use crate::{
    gipsat::{IncrementalQuery, IncrementalResult, TransysSolver},
    ic3::{
        frame::{Frame, FrameLemma},
        mic::MicType,
        IC3,
    },
    transys::TransysIf,
};
use logicrs::{LitOrdVec, LitVec, satif::Satif};
use rand::seq::SliceRandom;
use std::time::Instant;

impl IC3 {
    pub fn propagate(&mut self, from: Option<usize>) -> bool {
        let level = self.level();
        let from = from.unwrap_or(self.frame.early).max(1);
        // Prepare the whole propagation pass before mutating any frame. SAT
        // models are safe to speculate this far ahead because every model is
        // checked again against the live solver immediately before use; a
        // lemma learned by an earlier frame can only reject a stale model and
        // trigger CPU fallback.
        let active_enabled = crate::accel::cdcl_host::active_enabled();
        let mut work: Vec<(usize, Frame, Vec<IncrementalQuery>)> = Vec::new();
        for frame_idx in from..level {
            let mut frame = self.frame[frame_idx].clone();
            frame.sort_by_key(|x| x.len());
            let active_queries = if active_enabled {
                frame
                    .iter()
                    .map(|lemma| {
                        self.solvers[frame_idx]
                            .incremental_inductive_query(lemma, true, vec![])
                    })
                    .collect()
            } else {
                Vec::new()
            };
            work.push((frame_idx, frame, active_queries));
        }
        let active_results = if active_enabled {
            let mut requests = Vec::new();
            for (frame_idx, _, queries) in &work {
                let solver = &self.solvers[*frame_idx].dcs;
                for query in queries {
                    requests.push((solver, query.clone()));
                }
            }
            crate::accel::cdcl_host::solve_active_batch(requests)
        } else {
            Vec::new()
        };

        let mut result_offset = 0usize;
        for (frame_idx, frame, active_queries) in work {
            let frame_result_offset = result_offset;
            result_offset += active_queries.len();
            let _op =
                crate::inductor::macro_scope(inductor_trace::Phase::Push, frame_idx + 1);
            for (lemma_index, mut lemma) in frame.into_iter().enumerate() {
                if self.frame[frame_idx].iter().all(|l| l.ne(&lemma)) {
                    continue;
                }
                for ctp in 0..3 {
                    // A validated SAT model means the lemma is not blocked;
                    // a CPU-reproved FPGA core means it is blocked. Any stale,
                    // malformed, budgeted, or inconclusive result falls back
                    // to the ordinary full GipSAT inquiry below.
                    let active_answer = if ctp == 0 {
                        match active_results.get(frame_result_offset + lemma_index) {
                            Some(IncrementalResult::Sat { model }) => {
                                let validation_start = Instant::now();
                                let accepted = self.solvers[frame_idx]
                                    .install_incremental_sat_model(
                                        &active_queries[lemma_index],
                                        model,
                                    );
                                crate::accel::cdcl_host::note_active_sat_model(
                                    accepted,
                                    validation_start.elapsed().as_nanos() as u64,
                                );
                                accepted.then_some(false)
                            }
                            Some(IncrementalResult::Unsat { core, .. }) => {
                                let validation_start = Instant::now();
                                let cpu_core_len = self.solvers[frame_idx]
                                    .validate_incremental_unsat_core(
                                        lemma.as_litvec(),
                                        &active_queries[lemma_index],
                                        core,
                                    );
                                crate::accel::cdcl_host::note_active_unsat_core(
                                    cpu_core_len.is_some(),
                                    active_queries[lemma_index].assumptions.len(),
                                    core.len(),
                                    cpu_core_len.unwrap_or(0),
                                    validation_start.elapsed().as_nanos() as u64,
                                );
                                cpu_core_len.map(|_| true)
                            }
                            Some(IncrementalResult::Unknown(_)) => {
                                crate::accel::cdcl_host::note_active_cpu_fallback();
                                None
                            }
                            None => None,
                        }
                    } else {
                        None
                    };
                    let blocked = match active_answer {
                        Some(blocked) => blocked,
                        None => self
                            .blocked(frame_idx + 1, &lemma)
                            .in_phase(inductor_trace::Phase::Push)
                            .with_act_order(false)
                            .check(),
                    };
                    if blocked {
                        let core = self.solvers[frame_idx]
                            .inductive_core()
                            .unwrap_or(lemma.as_litvec().clone());
                        if let Some(po) = &mut lemma.po
                            && po.frame < frame_idx + 2
                            && self.obligations.remove(po)
                        {
                            po.push_to(frame_idx + 2);
                            self.obligations.add(po.clone());
                        }
                        self.add_lemma(frame_idx + 1, core, true, lemma.po);
                        self.statistic.ctp.statistic(ctp > 0);
                        break;
                    }
                    if !self.cfg.ctp {
                        break;
                    }
                    let (ctp, _) = self.get_pred(frame_idx + 1, false);
                    if !self.tsctx.cube_subsume_init(&ctp)
                        && {
                            let _ctx = crate::inductor::set_context(
                                inductor_trace::Phase::Push,
                                frame_idx - 1,
                            );
                            self.solvers[frame_idx - 1].inductive(&ctp, true)
                        }
                    {
                        let core = self.solvers[frame_idx - 1].inductive_core().unwrap();
                        let mic =
                            self.mic(frame_idx, core, &[], MicType::DropVar(Default::default()));
                        if self.add_lemma(frame_idx, mic, false, None) {
                            return true;
                        }
                    } else {
                        break;
                    }
                }
            }
            if self.frame[frame_idx].is_empty() {
                return true;
            }
        }
        self.frame.early = self.level();
        false
    }

    pub fn propagete_to_inf_rec(&mut self, lastf: &mut Vec<FrameLemma>, ctp: LitVec) -> bool {
        let ctp = LitOrdVec::new(ctp);
        let Some(lidx) = lastf.iter().position(|l| l.subsume(&ctp)) else {
            return false;
        };
        let mut lemma = lastf.swap_remove(lidx);
        loop {
            if {
                let _ctx = crate::inductor::set_context(inductor_trace::Phase::Inf, usize::MAX);
                self.inf_solver.inductive(&lemma, true)
            } {
                if let Some(po) = &mut lemma.po {
                    self.obligations.remove(po);
                }
                self.add_inf_lemma(lemma.as_litvec().clone());
                return true;
            } else {
                let target = self.tsctx.lits_next(lemma.as_litvec());
                let (ctp, _) = self.lift.lift(
                    &mut self.inf_solver,
                    target.iter().chain(self.tsctx.constraint.iter()),
                    |i, _| i == 0,
                );
                if !self.propagete_to_inf_rec(lastf, ctp) {
                    return false;
                }
            }
        }
    }

    pub fn propagate_to_inf(&mut self) {
        let start = Instant::now();
        let mut lastf = self.frame.last().clone();
        lastf.shuffle(&mut self.rng);
        while let Some(mut lemma) = lastf.pop() {
            loop {
                if {
                    let _ctx =
                        crate::inductor::set_context(inductor_trace::Phase::Inf, usize::MAX);
                    self.inf_solver.inductive(&lemma, true)
                } {
                    if let Some(po) = &mut lemma.po {
                        self.obligations.remove(po);
                    }
                    self.add_inf_lemma(lemma.as_litvec().clone());
                    break;
                } else {
                    let target = self.tsctx.lits_next(lemma.as_litvec());
                    let (ctp, _) = self.lift.lift(
                        &mut self.inf_solver,
                        target.iter().chain(self.tsctx.constraint.iter()),
                        |i, _| i == 0,
                    );
                    if !self.propagete_to_inf_rec(&mut lastf, ctp) {
                        break;
                    }
                }
            }
        }
        self.statistic.propagate.push_inf_time += start.elapsed();
    }

    pub fn propagate_to_inf2(&mut self) {
        let start = Instant::now();
        let iter_max = 7;
        let mut cand: Vec<_> = self
            .frame
            .last()
            .iter()
            .map(|l| l.as_litvec().clone())
            .collect();
        if cand.is_empty() {
            return;
        }
        for k in 0..=iter_max {
            if k == iter_max {
                self.statistic.propagate.push_inf_time += start.elapsed();
                return;
            }
            let mut slv = TransysSolver::new(&self.tsctx);
            for i in self.frame.inf.iter() {
                slv.add_clause(&!i.as_litvec());
            }
            for c in cand.iter() {
                slv.add_clause(&!c);
            }
            let mut new_cand = Vec::new();
            for c in cand.iter() {
                if slv.inductive(c, false) {
                    new_cand.push(c.clone());
                }
            }
            if new_cand.len() == cand.len() {
                break;
            } else {
                cand = new_cand;
            }
        }
        for c in cand {
            self.add_inf_lemma(c);
        }
        self.statistic.propagate.push_inf_time += start.elapsed();
    }
}
