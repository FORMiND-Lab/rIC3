use crate::{
    accel::cdcl_host::ActivePreflight,
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
    fn consume_hardware_push_result(
        &mut self,
        frame_idx: usize,
        lemma: &LitOrdVec,
        query: &IncrementalQuery,
        result: &IncrementalResult,
    ) -> Option<bool> {
        if crate::accel::cdcl_host::active_skip_cpu_check() {
            return match result {
                IncrementalResult::Sat { .. } => {
                    Some(false)
                }
                IncrementalResult::Unsat { .. } => {
                    Some(true)
                }
                IncrementalResult::Unknown(_) => {
                    crate::accel::cdcl_host::note_active_cpu_fallback();
                    None
                }
            };
        }
        let answer = match result {
            IncrementalResult::Sat { model } => {
                let validation_start = Instant::now();
                let accepted = self.solvers[frame_idx].install_incremental_sat_model(query, model);
                crate::accel::cdcl_host::note_active_sat_model(
                    accepted,
                    validation_start.elapsed().as_nanos() as u64,
                );
                accepted.then_some(false)
            }
            IncrementalResult::Unsat { core, .. } => {
                let validation_start = Instant::now();
                let cpu_core_len =
                    self.solvers[frame_idx].validate_incremental_unsat_core(lemma, query, core);
                crate::accel::cdcl_host::note_active_unsat_core(
                    cpu_core_len.is_some(),
                    query.assumptions.len(),
                    core.len(),
                    cpu_core_len.unwrap_or(0),
                    validation_start.elapsed().as_nanos() as u64,
                );
                cpu_core_len.map(|_| true)
            }
            IncrementalResult::Unknown(_) => {
                crate::accel::cdcl_host::note_active_cpu_fallback();
                None
            }
        };
        answer
    }

    fn launch_push_prefetch(&mut self, from: usize, level: usize) {
        if !crate::accel::cdcl_host::push_prefetch_enabled() {
            return;
        }
        if self.push_prefetch.busy() {
            crate::accel::cdcl_host::note_active_push_prefetch_busy();
            return;
        }
        // The next successful outer iteration extends IC3 by one level and
        // starts propagation at today's top frame. Prefetch that exact
        // look-ahead first; old-frame repeats are only secondary candidates.
        let max_lemma_len = super::push_prefetch::PushPrefetchCache::max_lemma_len();
        let max_contexts = super::push_prefetch::PushPrefetchCache::max_contexts();
        let all_frames = std::iter::once(level)
            .chain(from..level)
            .collect::<Vec<_>>();
        let n_contexts = if max_contexts == 0 {
            all_frames.len()
        } else {
            all_frames.len().min(max_contexts)
        };
        let candidate_frames = &all_frames[..n_contexts];
        let skipped_context = all_frames[n_contexts..]
            .iter()
            .map(|frame_idx| self.frame[*frame_idx].len())
            .sum::<usize>();
        let mut eligible_candidates = 0usize;
        let mut skipped_long = 0usize;
        for frame_idx in candidate_frames {
            for lemma in self.frame[*frame_idx].iter() {
                if max_lemma_len != 0 && lemma.len() > max_lemma_len {
                    skipped_long += 1;
                } else {
                    eligible_candidates += 1;
                }
            }
        }
        let n_candidates = eligible_candidates
            .min(super::push_prefetch::PushPrefetchCache::launch_window());
        if n_candidates < crate::accel::cdcl_host::active_min_batch_size() {
            return;
        }
        if !self.push_prefetch.should_launch() {
            return;
        }
        crate::accel::cdcl_host::note_active_push_prefetch_skipped_long(skipped_long);
        crate::accel::cdcl_host::note_active_push_prefetch_skipped_context(skipped_context);

        let prepare_start = Instant::now();
        let mut keys = Vec::with_capacity(n_candidates);
        let mut solver_frames = Vec::new();
        let mut owned_solvers = Vec::new();
        let mut owned_requests = Vec::with_capacity(n_candidates);
        'frames: for frame_idx in candidate_frames {
            let frame_idx = *frame_idx;
            let mut lemmas: Vec<_> = self.frame[frame_idx].iter().collect();
            lemmas.sort_by_key(|lemma| lemma.len());
            for lemma in lemmas {
                if max_lemma_len != 0 && lemma.len() > max_lemma_len {
                    continue;
                }
                if keys.len() == n_candidates {
                    break 'frames;
                }
                let solver_index = match solver_frames
                    .iter()
                    .position(|candidate| *candidate == frame_idx)
                {
                    Some(solver_index) => solver_index,
                    None => {
                        solver_frames.push(frame_idx);
                        owned_solvers.push(self.solvers[frame_idx].dcs.clone());
                        owned_solvers.len() - 1
                    }
                };
                keys.push((frame_idx, LitOrdVec::new(lemma.as_litvec().clone())));
                owned_requests.push((
                    solver_index,
                    self.solvers[frame_idx].incremental_inductive_query(lemma, true, vec![]),
                ));
            }
        }
        let prepare_ns = prepare_start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        self.push_prefetch
            .start(keys, owned_solvers, owned_requests, prepare_ns);
    }

    pub fn propagate(&mut self, from: Option<usize>) -> bool {
        let level = self.level();
        let from = from.unwrap_or(self.frame.early).max(1);
        let push_prefetch_enabled =
            crate::accel::cdcl_host::push_prefetch_enabled();
        if push_prefetch_enabled {
            self.push_prefetch.begin_pass();
        }
        // Prepare the whole propagation pass before mutating any frame. SAT
        // models are safe to speculate this far ahead because every model is
        // checked again against the live solver immediately before use; a
        // lemma learned by an earlier frame can only reject a stale model and
        // trigger CPU fallback.
        let propagation_batch_enabled =
            crate::accel::cdcl_host::propagation_batch_enabled();
        let prepare_accel_queries = propagation_batch_enabled || push_prefetch_enabled;
        let mut work: Vec<(usize, Frame, Vec<IncrementalQuery>)> = Vec::new();
        for frame_idx in from..level {
            let mut frame = self.frame[frame_idx].clone();
            frame.sort_by_key(|x| x.len());
            let active_queries = if prepare_accel_queries {
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
        let n_queries = work
            .iter()
            .map(|(_, _, queries)| queries.len())
            .sum::<usize>();
        let mut preflight = vec![ActivePreflight::Fpga; n_queries];
        if propagation_batch_enabled
            && crate::accel::cdcl_host::active_preflight_should_run(n_queries)
        {
            let mut query_index = 0usize;
            for (frame_idx, _, queries) in &work {
                for query in queries {
                    preflight[query_index] =
                        crate::accel::cdcl_host::active_preflight_classify(
                            &mut self.solvers[*frame_idx].dcs,
                            query,
                        );
                    query_index += 1;
                }
            }
            let mut sample_requests = Vec::with_capacity(n_queries);
            for (frame_idx, _, queries) in &work {
                let solver = &self.solvers[*frame_idx].dcs;
                for query in queries {
                    sample_requests.push((solver, query));
                }
            }
            crate::accel::cdcl_host::active_sample_select_pass(
                &sample_requests,
                &mut preflight,
            );
        }
        let active_results = if propagation_batch_enabled {
            let mut requests = Vec::new();
            let mut request_indices = Vec::new();
            let mut query_index = 0usize;
            for (frame_idx, _, queries) in &work {
                let solver = &self.solvers[*frame_idx].dcs;
                for query in queries {
                    if matches!(&preflight[query_index], ActivePreflight::Fpga) {
                        requests.push((solver, query.clone()));
                        request_indices.push(query_index);
                    }
                    query_index += 1;
                }
            }
            let selected_results =
                crate::accel::cdcl_host::solve_active_batch(requests);
            let mut results = vec![
                IncrementalResult::Unknown(
                    crate::accel::cdcl::UnknownReason::BackendError,
                );
                n_queries
            ];
            for (index, result) in request_indices.into_iter().zip(selected_results) {
                results[index] = result;
            }
            results
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
                        let result_index = frame_result_offset + lemma_index;
                        let prefetched = push_prefetch_enabled
                            .then(|| self.push_prefetch.take(frame_idx, &lemma))
                            .flatten();
                        if let Some(result) = prefetched.as_ref() {
                            let answer = self.consume_hardware_push_result(
                                frame_idx,
                                &lemma,
                                &active_queries[lemma_index],
                                &result.result,
                            );
                            self.push_prefetch
                                .note_validation(result.batch_id, answer.is_some());
                            crate::accel::cdcl_host::note_active_push_prefetch_hit(
                                lemma.len(),
                                answer.is_some(),
                            );
                            answer
                        } else {
                            match preflight.get(result_index) {
                                Some(ActivePreflight::Conclusive(IncrementalResult::Sat {
                                    model,
                                })) => {
                                    let restore_start = Instant::now();
                                    let accepted = self.solvers[frame_idx]
                                        .install_incremental_sat_model(
                                            &active_queries[lemma_index],
                                            model,
                                        );
                                    crate::accel::cdcl_host::note_active_preflight_result(
                                        false,
                                        accepted,
                                        restore_start.elapsed().as_nanos() as u64,
                                    );
                                    accepted.then_some(false)
                                }
                                Some(ActivePreflight::Conclusive(IncrementalResult::Unsat {
                                    core,
                                    used_constraints,
                                })) => {
                                    let restore_start = Instant::now();
                                    let accepted = self.solvers[frame_idx]
                                        .install_incremental_cpu_unsat_core(
                                            lemma.as_litvec(),
                                            &active_queries[lemma_index],
                                            core,
                                            *used_constraints,
                                        );
                                    crate::accel::cdcl_host::note_active_preflight_result(
                                        true,
                                        accepted,
                                        restore_start.elapsed().as_nanos() as u64,
                                    );
                                    accepted.then_some(true)
                                }
                                _ => active_results.get(result_index).and_then(|result| {
                                    self.consume_hardware_push_result(
                                        frame_idx,
                                        &lemma,
                                        &active_queries[lemma_index],
                                        result,
                                    )
                                }),
                            }
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
        self.launch_push_prefetch(from, level);
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
