use super::IC3;
use crate::{
    gipsat::{IncrementalQuery, IncrementalResult},
    ic3::IC3Config,
    transys::TransysIf,
};
use giputils::hash::GHashSet;
use log::trace;
use logicrs::{Lit, LitOrdVec, LitVec, satif::Satif};
use rand::{RngExt, seq::SliceRandom};
use std::{collections::VecDeque, time::Instant};

#[derive(Clone, Copy, Debug, Default)]
pub struct DropVarParameter {
    pub limit: usize,
    max: usize,
    level: usize,
}

impl DropVarParameter {
    #[inline]
    pub fn new(limit: usize, max: usize, level: usize) -> Self {
        Self { limit, max, level }
    }

    fn sub_level(self) -> Self {
        Self {
            limit: self.limit,
            max: self.max,
            level: self.level - 1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum MicType {
    #[allow(unused)]
    NoMic,
    DropVar(DropVarParameter),
}

struct MicDropPrefetch {
    candidate_index: usize,
    query: IncrementalQuery,
    result: IncrementalResult,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MicBatchRoute {
    Reject,
    Probe,
    Offload,
}

#[derive(Clone, Copy, Debug)]
struct MicHardwareSample {
    service_per_query_ns: u64,
    unsat_percent: u64,
}

#[derive(Default)]
pub(super) struct MicBatchPolicy {
    cpu_samples_ns: VecDeque<u64>,
    hardware_samples: VecDeque<MicHardwareSample>,
    cpu_since_probe: usize,
}

impl MicBatchPolicy {
    fn economics_enabled() -> bool {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_MIC_BATCH_ECONOMICS")
                .ok()
                .is_none_or(|value| !matches!(value.as_str(), "0" | "false" | "off"))
        })
    }

    fn cpu_window() -> usize {
        use std::sync::OnceLock;
        static WINDOW: OnceLock<usize> = OnceLock::new();
        *WINDOW.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_MIC_BATCH_CPU_WINDOW")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(128)
                .clamp(8, 4096)
        })
    }

    fn min_cpu_samples() -> usize {
        use std::sync::OnceLock;
        static SAMPLES: OnceLock<usize> = OnceLock::new();
        *SAMPLES.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_MIC_BATCH_CPU_SAMPLES")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(8)
                .clamp(1, Self::cpu_window())
        })
    }

    fn hardware_window() -> usize {
        use std::sync::OnceLock;
        static WINDOW: OnceLock<usize> = OnceLock::new();
        *WINDOW.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_MIC_BATCH_HW_WINDOW")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(8)
                .clamp(1, 64)
        })
    }

    fn min_hardware_samples() -> usize {
        use std::sync::OnceLock;
        static SAMPLES: OnceLock<usize> = OnceLock::new();
        *SAMPLES.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_MIC_BATCH_HW_SAMPLES")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(2)
                .clamp(1, Self::hardware_window())
        })
    }

    fn speedup_percent() -> u64 {
        use std::sync::OnceLock;
        static PERCENT: OnceLock<u64> = OnceLock::new();
        *PERCENT.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_MIC_BATCH_SPEEDUP_PCT")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(125)
                .clamp(100, 10_000)
        })
    }

    fn reprobe_cpu_queries() -> usize {
        use std::sync::OnceLock;
        static QUERIES: OnceLock<usize> = OnceLock::new();
        *QUERIES.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_MIC_BATCH_REPROBE_CPU_QUERIES")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(65_536)
                .clamp(1, 1 << 24)
        })
    }

    fn lower_median(values: impl Iterator<Item = u64>) -> Option<u64> {
        let mut values: Vec<_> = values.collect();
        values.sort_unstable();
        values.get(values.len().saturating_sub(1) / 2).copied()
    }

    fn route(&self, queries: usize) -> (MicBatchRoute, u64, Option<u64>) {
        if !Self::economics_enabled() {
            return (MicBatchRoute::Offload, 0, None);
        }
        self.route_at(
            queries,
            Self::min_cpu_samples(),
            Self::min_hardware_samples(),
            Self::reprobe_cpu_queries(),
            Self::speedup_percent(),
        )
    }

    fn route_at(
        &self,
        queries: usize,
        min_cpu_samples: usize,
        min_hardware_samples: usize,
        reprobe_cpu_queries: usize,
        speedup_percent: u64,
    ) -> (MicBatchRoute, u64, Option<u64>) {
        let Some(cpu_per_query_ns) = (self.cpu_samples_ns.len() >= min_cpu_samples)
            .then(|| Self::lower_median(self.cpu_samples_ns.iter().copied()))
            .flatten()
        else {
            return (MicBatchRoute::Reject, 0, None);
        };
        if self.hardware_samples.len() < min_hardware_samples {
            return (
                MicBatchRoute::Probe,
                cpu_per_query_ns.saturating_mul(queries as u64),
                None,
            );
        }
        let Some(service_per_query_ns) =
            Self::lower_median(self.hardware_samples.iter().map(|sample| {
                sample.service_per_query_ns
            }))
        else {
            return (
                MicBatchRoute::Probe,
                cpu_per_query_ns.saturating_mul(queries as u64),
                None,
            );
        };
        let unsat_percent = Self::lower_median(
            self.hardware_samples
                .iter()
                .map(|sample| sample.unsat_percent),
        )
        .unwrap_or(0);
        // Only an UNSAT answer can currently replace the native MIC query.
        // Using all FPGA UNSAT answers as potentially useful is deliberately
        // optimistic; invalidated answers make the real crossover stricter.
        let projected_cpu_ns = cpu_per_query_ns
            .saturating_mul(queries as u64)
            .saturating_mul(unsat_percent)
            / 100;
        let projected_hardware_ns = service_per_query_ns.saturating_mul(queries as u64);
        if projected_cpu_ns.saturating_mul(100)
            >= projected_hardware_ns.saturating_mul(speedup_percent)
        {
            (
                MicBatchRoute::Offload,
                projected_cpu_ns,
                Some(projected_hardware_ns),
            )
        } else if self.cpu_since_probe >= reprobe_cpu_queries {
            (
                MicBatchRoute::Probe,
                projected_cpu_ns,
                Some(projected_hardware_ns),
            )
        } else {
            (
                MicBatchRoute::Reject,
                projected_cpu_ns,
                Some(projected_hardware_ns),
            )
        }
    }

    fn note_cpu(&mut self, elapsed_ns: u64) {
        if self.cpu_samples_ns.len() == Self::cpu_window() {
            self.cpu_samples_ns.pop_front();
        }
        self.cpu_samples_ns.push_back(elapsed_ns);
        self.cpu_since_probe = self.cpu_since_probe.saturating_add(1);
    }

    fn note_hardware(&mut self, queries: u64, service_ns: u64, unsat: u64) {
        self.cpu_since_probe = 0;
        if queries == 0 || service_ns == 0 {
            return;
        }
        if self.hardware_samples.len() == Self::hardware_window() {
            self.hardware_samples.pop_front();
        }
        self.hardware_samples.push_back(MicHardwareSample {
            service_per_query_ns: service_ns.div_ceil(queries),
            unsat_percent: unsat.min(queries).saturating_mul(100) / queries,
        });
    }
}

impl MicType {
    pub fn from_config(cfg: &IC3Config) -> Self {
        let p = if cfg.ctg {
            DropVarParameter {
                limit: cfg.ctg_limit,
                max: cfg.ctg_max,
                level: 1,
            }
        } else {
            DropVarParameter::default()
        };
        MicType::DropVar(p)
    }
}

impl IC3 {
    fn launch_mic_drop_wave(
        &mut self,
        frame: usize,
        cube: &LitVec,
        keep: &GHashSet<Lit>,
        constraint: &[LitVec],
        parameter_level: usize,
        start: usize,
    ) -> Vec<MicDropPrefetch> {
        if !crate::accel::cdcl_host::mic_batch_enabled() {
            return Vec::new();
        }
        let window = crate::accel::cdcl_host::mic_batch_window();
        let mut candidates = Vec::new();
        for candidate_index in start..cube.len() {
            if keep.contains(&cube[candidate_index]) {
                continue;
            }
            let mut removed_cube = cube.clone();
            removed_cube.remove(candidate_index);
            let query = self.solvers[frame - 1].incremental_inductive_query(
                &removed_cube,
                true,
                if parameter_level == 0 {
                    constraint.to_vec()
                } else {
                    Vec::new()
                },
            );
            candidates.push((candidate_index, query));
            if candidates.len() == window {
                break;
            }
        }
        let min_batch = crate::accel::cdcl_host::mic_batch_min_size();
        if candidates.len() < min_batch {
            return Vec::new();
        }

        let (route, projected_cpu_ns, projected_hardware_ns) =
            self.mic_batch_policy.route(candidates.len());
        crate::accel::cdcl_host::note_active_mic_batch_economics(
            projected_cpu_ns,
            projected_hardware_ns,
            route == MicBatchRoute::Probe,
            route == MicBatchRoute::Offload,
        );
        if route == MicBatchRoute::Reject {
            return Vec::new();
        }

        let owned_solver = self.solvers[frame - 1].dcs.clone();
        let requests = candidates
            .iter()
            .map(|(_, query)| (&owned_solver, query.clone()))
            .collect();
        let service_before = crate::accel::cdcl_host::active_batch_service_snapshot();
        let results = crate::accel::cdcl_host::solve_active_batch_with_min(requests, min_batch);
        let service_after = crate::accel::cdcl_host::active_batch_service_snapshot();
        let hardware_unsat = results
            .iter()
            .filter(|result| matches!(result, IncrementalResult::Unsat { .. }))
            .count() as u64;
        self.mic_batch_policy.note_hardware(
            service_after.1.saturating_sub(service_before.1),
            service_after.2.saturating_sub(service_before.2),
            hardware_unsat,
        );
        crate::accel::cdcl_host::note_active_mic_wave(&results);
        if route == MicBatchRoute::Probe {
            // Calibration must be proof-neutral. Even a valid core can send
            // IC3 down a different and much longer path, so a probe measures
            // service/yield only and never exposes its answers to MIC.
            crate::accel::cdcl_host::note_active_mic_invalidated(results.len());
            return Vec::new();
        }
        candidates
            .into_iter()
            .zip(results)
            .map(|((candidate_index, query), result)| MicDropPrefetch {
                candidate_index,
                query,
                result,
            })
            .collect()
    }

    fn consume_mic_drop_result(
        &mut self,
        frame: usize,
        cube: &LitVec,
        prefetched: &MicDropPrefetch,
    ) -> Option<bool> {
        match &prefetched.result {
            IncrementalResult::Sat { .. } => {
                // `down`/`ctg_down` feed the satisfying assignment into
                // model-sensitive cube shrinking and predecessor lifting.
                // The generic active-model importer is sufficient for a
                // boolean blocked/not-blocked answer, but it has not proved
                // equivalence to GipSAT's native model state for those later
                // operations. Keep the complete FPGA SAT result observable,
                // then rerun this inquiry on the live solver until that
                // stronger state-transfer invariant has its own validator.
                crate::accel::cdcl_host::note_active_cpu_fallback();
                crate::accel::cdcl_host::note_active_mic_consumed(false, false);
                None
            }
            IncrementalResult::Unsat { core, .. } => {
                let validation_start = Instant::now();
                let cpu_core_len = self.solvers[frame - 1].validate_incremental_unsat_core(
                    cube,
                    &prefetched.query,
                    core,
                );
                crate::accel::cdcl_host::note_active_unsat_core(
                    cpu_core_len.is_some(),
                    prefetched.query.assumptions.len(),
                    core.len(),
                    cpu_core_len.unwrap_or(0),
                    validation_start.elapsed().as_nanos() as u64,
                );
                crate::accel::cdcl_host::note_active_mic_consumed(
                    true,
                    cpu_core_len.is_some(),
                );
                cpu_core_len.map(|_| true)
            }
            IncrementalResult::Unknown(_) => {
                crate::accel::cdcl_host::note_active_cpu_fallback();
                None
            }
        }
    }

    fn down(
        &mut self,
        frame: usize,
        cube: &LitVec,
        keep: &GHashSet<Lit>,
        full: &LitVec,
        constraint: &[LitVec],
        cex: &mut Vec<(LitOrdVec, LitOrdVec)>,
        mut prefetched: Option<&MicDropPrefetch>,
        measure_cpu: bool,
    ) -> Option<LitVec> {
        let mut cube = cube.clone();
        self.statistic.num_down += 1;
        loop {
            if self.tsctx.cube_subsume_init(&cube) {
                if prefetched.take().is_some() {
                    crate::accel::cdcl_host::note_active_mic_invalidated(1);
                }
                return None;
            }
            let lemma = LitOrdVec::new(cube.clone());
            if cex
                .iter()
                .any(|(s, t)| !lemma.subsume(s) && lemma.subsume(t))
            {
                if prefetched.take().is_some() {
                    crate::accel::cdcl_host::note_active_mic_invalidated(1);
                }
                return None;
            }
            self.statistic.num_down_sat += 1;

            // Ask the card before the solver, because asking after saves
            // nothing. It returns the unsat core -- the subset of the
            // next-state assumptions that conflicts -- which is exactly what
            // `inductive_core()` reconstructs and what becomes the lemma.
            //
            // Treat it as a candidate, not a proof boundary. The intended
            // invariant is that the card holds a subset of this frame's
            // clauses and is therefore weaker, but a real gate-bitstream run
            // violated it and changed an unsafe result into safe. Candidate
            // cores are rechecked below by this exact frame solver, with the
            // same strengthen and per-query constraints, before IC3 sees one.
            if crate::accel::core_offload()
                && crate::accel::ready()
                && crate::accel::select_core_query(cube.len())
            {
                // The lemmas the frame has gained are not visible to the card
                // until its occurrence index is rebuilt, and that used to happen
                // in the shadow block. Gating the shadow took it with it, and
                // the card went on propagating over a stale index: cores fell
                // from 84 to 10 while the run got ten times faster, which is
                // the shape of an engine that has stopped seeing the lemmas
                // rather than one that got quicker.
                crate::accel::sync_index();
                let assump = self.tsctx.lits_next(&cube);
                let raw: Vec<u32> = assump.iter().map(|l| Into::<u32>::into(*l)).collect();
                // No domain restriction here.
                //
                // The solver's domain is the transitive closure `enable_local`
                // builds, and it is built inside `solve()` -- after this point.
                // Sending the surface set instead, the assumptions and the cube,
                // gave the card a domain so small it propagated almost nothing:
                // 5 cores from 2,892 asks with every constraint accepted.
                //
                // Dropping the restriction is sound in the direction that
                // matters. It only lets propagation reach further over clauses
                // that are all real constraints, so a conflict it derives is
                // still a conflict for the query; the domain can lose
                // implications, never invent them.
                crate::accel::set_domain(&[]);
                // The clauses this query carries. `down` calls `blocked` with
                // `.with_strengthen()`, which adds `!cube`, plus whatever the
                // caller passed; the card needs both or it is weaker than the
                // solver on exactly these queries.
                let mut flat: Vec<u32> = Vec::new();
                {
                    let mut push = |c: &LitVec| {
                        flat.push(c.len() as u32);
                        for l in c.iter() {
                            flat.push(Into::<u32>::into(*l));
                        }
                    };
                    push(&LitVec::from_iter(cube.iter().map(|l| !*l)));
                    for c in constraint.iter() {
                        push(c);
                    }
                }
                let mut got: Vec<u32> = Vec::new();
                let lvl = crate::accel::level_arg((frame - 1) as u32);
                // One round trip where there were four. The card installs the
                // constraint, answers, minimises, and takes the constraint
                // back out itself, so the drop below is only for the
                // bitstreams that predate the fused mode.
                let got_core = if crate::accel::have_down() {
                    matches!(crate::accel::down(&flat, &raw, lvl, &mut got), Some(n) if n > 0)
                } else {
                    crate::accel::set_constraint(&flat);
                    let r = crate::accel::core(&raw, lvl, &mut got).is_some();
                    crate::accel::set_constraint(&[]);
                    r
                };
                if got_core {
                    let inset: std::collections::HashSet<u32> = got.into_iter().collect();
                    let mut ans = LitVec::new();
                    for &l in cube.iter() {
                        if inset.contains(&Into::<u32>::into(self.tsctx.next(l))) {
                            ans.push(l);
                        }
                    }
                    // Only when the card actually generalized something.
                    //
                    // A core equal to the cube generalizes nothing, and taking
                    // it is no better than asking the exact solver directly,
                    // and taking it skips the solver's own `down`, which would
                    // have returned a smaller one. Measured on
                    // Problem03_label51, 8,153 such cores went in at 2.00
                    // literals and came back at 2.00, and IC3 then called mic
                    // 24,514 times against the CPU-only run's 2,157 -- an
                    // eleven-fold increase that cost more than everything the
                    // card saved.
                    //
                    // `INDUCTOR_CORE_GAIN` is how many literals the card has to
                    // remove to be worth believing. 0 restores the old
                    // behaviour of taking every core.
                    let gain = cube.len().saturating_sub(ans.len());
                    let worth_checking = !ans.is_empty()
                        && gain >= crate::accel::core_gain()
                        && !self.tsctx.cube_subsume_init(&ans);
                    if worth_checking {
                        // Validate on a clone and stop before decisions. The
                        // live solver must not inherit watcher movement,
                        // learnt clauses or heuristic state from a rejected
                        // hardware candidate; and a validation query which
                        // needs search can be harder than the original cube.
                        let validated = if crate::accel::unchecked_core() {
                            true
                        } else {
                            let mut verifier = self.solvers[frame - 1].clone();
                            verifier.inductive_by_propagation(
                                &ans,
                                true,
                                constraint.to_vec(),
                            )
                        };
                        if validated {
                            if !crate::accel::unchecked_core() {
                                crate::accel::CORE_VALIDATED.fetch_add(
                                    1,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                            }
                            crate::accel::CORE_USED
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            crate::accel::observe_card_core_query(cube.len(), true);
                            return Some(ans);
                        }
                        crate::accel::CORE_VALIDATION_FAILED.fetch_add(
                            1,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    } else {
                        crate::accel::CORE_THIN
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                // The selector learns the probability of a *usable* core, not
                // merely a hardware conflict. An invalid or non-generalizing
                // core saves no CPU work and must make future offload less
                // likely just like a card miss.
                crate::accel::observe_card_core_query(cube.len(), false);
            }

            let active_answer = prefetched
                .take()
                .and_then(|answer| self.consume_mic_drop_result(frame, &cube, answer));
            let used_prefetched = active_answer.is_some();
            let blocked = match active_answer {
                Some(blocked) => blocked,
                None if measure_cpu => {
                    let cpu_start = Instant::now();
                    let blocked = self
                        .blocked(frame, &cube)
                        .in_phase(inductor_trace::Phase::Gen)
                        .with_act_order(false)
                        .with_strengthen()
                        .with_constraint(constraint)
                        .check();
                    self.mic_batch_policy.note_cpu(
                        cpu_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                    );
                    blocked
                }
                None => self
                    .blocked(frame, &cube)
                    .in_phase(inductor_trace::Phase::Gen)
                    .with_act_order(false)
                    .with_strengthen()
                    .with_constraint(constraint)
                    .check(),
            };
            if !used_prefetched {
                crate::accel::observe_core_query(
                    cube.len(),
                    blocked,
                    self.solvers[frame - 1].dcs.probe.t_bcp_ns,
                );
            }
            if blocked {
                return Some(self.solvers[frame - 1].inductive_core().unwrap());
            }
            let mut ret = false;
            let mut cube_new = LitVec::new();
            for lit in cube {
                if keep.contains(&lit) {
                    if let Some(true) = self.solvers[frame - 1].sat_value(lit) {
                        cube_new.push(lit);
                    } else {
                        ret = true;
                        break;
                    }
                } else if let Some(true) = self.solvers[frame - 1].sat_value(lit)
                    && !self.solvers[frame - 1].flip_to_none(lit.var())
                {
                    cube_new.push(lit);
                }
            }
            cube = cube_new;
            let mut s = LitVec::new();
            let mut t = LitVec::new();
            for l in full.iter() {
                if let Some(v) = self.solvers[frame - 1].sat_value(*l)
                    && self.solvers[frame - 1].flip_to_none(l.var())
                {
                    s.push(l.not_if(!v));
                }
                if let Some(v) = self.solvers[frame - 1].sat_value(self.tsctx.next(*l)) {
                    t.push(l.not_if(!v));
                }
            }
            cex.push((LitOrdVec::new(s), LitOrdVec::new(t)));
            if ret {
                return None;
            }
        }
    }

    fn ctg_down(
        &mut self,
        frame: usize,
        cube: &LitVec,
        keep: &GHashSet<Lit>,
        full: &LitVec,
        parameter: DropVarParameter,
        mut prefetched: Option<&MicDropPrefetch>,
        measure_cpu: bool,
    ) -> Option<LitVec> {
        let mut cube = cube.clone();
        self.statistic.num_down += 1;
        let mut ctg = 0;
        loop {
            if self.tsctx.cube_subsume_init(&cube) {
                if prefetched.take().is_some() {
                    crate::accel::cdcl_host::note_active_mic_invalidated(1);
                }
                return None;
            }
            self.statistic.num_down_sat += 1;
            let active_answer = prefetched
                .take()
                .and_then(|answer| self.consume_mic_drop_result(frame, &cube, answer));
            let blocked = match active_answer {
                Some(blocked) => blocked,
                None if measure_cpu => {
                    let cpu_start = Instant::now();
                    let blocked = self
                        .blocked(frame, &cube)
                        .in_phase(inductor_trace::Phase::Gen)
                        .with_act_order(false)
                        .with_strengthen()
                        .check();
                    self.mic_batch_policy.note_cpu(
                        cpu_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                    );
                    blocked
                }
                None => self
                    .blocked(frame, &cube)
                    .in_phase(inductor_trace::Phase::Gen)
                    .with_act_order(false)
                    .with_strengthen()
                    .check(),
            };
            if blocked {
                return Some(self.solvers[frame - 1].inductive_core().unwrap());
            }
            for lit in cube.iter() {
                if keep.contains(lit) && !self.solvers[frame - 1].sat_value(*lit).is_some_and(|v| v)
                {
                    return None;
                }
            }
            let (model, _) = self.get_pred(frame, false);
            let cex_set: GHashSet<Lit> = GHashSet::from_iter(model.iter().cloned());
            // for lit in cube.iter() {
            //     if keep.contains(lit) && !cex_set.contains(lit) {
            //         return None;
            //     }
            // }
            if ctg < parameter.max
                && frame > 1
                && !self.tsctx.cube_subsume_init(&model)
                && self.trivial_block(
                    frame - 1,
                    LitOrdVec::new(model.clone()),
                    &[!full.clone()],
                    parameter.sub_level(),
                )
            {
                ctg += 1;
                continue;
            }
            ctg = 0;
            let mut cube_new = LitVec::new();
            for lit in cube {
                if cex_set.contains(&lit) {
                    cube_new.push(lit);
                } else if keep.contains(&lit) {
                    return None;
                }
            }
            cube = cube_new;
        }
    }

    fn handle_down_success(
        &mut self,
        _frame: usize,
        cube: LitVec,
        i: usize,
        mut new_cube: LitVec,
    ) -> (LitVec, usize) {
        new_cube = cube
            .iter()
            .filter(|l| new_cube.contains(l))
            .cloned()
            .collect();
        let new_i = new_cube
            .iter()
            .position(|l| !(cube[0..i]).contains(l))
            .unwrap_or(new_cube.len());
        if new_i < new_cube.len() {
            assert!(!(cube[0..=i]).contains(&new_cube[new_i]))
        }
        (new_cube, new_i)
    }

    fn mic_by_drop_var(
        &mut self,
        frame: usize,
        mut cube: LitVec,
        constraint: &[LitVec],
        parameter: DropVarParameter,
    ) -> LitVec {
        let start = Instant::now();
        let _op = crate::inductor::macro_scope(inductor_trace::Phase::Gen, frame);
        if parameter.level == 0 {
            self.solvers[frame - 1].set_domain(
                self.tsctx
                    .lits_next(&cube)
                    .iter()
                    .copied()
                    .chain(cube.iter().copied()),
            );
        }
        self.statistic.avg_mic_cube_len += cube.len();
        self.statistic.num_mic += 1;
        // How many independent queries one generalization could issue at once.
        // Every speculative drop is a subset of this cube, so they share the
        // domain set just above and the clause set, which is fixed for the
        // duration of a mic -- the two conditions RUN_BATCH needs and the two
        // that the earlier per-query batching attempt could not meet.
        crate::accel::note_mic(cube.len());
        let mut cex = Vec::new();
        if self.rng.random_bool(0.2) {
            cube.shuffle(&mut self.rng);
        } else {
            self.activity.sort_by_activity(&mut cube, true);
        }
        if self.cfg.parent_lemma
            && let Some(parent) = self.frame.parent_lemma(&cube, frame)
        {
            let parent = GHashSet::from_iter(parent);
            cube.sort_by_key(|x| parent.contains(x));
        }
        let mut keep = GHashSet::new();
        // Cache this once per MIC.  The default CPU path must not repeatedly
        // inspect the environment, clone the parent cube, time queries, or
        // maintain experimental telemetry when the opt-in experiment is off.
        let mic_batch_enabled = crate::accel::cdcl_host::mic_batch_enabled();
        let mut prefetched_parent: Option<LitVec> = None;
        let mut prefetched_wave = Vec::new();

        // Let the card run the drop loop first.
        //
        // It tries the same removals this loop does, with propagation instead
        // of the solver, and returns a sub-cube that still blocks. Weaker: on
        // the satisfiable branch the solver shrinks the cube from its model
        // and the card has no model, so it keeps the literal. But every
        // literal it did drop was dropped because a conflict survived without
        // it, so what comes back is a sound starting point -- and the loop
        // below still runs, so nothing this misses is lost.
        //
        // One call for the whole loop. The assumptions and the constraint are
        // both derived from the cube and both change every time it shrinks,
        // which is why this could not be a batch of queries prepared here.
        if crate::accel::mic_offload() && crate::accel::ready() && crate::accel::have_mic() {
            crate::accel::sync_index();
            let mut pairs: Vec<u32> = Vec::with_capacity(cube.len() * 2);
            for l in cube.iter() {
                pairs.push(Into::<u32>::into(*l));
                pairs.push(Into::<u32>::into(self.tsctx.next(*l)));
            }
            let mut extra: Vec<u32> = Vec::new();
            for c in constraint.iter() {
                extra.push(c.len() as u32);
                for l in c.iter() {
                    extra.push(Into::<u32>::into(*l));
                }
            }
            let mut got: Vec<u32> = Vec::new();
            let lvl = crate::accel::level_arg((frame - 1) as u32);
            if crate::accel::mic(&extra, &pairs, lvl, &mut got).is_some()
                && !got.is_empty()
                && got.len() < cube.len()
            {
                let inset: std::collections::HashSet<u32> = got.into_iter().collect();
                let mut shrunk = LitVec::new();
                for l in cube.iter() {
                    if inset.contains(&Into::<u32>::into(*l)) {
                        shrunk.push(*l);
                    }
                }
                // A cube that subsumes the initial states is not a lemma. The
                // card does not test that, and `down` would have handed such a
                // cube back rather than repaired it.
                if !shrunk.is_empty() && !self.tsctx.cube_subsume_init(&shrunk) {
                    crate::accel::MIC_TAKEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    cube = shrunk;
                }
            }
        }

        let mut i = 0;
        while i < cube.len() {
            if keep.contains(&cube[i]) {
                i += 1;
                continue;
            }
            if mic_batch_enabled && prefetched_parent.as_ref() != Some(&cube) {
                crate::accel::cdcl_host::note_active_mic_invalidated(prefetched_wave.len());
                prefetched_wave = self.launch_mic_drop_wave(
                    frame,
                    &cube,
                    &keep,
                    constraint,
                    parameter.level,
                    i,
                );
                prefetched_parent = Some(cube.clone());
            }
            let mut removed_cube = cube.clone();
            removed_cube.remove(i);
            let prefetched = if mic_batch_enabled {
                prefetched_wave
                    .iter()
                    .position(|entry| entry.candidate_index == i)
                    .map(|index| prefetched_wave.swap_remove(index))
            } else {
                None
            };
            let mic = if parameter.level == 0 {
                self.down(
                    frame,
                    &removed_cube,
                    &keep,
                    &cube,
                    constraint,
                    &mut cex,
                    prefetched.as_ref(),
                    mic_batch_enabled,
                )
            } else {
                self.ctg_down(
                    frame,
                    &removed_cube,
                    &keep,
                    &cube,
                    parameter,
                    prefetched.as_ref(),
                    mic_batch_enabled,
                )
            };
            if let Some(new_cube) = mic {
                self.statistic.mic_drop.success();
                (cube, i) = self.handle_down_success(frame, cube, i, new_cube);
                crate::accel::cdcl_host::note_active_mic_invalidated(prefetched_wave.len());
                prefetched_wave.clear();
                prefetched_parent = None;
                if parameter.level == 0 {
                    self.solvers[frame - 1].unset_domain();
                    self.solvers[frame - 1].set_domain(
                        self.tsctx
                            .lits_next(&cube)
                            .iter()
                            .copied()
                            .chain(cube.iter().copied()),
                    );
                }
            } else {
                self.statistic.mic_drop.fail();
                keep.insert(cube[i]);
                i += 1;
            }
        }
        crate::accel::cdcl_host::note_active_mic_invalidated(prefetched_wave.len());
        if parameter.level == 0 {
            self.solvers[frame - 1].unset_domain();
        }
        self.activity.bump_cube_activity(&cube);
        self.statistic.block.mic_time += start.elapsed();
        cube
    }

    pub(super) fn mic(
        &mut self,
        frame: usize,
        cube: LitVec,
        constraint: &[LitVec],
        mic_type: MicType,
    ) -> LitVec {
        let mic_olen = cube.len();
        let r = match mic_type {
            MicType::NoMic => cube,
            MicType::DropVar(parameter) => self.mic_by_drop_var(frame, cube, constraint, parameter),
        };
        trace!("mic from {} to {} len", mic_olen, r.len());
        r
    }
}

#[cfg(test)]
mod mic_batch_policy_tests {
    use super::{MicBatchPolicy, MicBatchRoute};

    fn cpu_trained() -> MicBatchPolicy {
        let mut policy = MicBatchPolicy::default();
        for _ in 0..8 {
            policy.cpu_samples_ns.push_back(200_000);
        }
        policy
    }

    #[test]
    fn mic_batch_economics_probes_once_then_rejects_slow_hardware() {
        let mut policy = cpu_trained();
        assert_eq!(
            policy.route_at(16, 8, 1, 4096, 125).0,
            MicBatchRoute::Probe
        );
        policy.note_hardware(16, 4_000_000, 8);
        let evaluation = policy.route_at(16, 8, 1, 4096, 125);
        assert_eq!(evaluation.0, MicBatchRoute::Reject);
        assert_eq!(evaluation.1, 1_600_000);
        assert_eq!(evaluation.2, Some(4_000_000));

        policy.cpu_since_probe = 4096;
        assert_eq!(
            policy.route_at(16, 8, 1, 4096, 125).0,
            MicBatchRoute::Probe
        );
    }

    #[test]
    fn mic_batch_economics_counts_only_replaceable_unsat_work() {
        let mut profitable = cpu_trained();
        profitable.note_hardware(16, 1_000_000, 16);
        assert_eq!(
            profitable.route_at(16, 8, 1, 4096, 125),
            (MicBatchRoute::Offload, 3_200_000, Some(1_000_000))
        );

        let mut mostly_sat = cpu_trained();
        mostly_sat.note_hardware(16, 1_000_000, 1);
        assert_eq!(
            mostly_sat.route_at(16, 8, 1, 4096, 125).0,
            MicBatchRoute::Reject
        );
    }

    #[test]
    fn mic_batch_economics_keeps_calibration_proof_neutral_until_trained() {
        let mut policy = cpu_trained();
        policy.note_hardware(16, 4_000_000, 8);
        assert_eq!(
            policy.route_at(16, 8, 2, 4096, 125).0,
            MicBatchRoute::Probe
        );
        policy.note_hardware(16, 4_000_000, 8);
        assert_eq!(
            policy.route_at(16, 8, 2, 4096, 125).0,
            MicBatchRoute::Reject
        );
    }
}
