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
    proof_neutral: bool,
}

enum MicDropAnswer<'a> {
    Blocked,
    Sat {
        query: &'a IncrementalQuery,
        model: &'a LitVec,
    },
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
}

#[derive(Default)]
pub(super) struct MicBatchPolicy {
    cpu_samples_ns: VecDeque<u64>,
    hardware_samples: VecDeque<MicHardwareSample>,
    cpu_since_probe: usize,
    shadow_queries: u64,
    shadow_replaceable: u64,
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
                .unwrap_or(8)
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
        let replaceable_percent = self
            .shadow_replaceable
            .min(self.shadow_queries)
            .saturating_mul(100)
            .checked_div(self.shadow_queries)
            .unwrap_or(0);
        // Both exact SAT witnesses and CPU-reproved UNSAT cores can replace
        // the first native MIC query. The proof-neutral shadow fraction has
        // already charged invalidated, inconclusive and mismatching answers
        // as zero yield. It remains optimistic for reached UNSAT answers,
        // whose cores are subset-checked here but re-proved only on adoption.
        let projected_cpu_ns = cpu_per_query_ns
            .saturating_mul(queries as u64)
            .saturating_mul(replaceable_percent)
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

    fn note_hardware(&mut self, queries: u64, service_ns: u64) {
        self.cpu_since_probe = 0;
        if queries == 0 || service_ns == 0 {
            return;
        }
        if self.hardware_samples.len() == Self::hardware_window() {
            self.hardware_samples.pop_front();
        }
        self.hardware_samples.push_back(MicHardwareSample {
            service_per_query_ns: service_ns.div_ceil(queries),
        });
    }

    fn note_shadow_result(&mut self, replaceable: bool) {
        self.shadow_queries = self.shadow_queries.saturating_add(1);
        if replaceable {
            self.shadow_replaceable = self.shadow_replaceable.saturating_add(1);
        }
    }

    fn note_shadow_invalidated(&mut self, count: usize) {
        self.shadow_queries = self.shadow_queries.saturating_add(count as u64);
    }
}

fn model_value(model: &[Lit], lit: Lit) -> Option<bool> {
    model
        .iter()
        .find(|candidate| candidate.var() == lit.var())
        .map(|candidate| candidate.polarity() == lit.polarity())
}

fn core_is_assumption_subset(query: &IncrementalQuery, core: &[Lit]) -> bool {
    let mut unmatched: Vec<Lit> = query.assumptions.iter().copied().collect();
    for &lit in core {
        let Some(position) = unmatched.iter().position(|candidate| *candidate == lit) else {
            return false;
        };
        unmatched.swap_remove(position);
    }
    true
}

/// Use a complete external SAT witness for the model-shrinking step of
/// ordinary `down`.  GipSAT additionally calls `flip_to_none` to minimize the
/// witness, but retaining every true cube literal is conservative and still
/// makes progress because the strengthen clause guarantees at least one is
/// false.  A false kept literal means this drop cannot preserve earlier MIC
/// decisions, exactly as on the native path.
fn shrink_down_cube_from_model(
    cube: &LitVec,
    keep: &GHashSet<Lit>,
    model: &[Lit],
) -> Option<LitVec> {
    let mut shrunk = LitVec::new();
    for &lit in cube.iter() {
        match model_value(model, lit) {
            Some(true) => shrunk.push(lit),
            Some(false) if keep.contains(&lit) => return None,
            Some(false) => {}
            None => return None,
        }
    }
    (shrunk.len() < cube.len()).then_some(shrunk)
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
    fn note_mic_shadow_result(
        &mut self,
        frame: usize,
        prefetched: &MicDropPrefetch,
        cpu_blocked: bool,
    ) {
        debug_assert!(prefetched.proof_neutral);
        let replaceable = match &prefetched.result {
            IncrementalResult::Sat { model } if !cpu_blocked => self.solvers[frame - 1]
                .validate_incremental_sat_model(&prefetched.query, model),
            IncrementalResult::Unsat { core, .. } if cpu_blocked => {
                core_is_assumption_subset(&prefetched.query, core)
            }
            _ => false,
        };
        self.mic_batch_policy.note_shadow_result(replaceable);
        crate::accel::cdcl_host::note_active_mic_shadow_result(replaceable);
    }

    fn note_mic_wave_invalidated(&mut self, wave: &[MicDropPrefetch]) {
        let shadow = wave
            .iter()
            .filter(|prefetched| prefetched.proof_neutral)
            .count();
        self.mic_batch_policy.note_shadow_invalidated(shadow);
        crate::accel::cdcl_host::note_active_mic_shadow_invalidated(shadow);
        crate::accel::cdcl_host::note_active_mic_invalidated(wave.len());
    }

    fn note_mic_prefetch_invalidated(&mut self, prefetched: &MicDropPrefetch) {
        if prefetched.proof_neutral {
            self.mic_batch_policy.note_shadow_invalidated(1);
            crate::accel::cdcl_host::note_active_mic_shadow_invalidated(1);
        }
        crate::accel::cdcl_host::note_active_mic_invalidated(1);
    }

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
        self.mic_batch_policy.note_hardware(
            service_after.1.saturating_sub(service_before.1),
            service_after.2.saturating_sub(service_before.2),
        );
        crate::accel::cdcl_host::note_active_mic_wave(&results);
        // A calibration wave remains attached to the unmodified CPU path so
        // we can observe which of its answers would actually be reached before
        // a cube shrink invalidates the tail. Its answers are never consumed.
        let proof_neutral = route == MicBatchRoute::Probe;
        candidates
            .into_iter()
            .zip(results)
            .map(|((candidate_index, query), result)| MicDropPrefetch {
                candidate_index,
                query,
                result,
                proof_neutral,
            })
            .collect()
    }

    fn consume_mic_drop_result<'a>(
        &mut self,
        frame: usize,
        cube: &LitVec,
        prefetched: &'a MicDropPrefetch,
    ) -> Option<MicDropAnswer<'a>> {
        debug_assert!(!prefetched.proof_neutral);
        match &prefetched.result {
            IncrementalResult::Sat { model } => {
                // Do not import the assignment into GipSAT's mutable trail.
                // The downstream MIC paths consume the exact witness directly
                // and, for CTG, independently certify its predecessor through
                // `TsLift::lift_model`.  This avoids the watcher/model-state
                // mismatch exposed by the first importer prototype.
                let validation_start = Instant::now();
                let accepted = self.solvers[frame - 1]
                    .validate_incremental_sat_model(&prefetched.query, model);
                crate::accel::cdcl_host::note_active_sat_model(
                    accepted,
                    validation_start.elapsed().as_nanos() as u64,
                );
                if accepted {
                    Some(MicDropAnswer::Sat {
                        query: &prefetched.query,
                        model,
                    })
                } else {
                    crate::accel::cdcl_host::note_active_cpu_fallback();
                    crate::accel::cdcl_host::note_active_mic_consumed(false, false);
                    None
                }
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
                cpu_core_len.map(|_| MicDropAnswer::Blocked)
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
                if let Some(prefetched) = prefetched.take() {
                    self.note_mic_prefetch_invalidated(prefetched);
                }
                return None;
            }
            let lemma = LitOrdVec::new(cube.clone());
            if cex
                .iter()
                .any(|(s, t)| !lemma.subsume(s) && lemma.subsume(t))
            {
                if let Some(prefetched) = prefetched.take() {
                    self.note_mic_prefetch_invalidated(prefetched);
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

            let offered = prefetched.take();
            let shadow = offered.filter(|prefetched| prefetched.proof_neutral);
            let active_answer = offered
                .filter(|prefetched| !prefetched.proof_neutral)
                .and_then(|answer| self.consume_mic_drop_result(frame, &cube, answer));
            let used_prefetched = active_answer.is_some();
            let mut active_model = None;
            let blocked = match active_answer {
                Some(MicDropAnswer::Blocked) => true,
                Some(MicDropAnswer::Sat { model, .. }) => {
                    active_model = Some(model);
                    crate::accel::cdcl_host::note_active_mic_consumed(false, true);
                    false
                }
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
            if let Some(shadow) = shadow {
                self.note_mic_shadow_result(frame, shadow, blocked);
            }
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
            if let Some(model) = active_model {
                // The exact model satisfies the strengthen clause, so unless
                // that falsified literal is protected by `keep`, retaining all
                // true cube literals strictly shrinks the next inquiry.  This
                // is the proof-safe external counterpart of GipSAT's more
                // aggressive `flip_to_none` model minimization.
                let Some(cube_new) = shrink_down_cube_from_model(&cube, keep, model) else {
                    return None;
                };
                cube = cube_new;
                continue;
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
                if let Some(prefetched) = prefetched.take() {
                    self.note_mic_prefetch_invalidated(prefetched);
                }
                return None;
            }
            self.statistic.num_down_sat += 1;
            let offered = prefetched.take();
            let shadow = offered.filter(|prefetched| prefetched.proof_neutral);
            let active_answer = offered
                .filter(|prefetched| !prefetched.proof_neutral)
                .and_then(|answer| self.consume_mic_drop_result(frame, &cube, answer));
            let mut active_model = None;
            let mut active_pred = None;
            let blocked = match active_answer {
                Some(MicDropAnswer::Blocked) => true,
                Some(MicDropAnswer::Sat { query, model }) => {
                    if let Some((pred, _inputs)) = self.pred_from_incremental_model(query, model) {
                        active_model = Some(model);
                        active_pred = Some(pred);
                        crate::accel::cdcl_host::note_active_mic_consumed(false, true);
                        false
                    } else {
                        // A complete clause-valid assignment should always
                        // yield at least its full latch predecessor. Fail
                        // closed if the transition-system view disagrees.
                        crate::accel::cdcl_host::note_active_cpu_fallback();
                        crate::accel::cdcl_host::note_active_mic_consumed(false, false);
                        let cpu_start = Instant::now();
                        let blocked = self
                            .blocked(frame, &cube)
                            .in_phase(inductor_trace::Phase::Gen)
                            .with_act_order(false)
                            .with_strengthen()
                            .check();
                        if measure_cpu {
                            self.mic_batch_policy.note_cpu(
                                cpu_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                            );
                        }
                        blocked
                    }
                }
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
            if let Some(shadow) = shadow {
                self.note_mic_shadow_result(frame, shadow, blocked);
            }
            if blocked {
                return Some(self.solvers[frame - 1].inductive_core().unwrap());
            }
            for lit in cube.iter() {
                let value = if let Some(model) = active_model {
                    model_value(model, *lit)
                } else {
                    self.solvers[frame - 1].sat_value(*lit)
                };
                if keep.contains(lit) && !value.is_some_and(|value| value) {
                    return None;
                }
            }
            let model = active_pred.unwrap_or_else(|| self.get_pred(frame, false).0);
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
        // Independent MIC batches lose almost all useful tail work as soon as
        // one early drop shrinks the cube. The full-CDCL MIC-chain command
        // instead rebuilds every later inquiry from the device's current
        // reduced cube. Its output is still only a candidate: one ordinary,
        // unbudgeted live GipSAT solve proves the complete returned cube before
        // IC3 may adopt it.
        let mut mic_chain_answered = false;
        let mut mic_chain_finished = false;
        if parameter.level == 0
            && crate::accel::cdcl_host::mic_chain_enabled()
            && cube.len() >= crate::accel::cdcl_host::mic_chain_min_cube()
        {
            let pairs: Vec<_> = cube
                .iter()
                .map(|lit| (*lit, self.tsctx.next(*lit)))
                .collect();
            if let Some(chain) = crate::accel::cdcl_host::solve_active_mic_chain(
                &self.solvers[frame - 1].dcs,
                &pairs,
                constraint,
            ) {
                mic_chain_answered = true;
                // A complete chain may legitimately return the input cube:
                // every attempted drop was SAT.  Prove that cube once as well
                // so a successful complete command can replace, rather than
                // precede, the CPU literal-by-literal traversal.
                if (chain.complete || chain.cube.len() < cube.len())
                    && !chain.cube.is_empty()
                    && !self.tsctx.cube_subsume_init(&chain.cube)
                {
                    let verify_start = Instant::now();
                    let blocked = self
                        .blocked(frame, &chain.cube)
                        .in_phase(inductor_trace::Phase::Gen)
                        .with_act_order(false)
                        .with_strengthen()
                        .with_constraint(constraint)
                        .check();
                    let verify_ns =
                        verify_start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                    let mut adopted = false;
                    if blocked {
                        let exact = self.solvers[frame - 1]
                            .inductive_core()
                            .unwrap_or_else(|| chain.cube.clone());
                        let exact = LitVec::from_iter(
                            cube.iter().filter(|lit| exact.contains(lit)).copied(),
                        );
                        if !exact.is_empty() && !self.tsctx.cube_subsume_init(&exact) {
                            adopted = true;
                            cube = exact;
                            // Once an exact solve proves the result of a
                            // complete hardware traversal, repeating every
                            // attempted drop on the CPU cannot improve
                            // correctness.  Partial traversals still fall
                            // through to the ordinary loop for the remaining
                            // minimisation work.
                            mic_chain_finished = chain.complete;
                            if mic_chain_finished {
                                crate::accel::cdcl_host::note_active_mic_chain_cpu_replaced(
                                    chain.trials,
                                );
                            }
                            if !mic_chain_finished {
                                self.solvers[frame - 1].unset_domain();
                                self.solvers[frame - 1].set_domain(
                                    self.tsctx
                                        .lits_next(&cube)
                                        .iter()
                                        .copied()
                                        .chain(cube.iter().copied()),
                                );
                            }
                        }
                    }
                    crate::accel::cdcl_host::note_active_mic_chain_validation(
                        adopted, verify_ns,
                    );
                }
            }
        }
        // Cache this once per MIC.  The default CPU path must not repeatedly
        // inspect the environment, clone the parent cube, time queries, or
        // maintain experimental telemetry when the opt-in experiment is off.
        let mic_batch_enabled = crate::accel::cdcl_host::mic_batch_enabled()
            && !mic_chain_answered;
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
        while !mic_chain_finished && i < cube.len() {
            if keep.contains(&cube[i]) {
                i += 1;
                continue;
            }
            if mic_batch_enabled && prefetched_parent.as_ref() != Some(&cube) {
                self.note_mic_wave_invalidated(&prefetched_wave);
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
                self.note_mic_wave_invalidated(&prefetched_wave);
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
        self.note_mic_wave_invalidated(&prefetched_wave);
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
    use super::{MicBatchPolicy, MicBatchRoute, shrink_down_cube_from_model};
    use giputils::hash::GHashSet;
    use logicrs::{LitVec, Var};

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
        for replaceable in [true; 8].into_iter().chain([false; 8]) {
            policy.note_shadow_result(replaceable);
        }
        policy.note_hardware(16, 4_000_000);
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
    fn mic_batch_economics_counts_conclusive_replaceable_work() {
        let mut profitable = cpu_trained();
        for _ in 0..16 {
            profitable.note_shadow_result(true);
        }
        profitable.note_hardware(16, 1_000_000);
        assert_eq!(
            profitable.route_at(16, 8, 1, 4096, 125),
            (MicBatchRoute::Offload, 3_200_000, Some(1_000_000))
        );

        let mut mostly_unknown = cpu_trained();
        mostly_unknown.note_shadow_result(true);
        for _ in 0..15 {
            mostly_unknown.note_shadow_result(false);
        }
        mostly_unknown.note_hardware(16, 1_000_000);
        assert_eq!(
            mostly_unknown.route_at(16, 8, 1, 4096, 125).0,
            MicBatchRoute::Reject
        );
    }

    #[test]
    fn external_sat_model_shrinks_down_without_violating_keep() {
        let a = Var::from(1).lit();
        let b = Var::from(2).lit();
        let c = Var::from(3).lit();
        let cube = LitVec::from([a, b, c]);
        let model = LitVec::from([a, !b, c]);

        let keep_a = GHashSet::from_iter([a]);
        assert_eq!(
            shrink_down_cube_from_model(&cube, &keep_a, &model)
                .unwrap()
                .as_slice(),
            &[a, c]
        );

        let keep_b = GHashSet::from_iter([b]);
        assert!(shrink_down_cube_from_model(&cube, &keep_b, &model).is_none());
        assert!(
            shrink_down_cube_from_model(&cube, &GHashSet::new(), &[a, !b]).is_none()
        );
    }

    #[test]
    fn mic_batch_economics_keeps_calibration_proof_neutral_until_trained() {
        let mut policy = cpu_trained();
        for replaceable in [true; 8].into_iter().chain([false; 8]) {
            policy.note_shadow_result(replaceable);
        }
        policy.note_hardware(16, 4_000_000);
        assert_eq!(
            policy.route_at(16, 8, 2, 4096, 125).0,
            MicBatchRoute::Probe
        );
        policy.note_hardware(16, 4_000_000);
        assert_eq!(
            policy.route_at(16, 8, 2, 4096, 125).0,
            MicBatchRoute::Reject
        );
    }

    #[test]
    fn mic_batch_economics_charges_invalidated_probe_tail() {
        let mut policy = cpu_trained();
        policy.note_shadow_result(true);
        policy.note_shadow_invalidated(15);
        policy.note_hardware(16, 1_000_000);

        let evaluation = policy.route_at(16, 8, 1, 4096, 125);
        assert_eq!(evaluation.0, MicBatchRoute::Reject);
        assert_eq!(evaluation.1, 192_000);
        assert_eq!(evaluation.2, Some(1_000_000));
    }
}
