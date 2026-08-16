use crate::ic3::{
    IC3,
    mab::branch_act,
    mic::{DropVarParameter, MicType},
    proofoblig::ProofObligation,
};
use giputils::TerminateCtrl;
use log::{debug, info};
use logicrs::{Lit, LitOrdVec, LitVec, satif::Satif};
use rand::seq::SliceRandom;
use std::{collections::VecDeque, time::Instant};

use crate::{
    accel::cdcl_host::ActivePreflight,
    gipsat::{IncrementalQuery, IncrementalResult},
};

#[derive(Clone)]
struct CachedBlockInquiry {
    frame: usize,
    state: LitOrdVec,
    query: IncrementalQuery,
    result: IncrementalResult,
    trusted_cpu: bool,
}

struct PendingBlockBatch {
    inquiries: Vec<CachedBlockInquiry>,
    hardware_indices: Vec<usize>,
    handle: Option<std::thread::JoinHandle<Vec<IncrementalResult>>>,
    launched_at: Instant,
}

impl PendingBlockBatch {
    fn is_finished(&self) -> bool {
        self.handle
            .as_ref()
            .is_none_or(std::thread::JoinHandle::is_finished)
    }

    fn finish(&mut self) -> Vec<CachedBlockInquiry> {
        let wait_start = Instant::now();
        let n_hardware = self.hardware_indices.len();
        let results = self
            .handle
            .take()
            .and_then(|handle| handle.join().ok())
            .unwrap_or_else(|| {
                vec![
                    IncrementalResult::Unknown(
                        crate::accel::cdcl::UnknownReason::BackendError,
                    );
                    n_hardware
                ]
            });
        crate::accel::cdcl_host::note_active_block_async_harvest(
            self.launched_at
                .elapsed()
                .as_nanos()
                .min(u64::MAX as u128) as u64,
            wait_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        );
        for (index, result) in self.hardware_indices.iter().copied().zip(results) {
            self.inquiries[index].result = result;
        }
        std::mem::take(&mut self.inquiries)
    }
}

impl Drop for PendingBlockBatch {
    fn drop(&mut self) {
        if self.handle.is_some() {
            let _ = self.finish();
        }
    }
}

#[derive(Default)]
struct BlockBatchCache {
    inquiries: Vec<CachedBlockInquiry>,
    pending: Vec<PendingBlockBatch>,
}

#[derive(Default)]
pub(super) struct BlockAccelPolicy {
    cpu_samples_ns: VecDeque<u64>,
    cpu_samples_scratch_ns: Vec<u64>,
    calibration_samples_ns: Vec<u64>,
    hardware_since_sample: usize,
    calibration_profitable: Option<bool>,
}

impl BlockAccelPolicy {
    fn min_samples() -> usize {
        use std::sync::OnceLock;
        static SAMPLES: OnceLock<usize> = OnceLock::new();
        *SAMPLES.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_BLOCK_COST_SAMPLES")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(8)
                .clamp(1, 64)
        })
    }

    fn min_cpu_ns() -> u64 {
        use std::sync::OnceLock;
        static MIN_NS: OnceLock<u64> = OnceLock::new();
        *MIN_NS.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_BLOCK_MIN_CPU_NS")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(100_000)
        })
    }

    fn disable_cpu_ns() -> u64 {
        use std::sync::OnceLock;
        static MAX_NS: OnceLock<u64> = OnceLock::new();
        *MAX_NS.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_BLOCK_DISABLE_CPU_NS")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or_else(|| Self::min_cpu_ns().saturating_mul(3) / 4)
                .min(Self::min_cpu_ns())
        })
    }

    fn calibration_samples() -> usize {
        use std::sync::OnceLock;
        static SAMPLES: OnceLock<usize> = OnceLock::new();
        *SAMPLES.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_BLOCK_CALIBRATION_SAMPLES")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(8)
                .clamp(1, 64)
        })
    }

    fn resample_interval() -> usize {
        use std::sync::OnceLock;
        static INTERVAL: OnceLock<usize> = OnceLock::new();
        *INTERVAL.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_BLOCK_RESAMPLE")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(64)
                .max(1)
        })
    }

    fn sample_window() -> usize {
        use std::sync::OnceLock;
        static WINDOW: OnceLock<usize> = OnceLock::new();
        *WINDOW.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_BLOCK_COST_WINDOW")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(256)
                .clamp(Self::min_samples(), 4096)
        })
    }

    fn representative_ns(samples: impl Iterator<Item = u64>) -> Option<u64> {
        let mut samples: Vec<_> = samples.collect();
        samples.sort_unstable();
        samples
            .get(samples.len().saturating_sub(1) / 2)
            .copied()
    }

    fn representative_cpu_ns(&mut self, min_samples: usize) -> Option<u64> {
        if self.cpu_samples_ns.len() < min_samples {
            return None;
        }
        // Use the complete retained distribution. A short tail of expensive
        // fifo inquiries is not enough evidence to reverse a route whose
        // typical inquiry is cheap. Reusing the scratch allocation and using
        // linear-time selection avoids sorting or allocating on every sample.
        self.cpu_samples_scratch_ns.clear();
        self.cpu_samples_scratch_ns
            .extend(self.cpu_samples_ns.iter().copied());
        let median_index = self.cpu_samples_scratch_ns.len().saturating_sub(1) / 2;
        let (_, median, _) = self
            .cpu_samples_scratch_ns
            .select_nth_unstable(median_index);
        Some(*median)
    }

    fn should_offload(&self) -> bool {
        self.should_offload_at(Self::resample_interval())
    }

    fn should_offload_at(&self, resample_interval: usize) -> bool {
        self.hardware_since_sample < resample_interval
            && self.calibration_profitable == Some(true)
    }

    fn needs_calibration(&self) -> bool {
        self.calibration_profitable.is_none()
    }

    fn note_calibration(&mut self, elapsed_ns: u64) -> bool {
        let above_threshold = elapsed_ns >= Self::min_cpu_ns();
        let before = self.calibration_profitable;
        let profitable = self.note_calibration_at(
            elapsed_ns,
            Self::calibration_samples(),
            Self::min_cpu_ns(),
        );
        crate::accel::cdcl_host::note_active_block_calibration(
            above_threshold,
            elapsed_ns,
        );
        if before != self.calibration_profitable
            && let Some(enabled) = self.calibration_profitable
        {
            let representative = Self::representative_ns(
                self.calibration_samples_ns.iter().copied(),
            )
            .unwrap_or(0);
            crate::accel::cdcl_host::note_active_block_route_observation(
                representative,
                enabled,
            );
            crate::accel::cdcl_host::note_active_block_route_decision(enabled);
        }
        profitable
    }

    fn note_cpu(&mut self, elapsed_ns: u64) {
        let before = self.calibration_profitable;
        let representative = self.note_cpu_at(
            elapsed_ns,
            Self::sample_window(),
            Self::min_samples(),
            Self::min_cpu_ns(),
            Self::disable_cpu_ns(),
        );
        if let (Some(representative), Some(enabled)) =
            (representative, self.calibration_profitable)
        {
            crate::accel::cdcl_host::note_active_block_route_observation(
                representative,
                enabled,
            );
        }
        if before != self.calibration_profitable
            && let Some(enabled) = self.calibration_profitable
        {
            crate::accel::cdcl_host::note_active_block_route_decision(enabled);
        }
        crate::accel::cdcl_host::note_active_block_cpu_sample(elapsed_ns);
    }

    fn note_calibration_at(
        &mut self,
        elapsed_ns: u64,
        required_samples: usize,
        enable_ns: u64,
    ) -> bool {
        if self.calibration_profitable.is_none() {
            self.calibration_samples_ns.push(elapsed_ns);
            if self.calibration_samples_ns.len() >= required_samples.max(1) {
                let representative = Self::representative_ns(
                    self.calibration_samples_ns.iter().copied(),
                )
                .unwrap_or(0);
                self.calibration_profitable = Some(representative >= enable_ns);
            }
        }
        self.calibration_profitable == Some(true)
    }

    fn note_cpu_at(
        &mut self,
        elapsed_ns: u64,
        window: usize,
        min_samples: usize,
        enable_ns: u64,
        disable_ns: u64,
    ) -> Option<u64> {
        let window = window.max(min_samples).max(1);
        if self.cpu_samples_ns.len() == window {
            self.cpu_samples_ns.pop_front();
        }
        self.cpu_samples_ns.push_back(elapsed_ns);
        self.hardware_since_sample = 0;
        let Some(representative) = self.representative_cpu_ns(min_samples.max(1)) else {
            return None;
        };
        match self.calibration_profitable {
            Some(true) if representative < disable_ns => {
                self.calibration_profitable = Some(false);
            }
            Some(false) if representative >= enable_ns => {
                self.calibration_profitable = Some(true);
            }
            _ => {}
        }
        Some(representative)
    }

    fn note_hardware(&mut self) {
        self.hardware_since_sample = self.hardware_since_sample.saturating_add(1);
    }
}

#[cfg(test)]
mod block_accel_policy_tests {
    use super::BlockAccelPolicy;
    use std::collections::VecDeque;

    #[test]
    fn calibrated_route_requires_periodic_cpu_resampling() {
        let fast = BlockAccelPolicy {
            cpu_samples_ns: VecDeque::from(vec![20_000; 64]),
            ..Default::default()
        };
        assert!(!fast.should_offload_at(64));

        let mut expensive = BlockAccelPolicy {
            cpu_samples_ns: VecDeque::from(vec![200_000; 64]),
            ..Default::default()
        };
        expensive.calibration_profitable = Some(true);
        assert!(expensive.should_offload_at(64));
        for _ in 0..64 {
            expensive.note_hardware();
        }
        assert!(!expensive.should_offload_at(64));

        let calibrated = BlockAccelPolicy {
            calibration_profitable: Some(true),
            ..Default::default()
        };
        assert!(calibrated.should_offload_at(64));
    }

    #[test]
    fn calibration_uses_a_representative_distribution_not_one_outlier() {
        let mut rejected = BlockAccelPolicy::default();
        assert!(!rejected.note_calibration_at(300_000, 3, 100_000));
        assert!(!rejected.note_calibration_at(20_000, 3, 100_000));
        assert!(!rejected.note_calibration_at(30_000, 3, 100_000));
        assert_eq!(rejected.calibration_profitable, Some(false));

        let mut accepted = BlockAccelPolicy::default();
        assert!(!accepted.note_calibration_at(250_000, 3, 100_000));
        assert!(!accepted.note_calibration_at(40_000, 3, 100_000));
        assert!(accepted.note_calibration_at(180_000, 3, 100_000));
        assert_eq!(accepted.calibration_profitable, Some(true));
    }

    #[test]
    fn rolling_median_has_enable_disable_hysteresis() {
        let mut policy = BlockAccelPolicy {
            calibration_profitable: Some(true),
            ..Default::default()
        };
        for _ in 0..8 {
            policy.note_cpu_at(80_000, 8, 8, 100_000, 75_000);
        }
        assert_eq!(policy.calibration_profitable, Some(true));
        for _ in 0..8 {
            policy.note_cpu_at(60_000, 8, 8, 100_000, 75_000);
        }
        assert_eq!(policy.calibration_profitable, Some(false));
        for _ in 0..8 {
            policy.note_cpu_at(90_000, 8, 8, 100_000, 75_000);
        }
        assert_eq!(policy.calibration_profitable, Some(false));
        for _ in 0..8 {
            policy.note_cpu_at(120_000, 8, 8, 100_000, 75_000);
        }
        assert_eq!(policy.calibration_profitable, Some(true));
    }

    #[test]
    fn rolling_median_ignores_a_short_expensive_tail() {
        let mut policy = BlockAccelPolicy {
            calibration_profitable: Some(false),
            ..Default::default()
        };
        for _ in 0..64 {
            policy.note_cpu_at(25_000, 64, 8, 100_000, 75_000);
        }
        for _ in 0..8 {
            policy.note_cpu_at(500_000, 64, 8, 100_000, 75_000);
        }
        assert_eq!(policy.calibration_profitable, Some(false));
        assert_eq!(policy.representative_cpu_ns(8), Some(25_000));
    }
}

impl BlockBatchCache {
    fn window() -> usize {
        use std::sync::OnceLock;
        static WINDOW: OnceLock<usize> = OnceLock::new();
        *WINDOW.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_BLOCK_WINDOW")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(64)
                .clamp(1, 64)
        })
    }

    fn min_context_vars() -> usize {
        use std::sync::OnceLock;
        static MIN_VARS: OnceLock<usize> = OnceLock::new();
        *MIN_VARS.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_BLOCK_MIN_VARS")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(128)
        })
    }

    fn model_lift_enabled() -> bool {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_BLOCK_MODEL_LIFT")
                .ok()
                .map(|value| !matches!(value.as_str(), "0" | "false" | "off"))
                .unwrap_or(true)
        })
    }

    fn async_enabled() -> bool {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_BLOCK_ASYNC")
                .ok()
                .map(|value| !matches!(value.as_str(), "0" | "false" | "off"))
                .unwrap_or(false)
        })
    }

    fn async_depth() -> usize {
        use std::sync::OnceLock;
        static DEPTH: OnceLock<usize> = OnceLock::new();
        *DEPTH.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_BLOCK_ASYNC_DEPTH")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(1)
                .clamp(1, 8)
        })
    }

    fn can_launch(&self) -> bool {
        !Self::async_enabled() || self.pending.len() < Self::async_depth()
    }

    fn contains(&self, frame: usize, state: &LitOrdVec) -> bool {
        self.inquiries
            .iter()
            .any(|entry| entry.frame == frame && &entry.state == state)
            || self.pending.iter().any(|pending| {
                pending
                    .inquiries
                    .iter()
                    .any(|entry| entry.frame == frame && &entry.state == state)
            })
    }

    fn insert(&mut self, inquiries: Vec<CachedBlockInquiry>) {
        for inquiry in inquiries {
            if Self::async_enabled()
                && matches!(inquiry.result, IncrementalResult::Unknown(_))
            {
                continue;
            }
            if let Some(index) = self.inquiries.iter().position(|entry| {
                entry.frame == inquiry.frame && entry.state == inquiry.state
            }) {
                self.inquiries.swap_remove(index);
            }
            self.inquiries.push(inquiry);
        }
        const MAX_CACHED_INQUIRIES: usize = 256;
        if self.inquiries.len() > MAX_CACHED_INQUIRIES {
            let overflow = self.inquiries.len() - MAX_CACHED_INQUIRIES;
            self.inquiries.drain(..overflow);
        }
    }

    fn harvest_ready(&mut self) {
        let mut index = 0;
        while index < self.pending.len() {
            if !self.pending[index].is_finished() {
                index += 1;
                continue;
            }
            let mut pending = self.pending.swap_remove(index);
            let inquiries = pending.finish();
            self.insert(inquiries);
        }
    }

    fn start(&mut self, pending: PendingBlockBatch) {
        debug_assert!(self.pending.len() < Self::async_depth());
        self.pending.push(pending);
    }

    fn take(&mut self, po: &ProofObligation) -> Option<CachedBlockInquiry> {
        let index = self
            .inquiries
            .iter()
            .position(|entry| entry.frame == po.frame && entry.state == po.state)?;
        Some(self.inquiries.swap_remove(index))
    }
}

pub enum BlockResult {
    Success,
    Failure(usize),
    Proved,
    BlockLimitExceeded,
    OverallTimeLimitExceeded,
}

impl IC3 {
    fn full_pred_from_incremental_model(&self, model: &[Lit]) -> Option<(LitVec, Vec<LitVec>)> {
        let value = |lit: Lit| {
            model
                .iter()
                .find(|candidate| candidate.var() == lit.var())
                .map(|candidate| candidate.polarity())
        };
        let inputs = self
            .tsctx
            .input
            .iter()
            .map(|input| {
                let lit = input.lit();
                value(lit).map(|polarity| lit.not_if(!polarity))
            })
            .collect::<Option<LitVec>>()?;
        let state = self
            .tsctx
            .latch
            .iter()
            .map(|latch| {
                let lit = latch.lit();
                value(lit).map(|polarity| lit.not_if(!polarity))
            })
            .collect::<Option<LitVec>>()?;
        Some((state, vec![inputs]))
    }

    fn pred_from_incremental_model(
        &mut self,
        query: &IncrementalQuery,
        model: &[Lit],
    ) -> Option<(LitVec, Vec<LitVec>)> {
        let full = self.full_pred_from_incremental_model(model)?;
        let full_lits = full.0.len();
        let attempted = BlockBatchCache::model_lift_enabled() && query.constraints.is_empty();
        let start = Instant::now();
        let lifted = attempted
            .then(|| {
                let mut target = query.assumptions.clone();
                target.extend(self.ts.constraint.iter().copied());
                target.retain(|lit| self.localabs.refine_has(lit.var()));
                let order = |mut iteration: usize, cube: &mut [Lit]| -> bool {
                    if self.cfg.inn || !self.auxiliary_var.is_empty() {
                        if iteration == 0 {
                            cube.sort_by(|a, b| {
                                self.ts_top_lv[b]
                                    .cmp(&self.ts_top_lv[a])
                                    .then_with(|| self.activity.cmp(b, a))
                            });
                            return true;
                        }
                        iteration -= 1;
                    }
                    match iteration {
                        0 => self.activity.sort_by_activity(cube, false),
                        1 => cube.reverse(),
                        _ => cube.shuffle(&mut self.rng),
                    }
                    true
                };
                self.lift.lift_model(model, target, order)
            })
            .flatten();
        let succeeded = lifted.is_some();
        let result = lifted.unwrap_or(full);
        crate::accel::cdcl_host::note_active_sat_lift(
            attempted,
            succeeded,
            full_lits,
            result.0.len(),
            start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        );
        Some(result)
    }

    fn push_lemma(&mut self, frame: usize, mut cube: LitVec) -> (usize, LitVec) {
        let start = Instant::now();
        let _op = crate::inductor::macro_scope(inductor_trace::Phase::Push, frame);
        for i in frame + 1..=self.level() {
            let _ctx = crate::inductor::set_context(inductor_trace::Phase::Push, i);
            if self.solvers[i - 1].inductive(&cube, true) {
                cube = self.solvers[i - 1].inductive_core().unwrap_or(cube);
            } else {
                return (i, cube);
            }
        }
        self.statistic.block.push_time += start.elapsed();
        (self.level() + 1, cube)
    }

    fn generalize(&mut self, mut po: ProofObligation, mic_type: MicType) -> bool {
        let Some(mut mic) = self.solvers[po.frame - 1].inductive_core() else {
            po.frame += 1;
            self.add_obligation(po.clone());
            return self.add_lemma(po.frame - 1, po.state.as_litvec().clone(), false, Some(po));
        };
        let original_cube_size = mic.len();
        mic = self.mic(po.frame, mic, &[], mic_type);
        let generalized_cube_size = mic.len();
        let (frame, mic) = self.push_lemma(po.frame, mic);
        if self.cfg.mab {
            self.mab_feedback(&po, original_cube_size, generalized_cube_size, frame);
        }
        self.statistic.avg_po_cube_len += po.state.len();
        po.push_to(frame);
        self.add_obligation(po.clone());
        if self.add_lemma(frame - 1, mic.clone(), false, Some(po)) {
            return true;
        }
        false
    }

    #[allow(unused)]
    fn block_with_restart(&mut self) -> BlockResult {
        let mut restart = 0;
        loop {
            let rest_base = luby(2.0, restart);
            match self.block(Some(rest_base * 100.0)) {
                BlockResult::BlockLimitExceeded => {
                    let bt = if let Some(a) = self.obligations.peak() {
                        (a.frame + 2).min(self.level() - 1)
                    } else {
                        self.level() - 1
                    };
                    self.obligations.clear_to(bt);
                    restart += 1;
                    if restart % 10 == 0 {
                        info!("rIC3 restarted {restart} times");
                    }
                }
                r => return r,
            }
        }
    }

    pub fn block(&mut self, limit: Option<f64>) -> BlockResult {
        if crate::accel::cdcl_host::block_batch_enabled() {
            crate::inductor::ThreadCpuTimer::enable();
        }
        let mut noc = 0;
        let mut block_batch = BlockBatchCache::default();
        while let Some(mut po) = self.obligations.pop(self.level()) {
            block_batch.harvest_ready();
            // Remove a previously speculated answer even when this obligation
            // is discarded by one of the cheap guards below. Otherwise stale
            // entries would prevent the cache from naturally draining.
            let mut cached_block = block_batch.take(&po);
            self.render_progress();
            if po.removed {
                continue;
            }
            if let Some(limit) = limit
                && noc as f64 > limit
            {
                return BlockResult::BlockLimitExceeded;
            }
            if self.ctrl.is_terminated() {
                return BlockResult::OverallTimeLimitExceeded;
            }
            if let Some(limit) = self.cfg.time_limit
                && self.statistic.time.time().as_secs() > limit
            {
                return BlockResult::OverallTimeLimitExceeded;
            }
            if self.tsctx.cube_subsume_init(&po.state) {
                if self.cfg.abs_cst || self.cfg.abs_trans {
                    self.add_obligation(po.clone());
                    if self.check_cex_by_bmc(po.depth) {
                        return BlockResult::Failure(po.depth);
                    }
                    self.obligations.clear();
                    self.frame.clear_po();
                    continue;
                } else if po.frame > 0 {
                    let lemma = po.state.as_litvec();
                    debug_assert!(!self.solvers[0].solve(lemma));
                } else {
                    self.add_obligation(po.clone());
                    return BlockResult::Failure(po.depth);
                }
            }
            if let Some((bf, _)) = self.frame.trivial_contained(Some(po.frame), &po.state) {
                if let Some(bf) = bf {
                    po.push_to(bf + 1);
                    self.add_obligation(po);
                }
                continue;
            }
            po.bump_act();
            if self.cfg.drop_po && po.act > 20.0 {
                continue;
            }
            let blocked_start = Instant::now();

            // The queue already contains multiple independent obligations.
            // Snapshot at most one hardware batch before mutating frames or
            // adding successors. Results remain candidates: a SAT model must
            // satisfy the exact live frame when this obligation is consumed,
            // and an UNSAT core is re-proved by exact GipSAT. New lemmas can
            // therefore invalidate speculation but cannot make it unsound.
            let needs_calibration = self.block_accel_policy.needs_calibration();
            let block_cost_eligible = po.frame > 0
                && (needs_calibration || self.block_accel_policy.should_offload());
            if po.frame > 0
                && !block_cost_eligible
                && crate::accel::cdcl_host::block_batch_enabled()
            {
                crate::accel::cdcl_host::note_active_block_cost_rejected();
            }
            if cached_block.is_none()
                && block_batch.can_launch()
                && po.frame > 0
                && crate::accel::cdcl_host::block_batch_enabled()
                && block_cost_eligible
                && self.solvers[po.frame - 1].dcs.num_var()
                    >= BlockBatchCache::min_context_vars()
            {
                if !BlockBatchCache::async_enabled() {
                    // The synchronous policy intentionally refreshes the
                    // frontier after every blocking step. Asynchronous mode
                    // retains completed answers while their obligations wait
                    // in the priority queue.
                    block_batch.inquiries.clear();
                }
                let mut candidates = Vec::new();
                if !BlockBatchCache::async_enabled()
                    || !block_batch.contains(po.frame, &po.state)
                {
                    candidates.push((po.frame, po.state.clone()));
                }
                for candidate in self.obligations.iter().rev() {
                    if candidates.len() >= BlockBatchCache::window() {
                        break;
                    }
                    if candidate.frame == 0
                        || candidate.frame > self.level()
                        || candidate.removed
                        || BlockBatchCache::async_enabled()
                            && block_batch.contains(candidate.frame, &candidate.state)
                        || candidates
                            .iter()
                            .any(|(_, state)| state == &candidate.state)
                    {
                        continue;
                    }
                    candidates.push((candidate.frame, candidate.state.clone()));
                }

                let requests: Vec<_> = candidates
                    .iter()
                    .map(|(frame, state)| {
                        let solver = &self.solvers[*frame - 1];
                        (
                            &solver.dcs,
                            solver.incremental_inductive_query(state, false, vec![]),
                        )
                    })
                    .collect();
                let mut decisions = vec![ActivePreflight::Fpga; requests.len()];
                if crate::accel::cdcl_host::active_enabled()
                    && needs_calibration
                    && requests.len() >= crate::accel::cdcl_host::active_min_batch_size()
                {
                    let sample_index = requests.len() / 2;
                    let (sample_solver, sample_query) = &requests[sample_index];
                    let mut sample_solver = (*sample_solver).clone();
                    let sample_start = crate::inductor::ThreadCpuTimer::start();
                    let sample_result = sample_solver.classify_incremental_exact(sample_query);
                    let sample_ns = sample_start.ns();
                    self.block_accel_policy.note_cpu(sample_ns);
                    let profitable = self.block_accel_policy.note_calibration(sample_ns);
                    decisions[sample_index] = match sample_result {
                        IncrementalResult::Sat { .. } | IncrementalResult::Unsat { .. } => {
                            ActivePreflight::Conclusive(sample_result)
                        }
                        IncrementalResult::Unknown(_) => ActivePreflight::CpuFallback,
                    };
                    if !profitable {
                        for (index, decision) in decisions.iter_mut().enumerate() {
                            if index != sample_index {
                                *decision = ActivePreflight::CpuFallback;
                            }
                        }
                    }
                }
                if crate::accel::cdcl_host::active_enabled() {
                    let sample_requests: Vec<_> = requests
                        .iter()
                        .map(|(solver, query)| (*solver, query))
                        .collect();
                    crate::accel::cdcl_host::active_sample_select_pass(
                        &sample_requests,
                        &mut decisions,
                    );
                }

                let mut results = vec![
                    IncrementalResult::Unknown(
                        crate::accel::cdcl::UnknownReason::BackendError,
                    );
                    requests.len()
                ];
                let mut trusted_cpu = vec![false; requests.len()];
                let mut hardware_indices = Vec::new();
                let mut hardware_requests = Vec::new();
                for (index, decision) in decisions.into_iter().enumerate() {
                    match decision {
                        ActivePreflight::Fpga => {
                            hardware_indices.push(index);
                            hardware_requests.push((requests[index].0, requests[index].1.clone()));
                        }
                        ActivePreflight::Conclusive(result) => {
                            results[index] = result;
                            trusted_cpu[index] = true;
                        }
                        ActivePreflight::CpuFallback => {}
                    }
                }
                let mut inquiries: Vec<_> = candidates
                    .iter()
                    .cloned()
                    .zip(requests.iter().map(|(_, query)| query.clone()))
                    .zip(results.into_iter().zip(trusted_cpu.iter().copied()))
                    .map(
                        |(((frame, state), query), (result, trusted_cpu))| CachedBlockInquiry {
                            frame,
                            state,
                            query,
                            result,
                            trusted_cpu,
                        },
                    )
                    .collect();
                let asynchronous = BlockBatchCache::async_enabled()
                    && crate::accel::cdcl_host::active_enabled()
                    && hardware_indices.len()
                        >= crate::accel::cdcl_host::active_min_batch_size();
                if asynchronous {
                    let prepare_start = Instant::now();
                    let mut solver_frames = Vec::new();
                    let mut owned_solvers = Vec::new();
                    let mut owned_requests = Vec::with_capacity(hardware_indices.len());
                    for index in &hardware_indices {
                        let frame = candidates[*index].0;
                        let solver_index = match solver_frames
                            .iter()
                            .position(|candidate| *candidate == frame)
                        {
                            Some(solver_index) => solver_index,
                            None => {
                                solver_frames.push(frame);
                                owned_solvers.push((*requests[*index].0).clone());
                                owned_solvers.len() - 1
                            }
                        };
                        owned_requests.push((solver_index, requests[*index].1.clone()));
                    }
                    let prepare_ns = prepare_start
                        .elapsed()
                        .as_nanos()
                        .min(u64::MAX as u128) as u64;
                    crate::accel::cdcl_host::note_active_block_async_launch(prepare_ns);
                    let launched_at = Instant::now();
                    let handle = std::thread::spawn(move || {
                        let hardware_requests = owned_requests
                            .into_iter()
                            .map(|(solver_index, query)| (&owned_solvers[solver_index], query))
                            .collect();
                        crate::accel::cdcl_host::solve_active_batch(hardware_requests)
                    });
                    if trusted_cpu.first() == Some(&true) {
                        cached_block = inquiries.first().cloned();
                    }
                    block_batch.start(PendingBlockBatch {
                        inquiries,
                        hardware_indices,
                        handle: Some(handle),
                        launched_at,
                    });
                } else {
                    let hardware_results =
                        crate::accel::cdcl_host::solve_active_batch(hardware_requests);
                    for (index, result) in
                        hardware_indices.iter().copied().zip(hardware_results)
                    {
                        inquiries[index].result = result;
                    }
                    block_batch.insert(inquiries);
                    cached_block = block_batch.take(&po);
                }
            }

            let mut speculative_pred = None;
            let mut speculative_trusted_cpu = false;
            let speculative = cached_block.and_then(|entry| {
                speculative_trusted_cpu = entry.trusted_cpu;
                match (entry.trusted_cpu, entry.result) {
                    (true, IncrementalResult::Sat { model }) => {
                        let validation_start = Instant::now();
                        let accepted = self.solvers[po.frame - 1]
                            .validate_incremental_sat_model(&entry.query, &model);
                        speculative_pred = accepted
                            .then(|| self.pred_from_incremental_model(&entry.query, &model))
                            .flatten();
                        let accepted = speculative_pred.is_some();
                        crate::accel::cdcl_host::note_active_preflight_result(
                            false,
                            accepted,
                            validation_start.elapsed().as_nanos() as u64,
                        );
                        accepted.then_some(false)
                    }
                    (
                        true,
                        IncrementalResult::Unsat {
                            core,
                            used_constraints,
                        },
                    ) => {
                        let restore_start = Instant::now();
                        let accepted = self.solvers[po.frame - 1]
                            .install_incremental_cpu_unsat_core(
                                &po.state,
                                &entry.query,
                                &core,
                                used_constraints,
                            );
                        crate::accel::cdcl_host::note_active_preflight_result(
                            true,
                            accepted,
                            restore_start.elapsed().as_nanos() as u64,
                        );
                        accepted.then_some(true)
                    }
                    (false, IncrementalResult::Sat { model }) => {
                        let validation_start = Instant::now();
                        let accepted = self.solvers[po.frame - 1]
                            .validate_incremental_sat_model(&entry.query, &model);
                        speculative_pred = accepted
                            .then(|| self.pred_from_incremental_model(&entry.query, &model))
                            .flatten();
                        let accepted = speculative_pred.is_some();
                        crate::accel::cdcl_host::note_active_sat_model(
                            accepted,
                            validation_start.elapsed().as_nanos() as u64,
                        );
                        accepted.then_some(false)
                    }
                    (false, IncrementalResult::Unsat { core, .. }) => {
                        let validation_start = Instant::now();
                        let cpu_core_len = self.solvers[po.frame - 1]
                            .validate_incremental_unsat_core(&po.state, &entry.query, &core);
                        crate::accel::cdcl_host::note_active_unsat_core(
                            cpu_core_len.is_some(),
                            entry.query.assumptions.len(),
                            core.len(),
                            cpu_core_len.unwrap_or(0),
                            validation_start.elapsed().as_nanos() as u64,
                        );
                        cpu_core_len.map(|_| true)
                    }
                    (_, IncrementalResult::Unknown(_)) => {
                        crate::accel::cdcl_host::note_active_cpu_fallback();
                        None
                    }
                }
            });
            let blocked = match speculative {
                Some(blocked) => {
                    if !speculative_trusted_cpu {
                        self.block_accel_policy.note_hardware();
                    }
                    blocked
                }
                None => {
                    let cpu_start = crate::inductor::ThreadCpuTimer::start();
                    let blocked = self
                        .blocked(po.frame, &po.state)
                        .in_phase(inductor_trace::Phase::Block)
                        .with_act_order(false)
                        .check();
                    self.block_accel_policy.note_cpu(cpu_start.ns());
                    blocked
                }
            };
            self.statistic.block.blocked_time += blocked_start.elapsed();
            if blocked {
                noc += 1;
                let mic_type = if self.cfg.mab {
                    self.mab_choose_mic_type(&po)
                } else if self.cfg.dynamic {
                    if let Some(act) = branch_act(&po) {
                        const CTG_THRESHOLD: f64 = 10.0;
                        const EXCTG_THRESHOLD: f64 = 40.0;
                        let (limit, max, level) = match act {
                            EXCTG_THRESHOLD.. => {
                                let limit = ((act - EXCTG_THRESHOLD).powf(0.45) * 2.0 + 5.0).round()
                                    as usize;
                                (limit, 5, 1)
                            }
                            CTG_THRESHOLD..EXCTG_THRESHOLD => {
                                let max = (act - CTG_THRESHOLD) as usize / 10 + 2;
                                (1, max, 1)
                            }
                            ..CTG_THRESHOLD => (0, 0, 0),
                            _ => panic!("unreachable activity range (act={act}), possibly NaN"),
                        };
                        let p = DropVarParameter::new(limit, max, level);
                        MicType::DropVar(p)
                    } else {
                        MicType::DropVar(Default::default())
                    }
                } else {
                    MicType::from_config(&self.cfg)
                };
                if self.generalize(po, mic_type) {
                    return BlockResult::Proved;
                }
                debug!("{}", self.frame.statistic(false));
            } else {
                let (model, inputs) = speculative_pred
                    .take()
                    .unwrap_or_else(|| self.get_pred(po.frame, true));
                self.add_obligation(ProofObligation::new(
                    po.frame - 1,
                    LitOrdVec::new(model),
                    inputs,
                    po.depth + 1,
                    Some(po.clone()),
                ));
                self.add_obligation(po);
            }
        }
        BlockResult::Success
    }

    #[allow(unused)]
    fn trivial_block_rec(
        &mut self,
        frame: usize,
        lemma: LitOrdVec,
        constraint: &[LitVec],
        limit: &mut usize,
        parameter: DropVarParameter,
    ) -> bool {
        if frame == 0 {
            return false;
        }
        if self.tsctx.cube_subsume_init(&lemma) {
            return false;
        }
        if *limit == 0 {
            return false;
        }
        *limit -= 1;
        loop {
            if self
                // CTG blocking is reached from `ctg_down` inside generalization,
                // so it counts as Q_gen in the paper's taxonomy even though the
                // query has the shape of a blocking query.
                .blocked(frame, &lemma)
                .in_phase(inductor_trace::Phase::Gen)
                .with_act_order(false)
                .with_strengthen()
                .with_constraint(constraint)
                .check()
            {
                let mut mic = self.solvers[frame - 1].inductive_core().unwrap();
                mic = self.mic(frame, mic, constraint, MicType::DropVar(parameter));
                let (frame, mic) = self.push_lemma(frame, mic);
                self.add_lemma(frame - 1, mic, false, None);
                return true;
            } else {
                if *limit == 0 {
                    return false;
                }
                let model = LitOrdVec::new(self.get_pred(frame, false).0);
                if !self.trivial_block_rec(frame - 1, model, constraint, limit, parameter) {
                    return false;
                }
            }
        }
    }

    pub fn trivial_block(
        &mut self,
        frame: usize,
        lemma: LitOrdVec,
        constraint: &[LitVec],
        parameter: DropVarParameter,
    ) -> bool {
        let mut limit = parameter.limit;
        self.trivial_block_rec(frame, lemma, constraint, &mut limit, parameter)
    }
}

fn luby(y: f64, mut x: usize) -> f64 {
    let mut size = 1;
    let mut seq = 0;
    while size < x + 1 {
        seq += 1;
        size = 2 * size + 1
    }
    while size - 1 != x {
        size = (size - 1) >> 1;
        seq -= 1;
        x %= size;
    }
    y.powi(seq)
}
