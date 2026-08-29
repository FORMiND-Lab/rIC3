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
    context_revision: u64,
    result: IncrementalResult,
    trusted_cpu: bool,
    hardware_selected: bool,
    cached_at: u64,
    cache_age: u64,
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
    epoch: u64,
}

const DEFAULT_BLOCK_ASYNC: bool = false;
const DEFAULT_BLOCK_WAVEFRONT: bool = true;

// Version-5 exact replay instruction codes for the software resident BLOCK
// controller. Keep these values in sync with cdcl_exact_replay_tb.cpp.
const BLOCK_STEP_DISCARD_REMOVED: u32 = 0;
const BLOCK_STEP_LIMIT: u32 = 1;
const BLOCK_STEP_SUBSUME_CLEAR: u32 = 2;
const BLOCK_STEP_TRIVIAL_REQUEUE: u32 = 3;
const BLOCK_STEP_DROP_ACTIVITY: u32 = 4;
const BLOCK_STEP_GENERALIZED: u32 = 5;
const BLOCK_STEP_PROVED: u32 = 6;
const BLOCK_STEP_PREDECESSOR: u32 = 7;
const BLOCK_STEP_SUCCESS: u32 = 8;
const BLOCK_STEP_FAILURE: u32 = 9;
const BLOCK_STEP_TIMEOUT: u32 = 10;

pub(super) struct BlockAccelPolicy {
    cpu_samples_ns: VecDeque<u64>,
    cpu_samples_scratch_ns: Vec<u64>,
    calibration_samples_ns: Vec<u64>,
    hardware_batch_samples: VecDeque<HardwareBatchSample>,
    hardware_batch_ns_scratch: Vec<u64>,
    hardware_batch_queries_scratch: Vec<usize>,
    hardware_since_sample: usize,
    cpu_since_batch_probe: usize,
    calibration_profitable: Option<bool>,
    batch_route_profitable: Option<bool>,
}

#[derive(Clone, Copy, Debug)]
struct HardwareBatchSample {
    service_per_batch_ns: u64,
    queries_per_batch: usize,
    conclusive_percent: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BatchRouteDecision {
    Reject,
    Probe,
    Offload,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BatchRouteEvaluation {
    decision: BatchRouteDecision,
    projected_cpu_ns: u64,
    projected_hardware_ns: Option<u64>,
}

impl Default for BlockAccelPolicy {
    fn default() -> Self {
        Self {
            cpu_samples_ns: VecDeque::new(),
            cpu_samples_scratch_ns: Vec::new(),
            calibration_samples_ns: Vec::new(),
            hardware_batch_samples: VecDeque::new(),
            hardware_batch_ns_scratch: Vec::new(),
            hardware_batch_queries_scratch: Vec::new(),
            hardware_since_sample: 0,
            // Permit exactly one bounded bootstrap probe once the aggregate
            // CPU floor is met; later probes require a real CPU cooldown.
            cpu_since_batch_probe: usize::MAX,
            calibration_profitable: None,
            batch_route_profitable: None,
        }
    }
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

    fn batch_hardware_window() -> usize {
        use std::sync::OnceLock;
        static WINDOW: OnceLock<usize> = OnceLock::new();
        *WINDOW.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_BLOCK_BATCH_COST_WINDOW")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(32)
                .clamp(1, 256)
        })
    }

    fn batch_enable_speedup_pct() -> u64 {
        use std::sync::OnceLock;
        static PERCENT: OnceLock<u64> = OnceLock::new();
        *PERCENT.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_BLOCK_BATCH_SPEEDUP_PCT")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(125)
                .clamp(100, 1000)
        })
    }

    fn batch_disable_speedup_pct() -> u64 {
        use std::sync::OnceLock;
        static PERCENT: OnceLock<u64> = OnceLock::new();
        *PERCENT.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_BLOCK_BATCH_DISABLE_PCT")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(105)
                .clamp(100, Self::batch_enable_speedup_pct())
        })
    }

    fn batch_probe_interval() -> usize {
        use std::sync::OnceLock;
        static INTERVAL: OnceLock<usize> = OnceLock::new();
        *INTERVAL.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_BLOCK_BATCH_PROBE_CPU_QUERIES")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(16_384)
                .max(1)
        })
    }

    fn batch_probe_min_cpu_ns() -> u64 {
        use std::sync::OnceLock;
        static NS: OnceLock<u64> = OnceLock::new();
        *NS.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_BLOCK_BATCH_PROBE_MIN_CPU_NS")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(50_000)
        })
    }

    fn batch_cpu_cap_ns() -> u64 {
        use std::sync::OnceLock;
        static NS: OnceLock<u64> = OnceLock::new();
        *NS.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_BLOCK_BATCH_CPU_CAP_NS")
                .ok()
                .and_then(|value| value.parse().ok())
                // A seconds-long GipSAT tail will still be retried after a
                // bounded hardware UNKNOWN. Cap its bootstrap influence until
                // the measured conclusive ratio proves the FPGA can replace it.
                .unwrap_or(10_000_000)
                .max(1)
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

    fn batch_probe_ready(&self, probe_interval: usize) -> bool {
        self.cpu_since_batch_probe >= probe_interval.max(1)
    }

    fn batch_cpu_per_query_ns(&self, min_samples: usize, cap_ns: u64) -> Option<u64> {
        if self.cpu_samples_ns.len() < min_samples.max(1) {
            return None;
        }
        let total = self.cpu_samples_ns.iter().fold(0u64, |total, sample| {
            total.saturating_add((*sample).min(cap_ns))
        });
        Some(total / self.cpu_samples_ns.len() as u64)
    }

    fn batch_has_minimum_cpu_work_at(
        &mut self,
        n_candidates: usize,
        min_batch: usize,
        min_samples: usize,
        min_batch_cpu_ns: u64,
        cpu_cap_ns: u64,
        probe_min_cpu_ns: u64,
    ) -> bool {
        if n_candidates < min_batch {
            return false;
        }
        self.batch_cpu_per_query_ns(min_samples, cpu_cap_ns)
            .is_some_and(|mean| {
                (mean >= probe_min_cpu_ns || !self.hardware_batch_samples.is_empty())
                    && mean.saturating_mul(n_candidates as u64) >= min_batch_cpu_ns
            })
    }

    fn batch_route_at(
        &mut self,
        n_candidates: usize,
        min_batch: usize,
        min_samples: usize,
        min_batch_cpu_ns: u64,
        cpu_cap_ns: u64,
        probe_min_cpu_ns: u64,
        enable_speedup_pct: u64,
        disable_speedup_pct: u64,
        probe_interval: usize,
    ) -> BatchRouteEvaluation {
        let Some(mean_cpu_ns) = self.batch_cpu_per_query_ns(min_samples, cpu_cap_ns) else {
            return BatchRouteEvaluation {
                decision: BatchRouteDecision::Reject,
                projected_cpu_ns: 0,
                projected_hardware_ns: None,
            };
        };
        let mut projected_cpu_ns = mean_cpu_ns.saturating_mul(n_candidates as u64);
        if n_candidates < min_batch
            || projected_cpu_ns < min_batch_cpu_ns
            || self.hardware_batch_samples.is_empty() && mean_cpu_ns < probe_min_cpu_ns
        {
            self.batch_route_profitable = Some(false);
            return BatchRouteEvaluation {
                decision: BatchRouteDecision::Reject,
                projected_cpu_ns,
                projected_hardware_ns: None,
            };
        }
        if self.hardware_since_sample >= Self::resample_interval() {
            return BatchRouteEvaluation {
                decision: BatchRouteDecision::Reject,
                projected_cpu_ns,
                projected_hardware_ns: None,
            };
        }
        if self.hardware_batch_samples.is_empty() {
            return BatchRouteEvaluation {
                decision: if self.batch_probe_ready(probe_interval) {
                    BatchRouteDecision::Probe
                } else {
                    BatchRouteDecision::Reject
                },
                projected_cpu_ns,
                projected_hardware_ns: None,
            };
        }

        self.hardware_batch_ns_scratch.clear();
        self.hardware_batch_queries_scratch.clear();
        self.hardware_batch_ns_scratch.extend(
            self.hardware_batch_samples
                .iter()
                .map(|sample| sample.service_per_batch_ns),
        );
        self.hardware_batch_queries_scratch.extend(
            self.hardware_batch_samples
                .iter()
                .map(|sample| sample.queries_per_batch),
        );
        let batch_index = self.hardware_batch_ns_scratch.len().saturating_sub(1) / 2;
        let (_, service_per_batch_ns, _) = self
            .hardware_batch_ns_scratch
            .select_nth_unstable(batch_index);
        let query_index = self
            .hardware_batch_queries_scratch
            .len()
            .saturating_sub(1)
            / 2;
        let (_, queries_per_batch, _) = self
            .hardware_batch_queries_scratch
            .select_nth_unstable(query_index);
        let mut conclusive_distribution: Vec<_> = self
            .hardware_batch_samples
            .iter()
            .map(|sample| sample.conclusive_percent)
            .collect();
        let conclusive_index = conclusive_distribution.len().saturating_sub(1) / 2;
        let (_, conclusive_percent, _) =
            conclusive_distribution.select_nth_unstable(conclusive_index);
        projected_cpu_ns = projected_cpu_ns
            .saturating_mul(*conclusive_percent)
            / 100;
        let estimated_batches = n_candidates.div_ceil((*queries_per_batch).max(1));
        let projected_hardware_ns = service_per_batch_ns
            .saturating_mul(estimated_batches as u64);
        let required_speedup = if self.batch_route_profitable == Some(true) {
            disable_speedup_pct.min(enable_speedup_pct)
        } else {
            enable_speedup_pct
        };
        let profitable = u128::from(projected_cpu_ns).saturating_mul(100)
            >= u128::from(projected_hardware_ns)
                .saturating_mul(u128::from(required_speedup));
        self.batch_route_profitable = Some(profitable);
        let decision = if profitable {
            BatchRouteDecision::Offload
        } else if self.batch_probe_ready(probe_interval) {
            BatchRouteDecision::Probe
        } else {
            BatchRouteDecision::Reject
        };
        BatchRouteEvaluation {
            decision,
            projected_cpu_ns,
            projected_hardware_ns: Some(projected_hardware_ns),
        }
    }

    fn batch_route(&mut self, n_candidates: usize) -> BatchRouteEvaluation {
        self.batch_route_at(
            n_candidates,
            crate::accel::cdcl_host::active_min_batch_size(),
            Self::min_samples(),
            crate::accel::cdcl_host::block_min_batch_cpu_ns(),
            Self::batch_cpu_cap_ns(),
            Self::batch_probe_min_cpu_ns(),
            Self::batch_enable_speedup_pct(),
            Self::batch_disable_speedup_pct(),
            Self::batch_probe_interval(),
        )
    }

    fn note_hardware_batch(
        &mut self,
        queries: u64,
        batches: u64,
        service_ns: u64,
        conclusive: u64,
    ) {
        // A selected probe consumed the opportunity even when transport or
        // device setup failed before a measurable batch. Do not hammer an
        // unavailable service on every following frontier.
        self.cpu_since_batch_probe = 0;
        if queries == 0 || batches == 0 || service_ns == 0 {
            return;
        }
        let sample = HardwareBatchSample {
            service_per_batch_ns: service_ns.div_ceil(batches),
            queries_per_batch: usize::try_from(queries.div_ceil(batches))
                .unwrap_or(usize::MAX)
                .max(1),
            conclusive_percent: conclusive.min(queries).saturating_mul(100) / queries,
        };
        if self.hardware_batch_samples.len() == Self::batch_hardware_window() {
            self.hardware_batch_samples.pop_front();
        }
        self.hardware_batch_samples.push_back(sample);
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
        self.cpu_since_batch_probe = self.cpu_since_batch_probe.saturating_add(1);
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
    use super::{
        BatchRouteDecision, BlockAccelPolicy, BlockBatchCache, CachedBlockInquiry,
        DEFAULT_BLOCK_ASYNC, DEFAULT_BLOCK_WAVEFRONT,
    };
    use crate::{
        accel::cdcl::UnknownReason,
        gipsat::{IncrementalQuery, IncrementalResult},
        ic3::proofoblig::ProofObligation,
    };
    use logicrs::{Lit, LitOrdVec, LitVec, Var};
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

    #[test]
    fn aggregate_batch_economics_probes_many_moderate_queries_only() {
        let mut aggregate = BlockAccelPolicy::default();
        for _ in 0..8 {
            aggregate.note_cpu_at(80_000, 64, 8, 100_000, 75_000);
        }
        assert_eq!(
            aggregate
                .batch_route_at(
                    64,
                    8,
                    8,
                    4_000_000,
                    10_000_000,
                    50_000,
                    125,
                    105,
                    256,
                )
                .decision,
            BatchRouteDecision::Probe,
        );

        let mut cheap = BlockAccelPolicy::default();
        for _ in 0..8 {
            cheap.note_cpu_at(25_000, 64, 8, 100_000, 75_000);
        }
        assert_eq!(
            cheap
                .batch_route_at(
                    64,
                    8,
                    8,
                    4_000_000,
                    10_000_000,
                    50_000,
                    125,
                    105,
                    256,
                )
                .decision,
            BatchRouteDecision::Reject,
        );

        let mut cheap_but_numerous = BlockAccelPolicy::default();
        for _ in 0..8 {
            cheap_but_numerous.note_cpu_at(40_000, 64, 8, 100_000, 75_000);
        }
        assert_eq!(
            cheap_but_numerous
                .batch_route_at(
                    64,
                    8,
                    8,
                    2_000_000,
                    10_000_000,
                    50_000,
                    125,
                    105,
                    256,
                )
                .decision,
            BatchRouteDecision::Reject,
        );
    }

    #[test]
    fn measured_batch_service_controls_route_and_probe_cooldown() {
        let mut profitable = BlockAccelPolicy::default();
        for _ in 0..8 {
            profitable.note_cpu_at(200_000, 64, 8, 100_000, 75_000);
        }
        // Two measured batches carried 16 queries each and cost 2 ms each.
        // A 64-query frontier therefore predicts 12.8 ms CPU versus 8 ms FPGA.
        profitable.note_hardware_batch(32, 2, 4_000_000, 32);
        let evaluation =
            profitable.batch_route_at(
                64,
                8,
                8,
                4_000_000,
                10_000_000,
                50_000,
                125,
                105,
                256,
            );
        assert_eq!(evaluation.decision, BatchRouteDecision::Offload);
        assert_eq!(evaluation.projected_cpu_ns, 12_800_000);
        assert_eq!(evaluation.projected_hardware_ns, Some(8_000_000));

        let mut all_unknown = BlockAccelPolicy::default();
        for _ in 0..8 {
            all_unknown.note_cpu_at(200_000, 64, 8, 100_000, 75_000);
        }
        all_unknown.note_hardware_batch(32, 2, 4_000_000, 0);
        let unknown_evaluation =
            all_unknown.batch_route_at(
                64,
                8,
                8,
                4_000_000,
                10_000_000,
                50_000,
                125,
                105,
                256,
            );
        assert_eq!(unknown_evaluation.decision, BatchRouteDecision::Reject);
        assert_eq!(unknown_evaluation.projected_cpu_ns, 0);

        let mut failed_probe = BlockAccelPolicy::default();
        for _ in 0..8 {
            failed_probe.note_cpu_at(200_000, 64, 8, 100_000, 75_000);
        }
        failed_probe.note_hardware_batch(0, 0, 0, 0);
        assert_eq!(
            failed_probe
                .batch_route_at(
                    64,
                    8,
                    8,
                    4_000_000,
                    10_000_000,
                    50_000,
                    125,
                    105,
                    256,
                )
                .decision,
            BatchRouteDecision::Reject,
        );

        let mut slow = BlockAccelPolicy::default();
        for _ in 0..8 {
            slow.note_cpu_at(80_000, 64, 8, 100_000, 75_000);
        }
        slow.note_hardware_batch(64, 1, 5_000_000, 64);
        assert_eq!(
            slow.batch_route_at(
                64,
                8,
                8,
                4_000_000,
                10_000_000,
                50_000,
                125,
                105,
                256,
            )
                .decision,
            BatchRouteDecision::Reject,
        );
        for _ in 0..255 {
            slow.note_cpu_at(80_000, 256, 8, 100_000, 75_000);
        }
        assert_eq!(
            slow.batch_route_at(
                64,
                8,
                8,
                4_000_000,
                10_000_000,
                50_000,
                125,
                105,
                256,
            )
                .decision,
            BatchRouteDecision::Reject,
        );
        slow.note_cpu_at(80_000, 256, 8, 100_000, 75_000);
        assert_eq!(
            slow.batch_route_at(
                64,
                8,
                8,
                4_000_000,
                10_000_000,
                50_000,
                125,
                105,
                256,
            )
                .decision,
            BatchRouteDecision::Probe,
        );
    }

    fn cached_inquiry(
        frame: usize,
        lit: Lit,
        result: IncrementalResult,
    ) -> CachedBlockInquiry {
        CachedBlockInquiry {
            frame,
            state: LitOrdVec::new(LitVec::from([lit])),
            query: IncrementalQuery::new(frame as u32, LitVec::from([lit])),
            context_revision: 7,
            result,
            trusted_cpu: false,
            hardware_selected: true,
            cached_at: 0,
            cache_age: 0,
        }
    }

    #[test]
    fn block_cache_survives_an_unrelated_obligation_detour() {
        let a = Lit::new(Var::from(0), true);
        let b = Lit::new(Var::from(1), true);
        let mut cache = BlockBatchCache::default();
        cache.insert(vec![cached_inquiry(
            2,
            a,
            IncrementalResult::Sat {
                model: LitVec::from([a]),
            },
        )]);

        cache.advance_step();
        let unrelated = ProofObligation::new(
            1,
            LitOrdVec::new(LitVec::from([b])),
            Vec::new(),
            0,
            None,
        );
        assert!(cache.take(&unrelated, true).is_none());

        cache.advance_step();
        let original = ProofObligation::new(
            2,
            LitOrdVec::new(LitVec::from([a])),
            Vec::new(),
            0,
            None,
        );
        let reused = cache.take(&original, true).unwrap();
        assert_eq!(reused.cache_age, 2);
    }

    #[test]
    fn revision_trust_requires_the_exact_captured_formula() {
        assert!(BlockBatchCache::trusted_sat_snapshot_fresh(0, 7, 8, false));
        assert!(!BlockBatchCache::trusted_sat_snapshot_fresh(2, 7, 7, false));
        assert!(BlockBatchCache::trusted_sat_snapshot_fresh(2, 7, 7, true));
        assert!(!BlockBatchCache::trusted_sat_snapshot_fresh(2, 7, 8, true));
    }

    #[test]
    fn block_window_defaults_to_the_multi_aiger_optimum() {
        assert_eq!(BlockBatchCache::window_setting(None), 8);
        assert_eq!(BlockBatchCache::window_setting(Some("16")), 16);
        assert_eq!(BlockBatchCache::window_setting(Some("0")), 1);
        assert_eq!(BlockBatchCache::window_setting(Some("100")), 64);
        assert_eq!(BlockBatchCache::window_setting(Some("bad")), 8);
    }

    #[test]
    fn block_cache_does_not_retain_unknown_answers() {
        let a = Lit::new(Var::from(0), true);
        let mut cache = BlockBatchCache::default();
        cache.insert(vec![cached_inquiry(
            2,
            a,
            IncrementalResult::Unknown(UnknownReason::ConflictBudget),
        )]);
        assert!(!cache.contains(2, &LitOrdVec::new(LitVec::from([a]))));
    }

    #[test]
    fn block_wave_prefers_pruning_unsat_answers() {
        let lit = Lit::new(Var::from(0), true);
        let sat = IncrementalResult::Sat {
            model: LitVec::from([lit]),
        };
        let unsat = IncrementalResult::Unsat {
            core: LitVec::from([lit]),
            used_constraints: false,
        };
        let unknown = IncrementalResult::Unknown(UnknownReason::ConflictBudget);

        assert!(!BlockBatchCache::wavefront_result_eligible(&sat, false));
        assert!(BlockBatchCache::wavefront_result_eligible(&sat, true));
        assert!(BlockBatchCache::wavefront_result_eligible(&unsat, false));
        assert!(!BlockBatchCache::wavefront_result_eligible(
            &unknown, false
        ));
    }

    #[test]
    fn block_wave_is_default_but_background_speculation_is_not() {
        assert!(DEFAULT_BLOCK_WAVEFRONT);
        assert!(!DEFAULT_BLOCK_ASYNC);
    }
}

impl BlockBatchCache {
    fn window_setting(value: Option<&str>) -> usize {
        value
            .and_then(|value| value.parse().ok())
            .unwrap_or(8)
            .clamp(1, 64)
    }

    fn window() -> usize {
        use std::sync::OnceLock;
        static WINDOW: OnceLock<usize> = OnceLock::new();
        *WINDOW.get_or_init(|| {
            let setting = std::env::var("INDUCTOR_CDCL_BLOCK_WINDOW").ok();
            // The fixed ten-AIGER board sweep found 64 wastes speculative
            // answers and 4 destabilizes zipcpu's proof path. Eight reduced
            // completed-model wall time by 12.2% while FPGA service remained
            // roughly 60% of the two main short-model wall times.
            Self::window_setting(setting.as_deref())
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

    fn revision_trust_enabled() -> bool {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_BLOCK_REVISION_TRUST")
                .ok()
                .is_some_and(|value| !matches!(value.as_str(), "0" | "false" | "off"))
        })
    }

    fn trusted_sat_snapshot_fresh(
        cache_age: u64,
        captured_revision: u64,
        current_revision: u64,
        revision_trust: bool,
    ) -> bool {
        cache_age == 0 || revision_trust && captured_revision == current_revision
    }

    fn async_enabled() -> bool {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_BLOCK_ASYNC")
                .ok()
                .map(|value| !matches!(value.as_str(), "0" | "false" | "off"))
                .unwrap_or(DEFAULT_BLOCK_ASYNC)
        })
    }

    fn reuse_enabled() -> bool {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_BLOCK_CACHE_REUSE")
                .ok()
                .map(|value| !matches!(value.as_str(), "0" | "false" | "off"))
                .unwrap_or(false)
        })
    }

    fn reuse_steps() -> u64 {
        use std::sync::OnceLock;
        static STEPS: OnceLock<u64> = OnceLock::new();
        *STEPS.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_BLOCK_CACHE_STEPS")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(4)
                .clamp(1, 256)
        })
    }

    fn wavefront_enabled() -> bool {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_BLOCK_WAVEFRONT")
                .ok()
                .map(|value| !matches!(value.as_str(), "0" | "false" | "off"))
                .unwrap_or(DEFAULT_BLOCK_WAVEFRONT)
        })
    }

    fn wavefront_steps() -> usize {
        use std::sync::OnceLock;
        static STEPS: OnceLock<usize> = OnceLock::new();
        *STEPS.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_BLOCK_WAVEFRONT_STEPS")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(8)
                .clamp(1, 64)
        })
    }

    fn wavefront_include_sat() -> bool {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_BLOCK_WAVEFRONT_SAT")
                .ok()
                .map(|value| !matches!(value.as_str(), "0" | "false" | "off"))
                .unwrap_or(false)
        })
    }

    fn wavefront_result_eligible(result: &IncrementalResult, include_sat: bool) -> bool {
        matches!(result, IncrementalResult::Unsat { .. })
            || include_sat && matches!(result, IncrementalResult::Sat { .. })
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
        for mut inquiry in inquiries {
            let hardware = inquiry.hardware_selected;
            let conclusive = matches!(
                inquiry.result,
                IncrementalResult::Sat { .. } | IncrementalResult::Unsat { .. }
            );
            if hardware {
                crate::accel::cdcl_host::note_active_block_selected_result(conclusive);
            }
            // UNKNOWN cannot answer a later obligation and must not shadow a
            // future retry of the same state/frame pair.
            if !conclusive {
                continue;
            }
            inquiry.cached_at = self.epoch;
            inquiry.cache_age = 0;
            if let Some(index) = self.inquiries.iter().position(|entry| {
                entry.frame == inquiry.frame && entry.state == inquiry.state
            }) {
                let replaced = self.inquiries.swap_remove(index);
                if replaced.hardware_selected {
                    crate::accel::cdcl_host::note_active_block_cache_replaced();
                }
            }
            self.inquiries.push(inquiry);
        }
        const MAX_CACHED_INQUIRIES: usize = 256;
        if self.inquiries.len() > MAX_CACHED_INQUIRIES {
            let overflow = self.inquiries.len() - MAX_CACHED_INQUIRIES;
            let hardware_evicted = self.inquiries[..overflow]
                .iter()
                .filter(|entry| entry.hardware_selected)
                .count();
            crate::accel::cdcl_host::note_active_block_cache_evicted(hardware_evicted);
            self.inquiries.drain(..overflow);
        }
    }

    fn clear_for_refresh(&mut self) {
        let hardware_evicted = self
            .inquiries
            .iter()
            .filter(|entry| entry.hardware_selected)
            .count();
        crate::accel::cdcl_host::note_active_block_cache_evicted(hardware_evicted);
        self.inquiries.clear();
    }

    fn advance_step(&mut self) {
        self.epoch = self.epoch.saturating_add(1);
        if !Self::reuse_enabled() {
            return;
        }
        let max_age = Self::reuse_steps();
        let mut hardware_evicted = 0usize;
        self.inquiries.retain(|entry| {
            let keep = self.epoch.saturating_sub(entry.cached_at) <= max_age;
            if !keep && entry.hardware_selected {
                hardware_evicted += 1;
            }
            keep
        });
        crate::accel::cdcl_host::note_active_block_cache_evicted(hardware_evicted);
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

    fn take(&mut self, po: &ProofObligation, reused: bool) -> Option<CachedBlockInquiry> {
        let index = self
            .inquiries
            .iter()
            .position(|entry| entry.frame == po.frame && entry.state == po.state)?;
        let mut inquiry = self.inquiries.swap_remove(index);
        inquiry.cache_age = self.epoch.saturating_sub(inquiry.cached_at);
        if reused && inquiry.hardware_selected {
            crate::accel::cdcl_host::note_active_block_cache_reused(inquiry.cache_age);
        }
        Some(inquiry)
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

    pub(super) fn pred_from_incremental_model(
        &mut self,
        query: &IncrementalQuery,
        model: &[Lit],
    ) -> Option<(LitVec, Vec<LitVec>)> {
        let full = self.full_pred_from_incremental_model(model)?;
        let full_lits = full.0.len();
        // Query-local clauses select the concrete witness but are not part of
        // predecessor lifting. Native `get_pred` likewise proves only that a
        // state/input premise implies the next-state assumptions through the
        // transition relation, so the independent external-model check is
        // valid even for strengthened MIC inquiries.
        let attempted = BlockBatchCache::model_lift_enabled();
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

    fn restore_block_wave(&mut self, block_wave: &mut VecDeque<ProofObligation>) {
        while let Some(po) = block_wave.pop_front() {
            if !po.removed && !self.obligations.contains(&po) {
                self.obligations.add(po);
            }
        }
    }

    fn note_exact_block_step(
        &self,
        progress: &mut Option<crate::accel::cdcl_host::ExactBlockProgressCapture>,
        event: u32,
    ) {
        let Some(progress) = progress.as_mut() else {
            return;
        };
        crate::accel::cdcl_host::note_exact_block_progress_step(
            Some(progress),
            event,
            self.obligations.progress_snapshot(),
            self.frame.progress_snapshot(),
            self.obligations.progress_image(),
            matches!(event, BLOCK_STEP_GENERALIZED | BLOCK_STEP_PROVED)
                .then(|| self.frame.progress_image()),
        );
    }

    pub fn block(&mut self, limit: Option<f64>) -> BlockResult {
        // One invocation drains an obligation wave and may contain dependent
        // MIC traversals. Scope the operation at its implementation boundary
        // so every caller and every early return retains the same exact-replay
        // root for a future resident BLOCK program.
        let _op = crate::inductor::macro_scope(
            inductor_trace::Phase::Block,
            self.level(),
        );
        let mut progress = crate::accel::cdcl_host::exact_block_progress_enabled()
            .then(|| {
                crate::accel::cdcl_host::begin_exact_block_progress(
                    self.level(),
                    self.obligations.progress_snapshot(),
                    self.frame.progress_snapshot(),
                    self.obligations.progress_image(),
                    self.frame.progress_image(),
                )
            })
            .flatten();
        let result = self.block_inner(limit, &mut progress);
        let (result_code, result_aux) = match &result {
            BlockResult::Success => (0, 0),
            BlockResult::Failure(depth) => (1, (*depth).min(u32::MAX as usize) as u32),
            BlockResult::Proved => (2, 0),
            BlockResult::BlockLimitExceeded => (3, 0),
            BlockResult::OverallTimeLimitExceeded => (4, 0),
        };
        crate::accel::cdcl_host::finish_exact_block_progress(
            progress,
            result_code,
            result_aux,
            self.obligations.progress_snapshot(),
            self.frame.progress_snapshot(),
        );
        result
    }

    fn block_inner(
        &mut self,
        limit: Option<f64>,
        progress: &mut Option<crate::accel::cdcl_host::ExactBlockProgressCapture>,
    ) -> BlockResult {
        if crate::accel::cdcl_host::block_batch_enabled() {
            crate::inductor::ThreadCpuTimer::enable();
        }
        let mut noc = 0;
        let mut block_batch = BlockBatchCache::default();
        let mut block_wave = VecDeque::new();
        let block_wave_enabled = crate::accel::cdcl_host::block_batch_enabled()
            && BlockBatchCache::wavefront_enabled();
        loop {
            let mut from_wave = false;
            let mut next = None;
            while let Some(candidate) = block_wave.pop_front() {
                // A result processed earlier in the wave may have queued a
                // fresher obligation for the same state. Prefer that record
                // (notably its predecessor chain), otherwise consume the
                // reservation removed from the global set at batch creation.
                next = Some(self.obligations.take(&candidate).unwrap_or(candidate));
                from_wave = true;
                break;
            }
            let Some(mut po) = next.or_else(|| self.obligations.pop(self.level())) else {
                self.note_exact_block_step(progress, BLOCK_STEP_SUCCESS);
                break;
            };
            if from_wave {
                crate::accel::cdcl_host::note_active_block_wave_taken();
            }
            block_batch.advance_step();
            block_batch.harvest_ready();
            // Remove a previously speculated answer even when this obligation
            // is discarded by one of the cheap guards below. Otherwise stale
            // entries would prevent the cache from naturally draining.
            let mut cached_block = block_batch.take(&po, true);
            self.render_progress();
            if po.removed {
                self.note_exact_block_step(progress, BLOCK_STEP_DISCARD_REMOVED);
                continue;
            }
            if let Some(limit) = limit
                && noc as f64 > limit
            {
                self.restore_block_wave(&mut block_wave);
                self.note_exact_block_step(progress, BLOCK_STEP_LIMIT);
                return BlockResult::BlockLimitExceeded;
            }
            if self.ctrl.is_terminated() {
                self.restore_block_wave(&mut block_wave);
                self.note_exact_block_step(progress, BLOCK_STEP_TIMEOUT);
                return BlockResult::OverallTimeLimitExceeded;
            }
            if let Some(limit) = self.cfg.time_limit
                && self.statistic.time.time().as_secs() > limit
            {
                self.restore_block_wave(&mut block_wave);
                self.note_exact_block_step(progress, BLOCK_STEP_TIMEOUT);
                return BlockResult::OverallTimeLimitExceeded;
            }
            if self.tsctx.cube_subsume_init(&po.state) {
                if self.cfg.abs_cst || self.cfg.abs_trans {
                    self.add_obligation(po.clone());
                    if self.check_cex_by_bmc(po.depth) {
                        self.note_exact_block_step(progress, BLOCK_STEP_FAILURE);
                        return BlockResult::Failure(po.depth);
                    }
                    self.obligations.clear();
                    self.frame.clear_po();
                    self.note_exact_block_step(progress, BLOCK_STEP_SUBSUME_CLEAR);
                    continue;
                } else if po.frame > 0 {
                    let lemma = po.state.as_litvec();
                    debug_assert!(!self.solvers[0].solve(lemma));
                } else {
                    self.add_obligation(po.clone());
                    self.note_exact_block_step(progress, BLOCK_STEP_FAILURE);
                    return BlockResult::Failure(po.depth);
                }
            }
            if let Some((bf, _)) = self.frame.trivial_contained(Some(po.frame), &po.state) {
                if let Some(bf) = bf {
                    po.push_to(bf + 1);
                    self.add_obligation(po);
                }
                self.note_exact_block_step(progress, BLOCK_STEP_TRIVIAL_REQUEUE);
                continue;
            }
            po.bump_act();
            if self.cfg.drop_po && po.act > 20.0 {
                self.note_exact_block_step(progress, BLOCK_STEP_DROP_ACTIVITY);
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
            let direct_cost_eligible =
                needs_calibration || self.block_accel_policy.should_offload();
            let batch_economics =
                crate::accel::cdcl_host::block_batch_economics_enabled();
            let batch_plan_eligible = batch_economics
                && (needs_calibration
                    || (self.block_accel_policy.batch_route_profitable != Some(false)
                        || self
                            .block_accel_policy
                            .batch_probe_ready(BlockAccelPolicy::batch_probe_interval()))
                        && self.block_accel_policy.batch_has_minimum_cpu_work_at(
                            BlockBatchCache::window(),
                            crate::accel::cdcl_host::active_min_batch_size(),
                            BlockAccelPolicy::min_samples(),
                            crate::accel::cdcl_host::block_min_batch_cpu_ns(),
                            BlockAccelPolicy::batch_cpu_cap_ns(),
                            BlockAccelPolicy::batch_probe_min_cpu_ns(),
                        ));
            let block_cost_eligible =
                po.frame > 0 && (direct_cost_eligible || batch_plan_eligible);
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
                if !BlockBatchCache::async_enabled() && !BlockBatchCache::reuse_enabled() {
                    // The synchronous policy intentionally refreshes the
                    // frontier after every blocking step when reuse is
                    // disabled. The TTL cache is diagnostic because retained
                    // SAT models quickly became stale in the board A/B.
                    block_batch.clear_for_refresh();
                }
                let mut candidates = Vec::new();
                let mut wave_candidates = Vec::new();
                if !block_batch.contains(po.frame, &po.state) {
                    candidates.push((po.frame, po.state.clone()));
                    wave_candidates.push(None);
                }
                let reserve_wave = block_wave_enabled
                    && !BlockBatchCache::async_enabled()
                    && block_wave.is_empty();
                for candidate in self.obligations.iter().rev() {
                    if candidates.len() >= BlockBatchCache::window() {
                        break;
                    }
                    if candidate.frame == 0
                        || candidate.frame > self.level()
                        || candidate.removed
                        || (BlockBatchCache::async_enabled()
                            || BlockBatchCache::reuse_enabled())
                            && block_batch.contains(candidate.frame, &candidate.state)
                        || candidates
                            .iter()
                            .any(|(_, state)| state == &candidate.state)
                    {
                        continue;
                    }
                    candidates.push((candidate.frame, candidate.state.clone()));
                    wave_candidates.push(reserve_wave.then(|| candidate.clone()));
                }

                let queries: Vec<_> = candidates
                    .iter()
                    .map(|(frame, state)| {
                        let solver = &self.solvers[*frame - 1];
                        solver.incremental_inductive_query(state, false, vec![])
                    })
                    .collect();
                let mut decisions = vec![ActivePreflight::Fpga; queries.len()];
                let mut direct_route_profitable = self.block_accel_policy.should_offload();
                if crate::accel::cdcl_host::active_enabled()
                    && needs_calibration
                    && queries.len() >= crate::accel::cdcl_host::active_min_batch_size()
                {
                    let sample_index = queries.len() / 2;
                    let sample_frame = candidates[sample_index].0;
                    let mut sample_solver = self.solvers[sample_frame - 1].dcs.clone();
                    let sample_start = crate::inductor::ThreadCpuTimer::start();
                    let sample_result =
                        sample_solver.classify_incremental_exact(&queries[sample_index]);
                    let sample_ns = sample_start.ns();
                    self.block_accel_policy.note_cpu(sample_ns);
                    direct_route_profitable =
                        self.block_accel_policy.note_calibration(sample_ns);
                    decisions[sample_index] = match sample_result {
                        IncrementalResult::Sat { .. } | IncrementalResult::Unsat { .. } => {
                            ActivePreflight::Conclusive(sample_result)
                        }
                        IncrementalResult::Unknown(_) => ActivePreflight::CpuFallback,
                    };
                }
                let batch_meets_cpu_floor = batch_economics
                    && self.block_accel_policy.batch_has_minimum_cpu_work_at(
                        queries.len(),
                        crate::accel::cdcl_host::active_min_batch_size(),
                        BlockAccelPolicy::min_samples(),
                        crate::accel::cdcl_host::block_min_batch_cpu_ns(),
                        BlockAccelPolicy::batch_cpu_cap_ns(),
                        BlockAccelPolicy::batch_probe_min_cpu_ns(),
                    );
                if (batch_economics && !batch_meets_cpu_floor)
                    || (!batch_economics && !direct_route_profitable)
                {
                    for decision in &mut decisions {
                        if matches!(decision, ActivePreflight::Fpga) {
                            *decision = ActivePreflight::CpuFallback;
                        }
                    }
                }
                if crate::accel::cdcl_host::active_preflight_should_run(queries.len()) {
                    for (index, ((frame, _), query)) in
                        candidates.iter().zip(queries.iter()).enumerate()
                    {
                        if !matches!(decisions[index], ActivePreflight::Fpga) {
                            continue;
                        }
                        decisions[index] =
                            crate::accel::cdcl_host::active_preflight_classify(
                                &mut self.solvers[*frame - 1].dcs,
                                query,
                            );
                        crate::accel::cdcl_host::note_active_block_preflight(
                            &decisions[index],
                        );
                    }
                }
                let requests: Vec<_> = candidates
                    .iter()
                    .zip(queries.iter())
                    .map(|((frame, _), query)| (&self.solvers[*frame - 1].dcs, query.clone()))
                    .collect();
                if crate::accel::cdcl_host::active_enabled() && !batch_economics {
                    let sample_requests: Vec<_> = requests
                        .iter()
                        .map(|(solver, query)| (*solver, query))
                        .collect();
                    crate::accel::cdcl_host::active_sample_select_pass(
                        &sample_requests,
                        &mut decisions,
                    );
                }

                if batch_economics {
                    let selected = decisions
                        .iter()
                        .filter(|decision| matches!(decision, ActivePreflight::Fpga))
                        .count();
                    let route_selected = if selected
                        >= crate::accel::cdcl_host::active_min_batch_size()
                    {
                        let evaluation = self.block_accel_policy.batch_route(selected);
                        let selected = evaluation.decision != BatchRouteDecision::Reject;
                        crate::accel::cdcl_host::note_active_block_batch_economics(
                            evaluation.projected_cpu_ns,
                            evaluation.projected_hardware_ns,
                            evaluation.decision == BatchRouteDecision::Probe,
                            selected,
                        );
                        selected
                    } else {
                        false
                    };
                    if !route_selected {
                        for decision in &mut decisions {
                            if matches!(decision, ActivePreflight::Fpga) {
                                *decision = ActivePreflight::CpuFallback;
                            }
                        }
                        crate::accel::cdcl_host::note_active_block_cost_rejected();
                    }
                }

                let mut results = vec![
                    IncrementalResult::Unknown(
                        crate::accel::cdcl::UnknownReason::BackendError,
                    );
                    requests.len()
                ];
                let mut trusted_cpu = vec![false; requests.len()];
                let mut hardware_selected = vec![false; requests.len()];
                let mut hardware_indices = Vec::new();
                let mut hardware_requests = Vec::new();
                for (index, decision) in decisions.into_iter().enumerate() {
                    match decision {
                        ActivePreflight::Fpga => {
                            hardware_selected[index] = true;
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
                    .zip(requests.iter().map(|(solver, query)| {
                        (query.clone(), solver.incremental_context_revision())
                    }))
                    .zip(
                        results
                            .into_iter()
                            .zip(trusted_cpu.iter().copied())
                            .zip(hardware_selected),
                    )
                    .map(
                        |(
                            ((frame, state), (query, context_revision)),
                            ((result, trusted_cpu), hardware_selected),
                        )| CachedBlockInquiry {
                            frame,
                            state,
                            query,
                            context_revision,
                            result,
                            trusted_cpu,
                            hardware_selected,
                            cached_at: 0,
                            cache_age: 0,
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
                    let service_before =
                        crate::accel::cdcl_host::active_batch_service_snapshot();
                    let hardware_results =
                        crate::accel::cdcl_host::solve_active_batch(hardware_requests);
                    let hardware_conclusive = hardware_results
                        .iter()
                        .filter(|result| {
                            matches!(
                                result,
                                IncrementalResult::Sat { .. }
                                    | IncrementalResult::Unsat { .. }
                            )
                        })
                        .count() as u64;
                    let service_after =
                        crate::accel::cdcl_host::active_batch_service_snapshot();
                    self.block_accel_policy.note_hardware_batch(
                        service_after.1.saturating_sub(service_before.1),
                        service_after.0.saturating_sub(service_before.0),
                        service_after.2.saturating_sub(service_before.2),
                        hardware_conclusive,
                    );
                    for (index, result) in
                        hardware_indices.iter().copied().zip(hardware_results)
                    {
                        inquiries[index].result = result;
                    }
                    if reserve_wave {
                        let mut reserved = 0;
                        for index in hardware_indices.iter().copied() {
                            if reserved >= BlockBatchCache::wavefront_steps() {
                                break;
                            }
                            if !BlockBatchCache::wavefront_result_eligible(
                                &inquiries[index].result,
                                BlockBatchCache::wavefront_include_sat(),
                            ) {
                                continue;
                            }
                            let Some(candidate) = wave_candidates[index].take() else {
                                continue;
                            };
                            if let Some(candidate) = self.obligations.take(&candidate) {
                                block_wave.push_back(candidate);
                                reserved += 1;
                            }
                        }
                        crate::accel::cdcl_host::note_active_block_wave_reserved(reserved);
                    }
                    block_batch.insert(inquiries);
                    cached_block = block_batch.take(&po, false);
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
                            .install_incremental_proven_unsat_core(
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
                        // The production rule accepts only the same block
                        // epoch. The research rule replaces that conservative
                        // proxy with the exact captured per-frame formula
                        // revision; any CPU strengthening changes the revision
                        // and still forces the ordinary fallback.
                        let current_revision = self.solvers[po.frame - 1]
                            .dcs
                            .incremental_context_revision();
                        let snapshot_fresh = BlockBatchCache::trusted_sat_snapshot_fresh(
                            entry.cache_age,
                            entry.context_revision,
                            current_revision,
                            BlockBatchCache::revision_trust_enabled(),
                        );
                        let direct_trust = crate::accel::cdcl_host::active_skip_cpu_check()
                            && snapshot_fresh
                            && !BlockBatchCache::async_enabled();
                        let stale_trusted = crate::accel::cdcl_host::active_skip_cpu_check()
                            && !direct_trust;
                        if direct_trust && entry.cache_age > 0 {
                            crate::accel::cdcl_host::note_active_trusted_sat_revision_reused();
                        }
                        if stale_trusted {
                            crate::accel::cdcl_host::note_active_trusted_sat_stale();
                        }
                        if stale_trusted {
                            // A strengthened frame can invalidate an older SAT
                            // model. In qualified no-replay mode discard it and
                            // run the ordinary CPU inquiry below; do not turn
                            // FPGA validation into a second CPU solve.
                            crate::accel::cdcl_host::note_active_trusted_sat(
                                false,
                                validation_start.elapsed().as_nanos() as u64,
                            );
                            crate::accel::cdcl_host::note_active_block_result_consumed(
                                false,
                                entry.cache_age,
                            );
                            None
                        } else {
                            let accepted = if direct_trust {
                                self.solvers[po.frame - 1]
                                    .trusted_incremental_sat_model_shape(&entry.query, &model)
                            } else {
                                self.solvers[po.frame - 1]
                                    .validate_incremental_sat_model(&entry.query, &model)
                            };
                            speculative_pred = accepted
                                .then(|| self.pred_from_incremental_model(&entry.query, &model))
                                .flatten();
                            let accepted = speculative_pred.is_some();
                            if direct_trust {
                                crate::accel::cdcl_host::note_active_trusted_sat(
                                    accepted,
                                    validation_start.elapsed().as_nanos() as u64,
                                );
                            } else {
                                crate::accel::cdcl_host::note_active_sat_model(
                                    accepted,
                                    validation_start.elapsed().as_nanos() as u64,
                                );
                            }
                            crate::accel::cdcl_host::note_active_block_result_consumed(
                                accepted,
                                entry.cache_age,
                            );
                            accepted.then_some(false)
                        }
                    }
                    (
                        false,
                        IncrementalResult::Unsat {
                            core,
                            used_constraints,
                        },
                    ) => {
                        let validation_start = Instant::now();
                        // UNSAT survives every later frame strengthening, so a
                        // qualified core may be restored even when a background
                        // or retained result is older than the current frame.
                        let direct_trust = crate::accel::cdcl_host::active_skip_cpu_check();
                        let (accepted, cpu_core_len) = if direct_trust {
                            (
                                self.solvers[po.frame - 1].install_incremental_proven_unsat_core(
                                    &po.state,
                                    &entry.query,
                                    &core,
                                    used_constraints,
                                ),
                                0,
                            )
                        } else {
                            let cpu_core_len = self.solvers[po.frame - 1]
                                .validate_incremental_unsat_core(
                                    &po.state,
                                    &entry.query,
                                    &core,
                                );
                            (cpu_core_len.is_some(), cpu_core_len.unwrap_or(0))
                        };
                        if direct_trust {
                            crate::accel::cdcl_host::note_active_trusted_unsat(
                                accepted,
                                entry.query.assumptions.len(),
                                core.len(),
                                validation_start.elapsed().as_nanos() as u64,
                            );
                        } else {
                            crate::accel::cdcl_host::note_active_unsat_core(
                                accepted,
                                entry.query.assumptions.len(),
                                core.len(),
                                cpu_core_len,
                                validation_start.elapsed().as_nanos() as u64,
                            );
                        }
                        crate::accel::cdcl_host::note_active_block_result_consumed(
                            accepted,
                            entry.cache_age,
                        );
                        accepted.then_some(true)
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
                    self.note_exact_block_step(progress, BLOCK_STEP_PROVED);
                    return BlockResult::Proved;
                }
                self.note_exact_block_step(progress, BLOCK_STEP_GENERALIZED);
                debug!("{}", self.frame.statistic(false));
            } else {
                let (model, inputs) = speculative_pred
                    .take()
                    .unwrap_or_else(|| self.get_pred(po.frame, true));
                let pred = ProofObligation::new(
                    po.frame - 1,
                    LitOrdVec::new(model),
                    inputs,
                    po.depth + 1,
                    Some(po.clone()),
                );
                if block_wave_enabled {
                    // Breadth-first processing can make two independent paths
                    // discover the same ordered obligation before either path
                    // reaches the head of the queue. One representative is
                    // sufficient; both SAT predecessor chains are valid.
                    self.add_obligation_if_new(pred);
                    self.add_obligation_if_new(po);
                } else {
                    self.add_obligation(pred);
                    self.add_obligation(po);
                }
                self.note_exact_block_step(progress, BLOCK_STEP_PREDECESSOR);
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
