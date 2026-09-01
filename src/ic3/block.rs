use crate::ic3::{
    IC3,
    frame::{BlockLemmaMutation, begin_block_lemma_journal, finish_block_lemma_journal},
    mab::branch_act,
    mic::{DropVarParameter, MicType},
    proofoblig::ProofObligation,
};
use giputils::TerminateCtrl;
use log::{debug, info};
use logicrs::{Lit, LitOrdVec, LitVec, Var, satif::Satif};
use rand::seq::SliceRandom;
use std::{collections::VecDeque, time::Instant};

use crate::{
    accel::{
        cdcl::{
            BlockFullRootEvent, BlockFullRootResponse, BlockFullRootStatus,
            BlockRootExecutionStatus,
        },
        cdcl_host::ActivePreflight,
    },
    gipsat::{IncrementalQuery, IncrementalResult, decode_batch_results},
};

#[derive(Clone)]
struct CachedBlockInquiry {
    frame: usize,
    state: LitOrdVec,
    query: IncrementalQuery,
    context_revision: u64,
    result: IncrementalResult,
    // True after the SAT witness was converted into predecessor/current proof
    // obligations at the immutable batch revision. Such an answer is already
    // consumed and must never enter the revision-sensitive result cache.
    materialized_sat_committed: bool,
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
                    IncrementalResult::Unknown(crate::accel::cdcl::UnknownReason::BackendError,);
                    n_hardware
                ]
            });
        crate::accel::cdcl_host::note_active_block_async_harvest(
            self.launched_at.elapsed().as_nanos().min(u64::MAX as u128) as u64,
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
const DEFAULT_BLOCK_FULL_ROOT_PORTFOLIO_WORKER: &str = "ic3_abs_all";

fn block_full_root_worker_allowed(worker: Option<&str>, allowlist: Option<&str>) -> bool {
    let Some(worker) = worker else {
        // A standalone IC3 engine has no sibling with which it can contend.
        return true;
    };
    let allowlist = allowlist.map(str::trim);
    if allowlist.is_none() || allowlist == Some("auto") {
        // Root-timeline replay shows that admitting every FPGA-enabled
        // portfolio worker turns the two resident lanes into a calibration
        // queue and regresses most cases. Keep complete-root ownership on the
        // worker that passed the physical admission gate; sibling workers may
        // still use the independent short-inquiry stream path.
        return worker == DEFAULT_BLOCK_FULL_ROOT_PORTFOLIO_WORKER;
    }
    allowlist
        .unwrap()
        .split(',')
        .map(str::trim)
        .any(|candidate| candidate == "*" || candidate == "all" || candidate == worker)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BlockFullRootRoute {
    Cpu,
    Fpga,
}

fn block_fpga_routes(full_root_eligible: bool, route: BlockFullRootRoute) -> (bool, bool) {
    let full_root = full_root_eligible && matches!(route, BlockFullRootRoute::Fpga);
    // Ineligible portfolio workers retain short-inquiry acceleration. An
    // eligible CPU route is different: it is a root-exclusive calibration or
    // fixed CPU decision and therefore suppresses nested FPGA work.
    (full_root, !full_root_eligible || full_root)
}

/// Per-engine admission for the complete resident BLOCK root. The two
/// calibration samples are different algorithm roots: the first executes on
/// GipSAT and the second on the trusted resident controller. No result is
/// solved twice. Once calibrated, the route stays fixed for this IC3 engine.
#[derive(Default)]
pub(super) struct BlockFullRootAdmission {
    cpu_sample: Option<(u64, u64)>,
    fpga_sample: Option<(u64, u64)>,
    route_fpga: Option<bool>,
}

impl BlockFullRootAdmission {
    fn worker_allowed() -> bool {
        let worker = std::env::var("INDUCTOR_CDCL_PORTFOLIO_WORKER").ok();
        let allowlist = std::env::var("INDUCTOR_CDCL_BLOCK_FULL_ROOT_WORKERS").ok();
        block_full_root_worker_allowed(worker.as_deref(), allowlist.as_deref())
    }

    fn eligible() -> bool {
        Self::worker_allowed() && crate::accel::cdcl_host::block_full_root_enabled()
    }

    fn enabled() -> bool {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_BLOCK_FULL_ROOT_ADMISSION")
                .ok()
                .is_some_and(|value| !matches!(value.as_str(), "0" | "false" | "off"))
        }) && Self::eligible()
    }

    fn margin_percent() -> u64 {
        use std::sync::OnceLock;
        static PERCENT: OnceLock<u64> = OnceLock::new();
        *PERCENT.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_BLOCK_FULL_ROOT_ADMISSION_MARGIN_PCT")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(110)
                .clamp(100, 1000)
        })
    }

    fn min_cpu_sample_ns() -> u64 {
        use std::sync::OnceLock;
        static NANOSECONDS: OnceLock<u64> = OnceLock::new();
        *NANOSECONDS.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_BLOCK_FULL_ROOT_MIN_CPU_SAMPLE_US")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(500)
                .min(60_000_000)
                .saturating_mul(1_000)
        })
    }

    fn route(&self) -> BlockFullRootRoute {
        if !Self::eligible() {
            return BlockFullRootRoute::Cpu;
        }
        if !Self::enabled() {
            return BlockFullRootRoute::Fpga;
        }
        self.sampled_route()
    }

    fn sampled_route(&self) -> BlockFullRootRoute {
        if let Some(route_fpga) = self.route_fpga {
            return if route_fpga {
                BlockFullRootRoute::Fpga
            } else {
                BlockFullRootRoute::Cpu
            };
        }
        match (self.cpu_sample, self.fpga_sample) {
            (None, _) => BlockFullRootRoute::Cpu,
            (Some(_), None) => BlockFullRootRoute::Fpga,
            _ => BlockFullRootRoute::Cpu,
        }
    }

    fn observe_at(
        &mut self,
        route: BlockFullRootRoute,
        elapsed_ns: u64,
        inquiries: u64,
        margin_percent: u64,
        min_cpu_sample_ns: u64,
    ) {
        if inquiries == 0 || self.route_fpga.is_some() {
            return;
        }
        match route {
            BlockFullRootRoute::Cpu if self.cpu_sample.is_none() => {
                self.cpu_sample = Some((elapsed_ns, inquiries));
                // A complete-root FPGA probe has a fixed descriptor/context
                // cost. Very short CPU roots are better served by the sibling
                // short-inquiry stream, so do not spend a second root merely
                // to rediscover that fixed-cost boundary.
                if elapsed_ns < min_cpu_sample_ns {
                    self.route_fpga = Some(false);
                    return;
                }
            }
            BlockFullRootRoute::Fpga if self.cpu_sample.is_some() && self.fpga_sample.is_none() => {
                self.fpga_sample = Some((elapsed_ns, inquiries));
            }
            _ => return,
        }
        let (Some((cpu_ns, cpu_inquiries)), Some((fpga_ns, fpga_inquiries))) =
            (self.cpu_sample, self.fpga_sample)
        else {
            return;
        };
        // FPGA must beat CPU by the configured safety margin. Cross
        // multiplication avoids rounding the one-root per-inquiry estimates.
        let fpga_weighted = u128::from(fpga_ns)
            .saturating_mul(u128::from(cpu_inquiries))
            .saturating_mul(u128::from(margin_percent));
        let cpu_weighted = u128::from(cpu_ns)
            .saturating_mul(u128::from(fpga_inquiries))
            .saturating_mul(100);
        self.route_fpga = Some(fpga_weighted < cpu_weighted);
    }

    fn observe(&mut self, route: BlockFullRootRoute, elapsed_ns: u64, inquiries: u64) {
        if !Self::enabled() {
            return;
        }
        let before = self.route_fpga;
        self.observe_at(
            route,
            elapsed_ns,
            inquiries,
            Self::margin_percent(),
            Self::min_cpu_sample_ns(),
        );
        if before.is_none()
            && let Some(route_fpga) = self.route_fpga
        {
            match (self.cpu_sample, self.fpga_sample) {
                (Some((cpu_ns, cpu_inquiries)), Some((fpga_ns, fpga_inquiries))) => {
                    eprintln!(
                        "inductor-cdcl: full-root admission CPU {:.3} ms/{} inquiries, FPGA {:.3} ms/{} inquiries, margin {}%, route {}",
                        cpu_ns as f64 / 1_000_000.0,
                        cpu_inquiries,
                        fpga_ns as f64 / 1_000_000.0,
                        fpga_inquiries,
                        Self::margin_percent(),
                        if route_fpga { "FPGA" } else { "CPU" },
                    );
                }
                (Some((cpu_ns, cpu_inquiries)), None) => {
                    eprintln!(
                        "inductor-cdcl: full-root admission CPU {:.3} ms/{} inquiries below {:.3} ms probe floor, route CPU",
                        cpu_ns as f64 / 1_000_000.0,
                        cpu_inquiries,
                        Self::min_cpu_sample_ns() as f64 / 1_000_000.0,
                    );
                }
                _ => {}
            }
        }
    }
}

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

// Version-7 event operands. Each operation is self-delimiting and is emitted
// where the algorithm actually mutates its queue/frame, never by comparing
// CPU boundary images. Keep these values in sync with the native replayer.
const BLOCK_SEMANTIC_REMOVE_OBLIGATION: u32 = 0;
const BLOCK_SEMANTIC_INSERT_OBLIGATION: u32 = 1;
const BLOCK_SEMANTIC_CLEAR_OBLIGATIONS: u32 = 2;
const BLOCK_SEMANTIC_REMOVE_LEMMA: u32 = 3;
const BLOCK_SEMANTIC_INSERT_LEMMA: u32 = 4;
// A queue-owned pop is distinct from an arbitrary descriptor removal. The
// resident controller must independently choose the same heap head and return
// its opaque tag before CPU proof processing continues.
const BLOCK_SEMANTIC_POP_OBLIGATION: u32 = 7;
type ExactBlockSemanticOps = Option<Vec<Vec<u32>>>;

fn exact_obligation_payload(po: &ProofObligation) -> Vec<u32> {
    let mut payload = Vec::new();
    payload.push(po.state.len().min(u32::MAX as usize) as u32);
    payload.extend(po.state.iter().map(|lit| u32::from(*lit)));
    payload.push(po.input.len().min(u32::MAX as usize) as u32);
    for inputs in &po.input {
        payload.push(inputs.len().min(u32::MAX as usize) as u32);
        payload.extend(inputs.iter().map(|lit| u32::from(*lit)));
    }
    payload
}

fn note_exact_obligation_op(
    operations: &mut ExactBlockSemanticOps,
    command: u32,
    po: &ProofObligation,
) {
    let Some(operations) = operations.as_mut() else {
        return;
    };
    let payload = exact_obligation_payload(po);
    let mut operation = vec![
        command,
        po.frame.min(u32::MAX as usize) as u32,
        po.depth.min(u32::MAX as usize) as u32,
        u32::from(po.removed),
        payload.len().min(u32::MAX as usize) as u32,
    ];
    operation.extend(payload);
    operations.push(operation);
}

fn note_exact_obligation_pop(
    operations: &mut ExactBlockSemanticOps,
    max_frame: usize,
    po: &ProofObligation,
) {
    let Some(operations) = operations.as_mut() else {
        return;
    };
    let payload = exact_obligation_payload(po);
    let mut operation = vec![
        BLOCK_SEMANTIC_POP_OBLIGATION,
        max_frame.min(u32::MAX as usize) as u32,
        po.frame.min(u32::MAX as usize) as u32,
        po.depth.min(u32::MAX as usize) as u32,
        u32::from(po.removed),
        payload.len().min(u32::MAX as usize) as u32,
    ];
    operation.extend(payload);
    operations.push(operation);
}

fn note_exact_clear_obligations(operations: &mut ExactBlockSemanticOps) {
    if let Some(operations) = operations.as_mut() {
        operations.push(vec![BLOCK_SEMANTIC_CLEAR_OBLIGATIONS]);
    }
}

fn note_exact_lemma_mutations(
    operations: &mut ExactBlockSemanticOps,
    mutations: Vec<BlockLemmaMutation>,
) {
    let Some(operations) = operations.as_mut() else {
        return;
    };
    for mutation in mutations {
        let mut operation = vec![
            if mutation.insert {
                BLOCK_SEMANTIC_INSERT_LEMMA
            } else {
                BLOCK_SEMANTIC_REMOVE_LEMMA
            },
            mutation.frame.min(u32::MAX as usize) as u32,
            mutation.lemma.len().min(u32::MAX as usize) as u32 + 1,
            mutation.lemma.len().min(u32::MAX as usize) as u32,
        ];
        operation.extend(mutation.lemma.iter().map(|lit| u32::from(*lit)));
        operations.push(operation);
    }
}

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
        samples.get(samples.len().saturating_sub(1) / 2).copied()
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
        let query_index = self.hardware_batch_queries_scratch.len().saturating_sub(1) / 2;
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
        projected_cpu_ns = projected_cpu_ns.saturating_mul(*conclusive_percent) / 100;
        let estimated_batches = n_candidates.div_ceil((*queries_per_batch).max(1));
        let projected_hardware_ns = service_per_batch_ns.saturating_mul(estimated_batches as u64);
        let required_speedup = if self.batch_route_profitable == Some(true) {
            disable_speedup_pct.min(enable_speedup_pct)
        } else {
            enable_speedup_pct
        };
        let profitable = u128::from(projected_cpu_ns).saturating_mul(100)
            >= u128::from(projected_hardware_ns).saturating_mul(u128::from(required_speedup));
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
        self.hardware_since_sample < resample_interval && self.calibration_profitable == Some(true)
    }

    fn needs_calibration(&self) -> bool {
        self.calibration_profitable.is_none()
    }

    fn note_calibration(&mut self, elapsed_ns: u64) -> bool {
        let above_threshold = elapsed_ns >= Self::min_cpu_ns();
        let before = self.calibration_profitable;
        let profitable =
            self.note_calibration_at(elapsed_ns, Self::calibration_samples(), Self::min_cpu_ns());
        crate::accel::cdcl_host::note_active_block_calibration(above_threshold, elapsed_ns);
        if before != self.calibration_profitable
            && let Some(enabled) = self.calibration_profitable
        {
            let representative =
                Self::representative_ns(self.calibration_samples_ns.iter().copied()).unwrap_or(0);
            crate::accel::cdcl_host::note_active_block_route_observation(representative, enabled);
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
        if let (Some(representative), Some(enabled)) = (representative, self.calibration_profitable)
        {
            crate::accel::cdcl_host::note_active_block_route_observation(representative, enabled);
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
                let representative =
                    Self::representative_ns(self.calibration_samples_ns.iter().copied())
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
        BatchRouteDecision, BlockAccelPolicy, BlockBatchCache, BlockFullRootAdmission,
        BlockFullRootRoute, CachedBlockInquiry, DEFAULT_BLOCK_ASYNC, DEFAULT_BLOCK_WAVEFRONT,
        block_fpga_routes, block_full_root_worker_allowed,
    };
    use crate::{
        accel::cdcl::UnknownReason,
        gipsat::{IncrementalQuery, IncrementalResult},
        ic3::proofoblig::ProofObligation,
    };
    use logicrs::{Lit, LitOrdVec, LitVec, Var};
    use std::collections::VecDeque;

    #[test]
    fn full_root_admission_uses_one_distinct_sample_per_route() {
        let mut policy = BlockFullRootAdmission::default();
        policy.observe_at(BlockFullRootRoute::Cpu, 20_000_000, 10, 110, 0);
        // A second CPU root is not used as the FPGA sample.
        policy.observe_at(BlockFullRootRoute::Cpu, 1, 10, 110, 0);
        assert_eq!(policy.cpu_sample, Some((20_000_000, 10)));
        assert_eq!(policy.fpga_sample, None);
        policy.observe_at(BlockFullRootRoute::Fpga, 10_000_000, 10, 110, 0);
        assert_eq!(policy.route_fpga, Some(true));
    }

    #[test]
    fn full_root_admission_skips_fpga_probe_below_cpu_cost_floor() {
        let mut policy = BlockFullRootAdmission::default();
        policy.observe_at(BlockFullRootRoute::Cpu, 499_999, 10, 110, 500_000);
        assert_eq!(policy.cpu_sample, Some((499_999, 10)));
        assert_eq!(policy.fpga_sample, None);
        assert_eq!(policy.route_fpga, Some(false));
        assert_eq!(policy.sampled_route(), BlockFullRootRoute::Cpu);
    }

    #[test]
    fn portfolio_full_root_defaults_to_the_physically_qualified_worker() {
        assert!(block_full_root_worker_allowed(None, None));
        assert!(block_full_root_worker_allowed(Some("ic3_abs_all"), None));
        assert!(block_full_root_worker_allowed(
            Some("ic3_abs_all"),
            Some("auto")
        ));
        assert!(!block_full_root_worker_allowed(Some("ic3"), None));
        assert!(!block_full_root_worker_allowed(
            Some("ic3_ctg_limit"),
            Some("auto")
        ));
    }

    #[test]
    fn portfolio_full_root_allowlist_is_explicit_and_exact() {
        assert!(block_full_root_worker_allowed(
            Some("ic3"),
            Some("ic3, ic3_abs_all")
        ));
        assert!(!block_full_root_worker_allowed(
            Some("ic3_inn"),
            Some("ic3, ic3_abs_all")
        ));
        assert!(block_full_root_worker_allowed(
            Some("anything"),
            Some("all")
        ));
        assert!(block_full_root_worker_allowed(Some("anything"), Some("*")));
        assert!(!block_full_root_worker_allowed(
            Some("ic3_abs_all"),
            Some("")
        ));
    }

    #[test]
    fn full_root_allowlist_does_not_disable_short_inquiry_acceleration() {
        assert_eq!(
            block_fpga_routes(false, BlockFullRootRoute::Cpu),
            (false, true)
        );
        assert_eq!(
            block_fpga_routes(true, BlockFullRootRoute::Cpu),
            (false, false)
        );
        assert_eq!(
            block_fpga_routes(true, BlockFullRootRoute::Fpga),
            (true, true)
        );
    }

    #[test]
    fn full_root_admission_rejects_slow_or_empty_fpga_samples() {
        let mut policy = BlockFullRootAdmission::default();
        policy.observe_at(BlockFullRootRoute::Cpu, 10_000_000, 10, 110, 0);
        policy.observe_at(BlockFullRootRoute::Fpga, 1, 0, 110, 0);
        assert_eq!(policy.fpga_sample, None);
        policy.observe_at(BlockFullRootRoute::Fpga, 12_000_000, 10, 110, 0);
        assert_eq!(policy.route_fpga, Some(false));
    }

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
                .batch_route_at(64, 8, 8, 4_000_000, 10_000_000, 50_000, 125, 105, 256,)
                .decision,
            BatchRouteDecision::Probe,
        );

        let mut cheap = BlockAccelPolicy::default();
        for _ in 0..8 {
            cheap.note_cpu_at(25_000, 64, 8, 100_000, 75_000);
        }
        assert_eq!(
            cheap
                .batch_route_at(64, 8, 8, 4_000_000, 10_000_000, 50_000, 125, 105, 256,)
                .decision,
            BatchRouteDecision::Reject,
        );

        let mut cheap_but_numerous = BlockAccelPolicy::default();
        for _ in 0..8 {
            cheap_but_numerous.note_cpu_at(40_000, 64, 8, 100_000, 75_000);
        }
        assert_eq!(
            cheap_but_numerous
                .batch_route_at(64, 8, 8, 2_000_000, 10_000_000, 50_000, 125, 105, 256,)
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
            profitable.batch_route_at(64, 8, 8, 4_000_000, 10_000_000, 50_000, 125, 105, 256);
        assert_eq!(evaluation.decision, BatchRouteDecision::Offload);
        assert_eq!(evaluation.projected_cpu_ns, 12_800_000);
        assert_eq!(evaluation.projected_hardware_ns, Some(8_000_000));

        let mut all_unknown = BlockAccelPolicy::default();
        for _ in 0..8 {
            all_unknown.note_cpu_at(200_000, 64, 8, 100_000, 75_000);
        }
        all_unknown.note_hardware_batch(32, 2, 4_000_000, 0);
        let unknown_evaluation =
            all_unknown.batch_route_at(64, 8, 8, 4_000_000, 10_000_000, 50_000, 125, 105, 256);
        assert_eq!(unknown_evaluation.decision, BatchRouteDecision::Reject);
        assert_eq!(unknown_evaluation.projected_cpu_ns, 0);

        let mut failed_probe = BlockAccelPolicy::default();
        for _ in 0..8 {
            failed_probe.note_cpu_at(200_000, 64, 8, 100_000, 75_000);
        }
        failed_probe.note_hardware_batch(0, 0, 0, 0);
        assert_eq!(
            failed_probe
                .batch_route_at(64, 8, 8, 4_000_000, 10_000_000, 50_000, 125, 105, 256,)
                .decision,
            BatchRouteDecision::Reject,
        );

        let mut slow = BlockAccelPolicy::default();
        for _ in 0..8 {
            slow.note_cpu_at(80_000, 64, 8, 100_000, 75_000);
        }
        slow.note_hardware_batch(64, 1, 5_000_000, 64);
        assert_eq!(
            slow.batch_route_at(64, 8, 8, 4_000_000, 10_000_000, 50_000, 125, 105, 256,)
                .decision,
            BatchRouteDecision::Reject,
        );
        for _ in 0..255 {
            slow.note_cpu_at(80_000, 256, 8, 100_000, 75_000);
        }
        assert_eq!(
            slow.batch_route_at(64, 8, 8, 4_000_000, 10_000_000, 50_000, 125, 105, 256,)
                .decision,
            BatchRouteDecision::Reject,
        );
        slow.note_cpu_at(80_000, 256, 8, 100_000, 75_000);
        assert_eq!(
            slow.batch_route_at(64, 8, 8, 4_000_000, 10_000_000, 50_000, 125, 105, 256,)
                .decision,
            BatchRouteDecision::Probe,
        );
    }

    fn cached_inquiry(frame: usize, lit: Lit, result: IncrementalResult) -> CachedBlockInquiry {
        CachedBlockInquiry {
            frame,
            state: LitOrdVec::new(LitVec::from([lit])),
            query: IncrementalQuery::new(frame as u32, LitVec::from([lit])),
            context_revision: 7,
            result,
            materialized_sat_committed: false,
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
        let unrelated =
            ProofObligation::new(1, LitOrdVec::new(LitVec::from([b])), Vec::new(), 0, None);
        assert!(cache.take(&unrelated, true).is_none());

        cache.advance_step();
        let original =
            ProofObligation::new(2, LitOrdVec::new(LitVec::from([a])), Vec::new(), 0, None);
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
    fn block_cache_does_not_retain_committed_sat_answers() {
        let a = Lit::new(Var::from(0), true);
        let mut inquiry = cached_inquiry(
            2,
            a,
            IncrementalResult::Sat {
                model: LitVec::from([a]),
            },
        );
        inquiry.materialized_sat_committed = true;
        let mut cache = BlockBatchCache::default();
        cache.insert(vec![inquiry]);
        assert!(!cache.contains(2, &LitOrdVec::new(LitVec::from([a]))));
    }

    #[test]
    fn sat_materialization_targets_small_contexts() {
        assert!(BlockBatchCache::materialize_sat_context_eligible(512, 512));
        assert!(!BlockBatchCache::materialize_sat_context_eligible(513, 512));
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
        assert!(!BlockBatchCache::wavefront_result_eligible(&unknown, false));
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

    fn materialize_sat_wave_enabled() -> bool {
        use std::sync::OnceLock;
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_BLOCK_MATERIALIZE_SAT_WAVE")
                .ok()
                .is_some_and(|value| !matches!(value.as_str(), "0" | "false" | "off"))
        })
    }

    fn materialize_sat_max_context_vars() -> usize {
        use std::sync::OnceLock;
        static MAX_VARS: OnceLock<usize> = OnceLock::new();
        *MAX_VARS.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_BLOCK_MATERIALIZE_SAT_MAX_VARS")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(512)
                .clamp(1, 32_768)
        })
    }

    fn materialize_sat_context_eligible(context_vars: usize, max_vars: usize) -> bool {
        context_vars <= max_vars
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
            if inquiry.materialized_sat_committed {
                crate::accel::cdcl_host::note_active_block_result_consumed(true, 0);
                continue;
            }
            // UNKNOWN cannot answer a later obligation and must not shadow a
            // future retry of the same state/frame pair.
            if !conclusive {
                continue;
            }
            inquiry.cached_at = self.epoch;
            inquiry.cache_age = 0;
            if let Some(index) = self
                .inquiries
                .iter()
                .position(|entry| entry.frame == inquiry.frame && entry.state == inquiry.state)
            {
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
    fn replay_resident_full_root(
        &mut self,
        response: &BlockFullRootResponse,
        source_keys: &[Vec<u32>],
    ) -> Result<(bool, Vec<Vec<u32>>), String> {
        if response.events.len() != source_keys.len() {
            return Err("resident full-root event/source journal length mismatch".to_string());
        }
        let n_var = self.tsctx.num_var();
        let decode = |words: &[u32]| -> Result<LitVec, String> {
            words
                .iter()
                .map(|word| {
                    let variable = Var::from(*word >> 1);
                    (usize::from(variable) < n_var)
                        .then(|| Lit::new(variable, *word & 1 == 0))
                        .ok_or_else(|| format!("resident full-root literal out of range: {word}"))
                })
                .collect()
        };
        let mut proved = false;
        let mut semantic_ops = Some(Vec::new());
        for (event, source_key) in response.events.iter().zip(source_keys) {
            let source = self
                .obligations
                .take_resident_key(source_key, self.level())
                .ok_or_else(|| "resident full-root consumed unknown CPU obligation".to_string())?;
            match event {
                BlockFullRootEvent::SatPredecessor {
                    parent_descriptor_handle,
                    child_descriptor_handle,
                    frame,
                    depth,
                    state,
                    input,
                    ..
                } => {
                    if source.frame == 0
                        || *frame as usize + 1 != source.frame
                        || *depth as usize != source.depth + 1
                    {
                        return Err("resident SAT predecessor metadata mismatch".to_string());
                    }
                    let state = decode(state)?;
                    let input = decode(input)?;
                    let predecessor = ProofObligation::new(
                        *frame as usize,
                        LitOrdVec::new(state.clone()),
                        vec![input],
                        *depth as usize,
                        Some(source.clone()),
                    );
                    let parent_inserted = self.add_obligation_if_new(source.clone());
                    let child_inserted = self.add_obligation_if_new(predecessor);
                    if parent_inserted != (*parent_descriptor_handle != u32::MAX)
                        || child_inserted != (*child_descriptor_handle != u32::MAX)
                    {
                        return Err(format!(
                            "resident SAT predecessor deduplication mismatch: parent CPU/device {parent_inserted}/{}, child CPU/device {child_inserted}/{}, source frame/depth/state {}/{}/{:?}, child frame/depth/state {}/{}/{:?}",
                            *parent_descriptor_handle != u32::MAX,
                            *child_descriptor_handle != u32::MAX,
                            source.frame,
                            source.depth,
                            source.state,
                            frame,
                            depth,
                            state,
                        ));
                    }
                }
                BlockFullRootEvent::UnsatLemma { frame, cube, .. } => {
                    if source.frame == 0 || *frame as usize != source.frame {
                        return Err("resident UNSAT lemma frame mismatch".to_string());
                    }
                    let cube = decode(cube)?;
                    if cube.is_empty() || self.tsctx.cube_subsume_init(&cube) {
                        return Err("resident UNSAT lemma violates Init guard".to_string());
                    }
                    // Simulation oracle: until the fused root is qualified on
                    // the AIGER matrix, prove every device-produced lemma
                    // against an independent clone of the exact CPU frame.
                    // This is deliberately outside the intended production
                    // fast path and can be removed once the architecture has
                    // zero oracle disagreements over repeated matrices.
                    if std::env::var_os("INDUCTOR_CDCL_BLOCK_FULL_ROOT_ORACLE").is_some() {
                        let solver = &self.solvers[*frame as usize - 1];
                        let source_query = solver.incremental_inductive_query(
                            source.state.as_litvec(),
                            false,
                            vec![],
                        );
                        let mut source_checker = solver.dcs.clone();
                        if !matches!(
                            source_checker.classify_incremental_exact(&source_query),
                            IncrementalResult::Unsat { .. }
                        ) {
                            return Err(format!(
                                "resident Q_block UNSAT failed exact CPU oracle at frame {} (source {} literals, lemma {} literals, CPU assumptions {:?})",
                                frame,
                                source.state.len(),
                                cube.len(),
                                source_query
                                    .assumptions
                                    .iter()
                                    .map(|literal| u32::from(*literal))
                                    .collect::<Vec<_>>()
                            ));
                        }
                        // MIC proves relative induction with the candidate
                        // blocking clause installed: F[i-1] & !cube & cube'.
                        // Checking strengthen=false here incorrectly rejects a
                        // valid generalized lemma that is not blocked by the
                        // weaker Q_block formula alone.
                        let query = solver.incremental_inductive_query(&cube, true, vec![]);
                        let mut checker = solver.dcs.clone();
                        if !matches!(
                            checker.classify_incremental_exact(&query),
                            IncrementalResult::Unsat { .. }
                        ) {
                            return Err(format!(
                                "resident UNSAT lemma failed exact CPU oracle at frame {} ({} literals)",
                                frame,
                                cube.len()
                            ));
                        }
                    }
                    // The card has already inserted this exact lemma. Replay
                    // add_lemma only to rebuild FrameLemma/proof ownership and
                    // capture the canonicalization removals it performs. One
                    // matching insertion is consumed by the resident event;
                    // all other mutations (normally subsumed old lemmas) must
                    // still be sent to the semantic handle state.
                    begin_block_lemma_journal(true);
                    proved |= self.add_lemma(*frame as usize, cube.clone(), false, Some(source));
                    let mutations = finish_block_lemma_journal();
                    let mut consumed_resident_insert = false;
                    let mut normalization = Vec::new();
                    for mutation in mutations {
                        if mutation.insert
                            && !consumed_resident_insert
                            && mutation.frame == *frame as usize
                            && mutation.lemma == cube
                        {
                            consumed_resident_insert = true;
                        } else {
                            normalization.push(mutation);
                        }
                    }
                    if !consumed_resident_insert {
                        return Err(
                            "resident UNSAT lemma was not installed in the CPU frame".to_string()
                        );
                    }
                    note_exact_lemma_mutations(&mut semantic_ops, normalization);
                }
            }
        }
        Ok((proved, semantic_ops.unwrap_or_default()))
    }

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

    fn generalize(
        &mut self,
        mut po: ProofObligation,
        mic_type: MicType,
        semantic_ops: &mut ExactBlockSemanticOps,
    ) -> bool {
        begin_block_lemma_journal(semantic_ops.is_some());
        let Some(mut mic) = self.solvers[po.frame - 1].inductive_core() else {
            po.frame += 1;
            self.add_obligation(po.clone());
            note_exact_obligation_op(semantic_ops, BLOCK_SEMANTIC_INSERT_OBLIGATION, &po);
            let proved =
                self.add_lemma(po.frame - 1, po.state.as_litvec().clone(), false, Some(po));
            note_exact_lemma_mutations(semantic_ops, finish_block_lemma_journal());
            return proved;
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
        note_exact_obligation_op(semantic_ops, BLOCK_SEMANTIC_INSERT_OBLIGATION, &po);
        let proved = self.add_lemma(frame - 1, mic.clone(), false, Some(po));
        note_exact_lemma_mutations(semantic_ops, finish_block_lemma_journal());
        if proved {
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

    fn restore_block_wave(
        &mut self,
        block_wave: &mut VecDeque<ProofObligation>,
        semantic_ops: &mut ExactBlockSemanticOps,
    ) {
        while let Some(po) = block_wave.pop_front() {
            if !po.removed && !self.obligations.contains(&po) {
                note_exact_obligation_op(semantic_ops, BLOCK_SEMANTIC_INSERT_OBLIGATION, &po);
                self.obligations.add(po);
            }
        }
    }

    fn note_exact_block_step(
        &self,
        progress: &mut Option<crate::accel::cdcl_host::ExactBlockProgressCapture>,
        event: u32,
        semantic_ops: ExactBlockSemanticOps,
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
            semantic_ops.unwrap_or_default(),
        );
    }

    pub fn block(&mut self, limit: Option<f64>) -> BlockResult {
        // One invocation drains an obligation wave and may contain dependent
        // MIC traversals. Scope the operation at its implementation boundary
        // so every caller and every early return retains the same exact-replay
        // root for a future resident BLOCK program.
        let _op = crate::inductor::macro_scope(inductor_trace::Phase::Block, self.level());
        let admission_route = self.block_full_root_admission.route();
        let admission_enabled = BlockFullRootAdmission::enabled();
        let full_root_eligible = BlockFullRootAdmission::eligible();
        let (full_root_fpga_enabled, leaf_fpga_enabled) =
            block_fpga_routes(full_root_eligible, admission_route);
        let root_started = admission_enabled.then(Instant::now);
        let cpu_query_counter = (admission_enabled
            && matches!(admission_route, BlockFullRootRoute::Cpu))
        .then(crate::inductor::RootQueryCounter::start);
        let mut resident_full_root_inquiries = 0u64;
        let root_timeline = crate::accel::cdcl_host::begin_block_root_timeline(self.level());
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
        let result = self.block_inner(
            limit,
            &mut progress,
            full_root_fpga_enabled,
            leaf_fpga_enabled,
            &mut resident_full_root_inquiries,
        );
        let root_elapsed_ns =
            root_started.map(|started| started.elapsed().as_nanos().min(u64::MAX as u128) as u64);
        let (result_code, result_aux) = match &result {
            BlockResult::Success => (0, 0),
            BlockResult::Failure(depth) => (1, (*depth).min(u32::MAX as usize) as u32),
            BlockResult::Proved => (2, 0),
            BlockResult::BlockLimitExceeded => (3, 0),
            BlockResult::OverallTimeLimitExceeded => (4, 0),
        };
        crate::accel::cdcl_host::finish_block_root_timeline(root_timeline, result_code, result_aux);
        crate::accel::cdcl_host::finish_exact_block_progress(
            progress,
            result_code,
            result_aux,
            self.obligations.progress_snapshot(),
            self.frame.progress_snapshot(),
        );
        if let Some(root_elapsed_ns) = root_elapsed_ns {
            let inquiries = match admission_route {
                BlockFullRootRoute::Cpu => cpu_query_counter
                    .map(crate::inductor::RootQueryCounter::finish)
                    .unwrap_or(0),
                BlockFullRootRoute::Fpga => resident_full_root_inquiries,
            };
            self.block_full_root_admission
                .observe(admission_route, root_elapsed_ns, inquiries);
        }
        result
    }

    fn block_inner(
        &mut self,
        limit: Option<f64>,
        progress: &mut Option<crate::accel::cdcl_host::ExactBlockProgressCapture>,
        full_root_fpga_enabled: bool,
        leaf_fpga_enabled: bool,
        resident_full_root_inquiries: &mut u64,
    ) -> BlockResult {
        if leaf_fpga_enabled && crate::accel::cdcl_host::block_batch_enabled() {
            crate::inductor::ThreadCpuTimer::enable();
        }
        let mut noc = 0;
        let mut block_batch = BlockBatchCache::default();
        let mut block_wave = VecDeque::new();
        let mut full_root_compacted_retry = false;
        let block_wave_enabled = leaf_fpga_enabled
            && crate::accel::cdcl_host::block_batch_enabled()
            && BlockBatchCache::wavefront_enabled();
        loop {
            let mut semantic_ops = progress.as_ref().map(|_| Vec::new());
            let mut from_wave = false;
            let mut next = None;
            while let Some(candidate) = block_wave.pop_front() {
                // A result processed earlier in the wave may have queued a
                // fresher obligation for the same state. Prefer that record
                // (notably its predecessor chain), otherwise consume the
                // reservation removed from the global set at batch creation.
                let taken = self.obligations.take(&candidate);
                if let Some(taken) = taken {
                    note_exact_obligation_op(
                        &mut semantic_ops,
                        BLOCK_SEMANTIC_REMOVE_OBLIGATION,
                        &taken,
                    );
                    next = Some(taken);
                } else {
                    next = Some(candidate);
                }
                from_wave = true;
                break;
            }
            let mut resident_root_inquiries = None;
            let mut resident_root_attempted = false;
            let mut full_root_pop = None;
            let mut full_root_sync_event = None;
            if next.is_none() && full_root_fpga_enabled {
                // A StepBudget response is normally followed immediately by
                // another resident root command.  Without checking the global
                // stop conditions here, that fast path can loop indefinitely
                // without reaching the ordinary CPU-obligation checks below.
                // Each completed command is a failure-atomic scheduling
                // boundary, so stopping here leaves the resident queue and
                // every journaled SAT/UNSAT commit in a consistent state.
                if self.ctrl.is_terminated()
                    || self
                        .cfg
                        .time_limit
                        .is_some_and(|limit| self.statistic.time.time().as_secs() > limit)
                {
                    self.note_exact_block_step(progress, BLOCK_STEP_TIMEOUT, semantic_ops);
                    return BlockResult::OverallTimeLimitExceeded;
                }
                let full_root = self.solvers.first().map_or(
                    crate::accel::cdcl_host::ResidentBlockFullRoot::Disabled,
                    |solver| {
                        let next_var_by_current = solver.resident_block_next_var_map();
                        let (init, latches, inputs) = solver.resident_block_projection_metadata();
                        let mut query_template =
                            solver.incremental_inductive_query(&[], false, vec![]);
                        query_template.budget.conflicts =
                            crate::accel::cdcl_host::block_full_root_conflict_budget();
                        let resident_solvers = self
                            .solvers
                            .iter()
                            .map(|solver| &solver.dcs)
                            .collect::<Vec<_>>();
                        let full_root_steps = std::env::var("INDUCTOR_CDCL_BLOCK_FULL_ROOT_STEPS")
                            .ok()
                            .and_then(|value| value.parse::<usize>().ok())
                            .filter(|steps| *steps != 0)
                            .unwrap_or(64);
                        crate::accel::cdcl_host::run_resident_block_full_root(
                            self.level(),
                            full_root_steps,
                            &resident_solvers,
                            &next_var_by_current,
                            &init,
                            &latches,
                            &inputs,
                            &query_template,
                            full_root_compacted_retry,
                        )
                    },
                );
                if let crate::accel::cdcl_host::ResidentBlockFullRoot::Wave {
                    response,
                    source_keys,
                } = full_root
                {
                    if response.status != BlockFullRootStatus::CompactionRequired {
                        full_root_compacted_retry = false;
                    }
                    *resident_full_root_inquiries = resident_full_root_inquiries
                        .saturating_add(u64::from(response.cdcl_inquiries));
                    let (proved, resident_semantic_ops) = self
                        .replay_resident_full_root(&response, &source_keys)
                        .unwrap_or_else(|error| panic!("resident full-root replay: {error}"));
                    let resident_solvers = self
                        .solvers
                        .iter()
                        .map(|solver| &solver.dcs)
                        .collect::<Vec<_>>();
                    crate::accel::cdcl_host::audit_resident_full_root_formula(&resident_solvers)
                        .unwrap_or_else(|error| {
                            panic!("resident full-root formula oracle: {error}")
                        });
                    let resident_ops = progress.as_ref().map(|_| resident_semantic_ops);
                    if proved {
                        self.note_exact_block_step(progress, BLOCK_STEP_PROVED, resident_ops);
                        return BlockResult::Proved;
                    }
                    match response.status {
                        BlockFullRootStatus::Drained => {
                            let event = if response.unsat_commits != 0 {
                                BLOCK_STEP_GENERALIZED
                            } else {
                                BLOCK_STEP_SUCCESS
                            };
                            self.note_exact_block_step(progress, event, resident_ops);
                            break;
                        }
                        BlockFullRootStatus::StepBudget => {
                            let event = if response.unsat_commits != 0 {
                                BLOCK_STEP_GENERALIZED
                            } else {
                                BLOCK_STEP_PREDECESSOR
                            };
                            self.note_exact_block_step(progress, event, resident_ops);
                            continue;
                        }
                        BlockFullRootStatus::CompactionRequired => {
                            full_root_compacted_retry = true;
                            let event = if response.unsat_commits != 0 {
                                BLOCK_STEP_GENERALIZED
                            } else {
                                BLOCK_STEP_PREDECESSOR
                            };
                            // The device restored the uncommitted head. Apply
                            // its conclusive prefix, let the simulation mirror
                            // rebuild the live arenas, then retry without a CPU
                            // solve or ownership transfer.
                            self.note_exact_block_step(progress, event, resident_ops);
                            continue;
                        }
                        BlockFullRootStatus::Proved => {
                            panic!("resident full-root proved status disagrees with CPU add_lemma")
                        }
                        BlockFullRootStatus::CpuResult
                        | BlockFullRootStatus::CpuHandoff
                        | BlockFullRootStatus::Fallback => {
                            let event = if response.unsat_commits != 0 {
                                BLOCK_STEP_GENERALIZED
                            } else {
                                BLOCK_STEP_PREDECESSOR
                            };
                            // The device has already POPed the handoff. Delay
                            // the image oracle until the CPU removes the same
                            // proof object; checking between those two halves
                            // reports a false one-obligation mismatch.
                            semantic_ops = resident_ops;
                            full_root_sync_event = Some(event);
                            let handoff = response
                                .handoff
                                .expect("resident full-root status requires handoff");
                            full_root_pop =
                                Some(crate::accel::cdcl_host::ResidentBlockPop::Selected {
                                    user_tag: handoff.user_tag(),
                                });
                        }
                        BlockFullRootStatus::Error => {
                            panic!("resident full-root returned an internal error")
                        }
                    }
                }
            }
            let full_root_handoff = full_root_pop.is_some();
            let resident_pop = if let Some(pop) = full_root_pop {
                pop
            } else if next.is_none() && leaf_fpga_enabled {
                let root = self.solvers.first().map_or(
                    crate::accel::cdcl_host::ResidentBlockRoot::Disabled,
                    |solver| {
                        let next_var_by_current = solver.resident_block_next_var_map();
                        let query_template = solver.incremental_inductive_query(&[], false, vec![]);
                        let resident_solvers = self
                            .solvers
                            .iter()
                            .map(|solver| &solver.dcs)
                            .collect::<Vec<_>>();
                        crate::accel::cdcl_host::run_resident_block_root(
                            self.level(),
                            BlockBatchCache::window().min(8),
                            &resident_solvers,
                            &next_var_by_current,
                            &query_template,
                        )
                    },
                );
                match root {
                    crate::accel::cdcl_host::ResidentBlockRoot::Wave { response, keys } => {
                        match response.status {
                            BlockRootExecutionStatus::Ok => {
                                resident_root_attempted = true;
                                let current_tag = response.work[0].user_tag();
                                if response.work.len() == keys.len() {
                                    let mut prepared = Vec::with_capacity(keys.len());
                                    let mut queries = Vec::with_capacity(keys.len());
                                    for (work, key) in response.work.iter().zip(&keys) {
                                        let Some(candidate) =
                                            self.obligations.clone_resident_key(key, self.level())
                                        else {
                                            prepared.clear();
                                            queries.clear();
                                            break;
                                        };
                                        if candidate.frame != work.frame as usize
                                            || candidate.depth != work.depth as usize
                                            || u32::from(candidate.removed) != work.removed
                                            || candidate.frame == 0
                                        {
                                            prepared.clear();
                                            queries.clear();
                                            break;
                                        }
                                        let query = self.solvers[candidate.frame - 1]
                                            .incremental_inductive_query(
                                                &candidate.state,
                                                false,
                                                vec![],
                                            );
                                        let revision = self.solvers[candidate.frame - 1]
                                            .dcs
                                            .incremental_context_revision();
                                        queries.push(query.clone());
                                        prepared.push((candidate, query, revision));
                                    }
                                    if prepared.len() == keys.len()
                                        && let Ok(results) =
                                            decode_batch_results(&queries, &response.batch)
                                        && results.len() == prepared.len()
                                    {
                                        resident_root_inquiries = Some(
                                            prepared
                                                .into_iter()
                                                .zip(results)
                                                .map(|((candidate, query, revision), result)| {
                                                    CachedBlockInquiry {
                                                        frame: candidate.frame,
                                                        state: candidate.state.clone(),
                                                        query,
                                                        context_revision: revision,
                                                        result,
                                                        materialized_sat_committed: false,
                                                        trusted_cpu: false,
                                                        hardware_selected: true,
                                                        cached_at: 0,
                                                        cache_age: 0,
                                                    }
                                                })
                                                .collect(),
                                        );
                                    }
                                }
                                crate::accel::cdcl_host::ResidentBlockPop::Selected {
                                    user_tag: current_tag,
                                }
                            }
                            BlockRootExecutionStatus::CpuHandoff => {
                                crate::accel::cdcl_host::ResidentBlockPop::Selected {
                                    user_tag: response.work[0].user_tag(),
                                }
                            }
                            BlockRootExecutionStatus::Empty => {
                                crate::accel::cdcl_host::ResidentBlockPop::Empty
                            }
                            _ => {
                                crate::accel::cdcl_host::pop_resident_block_obligation(self.level())
                            }
                        }
                    }
                    crate::accel::cdcl_host::ResidentBlockRoot::Disabled => {
                        crate::accel::cdcl_host::pop_resident_block_obligation(self.level())
                    }
                }
            } else {
                crate::accel::cdcl_host::ResidentBlockPop::Disabled
            };
            let mut po = if let Some(po) = next {
                po
            } else if let crate::accel::cdcl_host::ResidentBlockPop::Selected { user_tag } =
                &resident_pop
            {
                let po = self
                    .obligations
                    .take_resident_tag(*user_tag, self.level())
                    .unwrap_or_else(|| {
                        panic!("resident queue selected unknown proof-chain tag {user_tag}")
                    });
                if !full_root_handoff {
                    note_exact_obligation_pop(&mut semantic_ops, self.level(), &po);
                }
                po
            } else if matches!(
                resident_pop,
                crate::accel::cdcl_host::ResidentBlockPop::Empty
            ) {
                self.note_exact_block_step(progress, BLOCK_STEP_SUCCESS, semantic_ops);
                break;
            } else if let Some(po) = self.obligations.pop(self.level()) {
                note_exact_obligation_pop(&mut semantic_ops, self.level(), &po);
                po
            } else {
                self.note_exact_block_step(progress, BLOCK_STEP_SUCCESS, semantic_ops);
                break;
            };
            if let Some(event) = full_root_sync_event {
                self.note_exact_block_step(progress, event, semantic_ops.take());
                semantic_ops = progress.as_ref().map(|_| Vec::new());
            }
            if from_wave {
                crate::accel::cdcl_host::note_active_block_wave_taken();
            }
            block_batch.advance_step();
            block_batch.harvest_ready();
            if let Some(inquiries) = resident_root_inquiries.take() {
                block_batch.insert(inquiries);
            }
            // Remove a previously speculated answer even when this obligation
            // is discarded by one of the cheap guards below. Otherwise stale
            // entries would prevent the cache from naturally draining.
            let mut cached_block = block_batch.take(&po, true);
            let mut materialized_current_sat = false;
            self.render_progress();
            if po.removed {
                self.note_exact_block_step(progress, BLOCK_STEP_DISCARD_REMOVED, semantic_ops);
                continue;
            }
            if let Some(limit) = limit
                && noc as f64 > limit
            {
                self.restore_block_wave(&mut block_wave, &mut semantic_ops);
                self.note_exact_block_step(progress, BLOCK_STEP_LIMIT, semantic_ops);
                return BlockResult::BlockLimitExceeded;
            }
            if self.ctrl.is_terminated() {
                self.restore_block_wave(&mut block_wave, &mut semantic_ops);
                self.note_exact_block_step(progress, BLOCK_STEP_TIMEOUT, semantic_ops);
                return BlockResult::OverallTimeLimitExceeded;
            }
            if let Some(limit) = self.cfg.time_limit
                && self.statistic.time.time().as_secs() > limit
            {
                self.restore_block_wave(&mut block_wave, &mut semantic_ops);
                self.note_exact_block_step(progress, BLOCK_STEP_TIMEOUT, semantic_ops);
                return BlockResult::OverallTimeLimitExceeded;
            }
            if self.tsctx.cube_subsume_init(&po.state) {
                if self.cfg.abs_cst || self.cfg.abs_trans {
                    self.add_obligation(po.clone());
                    note_exact_obligation_op(
                        &mut semantic_ops,
                        BLOCK_SEMANTIC_INSERT_OBLIGATION,
                        &po,
                    );
                    if self.check_cex_by_bmc(po.depth) {
                        self.note_exact_block_step(progress, BLOCK_STEP_FAILURE, semantic_ops);
                        return BlockResult::Failure(po.depth);
                    }
                    self.obligations.clear();
                    note_exact_clear_obligations(&mut semantic_ops);
                    self.frame.clear_po();
                    self.note_exact_block_step(progress, BLOCK_STEP_SUBSUME_CLEAR, semantic_ops);
                    continue;
                } else if po.frame > 0 {
                    let lemma = po.state.as_litvec();
                    debug_assert!(!self.solvers[0].solve(lemma));
                } else {
                    self.add_obligation(po.clone());
                    note_exact_obligation_op(
                        &mut semantic_ops,
                        BLOCK_SEMANTIC_INSERT_OBLIGATION,
                        &po,
                    );
                    self.note_exact_block_step(progress, BLOCK_STEP_FAILURE, semantic_ops);
                    return BlockResult::Failure(po.depth);
                }
            }
            if let Some((bf, _)) = self.frame.trivial_contained(Some(po.frame), &po.state) {
                if let Some(bf) = bf {
                    po.push_to(bf + 1);
                    let inserted = if BlockBatchCache::materialize_sat_wave_enabled() {
                        self.add_obligation_if_new(po.clone())
                    } else {
                        self.add_obligation(po.clone());
                        true
                    };
                    if inserted {
                        note_exact_obligation_op(
                            &mut semantic_ops,
                            BLOCK_SEMANTIC_INSERT_OBLIGATION,
                            &po,
                        );
                    }
                }
                self.note_exact_block_step(progress, BLOCK_STEP_TRIVIAL_REQUEUE, semantic_ops);
                continue;
            }
            po.bump_act();
            if self.cfg.drop_po && po.act > 20.0 {
                self.note_exact_block_step(progress, BLOCK_STEP_DROP_ACTIVITY, semantic_ops);
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
            let batch_economics = crate::accel::cdcl_host::block_batch_economics_enabled();
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
            let block_cost_eligible = po.frame > 0 && (direct_cost_eligible || batch_plan_eligible);
            if po.frame > 0
                && !block_cost_eligible
                && leaf_fpga_enabled
                && crate::accel::cdcl_host::block_batch_enabled()
            {
                crate::accel::cdcl_host::note_active_block_cost_rejected();
            }
            if cached_block.is_none()
                && !resident_root_attempted
                && block_batch.can_launch()
                && po.frame > 0
                && leaf_fpga_enabled
                && crate::accel::cdcl_host::block_batch_enabled()
                && block_cost_eligible
                && self.solvers[po.frame - 1].dcs.num_var() >= BlockBatchCache::min_context_vars()
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
                        || (BlockBatchCache::async_enabled() || BlockBatchCache::reuse_enabled())
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
                    direct_route_profitable = self.block_accel_policy.note_calibration(sample_ns);
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
                        decisions[index] = crate::accel::cdcl_host::active_preflight_classify(
                            &mut self.solvers[*frame - 1].dcs,
                            query,
                        );
                        crate::accel::cdcl_host::note_active_block_preflight(&decisions[index]);
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
                    let route_selected =
                        if selected >= crate::accel::cdcl_host::active_min_batch_size() {
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
                            materialized_sat_committed: false,
                            trusted_cpu,
                            hardware_selected,
                            cached_at: 0,
                            cache_age: 0,
                        },
                    )
                    .collect();
                let asynchronous = BlockBatchCache::async_enabled()
                    && crate::accel::cdcl_host::active_enabled()
                    && hardware_indices.len() >= crate::accel::cdcl_host::active_min_batch_size();
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
                    let prepare_ns =
                        prepare_start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
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
                    let service_before = crate::accel::cdcl_host::active_batch_service_snapshot();
                    let hardware_results =
                        crate::accel::cdcl_host::solve_active_batch(hardware_requests);
                    let hardware_conclusive = hardware_results
                        .iter()
                        .filter(|result| {
                            matches!(
                                result,
                                IncrementalResult::Sat { .. } | IncrementalResult::Unsat { .. }
                            )
                        })
                        .count() as u64;
                    let service_after = crate::accel::cdcl_host::active_batch_service_snapshot();
                    self.block_accel_policy.note_hardware_batch(
                        service_after.1.saturating_sub(service_before.1),
                        service_after.0.saturating_sub(service_before.0),
                        service_after.2.saturating_sub(service_before.2),
                        hardware_conclusive,
                    );
                    for (index, result) in hardware_indices.iter().copied().zip(hardware_results) {
                        inquiries[index].result = result;
                    }
                    if BlockBatchCache::materialize_sat_wave_enabled()
                        && crate::accel::cdcl_host::active_skip_cpu_check()
                        && !BlockBatchCache::async_enabled()
                    {
                        // Commit every SAT witness before any result from this
                        // batch can strengthen a frame. This is the ordinary
                        // SAT transition (predecessor plus deferred current
                        // obligation), applied transactionally to the whole
                        // immutable-revision batch.
                        for index in hardware_indices.iter().copied() {
                            let materialization = match &inquiries[index].result {
                                IncrementalResult::Sat { model } => Some((
                                    inquiries[index].frame,
                                    inquiries[index].query.clone(),
                                    model.clone(),
                                )),
                                _ => None,
                            };
                            let Some((frame, query, model)) = materialization else {
                                continue;
                            };
                            if !BlockBatchCache::materialize_sat_context_eligible(
                                self.solvers[frame - 1].dcs.num_var(),
                                BlockBatchCache::materialize_sat_max_context_vars(),
                            ) {
                                continue;
                            }
                            let start = Instant::now();
                            let shape_ok = self.solvers[frame - 1]
                                .trusted_incremental_sat_model_shape(&query, &model);
                            let pred = shape_ok
                                .then(|| self.pred_from_incremental_model(&query, &model))
                                .flatten();
                            let candidate = if index == 0 {
                                Some(po.clone())
                            } else {
                                wave_candidates[index].clone()
                            };
                            let committed = pred.is_some() && candidate.is_some();
                            if let (Some((model, inputs)), Some(candidate)) = (pred, candidate) {
                                let candidate = if index == 0 {
                                    Some(candidate)
                                } else {
                                    self.obligations.take(&candidate).inspect(|candidate| {
                                        note_exact_obligation_op(
                                            &mut semantic_ops,
                                            BLOCK_SEMANTIC_REMOVE_OBLIGATION,
                                            candidate,
                                        );
                                    })
                                };
                                if let Some(candidate) = candidate {
                                    let pred = ProofObligation::new(
                                        candidate.frame - 1,
                                        LitOrdVec::new(model),
                                        inputs,
                                        candidate.depth + 1,
                                        Some(candidate.clone()),
                                    );
                                    if self.add_obligation_if_new(pred.clone()) {
                                        note_exact_obligation_op(
                                            &mut semantic_ops,
                                            BLOCK_SEMANTIC_INSERT_OBLIGATION,
                                            &pred,
                                        );
                                    }
                                    if self.add_obligation_if_new(candidate.clone()) {
                                        note_exact_obligation_op(
                                            &mut semantic_ops,
                                            BLOCK_SEMANTIC_INSERT_OBLIGATION,
                                            &candidate,
                                        );
                                    }
                                    inquiries[index].materialized_sat_committed = true;
                                    wave_candidates[index] = None;
                                    materialized_current_sat |= index == 0;
                                    crate::accel::cdcl_host::note_active_materialized_sat_used();
                                    crate::accel::cdcl_host::note_active_trusted_sat(
                                        true,
                                        start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                                    );
                                }
                            }
                            crate::accel::cdcl_host::note_active_materialized_sat_prepared(
                                committed && inquiries[index].materialized_sat_committed,
                                start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                            );
                        }
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
                                note_exact_obligation_op(
                                    &mut semantic_ops,
                                    BLOCK_SEMANTIC_REMOVE_OBLIGATION,
                                    &candidate,
                                );
                                block_wave.push_back(candidate);
                                reserved += 1;
                            }
                        }
                        crate::accel::cdcl_host::note_active_block_wave_reserved(reserved);
                    }
                    block_batch.insert(inquiries);
                    if materialized_current_sat {
                        self.note_exact_block_step(progress, BLOCK_STEP_PREDECESSOR, semantic_ops);
                        continue;
                    }
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
                        let stale_trusted =
                            crate::accel::cdcl_host::active_skip_cpu_check() && !direct_trust;
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
                                .validate_incremental_unsat_core(&po.state, &entry.query, &core);
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
                if self.generalize(po, mic_type, &mut semantic_ops) {
                    self.note_exact_block_step(progress, BLOCK_STEP_PROVED, semantic_ops);
                    return BlockResult::Proved;
                }
                self.note_exact_block_step(progress, BLOCK_STEP_GENERALIZED, semantic_ops);
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
                    if self.add_obligation_if_new(pred.clone()) {
                        note_exact_obligation_op(
                            &mut semantic_ops,
                            BLOCK_SEMANTIC_INSERT_OBLIGATION,
                            &pred,
                        );
                    }
                    if self.add_obligation_if_new(po.clone()) {
                        note_exact_obligation_op(
                            &mut semantic_ops,
                            BLOCK_SEMANTIC_INSERT_OBLIGATION,
                            &po,
                        );
                    }
                } else {
                    self.add_obligation(pred.clone());
                    self.add_obligation(po.clone());
                    note_exact_obligation_op(
                        &mut semantic_ops,
                        BLOCK_SEMANTIC_INSERT_OBLIGATION,
                        &pred,
                    );
                    note_exact_obligation_op(
                        &mut semantic_ops,
                        BLOCK_SEMANTIC_INSERT_OBLIGATION,
                        &po,
                    );
                }
                self.note_exact_block_step(progress, BLOCK_STEP_PREDECESSOR, semantic_ops);
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
