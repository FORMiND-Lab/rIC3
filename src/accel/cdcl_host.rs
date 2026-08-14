//! XRT-backed implementation of the incremental CDCL semantic boundary.
//!
//! The C++ bridge owns one persistent kernel context and reusable DMA buffers.
//! Transport or device failures become `Unknown(BackendError)` through the
//! `IncrementalCdcl` implementation; they are never interpreted as SAT/UNSAT.

use super::cdcl::{BatchHeader, RESPONSE_HEADER_WORDS, Status, UnknownReason};
use crate::gipsat::{
    BatchDecodeError, DagCnfSolver, IncrementalCdcl, IncrementalQuery, IncrementalResult,
    pack_batch, solve_on_cpu_after_hardware_unknown,
};
#[cfg(has_cdcl_accel)]
use crate::gipsat::decode_batch_results;
use logicrs::LitVec;
#[cfg(has_cdcl_accel)]
use std::ffi::CString;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(has_cdcl_accel)]
unsafe extern "C" {
    fn ind_cdcl_open(path: *const std::os::raw::c_char) -> i32;
    fn ind_cdcl_load_context(request: *const u32, request_words: u32) -> i32;
    fn ind_cdcl_add_frame_clauses(request: *const u32, request_words: u32) -> i32;
    fn ind_cdcl_solve_batch(
        request: *const u32,
        request_words: u32,
        response: *mut u32,
        response_capacity_words: u32,
        out_response_words: *mut u32,
    ) -> i32;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentClause {
    pub lo: u32,
    pub hi: u32,
    pub literals: LitVec,
}

impl ResidentClause {
    pub fn new(lo: u32, hi: u32, literals: impl Into<LitVec>) -> Self {
        Self {
            lo,
            hi,
            literals: literals.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HardwareError {
    Unavailable,
    InvalidPath,
    InvalidContext,
    Capacity,
    Open(i32),
    Command(i32),
    Decode(BatchDecodeError),
}

impl std::fmt::Display for HardwareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "incremental CDCL hardware error: {self:?}")
    }
}

impl std::error::Error for HardwareError {}

/// Encode frame-ranged clauses for LOAD_CONTEXT or ADD_FRAME_CLAUSES.
fn pack_clauses(
    prefix: &[u32],
    n_var: u32,
    clauses: &[ResidentClause],
) -> Result<Vec<u32>, HardwareError> {
    if n_var == 0 || clauses.len() > u32::MAX as usize {
        return Err(HardwareError::InvalidContext);
    }
    let mut capacity = prefix.len();
    for clause in clauses {
        if clause.lo > clause.hi || clause.literals.is_empty()
            || clause.literals.len() > u32::MAX as usize
            || clause
                .literals
                .iter()
                .any(|lit| u32::from(*lit) >> 1 >= n_var)
        {
            return Err(HardwareError::InvalidContext);
        }
        capacity = capacity
            .checked_add(3 + clause.literals.len())
            .ok_or(HardwareError::Capacity)?;
    }
    let mut words = Vec::with_capacity(capacity);
    words.extend_from_slice(prefix);
    for clause in clauses {
        words.push(clause.lo);
        words.push(clause.hi);
        words.push(clause.literals.len() as u32);
        words.extend(clause.literals.iter().map(|lit| u32::from(*lit)));
    }
    if words.len() > u32::MAX as usize {
        return Err(HardwareError::Capacity);
    }
    Ok(words)
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct HardwareWork {
    status: u32,
    reason: u32,
    decisions: u64,
    conflicts: u64,
    propagations: u64,
    learnt_clauses: u64,
}

fn decode_batch_work_records(words: &[u32], n_queries: usize) -> Option<Vec<HardwareWork>> {
    let prefix = words.get(..4)?;
    if usize::try_from(prefix[1]).ok()? != n_queries {
        return None;
    }
    let mut records = Vec::with_capacity(n_queries);
    let mut offset = 4usize;
    for _ in 0..n_queries {
        let header = words.get(offset..offset.checked_add(RESPONSE_HEADER_WORDS)?)?;
        records.push(HardwareWork {
            status: header[0],
            reason: header[1],
            decisions: u64::from(header[4]),
            conflicts: u64::from(header[5]),
            propagations: u64::from(header[6]),
            learnt_clauses: u64::from(header[7]),
        });
        let payload_words = usize::try_from(header[2]).ok()?
            .checked_add(usize::try_from(header[3]).ok()?)?;
        offset = offset
            .checked_add(RESPONSE_HEADER_WORDS)?
            .checked_add(payload_words)?;
        if offset > words.len() {
            return None;
        }
    }
    (offset == words.len()).then_some(records)
}

fn sum_hardware_work(records: &[HardwareWork]) -> HardwareWork {
    records
        .iter()
        .copied()
        .fold(HardwareWork::default(), |mut total, work| {
            total.decisions = total.decisions.saturating_add(work.decisions);
            total.conflicts = total.conflicts.saturating_add(work.conflicts);
            total.propagations = total.propagations.saturating_add(work.propagations);
            total.learnt_clauses = total.learnt_clauses.saturating_add(work.learnt_clauses);
            total
        })
}

pub struct HardwareCdcl {
    n_var: u32,
    last_batch_work: HardwareWork,
    last_batch_records: Vec<HardwareWork>,
}

impl HardwareCdcl {
    pub fn compiled() -> bool {
        cfg!(has_cdcl_accel)
    }

    /// Open the xclbin named explicitly by the caller.
    pub fn open(path: &str) -> Result<Self, HardwareError> {
        #[cfg(has_cdcl_accel)]
        {
            let path = CString::new(path).map_err(|_| HardwareError::InvalidPath)?;
            let rc = unsafe { ind_cdcl_open(path.as_ptr()) };
            if rc != 0 {
                return Err(HardwareError::Open(rc));
            }
            Ok(Self {
                n_var: 0,
                last_batch_work: HardwareWork::default(),
                last_batch_records: Vec::new(),
            })
        }
        #[cfg(not(has_cdcl_accel))]
        {
            let _ = path;
            Err(HardwareError::Unavailable)
        }
    }

    /// `INDUCTOR_CDCL_ACCEL` is separate from the older propagation-only
    /// xclbin so an incompatible bitstream cannot be selected accidentally.
    pub fn open_from_env() -> Result<Self, HardwareError> {
        let path = std::env::var("INDUCTOR_CDCL_ACCEL")
            .map_err(|_| HardwareError::Unavailable)?;
        Self::open(&path)
    }

    pub fn load_context(
        &mut self,
        n_var: u32,
        clauses: &[ResidentClause],
    ) -> Result<(), HardwareError> {
        let n_clause = u32::try_from(clauses.len()).map_err(|_| HardwareError::Capacity)?;
        let words = pack_clauses(&[n_var, n_clause], n_var, clauses)?;
        profile_resident_context(n_var, clauses);
        #[cfg(has_cdcl_accel)]
        {
            let rc = unsafe { ind_cdcl_load_context(words.as_ptr(), words.len() as u32) };
            if rc != 0 {
                return Err(HardwareError::Command(rc));
            }
            self.n_var = n_var;
            Ok(())
        }
        #[cfg(not(has_cdcl_accel))]
        {
            let _ = words;
            Err(HardwareError::Unavailable)
        }
    }

    /// Load the transition CNF and current frame lemmas from a real GipSAT
    /// instance. Learnts remain backend-local and temporary constraints remain
    /// query-local, matching `DagCnfSolver::incremental_resident_snapshot`.
    pub fn load_solver_context(
        &mut self,
        solver: &DagCnfSolver,
    ) -> Result<(), HardwareError> {
        let (n_var, frame, snapshot) = solver.incremental_resident_snapshot();
        let clauses: Vec<_> = snapshot
            .into_iter()
            .map(|literals| ResidentClause::new(frame, frame, literals))
            .collect();
        self.load_context(n_var, &clauses)
    }

    pub fn add_frame_clauses(
        &mut self,
        clauses: &[ResidentClause],
    ) -> Result<(), HardwareError> {
        if self.n_var == 0 {
            return Err(HardwareError::InvalidContext);
        }
        let n_clause = u32::try_from(clauses.len()).map_err(|_| HardwareError::Capacity)?;
        let words = pack_clauses(&[n_clause], self.n_var, clauses)?;
        #[cfg(has_cdcl_accel)]
        {
            let rc = unsafe {
                ind_cdcl_add_frame_clauses(words.as_ptr(), words.len() as u32)
            };
            if rc != 0 {
                return Err(HardwareError::Command(rc));
            }
            Ok(())
        }
        #[cfg(not(has_cdcl_accel))]
        {
            let _ = words;
            Err(HardwareError::Unavailable)
        }
    }

    /// Execute many short inquiries in one XRT command. Result capacity is
    /// exact for the ABI: a SAT model contains at most the domain and an UNSAT
    /// core at most the assumptions.
    pub fn solve_batch(
        &mut self,
        queries: &[IncrementalQuery],
    ) -> Result<Vec<IncrementalResult>, HardwareError> {
        self.last_batch_work = HardwareWork::default();
        self.last_batch_records.clear();
        if self.n_var == 0 {
            return Err(HardwareError::InvalidContext);
        }
        let result_words = queries.iter().try_fold(0usize, |total, query| {
            total.checked_add(
                RESPONSE_HEADER_WORDS + query.domain.len().max(query.assumptions.len()),
            )
        }).ok_or(HardwareError::Capacity)?;
        let result_words_u32 =
            u32::try_from(result_words).map_err(|_| HardwareError::Capacity)?;
        let (batch, payload) = pack_batch(queries, result_words_u32);
        let mut request = Vec::with_capacity(4 + payload.len());
        let BatchHeader {
            version,
            n_queries,
            n_request_words,
            result_capacity_words,
        } = batch;
        request.extend([version, n_queries, n_request_words, result_capacity_words]);
        request.extend(payload);

        let response_capacity = result_words.checked_add(4).ok_or(HardwareError::Capacity)?;
        let response_capacity_u32 =
            u32::try_from(response_capacity).map_err(|_| HardwareError::Capacity)?;
        #[cfg(has_cdcl_accel)]
        let mut response = vec![0u32; response_capacity];
        #[cfg(not(has_cdcl_accel))]
        let response = vec![0u32; response_capacity];
        #[cfg(has_cdcl_accel)]
        {
            let mut out_words = 0u32;
            let rc = unsafe {
                ind_cdcl_solve_batch(
                    request.as_ptr(),
                    request.len() as u32,
                    response.as_mut_ptr(),
                    response_capacity_u32,
                    &mut out_words,
                )
            };
            if rc != 0 {
                return Err(HardwareError::Command(rc));
            }
            let out_words = usize::try_from(out_words).map_err(|_| HardwareError::Capacity)?;
            if out_words > response.len() {
                return Err(HardwareError::Capacity);
            }
            response.truncate(out_words);
            let results = decode_batch_results(queries, &response)
                .map_err(HardwareError::Decode)?;
            self.last_batch_records = decode_batch_work_records(&response, queries.len())
                .ok_or(HardwareError::Decode(BatchDecodeError::InvalidResultShape))?;
            self.last_batch_work = sum_hardware_work(&self.last_batch_records);
            Ok(results)
        }
        #[cfg(not(has_cdcl_accel))]
        {
            let _ = (request, response_capacity_u32, response);
            Err(HardwareError::Unavailable)
        }
    }

    /// Batch first, then retry only inconclusive inquiries in GipSAT. All
    /// queries in this call must belong to the supplied CPU solver/frame.
    pub fn solve_batch_with_cpu_fallback(
        &mut self,
        cpu: &mut DagCnfSolver,
        queries: &[IncrementalQuery],
    ) -> Vec<IncrementalResult> {
        match self.solve_batch(queries) {
            Ok(results) => results
                .into_iter()
                .zip(queries)
                .map(|(result, query)| {
                    if matches!(result, IncrementalResult::Unknown(_)) {
                        solve_on_cpu_after_hardware_unknown(cpu, query)
                    } else {
                        result
                    }
                })
                .collect(),
            Err(_) => queries
                .iter()
                .map(|query| solve_on_cpu_after_hardware_unknown(cpu, query))
                .collect(),
        }
    }
}

impl IncrementalCdcl for HardwareCdcl {
    fn solve_incremental(&mut self, query: &IncrementalQuery) -> IncrementalResult {
        match self.solve_batch(std::slice::from_ref(query)) {
            Ok(mut result) => result.pop().unwrap_or(IncrementalResult::Unknown(
                super::cdcl::UnknownReason::BackendError,
            )),
            Err(_) => IncrementalResult::Unknown(super::cdcl::UnknownReason::BackendError),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ShadowContext {
    n_var: u32,
    clauses: Vec<ResidentClause>,
}

struct ShadowBatch {
    context: ShadowContext,
    pending: Vec<(IncrementalQuery, Option<bool>)>,
}

struct ShadowState {
    hardware: Option<HardwareCdcl>,
    loaded_context: Option<ShadowContext>,
    batches: Vec<ShadowBatch>,
}

struct ActiveState {
    hardware: Option<HardwareCdcl>,
    loaded_context: Option<ShadowContext>,
}

static SHADOW_STATE: std::sync::OnceLock<std::sync::Mutex<ShadowState>> =
    std::sync::OnceLock::new();
static ACTIVE_STATE: std::sync::OnceLock<std::sync::Mutex<ActiveState>> =
    std::sync::OnceLock::new();
static SHADOW_OFFERED: AtomicU64 = AtomicU64::new(0);
static SHADOW_BATCHES: AtomicU64 = AtomicU64::new(0);
static SHADOW_CONTEXT_LOADS: AtomicU64 = AtomicU64::new(0);
static SHADOW_AGREE: AtomicU64 = AtomicU64::new(0);
static SHADOW_MISMATCH: AtomicU64 = AtomicU64::new(0);
static SHADOW_HW_SAT_CPU_UNSAT: AtomicU64 = AtomicU64::new(0);
static SHADOW_HW_UNSAT_CPU_SAT: AtomicU64 = AtomicU64::new(0);
static SHADOW_UNKNOWN: AtomicU64 = AtomicU64::new(0);
static SHADOW_ERROR: AtomicU64 = AtomicU64::new(0);
static SHADOW_QUERY_LEMMAS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_OFFERED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_CANDIDATES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_SKIPPED_SMALL_BATCH: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PASSES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_SKIPPED_PASSES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_OFFERED_PASSES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MAX_READY: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BATCHES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_CONTEXT_LOADS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_HW_SAT: AtomicU64 = AtomicU64::new(0);
static ACTIVE_HW_UNSAT: AtomicU64 = AtomicU64::new(0);
static ACTIVE_UNKNOWN: AtomicU64 = AtomicU64::new(0);
static ACTIVE_ERROR: AtomicU64 = AtomicU64::new(0);
static ACTIVE_SAT_USED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_SAT_REJECTED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_CPU_FALLBACK: AtomicU64 = AtomicU64::new(0);
static ACTIVE_INIT_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_CONTEXT_LOAD_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BATCH_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_VALIDATE_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_HW_DECISIONS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_HW_CONFLICTS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_HW_PROPAGATIONS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_HW_LEARNTS: AtomicU64 = AtomicU64::new(0);
static PROFILE_QUERIES: AtomicU64 = AtomicU64::new(0);
static PROFILE_ZERO_DECISIONS: AtomicU64 = AtomicU64::new(0);
static PROFILE_CONFLICT_0: AtomicU64 = AtomicU64::new(0);
static PROFILE_CONFLICT_1: AtomicU64 = AtomicU64::new(0);
static PROFILE_CONFLICT_2_3: AtomicU64 = AtomicU64::new(0);
static PROFILE_CONFLICT_4_15: AtomicU64 = AtomicU64::new(0);
static PROFILE_CONFLICT_16_PLUS: AtomicU64 = AtomicU64::new(0);
static PROFILE_UNKNOWN: AtomicU64 = AtomicU64::new(0);
static PROFILE_UNKNOWN_CONFLICT: AtomicU64 = AtomicU64::new(0);
static PROFILE_UNKNOWN_CAPACITY: AtomicU64 = AtomicU64::new(0);
static PROFILE_ASSUMPTIONS: AtomicU64 = AtomicU64::new(0);
static PROFILE_CONSTRAINT_LITS: AtomicU64 = AtomicU64::new(0);
static PROFILE_DOMAIN: AtomicU64 = AtomicU64::new(0);
static PROFILE_MAX_ASSUMPTIONS: AtomicU64 = AtomicU64::new(0);
static PROFILE_MAX_CONSTRAINT_LITS: AtomicU64 = AtomicU64::new(0);
static PROFILE_MAX_DOMAIN: AtomicU64 = AtomicU64::new(0);
static PROFILE_CONTEXT_LOADS: AtomicU64 = AtomicU64::new(0);
static PROFILE_MAX_CONTEXT_VARS: AtomicU64 = AtomicU64::new(0);
static PROFILE_MAX_CONTEXT_CLAUSES: AtomicU64 = AtomicU64::new(0);
static PROFILE_MAX_CONTEXT_LITS: AtomicU64 = AtomicU64::new(0);

const DEFAULT_SHADOW_BATCH_SIZE: usize = 64;
const KERNEL_MAX_REQUEST_WORDS: usize = 1 << 15;
const DEFAULT_SHADOW_CONFLICT_BUDGET: u32 = 3;
const DEFAULT_ACTIVE_CONFLICT_BUDGET: u32 = 16;

fn configured_conflict_budget(mode_variable: &str, default: u32) -> u32 {
    std::env::var(mode_variable)
        .ok()
        .or_else(|| std::env::var("INDUCTOR_CDCL_CONFLICT_BUDGET").ok())
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(default)
}

/// Bound synchronous FPGA tail latency. UNKNOWN is retried by the ordinary
/// GipSAT path, so this is a scheduling policy rather than a proof limit.
pub fn shadow_conflict_budget() -> u32 {
    static BUDGET: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| {
        configured_conflict_budget(
            "INDUCTOR_CDCL_SHADOW_CONFLICT_BUDGET",
            DEFAULT_SHADOW_CONFLICT_BUDGET,
        )
    })
}

/// Active batches use the same short-query policy, with an independent
/// override for controlled A/B measurements.
pub fn active_conflict_budget() -> u32 {
    static BUDGET: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| {
        configured_conflict_budget(
            "INDUCTOR_CDCL_ACTIVE_CONFLICT_BUDGET",
            DEFAULT_ACTIVE_CONFLICT_BUDGET,
        )
    })
}

fn profile_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("INDUCTOR_CDCL_PROFILE").is_ok())
}

fn profile_every_batch() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("INDUCTOR_CDCL_PROFILE_EVERY_BATCH").is_ok())
}

fn profile_capacity_detail() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("INDUCTOR_CDCL_PROFILE_CAPACITY").is_ok())
}

fn print_profile() {
    let queries = PROFILE_QUERIES.load(Ordering::Relaxed);
    let divisor = queries.max(1) as f64;
    eprintln!(
        "inductor-cdcl: profile queries {}, zero-decision {}, conflicts [0/1/2-3/4-15/16+] [{}/{}/{}/{}/{}], unknown total/conflict/capacity {}/{}/{}, mean assumptions/constraint-lits/domain {:.2}/{:.2}/{:.2}, max {}/{}/{}, resident loads {}, max vars/clauses/lits {}/{}/{}, active batches/service-ms {}/{:.3}",
        queries,
        PROFILE_ZERO_DECISIONS.load(Ordering::Relaxed),
        PROFILE_CONFLICT_0.load(Ordering::Relaxed),
        PROFILE_CONFLICT_1.load(Ordering::Relaxed),
        PROFILE_CONFLICT_2_3.load(Ordering::Relaxed),
        PROFILE_CONFLICT_4_15.load(Ordering::Relaxed),
        PROFILE_CONFLICT_16_PLUS.load(Ordering::Relaxed),
        PROFILE_UNKNOWN.load(Ordering::Relaxed),
        PROFILE_UNKNOWN_CONFLICT.load(Ordering::Relaxed),
        PROFILE_UNKNOWN_CAPACITY.load(Ordering::Relaxed),
        PROFILE_ASSUMPTIONS.load(Ordering::Relaxed) as f64 / divisor,
        PROFILE_CONSTRAINT_LITS.load(Ordering::Relaxed) as f64 / divisor,
        PROFILE_DOMAIN.load(Ordering::Relaxed) as f64 / divisor,
        PROFILE_MAX_ASSUMPTIONS.load(Ordering::Relaxed),
        PROFILE_MAX_CONSTRAINT_LITS.load(Ordering::Relaxed),
        PROFILE_MAX_DOMAIN.load(Ordering::Relaxed),
        PROFILE_CONTEXT_LOADS.load(Ordering::Relaxed),
        PROFILE_MAX_CONTEXT_VARS.load(Ordering::Relaxed),
        PROFILE_MAX_CONTEXT_CLAUSES.load(Ordering::Relaxed),
        PROFILE_MAX_CONTEXT_LITS.load(Ordering::Relaxed),
        ACTIVE_BATCHES.load(Ordering::Relaxed),
        ACTIVE_BATCH_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
    );
}

/// Record the largest immutable transition context actually sent to the card.
/// The light lane must still hold this whole context, so conflict and query
/// profiling alone is insufficient for choosing its physical RAM capacity.
fn profile_resident_context(n_var: u32, clauses: &[ResidentClause]) {
    if !profile_enabled() {
        return;
    }
    let literals = clauses
        .iter()
        .map(|clause| clause.literals.len() as u64)
        .sum::<u64>();
    PROFILE_CONTEXT_LOADS.fetch_add(1, Ordering::Relaxed);
    PROFILE_MAX_CONTEXT_VARS.fetch_max(u64::from(n_var), Ordering::Relaxed);
    PROFILE_MAX_CONTEXT_CLAUSES.fetch_max(clauses.len() as u64, Ordering::Relaxed);
    PROFILE_MAX_CONTEXT_LITS.fetch_max(literals, Ordering::Relaxed);
}

/// Record the real per-query search work returned by the card. This is an
/// opt-in sizing probe for low-budget/lightweight lanes; it has no scheduling
/// or proof effect.
fn profile_hardware_batch(queries: &[IncrementalQuery], work: &[HardwareWork]) {
    if !profile_enabled() || queries.len() != work.len() {
        return;
    }
    for (batch_index, (query, work)) in queries.iter().zip(work).enumerate() {
        let assumptions = query.assumptions.len() as u64;
        let constraint_literals = query.constraints.iter()
            .map(|clause| clause.len() as u64)
            .sum::<u64>();
        let domain = query.domain.len() as u64;
        PROFILE_QUERIES.fetch_add(1, Ordering::Relaxed);
        PROFILE_ASSUMPTIONS.fetch_add(assumptions, Ordering::Relaxed);
        PROFILE_CONSTRAINT_LITS.fetch_add(constraint_literals, Ordering::Relaxed);
        PROFILE_DOMAIN.fetch_add(domain, Ordering::Relaxed);
        PROFILE_MAX_ASSUMPTIONS.fetch_max(assumptions, Ordering::Relaxed);
        PROFILE_MAX_CONSTRAINT_LITS.fetch_max(constraint_literals, Ordering::Relaxed);
        PROFILE_MAX_DOMAIN.fetch_max(domain, Ordering::Relaxed);
        if work.decisions == 0 {
            PROFILE_ZERO_DECISIONS.fetch_add(1, Ordering::Relaxed);
        }
        match work.conflicts {
            0 => &PROFILE_CONFLICT_0,
            1 => &PROFILE_CONFLICT_1,
            2..=3 => &PROFILE_CONFLICT_2_3,
            4..=15 => &PROFILE_CONFLICT_4_15,
            _ => &PROFILE_CONFLICT_16_PLUS,
        }.fetch_add(1, Ordering::Relaxed);
        if work.status == Status::Unknown as u32 {
            PROFILE_UNKNOWN.fetch_add(1, Ordering::Relaxed);
            match work.reason {
                reason if reason == UnknownReason::ConflictBudget as u32 => {
                    PROFILE_UNKNOWN_CONFLICT.fetch_add(1, Ordering::Relaxed);
                }
                reason if reason == UnknownReason::Capacity as u32 => {
                    PROFILE_UNKNOWN_CAPACITY.fetch_add(1, Ordering::Relaxed);
                    if profile_capacity_detail() {
                        eprintln!(
                            "inductor-cdcl: capacity batch-index {} frame {} assumptions {} constraints {} constraint-lits {} domain {} decisions {} conflicts {} propagations {} learnt-clauses {}",
                            batch_index,
                            query.frame,
                            assumptions,
                            query.constraints.len(),
                            constraint_literals,
                            domain,
                            work.decisions,
                            work.conflicts,
                            work.propagations,
                            work.learnt_clauses,
                        );
                    }
                }
                _ => {}
            }
        }
    }
    if profile_every_batch() {
        print_profile();
    }
}

fn shadow_batch_size() -> usize {
    static SIZE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SIZE.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_SHADOW_BATCH")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .map(|value| value.clamp(1, DEFAULT_SHADOW_BATCH_SIZE))
            .unwrap_or(DEFAULT_SHADOW_BATCH_SIZE)
    })
}

fn active_batch_size() -> usize {
    static SIZE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SIZE.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_ACTIVE_BATCH")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .map(|value| value.clamp(1, DEFAULT_SHADOW_BATCH_SIZE))
            .unwrap_or(DEFAULT_SHADOW_BATCH_SIZE)
    })
}

fn active_min_batch_size() -> usize {
    static SIZE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SIZE.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_ACTIVE_MIN_BATCH")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .map(|value| value.clamp(1, DEFAULT_SHADOW_BATCH_SIZE))
            // The measured batch-1 round trip is ~48 us while these GipSAT
            // push inquiries average only a few microseconds. Do not program
            // the card for a handful of queries by default.
            .unwrap_or(32)
    })
}

fn pair_scheduler_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_PAIR_SCHEDULER")
            .ok()
            .map(|value| !matches!(value.as_str(), "0" | "false" | "off"))
            // Static size is not a stable predictor once each physical lane
            // retains phase/activity history.  On the repaired loader the
            // identical 229-query multiplier campaign repeatedly took
            // ~54.66 ms sorted versus 47.8--48.1 ms in IC3 order.  Preserve
            // caller order by default; keep sorting as an explicit research
            // switch while a history-aware scheduler is developed.
            .unwrap_or(false)
    })
}

/// Estimate the work that one query contributes to a two-lane round. This is
/// deliberately cheap: assumptions create propagation events, temporary
/// literals must be inserted into the private clause arena, and domain
/// variables may participate in decision selection. The score is only used to
/// pair independent inquiries; it cannot change their formula or answer.
fn query_work_score(query: &IncrementalQuery) -> u64 {
    let constraint_literals = query
        .constraints
        .iter()
        .fold(0u64, |total, clause| {
            total.saturating_add(clause.len() as u64)
        });
    (query.assumptions.len() as u64)
        .saturating_mul(16)
        .saturating_add(constraint_literals.saturating_mul(4))
        .saturating_add(query.domain.len() as u64)
}

/// Two engines execute adjacent requests concurrently, so pairing the largest
/// estimates together minimizes the sum of pair maxima for a fixed set of
/// scores. The caller keeps an original index alongside each active query and
/// restores result order after the FPGA returns.
fn schedule_query_pairs<T>(
    pending: &mut [T],
    query_of: impl Fn(&T) -> &IncrementalQuery,
) {
    pending.sort_by_cached_key(|item| {
        std::cmp::Reverse(query_work_score(query_of(item)))
    });
}

fn query_request_words(query: &IncrementalQuery) -> Option<usize> {
    let constraints = query.constraints.iter().try_fold(0usize, |words, clause| {
        words.checked_add(1 + clause.len())
    })?;
    8usize
        .checked_add(query.assumptions.len())?
        .checked_add(constraints)?
        .checked_add(query.domain.len())
}

/// Keep the transition relation resident and encode this frame's permanent
/// lemmas as query-local clauses. That makes inquiries from different IC3
/// snapshots batch-compatible without weakening or strengthening any query.
/// If one expanded query would exceed the command buffer, retain the previous
/// exact-snapshot representation as a capacity-safe fallback.
fn prepare_batched_query(
    solver: &DagCnfSolver,
    mut query: IncrementalQuery,
    prefer_query_lemmas: bool,
) -> (ShadowContext, IncrementalQuery, bool) {
    let (n_var, frame, trans, lemmas) = solver.incremental_resident_partition();
    let n_existing_constraints = query.constraints.len();
    if prefer_query_lemmas {
        query.constraints.extend(lemmas.iter().cloned());
        let fits = query_request_words(&query)
            .and_then(|words| words.checked_add(4))
            .is_some_and(|words| words <= KERNEL_MAX_REQUEST_WORDS);
        if fits {
            let clauses = trans
                .into_iter()
                .map(|literals| ResidentClause::new(0, u32::MAX, literals))
                .collect();
            return (ShadowContext { n_var, clauses }, query, true);
        }
        query.constraints.truncate(n_existing_constraints);
    }

    let clauses = trans
        .into_iter()
        .chain(lemmas)
        .map(|literals| ResidentClause::new(frame, frame, literals))
        .collect();
    (ShadowContext { n_var, clauses }, query, false)
}

fn dump_mismatch_dimacs(context: &ShadowContext, query: &IncrementalQuery) {
    let Some(path) = std::env::var_os("INDUCTOR_CDCL_DUMP_MISMATCH") else {
        return;
    };
    let Ok(mut file) = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
    else {
        return;
    };
    use std::io::Write;
    let n_clause = context.clauses.len() + query.constraints.len() + query.assumptions.len();
    if writeln!(file, "p cnf {} {}", context.n_var, n_clause).is_err() {
        return;
    }
    let write_clause = |file: &mut std::fs::File, clause: &[logicrs::Lit]| {
        for lit in clause {
            let var = i64::from(u32::from(lit.var())) + 1;
            let dimacs = if lit.polarity() { var } else { -var };
            write!(file, "{dimacs} ")?;
        }
        writeln!(file, "0")
    };
    for clause in &context.clauses {
        if write_clause(&mut file, &clause.literals).is_err() {
            return;
        }
    }
    for clause in &query.constraints {
        if write_clause(&mut file, clause).is_err() {
            return;
        }
    }
    for assumption in &query.assumptions {
        if write_clause(&mut file, std::slice::from_ref(assumption)).is_err() {
            return;
        }
    }
}

/// `INDUCTOR_CDCL_SHADOW=<xclbin>` enables correctness-only batching after CPU
/// answers. It is mutually exclusive with the older propagation-only xclbin.
pub fn shadow_enabled() -> bool {
    std::env::var_os("INDUCTOR_CDCL_SHADOW").is_some()
        && std::env::var_os("INDUCTOR_ACCEL").is_none()
}

/// Enable the proof-safe active path. Only complete SAT models that pass an
/// exact CPU formula check may bypass GipSAT search; all other results fall
/// back. Active and shadow modes are separate so one process never opens the
/// singleton XRT bridge twice.
pub fn active_enabled() -> bool {
    std::env::var_os("INDUCTOR_CDCL_ACTIVE").is_some()
        && std::env::var_os("INDUCTOR_CDCL_SHADOW").is_none()
        && std::env::var_os("INDUCTOR_ACCEL").is_none()
}

fn shadow_state() -> &'static std::sync::Mutex<ShadowState> {
    SHADOW_STATE.get_or_init(|| {
        let hardware = std::env::var("INDUCTOR_CDCL_SHADOW")
            .ok()
            .and_then(|path| HardwareCdcl::open(&path).ok());
        std::sync::Mutex::new(ShadowState {
            hardware,
            loaded_context: None,
            batches: Vec::new(),
        })
    })
}

fn active_state() -> &'static std::sync::Mutex<ActiveState> {
    ACTIVE_STATE.get_or_init(|| {
        let start = std::time::Instant::now();
        let hardware = std::env::var("INDUCTOR_CDCL_ACTIVE")
            .ok()
            .and_then(|path| HardwareCdcl::open(&path).ok());
        ACTIVE_INIT_NS.store(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        std::sync::Mutex::new(ActiveState {
            hardware,
            loaded_context: None,
        })
    })
}

fn flush_batch_locked(state: &mut ShadowState, batch_index: usize) {
    let mut batch = state.batches.swap_remove(batch_index);
    if batch.pending.is_empty() {
        return;
    }
    if pair_scheduler_enabled() {
        schedule_query_pairs(&mut batch.pending, |(query, _)| query);
    }
    if state.loaded_context.as_ref() != Some(&batch.context) {
        let loaded = state
            .hardware
            .as_mut()
            .ok_or(HardwareError::Unavailable)
            .and_then(|hardware| {
                hardware.load_context(batch.context.n_var, &batch.context.clauses)
            });
        if loaded.is_err() {
            SHADOW_ERROR.fetch_add(batch.pending.len() as u64, Ordering::Relaxed);
            state.loaded_context = None;
            return;
        }
        SHADOW_CONTEXT_LOADS.fetch_add(1, Ordering::Relaxed);
        state.loaded_context = Some(batch.context.clone());
    }
    let queries: Vec<_> = batch.pending.iter().map(|(query, _)| query.clone()).collect();
    let cpu: Vec<_> = batch.pending.iter().map(|(_, result)| *result).collect();
    SHADOW_BATCHES.fetch_add(1, Ordering::Relaxed);
    let result = state
        .hardware
        .as_mut()
        .ok_or(HardwareError::Unavailable)
        .and_then(|hardware| hardware.solve_batch(&queries));
    if result.is_ok() {
        if let Some(hardware) = state.hardware.as_ref() {
            profile_hardware_batch(&queries, &hardware.last_batch_records);
        }
    }
    match result {
        Ok(results) => {
            for (query_index, (hardware, cpu)) in results.into_iter().zip(cpu).enumerate() {
                match (&hardware, cpu) {
                    (IncrementalResult::Sat { .. }, Some(true))
                    | (IncrementalResult::Unsat { .. }, Some(false)) => {
                        SHADOW_AGREE.fetch_add(1, Ordering::Relaxed);
                    }
                    (IncrementalResult::Unknown(_), _) | (_, None) => {
                        SHADOW_UNKNOWN.fetch_add(1, Ordering::Relaxed);
                    }
                    _ => {
                        if matches!(hardware, IncrementalResult::Sat { .. }) {
                            SHADOW_HW_SAT_CPU_UNSAT.fetch_add(1, Ordering::Relaxed);
                        } else {
                            SHADOW_HW_UNSAT_CPU_SAT.fetch_add(1, Ordering::Relaxed);
                        }
                        let prior = SHADOW_MISMATCH.fetch_add(1, Ordering::Relaxed);
                        if prior < 4 {
                            let query = &queries[query_index];
                            dump_mismatch_dimacs(&batch.context, query);
                            let n_context_lit: usize = batch
                                .context
                                .clauses
                                .iter()
                                .map(|clause| clause.literals.len())
                                .sum();
                            eprintln!(
                                "inductor-cdcl: mismatch #{}, cpu {:?}, hw {:?}, frame {}, context {} clauses/{} lits, assumptions {:?}, constraints {:?}, domain {:?}",
                                prior + 1,
                                cpu,
                                hardware,
                                query.frame,
                                batch.context.clauses.len(),
                                n_context_lit,
                                query.assumptions,
                                query.constraints,
                                query.domain,
                            );
                        }
                    }
                }
            }
        }
        Err(_) => {
            SHADOW_ERROR.fetch_add(batch.pending.len() as u64, Ordering::Relaxed);
        }
    }
}

/// Solve a set of already-independent IC3 inquiries in as few XRT submissions
/// as their resident contexts and the command-word limit allow. The returned
/// order matches the input order. This function deliberately does not decide
/// whether a result is proof-safe for IC3; callers may consume a SAT answer
/// only through `DagCnfSolver::install_incremental_sat_model`.
pub fn solve_active_batch(
    requests: Vec<(&DagCnfSolver, IncrementalQuery)>,
) -> Vec<IncrementalResult> {
    let unknown = IncrementalResult::Unknown(super::cdcl::UnknownReason::BackendError);
    let mut output = vec![unknown.clone(); requests.len()];
    if requests.is_empty() || !active_enabled() {
        return output;
    }
    ACTIVE_PASSES.fetch_add(1, Ordering::Relaxed);
    ACTIVE_MAX_READY.fetch_max(requests.len() as u64, Ordering::Relaxed);
    ACTIVE_CANDIDATES.fetch_add(requests.len() as u64, Ordering::Relaxed);
    if requests.len() < active_min_batch_size() {
        ACTIVE_SKIPPED_PASSES.fetch_add(1, Ordering::Relaxed);
        ACTIVE_SKIPPED_SMALL_BATCH.fetch_add(requests.len() as u64, Ordering::Relaxed);
        return output;
    }
    ACTIVE_OFFERED_PASSES.fetch_add(1, Ordering::Relaxed);
    ACTIVE_OFFERED.fetch_add(requests.len() as u64, Ordering::Relaxed);

    struct Group {
        context: ShadowContext,
        pending: Vec<(usize, IncrementalQuery)>,
    }
    let mut groups: Vec<Group> = Vec::new();
    for (index, (solver, query)) in requests.into_iter().enumerate() {
        let (context, query, _) = prepare_batched_query(solver, query, true);
        let fits = query_request_words(&query)
            .and_then(|words| words.checked_add(4))
            .is_some_and(|words| words <= KERNEL_MAX_REQUEST_WORDS);
        if !fits {
            ACTIVE_ERROR.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        if let Some(group) = groups.iter_mut().find(|group| group.context == context) {
            group.pending.push((index, query));
        } else {
            groups.push(Group {
                context,
                pending: vec![(index, query)],
            });
        }
    }

    let Ok(mut state) = active_state().lock() else {
        let n_pending: usize = groups.iter().map(|group| group.pending.len()).sum();
        ACTIVE_ERROR.fetch_add(n_pending as u64, Ordering::Relaxed);
        return output;
    };
    for mut group in groups {
        if pair_scheduler_enabled() {
            schedule_query_pairs(&mut group.pending, |(_, query)| query);
        }
        if state.loaded_context.as_ref() != Some(&group.context) {
            let load_start = std::time::Instant::now();
            let loaded = state
                .hardware
                .as_mut()
                .ok_or(HardwareError::Unavailable)
                .and_then(|hardware| {
                    hardware.load_context(group.context.n_var, &group.context.clauses)
                });
            if loaded.is_err() {
                ACTIVE_CONTEXT_LOAD_NS.fetch_add(
                    load_start.elapsed().as_nanos() as u64,
                    Ordering::Relaxed,
                );
                ACTIVE_ERROR.fetch_add(group.pending.len() as u64, Ordering::Relaxed);
                state.loaded_context = None;
                continue;
            }
            ACTIVE_CONTEXT_LOAD_NS.fetch_add(
                load_start.elapsed().as_nanos() as u64,
                Ordering::Relaxed,
            );
            ACTIVE_CONTEXT_LOADS.fetch_add(1, Ordering::Relaxed);
            state.loaded_context = Some(group.context.clone());
        }

        let mut start = 0usize;
        while start < group.pending.len() {
            let mut words = 4usize;
            let mut end = start;
            while end < group.pending.len() && end - start < active_batch_size() {
                let Some(query_words) = query_request_words(&group.pending[end].1) else {
                    break;
                };
                let Some(next_words) = words.checked_add(query_words) else {
                    break;
                };
                if next_words > KERNEL_MAX_REQUEST_WORDS {
                    break;
                }
                words = next_words;
                end += 1;
            }
            if end == start {
                ACTIVE_ERROR.fetch_add(1, Ordering::Relaxed);
                start += 1;
                continue;
            }
            let queries: Vec<_> = group.pending[start..end]
                .iter()
                .map(|(_, query)| query.clone())
                .collect();
            ACTIVE_BATCHES.fetch_add(1, Ordering::Relaxed);
            let batch_start = std::time::Instant::now();
            let result = state
                .hardware
                .as_mut()
                .ok_or(HardwareError::Unavailable)
                .and_then(|hardware| hardware.solve_batch(&queries));
            ACTIVE_BATCH_NS.fetch_add(
                batch_start.elapsed().as_nanos() as u64,
                Ordering::Relaxed,
            );
            if result.is_ok() {
                if let Some(hardware) = state.hardware.as_ref() {
                    profile_hardware_batch(&queries, &hardware.last_batch_records);
                    ACTIVE_HW_DECISIONS.fetch_add(
                        hardware.last_batch_work.decisions,
                        Ordering::Relaxed,
                    );
                    ACTIVE_HW_CONFLICTS.fetch_add(
                        hardware.last_batch_work.conflicts,
                        Ordering::Relaxed,
                    );
                    ACTIVE_HW_PROPAGATIONS.fetch_add(
                        hardware.last_batch_work.propagations,
                        Ordering::Relaxed,
                    );
                    ACTIVE_HW_LEARNTS.fetch_add(
                        hardware.last_batch_work.learnt_clauses,
                        Ordering::Relaxed,
                    );
                }
            }
            match result {
                Ok(results) => {
                    for ((index, _), result) in group.pending[start..end]
                        .iter()
                        .zip(results)
                    {
                        match &result {
                            IncrementalResult::Sat { .. } => {
                                ACTIVE_HW_SAT.fetch_add(1, Ordering::Relaxed);
                            }
                            IncrementalResult::Unsat { .. } => {
                                ACTIVE_HW_UNSAT.fetch_add(1, Ordering::Relaxed);
                            }
                            IncrementalResult::Unknown(_) => {
                                ACTIVE_UNKNOWN.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        output[*index] = result;
                    }
                }
                Err(_) => {
                    ACTIVE_ERROR.fetch_add((end - start) as u64, Ordering::Relaxed);
                }
            }
            start = end;
        }
    }
    output
}

pub fn note_active_sat_model(accepted: bool, validation_ns: u64) {
    ACTIVE_VALIDATE_NS.fetch_add(validation_ns, Ordering::Relaxed);
    if accepted {
        ACTIVE_SAT_USED.fetch_add(1, Ordering::Relaxed);
    } else {
        ACTIVE_SAT_REJECTED.fetch_add(1, Ordering::Relaxed);
        ACTIVE_CPU_FALLBACK.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn note_active_cpu_fallback() {
    ACTIVE_CPU_FALLBACK.fetch_add(1, Ordering::Relaxed);
}

/// Queue one already-answered GipSAT inquiry. Queries are grouped by their
/// exact resident snapshot rather than flushed when IC3 switches frames. This
/// exposes the batching available to a future multi-context scheduler while
/// shadow mode remains independent of the CPU control flow.
pub fn queue_shadow(
    solver: &DagCnfSolver,
    query: IncrementalQuery,
    cpu_result: Option<bool>,
) {
    if !shadow_enabled() || query.domain.is_empty() {
        return;
    }
    SHADOW_OFFERED.fetch_add(1, Ordering::Relaxed);
    let prefer_query_lemmas =
        std::env::var_os("INDUCTOR_CDCL_SHADOW_RESIDENT_LEMMAS").is_none();
    let (context, query, used_query_lemmas) =
        prepare_batched_query(solver, query, prefer_query_lemmas);
    if used_query_lemmas {
        SHADOW_QUERY_LEMMAS.fetch_add(1, Ordering::Relaxed);
    }
    let Some(query_words) = query_request_words(&query) else {
        SHADOW_ERROR.fetch_add(1, Ordering::Relaxed);
        return;
    };
    if query_words + 4 > KERNEL_MAX_REQUEST_WORDS {
        SHADOW_ERROR.fetch_add(1, Ordering::Relaxed);
        return;
    }
    let Ok(mut state) = shadow_state().lock() else {
        SHADOW_ERROR.fetch_add(1, Ordering::Relaxed);
        return;
    };

    let mut batch_index = state
        .batches
        .iter()
        .position(|batch| batch.context == context);
    if let Some(index) = batch_index {
        let queued_words = state.batches[index]
            .pending
            .iter()
            .try_fold(4usize, |words, (queued, _)| {
                words.checked_add(query_request_words(queued)?)
            });
        if queued_words
            .and_then(|words| words.checked_add(query_words))
            .is_none_or(|words| words > KERNEL_MAX_REQUEST_WORDS)
        {
            flush_batch_locked(&mut state, index);
            batch_index = None;
        }
    }
    let index = batch_index.unwrap_or_else(|| {
        state.batches.push(ShadowBatch {
            context,
            pending: Vec::with_capacity(shadow_batch_size()),
        });
        state.batches.len() - 1
    });
    state.batches[index].pending.push((query, cpu_result));
    if state.batches[index].pending.len() >= shadow_batch_size() {
        flush_batch_locked(&mut state, index);
    }
}

/// Flush pending shadow work and print compact shadow/active summaries.
pub fn flush_and_report() {
    if shadow_enabled() {
        if let Ok(mut state) = shadow_state().lock() {
            while !state.batches.is_empty() {
                flush_batch_locked(&mut state, 0);
            }
        }
        eprintln!(
            "inductor-cdcl: shadow offered {}, batches {}, context loads {}, query-lemma inquiries {}, agree {}, mismatch {} (hw-sat/cpu-unsat {}, hw-unsat/cpu-sat {}), unknown {}, errors {}",
            SHADOW_OFFERED.load(Ordering::Relaxed),
            SHADOW_BATCHES.load(Ordering::Relaxed),
            SHADOW_CONTEXT_LOADS.load(Ordering::Relaxed),
            SHADOW_QUERY_LEMMAS.load(Ordering::Relaxed),
            SHADOW_AGREE.load(Ordering::Relaxed),
            SHADOW_MISMATCH.load(Ordering::Relaxed),
            SHADOW_HW_SAT_CPU_UNSAT.load(Ordering::Relaxed),
            SHADOW_HW_UNSAT_CPU_SAT.load(Ordering::Relaxed),
            SHADOW_UNKNOWN.load(Ordering::Relaxed),
            SHADOW_ERROR.load(Ordering::Relaxed),
        );
    }
    if active_enabled() {
        eprintln!(
            "inductor-cdcl: active pair-scheduler {}, passes {} (skipped {}, offered {}, max-ready {}), candidates {}, skipped-small-batch {}, offered {}, batches {}, context loads {}, hw SAT {}, hw UNSAT {}, unknown {}, errors {}, hw work decisions/conflicts/propagations/learnts {}/{}/{}/{}, validated SAT used {}, rejected SAT {}, CPU fallbacks executed {}, init {:.3} ms, load {:.3} ms, batches {:.3} ms, validate {:.3} ms",
            if pair_scheduler_enabled() { "on" } else { "off" },
            ACTIVE_PASSES.load(Ordering::Relaxed),
            ACTIVE_SKIPPED_PASSES.load(Ordering::Relaxed),
            ACTIVE_OFFERED_PASSES.load(Ordering::Relaxed),
            ACTIVE_MAX_READY.load(Ordering::Relaxed),
            ACTIVE_CANDIDATES.load(Ordering::Relaxed),
            ACTIVE_SKIPPED_SMALL_BATCH.load(Ordering::Relaxed),
            ACTIVE_OFFERED.load(Ordering::Relaxed),
            ACTIVE_BATCHES.load(Ordering::Relaxed),
            ACTIVE_CONTEXT_LOADS.load(Ordering::Relaxed),
            ACTIVE_HW_SAT.load(Ordering::Relaxed),
            ACTIVE_HW_UNSAT.load(Ordering::Relaxed),
            ACTIVE_UNKNOWN.load(Ordering::Relaxed),
            ACTIVE_ERROR.load(Ordering::Relaxed),
            ACTIVE_HW_DECISIONS.load(Ordering::Relaxed),
            ACTIVE_HW_CONFLICTS.load(Ordering::Relaxed),
            ACTIVE_HW_PROPAGATIONS.load(Ordering::Relaxed),
            ACTIVE_HW_LEARNTS.load(Ordering::Relaxed),
            ACTIVE_SAT_USED.load(Ordering::Relaxed),
            ACTIVE_SAT_REJECTED.load(Ordering::Relaxed),
            ACTIVE_CPU_FALLBACK.load(Ordering::Relaxed),
            ACTIVE_INIT_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_CONTEXT_LOAD_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_BATCH_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_VALIDATE_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
        );
    }
    if profile_enabled() {
        print_profile();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use logicrs::DagCnf;
    use logicrs::satif::Satif;
    use logicrs::{Lit, Var};

    #[test]
    fn resident_context_packing_rejects_bad_ranges_and_variables() {
        let a = Lit::new(Var::from(0), true);
        let good = ResidentClause::new(0, 3, LitVec::from([a]));
        let words = pack_clauses(&[1, 1], 1, std::slice::from_ref(&good)).unwrap();
        assert_eq!(words, vec![1, 1, 0, 3, 1, u32::from(a)]);

        let bad_range = ResidentClause::new(4, 3, LitVec::from([a]));
        assert_eq!(
            pack_clauses(&[1], 1, &[bad_range]),
            Err(HardwareError::InvalidContext),
        );
        let outside = Lit::new(Var::from(1), true);
        assert_eq!(
            pack_clauses(&[1], 1, &[ResidentClause::new(0, 0, LitVec::from([outside]))]),
            Err(HardwareError::InvalidContext),
        );
    }

    #[test]
    fn batch_work_decoder_sums_variable_length_records() {
        let words = [
            super::super::cdcl::ABI_VERSION,
            2,
            19,
            0,
            1,
            0,
            1,
            0,
            5,
            6,
            7,
            8,
            0,
            42,
            2,
            0,
            0,
            0,
            11,
            12,
            13,
            14,
            0,
        ];
        let records = decode_batch_work_records(&words, 2).unwrap();
        assert_eq!(
            records,
            vec![
                HardwareWork {
                    status: 1,
                    reason: 0,
                    decisions: 5,
                    conflicts: 6,
                    propagations: 7,
                    learnt_clauses: 8,
                },
                HardwareWork {
                    status: 2,
                    reason: 0,
                    decisions: 11,
                    conflicts: 12,
                    propagations: 13,
                    learnt_clauses: 14,
                },
            ]
        );
        assert_eq!(
            sum_hardware_work(&records),
            HardwareWork {
                decisions: 16,
                conflicts: 18,
                propagations: 20,
                learnt_clauses: 22,
                ..HardwareWork::default()
            }
        );
        assert_eq!(decode_batch_work_records(&words[..22], 2), None);
    }

    #[test]
    fn shadow_batch_word_accounting_matches_query_wire_layout() {
        let a = Lit::new(Var::from(0), true);
        let b = Lit::new(Var::from(1), false);
        let mut query = IncrementalQuery::new(2, LitVec::from([a]));
        query.constraints.push(LitVec::from([a, b]));
        query.domain = vec![Var::from(0), Var::from(1)];

        let (header, payload) = query.pack();
        assert_eq!(query_request_words(&query), Some(14));
        assert_eq!(query_request_words(&query), Some(header.as_words().len() + payload.len()));
    }

    #[test]
    fn query_lemma_mode_shares_only_the_transition_context() {
        let mut dc = DagCnf::new();
        let a = dc.new_var().lit();
        let b = dc.new_var().lit();
        dc.add_rel(b.var(), &[LitVec::from([a, b])]);
        let mut solver = DagCnfSolver::new(&dc);
        solver.accel_level = 7;
        solver.add_clause(&[!a, b]);

        let mut query = IncrementalQuery::new(7, LitVec::from([a]));
        query.constraints.push(LitVec::from([b]));
        query.domain = vec![a.var(), b.var()];

        let (shared, expanded, used) = prepare_batched_query(&solver, query.clone(), true);
        assert!(used);
        assert!(shared
            .clauses
            .iter()
            .all(|clause| clause.lo == 0 && clause.hi == u32::MAX));
        assert!(shared
            .clauses
            .iter()
            .any(|clause| clause.literals.as_slice() == [a, b]));
        assert!(!shared
            .clauses
            .iter()
            .any(|clause| clause.literals.as_slice() == [!a, b]));
        assert_eq!(expanded.constraints.len(), 2);
        assert!(expanded
            .constraints
            .iter()
            .any(|clause| clause.as_slice() == [!a, b]));

        let (exact, unchanged, used) = prepare_batched_query(&solver, query.clone(), false);
        assert!(!used);
        assert_eq!(unchanged.constraints, query.constraints);
        assert!(exact
            .clauses
            .iter()
            .all(|clause| clause.lo == 7 && clause.hi == 7));
        assert!(exact
            .clauses
            .iter()
            .any(|clause| clause.literals.as_slice() == [!a, b]));
    }

    #[test]
    fn pair_scheduler_places_similar_heavy_queries_in_the_same_round() {
        let query_with_domain = |n: u32| {
            let mut query = IncrementalQuery::new(0, LitVec::new());
            query.domain = (0..n).map(Var::from).collect();
            query
        };
        let mut pending = vec![
            (0usize, query_with_domain(8)),
            (1usize, query_with_domain(1)),
            (2usize, query_with_domain(7)),
            (3usize, query_with_domain(2)),
        ];
        let pair_cost = |queries: &[(usize, IncrementalQuery)]| -> u64 {
            queries
                .chunks(2)
                .map(|pair| {
                    pair.iter()
                        .map(|(_, query)| query_work_score(query))
                        .max()
                        .unwrap()
                })
                .sum()
        };
        assert_eq!(pair_cost(&pending), 15);
        schedule_query_pairs(&mut pending, |(_, query)| query);
        assert_eq!(
            pending
                .iter()
                .map(|(index, _)| *index)
                .collect::<Vec<_>>(),
            [0, 2, 3, 1]
        );
        assert_eq!(pair_cost(&pending), 10);
    }
}
