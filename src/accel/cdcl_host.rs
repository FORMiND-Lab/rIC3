//! XRT-backed implementation of the incremental CDCL semantic boundary.
//!
//! The C++ bridge owns one persistent kernel context and reusable DMA buffers.
//! Transport or device failures become `Unknown(BackendError)` through the
//! `IncrementalCdcl` implementation; they are never interpreted as SAT/UNSAT.

use super::cdcl::{
    BatchHeader, PROFILE_ANALYZE, PROFILE_ANALYZED_LITERALS, PROFILE_BACKTRACK,
    PROFILE_CLEANUP, PROFILE_DECIDE, PROFILE_EMIT, PROFILE_EVALUATED_LITERALS,
    PROFILE_LEARN, PROFILE_LEARNT_LITERALS, PROFILE_OCCURRENCE_UPDATES,
    PROFILE_PARTIAL_OCCURRENCE_SCANS, PROFILE_PROPAGATE, PROFILE_ROOT,
    PROFILE_SETUP, PROFILE_UNDO_ASSIGNMENTS, PROFILE_UNDO_OCCURRENCES,
    PROFILE_UNIT_CANDIDATES, RESPONSE_HEADER_WORDS, STAGE_PROFILE_COUNTERS,
    STAGE_PROFILE_MAGIC, STAGE_PROFILE_STAGE_COUNTERS, STAGE_PROFILE_VERSION,
    STAGE_PROFILE_WORDS, Status, UnknownReason, WANT_STAGE_PROFILE,
};
use crate::gipsat::{
    BatchDecodeError, DagCnfSolver, IncrementalCdcl, IncrementalQuery, IncrementalResult,
    pack_batch, solve_on_cpu_after_hardware_unknown,
};
#[cfg(has_cdcl_accel)]
use crate::gipsat::decode_batch_results;
use logicrs::LitVec;
#[cfg(has_cdcl_accel)]
use std::ffi::CString;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(has_cdcl_accel)]
unsafe extern "C" {
    fn ind_cdcl_open(path: *const std::os::raw::c_char) -> i32;
    fn ind_cdcl_connect(path: *const std::os::raw::c_char) -> i32;
    fn ind_cdcl_load_context(request: *const u32, request_words: u32) -> i32;
    fn ind_cdcl_add_frame_clauses(request: *const u32, request_words: u32) -> i32;
    fn ind_cdcl_solve_batch(
        request: *const u32,
        request_words: u32,
        response: *mut u32,
        response_capacity_words: u32,
        out_response_words: *mut u32,
    ) -> i32;
    fn ind_cdcl_load_context_and_solve_batch(
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
    profile_counters: [u64; STAGE_PROFILE_COUNTERS],
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
            profile_counters: [0; STAGE_PROFILE_COUNTERS],
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

/// Validate a profiled variable-length response. The returned copy has its
/// diagnostic trailers removed so the stable ABI-v1 semantic decoder remains
/// unaware of profiling, while the work records retain all stage counters.
fn decode_profiled_batch_wire(
    words: &[u32],
    n_queries: usize,
) -> Option<(Vec<HardwareWork>, Vec<u32>)> {
    let prefix = words.get(..4)?;
    if usize::try_from(prefix[1]).ok()? != n_queries {
        return None;
    }
    let result_words = usize::try_from(prefix[2]).ok()?;
    if words.len() != 4usize.checked_add(result_words)? {
        return None;
    }
    let mut records = Vec::with_capacity(n_queries);
    let mut semantic = Vec::with_capacity(
        words
            .len()
            .saturating_sub(n_queries.saturating_mul(STAGE_PROFILE_WORDS)),
    );
    semantic.extend_from_slice(prefix);
    let mut offset = 4usize;
    for _ in 0..n_queries {
        let header = words.get(offset..offset.checked_add(RESPONSE_HEADER_WORDS)?)?;
        let mut work = HardwareWork {
            status: header[0],
            reason: header[1],
            decisions: u64::from(header[4]),
            conflicts: u64::from(header[5]),
            propagations: u64::from(header[6]),
            learnt_clauses: u64::from(header[7]),
            profile_counters: [0; STAGE_PROFILE_COUNTERS],
        };
        let payload_words = usize::try_from(header[2]).ok()?
            .checked_add(usize::try_from(header[3]).ok()?)?;
        let semantic_end = offset
            .checked_add(RESPONSE_HEADER_WORDS)?
            .checked_add(payload_words)?;
        semantic.extend_from_slice(words.get(offset..semantic_end)?);
        offset = semantic_end;
        let trailer = words.get(offset..offset.checked_add(STAGE_PROFILE_WORDS)?)?;
        if trailer[0] != STAGE_PROFILE_MAGIC
            || trailer[1] != STAGE_PROFILE_VERSION
            || usize::try_from(trailer[2]).ok()? != STAGE_PROFILE_COUNTERS
        {
            return None;
        }
        for (counter, value) in work.profile_counters.iter_mut().enumerate() {
            let at = 3 + 2 * counter;
            *value = u64::from(trailer[at]) | (u64::from(trailer[at + 1]) << 32);
        }
        offset += STAGE_PROFILE_WORDS;
        records.push(work);
    }
    if offset != words.len() {
        return None;
    }
    semantic[2] = u32::try_from(semantic.len().checked_sub(4)?).ok()?;
    Some((records, semantic))
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
            for (sum, value) in total
                .profile_counters
                .iter_mut()
                .zip(work.profile_counters)
            {
                *sum = sum.saturating_add(value);
            }
            total
        })
}

fn pack_batch_request(
    queries: &[IncrementalQuery],
    want_stage_profile: bool,
) -> Result<(Vec<u32>, usize), HardwareError> {
    let result_words = queries
        .iter()
        .try_fold(0usize, |total, query| {
            let record = RESPONSE_HEADER_WORDS
                .checked_add(query.domain.len().max(query.assumptions.len()))?
                .checked_add(if want_stage_profile {
                    STAGE_PROFILE_WORDS
                } else {
                    0
                })?;
            total.checked_add(
                record,
            )
        })
        .ok_or(HardwareError::Capacity)?;
    let result_words_u32 =
        u32::try_from(result_words).map_err(|_| HardwareError::Capacity)?;
    let (batch, mut payload) = pack_batch(queries, result_words_u32);
    if want_stage_profile {
        let mut offset = 0usize;
        for query in queries {
            payload[offset + 2] |= WANT_STAGE_PROFILE;
            offset = offset
                .checked_add(query_request_words(query).ok_or(HardwareError::Capacity)?)
                .ok_or(HardwareError::Capacity)?;
        }
        if offset != payload.len() {
            return Err(HardwareError::InvalidContext);
        }
    }
    let mut request = Vec::with_capacity(4 + payload.len());
    let BatchHeader {
        version,
        n_queries,
        n_request_words,
        result_capacity_words,
    } = batch;
    request.extend([version, n_queries, n_request_words, result_capacity_words]);
    request.extend(payload);
    u32::try_from(request.len()).map_err(|_| HardwareError::Capacity)?;
    let response_capacity = result_words.checked_add(4).ok_or(HardwareError::Capacity)?;
    u32::try_from(response_capacity).map_err(|_| HardwareError::Capacity)?;
    Ok((request, response_capacity))
}

fn pack_load_context_and_batch_request(
    n_var: u32,
    clauses: &[ResidentClause],
    queries: &[IncrementalQuery],
    want_stage_profile: bool,
) -> Result<(Vec<u32>, usize), HardwareError> {
    let n_clause = u32::try_from(clauses.len()).map_err(|_| HardwareError::Capacity)?;
    let context = pack_clauses(&[n_var, n_clause], n_var, clauses)?;
    let context_words = u32::try_from(context.len()).map_err(|_| HardwareError::Capacity)?;
    let (batch, response_capacity) = pack_batch_request(queries, want_stage_profile)?;
    let capacity = 1usize
        .checked_add(context.len())
        .and_then(|words| words.checked_add(batch.len()))
        .ok_or(HardwareError::Capacity)?;
    u32::try_from(capacity).map_err(|_| HardwareError::Capacity)?;
    let mut request = Vec::with_capacity(capacity);
    request.push(context_words);
    request.extend(context);
    request.extend(batch);
    Ok((request, response_capacity))
}

pub struct HardwareCdcl {
    n_var: u32,
    last_batch_work: HardwareWork,
    last_batch_records: Vec<HardwareWork>,
    stage_profile: bool,
}

impl HardwareCdcl {
    pub fn compiled() -> bool {
        cfg!(has_cdcl_accel)
    }

    /// Open the xclbin named explicitly by the caller.
    pub fn open(path: &str) -> Result<Self, HardwareError> {
        #[cfg(has_cdcl_accel)]
        {
            let rc = if let Ok(socket) = std::env::var("INDUCTOR_CDCL_SERVER") {
                let socket = CString::new(socket).map_err(|_| HardwareError::InvalidPath)?;
                unsafe { ind_cdcl_connect(socket.as_ptr()) }
            } else {
                let path = CString::new(path).map_err(|_| HardwareError::InvalidPath)?;
                unsafe { ind_cdcl_open(path.as_ptr()) }
            };
            if rc != 0 {
                return Err(HardwareError::Open(rc));
            }
            Ok(Self {
                n_var: 0,
                last_batch_work: HardwareWork::default(),
                last_batch_records: Vec::new(),
                stage_profile: stage_profile_enabled(),
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
        let (request, response_capacity) = pack_batch_request(queries, self.stage_profile)?;
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
            self.decode_batch_response(queries, &response)
        }
        #[cfg(not(has_cdcl_accel))]
        {
            let _ = (request, response_capacity_u32, response);
            Err(HardwareError::Unavailable)
        }
    }

    /// Atomically replace the resident context and execute its first batch in
    /// one kernel/RPC command. Older bitstreams reject this extension; callers
    /// may safely retry with `load_context` followed by `solve_batch`.
    pub fn load_context_and_solve_batch(
        &mut self,
        n_var: u32,
        clauses: &[ResidentClause],
        queries: &[IncrementalQuery],
    ) -> Result<Vec<IncrementalResult>, HardwareError> {
        self.last_batch_work = HardwareWork::default();
        self.last_batch_records.clear();
        let (request, response_capacity) =
            pack_load_context_and_batch_request(
                n_var, clauses, queries, self.stage_profile,
            )?;
        profile_resident_context(n_var, clauses);
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
                ind_cdcl_load_context_and_solve_batch(
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
            let results = self.decode_batch_response(queries, &response)?;
            self.n_var = n_var;
            Ok(results)
        }
        #[cfg(not(has_cdcl_accel))]
        {
            let _ = (request, response_capacity_u32, response);
            Err(HardwareError::Unavailable)
        }
    }

    #[cfg(has_cdcl_accel)]
    fn decode_batch_response(
        &mut self,
        queries: &[IncrementalQuery],
        response: &[u32],
    ) -> Result<Vec<IncrementalResult>, HardwareError> {
        let results = if self.stage_profile {
            let (records, semantic) = decode_profiled_batch_wire(response, queries.len())
                .ok_or(HardwareError::Decode(BatchDecodeError::InvalidResultShape))?;
            self.last_batch_records = records;
            decode_batch_results(queries, &semantic).map_err(HardwareError::Decode)?
        } else {
            self.last_batch_records = decode_batch_work_records(response, queries.len())
                .ok_or(HardwareError::Decode(BatchDecodeError::InvalidResultShape))?;
            decode_batch_results(queries, response).map_err(HardwareError::Decode)?
        };
        self.last_batch_work = sum_hardware_work(&self.last_batch_records);
        Ok(results)
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

#[derive(Clone, Copy, Debug, Default)]
struct PairedCpuWork {
    status: u32,
    elapsed_ns: u64,
    decisions: u64,
    conflicts: u64,
    propagations: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct PairedPreflightWork {
    status: u32,
    reason: u32,
    elapsed_ns: u64,
    clone_ns: u64,
    solve_ns: u64,
    decisions: u64,
    conflicts: u64,
    propagations: u64,
    selected: bool,
}

#[derive(Clone, Debug)]
pub enum ActivePreflight {
    /// No preflight was requested, or this inquiry exhausted its CPU budget.
    Fpga,
    /// Keep this inquiry on the ordinary CPU path without a reusable answer.
    CpuFallback,
    /// Exact GipSAT completed inside the budget; consume this answer directly.
    Conclusive(IncrementalResult),
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
static ACTIVE_COMBINED_BATCHES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_COMBINED_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_HW_SAT: AtomicU64 = AtomicU64::new(0);
static ACTIVE_HW_UNSAT: AtomicU64 = AtomicU64::new(0);
static ACTIVE_UNKNOWN: AtomicU64 = AtomicU64::new(0);
static ACTIVE_ERROR: AtomicU64 = AtomicU64::new(0);
static ACTIVE_SAT_USED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_SAT_REJECTED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_UNSAT_CORE_USED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_UNSAT_CORE_REJECTED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_UNSAT_ASSUMPTION_LITS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_UNSAT_HW_CORE_LITS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_UNSAT_CPU_CORE_LITS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_CPU_FALLBACK: AtomicU64 = AtomicU64::new(0);
static ACTIVE_INIT_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_STATE_WAIT_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_CONTEXT_LOAD_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_COMBINED_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_COMBINED_FALLBACK_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BATCH_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_VALIDATE_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_UNSAT_VALIDATE_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_HW_DECISIONS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_HW_CONFLICTS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_HW_PROPAGATIONS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_HW_LEARNTS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PREFLIGHT_CANDIDATES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PREFLIGHT_CONCLUSIVE: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PREFLIGHT_SELECTED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PREFLIGHT_STATIC_FILTERED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PREFLIGHT_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PREFLIGHT_CONFLICTS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PREFLIGHT_SAT_REUSED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PREFLIGHT_UNSAT_REUSED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PREFLIGHT_REJECTED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PREFLIGHT_RESTORE_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_SAMPLE_QUERIES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_SAMPLE_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_SAMPLE_CLONE_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_SAMPLE_SOLVE_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_SAMPLE_FPGA_BATCHES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_SAMPLE_CPU_BATCHES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_SAMPLE_FPGA_RETAINED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_SAMPLE_CPU_REJECTED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_SAMPLE_UNDERSIZED_REJECTED: AtomicU64 = AtomicU64::new(0);
static PAIRED_BATCH_ID: AtomicU64 = AtomicU64::new(0);
static PAIRED_QUERIES: AtomicU64 = AtomicU64::new(0);
static PAIRED_FILTERED: AtomicU64 = AtomicU64::new(0);
static PAIRED_AGREE: AtomicU64 = AtomicU64::new(0);
static PAIRED_MISMATCH: AtomicU64 = AtomicU64::new(0);
static PAIRED_UNKNOWN: AtomicU64 = AtomicU64::new(0);
static PAIRED_CPU_NS: AtomicU64 = AtomicU64::new(0);
static PAIRED_BASELINE_CPU_NS: AtomicU64 = AtomicU64::new(0);
static PAIRED_HW_NS: AtomicU64 = AtomicU64::new(0);
static PAIRED_HW_FASTER_BATCHES: AtomicU64 = AtomicU64::new(0);
static PAIRED_PREFLIGHT_CANDIDATES: AtomicU64 = AtomicU64::new(0);
static PAIRED_PREFLIGHT_CONCLUSIVE: AtomicU64 = AtomicU64::new(0);
static PAIRED_PREFLIGHT_SELECTED: AtomicU64 = AtomicU64::new(0);
static PAIRED_PREFLIGHT_NS: AtomicU64 = AtomicU64::new(0);
static PAIRED_PREFLIGHT_CLONE_NS: AtomicU64 = AtomicU64::new(0);
static PAIRED_PREFLIGHT_SOLVE_NS: AtomicU64 = AtomicU64::new(0);
static PAIRED_WRITER: std::sync::OnceLock<
    Option<std::sync::Mutex<BufWriter<std::fs::File>>>,
> = std::sync::OnceLock::new();
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

fn stage_profile_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("INDUCTOR_CDCL_STAGE_PROFILE").is_ok())
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
    if stage_profile_enabled() && queries.len() == work.len() {
        for (batch_index, (query, work)) in queries.iter().zip(work).enumerate() {
            let counters = &work.profile_counters;
            let total = counters[..STAGE_PROFILE_STAGE_COUNTERS]
                .iter()
                .copied()
                .fold(0u64, u64::saturating_add);
            eprintln!(
                "inductor-cdcl-stage: batch-index {} frame {} status {} reason {} assumptions {} constraints {} domain {} total-entries {} setup {} root {} propagate {} analyze {} backtrack {} learn {} decide {} emit {} cleanup {} occurrence-updates {} partial-occurrence-scans {} evaluated-literals {} unit-candidates {} analyzed-literals {} undo-occurrences {} undo-assignments {} learnt-literals {}",
                batch_index,
                query.frame,
                work.status,
                work.reason,
                query.assumptions.len(),
                query.constraints.len(),
                query.domain.len(),
                total,
                counters[PROFILE_SETUP],
                counters[PROFILE_ROOT],
                counters[PROFILE_PROPAGATE],
                counters[PROFILE_ANALYZE],
                counters[PROFILE_BACKTRACK],
                counters[PROFILE_LEARN],
                counters[PROFILE_DECIDE],
                counters[PROFILE_EMIT],
                counters[PROFILE_CLEANUP],
                counters[PROFILE_OCCURRENCE_UPDATES],
                counters[PROFILE_PARTIAL_OCCURRENCE_SCANS],
                counters[PROFILE_EVALUATED_LITERALS],
                counters[PROFILE_UNIT_CANDIDATES],
                counters[PROFILE_ANALYZED_LITERALS],
                counters[PROFILE_UNDO_OCCURRENCES],
                counters[PROFILE_UNDO_ASSIGNMENTS],
                counters[PROFILE_LEARNT_LITERALS],
            );
        }
    }
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

fn paired_min_frame() -> u32 {
    static FRAME: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *FRAME.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_PAIRED_MIN_FRAME")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(0)
    })
}

fn paired_max_assumptions() -> usize {
    static ASSUMPTIONS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ASSUMPTIONS.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_PAIRED_MAX_ASSUMPTIONS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(usize::MAX)
    })
}

fn paired_preflight_conflicts() -> Option<u32> {
    static CONFLICTS: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *CONFLICTS.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_PAIRED_PREFLIGHT_CONFLICTS")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .filter(|value| *value > 0)
    })
}

fn active_compare_cpu_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        active_enabled()
            && std::env::var("INDUCTOR_CDCL_ACTIVE_COMPARE_CPU")
                .ok()
                .is_some_and(|value| !matches!(value.as_str(), "0" | "false" | "off"))
    })
}

pub fn active_preflight_conflicts() -> Option<u32> {
    static CONFLICTS: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *CONFLICTS.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_ACTIVE_PREFLIGHT_CONFLICTS")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .filter(|value| *value > 0)
    })
}

fn active_preflight_max_assumptions() -> usize {
    static ASSUMPTIONS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ASSUMPTIONS.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_ACTIVE_PREFLIGHT_MAX_ASSUMPTIONS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(usize::MAX)
    })
}

fn active_sample_queries() -> usize {
    static QUERIES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *QUERIES.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_ACTIVE_SAMPLE_QUERIES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0)
    })
}

fn active_sample_min_cpu_ns() -> u64 {
    static NS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *NS.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_ACTIVE_SAMPLE_MIN_CPU_NS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(200_000)
    })
}

fn representative_sample_positions(n_candidates: usize, n_sample: usize) -> Vec<usize> {
    let n_sample = n_sample.min(n_candidates);
    if n_sample == 0 {
        return Vec::new();
    }
    // Midpoints of equal-width strata cover the entire frame instead of
    // measuring only its first (often unusually hard) inquiries. u128 keeps
    // the arithmetic defined even for an adversarially large host vector.
    (0..n_sample)
        .map(|sample| {
            (((2u128 * sample as u128 + 1) * n_candidates as u128)
                / (2u128 * n_sample as u128)) as usize
        })
        .collect()
}

fn sample_keeps_fpga(
    sample_solve_ns: &[u64],
    n_remaining: usize,
    min_batch: usize,
    min_cpu_ns: u64,
    all_conclusive: bool,
) -> bool {
    let mut distribution = sample_solve_ns.to_vec();
    distribution.sort_unstable();
    // The lower median is deliberately conservative for an even sample. One
    // expensive outlier must not route a frame whose typical inquiry is cheap.
    let Some(representative_ns) = distribution
        .get(distribution.len().saturating_sub(1) / 2)
        .copied()
    else {
        return false;
    };
    all_conclusive
        && n_remaining >= min_batch
        && representative_ns >= min_cpu_ns
}

/// Avoid paying even the bounded CPU classification cost when the raw
/// propagation pass cannot satisfy the hardware minimum batch threshold.
pub fn active_preflight_should_run(n_candidates: usize) -> bool {
    active_enabled()
        && active_preflight_conflicts().is_some()
        && n_candidates >= active_min_batch_size()
}

/// Classify one live inquiry. Budget exhaustion is an FPGA scheduling hint;
/// conclusive SAT/UNSAT is an exact CPU result that may be reused after its
/// model/core state is restored at the point where IC3 consumes it.
pub fn active_preflight_classify(
    solver: &mut DagCnfSolver,
    query: &IncrementalQuery,
) -> ActivePreflight {
    let Some(conflict_limit) = active_preflight_conflicts() else {
        return ActivePreflight::Fpga;
    };
    if query.assumptions.len() > active_preflight_max_assumptions() {
        ACTIVE_PREFLIGHT_STATIC_FILTERED.fetch_add(1, Ordering::Relaxed);
        return ActivePreflight::CpuFallback;
    }
    let start = std::time::Instant::now();
    let result = solver.classify_incremental_preflight(query, conflict_limit);
    let elapsed_ns = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
    let selected = matches!(
        &result,
        IncrementalResult::Unknown(UnknownReason::ConflictBudget)
    );
    ACTIVE_PREFLIGHT_CANDIDATES.fetch_add(1, Ordering::Relaxed);
    ACTIVE_PREFLIGHT_NS.fetch_add(elapsed_ns, Ordering::Relaxed);
    ACTIVE_PREFLIGHT_CONFLICTS.fetch_add(u64::from(solver.probe.n_conflict), Ordering::Relaxed);
    if selected {
        ACTIVE_PREFLIGHT_SELECTED.fetch_add(1, Ordering::Relaxed);
        ActivePreflight::Fpga
    } else {
        ACTIVE_PREFLIGHT_CONCLUSIVE.fetch_add(1, Ordering::Relaxed);
        match result {
            IncrementalResult::Sat { .. } | IncrementalResult::Unsat { .. } => {
                ActivePreflight::Conclusive(result)
            }
            IncrementalResult::Unknown(_) => ActivePreflight::CpuFallback,
        }
    }
}

/// Finish a small number of budget-exhausted inquiries exactly on GipSAT
/// clones and use a representative restart cost to route the rest of this
/// frame.
/// Cloning keeps the sample from changing live phase/activity and therefore
/// changing the cost it is trying to predict. Sampled answers are retained as
/// trusted CPU results rather than discarded. Samples are spread across the
/// frame and the lower median rejects a batch dominated by cheap inquiries
/// even if one prefix inquiry is an expensive outlier. This is opt-in while
/// the crossover is validated across models.
pub fn active_sample_select(
    solver: &DagCnfSolver,
    queries: &[IncrementalQuery],
    decisions: &mut [ActivePreflight],
) {
    let requested = active_sample_queries();
    if requested == 0 || queries.len() != decisions.len() {
        return;
    }
    let candidates: Vec<usize> = decisions
        .iter()
        .enumerate()
        .filter_map(|(index, decision)| {
            matches!(decision, ActivePreflight::Fpga).then_some(index)
        })
        .collect();
    if candidates.is_empty() {
        return;
    }
    if candidates.len() < active_min_batch_size() {
        ACTIVE_SAMPLE_UNDERSIZED_REJECTED
            .fetch_add(candidates.len() as u64, Ordering::Relaxed);
        for index in candidates {
            decisions[index] = ActivePreflight::CpuFallback;
        }
        return;
    }

    let n_sample = requested.min(candidates.len());
    let sample_positions = representative_sample_positions(candidates.len(), n_sample);
    let mut sampled = vec![false; candidates.len()];
    let mut sample_ns = 0u64;
    let mut sample_clone_ns = 0u64;
    let mut sample_solve_ns = 0u64;
    let mut sample_solve_distribution = Vec::with_capacity(n_sample);
    let mut all_conclusive = true;
    for position in sample_positions {
        sampled[position] = true;
        let index = candidates[position];
        let start = std::time::Instant::now();
        let mut sample_solver = solver.clone();
        let clone_ns = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        let solve_start = std::time::Instant::now();
        let result = sample_solver.classify_incremental_exact(&queries[index]);
        let solve_ns = solve_start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        sample_clone_ns = sample_clone_ns.saturating_add(clone_ns);
        sample_solve_ns = sample_solve_ns.saturating_add(solve_ns);
        sample_ns = sample_ns.saturating_add(clone_ns.saturating_add(solve_ns));
        sample_solve_distribution.push(solve_ns);
        decisions[index] = match result {
            IncrementalResult::Sat { .. } | IncrementalResult::Unsat { .. } => {
                ActivePreflight::Conclusive(result)
            }
            IncrementalResult::Unknown(_) => {
                all_conclusive = false;
                ActivePreflight::CpuFallback
            }
        };
    }
    ACTIVE_SAMPLE_QUERIES.fetch_add(n_sample as u64, Ordering::Relaxed);
    ACTIVE_SAMPLE_NS.fetch_add(sample_ns, Ordering::Relaxed);
    ACTIVE_SAMPLE_CLONE_NS.fetch_add(sample_clone_ns, Ordering::Relaxed);
    ACTIVE_SAMPLE_SOLVE_NS.fetch_add(sample_solve_ns, Ordering::Relaxed);

    let remaining: Vec<usize> = candidates
        .into_iter()
        .enumerate()
        .filter_map(|(position, index)| (!sampled[position]).then_some(index))
        .collect();
    if sample_keeps_fpga(
        &sample_solve_distribution,
        remaining.len(),
        active_min_batch_size(),
        active_sample_min_cpu_ns(),
        all_conclusive,
    ) {
        ACTIVE_SAMPLE_FPGA_BATCHES.fetch_add(1, Ordering::Relaxed);
        ACTIVE_SAMPLE_FPGA_RETAINED.fetch_add(remaining.len() as u64, Ordering::Relaxed);
    } else {
        ACTIVE_SAMPLE_CPU_BATCHES.fetch_add(1, Ordering::Relaxed);
        ACTIVE_SAMPLE_CPU_REJECTED.fetch_add(remaining.len() as u64, Ordering::Relaxed);
        for index in remaining {
            decisions[index] = ActivePreflight::CpuFallback;
        }
    }
}

/// Optional measurement-only filter. It lets experiments test whether an
/// observable query class packs into profitable FPGA batches without changing
/// the result used by IC3. Production active mode intentionally ignores it.
fn paired_static_selected(query: &IncrementalQuery) -> bool {
    query.frame >= paired_min_frame()
        && query.assumptions.len() <= paired_max_assumptions()
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

fn active_resident_lemmas() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_ACTIVE_RESIDENT_LEMMAS")
            .ok()
            .is_some_and(|value| !matches!(value.as_str(), "0" | "false" | "off"))
    })
}

fn query_lemma_word_limit() -> usize {
    static WORDS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *WORDS.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_QUERY_LEMMA_MAX_WORDS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .map(|value| value.clamp(1, KERNEL_MAX_REQUEST_WORDS))
            // Repeating frame lemmas is attractive only while several
            // inquiries still fit in one DMA command. `fifo.btor` reached
            // ~15k words/query and 1,210 batches; keeping at least eight-query
            // packing as the default crossover reduced that to 38 batches.
            .unwrap_or(KERNEL_MAX_REQUEST_WORDS / 8)
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
            .is_some_and(|words| {
                words <= KERNEL_MAX_REQUEST_WORDS
                    && words <= query_lemma_word_limit()
            });
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

fn result_status(result: &IncrementalResult) -> u32 {
    match result {
        IncrementalResult::Sat { .. } => Status::Sat as u32,
        IncrementalResult::Unsat { .. } => Status::Unsat as u32,
        IncrementalResult::Unknown(_) => Status::Unknown as u32,
    }
}

fn result_reason(result: &IncrementalResult) -> u32 {
    match result {
        IncrementalResult::Unknown(reason) => *reason as u32,
        IncrementalResult::Sat { .. } | IncrementalResult::Unsat { .. } => 0,
    }
}

fn measure_paired_preflight(
    requests: &[(&DagCnfSolver, IncrementalQuery)],
) -> Option<Vec<PairedPreflightWork>> {
    let conflict_limit = paired_preflight_conflicts()?;
    let work: Vec<_> = requests
        .iter()
        .map(|(solver, query)| {
            if !paired_static_selected(query) {
                return PairedPreflightWork::default();
            }
            // Include cloning in the selector cost. Production may eventually
            // run a resumable preflight on the live solver, so this is the
            // conservative proof-neutral measurement rather than a claimed
            // implementation overhead.
            let start = std::time::Instant::now();
            let mut cpu: DagCnfSolver = (**solver).clone();
            let clone_ns = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            let solve_start = std::time::Instant::now();
            let result = cpu.solve_incremental_preflight(query, conflict_limit);
            let solve_ns = solve_start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            let elapsed_ns = clone_ns.saturating_add(solve_ns);
            PairedPreflightWork {
                status: result_status(&result),
                reason: result_reason(&result),
                elapsed_ns,
                clone_ns,
                solve_ns,
                decisions: u64::from(cpu.probe.n_decide),
                conflicts: u64::from(cpu.probe.n_conflict),
                propagations: u64::from(cpu.probe.n_prop),
                selected: matches!(
                    result,
                    IncrementalResult::Unknown(UnknownReason::ConflictBudget)
                ),
            }
        })
        .collect();
    let eligible = work.iter().filter(|item| item.status != 0).count() as u64;
    let selected = work.iter().filter(|item| item.selected).count() as u64;
    PAIRED_PREFLIGHT_CANDIDATES.fetch_add(eligible, Ordering::Relaxed);
    PAIRED_PREFLIGHT_CONCLUSIVE.fetch_add(eligible.saturating_sub(selected), Ordering::Relaxed);
    PAIRED_PREFLIGHT_SELECTED.fetch_add(selected, Ordering::Relaxed);
    PAIRED_PREFLIGHT_NS.fetch_add(
        work.iter().map(|item| item.elapsed_ns).sum(),
        Ordering::Relaxed,
    );
    PAIRED_PREFLIGHT_CLONE_NS.fetch_add(
        work.iter().map(|item| item.clone_ns).sum(),
        Ordering::Relaxed,
    );
    PAIRED_PREFLIGHT_SOLVE_NS.fetch_add(
        work.iter().map(|item| item.solve_ns).sum(),
        Ordering::Relaxed,
    );
    Some(work)
}

fn measure_reference_cpu(
    requests: &[(&DagCnfSolver, IncrementalQuery)],
    apply_paired_filter: bool,
) -> Vec<PairedCpuWork> {
    requests
        .iter()
        .map(|(solver, query)| {
            if apply_paired_filter && !paired_static_selected(query) {
                return PairedCpuWork::default();
            }
            // Clone before starting the timer: the comparison is the work of
            // one independent GipSAT inquiry, not the cost of manufacturing a
            // profiling copy. Remove FPGA-only budgets for the exact CPU run.
            let mut cpu: DagCnfSolver = (**solver).clone();
            let mut cpu_query = query.clone();
            cpu_query.budget.decisions = 0;
            cpu_query.budget.conflicts = 0;
            let start = std::time::Instant::now();
            let result = cpu.solve_incremental(&cpu_query);
            let elapsed_ns = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            PairedCpuWork {
                status: result_status(&result),
                elapsed_ns,
                decisions: u64::from(cpu.probe.n_decide),
                conflicts: u64::from(cpu.probe.n_conflict),
                propagations: u64::from(cpu.probe.n_prop),
            }
        })
        .collect()
}

fn comparison_writer() -> Option<&'static std::sync::Mutex<BufWriter<std::fs::File>>> {
    PAIRED_WRITER
        .get_or_init(|| {
            let variable = if paired_enabled() {
                "INDUCTOR_CDCL_PAIRED_CSV"
            } else {
                "INDUCTOR_CDCL_ACTIVE_COMPARE_CSV"
            };
            let path = std::env::var_os(variable)?;
            let file = match std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&path)
            {
                Ok(file) => file,
                Err(error) => {
                    eprintln!(
                        "inductor-cdcl: cannot create comparison CSV {}: {error}",
                        std::path::Path::new(&path).display(),
                    );
                    return None;
                }
            };
            let mut writer = BufWriter::new(file);
            if writeln!(
                writer,
                "pass_id\tbatch_id\tbatch_size\tposition\toriginal_index\tframe\tcpu_status\tcpu_ns\tcpu_decisions\tcpu_conflicts\tcpu_propagations\thw_status\thw_reason\thw_decisions\thw_conflicts\thw_propagations\thw_learnts\tassumptions\tconstraint_clauses\tconstraint_literals\tdomain\trequest_words\tcontext_vars\tcontext_clauses\tcontext_literals\tbatch_cpu_ns\tbatch_hw_ns\tcontext_load_ns\tinit_ns\tpreflight_selected\tpreflight_status\tpreflight_reason\tpreflight_ns\tpreflight_clone_ns\tpreflight_solve_ns\tpreflight_decisions\tpreflight_conflicts\tpreflight_propagations"
            )
            .is_err()
            {
                return None;
            }
            Some(std::sync::Mutex::new(writer))
        })
        .as_ref()
}

fn record_comparison_batch(
    pass_id: u64,
    context: &ShadowContext,
    pending: &[(usize, IncrementalQuery)],
    queries: &[IncrementalQuery],
    cpu: &[PairedCpuWork],
    preflight: Option<&[PairedPreflightWork]>,
    hardware: &[HardwareWork],
    context_load_ns: u64,
    batch_hw_ns: u64,
) {
    if pending.len() != queries.len() || queries.len() != hardware.len() {
        return;
    }
    let batch_id = PAIRED_BATCH_ID.fetch_add(1, Ordering::Relaxed) + 1;
    let batch_cpu_ns = pending.iter().fold(0u64, |total, (index, _)| {
        total.saturating_add(cpu.get(*index).map_or(0, |work| work.elapsed_ns))
    });
    PAIRED_QUERIES.fetch_add(queries.len() as u64, Ordering::Relaxed);
    PAIRED_CPU_NS.fetch_add(batch_cpu_ns, Ordering::Relaxed);
    PAIRED_HW_NS.fetch_add(batch_hw_ns, Ordering::Relaxed);
    if batch_hw_ns < batch_cpu_ns {
        PAIRED_HW_FASTER_BATCHES.fetch_add(1, Ordering::Relaxed);
    }
    for ((index, _), work) in pending.iter().zip(hardware) {
        let Some(cpu_work) = cpu.get(*index) else {
            continue;
        };
        if work.status == Status::Unknown as u32 {
            PAIRED_UNKNOWN.fetch_add(1, Ordering::Relaxed);
        } else if work.status == cpu_work.status {
            PAIRED_AGREE.fetch_add(1, Ordering::Relaxed);
        } else {
            PAIRED_MISMATCH.fetch_add(1, Ordering::Relaxed);
        }
    }

    let Some(writer) = comparison_writer() else {
        return;
    };
    let Ok(mut writer) = writer.lock() else {
        return;
    };
    let context_literals = context
        .clauses
        .iter()
        .map(|clause| clause.literals.len() as u64)
        .sum::<u64>();
    let init_ns = ACTIVE_INIT_NS.load(Ordering::Relaxed);
    for (position, (((index, _), query), work)) in pending
        .iter()
        .zip(queries)
        .zip(hardware)
        .enumerate()
    {
        let Some(cpu_work) = cpu.get(*index) else {
            continue;
        };
        let preflight_work = preflight
            .and_then(|work| work.get(*index))
            .copied()
            .unwrap_or_default();
        let constraint_literals = query
            .constraints
            .iter()
            .map(|clause| clause.len() as u64)
            .sum::<u64>();
        let _ = writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            pass_id,
            batch_id,
            queries.len(),
            position,
            index,
            query.frame,
            cpu_work.status,
            cpu_work.elapsed_ns,
            cpu_work.decisions,
            cpu_work.conflicts,
            cpu_work.propagations,
            work.status,
            work.reason,
            work.decisions,
            work.conflicts,
            work.propagations,
            work.learnt_clauses,
            query.assumptions.len(),
            query.constraints.len(),
            constraint_literals,
            query.domain.len(),
            query_request_words(query).unwrap_or(0),
            context.n_var,
            context.clauses.len(),
            context_literals,
            batch_cpu_ns,
            batch_hw_ns,
            context_load_ns,
            init_ns,
            u8::from(preflight_work.selected),
            preflight_work.status,
            preflight_work.reason,
            preflight_work.elapsed_ns,
            preflight_work.clone_ns,
            preflight_work.solve_ns,
            preflight_work.decisions,
            preflight_work.conflicts,
            preflight_work.propagations,
        );
    }
}

fn flush_comparison_writer() {
    if let Some(writer) = comparison_writer()
        && let Ok(mut writer) = writer.lock()
    {
        let _ = writer.flush();
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
        && std::env::var_os("INDUCTOR_CDCL_PAIRED").is_none()
        && std::env::var_os("INDUCTOR_CDCL_SHADOW").is_none()
        && std::env::var_os("INDUCTOR_ACCEL").is_none()
}

/// Run the same independent propagation inquiries on a cloned GipSAT state
/// and on the FPGA, but return UNKNOWN to IC3 so the ordinary CPU path remains
/// authoritative. This is a measurement mode, not an alternative proof path.
pub fn paired_enabled() -> bool {
    std::env::var_os("INDUCTOR_CDCL_PAIRED").is_some()
        && std::env::var_os("INDUCTOR_CDCL_ACTIVE").is_none()
        && std::env::var_os("INDUCTOR_CDCL_SHADOW").is_none()
        && std::env::var_os("INDUCTOR_ACCEL").is_none()
}

/// Whether IC3 should prepare one speculative propagation batch. Active mode
/// may consume validated answers; paired mode deliberately never does.
pub fn propagation_batch_enabled() -> bool {
    active_enabled() || paired_enabled()
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
            .or_else(|_| std::env::var("INDUCTOR_CDCL_PAIRED"))
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
    if requests.is_empty() || !propagation_batch_enabled() {
        return output;
    }
    let paired = paired_enabled();
    let compare_cpu = paired || active_compare_cpu_enabled();
    let paired_preflight = paired
        .then(|| measure_paired_preflight(&requests))
        .flatten();
    let selected: Vec<bool> = requests
        .iter()
        .enumerate()
        .map(|(index, (_, query))| {
            !paired
                || paired_static_selected(query)
                    && paired_preflight
                        .as_ref()
                        .is_none_or(|work| work[index].selected)
        })
        .collect();
    let selected_count = selected.iter().filter(|selected| **selected).count();
    let pass_id = ACTIVE_PASSES.fetch_add(1, Ordering::Relaxed) + 1;
    ACTIVE_MAX_READY.fetch_max(requests.len() as u64, Ordering::Relaxed);
    ACTIVE_CANDIDATES.fetch_add(requests.len() as u64, Ordering::Relaxed);
    PAIRED_FILTERED.fetch_add(
        requests.len().saturating_sub(selected_count) as u64,
        Ordering::Relaxed,
    );
    if selected_count < active_min_batch_size() {
        ACTIVE_SKIPPED_PASSES.fetch_add(1, Ordering::Relaxed);
        ACTIVE_SKIPPED_SMALL_BATCH.fetch_add(selected_count as u64, Ordering::Relaxed);
        return output;
    }
    ACTIVE_OFFERED_PASSES.fetch_add(1, Ordering::Relaxed);
    ACTIVE_OFFERED.fetch_add(selected_count as u64, Ordering::Relaxed);
    let reference_cpu = compare_cpu.then(|| measure_reference_cpu(&requests, paired));
    if let Some(cpu) = reference_cpu.as_ref() {
        PAIRED_BASELINE_CPU_NS.fetch_add(
            cpu.iter().map(|work| work.elapsed_ns).sum(),
            Ordering::Relaxed,
        );
    }

    struct Group {
        context: ShadowContext,
        pending: Vec<(usize, IncrementalQuery)>,
    }
    let mut groups: Vec<Group> = Vec::new();
    let prefer_query_lemmas = !active_resident_lemmas();
    for (index, (solver, query)) in requests.into_iter().enumerate() {
        if !selected[index] {
            continue;
        }
        let (context, query, _) =
            prepare_batched_query(solver, query, prefer_query_lemmas);
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

    let state_wait_start = std::time::Instant::now();
    let state = active_state().lock();
    ACTIVE_STATE_WAIT_NS.fetch_add(
        state_wait_start
            .elapsed()
            .as_nanos()
            .min(u64::MAX as u128) as u64,
        Ordering::Relaxed,
    );
    let Ok(mut state) = state else {
        let n_pending: usize = groups.iter().map(|group| group.pending.len()).sum();
        ACTIVE_ERROR.fetch_add(n_pending as u64, Ordering::Relaxed);
        return output;
    };
    for mut group in groups {
        let mut context_load_ns = 0u64;
        if pair_scheduler_enabled() {
            schedule_query_pairs(&mut group.pending, |(_, query)| query);
        }
        let mut context_ready = state.loaded_context.as_ref() == Some(&group.context);

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
            let mut batch_ns = 0u64;
            let mut result = None;
            if !context_ready {
                let combined_start = std::time::Instant::now();
                let combined = state
                    .hardware
                    .as_mut()
                    .ok_or(HardwareError::Unavailable)
                    .and_then(|hardware| {
                        hardware.load_context_and_solve_batch(
                            group.context.n_var,
                            &group.context.clauses,
                            &queries,
                        )
                    });
                let combined_ns = combined_start
                    .elapsed()
                    .as_nanos()
                    .min(u64::MAX as u128) as u64;
                ACTIVE_COMBINED_NS.fetch_add(combined_ns, Ordering::Relaxed);
                match combined {
                    Ok(results) => {
                        ACTIVE_COMBINED_BATCHES.fetch_add(1, Ordering::Relaxed);
                        ACTIVE_CONTEXT_LOADS.fetch_add(1, Ordering::Relaxed);
                        context_ready = true;
                        state.loaded_context = Some(group.context.clone());
                        batch_ns = combined_ns;
                        result = Some(Ok(results));
                    }
                    Err(error) => {
                        eprintln!(
                            "inductor-cdcl: combined load/run failed: {error}"
                        );
                        ACTIVE_COMBINED_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                        ACTIVE_COMBINED_FALLBACK_NS.fetch_add(
                            combined_ns,
                            Ordering::Relaxed,
                        );
                        state.loaded_context = None;
                        let load_start = std::time::Instant::now();
                        let loaded = state
                            .hardware
                            .as_mut()
                            .ok_or(HardwareError::Unavailable)
                            .and_then(|hardware| {
                                hardware.load_context(
                                    group.context.n_var,
                                    &group.context.clauses,
                                )
                            });
                        context_load_ns = load_start
                            .elapsed()
                            .as_nanos()
                            .min(u64::MAX as u128) as u64;
                        ACTIVE_CONTEXT_LOAD_NS.fetch_add(context_load_ns, Ordering::Relaxed);
                        if let Err(error) = loaded {
                            eprintln!(
                                "inductor-cdcl: fallback context load failed: {error}"
                            );
                            ACTIVE_ERROR.fetch_add((end - start) as u64, Ordering::Relaxed);
                            start = end;
                            continue;
                        }
                        ACTIVE_CONTEXT_LOADS.fetch_add(1, Ordering::Relaxed);
                        context_ready = true;
                        state.loaded_context = Some(group.context.clone());
                    }
                }
            }
            let result = match result {
                Some(result) => result,
                None => {
                    let batch_start = std::time::Instant::now();
                    let result = state
                        .hardware
                        .as_mut()
                        .ok_or(HardwareError::Unavailable)
                        .and_then(|hardware| hardware.solve_batch(&queries));
                    batch_ns = batch_start
                        .elapsed()
                        .as_nanos()
                        .min(u64::MAX as u128) as u64;
                    result
                }
            };
            ACTIVE_BATCH_NS.fetch_add(batch_ns, Ordering::Relaxed);
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
                    if let Some(cpu) = reference_cpu.as_ref() {
                        record_comparison_batch(
                            pass_id,
                            &group.context,
                            &group.pending[start..end],
                            &queries,
                            cpu,
                            paired_preflight.as_deref(),
                            &hardware.last_batch_records,
                            std::mem::take(&mut context_load_ns),
                            batch_ns,
                        );
                    }
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
    if paired {
        output.fill(unknown);
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

pub fn note_active_unsat_core(
    accepted: bool,
    assumption_lits: usize,
    hardware_core_lits: usize,
    cpu_core_lits: usize,
    validation_ns: u64,
) {
    ACTIVE_UNSAT_VALIDATE_NS.fetch_add(validation_ns, Ordering::Relaxed);
    ACTIVE_UNSAT_ASSUMPTION_LITS.fetch_add(assumption_lits as u64, Ordering::Relaxed);
    ACTIVE_UNSAT_HW_CORE_LITS.fetch_add(hardware_core_lits as u64, Ordering::Relaxed);
    if accepted {
        ACTIVE_UNSAT_CORE_USED.fetch_add(1, Ordering::Relaxed);
        ACTIVE_UNSAT_CPU_CORE_LITS.fetch_add(cpu_core_lits as u64, Ordering::Relaxed);
    } else {
        ACTIVE_UNSAT_CORE_REJECTED.fetch_add(1, Ordering::Relaxed);
        ACTIVE_CPU_FALLBACK.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn note_active_cpu_fallback() {
    ACTIVE_CPU_FALLBACK.fetch_add(1, Ordering::Relaxed);
}

pub fn note_active_preflight_result(
    unsat: bool,
    accepted: bool,
    restore_ns: u64,
) {
    ACTIVE_PREFLIGHT_RESTORE_NS.fetch_add(restore_ns, Ordering::Relaxed);
    if accepted {
        if unsat {
            ACTIVE_PREFLIGHT_UNSAT_REUSED.fetch_add(1, Ordering::Relaxed);
        } else {
            ACTIVE_PREFLIGHT_SAT_REUSED.fetch_add(1, Ordering::Relaxed);
        }
    } else {
        ACTIVE_PREFLIGHT_REJECTED.fetch_add(1, Ordering::Relaxed);
        ACTIVE_CPU_FALLBACK.fetch_add(1, Ordering::Relaxed);
    }
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
    if paired_enabled() {
        flush_comparison_writer();
        let cpu_ns = PAIRED_CPU_NS.load(Ordering::Relaxed);
        let hw_ns = PAIRED_HW_NS.load(Ordering::Relaxed);
        let service_ratio = cpu_ns as f64 / hw_ns.max(1) as f64;
        eprintln!(
            "inductor-cdcl: paired selector frame >= {}, assumptions <= {}, passes {} (skipped {}, offered {}, max-ready {}), candidates {}, filtered {}, queries {}, batches {}, CPU/HW agree {}, mismatch {}, HW unknown {}, HW-faster batches {}, CPU-reference {:.3} ms, FPGA service {:.3} ms, service ratio {:.3}x, init {:.3} ms, context loads {} / {:.3} ms, combined ok/fallback {}/{} / {:.3} ms, errors {}, CSV {}",
            paired_min_frame(),
            paired_max_assumptions(),
            ACTIVE_PASSES.load(Ordering::Relaxed),
            ACTIVE_SKIPPED_PASSES.load(Ordering::Relaxed),
            ACTIVE_OFFERED_PASSES.load(Ordering::Relaxed),
            ACTIVE_MAX_READY.load(Ordering::Relaxed),
            ACTIVE_CANDIDATES.load(Ordering::Relaxed),
            PAIRED_FILTERED.load(Ordering::Relaxed),
            PAIRED_QUERIES.load(Ordering::Relaxed),
            PAIRED_BATCH_ID.load(Ordering::Relaxed),
            PAIRED_AGREE.load(Ordering::Relaxed),
            PAIRED_MISMATCH.load(Ordering::Relaxed),
            PAIRED_UNKNOWN.load(Ordering::Relaxed),
            PAIRED_HW_FASTER_BATCHES.load(Ordering::Relaxed),
            cpu_ns as f64 / 1_000_000.0,
            hw_ns as f64 / 1_000_000.0,
            service_ratio,
            ACTIVE_INIT_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_CONTEXT_LOADS.load(Ordering::Relaxed),
            ACTIVE_CONTEXT_LOAD_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_COMBINED_BATCHES.load(Ordering::Relaxed),
            ACTIVE_COMBINED_FALLBACKS.load(Ordering::Relaxed),
            ACTIVE_COMBINED_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_ERROR.load(Ordering::Relaxed),
            std::env::var("INDUCTOR_CDCL_PAIRED_CSV")
                .unwrap_or_else(|_| "disabled".to_string()),
        );
        if let Some(conflict_limit) = paired_preflight_conflicts() {
            let baseline_ns = PAIRED_BASELINE_CPU_NS.load(Ordering::Relaxed);
            let preflight_ns = PAIRED_PREFLIGHT_NS.load(Ordering::Relaxed);
            let hybrid_service_ns = preflight_ns.saturating_add(hw_ns);
            let hybrid_with_load_ns = hybrid_service_ns
                .saturating_add(ACTIVE_CONTEXT_LOAD_NS.load(Ordering::Relaxed))
                .saturating_add(ACTIVE_COMBINED_FALLBACK_NS.load(Ordering::Relaxed));
            eprintln!(
                "inductor-cdcl: paired preflight conflicts {}, candidates {}, conclusive {}, selected {}, preflight total/clone/solve {:.3}/{:.3}/{:.3} ms, all-candidate CPU baseline {:.3} ms, projected hybrid service {:.3} ms ({:.3}x), with context load {:.3} ms ({:.3}x)",
                conflict_limit,
                PAIRED_PREFLIGHT_CANDIDATES.load(Ordering::Relaxed),
                PAIRED_PREFLIGHT_CONCLUSIVE.load(Ordering::Relaxed),
                PAIRED_PREFLIGHT_SELECTED.load(Ordering::Relaxed),
                preflight_ns as f64 / 1_000_000.0,
                PAIRED_PREFLIGHT_CLONE_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
                PAIRED_PREFLIGHT_SOLVE_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
                baseline_ns as f64 / 1_000_000.0,
                hybrid_service_ns as f64 / 1_000_000.0,
                baseline_ns as f64 / hybrid_service_ns.max(1) as f64,
                hybrid_with_load_ns as f64 / 1_000_000.0,
                baseline_ns as f64 / hybrid_with_load_ns.max(1) as f64,
            );
        }
    } else if active_enabled() {
        if active_compare_cpu_enabled() {
            flush_comparison_writer();
        }
        eprintln!(
            "inductor-cdcl: active pair-scheduler {}, passes {} (skipped {}, offered {}, max-ready {}), candidates {}, skipped-small-batch {}, offered {}, batches {}, context loads {}, combined ok/fallback {}/{}, hw SAT {}, hw UNSAT {}, unknown {}, errors {}, hw work decisions/conflicts/propagations/learnts {}/{}/{}/{}, validated SAT used {}, rejected SAT {}, validated UNSAT cores used {}, rejected {}, UNSAT lits assumptions/hw-core/cpu-core {}/{}/{}, CPU fallbacks executed {}, init/wait {:.3}/{:.3} ms, load {:.3} ms, combined attempts {:.3} ms, batches {:.3} ms, SAT-validate {:.3} ms, UNSAT-validate {:.3} ms",
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
            ACTIVE_COMBINED_BATCHES.load(Ordering::Relaxed),
            ACTIVE_COMBINED_FALLBACKS.load(Ordering::Relaxed),
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
            ACTIVE_UNSAT_CORE_USED.load(Ordering::Relaxed),
            ACTIVE_UNSAT_CORE_REJECTED.load(Ordering::Relaxed),
            ACTIVE_UNSAT_ASSUMPTION_LITS.load(Ordering::Relaxed),
            ACTIVE_UNSAT_HW_CORE_LITS.load(Ordering::Relaxed),
            ACTIVE_UNSAT_CPU_CORE_LITS.load(Ordering::Relaxed),
            ACTIVE_CPU_FALLBACK.load(Ordering::Relaxed),
            ACTIVE_INIT_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_STATE_WAIT_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_CONTEXT_LOAD_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_COMBINED_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_BATCH_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_VALIDATE_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_UNSAT_VALIDATE_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
        );
        if let Some(conflict_limit) = active_preflight_conflicts() {
            let candidates = ACTIVE_PREFLIGHT_CANDIDATES.load(Ordering::Relaxed);
            eprintln!(
                "inductor-cdcl: active preflight conflicts {}, max assumptions {}, static-filtered {}, candidates {}, conclusive {}, selected {}, reused SAT/UNSAT {}, {}, rejected {}, service {:.3} ms, restore {:.3} ms, mean conflicts/query {:.2}",
                conflict_limit,
                active_preflight_max_assumptions(),
                ACTIVE_PREFLIGHT_STATIC_FILTERED.load(Ordering::Relaxed),
                candidates,
                ACTIVE_PREFLIGHT_CONCLUSIVE.load(Ordering::Relaxed),
                ACTIVE_PREFLIGHT_SELECTED.load(Ordering::Relaxed),
                ACTIVE_PREFLIGHT_SAT_REUSED.load(Ordering::Relaxed),
                ACTIVE_PREFLIGHT_UNSAT_REUSED.load(Ordering::Relaxed),
                ACTIVE_PREFLIGHT_REJECTED.load(Ordering::Relaxed),
                ACTIVE_PREFLIGHT_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
                ACTIVE_PREFLIGHT_RESTORE_NS.load(Ordering::Relaxed) as f64
                    / 1_000_000.0,
                ACTIVE_PREFLIGHT_CONFLICTS.load(Ordering::Relaxed) as f64
                    / candidates.max(1) as f64,
            );
        }
        let sampled = ACTIVE_SAMPLE_QUERIES.load(Ordering::Relaxed);
        let undersized = ACTIVE_SAMPLE_UNDERSIZED_REJECTED.load(Ordering::Relaxed);
        if sampled != 0 || undersized != 0 {
            eprintln!(
                "inductor-cdcl: active CPU sample queries {}, mean total/clone/solve {:.3}/{:.3}/{:.3} us, solve threshold {:.3} us, FPGA/CPU batches {}/{}, FPGA retained {}, CPU rejected {}, undersized rejected {}",
                sampled,
                ACTIVE_SAMPLE_NS.load(Ordering::Relaxed) as f64
                    / sampled.max(1) as f64
                    / 1_000.0,
                ACTIVE_SAMPLE_CLONE_NS.load(Ordering::Relaxed) as f64
                    / sampled.max(1) as f64
                    / 1_000.0,
                ACTIVE_SAMPLE_SOLVE_NS.load(Ordering::Relaxed) as f64
                    / sampled.max(1) as f64
                    / 1_000.0,
                active_sample_min_cpu_ns() as f64 / 1_000.0,
                ACTIVE_SAMPLE_FPGA_BATCHES.load(Ordering::Relaxed),
                ACTIVE_SAMPLE_CPU_BATCHES.load(Ordering::Relaxed),
                ACTIVE_SAMPLE_FPGA_RETAINED.load(Ordering::Relaxed),
                ACTIVE_SAMPLE_CPU_REJECTED.load(Ordering::Relaxed),
                undersized,
            );
        }
        if active_compare_cpu_enabled() {
            let cpu_ns = PAIRED_CPU_NS.load(Ordering::Relaxed);
            let hw_ns = PAIRED_HW_NS.load(Ordering::Relaxed);
            eprintln!(
                "inductor-cdcl: active CPU comparison queries {}, batches {}, CPU/HW status agree {}, mismatch {}, HW unknown {}, HW-faster batches {}, CPU-reference {:.3} ms, FPGA service {:.3} ms, service ratio {:.3}x, with context load {:.3}x, CSV {}",
                PAIRED_QUERIES.load(Ordering::Relaxed),
                PAIRED_BATCH_ID.load(Ordering::Relaxed),
                PAIRED_AGREE.load(Ordering::Relaxed),
                PAIRED_MISMATCH.load(Ordering::Relaxed),
                PAIRED_UNKNOWN.load(Ordering::Relaxed),
                PAIRED_HW_FASTER_BATCHES.load(Ordering::Relaxed),
                cpu_ns as f64 / 1_000_000.0,
                hw_ns as f64 / 1_000_000.0,
                cpu_ns as f64 / hw_ns.max(1) as f64,
                cpu_ns as f64
                    / hw_ns
                        .saturating_add(ACTIVE_CONTEXT_LOAD_NS.load(Ordering::Relaxed))
                        .saturating_add(ACTIVE_COMBINED_FALLBACK_NS.load(Ordering::Relaxed))
                        .max(1) as f64,
                std::env::var("INDUCTOR_CDCL_ACTIVE_COMPARE_CSV")
                    .unwrap_or_else(|_| "disabled".to_string()),
            );
        }
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
    fn cpu_sample_requires_cost_and_a_remaining_hardware_batch() {
        assert_eq!(representative_sample_positions(10, 1), vec![5]);
        assert_eq!(representative_sample_positions(10, 3), vec![1, 5, 8]);
        assert_eq!(representative_sample_positions(3, 8), vec![0, 1, 2]);
        assert!(sample_keeps_fpga(
            &[250_000, 300_000, 900_000], 16, 8, 200_000, true
        ));
        assert!(!sample_keeps_fpga(
            &[50_000, 100_000, 900_000], 16, 8, 200_000, true
        ));
        // With an even sample the lower median prevents one expensive half
        // from routing a frame whose other half is cheap.
        assert!(!sample_keeps_fpga(
            &[150_000, 600_000], 16, 8, 200_000, true
        ));
        assert!(!sample_keeps_fpga(
            &[250_000, 300_000], 7, 8, 200_000, true
        ));
        assert!(!sample_keeps_fpga(&[], 16, 8, 200_000, true));
        assert!(!sample_keeps_fpga(
            &[250_000, 300_000], 16, 8, 200_000, false
        ));
    }

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
    fn combined_request_has_exact_context_and_batch_boundaries() {
        let a = Lit::new(Var::from(0), true);
        let clauses = [ResidentClause::new(0, 2, LitVec::from([a]))];
        let mut query = IncrementalQuery::new(2, LitVec::from([a]));
        query.domain = vec![Var::from(0)];
        let queries = [query];
        let context = pack_clauses(&[1, 1], 1, &clauses).unwrap();
        let (batch, response_capacity) = pack_batch_request(&queries, false).unwrap();
        let (profile_batch, profile_capacity) = pack_batch_request(&queries, true).unwrap();
        let (combined, combined_capacity) =
            pack_load_context_and_batch_request(1, &clauses, &queries, false).unwrap();

        assert_eq!(profile_batch[4 + 2] & WANT_STAGE_PROFILE, WANT_STAGE_PROFILE);
        assert_eq!(profile_capacity, response_capacity + STAGE_PROFILE_WORDS);
        assert_eq!(combined[0] as usize, context.len());
        assert_eq!(&combined[1..1 + context.len()], context.as_slice());
        assert_eq!(&combined[1 + context.len()..], batch.as_slice());
        assert_eq!(combined_capacity, response_capacity);
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
                    profile_counters: [0; STAGE_PROFILE_COUNTERS],
                },
                HardwareWork {
                    status: 2,
                    reason: 0,
                    decisions: 11,
                    conflicts: 12,
                    propagations: 13,
                    learnt_clauses: 14,
                    profile_counters: [0; STAGE_PROFILE_COUNTERS],
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
    fn stage_profile_trailer_is_validated_and_stripped() {
        let mut words = vec![
            super::super::cdcl::ABI_VERSION,
            1,
            (RESPONSE_HEADER_WORDS + 1 + STAGE_PROFILE_WORDS) as u32,
            0,
            Status::Sat as u32,
            0,
            1,
            0,
            2,
            3,
            4,
            5,
            0,
            42,
            STAGE_PROFILE_MAGIC,
            STAGE_PROFILE_VERSION,
            STAGE_PROFILE_COUNTERS as u32,
        ];
        for stage in 0..STAGE_PROFILE_COUNTERS {
            let entries = (stage as u64 + 1) << 32 | (100 + stage as u64);
            words.push(entries as u32);
            words.push((entries >> 32) as u32);
        }
        let (records, semantic) = decode_profiled_batch_wire(&words, 1).unwrap();
        assert_eq!(
            records[0].profile_counters[PROFILE_SETUP],
            (1u64 << 32) | 100
        );
        assert_eq!(
            records[0].profile_counters[PROFILE_CLEANUP],
            (PROFILE_CLEANUP as u64 + 1) << 32 | 108,
        );
        assert_eq!(
            records[0].profile_counters[PROFILE_LEARNT_LITERALS],
            (STAGE_PROFILE_COUNTERS as u64) << 32 | 116,
        );
        assert_eq!(semantic[2], (RESPONSE_HEADER_WORDS + 1) as u32);
        assert_eq!(semantic.len(), 4 + RESPONSE_HEADER_WORDS + 1);
        words[14] ^= 1;
        assert_eq!(decode_profiled_batch_wire(&words, 1), None);
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
    fn oversized_query_lemma_expansion_uses_resident_frame_context() {
        let mut dc = DagCnf::new();
        let a = dc.new_var().lit();
        let b = dc.new_var().lit();
        let c = dc.new_var().lit();
        let extra: Vec<_> = (0..query_lemma_word_limit() / 5 + 1)
            .map(|_| dc.new_var().lit())
            .collect();
        let mut solver = DagCnfSolver::new(&dc);
        solver.accel_level = 5;
        for d in extra {
            solver.add_clause(&[a, b, c, d]);
        }

        let query = IncrementalQuery::new(5, LitVec::new());
        let (context, unchanged, used) =
            prepare_batched_query(&solver, query.clone(), true);
        assert!(!used);
        assert_eq!(unchanged.frame, query.frame);
        assert_eq!(unchanged.assumptions, query.assumptions);
        assert_eq!(unchanged.constraints, query.constraints);
        assert_eq!(unchanged.domain, query.domain);
        assert!(context
            .clauses
            .iter()
            .all(|clause| clause.lo == 5 && clause.hi == 5));
        assert!(context.clauses.len() > query_lemma_word_limit() / 5);
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
