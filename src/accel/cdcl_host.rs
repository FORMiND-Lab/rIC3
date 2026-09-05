//! XRT-backed implementation of the incremental CDCL semantic boundary.
//!
//! The C++ bridge owns one persistent kernel context and reusable DMA buffers.
//! Transport or device failures become `Unknown(BackendError)` through the
//! `IncrementalCdcl` implementation; they are never interpreted as SAT/UNSAT.

use super::cdcl::{
    ABI_VERSION, BANK_ALIGNED_DOMAIN, BLOCK_ROOT_BATCH_OFFSET, BLOCK_SEMANTIC_EVENT_RESET_EPOCH,
    BLOCK_SEMANTIC_RESET, BatchHeader, BlockFullRootEvent, BlockFullRootResponse,
    BlockFullRootStatus, BlockRootExecutionStatus, BlockRootResponse, BlockSemanticCommand,
    BlockSemanticCommandResponse, MIC_BATCH_HEADER_WORDS, MIC_BATCH_RESPONSE_HEADER_WORDS,
    MIC_MODEL_SHRINK, MIC_PROTECT_INDEX, MIC_PROTECTED_INDEX_SHIFT, MIC_RESPONSE_HEADER_WORDS,
    MicHeader, MicResponseHeader, PROFILE_ANALYZE, PROFILE_ANALYZED_LITERALS, PROFILE_BACKTRACK,
    PROFILE_CLEANUP, PROFILE_DECIDE, PROFILE_EMIT, PROFILE_EVALUATED_LITERALS, PROFILE_LEARN,
    PROFILE_LEARNT_LITERALS, PROFILE_OCCURRENCE_PAIRS, PROFILE_OCCURRENCE_ROUNDS,
    PROFILE_OCCURRENCE_UPDATES, PROFILE_PARTIAL_OCCURRENCE_SCANS, PROFILE_PROPAGATE, PROFILE_ROOT,
    PROFILE_SETUP, PROFILE_UNDO_ASSIGNMENTS, PROFILE_UNDO_OCCURRENCES, PROFILE_UNIT_CANDIDATES,
    RESPONSE_HEADER_WORDS, STAGE_PROFILE_COUNTERS, STAGE_PROFILE_MAGIC,
    STAGE_PROFILE_STAGE_COUNTERS, STAGE_PROFILE_VERSION, STAGE_PROFILE_WORDS, Status,
    UnknownReason, WANT_STAGE_PROFILE, block_full_root_required_response_capacity,
    decode_block_full_root_response, decode_block_root_response,
    decode_block_semantic_batch_response, pack_block_full_root_continuation,
    pack_block_full_root_request, pack_block_root_request, pack_block_semantic_batch,
};
#[cfg(has_cdcl_accel)]
use crate::gipsat::decode_batch_results;
use crate::gipsat::{
    BatchDecodeError, DagCnfSolver, IncrementalCdcl, IncrementalQuery, IncrementalResult,
    QueryBudget, bank_aligned_domain_enabled, encoded_domain_words, pack_batch,
    solve_on_cpu_after_hardware_unknown,
};
use logicrs::{Lit, LitVec, Var};
use std::collections::{HashMap, HashSet};
#[cfg(has_cdcl_accel)]
use std::ffi::CString;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[cfg(has_cdcl_accel)]
unsafe extern "C" {
    fn ind_cdcl_open(path: *const std::os::raw::c_char) -> i32;
    fn ind_cdcl_open_with_device(path: *const std::os::raw::c_char, device_index: i32) -> i32;
    fn ind_cdcl_connect(path: *const std::os::raw::c_char) -> i32;
    fn ind_cdcl_load_context(request: *const u32, request_words: u32) -> i32;
    fn ind_cdcl_add_frame_clauses(request: *const u32, request_words: u32) -> i32;
    fn ind_cdcl_append_and_solve_mic_chain(
        request: *const u32,
        request_words: u32,
        response: *mut u32,
        response_capacity_words: u32,
        out_response_words: *mut u32,
    ) -> i32;
    fn ind_cdcl_materialize_frame(frame: u32) -> i32;
    fn ind_cdcl_solve_batch(
        request: *const u32,
        request_words: u32,
        response: *mut u32,
        response_capacity_words: u32,
        out_response_words: *mut u32,
    ) -> i32;
    fn ind_cdcl_solve_arena_batch(
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
    fn ind_cdcl_solve_mic_chain(
        request: *const u32,
        request_words: u32,
        response: *mut u32,
        response_capacity_words: u32,
        out_response_words: *mut u32,
    ) -> i32;
    fn ind_cdcl_solve_arena_mic(
        request: *const u32,
        request_words: u32,
        response: *mut u32,
        response_capacity_words: u32,
        out_response_words: *mut u32,
    ) -> i32;
    fn ind_cdcl_solve_mic_chains(
        request: *const u32,
        request_words: u32,
        response: *mut u32,
        response_capacity_words: u32,
        out_response_words: *mut u32,
    ) -> i32;
    fn ind_cdcl_run_block_semantic_batch(
        request: *const u32,
        request_words: u32,
        response: *mut u32,
        response_capacity_words: u32,
        out_response_words: *mut u32,
    ) -> i32;
    fn ind_cdcl_run_block_root(
        request: *const u32,
        request_words: u32,
        response: *mut u32,
        response_capacity_words: u32,
        out_response_words: *mut u32,
    ) -> i32;
    fn ind_cdcl_run_block_full_root(
        request: *const u32,
        request_words: u32,
        response: *mut u32,
        response_capacity_words: u32,
        out_response_words: *mut u32,
    ) -> i32;
    fn ind_cdcl_total_kernel_ns() -> u64;
}

#[cfg(has_cdcl_accel)]
fn direct_kernel_ns() -> u64 {
    // The bridge serializes commands and owns this process's XRT context, so
    // the cumulative value is a coherent kernel-busy counter at report time.
    unsafe { ind_cdcl_total_kernel_ns() }
}

#[cfg(not(has_cdcl_accel))]
fn direct_kernel_ns() -> u64 {
    0
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

#[derive(Default)]
struct FrameRangeRegistry {
    n_var: u32,
    transition: Vec<LitVec>,
    clauses: Vec<ResidentClause>,
}

static FRAME_RANGE_REGISTRY: std::sync::OnceLock<std::sync::Mutex<FrameRangeRegistry>> =
    std::sync::OnceLock::new();

#[derive(Default)]
struct BlockRootRangeRegistry {
    n_var: u32,
    transition: Vec<LitVec>,
    clauses: Vec<ResidentClause>,
    seen: HashSet<(u32, Vec<u32>)>,
}

static BLOCK_ROOT_RANGE_REGISTRY: std::sync::OnceLock<std::sync::Mutex<BlockRootRangeRegistry>> =
    std::sync::OnceLock::new();

fn block_root_range_registry() -> &'static std::sync::Mutex<BlockRootRangeRegistry> {
    BLOCK_ROOT_RANGE_REGISTRY
        .get_or_init(|| std::sync::Mutex::new(BlockRootRangeRegistry::default()))
}

fn frame_range_registry() -> &'static std::sync::Mutex<FrameRangeRegistry> {
    FRAME_RANGE_REGISTRY.get_or_init(|| std::sync::Mutex::new(FrameRangeRegistry::default()))
}

/// Start one IC3 run's append-only, frame-ranged resident formula. The
/// transition relation is permanent; subsequent calls record the exact frame
/// intervals into which IC3 inserts each lemma.
pub fn reset_frame_resident_context(solver: &DagCnfSolver) {
    let (n_var, _, transition, _) = solver.incremental_resident_partition();
    if let Ok(mut registry) = block_root_range_registry().lock() {
        registry.n_var = n_var;
        registry.transition = transition.clone();
        registry.clauses.clear();
        registry.seen.clear();
    }
    if !active_frame_ranges() {
        return;
    }
    if let Ok(mut registry) = frame_range_registry().lock() {
        registry.n_var = n_var;
        registry.transition = transition;
        registry.clauses.clear();
    }
}

/// Record only interval portions not already covered by an identical clause.
/// The log stays append-only so the card normally advances with
/// ADD_FRAME_CLAUSES instead of replacing the whole resident context.
pub fn register_frame_resident_clause(literals: &[Lit], lo: u32, hi: u32) {
    if !active_frame_ranges() || lo > hi || literals.is_empty() {
        return;
    }
    let Ok(mut registry) = frame_range_registry().lock() else {
        return;
    };
    let literals = LitVec::from(literals);
    let mut covered: Vec<_> = registry
        .clauses
        .iter()
        .filter(|clause| clause.literals == literals)
        .map(|clause| (clause.lo, clause.hi))
        .collect();
    covered.sort_unstable();

    let mut cursor = lo;
    for (begin, end) in covered {
        if end < cursor || begin > hi {
            continue;
        }
        if begin > cursor {
            registry.clauses.push(ResidentClause::new(
                cursor,
                hi.min(begin - 1),
                literals.clone(),
            ));
        }
        if end == u32::MAX {
            return;
        }
        cursor = cursor.max(end + 1);
        if cursor > hi {
            return;
        }
    }
    registry
        .clauses
        .push(ResidentClause::new(cursor, hi, literals));
}

fn canonical_clause_set<'a>(clauses: impl Iterator<Item = &'a LitVec>) -> Vec<Vec<u32>> {
    let mut set: Vec<Vec<u32>> = clauses
        .map(|clause| {
            let mut raw: Vec<u32> = clause.iter().map(|lit| u32::from(*lit)).collect();
            raw.sort_unstable();
            raw.dedup();
            raw
        })
        .collect();
    set.sort_unstable();
    set.dedup();
    set
}

fn ranged_snapshot_matches(
    clauses: &[ResidentClause],
    frame: u32,
    exact_lemmas: &[LitVec],
) -> bool {
    canonical_clause_set(
        clauses
            .iter()
            .filter(|clause| clause.lo <= frame && frame <= clause.hi)
            .map(|clause| &clause.literals),
    ) == canonical_clause_set(exact_lemmas.iter())
}

fn frame_ranged_context(
    n_var: u32,
    transition: &[LitVec],
    frame: u32,
    exact_lemmas: &[LitVec],
) -> Option<ShadowContext> {
    if !active_frame_ranges() {
        return None;
    }
    let registry = frame_range_registry().lock().ok()?;
    if registry.n_var != n_var || registry.transition != transition {
        return None;
    }
    // The append-only range registry is a transport optimization, not the
    // semantic authority. Clause subsumption, solver cloning and promotion to
    // infinity can make its event log diverge from one frame solver's exact
    // resident lemma snapshot. Only use the compact ranged representation when
    // the active formula is set-equivalent; otherwise the caller falls back to
    // an exact-frame snapshot. This is context maintenance, not result checking.
    if !ranged_snapshot_matches(&registry.clauses, frame, exact_lemmas) {
        FRAME_RANGE_SNAPSHOT_MISMATCH.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    // Preserve registration order exactly. IC3 revisits lower frames, so `hi`
    // is not globally monotonic; sorting would insert new clauses into the
    // loaded prefix and force a full reload. The kernel's frame buckets skip
    // expired ranges without changing this physical append order.
    let clauses = transition
        .iter()
        .cloned()
        .map(|literals| ResidentClause::new(0, u32::MAX, literals))
        .chain(registry.clauses.iter().cloned())
        .collect();
    Some(ShadowContext {
        n_var,
        clauses,
        scope: ShadowContextScope::FrameRanged,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HardwareError {
    Unavailable,
    InvalidPath,
    InvalidContext,
    Capacity,
    Timeout,
    Open(i32),
    Command(i32),
    Decode(BatchDecodeError),
    BlockSemantic {
        error: u32,
        completed: u32,
        command: u32,
        command_status: u32,
        obligation_count: u32,
        lemma_count: u32,
    },
    InvalidResponse,
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
        if clause.lo > clause.hi
            || clause.literals.is_empty()
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MicChainResult {
    pub cube: LitVec,
    pub trials: u32,
    /// Physical engine rounds reported by command 5. This is smaller than
    /// `trials` when adjacent speculative SAT answers were consumable.
    pub physical_rounds: u32,
    pub complete: bool,
    pub client_ns: u64,
    pub context_reused: bool,
    pub reason: UnknownReason,
    pub decisions: u64,
    pub conflicts: u64,
    pub propagations: u64,
    pub learnt_clauses: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct MicChainQuery<'a> {
    pub frame: u32,
    pub cube: &'a [(Lit, Lit)],
    pub constraints: &'a [LitVec],
    pub protected_index: usize,
    pub decision_budget: u32,
    pub conflict_budget: u32,
    pub max_trials: u32,
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
        let payload_words = usize::try_from(header[2])
            .ok()?
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
        let payload_words = usize::try_from(header[2])
            .ok()?
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
            for (sum, value) in total.profile_counters.iter_mut().zip(work.profile_counters) {
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
                .checked_add(encoded_domain_words(&query.domain).max(query.assumptions.len()))?
                .checked_add(if want_stage_profile {
                    STAGE_PROFILE_WORDS
                } else {
                    0
                })?;
            total.checked_add(record)
        })
        .ok_or(HardwareError::Capacity)?;
    let result_words_u32 = u32::try_from(result_words).map_err(|_| HardwareError::Capacity)?;
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

const ARENA_VIEW_PREFIX_WORDS: usize = 5;
const ARENA_VIEW_REUSE: u32 = 0;
const ARENA_VIEW_TOGGLE: u32 = 1;
const ARENA_VIEW_BITMAP: u32 = 2;
const QUALIFIED_ARENA_MAX_VARS: u32 = 32_768;
// Admission bound of the qualified full VCK5000 image. Using the complete
// bitmap here is conservative: most commands send sparse toggles, but the
// host planner must not form a batch that only fits when every view update is
// unusually small.
const QUALIFIED_ARENA_MAX_CLAUSES: usize = 75_000;

/// Insert one independently planned physical-lane view immediately before
/// each complete query record. The ordinary batch packer remains the single
/// source of truth for query flags and response capacity.
fn pack_arena_batch_request(
    queries: &[IncrementalQuery],
    views: &[ArenaViewUpdate],
    want_stage_profile: bool,
) -> Result<(Vec<u32>, usize), HardwareError> {
    if queries.is_empty() || queries.len() != views.len() {
        return Err(HardwareError::InvalidContext);
    }
    let (plain, response_capacity) = pack_batch_request(queries, want_stage_profile)?;
    let mut payload_words = 0usize;
    for (query, view) in queries.iter().zip(views) {
        let query_words = query_request_words(query).ok_or(HardwareError::Capacity)?;
        payload_words = payload_words
            .checked_add(view.words.len())
            .and_then(|words| words.checked_add(query_words))
            .ok_or(HardwareError::Capacity)?;
    }
    let total_words = 4usize
        .checked_add(payload_words)
        .ok_or(HardwareError::Capacity)?;
    if total_words > KERNEL_MAX_REQUEST_WORDS || total_words > u32::MAX as usize {
        return Err(HardwareError::Capacity);
    }
    let mut request = Vec::with_capacity(total_words);
    request.extend([plain[0], plain[1], payload_words as u32, plain[3]]);
    let mut offset = 4usize;
    for (query, view) in queries.iter().zip(views) {
        let query_words = query_request_words(query).ok_or(HardwareError::Capacity)?;
        request.extend_from_slice(&view.words);
        request.extend_from_slice(
            plain
                .get(offset..offset + query_words)
                .ok_or(HardwareError::InvalidContext)?,
        );
        offset += query_words;
    }
    if offset != plain.len() {
        return Err(HardwareError::InvalidContext);
    }
    Ok((request, response_capacity))
}

fn pack_arena_mic_request(view: &ArenaViewUpdate, mic: &[u32]) -> Result<Vec<u32>, HardwareError> {
    let total_words = view
        .words
        .len()
        .checked_add(mic.len())
        .ok_or(HardwareError::Capacity)?;
    if total_words > KERNEL_MAX_REQUEST_WORDS || total_words > u32::MAX as usize {
        return Err(HardwareError::Capacity);
    }
    let mut request = Vec::with_capacity(total_words);
    request.extend_from_slice(&view.words);
    request.extend_from_slice(mic);
    Ok(request)
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

fn pack_mic_chain_request(
    n_var: u32,
    frame: u32,
    cube: &[(Lit, Lit)],
    constraints: &[LitVec],
    protected_index: usize,
    decision_budget: u32,
    conflict_budget: u32,
    max_trials: u32,
) -> Result<Vec<u32>, HardwareError> {
    if n_var == 0
        || cube.len() < 2
        || cube.len() > u32::MAX as usize
        || protected_index >= cube.len()
        || protected_index > u16::MAX as usize
    {
        return Err(HardwareError::InvalidContext);
    }
    let mut constraint_words = 0usize;
    for clause in constraints {
        if clause.is_empty()
            || clause.len() > u32::MAX as usize
            || clause.iter().any(|lit| u32::from(*lit) >> 1 >= n_var)
        {
            return Err(HardwareError::InvalidContext);
        }
        constraint_words = constraint_words
            .checked_add(1 + clause.len())
            .ok_or(HardwareError::Capacity)?;
    }
    if cube.iter().any(|(current, next)| {
        (u32::from(*current) >> 1) >= n_var || (u32::from(*next) >> 1) >= n_var
    }) {
        return Err(HardwareError::InvalidContext);
    }
    // Match GipSAT's MIC-local decision domain exactly: next-state cube
    // variables first, then current-state cube variables, with stable
    // first-occurrence deduplication.  Sending 0..n_var here lets the FPGA
    // branch on hundreds of unrelated transition variables and turns the
    // otherwise short SAT inquiries into long, low-conflict searches.
    let mut domain = Vec::with_capacity(2 * cube.len());
    let mut seen = vec![false; n_var as usize];
    for literal in cube
        .iter()
        .map(|(_, next)| *next)
        .chain(cube.iter().map(|(current, _)| *current))
    {
        let var = (u32::from(literal) >> 1) as usize;
        if !seen[var] {
            seen[var] = true;
            domain.push(var as u32);
        }
    }
    let bank_aligned = bank_aligned_domain_enabled();
    let encoded_domain = encode_mic_domain(&domain, bank_aligned)?;
    let payload_words = constraint_words
        .checked_add(encoded_domain.len())
        .and_then(|words| words.checked_add(2 * cube.len()))
        .ok_or(HardwareError::Capacity)?;
    let total_words = super::cdcl::MIC_HEADER_WORDS
        .checked_add(payload_words)
        .ok_or(HardwareError::Capacity)?;
    if total_words > KERNEL_MAX_REQUEST_WORDS || total_words > u32::MAX as usize {
        return Err(HardwareError::Capacity);
    }
    let header = MicHeader {
        version: ABI_VERSION,
        frame,
        flags: MIC_PROTECT_INDEX
            | ((protected_index as u32) << MIC_PROTECTED_INDEX_SHIFT)
            | if bank_aligned { BANK_ALIGNED_DOMAIN } else { 0 }
            | if mic_chain_model_shrink() {
                MIC_MODEL_SHRINK
            } else {
                0
            },
        n_cube: cube.len() as u32,
        n_constraint_words: constraint_words as u32,
        n_domain: encoded_domain.len() as u32,
        decision_budget,
        conflict_budget,
        max_trials,
    };
    let mut request = Vec::with_capacity(total_words);
    request.extend(header.as_words());
    for clause in constraints {
        request.push(clause.len() as u32);
        request.extend(clause.iter().map(|lit| u32::from(*lit)));
    }
    request.extend(encoded_domain);
    for &(current, next) in cube {
        request.push(current.into());
        request.push(next.into());
    }
    debug_assert!(header.valid_for(&request[super::cdcl::MIC_HEADER_WORDS..]));
    Ok(request)
}

fn encode_mic_domain(domain: &[u32], bank_aligned: bool) -> Result<Vec<u32>, HardwareError> {
    if !bank_aligned {
        return Ok(domain.to_vec());
    }
    if domain.len() > 32768 || domain.iter().any(|&variable| variable >= 32768) {
        return Err(HardwareError::Capacity);
    }
    let mut banks: [Vec<(u16, u16)>; 4] = std::array::from_fn(|_| Vec::new());
    for (rank, &variable) in domain.iter().enumerate() {
        banks[(variable & 3) as usize].push((rank as u16, variable as u16));
    }
    let lines = banks.iter().map(Vec::len).max().unwrap_or(0);
    let mut encoded = Vec::with_capacity(4 * lines);
    for line in 0..lines {
        for bank in &banks {
            encoded.push(bank.get(line).map_or(0, |&(rank, variable)| {
                0x8000_0000 | (u32::from(rank) << 16) | u32::from(variable)
            }));
        }
    }
    Ok(encoded)
}

fn pack_mic_chains_request(
    n_var: u32,
    chains: &[MicChainQuery<'_>],
) -> Result<(Vec<u32>, usize), HardwareError> {
    if chains.is_empty() || chains.len() > 4 {
        return Err(HardwareError::InvalidContext);
    }
    let mut records = Vec::with_capacity(chains.len());
    let mut request_words = 0usize;
    let mut result_words = 0usize;
    for chain in chains {
        let record = pack_mic_chain_request(
            n_var,
            chain.frame,
            chain.cube,
            chain.constraints,
            chain.protected_index,
            chain.decision_budget,
            chain.conflict_budget,
            chain.max_trials,
        )?;
        request_words = request_words
            .checked_add(record.len())
            .ok_or(HardwareError::Capacity)?;
        result_words = result_words
            .checked_add(MIC_RESPONSE_HEADER_WORDS)
            .and_then(|words| words.checked_add(chain.cube.len()))
            .ok_or(HardwareError::Capacity)?;
        records.push(record);
    }
    let total_request = MIC_BATCH_HEADER_WORDS
        .checked_add(request_words)
        .ok_or(HardwareError::Capacity)?;
    let response_capacity = MIC_BATCH_RESPONSE_HEADER_WORDS
        .checked_add(result_words)
        .ok_or(HardwareError::Capacity)?;
    if total_request > KERNEL_MAX_REQUEST_WORDS
        || total_request > u32::MAX as usize
        || result_words > u32::MAX as usize
    {
        return Err(HardwareError::Capacity);
    }
    let header = BatchHeader {
        version: ABI_VERSION,
        n_queries: chains.len() as u32,
        n_request_words: request_words as u32,
        result_capacity_words: result_words as u32,
    };
    let mut request = Vec::with_capacity(total_request);
    request.extend([
        header.version,
        header.n_queries,
        header.n_request_words,
        header.result_capacity_words,
    ]);
    for record in records {
        request.extend(record);
    }
    Ok((request, response_capacity))
}

fn decode_mic_chain_record(
    cube: &[(Lit, Lit)],
    response: &[u32],
) -> Result<(MicChainResult, usize), HardwareError> {
    let header = response
        .get(..MIC_RESPONSE_HEADER_WORDS)
        .and_then(MicResponseHeader::from_words)
        .ok_or(HardwareError::InvalidResponse)?;
    let n_output = usize::try_from(header.n_output).map_err(|_| HardwareError::InvalidResponse)?;
    let record_words = MIC_RESPONSE_HEADER_WORDS
        .checked_add(n_output)
        .ok_or(HardwareError::InvalidResponse)?;
    if header.version != ABI_VERSION
        || usize::try_from(header.n_input).ok() != Some(cube.len())
        || n_output == 0
        || n_output > cube.len()
        || header.trials > header.n_input
        || (header.trials == 0) != (header.physical_rounds == 0)
        || header.physical_rounds.saturating_mul(4) < header.trials
        || header.complete > 1
        || header.error != 0
        || record_words > response.len()
    {
        return Err(HardwareError::InvalidResponse);
    }
    let reason = UnknownReason::from_word(header.reason).ok_or(HardwareError::InvalidResponse)?;
    if header.complete == 1 && reason != UnknownReason::None {
        return Err(HardwareError::InvalidResponse);
    }
    let returned = &response[MIC_RESPONSE_HEADER_WORDS..record_words];
    let mut next_input = 0usize;
    let mut decoded = LitVec::new();
    for &word in returned {
        let Some(relative) = cube[next_input..]
            .iter()
            .position(|(current, _)| u32::from(*current) == word)
        else {
            return Err(HardwareError::InvalidResponse);
        };
        next_input += relative + 1;
        decoded.push(Lit::new(Var::from(word >> 1), word & 1 == 0));
    }
    Ok((
        MicChainResult {
            cube: decoded,
            trials: header.trials,
            physical_rounds: header.physical_rounds,
            complete: header.complete == 1,
            client_ns: 0,
            context_reused: false,
            reason,
            decisions: u64::from(header.decisions),
            conflicts: u64::from(header.conflicts),
            propagations: u64::from(header.propagations),
            learnt_clauses: u64::from(header.learnt_clauses),
        },
        record_words,
    ))
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct ArenaLaneView {
    bitmap: Vec<u32>,
    key: u64,
    valid: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ArenaViewUpdate {
    words: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ArenaMappedClause {
    id: u32,
    lo: u32,
    hi: u32,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct ArenaContextMapping {
    clauses: Vec<ArenaMappedClause>,
}

impl ArenaContextMapping {
    fn active(&self, frame: u32, ranged: bool) -> Vec<u32> {
        self.clauses
            .iter()
            .filter(|clause| !ranged || clause.lo <= frame && frame <= clause.hi)
            .map(|clause| clause.id)
            .collect()
    }
}

#[derive(Clone, Debug, Default)]
struct ResidentArena {
    n_var: u32,
    /// One normalized body can occur more than once in a CNF. Stable IDs are
    /// assigned by occurrence ordinal, not merely by body, so duplicate
    /// clauses remain duplicate physical occurrences exactly as in replay.
    instances: HashMap<Vec<u32>, Vec<u32>>,
    n_clause: u32,
    lanes: [ArenaLaneView; 2],
    next_view_key: u64,
}

impl ResidentArena {
    fn reset(&mut self, n_var: u32) {
        *self = Self {
            n_var,
            ..Self::default()
        };
    }

    fn normalize_clause(n_var: u32, literals: &LitVec) -> Result<Option<Vec<u32>>, HardwareError> {
        if n_var == 0 || literals.is_empty() {
            return Err(HardwareError::InvalidContext);
        }
        let mut normalized = Vec::with_capacity(literals.len());
        for &literal in literals.iter() {
            let literal = u32::from(literal);
            if literal >> 1 >= n_var {
                return Err(HardwareError::InvalidContext);
            }
            if normalized.contains(&(literal ^ 1)) {
                return Ok(None);
            }
            if !normalized.contains(&literal) {
                normalized.push(literal);
            }
        }
        Ok(Some(normalized))
    }

    fn intern_context(
        &mut self,
        context: &ShadowContext,
    ) -> Result<(ArenaContextMapping, Vec<ResidentClause>), HardwareError> {
        if self.n_var != context.n_var {
            self.reset(context.n_var);
        }
        let mut ordinals: HashMap<Vec<u32>, usize> = HashMap::new();
        let mut mapping = ArenaContextMapping::default();
        let mut appended = Vec::new();
        for clause in &context.clauses {
            let Some(body) = Self::normalize_clause(context.n_var, &clause.literals)? else {
                continue;
            };
            let ordinal = ordinals.entry(body.clone()).or_default();
            let ids = self.instances.entry(body.clone()).or_default();
            let id = if *ordinal < ids.len() {
                ids[*ordinal]
            } else {
                let id = self.n_clause;
                self.n_clause = self
                    .n_clause
                    .checked_add(1)
                    .ok_or(HardwareError::Capacity)?;
                ids.push(id);
                let literals = body
                    .iter()
                    .map(|literal| Lit::new(Var::from(literal >> 1), literal & 1 == 0))
                    .collect::<LitVec>();
                appended.push(ResidentClause::new(0, u32::MAX, literals));
                id
            };
            *ordinal += 1;
            mapping.clauses.push(ArenaMappedClause {
                id,
                lo: clause.lo,
                hi: clause.hi,
            });
        }
        Ok((mapping, appended))
    }

    fn plan_view(&mut self, lane: usize, active: &[u32]) -> Result<ArenaViewUpdate, HardwareError> {
        let lane = self
            .lanes
            .get_mut(lane)
            .ok_or(HardwareError::InvalidContext)?;
        let bitmap_words =
            usize::try_from((self.n_clause + 31) / 32).map_err(|_| HardwareError::Capacity)?;
        let mut target = vec![0u32; bitmap_words];
        for &clause in active {
            if clause >= self.n_clause {
                return Err(HardwareError::InvalidContext);
            }
            target[(clause >> 5) as usize] |= 1 << (clause & 31);
        }
        lane.bitmap.resize(bitmap_words, 0);
        let changed = !lane.valid || lane.bitmap != target;
        let (mode, key, update) = if changed {
            self.next_view_key = self
                .next_view_key
                .checked_add(1)
                .ok_or(HardwareError::Capacity)?;
            let mut toggles = Vec::new();
            for clause in 0..self.n_clause {
                let word = (clause >> 5) as usize;
                let mask = 1 << (clause & 31);
                if (lane.bitmap[word] ^ target[word]) & mask != 0 {
                    toggles.push(clause);
                }
            }
            if toggles.len() <= bitmap_words {
                (ARENA_VIEW_TOGGLE, self.next_view_key, toggles)
            } else {
                (ARENA_VIEW_BITMAP, self.next_view_key, target.clone())
            }
        } else {
            (ARENA_VIEW_REUSE, lane.key, Vec::new())
        };
        let update_words = u32::try_from(update.len()).map_err(|_| HardwareError::Capacity)?;
        let mut words = Vec::with_capacity(ARENA_VIEW_PREFIX_WORDS + update.len());
        words.extend([
            mode,
            key as u32,
            (key >> 32) as u32,
            self.n_clause,
            update_words,
        ]);
        words.extend(update);
        lane.bitmap = target;
        lane.key = key;
        lane.valid = true;
        Ok(ArenaViewUpdate { words })
    }
}

pub struct HardwareCdcl {
    n_var: u32,
    materialized_frame: Option<u32>,
    last_batch_work: HardwareWork,
    last_batch_records: Vec<HardwareWork>,
    stage_profile: bool,
    arena: ResidentArena,
    full_root_projection: Option<FullRootProjectionLease>,
    next_full_root_projection_handle: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct FullRootProjectionLease {
    handle: u32,
    next_var_by_current: Vec<u32>,
    init_value_by_current: Vec<u32>,
    decision_domain: Vec<u32>,
    latch_variables: Vec<u32>,
    input_variables: Vec<u32>,
}

fn semantic_batch_invalidates_projection(commands: &[BlockSemanticCommand]) -> bool {
    commands.iter().any(|command| {
        matches!(
            command.command,
            BLOCK_SEMANTIC_RESET | BLOCK_SEMANTIC_EVENT_RESET_EPOCH
        )
    })
}

impl FullRootProjectionLease {
    fn matches(
        &self,
        next_var_by_current: &[u32],
        init_value_by_current: &[u32],
        decision_domain: &[u32],
        latch_variables: &[u32],
        input_variables: &[u32],
    ) -> bool {
        self.next_var_by_current == next_var_by_current
            && self.init_value_by_current == init_value_by_current
            && self.decision_domain == decision_domain
            && self.latch_variables == latch_variables
            && self.input_variables == input_variables
    }
}

impl HardwareCdcl {
    pub fn compiled() -> bool {
        cfg!(has_cdcl_accel)
    }

    fn throughput_enabled() -> bool {
        std::env::var("INDUCTOR_CDCL_FPGA_THROUGHPUT")
            .ok()
            .is_some_and(|value| !matches!(value.as_str(), "0" | "false" | "off"))
    }

    fn selected_device() -> Option<i32> {
        std::env::var("INDUCTOR_CDCL_DEVICE")
            .ok()
            .and_then(|value| value.parse::<i32>().ok())
            .and_then(|device| (device >= 0).then_some(device))
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
                if let Some(device_index) = Self::selected_device() {
                    unsafe { ind_cdcl_open_with_device(path.as_ptr(), device_index) }
                } else {
                    unsafe { ind_cdcl_open(path.as_ptr()) }
                }
            };
            if rc != 0 {
                return Err(HardwareError::Open(rc));
            }
            if let Ok(worker) = std::env::var("INDUCTOR_CDCL_PORTFOLIO_WORKER") {
                eprintln!("inductor-cdcl: portfolio worker {worker} connected to FPGA service");
            }
            Ok(Self {
                n_var: 0,
                materialized_frame: None,
                last_batch_work: HardwareWork::default(),
                last_batch_records: Vec::new(),
                stage_profile: stage_profile_enabled(),
                arena: ResidentArena::default(),
                full_root_projection: None,
                next_full_root_projection_handle: 1,
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
        let path = std::env::var("INDUCTOR_CDCL_ACCEL").map_err(|_| HardwareError::Unavailable)?;
        Self::open(&path)
    }

    /// Execute one atomic, packed proof-state transaction against the
    /// resident BLOCK interpreter owned by the native/RPC service. This is a
    /// simulation-first transport today; the direct XRT bridge deliberately
    /// reports unsupported until the same opcode exists in the persistent
    /// hardware command ring.
    pub fn run_block_semantic_batch(
        &mut self,
        commands: &[BlockSemanticCommand],
    ) -> Result<Vec<BlockSemanticCommandResponse>, HardwareError> {
        // Ordinary journal commands mutate obligations and lemmas, not the
        // transition projection. Keep the lease across those hot-path syncs;
        // only an epoch reset makes the controller discard the cached arrays.
        if semantic_batch_invalidates_projection(commands) {
            self.full_root_projection = None;
        }
        let request = pack_block_semantic_batch(commands).ok_or(HardwareError::Capacity)?;
        let response_capacity = usize::try_from(request[3]).map_err(|_| HardwareError::Capacity)?;
        let request_words = u32::try_from(request.len()).map_err(|_| HardwareError::Capacity)?;
        let response_capacity_words =
            u32::try_from(response_capacity).map_err(|_| HardwareError::Capacity)?;
        #[cfg(has_cdcl_accel)]
        {
            let mut response = vec![0u32; response_capacity];
            let mut out_words = 0u32;
            let rc = unsafe {
                ind_cdcl_run_block_semantic_batch(
                    request.as_ptr(),
                    request_words,
                    response.as_mut_ptr(),
                    response_capacity_words,
                    &mut out_words,
                )
            };
            if rc != 0 {
                return Err(HardwareError::Command(rc));
            }
            let out_words = usize::try_from(out_words).map_err(|_| HardwareError::Capacity)?;
            if out_words > response.len() {
                return Err(HardwareError::InvalidResponse);
            }
            response.truncate(out_words);
            let (error, records) = decode_block_semantic_batch_response(&response)
                .ok_or(HardwareError::InvalidResponse)?;
            let command_status = records
                .iter()
                .find(|record| record.status != 0)
                .map_or(0, |record| record.status);
            if error != 0 || command_status != 0 || records.len() != commands.len() {
                return Err(HardwareError::BlockSemantic {
                    error,
                    completed: records.len().min(u32::MAX as usize) as u32,
                    command: commands
                        .get(records.len().saturating_sub(1))
                        .map_or(u32::MAX, |command| command.command),
                    command_status,
                    obligation_count: records.last().map_or(0, |record| record.obligation_count),
                    lemma_count: records.last().map_or(0, |record| record.lemma_count),
                });
            }
            Ok(records)
        }
        #[cfg(not(has_cdcl_accel))]
        {
            let _ = (request, request_words, response_capacity_words);
            Err(HardwareError::Unavailable)
        }
    }

    /// Run one failure-atomic resident queue-to-CDCL wave through the native
    /// or RPC service. `query_template` contributes the already-qualified
    /// decision domain, flags and budgets; assumptions are constructed from
    /// the tagged resident state cubes by the controller itself.
    pub fn run_block_root(
        &mut self,
        max_frame: u32,
        requested_queries: usize,
        next_var_by_current: &[u32],
        query_template: &IncrementalQuery,
    ) -> Result<BlockRootResponse, HardwareError> {
        if next_var_by_current.len() != self.n_var as usize
            || query_template.domain.is_empty()
            || query_template.constraints.len() != 0
        {
            return Err(HardwareError::InvalidContext);
        }
        let mut domain_query = query_template.clone();
        domain_query.frame = 0;
        domain_query.assumptions.clear();
        domain_query.constraints.clear();
        let (domain_header, domain_words) = domain_query.pack();
        if domain_header.n_assumptions != 0
            || domain_header.n_constraint_words != 0
            || usize::try_from(domain_header.n_domain).ok() != Some(domain_words.len())
        {
            return Err(HardwareError::InvalidContext);
        }
        let request = pack_block_root_request(
            max_frame,
            requested_queries,
            next_var_by_current,
            &domain_words,
            domain_header.flags,
            domain_header.decision_budget,
            domain_header.conflict_budget,
        )
        .ok_or(HardwareError::Capacity)?;
        let per_result = RESPONSE_HEADER_WORDS
            .checked_add(next_var_by_current.len().max(domain_words.len()))
            .ok_or(HardwareError::Capacity)?;
        let response_capacity = BLOCK_ROOT_BATCH_OFFSET
            .checked_add(4)
            .and_then(|words| {
                requested_queries
                    .checked_mul(per_result)
                    .and_then(|results| words.checked_add(results))
            })
            .ok_or(HardwareError::Capacity)?;
        let request_words = u32::try_from(request.len()).map_err(|_| HardwareError::Capacity)?;
        let response_capacity_words =
            u32::try_from(response_capacity).map_err(|_| HardwareError::Capacity)?;
        #[cfg(has_cdcl_accel)]
        {
            let mut response = vec![0u32; response_capacity];
            let mut out_words = 0u32;
            let rc = unsafe {
                ind_cdcl_run_block_root(
                    request.as_ptr(),
                    request_words,
                    response.as_mut_ptr(),
                    response_capacity_words,
                    &mut out_words,
                )
            };
            if rc != 0 {
                return Err(HardwareError::Command(rc));
            }
            let out_words = usize::try_from(out_words).map_err(|_| HardwareError::Capacity)?;
            if out_words > response.len() {
                return Err(HardwareError::InvalidResponse);
            }
            response.truncate(out_words);
            let decoded =
                decode_block_root_response(&response).ok_or(HardwareError::InvalidResponse)?;
            if decoded.status == BlockRootExecutionStatus::Ok
                && (decoded.batch.len() < 4
                    || decoded.batch[0] != ABI_VERSION
                    || usize::try_from(decoded.batch[1]).ok() != Some(decoded.work.len())
                    || usize::try_from(decoded.batch[2]).ok() != decoded.batch.len().checked_sub(4))
            {
                return Err(HardwareError::InvalidResponse);
            }
            Ok(decoded)
        }
        #[cfg(not(has_cdcl_accel))]
        {
            let _ = (request, request_words, response_capacity_words);
            Err(HardwareError::Unavailable)
        }
    }

    /// Execute a bounded, complete resident BLOCK root. SAT predecessors,
    /// UNSAT core reconstruction, MIC and lemma append remain inside one
    /// native/RPC command; the ordered journal is the only CPU sync payload.
    pub fn run_block_full_root(
        &mut self,
        max_frame: u32,
        step_limit: usize,
        frontier_limit: usize,
        next_var_by_current: &[u32],
        init_value_by_current: &[u32],
        latch_variables: &[u32],
        input_variables: &[u32],
        query_template: &IncrementalQuery,
        compacted_retry: bool,
        predecessor_lift: bool,
    ) -> Result<BlockFullRootResponse, HardwareError> {
        if next_var_by_current.len() != self.n_var as usize
            || init_value_by_current.len() != next_var_by_current.len()
            || query_template.domain.is_empty()
            || !query_template.constraints.is_empty()
        {
            if std::env::var_os("INDUCTOR_CDCL_NATIVE_DIAG_FIRST_ERROR").is_some() {
                eprintln!(
                    concat!(
                        "inductor-cdcl: full-root invalid context: resident_n_var={} ",
                        "next_map={} init={} domain={} constraints={}"
                    ),
                    self.n_var,
                    next_var_by_current.len(),
                    init_value_by_current.len(),
                    query_template.domain.len(),
                    query_template.constraints.len(),
                );
            }
            return Err(HardwareError::InvalidContext);
        }
        let mut domain_query = query_template.clone();
        domain_query.frame = 0;
        domain_query.assumptions.clear();
        domain_query.constraints.clear();
        let (domain_header, domain_words) = domain_query.pack();
        if domain_header.n_assumptions != 0
            || domain_header.n_constraint_words != 0
            || usize::try_from(domain_header.n_domain).ok() != Some(domain_words.len())
        {
            return Err(HardwareError::InvalidContext);
        }
        let reuse_handle = (!compacted_retry)
            .then(|| self.full_root_projection.as_ref())
            .flatten()
            .filter(|lease| {
                lease.matches(
                    next_var_by_current,
                    init_value_by_current,
                    &domain_words,
                    latch_variables,
                    input_variables,
                )
            })
            .map(|lease| lease.handle);
        let projection_handle = reuse_handle.unwrap_or_else(|| {
            let handle = self.next_full_root_projection_handle.max(1);
            self.next_full_root_projection_handle = handle.wrapping_add(1).max(1);
            handle
        });
        let request = if reuse_handle.is_some() {
            pack_block_full_root_continuation(
                max_frame,
                step_limit,
                frontier_limit,
                next_var_by_current.len(),
                domain_words.len(),
                latch_variables.len(),
                input_variables.len(),
                domain_header.flags,
                domain_header.decision_budget,
                domain_header.conflict_budget,
                projection_handle,
                compacted_retry,
            )
        } else {
            pack_block_full_root_request(
                max_frame,
                step_limit,
                frontier_limit,
                next_var_by_current,
                init_value_by_current,
                &domain_words,
                latch_variables,
                input_variables,
                domain_header.flags,
                domain_header.decision_budget,
                domain_header.conflict_budget,
                projection_handle,
                compacted_retry,
            )
        };
        if request.is_none() && std::env::var_os("INDUCTOR_CDCL_NATIVE_DIAG_FIRST_ERROR").is_some()
        {
            eprintln!(
                concat!(
                    "inductor-cdcl: full-root pack rejected: steps={} frontier={} ",
                    "next={} init={} init_max={} domain={} latches={} inputs={} ",
                    "flags=0x{:x} budgets={}/{}"
                ),
                step_limit,
                frontier_limit,
                next_var_by_current.len(),
                init_value_by_current.len(),
                init_value_by_current.iter().copied().max().unwrap_or(0),
                domain_words.len(),
                latch_variables.len(),
                input_variables.len(),
                domain_header.flags,
                domain_header.decision_budget,
                domain_header.conflict_budget,
            );
        }
        let mut request = request.ok_or(HardwareError::InvalidContext)?;
        if predecessor_lift {
            request[5] |= crate::accel::cdcl::BLOCK_PREDECESSOR_LIFT;
        }
        if std::env::var("INDUCTOR_CDCL_BLOCK_FULL_ROOT_SKIP_MIC")
            .ok()
            .is_some_and(|value| !matches!(value.as_str(), "0" | "false" | "off"))
        {
            request[11] |= crate::accel::cdcl::BLOCK_FULL_ROOT_SKIP_MIC;
        }
        if std::env::var("INDUCTOR_CDCL_BLOCK_FULL_ROOT_CPU_MIC")
            .ok()
            .is_some_and(|value| !matches!(value.as_str(), "0" | "false" | "off"))
        {
            request[11] |= crate::accel::cdcl::BLOCK_FULL_ROOT_CPU_MIC;
            // Only the head is popped/owned at the hybrid boundary. Keep one
            // temporal resident stream instead of producing non-head results
            // whose MIC work cannot be committed transactionally.
            request[10] = 1;
        }
        let response_capacity = block_full_root_required_response_capacity(
            step_limit,
            latch_variables.len(),
            input_variables.len(),
        )
        .ok_or(HardwareError::Capacity)?;
        let request_words = u32::try_from(request.len()).map_err(|_| HardwareError::Capacity)?;
        let response_capacity_words =
            u32::try_from(response_capacity).map_err(|_| HardwareError::Capacity)?;
        #[cfg(has_cdcl_accel)]
        {
            let mut response = vec![0u32; response_capacity];
            let mut out_words = 0u32;
            let rc = unsafe {
                ind_cdcl_run_block_full_root(
                    request.as_ptr(),
                    request_words,
                    response.as_mut_ptr(),
                    response_capacity_words,
                    &mut out_words,
                )
            };
            if rc != 0 {
                record_full_root_transaction(&request, &[], rc);
                self.full_root_projection = None;
                return Err(if rc == -27 {
                    HardwareError::Timeout
                } else {
                    HardwareError::Command(rc)
                });
            }
            let out_words = usize::try_from(out_words).map_err(|_| HardwareError::Capacity)?;
            if out_words > response.len() {
                record_full_root_transaction(&request, &[], rc);
                return Err(HardwareError::InvalidResponse);
            }
            response.truncate(out_words);
            record_full_root_transaction(&request, &response, rc);
            let decoded =
                decode_block_full_root_response(&response).ok_or(HardwareError::InvalidResponse)?;
            if decoded.status == BlockFullRootStatus::StepBudget {
                if reuse_handle.is_none() {
                    self.full_root_projection = Some(FullRootProjectionLease {
                        handle: projection_handle,
                        next_var_by_current: next_var_by_current.to_vec(),
                        init_value_by_current: init_value_by_current.to_vec(),
                        decision_domain: domain_words,
                        latch_variables: latch_variables.to_vec(),
                        input_variables: input_variables.to_vec(),
                    });
                }
            } else {
                self.full_root_projection = None;
            }
            Ok(decoded)
        }
        #[cfg(not(has_cdcl_accel))]
        {
            let _ = (
                request,
                request_words,
                response_capacity_words,
                projection_handle,
            );
            Err(HardwareError::Unavailable)
        }
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
                self.materialized_frame = None;
                self.arena = ResidentArena::default();
                self.full_root_projection = None;
                return Err(HardwareError::Command(rc));
            }
            self.n_var = n_var;
            self.materialized_frame = None;
            self.arena = ResidentArena::default();
            // Formula reload does not alter transition projection metadata.
            // Exact vector matching below prevents reuse after a true mapping
            // change; the RPC server arbitrates the physical cache owner.
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
    pub fn load_solver_context(&mut self, solver: &DagCnfSolver) -> Result<(), HardwareError> {
        let (n_var, _frame, snapshot) = solver.incremental_resident_snapshot();
        let clauses: Vec<_> = snapshot
            .into_iter()
            .map(|literals| ResidentClause::new(0, u32::MAX, literals))
            .collect();
        self.load_context(n_var, &clauses)
    }

    pub fn add_frame_clauses(&mut self, clauses: &[ResidentClause]) -> Result<(), HardwareError> {
        if self.n_var == 0 {
            return Err(HardwareError::InvalidContext);
        }
        let n_clause = u32::try_from(clauses.len()).map_err(|_| HardwareError::Capacity)?;
        let words = pack_clauses(&[n_clause], self.n_var, clauses)?;
        #[cfg(has_cdcl_accel)]
        {
            let rc = unsafe { ind_cdcl_add_frame_clauses(words.as_ptr(), words.len() as u32) };
            if rc != 0 {
                self.materialized_frame = None;
                return Err(HardwareError::Command(rc));
            }
            // The device adds the resident suffix to its range-checked delta
            // occurrence overlay.  An already materialized frame therefore
            // remains valid; periodic overlay merges rematerialize that same
            // frame inside the append command.
            Ok(())
        }
        #[cfg(not(has_cdcl_accel))]
        {
            let _ = words;
            Err(HardwareError::Unavailable)
        }
    }

    /// Rebuild the hot occurrence view for one resident frame. Queries still
    /// perform the same check on device, so this is a timing/accounting split,
    /// not a semantic dependency on host state.
    pub fn materialize_frame(&mut self, frame: u32) -> Result<bool, HardwareError> {
        if self.n_var == 0 {
            return Err(HardwareError::InvalidContext);
        }
        if self.materialized_frame == Some(frame) {
            return Ok(false);
        }
        #[cfg(has_cdcl_accel)]
        {
            let rc = unsafe { ind_cdcl_materialize_frame(frame) };
            if rc != 0 {
                return Err(HardwareError::Command(rc));
            }
            self.materialized_frame = Some(frame);
            Ok(true)
        }
        #[cfg(not(has_cdcl_accel))]
        {
            let _ = frame;
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
            // Lane zero may now hold the last frame assigned by the batch.
            // Do not let a later explicit MIC materialization rely on stale
            // host-side knowledge of the on-card hot view.
            self.materialized_frame = None;
            self.decode_batch_response(queries, &response)
        }
        #[cfg(not(has_cdcl_accel))]
        {
            let _ = (request, response_capacity_u32, response);
            Err(HardwareError::Unavailable)
        }
    }

    fn invalidate_arena_context(&mut self) {
        self.n_var = 0;
        self.materialized_frame = None;
        self.arena = ResidentArena::default();
    }

    /// Extend one process-lifetime union arena and return the stable IDs that
    /// represent this logical snapshot. Context switches and frame shrinkage
    /// only change the subsequent private lane views; the physical clause
    /// store is reset solely when the variable universe changes.
    fn prepare_arena_context(
        &mut self,
        context: &ShadowContext,
    ) -> Result<ArenaContextMapping, HardwareError> {
        if context.n_var == 0 {
            return Err(HardwareError::InvalidContext);
        }
        if self.arena.n_var != context.n_var || self.n_var != context.n_var {
            self.load_context(context.n_var, &[])?;
            self.arena.reset(context.n_var);
        }
        let mut candidate = self.arena.clone();
        let (mapping, appended) = candidate.intern_context(context)?;
        if !appended.is_empty()
            && let Err(error) = self.add_frame_clauses(&appended)
        {
            self.invalidate_arena_context();
            return Err(error);
        }
        self.arena = candidate;
        Ok(mapping)
    }

    /// Batch inquiries from different IC3 frame snapshots through one union
    /// arena. Each query receives the mapping of its own exact/ranged context;
    /// clauses interned for another snapshot remain physically resident but
    /// disabled in that query's lane-local bitmap.
    fn solve_arena_batch_contexts(
        &mut self,
        contexts: &[ShadowContext],
        queries: &[IncrementalQuery],
    ) -> Result<Vec<IncrementalResult>, HardwareError> {
        self.last_batch_work = HardwareWork::default();
        self.last_batch_records.clear();
        if queries.is_empty() || contexts.len() != queries.len() {
            return Err(HardwareError::InvalidContext);
        }
        let required_n_var = contexts
            .iter()
            .map(|context| context.n_var)
            .max()
            .filter(|n_var| *n_var != 0 && *n_var <= QUALIFIED_ARENA_MAX_VARS)
            .ok_or(HardwareError::InvalidContext)?;
        // Solver clones in one IC3 frontier can own different counts of
        // temporary activation variables. The resident engine only needs one
        // upper bound, so retain an already-larger arena or grow once to the
        // maximum instead of splitting/resetting on each logical snapshot.
        let arena_n_var = if self.n_var >= required_n_var
            && self.n_var <= QUALIFIED_ARENA_MAX_VARS
            && self.arena.n_var == self.n_var
        {
            self.n_var
        } else {
            required_n_var
        };
        let normalized_contexts: Vec<_> = contexts
            .iter()
            .cloned()
            .map(|mut context| {
                context.n_var = arena_n_var;
                context
            })
            .collect();
        let mut mappings = Vec::with_capacity(normalized_contexts.len());
        for context in &normalized_contexts {
            mappings.push(self.prepare_arena_context(context)?);
        }
        let mut candidate = self.arena.clone();
        let mut views = Vec::with_capacity(queries.len());
        for (index, ((context, mapping), query)) in normalized_contexts
            .iter()
            .zip(mappings.iter())
            .zip(queries.iter())
            .enumerate()
        {
            let active = mapping.active(
                query.frame,
                context.scope == ShadowContextScope::FrameRanged,
            );
            views.push(candidate.plan_view(index & 1, &active)?);
        }
        let (request, response_capacity) =
            pack_arena_batch_request(queries, &views, self.stage_profile)?;
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
                ind_cdcl_solve_arena_batch(
                    request.as_ptr(),
                    request.len() as u32,
                    response.as_mut_ptr(),
                    response_capacity_u32,
                    &mut out_words,
                )
            };
            if rc != 0 {
                self.invalidate_arena_context();
                return Err(HardwareError::Command(rc));
            }
            let out_words = usize::try_from(out_words).map_err(|_| HardwareError::Capacity)?;
            if out_words > response.len() {
                self.invalidate_arena_context();
                return Err(HardwareError::Capacity);
            }
            response.truncate(out_words);
            self.arena = candidate;
            self.materialized_frame = None;
            self.decode_batch_response(queries, &response)
        }
        #[cfg(not(has_cdcl_accel))]
        {
            let _ = (request, response_capacity_u32, response, candidate);
            Err(HardwareError::Unavailable)
        }
    }

    fn solve_arena_mic_chain(
        &mut self,
        context: &ShadowContext,
        frame: u32,
        cube: &[(Lit, Lit)],
        constraints: &[LitVec],
        protected_index: usize,
        decision_budget: u32,
        conflict_budget: u32,
        max_trials: u32,
    ) -> Result<MicChainResult, HardwareError> {
        let mapping = self.prepare_arena_context(context)?;
        let active = mapping.active(frame, context.scope == ShadowContextScope::FrameRanged);
        let mut candidate = self.arena.clone();
        let view = candidate.plan_view(0, &active)?;
        let mic = pack_mic_chain_request(
            self.n_var,
            frame,
            cube,
            constraints,
            protected_index,
            decision_budget,
            conflict_budget,
            max_trials,
        )?;
        let request = pack_arena_mic_request(&view, &mic)?;
        let response_capacity = MIC_RESPONSE_HEADER_WORDS
            .checked_add(cube.len())
            .ok_or(HardwareError::Capacity)?;
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
                ind_cdcl_solve_arena_mic(
                    request.as_ptr(),
                    request.len() as u32,
                    response.as_mut_ptr(),
                    response_capacity_u32,
                    &mut out_words,
                )
            };
            if rc != 0 {
                self.invalidate_arena_context();
                return Err(HardwareError::Command(rc));
            }
            let out_words = usize::try_from(out_words).map_err(|_| HardwareError::Capacity)?;
            if out_words > response.len() {
                self.invalidate_arena_context();
                return Err(HardwareError::Capacity);
            }
            let (result, record_words) = match decode_mic_chain_record(cube, &response[..out_words])
            {
                Ok(decoded) => decoded,
                Err(error) => {
                    self.invalidate_arena_context();
                    return Err(error);
                }
            };
            if record_words != out_words {
                self.invalidate_arena_context();
                return Err(HardwareError::InvalidResponse);
            }
            self.arena = candidate;
            self.materialized_frame = None;
            Ok(result)
        }
        #[cfg(not(has_cdcl_accel))]
        {
            let _ = (request, response_capacity_u32, response, candidate);
            Err(HardwareError::Unavailable)
        }
    }

    /// Execute the dependent drop traversal in one device command. The
    /// returned cube is only a candidate; the IC3 caller must still prove it
    /// exactly against the live GipSAT frame before adoption.
    pub fn solve_mic_chain(
        &mut self,
        frame: u32,
        cube: &[(Lit, Lit)],
        constraints: &[LitVec],
        protected_index: usize,
        decision_budget: u32,
        conflict_budget: u32,
        max_trials: u32,
    ) -> Result<MicChainResult, HardwareError> {
        if self.n_var == 0 {
            return Err(HardwareError::InvalidContext);
        }
        let request = pack_mic_chain_request(
            self.n_var,
            frame,
            cube,
            constraints,
            protected_index,
            decision_budget,
            conflict_budget,
            max_trials,
        )?;
        let response_capacity = MIC_RESPONSE_HEADER_WORDS
            .checked_add(cube.len())
            .ok_or(HardwareError::Capacity)?;
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
                ind_cdcl_solve_mic_chain(
                    request.as_ptr(),
                    request.len() as u32,
                    response.as_mut_ptr(),
                    response_capacity_u32,
                    &mut out_words,
                )
            };
            if rc != 0 {
                self.materialized_frame = None;
                return Err(HardwareError::Command(rc));
            }
            let out_words = usize::try_from(out_words).map_err(|_| HardwareError::Capacity)?;
            if out_words > response.len() {
                return Err(HardwareError::Capacity);
            }
            let (result, record_words) = decode_mic_chain_record(cube, &response[..out_words])?;
            if record_words != out_words {
                return Err(HardwareError::InvalidResponse);
            }
            self.materialized_frame = Some(frame);
            Ok(result)
        }
        #[cfg(not(has_cdcl_accel))]
        {
            let _ = (request, response_capacity_u32, response);
            Err(HardwareError::Unavailable)
        }
    }

    /// Atomically append the next resident lemma suffix and execute the
    /// immediately following dependent MIC traversal in one XRT submission.
    /// The device validates both payloads before committing the append, and
    /// the MIC command itself selects/materializes `frame` when necessary.
    pub fn append_and_solve_mic_chain(
        &mut self,
        clauses: &[ResidentClause],
        frame: u32,
        cube: &[(Lit, Lit)],
        constraints: &[LitVec],
        protected_index: usize,
        decision_budget: u32,
        conflict_budget: u32,
        max_trials: u32,
    ) -> Result<MicChainResult, HardwareError> {
        if self.n_var == 0 || clauses.is_empty() {
            return Err(HardwareError::InvalidContext);
        }
        let n_clause = u32::try_from(clauses.len()).map_err(|_| HardwareError::Capacity)?;
        let append = pack_clauses(&[n_clause], self.n_var, clauses)?;
        let append_words = u32::try_from(append.len()).map_err(|_| HardwareError::Capacity)?;
        let mic = pack_mic_chain_request(
            self.n_var,
            frame,
            cube,
            constraints,
            protected_index,
            decision_budget,
            conflict_budget,
            max_trials,
        )?;
        let request_words = 1usize
            .checked_add(append.len())
            .and_then(|words| words.checked_add(mic.len()))
            .ok_or(HardwareError::Capacity)?;
        u32::try_from(request_words).map_err(|_| HardwareError::Capacity)?;
        let mut request = Vec::with_capacity(request_words);
        request.push(append_words);
        request.extend(append);
        request.extend(mic);

        let response_capacity = MIC_RESPONSE_HEADER_WORDS
            .checked_add(cube.len())
            .ok_or(HardwareError::Capacity)?;
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
                ind_cdcl_append_and_solve_mic_chain(
                    request.as_ptr(),
                    request.len() as u32,
                    response.as_mut_ptr(),
                    response_capacity_u32,
                    &mut out_words,
                )
            };
            if rc != 0 {
                self.materialized_frame = None;
                return Err(HardwareError::Command(rc));
            }
            let out_words = usize::try_from(out_words).map_err(|_| HardwareError::Capacity)?;
            if out_words > response.len() {
                return Err(HardwareError::Capacity);
            }
            let (result, record_words) = decode_mic_chain_record(cube, &response[..out_words])?;
            if record_words != out_words {
                return Err(HardwareError::InvalidResponse);
            }
            self.materialized_frame = Some(frame);
            Ok(result)
        }
        #[cfg(not(has_cdcl_accel))]
        {
            let _ = (request, response_capacity_u32, response);
            Err(HardwareError::Unavailable)
        }
    }

    /// Execute up to four independent dependent-drop traversals in one
    /// physical command. Every chain selects its own frame from one resident
    /// ranged-clause context; mutable frame/cube/search state remains private
    /// to its fixed hardware lane.
    pub fn solve_mic_chains(
        &mut self,
        chains: &[MicChainQuery<'_>],
    ) -> Result<Vec<MicChainResult>, HardwareError> {
        if self.n_var == 0 {
            return Err(HardwareError::InvalidContext);
        }
        let (request, response_capacity) = pack_mic_chains_request(self.n_var, chains)?;
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
                ind_cdcl_solve_mic_chains(
                    request.as_ptr(),
                    request.len() as u32,
                    response.as_mut_ptr(),
                    response_capacity_u32,
                    &mut out_words,
                )
            };
            if rc != 0 {
                self.materialized_frame = None;
                return Err(HardwareError::Command(rc));
            }
            let out_words = usize::try_from(out_words).map_err(|_| HardwareError::Capacity)?;
            if out_words > response.len() || out_words < MIC_BATCH_RESPONSE_HEADER_WORDS {
                return Err(HardwareError::InvalidResponse);
            }
            let prefix = &response[..MIC_BATCH_RESPONSE_HEADER_WORDS];
            let result_words =
                usize::try_from(prefix[2]).map_err(|_| HardwareError::InvalidResponse)?;
            if prefix[0] != ABI_VERSION
                || usize::try_from(prefix[1]).ok() != Some(chains.len())
                || prefix[3] != 0
                || MIC_BATCH_RESPONSE_HEADER_WORDS.checked_add(result_words) != Some(out_words)
            {
                return Err(HardwareError::InvalidResponse);
            }
            let mut results = Vec::with_capacity(chains.len());
            let mut at = MIC_BATCH_RESPONSE_HEADER_WORDS;
            for (chain_index, chain) in chains.iter().enumerate() {
                let decoded = decode_mic_chain_record(chain.cube, &response[at..out_words]);
                let (result, words) = decoded.map_err(|error| {
                    let header_end = at
                        .saturating_add(MIC_RESPONSE_HEADER_WORDS)
                        .min(out_words);
                    eprintln!(
                        "inductor-cdcl: invalid MIC batch record lane {chain_index}/{}, input {}, offset {at}/{out_words}, header {:?}",
                        chains.len(),
                        chain.cube.len(),
                        &response[at..header_end],
                    );
                    error
                })?;
                results.push(result);
                at = at
                    .checked_add(words)
                    .ok_or(HardwareError::InvalidResponse)?;
            }
            if at != out_words {
                return Err(HardwareError::InvalidResponse);
            }
            self.materialized_frame = chains.first().map(|chain| chain.frame);
            Ok(results)
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
            pack_load_context_and_batch_request(n_var, clauses, queries, self.stage_profile)?;
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
            self.materialized_frame = None;
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
    scope: ShadowContextScope,
}

/// Materialize the exact union of the live IC3 frame solvers as one ranged
/// resident formula. The append-only event registry is the fast path, but
/// subsumption and frame promotion can legitimately make that history differ
/// from an exact solver snapshot. A fused root still needs a correct resync
/// path when that happens.
fn block_root_ranged_context(solvers: &[&DagCnfSolver]) -> Option<ShadowContext> {
    if solvers.is_empty() {
        return None;
    }
    let mut registry = block_root_range_registry().lock().ok()?;
    if registry.n_var == 0 {
        return None;
    }
    for solver in solvers {
        let (solver_n_var, frame, solver_transition, lemmas) =
            solver.incremental_resident_partition();
        if solver_n_var != registry.n_var || solver_transition != registry.transition {
            return None;
        }
        for literals in lemmas {
            let mut key: Vec<u32> = literals.iter().map(|literal| u32::from(*literal)).collect();
            key.sort_unstable();
            key.dedup();
            if registry.seen.insert((frame, key)) {
                // Frame solvers only gain clauses; IC3's bookkeeping may
                // remove a subsumed FrameLemma, but DagCnfSolver deliberately
                // retains the already-proved clause. Keeping each first-seen
                // exact-frame occurrence therefore gives a monotonic physical
                // prefix without changing the logical formula at that frame.
                registry
                    .clauses
                    .push(ResidentClause::new(frame, frame, literals));
            }
        }
    }
    let clauses = registry
        .transition
        .iter()
        .cloned()
        .map(|literals| ResidentClause::new(0, u32::MAX, literals))
        .chain(registry.clauses.iter().cloned())
        .collect();
    Some(ShadowContext {
        n_var: registry.n_var,
        clauses,
        scope: ShadowContextScope::FrameRanged,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ShadowContextScope {
    /// Only the immutable transition CNF is resident. Frame lemmas travel in
    /// every query, so no other resident clause may be active at its frame.
    SharedTransition,
    /// Transition clauses are shared across all frames and every remaining
    /// clause is active only at this exact frame.
    ExactFrame(u32),
    /// One append-only formula contains permanent clauses and exact IC3 lemma
    /// validity intervals; the query header selects the active frame on chip.
    FrameRanged,
}

/// Exact physical context currently resident on the FPGA. FrameRanged mode is
/// indexed on chip so expired frame buckets can be skipped without replacing
/// the physical clause store.
#[derive(Clone, Debug, Eq, PartialEq)]
struct LoadedContext {
    n_var: u32,
    clauses: Vec<ResidentClause>,
    scope: ShadowContextScope,
}

impl From<&ShadowContext> for LoadedContext {
    fn from(context: &ShadowContext) -> Self {
        Self {
            n_var: context.n_var,
            clauses: context.clauses.clone(),
            scope: context.scope,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ContextUpdate {
    Ready,
    Append(Vec<ResidentClause>),
    Reload,
}

/// Decide whether a logical query snapshot is already represented by the
/// physical resident snapshot, can be reached by a monotonic clause append, or
/// needs an exact reload. This is deliberately strict: any transition change,
/// frame change, lemma deletion/reordering, or unexpected clause range forces
/// a reload.
fn plan_context_update(loaded: Option<&LoadedContext>, target: &ShadowContext) -> ContextUpdate {
    let Some(loaded) = loaded else {
        return ContextUpdate::Reload;
    };
    if loaded.n_var != target.n_var {
        return ContextUpdate::Reload;
    }
    if target.scope == ShadowContextScope::SharedTransition {
        return if loaded.scope == target.scope && loaded.clauses == target.clauses {
            ContextUpdate::Ready
        } else {
            ContextUpdate::Reload
        };
    }
    if target.scope == ShadowContextScope::FrameRanged {
        if loaded.scope != ShadowContextScope::FrameRanged {
            return ContextUpdate::Reload;
        }
        if target.clauses.starts_with(&loaded.clauses) {
            let delta = target.clauses[loaded.clauses.len()..].to_vec();
            return if delta.is_empty() {
                ContextUpdate::Ready
            } else {
                ContextUpdate::Append(delta)
            };
        }
        return ContextUpdate::Reload;
    }
    let ShadowContextScope::ExactFrame(frame) = target.scope else {
        unreachable!();
    };
    if loaded.scope != ShadowContextScope::ExactFrame(frame) {
        return ContextUpdate::Reload;
    }
    let is_shared = |clause: &&ResidentClause| clause.lo == 0 && clause.hi == u32::MAX;
    if !loaded
        .clauses
        .iter()
        .filter(is_shared)
        .eq(target.clauses.iter().filter(is_shared))
    {
        return ContextUpdate::Reload;
    }
    if target.clauses.iter().any(|clause| {
        !(clause.lo == 0 && clause.hi == u32::MAX) && !(clause.lo == frame && clause.hi == frame)
    }) || loaded.clauses.iter().any(|clause| {
        !(clause.lo == 0 && clause.hi == u32::MAX) && !(clause.lo == frame && clause.hi == frame)
    }) {
        return ContextUpdate::Reload;
    }
    let is_frame = |clause: &&ResidentClause| clause.lo == frame && clause.hi == frame;
    let mut target_frame = target.clauses.iter().filter(is_frame);
    for resident in loaded.clauses.iter().filter(is_frame) {
        if target_frame.next() != Some(resident) {
            return ContextUpdate::Reload;
        }
    }
    let delta: Vec<_> = target_frame.cloned().collect();
    if delta.is_empty() {
        ContextUpdate::Ready
    } else {
        ContextUpdate::Append(delta)
    }
}

/// Canonical logical clause view selected by one frame from the exact physical
/// resident image. The physical image remains ordered and may contain duplicate
/// occurrences; formula equivalence deliberately compares a sorted clause set.
fn resident_formula_view(loaded: &LoadedContext, frame: u32) -> Result<Vec<Vec<u32>>, String> {
    let selected = match loaded.scope {
        ShadowContextScope::FrameRanged => loaded
            .clauses
            .iter()
            .filter(|clause| clause.lo <= frame && frame <= clause.hi)
            .map(|clause| &clause.literals)
            .collect::<Vec<_>>(),
        ShadowContextScope::ExactFrame(exact) if exact == frame => loaded
            .clauses
            .iter()
            .map(|clause| &clause.literals)
            .collect(),
        scope => {
            return Err(format!(
                "resident full-root formula oracle cannot select frame {frame} from {scope:?}"
            ));
        }
    };
    Ok(canonical_clause_set(selected.into_iter()))
}

/// A stable diagnostic fingerprint. Equality is always decided by the exact
/// canonical vectors above; this hash is only a compact way to identify the
/// two disagreeing frame views in logs and artifacts.
fn formula_view_fingerprint(view: &[Vec<u32>]) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut hash = OFFSET;
    let mut feed = |word: u32| {
        for byte in word.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(PRIME);
        }
    };
    feed(view.len().min(u32::MAX as usize) as u32);
    for clause in view {
        feed(clause.len().min(u32::MAX as usize) as u32);
        for &literal in clause {
            feed(literal);
        }
        feed(u32::MAX);
    }
    hash
}

fn compare_resident_formula_view(
    loaded: &LoadedContext,
    n_var: u32,
    frame: u32,
    cpu_clauses: &[LitVec],
) -> Result<(), String> {
    if loaded.n_var != n_var {
        return Err(format!(
            "resident full-root formula n_var mismatch at frame {frame}: device={} CPU={n_var}",
            loaded.n_var
        ));
    }
    let device = resident_formula_view(loaded, frame)?;
    let cpu = canonical_clause_set(cpu_clauses.iter());
    if device == cpu {
        return Ok(());
    }
    let cpu_only = cpu
        .iter()
        .find(|clause| device.binary_search(clause).is_err());
    let device_only = device
        .iter()
        .find(|clause| cpu.binary_search(clause).is_err());
    let device_only_ranges = device_only.map(|body| {
        loaded
            .clauses
            .iter()
            .filter_map(|clause| {
                let mut raw: Vec<u32> = clause
                    .literals
                    .iter()
                    .map(|literal| u32::from(*literal))
                    .collect();
                raw.sort_unstable();
                raw.dedup();
                (raw == **body).then_some((clause.lo, clause.hi))
            })
            .collect::<Vec<_>>()
    });
    Err(format!(
        "resident full-root formula mismatch at frame {frame}: device clauses={} fingerprint={:016x}, CPU clauses={} fingerprint={:016x}, first device-only={device_only:?} physical-ranges={device_only_ranges:?}, first CPU-only={cpu_only:?}",
        device.len(),
        formula_view_fingerprint(&device),
        cpu.len(),
        formula_view_fingerprint(&cpu),
    ))
}

/// Compare the device mutation journal's exact physical frame views with the
/// authoritative GipSAT frame solvers after replaying one full-root response.
/// This is a simulation qualification oracle, not a production double-check.
pub fn audit_resident_full_root_formula(solvers: &[&DagCnfSolver]) -> Result<(), String> {
    if std::env::var_os("INDUCTOR_CDCL_BLOCK_FULL_ROOT_FORMULA_ORACLE").is_none() {
        return Ok(());
    }
    let state = active_state()
        .lock()
        .map_err(|_| "resident full-root formula oracle hardware lock poisoned".to_string())?;
    let loaded = state
        .loaded_context
        .as_ref()
        .ok_or_else(|| "resident full-root formula oracle has no physical context".to_string())?;
    for solver in solvers {
        let (n_var, frame, cpu_clauses) = solver.incremental_resident_snapshot();
        compare_resident_formula_view(loaded, n_var, frame, &cpu_clauses)?;
    }
    Ok(())
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
    loaded_context: Option<LoadedContext>,
}

#[derive(Clone, Debug, Default)]
struct PairedCpuWork {
    status: u32,
    elapsed_ns: u64,
    decisions: u64,
    conflicts: u64,
    propagations: u64,
    /// Exact GipSAT payload for native semantic replay.  CSV consumers keep
    /// using the scalar fields above; the binary replay stream additionally
    /// uses the model/core as an independently computed oracle.
    result: Option<IncrementalResult>,
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
static ACTIVE_HARDWARE_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
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
static ACTIVE_SHARED_DOMAIN_PROJECTED_QUERIES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_SHARED_DOMAIN_PROJECTED_BATCHES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_SHARED_DOMAIN_PROJECTED_SAVED_WORDS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_CONTEXT_LOADS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_CONTEXT_APPENDS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_CONTEXT_APPEND_CLAUSES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_CONTEXT_APPEND_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_COMBINED_BATCHES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_COMBINED_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_HW_SAT: AtomicU64 = AtomicU64::new(0);
static ACTIVE_HW_UNSAT: AtomicU64 = AtomicU64::new(0);
static ACTIVE_UNKNOWN: AtomicU64 = AtomicU64::new(0);
static ACTIVE_ERROR: AtomicU64 = AtomicU64::new(0);
static ACTIVE_TRANSPORT_UNAVAILABLE: AtomicBool = AtomicBool::new(false);
static ACTIVE_HARDWARE_DISABLED: AtomicBool = AtomicBool::new(false);
static ACTIVE_HARDWARE_DISABLES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_UNAVAILABLE_CALLS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_UNAVAILABLE_QUERIES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_SAT_USED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_SAT_REJECTED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_TRUSTED_SAT_INSTALLED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_TRUSTED_SAT_REJECTED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_TRUSTED_SAT_STALE_REVALIDATED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_TRUSTED_SAT_REVISION_REUSED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MATERIALIZED_SAT_PREPARED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MATERIALIZED_SAT_REJECTED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MATERIALIZED_SAT_USED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MATERIALIZED_SAT_PREPARE_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_SAT_LIFT_ATTEMPTED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_SAT_LIFT_SUCCEEDED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_SAT_FULL_LITS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_SAT_LIFTED_LITS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_SAT_LIFT_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_UNSAT_CORE_USED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_UNSAT_CORE_REJECTED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_TRUSTED_UNSAT_INSTALLED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_TRUSTED_UNSAT_REJECTED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_UNSAT_ASSUMPTION_LITS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_UNSAT_HW_CORE_LITS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_UNSAT_CPU_CORE_LITS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_CPU_FALLBACK: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_COST_REJECTED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_CPU_SAMPLES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_CPU_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_CALIBRATIONS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_CALIBRATION_PROFITABLE: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_CALIBRATION_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_ROUTE_ENABLES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_ROUTE_DISABLES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_ROUTE_REPRESENTATIVE_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_ROUTE_ENABLED: AtomicBool = AtomicBool::new(false);
static ACTIVE_BLOCK_BATCH_ECON_PROBES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_BATCH_ECON_OFFLOADS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_BATCH_ECON_REJECTS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_BATCH_ECON_CPU_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_BATCH_ECON_HW_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_BATCH_ECON_HW_VALID: AtomicBool = AtomicBool::new(false);
static ACTIVE_BLOCK_BATCH_ECON_ROUTE: AtomicBool = AtomicBool::new(false);
const DEFAULT_BLOCK_BATCH_ECONOMICS: bool = true;
static ACTIVE_BLOCK_HW_CONCLUSIVE: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_SELECTED_NO_ANSWER: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_RESULT_USED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_RESULT_REJECTED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_CACHE_REUSED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_CACHE_REUSE_AGE: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_CACHE_REUSE_USED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_CACHE_REUSE_REJECTED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_CACHE_REPLACED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_CACHE_EVICTED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_PREFLIGHT_CONCLUSIVE: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_PREFLIGHT_SELECTED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_PREFLIGHT_FALLBACK: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_WAVE_RESERVED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_WAVE_TAKEN: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_ASYNC_LAUNCHED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_ASYNC_HARVESTED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_ASYNC_DISCARDED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_ASYNC_CPU_RACES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_ASYNC_ROOT_TAIL: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_ASYNC_ROOT_UNUSED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_ASYNC_DEMANDS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_ASYNC_DEMAND_READY: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_ASYNC_DEMAND_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_BROKER_GROUPS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_BROKER_JOBS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_BROKER_QUERIES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_BROKER_QUEUE_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_BROKER_REPLY_ERRORS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_BROKER_STREAM_REPLIES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_BROKER_REPLY_TAIL_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_ASYNC_PREPARE_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_ASYNC_WALL_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BLOCK_ASYNC_JOIN_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PUSH_PREFETCH_LAUNCHED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PUSH_PREFETCH_QUERIES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PUSH_PREFETCH_HARVESTED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PUSH_PREFETCH_READY: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PUSH_PREFETCH_HITS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PUSH_PREFETCH_USED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PUSH_PREFETCH_REJECTED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PUSH_PREFETCH_EVICTED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PUSH_PREFETCH_BUSY: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PUSH_PREFETCH_PREPARE_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PUSH_PREFETCH_WALL_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PUSH_PREFETCH_JOIN_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PUSH_PREFETCH_ADMITTED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PUSH_PREFETCH_SUPPRESSED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PUSH_PREFETCH_REPROBES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PUSH_PREFETCH_SKIPPED_LONG: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PUSH_PREFETCH_SKIPPED_CONTEXT: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PUSH_PREFETCH_EVAL_QUERIES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PUSH_PREFETCH_EVAL_USED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PUSH_PREFETCH_SUBMITTED_BY_LEN: [AtomicU64; 5] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static ACTIVE_PUSH_PREFETCH_READY_BY_LEN: [AtomicU64; 5] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static ACTIVE_PUSH_PREFETCH_HITS_BY_LEN: [AtomicU64; 5] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static ACTIVE_PUSH_PREFETCH_USED_BY_LEN: [AtomicU64; 5] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static ACTIVE_MIC_BATCH_WAVES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_BATCH_QUERIES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_BATCH_SAT: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_BATCH_UNSAT: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_BATCH_UNKNOWN: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_BATCH_SAT_USED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_BATCH_UNSAT_USED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_BATCH_REJECTED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_BATCH_INVALIDATED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_BATCH_SHADOW_REACHED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_BATCH_SHADOW_REPLACEABLE: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_BATCH_SHADOW_INVALIDATED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_BATCH_ECON_PROBES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_BATCH_ECON_OFFLOADS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_BATCH_ECON_REJECTS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_BATCH_ECON_CPU_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_BATCH_ECON_HW_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_BATCH_ECON_HW_VALID: AtomicBool = AtomicBool::new(false);
static ACTIVE_MIC_CHAIN_COMMANDS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_CHAIN_INPUT_LITS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_CHAIN_OUTPUT_LITS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_CHAIN_TRIALS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_CHAIN_PHYSICAL_ROUNDS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_CHAIN_COMPLETE: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_CHAIN_PARTIAL: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_CHAIN_ERRORS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_CHAIN_SERVICE_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_CHAIN_KERNEL_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_FUSED_APPEND_MIC_COMMANDS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_FUSED_APPEND_MIC_CLAUSES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_FUSED_APPEND_MIC_SERVICE_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_FUSED_APPEND_MIC_KERNEL_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_FRAME_MATERIALIZATIONS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_FRAME_MATERIALIZE_SERVICE_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_FRAME_MATERIALIZE_KERNEL_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_CHAIN_DECISIONS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_CHAIN_CONFLICTS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_CHAIN_PROPAGATIONS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_CHAIN_LEARNTS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_CHAIN_VALIDATED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_CHAIN_REJECTED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_CHAIN_VERIFY_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_CHAIN_CPU_LOOPS_REPLACED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_CHAIN_CPU_TRIALS_REPLACED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_CHAIN_ECON_PROBES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_CHAIN_ECON_OFFLOADS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_CHAIN_ECON_REJECTS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_CHAIN_ECON_WARM_PROBES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_CHAIN_ECON_WARM_OFFLOADS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_CHAIN_ECON_WARM_REJECTS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_CHAIN_ECON_CPU_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_CHAIN_ECON_HW_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_CHAIN_ECON_HW_VALID: AtomicBool = AtomicBool::new(false);
static ACTIVE_MIC_CHAIN_CLIENT_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MIC_CHAIN_CONTEXT_REUSED: AtomicU64 = AtomicU64::new(0);
static ACTIVE_INIT_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_STATE_WAIT_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_CONTEXT_LOAD_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_CONTEXT_APPEND_NS: AtomicU64 = AtomicU64::new(0);
// Direct-XRT kernel intervals are recorded separately from the end-to-end
// command timers above.  This keeps DMA/packing/lock time out of the FPGA
// occupancy numerator and makes resident maintenance directly comparable with
// useful CDCL execution. RPC mode reports zero because the kernel is owned by
// the server process.
static ACTIVE_CONTEXT_LOAD_KERNEL_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_CONTEXT_APPEND_KERNEL_NS: AtomicU64 = AtomicU64::new(0);
static FRAME_RANGE_SNAPSHOT_MISMATCH: AtomicU64 = AtomicU64::new(0);
static ACTIVE_COMBINED_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_COMBINED_KERNEL_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_COMBINED_FALLBACK_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BATCH_NS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_BATCH_KERNEL_NS: AtomicU64 = AtomicU64::new(0);
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
static ACTIVE_SAMPLE_CONTEXT_GROUPS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_SAMPLE_PLAN_NS: AtomicU64 = AtomicU64::new(0);
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
static PAIRED_WRITER: std::sync::OnceLock<Option<std::sync::Mutex<BufWriter<std::fs::File>>>> =
    std::sync::OnceLock::new();
static ARCH_TRACE_WRITER: std::sync::OnceLock<Option<std::sync::Mutex<BufWriter<std::fs::File>>>> =
    std::sync::OnceLock::new();
static ROOT_TRACE_WRITER: std::sync::OnceLock<Option<std::sync::Mutex<BufWriter<std::fs::File>>>> =
    std::sync::OnceLock::new();
static ARCH_TRACE_BATCH_ID: AtomicU64 = AtomicU64::new(0);
static ARCH_TRACE_QUERIES: AtomicU64 = AtomicU64::new(0);
static ROOT_TRACE_ROOTS: AtomicU64 = AtomicU64::new(0);
static EXACT_REPLAY_WRITER: std::sync::OnceLock<
    Option<std::sync::Mutex<BufWriter<std::fs::File>>>,
> = std::sync::OnceLock::new();
static FULL_ROOT_TRANSCRIPT_WRITER: std::sync::OnceLock<
    Option<std::sync::Mutex<BufWriter<std::fs::File>>>,
> = std::sync::OnceLock::new();
static FULL_ROOT_TRANSCRIPT_COMMANDS: AtomicU64 = AtomicU64::new(0);
static FULL_ROOT_TRANSCRIPT_REQUEST_WORDS: AtomicU64 = AtomicU64::new(0);
static FULL_ROOT_TRANSCRIPT_RESPONSE_WORDS: AtomicU64 = AtomicU64::new(0);
static FULL_ROOT_WIRE_REJECTS: AtomicU64 = AtomicU64::new(0);
static FULL_ROOT_STEP_CAPS: AtomicU64 = AtomicU64::new(0);
static EXACT_REPLAY_BATCHES: AtomicU64 = AtomicU64::new(0);
static EXACT_REPLAY_QUERIES: AtomicU64 = AtomicU64::new(0);
static EXACT_REPLAY_MICS: AtomicU64 = AtomicU64::new(0);
static EXACT_REPLAY_BLOCK_PROGRESS: AtomicU64 = AtomicU64::new(0);
static EXACT_REPLAY_BLOCK_EVENTS: AtomicU64 = AtomicU64::new(0);
static EXACT_REPLAY_FRAME_EVENTS: AtomicU64 = AtomicU64::new(0);
static EXACT_REPLAY_ROOTS: std::sync::OnceLock<std::sync::Mutex<HashSet<u32>>> =
    std::sync::OnceLock::new();
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
const DEFAULT_FULL_ROOT_MAX_RESPONSE_WORDS: usize = 1 << 16;
const DEFAULT_SHADOW_CONFLICT_BUDGET: u32 = 3;
const DEFAULT_ACTIVE_CONFLICT_BUDGET: u32 = 16;
const DEFAULT_BLOCK_FULL_ROOT_CONFLICT_BUDGET: u32 = 128;
const DEFAULT_BLOCK_FULL_ROOT_DECISION_BUDGET: u32 = 4096;

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

/// A resident root amortizes one slightly deeper short inquiry across an
/// entire on-device BLOCK traversal.  Keep this budget independent from the
/// leaf-batch cap: native multi-AIGER sweeps show that 16/32 conflicts cause
/// repeated CPU handoffs, 64 still misses the mod3/token tail, while 128 keeps
/// handoff below the simulation gate without paying the 256-conflict cap.
pub fn block_full_root_conflict_budget() -> u32 {
    static BUDGET: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| {
        configured_conflict_budget(
            "INDUCTOR_CDCL_BLOCK_FULL_ROOT_CONFLICT_BUDGET",
            DEFAULT_BLOCK_FULL_ROOT_CONFLICT_BUDGET,
        )
    })
}

/// Conflict-only limits do not bound low-conflict SAT walks. Keep one root
/// inquiry finite so a resident actor can hand an unsuitable tail back to
/// GipSAT instead of monopolizing the synchronous FPGA service.
pub fn block_full_root_decision_budget() -> u32 {
    static BUDGET: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_BLOCK_FULL_ROOT_DECISION_BUDGET")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(DEFAULT_BLOCK_FULL_ROOT_DECISION_BUDGET)
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

fn throughput_enabled() -> bool {
    HardwareCdcl::throughput_enabled()
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
                "inductor-cdcl-stage: batch-index {} frame {} status {} reason {} assumptions {} constraints {} domain {} total-entries {} setup {} root {} propagate {} analyze {} backtrack {} learn {} decide {} emit {} cleanup {} occurrence-updates {} partial-occurrence-scans {} evaluated-literals {} unit-candidates {} analyzed-literals {} undo-occurrences {} undo-assignments {} learnt-literals {} occurrence-rounds {} occurrence-pairs {}",
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
                counters[PROFILE_OCCURRENCE_ROUNDS],
                counters[PROFILE_OCCURRENCE_PAIRS],
            );
        }
    }
    if !profile_enabled() || queries.len() != work.len() {
        return;
    }
    for (batch_index, (query, work)) in queries.iter().zip(work).enumerate() {
        let assumptions = query.assumptions.len() as u64;
        let constraint_literals = query
            .constraints
            .iter()
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
        }
        .fetch_add(1, Ordering::Relaxed);
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

pub fn active_min_batch_size() -> usize {
    let fpga_throughput = std::env::var("INDUCTOR_CDCL_FPGA_THROUGHPUT")
        .ok()
        .is_some_and(|value| !matches!(value.as_str(), "0" | "false" | "off"));
    static SIZE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SIZE.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_ACTIVE_MIN_BATCH")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .map(|value| value.clamp(1, DEFAULT_SHADOW_BATCH_SIZE))
            // The measured batch-1 round trip is ~48 us while these GipSAT
            // push inquiries average only a few microseconds. Do not program
            // the card for a handful of queries by default.
            .unwrap_or(if fpga_throughput { 1 } else { 32 })
    })
}

pub fn block_batch_economics_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_BLOCK_BATCH_ECONOMICS")
            .ok()
            .map(|value| !matches!(value.as_str(), "0" | "false" | "off"))
            // Learn the end-to-end batch cost and fall back to exact GipSAT
            // when the VCK5000 cannot repay it. Set the variable to 0 only
            // for controlled comparisons with the legacy per-query route.
            .unwrap_or(DEFAULT_BLOCK_BATCH_ECONOMICS)
    })
}

pub fn block_min_batch_cpu_ns() -> u64 {
    static NS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *NS.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_BLOCK_MIN_BATCH_CPU_NS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            // A full batch below this aggregate CPU cost cannot amortize the
            // measured VCK5000 service and transport floor. The adaptive
            // hardware estimate takes over after the first bounded probe.
            .unwrap_or(2_500_000)
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
            (((2u128 * sample as u128 + 1) * n_candidates as u128) / (2u128 * n_sample as u128))
                as usize
        })
        .collect()
}

fn sample_keeps_fpga(
    sample_solve_ns: &[u64],
    n_remaining: usize,
    min_batch: usize,
    min_cpu_ns: u64,
    min_batch_cpu_ns: Option<u64>,
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
    let aggregate_profitable = min_batch_cpu_ns
        .is_some_and(|minimum| representative_ns.saturating_mul(n_remaining as u64) >= minimum);
    all_conclusive
        && n_remaining >= min_batch
        && (representative_ns >= min_cpu_ns || aggregate_profitable)
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
/// clones and use a representative restart cost to route the rest of each
/// context-compatible group in this propagation pass.
/// Cloning keeps the sample from changing live phase/activity and therefore
/// changing the cost it is trying to predict. Sampled answers are retained as
/// trusted CPU results rather than discarded. Samples are spread across the
/// compatible group and the lower median rejects a batch dominated by cheap
/// inquiries even if one prefix inquiry is an expensive outlier. Planning is
/// deliberately bounded by one propagation pass: carrying queries across
/// passes would cross IC3 frame mutations and invalidate their proof context.
pub fn active_sample_select_pass(
    requests: &[(&DagCnfSolver, &IncrementalQuery)],
    decisions: &mut [ActivePreflight],
) {
    let requested = active_sample_queries();
    if requested == 0 || requests.len() != decisions.len() {
        return;
    }

    struct SampleGroup {
        context: ShadowContext,
        candidates: Vec<(usize, usize)>,
    }

    let plan_start = std::time::Instant::now();
    let prefer_query_lemmas = !(active_resident_lemmas() || active_frame_ranges());
    let mut caches = Vec::new();
    let mut groups: Vec<SampleGroup> = Vec::new();
    for (index, ((solver, query), decision)) in
        requests.iter().zip(decisions.iter_mut()).enumerate()
    {
        if !matches!(decision, ActivePreflight::Fpga) {
            continue;
        }
        let cache_index = batched_solver_cache_index(&mut caches, solver);
        let Some((use_query_lemmas, words)) =
            caches[cache_index].query_plan(query, prefer_query_lemmas)
        else {
            *decision = ActivePreflight::CpuFallback;
            ACTIVE_SAMPLE_UNDERSIZED_REJECTED.fetch_add(1, Ordering::Relaxed);
            continue;
        };
        if words
            .checked_add(4)
            .is_none_or(|total| total > KERNEL_MAX_REQUEST_WORDS)
        {
            *decision = ActivePreflight::CpuFallback;
            ACTIVE_SAMPLE_UNDERSIZED_REJECTED.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        let context = caches[cache_index].context(use_query_lemmas);
        if let Some(group) = groups.iter_mut().find(|group| &group.context == context) {
            group.candidates.push((index, words));
        } else {
            groups.push(SampleGroup {
                context: context.clone(),
                candidates: vec![(index, words)],
            });
        }
    }
    ACTIVE_SAMPLE_CONTEXT_GROUPS.fetch_add(groups.len() as u64, Ordering::Relaxed);
    ACTIVE_SAMPLE_PLAN_NS.fetch_add(
        plan_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        Ordering::Relaxed,
    );

    for group in groups {
        let n_sample = requested.min(group.candidates.len());
        if group.candidates.len().saturating_sub(n_sample) < active_min_batch_size() {
            ACTIVE_SAMPLE_UNDERSIZED_REJECTED
                .fetch_add(group.candidates.len() as u64, Ordering::Relaxed);
            for (index, _) in group.candidates {
                decisions[index] = ActivePreflight::CpuFallback;
            }
            continue;
        }

        let sample_positions = representative_sample_positions(group.candidates.len(), n_sample);
        let mut sampled = vec![false; group.candidates.len()];
        let mut sample_ns = 0u64;
        let mut sample_clone_ns = 0u64;
        let mut sample_solve_ns = 0u64;
        let mut sample_solve_distribution = Vec::with_capacity(n_sample);
        let mut all_conclusive = true;
        for position in sample_positions {
            sampled[position] = true;
            let index = group.candidates[position].0;
            let (solver, query) = requests[index];
            let start = std::time::Instant::now();
            let mut sample_solver = solver.clone();
            let clone_ns = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            let solve_start = std::time::Instant::now();
            let result = sample_solver.classify_incremental_exact(query);
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

        let remaining: Vec<(usize, usize)> = group
            .candidates
            .into_iter()
            .enumerate()
            .filter_map(|(position, candidate)| (!sampled[position]).then_some(candidate))
            .collect();
        if !sample_keeps_fpga(
            &sample_solve_distribution,
            remaining.len(),
            active_min_batch_size(),
            active_sample_min_cpu_ns(),
            block_batch_economics_enabled().then_some(block_min_batch_cpu_ns()),
            all_conclusive,
        ) {
            ACTIVE_SAMPLE_CPU_BATCHES.fetch_add(1, Ordering::Relaxed);
            ACTIVE_SAMPLE_CPU_REJECTED.fetch_add(remaining.len() as u64, Ordering::Relaxed);
            for (index, _) in remaining {
                decisions[index] = ActivePreflight::CpuFallback;
            }
            continue;
        }

        let query_words: Vec<_> = remaining.iter().map(|(_, words)| *words).collect();
        let ranges = plan_full_batch_ranges(
            &query_words,
            active_min_batch_size(),
            active_batch_size(),
            KERNEL_MAX_REQUEST_WORDS,
        );
        let mut planned = vec![false; remaining.len()];
        for range in &ranges {
            planned[range.clone()].fill(true);
        }
        let planned_count = planned.iter().filter(|planned| **planned).count();
        ACTIVE_SAMPLE_FPGA_BATCHES.fetch_add(ranges.len() as u64, Ordering::Relaxed);
        ACTIVE_SAMPLE_FPGA_RETAINED.fetch_add(planned_count as u64, Ordering::Relaxed);
        ACTIVE_SAMPLE_UNDERSIZED_REJECTED.fetch_add(
            remaining.len().saturating_sub(planned_count) as u64,
            Ordering::Relaxed,
        );
        for ((index, _), planned) in remaining.into_iter().zip(planned) {
            if !planned {
                decisions[index] = ActivePreflight::CpuFallback;
            }
        }
    }
}

/// Optional measurement-only filter. It lets experiments test whether an
/// observable query class packs into profitable FPGA batches without changing
/// the result used by IC3. Production active mode intentionally ignores it.
fn paired_static_selected(query: &IncrementalQuery) -> bool {
    query.frame >= paired_min_frame() && query.assumptions.len() <= paired_max_assumptions()
}

fn pair_scheduler_setting(value: Option<&str>, throughput: bool) -> bool {
    value
        .map(|value| !matches!(value, "0" | "false" | "off"))
        .unwrap_or(throughput)
}

fn pair_scheduler_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        let setting = std::env::var("INDUCTOR_CDCL_PAIR_SCHEDULER").ok();
        // A single multiplier diagnostic originally regressed, so this
        // remained research-only. The fixed ten-AIGER qualification then
        // reproduced lower completed-model wall time on both the 125 MHz
        // production image (-8.1%) and the widened 120 MHz candidate (-12.4%).
        // Enable it only for the explicitly qualified throughput profile;
        // ordinary active/shadow diagnostics preserve caller order, and the
        // explicit switch remains an exact opt-out.
        pair_scheduler_setting(setting.as_deref(), HardwareCdcl::throughput_enabled())
    })
}

fn heterogeneous_full_lanes_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_HETEROGENEOUS_FULL_LANES")
            .ok()
            .is_some_and(|value| !matches!(value.as_str(), "0" | "false" | "off"))
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

fn active_frame_ranges() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_ACTIVE_FRAME_RANGES")
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
            // The local-lemma representation is useful only while a complete
            // minimum-size hardware batch fits in one command. Otherwise use
            // the exact resident-frame representation instead of satisfying
            // the global candidate guard and then emitting tiny DMA batches.
            .unwrap_or(KERNEL_MAX_REQUEST_WORDS / active_min_batch_size())
    })
}

/// Estimate the work that one query contributes to a two-lane round. This is
/// deliberately cheap: assumptions create propagation events, temporary
/// literals must be inserted into the private clause arena, and domain
/// variables may participate in decision selection. The score is only used to
/// pair independent inquiries; it cannot change their formula or answer.
fn query_work_score(query: &IncrementalQuery) -> u64 {
    let constraint_literals = query.constraints.iter().fold(0u64, |total, clause| {
        total.saturating_add(clause.len() as u64)
    });
    (query.assumptions.len() as u64)
        .saturating_mul(16)
        .saturating_add(constraint_literals.saturating_mul(4))
        .saturating_add(query.domain.len() as u64)
}

const COMPACT_FULL_LANE_LOCAL_CLAUSES: usize = 512;
const COMPACT_FULL_LANE_LOCAL_LITERALS: usize = 8192;

fn query_private_literal_count(query: &IncrementalQuery) -> usize {
    query
        .constraints
        .iter()
        .fold(0usize, |total, clause| total.saturating_add(clause.len()))
}

fn query_exceeds_compact_private_arena(query: &IncrementalQuery) -> bool {
    query.constraints.len() > COMPACT_FULL_LANE_LOCAL_CLAUSES
        || query_private_literal_count(query) > COMPACT_FULL_LANE_LOCAL_LITERALS
}

/// Rank pressure on the two independent private-arena dimensions. Multiplying
/// by the opposite compact-lane bound compares each query's fraction of the
/// 512-clause / 8192-literal arena without floating point. This score is used
/// only when neither member of an already-selected pair fits the compact lane.
fn query_private_arena_score(query: &IncrementalQuery) -> u64 {
    let constraint_literals = query_private_literal_count(query) as u64;
    (query.constraints.len() as u64)
        .saturating_mul(COMPACT_FULL_LANE_LOCAL_LITERALS as u64)
        .max(constraint_literals.saturating_mul(COMPACT_FULL_LANE_LOCAL_CLAUSES as u64))
}

/// Two engines execute adjacent requests concurrently, so pairing the largest
/// estimates together minimizes the sum of pair maxima for a fixed set of
/// scores. The caller keeps an original index alongside each active query and
/// restores result order after the FPGA returns.
fn schedule_query_pairs_for_layout<T>(
    pending: &mut [T],
    query_of: impl Fn(&T) -> &IncrementalQuery,
    heterogeneous_full_lanes: bool,
) {
    pending.sort_by_cached_key(|item| std::cmp::Reverse(query_work_score(query_of(item))));
    if heterogeneous_full_lanes {
        for pair in pending.chunks_mut(2) {
            if pair.len() != 2 {
                continue;
            }
            let first_exceeds = query_exceeds_compact_private_arena(query_of(&pair[0]));
            let second_exceeds = query_exceeds_compact_private_arena(query_of(&pair[1]));
            let swap = (first_exceeds && !second_exceeds)
                || (first_exceeds
                    && second_exceeds
                    && query_private_arena_score(query_of(&pair[0]))
                        > query_private_arena_score(query_of(&pair[1])));
            if swap {
                // Batch order maps the first record to the compact lane 0 and
                // the second to the capacity lane 1. Preserve the qualified
                // default order when both queries fit lane 0: merely swapping
                // fitting queries changes lane-local learnt/search history.
                // Original result indices travel with each query and are
                // restored by the caller.
                pair.swap(0, 1);
            }
        }
    }
}

fn schedule_query_pairs<T>(pending: &mut [T], query_of: impl Fn(&T) -> &IncrementalQuery) {
    schedule_query_pairs_for_layout(pending, query_of, heterogeneous_full_lanes_enabled());
}

fn query_request_words(query: &IncrementalQuery) -> Option<usize> {
    let constraints = query
        .constraints
        .iter()
        .try_fold(0usize, |words, clause| words.checked_add(1 + clause.len()))?;
    8usize
        .checked_add(query.assumptions.len())?
        .checked_add(constraints)?
        .checked_add(encoded_domain_words(&query.domain))
}

/// Partition one context-compatible query sequence into complete hardware
/// batches. Every returned range satisfies the configured query-count and
/// command-word limits. A bounded dynamic program maximizes the number of
/// planned queries, then minimizes submission count; any queries that cannot
/// participate in a complete batch are intentionally left for CPU fallback.
///
/// Ranges preserve caller order. Positions omitted between ranges correspond
/// to a query that could not participate in a full batch without exceeding the
/// command buffer.
fn plan_full_batch_ranges(
    query_words: &[usize],
    min_batch: usize,
    max_batch: usize,
    max_words: usize,
) -> Vec<std::ops::Range<usize>> {
    if query_words.is_empty() || min_batch == 0 || min_batch > max_batch || max_words < 4 {
        return Vec::new();
    }
    let n_query = query_words.len();
    let mut covered = vec![0usize; n_query + 1];
    let mut batches = vec![0usize; n_query + 1];
    let mut take = vec![0usize; n_query];
    for start in (0..n_query).rev() {
        // Skipping this query is always a valid CPU-fallback plan.
        covered[start] = covered[start + 1];
        batches[start] = batches[start + 1];
        let mut words = 4usize;
        let limit = n_query.min(start.saturating_add(max_batch));
        for end in start..limit {
            let Some(next) = words.checked_add(query_words[end]) else {
                break;
            };
            if next > max_words {
                break;
            }
            words = next;
            let len = end + 1 - start;
            if len < min_batch {
                continue;
            }
            let candidate_covered = len.saturating_add(covered[end + 1]);
            let candidate_batches = 1usize.saturating_add(batches[end + 1]);
            if candidate_covered > covered[start]
                || candidate_covered == covered[start]
                    && (candidate_batches < batches[start]
                        || candidate_batches == batches[start] && len > take[start])
            {
                covered[start] = candidate_covered;
                batches[start] = candidate_batches;
                take[start] = len;
            }
        }
    }

    let mut ranges = Vec::with_capacity(batches[0]);
    let mut start = 0usize;
    while start < n_query {
        let len = take[start];
        if len == 0 {
            start += 1;
        } else {
            ranges.push(start..start + len);
            start += len;
        }
    }
    ranges
}

/// Architecture-only planner for an ABI where one identical decision domain
/// is carried once per batch instead of once per query. It does not alter the
/// production request. Live native/board telemetry uses it to decide whether
/// implementing the new wire command is worth an HLS iteration.
fn plan_shared_domain_batch_ranges(
    domains: &[&[Var]],
    query_words: &[usize],
    min_batch: usize,
    max_batch: usize,
    max_words: usize,
) -> Vec<std::ops::Range<usize>> {
    if domains.len() != query_words.len()
        || domains.is_empty()
        || min_batch == 0
        || min_batch > max_batch
        || max_words < 4
    {
        return Vec::new();
    }
    let n_query = domains.len();
    let mut covered = vec![0usize; n_query + 1];
    let mut batches = vec![0usize; n_query + 1];
    let mut take = vec![0usize; n_query];
    for start in (0..n_query).rev() {
        covered[start] = covered[start + 1];
        batches[start] = batches[start + 1];
        let shared_domain_words = encoded_domain_words(domains[start]);
        let Some(mut words) = 4usize.checked_add(shared_domain_words) else {
            continue;
        };
        let limit = n_query.min(start.saturating_add(max_batch));
        for end in start..limit {
            if domains[end] != domains[start] {
                break;
            }
            let Some(private_words) = query_words[end].checked_sub(shared_domain_words) else {
                break;
            };
            let Some(next) = words.checked_add(private_words) else {
                break;
            };
            if next > max_words {
                break;
            }
            words = next;
            let count = end + 1 - start;
            if count < min_batch {
                continue;
            }
            let candidate_covered = count.saturating_add(covered[end + 1]);
            let candidate_batches = 1usize.saturating_add(batches[end + 1]);
            if candidate_covered > covered[start]
                || candidate_covered == covered[start]
                    && (candidate_batches < batches[start]
                        || candidate_batches == batches[start] && count > take[start])
            {
                covered[start] = candidate_covered;
                batches[start] = candidate_batches;
                take[start] = count;
            }
        }
    }
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < n_query {
        let count = take[start];
        if count == 0 {
            start += 1;
        } else {
            ranges.push(start..start + count);
            start += count;
        }
    }
    ranges
}

/// Cache the resident partition once per live frame solver while one
/// propagation pass is being planned. `incremental_resident_partition`
/// returns owned clause vectors, so invoking it for every query turns a cheap
/// compatibility check into repeated copies of the complete transition CNF.
struct BatchedSolverContext {
    solver_addr: usize,
    n_var: u32,
    frame: u32,
    trans: Vec<logicrs::LitVec>,
    lemmas: Vec<logicrs::LitVec>,
    lemma_words: usize,
    shared: Option<ShadowContext>,
    exact: Option<ShadowContext>,
    ranged: Option<ShadowContext>,
}

impl BatchedSolverContext {
    fn new(solver: &DagCnfSolver) -> Self {
        let (n_var, frame, trans, lemmas) = solver.incremental_resident_partition();
        let lemma_words = lemmas.iter().fold(0usize, |total, clause| {
            total.saturating_add(1usize.saturating_add(clause.len()))
        });
        Self {
            solver_addr: std::ptr::from_ref(solver) as usize,
            n_var,
            frame,
            trans,
            lemmas,
            lemma_words,
            shared: None,
            exact: None,
            ranged: None,
        }
    }

    fn query_plan(
        &self,
        query: &IncrementalQuery,
        prefer_query_lemmas: bool,
    ) -> Option<(bool, usize)> {
        let base_words = query_request_words(query)?;
        let expanded_words = base_words.checked_add(self.lemma_words);
        let use_query_lemmas = prefer_query_lemmas
            && expanded_words
                .and_then(|words| words.checked_add(4))
                .is_some_and(|total| total <= KERNEL_MAX_REQUEST_WORDS)
            && expanded_words.is_some_and(|words| words <= query_lemma_word_limit());
        Some((
            use_query_lemmas,
            if use_query_lemmas {
                // The fit checks above prove this branch has a value.
                expanded_words.unwrap_or(base_words)
            } else {
                base_words
            },
        ))
    }

    fn prepare_query(
        &self,
        mut query: IncrementalQuery,
        use_query_lemmas: bool,
    ) -> IncrementalQuery {
        if use_query_lemmas {
            query.constraints.extend(self.lemmas.iter().cloned());
        }
        query
    }

    fn context(&mut self, use_query_lemmas: bool) -> &ShadowContext {
        if use_query_lemmas {
            self.shared.get_or_insert_with(|| ShadowContext {
                n_var: self.n_var,
                clauses: self
                    .trans
                    .iter()
                    .cloned()
                    .map(|literals| ResidentClause::new(0, u32::MAX, literals))
                    .collect(),
                scope: ShadowContextScope::SharedTransition,
            })
        } else if active_frame_ranges() {
            if let Some(context) =
                frame_ranged_context(self.n_var, &self.trans, self.frame, &self.lemmas)
            {
                // Refresh because IC3 can append a ranged lemma after this
                // per-pass solver cache was built.
                self.ranged = Some(context);
                return self.ranged.as_ref().unwrap();
            }
            self.exact.get_or_insert_with(|| ShadowContext {
                n_var: self.n_var,
                clauses: self
                    .trans
                    .iter()
                    .cloned()
                    .map(|literals| ResidentClause::new(0, u32::MAX, literals))
                    .chain(
                        self.lemmas
                            .iter()
                            .cloned()
                            .map(|literals| ResidentClause::new(0, u32::MAX, literals)),
                    )
                    .collect(),
                scope: ShadowContextScope::ExactFrame(self.frame),
            })
        } else {
            self.exact.get_or_insert_with(|| ShadowContext {
                n_var: self.n_var,
                clauses: self
                    .trans
                    .iter()
                    .cloned()
                    .map(|literals| ResidentClause::new(0, u32::MAX, literals))
                    .chain(
                        self.lemmas
                            .iter()
                            .cloned()
                            .map(|literals| ResidentClause::new(0, u32::MAX, literals)),
                    )
                    .collect(),
                scope: ShadowContextScope::ExactFrame(self.frame),
            })
        }
    }
}

fn batched_solver_cache_index(
    caches: &mut Vec<BatchedSolverContext>,
    solver: &DagCnfSolver,
) -> usize {
    let solver_addr = std::ptr::from_ref(solver) as usize;
    caches
        .iter()
        .position(|cache| cache.solver_addr == solver_addr)
        .unwrap_or_else(|| {
            caches.push(BatchedSolverContext::new(solver));
            caches.len() - 1
        })
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
                words <= KERNEL_MAX_REQUEST_WORDS && words <= query_lemma_word_limit()
            });
        if fits {
            let clauses = trans
                .into_iter()
                .map(|literals| ResidentClause::new(0, u32::MAX, literals))
                .collect();
            return (
                ShadowContext {
                    n_var,
                    clauses,
                    scope: ShadowContextScope::SharedTransition,
                },
                query,
                true,
            );
        }
        query.constraints.truncate(n_existing_constraints);
    }

    if let Some(context) = frame_ranged_context(n_var, &trans, frame, &lemmas) {
        return (context, query, false);
    }

    let clauses = trans
        .into_iter()
        .map(|literals| ResidentClause::new(0, u32::MAX, literals))
        .chain(
            lemmas
                .into_iter()
                .map(|literals| ResidentClause::new(0, u32::MAX, literals)),
        )
        .collect();
    (
        ShadowContext {
            n_var,
            clauses,
            scope: ShadowContextScope::ExactFrame(frame),
        },
        query,
        false,
    )
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
    selected: &[bool],
    apply_paired_filter: bool,
    retain_result: bool,
) -> Vec<PairedCpuWork> {
    requests
        .iter()
        .enumerate()
        .map(|(index, (solver, query))| {
            if !selected.get(index).copied().unwrap_or(false)
                || apply_paired_filter && !paired_static_selected(query)
            {
                return PairedCpuWork::default();
            }
            // Clone before starting the timer: the comparison is the work of
            // one independent GipSAT inquiry, not the cost of manufacturing a
            // profiling copy. Remove FPGA-only budgets for the exact CPU run.
            let mut cpu: DagCnfSolver = (**solver).clone();
            let mut cpu_query = query.clone();
            cpu_query.budget = QueryBudget::default();
            let start = std::time::Instant::now();
            let result = cpu.solve_incremental(&cpu_query);
            let elapsed_ns = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            let status = result_status(&result);
            PairedCpuWork {
                status,
                elapsed_ns,
                decisions: u64::from(cpu.probe.n_decide),
                conflicts: u64::from(cpu.probe.n_conflict),
                propagations: u64::from(cpu.probe.n_prop),
                result: retain_result.then_some(result),
            }
        })
        .collect()
}

// Version 2 adds phase/op_id/dependency metadata ahead of each batch or MIC
// context. Version 3 records the stable producer ticket used by the persistent
// ring model for every query. The server remains the sole SQ producer: it
// resolves the lease/epoch and assigns a physical lane before constructing the
// 64-byte ABI-v2 descriptor.
// Version 7 appends an event-owned semantic operation stream to every BLOCK
// instruction.  Unlike the version-6 image patch, these operands are emitted
// at the queue/frame mutation sites and can therefore drive a resident
// controller without reconstructing work from CPU post-images.
// Version 8 also emits compact event-only records for BLOCK roots that did not
// launch an exact SAT inquiry. This keeps the resident proof state continuous
// instead of accumulating their mutations into the next full checkpoint.
// Version 9 emits the direct extend/propagate/push-to-infinity maintenance
// stream between BLOCK roots.  Initial/final images are trace-only oracles;
// production consumes only the mutation commands.
const EXACT_REPLAY_VERSION: u32 = 9;
const EXACT_REPLAY_BATCH: u32 = 1;
const EXACT_REPLAY_MIC: u32 = 2;
const EXACT_REPLAY_BLOCK_PROGRESS_RECORD: u32 = 3;
const EXACT_REPLAY_BLOCK_EVENT_RECORD: u32 = 4;
const EXACT_REPLAY_FRAME_EVENT_RECORD: u32 = 5;
const RING_INDEPENDENT_SET: u32 = 1 << 3;
const RING_END_OF_BATCH: u32 = 1 << 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PersistentRingQueryTicket {
    batch_id: u32,
    position: u32,
    flags: u32,
    user_tag: u64,
}

fn persistent_ring_batch_id(pass_id: u64) -> u32 {
    let folded = (pass_id as u32) ^ (pass_id >> 32) as u32;
    folded.max(1)
}

fn persistent_ring_query_ticket(
    pass_id: u64,
    position: usize,
    end_of_batch: bool,
) -> PersistentRingQueryTicket {
    let batch_id = persistent_ring_batch_id(pass_id);
    let position = position.min(u32::MAX as usize) as u32;
    PersistentRingQueryTicket {
        batch_id,
        position,
        flags: RING_INDEPENDENT_SET | if end_of_batch { RING_END_OF_BATCH } else { 0 },
        user_tag: (u64::from(batch_id) << 32) | u64::from(position),
    }
}

pub struct ExactMicReplayCapture {
    context: ShadowContext,
    request: Vec<u32>,
}

pub struct ExactBlockProgressCapture {
    op_id: u32,
    frame: u32,
    obligations: (usize, u64),
    lemmas: (usize, u64),
    obligation_image: Vec<u32>,
    lemma_image: Vec<u32>,
    current_obligation_image: Vec<u32>,
    current_lemma_image: Vec<u32>,
    steps: Vec<ExactBlockProgressStep>,
}

/// Lightweight wall-clock boundary for discrete-event architecture replay.
/// Unlike exact replay, this does not clone proof images or change the live
/// solver. Query rows with the same macro_op_id provide the measured SAT part.
pub struct BlockRootTimelineCapture {
    op_id: u32,
    frame: u32,
    ready_unix_ns: u128,
    start: std::time::Instant,
}

pub struct ExactFrameEventCapture {
    wave_id: u32,
    input_obligations: (usize, u64),
    input_lemmas: (usize, u64),
    initial_obligation_image: Vec<u32>,
    initial_lemma_image: Vec<u32>,
}

static BLOCK_CONTROLLER_SIM_FAILED: AtomicBool = AtomicBool::new(false);

fn block_root_timeline_enabled() -> bool {
    std::env::var_os("INDUCTOR_CDCL_ROOT_TRACE_TSV").is_some()
}

fn block_root_timeline_writer() -> Option<&'static std::sync::Mutex<BufWriter<std::fs::File>>> {
    ROOT_TRACE_WRITER
        .get_or_init(|| {
            let path = worker_scoped_trace_path(std::env::var_os("INDUCTOR_CDCL_ROOT_TRACE_TSV")?);
            let file = match std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&path)
            {
                Ok(file) => file,
                Err(error) => {
                    eprintln!(
                        "inductor-cdcl: cannot create BLOCK root timeline {}: {error}",
                        path.display(),
                    );
                    return None;
                }
            };
            let mut writer = BufWriter::new(file);
            writeln!(
                writer,
                "op_id\tframe\tresult\tresult_aux\tready_unix_ns\tfinish_unix_ns\tcpu_root_ns"
            )
            .ok()?;
            Some(std::sync::Mutex::new(writer))
        })
        .as_ref()
}

pub fn begin_block_root_timeline(frame: usize) -> Option<BlockRootTimelineCapture> {
    if !block_root_timeline_enabled() {
        return None;
    }
    let (phase, op_id) = crate::inductor::current_macro_context();
    if phase != inductor_trace::Phase::Block || op_id == 0 {
        return None;
    }
    Some(BlockRootTimelineCapture {
        op_id,
        frame: frame.min(u32::MAX as usize) as u32,
        ready_unix_ns: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
        start: std::time::Instant::now(),
    })
}

pub fn finish_block_root_timeline(
    capture: Option<BlockRootTimelineCapture>,
    result: u32,
    result_aux: u32,
) {
    let Some(capture) = capture else { return };
    let elapsed_ns = capture.start.elapsed().as_nanos();
    let finish_unix_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let Some(writer) = block_root_timeline_writer() else {
        return;
    };
    let Ok(mut writer) = writer.lock() else {
        return;
    };
    if writeln!(
        writer,
        "{}\t{}\t{}\t{}\t{}\t{}\t{}",
        capture.op_id,
        capture.frame,
        result,
        result_aux,
        capture.ready_unix_ns,
        finish_unix_ns,
        elapsed_ns,
    )
    .is_ok()
    {
        ROOT_TRACE_ROOTS.fetch_add(1, Ordering::Relaxed);
    }
}

fn block_controller_sim_requested() -> bool {
    std::env::var("INDUCTOR_CDCL_BLOCK_CONTROLLER_SIM")
        .ok()
        .is_some_and(|value| !matches!(value.as_str(), "0" | "false" | "off"))
}

fn block_controller_sim_enabled() -> bool {
    block_controller_sim_requested() && !BLOCK_CONTROLLER_SIM_FAILED.load(Ordering::Relaxed)
}

fn block_controller_sim_strict() -> bool {
    std::env::var("INDUCTOR_CDCL_BLOCK_CONTROLLER_SIM_STRICT")
        .ok()
        .is_some_and(|value| !matches!(value.as_str(), "0" | "false" | "off"))
}

fn block_controller_owns_queue() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_BLOCK_CONTROLLER_OWNS_QUEUE")
            .ok()
            .is_some_and(|value| !matches!(value.as_str(), "0" | "false" | "off"))
    })
}

pub fn block_root_executor_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_BLOCK_ROOT_EXECUTOR")
            .ok()
            .is_some_and(|value| !matches!(value.as_str(), "0" | "false" | "off"))
    })
}

pub fn block_full_root_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_BLOCK_FULL_ROOT")
            .ok()
            .is_some_and(|value| !matches!(value.as_str(), "0" | "false" | "off"))
    })
}

fn full_root_wire_limit(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value != 0)
        .unwrap_or(default)
}

fn admitted_full_root_steps_with_limits(
    requested_steps: usize,
    n_var: usize,
    domain_words: usize,
    latch_count: usize,
    input_count: usize,
    max_request_words: usize,
    max_response_words: usize,
) -> Option<usize> {
    let request_words = crate::accel::cdcl::BLOCK_FULL_ROOT_REQUEST_HEADER_WORDS
        .checked_add(2usize.checked_mul(n_var)?)?
        .checked_add(domain_words)?
        .checked_add(latch_count)?
        .checked_add(input_count)?;
    if requested_steps == 0 || request_words > max_request_words {
        return None;
    }
    let fixed_response_words = crate::accel::cdcl::BLOCK_FULL_ROOT_RESPONSE_HEADER_WORDS
        .checked_add(crate::accel::cdcl::BLOCK_FULL_ROOT_WORK_WORDS)?;
    let lemma_words =
        crate::accel::cdcl::BLOCK_FULL_ROOT_LEMMA_HEADER_WORDS.checked_add(latch_count)?;
    let sat_words = crate::accel::cdcl::BLOCK_FULL_ROOT_SAT_HEADER_WORDS
        .checked_add(latch_count)?
        .checked_add(input_count)?;
    let event_words = crate::accel::cdcl::BLOCK_FULL_ROOT_EVENT_HEADER_WORDS
        .checked_add(lemma_words.max(sat_words))?;
    let journal_words = max_response_words.checked_sub(fixed_response_words)?;
    let admitted_steps = (journal_words / event_words)
        .min(requested_steps)
        .min(crate::accel::cdcl::BLOCK_FULL_ROOT_MAX_STEPS);
    (admitted_steps != 0).then_some(admitted_steps)
}

fn admitted_full_root_steps(
    requested_steps: usize,
    n_var: usize,
    domain_words: usize,
    latch_count: usize,
    input_count: usize,
) -> Option<usize> {
    admitted_full_root_steps_with_limits(
        requested_steps,
        n_var,
        domain_words,
        latch_count,
        input_count,
        full_root_wire_limit(
            "INDUCTOR_CDCL_BLOCK_FULL_ROOT_MAX_REQUEST_WORDS",
            KERNEL_MAX_REQUEST_WORDS,
        ),
        full_root_wire_limit(
            "INDUCTOR_CDCL_BLOCK_FULL_ROOT_MAX_RESPONSE_WORDS",
            DEFAULT_FULL_ROOT_MAX_RESPONSE_WORDS,
        ),
    )
}

pub enum ResidentBlockPop {
    Disabled,
    Empty,
    Selected { user_tag: u64 },
}

pub enum ResidentBlockRoot {
    Disabled,
    Wave {
        response: BlockRootResponse,
        keys: Vec<Vec<u32>>,
    },
}

pub enum ResidentBlockFullRoot {
    Disabled,
    Wave {
        response: BlockFullRootResponse,
        source_keys: Vec<Vec<u32>>,
    },
}

/// Execute one controller-owned resident queue/CDCL wave. The CPU-facing
/// return keeps opaque proof keys aligned with device work records; only the
/// first key is consumed from the queue, while later records remain cached
/// speculative inquiries.
pub fn run_resident_block_root(
    max_frame: usize,
    requested_queries: usize,
    solvers: &[&DagCnfSolver],
    next_var_by_current: &[u32],
    query_template: &IncrementalQuery,
) -> ResidentBlockRoot {
    if !block_root_executor_enabled()
        || !block_controller_owns_queue()
        || !block_controller_sim_enabled()
    {
        return ResidentBlockRoot::Disabled;
    }
    // A root wave may be the first CDCL operation in a BLOCK traversal.  The
    // semantic queue is a separate resident state machine, so successfully
    // rebasing it does not imply that the formula arena has been installed.
    // Build an exact ranged snapshot here and make residency an explicit part
    // of the root transaction's host-side precondition.
    let Some(context) = block_root_ranged_context(solvers) else {
        return ResidentBlockRoot::Disabled;
    };
    let state = active_state().lock();
    let result = state
        .map_err(|_| "active hardware lock poisoned".to_string())
        .and_then(|mut state| {
            let update = plan_context_update(state.loaded_context.as_ref(), &context);
            let ready = match update {
                ContextUpdate::Ready => Ok(()),
                ContextUpdate::Append(clauses) => {
                    let started = std::time::Instant::now();
                    let result = state
                        .hardware
                        .as_mut()
                        .ok_or(HardwareError::Unavailable)
                        .and_then(|hardware| hardware.add_frame_clauses(&clauses));
                    ACTIVE_CONTEXT_APPEND_NS.fetch_add(
                        started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                        Ordering::Relaxed,
                    );
                    if result.is_ok() {
                        ACTIVE_CONTEXT_APPENDS.fetch_add(1, Ordering::Relaxed);
                        ACTIVE_CONTEXT_APPEND_CLAUSES
                            .fetch_add(clauses.len() as u64, Ordering::Relaxed);
                        if let Some(loaded) = state.loaded_context.as_mut() {
                            loaded.clauses.extend(clauses.clone());
                        }
                    } else {
                        ACTIVE_CONTEXT_APPEND_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                    }
                    result
                }
                ContextUpdate::Reload => {
                    let started = std::time::Instant::now();
                    let result = state
                        .hardware
                        .as_mut()
                        .ok_or(HardwareError::Unavailable)
                        .and_then(|hardware| {
                            hardware.load_context(context.n_var, &context.clauses)
                        });
                    ACTIVE_CONTEXT_LOAD_NS.fetch_add(
                        started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                        Ordering::Relaxed,
                    );
                    if result.is_ok() {
                        ACTIVE_CONTEXT_LOADS.fetch_add(1, Ordering::Relaxed);
                        state.loaded_context = Some(LoadedContext::from(&context));
                    }
                    result
                }
            };
            if let Err(error) = ready {
                state.loaded_context = None;
                return Err(format!("resident BLOCK root context failed: {error}"));
            }
            let result = state
                .hardware
                .as_mut()
                .ok_or_else(|| "active hardware transport unavailable".to_string())
                .and_then(|hardware| {
                    super::block_controller_sim::run_owned_root(
                        hardware,
                        max_frame.min(u32::MAX as usize) as u32,
                        requested_queries,
                        next_var_by_current,
                        query_template,
                    )
                });
            if result.is_err() {
                // A failed command also means a shared service lease can no
                // longer be assumed.  The next safe attempt will reinstall
                // the exact ranged formula before touching the queue.
                state.loaded_context = None;
            }
            result
        });
    match result {
        Ok(wave) => ResidentBlockRoot::Wave {
            response: wave.response,
            keys: wave.keys,
        },
        Err(error) => {
            finish_block_controller_sim(Err(error));
            ResidentBlockRoot::Disabled
        }
    }
}

pub fn run_resident_block_full_root(
    max_frame: usize,
    step_limit: usize,
    solvers: &[&DagCnfSolver],
    next_var_by_current: &[u32],
    init_value_by_current: &[u32],
    latch_variables: &[u32],
    input_variables: &[u32],
    query_template: &IncrementalQuery,
    compacted_retry: bool,
    allow_predecessor_lift: bool,
) -> ResidentBlockFullRoot {
    if !block_full_root_enabled()
        || !block_controller_owns_queue()
        || !block_controller_sim_enabled()
    {
        return ResidentBlockFullRoot::Disabled;
    }
    let Some(admitted_step_limit) = admitted_full_root_steps(
        step_limit,
        next_var_by_current.len(),
        query_template.domain.len(),
        latch_variables.len(),
        input_variables.len(),
    ) else {
        FULL_ROOT_WIRE_REJECTS.fetch_add(1, Ordering::Relaxed);
        return ResidentBlockFullRoot::Disabled;
    };
    if admitted_step_limit != step_limit {
        FULL_ROOT_STEP_CAPS.fetch_add(1, Ordering::Relaxed);
    }
    let context = if std::env::var_os("INDUCTOR_CDCL_BLOCK_FULL_ROOT_EXACT_MAX_FRAME").is_some() {
        let Some(solver) = max_frame
            .checked_sub(1)
            .and_then(|frame| solvers.get(frame))
        else {
            return ResidentBlockFullRoot::Disabled;
        };
        let (n_var, frame, snapshot) = solver.incremental_resident_snapshot();
        ShadowContext {
            n_var,
            clauses: snapshot
                .into_iter()
                .map(|literals| ResidentClause::new(0, u32::MAX, literals))
                .collect(),
            scope: ShadowContextScope::ExactFrame(frame),
        }
    } else if let Some(context) = block_root_ranged_context(solvers) {
        context
    } else {
        return ResidentBlockFullRoot::Disabled;
    };
    let frontier_limit = std::env::var("INDUCTOR_CDCL_BLOCK_FULL_ROOT_LANES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(crate::accel::cdcl::BLOCK_ROOT_MAX_WORK)
        .clamp(1, crate::accel::cdcl::BLOCK_ROOT_MAX_WORK);
    // The reserved view must contain T alone, never Init or frame lemmas.
    // Constrained systems need constraints in the implication target too;
    // their caller deliberately keeps complete predecessors for now.
    let predecessor_lift = allow_predecessor_lift
        && context.scope == ShadowContextScope::FrameRanged
        && max_frame < u32::MAX as usize
        && std::env::var("INDUCTOR_CDCL_BLOCK_FULL_ROOT_PREDECESSOR_LIFT")
            .ok()
            .is_some_and(|value| !matches!(value.as_str(), "0" | "false" | "off"));
    let state = active_state().lock();
    let result = state
        .map_err(|_| "active hardware lock poisoned".to_string())
        .and_then(|mut state| {
            let update =
                if std::env::var_os("INDUCTOR_CDCL_BLOCK_FULL_ROOT_RELOAD_EACH_ROOT").is_some() {
                    ContextUpdate::Reload
                } else {
                    plan_context_update(state.loaded_context.as_ref(), &context)
                };
            let ready = match update {
                ContextUpdate::Ready => Ok(()),
                ContextUpdate::Append(clauses) => {
                    let result = state
                        .hardware
                        .as_mut()
                        .ok_or(HardwareError::Unavailable)
                        .and_then(|hardware| hardware.add_frame_clauses(&clauses));
                    if result.is_ok() {
                        if let Some(loaded) = state.loaded_context.as_mut() {
                            loaded.clauses.extend(clauses);
                        }
                    }
                    result
                }
                ContextUpdate::Reload => {
                    let result = state
                        .hardware
                        .as_mut()
                        .ok_or(HardwareError::Unavailable)
                        .and_then(|hardware| {
                            hardware.load_context(context.n_var, &context.clauses)
                        });
                    if result.is_ok() {
                        state.loaded_context = Some(LoadedContext::from(&context));
                    }
                    result
                }
            };
            if let Err(error) = ready {
                state.loaded_context = None;
                return Err(format!("resident BLOCK full-root context failed: {error}"));
            }
            let result = state
                .hardware
                .as_mut()
                .ok_or_else(|| "active hardware transport unavailable".to_string())
                .and_then(|hardware| {
                    super::block_controller_sim::run_owned_full_root(
                        hardware,
                        max_frame.min(u32::MAX as usize) as u32,
                        admitted_step_limit,
                        frontier_limit,
                        next_var_by_current,
                        init_value_by_current,
                        latch_variables,
                        input_variables,
                        query_template,
                        compacted_retry,
                        predecessor_lift,
                    )
                });
            if let Ok(wave) = &result
                && let Some(loaded) = state.loaded_context.as_mut()
            {
                // The resident controller has already appended every UNSAT
                // journal lemma to the physical formula before publishing the
                // response. Mirror those exact ranged clauses in the client
                // lease so the next root does not append them a second time.
                for event in &wave.response.events {
                    if let BlockFullRootEvent::UnsatLemma {
                        frame,
                        begin_frame,
                        cube,
                        ..
                    } = event
                        && begin_frame <= frame
                    {
                        loaded.clauses.push(ResidentClause::new(
                            *begin_frame,
                            *frame,
                            cube.iter()
                                .map(|literal| {
                                    let word = *literal ^ 1;
                                    Lit::new(Var::from(word >> 1), word & 1 == 0)
                                })
                                .collect::<LitVec>(),
                        ));
                    }
                }
            } else if result.is_err() {
                state.loaded_context = None;
            }
            result
        });
    match result {
        Ok(wave) => ResidentBlockFullRoot::Wave {
            response: wave.response,
            source_keys: wave.source_keys,
        },
        Err(error) => {
            if error.ends_with("incremental CDCL hardware error: Timeout") {
                BLOCK_CONTROLLER_SIM_FAILED.store(true, Ordering::Relaxed);
                if !ACTIVE_HARDWARE_DISABLED.swap(true, Ordering::Relaxed) {
                    ACTIVE_HARDWARE_DISABLES.fetch_add(1, Ordering::Relaxed);
                    eprintln!(
                        "inductor-cdcl: full-root service timed out; closing its resident lease and continuing this solver on CPU"
                    );
                }
                return ResidentBlockFullRoot::Disabled;
            }
            finish_block_controller_sim(Err(error));
            ResidentBlockFullRoot::Disabled
        }
    }
}

/// Let the resident controller select and remove the next BLOCK obligation.
/// Only the opaque tag crosses this scheduling boundary; proof payloads remain
/// resident on each side.
pub fn pop_resident_block_obligation(max_frame: usize) -> ResidentBlockPop {
    if !block_controller_owns_queue() || !block_controller_sim_enabled() {
        return ResidentBlockPop::Disabled;
    }
    let state = active_state().lock();
    let result = state
        .map_err(|_| "active hardware lock poisoned".to_string())
        .and_then(|mut state| {
            let hardware = state
                .hardware
                .as_mut()
                .ok_or_else(|| "active hardware transport unavailable".to_string())?;
            super::block_controller_sim::pop_owned(
                hardware,
                max_frame.min(u32::MAX as usize) as u32,
            )
        });
    match result {
        Ok(Some(user_tag)) => ResidentBlockPop::Selected { user_tag },
        Ok(None) => ResidentBlockPop::Empty,
        Err(error) => {
            finish_block_controller_sim(Err(error));
            ResidentBlockPop::Disabled
        }
    }
}

/// Resolve one controller-selected tag inside the simulation adapter. The
/// production host will own this registry when it assigns tags to insertions.
pub fn take_resident_block_selection(user_tag: u64) -> Option<Vec<u32>> {
    match super::block_controller_sim::take_owned_key(user_tag) {
        Ok(key) => Some(key),
        Err(error) => {
            finish_block_controller_sim(Err(error));
            None
        }
    }
}

fn finish_block_controller_sim(result: Result<(), String>) {
    if let Err(error) = result {
        if !BLOCK_CONTROLLER_SIM_FAILED.swap(true, Ordering::Relaxed) {
            eprintln!("inductor-cdcl: live BLOCK controller disabled: {error}");
        }
        assert!(
            !block_controller_sim_strict(),
            "live BLOCK controller: {error}"
        );
    }
}

fn reconcile_block_controller_sim(obligation_image: &[u32], lemma_image: &[u32]) {
    if !block_controller_sim_enabled() {
        return;
    }
    let state = active_state().lock();
    let result = state
        .map_err(|_| "active hardware lock poisoned".to_string())
        .and_then(|mut state| {
            let hardware = state
                .hardware
                .as_mut()
                .ok_or_else(|| "active hardware transport unavailable".to_string())?;
            super::block_controller_sim::reconcile(hardware, obligation_image, lemma_image)
        });
    finish_block_controller_sim(result);
}

fn apply_block_controller_sim(
    semantic_ops: &[Vec<u32>],
    obligation_image: &[u32],
    lemma_image: &[u32],
) {
    if !block_controller_sim_enabled() {
        return;
    }
    let state = active_state().lock();
    let result = state
        .map_err(|_| "active hardware lock poisoned".to_string())
        .and_then(|mut state| {
            let hardware = state
                .hardware
                .as_mut()
                .ok_or_else(|| "active hardware transport unavailable".to_string())?;
            super::block_controller_sim::apply(
                hardware,
                semantic_ops,
                obligation_image,
                lemma_image,
            )
        });
    finish_block_controller_sim(result);
}

struct ExactImagePatch {
    prefix: u32,
    suffix: u32,
    append: Vec<u32>,
}

struct ExactBlockProgressStep {
    event: u32,
    obligations: (usize, u64),
    lemmas: (usize, u64),
    obligation_patch: ExactImagePatch,
    lemma_patch: Option<ExactImagePatch>,
    semantic_ops: Vec<Vec<u32>>,
}

fn exact_replay_roots() -> &'static std::sync::Mutex<HashSet<u32>> {
    EXACT_REPLAY_ROOTS.get_or_init(Default::default)
}

fn exact_replay_limit() -> u64 {
    static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_EXACT_REPLAY_QUERIES")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(256)
    })
}

fn exact_mic_replay_limit() -> u64 {
    static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_EXACT_REPLAY_MICS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(16)
    })
}

fn scope_trace_path(
    path: std::ffi::OsString,
    worker: Option<&std::ffi::OsStr>,
) -> std::path::PathBuf {
    let mut path = std::path::PathBuf::from(path);
    let Some(worker) = worker else {
        return path;
    };
    let worker = worker
        .to_string_lossy()
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '_' | '-') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if worker.is_empty() {
        return path;
    }
    let stem = path
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| "trace".to_string());
    let extension = path
        .extension()
        .map(|extension| format!(".{}", extension.to_string_lossy()))
        .unwrap_or_default();
    path.set_file_name(format!("{stem}.{worker}{extension}"));
    path
}

fn worker_scoped_trace_path(path: std::ffi::OsString) -> std::path::PathBuf {
    let worker = std::env::var_os("INDUCTOR_CDCL_PORTFOLIO_WORKER");
    scope_trace_path(path, worker.as_deref())
}

fn exact_replay_writer() -> Option<&'static std::sync::Mutex<BufWriter<std::fs::File>>> {
    EXACT_REPLAY_WRITER
        .get_or_init(|| {
            let path = worker_scoped_trace_path(std::env::var_os("INDUCTOR_CDCL_EXACT_REPLAY")?);
            let file = match std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&path)
            {
                Ok(file) => file,
                Err(error) => {
                    eprintln!(
                        "inductor-cdcl: cannot create exact replay {}: {error}",
                        std::path::Path::new(&path).display(),
                    );
                    return None;
                }
            };
            let mut writer = BufWriter::new(file);
            // Eight-byte magic, stream version and the request ABI version.
            if writer.write_all(b"INDEXACT").is_err()
                || writer
                    .write_all(&EXACT_REPLAY_VERSION.to_le_bytes())
                    .is_err()
                || writer.write_all(&ABI_VERSION.to_le_bytes()).is_err()
            {
                return None;
            }
            Some(std::sync::Mutex::new(writer))
        })
        .as_ref()
}

/// Record the exact production-facing RUN_BLOCK_FULL_ROOT transaction.  This
/// is deliberately a separate stream from the CPU-oracle exact replay: it is
/// emitted only after the real native/RPC transport has returned, so every
/// word was observed at the same FFI boundary used by an xclbin.
fn full_root_transcript_writer() -> Option<&'static std::sync::Mutex<BufWriter<std::fs::File>>> {
    FULL_ROOT_TRANSCRIPT_WRITER
        .get_or_init(|| {
            let path =
                worker_scoped_trace_path(std::env::var_os("INDUCTOR_CDCL_FULL_ROOT_TRANSCRIPT")?);
            let file = match std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&path)
            {
                Ok(file) => file,
                Err(error) => {
                    eprintln!(
                        "inductor-cdcl: cannot create full-root transcript {}: {error}",
                        path.display(),
                    );
                    return None;
                }
            };
            let mut writer = BufWriter::new(file);
            // Eight-byte magic followed by the transcript format version.
            if writer.write_all(b"INDFROOT").is_err()
                || writer.write_all(&1u32.to_le_bytes()).is_err()
            {
                return None;
            }
            Some(std::sync::Mutex::new(writer))
        })
        .as_ref()
}

fn record_full_root_transaction(request: &[u32], response: &[u32], rc: i32) {
    let Some(writer) = full_root_transcript_writer() else {
        return;
    };
    let (_, op_id) = crate::inductor::current_macro_context();
    let Ok(request_words) = u32::try_from(request.len()) else {
        return;
    };
    let Ok(response_words) = u32::try_from(response.len()) else {
        return;
    };
    let mut words = Vec::with_capacity(5 + request.len() + response.len());
    words.push(0);
    words.push(op_id);
    words.push(rc as u32);
    words.push(request_words);
    words.push(response_words);
    words.extend_from_slice(request);
    words.extend_from_slice(response);
    words[0] = (words.len() - 1) as u32;
    let mut encoded = Vec::with_capacity(words.len().saturating_mul(4));
    for word in words {
        encoded.extend_from_slice(&word.to_le_bytes());
    }
    let Ok(mut writer) = writer.lock() else {
        return;
    };
    // A correctness panic may follow immediately while the CPU mirror checks
    // the returned journal. Keep the just-observed transaction durable as one
    // complete record so the failure itself remains replayable.
    if writer.write_all(&encoded).is_err() || writer.flush().is_err() {
        return;
    }
    FULL_ROOT_TRANSCRIPT_COMMANDS.fetch_add(1, Ordering::Relaxed);
    FULL_ROOT_TRANSCRIPT_REQUEST_WORDS.fetch_add(u64::from(request_words), Ordering::Relaxed);
    FULL_ROOT_TRANSCRIPT_RESPONSE_WORDS.fetch_add(u64::from(response_words), Ordering::Relaxed);
}

fn flush_full_root_transcript_writer() {
    if let Some(writer) = full_root_transcript_writer()
        && let Ok(mut writer) = writer.lock()
    {
        let _ = writer.flush();
    }
}

#[inline]
fn exact_push_u64(words: &mut Vec<u32>, value: u64) {
    words.push(value as u32);
    words.push((value >> 32) as u32);
}

/// Serialize one production-scheduled batch in the exact word encoding used
/// by the C++/HLS boundary.  The stream deliberately duplicates its logical
/// context per record: records are self-contained, while the native replayer
/// can still model a two-entry resident-context cache and distinguish hits,
/// prefix appends and replacements.
fn record_exact_replay_batch(
    pass_id: u64,
    context: &ShadowContext,
    pending: &[(usize, IncrementalQuery, ShadowContext)],
    cpu: &[PairedCpuWork],
    end_of_pass: bool,
) {
    let limit = exact_replay_limit();
    if limit == 0 {
        return;
    }
    let written = EXACT_REPLAY_QUERIES.load(Ordering::Relaxed);
    if written >= limit {
        return;
    }
    let take = pending.len().min((limit - written) as usize);
    if take == 0 {
        return;
    }
    let Some(writer) = exact_replay_writer() else {
        return;
    };

    let batch_id = EXACT_REPLAY_BATCHES.fetch_add(1, Ordering::Relaxed) + 1;
    let mut words = Vec::new();
    // Filled after the record has been assembled; length excludes itself.
    words.push(0);
    words.push(EXACT_REPLAY_BATCH);
    exact_push_u64(&mut words, pass_id);
    exact_push_u64(&mut words, batch_id);
    let (phase, scoped_op_id) = crate::inductor::current_macro_context();
    // solve_active_batch accepts only an already-independent inquiry set.
    // pass_id spans every context/word-limited record produced by this one
    // call, so it is the exact persistent-queue scope when a caller launched
    // outside a thread-local macro_scope (notably async PUSH prefetch).
    let op_id = if scoped_op_id != 0 {
        scoped_op_id
    } else {
        (pass_id as u32).max(1)
    };
    words.push(phase as u32);
    words.push(op_id);
    words.push(1); // independent inquiry set
    words.push(match context.scope {
        ShadowContextScope::SharedTransition => 0,
        ShadowContextScope::ExactFrame(_) => 1,
        ShadowContextScope::FrameRanged => 2,
    });
    words.push(context.n_var);
    words.push(context.clauses.len() as u32);
    words.push(take as u32);
    for clause in &context.clauses {
        words.push(clause.lo);
        words.push(clause.hi);
        words.push(clause.literals.len() as u32);
        words.extend(clause.literals.iter().map(|lit| u32::from(*lit)));
    }
    for (query_position, (index, query, _)) in pending.iter().take(take).enumerate() {
        let Some(work) = cpu.get(*index) else {
            return;
        };
        let Some(result) = work.result.as_ref() else {
            return;
        };
        // Exact replay is a correctness gate, not a latency-limit experiment:
        // remove scheduling budgets and cross-query learnt retention so every
        // record must reach the CPU oracle's conclusive result independently.
        let mut query = query.clone();
        query.budget = QueryBudget::default();
        query.keep_learnts = false;
        let (header, payload) = query.pack();
        let query_words = header.as_words().len() + payload.len();
        words.push(*index as u32);
        // A capture truncated by its query limit is a closed simulation batch,
        // even when the live independent wave had more inquiries.
        let ticket = persistent_ring_query_ticket(
            pass_id,
            *index,
            query_position + 1 == take
                && (end_of_pass || take < pending.len() || written + take as u64 >= limit),
        );
        words.push(ticket.batch_id);
        words.push(ticket.position);
        words.push(ticket.flags);
        words.push(ticket.user_tag as u32);
        words.push((ticket.user_tag >> 32) as u32);
        words.push(query_words as u32);
        words.extend(header.as_words());
        words.extend(payload);
        words.push(result_status(result));
        words.push(result_reason(result));
        match result {
            IncrementalResult::Sat { model } => {
                words.push(0);
                words.push(model.len() as u32);
                words.extend(model.iter().map(|lit| u32::from(*lit)));
            }
            IncrementalResult::Unsat {
                core,
                used_constraints,
            } => {
                words.push(u32::from(*used_constraints));
                words.push(core.len() as u32);
                words.extend(core.iter().map(|lit| u32::from(*lit)));
            }
            IncrementalResult::Unknown(_) => {
                words.push(0);
                words.push(0);
            }
        }
    }
    words[0] = (words.len() - 1) as u32;
    let Ok(mut writer) = writer.lock() else {
        return;
    };
    for word in words {
        if writer.write_all(&word.to_le_bytes()).is_err() {
            return;
        }
    }
    if let Ok(mut roots) = exact_replay_roots().lock() {
        roots.insert(op_id);
    }
    EXACT_REPLAY_QUERIES.fetch_add(take as u64, Ordering::Relaxed);
}

/// Snapshot one ordinary CPU MIC before it mutates the solver. The request is
/// deliberately unlimited, model-guided and proof-neutral: the native C++
/// replayer must finish the whole dependent traversal from this exact formula.
pub fn begin_exact_mic_replay(
    solver: &DagCnfSolver,
    cube: &[(Lit, Lit)],
    constraints: &[LitVec],
    protected_index: usize,
) -> Option<ExactMicReplayCapture> {
    let limit = exact_mic_replay_limit();
    if std::env::var_os("INDUCTOR_CDCL_EXACT_REPLAY").is_none()
        || limit == 0
        || EXACT_REPLAY_MICS.load(Ordering::Relaxed) >= limit
        || cube.len() < 2
    {
        return None;
    }
    let mut cache = BatchedSolverContext::new(solver);
    let context = cache.context(false).clone();
    let frame = match context.scope {
        ShadowContextScope::ExactFrame(frame) => frame,
        ShadowContextScope::FrameRanged => solver.accel_level,
        ShadowContextScope::SharedTransition => return None,
    };
    let mut request = pack_mic_chain_request(
        context.n_var,
        frame,
        cube,
        constraints,
        protected_index,
        0,
        0,
        0,
    )
    .ok()?;
    request[2] |= MIC_MODEL_SHRINK;
    Some(ExactMicReplayCapture { context, request })
}

/// Complete one exact MIC record with the final cube produced by ordinary
/// GipSAT. CSim either has to reproduce it or emit an independently provable
/// alternative inductive cube; the live IC3 result remains untouched.
pub fn finish_exact_mic_replay(capture: Option<ExactMicReplayCapture>, output: &LitVec) {
    let Some(capture) = capture else { return };
    if output.is_empty() {
        return;
    }
    let limit = exact_mic_replay_limit();
    let Some(writer) = exact_replay_writer() else {
        return;
    };
    let Ok(previous) =
        EXACT_REPLAY_MICS.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            (current < limit).then_some(current + 1)
        })
    else {
        return;
    };
    let mic_id = previous + 1;
    let mut words = Vec::new();
    words.push(0);
    words.push(EXACT_REPLAY_MIC);
    exact_push_u64(&mut words, mic_id);
    let (phase, op_id) = crate::inductor::current_macro_context();
    words.push(phase as u32);
    words.push(op_id);
    words.push(2); // dependent MIC chain
    words.push(match capture.context.scope {
        ShadowContextScope::SharedTransition => 0,
        ShadowContextScope::ExactFrame(_) => 1,
        ShadowContextScope::FrameRanged => 2,
    });
    words.push(capture.context.n_var);
    words.push(capture.context.clauses.len() as u32);
    words.push(capture.request.len() as u32);
    words.push(output.len() as u32);
    for clause in &capture.context.clauses {
        words.push(clause.lo);
        words.push(clause.hi);
        words.push(clause.literals.len() as u32);
        words.extend(clause.literals.iter().map(|lit| u32::from(*lit)));
    }
    words.extend(capture.request);
    words.extend(output.iter().map(|lit| u32::from(*lit)));
    words[0] = (words.len() - 1) as u32;
    let Ok(mut writer) = writer.lock() else {
        return;
    };
    for word in words {
        if writer.write_all(&word.to_le_bytes()).is_err() {
            return;
        }
    }
    if let Ok(mut roots) = exact_replay_roots().lock() {
        roots.insert(op_id);
    }
}

/// Capture the CPU proof-state boundary for one algorithm-owned BLOCK root.
/// Roots with exact query/MIC work become full image checkpoints; all others
/// become compact event-only records in version 8.
pub fn begin_exact_block_progress(
    frame: usize,
    obligations: (usize, u64),
    lemmas: (usize, u64),
    obligation_image: Vec<u32>,
    lemma_image: Vec<u32>,
) -> Option<ExactBlockProgressCapture> {
    if std::env::var_os("INDUCTOR_CDCL_EXACT_REPLAY").is_none() && !block_controller_sim_requested()
    {
        return None;
    }
    let (phase, op_id) = crate::inductor::current_macro_context();
    if phase != inductor_trace::Phase::Block || op_id == 0 {
        return None;
    }
    reconcile_block_controller_sim(&obligation_image, &lemma_image);
    Some(ExactBlockProgressCapture {
        op_id,
        frame: frame.min(u32::MAX as usize) as u32,
        obligations,
        lemmas,
        current_obligation_image: obligation_image.clone(),
        current_lemma_image: lemma_image.clone(),
        obligation_image,
        lemma_image,
        steps: Vec::new(),
    })
}

pub fn exact_block_progress_enabled() -> bool {
    std::env::var_os("INDUCTOR_CDCL_EXACT_REPLAY").is_some() || block_controller_sim_requested()
}

/// Begin one direct proof-state maintenance wave.  This scope deliberately
/// starts before IC3 extends the delta-frame vector and closes only after
/// propagation and infinity promotion, so no inter-BLOCK mutation needs to be
/// inferred at the next root boundary.
pub fn begin_exact_frame_events(
    obligations: (usize, u64),
    lemmas: (usize, u64),
    obligation_image: Vec<u32>,
    lemma_image: Vec<u32>,
) -> Option<ExactFrameEventCapture> {
    (std::env::var_os("INDUCTOR_CDCL_EXACT_REPLAY").is_some() || block_controller_sim_requested())
        .then(|| {
            reconcile_block_controller_sim(&obligation_image, &lemma_image);
            let wave_id = (EXACT_REPLAY_FRAME_EVENTS.fetch_add(1, Ordering::Relaxed) + 1)
                .min(u64::from(u32::MAX)) as u32;
            ExactFrameEventCapture {
                wave_id,
                input_obligations: obligations,
                input_lemmas: lemmas,
                initial_obligation_image: obligation_image,
                initial_lemma_image: lemma_image,
            }
        })
}

/// Close a maintenance wave with CPU images used only as an exact simulation
/// oracle. The serialized production-shape program is `semantic_ops`.
pub fn finish_exact_frame_events(
    capture: Option<ExactFrameEventCapture>,
    proved: bool,
    obligations: (usize, u64),
    lemmas: (usize, u64),
    obligation_image: Vec<u32>,
    lemma_image: Vec<u32>,
    semantic_ops: Vec<Vec<u32>>,
) {
    let Some(capture) = capture else { return };
    apply_block_controller_sim(&semantic_ops, &obligation_image, &lemma_image);
    let Some(writer) = exact_replay_writer() else {
        return;
    };
    let mut words = vec![
        0,
        EXACT_REPLAY_FRAME_EVENT_RECORD,
        capture.wave_id,
        u32::from(proved),
        capture.input_obligations.0.min(u32::MAX as usize) as u32,
    ];
    exact_push_u64(&mut words, capture.input_obligations.1);
    words.push(obligations.0.min(u32::MAX as usize) as u32);
    exact_push_u64(&mut words, obligations.1);
    words.push(capture.input_lemmas.0.min(u32::MAX as usize) as u32);
    exact_push_u64(&mut words, capture.input_lemmas.1);
    words.push(lemmas.0.min(u32::MAX as usize) as u32);
    exact_push_u64(&mut words, lemmas.1);
    words.push(semantic_ops.len().min(u32::MAX as usize) as u32);
    words.push(
        capture
            .initial_obligation_image
            .len()
            .min(u32::MAX as usize) as u32,
    );
    words.extend(capture.initial_obligation_image);
    words.push(capture.initial_lemma_image.len().min(u32::MAX as usize) as u32);
    words.extend(capture.initial_lemma_image);
    words.push(obligation_image.len().min(u32::MAX as usize) as u32);
    words.extend(obligation_image);
    words.push(lemma_image.len().min(u32::MAX as usize) as u32);
    words.extend(lemma_image);
    for operation in semantic_ops {
        words.extend(operation);
    }
    words[0] = (words.len() - 1) as u32;
    let Ok(mut writer) = writer.lock() else {
        return;
    };
    for word in words {
        if writer.write_all(&word.to_le_bytes()).is_err() {
            return;
        }
    }
}

/// Append one algorithm-level instruction to the buffered BLOCK program.
pub fn note_exact_block_progress_step(
    capture: Option<&mut ExactBlockProgressCapture>,
    event: u32,
    obligations: (usize, u64),
    lemmas: (usize, u64),
    obligation_image: Vec<u32>,
    lemma_image: Option<Vec<u32>>,
    semantic_ops: Vec<Vec<u32>>,
) {
    let Some(capture) = capture else { return };
    let live_lemma_image = lemma_image
        .as_deref()
        .unwrap_or(&capture.current_lemma_image);
    apply_block_controller_sim(&semantic_ops, &obligation_image, live_lemma_image);
    fn patch(previous: &mut Vec<u32>, next: Vec<u32>) -> ExactImagePatch {
        let prefix = previous
            .iter()
            .zip(&next)
            .take_while(|(left, right)| left == right)
            .count();
        let suffix = previous[prefix..]
            .iter()
            .rev()
            .zip(next[prefix..].iter().rev())
            .take_while(|(left, right)| left == right)
            .count();
        let append = next[prefix..next.len() - suffix].to_vec();
        *previous = next;
        ExactImagePatch {
            prefix: prefix.min(u32::MAX as usize) as u32,
            suffix: suffix.min(u32::MAX as usize) as u32,
            append,
        }
    }
    let obligation_patch = patch(&mut capture.current_obligation_image, obligation_image);
    let lemma_patch = lemma_image.map(|image| patch(&mut capture.current_lemma_image, image));
    capture.steps.push(ExactBlockProgressStep {
        event,
        obligations,
        lemmas,
        obligation_patch,
        lemma_patch,
        semantic_ops,
    });
}

pub fn finish_exact_block_progress(
    capture: Option<ExactBlockProgressCapture>,
    result: u32,
    result_aux: u32,
    obligations: (usize, u64),
    lemmas: (usize, u64),
) {
    let Some(capture) = capture else { return };
    let captured = exact_replay_roots()
        .lock()
        .is_ok_and(|roots| roots.contains(&capture.op_id));
    let Some(writer) = exact_replay_writer() else {
        return;
    };
    let mut words = vec![
        0,
        if captured {
            EXACT_REPLAY_BLOCK_PROGRESS_RECORD
        } else {
            EXACT_REPLAY_BLOCK_EVENT_RECORD
        },
        capture.op_id,
        capture.frame,
        result,
        result_aux,
        capture.obligations.0.min(u32::MAX as usize) as u32,
    ];
    exact_push_u64(&mut words, capture.obligations.1);
    words.push(obligations.0.min(u32::MAX as usize) as u32);
    exact_push_u64(&mut words, obligations.1);
    words.push(capture.lemmas.0.min(u32::MAX as usize) as u32);
    exact_push_u64(&mut words, capture.lemmas.1);
    words.push(lemmas.0.min(u32::MAX as usize) as u32);
    exact_push_u64(&mut words, lemmas.1);
    words.push(capture.steps.len().min(u32::MAX as usize) as u32);
    // Both checkpoint and event-only roots carry their initial images as a
    // trace-only oracle. Event-only roots omit every per-step image patch;
    // their production model is still the direct semantic handle stream.
    words.push(capture.obligation_image.len().min(u32::MAX as usize) as u32);
    words.extend(capture.obligation_image);
    words.push(capture.lemma_image.len().min(u32::MAX as usize) as u32);
    words.extend(capture.lemma_image);
    for step in capture.steps {
        words.push(step.event);
        words.push(step.obligations.0.min(u32::MAX as usize) as u32);
        exact_push_u64(&mut words, step.obligations.1);
        words.push(step.lemmas.0.min(u32::MAX as usize) as u32);
        exact_push_u64(&mut words, step.lemmas.1);
        if captured {
            words.push(step.obligation_patch.prefix);
            words.push(step.obligation_patch.suffix);
            words.push(step.obligation_patch.append.len().min(u32::MAX as usize) as u32);
            words.extend(step.obligation_patch.append);
            match step.lemma_patch {
                Some(patch) => {
                    words.push(1);
                    words.push(patch.prefix);
                    words.push(patch.suffix);
                    words.push(patch.append.len().min(u32::MAX as usize) as u32);
                    words.extend(patch.append);
                }
                None => words.push(0),
            }
        }
        words.push(step.semantic_ops.len().min(u32::MAX as usize) as u32);
        for operation in step.semantic_ops {
            words.extend(operation);
        }
    }
    words[0] = (words.len() - 1) as u32;
    let Ok(mut writer) = writer.lock() else {
        return;
    };
    for word in words {
        if writer.write_all(&word.to_le_bytes()).is_err() {
            return;
        }
    }
    if captured {
        EXACT_REPLAY_BLOCK_PROGRESS.fetch_add(1, Ordering::Relaxed);
    } else {
        EXACT_REPLAY_BLOCK_EVENTS.fetch_add(1, Ordering::Relaxed);
    }
}

fn flush_exact_replay_writer() {
    if let Some(writer) = exact_replay_writer()
        && let Ok(mut writer) = writer.lock()
    {
        let _ = writer.flush();
    }
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

fn architecture_trace_writer() -> Option<&'static std::sync::Mutex<BufWriter<std::fs::File>>> {
    ARCH_TRACE_WRITER
        .get_or_init(|| {
            let path = worker_scoped_trace_path(
                std::env::var_os("INDUCTOR_CDCL_TRACE_CSV")?,
            );
            let file = match std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&path)
            {
                Ok(file) => file,
                Err(error) => {
                    eprintln!(
                        "inductor-cdcl: cannot create architecture trace {}: {error}",
                        std::path::Path::new(&path).display(),
                    );
                    return None;
                }
            };
            let mut writer = BufWriter::new(file);
            if writeln!(
                writer,
                "pass_id\tmacro_op_id\tbatch_id\tbatch_size\tposition\toriginal_index\tframe\tcpu_status\tcpu_ns\tcpu_decisions\tcpu_conflicts\tcpu_propagations\tassumptions\tconstraint_clauses\tconstraint_literals\tdomain\trequest_words\tquery_fingerprint\tcontext_scope\tcontext_fingerprint\tcontext_vars\tcontext_clauses\tcontext_literals\tcontext_words\tready_unix_ns"
            )
            .is_err()
            {
                return None;
            }
            Some(std::sync::Mutex::new(writer))
        })
        .as_ref()
}

// Observational FNV-1a fingerprints let the no-HLS architecture simulator
// distinguish a useful speculative answer from a merely shape-compatible
// query. They never participate in a proof decision or in the device ABI.
fn architecture_trace_hash_word(hash: &mut u64, value: u64) {
    const PRIME: u64 = 0x100000001b3;
    *hash ^= value;
    *hash = hash.wrapping_mul(PRIME);
}

fn architecture_trace_query_fingerprint(query: &IncrementalQuery) -> u64 {
    let mut hash = 0xcbf29ce484222325;
    architecture_trace_hash_word(&mut hash, query.frame as u64);
    architecture_trace_hash_word(&mut hash, query.assumptions.len() as u64);
    for literal in &query.assumptions {
        architecture_trace_hash_word(&mut hash, u32::from(*literal) as u64);
    }
    architecture_trace_hash_word(&mut hash, query.constraints.len() as u64);
    for clause in &query.constraints {
        architecture_trace_hash_word(&mut hash, clause.len() as u64);
        for literal in clause {
            architecture_trace_hash_word(&mut hash, u32::from(*literal) as u64);
        }
    }
    architecture_trace_hash_word(&mut hash, query.domain.len() as u64);
    for variable in &query.domain {
        architecture_trace_hash_word(&mut hash, u32::from(*variable) as u64);
    }
    architecture_trace_hash_word(&mut hash, query.budget.decisions as u64);
    architecture_trace_hash_word(&mut hash, query.budget.conflicts as u64);
    architecture_trace_hash_word(
        &mut hash,
        query.budget.restarts.map_or(u64::MAX, |value| value as u64),
    );
    architecture_trace_hash_word(&mut hash, u64::from(query.keep_learnts));
    hash
}

fn architecture_trace_context_fingerprint(context: &ShadowContext) -> u64 {
    let mut hash = 0xcbf29ce484222325;
    architecture_trace_hash_word(&mut hash, context.n_var as u64);
    architecture_trace_hash_word(
        &mut hash,
        match context.scope {
            ShadowContextScope::SharedTransition => 0,
            ShadowContextScope::ExactFrame(frame) => 1u64 << 32 | frame as u64,
            ShadowContextScope::FrameRanged => 2,
        },
    );
    architecture_trace_hash_word(&mut hash, context.clauses.len() as u64);
    for clause in &context.clauses {
        architecture_trace_hash_word(&mut hash, clause.lo as u64);
        architecture_trace_hash_word(&mut hash, clause.hi as u64);
        architecture_trace_hash_word(&mut hash, clause.literals.len() as u64);
        for literal in &clause.literals {
            architecture_trace_hash_word(&mut hash, u32::from(*literal) as u64);
        }
    }
    hash
}

fn record_architecture_trace_batch(
    pass_id: u64,
    ready_unix_ns: u128,
    context: &ShadowContext,
    pending: &[(usize, IncrementalQuery, ShadowContext)],
    cpu: &[PairedCpuWork],
) {
    if pending.is_empty() {
        return;
    }
    let Some(writer) = architecture_trace_writer() else {
        return;
    };
    let Ok(mut writer) = writer.lock() else {
        return;
    };
    let batch_id = ARCH_TRACE_BATCH_ID.fetch_add(1, Ordering::Relaxed) + 1;
    ARCH_TRACE_QUERIES.fetch_add(pending.len() as u64, Ordering::Relaxed);
    let (_, macro_op_id) = crate::inductor::current_macro_context();
    let context_literals = context
        .clauses
        .iter()
        .map(|clause| clause.literals.len() as u64)
        .sum::<u64>();
    let context_words = context.clauses.iter().fold(2u64, |words, clause| {
        words.saturating_add(3u64.saturating_add(clause.literals.len() as u64))
    });
    let context_scope = match context.scope {
        ShadowContextScope::SharedTransition => "shared",
        ShadowContextScope::ExactFrame(_) => "exact-frame",
        ShadowContextScope::FrameRanged => "frame-ranged",
    };
    let context_fingerprint = architecture_trace_context_fingerprint(context);
    for (position, (index, query, _)) in pending.iter().enumerate() {
        let Some(cpu_work) = cpu.get(*index) else {
            continue;
        };
        let constraint_literals = query
            .constraints
            .iter()
            .map(|clause| clause.len() as u64)
            .sum::<u64>();
        let _ = writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            pass_id,
            macro_op_id,
            batch_id,
            pending.len(),
            position,
            index,
            query.frame,
            cpu_work.status,
            cpu_work.elapsed_ns,
            cpu_work.decisions,
            cpu_work.conflicts,
            cpu_work.propagations,
            query.assumptions.len(),
            query.constraints.len(),
            constraint_literals,
            query.domain.len(),
            query_request_words(query).unwrap_or(0),
            architecture_trace_query_fingerprint(query),
            context_scope,
            context_fingerprint,
            context.n_var,
            context.clauses.len(),
            context_literals,
            context_words,
            ready_unix_ns,
        );
    }
}

fn flush_architecture_trace_writer() {
    if let Some(writer) = architecture_trace_writer()
        && let Ok(mut writer) = writer.lock()
    {
        let _ = writer.flush();
    }
}

fn flush_block_root_timeline_writer() {
    if let Some(writer) = block_root_timeline_writer()
        && let Ok(mut writer) = writer.lock()
    {
        let _ = writer.flush();
    }
}

fn record_comparison_batch(
    pass_id: u64,
    context: &ShadowContext,
    pending: &[(usize, IncrementalQuery, ShadowContext)],
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
    let batch_cpu_ns = pending.iter().fold(0u64, |total, (index, _, _)| {
        total.saturating_add(cpu.get(*index).map_or(0, |work| work.elapsed_ns))
    });
    PAIRED_QUERIES.fetch_add(queries.len() as u64, Ordering::Relaxed);
    PAIRED_CPU_NS.fetch_add(batch_cpu_ns, Ordering::Relaxed);
    PAIRED_HW_NS.fetch_add(batch_hw_ns, Ordering::Relaxed);
    if batch_hw_ns < batch_cpu_ns {
        PAIRED_HW_FASTER_BATCHES.fetch_add(1, Ordering::Relaxed);
    }
    for ((index, _, _), work) in pending.iter().zip(hardware) {
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
    for (position, (((index, _, _), query), work)) in
        pending.iter().zip(queries).zip(hardware).enumerate()
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

/// Enable the proof-safe active path. The default validates device answers on
/// GipSAT; qualified throughput mode may instead restore structurally complete
/// SAT models and UNSAT cores directly. Active and shadow modes are separate so
/// one process never opens the singleton XRT bridge twice.
fn architecture_trace_enabled() -> bool {
    std::env::var_os("INDUCTOR_CDCL_TRACE_CSV").is_some()
}

// Capture propagation traffic without letting the earlier BLOCK phase exhaust
// the bounded exact-replay query budget. This remains observational: the CPU
// solver is authoritative and solve_active_batch returns UNKNOWN to IC3.
fn architecture_frame_trace_only() -> bool {
    architecture_trace_enabled()
        && std::env::var("INDUCTOR_CDCL_TRACE_FRAME_ONLY")
            .ok()
            .is_some_and(|value| !matches!(value.as_str(), "" | "0" | "false" | "off"))
}

pub fn active_enabled() -> bool {
    (std::env::var_os("INDUCTOR_CDCL_ACTIVE").is_some() || architecture_trace_enabled())
        && std::env::var_os("INDUCTOR_CDCL_PAIRED").is_none()
        && std::env::var_os("INDUCTOR_CDCL_SHADOW").is_none()
        && std::env::var_os("INDUCTOR_ACCEL").is_none()
}

/// Command 12/13 require the shared-frame-view image. Keep the feature
/// explicit so an older, otherwise ABI-compatible xclbin cannot be selected
/// accidentally while simulation and board qualification overlap.
fn active_arena_views_enabled() -> bool {
    active_enabled()
        && std::env::var("INDUCTOR_CDCL_ARENA_VIEWS")
            .ok()
            .is_some_and(|value| !matches!(value.as_str(), "0" | "false" | "off"))
}

fn shared_domain_projection_enabled() -> bool {
    std::env::var("INDUCTOR_CDCL_SHARED_DOMAIN_PROJECTION")
        .ok()
        .is_some_and(|value| !matches!(value.as_str(), "" | "0" | "false" | "off"))
}

fn cross_context_batch_enabled() -> bool {
    shared_domain_projection_enabled()
        || std::env::var("INDUCTOR_CDCL_CROSS_CONTEXT_BATCH")
            .ok()
            .is_some_and(|value| !matches!(value.as_str(), "" | "0" | "false" | "off"))
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
    (active_enabled() || paired_enabled())
        && (std::env::var_os("INDUCTOR_CDCL_BLOCK_ONLY").is_none()
            || architecture_frame_trace_only())
        && propagation_batch_requested()
        && !push_prefetch_enabled()
        && active_hardware_available()
}

fn propagation_batch_requested() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_PROPAGATION_BATCH")
            .ok()
            .map(|value| !matches!(value.as_str(), "0" | "false" | "off"))
            .unwrap_or_else(|| {
                let mic_chain_requested = std::env::var("INDUCTOR_CDCL_MIC_CHAIN")
                    .ok()
                    .is_some_and(|value| !matches!(value.as_str(), "0" | "false" | "off"));
                // Board measurements show that combining aggressive
                // propagation takeover with MIC-chain floods large designs
                // with stale SAT models and lane-tail service. Throughput mode
                // therefore defaults to the dependent short-chain path when
                // it is explicitly requested. A/B runs can opt propagation
                // back in with INDUCTOR_CDCL_PROPAGATION_BATCH=1.
                !(throughput_enabled() && mic_chain_requested)
            })
    })
}

/// Run lemma-push inquiries in the background after one pass and consume only
/// answers in a later pass. A prefetched SAT result is always revalidated after
/// frame mutation; an old UNSAT result stays valid under monotonic strengthening.
/// Active mode defaults to the measured context-local short-query policy; an
/// explicit false value restores the synchronous propagation scheduler for A/B.
pub fn push_prefetch_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    active_enabled()
        && std::env::var_os("INDUCTOR_CDCL_BLOCK_ONLY").is_none()
        && !crate::accel::cdcl_host::throughput_enabled()
        && active_hardware_available()
        && *ENABLED.get_or_init(|| {
            std::env::var("INDUCTOR_CDCL_PUSH_PREFETCH")
                .ok()
                .is_none_or(|value| !matches!(value.as_str(), "0" | "false" | "off"))
        })
}

pub fn active_skip_cpu_check() -> bool {
    static SKIP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SKIP.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_ACTIVE_SKIP_CPU_CHECK")
            .ok()
            .map(|value| !matches!(value.as_str(), "0" | "false" | "off"))
            .unwrap_or_else(|| throughput_enabled())
    })
}

/// Speculate over the first inductiveness check for several literal-drop
/// candidates from the same MIC cube. This is deliberately opt-in while the
/// proof-path effect is measured. The conservative mode validates on live
/// GipSAT; qualified throughput mode may restore result state directly.
pub fn mic_batch_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        active_enabled()
            && active_hardware_available()
            && std::env::var("INDUCTOR_CDCL_MIC_BATCH")
                .ok()
                .is_some_and(|value| !matches!(value.as_str(), "0" | "false" | "off"))
    })
}

/// Move the dependent literal-drop sequence, not only isolated BCP, into one
/// full-CDCL device command. This remains opt-in because it requires the MIC
/// ABI extension in the selected xclbin.
pub fn mic_chain_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        active_enabled()
            && active_hardware_available()
            && std::env::var("INDUCTOR_CDCL_MIC_CHAIN")
                .ok()
                .is_some_and(|value| !matches!(value.as_str(), "0" | "false" | "off"))
    })
}

pub fn mic_chain_min_cube() -> usize {
    static SIZE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SIZE.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_MIC_CHAIN_MIN_CUBE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(2)
            .clamp(1, 4096)
    })
}

pub fn mic_chain_max_cube() -> usize {
    static SIZE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SIZE.get_or_init(|| {
        let fpga_throughput = std::env::var("INDUCTOR_CDCL_FPGA_THROUGHPUT")
            .ok()
            .is_some_and(|value| !matches!(value.as_str(), "0" | "false" | "off"));
        std::env::var("INDUCTOR_CDCL_MIC_CHAIN_MAX_CUBE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(if fpga_throughput { 5 } else { 4096 })
            .clamp(1, 4096)
    })
}

pub fn mic_chain_skip_cpu_check() -> bool {
    static SKIP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SKIP.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_MIC_CHAIN_SKIP_CPU_CHECK")
            .ok()
            .map(|value| !matches!(value.as_str(), "0" | "false" | "off"))
            .unwrap_or_else(|| {
                std::env::var("INDUCTOR_CDCL_FPGA_THROUGHPUT")
                    .ok()
                    .is_some_and(|value| !matches!(value.as_str(), "0" | "false" | "off"))
            })
    })
}

fn mic_chain_model_shrink() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_MIC_MODEL_SHRINK")
            .ok()
            .is_some_and(|value| !matches!(value.as_str(), "0" | "false" | "off"))
    })
}

fn mic_chain_parallel_lanes() -> usize {
    static LANES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *LANES.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_MIC_CHAIN_LANES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1)
            .clamp(1, 4)
    })
}

fn mic_chain_experimental_reorder() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_MIC_CHAIN_EXPERIMENTAL_REORDER")
            .ok()
            .is_some_and(|value| !matches!(value.as_str(), "0" | "false" | "off"))
    })
}

fn mic_chain_conflict_budget() -> u32 {
    static BUDGET: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_MIC_CHAIN_CONFLICT_BUDGET")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or_else(active_conflict_budget)
    })
}

fn mic_chain_decision_budget() -> u32 {
    static BUDGET: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_MIC_CHAIN_DECISION_BUDGET")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(0)
    })
}

fn mic_chain_max_trials() -> u32 {
    static TRIALS: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *TRIALS.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_MIC_CHAIN_MAX_TRIALS")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(0)
    })
}

pub fn mic_batch_min_size() -> usize {
    let fpga_throughput = std::env::var("INDUCTOR_CDCL_FPGA_THROUGHPUT")
        .ok()
        .is_some_and(|value| !matches!(value.as_str(), "0" | "false" | "off"));
    static SIZE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SIZE.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_MIC_BATCH_MIN")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(if fpga_throughput { 1 } else { 8 })
            .clamp(1, DEFAULT_SHADOW_BATCH_SIZE)
    })
}

pub fn mic_batch_window() -> usize {
    static SIZE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SIZE.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_MIC_BATCH_WINDOW")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(DEFAULT_SHADOW_BATCH_SIZE)
            .clamp(1, DEFAULT_SHADOW_BATCH_SIZE)
    })
}

/// Whether IC3 should speculate over the currently queued proof obligations.
/// Conservative mode checks every answer against the live frame. Qualified
/// synchronous results may be restored directly; asynchronous SAT remains
/// subject to live-frame validation.
pub fn block_batch_enabled() -> bool {
    (active_enabled() || paired_enabled())
        && std::env::var_os("INDUCTOR_CDCL_BLOCK_BATCH").is_some()
        && !architecture_frame_trace_only()
        && active_hardware_available()
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

/// Probe the process-lifetime transport once and fail over before manufacturing
/// contexts or batches when the configured card/server could not be opened.
/// `ACTIVE_STATE` is a `OnceLock`, so a missing device never triggers another
/// XRT initialization attempt in the same rIC3 process.
fn active_hardware_available() -> bool {
    if architecture_trace_enabled() {
        return true;
    }
    if ACTIVE_HARDWARE_DISABLED.load(Ordering::Relaxed) {
        return false;
    }
    let available = *ACTIVE_HARDWARE_AVAILABLE.get_or_init(|| {
        let wait_start = std::time::Instant::now();
        let state = active_state().lock();
        ACTIVE_STATE_WAIT_NS.fetch_add(
            wait_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        state.is_ok_and(|state| state.hardware.is_some())
    });
    if !available {
        ACTIVE_TRANSPORT_UNAVAILABLE.store(true, Ordering::Relaxed);
    }
    available
}

fn deterministic_active_failure(error: &HardwareError) -> bool {
    matches!(
        error,
        // The command completed and returned a semantic backend error; the
        // same frontier shape will not become supported on a retry.
        HardwareError::Decode(_)
            | HardwareError::Capacity
            // The C++ bridge maps the kernel's top-level BAD_COMMAND,
            // BAD_PAYLOAD, CAPACITY and RESPONSE_CAPACITY statuses to
            // -101..=-104. Retrying the same solver/image/shape cannot change
            // any of them; an older image otherwise produces one error per
            // frontier batch instead of falling back once to the CPU.
            | HardwareError::Command(-104..=-101)
    )
}

fn active_failure_is_capacity(error: &HardwareError) -> bool {
    matches!(
        error,
        HardwareError::Capacity | HardwareError::Command(-103)
    )
}

fn disable_active_hardware(error: &HardwareError) {
    if !ACTIVE_HARDWARE_DISABLED.swap(true, Ordering::Relaxed) {
        ACTIVE_HARDWARE_DISABLES.fetch_add(1, Ordering::Relaxed);
        eprintln!(
            "inductor-cdcl: disabling active hardware for this solver after deterministic failure: {error}"
        );
    }
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
    let queries: Vec<_> = batch
        .pending
        .iter()
        .map(|(query, _)| query.clone())
        .collect();
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

/// Execute one dependent MIC traversal against the solver's exact resident
/// frame. Context changes use the same monotonic append/reload discipline as
/// ordinary active batches. Any transport, lease or decode failure returns no
/// candidate and leaves the native CPU MIC path authoritative.
pub fn solve_active_mic_chain(
    solver: &DagCnfSolver,
    cube: &[(Lit, Lit)],
    constraints: &[LitVec],
    protected_index: usize,
) -> Option<MicChainResult> {
    let client_started = std::time::Instant::now();
    if !mic_chain_enabled()
        || cube.len() < mic_chain_min_cube()
        || cube.len() > mic_chain_max_cube()
        || protected_index >= cube.len()
    {
        return None;
    }
    if !active_hardware_available() {
        ACTIVE_UNAVAILABLE_CALLS.fetch_add(1, Ordering::Relaxed);
        ACTIVE_UNAVAILABLE_QUERIES.fetch_add(cube.len() as u64, Ordering::Relaxed);
        return None;
    }
    let mut cache = BatchedSolverContext::new(solver);
    let context = cache.context(false).clone();
    let frame = match context.scope {
        ShadowContextScope::ExactFrame(frame) => frame,
        ShadowContextScope::FrameRanged => solver.accel_level,
        ShadowContextScope::SharedTransition => return None,
    };

    let wait_start = std::time::Instant::now();
    let state = active_state().lock();
    ACTIVE_STATE_WAIT_NS.fetch_add(
        wait_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        Ordering::Relaxed,
    );
    let Ok(mut state) = state else {
        ACTIVE_MIC_CHAIN_ERRORS.fetch_add(1, Ordering::Relaxed);
        return None;
    };

    let arena_views = active_arena_views_enabled() && !mic_chain_experimental_reorder();
    let update = if arena_views {
        ContextUpdate::Ready
    } else {
        plan_context_update(state.loaded_context.as_ref(), &context)
    };
    let context_reused = if arena_views {
        state
            .hardware
            .as_ref()
            .is_some_and(|hardware| hardware.arena.n_var == context.n_var)
    } else {
        update != ContextUpdate::Reload
    };
    let mut ready = update == ContextUpdate::Ready;
    let mut fused_append_clauses = None;
    if arena_views {
        state.loaded_context = None;
    }
    if let ContextUpdate::Append(clauses) = update {
        // The production traversal is one order-dependent chain. Fuse its
        // monotonic resident append with the immediately following MIC so the
        // hundreds of tiny lemma updates do not each pay an XRT submission.
        // Explicit reordered multi-chain experiments retain the standalone
        // append path because command 8 deliberately preserves one chain's
        // exact caller order.
        if !mic_chain_experimental_reorder() {
            ready = true;
            fused_append_clauses = Some(clauses);
        } else {
            let started = std::time::Instant::now();
            let kernel_before = direct_kernel_ns();
            let appended = state
                .hardware
                .as_mut()
                .ok_or(HardwareError::Unavailable)
                .and_then(|hardware| hardware.add_frame_clauses(&clauses));
            ACTIVE_CONTEXT_APPEND_KERNEL_NS.fetch_add(
                direct_kernel_ns().saturating_sub(kernel_before),
                Ordering::Relaxed,
            );
            let elapsed = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            ACTIVE_CONTEXT_APPEND_NS.fetch_add(elapsed, Ordering::Relaxed);
            match appended {
                Ok(()) => {
                    ACTIVE_CONTEXT_APPENDS.fetch_add(1, Ordering::Relaxed);
                    ACTIVE_CONTEXT_APPEND_CLAUSES
                        .fetch_add(clauses.len() as u64, Ordering::Relaxed);
                    if let Some(loaded) = state.loaded_context.as_mut() {
                        loaded.clauses.extend(clauses);
                        ready = true;
                    }
                }
                Err(_) => {
                    ACTIVE_CONTEXT_APPEND_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                    state.loaded_context = None;
                }
            }
        }
    }
    if !ready {
        let started = std::time::Instant::now();
        let kernel_before = direct_kernel_ns();
        let loaded = state
            .hardware
            .as_mut()
            .ok_or(HardwareError::Unavailable)
            .and_then(|hardware| hardware.load_context(context.n_var, &context.clauses));
        ACTIVE_CONTEXT_LOAD_KERNEL_NS.fetch_add(
            direct_kernel_ns().saturating_sub(kernel_before),
            Ordering::Relaxed,
        );
        let elapsed = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        ACTIVE_CONTEXT_LOAD_NS.fetch_add(elapsed, Ordering::Relaxed);
        if loaded.is_err() {
            state.loaded_context = None;
            ACTIVE_MIC_CHAIN_ERRORS.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        ACTIVE_CONTEXT_LOADS.fetch_add(1, Ordering::Relaxed);
        state.loaded_context = Some(LoadedContext::from(&context));
    }

    // Select the active frame in a separate device command. The query retains
    // an on-card guard and would materialize automatically if this call were
    // ever omitted, but the split prevents frame maintenance from being
    // mislabeled as useful MIC/CDCL occupancy.
    if !arena_views && fused_append_clauses.is_none() {
        let materialize_started = std::time::Instant::now();
        let materialize_kernel_before = direct_kernel_ns();
        let materialized = state
            .hardware
            .as_mut()
            .ok_or(HardwareError::Unavailable)
            .and_then(|hardware| hardware.materialize_frame(frame));
        let materialize_kernel_ns = direct_kernel_ns().saturating_sub(materialize_kernel_before);
        let materialize_service_ns = materialize_started
            .elapsed()
            .as_nanos()
            .min(u64::MAX as u128) as u64;
        match materialized {
            Ok(true) => {
                ACTIVE_FRAME_MATERIALIZATIONS.fetch_add(1, Ordering::Relaxed);
                ACTIVE_FRAME_MATERIALIZE_KERNEL_NS
                    .fetch_add(materialize_kernel_ns, Ordering::Relaxed);
                ACTIVE_FRAME_MATERIALIZE_SERVICE_NS
                    .fetch_add(materialize_service_ns, Ordering::Relaxed);
            }
            Ok(false) => {}
            Err(error) => {
                eprintln!("inductor-cdcl: frame materialization failed: {error}");
                state.loaded_context = None;
                ACTIVE_MIC_CHAIN_ERRORS.fetch_add(1, Ordering::Relaxed);
                return None;
            }
        }
    }

    // The Init guard stays at its original position. The ABI carries its index
    // in the flag word and the kernel skips that one drop in place, preserving
    // GipSAT's existing literal order and proof path.
    let eligible_trials = (cube.len() - 1) as u32;
    let configured_trials = mic_chain_max_trials();
    let max_trials = if configured_trials == 0 {
        eligible_trials
    } else {
        configured_trials.min(eligible_trials)
    };
    let started = std::time::Instant::now();
    let kernel_before = direct_kernel_ns();
    // Rotating the literal order explores different, independently sound MIC
    // traversals, but it changes IC3's proof path and therefore cannot be used
    // to claim stable hardware speedup. It also exposed a model-guided 3/4-lane
    // BAD_QUERY response on the first routed command-6 image. Keep the ABI and
    // board-qualified capacity path available for explicit experiments while
    // the production path preserves the original GipSAT literal order.
    let parallel_lanes = if mic_chain_experimental_reorder() {
        mic_chain_parallel_lanes()
            .min(eligible_trials as usize)
            .max(1)
    } else {
        1
    };
    let result = state
        .hardware
        .as_mut()
        .ok_or(HardwareError::Unavailable)
        .and_then(|hardware| {
            if parallel_lanes == 1 {
                if arena_views {
                    return hardware.solve_arena_mic_chain(
                        &context,
                        frame,
                        cube,
                        constraints,
                        protected_index,
                        mic_chain_decision_budget(),
                        mic_chain_conflict_budget(),
                        max_trials,
                    );
                }
                if let Some(clauses) = fused_append_clauses.as_deref() {
                    return hardware.append_and_solve_mic_chain(
                        clauses,
                        frame,
                        cube,
                        constraints,
                        protected_index,
                        mic_chain_decision_budget(),
                        mic_chain_conflict_budget(),
                        max_trials,
                    );
                }
                return hardware.solve_mic_chain(
                    frame,
                    cube,
                    constraints,
                    protected_index,
                    mic_chain_decision_budget(),
                    mic_chain_conflict_budget(),
                    max_trials,
                );
            }

            // The model-guided chain is order dependent. Launch rotated
            // prefix orders on fixed lanes while leaving the protected Init
            // literal at its original index. Every returned cube is independently proved by its
            // own chain; choosing the smallest result is therefore a search
            // policy, not a vote or CPU replay.
            let eligible = cube.len() - 1;
            let mut lane_cubes = Vec::with_capacity(parallel_lanes);
            for lane in 0..parallel_lanes {
                let mut reordered = cube.to_vec();
                let protected = reordered.remove(protected_index);
                reordered.rotate_left(lane * eligible / parallel_lanes);
                reordered.insert(protected_index, protected);
                lane_cubes.push(reordered);
            }
            let queries: Vec<_> = lane_cubes
                .iter()
                .map(|lane_cube| MicChainQuery {
                    frame,
                    cube: lane_cube,
                    constraints,
                    protected_index,
                    decision_budget: mic_chain_decision_budget(),
                    conflict_budget: mic_chain_conflict_budget(),
                    max_trials,
                })
                .collect();
            hardware.solve_mic_chains(&queries).and_then(|results| {
                let decisions = results.iter().map(|result| result.decisions).sum();
                let conflicts = results.iter().map(|result| result.conflicts).sum();
                let propagations = results.iter().map(|result| result.propagations).sum();
                let learnt_clauses = results.iter().map(|result| result.learnt_clauses).sum();
                let physical_rounds = results
                    .iter()
                    .map(|result| result.physical_rounds)
                    .max()
                    .unwrap_or(0);
                let mut selected = results
                    .into_iter()
                    .min_by_key(|result| {
                        (
                            u8::from(!result.complete),
                            result.cube.len(),
                            u32::MAX - result.trials,
                        )
                    })
                    .ok_or(HardwareError::InvalidResponse)?;
                // Work counters describe the whole device command, while the
                // semantic trial count/cube belong to the selected traversal.
                selected.decisions = decisions;
                selected.conflicts = conflicts;
                selected.propagations = propagations;
                selected.learnt_clauses = learnt_clauses;
                selected.physical_rounds = physical_rounds;
                Ok(selected)
            })
        });
    let service_ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
    // Read the direct-XRT command interval while the active-state lock still
    // excludes another hardware command. RPC mode deliberately reports zero:
    // its kernel lives in the server process and cannot be inferred from
    // client wall time.
    let kernel_elapsed_ns = direct_kernel_ns().saturating_sub(kernel_before);
    if let Some(clauses) = fused_append_clauses.as_ref() {
        ACTIVE_FUSED_APPEND_MIC_COMMANDS.fetch_add(1, Ordering::Relaxed);
        ACTIVE_FUSED_APPEND_MIC_CLAUSES.fetch_add(clauses.len() as u64, Ordering::Relaxed);
        ACTIVE_FUSED_APPEND_MIC_KERNEL_NS.fetch_add(kernel_elapsed_ns, Ordering::Relaxed);
        ACTIVE_FUSED_APPEND_MIC_SERVICE_NS.fetch_add(service_ns, Ordering::Relaxed);
        if result.is_ok() {
            ACTIVE_CONTEXT_APPENDS.fetch_add(1, Ordering::Relaxed);
            ACTIVE_CONTEXT_APPEND_CLAUSES.fetch_add(clauses.len() as u64, Ordering::Relaxed);
            if let Some(loaded) = state.loaded_context.as_mut() {
                loaded.clauses.extend(clauses.iter().cloned());
            }
        } else {
            ACTIVE_CONTEXT_APPEND_FALLBACKS.fetch_add(1, Ordering::Relaxed);
        }
    } else {
        ACTIVE_MIC_CHAIN_KERNEL_NS.fetch_add(kernel_elapsed_ns, Ordering::Relaxed);
        ACTIVE_MIC_CHAIN_SERVICE_NS.fetch_add(service_ns, Ordering::Relaxed);
    }
    match result {
        Ok(mut result) => {
            // The kernel reports a max-trials stop as partial. When the limit
            // exactly equals every eligible prefix literal, the protected
            // suffix is deliberately not a candidate and the host traversal
            // is nevertheless complete. An explicit smaller user cap remains
            // partial and falls through to the CPU loop.
            result.complete = mic_chain_effectively_complete(
                result.complete,
                result.reason,
                result.trials,
                max_trials,
                eligible_trials,
            );
            result.client_ns = client_started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            result.context_reused = context_reused;
            ACTIVE_MIC_CHAIN_COMMANDS.fetch_add(1, Ordering::Relaxed);
            ACTIVE_MIC_CHAIN_INPUT_LITS.fetch_add(cube.len() as u64, Ordering::Relaxed);
            ACTIVE_MIC_CHAIN_OUTPUT_LITS.fetch_add(result.cube.len() as u64, Ordering::Relaxed);
            ACTIVE_MIC_CHAIN_TRIALS.fetch_add(u64::from(result.trials), Ordering::Relaxed);
            ACTIVE_MIC_CHAIN_PHYSICAL_ROUNDS
                .fetch_add(u64::from(result.physical_rounds), Ordering::Relaxed);
            if result.complete {
                ACTIVE_MIC_CHAIN_COMPLETE.fetch_add(1, Ordering::Relaxed);
            } else {
                ACTIVE_MIC_CHAIN_PARTIAL.fetch_add(1, Ordering::Relaxed);
            }
            ACTIVE_MIC_CHAIN_DECISIONS.fetch_add(result.decisions, Ordering::Relaxed);
            ACTIVE_MIC_CHAIN_CONFLICTS.fetch_add(result.conflicts, Ordering::Relaxed);
            ACTIVE_MIC_CHAIN_PROPAGATIONS.fetch_add(result.propagations, Ordering::Relaxed);
            ACTIVE_MIC_CHAIN_LEARNTS.fetch_add(result.learnt_clauses, Ordering::Relaxed);
            ACTIVE_MIC_CHAIN_CLIENT_NS.fetch_add(result.client_ns, Ordering::Relaxed);
            if result.context_reused {
                ACTIVE_MIC_CHAIN_CONTEXT_REUSED.fetch_add(1, Ordering::Relaxed);
            }
            Some(result)
        }
        Err(error) => {
            state.loaded_context = None;
            let prior = ACTIVE_MIC_CHAIN_ERRORS.fetch_add(1, Ordering::Relaxed);
            if prior < 4 {
                eprintln!("inductor-cdcl: MIC chain command failed: {error}");
            }
            None
        }
    }
}

pub fn active_mic_chain_context_reusable(solver: &DagCnfSolver) -> bool {
    if !mic_chain_enabled() || !active_hardware_available() {
        return false;
    }
    active_state()
        .lock()
        .ok()
        .and_then(|state| {
            state
                .loaded_context
                .as_ref()
                .map(|context| match context.scope {
                    ShadowContextScope::FrameRanged => true,
                    ShadowContextScope::ExactFrame(frame) => frame == solver.accel_level,
                    ShadowContextScope::SharedTransition => false,
                })
        })
        .unwrap_or(false)
}

fn mic_chain_effectively_complete(
    device_complete: bool,
    reason: UnknownReason,
    trials: u32,
    max_trials: u32,
    eligible_trials: u32,
) -> bool {
    device_complete
        || (reason == UnknownReason::None
            && max_trials == eligible_trials
            && trials == eligible_trials)
}

fn batch_context_compatible(
    left: &ShadowContext,
    right: &ShadowContext,
    arena_views: bool,
) -> bool {
    if arena_views {
        left.n_var != 0
            && right.n_var != 0
            && left.n_var <= QUALIFIED_ARENA_MAX_VARS
            && right.n_var <= QUALIFIED_ARENA_MAX_VARS
    } else {
        left == right
    }
}

/// Solve a set of already-independent IC3 inquiries in as few XRT submissions
/// as their resident contexts and the command-word limit allow. The returned
/// order matches the input order. This function deliberately does not decide
/// whether a result is proof-safe for IC3; callers may consume a SAT answer
/// only after `DagCnfSolver::validate_incremental_sat_model`, either by the
/// legacy live-trail importer or through an independently certified external
/// predecessor/model-shrinking path.
pub fn solve_active_batch(
    requests: Vec<(&DagCnfSolver, IncrementalQuery)>,
) -> Vec<IncrementalResult> {
    solve_active_batch_with_min(requests, active_min_batch_size())
}

/// Variant used by an explicitly enabled producer whose independent wave has
/// a different measured crossover from propagation. Context partitioning and
/// command-size planning still enforce this minimum for every device batch.
pub fn solve_active_batch_with_min(
    requests: Vec<(&DagCnfSolver, IncrementalQuery)>,
    min_batch_size: usize,
) -> Vec<IncrementalResult> {
    solve_active_batch_reporting(requests, min_batch_size, &mut |_, _| {})
}

/// Publish indexed answers as physical batches finish. All solver-dependent
/// planning and diagnostics finish before the first callback. The execution
/// function accepts owned data only: a completed client's DagCnf may go away
/// while other clients' batches are still running. Callbacks must not block or
/// re-enter the hardware backend (its device-state lock is held).
pub fn solve_active_batch_stream(
    requests: Vec<(&DagCnfSolver, IncrementalQuery)>,
    mut completed: impl FnMut(usize, &IncrementalResult),
) {
    let mut emitted = vec![false; requests.len()];
    let output = solve_active_batch_reporting(requests, active_min_batch_size(), &mut |index, result| {
        emitted[index] = true;
        completed(index, result);
    });
    // Admission failures, trace-only/paired modes and an interrupted device
    // group still terminate every request, with their ordinary UNKNOWN reply.
    for (index, result) in output.iter().enumerate() {
        if !emitted[index] { completed(index, result); }
    }
}

struct ActiveBatchGroup {
    context: ShadowContext,
    pending: Vec<(usize, IncrementalQuery, ShadowContext)>,
    batches: Vec<std::ops::Range<usize>>,
}

fn solve_active_batch_reporting(
    requests: Vec<(&DagCnfSolver, IncrementalQuery)>,
    min_batch_size: usize,
    completed: &mut impl FnMut(usize, &IncrementalResult),
) -> Vec<IncrementalResult> {
    let min_batch_size = min_batch_size.clamp(1, DEFAULT_SHADOW_BATCH_SIZE);
    let unknown = IncrementalResult::Unknown(super::cdcl::UnknownReason::BackendError);
    let output = vec![unknown.clone(); requests.len()];
    if requests.is_empty() || !(active_enabled() || paired_enabled()) {
        return output;
    }
    if !active_hardware_available() {
        ACTIVE_UNAVAILABLE_CALLS.fetch_add(1, Ordering::Relaxed);
        ACTIVE_UNAVAILABLE_QUERIES.fetch_add(requests.len() as u64, Ordering::Relaxed);
        return output;
    }
    let paired = paired_enabled();
    let trace_only = architecture_trace_enabled();
    let compare_cpu = paired || active_compare_cpu_enabled() || trace_only;
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
    if selected_count < min_batch_size {
        ACTIVE_SKIPPED_PASSES.fetch_add(1, Ordering::Relaxed);
        ACTIVE_SKIPPED_SMALL_BATCH.fetch_add(selected_count as u64, Ordering::Relaxed);
        return output;
    }
    let mut groups: Vec<ActiveBatchGroup> = Vec::new();
    let mut caches = Vec::new();
    let prefer_query_lemmas = !(active_resident_lemmas() || active_frame_ranges());
    // Exact/architecture traces encode one context per recorded batch. Keep
    // those diagnostic streams on their original grouping even if the arena
    // flag is present; production active mode can carry one context per query.
    let arena_views = active_arena_views_enabled() && !trace_only;
    // Keep the measured production path unchanged until the shared-domain ABI
    // closes the queue-economics gate. The projection flag opts into the
    // prerequisite cross-snapshot merge automatically for native simulation.
    let merge_contexts = arena_views && cross_context_batch_enabled();
    for (index, (solver, query)) in requests.iter().enumerate() {
        if !selected[index] {
            continue;
        }
        let cache_index = batched_solver_cache_index(&mut caches, solver);
        let Some((use_query_lemmas, query_words)) =
            caches[cache_index].query_plan(query, prefer_query_lemmas)
        else {
            ACTIVE_ERROR.fetch_add(1, Ordering::Relaxed);
            continue;
        };
        if query_words
            .checked_add(4)
            .is_none_or(|words| words > KERNEL_MAX_REQUEST_WORDS)
        {
            ACTIVE_ERROR.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        let query = caches[cache_index].prepare_query(query.clone(), use_query_lemmas);
        let context = caches[cache_index].context(use_query_lemmas);
        // An arena command does not require identical logical snapshots: it
        // interns their union once and carries one exact bitmap per query.
        // Grouping by full ShadowContext here defeated that protocol and
        // turned an eight-inquiry IC3 frontier into eight device launches.
        if let Some(group) = groups
            .iter_mut()
            .find(|group| batch_context_compatible(&group.context, &context, merge_contexts))
        {
            group.pending.push((index, query, context.clone()));
        } else {
            groups.push(ActiveBatchGroup {
                context: context.clone(),
                pending: vec![(index, query, context.clone())],
                batches: Vec::new(),
            });
        }
    }

    let mut planned = vec![false; requests.len()];
    let mut planned_count = 0usize;
    for group in &mut groups {
        if pair_scheduler_enabled() {
            schedule_query_pairs(&mut group.pending, |(_, query, _)| query);
        }
        let query_words: Vec<_> = group
            .pending
            .iter()
            .map(|(_, query, _)| {
                let words = query_request_words(query).unwrap_or(KERNEL_MAX_REQUEST_WORDS);
                if arena_views {
                    // Dense bitmap is the largest legal view update. Runtime
                    // still rechecks the process-lifetime union arena, which
                    // can be larger after switching between context groups.
                    words.saturating_add(
                        ARENA_VIEW_PREFIX_WORDS + QUALIFIED_ARENA_MAX_CLAUSES.div_ceil(32),
                    )
                } else {
                    words
                }
            })
            .collect();
        if merge_contexts && shared_domain_projection_enabled() {
            let domains: Vec<&[Var]> = group
                .pending
                .iter()
                .map(|(_, query, _)| query.domain.as_slice())
                .collect();
            let projected = plan_shared_domain_batch_ranges(
                &domains,
                &query_words,
                min_batch_size,
                active_batch_size(),
                KERNEL_MAX_REQUEST_WORDS,
            );
            let projected_queries = projected.iter().map(|range| range.len()).sum::<usize>();
            let saved_words = projected.iter().fold(0usize, |total, range| {
                total.saturating_add(
                    encoded_domain_words(domains[range.start])
                        .saturating_mul(range.len().saturating_sub(1)),
                )
            });
            ACTIVE_SHARED_DOMAIN_PROJECTED_QUERIES
                .fetch_add(projected_queries as u64, Ordering::Relaxed);
            ACTIVE_SHARED_DOMAIN_PROJECTED_BATCHES
                .fetch_add(projected.len() as u64, Ordering::Relaxed);
            ACTIVE_SHARED_DOMAIN_PROJECTED_SAVED_WORDS
                .fetch_add(saved_words as u64, Ordering::Relaxed);
        }
        group.batches = plan_full_batch_ranges(
            &query_words,
            min_batch_size,
            active_batch_size(),
            KERNEL_MAX_REQUEST_WORDS,
        );
        let group_planned: usize = group.batches.iter().map(|range| range.len()).sum();
        ACTIVE_SKIPPED_SMALL_BATCH.fetch_add(
            group.pending.len().saturating_sub(group_planned) as u64,
            Ordering::Relaxed,
        );
        for range in &group.batches {
            for (index, _, _) in &group.pending[range.clone()] {
                planned[*index] = true;
            }
        }
        planned_count += group_planned;
    }
    groups.retain(|group| !group.batches.is_empty());
    if planned_count == 0 {
        ACTIVE_SKIPPED_PASSES.fetch_add(1, Ordering::Relaxed);
        return output;
    }
    ACTIVE_OFFERED_PASSES.fetch_add(1, Ordering::Relaxed);
    ACTIVE_OFFERED.fetch_add(planned_count as u64, Ordering::Relaxed);
    let retain_exact_result =
        trace_only && std::env::var_os("INDUCTOR_CDCL_EXACT_REPLAY").is_some();
    let trace_ready_unix_ns = trace_only
        .then(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        })
        .unwrap_or(0);
    let reference_cpu = compare_cpu
        .then(|| measure_reference_cpu(&requests, &planned, paired, retain_exact_result));
    if let Some(cpu) = reference_cpu.as_ref() {
        PAIRED_BASELINE_CPU_NS.fetch_add(
            cpu.iter().map(|work| work.elapsed_ns).sum(),
            Ordering::Relaxed,
        );
    }

    if trace_only {
        if let Some(cpu) = reference_cpu.as_ref() {
            let mut remaining: usize = groups
                .iter()
                .flat_map(|group| group.batches.iter())
                .map(std::ops::Range::len)
                .sum();
            for group in &groups {
                for range in &group.batches {
                    record_architecture_trace_batch(
                        pass_id,
                        trace_ready_unix_ns,
                        &group.context,
                        &group.pending[range.clone()],
                        cpu,
                    );
                    remaining = remaining.saturating_sub(range.len());
                    record_exact_replay_batch(
                        pass_id,
                        &group.context,
                        &group.pending[range.clone()],
                        cpu,
                        remaining == 0,
                    );
                }
            }
        }
        // The trace is observational. Returning UNKNOWN keeps the live IC3
        // solver on its ordinary exact GipSAT path and prevents simulation
        // parameters from affecting proof state or verdict.
        return output;
    }

    // Nothing below this boundary can read a source solver or its non-owning
    // DagCnf pointer, even after the callback releases an original client.
    drop(caches);
    drop(requests);
    if !paired {
        for (index, is_planned) in planned.iter().enumerate() {
            if !is_planned { completed(index, &output[index]); }
        }
    }
    execute_prepared_active_batch(groups, output, arena_views, paired, pass_id,
                                  reference_cpu, paired_preflight, completed)
}

fn execute_prepared_active_batch(
    groups: Vec<ActiveBatchGroup>,
    mut output: Vec<IncrementalResult>,
    arena_views: bool,
    paired: bool,
    pass_id: u64,
    reference_cpu: Option<Vec<PairedCpuWork>>,
    paired_preflight: Option<Vec<PairedPreflightWork>>,
    completed: &mut impl FnMut(usize, &IncrementalResult),
) -> Vec<IncrementalResult> {
    let state_wait_start = std::time::Instant::now();
    let state = active_state().lock();
    ACTIVE_STATE_WAIT_NS.fetch_add(
        state_wait_start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        Ordering::Relaxed,
    );
    let Ok(mut state) = state else {
        let n_pending: usize = groups.iter().map(|group| group.pending.len()).sum();
        ACTIVE_ERROR.fetch_add(n_pending as u64, Ordering::Relaxed);
        return output;
    };
    'groups: for mut group in groups {
        let mut context_load_ns = 0u64;
        let context_update = plan_context_update(state.loaded_context.as_ref(), &group.context);
        let mut context_ready = context_update == ContextUpdate::Ready;
        let mut append_clauses = match context_update {
            ContextUpdate::Append(clauses) => Some(clauses),
            ContextUpdate::Ready | ContextUpdate::Reload => None,
        };
        if arena_views {
            // The physical union and private views are owned by HardwareCdcl;
            // the legacy exact-snapshot cache must not claim residency.
            state.loaded_context = None;
            context_ready = true;
            append_clauses = None;
        }
        let batches = std::mem::take(&mut group.batches);
        for range in batches {
            let start = range.start;
            let end = range.end;
            let queries: Vec<_> = group.pending[start..end]
                .iter()
                .map(|(_, query, _)| query.clone())
                .collect();
            let contexts: Vec<_> = group.pending[start..end]
                .iter()
                .map(|(_, _, context)| context.clone())
                .collect();
            ACTIVE_BATCHES.fetch_add(1, Ordering::Relaxed);
            let mut batch_ns = 0u64;
            let mut result = None;
            if arena_views {
                let batch_start = std::time::Instant::now();
                let kernel_before = direct_kernel_ns();
                result = Some(
                    state
                        .hardware
                        .as_mut()
                        .ok_or(HardwareError::Unavailable)
                        .and_then(|hardware| {
                            hardware.solve_arena_batch_contexts(&contexts, &queries)
                        }),
                );
                ACTIVE_BATCH_KERNEL_NS.fetch_add(
                    direct_kernel_ns().saturating_sub(kernel_before),
                    Ordering::Relaxed,
                );
                batch_ns = batch_start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            }
            if !context_ready && let Some(clauses) = append_clauses.take() {
                let append_start = std::time::Instant::now();
                let kernel_before = direct_kernel_ns();
                let appended = state
                    .hardware
                    .as_mut()
                    .ok_or(HardwareError::Unavailable)
                    .and_then(|hardware| hardware.add_frame_clauses(&clauses));
                ACTIVE_CONTEXT_APPEND_KERNEL_NS.fetch_add(
                    direct_kernel_ns().saturating_sub(kernel_before),
                    Ordering::Relaxed,
                );
                let append_ns = append_start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                ACTIVE_CONTEXT_APPEND_NS.fetch_add(append_ns, Ordering::Relaxed);
                match appended {
                    Ok(()) => {
                        ACTIVE_CONTEXT_APPENDS.fetch_add(1, Ordering::Relaxed);
                        ACTIVE_CONTEXT_APPEND_CLAUSES
                            .fetch_add(clauses.len() as u64, Ordering::Relaxed);
                        if let Some(loaded) = state.loaded_context.as_mut() {
                            loaded.clauses.extend(clauses);
                            context_ready = true;
                            context_load_ns = append_ns;
                        } else {
                            // The planner can only return Append for a loaded
                            // context. Keep this defensive branch exact if the
                            // state representation changes later.
                            ACTIVE_CONTEXT_APPEND_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    Err(error) => {
                        eprintln!("inductor-cdcl: incremental context append failed: {error}");
                        ACTIVE_CONTEXT_APPEND_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                        // ADD_FRAME_CLAUSES is atomic in the kernel, but a
                        // transport/server failure also invalidates the lease.
                        state.loaded_context = None;
                    }
                }
            }
            if !context_ready {
                let combined_start = std::time::Instant::now();
                let kernel_before = direct_kernel_ns();
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
                ACTIVE_COMBINED_KERNEL_NS.fetch_add(
                    direct_kernel_ns().saturating_sub(kernel_before),
                    Ordering::Relaxed,
                );
                let combined_ns = combined_start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                ACTIVE_COMBINED_NS.fetch_add(combined_ns, Ordering::Relaxed);
                match combined {
                    Ok(results) => {
                        ACTIVE_COMBINED_BATCHES.fetch_add(1, Ordering::Relaxed);
                        ACTIVE_CONTEXT_LOADS.fetch_add(1, Ordering::Relaxed);
                        context_ready = true;
                        state.loaded_context = Some(LoadedContext::from(&group.context));
                        batch_ns = combined_ns;
                        result = Some(Ok(results));
                    }
                    Err(error) => {
                        eprintln!("inductor-cdcl: combined load/run failed: {error}");
                        ACTIVE_COMBINED_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                        ACTIVE_COMBINED_FALLBACK_NS.fetch_add(combined_ns, Ordering::Relaxed);
                        state.loaded_context = None;
                        if deterministic_active_failure(&error) {
                            if let Some(query) = queries.first() {
                                let max_lit_var = query
                                    .assumptions
                                    .iter()
                                    .chain(query.constraints.iter().flatten())
                                    .map(|lit| u32::from(*lit) >> 1)
                                    .max()
                                    .unwrap_or(0);
                                let max_domain_var = query
                                    .domain
                                    .iter()
                                    .map(|var| u32::from(*var))
                                    .max()
                                    .unwrap_or(0);
                                let constraint_words: usize = query
                                    .constraints
                                    .iter()
                                    .map(|clause| 1 + clause.len())
                                    .sum();
                                eprintln!(
                                    "inductor-cdcl: rejected context/query shape vars {} clauses {} assumptions {} constraint-words {} domain {} max-lit-var {} max-domain-var {} frame {}",
                                    group.context.n_var,
                                    group.context.clauses.len(),
                                    query.assumptions.len(),
                                    constraint_words,
                                    query.domain.len(),
                                    max_lit_var,
                                    max_domain_var,
                                    query.frame,
                                );
                            }
                            // A context outside the compiled hardware envelope
                            // is an ordinary UNKNOWN/CAPACITY outcome. It must
                            // route to CPU once, not manufacture one protocol
                            // error for every inquiry already packed in the
                            // frontier batch.
                            if active_failure_is_capacity(&error) {
                                ACTIVE_UNKNOWN.fetch_add((end - start) as u64, Ordering::Relaxed);
                            } else {
                                ACTIVE_ERROR.fetch_add((end - start) as u64, Ordering::Relaxed);
                            }
                            disable_active_hardware(&error);
                            break 'groups;
                        }
                        let load_start = std::time::Instant::now();
                        let kernel_before = direct_kernel_ns();
                        let loaded = state
                            .hardware
                            .as_mut()
                            .ok_or(HardwareError::Unavailable)
                            .and_then(|hardware| {
                                hardware.load_context(group.context.n_var, &group.context.clauses)
                            });
                        ACTIVE_CONTEXT_LOAD_KERNEL_NS.fetch_add(
                            direct_kernel_ns().saturating_sub(kernel_before),
                            Ordering::Relaxed,
                        );
                        context_load_ns =
                            load_start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                        ACTIVE_CONTEXT_LOAD_NS.fetch_add(context_load_ns, Ordering::Relaxed);
                        if let Err(error) = loaded {
                            eprintln!("inductor-cdcl: fallback context load failed: {error}");
                            ACTIVE_ERROR.fetch_add((end - start) as u64, Ordering::Relaxed);
                            // One solver process owns one transition system.
                            // Repeating an exact-load failure for every block
                            // frontier only manufactures RPC/stale-lease work.
                            disable_active_hardware(&error);
                            break 'groups;
                        }
                        ACTIVE_CONTEXT_LOADS.fetch_add(1, Ordering::Relaxed);
                        context_ready = true;
                        state.loaded_context = Some(LoadedContext::from(&group.context));
                    }
                }
            }
            let result = match result {
                Some(result) => result,
                None => {
                    let batch_start = std::time::Instant::now();
                    let kernel_before = direct_kernel_ns();
                    let result = state
                        .hardware
                        .as_mut()
                        .ok_or(HardwareError::Unavailable)
                        .and_then(|hardware| hardware.solve_batch(&queries));
                    ACTIVE_BATCH_KERNEL_NS.fetch_add(
                        direct_kernel_ns().saturating_sub(kernel_before),
                        Ordering::Relaxed,
                    );
                    batch_ns = batch_start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                    result
                }
            };
            // A shared persistent service can reject RUN_BATCH when another
            // client replaced the resident context between commands. Treat
            // every command failure as a lost context lease: this batch stays
            // UNKNOWN and the next one atomically reloads its exact snapshot.
            if result.is_err() {
                context_ready = false;
                state.loaded_context = None;
            }
            ACTIVE_BATCH_NS.fetch_add(batch_ns, Ordering::Relaxed);
            if result.is_ok() {
                if let Some(hardware) = state.hardware.as_ref() {
                    profile_hardware_batch(&queries, &hardware.last_batch_records);
                    ACTIVE_HW_DECISIONS
                        .fetch_add(hardware.last_batch_work.decisions, Ordering::Relaxed);
                    ACTIVE_HW_CONFLICTS
                        .fetch_add(hardware.last_batch_work.conflicts, Ordering::Relaxed);
                    ACTIVE_HW_PROPAGATIONS
                        .fetch_add(hardware.last_batch_work.propagations, Ordering::Relaxed);
                    ACTIVE_HW_LEARNTS
                        .fetch_add(hardware.last_batch_work.learnt_clauses, Ordering::Relaxed);
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
                Ok(results) if results.len() == end - start => {
                    for ((index, _, _), result) in group.pending[start..end].iter().zip(results) {
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
                Ok(_) => {
                    // Do not publish a prefix of a malformed physical reply.
                    ACTIVE_ERROR.fetch_add((end - start) as u64, Ordering::Relaxed);
                    note_active_block_broker_reply_error();
                    context_ready = false;
                    state.loaded_context = None;
                }
                Err(error) => {
                    // A top-level request/arena capacity rejection means that
                    // none of these inquiries ran.  Report ordinary UNKNOWNs
                    // so every caller takes its exact CPU path; do not count a
                    // proof-safe heterogeneous fallback as a wrong hardware
                    // answer.  The same resident arena/request shape cannot
                    // become legal on a retry, so stop submitting further
                    // batches from this solver process.  Per-query capacity
                    // results arrive through Ok(results) above and therefore
                    // remain local to just that inquiry.
                    if active_failure_is_capacity(&error) {
                        ACTIVE_UNKNOWN.fetch_add((end - start) as u64, Ordering::Relaxed);
                    } else {
                        ACTIVE_ERROR.fetch_add((end - start) as u64, Ordering::Relaxed);
                    }
                    if deterministic_active_failure(&error) {
                        disable_active_hardware(&error);
                        break 'groups;
                    }
                }
            }
            if !paired {
                for (index, _, _) in &group.pending[start..end] {
                    completed(*index, &output[*index]);
                }
            }
        }
    }
    if paired {
        output.fill(IncrementalResult::Unknown(UnknownReason::BackendError));
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

pub fn note_active_trusted_sat(accepted: bool, install_ns: u64) {
    if accepted {
        ACTIVE_TRUSTED_SAT_INSTALLED.fetch_add(1, Ordering::Relaxed);
    } else {
        ACTIVE_TRUSTED_SAT_REJECTED.fetch_add(1, Ordering::Relaxed);
    }
    note_active_sat_model(accepted, install_ns);
}

pub fn note_active_trusted_sat_stale() {
    ACTIVE_TRUSTED_SAT_STALE_REVALIDATED.fetch_add(1, Ordering::Relaxed);
}

pub fn note_active_trusted_sat_revision_reused() {
    ACTIVE_TRUSTED_SAT_REVISION_REUSED.fetch_add(1, Ordering::Relaxed);
}

pub fn note_active_materialized_sat_prepared(accepted: bool, elapsed_ns: u64) {
    if accepted {
        ACTIVE_MATERIALIZED_SAT_PREPARED.fetch_add(1, Ordering::Relaxed);
    } else {
        ACTIVE_MATERIALIZED_SAT_REJECTED.fetch_add(1, Ordering::Relaxed);
    }
    ACTIVE_MATERIALIZED_SAT_PREPARE_NS.fetch_add(elapsed_ns, Ordering::Relaxed);
}

pub fn note_active_materialized_sat_used() {
    ACTIVE_MATERIALIZED_SAT_USED.fetch_add(1, Ordering::Relaxed);
}

pub fn note_active_sat_lift(
    attempted: bool,
    succeeded: bool,
    full_lits: usize,
    result_lits: usize,
    elapsed_ns: u64,
) {
    if attempted {
        ACTIVE_SAT_LIFT_ATTEMPTED.fetch_add(1, Ordering::Relaxed);
    }
    if succeeded {
        ACTIVE_SAT_LIFT_SUCCEEDED.fetch_add(1, Ordering::Relaxed);
    }
    ACTIVE_SAT_FULL_LITS.fetch_add(full_lits as u64, Ordering::Relaxed);
    ACTIVE_SAT_LIFTED_LITS.fetch_add(result_lits as u64, Ordering::Relaxed);
    ACTIVE_SAT_LIFT_NS.fetch_add(elapsed_ns, Ordering::Relaxed);
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

pub fn note_active_trusted_unsat(
    accepted: bool,
    assumption_lits: usize,
    hardware_core_lits: usize,
    install_ns: u64,
) {
    ACTIVE_UNSAT_VALIDATE_NS.fetch_add(install_ns, Ordering::Relaxed);
    ACTIVE_UNSAT_ASSUMPTION_LITS.fetch_add(assumption_lits as u64, Ordering::Relaxed);
    ACTIVE_UNSAT_HW_CORE_LITS.fetch_add(hardware_core_lits as u64, Ordering::Relaxed);
    if accepted {
        ACTIVE_TRUSTED_UNSAT_INSTALLED.fetch_add(1, Ordering::Relaxed);
        ACTIVE_UNSAT_CORE_USED.fetch_add(1, Ordering::Relaxed);
    } else {
        ACTIVE_TRUSTED_UNSAT_REJECTED.fetch_add(1, Ordering::Relaxed);
        ACTIVE_UNSAT_CORE_REJECTED.fetch_add(1, Ordering::Relaxed);
        ACTIVE_CPU_FALLBACK.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn note_active_cpu_fallback() {
    ACTIVE_CPU_FALLBACK.fetch_add(1, Ordering::Relaxed);
}

pub fn note_active_mic_wave(results: &[IncrementalResult]) {
    ACTIVE_MIC_BATCH_WAVES.fetch_add(1, Ordering::Relaxed);
    ACTIVE_MIC_BATCH_QUERIES.fetch_add(results.len() as u64, Ordering::Relaxed);
    for result in results {
        match result {
            IncrementalResult::Sat { .. } => {
                ACTIVE_MIC_BATCH_SAT.fetch_add(1, Ordering::Relaxed);
            }
            IncrementalResult::Unsat { .. } => {
                ACTIVE_MIC_BATCH_UNSAT.fetch_add(1, Ordering::Relaxed);
            }
            IncrementalResult::Unknown(_) => {
                ACTIVE_MIC_BATCH_UNKNOWN.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

pub fn note_active_mic_consumed(unsat: bool, accepted: bool) {
    if accepted {
        if unsat {
            ACTIVE_MIC_BATCH_UNSAT_USED.fetch_add(1, Ordering::Relaxed);
        } else {
            ACTIVE_MIC_BATCH_SAT_USED.fetch_add(1, Ordering::Relaxed);
        }
    } else {
        ACTIVE_MIC_BATCH_REJECTED.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn note_active_mic_invalidated(count: usize) {
    if count == 0 {
        return;
    }
    ACTIVE_MIC_BATCH_INVALIDATED.fetch_add(count as u64, Ordering::Relaxed);
}

pub fn note_active_mic_shadow_result(replaceable: bool) {
    ACTIVE_MIC_BATCH_SHADOW_REACHED.fetch_add(1, Ordering::Relaxed);
    if replaceable {
        ACTIVE_MIC_BATCH_SHADOW_REPLACEABLE.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn note_active_mic_shadow_invalidated(count: usize) {
    if count != 0 {
        ACTIVE_MIC_BATCH_SHADOW_INVALIDATED.fetch_add(count as u64, Ordering::Relaxed);
    }
}

pub fn note_active_mic_batch_economics(
    projected_cpu_ns: u64,
    projected_hardware_ns: Option<u64>,
    probe: bool,
    selected: bool,
) {
    ACTIVE_MIC_BATCH_ECON_CPU_NS.store(projected_cpu_ns, Ordering::Relaxed);
    ACTIVE_MIC_BATCH_ECON_HW_NS.store(projected_hardware_ns.unwrap_or(0), Ordering::Relaxed);
    ACTIVE_MIC_BATCH_ECON_HW_VALID.store(projected_hardware_ns.is_some(), Ordering::Relaxed);
    if probe {
        ACTIVE_MIC_BATCH_ECON_PROBES.fetch_add(1, Ordering::Relaxed);
    } else if selected {
        ACTIVE_MIC_BATCH_ECON_OFFLOADS.fetch_add(1, Ordering::Relaxed);
    } else {
        ACTIVE_MIC_BATCH_ECON_REJECTS.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn note_active_mic_chain_validation(accepted: bool, verify_ns: u64) {
    ACTIVE_MIC_CHAIN_VERIFY_NS.fetch_add(verify_ns, Ordering::Relaxed);
    if accepted {
        ACTIVE_MIC_CHAIN_VALIDATED.fetch_add(1, Ordering::Relaxed);
    } else {
        ACTIVE_MIC_CHAIN_REJECTED.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn note_active_mic_chain_cpu_replaced(trials: u32) {
    ACTIVE_MIC_CHAIN_CPU_LOOPS_REPLACED.fetch_add(1, Ordering::Relaxed);
    ACTIVE_MIC_CHAIN_CPU_TRIALS_REPLACED.fetch_add(u64::from(trials), Ordering::Relaxed);
}

pub fn note_active_mic_chain_economics(
    route: u8,
    context_reusable: bool,
    projected_cpu_ns: u64,
    projected_hardware_ns: Option<u64>,
) {
    let (total, warm) = match route {
        0 => (
            &ACTIVE_MIC_CHAIN_ECON_REJECTS,
            &ACTIVE_MIC_CHAIN_ECON_WARM_REJECTS,
        ),
        1 => (
            &ACTIVE_MIC_CHAIN_ECON_PROBES,
            &ACTIVE_MIC_CHAIN_ECON_WARM_PROBES,
        ),
        2 => (
            &ACTIVE_MIC_CHAIN_ECON_OFFLOADS,
            &ACTIVE_MIC_CHAIN_ECON_WARM_OFFLOADS,
        ),
        _ => return,
    };
    total.fetch_add(1, Ordering::Relaxed);
    if context_reusable {
        warm.fetch_add(1, Ordering::Relaxed);
    }
    ACTIVE_MIC_CHAIN_ECON_CPU_NS.store(projected_cpu_ns, Ordering::Relaxed);
    ACTIVE_MIC_CHAIN_ECON_HW_NS.store(projected_hardware_ns.unwrap_or(0), Ordering::Relaxed);
    ACTIVE_MIC_CHAIN_ECON_HW_VALID.store(projected_hardware_ns.is_some(), Ordering::Relaxed);
}

pub fn note_active_block_cost_rejected() {
    ACTIVE_BLOCK_COST_REJECTED.fetch_add(1, Ordering::Relaxed);
}

pub fn note_active_block_cpu_sample(elapsed_ns: u64) {
    ACTIVE_BLOCK_CPU_SAMPLES.fetch_add(1, Ordering::Relaxed);
    ACTIVE_BLOCK_CPU_NS.fetch_add(elapsed_ns, Ordering::Relaxed);
}

pub fn note_active_block_calibration(profitable: bool, elapsed_ns: u64) {
    ACTIVE_BLOCK_CALIBRATIONS.fetch_add(1, Ordering::Relaxed);
    if profitable {
        ACTIVE_BLOCK_CALIBRATION_PROFITABLE.fetch_add(1, Ordering::Relaxed);
    }
    ACTIVE_BLOCK_CALIBRATION_NS.fetch_add(elapsed_ns, Ordering::Relaxed);
}

pub fn note_active_block_route_decision(enabled: bool) {
    if enabled {
        ACTIVE_BLOCK_ROUTE_ENABLES.fetch_add(1, Ordering::Relaxed);
    } else {
        ACTIVE_BLOCK_ROUTE_DISABLES.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn note_active_block_route_observation(representative_ns: u64, enabled: bool) {
    ACTIVE_BLOCK_ROUTE_REPRESENTATIVE_NS.store(representative_ns, Ordering::Relaxed);
    ACTIVE_BLOCK_ROUTE_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Monotonic process-local service counters around one synchronous block
/// submission. The elapsed time includes the persistent-service round trip,
/// queueing behind another portfolio client, and a combined context load when
/// one was required, so the adaptive route learns the cost the caller really
/// paid instead of a kernel-only idealization.
pub fn active_batch_service_snapshot() -> (u64, u64, u64) {
    (
        ACTIVE_BATCHES.load(Ordering::Relaxed),
        ACTIVE_OFFERED.load(Ordering::Relaxed),
        ACTIVE_BATCH_NS.load(Ordering::Relaxed),
    )
}

pub fn note_active_block_batch_economics(
    projected_cpu_ns: u64,
    projected_hardware_ns: Option<u64>,
    probe: bool,
    selected: bool,
) {
    ACTIVE_BLOCK_BATCH_ECON_CPU_NS.store(projected_cpu_ns, Ordering::Relaxed);
    ACTIVE_BLOCK_BATCH_ECON_HW_NS.store(projected_hardware_ns.unwrap_or(0), Ordering::Relaxed);
    ACTIVE_BLOCK_BATCH_ECON_HW_VALID.store(projected_hardware_ns.is_some(), Ordering::Relaxed);
    ACTIVE_BLOCK_BATCH_ECON_ROUTE.store(selected && !probe, Ordering::Relaxed);
    if probe {
        ACTIVE_BLOCK_BATCH_ECON_PROBES.fetch_add(1, Ordering::Relaxed);
    } else if selected {
        ACTIVE_BLOCK_BATCH_ECON_OFFLOADS.fetch_add(1, Ordering::Relaxed);
    } else {
        ACTIVE_BLOCK_BATCH_ECON_REJECTS.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn note_active_block_selected_result(conclusive: bool) {
    if conclusive {
        ACTIVE_BLOCK_HW_CONCLUSIVE.fetch_add(1, Ordering::Relaxed);
    } else {
        ACTIVE_BLOCK_SELECTED_NO_ANSWER.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn note_active_block_result_consumed(accepted: bool, cache_age: u64) {
    if accepted {
        ACTIVE_BLOCK_RESULT_USED.fetch_add(1, Ordering::Relaxed);
        if cache_age != 0 {
            ACTIVE_BLOCK_CACHE_REUSE_USED.fetch_add(1, Ordering::Relaxed);
        }
    } else {
        ACTIVE_BLOCK_RESULT_REJECTED.fetch_add(1, Ordering::Relaxed);
        if cache_age != 0 {
            ACTIVE_BLOCK_CACHE_REUSE_REJECTED.fetch_add(1, Ordering::Relaxed);
        }
    }
}

pub fn note_active_block_cache_reused(age: u64) {
    ACTIVE_BLOCK_CACHE_REUSED.fetch_add(1, Ordering::Relaxed);
    ACTIVE_BLOCK_CACHE_REUSE_AGE.fetch_add(age, Ordering::Relaxed);
}

pub fn note_active_block_cache_replaced() {
    ACTIVE_BLOCK_CACHE_REPLACED.fetch_add(1, Ordering::Relaxed);
}

pub fn note_active_block_cache_evicted(n: usize) {
    ACTIVE_BLOCK_CACHE_EVICTED.fetch_add(n as u64, Ordering::Relaxed);
}

pub fn note_active_block_preflight(decision: &ActivePreflight) {
    match decision {
        ActivePreflight::Conclusive(_) => {
            ACTIVE_BLOCK_PREFLIGHT_CONCLUSIVE.fetch_add(1, Ordering::Relaxed);
        }
        ActivePreflight::Fpga => {
            ACTIVE_BLOCK_PREFLIGHT_SELECTED.fetch_add(1, Ordering::Relaxed);
        }
        ActivePreflight::CpuFallback => {
            ACTIVE_BLOCK_PREFLIGHT_FALLBACK.fetch_add(1, Ordering::Relaxed);
        }
    }
}

pub fn note_active_block_wave_reserved(n: usize) {
    ACTIVE_BLOCK_WAVE_RESERVED.fetch_add(n as u64, Ordering::Relaxed);
}

pub fn note_active_block_wave_taken() {
    ACTIVE_BLOCK_WAVE_TAKEN.fetch_add(1, Ordering::Relaxed);
}

pub fn note_active_block_async_launch(prepare_ns: u64) {
    ACTIVE_BLOCK_ASYNC_LAUNCHED.fetch_add(1, Ordering::Relaxed);
    ACTIVE_BLOCK_ASYNC_PREPARE_NS.fetch_add(prepare_ns, Ordering::Relaxed);
}

pub fn note_active_block_async_discarded(queries: usize) {
    ACTIVE_BLOCK_ASYNC_DISCARDED.fetch_add(queries as u64, Ordering::Relaxed);
}

pub fn note_active_block_async_cpu_race() {
    ACTIVE_BLOCK_ASYNC_CPU_RACES.fetch_add(1, Ordering::Relaxed);
}

pub fn note_active_block_broker_dispatch(jobs: usize, queries: usize, queued_ns: u64) {
    ACTIVE_BLOCK_BROKER_GROUPS.fetch_add(1, Ordering::Relaxed);
    ACTIVE_BLOCK_BROKER_JOBS.fetch_add(jobs as u64, Ordering::Relaxed);
    ACTIVE_BLOCK_BROKER_QUERIES.fetch_add(queries as u64, Ordering::Relaxed);
    ACTIVE_BLOCK_BROKER_QUEUE_NS.fetch_add(queued_ns, Ordering::Relaxed);
}

pub fn note_active_block_broker_reply_error() {
    ACTIVE_BLOCK_BROKER_REPLY_ERRORS.fetch_add(1, Ordering::Relaxed);
}

pub fn note_active_block_broker_stream(replies: usize, tail_ns: u64) {
    ACTIVE_BLOCK_BROKER_STREAM_REPLIES.fetch_add(replies as u64, Ordering::Relaxed);
    ACTIVE_BLOCK_BROKER_REPLY_TAIL_NS.fetch_add(tail_ns, Ordering::Relaxed);
}

pub fn note_active_block_async_root_tail(queries: usize) {
    ACTIVE_BLOCK_ASYNC_ROOT_TAIL.fetch_add(queries as u64, Ordering::Relaxed);
}

pub fn note_active_block_async_root_unused(queries: usize) {
    ACTIVE_BLOCK_ASYNC_ROOT_UNUSED.fetch_add(queries as u64, Ordering::Relaxed);
}

pub fn note_active_block_async_demand(ready: bool, elapsed_ns: u64) {
    ACTIVE_BLOCK_ASYNC_DEMANDS.fetch_add(1, Ordering::Relaxed);
    ACTIVE_BLOCK_ASYNC_DEMAND_READY.fetch_add(u64::from(ready), Ordering::Relaxed);
    ACTIVE_BLOCK_ASYNC_DEMAND_NS.fetch_add(elapsed_ns, Ordering::Relaxed);
}

pub fn note_active_block_async_harvest(wall_ns: u64, join_ns: u64) {
    ACTIVE_BLOCK_ASYNC_HARVESTED.fetch_add(1, Ordering::Relaxed);
    ACTIVE_BLOCK_ASYNC_WALL_NS.fetch_add(wall_ns, Ordering::Relaxed);
    ACTIVE_BLOCK_ASYNC_JOIN_NS.fetch_add(join_ns, Ordering::Relaxed);
}

pub fn note_active_push_prefetch_launch(n_queries: usize, prepare_ns: u64) {
    ACTIVE_PUSH_PREFETCH_LAUNCHED.fetch_add(1, Ordering::Relaxed);
    ACTIVE_PUSH_PREFETCH_QUERIES.fetch_add(n_queries as u64, Ordering::Relaxed);
    ACTIVE_PUSH_PREFETCH_PREPARE_NS.fetch_add(prepare_ns, Ordering::Relaxed);
}

pub fn note_active_push_prefetch_harvest(n_ready: usize, wall_ns: u64, join_ns: u64) {
    ACTIVE_PUSH_PREFETCH_HARVESTED.fetch_add(1, Ordering::Relaxed);
    ACTIVE_PUSH_PREFETCH_READY.fetch_add(n_ready as u64, Ordering::Relaxed);
    ACTIVE_PUSH_PREFETCH_WALL_NS.fetch_add(wall_ns, Ordering::Relaxed);
    ACTIVE_PUSH_PREFETCH_JOIN_NS.fetch_add(join_ns, Ordering::Relaxed);
}

fn push_prefetch_length_bucket(lemma_len: usize) -> usize {
    match lemma_len {
        0..=4 => 0,
        5..=8 => 1,
        9..=16 => 2,
        17..=32 => 3,
        _ => 4,
    }
}

pub fn note_active_push_prefetch_submit_length(lemma_len: usize) {
    ACTIVE_PUSH_PREFETCH_SUBMITTED_BY_LEN[push_prefetch_length_bucket(lemma_len)]
        .fetch_add(1, Ordering::Relaxed);
}

pub fn note_active_push_prefetch_ready_length(lemma_len: usize) {
    ACTIVE_PUSH_PREFETCH_READY_BY_LEN[push_prefetch_length_bucket(lemma_len)]
        .fetch_add(1, Ordering::Relaxed);
}

pub fn note_active_push_prefetch_hit(lemma_len: usize, accepted: bool) {
    let bucket = push_prefetch_length_bucket(lemma_len);
    ACTIVE_PUSH_PREFETCH_HITS.fetch_add(1, Ordering::Relaxed);
    ACTIVE_PUSH_PREFETCH_HITS_BY_LEN[bucket].fetch_add(1, Ordering::Relaxed);
    if accepted {
        ACTIVE_PUSH_PREFETCH_USED.fetch_add(1, Ordering::Relaxed);
        ACTIVE_PUSH_PREFETCH_USED_BY_LEN[bucket].fetch_add(1, Ordering::Relaxed);
    } else {
        ACTIVE_PUSH_PREFETCH_REJECTED.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn note_active_push_prefetch_admission(
    admitted: bool,
    reprobe: bool,
    evaluated_queries: usize,
    evaluated_used: usize,
) {
    if admitted {
        ACTIVE_PUSH_PREFETCH_ADMITTED.fetch_add(1, Ordering::Relaxed);
    } else {
        ACTIVE_PUSH_PREFETCH_SUPPRESSED.fetch_add(1, Ordering::Relaxed);
    }
    if reprobe {
        ACTIVE_PUSH_PREFETCH_REPROBES.fetch_add(1, Ordering::Relaxed);
    }
    ACTIVE_PUSH_PREFETCH_EVAL_QUERIES.store(evaluated_queries as u64, Ordering::Relaxed);
    ACTIVE_PUSH_PREFETCH_EVAL_USED.store(evaluated_used as u64, Ordering::Relaxed);
}

pub fn note_active_push_prefetch_skipped_long(n: usize) {
    ACTIVE_PUSH_PREFETCH_SKIPPED_LONG.fetch_add(n as u64, Ordering::Relaxed);
}

pub fn note_active_push_prefetch_skipped_context(n: usize) {
    ACTIVE_PUSH_PREFETCH_SKIPPED_CONTEXT.fetch_add(n as u64, Ordering::Relaxed);
}

pub fn note_active_push_prefetch_evicted(n: usize) {
    ACTIVE_PUSH_PREFETCH_EVICTED.fetch_add(n as u64, Ordering::Relaxed);
}

pub fn note_active_push_prefetch_busy() {
    ACTIVE_PUSH_PREFETCH_BUSY.fetch_add(1, Ordering::Relaxed);
}

pub fn note_active_preflight_result(unsat: bool, accepted: bool, restore_ns: u64) {
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
pub fn queue_shadow(solver: &DagCnfSolver, query: IncrementalQuery, cpu_result: Option<bool>) {
    if !shadow_enabled() || query.domain.is_empty() {
        return;
    }
    SHADOW_OFFERED.fetch_add(1, Ordering::Relaxed);
    let prefer_query_lemmas = std::env::var_os("INDUCTOR_CDCL_SHADOW_RESIDENT_LEMMAS").is_none();
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
    if architecture_trace_enabled()
        || exact_block_progress_enabled()
        || block_root_timeline_enabled()
        || std::env::var_os("INDUCTOR_CDCL_FULL_ROOT_TRANSCRIPT").is_some()
    {
        if architecture_trace_enabled() {
            flush_architecture_trace_writer();
        }
        if block_root_timeline_enabled() {
            flush_block_root_timeline_writer();
        }
        flush_exact_replay_writer();
        flush_full_root_transcript_writer();
        eprintln!(
            "inductor-cdcl: architecture trace batches {}, queries {}, roots {}, CSV {}; exact replay batches {}, queries {}, MICs {}, BLOCK checkpoints {}, event-only roots {}, frame-event waves {}, file {}; ranged snapshot fallbacks {}",
            ARCH_TRACE_BATCH_ID.load(Ordering::Relaxed),
            ARCH_TRACE_QUERIES.load(Ordering::Relaxed),
            ROOT_TRACE_ROOTS.load(Ordering::Relaxed),
            std::env::var("INDUCTOR_CDCL_TRACE_CSV").unwrap_or_else(|_| "disabled".to_string()),
            EXACT_REPLAY_BATCHES.load(Ordering::Relaxed),
            EXACT_REPLAY_QUERIES.load(Ordering::Relaxed),
            EXACT_REPLAY_MICS.load(Ordering::Relaxed),
            EXACT_REPLAY_BLOCK_PROGRESS.load(Ordering::Relaxed),
            EXACT_REPLAY_BLOCK_EVENTS.load(Ordering::Relaxed),
            EXACT_REPLAY_FRAME_EVENTS.load(Ordering::Relaxed),
            std::env::var("INDUCTOR_CDCL_EXACT_REPLAY").unwrap_or_else(|_| "disabled".to_string()),
            FRAME_RANGE_SNAPSHOT_MISMATCH.load(Ordering::Relaxed),
        );
        if std::env::var_os("INDUCTOR_CDCL_FULL_ROOT_TRANSCRIPT").is_some() {
            eprintln!(
                "inductor-cdcl: full-root transcript commands {}, request words {}, response words {}, wire rejects {}, step caps {}, file {}",
                FULL_ROOT_TRANSCRIPT_COMMANDS.load(Ordering::Relaxed),
                FULL_ROOT_TRANSCRIPT_REQUEST_WORDS.load(Ordering::Relaxed),
                FULL_ROOT_TRANSCRIPT_RESPONSE_WORDS.load(Ordering::Relaxed),
                FULL_ROOT_WIRE_REJECTS.load(Ordering::Relaxed),
                FULL_ROOT_STEP_CAPS.load(Ordering::Relaxed),
                std::env::var("INDUCTOR_CDCL_FULL_ROOT_TRANSCRIPT")
                    .unwrap_or_else(|_| "disabled".to_string()),
            );
        }
        if block_controller_sim_requested() {
            super::block_controller_sim::report();
        }
    }
    if paired_enabled() {
        flush_comparison_writer();
        let cpu_ns = PAIRED_CPU_NS.load(Ordering::Relaxed);
        let hw_ns = PAIRED_HW_NS.load(Ordering::Relaxed);
        let service_ratio = cpu_ns as f64 / hw_ns.max(1) as f64;
        eprintln!(
            "inductor-cdcl: paired selector frame >= {}, assumptions <= {}, passes {} (skipped {}, offered {}, max-ready {}), candidates {}, filtered {}, queries {}, batches {}, CPU/HW agree {}, mismatch {}, HW unknown {}, HW-faster batches {}, CPU-reference {:.3} ms, FPGA service {:.3} ms, service ratio {:.3}x, init {:.3} ms, context loads {} / {:.3} ms, appends {}/{} clauses (fallbacks {}) / {:.3} ms, combined ok/fallback {}/{} / {:.3} ms, errors {}, CSV {}",
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
            ACTIVE_CONTEXT_APPENDS.load(Ordering::Relaxed),
            ACTIVE_CONTEXT_APPEND_CLAUSES.load(Ordering::Relaxed),
            ACTIVE_CONTEXT_APPEND_FALLBACKS.load(Ordering::Relaxed),
            ACTIVE_CONTEXT_APPEND_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_COMBINED_BATCHES.load(Ordering::Relaxed),
            ACTIVE_COMBINED_FALLBACKS.load(Ordering::Relaxed),
            ACTIVE_COMBINED_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_ERROR.load(Ordering::Relaxed),
            std::env::var("INDUCTOR_CDCL_PAIRED_CSV").unwrap_or_else(|_| "disabled".to_string()),
        );
        if let Some(conflict_limit) = paired_preflight_conflicts() {
            let baseline_ns = PAIRED_BASELINE_CPU_NS.load(Ordering::Relaxed);
            let preflight_ns = PAIRED_PREFLIGHT_NS.load(Ordering::Relaxed);
            let hybrid_service_ns = preflight_ns.saturating_add(hw_ns);
            let hybrid_with_load_ns = hybrid_service_ns
                .saturating_add(ACTIVE_CONTEXT_LOAD_NS.load(Ordering::Relaxed))
                .saturating_add(ACTIVE_CONTEXT_APPEND_NS.load(Ordering::Relaxed))
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
        let hw_conclusive = ACTIVE_HW_SAT
            .load(Ordering::Relaxed)
            .saturating_add(ACTIVE_HW_UNSAT.load(Ordering::Relaxed));
        let hw_used = ACTIVE_SAT_USED
            .load(Ordering::Relaxed)
            .saturating_add(ACTIVE_UNSAT_CORE_USED.load(Ordering::Relaxed));
        let hw_rejected = ACTIVE_SAT_REJECTED
            .load(Ordering::Relaxed)
            .saturating_add(ACTIVE_UNSAT_CORE_REJECTED.load(Ordering::Relaxed));
        let hw_unconsumed = hw_conclusive.saturating_sub(hw_used.saturating_add(hw_rejected));
        let block_conclusive = ACTIVE_BLOCK_HW_CONCLUSIVE.load(Ordering::Relaxed);
        let block_used = ACTIVE_BLOCK_RESULT_USED.load(Ordering::Relaxed);
        let block_rejected = ACTIVE_BLOCK_RESULT_REJECTED.load(Ordering::Relaxed);
        let block_unconsumed =
            block_conclusive.saturating_sub(block_used.saturating_add(block_rejected));
        let push_conclusive = hw_conclusive.saturating_sub(block_conclusive);
        let push_used = hw_used.saturating_sub(block_used);
        // Keep this line comfortably below PIPE_BUF. Portfolio workers share
        // one redirected stderr; the detailed multi-kilobyte report can be
        // interleaved at a write boundary and is not a reliable qualification
        // record. The matrix runner aggregates one compact line per selected
        // FPGA worker.
        let (root_waves, root_work, root_service_ns) = super::block_controller_sim::root_metrics();
        let qualification = format!(
            "inductor-cdcl: qualification worker={} transport_attempted={} transport_unavailable={} hardware_disabled={} disable_count={} candidates={} batches={} hw_sat={} hw_unsat={} hw_unknown={} hw_errors={} batch_service_ms={:.3} mic_service_ms={:.3} root_waves={} root_work={} root_service_ms={:.3} block_conclusive={} block_used={} push_conclusive={} push_used={} cpu_fallback={} trusted_sat={} trusted_unsat={} trusted_rejected={} stale_sat={}\n",
            std::env::var("INDUCTOR_CDCL_PORTFOLIO_WORKER")
                .unwrap_or_else(|_| "standalone".to_string()),
            ACTIVE_INIT_NS.load(Ordering::Relaxed) != 0,
            ACTIVE_TRANSPORT_UNAVAILABLE.load(Ordering::Relaxed),
            ACTIVE_HARDWARE_DISABLED.load(Ordering::Relaxed),
            ACTIVE_HARDWARE_DISABLES.load(Ordering::Relaxed),
            ACTIVE_CANDIDATES.load(Ordering::Relaxed),
            ACTIVE_BATCHES.load(Ordering::Relaxed),
            ACTIVE_HW_SAT.load(Ordering::Relaxed),
            ACTIVE_HW_UNSAT.load(Ordering::Relaxed),
            ACTIVE_UNKNOWN.load(Ordering::Relaxed),
            ACTIVE_ERROR.load(Ordering::Relaxed),
            ACTIVE_BATCH_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_MIC_CHAIN_CLIENT_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            root_waves,
            root_work,
            root_service_ns as f64 / 1_000_000.0,
            block_conclusive,
            block_used,
            push_conclusive,
            push_used,
            ACTIVE_CPU_FALLBACK.load(Ordering::Relaxed),
            ACTIVE_TRUSTED_SAT_INSTALLED.load(Ordering::Relaxed),
            ACTIVE_TRUSTED_UNSAT_INSTALLED.load(Ordering::Relaxed),
            ACTIVE_TRUSTED_SAT_REJECTED
                .load(Ordering::Relaxed)
                .saturating_add(ACTIVE_TRUSTED_UNSAT_REJECTED.load(Ordering::Relaxed)),
            ACTIVE_TRUSTED_SAT_STALE_REVALIDATED.load(Ordering::Relaxed),
        );
        let _ = std::io::stderr().lock().write_all(qualification.as_bytes());
        eprintln!(
            "inductor-cdcl: async ablation discarded_queries={}",
            ACTIVE_BLOCK_ASYNC_DISCARDED.load(Ordering::Relaxed),
        );
        eprintln!(
            "inductor-cdcl: async lifecycle cpu_races={} root_tail_queries={} root_unused_answers={} demand_attempts={} demand_ready={} demand_wait_ms={:.3}",
            ACTIVE_BLOCK_ASYNC_CPU_RACES.load(Ordering::Relaxed),
            ACTIVE_BLOCK_ASYNC_ROOT_TAIL.load(Ordering::Relaxed),
            ACTIVE_BLOCK_ASYNC_ROOT_UNUSED.load(Ordering::Relaxed),
            ACTIVE_BLOCK_ASYNC_DEMANDS.load(Ordering::Relaxed),
            ACTIVE_BLOCK_ASYNC_DEMAND_READY.load(Ordering::Relaxed),
            ACTIVE_BLOCK_ASYNC_DEMAND_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
        );
        eprintln!(
            "inductor-cdcl: async broker groups={} jobs={} queries={} queue_wait_ms={:.3} state_wait_ms={:.3} reply_errors={} stream_replies={} reply_tail_ms={:.3}",
            ACTIVE_BLOCK_BROKER_GROUPS.load(Ordering::Relaxed),
            ACTIVE_BLOCK_BROKER_JOBS.load(Ordering::Relaxed),
            ACTIVE_BLOCK_BROKER_QUERIES.load(Ordering::Relaxed),
            ACTIVE_BLOCK_BROKER_QUEUE_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_STATE_WAIT_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_BLOCK_BROKER_REPLY_ERRORS.load(Ordering::Relaxed),
            ACTIVE_BLOCK_BROKER_STREAM_REPLIES.load(Ordering::Relaxed),
            ACTIVE_BLOCK_BROKER_REPLY_TAIL_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
        );
        let kernel_ns = direct_kernel_ns();
        if kernel_ns != 0 {
            let load_kernel_ns = ACTIVE_CONTEXT_LOAD_KERNEL_NS.load(Ordering::Relaxed);
            let append_kernel_ns = ACTIVE_CONTEXT_APPEND_KERNEL_NS.load(Ordering::Relaxed);
            let combined_kernel_ns = ACTIVE_COMBINED_KERNEL_NS.load(Ordering::Relaxed);
            let batch_kernel_ns = ACTIVE_BATCH_KERNEL_NS.load(Ordering::Relaxed);
            let mic_kernel_ns = ACTIVE_MIC_CHAIN_KERNEL_NS.load(Ordering::Relaxed);
            let fused_append_mic_kernel_ns =
                ACTIVE_FUSED_APPEND_MIC_KERNEL_NS.load(Ordering::Relaxed);
            let materialize_kernel_ns = ACTIVE_FRAME_MATERIALIZE_KERNEL_NS.load(Ordering::Relaxed);
            let useful_kernel_ns = batch_kernel_ns.saturating_add(mic_kernel_ns);
            let maintenance_kernel_ns = load_kernel_ns
                .saturating_add(append_kernel_ns)
                .saturating_add(materialize_kernel_ns);
            let attributed_kernel_ns = load_kernel_ns
                .saturating_add(append_kernel_ns)
                .saturating_add(combined_kernel_ns)
                .saturating_add(batch_kernel_ns)
                .saturating_add(materialize_kernel_ns)
                .saturating_add(mic_kernel_ns)
                .saturating_add(fused_append_mic_kernel_ns);
            eprintln!(
                "inductor-cdcl: direct XRT kernel busy {:.3} ms (kernel start-to-completion only; excludes host packing and DMA)",
                kernel_ns as f64 / 1_000_000.0,
            );
            eprintln!(
                "inductor-cdcl: direct XRT kernel split load/append/combined-load-run/batch/frame-materialize/useful-MIC/fused-append+MIC-mixed/unattributed {:.3}/{:.3}/{:.3}/{:.3}/{:.3}/{:.3}/{:.3}/{:.3} ms",
                load_kernel_ns as f64 / 1_000_000.0,
                append_kernel_ns as f64 / 1_000_000.0,
                combined_kernel_ns as f64 / 1_000_000.0,
                batch_kernel_ns as f64 / 1_000_000.0,
                materialize_kernel_ns as f64 / 1_000_000.0,
                mic_kernel_ns as f64 / 1_000_000.0,
                fused_append_mic_kernel_ns as f64 / 1_000_000.0,
                kernel_ns.saturating_sub(attributed_kernel_ns) as f64 / 1_000_000.0,
            );
            eprintln!(
                "inductor-cdcl: direct XRT classified busy useful-CDCL {:.3} ms ({:.1}%), resident/frame maintenance {:.3} ms ({:.1}%), fused append+MIC mixed {:.3} ms ({:.1}%); mixed commands are not counted as useful or maintenance",
                useful_kernel_ns as f64 / 1_000_000.0,
                100.0 * useful_kernel_ns as f64 / kernel_ns.max(1) as f64,
                maintenance_kernel_ns as f64 / 1_000_000.0,
                100.0 * maintenance_kernel_ns as f64 / kernel_ns.max(1) as f64,
                fused_append_mic_kernel_ns as f64 / 1_000_000.0,
                100.0 * fused_append_mic_kernel_ns as f64 / kernel_ns.max(1) as f64,
            );
        }
        eprintln!(
            "inductor-cdcl: active pair-scheduler {}, transport unavailable {}, hardware disabled {}/{}, passes {} (skipped {}, offered {}, max-ready {}), candidates {}, skipped-small-batch {}, offered {}, batches {}, unavailable calls/queries {}/{}, context loads {}, appends {}/{} clauses (fallbacks {}), combined ok/fallback {}/{}, hw SAT {}, hw UNSAT {}, unknown {}, errors {}, effective conclusive used/generated {}/{}, validation rejected {}, unconsumed {}, hw work decisions/conflicts/propagations/learnts {}/{}/{}/{}, SAT used {}, rejected SAT {}, model lift succeeded/attempted {}/{}, predecessor lits full/result {}/{}, lift {:.3} ms, UNSAT cores used {}, rejected {}, UNSAT lits assumptions/hw-core/cpu-core {}/{}/{}, CPU fallbacks executed {}, block cost-gate rejected {}, block CPU samples {} mean {:.3} us, calibrations above-threshold/total {}/{}, route enable/disable {}/{}, latest representative {:.3} us route {}, calibration {:.3} ms, async harvested/launched {}/{}, prepare/wall/join {:.3}/{:.3}/{:.3} ms, init/wait {:.3}/{:.3} ms, load/append {:.3}/{:.3} ms, combined attempts {:.3} ms, batches {:.3} ms, SAT-state {:.3} ms, UNSAT-state {:.3} ms",
            if pair_scheduler_enabled() {
                "on"
            } else {
                "off"
            },
            ACTIVE_TRANSPORT_UNAVAILABLE.load(Ordering::Relaxed),
            ACTIVE_HARDWARE_DISABLED.load(Ordering::Relaxed),
            ACTIVE_HARDWARE_DISABLES.load(Ordering::Relaxed),
            ACTIVE_PASSES.load(Ordering::Relaxed),
            ACTIVE_SKIPPED_PASSES.load(Ordering::Relaxed),
            ACTIVE_OFFERED_PASSES.load(Ordering::Relaxed),
            ACTIVE_MAX_READY.load(Ordering::Relaxed),
            ACTIVE_CANDIDATES.load(Ordering::Relaxed),
            ACTIVE_SKIPPED_SMALL_BATCH.load(Ordering::Relaxed),
            ACTIVE_OFFERED.load(Ordering::Relaxed),
            ACTIVE_BATCHES.load(Ordering::Relaxed),
            ACTIVE_UNAVAILABLE_CALLS.load(Ordering::Relaxed),
            ACTIVE_UNAVAILABLE_QUERIES.load(Ordering::Relaxed),
            ACTIVE_CONTEXT_LOADS.load(Ordering::Relaxed),
            ACTIVE_CONTEXT_APPENDS.load(Ordering::Relaxed),
            ACTIVE_CONTEXT_APPEND_CLAUSES.load(Ordering::Relaxed),
            ACTIVE_CONTEXT_APPEND_FALLBACKS.load(Ordering::Relaxed),
            ACTIVE_COMBINED_BATCHES.load(Ordering::Relaxed),
            ACTIVE_COMBINED_FALLBACKS.load(Ordering::Relaxed),
            ACTIVE_HW_SAT.load(Ordering::Relaxed),
            ACTIVE_HW_UNSAT.load(Ordering::Relaxed),
            ACTIVE_UNKNOWN.load(Ordering::Relaxed),
            ACTIVE_ERROR.load(Ordering::Relaxed),
            hw_used,
            hw_conclusive,
            hw_rejected,
            hw_unconsumed,
            ACTIVE_HW_DECISIONS.load(Ordering::Relaxed),
            ACTIVE_HW_CONFLICTS.load(Ordering::Relaxed),
            ACTIVE_HW_PROPAGATIONS.load(Ordering::Relaxed),
            ACTIVE_HW_LEARNTS.load(Ordering::Relaxed),
            ACTIVE_SAT_USED.load(Ordering::Relaxed),
            ACTIVE_SAT_REJECTED.load(Ordering::Relaxed),
            ACTIVE_SAT_LIFT_SUCCEEDED.load(Ordering::Relaxed),
            ACTIVE_SAT_LIFT_ATTEMPTED.load(Ordering::Relaxed),
            ACTIVE_SAT_FULL_LITS.load(Ordering::Relaxed),
            ACTIVE_SAT_LIFTED_LITS.load(Ordering::Relaxed),
            ACTIVE_SAT_LIFT_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_UNSAT_CORE_USED.load(Ordering::Relaxed),
            ACTIVE_UNSAT_CORE_REJECTED.load(Ordering::Relaxed),
            ACTIVE_UNSAT_ASSUMPTION_LITS.load(Ordering::Relaxed),
            ACTIVE_UNSAT_HW_CORE_LITS.load(Ordering::Relaxed),
            ACTIVE_UNSAT_CPU_CORE_LITS.load(Ordering::Relaxed),
            ACTIVE_CPU_FALLBACK.load(Ordering::Relaxed),
            ACTIVE_BLOCK_COST_REJECTED.load(Ordering::Relaxed),
            ACTIVE_BLOCK_CPU_SAMPLES.load(Ordering::Relaxed),
            ACTIVE_BLOCK_CPU_NS.load(Ordering::Relaxed) as f64
                / ACTIVE_BLOCK_CPU_SAMPLES.load(Ordering::Relaxed).max(1) as f64
                / 1_000.0,
            ACTIVE_BLOCK_CALIBRATION_PROFITABLE.load(Ordering::Relaxed),
            ACTIVE_BLOCK_CALIBRATIONS.load(Ordering::Relaxed),
            ACTIVE_BLOCK_ROUTE_ENABLES.load(Ordering::Relaxed),
            ACTIVE_BLOCK_ROUTE_DISABLES.load(Ordering::Relaxed),
            ACTIVE_BLOCK_ROUTE_REPRESENTATIVE_NS.load(Ordering::Relaxed) as f64 / 1_000.0,
            if ACTIVE_BLOCK_ROUTE_ENABLED.load(Ordering::Relaxed) {
                "FPGA"
            } else {
                "CPU"
            },
            ACTIVE_BLOCK_CALIBRATION_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_BLOCK_ASYNC_HARVESTED.load(Ordering::Relaxed),
            ACTIVE_BLOCK_ASYNC_LAUNCHED.load(Ordering::Relaxed),
            ACTIVE_BLOCK_ASYNC_PREPARE_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_BLOCK_ASYNC_WALL_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_BLOCK_ASYNC_JOIN_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_INIT_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_STATE_WAIT_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_CONTEXT_LOAD_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_CONTEXT_APPEND_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_COMBINED_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_BATCH_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_VALIDATE_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_UNSAT_VALIDATE_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
        );
        eprintln!(
            "inductor-cdcl: active source split block selected conclusive/used/rejected/unconsumed {}/{}/{}/{}, no-answer {}, cache reused/mean-age {}/{:.2}, used/rejected {}/{}, replaced/evicted {}/{}, propagation conclusive/used/rejected {}/{}/{}",
            block_conclusive,
            block_used,
            block_rejected,
            block_unconsumed,
            ACTIVE_BLOCK_SELECTED_NO_ANSWER.load(Ordering::Relaxed),
            ACTIVE_BLOCK_CACHE_REUSED.load(Ordering::Relaxed),
            ACTIVE_BLOCK_CACHE_REUSE_AGE.load(Ordering::Relaxed) as f64
                / ACTIVE_BLOCK_CACHE_REUSED.load(Ordering::Relaxed).max(1) as f64,
            ACTIVE_BLOCK_CACHE_REUSE_USED.load(Ordering::Relaxed),
            ACTIVE_BLOCK_CACHE_REUSE_REJECTED.load(Ordering::Relaxed),
            ACTIVE_BLOCK_CACHE_REPLACED.load(Ordering::Relaxed),
            ACTIVE_BLOCK_CACHE_EVICTED.load(Ordering::Relaxed),
            hw_conclusive.saturating_sub(block_conclusive),
            hw_used.saturating_sub(block_used),
            hw_rejected.saturating_sub(block_rejected),
        );
        if shared_domain_projection_enabled() {
            let projected_queries = ACTIVE_SHARED_DOMAIN_PROJECTED_QUERIES.load(Ordering::Relaxed);
            let projected_batches = ACTIVE_SHARED_DOMAIN_PROJECTED_BATCHES.load(Ordering::Relaxed);
            eprintln!(
                "inductor-cdcl: shared-domain ABI projection queries/batches/fill {}/{}/{:.3}, repeated request words removed {} (planning only; production wire unchanged)",
                projected_queries,
                projected_batches,
                projected_queries as f64 / projected_batches.max(1) as f64,
                ACTIVE_SHARED_DOMAIN_PROJECTED_SAVED_WORDS.load(Ordering::Relaxed),
            );
        }
        if active_skip_cpu_check() {
            eprintln!(
                "inductor-cdcl: active trusted direct results SAT accepted/rejected {}/{}, revision-fresh SAT reused {}, stale SAT discarded {}, materialized SAT prepared/rejected/used {}/{}/{}, prepare {:.3} ms, UNSAT core accepted/rejected {}/{} (transport/state restoration only; no CPU semantic replay)",
                ACTIVE_TRUSTED_SAT_INSTALLED.load(Ordering::Relaxed),
                ACTIVE_TRUSTED_SAT_REJECTED.load(Ordering::Relaxed),
                ACTIVE_TRUSTED_SAT_REVISION_REUSED.load(Ordering::Relaxed),
                ACTIVE_TRUSTED_SAT_STALE_REVALIDATED.load(Ordering::Relaxed),
                ACTIVE_MATERIALIZED_SAT_PREPARED.load(Ordering::Relaxed),
                ACTIVE_MATERIALIZED_SAT_REJECTED.load(Ordering::Relaxed),
                ACTIVE_MATERIALIZED_SAT_USED.load(Ordering::Relaxed),
                ACTIVE_MATERIALIZED_SAT_PREPARE_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
                ACTIVE_TRUSTED_UNSAT_INSTALLED.load(Ordering::Relaxed),
                ACTIVE_TRUSTED_UNSAT_REJECTED.load(Ordering::Relaxed),
            );
        }
        eprintln!(
            "inductor-cdcl: active push prefetch launched/harvested/busy {}/{}/{}, queries/ready/hits {}/{}/{}, used/rejected/evicted {}/{}/{}, prepare/wall/join {:.3}/{:.3}/{:.3} ms",
            ACTIVE_PUSH_PREFETCH_LAUNCHED.load(Ordering::Relaxed),
            ACTIVE_PUSH_PREFETCH_HARVESTED.load(Ordering::Relaxed),
            ACTIVE_PUSH_PREFETCH_BUSY.load(Ordering::Relaxed),
            ACTIVE_PUSH_PREFETCH_QUERIES.load(Ordering::Relaxed),
            ACTIVE_PUSH_PREFETCH_READY.load(Ordering::Relaxed),
            ACTIVE_PUSH_PREFETCH_HITS.load(Ordering::Relaxed),
            ACTIVE_PUSH_PREFETCH_USED.load(Ordering::Relaxed),
            ACTIVE_PUSH_PREFETCH_REJECTED.load(Ordering::Relaxed),
            ACTIVE_PUSH_PREFETCH_EVICTED.load(Ordering::Relaxed),
            ACTIVE_PUSH_PREFETCH_PREPARE_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_PUSH_PREFETCH_WALL_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_PUSH_PREFETCH_JOIN_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
        );
        let push_bucket_values = |buckets: &[AtomicU64; 5]| {
            buckets
                .iter()
                .map(|value| value.load(Ordering::Relaxed))
                .collect::<Vec<_>>()
        };
        eprintln!(
            "inductor-cdcl: active push prefetch len buckets <=4/5-8/9-16/17-32/>32 submitted {:?}, ready {:?}, hits {:?}, used {:?}",
            push_bucket_values(&ACTIVE_PUSH_PREFETCH_SUBMITTED_BY_LEN),
            push_bucket_values(&ACTIVE_PUSH_PREFETCH_READY_BY_LEN),
            push_bucket_values(&ACTIVE_PUSH_PREFETCH_HITS_BY_LEN),
            push_bucket_values(&ACTIVE_PUSH_PREFETCH_USED_BY_LEN),
        );
        eprintln!(
            "inductor-cdcl: active push prefetch adaptive admitted/suppressed/reprobes {}/{}/{}, latest used/query {}/{}, skipped long/context {}/{}",
            ACTIVE_PUSH_PREFETCH_ADMITTED.load(Ordering::Relaxed),
            ACTIVE_PUSH_PREFETCH_SUPPRESSED.load(Ordering::Relaxed),
            ACTIVE_PUSH_PREFETCH_REPROBES.load(Ordering::Relaxed),
            ACTIVE_PUSH_PREFETCH_EVAL_USED.load(Ordering::Relaxed),
            ACTIVE_PUSH_PREFETCH_EVAL_QUERIES.load(Ordering::Relaxed),
            ACTIVE_PUSH_PREFETCH_SKIPPED_LONG.load(Ordering::Relaxed),
            ACTIVE_PUSH_PREFETCH_SKIPPED_CONTEXT.load(Ordering::Relaxed),
        );
        eprintln!(
            "inductor-cdcl: active MIC waves/queries {}/{}, SAT/UNSAT/UNKNOWN {}/{}/{}, used SAT/UNSAT {}/{}, rejected {}, invalidated {}, proof-neutral reached/replaceable/invalidated {}/{}/{}",
            ACTIVE_MIC_BATCH_WAVES.load(Ordering::Relaxed),
            ACTIVE_MIC_BATCH_QUERIES.load(Ordering::Relaxed),
            ACTIVE_MIC_BATCH_SAT.load(Ordering::Relaxed),
            ACTIVE_MIC_BATCH_UNSAT.load(Ordering::Relaxed),
            ACTIVE_MIC_BATCH_UNKNOWN.load(Ordering::Relaxed),
            ACTIVE_MIC_BATCH_SAT_USED.load(Ordering::Relaxed),
            ACTIVE_MIC_BATCH_UNSAT_USED.load(Ordering::Relaxed),
            ACTIVE_MIC_BATCH_REJECTED.load(Ordering::Relaxed),
            ACTIVE_MIC_BATCH_INVALIDATED.load(Ordering::Relaxed),
            ACTIVE_MIC_BATCH_SHADOW_REACHED.load(Ordering::Relaxed),
            ACTIVE_MIC_BATCH_SHADOW_REPLACEABLE.load(Ordering::Relaxed),
            ACTIVE_MIC_BATCH_SHADOW_INVALIDATED.load(Ordering::Relaxed),
        );
        eprintln!(
            "inductor-cdcl: active MIC economics probes/offloads/rejects {}/{}/{}, projected replaceable CPU/HW {:.3}/{}",
            ACTIVE_MIC_BATCH_ECON_PROBES.load(Ordering::Relaxed),
            ACTIVE_MIC_BATCH_ECON_OFFLOADS.load(Ordering::Relaxed),
            ACTIVE_MIC_BATCH_ECON_REJECTS.load(Ordering::Relaxed),
            ACTIVE_MIC_BATCH_ECON_CPU_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            if ACTIVE_MIC_BATCH_ECON_HW_VALID.load(Ordering::Relaxed) {
                format!(
                    "{:.3} ms",
                    ACTIVE_MIC_BATCH_ECON_HW_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0
                )
            } else {
                "untrained".to_string()
            },
        );
        eprintln!(
            "inductor-cdcl: active MIC chain commands complete/partial/errors {}/{}/{}/{}, trials/physical rounds {}/{}, input/output lits {}/{}, standalone useful kernel/service {:.3}/{:.3} ms, fused append+MIC commands/clauses kernel/service {}/{} {:.3}/{:.3} ms, client service {:.3} ms, frame materializations {} kernel/service {:.3}/{:.3} ms, context reused {}, work decisions/conflicts/propagations/learnts {}/{}/{}/{}, adoption accepted/rejected {}/{}, CPU verify {:.3} ms, CPU MIC loops/trials replaced {}/{}",
            ACTIVE_MIC_CHAIN_COMMANDS.load(Ordering::Relaxed),
            ACTIVE_MIC_CHAIN_COMPLETE.load(Ordering::Relaxed),
            ACTIVE_MIC_CHAIN_PARTIAL.load(Ordering::Relaxed),
            ACTIVE_MIC_CHAIN_ERRORS.load(Ordering::Relaxed),
            ACTIVE_MIC_CHAIN_TRIALS.load(Ordering::Relaxed),
            ACTIVE_MIC_CHAIN_PHYSICAL_ROUNDS.load(Ordering::Relaxed),
            ACTIVE_MIC_CHAIN_INPUT_LITS.load(Ordering::Relaxed),
            ACTIVE_MIC_CHAIN_OUTPUT_LITS.load(Ordering::Relaxed),
            ACTIVE_MIC_CHAIN_KERNEL_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_MIC_CHAIN_SERVICE_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_FUSED_APPEND_MIC_COMMANDS.load(Ordering::Relaxed),
            ACTIVE_FUSED_APPEND_MIC_CLAUSES.load(Ordering::Relaxed),
            ACTIVE_FUSED_APPEND_MIC_KERNEL_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_FUSED_APPEND_MIC_SERVICE_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_MIC_CHAIN_CLIENT_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_FRAME_MATERIALIZATIONS.load(Ordering::Relaxed),
            ACTIVE_FRAME_MATERIALIZE_KERNEL_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_FRAME_MATERIALIZE_SERVICE_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_MIC_CHAIN_CONTEXT_REUSED.load(Ordering::Relaxed),
            ACTIVE_MIC_CHAIN_DECISIONS.load(Ordering::Relaxed),
            ACTIVE_MIC_CHAIN_CONFLICTS.load(Ordering::Relaxed),
            ACTIVE_MIC_CHAIN_PROPAGATIONS.load(Ordering::Relaxed),
            ACTIVE_MIC_CHAIN_LEARNTS.load(Ordering::Relaxed),
            ACTIVE_MIC_CHAIN_VALIDATED.load(Ordering::Relaxed),
            ACTIVE_MIC_CHAIN_REJECTED.load(Ordering::Relaxed),
            ACTIVE_MIC_CHAIN_VERIFY_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            ACTIVE_MIC_CHAIN_CPU_LOOPS_REPLACED.load(Ordering::Relaxed),
            ACTIVE_MIC_CHAIN_CPU_TRIALS_REPLACED.load(Ordering::Relaxed),
        );
        eprintln!(
            "inductor-cdcl: active MIC chain economics probes/offloads/rejects {}/{}/{}, warm {}/{}/{}, projected replaceable CPU/HW {:.3}/{}",
            ACTIVE_MIC_CHAIN_ECON_PROBES.load(Ordering::Relaxed),
            ACTIVE_MIC_CHAIN_ECON_OFFLOADS.load(Ordering::Relaxed),
            ACTIVE_MIC_CHAIN_ECON_REJECTS.load(Ordering::Relaxed),
            ACTIVE_MIC_CHAIN_ECON_WARM_PROBES.load(Ordering::Relaxed),
            ACTIVE_MIC_CHAIN_ECON_WARM_OFFLOADS.load(Ordering::Relaxed),
            ACTIVE_MIC_CHAIN_ECON_WARM_REJECTS.load(Ordering::Relaxed),
            ACTIVE_MIC_CHAIN_ECON_CPU_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            if ACTIVE_MIC_CHAIN_ECON_HW_VALID.load(Ordering::Relaxed) {
                format!(
                    "{:.3} ms",
                    ACTIVE_MIC_CHAIN_ECON_HW_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0
                )
            } else {
                "untrained".to_string()
            },
        );
        eprintln!(
            "inductor-cdcl: active block preflight conclusive/selected/fallback {}/{}/{}, wave reserved/taken {}/{}",
            ACTIVE_BLOCK_PREFLIGHT_CONCLUSIVE.load(Ordering::Relaxed),
            ACTIVE_BLOCK_PREFLIGHT_SELECTED.load(Ordering::Relaxed),
            ACTIVE_BLOCK_PREFLIGHT_FALLBACK.load(Ordering::Relaxed),
            ACTIVE_BLOCK_WAVE_RESERVED.load(Ordering::Relaxed),
            ACTIVE_BLOCK_WAVE_TAKEN.load(Ordering::Relaxed),
        );
        let batch_hw_valid = ACTIVE_BLOCK_BATCH_ECON_HW_VALID.load(Ordering::Relaxed);
        eprintln!(
            "inductor-cdcl: active block batch economics probes/offloads/rejects {}/{}/{}, projected CPU/HW {:.3}/{}, route {}",
            ACTIVE_BLOCK_BATCH_ECON_PROBES.load(Ordering::Relaxed),
            ACTIVE_BLOCK_BATCH_ECON_OFFLOADS.load(Ordering::Relaxed),
            ACTIVE_BLOCK_BATCH_ECON_REJECTS.load(Ordering::Relaxed),
            ACTIVE_BLOCK_BATCH_ECON_CPU_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            if batch_hw_valid {
                format!(
                    "{:.3} ms",
                    ACTIVE_BLOCK_BATCH_ECON_HW_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0
                )
            } else {
                "untrained".to_string()
            },
            if ACTIVE_BLOCK_BATCH_ECON_ROUTE.load(Ordering::Relaxed) {
                "FPGA"
            } else {
                "CPU"
            },
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
                ACTIVE_PREFLIGHT_RESTORE_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
                ACTIVE_PREFLIGHT_CONFLICTS.load(Ordering::Relaxed) as f64
                    / candidates.max(1) as f64,
            );
        }
        let sampled = ACTIVE_SAMPLE_QUERIES.load(Ordering::Relaxed);
        let undersized = ACTIVE_SAMPLE_UNDERSIZED_REJECTED.load(Ordering::Relaxed);
        if sampled != 0 || undersized != 0 {
            eprintln!(
                "inductor-cdcl: active CPU sample compatible groups {}, planning {:.3} ms, queries {}, mean total/clone/solve {:.3}/{:.3}/{:.3} us, solve threshold {:.3} us, FPGA/CPU batches {}/{}, FPGA retained {}, CPU rejected {}, undersized rejected {}",
                ACTIVE_SAMPLE_CONTEXT_GROUPS.load(Ordering::Relaxed),
                ACTIVE_SAMPLE_PLAN_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
                sampled,
                ACTIVE_SAMPLE_NS.load(Ordering::Relaxed) as f64 / sampled.max(1) as f64 / 1_000.0,
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
                        .saturating_add(ACTIVE_CONTEXT_APPEND_NS.load(Ordering::Relaxed))
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
    use crate::accel::cdcl::MIC_HEADER_WORDS;
    use logicrs::DagCnf;
    use logicrs::satif::Satif;
    use logicrs::{Lit, Var};

    #[test]
    fn full_root_projection_lease_requires_exact_projection_vectors() {
        let lease = FullRootProjectionLease {
            handle: 9,
            next_var_by_current: vec![1, 0, 2],
            init_value_by_current: vec![0, 1, 2],
            decision_domain: vec![0x5, 0xa],
            latch_variables: vec![0, 1],
            input_variables: vec![2],
        };
        assert!(lease.matches(&[1, 0, 2], &[0, 1, 2], &[0x5, 0xa], &[0, 1], &[2]));
        assert!(!lease.matches(&[1, 2, 0], &[0, 1, 2], &[0x5, 0xa], &[0, 1], &[2]));
        assert!(!lease.matches(&[1, 0, 2], &[0, 2, 1], &[0x5, 0xa], &[0, 1], &[2]));
        assert!(!lease.matches(&[1, 0, 2], &[0, 1, 2], &[0x5], &[0, 1], &[2]));
        assert!(!lease.matches(&[1, 0, 2], &[0, 1, 2], &[0x5, 0xa], &[1, 0], &[2]));
        assert!(!lease.matches(&[1, 0, 2], &[0, 1, 2], &[0x5, 0xa], &[0, 1], &[1]));
    }

    #[test]
    fn only_epoch_reset_semantics_invalidate_full_root_projection() {
        assert!(!semantic_batch_invalidates_projection(&[
            BlockSemanticCommand::new(crate::accel::cdcl::BLOCK_SEMANTIC_STATS),
            BlockSemanticCommand::new(crate::accel::cdcl::BLOCK_SEMANTIC_EVENT_INSERT_LEMMA,),
        ]));
        assert!(semantic_batch_invalidates_projection(&[
            BlockSemanticCommand::new(crate::accel::cdcl::BLOCK_SEMANTIC_STATS),
            BlockSemanticCommand::new(BLOCK_SEMANTIC_RESET),
        ]));
        assert!(semantic_batch_invalidates_projection(&[
            BlockSemanticCommand::new(BLOCK_SEMANTIC_EVENT_RESET_EPOCH),
        ]));
    }

    #[test]
    fn full_root_wire_admission_caps_journal_and_rejects_repeated_large_metadata() {
        assert_eq!(
            admitted_full_root_steps_with_limits(64, 1_000, 1_000, 200, 100, 32_768, 65_536),
            Some(64),
        );
        assert_eq!(
            admitted_full_root_steps_with_limits(64, 10_000, 4_000, 4_799, 3_110, 32_768, 65_536,),
            Some(8),
        );
        assert_eq!(
            admitted_full_root_steps_with_limits(64, 20_000, 4_000, 4_799, 3_110, 32_768, 65_536,),
            None,
        );
    }

    #[test]
    fn portfolio_trace_paths_are_worker_scoped_without_losing_extension() {
        assert_eq!(
            scope_trace_path(
                "/tmp/quad.tsv".into(),
                Some(std::ffi::OsStr::new("ic3/pred")),
            ),
            std::path::PathBuf::from("/tmp/quad.ic3_pred.tsv"),
        );
        assert_eq!(
            scope_trace_path("/tmp/quad".into(), None),
            std::path::PathBuf::from("/tmp/quad"),
        );
    }

    #[test]
    fn persistent_ring_tickets_keep_one_independent_wave_correlated() {
        let pass = (7u64 << 32) | 11u64;
        let batch = persistent_ring_batch_id(pass);
        assert_ne!(batch, 0);
        let tickets: Vec<_> = (0..19)
            .map(|position| persistent_ring_query_ticket(pass, position, position == 18))
            .collect();
        assert!(tickets.iter().all(|ticket| ticket.batch_id == batch));
        assert!(
            tickets
                .iter()
                .all(|ticket| ticket.flags & RING_INDEPENDENT_SET != 0)
        );
        assert!(
            tickets[..18]
                .iter()
                .all(|ticket| ticket.flags & RING_END_OF_BATCH == 0)
        );
        assert_ne!(tickets[18].flags & RING_END_OF_BATCH, 0);
        for (position, ticket) in tickets.iter().enumerate() {
            assert_eq!(ticket.position, position as u32);
            assert_eq!(ticket.user_tag >> 32, u64::from(batch));
            assert_eq!(ticket.user_tag as u32, position as u32);
        }
    }

    #[test]
    fn persistent_ring_batch_id_never_uses_protocol_sentinel() {
        assert_eq!(persistent_ring_batch_id(0), 1);
        assert_eq!(
            persistent_ring_batch_id(u64::from(u32::MAX) << 32 | u64::from(u32::MAX)),
            1
        );
    }

    #[test]
    fn deterministic_device_failures_trip_the_solver_run_circuit_breaker() {
        assert!(deterministic_active_failure(&HardwareError::Capacity));
        for status in -104..=-101 {
            assert!(deterministic_active_failure(&HardwareError::Command(
                status
            )));
        }
        assert!(deterministic_active_failure(&HardwareError::Decode(
            BatchDecodeError::Backend(1),
        )));
        assert!(!deterministic_active_failure(&HardwareError::Command(-105)));
        assert!(!deterministic_active_failure(&HardwareError::Command(-100)));
        assert!(!deterministic_active_failure(&HardwareError::Command(-32)));
        assert!(!deterministic_active_failure(&HardwareError::Unavailable));
    }

    #[test]
    fn only_capacity_failures_are_accounted_as_hardware_unknowns() {
        assert!(active_failure_is_capacity(&HardwareError::Capacity));
        assert!(active_failure_is_capacity(&HardwareError::Command(-103)));
        assert!(!active_failure_is_capacity(&HardwareError::Command(-104)));
        assert!(!active_failure_is_capacity(&HardwareError::Command(-102)));
        assert!(!active_failure_is_capacity(&HardwareError::Command(-101)));
        assert!(!active_failure_is_capacity(&HardwareError::Unavailable));
    }

    #[test]
    fn mic_chain_batch_wire_round_trip_keeps_records_independent() {
        let l = |var: u32| Lit::new(Var::from(var), true);
        let cube0 = [(l(0), l(3)), (l(1), l(4)), (l(2), l(5))];
        let cube1 = [(l(6), l(9)), (l(7), l(10)), (l(8), l(11))];
        let chains = [
            MicChainQuery {
                frame: 4,
                cube: &cube0,
                constraints: &[],
                protected_index: 2,
                decision_budget: 32,
                conflict_budget: 16,
                max_trials: 2,
            },
            MicChainQuery {
                frame: 5,
                cube: &cube1,
                constraints: &[],
                protected_index: 2,
                decision_budget: 32,
                conflict_budget: 16,
                max_trials: 2,
            },
        ];
        let (request, capacity) = pack_mic_chains_request(12, &chains).unwrap();
        assert_eq!(request[0], ABI_VERSION);
        assert_eq!(request[1], 2);
        assert_eq!(request[2] as usize, request.len() - MIC_BATCH_HEADER_WORDS);
        assert_eq!(capacity, MIC_BATCH_RESPONSE_HEADER_WORDS + 2 * 15);
        let first = MIC_BATCH_HEADER_WORDS;
        let first_words = MIC_HEADER_WORDS
            + request[first + 4] as usize
            + request[first + 5] as usize
            + 2 * request[first + 3] as usize;
        assert_eq!(request[first + 1], 4);
        assert_eq!(request[first + first_words + 1], 5);

        let mut response = vec![ABI_VERSION, 2, 26, 0];
        for protected in [l(2), l(8)] {
            response.extend([
                ABI_VERSION,
                3,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                2,
                u32::from(protected),
            ]);
        }
        let (result0, words0) = decode_mic_chain_record(&cube0, &response[4..]).unwrap();
        let (result1, words1) = decode_mic_chain_record(&cube1, &response[4 + words0..]).unwrap();
        assert_eq!(words0, 13);
        assert_eq!(words1, 13);
        assert_eq!(result0.cube.len(), 1);
        assert_eq!(result1.cube.len(), 1);
        assert_eq!(result0.cube[0], l(2));
        assert_eq!(result1.cube[0], l(8));
        assert_eq!(result0.physical_rounds, 2);
        assert_eq!(result1.physical_rounds, 2);
    }

    #[test]
    fn mic_chain_wire_protects_a_middle_literal_without_reordering() {
        let l = |var: u32| Lit::new(Var::from(var), true);
        let cube = [(l(0), l(3)), (l(1), l(4)), (l(2), l(5))];
        let request = pack_mic_chain_request(6, 4, &cube, &[], 1, 32, 16, 2).unwrap();
        assert_eq!(request[6], 32);
        assert_eq!(request[2] & MIC_PROTECT_INDEX, MIC_PROTECT_INDEX);
        assert_eq!(request[2] >> MIC_PROTECTED_INDEX_SHIFT, 1);
        assert_eq!(request[5], 6);
        let pairs_at = MIC_HEADER_WORDS + 6;
        assert_eq!(
            &request[pairs_at..],
            &[
                u32::from(l(0)),
                u32::from(l(3)),
                u32::from(l(1)),
                u32::from(l(4)),
                u32::from(l(2)),
                u32::from(l(5)),
            ]
        );
    }

    #[test]
    fn mic_domain_uses_the_same_four_bank_schedule_as_short_queries() {
        let encoded = encode_mic_domain(&[5, 2, 8, 3, 4], true).unwrap();
        assert_eq!(encoded.len(), 8);
        assert_eq!(encoded[0], 0x8002_0008);
        assert_eq!(encoded[1], 0x8000_0005);
        assert_eq!(encoded[2], 0x8001_0002);
        assert_eq!(encoded[3], 0x8003_0003);
        assert_eq!(encoded[4], 0x8004_0004);
        assert_eq!(&encoded[5..], &[0, 0, 0]);
    }

    #[test]
    fn protected_mic_entry_counts_as_complete_only_after_all_candidates() {
        assert!(mic_chain_effectively_complete(
            false,
            UnknownReason::None,
            7,
            7,
            7,
        ));
        assert!(!mic_chain_effectively_complete(
            false,
            UnknownReason::None,
            4,
            4,
            7,
        ));
        assert!(!mic_chain_effectively_complete(
            false,
            UnknownReason::ConflictBudget,
            7,
            7,
            7,
        ));
        assert!(mic_chain_effectively_complete(
            true,
            UnknownReason::None,
            2,
            4,
            7,
        ));
    }

    #[test]
    fn cpu_sample_requires_cost_and_a_remaining_hardware_batch() {
        assert_eq!(representative_sample_positions(10, 1), vec![5]);
        assert_eq!(representative_sample_positions(10, 3), vec![1, 5, 8]);
        assert_eq!(representative_sample_positions(3, 8), vec![0, 1, 2]);
        assert!(sample_keeps_fpga(
            &[250_000, 300_000, 900_000],
            16,
            8,
            200_000,
            None,
            true
        ));
        assert!(!sample_keeps_fpga(
            &[50_000, 100_000, 900_000],
            16,
            8,
            200_000,
            None,
            true
        ));
        // With an even sample the lower median prevents one expensive half
        // from routing a frame whose other half is cheap.
        assert!(!sample_keeps_fpga(
            &[150_000, 600_000],
            16,
            8,
            200_000,
            None,
            true
        ));
        assert!(!sample_keeps_fpga(
            &[250_000, 300_000],
            7,
            8,
            200_000,
            None,
            true
        ));
        assert!(!sample_keeps_fpga(&[], 16, 8, 200_000, None, true));
        assert!(!sample_keeps_fpga(
            &[250_000, 300_000],
            16,
            8,
            200_000,
            None,
            false
        ));
        // Many individually cheap inquiries can still amortize one FPGA
        // submission; a genuinely cheap aggregate remains on the CPU.
        assert!(sample_keeps_fpga(
            &[80_000, 90_000, 100_000],
            64,
            8,
            200_000,
            Some(4_000_000),
            true,
        ));
        assert!(!sample_keeps_fpga(
            &[20_000, 25_000, 30_000],
            64,
            8,
            200_000,
            Some(4_000_000),
            true,
        ));
    }

    #[test]
    fn full_batch_planner_rebalances_and_drops_only_short_tails() {
        assert_eq!(
            plan_full_batch_ranges(&vec![1; 95], 32, 64, 32_768),
            vec![0..63, 63..95]
        );
        assert_eq!(
            plan_full_batch_ranges(&vec![1; 31], 32, 64, 32_768),
            Vec::<std::ops::Range<usize>>::new()
        );
        assert_eq!(
            plan_full_batch_ranges(&vec![1; 32], 32, 64, 32_768),
            vec![0..32]
        );
        assert_eq!(
            plan_full_batch_ranges(&[200, 10, 10], 2, 4, 100),
            vec![1..3]
        );
        let capped = plan_full_batch_ranges(&vec![10; 12], 4, 8, 54);
        assert_eq!(capped, vec![0..4, 4..8, 8..12]);
        assert!(capped.iter().all(|range| range.len() >= 4));
    }

    #[test]
    fn shared_domain_projection_recovers_large_frontier_batches() {
        let domain: Vec<_> = (0..16_326).map(Var::from).collect();
        let domains = vec![domain.as_slice(); 8];
        // The measured p187 shape is approximately 16,326 repeated domain
        // words plus 2,474 private query/view words. Ordinary ABI-v2 can fit
        // only one record; a shared domain fits six in the same 32K command.
        let words = vec![18_800; 8];
        assert_eq!(
            plan_full_batch_ranges(&words, 1, 8, 32_768),
            (0..8).map(|index| index..index + 1).collect::<Vec<_>>()
        );
        assert_eq!(
            plan_shared_domain_batch_ranges(&domains, &words, 1, 8, 32_768),
            vec![0..6, 6..8]
        );

        let other = vec![Var::from(1), Var::from(2)];
        let split_domains = [
            domain.as_slice(),
            domain.as_slice(),
            other.as_slice(),
            other.as_slice(),
        ];
        assert_eq!(
            plan_shared_domain_batch_ranges(
                &split_domains,
                &[16_426, 16_426, 102, 102],
                2,
                8,
                32_768,
            ),
            vec![0..2, 2..4]
        );
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
            pack_clauses(
                &[1],
                1,
                &[ResidentClause::new(0, 0, LitVec::from([outside]))]
            ),
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

        assert_eq!(
            profile_batch[4 + 2] & WANT_STAGE_PROFILE,
            WANT_STAGE_PROFILE
        );
        assert_eq!(profile_capacity, response_capacity + STAGE_PROFILE_WORDS);
        assert_eq!(combined[0] as usize, context.len());
        assert_eq!(&combined[1..1 + context.len()], context.as_slice());
        assert_eq!(&combined[1 + context.len()..], batch.as_slice());
        assert_eq!(combined_capacity, response_capacity);
    }

    #[test]
    fn resident_arena_preserves_duplicate_occurrence_ids_across_shrink() {
        let a = Lit::new(Var::from(0), true);
        let b = Lit::new(Var::from(1), true);
        let duplicate = ResidentClause::new(0, u32::MAX, LitVec::from([a, b]));
        let tautology = ResidentClause::new(0, u32::MAX, LitVec::from([a, !a]));
        let context = |clauses| ShadowContext {
            n_var: 2,
            clauses,
            scope: ShadowContextScope::ExactFrame(3),
        };
        let mut arena = ResidentArena::default();
        let (full, appended) = arena
            .intern_context(&context(vec![
                duplicate.clone(),
                duplicate.clone(),
                tautology,
            ]))
            .unwrap();
        assert_eq!(full.active(3, false), vec![0, 1]);
        assert_eq!(appended.len(), 2);
        assert_eq!(arena.n_clause, 2);

        let (short, appended) = arena
            .intern_context(&context(vec![duplicate.clone()]))
            .unwrap();
        assert_eq!(short.active(3, false), vec![0]);
        assert!(appended.is_empty());

        let (restored, appended) = arena
            .intern_context(&context(vec![duplicate.clone(), duplicate]))
            .unwrap();
        assert_eq!(restored.active(3, false), vec![0, 1]);
        assert!(appended.is_empty());
    }

    #[test]
    fn resident_arena_isolates_different_frame_snapshots_in_one_batch() {
        let a = Lit::new(Var::from(0), true);
        let b = Lit::new(Var::from(1), true);
        let context = |literal, frame| ShadowContext {
            n_var: 2,
            clauses: vec![ResidentClause::new(0, u32::MAX, LitVec::from([literal]))],
            scope: ShadowContextScope::ExactFrame(frame),
        };
        let first_context = context(a, 3);
        let second_context = context(b, 4);
        assert!(batch_context_compatible(
            &first_context,
            &second_context,
            true
        ));
        assert!(!batch_context_compatible(
            &first_context,
            &second_context,
            false
        ));
        assert!(batch_context_compatible(
            &first_context,
            &ShadowContext {
                n_var: 3,
                clauses: second_context.clauses.clone(),
                scope: second_context.scope,
            },
            true
        ));
        assert!(!batch_context_compatible(
            &first_context,
            &ShadowContext {
                n_var: QUALIFIED_ARENA_MAX_VARS + 1,
                clauses: second_context.clauses.clone(),
                scope: second_context.scope,
            },
            true
        ));

        let mut arena = ResidentArena::default();
        let (first_mapping, first_append) = arena.intern_context(&first_context).unwrap();
        let (second_mapping, second_append) = arena.intern_context(&second_context).unwrap();
        assert_eq!(arena.n_clause, 2);
        assert_eq!(first_append.len(), 1);
        assert_eq!(second_append.len(), 1);
        assert_eq!(first_mapping.active(3, false), vec![0]);
        assert_eq!(second_mapping.active(4, false), vec![1]);

        let first_view = arena.plan_view(0, &[0]).unwrap();
        let second_view = arena.plan_view(1, &[1]).unwrap();
        assert_eq!(first_view.words[0], ARENA_VIEW_TOGGLE);
        assert_eq!(second_view.words[0], ARENA_VIEW_TOGGLE);
        assert_eq!(&first_view.words[ARENA_VIEW_PREFIX_WORDS..], &[0]);
        assert_eq!(&second_view.words[ARENA_VIEW_PREFIX_WORDS..], &[1]);
        assert_eq!(arena.lanes[0].bitmap, vec![0b01]);
        assert_eq!(arena.lanes[1].bitmap, vec![0b10]);
    }

    #[test]
    fn arena_views_choose_toggle_bitmap_and_reuse_across_inactive_growth() {
        let mut arena = ResidentArena {
            n_var: 1,
            n_clause: 33,
            ..ResidentArena::default()
        };
        let first = arena.plan_view(0, &[0]).unwrap();
        assert_eq!(first.words[0], ARENA_VIEW_TOGGLE);
        assert_eq!(&first.words[ARENA_VIEW_PREFIX_WORDS..], &[0]);

        let reuse = arena.plan_view(0, &[0]).unwrap();
        assert_eq!(reuse.words[0], ARENA_VIEW_REUSE);
        assert_eq!(reuse.words[4], 0);
        assert_eq!(reuse.words[1], first.words[1]);

        let dense: Vec<_> = (0..12).collect();
        let bitmap = arena.plan_view(0, &dense).unwrap();
        assert_eq!(bitmap.words[0], ARENA_VIEW_BITMAP);
        assert_eq!(bitmap.words[4], 2);

        // The device append path extends an external view with zero bits for
        // new clauses and updates its physical count. An inactive append can
        // therefore keep both bitmap and key through REUSE.
        let before_growth_key = (u64::from(bitmap.words[2]) << 32) | u64::from(bitmap.words[1]);
        arena.n_clause = 34;
        let growth = arena.plan_view(0, &dense).unwrap();
        let growth_key = (u64::from(growth.words[2]) << 32) | u64::from(growth.words[1]);
        assert_eq!(growth.words[0], ARENA_VIEW_REUSE);
        assert_eq!(growth.words[3], 34);
        assert_eq!(growth.words[4], 0);
        assert_eq!(growth_key, before_growth_key);
    }

    #[test]
    fn arena_batch_packer_interleaves_lane_views_without_changing_queries() {
        let a = Lit::new(Var::from(0), true);
        let mut first = IncrementalQuery::new(2, LitVec::from([a]));
        first.domain = vec![Var::from(0)];
        let second = IncrementalQuery::new(5, LitVec::from([!a]));
        let queries = [first, second];
        let views = [
            ArenaViewUpdate {
                words: vec![ARENA_VIEW_TOGGLE, 11, 0, 3, 1, 2],
            },
            ArenaViewUpdate {
                words: vec![ARENA_VIEW_REUSE, 7, 0, 3, 0],
            },
        ];
        let (plain, plain_capacity) = pack_batch_request(&queries, false).unwrap();
        let (arena, arena_capacity) = pack_arena_batch_request(&queries, &views, false).unwrap();
        let first_words = query_request_words(&queries[0]).unwrap();
        let second_words = query_request_words(&queries[1]).unwrap();

        assert_eq!(arena[0], ABI_VERSION);
        assert_eq!(arena[1], 2);
        assert_eq!(arena[2] as usize, arena.len() - 4);
        assert_eq!(arena_capacity, plain_capacity);
        assert_eq!(&arena[4..4 + views[0].words.len()], &views[0].words);
        assert_eq!(
            &arena[4 + views[0].words.len()..4 + views[0].words.len() + first_words],
            &plain[4..4 + first_words]
        );
        let second_offset = 4 + views[0].words.len() + first_words;
        assert_eq!(
            &arena[second_offset..second_offset + views[1].words.len()],
            &views[1].words
        );
        assert_eq!(
            &arena[second_offset + views[1].words.len()..],
            &plain[4 + first_words..4 + first_words + second_words]
        );
    }

    #[test]
    fn arena_view_planning_is_transactional_until_candidate_commit() {
        let arena = ResidentArena {
            n_var: 1,
            n_clause: 1,
            ..ResidentArena::default()
        };
        let mut candidate = arena.clone();
        let update = candidate.plan_view(0, &[0]).unwrap();
        assert_eq!(update.words[0], ARENA_VIEW_TOGGLE);
        assert!(!arena.lanes[0].valid);
        assert!(candidate.lanes[0].valid);
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
            (PROFILE_LEARNT_LITERALS as u64 + 1) << 32 | 116,
        );
        assert_eq!(
            records[0].profile_counters[PROFILE_OCCURRENCE_PAIRS],
            (PROFILE_OCCURRENCE_PAIRS as u64 + 1) << 32 | 118,
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
        assert_eq!(
            query_request_words(&query),
            Some(header.as_words().len() + payload.len())
        );
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
        assert!(
            shared
                .clauses
                .iter()
                .all(|clause| clause.lo == 0 && clause.hi == u32::MAX)
        );
        assert!(
            shared
                .clauses
                .iter()
                .any(|clause| clause.literals.as_slice() == [a, b])
        );
        assert!(
            !shared
                .clauses
                .iter()
                .any(|clause| clause.literals.as_slice() == [!a, b])
        );
        assert_eq!(expanded.constraints.len(), 2);
        assert!(
            expanded
                .constraints
                .iter()
                .any(|clause| clause.as_slice() == [!a, b])
        );

        let mut cache = BatchedSolverContext::new(&solver);
        let (cached_uses_lemmas, cached_words) = cache.query_plan(&query, true).unwrap();
        let cached_query = cache.prepare_query(query.clone(), cached_uses_lemmas);
        let cached_context = cache.context(cached_uses_lemmas).clone();
        assert!(cached_uses_lemmas);
        assert_eq!(cached_words, query_request_words(&expanded).unwrap());
        assert_eq!(cached_query.constraints, expanded.constraints);
        assert_eq!(cached_context, shared);

        let (exact, unchanged, used) = prepare_batched_query(&solver, query.clone(), false);
        assert!(!used);
        assert_eq!(unchanged.constraints, query.constraints);
        assert_eq!(exact.scope, ShadowContextScope::ExactFrame(7));
        assert!(
            exact
                .clauses
                .iter()
                .all(|clause| clause.lo == 0 && clause.hi == u32::MAX)
        );
        assert!(
            exact
                .clauses
                .iter()
                .any(|clause| clause.lo == 0 && clause.hi == u32::MAX)
        );
        assert!(
            exact
                .clauses
                .iter()
                .any(|clause| clause.literals.as_slice() == [!a, b])
        );

        let mut later_solver = solver.clone();
        later_solver.accel_level = 8;
        later_solver.add_clause(&[a, !b]);
        let mut later_query = query;
        later_query.frame = 8;
        let (later_shared, later_expanded, later_used) =
            prepare_batched_query(&later_solver, later_query, true);
        assert!(later_used);
        assert_eq!(shared, later_shared);
        assert!(
            later_expanded
                .constraints
                .iter()
                .any(|clause| clause.as_slice() == [a, !b])
        );
    }

    #[test]
    fn prepared_group_keeps_exact_formula_after_source_solvers_and_cnf_are_dropped() {
        let (group, a, b) = {
            let mut dc = DagCnf::new();
            let a = dc.new_var().lit();
            let b = dc.new_var().lit();
            dc.add_rel(b.var(), &[LitVec::from([a, b])]);
            let mut first = DagCnfSolver::new(&dc);
            first.add_clause(&[!a, b]);
            let mut second = first.clone();
            second.add_clause(&[a, !b]);
            let mut pending = Vec::new();
            for (index, solver) in [&first, &second].into_iter().enumerate() {
                let mut cache = BatchedSolverContext::new(solver);
                let query = cache.prepare_query(IncrementalQuery::new(0, LitVec::from([a])), true);
                pending.push((index, query, cache.context(true).clone()));
            }
            let group = ActiveBatchGroup {
                context: pending[0].2.clone(), pending, batches: vec![0..2],
            };
            drop(second);
            drop(first);
            drop(dc);
            (group, a, b)
        };
        assert!(group.context.clauses.iter().any(|clause| clause.literals.as_slice() == [a, b]));
        assert_eq!(group.pending[0].2, group.pending[1].2);
        assert!(group.pending[0].1.constraints.iter().any(|clause| clause.as_slice() == [!a, b]));
        assert!(!group.pending[0].1.constraints.iter().any(|clause| clause.as_slice() == [a, !b]));
        assert!(group.pending[1].1.constraints.iter().any(|clause| clause.as_slice() == [a, !b]));
        assert_eq!(group.batches[0], 0..2);
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
        let (context, unchanged, used) = prepare_batched_query(&solver, query.clone(), true);
        assert!(!used);
        assert_eq!(unchanged.frame, query.frame);
        assert_eq!(unchanged.assumptions, query.assumptions);
        assert_eq!(unchanged.constraints, query.constraints);
        assert_eq!(unchanged.domain, query.domain);
        assert_eq!(context.scope, ShadowContextScope::ExactFrame(5));
        assert!(
            context
                .clauses
                .iter()
                .all(|clause| clause.lo == 0 && clause.hi == u32::MAX)
        );
        assert!(context.clauses.len() > query_lemma_word_limit() / 5);
    }

    #[test]
    fn resident_context_appends_only_monotonic_same_frame_deltas() {
        let a = Lit::new(Var::from(0), true);
        let b = Lit::new(Var::from(1), true);
        let c = Lit::new(Var::from(2), true);
        let transition = ResidentClause::new(0, u32::MAX, LitVec::from([a, b]));
        let frame_one_a = ResidentClause::new(1, 1, LitVec::from([!a]));
        let frame_one_b = ResidentClause::new(1, 1, LitVec::from([!b]));
        let frame_two = ResidentClause::new(2, 2, LitVec::from([c]));
        let frame_one = |lemmas: Vec<ResidentClause>| ShadowContext {
            n_var: 3,
            clauses: std::iter::once(transition.clone()).chain(lemmas).collect(),
            scope: ShadowContextScope::ExactFrame(1),
        };
        let target_one = frame_one(vec![frame_one_a.clone(), frame_one_b.clone()]);
        let mut loaded = LoadedContext::from(&frame_one(vec![frame_one_a.clone()]));

        assert_eq!(
            plan_context_update(Some(&loaded), &target_one),
            ContextUpdate::Append(vec![frame_one_b.clone()]),
        );
        loaded.clauses.push(frame_one_b.clone());
        assert_eq!(
            plan_context_update(Some(&loaded), &target_one),
            ContextUpdate::Ready,
        );

        let target_two = ShadowContext {
            n_var: 3,
            clauses: vec![transition.clone(), frame_two.clone()],
            scope: ShadowContextScope::ExactFrame(2),
        };
        assert_eq!(
            plan_context_update(Some(&loaded), &target_two),
            ContextUpdate::Reload,
        );

        // A shorter/reordered frame log or a transition change can no longer
        // be represented by append-only state and must replace the context.
        assert_eq!(
            plan_context_update(Some(&loaded), &frame_one(vec![])),
            ContextUpdate::Reload,
        );
        let changed_transition = ShadowContext {
            n_var: 3,
            clauses: vec![
                ResidentClause::new(0, u32::MAX, LitVec::from([a, c])),
                frame_one_a,
                frame_one_b,
            ],
            scope: ShadowContextScope::ExactFrame(1),
        };
        assert_eq!(
            plan_context_update(Some(&loaded), &changed_transition),
            ContextUpdate::Reload,
        );
    }

    #[test]
    fn frame_ranged_context_appends_only_physical_prefix_deltas() {
        let a = Lit::new(Var::from(0), true);
        let b = Lit::new(Var::from(1), true);
        let transition = ResidentClause::new(0, u32::MAX, LitVec::from([a, b]));
        let early = ResidentClause::new(1, 4, LitVec::from([!a]));
        let later = ResidentClause::new(3, 8, LitVec::from([!b]));
        let context = |clauses| ShadowContext {
            n_var: 2,
            clauses,
            scope: ShadowContextScope::FrameRanged,
        };
        let loaded = LoadedContext::from(&context(vec![transition.clone(), early.clone()]));
        let target = context(vec![transition.clone(), early.clone(), later.clone()]);

        assert_eq!(
            plan_context_update(Some(&loaded), &target),
            ContextUpdate::Append(vec![later.clone()]),
        );
        assert_eq!(
            plan_context_update(Some(&LoadedContext::from(&target)), &target),
            ContextUpdate::Ready,
        );

        // Frame selection happens in the query header, so changing frames does
        // not reorder physical clauses. Any actual reordering still reloads.
        let reordered = context(vec![transition, later, early]);
        assert_eq!(
            plan_context_update(Some(&loaded), &reordered),
            ContextUpdate::Reload,
        );
    }

    #[test]
    fn full_root_formula_oracle_compares_exact_frame_views() {
        let a = Lit::new(Var::from(0), true);
        let b = Lit::new(Var::from(1), true);
        let transition = LitVec::from([a, b]);
        let early = LitVec::from([!a]);
        let later = LitVec::from([!b]);
        let exact = LoadedContext {
            n_var: 2,
            clauses: vec![
                ResidentClause::new(0, u32::MAX, transition.clone()),
                ResidentClause::new(1, 4, early.clone()),
                ResidentClause::new(3, 8, later.clone()),
            ],
            scope: ShadowContextScope::FrameRanged,
        };
        assert!(
            compare_resident_formula_view(&exact, 2, 2, &[early.clone(), transition.clone()],)
                .is_ok()
        );
        assert!(
            compare_resident_formula_view(
                &exact,
                2,
                3,
                &[later.clone(), transition.clone(), early.clone()],
            )
            .is_ok()
        );

        // This is the failure mode the full-root architecture gate targets:
        // a device-side 1..frame append is stronger than the CPU's actual
        // begin..frame insertion at an earlier frame.
        let too_broad = LoadedContext {
            clauses: vec![
                ResidentClause::new(0, u32::MAX, transition.clone()),
                ResidentClause::new(1, 4, early.clone()),
                ResidentClause::new(1, 8, later.clone()),
            ],
            ..exact.clone()
        };
        let error =
            compare_resident_formula_view(&too_broad, 2, 2, &[transition.clone(), early.clone()])
                .unwrap_err();
        assert!(
            error.contains("frame 2")
                && error.contains("device-only")
                && error.contains("physical-ranges=Some([(1, 8)])")
        );

        let first = canonical_clause_set([&transition, &early].into_iter());
        let second = canonical_clause_set([&early, &transition].into_iter());
        assert_eq!(first, second);
        assert_eq!(
            formula_view_fingerprint(&first),
            formula_view_fingerprint(&second)
        );
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
            pending.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
            [0, 2, 3, 1]
        );
        assert_eq!(pair_cost(&pending), 10);
    }

    #[test]
    fn heterogeneous_pair_scheduler_places_arena_pressure_on_lane_one() {
        let mut pressure = IncrementalQuery::new(0, LitVec::new());
        pressure.constraints = (0..513)
            .map(|_| LitVec::from([Lit::new(Var::from(0), true)]))
            .collect();
        let mut short = IncrementalQuery::new(0, LitVec::new());
        short.domain.push(Var::from(0));
        let mut pending = vec![(7usize, pressure), (3usize, short)];

        schedule_query_pairs_for_layout(&mut pending, |(_, query)| query, true);

        assert_eq!(
            pending.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
            [3, 7],
        );
        assert!(
            query_private_arena_score(&pending[1].1) > query_private_arena_score(&pending[0].1)
        );
    }

    #[test]
    fn heterogeneous_pair_scheduler_preserves_fitting_pair_order() {
        let mut more_pressure = IncrementalQuery::new(0, LitVec::new());
        more_pressure.constraints = (0..511)
            .map(|_| LitVec::from([Lit::new(Var::from(0), true)]))
            .collect();
        let mut less_pressure = IncrementalQuery::new(0, LitVec::new());
        less_pressure.constraints = (0..510)
            .map(|_| LitVec::from([Lit::new(Var::from(0), true)]))
            .collect();
        let mut pending = vec![(7usize, more_pressure), (3usize, less_pressure)];

        schedule_query_pairs_for_layout(&mut pending, |(_, query)| query, true);

        assert_eq!(
            pending.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
            [7, 3],
        );
        assert!(
            query_private_arena_score(&pending[0].1) > query_private_arena_score(&pending[1].1)
        );
    }

    #[test]
    fn pair_scheduler_defaults_only_in_throughput_mode() {
        assert!(!pair_scheduler_setting(None, false));
        assert!(pair_scheduler_setting(None, true));
        assert!(pair_scheduler_setting(Some("1"), false));
        assert!(!pair_scheduler_setting(Some("0"), true));
        assert!(!pair_scheduler_setting(Some("false"), true));
        assert!(!pair_scheduler_setting(Some("off"), true));
    }

    #[test]
    fn ranged_context_requires_exact_active_lemma_set() {
        let a = Lit::new(Var::from(1), true);
        let b = Lit::new(Var::from(2), false);
        let clauses = vec![
            ResidentClause::new(1, 3, LitVec::from([a, b])),
            // A duplicate with different literal order is semantically the
            // same clause and must not force a reload.
            ResidentClause::new(2, 2, LitVec::from([b, a])),
            ResidentClause::new(4, 5, LitVec::from([!a])),
        ];
        assert!(ranged_snapshot_matches(
            &clauses,
            2,
            &[LitVec::from([a, b])],
        ));
        assert!(!ranged_snapshot_matches(
            &clauses,
            4,
            &[LitVec::from([a, b])],
        ));
    }
}
