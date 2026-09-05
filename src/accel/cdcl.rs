//! Stable wire contract for a resident incremental CDCL engine.
//!
//! The existing accelerator calls grew around propagation and return a verdict
//! or a candidate core. A SAT-Accel-derived backend needs a different proof
//! boundary: a query names its IC3 frame, carries assumptions, temporary
//! clauses and a decision domain, and returns either a consumable model, an
//! assumption core, or an explicit fallback reason.
//!
//! Keep this module free of `logicrs` types. The same records are intended to
//! be copied into the XRT host and HLS kernel without translating Rust layout.

/// Increment when a header or payload changes incompatibly.
pub const ABI_VERSION: u32 = 2;

/// Return a sparse model over the variables assigned by the search.
pub const WANT_MODEL: u32 = 1 << 0;
/// Return the subset of assumptions used by an UNSAT result.
pub const WANT_CORE: u32 = 1 << 1;
/// Retain bounded learnt clauses in this frame's resident context.
pub const KEEP_LEARNTS: u32 = 1 << 2;
/// Ask a diagnostic image to append per-stage entry counters.
pub const WANT_STAGE_PROFILE: u32 = 1 << 3;
/// Let RUN_MIC_CHAIN use complete SAT models to continue GipSAT's `down`
/// loop on the device instead of merely retaining the attempted literal.
pub const MIC_MODEL_SHRINK: u32 = 1 << 4;
/// The last MIC pair is an Init guard and participates in every inquiry but
/// must never be removed by model-guided shrinking.
pub const MIC_PROTECT_LAST: u32 = 1 << 5;
/// Return a sparse SAT projection over the final temporary clause. This is an
/// internal query flag used by model-guided RUN_MIC_CHAIN refinements.
pub const WANT_LAST_CONSTRAINT_MODEL: u32 = 1 << 6;
/// Protect one cube entry in place. Its zero-based index occupies flags[31:16]
/// so the FPGA follows GipSAT's existing literal/drop order.
pub const MIC_PROTECT_INDEX: u32 = 1 << 7;
/// Domain payload is four bank-aligned scheduled slots per line. This is an
/// internal candidate-image contract; ordinary ABI-v2 images keep it clear.
pub const BANK_ALIGNED_DOMAIN: u32 = 1 << 8;
/// Return one SAT assignment bit per logical domain variable in lane-major
/// bank order. `ResponseHeader::n_model` then counts packed 32-bit words.
pub const PACKED_SAT_MODEL: u32 = 1 << 9;
/// Full-root control only; stripped before ordinary CDCL inquiries.
pub const BLOCK_PREDECESSOR_LIFT: u32 = 1 << 10;
pub const BLOCK_PUSH_LEMMA: u32 = 1 << 11;
pub const BLOCK_REQUEUE: u32 = 1 << 12;
pub const MIC_PROTECTED_INDEX_SHIFT: u32 = 16;

pub const STAGE_PROFILE_MAGIC: u32 = 0x4344_5031; // "CDP1"
pub const STAGE_PROFILE_VERSION: u32 = 3;
pub const STAGE_PROFILE_STAGE_COUNTERS: usize = 9;
pub const STAGE_PROFILE_WORK_COUNTERS: usize = 10;
pub const STAGE_PROFILE_COUNTERS: usize =
    STAGE_PROFILE_STAGE_COUNTERS + STAGE_PROFILE_WORK_COUNTERS;
pub const STAGE_PROFILE_WORDS: usize = 3 + 2 * STAGE_PROFILE_COUNTERS;

pub const PROFILE_SETUP: usize = 0;
pub const PROFILE_ROOT: usize = 1;
pub const PROFILE_PROPAGATE: usize = 2;
pub const PROFILE_ANALYZE: usize = 3;
pub const PROFILE_BACKTRACK: usize = 4;
pub const PROFILE_LEARN: usize = 5;
pub const PROFILE_DECIDE: usize = 6;
pub const PROFILE_EMIT: usize = 7;
pub const PROFILE_CLEANUP: usize = 8;
pub const PROFILE_OCCURRENCE_UPDATES: usize = 9;
pub const PROFILE_PARTIAL_OCCURRENCE_SCANS: usize = 10;
pub const PROFILE_EVALUATED_LITERALS: usize = 11;
pub const PROFILE_UNIT_CANDIDATES: usize = 12;
pub const PROFILE_ANALYZED_LITERALS: usize = 13;
pub const PROFILE_UNDO_OCCURRENCES: usize = 14;
pub const PROFILE_UNDO_ASSIGNMENTS: usize = 15;
pub const PROFILE_LEARNT_LITERALS: usize = 16;
pub const PROFILE_OCCURRENCE_ROUNDS: usize = 17;
pub const PROFILE_OCCURRENCE_PAIRS: usize = 18;

pub const QUERY_HEADER_WORDS: usize = 8;
pub const RESPONSE_HEADER_WORDS: usize = 9;
pub const MIC_HEADER_WORDS: usize = 9;
pub const MIC_RESPONSE_HEADER_WORDS: usize = 12;
pub const MIC_BATCH_HEADER_WORDS: usize = 4;
pub const MIC_BATCH_RESPONSE_HEADER_WORDS: usize = 4;
/// Each lane-affine portfolio record starts with its physical lane id and
/// optional append extent, followed by the append and one complete MIC record.
pub const PORTFOLIO_MIC_RECORD_PREFIX_WORDS: usize = 2;

// Resident BLOCK-program ABI. This is intentionally separate from ABI-v2 SAT
// inquiries: one packet carries a bounded sequence of proof-state mutations,
// and every command receives the exact 14-word response emitted by the shared
// C++/HLS command interpreter. ABI v2 adds a 64-bit obligation user tag to
// each response while retaining all v1 field positions.
pub const BLOCK_SEMANTIC_BATCH_VERSION: u32 = 2;
pub const BLOCK_SEMANTIC_BATCH_HEADER_WORDS: usize = 4;
pub const BLOCK_SEMANTIC_COMMAND_HEADER_WORDS: usize = 6;
pub const BLOCK_SEMANTIC_RESPONSE_HEADER_WORDS: usize = 4;
pub const BLOCK_SEMANTIC_COMMAND_RESPONSE_WORDS: usize = 16;

pub const BLOCK_SEMANTIC_RESET: u32 = 0;
pub const BLOCK_SEMANTIC_REGISTER_OBLIGATION: u32 = 1;
pub const BLOCK_SEMANTIC_INSERT_OBLIGATION: u32 = 2;
pub const BLOCK_SEMANTIC_REMOVE_OBLIGATION: u32 = 3;
pub const BLOCK_SEMANTIC_POP_OBLIGATION: u32 = 4;
pub const BLOCK_SEMANTIC_SET_LEMMA_FRAMES: u32 = 5;
pub const BLOCK_SEMANTIC_REGISTER_LEMMA: u32 = 6;
pub const BLOCK_SEMANTIC_INSERT_LEMMA: u32 = 7;
pub const BLOCK_SEMANTIC_REMOVE_LEMMA: u32 = 8;
pub const BLOCK_SEMANTIC_STATS: u32 = 9;
pub const BLOCK_SEMANTIC_EVENT_REMOVE_OBLIGATION: u32 = 10;
pub const BLOCK_SEMANTIC_EVENT_INSERT_OBLIGATION: u32 = 11;
pub const BLOCK_SEMANTIC_EVENT_CLEAR_OBLIGATIONS: u32 = 12;
pub const BLOCK_SEMANTIC_EVENT_REMOVE_LEMMA: u32 = 13;
pub const BLOCK_SEMANTIC_EVENT_INSERT_LEMMA: u32 = 14;
pub const BLOCK_SEMANTIC_EVENT_SET_LEMMA_FRAMES: u32 = 15;
pub const BLOCK_SEMANTIC_EVENT_RESET_EPOCH: u32 = 16;
pub const BLOCK_SEMANTIC_EVENT_PROMOTE_LEMMA_FRAME_TO_INF: u32 = 17;
pub const BLOCK_SEMANTIC_EVENT_SHIFT_FRAME_SUFFIX_UP: u32 = 18;
pub const BLOCK_SEMANTIC_EVENT_MOVE_LEMMA: u32 = 19;
pub const BLOCK_SEMANTIC_REGISTER_STATE_FULL: u32 = 20;
pub const BLOCK_SEMANTIC_REGISTER_STATE_DELTA: u32 = 21;
pub const BLOCK_SEMANTIC_REGISTER_INPUT_FULL: u32 = 22;
pub const BLOCK_SEMANTIC_REGISTER_INPUT_DELTA: u32 = 23;
pub const BLOCK_SEMANTIC_COMPOSE_OBLIGATION: u32 = 24;
pub const BLOCK_SEMANTIC_INSERT_OBLIGATION_TAGGED: u32 = 25;
pub const BLOCK_SEMANTIC_PEEK_OBLIGATION: u32 = 26;

pub const BLOCK_ROOT_PROTOCOL_VERSION: u32 = 2;
pub const BLOCK_ROOT_REQUEST_HEADER_WORDS: usize = 9;
pub const BLOCK_ROOT_RESPONSE_HEADER_WORDS: usize = 6;
pub const BLOCK_ROOT_WORK_WORDS: usize = 7;
pub const BLOCK_ROOT_MAX_WORK: usize = 8;
pub const BLOCK_ROOT_BATCH_OFFSET: usize =
    BLOCK_ROOT_RESPONSE_HEADER_WORDS + BLOCK_ROOT_MAX_WORK * BLOCK_ROOT_WORK_WORDS;

pub const BLOCK_FULL_ROOT_PROTOCOL_VERSION: u32 = 5;
pub const BLOCK_FULL_ROOT_REQUEST_HEADER_WORDS: usize = 13;
pub const BLOCK_FULL_ROOT_RESPONSE_HEADER_WORDS: usize = 10;
pub const BLOCK_FULL_ROOT_WORK_WORDS: usize = 7;
pub const BLOCK_FULL_ROOT_EVENT_HEADER_WORDS: usize = 2;
pub const BLOCK_FULL_ROOT_LEMMA_HEADER_WORDS: usize = 7;
pub const BLOCK_FULL_ROOT_SAT_HEADER_WORDS: usize = 11;
pub const BLOCK_FULL_ROOT_MAX_STEPS: usize = 256;
pub const BLOCK_FULL_ROOT_CPU_MIC: u32 = 1 << 28;
pub const BLOCK_FULL_ROOT_SKIP_MIC: u32 = 1 << 29;
pub const BLOCK_FULL_ROOT_REUSE_PROJECTION: u32 = 1 << 30;
pub const BLOCK_FULL_ROOT_COMPACTED_RETRY: u32 = 1 << 31;
pub const BLOCK_FULL_ROOT_EVENT_SAT_PREDECESSOR: u32 = 1;
pub const BLOCK_FULL_ROOT_EVENT_UNSAT_LEMMA: u32 = 2;
pub const BLOCK_FULL_ROOT_EVENT_UNSAT_REQUEUE: u32 = 3;

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlockRootExecutionStatus {
    Ok = 0,
    Empty = 1,
    CpuHandoff = 2,
    BadArgument = 3,
    QueryBuild = 4,
    SolverError = 5,
    BadResponse = 6,
    QueueChanged = 7,
}

impl BlockRootExecutionStatus {
    fn from_word(word: u32) -> Option<Self> {
        Some(match word {
            0 => Self::Ok,
            1 => Self::Empty,
            2 => Self::CpuHandoff,
            3 => Self::BadArgument,
            4 => Self::QueryBuild,
            5 => Self::SolverError,
            6 => Self::BadResponse,
            7 => Self::QueueChanged,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockRootWork {
    pub frame: u32,
    pub depth: u32,
    pub removed: u32,
    pub descriptor_handle: u32,
    pub user_tag_lo: u32,
    pub user_tag_hi: u32,
    pub payload_handle: u32,
}

impl BlockRootWork {
    pub fn user_tag(self) -> u64 {
        u64::from(self.user_tag_lo) | (u64::from(self.user_tag_hi) << 32)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockRootResponse {
    pub status: BlockRootExecutionStatus,
    pub work: Vec<BlockRootWork>,
    pub batch: Vec<u32>,
}

pub fn pack_block_root_request(
    max_frame: u32,
    requested_queries: usize,
    next_var_by_current: &[u32],
    decision_domain: &[u32],
    query_flags: u32,
    decision_budget: u32,
    conflict_budget: u32,
) -> Option<Vec<u32>> {
    if requested_queries == 0
        || requested_queries > BLOCK_ROOT_MAX_WORK
        || next_var_by_current.is_empty()
    {
        return None;
    }
    let mut words = Vec::with_capacity(
        BLOCK_ROOT_REQUEST_HEADER_WORDS
            .checked_add(next_var_by_current.len())?
            .checked_add(decision_domain.len())?,
    );
    words.extend([
        BLOCK_ROOT_PROTOCOL_VERSION,
        max_frame,
        u32::try_from(requested_queries).ok()?,
        u32::try_from(next_var_by_current.len()).ok()?,
        u32::try_from(decision_domain.len()).ok()?,
        query_flags,
        decision_budget,
        conflict_budget,
        0,
    ]);
    words.extend_from_slice(next_var_by_current);
    words.extend_from_slice(decision_domain);
    Some(words)
}

pub fn decode_block_root_response(words: &[u32]) -> Option<BlockRootResponse> {
    let header = words.get(..BLOCK_ROOT_RESPONSE_HEADER_WORDS)?;
    if header[0] != BLOCK_ROOT_PROTOCOL_VERSION {
        return None;
    }
    let status = BlockRootExecutionStatus::from_word(header[1])?;
    let work_count = usize::try_from(header[2]).ok()?;
    let metadata_words = usize::try_from(header[3]).ok()?;
    let batch_offset = usize::try_from(header[4]).ok()?;
    let batch_words = usize::try_from(header[5]).ok()?;
    if work_count > BLOCK_ROOT_MAX_WORK
        || metadata_words != work_count.checked_mul(BLOCK_ROOT_WORK_WORDS)?
        || words.len() < BLOCK_ROOT_RESPONSE_HEADER_WORDS.checked_add(metadata_words)?
    {
        return None;
    }
    let mut work = Vec::with_capacity(work_count);
    for record in words
        [BLOCK_ROOT_RESPONSE_HEADER_WORDS..BLOCK_ROOT_RESPONSE_HEADER_WORDS + metadata_words]
        .chunks_exact(BLOCK_ROOT_WORK_WORDS)
    {
        work.push(BlockRootWork {
            frame: record[0],
            depth: record[1],
            removed: record[2],
            descriptor_handle: record[3],
            user_tag_lo: record[4],
            user_tag_hi: record[5],
            payload_handle: record[6],
        });
    }
    let batch = if status == BlockRootExecutionStatus::Ok {
        if work.is_empty()
            || batch_offset != BLOCK_ROOT_BATCH_OFFSET
            || batch_words < 4
            || words.len() != batch_offset.checked_add(batch_words)?
        {
            return None;
        }
        words.get(batch_offset..)?.to_vec()
    } else {
        if batch_offset != 0
            || batch_words != 0
            || words.len() != BLOCK_ROOT_RESPONSE_HEADER_WORDS + metadata_words
            || status == BlockRootExecutionStatus::CpuHandoff && work_count != 1
            || status != BlockRootExecutionStatus::CpuHandoff && work_count != 0
        {
            return None;
        }
        Vec::new()
    };
    Some(BlockRootResponse {
        status,
        work,
        batch,
    })
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlockFullRootStatus {
    Drained = 0,
    CpuResult = 1,
    CpuHandoff = 2,
    StepBudget = 3,
    Fallback = 4,
    Error = 5,
    CompactionRequired = 6,
    Proved = 7,
    CpuMic = 8,
}

impl BlockFullRootStatus {
    fn from_word(word: u32) -> Option<Self> {
        Some(match word {
            0 => Self::Drained,
            1 => Self::CpuResult,
            2 => Self::CpuHandoff,
            3 => Self::StepBudget,
            4 => Self::Fallback,
            5 => Self::Error,
            6 => Self::CompactionRequired,
            7 => Self::Proved,
            8 => Self::CpuMic,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BlockFullRootEvent {
    SatPredecessor {
        child_tag: u64,
        parent_tag: u64,
        parent_descriptor_handle: u32,
        child_descriptor_handle: u32,
        child_payload_handle: u32,
        frame: u32,
        depth: u32,
        state: Vec<u32>,
        input: Vec<u32>,
    },
    UnsatLemma {
        frame: u32,
        begin_frame: u32,
        proof_tag: u64,
        payload_handle: u32,
        descriptor_handle: u32,
        requeued_descriptor: Option<u32>,
        cube: Vec<u32>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockFullRootResponse {
    pub status: BlockFullRootStatus,
    pub handoff: Option<BlockRootWork>,
    /// Reconstructed current-state failed-assumption core for `CpuMic`.
    /// CPU consumes this trusted Q_block result as MIC input without solving
    /// the original inquiry again.
    pub handoff_core: Option<Vec<u32>>,
    pub events: Vec<BlockFullRootEvent>,
    pub cdcl_waves: u32,
    pub cdcl_inquiries: u32,
    pub mic_waves: u32,
    pub sat_commits: u32,
    pub unsat_commits: u32,
}

pub fn block_full_root_required_response_capacity(
    step_limit: usize,
    latch_count: usize,
    input_count: usize,
) -> Option<usize> {
    let lemma = BLOCK_FULL_ROOT_LEMMA_HEADER_WORDS.checked_add(latch_count)?;
    let sat = BLOCK_FULL_ROOT_SAT_HEADER_WORDS
        .checked_add(latch_count)?
        .checked_add(input_count)?;
    BLOCK_FULL_ROOT_RESPONSE_HEADER_WORDS
        .checked_add(BLOCK_FULL_ROOT_WORK_WORDS)?
        .checked_add(
            step_limit
                .checked_mul(BLOCK_FULL_ROOT_EVENT_HEADER_WORDS.checked_add(lemma.max(sat))?)?,
        )
}

pub fn pack_block_full_root_request(
    max_frame: u32,
    step_limit: usize,
    frontier_limit: usize,
    next_var_by_current: &[u32],
    init_value_by_current: &[u32],
    decision_domain: &[u32],
    latch_variables: &[u32],
    input_variables: &[u32],
    query_flags: u32,
    decision_budget: u32,
    conflict_budget: u32,
    projection_handle: u32,
    compacted_retry: bool,
) -> Option<Vec<u32>> {
    if step_limit == 0
        || step_limit > BLOCK_FULL_ROOT_MAX_STEPS
        || frontier_limit == 0
        || frontier_limit > BLOCK_ROOT_MAX_WORK
        || next_var_by_current.is_empty()
        || init_value_by_current.len() != next_var_by_current.len()
        || init_value_by_current.iter().any(|value| *value > 2)
        || latch_variables.len() > next_var_by_current.len()
        || input_variables.len() > next_var_by_current.len()
        || projection_handle == 0
        || query_flags & (BANK_ALIGNED_DOMAIN | PACKED_SAT_MODEL)
            != (BANK_ALIGNED_DOMAIN | PACKED_SAT_MODEL)
    {
        return None;
    }
    let capacity = BLOCK_FULL_ROOT_REQUEST_HEADER_WORDS
        .checked_add(2usize.checked_mul(next_var_by_current.len())?)?
        .checked_add(decision_domain.len())?
        .checked_add(latch_variables.len())?
        .checked_add(input_variables.len())?;
    let mut words = Vec::with_capacity(capacity);
    words.extend([
        BLOCK_FULL_ROOT_PROTOCOL_VERSION,
        max_frame,
        u32::try_from(step_limit).ok()?,
        u32::try_from(next_var_by_current.len()).ok()?,
        u32::try_from(decision_domain.len()).ok()?,
        query_flags,
        decision_budget,
        conflict_budget,
        u32::try_from(latch_variables.len()).ok()?,
        u32::try_from(input_variables.len()).ok()?,
        u32::try_from(frontier_limit).ok()?,
        if compacted_retry {
            BLOCK_FULL_ROOT_COMPACTED_RETRY
        } else {
            0
        },
        projection_handle,
    ]);
    words.extend_from_slice(next_var_by_current);
    words.extend_from_slice(init_value_by_current);
    words.extend_from_slice(decision_domain);
    words.extend_from_slice(latch_variables);
    words.extend_from_slice(input_variables);
    Some(words)
}

pub fn pack_block_full_root_continuation(
    max_frame: u32,
    step_limit: usize,
    frontier_limit: usize,
    n_var: usize,
    domain_words: usize,
    latch_count: usize,
    input_count: usize,
    query_flags: u32,
    decision_budget: u32,
    conflict_budget: u32,
    projection_handle: u32,
    compacted_retry: bool,
) -> Option<Vec<u32>> {
    if step_limit == 0
        || step_limit > BLOCK_FULL_ROOT_MAX_STEPS
        || frontier_limit == 0
        || frontier_limit > BLOCK_ROOT_MAX_WORK
        || n_var == 0
        || latch_count > n_var
        || input_count > n_var
        || projection_handle == 0
        || query_flags & (BANK_ALIGNED_DOMAIN | PACKED_SAT_MODEL)
            != (BANK_ALIGNED_DOMAIN | PACKED_SAT_MODEL)
    {
        return None;
    }
    Some(vec![
        BLOCK_FULL_ROOT_PROTOCOL_VERSION,
        max_frame,
        u32::try_from(step_limit).ok()?,
        u32::try_from(n_var).ok()?,
        u32::try_from(domain_words).ok()?,
        query_flags,
        decision_budget,
        conflict_budget,
        u32::try_from(latch_count).ok()?,
        u32::try_from(input_count).ok()?,
        u32::try_from(frontier_limit).ok()?,
        BLOCK_FULL_ROOT_REUSE_PROJECTION
            | if compacted_retry {
                BLOCK_FULL_ROOT_COMPACTED_RETRY
            } else {
                0
            },
        projection_handle,
    ])
}

pub fn decode_block_full_root_response(words: &[u32]) -> Option<BlockFullRootResponse> {
    let header = words.get(..BLOCK_FULL_ROOT_RESPONSE_HEADER_WORDS)?;
    if header[0] != BLOCK_FULL_ROOT_PROTOCOL_VERSION {
        return None;
    }
    let status = BlockFullRootStatus::from_word(header[1])?;
    let handoff_count = usize::try_from(header[2]).ok()?;
    let event_count = usize::try_from(header[3]).ok()?;
    let payload_words = usize::try_from(header[8]).ok()?;
    if handoff_count > 1
        || words.len() != BLOCK_FULL_ROOT_RESPONSE_HEADER_WORDS.checked_add(payload_words)?
    {
        return None;
    }
    let mut at = BLOCK_FULL_ROOT_RESPONSE_HEADER_WORDS;
    let handoff = if handoff_count == 1 {
        let record = words.get(at..at.checked_add(BLOCK_FULL_ROOT_WORK_WORDS)?)?;
        at += BLOCK_FULL_ROOT_WORK_WORDS;
        Some(BlockRootWork {
            frame: record[0],
            depth: record[1],
            removed: record[2],
            descriptor_handle: record[3],
            user_tag_lo: record[4],
            user_tag_hi: record[5],
            payload_handle: record[6],
        })
    } else {
        None
    };
    let handoff_core = if status == BlockFullRootStatus::CpuMic {
        let core_words = usize::try_from(*words.get(at)?).ok()?;
        at = at.checked_add(1)?;
        if core_words == 0 {
            return None;
        }
        let core = words.get(at..at.checked_add(core_words)?)?.to_vec();
        at += core_words;
        Some(core)
    } else {
        None
    };
    if matches!(
        status,
        BlockFullRootStatus::CpuResult
            | BlockFullRootStatus::CpuHandoff
            | BlockFullRootStatus::CpuMic
            | BlockFullRootStatus::Fallback
    ) != handoff.is_some()
        || matches!(
            status,
            BlockFullRootStatus::Drained
                | BlockFullRootStatus::StepBudget
                | BlockFullRootStatus::CompactionRequired
                | BlockFullRootStatus::Proved
        ) && handoff.is_some()
    {
        return None;
    }

    let mut events = Vec::with_capacity(event_count);
    let mut sat_events = 0u32;
    let mut unsat_events = 0u32;
    for _ in 0..event_count {
        let event_header = words.get(at..at.checked_add(BLOCK_FULL_ROOT_EVENT_HEADER_WORDS)?)?;
        at += BLOCK_FULL_ROOT_EVENT_HEADER_WORDS;
        let kind = event_header[0];
        let record_words = usize::try_from(event_header[1]).ok()?;
        let end = at.checked_add(record_words)?;
        let record = words.get(at..end)?;
        let event = match kind {
            BLOCK_FULL_ROOT_EVENT_UNSAT_LEMMA | BLOCK_FULL_ROOT_EVENT_UNSAT_REQUEUE => {
                if record.len() < BLOCK_FULL_ROOT_LEMMA_HEADER_WORDS {
                    return None;
                }
                let cube_words = usize::try_from(record[6]).ok()?;
                let requeue = kind == BLOCK_FULL_ROOT_EVENT_UNSAT_REQUEUE;
                if requeue && record[0] == u32::MAX { return None; }
                let header_words = BLOCK_FULL_ROOT_LEMMA_HEADER_WORDS + usize::from(requeue);
                if record.len() != header_words.checked_add(cube_words)? {
                    return None;
                }
                unsat_events = unsat_events.checked_add(1)?;
                BlockFullRootEvent::UnsatLemma {
                    frame: record[0],
                    begin_frame: record[1],
                    proof_tag: u64::from(record[2]) | (u64::from(record[3]) << 32),
                    payload_handle: record[4],
                    descriptor_handle: record[5],
                    requeued_descriptor: requeue.then(|| record[7]),
                    cube: record[header_words..].to_vec(),
                }
            }
            BLOCK_FULL_ROOT_EVENT_SAT_PREDECESSOR => {
                if record.len() < BLOCK_FULL_ROOT_SAT_HEADER_WORDS {
                    return None;
                }
                let state_words = usize::try_from(record[9]).ok()?;
                let input_words = usize::try_from(record[10]).ok()?;
                if record.len()
                    != BLOCK_FULL_ROOT_SAT_HEADER_WORDS
                        .checked_add(state_words)?
                        .checked_add(input_words)?
                {
                    return None;
                }
                sat_events = sat_events.checked_add(1)?;
                let state_begin = BLOCK_FULL_ROOT_SAT_HEADER_WORDS;
                let input_begin = state_begin.checked_add(state_words)?;
                BlockFullRootEvent::SatPredecessor {
                    child_tag: u64::from(record[0]) | (u64::from(record[1]) << 32),
                    parent_tag: u64::from(record[2]) | (u64::from(record[3]) << 32),
                    parent_descriptor_handle: record[4],
                    child_descriptor_handle: record[5],
                    child_payload_handle: record[6],
                    frame: record[7],
                    depth: record[8],
                    state: record[state_begin..input_begin].to_vec(),
                    input: record[input_begin..].to_vec(),
                }
            }
            _ => return None,
        };
        events.push(event);
        at = end;
    }
    if at != words.len() || sat_events != header[6] || unsat_events != header[7] {
        return None;
    }
    let committed = header[6].checked_add(header[7])?;
    if committed > header[9] || header[4] > header[9] {
        return None;
    }
    Some(BlockFullRootResponse {
        status,
        handoff,
        handoff_core,
        events,
        cdcl_waves: header[4],
        cdcl_inquiries: header[9],
        mic_waves: header[5],
        sat_commits: header[6],
        unsat_commits: header[7],
    })
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BlockSemanticCommand {
    pub command: u32,
    pub frame: u32,
    pub depth: u32,
    pub removed: u32,
    pub handle: u32,
    pub payload: Vec<u32>,
}

impl BlockSemanticCommand {
    pub fn new(command: u32) -> Self {
        Self {
            command,
            ..Self::default()
        }
    }

    pub fn packed_words(&self) -> Option<usize> {
        BLOCK_SEMANTIC_COMMAND_HEADER_WORDS.checked_add(self.payload.len())
    }

    fn append_words(&self, words: &mut Vec<u32>) -> Option<()> {
        words.extend([
            self.command,
            self.frame,
            self.depth,
            self.removed,
            self.handle,
            u32::try_from(self.payload.len()).ok()?,
        ]);
        words.extend_from_slice(&self.payload);
        Some(())
    }
}

/// Pack a complete resident proof-state transaction. The declared response
/// capacity is part of the request so RPC, native simulation and a future ring
/// transport reject truncation identically.
pub fn pack_block_semantic_batch(commands: &[BlockSemanticCommand]) -> Option<Vec<u32>> {
    let command_words = commands.iter().try_fold(0usize, |total, command| {
        total.checked_add(command.packed_words()?)
    })?;
    let response_words = BLOCK_SEMANTIC_RESPONSE_HEADER_WORDS.checked_add(
        commands
            .len()
            .checked_mul(BLOCK_SEMANTIC_COMMAND_RESPONSE_WORDS)?,
    )?;
    let mut words =
        Vec::with_capacity(BLOCK_SEMANTIC_BATCH_HEADER_WORDS.checked_add(command_words)?);
    words.extend([
        BLOCK_SEMANTIC_BATCH_VERSION,
        u32::try_from(commands.len()).ok()?,
        u32::try_from(command_words).ok()?,
        u32::try_from(response_words).ok()?,
    ]);
    for command in commands {
        command.append_words(&mut words)?;
    }
    Some(words)
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BlockSemanticCommandResponse {
    pub status: u32,
    pub found: u32,
    pub obligation_count: u32,
    pub lemma_count: u32,
    pub obligation_arena_words: u32,
    pub lemma_arena_words: u32,
    pub output_handle: u32,
    pub popped_frame: u32,
    pub popped_depth: u32,
    pub popped_removed: u32,
    pub popped_sample: u32,
    pub lemma_frame_count: u32,
    pub state_arena_words: u32,
    pub input_arena_words: u32,
    pub popped_user_tag_lo: u32,
    pub popped_user_tag_hi: u32,
}

impl BlockSemanticCommandResponse {
    pub fn from_words(words: &[u32]) -> Option<Self> {
        let words: &[u32; BLOCK_SEMANTIC_COMMAND_RESPONSE_WORDS] = words.try_into().ok()?;
        Some(Self {
            status: words[0],
            found: words[1],
            obligation_count: words[2],
            lemma_count: words[3],
            obligation_arena_words: words[4],
            lemma_arena_words: words[5],
            output_handle: words[6],
            popped_frame: words[7],
            popped_depth: words[8],
            popped_removed: words[9],
            popped_sample: words[10],
            lemma_frame_count: words[11],
            state_arena_words: words[12],
            input_arena_words: words[13],
            popped_user_tag_lo: words[14],
            popped_user_tag_hi: words[15],
        })
    }

    pub fn popped_user_tag(&self) -> u64 {
        u64::from(self.popped_user_tag_lo) | (u64::from(self.popped_user_tag_hi) << 32)
    }
}

pub fn decode_block_semantic_batch_response(
    words: &[u32],
) -> Option<(u32, Vec<BlockSemanticCommandResponse>)> {
    let header = words.get(..BLOCK_SEMANTIC_RESPONSE_HEADER_WORDS)?;
    if header[0] != BLOCK_SEMANTIC_BATCH_VERSION {
        return None;
    }
    let completed = usize::try_from(header[1]).ok()?;
    let result_words = usize::try_from(header[2]).ok()?;
    if result_words != completed.checked_mul(BLOCK_SEMANTIC_COMMAND_RESPONSE_WORDS)?
        || words.len() != BLOCK_SEMANTIC_RESPONSE_HEADER_WORDS.checked_add(result_words)?
    {
        return None;
    }
    let mut responses = Vec::with_capacity(completed);
    for record in words[BLOCK_SEMANTIC_RESPONSE_HEADER_WORDS..]
        .chunks_exact(BLOCK_SEMANTIC_COMMAND_RESPONSE_WORDS)
    {
        responses.push(BlockSemanticCommandResponse::from_words(record)?);
    }
    Some((header[3], responses))
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Status {
    Sat = 1,
    Unsat = 2,
    Unknown = 3,
    Error = 4,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum UnknownReason {
    #[default]
    None = 0,
    DecisionBudget = 1,
    ConflictBudget = 2,
    Capacity = 3,
    FrameMiss = 4,
    Unsupported = 5,
    BackendError = 6,
    RestartBudget = 7,
}

/// A batch is one DMA submission containing consecutive query records. Each
/// record is `QueryHeader::as_words()` followed immediately by its payload.
/// Results occupy a separate caller-sized buffer and remain in query order.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BatchHeader {
    pub version: u32,
    pub n_queries: u32,
    pub n_request_words: u32,
    pub result_capacity_words: u32,
}

/// Batch completion prefix followed by `n_queries` variable-length records.
/// Each record is a `ResponseHeader`, then its model and core words.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BatchResponseHeader {
    pub version: u32,
    pub n_queries: u32,
    pub n_result_words: u32,
    pub error: u32,
}

impl BatchHeader {
    pub fn valid_for(&self, request_words: &[u32]) -> bool {
        if self.version != ABI_VERSION
            || usize::try_from(self.n_request_words).ok() != Some(request_words.len())
        {
            return false;
        }

        let mut offset = 0usize;
        for _ in 0..self.n_queries {
            let Some(header_words) = request_words.get(offset..offset + QUERY_HEADER_WORDS) else {
                return false;
            };
            let Some(header) = QueryHeader::from_words(header_words) else {
                return false;
            };
            offset += QUERY_HEADER_WORDS;
            let Some(n_payload) = header.payload_words() else {
                return false;
            };
            let Some(payload) = request_words.get(offset..offset + n_payload) else {
                return false;
            };
            if !header.valid_for(payload) {
                return false;
            }
            offset += n_payload;
        }
        offset == request_words.len()
    }
}

/// Header followed by three packed regions in this exact order:
///
/// 1. `n_assumptions` encoded literals;
/// 2. `n_constraint_words` words, each clause encoded as `[len, literals...]`;
/// 3. `n_domain` encoded variable identifiers.
///
/// A zero budget means unlimited. The FPGA may still apply a build-time hard
/// cap, but reaching it must return `Unknown`, never a guessed verdict.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct QueryHeader {
    pub version: u32,
    pub frame: u32,
    pub flags: u32,
    pub n_assumptions: u32,
    pub n_constraint_words: u32,
    pub n_domain: u32,
    pub decision_budget: u32,
    pub conflict_budget: u32,
}

/// One device-resident dependent MIC traversal. The payload contains packed
/// temporary constraints, full-domain variables, then current/next literal
/// pairs. A zero `max_trials` means traverse the whole input cube.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MicHeader {
    pub version: u32,
    pub frame: u32,
    pub flags: u32,
    pub n_cube: u32,
    pub n_constraint_words: u32,
    pub n_domain: u32,
    pub decision_budget: u32,
    pub conflict_budget: u32,
    pub max_trials: u32,
}

impl MicHeader {
    pub fn as_words(&self) -> [u32; MIC_HEADER_WORDS] {
        [
            self.version,
            self.frame,
            self.flags,
            self.n_cube,
            self.n_constraint_words,
            self.n_domain,
            self.decision_budget,
            self.conflict_budget,
            self.max_trials,
        ]
    }

    pub fn payload_words(&self) -> Option<usize> {
        usize::try_from(self.n_constraint_words)
            .ok()?
            .checked_add(usize::try_from(self.n_domain).ok()?)?
            .checked_add(usize::try_from(self.n_cube).ok()?.checked_mul(2)?)
    }

    pub fn valid_for(&self, payload: &[u32]) -> bool {
        if self.version != ABI_VERSION
            || self.n_cube < 2
            || self.flags & BANK_ALIGNED_DOMAIN != 0 && self.n_domain & 3 != 0
            || self.payload_words() != Some(payload.len())
        {
            return false;
        }
        let end = self.n_constraint_words as usize;
        let mut p = 0usize;
        while p < end {
            let len = payload[p] as usize;
            p += 1;
            if len == 0 || p.checked_add(len).is_none_or(|next| next > end) {
                return false;
            }
            p += len;
        }
        p == end
    }
}

impl QueryHeader {
    pub fn as_words(&self) -> [u32; QUERY_HEADER_WORDS] {
        [
            self.version,
            self.frame,
            self.flags,
            self.n_assumptions,
            self.n_constraint_words,
            self.n_domain,
            self.decision_budget,
            self.conflict_budget,
        ]
    }

    pub fn from_words(words: &[u32]) -> Option<Self> {
        let words: &[u32; QUERY_HEADER_WORDS] = words.try_into().ok()?;
        Some(Self {
            version: words[0],
            frame: words[1],
            flags: words[2],
            n_assumptions: words[3],
            n_constraint_words: words[4],
            n_domain: words[5],
            decision_budget: words[6],
            conflict_budget: words[7],
        })
    }

    pub fn payload_words(&self) -> Option<usize> {
        let assumptions = usize::try_from(self.n_assumptions).ok()?;
        let constraints = usize::try_from(self.n_constraint_words).ok()?;
        let domain = usize::try_from(self.n_domain).ok()?;
        assumptions.checked_add(constraints)?.checked_add(domain)
    }

    pub fn valid_for(&self, payload: &[u32]) -> bool {
        if self.version != ABI_VERSION
            || self.payload_words() != Some(payload.len())
            || self.flags & PACKED_SAT_MODEL != 0 && self.flags & BANK_ALIGNED_DOMAIN == 0
            || self.flags & BANK_ALIGNED_DOMAIN != 0 && self.n_domain & 3 != 0
        {
            return false;
        }
        let begin = self.n_assumptions as usize;
        let end = begin + self.n_constraint_words as usize;
        let mut p = begin;
        while p < end {
            let len = payload[p] as usize;
            p += 1;
            if len == 0 || p.checked_add(len).is_none_or(|next| next > end) {
                return false;
            }
            p += len;
        }
        p == end
    }
}

impl ResponseHeader {
    pub fn from_words(words: &[u32]) -> Option<Self> {
        let words: &[u32; RESPONSE_HEADER_WORDS] = words.try_into().ok()?;
        Some(Self {
            status: words[0],
            reason: words[1],
            n_model: words[2],
            n_core: words[3],
            decisions: words[4],
            conflicts: words[5],
            propagations: words[6],
            learnt_clauses: words[7],
            error: words[8],
        })
    }
}

impl Status {
    pub fn from_word(word: u32) -> Option<Self> {
        match word {
            1 => Some(Self::Sat),
            2 => Some(Self::Unsat),
            3 => Some(Self::Unknown),
            4 => Some(Self::Error),
            _ => None,
        }
    }
}

impl UnknownReason {
    pub fn from_word(word: u32) -> Option<Self> {
        match word {
            0 => Some(Self::None),
            1 => Some(Self::DecisionBudget),
            2 => Some(Self::ConflictBudget),
            3 => Some(Self::Capacity),
            4 => Some(Self::FrameMiss),
            5 => Some(Self::Unsupported),
            6 => Some(Self::BackendError),
            7 => Some(Self::RestartBudget),
            _ => None,
        }
    }
}

/// Fixed-size completion record. Model and core literals follow in the result
/// buffer, model first. Work counters make batch comparisons meaningful even
/// when different queries take different search paths.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResponseHeader {
    pub status: u32,
    pub reason: u32,
    pub n_model: u32,
    pub n_core: u32,
    pub decisions: u32,
    pub conflicts: u32,
    pub propagations: u32,
    pub learnt_clauses: u32,
    pub error: u32,
}

/// MIC completion prefix followed by exactly `n_output` current-state
/// literals. `complete == 0` is a usable partial traversal, subject to the
/// same exact CPU re-proof as a complete result.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MicResponseHeader {
    pub version: u32,
    pub n_input: u32,
    pub n_output: u32,
    pub trials: u32,
    pub complete: u32,
    pub reason: u32,
    pub decisions: u32,
    pub conflicts: u32,
    pub propagations: u32,
    pub learnt_clauses: u32,
    pub error: u32,
    pub physical_rounds: u32,
}

impl MicResponseHeader {
    pub fn from_words(words: &[u32]) -> Option<Self> {
        let words: &[u32; MIC_RESPONSE_HEADER_WORDS] = words.try_into().ok()?;
        Some(Self {
            version: words[0],
            n_input: words[1],
            n_output: words[2],
            trials: words[3],
            complete: words[4],
            reason: words[5],
            decisions: words[6],
            conflicts: words[7],
            propagations: words[8],
            learnt_clauses: words[9],
            error: words[10],
            physical_rounds: words[11],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_layout_is_fixed_and_payload_is_checked() {
        assert_eq!(std::mem::size_of::<QueryHeader>(), 8 * 4);
        assert_eq!(std::mem::size_of::<ResponseHeader>(), 9 * 4);
        assert_eq!(std::mem::size_of::<BatchHeader>(), 4 * 4);
        assert_eq!(std::mem::size_of::<BatchResponseHeader>(), 4 * 4);
        assert_eq!(std::mem::size_of::<MicHeader>(), 9 * 4);
        assert_eq!(std::mem::size_of::<MicResponseHeader>(), 12 * 4);

        let mic_response =
            MicResponseHeader::from_words(&[ABI_VERSION, 4, 4, 4, 1, 0, 7, 2, 19, 1, 0, 1])
                .unwrap();
        assert_eq!(mic_response.trials, 4);
        assert_eq!(mic_response.physical_rounds, 1);

        let header = QueryHeader {
            version: ABI_VERSION,
            frame: 3,
            flags: WANT_MODEL | WANT_CORE | KEEP_LEARNTS,
            n_assumptions: 2,
            n_constraint_words: 4,
            n_domain: 2,
            decision_budget: 32,
            conflict_budget: 8,
        };
        // assumptions, then one 3-literal temporary clause, then the domain.
        let payload = [10, 12, 3, 20, 22, 24, 5, 6];
        assert_eq!(header.payload_words(), Some(payload.len()));
        assert!(header.valid_for(&payload));

        let mut malformed = payload;
        malformed[2] = 4;
        assert!(!header.valid_for(&malformed));

        let mut request_words = header.as_words().to_vec();
        request_words.extend(payload);
        request_words.extend(header.as_words());
        request_words.extend(payload);
        let batch = BatchHeader {
            version: ABI_VERSION,
            n_queries: 2,
            n_request_words: request_words.len() as u32,
            result_capacity_words: 128,
        };
        assert!(batch.valid_for(&request_words));
        request_words.pop();
        assert!(!batch.valid_for(&request_words));

        let mic = MicHeader {
            version: ABI_VERSION,
            frame: 3,
            n_cube: 2,
            n_constraint_words: 3,
            n_domain: 4,
            conflict_budget: 16,
            ..MicHeader::default()
        };
        let mic_payload = [2, 10, 12, 0, 1, 2, 3, 10, 20, 12, 22];
        assert_eq!(mic.payload_words(), Some(mic_payload.len()));
        assert!(mic.valid_for(&mic_payload));
        let mut malformed_mic = mic_payload;
        malformed_mic[0] = 3;
        assert!(!mic.valid_for(&malformed_mic));
    }

    #[test]
    fn block_semantic_batch_matches_cpp_word_abi() {
        let mut state_full = BlockSemanticCommand::new(BLOCK_SEMANTIC_REGISTER_STATE_FULL);
        state_full.payload = vec![2, 4];
        let mut input_full = BlockSemanticCommand::new(BLOCK_SEMANTIC_REGISTER_INPUT_FULL);
        input_full.payload = vec![3, 7];
        let mut compose = BlockSemanticCommand::new(BLOCK_SEMANTIC_COMPOSE_OBLIGATION);
        compose.payload = vec![0];
        let mut frames = BlockSemanticCommand::new(BLOCK_SEMANTIC_SET_LEMMA_FRAMES);
        frames.frame = 2;
        let mut insert = BlockSemanticCommand::new(BLOCK_SEMANTIC_INSERT_OBLIGATION);
        insert.frame = 1;
        insert.depth = 3;
        let commands = [
            BlockSemanticCommand::new(BLOCK_SEMANTIC_RESET),
            state_full,
            input_full,
            compose,
            frames,
            insert,
            BlockSemanticCommand::new(BLOCK_SEMANTIC_STATS),
        ];
        let words = pack_block_semantic_batch(&commands).unwrap();
        assert_eq!(words[..4], [2, 7, 47, 116]);
        assert_eq!(words.len(), 51);
        assert_eq!(&words[4..10], &[BLOCK_SEMANTIC_RESET, 0, 0, 0, 0, 0]);
        assert_eq!(
            &words[10..18],
            &[BLOCK_SEMANTIC_REGISTER_STATE_FULL, 0, 0, 0, 0, 2, 2, 4],
        );
        assert_eq!(
            &words[18..26],
            &[BLOCK_SEMANTIC_REGISTER_INPUT_FULL, 0, 0, 0, 0, 2, 3, 7],
        );

        let mut response = vec![BLOCK_SEMANTIC_BATCH_VERSION, 1, 16, 0];
        response.extend([
            0, 1, 2, 3, 10, 11, 7, 4, 5, 0, 9, 6, 2, 8, 0x89abcdef, 0x01234567,
        ]);
        let (error, decoded) = decode_block_semantic_batch_response(&response).unwrap();
        assert_eq!(error, 0);
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].output_handle, 7);
        assert_eq!(decoded[0].state_arena_words, 2);
        assert_eq!(decoded[0].popped_user_tag(), 0x0123456789abcdef);
        response[2] = 15;
        assert!(decode_block_semantic_batch_response(&response).is_none());
    }

    #[test]
    fn block_root_protocol_matches_cpp_word_abi() {
        let request = pack_block_root_request(
            3,
            2,
            &[4, 6, u32::MAX, u32::MAX],
            &[0, 1, 2, 3],
            WANT_MODEL | WANT_CORE,
            11,
            17,
        )
        .unwrap();
        assert_eq!(request.len(), 17);
        assert_eq!(
            &request[..9],
            &[BLOCK_ROOT_PROTOCOL_VERSION, 3, 2, 4, 4, 3, 11, 17, 0],
        );

        let mut response = vec![0u32; BLOCK_ROOT_BATCH_OFFSET];
        response[..6].copy_from_slice(&[
            BLOCK_ROOT_PROTOCOL_VERSION,
            BlockRootExecutionStatus::Ok as u32,
            2,
            14,
            BLOCK_ROOT_BATCH_OFFSET as u32,
            4,
        ]);
        response[6..20].copy_from_slice(&[
            1, 10, 0, 7, 0x89abcdef, 0x01234567, 3, 2, 5, 0, 8, 0x76543210, 0xfedcba98, 4,
        ]);
        response.extend([ABI_VERSION, 2, 0, 0]);
        let decoded = decode_block_root_response(&response).unwrap();
        assert_eq!(decoded.status, BlockRootExecutionStatus::Ok);
        assert_eq!(decoded.work.len(), 2);
        assert_eq!(decoded.work[0].user_tag(), 0x0123456789abcdef);
        assert_eq!(decoded.batch, [ABI_VERSION, 2, 0, 0]);

        response[4] = 0;
        assert!(decode_block_root_response(&response).is_none());
        assert_eq!(
            decode_block_root_response(&[
                BLOCK_ROOT_PROTOCOL_VERSION,
                BlockRootExecutionStatus::Empty as u32,
                0,
                0,
                0,
                0,
            ])
            .unwrap()
            .status,
            BlockRootExecutionStatus::Empty,
        );
    }

    #[test]
    fn block_full_root_protocol_preserves_ordered_sat_and_unsat_journal() {
        let flags = WANT_MODEL | WANT_CORE | BANK_ALIGNED_DOMAIN | PACKED_SAT_MODEL;
        let request = pack_block_full_root_request(
            3,
            4,
            2,
            &[4, 6, u32::MAX, u32::MAX],
            &[1, 0, 2, 2],
            &[0x8000_0000, 0x8001_0001, 0x8002_0002, 0x8003_0003],
            &[0, 1],
            &[3],
            flags,
            11,
            17,
            7,
            false,
        )
        .unwrap();
        assert_eq!(request.len(), 28);
        assert_eq!(
            &request[..13],
            &[
                BLOCK_FULL_ROOT_PROTOCOL_VERSION,
                3,
                4,
                4,
                4,
                flags,
                11,
                17,
                2,
                1,
                2,
                0,
                7,
            ],
        );
        assert_eq!(
            pack_block_full_root_continuation(3, 4, 2, 4, 4, 2, 1, flags, 11, 17, 7, false,)
                .unwrap(),
            vec![
                BLOCK_FULL_ROOT_PROTOCOL_VERSION,
                3,
                4,
                4,
                4,
                flags,
                11,
                17,
                2,
                1,
                2,
                BLOCK_FULL_ROOT_REUSE_PROJECTION,
                7,
            ],
        );
        assert_eq!(
            block_full_root_required_response_capacity(4, 2, 1),
            Some(81)
        );

        let response = [
            BLOCK_FULL_ROOT_PROTOCOL_VERSION,
            BlockFullRootStatus::CpuHandoff as u32,
            1,
            2,
            2,
            1,
            1,
            1,
            34,
            2,
            // CPU handoff work.
            0,
            6,
            0,
            9,
            0x22,
            0x8000_0000,
            4,
            // SAT event and its 11-word record.
            BLOCK_FULL_ROOT_EVENT_SAT_PREDECESSOR,
            14,
            0x22,
            0x8000_0000,
            0x11,
            0,
            7,
            8,
            9,
            0,
            6,
            2,
            1,
            0,
            3,
            3,
            // UNSAT lemma event and its 7-word fixed header.
            BLOCK_FULL_ROOT_EVENT_UNSAT_LEMMA,
            9,
            2,
            2,
            0x33,
            0,
            4,
            5,
            2,
            0,
            2,
        ];
        let decoded = decode_block_full_root_response(&response).unwrap();
        assert_eq!(decoded.status, BlockFullRootStatus::CpuHandoff);
        assert_eq!(decoded.handoff.unwrap().user_tag(), 0x8000_0000_0000_0022);
        assert_eq!(decoded.events.len(), 2);
        assert_eq!(decoded.cdcl_inquiries, 2);
        assert!(matches!(
            &decoded.events[0],
            BlockFullRootEvent::SatPredecessor { state, input, .. }
                if state == &[0, 3] && input == &[3]
        ));
        assert!(matches!(
            &decoded.events[1],
            BlockFullRootEvent::UnsatLemma { frame: 2, begin_frame: 2, cube, .. }
                if cube == &[0, 2]
        ));

        // Kind 3 extends only this lemma record; old kinds/offsets stay valid.
        let mut requeue_response = response.to_vec();
        let lemma_at = requeue_response.len() - 11;
        requeue_response[lemma_at] = BLOCK_FULL_ROOT_EVENT_UNSAT_REQUEUE;
        requeue_response[lemma_at + 1] += 1;
        requeue_response[8] += 1;
        requeue_response.insert(lemma_at + 2 + 7, 17);
        assert!(matches!(
            &decode_block_full_root_response(&requeue_response).unwrap().events[1],
            BlockFullRootEvent::UnsatLemma { requeued_descriptor: Some(17), cube, .. }
                if cube == &[0, 2]
        ));
        requeue_response[lemma_at + 2 + 7] = u32::MAX;
        assert!(decode_block_full_root_response(&requeue_response).is_some());
        requeue_response.pop();
        assert!(decode_block_full_root_response(&requeue_response).is_none());

        let compact = decode_block_full_root_response(&[
            BLOCK_FULL_ROOT_PROTOCOL_VERSION,
            BlockFullRootStatus::CompactionRequired as u32,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
        ])
        .unwrap();
        assert_eq!(compact.status, BlockFullRootStatus::CompactionRequired);
        assert!(compact.handoff.is_none() && compact.events.is_empty());

        let proved = decode_block_full_root_response(&[
            BLOCK_FULL_ROOT_PROTOCOL_VERSION,
            BlockFullRootStatus::Proved as u32,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
        ])
        .unwrap();
        assert_eq!(proved.status, BlockFullRootStatus::Proved);
        assert!(proved.handoff.is_none() && proved.events.is_empty());

        let cpu_mic = decode_block_full_root_response(&[
            BLOCK_FULL_ROOT_PROTOCOL_VERSION,
            BlockFullRootStatus::CpuMic as u32,
            1,
            0,
            1,
            0,
            0,
            0,
            (BLOCK_FULL_ROOT_WORK_WORDS + 3) as u32,
            1,
            3,
            7,
            0,
            11,
            0x44,
            0,
            5,
            2,
            0,
            2,
        ])
        .unwrap();
        assert_eq!(cpu_mic.status, BlockFullRootStatus::CpuMic);
        assert_eq!(cpu_mic.handoff.unwrap().user_tag(), 0x44);
        assert_eq!(cpu_mic.handoff_core.as_deref(), Some(&[0, 2][..]));
        assert!(cpu_mic.events.is_empty());
    }
}
