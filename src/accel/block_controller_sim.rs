//! Live, simulation-only mirror of rIC3's resident BLOCK proof state.
//!
//! CPU IC3 remains authoritative. The mirror translates the algorithm-owned
//! semantic journal into the same packed commands intended for the persistent
//! FPGA controller, then checks the complete obligation/lemma multiset after
//! every step. It is deliberately transport-backed rather than a second Rust
//! model, so the native server exercises the synthesizable C++ state machine.

use super::{
    cdcl::{
        BLOCK_SEMANTIC_COMPOSE_OBLIGATION, BLOCK_SEMANTIC_EVENT_CLEAR_OBLIGATIONS,
        BLOCK_SEMANTIC_EVENT_INSERT_LEMMA, BLOCK_SEMANTIC_EVENT_MOVE_LEMMA,
        BLOCK_SEMANTIC_EVENT_REMOVE_LEMMA, BLOCK_SEMANTIC_EVENT_REMOVE_OBLIGATION,
        BLOCK_SEMANTIC_EVENT_SET_LEMMA_FRAMES, BLOCK_SEMANTIC_INSERT_LEMMA,
        BLOCK_SEMANTIC_INSERT_OBLIGATION_TAGGED, BLOCK_SEMANTIC_POP_OBLIGATION,
        BLOCK_SEMANTIC_REGISTER_LEMMA, BLOCK_SEMANTIC_REGISTER_STATE_FULL, BLOCK_SEMANTIC_RESET,
        BLOCK_SEMANTIC_SET_LEMMA_FRAMES, BLOCK_SEMANTIC_STATS, BlockFullRootEvent,
        BlockFullRootResponse, BlockRootExecutionStatus, BlockRootResponse, BlockSemanticCommand,
        BlockSemanticCommandResponse,
    },
    cdcl_host::HardwareCdcl,
};
use crate::gipsat::IncrementalQuery;
use std::{
    collections::{HashMap, HashSet},
    sync::atomic::{AtomicU64, Ordering},
    sync::{Mutex, OnceLock},
    time::Instant,
};

// Journal-local command numbers emitted by ic3/block.rs and ic3/frame.rs.
const JOURNAL_REMOVE_OBLIGATION: u32 = 0;
const JOURNAL_INSERT_OBLIGATION: u32 = 1;
const JOURNAL_CLEAR_OBLIGATIONS: u32 = 2;
const JOURNAL_REMOVE_LEMMA: u32 = 3;
const JOURNAL_INSERT_LEMMA: u32 = 4;
const JOURNAL_SET_LEMMA_FRAMES: u32 = 5;
const JOURNAL_MOVE_LEMMA: u32 = 6;
const JOURNAL_POP_OBLIGATION: u32 = 7;

static MIRROR: OnceLock<Mutex<ResidentBlockMirror>> = OnceLock::new();
static BATCHES: AtomicU64 = AtomicU64::new(0);
static COMMANDS: AtomicU64 = AtomicU64::new(0);
static REBASES: AtomicU64 = AtomicU64::new(0);
static CAPACITY_COMPACTIONS: AtomicU64 = AtomicU64::new(0);
static ROOT_RECONCILES: AtomicU64 = AtomicU64::new(0);
static STEPS: AtomicU64 = AtomicU64::new(0);
static SERVICE_NS: AtomicU64 = AtomicU64::new(0);
static MAX_BATCH: AtomicU64 = AtomicU64::new(0);
static MAX_OBLIGATIONS: AtomicU64 = AtomicU64::new(0);
static MAX_LEMMAS: AtomicU64 = AtomicU64::new(0);
static MAX_OBLIGATION_ARENA: AtomicU64 = AtomicU64::new(0);
static MAX_LEMMA_ARENA: AtomicU64 = AtomicU64::new(0);
static MAX_STATE_ARENA: AtomicU64 = AtomicU64::new(0);
static MAX_INPUT_ARENA: AtomicU64 = AtomicU64::new(0);
static QUEUE_POPS: AtomicU64 = AtomicU64::new(0);
static OWNED_QUEUE_POPS: AtomicU64 = AtomicU64::new(0);
static ROOT_WAVES: AtomicU64 = AtomicU64::new(0);
static ROOT_WORK: AtomicU64 = AtomicU64::new(0);
static ROOT_OK: AtomicU64 = AtomicU64::new(0);
static ROOT_CPU_HANDOFFS: AtomicU64 = AtomicU64::new(0);
static ROOT_SERVICE_NS: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug)]
struct Obligation {
    frame: u32,
    depth: u32,
    removed: u32,
    payload: Vec<u32>,
}

impl Obligation {
    fn key(&self) -> Vec<u32> {
        let mut key = vec![self.frame, self.depth, self.removed];
        key.extend_from_slice(&self.payload);
        key
    }
}

#[derive(Clone, Debug)]
struct Lemma {
    frame: u32,
    payload: Vec<u32>,
}

impl Lemma {
    fn key(&self) -> Vec<u32> {
        let mut key = vec![self.frame];
        key.extend_from_slice(&self.payload);
        key
    }
}

enum JournalOperation {
    RemoveObligation(Obligation),
    InsertObligation(Obligation),
    ClearObligations,
    RemoveLemma(Lemma),
    InsertLemma(Lemma),
    SetLemmaFrames(u32),
    MoveLemma {
        source: Lemma,
        destination: u32,
    },
    PopObligation {
        max_frame: u32,
        expected: Obligation,
    },
}

enum PendingAction {
    None,
    InsertObligation { key: Vec<u32>, user_tag: u64 },
    InsertLemma(Vec<u32>),
    MoveLemma { key: Vec<u32>, handle: u32 },
}

#[derive(Clone, Copy, Debug)]
struct ResidentObligationDescriptor {
    handle: u32,
    user_tag: u64,
}

#[derive(Clone, Debug)]
struct PendingOwnedPop {
    max_frame: u32,
    key: Vec<u32>,
    response: BlockSemanticCommandResponse,
}

#[derive(Default)]
struct ResidentBlockMirror {
    initialized: bool,
    obligation_image: Vec<u32>,
    lemma_image: Vec<u32>,
    obligation_payloads: HashMap<Vec<u32>, u32>,
    obligation_state_payloads: HashMap<Vec<u32>, u32>,
    lemma_payloads: HashMap<Vec<u32>, u32>,
    obligation_descriptors: HashMap<Vec<u32>, Vec<ResidentObligationDescriptor>>,
    lemma_descriptors: HashMap<Vec<u32>, Vec<u32>>,
    next_user_tag: u64,
    pending_owned_pop: Option<PendingOwnedPop>,
    owned_selection_keys: HashMap<u64, Vec<u32>>,
}

pub(super) struct OwnedBlockRootWave {
    pub response: BlockRootResponse,
    pub keys: Vec<Vec<u32>>,
}

pub(super) struct OwnedBlockFullRootWave {
    pub response: BlockFullRootResponse,
    /// CPU proof-obligation keys consumed by each ordered resident event.
    pub source_keys: Vec<Vec<u32>>,
}

fn take(words: &[u32], at: &mut usize) -> Result<u32, String> {
    let word = *words
        .get(*at)
        .ok_or_else(|| "truncated image".to_string())?;
    *at += 1;
    Ok(word)
}

fn take_payload(words: &[u32], at: &mut usize, count: usize) -> Result<Vec<u32>, String> {
    let end = at
        .checked_add(count)
        .ok_or_else(|| "image extent overflow".to_string())?;
    let payload = words
        .get(*at..end)
        .ok_or_else(|| "truncated payload".to_string())?
        .to_vec();
    *at = end;
    Ok(payload)
}

fn decode_obligations(image: &[u32]) -> Result<Vec<Obligation>, String> {
    let mut at = 0usize;
    let count = take(image, &mut at)?;
    let mut obligations = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let frame = take(image, &mut at)?;
        let depth = take(image, &mut at)?;
        let removed = take(image, &mut at)?;
        if removed > 1 {
            return Err("invalid obligation removed flag".to_string());
        }
        let payload_begin = at;
        let state_words = take(image, &mut at)? as usize;
        let _ = take_payload(image, &mut at, state_words)?;
        let input_count = take(image, &mut at)?;
        for _ in 0..input_count {
            let input_words = take(image, &mut at)? as usize;
            let _ = take_payload(image, &mut at, input_words)?;
        }
        obligations.push(Obligation {
            frame,
            depth,
            removed,
            payload: image[payload_begin..at].to_vec(),
        });
    }
    if at != image.len() {
        return Err("trailing obligation image words".to_string());
    }
    Ok(obligations)
}

fn decode_lemmas(image: &[u32]) -> Result<(u32, Vec<Lemma>), String> {
    let mut at = 0usize;
    let frame_count = take(image, &mut at)?;
    let mut lemmas = Vec::new();
    for section in 0..=frame_count {
        let count = take(image, &mut at)?;
        let frame = if section == frame_count {
            u32::MAX
        } else {
            section
        };
        for _ in 0..count {
            let payload_begin = at;
            let literal_count = take(image, &mut at)? as usize;
            let _ = take_payload(image, &mut at, literal_count)?;
            lemmas.push(Lemma {
                frame,
                payload: image[payload_begin..at].to_vec(),
            });
        }
    }
    if at != image.len() {
        return Err("trailing lemma image words".to_string());
    }
    Ok((frame_count, lemmas))
}

fn decode_operation(words: &[u32]) -> Result<JournalOperation, String> {
    let mut at = 0usize;
    let command = take(words, &mut at)?;
    let operation = match command {
        JOURNAL_CLEAR_OBLIGATIONS => JournalOperation::ClearObligations,
        JOURNAL_REMOVE_OBLIGATION | JOURNAL_INSERT_OBLIGATION => {
            let frame = take(words, &mut at)?;
            let depth = take(words, &mut at)?;
            let removed = take(words, &mut at)?;
            let payload_words = take(words, &mut at)? as usize;
            let obligation = Obligation {
                frame,
                depth,
                removed,
                payload: take_payload(words, &mut at, payload_words)?,
            };
            if command == JOURNAL_REMOVE_OBLIGATION {
                JournalOperation::RemoveObligation(obligation)
            } else {
                JournalOperation::InsertObligation(obligation)
            }
        }
        JOURNAL_REMOVE_LEMMA | JOURNAL_INSERT_LEMMA => {
            let frame = take(words, &mut at)?;
            let payload_words = take(words, &mut at)? as usize;
            let lemma = Lemma {
                frame,
                payload: take_payload(words, &mut at, payload_words)?,
            };
            if command == JOURNAL_REMOVE_LEMMA {
                JournalOperation::RemoveLemma(lemma)
            } else {
                JournalOperation::InsertLemma(lemma)
            }
        }
        JOURNAL_SET_LEMMA_FRAMES => JournalOperation::SetLemmaFrames(take(words, &mut at)?),
        JOURNAL_MOVE_LEMMA => {
            let frame = take(words, &mut at)?;
            let destination = take(words, &mut at)?;
            let payload_words = take(words, &mut at)? as usize;
            JournalOperation::MoveLemma {
                source: Lemma {
                    frame,
                    payload: take_payload(words, &mut at, payload_words)?,
                },
                destination,
            }
        }
        JOURNAL_POP_OBLIGATION => {
            let max_frame = take(words, &mut at)?;
            let frame = take(words, &mut at)?;
            let depth = take(words, &mut at)?;
            let removed = take(words, &mut at)?;
            let payload_words = take(words, &mut at)? as usize;
            JournalOperation::PopObligation {
                max_frame,
                expected: Obligation {
                    frame,
                    depth,
                    removed,
                    payload: take_payload(words, &mut at, payload_words)?,
                },
            }
        }
        _ => return Err(format!("unknown BLOCK journal operation {command}")),
    };
    if at != words.len() {
        return Err("trailing BLOCK journal operation words".to_string());
    }
    Ok(operation)
}

fn obligation_state(payload: &[u32]) -> Result<Vec<u32>, String> {
    let state_words = payload
        .first()
        .copied()
        .ok_or_else(|| "empty obligation payload".to_string())? as usize;
    payload
        .get(1..1usize.saturating_add(state_words))
        .map(<[u32]>::to_vec)
        .ok_or_else(|| "truncated obligation state cube".to_string())
}

fn command(command: u32) -> BlockSemanticCommand {
    BlockSemanticCommand::new(command)
}

fn command_batch_limit() -> usize {
    static LIMIT: OnceLock<usize> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_BLOCK_CONTROLLER_BATCH")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(256)
            .clamp(1, 1024)
    })
}

fn issue(
    hardware: &mut HardwareCdcl,
    commands: &[BlockSemanticCommand],
) -> Result<Vec<BlockSemanticCommandResponse>, String> {
    if commands.is_empty() {
        return Ok(Vec::new());
    }
    let mut combined = Vec::with_capacity(commands.len());
    for chunk in commands.chunks(command_batch_limit()) {
        let started = Instant::now();
        let response = hardware
            .run_block_semantic_batch(chunk)
            .map_err(|error| error.to_string())?;
        let elapsed_ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        SERVICE_NS.fetch_add(elapsed_ns, Ordering::Relaxed);
        BATCHES.fetch_add(1, Ordering::Relaxed);
        COMMANDS.fetch_add(chunk.len() as u64, Ordering::Relaxed);
        MAX_BATCH.fetch_max(chunk.len() as u64, Ordering::Relaxed);
        for record in &response {
            MAX_OBLIGATIONS.fetch_max(record.obligation_count as u64, Ordering::Relaxed);
            MAX_LEMMAS.fetch_max(record.lemma_count as u64, Ordering::Relaxed);
            MAX_OBLIGATION_ARENA.fetch_max(record.obligation_arena_words as u64, Ordering::Relaxed);
            MAX_LEMMA_ARENA.fetch_max(record.lemma_arena_words as u64, Ordering::Relaxed);
            MAX_STATE_ARENA.fetch_max(record.state_arena_words as u64, Ordering::Relaxed);
            MAX_INPUT_ARENA.fetch_max(record.input_arena_words as u64, Ordering::Relaxed);
        }
        combined.extend(response);
    }
    Ok(combined)
}

fn semantic_capacity_error(error: &str) -> bool {
    // HardwareError::BlockSemantic preserves the command's ordinary semantic
    // status in its Debug/Display representation. Status 2 is the fixed-arena
    // capacity result; batch/transport errors must remain fatal so a reset
    // cannot hide malformed journal traffic or a broken RPC boundary.
    error.contains("BlockSemantic") && error.contains("command_status: 2")
}

fn multiset<T>(items: &[T], key: impl Fn(&T) -> Vec<u32>) -> HashMap<Vec<u32>, usize> {
    let mut counts = HashMap::new();
    for item in items {
        *counts.entry(key(item)).or_insert(0) += 1;
    }
    counts
}

impl ResidentBlockMirror {
    fn allocate_user_tag(&mut self) -> u64 {
        self.next_user_tag = self.next_user_tag.wrapping_add(1).max(1);
        self.next_user_tag
    }

    fn remove_obligation_descriptor_for_tag(
        &mut self,
        user_tag: u64,
    ) -> Result<(Vec<u32>, ResidentObligationDescriptor), String> {
        let selected = self
            .obligation_descriptors
            .iter()
            .find_map(|(key, descriptors)| {
                descriptors
                    .iter()
                    .position(|descriptor| descriptor.user_tag == user_tag)
                    .map(|position| (key.clone(), position))
            })
            .ok_or_else(|| format!("resident full root returned unknown source tag {user_tag}"))?;
        let (key, position) = selected;
        let descriptors = self
            .obligation_descriptors
            .get_mut(&key)
            .ok_or_else(|| "resident full-root source disappeared".to_string())?;
        let descriptor = descriptors.swap_remove(position);
        if descriptors.is_empty() {
            self.obligation_descriptors.remove(&key);
        }
        Ok((key, descriptor))
    }

    fn tagged_obligation_insert(
        &mut self,
        obligation: &Obligation,
        payload_handle: u32,
    ) -> (BlockSemanticCommand, u64) {
        let user_tag = self.allocate_user_tag();
        let mut insert = command(BLOCK_SEMANTIC_INSERT_OBLIGATION_TAGGED);
        insert.frame = obligation.frame;
        insert.depth = obligation.depth;
        insert.removed = obligation.removed;
        insert.handle = payload_handle;
        insert.payload = vec![user_tag as u32, (user_tag >> 32) as u32];
        (insert, user_tag)
    }

    fn pop_owned(
        &mut self,
        hardware: &mut HardwareCdcl,
        max_frame: u32,
    ) -> Result<Option<u64>, String> {
        if !self.initialized {
            return Err("resident BLOCK mirror was not rebased".to_string());
        }
        if self.pending_owned_pop.is_some() {
            return Err("resident queue has an unconsumed owned pop".to_string());
        }
        let mut pop = command(BLOCK_SEMANTIC_POP_OBLIGATION);
        pop.frame = max_frame;
        let response = issue(hardware, &[pop])?
            .into_iter()
            .next()
            .ok_or_else(|| "missing resident queue pop response".to_string())?;
        if response.found == 0 {
            return Ok(None);
        }
        let user_tag = response.popped_user_tag();
        let selected = self
            .obligation_descriptors
            .iter()
            .find_map(|(key, descriptors)| {
                descriptors
                    .iter()
                    .position(|descriptor| {
                        descriptor.handle == response.output_handle
                            && descriptor.user_tag == user_tag
                    })
                    .map(|position| (key.clone(), position))
            })
            .ok_or_else(|| {
                format!(
                    "resident queue returned unknown handle {} tag {}",
                    response.output_handle, user_tag,
                )
            })?;
        let (key, position) = selected;
        let descriptors = self
            .obligation_descriptors
            .get_mut(&key)
            .ok_or_else(|| "resident queue selection disappeared".to_string())?;
        descriptors.swap_remove(position);
        if descriptors.is_empty() {
            self.obligation_descriptors.remove(&key);
        }
        self.pending_owned_pop = Some(PendingOwnedPop {
            max_frame,
            key: key.clone(),
            response,
        });
        if self.owned_selection_keys.insert(user_tag, key).is_some() {
            return Err(format!("duplicate resident queue selection tag {user_tag}"));
        }
        QUEUE_POPS.fetch_add(1, Ordering::Relaxed);
        OWNED_QUEUE_POPS.fetch_add(1, Ordering::Relaxed);
        Ok(Some(user_tag))
    }

    fn run_owned_root(
        &mut self,
        hardware: &mut HardwareCdcl,
        max_frame: u32,
        requested_queries: usize,
        next_var_by_current: &[u32],
        query_template: &IncrementalQuery,
    ) -> Result<OwnedBlockRootWave, String> {
        if !self.initialized {
            return Err("resident BLOCK mirror was not rebased".to_string());
        }
        if self.pending_owned_pop.is_some() {
            return Err("resident queue has an unconsumed owned pop".to_string());
        }
        let started = Instant::now();
        let response = hardware
            .run_block_root(
                max_frame,
                requested_queries,
                next_var_by_current,
                query_template,
            )
            .map_err(|error| format!("resident BLOCK root command failed: {error}"))?;
        let elapsed_ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        SERVICE_NS.fetch_add(elapsed_ns, Ordering::Relaxed);
        ROOT_SERVICE_NS.fetch_add(elapsed_ns, Ordering::Relaxed);
        let mut selections = Vec::with_capacity(response.work.len());
        for work in &response.work {
            let user_tag = work.user_tag();
            let selected = self
                .obligation_descriptors
                .iter()
                .find_map(|(key, descriptors)| {
                    descriptors
                        .iter()
                        .position(|descriptor| {
                            descriptor.handle == work.descriptor_handle
                                && descriptor.user_tag == user_tag
                        })
                        .map(|position| (key.clone(), position))
                })
                .ok_or_else(|| {
                    format!(
                        "resident root returned unknown handle {} tag {}",
                        work.descriptor_handle, user_tag,
                    )
                })?;
            selections.push((selected.0, selected.1, user_tag));
        }
        if matches!(
            response.status,
            BlockRootExecutionStatus::Ok | BlockRootExecutionStatus::CpuHandoff
        ) {
            let (key, position, user_tag) = selections
                .first()
                .cloned()
                .ok_or_else(|| "resident root committed without work".to_string())?;
            let descriptors = self
                .obligation_descriptors
                .get_mut(&key)
                .ok_or_else(|| "resident root selection disappeared".to_string())?;
            descriptors.swap_remove(position);
            if descriptors.is_empty() {
                self.obligation_descriptors.remove(&key);
            }
            let work = response.work[0];
            let (lemma_frame_count, _) = decode_lemmas(&self.lemma_image)?;
            let semantic = BlockSemanticCommandResponse {
                found: 1,
                obligation_count: self
                    .obligation_descriptors
                    .values()
                    .map(Vec::len)
                    .sum::<usize>()
                    .min(u32::MAX as usize) as u32,
                lemma_count: self
                    .lemma_descriptors
                    .values()
                    .map(Vec::len)
                    .sum::<usize>()
                    .min(u32::MAX as usize) as u32,
                output_handle: work.descriptor_handle,
                popped_frame: work.frame,
                popped_depth: work.depth,
                popped_removed: work.removed,
                lemma_frame_count,
                popped_user_tag_lo: work.user_tag_lo,
                popped_user_tag_hi: work.user_tag_hi,
                ..BlockSemanticCommandResponse::default()
            };
            self.pending_owned_pop = Some(PendingOwnedPop {
                max_frame,
                key: key.clone(),
                response: semantic,
            });
            if self.owned_selection_keys.insert(user_tag, key).is_some() {
                return Err(format!("duplicate resident root selection tag {user_tag}"));
            }
            QUEUE_POPS.fetch_add(1, Ordering::Relaxed);
            OWNED_QUEUE_POPS.fetch_add(1, Ordering::Relaxed);
        }
        ROOT_WAVES.fetch_add(1, Ordering::Relaxed);
        ROOT_WORK.fetch_add(response.work.len() as u64, Ordering::Relaxed);
        match response.status {
            BlockRootExecutionStatus::Ok => {
                ROOT_OK.fetch_add(1, Ordering::Relaxed);
            }
            BlockRootExecutionStatus::CpuHandoff => {
                ROOT_CPU_HANDOFFS.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
        let keys = selections.into_iter().map(|(key, _, _)| key).collect();
        Ok(OwnedBlockRootWave { response, keys })
    }

    fn run_owned_full_root(
        &mut self,
        hardware: &mut HardwareCdcl,
        max_frame: u32,
        step_limit: usize,
        next_var_by_current: &[u32],
        init_value_by_current: &[u32],
        latch_variables: &[u32],
        input_variables: &[u32],
        query_template: &IncrementalQuery,
    ) -> Result<OwnedBlockFullRootWave, String> {
        if !self.initialized {
            return Err("resident BLOCK mirror was not rebased".to_string());
        }
        if self.pending_owned_pop.is_some() {
            return Err("resident queue has an unconsumed owned pop".to_string());
        }
        let started = Instant::now();
        let response = hardware
            .run_block_full_root(
                max_frame,
                step_limit,
                next_var_by_current,
                init_value_by_current,
                latch_variables,
                input_variables,
                query_template,
            )
            .map_err(|error| format!("resident BLOCK full-root command failed: {error}"))?;
        let elapsed_ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        SERVICE_NS.fetch_add(elapsed_ns, Ordering::Relaxed);
        ROOT_SERVICE_NS.fetch_add(elapsed_ns, Ordering::Relaxed);

        let mut source_keys = Vec::with_capacity(response.events.len());
        for event in &response.events {
            match event {
                BlockFullRootEvent::SatPredecessor {
                    child_tag,
                    parent_tag,
                    parent_descriptor_handle,
                    child_descriptor_handle,
                    child_payload_handle,
                    frame,
                    depth,
                    state,
                    input,
                } => {
                    let (source_key, source_descriptor) =
                        self.remove_obligation_descriptor_for_tag(*parent_tag)?;
                    if source_descriptor.user_tag != *parent_tag {
                        return Err("resident SAT journal source tag mismatch".to_string());
                    }
                    source_keys.push(source_key.clone());
                    self.obligation_descriptors
                        .entry(source_key)
                        .or_default()
                        .push(ResidentObligationDescriptor {
                            handle: *parent_descriptor_handle,
                            user_tag: *parent_tag,
                        });

                    let mut payload = Vec::with_capacity(state.len() + input.len() + 3);
                    payload.push(state.len().min(u32::MAX as usize) as u32);
                    payload.extend_from_slice(state);
                    payload.push(1);
                    payload.push(input.len().min(u32::MAX as usize) as u32);
                    payload.extend_from_slice(input);
                    let mut child_key = vec![*frame, *depth, 0];
                    child_key.extend_from_slice(&payload);
                    self.obligation_payloads
                        .insert(payload.clone(), *child_payload_handle);
                    self.obligation_state_payloads
                        .insert(state.clone(), *child_payload_handle);
                    self.obligation_descriptors
                        .entry(child_key)
                        .or_default()
                        .push(ResidentObligationDescriptor {
                            handle: *child_descriptor_handle,
                            user_tag: *child_tag,
                        });
                }
                BlockFullRootEvent::UnsatLemma {
                    frame,
                    proof_tag,
                    payload_handle,
                    descriptor_handle,
                    cube,
                } => {
                    let (source_key, _) = self.remove_obligation_descriptor_for_tag(*proof_tag)?;
                    source_keys.push(source_key);
                    let mut payload = Vec::with_capacity(cube.len() + 1);
                    payload.push(cube.len().min(u32::MAX as usize) as u32);
                    payload.extend_from_slice(cube);
                    let mut key = vec![*frame];
                    key.extend_from_slice(&payload);
                    self.lemma_payloads.insert(payload, *payload_handle);
                    self.lemma_descriptors
                        .entry(key)
                        .or_default()
                        .push(*descriptor_handle);
                }
            }
        }
        if let Some(handoff) = response.handoff {
            let user_tag = handoff.user_tag();
            let (key, descriptor) = self.remove_obligation_descriptor_for_tag(user_tag)?;
            if descriptor.handle != handoff.descriptor_handle {
                return Err("resident full-root handoff descriptor mismatch".to_string());
            }
            if self.owned_selection_keys.insert(user_tag, key).is_some() {
                return Err(format!(
                    "duplicate resident full-root handoff tag {user_tag}"
                ));
            }
        }
        ROOT_WAVES.fetch_add(1, Ordering::Relaxed);
        ROOT_WORK.fetch_add(response.cdcl_waves as u64, Ordering::Relaxed);
        ROOT_OK.fetch_add(1, Ordering::Relaxed);
        Ok(OwnedBlockFullRootWave {
            response,
            source_keys,
        })
    }

    fn take_owned_key(&mut self, user_tag: u64) -> Result<Vec<u32>, String> {
        self.owned_selection_keys
            .remove(&user_tag)
            .ok_or_else(|| format!("unknown resident queue selection tag {user_tag}"))
    }

    fn register_obligation_payloads(
        &mut self,
        hardware: &mut HardwareCdcl,
        payloads: impl IntoIterator<Item = Vec<u32>>,
    ) -> Result<(), String> {
        let mut full_to_state = Vec::new();
        let mut new_states = Vec::new();
        let mut pending_states = HashSet::new();
        for payload in payloads {
            if self.obligation_payloads.contains_key(&payload) {
                continue;
            }
            let state = obligation_state(&payload)?;
            if !self.obligation_state_payloads.contains_key(&state)
                && pending_states.insert(state.clone())
            {
                new_states.push(state.clone());
            }
            full_to_state.push((payload, state));
        }
        if !new_states.is_empty() {
            let registrations: Vec<_> = new_states
                .iter()
                .map(|state| {
                    let mut register = command(BLOCK_SEMANTIC_REGISTER_STATE_FULL);
                    register.payload = state.clone();
                    register
                })
                .collect();
            let state_records = issue(hardware, &registrations)?;
            let compositions: Vec<_> = state_records
                .iter()
                .map(|record| {
                    let mut compose = command(BLOCK_SEMANTIC_COMPOSE_OBLIGATION);
                    compose.handle = record.output_handle;
                    compose
                })
                .collect();
            let payload_records = issue(hardware, &compositions)?;
            for (state, record) in new_states.into_iter().zip(payload_records) {
                self.obligation_state_payloads
                    .insert(state, record.output_handle);
            }
        }
        for (payload, state) in full_to_state {
            let handle = *self
                .obligation_state_payloads
                .get(&state)
                .ok_or_else(|| "missing state-only obligation payload".to_string())?;
            self.obligation_payloads.insert(payload, handle);
        }
        Ok(())
    }

    fn validate(
        &self,
        obligation_image: &[u32],
        lemma_image: &[u32],
        response: Option<&BlockSemanticCommandResponse>,
    ) -> Result<(), String> {
        let obligations = decode_obligations(obligation_image)?;
        let (frame_count, lemmas) = decode_lemmas(lemma_image)?;
        let expected_obligations = multiset(&obligations, Obligation::key);
        let expected_lemmas = multiset(&lemmas, Lemma::key);
        let actual_obligations: HashMap<_, _> = self
            .obligation_descriptors
            .iter()
            .map(|(key, handles)| (key.clone(), handles.len()))
            .collect();
        let actual_lemmas: HashMap<_, _> = self
            .lemma_descriptors
            .iter()
            .map(|(key, handles)| (key.clone(), handles.len()))
            .collect();
        if expected_obligations != actual_obligations || expected_lemmas != actual_lemmas {
            let obligation_difference = expected_obligations
                .iter()
                .find(|(key, expected)| actual_obligations.get(*key) != Some(*expected))
                .map(|(key, expected)| {
                    format!(
                        "expected obligation {:?} x{}, actual {}",
                        key,
                        expected,
                        actual_obligations.get(key).copied().unwrap_or(0)
                    )
                })
                .or_else(|| {
                    actual_obligations
                        .iter()
                        .find(|(key, actual)| expected_obligations.get(*key) != Some(*actual))
                        .map(|(key, actual)| {
                            format!(
                                "extra obligation {:?} x{}, expected {}",
                                key,
                                actual,
                                expected_obligations.get(key).copied().unwrap_or(0)
                            )
                        })
                })
                .unwrap_or_else(|| "obligations match".to_string());
            return Err(format!(
                "resident BLOCK multiset mismatch obligations {}/{} lemmas {}/{}; {}",
                actual_obligations.values().sum::<usize>(),
                obligations.len(),
                actual_lemmas.values().sum::<usize>(),
                lemmas.len(),
                obligation_difference,
            ));
        }
        if let Some(response) = response
            && (response.obligation_count as usize != obligations.len()
                || response.lemma_count as usize != lemmas.len()
                || response.lemma_frame_count != frame_count)
        {
            return Err(format!(
                "resident BLOCK counter mismatch obligations {}/{} lemmas {}/{} frames {}/{}",
                response.obligation_count,
                obligations.len(),
                response.lemma_count,
                lemmas.len(),
                response.lemma_frame_count,
                frame_count,
            ));
        }
        Ok(())
    }

    fn rebase(
        &mut self,
        hardware: &mut HardwareCdcl,
        obligation_image: &[u32],
        lemma_image: &[u32],
    ) -> Result<(), String> {
        let obligations = decode_obligations(obligation_image)?;
        let (frame_count, lemmas) = decode_lemmas(lemma_image)?;
        self.obligation_payloads.clear();
        self.obligation_state_payloads.clear();
        self.lemma_payloads.clear();
        self.obligation_descriptors.clear();
        self.lemma_descriptors.clear();
        self.pending_owned_pop = None;
        self.owned_selection_keys.clear();

        let mut registration = vec![command(BLOCK_SEMANTIC_RESET)];
        let mut set_frames = command(BLOCK_SEMANTIC_SET_LEMMA_FRAMES);
        set_frames.frame = frame_count;
        registration.push(set_frames);
        let _ = issue(hardware, &registration)?;
        self.register_obligation_payloads(
            hardware,
            obligations
                .iter()
                .map(|obligation| obligation.payload.clone()),
        )?;
        let mut seen = HashSet::new();
        let mut lemma_keys = Vec::new();
        for lemma in &lemmas {
            if seen.insert(lemma.payload.clone()) {
                let mut register = command(BLOCK_SEMANTIC_REGISTER_LEMMA);
                register.payload = lemma.payload.clone();
                registration.push(register);
                lemma_keys.push(lemma.payload.clone());
            }
        }
        // RESET/SET were already issued; reuse the vector allocation for the
        // independent lemma registrations.
        registration.drain(..2);
        let response = issue(hardware, &registration)?;
        for (key, record) in lemma_keys.into_iter().zip(response.iter()) {
            self.lemma_payloads.insert(key, record.output_handle);
        }

        let mut insertion = Vec::with_capacity(obligations.len() + lemmas.len() + 1);
        let mut obligation_descriptor_keys = Vec::new();
        for obligation in &obligations {
            let payload_handle = *self
                .obligation_payloads
                .get(&obligation.payload)
                .ok_or_else(|| "missing registered obligation payload".to_string())?;
            let (insert, user_tag) = self.tagged_obligation_insert(obligation, payload_handle);
            insertion.push(insert);
            obligation_descriptor_keys.push((obligation.key(), user_tag));
        }
        let obligation_insertion_end = insertion.len();
        let mut lemma_descriptor_keys = Vec::new();
        for lemma in &lemmas {
            let mut insert = command(BLOCK_SEMANTIC_INSERT_LEMMA);
            insert.frame = lemma.frame;
            insert.handle = *self
                .lemma_payloads
                .get(&lemma.payload)
                .ok_or_else(|| "missing registered lemma payload".to_string())?;
            insertion.push(insert);
            lemma_descriptor_keys.push(lemma.key());
        }
        insertion.push(command(BLOCK_SEMANTIC_STATS));
        let response = issue(hardware, &insertion)?;
        for ((key, user_tag), record) in obligation_descriptor_keys
            .into_iter()
            .zip(response[..obligation_insertion_end].iter())
        {
            self.obligation_descriptors.entry(key).or_default().push(
                ResidentObligationDescriptor {
                    handle: record.output_handle,
                    user_tag,
                },
            );
        }
        for (key, record) in lemma_descriptor_keys
            .into_iter()
            .zip(response[obligation_insertion_end..response.len() - 1].iter())
        {
            self.lemma_descriptors
                .entry(key)
                .or_default()
                .push(record.output_handle);
        }
        self.validate(obligation_image, lemma_image, response.last())?;
        self.initialized = true;
        self.obligation_image = obligation_image.to_vec();
        self.lemma_image = lemma_image.to_vec();
        REBASES.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn reconcile(
        &mut self,
        hardware: &mut HardwareCdcl,
        obligation_image: &[u32],
        lemma_image: &[u32],
    ) -> Result<(), String> {
        if self.initialized
            && self.obligation_image == obligation_image
            && self.lemma_image == lemma_image
        {
            return Ok(());
        }
        if !self.initialized {
            return self.rebase(hardware, obligation_image, lemma_image);
        }

        // Mutations between algorithm-owned BLOCK scopes (for example adding
        // the next bad-state obligation) are reconciled as ordinary resident
        // remove/insert events. Never reset a healthy payload epoch merely
        // because the CPU crossed a macro boundary.
        let expected_obligations = decode_obligations(obligation_image)?;
        let (frame_count, expected_lemmas) = decode_lemmas(lemma_image)?;
        let expected_obligation_counts = multiset(&expected_obligations, Obligation::key);
        let expected_lemma_counts = multiset(&expected_lemmas, Lemma::key);
        let mut operations = Vec::new();

        for (key, handles) in &self.obligation_descriptors {
            let expected = expected_obligation_counts.get(key).copied().unwrap_or(0);
            for _ in expected..handles.len() {
                operations.push(JournalOperation::RemoveObligation(Obligation {
                    frame: key[0],
                    depth: key[1],
                    removed: key[2],
                    payload: key[3..].to_vec(),
                }));
            }
        }
        for (key, handles) in &self.lemma_descriptors {
            let expected = expected_lemma_counts.get(key).copied().unwrap_or(0);
            for _ in expected..handles.len() {
                operations.push(JournalOperation::RemoveLemma(Lemma {
                    frame: key[0],
                    payload: key[1..].to_vec(),
                }));
            }
        }
        operations.push(JournalOperation::SetLemmaFrames(frame_count));
        let actual_obligation_counts: HashMap<_, _> = self
            .obligation_descriptors
            .iter()
            .map(|(key, handles)| (key.clone(), handles.len()))
            .collect();
        for obligation in expected_obligations {
            let key = obligation.key();
            let actual = actual_obligation_counts.get(&key).copied().unwrap_or(0);
            let already_planned = operations
                .iter()
                .filter(|operation| {
                    matches!(operation, JournalOperation::InsertObligation(item) if item.key() == key)
                })
                .count();
            if actual + already_planned < expected_obligation_counts[&key] {
                operations.push(JournalOperation::InsertObligation(obligation));
            }
        }
        let actual_lemma_counts: HashMap<_, _> = self
            .lemma_descriptors
            .iter()
            .map(|(key, handles)| (key.clone(), handles.len()))
            .collect();
        for lemma in expected_lemmas {
            let key = lemma.key();
            let actual = actual_lemma_counts.get(&key).copied().unwrap_or(0);
            let already_planned = operations
                .iter()
                .filter(|operation| {
                    matches!(operation, JournalOperation::InsertLemma(item) if item.key() == key)
                })
                .count();
            if actual + already_planned < expected_lemma_counts[&key] {
                operations.push(JournalOperation::InsertLemma(lemma));
            }
        }
        self.apply_operations(hardware, operations, obligation_image, lemma_image, false)?;
        ROOT_RECONCILES.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn register_missing_payloads(
        &mut self,
        hardware: &mut HardwareCdcl,
        operations: &[JournalOperation],
    ) -> Result<(), String> {
        let mut commands = Vec::new();
        let mut lemma_keys = Vec::new();
        let mut obligation_payloads = Vec::new();
        let mut pending_lemmas = HashSet::new();
        for operation in operations {
            match operation {
                JournalOperation::InsertObligation(obligation)
                    if !self.obligation_payloads.contains_key(&obligation.payload) =>
                {
                    obligation_payloads.push(obligation.payload.clone());
                }
                JournalOperation::InsertLemma(lemma)
                    if !self.lemma_payloads.contains_key(&lemma.payload)
                        && pending_lemmas.insert(lemma.payload.clone()) =>
                {
                    let mut register = command(BLOCK_SEMANTIC_REGISTER_LEMMA);
                    register.payload = lemma.payload.clone();
                    commands.push(register);
                    lemma_keys.push(lemma.payload.clone());
                }
                _ => {}
            }
        }
        self.register_obligation_payloads(hardware, obligation_payloads)?;
        let response = issue(hardware, &commands)?;
        for (key, record) in lemma_keys.into_iter().zip(response.iter()) {
            self.lemma_payloads.insert(key, record.output_handle);
        }
        Ok(())
    }

    fn flush_pending(
        &mut self,
        hardware: &mut HardwareCdcl,
        commands: &mut Vec<BlockSemanticCommand>,
        actions: &mut Vec<PendingAction>,
        pending_obligations: &mut HashSet<Vec<u32>>,
        pending_lemmas: &mut HashSet<Vec<u32>>,
    ) -> Result<Option<BlockSemanticCommandResponse>, String> {
        if commands.is_empty() {
            return Ok(None);
        }
        let response = issue(hardware, commands)?;
        for (action, record) in actions.drain(..).zip(&response) {
            match action {
                PendingAction::None => {}
                PendingAction::InsertObligation { key, user_tag } => {
                    self.obligation_descriptors.entry(key).or_default().push(
                        ResidentObligationDescriptor {
                            handle: record.output_handle,
                            user_tag,
                        },
                    )
                }
                PendingAction::InsertLemma(key) => self
                    .lemma_descriptors
                    .entry(key)
                    .or_default()
                    .push(record.output_handle),
                PendingAction::MoveLemma { key, handle } => {
                    self.lemma_descriptors.entry(key).or_default().push(handle)
                }
            }
        }
        commands.clear();
        pending_obligations.clear();
        pending_lemmas.clear();
        Ok(response.last().copied())
    }

    fn apply_operations(
        &mut self,
        hardware: &mut HardwareCdcl,
        operations: Vec<JournalOperation>,
        obligation_image: &[u32],
        lemma_image: &[u32],
        count_step: bool,
    ) -> Result<(), String> {
        if !self.initialized {
            return Err("resident BLOCK mirror was not rebased".to_string());
        }
        self.register_missing_payloads(hardware, &operations)?;

        let mut commands = Vec::new();
        let mut actions = Vec::new();
        let mut pending_obligations = HashSet::new();
        let mut pending_lemmas = HashSet::new();
        let mut last_response = None;
        for operation in operations {
            match operation {
                JournalOperation::RemoveObligation(obligation) => {
                    let key = obligation.key();
                    if pending_obligations.contains(&key) {
                        last_response = self.flush_pending(
                            hardware,
                            &mut commands,
                            &mut actions,
                            &mut pending_obligations,
                            &mut pending_lemmas,
                        )?;
                    }
                    let handles = self
                        .obligation_descriptors
                        .get_mut(&key)
                        .ok_or_else(|| "remove of unknown obligation".to_string())?;
                    let descriptor = handles
                        .pop()
                        .ok_or_else(|| "empty obligation descriptor stack".to_string())?;
                    if handles.is_empty() {
                        self.obligation_descriptors.remove(&key);
                    }
                    let mut remove = command(BLOCK_SEMANTIC_EVENT_REMOVE_OBLIGATION);
                    remove.handle = descriptor.handle;
                    commands.push(remove);
                    actions.push(PendingAction::None);
                }
                JournalOperation::InsertObligation(obligation) => {
                    let key = obligation.key();
                    let payload_handle = *self
                        .obligation_payloads
                        .get(&obligation.payload)
                        .ok_or_else(|| "insert uses unknown obligation payload".to_string())?;
                    let (insert, user_tag) =
                        self.tagged_obligation_insert(&obligation, payload_handle);
                    commands.push(insert);
                    actions.push(PendingAction::InsertObligation {
                        key: key.clone(),
                        user_tag,
                    });
                    pending_obligations.insert(key);
                }
                JournalOperation::PopObligation {
                    max_frame,
                    expected,
                } => {
                    let _ = self.flush_pending(
                        hardware,
                        &mut commands,
                        &mut actions,
                        &mut pending_obligations,
                        &mut pending_lemmas,
                    )?;
                    let key = expected.key();
                    if let Some(pending) = self.pending_owned_pop.take() {
                        if pending.max_frame != max_frame || pending.key != key {
                            return Err(format!(
                                "owned resident pop diverged max-frame {}/{} tag {}",
                                pending.max_frame,
                                max_frame,
                                pending.response.popped_user_tag(),
                            ));
                        }
                        last_response = Some(pending.response);
                        continue;
                    }
                    let mut pop = command(BLOCK_SEMANTIC_POP_OBLIGATION);
                    pop.frame = max_frame;
                    let response = issue(hardware, &[pop])?
                        .into_iter()
                        .next()
                        .ok_or_else(|| "missing resident queue pop response".to_string())?;
                    if response.found == 0 {
                        return Err("resident queue missed CPU obligation pop".to_string());
                    }
                    let handles = self.obligation_descriptors.get_mut(&key).ok_or_else(|| {
                        "resident queue popped a different obligation".to_string()
                    })?;
                    let position = handles
                        .iter()
                        .position(|descriptor| {
                            descriptor.handle == response.output_handle
                                && descriptor.user_tag == response.popped_user_tag()
                        })
                        .ok_or_else(|| {
                            format!(
                                "resident queue tag mismatch handle {} tag {}",
                                response.output_handle,
                                response.popped_user_tag(),
                            )
                        })?;
                    handles.swap_remove(position);
                    if handles.is_empty() {
                        self.obligation_descriptors.remove(&key);
                    }
                    QUEUE_POPS.fetch_add(1, Ordering::Relaxed);
                    last_response = Some(response);
                }
                JournalOperation::ClearObligations => {
                    last_response = self.flush_pending(
                        hardware,
                        &mut commands,
                        &mut actions,
                        &mut pending_obligations,
                        &mut pending_lemmas,
                    )?;
                    self.obligation_descriptors.clear();
                    commands.push(command(BLOCK_SEMANTIC_EVENT_CLEAR_OBLIGATIONS));
                    actions.push(PendingAction::None);
                }
                JournalOperation::RemoveLemma(lemma) => {
                    let key = lemma.key();
                    if pending_lemmas.contains(&key) {
                        last_response = self.flush_pending(
                            hardware,
                            &mut commands,
                            &mut actions,
                            &mut pending_obligations,
                            &mut pending_lemmas,
                        )?;
                    }
                    let handles = self
                        .lemma_descriptors
                        .get_mut(&key)
                        .ok_or_else(|| "remove of unknown lemma".to_string())?;
                    let handle = handles
                        .pop()
                        .ok_or_else(|| "empty lemma descriptor stack".to_string())?;
                    if handles.is_empty() {
                        self.lemma_descriptors.remove(&key);
                    }
                    let mut remove = command(BLOCK_SEMANTIC_EVENT_REMOVE_LEMMA);
                    remove.handle = handle;
                    commands.push(remove);
                    actions.push(PendingAction::None);
                }
                JournalOperation::InsertLemma(lemma) => {
                    let key = lemma.key();
                    let mut insert = command(BLOCK_SEMANTIC_EVENT_INSERT_LEMMA);
                    insert.frame = lemma.frame;
                    insert.handle = *self
                        .lemma_payloads
                        .get(&lemma.payload)
                        .ok_or_else(|| "insert uses unknown lemma payload".to_string())?;
                    commands.push(insert);
                    actions.push(PendingAction::InsertLemma(key.clone()));
                    pending_lemmas.insert(key);
                }
                JournalOperation::SetLemmaFrames(frame_count) => {
                    let mut set = command(BLOCK_SEMANTIC_EVENT_SET_LEMMA_FRAMES);
                    set.frame = frame_count;
                    commands.push(set);
                    actions.push(PendingAction::None);
                }
                JournalOperation::MoveLemma {
                    source,
                    destination,
                } => {
                    let source_key = source.key();
                    if pending_lemmas.contains(&source_key) {
                        last_response = self.flush_pending(
                            hardware,
                            &mut commands,
                            &mut actions,
                            &mut pending_obligations,
                            &mut pending_lemmas,
                        )?;
                    }
                    let handles = self
                        .lemma_descriptors
                        .get_mut(&source_key)
                        .ok_or_else(|| "move of unknown lemma".to_string())?;
                    let handle = handles
                        .pop()
                        .ok_or_else(|| "empty moved lemma descriptor stack".to_string())?;
                    if handles.is_empty() {
                        self.lemma_descriptors.remove(&source_key);
                    }
                    let destination_key = Lemma {
                        frame: destination,
                        payload: source.payload,
                    }
                    .key();
                    let mut move_command = command(BLOCK_SEMANTIC_EVENT_MOVE_LEMMA);
                    move_command.frame = destination;
                    move_command.handle = handle;
                    commands.push(move_command);
                    actions.push(PendingAction::MoveLemma {
                        key: destination_key.clone(),
                        handle,
                    });
                    pending_lemmas.insert(destination_key);
                }
            }
        }
        if let Some(response) = self.flush_pending(
            hardware,
            &mut commands,
            &mut actions,
            &mut pending_obligations,
            &mut pending_lemmas,
        )? {
            last_response = Some(response);
        }
        if last_response.is_none() {
            last_response = issue(hardware, &[command(BLOCK_SEMANTIC_STATS)])?
                .last()
                .copied();
        }
        self.validate(obligation_image, lemma_image, last_response.as_ref())?;
        self.obligation_image = obligation_image.to_vec();
        self.lemma_image = lemma_image.to_vec();
        if count_step {
            STEPS.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    fn apply(
        &mut self,
        hardware: &mut HardwareCdcl,
        semantic_ops: &[Vec<u32>],
        obligation_image: &[u32],
        lemma_image: &[u32],
    ) -> Result<(), String> {
        let operations = semantic_ops
            .iter()
            .map(|words| decode_operation(words))
            .collect::<Result<Vec<_>, _>>()?;
        match self.apply_operations(hardware, operations, obligation_image, lemma_image, true) {
            Ok(()) => Ok(()),
            Err(error) if semantic_capacity_error(&error) => {
                // Immutable payload arenas intentionally avoid free-list and
                // lifetime hazards in the hot path. When append space is
                // exhausted, RESET and rebuild only the exact live image: a
                // copying-GC epoch boundary. The failed batch may have
                // committed a prefix, so retrying individual operations is
                // unsafe; a complete rebase is the transactional recovery.
                CAPACITY_COMPACTIONS.fetch_add(1, Ordering::Relaxed);
                self.rebase(hardware, obligation_image, lemma_image)
            }
            Err(error) => Err(error),
        }
    }
}

pub(super) fn reconcile(
    hardware: &mut HardwareCdcl,
    obligation_image: &[u32],
    lemma_image: &[u32],
) -> Result<(), String> {
    MIRROR
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| "resident BLOCK mirror lock poisoned".to_string())?
        .reconcile(hardware, obligation_image, lemma_image)
}

pub(super) fn apply(
    hardware: &mut HardwareCdcl,
    semantic_ops: &[Vec<u32>],
    obligation_image: &[u32],
    lemma_image: &[u32],
) -> Result<(), String> {
    MIRROR
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| "resident BLOCK mirror lock poisoned".to_string())?
        .apply(hardware, semantic_ops, obligation_image, lemma_image)
}

pub(super) fn pop_owned(
    hardware: &mut HardwareCdcl,
    max_frame: u32,
) -> Result<Option<u64>, String> {
    MIRROR
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| "resident BLOCK mirror lock poisoned".to_string())?
        .pop_owned(hardware, max_frame)
}

pub(super) fn run_owned_root(
    hardware: &mut HardwareCdcl,
    max_frame: u32,
    requested_queries: usize,
    next_var_by_current: &[u32],
    query_template: &IncrementalQuery,
) -> Result<OwnedBlockRootWave, String> {
    MIRROR
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| "resident BLOCK mirror lock poisoned".to_string())?
        .run_owned_root(
            hardware,
            max_frame,
            requested_queries,
            next_var_by_current,
            query_template,
        )
}

pub(super) fn run_owned_full_root(
    hardware: &mut HardwareCdcl,
    max_frame: u32,
    step_limit: usize,
    next_var_by_current: &[u32],
    init_value_by_current: &[u32],
    latch_variables: &[u32],
    input_variables: &[u32],
    query_template: &IncrementalQuery,
) -> Result<OwnedBlockFullRootWave, String> {
    MIRROR
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| "resident BLOCK mirror lock poisoned".to_string())?
        .run_owned_full_root(
            hardware,
            max_frame,
            step_limit,
            next_var_by_current,
            init_value_by_current,
            latch_variables,
            input_variables,
            query_template,
        )
}

pub(super) fn take_owned_key(user_tag: u64) -> Result<Vec<u32>, String> {
    MIRROR
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| "resident BLOCK mirror lock poisoned".to_string())?
        .take_owned_key(user_tag)
}

pub(super) fn report() {
    eprintln!(
        "inductor-cdcl: live BLOCK controller steps {}, rebases {} (capacity compactions {}), root-reconciles {}, batches {}, commands {}, max-batch {}, service {:.3} ms, peak obligations/lemmas {}/{}, arena obligation/lemma/state/input {}/{}/{}/{} words, queue-pops {}, owned-pops {}, root-waves {}, root-work {}, root-ok {}, root-cpu-handoffs {}, root-service {:.3} ms",
        STEPS.load(Ordering::Relaxed),
        REBASES.load(Ordering::Relaxed),
        CAPACITY_COMPACTIONS.load(Ordering::Relaxed),
        ROOT_RECONCILES.load(Ordering::Relaxed),
        BATCHES.load(Ordering::Relaxed),
        COMMANDS.load(Ordering::Relaxed),
        MAX_BATCH.load(Ordering::Relaxed),
        SERVICE_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
        MAX_OBLIGATIONS.load(Ordering::Relaxed),
        MAX_LEMMAS.load(Ordering::Relaxed),
        MAX_OBLIGATION_ARENA.load(Ordering::Relaxed),
        MAX_LEMMA_ARENA.load(Ordering::Relaxed),
        MAX_STATE_ARENA.load(Ordering::Relaxed),
        MAX_INPUT_ARENA.load(Ordering::Relaxed),
        QUEUE_POPS.load(Ordering::Relaxed),
        OWNED_QUEUE_POPS.load(Ordering::Relaxed),
        ROOT_WAVES.load(Ordering::Relaxed),
        ROOT_WORK.load(Ordering::Relaxed),
        ROOT_OK.load(Ordering::Relaxed),
        ROOT_CPU_HANDOFFS.load(Ordering::Relaxed),
        ROOT_SERVICE_NS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
    );
}

pub(super) fn root_metrics() -> (u64, u64, u64) {
    (
        ROOT_WAVES.load(Ordering::Relaxed),
        ROOT_WORK.load(Ordering::Relaxed),
        ROOT_SERVICE_NS.load(Ordering::Relaxed),
    )
}

#[cfg(test)]
mod tests {
    use super::semantic_capacity_error;

    #[test]
    fn only_semantic_arena_capacity_triggers_epoch_compaction() {
        assert!(semantic_capacity_error(
            "incremental CDCL hardware error: BlockSemantic { error: 3, completed: 1, command: 20, command_status: 2, obligation_count: 0, lemma_count: 789 }"
        ));
        assert!(!semantic_capacity_error(
            "incremental CDCL hardware error: BlockSemantic { error: 3, completed: 1, command: 20, command_status: 1, obligation_count: 0, lemma_count: 789 }"
        ));
        assert!(!semantic_capacity_error(
            "incremental CDCL hardware error: InvalidResponse"
        ));
    }
}
