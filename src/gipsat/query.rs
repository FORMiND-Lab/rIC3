use super::DagCnfSolver;
use super::cdb::CREF_NONE;
use crate::accel::cdcl::{
    ABI_VERSION, BANK_ALIGNED_DOMAIN, BatchHeader, BatchResponseHeader, KEEP_LEARNTS,
    PACKED_SAT_MODEL, QueryHeader, RESPONSE_HEADER_WORDS, ResponseHeader, Status, UnknownReason,
    WANT_CORE, WANT_MODEL,
};
use logicrs::{Lbool, Lit, LitVec, Var, satif::Satif};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct QueryBudget {
    /// Zero means unlimited, matching the FPGA wire contract.
    pub decisions: u32,
    /// Zero means unlimited, matching the FPGA wire contract.
    pub conflicts: u32,
    /// CPU-reference escape hatch. Hardware does not consume this field.
    pub restarts: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct IncrementalQuery {
    pub frame: u32,
    pub assumptions: LitVec,
    pub constraints: Vec<LitVec>,
    pub domain: Vec<Var>,
    pub budget: QueryBudget,
    pub keep_learnts: bool,
}

pub fn bank_aligned_domain_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_BANK_ALIGNED_DOMAIN")
            .ok()
            .is_some_and(|value| !matches!(value.as_str(), "" | "0" | "false" | "off"))
    })
}

pub fn packed_sat_model_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_PACKED_SAT_MODEL")
            .ok()
            .is_some_and(|value| !matches!(value.as_str(), "" | "0" | "false" | "off"))
    })
}

pub fn local_incremental_domain_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("INDUCTOR_CDCL_LOCAL_DOMAIN")
            .ok()
            .is_some_and(|value| !matches!(value.as_str(), "" | "0" | "false" | "off"))
    })
}

pub fn encoded_domain_words(domain: &[Var]) -> usize {
    if !bank_aligned_domain_enabled() {
        return domain.len();
    }
    let mut banks = [0usize; 4];
    for variable in domain {
        banks[(u32::from(*variable) & 3) as usize] += 1;
    }
    4 * banks.into_iter().max().unwrap_or(0)
}

impl IncrementalQuery {
    pub fn new(frame: u32, assumptions: impl Into<LitVec>) -> Self {
        Self {
            frame,
            assumptions: assumptions.into(),
            constraints: Vec::new(),
            domain: Vec::new(),
            budget: QueryBudget::default(),
            keep_learnts: true,
        }
    }

    /// Encode the query exactly as the future XRT/HLS backend will receive it.
    pub fn pack(&self) -> (QueryHeader, Vec<u32>) {
        let constraint_words = self.constraints.iter().fold(0usize, |n, c| {
            n.checked_add(1 + c.len())
                .expect("incremental CDCL constraint payload overflow")
        });
        let bank_aligned = bank_aligned_domain_enabled();
        let domain_words = encoded_domain_words(&self.domain);
        let mut payload =
            Vec::with_capacity(self.assumptions.len() + constraint_words + domain_words);
        payload.extend(self.assumptions.iter().map(|l| Into::<u32>::into(*l)));
        for clause in &self.constraints {
            payload.push(clause.len() as u32);
            payload.extend(clause.iter().map(|l| Into::<u32>::into(*l)));
        }
        if bank_aligned {
            assert!(
                self.domain.len() <= 32768,
                "bank-aligned CDCL domain exceeds the 15-bit schedule ABI"
            );
            let mut banks: [Vec<(u16, u16)>; 4] = std::array::from_fn(|_| Vec::new());
            for (rank, variable) in self.domain.iter().enumerate() {
                let variable = u32::from(*variable);
                assert!(
                    variable < 32768,
                    "bank-aligned CDCL variable exceeds the 15-bit schedule ABI"
                );
                banks[(variable & 3) as usize].push((rank as u16, variable as u16));
            }
            for line in 0..domain_words / 4 {
                for bank in &banks {
                    let slot = bank.get(line).map_or(0, |&(rank, variable)| {
                        0x8000_0000 | (u32::from(rank) << 16) | u32::from(variable)
                    });
                    payload.push(slot);
                }
            }
        } else {
            payload.extend(self.domain.iter().map(|v| Into::<u32>::into(*v)));
        }

        let mut flags = WANT_MODEL | WANT_CORE;
        if self.keep_learnts {
            flags |= KEEP_LEARNTS;
        }
        if bank_aligned {
            flags |= BANK_ALIGNED_DOMAIN;
            if packed_sat_model_enabled() {
                flags |= PACKED_SAT_MODEL;
            }
        }
        let header = QueryHeader {
            version: ABI_VERSION,
            frame: self.frame,
            flags,
            n_assumptions: self.assumptions.len() as u32,
            n_constraint_words: constraint_words as u32,
            n_domain: domain_words as u32,
            decision_budget: self.budget.decisions,
            conflict_budget: self.budget.conflicts,
        };
        debug_assert!(header.valid_for(&payload));
        (header, payload)
    }
}

/// Pack many short inquiries into one DMA request. The result buffer is sized
/// by the caller because a full sparse model can be larger than the query.
pub fn pack_batch(
    queries: &[IncrementalQuery],
    result_capacity_words: u32,
) -> (BatchHeader, Vec<u32>) {
    let mut words = Vec::new();
    for query in queries {
        let (header, payload) = query.pack();
        words.extend(header.as_words());
        words.extend(payload);
    }
    let header = BatchHeader {
        version: ABI_VERSION,
        n_queries: queries
            .len()
            .try_into()
            .expect("incremental CDCL batch query count exceeds u32"),
        n_request_words: words
            .len()
            .try_into()
            .expect("incremental CDCL batch payload exceeds u32 words"),
        result_capacity_words,
    };
    debug_assert!(header.valid_for(&words));
    (header, words)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchDecodeError {
    Truncated,
    Version,
    QueryCount,
    Backend(u32),
    InvalidStatus,
    InvalidReason,
    InvalidResultShape,
}

fn decode_lit(word: u32) -> Lit {
    Lit::new(Var::from(word >> 1), word & 1 == 0)
}

fn decode_packed_sat_model(
    query: &IncrementalQuery,
    payload: &[u32],
) -> Result<LitVec, BatchDecodeError> {
    let expected = query.domain.len().div_ceil(32);
    if payload.len() != expected {
        return Err(BatchDecodeError::InvalidResultShape);
    }
    if let Some(&last) = payload.last() {
        let used = query.domain.len() & 31;
        if used != 0 && last & (!0u32 << used) != 0 {
            return Err(BatchDecodeError::InvalidResultShape);
        }
    }
    let mut model = LitVec::new();
    let mut bit = 0usize;
    for lane in 0..4u32 {
        for &variable in &query.domain {
            if u32::from(variable) & 3 != lane {
                continue;
            }
            let positive = payload[bit >> 5] & (1u32 << (bit & 31)) != 0;
            model.push(Lit::new(variable, positive));
            bit += 1;
        }
    }
    if bit != query.domain.len() {
        return Err(BatchDecodeError::InvalidResultShape);
    }
    Ok(model)
}

/// Decode the variable-length records emitted by the persistent HLS batch
/// command. Malformed device output is an error, never a partial SAT result.
pub fn decode_batch_results(
    queries: &[IncrementalQuery],
    words: &[u32],
) -> Result<Vec<IncrementalResult>, BatchDecodeError> {
    let prefix = words.get(..4).ok_or(BatchDecodeError::Truncated)?;
    let batch = BatchResponseHeader {
        version: prefix[0],
        n_queries: prefix[1],
        n_result_words: prefix[2],
        error: prefix[3],
    };
    if batch.version != ABI_VERSION {
        return Err(BatchDecodeError::Version);
    }
    if usize::try_from(batch.n_queries).ok() != Some(queries.len()) {
        return Err(BatchDecodeError::QueryCount);
    }
    if batch.error != 0 {
        return Err(BatchDecodeError::Backend(batch.error));
    }
    let result_words =
        usize::try_from(batch.n_result_words).map_err(|_| BatchDecodeError::InvalidResultShape)?;
    if words.len() != 4 + result_words {
        return Err(BatchDecodeError::InvalidResultShape);
    }

    let mut offset = 4usize;
    let mut results = Vec::with_capacity(queries.len());
    for query in queries {
        let header_words = words
            .get(offset..offset + RESPONSE_HEADER_WORDS)
            .ok_or(BatchDecodeError::Truncated)?;
        let header = ResponseHeader::from_words(header_words).ok_or(BatchDecodeError::Truncated)?;
        offset += RESPONSE_HEADER_WORDS;
        let n_model =
            usize::try_from(header.n_model).map_err(|_| BatchDecodeError::InvalidResultShape)?;
        let n_core =
            usize::try_from(header.n_core).map_err(|_| BatchDecodeError::InvalidResultShape)?;
        let payload = words
            .get(offset..offset + n_model + n_core)
            .ok_or(BatchDecodeError::Truncated)?;
        offset += payload.len();

        let result =
            match Status::from_word(header.status).ok_or(BatchDecodeError::InvalidStatus)? {
                Status::Sat if n_core == 0 && header.error == 0 => {
                    let model = if bank_aligned_domain_enabled() && packed_sat_model_enabled() {
                        decode_packed_sat_model(query, &payload[..n_model])?
                    } else {
                        payload[..n_model].iter().map(|w| decode_lit(*w)).collect()
                    };
                    IncrementalResult::Sat { model }
                }
                Status::Unsat if n_model == 0 && header.error == 0 => IncrementalResult::Unsat {
                    core: payload[..n_core].iter().map(|w| decode_lit(*w)).collect(),
                    used_constraints: !query.constraints.is_empty(),
                },
                Status::Unknown if n_model == 0 && n_core == 0 && header.error == 0 => {
                    let reason = UnknownReason::from_word(header.reason)
                        .ok_or(BatchDecodeError::InvalidReason)?;
                    if reason == UnknownReason::None {
                        return Err(BatchDecodeError::InvalidReason);
                    }
                    IncrementalResult::Unknown(reason)
                }
                Status::Error => return Err(BatchDecodeError::Backend(header.error)),
                _ => return Err(BatchDecodeError::InvalidResultShape),
            };
        results.push(result);
    }
    if offset != words.len() {
        return Err(BatchDecodeError::InvalidResultShape);
    }
    Ok(results)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IncrementalResult {
    /// GipSAT models are intentionally sparse: unassigned variables remain
    /// `None`, just as they do after the ordinary CPU search.
    Sat {
        model: LitVec,
    },
    /// The core contains only caller assumptions. Temporary clauses remain
    /// part of the query even when the activation literal participates.
    Unsat {
        core: LitVec,
        used_constraints: bool,
    },
    Unknown(crate::accel::cdcl::UnknownReason),
}

/// Common semantic boundary for the CPU reference and the future resident
/// SAT-Accel-derived backend.
pub trait IncrementalCdcl {
    fn solve_incremental(&mut self, query: &IncrementalQuery) -> IncrementalResult;
}

impl DagCnfSolver {
    /// Run an exact GipSAT inquiry until it completes or reaches a small
    /// conflict limit. The proof-neutral profiler invokes this on a clone;
    /// active dispatch may invoke it on the live solver and then reset the
    /// query state. A conclusive result has ordinary CPU semantics; hitting
    /// the limit returns `UNKNOWN/ConflictBudget`.
    pub fn solve_incremental_preflight(
        &mut self,
        query: &IncrementalQuery,
        conflict_limit: u32,
    ) -> IncrementalResult {
        if query.frame != self.accel_level {
            return IncrementalResult::Unknown(crate::accel::cdcl::UnknownReason::FrameMiss);
        }
        if conflict_limit == 0 {
            return IncrementalResult::Unknown(crate::accel::cdcl::UnknownReason::ConflictBudget);
        }
        let result = self.solve_with_limits(
            &query.assumptions,
            query.constraints.clone(),
            query.domain.iter().copied(),
            None,
            Some(conflict_limit),
            false,
        );
        match result {
            Some(true) => IncrementalResult::Sat {
                // Match the accelerator ABI: root assignments outside the
                // dependency-closed query domain are solver-internal state,
                // not part of this inquiry's witness.
                model: query
                    .domain
                    .iter()
                    .filter_map(|variable| {
                        self.sat_value(variable.lit())
                            .map(|polarity| variable.lit().not_if(!polarity))
                    })
                    .collect(),
            },
            Some(false) => IncrementalResult::Unsat {
                core: query
                    .assumptions
                    .iter()
                    .filter(|lit| self.unsat_core.has(**lit))
                    .copied()
                    .collect(),
                used_constraints: !query.constraints.is_empty()
                    && self.unsat_core.has(self.constrain_act.lit()),
            },
            None => IncrementalResult::Unknown(crate::accel::cdcl::UnknownReason::ConflictBudget),
        }
    }

    /// Classify a live inquiry and restore the solver to a clean query
    /// boundary. Preflight learnts are deliberately temporary, so neither
    /// they nor assumptions and temporary constraints leak into later FPGA
    /// validation or CPU fallback.
    pub fn classify_incremental_preflight(
        &mut self,
        query: &IncrementalQuery,
        conflict_limit: u32,
    ) -> IncrementalResult {
        let result = self.solve_incremental_preflight(query, conflict_limit);
        self.reset();
        result
    }

    /// Finish one speculative inquiry exactly on the live CPU solver and
    /// restore a clean query boundary. This is used by the active dispatch
    /// sampler: the answer is reusable, while its measured cost predicts
    /// whether the remaining compatible inquiries belong on CPU or FPGA.
    pub fn classify_incremental_exact(&mut self, query: &IncrementalQuery) -> IncrementalResult {
        let mut exact = query.clone();
        exact.budget = QueryBudget::default();
        let result = self.solve_incremental(&exact);
        self.reset();
        result
    }

    /// Restore the failed-assumption core of an already-qualified result
    /// without repeating the solve. This covers exact CPU preflight and the
    /// explicit trusted-accelerator policy. IC3 may only have strengthened the
    /// frame since classification, so the UNSAT implication remains valid.
    /// This entry point is separate from the untrusted hardware-core validator.
    pub fn install_incremental_proven_unsat_core(
        &mut self,
        query: &IncrementalQuery,
        core: &[Lit],
        used_constraints: bool,
    ) -> bool {
        if query.frame != self.accel_level {
            return false;
        }
        let mut unmatched: Vec<Lit> = query.assumptions.iter().copied().collect();
        for &lit in core {
            let Some(position) = unmatched.iter().position(|candidate| *candidate == lit) else {
                return false;
            };
            unmatched.swap_remove(position);
        }

        self.reset();
        self.assump = query.assumptions.clone();
        self.constraint = query.constraints.clone();
        self.unsat_core.clear();
        for &lit in core {
            self.unsat_core.insert(lit);
        }
        if used_constraints {
            self.unsat_core.insert(self.constrain_act.lit());
        }
        true
    }

    /// Decode an accelerator model without proving its clauses on the CPU.
    /// GipSAT searches only the dependency-closed query domain; variables
    /// outside it are deliberately absent from an ordinary CPU model too. A
    /// trusted device must return every query-domain variable exactly once and
    /// no variable outside that domain.
    fn incremental_domain_assignment(
        &self,
        query: &IncrementalQuery,
        model: &[Lit],
    ) -> Option<Vec<Option<bool>>> {
        if query.frame != self.accel_level {
            return None;
        }
        let n_var = self.num_var();
        let mut assignment = vec![None; n_var];
        let mut required = vec![false; n_var];
        for &variable in &query.domain {
            let var: usize = variable.into();
            let slot = required.get_mut(var)?;
            if *slot {
                return None;
            }
            *slot = true;
        }
        for &lit in model {
            let var: usize = lit.var().into();
            if !required.get(var).copied().unwrap_or(false) {
                return None;
            }
            let slot = assignment.get_mut(var)?;
            if slot.is_some() {
                return None;
            }
            *slot = Some(lit.polarity());
        }
        if required
            .iter()
            .zip(&assignment)
            .any(|(required, value)| *required && value.is_none())
        {
            return None;
        }
        Some(assignment)
    }

    /// Check a hardware SAT model against the exact, current CPU formula.
    ///
    /// This is intentionally stricter than checking the sparse values GipSAT
    /// normally exposes: an untrusted accelerator answer may bypass CPU search
    /// only if it assigns every variable exactly once and satisfies the
    /// transition CNF, every current permanent lemma, the assumptions, and all
    /// query-local constraints. A stale model from a batch prepared before IC3
    /// added another lemma therefore fails closed.
    fn validated_incremental_assignment(
        &self,
        query: &IncrementalQuery,
        model: &[Lit],
    ) -> Option<Vec<Option<bool>>> {
        let assignment = self.incremental_domain_assignment(query, model)?;
        // Clause-by-clause CPU validation requires a total formula assignment.
        // A sparse dependency-domain witness is accepted only by the explicit
        // qualified-device path; conservative mode falls back to CPU search.
        if assignment.iter().any(Option::is_none) {
            return None;
        }
        let lit_true = |lit: Lit| {
            let var: usize = lit.var().into();
            assignment
                .get(var)
                .and_then(|value| *value)
                .is_some_and(|value| value == lit.polarity())
        };
        if !query.assumptions.iter().copied().all(&lit_true) {
            return None;
        }
        if self
            .resident_trans
            .iter()
            .chain(&self.resident_lemmas)
            .chain(&query.constraints)
            .any(|clause| !clause.iter().copied().any(&lit_true))
        {
            return None;
        }
        Some(assignment)
    }

    fn install_incremental_assignment(
        &mut self,
        query: &IncrementalQuery,
        assignment: Vec<Option<bool>>,
    ) -> bool {
        if self.trivial_unsat || self.propagate() != CREF_NONE {
            return false;
        }

        self.assump = query.assumptions.clone();
        self.constraint = query.constraints.clone();
        let constraint_vars: Vec<Var> = query
            .constraints
            .iter()
            .flatten()
            .map(|lit| lit.var())
            .collect();
        let setup_ok = self.new_round(
            query
                .domain
                .iter()
                .copied()
                .chain(query.assumptions.iter().map(|lit| lit.var()))
                .chain(constraint_vars),
            query.constraints.clone(),
            true,
        );
        if !setup_ok {
            self.reset();
            return false;
        }
        self.clean_learnt(true);
        self.simplify();

        self.new_level();
        if !query.constraints.is_empty() {
            let activation = self.constrain_act.lit();
            match self.value.v(activation) {
                Lbool::TRUE => {}
                Lbool::FALSE => {
                    self.reset();
                    return false;
                }
                _ => self.assign(activation, CREF_NONE),
            }
        }
        for (var, polarity) in assignment.into_iter().enumerate() {
            let Some(polarity) = polarity else {
                continue;
            };
            let lit = Lit::new(Var::from(var), polarity);
            match self.value.v(lit) {
                Lbool::TRUE => {}
                Lbool::FALSE => {
                    self.reset();
                    return false;
                }
                _ => self.assign(lit, CREF_NONE),
            }
        }
        // The ordinary path checked the model clause-by-clause; the trusted
        // path relies on the already-qualified device. In both cases feeding a
        // complete assignment into GipSAT's propagation queue is redundant and
        // violates its normal one-decision-at-a-time invariant. Downstream
        // predecessor lifting only reads values and selectively calls
        // `flip_to_none`.
        self.propagated = self.trail.len() as u32;
        true
    }

    pub fn validate_incremental_sat_model(&self, query: &IncrementalQuery, model: &[Lit]) -> bool {
        self.validated_incremental_assignment(query, model)
            .is_some()
    }

    /// Verify only the fixed-width transport contract for a model returned by
    /// a qualified accelerator. No resident/query clause is evaluated here.
    pub fn trusted_incremental_sat_model_shape(
        &self,
        query: &IncrementalQuery,
        model: &[Lit],
    ) -> bool {
        self.incremental_domain_assignment(query, model).is_some()
    }

    /// Validate and import a complete FPGA SAT model into GipSAT's live trail.
    /// Downstream IC3 code can then use `sat_value` and `flip_to_none` exactly
    /// as after an ordinary CPU SAT search. Any malformed, stale, or internally
    /// inconsistent model is rejected and the next ordinary solve resets the
    /// temporary setup before falling back to CPU.
    pub fn install_incremental_sat_model(
        &mut self,
        query: &IncrementalQuery,
        model: &[Lit],
    ) -> bool {
        let Some(assignment) = self.validated_incremental_assignment(query, model) else {
            return false;
        };
        self.install_incremental_assignment(query, assignment)
    }

    /// Install a SAT assignment returned by an already-qualified FPGA without
    /// re-evaluating every resident and temporary clause on the CPU. This is
    /// not a solver fallback: only frame/model transport shape is checked, then
    /// GipSAT's live model state is reconstructed for existing IC3 consumers.
    /// Use only behind the explicit accelerator trust policy.
    pub fn install_trusted_incremental_sat_model(
        &mut self,
        query: &IncrementalQuery,
        model: &[Lit],
    ) -> bool {
        let Some(assignment) = self.incremental_domain_assignment(query, model) else {
            return false;
        };
        self.install_incremental_assignment(query, assignment)
    }

    /// Re-prove an FPGA failed-assumption core against the exact live CPU
    /// formula. The hardware result is only a candidate: literals must form a
    /// multiset subset of the original assumptions, and GipSAT must establish
    /// UNSAT again with the same frame, temporary constraints, and domain.
    /// Hardware scheduling budgets are removed for this proof step.
    ///
    /// On success GipSAT's live `unsat_core` is the proof core of the reduced
    /// query, so downstream IC3 generalization can consume it exactly as after
    /// an ordinary CPU solve. The returned size is that CPU proof core size;
    /// `Some(0)` is a valid result when the formula is UNSAT independently of
    /// assumptions.
    pub fn validate_incremental_unsat_core(
        &mut self,
        query: &IncrementalQuery,
        hardware_core: &[Lit],
    ) -> Option<usize> {
        if query.frame != self.accel_level {
            return None;
        }
        let mut unmatched: Vec<Lit> = query.assumptions.iter().copied().collect();
        for &lit in hardware_core {
            let position = unmatched.iter().position(|candidate| *candidate == lit)?;
            unmatched.swap_remove(position);
        }

        let mut reduced = query.clone();
        reduced.assumptions = LitVec::from(hardware_core);
        reduced.budget = QueryBudget::default();
        match self.solve_incremental(&reduced) {
            IncrementalResult::Unsat { core, .. } => Some(core.len()),
            IncrementalResult::Sat { .. } | IncrementalResult::Unknown(_) => None,
        }
    }
}

/// Use a hardware answer only when it is conclusive. Decision/conflict limits
/// are hardware scheduling budgets, so the ordinary GipSAT retry removes them
/// instead of returning the same `UNKNOWN` without doing any CPU search.
pub fn solve_with_cpu_fallback(
    hardware: &mut impl IncrementalCdcl,
    cpu: &mut DagCnfSolver,
    query: &IncrementalQuery,
) -> IncrementalResult {
    let result = hardware.solve_incremental(query);
    if !matches!(result, IncrementalResult::Unknown(_)) {
        return result;
    }
    solve_on_cpu_after_hardware_unknown(cpu, query)
}

/// Retry an already-inconclusive inquiry without hardware-only budgets.
pub fn solve_on_cpu_after_hardware_unknown(
    cpu: &mut DagCnfSolver,
    query: &IncrementalQuery,
) -> IncrementalResult {
    let mut cpu_query = query.clone();
    cpu_query.budget.decisions = 0;
    cpu_query.budget.conflicts = 0;
    cpu.solve_incremental(&cpu_query)
}

impl IncrementalCdcl for DagCnfSolver {
    fn solve_incremental(&mut self, query: &IncrementalQuery) -> IncrementalResult {
        if query.frame != self.accel_level {
            return IncrementalResult::Unknown(crate::accel::cdcl::UnknownReason::FrameMiss);
        }

        // GipSAT currently exposes restart limits, while the hardware contract
        // budgets decisions and conflicts. Until those counters are plumbed
        // through search.rs, a non-zero hardware-only budget is explicit
        // unsupported rather than silently ignored by the reference backend.
        if query.budget.decisions != 0 || query.budget.conflicts != 0 {
            return IncrementalResult::Unknown(crate::accel::cdcl::UnknownReason::Unsupported);
        }

        let result = self.solve_with_param(
            &query.assumptions,
            query.constraints.clone(),
            query.domain.iter().copied(),
            query.budget.restarts,
        );
        match result {
            Some(true) => IncrementalResult::Sat {
                model: self.sat_value_iter().copied().collect(),
            },
            Some(false) => IncrementalResult::Unsat {
                core: query
                    .assumptions
                    .iter()
                    .filter(|l| self.unsat_core.has(**l))
                    .copied()
                    .collect(),
                used_constraints: !query.constraints.is_empty()
                    && self.unsat_core.has(self.constrain_act.lit()),
            },
            None => IncrementalResult::Unknown(crate::accel::cdcl::UnknownReason::RestartBudget),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use logicrs::satif::Satif;
    use logicrs::{DagCnf, LitVec};

    fn implication_solver() -> (DagCnfSolver, logicrs::Lit, logicrs::Lit) {
        let mut dc = DagCnf::new();
        let a = dc.new_var().lit();
        let b = dc.new_var().lit();
        let mut solver = DagCnfSolver::new(&dc);
        solver.add_clause(&[!a, b]);
        (solver, a, b)
    }

    #[test]
    fn cpu_reference_returns_sparse_model_and_assumption_core() {
        let (mut solver, a, b) = implication_solver();
        let (n_var, frame, resident) = solver.incremental_resident_snapshot();
        assert!(n_var >= 2 && frame == 0);
        assert!(resident.iter().any(|clause| clause.as_slice() == [!a, b]));
        let sat = IncrementalQuery::new(0, LitVec::from([a]));
        let IncrementalResult::Sat { model } = solver.solve_incremental(&sat) else {
            panic!("expected SAT");
        };
        assert!(model.contains(&a));
        assert!(model.contains(&b));

        let unsat = IncrementalQuery::new(0, LitVec::from([a, !b]));
        let IncrementalResult::Unsat { core, .. } = solver.solve_incremental(&unsat) else {
            panic!("expected UNSAT");
        };
        assert!(core.contains(&a));
        assert!(core.contains(&!b));
    }

    #[test]
    fn cpu_reference_reports_frame_budget_and_constraint_semantics() {
        let (mut solver, a, _) = implication_solver();

        let wrong_frame = IncrementalQuery::new(1, LitVec::from([a]));
        assert_eq!(
            solver.solve_incremental(&wrong_frame),
            IncrementalResult::Unknown(crate::accel::cdcl::UnknownReason::FrameMiss),
        );

        let mut budgeted = IncrementalQuery::new(0, LitVec::from([a]));
        budgeted.budget.decisions = 1;
        assert_eq!(
            solver.solve_incremental(&budgeted),
            IncrementalResult::Unknown(crate::accel::cdcl::UnknownReason::Unsupported),
        );

        let mut constrained = IncrementalQuery::new(0, LitVec::from([a]));
        constrained.constraints.push(LitVec::from([!a]));
        let (header, payload) = constrained.pack();
        assert!(header.valid_for(&payload));
        let (batch, words) = pack_batch(&[constrained.clone(), constrained.clone()], 256);
        assert!(batch.valid_for(&words));
        assert!(matches!(
            solver.solve_incremental(&constrained),
            IncrementalResult::Unsat { .. }
        ));
    }

    #[test]
    fn cpu_preflight_stops_at_the_conflict_limit() {
        let mut dc = DagCnf::new();
        let a = dc.new_var().lit();
        let b = dc.new_var().lit();
        let solver = DagCnfSolver::new(&dc);
        let mut query = IncrementalQuery::new(0, LitVec::new());
        query.domain = (0..solver.num_var()).map(Var::from).collect();
        query.constraints = vec![
            LitVec::from([a, b]),
            LitVec::from([!a, b]),
            LitVec::from([a, !b]),
            LitVec::from([!a, !b]),
        ];

        let mut preflight = solver.clone();
        assert_eq!(
            preflight.solve_incremental_preflight(&query, 1),
            IncrementalResult::Unknown(UnknownReason::ConflictBudget),
        );
        let mut exact = solver.clone();
        assert!(matches!(
            exact.solve_incremental(&query),
            IncrementalResult::Unsat { .. }
        ));

        let mut live = solver.clone();
        let learnts_before = live.cdb.num_learnt();
        assert_eq!(
            live.classify_incremental_preflight(&query, 1),
            IncrementalResult::Unknown(UnknownReason::ConflictBudget),
        );
        assert_eq!(live.cdb.num_learnt(), learnts_before);
        assert!(matches!(
            live.solve_incremental(&query),
            IncrementalResult::Unsat { .. }
        ));
    }

    #[test]
    fn conclusive_cpu_preflight_results_can_be_restored_without_resolving() {
        // Keep DagCnf alive: DagCnfSolver intentionally stores a non-owning
        // pointer to it.
        let mut dc = DagCnf::new();
        let a = dc.new_var().lit();
        let b = dc.new_var().lit();
        let mut solver = DagCnfSolver::new(&dc);
        solver.add_clause(&[!a, b]);
        let mut sat = IncrementalQuery::new(0, LitVec::from([a]));
        sat.domain = (0..solver.num_var()).map(Var::from).collect();
        let IncrementalResult::Sat { model } = solver.classify_incremental_preflight(&sat, 8)
        else {
            panic!("expected conclusive SAT preflight");
        };
        assert!(solver.install_incremental_sat_model(&sat, &model));
        assert_eq!(solver.sat_value(a), Some(true));
        assert_eq!(solver.sat_value(b), Some(true));

        let mut unsat = IncrementalQuery::new(0, LitVec::from([a, !b]));
        unsat.domain = (0..solver.num_var()).map(Var::from).collect();
        let IncrementalResult::Unsat {
            core,
            used_constraints,
        } = solver.classify_incremental_preflight(&unsat, 8)
        else {
            panic!("expected conclusive UNSAT preflight");
        };
        // Simulate an earlier push strengthening this frame between the
        // speculative preflight and ordered IC3 consumption.
        solver.add_clause(&[b]);
        assert!(solver.install_incremental_proven_unsat_core(&unsat, &core, used_constraints,));
        assert!(core.iter().all(|lit| solver.unsat_has(*lit)));
        assert!(!solver.install_incremental_proven_unsat_core(&unsat, &[!a], used_constraints,));
    }

    #[test]
    fn exact_live_classifier_clears_hardware_budgets_and_restores_boundary() {
        let (mut solver, a, _) = implication_solver();
        let mut query = IncrementalQuery::new(0, LitVec::from([a]));
        query.domain = (0..solver.num_var()).map(Var::from).collect();
        query.budget.conflicts = 1;
        assert!(matches!(
            solver.classify_incremental_exact(&query),
            IncrementalResult::Sat { .. }
        ));
        assert!(solver.assert_value(a).is_none());
    }

    #[test]
    fn hardware_batch_records_decode_without_trusting_partial_output() {
        let (solver, a, b) = implication_solver();
        drop(solver);
        let queries = vec![
            IncrementalQuery::new(0, LitVec::from([a])),
            IncrementalQuery::new(0, LitVec::from([a, !b])),
        ];
        let mut words = vec![ABI_VERSION, 2, 0, 0];
        // SAT record with a two-literal sparse model.
        words.extend([Status::Sat as u32, 0, 2, 0, 1, 0, 1, 0, 0]);
        words.extend([u32::from(a), u32::from(b)]);
        // UNSAT record with a two-literal assumption core.
        words.extend([Status::Unsat as u32, 0, 0, 2, 0, 1, 2, 1, 0]);
        words.extend([u32::from(a), u32::from(!b)]);
        words[2] = (words.len() - 4) as u32;

        let decoded = decode_batch_results(&queries, &words).unwrap();
        assert!(matches!(&decoded[0], IncrementalResult::Sat { model }
            if model.contains(&a) && model.contains(&b)));
        assert!(matches!(&decoded[1], IncrementalResult::Unsat { core, .. }
            if core.contains(&a) && core.contains(&!b)));

        words[1] = 1;
        assert_eq!(
            decode_batch_results(&queries, &words),
            Err(BatchDecodeError::QueryCount),
        );
    }

    #[test]
    fn packed_sat_models_decode_lane_major_across_word_boundaries() {
        let mut query = IncrementalQuery::new(0, LitVec::new());
        query.domain = (0..33).map(Var::from).collect();
        let lane_major: Vec<Var> = (0..4u32)
            .flat_map(|lane| {
                query
                    .domain
                    .iter()
                    .copied()
                    .filter(move |variable| u32::from(*variable) & 3 == lane)
            })
            .collect();
        let mut payload = vec![0u32; 2];
        for (bit, variable) in lane_major.iter().enumerate() {
            if u32::from(*variable) & 1 == 0 {
                payload[bit >> 5] |= 1u32 << (bit & 31);
            }
        }

        let model = decode_packed_sat_model(&query, &payload).unwrap();
        assert_eq!(model.len(), 33);
        for (literal, variable) in model.iter().zip(lane_major) {
            assert_eq!(literal.var(), variable);
            assert_eq!(literal.polarity(), u32::from(variable) & 1 == 0);
        }
    }

    #[test]
    fn packed_sat_models_reject_bad_lengths_and_nonzero_tail_bits() {
        let mut query = IncrementalQuery::new(0, LitVec::new());
        query.domain = (0..33).map(Var::from).collect();
        assert_eq!(
            decode_packed_sat_model(&query, &[0]),
            Err(BatchDecodeError::InvalidResultShape),
        );
        assert_eq!(
            decode_packed_sat_model(&query, &[0, 2]),
            Err(BatchDecodeError::InvalidResultShape),
        );
    }

    #[test]
    fn hardware_unknown_retries_on_cpu_without_hardware_budgets() {
        struct BudgetUnknown;
        impl IncrementalCdcl for BudgetUnknown {
            fn solve_incremental(&mut self, _: &IncrementalQuery) -> IncrementalResult {
                IncrementalResult::Unknown(UnknownReason::DecisionBudget)
            }
        }

        let (mut cpu, a, _) = implication_solver();
        let mut query = IncrementalQuery::new(0, LitVec::from([a]));
        query.budget.decisions = 1;
        query.budget.conflicts = 1;
        let result = solve_with_cpu_fallback(&mut BudgetUnknown, &mut cpu, &query);
        assert!(matches!(result, IncrementalResult::Sat { model } if model.contains(&a)));
    }

    #[test]
    fn hardware_unsat_core_is_subset_checked_and_reproved_without_budgets() {
        let mut dc = DagCnf::new();
        let a = dc.new_var().lit();
        let b = dc.new_var().lit();
        let c = dc.new_var().lit();
        let mut solver = DagCnfSolver::new(&dc);
        solver.add_clause(&[!a, b]);

        let mut query = IncrementalQuery::new(0, LitVec::from([c, a, !b]));
        query.budget.decisions = 7;
        query.budget.conflicts = 3;
        query.domain = (0..solver.num_var()).map(Var::from).collect();

        assert_eq!(
            solver.validate_incremental_unsat_core(&query, &[a, !b]),
            Some(2),
        );
        assert!(solver.unsat_has(a));
        assert!(solver.unsat_has(!b));
        assert!(!solver.unsat_has(c));

        // A literal outside the original assumption multiset is malformed,
        // while an insufficient subset is rejected by the exact CPU solve.
        assert_eq!(solver.validate_incremental_unsat_core(&query, &[!c]), None,);
        assert_eq!(solver.validate_incremental_unsat_core(&query, &[a]), None);

        let single_a = IncrementalQuery::new(0, LitVec::from([a]));
        assert_eq!(
            solver.validate_incremental_unsat_core(&single_a, &[a, a]),
            None,
        );
    }

    #[test]
    fn complete_hardware_model_is_validated_and_imported() {
        // Keep DagCnf alive: DagCnfSolver intentionally stores a non-owning
        // pointer to it.
        let mut dc = DagCnf::new();
        let a = dc.new_var().lit();
        let b = dc.new_var().lit();
        let mut solver = DagCnfSolver::new(&dc);
        solver.add_clause(&[!a, b]);
        let mut query = IncrementalQuery::new(0, LitVec::from([a]));
        query.constraints.push(LitVec::from([b]));
        query.domain = (0..solver.num_var()).map(Var::from).collect();
        let model: LitVec = (0..solver.num_var())
            .map(|var| {
                if var == 0 {
                    Lit::constant(true)
                } else {
                    Var::from(var).lit()
                }
            })
            .collect();

        assert!(solver.install_incremental_sat_model(&query, &model));
        assert_eq!(solver.sat_value(a), Some(true));
        assert_eq!(solver.sat_value(b), Some(true));

        let incomplete = LitVec::from([a, b]);
        assert!(!solver.install_incremental_sat_model(&query, &incomplete));

        solver.add_clause(&[!b]);
        assert!(!solver.install_incremental_sat_model(&query, &model));
    }

    #[test]
    fn qualified_hardware_model_can_be_imported_without_clause_replay() {
        // The trusted entry point is intentionally distinct from the ordinary
        // validator. Use a complete but semantically false assignment to prove
        // this test exercises only transport shape and live-state restoration.
        let mut dc = DagCnf::new();
        let a = dc.new_var().lit();
        let b = dc.new_var().lit();
        let mut solver = DagCnfSolver::new(&dc);
        solver.add_clause(&[!a, b]);
        let mut query = IncrementalQuery::new(0, LitVec::from([!a]));
        query.domain = (0..solver.num_var()).map(Var::from).collect();
        let IncrementalResult::Sat { mut model } = solver.classify_incremental_exact(&query) else {
            panic!("expected a source model");
        };
        let a_slot = model
            .iter_mut()
            .find(|lit| lit.var() == a.var())
            .expect("complete source model");
        *a_slot = a;

        assert!(!solver.validate_incremental_sat_model(&query, &model));
        assert!(solver.install_trusted_incremental_sat_model(&query, &model));
        assert_eq!(solver.sat_value(a), Some(true));

        let incomplete = LitVec::from([a, b]);
        assert!(!solver.install_trusted_incremental_sat_model(&query, &incomplete));
        let wrong_frame = IncrementalQuery::new(1, LitVec::new());
        assert!(!solver.install_trusted_incremental_sat_model(&wrong_frame, &model));
    }

    #[test]
    fn dependency_domain_trusted_model_leaves_unrelated_variables_unassigned() {
        let mut dc = DagCnf::new();
        let a = dc.new_var().lit();
        let b = dc.new_var().lit();
        let unrelated = dc.new_var().lit();
        let mut solver = DagCnfSolver::new(&dc);
        solver.add_clause(&[!a, b]);

        let mut query = IncrementalQuery::new(0, LitVec::from([a]));
        query.domain = solver.incremental_local_domain(std::iter::once(a.var()));
        assert!(query.domain.contains(&a.var()));
        assert!(query.domain.contains(&b.var()));
        assert!(!query.domain.contains(&unrelated.var()));

        let IncrementalResult::Sat { model } = solver.classify_incremental_exact(&query) else {
            panic!("expected a dependency-domain SAT model");
        };
        assert_eq!(model.len(), query.domain.len());
        assert!(!solver.validate_incremental_sat_model(&query, &model));
        assert!(solver.install_trusted_incremental_sat_model(&query, &model));
        assert_eq!(solver.sat_value(a), Some(true));
        assert_eq!(solver.sat_value(b), Some(true));
        assert_eq!(solver.sat_value(unrelated), None);

        let mut missing = model.clone();
        missing.pop();
        assert!(!solver.install_trusted_incremental_sat_model(&query, &missing));
        let mut extraneous = model;
        extraneous.push(unrelated);
        assert!(!solver.install_trusted_incremental_sat_model(&query, &extraneous));
    }

    #[test]
    fn incremental_context_revision_tracks_permanent_strengthening() {
        let mut dc = DagCnf::new();
        let a = dc.new_var().lit();
        let b = dc.new_var().lit();
        let mut solver = DagCnfSolver::new(&dc);
        assert_eq!(solver.incremental_context_revision(), 0);
        solver.add_clause(&[a, b]);
        assert_eq!(solver.incremental_context_revision(), 1);

        let mut clone = solver.clone();
        assert_eq!(clone.incremental_context_revision(), 1);
        clone.add_clause(&[!a, b]);
        assert_eq!(clone.incremental_context_revision(), 2);
        assert_eq!(solver.incremental_context_revision(), 1);
    }
}
