use super::DagCnfSolver;
use super::cdb::CREF_NONE;
use crate::accel::cdcl::{
    ABI_VERSION, BatchHeader, BatchResponseHeader, KEEP_LEARNTS, QueryHeader,
    RESPONSE_HEADER_WORDS, ResponseHeader, Status, UnknownReason, WANT_CORE, WANT_MODEL,
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
        let mut payload = Vec::with_capacity(
            self.assumptions.len() + constraint_words + self.domain.len(),
        );
        payload.extend(self.assumptions.iter().map(|l| Into::<u32>::into(*l)));
        for clause in &self.constraints {
            payload.push(clause.len() as u32);
            payload.extend(clause.iter().map(|l| Into::<u32>::into(*l)));
        }
        payload.extend(self.domain.iter().map(|v| Into::<u32>::into(*v)));

        let mut flags = WANT_MODEL | WANT_CORE;
        if self.keep_learnts {
            flags |= KEEP_LEARNTS;
        }
        let header = QueryHeader {
            version: ABI_VERSION,
            frame: self.frame,
            flags,
            n_assumptions: self.assumptions.len() as u32,
            n_constraint_words: constraint_words as u32,
            n_domain: self.domain.len() as u32,
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
    let result_words = usize::try_from(batch.n_result_words)
        .map_err(|_| BatchDecodeError::InvalidResultShape)?;
    if words.len() != 4 + result_words {
        return Err(BatchDecodeError::InvalidResultShape);
    }

    let mut offset = 4usize;
    let mut results = Vec::with_capacity(queries.len());
    for query in queries {
        let header_words = words
            .get(offset..offset + RESPONSE_HEADER_WORDS)
            .ok_or(BatchDecodeError::Truncated)?;
        let header = ResponseHeader::from_words(header_words)
            .ok_or(BatchDecodeError::Truncated)?;
        offset += RESPONSE_HEADER_WORDS;
        let n_model = usize::try_from(header.n_model)
            .map_err(|_| BatchDecodeError::InvalidResultShape)?;
        let n_core = usize::try_from(header.n_core)
            .map_err(|_| BatchDecodeError::InvalidResultShape)?;
        let payload = words
            .get(offset..offset + n_model + n_core)
            .ok_or(BatchDecodeError::Truncated)?;
        offset += payload.len();

        let result = match Status::from_word(header.status)
            .ok_or(BatchDecodeError::InvalidStatus)?
        {
            Status::Sat if n_core == 0 && header.error == 0 => IncrementalResult::Sat {
                model: payload[..n_model].iter().map(|w| decode_lit(*w)).collect(),
            },
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
    Sat { model: LitVec },
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
    /// Check a hardware SAT model against the exact, current CPU formula.
    ///
    /// This is intentionally stricter than checking the sparse values GipSAT
    /// normally exposes: an accelerator answer may bypass CPU search only if
    /// it assigns every non-activation variable exactly once and satisfies the
    /// transition CNF, every current permanent lemma, the assumptions, and all
    /// query-local constraints. A stale model from a batch prepared before IC3
    /// added another lemma therefore fails closed.
    fn validated_incremental_assignment(
        &self,
        query: &IncrementalQuery,
        model: &[Lit],
    ) -> Option<Vec<bool>> {
        if query.frame != self.accel_level {
            return None;
        }
        let n_var = self.num_var();
        let mut assignment = vec![None; n_var];
        for &lit in model {
            let var: usize = lit.var().into();
            let slot = assignment.get_mut(var)?;
            if slot.is_some() {
                return None;
            }
            *slot = Some(lit.polarity());
        }
        let assignment: Vec<bool> = assignment.into_iter().collect::<Option<_>>()?;
        let lit_true = |lit: Lit| {
            let var: usize = lit.var().into();
            assignment
                .get(var)
                .is_some_and(|value| *value == lit.polarity())
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
        // The model was checked clause-by-clause above. Feeding a complete
        // assignment into GipSAT's propagation queue is both redundant and
        // violates its normal invariant that each decision is propagated
        // before the next one is enqueued. Keep watchers untouched and mark
        // the imported trail consumed; downstream model lifting only reads
        // values and selectively calls `flip_to_none`.
        self.propagated = self.trail.len() as u32;
        true
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
            None => IncrementalResult::Unknown(
                crate::accel::cdcl::UnknownReason::RestartBudget,
            ),
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
}
