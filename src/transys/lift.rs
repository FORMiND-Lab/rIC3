use crate::{
    gipsat::DagCnfSolver,
    transys::{Transys, TransysIf, unroll::TransysUnroll},
};
use giputils::hash::GHashSet;
use logicrs::{Lit, LitVec, Var, satif::Satif};

pub struct TsLift {
    _ts: Box<Transys>,
    uts: TransysUnroll<Transys>,
    slv: DagCnfSolver,
}

impl TsLift {
    pub fn new(uts: TransysUnroll<Transys>) -> Self {
        let ts = Box::new(uts.compile());
        let slv = DagCnfSolver::new(&ts.rel);
        Self { _ts: ts, uts, slv }
    }

    pub fn lift(
        &mut self,
        satif: &mut impl Satif,
        target: impl IntoIterator<Item = impl AsRef<Lit>>,
        order: impl FnMut(usize, &mut [Lit]) -> bool,
    ) -> (LitVec, Vec<LitVec>) {
        self.complex_lift(satif, self.uts.latch.clone(), target, order)
    }

    /// Lift a clause-valid external model without importing it into GipSAT's
    /// trail. The external assignment supplies only candidate state/input
    /// values; `minimal_premise` independently proves that the returned state,
    /// together with the recorded inputs, implies the target through the
    /// transition relation. A missing value or failed implication check is a
    /// proof-safe `None` for the caller to replace with the full predecessor.
    pub fn lift_model(
        &mut self,
        model: &[Lit],
        target: impl IntoIterator<Item = impl AsRef<Lit>>,
        mut order: impl FnMut(usize, &mut [Lit]) -> bool,
    ) -> Option<(LitVec, Vec<LitVec>)> {
        let mut cls: LitVec = target.into_iter().map(|lit| *lit.as_ref()).collect();
        if cls.is_empty() {
            return Some((LitVec::new(), vec![]));
        }
        cls = !cls;
        let value = |lit: Lit| {
            model
                .iter()
                .find(|candidate| candidate.var() == lit.var())
                .map(|candidate| candidate.polarity())
        };

        let mut inputs = Vec::new();
        let mut inputs_flatten = LitVec::new();
        for k in 0..=self.uts.num_unroll {
            let mut input = LitVec::new();
            for var in self.uts.input() {
                let unrolled = self.uts.lit_next(var.lit(), k);
                let polarity = value(unrolled)?;
                input.push(var.lit().not_if(!polarity));
                inputs_flatten.push(unrolled.not_if(!polarity));
            }
            inputs.push(input);
        }

        self.slv.set_domain(cls.iter().copied());
        let mut states = LitVec::new();
        for var in self.uts.latch.iter().copied() {
            let lit = var.lit();
            if self.slv.domain_has(lit.var()) {
                let polarity = match value(lit) {
                    Some(polarity) => polarity,
                    None => {
                        self.slv.unset_domain();
                        return None;
                    }
                };
                states.push(lit.not_if(!polarity));
            }
        }
        for iteration in 0.. {
            // Always execute at least one exact implication check, including
            // the useful case where fixed inputs alone imply the target.
            // `order` controls additional core-shrinking passes, never whether
            // the external model is independently certified.
            let continue_after = !states.is_empty() && order(iteration, &mut states);
            let previous_len = states.len();
            states = match self.slv.minimal_premise(&inputs_flatten, &states, &cls) {
                Some(states) => states,
                None => {
                    self.slv.unset_domain();
                    return None;
                }
            };
            if !continue_after || states.is_empty() || states.len() == previous_len {
                break;
            }
        }
        self.slv.unset_domain();
        Some((states, inputs))
    }

    pub fn complex_lift(
        &mut self,
        satif: &mut impl Satif,
        state: impl IntoIterator<Item = impl AsRef<Var>>,
        target: impl IntoIterator<Item = impl AsRef<Lit>>,
        mut order: impl FnMut(usize, &mut [Lit]) -> bool,
    ) -> (LitVec, Vec<LitVec>) {
        let mut cls: LitVec = target.into_iter().map(|l| *l.as_ref()).collect();
        if cls.is_empty() {
            return (LitVec::new(), vec![]);
        }
        cls = !cls;
        let in_cls: GHashSet<Var> = GHashSet::from_iter(cls.iter().map(|l| l.var()));
        let mut inputs = Vec::new();
        let mut inputs_flatten = LitVec::new();
        for k in 0..=self.uts.num_unroll {
            let mut input = LitVec::new();
            for i in self.uts.input() {
                let lit = self.uts.lit_next(i.lit(), k);
                if let Some(v) = satif.sat_value(lit) {
                    input.push(i.lit().not_if(!v));
                    inputs_flatten.push(lit.not_if(!v));
                }
            }
            inputs.push(input);
        }
        self.slv.set_domain(cls.iter().cloned());
        let mut states = LitVec::new();
        for s in state.into_iter() {
            let s = *s.as_ref();
            let lit = s.lit();
            if self.slv.domain_has(lit.var())
                && let Some(v) = satif.sat_value(lit)
                && (in_cls.contains(&s) || !satif.flip_to_none(s))
            {
                states.push(lit.not_if(!v));
            }
        }
        for i in 0.. {
            if states.is_empty() {
                break;
            }
            if !order(i, &mut states) {
                break;
            }
            let olen = states.len();
            states = self
                .slv
                .minimal_premise(&inputs_flatten, &states, &cls)
                .unwrap();
            if states.len() == olen {
                break;
            }
        }
        self.slv.unset_domain();
        (states, inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::TsLift;
    use crate::transys::{Transys, TransysIf, unroll::TransysUnroll};
    use logicrs::Lit;

    #[test]
    fn external_model_is_reduced_only_by_an_exact_implication_check() {
        let mut ts = Transys::new();
        let input = ts.new_var();
        let relevant = ts.new_var();
        let irrelevant = ts.new_var();
        let output = ts.new_var();
        let next = ts.rel.new_and([input.lit(), relevant.lit()]);
        ts.add_input(input);
        ts.add_latch(relevant, None, relevant.lit());
        ts.add_latch(irrelevant, None, irrelevant.lit());
        ts.add_latch(output, None, next);

        let mut lift = TsLift::new(TransysUnroll::new(&ts));
        let model = [
            input.lit(),
            relevant.lit(),
            !irrelevant.lit(),
            output.lit(),
            next,
        ];
        let (state, inputs) = lift
            .lift_model(&model, [next], |iteration, _| iteration == 0)
            .unwrap();

        assert_eq!(state.as_slice(), &[relevant.lit()]);
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].as_slice(), &[input.lit()]);

        let missing_input: Vec<Lit> = model
            .iter()
            .copied()
            .filter(|lit| lit.var() != input)
            .collect();
        assert!(
            lift.lift_model(&missing_input, [next], |iteration, _| iteration == 0)
                .is_none()
        );

        let inconsistent = [
            !input.lit(),
            relevant.lit(),
            !irrelevant.lit(),
            output.lit(),
            next,
        ];
        assert!(
            lift.lift_model(&inconsistent, [next], |iteration, _| iteration == 0)
                .is_none()
        );
    }
}
