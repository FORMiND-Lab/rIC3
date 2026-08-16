use crate::{
    BlCex, BlEngine, BlProof, Engine, McResult,
    config::{EngineConfig, EngineConfigBase, PreprocConfig},
    gipsat::{SolverStatistic, TransysSolver},
    ic3::{block::BlockResult, localabs::LocalAbs, predprop::PredProp},
    impl_config_deref,
    tracer::{Tracer, TracerIf},
    transys::{
        Transys, TransysCtx, TransysIf, certify::Restore, lift::TsLift, unroll::TransysUnroll,
    },
    ui::UiRenderer,
    utils::EngineCtrl,
};
use activity::Activity;
use clap::{ArgAction, Args, Parser};
use frame::Frames;
use giputils::{TerminateCtrl, logger::IntervalLogger, ptr::Grc};
use log::{Level, debug, error, info, trace};
use logicrs::{Lit, LitOrdVec, LitVec, LitVvec, Var, VarMap, VarSymbols, satif::Satif};
use proofoblig::{ProofObligation, ProofObligationQueue};
use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use std::{ops::Deref, sync::Arc, time::Instant};
use utils::Statistic;

mod activity;
mod auxv;
mod block;
mod frame;
mod localabs;
mod mab;
mod mic;
mod predprop;
mod proofoblig;
mod propagate;
mod push_prefetch;
mod solver;
mod ui;
mod utils;

#[derive(Args, Clone, Debug, Serialize, Deserialize)]
pub struct IC3Config {
    #[command(flatten)]
    pub base: EngineConfigBase,

    #[command(flatten)]
    pub preproc: PreprocConfig,

    /// dynamic generalization
    #[arg(long = "dynamic", default_value_t = false)]
    pub dynamic: bool,

    /// contextual-MAB (LinUCB) adaptive generalization (A-IC3)
    #[arg(long = "mab", default_value_t = false)]
    pub mab: bool,

    /// LinUCB exploration parameter alpha
    #[arg(long = "mab-alpha", default_value_t = 1.0)]
    pub mab_alpha: f64,

    /// LinUCB regularization parameter lambda
    #[arg(long = "mab-lambda", default_value_t = 0.1)]
    pub mab_lambda: f64,

    /// counterexample to generalization
    #[arg(long = "ctg", action = ArgAction::Set, default_value_t = true)]
    pub ctg: bool,

    /// max number of ctg
    #[arg(long = "ctg-max", default_value_t = 3)]
    pub ctg_max: usize,

    /// ctg limit
    #[arg(long = "ctg-limit", default_value_t = 1)]
    pub ctg_limit: usize,

    /// counterexample to propagation
    #[arg(long = "ctp", default_value_t = false)]
    pub ctp: bool,

    /// internal signals (FMCAD'21 https://doi.org/10.34727/2021/isbn.978-3-85448-046-4_14)
    #[arg(long = "inn", default_value_t = false)]
    pub inn: bool,

    /// abstract constrains
    #[arg(long = "abs-cst", default_value_t = false)]
    pub abs_cst: bool,

    /// abstract trans
    #[arg(long = "abs-trans", default_value_t = false)]
    pub abs_trans: bool,

    /// dropping proof-obligation
    #[arg(
        long = "drop-po", action = ArgAction::Set, default_value_t = true,
    )]
    pub drop_po: bool,

    /// full assignment of last bad (internal parameter)
    #[arg(skip)]
    pub full_bad: bool,

    /// abstract array
    #[arg(long = "abs-array", default_value_t = false)]
    pub abs_array: bool,

    /// finding parent lemma in mic (CAV'23 https://doi.org/10.1007/978-3-031-37703-7_14)
    #[arg(long = "parent-lemma", action = ArgAction::Set, default_value_t = true)]
    pub parent_lemma: bool,

    /// predicate property
    #[arg(long = "pred-prop", default_value_t = false)]
    pub pred_prop: bool,

    /// Local proof (internal parameter)
    #[arg(skip)]
    pub local_proof: bool,
}

impl_config_deref!(IC3Config);

impl Default for IC3Config {
    fn default() -> Self {
        let cfg = EngineConfig::parse_from(["", "ic3"]);
        cfg.into_ic3().unwrap()
    }
}

impl IC3Config {
    fn validate(&self) {
        if self.dynamic && self.drop_po {
            error!("cannot enable both dynamic and drop-po");
            panic!();
        }
        if self.mab && self.drop_po {
            error!("cannot enable both mab and drop-po");
            panic!();
        }
        if self.inn {
            let pre = "cannot enable both inn and";
            if self.abs_cst || self.abs_trans {
                error!("{pre} (abs_cst or abs_trans)");
                panic!();
            }
            if self.pred_prop {
                error!("{pre} pred-prop");
                panic!();
            }
        }
        if self.full_bad {
            error!("full-bad can't be used now");
            panic!();
        }
        if self.local_proof {
            if !self.pred_prop {
                error!("local-proof should used with pred-prop");
                panic!();
            }
            if self.prop.is_none() {
                error!("A property ID must be specified for local proof.");
                panic!();
            }
        }
    }
}

pub struct IC3 {
    cfg: IC3Config,
    ts: Grc<Transys>,
    #[allow(unused)]
    symbols: VarSymbols,
    tsctx: Grc<TransysCtx>,
    solvers: Vec<TransysSolver>,
    inf_solver: TransysSolver,
    ts_top_lv: VarMap<usize>,
    lift: TsLift,
    frame: Frames,
    obligations: ProofObligationQueue,
    activity: Activity,
    statistic: Statistic,
    localabs: LocalAbs,
    ots: Transys,
    rst: Restore,
    auxiliary_var: Vec<Var>,
    predprop: Option<PredProp>,
    mab: mab::CtxMab,
    block_accel_policy: block::BlockAccelPolicy,
    push_prefetch: push_prefetch::PushPrefetchCache,

    rng: StdRng,
    filog: IntervalLogger,
    tracer: Tracer,
    ctrl: Arc<EngineCtrl>,
    renderer: Option<UiRenderer>,
}

impl IC3 {
    #[inline]
    pub fn level(&self) -> usize {
        self.solvers.len() - 1
    }

    fn extend(&mut self) {
        let nl = self.solvers.len();
        debug!("extending IC3 to level {nl}");
        if let Some(predprop) = self.predprop.as_mut() {
            predprop.extend(self.frame.inf.iter().map(|l| l.as_litvec()));
        }
        let mut solver = self.inf_solver.clone();
        // A clone is a new solver and needs its own identity, or one id names
        // every frame and the card mirrors all of their lemmas.
        solver.dcs.accel_id = crate::gipsat::DagCnfSolver::fresh_accel_id();
        self.solvers.push(solver);
        // Every solver knows its frame. The card holds all frames' lemmas
        // with the range each is valid over, so no binding is needed: a query
        // names its frame and the engine skips what that frame does not hold.
        let frontier = self.solvers.len() - 1;
        self.solvers[frontier].dcs.accel_level = frontier as u32;
        self.frame.extend();
        if self.level() == 0 {
            for init in self.tsctx.init.clone() {
                self.add_lemma(0, !init, true, None);
            }
            let init: LitVec = self
                .tsctx
                .latch
                .iter()
                .filter(|l| self.tsctx.init_map[**l].is_none())
                .filter_map(|l| {
                    self.solvers[0]
                        .sat_value(l.lit())
                        .map(|v| l.lit().not_if(!v))
                })
                .collect();
            for i in init {
                self.ts.add_init(i.var(), Lit::constant(i.polarity()));
                self.tsctx.add_init(i.var(), Lit::constant(i.polarity()));
            }
        }
    }
}

impl IC3 {
    pub fn new(cfg: IC3Config, mut ts: Transys, symbols: VarSymbols) -> Self {
        cfg.validate();
        let ots = ts.clone();
        if let Some(prop) = cfg.prop {
            if !cfg.local_proof {
                ts.bad = LitVec::from(ts.bad[prop]);
            }
        } else {
            ts.compress_bads();
        }
        let rst = Restore::new(&ts);
        let rng = StdRng::seed_from_u64(cfg.rseed);
        let statistic = Statistic::default();
        let (mut ts, mut rst) = ts.preproc(&cfg.preproc, rst);
        ts.remove_gate_init(&mut rst);
        let ts_top_lv = ts.rel.level();
        if cfg.inn {
            let mut u = TransysUnroll::new(&ts);
            u.unroll();
            ts = u.internal_signals();
        }
        let ts = Grc::new(ts);
        let predprop = if cfg.pred_prop {
            let mut uts = TransysUnroll::new(ts.deref());
            uts.unroll();
            Some(PredProp::new(
                uts,
                cfg.local_proof.then(|| cfg.prop.unwrap()),
            ))
        } else {
            None
        };
        let tsctx = Grc::new(ts.ctx());
        let activity = Activity::new(&tsctx);
        let frame = Frames::new(&tsctx);
        let inf_solver = TransysSolver::new(&tsctx);
        let lift = TsLift::new(TransysUnroll::new(&ts));
        let localabs = LocalAbs::new(&ts, &cfg);
        let mab = mab::CtxMab::new(cfg.mab_alpha, cfg.mab_lambda);
        Self {
            cfg,
            ts,
            symbols,
            tsctx,
            activity,
            solvers: Vec::new(),
            inf_solver,
            lift,
            ts_top_lv,
            statistic,
            obligations: ProofObligationQueue::new(),
            frame,
            localabs,
            auxiliary_var: Vec::new(),
            ots,
            rst,
            predprop,
            mab,
            block_accel_policy: Default::default(),
            push_prefetch: Default::default(),
            rng,
            filog: Default::default(),
            tracer: Tracer::new(),
            ctrl: Arc::new(EngineCtrl::new()),
            renderer: None,
        }
    }

    pub fn invariant(&mut self) -> Vec<LitVec> {
        self.inner_invariant()
            .iter()
            .map(|l| l.map_var(|l| self.rst.restore_var(l)))
            .collect()
    }
}

impl IC3 {
    /// Inductor: the real check loop. `Engine::check` wraps this so the query
    /// trace is opened and closed exactly once no matter which of the many
    /// exits below fires.
    fn check_traced(&mut self) -> McResult {
        if !self.prep_prop_base() {
            self.tracer.trace_state(None, McResult::SAT(0));
            self.finish_progress(McResult::SAT(0));
            return McResult::SAT(0);
        }
        self.extend();
        self.render_progress();
        loop {
            let start = Instant::now();
            debug!("blocking phase begin");
            loop {
                let lvl = self.level();
                let terminal = match crate::inductor::in_phase(
                    inductor_trace::Phase::Block,
                    lvl,
                    || self.block(None),
                ) {
                    BlockResult::Failure(depth) => Some(McResult::SAT(depth)),
                    BlockResult::Proved => Some(McResult::UNSAT),
                    BlockResult::OverallTimeLimitExceeded => {
                        Some(McResult::Unknown(Some(self.level())))
                    }
                    _ => None,
                };
                if let Some(result) = terminal {
                    self.statistic.block.overall_time += start.elapsed();
                    if !matches!(result, McResult::Unknown(_)) {
                        self.tracer.trace_state(None, result);
                    }
                    self.finish_progress(result);
                    return result;
                }
                if let Some((bad, inputs)) = self.get_bad() {
                    debug!("bad state found in frame {}", self.level());
                    trace!("bad = {bad}");
                    let bad = LitOrdVec::new(bad);
                    let depth = inputs.len() - 1;
                    self.add_obligation(ProofObligation::new(
                        self.level(),
                        bad,
                        inputs,
                        depth,
                        None,
                    ))
                } else {
                    break;
                }
            }
            debug!("blocking phase end");
            self.statistic.block.overall_time += start.elapsed();
            self.filog.log(Level::Info, self.frame.statistic(true));
            self.tracer
                .trace_state(None, McResult::Unknown(Some(self.level())));
            self.extend();
            self.render_progress();
            let start = Instant::now();
            let lvl = self.level();
            let propagate = crate::inductor::in_phase(
                inductor_trace::Phase::Push,
                lvl,
                || self.propagate(None),
            );
            self.statistic.propagate.overall_time += start.elapsed();
            if propagate {
                self.tracer.trace_state(None, McResult::UNSAT);
                self.finish_progress(McResult::UNSAT);
                return McResult::UNSAT;
            }
            self.propagate_to_inf();
            self.render_progress();
        }
    }
}

impl Engine for IC3 {
    fn check(&mut self) -> McResult {
        let shape = {
            let rel = &self.tsctx.rel;
            let gates = rel.var_iter().filter(|v| !rel.is_leaf(*v)).count() as u64;
            // Size of the inverted fanin lists, counted exactly as the
            // gate-implication path walks them: v contributes one entry to each
            // of its fanins, plus one self-entry for the gate it defines. This
            // must stay in step with how `fanout_len` is built in gipsat/mod.rs.
            let deps: u64 = rel.var_iter().map(|v| rel.dep(v).len() as u64).sum();

            // Gate shape, as the HLS evaluator has to unroll over it. Clauses
            // are grouped by defining variable, which is exactly the grouping
            // the hardware's per-gate record uses, so this counts the same
            // thing the kernel would.
            let mut max_gate_clauses = 0u32;
            let mut max_gate_slots = 0u32;
            let mut max_clause_len = 0u32;
            let mut n_gate_unfit = 0u32;
            let mut gate_clause_hist = [0u32; 6];
            let mut visit_clause_hist = [0u64; 6];
            let mut pool_words = 0u64;
            let mut total_lits = 0u64;
            let mut pool_words_aligned = [0u64; 3];
            let mut slots: Vec<Var> = Vec::with_capacity(8);
            for (_v, cls_of_gate) in rel.iter() {
                if cls_of_gate.is_empty() {
                    continue;
                }
                let n_cls = cls_of_gate.len() as u32;
                if n_cls > max_gate_clauses {
                    max_gate_clauses = n_cls;
                }
                slots.clear();
                let mut longest = 0u32;
                for cls in cls_of_gate.iter() {
                    let l = cls.len() as u32;
                    if l > longest {
                        longest = l;
                    }
                    // The pool stores each clause as one length word followed by
                    // its literals. The aligned variants pad the literal run to
                    // a lane boundary, which is what would make a lane's bank
                    // index a compile-time constant.
                    total_lits += l as u64;
                    pool_words += 1 + l as u64;
                    for (idx, lanes) in [4u64, 8, 16].iter().enumerate() {
                        let padded = (l as u64).div_ceil(*lanes) * lanes;
                        pool_words_aligned[idx] += 1 + padded;
                    }
                    for lit in cls.iter() {
                        if !slots.contains(&lit.var()) {
                            slots.push(lit.var());
                        }
                    }
                }
                if longest > max_clause_len {
                    max_clause_len = longest;
                }
                if slots.len() as u32 > max_gate_slots {
                    max_gate_slots = slots.len() as u32;
                }
                // The kernel's fixed-size record holds 6 clauses over 4 distinct
                // variables with at most 3 literals each. Past any of those, the
                // gate cannot be stored as one entry.
                if n_cls > 6 || slots.len() > 4 || longest > 3 {
                    n_gate_unfit += 1;
                }
                // A gate sits in the fanout list of every variable it mentions,
                // so its fanout degree -- how often propagation has to evaluate
                // it -- is exactly its distinct-variable count. Weighting by
                // that gives the size distribution the datapath actually sees,
                // as opposed to the one the netlist merely contains.
                let bucket = match n_cls {
                    0..=4 => 0,
                    5..=6 => 1,
                    7..=8 => 2,
                    9..=16 => 3,
                    17..=64 => 4,
                    _ => 5,
                };
                gate_clause_hist[bucket] += 1;
                visit_clause_hist[bucket] += slots.len() as u64;
            }

            crate::inductor::NetlistShape {
                n_var: rel.num_var() as u32,
                n_clause: rel.num_clause() as u32,
                n_gate: gates as u32,
                n_fanout_total: deps + gates,
                max_gate_clauses,
                max_gate_slots,
                max_clause_len,
                n_gate_unfit,
                gate_clause_hist,
                visit_clause_hist,
                pool_words,
                total_lits,
                pool_words_aligned,
            }
        };
        crate::inductor::dump_netlist(&self.tsctx.rel);
        // Bring the card up on the same transition relation the solver is about
        // to use. Shadow mode from here: the solver answers its own queries and
        // the card is asked the same ones, so the state on the card is the
        // state the solver has -- which a replay cannot guarantee (7v).
        if let Some(path) = crate::accel::xclbin() {
            let (n_var, flat) = crate::inductor::netlist_flat(&self.tsctx.rel);
            match crate::accel::init(&path, n_var, &flat) {
                Ok(()) => {
                    log::info!("inductor: accelerator ready on {path}, {n_var} vars");
                    // Binding happens in `extend`, where frame 1's solver is
                    // created. Here the vector is still empty -- solvers are
                    // built as the run extends -- and binding on an empty
                    // vector is why the first attempt reported solver 0 and
                    // never called the card.
                }
                Err(e) => log::warn!("inductor: accelerator unavailable ({e}); CPU only"),
            }
        }
        crate::inductor::init("", shape);
        let t0 = Instant::now();
        let res = self.check_traced();
        let name = match res {
            McResult::UNSAT => "safe",
            McResult::SAT(_) => "unsafe",
            McResult::Unknown(_) => "unknown",
        };
        // One solver per frame, each with its own lemma set: `add_lemma` writes
        // a clause into `solvers[begin..=frame]`. A replay that loads one
        // snapshot is loading one of these, while the queries it replays came
        // from many. This reports how far apart those are.
        {
            let per: Vec<(usize, u64)> = self
                .solvers
                .iter()
                .map(|s| s.dcs.lemma_size())
                .collect();
            crate::inductor::report_solver_fanout(self.solvers.len(), &per);
        }
        self.push_prefetch.finish();
        crate::accel::report();
        crate::inductor::finish(
            name,
            t0.elapsed().as_nanos() as u64,
            (
                self.statistic.mic_drop.succ() as u64,
                self.statistic.mic_drop.total() as u64,
            ),
            self.statistic.num_down as u64,
            self.statistic.num_down_sat as u64,
        );
        res
    }

    fn add_tracer(&mut self, tracer: Box<dyn TracerIf>) {
        self.tracer.add_tracer(tracer);
    }

    fn set_ui(&mut self, renderer: UiRenderer) {
        self.renderer = Some(renderer);
    }

    fn statistic(&mut self) {
        self.statistic.num_auxiliary_var = self.auxiliary_var.len();
        if self.cfg.mab {
            info!("{}", self.mab.statistic());
        }
        info!("obligations: {}", self.obligations.statistic());
        info!("{}", self.frame.statistic(false));
        let statistic = self
            .solvers
            .iter()
            .fold(SolverStatistic::default(), |mut acc, s| {
                acc += *s.statistic();
                acc
            });
        info!("{statistic:#?}");
        info!("{:#?}", self.statistic);
    }

    fn get_ctrl(&self) -> Arc<dyn TerminateCtrl> {
        self.ctrl.clone()
    }
}

impl BlEngine for IC3 {
    fn proof(&mut self) -> BlProof {
        let mut proof = self.ots.clone();
        if let Some(iv) = self.rst.init_var() {
            let piv = proof.add_init_var();
            self.rst.add_restore(iv, piv);
        }
        let mut invariants = self.inner_invariant();
        for c in self.ts.constraint.clone() {
            proof
                .rel
                .migrate(&self.ts.rel, c.var(), &mut self.rst.bvmap);
            invariants.push(LitVec::from(!c));
        }
        let mut invariants: LitVvec = invariants
            .iter()
            .map(|l| LitVec::from_iter(l.iter().map(|l| self.rst.restore(*l))))
            .collect();
        invariants.extend(self.rst.eq_invariant());
        let certifaiger_dnf: Vec<_> = invariants
            .into_iter()
            .map(|c| proof.rel.new_and(c))
            .collect();
        let invariants = proof.rel.new_or(certifaiger_dnf);
        let bad = proof.rel.new_or(proof.bad);
        proof.bad = LitVec::from(proof.rel.new_or([invariants, bad]));
        BlProof { proof }
    }

    fn cex(&mut self) -> BlCex {
        let mut res = if let Some(res) = self.localabs.cex() {
            res
        } else {
            let mut res = BlCex::default();
            let b = self.obligations.peak().unwrap();
            assert!(b.frame == 0);
            let mut b = Some(b);
            while let Some(bad) = b {
                res.state.push(bad.state.as_litvec().clone());
                res.input.push(bad.input[0].clone());
                for i in &bad.input[1..] {
                    res.input.push(i.clone());
                    res.state.push(LitVec::new());
                }
                b = bad.next.clone();
            }
            res
        };
        let iv = self.rst.init_var();
        res = res.filter_map(|l| {
            (iv != Some(l.var()))
                .then(|| self.rst.try_restore(l))
                .flatten()
        });
        for s in res.state.iter_mut() {
            *s = self.rst.restore_eq_state(s);
        }
        res.exact_state(&self.ots, true);
        res
    }
}
