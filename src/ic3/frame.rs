use super::{IC3, proofoblig::ProofObligation};
use crate::transys::TransysCtx;
use giputils::hash::GHashSet;
use giputils::ptr::Grc;
use logicrs::{Lit, LitOrdVec, LitSet, LitVec, Var, satif::Satif};
use std::{
    cell::RefCell,
    fmt::Write,
    ops::{Deref, DerefMut, Index},
    vec,
};

#[derive(Clone)]
pub struct FrameLemma {
    lemma: LitOrdVec,
    pub po: Option<ProofObligation>,
    pub _ctp: Option<LitVec>,
}

/// Exact-replay journal entry produced at the real lemma mutation site.  The
/// BLOCK resident-controller simulation consumes these entries directly;
/// canonical frame images remain an independent correctness oracle only.
#[derive(Clone)]
pub(super) struct BlockLemmaMutation {
    pub insert: bool,
    pub frame: usize,
    pub lemma: LitVec,
}

thread_local! {
    static BLOCK_LEMMA_JOURNAL: RefCell<Option<Vec<BlockLemmaMutation>>> =
        const { RefCell::new(None) };
    // Mutations outside BLOCK used to be rediscovered by diffing the next
    // BLOCK root image.  Keep one direct semantic stream across the complete
    // extend/propagate/push-to-infinity maintenance wave instead.  The final
    // images remain an independent simulation oracle only.
    static FRAME_MAINTENANCE_JOURNAL: RefCell<Option<Vec<Vec<u32>>>> =
        const { RefCell::new(None) };
}

const RESIDENT_REMOVE_OBLIGATION: u32 = 0;
const RESIDENT_INSERT_OBLIGATION: u32 = 1;
const RESIDENT_CLEAR_OBLIGATIONS: u32 = 2;
const RESIDENT_REMOVE_LEMMA: u32 = 3;
const RESIDENT_INSERT_LEMMA: u32 = 4;
const RESIDENT_SET_LEMMA_FRAMES: u32 = 5;
const RESIDENT_MOVE_LEMMA: u32 = 6;

pub(super) fn begin_frame_maintenance_journal(enabled: bool) {
    FRAME_MAINTENANCE_JOURNAL.with(|journal| {
        *journal.borrow_mut() = enabled.then(Vec::new);
    });
}

pub(super) fn finish_frame_maintenance_journal() -> Vec<Vec<u32>> {
    FRAME_MAINTENANCE_JOURNAL.with(|journal| {
        journal.borrow_mut().take().unwrap_or_default()
    })
}

pub(super) fn frame_maintenance_journal_enabled() -> bool {
    FRAME_MAINTENANCE_JOURNAL.with(|journal| journal.borrow().is_some())
}

pub(super) fn note_frame_obligation_mutation(
    insert: bool,
    po: &ProofObligation,
) {
    FRAME_MAINTENANCE_JOURNAL.with(|journal| {
        let mut journal = journal.borrow_mut();
        let Some(journal) = journal.as_mut() else { return };
        let mut payload = Vec::new();
        payload.push(po.state.len().min(u32::MAX as usize) as u32);
        payload.extend(po.state.iter().map(|lit| u32::from(*lit)));
        payload.push(po.input.len().min(u32::MAX as usize) as u32);
        for inputs in &po.input {
            payload.push(inputs.len().min(u32::MAX as usize) as u32);
            payload.extend(inputs.iter().map(|lit| u32::from(*lit)));
        }
        let mut operation = vec![
            if insert {
                RESIDENT_INSERT_OBLIGATION
            } else {
                RESIDENT_REMOVE_OBLIGATION
            },
            po.frame.min(u32::MAX as usize) as u32,
            po.depth.min(u32::MAX as usize) as u32,
            u32::from(po.removed),
            payload.len().min(u32::MAX as usize) as u32,
        ];
        operation.extend(payload);
        journal.push(operation);
    });
}

pub(super) fn note_frame_clear_obligations() {
    FRAME_MAINTENANCE_JOURNAL.with(|journal| {
        if let Some(journal) = journal.borrow_mut().as_mut() {
            journal.push(vec![RESIDENT_CLEAR_OBLIGATIONS]);
        }
    });
}

fn note_frame_lemma_mutation(insert: bool, frame: usize, lemma: &LitOrdVec) {
    FRAME_MAINTENANCE_JOURNAL.with(|journal| {
        let mut journal = journal.borrow_mut();
        let Some(journal) = journal.as_mut() else { return };
        let mut operation = vec![
            if insert {
                RESIDENT_INSERT_LEMMA
            } else {
                RESIDENT_REMOVE_LEMMA
            },
            frame.min(u32::MAX as usize) as u32,
            lemma.len().min(u32::MAX as usize) as u32 + 1,
            lemma.len().min(u32::MAX as usize) as u32,
        ];
        operation.extend(lemma.iter().map(|lit| u32::from(*lit)));
        journal.push(operation);
    });
}

fn note_frame_lemma_move(source: usize, destination: usize, lemma: &LitOrdVec) {
    FRAME_MAINTENANCE_JOURNAL.with(|journal| {
        let mut journal = journal.borrow_mut();
        let Some(journal) = journal.as_mut() else { return };
        let mut operation = vec![
            RESIDENT_MOVE_LEMMA,
            source.min(u32::MAX as usize) as u32,
            destination.min(u32::MAX as usize) as u32,
            lemma.len().min(u32::MAX as usize) as u32 + 1,
            lemma.len().min(u32::MAX as usize) as u32,
        ];
        operation.extend(lemma.iter().map(|lit| u32::from(*lit)));
        journal.push(operation);
    });
}

fn note_frame_count(frame_count: usize) {
    FRAME_MAINTENANCE_JOURNAL.with(|journal| {
        if let Some(journal) = journal.borrow_mut().as_mut() {
            journal.push(vec![
                RESIDENT_SET_LEMMA_FRAMES,
                frame_count.min(u32::MAX as usize) as u32,
            ]);
        }
    });
}

pub(super) fn begin_block_lemma_journal(enabled: bool) {
    BLOCK_LEMMA_JOURNAL.with(|journal| {
        *journal.borrow_mut() = enabled.then(Vec::new);
    });
}

pub(super) fn finish_block_lemma_journal() -> Vec<BlockLemmaMutation> {
    BLOCK_LEMMA_JOURNAL.with(|journal| journal.borrow_mut().take().unwrap_or_default())
}

impl FrameLemma {
    #[inline]
    pub fn new(lemma: LitOrdVec, po: Option<ProofObligation>, ctp: Option<LitVec>) -> Self {
        Self {
            lemma,
            po,
            _ctp: ctp,
        }
    }
}

impl Deref for FrameLemma {
    type Target = LitOrdVec;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.lemma
    }
}

impl DerefMut for FrameLemma {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.lemma
    }
}

#[derive(Default, Clone)]
pub struct Frame {
    lemmas: Vec<FrameLemma>,
}

impl Frame {
    pub fn new() -> Self {
        Self { lemmas: Vec::new() }
    }
}

impl Deref for Frame {
    type Target = Vec<FrameLemma>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.lemmas
    }
}

impl DerefMut for Frame {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.lemmas
    }
}

impl IntoIterator for Frame {
    type Item = FrameLemma;
    type IntoIter = vec::IntoIter<FrameLemma>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.lemmas.into_iter()
    }
}

pub struct Frames {
    frames: Vec<Frame>,
    pub inf: Frame,
    pub early: usize,
    tmp_lit_set: LitSet,
}

impl Frames {
    pub fn new(ts: &Grc<TransysCtx>) -> Self {
        let mut tmp_lit_set = LitSet::new();
        tmp_lit_set.reserve(ts.max_latch);
        Self {
            frames: Default::default(),
            inf: Default::default(),
            early: 1,
            tmp_lit_set,
        }
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Count and fingerprint the lemma image consumed by BLOCK. This is a
    /// simulation oracle for a future FPGA-resident controller, not a proof
    /// hash: the live CPU structures remain authoritative.
    pub fn progress_snapshot(&self) -> (usize, u64) {
        const OFFSET: u64 = 0xcbf29ce484222325;
        const PRIME: u64 = 0x100000001b3;
        let mut hash = OFFSET;
        let mut count = 0usize;
        let mut word = |value: u64| {
            hash ^= value;
            hash = hash.wrapping_mul(PRIME);
        };
        word(self.frames.len() as u64);
        for (index, frame) in self.frames.iter().enumerate() {
            word(index as u64);
            word(frame.len() as u64);
            count += frame.len();
            for lemma in frame.iter() {
                word(lemma.len() as u64);
                for lit in lemma.iter() {
                    word(u64::from(u32::from(*lit)));
                }
            }
        }
        word(u64::MAX);
        word(self.inf.len() as u64);
        count += self.inf.len();
        for lemma in self.inf.iter() {
            word(lemma.len() as u64);
            for lit in lemma.iter() {
                word(u64::from(u32::from(*lit)));
            }
        }
        (count, hash)
    }

    /// Canonical word image paired with `progress_snapshot`. Frame indices are
    /// implicit in vector order; the final section contains infinite-frame
    /// lemmas. The exact replay interpreter decodes this independently.
    pub fn progress_image(&self) -> Vec<u32> {
        let mut words = Vec::new();
        words.push(self.frames.len().min(u32::MAX as usize) as u32);
        for frame in &self.frames {
            words.push(frame.len().min(u32::MAX as usize) as u32);
            for lemma in frame.iter() {
                words.push(lemma.len().min(u32::MAX as usize) as u32);
                words.extend(lemma.iter().map(|lit| u32::from(*lit)));
            }
        }
        words.push(self.inf.len().min(u32::MAX as usize) as u32);
        for lemma in self.inf.iter() {
            words.push(lemma.len().min(u32::MAX as usize) as u32);
            words.extend(lemma.iter().map(|lit| u32::from(*lit)));
        }
        words
    }

    pub fn last(&self) -> &Frame {
        self.frames.last().unwrap()
    }

    pub fn extend(&mut self) {
        self.frames.push(Frame::new());
        note_frame_count(self.frames.len());
    }

    pub fn reserve(&mut self, var: Var) {
        self.tmp_lit_set.reserve(var);
    }

    /// Check whether `lemma` is already syntactically subsumed by an existing lemma.
    ///
    /// If `frame` is `Some(i)`, search delta frames `i..` and then the infinite frame.
    /// If `frame` is `None`, search only the infinite frame.
    /// Returns the matched frame index (`None` for the infinite frame) together
    /// with a mutable reference to the matched lemma's proof obligation.
    #[inline]
    pub fn trivial_contained<'a>(
        &'a mut self,
        frame: Option<usize>,
        lemma: &LitOrdVec,
    ) -> Option<(Option<usize>, &'a mut Option<ProofObligation>)> {
        for l in lemma.iter() {
            self.tmp_lit_set.insert(*l);
        }
        if let Some(frame) = frame {
            for (i, fi) in self.frames.iter_mut().enumerate().skip(frame) {
                for j in 0..fi.len() {
                    if fi[j].lemma.subsume_set(lemma, &self.tmp_lit_set) {
                        self.tmp_lit_set.clear();
                        return Some((Some(i), &mut fi[j].po));
                    }
                }
            }
        }
        for j in 0..self.inf.len() {
            if self.inf[j].lemma.subsume_set(lemma, &self.tmp_lit_set) {
                self.tmp_lit_set.clear();
                return Some((None, &mut self.inf[j].po));
            }
        }
        self.tmp_lit_set.clear();
        None
    }

    pub fn parent_lemma(&self, lemma: &[Lit], frame: usize) -> Option<LitOrdVec> {
        if frame == 1 {
            return None;
        }
        let lemma = LitOrdVec::new(LitVec::from(lemma));
        for c in self.frames[frame - 1].iter() {
            if c.subsume(&lemma) {
                return Some(c.lemma.clone());
            }
        }
        None
    }

    pub fn _parent_lemmas(&self, lemma: &LitOrdVec, frame: usize) -> Vec<LitOrdVec> {
        let mut res = Vec::new();
        if frame == 1 {
            return res;
        }
        for c in self.frames[frame - 1].iter() {
            if c.subsume(lemma) {
                res.push(c.lemma.clone());
            }
        }
        res
    }

    #[allow(unused)]
    pub fn similar(&self, cube: &[Lit], frame: usize) -> Vec<LitVec> {
        let cube_set: GHashSet<Lit> = GHashSet::from_iter(cube.iter().copied());
        let mut res = GHashSet::new();
        for frame in self.frames[frame..].iter() {
            for lemma in frame.iter() {
                let sec: LitVec = lemma
                    .iter()
                    .filter(|l| cube_set.contains(l))
                    .copied()
                    .collect();
                if sec.len() != cube.len() && sec.len() * 2 >= cube.len() {
                    res.insert(sec);
                }
            }
        }
        let mut res = Vec::from_iter(res);
        res.sort_by_key(|x| x.len());
        res.reverse();
        if res.len() > 3 {
            res.truncate(3);
        }
        res
    }

    pub fn clear_po(&mut self) {
        for f in self.frames.iter_mut() {
            for l in f.iter_mut() {
                l.po = None;
            }
        }
    }

    #[inline]
    pub fn statistic(&self, compact: bool) -> String {
        const COMPACT_FRAME_LIMIT: usize = 50;
        let mut s = String::new();
        let total = self.frames.len() + 1;
        write!(s, "frames [{total}]: ").unwrap();
        let frames_iter: Box<dyn Iterator<Item = &Frame>> =
            if compact && total > COMPACT_FRAME_LIMIT {
                s.push_str("... ");
                Box::new(self.frames.iter().skip(total - COMPACT_FRAME_LIMIT))
            } else {
                Box::new(self.frames.iter())
            };
        for f in frames_iter {
            write!(s, "{} ", f.len()).unwrap();
        }
        write!(s, "{} ", self.inf.len()).unwrap();
        s
    }
}

impl Index<usize> for Frames {
    type Output = Frame;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.frames[index]
    }
}

impl IC3 {
    #[inline]
    pub(super) fn add_lemma(
        &mut self,
        frame: usize,
        lemma: LitVec,
        contained_check: bool,
        po: Option<ProofObligation>,
    ) -> bool {
        self.add_lemma_with_mutations(frame, lemma, contained_check, po, None)
    }

    #[inline]
    pub(super) fn add_lemma_with_mutations(
        &mut self,
        frame: usize,
        lemma: LitVec,
        contained_check: bool,
        po: Option<ProofObligation>,
        mut mutations: Option<&mut Vec<BlockLemmaMutation>>,
    ) -> bool {
        fn record(
            mutations: &mut Option<&mut Vec<BlockLemmaMutation>>,
            insert: bool,
            frame: usize,
            lemma: &LitOrdVec,
        ) {
            let mutation = BlockLemmaMutation {
                insert,
                frame,
                lemma: lemma.as_litvec().clone(),
            };
            note_frame_lemma_mutation(insert, frame, lemma);
            BLOCK_LEMMA_JOURNAL.with(|journal| {
                if let Some(journal) = journal.borrow_mut().as_mut() {
                    journal.push(mutation.clone());
                }
            });
            if let Some(mutations) = mutations.as_deref_mut() {
                mutations.push(mutation);
            }
        }
        let lemma = LitOrdVec::new(lemma);
        if frame == 0 {
            assert_eq!(self.frame.len(), 1);
            if self.level() == frame
                && let Some(predprop) = self.predprop.as_mut()
            {
                predprop.add_lemma(&lemma);
            }
            let clause = !lemma.as_litvec();
            crate::accel::cdcl_host::register_frame_resident_clause(&clause, 0, 0);
            self.solvers[0].add_clause(&clause);
            record(&mut mutations, true, 0, &lemma);
            self.frame.frames[0].push(FrameLemma::new(lemma, po, None));
            return false;
        }
        if contained_check && self.frame.trivial_contained(Some(frame), &lemma).is_some() {
            return false;
        }
        let mut begin = None;
        let mut inv_found = false;
        'fl: for i in (1..=frame).rev() {
            let mut j = 0;
            while j < self.frame[i].len() {
                let l = &self.frame[i][j];
                if begin.is_none() && l.subsume(&lemma) {
                    if l.eq(&lemma) {
                        let removed = self.frame.frames[i].swap_remove(j);
                        record(&mut mutations, false, i, &removed);
                        let clause = !lemma.as_litvec();
                        crate::accel::cdcl_host::register_frame_resident_clause(
                            &clause,
                            (i + 1) as u32,
                            frame as u32,
                        );
                        for k in i + 1..=frame {
                            self.solvers[k].add_clause(&clause);
                        }
                        if self.level() == frame
                            && let Some(predprop) = self.predprop.as_mut()
                        {
                            predprop.add_lemma(&lemma);
                        }
                        record(&mut mutations, true, frame, &lemma);
                        self.frame.frames[frame].push(FrameLemma::new(lemma, po, None));
                        self.frame.early = self.frame.early.min(i + 1);
                        return self.frame[i].is_empty();
                    } else {
                        begin = Some(i + 1);
                        break 'fl;
                    }
                }
                if lemma.subsume(l) {
                    let remove = self.frame.frames[i].swap_remove(j);
                    record(&mut mutations, false, i, &remove);
                    // self.solvers[i].remove_lemma(&remove);
                    continue;
                }
                j += 1;
            }
            if i != frame && self.frame[i].is_empty() {
                inv_found = true;
            }
        }
        let clause = !lemma.as_litvec();
        let begin = begin.unwrap_or(1);
        crate::accel::cdcl_host::register_frame_resident_clause(
            &clause,
            begin as u32,
            frame as u32,
        );
        // Mirrored once, with the frames it belongs to. The card holds one
        // clause set and each query names its frame, so `begin..=frame` is
        // exactly what the engine needs to decide whether this lemma applies.
        //
        // This used to mirror only the one solver the card was bound to, which
        // put a single frame's clause set on the card. Frame 1 is the sound
        // choice for that -- its lemmas are a subset of every frame's -- and on
        // cal97 it amounted to one lemma and settled none of 2033 unsat
        // queries.
        if crate::accel::ready() {
            let raw: Vec<u32> = clause.iter().map(|l| Into::<u32>::into(*l)).collect();
            if !crate::accel::add_lemma(&raw, begin as u32, frame as u32) {
                // Out of room. Unbind rather than carry on with a clause set
                // that silently differs from the solvers'.
                crate::accel::unbind();
            } else {
                crate::accel::mark_dirty();
            }
        }
        for i in begin..=frame {
            self.solvers[i].add_clause(&clause);
        }
        if self.level() == frame
            && let Some(predprop) = self.predprop.as_mut()
        {
            predprop.add_lemma(&lemma);
        }
        record(&mut mutations, true, frame, &lemma);
        self.frame.frames[frame].push(FrameLemma::new(lemma, po, None));
        self.frame.early = self.frame.early.min(begin);
        inv_found
    }

    pub(super) fn add_inf_lemma(&mut self, lemma: LitVec) {
        self.tracer.trace_lemma(
            &lemma
                .iter()
                .map(|l| !l.map_var(|v| self.rst.restore_var(v)))
                .collect(),
            None,
        );
        let lemma = LitOrdVec::new(lemma);
        assert!(self.frame.trivial_contained(None, &lemma).is_none());
        let last_frame = self.frame.frames.len() - 1;
        let lastf = self.frame.frames.last_mut().unwrap();
        let olen = lastf.len();
        lastf.retain(|l| !l.eq(&lemma));
        assert_eq!(lastf.len() + 1, olen);
        note_frame_lemma_move(last_frame, usize::MAX, &lemma);
        let clause = !lemma.as_litvec();
        crate::accel::cdcl_host::register_frame_resident_clause(
            &clause,
            0,
            u32::MAX,
        );
        // Mirror at every frame. An infinity lemma holds unconditionally, and
        // every frame solver is a clone of `inf_solver`, so a card that never
        // saw these holds strictly less than the solver it is shadowing. That
        // showed up as the card completing a search and answering satisfiable
        // on ~6800 queries the solver found unsat: not unsound, since only its
        // conflicts are relied on, but it is the reason it finds so few.
        if crate::accel::ready() {
            let raw: Vec<u32> = clause.iter().map(|l| Into::<u32>::into(*l)).collect();
            if !crate::accel::add_lemma(&raw, 0, 0xffff) {
                crate::accel::unbind();
            } else {
                crate::accel::mark_dirty();
            }
        }
        self.inf_solver.add_clause(&clause);
        self.frame.inf.push(FrameLemma::new(lemma, None, None));
    }

    pub fn inner_invariant(&mut self) -> Vec<LitVec> {
        let mut invariants: Vec<_> = self
            .frame
            .inf
            .iter()
            .map(|c| c.as_litvec().clone())
            .collect();
        if let Some(invariant) = self.frame.frames.iter().position(|frame| frame.is_empty()) {
            for i in invariant..self.frame.len() {
                for cube in self.frame[i].iter() {
                    invariants.push(cube.as_litvec().clone());
                }
            }
            invariants
        } else {
            self.propagate_to_inf2();
            self.frame
                .inf
                .iter()
                .map(|c| c.as_litvec().clone())
                .collect()
        }
    }
}
