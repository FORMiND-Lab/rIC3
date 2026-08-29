use super::IC3;
use giputils::ptr::Grc;
use log::trace;
use logicrs::{LitOrdVec, LitVec};
use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap, btree_set};
use std::fmt::{self, Debug};
use std::ops::{Deref, DerefMut};

#[derive(Default)]
pub struct ProofObligationInner {
    pub frame: usize,
    pub input: Vec<LitVec>,
    pub state: LitOrdVec,
    pub depth: usize,
    pub next: Option<ProofObligation>,
    pub removed: bool,
    pub act: f64,
}

impl PartialEq for ProofObligationInner {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.state == other.state && self.removed == other.removed
    }
}

impl Eq for ProofObligationInner {}

impl PartialOrd for ProofObligationInner {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ProofObligationInner {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        match other.frame.cmp(&self.frame) {
            Ordering::Equal => match self.depth.cmp(&other.depth) {
                Ordering::Equal => match other.state.len().cmp(&self.state.len()) {
                    Ordering::Equal => match other.state.cmp(&self.state) {
                        Ordering::Equal => self.removed.cmp(&other.removed),
                        ord => ord,
                    },
                    ord => ord,
                },
                ord => ord,
            },
            ord => ord,
        }
    }
}

impl Debug for ProofObligationInner {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProofObligation")
            .field("frame", &self.frame)
            .field("lemma", &self.state)
            .field("depth", &self.depth)
            .finish()
    }
}

#[derive(Clone, Default)]
pub struct ProofObligation {
    inner: Grc<ProofObligationInner>,
}

impl ProofObligation {
    pub fn new(
        frame: usize,
        lemma: LitOrdVec,
        input: Vec<LitVec>,
        depth: usize,
        next: Option<Self>,
    ) -> Self {
        Self {
            inner: Grc::new(ProofObligationInner {
                frame,
                input,
                state: lemma,
                depth,
                next,
                removed: false,
                act: 0.0,
            }),
        }
    }

    pub fn bump_act(&mut self) {
        self.act += 1.0;
    }

    pub fn push_to(&mut self, frame: usize) {
        for _ in self.frame..frame {
            self.act *= 0.6;
        }
        self.frame = frame;
    }

    fn resident_key(&self) -> Vec<u32> {
        let mut key = vec![
            self.frame.min(u32::MAX as usize) as u32,
            self.depth.min(u32::MAX as usize) as u32,
            u32::from(self.removed),
            self.state.len().min(u32::MAX as usize) as u32,
        ];
        key.extend(self.state.iter().map(|lit| u32::from(*lit)));
        key.push(self.input.len().min(u32::MAX as usize) as u32);
        for inputs in &self.input {
            key.push(inputs.len().min(u32::MAX as usize) as u32);
            key.extend(inputs.iter().map(|lit| u32::from(*lit)));
        }
        key
    }
}

impl Deref for ProofObligation {
    type Target = ProofObligationInner;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for ProofObligation {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl PartialEq for ProofObligation {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl Eq for ProofObligation {}

impl PartialOrd for ProofObligation {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ProofObligation {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.inner.cmp(&other.inner)
    }
}

impl Debug for ProofObligation {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

#[derive(Default, Debug)]
pub struct ProofObligationQueue {
    obligations: BTreeSet<ProofObligation>,
    resident_index: HashMap<Vec<u32>, ProofObligation>,
    num: Vec<usize>,
}

impl ProofObligationQueue {
    pub fn new() -> Self {
        Self::default()
    }

    /// Compact deterministic oracle for a resident BLOCK-program simulation.
    /// The fingerprint is observational only; it never participates in proof
    /// decisions. BTreeSet order makes equal queues reproduce the same value.
    pub fn progress_snapshot(&self) -> (usize, u64) {
        const OFFSET: u64 = 0xcbf29ce484222325;
        const PRIME: u64 = 0x100000001b3;
        let mut hash = OFFSET;
        let mut word = |value: u64| {
            hash ^= value;
            hash = hash.wrapping_mul(PRIME);
        };
        word(self.obligations.len() as u64);
        for po in &self.obligations {
            word(po.frame as u64);
            word(po.depth as u64);
            word(u64::from(po.removed));
            word(po.state.len() as u64);
            for lit in po.state.iter() {
                word(u64::from(u32::from(*lit)));
            }
            word(po.input.len() as u64);
            for inputs in &po.input {
                word(inputs.len() as u64);
                for lit in inputs {
                    word(u64::from(u32::from(*lit)));
                }
            }
        }
        (self.obligations.len(), hash)
    }

    /// Canonical word image paired with `progress_snapshot`. The image is
    /// trace-only input for an independent resident-controller interpreter;
    /// it is never consumed by the proof algorithm itself.
    pub fn progress_image(&self) -> Vec<u32> {
        let mut words = Vec::new();
        words.push(self.obligations.len().min(u32::MAX as usize) as u32);
        for po in &self.obligations {
            words.push(po.frame.min(u32::MAX as usize) as u32);
            words.push(po.depth.min(u32::MAX as usize) as u32);
            words.push(u32::from(po.removed));
            words.push(po.state.len().min(u32::MAX as usize) as u32);
            words.extend(po.state.iter().map(|lit| u32::from(*lit)));
            words.push(po.input.len().min(u32::MAX as usize) as u32);
            for inputs in &po.input {
                words.push(inputs.len().min(u32::MAX as usize) as u32);
                words.extend(inputs.iter().map(|lit| u32::from(*lit)));
            }
        }
        words
    }

    pub fn add(&mut self, po: ProofObligation) {
        if self.num.len() <= po.frame {
            self.num.resize(po.frame + 1, 0);
        }
        self.num[po.frame] += 1;
        trace!("add obligation: {}", po.state);
        super::frame::note_frame_obligation_mutation(true, &po);
        let key = po.resident_key();
        assert!(self.obligations.insert(po.clone()));
        assert!(self.resident_index.insert(key, po).is_none());
    }

    pub fn add_if_new(&mut self, po: ProofObligation) -> bool {
        let frame = po.frame;
        trace!("add obligation if new: {}", po.state);
        if !self.obligations.insert(po.clone()) {
            return false;
        }
        if super::frame::frame_maintenance_journal_enabled() {
            super::frame::note_frame_obligation_mutation(true, &po);
        }
        assert!(self.resident_index.insert(po.resident_key(), po).is_none());
        if self.num.len() <= frame {
            self.num.resize(frame + 1, 0);
        }
        self.num[frame] += 1;
        true
    }

    pub fn pop(&mut self, depth: usize) -> Option<ProofObligation> {
        if let Some(po) = self.obligations.last().filter(|po| po.frame <= depth) {
            self.num[po.frame] -= 1;
            let popped = self.obligations.pop_last();
            if let Some(po) = &popped {
                assert!(self.resident_index.remove(&po.resident_key()).is_some());
                super::frame::note_frame_obligation_mutation(false, po);
            }
            popped
        } else {
            None
        }
    }

    /// Simulation bridge for FPGA-owned work scheduling. The controller has
    /// already removed this descriptor; find the matching CPU proof-chain
    /// object without imposing the CPU BTreeSet's choice on the device.
    fn take_resident_key(&mut self, key: &[u32], max_frame: usize) -> Option<ProofObligation> {
        let selected = self
            .resident_index
            .get(key)
            .filter(|po| po.frame <= max_frame)?
            .clone();
        let ret = self.obligations.take(&selected);
        if let Some(taken) = &ret {
            assert!(self.resident_index.remove(key).is_some());
            self.num[taken.frame] -= 1;
            super::frame::note_frame_obligation_mutation(false, taken);
        }
        ret
    }

    pub fn clone_resident_key(&self, key: &[u32], max_frame: usize) -> Option<ProofObligation> {
        self.resident_index
            .get(key)
            .filter(|po| po.frame <= max_frame)
            .cloned()
    }

    /// Consume a controller-selected proof chain through its opaque tag. The
    /// simulation adapter resolves the tag; the queue lookup itself is indexed
    /// and does not scan or serialize every resident obligation.
    pub fn take_resident_tag(
        &mut self,
        user_tag: u64,
        max_frame: usize,
    ) -> Option<ProofObligation> {
        let key = crate::accel::cdcl_host::take_resident_block_selection(user_tag)?;
        self.take_resident_key(&key, max_frame)
    }

    pub fn peak(&mut self) -> Option<ProofObligation> {
        self.obligations.last().cloned()
    }

    pub fn remove(&mut self, po: &ProofObligation) -> bool {
        let ret = self.obligations.take(po);
        if let Some(taken) = &ret {
            assert!(self.resident_index.remove(&taken.resident_key()).is_some());
            self.num[taken.frame] -= 1;
            super::frame::note_frame_obligation_mutation(false, taken);
        }
        ret.is_some()
    }

    pub fn take(&mut self, po: &ProofObligation) -> Option<ProofObligation> {
        let ret = self.obligations.take(po);
        if let Some(taken) = &ret {
            assert!(self.resident_index.remove(&taken.resident_key()).is_some());
            self.num[taken.frame] -= 1;
            super::frame::note_frame_obligation_mutation(false, taken);
        }
        ret
    }

    pub fn contains(&self, po: &ProofObligation) -> bool {
        self.obligations.contains(po)
    }

    pub fn clear(&mut self) {
        if !self.obligations.is_empty() {
            super::frame::note_frame_clear_obligations();
        }
        self.obligations.clear();
        self.resident_index.clear();
        for n in self.num.iter_mut() {
            *n = 0;
        }
    }

    pub fn clear_to(&mut self, frame: usize) {
        while self.pop(frame).is_some() {}
    }

    #[allow(unused)]
    pub fn iter(&self) -> btree_set::Iter<'_, ProofObligation> {
        self.obligations.iter()
    }

    pub fn statistic(&self) -> String {
        format!("{:?}", self.num)
    }
}

impl IC3 {
    pub(super) fn add_obligation(&mut self, po: ProofObligation) {
        self.statistic.avg_po_cube_len += po.state.len();
        self.obligations.add(po)
    }

    pub(super) fn add_obligation_if_new(&mut self, po: ProofObligation) -> bool {
        let cube_len = po.state.len();
        if !self.obligations.add_if_new(po) {
            return false;
        }
        self.statistic.avg_po_cube_len += cube_len;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::{ProofObligation, ProofObligationQueue};
    use logicrs::{Lit, LitOrdVec, LitVec, Var};

    #[test]
    fn conditional_insert_keeps_frame_counts_consistent() {
        let lit = Lit::new(Var::from(0), true);
        let po = ProofObligation::new(2, LitOrdVec::new(LitVec::from([lit])), Vec::new(), 1, None);
        let mut queue = ProofObligationQueue::new();

        assert!(queue.add_if_new(po.clone()));
        assert!(!queue.add_if_new(po));
        assert_eq!(queue.num[2], 1);
        assert!(queue.pop(2).is_some());
        assert_eq!(queue.num[2], 0);
    }

    #[test]
    fn resident_selection_can_override_cpu_btree_tie_break() {
        let a = Lit::new(Var::from(0), true);
        let b = Lit::new(Var::from(1), true);
        let po_a = ProofObligation::new(2, LitOrdVec::new(LitVec::from([a])), Vec::new(), 1, None);
        let po_b = ProofObligation::new(2, LitOrdVec::new(LitVec::from([b])), Vec::new(), 1, None);
        let key_b = po_b.resident_key();
        let mut queue = ProofObligationQueue::new();
        queue.add(po_a.clone());
        queue.add(po_b.clone());

        let selected = queue.take_resident_key(&key_b, 2).unwrap();
        assert_eq!(selected.state, po_b.state);
        assert!(queue.contains(&po_a));
        assert!(!queue.contains(&po_b));
        assert_eq!(queue.num[2], 1);
    }
}
