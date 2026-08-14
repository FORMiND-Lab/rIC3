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
pub const ABI_VERSION: u32 = 1;

/// Return a sparse model over the variables assigned by the search.
pub const WANT_MODEL: u32 = 1 << 0;
/// Return the subset of assumptions used by an UNSAT result.
pub const WANT_CORE: u32 = 1 << 1;
/// Retain bounded learnt clauses in this frame's resident context.
pub const KEEP_LEARNTS: u32 = 1 << 2;

pub const QUERY_HEADER_WORDS: usize = 8;
pub const RESPONSE_HEADER_WORDS: usize = 9;

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
        if self.version != ABI_VERSION || self.payload_words() != Some(payload.len()) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_layout_is_fixed_and_payload_is_checked() {
        assert_eq!(std::mem::size_of::<QueryHeader>(), 8 * 4);
        assert_eq!(std::mem::size_of::<ResponseHeader>(), 9 * 4);
        assert_eq!(std::mem::size_of::<BatchHeader>(), 4 * 4);
        assert_eq!(std::mem::size_of::<BatchResponseHeader>(), 4 * 4);

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
    }
}
