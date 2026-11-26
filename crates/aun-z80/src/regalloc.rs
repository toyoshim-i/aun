//! Z80 register-first local allocator (`B`, `C`, `BC`) and interprocedural
//! register clobber analysis (IPRA).

use crate::analysis::FnAnalysis;
use aun_ir::*;
use std::collections::{BTreeSet, HashMap};

/// Bitmask of Z80 registers tracked for local allocation and IPRA.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RegMask(pub u8);

impl RegMask {
    pub const NONE: Self = Self(0);
    pub const B: Self = Self(1 << 0);
    pub const C: Self = Self(1 << 1);
    pub const BC: Self = Self(Self::B.0 | Self::C.0);
    pub const IX: Self = Self(1 << 2);

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    pub const fn intersects(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// Hardware register location assigned to a promotable local slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegLoc {
    C,
    B,
    BC,
}

impl RegLoc {
    pub const fn mask(self) -> RegMask {
        match self {
            RegLoc::C => RegMask::C,
            RegLoc::B => RegMask::B,
            RegLoc::BC => RegMask::BC,
        }
    }

    pub const fn byte_size(self) -> u16 {
        match self {
            RegLoc::C | RegLoc::B => 1,
            RegLoc::BC => 2,
        }
    }

    pub const fn reg8_name(self) -> &'static str {
        match self {
            RegLoc::B => "b",
            RegLoc::C | RegLoc::BC => "c",
        }
    }
}

/// Per-function register allocation assignments and per-call live masks.
#[derive(Debug, Clone, Default)]
pub struct FnRegAlloc {
    pub slot_regs: HashMap<SlotId, RegLoc>,
    pub call_live_masks: HashMap<(usize, usize), RegMask>,
}

impl FnRegAlloc {
    pub fn slot_reg(&self, slot: SlotId) -> Option<RegLoc> {
        self.slot_regs.get(&slot).copied()
    }

    pub fn uses_c(&self) -> bool {
        self.slot_regs
            .values()
            .any(|r| r.mask().intersects(RegMask::C))
    }

    pub fn uses_b(&self) -> bool {
        self.slot_regs
            .values()
            .any(|r| r.mask().intersects(RegMask::B))
    }

    pub fn uses_bc(&self) -> bool {
        !self.slot_regs.is_empty()
    }

    pub fn used_mask(&self) -> RegMask {
        self.slot_regs
            .values()
            .fold(RegMask::NONE, |acc, r| acc.union(r.mask()))
    }

    pub fn call_live_mask(&self, bb_idx: usize, inst_idx: usize) -> RegMask {
        self.call_live_masks
            .get(&(bb_idx, inst_idx))
            .copied()
            .unwrap_or(RegMask::NONE)
    }
}

fn parse_reg_mask(reg: &str) -> RegMask {
    match reg.to_ascii_lowercase().as_str() {
        "b" => RegMask::B,
        "c" => RegMask::C,
        "bc" => RegMask::BC,
        "ix" => RegMask::IX,
        "all" => RegMask(RegMask::BC.0 | RegMask::IX.0),
        _ => RegMask::NONE,
    }
}

/// Returns the mask of tracked registers read as args, clobbered, or returned by `ef`.
pub fn extern_clobber_mask(ef: &IrExternFn) -> RegMask {
    let mut mask = RegMask::NONE;
    for (_, _, r) in &ef.params {
        mask = mask.union(parse_reg_mask(r));
    }
    for r in &ef.clobbers {
        mask = mask.union(parse_reg_mask(r));
    }
    if let Some((_, r)) = &ef.ret {
        mask = mask.union(parse_reg_mask(r));
    }
    mask
}

/// Returns the `RegMask` of fixed scratch/loop registers clobbered internally by emitting `inst`.
/// Excludes interprocedural `Call`/`ExternCall` targets (handled by `fn_clobbers`/`call_sites`).
pub(crate) fn inst_internal_clobber_mask(
    program: &IrProgram,
    inst: &IrInst,
    const_ints: &HashMap<ValueId, i64>,
) -> RegMask {
    match inst {
        IrInst::InlineAsm { .. } => RegMask(RegMask::BC.0 | RegMask::IX.0),
        IrInst::PoolAlloc { .. }
        | IrInst::PoolFree { .. }
        | IrInst::PoolElemAddr { .. }
        | IrInst::StateTransition { .. }
        | IrInst::StateDispatch { .. } => RegMask::BC,
        IrInst::PortIn { port, .. } | IrInst::PortOut { port, .. } => {
            if const_ints.contains_key(port) {
                RegMask::NONE
            } else {
                RegMask::BC
            }
        }
        IrInst::FixedInt { .. } => RegMask::B,
        IrInst::BinOp {
            op: IrBinOp::Mul,
            ty,
            lhs,
            rhs,
            ..
        } => {
            let sz = program.type_size(ty);
            if sz == 1 {
                RegMask::B
            } else if sz == 2 && !const_ints.contains_key(lhs) && !const_ints.contains_key(rhs) {
                RegMask::BC
            } else {
                RegMask::NONE
            }
        }
        IrInst::BinOp {
            op: op @ (IrBinOp::Shl | IrBinOp::Shr),
            ty,
            rhs,
            ..
        } => {
            let sz = program.type_size(ty);
            let signed = crate::analysis::is_signed_ty(ty);
            let unrolled = if let Some(&k_raw) = const_ints.get(rhs) {
                match (sz, *op, k_raw) {
                    (1, IrBinOp::Shl, 0..=4 | 8) => true,
                    (1, IrBinOp::Shr, 0..=3 | 8) => true,
                    (2, IrBinOp::Shl, 0..=4 | 7..=16) => true,
                    (2, IrBinOp::Shr, 0..=3) => true,
                    (2, IrBinOp::Shr, 8..=16) if !signed => true,
                    _ => false,
                }
            } else {
                false
            };
            if unrolled { RegMask::NONE } else { RegMask::B }
        }
        _ => RegMask::NONE,
    }
}

/// Returns `(internal_clobber_mask, can_use_c, can_use_b)` for `func`.
/// Function-wide disables (`can_use_* = false`) apply only where codegen cannot isolate fixed
/// register use cleanly (inline asm, pool/state ops, and extern FFI args/rets using `B`/`C`/`BC`);
/// intra-block shift/mul/fixed-int clobbers are modeled as point intervals in `allocate_fn_regs`.
fn analyze_internal_reg_constraints(program: &IrProgram, func: &IrFunction) -> (RegMask, bool, bool) {
    let mut clob = RegMask::NONE;
    let mut can_use_c = true;
    let mut can_use_b = true;
    let mut const_ints: HashMap<ValueId, i64> = HashMap::new();

    for bb in &func.blocks {
        for inst in &bb.insts {
            if let IrInst::ConstInt { dst, val, .. } = inst {
                const_ints.insert(*dst, *val);
                continue;
            }
            clob = clob.union(inst_internal_clobber_mask(program, inst, &const_ints));
            match inst {
                IrInst::InlineAsm { .. }
                | IrInst::PoolAlloc { .. }
                | IrInst::PoolFree { .. }
                | IrInst::PoolElemAddr { .. }
                | IrInst::StateTransition { .. }
                | IrInst::StateDispatch { .. } => {
                    can_use_c = false;
                    can_use_b = false;
                }
                IrInst::ExternCall { extern_fn, .. } => {
                    let ef = &program.externs[*extern_fn as usize];
                    let em = extern_clobber_mask(ef);
                    clob = clob.union(em);
                    if ef.params.iter().any(|(_, _, r)| {
                        matches!(r.to_ascii_lowercase().as_str(), "c" | "bc")
                    }) || ef.ret.as_ref().is_some_and(|(_, r)| {
                        matches!(r.to_ascii_lowercase().as_str(), "b" | "c" | "bc")
                    }) {
                        can_use_c = false;
                        can_use_b = false;
                    }
                    if ef.params.iter().any(|(_, _, r)| {
                        matches!(r.to_ascii_lowercase().as_str(), "b" | "bc")
                    }) {
                        can_use_b = false;
                    }
                }
                _ => {}
            }
        }
    }

    if !can_use_c {
        can_use_b = false;
    }
    (clob, can_use_c, can_use_b)
}

fn intervals_overlap(a: (usize, usize), b: (usize, usize)) -> bool {
    a.0 <= b.1 && b.0 <= a.1
}

/// Allocates non-overlapping local slots of `func` to hardware registers `C`, `B`, and `BC`
/// using live-interval coloring and computes live register masks across every call site.
pub fn allocate_fn_regs(
    program: &IrProgram,
    func: &IrFunction,
    reg_p1: bool,
    ix_param_slot: Option<SlotId>,
) -> FnRegAlloc {
    allocate_fn_regs_with_clobbers(program, func, reg_p1, ix_param_slot, None)
}

/// Same as `allocate_fn_regs`, using whole-program `fn_clobbers` (when known) so per-register
/// `live_calls` profitability reflects whether callees actually clobber `B` vs `C`.
pub fn allocate_fn_regs_with_clobbers(
    program: &IrProgram,
    func: &IrFunction,
    reg_p1: bool,
    ix_param_slot: Option<SlotId>,
    fn_clobbers: Option<&[RegMask]>,
) -> FnRegAlloc {
    if func.locals.is_empty() || func.blocks.iter().all(|b| b.insts.is_empty()) {
        return FnRegAlloc::default();
    }

    let (_, can_use_c, can_use_b) = analyze_internal_reg_constraints(program, func);
    if !can_use_c && !can_use_b {
        return FnRegAlloc::default();
    }

    let n_slots = func.locals.len();
    let mut addr_taken = vec![false; n_slots];
    let mut uses = vec![0usize; n_slots];
    let mut min_pos: Vec<Option<usize>> = vec![None; n_slots];
    let mut max_pos: Vec<Option<usize>> = vec![None; n_slots];
    let mut slot_accesses: Vec<Vec<(usize, bool /* is_load */, usize /* bb_idx */)>> =
        vec![Vec::new(); n_slots];
    let mut bb_pos: HashMap<BlockId, (usize, usize)> = HashMap::new();
    let mut call_sites: Vec<(usize, RegMask)> = Vec::new(); // (pos, callee_clobber_mask)
    let mut const_ints: HashMap<ValueId, i64> = HashMap::new();
    let mut fixed_b_clobbers: Vec<(usize, usize)> = Vec::new();
    let mut fixed_c_clobbers: Vec<(usize, usize)> = Vec::new();

    let mut cur_pos = 0usize;
    for (bi, bb) in func.blocks.iter().enumerate() {
        let start = cur_pos;
        for inst in &bb.insts {
            if let IrInst::ConstInt { dst, val, .. } = inst {
                const_ints.insert(*dst, *val);
            } else {
                let ic = inst_internal_clobber_mask(program, inst, &const_ints);
                if ic.intersects(RegMask::B) {
                    fixed_b_clobbers.push((cur_pos, cur_pos));
                }
                if ic.intersects(RegMask::C) {
                    fixed_c_clobbers.push((cur_pos, cur_pos));
                }
            }
            match inst {
                IrInst::AddrLocal { slot, .. } => {
                    if let Some(f) = addr_taken.get_mut(*slot as usize) {
                        *f = true;
                    }
                }
                IrInst::LoadLocal { slot, .. } => {
                    let s = *slot as usize;
                    if s < n_slots {
                        uses[s] += 1;
                        min_pos[s] = Some(min_pos[s].map_or(cur_pos, |m| m.min(cur_pos)));
                        max_pos[s] = Some(max_pos[s].map_or(cur_pos, |m| m.max(cur_pos)));
                        slot_accesses[s].push((cur_pos, true, bi));
                    }
                }
                IrInst::StoreLocal { slot, .. } => {
                    let s = *slot as usize;
                    if s < n_slots {
                        uses[s] += 1;
                        min_pos[s] = Some(min_pos[s].map_or(cur_pos, |m| m.min(cur_pos)));
                        max_pos[s] = Some(max_pos[s].map_or(cur_pos, |m| m.max(cur_pos)));
                        slot_accesses[s].push((cur_pos, false, bi));
                    }
                }
                IrInst::Call { func: cid, .. } => {
                    let cm = fn_clobbers
                        .and_then(|m| m.get(*cid as usize).copied())
                        .unwrap_or(RegMask::BC)
                        .0
                        & RegMask::BC.0;
                    if cm != 0 {
                        call_sites.push((cur_pos, RegMask(cm)));
                    }
                }
                IrInst::ExternCall { extern_fn, .. } => {
                    let em = extern_clobber_mask(&program.externs[*extern_fn as usize]);
                    if em.intersects(RegMask::BC) {
                        call_sites.push((cur_pos, RegMask(em.0 & RegMask::BC.0)));
                    }
                }
                _ => {}
            }
            cur_pos += 1;
        }
        bb_pos.insert(bb.id, (start, cur_pos));
        cur_pos += 1;
    }

    // Register-passed params (0 and optionally 1) are live from prologue entry (`pos = 0`).
    for s in 0..func.param_count.min(2) {
        let is_reg_param = s == 0 || (s == 1 && reg_p1);
        if is_reg_param && !addr_taken[s] && ix_param_slot != Some(s as SlotId) {
            min_pos[s] = Some(0);
            max_pos[s] = Some(max_pos[s].unwrap_or(0));
        }
    }

    // Extend live intervals across loop back-edges.
    let mut loops: Vec<(usize, usize)> = Vec::new();
    for bb in &func.blocks {
        let (_, src_end) = bb_pos[&bb.id];
        let mut check_edge = |target: BlockId| {
            if let Some(&(t_start, _)) = bb_pos.get(&target)
                && t_start <= src_end
            {
                loops.push((t_start, src_end));
            }
        };
        match bb.term {
            Terminator::Goto(t) => check_edge(t),
            Terminator::Branch {
                then_bb, else_bb, ..
            } => {
                check_edge(then_bb);
                check_edge(else_bb);
            }
            _ => {}
        }
    }
    for s in 0..n_slots {
        if let (Some(mut lo), Some(mut hi)) = (min_pos[s], max_pos[s]) {
            let is_reg_param =
                s < func.param_count.min(2) && (s == 0 || (s == 1 && reg_p1));
            let mut changed = true;
            while changed {
                changed = false;
                for &(l_lo, l_hi) in &loops {
                    let crosses_in = lo < l_lo && hi >= l_lo && hi < l_hi;
                    let crosses_out = lo > l_lo && lo <= l_hi && hi > l_hi;
                    let loop_carried_inside = if lo >= l_lo && hi <= l_hi && (lo > l_lo || hi < l_hi) {
                        if is_reg_param && l_lo == 0 {
                            true
                        } else {
                            let mut first_bb: Option<usize> = None;
                            let mut multi_bb = false;
                            let mut seen_store = false;
                            let mut load_before_store = false;
                            for &(p, is_load, bi) in &slot_accesses[s] {
                                if p < l_lo || p > l_hi {
                                    continue;
                                }
                                match first_bb {
                                    None => first_bb = Some(bi),
                                    Some(fb) if fb != bi => multi_bb = true,
                                    _ => {}
                                }
                                if is_load {
                                    if !seen_store {
                                        load_before_store = true;
                                    }
                                } else {
                                    seen_store = true;
                                }
                            }
                            multi_bb || load_before_store
                        }
                    } else {
                        false
                    };
                    if crosses_in && hi < l_hi {
                        hi = l_hi;
                        changed = true;
                    }
                    if crosses_out && lo > l_lo {
                        lo = l_lo;
                        changed = true;
                    }
                    if loop_carried_inside {
                        if lo > l_lo {
                            lo = l_lo;
                            changed = true;
                        }
                        if hi < l_hi {
                            hi = l_hi;
                            changed = true;
                        }
                    }
                }
            }
            min_pos[s] = Some(lo);
            max_pos[s] = Some(hi);
        }
    }

    // Collect profitable candidates sorted by descending net savings score.
    #[derive(Clone, Copy)]
    struct Cand {
        slot: SlotId,
        lo: usize,
        hi: usize,
        sz: u16,
        uses: usize,
        live_c: usize,
        live_b: usize,
        live_bc: usize,
        score: isize,
    }
    let is_profitable = |u: usize, lc: usize| if lc == 0 { u >= 2 } else { u > lc + 1 };
    let mut cands: Vec<Cand> = Vec::new();
    for (s, local) in func.locals.iter().enumerate() {
        let sid = s as SlotId;
        if addr_taken[s] || ix_param_slot == Some(sid) {
            continue;
        }
        let sz = program.type_size(&local.ty);
        if sz != 1 && sz != 2 {
            continue;
        }
        if s < func.param_count && !(s == 0 || (s == 1 && reg_p1)) {
            continue;
        }
        let (Some(lo), Some(hi)) = (min_pos[s], max_pos[s]) else {
            continue;
        };
        let u = uses[s];
        let mut live_c = 0usize;
        let mut live_b = 0usize;
        let mut live_bc = 0usize;
        for &(pos, cm) in &call_sites {
            if lo <= pos && pos < hi {
                if cm.intersects(RegMask::C) {
                    live_c += 1;
                }
                if cm.intersects(RegMask::B) {
                    live_b += 1;
                }
                if cm.intersects(RegMask::BC) {
                    live_bc += 1;
                }
            }
        }
        let min_calls_opt = match sz {
            1 => {
                let ok_c = can_use_c && is_profitable(u, live_c);
                let ok_b = can_use_b && is_profitable(u, live_b);
                match (ok_c, ok_b) {
                    (true, true) => Some(live_c.min(live_b)),
                    (true, false) => Some(live_c),
                    (false, true) => Some(live_b),
                    (false, false) => None,
                }
            }
            2 if can_use_b && can_use_c && is_profitable(u, live_bc) => Some(live_bc),
            _ => None,
        };
        if let Some(min_calls) = min_calls_opt {
            let score = (u as isize) * 2 - (min_calls as isize) * 2;
            cands.push(Cand {
                slot: sid,
                lo,
                hi,
                sz,
                uses: u,
                live_c,
                live_b,
                live_bc,
                score,
            });
        }
    }
    cands.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then_with(|| b.uses.cmp(&a.uses))
            .then_with(|| a.slot.cmp(&b.slot))
    });

    let mut intervals_c: Vec<(usize, usize)> = fixed_c_clobbers;
    let mut intervals_b: Vec<(usize, usize)> = fixed_b_clobbers;
    let mut slot_regs: HashMap<SlotId, RegLoc> = HashMap::new();
    let mut assigned_intervals: Vec<(SlotId, (usize, usize), RegLoc)> = Vec::new();

    for c in cands {
        let iv = (c.lo, c.hi);
        match c.sz {
            1 => {
                let ok_c = can_use_c
                    && is_profitable(c.uses, c.live_c)
                    && !intervals_c.iter().any(|&x| intervals_overlap(x, iv));
                let ok_b = can_use_b
                    && is_profitable(c.uses, c.live_b)
                    && !intervals_b.iter().any(|&x| intervals_overlap(x, iv));
                let pick = match (ok_c, ok_b) {
                    (true, true) => {
                        if c.live_b < c.live_c {
                            Some(RegLoc::B)
                        } else {
                            Some(RegLoc::C)
                        }
                    }
                    (true, false) => Some(RegLoc::C),
                    (false, true) => Some(RegLoc::B),
                    (false, false) => None,
                };
                if let Some(reg) = pick {
                    match reg {
                        RegLoc::C => intervals_c.push(iv),
                        RegLoc::B => intervals_b.push(iv),
                        RegLoc::BC => {}
                    }
                    slot_regs.insert(c.slot, reg);
                    assigned_intervals.push((c.slot, iv, reg));
                }
            }
            2 => {
                if can_use_b
                    && can_use_c
                    && is_profitable(c.uses, c.live_bc)
                    && !intervals_b.iter().any(|&x| intervals_overlap(x, iv))
                    && !intervals_c.iter().any(|&x| intervals_overlap(x, iv))
                {
                    intervals_b.push(iv);
                    intervals_c.push(iv);
                    slot_regs.insert(c.slot, RegLoc::BC);
                    assigned_intervals.push((c.slot, iv, RegLoc::BC));
                }
            }
            _ => {}
        }
    }

    // Compute per-call-site live register mask from assigned slot intervals.
    let mut all_call_sites: Vec<(usize, usize, usize)> = Vec::new();
    let mut pos = 0usize;
    for (bi, bb) in func.blocks.iter().enumerate() {
        for (ii, inst) in bb.insts.iter().enumerate() {
            if matches!(inst, IrInst::Call { .. } | IrInst::ExternCall { .. }) {
                all_call_sites.push((pos, bi, ii));
            }
            pos += 1;
        }
        pos += 1;
    }

    let mut call_live_masks: HashMap<(usize, usize), RegMask> = HashMap::new();
    for (c_pos, bi, ii) in all_call_sites {
        let mut mask = RegMask::NONE;
        for &(_slot, (lo, hi), reg) in &assigned_intervals {
            let is_live = lo <= c_pos && c_pos < hi;
            if is_live {
                mask = mask.union(reg.mask());
            }
        }
        if mask != RegMask::NONE {
            call_live_masks.insert((bi, ii), mask);
        }
    }

    FnRegAlloc {
        slot_regs,
        call_live_masks,
    }
}

/// Computes bottom-up interprocedural clobber masks (`B`, `C`, `IX`) for every function.
pub fn compute_fn_clobbers(program: &IrProgram, fns: &[FnAnalysis], any_nonleaf_uses_ix: bool) -> Vec<RegMask> {
    let n = program.functions.len();
    let mut clobbers = vec![RegMask::NONE; n];
    let mut callees: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];

    for (i, func) in program.functions.iter().enumerate() {
        let (mut mask, _, _) = analyze_internal_reg_constraints(program, func);
        mask = mask.union(fns[i].reg_alloc.used_mask());
        // Check whether IX is modified without callee-save.
        if fns[i].ix_param_slot.is_some() {
            let save_ix = !fns[i].is_leaf || any_nonleaf_uses_ix || func.is_pub;
            if !save_ix {
                mask = mask.union(RegMask::IX);
            }
        }
        clobbers[i] = mask;

        for bb in &func.blocks {
            for inst in &bb.insts {
                if let IrInst::Call { func: cid, .. } = inst {
                    callees[i].insert(*cid as usize);
                }
            }
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..n {
            let mut m = clobbers[i];
            for &c in &callees[i] {
                if c < n {
                    m = m.union(clobbers[c]);
                }
            }
            if m != clobbers[i] {
                clobbers[i] = m;
                changed = true;
            }
        }
    }
    clobbers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::ProgramAnalysis;

    #[test]
    fn multi_local_coloring_shares_non_overlapping_c_and_splits_overlapping_bc() {
        let u8t = IrType::Int(IntType::U8);
        // Sequential non-overlapping a, b -> both color to C; overlapping x, y -> C and B; 16-bit w -> BC!
        let f = IrFunction {
            id: 0,
            name: "f".into(),
            is_pub: false,
            param_count: 0,
            locals: vec![
                LocalSlot {
                    id: 0,
                    name: "a".into(),
                    ty: u8t.clone(),
                    mutable: true,
                },
                LocalSlot {
                    id: 1,
                    name: "b".into(),
                    ty: u8t.clone(),
                    mutable: true,
                },
                LocalSlot {
                    id: 2,
                    name: "w".into(),
                    ty: IrType::Int(IntType::U16),
                    mutable: true,
                },
            ],
            ret_ty: IrType::Int(IntType::U8),
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            blocks: vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::ConstInt {
                        dst: 0,
                        ty: u8t.clone(),
                        val: 1,
                    },
                    IrInst::StoreLocal {
                        slot: 0,
                        val: 0,
                        ty: u8t.clone(),
                    },
                    IrInst::LoadLocal {
                        dst: 1,
                        slot: 0,
                        ty: u8t.clone(),
                    },
                    IrInst::StoreLocal {
                        slot: 1,
                        val: 0,
                        ty: u8t.clone(),
                    },
                    IrInst::LoadLocal {
                        dst: 2,
                        slot: 1,
                        ty: u8t.clone(),
                    },
                    IrInst::ConstInt {
                        dst: 3,
                        ty: IrType::Int(IntType::U16),
                        val: 2,
                    },
                    IrInst::StoreLocal {
                        slot: 2,
                        val: 3,
                        ty: IrType::Int(IntType::U16),
                    },
                    IrInst::LoadLocal {
                        dst: 4,
                        slot: 2,
                        ty: IrType::Int(IntType::U16),
                    },
                ],
                term: Terminator::Return(Some(2)),
            }],
            next_val: 5,
        };
        let prog = IrProgram {
            functions: vec![f],
            ..Default::default()
        };
        let ra = allocate_fn_regs(&prog, &prog.functions[0], false, None);
        assert_eq!(ra.slot_reg(0), Some(RegLoc::C));
        assert_eq!(ra.slot_reg(1), Some(RegLoc::C));
        assert_eq!(ra.slot_reg(2), Some(RegLoc::BC));
    }

    #[test]
    fn ipra_clobber_masks_distinguish_clean_and_clobbering_callees() {
        let u8t = IrType::Int(IntType::U8);
        let clean_callee = IrFunction {
            id: 0,
            name: "clean".into(),
            is_pub: false,
            param_count: 1,
            locals: vec![LocalSlot {
                id: 0,
                name: "x".into(),
                ty: u8t.clone(),
                mutable: false,
            }],
            ret_ty: u8t.clone(),
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            blocks: vec![BasicBlock {
                id: 0,
                insts: vec![IrInst::LoadLocal {
                    dst: 0,
                    slot: 0,
                    ty: u8t.clone(),
                }],
                term: Terminator::Return(Some(0)),
            }],
            next_val: 1,
        };
        let bc_callee = IrFunction {
            id: 1,
            name: "uses_c".into(),
            is_pub: false,
            param_count: 1,
            locals: vec![
                LocalSlot {
                    id: 0,
                    name: "p".into(),
                    ty: u8t.clone(),
                    mutable: true,
                },
            ],
            ret_ty: u8t.clone(),
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            blocks: vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::LoadLocal {
                        dst: 0,
                        slot: 0,
                        ty: u8t.clone(),
                    },
                    IrInst::LoadLocal {
                        dst: 1,
                        slot: 0,
                        ty: u8t.clone(),
                    },
                    IrInst::StoreLocal {
                        slot: 0,
                        val: 1,
                        ty: u8t,
                    },
                ],
                term: Terminator::Return(Some(1)),
            }],
            next_val: 2,
        };
        let prog = IrProgram {
            functions: vec![clean_callee, bc_callee],
            ..Default::default()
        };
        let analysis = ProgramAnalysis::analyze(&prog);
        assert!(!analysis.fn_clobbers[0].intersects(RegMask::BC));
        assert!(analysis.fn_clobbers[1].contains(RegMask::C));
        // Even a B-only or C-only clobbering fn must remain in `no_bc_fns` until BOTH B & C are clobbered.
        let no_bc = analysis.no_bc_fns(&prog);
        assert!(no_bc.contains("clean") && no_bc.contains("uses_c"));
    }

    #[test]
    fn signed_i16_shr_marks_b_clobbered_and_loop_carried_slot_saves_call_at_header() {
        let i16t = IrType::Int(IntType::I16);
        let u8t = IrType::Int(IntType::U8);
        let ext_c = IrExternFn {
            name: "_ext".into(),
            params: vec![],
            ret: None,
            clobbers: vec!["B".into(), "C".into()],
            requires_di: false,
            requires_banks: vec![],
        };
        let shift_fn = IrFunction {
            id: 0,
            name: "signed_shr".into(),
            is_pub: false,
            param_count: 1,
            locals: vec![
                LocalSlot { id: 0, name: "x".into(), ty: i16t.clone(), mutable: false },
                LocalSlot { id: 1, name: "k".into(), ty: u8t.clone(), mutable: true },
            ],
            ret_ty: i16t.clone(),
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            blocks: vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::ConstInt { dst: 0, ty: u8t.clone(), val: 9 },
                    IrInst::LoadLocal { dst: 1, slot: 0, ty: i16t.clone() },
                    IrInst::BinOp {
                        dst: 2,
                        op: IrBinOp::Shr,
                        ty: i16t.clone(),
                        mode: aun_ir::ArithMode::Wrapping,
                        lhs: 1,
                        rhs: 0,
                        site_id: 0,
                    },
                    IrInst::ConstInt { dst: 3, ty: u8t.clone(), val: 1 },
                    IrInst::StoreLocal { slot: 1, val: 3, ty: u8t.clone() },
                    IrInst::LoadLocal { dst: 4, slot: 1, ty: u8t.clone() },
                    IrInst::StoreLocal { slot: 1, val: 4, ty: u8t.clone() },
                ],
                term: Terminator::Return(Some(2)),
            }],
            next_val: 5,
        };
        // Loop where bb0 is the loop header, Call at pos 0, and local slot 1 is read & written inside loop.
        let loop_fn = IrFunction {
            id: 1,
            name: "loop_hdr_call".into(),
            is_pub: false,
            param_count: 0,
            locals: vec![
                LocalSlot { id: 0, name: "acc".into(), ty: u8t.clone(), mutable: true },
            ],
            ret_ty: u8t.clone(),
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            blocks: vec![
                BasicBlock {
                    id: 0,
                    insts: vec![
                        IrInst::ExternCall { dst: None, extern_fn: 0, args: vec![] },
                        IrInst::LoadLocal { dst: 0, slot: 0, ty: u8t.clone() },
                        IrInst::LoadLocal { dst: 1, slot: 0, ty: u8t.clone() },
                        IrInst::ConstInt { dst: 2, ty: u8t.clone(), val: 1 },
                        IrInst::StoreLocal { slot: 0, val: 2, ty: u8t.clone() },
                    ],
                    term: Terminator::Goto(0),
                },
            ],
            next_val: 3,
        };
        // `can_alias_reg_load` must return false when Call lies between LoadLocal and its last use.
        let alias_fn = IrFunction {
            id: 2,
            name: "alias_across_call".into(),
            is_pub: false,
            param_count: 1,
            locals: vec![LocalSlot { id: 0, name: "p".into(), ty: u8t.clone(), mutable: true }],
            ret_ty: u8t.clone(),
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            blocks: vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::LoadLocal { dst: 0, slot: 0, ty: u8t.clone() },
                    IrInst::LoadLocal { dst: 1, slot: 0, ty: u8t.clone() },
                    IrInst::ExternCall { dst: None, extern_fn: 0, args: vec![] },
                ],
                term: Terminator::Return(Some(1)),
            }],
            next_val: 2,
        };
        let prog = IrProgram {
            externs: vec![ext_c],
            functions: vec![shift_fn, loop_fn, alias_fn],
            ..Default::default()
        };
        let analysis = ProgramAnalysis::analyze(&prog);
        assert!(
            analysis.fn_clobbers[0].contains(RegMask::B),
            "signed i16 >> 9 uses djnz B and must mark B clobbered"
        );
        assert_eq!(analysis.fns[0].reg_alloc.slot_reg(1), Some(RegLoc::C));
        assert_eq!(analysis.fns[1].reg_alloc.slot_reg(0), Some(RegLoc::C));
        assert_eq!(
            analysis.fns[1].reg_alloc.call_live_mask(0, 0),
            RegMask::C,
            "loop-carried slot with Call at l_lo must save C at pos 0"
        );
        assert!(!crate::analysis::can_alias_reg_load(
            &prog,
            &prog.functions[2],
            &analysis.fns[2].reg_alloc,
            0,
            1,
            1,
            RegLoc::C,
        ));
    }

    #[test]
    fn allocates_b_outside_fixed_clobber_point_and_rejects_spanning_or_aliasing_b() {
        let u8t = IrType::Int(IntType::U8);
        let i16t = IrType::Int(IntType::I16);
        let f = IrFunction {
            id: 0,
            name: "b_point_clobber".into(),
            is_pub: false,
            param_count: 1,
            locals: vec![
                // slot 0 (x: i16, param in HL): triggers `i16 >> 9` at pos 4 (clobbers B at pos 4)
                LocalSlot { id: 0, name: "x".into(), ty: i16t.clone(), mutable: false },
                // slot 1 (c_hold: u8): [1, 10] gets C
                LocalSlot { id: 1, name: "c_hold".into(), ty: u8t.clone(), mutable: true },
                // slot 2 (spans pos 4): [3, 6] overlaps C and B-clobber at 4 -> stays spilled!
                LocalSlot { id: 2, name: "span_b".into(), ty: u8t.clone(), mutable: true },
                // slot 3 (disjoint after pos 4): [7, 9] overlaps C but NOT pos 4 -> gets B!
                LocalSlot { id: 3, name: "disjoint_b".into(), ty: u8t.clone(), mutable: true },
            ],
            ret_ty: u8t.clone(),
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            blocks: vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::ConstInt { dst: 0, ty: u8t.clone(), val: 9 },          // 0
                    IrInst::StoreLocal { slot: 1, val: 0, ty: u8t.clone() },       // 1 (c_hold: 4 uses)
                    IrInst::LoadLocal { dst: 1, slot: 0, ty: i16t.clone() },       // 2
                    IrInst::StoreLocal { slot: 2, val: 0, ty: u8t.clone() },       // 3 (span_b: 3 uses [3, 6])
                    IrInst::BinOp {                                                // 4 (clobbers B!)
                        dst: 2,
                        op: IrBinOp::Shr,
                        ty: i16t,
                        mode: aun_ir::ArithMode::Wrapping,
                        lhs: 1,
                        rhs: 0,
                        site_id: 0,
                    },
                    IrInst::LoadLocal { dst: 3, slot: 2, ty: u8t.clone() },        // 5
                    IrInst::StoreLocal { slot: 2, val: 3, ty: u8t.clone() },       // 6 (span_b end)
                    IrInst::StoreLocal { slot: 3, val: 0, ty: u8t.clone() },       // 7 (disjoint_b: 3 uses [7, 9])
                    IrInst::LoadLocal { dst: 4, slot: 3, ty: u8t.clone() },        // 8
                    IrInst::LoadLocal { dst: 5, slot: 3, ty: u8t.clone() },        // 9 (disjoint_b end)
                    IrInst::LoadLocal { dst: 6, slot: 1, ty: u8t.clone() },        // 10
                    IrInst::LoadLocal { dst: 7, slot: 1, ty: u8t.clone() },        // 11
                    IrInst::LoadLocal { dst: 8, slot: 1, ty: u8t.clone() },        // 12 (c_hold end)
                ],
                term: Terminator::Return(Some(8)),
            }],
            next_val: 9,
        };
        // Also test `can_alias_reg_load` rejecting aliasing `B` across a B-clobbering instruction:
        let alias_b_fn = IrFunction {
            id: 1,
            name: "alias_across_b_clobber".into(),
            is_pub: false,
            param_count: 1,
            locals: vec![
                LocalSlot { id: 0, name: "p".into(), ty: u8t.clone(), mutable: true },
                LocalSlot { id: 1, name: "unused_i16".into(), ty: u8t.clone(), mutable: true },
            ],
            ret_ty: u8t.clone(),
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            blocks: vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::ConstInt { dst: 0, ty: u8t.clone(), val: 9 },          // 0
                    IrInst::LoadLocal { dst: 1, slot: 0, ty: u8t.clone() },        // 1 (load B)
                    IrInst::BinOp {                                                // 2 (clobbers B!)
                        dst: 2,
                        op: IrBinOp::Shr,
                        ty: IrType::Int(IntType::I16),
                        mode: aun_ir::ArithMode::Wrapping,
                        lhs: 1,
                        rhs: 0,
                        site_id: 0,
                    },
                    IrInst::StoreLocal {                                           // 3 (uses dst 1)
                        slot: 1,
                        val: 1,
                        ty: u8t,
                    },
                ],
                term: Terminator::Return(Some(2)),
            }],
            next_val: 4,
        };
        let prog = IrProgram {
            functions: vec![f, alias_b_fn],
            ..Default::default()
        };
        let analysis = ProgramAnalysis::analyze(&prog);
        assert_eq!(analysis.fns[0].reg_alloc.slot_reg(1), Some(RegLoc::C));
        assert_eq!(analysis.fns[0].reg_alloc.slot_reg(2), None);
        assert_eq!(analysis.fns[0].reg_alloc.slot_reg(3), Some(RegLoc::B));
        assert!(!crate::analysis::can_alias_reg_load(
            &prog,
            &prog.functions[1],
            &analysis.fns[1].reg_alloc,
            0,
            1,
            1,
            RegLoc::B,
        ));
    }

    #[test]
    fn ipra_zero_c_clobbers_allocates_c_across_b_clobbering_extern_calls() {
        let u8t = IrType::Int(IntType::U8);
        let opn_write = IrExternFn {
            name: "_opn_write".into(),
            params: vec![
                ("reg".into(), u8t.clone(), "A".into()),
                ("val".into(), u8t.clone(), "L".into()),
            ],
            ret: None,
            clobbers: vec!["A".into(), "B".into(), "D".into(), "F".into()],
            requires_di: false,
            requires_banks: vec![],
        };
        // Local `slot 0` has only 2 accesses (`u = 2`) spanning 3 B-clobbering calls:
        // globally `u <= live_calls + 1` (2 <= 4), yet `live_c == 0` so C is profitable!
        let f = IrFunction {
            id: 0,
            name: "opn_burst".into(),
            is_pub: false,
            param_count: 1,
            locals: vec![LocalSlot {
                id: 0,
                name: "v".into(),
                ty: u8t.clone(),
                mutable: false,
            }],
            ret_ty: u8t.clone(),
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            blocks: vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::LoadLocal { dst: 0, slot: 0, ty: u8t.clone() },
                    IrInst::ExternCall { dst: None, extern_fn: 0, args: vec![] },
                    IrInst::ExternCall { dst: None, extern_fn: 0, args: vec![] },
                    IrInst::ExternCall { dst: None, extern_fn: 0, args: vec![] },
                    IrInst::LoadLocal { dst: 1, slot: 0, ty: u8t },
                ],
                term: Terminator::Return(Some(1)),
            }],
            next_val: 2,
        };
        let prog = IrProgram {
            externs: vec![opn_write],
            functions: vec![f],
            ..Default::default()
        };
        let analysis = ProgramAnalysis::analyze(&prog);
        assert_eq!(analysis.fns[0].reg_alloc.slot_reg(0), Some(RegLoc::C));
        assert_eq!(analysis.fns[0].reg_alloc.call_live_mask(0, 1), RegMask::C);
    }
}
