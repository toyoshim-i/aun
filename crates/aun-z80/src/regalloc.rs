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
    match &ef.clobbers {
        Some(cs) => {
            for r in cs {
                mask = mask.union(parse_reg_mask(r));
            }
        }
        None => {
            // Omitted `clobbers(...)` on Z80 defaults to caller-saved `AF, BC, DE, HL`.
            mask = mask.union(RegMask::BC);
        }
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
        IrInst::QueuePush { elem_ty, .. } => {
            if !elem_ty.is_scalar() || program.type_size(elem_ty) >= 4 {
                RegMask::BC
            } else {
                RegMask::NONE
            }
        }
        IrInst::QueuePop { elem_size, .. } => {
            if *elem_size >= 3 {
                RegMask::BC
            } else {
                RegMask::NONE
            }
        }
        IrInst::PortIn { port, .. } | IrInst::PortOut { port, .. } => {
            if const_ints.contains_key(port) {
                RegMask::NONE
            } else {
                RegMask::BC
            }
        }
        IrInst::MemSet { len, .. } => {
            if *len <= 1 {
                RegMask::NONE
            } else if *len < 12 {
                RegMask::B
            } else {
                RegMask::BC
            }
        }
        IrInst::MemCopy { len, .. } => {
            if *len == 0 {
                RegMask::NONE
            } else {
                RegMask::BC
            }
        }
        IrInst::MemFill { count, elem_size, .. } => {
            if *count <= 1 || *elem_size == 0 {
                RegMask::NONE
            } else {
                RegMask::BC
            }
        }
        IrInst::FixedInt { .. } => RegMask::B,
        IrInst::BinOp {
            op: op @ (IrBinOp::Mul | IrBinOp::Div | IrBinOp::Rem),
            ty,
            lhs,
            rhs,
            ..
        } => {
            let sz = program.type_size(ty);
            if sz == 4 || matches!(*op, IrBinOp::Div | IrBinOp::Rem) {
                RegMask::BC
            } else if sz == 1
                && *op == IrBinOp::Mul
                && !const_ints.contains_key(lhs)
                && !const_ints.contains_key(rhs)
            {
                RegMask::B
            } else if sz == 2
                && *op == IrBinOp::Mul
                && !const_ints.contains_key(lhs)
                && !const_ints.contains_key(rhs)
            {
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
            let unrolled = if let Some(&k_raw) = const_ints.get(rhs) {
                match (sz, *op, k_raw) {
                    (1, IrBinOp::Shl, _) => true,
                    (1, IrBinOp::Shr, k) => {
                        let kc = k.clamp(0, 8);
                        matches!(kc, 0..=3 | 8)
                            || (matches!(kc, 5..=7) && !crate::analysis::is_signed_ty(ty))
                    }
                    (2, IrBinOp::Shl | IrBinOp::Shr, _) => true,
                    (4, IrBinOp::Shl, k) => k <= 3 || (8..=11).contains(&k) || k >= 16,
                    (4, IrBinOp::Shr, k) => k <= 2 || (8..=10).contains(&k) || k >= 16,
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
                IrInst::CallIndirect { .. } => {
                    clob = clob.union(RegMask::BC);
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
    allocate_fn_regs_full(program, func, reg_p1, ix_param_slot, fn_clobbers, &[])
}

/// Full local register allocator accounting for IPRA callee clobber masks and any
/// `ByteCopyLoop` regions whose internal `B` / `BC` registers cannot be shared across the loop.
pub fn allocate_fn_regs_full(
    program: &IrProgram,
    func: &IrFunction,
    reg_p1: bool,
    ix_param_slot: Option<SlotId>,
    fn_clobbers: Option<&[RegMask]>,
    byte_copy_loops: &[crate::analysis::ByteCopyLoop],
) -> FnRegAlloc {
    if func.locals.is_empty() || func.blocks.iter().all(|b| b.insts.is_empty()) {
        return FnRegAlloc::default();
    }

    let (_, can_use_c, can_use_b) = analyze_internal_reg_constraints(program, func);
    if !can_use_c && !can_use_b {
        return FnRegAlloc::default();
    }

    let n_slots = func.locals.len();
    let cfg = aun_ir::cfg::Cfg::build(func);
    let mut id_to_idx: HashMap<BlockId, usize> = HashMap::with_capacity(func.blocks.len());
    for (bi, bb) in func.blocks.iter().enumerate() {
        id_to_idx.insert(bb.id, bi);
    }
    let mut loop_depth = vec![0u32; func.blocks.len()];
    for lp in cfg.natural_loops() {
        for &b in &lp.body {
            if let Some(&bi) = id_to_idx.get(&b) {
                loop_depth[bi] += 1;
            }
        }
    }
    let bb_weight: Vec<usize> = loop_depth
        .iter()
        .map(|&d| 1usize << d.min(3))
        .collect();

    let mut addr_taken = vec![false; n_slots];
    let mut uses = vec![0usize; n_slots];
    let mut w_uses = vec![0usize; n_slots];
    let mut const_ints: HashMap<ValueId, i64> = HashMap::new();
    let mut use_sets: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); func.blocks.len()];
    let mut def_sets: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); func.blocks.len()];

    for &bi in cfg.rpo_indices() {
        let bb = &func.blocks[bi];
        let bw = bb_weight[bi];
        for inst in &bb.insts {
            match inst {
                IrInst::ConstInt { dst, val, .. } => {
                    const_ints.insert(*dst, *val);
                }
                IrInst::AddrLocal { slot, .. } => {
                    if let Some(f) = addr_taken.get_mut(*slot as usize) {
                        *f = true;
                    }
                }
                IrInst::LoadLocal { slot, .. } => {
                    let s = *slot as usize;
                    if s < n_slots {
                        uses[s] += 1;
                        w_uses[s] += bw;
                        if !def_sets[bi].contains(&s) {
                            use_sets[bi].insert(s);
                        }
                    }
                }
                IrInst::StoreLocal { slot, .. } => {
                    let s = *slot as usize;
                    if s < n_slots {
                        uses[s] += 1;
                        w_uses[s] += bw;
                        def_sets[bi].insert(s);
                    }
                }
                _ => {}
            }
        }
    }

    // Iterative backward dataflow liveness over reachable CFG blocks:
    //   live_out[B] = union_{S in succ(B)} live_in[S]
    //   live_in[B]  = use[B] union (live_out[B] - def[B])
    let mut live_in: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); func.blocks.len()];
    let mut live_out: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); func.blocks.len()];
    loop {
        let mut changed = false;
        for &bi in cfg.rpo_indices().iter().rev() {
            let mut out = BTreeSet::new();
            for &succ_id in cfg.succs(func.blocks[bi].id) {
                if let Some(&sbi) = id_to_idx.get(&succ_id) {
                    out.extend(live_in[sbi].iter().copied());
                }
            }
            let mut in_set = use_sets[bi].clone();
            for &s in &out {
                if !def_sets[bi].contains(&s) {
                    in_set.insert(s);
                }
            }
            if out != live_out[bi] {
                live_out[bi] = out;
                changed = true;
            }
            if in_set != live_in[bi] {
                live_in[bi] = in_set;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // Build exact per-slot interference graph, per-slot clobber blocks, call-clobber counts,
    // and live-across-call slot sets by walking each reachable block backward.
    let mut adj: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n_slots];
    let mut blocked_b = vec![false; n_slots];
    let mut blocked_c = vec![false; n_slots];
    let mut live_c = vec![0usize; n_slots];
    let mut live_b = vec![0usize; n_slots];
    let mut live_bc = vec![0usize; n_slots];
    let mut w_live_c = vec![0usize; n_slots];
    let mut w_live_b = vec![0usize; n_slots];
    let mut w_live_bc = vec![0usize; n_slots];
    let mut call_live_slots: Vec<((usize, usize), BTreeSet<usize>)> = Vec::new();

    let mut bcl_slots: BTreeSet<SlotId> = BTreeSet::new();
    let mut bcl_bb_clobbers: HashMap<usize, RegMask> = HashMap::new();
    for bcl in byte_copy_loops {
        bcl_slots.insert(bcl.src_slot);
        bcl_slots.insert(bcl.dst_slot);
        bcl_slots.insert(bcl.cnt_slot);
        let mask = if bcl.is_u8_cnt {
            RegMask::B
        } else {
            RegMask::BC
        };
        for bid in [bcl.hdr_bb, bcl.step_bb] {
            if let Some(&bi) = id_to_idx.get(&bid) {
                bcl_bb_clobbers
                    .entry(bi)
                    .and_modify(|m| *m = m.union(mask))
                    .or_insert(mask);
            }
        }
    }

    for &bi in cfg.rpo_indices() {
        let bb = &func.blocks[bi];
        let bw = bb_weight[bi];
        let mut live = live_out[bi].clone();
        if let Some(&bm) = bcl_bb_clobbers.get(&bi) {
            for &s in &live {
                if bm.intersects(RegMask::B) {
                    blocked_b[s] = true;
                }
                if bm.intersects(RegMask::C) {
                    blocked_c[s] = true;
                }
            }
        }
        for (ii, inst) in bb.insts.iter().enumerate().rev() {
            match inst {
                IrInst::StoreLocal { slot, .. } => {
                    let s = *slot as usize;
                    if s < n_slots {
                        for &other in &live {
                            if other != s {
                                adj[s].insert(other);
                                adj[other].insert(s);
                            }
                        }
                        live.remove(&s);
                    }
                }
                IrInst::Call { func: cid, .. } => {
                    let cm = fn_clobbers
                        .and_then(|m| m.get(*cid as usize).copied())
                        .unwrap_or(RegMask::BC);
                    for &s in &live {
                        if cm.intersects(RegMask::C) {
                            live_c[s] += 1;
                            w_live_c[s] += bw;
                        }
                        if cm.intersects(RegMask::B) {
                            live_b[s] += 1;
                            w_live_b[s] += bw;
                        }
                        if cm.intersects(RegMask::BC) {
                            live_bc[s] += 1;
                            w_live_bc[s] += bw;
                        }
                    }
                    if !live.is_empty() {
                        call_live_slots.push(((bi, ii), live.clone()));
                    }
                }
                IrInst::CallIndirect { .. } => {
                    for &s in &live {
                        live_c[s] += 1;
                        live_b[s] += 1;
                        live_bc[s] += 1;
                        w_live_c[s] += bw;
                        w_live_b[s] += bw;
                        w_live_bc[s] += bw;
                    }
                    if !live.is_empty() {
                        call_live_slots.push(((bi, ii), live.clone()));
                    }
                }
                IrInst::ExternCall { extern_fn, .. } => {
                    let em = extern_clobber_mask(&program.externs[*extern_fn as usize]);
                    for &s in &live {
                        if em.intersects(RegMask::C) {
                            live_c[s] += 1;
                            w_live_c[s] += bw;
                        }
                        if em.intersects(RegMask::B) {
                            live_b[s] += 1;
                            w_live_b[s] += bw;
                        }
                        if em.intersects(RegMask::BC) {
                            live_bc[s] += 1;
                            w_live_bc[s] += bw;
                        }
                    }
                    if !live.is_empty() {
                        call_live_slots.push(((bi, ii), live.clone()));
                    }
                }
                IrInst::ConstInt { .. } => {}
                IrInst::LoadLocal { slot, .. } => {
                    let s = *slot as usize;
                    if s < n_slots {
                        live.insert(s);
                    }
                }
                other => {
                    let ic = inst_internal_clobber_mask(program, other, &const_ints);
                    if ic.intersects(RegMask::B) {
                        for &s in &live {
                            blocked_b[s] = true;
                        }
                    }
                    if ic.intersects(RegMask::C) {
                        for &s in &live {
                            blocked_c[s] = true;
                        }
                    }
                }
            }
        }
        if let Some(&bm) = bcl_bb_clobbers.get(&bi) {
            for &s in &live {
                if bm.intersects(RegMask::B) {
                    blocked_b[s] = true;
                }
                if bm.intersects(RegMask::C) {
                    blocked_c[s] = true;
                }
            }
        }
    }

    // Register-passed params (0 and optionally 1) are initialized at prologue entry (`bb0`),
    // so they mutually interfere with each other and with every slot in `live_in[entry]`.
    if let Some(&entry_bi) = cfg.rpo_indices().first() {
        let mut reg_params: Vec<usize> = Vec::new();
        for s in 0..func.param_count.min(2) {
            let is_reg_param = s == 0 || (s == 1 && reg_p1);
            if is_reg_param && !addr_taken[s] && ix_param_slot != Some(s as SlotId) {
                reg_params.push(s);
            }
        }
        for (idx, &p) in reg_params.iter().enumerate() {
            for &q in &reg_params[idx + 1..] {
                adj[p].insert(q);
                adj[q].insert(p);
            }
            for &other in &live_in[entry_bi] {
                if other != p {
                    adj[p].insert(other);
                    adj[other].insert(p);
                }
            }
        }
    }

    // Identify 1-byte countdown latch counters (`slot -= 1; slot != 0`) so they prefer `B` for `djnz`.
    let countdown_slots: std::collections::HashSet<SlotId> = func
        .blocks
        .iter()
        .filter_map(|bb| {
            crate::codegen::func::detect_countdown_latch(program, func, bb, &const_ints)
                .map(|(_, slot, _, _)| slot)
        })
        .collect();

    // Collect profitable candidates sorted by descending net savings score.
    #[derive(Clone, Copy)]
    struct Cand {
        slot: SlotId,
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
        if addr_taken[s]
            || ix_param_slot == Some(sid)
            || bcl_slots.contains(&sid)
            || uses[s] == 0
        {
            continue;
        }
        let sz = program.type_size(&local.ty);
        if sz != 1 && sz != 2 {
            continue;
        }
        if s < func.param_count && !(s == 0 || (s == 1 && reg_p1)) {
            continue;
        }
        let u = uses[s];
        let lc = live_c[s];
        let lb = live_b[s];
        let lbc = live_bc[s];
        let w_min_calls_opt = match sz {
            1 => {
                let ok_c = can_use_c && !blocked_c[s] && is_profitable(u, lc);
                let ok_b = can_use_b && !blocked_b[s] && is_profitable(u, lb);
                match (ok_c, ok_b) {
                    (true, true) => Some(w_live_b[s].min(w_live_c[s])),
                    (true, false) => Some(w_live_c[s]),
                    (false, true) => Some(w_live_b[s]),
                    (false, false) => None,
                }
            }
            2 if can_use_b
                && can_use_c
                && !blocked_b[s]
                && !blocked_c[s]
                && is_profitable(u, lbc) =>
            {
                Some(w_live_bc[s])
            }
            _ => None,
        };
        if let Some(w_min_calls) = w_min_calls_opt {
            let mut score = (w_uses[s] as isize) * 2 - (w_min_calls as isize) * 2;
            if countdown_slots.contains(&sid) {
                // Countdown latch counters (`djnz` / `dec r; jp nz` / `dec bc`) save instructions
                // and cycles on every loop back-edge and should beat generic RAM-slot spills.
                score += 6;
            }
            cands.push(Cand {
                slot: sid,
                sz,
                uses: u,
                live_c: lc,
                live_b: lb,
                live_bc: lbc,
                score,
            });
        }
    }
    cands.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then_with(|| a.sz.cmp(&b.sz))
            .then_with(|| b.uses.cmp(&a.uses))
            .then_with(|| a.slot.cmp(&b.slot))
    });

    let mut slot_regs: HashMap<SlotId, RegLoc> = HashMap::new();
    let neighbors_use_mask = |s: usize, slot_regs: &HashMap<SlotId, RegLoc>| -> RegMask {
        let mut m = RegMask::NONE;
        for &n in &adj[s] {
            if let Some(&r) = slot_regs.get(&(n as SlotId)) {
                m = m.union(r.mask());
            }
        }
        m
    };

    for (ci, &c) in cands.iter().enumerate() {
        let s = c.slot as usize;
        let used = neighbors_use_mask(s, &slot_regs);
        match c.sz {
            1 => {
                let ok_c = can_use_c
                    && !blocked_c[s]
                    && is_profitable(c.uses, c.live_c)
                    && !used.intersects(RegMask::C);
                let ok_b = can_use_b
                    && !blocked_b[s]
                    && is_profitable(c.uses, c.live_b)
                    && !used.intersects(RegMask::B);
                let pick = match (ok_c, ok_b) {
                    (true, true) => {
                        if (countdown_slots.contains(&c.slot) && c.live_b <= c.live_c)
                            || c.live_b < c.live_c
                        {
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
                    slot_regs.insert(c.slot, reg);
                }
            }
            2 => {
                if can_use_b
                    && can_use_c
                    && !blocked_b[s]
                    && !blocked_c[s]
                    && is_profitable(c.uses, c.live_bc)
                    && !used.intersects(RegMask::BC)
                {
                    // Do not let one `u16` (`BC`) candidate starve two interfering `u8` loop locals
                    // whose combined savings exceed the single `BC` allocation.
                    let mut sim_regs = slot_regs.clone();
                    let mut u8_score_sum = 0isize;
                    let mut u8_count = 0usize;
                    for &nc in &cands[ci + 1..] {
                        if nc.sz != 1 || !adj[s].contains(&(nc.slot as usize)) {
                            continue;
                        }
                        let ns = nc.slot as usize;
                        let nused = neighbors_use_mask(ns, &sim_regs);
                        let ok_c = can_use_c
                            && !blocked_c[ns]
                            && is_profitable(nc.uses, nc.live_c)
                            && !nused.intersects(RegMask::C);
                        let ok_b = can_use_b
                            && !blocked_b[ns]
                            && is_profitable(nc.uses, nc.live_b)
                            && !nused.intersects(RegMask::B);
                        let npick = match (ok_c, ok_b) {
                            (true, true) => {
                                if nc.live_b < nc.live_c {
                                    Some(RegLoc::B)
                                } else {
                                    Some(RegLoc::C)
                                }
                            }
                            (true, false) => Some(RegLoc::C),
                            (false, true) => Some(RegLoc::B),
                            (false, false) => None,
                        };
                        if let Some(nreg) = npick {
                            sim_regs.insert(nc.slot, nreg);
                            u8_score_sum += nc.score;
                            u8_count += 1;
                        }
                    }
                    if !(u8_count >= 2 && u8_score_sum > c.score) {
                        slot_regs.insert(c.slot, RegLoc::BC);
                    }
                }
            }
            _ => {}
        }
    }

    // Compute per-call-site live register mask from exact live slot sets.
    let mut call_live_masks: HashMap<(usize, usize), RegMask> = HashMap::new();
    for (site, live_slots) in call_live_slots {
        let mut mask = RegMask::NONE;
        for s in live_slots {
            if let Some(&reg) = slot_regs.get(&(s as SlotId)) {
                mask = mask.union(reg.mask());
            }
        }
        if mask != RegMask::NONE {
            call_live_masks.insert(site, mask);
        }
    }

    FnRegAlloc {
        slot_regs,
        call_live_masks,
    }
}

/// Computes bottom-up interprocedural clobber masks (`B`, `C`, `IX`) for every function.
pub fn compute_fn_clobbers(program: &IrProgram, fns: &[FnAnalysis]) -> Vec<RegMask> {
    let n = program.functions.len();
    let mut clobbers = vec![RegMask::NONE; n];
    let mut callees: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];

    for (i, func) in program.functions.iter().enumerate() {
        let (mut mask, _, _) = analyze_internal_reg_constraints(program, func);
        mask = mask.union(fns[i].reg_alloc.used_mask());
        for bcl in &fns[i].byte_copy_loops {
            mask = mask.union(if bcl.is_u8_cnt {
                RegMask::B
            } else {
                RegMask::BC
            });
        }
        clobbers[i] = mask;

        for bb in &func.blocks {
            for inst in &bb.insts {
                if let IrInst::Call { func: cid, .. } = inst {
                    callees[i].insert(*cid as usize);
                }
            }
        }

        let calls_ix_in_reg = callees[i]
            .iter()
            .any(|&c| fns.get(c).is_some_and(|cf| cf.ix_in_reg));
        let modifies_ix = (fns[i].uses_ix() && !fns[i].ix_in_reg)
            || (!fns[i].uses_ix() && calls_ix_in_reg);
        if modifies_ix && !fns[i].save_ix {
            clobbers[i] = clobbers[i].union(RegMask::IX);
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
            isr_using: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            owner_state: None,
            mock_target: None,
            export_name: None,
            cycle_budget: None,
            loop_bounds: HashMap::new(),
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
            isr_using: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            owner_state: None,
            mock_target: None,
            export_name: None,
            cycle_budget: None,
            loop_bounds: HashMap::new(),
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
            isr_using: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            owner_state: None,
            mock_target: None,
            export_name: None,
            cycle_budget: None,
            loop_bounds: HashMap::new(),
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
            clobbers: Some(vec!["B".into(), "C".into()]),
            requires_di: false,
            requires_banks: vec![],
            fixed_addr: None,
            no_callback: false,
            owner_state: None,
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
            isr_using: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            owner_state: None,
            mock_target: None,
            export_name: None,
            cycle_budget: None,
            loop_bounds: HashMap::new(),
            blocks: vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::ConstInt { dst: 0, ty: u8t.clone(), val: 4 },
                    IrInst::LoadLocal { dst: 1, slot: 0, ty: i16t.clone() },
                    IrInst::BinOp {
                        dst: 2,
                        op: IrBinOp::Shr,
                        ty: u8t.clone(),
                        mode: aun_ir::ArithMode::Wrapping,
                        lhs: 0,
                        rhs: 0,
                        site_id: 0,
                    },
                    IrInst::ConstInt { dst: 3, ty: u8t.clone(), val: 1 },
                    IrInst::StoreLocal { slot: 1, val: 3, ty: u8t.clone() },
                    IrInst::LoadLocal { dst: 4, slot: 1, ty: u8t.clone() },
                    IrInst::StoreLocal { slot: 1, val: 4, ty: u8t.clone() },
                ],
                term: Terminator::Return(Some(1)),
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
            isr_using: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            owner_state: None,
            mock_target: None,
            export_name: None,
            cycle_budget: None,
            loop_bounds: HashMap::new(),
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
            isr_using: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            owner_state: None,
            mock_target: None,
            export_name: None,
            cycle_budget: None,
            loop_bounds: HashMap::new(),
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
            "u8 >> 4 uses djnz B and must mark B clobbered"
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
            &analysis.fn_clobbers,
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
            isr_using: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            owner_state: None,
            mock_target: None,
            export_name: None,
            cycle_budget: None,
            loop_bounds: HashMap::new(),
            blocks: vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::ConstInt { dst: 0, ty: u8t.clone(), val: 4 },          // 0
                    IrInst::StoreLocal { slot: 1, val: 0, ty: u8t.clone() },       // 1 (c_hold: 4 uses)
                    IrInst::LoadLocal { dst: 1, slot: 0, ty: i16t.clone() },       // 2
                    IrInst::StoreLocal { slot: 2, val: 0, ty: u8t.clone() },       // 3 (span_b: 3 uses [3, 6])
                    IrInst::BinOp {                                                // 4 (clobbers B via u8 >> 4!)
                        dst: 2,
                        op: IrBinOp::Shr,
                        ty: u8t.clone(),
                        mode: aun_ir::ArithMode::Wrapping,
                        lhs: 0,
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
            isr_using: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            owner_state: None,
            mock_target: None,
            export_name: None,
            cycle_budget: None,
            loop_bounds: HashMap::new(),
            blocks: vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::ConstInt { dst: 0, ty: u8t.clone(), val: 4 },          // 0
                    IrInst::LoadLocal { dst: 1, slot: 0, ty: u8t.clone() },        // 1 (load B)
                    IrInst::BinOp {                                                // 2 (clobbers B via u8 >> 4!)
                        dst: 2,
                        op: IrBinOp::Shr,
                        ty: u8t.clone(),
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
            &analysis.fn_clobbers,
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
            clobbers: Some(vec!["A".into(), "B".into(), "D".into(), "F".into()]),
            requires_di: false,
            requires_banks: vec![],
            fixed_addr: None,
            no_callback: false,
            owner_state: None,
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
            isr_using: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            owner_state: None,
            mock_target: None,
            export_name: None,
            cycle_budget: None,
            loop_bounds: HashMap::new(),
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

    #[test]
    fn mutable_local_loop_counter_is_not_addr_taken_and_allocates_c() {
        let src = "\
pub fn loop_add(n: u8) -> u8 {
    let mut i: u8 = 0u8;
    let mut acc: u8 = 0u8;
    while i < n {
        acc = acc +% i;
        i = i +% 1u8;
    }
    acc
}
";
        let mut prog = aun_front::compile_str(0, src).expect("should compile");
        aun_ir::opt::optimize_program(&mut prog);
        let f = &prog.functions[0];
        assert!(
            !f.blocks
                .iter()
                .any(|b| b.insts.iter().any(|inst| matches!(inst, IrInst::AddrLocal { .. }))),
            "scalar mutable local assignments must lower to StoreLocal, not AddrLocal:\n{:?}",
            f.blocks
        );
        let i_slot = f.locals.iter().position(|l| l.name == "i").unwrap() as SlotId;
        let analysis = ProgramAnalysis::analyze(&prog);
        assert!(
            matches!(
                analysis.fns[0].reg_alloc.slot_reg(i_slot),
                Some(RegLoc::C | RegLoc::B)
            ),
            "loop counter `i` should allocate to B or C, got {:?}",
            analysis.fns[0].reg_alloc.slot_reg(i_slot)
        );
    }

    #[test]
    fn allocates_b_across_short_circuit_without_fake_loop() {
        let src = "\
extern \"asm\" {
    fn ext_a_only() -> u8 in A clobbers(A, F);
}

pub fn short_circuit(a: u8, b: u8) -> u8 {
    let mut x: u8 = a +% 1u8;
    if a < b && ext_a_only() != 0u8 {
        return x;
    }
    x = (b >> 5u8) +% 2u8;
    x
}
";
        let mut prog = aun_front::compile_str(0, src).expect("should compile");
        aun_ir::opt::optimize_program(&mut prog);
        let f = &prog.functions[0];
        let x_slot = f.locals.iter().position(|l| l.name == "x").unwrap() as SlotId;
        let analysis = ProgramAnalysis::analyze(&prog);
        // `x` is dead before `(b >> 5u8)` (which clobbers B) and never crosses any real loop!
        assert!(
            matches!(
                analysis.fns[0].reg_alloc.slot_reg(x_slot),
                Some(RegLoc::B | RegLoc::C)
            ),
            "expected `x` to allocate to B or C (got {:?})",
            analysis.fns[0].reg_alloc.slot_reg(x_slot)
        );
    }

    #[test]
    fn two_u8_locals_beat_single_u16_for_bc() {
        let src = "\
pub fn two_hot_u8s(init1: u8, init2: u8, init_w: u16) -> u8 {
    let mut c1: u8 = init1;
    let mut c2: u8 = init2;
    let mut w: u16 = init_w;
    while c1 > 0u8 {
        w = w +% 1u16;
        w = w +% 3u16;
        w = w +% (c2 as u16);
        c1 = c1 -% 1u8;
        c2 = c2 +% 5u8;
    }
    c1 +% c2 +% (w as u8)
}
";
        let mut prog = aun_front::compile_str(0, src).expect("should compile");
        aun_ir::opt::optimize_program(&mut prog);
        let f = &prog.functions[0];
        let c1_slot = f.locals.iter().position(|l| l.name == "c1").unwrap() as SlotId;
        let c2_slot = f.locals.iter().position(|l| l.name == "c2").unwrap() as SlotId;
        let w_slot = f.locals.iter().position(|l| l.name == "w").unwrap() as SlotId;
        let analysis = ProgramAnalysis::analyze(&prog);
        let ra = &analysis.fns[0].reg_alloc;
        assert!(
            matches!(ra.slot_reg(c1_slot), Some(RegLoc::B | RegLoc::C))
                && matches!(ra.slot_reg(c2_slot), Some(RegLoc::B | RegLoc::C))
                && ra.slot_reg(c1_slot) != ra.slot_reg(c2_slot),
            "two hot interfering u8 scalars should take B and C instead of being starved by u16 `w` (c1={:?}, c2={:?}, w={:?})",
            ra.slot_reg(c1_slot),
            ra.slot_reg(c2_slot),
            ra.slot_reg(w_slot)
        );
    }

    #[test]
    fn issue42_u8_rot_shifts_do_not_clobber_b_and_memset_ldir_clobbers_bc() {
        let prog = IrProgram::default();
        let u8t = IrType::Int(IntType::U8);
        let i8t = IrType::Int(IntType::I8);
        let mut consts = HashMap::new();
        for k in [5i64, 6, 7] {
            consts.insert(k as ValueId, k);
        }
        for k in [5u32, 6, 7] {
            let shl_u8 = IrInst::BinOp {
                dst: 10,
                op: IrBinOp::Shl,
                mode: ArithMode::Wrapping,
                ty: u8t.clone(),
                lhs: 0,
                rhs: k,
                site_id: 0,
            };
            let shr_u8 = IrInst::BinOp {
                dst: 11,
                op: IrBinOp::Shr,
                mode: ArithMode::Wrapping,
                ty: u8t.clone(),
                lhs: 0,
                rhs: k,
                site_id: 0,
            };
            let shr_i8 = IrInst::BinOp {
                dst: 12,
                op: IrBinOp::Shr,
                mode: ArithMode::Wrapping,
                ty: i8t.clone(),
                lhs: 0,
                rhs: k,
                site_id: 0,
            };
            assert_eq!(inst_internal_clobber_mask(&prog, &shl_u8, &consts), RegMask::NONE);
            assert_eq!(inst_internal_clobber_mask(&prog, &shr_u8, &consts), RegMask::NONE);
            assert_eq!(inst_internal_clobber_mask(&prog, &shr_i8, &consts), RegMask::B);
        }
        let short_memset = IrInst::MemSet {
            dst_addr: 0,
            val: 1,
            len: 11,
        };
        let ldir_memset = IrInst::MemSet {
            dst_addr: 0,
            val: 1,
            len: 12,
        };
        assert_eq!(inst_internal_clobber_mask(&prog, &short_memset, &consts), RegMask::B);
        assert_eq!(inst_internal_clobber_mask(&prog, &ldir_memset, &consts), RegMask::BC);
    }
}
