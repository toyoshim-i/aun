//! Single-source-of-truth Z80 whole-program and per-function static analyses.
//!
//! Computed once per compilation unit after Layer 2 (`aun-ir::opt`) and shared across
//! *Uchiguri* static RAM allocation, Z80 codegen, and peephole/SDASZ80 emission.

use crate::regalloc::{self, FnRegAlloc, RegLoc, RegMask};
use aun_ir::*;
use std::collections::{BTreeSet, HashMap, HashSet};

/// Converts an IR function/method name (which may contain `::`) into a valid Z80 label.
pub fn z80_sym(name: &str) -> String {
    name.replace("::", "__")
}

/// Static struct-array element pointer (`&STATIC_GLOBAL[.field*][idx]`) promoted to `IX`.
#[derive(Debug, Clone)]
pub struct IxStaticElem {
    pub global: u32,
    pub base_offset: u16,
    pub elem_size: u16,
    pub len: u16,
    /// When `Some(param_slot)`, `IX` can be initialized once in the function prologue
    /// (whenever bounds checks are disabled or `len == 0`).
    pub hoist_param: Option<SlotId>,
    /// All `IndexAddr` destination `ValueId`s in `func` promoted to `IX`.
    pub index_vals: BTreeSet<ValueId>,
    /// Non-param local slots (e.g. inlined `&mut T` ref args) whose only stores copy `IX + 0`.
    pub alias_slots: BTreeSet<SlotId>,
}

/// Per-function static Z80 backend facts derived solely from `IrProgram` + `IrFunction`.
#[derive(Debug, Clone)]
pub struct FnAnalysis {
    pub is_leaf: bool,
    pub has_inline_asm: bool,
    pub uses_reg_param1: bool,
    pub ix_param_slot: Option<SlotId>,
    pub ix_local_slot: Option<SlotId>,
    pub ix_static_elem: Option<IxStaticElem>,
    /// True when private whole-program callers pass `ix_param_slot` directly in `IX`.
    pub ix_in_reg: bool,
    /// True when this function must `push ix` / `pop ix` around its own `IX` modifications.
    pub save_ix: bool,
    pub reg_alloc: FnRegAlloc,
    pub escaped_locals: Vec<bool>,
    pub val_sizes: HashMap<ValueId, u16>,
    pub fused_cmp_dsts: BTreeSet<ValueId>,
}

impl FnAnalysis {
    /// Returns the parameter or local slot promoted to `IX`, if any.
    pub fn ix_slot(&self) -> Option<SlotId> {
        self.ix_param_slot.or(self.ix_local_slot)
    }

    /// Returns `true` when this function uses `IX` (parameter, local, or static element pointer).
    pub fn uses_ix(&self) -> bool {
        self.ix_param_slot.is_some()
            || self.ix_local_slot.is_some()
            || self.ix_static_elem.is_some()
    }

    /// Returns `true` when `slot` is an `IX + 0` alias slot from an inlined reference.
    pub fn is_ix_alias_slot(&self, slot: SlotId) -> bool {
        self.ix_static_elem
            .as_ref()
            .is_some_and(|ise| ise.alias_slots.contains(&slot))
    }
}

/// Whole-program static Z80 analysis shared by `uchiguri`, `codegen`, and `lib`.
#[derive(Debug, Clone)]
pub struct ProgramAnalysis {
    pub fns: Vec<FnAnalysis>,
    pub fn_clobbers: Vec<RegMask>,
}

impl ProgramAnalysis {
    /// Analyzes every function in `program` once in a single bottom-up pass.
    pub fn analyze(program: &IrProgram) -> Self {
        let n = program.functions.len();
        let mut addr_taken_fns: HashSet<usize> = HashSet::new();
        let mut callees: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, func) in program.functions.iter().enumerate() {
            for bb in &func.blocks {
                for inst in &bb.insts {
                    match inst {
                        IrInst::AddrFn {
                            target: IrFnTarget::Fn(fid),
                            ..
                        } => {
                            addr_taken_fns.insert(*fid as usize);
                        }
                        IrInst::Call { func: cid, .. } if (*cid as usize) < n => {
                            callees[i].push(*cid as usize);
                        }
                        _ => {}
                    }
                }
            }
        }
        let mut machine_method_fns: HashSet<usize> = HashSet::new();
        for mdef in &program.machines {
            let prefix = format!("{}::", mdef.name);
            for (i, func) in program.functions.iter().enumerate() {
                if func.name.starts_with(&prefix) {
                    machine_method_fns.insert(i);
                }
            }
            for st in &mdef.states {
                if let Some(fid) = st.on_enter {
                    machine_method_fns.insert(fid as usize);
                }
                if let Some(fid) = st.on_exit {
                    machine_method_fns.insert(fid as usize);
                }
                for &fid in st.methods.values() {
                    machine_method_fns.insert(fid as usize);
                }
            }
        }
        if let Some(target) = &program.target {
            for (i, func) in program.functions.iter().enumerate() {
                if target.on_swap_fn.as_deref() == Some(func.name.as_str())
                    || target
                        .overlay_slots
                        .iter()
                        .any(|s| s.load_fn.as_deref() == Some(func.name.as_str()))
                {
                    machine_method_fns.insert(i);
                }
            }
        }

        let mut fns = Vec::with_capacity(n);
        for (i, func) in program.functions.iter().enumerate() {
            debug_assert_eq!(
                func.id as usize, i,
                "IrFunction.id must match its index in program.functions"
            );
            let is_leaf = is_leaf_fn(func);
            let has_inline_asm = fn_has_inline_asm(func);
            let uses_p1 = uses_reg_param1(program, func);
            let ix_param_slot = detect_ix_param(program, func, uses_p1);
            let ix_local_slot = if ix_param_slot.is_none() {
                detect_ix_local(program, func)
            } else {
                None
            };
            let ix_static_elem = if ix_param_slot.is_none() && ix_local_slot.is_none() {
                detect_ix_static_elem(program, func)
            } else {
                None
            };
            let ix_in_reg = ix_param_slot.is_some()
                && !func.is_pub
                && !func.is_test
                && func.interrupt_source.is_none()
                && func.overlay_state.is_none()
                && func.owner_state.is_none()
                && !addr_taken_fns.contains(&i)
                && !machine_method_fns.contains(&i);
            let reg_alloc = regalloc::allocate_fn_regs(
                program,
                func,
                uses_p1,
                ix_param_slot.or(ix_local_slot),
            );
            let escaped_locals = func.escaped_locals();
            let val_sizes = collect_val_sizes(program, func);
            let fused_cmp_dsts = collect_fused_cmp_dsts(func);
            fns.push(FnAnalysis {
                is_leaf,
                has_inline_asm,
                uses_reg_param1: uses_p1,
                ix_param_slot,
                ix_local_slot,
                ix_static_elem,
                ix_in_reg,
                save_ix: false,
                reg_alloc,
                escaped_locals,
                val_sizes,
                fused_cmp_dsts,
            });
        }

        // Compute transitive callees of functions where `IX` must remain preserved across calls
        // (`uses_ix()`, `pub`, `addr_taken`, or `machine_method`), and set `save_ix`
        // only on functions that actually modify `IX` in their own prologue/calls.
        let mut must_preserve_ix = vec![false; n];
        let mut stack: Vec<usize> = Vec::new();
        for (i, func) in program.functions.iter().enumerate() {
            if fns[i].uses_ix()
                || func.is_pub
                || addr_taken_fns.contains(&i)
                || machine_method_fns.contains(&i)
            {
                must_preserve_ix[i] = true;
                stack.push(i);
            }
        }
        while let Some(u) = stack.pop() {
            for &v in &callees[u] {
                if !must_preserve_ix[v] {
                    must_preserve_ix[v] = true;
                    stack.push(v);
                }
            }
        }
        for (i, func) in program.functions.iter().enumerate() {
            let calls_ix_in_reg = callees[i].iter().any(|&c| fns[c].ix_in_reg);
            let modifies_ix = (fns[i].uses_ix() && !fns[i].ix_in_reg)
                || (!fns[i].uses_ix() && calls_ix_in_reg);
            fns[i].save_ix = modifies_ix && func.interrupt_source.is_none() && must_preserve_ix[i];
        }

        let mut fn_clobbers = regalloc::compute_fn_clobbers(program, &fns);
        // Refine register allocation using converged IPRA callee clobber masks so per-register
        // `live_calls` profitability reflects whether callees actually clobber `B` vs `C`.
        for _ in 0..4 {
            for (i, func) in program.functions.iter().enumerate() {
                fns[i].reg_alloc = regalloc::allocate_fn_regs_with_clobbers(
                    program,
                    func,
                    fns[i].uses_reg_param1,
                    fns[i].ix_slot(),
                    Some(&fn_clobbers),
                );
            }
            let next_clobbers = regalloc::compute_fn_clobbers(program, &fns);
            if next_clobbers == fn_clobbers {
                break;
            }
            fn_clobbers = next_clobbers;
        }
        Self { fns, fn_clobbers }
    }

    /// Returns the set of Z80 function symbols that preserve `B`, `C`, or both across calls,
    /// so `peephole::promote_single_block_spills_to_bc` must NOT introduce new `B`/`C` clobbers.
    pub fn no_bc_fns(&self, program: &IrProgram) -> HashSet<String> {
        let mut out = HashSet::new();
        for (i, func) in program.functions.iter().enumerate() {
            if !func.is_test && !self.fn_clobbers[i].contains(RegMask::BC) {
                out.insert(z80_sym(&func.name));
            }
        }
        out
    }
}

pub(crate) fn is_signed_ty(ty: &IrType) -> bool {
    match ty {
        IrType::Int(it) | IrType::Fixed { base: it, .. } => it.is_signed(),
        _ => false,
    }
}

pub(crate) fn bb_uses_val(bb: &BasicBlock, target: ValueId) -> bool {
    if bb.insts.iter().any(|i| i.uses_val(target)) {
        return true;
    }
    matches!(
        bb.term,
        Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } if v == target
    )
}

pub(crate) fn count_val_uses(func: &IrFunction, target: ValueId) -> usize {
    let mut n = 0usize;
    for bb in &func.blocks {
        for inst in &bb.insts {
            if inst.uses_val(target) {
                n += 1;
            }
        }
        if matches!(
            bb.term,
            Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } if v == target
        ) {
            n += 1;
        }
    }
    n
}

pub(crate) fn is_val_used_outside_term(
    func: &IrFunction,
    target: ValueId,
    skip_bb_idx: usize,
) -> bool {
    for (idx, bb) in func.blocks.iter().enumerate() {
        if bb.insts.iter().any(|inst| inst.uses_val(target)) {
            return true;
        }
        if idx != skip_bb_idx
            && matches!(
                bb.term,
                Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } if v == target
            )
        {
            return true;
        }
    }
    false
}

/// Checks if a `LoadLocal` of a register-allocated slot (`B`/`C`/`BC`) at
/// `(bb_idx, inst_idx)` can stay in its assigned register without spilling to scratch RAM.
pub(crate) fn can_alias_reg_load(
    program: &IrProgram,
    func: &IrFunction,
    reg_alloc: &FnRegAlloc,
    fn_clobbers: &[RegMask],
    bb_idx: usize,
    inst_idx: usize,
    dst: ValueId,
    reg: RegLoc,
) -> bool {
    for (i, bb) in func.blocks.iter().enumerate() {
        if i != bb_idx && bb_uses_val(bb, dst) {
            return false;
        }
    }
    let reg_sz = reg.byte_size();
    let bb = &func.blocks[bb_idx];
    let mut last_use: Option<usize> = None;
    for (j, inst) in bb.insts.iter().enumerate().skip(inst_idx + 1) {
        if !inst.uses_val(dst) {
            continue;
        }
        let supported = match inst {
            IrInst::StoreLocal { val, .. } => *val == dst,
            IrInst::StorePtr { addr, val, .. } => *val == dst && *addr != dst,
            IrInst::LoadPtr { addr, .. } => reg_sz == 2 && *addr == dst,
            IrInst::FieldAddr { base, .. } => reg_sz == 2 && *base == dst,
            IrInst::IndexAddr { base, index, .. } => {
                (reg_sz == 1 && *index == dst && *base != dst)
                    || (reg_sz == 2 && *base == dst && *index != dst)
            }
            IrInst::Cast {
                from_ty, to_ty, val, ..
            } => {
                *val == dst
                    && if reg_sz == 1 {
                        matches!(
                            (program.type_size(from_ty), program.type_size(to_ty)),
                            (1, 1) | (1, 2)
                        )
                    } else {
                        matches!(
                            (program.type_size(from_ty), program.type_size(to_ty)),
                            (2, 1) | (2, 2)
                        )
                    }
            }
            IrInst::Call { .. } | IrInst::ExternCall { .. } | IrInst::UnOp { .. } => true,
            IrInst::PortOut { val, port } => reg_sz == 1 && *val == dst && *port != dst,
            IrInst::BinOp {
                op,
                ty,
                lhs,
                rhs,
                ..
            } => {
                let is_cmp = matches!(
                    op,
                    IrBinOp::Eq
                        | IrBinOp::Ne
                        | IrBinOp::Lt
                        | IrBinOp::Le
                        | IrBinOp::Gt
                        | IrBinOp::Ge
                );
                if is_cmp {
                    true
                } else if reg_sz == 1 {
                    program.type_size(ty) == 1
                        && (matches!(
                            op,
                            IrBinOp::Add
                                | IrBinOp::Sub
                                | IrBinOp::BitAnd
                                | IrBinOp::BitOr
                                | IrBinOp::BitXor
                        ) || (matches!(op, IrBinOp::Shl | IrBinOp::Shr)
                            && *lhs == dst
                            && *rhs != dst
                            && func.blocks[bb_idx].insts[..inst_idx]
                                .iter()
                                .any(|pi| matches!(pi, IrInst::ConstInt { dst: cd, .. } if *cd == *rhs))))
                        && !(*lhs == dst && *rhs == dst)
                } else {
                    program.type_size(ty) == 2
                        && matches!(
                            op,
                            IrBinOp::Add
                                | IrBinOp::Sub
                                | IrBinOp::BitAnd
                                | IrBinOp::BitOr
                                | IrBinOp::BitXor
                        )
                }
            }
            _ => false,
        };
        if !supported {
            return false;
        }
        last_use = Some(j);
    }
    let used_in_term = match bb.term {
        Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } => v == dst,
        _ => false,
    };
    let end_idx = if used_in_term {
        bb.insts.len()
    } else if let Some(lu) = last_use {
        lu
    } else {
        return true;
    };
    let mut const_ints: HashMap<ValueId, i64> = HashMap::new();
    for pi in &bb.insts[..=inst_idx] {
        if let IrInst::ConstInt { dst: cd, val, .. } = pi {
            const_ints.insert(*cd, *val);
        }
    }
    for (j, inst) in bb.insts.iter().enumerate().take(end_idx).skip(inst_idx + 1) {
        if let IrInst::ConstInt { dst: cd, val, .. } = inst {
            const_ints.insert(*cd, *val);
            continue;
        }
        if regalloc::inst_internal_clobber_mask(program, inst, &const_ints).intersects(reg.mask()) {
            return false;
        }
        match inst {
            IrInst::StoreLocal { slot, .. } => {
                if let Some(other_reg) = reg_alloc.slot_reg(*slot)
                    && other_reg.mask().intersects(reg.mask())
                {
                    return false;
                }
            }
            IrInst::Call { func: cid, .. } => {
                let cm = fn_clobbers
                    .get(*cid as usize)
                    .copied()
                    .unwrap_or(RegMask::BC);
                if cm.intersects(reg.mask())
                    && !reg_alloc.call_live_mask(bb_idx, j).intersects(reg.mask())
                {
                    return false;
                }
            }
            IrInst::ExternCall { extern_fn, .. } => {
                let em = regalloc::extern_clobber_mask(&program.externs[*extern_fn as usize]);
                if em.intersects(reg.mask())
                    && !reg_alloc.call_live_mask(bb_idx, j).intersects(reg.mask())
                {
                    return false;
                }
            }
            IrInst::CallIndirect { .. }
            | IrInst::StateDispatch { .. }
            | IrInst::StateTransition { .. } => {
                return false;
            }
            _ => {}
        }
    }
    true
}

/// Checks if `LoadLocal { dst, slot }` at `inst_idx` in `bb_idx` can safely alias `slot_addr`
/// directly (0 instructions emitted!) without being clobbered before its last use.
pub(crate) fn can_alias_local_load(
    func: &IrFunction,
    bb_idx: usize,
    inst_idx: usize,
    dst: ValueId,
    slot: SlotId,
) -> bool {
    let is_slice_slot = matches!(
        func.locals.get(slot as usize).map(|l| &l.ty),
        Some(IrType::Slice { .. } | IrType::ConstSlice(_))
    );
    if is_slice_slot {
        let unmodified_unescaped = !func.blocks.iter().any(|b| {
            b.insts.iter().any(|inst| {
                matches!(
                    inst,
                    IrInst::StoreLocal { slot: s, .. } | IrInst::AddrLocal { slot: s, .. }
                        if *s == slot
                )
            })
        });
        if unmodified_unescaped {
            return true;
        }
    }
    for (i, bb) in func.blocks.iter().enumerate() {
        if i != bb_idx && bb_uses_val(bb, dst) {
            return false;
        }
    }
    let bb = &func.blocks[bb_idx];
    let mut last_use: Option<usize> = None;
    for (j, inst) in bb.insts.iter().enumerate().skip(inst_idx + 1) {
        if inst.uses_val(dst) {
            last_use = Some(j);
        }
    }
    let used_in_term = match bb.term {
        Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } => v == dst,
        _ => false,
    };
    let end_idx = if used_in_term {
        bb.insts.len()
    } else if let Some(lu) = last_use {
        if matches!(bb.insts[lu], IrInst::StateTransition { .. }) {
            return false;
        }
        lu
    } else {
        return true;
    };
    for inst in &bb.insts[inst_idx + 1..end_idx] {
        match inst {
            IrInst::StoreLocal { slot: s, .. } if *s == slot => return false,
            IrInst::StorePtr { .. }
            | IrInst::MemSet { .. }
            | IrInst::MemCopy { .. }
            | IrInst::MemFill { .. }
            | IrInst::PoolAlloc { .. }
            | IrInst::PoolFree { .. }
            | IrInst::QueuePush { .. }
            | IrInst::QueuePop { .. }
            | IrInst::BankEnter { .. }
            | IrInst::BankExit { .. }
            | IrInst::CriticalEnter
            | IrInst::CriticalExit
            | IrInst::Call { .. }
            | IrInst::CallIndirect { .. }
            | IrInst::ExternCall { .. }
            | IrInst::StateDispatch { .. }
            | IrInst::StateTransition { .. }
            | IrInst::InlineAsm { .. } => return false,
            _ => {}
        }
    }
    true
}

/// Checks if a `LoadPtr` from `(ix + d)` at `(bb_idx, inst_idx)` can be kept as a lazy
/// `ix_val_loads` entry (0 scratch spill!) and loaded directly by every consumer in `bb_idx`.
pub(crate) fn can_alias_ix_load(
    program: &IrProgram,
    func: &IrFunction,
    bb_idx: usize,
    inst_idx: usize,
    dst: ValueId,
    fused_cmp_dsts: &BTreeSet<ValueId>,
    ix_local_slot: Option<SlotId>,
) -> bool {
    for (i, bb) in func.blocks.iter().enumerate() {
        if i != bb_idx && bb_uses_val(bb, dst) {
            return false;
        }
    }
    let bb = &func.blocks[bb_idx];
    let mut last_use: Option<usize> = None;
    for (j, inst) in bb.insts.iter().enumerate().skip(inst_idx + 1) {
        if !inst.uses_val(dst) {
            continue;
        }
        let supported = match inst {
            IrInst::StoreLocal { val, .. } => *val == dst,
            IrInst::StorePtr { addr, val, .. } => *val == dst && *addr != dst,
            IrInst::LoadPtr { addr, .. } => *addr == dst,
            IrInst::Cast {
                from_ty, to_ty, val, ..
            } => {
                *val == dst
                    && matches!(
                        (program.type_size(from_ty), program.type_size(to_ty)),
                        (1, 1) | (1, 2) | (2, 1) | (2, 2)
                    )
            }
            IrInst::BinOp {
                dst: bd,
                op,
                mode,
                ty,
                lhs,
                rhs,
                ..
            } => {
                if fused_cmp_dsts.contains(bd)
                    || matches!(
                        op,
                        IrBinOp::Eq
                            | IrBinOp::Ne
                            | IrBinOp::Lt
                            | IrBinOp::Le
                            | IrBinOp::Gt
                            | IrBinOp::Ge
                    )
                {
                    true
                } else {
                    let sz = program.type_size(ty);
                    match sz {
                        1 => {
                            *mode != ArithMode::Saturating
                                && matches!(
                                    op,
                                    IrBinOp::Add
                                        | IrBinOp::Sub
                                        | IrBinOp::BitAnd
                                        | IrBinOp::BitOr
                                        | IrBinOp::BitXor
                                )
                                && !(*lhs == dst && *rhs == dst)
                        }
                        2 => matches!(
                            op,
                            IrBinOp::Add
                                | IrBinOp::Sub
                                | IrBinOp::BitAnd
                                | IrBinOp::BitOr
                                | IrBinOp::BitXor
                        ),
                        _ => false,
                    }
                }
            }
            _ => false,
        };
        if !supported {
            return false;
        }
        last_use = Some(j);
    }
    let used_in_term = match bb.term {
        Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } => v == dst,
        _ => false,
    };
    let end_idx = if used_in_term {
        bb.insts.len()
    } else if let Some(lu) = last_use {
        lu
    } else {
        return true;
    };
    for inst in &bb.insts[inst_idx + 1..end_idx] {
        match inst {
            IrInst::StoreLocal { slot, .. } if ix_local_slot == Some(*slot) => return false,
            IrInst::StorePtr { .. }
            | IrInst::MemSet { .. }
            | IrInst::MemCopy { .. }
            | IrInst::MemFill { .. }
            | IrInst::PoolAlloc { .. }
            | IrInst::PoolFree { .. }
            | IrInst::QueuePush { .. }
            | IrInst::QueuePop { .. }
            | IrInst::BankEnter { .. }
            | IrInst::BankExit { .. }
            | IrInst::CriticalEnter
            | IrInst::CriticalExit
            | IrInst::Call { .. }
            | IrInst::CallIndirect { .. }
            | IrInst::ExternCall { .. }
            | IrInst::StateDispatch { .. }
            | IrInst::StateTransition { .. }
            | IrInst::InlineAsm { .. } => return false,
            _ => {}
        }
    }
    true
}

pub(crate) fn is_const_always_folded(
    program: &IrProgram,
    func: &IrFunction,
    val: ValueId,
    fused_cmp_dsts: &BTreeSet<ValueId>,
    skip_insts: &HashSet<(usize, usize)>,
) -> bool {
    for (bi, bb) in func.blocks.iter().enumerate() {
        for (ii, inst) in bb.insts.iter().enumerate() {
            if skip_insts.contains(&(bi, ii)) || !inst.uses_val(val) {
                continue;
            }
            let folded = match inst {
                IrInst::StoreLocal { val: v, .. } => *v == val,
                IrInst::LoadPtr { addr, .. } => *addr == val,
                IrInst::StorePtr { addr, val: v, .. } => *v == val || *addr == val,
                IrInst::BinOp {
                    dst: _,
                    op,
                    mode,
                    ty,
                    lhs,
                    rhs,
                    ..
                } => {
                    let sz = program.type_size(ty);
                    match op {
                        IrBinOp::Shl | IrBinOp::Shr => {
                            ((sz == 2 || sz == 1) && *rhs == val && *lhs != val) || sz == 4
                        }
                        IrBinOp::Add
                        | IrBinOp::Sub
                        | IrBinOp::BitAnd
                        | IrBinOp::BitOr
                        | IrBinOp::BitXor => {
                            sz == 4
                                || sz == 2
                                || (sz == 1
                                    && *mode != ArithMode::Saturating
                                    && ((*rhs == val && *lhs != val)
                                        || (*op == IrBinOp::Sub && *lhs == val && *rhs != val)))
                        }
                        IrBinOp::Mul => sz == 4 || sz == 2,
                        IrBinOp::Div | IrBinOp::Rem => sz == 4,
                        IrBinOp::Eq
                        | IrBinOp::Ne
                        | IrBinOp::Lt
                        | IrBinOp::Le
                        | IrBinOp::Gt
                        | IrBinOp::Ge => {
                            let _ = fused_cmp_dsts;
                            sz == 4 || sz == 2 || sz == 1
                        }
                    }
                }
                IrInst::PortIn { port, .. } => *port == val,
                IrInst::MemSet { val: v, .. } => *v == val,
                IrInst::MakeSlice { ptr, len, .. } => *len == val && *ptr != val,
                IrInst::Call { args, .. } | IrInst::ExternCall { args, .. } => args.contains(&val),
                _ => false,
            };
            if !folded {
                return false;
            }
        }
        if matches!(
            bb.term,
            Terminator::Branch { cond: v, .. } if v == val
        ) {
            return false;
        }
    }
    true
}

/// Returns `true` if every use of a `const_addrs` / `rom_addrs` value `val` across `func`
/// folds the address directly without ever reading `val`'s scratch RAM slot.
pub(crate) fn is_const_addr_always_folded(
    program: &IrProgram,
    func: &IrFunction,
    val: ValueId,
    skip_insts: &HashSet<(usize, usize)>,
) -> bool {
    for (bi, bb) in func.blocks.iter().enumerate() {
        for (ii, inst) in bb.insts.iter().enumerate() {
            if skip_insts.contains(&(bi, ii)) || !inst.uses_val(val) {
                continue;
            }
            let folded = match inst {
                IrInst::FieldAddr { base, .. } => *base == val,
                IrInst::IndexAddr { base, index, .. } => *base == val && *index != val,
                IrInst::LoadPtr { addr, .. } => *addr == val,
                IrInst::StorePtr { addr, val: v, ty, .. } => {
                    *addr == val || (*v == val && program.type_size(ty) == 2)
                }
                IrInst::Cast {
                    val: v,
                    from_ty,
                    to_ty,
                    ..
                } if *v == val
                    && program.type_size(from_ty) == 2
                    && program.type_size(to_ty) == 2 =>
                {
                    true
                }
                IrInst::MemSet { dst_addr, val: v, .. } => *dst_addr == val && *v != val,
                IrInst::MemCopy {
                    dst_addr, src_addr, ..
                } => *dst_addr == val || *src_addr == val,
                IrInst::MemFill { dst_addr, .. } => *dst_addr == val,
                IrInst::MakeSlice { ptr, len, .. } => *ptr == val && *len != val,
                IrInst::PoolElemAddr {
                    pool_addr, handle, ..
                } => *pool_addr == val && *handle != val,
                IrInst::QueuePush {
                    queue_addr,
                    val: qval,
                    elem_ty,
                    ..
                } => *queue_addr == val || (*qval == val && !elem_ty.is_scalar()),
                IrInst::QueuePop {
                    queue_addr,
                    out_addr,
                    ..
                } => *queue_addr == val || *out_addr == val,
                IrInst::QueueIsEmpty { queue_addr, .. } | IrInst::QueueLen { queue_addr, .. } => {
                    *queue_addr == val
                }
                IrInst::StoreLocal { val: v, ty, .. } => {
                    *v == val && program.type_size(ty) == 2
                }
                IrInst::Call { args, .. } | IrInst::ExternCall { args, .. } => {
                    args.contains(&val)
                }
                IrInst::CallIndirect { addr, args } => {
                    *addr == val || args.iter().any(|(a, _)| *a == val)
                }
                IrInst::BinOp {
                    op:
                        IrBinOp::Add
                        | IrBinOp::Sub
                        | IrBinOp::BitAnd
                        | IrBinOp::BitOr
                        | IrBinOp::BitXor,
                    ty,
                    ..
                } if program.type_size(ty) == 2 => true,
                _ => false,
            };
            if !folded {
                return false;
            }
        }
        if matches!(
            bb.term,
            Terminator::Branch { cond: v, .. } if v == val
        ) {
            return false;
        }
    }
    true
}

fn can_use_ix_in_fn(program: &IrProgram, func: &IrFunction) -> bool {
    if func.interrupt_source.is_some() {
        return false;
    }
    for bb in &func.blocks {
        for inst in &bb.insts {
            match inst {
                IrInst::InlineAsm { .. } => return false,
                IrInst::ExternCall { extern_fn, .. } => {
                    if let Some(ef) = program.externs.get(*extern_fn as usize)
                        && ef
                            .clobbers
                            .as_deref()
                            .is_some_and(|cs| cs.iter().any(|c| c.eq_ignore_ascii_case("ix")))
                    {
                        return false;
                    }
                }
                _ => {}
            }
        }
    }
    true
}

fn is_leaf_fn(func: &IrFunction) -> bool {
    for bb in &func.blocks {
        for inst in &bb.insts {
            if matches!(
                inst,
                IrInst::Call { .. }
                    | IrInst::CallIndirect { .. }
                    | IrInst::ExternCall { .. }
                    | IrInst::StateDispatch { .. }
                    | IrInst::StateTransition { .. }
                    | IrInst::InlineAsm { .. }
            ) {
                return false;
            }
        }
    }
    true
}

fn fn_has_inline_asm(func: &IrFunction) -> bool {
    func.blocks
        .iter()
        .any(|bb| bb.insts.iter().any(|i| matches!(i, IrInst::InlineAsm { .. })))
}

/// Returns `true` if `func` uses the WPO-ABI 2nd-parameter register convention
/// (`DE` for 2-byte, `E` for 1-byte) instead of passing param 1 through Uchiguri RAM.
/// State-machine methods/hooks keep RAM passing because table-driven dispatch uses `DE`.
fn uses_reg_param1(program: &IrProgram, func: &IrFunction) -> bool {
    if func.blocks.is_empty()
        || func.param_count < 2
        || func.locals.len() < func.param_count
        || func.interrupt_source.is_some()
    {
        return false;
    }
    let p0_sz = program.type_size(&func.locals[0].ty);
    let p1_sz = program.type_size(&func.locals[1].ty);
    if !matches!(p0_sz, 1 | 2) || !matches!(p1_sz, 1 | 2) {
        return false;
    }
    // SDCC `--sdcccall 1` passes `(u16, u8)` as `(HL, stack 1B)` with callee cleanup, not `(HL, E)`.
    if func.is_pub && p0_sz == 2 && p1_sz == 1 {
        return false;
    }
    for mdef in &program.machines {
        for st in &mdef.states {
            if st.on_enter == Some(func.id)
                || st.on_exit == Some(func.id)
                || st.methods.values().any(|&fid| fid == func.id)
            {
                return false;
            }
        }
    }
    true
}

/// Returns `Some(slot)` (`0` or `1`) if param `slot` is a 2-byte `&T`/`&mut T` whose value
/// (and every transitive `FieldAddr` derived from it) is strictly consumed by `FieldAddr`
/// (within `-128..=125`), `IndexAddr(base)`, `LoadPtr(addr)`, `StorePtr(addr)`, or `Call` (`disp == 0`).
fn detect_ix_param(program: &IrProgram, func: &IrFunction, reg_p1: bool) -> Option<SlotId> {
    if !can_use_ix_in_fn(program, func) || func.param_count == 0 {
        return None;
    }
    let cfg = aun_ir::cfg::Cfg::build(func);
    let mut visit_order: Vec<usize> = cfg.rpo_indices().to_vec();
    let mut seen_bi = vec![false; func.blocks.len()];
    for &bi in &visit_order {
        if bi < seen_bi.len() {
            seen_bi[bi] = true;
        }
    }
    for (bi, &visited) in seen_bi.iter().enumerate() {
        if !visited {
            visit_order.push(bi);
        }
    }
    let max_cand = if reg_p1 { func.param_count.min(2) } else { 1 };
    for cand in 0..max_cand {
        if !matches!(func.locals[cand].ty, IrType::Ref { .. }) {
            continue;
        }
        let cslot = cand as SlotId;
        let mut addr_of_p: HashMap<ValueId, ()> = HashMap::new();
        let mut ix_vals: HashMap<ValueId, i32> = HashMap::new();
        let mut bad = false;

        for &bi in &visit_order {
            let bb = &func.blocks[bi];
            for inst in &bb.insts {
                match inst {
                    IrInst::StoreLocal { slot, .. } if *slot == cslot => {
                        bad = true;
                        break;
                    }
                    IrInst::AddrLocal { dst, slot } if *slot == cslot => {
                        addr_of_p.insert(*dst, ());
                    }
                    IrInst::LoadLocal { dst, slot, .. } if *slot == cslot => {
                        ix_vals.insert(*dst, 0);
                    }
                    IrInst::LoadPtr { dst, addr, .. } if addr_of_p.contains_key(addr) => {
                        ix_vals.insert(*dst, 0);
                    }
                    IrInst::FieldAddr {
                        dst,
                        base,
                        byte_offset,
                    } if ix_vals.contains_key(base) => {
                        let new_disp = ix_vals[base] + (*byte_offset as i32);
                        if !(-128..=124).contains(&new_disp) {
                            bad = true;
                            break;
                        }
                        ix_vals.insert(*dst, new_disp);
                    }
                    _ => {}
                }
            }
            if bad {
                break;
            }
        }
        if bad {
            continue;
        }

        for bb in &func.blocks {
            for inst in &bb.insts {
                for &v in addr_of_p.keys() {
                    if inst.uses_val(v)
                        && !matches!(inst, IrInst::LoadPtr { addr, .. } if *addr == v)
                    {
                        bad = true;
                        break;
                    }
                }
                if bad {
                    break;
                }
                for (&v, &disp) in &ix_vals {
                    if inst.uses_val(v) {
                        let ok = match inst {
                            IrInst::FieldAddr { base, .. } => *base == v,
                            IrInst::IndexAddr { base, index, .. } => *base == v && *index != v,
                            IrInst::LoadPtr { addr, .. } => *addr == v,
                            IrInst::StorePtr { addr, val, .. } => *addr == v && *val != v,
                            IrInst::MemSet { dst_addr, val, .. } => *dst_addr == v && *val != v,
                            IrInst::MemCopy {
                                dst_addr, src_addr, ..
                            } => *dst_addr == v || *src_addr == v,
                            IrInst::Call { .. } => disp == 0,
                            _ => false,
                        };
                        if !ok {
                            bad = true;
                            break;
                        }
                    }
                }
                if bad {
                    break;
                }
            }
            if bad {
                break;
            }
            if let Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } = bb.term
                && (addr_of_p.contains_key(&v) || ix_vals.contains_key(&v))
            {
                bad = true;
                break;
            }
        }
        if !bad {
            return Some(cslot);
        }
    }
    None
}

/// Promotes a hot non-param 2-byte local pointer (or loop-OSR pointer induction variable)
/// to `IX` when every loaded value and `FieldAddr` derived from it is consumed in the same
/// basic block and before any `StoreLocal` to that slot, and updates are either arbitrary
/// reloads or single-use `p + const_i16` steps.
fn detect_ix_local(program: &IrProgram, func: &IrFunction) -> Option<SlotId> {
    if !can_use_ix_in_fn(program, func) {
        return None;
    }
    let cfg = aun_ir::cfg::Cfg::build(func);
    let mut in_loop: HashSet<BlockId> = HashSet::new();
    for lp in cfg.natural_loops() {
        in_loop.extend(lp.body.iter().copied());
    }
    let mut addr_escaped = vec![false; func.locals.len()];
    for bb in &func.blocks {
        for inst in &bb.insts {
            if let IrInst::AddrLocal { slot, .. } = inst
                && (*slot as usize) < addr_escaped.len()
            {
                addr_escaped[*slot as usize] = true;
            }
        }
    }

    let mut best: Option<(usize, SlotId)> = None;
    for s in func.param_count..func.locals.len() {
        let local = &func.locals[s];
        if addr_escaped[s]
            || program.type_size(&local.ty) != 2
            || !matches!(local.ty, IrType::Ref { .. } | IrType::Int(IntType::U16))
        {
            continue;
        }
        let cslot = s as SlotId;
        let mut ix_vals: HashMap<ValueId, (i32, usize, u32)> = HashMap::new();
        let mut step_vals: HashMap<ValueId, (usize, u32)> = HashMap::new();
        let mut store_count = 0usize;
        let mut bad = false;

        for (bi, bb) in func.blocks.iter().enumerate() {
            let mut epoch = 0u32;
            let mut const_ints: HashMap<ValueId, i64> = HashMap::new();
            for inst in &bb.insts {
                match inst {
                    IrInst::ConstInt { dst, val, .. } => {
                        const_ints.insert(*dst, *val);
                    }
                    IrInst::LoadLocal { dst, slot, .. } if *slot == cslot => {
                        ix_vals.insert(*dst, (0, bi, epoch));
                    }
                    IrInst::FieldAddr {
                        dst,
                        base,
                        byte_offset,
                    } if ix_vals.contains_key(base) => {
                        let (d, b, e) = ix_vals[base];
                        if b != bi || e != epoch {
                            bad = true;
                            break;
                        }
                        let new_disp = d + (*byte_offset as i32);
                        if !(-128..=124).contains(&new_disp) {
                            bad = true;
                            break;
                        }
                        ix_vals.insert(*dst, (new_disp, bi, epoch));
                    }
                    IrInst::BinOp {
                        dst,
                        op: IrBinOp::Add,
                        ty,
                        lhs,
                        rhs,
                        ..
                    } if program.type_size(ty) == 2 => {
                        let is_step = (ix_vals.get(lhs) == Some(&(0, bi, epoch))
                            && const_ints
                                .get(rhs)
                                .is_some_and(|&k| (-32768..=32767).contains(&k)))
                            || (ix_vals.get(rhs) == Some(&(0, bi, epoch))
                                && const_ints
                                    .get(lhs)
                                    .is_some_and(|&k| (-32768..=32767).contains(&k)));
                        if is_step {
                            step_vals.insert(*dst, (bi, epoch));
                        }
                    }
                    IrInst::StoreLocal { slot, val, .. } if *slot == cslot => {
                        store_count += 1;
                        if let Some(&(sb, se)) = step_vals.get(val)
                            && (sb != bi || se != epoch)
                        {
                            bad = true;
                            break;
                        }
                        if let Some(&(_, vb, ve)) = ix_vals.get(val)
                            && (vb != bi || ve != epoch)
                        {
                            bad = true;
                            break;
                        }
                        epoch += 1;
                    }
                    _ => {}
                }
            }
            if bad {
                break;
            }
        }
        if bad || store_count == 0 {
            continue;
        }
        if step_vals.keys().any(|&v| count_val_uses(func, v) != 1) {
            continue;
        }

        let mut score = 0usize;
        let mut deref_or_field_count = 0usize;
        for (bi, bb) in func.blocks.iter().enumerate() {
            let weight = if in_loop.contains(&bb.id) { 8 } else { 1 };
            let mut epoch = 0u32;
            for inst in &bb.insts {
                for (&v, &(def_bi, def_ep)) in &step_vals {
                    if inst.uses_val(v) {
                        let ok = bi == def_bi
                            && epoch == def_ep
                            && matches!(
                                inst,
                                IrInst::StoreLocal { slot, val, .. } if *slot == cslot && *val == v
                            );
                        if !ok {
                            bad = true;
                            break;
                        }
                    }
                }
                if bad {
                    break;
                }
                for (&v, &(disp, def_bi, def_ep)) in &ix_vals {
                    if inst.uses_val(v) {
                        if bi != def_bi || epoch != def_ep {
                            bad = true;
                            break;
                        }
                        let ok = match inst {
                            IrInst::FieldAddr { base, .. } => {
                                if *base == v {
                                    score += weight;
                                    deref_or_field_count += 1;
                                    true
                                } else {
                                    false
                                }
                            }
                            IrInst::IndexAddr { base, index, .. } => *base == v && *index != v,
                            IrInst::LoadPtr { addr, .. } => {
                                if *addr == v {
                                    score += weight;
                                    deref_or_field_count += 1;
                                    true
                                } else {
                                    false
                                }
                            }
                            IrInst::StorePtr { addr, val, .. } => {
                                if *addr == v && *val != v {
                                    score += weight;
                                    deref_or_field_count += 1;
                                    true
                                } else {
                                    false
                                }
                            }
                            IrInst::MemSet { dst_addr, val, .. } => {
                                if *dst_addr == v && *val != v {
                                    score += weight;
                                    deref_or_field_count += 1;
                                    true
                                } else {
                                    false
                                }
                            }
                            IrInst::MemCopy {
                                dst_addr, src_addr, ..
                            } => {
                                if *dst_addr == v || *src_addr == v {
                                    score += weight;
                                    deref_or_field_count += 1;
                                    true
                                } else {
                                    false
                                }
                            }
                            IrInst::Call { .. } => disp == 0,
                            IrInst::BinOp { dst, .. } if step_vals.contains_key(dst) => disp == 0,
                            IrInst::StoreLocal { slot, val, .. } => *slot == cslot && *val == v,
                            _ => false,
                        };
                        if !ok {
                            bad = true;
                            break;
                        }
                    }
                }
                if bad {
                    break;
                }
                if matches!(inst, IrInst::StoreLocal { slot, .. } if *slot == cslot) {
                    epoch += 1;
                }
            }
            if bad {
                break;
            }
            if let Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } = bb.term
                && (ix_vals.contains_key(&v) || step_vals.contains_key(&v))
            {
                bad = true;
                break;
            }
        }
        if !bad
            && score >= 4
            && deref_or_field_count >= 1
            && best.map_or(true, |(best_score, _)| score > best_score)
        {
            best = Some((score, cslot));
        }
    }
    best.map(|(_, s)| s)
}

/// Promotes a hot invariant `&STATIC_GLOBAL[.field*][idx]` element pointer to `IX` when `IX`
/// is otherwise unused and all derived field accesses fit `-128..=124` (`>= 3` field/ptr accesses,
/// or `>= 2` inside a loop).
fn detect_ix_static_elem(program: &IrProgram, func: &IrFunction) -> Option<IxStaticElem> {
    if !can_use_ix_in_fn(program, func) || func.blocks.is_empty() {
        return None;
    }
    let cfg = aun_ir::cfg::Cfg::build(func);
    let mut in_loop: HashSet<BlockId> = HashSet::new();
    for lp in cfg.natural_loops() {
        in_loop.extend(lp.body.iter().copied());
    }
    let mut addr_escaped = vec![false; func.locals.len()];
    let mut store_counts = vec![0usize; func.locals.len()];
    for bb in &func.blocks {
        for inst in &bb.insts {
            match inst {
                IrInst::AddrLocal { slot, .. } if (*slot as usize) < addr_escaped.len() => {
                    addr_escaped[*slot as usize] = true;
                }
                IrInst::StoreLocal { slot, .. } if (*slot as usize) < store_counts.len() => {
                    store_counts[*slot as usize] += 1;
                }
                _ => {}
            }
        }
    }

    let mut static_bases: HashMap<ValueId, (u32, u16)> = HashMap::new();
    let mut const_ints: HashSet<ValueId> = HashSet::new();
    let mut param_idx_vals: HashMap<ValueId, SlotId> = HashMap::new();

    #[derive(Clone, Copy)]
    struct SiteInfo {
        dst: ValueId,
        global: u32,
        base_offset: u16,
        elem_size: u16,
        len: u16,
        param_slot: Option<SlotId>,
    }
    let mut sites: Vec<SiteInfo> = Vec::new();

    for &bi in cfg.rpo_indices() {
        let bb = &func.blocks[bi];
        for inst in &bb.insts {
            match inst {
                IrInst::ConstInt { dst, .. } => {
                    const_ints.insert(*dst);
                }
                IrInst::AddrGlobal { dst, global } => {
                    static_bases.insert(*dst, (*global, 0));
                }
                IrInst::FieldAddr {
                    dst,
                    base,
                    byte_offset,
                } => {
                    if let Some(&(gid, off)) = static_bases.get(base)
                        && let Some(new_off) = off.checked_add(*byte_offset)
                    {
                        static_bases.insert(*dst, (gid, new_off));
                    }
                }
                IrInst::Cast {
                    dst,
                    val,
                    from_ty,
                    to_ty,
                } => {
                    if program.type_size(from_ty) == 2
                        && program.type_size(to_ty) == 2
                        && let Some(&b) = static_bases.get(val)
                    {
                        static_bases.insert(*dst, b);
                    } else if let Some(&slot) = param_idx_vals.get(val)
                        && !is_signed_ty(from_ty)
                        && program.type_size(from_ty) == 1
                        && program.type_size(to_ty) == 2
                    {
                        param_idx_vals.insert(*dst, slot);
                    }
                }
                IrInst::LoadLocal { dst, slot, ty } => {
                    let s = *slot as usize;
                    if s < func.param_count
                        && !addr_escaped[s]
                        && store_counts[s] == 0
                        && matches!(program.type_size(ty), 1 | 2)
                        && !is_signed_ty(ty)
                    {
                        param_idx_vals.insert(*dst, *slot);
                    }
                }
                IrInst::IndexAddr {
                    dst,
                    base,
                    index,
                    elem_size,
                    len,
                    ..
                } if *elem_size >= 1
                    && !const_ints.contains(index)
                    && static_bases.contains_key(base) =>
                {
                    let (gid, base_off) = static_bases[base];
                    sites.push(SiteInfo {
                        dst: *dst,
                        global: gid,
                        base_offset: base_off,
                        elem_size: *elem_size,
                        len: *len,
                        param_slot: param_idx_vals.get(index).copied(),
                    });
                }
                _ => {}
            }
        }
    }

    if sites.is_empty() {
        return None;
    }

    struct Cand {
        global: u32,
        base_offset: u16,
        elem_size: u16,
        len: u16,
        hoist_param: Option<SlotId>,
        roots: BTreeSet<ValueId>,
    }
    let mut cands: Vec<Cand> = Vec::new();
    let mut by_param_key: HashMap<(u32, u16, SlotId, u16), Vec<SiteInfo>> = HashMap::new();
    for &s in &sites {
        if let Some(pslot) = s.param_slot {
            by_param_key
                .entry((s.global, s.base_offset, pslot, s.elem_size))
                .or_default()
                .push(s);
        }
        // Single-root candidate (works for any dynamic index, not just parameters).
        cands.push(Cand {
            global: s.global,
            base_offset: s.base_offset,
            elem_size: s.elem_size,
            len: s.len,
            hoist_param: s.param_slot,
            roots: BTreeSet::from([s.dst]),
        });
    }
    for ((gid, base_off, pslot, esz), group) in by_param_key {
        if group.len() >= 2 {
            let group_len = if group.iter().all(|s| s.len == 0) {
                0
            } else {
                1
            };
            cands.push(Cand {
                global: gid,
                base_offset: base_off,
                elem_size: esz,
                len: group_len,
                hoist_param: Some(pslot),
                roots: group.into_iter().map(|s| s.dst).collect(),
            });
        }
    }

    let mut best: Option<((usize, bool, usize), IxStaticElem)> = None;
    for cand in cands {
        let mut ix_vals: HashMap<ValueId, i32> = HashMap::new();
        for &r in &cand.roots {
            ix_vals.insert(r, 0);
        }
        // Discover local slots (e.g., inlined `&T`/`&mut T` params) whose ONLY stores copy `IX + 0`
        // and whose `AddrLocal`s are only consumed by `LoadPtr`.
        let mut candidate_alias_slots: BTreeSet<SlotId> = BTreeSet::new();
        let mut slot_store_vals: HashMap<SlotId, Vec<ValueId>> = HashMap::new();
        let mut slot_addr_vals: HashMap<SlotId, Vec<ValueId>> = HashMap::new();
        for bb in &func.blocks {
            for inst in &bb.insts {
                match inst {
                    IrInst::StoreLocal { slot, val, .. } if (*slot as usize) >= func.param_count => {
                        slot_store_vals.entry(*slot).or_default().push(*val);
                    }
                    IrInst::AddrLocal { dst, slot } => {
                        slot_addr_vals.entry(*slot).or_default().push(*dst);
                    }
                    _ => {}
                }
            }
        }
        let mut alias_addr_vals: HashSet<ValueId> = HashSet::new();
        for (&slot, vals) in &slot_store_vals {
            if vals.iter().all(|v| ix_vals.get(v) == Some(&0)) {
                let addrs = slot_addr_vals.get(&slot).map(Vec::as_slice).unwrap_or(&[]);
                let addrs_only_loadptr = addrs.iter().all(|&av| {
                    func.blocks.iter().all(|b| {
                        b.insts.iter().all(|u| {
                            !u.uses_val(av)
                                || matches!(u, IrInst::LoadPtr { addr, volatile: false, .. } if *addr == av)
                        }) && !matches!(
                            b.term,
                            Terminator::Return(Some(tv)) | Terminator::Branch { cond: tv, .. } if tv == av
                        )
                    })
                });
                if addrs_only_loadptr {
                    candidate_alias_slots.insert(slot);
                    alias_addr_vals.extend(addrs.iter().copied());
                }
            }
        }

        let mut bad = false;
        for &bi in cfg.rpo_indices() {
            let bb = &func.blocks[bi];
            for inst in &bb.insts {
                match inst {
                    IrInst::LoadLocal { dst, slot, .. }
                        if candidate_alias_slots.contains(slot) =>
                    {
                        ix_vals.insert(*dst, 0);
                    }
                    IrInst::LoadPtr { dst, addr, .. } if alias_addr_vals.contains(addr) => {
                        ix_vals.insert(*dst, 0);
                    }
                    IrInst::FieldAddr {
                        dst,
                        base,
                        byte_offset,
                    } if ix_vals.contains_key(base) => {
                        let d = ix_vals[base];
                        let new_disp = d + (*byte_offset as i32);
                        if !(-128..=124).contains(&new_disp) {
                            bad = true;
                            break;
                        }
                        ix_vals.insert(*dst, new_disp);
                    }
                    _ => {}
                }
            }
            if bad {
                break;
            }
        }
        if bad {
            continue;
        }

        let mut access_count = 0usize;
        let mut in_loop_access_count = 0usize;
        let mut access_score = 0usize;
        for bb in &func.blocks {
            let in_lp = in_loop.contains(&bb.id);
            let weight = if in_lp { 8 } else { 1 };
            for inst in &bb.insts {
                for (&v, &disp) in &ix_vals {
                    if !inst.uses_val(v) {
                        continue;
                    }
                    let ok = match inst {
                        IrInst::FieldAddr { base, .. } => *base == v,
                        IrInst::IndexAddr { base, index, .. } => *base == v && *index != v,
                        IrInst::StoreLocal { slot, val, .. } => {
                            candidate_alias_slots.contains(slot) && *val == v && disp == 0
                        }
                        IrInst::LoadPtr { addr, .. } => {
                            if *addr == v {
                                access_count += 1;
                                if in_lp {
                                    in_loop_access_count += 1;
                                }
                                access_score += weight;
                                true
                            } else {
                                false
                            }
                        }
                        IrInst::StorePtr { addr, val, .. } => {
                            if *addr == v && *val != v {
                                access_count += 1;
                                if in_lp {
                                    in_loop_access_count += 1;
                                }
                                access_score += weight;
                                true
                            } else {
                                false
                            }
                        }
                        IrInst::MemSet { dst_addr, val, .. } => {
                            if *dst_addr == v && *val != v {
                                access_count += 1;
                                if in_lp {
                                    in_loop_access_count += 1;
                                }
                                access_score += weight;
                                true
                            } else {
                                false
                            }
                        }
                        IrInst::MemCopy {
                            dst_addr, src_addr, ..
                        } => {
                            if *dst_addr == v || *src_addr == v {
                                access_count += 1;
                                if in_lp {
                                    in_loop_access_count += 1;
                                }
                                access_score += weight;
                                true
                            } else {
                                false
                            }
                        }
                        IrInst::MemFill { dst_addr, .. } => {
                            if *dst_addr == v {
                                access_count += 1;
                                if in_lp {
                                    in_loop_access_count += 1;
                                }
                                access_score += weight;
                                true
                            } else {
                                false
                            }
                        }
                        IrInst::Call { .. } => disp == 0,
                        _ => false,
                    };
                    if !ok {
                        bad = true;
                        break;
                    }
                }
                if bad {
                    break;
                }
            }
            if bad {
                break;
            }
            if let Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } = bb.term
                && ix_vals.contains_key(&v)
            {
                bad = true;
                break;
            }
        }
        if bad || (access_count < 3 && in_loop_access_count < 2) {
            continue;
        }
        let rank = (access_score, cand.hoist_param.is_some(), access_count);
        if best.as_ref().is_none_or(|(br, _)| rank > *br) {
            best = Some((
                rank,
                IxStaticElem {
                    global: cand.global,
                    base_offset: cand.base_offset,
                    elem_size: cand.elem_size,
                    len: cand.len,
                    hoist_param: cand.hoist_param,
                    index_vals: cand.roots,
                    alias_slots: candidate_alias_slots,
                },
            ));
        }
    }

    best.map(|(_, elem)| elem)
}

/// Selects the hottest eligible 1-byte non-escaping local slot in `func` to live permanently
/// in Z80 register `C` (0 bytes of Uchiguri RAM, 1-byte loads/stores/ALU).
fn collect_val_sizes(program: &IrProgram, func: &IrFunction) -> HashMap<ValueId, u16> {
    let mut sizes = HashMap::new();
    for bb in &func.blocks {
        for inst in &bb.insts {
            match inst {
                IrInst::ConstInt { dst, ty, .. } => {
                    sizes.insert(*dst, program.type_size(ty));
                }
                IrInst::ConstBool { dst, .. } | IrInst::StateTag { dst, .. } => {
                    sizes.insert(*dst, 1);
                }
                IrInst::BinOp { dst, op, ty, .. } => {
                    let sz = match op {
                        IrBinOp::Eq
                        | IrBinOp::Ne
                        | IrBinOp::Lt
                        | IrBinOp::Le
                        | IrBinOp::Gt
                        | IrBinOp::Ge => 1,
                        _ => program.type_size(ty),
                    };
                    sizes.insert(*dst, sz);
                }
                IrInst::UnOp { dst, ty, .. } => {
                    sizes.insert(*dst, program.type_size(ty));
                }
                IrInst::Cast { dst, to_ty, .. }
                | IrInst::MakeSlice { dst, ty: to_ty, .. } => {
                    sizes.insert(*dst, program.type_size(to_ty));
                }
                IrInst::FixedInt { dst, base, .. } => {
                    sizes.insert(*dst, base.byte_size());
                }
                IrInst::LoadLocal { dst, ty, .. }
                | IrInst::LoadPtr { dst, ty, .. }
                | IrInst::Phi { dst, ty, .. } => {
                    sizes.insert(*dst, program.type_size(ty));
                }
                IrInst::AddrLocal { dst, .. }
                | IrInst::AddrGlobal { dst, .. }
                | IrInst::AddrFn { dst, .. }
                | IrInst::FieldAddr { dst, .. }
                | IrInst::IndexAddr { dst, .. }
                | IrInst::SlicePtr { dst, .. }
                | IrInst::SliceLen { dst, .. }
                | IrInst::PoolElemAddr { dst, .. } => {
                    sizes.insert(*dst, 2);
                }
                IrInst::PoolAlloc { dst, .. }
                | IrInst::PortIn { dst, .. }
                | IrInst::QueuePush { dst, .. }
                | IrInst::QueuePop { dst, .. }
                | IrInst::QueueIsEmpty { dst, .. }
                | IrInst::QueueLen { dst, .. } => {
                    sizes.insert(*dst, 1);
                }
                IrInst::Call {
                    dst: Some(d),
                    func: callee_id,
                    ..
                } => {
                    let sz = program
                        .functions
                        .get(*callee_id as usize)
                        .map(|f| program.type_size(&f.ret_ty))
                        .unwrap_or(2);
                    sizes.insert(*d, sz);
                }
                IrInst::ExternCall {
                    dst: Some(d),
                    extern_fn,
                    ..
                } => {
                    let sz = program
                        .externs
                        .get(*extern_fn as usize)
                        .and_then(|e| e.ret.as_ref())
                        .map(|(ty, _)| program.type_size(ty))
                        .unwrap_or(2);
                    sizes.insert(*d, sz);
                }
                _ => {}
            }
        }
    }
    sizes
}

fn collect_fused_cmp_dsts(func: &IrFunction) -> BTreeSet<ValueId> {
    let mut fused_cmp_dsts = BTreeSet::new();
    for (bb_idx, bb) in func.blocks.iter().enumerate() {
        if let Terminator::Branch { cond, .. } = bb.term
            && let Some(IrInst::BinOp { dst, op, .. }) = bb.insts.last()
            && *dst == cond
            && matches!(
                op,
                IrBinOp::Eq | IrBinOp::Ne | IrBinOp::Lt | IrBinOp::Le | IrBinOp::Gt | IrBinOp::Ge
            )
            && !is_val_used_outside_term(func, cond, bb_idx)
        {
            fused_cmp_dsts.insert(*dst);
        }
    }
    fused_cmp_dsts
}
