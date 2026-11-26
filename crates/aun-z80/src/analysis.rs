//! Single-source-of-truth Z80 whole-program and per-function static analyses.
//!
//! Computed once per compilation unit after Layer 2 (`aun-ir::opt`) and shared across
//! *Uchiguri* static RAM allocation, Z80 codegen, and peephole/SDASZ80 emission.

use aun_ir::*;
use std::collections::{BTreeSet, HashMap, HashSet};

/// Converts an IR function/method name (which may contain `::`) into a valid Z80 label.
pub fn z80_sym(name: &str) -> String {
    name.replace("::", "__")
}

/// Per-function static Z80 backend facts derived solely from `IrProgram` + `IrFunction`.
#[derive(Debug, Clone)]
pub struct FnAnalysis {
    pub is_leaf: bool,
    pub has_inline_asm: bool,
    pub uses_reg_param1: bool,
    pub ix_param_slot: Option<SlotId>,
    pub reg_c_slot: Option<SlotId>,
    pub escaped_locals: Vec<bool>,
    pub val_sizes: HashMap<ValueId, u16>,
    pub fused_cmp_dsts: BTreeSet<ValueId>,
}

/// Whole-program static Z80 analysis shared by `uchiguri`, `codegen`, and `lib`.
#[derive(Debug, Clone)]
pub struct ProgramAnalysis {
    pub fns: Vec<FnAnalysis>,
    pub fn_clobbers_c: Vec<bool>,
    pub any_nonleaf_uses_ix: bool,
}

impl ProgramAnalysis {
    /// Analyzes every function in `program` once in a single bottom-up pass.
    pub fn analyze(program: &IrProgram) -> Self {
        let mut fns = Vec::with_capacity(program.functions.len());
        for func in &program.functions {
            let is_leaf = is_leaf_fn(func);
            let has_inline_asm = fn_has_inline_asm(func);
            let uses_p1 = uses_reg_param1(program, func);
            let ix_param_slot = detect_ix_param(program, func, uses_p1);
            let reg_c_slot = compute_reg_c_slot(program, func, uses_p1);
            let escaped_locals = func.escaped_locals();
            let val_sizes = collect_val_sizes(program, func);
            let fused_cmp_dsts = collect_fused_cmp_dsts(func);
            fns.push(FnAnalysis {
                is_leaf,
                has_inline_asm,
                uses_reg_param1: uses_p1,
                ix_param_slot,
                reg_c_slot,
                escaped_locals,
                val_sizes,
                fused_cmp_dsts,
            });
        }
        let fn_clobbers_c = compute_fn_clobbers_c(program, &fns);
        let any_nonleaf_uses_ix = program
            .functions
            .iter()
            .enumerate()
            .any(|(i, f)| !f.is_test && !fns[i].is_leaf && fns[i].ix_param_slot.is_some());
        Self {
            fns,
            fn_clobbers_c,
            any_nonleaf_uses_ix,
        }
    }

    /// Returns the set of Z80 function symbols guaranteed not to clobber register `BC`/`C`.
    pub fn no_bc_fns(&self, program: &IrProgram) -> HashSet<String> {
        let mut out = HashSet::new();
        for (i, func) in program.functions.iter().enumerate() {
            if !func.is_test && !self.fn_clobbers_c[i] {
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

/// Returns `true` if `ef` reads a parameter in `C`/`BC`, clobbers `C`/`BC`/`ALL`, or
/// returns in `B`/`C`/`BC` (requiring caller `C`-register preservation).
pub(crate) fn extern_clobbers_c(ef: &IrExternFn) -> bool {
    ef.params.iter().any(|(_, _, r)| {
        matches!(r.to_ascii_lowercase().as_str(), "c" | "bc")
    }) || ef.clobbers.iter().any(|r| {
        matches!(r.to_ascii_lowercase().as_str(), "c" | "bc" | "all")
    }) || ef.ret.as_ref().is_some_and(|(_, r)| {
        matches!(r.to_ascii_lowercase().as_str(), "b" | "c" | "bc")
    })
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
            | IrInst::PoolAlloc { .. }
            | IrInst::PoolFree { .. }
            | IrInst::Call { .. }
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
            IrInst::StorePtr { .. }
            | IrInst::PoolAlloc { .. }
            | IrInst::PoolFree { .. }
            | IrInst::Call { .. }
            | IrInst::ExternCall { .. }
            | IrInst::StateDispatch { .. }
            | IrInst::StateTransition { .. }
            | IrInst::InlineAsm { .. } => return false,
            _ => {}
        }
    }
    true
}

/// Checks if a `LoadLocal` of the `C`-allocated slot at `(bb_idx, inst_idx)` can stay
/// directly in `C` (`val_in_c`) without spilling to scratch RAM.
pub(crate) fn can_alias_c_load(
    program: &IrProgram,
    func: &IrFunction,
    bb_idx: usize,
    inst_idx: usize,
    dst: ValueId,
    eff_slot: SlotId,
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
            IrInst::IndexAddr { base, index, .. } => *index == dst && *base != dst,
            IrInst::Cast {
                from_ty, to_ty, val, ..
            } => {
                *val == dst
                    && matches!(
                        (program.type_size(from_ty), program.type_size(to_ty)),
                        (1, 1) | (1, 2)
                    )
            }
            IrInst::Call { .. } | IrInst::ExternCall { .. } => true,
            IrInst::BinOp {
                op,
                mode,
                ty,
                lhs,
                rhs,
                ..
            } => {
                matches!(
                    op,
                    IrBinOp::Eq
                        | IrBinOp::Ne
                        | IrBinOp::Lt
                        | IrBinOp::Le
                        | IrBinOp::Gt
                        | IrBinOp::Ge
                ) || (program.type_size(ty) == 1
                    && *mode != ArithMode::Saturating
                    && matches!(
                        op,
                        IrBinOp::Add
                            | IrBinOp::Sub
                            | IrBinOp::BitAnd
                            | IrBinOp::BitOr
                            | IrBinOp::BitXor
                    )
                    && !(*lhs == dst && *rhs == dst))
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
        if let IrInst::StoreLocal { slot, .. } = inst
            && *slot == eff_slot
        {
            return false;
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
                IrInst::StorePtr { addr, val: v, .. } => *v == val && *addr != val,
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
                            (sz == 2 || sz == 1) && *rhs == val && *lhs != val
                        }
                        IrBinOp::Add
                        | IrBinOp::Sub
                        | IrBinOp::BitAnd
                        | IrBinOp::BitOr
                        | IrBinOp::BitXor => {
                            sz == 2
                                || (sz == 1
                                    && *mode != ArithMode::Saturating
                                    && ((*rhs == val && *lhs != val)
                                        || (*op == IrBinOp::Sub && *lhs == val && *rhs != val)))
                        }
                        IrBinOp::Eq
                        | IrBinOp::Ne
                        | IrBinOp::Lt
                        | IrBinOp::Le
                        | IrBinOp::Gt
                        | IrBinOp::Ge => {
                            let _ = fused_cmp_dsts;
                            sz == 2 || sz == 1
                        }
                        _ => false,
                    }
                }
                IrInst::PortIn { port, .. } => *port == val,
                IrInst::Call { args, .. } | IrInst::ExternCall { args, .. } => args.contains(&val),
                _ => false,
            };
            if !folded {
                return false;
            }
        }
        if matches!(
            bb.term,
            Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } if v == val
        ) {
            return false;
        }
    }
    true
}

/// Returns `true` if every use of a `const_addrs` value `val` across the function
/// folds `const_addrs` directly without ever reading `val`'s scratch RAM slot.
pub(crate) fn is_const_addr_always_folded(
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
                IrInst::StorePtr { addr, val: v, .. } => *addr == val && *v != val,
                IrInst::PoolElemAddr {
                    pool_addr, handle, ..
                } => *pool_addr == val && *handle != val,
                _ => false,
            };
            if !folded {
                return false;
            }
        }
        if matches!(
            bb.term,
            Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } if v == val
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
                        && ef.clobbers.iter().any(|c| c.eq_ignore_ascii_case("ix"))
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
    if func.param_count < 2 || func.interrupt_source.is_some() {
        return false;
    }
    if !matches!(program.type_size(&func.locals[1].ty), 1 | 2) {
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
    let max_cand = if reg_p1 { func.param_count.min(2) } else { 1 };
    for cand in 0..max_cand {
        if !matches!(func.locals[cand].ty, IrType::Ref { .. }) {
            continue;
        }
        let cslot = cand as SlotId;
        let mut addr_of_p: HashMap<ValueId, ()> = HashMap::new();
        let mut ix_vals: HashMap<ValueId, i32> = HashMap::new();
        let mut bad = false;

        for bb in &func.blocks {
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
                        if !(-128..=125).contains(&new_disp) {
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

/// Selects the hottest eligible 1-byte non-escaping local slot in `func` to live permanently
/// in Z80 register `C` (0 bytes of Uchiguri RAM, 1-byte loads/stores/ALU).
fn compute_reg_c_slot(program: &IrProgram, func: &IrFunction, reg_p1: bool) -> Option<SlotId> {
    if func.locals.is_empty() || func.blocks.iter().all(|b| b.insts.is_empty()) {
        return None;
    }
    let mut addr_taken = vec![false; func.locals.len()];
    let mut uses = vec![0usize; func.locals.len()];
    let mut calls = 0usize;
    let mut const_ints: BTreeSet<ValueId> = BTreeSet::new();
    for bb in &func.blocks {
        for inst in &bb.insts {
            match inst {
                IrInst::ConstInt { dst, .. } => {
                    const_ints.insert(*dst);
                }
                IrInst::InlineAsm { .. }
                | IrInst::PoolAlloc { .. }
                | IrInst::PoolFree { .. }
                | IrInst::PoolElemAddr { .. }
                | IrInst::StateTransition { .. }
                | IrInst::StateDispatch { .. } => return None,
                IrInst::PortIn { port, .. } | IrInst::PortOut { port, .. } => {
                    if !const_ints.contains(port) {
                        return None;
                    }
                }
                IrInst::BinOp {
                    op: IrBinOp::Mul,
                    ty,
                    lhs,
                    rhs,
                    ..
                } if program.type_size(ty) == 2
                    && !const_ints.contains(lhs)
                    && !const_ints.contains(rhs) =>
                {
                    return None
                }
                IrInst::ExternCall { extern_fn, .. } => {
                    let ef = &program.externs[*extern_fn as usize];
                    if ef.params.iter().any(|(_, _, r)| {
                        matches!(r.to_ascii_lowercase().as_str(), "c" | "bc")
                    }) || ef.ret.as_ref().is_some_and(|(_, r)| {
                        matches!(r.to_ascii_lowercase().as_str(), "b" | "c" | "bc")
                    }) {
                        return None;
                    }
                    if extern_clobbers_c(ef) {
                        calls += 1;
                    }
                }
                IrInst::Call { .. } => {
                    calls += 1;
                }
                IrInst::AddrLocal { slot, .. } => {
                    if let Some(f) = addr_taken.get_mut(*slot as usize) {
                        *f = true;
                    }
                }
                IrInst::LoadLocal { slot, .. } | IrInst::StoreLocal { slot, .. } => {
                    if let Some(c) = uses.get_mut(*slot as usize) {
                        *c += 1;
                    }
                }
                _ => {}
            }
        }
    }

    let mut best: Option<(SlotId, usize)> = None;
    for (s, local) in func.locals.iter().enumerate() {
        let sid = s as SlotId;
        if program.type_size(&local.ty) != 1 || addr_taken[s] {
            continue;
        }
        if s < func.param_count && !(s == 0 || (s == 1 && reg_p1)) {
            continue;
        }
        let u = uses[s];
        let profitable = if calls == 0 { u >= 2 } else { u > calls + 1 };
        if profitable && best.is_none_or(|(_, bu)| u > bu) {
            best = Some((sid, u));
        }
    }
    best.map(|(sid, _)| sid)
}

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
                IrInst::Cast { dst, to_ty, .. } => {
                    sizes.insert(*dst, program.type_size(to_ty));
                }
                IrInst::FixedInt { dst, base, .. } => {
                    sizes.insert(*dst, base.byte_size());
                }
                IrInst::LoadLocal { dst, ty, .. } | IrInst::LoadPtr { dst, ty, .. } => {
                    sizes.insert(*dst, program.type_size(ty));
                }
                IrInst::AddrLocal { dst, .. }
                | IrInst::AddrGlobal { dst, .. }
                | IrInst::FieldAddr { dst, .. }
                | IrInst::IndexAddr { dst, .. }
                | IrInst::PoolElemAddr { dst, .. } => {
                    sizes.insert(*dst, 2);
                }
                IrInst::PoolAlloc { dst, .. } | IrInst::PortIn { dst, .. } => {
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

fn compute_fn_clobbers_c(program: &IrProgram, fns: &[FnAnalysis]) -> Vec<bool> {
    let n = program.functions.len();
    let mut clobbers = vec![false; n];
    let mut callees: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, func) in program.functions.iter().enumerate() {
        if fns[i].reg_c_slot.is_some() {
            clobbers[i] = true;
        }
        let mut const_ints: BTreeSet<ValueId> = BTreeSet::new();
        for bb in &func.blocks {
            for inst in &bb.insts {
                match inst {
                    IrInst::ConstInt { dst, .. } => {
                        const_ints.insert(*dst);
                    }
                    IrInst::Call { func: cid, .. } => {
                        callees[i].push(*cid as usize);
                    }
                    IrInst::ExternCall { extern_fn, .. } => {
                        if extern_clobbers_c(&program.externs[*extern_fn as usize]) {
                            clobbers[i] = true;
                        }
                    }
                    IrInst::InlineAsm { .. }
                    | IrInst::PoolAlloc { .. }
                    | IrInst::PoolFree { .. }
                    | IrInst::PoolElemAddr { .. }
                    | IrInst::StateTransition { .. }
                    | IrInst::StateDispatch { .. } => {
                        clobbers[i] = true;
                    }
                    IrInst::PortIn { port, .. } | IrInst::PortOut { port, .. } => {
                        if !const_ints.contains(port) {
                            clobbers[i] = true;
                        }
                    }
                    IrInst::BinOp {
                        op: IrBinOp::Mul,
                        ty,
                        lhs,
                        rhs,
                        ..
                    } if program.type_size(ty) == 2
                        && !const_ints.contains(lhs)
                        && !const_ints.contains(rhs) =>
                    {
                        clobbers[i] = true;
                    }
                    _ => {}
                }
            }
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..n {
            if !clobbers[i] && callees[i].iter().any(|&c| clobbers[c]) {
                clobbers[i] = true;
                changed = true;
            }
        }
    }
    clobbers
}
