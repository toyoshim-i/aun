//! Layer 2: Target-independent MIR optimization passes (`aun-ir::opt`).
//!
//! Runs before backend allocation (*Uchiguri*) and code generation so that both
//! RAM overlay planning and Z80 instruction selection see a single clean,
//! canonical IR:
//! - CFG simplification (constant branch folding, empty `Goto` threading,
//!   unreachable block removal, single-predecessor `Goto` block merging).
//! - Local-to-SSA forwarding (`StoreLocal` -> `LoadLocal` within a basic block),
//!   dead local store elimination, and transitive copy coalescing (`tmp -> dst`).
//! - Constant folding (wrapping/bitwise/comparison `BinOp`, `UnOp::Not`, `Cast`).
//! - Pure SSA dead-code elimination (DCE).

use crate::*;
use std::collections::{HashMap, HashSet, VecDeque};

/// Target optimization configuration controlling target-biased MIR passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OptConfig {
    pub rematerialize_local_loads: bool,
    pub demote_cross_block_ssa: bool,
    pub reduce_u8_pow2_mul_div: bool,
    pub enable_osr_u8: bool,
}

impl OptConfig {
    pub const fn z80() -> Self {
        Self {
            rematerialize_local_loads: true,
            demote_cross_block_ssa: true,
            reduce_u8_pow2_mul_div: true,
            enable_osr_u8: true,
        }
    }

    pub const fn mcs51() -> Self {
        Self {
            rematerialize_local_loads: false,
            demote_cross_block_ssa: false,
            reduce_u8_pow2_mul_div: false,
            enable_osr_u8: false,
        }
    }

    pub fn for_target(target: Option<&TargetDef>) -> Self {
        match target.map(|t| t.arch.as_str()) {
            Some("mcs51" | "ch559" | "8051") => Self::mcs51(),
            _ => Self::z80(),
        }
    }
}

impl Default for OptConfig {
    fn default() -> Self {
        Self::z80()
    }
}

/// Optimizes all functions in `program` in-place until fixed point using target defaults.
pub fn optimize_program(program: &mut IrProgram) {
    let cfg = OptConfig::for_target(program.target.as_ref());
    optimize_program_with_config(program, &cfg);
}

/// Optimizes all functions in `program` in-place with an explicit `OptConfig`.
pub fn optimize_program_with_config(program: &mut IrProgram, cfg: &OptConfig) {
    for func in &mut program.functions {
        optimize_function_with_config(func, cfg);
    }
    crate::inline::inline_functions(program, cfg);
}

/// Optimizes a single `IrFunction` in-place until fixed point with Z80-default config.
pub fn optimize_function(func: &mut IrFunction) {
    optimize_function_with_config(func, &OptConfig::default());
}

/// Optimizes a single `IrFunction` in-place until fixed point with `cfg`.
pub fn optimize_function_with_config(func: &mut IrFunction, cfg: &OptConfig) {
    if func.blocks.is_empty() || func.blocks.iter().all(|b| b.insts.is_empty()) {
        return;
    }
    run_cleanup_loop(func, cfg);
    optimize_function_ssa_with_config(func, cfg);
    run_cleanup_loop(func, cfg);
}

fn run_cleanup_loop(func: &mut IrFunction, cfg: &OptConfig) {
    for _ in 0..8 {
        let mut changed = false;
        changed |= fold_constants(func, cfg);
        changed |= simplify_insts_and_branches(func);
        changed |= narrow_truncating_binops(func);
        changed |= simplify_cfg(func);
        changed |= forward_and_coalesce_locals(func);
        changed |= forward_memory_loads(func);
        changed |= dce(func);
        if !changed {
            break;
        }
    }
}

/// Demand-narrows 16-bit bitwise/wrapping add/sub expressions whose sole consumer is a
/// truncating cast to `u8`/`i8`, and collapses round-trip `u8/i8 -> u16/i16 -> u8/i8` casts.
fn narrow_truncating_binops(func: &mut IrFunction) -> bool {
    if func.blocks.is_empty() {
        return false;
    }
    let mut use_counts: HashMap<ValueId, usize> = HashMap::new();
    for bb in &func.blocks {
        for inst in &bb.insts {
            inst.for_each_use(|v| *use_counts.entry(v).or_insert(0) += 1);
        }
        match bb.term {
            Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } => {
                *use_counts.entry(v).or_insert(0) += 1;
            }
            _ => {}
        }
    }

    let mut changed = false;
    let mut next_val = func.next_val;
    for bb in &mut func.blocks {
        // Map ValueId -> Cast source within this block to collapse widening->narrowing round trips.
        let mut blk_casts: HashMap<ValueId, (IrType, ValueId)> = HashMap::new();
        let mut blk_narrowable: HashMap<ValueId, (usize, IrBinOp, IrType, ValueId, ValueId, u16)> =
            HashMap::new();

        for (ii, inst) in bb.insts.iter().enumerate() {
            match inst {
                IrInst::Cast {
                    dst,
                    val,
                    from_ty,
                    to_ty,
                } if matches!(from_ty, IrType::Int(IntType::U8 | IntType::I8))
                    && matches!(to_ty, IrType::Int(IntType::U16 | IntType::I16)) =>
                {
                    blk_casts.insert(*dst, (from_ty.clone(), *val));
                }
                IrInst::BinOp {
                    dst,
                    op: op @ (IrBinOp::BitAnd | IrBinOp::BitOr | IrBinOp::BitXor | IrBinOp::Add | IrBinOp::Sub),
                    mode,
                    ty: ty @ IrType::Int(IntType::U16 | IntType::I16),
                    lhs,
                    rhs,
                    site_id,
                } if (matches!(op, IrBinOp::BitAnd | IrBinOp::BitOr | IrBinOp::BitXor)
                    || *mode == ArithMode::Wrapping)
                    && use_counts.get(dst).copied().unwrap_or(0) == 1 =>
                {
                    blk_narrowable.insert(*dst, (ii, *op, ty.clone(), *lhs, *rhs, *site_id));
                }
                _ => {}
            }
        }

        let mut i = 0;
        while i < bb.insts.len() {
            let mut handled_roundtrip = false;
            if let IrInst::Cast {
                dst,
                val,
                from_ty: IrType::Int(IntType::U16 | IntType::I16),
                to_ty: to_ty @ IrType::Int(IntType::U8 | IntType::I8),
            } = &bb.insts[i]
                && let Some((orig_ty, orig_val)) = blk_casts.get(val).cloned()
            {
                let d = *dst;
                let t = to_ty.clone();
                bb.insts[i] = IrInst::Cast {
                    dst: d,
                    val: orig_val,
                    from_ty: orig_ty,
                    to_ty: t,
                };
                changed = true;
                handled_roundtrip = true;
            }
            if handled_roundtrip {
                i += 1;
                continue;
            }

            if let IrInst::Cast {
                dst,
                val,
                from_ty: IrType::Int(IntType::U16 | IntType::I16),
                to_ty: to_ty @ IrType::Int(IntType::U8 | IntType::I8),
            } = &bb.insts[i]
                && let Some((b_ii, op, from_ty, lhs, rhs, site_id)) = blk_narrowable.remove(val)
            {
                let d = *dst;
                let t = to_ty.clone();
                let c_lhs = next_val;
                next_val += 1;
                let c_rhs = next_val;
                next_val += 1;
                // Place narrowed BinOp in-place at `i` (where `Cast` was), with `c_lhs`/`c_rhs` right before it.
                // Mark old 16-bit BinOp as a self-identity Cast or replace directly? Replacing `b_ii` with
                // `ConstInt(0)` / noop isn't clean; remove `b_ii` if earlier than `i` and adjust `i`.
                let mut replacement = vec![
                    IrInst::Cast {
                        dst: c_lhs,
                        val: lhs,
                        from_ty: from_ty.clone(),
                        to_ty: t.clone(),
                    },
                    IrInst::Cast {
                        dst: c_rhs,
                        val: rhs,
                        from_ty,
                        to_ty: t.clone(),
                    },
                    IrInst::BinOp {
                        dst: d,
                        op,
                        mode: ArithMode::Wrapping,
                        ty: t,
                        lhs: c_lhs,
                        rhs: c_rhs,
                        site_id,
                    },
                ];
                bb.insts.splice(i..=i, replacement.drain(..));
                if b_ii < i {
                    bb.insts.remove(b_ii);
                    // Adjust remaining indices in `blk_narrowable`? Break & let outer loop re-run!
                    changed = true;
                    break;
                }
                changed = true;
                i += 3;
                continue;
            }
            i += 1;
        }
    }
    func.next_val = next_val;
    changed
}

/// Promotes promotable locals to Pruned SSA, runs SCCP + GVN + LICM + OSR + DCE to
/// fixed point, and destructs SSA form back to non-`Phi` MIR.
pub fn optimize_function_ssa(func: &mut IrFunction) {
    optimize_function_ssa_with_config(func, &OptConfig::default());
}

/// Target-configurable variant of `optimize_function_ssa`.
pub fn optimize_function_ssa_with_config(func: &mut IrFunction, cfg: &OptConfig) {
    if func.blocks.is_empty() || func.blocks.iter().all(|b| b.insts.is_empty()) {
        return;
    }
    crate::ssa::mem2reg(func);
    for _ in 0..4 {
        let mut changed = false;
        changed |= crate::sccp::sccp(func);
        changed |= crate::gvn::gvn(func);
        changed |= crate::licm::licm(func);
        changed |= crate::osr::osr_with_config(func, cfg.enable_osr_u8);
        changed |= dce(func);
        if !changed {
            break;
        }
    }
    crate::ssa::destruct_ssa_with_options(
        func,
        cfg.rematerialize_local_loads,
        cfg.demote_cross_block_ssa,
    );
}

pub(crate) fn apply_val_remap(func: &mut IrFunction, remap: &HashMap<ValueId, ValueId>) {
    if remap.is_empty() {
        return;
    }
    let resolve = |mut v: ValueId| -> ValueId {
        let mut hops = 0;
        while let Some(&next) = remap.get(&v) {
            if next == v || hops > 16 {
                break;
            }
            v = next;
            hops += 1;
        }
        v
    };
    for bb in &mut func.blocks {
        for inst in &mut bb.insts {
            inst.for_each_use_mut(|v| *v = resolve(*v));
        }
        match &mut bb.term {
            Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } => {
                *v = resolve(*v);
            }
            _ => {}
        }
    }
}

pub(crate) fn truncate_int(val: i64, it: IntType) -> i64 {
    match it {
        IntType::U8 => (val as u8) as i64,
        IntType::I8 => (val as i8) as i64,
        IntType::U16 => (val as u16) as i64,
        IntType::I16 => (val as i16) as i64,
        IntType::U32 => (val as u32) as i64,
        IntType::I32 => (val as i32) as i64,
    }
}

fn fold_constants(func: &mut IrFunction, cfg: &OptConfig) -> bool {
    let mut const_ints: HashMap<ValueId, (IntType, i64)> = HashMap::new();
    let mut const_bools: HashMap<ValueId, bool> = HashMap::new();
    for bb in &func.blocks {
        for inst in &bb.insts {
            match inst {
                IrInst::ConstInt {
                    dst,
                    ty: IrType::Int(it),
                    val,
                } => {
                    const_ints.insert(*dst, (*it, truncate_int(*val, *it)));
                }
                IrInst::ConstBool { dst, val } => {
                    const_bools.insert(*dst, *val);
                }
                _ => {}
            }
        }
    }

    let mut changed = false;
    let mut next_val = func.next_val;
    for bb in &mut func.blocks {
        let mut new_insts: Option<Vec<IrInst>> = None;
        let orig_len = bb.insts.len();
        for idx in 0..orig_len {
            let mut pre_insert: Option<IrInst> = None;
            {
                let inst = &mut bb.insts[idx];
                let replacement = match inst {
                    IrInst::UnOp {
                        dst,
                        op: IrUnOp::Not,
                        ty: IrType::Bool,
                        val,
                        ..
                    } => const_bools.get(val).map(|&b| IrInst::ConstBool {
                        dst: *dst,
                        val: !b,
                    }),
                    IrInst::UnOp {
                        dst,
                        op: IrUnOp::Not,
                        ty: IrType::Int(it),
                        val,
                        ..
                    } => const_ints.get(val).map(|&(_, v)| IrInst::ConstInt {
                        dst: *dst,
                        ty: IrType::Int(*it),
                        val: truncate_int(!v, *it),
                    }),
                    IrInst::UnOp {
                        dst,
                        op: IrUnOp::Neg,
                        mode,
                        ty: IrType::Int(it),
                        val,
                        ..
                    } => const_ints.get(val).and_then(|&(_, v)| {
                        let raw = -(v as i128);
                        let min = it.min_i64() as i128;
                        let max = it.max_i64() as i128;
                        let folded = match *mode {
                            ArithMode::Wrapping => Some(truncate_int(raw as i64, *it)),
                            ArithMode::Saturating => {
                                Some(truncate_int(raw.clamp(min, max) as i64, *it))
                            }
                            ArithMode::Checked => {
                                (raw >= min && raw <= max).then(|| truncate_int(raw as i64, *it))
                            }
                        };
                        folded.map(|fv| IrInst::ConstInt {
                            dst: *dst,
                            ty: IrType::Int(*it),
                            val: fv,
                        })
                    }),
                    IrInst::Cast {
                        dst,
                        from_ty,
                        to_ty: IrType::Int(to_it),
                        val,
                    } => {
                        if let Some(&(_, v)) = const_ints.get(val) {
                            Some(IrInst::ConstInt {
                                dst: *dst,
                                ty: IrType::Int(*to_it),
                                val: truncate_int(v, *to_it),
                            })
                        } else if *from_ty == IrType::Bool
                            && let Some(&b) = const_bools.get(val)
                        {
                            Some(IrInst::ConstInt {
                                dst: *dst,
                                ty: IrType::Int(*to_it),
                                val: if b { 1 } else { 0 },
                            })
                        } else {
                            None
                        }
                    }
                    IrInst::Cast {
                        dst,
                        to_ty: IrType::Bool,
                        val,
                        ..
                    } => {
                        if let Some(&b) = const_bools.get(val) {
                            Some(IrInst::ConstBool { dst: *dst, val: b })
                        } else {
                            const_ints.get(val).map(|&(_, v)| IrInst::ConstBool {
                                dst: *dst,
                                val: v != 0,
                            })
                        }
                    }
                    IrInst::BinOp {
                        dst,
                        op,
                        mode,
                        ty,
                        lhs,
                        rhs,
                        ..
                    } => {
                        let l_cv = const_ints
                            .get(lhs)
                            .map(|&(it, v)| crate::sccp::ConstVal::Int(it, v))
                            .or_else(|| const_bools.get(lhs).map(|&b| crate::sccp::ConstVal::Bool(b)));
                        let r_cv = const_ints
                            .get(rhs)
                            .map(|&(it, v)| crate::sccp::ConstVal::Int(it, v))
                            .or_else(|| const_bools.get(rhs).map(|&b| crate::sccp::ConstVal::Bool(b)));
                        if let (Some(lc), Some(rc)) = (l_cv, r_cv) {
                            crate::sccp::eval_binop(*op, *mode, ty, lc, rc).map(|cv| match cv {
                                crate::sccp::ConstVal::Int(it, val) => IrInst::ConstInt {
                                    dst: *dst,
                                    ty: IrType::Int(it),
                                    val,
                                },
                                crate::sccp::ConstVal::Bool(val) => {
                                    IrInst::ConstBool { dst: *dst, val }
                                }
                            })
                        } else {
                            None
                        }
                    }
                    _ => None,
                };

                if let Some(new_inst) = replacement {
                    match &new_inst {
                        IrInst::ConstInt {
                            dst,
                            ty: IrType::Int(it),
                            val,
                        } => {
                            const_ints.insert(*dst, (*it, *val));
                        }
                        IrInst::ConstBool { dst, val } => {
                            const_bools.insert(*dst, *val);
                        }
                        _ => {}
                    }
                    *inst = new_inst;
                    changed = true;
                } else if let IrInst::BinOp {
                    op,
                    mode,
                    ty: IrType::Int(it),
                    lhs,
                    rhs,
                    ..
                } = inst
                {
                    if const_ints.contains_key(lhs) && !const_ints.contains_key(rhs) {
                        match op {
                            IrBinOp::Add
                            | IrBinOp::Mul
                            | IrBinOp::BitAnd
                            | IrBinOp::BitOr
                            | IrBinOp::BitXor
                            | IrBinOp::Eq
                            | IrBinOp::Ne => {
                                std::mem::swap(lhs, rhs);
                                changed = true;
                            }
                            IrBinOp::Lt | IrBinOp::Le | IrBinOp::Gt | IrBinOp::Ge => {
                                std::mem::swap(lhs, rhs);
                                *op = match *op {
                                    IrBinOp::Lt => IrBinOp::Gt,
                                    IrBinOp::Le => IrBinOp::Ge,
                                    IrBinOp::Gt => IrBinOp::Lt,
                                    IrBinOp::Ge => IrBinOp::Le,
                                    _ => unreachable!(),
                                };
                                changed = true;
                            }
                            _ => {}
                        }
                    }
                    // Unsigned power-of-two strength reduction: x *% 2^k -> x <<% k,
                    // x / 2^k -> x >> k, x % 2^k -> x & (2^k - 1).
                    if !it.is_signed()
                        && let Some(&(_, c)) = const_ints.get(rhs)
                        && c > 0
                        && (c as u64).is_power_of_two()
                    {
                        let shift = (c as u64).trailing_zeros() as i64;
                        if shift > 0 && shift < it.bit_width() as i64 {
                            match *op {
                                IrBinOp::Mul
                                    if *mode == ArithMode::Wrapping
                                        && (cfg.reduce_u8_pow2_mul_div || *it != IntType::U8) =>
                                {
                                    let k_val = next_val;
                                    next_val += 1;
                                    const_ints.insert(k_val, (*it, shift));
                                    pre_insert = Some(IrInst::ConstInt {
                                        dst: k_val,
                                        ty: IrType::Int(*it),
                                        val: shift,
                                    });
                                    *op = IrBinOp::Shl;
                                    *rhs = k_val;
                                    changed = true;
                                }
                                IrBinOp::Div
                                    if cfg.reduce_u8_pow2_mul_div || *it != IntType::U8 =>
                                {
                                    let k_val = next_val;
                                    next_val += 1;
                                    const_ints.insert(k_val, (*it, shift));
                                    pre_insert = Some(IrInst::ConstInt {
                                        dst: k_val,
                                        ty: IrType::Int(*it),
                                        val: shift,
                                    });
                                    *op = IrBinOp::Shr;
                                    *mode = ArithMode::Wrapping;
                                    *rhs = k_val;
                                    changed = true;
                                }
                                IrBinOp::Rem => {
                                    let mask = c - 1;
                                    let k_val = next_val;
                                    next_val += 1;
                                    const_ints.insert(k_val, (*it, mask));
                                    pre_insert = Some(IrInst::ConstInt {
                                        dst: k_val,
                                        ty: IrType::Int(*it),
                                        val: mask,
                                    });
                                    *op = IrBinOp::BitAnd;
                                    *mode = ArithMode::Wrapping;
                                    *rhs = k_val;
                                    changed = true;
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }

            if let Some(pre) = pre_insert {
                let vec = new_insts.get_or_insert_with(|| bb.insts[..idx].to_vec());
                vec.push(pre);
                vec.push(bb.insts[idx].clone());
            } else if let Some(ref mut vec) = new_insts {
                vec.push(bb.insts[idx].clone());
            }
        }
        if let Some(vec) = new_insts {
            bb.insts = vec;
        }
    }
    func.next_val = next_val;
    changed
}

fn invert_cmp_op(op: IrBinOp) -> Option<IrBinOp> {
    match op {
        IrBinOp::Eq => Some(IrBinOp::Ne),
        IrBinOp::Ne => Some(IrBinOp::Eq),
        IrBinOp::Lt => Some(IrBinOp::Ge),
        IrBinOp::Ge => Some(IrBinOp::Lt),
        IrBinOp::Le => Some(IrBinOp::Gt),
        IrBinOp::Gt => Some(IrBinOp::Le),
        _ => None,
    }
}

fn simplify_insts_and_branches(func: &mut IrFunction) -> bool {
    if func.blocks.is_empty() {
        return false;
    }
    let cfg = crate::cfg::Cfg::build(func);
    let mut field_addrs: HashMap<ValueId, (ValueId, u16)> = HashMap::new();
    let mut slices: HashMap<ValueId, (ValueId, ValueId)> = HashMap::new();
    let mut not_vals: HashMap<ValueId, ValueId> = HashMap::new();
    let mut neg_wrapping_vals: HashMap<ValueId, ValueId> = HashMap::new();
    let mut cmp_defs: HashMap<ValueId, (IrBinOp, ArithMode, IrType, ValueId, ValueId, u16)> =
        HashMap::new();

    for &bi in cfg.rpo_indices() {
        for inst in &func.blocks[bi].insts {
            match inst {
                IrInst::MakeSlice { dst, ptr, len, .. } => {
                    slices.insert(*dst, (*ptr, *len));
                }
                IrInst::FieldAddr {
                    dst,
                    base,
                    byte_offset,
                } => {
                    if let Some(&(root, off1)) = field_addrs.get(base)
                        && let Some(sum) = off1.checked_add(*byte_offset)
                    {
                        field_addrs.insert(*dst, (root, sum));
                    } else {
                        field_addrs.insert(*dst, (*base, *byte_offset));
                    }
                }
                IrInst::UnOp {
                    dst,
                    op: IrUnOp::Not,
                    ty: IrType::Bool,
                    val,
                    ..
                } => {
                    not_vals.insert(*dst, *val);
                }
                IrInst::UnOp {
                    dst,
                    op: IrUnOp::Neg,
                    mode: ArithMode::Wrapping,
                    val,
                    ..
                } => {
                    neg_wrapping_vals.insert(*dst, *val);
                }
                IrInst::BinOp {
                    dst,
                    op,
                    mode,
                    ty,
                    lhs,
                    rhs,
                    site_id,
                } if invert_cmp_op(*op).is_some() => {
                    cmp_defs.insert(*dst, (*op, *mode, ty.clone(), *lhs, *rhs, *site_id));
                }
                _ => {}
            }
        }
    }

    let mut changed = false;
    let mut remap: HashMap<ValueId, ValueId> = HashMap::new();
    for bb in &mut func.blocks {
        let mut drop = vec![false; bb.insts.len()];
        for (ii, inst) in bb.insts.iter_mut().enumerate() {
            match inst {
                IrInst::FieldAddr {
                    base, byte_offset, ..
                } => {
                    if let Some(&(root, sum)) = field_addrs.get(base)
                        && let Some(total) = sum.checked_add(*byte_offset)
                    {
                        *base = root;
                        *byte_offset = total;
                        changed = true;
                    }
                }
                IrInst::UnOp {
                    dst,
                    op: IrUnOp::Not,
                    ty: IrType::Bool,
                    val,
                    ..
                } => {
                    if let Some(&orig) = not_vals.get(val) {
                        remap.insert(*dst, orig);
                        drop[ii] = true;
                        changed = true;
                    } else if let Some(&(cop, cmode, ref cty, clhs, crhs, csite)) =
                        cmp_defs.get(val)
                        && let Some(inv_op) = invert_cmp_op(cop)
                    {
                        *inst = IrInst::BinOp {
                            dst: *dst,
                            op: inv_op,
                            mode: cmode,
                            ty: cty.clone(),
                            lhs: clhs,
                            rhs: crhs,
                            site_id: csite,
                        };
                        changed = true;
                    }
                }
                IrInst::UnOp {
                    dst,
                    op: IrUnOp::Neg,
                    mode: ArithMode::Wrapping,
                    val,
                    ..
                } => {
                    if let Some(&orig) = neg_wrapping_vals.get(val) {
                        remap.insert(*dst, orig);
                        drop[ii] = true;
                        changed = true;
                    }
                }
                IrInst::SlicePtr { dst, slice, .. } => {
                    if let Some(&(ptr, _)) = slices.get(slice) {
                        remap.insert(*dst, ptr);
                        drop[ii] = true;
                        changed = true;
                    }
                }
                IrInst::SliceLen { dst, slice } => {
                    if let Some(&(_, len)) = slices.get(slice) {
                        remap.insert(*dst, len);
                        drop[ii] = true;
                        changed = true;
                    }
                }
                _ => {}
            }
        }
        if drop.iter().any(|&d| d) {
            let mut idx = 0usize;
            bb.insts.retain(|_| {
                let keep = !drop[idx];
                idx += 1;
                keep
            });
        }
        if let Terminator::Branch {
            cond,
            then_bb,
            else_bb,
        } = &mut bb.term
            && let Some(&inner) = not_vals.get(cond)
        {
            *cond = inner;
            std::mem::swap(then_bb, else_bb);
            changed = true;
        }
    }
    if !remap.is_empty() {
        apply_val_remap(func, &remap);
    }
    changed
}

fn simplify_cfg(func: &mut IrFunction) -> bool {
    if func.blocks.is_empty() {
        return false;
    }
    let mut changed = false;

    // 1. Fold constant branches & identical-target branches
    let mut const_bools: HashMap<ValueId, bool> = HashMap::new();
    for bb in &func.blocks {
        for inst in &bb.insts {
            if let IrInst::ConstBool { dst, val } = inst {
                const_bools.insert(*dst, *val);
            }
        }
    }
    for bb in &mut func.blocks {
        if let Terminator::Branch {
            cond,
            then_bb,
            else_bb,
        } = bb.term
        {
            if then_bb == else_bb {
                bb.term = Terminator::Goto(then_bb);
                changed = true;
            } else if let Some(&b) = const_bools.get(&cond) {
                bb.term = Terminator::Goto(if b { then_bb } else { else_bb });
                changed = true;
            }
        }
    }

    // 2. Thread empty `Goto` trampoline blocks (never redirect away from entry block 0)
    let entry_id = func.blocks[0].id;
    let mut empty_goto: HashMap<BlockId, BlockId> = HashMap::new();
    for bb in &func.blocks {
        if bb.id != entry_id
            && bb.insts.is_empty()
            && let Terminator::Goto(t) = bb.term
            && t != bb.id
        {
            empty_goto.insert(bb.id, t);
        }
    }
    if !empty_goto.is_empty() {
        let resolve = |mut b: BlockId| -> BlockId {
            let mut seen = HashSet::new();
            while let Some(&next) = empty_goto.get(&b) {
                if !seen.insert(b) {
                    break;
                }
                b = next;
            }
            b
        };
        for bb in &mut func.blocks {
            match &mut bb.term {
                Terminator::Goto(t) => {
                    let nt = resolve(*t);
                    if nt != *t {
                        *t = nt;
                        changed = true;
                    }
                }
                Terminator::Branch {
                    then_bb, else_bb, ..
                } => {
                    let nt = resolve(*then_bb);
                    let ne = resolve(*else_bb);
                    if nt != *then_bb {
                        *then_bb = nt;
                        changed = true;
                    }
                    if ne != *else_bb {
                        *else_bb = ne;
                        changed = true;
                    }
                }
                _ => {}
            }
        }
    }

    // 3. Remove unreachable blocks (BFS from entry)
    let id_to_idx: HashMap<BlockId, usize> = func
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id, i))
        .collect();
    let mut reachable: HashSet<BlockId> = HashSet::new();
    let mut q = VecDeque::new();
    reachable.insert(entry_id);
    q.push_back(entry_id);
    while let Some(bid) = q.pop_front() {
        let Some(&idx) = id_to_idx.get(&bid) else {
            continue;
        };
        match func.blocks[idx].term {
            Terminator::Goto(t) => {
                if reachable.insert(t) {
                    q.push_back(t);
                }
            }
            Terminator::Branch {
                then_bb, else_bb, ..
            } => {
                if reachable.insert(then_bb) {
                    q.push_back(then_bb);
                }
                if reachable.insert(else_bb) {
                    q.push_back(else_bb);
                }
            }
            _ => {}
        }
    }
    if reachable.len() < func.blocks.len() {
        func.blocks.retain(|b| reachable.contains(&b.id));
        changed = true;
    }

    // 4. Merge single-predecessor `Goto` chains (`A -> B` where `in_deg(B) == 1 && B != entry`)
    let mut in_deg: HashMap<BlockId, usize> = HashMap::new();
    for bb in &func.blocks {
        match bb.term {
            Terminator::Goto(t) => *in_deg.entry(t).or_default() += 1,
            Terminator::Branch {
                then_bb, else_bb, ..
            } => {
                *in_deg.entry(then_bb).or_default() += 1;
                *in_deg.entry(else_bb).or_default() += 1;
            }
            _ => {}
        }
    }
    let mut merged = true;
    while merged {
        merged = false;
        for i in 0..func.blocks.len() {
            if let Terminator::Goto(target) = func.blocks[i].term
                && target != entry_id
                && target != func.blocks[i].id
                && in_deg.get(&target).copied() == Some(1)
                && let Some(tidx) = func.blocks.iter().position(|b| b.id == target)
            {
                let mut target_bb = func.blocks.remove(tidx);
                let src_idx = if tidx < i { i - 1 } else { i };
                func.blocks[src_idx].insts.append(&mut target_bb.insts);
                func.blocks[src_idx].term = target_bb.term;
                in_deg.remove(&target);
                changed = true;
                merged = true;
                break;
            }
        }
    }

    changed
}

fn forward_and_coalesce_locals(func: &mut IrFunction) -> bool {
    let n_slots = func.locals.len();
    if n_slots == 0 {
        return false;
    }
    let mut changed = false;

    let mut addr_taken = vec![false; n_slots];
    for bb in &func.blocks {
        for inst in &bb.insts {
            if let IrInst::AddrLocal { slot, .. } = inst
                && let Some(f) = addr_taken.get_mut(*slot as usize)
            {
                *f = true;
            }
        }
    }

    // 1. Intra-BB `StoreLocal(s, v)` -> `LoadLocal(dst, s)` forwarding + same-BB dead overwrite
    let mut val_remap: HashMap<ValueId, ValueId> = HashMap::new();
    for bb in &mut func.blocks {
        let mut defined_in_bb: HashSet<ValueId> = HashSet::new();
        let mut last_val: HashMap<SlotId, ValueId> = HashMap::new();
        let mut last_store_idx: HashMap<SlotId, usize> = HashMap::new();
        let mut drop = vec![false; bb.insts.len()];

        for (ii, inst) in bb.insts.iter().enumerate() {
            if let Some(d) = inst.dst() {
                defined_in_bb.insert(d);
            }
            match inst {
                IrInst::StoreLocal { slot, val, .. } => {
                    let s = *slot as usize;
                    if s < n_slots && !addr_taken[s] {
                        if last_val.get(slot) == Some(val) {
                            drop[ii] = true;
                            changed = true;
                            continue;
                        }
                        if let Some(prev_si) = last_store_idx.insert(*slot, ii) {
                            drop[prev_si] = true;
                            changed = true;
                        }
                        if defined_in_bb.contains(val) {
                            last_val.insert(*slot, *val);
                        } else {
                            last_val.remove(slot);
                        }
                    }
                }
                IrInst::LoadLocal { dst, slot, .. } => {
                    let s = *slot as usize;
                    if s < n_slots && !addr_taken[s] {
                        if let Some(&pv) = last_val.get(slot) {
                            val_remap.insert(*dst, pv);
                            drop[ii] = true;
                            changed = true;
                        } else {
                            // `slot` was read before any subsequent overwrite
                            last_store_idx.remove(slot);
                            last_val.insert(*slot, *dst);
                        }
                    }
                }
                _ => {}
            }
        }
        if drop.iter().any(|&d| d) {
            let mut keep_idx = 0usize;
            bb.insts.retain(|_| {
                let keep = !drop[keep_idx];
                keep_idx += 1;
                keep
            });
        }
    }
    if !val_remap.is_empty() {
        apply_val_remap(func, &val_remap);
    }

    // 2. Copy-coalesce single-read non-escaping temporaries (`tmp -> dst`) directly in MIR
    for _ in 0..n_slots {
        let mut read_locs: HashMap<SlotId, Vec<(usize, usize)>> = HashMap::new();
        let mut write_locs: HashMap<SlotId, Vec<(usize, usize)>> = HashMap::new();
        let mut val_uses: HashMap<ValueId, usize> = HashMap::new();
        for (bi, bb) in func.blocks.iter().enumerate() {
            for (ii, inst) in bb.insts.iter().enumerate() {
                inst.for_each_use(|v| *val_uses.entry(v).or_default() += 1);
                match inst {
                    IrInst::LoadLocal { slot, .. } => {
                        read_locs.entry(*slot).or_default().push((bi, ii));
                    }
                    IrInst::StoreLocal { slot, .. } => {
                        write_locs.entry(*slot).or_default().push((bi, ii));
                    }
                    _ => {}
                }
            }
            if let Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } = bb.term {
                *val_uses.entry(v).or_default() += 1;
            }
        }

        let mut coalesce_pair: Option<(SlotId, SlotId, usize, usize)> = None;
        for s in func.param_count..n_slots {
            let tmp = s as SlotId;
            if addr_taken[s] {
                continue;
            }
            let Some(rlocs) = read_locs.get(&tmp) else {
                continue;
            };
            if rlocs.len() != 1 {
                continue;
            }
            let (rb, ri) = rlocs[0];
            let bb = &func.blocks[rb];
            let IrInst::LoadLocal { dst: v, .. } = bb.insts[ri] else {
                continue;
            };
            let Some(IrInst::StoreLocal {
                slot: dst_slot,
                val: v2,
                ..
            }) = bb.insts.get(ri + 1)
            else {
                continue;
            };
            if *v2 != v
                || *dst_slot == tmp
                || (*dst_slot as usize) < func.param_count
                || addr_taken.get(*dst_slot as usize).copied().unwrap_or(true)
                || val_uses.get(&v).copied() != Some(1)
            {
                continue;
            }
            let dst_only_written_here = write_locs
                .get(dst_slot)
                .is_some_and(|wl| wl.len() == 1 && wl[0] == (rb, ri + 1));
            let dst_read_before = read_locs.get(dst_slot).is_some_and(|dlocs| {
                dlocs
                    .iter()
                    .any(|&(db, di)| db < rb || (db == rb && di <= ri))
            });
            // Soundness: ensure NO `StoreLocal(tmp, ...)` can execute while `dst_slot` is live!
            // Compute blocks from which a `LoadLocal(dst_slot)` is reachable without first passing
            // through `rb` (the unique write of `dst_slot`), and also check `rb` after `ri + 1`.
            let tmp_clobbers_live_dst = if dst_only_written_here && !dst_read_before {
                let id_to_idx: HashMap<BlockId, usize> = func
                    .blocks
                    .iter()
                    .enumerate()
                    .map(|(i, b)| (b.id, i))
                    .collect();
                let mut preds: Vec<Vec<usize>> = vec![Vec::new(); func.blocks.len()];
                for (bi, b) in func.blocks.iter().enumerate() {
                    match b.term {
                        Terminator::Goto(t) => {
                            if let Some(&ti) = id_to_idx.get(&t) {
                                preds[ti].push(bi);
                            }
                        }
                        Terminator::Branch {
                            then_bb, else_bb, ..
                        } => {
                            if let Some(&ti) = id_to_idx.get(&then_bb) {
                                preds[ti].push(bi);
                            }
                            if let Some(&ei) = id_to_idx.get(&else_bb) {
                                preds[ei].push(bi);
                            }
                        }
                        _ => {}
                    }
                }
                let mut dst_live_in = vec![false; func.blocks.len()];
                let mut dst_live_out = vec![false; func.blocks.len()];
                let mut q: VecDeque<usize> = VecDeque::new();
                if let Some(dlocs) = read_locs.get(dst_slot) {
                    for &(db, _) in dlocs {
                        if db != rb && !dst_live_in[db] {
                            dst_live_in[db] = true;
                            q.push_back(db);
                        }
                    }
                }
                while let Some(b) = q.pop_front() {
                    for &p in &preds[b] {
                        dst_live_out[p] = true;
                        if p != rb && !dst_live_in[p] {
                            dst_live_in[p] = true;
                            q.push_back(p);
                        }
                    }
                }
                write_locs.get(&tmp).is_some_and(|wlocs| {
                    wlocs.iter().any(|&(wb, wi)| {
                        if wb == rb {
                            wi > ri + 1
                                && (dst_live_out[rb]
                                    || read_locs
                                        .get(dst_slot)
                                        .is_some_and(|dl| dl.iter().any(|&(db, di)| db == rb && di > wi)))
                        } else {
                            dst_live_out[wb]
                                || read_locs
                                    .get(dst_slot)
                                    .is_some_and(|dl| dl.iter().any(|&(db, di)| db == wb && di > wi))
                        }
                    })
                })
            } else {
                true
            };
            if dst_only_written_here && !dst_read_before && !tmp_clobbers_live_dst {
                coalesce_pair = Some((tmp, *dst_slot, rb, ri));
                break;
            }
        }
        let Some((tmp, dst_slot, rb, ri)) = coalesce_pair else {
            break;
        };
        for bb in &mut func.blocks {
            for inst in &mut bb.insts {
                if let IrInst::StoreLocal { slot, .. } = inst
                    && *slot == tmp
                {
                    *slot = dst_slot;
                }
            }
        }
        func.blocks[rb].insts.remove(ri + 1);
        func.blocks[rb].insts.remove(ri);
        changed = true;
    }

    // 2b. Tail-return forwarding: if non-entry `bb` is `[v = LoadLocal(tmp); Return(Some(v))]`
    //     and `tmp` is only read there, forward predecessor `StoreLocal(tmp, val); Goto(bb)`
    //     directly into `Return(Some(val))`. Unreachable `bb` is later removed by `simplify_cfg`.
    let mut read_counts = vec![0usize; n_slots];
    for bb in &func.blocks {
        for inst in &bb.insts {
            if let IrInst::LoadLocal { slot, .. } = inst
                && let Some(c) = read_counts.get_mut(*slot as usize)
            {
                *c += 1;
            }
        }
    }
    let mut tail_candidates: Vec<(BlockId, SlotId)> = Vec::new();
    for bb in func.blocks.iter().skip(1) {
        if bb.insts.len() != 1 {
            continue;
        }
        let IrInst::LoadLocal { dst: v, slot: tmp, .. } = bb.insts[0] else {
            continue;
        };
        let ts = tmp as usize;
        if ts >= func.param_count
            && ts < n_slots
            && !addr_taken[ts]
            && read_counts[ts] == 1
            && matches!(bb.term, Terminator::Return(Some(rv)) if rv == v)
        {
            tail_candidates.push((bb.id, tmp));
        }
    }
    for (target_bb, tmp) in tail_candidates {
        for pbb in &mut func.blocks {
            if matches!(pbb.term, Terminator::Goto(t) if t == target_bb)
                && matches!(pbb.insts.last(), Some(IrInst::StoreLocal { slot, .. }) if *slot == tmp)
                && let Some(IrInst::StoreLocal { val, .. }) = pbb.insts.pop()
            {
                pbb.term = Terminator::Return(Some(val));
                changed = true;
            }
        }
    }

    // 3. Exact CFG Global Local DSE: backward liveness over non-escaping locals across blocks.
    let cfg = crate::cfg::Cfg::build(func);
    let mut id_to_idx: HashMap<BlockId, usize> = HashMap::with_capacity(func.blocks.len());
    for (bi, bb) in func.blocks.iter().enumerate() {
        id_to_idx.insert(bb.id, bi);
    }
    let mut use_sets: Vec<HashSet<usize>> = vec![HashSet::new(); func.blocks.len()];
    let mut def_sets: Vec<HashSet<usize>> = vec![HashSet::new(); func.blocks.len()];
    for &bi in cfg.rpo_indices() {
        for inst in &func.blocks[bi].insts {
            match inst {
                IrInst::LoadLocal { slot, .. } => {
                    let s = *slot as usize;
                    if s < n_slots && !addr_taken[s] && !def_sets[bi].contains(&s) {
                        use_sets[bi].insert(s);
                    }
                }
                IrInst::StoreLocal { slot, .. } => {
                    let s = *slot as usize;
                    if s < n_slots && !addr_taken[s] {
                        def_sets[bi].insert(s);
                    }
                }
                _ => {}
            }
        }
    }
    let mut live_in: Vec<HashSet<usize>> = vec![HashSet::new(); func.blocks.len()];
    let mut live_out: Vec<HashSet<usize>> = vec![HashSet::new(); func.blocks.len()];
    loop {
        let mut df_changed = false;
        for &bi in cfg.rpo_indices().iter().rev() {
            let mut out = HashSet::new();
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
                df_changed = true;
            }
            if in_set != live_in[bi] {
                live_in[bi] = in_set;
                df_changed = true;
            }
        }
        if !df_changed {
            break;
        }
    }
    for &bi in cfg.rpo_indices() {
        let bb = &mut func.blocks[bi];
        let mut live = live_out[bi].clone();
        let mut drop = vec![false; bb.insts.len()];
        for (ii, inst) in bb.insts.iter().enumerate().rev() {
            match inst {
                IrInst::StoreLocal { slot, .. } => {
                    let s = *slot as usize;
                    if s < n_slots && !addr_taken[s] {
                        if !live.contains(&s) {
                            drop[ii] = true;
                            changed = true;
                        } else {
                            live.remove(&s);
                        }
                    }
                }
                IrInst::LoadLocal { slot, .. } => {
                    let s = *slot as usize;
                    if s < n_slots && !addr_taken[s] {
                        live.insert(s);
                    }
                }
                _ => {}
            }
        }
        if drop.iter().any(|&d| d) {
            let mut idx = 0usize;
            bb.insts.retain(|_| {
                let keep = !drop[idx];
                idx += 1;
                keep
            });
        }
    }

    changed
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MemBase {
    Global(u32),
    Local(u32),
}

fn scalar_mem_size(ty: &IrType) -> u16 {
    match ty {
        IrType::Bool | IrType::Handle(_) => 1,
        IrType::Int(it) | IrType::Fixed { base: it, .. } => it.byte_size(),
        IrType::Ref { .. } => 2,
        _ => 0,
    }
}

fn mem_ranges_overlap(o0: u16, s0: u16, o1: u16, s1: u16) -> bool {
    (o0 as u32) < (o1 as u32 + s1 as u32) && (o1 as u32) < (o0 as u32 + s0 as u32)
}

/// Scalar store-to-load forwarding and redundant load elimination on `AddrGlobal`/`AddrLocal`
/// derived pointers. Never caches raw `ConstInt` pointers, preserving MMIO semantics.
fn forward_memory_loads(func: &mut IrFunction) -> bool {
    if func.blocks.is_empty() {
        return false;
    }
    let cfg = crate::cfg::Cfg::build(func);
    let rpo_indices = cfg.rpo_indices();
    if rpo_indices.is_empty() {
        return false;
    }
    let escaped_locals = func.escaped_locals();

    let mut const_vals: HashMap<ValueId, i64> = HashMap::new();
    let mut is_const_val: HashSet<ValueId> = HashSet::new();
    let mut ptr_map: HashMap<ValueId, (MemBase, u16)> = HashMap::new();

    for &bi in rpo_indices {
        for inst in &func.blocks[bi].insts {
            match inst {
                IrInst::ConstInt { dst, val, .. } => {
                    const_vals.insert(*dst, *val);
                    is_const_val.insert(*dst);
                }
                IrInst::ConstBool { dst, .. } => {
                    is_const_val.insert(*dst);
                }
                IrInst::AddrGlobal { dst, global } => {
                    ptr_map.insert(*dst, (MemBase::Global(*global), 0));
                }
                IrInst::AddrLocal { dst, slot } => {
                    ptr_map.insert(*dst, (MemBase::Local(*slot), 0));
                }
                IrInst::FieldAddr {
                    dst,
                    base,
                    byte_offset,
                } => {
                    if let Some(&(b, off)) = ptr_map.get(base)
                        && let Some(noff) = off.checked_add(*byte_offset)
                    {
                        ptr_map.insert(*dst, (b, noff));
                    }
                }
                IrInst::IndexAddr {
                    dst,
                    base,
                    index,
                    elem_size,
                    len,
                    ..
                } => {
                    if let Some(&(b, off)) = ptr_map.get(base)
                        && let Some(&c_idx) = const_vals.get(index)
                        && (*len == 0 || (c_idx >= 0 && (c_idx as u16) < *len))
                        && let Some(ioff) =
                            u16::try_from(c_idx).ok().and_then(|i| i.checked_mul(*elem_size))
                        && let Some(noff) = off.checked_add(ioff)
                    {
                        ptr_map.insert(*dst, (b, noff));
                    }
                }
                _ => {}
            }
        }
    }

    if ptr_map.is_empty() {
        return false;
    }

    type MemState = HashMap<(MemBase, u16), (IrType, ValueId, u16)>;
    let resolve = |mut v: ValueId, remap: &HashMap<ValueId, ValueId>| -> ValueId {
        let mut hops = 0;
        while let Some(&next) = remap.get(&v) {
            if next == v || hops > 16 {
                break;
            }
            v = next;
            hops += 1;
        }
        v
    };

    let mut out_states: HashMap<BlockId, MemState> = HashMap::new();
    let mut remap: HashMap<ValueId, ValueId> = HashMap::new();
    let mut drop_blocks: HashMap<usize, Vec<bool>> = HashMap::new();
    let mut changed = false;

    for &bi in rpo_indices {
        let bb = &func.blocks[bi];
        let preds = cfg.reachable_preds(bb.id);
        let mut state: MemState = if preds.len() == 1
            && let Some(p_state) = out_states.get(&preds[0])
        {
            p_state
                .iter()
                .filter(|&(_, (_, v, _))| is_const_val.contains(v))
                .map(|(&k, v)| (k, v.clone()))
                .collect()
        } else {
            HashMap::new()
        };

        let mut drops = vec![false; bb.insts.len()];
        for (ii, inst) in bb.insts.iter().enumerate() {
            match inst {
                IrInst::LoadPtr {
                    dst,
                    addr,
                    ty,
                    volatile,
                } => {
                    let sz = scalar_mem_size(ty);
                    if sz > 0
                        && let Some(&(base, off)) = ptr_map.get(addr)
                    {
                        if *volatile {
                            state.retain(|&(b, o), &mut (_, _, esz)| {
                                b != base || !mem_ranges_overlap(o, esz, off, sz)
                            });
                        } else if let Some((cty, cval, _)) = state.get(&(base, off))
                            && cty == ty
                        {
                            remap.insert(*dst, *cval);
                            drops[ii] = true;
                            changed = true;
                        } else {
                            state.insert((base, off), (ty.clone(), *dst, sz));
                        }
                    }
                }
                IrInst::StorePtr {
                    addr,
                    val,
                    ty,
                    volatile,
                } => {
                    let sz = scalar_mem_size(ty);
                    if sz > 0
                        && let Some(&(base, off)) = ptr_map.get(addr)
                    {
                        state.retain(|&(b, o), &mut (_, _, esz)| {
                            if b == base {
                                !mem_ranges_overlap(o, esz, off, sz)
                            } else {
                                !matches!((base, b), (MemBase::Global(_), MemBase::Global(_)))
                            }
                        });
                        if !*volatile {
                            let resolved_val = resolve(*val, &remap);
                            state.insert((base, off), (ty.clone(), resolved_val, sz));
                        }
                    } else {
                        state.retain(|&(b, _), _| {
                            matches!(
                                b,
                                MemBase::Local(s)
                                    if !escaped_locals.get(s as usize).copied().unwrap_or(true)
                            )
                        });
                    }
                }
                IrInst::Call { .. }
                | IrInst::CallIndirect { .. }
                | IrInst::ExternCall { .. }
                | IrInst::MemSet { .. }
                | IrInst::MemCopy { .. }
                | IrInst::MemFill { .. }
                | IrInst::StateDispatch { .. }
                | IrInst::StateTransition { .. }
                | IrInst::PoolAlloc { .. }
                | IrInst::PoolFree { .. } => {
                    state.retain(|&(b, _), _| {
                        matches!(
                            b,
                            MemBase::Local(s)
                                if !escaped_locals.get(s as usize).copied().unwrap_or(true)
                        )
                    });
                }
                IrInst::InlineAsm { .. } => {
                    state.clear();
                }
                _ => {}
            }
        }
        drop_blocks.insert(bi, drops);
        out_states.insert(bb.id, state);
    }

    if changed {
        for (bi, drops) in drop_blocks {
            if drops.iter().any(|&d| d) {
                let mut idx = 0usize;
                func.blocks[bi].insts.retain(|_| {
                    let keep = !drops[idx];
                    idx += 1;
                    keep
                });
            }
        }
        apply_val_remap(func, &remap);
    }

    changed
}

fn is_pure_inst(inst: &IrInst) -> bool {
    match inst {
        IrInst::ConstInt { .. }
        | IrInst::ConstBool { .. }
        | IrInst::LoadLocal { .. }
        | IrInst::AddrLocal { .. }
        | IrInst::AddrGlobal { .. }
        | IrInst::AddrFn { .. }
        | IrInst::FieldAddr { .. }
        | IrInst::LoadPtr {
            volatile: false, ..
        }
        | IrInst::Cast { .. }
        | IrInst::FixedInt { .. }
        | IrInst::MakeSlice { .. }
        | IrInst::SlicePtr { .. }
        | IrInst::SliceLen { .. }
        | IrInst::StateTag { .. }
        | IrInst::Phi { .. } => true,
        IrInst::UnOp { op, mode, .. } => *op == IrUnOp::Not || *mode == ArithMode::Wrapping,
        IrInst::BinOp { op, mode, .. } => {
            (*mode == ArithMode::Wrapping
                && matches!(
                    op,
                    IrBinOp::Add | IrBinOp::Sub | IrBinOp::Mul | IrBinOp::Shl | IrBinOp::Shr
                ))
                || matches!(
                    op,
                    IrBinOp::Eq
                        | IrBinOp::Ne
                        | IrBinOp::Lt
                        | IrBinOp::Le
                        | IrBinOp::Gt
                        | IrBinOp::Ge
                        | IrBinOp::BitAnd
                        | IrBinOp::BitOr
                        | IrBinOp::BitXor
                )
        }
        _ => false,
    }
}

pub fn dce(func: &mut IrFunction) -> bool {
    let mut uses: HashMap<ValueId, usize> = HashMap::new();
    for bb in &func.blocks {
        for inst in &bb.insts {
            inst.for_each_use(|v| *uses.entry(v).or_default() += 1);
        }
        if let Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } = bb.term {
            *uses.entry(v).or_default() += 1;
        }
    }

    let mut changed = false;
    for bb in &mut func.blocks {
        let before = bb.insts.len();
        bb.insts.retain(|inst| {
            if is_pure_inst(inst)
                && let Some(d) = inst.dst()
                && uses.get(&d).copied().unwrap_or(0) == 0
            {
                return false;
            }
            true
        });
        if bb.insts.len() < before {
            changed = true;
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mir_opt_folds_constants_forwards_locals_and_simplifies_cfg() {
        let mut prog = IrProgram::default();
        prog.functions.push(IrFunction {
            id: 0,
            name: "f".into(),
            param_count: 0,
            is_pub: true,
            locals: vec![LocalSlot {
                id: 0,
                name: "x".into(),
                ty: IrType::Int(IntType::U8),
                mutable: true,
            }],
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
            blocks: vec![
                BasicBlock {
                    id: 0,
                    insts: vec![
                        IrInst::ConstInt {
                            dst: 0,
                            ty: IrType::Int(IntType::U8),
                            val: 3,
                        },
                        IrInst::ConstInt {
                            dst: 1,
                            ty: IrType::Int(IntType::U8),
                            val: 4,
                        },
                        IrInst::BinOp {
                            dst: 2,
                            op: IrBinOp::Add,
                            mode: ArithMode::Wrapping,
                            ty: IrType::Int(IntType::U8),
                            lhs: 0,
                            rhs: 1,
                            site_id: 0,
                        },
                        IrInst::StoreLocal {
                            slot: 0,
                            val: 2,
                            ty: IrType::Int(IntType::U8),
                        },
                    ],
                    term: Terminator::Goto(1),
                },
                BasicBlock {
                    id: 1,
                    insts: vec![],
                    term: Terminator::Goto(2),
                },
                BasicBlock {
                    id: 2,
                    insts: vec![IrInst::LoadLocal {
                        dst: 3,
                        slot: 0,
                        ty: IrType::Int(IntType::U8),
                    }],
                    term: Terminator::Return(Some(3)),
                },
            ],
            next_val: 4,
        });

        optimize_program(&mut prog);
        let f = &prog.functions[0];
        assert_eq!(f.blocks.len(), 1, "CFG should merge into 1 block");
        assert_eq!(f.blocks[0].insts.len(), 1, "only folded ConstInt(7) remains");
        assert!(matches!(
            f.blocks[0].insts[0],
            IrInst::ConstInt { val: 7, .. }
        ));
    }

    #[test]
    fn mir_opt_forwards_tail_returns_and_drops_dead_stores_before_return() {
        let mut prog = IrProgram::default();
        prog.functions.push(IrFunction {
            id: 0,
            name: "g".into(),
            param_count: 1,
            is_pub: true,
            locals: vec![
                LocalSlot {
                    id: 0,
                    name: "cond".into(),
                    ty: IrType::Bool,
                    mutable: false,
                },
                LocalSlot {
                    id: 1,
                    name: "ret_tmp".into(),
                    ty: IrType::Int(IntType::U8),
                    mutable: true,
                },
                LocalSlot {
                    id: 2,
                    name: "dead_local".into(),
                    ty: IrType::Int(IntType::U8),
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
            blocks: vec![
                BasicBlock {
                    id: 0,
                    insts: vec![IrInst::LoadLocal {
                        dst: 0,
                        slot: 0,
                        ty: IrType::Bool,
                    }],
                    term: Terminator::Branch {
                        cond: 0,
                        then_bb: 1,
                        else_bb: 2,
                    },
                },
                BasicBlock {
                    id: 1,
                    insts: vec![
                        IrInst::ConstInt {
                            dst: 1,
                            ty: IrType::Int(IntType::U8),
                            val: 10,
                        },
                        IrInst::StoreLocal {
                            slot: 2,
                            val: 1,
                            ty: IrType::Int(IntType::U8),
                        },
                        IrInst::StoreLocal {
                            slot: 1,
                            val: 1,
                            ty: IrType::Int(IntType::U8),
                        },
                    ],
                    term: Terminator::Goto(3),
                },
                BasicBlock {
                    id: 2,
                    insts: vec![
                        IrInst::ConstInt {
                            dst: 2,
                            ty: IrType::Int(IntType::U8),
                            val: 20,
                        },
                        IrInst::StoreLocal {
                            slot: 1,
                            val: 2,
                            ty: IrType::Int(IntType::U8),
                        },
                    ],
                    term: Terminator::Goto(3),
                },
                BasicBlock {
                    id: 3,
                    insts: vec![IrInst::LoadLocal {
                        dst: 3,
                        slot: 1,
                        ty: IrType::Int(IntType::U8),
                    }],
                    term: Terminator::Return(Some(3)),
                },
            ],
            next_val: 4,
        });

        optimize_program(&mut prog);
        let f = &prog.functions[0];
        assert_eq!(f.blocks.len(), 3, "unreachable tail return block should be removed");
        assert!(f.blocks.iter().all(|b| b.id != 3));
        for bi in [1usize, 2usize] {
            assert!(matches!(f.blocks[bi].term, Terminator::Return(Some(_))));
            assert!(
                f.blocks[bi]
                    .insts
                    .iter()
                    .all(|i| !matches!(i, IrInst::StoreLocal { .. })),
                "all dead StoreLocals should be removed"
            );
        }
    }

    #[test]
    fn canonicalizes_const_lhs_binops_and_preserves_non_commutative() {
        let u8t = IrType::Int(IntType::U8);
        let mut prog = IrProgram::default();
        prog.functions.push(IrFunction {
            id: 0,
            name: "canon".into(),
            is_pub: false,
            param_count: 1,
            locals: vec![LocalSlot {
                id: 0,
                name: "x".into(),
                ty: u8t.clone(),
                mutable: false,
            }],
            ret_ty: IrType::Bool,
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            isr_using: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            owner_state: None,
            blocks: vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::LoadLocal {
                        dst: 0,
                        slot: 0,
                        ty: u8t.clone(),
                    },
                    IrInst::ConstInt {
                        dst: 1,
                        ty: u8t.clone(),
                        val: 0x30,
                    },
                    IrInst::ConstInt {
                        dst: 2,
                        ty: u8t.clone(),
                        val: 5,
                    },
                    IrInst::BinOp {
                        dst: 3,
                        op: IrBinOp::Add,
                        mode: ArithMode::Wrapping,
                        ty: u8t.clone(),
                        lhs: 1,
                        rhs: 0,
                        site_id: 0,
                    },
                    IrInst::BinOp {
                        dst: 4,
                        op: IrBinOp::Sub,
                        mode: ArithMode::Wrapping,
                        ty: u8t.clone(),
                        lhs: 1,
                        rhs: 3,
                        site_id: 0,
                    },
                    IrInst::BinOp {
                        dst: 5,
                        op: IrBinOp::Lt,
                        mode: ArithMode::Wrapping,
                        ty: u8t,
                        lhs: 2,
                        rhs: 4,
                        site_id: 0,
                    },
                ],
                term: Terminator::Return(Some(5)),
            }],
            next_val: 6,
        });

        optimize_program(&mut prog);
        let insts = &prog.functions[0].blocks[0].insts;
        // 0x30 + x -> x + 0x30
        assert!(matches!(
            insts[3],
            IrInst::BinOp {
                op: IrBinOp::Add,
                lhs: 0,
                rhs: 1,
                ..
            }
        ));
        // Non-commutative `0x30 - ...` must NOT be swapped
        assert!(matches!(
            insts[4],
            IrInst::BinOp {
                op: IrBinOp::Sub,
                lhs: 1,
                rhs: 3,
                ..
            }
        ));
        // `5 < y` -> `y > 5`
        assert!(matches!(
            insts[5],
            IrInst::BinOp {
                op: IrBinOp::Gt,
                lhs: 4,
                rhs: 2,
                ..
            }
        ));
    }

    #[test]
    fn strength_reduces_unsigned_pow2_mul_div_rem() {
        let u16t = IrType::Int(IntType::U16);
        let mut prog = IrProgram::default();
        prog.functions.push(IrFunction {
            id: 0,
            name: "sr".into(),
            is_pub: false,
            param_count: 1,
            locals: vec![LocalSlot {
                id: 0,
                name: "x".into(),
                ty: u16t.clone(),
                mutable: false,
            }],
            ret_ty: u16t.clone(),
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            isr_using: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            owner_state: None,
            blocks: vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::LoadLocal {
                        dst: 0,
                        slot: 0,
                        ty: u16t.clone(),
                    },
                    IrInst::ConstInt {
                        dst: 1,
                        ty: u16t.clone(),
                        val: 8,
                    },
                    IrInst::ConstInt {
                        dst: 2,
                        ty: u16t.clone(),
                        val: 4,
                    },
                    IrInst::ConstInt {
                        dst: 3,
                        ty: u16t.clone(),
                        val: 16,
                    },
                    IrInst::BinOp {
                        dst: 4,
                        op: IrBinOp::Mul,
                        mode: ArithMode::Wrapping,
                        ty: u16t.clone(),
                        lhs: 0,
                        rhs: 1,
                        site_id: 0,
                    },
                    IrInst::BinOp {
                        dst: 5,
                        op: IrBinOp::Div,
                        mode: ArithMode::Checked,
                        ty: u16t.clone(),
                        lhs: 4,
                        rhs: 2,
                        site_id: 0,
                    },
                    IrInst::BinOp {
                        dst: 6,
                        op: IrBinOp::Rem,
                        mode: ArithMode::Checked,
                        ty: u16t,
                        lhs: 5,
                        rhs: 3,
                        site_id: 0,
                    },
                ],
                term: Terminator::Return(Some(6)),
            }],
            next_val: 7,
        });

        optimize_program(&mut prog);
        let insts = &prog.functions[0].blocks[0].insts;
        assert!(insts.iter().any(|i| matches!(i, IrInst::BinOp { op: IrBinOp::Shl, .. })));
        assert!(insts.iter().any(|i| matches!(i, IrInst::BinOp { op: IrBinOp::Shr, .. })));
        assert!(insts.iter().any(|i| matches!(i, IrInst::BinOp { op: IrBinOp::BitAnd, .. })));
        assert!(!insts.iter().any(|i| matches!(
            i,
            IrInst::BinOp {
                op: IrBinOp::Mul | IrBinOp::Div | IrBinOp::Rem,
                ..
            }
        )));
    }

    #[test]
    fn inverts_not_cmp_and_branch_not_and_global_local_dse() {
        let u8t = IrType::Int(IntType::U8);
        let mut prog = IrProgram::default();
        prog.functions.push(IrFunction {
            id: 0,
            name: "inv_and_dse".into(),
            is_pub: false,
            param_count: 2,
            locals: vec![
                LocalSlot {
                    id: 0,
                    name: "a".into(),
                    ty: u8t.clone(),
                    mutable: false,
                },
                LocalSlot {
                    id: 1,
                    name: "b".into(),
                    ty: u8t.clone(),
                    mutable: false,
                },
                LocalSlot {
                    id: 2,
                    name: "dead_cross".into(),
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
            blocks: vec![
                BasicBlock {
                    id: 0,
                    insts: vec![
                        IrInst::LoadLocal {
                            dst: 0,
                            slot: 0,
                            ty: u8t.clone(),
                        },
                        IrInst::LoadLocal {
                            dst: 1,
                            slot: 1,
                            ty: u8t.clone(),
                        },
                        IrInst::ConstInt {
                            dst: 2,
                            ty: u8t.clone(),
                            val: 99,
                        },
                        // Written here, overwritten in bb1 and never read in bb2 -> dead!
                        IrInst::StoreLocal {
                            slot: 2,
                            val: 2,
                            ty: u8t.clone(),
                        },
                        IrInst::BinOp {
                            dst: 3,
                            op: IrBinOp::Lt,
                            mode: ArithMode::Checked,
                            ty: u8t.clone(),
                            lhs: 0,
                            rhs: 1,
                            site_id: 0,
                        },
                        IrInst::UnOp {
                            dst: 4,
                            op: IrUnOp::Not,
                            mode: ArithMode::Wrapping,
                            ty: IrType::Bool,
                            val: 3,
                            site_id: 0,
                        },
                        IrInst::UnOp {
                            dst: 5,
                            op: IrUnOp::Not,
                            mode: ArithMode::Wrapping,
                            ty: IrType::Bool,
                            val: 4,
                            site_id: 0,
                        },
                    ],
                    term: Terminator::Branch {
                        cond: 5,
                        then_bb: 1,
                        else_bb: 2,
                    },
                },
                BasicBlock {
                    id: 1,
                    insts: vec![
                        IrInst::StoreLocal {
                            slot: 2,
                            val: 0,
                            ty: u8t.clone(),
                        },
                        IrInst::LoadLocal {
                            dst: 6,
                            slot: 2,
                            ty: u8t,
                        },
                    ],
                    term: Terminator::Return(Some(6)),
                },
                BasicBlock {
                    id: 2,
                    insts: vec![],
                    term: Terminator::Return(Some(1)),
                },
            ],
            next_val: 7,
        });

        optimize_program(&mut prog);
        let f = &prog.functions[0];
        // No UnOp::Not should survive, and no StoreLocal to dead_cross should survive.
        for bb in &f.blocks {
            assert!(!bb.insts.iter().any(|i| matches!(i, IrInst::UnOp { op: IrUnOp::Not, .. })));
            assert!(!bb.insts.iter().any(|i| matches!(i, IrInst::StoreLocal { slot: 2, .. })));
        }
    }

    #[test]
    fn forwards_global_memory_loads_and_preserves_const_mmio() {
        let u8t = IrType::Int(IntType::U8);
        let mut prog = IrProgram::default();
        prog.globals.push(IrGlobal {
            name: "G".into(),
            ty: u8t.clone(),
            mutable: true,
            in_rom: false,
            volatile: false,
            fixed_addr: None,
            is_extern: false,
            align: 1,
            soa: false,
            space: None,
            owner_state: None,
            init: vec![],
        });
        prog.functions.push(IrFunction {
            id: 0,
            name: "mem_fwd".into(),
            is_pub: true,
            param_count: 0,
            locals: vec![],
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
            blocks: vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::AddrGlobal { dst: 0, global: 0 },
                    IrInst::ConstInt {
                        dst: 1,
                        ty: u8t.clone(),
                        val: 42,
                    },
                    IrInst::StorePtr {
                        addr: 0,
                        val: 1,
                        ty: u8t.clone(),
                        volatile: false,
                    },
                    IrInst::LoadPtr {
                        dst: 2,
                        addr: 0,
                        ty: u8t.clone(),
                        volatile: false,
                    },
                    // Raw ConstInt MMIO read MUST NOT be forwarded or eliminated!
                    IrInst::ConstInt {
                        dst: 3,
                        ty: IrType::Int(IntType::U16),
                        val: 0xF3C8,
                    },
                    IrInst::LoadPtr {
                        dst: 4,
                        addr: 3,
                        ty: u8t.clone(),
                        volatile: false,
                    },
                    IrInst::LoadPtr {
                        dst: 5,
                        addr: 3,
                        ty: u8t.clone(),
                        volatile: false,
                    },
                    // Unused volatile global load must NOT be forwarded or dropped by DCE!
                    IrInst::LoadPtr {
                        dst: 8,
                        addr: 0,
                        ty: u8t.clone(),
                        volatile: true,
                    },
                    IrInst::BinOp {
                        dst: 6,
                        op: IrBinOp::Add,
                        mode: ArithMode::Wrapping,
                        ty: u8t.clone(),
                        lhs: 4,
                        rhs: 5,
                        site_id: 0,
                    },
                    IrInst::BinOp {
                        dst: 7,
                        op: IrBinOp::Add,
                        mode: ArithMode::Wrapping,
                        ty: u8t,
                        lhs: 2,
                        rhs: 6,
                        site_id: 0,
                    },
                ],
                term: Terminator::Return(Some(7)),
            }],
            next_val: 9,
        });

        optimize_program(&mut prog);
        let insts = &prog.functions[0].blocks[0].insts;
        // Global non-volatile LoadPtr (dst 2) must be forwarded to 42 and removed;
        // the two MMIO loads + 1 unused volatile LoadPtr must remain.
        let load_ptrs: Vec<_> = insts
            .iter()
            .filter(|i| matches!(i, IrInst::LoadPtr { .. }))
            .collect();
        assert_eq!(
            load_ptrs.len(),
            3,
            "the 2 raw MMIO LoadPtrs and 1 unused volatile LoadPtr must remain"
        );
    }

    #[test]
    fn optimize_program_runs_ssa_sccp_gvn_and_rematerializes_params() {
        let u8t = IrType::Int(IntType::U8);
        let mut prog = IrProgram::default();
        prog.functions.push(IrFunction {
            id: 0,
            name: "ssa_opt".into(),
            is_pub: false,
            param_count: 1,
            locals: vec![
                LocalSlot {
                    id: 0,
                    name: "p".into(),
                    ty: u8t.clone(),
                    mutable: false,
                },
                LocalSlot {
                    id: 1,
                    name: "flag".into(),
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
            blocks: vec![
                BasicBlock {
                    id: 0,
                    insts: vec![
                        IrInst::ConstInt {
                            dst: 0,
                            ty: u8t.clone(),
                            val: 5,
                        },
                        IrInst::ConstInt {
                            dst: 1,
                            ty: u8t.clone(),
                            val: 3,
                        },
                        IrInst::BinOp {
                            dst: 2,
                            op: IrBinOp::Lt,
                            mode: ArithMode::Checked,
                            ty: u8t.clone(),
                            lhs: 1,
                            rhs: 0,
                            site_id: 0,
                        },
                    ],
                    term: Terminator::Branch {
                        cond: 2,
                        then_bb: 1,
                        else_bb: 2,
                    },
                },
                BasicBlock {
                    id: 1,
                    insts: vec![
                        IrInst::ConstInt {
                            dst: 3,
                            ty: u8t.clone(),
                            val: 10,
                        },
                        IrInst::StoreLocal {
                            slot: 1,
                            val: 3,
                            ty: u8t.clone(),
                        },
                    ],
                    term: Terminator::Goto(3),
                },
                BasicBlock {
                    id: 2,
                    insts: vec![
                        IrInst::ConstInt {
                            dst: 4,
                            ty: u8t.clone(),
                            val: 20,
                        },
                        IrInst::StoreLocal {
                            slot: 1,
                            val: 4,
                            ty: u8t.clone(),
                        },
                    ],
                    term: Terminator::Goto(3),
                },
                BasicBlock {
                    id: 3,
                    insts: vec![
                        IrInst::LoadLocal {
                            dst: 5,
                            slot: 1,
                            ty: u8t.clone(),
                        },
                        IrInst::LoadLocal {
                            dst: 6,
                            slot: 0,
                            ty: u8t.clone(),
                        },
                        IrInst::BinOp {
                            dst: 7,
                            op: IrBinOp::Add,
                            mode: ArithMode::Wrapping,
                            ty: u8t,
                            lhs: 5,
                            rhs: 6,
                            site_id: 0,
                        },
                    ],
                    term: Terminator::Return(Some(7)),
                },
            ],
            next_val: 8,
        });

        optimize_program(&mut prog);
        crate::verify::verify_program(&prog).expect("optimized program must verify");
        let f = &prog.functions[0];
        assert_eq!(f.blocks.len(), 1);
        assert!(f.blocks[0]
            .insts
            .iter()
            .any(|i| matches!(i, IrInst::ConstInt { val: 10, .. })));
    }

    #[test]
    fn opt_config_mcs51_preserves_u8_pow2_mul_div_and_skips_u8_osr() {
        let u8t = IrType::Int(IntType::U8);
        let u16t = IrType::Int(IntType::U16);
        let mut prog = IrProgram {
            target: Some(TargetDef {
                name: "ch559".into(),
                arch: "mcs51".into(),
                spaces: Vec::new(),
                stacks: Vec::new(),
                bank_groups: Vec::new(),
                interrupt_controllers: Vec::new(),
                overlay_slots: Vec::new(),
                on_swap_fn: None,
            }),
            ..IrProgram::default()
        };
        prog.functions.push(IrFunction {
            id: 0,
            name: "u8_math".into(),
            is_pub: true,
            param_count: 2,
            locals: vec![
                LocalSlot {
                    id: 0,
                    name: "x8".into(),
                    ty: u8t.clone(),
                    mutable: false,
                },
                LocalSlot {
                    id: 1,
                    name: "x16".into(),
                    ty: u16t.clone(),
                    mutable: false,
                },
            ],
            ret_ty: u16t.clone(),
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            isr_using: None,
            requires_banks: Vec::new(),
            requires_di: false,
            overlay_state: None,
            owner_state: None,
            blocks: vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::LoadLocal {
                        dst: 0,
                        slot: 0,
                        ty: u8t.clone(),
                    },
                    IrInst::ConstInt {
                        dst: 1,
                        ty: u8t.clone(),
                        val: 8,
                    },
                    IrInst::BinOp {
                        dst: 2,
                        op: IrBinOp::Mul,
                        mode: ArithMode::Wrapping,
                        ty: u8t.clone(),
                        lhs: 0,
                        rhs: 1,
                        site_id: 0,
                    },
                    IrInst::BinOp {
                        dst: 3,
                        op: IrBinOp::Div,
                        mode: ArithMode::Checked,
                        ty: u8t.clone(),
                        lhs: 2,
                        rhs: 1,
                        site_id: 0,
                    },
                    IrInst::BinOp {
                        dst: 4,
                        op: IrBinOp::Rem,
                        mode: ArithMode::Checked,
                        ty: u8t.clone(),
                        lhs: 3,
                        rhs: 1,
                        site_id: 0,
                    },
                    IrInst::LoadLocal {
                        dst: 5,
                        slot: 1,
                        ty: u16t.clone(),
                    },
                    IrInst::ConstInt {
                        dst: 6,
                        ty: u16t.clone(),
                        val: 8,
                    },
                    IrInst::BinOp {
                        dst: 7,
                        op: IrBinOp::Mul,
                        mode: ArithMode::Wrapping,
                        ty: u16t.clone(),
                        lhs: 5,
                        rhs: 6,
                        site_id: 0,
                    },
                    IrInst::Cast {
                        dst: 8,
                        from_ty: u8t.clone(),
                        to_ty: u16t.clone(),
                        val: 4,
                    },
                    IrInst::BinOp {
                        dst: 9,
                        op: IrBinOp::Add,
                        mode: ArithMode::Wrapping,
                        ty: u16t.clone(),
                        lhs: 7,
                        rhs: 8,
                        site_id: 0,
                    },
                ],
                term: Terminator::Return(Some(9)),
            }],
            next_val: 10,
        });

        optimize_program(&mut prog);
        let insts = &prog.functions[0].blocks[0].insts;
        assert!(insts.iter().any(|i| matches!(
            i,
            IrInst::BinOp {
                op: IrBinOp::Mul,
                ty: IrType::Int(IntType::U8),
                ..
            }
        )));
        assert!(insts.iter().any(|i| matches!(
            i,
            IrInst::BinOp {
                op: IrBinOp::Div,
                ty: IrType::Int(IntType::U8),
                ..
            }
        )));
        assert!(insts.iter().any(|i| matches!(
            i,
            IrInst::BinOp {
                op: IrBinOp::BitAnd,
                ty: IrType::Int(IntType::U8),
                ..
            }
        )));
        assert!(insts.iter().any(|i| matches!(
            i,
            IrInst::BinOp {
                op: IrBinOp::Shl,
                ty: IrType::Int(IntType::U16),
                ..
            }
        )));

        // Also test OSR u8 gating directly via osr_with_config vs osr
        let mut loop_fn = IrFunction {
            id: 1,
            name: "u8_loop".into(),
            is_pub: false,
            param_count: 1,
            locals: vec![LocalSlot {
                id: 0,
                name: "n".into(),
                ty: u8t.clone(),
                mutable: false,
            }],
            ret_ty: u8t.clone(),
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            isr_using: None,
            requires_banks: Vec::new(),
            requires_di: false,
            overlay_state: None,
            owner_state: None,
            blocks: vec![
                BasicBlock {
                    id: 0,
                    insts: vec![
                        IrInst::LoadLocal {
                            dst: 0,
                            slot: 0,
                            ty: u8t.clone(),
                        },
                        IrInst::ConstInt {
                            dst: 1,
                            ty: u8t.clone(),
                            val: 0,
                        },
                        IrInst::ConstInt {
                            dst: 2,
                            ty: u8t.clone(),
                            val: 1,
                        },
                        IrInst::ConstInt {
                            dst: 3,
                            ty: u8t.clone(),
                            val: 3,
                        },
                    ],
                    term: Terminator::Goto(1),
                },
                BasicBlock {
                    id: 1,
                    insts: vec![
                        IrInst::Phi {
                            dst: 4,
                            ty: u8t.clone(),
                            incoming: vec![(0, 1), (1, 7)],
                        },
                        IrInst::Phi {
                            dst: 5,
                            ty: u8t.clone(),
                            incoming: vec![(0, 1), (1, 8)],
                        },
                        IrInst::BinOp {
                            dst: 6,
                            op: IrBinOp::Mul,
                            mode: ArithMode::Wrapping,
                            ty: u8t.clone(),
                            lhs: 4,
                            rhs: 3,
                            site_id: 0,
                        },
                        IrInst::BinOp {
                            dst: 8,
                            op: IrBinOp::Add,
                            mode: ArithMode::Wrapping,
                            ty: u8t.clone(),
                            lhs: 5,
                            rhs: 6,
                            site_id: 0,
                        },
                        IrInst::BinOp {
                            dst: 7,
                            op: IrBinOp::Add,
                            mode: ArithMode::Wrapping,
                            ty: u8t.clone(),
                            lhs: 4,
                            rhs: 2,
                            site_id: 0,
                        },
                        IrInst::BinOp {
                            dst: 9,
                            op: IrBinOp::Lt,
                            mode: ArithMode::Wrapping,
                            ty: u8t.clone(),
                            lhs: 7,
                            rhs: 0,
                            site_id: 0,
                        },
                    ],
                    term: Terminator::Branch {
                        cond: 9,
                        then_bb: 1,
                        else_bb: 2,
                    },
                },
                BasicBlock {
                    id: 2,
                    insts: vec![],
                    term: Terminator::Return(Some(8)),
                },
            ],
            next_val: 10,
        };
        let mut loop_mcs = loop_fn.clone();
        assert!(!crate::osr::osr_with_config(&mut loop_mcs, false));
        assert!(crate::osr::osr(&mut loop_fn));
    }

    #[test]
    fn narrows_16bit_bitwise_and_wrapping_arith_before_u8_cast() {
        let u8t = IrType::Int(IntType::U8);
        let u16t = IrType::Int(IntType::U16);
        let mut func = IrFunction {
            id: 0,
            name: "hi_nibble".into(),
            is_pub: true,
            param_count: 1,
            locals: vec![LocalSlot {
                id: 0,
                name: "tp".into(),
                ty: u16t.clone(),
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
            blocks: vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::LoadLocal {
                        dst: 0,
                        slot: 0,
                        ty: u16t.clone(),
                    },
                    IrInst::ConstInt {
                        dst: 1,
                        ty: u16t.clone(),
                        val: 8,
                    },
                    IrInst::BinOp {
                        dst: 2,
                        op: IrBinOp::Shr,
                        mode: ArithMode::Wrapping,
                        ty: u16t.clone(),
                        lhs: 0,
                        rhs: 1,
                        site_id: 0,
                    },
                    IrInst::ConstInt {
                        dst: 3,
                        ty: u16t.clone(),
                        val: 0x0F,
                    },
                    IrInst::BinOp {
                        dst: 4,
                        op: IrBinOp::BitAnd,
                        mode: ArithMode::Wrapping,
                        ty: u16t.clone(),
                        lhs: 2,
                        rhs: 3,
                        site_id: 0,
                    },
                    IrInst::Cast {
                        dst: 5,
                        val: 4,
                        from_ty: u16t,
                        to_ty: u8t.clone(),
                    },
                ],
                term: Terminator::Return(Some(5)),
            }],
            next_val: 6,
        };
        optimize_function(&mut func);
        assert!(
            func.blocks[0]
                .insts
                .iter()
                .any(|i| matches!(i, IrInst::BinOp { op: IrBinOp::BitAnd, ty: IrType::Int(IntType::U8), .. })),
            "BitAnd before u8 cast must narrow to 8-bit BitAnd: {:?}",
            func.blocks[0].insts
        );
    }
}
