use super::runtime::emit_port_write_op;
use super::{EmitCtx, EmitGlobalEnv, RuntimeNeeds, fmt_uchiguri_mem};
use crate::analysis::{count_val_uses, is_signed_ty, z80_sym};
use crate::regalloc::RegLoc;
use aun_ir::*;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write;

pub(super) fn emit_function(
    env: &EmitGlobalEnv<'_>,
    func: &IrFunction,
    runtime_needs: &mut RuntimeNeeds,
    asm: &mut String,
) -> Result<(), String> {
    let program = env.program;
    let analysis = env.analysis;
    let layout = env.layout;
    let scratch_base = layout
        .fn_scratch_bases
        .get(&func.id)
        .copied()
        .unwrap_or(layout.ram_end);

    let fname = z80_sym(&func.name);
    writeln!(asm, "\n{fname}:").unwrap();

    if func.interrupt_source.is_some() {
        writeln!(asm, "    push af").unwrap();
        writeln!(asm, "    push bc").unwrap();
        writeln!(asm, "    push de").unwrap();
        writeln!(asm, "    push hl").unwrap();
        writeln!(asm, "    push ix").unwrap();
    }

    let fa = &analysis.fns[func.id as usize];
    let ix_param_slot = fa.ix_param_slot;
    let save_ix = fa.save_ix;
    if ix_param_slot.is_none() && save_ix {
        writeln!(asm, "    push ix").unwrap();
    }
    let mut fused_skip_insts: HashSet<(usize, usize)> = HashSet::new();

    // Pre-collect same-block fused `p = p + k` pointer step updates for `ix_local_slot`.
    let mut fused_ix_step: HashMap<ValueId, i16> = HashMap::new();
    if let Some(cslot) = fa.ix_local_slot {
        for (bi, bb) in func.blocks.iter().enumerate() {
            let mut blk_consts: HashMap<ValueId, i64> = HashMap::new();
            let mut blk_ix_loads: HashSet<ValueId> = HashSet::new();
            let mut blk_steps: HashMap<ValueId, (usize, i16)> = HashMap::new();
            for (ii, inst) in bb.insts.iter().enumerate() {
                match inst {
                    IrInst::ConstInt { dst, val, .. } => {
                        blk_consts.insert(*dst, *val);
                    }
                    IrInst::LoadLocal { dst, slot, .. } if *slot == cslot => {
                        blk_ix_loads.insert(*dst);
                    }
                    IrInst::BinOp {
                        dst,
                        op: IrBinOp::Add,
                        ty,
                        lhs,
                        rhs,
                        ..
                    } if program.type_size(ty) == 2 && count_val_uses(func, *dst) == 1 => {
                        let step_k = if blk_ix_loads.contains(lhs)
                            && let Some(&k) = blk_consts.get(rhs)
                            && (-32768..=32767).contains(&k)
                        {
                            Some(k as i16)
                        } else if blk_ix_loads.contains(rhs)
                            && let Some(&k) = blk_consts.get(lhs)
                            && (-32768..=32767).contains(&k)
                        {
                            Some(k as i16)
                        } else {
                            None
                        };
                        if let Some(k) = step_k {
                            blk_steps.insert(*dst, (ii, k));
                        }
                    }
                    IrInst::StoreLocal { slot, val, .. } if *slot == cslot => {
                        if let Some(&(binop_ii, k)) = blk_steps.get(val) {
                            fused_skip_insts.insert((bi, binop_ii));
                            fused_ix_step.insert(*val, k);
                        }
                        blk_ix_loads.clear();
                        blk_steps.clear();
                    }
                    _ => {}
                }
            }
        }
    }

    // Function-wide constant integer map (captures LICM-hoisted constants for latch countdown matching).
    let mut fn_consts: HashMap<ValueId, i64> = HashMap::new();
    for bb in &func.blocks {
        for inst in &bb.insts {
            if let IrInst::ConstInt { dst, val, .. } = inst {
                fn_consts.insert(*dst, *val);
            }
        }
    }
    let countdown_branches: Vec<Option<(usize, SlotId, RegLoc, BlockId, BlockId)>> = func
        .blocks
        .iter()
        .map(|bb| {
            detect_countdown_latch(program, func, bb, &fn_consts).and_then(
                |(skip_from, slot, then_bb, else_bb)| {
                    let reg = fa.reg_alloc.slot_reg(slot)?;
                    Some((skip_from, slot, reg, then_bb, else_bb))
                },
            )
        })
        .collect();

    // Pre-collect same-block fused `r = r + k` step updates for register-allocated locals (`B`, `C`, `BC`).
    let mut fused_reg_step: HashMap<ValueId, i16> = HashMap::new();
    if !fa.reg_alloc.slot_regs.is_empty() {
        for (bi, bb) in func.blocks.iter().enumerate() {
            let limit_ii = countdown_branches[bi]
                .map_or(bb.insts.len(), |(skip_from, _, _, _, _)| skip_from);
            let mut blk_consts: HashMap<ValueId, i64> = HashMap::new();
            let mut blk_reg_loads: HashMap<ValueId, (SlotId, RegLoc)> = HashMap::new();
            let mut blk_reg_steps: HashMap<ValueId, (SlotId, RegLoc, usize, i16)> = HashMap::new();
            let invalidate_reg_mask = |mask: crate::regalloc::RegMask,
                                       loads: &mut HashMap<ValueId, (SlotId, RegLoc)>,
                                       steps: &mut HashMap<ValueId, (SlotId, RegLoc, usize, i16)>| {
                loads.retain(|_, (_, r)| !r.mask().intersects(mask));
                steps.retain(|_, (_, r, _, _)| !r.mask().intersects(mask));
            };
            for (ii, inst) in bb.insts[..limit_ii].iter().enumerate() {
                if fused_skip_insts.contains(&(bi, ii)) {
                    continue;
                }
                match inst {
                    IrInst::ConstInt { dst, val, .. } => {
                        blk_consts.insert(*dst, *val);
                    }
                    IrInst::LoadLocal { dst, slot, .. } => {
                        if let Some(reg) = fa.reg_alloc.slot_reg(*slot) {
                            blk_reg_loads.insert(*dst, (*slot, reg));
                        }
                    }
                    IrInst::BinOp {
                        dst,
                        op: op @ (IrBinOp::Add | IrBinOp::Sub),
                        mode,
                        ty,
                        lhs,
                        rhs,
                        ..
                    } if *mode != ArithMode::Saturating && count_val_uses(func, *dst) == 1 => {
                        let sz = program.type_size(ty);
                        let matched = if let Some(&(slot, reg)) = blk_reg_loads.get(lhs)
                            && let Some(&k_raw) = blk_consts.get(rhs)
                        {
                            let k = if *op == IrBinOp::Sub { -k_raw } else { k_raw };
                            Some((slot, reg, k))
                        } else if *op == IrBinOp::Add
                            && let Some(&(slot, reg)) = blk_reg_loads.get(rhs)
                            && let Some(&k) = blk_consts.get(lhs)
                        {
                            Some((slot, reg, k))
                        } else {
                            None
                        };
                        if let Some((slot, reg, k)) = matched {
                            let ok = match reg {
                                RegLoc::BC => sz == 2 && (-2..=2).contains(&k) && k != 0,
                                RegLoc::B | RegLoc::C => sz == 1 && (-1..=1).contains(&k) && k != 0,
                            };
                            if ok {
                                blk_reg_steps.insert(*dst, (slot, reg, ii, k as i16));
                            }
                        }
                        let ic = crate::regalloc::inst_internal_clobber_mask(
                            program,
                            inst,
                            &blk_consts,
                        );
                        if ic != crate::regalloc::RegMask::NONE {
                            invalidate_reg_mask(ic, &mut blk_reg_loads, &mut blk_reg_steps);
                        }
                    }
                    IrInst::StoreLocal { slot, val, .. } => {
                        if let Some(reg) = fa.reg_alloc.slot_reg(*slot) {
                            if let Some(&(s_slot, _, binop_ii, k)) = blk_reg_steps.get(val)
                                && s_slot == *slot
                            {
                                fused_skip_insts.insert((bi, binop_ii));
                                fused_reg_step.insert(*val, k);
                            }
                            invalidate_reg_mask(reg.mask(), &mut blk_reg_loads, &mut blk_reg_steps);
                        }
                    }
                    IrInst::Call { func: cid, .. } => {
                        let cm = analysis
                            .fn_clobbers
                            .get(*cid as usize)
                            .copied()
                            .unwrap_or(crate::regalloc::RegMask::BC);
                        invalidate_reg_mask(cm, &mut blk_reg_loads, &mut blk_reg_steps);
                    }
                    IrInst::CallIndirect { .. } => {
                        invalidate_reg_mask(
                            crate::regalloc::RegMask::BC,
                            &mut blk_reg_loads,
                            &mut blk_reg_steps,
                        );
                    }
                    IrInst::ExternCall { extern_fn, .. } => {
                        let em = crate::regalloc::extern_clobber_mask(
                            &program.externs[*extern_fn as usize],
                        );
                        invalidate_reg_mask(em, &mut blk_reg_loads, &mut blk_reg_steps);
                    }
                    other => {
                        let ic = crate::regalloc::inst_internal_clobber_mask(
                            program,
                            other,
                            &blk_consts,
                        );
                        if ic != crate::regalloc::RegMask::NONE {
                            invalidate_reg_mask(ic, &mut blk_reg_loads, &mut blk_reg_steps);
                        }
                    }
                }
            }
        }
    }

    // Pre-collect same-block unsigned `Cast(u8 -> u16)` whose sole consumer is `IndexAddr`.
    let mut fused_index_u8_cast: HashMap<ValueId, ValueId> = HashMap::new();
    for (bi, bb) in func.blocks.iter().enumerate() {
        let mut val_slot: HashMap<ValueId, SlotId> = HashMap::new();
        let mut const_u8: HashSet<ValueId> = HashSet::new();
        let mut cands: HashMap<ValueId, (usize, ValueId, Option<SlotId>)> = HashMap::new();
        for (ii, inst) in bb.insts.iter().enumerate() {
            if fused_skip_insts.contains(&(bi, ii)) {
                continue;
            }
            match inst {
                IrInst::ConstInt { dst, ty, .. } if program.type_size(ty) == 1 => {
                    const_u8.insert(*dst);
                }
                IrInst::LoadLocal { dst, slot, .. } => {
                    val_slot.insert(*dst, *slot);
                }
                IrInst::Cast {
                    dst,
                    val,
                    from_ty,
                    to_ty,
                } if !is_signed_ty(from_ty)
                    && program.type_size(from_ty) == 1
                    && program.type_size(to_ty) == 2
                    && (val_slot.contains_key(val) || const_u8.contains(val))
                    && count_val_uses(func, *dst) == 1 =>
                {
                    cands.insert(*dst, (ii, *val, val_slot.get(val).copied()));
                }
                IrInst::IndexAddr { index, .. } => {
                    if let Some((cast_ii, u8_src, _)) = cands.remove(index) {
                        fused_skip_insts.insert((bi, cast_ii));
                        fused_index_u8_cast.insert(*index, u8_src);
                    }
                }
                IrInst::StoreLocal { slot, .. } => {
                    cands.retain(|_, (_, _, s)| *s != Some(*slot));
                }
                IrInst::StorePtr { .. }
                | IrInst::MemSet { .. }
                | IrInst::MemCopy { .. }
                | IrInst::MemFill { .. }
                | IrInst::Call { .. }
                | IrInst::CallIndirect { .. }
                | IrInst::ExternCall { .. }
                | IrInst::StateDispatch { .. }
                | IrInst::StateTransition { .. }
                | IrInst::PoolAlloc { .. }
                | IrInst::PoolFree { .. }
                | IrInst::QueuePush { .. }
                | IrInst::QueuePop { .. }
                | IrInst::BankEnter { .. }
                | IrInst::BankExit { .. }
                | IrInst::CriticalEnter
                | IrInst::CriticalExit
                | IrInst::InlineAsm { .. } => {
                    cands.clear();
                }
                _ => {}
            }
        }
    }

    // Entry: 1st param passed in A/HL (or directly in IX if `fa.ix_in_reg`); 2nd param passed
    // in E/DE (or IX) for eligible functions; remaining params live in Uchiguri parameter slots.
    if func.param_count > 0 {
        let p0 = &func.locals[0];
        let p0_addr = layout.slot_addrs[&(func.id, p0.id)];
        if ix_param_slot == Some(0) {
            if !fa.ix_in_reg {
                if save_ix {
                    writeln!(asm, "    push ix").unwrap();
                }
                writeln!(asm, "    push hl").unwrap();
                writeln!(asm, "    pop ix").unwrap();
            }
        } else if let Some(reg) = fa.reg_alloc.slot_reg(0) {
            match reg {
                crate::regalloc::RegLoc::C => writeln!(asm, "    ld c, a").unwrap(),
                crate::regalloc::RegLoc::B => writeln!(asm, "    ld b, a").unwrap(),
                crate::regalloc::RegLoc::BC => {
                    writeln!(asm, "    ld b, h").unwrap();
                    writeln!(asm, "    ld c, l").unwrap();
                }
            }
        } else {
            let m0 = fmt_uchiguri_mem(layout.ram_base, p0_addr);
            match program.type_size(&p0.ty) {
                1 => writeln!(asm, "    ld {m0}, a").unwrap(),
                2 => writeln!(asm, "    ld {m0}, hl").unwrap(),
                4 => {
                    let m2 = fmt_uchiguri_mem(layout.ram_base, p0_addr.wrapping_add(2));
                    writeln!(asm, "    ld {m0}, de").unwrap();
                    writeln!(asm, "    ld {m2}, hl").unwrap();
                }
                _ => {}
            }
        }
    }
    if fa.uses_reg_param1 {
        let p1 = &func.locals[1];
        let p1_addr = layout.slot_addrs[&(func.id, p1.id)];
        if ix_param_slot == Some(1) {
            if !fa.ix_in_reg {
                if save_ix {
                    writeln!(asm, "    push ix").unwrap();
                }
                writeln!(asm, "    push de").unwrap();
                writeln!(asm, "    pop ix").unwrap();
            }
        } else if let Some(reg) = fa.reg_alloc.slot_reg(1) {
            match reg {
                crate::regalloc::RegLoc::C => writeln!(asm, "    ld c, e").unwrap(),
                crate::regalloc::RegLoc::B => writeln!(asm, "    ld b, e").unwrap(),
                crate::regalloc::RegLoc::BC => {
                    writeln!(asm, "    ld b, d").unwrap();
                    writeln!(asm, "    ld c, e").unwrap();
                }
            }
        } else {
            let m1 = fmt_uchiguri_mem(layout.ram_base, p1_addr);
            match program.type_size(&p1.ty) {
                1 => {
                    writeln!(asm, "    ld a, e").unwrap();
                    writeln!(asm, "    ld {m1}, a").unwrap();
                }
                2 => writeln!(asm, "    ld {m1}, de").unwrap(),
                _ => {}
            }
        }
    }

    let hoisted_static_ix = fa.ix_static_elem.as_ref().is_some_and(|ise| {
        ise.hoist_param.is_some() && (!env.opts.check_bounds || ise.len == 0)
    });
    if hoisted_static_ix
        && let Some(ise) = fa.ix_static_elem.as_ref()
        && let Some(idx_slot) = ise.hoist_param
    {
        let idx_size = program.type_size(&func.locals[idx_slot as usize].ty);
        let small_u8_mul = super::runtime::is_small_u8_array_mul(ise.len, ise.elem_size);
        let use_mul_a =
            idx_size == 1 && !small_u8_mul && env.shared_muls.contains(&ise.elem_size);
        let reg_opt = fa.reg_alloc.slot_reg(idx_slot);
        if small_u8_mul {
            if idx_size == 1 {
                if let Some(r) = reg_opt {
                    let rname = r.reg8_name();
                    writeln!(asm, "    ld a, {rname}").unwrap();
                    super::runtime::emit_mul_u8_const(ise.elem_size, rname, asm);
                } else {
                    let sa = layout.slot_addrs[&(func.id, idx_slot)];
                    let sa_mem = fmt_uchiguri_mem(layout.ram_base, sa);
                    writeln!(asm, "    ld a, {sa_mem}").unwrap();
                    writeln!(asm, "    ld l, a").unwrap();
                    super::runtime::emit_mul_u8_const(ise.elem_size, "l", asm);
                }
            } else if reg_opt == Some(RegLoc::BC) {
                writeln!(asm, "    ld a, c").unwrap();
                super::runtime::emit_mul_u8_const(ise.elem_size, "c", asm);
            } else {
                let sa = layout.slot_addrs[&(func.id, idx_slot)];
                let sa_mem = fmt_uchiguri_mem(layout.ram_base, sa);
                writeln!(asm, "    ld a, {sa_mem}").unwrap();
                writeln!(asm, "    ld l, a").unwrap();
                super::runtime::emit_mul_u8_const(ise.elem_size, "l", asm);
            }
        } else if use_mul_a {
            if let Some(r) = reg_opt {
                writeln!(asm, "    ld a, {}", r.reg8_name()).unwrap();
            } else {
                let sa = layout.slot_addrs[&(func.id, idx_slot)];
                let sa_mem = fmt_uchiguri_mem(layout.ram_base, sa);
                writeln!(asm, "    ld a, {sa_mem}").unwrap();
            }
            writeln!(asm, "    call __aun_mul_a_{}", ise.elem_size).unwrap();
        } else {
            if idx_size == 1 {
                if let Some(r) = reg_opt {
                    writeln!(asm, "    ld l, {}", r.reg8_name()).unwrap();
                } else {
                    let sa = layout.slot_addrs[&(func.id, idx_slot)];
                    let sa_mem = fmt_uchiguri_mem(layout.ram_base, sa);
                    writeln!(asm, "    ld a, {sa_mem}").unwrap();
                    writeln!(asm, "    ld l, a").unwrap();
                }
                writeln!(asm, "    ld h, 0").unwrap();
            } else if reg_opt == Some(RegLoc::BC) {
                writeln!(asm, "    ld l, c").unwrap();
                writeln!(asm, "    ld h, b").unwrap();
            } else {
                let sa = layout.slot_addrs[&(func.id, idx_slot)];
                let sa_mem = fmt_uchiguri_mem(layout.ram_base, sa);
                writeln!(asm, "    ld hl, {sa_mem}").unwrap();
            }
            if env.shared_muls.contains(&ise.elem_size) {
                writeln!(asm, "    call __aun_mul_hl_{}", ise.elem_size).unwrap();
            } else {
                super::runtime::emit_mul_hl_const(ise.elem_size, asm);
            }
        }
        let g = &program.globals[ise.global as usize];
        if g.in_rom || g.is_extern {
            if ise.base_offset == 0 {
                writeln!(asm, "    ld de, {}", g.name).unwrap();
            } else {
                writeln!(asm, "    ld de, {} + {}", g.name, ise.base_offset).unwrap();
            }
        } else {
            let ga = layout.global_addrs[&ise.global].wrapping_add(ise.base_offset);
            let is_u = g.fixed_addr.is_none() && g.space.is_none();
            let imm = if is_u {
                super::fmt_uchiguri_imm(layout.ram_base, ga)
            } else {
                format!("0x{ga:04X}")
            };
            writeln!(asm, "    ld de, {imm}").unwrap();
        }
        writeln!(asm, "    add hl, de").unwrap();
        writeln!(asm, "    push hl").unwrap();
        writeln!(asm, "    pop ix").unwrap();
    }

    if hoisted_static_ix && let Some(ise) = fa.ix_static_elem.as_ref() {
        let mut dead_idx_vals: HashSet<ValueId> = HashSet::new();
        let all_uses_dead = |v: ValueId, skip_set: &HashSet<(usize, usize)>, dead_set: &HashSet<ValueId>| {
            for (bi, bb) in func.blocks.iter().enumerate() {
                for (ii, inst) in bb.insts.iter().enumerate() {
                    if skip_set.contains(&(bi, ii)) || !inst.uses_val(v) {
                        continue;
                    }
                    let ok = match inst {
                        IrInst::IndexAddr { dst, index, .. }
                            if ise.index_vals.contains(dst) && *index == v =>
                        {
                            true
                        }
                        IrInst::Cast { dst, val, .. } if dead_set.contains(dst) && *val == v => true,
                        _ => false,
                    };
                    if !ok {
                        return false;
                    }
                }
                if matches!(
                    bb.term,
                    Terminator::Return(Some(tv)) | Terminator::Branch { cond: tv, .. } if tv == v
                ) {
                    return false;
                }
            }
            true
        };
        for (bi, bb) in func.blocks.iter().enumerate() {
            for (ii, inst) in bb.insts.iter().enumerate() {
                if let IrInst::Cast { dst, .. } = inst
                    && all_uses_dead(*dst, &fused_skip_insts, &dead_idx_vals)
                {
                    fused_skip_insts.insert((bi, ii));
                    dead_idx_vals.insert(*dst);
                }
            }
        }
        for (bi, bb) in func.blocks.iter().enumerate() {
            for (ii, inst) in bb.insts.iter().enumerate() {
                if let IrInst::LoadLocal { dst, .. } = inst
                    && all_uses_dead(*dst, &fused_skip_insts, &dead_idx_vals)
                {
                    fused_skip_insts.insert((bi, ii));
                }
            }
        }
    }

    let fused_cmp_dsts = &fa.fused_cmp_dsts;

    // Pre-collect same-block fused 16-bit `x ^ (x << 7|8)` and unsigned `x ^ (x >> 8..=15)`.
    let mut fused_xor_shift: HashMap<ValueId, (ValueId, IrBinOp, u8)> = HashMap::new();
    {
        let mut quick_consts: HashMap<ValueId, i64> = HashMap::new();
        for (bi, bb) in func.blocks.iter().enumerate() {
            // Map shift dst -> (inst_idx, src_val, src_slot_opt, op, k)
            let mut blk_shifts: HashMap<ValueId, (usize, ValueId, Option<SlotId>, IrBinOp, u8)> =
                HashMap::new();
            // Track which local slot each `LoadLocal` came from (invalidated on writes to that slot)
            let mut val_from_slot: HashMap<ValueId, SlotId> = HashMap::new();
            for (ii, inst) in bb.insts.iter().enumerate() {
                if fused_skip_insts.contains(&(bi, ii)) {
                    continue;
                }
                match inst {
                    IrInst::ConstInt { dst, val, .. } => {
                        quick_consts.insert(*dst, *val);
                    }
                    IrInst::LoadLocal { dst, slot, .. } => {
                        val_from_slot.insert(*dst, *slot);
                    }
                    IrInst::StoreLocal { slot, .. } => {
                        val_from_slot.retain(|_, s| *s != *slot);
                        blk_shifts.retain(|_, (_, _, s_opt, _, _)| *s_opt != Some(*slot));
                    }
                    IrInst::StorePtr { .. }
                    | IrInst::MemSet { .. }
                    | IrInst::MemCopy { .. }
                    | IrInst::MemFill { .. }
                    | IrInst::Call { .. }
                    | IrInst::CallIndirect { .. }
                    | IrInst::ExternCall { .. }
                    | IrInst::StateDispatch { .. }
                    | IrInst::StateTransition { .. }
                    | IrInst::PoolAlloc { .. }
                    | IrInst::PoolFree { .. }
                    | IrInst::QueuePush { .. }
                    | IrInst::QueuePop { .. }
                    | IrInst::BankEnter { .. }
                    | IrInst::BankExit { .. }
                    | IrInst::CriticalEnter
                    | IrInst::CriticalExit
                    | IrInst::InlineAsm { .. } => {
                        val_from_slot.clear();
                        blk_shifts.clear();
                    }
                    IrInst::BinOp {
                        dst,
                        op,
                        ty,
                        lhs,
                        rhs,
                        ..
                    } if program.type_size(ty) == 2 => {
                        if matches!(op, IrBinOp::Shl | IrBinOp::Shr)
                            && let Some(&k_raw) = quick_consts.get(rhs)
                            && count_val_uses(func, *dst) == 1
                        {
                            let k = k_raw.clamp(0, 16) as u8;
                            let supported = matches!(
                                (op, k, is_signed_ty(ty)),
                                (IrBinOp::Shl, 7 | 8, _) | (IrBinOp::Shr, 8..=15, false)
                            );
                            if supported {
                                let src_slot = val_from_slot.get(lhs).copied();
                                blk_shifts.insert(*dst, (ii, *lhs, src_slot, *op, k));
                            }
                        } else if *op == IrBinOp::BitXor {
                            let same_src = |sx: ValueId, s_slot: Option<SlotId>, other: ValueId| {
                                sx == other
                                    || (s_slot.is_some()
                                        && s_slot == val_from_slot.get(&other).copied())
                            };
                            let matched =
                                if let Some(&(s_ii, sx, s_slot, sop, sk)) = blk_shifts.get(rhs)
                                    && same_src(sx, s_slot, *lhs)
                                {
                                    Some((s_ii, *lhs, sop, sk))
                                } else if let Some(&(s_ii, sx, s_slot, sop, sk)) =
                                    blk_shifts.get(lhs)
                                    && same_src(sx, s_slot, *rhs)
                                {
                                    Some((s_ii, *rhs, sop, sk))
                                } else {
                                    None
                                };
                            if let Some((s_ii, sx, sop, sk)) = matched {
                                fused_skip_insts.insert((bi, s_ii));
                                fused_xor_shift.insert(*dst, (sx, sop, sk));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Pre-collect same-block fused `FixedInt(2, 6)` -> `Cast(2 -> 1, non-Bool)` (`add hl, hl; add hl, hl; ld a, h`),
    // and fused `BinOp(Shr, 2, const 8)` -> `Cast(2 -> 1, non-Bool)` high-byte extraction (`ld a, (src+1)`).
    let mut fused_q10_6_u8: HashMap<ValueId, ValueId> = HashMap::new();
    let mut fused_hi8_u8: HashMap<ValueId, ValueId> = HashMap::new();
    for (bi, bb) in func.blocks.iter().enumerate() {
        let mut blk_fx6: HashMap<ValueId, (usize, ValueId)> = HashMap::new();
        let mut blk_shr8: HashMap<ValueId, (usize, ValueId)> = HashMap::new();
        let mut blk_consts: HashMap<ValueId, i64> = HashMap::new();
        for (ii, inst) in bb.insts.iter().enumerate() {
            if fused_skip_insts.contains(&(bi, ii)) {
                continue;
            }
            match inst {
                IrInst::ConstInt { dst, val, .. } => {
                    blk_consts.insert(*dst, *val);
                }
                IrInst::FixedInt {
                    dst,
                    val,
                    base,
                    frac_bits: 6,
                } if base.byte_size() == 2 && count_val_uses(func, *dst) == 1 => {
                    blk_fx6.insert(*dst, (ii, *val));
                }
                IrInst::BinOp {
                    dst,
                    op: IrBinOp::Shr,
                    ty,
                    lhs,
                    rhs,
                    ..
                } if program.type_size(ty) == 2
                    && blk_consts.get(rhs) == Some(&8)
                    && count_val_uses(func, *dst) == 1 =>
                {
                    blk_shr8.insert(*dst, (ii, *lhs));
                }
                IrInst::Cast {
                    dst,
                    val,
                    from_ty,
                    to_ty,
                } if program.type_size(from_ty) == 2
                    && program.type_size(to_ty) == 1
                    && *to_ty != IrType::Bool =>
                {
                    // Require immediately adjacent instruction so `src_v`'s colored
                    // scratch slot cannot be overwritten by an intervening instruction.
                    if let Some(&(fx_ii, src_v)) = blk_fx6.get(val)
                        && ii == fx_ii + 1
                    {
                        fused_skip_insts.insert((bi, fx_ii));
                        fused_q10_6_u8.insert(*dst, src_v);
                    } else if let Some(&(sh_ii, src_v)) = blk_shr8.get(val)
                        && ii == sh_ii + 1
                    {
                        fused_skip_insts.insert((bi, sh_ii));
                        fused_hi8_u8.insert(*dst, src_v);
                    }
                }
                IrInst::StoreLocal { .. }
                | IrInst::StorePtr { .. }
                | IrInst::MemSet { .. }
                | IrInst::MemCopy { .. }
                | IrInst::MemFill { .. }
                | IrInst::Call { .. }
                | IrInst::CallIndirect { .. }
                | IrInst::ExternCall { .. }
                | IrInst::StateDispatch { .. }
                | IrInst::StateTransition { .. }
                | IrInst::PoolAlloc { .. }
                | IrInst::PoolFree { .. }
                | IrInst::QueuePush { .. }
                | IrInst::QueuePop { .. }
                | IrInst::BankEnter { .. }
                | IrInst::BankExit { .. }
                | IrInst::CriticalEnter
                | IrInst::CriticalExit
                | IrInst::InlineAsm { .. } => {
                    blk_fx6.clear();
                    blk_shr8.clear();
                }
                _ => {}
            }
        }
    }

    // Pre-collect same-block `(u8 as u16) << k` and `u16 | ((u8 as u16) << k)` / `u16 | (u8 as u16)` packs.
    let mut fused_u8_shl16: HashMap<ValueId, (ValueId, u8)> = HashMap::new();
    let mut fused_bitor_u8_term: HashMap<ValueId, (ValueId, ValueId, u8)> = HashMap::new();
    for (bi, bb) in func.blocks.iter().enumerate() {
        let mut u8_casts: HashMap<ValueId, (usize, ValueId)> = HashMap::new();
        let mut shl16_cands: HashMap<ValueId, (usize, ValueId, u8)> = HashMap::new();
        let only_folded_consts_between =
            |start_ii: usize, end_ii: usize, skip_set: &HashSet<(usize, usize)>| -> bool {
                ((start_ii + 1)..end_ii).all(|k| {
                    skip_set.contains(&(bi, k))
                        || matches!(
                            bb.insts[k],
                            IrInst::ConstInt { dst, .. }
                                if crate::analysis::is_const_always_folded(
                                    program,
                                    func,
                                    dst,
                                    fused_cmp_dsts,
                                    skip_set,
                                )
                        )
                })
            };
        for (ii, inst) in bb.insts.iter().enumerate() {
            if fused_skip_insts.contains(&(bi, ii)) {
                continue;
            }
            match inst {
                IrInst::Cast {
                    dst,
                    val,
                    from_ty,
                    to_ty,
                } if !is_signed_ty(from_ty)
                    && program.type_size(from_ty) == 1
                    && program.type_size(to_ty) == 2
                    && *to_ty != IrType::Bool
                    && count_val_uses(func, *dst) == 1
                    && !fa.direct_reg_dsts.contains_key(dst) =>
                {
                    u8_casts.insert(*dst, (ii, *val));
                }
                IrInst::BinOp {
                    dst,
                    op: IrBinOp::Shl,
                    ty,
                    lhs,
                    rhs,
                    ..
                } if program.type_size(ty) == 2 => {
                    if let Some(&k_raw) = fn_consts.get(rhs)
                        && k_raw >= 0
                        && let Some(&(c_ii, x_v)) = u8_casts.get(lhs)
                        && only_folded_consts_between(c_ii, ii, &fused_skip_insts)
                    {
                        let k = k_raw.clamp(0, 16) as u8;
                        fused_skip_insts.insert((bi, c_ii));
                        fused_u8_shl16.insert(*dst, (x_v, k));
                        if count_val_uses(func, *dst) == 1
                            && !fa.direct_reg_dsts.contains_key(dst)
                        {
                            shl16_cands.insert(*dst, (ii, x_v, k));
                        }
                    }
                }
                IrInst::BinOp {
                    dst,
                    op: IrBinOp::BitOr,
                    ty,
                    lhs,
                    rhs,
                    ..
                } if program.type_size(ty) == 2 && *lhs != *rhs => {
                    for &(side, other) in &[(*rhs, *lhs), (*lhs, *rhs)] {
                        if let Some(&(term_ii, x_v, k)) = shl16_cands.get(&side)
                            && only_folded_consts_between(term_ii, ii, &fused_skip_insts)
                        {
                            fused_skip_insts.insert((bi, term_ii));
                            fused_u8_shl16.remove(&side);
                            fused_bitor_u8_term.insert(*dst, (other, x_v, k));
                            break;
                        }
                        if let Some(&(term_ii, x_v)) = u8_casts.get(&side)
                            && only_folded_consts_between(term_ii, ii, &fused_skip_insts)
                        {
                            fused_skip_insts.insert((bi, term_ii));
                            fused_bitor_u8_term.insert(*dst, (other, x_v, 0));
                            break;
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // Pre-collect adjacent 32-bit unsigned `(Div, Rem)` / `(Rem, Div)` pairs for `__aun_udivrem32`
    // fusion, plus function-wide `u8` / `u16` range bounds on `u32` values.
    let mut fused_divrem_partner: HashMap<ValueId, (IrBinOp, ValueId)> = HashMap::new();
    let mut u8_bound_vals: HashSet<ValueId> = HashSet::new();
    let mut u16_bound_vals: HashSet<ValueId> = HashSet::new();
    let mut i16_bound_vals: HashSet<ValueId> = HashSet::new();
    for (bi, bb) in func.blocks.iter().enumerate() {
        let mut ii = 0;
        while ii + 1 < bb.insts.len() {
            if !fused_skip_insts.contains(&(bi, ii))
                && !fused_skip_insts.contains(&(bi, ii + 1))
                && let (
                    IrInst::BinOp {
                        dst: d1,
                        op: op1 @ (IrBinOp::Div | IrBinOp::Rem),
                        mode: m1,
                        ty: ty1 @ IrType::Int(IntType::U32),
                        lhs: l1,
                        rhs: r1,
                        ..
                    },
                    IrInst::BinOp {
                        dst: d2,
                        op: op2 @ (IrBinOp::Div | IrBinOp::Rem),
                        mode: m2,
                        ty: IrType::Int(IntType::U32),
                        lhs: l2,
                        rhs: r2,
                        ..
                    },
                ) = (&bb.insts[ii], &bb.insts[ii + 1])
                && op1 != op2
                && m1 == m2
                && !is_signed_ty(ty1)
                && l1 == l2
                && r1 == r2
                && d1 != d2
                && count_val_uses(func, *d1) > 0
                && count_val_uses(func, *d2) > 0
            {
                fused_skip_insts.insert((bi, ii + 1));
                fused_divrem_partner.insert(*d1, (*op2, *d2));
                ii += 2;
                continue;
            }
            ii += 1;
        }
        for inst in &bb.insts {
            match inst {
                IrInst::ConstInt {
                    dst,
                    ty: IrType::Int(IntType::U32),
                    val,
                } => {
                    if (0..=255).contains(val) {
                        u8_bound_vals.insert(*dst);
                        u16_bound_vals.insert(*dst);
                    } else if (0..=65535).contains(val) {
                        u16_bound_vals.insert(*dst);
                    }
                }
                IrInst::Cast {
                    dst,
                    from_ty,
                    to_ty: IrType::Int(IntType::U32),
                    ..
                } if !is_signed_ty(from_ty) => {
                    match program.type_size(from_ty) {
                        1 => {
                            u8_bound_vals.insert(*dst);
                            u16_bound_vals.insert(*dst);
                        }
                        2 => {
                            u16_bound_vals.insert(*dst);
                        }
                        _ => {}
                    }
                }
                IrInst::ConstInt {
                    dst,
                    ty: IrType::Int(IntType::I32),
                    val,
                } if (-32768..=32767).contains(val) => {
                    i16_bound_vals.insert(*dst);
                }
                IrInst::Cast {
                    dst,
                    from_ty,
                    to_ty: IrType::Int(IntType::I32),
                    ..
                } => {
                    let sz = program.type_size(from_ty);
                    if sz == 1 || (sz == 2 && is_signed_ty(from_ty)) {
                        i16_bound_vals.insert(*dst);
                    }
                }
                _ => {}
            }
        }
    }

    // Pre-collect single-predecessor blocks where a signed 16-bit `(ix + d)` field is known >= 0.
    let mut nonneg_ix_blocks: HashMap<BlockId, i8> = HashMap::new();
    if ix_param_slot == Some(0) && !func.blocks.is_empty() {
        let mut pred_counts: HashMap<BlockId, usize> = HashMap::new();
        // Implicit function-entry predecessor edge into block 0
        *pred_counts.entry(func.blocks[0].id).or_insert(0) += 1;
        for bb in &func.blocks {
            match bb.term {
                Terminator::Goto(t) => *pred_counts.entry(t).or_insert(0) += 1,
                Terminator::Branch {
                    then_bb, else_bb, ..
                } => {
                    *pred_counts.entry(then_bb).or_insert(0) += 1;
                    *pred_counts.entry(else_bb).or_insert(0) += 1;
                }
                _ => {}
            }
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
        let mut quick_consts: HashMap<ValueId, i64> = HashMap::new();
        let mut quick_p0_addrs: HashMap<ValueId, ()> = HashMap::new();
        let mut quick_ix: HashMap<ValueId, i8> = HashMap::new();
        for &bi in &visit_order {
            let bb = &func.blocks[bi];
            // Reset per-block loaded `(ix + d)` tracking so cross-block mutations cannot leak
            let mut quick_loaded_ix: HashMap<ValueId, i8> = HashMap::new();
            for inst in &bb.insts {
                match inst {
                    IrInst::ConstInt { dst, val, .. } => {
                        quick_consts.insert(*dst, *val);
                    }
                    IrInst::AddrLocal { dst, slot: 0 } => {
                        quick_p0_addrs.insert(*dst, ());
                    }
                    IrInst::LoadLocal { dst, slot: 0, .. } => {
                        quick_ix.insert(*dst, 0);
                    }
                    IrInst::LoadPtr { dst, addr, .. } if quick_p0_addrs.contains_key(addr) => {
                        quick_ix.insert(*dst, 0);
                    }
                    IrInst::FieldAddr {
                        dst,
                        base,
                        byte_offset,
                    } if quick_ix.contains_key(base) => {
                        if let Some(d) = (quick_ix[base] as i32)
                            .checked_add(*byte_offset as i32)
                            .filter(|&x| (-128..=125).contains(&x))
                        {
                            quick_ix.insert(*dst, d as i8);
                        }
                    }
                    IrInst::LoadPtr { dst, addr, ty, .. }
                        if program.type_size(ty) == 2 && quick_ix.contains_key(addr) =>
                    {
                        quick_loaded_ix.insert(*dst, quick_ix[addr]);
                    }
                    IrInst::StorePtr { .. }
                    | IrInst::MemSet { .. }
                    | IrInst::MemCopy { .. }
                    | IrInst::MemFill { .. }
                    | IrInst::Call { .. }
                    | IrInst::CallIndirect { .. }
                    | IrInst::ExternCall { .. }
                    | IrInst::StateDispatch { .. }
                    | IrInst::StateTransition { .. }
                    | IrInst::PoolAlloc { .. }
                    | IrInst::PoolFree { .. }
                    | IrInst::QueuePush { .. }
                    | IrInst::QueuePop { .. }
                    | IrInst::BankEnter { .. }
                    | IrInst::BankExit { .. }
                    | IrInst::CriticalEnter
                    | IrInst::CriticalExit
                    | IrInst::InlineAsm { .. } => {
                        quick_loaded_ix.clear();
                    }
                    _ => {}
                }
            }
            if let Terminator::Branch {
                cond,
                then_bb,
                else_bb,
            } = bb.term
                && let Some(IrInst::BinOp {
                    dst,
                    op,
                    ty,
                    lhs,
                    rhs,
                    ..
                }) = bb.insts.last()
                && *dst == cond
                && program.type_size(ty) == 2
                && is_signed_ty(ty)
                && quick_consts.get(rhs) == Some(&0)
                && let Some(&disp) = quick_loaded_ix.get(lhs)
            {
                let nn_bb = match op {
                    IrBinOp::Lt => Some(else_bb),
                    IrBinOp::Ge => Some(then_bb),
                    _ => None,
                };
                if let Some(target_bb) = nn_bb
                    && pred_counts.get(&target_bb).copied() == Some(1)
                {
                    nonneg_ix_blocks.insert(target_bb, disp);
                }
            }
        }
    }

    let val_offsets: &[u16] = layout
        .fn_val_offsets
        .get(&func.id)
        .map(Vec::as_slice)
        .unwrap_or(&[]);

    let escaped_mask = func.escaped_locals();
    let escaped_slot_ranges: Vec<(u16, u16)> = func
        .locals
        .iter()
        .enumerate()
        .filter_map(|(i, l)| {
            if escaped_mask.get(i).copied().unwrap_or(false)
                && fa.reg_alloc.slot_reg(l.id).is_none()
            {
                Some((
                    layout.slot_addrs[&(func.id, l.id)],
                    program.type_size(&l.ty).max(1),
                ))
            } else {
                None
            }
        })
        .collect();

    let mut ctx = EmitCtx {
        program,
        analysis,
        func,
        fa,
        layout,
        scratch_base,
        val_offsets,
        opts: env.opts.clone(),
        shared_muls: env.shared_muls,
        fused_skip_insts: &fused_skip_insts,
        fused_ix_step: &fused_ix_step,
        fused_reg_step: &fused_reg_step,
        fused_index_u8_cast: &fused_index_u8_cast,
        fused_xor_shift: &fused_xor_shift,
        fused_q10_6_u8: &fused_q10_6_u8,
        fused_hi8_u8: &fused_hi8_u8,
        fused_u8_shl16: &fused_u8_shl16,
        fused_bitor_u8_term: &fused_bitor_u8_term,
        fused_divrem_partner: &fused_divrem_partner,
        u8_bound_vals: &u8_bound_vals,
        u16_bound_vals: &u16_bound_vals,
        i16_bound_vals: &i16_bound_vals,
        escaped_slot_ranges: &escaped_slot_ranges,
        block_nonneg_ix: None,
        nonneg_vals: BTreeSet::new(),
        const_vals: HashMap::new(),
        possible_bits: HashMap::new(),
        const_addrs: HashMap::new(),
        const_addr_srcs: HashMap::new(),
        const_addr_loads: HashMap::new(),
        rom_addrs: HashMap::new(),
        val_aliases: HashMap::new(),
        val_from_local: HashMap::new(),
        const_slices: HashMap::new(),
        ix_addrs: HashMap::new(),
        ix_param_addr_vals: BTreeSet::new(),
        ix_val_loads: HashMap::new(),
        aligned256_addrs: HashMap::new(),
        val_in_reg: HashMap::new(),
        shift_cse: HashMap::new(),
        index_cse: HashMap::new(),
        hl_ptr: None,
        hl_val: None,
        label_counter: 0,
        plan: env.plan,
        resolved_overlay_syms: env.resolved_overlay_syms,
        pass1_placeholders: env.pass1_placeholders,
        cur_bb_is_tail_return_none: false,
        cur_bb_inst_len: 0,
        tail_jumped_trampoline: false,
        tail_cold_stubs: String::new(),
        runtime_needs,
        asm,
    };

    let bcl_by_hdr: HashMap<BlockId, &crate::analysis::ByteCopyLoop> = fa
        .byte_copy_loops
        .iter()
        .map(|b| (b.hdr_bb, b))
        .collect();
    let bcl_skip_steps: HashSet<BlockId> = fa
        .byte_copy_loops
        .iter()
        .filter(|b| b.step_bb != b.hdr_bb)
        .map(|b| b.step_bb)
        .collect();
    let mut bcl_spill_tramps: Vec<(String, SlotId, BlockId)> = Vec::new();

    let scan_loops = detect_scan_loops(
        program,
        func,
        fa,
        &layout.slot_addrs,
        &countdown_branches,
        &fn_consts,
    );
    let mut scan_by_preheader: HashMap<usize, usize> = HashMap::new();
    let mut scan_by_header: HashMap<usize, usize> = HashMap::new();
    let mut scan_by_latch: HashMap<usize, usize> = HashMap::new();
    let mut scan_by_cold_end: HashMap<usize, usize> = HashMap::new();
    let mut scan_block_to_loop: HashMap<BlockId, usize> = HashMap::new();
    for (si, sl) in scan_loops.iter().enumerate() {
        scan_by_preheader.insert(sl.preheader_bi, si);
        scan_by_header.insert(sl.header_bi, si);
        scan_by_latch.insert(sl.latch_bi, si);
        for &cid in &sl.cold_region {
            scan_block_to_loop.insert(cid, si);
        }
    }

    let mut bb_order = compute_bb_order(func, &scan_loops);
    bb_order.retain(|&idx| !bcl_skip_steps.contains(&func.blocks[idx].id));
    // Find the last block in `bb_order` belonging to each scan loop's `cold_region` so `__scan_reenter`
    // is emitted right after it (allowing fall-through when it ends in `Goto(latch_id)`).
    for (si, sl) in scan_loops.iter().enumerate() {
        if sl.reenters_latch
            && let Some(last_pos) = bb_order
                .iter()
                .rposition(|&bi| sl.cold_region.contains(&func.blocks[bi].id))
        {
            scan_by_cold_end.insert(bb_order[last_pos], si);
        }
    }
    for (order_pos, &bb_idx) in bb_order.iter().enumerate() {
        let bb = &func.blocks[bb_idx];
        writeln!(ctx.asm, "{fname}_bb{}:", bb.id).unwrap();
        let next_bb_id = bb_order.get(order_pos + 1).map(|&idx| func.blocks[idx].id);

        if let Some(&bcl) = bcl_by_hdr.get(&bb.id) {
            writeln!(ctx.asm, "    ld a, (hl)").unwrap();
            if let Some((sval, exit_on_eq)) = bcl.sentinel {
                if sval == 0 {
                    writeln!(ctx.asm, "    or a").unwrap();
                } else {
                    writeln!(ctx.asm, "    cp 0x{sval:02X}").unwrap();
                }
                let exit_s = bcl.exit_sentinel_bb.unwrap();
                let cc = if exit_on_eq { "z" } else { "nz" };
                if bcl.dst_live_out {
                    let spill_lbl = ctx.fresh_label("bcl_exit");
                    writeln!(ctx.asm, "    jp {cc}, {spill_lbl}").unwrap();
                    bcl_spill_tramps.push((spill_lbl, bcl.dst_slot, exit_s));
                } else {
                    writeln!(ctx.asm, "    jp {cc}, {fname}_bb{exit_s}").unwrap();
                }
            }
            writeln!(ctx.asm, "    ld (de), a").unwrap();
            writeln!(ctx.asm, "    inc hl").unwrap();
            let inc_e = bcl.inc_e_fixed || (bcl.inc_e_ram_global && !ctx.opts.relocatable_ram);
            if inc_e {
                writeln!(ctx.asm, "    inc e").unwrap();
            } else {
                writeln!(ctx.asm, "    inc de").unwrap();
            }
            if bcl.is_u8_cnt {
                writeln!(ctx.asm, "    djnz {fname}_bb{}", bcl.hdr_bb).unwrap();
            } else {
                writeln!(ctx.asm, "    dec bc").unwrap();
                writeln!(ctx.asm, "    ld a, b").unwrap();
                writeln!(ctx.asm, "    or c").unwrap();
                writeln!(ctx.asm, "    jp nz, {fname}_bb{}", bcl.hdr_bb).unwrap();
            }
            if bcl.dst_live_out {
                let dst_mem =
                    fmt_uchiguri_mem(layout.ram_base, layout.slot_addrs[&(func.id, bcl.dst_slot)]);
                writeln!(ctx.asm, "    ld {dst_mem}, de").unwrap();
            }
            if next_bb_id != Some(bcl.exit_bound_bb) {
                writeln!(ctx.asm, "    jp {fname}_bb{}", bcl.exit_bound_bb).unwrap();
            }
            continue;
        }

        if let Some(&si) = scan_by_header.get(&bb_idx) {
            let sl = &scan_loops[si];
            ctx.reset_bb(None);
            writeln!(ctx.asm, "    ld a, (hl)").unwrap();
            match sl.cmp_imm {
                None | Some(0) => writeln!(ctx.asm, "    or a").unwrap(),
                Some(k) => writeln!(ctx.asm, "    cp 0x{k:02X}").unwrap(),
            }
            let cold_entry = if sl.needs_cold_save {
                format!("__scan_cold_save_{fname}_{}", sl.header_id)
            } else {
                format!("{fname}_bb{}", sl.cold_id)
            };
            writeln!(ctx.asm, "    jp {}, {cold_entry}", sl.jp_cc_to_cold).unwrap();
            if next_bb_id != Some(sl.latch_id) {
                writeln!(ctx.asm, "    jp {fname}_bb{}", sl.latch_id).unwrap();
            }
            continue;
        }

        if let Some(&si) = scan_by_latch.get(&bb_idx) {
            let sl = &scan_loops[si];
            ctx.reset_bb(None);
            writeln!(ctx.asm, "    add hl, de").unwrap();
            match sl.countdown_reg {
                RegLoc::B => writeln!(ctx.asm, "    djnz {fname}_bb{}", sl.header_id).unwrap(),
                RegLoc::C => {
                    writeln!(ctx.asm, "    dec c").unwrap();
                    writeln!(ctx.asm, "    jp nz, {fname}_bb{}", sl.header_id).unwrap();
                }
                RegLoc::BC => unreachable!(),
            }
            if next_bb_id != Some(sl.exit_id) {
                writeln!(ctx.asm, "    jp {fname}_bb{}", sl.exit_id).unwrap();
            }
            if sl.needs_cold_save {
                writeln!(ctx.asm, "__scan_cold_save_{fname}_{}:", sl.header_id).unwrap();
                emit_scan_hl_sub_field_off(ctx.asm, sl.field_off);
                if sl.ptr_is_ix {
                    writeln!(ctx.asm, "    push hl").unwrap();
                    writeln!(ctx.asm, "    pop ix").unwrap();
                } else {
                    let mem = fmt_uchiguri_mem(layout.ram_base, sl.ptr_addr);
                    writeln!(ctx.asm, "    ld {mem}, hl").unwrap();
                }
                if next_bb_id != Some(sl.cold_id) {
                    writeln!(ctx.asm, "    jp {fname}_bb{}", sl.cold_id).unwrap();
                }
            }
            continue;
        }

        // Detect countdown loop latch (`slot = slot - 1; slot != 0 -> branch then_bb / else_bb`)
        // where `slot` is allocated to `B`, `C`, or `BC` so we can lower directly to `djnz` / `dec r; jp nz` / `dec bc; ld a, b; or c; jp nz`.
        let countdown_branch = countdown_branches[bb_idx];

        // Detect fused Compare + Branch at the end of the basic block!
        let fused_cmp_branch = if countdown_branch.is_some() {
            None
        } else if let Terminator::Branch {
            cond,
            then_bb,
            else_bb,
        } = bb.term
            && let Some(IrInst::BinOp {
                dst,
                op,
                ty,
                lhs,
                rhs,
                ..
            }) = bb.insts.last()
            && *dst == cond
            && fused_cmp_dsts.contains(dst)
        {
            Some((*op, ty.clone(), *lhs, *rhs, then_bb, else_bb))
        } else {
            None
        };

        let inst_count = if let Some((skip_from, _, _, _, _)) = countdown_branch {
            skip_from
        } else if fused_cmp_branch.is_some() {
            bb.insts.len() - 1
        } else {
            bb.insts.len()
        };

        ctx.reset_bb(nonneg_ix_blocks.get(&bb.id).copied());
        ctx.cur_bb_is_tail_return_none = matches!(bb.term, Terminator::Return(None))
            && func.interrupt_source.is_none()
            && inst_count == bb.insts.len();
        ctx.cur_bb_inst_len = inst_count;

        for (inst_idx, inst) in bb.insts[..inst_count].iter().enumerate() {
            if ctx.fused_skip_insts.contains(&(bb_idx, inst_idx)) {
                continue;
            }
            ctx.emit_inst(bb_idx, inst_idx, inst)?;
        }

        if let Some(&si) = scan_by_preheader.get(&bb_idx) {
            let sl = &scan_loops[si];
            emit_scan_load_hl_and_de(ctx.asm, layout.ram_base, sl);
            if next_bb_id != Some(sl.header_id) {
                writeln!(ctx.asm, "    jp {fname}_bb{}", sl.header_id).unwrap();
            }
            continue;
        }

        if let Some((_, _, reg, then_bb, else_bb)) = countdown_branch {
            match reg {
                RegLoc::B if can_use_direct_djnz(program, func, &bb_order, order_pos, then_bb) => {
                    writeln!(ctx.asm, "    djnz {fname}_bb{then_bb}").unwrap();
                }
                RegLoc::B | RegLoc::C => {
                    let r_ch = if reg == RegLoc::B { 'b' } else { 'c' };
                    writeln!(ctx.asm, "    dec {r_ch}").unwrap();
                    writeln!(ctx.asm, "    jp nz, {fname}_bb{then_bb}").unwrap();
                }
                RegLoc::BC => {
                    writeln!(ctx.asm, "    dec bc").unwrap();
                    writeln!(ctx.asm, "    ld a, b").unwrap();
                    writeln!(ctx.asm, "    or c").unwrap();
                    writeln!(ctx.asm, "    jp nz, {fname}_bb{then_bb}").unwrap();
                }
            }
            if next_bb_id != Some(else_bb) {
                writeln!(ctx.asm, "    jp {fname}_bb{else_bb}").unwrap();
            }
            continue;
        }

        if let Some((op, ty, lhs, rhs, then_bb, else_bb)) = fused_cmp_branch {
            ctx.emit_fused_cmp_branch(&fname, op, &ty, lhs, rhs, (then_bb, else_bb, next_bb_id));
            continue;
        }
        match &bb.term {
            Terminator::Return(Some(v)) => {
                let ret_sz = program.type_size(&func.ret_ty);
                match ret_sz {
                    1 => ctx.emit_load_a_from(*v),
                    2 => ctx.emit_load_hl_from(*v),
                    4 => ctx.emit_load_hlde_from(*v),
                    _ => {
                        debug_assert!(false, "unexpected non-unit return size {ret_sz}");
                    }
                }
                if save_ix {
                    writeln!(ctx.asm, "    pop ix").unwrap();
                }
                writeln!(ctx.asm, "    ret").unwrap();
            }
            Terminator::Return(None) => {
                if let Some(src_name) = &func.interrupt_source {
                    // Emit interrupt controller rearm port writes, restore regs, EI + RETI
                    if let Some(target) = &program.target {
                        for ic in &target.interrupt_controllers {
                            if let Some(src) = ic.sources.iter().find(|s| &s.name == src_name) {
                                for op in &src.rearm {
                                    emit_port_write_op(op, ctx.asm);
                                }
                            }
                        }
                    }
                    writeln!(ctx.asm, "    pop ix").unwrap();
                    writeln!(ctx.asm, "    pop hl").unwrap();
                    writeln!(ctx.asm, "    pop de").unwrap();
                    writeln!(ctx.asm, "    pop bc").unwrap();
                    writeln!(ctx.asm, "    pop af").unwrap();
                    writeln!(ctx.asm, "    ei").unwrap();
                    writeln!(ctx.asm, "    reti").unwrap();
                } else if !ctx.tail_jumped_trampoline {
                    if save_ix {
                        writeln!(ctx.asm, "    pop ix").unwrap();
                    }
                    writeln!(ctx.asm, "    ret").unwrap();
                }
            }
            Terminator::Goto(target) => {
                if let Some(&bcl) = bcl_by_hdr.get(target) {
                    let src_mem = fmt_uchiguri_mem(
                        layout.ram_base,
                        layout.slot_addrs[&(func.id, bcl.src_slot)],
                    );
                    let dst_mem = fmt_uchiguri_mem(
                        layout.ram_base,
                        layout.slot_addrs[&(func.id, bcl.dst_slot)],
                    );
                    let cnt_mem = fmt_uchiguri_mem(
                        layout.ram_base,
                        layout.slot_addrs[&(func.id, bcl.cnt_slot)],
                    );
                    writeln!(ctx.asm, "    ld hl, {src_mem}").unwrap();
                    writeln!(ctx.asm, "    ld de, {dst_mem}").unwrap();
                    if bcl.is_u8_cnt {
                        writeln!(ctx.asm, "    ld a, {cnt_mem}").unwrap();
                        writeln!(ctx.asm, "    ld b, a").unwrap();
                    } else {
                        writeln!(ctx.asm, "    ld bc, {cnt_mem}").unwrap();
                    }
                }
                if let Some(&si) = scan_block_to_loop.get(&bb.id)
                    && *target == scan_loops[si].latch_id
                {
                    let is_cold_end = scan_by_cold_end.get(&bb_idx) == Some(&si);
                    if !is_cold_end {
                        writeln!(
                            ctx.asm,
                            "    jp __scan_reenter_{fname}_{}",
                            scan_loops[si].header_id
                        )
                        .unwrap();
                    }
                } else if next_bb_id != Some(*target) {
                    writeln!(ctx.asm, "    jp {fname}_bb{target}").unwrap();
                }
            }
            Terminator::Branch {
                cond,
                then_bb,
                else_bb,
            } => {
                ctx.emit_load_a_from(*cond);
                writeln!(ctx.asm, "    or a").unwrap();
                if next_bb_id == Some(*then_bb) {
                    writeln!(ctx.asm, "    jp z, {fname}_bb{else_bb}").unwrap();
                } else {
                    writeln!(ctx.asm, "    jp nz, {fname}_bb{then_bb}").unwrap();
                    if next_bb_id != Some(*else_bb) {
                        writeln!(ctx.asm, "    jp {fname}_bb{else_bb}").unwrap();
                    }
                }
            }
            Terminator::Unreachable => {
                writeln!(ctx.asm, "    halt").unwrap();
            }
        }
        if let Some(&si) = scan_by_cold_end.get(&bb_idx) {
            let sl = &scan_loops[si];
            writeln!(ctx.asm, "__scan_reenter_{fname}_{}:", sl.header_id).unwrap();
            emit_scan_load_hl_and_de(ctx.asm, layout.ram_base, sl);
            writeln!(ctx.asm, "    jp {fname}_bb{}", sl.latch_id).unwrap();
        }
    }
    if !ctx.tail_cold_stubs.is_empty() {
        ctx.asm.push_str(&ctx.tail_cold_stubs);
    }
    for (spill_lbl, dst_slot, exit_s) in bcl_spill_tramps {
        writeln!(ctx.asm, "{spill_lbl}:").unwrap();
        let dst_mem = fmt_uchiguri_mem(layout.ram_base, layout.slot_addrs[&(func.id, dst_slot)]);
        writeln!(ctx.asm, "    ld {dst_mem}, de").unwrap();
        writeln!(ctx.asm, "    jp {fname}_bb{exit_s}").unwrap();
    }
    Ok(())
}

struct ScanLoopInfo {
    preheader_bi: usize,
    header_bi: usize,
    header_id: BlockId,
    latch_bi: usize,
    latch_id: BlockId,
    exit_id: BlockId,
    cold_id: BlockId,
    cold_region: BTreeSet<BlockId>,
    ptr_addr: u16,
    ptr_is_ix: bool,
    countdown_reg: RegLoc,
    field_off: u16,
    stride: i16,
    cmp_imm: Option<u8>,
    jp_cc_to_cold: &'static str,
    needs_cold_save: bool,
    reenters_latch: bool,
}

fn emit_scan_hl_sub_field_off(asm: &mut String, field_off: u16) {
    if field_off == 0 {
        return;
    }
    if field_off <= 4 {
        for _ in 0..field_off {
            writeln!(asm, "    dec hl").unwrap();
        }
    } else {
        writeln!(asm, "    or a").unwrap();
        writeln!(asm, "    ld de, 0x{field_off:04X}").unwrap();
        writeln!(asm, "    sbc hl, de").unwrap();
    }
}

fn emit_scan_load_hl_and_de(asm: &mut String, ram_base: u16, sl: &ScanLoopInfo) {
    if sl.ptr_is_ix {
        writeln!(asm, "    push ix").unwrap();
        writeln!(asm, "    pop hl").unwrap();
    } else {
        let mem = fmt_uchiguri_mem(ram_base, sl.ptr_addr);
        writeln!(asm, "    ld hl, {mem}").unwrap();
    }
    if sl.field_off > 0 && sl.field_off <= 3 {
        for _ in 0..sl.field_off {
            writeln!(asm, "    inc hl").unwrap();
        }
    } else if sl.field_off > 3 {
        writeln!(asm, "    ld de, 0x{:04X}", sl.field_off).unwrap();
        writeln!(asm, "    add hl, de").unwrap();
    }
    writeln!(asm, "    ld de, 0x{:04X}", sl.stride as u16).unwrap();
}

fn detect_scan_loops(
    program: &IrProgram,
    func: &IrFunction,
    fa: &crate::analysis::FnAnalysis,
    slot_addrs: &HashMap<(u32, SlotId), u16>,
    countdown_branches: &[Option<(usize, SlotId, RegLoc, BlockId, BlockId)>],
    fn_consts: &HashMap<ValueId, i64>,
) -> Vec<ScanLoopInfo> {
    let id_to_idx: HashMap<BlockId, usize> =
        func.blocks.iter().enumerate().map(|(i, b)| (b.id, i)).collect();
    let escaped = func.escaped_locals();
    let mut preds: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
    let mut succs: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
    for bb in &func.blocks {
        preds.entry(bb.id).or_default();
        let sv = succs.entry(bb.id).or_default();
        match bb.term {
            Terminator::Goto(t) => {
                sv.push(t);
                preds.entry(t).or_default().push(bb.id);
            }
            Terminator::Branch {
                then_bb, else_bb, ..
            } => {
                sv.push(then_bb);
                preds.entry(then_bb).or_default().push(bb.id);
                if else_bb != then_bb {
                    sv.push(else_bb);
                    preds.entry(else_bb).or_default().push(bb.id);
                }
            }
            _ => {}
        }
    }

    let mut result: Vec<ScanLoopInfo> = Vec::new();
    let mut used_blocks: HashSet<BlockId> = HashSet::new();

    for (latch_bi, latch_bb) in func.blocks.iter().enumerate() {
        let Some((skip_from, counter_slot, countdown_reg @ (RegLoc::B | RegLoc::C), header_id, exit_id)) =
            countdown_branches[latch_bi]
        else {
            continue;
        };
        let latch_id = latch_bb.id;
        if header_id == latch_id || exit_id == header_id || exit_id == latch_id {
            continue;
        }
        let Some(&header_bi) = id_to_idx.get(&header_id) else {
            continue;
        };
        let Some(h_preds) = preds.get(&header_id) else {
            continue;
        };
        if h_preds.len() != 2 || !h_preds.contains(&latch_id) {
            continue;
        }
        let preheader_id = if h_preds[0] == latch_id {
            h_preds[1]
        } else {
            h_preds[0]
        };
        if preheader_id == header_id || preheader_id == latch_id {
            continue;
        }
        let Some(&preheader_bi) = id_to_idx.get(&preheader_id) else {
            continue;
        };
        if !matches!(func.blocks[preheader_bi].term, Terminator::Goto(t) if t == header_id) {
            continue;
        }

        // Match `latch_bb.insts[..skip_from]` as a single pointer step `ptr = ptr + stride`.
        let mut latch_non_const: Vec<usize> = Vec::new();
        let mut latch_const_ok = true;
        for (ii, inst) in latch_bb.insts[..skip_from].iter().enumerate() {
            if let IrInst::ConstInt { dst, .. } = inst {
                for (obi, obb) in func.blocks.iter().enumerate() {
                    if obi != latch_bi
                        && (obb.insts.iter().any(|i| i.uses_val(*dst))
                            || matches!(
                                obb.term,
                                Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } if v == *dst
                            ))
                    {
                        latch_const_ok = false;
                        break;
                    }
                }
            } else {
                latch_non_const.push(ii);
            }
        }
        if !latch_const_ok || latch_non_const.len() != 3 {
            continue;
        }
        let IrInst::LoadLocal {
            dst: lp0,
            slot: ptr_slot,
            ..
        } = latch_bb.insts[latch_non_const[0]]
        else {
            continue;
        };
        let IrInst::BinOp {
            dst: lp1,
            op: IrBinOp::Add,
            ty: step_ty,
            lhs: step_lhs,
            rhs: step_rhs,
            ..
        } = &latch_bb.insts[latch_non_const[1]]
        else {
            continue;
        };
        let IrInst::StoreLocal {
            slot: store_slot,
            val: store_val,
            ..
        } = latch_bb.insts[latch_non_const[2]]
        else {
            continue;
        };
        if store_slot != ptr_slot
            || store_val != *lp1
            || program.type_size(step_ty) != 2
            || count_val_uses(func, lp0) != 1
            || count_val_uses(func, *lp1) != 1
        {
            continue;
        }
        let stride_opt = if *step_lhs == lp0 {
            fn_consts.get(step_rhs).copied()
        } else if *step_rhs == lp0 {
            fn_consts.get(step_lhs).copied()
        } else {
            None
        };
        let Some(stride_i64) = stride_opt else {
            continue;
        };
        if !(-32768..=32767).contains(&stride_i64) || stride_i64 == 0 {
            continue;
        }
        let stride = stride_i64 as i16;

        let ps_idx = ptr_slot as usize;
        if (ps_idx < escaped.len() && escaped[ps_idx])
            || fa.reg_alloc.slot_reg(ptr_slot).is_some()
            || fa.ix_param_slot == Some(ptr_slot)
        {
            continue;
        }
        let ptr_is_ix = fa.ix_local_slot == Some(ptr_slot);
        let ptr_addr = if ptr_is_ix {
            0
        } else {
            let Some(&addr) = slot_addrs.get(&(func.id, ptr_slot)) else {
                continue;
            };
            addr
        };

        // Match `header_bi` as a single 1-byte field guard jumping either to `latch_id` or `cold_id`.
        let header_bb = &func.blocks[header_bi];
        if header_bb.insts.iter().any(|i| matches!(i, IrInst::Phi { .. })) {
            continue;
        }
        let Terminator::Branch {
            cond: h_cond,
            then_bb: h_then,
            else_bb: h_else,
        } = header_bb.term
        else {
            continue;
        };
        let (skip_on_true, cold_id) = if h_then == latch_id && h_else != latch_id {
            (true, h_else)
        } else if h_else == latch_id && h_then != latch_id {
            (false, h_then)
        } else {
            continue;
        };
        if cold_id == header_id || cold_id == preheader_id || cold_id == exit_id {
            continue;
        }

        // Verify all header values are used strictly inside `header_bi` and match the 1-byte guard.
        let mut hdr_vals_ok = true;
        for inst in &header_bb.insts {
            if let Some(d) = inst.dst() {
                for (obi, obb) in func.blocks.iter().enumerate() {
                    if obi != header_bi
                        && (obb.insts.iter().any(|i| i.uses_val(d))
                            || matches!(
                                obb.term,
                                Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } if v == d
                            ))
                    {
                        hdr_vals_ok = false;
                        break;
                    }
                }
            }
            if !hdr_vals_ok {
                break;
            }
        }
        if !hdr_vals_ok {
            continue;
        }

        // Parse `header_bb.insts` strictly.
        let mut h_non_const: Vec<usize> = Vec::new();
        let mut h_consts: HashMap<ValueId, i64> = HashMap::new();
        for (ii, inst) in header_bb.insts.iter().enumerate() {
            match inst {
                IrInst::ConstInt { dst, val, .. } => {
                    h_consts.insert(*dst, *val);
                }
                IrInst::ConstBool { dst, val } => {
                    h_consts.insert(*dst, if *val { 1 } else { 0 });
                }
                _ => h_non_const.push(ii),
            }
        }
        if !matches!(h_non_const.len(), 2 | 3 | 4) {
            continue;
        }

        let IrInst::LoadLocal {
            dst: hv_ptr,
            slot: h_ptr_slot,
            ..
        } = header_bb.insts[h_non_const[0]]
        else {
            continue;
        };
        if h_ptr_slot != ptr_slot {
            continue;
        }
        let (expected_load_ptr, field_off, load_idx) = if matches!(
            h_non_const.get(1).map(|&i| &header_bb.insts[i]),
            Some(IrInst::FieldAddr { .. })
        ) {
            let IrInst::FieldAddr {
                dst: hv_fld,
                base,
                byte_offset,
                ..
            } = header_bb.insts[h_non_const[1]]
            else {
                unreachable!()
            };
            let off_u16 = u16::try_from(byte_offset).unwrap();
            if base != hv_ptr {
                continue;
            }
            (hv_fld, off_u16, 2usize)
        } else {
            (hv_ptr, 0u16, 1usize)
        };
        let Some(&load_inst_ii) = h_non_const.get(load_idx) else {
            continue;
        };
        let IrInst::LoadPtr {
            dst: hv_byte,
            addr: load_ptr,
            ty: byte_ty,
            volatile: false,
        } = &header_bb.insts[load_inst_ii]
        else {
            continue;
        };
        if *load_ptr != expected_load_ptr
            || program.type_size(byte_ty) != 1
            || !matches!(byte_ty, IrType::Int(_) | IrType::Bool)
        {
            continue;
        }
        let (cmp_imm, jp_cc_to_cold) = if let Some(&cmp_ii) = h_non_const.get(load_idx + 1) {
            if cmp_ii != *h_non_const.last().unwrap() {
                continue;
            }
            let IrInst::BinOp {
                dst: hv_cmp,
                op: cmp_op @ (IrBinOp::Eq | IrBinOp::Ne),
                ty: cmp_ty,
                lhs: cmp_lhs,
                rhs: cmp_rhs,
                ..
            } = &header_bb.insts[cmp_ii]
            else {
                continue;
            };
            if *hv_cmp != h_cond || program.type_size(cmp_ty) != 1 {
                continue;
            }
            let imm_opt = if *cmp_lhs == *hv_byte {
                h_consts.get(cmp_rhs).or_else(|| fn_consts.get(cmp_rhs)).copied()
            } else if *cmp_rhs == *hv_byte {
                h_consts.get(cmp_lhs).or_else(|| fn_consts.get(cmp_lhs)).copied()
            } else {
                None
            };
            let Some(imm_val) = imm_opt else {
                continue;
            };
            let imm_u8 = (imm_val & 0xFF) as u8;
            let jp_cc = match (cmp_op, skip_on_true) {
                (IrBinOp::Eq, true) | (IrBinOp::Ne, false) => "nz",
                (IrBinOp::Eq, false) | (IrBinOp::Ne, true) => "z",
                _ => unreachable!(),
            };
            (Some(imm_u8), jp_cc)
        } else {
            if *hv_byte != h_cond || *byte_ty != IrType::Bool {
                continue;
            }
            let jp_cc = if skip_on_true { "z" } else { "nz" };
            (None, jp_cc)
        };

        // Collect dominated `cold_region` starting at `cold_id` (all preds inside `{header_id} U cold_region`).
        if preds.get(&cold_id).is_none_or(|ps| ps.as_slice() != [header_id]) {
            continue;
        }
        let mut cold_region: BTreeSet<BlockId> = BTreeSet::new();
        cold_region.insert(cold_id);
        loop {
            let mut added = false;
            let cur: Vec<BlockId> = cold_region.iter().copied().collect();
            for b in cur {
                if let Some(sv) = succs.get(&b) {
                    for &s in sv {
                        if s == latch_id || s == exit_id || s == header_id || s == preheader_id || cold_region.contains(&s) {
                            continue;
                        }
                        if preds.get(&s).is_some_and(|ps| {
                            !ps.is_empty() && ps.iter().all(|p| *p == header_id || cold_region.contains(p))
                        }) {
                            cold_region.insert(s);
                            added = true;
                        }
                    }
                }
            }
            if !added {
                break;
            }
        }

        // Verify `cold_region` invariants:
        // - no direct edge back to `header_id` or `preheader_id`
        // - no `Branch` targeting `latch_id` (edges to `latch_id` must be `Terminator::Goto(latch_id)`)
        // - no `Phi`s inside `cold_region`
        // - no `StoreLocal` to `counter_slot` or `ptr_slot` inside `cold_region`
        let mut cold_ok = true;
        let mut reenters_latch = false;
        let mut cold_reads_ptr = false;
        for &cid in &cold_region {
            let Some(&cbi) = id_to_idx.get(&cid) else {
                cold_ok = false;
                break;
            };
            let cbb = &func.blocks[cbi];
            match cbb.term {
                Terminator::Goto(t) => {
                    if t == header_id || t == preheader_id {
                        cold_ok = false;
                        break;
                    }
                    if t == latch_id {
                        reenters_latch = true;
                    }
                }
                Terminator::Branch {
                    then_bb, else_bb, ..
                } => {
                    if then_bb == header_id
                        || else_bb == header_id
                        || then_bb == preheader_id
                        || else_bb == preheader_id
                        || then_bb == latch_id
                        || else_bb == latch_id
                    {
                        cold_ok = false;
                        break;
                    }
                }
                _ => {}
            }
            for inst in &cbb.insts {
                match inst {
                    IrInst::Phi { .. } => {
                        cold_ok = false;
                        break;
                    }
                    IrInst::StoreLocal { slot, .. }
                        if *slot == counter_slot || *slot == ptr_slot =>
                    {
                        cold_ok = false;
                        break;
                    }
                    IrInst::LoadLocal { slot, .. } if *slot == ptr_slot => {
                        cold_reads_ptr = true;
                    }
                    _ => {}
                }
            }
            if !cold_ok {
                break;
            }
        }
        if !cold_ok {
            continue;
        }

        // Verify `ptr_slot` is not accessed outside `{preheader_id, header_id, latch_id} U cold_region`
        // (and not read in `preheader_id`).
        let mut ptr_outside_ok = true;
        for bb in &func.blocks {
            if bb.id == header_id || bb.id == latch_id || cold_region.contains(&bb.id) {
                continue;
            }
            for inst in &bb.insts {
                match inst {
                    IrInst::LoadLocal { slot, .. } | IrInst::AddrLocal { slot, .. }
                        if *slot == ptr_slot =>
                    {
                        ptr_outside_ok = false;
                        break;
                    }
                    IrInst::StoreLocal { slot, .. }
                        if *slot == ptr_slot && bb.id != preheader_id =>
                    {
                        ptr_outside_ok = false;
                        break;
                    }
                    _ => {}
                }
            }
            if !ptr_outside_ok {
                break;
            }
        }
        if !ptr_outside_ok {
            continue;
        }

        // Ensure disjointness with any previously matched scan loop.
        let mut touches = vec![preheader_id, header_id, latch_id];
        touches.extend(cold_region.iter().copied());
        if touches.iter().any(|b| used_blocks.contains(b)) {
            continue;
        }
        used_blocks.extend(touches);

        result.push(ScanLoopInfo {
            preheader_bi,
            header_bi,
            header_id,
            latch_bi,
            latch_id,
            exit_id,
            cold_id,
            cold_region,
            ptr_addr,
            ptr_is_ix,
            countdown_reg,
            field_off,
            stride,
            cmp_imm,
            jp_cc_to_cold,
            needs_cold_save: ptr_is_ix || cold_reads_ptr || reenters_latch,
            reenters_latch,
        });
    }
    result
}

/// Computes a dominance-preserving greedy fallthrough basic-block emission order:
/// uses RPO to identify back-edges, then emits blocks in topological DFS trace order
/// preferring `else_bb` / `Goto` fallthrough to eliminate redundant unconditional jumps.
fn compute_bb_order(func: &IrFunction, scan_loops: &[ScanLoopInfo]) -> Vec<usize> {
    let n = func.blocks.len();
    if n <= 2 {
        return (0..n).collect();
    }
    let id_to_idx: HashMap<BlockId, usize> = func
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id, i))
        .collect();
    let succs = |u: usize| -> Vec<usize> {
        let mut s = Vec::new();
        match func.blocks[u].term {
            Terminator::Goto(t) => {
                if let Some(&idx) = id_to_idx.get(&t) {
                    s.push(idx);
                }
            }
            Terminator::Branch {
                then_bb, else_bb, ..
            } => {
                if let Some(&idx) = id_to_idx.get(&then_bb) {
                    s.push(idx);
                }
                if let Some(&idx) = id_to_idx.get(&else_bb)
                    && !s.contains(&idx)
                {
                    s.push(idx);
                }
            }
            _ => {}
        }
        s
    };

    // 1. Compute RPO from entry block 0
    let mut visited = vec![false; n];
    let mut postorder = Vec::with_capacity(n);
    fn dfs_po(
        u: usize,
        succs: &impl Fn(usize) -> Vec<usize>,
        visited: &mut [bool],
        po: &mut Vec<usize>,
    ) {
        visited[u] = true;
        for v in succs(u) {
            if !visited[v] {
                dfs_po(v, succs, visited, po);
            }
        }
        po.push(u);
    }
    dfs_po(0, &succs, &mut visited, &mut postorder);
    postorder.reverse();
    let mut rpo_pos = vec![usize::MAX; n];
    for (pos, &u) in postorder.iter().enumerate() {
        rpo_pos[u] = pos;
    }

    // 2. Count forward (non-backedge) in-degree for reachable blocks
    let mut in_deg = vec![0usize; n];
    let mut fwd_succs: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &u in &postorder {
        for v in succs(u) {
            if rpo_pos[u] < rpo_pos[v] {
                fwd_succs[u].push(v);
                in_deg[v] += 1;
            }
        }
    }

    // 3. Greedy stack-based topological trace: push `then_bb` first, `else_bb` last so
    //    `else_bb` is popped immediately for zero-jump fallthrough.
    //    For matched scan loops (`header -> latch -> cold_region -> exit`), emit `latch`
    //    immediately after `header` so `header` falls straight into `latch` and `djnz`
    //    only spans those two tiny blocks before `cold_region`.
    let scan_by_hdr: HashMap<usize, &ScanLoopInfo> = scan_loops
        .iter()
        .map(|sl| (sl.header_bi, sl))
        .collect();
    let mut order = Vec::with_capacity(n);
    let mut emitted = vec![false; n];
    let mut ready: Vec<usize> = vec![0];
    while let Some(u) = ready.pop() {
        if emitted[u] {
            continue;
        }
        emitted[u] = true;
        order.push(u);
        if let Some(sl) = scan_by_hdr.get(&u)
            && !emitted[sl.latch_bi]
        {
            let latch_bi = sl.latch_bi;
            emitted[latch_bi] = true;
            order.push(latch_bi);
            // Push any newly ready forward succs from `latch` (i.e. `exit`) first,
            // then forward succs from `header` (`cold_id`) on top so `cold_region` emits
            // right between `latch` and `exit`.
            for &v in &fwd_succs[latch_bi] {
                in_deg[v] = in_deg[v].saturating_sub(1);
                if in_deg[v] == 0 && !emitted[v] {
                    ready.push(v);
                }
            }
            for &v in &fwd_succs[u] {
                in_deg[v] = in_deg[v].saturating_sub(1);
                if in_deg[v] == 0 && !emitted[v] {
                    ready.push(v);
                }
            }
            continue;
        }
        for &v in &fwd_succs[u] {
            in_deg[v] = in_deg[v].saturating_sub(1);
            if in_deg[v] == 0 && !emitted[v] {
                ready.push(v);
            }
        }
    }
    // Append any remaining blocks (unreachable or irreducible cycles) in original order
    for i in 0..n {
        if !emitted[i] {
            order.push(i);
        }
    }
    order
}

pub(crate) fn detect_countdown_latch(
    program: &IrProgram,
    func: &IrFunction,
    bb: &BasicBlock,
    fn_consts: &HashMap<ValueId, i64>,
) -> Option<(usize, SlotId, BlockId, BlockId)> {
    let Terminator::Branch {
        cond,
        then_bb,
        else_bb,
    } = bb.term
    else {
        return None;
    };

    // Walk backwards from end of `bb.insts`, ignoring ConstInt definitions,
    // expecting: [LoadLocal(slot) -> v_cur, BinOp(Sub, u8, v_cur, 1) -> v_dec,
    //             StoreLocal(slot, v_dec), BinOp(Ne, u8, v_dec, 0) -> cond].
    let mut non_const: Vec<usize> = Vec::with_capacity(4);
    for idx in (0..bb.insts.len()).rev() {
        if !matches!(bb.insts[idx], IrInst::ConstInt { .. }) {
            non_const.push(idx);
            if non_const.len() == 4 {
                break;
            }
        }
    }
    if non_const.len() < 4 {
        return None;
    }
    let idx_ne = non_const[0];
    let idx_store = non_const[1];
    let idx_sub = non_const[2];
    let idx_load = non_const[3];

    let IrInst::BinOp {
        dst: ne_dst,
        op: IrBinOp::Ne,
        ty: ne_ty,
        lhs: ne_lhs,
        rhs: ne_rhs,
        ..
    } = &bb.insts[idx_ne]
    else {
        return None;
    };
    let ne_sz = program.type_size(ne_ty);
    if *ne_dst != cond || !matches!(ne_sz, 1 | 2) || *ne_ty == IrType::Bool {
        return None;
    }
    let IrInst::StoreLocal {
        slot,
        val: store_val,
        ..
    } = &bb.insts[idx_store]
    else {
        return None;
    };
    let IrInst::BinOp {
        dst: sub_dst,
        op: IrBinOp::Sub,
        ty: sub_ty,
        lhs: sub_lhs,
        rhs: sub_rhs,
        ..
    } = &bb.insts[idx_sub]
    else {
        return None;
    };
    if program.type_size(sub_ty) != ne_sz || *sub_ty == IrType::Bool || *store_val != *sub_dst {
        return None;
    }
    let IrInst::LoadLocal {
        dst: load_dst,
        slot: load_slot,
        ..
    } = &bb.insts[idx_load]
    else {
        return None;
    };
    if *load_slot != *slot || *sub_lhs != *load_dst || fn_consts.get(sub_rhs) != Some(&1) {
        return None;
    }
    let ne_ok = (*ne_lhs == *sub_dst && fn_consts.get(ne_rhs) == Some(&0))
        || (*ne_rhs == *sub_dst && fn_consts.get(ne_lhs) == Some(&0));
    if !ne_ok {
        return None;
    }
    if count_val_uses(func, *load_dst) != 1
        || count_val_uses(func, *sub_dst) != 2
        || count_val_uses(func, *ne_dst) != 1
    {
        return None;
    }
    // Skip any trailing ConstInts immediately before `idx_load` that were only used by this latch.
    let mut skip_from = idx_load;
    for ii in idx_load..bb.insts.len() {
        if let IrInst::ConstInt { dst, .. } = bb.insts[ii]
            && count_val_uses(func, dst) > 1
        {
            return None;
        }
    }
    while skip_from > 0 {
        if let IrInst::ConstInt { dst, .. } = bb.insts[skip_from - 1]
            && count_val_uses(func, dst) == 1
            && (dst == *sub_rhs || dst == *ne_rhs)
        {
            skip_from -= 1;
        } else {
            break;
        }
    }
    Some((skip_from, *slot, then_bb, else_bb))
}

fn can_use_direct_djnz(
    program: &IrProgram,
    func: &IrFunction,
    bb_order: &[usize],
    cur_pos: usize,
    then_bb: BlockId,
) -> bool {
    let Some(then_pos) = bb_order[..=cur_pos]
        .iter()
        .position(|&bi| func.blocks[bi].id == then_bb)
    else {
        return false;
    };
    let mut total_insts = 0usize;
    for &bi in &bb_order[then_pos..=cur_pos] {
        let b = &func.blocks[bi];
        total_insts += b.insts.len() + 1;
        if total_insts > 16 {
            return false;
        }
        for inst in &b.insts {
            match inst {
                IrInst::Call { .. }
                | IrInst::CallIndirect { .. }
                | IrInst::ExternCall { .. }
                | IrInst::MemCopy { .. }
                | IrInst::MemFill { .. }
                | IrInst::MemSet { .. }
                | IrInst::PoolAlloc { .. }
                | IrInst::PoolFree { .. }
                | IrInst::QueuePush { .. }
                | IrInst::QueuePop { .. }
                | IrInst::QueueIsEmpty { .. }
                | IrInst::QueueLen { .. }
                | IrInst::StateDispatch { .. }
                | IrInst::StateTransition { .. }
                | IrInst::InlineAsm { .. } => return false,
                IrInst::BinOp {
                    op: IrBinOp::Mul | IrBinOp::Div | IrBinOp::Rem,
                    ty,
                    ..
                } if program.type_size(ty) >= 4 => return false,
                _ => {}
            }
        }
    }
    true
}
