use super::runtime::emit_port_write_op;
use super::{EmitCtx, EmitGlobalEnv, RuntimeNeeds, fmt_uchiguri_mem};
use crate::analysis::{count_val_uses, is_signed_ty, z80_sym};
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

    // Pre-collect same-block fused `FixedInt(2, 6)` -> `Cast(2 -> 1, non-Bool)` (`add hl, hl; add hl, hl; ld a, h`).
    let mut fused_q10_6_u8: HashMap<ValueId, ValueId> = HashMap::new();
    for (bi, bb) in func.blocks.iter().enumerate() {
        let mut blk_fx6: HashMap<ValueId, (usize, ValueId)> = HashMap::new();
        for (ii, inst) in bb.insts.iter().enumerate() {
            if fused_skip_insts.contains(&(bi, ii)) {
                continue;
            }
            match inst {
                IrInst::FixedInt {
                    dst,
                    val,
                    base,
                    frac_bits: 6,
                } if base.byte_size() == 2 && count_val_uses(func, *dst) == 1 => {
                    blk_fx6.insert(*dst, (ii, *val));
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
                    // Require immediately adjacent `FixedInt` -> `Cast` so `src_v`'s colored
                    // scratch slot cannot be overwritten by an intervening instruction.
                    if let Some(&(fx_ii, src_v)) = blk_fx6.get(val)
                        && ii == fx_ii + 1
                    {
                        fused_skip_insts.insert((bi, fx_ii));
                        fused_q10_6_u8.insert(*dst, src_v);
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
                | IrInst::InlineAsm { .. } => {
                    blk_fx6.clear();
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

    let mut ctx = EmitCtx {
        program,
        analysis,
        func,
        fa,
        layout,
        scratch_base,
        val_offsets,
        opts: env.opts,
        shared_muls: env.shared_muls,
        fused_skip_insts: &fused_skip_insts,
        fused_xor_shift: &fused_xor_shift,
        fused_q10_6_u8: &fused_q10_6_u8,
        block_nonneg_ix: None,
        nonneg_vals: BTreeSet::new(),
        const_vals: HashMap::new(),
        const_addrs: HashMap::new(),
        rom_addrs: HashMap::new(),
        val_aliases: HashMap::new(),
        ix_addrs: HashMap::new(),
        ix_param_addr_vals: BTreeSet::new(),
        ix_val_loads: HashMap::new(),
        val_in_reg: HashMap::new(),
        shift_cse: HashMap::new(),
        index_cse: HashMap::new(),
        label_counter: 0,
        runtime_needs,
        asm,
    };

    let bb_order = compute_bb_order(func);
    for (order_pos, &bb_idx) in bb_order.iter().enumerate() {
        let bb = &func.blocks[bb_idx];
        writeln!(ctx.asm, "{fname}_bb{}:", bb.id).unwrap();
        let next_bb_id = bb_order.get(order_pos + 1).map(|&idx| func.blocks[idx].id);

        // Detect fused Compare + Branch at the end of the basic block!
        let fused_cmp_branch = if let Terminator::Branch {
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

        let inst_count = if fused_cmp_branch.is_some() {
            bb.insts.len() - 1
        } else {
            bb.insts.len()
        };

        ctx.reset_bb(nonneg_ix_blocks.get(&bb.id).copied());

        for (inst_idx, inst) in bb.insts[..inst_count].iter().enumerate() {
            if ctx.fused_skip_insts.contains(&(bb_idx, inst_idx)) {
                continue;
            }
            ctx.emit_inst(bb_idx, inst_idx, inst)?;
        }

        if let Some((op, ty, lhs, rhs, then_bb, else_bb)) = fused_cmp_branch {
            ctx.emit_fused_cmp_branch(&fname, op, &ty, lhs, rhs, (then_bb, else_bb, next_bb_id));
            continue;
        }
        match &bb.term {
            Terminator::Return(Some(v)) => {
                match program.type_size(&func.ret_ty) {
                    1 => ctx.emit_load_a_from(*v),
                    2 => ctx.emit_load_hl_from(*v),
                    4 => ctx.emit_load_hlde_from(*v),
                    _ => {}
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
                } else {
                    if save_ix {
                        writeln!(ctx.asm, "    pop ix").unwrap();
                    }
                    writeln!(ctx.asm, "    ret").unwrap();
                }
            }
            Terminator::Goto(target) => {
                if next_bb_id != Some(*target) {
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
    }
    Ok(())
}

/// Computes a dominance-preserving greedy fallthrough basic-block emission order:
/// uses RPO to identify back-edges, then emits blocks in topological DFS trace order
/// preferring `else_bb` / `Goto` fallthrough to eliminate redundant unconditional jumps.
fn compute_bb_order(func: &IrFunction) -> Vec<usize> {
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
    let mut order = Vec::with_capacity(n);
    let mut emitted = vec![false; n];
    let mut ready: Vec<usize> = vec![0];
    while let Some(u) = ready.pop() {
        if emitted[u] {
            continue;
        }
        emitted[u] = true;
        order.push(u);
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
