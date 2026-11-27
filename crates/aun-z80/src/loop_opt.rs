//! Z80-specific MIR loop and arithmetic pre-lowering passes (`aun-z80`).
//!
//! Runs after target-independent `aun-ir` optimization and before `ProgramAnalysis` /
//! Uchiguri static RAM allocation so transformations keep exact IR semantics and scratch liveness.

use aun_ir::cfg::Cfg;
use aun_ir::*;
use std::collections::{BTreeSet, HashMap, HashSet};

/// Entry point for all Z80-specific pre-codegen MIR optimizations.
pub fn optimize_z80_program(program: &mut IrProgram, is_release: bool) {
    for func in &mut program.functions {
        if func.blocks.is_empty() {
            continue;
        }
        reduce_loop_index_addrs(func, is_release);
        rotate_counted_loops(func);
        convert_trip_counters_to_countdown(func);
        colocate_div_rem(func);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum BaseKey {
    Val(ValueId),
    Global(u32),
}

fn find_loop_iv_init(
    func: &IrFunction,
    cfg: &Cfg,
    preheader_id: BlockId,
    iv_slot: SlotId,
    consts: &HashMap<ValueId, i64>,
) -> Option<i64> {
    let bb_idx_map: HashMap<BlockId, usize> =
        func.blocks.iter().enumerate().map(|(i, b)| (b.id, i)).collect();
    let mut cur = preheader_id;
    for _ in 0..3 {
        let idx = *bb_idx_map.get(&cur)?;
        for inst in func.blocks[idx].insts.iter().rev() {
            if let IrInst::StoreLocal { slot, val, .. } = inst
                && *slot == iv_slot
            {
                return consts.get(val).copied();
            }
        }
        let preds = cfg.preds(cur);
        if preds.len() == 1 {
            cur = preds[0];
        } else {
            break;
        }
    }
    None
}

fn find_loop_iv_upper_bound(
    func: &IrFunction,
    header_id: BlockId,
    loop_body: &BTreeSet<BlockId>,
    iv_slot: SlotId,
    consts: &HashMap<ValueId, i64>,
) -> Option<i64> {
    let hdr = func.blocks.iter().find(|b| b.id == header_id)?;
    let Terminator::Branch {
        cond,
        then_bb,
        else_bb,
    } = hdr.term
    else {
        return None;
    };
    if !(loop_body.contains(&then_bb) && !loop_body.contains(&else_bb)) {
        return None;
    }
    let mut local_loads: HashMap<ValueId, SlotId> = HashMap::new();
    for inst in &hdr.insts {
        match inst {
            IrInst::LoadLocal { dst, slot, .. } => {
                local_loads.insert(*dst, *slot);
            }
            IrInst::BinOp {
                dst,
                op: op @ (IrBinOp::Lt | IrBinOp::Le),
                ty,
                lhs,
                rhs,
                ..
            } if *dst == cond
                && matches!(ty, IrType::Int(IntType::U8 | IntType::U16))
                && local_loads.get(lhs) == Some(&iv_slot) =>
            {
                let bound = consts.get(rhs).copied()?;
                return Some(if *op == IrBinOp::Lt {
                    bound
                } else {
                    bound.checked_add(1)?
                });
            }
            _ => {}
        }
    }
    None
}

fn remove_dead_pure_insts(func: &mut IrFunction) {
    loop {
        let mut uses: HashMap<ValueId, usize> = HashMap::new();
        for bb in &func.blocks {
            for inst in &bb.insts {
                inst.for_each_use(|v| *uses.entry(v).or_insert(0) += 1);
            }
            match bb.term {
                Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } => {
                    *uses.entry(v).or_insert(0) += 1;
                }
                _ => {}
            }
        }
        let mut changed = false;
        for bb in &mut func.blocks {
            let before = bb.insts.len();
            bb.insts.retain(|inst| {
                let is_pure = matches!(
                    inst,
                    IrInst::ConstInt { .. }
                        | IrInst::ConstBool { .. }
                        | IrInst::LoadLocal { .. }
                        | IrInst::AddrLocal { .. }
                        | IrInst::AddrGlobal { .. }
                        | IrInst::Cast { .. }
                );
                if is_pure && let Some(d) = inst.dst() {
                    return uses.get(&d).copied().unwrap_or(0) > 0;
                }
                true
            });
            if bb.insts.len() != before {
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

/// Strength-reduces loop `IndexAddr` instructions indexed by a constant-step induction
/// variable (`base + iv * elem_size`) into an incrementally stepped 16-bit pointer local.
pub fn reduce_loop_index_addrs(func: &mut IrFunction, is_release: bool) -> bool {
    if func.blocks.is_empty() {
        return false;
    }

    let mut any_changed = false;
    let mut processed_headers: HashSet<BlockId> = HashSet::new();
    const MAX_OSR_PTRS_PER_LOOP: usize = 3;

    loop {
        let cfg = Cfg::build(func);
        let Some(entry_id) = cfg.entry_block() else {
            break;
        };

        let mut by_header: HashMap<BlockId, (Vec<BlockId>, BTreeSet<BlockId>)> = HashMap::new();
        for lp in cfg.natural_loops() {
            let entry = by_header
                .entry(lp.header)
                .or_insert_with(|| (Vec::new(), BTreeSet::new()));
            entry.0.push(lp.latch);
            entry.1.extend(&lp.body);
        }
        let mut loops: Vec<(BlockId, BlockId, BTreeSet<BlockId>)> = by_header
            .into_iter()
            .filter_map(|(h, (latches, body))| {
                (h != entry_id && latches.len() == 1 && !processed_headers.contains(&h))
                    .then_some((h, latches[0], body))
            })
            .collect();
        loops.sort_by_key(|(h, _, body)| (body.len(), *h));

        let Some((header, latch, body)) = loops.into_iter().next() else {
            break;
        };
        processed_headers.insert(header);

        let outside_preds: Vec<BlockId> = cfg
            .preds(header)
            .iter()
            .copied()
            .filter(|p| !body.contains(p))
            .collect();
        if outside_preds.len() != 1 {
            continue;
        }
        let preheader_id = outside_preds[0];
        let bb_idx_map: HashMap<BlockId, usize> =
            func.blocks.iter().enumerate().map(|(i, b)| (b.id, i)).collect();
        let Some(&pre_idx) = bb_idx_map.get(&preheader_id) else {
            continue;
        };
        if !matches!(func.blocks[pre_idx].term, Terminator::Goto(t) if t == header) {
            continue;
        }

        let escaped = func.escaped_locals();
        let mut consts: HashMap<ValueId, i64> = HashMap::new();
        let mut def_block: HashMap<ValueId, BlockId> = HashMap::new();
        let mut addr_global_of: HashMap<ValueId, u32> = HashMap::new();
        for bb in &func.blocks {
            for inst in &bb.insts {
                if let Some(d) = inst.dst() {
                    def_block.insert(d, bb.id);
                }
                match inst {
                    IrInst::ConstInt { dst, val, .. } => {
                        consts.insert(*dst, *val);
                    }
                    IrInst::AddrGlobal { dst, global } => {
                        addr_global_of.insert(*dst, *global);
                    }
                    _ => {}
                }
            }
        }

        // Find single-update non-escaped integer induction variable inside `body`:
        // exactly one `StoreLocal(iv_slot, iv_next)` in `body`, located in `latch`,
        // where `iv_next = LoadLocal(iv_slot) + const_step` with `const_step > 0`.
        let mut slot_stores: HashMap<SlotId, Vec<(BlockId, usize, ValueId)>> = HashMap::new();
        for &bid in &body {
            let Some(&bi) = bb_idx_map.get(&bid) else {
                continue;
            };
            for (ii, inst) in func.blocks[bi].insts.iter().enumerate() {
                if let IrInst::StoreLocal { slot, val, .. } = inst {
                    slot_stores.entry(*slot).or_default().push((bid, ii, *val));
                }
            }
        }

        let mut iv_candidates: HashMap<SlotId, (usize, i64)> = HashMap::new(); // iv_slot -> (latch_store_idx, step)
        if let Some(&latch_idx) = bb_idx_map.get(&latch) {
            let latch_bb = &func.blocks[latch_idx];
            let mut local_loads: HashMap<ValueId, SlotId> = HashMap::new();
            let mut add_defs: HashMap<ValueId, (IrType, ValueId, ValueId)> = HashMap::new();
            for inst in &latch_bb.insts {
                match inst {
                    IrInst::LoadLocal { dst, slot, .. } => {
                        local_loads.insert(*dst, *slot);
                    }
                    IrInst::BinOp {
                        dst,
                        op: IrBinOp::Add,
                        ty,
                        lhs,
                        rhs,
                        ..
                    } if matches!(ty, IrType::Int(IntType::U8 | IntType::U16)) => {
                        add_defs.insert(*dst, (ty.clone(), *lhs, *rhs));
                    }
                    _ => {}
                }
            }
            for (&slot, stores) in &slot_stores {
                if stores.len() != 1 || (slot as usize) < escaped.len() && escaped[slot as usize] {
                    continue;
                }
                let (s_bid, s_ii, next_val) = stores[0];
                if s_bid != latch {
                    continue;
                }
                let Some((_, lhs, rhs)) = add_defs.get(&next_val) else {
                    continue;
                };
                let step = if local_loads.get(lhs) == Some(&slot) {
                    consts.get(rhs).copied()
                } else if local_loads.get(rhs) == Some(&slot) {
                    consts.get(lhs).copied()
                } else {
                    None
                };
                if let Some(step) = step
                    && step > 0
                {
                    iv_candidates.insert(slot, (s_ii, step));
                }
            }
        }
        if iv_candidates.is_empty() {
            continue;
        }

        // Collect all IndexAddr inside `body` matching `(base_key, iv_slot, elem_size)`
        // strictly before any `latch` IV store.
        let mut init_cache: HashMap<SlotId, Option<i64>> = HashMap::new();
        let mut bound_cache: HashMap<SlotId, Option<i64>> = HashMap::new();

        struct OsrGroup {
            base_key: BaseKey,
            iv_slot: SlotId,
            elem_size: u16,
            init: i64,
            step: i64,
            sites: Vec<(BlockId, usize, ValueId)>,
        }
        let mut groups: Vec<OsrGroup> = Vec::new();
        let mut group_idx_map: HashMap<(BaseKey, SlotId, u16), usize> = HashMap::new();

        let mut val_iv: HashMap<ValueId, SlotId> = HashMap::new();
        for &bid in &body {
            let Some(&bi) = bb_idx_map.get(&bid) else {
                continue;
            };
            for (ii, inst) in func.blocks[bi].insts.iter().enumerate() {
                match inst {
                    IrInst::LoadLocal { dst, slot, .. } if iv_candidates.contains_key(slot) => {
                        let (latch_store_ii, _) = iv_candidates[slot];
                        if bid != latch || ii < latch_store_ii {
                            val_iv.insert(*dst, *slot);
                        }
                    }
                    IrInst::Cast {
                        dst,
                        val,
                        from_ty: IrType::Int(IntType::U8),
                        to_ty: IrType::Int(IntType::U16),
                    } => {
                        if let Some(&slot) = val_iv.get(val) {
                            val_iv.insert(*dst, slot);
                        }
                    }
                    _ => {}
                }
            }
        }

        for &bid in &body {
            let Some(&bi) = bb_idx_map.get(&bid) else {
                continue;
            };
            let bb = &func.blocks[bi];
            for (ii, inst) in bb.insts.iter().enumerate() {
                if let IrInst::IndexAddr {
                    dst,
                    base,
                    index,
                    elem_size,
                    len,
                    ..
                } = inst
                    && *elem_size >= 1
                {
                    let Some(&iv_slot) = val_iv.get(index) else {
                        continue;
                    };
                    let (latch_store_ii, step) = iv_candidates[&iv_slot];
                    if bid == latch && ii >= latch_store_ii {
                        continue;
                    }
                    let base_key = if let Some(&gid) = addr_global_of.get(base) {
                        BaseKey::Global(gid)
                    } else if def_block
                        .get(base)
                        .is_some_and(|db| !body.contains(db) && cfg.dominates(*db, preheader_id))
                    {
                        BaseKey::Val(*base)
                    } else {
                        continue;
                    };
                    let init_opt = *init_cache
                        .entry(iv_slot)
                        .or_insert_with(|| find_loop_iv_init(func, &cfg, preheader_id, iv_slot, &consts));
                    let Some(init) = init_opt else {
                        continue;
                    };
                    if init < 0 {
                        continue;
                    }
                    if !is_release {
                        let bound_opt = *bound_cache.entry(iv_slot).or_insert_with(|| {
                            find_loop_iv_upper_bound(func, header, &body, iv_slot, &consts)
                        });
                        let Some(excl_bound) = bound_opt else {
                            continue;
                        };
                        if excl_bound < 0 || excl_bound > *len as i64 {
                            continue;
                        }
                    }
                    let key = (base_key, iv_slot, *elem_size);
                    if let Some(&gi) = group_idx_map.get(&key) {
                        groups[gi].sites.push((bid, ii, *dst));
                    } else if groups.len() < MAX_OSR_PTRS_PER_LOOP {
                        let gi = groups.len();
                        group_idx_map.insert(key, gi);
                        groups.push(OsrGroup {
                            base_key,
                            iv_slot,
                            elem_size: *elem_size,
                            init,
                            step,
                            sites: vec![(bid, ii, *dst)],
                        });
                    }
                }
            }
        }

        groups.retain(|g| g.elem_size >= 2 || g.sites.len() >= 2);
        if groups.is_empty() {
            continue;
        }

        let u16t = IrType::Int(IntType::U16);
        let mut next_slot: SlotId = func.locals.iter().map(|l| l.id + 1).max().unwrap_or(0);
        let mut next_val = func.next_val;
        let mut alloc_val = || {
            let v = next_val;
            next_val += 1;
            v
        };

        // Track latch pointer increments keyed by `latch_store_ii` so insertions preserve indices.
        let mut latch_appends: HashMap<SlotId, Vec<IrInst>> = HashMap::new();

        for g in &groups {
            let ptr_slot = next_slot;
            next_slot += 1;
            func.locals.push(LocalSlot {
                id: ptr_slot,
                name: format!("__osr_ptr_{ptr_slot}"),
                ty: u16t.clone(),
                mutable: true,
            });

            let base_val = match g.base_key {
                BaseKey::Global(gid) => {
                    let bv = alloc_val();
                    func.blocks[pre_idx]
                        .insts
                        .push(IrInst::AddrGlobal { dst: bv, global: gid });
                    bv
                }
                BaseKey::Val(v) => v,
            };

            let init_ptr_val = if g.init == 0 {
                base_val
            } else {
                let off_v = alloc_val();
                let sum_v = alloc_val();
                func.blocks[pre_idx].insts.push(IrInst::ConstInt {
                    dst: off_v,
                    ty: u16t.clone(),
                    val: g.init.wrapping_mul(g.elem_size as i64),
                });
                func.blocks[pre_idx].insts.push(IrInst::BinOp {
                    dst: sum_v,
                    op: IrBinOp::Add,
                    mode: ArithMode::Wrapping,
                    ty: u16t.clone(),
                    lhs: base_val,
                    rhs: off_v,
                    site_id: 0,
                });
                sum_v
            };
            func.blocks[pre_idx].insts.push(IrInst::StoreLocal {
                slot: ptr_slot,
                val: init_ptr_val,
                ty: u16t.clone(),
            });

            for &(bid, ii, dst) in &g.sites {
                let bi = bb_idx_map[&bid];
                func.blocks[bi].insts[ii] = IrInst::LoadLocal {
                    dst,
                    slot: ptr_slot,
                    ty: u16t.clone(),
                };
            }

            let cur_p = alloc_val();
            let step_c = alloc_val();
            let next_p = alloc_val();
            let inc_insts = latch_appends.entry(g.iv_slot).or_default();
            inc_insts.push(IrInst::LoadLocal {
                dst: cur_p,
                slot: ptr_slot,
                ty: u16t.clone(),
            });
            inc_insts.push(IrInst::ConstInt {
                dst: step_c,
                ty: u16t.clone(),
                val: g.step.wrapping_mul(g.elem_size as i64),
            });
            inc_insts.push(IrInst::BinOp {
                dst: next_p,
                op: IrBinOp::Add,
                mode: ArithMode::Wrapping,
                ty: u16t.clone(),
                lhs: cur_p,
                rhs: step_c,
                site_id: 0,
            });
            inc_insts.push(IrInst::StoreLocal {
                slot: ptr_slot,
                val: next_p,
                ty: u16t.clone(),
            });
        }

        if let Some(&latch_idx) = bb_idx_map.get(&latch) {
            let mut insertions: Vec<(usize, Vec<IrInst>)> = latch_appends
                .into_iter()
                .map(|(iv_s, insts)| (iv_candidates[&iv_s].0, insts))
                .collect();
            insertions.sort_by_key(|(idx, _)| std::cmp::Reverse(*idx));
            for (store_idx, insts) in insertions {
                let pos = store_idx + 1;
                func.blocks[latch_idx].insts.splice(pos..pos, insts);
            }
        }

        func.next_val = next_val;
        any_changed = true;
    }

    if any_changed {
        remove_dead_pure_insts(func);
    }
    any_changed
}

fn clone_pure_inst_remapped(
    inst: &IrInst,
    val_map: &mut HashMap<ValueId, ValueId>,
    alloc_val: &mut dyn FnMut() -> ValueId,
) -> IrInst {
    let mut cloned = inst.clone();
    cloned.for_each_use_mut(|v| {
        if let Some(&mv) = val_map.get(v) {
            *v = mv;
        }
    });
    if let Some(d) = cloned.dst_mut() {
        let orig = *d;
        let nd = *val_map.entry(orig).or_insert_with(alloc_val);
        *d = nd;
    }
    cloned
}

/// Rotates top-tested natural loops whose header contains only pure compare instructions
/// into bottom-tested do-while form by cloning the exit test into the single latch.
pub fn rotate_counted_loops(func: &mut IrFunction) -> bool {
    if func.blocks.is_empty() {
        return false;
    }
    let mut any_changed = false;
    let mut processed_headers: HashSet<BlockId> = HashSet::new();

    loop {
        let cfg = Cfg::build(func);
        let Some(entry_id) = cfg.entry_block() else {
            break;
        };

        let mut by_header: HashMap<BlockId, (Vec<BlockId>, BTreeSet<BlockId>)> = HashMap::new();
        for lp in cfg.natural_loops() {
            let entry = by_header
                .entry(lp.header)
                .or_insert_with(|| (Vec::new(), BTreeSet::new()));
            entry.0.push(lp.latch);
            entry.1.extend(&lp.body);
        }
        let mut loops: Vec<(BlockId, BlockId, BTreeSet<BlockId>)> = by_header
            .into_iter()
            .filter_map(|(h, (latches, body))| {
                (h != entry_id && latches.len() == 1 && !processed_headers.contains(&h))
                    .then_some((h, latches[0], body))
            })
            .collect();
        loops.sort_by_key(|(h, _, body)| (body.len(), *h));

        let Some((header, latch, body)) = loops.into_iter().next() else {
            break;
        };
        processed_headers.insert(header);

        if header == latch {
            continue;
        }
        let outside_preds: Vec<BlockId> = cfg
            .preds(header)
            .iter()
            .copied()
            .filter(|p| !body.contains(p))
            .collect();
        if outside_preds.len() != 1 {
            continue;
        }
        let preheader_id = outside_preds[0];
        let bb_idx_map: HashMap<BlockId, usize> =
            func.blocks.iter().enumerate().map(|(i, b)| (b.id, i)).collect();
        let (Some(&hdr_idx), Some(&latch_idx), Some(&pre_idx)) = (
            bb_idx_map.get(&header),
            bb_idx_map.get(&latch),
            bb_idx_map.get(&preheader_id),
        ) else {
            continue;
        };
        if !matches!(func.blocks[latch_idx].term, Terminator::Goto(t) if t == header) {
            continue;
        }
        if !matches!(func.blocks[pre_idx].term, Terminator::Goto(t) if t == header) {
            continue;
        }

        let Terminator::Branch {
            cond,
            then_bb,
            else_bb,
        } = func.blocks[hdr_idx].term
        else {
            continue;
        };
        if !(body.contains(&then_bb) && !body.contains(&else_bb)) {
            continue;
        }
        if func.blocks[hdr_idx].insts.len() > 6 {
            continue;
        }
        let hdr_all_pure = func.blocks[hdr_idx].insts.iter().all(|inst| {
            matches!(
                inst,
                IrInst::ConstInt { .. }
                    | IrInst::ConstBool { .. }
                    | IrInst::LoadLocal { .. }
                    | IrInst::Cast { .. }
                    | IrInst::BinOp {
                        op: IrBinOp::Eq
                            | IrBinOp::Ne
                            | IrBinOp::Lt
                            | IrBinOp::Le
                            | IrBinOp::Gt
                            | IrBinOp::Ge,
                        ..
                    }
            )
        });
        if !hdr_all_pure {
            continue;
        }

        // Check whether any SSA value defined inside `header` is used outside `header`.
        let mut hdr_defs: HashSet<ValueId> = HashSet::new();
        for inst in &func.blocks[hdr_idx].insts {
            if let Some(d) = inst.dst() {
                hdr_defs.insert(d);
            }
        }
        let mut hdr_used_outside = false;
        for bb in &func.blocks {
            if bb.id == header {
                continue;
            }
            for inst in &bb.insts {
                inst.for_each_use(|v| {
                    if hdr_defs.contains(&v) {
                        hdr_used_outside = true;
                    }
                });
            }
            match bb.term {
                Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } => {
                    if hdr_defs.contains(&v) {
                        hdr_used_outside = true;
                    }
                }
                _ => {}
            }
        }

        let hdr_insts = &func.blocks[hdr_idx].insts;
        let mut entry_ok = false;
        if !hdr_used_outside {
            let mut consts: HashMap<ValueId, i64> = HashMap::new();
            for bb in &func.blocks {
                for inst in &bb.insts {
                    if let IrInst::ConstInt { dst, val, .. } = inst {
                        consts.insert(*dst, *val);
                    }
                }
            }
            let mut hdr_loads: HashMap<ValueId, SlotId> = HashMap::new();
            for inst in hdr_insts {
                if let IrInst::LoadLocal { dst, slot, .. } = inst {
                    hdr_loads.insert(*dst, *slot);
                }
            }
            if let Some(IrInst::BinOp {
                dst,
                op: cmp_op @ (IrBinOp::Lt | IrBinOp::Le),
                ty: IrType::Int(IntType::U8 | IntType::U16),
                lhs,
                rhs,
                ..
            }) = hdr_insts.last()
                && *dst == cond
                && let Some(&iv_slot) = hdr_loads.get(lhs)
                && let Some(bound) = consts.get(rhs).copied()
                && let Some(init) = find_loop_iv_init(func, &cfg, preheader_id, iv_slot, &consts)
            {
                entry_ok = if *cmp_op == IrBinOp::Lt {
                    init >= 0 && init < bound
                } else {
                    init >= 0 && init <= bound
                };
            }
        }
        let should_rotate = (!hdr_used_outside && entry_ok)
            || (hdr_insts.len() <= 3 && body.len() <= 3);
        if !should_rotate {
            continue;
        }

        // Clone `header` test instructions into `latch` and make `latch` jump straight to `then_bb`.
        let mut next_val = func.next_val;
        let mut val_map: HashMap<ValueId, ValueId> = HashMap::new();
        let hdr_insts = func.blocks[hdr_idx].insts.clone();
        let mut cloned_insts = Vec::with_capacity(hdr_insts.len());
        let mut alloc_val = || {
            let v = next_val;
            next_val += 1;
            v
        };
        for inst in &hdr_insts {
            cloned_insts.push(clone_pure_inst_remapped(inst, &mut val_map, &mut alloc_val));
        }
        let cloned_cond = *val_map.get(&cond).unwrap_or(&cond);
        func.blocks[latch_idx].insts.extend(cloned_insts);
        func.blocks[latch_idx].term = Terminator::Branch {
            cond: cloned_cond,
            then_bb,
            else_bb,
        };

        if !hdr_used_outside && entry_ok {
            func.blocks[pre_idx].term = Terminator::Goto(then_bb);
        }

        func.next_val = next_val;
        any_changed = true;
    }

    if any_changed {
        remove_dead_unreachable_blocks(func);
    }
    any_changed
}

fn remove_dead_unreachable_blocks(func: &mut IrFunction) {
    if func.blocks.is_empty() {
        return;
    }
    let cfg = Cfg::build(func);
    func.blocks.retain(|bb| cfg.is_reachable(bb.id));
}

fn slot_live_out_at_exit(
    func: &IrFunction,
    cfg: &Cfg,
    exit_bb: BlockId,
    slot: SlotId,
) -> bool {
    let bb_idx_map: HashMap<BlockId, usize> =
        func.blocks.iter().enumerate().map(|(i, b)| (b.id, i)).collect();
    let mut visited: HashSet<BlockId> = HashSet::new();
    let mut stack = vec![exit_bb];
    while let Some(bid) = stack.pop() {
        if !visited.insert(bid) {
            continue;
        }
        let Some(&bi) = bb_idx_map.get(&bid) else {
            continue;
        };
        let mut overwritten = false;
        for inst in &func.blocks[bi].insts {
            match inst {
                IrInst::LoadLocal { slot: s, .. } if *s == slot => {
                    return true;
                }
                IrInst::StoreLocal { slot: s, .. } if *s == slot => {
                    overwritten = true;
                    break;
                }
                _ => {}
            }
        }
        if !overwritten {
            for &succ in cfg.succs(bid) {
                if !visited.contains(&succ) {
                    stack.push(succ);
                }
            }
        }
    }
    false
}

/// Converts rotated single-latch count-up `u8` trip counters (`iv += 1; iv < bound`)
/// whose index variable is otherwise dead into `u8` countdowns (`iv -= 1; iv != 0`)
/// so Z80 codegen can lower the backedge to `djnz` / `dec r; jp nz`.
pub fn convert_trip_counters_to_countdown(func: &mut IrFunction) -> bool {
    if func.blocks.is_empty() {
        return false;
    }
    let mut any_changed = false;
    let mut processed_headers: HashSet<BlockId> = HashSet::new();

    loop {
        let cfg = Cfg::build(func);
        let Some(entry_id) = cfg.entry_block() else {
            break;
        };

        let mut by_header: HashMap<BlockId, (Vec<BlockId>, BTreeSet<BlockId>)> = HashMap::new();
        for lp in cfg.natural_loops() {
            let entry = by_header
                .entry(lp.header)
                .or_insert_with(|| (Vec::new(), BTreeSet::new()));
            entry.0.push(lp.latch);
            entry.1.extend(&lp.body);
        }
        let mut loops: Vec<(BlockId, BlockId, BTreeSet<BlockId>)> = by_header
            .into_iter()
            .filter_map(|(h, (latches, body))| {
                (h != entry_id && latches.len() == 1 && !processed_headers.contains(&h))
                    .then_some((h, latches[0], body))
            })
            .collect();
        loops.sort_by_key(|(h, _, body)| (body.len(), *h));

        let Some((header, latch, body)) = loops.into_iter().next() else {
            break;
        };
        processed_headers.insert(header);

        let outside_preds: Vec<BlockId> = cfg
            .preds(header)
            .iter()
            .copied()
            .filter(|p| !body.contains(p))
            .collect();
        if outside_preds.len() != 1 {
            continue;
        }
        let pre_or_guard_id = outside_preds[0];
        let bb_idx_map: HashMap<BlockId, usize> =
            func.blocks.iter().enumerate().map(|(i, b)| (b.id, i)).collect();
        let Some(&latch_idx) = bb_idx_map.get(&latch) else {
            continue;
        };
        let Terminator::Branch {
            cond,
            then_bb,
            else_bb,
        } = func.blocks[latch_idx].term
        else {
            continue;
        };
        if !(then_bb == header && !body.contains(&else_bb)) {
            continue;
        }

        let escaped = func.escaped_locals();
        let mut consts: HashMap<ValueId, i64> = HashMap::new();
        for bb in &func.blocks {
            for inst in &bb.insts {
                if let IrInst::ConstInt { dst, val, .. } = inst {
                    consts.insert(*dst, *val);
                }
            }
        }

        // Match count-up +1 store and cloned test in `latch`.
        let latch_bb = &func.blocks[latch_idx];
        let mut local_loads: HashMap<ValueId, SlotId> = HashMap::new();
        let mut add1_defs: HashMap<ValueId, SlotId> = HashMap::new();
        let mut cmp_info: Option<(SlotId, ValueId, IrBinOp)> = None;
        for inst in &latch_bb.insts {
            match inst {
                IrInst::LoadLocal { dst, slot, .. } => {
                    local_loads.insert(*dst, *slot);
                }
                IrInst::BinOp {
                    dst,
                    op: IrBinOp::Add,
                    ty: IrType::Int(IntType::U8),
                    lhs,
                    rhs,
                    ..
                } => {
                    if let Some(&s) = local_loads.get(lhs)
                        && consts.get(rhs) == Some(&1)
                    {
                        add1_defs.insert(*dst, s);
                    } else if let Some(&s) = local_loads.get(rhs)
                        && consts.get(lhs) == Some(&1)
                    {
                        add1_defs.insert(*dst, s);
                    }
                }
                IrInst::BinOp {
                    dst,
                    op: op @ (IrBinOp::Lt | IrBinOp::Le),
                    ty: IrType::Int(IntType::U8),
                    lhs,
                    rhs,
                    ..
                } if *dst == cond => {
                    if let Some(&s) = local_loads.get(lhs) {
                        cmp_info = Some((s, *rhs, *op));
                    }
                }
                _ => {}
            }
        }
        let Some((iv_slot, bound_val, cmp_op)) = cmp_info else {
            continue;
        };
        if (iv_slot as usize) < escaped.len() && escaped[iv_slot as usize] {
            continue;
        }
        if func.locals.get(iv_slot as usize).map(|l| &l.ty) != Some(&IrType::Int(IntType::U8))
            && !func
                .locals
                .iter()
                .any(|l| l.id == iv_slot && l.ty == IrType::Int(IntType::U8))
        {
            continue;
        }

        // Ensure `iv_slot` has a unique store in `body` (at `latch_store_idx`),
        // NO other reads anywhere in `body` besides the +1 load and latch test load,
        // and no uses of the post-store latch test outside `cond`.
        let mut store_pos: Option<usize> = None;
        let mut abort = false;
        let mut other_loads_in_latch: Vec<ValueId> = Vec::new();
        for &bid in &body {
            let Some(&bi) = bb_idx_map.get(&bid) else {
                abort = true;
                break;
            };
            for (ii, inst) in func.blocks[bi].insts.iter().enumerate() {
                match inst {
                    IrInst::StoreLocal { slot, val, .. } if *slot == iv_slot => {
                        if bid != latch || store_pos.is_some() || add1_defs.get(val) != Some(&iv_slot) {
                            abort = true;
                            break;
                        }
                        store_pos = Some(ii);
                    }
                    IrInst::LoadLocal { dst, slot, .. } if *slot == iv_slot => {
                        if bid != latch {
                            abort = true;
                            break;
                        }
                        other_loads_in_latch.push(*dst);
                    }
                    _ => {}
                }
            }
            if abort {
                break;
            }
        }
        let Some(latch_store_idx) = store_pos else {
            continue;
        };
        if abort || other_loads_in_latch.len() != 2 {
            continue;
        }
        if slot_live_out_at_exit(func, &cfg, else_bb, iv_slot) {
            continue;
        }

        // Verify `pre_or_guard_id` initializes `iv_slot = init` (preheader -> Goto(header))
        // with `0 <= init < bound <= 255`. (Handle either direct preheader or preheader->guard.)
        let bound_const_opt = consts.get(&bound_val).copied().map(|b| {
            if cmp_op == IrBinOp::Lt { b } else { b + 1 }
        });
        let Some(excl_bound) = bound_const_opt else {
            continue;
        };
        if excl_bound <= 0 || excl_bound > 255 {
            continue;
        }

        let Some(&pg_idx) = bb_idx_map.get(&pre_or_guard_id) else {
            continue;
        };
        let mut init_store_loc: Option<(usize, usize, i64)> = None;
        if matches!(func.blocks[pg_idx].term, Terminator::Goto(t) if t == header) {
            for (ii, inst) in func.blocks[pg_idx].insts.iter().enumerate().rev() {
                if let IrInst::StoreLocal { slot, val, .. } = inst
                    && *slot == iv_slot
                    && let Some(&init_v) = consts.get(val)
                {
                    init_store_loc = Some((pg_idx, ii, init_v));
                    break;
                }
            }
        } else if matches!(func.blocks[pg_idx].term, Terminator::Branch { then_bb: t, else_bb: e, .. } if t == header && e == else_bb)
            && cfg.preds(pre_or_guard_id).len() == 1
        {
            let pre_id = cfg.preds(pre_or_guard_id)[0];
            if let Some(&p_idx) = bb_idx_map.get(&pre_id)
                && matches!(func.blocks[p_idx].term, Terminator::Goto(t) if t == pre_or_guard_id)
            {
                for (ii, inst) in func.blocks[p_idx].insts.iter().enumerate().rev() {
                    if let IrInst::StoreLocal { slot, val, .. } = inst
                        && *slot == iv_slot
                        && let Some(&init_v) = consts.get(val)
                    {
                        init_store_loc = Some((p_idx, ii, init_v));
                        break;
                    }
                }
            }
        }
        let Some((init_bb_idx, _init_inst_idx, init_v)) = init_store_loc else {
            continue;
        };
        if init_v < 0 || init_v >= excl_bound {
            continue;
        }
        let trip_count = excl_bound - init_v;
        if !(1..=255).contains(&trip_count) {
            continue;
        }

        // Check all preconditions before allocating ValueIds or mutating any block.
        let mut bypass_guard = false;
        if pre_or_guard_id != func.blocks[init_bb_idx].id {
            let mut pg_defs: HashSet<ValueId> = HashSet::new();
            for inst in &func.blocks[pg_idx].insts {
                if let Some(d) = inst.dst() {
                    pg_defs.insert(d);
                }
            }
            let mut used_out = false;
            for (bi, bb) in func.blocks.iter().enumerate() {
                if bi == pg_idx {
                    continue;
                }
                for inst in &bb.insts {
                    inst.for_each_use(|v| {
                        if pg_defs.contains(&v) {
                            used_out = true;
                        }
                    });
                }
            }
            if used_out {
                continue;
            }
            bypass_guard = true;
        }

        let (v_cur_load, v_cmp_load) = (other_loads_in_latch[0], other_loads_in_latch[1]);
        let old_add_val = match &func.blocks[latch_idx].insts[latch_store_idx] {
            IrInst::StoreLocal { val, .. } => *val,
            _ => unreachable!(),
        };
        let drop_vals: HashSet<ValueId> = [v_cur_load, old_add_val, v_cmp_load, cond]
            .into_iter()
            .collect();
        // Check none of `drop_vals` are used outside `latch` or in other insts in `latch`!
        let mut extra_use = false;
        for (bi, bb) in func.blocks.iter().enumerate() {
            for (ii, inst) in bb.insts.iter().enumerate() {
                let is_self_pat = bi == latch_idx
                    && (ii == latch_store_idx
                        || inst.dst().is_some_and(|d| drop_vals.contains(&d)));
                if !is_self_pat {
                    inst.for_each_use(|v| {
                        if drop_vals.contains(&v) {
                            extra_use = true;
                        }
                    });
                }
            }
            if bi != latch_idx {
                match bb.term {
                    Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } => {
                        if drop_vals.contains(&v) {
                            extra_use = true;
                        }
                    }
                    _ => {}
                }
            }
        }
        if extra_use {
            continue;
        }

        // Preserve any unrelated instructions (e.g. OSR pointer updates) in `latch` and emit the
        // countdown `iv -= 1; iv != 0` at the end of `latch` so flags and counter match `djnz`.
        let u8t = IrType::Int(IntType::U8);
        let mut next_val = func.next_val;
        let mut alloc_val = || {
            let v = next_val;
            next_val += 1;
            v
        };
        let count_c = alloc_val();
        func.blocks[init_bb_idx].insts.push(IrInst::ConstInt {
            dst: count_c,
            ty: u8t.clone(),
            val: trip_count,
        });
        func.blocks[init_bb_idx].insts.push(IrInst::StoreLocal {
            slot: iv_slot,
            val: count_c,
            ty: u8t.clone(),
        });
        if bypass_guard {
            func.blocks[init_bb_idx].term = Terminator::Goto(header);
        }

        func.blocks[latch_idx].insts.retain(|inst| {
            !matches!(inst, IrInst::StoreLocal { slot, .. } if *slot == iv_slot)
                && !inst.dst().is_some_and(|d| drop_vals.contains(&d))
        });

        let c_load = alloc_val();
        let c_1 = alloc_val();
        let c_dec = alloc_val();
        let c_0 = alloc_val();
        let c_nz = alloc_val();
        func.blocks[latch_idx].insts.extend([
            IrInst::LoadLocal {
                dst: c_load,
                slot: iv_slot,
                ty: u8t.clone(),
            },
            IrInst::ConstInt {
                dst: c_1,
                ty: u8t.clone(),
                val: 1,
            },
            IrInst::BinOp {
                dst: c_dec,
                op: IrBinOp::Sub,
                mode: ArithMode::Wrapping,
                ty: u8t.clone(),
                lhs: c_load,
                rhs: c_1,
                site_id: 0,
            },
            IrInst::StoreLocal {
                slot: iv_slot,
                val: c_dec,
                ty: u8t.clone(),
            },
            IrInst::ConstInt {
                dst: c_0,
                ty: u8t,
                val: 0,
            },
            IrInst::BinOp {
                dst: c_nz,
                op: IrBinOp::Ne,
                mode: ArithMode::Wrapping,
                ty: IrType::Int(IntType::U8),
                lhs: c_dec,
                rhs: c_0,
                site_id: 0,
            },
        ]);
        func.blocks[latch_idx].term = Terminator::Branch {
            cond: c_nz,
            then_bb,
            else_bb,
        };
        func.next_val = next_val;
        any_changed = true;
    }

    if any_changed {
        remove_dead_unreachable_blocks(func);
        remove_dead_pure_insts(func);
    }
    any_changed
}

/// Moves matching 32-bit unsigned `(Div, Rem)` / `(Rem, Div)` instructions in the same basic
/// block adjacent to each other so codegen can fuse both into a single `__aun_udivrem32` call.
pub fn colocate_div_rem(func: &mut IrFunction) {
    let escaped = func.escaped_locals();
    // Collect function-wide constants first so LICM-hoisted constants remain visible across blocks.
    let mut fn_consts: HashMap<ValueId, (IrType, i64)> = HashMap::new();
    for bb in &func.blocks {
        for inst in &bb.insts {
            if let IrInst::ConstInt { dst, ty, val } = inst {
                fn_consts.insert(*dst, (ty.clone(), *val));
            }
        }
    }
    for bb in &mut func.blocks {
        let mut i = 0;
        while i < bb.insts.len() {
            let Some((op1, mode1, lhs1, rhs1)) = (match &bb.insts[i] {
                IrInst::BinOp {
                    op: op @ (IrBinOp::Div | IrBinOp::Rem),
                    mode,
                    ty: IrType::Int(IntType::U32),
                    lhs,
                    rhs,
                    ..
                } => Some((*op, *mode, *lhs, *rhs)),
                _ => None,
            }) else {
                i += 1;
                continue;
            };

            // Track unmodified non-escaped local loads up to `i`,
            // then scan forward for a matching partner op whose operands are equivalent.
            let mut slot_rep: HashMap<SlotId, ValueId> = HashMap::new();
            let mut val_from_slot: HashMap<ValueId, SlotId> = HashMap::new();
            for inst in &bb.insts[..i] {
                match inst {
                    IrInst::LoadLocal { dst, slot, .. } => {
                        if (*slot as usize) >= escaped.len() || !escaped[*slot as usize] {
                            slot_rep.entry(*slot).or_insert(*dst);
                            val_from_slot.insert(*dst, *slot);
                        }
                    }
                    IrInst::StoreLocal { slot, .. } => {
                        slot_rep.remove(slot);
                    }
                    _ => {}
                }
            }

            let mut partner_idx: Option<usize> = None;
            let mut cur_slot_rep = slot_rep;
            let mut cur_val_slot = val_from_slot;
            let lhs1_const = fn_consts.get(&lhs1).cloned();
            let rhs1_const = fn_consts.get(&rhs1).cloned();
            let lhs1_slot = cur_val_slot.get(&lhs1).copied();
            let rhs1_slot = cur_val_slot.get(&rhs1).copied();
            let mut lhs1_slot_clobbered = false;
            let mut rhs1_slot_clobbered = false;

            for (j, inst) in bb.insts.iter().enumerate().skip(i + 1) {
                match inst {
                    IrInst::LoadLocal { dst, slot, .. } => {
                        if (*slot as usize) >= escaped.len() || !escaped[*slot as usize] {
                            cur_slot_rep.entry(*slot).or_insert(*dst);
                            cur_val_slot.insert(*dst, *slot);
                        }
                    }
                    IrInst::StoreLocal { slot, .. } => {
                        if lhs1_slot == Some(*slot) {
                            lhs1_slot_clobbered = true;
                        }
                        if rhs1_slot == Some(*slot) {
                            rhs1_slot_clobbered = true;
                        }
                        cur_slot_rep.remove(slot);
                    }
                    IrInst::BinOp {
                        op: op2 @ (IrBinOp::Div | IrBinOp::Rem),
                        mode: mode2,
                        ty: IrType::Int(IntType::U32),
                        lhs: lhs2,
                        rhs: rhs2,
                        ..
                    } if *op2 != op1 && *mode2 == mode1 => {
                        let lhs_eq = *lhs2 == lhs1
                            || (lhs1_const.is_some() && fn_consts.get(lhs2).cloned() == lhs1_const)
                            || (!lhs1_slot_clobbered
                                && lhs1_slot.is_some()
                                && cur_val_slot.get(lhs2).copied() == lhs1_slot);
                        let rhs_eq = *rhs2 == rhs1
                            || (rhs1_const.is_some() && fn_consts.get(rhs2).cloned() == rhs1_const)
                            || (!rhs1_slot_clobbered
                                && rhs1_slot.is_some()
                                && cur_val_slot.get(rhs2).copied() == rhs1_slot);
                        if lhs_eq && rhs_eq {
                            partner_idx = Some(j);
                            break;
                        }
                    }
                    _ => {}
                }
            }

            if let Some(j) = partner_idx {
                if j > i + 1 {
                    let mut moved = bb.insts.remove(j);
                    if let IrInst::BinOp { lhs, rhs, .. } = &mut moved {
                        *lhs = lhs1;
                        *rhs = rhs1;
                    }
                    bb.insts.insert(i + 1, moved);
                } else if let IrInst::BinOp { lhs, rhs, .. } = &mut bb.insts[j] {
                    *lhs = lhs1;
                    *rhs = rhs1;
                }
                i += 2;
                continue;
            }
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colocates_u32_rem_and_div_across_intervening_store_ptr() {
        let u32t = IrType::Int(IntType::U32);
        let u16t = IrType::Int(IntType::U16);
        let mut f = IrFunction {
            id: 0,
            name: "draw".into(),
            is_pub: true,
            param_count: 1,
            locals: vec![
                LocalSlot {
                    id: 0,
                    name: "s".into(),
                    ty: u32t.clone(),
                    mutable: true,
                },
                LocalSlot {
                    id: 1,
                    name: "p".into(),
                    ty: u16t.clone(),
                    mutable: false,
                },
            ],
            ret_ty: IrType::Unit,
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            isr_using: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            blocks: vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::LoadLocal {
                        dst: 0,
                        slot: 0,
                        ty: u32t.clone(),
                    },
                    IrInst::ConstInt {
                        dst: 1,
                        ty: u32t.clone(),
                        val: 10,
                    },
                    IrInst::BinOp {
                        dst: 2,
                        op: IrBinOp::Rem,
                        mode: ArithMode::Checked,
                        ty: u32t.clone(),
                        lhs: 0,
                        rhs: 1,
                        site_id: 0,
                    },
                    IrInst::LoadLocal {
                        dst: 3,
                        slot: 1,
                        ty: u16t,
                    },
                    IrInst::StorePtr {
                        addr: 3,
                        val: 2,
                        ty: u32t.clone(),
                        volatile: false,
                    },
                    IrInst::LoadLocal {
                        dst: 4,
                        slot: 0,
                        ty: u32t.clone(),
                    },
                    IrInst::ConstInt {
                        dst: 5,
                        ty: u32t.clone(),
                        val: 10,
                    },
                    IrInst::BinOp {
                        dst: 6,
                        op: IrBinOp::Div,
                        mode: ArithMode::Checked,
                        ty: u32t.clone(),
                        lhs: 4,
                        rhs: 5,
                        site_id: 1,
                    },
                    IrInst::StoreLocal {
                        slot: 0,
                        val: 6,
                        ty: u32t,
                    },
                ],
                term: Terminator::Return(None),
            }],
            next_val: 7,
        };

        colocate_div_rem(&mut f);
        assert!(matches!(
            f.blocks[0].insts[2],
            IrInst::BinOp {
                dst: 2,
                op: IrBinOp::Rem,
                ..
            }
        ));
        assert!(matches!(
            f.blocks[0].insts[3],
            IrInst::BinOp {
                dst: 6,
                op: IrBinOp::Div,
                lhs: 0,
                rhs: 1,
                ..
            }
        ));
    }

    #[test]
    fn osr_replaces_grouped_loop_index_addr_with_strided_ptr() {
        let mut prog = IrProgram::default();
        prog.globals.push(IrGlobal {
            name: "ARR".into(),
            ty: IrType::Array {
                elem: Box::new(IrType::Int(IntType::U8)),
                len: 8,
            },
            mutable: true,
            in_rom: false,
            volatile: false,
            fixed_addr: None,
            is_extern: false,
            align: 1,
            soa: false,
            space: None,
            init: vec![],
        });
        let u8t = IrType::Int(IntType::U8);
        let u16t = IrType::Int(IntType::U16);
        prog.functions.push(IrFunction {
            id: 0,
            name: "clear_structs".into(),
            is_pub: true,
            param_count: 0,
            locals: vec![LocalSlot {
                id: 0,
                name: "i".into(),
                ty: u8t.clone(),
                mutable: true,
            }],
            ret_ty: IrType::Unit,
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            isr_using: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            blocks: vec![
                // bb0: i = 0; goto bb1
                BasicBlock {
                    id: 0,
                    insts: vec![
                        IrInst::ConstInt {
                            dst: 0,
                            ty: u8t.clone(),
                            val: 0,
                        },
                        IrInst::StoreLocal {
                            slot: 0,
                            val: 0,
                            ty: u8t.clone(),
                        },
                    ],
                    term: Terminator::Goto(1),
                },
                // bb1 (header): if i < 8 goto bb2 else goto bb3
                BasicBlock {
                    id: 1,
                    insts: vec![
                        IrInst::LoadLocal {
                            dst: 1,
                            slot: 0,
                            ty: u8t.clone(),
                        },
                        IrInst::ConstInt {
                            dst: 2,
                            ty: u8t.clone(),
                            val: 8,
                        },
                        IrInst::BinOp {
                            dst: 3,
                            op: IrBinOp::Lt,
                            mode: ArithMode::Wrapping,
                            ty: u8t.clone(),
                            lhs: 1,
                            rhs: 2,
                            site_id: 0,
                        },
                    ],
                    term: Terminator::Branch {
                        cond: 3,
                        then_bb: 2,
                        else_bb: 3,
                    },
                },
                // bb2 (body/latch): ARR[i as u16] twice (elem_size 6), i += 1
                BasicBlock {
                    id: 2,
                    insts: vec![
                        IrInst::AddrGlobal { dst: 4, global: 0 },
                        IrInst::LoadLocal {
                            dst: 5,
                            slot: 0,
                            ty: u8t.clone(),
                        },
                        IrInst::Cast {
                            dst: 6,
                            val: 5,
                            from_ty: u8t.clone(),
                            to_ty: u16t.clone(),
                        },
                        IrInst::IndexAddr {
                            dst: 7,
                            base: 4,
                            index: 6,
                            elem_size: 6,
                            len: 8,
                            site_id: 0,
                        },
                        IrInst::ConstInt {
                            dst: 8,
                            ty: u8t.clone(),
                            val: 0,
                        },
                        IrInst::StorePtr {
                            addr: 7,
                            val: 8,
                            ty: u8t.clone(),
                            volatile: false,
                        },
                        IrInst::IndexAddr {
                            dst: 9,
                            base: 4,
                            index: 6,
                            elem_size: 6,
                            len: 8,
                            site_id: 1,
                        },
                        IrInst::StorePtr {
                            addr: 9,
                            val: 8,
                            ty: u8t.clone(),
                            volatile: false,
                        },
                        IrInst::LoadLocal {
                            dst: 10,
                            slot: 0,
                            ty: u8t.clone(),
                        },
                        IrInst::ConstInt {
                            dst: 11,
                            ty: u8t.clone(),
                            val: 1,
                        },
                        IrInst::BinOp {
                            dst: 12,
                            op: IrBinOp::Add,
                            mode: ArithMode::Wrapping,
                            ty: u8t.clone(),
                            lhs: 10,
                            rhs: 11,
                            site_id: 2,
                        },
                        IrInst::StoreLocal {
                            slot: 0,
                            val: 12,
                            ty: u8t,
                        },
                    ],
                    term: Terminator::Goto(1),
                },
                BasicBlock {
                    id: 3,
                    insts: vec![],
                    term: Terminator::Return(None),
                },
            ],
            next_val: 13,
        });

        let changed = reduce_loop_index_addrs(&mut prog.functions[0], false);
        assert!(changed);
        assert!(
            !prog.functions[0].blocks[2]
                .insts
                .iter()
                .any(|i| matches!(i, IrInst::IndexAddr { .. })),
            "both grouped IndexAddrs must be replaced by pointer OSR"
        );
        // Only ONE __osr_ptr local should have been created for both sites.
        assert_eq!(prog.functions[0].locals.len(), 2);
        assert!(
            prog.functions[0].blocks[2].insts.iter().any(|i| matches!(
                i,
                IrInst::ConstInt { val: 6, .. }
            )),
            "pointer increment must use stride step*elem_size = 6"
        );
        aun_ir::verify::verify_program(&prog).expect("OSR output must pass MIR verifier");
    }

    #[test]
    fn rotate_and_convert_counted_loop_to_countdown() {
        let mut prog = IrProgram::default();
        prog.globals.push(IrGlobal {
            name: "BUF".into(),
            ty: IrType::Array {
                elem: Box::new(IrType::Int(IntType::U16)),
                len: 8,
            },
            mutable: true,
            in_rom: false,
            volatile: false,
            fixed_addr: None,
            is_extern: false,
            align: 1,
            soa: false,
            space: None,
            init: vec![],
        });
        let u8t = IrType::Int(IntType::U8);
        let u16t = IrType::Int(IntType::U16);
        prog.functions.push(IrFunction {
            id: 0,
            name: "zero_buf".into(),
            is_pub: true,
            param_count: 0,
            locals: vec![LocalSlot {
                id: 0,
                name: "i".into(),
                ty: u8t.clone(),
                mutable: true,
            }],
            ret_ty: IrType::Unit,
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            isr_using: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            blocks: vec![
                BasicBlock {
                    id: 0,
                    insts: vec![
                        IrInst::ConstInt {
                            dst: 0,
                            ty: u8t.clone(),
                            val: 0,
                        },
                        IrInst::StoreLocal {
                            slot: 0,
                            val: 0,
                            ty: u8t.clone(),
                        },
                    ],
                    term: Terminator::Goto(1),
                },
                BasicBlock {
                    id: 1,
                    insts: vec![
                        IrInst::LoadLocal {
                            dst: 1,
                            slot: 0,
                            ty: u8t.clone(),
                        },
                        IrInst::ConstInt {
                            dst: 2,
                            ty: u8t.clone(),
                            val: 8,
                        },
                        IrInst::BinOp {
                            dst: 3,
                            op: IrBinOp::Lt,
                            mode: ArithMode::Wrapping,
                            ty: u8t.clone(),
                            lhs: 1,
                            rhs: 2,
                            site_id: 0,
                        },
                    ],
                    term: Terminator::Branch {
                        cond: 3,
                        then_bb: 2,
                        else_bb: 3,
                    },
                },
                BasicBlock {
                    id: 2,
                    insts: vec![
                        IrInst::AddrGlobal { dst: 4, global: 0 },
                        IrInst::LoadLocal {
                            dst: 5,
                            slot: 0,
                            ty: u8t.clone(),
                        },
                        IrInst::Cast {
                            dst: 6,
                            val: 5,
                            from_ty: u8t.clone(),
                            to_ty: u16t.clone(),
                        },
                        IrInst::IndexAddr {
                            dst: 7,
                            base: 4,
                            index: 6,
                            elem_size: 2,
                            len: 8,
                            site_id: 0,
                        },
                        IrInst::ConstInt {
                            dst: 8,
                            ty: u16t.clone(),
                            val: 0,
                        },
                        IrInst::StorePtr {
                            addr: 7,
                            val: 8,
                            ty: u16t,
                            volatile: false,
                        },
                        IrInst::LoadLocal {
                            dst: 9,
                            slot: 0,
                            ty: u8t.clone(),
                        },
                        IrInst::ConstInt {
                            dst: 10,
                            ty: u8t.clone(),
                            val: 1,
                        },
                        IrInst::BinOp {
                            dst: 11,
                            op: IrBinOp::Add,
                            mode: ArithMode::Wrapping,
                            ty: u8t.clone(),
                            lhs: 9,
                            rhs: 10,
                            site_id: 1,
                        },
                        IrInst::StoreLocal {
                            slot: 0,
                            val: 11,
                            ty: u8t,
                        },
                    ],
                    term: Terminator::Goto(1),
                },
                BasicBlock {
                    id: 3,
                    insts: vec![],
                    term: Terminator::Return(None),
                },
            ],
            next_val: 12,
        });

        optimize_z80_program(&mut prog, true);
        let f = &prog.functions[0];
        // Preheader sets countdown = 8, latch subtracts 1 and branches `!= 0` directly to bb2.
        assert!(
            f.blocks[0]
                .insts
                .iter()
                .any(|i| matches!(i, IrInst::ConstInt { val: 8, .. })),
            "preheader must initialize trip counter to 8"
        );
        let latch = f.blocks.iter().find(|b| b.id == 2).expect("bb2 retained");
        assert!(matches!(
            latch.term,
            Terminator::Branch {
                then_bb: 2,
                else_bb: 3,
                ..
            }
        ));
        assert!(latch.insts.iter().any(|i| matches!(
            i,
            IrInst::BinOp {
                op: IrBinOp::Sub,
                ..
            }
        )));
        aun_ir::verify::verify_program(&prog).expect("rotated countdown loop must pass MIR verifier");
    }
}
