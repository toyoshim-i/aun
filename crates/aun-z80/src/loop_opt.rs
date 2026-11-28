//! Z80-specific MIR loop and arithmetic pre-lowering passes (`aun-z80`).
//!
//! Runs after target-independent `aun-ir` optimization and before `ProgramAnalysis` /
//! Uchiguri static RAM allocation so transformations keep exact IR semantics and scratch liveness.

use aun_ir::cfg::Cfg;
use aun_ir::*;
use std::collections::{BTreeSet, HashMap, HashSet};

/// Entry point for all Z80-specific pre-codegen MIR optimizations.
pub fn optimize_z80_program(program: &mut IrProgram, is_release: bool) {
    let structs = &program.structs;
    let machines = &program.machines;
    let globals = &program.globals;
    for func in &mut program.functions {
        if func.blocks.is_empty() {
            continue;
        }
        thread_short_circuit_joins(func);
        lower_memset_memcpy_loops_with_globals(func, globals, is_release);
        osr_byte_copy_loops(func, globals, is_release);
        reduce_loop_index_addrs(func, is_release);
        hoist_invariant_static_index_addrs(func, is_release);
        reduce_loop_bitmask_shifts(func);
        coalesce_zero_field_stores(func);
        sink_cold_guard_spills_with_context(func, structs, machines, globals);
        sink_cold_loads_with_context(func, structs, machines, globals);
        rotate_counted_loops(func);
        convert_trip_counters_to_countdown(func);
        convert_u16_countup_to_countdown(func);
        colocate_div_rem(func);
        schedule_loads_near_use_with_context(func, structs, machines, globals);
    }
}

/// Converts loop-invariant `const_mask << iv` shifts (`u8` `iv` starting at 0, trip count <= 8)
/// into a rotating bitmask local (`mask <<= 1` at the loop latch) so codegen emits `add a, a` /
/// register shift instead of an inner `djnz` variable-shift loop.
pub fn reduce_loop_bitmask_shifts(func: &mut IrFunction) -> bool {
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
        let preheader_id = outside_preds[0];
        let bb_idx_map: HashMap<BlockId, usize> =
            func.blocks.iter().enumerate().map(|(i, b)| (b.id, i)).collect();
        let (Some(&pre_idx), Some(&latch_idx)) =
            (bb_idx_map.get(&preheader_id), bb_idx_map.get(&latch))
        else {
            continue;
        };
        if !matches!(func.blocks[pre_idx].term, Terminator::Goto(t) if t == header) {
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

        // Find single-update non-escaped u8 IV incremented by +1 in `latch`.
        let mut slot_stores: HashMap<SlotId, Vec<(BlockId, usize, ValueId)>> = HashMap::new();
        for &bid in &body {
            if let Some(&bi) = bb_idx_map.get(&bid) {
                for (ii, inst) in func.blocks[bi].insts.iter().enumerate() {
                    if let IrInst::StoreLocal { slot, val, .. } = inst {
                        slot_stores.entry(*slot).or_default().push((bid, ii, *val));
                    }
                }
            }
        }

        let mut local_loads_latch: HashMap<ValueId, SlotId> = HashMap::new();
        let mut add_defs_latch: HashMap<ValueId, (IrType, ValueId, ValueId)> = HashMap::new();
        for inst in &func.blocks[latch_idx].insts {
            match inst {
                IrInst::LoadLocal { dst, slot, .. } => {
                    local_loads_latch.insert(*dst, *slot);
                }
                IrInst::BinOp {
                    dst,
                    op: IrBinOp::Add,
                    ty,
                    lhs,
                    rhs,
                    ..
                } => {
                    add_defs_latch.insert(*dst, (ty.clone(), *lhs, *rhs));
                }
                _ => {}
            }
        }

        let mut candidate_iv: Option<(SlotId, usize, i64)> = None;
        for (&slot, stores) in &slot_stores {
            if stores.len() != 1 || (slot as usize) >= escaped.len() || escaped[slot as usize] {
                continue;
            }
            let (sbid, sii, sval) = stores[0];
            if sbid != latch {
                continue;
            }
            let Some((ty, l, r)) = add_defs_latch.get(&sval) else {
                continue;
            };
            if *ty != IrType::Int(IntType::U8) {
                continue;
            }
            let step_one = (local_loads_latch.get(l) == Some(&slot) && consts.get(r) == Some(&1))
                || (local_loads_latch.get(r) == Some(&slot) && consts.get(l) == Some(&1));
            if !step_one {
                continue;
            }
            if find_loop_iv_init(func, &cfg, preheader_id, slot, &consts) != Some(0) {
                continue;
            }
            let Some(bound) = find_loop_iv_upper_bound(func, header, &body, slot, &consts) else {
                continue;
            };
            if (2..=8).contains(&bound) {
                candidate_iv = Some((slot, sii, bound));
                break;
            }
        }
        let Some((iv_slot, latch_store_idx, bound)) = candidate_iv else {
            continue;
        };

        // Find all `(base_const: u8) << iv` instructions inside `body` before the latch step.
        let mut cands_by_base: HashMap<u8, Vec<(usize, usize)>> = HashMap::new();
        for &bid in &body {
            let bi = bb_idx_map[&bid];
            let mut iv_loads: HashSet<ValueId> = HashSet::new();
            for (ii, inst) in func.blocks[bi].insts.iter().enumerate() {
                if bid == latch && ii >= latch_store_idx {
                    break;
                }
                match inst {
                    IrInst::LoadLocal { dst, slot, .. } if *slot == iv_slot => {
                        iv_loads.insert(*dst);
                    }
                    IrInst::BinOp {
                        op: IrBinOp::Shl,
                        ty: IrType::Int(IntType::U8),
                        lhs,
                        rhs,
                        ..
                    } if iv_loads.contains(rhs) => {
                        if let Some(&c) = consts.get(lhs)
                            && (1..=255).contains(&c)
                            && ((c as u16) << ((bound - 1) as u32)) <= 255
                        {
                            cands_by_base.entry(c as u8).or_default().push((bi, ii));
                        }
                    }
                    _ => {}
                }
            }
        }
        if cands_by_base.is_empty() {
            continue;
        }

        let u8t = IrType::Int(IntType::U8);
        let mut bases: Vec<u8> = cands_by_base.keys().copied().collect();
        bases.sort_unstable();
        let mut insert_at_latch_offset = 0usize;
        for base_c in bases {
            let sites = &cands_by_base[&base_c];
            let mask_slot = func.locals.len() as SlotId;
            func.locals.push(LocalSlot {
                id: mask_slot,
                name: format!("_osr_mask_{mask_slot}"),
                ty: u8t.clone(),
                mutable: true,
            });
            let init_v = func.next_val;
            func.next_val += 1;
            func.blocks[pre_idx].insts.extend([
                IrInst::ConstInt {
                    dst: init_v,
                    ty: u8t.clone(),
                    val: base_c as i64,
                },
                IrInst::StoreLocal {
                    slot: mask_slot,
                    val: init_v,
                    ty: u8t.clone(),
                },
            ]);
            for &(bi, ii) in sites {
                if let Some(d) = func.blocks[bi].insts[ii].dst() {
                    func.blocks[bi].insts[ii] = IrInst::LoadLocal {
                        dst: d,
                        slot: mask_slot,
                        ty: u8t.clone(),
                    };
                }
            }
            let m_v = func.next_val;
            let one_v = func.next_val + 1;
            let sh_v = func.next_val + 2;
            func.next_val += 3;
            let pos = latch_store_idx + 1 + insert_at_latch_offset;
            func.blocks[latch_idx].insts.splice(
                pos..pos,
                [
                    IrInst::LoadLocal {
                        dst: m_v,
                        slot: mask_slot,
                        ty: u8t.clone(),
                    },
                    IrInst::ConstInt {
                        dst: one_v,
                        ty: u8t.clone(),
                        val: 1,
                    },
                    IrInst::BinOp {
                        dst: sh_v,
                        op: IrBinOp::Shl,
                        mode: ArithMode::Wrapping,
                        ty: u8t.clone(),
                        lhs: m_v,
                        rhs: one_v,
                        site_id: 0,
                    },
                    IrInst::StoreLocal {
                        slot: mask_slot,
                        val: sh_v,
                        ty: u8t.clone(),
                    },
                ],
            );
            insert_at_latch_offset += 4;
        }
        any_changed = true;
    }

    if any_changed {
        remove_dead_pure_insts(func);
    }
    any_changed
}

/// Coalesces contiguous runs of non-volatile zero stores (`0u8`, `0i8`, `false`, `0u16`, `0i16`)
/// into adjacent struct fields on the same base pointer into a single `IrInst::MemSet` (>= 6 bytes).
pub fn coalesce_zero_field_stores(func: &mut IrFunction) -> bool {
    let mut changed = false;
    let mut next_val = func.next_val;
    let param_count = func.param_count;
    let mut ref_param_loads: HashSet<ValueId> = HashSet::new();
    let mut ref_param_addrs: HashSet<ValueId> = HashSet::new();
    for _ in 0..3 {
        let prev = ref_param_loads.len() + ref_param_addrs.len();
        for bb in &func.blocks {
            for inst in &bb.insts {
                match inst {
                    IrInst::LoadLocal { dst, slot, .. }
                        if (*slot as usize) < param_count
                            && matches!(func.locals[*slot as usize].ty, IrType::Ref { .. }) =>
                    {
                        ref_param_loads.insert(*dst);
                    }
                    IrInst::AddrLocal { dst, slot }
                        if (*slot as usize) < param_count
                            && matches!(func.locals[*slot as usize].ty, IrType::Ref { .. }) =>
                    {
                        ref_param_addrs.insert(*dst);
                    }
                    IrInst::LoadPtr { dst, addr, .. } if ref_param_addrs.contains(addr) => {
                        ref_param_loads.insert(*dst);
                    }
                    IrInst::FieldAddr { dst, base, .. } if ref_param_loads.contains(base) => {
                        ref_param_loads.insert(*dst);
                    }
                    _ => {}
                }
            }
        }
        if ref_param_loads.len() + ref_param_addrs.len() == prev {
            break;
        }
    }

    for bb in &mut func.blocks {
        let mut field_base_off: HashMap<ValueId, (ValueId, u16)> = HashMap::new();
        let mut zero_consts: HashSet<ValueId> = HashSet::new();

        #[derive(Clone, Copy)]
        struct ZeroStoreRec {
            inst_idx: usize,
            root_base: ValueId,
            off: u16,
            size: u16,
        }

        for inst in &bb.insts {
            match inst {
                IrInst::ConstInt { dst, val: 0, .. } | IrInst::ConstBool { dst, val: false } => {
                    zero_consts.insert(*dst);
                }
                IrInst::FieldAddr {
                    dst,
                    base,
                    byte_offset,
                } => {
                    if let Some(&(root, base_off)) = field_base_off.get(base) {
                        if let Some(tot) = base_off.checked_add(*byte_offset) {
                            field_base_off.insert(*dst, (root, tot));
                        }
                    } else {
                        field_base_off.insert(*dst, (*base, *byte_offset));
                    }
                }
                _ => {}
            }
        }

        let mut runs: Vec<Vec<ZeroStoreRec>> = Vec::new();
        let mut cur_run: Vec<ZeroStoreRec> = Vec::new();

        for (ii, inst) in bb.insts.iter().enumerate() {
            match inst {
                IrInst::ConstInt { .. }
                | IrInst::ConstBool { .. }
                | IrInst::AddrLocal { .. }
                | IrInst::AddrGlobal { .. }
                | IrInst::FieldAddr { .. } => {}
                IrInst::StorePtr {
                    addr,
                    val,
                    ty,
                    volatile: false,
                } if zero_consts.contains(val) => {
                    let w: u16 = match ty {
                        IrType::Bool | IrType::Int(IntType::U8 | IntType::I8) => 1,
                        IrType::Int(IntType::U16 | IntType::I16) => 2,
                        _ => 0,
                    };
                    if w > 0 {
                        let (root_base, off) =
                            field_base_off.get(addr).copied().unwrap_or((*addr, 0));
                        if cur_run
                            .last()
                            .is_some_and(|prev| prev.root_base != root_base)
                        {
                            runs.push(std::mem::take(&mut cur_run));
                        }
                        cur_run.push(ZeroStoreRec {
                            inst_idx: ii,
                            root_base,
                            off,
                            size: w,
                        });
                    } else if !cur_run.is_empty() {
                        runs.push(std::mem::take(&mut cur_run));
                    }
                }
                _ => {
                    if !cur_run.is_empty() {
                        runs.push(std::mem::take(&mut cur_run));
                    }
                }
            }
        }
        if !cur_run.is_empty() {
            runs.push(cur_run);
        }

        // Split any run that isn't strictly contiguous or has duplicates, and apply valid >= 6B runs
        // from right to left so instruction indices remain stable while splicing.
        let mut splices: Vec<(Vec<usize>, usize, ValueId, u16, u16)> = Vec::new();
        for run in runs {
            if run.len() < 3 {
                continue;
            }
            let root = run[0].root_base;
            let mut sorted = run;
            sorted.sort_by_key(|r| r.off);
            let mut valid = true;
            let mut cursor = sorted[0].off;
            for r in &sorted {
                if r.off != cursor {
                    valid = false;
                    break;
                }
                let Some(next) = cursor.checked_add(r.size) else {
                    valid = false;
                    break;
                };
                cursor = next;
            }
            let start_off = sorted[0].off;
            let total_bytes = cursor.saturating_sub(start_off);
            let min_bytes = if ref_param_loads.contains(&root) { 12 } else { 6 };
            if !valid || total_bytes < min_bytes {
                continue;
            }
            let mut idxs: Vec<usize> = sorted.iter().map(|r| r.inst_idx).collect();
            idxs.sort_unstable();
            let emit_after = *idxs.last().unwrap();
            splices.push((idxs, emit_after, root, start_off, total_bytes));
        }

        if splices.is_empty() {
            continue;
        }
        let drop_set: HashSet<usize> = splices.iter().flat_map(|s| s.0.iter().copied()).collect();
        let mut insert_at: HashMap<usize, Vec<IrInst>> = HashMap::new();
        let u8t = IrType::Int(IntType::U8);
        for (_, anchor, root, start_off, total_bytes) in splices {
            let mut seq = Vec::new();
            let dst_addr = if start_off == 0 {
                root
            } else {
                let fa_v = next_val;
                next_val += 1;
                seq.push(IrInst::FieldAddr {
                    dst: fa_v,
                    base: root,
                    byte_offset: start_off,
                });
                fa_v
            };
            let zero_v = next_val;
            next_val += 1;
            seq.push(IrInst::ConstInt {
                dst: zero_v,
                ty: u8t.clone(),
                val: 0,
            });
            seq.push(IrInst::MemSet {
                dst_addr,
                val: zero_v,
                len: total_bytes,
            });
            insert_at.entry(anchor).or_default().extend(seq);
        }

        let mut new_insts = Vec::with_capacity(bb.insts.len());
        for (ii, inst) in bb.insts.iter().enumerate() {
            if !drop_set.contains(&ii) {
                new_insts.push(inst.clone());
            }
            if let Some(seq) = insert_at.remove(&ii) {
                new_insts.extend(seq);
            }
        }
        bb.insts = new_insts;
        changed = true;
    }

    func.next_val = next_val;
    if changed {
        remove_dead_pure_insts(func);
    }
    changed
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BaseRoot {
    Global(u32, u32),
    RomGlobal(u32, u32),
    Fixed(u32),
    Local(SlotId, u32),
    Slot(SlotId, u32),
    Val(ValueId, u32),
    Unknown,
}

fn resolve_base_root(
    defs: &HashMap<ValueId, IrInst>,
    globals: &[IrGlobal],
    consts: &HashMap<ValueId, i64>,
    mut v: ValueId,
) -> BaseRoot {
    let mut off = 0u32;
    for _ in 0..8 {
        match defs.get(&v) {
            Some(IrInst::AddrGlobal { global, .. }) => {
                if globals.get(*global as usize).is_some_and(|g| g.in_rom) {
                    return BaseRoot::RomGlobal(*global, off);
                }
                if let Some(fa) = globals.get(*global as usize).and_then(|g| g.fixed_addr) {
                    return BaseRoot::Fixed((fa as u32).saturating_add(off));
                }
                return BaseRoot::Global(*global, off);
            }
            Some(IrInst::AddrLocal { slot, .. }) => return BaseRoot::Local(*slot, off),
            Some(IrInst::ConstInt { val, .. }) => {
                return BaseRoot::Fixed(((*val as u16) as u32).saturating_add(off));
            }
            Some(IrInst::FieldAddr {
                base, byte_offset, ..
            }) => {
                off = off.saturating_add(*byte_offset as u32);
                v = *base;
            }
            Some(IrInst::Cast {
                val,
                from_ty,
                to_ty,
                ..
            }) if matches!(
                from_ty,
                IrType::Int(IntType::U16 | IntType::I16) | IrType::Ref { .. }
            ) && matches!(
                to_ty,
                IrType::Int(IntType::U16 | IntType::I16) | IrType::Ref { .. }
            ) =>
            {
                v = *val;
            }
            Some(IrInst::BinOp {
                op: IrBinOp::Add,
                lhs,
                rhs,
                ..
            }) => {
                if let Some(&c) = consts.get(rhs) {
                    off = off.saturating_add((c as u16) as u32);
                    v = *lhs;
                } else if let Some(&c) = consts.get(lhs) {
                    off = off.saturating_add((c as u16) as u32);
                    v = *rhs;
                } else {
                    return BaseRoot::Val(v, off);
                }
            }
            Some(IrInst::LoadLocal { slot, .. }) => return BaseRoot::Slot(*slot, off),
            _ => return BaseRoot::Val(v, off),
        }
    }
    BaseRoot::Unknown
}

fn non_overlapping_bases(
    defs: &HashMap<ValueId, IrInst>,
    globals: &[IrGlobal],
    consts: &HashMap<ValueId, i64>,
    src_base: ValueId,
    dst_base: ValueId,
    len: u16,
) -> bool {
    let n = len as u32;
    let r_src = resolve_base_root(defs, globals, consts, src_base);
    let r_dst = resolve_base_root(defs, globals, consts, dst_base);
    match (r_src, r_dst) {
        (BaseRoot::RomGlobal(..), _) => true,
        (BaseRoot::Global(g1, o1), BaseRoot::Global(g2, o2)) if g1 == g2 => {
            o1 + n <= o2 || o2 + n <= o1
        }
        (BaseRoot::Local(s1, o1), BaseRoot::Local(s2, o2)) if s1 == s2 => {
            o1 + n <= o2 || o2 + n <= o1
        }
        (BaseRoot::Fixed(a1), BaseRoot::Fixed(a2)) => a1 + n <= a2 || a2 + n <= a1,
        (BaseRoot::Slot(s1, o1), BaseRoot::Slot(s2, o2)) if s1 == s2 => {
            o1 + n <= o2 || o2 + n <= o1
        }
        (BaseRoot::Val(v1, o1), BaseRoot::Val(v2, o2)) if v1 == v2 => {
            o1 + n <= o2 || o2 + n <= o1
        }
        (BaseRoot::Unknown, _) | (_, BaseRoot::Unknown) => false,
        _ => true,
    }
}

/// Lowers simple constant-trip counted byte fill/copy loops (`for i in 0..N`) directly into
/// `IrInst::MemSet` / `IrInst::MemCopy` (`LDIR`) before loop strength reduction/rotation.
pub fn lower_memset_memcpy_loops(func: &mut IrFunction, is_release: bool) -> bool {
    lower_memset_memcpy_loops_with_globals(func, &[], is_release)
}

/// Same as `lower_memset_memcpy_loops`, using `globals` metadata (`in_rom`, `fixed_addr`)
/// to validate aliasing for `mem_write_u8` / `mem_read_u8` copy loops.
pub fn lower_memset_memcpy_loops_with_globals(
    func: &mut IrFunction,
    globals: &[IrGlobal],
    is_release: bool,
) -> bool {
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
                (h != entry_id
                    && latches.len() == 1
                    && body.len() == 2
                    && latches[0] != h
                    && !processed_headers.contains(&h))
                .then_some((h, latches[0], body))
            })
            .collect();
        loops.sort_by_key(|(h, _, _)| *h);
        let Some((header, body_id, body)) = loops.into_iter().next() else {
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
        let (Some(&pre_idx), Some(&hdr_idx), Some(&body_idx)) = (
            bb_idx_map.get(&preheader_id),
            bb_idx_map.get(&header),
            bb_idx_map.get(&body_id),
        ) else {
            continue;
        };
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
        if then_bb != body_id || body.contains(&else_bb) {
            continue;
        }
        if !matches!(func.blocks[body_idx].term, Terminator::Goto(t) if t == header) {
            continue;
        }
        let exit_bb = else_bb;

        let hdr_pure = func.blocks[hdr_idx].insts.iter().all(|inst| {
            matches!(
                inst,
                IrInst::ConstInt { .. }
                    | IrInst::ConstBool { .. }
                    | IrInst::LoadLocal { .. }
                    | IrInst::Cast { .. }
                    | IrInst::BinOp {
                        op: IrBinOp::Lt | IrBinOp::Le,
                        ..
                    }
            )
        });
        if !hdr_pure {
            continue;
        }

        let escaped = func.escaped_locals();
        let mut consts: HashMap<ValueId, i64> = HashMap::new();
        let mut all_defs: HashMap<ValueId, IrInst> = HashMap::new();
        let mut def_bb: HashMap<ValueId, BlockId> = HashMap::new();
        for bb in &func.blocks {
            for inst in &bb.insts {
                if let Some(d) = inst.dst() {
                    all_defs.insert(d, inst.clone());
                    def_bb.insert(d, bb.id);
                }
                if let IrInst::ConstInt { dst, val, .. } = inst {
                    consts.insert(*dst, *val);
                }
            }
        }

        // Verify no value defined inside `header` or `body_id` is used outside the loop.
        let mut loop_defs: HashSet<ValueId> = HashSet::new();
        for &bid in &[header, body_id] {
            for inst in &func.blocks[bb_idx_map[&bid]].insts {
                if let Some(d) = inst.dst() {
                    loop_defs.insert(d);
                }
            }
        }
        let mut used_outside = false;
        for bb in &func.blocks {
            if body.contains(&bb.id) {
                continue;
            }
            for inst in &bb.insts {
                inst.for_each_use(|v| {
                    if loop_defs.contains(&v) {
                        used_outside = true;
                    }
                });
            }
            if let Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } = bb.term
                && loop_defs.contains(&v)
            {
                used_outside = true;
            }
        }
        if used_outside {
            continue;
        }

        // Extract counted header: `iv < N` / `iv <= N - 1` starting at `0`.
        let mut hdr_loads: HashMap<ValueId, SlotId> = HashMap::new();
        let mut hdr_defs: HashSet<ValueId> = HashSet::new();
        for inst in &func.blocks[hdr_idx].insts {
            if let Some(d) = inst.dst() {
                hdr_defs.insert(d);
            }
            if let IrInst::LoadLocal { dst, slot, .. } = inst {
                hdr_loads.insert(*dst, *slot);
            }
        }
        let Some(IrInst::BinOp {
            dst: cmp_dst,
            op: cmp_op @ (IrBinOp::Lt | IrBinOp::Le),
            ty: IrType::Int(iv_it @ (IntType::U8 | IntType::U16)),
            lhs,
            rhs,
            ..
        }) = func.blocks[hdr_idx].insts.last()
        else {
            continue;
        };
        if *cmp_dst != cond {
            continue;
        }
        let Some(&iv_slot) = hdr_loads.get(lhs) else {
            continue;
        };
        if (iv_slot as usize) >= escaped.len() || escaped[iv_slot as usize] {
            continue;
        }
        let Some(bound) = consts.get(rhs).copied() else {
            continue;
        };
        let Some(trip_count) = (if *cmp_op == IrBinOp::Lt {
            Some(bound)
        } else {
            bound.checked_add(1)
        }) else {
            continue;
        };
        let max_n = if *iv_it == IntType::U8 { 255 } else { 0xFFFF };
        if !(2..=max_n).contains(&trip_count) {
            continue;
        }
        if find_loop_iv_init(func, &cfg, preheader_id, iv_slot, &consts) != Some(0) {
            continue;
        }

        // Validate `body_bb` instructions: pure address arithmetic, optional single non-volatile
        // LoadPtr(u8), one StorePtr(u8, non-volatile), and one `iv = iv + 1` update.
        let body_bb = &func.blocks[body_idx];
        let mut has_disallowed = false;
        let mut store_locals: Vec<(usize, SlotId, ValueId)> = Vec::new();
        let mut store_ptrs: Vec<(usize, ValueId, ValueId)> = Vec::new();
        let mut load_ptrs: Vec<usize> = Vec::new();
        let mut body_inst_by_dst: HashMap<ValueId, (usize, IrInst)> = HashMap::new();

        for (ii, inst) in body_bb.insts.iter().enumerate() {
            if let Some(d) = inst.dst() {
                body_inst_by_dst.insert(d, (ii, inst.clone()));
            }
            match inst {
                IrInst::ConstInt { .. }
                | IrInst::ConstBool { .. }
                | IrInst::LoadLocal { .. }
                | IrInst::AddrGlobal { .. }
                | IrInst::AddrLocal { .. }
                | IrInst::FieldAddr { .. }
                | IrInst::SlicePtr { .. }
                | IrInst::SliceLen { .. }
                | IrInst::Cast { .. }
                | IrInst::UnOp { .. }
                | IrInst::BinOp { .. } => {}
                IrInst::IndexAddr { elem_size, len, .. } => {
                    if *elem_size != 1 || (!is_release && *len > 0 && trip_count > *len as i64) {
                        has_disallowed = true;
                        break;
                    }
                }
                IrInst::LoadPtr { ty, volatile, .. } => {
                    if *volatile
                        || !matches!(
                            ty,
                            IrType::Int(IntType::U8 | IntType::I8) | IrType::Bool
                        )
                    {
                        has_disallowed = true;
                        break;
                    }
                    load_ptrs.push(ii);
                }
                IrInst::StoreLocal { slot, val, .. } => {
                    store_locals.push((ii, *slot, *val));
                }
                IrInst::StorePtr {
                    addr,
                    val,
                    ty,
                    volatile,
                } => {
                    if *volatile
                        || !matches!(
                            ty,
                            IrType::Int(IntType::U8 | IntType::I8) | IrType::Bool
                        )
                    {
                        has_disallowed = true;
                        break;
                    }
                    store_ptrs.push((ii, *addr, *val));
                }
                _ => {
                    has_disallowed = true;
                    break;
                }
            }
        }
        if has_disallowed || store_locals.len() != 1 || store_ptrs.len() != 1 || load_ptrs.len() > 1
        {
            continue;
        }
        let (_, sl_slot, sl_val) = store_locals[0];
        if sl_slot != iv_slot {
            continue;
        }
        let iv_step_ok = matches!(
            body_inst_by_dst.get(&sl_val),
            Some((_, IrInst::BinOp { op: IrBinOp::Add, lhs, rhs, .. }))
                if (matches!(body_inst_by_dst.get(lhs), Some((_, IrInst::LoadLocal { slot, .. })) if *slot == iv_slot)
                    && consts.get(rhs) == Some(&1))
                || (matches!(body_inst_by_dst.get(rhs), Some((_, IrInst::LoadLocal { slot, .. })) if *slot == iv_slot)
                    && consts.get(lhs) == Some(&1))
        );
        if !iv_step_ok {
            continue;
        }

        let is_iv_index = |idx_v: ValueId| -> bool {
            let mut cur = idx_v;
            if let Some((_, IrInst::Cast {
                val,
                from_ty: IrType::Int(IntType::U8),
                to_ty: IrType::Int(IntType::U16),
                ..
            })) = body_inst_by_dst.get(&cur)
            {
                cur = *val;
            }
            matches!(
                body_inst_by_dst.get(&cur),
                Some((_, IrInst::LoadLocal { slot, .. })) if *slot == iv_slot
            )
        };

        let extract_byte_indexed_base = |addr_v: ValueId| -> Option<(ValueId, Option<ValueId>)> {
            match body_inst_by_dst.get(&addr_v) {
                Some((
                    _,
                    IrInst::IndexAddr {
                        base,
                        index,
                        elem_size: 1,
                        ..
                    },
                )) if is_iv_index(*index) => Some((*base, None)),
                Some((
                    _,
                    IrInst::BinOp {
                        op: IrBinOp::Add,
                        mode,
                        ty: IrType::Int(IntType::U16),
                        lhs,
                        rhs,
                        ..
                    },
                )) if *mode != ArithMode::Saturating => {
                    if is_iv_index(*rhs) {
                        Some((*lhs, None))
                    } else if is_iv_index(*lhs) {
                        Some((*rhs, None))
                    } else {
                        let match_inner = |cand: ValueId| -> Option<ValueId> {
                            if let Some((
                                _,
                                IrInst::BinOp {
                                    op: IrBinOp::Add,
                                    mode: m2,
                                    ty: IrType::Int(IntType::U16),
                                    lhs: l2,
                                    rhs: r2,
                                    ..
                                },
                            )) = body_inst_by_dst.get(&cand)
                                && *m2 != ArithMode::Saturating
                            {
                                if is_iv_index(*r2) {
                                    return Some(*l2);
                                }
                                if is_iv_index(*l2) {
                                    return Some(*r2);
                                }
                            }
                            None
                        };
                        if let Some(k) = match_inner(*rhs) {
                            Some((*lhs, Some(k)))
                        } else {
                            match_inner(*lhs).map(|k| (*rhs, Some(k)))
                        }
                    }
                }
                _ => None,
            }
        };

        let (_, dst_elem, store_val) = store_ptrs[0];
        let Some((dst_base, dst_extra_off)) = extract_byte_indexed_base(dst_elem) else {
            continue;
        };

        // Check whether `v` is loop-invariant and gather any `body_bb` helper instructions needed
        // to rematerialize `v` in `header`.
        fn can_materialize_in_header(
            v: ValueId,
            body_id: BlockId,
            iv_slot: SlotId,
            escaped: &[bool],
            hdr_defs: &HashSet<ValueId>,
            def_bb: &HashMap<ValueId, BlockId>,
            body_inst_by_dst: &HashMap<ValueId, (usize, IrInst)>,
            needed: &mut BTreeSet<usize>,
        ) -> bool {
            if hdr_defs.contains(&v) {
                return false;
            }
            if def_bb.get(&v).copied() != Some(body_id) {
                return true;
            }
            let Some((ii, inst)) = body_inst_by_dst.get(&v) else {
                return false;
            };
            let ok = match inst {
                IrInst::ConstInt { .. }
                | IrInst::ConstBool { .. }
                | IrInst::AddrGlobal { .. } => true,
                IrInst::AddrLocal { slot, .. } => *slot != iv_slot,
                IrInst::LoadLocal { slot, .. } => {
                    *slot != iv_slot && ((*slot as usize) >= escaped.len() || !escaped[*slot as usize])
                }
                IrInst::FieldAddr { base, .. } => can_materialize_in_header(
                    *base,
                    body_id,
                    iv_slot,
                    escaped,
                    hdr_defs,
                    def_bb,
                    body_inst_by_dst,
                    needed,
                ),
                IrInst::SlicePtr { slice, .. } | IrInst::SliceLen { slice, .. } => {
                    can_materialize_in_header(
                        *slice,
                        body_id,
                        iv_slot,
                        escaped,
                        hdr_defs,
                        def_bb,
                        body_inst_by_dst,
                        needed,
                    )
                }
                IrInst::Cast { val, .. } => can_materialize_in_header(
                    *val,
                    body_id,
                    iv_slot,
                    escaped,
                    hdr_defs,
                    def_bb,
                    body_inst_by_dst,
                    needed,
                ),
                IrInst::UnOp { mode, val, .. } if *mode != ArithMode::Saturating => {
                    can_materialize_in_header(
                        *val,
                        body_id,
                        iv_slot,
                        escaped,
                        hdr_defs,
                        def_bb,
                        body_inst_by_dst,
                        needed,
                    )
                }
                IrInst::BinOp {
                    op,
                    mode,
                    lhs,
                    rhs,
                    ..
                } if !matches!(op, IrBinOp::Div | IrBinOp::Rem)
                    && *mode != ArithMode::Saturating =>
                {
                    can_materialize_in_header(
                        *lhs,
                        body_id,
                        iv_slot,
                        escaped,
                        hdr_defs,
                        def_bb,
                        body_inst_by_dst,
                        needed,
                    ) && can_materialize_in_header(
                        *rhs,
                        body_id,
                        iv_slot,
                        escaped,
                        hdr_defs,
                        def_bb,
                        body_inst_by_dst,
                        needed,
                    )
                }
                _ => false,
            };
            if ok {
                needed.insert(*ii);
            }
            ok
        }

        let mut needed_body_insts: BTreeSet<usize> = BTreeSet::new();
        if !can_materialize_in_header(
            dst_base,
            body_id,
            iv_slot,
            &escaped,
            &hdr_defs,
            &def_bb,
            &body_inst_by_dst,
            &mut needed_body_insts,
        ) {
            continue;
        }
        if let Some(k) = dst_extra_off
            && !can_materialize_in_header(
                k,
                body_id,
                iv_slot,
                &escaped,
                &hdr_defs,
                &def_bb,
                &body_inst_by_dst,
                &mut needed_body_insts,
            )
        {
            continue;
        }

        enum BulkKind {
            Copy {
                src_base: ValueId,
                src_extra_off: Option<ValueId>,
            },
            Set {
                val: ValueId,
            },
        }

        let bulk_kind: Option<BulkKind> = if let Some((
            _,
            IrInst::LoadPtr {
                addr: src_elem, ..
            },
        )) = body_inst_by_dst.get(&store_val)
        {
            let Some((src_base, src_extra_off)) = extract_byte_indexed_base(*src_elem) else {
                continue;
            };
            let extra_ok = src_extra_off.is_none_or(|k| {
                can_materialize_in_header(
                    k,
                    body_id,
                    iv_slot,
                    &escaped,
                    &hdr_defs,
                    &def_bb,
                    &body_inst_by_dst,
                    &mut needed_body_insts,
                )
            });
            if !extra_ok
                || !can_materialize_in_header(
                    src_base,
                    body_id,
                    iv_slot,
                    &escaped,
                    &hdr_defs,
                    &def_bb,
                    &body_inst_by_dst,
                    &mut needed_body_insts,
                )
                || !non_overlapping_bases(
                    &all_defs,
                    globals,
                    &consts,
                    src_base,
                    dst_base,
                    trip_count as u16,
                )
            {
                None
            } else {
                Some(BulkKind::Copy {
                    src_base,
                    src_extra_off,
                })
            }
        } else if load_ptrs.is_empty()
            && can_materialize_in_header(
                store_val,
                body_id,
                iv_slot,
                &escaped,
                &hdr_defs,
                &def_bb,
                &body_inst_by_dst,
                &mut needed_body_insts,
            )
        {
            Some(BulkKind::Set { val: store_val })
        } else {
            None
        };

        let Some(bulk_kind) = bulk_kind else {
            continue;
        };

        let iv_read_after = func.blocks.iter().any(|b| {
            !body.contains(&b.id)
                && b.insts
                    .iter()
                    .any(|i| matches!(i, IrInst::LoadLocal { slot, .. } if *slot == iv_slot))
        });

        let mut new_hdr_insts: Vec<IrInst> = needed_body_insts
            .into_iter()
            .map(|ii| body_bb.insts[ii].clone())
            .collect();
        let u16t = IrType::Int(IntType::U16);
        let eff_dst_base = if let Some(k) = dst_extra_off {
            let d_v = func.next_val;
            func.next_val += 1;
            new_hdr_insts.push(IrInst::BinOp {
                dst: d_v,
                op: IrBinOp::Add,
                mode: ArithMode::Wrapping,
                ty: u16t.clone(),
                lhs: dst_base,
                rhs: k,
                site_id: 0,
            });
            d_v
        } else {
            dst_base
        };
        let bulk_inst = match bulk_kind {
            BulkKind::Copy {
                src_base,
                src_extra_off,
            } => {
                let eff_src_base = if let Some(k) = src_extra_off {
                    let s_v = func.next_val;
                    func.next_val += 1;
                    new_hdr_insts.push(IrInst::BinOp {
                        dst: s_v,
                        op: IrBinOp::Add,
                        mode: ArithMode::Wrapping,
                        ty: u16t,
                        lhs: src_base,
                        rhs: k,
                        site_id: 0,
                    });
                    s_v
                } else {
                    src_base
                };
                IrInst::MemCopy {
                    dst_addr: eff_dst_base,
                    src_addr: eff_src_base,
                    len: trip_count as u16,
                }
            }
            BulkKind::Set { val } => IrInst::MemSet {
                dst_addr: eff_dst_base,
                val,
                len: trip_count as u16,
            },
        };
        new_hdr_insts.push(bulk_inst);
        if iv_read_after {
            let n_v = func.next_val;
            func.next_val += 1;
            let iv_ty = IrType::Int(*iv_it);
            new_hdr_insts.push(IrInst::ConstInt {
                dst: n_v,
                ty: iv_ty.clone(),
                val: trip_count,
            });
            new_hdr_insts.push(IrInst::StoreLocal {
                slot: iv_slot,
                val: n_v,
                ty: iv_ty,
            });
        }
        func.blocks[hdr_idx].insts = new_hdr_insts;
        func.blocks[hdr_idx].term = Terminator::Goto(exit_bb);
        any_changed = true;
    }

    if any_changed {
        remove_dead_unreachable_blocks(func);
        remove_dead_pure_insts(func);
    }
    any_changed
}

/// Threads short-circuit boolean join blocks (`tmp = false/cond2; goto join; join: v = LoadLocal(tmp); Branch(v, T, F)`)
/// directly into branches/gotos so `while a && b` exposes clean loop guard blocks to OSR and countdown passes.
pub fn thread_short_circuit_joins(func: &mut IrFunction) -> bool {
    if func.blocks.is_empty()
        || func
            .blocks
            .iter()
            .any(|b| b.insts.iter().any(|i| matches!(i, IrInst::Phi { .. })))
    {
        return false;
    }
    let mut any_changed = false;
    for _ in 0..16 {
        let cfg = Cfg::build(func);
        let Some(entry_id) = cfg.entry_block() else {
            break;
        };
        let mut const_bools: HashMap<ValueId, bool> = HashMap::new();
        let mut load_counts: HashMap<SlotId, usize> = HashMap::new();
        let mut addr_taken: HashSet<SlotId> = HashSet::new();
        let mut val_uses: HashMap<ValueId, usize> = HashMap::new();
        for bb in &func.blocks {
            for inst in &bb.insts {
                if let IrInst::ConstBool { dst, val } = inst {
                    const_bools.insert(*dst, *val);
                }
                if let IrInst::LoadLocal { slot, .. } = inst {
                    *load_counts.entry(*slot).or_insert(0) += 1;
                }
                if let IrInst::AddrLocal { slot, .. } = inst {
                    addr_taken.insert(*slot);
                }
                inst.for_each_use(|v| *val_uses.entry(v).or_insert(0) += 1);
            }
            if let Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } = bb.term {
                *val_uses.entry(v).or_insert(0) += 1;
            }
        }
        let bb_map: HashMap<BlockId, usize> =
            func.blocks.iter().enumerate().map(|(i, b)| (b.id, i)).collect();

        let mut target_join: Option<(BlockId, SlotId, BlockId, BlockId, Vec<usize>)> = None;
        for bb in &func.blocks {
            if bb.id == entry_id || bb.insts.len() != 1 {
                continue;
            }
            let IrInst::LoadLocal {
                dst: v,
                slot: tmp,
                ty: IrType::Bool,
            } = bb.insts[0]
            else {
                continue;
            };
            let Terminator::Branch {
                cond,
                then_bb,
                else_bb,
            } = bb.term
            else {
                continue;
            };
            if cond != v
                || then_bb == bb.id
                || else_bb == bb.id
                || addr_taken.contains(&tmp)
                || load_counts.get(&tmp).copied() != Some(1)
                || val_uses.get(&v).copied() != Some(1)
            {
                continue;
            }
            let preds = cfg.reachable_preds(bb.id);
            if preds.is_empty() {
                continue;
            }
            let mut pred_idxs = Vec::with_capacity(preds.len());
            let mut ok = true;
            for p in preds {
                if p == bb.id {
                    ok = false;
                    break;
                }
                let Some(&pi) = bb_map.get(&p) else {
                    ok = false;
                    break;
                };
                if !matches!(func.blocks[pi].term, Terminator::Goto(t) if t == bb.id) {
                    ok = false;
                    break;
                }
                if !matches!(
                    func.blocks[pi].insts.last(),
                    Some(IrInst::StoreLocal { slot, .. }) if *slot == tmp
                ) {
                    ok = false;
                    break;
                }
                pred_idxs.push(pi);
            }
            if ok {
                target_join = Some((bb.id, tmp, then_bb, else_bb, pred_idxs));
                break;
            }
        }
        let Some((_join_id, _tmp, then_bb, else_bb, pred_idxs)) = target_join else {
            break;
        };
        for pi in pred_idxs {
            let Some(IrInst::StoreLocal { val, .. }) = func.blocks[pi].insts.pop() else {
                unreachable!();
            };
            func.blocks[pi].term = if let Some(&b) = const_bools.get(&val) {
                Terminator::Goto(if b { then_bb } else { else_bb })
            } else {
                Terminator::Branch {
                    cond: val,
                    then_bb,
                    else_bb,
                }
            };
        }
        // Thread empty Goto trampolines created by folding constant StoreLocal branches.
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
                    Terminator::Goto(t) => *t = resolve(*t),
                    Terminator::Branch {
                        then_bb: tb,
                        else_bb: eb,
                        ..
                    } => {
                        *tb = resolve(*tb);
                        *eb = resolve(*eb);
                    }
                    _ => {}
                }
            }
        }
        remove_dead_unreachable_blocks(func);
        remove_dead_pure_insts(func);
        any_changed = true;
    }
    any_changed
}

/// Lowers call-free byte-slice / byte-array copy loops with a `u8` column cap (`col < limit`,
/// `limit <= 255`) or an early byte-sentinel exit (e.g. NUL-terminated `if ch == 0 { break/return; }`)
/// into a canonical walking-pointer countdown loop (`src_ptr`, `dst_ptr`, `cnt`) so Z80 codegen
/// keeps `HL = src`, `DE = dst`, `B/BC = cnt`, and `A = ch` without per-character RAM spills.
pub fn osr_byte_copy_loops(
    func: &mut IrFunction,
    globals: &[IrGlobal],
    is_release: bool,
) -> bool {
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
        let mut loops: Vec<(BlockId, Vec<BlockId>, BTreeSet<BlockId>)> = by_header
            .into_iter()
            .filter_map(|(h, (latches, body))| {
                (h != entry_id
                    && (1..=2).contains(&latches.len())
                    && !processed_headers.contains(&h))
                .then_some((h, latches, body))
            })
            .collect();
        loops.sort_by_key(|(h, _, body)| (body.len(), *h));
        let Some((header, latches, body)) = loops.into_iter().next() else {
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
        let Some(&pg_idx) = bb_idx_map.get(&pre_or_guard_id) else {
            continue;
        };
        let pg_edge_ok = match func.blocks[pg_idx].term {
            Terminator::Goto(t) => t == header,
            Terminator::Branch {
                then_bb, else_bb, ..
            } => (then_bb == header) ^ (else_bb == header),
            _ => false,
        };
        if !pg_edge_ok {
            continue;
        }

        let escaped = func.escaped_locals();
        let mut consts: HashMap<ValueId, i64> = HashMap::new();
        let mut def_bb: HashMap<ValueId, BlockId> = HashMap::new();
        let mut all_defs: HashMap<ValueId, IrInst> = HashMap::new();
        let mut addr_local_of: HashMap<ValueId, SlotId> = HashMap::new();
        let mut val_uses: HashMap<ValueId, usize> = HashMap::new();
        for bb in &func.blocks {
            for inst in &bb.insts {
                if let Some(d) = inst.dst() {
                    def_bb.insert(d, bb.id);
                    all_defs.insert(d, inst.clone());
                }
                match inst {
                    IrInst::ConstInt { dst, val, .. } => {
                        consts.insert(*dst, *val);
                    }
                    IrInst::AddrLocal { dst, slot } => {
                        addr_local_of.insert(*dst, *slot);
                    }
                    _ => {}
                }
                inst.for_each_use(|v| *val_uses.entry(v).or_insert(0) += 1);
            }
            if let Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } = bb.term {
                *val_uses.entry(v).or_insert(0) += 1;
            }
        }

        // Verify no value defined inside `body` is used outside `body`.
        let mut body_defs: HashSet<ValueId> = HashSet::new();
        let mut body_inst_by_dst: HashMap<ValueId, (BlockId, usize, IrInst)> = HashMap::new();
        for &bid in &body {
            if let Some(&bi) = bb_idx_map.get(&bid) {
                for (ii, inst) in func.blocks[bi].insts.iter().enumerate() {
                    if let Some(d) = inst.dst() {
                        body_defs.insert(d);
                        body_inst_by_dst.insert(d, (bid, ii, inst.clone()));
                    }
                }
            }
        }
        let used_outside = func.blocks.iter().any(|bb| {
            !body.contains(&bb.id)
                && (bb.insts.iter().any(|inst| {
                    let mut hit = false;
                    inst.for_each_use(|v| {
                        if body_defs.contains(&v) {
                            hit = true;
                        }
                    });
                    hit
                }) || matches!(
                    bb.term,
                    Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. }
                    if body_defs.contains(&v)
                ))
        });
        if used_outside {
            continue;
        }

        // Validate allowed instructions inside `body` and locate the single 1B LoadPtr and StorePtr.
        let mut bad_inst = false;
        let mut elem_load: Option<(BlockId, usize, ValueId, ValueId, IrType)> = None;
        let mut elem_store: Option<(BlockId, usize, ValueId, ValueId, IrType)> = None;
        let mut slot_stores: HashMap<SlotId, Vec<(BlockId, usize, ValueId)>> = HashMap::new();
        let mut slot_loads: HashMap<SlotId, Vec<(BlockId, usize, ValueId)>> = HashMap::new();
        let mut local_ptr_stored_in_body: HashSet<SlotId> = HashSet::new();

        for &bid in &body {
            let Some(&bi) = bb_idx_map.get(&bid) else {
                bad_inst = true;
                break;
            };
            for (ii, inst) in func.blocks[bi].insts.iter().enumerate() {
                match inst {
                    IrInst::ConstInt { .. }
                    | IrInst::ConstBool { .. }
                    | IrInst::AddrLocal { .. }
                    | IrInst::AddrGlobal { .. }
                    | IrInst::FieldAddr { .. }
                    | IrInst::SlicePtr { .. }
                    | IrInst::SliceLen { .. }
                    | IrInst::Cast { .. }
                    | IrInst::UnOp { .. }
                    | IrInst::BinOp { .. } => {}
                    IrInst::IndexAddr { elem_size, .. } => {
                        if *elem_size != 1 {
                            bad_inst = true;
                            break;
                        }
                    }
                    IrInst::LoadLocal { dst, slot, .. } => {
                        slot_loads.entry(*slot).or_default().push((bid, ii, *dst));
                    }
                    IrInst::StoreLocal { slot, val, .. } => {
                        if (*slot as usize) < escaped.len() && escaped[*slot as usize] {
                            bad_inst = true;
                            break;
                        }
                        slot_stores.entry(*slot).or_default().push((bid, ii, *val));
                    }
                    IrInst::LoadPtr {
                        dst,
                        addr,
                        ty,
                        volatile: false,
                    } => {
                        if matches!(ty, IrType::Slice { .. } | IrType::ConstSlice(_))
                            && addr_local_of.contains_key(addr)
                        {
                            // Invariant slice parameter/local load
                        } else if matches!(ty, IrType::Int(IntType::U8 | IntType::I8) | IrType::Bool)
                        {
                            if elem_load.is_some() {
                                bad_inst = true;
                                break;
                            }
                            elem_load = Some((bid, ii, *dst, *addr, ty.clone()));
                        } else {
                            bad_inst = true;
                            break;
                        }
                    }
                    IrInst::StorePtr {
                        addr,
                        val,
                        ty,
                        volatile: false,
                    } if matches!(ty, IrType::Int(IntType::U8 | IntType::I8) | IrType::Bool) => {
                        if let Some(&slot) = addr_local_of.get(addr) {
                            local_ptr_stored_in_body.insert(slot);
                        }
                        if elem_store.is_some() {
                            bad_inst = true;
                            break;
                        }
                        elem_store = Some((bid, ii, *addr, *val, ty.clone()));
                    }
                    _ => {
                        bad_inst = true;
                        break;
                    }
                }
            }
            if bad_inst {
                break;
            }
        }
        if bad_inst {
            continue;
        }
        let (Some((load_bid, load_ii, load_dst, src_addr, load_ty)), Some((store_bid, _store_ii, dst_addr, store_val, store_ty))) =
            (elem_load, elem_store)
        else {
            continue;
        };

        // Identify +1 induction variable(s) updated once in a loop latch (or pre-join step block),
        // optionally with a second early-exit assignment (`iv = bound`) in another block of `body`.
        let latch_set: HashSet<BlockId> = latches.iter().copied().collect();
        let peel_empty_to_header = |mut cur: BlockId| -> Option<Vec<BlockId>> {
            let mut trail = Vec::new();
            let mut seen = HashSet::new();
            while cur != header {
                if !body.contains(&cur) || !seen.insert(cur) {
                    return None;
                }
                let &bi = bb_idx_map.get(&cur)?;
                let bb = &func.blocks[bi];
                if !bb.insts.is_empty() {
                    return None;
                }
                let Terminator::Goto(next) = bb.term else {
                    return None;
                };
                trail.push(cur);
                cur = next;
            }
            Some(trail)
        };
        let reaches_latch_via_empty = |bid: BlockId| -> bool {
            if latch_set.contains(&bid) {
                return true;
            }
            let Some(&bi) = bb_idx_map.get(&bid) else {
                return false;
            };
            let Terminator::Goto(next) = func.blocks[bi].term else {
                return false;
            };
            peel_empty_to_header(next).is_some()
        };

        type IvCandInfo = (BlockId, Option<(BlockId, usize, ValueId)>);
        let mut iv_cands_info: HashMap<SlotId, IvCandInfo> = HashMap::new();
        for (&slot, stores) in &slot_stores {
            if !(1..=2).contains(&stores.len())
                || (slot as usize) >= escaped.len()
                || escaped[slot as usize]
            {
                continue;
            }
            if !matches!(
                func.locals.get(slot as usize).map(|l| &l.ty),
                Some(IrType::Int(IntType::U8 | IntType::U16))
            ) {
                continue;
            }
            let is_step_one = |sbid: BlockId, sval: ValueId| -> bool {
                if !reaches_latch_via_empty(sbid) {
                    return false;
                }
                let Some((_, _, IrInst::BinOp {
                    op: IrBinOp::Add,
                    mode,
                    ty,
                    lhs,
                    rhs,
                    ..
                })) = body_inst_by_dst.get(&sval)
                else {
                    return false;
                };
                if *mode == ArithMode::Saturating
                    || !matches!(ty, IrType::Int(IntType::U8 | IntType::U16))
                {
                    return false;
                }
                let is_slot_load = |v: ValueId| {
                    matches!(
                        body_inst_by_dst.get(&v),
                        Some((lb, _, IrInst::LoadLocal { slot: ls, .. }))
                            if *lb == sbid && *ls == slot
                    )
                };
                (is_slot_load(*lhs) && consts.get(rhs) == Some(&1))
                    || (is_slot_load(*rhs) && consts.get(lhs) == Some(&1))
            };
            if stores.len() == 1 {
                let (sbid, _, sval) = stores[0];
                if is_step_one(sbid, sval) {
                    iv_cands_info.insert(slot, (sbid, None));
                }
            } else {
                let (s0, s1) = (stores[0], stores[1]);
                if is_step_one(s0.0, s0.2) && s1.0 != s0.0 && reaches_latch_via_empty(s1.0) {
                    iv_cands_info.insert(slot, (s0.0, Some(s1)));
                } else if is_step_one(s1.0, s1.2)
                    && s0.0 != s1.0
                    && reaches_latch_via_empty(s0.0)
                {
                    iv_cands_info.insert(slot, (s1.0, Some(s0)));
                }
            }
        }
        let iv_cands: HashSet<SlotId> = iv_cands_info.keys().copied().collect();
        if iv_cands.is_empty() {
            continue;
        }

        // Helper: extract `(base, extra_off, addr_len, idx_slot)` from an `IndexAddr` or `base + (idx as u16)` address.
        let extract_indexed = |addr_v: ValueId| -> Option<(ValueId, Option<ValueId>, u16, SlotId)> {
            let unwrap_idx_slot = |iv: ValueId| -> Option<SlotId> {
                let mut cur = iv;
                if let Some((_, _, IrInst::Cast {
                    val,
                    from_ty: IrType::Int(IntType::U8),
                    to_ty: IrType::Int(IntType::U16),
                    ..
                })) = body_inst_by_dst.get(&cur)
                {
                    cur = *val;
                }
                match body_inst_by_dst.get(&cur) {
                    Some((_, _, IrInst::LoadLocal { slot, .. })) if iv_cands.contains(slot) => {
                        Some(*slot)
                    }
                    _ => None,
                }
            };
            match body_inst_by_dst.get(&addr_v) {
                Some((_, _, IrInst::IndexAddr {
                    base,
                    index,
                    elem_size: 1,
                    len,
                    ..
                })) => unwrap_idx_slot(*index).map(|s| (*base, None, *len, s)),
                Some((_, _, IrInst::BinOp {
                    op: IrBinOp::Add,
                    mode,
                    ty: IrType::Int(IntType::U16),
                    lhs,
                    rhs,
                    ..
                })) if *mode != ArithMode::Saturating => {
                    if let Some(s) = unwrap_idx_slot(*rhs) {
                        Some((*lhs, None, 0, s))
                    } else if let Some(s) = unwrap_idx_slot(*lhs) {
                        Some((*rhs, None, 0, s))
                    } else {
                        let match_inner = |cand: ValueId| -> Option<(ValueId, SlotId)> {
                            if let Some((_, _, IrInst::BinOp {
                                op: IrBinOp::Add,
                                mode: m2,
                                ty: IrType::Int(IntType::U16),
                                lhs: l2,
                                rhs: r2,
                                ..
                            })) = body_inst_by_dst.get(&cand)
                                && *m2 != ArithMode::Saturating
                            {
                                if let Some(s) = unwrap_idx_slot(*r2) {
                                    return Some((*l2, s));
                                }
                                if let Some(s) = unwrap_idx_slot(*l2) {
                                    return Some((*r2, s));
                                }
                            }
                            None
                        };
                        if let Some((k, s)) = match_inner(*rhs) {
                            Some((*lhs, Some(k), 0, s))
                        } else if let Some((k, s)) = match_inner(*lhs) {
                            Some((*rhs, Some(k), 0, s))
                        } else {
                            None
                        }
                    }
                }
                _ => None,
            }
        };

        let Some((src_base, src_extra_off, src_addr_len, iv_slot)) = extract_indexed(src_addr)
        else {
            continue;
        };
        if !is_release && src_addr_len > 0 {
            continue;
        }
        if find_loop_iv_init(func, &cfg, pre_or_guard_id, iv_slot, &consts) != Some(0) {
            continue;
        }
        let Some((dst_base, dst_extra_off, dst_addr_len, dst_idx_slot)) = extract_indexed(dst_addr)
        else {
            continue;
        };
        let col_slot = if dst_idx_slot == iv_slot {
            None
        } else {
            Some(dst_idx_slot)
        };
        let (step_store_bid, iv_early_assign) = iv_cands_info[&iv_slot];
        if let Some(cs) = col_slot {
            let (col_step_bid, col_early) = iv_cands_info[&cs];
            if col_early.is_some() || col_step_bid != step_store_bid {
                continue;
            }
        }

        // Any other `StoreLocal` in `body` may only be a single-iteration temporary holding `load_dst`.
        let mut ch_slot_opt: Option<SlotId> = None;
        let mut bad_stores = false;
        for (&slot, stores) in &slot_stores {
            if slot == iv_slot || Some(slot) == col_slot {
                continue;
            }
            if stores.len() != 1 || ch_slot_opt.is_some() {
                bad_stores = true;
                break;
            }
            let (sbid, sii, sval) = stores[0];
            if sbid != load_bid || sii <= load_ii || sval != load_dst {
                bad_stores = true;
                break;
            }
            let all_loads_after = slot_loads.get(&slot).is_some_and(|ls| {
                ls.iter().all(|&(lbid, lii, _)| {
                    (lbid == load_bid && lii > sii) || (lbid == store_bid && lbid != load_bid)
                })
            });
            if !all_loads_after {
                bad_stores = true;
                break;
            }
            ch_slot_opt = Some(slot);
        }
        if bad_stores {
            continue;
        }

        // Verify `store_val` is `load_dst` (or `LoadLocal(ch_slot)`).
        let is_ch_val = |v: ValueId| -> bool {
            let mut cur = v;
            if let Some((_, _, IrInst::Cast {
                val,
                from_ty,
                to_ty,
                ..
            })) = body_inst_by_dst.get(&cur)
                && matches!(
                    from_ty,
                    IrType::Int(IntType::U8 | IntType::I8) | IrType::Bool
                )
                && matches!(
                    to_ty,
                    IrType::Int(IntType::U8 | IntType::I8) | IrType::Bool
                )
            {
                cur = *val;
            }
            cur == load_dst
                || ch_slot_opt.is_some_and(|cs| {
                    matches!(
                        body_inst_by_dst.get(&cur),
                        Some((_, _, IrInst::LoadLocal { slot, .. })) if *slot == cs
                    )
                })
        };
        if !is_ch_val(store_val) {
            continue;
        }

        // Walk CFG inside `body`: guard_blocks* -> load_bid [-> store_bid] -> header.
        struct GuardInfo {
            slot: SlotId,
            limit_val: ValueId,
            inclusive: bool,
            exit_bb: BlockId,
        }
        let mut visited_body: HashSet<BlockId> = HashSet::new();
        let mut guards: Vec<GuardInfo> = Vec::new();
        let mut cur_bid = header;
        let mut cfg_ok = true;

        while cur_bid != load_bid {
            if !visited_body.insert(cur_bid) {
                cfg_ok = false;
                break;
            }
            let bb = &func.blocks[bb_idx_map[&cur_bid]];
            let pure_guard = bb.insts.iter().all(|inst| {
                matches!(
                    inst,
                    IrInst::ConstInt { .. }
                        | IrInst::ConstBool { .. }
                        | IrInst::LoadLocal { .. }
                        | IrInst::LoadPtr { volatile: false, .. }
                        | IrInst::SliceLen { .. }
                        | IrInst::Cast { .. }
                        | IrInst::BinOp {
                            op: IrBinOp::Lt | IrBinOp::Le | IrBinOp::Gt | IrBinOp::Ge,
                            ..
                        }
                )
            });
            if !pure_guard {
                cfg_ok = false;
                break;
            }
            let Terminator::Branch {
                cond,
                then_bb,
                else_bb,
            } = bb.term
            else {
                cfg_ok = false;
                break;
            };
            let (next_bb, exit_bb, cont_on_true) =
                if body.contains(&then_bb) && !body.contains(&else_bb) {
                    (then_bb, else_bb, true)
                } else if body.contains(&else_bb) && !body.contains(&then_bb) {
                    (else_bb, then_bb, false)
                } else {
                    cfg_ok = false;
                    break;
                };
            let Some(IrInst::BinOp {
                dst: cmp_dst,
                op,
                ty: IrType::Int(IntType::U8 | IntType::U16),
                lhs,
                rhs,
                ..
            }) = bb.insts.last()
            else {
                cfg_ok = false;
                break;
            };
            if *cmp_dst != cond {
                cfg_ok = false;
                break;
            }
            // Normalize condition when continuing into loop (`cont_on_true == true` vs `false`):
            let eff_op = if cont_on_true {
                *op
            } else {
                match *op {
                    IrBinOp::Lt => IrBinOp::Ge,
                    IrBinOp::Le => IrBinOp::Gt,
                    IrBinOp::Gt => IrBinOp::Le,
                    IrBinOp::Ge => IrBinOp::Lt,
                    _ => unreachable!(),
                }
            };
            let slot_of_load = |v: ValueId| -> Option<SlotId> {
                match body_inst_by_dst.get(&v) {
                    Some((_, _, IrInst::LoadLocal { slot, .. })) => Some(*slot),
                    _ => None,
                }
            };
            let matched_guard = match eff_op {
                IrBinOp::Lt | IrBinOp::Le => slot_of_load(*lhs)
                    .filter(|&s| s == iv_slot || Some(s) == col_slot)
                    .map(|s| (s, *rhs, eff_op == IrBinOp::Le)),
                IrBinOp::Gt | IrBinOp::Ge => slot_of_load(*rhs)
                    .filter(|&s| s == iv_slot || Some(s) == col_slot)
                    .map(|s| (s, *lhs, eff_op == IrBinOp::Ge)),
                _ => None,
            };
            let Some((g_slot, g_limit, inclusive)) = matched_guard else {
                cfg_ok = false;
                break;
            };
            guards.push(GuardInfo {
                slot: g_slot,
                limit_val: g_limit,
                inclusive,
                exit_bb,
            });
            cur_bid = next_bb;
        }
        if !cfg_ok || !visited_body.insert(load_bid) {
            continue;
        }

        // Check `load_bid` -> optional sentinel exit (`break`/`return` or `iv = bound` idiom)
        // -> `store_bid` -> (optional empty join) -> `header`.
        // When `sentinel_exit_bb` is `None`, the sentinel exit uses `iv = bound` and exits to `iv_g.exit_bb`.
        let mut sentinel_info: Option<(u8, bool, Option<BlockId>)> = None;
        let mut sentinel_early_assign_val: Option<ValueId> = None;
        let load_bb = &func.blocks[bb_idx_map[&load_bid]];
        if load_bid == store_bid {
            if iv_early_assign.is_some()
                || latches.len() != 1
                || load_bid != latches[0]
                || step_store_bid != load_bid
                || !matches!(load_bb.term, Terminator::Goto(t) if t == header)
            {
                continue;
            }
        } else {
            if step_store_bid != store_bid || !visited_body.insert(store_bid) {
                continue;
            }
            let store_bb = &func.blocks[bb_idx_map[&store_bid]];
            let Terminator::Goto(store_next) = store_bb.term else {
                continue;
            };
            let Some(store_trail) = peel_empty_to_header(store_next) else {
                continue;
            };
            if store_trail.iter().any(|b| visited_body.contains(b)) {
                continue;
            }
            let Terminator::Branch {
                cond,
                then_bb,
                else_bb,
            } = load_bb.term
            else {
                continue;
            };
            let (cont_bb, other_bb, cont_on_true) = if then_bb == store_bid {
                (then_bb, else_bb, true)
            } else if else_bb == store_bid {
                (else_bb, then_bb, false)
            } else {
                continue;
            };
            if cont_bb != store_bid {
                continue;
            }
            let Some(IrInst::BinOp {
                dst: cmp_dst,
                op: cmp_op @ (IrBinOp::Eq | IrBinOp::Ne),
                ty: cmp_ty,
                lhs,
                rhs,
                ..
            }) = load_bb.insts.last()
            else {
                continue;
            };
            if *cmp_dst != cond
                || !matches!(cmp_ty, IrType::Int(IntType::U8 | IntType::I8))
            {
                continue;
            }
            let sval = if is_ch_val(*lhs) {
                consts.get(rhs).copied()
            } else if is_ch_val(*rhs) {
                consts.get(lhs).copied()
            } else {
                None
            };
            let Some(sval) = sval.filter(|&v| (0..=255).contains(&v)) else {
                continue;
            };
            let exit_on_eq = if cont_on_true {
                *cmp_op == IrBinOp::Ne
            } else {
                *cmp_op == IrBinOp::Eq
            };
            if !body.contains(&other_bb) {
                if iv_early_assign.is_some()
                    || latches.len() != 1
                    || (!store_trail.is_empty() && store_trail.last() != Some(&latches[0]))
                    || (store_trail.is_empty() && store_bid != latches[0])
                {
                    continue;
                }
                visited_body.extend(store_trail);
                sentinel_info = Some((sval as u8, exit_on_eq, Some(other_bb)));
            } else {
                let early_bb = other_bb;
                if early_bb == store_bid || !visited_body.insert(early_bb) {
                    continue;
                }
                let Some((e_bid, e_ii, assigned_val)) = iv_early_assign else {
                    continue;
                };
                if e_bid != early_bb {
                    continue;
                }
                let early_block = &func.blocks[bb_idx_map[&early_bb]];
                let early_insts_ok = early_block.insts.iter().enumerate().all(|(ii, inst)| {
                    if ii == e_ii {
                        matches!(
                            inst,
                            IrInst::StoreLocal { slot, val, .. }
                                if *slot == iv_slot && *val == assigned_val
                        )
                    } else {
                        match inst {
                            IrInst::ConstInt { .. }
                            | IrInst::ConstBool { .. }
                            | IrInst::AddrLocal { .. }
                            | IrInst::AddrGlobal { .. }
                            | IrInst::FieldAddr { .. }
                            | IrInst::SlicePtr { .. }
                            | IrInst::SliceLen { .. }
                            | IrInst::Cast { .. }
                            | IrInst::UnOp { .. } => true,
                            IrInst::BinOp { mode, .. } => *mode != ArithMode::Saturating,
                            IrInst::LoadLocal { slot, .. } => {
                                *slot != iv_slot && Some(*slot) != col_slot
                            }
                            IrInst::LoadPtr {
                                addr,
                                volatile: false,
                                ..
                            } => addr_local_of.contains_key(addr),
                            _ => false,
                        }
                    }
                });
                if !early_insts_ok {
                    continue;
                }
                let Terminator::Goto(early_next) = early_block.term else {
                    continue;
                };
                let Some(early_trail) = peel_empty_to_header(early_next) else {
                    continue;
                };
                if early_trail.iter().any(|b| visited_body.contains(b)) {
                    continue;
                }
                visited_body.extend(store_trail);
                visited_body.extend(early_trail);
                sentinel_early_assign_val = Some(assigned_val);
                sentinel_info = Some((sval as u8, exit_on_eq, None));
            }
        }

        if visited_body.len() != body.len() {
            continue;
        }

        // Extract `iv_guard` (`i < len`) and optional `col_guard` (`col < limit`).
        let mut iv_guard: Option<&GuardInfo> = None;
        let mut col_guard: Option<&GuardInfo> = None;
        let mut dup_guard = false;
        for g in &guards {
            if g.slot == iv_slot {
                if iv_guard.is_some() {
                    dup_guard = true;
                    break;
                }
                iv_guard = Some(g);
            } else if Some(g.slot) == col_slot {
                if col_guard.is_some() {
                    dup_guard = true;
                    break;
                }
                col_guard = Some(g);
            }
        }
        let Some(iv_g) = iv_guard else {
            continue;
        };
        if dup_guard {
            continue;
        }
        if iv_g.inclusive {
            // Only handle strict `<` for runtime `len` unless constant `<= k`!
            if !consts.contains_key(&iv_g.limit_val) {
                continue;
            }
        }
        let is_bare_ret_none = |bid: BlockId| -> bool {
            bb_idx_map.get(&bid).is_some_and(|&bi| {
                func.blocks[bi].insts.is_empty()
                    && matches!(func.blocks[bi].term, Terminator::Return(None))
            })
        };
        if let Some(cg) = col_guard
            && cg.exit_bb != iv_g.exit_bb
            && !(is_bare_ret_none(cg.exit_bb) && is_bare_ret_none(iv_g.exit_bb))
        {
            continue;
        }
        let exit_bound_bb = iv_g.exit_bb;
        let sentinel_info: Option<(u8, bool, BlockId)> =
            sentinel_info.map(|(sval, exit_on_eq, es_opt)| {
                (sval, exit_on_eq, es_opt.unwrap_or(exit_bound_bb))
            });

        // Determine if `col_guard` provides a `u8` limit in `1..=255`.
        let col_u8_limit: Option<u8> = if let Some(cg) = col_guard {
            let is_u8_col = func.locals[cg.slot as usize].ty == IrType::Int(IntType::U8);
            if !is_u8_col {
                continue;
            }
            let Some(raw_lim) = consts.get(&cg.limit_val).copied() else {
                continue;
            };
            let excl = if cg.inclusive { raw_lim + 1 } else { raw_lim };
            if !(1..=255).contains(&excl) {
                continue;
            }
            Some(excl as u8)
        } else {
            None
        };

        // Check debug-mode fixed array bounds safety for `dst_addr_len > 0`.
        if !is_release
            && dst_addr_len > 0
            && col_u8_limit.is_none_or(|lim| (lim as u16) > dst_addr_len)
        {
            continue;
        }

        // Determine if `len` is a `u8` constant `<= 255` or `u8` type.
        let iv_ty = func.locals[iv_slot as usize].ty.clone();
        let iv_excl_const: Option<i64> = consts
            .get(&iv_g.limit_val)
            .copied()
            .map(|v| if iv_g.inclusive { v + 1 } else { v });
        if let Some(c) = iv_excl_const
            && c <= 0
        {
            continue;
        }

        // Preserve `z80_e2e_256b_aligned_hud_and_vram_copy_loops` (`G_HUD.line[c] = s[i]`,
        // no sentinel check, no `col_guard`, `u16` runtime length) for `inc ix` + `dec bc` + `ld h, 0xC1`.
        if col_u8_limit.is_none()
            && sentinel_info.is_none()
            && iv_excl_const.is_none_or(|c| c > 255)
        {
            continue;
        }

        // Check live-out constraints on all exits.
        let mut all_exits = vec![exit_bound_bb];
        if let Some((_, _, es)) = sentinel_info {
            if !all_exits.contains(&es) {
                all_exits.push(es);
            }
        }
        if all_exits
            .iter()
            .any(|&eb| slot_live_out_at_exit(func, &cfg, eb, iv_slot))
        {
            continue;
        }
        if let Some(cs) = ch_slot_opt
            && all_exits
                .iter()
                .any(|&eb| slot_live_out_at_exit(func, &cfg, eb, cs))
        {
            continue;
        }
        let col_live_out = col_slot.is_some_and(|cs| {
            all_exits
                .iter()
                .any(|&eb| slot_live_out_at_exit(func, &cfg, eb, cs))
        });

        // Verify all base/limit expressions can be materialized in `pre_or_guard_id`.
        fn can_mat_pre(
            v: ValueId,
            body: &BTreeSet<BlockId>,
            pre_id: BlockId,
            iv_slot: SlotId,
            col_slot: Option<SlotId>,
            cfg: &Cfg,
            def_bb: &HashMap<ValueId, BlockId>,
            body_inst_by_dst: &HashMap<ValueId, (BlockId, usize, IrInst)>,
            escaped: &[bool],
            slot_stores: &HashMap<SlotId, Vec<(BlockId, usize, ValueId)>>,
            addr_local_of: &HashMap<ValueId, SlotId>,
            local_ptr_stored_in_body: &HashSet<SlotId>,
        ) -> bool {
            if let Some(db) = def_bb.get(&v)
                && !body.contains(db)
            {
                return cfg.dominates(*db, pre_id);
            }
            let Some((_, _, inst)) = body_inst_by_dst.get(&v) else {
                return false;
            };
            match inst {
                IrInst::ConstInt { .. }
                | IrInst::ConstBool { .. }
                | IrInst::AddrGlobal { .. }
                | IrInst::AddrLocal { .. } => true,
                IrInst::LoadLocal { slot, .. } => {
                    *slot != iv_slot
                        && Some(*slot) != col_slot
                        && ((*slot as usize) >= escaped.len() || !escaped[*slot as usize])
                        && !slot_stores.contains_key(slot)
                }
                IrInst::LoadPtr {
                    addr,
                    volatile: false,
                    ..
                } => addr_local_of.get(addr).is_some_and(|&slot| {
                    slot != iv_slot
                        && Some(slot) != col_slot
                        && ((slot as usize) >= escaped.len() || !escaped[slot as usize])
                        && !slot_stores.contains_key(&slot)
                        && !local_ptr_stored_in_body.contains(&slot)
                        && can_mat_pre(
                            *addr,
                            body,
                            pre_id,
                            iv_slot,
                            col_slot,
                            cfg,
                            def_bb,
                            body_inst_by_dst,
                            escaped,
                            slot_stores,
                            addr_local_of,
                            local_ptr_stored_in_body,
                        )
                }),
                IrInst::FieldAddr { base, .. }
                | IrInst::SlicePtr { slice: base, .. }
                | IrInst::SliceLen { slice: base, .. } => can_mat_pre(
                    *base,
                    body,
                    pre_id,
                    iv_slot,
                    col_slot,
                    cfg,
                    def_bb,
                    body_inst_by_dst,
                    escaped,
                    slot_stores,
                    addr_local_of,
                    local_ptr_stored_in_body,
                ),
                IrInst::Cast { val, .. } | IrInst::UnOp { val, .. } => can_mat_pre(
                    *val,
                    body,
                    pre_id,
                    iv_slot,
                    col_slot,
                    cfg,
                    def_bb,
                    body_inst_by_dst,
                    escaped,
                    slot_stores,
                    addr_local_of,
                    local_ptr_stored_in_body,
                ),
                IrInst::BinOp {
                    mode, lhs, rhs, ..
                } if *mode != ArithMode::Saturating => {
                    can_mat_pre(
                        *lhs,
                        body,
                        pre_id,
                        iv_slot,
                        col_slot,
                        cfg,
                        def_bb,
                        body_inst_by_dst,
                        escaped,
                        slot_stores,
                        addr_local_of,
                        local_ptr_stored_in_body,
                    ) && can_mat_pre(
                        *rhs,
                        body,
                        pre_id,
                        iv_slot,
                        col_slot,
                        cfg,
                        def_bb,
                        body_inst_by_dst,
                        escaped,
                        slot_stores,
                        addr_local_of,
                        local_ptr_stored_in_body,
                    )
                }
                _ => false,
            }
        }

        let check_mat = |v: ValueId| {
            can_mat_pre(
                v,
                &body,
                pre_or_guard_id,
                iv_slot,
                col_slot,
                &cfg,
                &def_bb,
                &body_inst_by_dst,
                &escaped,
                &slot_stores,
                &addr_local_of,
                &local_ptr_stored_in_body,
            )
        };
        if !check_mat(src_base)
            || !src_extra_off.is_none_or(check_mat)
            || !check_mat(dst_base)
            || !dst_extra_off.is_none_or(check_mat)
            || !check_mat(iv_g.limit_val)
            || !sentinel_early_assign_val.is_none_or(check_mat)
        {
            continue;
        }

        // Verify `iv = assigned_val` provably fails `iv_g` immediately at the loop header.
        if let Some(assigned_val) = sentinel_early_assign_val {
            #[allow(clippy::too_many_arguments)]
            fn equiv_invariant_val(
                v1: ValueId,
                v2: ValueId,
                iv_slot: SlotId,
                col_slot: Option<SlotId>,
                all_defs: &HashMap<ValueId, IrInst>,
                consts: &HashMap<ValueId, i64>,
                escaped: &[bool],
                slot_stores: &HashMap<SlotId, Vec<(BlockId, usize, ValueId)>>,
                addr_local_of: &HashMap<ValueId, SlotId>,
                local_ptr_stored_in_body: &HashSet<SlotId>,
                depth: u8,
            ) -> bool {
                if depth > 8 {
                    return false;
                }
                if v1 == v2 {
                    return true;
                }
                if let (Some(c1), Some(c2)) = (consts.get(&v1), consts.get(&v2)) {
                    return c1 == c2;
                }
                match (all_defs.get(&v1), all_defs.get(&v2)) {
                    (
                        Some(IrInst::LoadLocal { slot: s1, .. }),
                        Some(IrInst::LoadLocal { slot: s2, .. }),
                    ) => {
                        *s1 == *s2
                            && *s1 != iv_slot
                            && Some(*s1) != col_slot
                            && ((*s1 as usize) >= escaped.len() || !escaped[*s1 as usize])
                            && !slot_stores.contains_key(s1)
                    }
                    (
                        Some(IrInst::LoadPtr {
                            addr: a1,
                            volatile: false,
                            ..
                        }),
                        Some(IrInst::LoadPtr {
                            addr: a2,
                            volatile: false,
                            ..
                        }),
                    ) => {
                        if let (Some(&s1), Some(&s2)) =
                            (addr_local_of.get(a1), addr_local_of.get(a2))
                        {
                            s1 == s2
                                && s1 != iv_slot
                                && Some(s1) != col_slot
                                && ((s1 as usize) >= escaped.len() || !escaped[s1 as usize])
                                && !slot_stores.contains_key(&s1)
                                && !local_ptr_stored_in_body.contains(&s1)
                        } else {
                            false
                        }
                    }
                    (
                        Some(IrInst::SliceLen { slice: sl1, .. }),
                        Some(IrInst::SliceLen { slice: sl2, .. }),
                    )
                    | (
                        Some(IrInst::SlicePtr { slice: sl1, .. }),
                        Some(IrInst::SlicePtr { slice: sl2, .. }),
                    ) => equiv_invariant_val(
                        *sl1,
                        *sl2,
                        iv_slot,
                        col_slot,
                        all_defs,
                        consts,
                        escaped,
                        slot_stores,
                        addr_local_of,
                        local_ptr_stored_in_body,
                        depth + 1,
                    ),
                    (
                        Some(IrInst::FieldAddr {
                            base: b1,
                            byte_offset: o1,
                            ..
                        }),
                        Some(IrInst::FieldAddr {
                            base: b2,
                            byte_offset: o2,
                            ..
                        }),
                    ) => {
                        o1 == o2
                            && equiv_invariant_val(
                                *b1,
                                *b2,
                                iv_slot,
                                col_slot,
                                all_defs,
                                consts,
                                escaped,
                                slot_stores,
                                addr_local_of,
                                local_ptr_stored_in_body,
                                depth + 1,
                            )
                    }
                    (
                        Some(IrInst::Cast {
                            val: a1,
                            from_ty: f1,
                            to_ty: t1,
                            ..
                        }),
                        Some(IrInst::Cast {
                            val: a2,
                            from_ty: f2,
                            to_ty: t2,
                            ..
                        }),
                    ) => {
                        f1 == f2
                            && t1 == t2
                            && equiv_invariant_val(
                                *a1,
                                *a2,
                                iv_slot,
                                col_slot,
                                all_defs,
                                consts,
                                escaped,
                                slot_stores,
                                addr_local_of,
                                local_ptr_stored_in_body,
                                depth + 1,
                            )
                    }
                    _ => false,
                }
            }

            let exits_immediately = if iv_g.inclusive {
                matches!(
                    (consts.get(&assigned_val), consts.get(&iv_g.limit_val)),
                    (Some(&c1), Some(&c2)) if c1 > c2
                )
            } else if let (Some(&c1), Some(&c2)) =
                (consts.get(&assigned_val), consts.get(&iv_g.limit_val))
            {
                c1 >= c2 && c2 >= 0
            } else {
                equiv_invariant_val(
                    assigned_val,
                    iv_g.limit_val,
                    iv_slot,
                    col_slot,
                    &all_defs,
                    &consts,
                    &escaped,
                    &slot_stores,
                    &addr_local_of,
                    &local_ptr_stored_in_body,
                    0,
                )
            };
            if !exits_immediately {
                continue;
            }
        }

        // Check `dst_base` alignment when `col_u8_limit` is active (`col_0 < limit <= 255`)
        // or when indexing by `iv` directly with a proven constant bound `<= 255`.
        let dst_page_bounded = col_u8_limit.is_some()
            || (col_slot.is_none() && iv_excl_const.is_some_and(|c| (1..=255).contains(&c)));
        let dst_aligned_kind = if dst_page_bounded && dst_extra_off.is_none() {
            fn resolve_aligned256_dst(
                defs: &HashMap<ValueId, IrInst>,
                globals: &[IrGlobal],
                consts: &HashMap<ValueId, i64>,
                mut v: ValueId,
            ) -> Option<&'static str> {
                let mut off = 0u32;
                for _ in 0..8 {
                    match defs.get(&v) {
                        Some(IrInst::ConstInt { val, .. }) => {
                            let addr = ((*val as u16) as u32).checked_add(off)?;
                            return (addr <= 0xFFFF && (addr & 0xFF) == 0).then_some("fixed");
                        }
                        Some(IrInst::AddrGlobal { global, .. }) => {
                            let g = globals.get(*global as usize)?;
                            if let Some(fa) = g.fixed_addr {
                                let addr = (fa as u32).checked_add(off)?;
                                return (addr <= 0xFFFF && (addr & 0xFF) == 0).then_some("fixed");
                            }
                            if !g.in_rom
                                && !g.is_extern
                                && g.fixed_addr.is_none()
                                && g.align >= 256
                                && (off & 0xFF) == 0
                            {
                                return Some("global");
                            }
                            return None;
                        }
                        Some(IrInst::FieldAddr {
                            base, byte_offset, ..
                        }) => {
                            off = off.checked_add(*byte_offset as u32)?;
                            v = *base;
                        }
                        Some(IrInst::Cast {
                            val,
                            from_ty,
                            to_ty,
                            ..
                        }) if matches!(
                            from_ty,
                            IrType::Int(IntType::U16 | IntType::I16) | IrType::Ref { .. }
                        ) && matches!(
                            to_ty,
                            IrType::Int(IntType::U16 | IntType::I16) | IrType::Ref { .. }
                        ) =>
                        {
                            v = *val;
                        }
                        Some(IrInst::BinOp {
                            op: IrBinOp::Add,
                            mode,
                            lhs,
                            rhs,
                            ..
                        }) if *mode != ArithMode::Saturating => {
                            if let Some(&c) = consts.get(rhs) {
                                off = off.checked_add((c as u16) as u32)?;
                                v = *lhs;
                            } else if let Some(&c) = consts.get(lhs) {
                                off = off.checked_add((c as u16) as u32)?;
                                v = *rhs;
                            } else {
                                return None;
                            }
                        }
                        _ => return None,
                    }
                }
                None
            }
            resolve_aligned256_dst(&all_defs, globals, &consts, dst_base)
        } else {
            None
        };

        // --- All checks passed! Perform the MIR transformation. ---
        let u8t = IrType::Int(IntType::U8);
        let u16t = IrType::Int(IntType::U16);
        let is_u8_cnt = col_u8_limit.is_some()
            || iv_ty == u8t
            || iv_excl_const.is_some_and(|c| (1..=255).contains(&c));
        let cnt_ty = if is_u8_cnt { u8t.clone() } else { u16t.clone() };

        let mut next_block_id: BlockId = func.blocks.iter().map(|b| b.id + 1).max().unwrap_or(1);
        let mut alloc_block = || {
            let id = next_block_id;
            next_block_id += 1;
            id
        };
        let mut next_val = func.next_val;
        let mut alloc_val = || {
            let v = next_val;
            next_val += 1;
            v
        };
        let mut next_slot: SlotId = func.locals.iter().map(|l| l.id + 1).max().unwrap_or(0);
        let mut alloc_slot = |prefix: &str, ty: IrType| -> SlotId {
            let id = next_slot;
            next_slot += 1;
            func.locals.push(LocalSlot {
                id,
                name: format!("{prefix}_{id}"),
                ty,
                mutable: true,
            });
            id
        };

        let src_slot = alloc_slot("__bcl_src", u16t.clone());
        let dst_slot_prefix = match dst_aligned_kind {
            Some("fixed") => "__bcl_dst256_fixed",
            Some("global") => "__bcl_dst256_global",
            _ => "__bcl_dst",
        };
        let dst_slot = alloc_slot(dst_slot_prefix, u16t.clone());
        let cnt_slot = alloc_slot("__bcl_cnt", cnt_ty.clone());

        // Split preheader edge if `pre_or_guard_id` ends in a Branch.
        let pre_id = match &mut func.blocks[pg_idx].term {
            Terminator::Goto(t) if *t == header => pre_or_guard_id,
            Terminator::Branch {
                then_bb, else_bb, ..
            } => {
                let pid = alloc_block();
                if *then_bb == header {
                    *then_bb = pid;
                } else {
                    *else_bb = pid;
                }
                func.blocks.push(BasicBlock {
                    id: pid,
                    insts: Vec::new(),
                    term: Terminator::Goto(header),
                });
                pid
            }
            _ => unreachable!(),
        };

        let mut new_blocks: Vec<BasicBlock> = Vec::new();
        let mut pre_val_map: HashMap<ValueId, ValueId> = HashMap::new();
        fn mat_in_pre(
            v: ValueId,
            body_inst_by_dst: &HashMap<ValueId, (BlockId, usize, IrInst)>,
            val_map: &mut HashMap<ValueId, ValueId>,
            out: &mut Vec<IrInst>,
            alloc_val: &mut dyn FnMut() -> ValueId,
        ) -> ValueId {
            if let Some(&mapped) = val_map.get(&v) {
                return mapped;
            }
            let Some((_, _, inst)) = body_inst_by_dst.get(&v) else {
                return v;
            };
            let cloned = inst.clone();
            let mut deps = Vec::new();
            cloned.for_each_use(|u| deps.push(u));
            for u in deps {
                let mu = mat_in_pre(u, body_inst_by_dst, val_map, out, alloc_val);
                val_map.insert(u, mu);
            }
            let out_inst = clone_pure_inst_remapped(&cloned, val_map, alloc_val);
            let nd = out_inst.dst().unwrap();
            out.push(out_inst);
            nd
        }

        let mut pre_insts: Vec<IrInst> = Vec::new();
        let m_src_base = mat_in_pre(
            src_base,
            &body_inst_by_dst,
            &mut pre_val_map,
            &mut pre_insts,
            &mut alloc_val,
        );
        let src_init_v = if let Some(extra) = src_extra_off {
            let m_ex = mat_in_pre(
                extra,
                &body_inst_by_dst,
                &mut pre_val_map,
                &mut pre_insts,
                &mut alloc_val,
            );
            let sum = alloc_val();
            pre_insts.push(IrInst::BinOp {
                dst: sum,
                op: IrBinOp::Add,
                mode: ArithMode::Wrapping,
                ty: u16t.clone(),
                lhs: m_src_base,
                rhs: m_ex,
                site_id: 0,
            });
            sum
        } else {
            m_src_base
        };

        let m_dst_base = mat_in_pre(
            dst_base,
            &body_inst_by_dst,
            &mut pre_val_map,
            &mut pre_insts,
            &mut alloc_val,
        );
        let dst_base_full = if let Some(extra) = dst_extra_off {
            let m_ex = mat_in_pre(
                extra,
                &body_inst_by_dst,
                &mut pre_val_map,
                &mut pre_insts,
                &mut alloc_val,
            );
            let sum = alloc_val();
            pre_insts.push(IrInst::BinOp {
                dst: sum,
                op: IrBinOp::Add,
                mode: ArithMode::Wrapping,
                ty: u16t.clone(),
                lhs: m_dst_base,
                rhs: m_ex,
                site_id: 0,
            });
            sum
        } else {
            m_dst_base
        };

        let col0_u8_opt: Option<ValueId> = if let Some(cs) = col_slot {
            let c0 = alloc_val();
            let cty = func.locals[cs as usize].ty.clone();
            pre_insts.push(IrInst::LoadLocal {
                dst: c0,
                slot: cs,
                ty: cty.clone(),
            });
            Some(c0)
        } else {
            None
        };
        let dst_init_v = if let Some(c0) = col0_u8_opt {
            let cty = func.locals[col_slot.unwrap() as usize].ty.clone();
            let c0_u16 = if cty == u16t {
                c0
            } else {
                let w = alloc_val();
                pre_insts.push(IrInst::Cast {
                    dst: w,
                    val: c0,
                    from_ty: cty,
                    to_ty: u16t.clone(),
                });
                w
            };
            let sum = alloc_val();
            pre_insts.push(IrInst::BinOp {
                dst: sum,
                op: IrBinOp::Add,
                mode: ArithMode::Wrapping,
                ty: u16t.clone(),
                lhs: dst_base_full,
                rhs: c0_u16,
                site_id: 0,
            });
            sum
        } else {
            dst_base_full
        };

        let m_len_raw = mat_in_pre(
            iv_g.limit_val,
            &body_inst_by_dst,
            &mut pre_val_map,
            &mut pre_insts,
            &mut alloc_val,
        );
        let len_v = if iv_g.inclusive {
            let c = iv_excl_const.unwrap();
            let cv = alloc_val();
            pre_insts.push(IrInst::ConstInt {
                dst: cv,
                ty: iv_ty.clone(),
                val: c,
            });
            cv
        } else {
            m_len_raw
        };

        let hdr_id = alloc_block();
        let step_id = if sentinel_info.is_some() {
            alloc_block()
        } else {
            hdr_id
        };
        let enter_id = alloc_block();

        // Optional exit trampolines if `col_slot` is live-out after the loop.
        let mut mk_col_tramp = |real_exit: BlockId| -> BlockId {
            if !col_live_out {
                return real_exit;
            }
            let cs = col_slot.unwrap();
            let cty = func.locals[cs as usize].ty.clone();
            let tid = alloc_block();
            let p_end = alloc_val();
            let diff = alloc_val();
            let col_res = if cty == u16t {
                diff
            } else {
                alloc_val()
            };
            let mut t_insts = vec![
                IrInst::LoadLocal {
                    dst: p_end,
                    slot: dst_slot,
                    ty: u16t.clone(),
                },
                IrInst::BinOp {
                    dst: diff,
                    op: IrBinOp::Sub,
                    mode: ArithMode::Wrapping,
                    ty: u16t.clone(),
                    lhs: p_end,
                    rhs: dst_base_full,
                    site_id: 0,
                },
            ];
            if cty != u16t {
                t_insts.push(IrInst::Cast {
                    dst: col_res,
                    val: diff,
                    from_ty: u16t.clone(),
                    to_ty: cty.clone(),
                });
            }
            t_insts.push(IrInst::StoreLocal {
                slot: cs,
                val: col_res,
                ty: cty,
            });
            new_blocks.push(BasicBlock {
                id: tid,
                insts: t_insts,
                term: Terminator::Goto(real_exit),
            });
            tid
        };

        let exit_bound_tgt = mk_col_tramp(exit_bound_bb);
        let sentinel_tgt = sentinel_info.map(|(sval, exit_on_eq, es)| {
            let etgt = if es == exit_bound_bb {
                exit_bound_tgt
            } else {
                mk_col_tramp(es)
            };
            (sval, exit_on_eq, etgt)
        });

        // Build preheader entry guard + count calculation.
        let pre_idx = func.blocks.iter().position(|b| b.id == pre_id).unwrap();
        func.blocks[pre_idx].insts.extend(pre_insts);

        if let Some(lim) = col_u8_limit {
            let c0 = col0_u8_opt.unwrap();
            let lim_c = alloc_val();
            let col_ok = alloc_val();
            let len_chk_id = alloc_block();
            func.blocks[pre_idx].insts.extend([
                IrInst::ConstInt {
                    dst: lim_c,
                    ty: u8t.clone(),
                    val: lim as i64,
                },
                IrInst::BinOp {
                    dst: col_ok,
                    op: IrBinOp::Lt,
                    mode: ArithMode::Wrapping,
                    ty: u8t.clone(),
                    lhs: c0,
                    rhs: lim_c,
                    site_id: 0,
                },
            ]);
            func.blocks[pre_idx].term = Terminator::Branch {
                cond: col_ok,
                then_bb: len_chk_id,
                else_bb: exit_bound_bb,
            };

            let rem_col = alloc_val();
            let mut lc_insts = vec![IrInst::BinOp {
                dst: rem_col,
                op: IrBinOp::Sub,
                mode: ArithMode::Wrapping,
                ty: u8t.clone(),
                lhs: lim_c,
                rhs: c0,
                site_id: 0,
            }];

            let min_id = if iv_excl_const.is_some_and(|c| c > 0) {
                len_chk_id
            } else {
                let mid = alloc_block();
                let z_len = alloc_val();
                let len_nz = alloc_val();
                lc_insts.extend([
                    IrInst::ConstInt {
                        dst: z_len,
                        ty: iv_ty.clone(),
                        val: 0,
                    },
                    IrInst::BinOp {
                        dst: len_nz,
                        op: IrBinOp::Ne,
                        mode: ArithMode::Wrapping,
                        ty: iv_ty.clone(),
                        lhs: len_v,
                        rhs: z_len,
                        site_id: 0,
                    },
                ]);
                new_blocks.push(BasicBlock {
                    id: len_chk_id,
                    insts: std::mem::take(&mut lc_insts),
                    term: Terminator::Branch {
                        cond: len_nz,
                        then_bb: mid,
                        else_bb: exit_bound_bb,
                    },
                });
                mid
            };

            let rem_w = if iv_ty == u8t {
                rem_col
            } else {
                let w = alloc_val();
                lc_insts.push(IrInst::Cast {
                    dst: w,
                    val: rem_col,
                    from_ty: u8t.clone(),
                    to_ty: iv_ty.clone(),
                });
                w
            };
            let use_len = alloc_val();
            lc_insts.push(IrInst::BinOp {
                dst: use_len,
                op: IrBinOp::Lt,
                mode: ArithMode::Wrapping,
                ty: iv_ty.clone(),
                lhs: len_v,
                rhs: rem_w,
                site_id: 0,
            });
            let arm_len_id = alloc_block();
            let arm_rem_id = alloc_block();
            new_blocks.push(BasicBlock {
                id: min_id,
                insts: lc_insts,
                term: Terminator::Branch {
                    cond: use_len,
                    then_bb: arm_len_id,
                    else_bb: arm_rem_id,
                },
            });
            let len_u8 = if iv_ty == u8t {
                len_v
            } else {
                let trunc_v = alloc_val();
                new_blocks.push(BasicBlock {
                    id: arm_len_id,
                    insts: vec![
                        IrInst::Cast {
                            dst: trunc_v,
                            val: len_v,
                            from_ty: iv_ty.clone(),
                            to_ty: u8t.clone(),
                        },
                        IrInst::StoreLocal {
                            slot: cnt_slot,
                            val: trunc_v,
                            ty: u8t.clone(),
                        },
                    ],
                    term: Terminator::Goto(enter_id),
                });
                trunc_v
            };
            if iv_ty == u8t {
                new_blocks.push(BasicBlock {
                    id: arm_len_id,
                    insts: vec![IrInst::StoreLocal {
                        slot: cnt_slot,
                        val: len_u8,
                        ty: u8t.clone(),
                    }],
                    term: Terminator::Goto(enter_id),
                });
            }
            new_blocks.push(BasicBlock {
                id: arm_rem_id,
                insts: vec![IrInst::StoreLocal {
                    slot: cnt_slot,
                    val: rem_col,
                    ty: u8t.clone(),
                }],
                term: Terminator::Goto(enter_id),
            });
            new_blocks.push(BasicBlock {
                id: enter_id,
                insts: vec![
                    IrInst::StoreLocal {
                        slot: src_slot,
                        val: src_init_v,
                        ty: u16t.clone(),
                    },
                    IrInst::StoreLocal {
                        slot: dst_slot,
                        val: dst_init_v,
                        ty: u16t.clone(),
                    },
                ],
                term: Terminator::Goto(hdr_id),
            });
        } else {
            let cnt_init_v = if is_u8_cnt && iv_ty != u8t {
                let c_u8 = alloc_val();
                func.blocks[pre_idx].insts.push(IrInst::Cast {
                    dst: c_u8,
                    val: len_v,
                    from_ty: iv_ty.clone(),
                    to_ty: u8t.clone(),
                });
                c_u8
            } else {
                len_v
            };
            if iv_excl_const.is_some_and(|c| c > 0) {
                func.blocks[pre_idx].insts.extend([
                    IrInst::StoreLocal {
                        slot: src_slot,
                        val: src_init_v,
                        ty: u16t.clone(),
                    },
                    IrInst::StoreLocal {
                        slot: dst_slot,
                        val: dst_init_v,
                        ty: u16t.clone(),
                    },
                    IrInst::StoreLocal {
                        slot: cnt_slot,
                        val: cnt_init_v,
                        ty: cnt_ty.clone(),
                    },
                ]);
                func.blocks[pre_idx].term = Terminator::Goto(hdr_id);
            } else {
                let z_len = alloc_val();
                let len_nz = alloc_val();
                func.blocks[pre_idx].insts.extend([
                    IrInst::ConstInt {
                        dst: z_len,
                        ty: iv_ty.clone(),
                        val: 0,
                    },
                    IrInst::BinOp {
                        dst: len_nz,
                        op: IrBinOp::Ne,
                        mode: ArithMode::Wrapping,
                        ty: iv_ty.clone(),
                        lhs: len_v,
                        rhs: z_len,
                        site_id: 0,
                    },
                ]);
                func.blocks[pre_idx].term = Terminator::Branch {
                    cond: len_nz,
                    then_bb: enter_id,
                    else_bb: exit_bound_bb,
                };
                new_blocks.push(BasicBlock {
                    id: enter_id,
                    insts: vec![
                        IrInst::StoreLocal {
                            slot: src_slot,
                            val: src_init_v,
                            ty: u16t.clone(),
                        },
                        IrInst::StoreLocal {
                            slot: dst_slot,
                            val: dst_init_v,
                            ty: u16t.clone(),
                        },
                        IrInst::StoreLocal {
                            slot: cnt_slot,
                            val: cnt_init_v,
                            ty: cnt_ty.clone(),
                        },
                    ],
                    term: Terminator::Goto(hdr_id),
                });
            }
        }

        // Build canonical `hdr_id` and `step_id` loop blocks.
        let p_src = alloc_val();
        let ch = alloc_val();
        let p_dst = alloc_val();
        let s1 = alloc_val();
        let o1 = alloc_val();
        let sn = alloc_val();
        let d1 = alloc_val();
        let o2 = alloc_val();
        let dn = alloc_val();
        let c1 = alloc_val();
        let o3 = alloc_val();
        let cn = alloc_val();
        let z0 = alloc_val();
        let cnz = alloc_val();

        let step_insts = vec![
            IrInst::LoadLocal {
                dst: p_dst,
                slot: dst_slot,
                ty: u16t.clone(),
            },
            IrInst::StorePtr {
                addr: p_dst,
                val: ch,
                ty: store_ty,
                volatile: false,
            },
            IrInst::LoadLocal {
                dst: s1,
                slot: src_slot,
                ty: u16t.clone(),
            },
            IrInst::ConstInt {
                dst: o1,
                ty: u16t.clone(),
                val: 1,
            },
            IrInst::BinOp {
                dst: sn,
                op: IrBinOp::Add,
                mode: ArithMode::Wrapping,
                ty: u16t.clone(),
                lhs: s1,
                rhs: o1,
                site_id: 0,
            },
            IrInst::StoreLocal {
                slot: src_slot,
                val: sn,
                ty: u16t.clone(),
            },
            IrInst::LoadLocal {
                dst: d1,
                slot: dst_slot,
                ty: u16t.clone(),
            },
            IrInst::ConstInt {
                dst: o2,
                ty: u16t.clone(),
                val: 1,
            },
            IrInst::BinOp {
                dst: dn,
                op: IrBinOp::Add,
                mode: ArithMode::Wrapping,
                ty: u16t.clone(),
                lhs: d1,
                rhs: o2,
                site_id: 0,
            },
            IrInst::StoreLocal {
                slot: dst_slot,
                val: dn,
                ty: u16t.clone(),
            },
            IrInst::LoadLocal {
                dst: c1,
                slot: cnt_slot,
                ty: cnt_ty.clone(),
            },
            IrInst::ConstInt {
                dst: o3,
                ty: cnt_ty.clone(),
                val: 1,
            },
            IrInst::BinOp {
                dst: cn,
                op: IrBinOp::Sub,
                mode: ArithMode::Wrapping,
                ty: cnt_ty.clone(),
                lhs: c1,
                rhs: o3,
                site_id: 0,
            },
            IrInst::StoreLocal {
                slot: cnt_slot,
                val: cn,
                ty: cnt_ty.clone(),
            },
            IrInst::ConstInt {
                dst: z0,
                ty: cnt_ty.clone(),
                val: 0,
            },
            IrInst::BinOp {
                dst: cnz,
                op: IrBinOp::Ne,
                mode: ArithMode::Wrapping,
                ty: cnt_ty,
                lhs: cn,
                rhs: z0,
                site_id: 0,
            },
        ];

        let mut hdr_insts = vec![
            IrInst::LoadLocal {
                dst: p_src,
                slot: src_slot,
                ty: u16t.clone(),
            },
            IrInst::LoadPtr {
                dst: ch,
                addr: p_src,
                ty: load_ty.clone(),
                volatile: false,
            },
        ];
        if let Some((sval, exit_on_eq, etgt)) = sentinel_tgt {
            let cs = alloc_val();
            let cmp = alloc_val();
            let cmp_ty = if load_ty == IrType::Bool {
                u8t.clone()
            } else {
                load_ty
            };
            hdr_insts.extend([
                IrInst::ConstInt {
                    dst: cs,
                    ty: cmp_ty.clone(),
                    val: sval as i64,
                },
                IrInst::BinOp {
                    dst: cmp,
                    op: if exit_on_eq { IrBinOp::Eq } else { IrBinOp::Ne },
                    mode: ArithMode::Wrapping,
                    ty: cmp_ty,
                    lhs: ch,
                    rhs: cs,
                    site_id: 0,
                },
            ]);
            new_blocks.push(BasicBlock {
                id: hdr_id,
                insts: hdr_insts,
                term: Terminator::Branch {
                    cond: cmp,
                    then_bb: etgt,
                    else_bb: step_id,
                },
            });
            new_blocks.push(BasicBlock {
                id: step_id,
                insts: step_insts,
                term: Terminator::Branch {
                    cond: cnz,
                    then_bb: hdr_id,
                    else_bb: exit_bound_tgt,
                },
            });
        } else {
            hdr_insts.extend(step_insts);
            new_blocks.push(BasicBlock {
                id: hdr_id,
                insts: hdr_insts,
                term: Terminator::Branch {
                    cond: cnz,
                    then_bb: hdr_id,
                    else_bb: exit_bound_tgt,
                },
            });
        }

        func.blocks.retain(|b| !body.contains(&b.id));
        func.blocks.extend(new_blocks);
        if let Some(bound) = func.loop_bounds.remove(&header) {
            func.loop_bounds.insert(hdr_id, bound);
        }
        func.next_val = next_val;
        any_changed = true;
    }

    if any_changed {
        remove_dead_unreachable_blocks(func);
        remove_dead_pure_insts(func);
    }
    any_changed
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
                        | IrInst::FieldAddr { .. }
                        | IrInst::SlicePtr { .. }
                        | IrInst::SliceLen { .. }
                        | IrInst::LoadPtr { volatile: false, .. }
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
        let mut addr_local_of: HashMap<ValueId, SlotId> = HashMap::new();
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
                    IrInst::AddrLocal { dst, slot } => {
                        addr_local_of.insert(*dst, *slot);
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

        #[derive(Clone, PartialEq, Eq, Hash)]
        enum OsrBaseKey {
            Val(ValueId),
            Global(u32),
            GlobalField(u32, u16),
            Slice(SlotId),
            LocalPtr(SlotId),
            BodyExpr(ValueId),
        }

        struct OsrGroup {
            base_key: OsrBaseKey,
            base_val: ValueId,
            is_runtime_ptr: bool,
            iv_slot: SlotId,
            elem_size: u16,
            init: i64,
            step: i64,
            sites: Vec<(BlockId, usize, ValueId)>,
        }
        let mut groups: Vec<OsrGroup> = Vec::new();
        let mut group_idx_map: HashMap<(OsrBaseKey, SlotId, u16), usize> = HashMap::new();

        let mut body_inst_by_dst: HashMap<ValueId, (BlockId, usize, IrInst)> = HashMap::new();
        let mut local_ptr_stored_in_body: HashSet<SlotId> = HashSet::new();
        for &bid in &body {
            if let Some(&bi) = bb_idx_map.get(&bid) {
                for (ii, inst) in func.blocks[bi].insts.iter().enumerate() {
                    if let Some(d) = inst.dst() {
                        body_inst_by_dst.insert(d, (bid, ii, inst.clone()));
                    }
                    if let IrInst::StorePtr { addr, .. } = inst
                        && let Some(&slot) = addr_local_of.get(addr)
                    {
                        local_ptr_stored_in_body.insert(slot);
                    }
                }
            }
        }

        let can_materialize_osr_base = |root_v: ValueId| -> Option<(OsrBaseKey, bool)> {
            if let Some(&gid) = addr_global_of.get(&root_v) {
                return Some((OsrBaseKey::Global(gid), false));
            }
            if def_block
                .get(&root_v)
                .is_some_and(|db| !body.contains(db) && cfg.dominates(*db, preheader_id))
            {
                return Some((OsrBaseKey::Val(root_v), true));
            }
            // Check pure loop-invariant expressions inside `body` (e.g. SlicePtr(LoadLocal(s)), FieldAddr(AddrGlobal(g), off)).
            fn check_inv(
                v: ValueId,
                body: &BTreeSet<BlockId>,
                preheader_id: BlockId,
                cfg: &Cfg,
                def_block: &HashMap<ValueId, BlockId>,
                body_inst_by_dst: &HashMap<ValueId, (BlockId, usize, IrInst)>,
                escaped: &[bool],
                slot_stores: &HashMap<SlotId, Vec<(BlockId, usize, ValueId)>>,
                addr_local_of: &HashMap<ValueId, SlotId>,
                local_ptr_stored_in_body: &HashSet<SlotId>,
            ) -> bool {
                if def_block
                    .get(&v)
                    .is_some_and(|db| !body.contains(db) && cfg.dominates(*db, preheader_id))
                {
                    return true;
                }
                let Some((_, _, inst)) = body_inst_by_dst.get(&v) else {
                    return false;
                };
                match inst {
                    IrInst::ConstInt { .. }
                    | IrInst::ConstBool { .. }
                    | IrInst::AddrGlobal { .. } => true,
                    IrInst::AddrLocal { .. } => true,
                    IrInst::LoadLocal { slot, .. } => {
                        ((*slot as usize) >= escaped.len() || !escaped[*slot as usize])
                            && !slot_stores.contains_key(slot)
                    }
                    IrInst::LoadPtr {
                        addr,
                        volatile: false,
                        ..
                    } => {
                        if let Some(&slot) = addr_local_of.get(addr) {
                            ((slot as usize) >= escaped.len() || !escaped[slot as usize])
                                && !slot_stores.contains_key(&slot)
                                && !local_ptr_stored_in_body.contains(&slot)
                                && check_inv(
                                    *addr,
                                    body,
                                    preheader_id,
                                    cfg,
                                    def_block,
                                    body_inst_by_dst,
                                    escaped,
                                    slot_stores,
                                    addr_local_of,
                                    local_ptr_stored_in_body,
                                )
                        } else {
                            false
                        }
                    }
                    IrInst::FieldAddr { base, .. } => check_inv(
                        *base,
                        body,
                        preheader_id,
                        cfg,
                        def_block,
                        body_inst_by_dst,
                        escaped,
                        slot_stores,
                        addr_local_of,
                        local_ptr_stored_in_body,
                    ),
                    IrInst::SlicePtr { slice, .. } => check_inv(
                        *slice,
                        body,
                        preheader_id,
                        cfg,
                        def_block,
                        body_inst_by_dst,
                        escaped,
                        slot_stores,
                        addr_local_of,
                        local_ptr_stored_in_body,
                    ),
                    IrInst::Cast {
                        val,
                        from_ty,
                        to_ty,
                        ..
                    } if matches!(
                        from_ty,
                        IrType::Int(IntType::U16 | IntType::I16) | IrType::Ref { .. }
                    ) && matches!(
                        to_ty,
                        IrType::Int(IntType::U16 | IntType::I16) | IrType::Ref { .. }
                    ) =>
                    {
                        check_inv(
                            *val,
                            body,
                            preheader_id,
                            cfg,
                            def_block,
                            body_inst_by_dst,
                            escaped,
                            slot_stores,
                            addr_local_of,
                            local_ptr_stored_in_body,
                        )
                    }
                    _ => false,
                }
            }

            if !check_inv(
                root_v,
                &body,
                preheader_id,
                &cfg,
                &def_block,
                &body_inst_by_dst,
                &escaped,
                &slot_stores,
                &addr_local_of,
                &local_ptr_stored_in_body,
            ) {
                return None;
            }
            match body_inst_by_dst.get(&root_v).map(|(_, _, i)| i) {
                Some(IrInst::SlicePtr { slice, .. }) => {
                    if let Some((_, _, IrInst::LoadLocal { slot, .. })) =
                        body_inst_by_dst.get(slice)
                    {
                        Some((OsrBaseKey::Slice(*slot), true))
                    } else if let Some((_, _, IrInst::LoadPtr { addr, .. })) =
                        body_inst_by_dst.get(slice)
                        && let Some(&slot) = addr_local_of.get(addr)
                    {
                        Some((OsrBaseKey::Slice(slot), true))
                    } else {
                        Some((OsrBaseKey::BodyExpr(root_v), true))
                    }
                }
                Some(IrInst::FieldAddr {
                    base, byte_offset, ..
                }) => {
                    if let Some(&gid) = addr_global_of.get(base) {
                        Some((OsrBaseKey::GlobalField(gid, *byte_offset), false))
                    } else {
                        Some((OsrBaseKey::BodyExpr(root_v), true))
                    }
                }
                Some(IrInst::LoadLocal { slot, .. }) => {
                    Some((OsrBaseKey::LocalPtr(*slot), true))
                }
                _ => Some((OsrBaseKey::BodyExpr(root_v), true)),
            }
        };

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
                    let Some((base_key, is_runtime_ptr)) = can_materialize_osr_base(*base) else {
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
                    if !is_release && *len > 0 {
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
                    let key = (base_key.clone(), iv_slot, *elem_size);
                    if let Some(&gi) = group_idx_map.get(&key) {
                        groups[gi].sites.push((bid, ii, *dst));
                    } else if groups.len() < MAX_OSR_PTRS_PER_LOOP {
                        let gi = groups.len();
                        group_idx_map.insert(key, gi);
                        groups.push(OsrGroup {
                            base_key,
                            base_val: *base,
                            is_runtime_ptr,
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

        groups.retain(|g| g.elem_size >= 2 || g.sites.len() >= 2 || g.is_runtime_ptr);
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

        fn materialize_in_preheader(
            v: ValueId,
            body_inst_by_dst: &HashMap<ValueId, (BlockId, usize, IrInst)>,
            val_map: &mut HashMap<ValueId, ValueId>,
            pre_insts: &mut Vec<IrInst>,
            alloc_val: &mut dyn FnMut() -> ValueId,
        ) -> ValueId {
            if let Some(&mapped) = val_map.get(&v) {
                return mapped;
            }
            let Some((_, _, inst)) = body_inst_by_dst.get(&v) else {
                return v;
            };
            let cloned = inst.clone();
            let mut deps = Vec::new();
            cloned.for_each_use(|u| deps.push(u));
            for u in deps {
                let mu = materialize_in_preheader(u, body_inst_by_dst, val_map, pre_insts, alloc_val);
                val_map.insert(u, mu);
            }
            let out_v = clone_pure_inst_remapped(&cloned, val_map, alloc_val);
            let new_dst = out_v.dst().unwrap();
            pre_insts.push(out_v);
            new_dst
        }

        // Track latch pointer increments keyed by `latch_store_ii` so insertions preserve indices.
        let mut latch_appends: HashMap<SlotId, Vec<IrInst>> = HashMap::new();
        let mut pre_val_map: HashMap<ValueId, ValueId> = HashMap::new();

        for g in &groups {
            let ptr_slot = next_slot;
            next_slot += 1;
            func.locals.push(LocalSlot {
                id: ptr_slot,
                name: format!("__osr_ptr_{ptr_slot}"),
                ty: u16t.clone(),
                mutable: true,
            });

            let base_val = match &g.base_key {
                OsrBaseKey::Global(gid) => {
                    let bv = alloc_val();
                    func.blocks[pre_idx]
                        .insts
                        .push(IrInst::AddrGlobal { dst: bv, global: *gid });
                    bv
                }
                OsrBaseKey::Val(v) => *v,
                OsrBaseKey::GlobalField(..)
                | OsrBaseKey::Slice(..)
                | OsrBaseKey::LocalPtr(..)
                | OsrBaseKey::BodyExpr(..) => {
                    let mut pre_extra = Vec::new();
                    let bv = materialize_in_preheader(
                        g.base_val,
                        &body_inst_by_dst,
                        &mut pre_val_map,
                        &mut pre_extra,
                        &mut alloc_val,
                    );
                    func.blocks[pre_idx].insts.extend(pre_extra);
                    bv
                }
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

/// Hoists loop-invariant `&STATIC_GLOBAL[.field*][idx]` `IndexAddr` computations out of
/// natural loops into the loop preheader in release mode so `detect_ix_static_elem` can
/// promote the hoisted element pointer to `IX` across the entire loop body.
pub fn hoist_invariant_static_index_addrs(func: &mut IrFunction, is_release: bool) -> bool {
    if !is_release || func.blocks.is_empty() {
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

        let Some((header, _latch, body)) = loops.into_iter().next() else {
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
        let mut modified_in_body: HashSet<SlotId> = HashSet::new();
        let mut store_blocks_outside: HashMap<SlotId, Vec<BlockId>> = HashMap::new();
        let mut def_block: HashMap<ValueId, BlockId> = HashMap::new();
        let mut consts: HashSet<ValueId> = HashSet::new();
        let mut static_bases: HashMap<ValueId, (u32, u16)> = HashMap::new();
        let mut local_loads: HashMap<ValueId, (SlotId, IrType)> = HashMap::new();
        let mut widening_casts: HashMap<ValueId, (ValueId, IrType, IrType)> = HashMap::new();

        for &bi in cfg.rpo_indices() {
            let bb = &func.blocks[bi];
            let in_b = body.contains(&bb.id);
            for inst in &bb.insts {
                if let Some(d) = inst.dst() {
                    def_block.insert(d, bb.id);
                }
                match inst {
                    IrInst::ConstInt { dst, .. } => {
                        consts.insert(*dst);
                    }
                    IrInst::StoreLocal { slot, .. } => {
                        if in_b {
                            modified_in_body.insert(*slot);
                        } else {
                            store_blocks_outside.entry(*slot).or_default().push(bb.id);
                        }
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
                    IrInst::LoadLocal { dst, slot, ty } => {
                        local_loads.insert(*dst, (*slot, ty.clone()));
                    }
                    IrInst::Cast {
                        dst,
                        val,
                        from_ty,
                        to_ty,
                    } => {
                        if matches!(from_ty, IrType::Int(IntType::U8))
                            && matches!(to_ty, IrType::Int(IntType::U16))
                        {
                            widening_casts.insert(*dst, (*val, from_ty.clone(), to_ty.clone()));
                        } else if matches!(from_ty, IrType::Ref { .. })
                            && matches!(to_ty, IrType::Ref { .. })
                            && let Some(&b) = static_bases.get(val)
                        {
                            static_bases.insert(*dst, b);
                        }
                    }
                    _ => {}
                }
            }
        }

        #[derive(Clone, PartialEq, Eq, Hash)]
        enum InvIdx {
            Val(ValueId),
            Slot(SlotId, bool), // (slot, widen_u8_to_u16)
        }

        struct HoistGroup {
            gid: u32,
            base_off: u16,
            idx: InvIdx,
            idx_ty: Option<IrType>,
            elem_size: u16,
            len: u16,
            site_id: u16,
            sites: Vec<(usize, usize, ValueId)>,
        }

        let mut groups: Vec<HoistGroup> = Vec::new();
        let mut group_map: HashMap<(u32, u16, InvIdx, u16), usize> = HashMap::new();

        for &bid in &body {
            let Some(&bi) = bb_idx_map.get(&bid) else {
                continue;
            };
            for (ii, inst) in func.blocks[bi].insts.iter().enumerate() {
                let IrInst::IndexAddr {
                    dst,
                    base,
                    index,
                    elem_size,
                    len,
                    site_id,
                } = inst
                else {
                    continue;
                };
                if *elem_size == 0 || consts.contains(index) {
                    continue;
                }
                let Some(&(gid, base_off)) = static_bases.get(base) else {
                    continue;
                };

                let (inv_idx, idx_ty) = if let Some(&db) = def_block.get(index)
                    && !body.contains(&db)
                    && cfg.dominates(db, preheader_id)
                {
                    (InvIdx::Val(*index), None)
                } else {
                    let (inner_v, widen) = if let Some((v, _, _)) = widening_casts.get(index) {
                        (*v, true)
                    } else {
                        (*index, false)
                    };
                    let Some((slot, slot_ty)) = local_loads.get(&inner_v) else {
                        continue;
                    };
                    let s = *slot as usize;
                    if s >= escaped.len()
                        || escaped[s]
                        || modified_in_body.contains(slot)
                        || !(s < func.param_count
                            || store_blocks_outside
                                .get(slot)
                                .is_some_and(|bs| bs.iter().any(|&b| cfg.dominates(b, preheader_id))))
                    {
                        continue;
                    }
                    (InvIdx::Slot(*slot, widen), Some(slot_ty.clone()))
                };

                let key = (gid, base_off, inv_idx.clone(), *elem_size);
                if let Some(&gi) = group_map.get(&key) {
                    groups[gi].sites.push((bi, ii, *dst));
                } else {
                    let gi = groups.len();
                    group_map.insert(key, gi);
                    groups.push(HoistGroup {
                        gid,
                        base_off,
                        idx: inv_idx,
                        idx_ty,
                        elem_size: *elem_size,
                        len: *len,
                        site_id: *site_id,
                        sites: vec![(bi, ii, *dst)],
                    });
                }
            }
        }

        if groups.is_empty() {
            continue;
        }

        let mut next_val = func.next_val;
        let mut alloc_val = || {
            let v = next_val;
            next_val += 1;
            v
        };
        let mut remap: HashMap<ValueId, ValueId> = HashMap::new();
        let mut to_remove: HashSet<(usize, usize)> = HashSet::new();

        for g in groups {
            let gv = alloc_val();
            func.blocks[pre_idx]
                .insts
                .push(IrInst::AddrGlobal { dst: gv, global: g.gid });
            let base_v = if g.base_off == 0 {
                gv
            } else {
                let fv = alloc_val();
                func.blocks[pre_idx].insts.push(IrInst::FieldAddr {
                    dst: fv,
                    base: gv,
                    byte_offset: g.base_off,
                });
                fv
            };
            let idx_v = match g.idx {
                InvIdx::Val(v) => v,
                InvIdx::Slot(slot, widen) => {
                    let lv = alloc_val();
                    let lty = g.idx_ty.unwrap_or(IrType::Int(IntType::U16));
                    func.blocks[pre_idx].insts.push(IrInst::LoadLocal {
                        dst: lv,
                        slot,
                        ty: lty.clone(),
                    });
                    if widen {
                        let cv = alloc_val();
                        func.blocks[pre_idx].insts.push(IrInst::Cast {
                            dst: cv,
                            val: lv,
                            from_ty: lty,
                            to_ty: IrType::Int(IntType::U16),
                        });
                        cv
                    } else {
                        lv
                    }
                }
            };
            let dst_v = alloc_val();
            func.blocks[pre_idx].insts.push(IrInst::IndexAddr {
                dst: dst_v,
                base: base_v,
                index: idx_v,
                elem_size: g.elem_size,
                len: g.len,
                site_id: g.site_id,
            });
            for (bi, ii, old_dst) in g.sites {
                remap.insert(old_dst, dst_v);
                to_remove.insert((bi, ii));
            }
        }

        func.next_val = next_val;
        for (bi, bb) in func.blocks.iter_mut().enumerate() {
            let mut new_insts = Vec::with_capacity(bb.insts.len());
            for (ii, mut inst) in bb.insts.drain(..).enumerate() {
                if to_remove.contains(&(bi, ii)) {
                    continue;
                }
                inst.for_each_use_mut(|v| {
                    if let Some(&nv) = remap.get(v) {
                        *v = nv;
                    }
                });
                new_insts.push(inst);
            }
            bb.insts = new_insts;
            match &mut bb.term {
                Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } => {
                    if let Some(&nv) = remap.get(v) {
                        *v = nv;
                    }
                }
                _ => {}
            }
        }
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
            if let Some(bound) = func.loop_bounds.remove(&header) {
                func.loop_bounds.entry(then_bb).or_insert(bound);
            }
        } else if let Some(&bound) = func.loop_bounds.get(&header) {
            func.loop_bounds.entry(then_bb).or_insert(bound);
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

fn collect_loop_exit_blocks(cfg: &Cfg, body: &BTreeSet<BlockId>) -> Vec<BlockId> {
    let mut exits = Vec::new();
    for &bid in body {
        for &succ in cfg.succs(bid) {
            if !body.contains(&succ) && !exits.contains(&succ) {
                exits.push(succ);
            }
        }
    }
    exits
}

fn slot_live_out_at_any_exit(
    func: &IrFunction,
    cfg: &Cfg,
    body: &BTreeSet<BlockId>,
    slot: SlotId,
) -> bool {
    collect_loop_exit_blocks(cfg, body)
        .into_iter()
        .any(|eb| slot_live_out_at_exit(func, cfg, eb, slot))
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
        // no reads in dominating hot blocks of `body` besides `latch` (+1 & latch test),
        // and no uses of the post-store latch test outside `cond`.
        // Reads inside guarded cold blocks of `body` (`!cfg.dominates(bid, latch)`) are
        // allowed when `excl_bound <= 255` and rewritten later as `(excl_bound as u8) - rem`.
        let mut store_pos: Option<usize> = None;
        let mut abort = false;
        let mut other_loads_in_latch: Vec<ValueId> = Vec::new();
        let mut guarded_loads_in_body: Vec<(usize, usize, ValueId)> = Vec::new();
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
                        if bid == latch {
                            other_loads_in_latch.push(*dst);
                        } else if !cfg.dominates(bid, latch) && !cfg.dominates(latch, bid) {
                            guarded_loads_in_body.push((bi, ii, *dst));
                        } else {
                            abort = true;
                            break;
                        }
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
        let exit_bbs = collect_loop_exit_blocks(&cfg, &body);
        // Any block outside `body` reachable from a post-loop `StoreLocal(iv_slot)` is poisoned
        // (e.g. a subsequent loop updating `iv_slot`), so loads in or after it cannot use `N - rem`.
        let mut reachable_outside: HashSet<BlockId> = HashSet::new();
        let mut rstack = exit_bbs.clone();
        while let Some(bid) = rstack.pop() {
            if body.contains(&bid) || !reachable_outside.insert(bid) {
                continue;
            }
            for &succ in cfg.succs(bid) {
                if !body.contains(&succ) && !reachable_outside.contains(&succ) {
                    rstack.push(succ);
                }
            }
        }
        let mut poisoned_by_post_store: HashSet<BlockId> = HashSet::new();
        let mut pstack: Vec<BlockId> = Vec::new();
        for &bid in &reachable_outside {
            let Some(&bi) = bb_idx_map.get(&bid) else {
                continue;
            };
            if func.blocks[bi]
                .insts
                .iter()
                .any(|inst| matches!(inst, IrInst::StoreLocal { slot, .. } if *slot == iv_slot))
            {
                for &succ in cfg.succs(bid) {
                    if !body.contains(&succ) && poisoned_by_post_store.insert(succ) {
                        pstack.push(succ);
                    }
                }
            }
        }
        while let Some(bid) = pstack.pop() {
            for &succ in cfg.succs(bid) {
                if !body.contains(&succ) && poisoned_by_post_store.insert(succ) {
                    pstack.push(succ);
                }
            }
        }
        let mut visited_exits: HashSet<BlockId> = HashSet::new();
        let mut exit_stack = exit_bbs;
        while let Some(bid) = exit_stack.pop() {
            if body.contains(&bid) || !visited_exits.insert(bid) {
                continue;
            }
            let Some(&bi) = bb_idx_map.get(&bid) else {
                continue;
            };
            let mut loads_here: Vec<(usize, ValueId)> = Vec::new();
            let mut overwritten = false;
            for (ii, inst) in func.blocks[bi].insts.iter().enumerate() {
                match inst {
                    IrInst::LoadLocal { dst, slot, .. } if *slot == iv_slot => {
                        loads_here.push((ii, *dst));
                    }
                    IrInst::StoreLocal { slot, .. } if *slot == iv_slot => {
                        overwritten = true;
                        break;
                    }
                    _ => {}
                }
            }
            if !loads_here.is_empty() {
                if !cfg.dominates(header, bid) || poisoned_by_post_store.contains(&bid) {
                    abort = true;
                    break;
                }
                for (ii, dst) in loads_here {
                    guarded_loads_in_body.push((bi, ii, dst));
                }
            }
            if !overwritten {
                for &succ in cfg.succs(bid) {
                    if !body.contains(&succ) && !visited_exits.contains(&succ) {
                        exit_stack.push(succ);
                    }
                }
            }
        }
        if abort {
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

        // Rewrite guarded reads `k` inside conditional body blocks as `(excl_bound as u8) - rem`.
        guarded_loads_in_body.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
        for (bi, ii, old_dst) in guarded_loads_in_body {
            let v_rem = alloc_val();
            let v_excl = alloc_val();
            func.blocks[bi].insts[ii] = IrInst::LoadLocal {
                dst: v_rem,
                slot: iv_slot,
                ty: u8t.clone(),
            };
            func.blocks[bi].insts.insert(
                ii + 1,
                IrInst::ConstInt {
                    dst: v_excl,
                    ty: u8t.clone(),
                    val: excl_bound,
                },
            );
            func.blocks[bi].insts.insert(
                ii + 2,
                IrInst::BinOp {
                    dst: old_dst,
                    op: IrBinOp::Sub,
                    mode: ArithMode::Wrapping,
                    ty: u8t.clone(),
                    lhs: v_excl,
                    rhs: v_rem,
                    site_id: 0,
                },
            );
        }

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

/// Computes `(live_in, live_out)` slot sets per basic block for all non-escaped locals.
fn compute_non_escaped_slot_liveness(
    func: &IrFunction,
    cfg: &Cfg,
    escaped: &[bool],
) -> (Vec<HashSet<SlotId>>, Vec<HashSet<SlotId>>) {
    let n = func.blocks.len();
    let mut bb_use: Vec<HashSet<SlotId>> = vec![HashSet::new(); n];
    let mut bb_def: Vec<HashSet<SlotId>> = vec![HashSet::new(); n];
    for (bi, bb) in func.blocks.iter().enumerate() {
        for inst in &bb.insts {
            match inst {
                IrInst::LoadLocal { slot, .. } => {
                    let s = *slot as usize;
                    if (s >= escaped.len() || !escaped[s]) && !bb_def[bi].contains(slot) {
                        bb_use[bi].insert(*slot);
                    }
                }
                IrInst::StoreLocal { slot, .. } => {
                    let s = *slot as usize;
                    if s >= escaped.len() || !escaped[s] {
                        bb_def[bi].insert(*slot);
                    }
                }
                _ => {}
            }
        }
    }
    let id_to_idx: HashMap<BlockId, usize> =
        func.blocks.iter().enumerate().map(|(i, b)| (b.id, i)).collect();
    let mut live_in: Vec<HashSet<SlotId>> = vec![HashSet::new(); n];
    let mut live_out: Vec<HashSet<SlotId>> = vec![HashSet::new(); n];
    loop {
        let mut changed = false;
        for bi in (0..n).rev() {
            let bid = func.blocks[bi].id;
            if !cfg.is_reachable(bid) {
                continue;
            }
            let mut new_out: HashSet<SlotId> = HashSet::new();
            for &succ in cfg.succs(bid) {
                if let Some(&si) = id_to_idx.get(&succ) {
                    new_out.extend(live_in[si].iter().copied());
                }
            }
            let mut new_in = bb_use[bi].clone();
            for &s in &new_out {
                if !bb_def[bi].contains(&s) {
                    new_in.insert(s);
                }
            }
            if new_out != live_out[bi] || new_in != live_in[bi] {
                live_out[bi] = new_out;
                live_in[bi] = new_in;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    (live_in, live_out)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum PtrValRoot {
    UnmodSlot(SlotId),
    BlockSlotEpoch(SlotId, u32),
    Ssa(ValueId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PtrOrigin {
    Global(u32, Option<u16>),
    Local(SlotId, Option<u16>),
    Val(PtrValRoot, Option<u16>),
    Unknown,
}

fn type_size_opt(ty: &IrType, structs: &[StructDef], machines: &[MachineDef]) -> Option<u16> {
    match ty {
        IrType::Struct(id) if (*id as usize) >= structs.len() => None,
        IrType::Machine(id) if (*id as usize) >= machines.len() => None,
        IrType::Array { elem, .. } | IrType::Pool { elem, .. } | IrType::Queue { elem, .. } => {
            type_size_opt(elem, structs, machines)?;
            Some(ty.byte_size_with_machines(structs, machines))
        }
        _ => Some(ty.byte_size_with_machines(structs, machines)),
    }
}

fn collect_escaped_slots(func: &IrFunction) -> Vec<bool> {
    let mut escaped = func.escaped_locals();
    for bb in &func.blocks {
        for inst in &bb.insts {
            if let IrInst::AddrLocal { slot, .. } = inst {
                let s = *slot as usize;
                if s < escaped.len() {
                    escaped[s] = true;
                }
            }
        }
    }
    escaped
}

fn compute_ptr_origins(
    func: &IrFunction,
    escaped: &[bool],
    intra_block_idx: Option<usize>,
) -> HashMap<ValueId, PtrOrigin> {
    let mut const_vals: HashMap<ValueId, i64> = HashMap::new();
    let mut written_slots: HashSet<SlotId> = HashSet::new();
    for bb in &func.blocks {
        for inst in &bb.insts {
            match inst {
                IrInst::ConstInt { dst, val, .. } => {
                    const_vals.insert(*dst, *val);
                }
                IrInst::StoreLocal { slot, .. } => {
                    written_slots.insert(*slot);
                }
                _ => {}
            }
        }
    }
    let mut origins: HashMap<ValueId, PtrOrigin> = HashMap::new();
    let cfg = Cfg::build(func);
    let blocks_to_scan: Vec<usize> = if let Some(bi) = intra_block_idx {
        vec![bi]
    } else {
        cfg.rpo_indices().to_vec()
    };

    for bi in blocks_to_scan {
        let mut slot_epochs: HashMap<SlotId, u32> = HashMap::new();
        let mut slot_origins: HashMap<SlotId, PtrOrigin> = HashMap::new();
        for inst in &func.blocks[bi].insts {
            match inst {
                IrInst::AddrGlobal { dst, global } => {
                    origins.insert(*dst, PtrOrigin::Global(*global, Some(0)));
                }
                IrInst::AddrLocal { dst, slot } => {
                    origins.insert(*dst, PtrOrigin::Local(*slot, Some(0)));
                }
                IrInst::LoadLocal { dst, slot, .. } => {
                    let s = *slot as usize;
                    let is_esc = s < escaped.len() && escaped[s];
                    if intra_block_idx.is_some()
                        && !is_esc
                        && let Some(&orig) = slot_origins.get(slot)
                    {
                        origins.insert(*dst, orig);
                    } else {
                        let root = if intra_block_idx.is_some() && !is_esc {
                            let ep = slot_epochs.get(slot).copied().unwrap_or(0);
                            PtrValRoot::BlockSlotEpoch(*slot, ep)
                        } else if !is_esc && !written_slots.contains(slot) {
                            PtrValRoot::UnmodSlot(*slot)
                        } else {
                            PtrValRoot::Ssa(*dst)
                        };
                        origins.insert(*dst, PtrOrigin::Val(root, Some(0)));
                    }
                }
                IrInst::StoreLocal { slot, val, .. } => {
                    if intra_block_idx.is_some() {
                        let ep = slot_epochs.entry(*slot).or_insert(0);
                        *ep += 1;
                        let s = *slot as usize;
                        let is_esc = s < escaped.len() && escaped[s];
                        if !is_esc && let Some(&orig) = origins.get(val) {
                            slot_origins.insert(*slot, orig);
                        } else {
                            slot_origins.remove(slot);
                        }
                    }
                }
                IrInst::FieldAddr {
                    dst,
                    base,
                    byte_offset,
                } => {
                    let parent = origins
                        .get(base)
                        .copied()
                        .unwrap_or(PtrOrigin::Val(PtrValRoot::Ssa(*base), Some(0)));
                    let next = match parent {
                        PtrOrigin::Global(g, o) => {
                            PtrOrigin::Global(g, o.and_then(|v| v.checked_add(*byte_offset)))
                        }
                        PtrOrigin::Local(s, o) => {
                            PtrOrigin::Local(s, o.and_then(|v| v.checked_add(*byte_offset)))
                        }
                        PtrOrigin::Val(r, o) => {
                            PtrOrigin::Val(r, o.and_then(|v| v.checked_add(*byte_offset)))
                        }
                        PtrOrigin::Unknown => PtrOrigin::Unknown,
                    };
                    origins.insert(*dst, next);
                }
                IrInst::IndexAddr {
                    dst,
                    base,
                    index,
                    elem_size,
                    len,
                    ..
                } => {
                    let const_off = const_vals
                        .get(index)
                        .copied()
                        .filter(|&c| *len == 0 || (c >= 0 && (c as u16) < *len))
                        .and_then(|c| u16::try_from(c).ok())
                        .and_then(|i| i.checked_mul(*elem_size));
                    let parent = origins
                        .get(base)
                        .copied()
                        .unwrap_or(PtrOrigin::Val(PtrValRoot::Ssa(*base), Some(0)));
                    let next = match parent {
                        PtrOrigin::Global(g, o) => {
                            PtrOrigin::Global(g, o.and_then(|v| const_off.and_then(|k| v.checked_add(k))))
                        }
                        PtrOrigin::Local(s, o) => {
                            PtrOrigin::Local(s, o.and_then(|v| const_off.and_then(|k| v.checked_add(k))))
                        }
                        PtrOrigin::Val(r, o) => {
                            PtrOrigin::Val(r, o.and_then(|v| const_off.and_then(|k| v.checked_add(k))))
                        }
                        PtrOrigin::Unknown => PtrOrigin::Unknown,
                    };
                    origins.insert(*dst, next);
                }
                IrInst::Cast { dst, val, .. } => {
                    if let Some(&o) = origins.get(val) {
                        origins.insert(*dst, o);
                    }
                }
                _ => {}
            }
        }
    }
    origins
}

fn ranges_overlap_opt(o1: Option<u16>, s1: Option<u16>, o2: Option<u16>, s2: Option<u16>) -> bool {
    let (Some(a), Some(sa), Some(b), Some(sb)) = (o1, s1, o2, s2) else {
        return true;
    };
    let a_end = (a as u32) + (sa.max(1) as u32);
    let b_end = (b as u32) + (sb.max(1) as u32);
    (a as u32) < b_end && (b as u32) < a_end
}

fn ptr_may_alias(
    origins: &HashMap<ValueId, PtrOrigin>,
    a1: ValueId,
    s1: Option<u16>,
    a2: ValueId,
    s2: Option<u16>,
    globals: &[IrGlobal],
) -> bool {
    let o1 = origins.get(&a1).copied().unwrap_or(PtrOrigin::Unknown);
    let o2 = origins.get(&a2).copied().unwrap_or(PtrOrigin::Unknown);
    match (o1, o2) {
        (PtrOrigin::Global(g1, off1), PtrOrigin::Global(g2, off2)) => {
            if g1 == g2 {
                ranges_overlap_opt(off1, s1, off2, s2)
            } else {
                let disjoint = globals.get(g1 as usize).is_some_and(|ga| {
                    ga.fixed_addr.is_none() && !ga.is_extern && ga.owner_state.is_none()
                }) && globals.get(g2 as usize).is_some_and(|gb| {
                    gb.fixed_addr.is_none() && !gb.is_extern && gb.owner_state.is_none()
                });
                !disjoint
            }
        }
        (PtrOrigin::Local(l1, off1), PtrOrigin::Local(l2, off2)) => {
            l1 == l2 && ranges_overlap_opt(off1, s1, off2, s2)
        }
        (PtrOrigin::Global(..), PtrOrigin::Local(..))
        | (PtrOrigin::Local(..), PtrOrigin::Global(..)) => false,
        (PtrOrigin::Val(r1, off1), PtrOrigin::Val(r2, off2)) if r1 == r2 => {
            ranges_overlap_opt(off1, s1, off2, s2)
        }
        _ => true,
    }
}

fn escaped_slot_may_alias(origins: &HashMap<ValueId, PtrOrigin>, addr: ValueId, slot: SlotId) -> bool {
    match origins.get(&addr).copied().unwrap_or(PtrOrigin::Unknown) {
        PtrOrigin::Global(..) => false,
        PtrOrigin::Local(s, _) => s == slot,
        _ => true,
    }
}

fn is_hard_mem_barrier_for_load_sink(inst: &IrInst) -> bool {
    matches!(
        inst,
        IrInst::LoadPtr { volatile: true, .. }
            | IrInst::StorePtr { volatile: true, .. }
            | IrInst::MemSet { .. }
            | IrInst::MemCopy { .. }
            | IrInst::MemFill { .. }
            | IrInst::PoolAlloc { .. }
            | IrInst::PoolFree { .. }
            | IrInst::QueuePush { .. }
            | IrInst::QueuePop { .. }
            | IrInst::Call { .. }
            | IrInst::CallIndirect { .. }
            | IrInst::ExternCall { .. }
            | IrInst::StateDispatch { .. }
            | IrInst::StateTransition { .. }
            | IrInst::BankEnter { .. }
            | IrInst::BankExit { .. }
            | IrInst::CriticalEnter
            | IrInst::CriticalExit
            | IrInst::InlineAsm { .. }
            | IrInst::PortIn { .. }
            | IrInst::PortOut { .. }
            | IrInst::Assert { .. }
            | IrInst::Panic { .. }
    )
}

fn is_sinkable_pure_inst(inst: &IrInst, escaped: &[bool], kept_slots_written: &HashSet<SlotId>) -> bool {
    match inst {
        IrInst::ConstInt { .. }
        | IrInst::ConstBool { .. }
        | IrInst::AddrLocal { .. }
        | IrInst::AddrGlobal { .. }
        | IrInst::FieldAddr { .. }
        | IrInst::Cast { .. }
        | IrInst::FixedInt { .. }
        | IrInst::SlicePtr { .. }
        | IrInst::SliceLen { .. } => true,
        IrInst::UnOp { mode, .. } => *mode != ArithMode::Checked,
        IrInst::BinOp { op, mode, .. } => {
            *mode != ArithMode::Checked && !matches!(op, IrBinOp::Div | IrBinOp::Rem)
        }
        IrInst::LoadLocal { slot, .. } => {
            let s = *slot as usize;
            (s >= escaped.len() || !escaped[s]) && !kept_slots_written.contains(slot)
        }
        _ => false,
    }
}

/// Sinks cold-only local stores, non-volatile `LoadPtr`s, and pure computations executed before
/// a conditional branch down into single-predecessor cold targets, then runs intra-block
/// store-to-load forwarding and dead local-store elimination.
pub fn sink_cold_guard_spills(func: &mut IrFunction) -> bool {
    sink_cold_guard_spills_with_context(func, &[], &[], &[])
}

pub fn sink_cold_guard_spills_with_context(
    func: &mut IrFunction,
    structs: &[StructDef],
    machines: &[MachineDef],
    globals: &[IrGlobal],
) -> bool {
    if func.blocks.len() < 2 {
        return false;
    }
    let mut any_changed = false;

    for _round in 0..16 {
        let cfg = Cfg::build(func);
        // Treat ANY slot referenced by `AddrLocal` (not only `escaped_locals()`, since inlined
        // `LoadPtr`/`StorePtr` on `AddrLocal` also accesses the slot indirectly) as opaque.
        let escaped = collect_escaped_slots(func);
        let is_escaped = |slot: SlotId| -> bool {
            let s = slot as usize;
            s < escaped.len() && escaped[s]
        };
        let (live_in, _live_out) = compute_non_escaped_slot_liveness(func, &cfg, &escaped);
        let id_to_idx: HashMap<BlockId, usize> =
            func.blocks.iter().enumerate().map(|(i, b)| (b.id, i)).collect();

        // Collect all ValueId uses across the function: (use_block_for_dom_check, is_inside_bb).
        // For Phi incoming `(pred_bb, val)`, dominance must cover `pred_bb`.
        let mut ext_uses: HashMap<ValueId, Vec<BlockId>> = HashMap::new();
        let mut fn_consts: HashMap<ValueId, IrInst> = HashMap::new();
        for bb in &func.blocks {
            if !cfg.is_reachable(bb.id) {
                continue;
            }
            for inst in &bb.insts {
                match inst {
                    IrInst::ConstInt { dst, .. }
                    | IrInst::ConstBool { dst, .. }
                    | IrInst::AddrLocal { dst, .. }
                    | IrInst::AddrGlobal { dst, .. } => {
                        fn_consts.insert(*dst, inst.clone());
                    }
                    _ => {}
                }
                if let IrInst::Phi { incoming, .. } = inst {
                    for &(pred_bb, v) in incoming {
                        ext_uses.entry(v).or_default().push(pred_bb);
                    }
                } else {
                    inst.for_each_use(|v| {
                        ext_uses.entry(v).or_default().push(bb.id);
                    });
                }
            }
            match bb.term {
                Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } => {
                    ext_uses.entry(v).or_default().push(bb.id);
                }
                _ => {}
            }
        }

        let mut modified_targets: Vec<usize> = Vec::new();
        let mut round_changed = false;

        for b_idx in 0..func.blocks.len() {
            let b_id = func.blocks[b_idx].id;
            if !cfg.is_reachable(b_id) {
                continue;
            }
            let Terminator::Branch {
                cond,
                then_bb,
                else_bb,
            } = func.blocks[b_idx].term
            else {
                continue;
            };
            if then_bb == else_bb {
                continue;
            }
            let b_origins = compute_ptr_origins(func, &escaped, Some(b_idx));

            let candidates = [(then_bb, else_bb), (else_bb, then_bb)];
            for &(target_id, other_id) in &candidates {
                if target_id == b_id || cfg.preds(target_id) != [b_id] {
                    continue;
                }
                let Some(&target_idx) = id_to_idx.get(&target_id) else {
                    continue;
                };
                let Some(&other_idx) = id_to_idx.get(&other_id) else {
                    continue;
                };

                // Record last store position and local LoadLocal info in `b_idx` so operand
                // cloning checks whether a local read is unmodified up to `b_idx` exit.
                let mut last_store_in_b: HashMap<SlotId, usize> = HashMap::new();
                let mut b_loads: HashMap<ValueId, (usize, SlotId, IrType)> = HashMap::new();
                for (ii, inst) in func.blocks[b_idx].insts.iter().enumerate() {
                    match inst {
                        IrInst::StoreLocal { slot, .. } => {
                            last_store_in_b.insert(*slot, ii);
                        }
                        IrInst::LoadLocal { dst, slot, ty } => {
                            b_loads.insert(*dst, (ii, *slot, ty.clone()));
                        }
                        _ => {}
                    }
                }

                // Values whose non-B uses are NOT all dominated by `target_id` (plus `Phi` in `b_id`).
                let only_used_in_target_dom = |v: ValueId, kept_uses: &HashSet<ValueId>| -> bool {
                    if kept_uses.contains(&v) {
                        return false;
                    }
                    let Some(uses) = ext_uses.get(&v) else {
                        return true;
                    };
                    uses.iter().all(|&u_bb| {
                        // Normal uses inside `b_id` were either checked via `kept_uses` or
                        // belonged to later insts in `b_id` that already moved to `target_id`.
                        // Self-loop `other_id == b_id` is handled conservatively via `!cfg.preds(b_id).contains(&b_id)`.
                        if u_bb == b_id {
                            !cfg.preds(b_id).contains(&b_id)
                        } else {
                            cfg.dominates(target_id, u_bb)
                        }
                    })
                };

                let mut kept_uses: HashSet<ValueId> = HashSet::new();
                kept_uses.insert(cond);
                // Never sink a definition past a Phi in `target_id` that reads it.
                for inst in &func.blocks[target_idx].insts {
                    if let IrInst::Phi { incoming, .. } = inst {
                        for &(_, v) in incoming {
                            kept_uses.insert(v);
                        }
                    } else {
                        break;
                    }
                }
                let mut kept_slots_rw: HashSet<SlotId> = HashSet::new();
                let mut kept_slots_written: HashSet<SlotId> = HashSet::new();
                let mut has_hard_barrier_after = false;
                let mut kept_stores_after: Vec<(ValueId, Option<u16>)> = Vec::new();
                let mut kept_escaped_slot_stores: HashSet<SlotId> = HashSet::new();
                let mut sink_mask = vec![false; func.blocks[b_idx].insts.len()];
                let mut sunk_rev: Vec<IrInst> = Vec::new();
                let mut next_val = func.next_val;
                let mut cloned_val_map: HashMap<ValueId, ValueId> = HashMap::new();

                // Iterate backwards so sinking later stores/loads frees their dependencies before we inspect them.
                for (ii, inst) in func.blocks[b_idx].insts.iter().enumerate().rev() {
                    if matches!(inst, IrInst::Phi { .. }) {
                        continue;
                    }

                    let can_sink = match inst {
                        IrInst::StoreLocal { slot, .. } => {
                            !is_escaped(*slot)
                                && !kept_slots_rw.contains(slot)
                                && !live_in[other_idx].contains(slot)
                        }
                        IrInst::LoadPtr {
                            dst,
                            addr,
                            ty,
                            volatile: false,
                        } => {
                            let load_sz = type_size_opt(ty, structs, machines);
                            only_used_in_target_dom(*dst, &kept_uses)
                                && !has_hard_barrier_after
                                && kept_stores_after.iter().all(|&(sa, s_sz)| {
                                    !ptr_may_alias(&b_origins, *addr, load_sz, sa, s_sz, globals)
                                })
                                && kept_escaped_slot_stores.iter().all(|&es| {
                                    !escaped_slot_may_alias(&b_origins, *addr, es)
                                })
                        }
                        _ if is_sinkable_pure_inst(inst, &escaped, &kept_slots_written) => inst
                            .dst()
                            .is_some_and(|d| only_used_in_target_dom(d, &kept_uses)),
                        _ => false,
                    };

                    if can_sink {
                        sink_mask[ii] = true;
                        let mut moved = inst.clone();
                        // Localize cheap cross-block operands (`ConstInt`/`ConstBool` or unmodified
                        // non-escaped `LoadLocal`) when those operands stay live outside `target_id`.
                        let mut op_clones: Vec<IrInst> = Vec::new();
                        moved.for_each_use_mut(|v| {
                            if let Some(&c_v) = cloned_val_map.get(v) {
                                *v = c_v;
                                return;
                            }
                            if let Some(c_inst) = fn_consts.get(v) {
                                let stays_outside = ext_uses.get(v).is_some_and(|us| {
                                    us.iter().any(|&ub| ub != b_id && !cfg.dominates(target_id, ub))
                                }) || kept_uses.contains(v);
                                if stays_outside {
                                    let nv = next_val;
                                    next_val += 1;
                                    let mut cloned = c_inst.clone();
                                    if let Some(d) = cloned.dst_mut() {
                                        *d = nv;
                                    }
                                    op_clones.push(cloned);
                                    cloned_val_map.insert(*v, nv);
                                    *v = nv;
                                }
                            } else if let Some((def_pos, src_slot, src_ty)) = b_loads.get(v)
                                && !is_escaped(*src_slot)
                                && !kept_slots_written.contains(src_slot)
                                && last_store_in_b
                                    .get(src_slot)
                                    .is_none_or(|&sp| sp < *def_pos)
                            {
                                // Only clone `LoadLocal` if `v` will remain used outside `target_id`
                                // (either already known in `kept_uses`/external, or conservatively when
                                // `b_loads` is live out). Notice `v` might still be sunk later in this
                                // scan if used ONLY in `target_id`; avoid cloning if all external uses
                                // are already inside `target_id` and not in `kept_uses`!
                                let stays_outside = kept_uses.contains(v)
                                    || ext_uses.get(v).is_some_and(|us| {
                                        us.iter().any(|&ub| {
                                            ub != b_id && !cfg.dominates(target_id, ub)
                                        })
                                    });
                                if stays_outside {
                                    let nv = next_val;
                                    next_val += 1;
                                    op_clones.push(IrInst::LoadLocal {
                                        dst: nv,
                                        slot: *src_slot,
                                        ty: src_ty.clone(),
                                    });
                                    cloned_val_map.insert(*v, nv);
                                    *v = nv;
                                }
                            }
                        });
                        // Push `moved` first, then `op_clones` so reversing `sunk_rev` places clones BEFORE `moved`.
                        sunk_rev.push(moved);
                        for cl in op_clones.into_iter().rev() {
                            sunk_rev.push(cl);
                        }
                    } else {
                        inst.for_each_use(|v| {
                            kept_uses.insert(v);
                        });
                        if is_hard_mem_barrier_for_load_sink(inst) {
                            has_hard_barrier_after = true;
                        }
                        match inst {
                            IrInst::LoadLocal { slot, .. } => {
                                kept_slots_rw.insert(*slot);
                            }
                            IrInst::StoreLocal { slot, .. } => {
                                kept_slots_rw.insert(*slot);
                                kept_slots_written.insert(*slot);
                                if is_escaped(*slot) {
                                    kept_escaped_slot_stores.insert(*slot);
                                }
                            }
                            IrInst::StorePtr {
                                addr,
                                ty,
                                volatile: false,
                                ..
                            } => {
                                kept_stores_after
                                    .push((*addr, type_size_opt(ty, structs, machines)));
                            }
                            _ => {}
                        }
                    }
                }

                // Localize any `LoadLocal` in `b_idx` (e.g. from GVN) that stays used in `b_idx`
                // (or `other_id`) AND is also reused inside `target_id`'s dominated region.
                // Re-materializing `LoadLocal` at `phi_end` in `target_id` frees `b_idx`'s SSA
                // value from crossing into `target_id`.
                let mut gvn_loc_clones: Vec<IrInst> = Vec::new();
                let mut gvn_remap: HashMap<ValueId, ValueId> = HashMap::new();
                let mut ordered_loads: Vec<(ValueId, usize, SlotId, IrType)> = b_loads
                    .iter()
                    .map(|(&v, &(pos, sl, ref ty))| (v, pos, sl, ty.clone()))
                    .collect();
                ordered_loads.sort_by_key(|t| t.1);
                for (old_v, def_pos, src_slot, src_ty) in ordered_loads {
                    if sink_mask[def_pos]
                        || is_escaped(src_slot)
                        || last_store_in_b
                            .get(&src_slot)
                            .is_some_and(|&sp| sp >= def_pos)
                    {
                        continue;
                    }
                    let used_in_target = ext_uses.get(&old_v).is_some_and(|us| {
                        us.iter()
                            .any(|&ub| ub != b_id && cfg.dominates(target_id, ub))
                    });
                    if used_in_target {
                        let nv = next_val;
                        next_val += 1;
                        gvn_loc_clones.push(IrInst::LoadLocal {
                            dst: nv,
                            slot: src_slot,
                            ty: src_ty,
                        });
                        gvn_remap.insert(old_v, nv);
                    }
                }

                if !sunk_rev.is_empty() || !gvn_remap.is_empty() {
                    func.next_val = next_val;
                    // Remove sunk instructions from `b_idx`
                    if !sunk_rev.is_empty() {
                        let mut idx = 0;
                        func.blocks[b_idx].insts.retain(|_| {
                            let keep = !sink_mask[idx];
                            idx += 1;
                            keep
                        });
                    }
                    if !gvn_remap.is_empty() {
                        for inst in &mut sunk_rev {
                            inst.for_each_use_mut(|v| {
                                if let Some(&nv) = gvn_remap.get(v) {
                                    *v = nv;
                                }
                            });
                        }
                        for bb in &mut func.blocks {
                            let bid = bb.id;
                            let dom_all = bid != b_id && cfg.dominates(target_id, bid);
                            for inst in &mut bb.insts {
                                if let IrInst::Phi { incoming, .. } = inst {
                                    for (pred_bb, v) in incoming {
                                        if *pred_bb != b_id && cfg.dominates(target_id, *pred_bb)
                                            && let Some(&nv) = gvn_remap.get(v)
                                        {
                                            *v = nv;
                                        }
                                    }
                                } else if dom_all {
                                    inst.for_each_use_mut(|v| {
                                        if let Some(&nv) = gvn_remap.get(v) {
                                            *v = nv;
                                        }
                                    });
                                }
                            }
                            if dom_all {
                                match &mut bb.term {
                                    Terminator::Return(Some(v))
                                    | Terminator::Branch { cond: v, .. } => {
                                        if let Some(&nv) = gvn_remap.get(v) {
                                            *v = nv;
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    // Insert cloned loads + sunk instructions (in original order) after any Phi at start of `target_idx`.
                    let phi_end = func.blocks[target_idx]
                        .insts
                        .iter()
                        .position(|i| !matches!(i, IrInst::Phi { .. }))
                        .unwrap_or(func.blocks[target_idx].insts.len());
                    sunk_rev.reverse();
                    let mut to_insert = gvn_loc_clones;
                    to_insert.extend(sunk_rev);
                    func.blocks[target_idx].insts.splice(phi_end..phi_end, to_insert);

                    modified_targets.push(target_idx);
                    round_changed = true;
                    any_changed = true;
                    break; // Recompute uses/liveness cleanly on next pass or outer round
                }
            }
            if round_changed {
                break;
            }
        }

        // Clean up newly exposed redundant StoreLocal -> LoadLocal pairs inside modified targets.
        for target_idx in modified_targets {
            let fresh_cfg = Cfg::build(func);
            let (_in_l, out_l) = compute_non_escaped_slot_liveness(func, &fresh_cfg, &escaped);
            let target_live_out = out_l[target_idx].clone();

            let mut used_outside_target: HashSet<ValueId> = HashSet::new();
            for (bi, bb) in func.blocks.iter().enumerate() {
                if bi == target_idx {
                    for inst in &bb.insts {
                        if matches!(inst, IrInst::Phi { .. }) {
                            inst.for_each_use(|v| {
                                used_outside_target.insert(v);
                            });
                        }
                    }
                    continue;
                }
                for inst in &bb.insts {
                    inst.for_each_use(|v| {
                        used_outside_target.insert(v);
                    });
                }
                if let Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } = bb.term {
                    used_outside_target.insert(v);
                }
            }

            // 1. Forward StoreLocal(s, val) -> LoadLocal(s) inside `target_idx` for non-escaped slots
            //    when `dst` is strictly local to `target_idx` (so cross-block `AddrLocal`/`ConstInt`
            //    values are never exposed to `ctx.reset_bb()` invalidation across blocks).
            let mut cur_val: HashMap<SlotId, ValueId> = HashMap::new();
            let mut val_subst: HashMap<ValueId, ValueId> = HashMap::new();
            let mut drop_inst = vec![false; func.blocks[target_idx].insts.len()];
            for (ii, inst) in func.blocks[target_idx].insts.iter_mut().enumerate() {
                if !matches!(inst, IrInst::Phi { .. }) {
                    inst.for_each_use_mut(|v| {
                        while let Some(&repl) = val_subst.get(v) {
                            *v = repl;
                        }
                    });
                }
                match inst {
                    IrInst::StoreLocal { slot, val, .. } if !is_escaped(*slot) => {
                        cur_val.insert(*slot, *val);
                    }
                    IrInst::LoadLocal { dst, slot, .. } if !is_escaped(*slot) => {
                        if let Some(&sv) = cur_val.get(slot)
                            && !used_outside_target.contains(dst)
                        {
                            val_subst.insert(*dst, sv);
                            drop_inst[ii] = true;
                        } else {
                            cur_val.insert(*slot, *dst);
                        }
                    }
                    _ => {}
                }
            }
            if !drop_inst.iter().all(|&d| !d) {
                let mut ii = 0;
                func.blocks[target_idx].insts.retain(|_| {
                    let keep = !drop_inst[ii];
                    ii += 1;
                    keep
                });
            }
            if !val_subst.is_empty() {
                for bb in &mut func.blocks {
                    for inst in &mut bb.insts {
                        inst.for_each_use_mut(|v| {
                            while let Some(&repl) = val_subst.get(v) {
                                *v = repl;
                            }
                        });
                    }
                    match &mut bb.term {
                        Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } => {
                            while let Some(&repl) = val_subst.get(v) {
                                *v = repl;
                            }
                        }
                        _ => {}
                    }
                }
            }

            // 2. Eliminate dead non-escaped StoreLocals inside `target_idx`.
            let mut live_after = target_live_out;
            let mut dead_stores = vec![false; func.blocks[target_idx].insts.len()];
            for (ii, inst) in func.blocks[target_idx].insts.iter().enumerate().rev() {
                match inst {
                    IrInst::LoadLocal { slot, .. } if !is_escaped(*slot) => {
                        live_after.insert(*slot);
                    }
                    IrInst::StoreLocal { slot, .. } if !is_escaped(*slot) => {
                        if !live_after.contains(slot) {
                            dead_stores[ii] = true;
                        } else {
                            live_after.remove(slot);
                        }
                    }
                    _ => {}
                }
            }
            if dead_stores.iter().any(|&d| d) {
                let mut ii = 0;
                func.blocks[target_idx].insts.retain(|_| {
                    let keep = !dead_stores[ii];
                    ii += 1;
                    keep
                });
            }
        }

        if !round_changed {
            break;
        }
    }

    if any_changed {
        remove_dead_pure_insts(func);
    }
    any_changed
}

/// Computes the nearest common dominator of a slice of reachable blocks.
fn dom_lca(cfg: &Cfg, blocks: &[BlockId]) -> Option<BlockId> {
    let mut cur = *blocks.first()?;
    if !cfg.is_reachable(cur) {
        return None;
    }
    for &b in blocks.iter().skip(1) {
        if !cfg.is_reachable(b) {
            return None;
        }
        let mut a = cur;
        let mut c = b;
        while a != c {
            let ra = cfg.rpo_pos(a)?;
            let rc = cfg.rpo_pos(c)?;
            if ra > rc {
                a = cfg.idom(a)?;
            } else {
                c = cfg.idom(c)?;
            }
        }
        cur = a;
    }
    Some(cur)
}

/// Verifies that no instruction in `insts` conflicts with sinking a non-volatile `LoadPtr(addr, load_sz)`.
fn insts_invariant_for_load(
    insts: &[IrInst],
    origins: &HashMap<ValueId, PtrOrigin>,
    escaped: &[bool],
    addr: ValueId,
    load_sz: Option<u16>,
    structs: &[StructDef],
    machines: &[MachineDef],
    globals: &[IrGlobal],
) -> bool {
    for inst in insts {
        if is_hard_mem_barrier_for_load_sink(inst) {
            return false;
        }
        match inst {
            IrInst::StorePtr {
                addr: sa,
                ty: sty,
                volatile: false,
                ..
            } => {
                let store_sz = type_size_opt(sty, structs, machines);
                if ptr_may_alias(origins, addr, load_sz, *sa, store_sz, globals) {
                    return false;
                }
            }
            IrInst::StoreLocal { slot, .. } => {
                let s = *slot as usize;
                if s < escaped.len() && escaped[s] && escaped_slot_may_alias(origins, addr, *slot) {
                    return false;
                }
            }
            _ => {}
        }
    }
    true
}

/// Sinks non-volatile `LoadPtr`s (and pure address computations feeding them) across blocks
/// down to the nearest dominating block of their cold-branch uses, provided every CFG path from
/// the original load position to the target block is provably memory-invariant and at no deeper
/// loop nesting depth.
pub fn sink_cold_loads(func: &mut IrFunction) -> bool {
    sink_cold_loads_with_context(func, &[], &[], &[])
}

pub fn sink_cold_loads_with_context(
    func: &mut IrFunction,
    structs: &[StructDef],
    machines: &[MachineDef],
    globals: &[IrGlobal],
) -> bool {
    if func.blocks.len() < 2 {
        return false;
    }
    let mut any_changed = false;

    for _round in 0..8 {
        let cfg = Cfg::build(func);
        let escaped = collect_escaped_slots(func);
        let origins = compute_ptr_origins(func, &escaped, None);
        let id_to_idx: HashMap<BlockId, usize> =
            func.blocks.iter().enumerate().map(|(i, b)| (b.id, i)).collect();

        let mut use_blocks: HashMap<ValueId, Vec<BlockId>> = HashMap::new();
        let mut phi_blocks: HashMap<ValueId, HashSet<BlockId>> = HashMap::new();
        for bb in &func.blocks {
            if !cfg.is_reachable(bb.id) {
                continue;
            }
            for inst in &bb.insts {
                if let IrInst::Phi { incoming, .. } = inst {
                    for &(pred_bb, v) in incoming {
                        use_blocks.entry(v).or_default().push(pred_bb);
                        phi_blocks.entry(v).or_default().insert(bb.id);
                    }
                } else {
                    inst.for_each_use(|v| {
                        use_blocks.entry(v).or_default().push(bb.id);
                    });
                }
            }
            if let Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } = bb.term {
                use_blocks.entry(v).or_default().push(bb.id);
            }
        }

        let mut round_changed = false;
        let rpo = cfg.rpo().to_vec();
        for &def_bb_id in rpo.iter().rev() {
            let Some(&def_bi) = id_to_idx.get(&def_bb_id) else {
                continue;
            };
            let len = func.blocks[def_bi].insts.len();
            for ii in (0..len).rev() {
                let inst = &func.blocks[def_bi].insts[ii];
                let (dst, load_info) = match inst {
                    IrInst::LoadPtr {
                        dst,
                        addr,
                        ty,
                        volatile: false,
                    } => (*dst, Some((*addr, type_size_opt(ty, structs, machines)))),
                    IrInst::FieldAddr { dst, .. }
                    | IrInst::AddrGlobal { dst, .. }
                    | IrInst::AddrLocal { dst, .. } => (*dst, None),
                    _ => continue,
                };
                let Some(uses) = use_blocks.get(&dst) else {
                    continue;
                };
                if uses.is_empty() || uses.iter().any(|&ub| ub == def_bb_id) {
                    continue;
                }
                let Some(lca) = dom_lca(&cfg, uses) else {
                    continue;
                };
                if lca == def_bb_id || !cfg.strictly_dominates(def_bb_id, lca) {
                    continue;
                }

                let mut chosen_target: Option<BlockId> = None;
                let mut cand = lca;
                while cand != def_bb_id {
                    let phi_conflict = phi_blocks.get(&dst).is_some_and(|s| s.contains(&cand));
                    let loop_ok = cfg
                        .natural_loops()
                        .iter()
                        .all(|lp| !lp.body.contains(&cand) || lp.body.contains(&def_bb_id));
                    if !phi_conflict && loop_ok {
                        let mem_ok = if let Some((addr, load_sz)) = load_info {
                            let tail_ok = insts_invariant_for_load(
                                &func.blocks[def_bi].insts[ii + 1..],
                                &origins,
                                &escaped,
                                addr,
                                load_sz,
                                structs,
                                machines,
                                globals,
                            );
                            if !tail_ok {
                                false
                            } else {
                                let mut worklist: Vec<BlockId> = cfg.reachable_preds(cand);
                                let mut visited: HashSet<BlockId> = HashSet::new();
                                let mut all_paths_ok = !worklist.is_empty();
                                while let Some(u) = worklist.pop() {
                                    if u == def_bb_id {
                                        continue;
                                    }
                                    if !visited.insert(u) {
                                        continue;
                                    }
                                    if !cfg.strictly_dominates(def_bb_id, u) {
                                        all_paths_ok = false;
                                        break;
                                    }
                                    let Some(&ui) = id_to_idx.get(&u) else {
                                        all_paths_ok = false;
                                        break;
                                    };
                                    if !insts_invariant_for_load(
                                        &func.blocks[ui].insts,
                                        &origins,
                                        &escaped,
                                        addr,
                                        load_sz,
                                        structs,
                                        machines,
                                        globals,
                                    ) {
                                        all_paths_ok = false;
                                        break;
                                    }
                                    worklist.extend(cfg.reachable_preds(u));
                                }
                                all_paths_ok
                            }
                        } else {
                            true
                        };
                        if mem_ok {
                            chosen_target = Some(cand);
                            break;
                        }
                    }
                    let Some(parent) = cfg.idom(cand) else {
                        break;
                    };
                    cand = parent;
                }

                let Some(target_id) = chosen_target else {
                    continue;
                };
                let Some(&target_bi) = id_to_idx.get(&target_id) else {
                    continue;
                };
                let moved = func.blocks[def_bi].insts.remove(ii);
                let phi_end = func.blocks[target_bi]
                    .insts
                    .iter()
                    .position(|i| !matches!(i, IrInst::Phi { .. }))
                    .unwrap_or(func.blocks[target_bi].insts.len());
                moved.for_each_use(|op_v| {
                    if let Some(vec_u) = use_blocks.get_mut(&op_v) {
                        let mut replaced = false;
                        for ub in vec_u.iter_mut() {
                            if *ub == def_bb_id && !replaced {
                                *ub = target_id;
                                replaced = true;
                            }
                        }
                    }
                });
                func.blocks[target_bi].insts.insert(phi_end, moved);
                round_changed = true;
                any_changed = true;
            }
        }

        if !round_changed {
            break;
        }
    }
    any_changed
}

/// Schedules independent non-volatile `LoadPtr`s and pure computations closer to their first
/// use within each basic block, avoiding simultaneous live ranges and immediate `__aun_ram`
/// spill-then-reload pairs when loading multiple struct fields back-to-back.
pub fn schedule_loads_near_use(func: &mut IrFunction) -> bool {
    schedule_loads_near_use_with_context(func, &[], &[], &[])
}

fn is_movable_pure_for_schedule(inst: &IrInst) -> bool {
    match inst {
        IrInst::AddrLocal { .. }
        | IrInst::AddrGlobal { .. }
        | IrInst::AddrFn { .. }
        | IrInst::FieldAddr { .. }
        | IrInst::Cast { .. }
        | IrInst::FixedInt { .. }
        | IrInst::SlicePtr { .. }
        | IrInst::SliceLen { .. } => true,
        IrInst::UnOp { mode, .. } => *mode != ArithMode::Checked,
        IrInst::BinOp { op, mode, .. } => {
            *mode != ArithMode::Checked && !matches!(op, IrBinOp::Div | IrBinOp::Rem)
        }
        _ => false,
    }
}

pub fn schedule_loads_near_use_with_context(
    func: &mut IrFunction,
    structs: &[StructDef],
    machines: &[MachineDef],
    globals: &[IrGlobal],
) -> bool {
    let escaped = collect_escaped_slots(func);
    let is_escaped = |slot: SlotId| -> bool {
        let s = slot as usize;
        s < escaped.len() && escaped[s]
    };
    let mut any_changed = false;

    for bi in 0..func.blocks.len() {
        let len = func.blocks[bi].insts.len();
        if len < 2 {
            continue;
        }
        enum MoveKind {
            Pure,
            Local(SlotId),
            LoadPtr(ValueId, Option<u16>),
        }
        let origins = compute_ptr_origins(func, &escaped, Some(bi));

        for ii in (0..len).rev() {
            let (dst, kind) = {
                let inst = &func.blocks[bi].insts[ii];
                let mk = if is_movable_pure_for_schedule(inst) {
                    Some(MoveKind::Pure)
                } else {
                    match inst {
                        IrInst::LoadLocal { slot, .. } if !is_escaped(*slot) => {
                            Some(MoveKind::Local(*slot))
                        }
                        IrInst::LoadPtr {
                            addr,
                            ty,
                            volatile: false,
                            ..
                        } => Some(MoveKind::LoadPtr(
                            *addr,
                            type_size_opt(ty, structs, machines),
                        )),
                        _ => None,
                    }
                };
                let (Some(d), Some(k)) = (inst.dst(), mk) else {
                    continue;
                };
                (d, k)
            };

            let insts = &func.blocks[bi].insts;
            let first_use_inst = (ii + 1..insts.len()).find(|&j| insts[j].uses_val(dst));
            let limit = if let Some(fu) = first_use_inst {
                fu
            } else if matches!(
                func.blocks[bi].term,
                Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } if v == dst
            ) {
                insts.len()
            } else {
                continue;
            };

            let mut target_pos = limit;
            match kind {
                MoveKind::Pure => {}
                MoveKind::Local(slot) => {
                    for (j, mid) in insts.iter().enumerate().take(limit).skip(ii + 1) {
                        if matches!(mid, IrInst::StoreLocal { slot: s, .. } if *s == slot) {
                            target_pos = j;
                            break;
                        }
                    }
                }
                MoveKind::LoadPtr(addr, load_sz) => {
                    for (j, mid) in insts.iter().enumerate().take(limit).skip(ii + 1) {
                        let blocked = if is_hard_mem_barrier_for_load_sink(mid) {
                            true
                        } else {
                            match mid {
                                IrInst::StorePtr {
                                    addr: sa,
                                    ty: sty,
                                    volatile: false,
                                    ..
                                } => {
                                    let store_sz = type_size_opt(sty, structs, machines);
                                    ptr_may_alias(&origins, addr, load_sz, *sa, store_sz, globals)
                                }
                                IrInst::StoreLocal { slot: s, .. } => {
                                    is_escaped(*s) && escaped_slot_may_alias(&origins, addr, *s)
                                }
                                _ => false,
                            }
                        };
                        if blocked {
                            target_pos = j;
                            break;
                        }
                    }
                }
            }

            let only_consts_between = target_pos > ii + 1
                && func.blocks[bi].insts[ii + 1..target_pos]
                    .iter()
                    .all(|m| matches!(m, IrInst::ConstInt { .. } | IrInst::ConstBool { .. }));
            // Also avoid swapping an instruction past a sibling operand chain that solely builds
            // another operand of the same `limit` instruction (e.g., `w1 = *p11; w2 = *p13; w1 + w2`):
            // reversing `(w1, w2)` -> `(w2, w1)` saves no spills and breaks sequential HL walking.
            let only_sibling_operands_for_limit = target_pos == limit
                && target_pos > ii + 1
                && {
                    let mid = &func.blocks[bi].insts[ii + 1..target_pos];
                    let side_effect_free = mid.iter().all(|m| {
                        matches!(
                            m,
                            IrInst::ConstInt { .. }
                                | IrInst::ConstBool { .. }
                                | IrInst::LoadPtr {
                                    volatile: false,
                                    ..
                                }
                                | IrInst::LoadLocal { .. }
                        ) || is_movable_pure_for_schedule(m)
                    });
                    if !side_effect_free {
                        false
                    } else {
                        let tail_start = (limit + 1).min(func.blocks[bi].insts.len());
                        let tail = &func.blocks[bi].insts[tail_start..];
                        // Walk backwards from `limit`: every instruction in `mid` must feed
                        // `limit` (either directly or through later instructions in `mid`) and
                        // must not merely compute an unrelated value consumed strictly after `limit`.
                        let mut feeds_limit: Vec<ValueId> = Vec::new();
                        if let Some(lim) = func.blocks[bi].insts.get(limit) {
                            lim.for_each_use(|u| feeds_limit.push(u));
                        } else if let Terminator::Return(Some(v))
                        | Terminator::Branch { cond: v, .. } = &func.blocks[bi].term
                        {
                            feeds_limit.push(*v);
                        }
                        let mut ok = true;
                        for m in mid.iter().rev() {
                            let Some(md) = m.dst() else {
                                ok = false;
                                break;
                            };
                            if !feeds_limit.contains(&md) {
                                let term_uses = matches!(
                                    &func.blocks[bi].term,
                                    Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } if *v == md
                                );
                                let used_later =
                                    term_uses || tail.iter().any(|t| t.uses_val(md));
                                if used_later {
                                    ok = false;
                                    break;
                                }
                            }
                            m.for_each_use(|u| {
                                if !feeds_limit.contains(&u) {
                                    feeds_limit.push(u);
                                }
                            });
                        }
                        ok
                    }
                };
            if target_pos > ii + 1 && !only_consts_between && !only_sibling_operands_for_limit {
                let moved = func.blocks[bi].insts.remove(ii);
                func.blocks[bi].insts.insert(target_pos - 1, moved);
                any_changed = true;
            }
        }
    }
    any_changed
}

/// Converts dead count-up `u16` induction variables (`i = 0; i < limit; i += 1`) into
/// `u16` countdowns (`rem = limit; rem != 0; rem -= 1`) so Z80 codegen keeps `rem` in `BC`
/// (`dec bc; ld a, b; or c; jp nz`) without reloading `limit` (e.g. `s.len()`) per iteration.
pub fn convert_u16_countup_to_countdown(func: &mut IrFunction) -> bool {
    if func.blocks.is_empty() {
        return false;
    }
    let mut any_changed = false;
    let mut processed_headers: HashSet<BlockId> = HashSet::new();
    let u16t = IrType::Int(IntType::U16);

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
        let (Some(&hdr_idx), Some(&latch_idx), Some(&pg_idx)) = (
            bb_idx_map.get(&header),
            bb_idx_map.get(&latch),
            bb_idx_map.get(&pre_or_guard_id),
        ) else {
            continue;
        };

        let escaped = func.escaped_locals();
        let mut consts: HashMap<ValueId, i64> = HashMap::new();
        let mut def_bb: HashMap<ValueId, BlockId> = HashMap::new();
        let mut val_uses: HashMap<ValueId, usize> = HashMap::new();
        for bb in &func.blocks {
            for inst in &bb.insts {
                if let Some(d) = inst.dst() {
                    def_bb.insert(d, bb.id);
                }
                if let IrInst::ConstInt { dst, val, .. } = inst {
                    consts.insert(*dst, *val);
                }
                inst.for_each_use(|v| *val_uses.entry(v).or_insert(0) += 1);
            }
            if let Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } = bb.term {
                *val_uses.entry(v).or_insert(0) += 1;
            }
        }

        let mut body_slot_stores: HashSet<SlotId> = HashSet::new();
        for &bid in &body {
            if let Some(&bi) = bb_idx_map.get(&bid) {
                for inst in &func.blocks[bi].insts {
                    if let IrInst::StoreLocal { slot, .. } = inst {
                        body_slot_stores.insert(*slot);
                    }
                }
            }
        }

        let is_u16_slot = |slot: SlotId| -> bool {
            func.locals
                .iter()
                .any(|l| l.id == slot && l.ty == u16t)
        };

        // Case A: Unrotated `header -> Branch(iv < limit, then_bb, exit_bb)` with `latch -> Goto(header)`.
        if header != latch
            && matches!(func.blocks[pg_idx].term, Terminator::Goto(t) if t == header)
            && let Terminator::Branch {
                cond,
                then_bb,
                else_bb,
            } = func.blocks[hdr_idx].term
            && body.contains(&then_bb)
            && !body.contains(&else_bb)
            && matches!(func.blocks[latch_idx].term, Terminator::Goto(t) if t == header)
        {
            let hdr_insts = &func.blocks[hdr_idx].insts;
            let hdr_pure = hdr_insts.iter().all(|inst| {
                matches!(
                    inst,
                    IrInst::ConstInt { .. }
                        | IrInst::ConstBool { .. }
                        | IrInst::LoadLocal { .. }
                        | IrInst::SliceLen { .. }
                        | IrInst::Cast { .. }
                        | IrInst::BinOp {
                            op: IrBinOp::Lt,
                            ..
                        }
                )
            });
            if !hdr_pure {
                continue;
            }
            let hdr_defs: HashSet<ValueId> =
                hdr_insts.iter().filter_map(|i| i.dst()).collect();
            let hdr_used_outside = func.blocks.iter().any(|bb| {
                bb.id != header
                    && (bb.insts.iter().any(|inst| {
                        let mut hit = false;
                        inst.for_each_use(|v| {
                            if hdr_defs.contains(&v) {
                                hit = true;
                            }
                        });
                        hit
                    }) || matches!(
                        bb.term,
                        Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. }
                        if hdr_defs.contains(&v)
                    ))
            });
            if hdr_used_outside {
                continue;
            }
            let Some(IrInst::BinOp {
                dst: cmp_dst,
                op: IrBinOp::Lt,
                ty,
                lhs,
                rhs,
                ..
            }) = hdr_insts.last()
            else {
                continue;
            };
            if *cmp_dst != cond || *ty != u16t {
                continue;
            }
            let mut hdr_by_dst: HashMap<ValueId, IrInst> = HashMap::new();
            for inst in hdr_insts {
                if let Some(d) = inst.dst() {
                    hdr_by_dst.insert(d, inst.clone());
                }
            }
            let Some(IrInst::LoadLocal { slot: iv_slot, .. }) = hdr_by_dst.get(lhs) else {
                continue;
            };
            let iv_slot = *iv_slot;
            if ((iv_slot as usize) < escaped.len() && escaped[iv_slot as usize])
                || !is_u16_slot(iv_slot)
                || find_loop_iv_init(func, &cfg, pre_or_guard_id, iv_slot, &consts) != Some(0)
                || slot_live_out_at_any_exit(func, &cfg, &body, iv_slot)
            {
                continue;
            }

            // Check `limit` (`rhs`) can be materialized in `pre_or_guard_id`.
            fn can_inv_hdr(
                v: ValueId,
                header: BlockId,
                pre_id: BlockId,
                iv_slot: SlotId,
                cfg: &Cfg,
                def_bb: &HashMap<ValueId, BlockId>,
                hdr_by_dst: &HashMap<ValueId, IrInst>,
                escaped: &[bool],
                body_slot_stores: &HashSet<SlotId>,
            ) -> bool {
                if def_bb.get(&v).copied() != Some(header) {
                    return def_bb
                        .get(&v)
                        .is_some_and(|db| cfg.dominates(*db, pre_id));
                }
                match hdr_by_dst.get(&v) {
                    Some(IrInst::ConstInt { .. } | IrInst::ConstBool { .. }) => true,
                    Some(IrInst::LoadLocal { slot, .. }) => {
                        *slot != iv_slot
                            && ((*slot as usize) >= escaped.len() || !escaped[*slot as usize])
                            && !body_slot_stores.contains(slot)
                    }
                    Some(IrInst::SliceLen { slice, .. }) => can_inv_hdr(
                        *slice,
                        header,
                        pre_id,
                        iv_slot,
                        cfg,
                        def_bb,
                        hdr_by_dst,
                        escaped,
                        body_slot_stores,
                    ),
                    Some(IrInst::Cast { val, .. }) => can_inv_hdr(
                        *val,
                        header,
                        pre_id,
                        iv_slot,
                        cfg,
                        def_bb,
                        hdr_by_dst,
                        escaped,
                        body_slot_stores,
                    ),
                    _ => false,
                }
            }
            if !can_inv_hdr(
                *rhs,
                header,
                pre_or_guard_id,
                iv_slot,
                &cfg,
                &def_bb,
                &hdr_by_dst,
                &escaped,
                &body_slot_stores,
            ) {
                continue;
            }

            // Verify `iv_slot` has only 1 store in `body` (at `latch`: `iv = iv + 1`) and 2 loads total.
            let mut iv_loads_body = 0usize;
            let mut latch_add_info: Option<(usize, usize, ValueId, ValueId, ValueId)> = None;
            let mut bad_iv = false;
            for &bid in &body {
                let bi = bb_idx_map[&bid];
                let mut local_loads: HashMap<ValueId, usize> = HashMap::new();
                let mut add1s: HashMap<ValueId, (usize, ValueId, ValueId)> = HashMap::new();
                for (ii, inst) in func.blocks[bi].insts.iter().enumerate() {
                    match inst {
                        IrInst::LoadLocal { dst, slot, .. } if *slot == iv_slot => {
                            iv_loads_body += 1;
                            local_loads.insert(*dst, ii);
                        }
                        IrInst::BinOp {
                            dst,
                            op: IrBinOp::Add,
                            ty,
                            lhs: al,
                            rhs: ar,
                            ..
                        } if *ty == u16t => {
                            if local_loads.contains_key(al) && consts.get(ar) == Some(&1) {
                                add1s.insert(*dst, (ii, *al, *ar));
                            } else if local_loads.contains_key(ar) && consts.get(al) == Some(&1) {
                                add1s.insert(*dst, (ii, *ar, *al));
                            }
                        }
                        IrInst::StoreLocal { slot, val, .. } if *slot == iv_slot => {
                            if bid != latch || latch_add_info.is_some() {
                                bad_iv = true;
                                break;
                            }
                            if let Some(&(add_ii, l_v, one_v)) = add1s.get(val)
                                && val_uses.get(val).copied() == Some(1)
                                && val_uses.get(&l_v).copied() == Some(1)
                            {
                                latch_add_info = Some((ii, add_ii, *val, l_v, one_v));
                            } else {
                                bad_iv = true;
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                if bad_iv {
                    break;
                }
            }
            let Some((_, latch_add_ii, _, l_v, one_v)) = latch_add_info else {
                continue;
            };
            if bad_iv || iv_loads_body != 2 || val_uses.get(lhs).copied() != Some(1) {
                continue;
            }

            let mut next_val = func.next_val;
            let mut alloc_val = || {
                let v = next_val;
                next_val += 1;
                v
            };
            let mut val_map: HashMap<ValueId, ValueId> = HashMap::new();
            let mut pre_extra: Vec<IrInst> = Vec::new();
            fn emit_hdr_val(
                v: ValueId,
                header: BlockId,
                def_bb: &HashMap<ValueId, BlockId>,
                hdr_by_dst: &HashMap<ValueId, IrInst>,
                val_map: &mut HashMap<ValueId, ValueId>,
                out: &mut Vec<IrInst>,
                alloc_val: &mut dyn FnMut() -> ValueId,
            ) -> ValueId {
                if let Some(&m) = val_map.get(&v) {
                    return m;
                }
                if def_bb.get(&v).copied() != Some(header) {
                    return v;
                }
                let inst = hdr_by_dst[&v].clone();
                let mut deps = Vec::new();
                inst.for_each_use(|u| deps.push(u));
                for u in deps {
                    let mu = emit_hdr_val(u, header, def_bb, hdr_by_dst, val_map, out, alloc_val);
                    val_map.insert(u, mu);
                }
                let cloned = clone_pure_inst_remapped(&inst, val_map, alloc_val);
                let d = cloned.dst().unwrap();
                out.push(cloned);
                d
            }
            let pre_limit_v = emit_hdr_val(
                *rhs,
                header,
                &def_bb,
                &hdr_by_dst,
                &mut val_map,
                &mut pre_extra,
                &mut alloc_val,
            );
            // Replace trailing `iv_slot = 0` store in preheader if nothing after it reads `iv_slot`.
            let pre_insts = &mut func.blocks[pg_idx].insts;
            if let Some(pos) = pre_insts.iter().rposition(
                |i| matches!(i, IrInst::StoreLocal { slot, .. } if *slot == iv_slot),
            ) && !pre_insts[pos + 1..]
                .iter()
                .any(|i| matches!(i, IrInst::LoadLocal { slot, .. } if *slot == iv_slot))
            {
                pre_insts.remove(pos);
            }
            pre_insts.extend(pre_extra);
            pre_insts.push(IrInst::StoreLocal {
                slot: iv_slot,
                val: pre_limit_v,
                ty: u16t.clone(),
            });

            let h_cur = alloc_val();
            let h_zero = alloc_val();
            let h_nz = alloc_val();
            func.blocks[hdr_idx].insts = vec![
                IrInst::LoadLocal {
                    dst: h_cur,
                    slot: iv_slot,
                    ty: u16t.clone(),
                },
                IrInst::ConstInt {
                    dst: h_zero,
                    ty: u16t.clone(),
                    val: 0,
                },
                IrInst::BinOp {
                    dst: h_nz,
                    op: IrBinOp::Ne,
                    mode: ArithMode::Wrapping,
                    ty: u16t.clone(),
                    lhs: h_cur,
                    rhs: h_zero,
                    site_id: 0,
                },
            ];
            func.blocks[hdr_idx].term = Terminator::Branch {
                cond: h_nz,
                then_bb,
                else_bb,
            };
            if let IrInst::BinOp { op, lhs: al, rhs: ar, .. } = &mut func.blocks[latch_idx].insts[latch_add_ii] {
                *op = IrBinOp::Sub;
                *al = l_v;
                *ar = one_v;
            }
            func.next_val = next_val;
            any_changed = true;
            continue;
        }

        // Case B: Rotated `latch -> Branch(iv < limit, header, else_bb)` with guard/preheader `pg_idx`.
        if let Terminator::Branch {
            cond,
            then_bb,
            else_bb,
        } = func.blocks[latch_idx].term
            && then_bb == header
            && !body.contains(&else_bb)
        {
            let latch_insts = &func.blocks[latch_idx].insts;
            let mut local_loads: HashMap<ValueId, SlotId> = HashMap::new();
            let mut add1_defs: HashMap<ValueId, SlotId> = HashMap::new();
            let mut cmp_info: Option<(SlotId, ValueId)> = None;
            for inst in latch_insts {
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
                    } if *ty == u16t => {
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
                        op: IrBinOp::Lt,
                        ty,
                        lhs,
                        rhs,
                        ..
                    } if *dst == cond && *ty == u16t => {
                        if let Some(&s) = local_loads.get(lhs) {
                            cmp_info = Some((s, *rhs));
                        }
                    }
                    _ => {}
                }
            }
            let Some((iv_slot, bound_val)) = cmp_info else {
                continue;
            };
            if ((iv_slot as usize) < escaped.len() && escaped[iv_slot as usize])
                || !is_u16_slot(iv_slot)
                || slot_live_out_at_any_exit(func, &cfg, &body, iv_slot)
            {
                continue;
            }
            // Verify single store and 2 loads of `iv_slot` in `body`, all in `latch`.
            let mut store_pos: Option<usize> = None;
            let mut other_loads: Vec<ValueId> = Vec::new();
            let mut bad_iv = false;
            for &bid in &body {
                let bi = bb_idx_map[&bid];
                for (ii, inst) in func.blocks[bi].insts.iter().enumerate() {
                    match inst {
                        IrInst::StoreLocal { slot, val, .. } if *slot == iv_slot => {
                            if bid != latch || store_pos.is_some() || add1_defs.get(val) != Some(&iv_slot) {
                                bad_iv = true;
                                break;
                            }
                            store_pos = Some(ii);
                        }
                        IrInst::LoadLocal { dst, slot, .. } if *slot == iv_slot => {
                            if bid != latch {
                                bad_iv = true;
                                break;
                            }
                            other_loads.push(*dst);
                        }
                        _ => {}
                    }
                }
                if bad_iv {
                    break;
                }
            }
            let Some(latch_store_idx) = store_pos else {
                continue;
            };
            if bad_iv || other_loads.len() != 2 {
                continue;
            }
            let old_add_val = match &latch_insts[latch_store_idx] {
                IrInst::StoreLocal { val, .. } => *val,
                _ => unreachable!(),
            };
            let drop_vals: HashSet<ValueId> =
                [other_loads[0], old_add_val, other_loads[1], cond].into_iter().collect();
            let extra_use = func.blocks.iter().enumerate().any(|(bi, bb)| {
                bb.insts.iter().enumerate().any(|(ii, inst)| {
                    let is_self = bi == latch_idx
                        && (ii == latch_store_idx
                            || inst.dst().is_some_and(|d| drop_vals.contains(&d)));
                    if is_self {
                        return false;
                    }
                    let mut hit = false;
                    inst.for_each_use(|v| {
                        if drop_vals.contains(&v) {
                            hit = true;
                        }
                    });
                    hit
                }) || (bi != latch_idx
                    && matches!(
                        bb.term,
                        Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. }
                        if drop_vals.contains(&v)
                    ))
            });
            if extra_use {
                continue;
            }

            // Identify `limit` kind: either constant `bound_c`, or unmodified non-escaped `end_slot`.
            let bound_const = consts.get(&bound_val).copied();
            let bound_slot = latch_insts.iter().rev().find_map(|inst| match inst {
                IrInst::LoadLocal { dst, slot, .. }
                    if *dst == bound_val
                        && ((*slot as usize) >= escaped.len() || !escaped[*slot as usize])
                        && !body_slot_stores.contains(slot) =>
                {
                    Some(*slot)
                }
                _ => None,
            });
            if bound_const.is_none() && bound_slot.is_none() {
                continue;
            }

            // Validate `pg_idx` is either `Goto(header)` with const `bound_c > 0`,
            // or a guard `Branch(iv < limit, header, else_bb)` whose limit matches `bound_const` / `bound_slot`.
            enum PgKind {
                UncondConst(i64),
                Guard {
                    init_bb_idx: usize,
                    cmp_inst_idx: usize,
                    cmp_lhs_load: ValueId,
                },
            }
            let pg_kind = if matches!(func.blocks[pg_idx].term, Terminator::Goto(t) if t == header) {
                let Some(bc) = bound_const else {
                    continue;
                };
                if bc <= 0
                    || find_loop_iv_init(func, &cfg, pre_or_guard_id, iv_slot, &consts) != Some(0)
                {
                    continue;
                }
                PgKind::UncondConst(bc)
            } else if let Terminator::Branch {
                cond: g_cond,
                then_bb: g_then,
                else_bb: g_else,
            } = func.blocks[pg_idx].term
                && g_then == header
                && g_else == else_bb
                && val_uses.get(&g_cond).copied() == Some(1)
            {
                let mut init_bb_opt: Option<usize> = None;
                let mut g_loads: HashMap<ValueId, SlotId> = HashMap::new();
                for inst in &func.blocks[pg_idx].insts {
                    if let IrInst::LoadLocal { dst, slot, .. } = inst {
                        g_loads.insert(*dst, *slot);
                    }
                    if let IrInst::StoreLocal { slot, val, .. } = inst
                        && *slot == iv_slot
                        && consts.get(val) == Some(&0)
                    {
                        init_bb_opt = Some(pg_idx);
                    }
                }
                if init_bb_opt.is_none() && cfg.preds(pre_or_guard_id).len() == 1 {
                    let pid = cfg.preds(pre_or_guard_id)[0];
                    if let Some(&pidx) = bb_idx_map.get(&pid)
                        && matches!(func.blocks[pidx].term, Terminator::Goto(t) if t == pre_or_guard_id)
                        && find_loop_iv_init(func, &cfg, pid, iv_slot, &consts) == Some(0)
                    {
                        init_bb_opt = Some(pidx);
                    }
                }
                let Some(init_bb_idx) = init_bb_opt else {
                    continue;
                };
                let pg_insts = &func.blocks[pg_idx].insts;
                let Some(IrInst::BinOp {
                    dst: g_dst,
                    op: IrBinOp::Lt,
                    ty: g_ty,
                    lhs: g_lhs,
                    rhs: g_rhs,
                    ..
                }) = pg_insts.last()
                else {
                    continue;
                };
                if *g_dst != g_cond || *g_ty != u16t || g_loads.get(g_lhs) != Some(&iv_slot) {
                    continue;
                }
                let limit_matches = match (bound_const, bound_slot) {
                    (Some(bc), _) => consts.get(g_rhs) == Some(&bc),
                    (None, Some(es)) => g_loads.get(g_rhs) == Some(&es),
                    _ => false,
                };
                if !limit_matches || val_uses.get(g_lhs).copied() != Some(1) {
                    continue;
                }
                PgKind::Guard {
                    init_bb_idx,
                    cmp_inst_idx: pg_insts.len() - 1,
                    cmp_lhs_load: *g_lhs,
                }
            } else {
                continue;
            };

            let mut next_val = func.next_val;
            let mut alloc_val = || {
                let v = next_val;
                next_val += 1;
                v
            };

            match pg_kind {
                PgKind::UncondConst(bc) => {
                    let pre_insts = &mut func.blocks[pg_idx].insts;
                    if let Some(pos) = pre_insts.iter().rposition(
                        |i| matches!(i, IrInst::StoreLocal { slot, .. } if *slot == iv_slot),
                    ) && !pre_insts[pos + 1..]
                        .iter()
                        .any(|i| matches!(i, IrInst::LoadLocal { slot, .. } if *slot == iv_slot))
                    {
                        pre_insts.remove(pos);
                    }
                    let cv = alloc_val();
                    pre_insts.push(IrInst::ConstInt {
                        dst: cv,
                        ty: u16t.clone(),
                        val: bc,
                    });
                    pre_insts.push(IrInst::StoreLocal {
                        slot: iv_slot,
                        val: cv,
                        ty: u16t.clone(),
                    });
                }
                PgKind::Guard {
                    init_bb_idx,
                    cmp_inst_idx,
                    cmp_lhs_load,
                } => {
                    let init_insts = &mut func.blocks[init_bb_idx].insts;
                    if let Some(pos) = init_insts.iter().rposition(
                        |i| matches!(i, IrInst::StoreLocal { slot, .. } if *slot == iv_slot),
                    ) && !init_insts[pos + 1..]
                        .iter()
                        .any(|i| matches!(i, IrInst::LoadLocal { slot, .. } if *slot == iv_slot))
                    {
                        init_insts.remove(pos);
                    }
                    let g_limit_v = if let Some(bc) = bound_const {
                        let cv = alloc_val();
                        func.blocks[init_bb_idx].insts.push(IrInst::ConstInt {
                            dst: cv,
                            ty: u16t.clone(),
                            val: bc,
                        });
                        cv
                    } else {
                        let lv = alloc_val();
                        func.blocks[init_bb_idx].insts.push(IrInst::LoadLocal {
                            dst: lv,
                            slot: bound_slot.unwrap(),
                            ty: u16t.clone(),
                        });
                        lv
                    };
                    func.blocks[init_bb_idx].insts.push(IrInst::StoreLocal {
                        slot: iv_slot,
                        val: g_limit_v,
                        ty: u16t.clone(),
                    });
                    // Replace `pg_idx`'s `iv < limit` with `iv != 0`.
                    let g_cur = alloc_val();
                    let g_zero = alloc_val();
                    let g_nz = alloc_val();
                    let pg_insts = &mut func.blocks[pg_idx].insts;
                    pg_insts.truncate(cmp_inst_idx);
                    pg_insts.retain(|i| i.dst() != Some(cmp_lhs_load));
                    pg_insts.extend([
                        IrInst::LoadLocal {
                            dst: g_cur,
                            slot: iv_slot,
                            ty: u16t.clone(),
                        },
                        IrInst::ConstInt {
                            dst: g_zero,
                            ty: u16t.clone(),
                            val: 0,
                        },
                        IrInst::BinOp {
                            dst: g_nz,
                            op: IrBinOp::Ne,
                            mode: ArithMode::Wrapping,
                            ty: u16t.clone(),
                            lhs: g_cur,
                            rhs: g_zero,
                            site_id: 0,
                        },
                    ]);
                    func.blocks[pg_idx].term = Terminator::Branch {
                        cond: g_nz,
                        then_bb: header,
                        else_bb,
                    };
                }
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
                    ty: u16t.clone(),
                },
                IrInst::ConstInt {
                    dst: c_1,
                    ty: u16t.clone(),
                    val: 1,
                },
                IrInst::BinOp {
                    dst: c_dec,
                    op: IrBinOp::Sub,
                    mode: ArithMode::Wrapping,
                    ty: u16t.clone(),
                    lhs: c_load,
                    rhs: c_1,
                    site_id: 0,
                },
                IrInst::StoreLocal {
                    slot: iv_slot,
                    val: c_dec,
                    ty: u16t.clone(),
                },
                IrInst::ConstInt {
                    dst: c_0,
                    ty: u16t.clone(),
                    val: 0,
                },
                IrInst::BinOp {
                    dst: c_nz,
                    op: IrBinOp::Ne,
                    mode: ArithMode::Wrapping,
                    ty: u16t.clone(),
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
            owner_state: None,
            owned_by_irq: None,
            export_name: None,
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
            owner_state: None,
            mock_target: None,
            export_name: None,
            cycle_budget: None,
            loop_bounds: HashMap::new(),
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
            owner_state: None,
            owned_by_irq: None,
            export_name: None,
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
            owner_state: None,
            mock_target: None,
            export_name: None,
            cycle_budget: None,
            loop_bounds: HashMap::new(),
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

    #[test]
    fn lowers_constant_fill_and_copy_loops_to_memset_and_memcpy_and_skips_volatile_or_overlap() {
        let u8t = IrType::Int(IntType::U8);
        let u16t = IrType::Int(IntType::U16);
        let mk_arr = |name: &str, len: u16| IrGlobal {
            name: name.into(),
            ty: IrType::Array {
                elem: Box::new(u8t.clone()),
                len,
            },
            mutable: true,
            in_rom: false,
            volatile: false,
            fixed_addr: None,
            is_extern: false,
            align: 1,
            soa: false,
            space: None,
            owner_state: None,
            owned_by_irq: None,
            export_name: None,
            init: vec![],
        };

        // 1. Byte fill loop -> MemSet
        let mut fill_prog = IrProgram::default();
        fill_prog.globals.push(mk_arr("DST", 64));
        fill_prog.functions.push(IrFunction {
            id: 0,
            name: "fill64".into(),
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
            owner_state: None,
            mock_target: None,
            export_name: None,
            cycle_budget: None,
            loop_bounds: HashMap::new(),
            blocks: vec![
                BasicBlock {
                    id: 0,
                    insts: vec![
                        IrInst::ConstInt { dst: 0, ty: u8t.clone(), val: 0 },
                        IrInst::StoreLocal { slot: 0, val: 0, ty: u8t.clone() },
                    ],
                    term: Terminator::Goto(1),
                },
                BasicBlock {
                    id: 1,
                    insts: vec![
                        IrInst::LoadLocal { dst: 1, slot: 0, ty: u8t.clone() },
                        IrInst::ConstInt { dst: 2, ty: u8t.clone(), val: 64 },
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
                    term: Terminator::Branch { cond: 3, then_bb: 2, else_bb: 3 },
                },
                BasicBlock {
                    id: 2,
                    insts: vec![
                        IrInst::AddrGlobal { dst: 4, global: 0 },
                        IrInst::LoadLocal { dst: 5, slot: 0, ty: u8t.clone() },
                        IrInst::Cast { dst: 6, val: 5, from_ty: u8t.clone(), to_ty: u16t.clone() },
                        IrInst::IndexAddr { dst: 7, base: 4, index: 6, elem_size: 1, len: 64, site_id: 0 },
                        IrInst::ConstInt { dst: 8, ty: u8t.clone(), val: 0xAA },
                        IrInst::StorePtr { addr: 7, val: 8, ty: u8t.clone(), volatile: false },
                        IrInst::LoadLocal { dst: 9, slot: 0, ty: u8t.clone() },
                        IrInst::ConstInt { dst: 10, ty: u8t.clone(), val: 1 },
                        IrInst::BinOp {
                            dst: 11,
                            op: IrBinOp::Add,
                            mode: ArithMode::Wrapping,
                            ty: u8t.clone(),
                            lhs: 9,
                            rhs: 10,
                            site_id: 1,
                        },
                        IrInst::StoreLocal { slot: 0, val: 11, ty: u8t.clone() },
                    ],
                    term: Terminator::Goto(1),
                },
                BasicBlock { id: 3, insts: vec![], term: Terminator::Return(None) },
            ],
            next_val: 12,
        });
        assert!(lower_memset_memcpy_loops(&mut fill_prog.functions[0], false));
        assert!(fill_prog.functions[0].blocks.iter().any(|b| b.insts.iter().any(|i| matches!(i, IrInst::MemSet { len: 64, .. }))));
        aun_ir::verify::verify_program(&fill_prog).expect("MemSet lowered prog verifies");

        // 2. Disjoint copy loop -> MemCopy; same-global overlapping copy -> NOT lowered.
        let mut copy_prog = IrProgram::default();
        copy_prog.globals.push(mk_arr("SRC", 32));
        copy_prog.globals.push(mk_arr("DST", 32));
        copy_prog.functions.push(IrFunction {
            id: 0,
            name: "copy32".into(),
            is_pub: true,
            param_count: 0,
            locals: vec![LocalSlot { id: 0, name: "i".into(), ty: u8t.clone(), mutable: true }],
            ret_ty: IrType::Unit,
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
                        IrInst::ConstInt { dst: 0, ty: u8t.clone(), val: 0 },
                        IrInst::StoreLocal { slot: 0, val: 0, ty: u8t.clone() },
                    ],
                    term: Terminator::Goto(1),
                },
                BasicBlock {
                    id: 1,
                    insts: vec![
                        IrInst::LoadLocal { dst: 1, slot: 0, ty: u8t.clone() },
                        IrInst::ConstInt { dst: 2, ty: u8t.clone(), val: 32 },
                        IrInst::BinOp { dst: 3, op: IrBinOp::Lt, mode: ArithMode::Wrapping, ty: u8t.clone(), lhs: 1, rhs: 2, site_id: 0 },
                    ],
                    term: Terminator::Branch { cond: 3, then_bb: 2, else_bb: 3 },
                },
                BasicBlock {
                    id: 2,
                    insts: vec![
                        IrInst::AddrGlobal { dst: 4, global: 0 },
                        IrInst::AddrGlobal { dst: 5, global: 1 },
                        IrInst::LoadLocal { dst: 6, slot: 0, ty: u8t.clone() },
                        IrInst::IndexAddr { dst: 7, base: 4, index: 6, elem_size: 1, len: 32, site_id: 0 },
                        IrInst::LoadPtr { dst: 8, addr: 7, ty: u8t.clone(), volatile: false },
                        IrInst::IndexAddr { dst: 9, base: 5, index: 6, elem_size: 1, len: 32, site_id: 1 },
                        IrInst::StorePtr { addr: 9, val: 8, ty: u8t.clone(), volatile: false },
                        IrInst::ConstInt { dst: 10, ty: u8t.clone(), val: 1 },
                        IrInst::BinOp { dst: 11, op: IrBinOp::Add, mode: ArithMode::Wrapping, ty: u8t.clone(), lhs: 6, rhs: 10, site_id: 2 },
                        IrInst::StoreLocal { slot: 0, val: 11, ty: u8t.clone() },
                    ],
                    term: Terminator::Goto(1),
                },
                BasicBlock { id: 3, insts: vec![], term: Terminator::Return(None) },
            ],
            next_val: 12,
        });
        assert!(lower_memset_memcpy_loops(&mut copy_prog.functions[0], false));
        assert!(copy_prog.functions[0].blocks.iter().any(|b| b.insts.iter().any(|i| matches!(i, IrInst::MemCopy { len: 32, .. }))));
        aun_ir::verify::verify_program(&copy_prog).expect("MemCopy lowered prog verifies");

        // Same global copy (could overlap) and volatile fill must NOT lower.
        let mut alias_copy = copy_prog.clone();
        // Rebuild original function targeting global 0 -> global 0 before lowering
        alias_copy.functions[0] = IrFunction {
            id: 0,
            name: "overlap".into(),
            is_pub: true,
            param_count: 0,
            locals: vec![LocalSlot { id: 0, name: "i".into(), ty: u8t.clone(), mutable: true }],
            ret_ty: IrType::Unit,
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
                        IrInst::ConstInt { dst: 0, ty: u8t.clone(), val: 0 },
                        IrInst::StoreLocal { slot: 0, val: 0, ty: u8t.clone() },
                    ],
                    term: Terminator::Goto(1),
                },
                BasicBlock {
                    id: 1,
                    insts: vec![
                        IrInst::LoadLocal { dst: 1, slot: 0, ty: u8t.clone() },
                        IrInst::ConstInt { dst: 2, ty: u8t.clone(), val: 32 },
                        IrInst::BinOp { dst: 3, op: IrBinOp::Lt, mode: ArithMode::Wrapping, ty: u8t.clone(), lhs: 1, rhs: 2, site_id: 0 },
                    ],
                    term: Terminator::Branch { cond: 3, then_bb: 2, else_bb: 3 },
                },
                BasicBlock {
                    id: 2,
                    insts: vec![
                        IrInst::AddrGlobal { dst: 4, global: 0 },
                        IrInst::FieldAddr { dst: 5, base: 4, byte_offset: 1 },
                        IrInst::LoadLocal { dst: 6, slot: 0, ty: u8t.clone() },
                        IrInst::IndexAddr { dst: 7, base: 4, index: 6, elem_size: 1, len: 32, site_id: 0 },
                        IrInst::LoadPtr { dst: 8, addr: 7, ty: u8t.clone(), volatile: false },
                        IrInst::IndexAddr { dst: 9, base: 5, index: 6, elem_size: 1, len: 32, site_id: 1 },
                        IrInst::StorePtr { addr: 9, val: 8, ty: u8t.clone(), volatile: false },
                        IrInst::ConstInt { dst: 10, ty: u8t.clone(), val: 1 },
                        IrInst::BinOp { dst: 11, op: IrBinOp::Add, mode: ArithMode::Wrapping, ty: u8t.clone(), lhs: 6, rhs: 10, site_id: 2 },
                        IrInst::StoreLocal { slot: 0, val: 11, ty: u8t.clone() },
                    ],
                    term: Terminator::Goto(1),
                },
                BasicBlock { id: 3, insts: vec![], term: Terminator::Return(None) },
            ],
            next_val: 12,
        };
        assert!(!lower_memset_memcpy_loops(&mut alias_copy.functions[0], false));
    }

    #[test]
    fn reduces_loop_bitmask_shifts_and_coalesces_zero_field_stores() {
        let u8t = IrType::Int(IntType::U8);
        let u16t = IrType::Int(IntType::U16);
        let mut mask_prog = IrProgram::default();
        mask_prog.functions.push(IrFunction {
            id: 0,
            name: "mask_loop".into(),
            is_pub: true,
            param_count: 1,
            locals: vec![
                LocalSlot { id: 0, name: "acc".into(), ty: u8t.clone(), mutable: true },
                LocalSlot { id: 1, name: "op".into(), ty: u8t.clone(), mutable: true },
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
                        IrInst::ConstInt { dst: 0, ty: u8t.clone(), val: 0 },
                        IrInst::StoreLocal { slot: 1, val: 0, ty: u8t.clone() },
                    ],
                    term: Terminator::Goto(1),
                },
                BasicBlock {
                    id: 1,
                    insts: vec![
                        IrInst::LoadLocal { dst: 1, slot: 1, ty: u8t.clone() },
                        IrInst::ConstInt { dst: 2, ty: u8t.clone(), val: 4 },
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
                    term: Terminator::Branch { cond: 3, then_bb: 2, else_bb: 3 },
                },
                BasicBlock {
                    id: 2,
                    insts: vec![
                        IrInst::ConstInt { dst: 4, ty: u8t.clone(), val: 1 },
                        IrInst::LoadLocal { dst: 5, slot: 1, ty: u8t.clone() },
                        IrInst::BinOp {
                            dst: 6,
                            op: IrBinOp::Shl,
                            mode: ArithMode::Wrapping,
                            ty: u8t.clone(),
                            lhs: 4,
                            rhs: 5,
                            site_id: 0,
                        },
                        IrInst::LoadLocal { dst: 7, slot: 0, ty: u8t.clone() },
                        IrInst::BinOp {
                            dst: 8,
                            op: IrBinOp::BitOr,
                            mode: ArithMode::Wrapping,
                            ty: u8t.clone(),
                            lhs: 7,
                            rhs: 6,
                            site_id: 1,
                        },
                        IrInst::StoreLocal { slot: 0, val: 8, ty: u8t.clone() },
                        IrInst::ConstInt { dst: 9, ty: u8t.clone(), val: 1 },
                        IrInst::BinOp {
                            dst: 10,
                            op: IrBinOp::Add,
                            mode: ArithMode::Wrapping,
                            ty: u8t.clone(),
                            lhs: 5,
                            rhs: 9,
                            site_id: 2,
                        },
                        IrInst::StoreLocal { slot: 1, val: 10, ty: u8t.clone() },
                    ],
                    term: Terminator::Goto(1),
                },
                BasicBlock {
                    id: 3,
                    insts: vec![IrInst::LoadLocal { dst: 11, slot: 0, ty: u8t.clone() }],
                    term: Terminator::Return(Some(11)),
                },
            ],
            next_val: 12,
        });
        assert!(reduce_loop_bitmask_shifts(&mut mask_prog.functions[0]));
        assert!(
            !mask_prog.functions[0].blocks[2]
                .insts
                .iter()
                .take(3)
                .any(|i| matches!(i, IrInst::BinOp { op: IrBinOp::Shl, .. })),
            "body Shl by iv should be replaced with LoadLocal(_osr_mask)"
        );
        aun_ir::verify::verify_program(&mask_prog).expect("bitmask OSR prog verifies");

        // Zero-field StorePtr coalescing -> MemSet(len = 8) on global struct pointer
        let mut zf_prog = IrProgram::default();
        zf_prog.globals.push(IrGlobal {
            name: "STRUCT".into(),
            ty: IrType::Array {
                elem: Box::new(u8t.clone()),
                len: 32,
            },
            mutable: true,
            in_rom: false,
            volatile: false,
            fixed_addr: None,
            is_extern: false,
            align: 1,
            soa: false,
            space: None,
            owner_state: None,
            owned_by_irq: None,
            export_name: None,
            init: vec![],
        });
        zf_prog.functions.push(IrFunction {
            id: 0,
            name: "zero_fields".into(),
            is_pub: true,
            param_count: 0,
            locals: vec![],
            ret_ty: IrType::Unit,
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
                    IrInst::AddrGlobal { dst: 0, global: 0 },
                    IrInst::ConstInt { dst: 1, ty: u8t.clone(), val: 0 },
                    IrInst::ConstInt { dst: 2, ty: u16t.clone(), val: 0 },
                    IrInst::FieldAddr { dst: 3, base: 0, byte_offset: 13 },
                    IrInst::StorePtr { addr: 3, val: 1, ty: u8t.clone(), volatile: false },
                    IrInst::FieldAddr { dst: 4, base: 0, byte_offset: 14 },
                    IrInst::StorePtr { addr: 4, val: 1, ty: u8t.clone(), volatile: false },
                    IrInst::FieldAddr { dst: 5, base: 0, byte_offset: 15 },
                    IrInst::StorePtr { addr: 5, val: 2, ty: u16t.clone(), volatile: false },
                    IrInst::FieldAddr { dst: 6, base: 0, byte_offset: 17 },
                    IrInst::StorePtr { addr: 6, val: 2, ty: u16t, volatile: false },
                    IrInst::FieldAddr { dst: 7, base: 0, byte_offset: 19 },
                    IrInst::StorePtr { addr: 7, val: 1, ty: u8t.clone(), volatile: false },
                    IrInst::FieldAddr { dst: 8, base: 0, byte_offset: 20 },
                    IrInst::StorePtr { addr: 8, val: 1, ty: u8t, volatile: false },
                ],
                term: Terminator::Return(None),
            }],
            next_val: 9,
        });
        assert!(coalesce_zero_field_stores(&mut zf_prog.functions[0]));
        assert!(zf_prog.functions[0].blocks[0]
            .insts
            .iter()
            .any(|i| matches!(i, IrInst::MemSet { len: 8, .. })));
        aun_ir::verify::verify_program(&zf_prog).expect("coalesced zero-field prog verifies");
    }

    #[test]
    fn lower_mem_write_u8_and_mem_read_u8_copy_loops() {
        let src = r#"
struct Hud {
    line: [u8; 44],
}

#[align(256)]
static mut G_HUD: Hud = Hud { line: [0u8; 44] };

pub fn tvram_flush(dst: u16) {
    let col0: u16 = 2u16;
    for i in 0u8..44u8 {
        mem_write_u8(dst +% col0 +% (i as u16), G_HUD.line[i as u16]);
    }
}

pub fn read_vram_loop(src: u16) {
    for i in 0u8..44u8 {
        G_HUD.line[i as u16] = mem_read_u8(src +% (i as u16));
    }
}

pub fn fill_vram_loop(dst: u16) {
    for i in 0u8..32u8 {
        mem_write_u8(dst +% (i as u16), 0x20u8);
    }
}

pub fn fixed_overlap_copy() {
    for i in 0u8..16u8 {
        mem_write_u8(0xD001u16 +% (i as u16), mem_read_u8(0xD000u16 +% (i as u16)));
    }
}
"#;
        let mut prog = aun_front::compile_str(0, src).expect("front compile");
        aun_ir::opt::optimize_program(&mut prog);
        let globals = prog.globals.clone();

        // 1. tvram_flush: mem_write_u8(dst + col0 + i, G_HUD.line[i]) -> MemCopy(len = 44)
        let f_flush = prog
            .functions
            .iter_mut()
            .find(|f| f.name == "tvram_flush")
            .unwrap();
        assert!(lower_memset_memcpy_loops_with_globals(f_flush, &globals, true));
        assert!(
            f_flush
                .blocks
                .iter()
                .flat_map(|b| b.insts.iter())
                .any(|i| matches!(i, IrInst::MemCopy { len: 44, .. })),
            "tvram_flush should lower to MemCopy(len=44)"
        );

        // 2. read_vram_loop: G_HUD.line[i] = mem_read_u8(src + i) -> MemCopy(len = 44)
        let f_read = prog
            .functions
            .iter_mut()
            .find(|f| f.name == "read_vram_loop")
            .unwrap();
        assert!(lower_memset_memcpy_loops_with_globals(f_read, &globals, true));
        assert!(
            f_read
                .blocks
                .iter()
                .flat_map(|b| b.insts.iter())
                .any(|i| matches!(i, IrInst::MemCopy { len: 44, .. })),
            "read_vram_loop should lower to MemCopy(len=44)"
        );

        // 3. fill_vram_loop: mem_write_u8(dst + i, 0x20) -> MemSet(len = 32)
        let f_fill = prog
            .functions
            .iter_mut()
            .find(|f| f.name == "fill_vram_loop")
            .unwrap();
        assert!(lower_memset_memcpy_loops_with_globals(f_fill, &globals, true));
        assert!(
            f_fill
                .blocks
                .iter()
                .flat_map(|b| b.insts.iter())
                .any(|i| matches!(i, IrInst::MemSet { len: 32, .. })),
            "fill_vram_loop should lower to MemSet(len=32)"
        );

        // 4. fixed_overlap_copy: overlapping fixed addresses must NOT lower to MemCopy
        let f_alias = prog
            .functions
            .iter_mut()
            .find(|f| f.name == "fixed_overlap_copy")
            .unwrap();
        assert!(!lower_memset_memcpy_loops_with_globals(f_alias, &globals, true));
        assert!(
            !f_alias
                .blocks
                .iter()
                .flat_map(|b| b.insts.iter())
                .any(|i| matches!(i, IrInst::MemCopy { .. })),
            "fixed_overlap_copy must not lower overlapping fixed addresses to MemCopy"
        );

        aun_ir::verify::verify_program(&prog).expect("lowered MIR must pass verifier");
    }

    #[test]
    fn osr_slice_loadptr_and_u16_countdown() {
        let src = r#"
struct Hud {
    line: [u8; 44],
}

#[align(256)]
static mut G_HUD: Hud = Hud { line: [0u8; 44] };

pub fn put_str(col: u8, s: &const [u8]) {
    let mut c: u8 = col;
    let mut i: u16 = 0u16;
    while i < s.len() {
        G_HUD.line[c as u16] = s[i];
        c = c +% 1u8;
        i = i +% 1u16;
    }
}

pub fn put_str_for(col: u8, s: &const [u8]) {
    let mut c: u8 = col;
    for i in 0u16..s.len() {
        G_HUD.line[c as u16] = s[i];
        c = c +% 1u8;
    }
}
"#;
        let mut prog = aun_front::compile_str(0, src).expect("front compile");
        aun_ir::opt::optimize_program(&mut prog);
        optimize_z80_program(&mut prog, true);
        aun_ir::verify::verify_program(&prog).expect("optimized MIR must pass verifier");

        for fname in ["put_str", "put_str_for"] {
            let f = prog.functions.iter().find(|f| f.name == fname).unwrap();
            assert!(
                f.locals.iter().any(|l| l.name.starts_with("__osr_ptr_")),
                "{fname} should promote slice iteration to __osr_ptr_*"
            );
            assert!(
                !f.blocks
                    .iter()
                    .flat_map(|b| b.insts.iter())
                    .any(|i| matches!(i, IrInst::IndexAddr { len: 0, .. })),
                "{fname} should eliminate slice IndexAddr via OSR"
            );
            assert!(
                f.blocks.iter().flat_map(|b| b.insts.iter()).any(|i| {
                    matches!(i, IrInst::BinOp { op: IrBinOp::Sub, .. })
                }) && f.blocks.iter().flat_map(|b| b.insts.iter()).any(|i| {
                    matches!(i, IrInst::BinOp { op: IrBinOp::Ne, .. })
                }),
                "{fname} should convert u16 countup loop to countdown"
            );
        }
    }

    #[test]
    fn osr_byte_copy_loop_put_str_nul_and_col_cap() {
        let src = r#"
pub fn put_str(col: u8, s: &const [u8]) -> u8 {
    let mut c: u8 = col;
    let mut i: u16 = 0u16;
    while i < s.len() && c < 44u8 {
        let ch: u8 = s[i];
        if ch == 0u8 {
            return c;
        }
        mem_write_u8(0xF100u16 +% (c as u16), ch);
        c = c +% 1u8;
        i = i +% 1u16;
    }
    c
}
"#;
        let mut prog = aun_front::compile_str(0, src).expect("front compile");
        aun_ir::opt::optimize_program(&mut prog);
        optimize_z80_program(&mut prog, true);
        aun_ir::verify::verify_program(&prog).expect("optimized MIR must pass verifier");

        let analysis = crate::analysis::ProgramAnalysis::analyze(&prog);
        let fa = &analysis.fns[0];
        assert_eq!(
            fa.byte_copy_loops.len(),
            1,
            "put_str should be recognized as a ByteCopyLoop:\n{:?}",
            prog.functions[0].blocks
        );
        let bcl = &fa.byte_copy_loops[0];
        assert!(bcl.is_u8_cnt, "col < 44 cap guarantees u8 djnz trip count");
        assert!(
            matches!(bcl.sentinel, Some((0, _))),
            "ch == 0 early exit must be preserved, got {:?}",
            bcl.sentinel
        );
        assert!(bcl.inc_e_fixed, "0xF100 + c stays within a 256-byte page");
    }

    #[test]
    fn osr_byte_copy_loop_iv_bound_exit_and_256b_static_struct_field() {
        let src = r#"
const HUD_WIDTH: u8 = 44u8;

pub struct HudWork {
    line: [u8; 44],
    dirty: u8,
}

static mut G_HUD: HudWork @ 0xF100;

pub struct HudOffset {
    pad: [u8; 256],
    line: [u8; 44],
}

static mut G_HUD_OFF: HudOffset @ 0xF000;

#[align(256)]
static mut G_HUD_RAM: HudWork = HudWork { line: [0u8; 44], dirty: 0u8 };

static mut G_PLAIN_ARR: [u8; 44] @ 0xF300;

pub struct UnalignedHud {
    pad: [u8; 4],
    line: [u8; 44],
}

static mut G_HUD_UNALIGNED: UnalignedHud @ 0xF100;

pub fn put_str_hud(pos_in: u8, s: &const [u8]) {
    let mut pos: u8 = pos_in;
    let n: u16 = s.len();
    let mut i: u16 = 0u16;
    while i < n && pos < HUD_WIDTH {
        let ch: u8 = s[i];
        if ch == 0u8 {
            i = n;
        } else {
            G_HUD.line[pos as u16] = ch;
            pos = pos +% 1u8;
            i = i +% 1u16;
        }
    }
}

pub fn put_str_offset256(pos_in: u8, s: &const [u8]) {
    let mut pos: u8 = pos_in;
    let n: u16 = s.len();
    let mut i: u16 = 0u16;
    while i < n && pos < HUD_WIDTH {
        let ch: u8 = s[i];
        if ch == 0u8 {
            i = n;
        } else {
            G_HUD_OFF.line[pos as u16] = ch;
            pos = pos +% 1u8;
            i = i +% 1u16;
        }
    }
}

pub fn put_str_ram256(pos_in: u8, s: &const [u8]) {
    let mut pos: u8 = pos_in;
    let n: u16 = s.len();
    let mut i: u16 = 0u16;
    while i < n && pos < HUD_WIDTH {
        let ch: u8 = s[i];
        if ch == 0u8 {
            i = n;
        } else {
            G_HUD_RAM.line[pos as u16] = ch;
            pos = pos +% 1u8;
            i = i +% 1u16;
        }
    }
}

pub fn put_str_plain_arr(pos_in: u8, s: &const [u8]) {
    let mut pos: u8 = pos_in;
    let n: u16 = s.len();
    let mut i: u16 = 0u16;
    while i < n && pos < HUD_WIDTH {
        let ch: u8 = s[i];
        if ch == 0u8 {
            i = n;
        } else {
            G_PLAIN_ARR[pos as u16] = ch;
            pos = pos +% 1u8;
            i = i +% 1u16;
        }
    }
}

pub fn put_str_unaligned(pos_in: u8, s: &const [u8]) {
    let mut pos: u8 = pos_in;
    let n: u16 = s.len();
    let mut i: u16 = 0u16;
    while i < n && pos < HUD_WIDTH {
        let ch: u8 = s[i];
        if ch == 0u8 {
            i = n;
        } else {
            G_HUD_UNALIGNED.line[pos as u16] = ch;
            pos = pos +% 1u8;
            i = i +% 1u16;
        }
    }
}

// Negative soundness tests: MUST NOT become ByteCopyLoop!
pub fn bad_wrong_bound_assign(pos_in: u8, s: &const [u8], m: u16) {
    let mut pos: u8 = pos_in;
    let n: u16 = s.len();
    let mut i: u16 = 0u16;
    while i < n && pos < HUD_WIDTH {
        let ch: u8 = s[i];
        if ch == 0u8 {
            i = m;
        } else {
            G_HUD.line[pos as u16] = ch;
            pos = pos +% 1u8;
            i = i +% 1u16;
        }
    }
}

pub fn bad_iv_live_out(pos_in: u8, s: &const [u8]) -> u16 {
    let mut pos: u8 = pos_in;
    let n: u16 = s.len();
    let mut i: u16 = 0u16;
    while i < n && pos < HUD_WIDTH {
        let ch: u8 = s[i];
        if ch == 0u8 {
            i = n;
        } else {
            G_HUD.line[pos as u16] = ch;
            pos = pos +% 1u8;
            i = i +% 1u16;
        }
    }
    i
}

pub fn bad_inclusive_equal_exit(pos_in: u8, s: &const [u8]) {
    let mut pos: u8 = pos_in;
    let mut i: u16 = 0u16;
    while i <= 10u16 && pos < HUD_WIDTH {
        let ch: u8 = s[i];
        if ch == 0u8 {
            i = 10u16;
        } else {
            G_HUD.line[pos as u16] = ch;
            pos = pos +% 1u8;
            i = i +% 1u16;
        }
    }
}

pub fn bad_side_effect_in_early_arm(pos_in: u8, s: &const [u8]) -> u8 {
    let mut pos: u8 = pos_in;
    let n: u16 = s.len();
    let mut i: u16 = 0u16;
    while i < n && pos < HUD_WIDTH {
        let ch: u8 = s[i];
        if ch == 0u8 {
            pos = pos +% 1u8;
            i = n;
        } else {
            G_HUD.line[pos as u16] = ch;
            pos = pos +% 1u8;
            i = i +% 1u16;
        }
    }
    pos
}
"#;
        for is_release in [false, true] {
            let mut prog = aun_front::compile_str(0, src).expect("front compile");
            aun_ir::opt::optimize_program(&mut prog);
            optimize_z80_program(&mut prog, is_release);
            aun_ir::verify::verify_program(&prog).expect("optimized MIR must pass verifier");

            let analysis = crate::analysis::ProgramAnalysis::analyze(&prog);
            let get_fa = |name: &str| {
                let idx = prog.functions.iter().position(|f| f.name == name).unwrap();
                (&prog.functions[idx], &analysis.fns[idx])
            };

            let (f_hud, fa_hud) = get_fa("put_str_hud");
            assert_eq!(
                fa_hud.byte_copy_loops.len(),
                1,
                "put_str_hud (release={is_release}) should trigger ByteCopyLoop:\n{:?}",
                f_hud.blocks
            );
            let bcl_hud = &fa_hud.byte_copy_loops[0];
            assert!(bcl_hud.is_u8_cnt);
            assert!(matches!(bcl_hud.sentinel, Some((0, true))));
            assert!(bcl_hud.inc_e_fixed, "0xF100 + 0 is 256B-aligned fixed");
            assert!(!bcl_hud.dst_live_out);

            let (_, fa_off) = get_fa("put_str_offset256");
            assert_eq!(fa_off.byte_copy_loops.len(), 1);
            assert!(
                fa_off.byte_copy_loops[0].inc_e_fixed,
                "0xF000 + 256 is 256B-aligned fixed"
            );

            let (_, fa_ram) = get_fa("put_str_ram256");
            assert_eq!(fa_ram.byte_copy_loops.len(), 1);
            assert!(
                fa_ram.byte_copy_loops[0].inc_e_ram_global,
                "#[align(256)] G_HUD_RAM.line at offset 0 should set inc_e_ram_global"
            );

            let (_, fa_arr) = get_fa("put_str_plain_arr");
            assert_eq!(fa_arr.byte_copy_loops.len(), 1);
            assert!(
                fa_arr.byte_copy_loops[0].inc_e_fixed,
                "0xF300 plain static array should set inc_e_fixed"
            );

            let (_, fa_unaligned) = get_fa("put_str_unaligned");
            assert_eq!(fa_unaligned.byte_copy_loops.len(), 1);
            assert!(
                !fa_unaligned.byte_copy_loops[0].inc_e_fixed,
                "0xF100 + 4 is NOT 256B-aligned, must use inc de"
            );

            for bad in [
                "bad_wrong_bound_assign",
                "bad_iv_live_out",
                "bad_inclusive_equal_exit",
                "bad_side_effect_in_early_arm",
            ] {
                let (_, fa_bad) = get_fa(bad);
                assert!(
                    fa_bad.byte_copy_loops.is_empty(),
                    "{bad} (release={is_release}) must NOT trigger ByteCopyLoop"
                );
            }
        }
    }

    #[test]
    fn sinks_cold_guard_spills_and_allows_guarded_iv_in_djnz_loop() {
        let src = r#"
pub struct Slot {
    active: u8,
    timer: u8,
    tag: u8,
    pad: u8,
}

static mut SLOTS: [Slot; 6] = [
    Slot { active: 0u8, timer: 0u8, tag: 0u8, pad: 0u8 }; 6
];
static mut OUT: [u8; 6] = [0u8; 6];

pub fn step_slots() {
    for k in 0u8..6u8 {
        let ki: u16 = k as u16;
        if SLOTS[ki].active != 0u8 {
            SLOTS[ki].timer = SLOTS[ki].timer +% 1u8;
            OUT[k as u16] = k +% SLOTS[ki].timer;
        }
    }
}
"#;
        let mut prog = aun_front::compile_str(0, src).expect("front compile");
        aun_ir::opt::optimize_program(&mut prog);
        optimize_z80_program(&mut prog, true);
        aun_ir::verify::verify_program(&prog).expect("optimized MIR must pass verifier");

        let f = prog.functions.iter().find(|f| f.name == "step_slots").unwrap();
        // Loop header (branch testing `active != 0`) should NOT contain Cast(u8 -> u16) for cold indexing.
        let header_bb = f
            .blocks
            .iter()
            .find(|b| matches!(b.term, Terminator::Branch { .. })
                && b.insts.iter().any(|i| matches!(i, IrInst::LoadPtr { .. })))
            .expect("expected guard header block");
        assert!(
            !header_bb.insts.iter().any(|i| matches!(i, IrInst::Cast { .. })),
            "cold cast `k as u16` should be sunk out of hot guard header:\n{header_bb:?}"
        );
        // Loop latch should be converted to countdown (`Sub 1`, `Ne 0`) despite cold block using `k`.
        assert!(
            f.blocks.iter().any(|b| {
                matches!(b.term, Terminator::Branch { .. })
                    && b.insts.iter().any(|i| matches!(i, IrInst::BinOp { op: IrBinOp::Sub, .. }))
                    && b.insts.iter().any(|i| matches!(i, IrInst::BinOp { op: IrBinOp::Ne, .. }))
            }),
            "guarded use of `k` should still convert loop to countdown:\n{f:?}"
        );
    }

    #[test]
    fn rewrites_or_aborts_countdown_on_early_exit_iv_uses() {
        let src = r#"
struct Slot { active: u8, val: u8 }
static mut SLOTS: [Slot; 3] = [Slot { active: 0u8, val: 0u8 }; 3];
static mut FOUND: u8 = 0xFFu8;

pub fn find_return(v: u8) {
    for i in 0u8..3u8 {
        let ii: u16 = i as u16;
        if SLOTS[ii].active == 0u8 {
            SLOTS[ii].active = 1u8;
            SLOTS[ii].val = v;
            FOUND = i;
            return;
        }
    }
}

pub fn reuse_after_break(stop: u8) -> u8 {
    let mut i: u8 = 0u8;
    while i < 4u8 {
        if i == stop {
            break;
        }
        i = i +% 1u8;
    }
    while i < 6u8 {
        i = i +% 1u8;
    }
    i
}

pub fn find_u16_return(n: u16, want: u16) -> u16 {
    for i in 0u16..n {
        if i == want {
            return i;
        }
    }
    0xFFFFu16
}
"#;
        let mut prog = aun_front::compile_str(0, src).expect("front compile");
        aun_ir::opt::optimize_program(&mut prog);
        optimize_z80_program(&mut prog, true);
        aun_ir::verify::verify_program(&prog).expect("optimized MIR must pass verifier");

        // 1. `find_return` converts to countdown AND rewrites early-exit LoadLocal(iv) to `3 - rem` (never raw `rem`).
        let f_ret = prog.functions.iter().find(|f| f.name == "find_return").unwrap();
        assert!(
            f_ret.blocks.iter().any(|b| {
                matches!(b.term, Terminator::Branch { .. })
                    && b.insts.iter().any(|i| matches!(i, IrInst::BinOp { op: IrBinOp::Sub, .. }))
                    && b.insts.iter().any(|i| matches!(i, IrInst::BinOp { op: IrBinOp::Ne, .. }))
            }),
            "find_return should still convert to countdown:\n{f_ret:?}"
        );
        let has_3_sub = f_ret.blocks.iter().any(|b| {
            matches!(b.term, Terminator::Return(_))
                && b.insts.iter().any(|i| matches!(i, IrInst::BinOp { op: IrBinOp::Sub, .. }))
        });
        assert!(
            has_3_sub,
            "find_return early-exit block must compute `3 - rem`:\n{f_ret:?}"
        );

        // 2. `reuse_after_break` must NOT convert loop 1 (first `i < 4u8` loop) to countdown
        // because `i` is later read & mutated by the second loop!
        let f_reuse = prog.functions.iter().find(|f| f.name == "reuse_after_break").unwrap();
        assert!(
            !f_reuse.blocks.iter().any(|b| {
                matches!(b.term, Terminator::Branch { .. })
                    && b.insts.iter().any(|i| matches!(i, IrInst::BinOp { op: IrBinOp::Sub, .. }))
                    && b.insts.iter().any(|i| matches!(i, IrInst::BinOp { op: IrBinOp::Ne, .. }))
            }),
            "reuse_after_break must abort countdown conversion when `i` is live & mutated after break:\n{f_reuse:?}"
        );

        // 3. `find_u16_return` must NOT convert `u16` loop to countdown because early-return reads `i`.
        let f_u16 = prog.functions.iter().find(|f| f.name == "find_u16_return").unwrap();
        assert!(
            !f_u16.blocks.iter().any(|b| {
                matches!(b.term, Terminator::Branch { .. })
                    && b.insts.iter().any(|i| matches!(i, IrInst::BinOp { op: IrBinOp::Ne, ty: IrType::Int(IntType::U16), .. }))
            }),
            "find_u16_return must abort u16 countdown conversion when early return reads `i`:\n{f_u16:?}"
        );
    }

    fn make_ball_struct() -> StructDef {
        StructDef {
            name: "Ball".into(),
            fields: vec![
                ("active".into(), IrType::Int(IntType::U8), 0),
                ("x".into(), IrType::Fixed { base: IntType::I16, frac_bits: 6 }, 1),
                ("y".into(), IrType::Fixed { base: IntType::I16, frac_bits: 6 }, 3),
                ("x_px".into(), IrType::Int(IntType::U8), 9),
                ("y_px".into(), IrType::Int(IntType::U8), 10),
                ("flag".into(), IrType::Int(IntType::U8), 12),
            ],
            size: 13,
        }
    }

    fn make_test_fn(
        name: &str,
        param_count: usize,
        locals: Vec<LocalSlot>,
        ret_ty: IrType,
        blocks: Vec<BasicBlock>,
        next_val: ValueId,
    ) -> IrFunction {
        IrFunction {
            id: 0,
            name: name.into(),
            is_pub: true,
            param_count,
            locals,
            ret_ty,
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            isr_using: None,
            requires_banks: Vec::new(),
            requires_di: false,
            overlay_state: None,
            owner_state: None,
            mock_target: None,
            export_name: None,
            cycle_budget: None,
            loop_bounds: HashMap::new(),
            blocks,
            next_val,
        }
    }

    #[test]
    fn issue51_sink_cold_loads_and_guard_spills_respects_soundness_barriers() {
        let u8t = IrType::Int(IntType::U8);
        let q10_6 = IrType::Fixed {
            base: IntType::I16,
            frac_bits: 6,
        };
        let ref_ball = IrType::Ref {
            mutable: true,
            inner: Box::new(IrType::Struct(0)),
        };
        let structs = vec![make_ball_struct()];

        // 1. Positive: cold-only LoadPtrs (`orig_x`, `orig_y`) before `if b.flag == 0 { return 0; }`
        //    must sink out of bb0 into bb2 (the cold target).
        let mut f_pos = make_test_fn(
            "step_cold",
            1,
            vec![
                LocalSlot { id: 0, name: "b".into(), ty: ref_ball.clone(), mutable: false },
                LocalSlot { id: 1, name: "orig_x".into(), ty: q10_6.clone(), mutable: false },
            ],
            IrType::Int(IntType::I16),
            vec![
                BasicBlock {
                    id: 0,
                    insts: vec![
                        IrInst::LoadLocal { dst: 0, slot: 0, ty: ref_ball.clone() },
                        IrInst::FieldAddr { dst: 1, base: 0, byte_offset: 1 },
                        IrInst::LoadPtr { dst: 2, addr: 1, ty: q10_6.clone(), volatile: false },
                        IrInst::StoreLocal { slot: 1, val: 2, ty: q10_6.clone() },
                        IrInst::FieldAddr { dst: 3, base: 0, byte_offset: 3 },
                        IrInst::LoadPtr { dst: 4, addr: 3, ty: q10_6.clone(), volatile: false },
                        IrInst::FieldAddr { dst: 5, base: 0, byte_offset: 12 },
                        IrInst::LoadPtr { dst: 6, addr: 5, ty: u8t.clone(), volatile: false },
                        IrInst::ConstInt { dst: 7, ty: u8t.clone(), val: 0 },
                        IrInst::BinOp {
                            dst: 8,
                            op: IrBinOp::Eq,
                            mode: ArithMode::Wrapping,
                            ty: u8t.clone(),
                            lhs: 6,
                            rhs: 7,
                            site_id: 0,
                        },
                    ],
                    term: Terminator::Branch {
                        cond: 8,
                        then_bb: 1,
                        else_bb: 2,
                    },
                },
                BasicBlock {
                    id: 1,
                    insts: vec![IrInst::ConstInt {
                        dst: 9,
                        ty: IrType::Int(IntType::I16),
                        val: 0,
                    }],
                    term: Terminator::Return(Some(9)),
                },
                BasicBlock {
                    id: 2,
                    insts: vec![
                        IrInst::LoadLocal { dst: 10, slot: 1, ty: q10_6.clone() },
                        IrInst::BinOp {
                            dst: 11,
                            op: IrBinOp::Add,
                            mode: ArithMode::Wrapping,
                            ty: q10_6.clone(),
                            lhs: 10,
                            rhs: 4,
                            site_id: 0,
                        },
                        IrInst::Cast {
                            dst: 12,
                            val: 11,
                            from_ty: q10_6.clone(),
                            to_ty: IrType::Int(IntType::I16),
                        },
                    ],
                    term: Terminator::Return(Some(12)),
                },
            ],
            13,
        );
        assert!(sink_cold_guard_spills_with_context(&mut f_pos, &structs, &[], &[]));
        assert_eq!(
            f_pos.blocks[0]
                .insts
                .iter()
                .filter(|i| matches!(i, IrInst::LoadPtr { .. }))
                .count(),
            1,
            "bb0 should only keep the guard LoadPtr (`flag`), sinking `orig_x` and `orig_y`:\n{:?}",
            f_pos.blocks[0]
        );
        assert_eq!(
            f_pos.blocks[2]
                .insts
                .iter()
                .filter(|i| matches!(i, IrInst::LoadPtr { .. }))
                .count(),
            2,
            "bb2 should receive both sunk LoadPtrs:\n{:?}",
            f_pos.blocks[2]
        );

        // 2. Positive multi-hop LCA: diamond bb0 -> (bb1 | bb2) -> bb3 (cold merge) -> LoadPtr in bb0
        //    only used in bb3 sinks to bb3 when paths are memory-invariant.
        let mut f_lca = make_test_fn(
            "lca_sink",
            2,
            vec![
                LocalSlot { id: 0, name: "b".into(), ty: ref_ball.clone(), mutable: false },
                LocalSlot { id: 1, name: "c".into(), ty: u8t.clone(), mutable: false },
            ],
            u8t.clone(),
            vec![
                BasicBlock {
                    id: 0,
                    insts: vec![
                        IrInst::LoadLocal { dst: 0, slot: 0, ty: ref_ball.clone() },
                        IrInst::FieldAddr { dst: 1, base: 0, byte_offset: 9 },
                        IrInst::LoadPtr { dst: 2, addr: 1, ty: u8t.clone(), volatile: false },
                        IrInst::LoadLocal { dst: 3, slot: 1, ty: u8t.clone() },
                        IrInst::ConstInt { dst: 4, ty: u8t.clone(), val: 0 },
                        IrInst::BinOp {
                            dst: 5,
                            op: IrBinOp::Eq,
                            mode: ArithMode::Wrapping,
                            ty: u8t.clone(),
                            lhs: 3,
                            rhs: 4,
                            site_id: 0,
                        },
                    ],
                    term: Terminator::Branch { cond: 5, then_bb: 1, else_bb: 2 },
                },
                BasicBlock {
                    id: 1,
                    insts: vec![],
                    term: Terminator::Goto(3),
                },
                BasicBlock {
                    id: 2,
                    insts: vec![],
                    term: Terminator::Goto(3),
                },
                BasicBlock {
                    id: 3,
                    insts: vec![],
                    term: Terminator::Return(Some(2)),
                },
            ],
            6,
        );
        assert!(sink_cold_loads_with_context(&mut f_lca, &structs, &[], &[]));
        assert!(
            !f_lca.blocks[0].insts.iter().any(|i| matches!(i, IrInst::LoadPtr { .. })),
            "bb0 should sink LoadPtr across diamond to bb3:\n{:?}",
            f_lca.blocks[0]
        );
        assert!(
            f_lca.blocks[3].insts.iter().any(|i| matches!(i, IrInst::LoadPtr { dst: 2, .. })),
            "bb3 should contain sunk LoadPtr:\n{:?}",
            f_lca.blocks[3]
        );

        // 3. Negative soundness tests: aliasing StorePtr, Call, MemSet, PoolAlloc, QueuePush,
        //    StateTransition, InlineAsm, volatile LoadPtr.
        let ptr_u8 = IrType::Ref {
            mutable: true,
            inner: Box::new(u8t.clone()),
        };
        for (label, barrier) in [
            (
                "unknown_store_ptr",
                IrInst::StorePtr {
                    addr: 20,
                    val: 21,
                    ty: u8t.clone(),
                    volatile: false,
                },
            ),
            (
                "memset",
                IrInst::MemSet {
                    dst_addr: 20,
                    val: 21,
                    len: 4,
                },
            ),
            (
                "call",
                IrInst::Call {
                    dst: None,
                    func: 1,
                    args: vec![],
                },
            ),
            (
                "pool_alloc",
                IrInst::PoolAlloc {
                    dst: 22,
                    pool_addr: 20,
                    init_addr: 20,
                    elem_size: 4,
                    capacity: 2,
                },
            ),
            (
                "queue_push",
                IrInst::QueuePush {
                    dst: 22,
                    queue_addr: 20,
                    val: 21,
                    elem_ty: u8t.clone(),
                    capacity: 2,
                },
            ),
            (
                "state_transition",
                IrInst::StateTransition {
                    machine_addr: 20,
                    machine_id: 0,
                    from_state: Some(0),
                    to_state: 1,
                    field_vals: Vec::new(),
                    site_id: 0,
                },
            ),
            (
                "inline_asm",
                IrInst::InlineAsm {
                    template: "nop".into(),
                    clobbers: Vec::new(),
                },
            ),
            (
                "volatile_load",
                IrInst::LoadPtr {
                    dst: 22,
                    addr: 20,
                    ty: u8t.clone(),
                    volatile: true,
                },
            ),
        ] {
            let mut f_neg = make_test_fn(
                &format!("neg_{label}"),
                2,
                vec![
                    LocalSlot { id: 0, name: "b".into(), ty: ref_ball.clone(), mutable: false },
                    LocalSlot { id: 1, name: "unk".into(), ty: ptr_u8.clone(), mutable: false },
                ],
                u8t.clone(),
                vec![
                    BasicBlock {
                        id: 0,
                        insts: vec![
                            IrInst::LoadLocal { dst: 0, slot: 0, ty: ref_ball.clone() },
                            IrInst::LoadLocal { dst: 20, slot: 1, ty: ptr_u8.clone() },
                            IrInst::ConstInt { dst: 21, ty: u8t.clone(), val: 7 },
                            IrInst::FieldAddr { dst: 1, base: 0, byte_offset: 9 },
                            IrInst::LoadPtr { dst: 2, addr: 1, ty: u8t.clone(), volatile: false },
                            barrier,
                            IrInst::FieldAddr { dst: 3, base: 0, byte_offset: 12 },
                            IrInst::LoadPtr { dst: 4, addr: 3, ty: u8t.clone(), volatile: false },
                            IrInst::ConstInt { dst: 5, ty: u8t.clone(), val: 0 },
                            IrInst::BinOp {
                                dst: 6,
                                op: IrBinOp::Eq,
                                mode: ArithMode::Wrapping,
                                ty: u8t.clone(),
                                lhs: 4,
                                rhs: 5,
                                site_id: 0,
                            },
                        ],
                        term: Terminator::Branch { cond: 6, then_bb: 1, else_bb: 2 },
                    },
                    BasicBlock {
                        id: 1,
                        insts: vec![IrInst::ConstInt { dst: 7, ty: u8t.clone(), val: 0 }],
                        term: Terminator::Return(Some(7)),
                    },
                    BasicBlock {
                        id: 2,
                        insts: vec![],
                        term: Terminator::Return(Some(2)),
                    },
                ],
                25,
            );
            sink_cold_guard_spills_with_context(&mut f_neg, &structs, &[], &[]);
            sink_cold_loads_with_context(&mut f_neg, &structs, &[], &[]);
            assert!(
                f_neg.blocks[0]
                    .insts
                    .iter()
                    .any(|i| matches!(i, IrInst::LoadPtr { dst: 2, .. })),
                "{label}: LoadPtr dst=2 MUST NOT sink past barrier:\n{:?}",
                f_neg.blocks[0]
            );
        }

        // 4. Negative: volatile LoadPtr itself must NEVER sink even with no barriers.
        let mut f_vol = make_test_fn(
            "neg_vol_self",
            1,
            vec![LocalSlot { id: 0, name: "b".into(), ty: ref_ball.clone(), mutable: false }],
            u8t.clone(),
            vec![
                BasicBlock {
                    id: 0,
                    insts: vec![
                        IrInst::LoadLocal { dst: 0, slot: 0, ty: ref_ball.clone() },
                        IrInst::FieldAddr { dst: 1, base: 0, byte_offset: 9 },
                        IrInst::LoadPtr { dst: 2, addr: 1, ty: u8t.clone(), volatile: true },
                        IrInst::FieldAddr { dst: 3, base: 0, byte_offset: 12 },
                        IrInst::LoadPtr { dst: 4, addr: 3, ty: u8t.clone(), volatile: false },
                        IrInst::ConstInt { dst: 5, ty: u8t.clone(), val: 0 },
                        IrInst::BinOp {
                            dst: 6,
                            op: IrBinOp::Eq,
                            mode: ArithMode::Wrapping,
                            ty: u8t.clone(),
                            lhs: 4,
                            rhs: 5,
                            site_id: 0,
                        },
                    ],
                    term: Terminator::Branch { cond: 6, then_bb: 1, else_bb: 2 },
                },
                BasicBlock {
                    id: 1,
                    insts: vec![IrInst::ConstInt { dst: 7, ty: u8t.clone(), val: 0 }],
                    term: Terminator::Return(Some(7)),
                },
                BasicBlock {
                    id: 2,
                    insts: vec![],
                    term: Terminator::Return(Some(2)),
                },
            ],
            8,
        );
        sink_cold_guard_spills_with_context(&mut f_vol, &structs, &[], &[]);
        sink_cold_loads_with_context(&mut f_vol, &structs, &[], &[]);
        assert!(
            f_vol.blocks[0]
                .insts
                .iter()
                .any(|i| matches!(i, IrInst::LoadPtr { dst: 2, volatile: true, .. })),
            "volatile LoadPtr must never sink"
        );

        // 5. Negative: LoadPtr outside a loop must NOT sink into the loop header/body.
        let mut f_loop = make_test_fn(
            "neg_into_loop",
            1,
            vec![LocalSlot { id: 0, name: "b".into(), ty: ref_ball.clone(), mutable: false }],
            u8t.clone(),
            vec![
                BasicBlock {
                    id: 0,
                    insts: vec![
                        IrInst::LoadLocal { dst: 0, slot: 0, ty: ref_ball.clone() },
                        IrInst::FieldAddr { dst: 1, base: 0, byte_offset: 9 },
                        IrInst::LoadPtr { dst: 2, addr: 1, ty: u8t.clone(), volatile: false },
                    ],
                    term: Terminator::Goto(1),
                },
                BasicBlock {
                    id: 1,
                    insts: vec![
                        IrInst::ConstInt { dst: 3, ty: u8t.clone(), val: 0 },
                        IrInst::BinOp {
                            dst: 4,
                            op: IrBinOp::Eq,
                            mode: ArithMode::Wrapping,
                            ty: u8t.clone(),
                            lhs: 2,
                            rhs: 3,
                            site_id: 0,
                        },
                    ],
                    term: Terminator::Branch { cond: 4, then_bb: 1, else_bb: 2 },
                },
                BasicBlock {
                    id: 2,
                    insts: vec![],
                    term: Terminator::Return(Some(2)),
                },
            ],
            5,
        );
        assert!(!sink_cold_loads_with_context(&mut f_loop, &structs, &[], &[]));
        assert!(
            f_loop.blocks[0]
                .insts
                .iter()
                .any(|i| matches!(i, IrInst::LoadPtr { dst: 2, .. })),
            "loop-invariant LoadPtr in preheader must NOT sink into loop header"
        );
    }

    #[test]
    fn issue51_schedule_loads_near_use_reorders_independent_and_respects_aliasing() {
        let u8t = IrType::Int(IntType::U8);
        let q10_6 = IrType::Fixed {
            base: IntType::I16,
            frac_bits: 6,
        };
        let ref_ball = IrType::Ref {
            mutable: true,
            inner: Box::new(IrType::Struct(0)),
        };
        let ptr_u8 = IrType::Ref {
            mutable: true,
            inner: Box::new(u8t.clone()),
        };
        let structs = vec![make_ball_struct()];

        // 1. Positive: `b.x` (+1, 2B) and `b.y` (+3, 2B) loaded back-to-back, then `b.x` is
        //    converted & written to `b.x_px` (+9, 1B) before `b.y` is converted & written to `+10`.
        //    `LoadPtr(b.y)` must move after `StorePtr(b.x_px)`.
        let mut f_pos = make_test_fn(
            "sync_px",
            1,
            vec![LocalSlot { id: 0, name: "b".into(), ty: ref_ball.clone(), mutable: false }],
            IrType::Unit,
            vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::LoadLocal { dst: 0, slot: 0, ty: ref_ball.clone() },
                    IrInst::FieldAddr { dst: 1, base: 0, byte_offset: 1 },
                    IrInst::LoadPtr { dst: 2, addr: 1, ty: q10_6.clone(), volatile: false },
                    IrInst::FieldAddr { dst: 3, base: 0, byte_offset: 3 },
                    IrInst::LoadPtr { dst: 4, addr: 3, ty: q10_6.clone(), volatile: false },
                    IrInst::FixedInt {
                        dst: 5,
                        val: 2,
                        base: IntType::I16,
                        frac_bits: 6,
                    },
                    IrInst::Cast {
                        dst: 6,
                        val: 5,
                        from_ty: IrType::Int(IntType::I16),
                        to_ty: u8t.clone(),
                    },
                    IrInst::FieldAddr { dst: 7, base: 0, byte_offset: 9 },
                    IrInst::StorePtr { addr: 7, val: 6, ty: u8t.clone(), volatile: false },
                    IrInst::FixedInt {
                        dst: 8,
                        val: 4,
                        base: IntType::I16,
                        frac_bits: 6,
                    },
                    IrInst::Cast {
                        dst: 9,
                        val: 8,
                        from_ty: IrType::Int(IntType::I16),
                        to_ty: u8t.clone(),
                    },
                    IrInst::FieldAddr { dst: 10, base: 0, byte_offset: 10 },
                    IrInst::StorePtr { addr: 10, val: 9, ty: u8t.clone(), volatile: false },
                ],
                term: Terminator::Return(None),
            }],
            11,
        );
        assert!(schedule_loads_near_use_with_context(&mut f_pos, &structs, &[], &[]));
        let pos_store_xpx = f_pos.blocks[0]
            .insts
            .iter()
            .position(|i| matches!(i, IrInst::StorePtr { addr: 7, .. }))
            .unwrap();
        let pos_load_y = f_pos.blocks[0]
            .insts
            .iter()
            .position(|i| matches!(i, IrInst::LoadPtr { dst: 4, .. }))
            .unwrap();
        assert!(
            pos_load_y > pos_store_xpx,
            "independent LoadPtr(b.y) must be scheduled after disjoint StorePtr(b.x_px):\n{:?}",
            f_pos.blocks[0].insts
        );

        // 2. Negative: StorePtr to an unknown pointer blocks LoadPtr(b.y) from moving past it.
        let mut f_neg = make_test_fn(
            "neg_alias_sched",
            2,
            vec![
                LocalSlot { id: 0, name: "b".into(), ty: ref_ball.clone(), mutable: false },
                LocalSlot { id: 1, name: "unk".into(), ty: ptr_u8.clone(), mutable: false },
            ],
            u8t.clone(),
            vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::LoadLocal { dst: 0, slot: 0, ty: ref_ball.clone() },
                    IrInst::FieldAddr { dst: 1, base: 0, byte_offset: 9 },
                    IrInst::LoadPtr { dst: 2, addr: 1, ty: u8t.clone(), volatile: false },
                    IrInst::LoadLocal { dst: 3, slot: 1, ty: ptr_u8.clone() },
                    IrInst::ConstInt { dst: 4, ty: u8t.clone(), val: 99 },
                    IrInst::StorePtr { addr: 3, val: 4, ty: u8t.clone(), volatile: false },
                ],
                term: Terminator::Return(Some(2)),
            }],
            5,
        );
        schedule_loads_near_use_with_context(&mut f_neg, &structs, &[], &[]);
        let pos_load = f_neg.blocks[0]
            .insts
            .iter()
            .position(|i| matches!(i, IrInst::LoadPtr { dst: 2, .. }))
            .unwrap();
        let pos_store = f_neg.blocks[0]
            .insts
            .iter()
            .position(|i| matches!(i, IrInst::StorePtr { addr: 3, .. }))
            .unwrap();
        assert!(
            pos_load < pos_store,
            "LoadPtr MUST NOT move past potentially aliasing StorePtr:\n{:?}",
            f_neg.blocks[0].insts
        );
    }
}
