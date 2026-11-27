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

/// Optimizes all functions in `program` in-place until fixed point.
pub fn optimize_program(program: &mut IrProgram) {
    for func in &mut program.functions {
        optimize_function(func);
    }
}

/// Optimizes a single `IrFunction` in-place until fixed point.
pub fn optimize_function(func: &mut IrFunction) {
    if func.blocks.is_empty() || func.blocks.iter().all(|b| b.insts.is_empty()) {
        return;
    }
    for _ in 0..8 {
        let mut changed = false;
        changed |= fold_constants(func);
        changed |= simplify_cfg(func);
        changed |= forward_and_coalesce_locals(func);
        changed |= dce(func);
        if !changed {
            break;
        }
    }
}

/// Promotes promotable locals to Pruned SSA, runs SCCP + GVN + LICM + OSR + DCE to
/// fixed point, and destructs SSA form back to non-`Phi` MIR.
pub fn optimize_function_ssa(func: &mut IrFunction) {
    if func.blocks.is_empty() || func.blocks.iter().all(|b| b.insts.is_empty()) {
        return;
    }
    crate::ssa::mem2reg(func);
    for _ in 0..4 {
        let mut changed = false;
        changed |= crate::sccp::sccp(func);
        changed |= crate::gvn::gvn(func);
        changed |= crate::licm::licm(func);
        changed |= crate::osr::osr(func);
        changed |= dce(func);
        if !changed {
            break;
        }
    }
    crate::ssa::destruct_ssa(func);
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

fn fold_constants(func: &mut IrFunction) -> bool {
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
    for bb in &mut func.blocks {
        for inst in &mut bb.insts {
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
                IrInst::Cast {
                    dst,
                    to_ty: IrType::Int(to_it),
                    val,
                    ..
                } => const_ints.get(val).map(|&(_, v)| IrInst::ConstInt {
                    dst: *dst,
                    ty: IrType::Int(*to_it),
                    val: truncate_int(v, *to_it),
                }),
                IrInst::BinOp {
                    dst,
                    op,
                    mode,
                    ty: IrType::Int(it),
                    lhs,
                    rhs,
                    ..
                } => {
                    if let (Some(&(_, lv)), Some(&(_, rv))) =
                        (const_ints.get(lhs), const_ints.get(rhs))
                    {
                        match op {
                            IrBinOp::Eq => Some(IrInst::ConstBool {
                                dst: *dst,
                                val: lv == rv,
                            }),
                            IrBinOp::Ne => Some(IrInst::ConstBool {
                                dst: *dst,
                                val: lv != rv,
                            }),
                            IrBinOp::Lt => Some(IrInst::ConstBool {
                                dst: *dst,
                                val: lv < rv,
                            }),
                            IrBinOp::Le => Some(IrInst::ConstBool {
                                dst: *dst,
                                val: lv <= rv,
                            }),
                            IrBinOp::Gt => Some(IrInst::ConstBool {
                                dst: *dst,
                                val: lv > rv,
                            }),
                            IrBinOp::Ge => Some(IrInst::ConstBool {
                                dst: *dst,
                                val: lv >= rv,
                            }),
                            IrBinOp::BitAnd => Some(IrInst::ConstInt {
                                dst: *dst,
                                ty: IrType::Int(*it),
                                val: truncate_int(lv & rv, *it),
                            }),
                            IrBinOp::BitOr => Some(IrInst::ConstInt {
                                dst: *dst,
                                ty: IrType::Int(*it),
                                val: truncate_int(lv | rv, *it),
                            }),
                            IrBinOp::BitXor => Some(IrInst::ConstInt {
                                dst: *dst,
                                ty: IrType::Int(*it),
                                val: truncate_int(lv ^ rv, *it),
                            }),
                            IrBinOp::Add if *mode == ArithMode::Wrapping => Some(IrInst::ConstInt {
                                dst: *dst,
                                ty: IrType::Int(*it),
                                val: truncate_int(lv.wrapping_add(rv), *it),
                            }),
                            IrBinOp::Sub if *mode == ArithMode::Wrapping => Some(IrInst::ConstInt {
                                dst: *dst,
                                ty: IrType::Int(*it),
                                val: truncate_int(lv.wrapping_sub(rv), *it),
                            }),
                            _ => None,
                        }
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
            } else if let IrInst::BinOp { op, lhs, rhs, .. } = inst
                && const_ints.contains_key(lhs)
                && !const_ints.contains_key(rhs)
            {
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
        }
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

    // 3. Dead `StoreLocal` elimination for unread non-escaping non-param locals, plus
    //    dead `StoreLocal` before `Return` in the same basic block.
    let mut ever_read = vec![false; n_slots];
    for bb in &func.blocks {
        for inst in &bb.insts {
            if let IrInst::LoadLocal { slot, .. } = inst
                && let Some(f) = ever_read.get_mut(*slot as usize)
            {
                *f = true;
            }
        }
    }
    for bb in &mut func.blocks {
        let ends_in_ret = matches!(bb.term, Terminator::Return(_));
        let mut drop = vec![false; bb.insts.len()];
        for (ii, inst) in bb.insts.iter().enumerate() {
            if let IrInst::StoreLocal { slot, .. } = inst {
                let s = *slot as usize;
                if s >= func.param_count && s < n_slots && !addr_taken[s] {
                    let unread_anywhere = !ever_read[s];
                    let dead_before_ret = ends_in_ret
                        && !bb.insts[ii + 1..].iter().any(
                            |later| matches!(later, IrInst::LoadLocal { slot: ls, .. } if ls == slot),
                        );
                    if unread_anywhere || dead_before_ret {
                        drop[ii] = true;
                        changed = true;
                    }
                }
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

fn is_pure_inst(inst: &IrInst) -> bool {
    match inst {
        IrInst::ConstInt { .. }
        | IrInst::ConstBool { .. }
        | IrInst::LoadLocal { .. }
        | IrInst::AddrLocal { .. }
        | IrInst::AddrGlobal { .. }
        | IrInst::AddrFn { .. }
        | IrInst::FieldAddr { .. }
        | IrInst::LoadPtr { .. }
        | IrInst::Cast { .. }
        | IrInst::FixedInt { .. }
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
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
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
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
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
}
