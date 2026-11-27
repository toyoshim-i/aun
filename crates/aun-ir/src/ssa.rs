//! Pruned SSA construction (`mem2reg`), trivial Phi cleanup, and SSA destruction (`destruct_ssa`).

use crate::cfg::Cfg;
use crate::opt::apply_val_remap;
use crate::{
    BasicBlock, BlockId, IrBinOp, IrFunction, IrInst, IrType, LocalSlot, SlotId, Terminator,
    ValueId,
};
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

fn is_promotable_scalar(ty: &IrType) -> bool {
    matches!(
        ty,
        IrType::Bool
            | IrType::Int(_)
            | IrType::Fixed { .. }
            | IrType::Handle(_)
            | IrType::Ref { .. }
    )
}

fn is_scalar_1_or_2_byte(ty: &IrType) -> bool {
    match ty {
        IrType::Bool | IrType::Handle(_) | IrType::Ref { .. } => true,
        IrType::Int(it) | IrType::Fixed { base: it, .. } => it.byte_size() <= 2,
        _ => false,
    }
}

fn scalar_def_ty(inst: &IrInst) -> Option<IrType> {
    let ty = match inst {
        IrInst::BinOp { op, ty, .. } => {
            if matches!(
                op,
                IrBinOp::Eq | IrBinOp::Ne | IrBinOp::Lt | IrBinOp::Le | IrBinOp::Gt | IrBinOp::Ge
            ) {
                IrType::Bool
            } else {
                ty.clone()
            }
        }
        IrInst::UnOp { ty, .. } => ty.clone(),
        IrInst::Cast { to_ty, .. } => to_ty.clone(),
        IrInst::LoadPtr { ty, .. } => ty.clone(),
        IrInst::FixedInt { base, .. } => IrType::Int(*base),
        IrInst::StateTag { .. } | IrInst::PortIn { .. } | IrInst::PoolAlloc { .. } => {
            IrType::Int(crate::IntType::U8)
        }
        _ => return None,
    };
    is_scalar_1_or_2_byte(&ty).then_some(ty)
}

fn alloc_or_reuse_slot(
    func: &mut IrFunction,
    used_slots: &mut Vec<bool>,
    ty: &IrType,
    fallback_name: String,
) -> u32 {
    if let Some(pos) = (func.param_count..func.locals.len())
        .find(|&s| !used_slots[s] && func.locals[s].ty == *ty)
    {
        used_slots[pos] = true;
        pos as u32
    } else {
        let slot_id = func.locals.len() as u32;
        func.locals.push(LocalSlot {
            id: slot_id,
            name: fallback_name,
            ty: ty.clone(),
            mutable: true,
        });
        used_slots.push(true);
        slot_id
    }
}

/// Promotes non-escaped scalar local slots into SSA values with `IrInst::Phi` nodes
/// (Cytron et al. iterated dominance frontier + Choi-Cytron-Ferrante liveness pruning).
///
/// Any non-parameter slot that is live-in to entry block `bb0` (e.g., potentially
/// uninitialized on some path) is left as a stack/RAM local slot so SSA dominance and
/// single-definition guarantees always hold cleanly.
pub fn mem2reg(func: &mut IrFunction) -> bool {
    if func.blocks.is_empty() || func.locals.is_empty() {
        return false;
    }

    let cfg = Cfg::build(func);
    let rpo = cfg.rpo();
    if rpo.is_empty() {
        return false;
    }
    let entry_id = cfg.entry_block().unwrap_or(func.blocks[0].id);
    let bb_map: HashMap<BlockId, usize> =
        func.blocks.iter().enumerate().map(|(i, b)| (b.id, i)).collect();

    let n_slots = func.locals.len();
    let escaped = func.escaped_locals();
    let mut addr_taken = vec![false; n_slots];
    let mut has_access = vec![false; n_slots];
    let mut has_store = vec![false; n_slots];
    let mut read_outside_entry = vec![false; n_slots];
    for &bb_id in rpo {
        let bb = &func.blocks[bb_map[&bb_id]];
        for inst in &bb.insts {
            match inst {
                IrInst::AddrLocal { slot, .. } => {
                    if let Some(f) = addr_taken.get_mut(*slot as usize) {
                        *f = true;
                    }
                }
                IrInst::LoadLocal { slot, .. } => {
                    let s = *slot as usize;
                    if let Some(f) = has_access.get_mut(s) {
                        *f = true;
                    }
                    if bb_id != entry_id
                        && let Some(f) = read_outside_entry.get_mut(s)
                    {
                        *f = true;
                    }
                }
                IrInst::StoreLocal { slot, .. } => {
                    let s = *slot as usize;
                    if let Some(f) = has_access.get_mut(s) {
                        *f = true;
                    }
                    if let Some(f) = has_store.get_mut(s) {
                        *f = true;
                    }
                }
                _ => {}
            }
        }
    }
    // Check unreachable blocks as well so AddrLocal anywhere conservatively blocks promotion.
    for bb in &func.blocks {
        for inst in &bb.insts {
            if let IrInst::AddrLocal { slot, .. } = inst
                && let Some(f) = addr_taken.get_mut(*slot as usize)
            {
                *f = true;
            }
        }
    }

    let entry_has_preds = !cfg.reachable_preds(entry_id).is_empty();
    let mut candidate = vec![false; n_slots];
    let mut any_cand = false;
    for (s, local) in func.locals.iter().enumerate() {
        if escaped[s] || addr_taken[s] || !has_access[s] || !is_promotable_scalar(&local.ty) {
            continue;
        }
        if s < func.param_count
            && (entry_has_preds
                || matches!(local.ty, IrType::Ref { .. })
                || (!has_store[s] && !read_outside_entry[s]))
        {
            continue;
        }
        candidate[s] = true;
        any_cand = true;
    }
    if !any_cand {
        return false;
    }

    // Backward local-slot liveness on reachable blocks for Pruned SSA.
    let n_rpo = rpo.len();
    let rpo_pos: HashMap<BlockId, usize> = rpo.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let mut use_before_def: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n_rpo];
    let mut def_in_bb: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n_rpo];
    let mut def_blocks: HashMap<usize, BTreeSet<BlockId>> = HashMap::new();

    for (pos, &bb_id) in rpo.iter().enumerate() {
        let bb = &func.blocks[bb_map[&bb_id]];
        if bb_id == entry_id {
            for (s, &is_cand) in candidate.iter().enumerate() {
                if is_cand && s < func.param_count {
                    def_in_bb[pos].insert(s);
                    def_blocks.entry(s).or_default().insert(bb_id);
                }
            }
        }
        for inst in &bb.insts {
            match inst {
                IrInst::LoadLocal { slot, .. } => {
                    let s = *slot as usize;
                    if s < n_slots && candidate[s] && !def_in_bb[pos].contains(&s) {
                        use_before_def[pos].insert(s);
                    }
                }
                IrInst::StoreLocal { slot, .. } => {
                    let s = *slot as usize;
                    if s < n_slots && candidate[s] {
                        def_in_bb[pos].insert(s);
                        def_blocks.entry(s).or_default().insert(bb_id);
                    }
                }
                _ => {}
            }
        }
    }

    let mut live_in = use_before_def.clone();
    let mut live_out: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n_rpo];
    let mut liveness_changed = true;
    while liveness_changed {
        liveness_changed = false;
        for pos in (0..n_rpo).rev() {
            let bb_id = rpo[pos];
            let mut new_out = BTreeSet::new();
            for &succ in cfg.succs(bb_id) {
                if let Some(&s_pos) = rpo_pos.get(&succ) {
                    new_out.extend(&live_in[s_pos]);
                }
            }
            let mut new_in = use_before_def[pos].clone();
            for &s in &new_out {
                if !def_in_bb[pos].contains(&s) {
                    new_in.insert(s);
                }
            }
            if new_out != live_out[pos] || new_in != live_in[pos] {
                live_out[pos] = new_out;
                live_in[pos] = new_in;
                liveness_changed = true;
            }
        }
    }

    // Exclude any non-parameter slot that is live-in at entry block `bb0` so every
    // reachable read and Phi incoming edge is strictly dominated by a definition.
    let entry_pos = rpo_pos[&entry_id];
    let mut promoted = BTreeSet::new();
    for (s, &is_cand) in candidate.iter().enumerate() {
        if !is_cand {
            continue;
        }
        if s >= func.param_count && live_in[entry_pos].contains(&s) {
            continue;
        }
        promoted.insert(s as u32);
    }
    if promoted.is_empty() {
        return false;
    }

    // Iterated Dominance Frontier (DF+) Phi placement (Pruned SSA).
    let mut pending_phis: HashMap<BlockId, Vec<(u32, ValueId, IrType)>> = HashMap::new();
    let mut phi_slot_map: HashMap<(BlockId, ValueId), u32> = HashMap::new();

    for &slot in &promoted {
        let s_idx = slot as usize;
        let defs = match def_blocks.get(&s_idx) {
            Some(d) => d,
            None => continue,
        };
        let mut worklist: Vec<BlockId> = defs.iter().copied().collect();
        let mut ever_in_wl: HashSet<BlockId> = worklist.iter().copied().collect();
        let mut has_phi: HashSet<BlockId> = HashSet::new();

        while let Some(x) = worklist.pop() {
            for &y in cfg.df(x) {
                let y_pos = match rpo_pos.get(&y) {
                    Some(&p) => p,
                    None => continue,
                };
                if !live_in[y_pos].contains(&s_idx) || !has_phi.insert(y) {
                    continue;
                }
                let phi_dst = func.next_val;
                func.next_val += 1;
                let ty = func.locals[s_idx].ty.clone();
                pending_phis.entry(y).or_default().push((slot, phi_dst, ty));
                phi_slot_map.insert((y, phi_dst), slot);
                if ever_in_wl.insert(y) {
                    worklist.push(y);
                }
            }
        }
    }

    for (bb_id, mut phis) in pending_phis {
        phis.sort_by_key(|(slot, _, _)| *slot);
        let bb = &mut func.blocks[bb_map[&bb_id]];
        let new_insts = phis.into_iter().map(|(_, dst, ty)| IrInst::Phi {
            dst,
            ty,
            incoming: Vec::new(),
        });
        bb.insts.splice(0..0, new_insts);
    }

    // Materialize one entry LoadLocal at bb0 for each promoted parameter slot.
    let mut stacks: HashMap<u32, Vec<ValueId>> = HashMap::new();
    let mut materialized_param_dsts: HashSet<ValueId> = HashSet::new();
    let entry_bb_idx = bb_map[&entry_id];
    let entry_phi_end = func.blocks[entry_bb_idx]
        .insts
        .iter()
        .take_while(|i| matches!(i, IrInst::Phi { .. }))
        .count();
    let mut param_loads = Vec::new();
    for &slot in &promoted {
        if (slot as usize) < func.param_count {
            let v_param = func.next_val;
            func.next_val += 1;
            let ty = func.locals[slot as usize].ty.clone();
            param_loads.push(IrInst::LoadLocal {
                dst: v_param,
                slot,
                ty,
            });
            materialized_param_dsts.insert(v_param);
            stacks.entry(slot).or_default().push(v_param);
        }
    }
    if !param_loads.is_empty() {
        func.blocks[entry_bb_idx]
            .insts
            .splice(entry_phi_end..entry_phi_end, param_loads);
    }

    // Dominator-tree DFS renaming.
    let mut remap: HashMap<ValueId, ValueId> = HashMap::new();
    let mut drop_insts: HashMap<BlockId, Vec<bool>> = HashMap::new();
    let mut phi_incoming_adds: HashMap<(BlockId, ValueId), Vec<(BlockId, ValueId)>> =
        HashMap::new();

    struct RenameCtx<'a> {
        func: &'a IrFunction,
        cfg: &'a Cfg,
        bb_map: &'a HashMap<BlockId, usize>,
        promoted: &'a BTreeSet<u32>,
        phi_slot_map: &'a HashMap<(BlockId, ValueId), u32>,
        materialized_param_dsts: &'a HashSet<ValueId>,
        stacks: &'a mut HashMap<u32, Vec<ValueId>>,
        remap: &'a mut HashMap<ValueId, ValueId>,
        drop_insts: &'a mut HashMap<BlockId, Vec<bool>>,
        phi_incoming_adds: &'a mut HashMap<(BlockId, ValueId), Vec<(BlockId, ValueId)>>,
    }

    fn resolve_val(mut v: ValueId, remap: &HashMap<ValueId, ValueId>) -> ValueId {
        let mut hops = 0;
        while let Some(&next) = remap.get(&v) {
            if next == v || hops > 32 {
                break;
            }
            v = next;
            hops += 1;
        }
        v
    }

    fn rename_bb(ctx: &mut RenameCtx<'_>, bb_id: BlockId) {
        let bb = &ctx.func.blocks[ctx.bb_map[&bb_id]];
        let mut drops = vec![false; bb.insts.len()];
        let mut pushed: Vec<u32> = Vec::new();

        for (idx, inst) in bb.insts.iter().enumerate() {
            match inst {
                IrInst::Phi { dst, .. } => {
                    if let Some(&slot) = ctx.phi_slot_map.get(&(bb_id, *dst)) {
                        ctx.stacks.entry(slot).or_default().push(*dst);
                        pushed.push(slot);
                    }
                }
                IrInst::LoadLocal { dst, slot, .. } => {
                    if !ctx.materialized_param_dsts.contains(dst)
                        && ctx.promoted.contains(slot)
                    {
                        let cur = *ctx
                            .stacks
                            .get(slot)
                            .and_then(|s| s.last())
                            .expect("pruned SSA guarantees active definition before load");
                        ctx.remap.insert(*dst, cur);
                        drops[idx] = true;
                    }
                }
                IrInst::StoreLocal { slot, val, .. } => {
                    if ctx.promoted.contains(slot) {
                        let resolved = resolve_val(*val, ctx.remap);
                        ctx.stacks.entry(*slot).or_default().push(resolved);
                        pushed.push(*slot);
                        drops[idx] = true;
                    }
                }
                _ => {}
            }
        }
        ctx.drop_insts.insert(bb_id, drops);

        for &succ_id in ctx.cfg.succs(bb_id) {
            let succ_bb = &ctx.func.blocks[ctx.bb_map[&succ_id]];
            for inst in &succ_bb.insts {
                match inst {
                    IrInst::Phi { dst, .. } => {
                        if let Some(&slot) = ctx.phi_slot_map.get(&(succ_id, *dst)) {
                            let cur = *ctx
                                .stacks
                                .get(&slot)
                                .and_then(|s| s.last())
                                .expect("pruned SSA guarantees definition dominates predecessor");
                            ctx.phi_incoming_adds
                                .entry((succ_id, *dst))
                                .or_default()
                                .push((bb_id, cur));
                        }
                    }
                    _ => break,
                }
            }
        }

        for &child in ctx.cfg.dom_children(bb_id) {
            rename_bb(ctx, child);
        }

        for slot in pushed.into_iter().rev() {
            ctx.stacks.get_mut(&slot).unwrap().pop();
        }
    }

    let mut ctx = RenameCtx {
        func,
        cfg: &cfg,
        bb_map: &bb_map,
        promoted: &promoted,
        phi_slot_map: &phi_slot_map,
        materialized_param_dsts: &materialized_param_dsts,
        stacks: &mut stacks,
        remap: &mut remap,
        drop_insts: &mut drop_insts,
        phi_incoming_adds: &mut phi_incoming_adds,
    };
    rename_bb(&mut ctx, entry_id);

    // Apply Phi incoming lists, drop promoted LoadLocal/StoreLocal instructions, and remap uses.
    for bb in &mut func.blocks {
        let bb_id = bb.id;
        for inst in &mut bb.insts {
            if let IrInst::Phi { dst, incoming, .. } = inst
                && let Some(adds) = phi_incoming_adds.get(&(bb_id, *dst))
            {
                incoming.extend(adds.iter().copied());
                incoming.sort_by_key(|&(pred, _)| pred);
            }
        }
        if let Some(drops) = drop_insts.get(&bb_id) {
            let mut idx = 0;
            bb.insts.retain(|_| {
                let keep = !drops[idx];
                idx += 1;
                keep
            });
        }
    }

    apply_val_remap(func, &remap);
    simplify_trivial_phis(func);
    true
}

/// Eliminates trivial `IrInst::Phi` nodes (where all non-self incoming values are identical)
/// to a fixed point, replacing their uses with the unique value.
pub fn simplify_trivial_phis(func: &mut IrFunction) -> bool {
    let mut any_changed = false;
    loop {
        let mut remap: HashMap<ValueId, ValueId> = HashMap::new();
        let mut drop_blocks: Vec<Vec<bool>> = Vec::with_capacity(func.blocks.len());

        for bb in &func.blocks {
            let mut drops = vec![false; bb.insts.len()];
            for (idx, inst) in bb.insts.iter().enumerate() {
                let IrInst::Phi { dst, incoming, .. } = inst else {
                    break;
                };
                let mut unique: Option<ValueId> = None;
                let mut trivial = true;
                for &(_, val) in incoming {
                    if val == *dst {
                        continue;
                    }
                    match unique {
                        None => unique = Some(val),
                        Some(u) if u == val => {}
                        Some(_) => {
                            trivial = false;
                            break;
                        }
                    }
                }
                if trivial && let Some(rep) = unique {
                    remap.insert(*dst, rep);
                    drops[idx] = true;
                }
            }
            drop_blocks.push(drops);
        }

        if remap.is_empty() {
            break;
        }
        any_changed = true;
        for (bb, drops) in func.blocks.iter_mut().zip(drop_blocks) {
            let mut idx = 0;
            bb.insts.retain(|_| {
                let keep = !drops[idx];
                idx += 1;
                keep
            });
        }
        apply_val_remap(func, &remap);
    }
    any_changed
}

/// Destructs SSA form by simplifying trivial Phis, splitting critical edges entering
/// Phi blocks, and lowering remaining `IrInst::Phi` nodes to per-Phi scratch `LocalSlot`s.
pub fn destruct_ssa(func: &mut IrFunction) -> bool {
    let mut changed = simplify_trivial_phis(func);

    let has_any_phi = func
        .blocks
        .iter()
        .any(|bb| bb.insts.iter().any(|i| matches!(i, IrInst::Phi { .. })));
    if !has_any_phi {
        if rematerialize_local_loads(func) {
            changed = true;
        }
        if demote_cross_block_ssa_to_locals(func) {
            changed = true;
        }
        return changed;
    }
    changed = true;

    // 1. Split critical edges: Branch predecessors (with distinct targets) jumping to a Phi block.
    let mut next_bb_id = func
        .blocks
        .iter()
        .map(|b| b.id)
        .max()
        .map_or(0, |m| m + 1);
    let orig_len = func.blocks.len();
    let phi_target_ids: HashSet<BlockId> = func
        .blocks
        .iter()
        .filter(|bb| matches!(bb.insts.first(), Some(IrInst::Phi { .. })))
        .map(|bb| bb.id)
        .collect();
    let bb_map: HashMap<BlockId, usize> =
        func.blocks.iter().enumerate().map(|(i, b)| (b.id, i)).collect();

    let mut trampolines: Vec<BasicBlock> = Vec::new();
    for p_idx in 0..orig_len {
        let p_id = func.blocks[p_idx].id;
        let Terminator::Branch {
            cond,
            mut then_bb,
            mut else_bb,
        } = func.blocks[p_idx].term
        else {
            continue;
        };
        if then_bb == else_bb {
            continue;
        }

        let mut updated = false;
        if phi_target_ids.contains(&then_bb) {
            let target = then_bb;
            let split_id = next_bb_id;
            next_bb_id += 1;
            then_bb = split_id;
            trampolines.push(BasicBlock {
                id: split_id,
                insts: Vec::new(),
                term: Terminator::Goto(target),
            });
            rewrite_phi_pred(&mut func.blocks[bb_map[&target]], p_id, split_id);
            updated = true;
        }
        if phi_target_ids.contains(&else_bb) {
            let target = else_bb;
            let split_id = next_bb_id;
            next_bb_id += 1;
            else_bb = split_id;
            trampolines.push(BasicBlock {
                id: split_id,
                insts: Vec::new(),
                term: Terminator::Goto(target),
            });
            rewrite_phi_pred(&mut func.blocks[bb_map[&target]], p_id, split_id);
            updated = true;
        }
        if updated {
            func.blocks[p_idx].term = Terminator::Branch {
                cond,
                then_bb,
                else_bb,
            };
        }
    }
    func.blocks.extend(trampolines);

    // 2. Lower every Phi into a reused unused non-param LocalSlot or a fresh `__phi_*` slot.
    let mut used_slots = vec![false; func.locals.len()];
    for bb in &func.blocks {
        for inst in &bb.insts {
            if let IrInst::LoadLocal { slot, .. }
            | IrInst::StoreLocal { slot, .. }
            | IrInst::AddrLocal { slot, .. } = inst
                && let Some(u) = used_slots.get_mut(*slot as usize)
            {
                *u = true;
            }
        }
    }
    let bb_map: HashMap<BlockId, usize> =
        func.blocks.iter().enumerate().map(|(i, b)| (b.id, i)).collect();
    let phi_block_indices: Vec<usize> = func
        .blocks
        .iter()
        .enumerate()
        .filter_map(|(idx, bb)| matches!(bb.insts.first(), Some(IrInst::Phi { .. })).then_some(idx))
        .collect();

    for b_idx in phi_block_indices {
        let phi_count = func.blocks[b_idx]
            .insts
            .iter()
            .take_while(|i| matches!(i, IrInst::Phi { .. }))
            .count();
        let drained_phis: Vec<IrInst> = func.blocks[b_idx].insts.drain(0..phi_count).collect();
        let mut loads = Vec::with_capacity(drained_phis.len());

        for phi in drained_phis {
            let IrInst::Phi { dst, ty, incoming } = phi else {
                unreachable!();
            };
            let slot_id = alloc_or_reuse_slot(
                func,
                &mut used_slots,
                &ty,
                format!("__phi_{dst}"),
            );
            for (pred_bb, val) in incoming {
                if let Some(&p_idx) = bb_map.get(&pred_bb) {
                    func.blocks[p_idx].insts.push(IrInst::StoreLocal {
                        slot: slot_id,
                        val,
                        ty: ty.clone(),
                    });
                }
            }
            loads.push(IrInst::LoadLocal {
                dst,
                slot: slot_id,
                ty,
            });
        }
        func.blocks[b_idx].insts.splice(0..0, loads);
    }

    if rematerialize_local_loads(func) {
        changed = true;
    }
    if demote_cross_block_ssa_to_locals(func) {
        changed = true;
    }

    changed
}

/// Rematerializes non-address-taken `LoadLocal(slot)` values into each downstream using block
/// whenever no `StoreLocal(slot)` can occur between the definition and the block's last use.
///
/// This keeps local reads intra-block so Z80 codegen can alias registers (`B`/`C`) and RAM slots
/// without allocating cross-block dedicated scratch spills.
pub fn rematerialize_local_loads(func: &mut IrFunction) -> bool {
    if func.blocks.is_empty() || func.locals.is_empty() {
        return false;
    }
    let n_slots = func.locals.len();
    let cfg = Cfg::build(func);
    let mut id_to_idx: HashMap<BlockId, usize> = HashMap::with_capacity(func.blocks.len());
    let mut addr_taken = vec![false; n_slots];
    let mut stores_in_bb: Vec<HashSet<SlotId>> = vec![HashSet::new(); func.blocks.len()];
    for (bi, bb) in func.blocks.iter().enumerate() {
        id_to_idx.insert(bb.id, bi);
        for inst in &bb.insts {
            match inst {
                IrInst::AddrLocal { slot, .. } => {
                    if let Some(f) = addr_taken.get_mut(*slot as usize) {
                        *f = true;
                    }
                }
                IrInst::StoreLocal { slot, .. } => {
                    stores_in_bb[bi].insert(*slot);
                }
                _ => {}
            }
        }
    }

    // (ui_idx, old_val, slot, ty)
    let mut remat_ops: Vec<(usize, ValueId, SlotId, IrType)> = Vec::new();
    for bi in 0..func.blocks.len() {
        let bi_id = func.blocks[bi].id;
        if !cfg.is_reachable(bi_id) {
            continue;
        }
        for (def_ii, inst) in func.blocks[bi].insts.iter().enumerate() {
            let IrInst::LoadLocal {
                dst: v,
                slot,
                ref ty,
            } = *inst
            else {
                continue;
            };
            if (slot as usize) >= n_slots || addr_taken[slot as usize] {
                continue;
            }
            if func.blocks[bi].insts[def_ii + 1..]
                .iter()
                .any(|i| matches!(i, IrInst::StoreLocal { slot: s, .. } if *s == slot))
            {
                continue;
            }

            // Compute blocks reachable from `succs(bi)` without re-entering `bi`.
            let mut fwd: HashSet<usize> = HashSet::new();
            let mut fq: VecDeque<usize> = VecDeque::new();
            for &succ_id in cfg.succs(bi_id) {
                if let Some(&si) = id_to_idx.get(&succ_id)
                    && fwd.insert(si)
                {
                    fq.push_back(si);
                }
            }
            while let Some(cur) = fq.pop_front() {
                if cur == bi {
                    continue;
                }
                for &succ_id in cfg.succs(func.blocks[cur].id) {
                    if let Some(&si) = id_to_idx.get(&succ_id)
                        && fwd.insert(si)
                    {
                        fq.push_back(si);
                    }
                }
            }

            for ui in 0..func.blocks.len() {
                if ui == bi {
                    continue;
                }
                let ui_id = func.blocks[ui].id;
                if !cfg.is_reachable(ui_id) || !fwd.contains(&ui) {
                    continue;
                }
                let used_in_term = matches!(
                    func.blocks[ui].term,
                    Terminator::Return(Some(tv)) | Terminator::Branch { cond: tv, .. } if tv == v
                );
                let last_inst_use = func.blocks[ui]
                    .insts
                    .iter()
                    .rposition(|i| i.uses_val(v));
                let Some(last_use_pos) = (if used_in_term {
                    Some(func.blocks[ui].insts.len())
                } else {
                    last_inst_use
                }) else {
                    continue;
                };
                if func.blocks[ui].insts[..last_use_pos]
                    .iter()
                    .any(|i| matches!(i, IrInst::StoreLocal { slot: s, .. } if *s == slot))
                {
                    continue;
                }

                // Reverse BFS from `preds(ui)` stopping at `bi` to find all intermediate path blocks.
                let mut rev: HashSet<usize> = HashSet::new();
                let mut rq: VecDeque<usize> = VecDeque::new();
                for pred_id in cfg.reachable_preds(ui_id) {
                    if let Some(&pi) = id_to_idx.get(&pred_id)
                        && rev.insert(pi)
                    {
                        rq.push_back(pi);
                    }
                }
                while let Some(cur) = rq.pop_front() {
                    if cur == bi {
                        continue;
                    }
                    for pred_id in cfg.reachable_preds(func.blocks[cur].id) {
                        if let Some(&pi) = id_to_idx.get(&pred_id)
                            && rev.insert(pi)
                        {
                            rq.push_back(pi);
                        }
                    }
                }

                let conflict_mid = fwd.iter().any(|&m| {
                    m != bi && m != ui && rev.contains(&m) && stores_in_bb[m].contains(&slot)
                });
                let conflict_cycle = rev.contains(&ui) && stores_in_bb[ui].contains(&slot);
                if conflict_mid || conflict_cycle {
                    continue;
                }

                remat_ops.push((ui, v, slot, ty.clone()));
            }
        }
    }

    if remat_ops.is_empty() {
        return false;
    }

    let mut per_block: HashMap<usize, Vec<(ValueId, SlotId, IrType)>> = HashMap::new();
    for (ui, v, slot, ty) in remat_ops {
        per_block.entry(ui).or_default().push((v, slot, ty));
    }

    for (ui, ops) in per_block {
        insert_remat_loads_in_block(&mut func.blocks[ui], &mut func.next_val, ops);
    }

    crate::opt::dce(func);
    true
}

/// Inserts per-block rematerialized `LoadLocal` instructions immediately before their first
/// non-`Phi` use in `bb` (or at the end of `bb.insts` if used only by `bb.term`), rewriting all
/// uses of the remapped `ValueId`s in `bb`.
fn insert_remat_loads_in_block(
    bb: &mut BasicBlock,
    next_val: &mut ValueId,
    ops: Vec<(ValueId, SlotId, IrType)>,
) {
    let mut local_remap: HashMap<ValueId, ValueId> = HashMap::with_capacity(ops.len());
    let mut pending_loads: Vec<(ValueId, IrInst, bool)> = Vec::with_capacity(ops.len());
    for (old_v, slot, ty) in ops {
        if local_remap.contains_key(&old_v) {
            continue;
        }
        let new_v = *next_val;
        *next_val += 1;
        local_remap.insert(old_v, new_v);
        pending_loads.push((
            new_v,
            IrInst::LoadLocal {
                dst: new_v,
                slot,
                ty,
            },
            false,
        ));
    }

    for inst in &mut bb.insts {
        inst.for_each_use_mut(|u| {
            if let Some(&nv) = local_remap.get(u) {
                *u = nv;
            }
        });
    }
    match &mut bb.term {
        Terminator::Return(Some(tv)) | Terminator::Branch { cond: tv, .. } => {
            if let Some(&nv) = local_remap.get(tv) {
                *tv = nv;
            }
        }
        _ => {}
    }

    let mut rebuilt: Vec<IrInst> = Vec::with_capacity(bb.insts.len() + pending_loads.len());
    for inst in bb.insts.drain(..) {
        if !matches!(inst, IrInst::Phi { .. }) {
            for (nv, load_inst, emitted) in &mut pending_loads {
                if !*emitted && inst.uses_val(*nv) {
                    *emitted = true;
                    rebuilt.push(load_inst.clone());
                }
            }
        }
        rebuilt.push(inst);
    }
    for (_, load_inst, emitted) in pending_loads {
        if !emitted {
            rebuilt.push(load_inst);
        }
    }
    bb.insts = rebuilt;
}

/// Demotes cross-block 1B/2B SSA temporary values into `LocalSlot`s (with `StoreLocal` after the
/// defining instruction and `LoadLocal` at the top of each using block) so `regalloc` can promote
/// them to `B`/`C`/`BC` or color them into ordinary RAM locals instead of dedicated cross-block scratch.
fn demote_cross_block_ssa_to_locals(func: &mut IrFunction) -> bool {
    if func.blocks.is_empty() {
        return false;
    }
    let mut cross_uses: HashMap<ValueId, Vec<usize>> = HashMap::new();
    let mut def_info: Vec<(usize, usize, ValueId, IrType)> = Vec::new();
    let mut def_block_of: HashMap<ValueId, usize> = HashMap::new();

    for (bi, bb) in func.blocks.iter().enumerate() {
        for (ii, inst) in bb.insts.iter().enumerate() {
            if let Some(d) = inst.dst()
                && let Some(ty) = scalar_def_ty(inst)
            {
                def_block_of.insert(d, bi);
                def_info.push((bi, ii, d, ty));
            }
        }
    }
    if def_info.is_empty() {
        return false;
    }

    for (ui, bb) in func.blocks.iter().enumerate() {
        let mut record = |v: ValueId| {
            if let Some(&dbi) = def_block_of.get(&v)
                && dbi != ui
            {
                let list = cross_uses.entry(v).or_default();
                if list.last().copied() != Some(ui) {
                    list.push(ui);
                }
            }
        };
        for inst in &bb.insts {
            inst.for_each_use(&mut record);
        }
        if let Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } = bb.term {
            record(v);
        }
    }

    if cross_uses.is_empty() {
        return false;
    }

    let mut used_slots = vec![false; func.locals.len()];
    for bb in &func.blocks {
        for inst in &bb.insts {
            if let IrInst::LoadLocal { slot, .. }
            | IrInst::StoreLocal { slot, .. }
            | IrInst::AddrLocal { slot, .. } = inst
                && let Some(u) = used_slots.get_mut(*slot as usize)
            {
                *u = true;
            }
        }
    }

    // Per-block stores (keyed by defining ValueId) and per-block loads
    let mut stores_by_block: HashMap<usize, HashMap<ValueId, (SlotId, IrType)>> = HashMap::new();
    let mut loads_by_block: HashMap<usize, Vec<(ValueId, SlotId, IrType)>> = HashMap::new();

    for (bi, _ii, v, ty) in def_info {
        let Some(uis) = cross_uses.get(&v) else {
            continue;
        };
        let slot_id = alloc_or_reuse_slot(
            func,
            &mut used_slots,
            &ty,
            format!("__ssa_tmp_{v}"),
        );
        stores_by_block
            .entry(bi)
            .or_default()
            .insert(v, (slot_id, ty.clone()));
        for &ui in uis {
            loads_by_block
                .entry(ui)
                .or_default()
                .push((v, slot_id, ty.clone()));
        }
    }

    for (bi, store_map) in stores_by_block {
        let orig = std::mem::take(&mut func.blocks[bi].insts);
        let mut rebuilt = Vec::with_capacity(orig.len() + store_map.len());
        for inst in orig {
            let dst_opt = inst.dst();
            rebuilt.push(inst);
            if let Some(d) = dst_opt
                && let Some((slot, ty)) = store_map.get(&d)
            {
                rebuilt.push(IrInst::StoreLocal {
                    slot: *slot,
                    val: d,
                    ty: ty.clone(),
                });
            }
        }
        func.blocks[bi].insts = rebuilt;
    }

    for (ui, ops) in loads_by_block {
        insert_remat_loads_in_block(&mut func.blocks[ui], &mut func.next_val, ops);
    }

    true
}

/// Re-materializes per-block `LoadLocal`s for immutable (never stored, never address-taken)
/// parameters when an entry-block `LoadLocal` value is used in downstream blocks.
///
/// Kept as a public entry point; general `rematerialize_local_loads` subsumes this pass.
pub fn rematerialize_immutable_params(func: &mut IrFunction) -> bool {
    if func.blocks.is_empty() || func.param_count == 0 {
        return false;
    }
    let param_limit = func.param_count.min(func.locals.len());
    let mut non_imm = vec![false; param_limit];
    for bb in &func.blocks {
        for inst in &bb.insts {
            if let IrInst::StoreLocal { slot, .. } | IrInst::AddrLocal { slot, .. } = inst {
                let s = *slot as usize;
                if s < param_limit {
                    non_imm[s] = true;
                }
            }
        }
    }

    let mut entry_param_loads: HashMap<ValueId, (u32, IrType)> = HashMap::new();
    for inst in &func.blocks[0].insts {
        if let IrInst::LoadLocal { dst, slot, ty } = inst {
            let s = *slot as usize;
            if s < param_limit && !non_imm[s] {
                entry_param_loads.insert(*dst, (*slot, ty.clone()));
            }
        }
    }
    if entry_param_loads.is_empty() {
        return false;
    }

    let mut changed = false;
    for bi in 1..func.blocks.len() {
        let mut used: BTreeSet<ValueId> = BTreeSet::new();
        for inst in &func.blocks[bi].insts {
            inst.for_each_use(|v| {
                if entry_param_loads.contains_key(&v) {
                    used.insert(v);
                }
            });
        }
        match func.blocks[bi].term {
            Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } => {
                if entry_param_loads.contains_key(&v) {
                    used.insert(v);
                }
            }
            _ => {}
        }
        if used.is_empty() {
            continue;
        }

        let mut local_remap: HashMap<ValueId, ValueId> = HashMap::with_capacity(used.len());
        let mut new_loads: Vec<IrInst> = Vec::with_capacity(used.len());
        for old_v in used {
            let (slot, ty) = &entry_param_loads[&old_v];
            let new_v = func.next_val;
            func.next_val += 1;
            local_remap.insert(old_v, new_v);
            new_loads.push(IrInst::LoadLocal {
                dst: new_v,
                slot: *slot,
                ty: ty.clone(),
            });
        }

        for inst in &mut func.blocks[bi].insts {
            inst.for_each_use_mut(|v| {
                if let Some(&nv) = local_remap.get(v) {
                    *v = nv;
                }
            });
        }
        match &mut func.blocks[bi].term {
            Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } => {
                if let Some(&nv) = local_remap.get(v) {
                    *v = nv;
                }
            }
            _ => {}
        }
        let insert_at = func.blocks[bi]
            .insts
            .iter()
            .take_while(|i| matches!(i, IrInst::Phi { .. }))
            .count();
        func.blocks[bi].insts.splice(insert_at..insert_at, new_loads);
        changed = true;
    }

    changed
}

fn rewrite_phi_pred(bb: &mut BasicBlock, old_pred: BlockId, new_pred: BlockId) {
    for inst in &mut bb.insts {
        let IrInst::Phi { incoming, .. } = inst else {
            break;
        };
        for (pred, _) in incoming {
            if *pred == old_pred {
                *pred = new_pred;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify::verify_program;
    use crate::{ArithMode, IntType, IrBinOp, IrProgram};

    fn make_program(func: IrFunction) -> IrProgram {
        IrProgram {
            functions: vec![func],
            ..Default::default()
        }
    }

    fn base_fn(locals: Vec<LocalSlot>, blocks: Vec<BasicBlock>, next_val: u32) -> IrFunction {
        IrFunction {
            id: 0,
            name: "test".into(),
            is_pub: false,
            param_count: 0,
            locals,
            ret_ty: IrType::Int(IntType::U8),
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            blocks,
            next_val,
        }
    }

    fn u8_local(id: u32, name: &str) -> LocalSlot {
        LocalSlot {
            id,
            name: name.into(),
            ty: IrType::Int(IntType::U8),
            mutable: true,
        }
    }

    #[test]
    fn mem2reg_and_destruct_diamond_if_else() {
        let u8t = IrType::Int(IntType::U8);
        let func = base_fn(
            vec![u8_local(0, "x")],
            vec![
                BasicBlock {
                    id: 0,
                    insts: vec![IrInst::ConstBool { dst: 0, val: true }],
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
                            ty: u8t.clone(),
                            val: 10,
                        },
                        IrInst::StoreLocal {
                            slot: 0,
                            val: 1,
                            ty: u8t.clone(),
                        },
                    ],
                    term: Terminator::Goto(3),
                },
                BasicBlock {
                    id: 2,
                    insts: vec![
                        IrInst::ConstInt {
                            dst: 2,
                            ty: u8t.clone(),
                            val: 20,
                        },
                        IrInst::StoreLocal {
                            slot: 0,
                            val: 2,
                            ty: u8t.clone(),
                        },
                    ],
                    term: Terminator::Goto(3),
                },
                BasicBlock {
                    id: 3,
                    insts: vec![IrInst::LoadLocal {
                        dst: 3,
                        slot: 0,
                        ty: u8t,
                    }],
                    term: Terminator::Return(Some(3)),
                },
            ],
            4,
        );

        let mut prog = make_program(func);
        assert!(mem2reg(&mut prog.functions[0]));
        verify_program(&prog).expect("SSA after mem2reg must verify");
        assert!(matches!(
            prog.functions[0].blocks[3].insts.first(),
            Some(IrInst::Phi { incoming, .. }) if incoming.len() == 2
        ));

        assert!(destruct_ssa(&mut prog.functions[0]));
        verify_program(&prog).expect("IR after destruct_ssa must verify");
        assert!(!prog.functions[0]
            .blocks
            .iter()
            .any(|b| b.insts.iter().any(|i| matches!(i, IrInst::Phi { .. }))));
    }

    #[test]
    fn mem2reg_and_destruct_loop_swap() {
        let u8t = IrType::Int(IntType::U8);
        // x=1, y=2, c=true; loop header bb1: if c goto bb2 else bb3;
        // bb2 swaps x and y (tmp = x; x = y; y = tmp) then goto bb1; bb3 returns x.
        let func = base_fn(
            vec![u8_local(0, "x"), u8_local(1, "y")],
            vec![
                BasicBlock {
                    id: 0,
                    insts: vec![
                        IrInst::ConstInt {
                            dst: 0,
                            ty: u8t.clone(),
                            val: 1,
                        },
                        IrInst::ConstInt {
                            dst: 1,
                            ty: u8t.clone(),
                            val: 2,
                        },
                        IrInst::ConstBool { dst: 2, val: true },
                        IrInst::StoreLocal {
                            slot: 0,
                            val: 0,
                            ty: u8t.clone(),
                        },
                        IrInst::StoreLocal {
                            slot: 1,
                            val: 1,
                            ty: u8t.clone(),
                        },
                    ],
                    term: Terminator::Goto(1),
                },
                BasicBlock {
                    id: 1,
                    insts: vec![],
                    term: Terminator::Branch {
                        cond: 2,
                        then_bb: 2,
                        else_bb: 3,
                    },
                },
                BasicBlock {
                    id: 2,
                    insts: vec![
                        IrInst::LoadLocal {
                            dst: 3,
                            slot: 0,
                            ty: u8t.clone(),
                        },
                        IrInst::LoadLocal {
                            dst: 4,
                            slot: 1,
                            ty: u8t.clone(),
                        },
                        IrInst::StoreLocal {
                            slot: 0,
                            val: 4,
                            ty: u8t.clone(),
                        },
                        IrInst::StoreLocal {
                            slot: 1,
                            val: 3,
                            ty: u8t.clone(),
                        },
                    ],
                    term: Terminator::Goto(1),
                },
                BasicBlock {
                    id: 3,
                    insts: vec![IrInst::LoadLocal {
                        dst: 5,
                        slot: 0,
                        ty: u8t,
                    }],
                    term: Terminator::Return(Some(5)),
                },
            ],
            6,
        );

        let mut prog = make_program(func);
        assert!(mem2reg(&mut prog.functions[0]));
        verify_program(&prog).expect("loop swap SSA must verify");
        let header_phis: Vec<_> = prog.functions[0].blocks[1]
            .insts
            .iter()
            .filter(|i| matches!(i, IrInst::Phi { .. }))
            .collect();
        assert_eq!(header_phis.len(), 2, "both x and y get loop-header Phis");

        assert!(destruct_ssa(&mut prog.functions[0]));
        verify_program(&prog).expect("destructed loop swap must verify");
    }

    #[test]
    fn destruct_ssa_splits_critical_edge() {
        let u8t = IrType::Int(IntType::U8);
        // bb0: x = 1; Branch(cond, then=bb1, else=bb2) -- edge bb0->bb2 is critical!
        // bb1: x = 2; Goto(bb2)
        // bb2: return x
        let func = base_fn(
            vec![u8_local(0, "x")],
            vec![
                BasicBlock {
                    id: 0,
                    insts: vec![
                        IrInst::ConstInt {
                            dst: 0,
                            ty: u8t.clone(),
                            val: 1,
                        },
                        IrInst::ConstBool { dst: 1, val: false },
                        IrInst::StoreLocal {
                            slot: 0,
                            val: 0,
                            ty: u8t.clone(),
                        },
                    ],
                    term: Terminator::Branch {
                        cond: 1,
                        then_bb: 1,
                        else_bb: 2,
                    },
                },
                BasicBlock {
                    id: 1,
                    insts: vec![
                        IrInst::ConstInt {
                            dst: 2,
                            ty: u8t.clone(),
                            val: 2,
                        },
                        IrInst::StoreLocal {
                            slot: 0,
                            val: 2,
                            ty: u8t.clone(),
                        },
                    ],
                    term: Terminator::Goto(2),
                },
                BasicBlock {
                    id: 2,
                    insts: vec![IrInst::LoadLocal {
                        dst: 3,
                        slot: 0,
                        ty: u8t,
                    }],
                    term: Terminator::Return(Some(3)),
                },
            ],
            4,
        );

        let mut prog = make_program(func);
        assert!(mem2reg(&mut prog.functions[0]));
        verify_program(&prog).expect("critical-edge SSA must verify");
        let orig_blocks = prog.functions[0].blocks.len();

        assert!(destruct_ssa(&mut prog.functions[0]));
        verify_program(&prog).expect("destruct_ssa with split critical edge must verify");
        assert_eq!(
            prog.functions[0].blocks.len(),
            orig_blocks + 1,
            "critical edge bb0->bb2 must be split into a trampoline block"
        );
    }

    #[test]
    fn mem2reg_skips_escaped_and_uninitialized_locals() {
        let u8t = IrType::Int(IntType::U8);
        // slot 0: address-taken (AddrLocal) -> skipped
        // slot 1: only stored in then-branch (bb1), loaded in merge (bb2) -> live-in at bb0 -> skipped
        let func = base_fn(
            vec![u8_local(0, "addr_taken"), u8_local(1, "uninit")],
            vec![
                BasicBlock {
                    id: 0,
                    insts: vec![
                        IrInst::ConstBool { dst: 0, val: true },
                        IrInst::AddrLocal { dst: 1, slot: 0 },
                        IrInst::ConstInt {
                            dst: 2,
                            ty: u8t.clone(),
                            val: 5,
                        },
                        IrInst::StorePtr {
                            addr: 1,
                            val: 2,
                            ty: u8t.clone(),
                            volatile: false,
                        },
                    ],
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
                            dst: 3,
                            ty: u8t.clone(),
                            val: 9,
                        },
                        IrInst::StoreLocal {
                            slot: 1,
                            val: 3,
                            ty: u8t.clone(),
                        },
                    ],
                    term: Terminator::Goto(2),
                },
                BasicBlock {
                    id: 2,
                    insts: vec![
                        IrInst::LoadLocal {
                            dst: 4,
                            slot: 0,
                            ty: u8t.clone(),
                        },
                        IrInst::LoadLocal {
                            dst: 5,
                            slot: 1,
                            ty: u8t.clone(),
                        },
                        IrInst::BinOp {
                            dst: 6,
                            op: IrBinOp::Add,
                            mode: ArithMode::Wrapping,
                            ty: u8t,
                            lhs: 4,
                            rhs: 5,
                            site_id: 0,
                        },
                    ],
                    term: Terminator::Return(Some(6)),
                },
            ],
            7,
        );

        let mut prog = make_program(func);
        assert!(!mem2reg(&mut prog.functions[0]));
        verify_program(&prog).expect("unpromoted IR must still verify");
    }
}
