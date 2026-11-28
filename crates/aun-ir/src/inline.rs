//! Target-independent MIR function inliner (`aun-ir`).
//!
//! Inlines non-public single-caller helpers (up to moderate size) and tiny leaf
//! helpers into callers before final target-specific optimization, then clears
//! dead non-public callees so backends reserve zero RAM/ROM for them.

use crate::{
    BlockId, IrFnTarget, IrFunction, IrInst, IrProgram, IrType, LocalSlot, SlotId, Terminator,
    ValueId,
    cfg::Cfg,
    opt::{self, OptConfig},
};
use std::collections::{HashMap, HashSet, VecDeque};

const MAX_SINGLE_CALLER_INSTS: usize = 64;
const MAX_HOT_SINGLE_CALLER_INSTS: usize = 140;
const MAX_TINY_LEAF_INSTS: usize = 6;
const MAX_HOT_LEAF_HELPER_INSTS: usize = 72;
const MAX_HOT_NONLEAF_HELPER_INSTS: usize = 36;
const MAX_HOT_LEAF_TOTAL_DUP_INSTS: usize = 280;
const MAX_HOT_NONLEAF_TOTAL_DUP_INSTS: usize = 120;
const MAX_SMALL_HOT_LEAF_PARTIAL_INSTS: usize = 64;
const MAX_REF_MISMATCH_INSTS: usize = 20;
const MAX_CALLER_INSTS_AFTER_INLINE: usize = 600;
const MAX_HOT_CALLER_INSTS_AFTER_INLINE: usize = 800;
const MAX_INLINE_ROUNDS: usize = 4;

fn fn_inst_cost(f: &IrFunction) -> usize {
    f.blocks.iter().map(|b| b.insts.len() + 1).sum()
}

fn has_uninlineable_inst(f: &IrFunction) -> bool {
    if f.blocks.first().is_some_and(|bb| {
        bb.insts
            .iter()
            .any(|inst| matches!(inst, IrInst::Phi { .. }))
    }) {
        return true;
    }
    f.blocks.iter().any(|bb| {
        bb.insts.iter().any(|inst| {
            matches!(
                inst,
                IrInst::InlineAsm { .. }
                    | IrInst::CriticalEnter
                    | IrInst::CriticalExit
                    | IrInst::BankEnter { .. }
                    | IrInst::BankExit { .. }
            )
        })
    })
}

fn is_leaf_fn(f: &IrFunction) -> bool {
    !f.blocks.iter().any(|bb| {
        bb.insts.iter().any(|inst| {
            matches!(
                inst,
                IrInst::Call { .. }
                    | IrInst::CallIndirect { .. }
                    | IrInst::ExternCall { .. }
                    | IrInst::StateDispatch { .. }
                    | IrInst::StateTransition { .. }
            )
        })
    })
}

fn collect_addr_taken_and_machine_fns(program: &IrProgram) -> (HashSet<u32>, HashSet<u32>) {
    let mut addr_taken = HashSet::new();
    for f in &program.functions {
        for bb in &f.blocks {
            for inst in &bb.insts {
                if let IrInst::AddrFn {
                    target: IrFnTarget::Fn(fid),
                    ..
                } = inst
                {
                    addr_taken.insert(*fid);
                }
            }
        }
    }
    let mut machine_fns = HashSet::new();
    for m in &program.machines {
        for st in &m.states {
            if let Some(fid) = st.on_enter {
                machine_fns.insert(fid);
            }
            if let Some(fid) = st.on_exit {
                machine_fns.insert(fid);
            }
            for &fid in st.methods.values() {
                machine_fns.insert(fid);
            }
        }
    }
    if let Some(target) = &program.target {
        let mut hook_names: HashSet<&str> = HashSet::new();
        if let Some(ref sfn) = target.on_swap_fn {
            hook_names.insert(sfn.as_str());
        }
        for slot in &target.overlay_slots {
            if let Some(ref lfn) = slot.load_fn {
                hook_names.insert(lfn.as_str());
            }
        }
        for f in &program.functions {
            if hook_names.contains(f.name.as_str()) {
                machine_fns.insert(f.id);
            }
        }
    }
    (addr_taken, machine_fns)
}

fn count_direct_calls(program: &IrProgram) -> Vec<usize> {
    let n = program.functions.len();
    let mut counts = vec![0usize; n];
    for f in &program.functions {
        if f.blocks.is_empty() {
            continue;
        }
        for bb in &f.blocks {
            for inst in &bb.insts {
                if let IrInst::Call { func, .. } = inst
                    && (*func as usize) < n
                {
                    counts[*func as usize] += 1;
                }
            }
        }
    }
    counts
}

fn count_hot_direct_calls(program: &IrProgram, hot_info: &LoopHotInfo) -> Vec<usize> {
    let n = program.functions.len();
    let mut counts = vec![0usize; n];
    for (caller_idx, f) in program.functions.iter().enumerate() {
        if f.blocks.is_empty() {
            continue;
        }
        for bb in &f.blocks {
            let is_hot = hot_info.loop_bbs[caller_idx].contains(&bb.id)
                || hot_info.loop_hot_fns[caller_idx];
            if !is_hot {
                continue;
            }
            for inst in &bb.insts {
                if let IrInst::Call { func, .. } = inst
                    && (*func as usize) < n
                {
                    counts[*func as usize] += 1;
                }
            }
        }
    }
    counts
}

fn caller_ref_param_origin_slots(caller: &IrFunction) -> HashSet<ValueId> {
    let mut param_ref_vals = HashSet::new();
    for bb in &caller.blocks {
        for inst in &bb.insts {
            if let IrInst::LoadLocal { dst, slot, .. } = inst
                && let Some(pos) = caller.locals.iter().position(|l| l.id == *slot)
                && pos < caller.param_count
                && matches!(caller.locals[pos].ty, IrType::Ref { .. })
            {
                param_ref_vals.insert(*dst);
            }
        }
    }
    param_ref_vals
}

struct LoopHotInfo {
    loop_bbs: Vec<HashSet<BlockId>>,
    loop_depth: Vec<HashMap<BlockId, usize>>,
    loop_hot_fns: Vec<bool>,
    callees: Vec<HashSet<usize>>,
}

fn is_main_event_loop(
    f: &IrFunction,
    cfg: &Cfg,
    loops: &[crate::cfg::NaturalLoop],
    lp: &crate::cfg::NaturalLoop,
) -> bool {
    let has_exit_edge = lp
        .body
        .iter()
        .any(|&bb| cfg.succs(bb).iter().any(|s| !lp.body.contains(s)));
    if !has_exit_edge {
        return true;
    }
    let is_outermost = !loops
        .iter()
        .any(|other| other.body.len() > lp.body.len() && lp.body.is_subset(&other.body));
    if !is_outermost {
        return false;
    }
    f.blocks.iter().any(|bb| {
        lp.body.contains(&bb.id)
            && bb.insts.iter().any(|inst| {
                matches!(
                    inst,
                    IrInst::ExternCall { .. }
                        | IrInst::StateDispatch { .. }
                        | IrInst::StateTransition { .. }
                )
            })
    })
}

fn compute_loop_hot_info(program: &IrProgram) -> LoopHotInfo {
    let n = program.functions.len();
    let mut loop_bbs = vec![HashSet::new(); n];
    let mut loop_depth = vec![HashMap::new(); n];
    let mut callees = vec![HashSet::new(); n];
    let mut loop_hot_fns = vec![false; n];

    for (fi, f) in program.functions.iter().enumerate() {
        if f.blocks.is_empty() {
            continue;
        }
        let cfg = Cfg::build(f);
        let loops = cfg.natural_loops();
        for lp in loops {
            if f.name == "main" && is_main_event_loop(f, &cfg, loops, lp) {
                continue;
            }
            loop_bbs[fi].extend(lp.body.iter().copied());
            for &bb_id in &lp.body {
                *loop_depth[fi].entry(bb_id).or_insert(0) += 1;
            }
        }
        for bb in &f.blocks {
            let bb_in_loop = loop_bbs[fi].contains(&bb.id);
            for inst in &bb.insts {
                match inst {
                    IrInst::Call { func, .. } => {
                        let cid = *func as usize;
                        if cid < n {
                            callees[fi].insert(cid);
                            if bb_in_loop {
                                loop_hot_fns[cid] = true;
                            }
                        }
                    }
                    IrInst::StateTransition {
                        machine_id,
                        from_state,
                        to_state,
                        ..
                    } => {
                        if let Some(mdef) = program.machines.get(*machine_id as usize) {
                            if let Some(enter_fn) = mdef
                                .states
                                .get(*to_state as usize)
                                .and_then(|s| s.on_enter)
                            {
                                let cid = enter_fn as usize;
                                if cid < n {
                                    callees[fi].insert(cid);
                                    if bb_in_loop {
                                        loop_hot_fns[cid] = true;
                                    }
                                }
                            }
                            if let Some(from) = from_state
                                && let Some(exit_fn) = mdef
                                    .states
                                    .get(*from as usize)
                                    .and_then(|s| s.on_exit)
                            {
                                let cid = exit_fn as usize;
                                if cid < n {
                                    callees[fi].insert(cid);
                                    if bb_in_loop {
                                        loop_hot_fns[cid] = true;
                                    }
                                }
                            }
                        }
                    }
                    IrInst::StateDispatch {
                        machine_id, method, ..
                    } => {
                        if let Some(mdef) = program.machines.get(*machine_id as usize) {
                            for st in &mdef.states {
                                if let Some(&fid) = st.methods.get(method) {
                                    let cid = fid as usize;
                                    if cid < n {
                                        callees[fi].insert(cid);
                                        if bb_in_loop {
                                            loop_hot_fns[cid] = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    let mut queue: VecDeque<usize> = loop_hot_fns
        .iter()
        .enumerate()
        .filter_map(|(i, &hot)| hot.then_some(i))
        .collect();
    while let Some(u) = queue.pop_front() {
        for &v in &callees[u] {
            if !loop_hot_fns[v] {
                loop_hot_fns[v] = true;
                queue.push_back(v);
            }
        }
    }

    LoopHotInfo {
        loop_bbs,
        loop_depth,
        loop_hot_fns,
        callees,
    }
}

fn can_reach(src: usize, dst: usize, callees: &[HashSet<usize>]) -> bool {
    if src == dst {
        return true;
    }
    let mut visited = vec![false; callees.len()];
    let mut stack = vec![src];
    visited[src] = true;
    while let Some(u) = stack.pop() {
        for &v in &callees[u] {
            if v == dst {
                return true;
            }
            if !visited[v] {
                visited[v] = true;
                stack.push(v);
            }
        }
    }
    false
}

fn is_in_call_cycle(fid: usize, callees: &[HashSet<usize>]) -> bool {
    callees[fid]
        .iter()
        .any(|&succ| can_reach(succ, fid, callees))
}

fn extern_requires_bank_or_di(program: &IrProgram, callee: &IrFunction) -> bool {
    callee.blocks.iter().any(|bb| {
        bb.insts.iter().any(|inst| {
            if let IrInst::ExternCall { extern_fn, .. } = inst
                && let Some(ef) = program.externs.get(*extern_fn as usize)
            {
                !ef.requires_banks.is_empty() || ef.requires_di
            } else {
                false
            }
        })
    })
}

fn has_state_op(f: &IrFunction) -> bool {
    f.blocks.iter().any(|bb| {
        bb.insts.iter().any(|inst| {
            matches!(
                inst,
                IrInst::StateTransition { .. } | IrInst::StateDispatch { .. }
            )
        })
    })
}

fn is_callee_basically_inlineable(
    program: &IrProgram,
    callee: &IrFunction,
    addr_taken: &HashSet<u32>,
    machine_fns: &HashSet<u32>,
    isr_roots: &HashSet<u32>,
) -> bool {
    !(callee.blocks.is_empty()
        || callee.is_pub
        || callee.export_name.is_some()
        || callee.name == "main"
        || callee.is_test
        || callee.interrupt_source.is_some()
        || isr_roots.contains(&callee.id)
        || callee.isr_using.is_some()
        || callee.overlay_state.is_some()
        || callee.owner_state.is_some()
        || callee.is_reentrant
        || !callee.requires_banks.is_empty()
        || callee.requires_di
        || addr_taken.contains(&callee.id)
        || machine_fns.contains(&callee.id)
        || callee.locals.len() < callee.param_count
        || has_uninlineable_inst(callee)
        || callee.cycle_budget.is_some()
        || callee.mock_target.is_some()
        || extern_requires_bank_or_di(program, callee))
}

#[allow(clippy::too_many_arguments)]
fn callee_has_inlineable_call(
    program: &IrProgram,
    callee: &IrFunction,
    call_counts: &[usize],
    snapshots: &[IrFunction],
    addr_taken: &HashSet<u32>,
    machine_fns: &HashSet<u32>,
    isr_roots: &HashSet<u32>,
    callees: &[HashSet<usize>],
) -> bool {
    callee.blocks.iter().any(|bb| {
        bb.insts.iter().any(|inst| {
            if let IrInst::Call { func, args, .. } = inst
                && let Some(sub) = snapshots.get(*func as usize)
                && sub.id != callee.id
                && args.len() == sub.param_count
                && is_callee_basically_inlineable(program, sub, addr_taken, machine_fns, isr_roots)
                && !can_reach(sub.id as usize, callee.id as usize, callees)
            {
                let sub_cost = fn_inst_cost(sub);
                let sub_leaf = is_leaf_fn(sub);
                let sub_calls = call_counts.get(*func as usize).copied().unwrap_or(0);
                (sub_calls == 1 && sub_cost <= MAX_HOT_SINGLE_CALLER_INSTS)
                    || (sub_leaf && sub_cost <= MAX_HOT_LEAF_HELPER_INSTS)
                    || (!sub_leaf && sub_cost <= MAX_HOT_NONLEAF_HELPER_INSTS)
            } else {
                false
            }
        })
    })
}

#[allow(clippy::too_many_arguments)]
fn should_inline_site(
    program: &IrProgram,
    caller: &IrFunction,
    caller_bb_id: BlockId,
    callee: &IrFunction,
    callee_cost: usize,
    callee_is_leaf: bool,
    callee_has_inlineable: bool,
    args: &[ValueId],
    call_count: usize,
    hot_call_count: usize,
    remaining_calls_to_callee: usize,
    inlined_dup_insts_so_far: usize,
    is_hot_site: bool,
    remaining_dup_budget: usize,
    addr_taken: &HashSet<u32>,
    machine_fns: &HashSet<u32>,
    isr_roots: &HashSet<u32>,
    callees: &[HashSet<usize>],
    caller_ref_vals: &HashSet<ValueId>,
) -> bool {
    if callee.id == caller.id
        || caller.is_test
        || args.len() != callee.param_count
        || caller.locals.len() < caller.param_count
        || !is_callee_basically_inlineable(program, callee, addr_taken, machine_fns, isr_roots)
        || can_reach(callee.id as usize, caller.id as usize, callees)
        || is_in_call_cycle(callee.id as usize, callees)
    {
        return false;
    }
    let _ = caller_bb_id;

    let caller_is_overlay = caller.overlay_state.is_some() || caller.owner_state.is_some();
    if has_state_op(callee) && (call_count != 1 || caller_is_overlay) {
        return false;
    }

    let caller_cost = fn_inst_cost(caller);
    let single_limit = if is_hot_site {
        MAX_HOT_SINGLE_CALLER_INSTS
    } else {
        MAX_SINGLE_CALLER_INSTS
    };
    let is_single_caller = call_count == 1 && callee_cost <= single_limit;
    let is_tiny_leaf = call_count >= 1 && callee_is_leaf && callee_cost <= MAX_TINY_LEAF_INSTS;
    let hot_dup_sites = if hot_call_count == call_count {
        hot_call_count.saturating_sub(1)
    } else {
        hot_call_count
    };
    let net_dup_cost = if remaining_calls_to_callee <= 1 {
        0
    } else {
        callee_cost
    };
    let hot_helper_limit = if callee_is_leaf {
        MAX_HOT_LEAF_HELPER_INSTS
    } else {
        MAX_HOT_NONLEAF_HELPER_INSTS
    };
    let total_dup_limit = if callee_is_leaf {
        MAX_HOT_LEAF_TOTAL_DUP_INSTS
    } else {
        MAX_HOT_NONLEAF_TOTAL_DUP_INSTS
    };
    let dup_budget_ok = if callee_is_leaf && callee_cost <= MAX_SMALL_HOT_LEAF_PARTIAL_INSTS {
        inlined_dup_insts_so_far.saturating_add(net_dup_cost) <= total_dup_limit
    } else {
        callee_cost.saturating_mul(hot_dup_sites) <= total_dup_limit
    };
    let is_hot_small = !is_single_caller
        && is_hot_site
        && !caller_is_overlay
        && hot_call_count >= 1
        && callee_cost <= hot_helper_limit
        && !callee_has_inlineable
        && dup_budget_ok
        && remaining_dup_budget >= net_dup_cost;
    if !is_single_caller && !is_tiny_leaf && !is_hot_small {
        return false;
    }
    let caller_limit = if is_hot_site {
        MAX_HOT_CALLER_INSTS_AFTER_INLINE
    } else {
        MAX_CALLER_INSTS_AFTER_INLINE
    };
    if (!is_single_caller || callee_cost > MAX_SINGLE_CALLER_INSTS)
        && caller_cost + callee_cost > caller_limit
    {
        return false;
    }

    // Preserve IX-register promotion in callers that already take a `&`/`&mut` reference param
    // when the callee is medium-sized (> 20 IR insts) and operates on a different Ref target.
    let caller_has_ref = caller.locals[..caller.param_count]
        .iter()
        .any(|l| matches!(l.ty, IrType::Ref { .. }));
    if caller_has_ref && callee_cost > MAX_REF_MISMATCH_INSTS {
        let mut has_diff_ref = false;
        for (p_idx, arg_val) in args.iter().enumerate() {
            if matches!(callee.locals[p_idx].ty, IrType::Ref { .. })
                && !caller_ref_vals.contains(arg_val)
            {
                has_diff_ref = true;
                break;
            }
        }
        if has_diff_ref {
            return false;
        }
    }

    true
}

fn remap_inst(
    inst: &IrInst,
    val_map: &HashMap<ValueId, ValueId>,
    slot_map: &HashMap<SlotId, SlotId>,
    bb_map: &HashMap<BlockId, BlockId>,
    alloc_val: &mut dyn FnMut() -> ValueId,
    val_map_mut: &mut HashMap<ValueId, ValueId>,
) -> IrInst {
    let mut cloned = inst.clone();

    // First remap uses in-place (including Phi incoming ValueIds)
    cloned.for_each_use_mut(|v| {
        if let Some(&mapped) = val_map_mut.get(v) {
            *v = mapped;
        } else if let Some(&mapped) = val_map.get(v) {
            *v = mapped;
        }
    });

    // Remap slot IDs and Phi incoming BlockIds
    match &mut cloned {
        IrInst::LoadLocal { slot, .. }
        | IrInst::StoreLocal { slot, .. }
        | IrInst::AddrLocal { slot, .. } => {
            if let Some(&new_slot) = slot_map.get(slot) {
                *slot = new_slot;
            }
        }
        IrInst::Phi { incoming, .. } => {
            for (pred_bb, _) in incoming {
                if let Some(&new_bb) = bb_map.get(pred_bb) {
                    *pred_bb = new_bb;
                }
            }
        }
        _ => {}
    }

    // Remap defined destination ValueId (allocating if not yet pre-mapped)
    if let Some(dst) = cloned.dst_mut() {
        let orig = *dst;
        let new_v = *val_map_mut.entry(orig).or_insert_with(alloc_val);
        *dst = new_v;
    }

    cloned
}

/// Inlines `callee` at `(bb_idx, call_idx)` and returns newly created `BlockId`s.
fn inline_one_call_site(
    caller: &mut IrFunction,
    callee: &IrFunction,
    bb_idx: usize,
    call_idx: usize,
) -> Vec<BlockId> {
    let (dst_opt, args) = match &caller.blocks[bb_idx].insts[call_idx] {
        IrInst::Call { dst, args, .. } => (*dst, args.clone()),
        _ => return Vec::new(),
    };

    // 1. Allocate fresh LocalSlots in caller for all callee locals
    let mut next_slot_id: SlotId = caller.locals.iter().map(|l| l.id + 1).max().unwrap_or(0);
    let mut slot_map: HashMap<SlotId, SlotId> = HashMap::new();
    let mut param_slots: Vec<SlotId> = Vec::with_capacity(callee.param_count);
    for (idx, l) in callee.locals.iter().enumerate() {
        let new_id = next_slot_id;
        next_slot_id += 1;
        slot_map.insert(l.id, new_id);
        if idx < callee.param_count {
            param_slots.push(new_id);
        }
        caller.locals.push(LocalSlot {
            id: new_id,
            name: format!("__inl_{}_{}", callee.name, l.name),
            ty: l.ty.clone(),
            mutable: l.mutable || idx < callee.param_count,
        });
    }

    let ret_slot = match dst_opt {
        Some(_) if callee.ret_ty != IrType::Unit => {
            let sid = next_slot_id;
            caller.locals.push(LocalSlot {
                id: sid,
                name: format!("__inl_{}_ret", callee.name),
                ty: callee.ret_ty.clone(),
                mutable: true,
            });
            Some(sid)
        }
        _ => None,
    };

    // 2. Pre-allocate fresh ValueIds for all defs in callee so forward/back references resolve cleanly
    let mut next_val = caller.next_val;
    let mut val_map: HashMap<ValueId, ValueId> = HashMap::new();
    for cb in &callee.blocks {
        for inst in &cb.insts {
            if let Some(d) = inst.dst() {
                val_map.entry(d).or_insert_with(|| {
                    let nv = next_val;
                    next_val += 1;
                    nv
                });
            }
        }
    }

    // 3. Allocate fresh BlockIds for cloned callee blocks + continuation block
    let max_bb_id = caller.blocks.iter().map(|b| b.id).max().unwrap_or(0);
    let mut bb_map: HashMap<BlockId, BlockId> = HashMap::new();
    for (ci, cb) in callee.blocks.iter().enumerate() {
        bb_map.insert(cb.id, max_bb_id + 1 + ci as u32);
    }
    let cont_bb_id = max_bb_id + 1 + callee.blocks.len() as u32;
    let callee_entry_bb = bb_map[&callee.blocks[0].id];

    // 4. Split caller.blocks[bb_idx] around `call_idx`
    let pre_bb_id = caller.blocks[bb_idx].id;
    let mut tail_insts = caller.blocks[bb_idx].insts.split_off(call_idx + 1);
    caller.blocks[bb_idx].insts.pop(); // remove the Call itself
    let orig_term = std::mem::replace(
        &mut caller.blocks[bb_idx].term,
        Terminator::Goto(callee_entry_bb),
    );
    // Any caller Phi in a successor of `pre_bb_id` now receives control from `cont_bb_id`.
    let orig_targets: [Option<BlockId>; 2] = match orig_term {
        Terminator::Goto(t) => [Some(t), None],
        Terminator::Branch {
            then_bb, else_bb, ..
        } => [
            Some(then_bb),
            (else_bb != then_bb).then_some(else_bb),
        ],
        Terminator::Return(_) | Terminator::Unreachable => [None, None],
    };
    for target_id in orig_targets.into_iter().flatten() {
        if let Some(target_bb) = caller.blocks.iter_mut().find(|b| b.id == target_id) {
            for inst in &mut target_bb.insts {
                if let IrInst::Phi { incoming, .. } = inst {
                    for (pred_bb, _) in incoming {
                        if *pred_bb == pre_bb_id {
                            *pred_bb = cont_bb_id;
                        }
                    }
                }
            }
        }
    }
    for (p_slot, arg_val) in param_slots.into_iter().zip(args.into_iter()) {
        let p_ty = callee.locals.iter().find(|l| Some(&l.id) == slot_map.iter().find(|&(_, &v)| v == p_slot).map(|(k, _)| k))
            .map(|l| l.ty.clone())
            .unwrap_or(IrType::Unit);
        caller.blocks[bb_idx].insts.push(IrInst::StoreLocal {
            slot: p_slot,
            val: arg_val,
            ty: p_ty,
        });
    }

    // 5. Build cloned callee blocks
    let mut val_map_mut = val_map.clone();
    let mut alloc_val = || {
        let nv = next_val;
        next_val += 1;
        nv
    };
    let mut new_blocks = Vec::with_capacity(callee.blocks.len() + 1);
    for cb in &callee.blocks {
        let new_id = bb_map[&cb.id];
        let mut new_insts = Vec::with_capacity(cb.insts.len() + 1);
        for inst in &cb.insts {
            new_insts.push(remap_inst(
                inst,
                &val_map,
                &slot_map,
                &bb_map,
                &mut alloc_val,
                &mut val_map_mut,
            ));
        }
        let new_term = match &cb.term {
            Terminator::Unreachable => Terminator::Unreachable,
            Terminator::Goto(t) => Terminator::Goto(bb_map[t]),
            Terminator::Branch {
                cond,
                then_bb,
                else_bb,
            } => Terminator::Branch {
                cond: *val_map_mut.get(cond).unwrap_or(cond),
                then_bb: bb_map[then_bb],
                else_bb: bb_map[else_bb],
            },
            Terminator::Return(ret_val) => {
                if let (Some(rs), Some(rv)) = (ret_slot, ret_val) {
                    let mapped_rv = *val_map_mut.get(rv).unwrap_or(rv);
                    new_insts.push(IrInst::StoreLocal {
                        slot: rs,
                        val: mapped_rv,
                        ty: callee.ret_ty.clone(),
                    });
                }
                Terminator::Goto(cont_bb_id)
            }
        };
        new_blocks.push(crate::BasicBlock {
            id: new_id,
            insts: new_insts,
            term: new_term,
        });
    }

    // 6. Continuation block receives optional return value load + caller tail instructions
    let mut cont_insts = Vec::with_capacity(tail_insts.len() + 1);
    if let Some(d) = dst_opt
        && let Some(rs) = ret_slot
    {
        cont_insts.push(IrInst::LoadLocal {
            dst: d,
            slot: rs,
            ty: callee.ret_ty.clone(),
        });
    }
    cont_insts.append(&mut tail_insts);
    new_blocks.push(crate::BasicBlock {
        id: cont_bb_id,
        insts: cont_insts,
        term: orig_term,
    });

    let added_bb_ids: Vec<BlockId> = new_blocks.iter().map(|b| b.id).collect();
    caller.next_val = next_val;
    caller.blocks.extend(new_blocks);
    for (&old_bb, &bound) in &callee.loop_bounds {
        if let Some(&new_bb) = bb_map.get(&old_bb) {
            caller.loop_bounds.entry(new_bb).or_insert(bound);
        }
    }
    added_bb_ids
}

/// Runs bounded single-caller, tiny-leaf, and loop-hot small-helper inlining over `program`,
/// then cleans up modified callers and eliminates unreferenced internal callees.
pub fn inline_functions(program: &mut IrProgram, cfg: &OptConfig) -> bool {
    let (addr_taken, machine_fns) = collect_addr_taken_and_machine_fns(program);
    let isr_roots: HashSet<u32> = program
        .functions
        .iter()
        .filter(|f| program.is_isr_root(f))
        .map(|f| f.id)
        .collect();
    let initial_program_insts: usize = program.functions.iter().map(fn_inst_cost).sum();
    let mut remaining_dup_budget: usize = (initial_program_insts / 5).max(340);
    let mut any_changed = false;
    let mut inlined_callees: HashSet<u32> = HashSet::new();

    for _round in 0..MAX_INLINE_ROUNDS {
        let mut round_changed = false;
        let call_counts = count_direct_calls(program);
        let mut hot_info = compute_loop_hot_info(program);
        let hot_call_counts = count_hot_direct_calls(program, &hot_info);
        let snapshots: Vec<IrFunction> = program.functions.clone();
        let n = program.functions.len();
        let snapshot_costs: Vec<usize> = snapshots.iter().map(fn_inst_cost).collect();
        let snapshot_is_leaf: Vec<bool> = snapshots.iter().map(is_leaf_fn).collect();
        let snapshot_has_inlineable_call: Vec<bool> = snapshots
            .iter()
            .map(|s| {
                callee_has_inlineable_call(
                    program,
                    s,
                    &call_counts,
                    &snapshots,
                    &addr_taken,
                    &machine_fns,
                    &isr_roots,
                    &hot_info.callees,
                )
            })
            .collect();
        let mut remaining_calls: Vec<usize> = call_counts.clone();
        let mut inlined_dup_insts: Vec<usize> = vec![0usize; n];
        let mut caller_ref_vals: Vec<HashSet<ValueId>> = program
            .functions
            .iter()
            .map(caller_ref_param_origin_slots)
            .collect();
        let mut modified_callers: Vec<bool> = vec![false; n];

        loop {
            type PriorityKey = (
                bool,
                usize,
                usize,
                std::cmp::Reverse<usize>,
                usize,
                usize,
                usize,
            );
            let mut best: Option<(PriorityKey, usize, usize, usize, usize, bool, usize)> = None;

            for caller_idx in 0..n {
                if program.functions[caller_idx].blocks.is_empty() {
                    continue;
                }
                for (bi, bb) in program.functions[caller_idx].blocks.iter().enumerate() {
                    let is_hot_site = hot_info.loop_bbs[caller_idx].contains(&bb.id)
                        || hot_info.loop_hot_fns[caller_idx];
                    let site_bb_depth = hot_info.loop_depth[caller_idx]
                        .get(&bb.id)
                        .copied()
                        .unwrap_or(0);
                    let site_depth =
                        site_bb_depth + usize::from(hot_info.loop_hot_fns[caller_idx]);
                    for (ii, inst) in bb.insts.iter().enumerate() {
                        let IrInst::Call { func, args, .. } = inst else {
                            continue;
                        };
                        let callee_idx = *func as usize;
                        if callee_idx >= n || callee_idx == caller_idx {
                            continue;
                        }
                        let callee = &snapshots[callee_idx];
                        let callee_cost = snapshot_costs[callee_idx];
                        let callee_is_leaf = snapshot_is_leaf[callee_idx];
                        let count = call_counts[callee_idx];
                        let hot_count = hot_call_counts[callee_idx];
                        if !should_inline_site(
                            program,
                            &program.functions[caller_idx],
                            bb.id,
                            callee,
                            callee_cost,
                            callee_is_leaf,
                            snapshot_has_inlineable_call[callee_idx],
                            args,
                            count,
                            hot_count,
                            remaining_calls[callee_idx],
                            inlined_dup_insts[callee_idx],
                            is_hot_site,
                            remaining_dup_budget,
                            &addr_taken,
                            &machine_fns,
                            &isr_roots,
                            &hot_info.callees,
                            &caller_ref_vals[caller_idx],
                        ) {
                            continue;
                        }
                        let hot_dup_sites = if count == 1 {
                            0
                        } else if hot_count == count {
                            hot_count.saturating_sub(1)
                        } else {
                            hot_count
                        };
                        let key: PriorityKey = (
                            !callee_is_leaf,
                            callee_cost,
                            hot_dup_sites,
                            std::cmp::Reverse(site_depth),
                            caller_idx,
                            bi,
                            ii,
                        );
                        if best.as_ref().is_none_or(|(bk, ..)| key < *bk) {
                            best = Some((
                                key,
                                caller_idx,
                                bi,
                                ii,
                                callee_idx,
                                is_hot_site,
                                site_bb_depth,
                            ));
                        }
                    }
                }
            }

            let Some((_, caller_idx, bi, ii, callee_idx, was_hot_site, site_bb_depth)) = best
            else {
                break;
            };
            let callee = &snapshots[callee_idx];
            let callee_cost = snapshot_costs[callee_idx];
            let count = call_counts[callee_idx];
            remaining_calls[callee_idx] = remaining_calls[callee_idx].saturating_sub(1);
            if count != 1 && callee_cost > MAX_TINY_LEAF_INSTS && remaining_calls[callee_idx] > 0 {
                remaining_dup_budget = remaining_dup_budget.saturating_sub(callee_cost);
                inlined_dup_insts[callee_idx] += callee_cost;
            }
            inlined_callees.insert(callee.id);
            let added_bbs =
                inline_one_call_site(&mut program.functions[caller_idx], callee, bi, ii);
            if was_hot_site {
                hot_info.loop_bbs[caller_idx].extend(added_bbs.iter().copied());
            }
            if site_bb_depth > 0 {
                for bb_id in added_bbs {
                    hot_info.loop_depth[caller_idx]
                        .entry(bb_id)
                        .or_insert(site_bb_depth);
                }
            }
            caller_ref_vals[caller_idx] =
                caller_ref_param_origin_slots(&program.functions[caller_idx]);
            modified_callers[caller_idx] = true;
            round_changed = true;
            any_changed = true;
        }

        for (caller_idx, &changed) in modified_callers.iter().enumerate() {
            if changed {
                opt::optimize_function_with_config(&mut program.functions[caller_idx], cfg);
            }
        }

        // Recompute call counts and clear dead non-public helpers whose callers were all inlined.
        let remaining_calls = count_direct_calls(program);
        for f in &mut program.functions {
            if inlined_callees.contains(&f.id)
                && !f.blocks.is_empty()
                && !f.is_pub
                && f.export_name.is_none()
                && f.name != "main"
                && !f.is_test
                && f.interrupt_source.is_none()
                && !isr_roots.contains(&f.id)
                && f.overlay_state.is_none()
                && f.owner_state.is_none()
                && !addr_taken.contains(&f.id)
                && !machine_fns.contains(&f.id)
                && remaining_calls[f.id as usize] == 0
            {
                f.blocks.clear();
                f.locals.clear();
                f.param_count = 0;
                f.next_val = 0;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArithMode, BasicBlock, IntType, IrBinOp, verify};

    fn make_test_fn(
        id: u32,
        name: &str,
        is_pub: bool,
        param_count: usize,
        locals: Vec<LocalSlot>,
        ret_ty: IrType,
        blocks: Vec<BasicBlock>,
        next_val: ValueId,
    ) -> IrFunction {
        IrFunction {
            id,
            name: name.into(),
            is_pub,
            param_count,
            locals,
            ret_ty,
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
            blocks,
            next_val,
        }
    }

    #[test]
    fn inlines_single_caller_multi_return_and_eliminates_callee() {
        let mut prog = IrProgram::default();
        // fn helper(x: u8) -> u8 { if x > 10 { return 10; } return x +% 1; }
        prog.functions.push(make_test_fn(
            0,
            "helper",
            false,
            1,
            vec![LocalSlot {
                id: 0,
                name: "x".into(),
                ty: IrType::Int(IntType::U8),
                mutable: false,
            }],
            IrType::Int(IntType::U8),
            vec![
                BasicBlock {
                    id: 0,
                    insts: vec![
                        IrInst::LoadLocal {
                            dst: 0,
                            slot: 0,
                            ty: IrType::Int(IntType::U8),
                        },
                        IrInst::ConstInt {
                            dst: 1,
                            ty: IrType::Int(IntType::U8),
                            val: 10,
                        },
                        IrInst::BinOp {
                            dst: 2,
                            op: IrBinOp::Gt,
                            mode: ArithMode::Wrapping,
                            ty: IrType::Int(IntType::U8),
                            lhs: 0,
                            rhs: 1,
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
                    insts: vec![IrInst::ConstInt {
                        dst: 3,
                        ty: IrType::Int(IntType::U8),
                        val: 10,
                    }],
                    term: Terminator::Return(Some(3)),
                },
                BasicBlock {
                    id: 2,
                    insts: vec![
                        IrInst::LoadLocal {
                            dst: 4,
                            slot: 0,
                            ty: IrType::Int(IntType::U8),
                        },
                        IrInst::ConstInt {
                            dst: 5,
                            ty: IrType::Int(IntType::U8),
                            val: 1,
                        },
                        IrInst::BinOp {
                            dst: 6,
                            op: IrBinOp::Add,
                            mode: ArithMode::Wrapping,
                            ty: IrType::Int(IntType::U8),
                            lhs: 4,
                            rhs: 5,
                            site_id: 0,
                        },
                    ],
                    term: Terminator::Return(Some(6)),
                },
            ],
            7,
        ));

        // pub fn caller(v: u8) -> u8 { helper(v) +% 2 }
        prog.functions.push(make_test_fn(
            1,
            "caller",
            true,
            1,
            vec![LocalSlot {
                id: 0,
                name: "v".into(),
                ty: IrType::Int(IntType::U8),
                mutable: false,
            }],
            IrType::Int(IntType::U8),
            vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::LoadLocal {
                        dst: 0,
                        slot: 0,
                        ty: IrType::Int(IntType::U8),
                    },
                    IrInst::Call {
                        dst: Some(1),
                        func: 0,
                        args: vec![0],
                    },
                    IrInst::ConstInt {
                        dst: 2,
                        ty: IrType::Int(IntType::U8),
                        val: 2,
                    },
                    IrInst::BinOp {
                        dst: 3,
                        op: IrBinOp::Add,
                        mode: ArithMode::Wrapping,
                        ty: IrType::Int(IntType::U8),
                        lhs: 1,
                        rhs: 2,
                        site_id: 0,
                    },
                ],
                term: Terminator::Return(Some(3)),
            }],
            4,
        ));

        let changed = inline_functions(&mut prog, &OptConfig::default());
        assert!(changed);
        assert!(prog.functions[0].blocks.is_empty(), "single-caller helper eliminated");
        assert!(
            !prog.functions[1]
                .blocks
                .iter()
                .any(|bb| bb.insts.iter().any(|i| matches!(i, IrInst::Call { .. }))),
            "call must be inlined into caller"
        );
        verify::verify_program(&prog).expect("MIR must verify after inlining");
    }

    #[test]
    fn preserves_public_tiny_leaf_and_inlines_tiny_private_leaf_at_multiple_sites() {
        let mut prog = IrProgram::default();
        // fn tiny(x: u8) -> u8 { x +% 1 }
        prog.functions.push(make_test_fn(
            0,
            "tiny",
            false,
            1,
            vec![LocalSlot {
                id: 0,
                name: "x".into(),
                ty: IrType::Int(IntType::U8),
                mutable: false,
            }],
            IrType::Int(IntType::U8),
            vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::LoadLocal {
                        dst: 0,
                        slot: 0,
                        ty: IrType::Int(IntType::U8),
                    },
                    IrInst::ConstInt {
                        dst: 1,
                        ty: IrType::Int(IntType::U8),
                        val: 1,
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
                ],
                term: Terminator::Return(Some(2)),
            }],
            3,
        ));
        // pub fn use2(a: u8, b: u8) -> u8 { tiny(a) +% tiny(b) }
        prog.functions.push(make_test_fn(
            1,
            "use2",
            true,
            2,
            vec![
                LocalSlot {
                    id: 0,
                    name: "a".into(),
                    ty: IrType::Int(IntType::U8),
                    mutable: false,
                },
                LocalSlot {
                    id: 1,
                    name: "b".into(),
                    ty: IrType::Int(IntType::U8),
                    mutable: false,
                },
            ],
            IrType::Int(IntType::U8),
            vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::LoadLocal {
                        dst: 0,
                        slot: 0,
                        ty: IrType::Int(IntType::U8),
                    },
                    IrInst::Call {
                        dst: Some(1),
                        func: 0,
                        args: vec![0],
                    },
                    IrInst::LoadLocal {
                        dst: 2,
                        slot: 1,
                        ty: IrType::Int(IntType::U8),
                    },
                    IrInst::Call {
                        dst: Some(3),
                        func: 0,
                        args: vec![2],
                    },
                    IrInst::BinOp {
                        dst: 4,
                        op: IrBinOp::Add,
                        mode: ArithMode::Wrapping,
                        ty: IrType::Int(IntType::U8),
                        lhs: 1,
                        rhs: 3,
                        site_id: 0,
                    },
                ],
                term: Terminator::Return(Some(4)),
            }],
            5,
        ));

        inline_functions(&mut prog, &OptConfig::default());
        assert!(prog.functions[0].blocks.is_empty());
        assert!(!prog.functions[1].blocks[0].insts.iter().any(|i| matches!(i, IrInst::Call { .. })));
        verify::verify_program(&prog).expect("MIR must verify after multi-site tiny inlining");
    }

    #[test]
    fn inlines_multi_caller_small_helper_in_loop_and_triggers_cse() {
        let mut prog = IrProgram::default();
        prog.globals.push(crate::IrGlobal {
            name: "ARR".into(),
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

        // fn helper(idx: u8) (> MAX_TINY_LEAF_INSTS = 6, ~11 insts):
        // ARR[idx as u16] = ARR[idx as u16] +% 1;
        prog.functions.push(make_test_fn(
            0,
            "helper",
            false,
            1,
            vec![LocalSlot {
                id: 0,
                name: "idx".into(),
                ty: IrType::Int(IntType::U8),
                mutable: false,
            }],
            IrType::Unit,
            vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::LoadLocal {
                        dst: 0,
                        slot: 0,
                        ty: IrType::Int(IntType::U8),
                    },
                    IrInst::Cast {
                        dst: 1,
                        val: 0,
                        from_ty: IrType::Int(IntType::U8),
                        to_ty: IrType::Int(IntType::U16),
                    },
                    IrInst::AddrGlobal { dst: 2, global: 0 },
                    IrInst::IndexAddr {
                        dst: 3,
                        base: 2,
                        index: 1,
                        elem_size: 2,
                        len: 8,
                        site_id: 0,
                    },
                    IrInst::LoadPtr {
                        dst: 4,
                        addr: 3,
                        ty: IrType::Int(IntType::U16),
                        volatile: false,
                    },
                    IrInst::ConstInt {
                        dst: 5,
                        ty: IrType::Int(IntType::U16),
                        val: 1,
                    },
                    IrInst::BinOp {
                        dst: 6,
                        op: IrBinOp::Add,
                        mode: ArithMode::Wrapping,
                        ty: IrType::Int(IntType::U16),
                        lhs: 4,
                        rhs: 5,
                        site_id: 0,
                    },
                    IrInst::StorePtr {
                        addr: 3,
                        val: 6,
                        ty: IrType::Int(IntType::U16),
                        volatile: false,
                    },
                ],
                term: Terminator::Return(None),
            }],
            7,
        ));

        // pub fn hot_loop(n: u8) -> u16: loops and calls helper(i) twice per iteration
        prog.functions.push(make_test_fn(
            1,
            "hot_loop",
            true,
            1,
            vec![
                LocalSlot {
                    id: 0,
                    name: "n".into(),
                    ty: IrType::Int(IntType::U8),
                    mutable: false,
                },
                LocalSlot {
                    id: 1,
                    name: "i".into(),
                    ty: IrType::Int(IntType::U8),
                    mutable: true,
                },
            ],
            IrType::Int(IntType::U16),
            vec![
                BasicBlock {
                    id: 0,
                    insts: vec![IrInst::ConstInt {
                        dst: 0,
                        ty: IrType::Int(IntType::U8),
                        val: 0,
                    }],
                    term: Terminator::Goto(1),
                },
                BasicBlock {
                    id: 1,
                    insts: vec![
                        IrInst::LoadLocal {
                            dst: 1,
                            slot: 1,
                            ty: IrType::Int(IntType::U8),
                        },
                        IrInst::LoadLocal {
                            dst: 2,
                            slot: 0,
                            ty: IrType::Int(IntType::U8),
                        },
                        IrInst::BinOp {
                            dst: 3,
                            op: IrBinOp::Lt,
                            mode: ArithMode::Wrapping,
                            ty: IrType::Int(IntType::U8),
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
                        IrInst::LoadLocal {
                            dst: 4,
                            slot: 1,
                            ty: IrType::Int(IntType::U8),
                        },
                        IrInst::Cast {
                            dst: 5,
                            val: 4,
                            from_ty: IrType::Int(IntType::U8),
                            to_ty: IrType::Int(IntType::U16),
                        },
                        IrInst::AddrGlobal { dst: 6, global: 0 },
                        IrInst::IndexAddr {
                            dst: 7,
                            base: 6,
                            index: 5,
                            elem_size: 2,
                            len: 8,
                            site_id: 0,
                        },
                        IrInst::LoadPtr {
                            dst: 8,
                            addr: 7,
                            ty: IrType::Int(IntType::U16),
                            volatile: false,
                        },
                        IrInst::Call {
                            dst: None,
                            func: 0,
                            args: vec![4],
                        },
                        IrInst::Call {
                            dst: None,
                            func: 0,
                            args: vec![4],
                        },
                        IrInst::ConstInt {
                            dst: 9,
                            ty: IrType::Int(IntType::U8),
                            val: 1,
                        },
                        IrInst::BinOp {
                            dst: 10,
                            op: IrBinOp::Add,
                            mode: ArithMode::Wrapping,
                            ty: IrType::Int(IntType::U8),
                            lhs: 4,
                            rhs: 9,
                            site_id: 0,
                        },
                        IrInst::StoreLocal {
                            slot: 1,
                            val: 10,
                            ty: IrType::Int(IntType::U8),
                        },
                    ],
                    term: Terminator::Goto(1),
                },
                BasicBlock {
                    id: 3,
                    insts: vec![IrInst::ConstInt {
                        dst: 11,
                        ty: IrType::Int(IntType::U16),
                        val: 0,
                    }],
                    term: Terminator::Return(Some(11)),
                },
            ],
            12,
        ));

        // pub fn cold_caller_a(x: u8) and cold_caller_b(x: u8): outside any loop, stay outlined!
        for (fid, name) in [(2u32, "cold_caller_a"), (3u32, "cold_caller_b")] {
            prog.functions.push(make_test_fn(
                fid,
                name,
                true,
                1,
                vec![LocalSlot {
                    id: 0,
                    name: "x".into(),
                    ty: IrType::Int(IntType::U8),
                    mutable: false,
                }],
                IrType::Unit,
                vec![BasicBlock {
                    id: 0,
                    insts: vec![
                        IrInst::LoadLocal {
                            dst: 0,
                            slot: 0,
                            ty: IrType::Int(IntType::U8),
                        },
                        IrInst::Call {
                            dst: None,
                            func: 0,
                            args: vec![0],
                        },
                    ],
                    term: Terminator::Return(None),
                }],
                1,
            ));
        }

        let changed = inline_functions(&mut prog, &OptConfig::default());
        assert!(changed);
        // hot_loop should have zero Calls left and CSE all 3 IndexAddrs into 1!
        let hot_calls: usize = prog.functions[1]
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|i| matches!(i, IrInst::Call { .. }))
            .count();
        assert_eq!(hot_calls, 0, "hot_loop calls to helper must inline");
        let hot_index_addrs: usize = prog.functions[1]
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|i| matches!(i, IrInst::IndexAddr { .. }))
            .count();
        assert_eq!(
            hot_index_addrs, 1,
            "repeated IndexAddr in hot_loop + inlined helper must CSE to 1"
        );
        // cold callers keep their calls, so helper remains defined.
        for cold_idx in [2usize, 3usize] {
            assert!(
                prog.functions[cold_idx]
                    .blocks
                    .iter()
                    .flat_map(|b| &b.insts)
                    .any(|i| matches!(i, IrInst::Call { func: 0, .. })),
                "cold caller {cold_idx} should keep call to multi-caller medium helper"
            );
        }
        assert!(!prog.functions[0].blocks.is_empty());
        verify::verify_program(&prog).expect("MIR must verify");
    }

    #[test]
    fn respects_inlining_invariants_in_hot_loops() {
        let mut prog = IrProgram::default();

        // 0: pub fn pub_leaf(x: u8) -> u8 { x +% 1 }
        let pub_leaf = make_test_fn(
            0,
            "pub_leaf",
            true,
            1,
            vec![LocalSlot {
                id: 0,
                name: "x".into(),
                ty: IrType::Int(IntType::U8),
                mutable: false,
            }],
            IrType::Int(IntType::U8),
            vec![BasicBlock {
                id: 0,
                insts: vec![IrInst::LoadLocal {
                    dst: 0,
                    slot: 0,
                    ty: IrType::Int(IntType::U8),
                }],
                term: Terminator::Return(Some(0)),
            }],
            1,
        );
        prog.functions.push(pub_leaf);

        // 1: #[reentrant] fn reent_leaf()
        let mut reent_leaf = make_test_fn(
            1,
            "reent_leaf",
            false,
            0,
            vec![],
            IrType::Unit,
            vec![BasicBlock {
                id: 0,
                insts: vec![],
                term: Terminator::Return(None),
            }],
            0,
        );
        reent_leaf.is_reentrant = true;
        prog.functions.push(reent_leaf);

        // 2: #[requires(di)] fn di_leaf()
        let mut di_leaf = make_test_fn(
            2,
            "di_leaf",
            false,
            0,
            vec![],
            IrType::Unit,
            vec![BasicBlock {
                id: 0,
                insts: vec![],
                term: Terminator::Return(None),
            }],
            0,
        );
        di_leaf.requires_di = true;
        prog.functions.push(di_leaf);

        // 3: #[overlay(M::S)] fn ovl_leaf()
        let mut ovl_leaf = make_test_fn(
            3,
            "ovl_leaf",
            false,
            0,
            vec![],
            IrType::Unit,
            vec![BasicBlock {
                id: 0,
                insts: vec![],
                term: Terminator::Return(None),
            }],
            0,
        );
        ovl_leaf.owner_state = Some((0, 0));
        prog.functions.push(ovl_leaf);

        // 4, 5: mutually recursive small functions ping <-> pong (must not infinitely expand)
        let ping = make_test_fn(
            4,
            "ping",
            false,
            0,
            vec![],
            IrType::Unit,
            vec![BasicBlock {
                id: 0,
                insts: vec![IrInst::Call {
                    dst: None,
                    func: 5,
                    args: vec![],
                }],
                term: Terminator::Return(None),
            }],
            0,
        );
        let pong = make_test_fn(
            5,
            "pong",
            false,
            0,
            vec![],
            IrType::Unit,
            vec![BasicBlock {
                id: 0,
                insts: vec![IrInst::Call {
                    dst: None,
                    func: 4,
                    args: vec![],
                }],
                term: Terminator::Return(None),
            }],
            0,
        );
        prog.functions.push(ping);
        prog.functions.push(pong);

        // 6: pub fn driver() with loop calling all of 0..=4
        prog.functions.push(make_test_fn(
            6,
            "driver",
            true,
            0,
            vec![],
            IrType::Unit,
            vec![
                BasicBlock {
                    id: 0,
                    insts: vec![],
                    term: Terminator::Goto(1),
                },
                BasicBlock {
                    id: 1,
                    insts: vec![
                        IrInst::ConstInt {
                            dst: 0,
                            ty: IrType::Int(IntType::U8),
                            val: 1,
                        },
                        IrInst::Call {
                            dst: Some(1),
                            func: 0,
                            args: vec![0],
                        },
                        IrInst::Call {
                            dst: None,
                            func: 1,
                            args: vec![],
                        },
                        IrInst::Call {
                            dst: None,
                            func: 2,
                            args: vec![],
                        },
                        IrInst::Call {
                            dst: None,
                            func: 3,
                            args: vec![],
                        },
                        IrInst::Call {
                            dst: None,
                            func: 4,
                            args: vec![],
                        },
                        IrInst::ConstBool { dst: 2, val: true },
                    ],
                    term: Terminator::Branch {
                        cond: 2,
                        then_bb: 1,
                        else_bb: 2,
                    },
                },
                BasicBlock {
                    id: 2,
                    insts: vec![],
                    term: Terminator::Return(None),
                },
            ],
            3,
        ));

        let changed = inline_functions(&mut prog, &OptConfig::default());
        assert!(!changed, "protected functions and cycles must not be inlined");
        for fid in 0..=5 {
            assert!(!prog.functions[fid].blocks.is_empty());
        }
        verify::verify_program(&prog).expect("MIR must verify");
    }

    #[test]
    fn main_event_loop_does_not_mark_cold_callees_loop_hot() {
        let mut prog = IrProgram::default();
        // Function 0: cold helper called from main's infinite outer event loop.
        let cold_fn = make_test_fn(
            0,
            "ui_screen",
            false,
            0,
            vec![],
            IrType::Unit,
            vec![BasicBlock {
                id: 0,
                insts: vec![],
                term: Terminator::Return(None),
            }],
            0,
        );
        prog.functions.push(cold_fn);

        // Function 1: inner compute helper called from an inner loop inside main.
        let inner_fn = make_test_fn(
            1,
            "inner_compute",
            false,
            0,
            vec![],
            IrType::Unit,
            vec![BasicBlock {
                id: 0,
                insts: vec![],
                term: Terminator::Return(None),
            }],
            0,
        );
        prog.functions.push(inner_fn);

        // Function 2: main() with outer infinite loop { call ui_screen; inner bounded loop { call inner_compute } }.
        let main_fn = make_test_fn(
            2,
            "main",
            true,
            0,
            vec![],
            IrType::Unit,
            vec![
                BasicBlock {
                    id: 0,
                    insts: vec![],
                    term: Terminator::Goto(1),
                },
                BasicBlock {
                    id: 1,
                    insts: vec![IrInst::Call {
                        dst: None,
                        func: 0,
                        args: vec![],
                    }],
                    term: Terminator::Goto(2),
                },
                BasicBlock {
                    id: 2,
                    insts: vec![
                        IrInst::Call {
                            dst: None,
                            func: 1,
                            args: vec![],
                        },
                        IrInst::ConstBool { dst: 0, val: true },
                    ],
                    term: Terminator::Branch {
                        cond: 0,
                        then_bb: 2,
                        else_bb: 1,
                    },
                },
            ],
            1,
        );
        prog.functions.push(main_fn);

        let info = compute_loop_hot_info(&prog);
        assert!(
            !info.loop_hot_fns[0],
            "cold callee called from main's infinite outer loop must NOT be loop_hot"
        );
        assert!(
            info.loop_hot_fns[1],
            "inner bounded compute loop callee inside main must STILL be loop_hot"
        );
    }

    #[test]
    fn inlines_multi_caller_phi_helper_in_loop_and_verifies() {
        let mut prog = IrProgram::default();

        // 0: fn clamp_hi(x: u8) -> u8 {
        //   if x > 10 { return 10 +% 1; } else { return x +% 1; }
        //   (expressed with Phi referencing callee_entry bb0 directly and bb1)
        // }
        // Size > MAX_TINY_LEAF_INSTS (6) and <= MAX_HOT_HELPER_INSTS (40).
        prog.functions.push(make_test_fn(
            0,
            "clamp_hi",
            false,
            1,
            vec![LocalSlot {
                id: 0,
                name: "x".into(),
                ty: IrType::Int(IntType::U8),
                mutable: false,
            }],
            IrType::Int(IntType::U8),
            vec![
                BasicBlock {
                    id: 0,
                    insts: vec![
                        IrInst::LoadLocal {
                            dst: 0,
                            slot: 0,
                            ty: IrType::Int(IntType::U8),
                        },
                        IrInst::ConstInt {
                            dst: 1,
                            ty: IrType::Int(IntType::U8),
                            val: 10,
                        },
                        IrInst::BinOp {
                            dst: 2,
                            op: IrBinOp::Gt,
                            mode: ArithMode::Wrapping,
                            ty: IrType::Int(IntType::U8),
                            lhs: 0,
                            rhs: 1,
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
                    insts: vec![IrInst::ConstInt {
                        dst: 3,
                        ty: IrType::Int(IntType::U8),
                        val: 10,
                    }],
                    term: Terminator::Goto(2),
                },
                BasicBlock {
                    id: 2,
                    insts: vec![
                        IrInst::Phi {
                            dst: 4,
                            ty: IrType::Int(IntType::U8),
                            incoming: vec![(0, 0), (1, 3)],
                        },
                        IrInst::ConstInt {
                            dst: 5,
                            ty: IrType::Int(IntType::U8),
                            val: 1,
                        },
                        IrInst::BinOp {
                            dst: 6,
                            op: IrBinOp::Add,
                            mode: ArithMode::Wrapping,
                            ty: IrType::Int(IntType::U8),
                            lhs: 4,
                            rhs: 5,
                            site_id: 1,
                        },
                        IrInst::BinOp {
                            dst: 7,
                            op: IrBinOp::BitXor,
                            mode: ArithMode::Wrapping,
                            ty: IrType::Int(IntType::U8),
                            lhs: 6,
                            rhs: 5,
                            site_id: 2,
                        },
                    ],
                    term: Terminator::Return(Some(7)),
                },
            ],
            8,
        ));

        // 1: pub fn hot_loop(n: u8) -> u8 with SSA loop Phi and two calls to clamp_hi(iv)
        prog.functions.push(make_test_fn(
            1,
            "hot_loop",
            true,
            1,
            vec![LocalSlot {
                id: 0,
                name: "n".into(),
                ty: IrType::Int(IntType::U8),
                mutable: false,
            }],
            IrType::Int(IntType::U8),
            vec![
                BasicBlock {
                    id: 0,
                    insts: vec![
                        IrInst::ConstInt {
                            dst: 0,
                            ty: IrType::Int(IntType::U8),
                            val: 0,
                        },
                        IrInst::LoadLocal {
                            dst: 1,
                            slot: 0,
                            ty: IrType::Int(IntType::U8),
                        },
                    ],
                    term: Terminator::Goto(1),
                },
                BasicBlock {
                    id: 1,
                    insts: vec![
                        IrInst::Phi {
                            dst: 2,
                            ty: IrType::Int(IntType::U8),
                            incoming: vec![(0, 0), (2, 7)],
                        },
                        IrInst::Phi {
                            dst: 3,
                            ty: IrType::Int(IntType::U8),
                            incoming: vec![(0, 0), (2, 8)],
                        },
                        IrInst::BinOp {
                            dst: 4,
                            op: IrBinOp::Lt,
                            mode: ArithMode::Wrapping,
                            ty: IrType::Int(IntType::U8),
                            lhs: 2,
                            rhs: 1,
                            site_id: 0,
                        },
                    ],
                    term: Terminator::Branch {
                        cond: 4,
                        then_bb: 2,
                        else_bb: 3,
                    },
                },
                BasicBlock {
                    id: 2,
                    insts: vec![
                        IrInst::Call {
                            dst: Some(5),
                            func: 0,
                            args: vec![2],
                        },
                        IrInst::ConstInt {
                            dst: 6,
                            ty: IrType::Int(IntType::U8),
                            val: 1,
                        },
                        IrInst::BinOp {
                            dst: 7,
                            op: IrBinOp::Add,
                            mode: ArithMode::Wrapping,
                            ty: IrType::Int(IntType::U8),
                            lhs: 2,
                            rhs: 6,
                            site_id: 1,
                        },
                        IrInst::BinOp {
                            dst: 8,
                            op: IrBinOp::Add,
                            mode: ArithMode::Wrapping,
                            ty: IrType::Int(IntType::U8),
                            lhs: 3,
                            rhs: 5,
                            site_id: 2,
                        },
                    ],
                    term: Terminator::Goto(1),
                },
                BasicBlock {
                    id: 3,
                    insts: vec![],
                    term: Terminator::Return(Some(3)),
                },
            ],
            9,
        ));

        // 2, 3: cold callers outside loops keep clamp_hi multi-caller
        for (fid, name) in [(2u32, "cold_a"), (3u32, "cold_b")] {
            prog.functions.push(make_test_fn(
                fid,
                name,
                true,
                1,
                vec![LocalSlot {
                    id: 0,
                    name: "x".into(),
                    ty: IrType::Int(IntType::U8),
                    mutable: false,
                }],
                IrType::Int(IntType::U8),
                vec![BasicBlock {
                    id: 0,
                    insts: vec![
                        IrInst::LoadLocal {
                            dst: 0,
                            slot: 0,
                            ty: IrType::Int(IntType::U8),
                        },
                        IrInst::Call {
                            dst: Some(1),
                            func: 0,
                            args: vec![0],
                        },
                    ],
                    term: Terminator::Return(Some(1)),
                }],
                2,
            ));
        }

        // First test direct inline_one_call_site before destruct_ssa to verify raw Phi remapping
        let mut direct_prog = prog.clone();
        let callee_snap = direct_prog.functions[0].clone();
        inline_one_call_site(&mut direct_prog.functions[1], &callee_snap, 2, 0);
        verify::verify_program(&direct_prog)
            .expect("MIR with both caller and callee Phis must verify immediately after inline_one_call_site");

        let changed = inline_functions(&mut prog, &OptConfig::default());
        assert!(changed, "Phi-bearing hot helper must be inlined");
        let hot_calls: usize = prog.functions[1]
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|i| matches!(i, IrInst::Call { .. }))
            .count();
        assert_eq!(hot_calls, 0, "hot_loop call to Phi-bearing helper must inline");
        for cold_idx in [2usize, 3usize] {
            assert!(
                prog.functions[cold_idx]
                    .blocks
                    .iter()
                    .flat_map(|b| &b.insts)
                    .any(|i| matches!(i, IrInst::Call { func: 0, .. })),
                "cold caller {cold_idx} should keep call to multi-caller helper"
            );
        }
        verify::verify_program(&prog).expect("MIR must verify after Phi inlining");
    }

    #[test]
    fn selective_hot_caller_inlines_nine_site_helper_and_large_hot_callees_without_deleting_callee() {
        let u8t = IrType::Int(IntType::U8);
        let u16t = IrType::Int(IntType::U16);
        let mut prog = IrProgram::default();
        prog.globals.push(crate::IrGlobal {
            name: "G_BALLS".into(),
            ty: IrType::Array {
                elem: Box::new(u16t.clone()),
                len: 4,
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

        let make_indexed_fn = |id: u32, name: &str, extra_insts: usize| {
            let mut insts = vec![
                IrInst::LoadLocal {
                    dst: 0,
                    slot: 0,
                    ty: u8t.clone(),
                },
                IrInst::Cast {
                    dst: 1,
                    val: 0,
                    from_ty: u8t.clone(),
                    to_ty: u16t.clone(),
                },
                IrInst::AddrGlobal { dst: 2, global: 0 },
                IrInst::IndexAddr {
                    dst: 3,
                    base: 2,
                    index: 1,
                    elem_size: 2,
                    len: 4,
                    site_id: 0,
                },
                IrInst::LoadPtr {
                    dst: 4,
                    addr: 3,
                    ty: u16t.clone(),
                    volatile: false,
                },
            ];
            let mut cur_val = 4u32;
            let mut next_v = 5u32;
            for k in 0..extra_insts {
                let c = next_v;
                let sum = next_v + 1;
                next_v += 2;
                insts.push(IrInst::ConstInt {
                    dst: c,
                    ty: u16t.clone(),
                    val: (k as i64) + 1,
                });
                insts.push(IrInst::BinOp {
                    dst: sum,
                    op: IrBinOp::Add,
                    mode: ArithMode::Wrapping,
                    ty: u16t.clone(),
                    lhs: cur_val,
                    rhs: c,
                    site_id: 0,
                });
                cur_val = sum;
            }
            insts.push(IrInst::StorePtr {
                addr: 3,
                val: cur_val,
                ty: u16t.clone(),
                volatile: false,
            });
            make_test_fn(
                id,
                name,
                false,
                1,
                vec![LocalSlot {
                    id: 0,
                    name: "bi".into(),
                    ty: u8t.clone(),
                    mutable: false,
                }],
                IrType::Unit,
                vec![BasicBlock {
                    id: 0,
                    insts,
                    term: Terminator::Return(None),
                }],
                next_v,
            )
        };

        // 0: sync_ball_px (~25 insts, > MAX_TINY_LEAF_INSTS, <= MAX_HOT_HELPER_INSTS):
        // 9 total call sites (4 hot + 5 cold) would exceed MAX_HOT_HELPER_TOTAL_DUP_INSTS if cold sites counted.
        prog.functions.push(make_indexed_fn(0, "sync_ball_px", 12));
        // 1: guard_right_divider (~47 insts, > 40, <= MAX_HOT_HELPER_INSTS): 2 hot call sites.
        prog.functions.push(make_indexed_fn(1, "guard_right_divider", 20));
        // 2: clamp_ball_speed (~127 insts, > 96, <= MAX_HOT_SINGLE_CALLER_INSTS): 1 hot call site.
        prog.functions.push(make_indexed_fn(2, "clamp_ball_speed", 60));

        // 3: step_ball_substep (~210+ insts): calls guard_right_divider, clamp_ball_speed, sync_ball_px x2.
        let mut step_insts = vec![IrInst::LoadLocal {
            dst: 0,
            slot: 0,
            ty: u8t.clone(),
        }];
        let mut step_v = 1u32;
        for _ in 0..105 {
            let c1 = step_v;
            let c2 = step_v + 1;
            step_v += 2;
            step_insts.push(IrInst::ConstInt {
                dst: c1,
                ty: u16t.clone(),
                val: 1,
            });
            step_insts.push(IrInst::BinOp {
                dst: c2,
                op: IrBinOp::Add,
                mode: ArithMode::Wrapping,
                ty: u16t.clone(),
                lhs: c1,
                rhs: c1,
                site_id: 0,
            });
            // Keep side effect via StoreLocal? Use AddrGlobal + StorePtr so DCE doesn't remove pre-pass.
            step_insts.push(IrInst::AddrGlobal {
                dst: step_v,
                global: 0,
            });
            step_insts.push(IrInst::StorePtr {
                addr: step_v,
                val: c2,
                ty: u16t.clone(),
                volatile: true,
            });
            step_v += 1;
        }
        step_insts.push(IrInst::Call {
            dst: None,
            func: 1,
            args: vec![0],
        });
        step_insts.push(IrInst::Call {
            dst: None,
            func: 2,
            args: vec![0],
        });
        step_insts.push(IrInst::Call {
            dst: None,
            func: 0,
            args: vec![0],
        });
        step_insts.push(IrInst::Call {
            dst: None,
            func: 0,
            args: vec![0],
        });
        prog.functions.push(make_test_fn(
            3,
            "step_ball_substep",
            false,
            1,
            vec![LocalSlot {
                id: 0,
                name: "bi".into(),
                ty: u8t.clone(),
                mutable: false,
            }],
            IrType::Unit,
            vec![BasicBlock {
                id: 0,
                insts: step_insts,
                term: Terminator::Return(None),
            }],
            step_v,
        ));

        // 4: collide_walls: calls guard_right_divider + sync_ball_px.
        prog.functions.push(make_test_fn(
            4,
            "collide_walls",
            false,
            1,
            vec![LocalSlot {
                id: 0,
                name: "bi".into(),
                ty: u8t.clone(),
                mutable: false,
            }],
            IrType::Unit,
            vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::LoadLocal {
                        dst: 0,
                        slot: 0,
                        ty: u8t.clone(),
                    },
                    IrInst::Call {
                        dst: None,
                        func: 1,
                        args: vec![0],
                    },
                    IrInst::Call {
                        dst: None,
                        func: 0,
                        args: vec![0],
                    },
                ],
                term: Terminator::Return(None),
            }],
            1,
        ));

        // 5: collide_flipper_one: calls sync_ball_px.
        prog.functions.push(make_test_fn(
            5,
            "collide_flipper_one",
            false,
            1,
            vec![LocalSlot {
                id: 0,
                name: "bi".into(),
                ty: u8t.clone(),
                mutable: false,
            }],
            IrType::Unit,
            vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::LoadLocal {
                        dst: 0,
                        slot: 0,
                        ty: u8t.clone(),
                    },
                    IrInst::Call {
                        dst: None,
                        func: 0,
                        args: vec![0],
                    },
                ],
                term: Terminator::Return(None),
            }],
            1,
        ));

        // 6: pub fn physics_step_frame(): loop calling step_ball_substep, collide_walls, collide_flipper_one.
        prog.functions.push(make_test_fn(
            6,
            "physics_step_frame",
            true,
            0,
            vec![LocalSlot {
                id: 0,
                name: "bi".into(),
                ty: u8t.clone(),
                mutable: true,
            }],
            IrType::Unit,
            vec![
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
                            val: 3,
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
                        IrInst::LoadLocal {
                            dst: 4,
                            slot: 0,
                            ty: u8t.clone(),
                        },
                        IrInst::Call {
                            dst: None,
                            func: 3,
                            args: vec![4],
                        },
                        IrInst::Call {
                            dst: None,
                            func: 4,
                            args: vec![4],
                        },
                        IrInst::Call {
                            dst: None,
                            func: 5,
                            args: vec![4],
                        },
                        IrInst::ConstInt {
                            dst: 5,
                            ty: u8t.clone(),
                            val: 1,
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
                        IrInst::StoreLocal {
                            slot: 0,
                            val: 6,
                            ty: u8t.clone(),
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
            7,
        ));

        // 7: pub fn physics_init_game(): cold caller calling sync_ball_px 3 times outside loops.
        // 8: pub fn physics_spawn_ball(): cold caller calling sync_ball_px 2 times outside loops.
        for (fid, name, n_calls) in [(7u32, "physics_init_game", 3usize), (8u32, "physics_spawn_ball", 2usize)] {
            let mut insts = vec![IrInst::ConstInt {
                dst: 0,
                ty: u8t.clone(),
                val: 0,
            }];
            for _ in 0..n_calls {
                insts.push(IrInst::Call {
                    dst: None,
                    func: 0,
                    args: vec![0],
                });
            }
            prog.functions.push(make_test_fn(
                fid,
                name,
                true,
                0,
                vec![],
                IrType::Unit,
                vec![BasicBlock {
                    id: 0,
                    insts,
                    term: Terminator::Return(None),
                }],
                1,
            ));
        }

        let changed = inline_functions(&mut prog, &OptConfig::default());
        assert!(changed);
        verify::verify_program(&prog).expect("MIR must verify after selective hot inlining");

        let count_calls_to = |fn_idx: usize, target: u32| -> usize {
            prog.functions[fn_idx]
                .blocks
                .iter()
                .flat_map(|b| &b.insts)
                .filter(|i| matches!(i, IrInst::Call { func, .. } if *func == target))
                .count()
        };

        // Hot callers must inline sync_ball_px, guard_right_divider, and clamp_ball_speed.
        for hot_idx in [3usize, 4usize, 5usize] {
            for helper_id in [0u32, 1u32, 2u32] {
                assert_eq!(
                    count_calls_to(hot_idx, helper_id),
                    0,
                    "hot fn {} must inline helper {helper_id}",
                    prog.functions[hot_idx].name
                );
            }
        }
        // Cold callers keep their out-of-line calls to sync_ball_px, and sync_ball_px remains defined.
        assert_eq!(count_calls_to(7, 0), 3, "cold physics_init_game keeps 3 calls");
        assert_eq!(count_calls_to(8, 0), 2, "cold physics_spawn_ball keeps 2 calls");
        assert!(
            !prog.functions[0].blocks.is_empty(),
            "sync_ball_px must remain in program.functions while cold callers exist"
        );
        assert!(
            prog.functions[1].blocks.is_empty() && prog.functions[2].blocks.is_empty(),
            "guard_right_divider and clamp_ball_speed have no remaining callers and must be eliminated"
        );
    }

    #[test]
    fn prioritizes_small_hot_leaves_before_large_helpers_and_caps_multisite_medium_bloat() {
        let u8t = IrType::Int(IntType::U8);
        let u16t = IrType::Int(IntType::U16);
        let mut prog = IrProgram::default();
        prog.globals.push(crate::IrGlobal {
            name: "G_BALLS".into(),
            ty: IrType::Array {
                elem: Box::new(u16t.clone()),
                len: 4,
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

        let make_indexed_leaf = |id: u32, name: &str, extra_insts: usize| {
            let mut insts = vec![
                IrInst::LoadLocal {
                    dst: 0,
                    slot: 0,
                    ty: u8t.clone(),
                },
                IrInst::Cast {
                    dst: 1,
                    val: 0,
                    from_ty: u8t.clone(),
                    to_ty: u16t.clone(),
                },
                IrInst::AddrGlobal { dst: 2, global: 0 },
                IrInst::IndexAddr {
                    dst: 3,
                    base: 2,
                    index: 1,
                    elem_size: 2,
                    len: 4,
                    site_id: 0,
                },
                IrInst::LoadPtr {
                    dst: 4,
                    addr: 3,
                    ty: u16t.clone(),
                    volatile: false,
                },
            ];
            let mut cur_val = 4u32;
            let mut next_v = 5u32;
            for k in 0..extra_insts {
                let c = next_v;
                let sum = next_v + 1;
                next_v += 2;
                insts.push(IrInst::ConstInt {
                    dst: c,
                    ty: u16t.clone(),
                    val: (k as i64) + 1,
                });
                insts.push(IrInst::BinOp {
                    dst: sum,
                    op: IrBinOp::Add,
                    mode: ArithMode::Wrapping,
                    ty: u16t.clone(),
                    lhs: cur_val,
                    rhs: c,
                    site_id: 0,
                });
                cur_val = sum;
            }
            insts.push(IrInst::StorePtr {
                addr: 3,
                val: cur_val,
                ty: u16t.clone(),
                volatile: false,
            });
            make_test_fn(
                id,
                name,
                false,
                1,
                vec![LocalSlot {
                    id: 0,
                    name: "bi".into(),
                    ty: u8t.clone(),
                    mutable: false,
                }],
                IrType::Unit,
                vec![BasicBlock {
                    id: 0,
                    insts,
                    term: Terminator::Return(None),
                }],
                next_v,
            )
        };

        // 0: sync_ball_px (25 insts, <= 36): hot small leaf called AFTER larger helpers.
        prog.functions.push(make_indexed_leaf(0, "sync_ball_px", 9));
        // 1: guard_right_divider (35 insts, <= 36): hot small leaf called AFTER larger helpers.
        prog.functions.push(make_indexed_leaf(1, "guard_right_divider", 14));
        // 2: medium_fit_leaf (65 insts, <= 72): larger than small leaves; tight caller budget must
        // inline #0 and #1 BEFORE #2.
        prog.functions.push(make_indexed_leaf(2, "medium_fit_leaf", 29));
        // 3: push_and_reflect (77 insts, > MAX_HOT_LEAF_HELPER_INSTS=72): 5 hot sites, must stay out-of-line.
        prog.functions.push(make_indexed_leaf(3, "push_and_reflect", 35));
        // 4: collide_defenses (179 insts, > MAX_HOT_SINGLE_CALLER_INSTS=140): 1 hot site, must stay out-of-line.
        prog.functions.push(make_indexed_leaf(4, "collide_defenses", 86));

        // 5: hot step_ball_substep (~735 insts before inlining) calls large/medium helpers FIRST,
        // then small leaves LAST. Budget (MAX_HOT_CALLER_INSTS_AFTER_INLINE = 800) only fits #0 + #1
        // (~795), NOT #2 (45 insts -> would starve #0 or #1 if order wasn't sorted by size).
        let mut step_insts = vec![
            IrInst::LoadLocal {
                dst: 0,
                slot: 0,
                ty: u8t.clone(),
            },
            IrInst::AddrGlobal { dst: 1, global: 0 },
            IrInst::ConstInt {
                dst: 2,
                ty: u16t.clone(),
                val: 1,
            },
        ];
        let mut step_v = 3u32;
        for _ in 0..241 {
            let lv = step_v;
            let av = step_v + 1;
            step_v += 2;
            step_insts.push(IrInst::LoadPtr {
                dst: lv,
                addr: 1,
                ty: u16t.clone(),
                volatile: true,
            });
            step_insts.push(IrInst::BinOp {
                dst: av,
                op: IrBinOp::Add,
                mode: ArithMode::Wrapping,
                ty: u16t.clone(),
                lhs: lv,
                rhs: 2,
                site_id: 0,
            });
            step_insts.push(IrInst::StorePtr {
                addr: 1,
                val: av,
                ty: u16t.clone(),
                volatile: true,
            });
        }
        // Order in IR: collide_defenses (4), push_and_reflect (3), medium_fit_leaf (2),
        // guard_right_divider (1), sync_ball_px (0).
        for callee_id in [4u32, 3, 2, 1, 0] {
            step_insts.push(IrInst::Call {
                dst: None,
                func: callee_id,
                args: vec![0],
            });
        }
        prog.functions.push(make_test_fn(
            5,
            "step_ball_substep",
            false,
            1,
            vec![LocalSlot {
                id: 0,
                name: "bi".into(),
                ty: u8t.clone(),
                mutable: false,
            }],
            IrType::Unit,
            vec![BasicBlock {
                id: 0,
                insts: step_insts,
                term: Terminator::Return(None),
            }],
            step_v,
        ));

        // 6: second hot helper calling push_and_reflect 4 times + medium_fit_leaf + guard_right_divider + sync_ball_px.
        prog.functions.push(make_test_fn(
            6,
            "collide_walls",
            false,
            1,
            vec![LocalSlot {
                id: 0,
                name: "bi".into(),
                ty: u8t.clone(),
                mutable: false,
            }],
            IrType::Unit,
            vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::LoadLocal {
                        dst: 0,
                        slot: 0,
                        ty: u8t.clone(),
                    },
                    IrInst::Call {
                        dst: None,
                        func: 3,
                        args: vec![0],
                    },
                    IrInst::Call {
                        dst: None,
                        func: 3,
                        args: vec![0],
                    },
                    IrInst::Call {
                        dst: None,
                        func: 3,
                        args: vec![0],
                    },
                    IrInst::Call {
                        dst: None,
                        func: 3,
                        args: vec![0],
                    },
                    IrInst::Call {
                        dst: None,
                        func: 2,
                        args: vec![0],
                    },
                    IrInst::Call {
                        dst: None,
                        func: 1,
                        args: vec![0],
                    },
                    IrInst::Call {
                        dst: None,
                        func: 0,
                        args: vec![0],
                    },
                ],
                term: Terminator::Return(None),
            }],
            1,
        ));

        // 7: pub fn physics_step_frame() loop calling #5 and #6 so both are loop_hot.
        prog.functions.push(make_test_fn(
            7,
            "physics_step_frame",
            true,
            0,
            vec![LocalSlot {
                id: 0,
                name: "bi".into(),
                ty: u8t.clone(),
                mutable: true,
            }],
            IrType::Unit,
            vec![
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
                            val: 3,
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
                        IrInst::LoadLocal {
                            dst: 4,
                            slot: 0,
                            ty: u8t.clone(),
                        },
                        IrInst::Call {
                            dst: None,
                            func: 5,
                            args: vec![4],
                        },
                        IrInst::Call {
                            dst: None,
                            func: 6,
                            args: vec![4],
                        },
                        IrInst::ConstInt {
                            dst: 5,
                            ty: u8t.clone(),
                            val: 1,
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
                        IrInst::StoreLocal {
                            slot: 0,
                            val: 6,
                            ty: u8t.clone(),
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
            7,
        ));

        // 8: cold caller keeps medium_fit_leaf multi-caller so it remains subject to caller_limit.
        prog.functions.push(make_test_fn(
            8,
            "cold_medium_user",
            true,
            0,
            vec![],
            IrType::Unit,
            vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::ConstInt {
                        dst: 0,
                        ty: u8t.clone(),
                        val: 0,
                    },
                    IrInst::Call {
                        dst: None,
                        func: 2,
                        args: vec![0],
                    },
                ],
                term: Terminator::Return(None),
            }],
            1,
        ));

        let changed = inline_functions(&mut prog, &OptConfig::default());
        assert!(changed);
        verify::verify_program(&prog).expect("MIR must verify after prioritized inlining");

        let count_calls_to = |fn_idx: usize, target: u32| -> usize {
            prog.functions[fn_idx]
                .blocks
                .iter()
                .flat_map(|b| &b.insts)
                .filter(|i| matches!(i, IrInst::Call { func, .. } if *func == target))
                .count()
        };

        // step_ball_substep must prioritize sync_ball_px (0) and guard_right_divider (1),
        // leaving medium_fit_leaf (2), push_and_reflect (3), and collide_defenses (4) out-of-line.
        assert_eq!(count_calls_to(5, 0), 0, "sync_ball_px must inline first");
        assert_eq!(
            count_calls_to(5, 1),
            0,
            "guard_right_divider must inline before medium_fit_leaf"
        );
        assert_eq!(
            count_calls_to(5, 2),
            1,
            "medium_fit_leaf must stay out-of-line once caller budget is nearly full"
        );
        assert_eq!(
            count_calls_to(5, 3),
            1,
            "5-site medium push_and_reflect must stay out-of-line"
        );
        assert_eq!(
            count_calls_to(5, 4),
            1,
            "oversize single-caller collide_defenses must stay out-of-line"
        );
        assert!(!prog.functions[3].blocks.is_empty());
        assert!(!prog.functions[4].blocks.is_empty());
    }
}
