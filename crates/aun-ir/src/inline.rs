//! Target-independent MIR function inliner (`aun-ir`).
//!
//! Inlines non-public single-caller helpers (up to moderate size) and tiny leaf
//! helpers into callers before final target-specific optimization, then clears
//! dead non-public callees so backends reserve zero RAM/ROM for them.

use crate::{
    BlockId, IrFnTarget, IrFunction, IrInst, IrProgram, IrType, LocalSlot, SlotId, Terminator,
    ValueId,
    opt::{self, OptConfig},
};
use std::collections::{HashMap, HashSet};

const MAX_SINGLE_CALLER_INSTS: usize = 50;
const MAX_TINY_LEAF_INSTS: usize = 6;
const MAX_REF_MISMATCH_INSTS: usize = 20;
const MAX_CALLER_INSTS_AFTER_INLINE: usize = 600;
const MAX_INLINE_ROUNDS: usize = 3;

fn fn_inst_cost(f: &IrFunction) -> usize {
    f.blocks.iter().map(|b| b.insts.len() + 1).sum()
}

fn has_inline_asm_or_phi(f: &IrFunction) -> bool {
    f.blocks.iter().any(|bb| {
        bb.insts
            .iter()
            .any(|inst| matches!(inst, IrInst::InlineAsm { .. } | IrInst::Phi { .. }))
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

fn should_inline_site(
    caller: &IrFunction,
    callee: &IrFunction,
    args: &[ValueId],
    call_count: usize,
    addr_taken: &HashSet<u32>,
    machine_fns: &HashSet<u32>,
    caller_ref_vals: &HashSet<ValueId>,
) -> bool {
    if callee.id == caller.id
        || callee.blocks.is_empty()
        || callee.is_pub
        || callee.is_test
        || caller.is_test
        || callee.interrupt_source.is_some()
        || callee.overlay_state.is_some()
        || callee.is_reentrant
        || !callee.requires_banks.is_empty()
        || callee.requires_di
        || addr_taken.contains(&callee.id)
        || machine_fns.contains(&callee.id)
        || args.len() != callee.param_count
        || caller.locals.len() < caller.param_count
        || callee.locals.len() < callee.param_count
        || has_inline_asm_or_phi(callee)
    {
        return false;
    }

    let callee_cost = fn_inst_cost(callee);
    let caller_cost = fn_inst_cost(caller);
    let is_single_caller = call_count == 1 && callee_cost <= MAX_SINGLE_CALLER_INSTS;
    let is_tiny_leaf = call_count >= 1 && is_leaf_fn(callee) && callee_cost <= MAX_TINY_LEAF_INSTS;
    if !is_single_caller && !is_tiny_leaf {
        return false;
    }
    if !is_single_caller && caller_cost + callee_cost > MAX_CALLER_INSTS_AFTER_INLINE {
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
    alloc_val: &mut dyn FnMut() -> ValueId,
    val_map_mut: &mut HashMap<ValueId, ValueId>,
) -> IrInst {
    let mut cloned = inst.clone();

    // First remap uses in-place
    cloned.for_each_use_mut(|v| {
        if let Some(&mapped) = val_map_mut.get(v) {
            *v = mapped;
        } else if let Some(&mapped) = val_map.get(v) {
            *v = mapped;
        }
    });

    // Remap slot IDs
    match &mut cloned {
        IrInst::LoadLocal { slot, .. }
        | IrInst::StoreLocal { slot, .. }
        | IrInst::AddrLocal { slot, .. } => {
            if let Some(&new_slot) = slot_map.get(slot) {
                *slot = new_slot;
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

fn inline_one_call_site(
    caller: &mut IrFunction,
    callee: &IrFunction,
    bb_idx: usize,
    call_idx: usize,
) {
    let (dst_opt, args) = match &caller.blocks[bb_idx].insts[call_idx] {
        IrInst::Call { dst, args, .. } => (*dst, args.clone()),
        _ => return,
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
    let mut tail_insts = caller.blocks[bb_idx].insts.split_off(call_idx + 1);
    caller.blocks[bb_idx].insts.pop(); // remove the Call itself
    let orig_term = std::mem::replace(
        &mut caller.blocks[bb_idx].term,
        Terminator::Goto(callee_entry_bb),
    );
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

    caller.next_val = next_val;
    caller.blocks.extend(new_blocks);
}

/// Runs bounded single-caller and tiny-leaf inlining over `program`, then cleans up
/// modified callers and eliminates unreferenced internal callees.
pub fn inline_functions(program: &mut IrProgram, cfg: &OptConfig) -> bool {
    let (addr_taken, machine_fns) = collect_addr_taken_and_machine_fns(program);
    let isr_roots: HashSet<u32> = program
        .functions
        .iter()
        .filter(|f| program.is_isr_root(f))
        .map(|f| f.id)
        .collect();
    let mut any_changed = false;
    let mut inlined_callees: HashSet<u32> = HashSet::new();

    for _round in 0..MAX_INLINE_ROUNDS {
        let mut round_changed = false;
        let call_counts = count_direct_calls(program);
        let snapshots: Vec<IrFunction> = program.functions.clone();
        let n = program.functions.len();

        for caller_idx in 0..n {
            if program.functions[caller_idx].blocks.is_empty() {
                continue;
            }
            let mut caller_changed = false;
            loop {
                let mut found: Option<(usize, usize, usize)> = None;
                let caller_ref_vals = caller_ref_param_origin_slots(&program.functions[caller_idx]);
                'scan: for (bi, bb) in program.functions[caller_idx].blocks.iter().enumerate() {
                    for (ii, inst) in bb.insts.iter().enumerate() {
                        if let IrInst::Call { func, args, .. } = inst {
                            let callee_idx = *func as usize;
                            if callee_idx < n && callee_idx != caller_idx {
                                let callee = &snapshots[callee_idx];
                                let count = call_counts[callee_idx];
                                if !isr_roots.contains(&callee.id)
                                    && should_inline_site(
                                        &program.functions[caller_idx],
                                        callee,
                                        args,
                                        count,
                                        &addr_taken,
                                        &machine_fns,
                                        &caller_ref_vals,
                                    )
                                {
                                    found = Some((bi, ii, callee_idx));
                                    break 'scan;
                                }
                            }
                        }
                    }
                }
                let Some((bi, ii, callee_idx)) = found else {
                    break;
                };
                inlined_callees.insert(snapshots[callee_idx].id);
                inline_one_call_site(
                    &mut program.functions[caller_idx],
                    &snapshots[callee_idx],
                    bi,
                    ii,
                );
                caller_changed = true;
                round_changed = true;
                any_changed = true;
            }

            if caller_changed {
                opt::optimize_function_with_config(&mut program.functions[caller_idx], cfg);
            }
        }

        // Recompute call counts and clear dead non-public helpers whose callers were all inlined.
        let remaining_calls = count_direct_calls(program);
        for f in &mut program.functions {
            if inlined_callees.contains(&f.id)
                && !f.blocks.is_empty()
                && !f.is_pub
                && f.name != "main"
                && !f.is_test
                && f.interrupt_source.is_none()
                && !isr_roots.contains(&f.id)
                && f.overlay_state.is_none()
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
}
