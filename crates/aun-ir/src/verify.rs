//! Structural and SSA dominance verifier for `aun-ir`.

use crate::cfg::Cfg;
use crate::{BlockId, IrFunction, IrInst, IrProgram, Terminator, ValueId};
use std::collections::{BTreeSet, HashMap, HashSet};

/// Verifies structural, SSA, and dominance invariants for every function in `program`.
pub fn verify_program(program: &IrProgram) -> Result<(), Vec<String>> {
    let mut all_errors = Vec::new();
    let dead_fns: HashSet<u32> = program
        .functions
        .iter()
        .filter(|f| f.blocks.is_empty() && f.locals.is_empty() && f.next_val == 0 && !f.is_pub)
        .map(|f| f.id)
        .collect();
    for func in &program.functions {
        if dead_fns.contains(&func.id) {
            continue;
        }
        if let Err(errs) = verify_function(program, func) {
            all_errors.extend(errs);
        }
        for bb in &func.blocks {
            for inst in &bb.insts {
                match inst {
                    IrInst::Call { func: cid, .. } if dead_fns.contains(cid) => {
                        all_errors.push(format!(
                            "fn `{}` bb{} calls eliminated empty function id {}",
                            func.name, bb.id, cid
                        ));
                    }
                    IrInst::AddrFn {
                        target: crate::IrFnTarget::Fn(cid),
                        ..
                    } if dead_fns.contains(cid) => {
                        all_errors.push(format!(
                            "fn `{}` bb{} takes address of eliminated empty function id {}",
                            func.name, bb.id, cid
                        ));
                    }
                    _ => {}
                }
            }
        }
    }
    if all_errors.is_empty() {
        Ok(())
    } else {
        Err(all_errors)
    }
}

/// Verifies structural, SSA, and dominance invariants for a single `IrFunction`.
///
/// Checks:
/// 1. Function has an entry block (`bb0`) and unique `BlockId`s.
/// 2. Every `Terminator` targets an existing `BlockId`.
/// 3. `LocalId` operands (`LoadLocal`, `StoreLocal`, `AddrLocal`) are within `func.locals.len()`.
/// 4. Every defined `ValueId` is `< func.next_val` and defined at most once (single assignment).
/// 5. All `IrInst::Phi` instructions appear strictly at the start of their basic block.
/// 6. In all reachable blocks:
///    - Each `Phi` has unique CFG predecessor keys, covers every reachable predecessor, and each
///      reachable predecessor's incoming value is defined and dominates the end of that predecessor.
///    - Every non-`Phi` operand is `< func.next_val`, defined in a reachable block, and strictly
///      precedes its use (either earlier in the same block or in a strictly dominating block).
///    - Every terminator operand is `< func.next_val`, defined in a reachable block, and either
///      defined in the same block or in a strictly dominating block.
pub fn verify_function(_program: &IrProgram, func: &IrFunction) -> Result<(), Vec<String>> {
    let mut errs = Vec::new();
    let fn_name = &func.name;

    if func.blocks.is_empty() {
        errs.push(format!("fn `{fn_name}` has no basic blocks"));
        return Err(errs);
    }

    let mut block_ids = HashMap::with_capacity(func.blocks.len());
    for (idx, bb) in func.blocks.iter().enumerate() {
        if block_ids.insert(bb.id, idx).is_some() {
            errs.push(format!("fn `{fn_name}` has duplicate BlockId {}", bb.id));
        }
    }
    if !block_ids.contains_key(&0) {
        errs.push(format!("fn `{fn_name}` is missing entry block bb0"));
    }

    // Check terminator targets, local bounds, single-def SSA values, and Phi placement.
    let mut defs: HashMap<ValueId, (BlockId, usize)> = HashMap::new();
    let num_locals = func.locals.len() as u32;

    for bb in &func.blocks {
        match bb.term {
            Terminator::Goto(target) => {
                if !block_ids.contains_key(&target) {
                    errs.push(format!(
                        "fn `{fn_name}` bb{}: Goto targets missing bb{target}",
                        bb.id
                    ));
                }
            }
            Terminator::Branch {
                then_bb, else_bb, ..
            } => {
                if !block_ids.contains_key(&then_bb) {
                    errs.push(format!(
                        "fn `{fn_name}` bb{}: Branch then_bb targets missing bb{then_bb}",
                        bb.id
                    ));
                }
                if !block_ids.contains_key(&else_bb) {
                    errs.push(format!(
                        "fn `{fn_name}` bb{}: Branch else_bb targets missing bb{else_bb}",
                        bb.id
                    ));
                }
            }
            Terminator::Return(_) | Terminator::Unreachable => {}
        }

        let mut seen_non_phi = false;
        for (inst_idx, inst) in bb.insts.iter().enumerate() {
            if matches!(inst, IrInst::Phi { .. }) {
                if seen_non_phi {
                    errs.push(format!(
                        "fn `{fn_name}` bb{} inst#{inst_idx}: Phi appears after non-Phi instruction",
                        bb.id
                    ));
                }
            } else {
                seen_non_phi = true;
            }

            match inst {
                IrInst::LoadLocal { slot, .. }
                | IrInst::StoreLocal { slot, .. }
                | IrInst::AddrLocal { slot, .. }
                    if *slot >= num_locals =>
                {
                    errs.push(format!(
                        "fn `{fn_name}` bb{} inst#{inst_idx}: invalid local slot {slot} (locals={num_locals})",
                        bb.id
                    ));
                }
                _ => {}
            }

            if let Some(dst) = inst.dst() {
                if dst >= func.next_val {
                    errs.push(format!(
                        "fn `{fn_name}` bb{} inst#{inst_idx}: dst v{dst} >= next_val ({})",
                        bb.id, func.next_val
                    ));
                }
                if let Some((prev_bb, prev_idx)) = defs.insert(dst, (bb.id, inst_idx)) {
                    errs.push(format!(
                        "fn `{fn_name}` bb{} inst#{inst_idx}: duplicate definition of v{dst} (previously at bb{prev_bb} inst#{prev_idx})",
                        bb.id
                    ));
                }
            }
        }
    }

    let cfg = Cfg::build(func);
    for bb in &func.blocks {
        if !cfg.is_reachable(bb.id) {
            continue;
        }
        let all_preds: BTreeSet<BlockId> = cfg.preds(bb.id).iter().copied().collect();
        let reach_preds: BTreeSet<BlockId> = cfg.reachable_preds(bb.id).into_iter().collect();

        for (inst_idx, inst) in bb.insts.iter().enumerate() {
            if let IrInst::Phi { incoming, .. } = inst {
                let mut seen_preds = HashSet::with_capacity(incoming.len());
                for &(pred_bb, val) in incoming {
                    if !seen_preds.insert(pred_bb) {
                        errs.push(format!(
                            "fn `{fn_name}` bb{} inst#{inst_idx}: Phi has duplicate entry for bb{pred_bb}",
                            bb.id
                        ));
                    }
                    if !all_preds.contains(&pred_bb) {
                        errs.push(format!(
                            "fn `{fn_name}` bb{} inst#{inst_idx}: Phi references non-predecessor bb{pred_bb}",
                            bb.id
                        ));
                        continue;
                    }
                    if cfg.is_reachable(pred_bb) {
                        check_phi_incoming(
                            fn_name,
                            bb.id,
                            inst_idx,
                            pred_bb,
                            val,
                            func.next_val,
                            &defs,
                            &cfg,
                            &mut errs,
                        );
                    }
                }
                for rp in &reach_preds {
                    if !seen_preds.contains(rp) {
                        errs.push(format!(
                            "fn `{fn_name}` bb{} inst#{inst_idx}: Phi missing reachable predecessor bb{rp}",
                            bb.id
                        ));
                    }
                }
            } else {
                inst.for_each_use(|u| {
                    check_use_dom(
                        fn_name,
                        bb.id,
                        Some(inst_idx),
                        u,
                        func.next_val,
                        &defs,
                        &cfg,
                        &mut errs,
                    );
                });
            }
        }

        match bb.term {
            Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } => {
                check_use_dom(
                    fn_name,
                    bb.id,
                    None,
                    v,
                    func.next_val,
                    &defs,
                    &cfg,
                    &mut errs,
                );
            }
            _ => {}
        }
    }

    if errs.is_empty() {
        Ok(())
    } else {
        Err(errs)
    }
}

#[allow(clippy::too_many_arguments)]
fn check_phi_incoming(
    fn_name: &str,
    bb_id: BlockId,
    inst_idx: usize,
    pred_bb: BlockId,
    val: ValueId,
    next_val: ValueId,
    defs: &HashMap<ValueId, (BlockId, usize)>,
    cfg: &Cfg,
    errs: &mut Vec<String>,
) {
    if val >= next_val {
        errs.push(format!(
            "fn `{fn_name}` bb{bb_id} inst#{inst_idx}: Phi uses v{val} >= next_val ({next_val})"
        ));
        return;
    }
    let Some(&(def_bb, _)) = defs.get(&val) else {
        errs.push(format!(
            "fn `{fn_name}` bb{bb_id} inst#{inst_idx}: Phi uses undefined value v{val} from bb{pred_bb}"
        ));
        return;
    };
    if !cfg.dominates(def_bb, pred_bb) {
        errs.push(format!(
            "fn `{fn_name}` bb{bb_id} inst#{inst_idx}: Phi incoming v{val} (defined in bb{def_bb}) does not dominate predecessor bb{pred_bb}"
        ));
    }
}

#[allow(clippy::too_many_arguments)]
fn check_use_dom(
    fn_name: &str,
    bb_id: BlockId,
    inst_idx: Option<usize>,
    val: ValueId,
    next_val: ValueId,
    defs: &HashMap<ValueId, (BlockId, usize)>,
    cfg: &Cfg,
    errs: &mut Vec<String>,
) {
    let loc = match inst_idx {
        Some(idx) => format!("inst#{idx}"),
        None => "terminator".to_string(),
    };
    if val >= next_val {
        errs.push(format!(
            "fn `{fn_name}` bb{bb_id} {loc}: uses v{val} >= next_val ({next_val})"
        ));
        return;
    }
    let Some(&(def_bb, def_idx)) = defs.get(&val) else {
        errs.push(format!(
            "fn `{fn_name}` bb{bb_id} {loc}: uses undefined value v{val}"
        ));
        return;
    };
    if !cfg.is_reachable(def_bb) {
        errs.push(format!(
            "fn `{fn_name}` bb{bb_id} {loc}: uses v{val} defined in unreachable block bb{def_bb}"
        ));
        return;
    }
    let ok = if def_bb == bb_id {
        match inst_idx {
            Some(use_idx) => def_idx < use_idx,
            None => true,
        }
    } else {
        cfg.strictly_dominates(def_bb, bb_id)
    };
    if !ok {
        errs.push(format!(
            "fn `{fn_name}` bb{bb_id} {loc}: definition of v{val} in bb{def_bb}#{def_idx} does not dominate use"
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArithMode, BasicBlock, IntType, IrBinOp, IrProgram, IrType};

    fn dummy_program(func: IrFunction) -> IrProgram {
        IrProgram {
            functions: vec![func],
            ..Default::default()
        }
    }

    fn base_fn(blocks: Vec<BasicBlock>, next_val: u32) -> IrFunction {
        IrFunction {
            id: 0,
            name: "test".into(),
            is_pub: false,
            param_count: 0,
            locals: vec![],
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
            mock_target: None,
            cycle_budget: None,
            loop_bounds: std::collections::HashMap::new(),
            blocks,
            next_val,
        }
    }

    #[test]
    fn valid_diamond_phi_passes() {
        let func = base_fn(
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
                    insts: vec![IrInst::ConstInt {
                        dst: 1,
                        ty: IrType::Int(IntType::U8),
                        val: 10,
                    }],
                    term: Terminator::Goto(3),
                },
                BasicBlock {
                    id: 2,
                    insts: vec![IrInst::ConstInt {
                        dst: 2,
                        ty: IrType::Int(IntType::U8),
                        val: 20,
                    }],
                    term: Terminator::Goto(3),
                },
                BasicBlock {
                    id: 3,
                    insts: vec![IrInst::Phi {
                        dst: 3,
                        ty: IrType::Int(IntType::U8),
                        incoming: vec![(1, 1), (2, 2)],
                    }],
                    term: Terminator::Return(Some(3)),
                },
            ],
            4,
        );
        let prog = dummy_program(func);
        assert!(verify_program(&prog).is_ok());
    }

    #[test]
    fn rejects_use_before_def() {
        let func = base_fn(
            vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::BinOp {
                        dst: 1,
                        op: IrBinOp::Add,
                        mode: ArithMode::Wrapping,
                        ty: IrType::Int(IntType::U8),
                        lhs: 0,
                        rhs: 0,
                        site_id: 0,
                    },
                    IrInst::ConstInt {
                        dst: 0,
                        ty: IrType::Int(IntType::U8),
                        val: 1,
                    },
                ],
                term: Terminator::Return(Some(1)),
            }],
            2,
        );
        let prog = dummy_program(func);
        assert!(verify_program(&prog).is_err());
    }

    #[test]
    fn rejects_duplicate_def() {
        let func = base_fn(
            vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::ConstInt {
                        dst: 0,
                        ty: IrType::Int(IntType::U8),
                        val: 1,
                    },
                    IrInst::ConstInt {
                        dst: 0,
                        ty: IrType::Int(IntType::U8),
                        val: 2,
                    },
                ],
                term: Terminator::Return(Some(0)),
            }],
            1,
        );
        let prog = dummy_program(func);
        assert!(verify_program(&prog).is_err());
    }

    #[test]
    fn rejects_non_leading_phi() {
        let func = base_fn(
            vec![
                BasicBlock {
                    id: 0,
                    insts: vec![IrInst::ConstInt {
                        dst: 0,
                        ty: IrType::Int(IntType::U8),
                        val: 1,
                    }],
                    term: Terminator::Goto(1),
                },
                BasicBlock {
                    id: 1,
                    insts: vec![
                        IrInst::ConstInt {
                            dst: 1,
                            ty: IrType::Int(IntType::U8),
                            val: 2,
                        },
                        IrInst::Phi {
                            dst: 2,
                            ty: IrType::Int(IntType::U8),
                            incoming: vec![(0, 0)],
                        },
                    ],
                    term: Terminator::Return(Some(2)),
                },
            ],
            3,
        );
        let prog = dummy_program(func);
        assert!(verify_program(&prog).is_err());
    }

    #[test]
    fn rejects_missing_phi_predecessor() {
        let func = base_fn(
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
                    insts: vec![IrInst::ConstInt {
                        dst: 1,
                        ty: IrType::Int(IntType::U8),
                        val: 10,
                    }],
                    term: Terminator::Goto(3),
                },
                BasicBlock {
                    id: 2,
                    insts: vec![IrInst::ConstInt {
                        dst: 2,
                        ty: IrType::Int(IntType::U8),
                        val: 20,
                    }],
                    term: Terminator::Goto(3),
                },
                BasicBlock {
                    id: 3,
                    insts: vec![IrInst::Phi {
                        dst: 3,
                        ty: IrType::Int(IntType::U8),
                        incoming: vec![(1, 1)],
                    }],
                    term: Terminator::Return(Some(3)),
                },
            ],
            4,
        );
        let prog = dummy_program(func);
        assert!(verify_program(&prog).is_err());
    }
}
