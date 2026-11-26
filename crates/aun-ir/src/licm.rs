//! Loop-Invariant Code Motion (LICM) with automatic loop preheader creation.

use crate::cfg::Cfg;
use crate::{ArithMode, BasicBlock, BlockId, IrBinOp, IrFunction, IrInst, IrUnOp, Terminator, ValueId};
use std::collections::{BTreeSet, HashMap, HashSet};

/// Ensures `header` has a dedicated single predecessor block outside `body` that
/// unconditionally jumps (`Terminator::Goto`) to `header`.
///
/// Returns `None` if `header` is the entry block or has no reachable predecessors
/// outside `body`.
pub(crate) fn ensure_loop_preheader(
    func: &mut IrFunction,
    cfg: &Cfg,
    header: BlockId,
    body: &BTreeSet<BlockId>,
) -> Option<BlockId> {
    let entry_id = cfg.entry_block()?;
    if header == entry_id {
        return None;
    }

    let outside_preds: Vec<BlockId> = cfg
        .reachable_preds(header)
        .into_iter()
        .filter(|p| !body.contains(p))
        .collect();
    if outside_preds.is_empty() {
        return None;
    }

    let bb_map: HashMap<BlockId, usize> =
        func.blocks.iter().enumerate().map(|(i, b)| (b.id, i)).collect();

    if outside_preds.len() == 1 {
        let p = outside_preds[0];
        if matches!(func.blocks[bb_map[&p]].term, Terminator::Goto(t) if t == header) {
            return Some(p);
        }
    }

    let pre_id = func
        .blocks
        .iter()
        .map(|b| b.id)
        .max()
        .map_or(0, |m| m + 1);
    let outside_set: HashSet<BlockId> = outside_preds.iter().copied().collect();

    // Redirect all outside predecessors' edges targeting `header` to `pre_id`.
    for &p in &outside_preds {
        let term = &mut func.blocks[bb_map[&p]].term;
        match term {
            Terminator::Goto(t) if *t == header => *t = pre_id,
            Terminator::Branch {
                then_bb, else_bb, ..
            } => {
                if *then_bb == header {
                    *then_bb = pre_id;
                }
                if *else_bb == header {
                    *else_bb = pre_id;
                }
            }
            _ => {}
        }
    }

    // Rewrite leading Phi nodes in `header` (splitting multi-predecessor outside incoming into `pre_id`).
    let mut pre_phis: Vec<IrInst> = Vec::new();
    let header_bb = &mut func.blocks[bb_map[&header]];
    for inst in &mut header_bb.insts {
        let IrInst::Phi { ty, incoming, .. } = inst else {
            break;
        };
        let mut out_entries = Vec::new();
        let mut in_entries = Vec::new();
        for &(pred, val) in incoming.iter() {
            if outside_set.contains(&pred) {
                out_entries.push((pred, val));
            } else {
                in_entries.push((pred, val));
            }
        }
        if out_entries.is_empty() {
            continue;
        }
        if out_entries.len() == 1 || out_entries.windows(2).all(|w| w[0].1 == w[1].1) {
            in_entries.push((pre_id, out_entries[0].1));
        } else {
            let pre_dst = func.next_val;
            func.next_val += 1;
            out_entries.sort_by_key(|&(p, _)| p);
            pre_phis.push(IrInst::Phi {
                dst: pre_dst,
                ty: ty.clone(),
                incoming: out_entries,
            });
            in_entries.push((pre_id, pre_dst));
        }
        in_entries.sort_by_key(|&(p, _)| p);
        *incoming = in_entries;
    }

    func.blocks.push(BasicBlock {
        id: pre_id,
        insts: pre_phis,
        term: Terminator::Goto(header),
    });

    Some(pre_id)
}

fn is_speculatively_hoistable_compute(inst: &IrInst) -> bool {
    match inst {
        IrInst::AddrLocal { .. }
        | IrInst::AddrGlobal { .. }
        | IrInst::FieldAddr { .. }
        | IrInst::Cast { .. }
        | IrInst::FixedInt { .. } => true,
        IrInst::UnOp { op, mode, .. } => *op == IrUnOp::Not || *mode != ArithMode::Checked,
        IrInst::BinOp { op, mode, .. } => {
            matches!(
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
                    | IrBinOp::Shl
                    | IrBinOp::Shr
            ) || (*mode != ArithMode::Checked && !matches!(op, IrBinOp::Div | IrBinOp::Rem))
        }
        _ => false,
    }
}

/// Hoists loop-invariant pure, non-trapping computations into dedicated loop preheaders
/// from innermost natural loops outward.
pub fn licm(func: &mut IrFunction) -> bool {
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

        // Merge back-edges sharing the same header into one natural loop body.
        let mut by_header: HashMap<BlockId, BTreeSet<BlockId>> = HashMap::new();
        for lp in cfg.natural_loops() {
            by_header
                .entry(lp.header)
                .or_default()
                .extend(&lp.body);
        }
        let mut loops: Vec<(BlockId, BTreeSet<BlockId>)> = by_header
            .into_iter()
            .filter(|(h, _)| *h != entry_id && !processed_headers.contains(h))
            .collect();
        loops.sort_by_key(|(h, body)| (body.len(), *h));

        let Some((header, body)) = loops.into_iter().next() else {
            break;
        };
        processed_headers.insert(header);

        let has_outside_pred = cfg
            .reachable_preds(header)
            .into_iter()
            .any(|p| !body.contains(&p));
        if !has_outside_pred {
            continue;
        }

        let bb_map: HashMap<BlockId, usize> =
            func.blocks.iter().enumerate().map(|(i, b)| (b.id, i)).collect();

        let mut defined_in_loop: HashSet<ValueId> = HashSet::new();
        for &bb_id in &body {
            if let Some(&idx) = bb_map.get(&bb_id) {
                for inst in &func.blocks[idx].insts {
                    if let Some(d) = inst.dst() {
                        defined_in_loop.insert(d);
                    }
                }
            }
        }

        // Fixed-point invariant analysis over blocks in `body` ordered by RPO.
        let loop_rpo: Vec<BlockId> = cfg
            .rpo()
            .iter()
            .copied()
            .filter(|id| body.contains(id))
            .collect();

        let mut invariant_vals: HashSet<ValueId> = HashSet::new();
        let mut const_defs: HashMap<ValueId, (BlockId, usize)> = HashMap::new();
        let mut compute_hoist: HashSet<(BlockId, usize)> = HashSet::new();

        loop {
            let mut found_new = false;
            for &bb_id in &loop_rpo {
                let bb = &func.blocks[bb_map[&bb_id]];
                for (inst_idx, inst) in bb.insts.iter().enumerate() {
                    let Some(dst) = inst.dst() else {
                        continue;
                    };
                    if invariant_vals.contains(&dst) {
                        continue;
                    }
                    if matches!(inst, IrInst::ConstInt { .. } | IrInst::ConstBool { .. }) {
                        invariant_vals.insert(dst);
                        const_defs.insert(dst, (bb_id, inst_idx));
                        found_new = true;
                        continue;
                    }
                    if !is_speculatively_hoistable_compute(inst) {
                        continue;
                    }
                    let mut all_inv = true;
                    inst.for_each_use(|u| {
                        if defined_in_loop.contains(&u) && !invariant_vals.contains(&u) {
                            all_inv = false;
                        }
                    });
                    if all_inv {
                        invariant_vals.insert(dst);
                        compute_hoist.insert((bb_id, inst_idx));
                        found_new = true;
                    }
                }
            }
            if !found_new {
                break;
            }
        }

        if compute_hoist.is_empty() {
            continue;
        }

        // Only hoist loop-defined constants that are actually required by hoisted computations.
        let mut to_hoist: HashSet<(BlockId, usize)> = compute_hoist.clone();
        let mut worklist: Vec<(BlockId, usize)> = compute_hoist.into_iter().collect();
        while let Some((bb_id, idx)) = worklist.pop() {
            let inst = &func.blocks[bb_map[&bb_id]].insts[idx];
            inst.for_each_use(|u| {
                if let Some(&loc) = const_defs.get(&u)
                    && to_hoist.insert(loc)
                {
                    worklist.push(loc);
                }
            });
        }

        let Some(pre_id) = ensure_loop_preheader(func, &cfg, header, &body) else {
            continue;
        };

        // Recompute bb_map since ensure_loop_preheader may have pushed a new preheader block.
        let bb_map: HashMap<BlockId, usize> =
            func.blocks.iter().enumerate().map(|(i, b)| (b.id, i)).collect();

        let mut hoisted_insts = Vec::with_capacity(to_hoist.len());
        for &bb_id in &loop_rpo {
            let bb = &mut func.blocks[bb_map[&bb_id]];
            let mut idx = 0usize;
            let mut new_insts = Vec::with_capacity(bb.insts.len());
            for inst in bb.insts.drain(..) {
                if to_hoist.contains(&(bb_id, idx)) {
                    hoisted_insts.push(inst);
                } else {
                    new_insts.push(inst);
                }
                idx += 1;
            }
            bb.insts = new_insts;
        }

        let pre_idx = bb_map[&pre_id];
        func.blocks[pre_idx].insts.extend(hoisted_insts);
        any_changed = true;
    }

    any_changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify::verify_program;
    use crate::{IntType, IrProgram, IrType, LocalSlot};

    fn verify_prog(func: IrFunction) -> IrProgram {
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
            param_count: 2,
            locals: vec![
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

    #[test]
    fn licm_hoists_invariant_arith_and_field_addr_to_preheader() {
        let u8t = IrType::Int(IntType::U8);
        // bb0: a = LoadLocal(0), b = LoadLocal(1), cond = ConstBool(true), addr0 = AddrLocal(0) -> Goto(bb1)
        // bb1 (header): Phi(iv), Branch(cond, bb2, bb3)
        // bb2 (latch): inv1 = a +% b; fa = FieldAddr(addr0, 4); StorePtr(fa, inv1); Goto(bb1)
        // bb3: Return(iv)
        let func = base_fn(
            vec![
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
                        IrInst::ConstBool { dst: 2, val: true },
                        IrInst::AddrLocal { dst: 3, slot: 0 },
                        IrInst::ConstInt {
                            dst: 4,
                            ty: u8t.clone(),
                            val: 0,
                        },
                    ],
                    term: Terminator::Goto(1),
                },
                BasicBlock {
                    id: 1,
                    insts: vec![IrInst::Phi {
                        dst: 5,
                        ty: u8t.clone(),
                        incoming: vec![(0, 4), (2, 7)],
                    }],
                    term: Terminator::Branch {
                        cond: 2,
                        then_bb: 2,
                        else_bb: 3,
                    },
                },
                BasicBlock {
                    id: 2,
                    insts: vec![
                        IrInst::BinOp {
                            dst: 6,
                            op: IrBinOp::Add,
                            mode: ArithMode::Wrapping,
                            ty: u8t.clone(),
                            lhs: 0,
                            rhs: 1,
                            site_id: 0,
                        },
                        IrInst::FieldAddr {
                            dst: 7,
                            base: 3,
                            byte_offset: 4,
                        },
                        IrInst::StorePtr {
                            addr: 7,
                            val: 6,
                            ty: u8t,
                        },
                    ],
                    term: Terminator::Goto(1),
                },
                BasicBlock {
                    id: 3,
                    insts: vec![],
                    term: Terminator::Return(Some(5)),
                },
            ],
            8,
        );

        let mut prog = verify_prog(func);
        assert!(licm(&mut prog.functions[0]));
        verify_program(&prog).expect("LICM output must verify");

        // inv1 (dst 6) and fa (dst 7) must have been hoisted into bb0; only StorePtr remains in bb2.
        assert_eq!(prog.functions[0].blocks[2].insts.len(), 1);
        assert!(matches!(
            prog.functions[0].blocks[2].insts[0],
            IrInst::StorePtr { .. }
        ));
        assert!(prog.functions[0].blocks[0].insts.len() == 7);
    }

    #[test]
    fn licm_creates_preheader_when_multiple_outside_entries() {
        let u8t = IrType::Int(IntType::U8);
        // bb0 -> Branch(c, bb1, bb2)
        // bb1 -> x = 10 -> Goto(bb3)
        // bb2 -> x = 20 -> Goto(bb3)
        // bb3 (header): p = Phi((bb1, 10), (bb2, 20), (bb4, p)); Branch(c, bb4, bb5)
        // bb4 (latch): inv = a +% b; Goto(bb3)
        // bb5: Return(p)
        let func = base_fn(
            vec![
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
                        IrInst::ConstBool { dst: 2, val: true },
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
                        ty: u8t.clone(),
                        val: 10,
                    }],
                    term: Terminator::Goto(3),
                },
                BasicBlock {
                    id: 2,
                    insts: vec![IrInst::ConstInt {
                        dst: 4,
                        ty: u8t.clone(),
                        val: 20,
                    }],
                    term: Terminator::Goto(3),
                },
                BasicBlock {
                    id: 3,
                    insts: vec![IrInst::Phi {
                        dst: 5,
                        ty: u8t.clone(),
                        incoming: vec![(1, 3), (2, 4), (4, 5)],
                    }],
                    term: Terminator::Branch {
                        cond: 2,
                        then_bb: 4,
                        else_bb: 5,
                    },
                },
                BasicBlock {
                    id: 4,
                    insts: vec![IrInst::BinOp {
                        dst: 6,
                        op: IrBinOp::Add,
                        mode: ArithMode::Wrapping,
                        ty: u8t,
                        lhs: 0,
                        rhs: 1,
                        site_id: 0,
                    }],
                    term: Terminator::Goto(3),
                },
                BasicBlock {
                    id: 5,
                    insts: vec![],
                    term: Terminator::Return(Some(5)),
                },
            ],
            7,
        );

        let mut prog = verify_prog(func);
        let orig_blocks = prog.functions[0].blocks.len();
        assert!(licm(&mut prog.functions[0]));
        verify_program(&prog).expect("LICM with created preheader must verify");
        assert_eq!(prog.functions[0].blocks.len(), orig_blocks + 1);
        assert!(prog.functions[0].blocks[4].insts.is_empty());
    }

    #[test]
    fn licm_does_not_hoist_trapping_or_memory_loads() {
        let u8t = IrType::Int(IntType::U8);
        // bb0 -> bb1 -> bb2 (Div, Checked Add, LoadLocal) -> bb1 | bb3
        let func = base_fn(
            vec![
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
                        IrInst::ConstBool { dst: 2, val: true },
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
                        IrInst::BinOp {
                            dst: 3,
                            op: IrBinOp::Div,
                            mode: ArithMode::Wrapping,
                            ty: u8t.clone(),
                            lhs: 0,
                            rhs: 1,
                            site_id: 0,
                        },
                        IrInst::BinOp {
                            dst: 4,
                            op: IrBinOp::Add,
                            mode: ArithMode::Checked,
                            ty: u8t.clone(),
                            lhs: 0,
                            rhs: 1,
                            site_id: 0,
                        },
                        IrInst::LoadLocal {
                            dst: 5,
                            slot: 0,
                            ty: u8t,
                        },
                    ],
                    term: Terminator::Goto(1),
                },
                BasicBlock {
                    id: 3,
                    insts: vec![],
                    term: Terminator::Return(Some(0)),
                },
            ],
            6,
        );

        let mut prog = verify_prog(func);
        assert!(!licm(&mut prog.functions[0]));
        verify_program(&prog).expect("non-hoistable loop must verify");
        assert_eq!(prog.functions[0].blocks[2].insts.len(), 3);
    }
}
