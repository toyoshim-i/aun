//! Sparse Conditional Constant Propagation (Wegman & Zadeck, 1991) for `aun-ir` SSA form.

use crate::opt::truncate_int;
use crate::ssa::simplify_trivial_phis;
use crate::{ArithMode, BlockId, IntType, IrBinOp, IrFunction, IrInst, IrType, IrUnOp, Terminator, ValueId};
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConstVal {
    Int(IntType, i64),
    Bool(bool),
}

impl ConstVal {
    pub(crate) fn matches_type(self, ty: &IrType) -> bool {
        match (self, ty) {
            (ConstVal::Int(it1, _), IrType::Int(it2)) => it1 == *it2,
            (ConstVal::Bool(_), IrType::Bool) => true,
            _ => false,
        }
    }

    pub(crate) fn to_inst(self, dst: ValueId) -> IrInst {
        match self {
            ConstVal::Int(it, val) => IrInst::ConstInt {
                dst,
                ty: IrType::Int(it),
                val,
            },
            ConstVal::Bool(val) => IrInst::ConstBool { dst, val },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lat {
    Top,
    Const(ConstVal),
    Bottom,
}

fn meet(a: Lat, b: Lat) -> Lat {
    match (a, b) {
        (Lat::Top, x) | (x, Lat::Top) => x,
        (Lat::Const(c1), Lat::Const(c2)) if c1 == c2 => Lat::Const(c1),
        _ => Lat::Bottom,
    }
}

fn apply_arith_const(raw: i128, it: IntType, mode: ArithMode) -> Option<ConstVal> {
    let min = it.min_i64() as i128;
    let max = it.max_i64() as i128;
    match mode {
        ArithMode::Wrapping => Some(ConstVal::Int(it, truncate_int(raw as i64, it))),
        ArithMode::Saturating => {
            let clamped = raw.clamp(min, max) as i64;
            Some(ConstVal::Int(it, truncate_int(clamped, it)))
        }
        ArithMode::Checked => {
            if raw < min || raw > max {
                None
            } else {
                Some(ConstVal::Int(it, truncate_int(raw as i64, it)))
            }
        }
    }
}

fn eval_unop(op: IrUnOp, mode: ArithMode, ty: &IrType, v: ConstVal) -> Option<ConstVal> {
    match (op, ty, v) {
        (IrUnOp::Not, IrType::Bool, ConstVal::Bool(b)) => Some(ConstVal::Bool(!b)),
        (IrUnOp::Not, IrType::Int(it), ConstVal::Int(_, x)) => {
            Some(ConstVal::Int(*it, truncate_int(!x, *it)))
        }
        (IrUnOp::Neg, IrType::Int(it), ConstVal::Int(_, x)) => {
            apply_arith_const(-(x as i128), *it, mode)
        }
        _ => None,
    }
}

fn eval_cast(from_ty: &IrType, to_ty: &IrType, v: ConstVal) -> Option<ConstVal> {
    match (from_ty, to_ty, v) {
        (_, IrType::Int(to_it), ConstVal::Int(_, x)) => {
            Some(ConstVal::Int(*to_it, truncate_int(x, *to_it)))
        }
        (IrType::Bool, IrType::Int(to_it), ConstVal::Bool(b)) => {
            Some(ConstVal::Int(*to_it, if b { 1 } else { 0 }))
        }
        (_, IrType::Bool, ConstVal::Bool(b)) => Some(ConstVal::Bool(b)),
        (_, IrType::Bool, ConstVal::Int(_, x)) => Some(ConstVal::Bool(x != 0)),
        _ => None,
    }
}

pub fn eval_binop(
    op: IrBinOp,
    mode: ArithMode,
    ty: &IrType,
    lv: ConstVal,
    rv: ConstVal,
) -> Option<ConstVal> {
    match (lv, rv) {
        (ConstVal::Bool(a), ConstVal::Bool(b)) => match op {
            IrBinOp::Eq => Some(ConstVal::Bool(a == b)),
            IrBinOp::Ne => Some(ConstVal::Bool(a != b)),
            IrBinOp::BitAnd => Some(ConstVal::Bool(a & b)),
            IrBinOp::BitOr => Some(ConstVal::Bool(a | b)),
            IrBinOp::BitXor => Some(ConstVal::Bool(a ^ b)),
            _ => None,
        },
        (ConstVal::Int(lit, a), ConstVal::Int(rit, b)) => {
            if lit != rit {
                return None;
            }
            match op {
                IrBinOp::Eq => return Some(ConstVal::Bool(a == b)),
                IrBinOp::Ne => return Some(ConstVal::Bool(a != b)),
                IrBinOp::Lt => return Some(ConstVal::Bool(a < b)),
                IrBinOp::Le => return Some(ConstVal::Bool(a <= b)),
                IrBinOp::Gt => return Some(ConstVal::Bool(a > b)),
                IrBinOp::Ge => return Some(ConstVal::Bool(a >= b)),
                _ => {}
            }
            let IrType::Int(it) = *ty else {
                return None;
            };
            match op {
                IrBinOp::BitAnd => Some(ConstVal::Int(it, truncate_int(a & b, it))),
                IrBinOp::BitOr => Some(ConstVal::Int(it, truncate_int(a | b, it))),
                IrBinOp::BitXor => Some(ConstVal::Int(it, truncate_int(a ^ b, it))),
                IrBinOp::Shl => {
                    if b >= 0 && b < it.bit_width() as i64 {
                        Some(ConstVal::Int(it, truncate_int(a << (b as u32), it)))
                    } else {
                        None
                    }
                }
                IrBinOp::Shr => {
                    if b >= 0 && b < it.bit_width() as i64 {
                        Some(ConstVal::Int(it, truncate_int(a >> (b as u32), it)))
                    } else {
                        None
                    }
                }
                IrBinOp::Add => apply_arith_const((a as i128) + (b as i128), it, mode),
                IrBinOp::Sub => apply_arith_const((a as i128) - (b as i128), it, mode),
                IrBinOp::Mul => apply_arith_const((a as i128) * (b as i128), it, mode),
                IrBinOp::Div => {
                    if b == 0 {
                        return None;
                    }
                    apply_arith_const((a as i128) / (b as i128), it, mode)
                }
                IrBinOp::Rem => {
                    if b == 0 {
                        return None;
                    }
                    apply_arith_const((a as i128) % (b as i128), it, mode)
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// Runs Sparse Conditional Constant Propagation (SCCP) on `func`.
///
/// Propagates constants through `Phi` nodes and conditional branches across loops
/// using the two-worklist Wegman-Zadeck algorithm, folds constant branches to `Goto`,
/// trims unreachable `Phi` incoming edges and dead blocks, and calls `simplify_trivial_phis`.
pub fn sccp(func: &mut IrFunction) -> bool {
    if func.blocks.is_empty() {
        return false;
    }

    let entry_id = func
        .blocks
        .iter()
        .find(|b| b.id == 0)
        .map_or(func.blocks[0].id, |b| b.id);
    let bb_map: HashMap<BlockId, usize> =
        func.blocks.iter().enumerate().map(|(i, b)| (b.id, i)).collect();

    let n_vals = func.next_val as usize;
    let mut val_state = vec![Lat::Top; n_vals];
    let mut use_sites: Vec<Vec<(BlockId, usize)>> = vec![Vec::new(); n_vals];
    let mut term_use_sites: Vec<Vec<BlockId>> = vec![Vec::new(); n_vals];

    for bb in &func.blocks {
        for (idx, inst) in bb.insts.iter().enumerate() {
            inst.for_each_use(|v| {
                if let Some(list) = use_sites.get_mut(v as usize) {
                    list.push((bb.id, idx));
                }
            });
        }
        if let Terminator::Branch { cond, .. } = bb.term
            && let Some(list) = term_use_sites.get_mut(cond as usize)
        {
            list.push(bb.id);
        }
    }

    let mut cfg_worklist: VecDeque<(Option<BlockId>, BlockId)> = VecDeque::new();
    let mut ssa_worklist: VecDeque<ValueId> = VecDeque::new();
    let mut exec_edges: HashSet<(Option<BlockId>, BlockId)> = HashSet::new();
    let mut exec_blocks: HashSet<BlockId> = HashSet::new();

    cfg_worklist.push_back((None, entry_id));

    let set_lat = |dst: ValueId,
                   new_lat: Lat,
                   val_state: &mut [Lat],
                   ssa_wl: &mut VecDeque<ValueId>| {
        let idx = dst as usize;
        if idx < val_state.len() && val_state[idx] != new_lat {
            val_state[idx] = new_lat;
            ssa_wl.push_back(dst);
        }
    };

    let eval_inst = |bb_id: BlockId,
                     inst: &IrInst,
                     val_state: &mut [Lat],
                     exec_edges: &HashSet<(Option<BlockId>, BlockId)>,
                     ssa_wl: &mut VecDeque<ValueId>| {
        match inst {
            IrInst::Phi { dst, ty, incoming } => {
                let mut acc = Lat::Top;
                for &(pred, val) in incoming {
                    if exec_edges.contains(&(Some(pred), bb_id)) {
                        let in_lat = val_state.get(val as usize).copied().unwrap_or(Lat::Bottom);
                        let normalized = match in_lat {
                            Lat::Const(cv) if !cv.matches_type(ty) => Lat::Bottom,
                            other => other,
                        };
                        acc = meet(acc, normalized);
                        if acc == Lat::Bottom {
                            break;
                        }
                    }
                }
                set_lat(*dst, acc, val_state, ssa_wl);
            }
            IrInst::ConstInt {
                dst,
                ty: IrType::Int(it),
                val,
            } => {
                let cv = ConstVal::Int(*it, truncate_int(*val, *it));
                set_lat(*dst, Lat::Const(cv), val_state, ssa_wl);
            }
            IrInst::ConstBool { dst, val } => {
                set_lat(*dst, Lat::Const(ConstVal::Bool(*val)), val_state, ssa_wl);
            }
            IrInst::UnOp {
                dst,
                op,
                mode,
                ty,
                val,
                ..
            } => {
                let v_lat = val_state.get(*val as usize).copied().unwrap_or(Lat::Bottom);
                let new_lat = match v_lat {
                    Lat::Top => Lat::Top,
                    Lat::Bottom => Lat::Bottom,
                    Lat::Const(cv) => match eval_unop(*op, *mode, ty, cv) {
                        Some(res) => Lat::Const(res),
                        None => Lat::Bottom,
                    },
                };
                set_lat(*dst, new_lat, val_state, ssa_wl);
            }
            IrInst::Cast {
                dst,
                val,
                from_ty,
                to_ty,
            } => {
                let v_lat = val_state.get(*val as usize).copied().unwrap_or(Lat::Bottom);
                let new_lat = match v_lat {
                    Lat::Top => Lat::Top,
                    Lat::Bottom => Lat::Bottom,
                    Lat::Const(cv) => match eval_cast(from_ty, to_ty, cv) {
                        Some(res) => Lat::Const(res),
                        None => Lat::Bottom,
                    },
                };
                set_lat(*dst, new_lat, val_state, ssa_wl);
            }
            IrInst::BinOp {
                dst,
                op,
                mode,
                ty,
                lhs,
                rhs,
                ..
            } => {
                let l_lat = val_state.get(*lhs as usize).copied().unwrap_or(Lat::Bottom);
                let r_lat = val_state.get(*rhs as usize).copied().unwrap_or(Lat::Bottom);
                let new_lat = match (l_lat, r_lat) {
                    (Lat::Const(lc), Lat::Const(rc)) => match eval_binop(*op, *mode, ty, lc, rc) {
                        Some(res) => Lat::Const(res),
                        None => Lat::Bottom,
                    },
                    (Lat::Bottom, _) | (_, Lat::Bottom) => Lat::Bottom,
                    _ => Lat::Top,
                };
                set_lat(*dst, new_lat, val_state, ssa_wl);
            }
            other => {
                if let Some(dst) = other.dst() {
                    set_lat(dst, Lat::Bottom, val_state, ssa_wl);
                }
            }
        }
    };

    let eval_term = |bb_id: BlockId,
                     term: &Terminator,
                     val_state: &[Lat],
                     cfg_wl: &mut VecDeque<(Option<BlockId>, BlockId)>| {
        match *term {
            Terminator::Goto(target) => {
                cfg_wl.push_back((Some(bb_id), target));
            }
            Terminator::Branch {
                cond,
                then_bb,
                else_bb,
            } => {
                match val_state.get(cond as usize).copied().unwrap_or(Lat::Bottom) {
                    Lat::Top => {}
                    Lat::Const(ConstVal::Bool(true)) => {
                        cfg_wl.push_back((Some(bb_id), then_bb));
                    }
                    Lat::Const(ConstVal::Bool(false)) => {
                        cfg_wl.push_back((Some(bb_id), else_bb));
                    }
                    _ => {
                        cfg_wl.push_back((Some(bb_id), then_bb));
                        if else_bb != then_bb {
                            cfg_wl.push_back((Some(bb_id), else_bb));
                        }
                    }
                }
            }
            Terminator::Return(_) | Terminator::Unreachable => {}
        }
    };

    while !cfg_worklist.is_empty() || !ssa_worklist.is_empty() {
        while let Some((pred_opt, bb_id)) = cfg_worklist.pop_front() {
            if !exec_edges.insert((pred_opt, bb_id)) {
                continue;
            }
            let Some(&bb_idx) = bb_map.get(&bb_id) else {
                continue;
            };
            let first_visit = exec_blocks.insert(bb_id);
            let bb = &func.blocks[bb_idx];

            for inst in &bb.insts {
                if matches!(inst, IrInst::Phi { .. }) {
                    eval_inst(bb_id, inst, &mut val_state, &exec_edges, &mut ssa_worklist);
                } else {
                    break;
                }
            }
            if first_visit {
                for inst in &bb.insts {
                    if !matches!(inst, IrInst::Phi { .. }) {
                        eval_inst(bb_id, inst, &mut val_state, &exec_edges, &mut ssa_worklist);
                    }
                }
                eval_term(bb_id, &bb.term, &val_state, &mut cfg_worklist);
            }
        }

        while let Some(v) = ssa_worklist.pop_front() {
            let v_idx = v as usize;
            if v_idx >= use_sites.len() {
                continue;
            }
            for &(bb_id, inst_idx) in &use_sites[v_idx] {
                if !exec_blocks.contains(&bb_id) {
                    continue;
                }
                let inst = &func.blocks[bb_map[&bb_id]].insts[inst_idx];
                eval_inst(bb_id, inst, &mut val_state, &exec_edges, &mut ssa_worklist);
            }
            for &bb_id in &term_use_sites[v_idx] {
                if !exec_blocks.contains(&bb_id) {
                    continue;
                }
                let term = &func.blocks[bb_map[&bb_id]].term;
                eval_term(bb_id, term, &val_state, &mut cfg_worklist);
            }
        }
    }

    // Apply discovered constants, dead edges, and unreachable blocks.
    let mut changed = false;
    let orig_block_count = func.blocks.len();
    func.blocks.retain(|bb| exec_blocks.contains(&bb.id));
    if func.blocks.len() != orig_block_count {
        changed = true;
    }

    for bb in &mut func.blocks {
        let bb_id = bb.id;
        let phi_end = bb
            .insts
            .iter()
            .take_while(|i| matches!(i, IrInst::Phi { .. }))
            .count();

        let mut promoted_consts: Vec<IrInst> = Vec::new();
        let mut keep_phi = vec![true; phi_end];
        for (idx, inst) in bb.insts[..phi_end].iter_mut().enumerate() {
            let IrInst::Phi { dst, ty, incoming } = inst else {
                unreachable!();
            };
            if let Some(&Lat::Const(cv)) = val_state.get(*dst as usize)
                && cv.matches_type(ty)
            {
                promoted_consts.push(cv.to_inst(*dst));
                keep_phi[idx] = false;
                changed = true;
                continue;
            }
            let before = incoming.len();
            incoming.retain(|&(pred, _)| exec_edges.contains(&(Some(pred), bb_id)));
            if incoming.len() != before {
                changed = true;
            }
        }

        // Fold non-Phi pure expressions (BinOp, UnOp, Cast) whose result became constant.
        for inst in &mut bb.insts[phi_end..] {
            let can_fold = matches!(
                inst,
                IrInst::BinOp { .. } | IrInst::UnOp { .. } | IrInst::Cast { .. }
            );
            if can_fold
                && let Some(dst) = inst.dst()
                && let Some(&Lat::Const(cv)) = val_state.get(dst as usize)
            {
                *inst = cv.to_inst(dst);
                changed = true;
            }
        }

        // Remove constant Phis and insert their materialized Const* after remaining Phis.
        if !promoted_consts.is_empty() {
            let mut idx = 0usize;
            bb.insts.retain(|_| {
                let keep = if idx < phi_end { keep_phi[idx] } else { true };
                idx += 1;
                keep
            });
            let new_phi_end = bb
                .insts
                .iter()
                .take_while(|i| matches!(i, IrInst::Phi { .. }))
                .count();
            bb.insts.splice(new_phi_end..new_phi_end, promoted_consts);
        }

        // Fold conditional Branch to Goto when condition lattice value is Const(Bool).
        if let Terminator::Branch {
            cond,
            then_bb,
            else_bb,
        } = bb.term
            && let Some(&Lat::Const(ConstVal::Bool(b))) = val_state.get(cond as usize)
        {
            bb.term = Terminator::Goto(if b { then_bb } else { else_bb });
            changed = true;
        }
    }

    if simplify_trivial_phis(func) {
        changed = true;
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify::verify_program;
    use crate::{BasicBlock, IrProgram};

    fn make_fn(blocks: Vec<BasicBlock>, next_val: u32) -> IrFunction {
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

    fn verify_prog(func: IrFunction) -> IrProgram {
        IrProgram {
            functions: vec![func],
            ..Default::default()
        }
    }

    #[test]
    fn sccp_folds_conditional_branch_and_phi() {
        let u8t = IrType::Int(IntType::U8);
        // bb0: c = (3 < 5) -> Branch(c, bb1, bb2)
        // bb1: v1 = 10 -> Goto(bb3)
        // bb2: v2 = 99 -> Goto(bb3)
        // bb3: v3 = Phi((bb1, v1), (bb2, v2)) -> Return(v3)
        let func = make_fn(
            vec![
                BasicBlock {
                    id: 0,
                    insts: vec![
                        IrInst::ConstInt {
                            dst: 0,
                            ty: u8t.clone(),
                            val: 3,
                        },
                        IrInst::ConstInt {
                            dst: 1,
                            ty: u8t.clone(),
                            val: 5,
                        },
                        IrInst::BinOp {
                            dst: 2,
                            op: IrBinOp::Lt,
                            mode: ArithMode::Checked,
                            ty: u8t.clone(),
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
                        val: 99,
                    }],
                    term: Terminator::Goto(3),
                },
                BasicBlock {
                    id: 3,
                    insts: vec![IrInst::Phi {
                        dst: 5,
                        ty: u8t,
                        incoming: vec![(1, 3), (2, 4)],
                    }],
                    term: Terminator::Return(Some(5)),
                },
            ],
            6,
        );

        let mut prog = verify_prog(func);
        assert!(sccp(&mut prog.functions[0]));
        verify_program(&prog).expect("SCCP output must verify");

        // Dead block bb2 removed, Branch in bb0 became Goto(1), Phi in bb3 folded to ConstInt(10).
        assert_eq!(prog.functions[0].blocks.len(), 3);
        assert!(!prog.functions[0].blocks.iter().any(|b| b.id == 2));
        assert!(matches!(prog.functions[0].blocks[0].term, Terminator::Goto(1)));
        let bb3 = prog.functions[0].blocks.iter().find(|b| b.id == 3).unwrap();
        assert!(matches!(
            bb3.insts.as_slice(),
            [IrInst::ConstInt { dst: 5, val: 10, .. }]
        ));
    }

    #[test]
    fn sccp_propagates_loop_invariant_constant_across_backedge() {
        let u8t = IrType::Int(IntType::U8);
        // bb0: v0 = 5 -> Goto(bb1)
        // bb1: v1 = Phi((bb0, v0), (bb2, v2)), cond = (v1 < 10) -> Branch(cond, bb2, bb3)
        // bb2: v2 = v1 (or 5 via v1 + 0) -> Goto(bb1)
        // bb3: Return(v1)
        // Optimistic Top lattice must recognize v1 == 5 and eliminate the dead loop condition!
        let func = make_fn(
            vec![
                BasicBlock {
                    id: 0,
                    insts: vec![
                        IrInst::ConstInt {
                            dst: 0,
                            ty: u8t.clone(),
                            val: 5,
                        },
                        IrInst::ConstInt {
                            dst: 1,
                            ty: u8t.clone(),
                            val: 10,
                        },
                        IrInst::ConstInt {
                            dst: 2,
                            ty: u8t.clone(),
                            val: 0,
                        },
                    ],
                    term: Terminator::Goto(1),
                },
                BasicBlock {
                    id: 1,
                    insts: vec![
                        IrInst::Phi {
                            dst: 3,
                            ty: u8t.clone(),
                            incoming: vec![(0, 0), (2, 5)],
                        },
                        IrInst::BinOp {
                            dst: 4,
                            op: IrBinOp::Lt,
                            mode: ArithMode::Checked,
                            ty: u8t.clone(),
                            lhs: 3,
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
                    insts: vec![IrInst::BinOp {
                        dst: 5,
                        op: IrBinOp::Add,
                        mode: ArithMode::Wrapping,
                        ty: u8t,
                        lhs: 3,
                        rhs: 2,
                        site_id: 0,
                    }],
                    term: Terminator::Goto(1),
                },
                BasicBlock {
                    id: 3,
                    insts: vec![],
                    term: Terminator::Return(Some(3)),
                },
            ],
            6,
        );

        let mut prog = verify_prog(func);
        assert!(sccp(&mut prog.functions[0]));
        verify_program(&prog).expect("SCCP loop constant propagation must verify");

        // bb1 branch (5 < 10) becomes Goto(2), bb3 is unreachable and eliminated, v3 = 5.
        assert!(!prog.functions[0].blocks.iter().any(|b| b.id == 3));
        let bb1 = prog.functions[0].blocks.iter().find(|b| b.id == 1).unwrap();
        assert!(matches!(bb1.term, Terminator::Goto(2)));
        assert!(bb1
            .insts
            .iter()
            .any(|i| matches!(i, IrInst::ConstInt { dst: 3, val: 5, .. })));
    }
}
