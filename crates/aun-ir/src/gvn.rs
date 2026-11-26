//! Scoped Dominator-Tree Global Value Numbering / CSE (Briggs, Cooper, Simpson 1997)
//! with local algebraic identity simplifications for `aun-ir` SSA form.

use crate::cfg::Cfg;
use crate::opt::{apply_val_remap, truncate_int};
use crate::sccp::ConstVal;
use crate::ssa::simplify_trivial_phis;
use crate::{BlockId, IntType, IrBinOp, IrFunction, IrInst, IrType, IrUnOp, ValueId};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ExprKey {
    ConstInt(IrType, i64),
    ConstBool(bool),
    BinOp {
        op: IrBinOp,
        mode: crate::ArithMode,
        ty: IrType,
        lhs: ValueId,
        rhs: ValueId,
    },
    UnOp {
        op: IrUnOp,
        mode: crate::ArithMode,
        ty: IrType,
        val: ValueId,
    },
    Cast {
        val: ValueId,
        from_ty: IrType,
        to_ty: IrType,
    },
    FixedInt {
        val: ValueId,
        base: IntType,
        frac_bits: u8,
    },
    AddrLocal(u32),
    AddrGlobal(u32),
    FieldAddr {
        base: ValueId,
        byte_offset: u16,
    },
    Phi {
        bb_id: BlockId,
        ty: IrType,
        incoming: Vec<(BlockId, ValueId)>,
    },
}

fn is_commutative(op: IrBinOp) -> bool {
    matches!(
        op,
        IrBinOp::Add
            | IrBinOp::Mul
            | IrBinOp::BitAnd
            | IrBinOp::BitOr
            | IrBinOp::BitXor
            | IrBinOp::Eq
            | IrBinOp::Ne
    )
}

enum IdentityResult {
    ReuseVal(ValueId),
    Const(ConstVal),
}

fn try_algebraic_identity(
    op: IrBinOp,
    ty: &IrType,
    lhs: ValueId,
    rhs: ValueId,
    consts: &HashMap<ValueId, ConstVal>,
) -> Option<IdentityResult> {
    let IrType::Int(it) = *ty else {
        return None;
    };
    let lc = match consts.get(&lhs) {
        Some(ConstVal::Int(_, v)) => Some(*v),
        _ => None,
    };
    let rc = match consts.get(&rhs) {
        Some(ConstVal::Int(_, v)) => Some(*v),
        _ => None,
    };

    match op {
        IrBinOp::Add => {
            if rc == Some(0) {
                return Some(IdentityResult::ReuseVal(lhs));
            }
            if lc == Some(0) {
                return Some(IdentityResult::ReuseVal(rhs));
            }
        }
        IrBinOp::Sub => {
            if rc == Some(0) {
                return Some(IdentityResult::ReuseVal(lhs));
            }
            if lhs == rhs {
                return Some(IdentityResult::Const(ConstVal::Int(it, 0)));
            }
        }
        IrBinOp::Mul => {
            if rc == Some(1) {
                return Some(IdentityResult::ReuseVal(lhs));
            }
            if lc == Some(1) {
                return Some(IdentityResult::ReuseVal(rhs));
            }
            if lc == Some(0) || rc == Some(0) {
                return Some(IdentityResult::Const(ConstVal::Int(it, 0)));
            }
        }
        IrBinOp::BitAnd => {
            if lhs == rhs {
                return Some(IdentityResult::ReuseVal(lhs));
            }
            if lc == Some(0) || rc == Some(0) {
                return Some(IdentityResult::Const(ConstVal::Int(it, 0)));
            }
        }
        IrBinOp::BitOr => {
            if lhs == rhs || rc == Some(0) {
                return Some(IdentityResult::ReuseVal(lhs));
            }
            if lc == Some(0) {
                return Some(IdentityResult::ReuseVal(rhs));
            }
        }
        IrBinOp::BitXor => {
            if rc == Some(0) {
                return Some(IdentityResult::ReuseVal(lhs));
            }
            if lc == Some(0) {
                return Some(IdentityResult::ReuseVal(rhs));
            }
            if lhs == rhs {
                return Some(IdentityResult::Const(ConstVal::Int(it, 0)));
            }
        }
        IrBinOp::Shl | IrBinOp::Shr => {
            if rc == Some(0) {
                return Some(IdentityResult::ReuseVal(lhs));
            }
        }
        IrBinOp::Eq | IrBinOp::Le | IrBinOp::Ge if lhs == rhs => {
            return Some(IdentityResult::Const(ConstVal::Bool(true)));
        }
        IrBinOp::Ne | IrBinOp::Lt | IrBinOp::Gt if lhs == rhs => {
            return Some(IdentityResult::Const(ConstVal::Bool(false)));
        }
        _ => {}
    }
    None
}

/// Runs scoped dominator-tree Global Value Numbering / CSE on `func`.
pub fn gvn(func: &mut IrFunction) -> bool {
    if func.blocks.is_empty() {
        return false;
    }

    let cfg = Cfg::build(func);
    let Some(entry_id) = cfg.entry_block() else {
        return false;
    };
    let bb_map: HashMap<BlockId, usize> =
        func.blocks.iter().enumerate().map(|(i, b)| (b.id, i)).collect();

    struct GvnCtx<'a> {
        func: &'a mut IrFunction,
        cfg: &'a Cfg,
        bb_map: &'a HashMap<BlockId, usize>,
        table: HashMap<ExprKey, ValueId>,
        consts: HashMap<ValueId, ConstVal>,
        remap: HashMap<ValueId, ValueId>,
        drops: HashMap<BlockId, Vec<bool>>,
        changed: bool,
    }

    fn resolve(mut v: ValueId, remap: &HashMap<ValueId, ValueId>) -> ValueId {
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

    fn insert_expr(
        table: &mut HashMap<ExprKey, ValueId>,
        scope_keys: &mut Vec<ExprKey>,
        key: ExprKey,
        dst: ValueId,
    ) -> Option<ValueId> {
        if let Some(&leader) = table.get(&key) {
            Some(leader)
        } else {
            table.insert(key.clone(), dst);
            scope_keys.push(key);
            None
        }
    }

    fn visit_bb(ctx: &mut GvnCtx<'_>, bb_id: BlockId) {
        let mut scope_keys: Vec<ExprKey> = Vec::new();
        let bb_idx = ctx.bb_map[&bb_id];
        let inst_len = ctx.func.blocks[bb_idx].insts.len();
        let mut drop = vec![false; inst_len];

        for idx in 0..inst_len {
            let inst = &mut ctx.func.blocks[bb_idx].insts[idx];
            // Rewrite uses in this non-Phi instruction immediately from currently active leaders.
            if !matches!(inst, IrInst::Phi { .. }) {
                inst.for_each_use_mut(|v| *v = resolve(*v, &ctx.remap));
            }

            let mut maybe_key: Option<(ValueId, ExprKey)> = None;
            let mut identity_const: Option<(ValueId, ConstVal)> = None;
            let mut direct_alias: Option<(ValueId, ValueId)> = None;

            match inst {
                IrInst::ConstInt { dst, ty, val } => {
                    let norm_val = match ty {
                        IrType::Int(it) => {
                            let v = truncate_int(*val, *it);
                            ctx.consts.insert(*dst, ConstVal::Int(*it, v));
                            v
                        }
                        _ => *val,
                    };
                    maybe_key = Some((*dst, ExprKey::ConstInt(ty.clone(), norm_val)));
                }
                IrInst::ConstBool { dst, val } => {
                    ctx.consts.insert(*dst, ConstVal::Bool(*val));
                    maybe_key = Some((*dst, ExprKey::ConstBool(*val)));
                }
                IrInst::Cast {
                    dst,
                    val,
                    from_ty,
                    to_ty,
                } => {
                    if from_ty == to_ty {
                        direct_alias = Some((*dst, *val));
                    } else {
                        maybe_key = Some((
                            *dst,
                            ExprKey::Cast {
                                val: *val,
                                from_ty: from_ty.clone(),
                                to_ty: to_ty.clone(),
                            },
                        ));
                    }
                }
                IrInst::FixedInt {
                    dst,
                    val,
                    base,
                    frac_bits,
                } => {
                    maybe_key = Some((
                        *dst,
                        ExprKey::FixedInt {
                            val: *val,
                            base: *base,
                            frac_bits: *frac_bits,
                        },
                    ));
                }
                IrInst::AddrLocal { dst, slot } => {
                    maybe_key = Some((*dst, ExprKey::AddrLocal(*slot)));
                }
                IrInst::AddrGlobal { dst, global } => {
                    maybe_key = Some((*dst, ExprKey::AddrGlobal(*global)));
                }
                IrInst::FieldAddr {
                    dst,
                    base,
                    byte_offset,
                } => {
                    maybe_key = Some((
                        *dst,
                        ExprKey::FieldAddr {
                            base: *base,
                            byte_offset: *byte_offset,
                        },
                    ));
                }
                IrInst::UnOp {
                    dst,
                    op,
                    mode,
                    ty,
                    val,
                    ..
                } => {
                    if *op == IrUnOp::Not || *mode != crate::ArithMode::Checked {
                        maybe_key = Some((
                            *dst,
                            ExprKey::UnOp {
                                op: *op,
                                mode: *mode,
                                ty: ty.clone(),
                                val: *val,
                            },
                        ));
                    }
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
                    if let Some(id_res) = try_algebraic_identity(*op, ty, *lhs, *rhs, &ctx.consts) {
                        match id_res {
                            IdentityResult::ReuseVal(rep) => {
                                direct_alias = Some((*dst, rep));
                            }
                            IdentityResult::Const(cv) => {
                                identity_const = Some((*dst, cv));
                            }
                        }
                    } else {
                        let non_trapping = matches!(
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
                        ) || (*mode != crate::ArithMode::Checked
                            && !matches!(op, IrBinOp::Div | IrBinOp::Rem));
                        if non_trapping {
                            let (l, r) = if is_commutative(*op) {
                                ((*lhs).min(*rhs), (*lhs).max(*rhs))
                            } else {
                                (*lhs, *rhs)
                            };
                            maybe_key = Some((
                                *dst,
                                ExprKey::BinOp {
                                    op: *op,
                                    mode: *mode,
                                    ty: ty.clone(),
                                    lhs: l,
                                    rhs: r,
                                },
                            ));
                        }
                    }
                }
                IrInst::Phi { dst, ty, incoming } => {
                    let mut sorted = incoming.clone();
                    sorted.sort_by_key(|&(pred, _)| pred);
                    maybe_key = Some((
                        *dst,
                        ExprKey::Phi {
                            bb_id,
                            ty: ty.clone(),
                            incoming: sorted,
                        },
                    ));
                }
                _ => {}
            }

            if let Some((dst, rep)) = direct_alias {
                ctx.remap.insert(dst, rep);
                drop[idx] = true;
                ctx.changed = true;
            } else if let Some((dst, cv)) = identity_const {
                ctx.consts.insert(dst, cv);
                let c_key = match cv {
                    ConstVal::Int(it, v) => ExprKey::ConstInt(IrType::Int(it), v),
                    ConstVal::Bool(b) => ExprKey::ConstBool(b),
                };
                if let Some(leader) = insert_expr(&mut ctx.table, &mut scope_keys, c_key, dst) {
                    ctx.remap.insert(dst, leader);
                    drop[idx] = true;
                } else {
                    *inst = cv.to_inst(dst);
                }
                ctx.changed = true;
            } else if let Some((dst, key)) = maybe_key
                && let Some(leader) = insert_expr(&mut ctx.table, &mut scope_keys, key, dst)
            {
                ctx.remap.insert(dst, leader);
                drop[idx] = true;
                ctx.changed = true;
            }
        }

        ctx.drops.insert(bb_id, drop);

        let children: Vec<BlockId> = ctx.cfg.dom_children(bb_id).to_vec();
        for child in children {
            visit_bb(ctx, child);
        }

        for k in scope_keys {
            ctx.table.remove(&k);
        }
    }

    let mut ctx = GvnCtx {
        func,
        cfg: &cfg,
        bb_map: &bb_map,
        table: HashMap::new(),
        consts: HashMap::new(),
        remap: HashMap::new(),
        drops: HashMap::new(),
        changed: false,
    };
    visit_bb(&mut ctx, entry_id);

    let changed = ctx.changed;
    let drops = ctx.drops;
    let remap = ctx.remap;
    if !changed {
        return false;
    }

    for bb in &mut func.blocks {
        if let Some(drop) = drops.get(&bb.id) {
            let mut idx = 0usize;
            bb.insts.retain(|_| {
                let keep = !drop[idx];
                idx += 1;
                keep
            });
        }
    }

    apply_val_remap(func, &remap);
    simplify_trivial_phis(func);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify::verify_program;
    use crate::{ArithMode, BasicBlock, IrProgram, Terminator};

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
            param_count: 1,
            locals: vec![crate::LocalSlot {
                id: 0,
                name: "x".into(),
                ty: IrType::Int(IntType::U8),
                mutable: false,
            }],
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
    fn gvn_eliminates_commutative_cse_and_field_addr_across_domtree() {
        let u8t = IrType::Int(IntType::U8);
        // bb0: x = LoadLocal(0), y = LoadLocal(0), addr0 = AddrLocal(0), fa1 = FieldAddr(addr0, 2)
        //      sum1 = x +% y -> Branch(cond, bb1, bb2)
        // bb1: sum2 = y +% x (commutative CSE with sum1!), fa2 = FieldAddr(addr0, 2) -> Return(sum2)
        // bb2: Return(sum1)
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
                        IrInst::ConstInt {
                            dst: 1,
                            ty: u8t.clone(),
                            val: 3,
                        },
                        IrInst::AddrLocal { dst: 2, slot: 0 },
                        IrInst::FieldAddr {
                            dst: 3,
                            base: 2,
                            byte_offset: 2,
                        },
                        IrInst::BinOp {
                            dst: 4,
                            op: IrBinOp::Add,
                            mode: ArithMode::Wrapping,
                            ty: u8t.clone(),
                            lhs: 0,
                            rhs: 1,
                            site_id: 0,
                        },
                        IrInst::ConstBool { dst: 5, val: true },
                    ],
                    term: Terminator::Branch {
                        cond: 5,
                        then_bb: 1,
                        else_bb: 2,
                    },
                },
                BasicBlock {
                    id: 1,
                    insts: vec![
                        IrInst::BinOp {
                            dst: 6,
                            op: IrBinOp::Add,
                            mode: ArithMode::Wrapping,
                            ty: u8t.clone(),
                            lhs: 1,
                            rhs: 0,
                            site_id: 0,
                        },
                        IrInst::FieldAddr {
                            dst: 7,
                            base: 2,
                            byte_offset: 2,
                        },
                        IrInst::StorePtr {
                            addr: 7,
                            val: 6,
                            ty: u8t,
                        },
                    ],
                    term: Terminator::Return(Some(6)),
                },
                BasicBlock {
                    id: 2,
                    insts: vec![],
                    term: Terminator::Return(Some(4)),
                },
            ],
            8,
        );

        let mut prog = verify_prog(func);
        assert!(gvn(&mut prog.functions[0]));
        verify_program(&prog).expect("GVN output must verify");

        // bb1 should have only StorePtr(addr=3, val=4) and Return(4).
        let bb1 = &prog.functions[0].blocks[1];
        assert_eq!(bb1.insts.len(), 1);
        assert!(matches!(
            bb1.insts[0],
            IrInst::StorePtr { addr: 3, val: 4, .. }
        ));
        assert!(matches!(bb1.term, Terminator::Return(Some(4))));
    }

    #[test]
    fn gvn_does_not_leak_across_sibling_branches() {
        let u8t = IrType::Int(IntType::U8);
        // bb0 -> bb1 | bb2; bb1 -> bb3; bb2 -> bb3
        // bb1 computes v3 = v0 +% v1; bb2 computes v4 = v0 +% v1 (sibling -> MUST NOT CSE with v3!)
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
                        IrInst::ConstInt {
                            dst: 1,
                            ty: u8t.clone(),
                            val: 4,
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
                    insts: vec![IrInst::BinOp {
                        dst: 3,
                        op: IrBinOp::Add,
                        mode: ArithMode::Wrapping,
                        ty: u8t.clone(),
                        lhs: 0,
                        rhs: 1,
                        site_id: 0,
                    }],
                    term: Terminator::Goto(3),
                },
                BasicBlock {
                    id: 2,
                    insts: vec![IrInst::BinOp {
                        dst: 4,
                        op: IrBinOp::Add,
                        mode: ArithMode::Wrapping,
                        ty: u8t.clone(),
                        lhs: 0,
                        rhs: 1,
                        site_id: 0,
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
        assert!(!gvn(&mut prog.functions[0]));
        verify_program(&prog).expect("sibling branches must remain valid");
        assert_eq!(prog.functions[0].blocks[2].insts.len(), 1);
    }

    #[test]
    fn gvn_simplifies_algebraic_identities() {
        let u8t = IrType::Int(IntType::U8);
        // v1 = v0 + 0 (alias v0); v2 = v0 - v0 (0); v3 = (v1 == v0) (true); return v2 + v3? return v2.
        let func = base_fn(
            vec![BasicBlock {
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
                        val: 0,
                    },
                    IrInst::BinOp {
                        dst: 2,
                        op: IrBinOp::Add,
                        mode: ArithMode::Checked,
                        ty: u8t.clone(),
                        lhs: 0,
                        rhs: 1,
                        site_id: 0,
                    },
                    IrInst::BinOp {
                        dst: 3,
                        op: IrBinOp::Sub,
                        mode: ArithMode::Checked,
                        ty: u8t.clone(),
                        lhs: 2,
                        rhs: 0,
                        site_id: 0,
                    },
                    IrInst::BinOp {
                        dst: 4,
                        op: IrBinOp::Eq,
                        mode: ArithMode::Checked,
                        ty: u8t,
                        lhs: 2,
                        rhs: 0,
                        site_id: 0,
                    },
                ],
                term: Terminator::Return(Some(3)),
            }],
            5,
        );

        let mut prog = verify_prog(func);
        assert!(gvn(&mut prog.functions[0]));
        verify_program(&prog).expect("algebraic identity output must verify");

        // v2 -> v0, v3 = v0 - v0 -> constant 0 -> CSE'd with v1 (ConstInt 0), v4 -> ConstBool(true).
        assert!(matches!(prog.functions[0].blocks[0].term, Terminator::Return(Some(1))));
        assert!(prog.functions[0].blocks[0]
            .insts
            .iter()
            .any(|i| matches!(i, IrInst::ConstBool { dst: 4, val: true })));
    }
}
