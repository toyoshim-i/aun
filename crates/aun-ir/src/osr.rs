//! Operator Strength Reduction (OSR) on basic induction variable multiplications
//! and constant shifts inside natural loops.

use crate::cfg::Cfg;
use crate::licm::ensure_loop_preheader;
use crate::opt::{apply_val_remap, truncate_int};
use crate::{ArithMode, BlockId, IntType, IrBinOp, IrFunction, IrInst, IrType, ValueId};
use std::collections::{BTreeSet, HashMap, HashSet};

#[derive(Clone, Copy)]
struct BasicIv {
    phi_dst: ValueId,
    ty: IntType,
    step: i64,
}

#[derive(Clone, Copy)]
enum SrKind {
    Mul(i64),
    Shl(u8),
}

impl SrKind {
    fn factor(self) -> i64 {
        match self {
            SrKind::Mul(k) => k,
            SrKind::Shl(shift) => 1i64 << shift,
        }
    }
}

struct SrCandidate {
    bb_id: BlockId,
    inst_idx: usize,
    dst: ValueId,
    biv: BasicIv,
    kind: SrKind,
}

/// Replaces loop induction-variable multiplications (`iv *% k`) and constant left shifts
/// (`iv <<% shift`) with incrementally updated shadow induction variables (`sr_iv +% (step * k)`).
pub fn osr(func: &mut IrFunction) -> bool {
    osr_with_config(func, true)
}

/// Same as `osr`, optionally disabling 8-bit (`u8`/`i8`) strength reduction for targets
/// with fast hardware `MUL AB` or tight 8-bit register pressure (e.g. MCS-51).
pub fn osr_with_config(func: &mut IrFunction, enable_u8: bool) -> bool {
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

        // Collect single-latch natural loops.
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

        let bb_map: HashMap<BlockId, usize> =
            func.blocks.iter().enumerate().map(|(i, b)| (b.id, i)).collect();

        // Collect constant integers and all BinOp definitions in `body` / function.
        let mut const_ints: HashMap<ValueId, (IntType, i64)> = HashMap::new();
        let mut loop_binops: HashMap<ValueId, (IrBinOp, ArithMode, IntType, ValueId, ValueId)> =
            HashMap::new();
        for bb in &func.blocks {
            let in_loop = body.contains(&bb.id);
            for inst in &bb.insts {
                match inst {
                    IrInst::ConstInt {
                        dst,
                        ty: IrType::Int(it),
                        val,
                    } => {
                        const_ints.insert(*dst, (*it, truncate_int(*val, *it)));
                    }
                    IrInst::BinOp {
                        dst,
                        op,
                        mode,
                        ty: IrType::Int(it),
                        lhs,
                        rhs,
                        ..
                    } if in_loop => {
                        loop_binops.insert(*dst, (*op, *mode, *it, *lhs, *rhs));
                    }
                    _ => {}
                }
            }
        }

        // Identify Basic Induction Variables (BIVs) at `header`.
        let mut bivs: HashMap<ValueId, BasicIv> = HashMap::new();
        for inst in &func.blocks[bb_map[&header]].insts {
            let IrInst::Phi {
                dst: iv,
                ty: IrType::Int(it),
                incoming,
            } = inst
            else {
                if !matches!(inst, IrInst::Phi { .. }) {
                    break;
                }
                continue;
            };
            let Some(&(_, next_iv)) = incoming.iter().find(|&&(p, _)| p == latch) else {
                continue;
            };
            if incoming
                .iter()
                .any(|&(p, _)| p != latch && body.contains(&p))
            {
                continue;
            }
            let Some(&(op, mode, it2, lhs, rhs)) = loop_binops.get(&next_iv) else {
                continue;
            };
            if it2 != *it || !matches!(mode, ArithMode::Wrapping | ArithMode::Checked) {
                continue;
            }
            let step = match op {
                IrBinOp::Add => {
                    if lhs == *iv {
                        const_ints
                            .get(&rhs)
                            .and_then(|&(cit, c)| (cit == *it).then_some(c))
                    } else if rhs == *iv {
                        const_ints
                            .get(&lhs)
                            .and_then(|&(cit, c)| (cit == *it).then_some(c))
                    } else {
                        None
                    }
                }
                IrBinOp::Sub => {
                    if lhs == *iv {
                        const_ints
                            .get(&rhs)
                            .and_then(|&(cit, c)| (cit == *it).then_some(c.wrapping_neg()))
                    } else {
                        None
                    }
                }
                _ => None,
            };
            if let Some(step) = step
                && truncate_int(step, *it) != 0
            {
                bivs.insert(
                    *iv,
                    BasicIv {
                        phi_dst: *iv,
                        ty: *it,
                        step,
                    },
                );
            }
        }
        if bivs.is_empty() {
            continue;
        }

        // Scan `body` for strength-reducible `iv *% k` or `iv <<% shift`.
        let mut candidates: Vec<SrCandidate> = Vec::new();
        for &bb_id in cfg.rpo() {
            if !body.contains(&bb_id) {
                continue;
            }
            let bb = &func.blocks[bb_map[&bb_id]];
            for (inst_idx, inst) in bb.insts.iter().enumerate() {
                let IrInst::BinOp {
                    dst,
                    op,
                    mode: ArithMode::Wrapping,
                    ty: IrType::Int(it),
                    lhs,
                    rhs,
                    ..
                } = inst
                else {
                    continue;
                };
                if !enable_u8 && matches!(it, IntType::U8 | IntType::I8) {
                    continue;
                }
                match op {
                    IrBinOp::Mul => {
                        if let Some(&biv) = bivs.get(lhs)
                            && biv.ty == *it
                            && let Some(&(cit, k)) = const_ints.get(rhs)
                            && cit == *it
                            && k != 0
                            && k != 1
                        {
                            candidates.push(SrCandidate {
                                bb_id,
                                inst_idx,
                                dst: *dst,
                                biv,
                                kind: SrKind::Mul(k),
                            });
                        } else if let Some(&biv) = bivs.get(rhs)
                            && biv.ty == *it
                            && let Some(&(cit, k)) = const_ints.get(lhs)
                            && cit == *it
                            && k != 0
                            && k != 1
                        {
                            candidates.push(SrCandidate {
                                bb_id,
                                inst_idx,
                                dst: *dst,
                                biv,
                                kind: SrKind::Mul(k),
                            });
                        }
                    }
                    IrBinOp::Shl => {
                        if let Some(&biv) = bivs.get(lhs)
                            && biv.ty == *it
                            && let Some(&(cit, shift)) = const_ints.get(rhs)
                            && cit == *it
                            && shift > 0
                            && shift < it.bit_width() as i64
                        {
                            candidates.push(SrCandidate {
                                bb_id,
                                inst_idx,
                                dst: *dst,
                                biv,
                                kind: SrKind::Shl(shift as u8),
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
        if candidates.is_empty() {
            continue;
        }

        let Some(pre_id) = ensure_loop_preheader(func, &cfg, header, &body) else {
            continue;
        };

        let bb_map: HashMap<BlockId, usize> =
            func.blocks.iter().enumerate().map(|(i, b)| (b.id, i)).collect();

        // Lookup each BIV's initial value from the collapsed `(pre_id, init_val)` edge.
        let mut biv_inits: HashMap<ValueId, ValueId> = HashMap::new();
        for inst in &func.blocks[bb_map[&header]].insts {
            let IrInst::Phi { dst, incoming, .. } = inst else {
                if !matches!(inst, IrInst::Phi { .. }) {
                    break;
                }
                continue;
            };
            if bivs.contains_key(dst)
                && let Some(&(_, init_val)) = incoming.iter().find(|&&(p, _)| p == pre_id)
            {
                biv_inits.insert(*dst, init_val);
            }
        }

        let mut remap: HashMap<ValueId, ValueId> = HashMap::new();
        let mut drop_set: HashSet<(BlockId, usize)> = HashSet::new();
        let mut pre_insts: Vec<IrInst> = Vec::new();
        let mut header_phis: Vec<IrInst> = Vec::new();
        let mut latch_insts: Vec<IrInst> = Vec::new();

        for cand in candidates {
            let Some(&init_val) = biv_inits.get(&cand.biv.phi_dst) else {
                continue;
            };
            let ty = IrType::Int(cand.biv.ty);
            let k = cand.kind.factor();

            // Compute sr_init in preheader.
            let sr_init = func.next_val;
            func.next_val += 1;
            if let Some(&(cit, init_c)) = const_ints.get(&init_val)
                && cit == cand.biv.ty
            {
                let v = truncate_int((init_c as i128 * k as i128) as i64, cand.biv.ty);
                pre_insts.push(IrInst::ConstInt {
                    dst: sr_init,
                    ty: ty.clone(),
                    val: v,
                });
            } else {
                let k_val = func.next_val;
                func.next_val += 1;
                match cand.kind {
                    SrKind::Mul(factor) => {
                        pre_insts.push(IrInst::ConstInt {
                            dst: k_val,
                            ty: ty.clone(),
                            val: factor,
                        });
                        pre_insts.push(IrInst::BinOp {
                            dst: sr_init,
                            op: IrBinOp::Mul,
                            mode: ArithMode::Wrapping,
                            ty: ty.clone(),
                            lhs: init_val,
                            rhs: k_val,
                            site_id: 0,
                        });
                    }
                    SrKind::Shl(shift) => {
                        pre_insts.push(IrInst::ConstInt {
                            dst: k_val,
                            ty: ty.clone(),
                            val: shift as i64,
                        });
                        pre_insts.push(IrInst::BinOp {
                            dst: sr_init,
                            op: IrBinOp::Shl,
                            mode: ArithMode::Wrapping,
                            ty: ty.clone(),
                            lhs: init_val,
                            rhs: k_val,
                            site_id: 0,
                        });
                    }
                }
            }

            let sr_iv = func.next_val;
            func.next_val += 1;
            let sr_step = func.next_val;
            func.next_val += 1;
            let sr_next = func.next_val;
            func.next_val += 1;

            header_phis.push(IrInst::Phi {
                dst: sr_iv,
                ty: ty.clone(),
                incoming: vec![(pre_id, sr_init), (latch, sr_next)],
            });

            let step_val =
                truncate_int((cand.biv.step as i128 * k as i128) as i64, cand.biv.ty);
            latch_insts.push(IrInst::ConstInt {
                dst: sr_step,
                ty: ty.clone(),
                val: step_val,
            });
            latch_insts.push(IrInst::BinOp {
                dst: sr_next,
                op: IrBinOp::Add,
                mode: ArithMode::Wrapping,
                ty,
                lhs: sr_iv,
                rhs: sr_step,
                site_id: 0,
            });

            remap.insert(cand.dst, sr_iv);
            drop_set.insert((cand.bb_id, cand.inst_idx));
        }

        if remap.is_empty() {
            continue;
        }

        func.blocks[bb_map[&pre_id]].insts.extend(pre_insts);
        func.blocks[bb_map[&header]].insts.splice(0..0, header_phis);
        func.blocks[bb_map[&latch]].insts.extend(latch_insts);

        for bb in &mut func.blocks {
            let bb_id = bb.id;
            let mut idx = 0usize;
            bb.insts.retain(|_| {
                let keep = !drop_set.contains(&(bb_id, idx));
                idx += 1;
                keep
            });
        }
        apply_val_remap(func, &remap);
        any_changed = true;
    }

    any_changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify::verify_program;
    use crate::{BasicBlock, IrProgram, Terminator};

    fn verify_prog(func: IrFunction) -> IrProgram {
        IrProgram {
            functions: vec![func],
            ..Default::default()
        }
    }

    fn loop_fn(op: IrBinOp, factor: i64) -> IrProgram {
        let u8t = IrType::Int(IntType::U8);
        // bb0: init=0, limit=5, step=1, factor -> Goto(bb1)
        // bb1: iv = Phi((bb0, 0), (bb2, next_iv)), cond = iv < limit -> Branch(cond, bb2, bb3)
        // bb2: prod = iv (op) factor; next_iv = iv + step -> Goto(bb1)
        // bb3: Return(iv)
        let func = IrFunction {
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
            blocks: vec![
                BasicBlock {
                    id: 0,
                    insts: vec![
                        IrInst::ConstInt {
                            dst: 0,
                            ty: u8t.clone(),
                            val: 0,
                        },
                        IrInst::ConstInt {
                            dst: 1,
                            ty: u8t.clone(),
                            val: 5,
                        },
                        IrInst::ConstInt {
                            dst: 2,
                            ty: u8t.clone(),
                            val: 1,
                        },
                        IrInst::ConstInt {
                            dst: 3,
                            ty: u8t.clone(),
                            val: factor,
                        },
                    ],
                    term: Terminator::Goto(1),
                },
                BasicBlock {
                    id: 1,
                    insts: vec![
                        IrInst::Phi {
                            dst: 4,
                            ty: u8t.clone(),
                            incoming: vec![(0, 0), (2, 7)],
                        },
                        IrInst::BinOp {
                            dst: 5,
                            op: IrBinOp::Lt,
                            mode: ArithMode::Checked,
                            ty: u8t.clone(),
                            lhs: 4,
                            rhs: 1,
                            site_id: 0,
                        },
                    ],
                    term: Terminator::Branch {
                        cond: 5,
                        then_bb: 2,
                        else_bb: 3,
                    },
                },
                BasicBlock {
                    id: 2,
                    insts: vec![
                        IrInst::BinOp {
                            dst: 6,
                            op,
                            mode: ArithMode::Wrapping,
                            ty: u8t.clone(),
                            lhs: 4,
                            rhs: 3,
                            site_id: 0,
                        },
                        IrInst::BinOp {
                            dst: 7,
                            op: IrBinOp::Add,
                            mode: ArithMode::Wrapping,
                            ty: u8t,
                            lhs: 4,
                            rhs: 2,
                            site_id: 0,
                        },
                        IrInst::PortOut { port: 0, val: 6 },
                    ],
                    term: Terminator::Goto(1),
                },
                BasicBlock {
                    id: 3,
                    insts: vec![],
                    term: Terminator::Return(Some(4)),
                },
            ],
            next_val: 8,
        };
        verify_prog(func)
    }

    #[test]
    fn osr_reduces_loop_mul_to_incremental_add() {
        let mut prog = loop_fn(IrBinOp::Mul, 3);
        assert!(osr(&mut prog.functions[0]));
        verify_program(&prog).expect("OSR Mul output must verify");

        // No Mul remains in latch bb2; sr_iv Phi is added in bb1 and step=3 add is in bb2.
        assert!(!prog.functions[0].blocks[2]
            .insts
            .iter()
            .any(|i| matches!(i, IrInst::BinOp { op: IrBinOp::Mul, .. })));
        assert_eq!(
            prog.functions[0].blocks[1]
                .insts
                .iter()
                .filter(|i| matches!(i, IrInst::Phi { .. }))
                .count(),
            2
        );
    }

    #[test]
    fn osr_reduces_loop_shl_to_incremental_add() {
        let mut prog = loop_fn(IrBinOp::Shl, 2);
        assert!(osr(&mut prog.functions[0]));
        verify_program(&prog).expect("OSR Shl output must verify");

        assert!(!prog.functions[0].blocks[2]
            .insts
            .iter()
            .any(|i| matches!(i, IrInst::BinOp { op: IrBinOp::Shl, .. })));
        // Step constant for (1 << 2) * 1 == 4 added in bb2.
        assert!(prog.functions[0].blocks[2]
            .insts
            .iter()
            .any(|i| matches!(i, IrInst::ConstInt { val: 4, .. })));
    }
}
