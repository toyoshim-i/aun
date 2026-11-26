//! LIR machine outliner for Z80 code-size compression.
//!
//! Identifies repeated straight-line `LirInst` sequences within single basic blocks,
//! extracts them into shared `__aun_out_<id>` helper routines when net encoded bytes
//! (accounting for `call` overhead + helper `ret`/tail-`jp`) decrease, and collapses
//! trailing `call target; ret` sequences into `jp target`.

use crate::lir::{LirInst, LirItem, Reg16};
use std::collections::HashMap;

fn is_outlinable(inst: &LirInst) -> bool {
    match inst {
        LirInst::Push(_)
        | LirInst::Pop(_)
        | LirInst::LdSp(_)
        | LirInst::Jp { .. }
        | LirInst::JpInd(_)
        | LirInst::Jr { .. }
        | LirInst::Djnz(_)
        | LirInst::Ret(_)
        | LirInst::Reti
        | LirInst::Retn
        | LirInst::Halt
        | LirInst::Di
        | LirInst::Ei
        | LirInst::InAImm(_)
        | LirInst::InAC
        | LirInst::OutImmA(_)
        | LirInst::OutCA
        | LirInst::Raw(_) => false,
        LirInst::Call { cond: Some(_), .. } => false,
        LirInst::Call { cond: None, .. } => true,
        LirInst::Ld16Imm { dst, .. }
        | LirInst::Ld16Mem { dst, .. }
        | LirInst::Inc16(dst)
        | LirInst::Dec16(dst) => *dst != Reg16::SP,
        LirInst::St16Mem { src, .. } => *src != Reg16::SP,
        LirInst::Add16 { dst, src } => *dst != Reg16::SP && *src != Reg16::SP,
        LirInst::SbcHl { src } | LirInst::AdcHl { src } => *src != Reg16::SP,
        _ => true,
    }
}

/// Outlines profitable repeated LIR instruction sequences into `__aun_out_<n>` helpers
/// and returns the number of helper routines generated.
pub fn outline_lir(items: &mut Vec<LirItem>) -> usize {
    const MIN_LEN: usize = 2;
    const MAX_LEN: usize = 12;
    const MAX_ITERS: usize = 128;

    let mut helpers: Vec<LirItem> = Vec::new();
    let mut helper_id = 0usize;

    for _ in 0..MAX_ITERS {
        // Partition `items` into contiguous runs of outlinable `LirItem::Inst` elements.
        let mut runs: Vec<(usize, Vec<LirInst>)> = Vec::new();
        let mut idx = 0usize;
        while idx < items.len() {
            if let LirItem::Inst(inst) = &items[idx]
                && is_outlinable(inst)
            {
                let start = idx;
                let mut seq = Vec::new();
                while idx < items.len()
                    && let LirItem::Inst(i2) = &items[idx]
                    && is_outlinable(i2)
                {
                    seq.push(i2.clone());
                    idx += 1;
                }
                if seq.len() >= MIN_LEN {
                    runs.push((start, seq));
                }
            } else {
                idx += 1;
            }
        }

        if runs.is_empty() {
            break;
        }

        // Count non-overlapping occurrences of each sub-slice of length `MIN_LEN..=MAX_LEN`.
        let mut occs: HashMap<&[LirInst], Vec<usize>> = HashMap::new();
        for (run_start, run_insts) in &runs {
            let rlen = run_insts.len();
            let max_w = MAX_LEN.min(rlen);
            for w in MIN_LEN..=max_w {
                for off in 0..=(rlen - w) {
                    let slice = &run_insts[off..(off + w)];
                    let item_pos = *run_start + off;
                    let entry = occs.entry(slice).or_default();
                    if entry.last().is_none_or(|&prev| prev + w <= item_pos) {
                        entry.push(item_pos);
                    }
                }
            }
        }

        struct BestCand {
            savings: isize,
            seq_bytes: usize,
            positions: Vec<usize>,
            seq: Vec<LirInst>,
        }
        let mut best: Option<BestCand> = None;

        for (slice, positions) in occs {
            if positions.len() < 2 {
                continue;
            }
            let seq_bytes: usize = slice.iter().map(LirInst::byte_len).sum();
            let ends_with_call = matches!(slice.last(), Some(LirInst::Call { cond: None, .. }));
            let helper_bytes = if ends_with_call {
                seq_bytes
            } else {
                seq_bytes + 1
            };
            let count = positions.len();
            let savings = (count * seq_bytes) as isize - (count * 3 + helper_bytes) as isize;
            if savings < 2 {
                continue;
            }
            let better = match &best {
                None => true,
                Some(b) => {
                    (savings, seq_bytes, std::cmp::Reverse(positions[0]))
                        > (b.savings, b.seq_bytes, std::cmp::Reverse(b.positions[0]))
                }
            };
            if better {
                best = Some(BestCand {
                    savings,
                    seq_bytes,
                    positions,
                    seq: slice.to_vec(),
                });
            }
        }

        let Some(chosen) = best else {
            break;
        };

        let helper_name = format!("__aun_out_{helper_id}");
        helper_id += 1;
        let len = chosen.seq.len();

        // Replace non-overlapping occurrences in `items`.
        let mut next_items: Vec<LirItem> = Vec::with_capacity(items.len());
        let mut cur = 0usize;
        for pos in chosen.positions {
            next_items.extend_from_slice(&items[cur..pos]);
            next_items.push(LirItem::Inst(LirInst::Call {
                cond: None,
                target: helper_name.clone(),
            }));
            cur = pos + len;
        }
        next_items.extend_from_slice(&items[cur..]);
        *items = next_items;

        // Emit outlined helper body.
        helpers.push(LirItem::Blank);
        helpers.push(LirItem::Label(helper_name));
        if let Some(LirInst::Call {
            cond: None,
            target,
        }) = chosen.seq.last()
        {
            for inst in &chosen.seq[..(len - 1)] {
                helpers.push(LirItem::Inst(inst.clone()));
            }
            helpers.push(LirItem::Inst(LirInst::Jp {
                cond: None,
                target: target.clone(),
            }));
        } else {
            for inst in chosen.seq {
                helpers.push(LirItem::Inst(inst));
            }
            helpers.push(LirItem::Inst(LirInst::Ret(None)));
        }
    }

    if !helpers.is_empty() {
        items.extend(helpers);
    }

    // Collapse trailing `call target; ret` -> `jp target`.
    collapse_tail_calls_lir(items);

    helper_id
}

fn collapse_tail_calls_lir(items: &mut Vec<LirItem>) {
    let mut drop = vec![false; items.len()];
    for i in 0..items.len() {
        if drop[i] {
            continue;
        }
        if let LirItem::Inst(LirInst::Call {
            cond: None,
            target,
        }) = &items[i]
        {
            let target = target.clone();
            let mut j = i + 1;
            while j < items.len() && matches!(items[j], LirItem::Blank | LirItem::Comment(_)) {
                j += 1;
            }
            if j < items.len() && matches!(items[j], LirItem::Inst(LirInst::Ret(None))) {
                items[i] = LirItem::Inst(LirInst::Jp {
                    cond: None,
                    target,
                });
                drop[j] = true;
            }
        }
    }
    if drop.iter().any(|&d| d) {
        let mut idx = 0usize;
        items.retain(|_| {
            let keep = !drop[idx];
            idx += 1;
            keep
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lir::{parse_zilog_to_lir, LirItem};

    #[test]
    fn outlines_repeated_sequences_and_tail_calls_helpers() {
        let asm = "\
fn_a:
    ld l, (ix + 1)
    ld h, (ix + 2)
    inc hl
    inc hl
    ld (ix + 1), l
    ld (ix + 2), h
    ret
fn_b:
    ld l, (ix + 1)
    ld h, (ix + 2)
    inc hl
    inc hl
    ld (ix + 1), l
    ld (ix + 2), h
    ld a, 5
    ret
";
        let mut items = parse_zilog_to_lir(asm);
        let helpers = outline_lir(&mut items);
        assert!(helpers >= 1);
        assert!(items.iter().any(|it| matches!(it, LirItem::Label(l) if l == "__aun_out_0")));
        // fn_a's trailing `call __aun_out_0; ret` collapses into `jp __aun_out_0`!
        assert!(items.iter().any(|it| matches!(
            it,
            LirItem::Inst(LirInst::Jp { cond: None, target }) if target == "__aun_out_0"
        )));
    }

    #[test]
    fn never_outlines_push_pop_branches_or_port_io() {
        let asm = "\
fn_a:
    push hl
    ld a, 1
    out (0x44), a
    pop hl
    ret
fn_b:
    push hl
    ld a, 1
    out (0x44), a
    pop hl
    ret
";
        let mut items = parse_zilog_to_lir(asm);
        let helpers = outline_lir(&mut items);
        assert_eq!(helpers, 0);
    }
}
