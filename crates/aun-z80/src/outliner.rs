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
        LirInst::LdSp(_)
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
        LirInst::Call { cond: None, target } => !target.starts_with("__aun_out_"),
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

fn is_stack_balanced(slice: &[LirInst]) -> bool {
    let mut depth: i32 = 0;
    for inst in slice {
        match inst {
            LirInst::Push(_) => depth += 1,
            LirInst::Pop(_) => {
                depth -= 1;
                if depth < 0 {
                    return false;
                }
            }
            _ => {}
        }
    }
    depth == 0
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
        // Identify natural loop intervals via intra-function backward branches (`jp`/`jr`/`djnz`
        // targeting an earlier label) so hot-loop sequences require higher net savings before
        // paying +27 T per `call` + `ret`.
        let mut in_loop = vec![false; items.len()];
        {
            let mut fn_labels: HashMap<&str, usize> = HashMap::new();
            for (idx, it) in items.iter().enumerate() {
                match it {
                    LirItem::Label(l) => {
                        if !l.contains("_bb") && !crate::lir::is_bb_label(l) {
                            fn_labels.clear();
                        }
                        fn_labels.insert(l.as_str(), idx);
                    }
                    LirItem::Inst(
                        LirInst::Jp { target, .. }
                        | LirInst::Jr { target, .. }
                        | LirInst::Djnz(target),
                    ) => {
                        if let Some(&t_idx) = fn_labels.get(target.as_str())
                            && t_idx < idx
                        {
                            for slot in &mut in_loop[t_idx..=idx] {
                                *slot = true;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        // Partition `items` into contiguous runs of outlinable `LirItem::Inst` elements.
        let mut runs: Vec<(usize, Vec<LirInst>)> = Vec::new();
        let mut idx = 0usize;
        let mut in_inline_asm = false;
        while idx < items.len() {
            if let LirItem::Comment(c) = &items[idx] {
                let t = c.trim();
                if t == "; @aun:asm_begin" {
                    in_inline_asm = true;
                } else if t == "; @aun:asm_end" {
                    in_inline_asm = false;
                }
                idx += 1;
            } else if !in_inline_asm
                && let LirItem::Inst(inst) = &items[idx]
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
                    if !is_stack_balanced(slice) {
                        continue;
                    }
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
            let any_in_loop = positions
                .iter()
                .any(|&p| p < in_loop.len() && in_loop[p..(p + slice.len()).min(in_loop.len())].iter().any(|&b| b));
            let min_savings = if any_in_loop { 6 } else { 2 };
            if savings < min_savings {
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

/// Computes the worst-case additional stack bytes (2 per nested `call __aun_out_*` return
/// address, counting `jp __aun_out_*` tail calls as +0) introduced by outlining.
pub fn max_outlined_stack_bytes(items: &[LirItem]) -> u16 {
    let mut helper_depth: std::collections::HashMap<String, u16> = std::collections::HashMap::new();

    // Pass 1 (fixed point): compute max stack depth within each `__aun_out_*` helper body
    // (note that helpers are appended at the end of `items`, after user functions).
    let mut changed = true;
    while changed {
        changed = false;
        let mut cur_helper: Option<String> = None;
        let mut cur_max: u16 = 0;
        let flush = |h: Option<String>,
                     mx: u16,
                     map: &mut std::collections::HashMap<String, u16>,
                     ch: &mut bool| {
            if let Some(name) = h
                && map.get(&name).copied().unwrap_or(0) < mx
            {
                map.insert(name, mx);
                *ch = true;
            }
        };
        for it in items {
            match it {
                LirItem::Label(l) => {
                    flush(cur_helper.take(), cur_max, &mut helper_depth, &mut changed);
                    if l.starts_with("__aun_out_") {
                        cur_helper = Some(l.clone());
                        cur_max = helper_depth.get(l).copied().unwrap_or(0);
                    }
                }
                LirItem::Inst(LirInst::Call { target, .. }) if cur_helper.is_some() => {
                    let sub = if target.starts_with("__aun_out_") {
                        helper_depth.get(target).copied().unwrap_or(0)
                    } else {
                        0
                    };
                    cur_max = cur_max.max(2u16.saturating_add(sub));
                }
                LirItem::Inst(LirInst::Jp { target, .. })
                    if cur_helper.is_some() && target.starts_with("__aun_out_") =>
                {
                    let sub = helper_depth.get(target).copied().unwrap_or(0);
                    cur_max = cur_max.max(sub);
                }
                _ => {}
            }
        }
        flush(cur_helper, cur_max, &mut helper_depth, &mut changed);
    }

    // Pass 2: compute max outlined stack depth reachable from non-helper functions.
    let mut top_max: u16 = 0;
    let mut in_helper = false;
    for it in items {
        match it {
            LirItem::Label(l) => in_helper = l.starts_with("__aun_out_"),
            LirItem::Inst(LirInst::Call { target, .. })
                if !in_helper && target.starts_with("__aun_out_") =>
            {
                let d = 2u16.saturating_add(helper_depth.get(target).copied().unwrap_or(0));
                top_max = top_max.max(d);
            }
            LirItem::Inst(LirInst::Jp { target, .. })
                if !in_helper && target.starts_with("__aun_out_") =>
            {
                top_max = top_max.max(helper_depth.get(target).copied().unwrap_or(0));
            }
            _ => {}
        }
    }
    top_max
}

fn update_sp_state_lir(inst: &LirInst, sp_depth: &mut isize, sp_unknown: &mut bool) {
    match inst {
        LirInst::Push(_) => *sp_depth += 1,
        LirInst::Pop(_) => *sp_depth -= 1,
        LirInst::LdSp(_) | LirInst::Raw(_) => *sp_unknown = true,
        LirInst::Inc16(r)
        | LirInst::Dec16(r)
        | LirInst::Ld16Imm { dst: r, .. }
        | LirInst::Ld16Mem { dst: r, .. }
        | LirInst::Add16 { dst: r, .. }
            if *r == Reg16::SP =>
        {
            *sp_unknown = true;
        }
        _ => {}
    }
}

fn collapse_tail_calls_lir(items: &mut Vec<LirItem>) {
    let mut drop = vec![false; items.len()];
    let mut in_inline_asm = false;
    let mut sp_depth: isize = 0;
    let mut sp_unknown = false;
    for i in 0..items.len() {
        if drop[i] {
            continue;
        }
        match &items[i] {
            LirItem::Comment(c) => {
                let t = c.trim();
                if t == "; @aun:asm_begin" {
                    in_inline_asm = true;
                } else if t == "; @aun:asm_end" {
                    in_inline_asm = false;
                }
                continue;
            }
            LirItem::Label(_)
            | LirItem::Org(_)
            | LirItem::Equ(_, _)
            | LirItem::DataBytes(_)
            | LirItem::DataWords(_) => {
                if !in_inline_asm {
                    sp_depth = 0;
                    sp_unknown = false;
                }
                continue;
            }
            LirItem::Blank => continue,
            LirItem::Inst(_) => {}
        }
        if !in_inline_asm
            && !sp_unknown
            && sp_depth == 0
            && let LirItem::Inst(LirInst::Call {
                cond: None,
                target,
            }) = &items[i]
        {
            let target = target.clone();
            let mut j = i + 1;
            let mut crosses_asm = false;
            while j < items.len() {
                match &items[j] {
                    LirItem::Blank => j += 1,
                    LirItem::Comment(c) => {
                        if matches!(c.trim(), "; @aun:asm_begin" | "; @aun:asm_end") {
                            crosses_asm = true;
                            break;
                        }
                        j += 1;
                    }
                    _ => break,
                }
            }
            if !crosses_asm
                && j < items.len()
                && matches!(items[j], LirItem::Inst(LirInst::Ret(None)))
            {
                items[i] = LirItem::Inst(LirInst::Jp {
                    cond: None,
                    target,
                });
                drop[j] = true;
                continue;
            }
        }
        if let LirItem::Inst(inst) = &items[i] {
            update_sp_state_lir(inst, &mut sp_depth, &mut sp_unknown);
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

    #[test]
    fn outlines_balanced_push_pop_and_rejects_unbalanced() {
        let balanced = "\
fn_a:
    push ix
    pop de
    add hl, de
    ld a, (hl)
    or a
    ld b, a
    ret
fn_b:
    push ix
    pop de
    add hl, de
    ld a, (hl)
    or a
    inc (hl)
    ret
fn_c:
    push ix
    pop de
    add hl, de
    ld a, (hl)
    or a
    ld (hl), 1
    ret
";
        let mut items = parse_zilog_to_lir(balanced);
        let helpers = outline_lir(&mut items);
        assert!(helpers >= 1, "balanced push/pop sequence should outline:\n{items:?}");

        let unbalanced = "\
fn_a:
    push hl
    ld a, 1
    di
    pop hl
    ret
fn_b:
    push hl
    ld a, 1
    ei
    pop hl
    ret
";
        let mut items_u = parse_zilog_to_lir(unbalanced);
        let helpers_u = outline_lir(&mut items_u);
        assert_eq!(helpers_u, 0, "unbalanced push/pop run must not outline:\n{items_u:?}");
    }

    #[test]
    fn collapse_tail_calls_respects_inline_asm_and_unbalanced_stack() {
        let src = "\
f_unbal:
    push hl
    call _callee
    ret
f_asm:
; @aun:asm_begin
    push hl
    call _callee2
; @aun:asm_end
    ret
f_ok:
    call _callee3
    ret
";
        let mut items = parse_zilog_to_lir(src);
        let helpers = outline_lir(&mut items);
        assert_eq!(helpers, 0);
        collapse_tail_calls_lir(&mut items);
        let formatted: Vec<String> = items
            .iter()
            .filter_map(|it| match it {
                LirItem::Inst(i) => Some(i.format_zilog()),
                _ => None,
            })
            .collect();
        let joined = formatted.join("\n");
        assert!(joined.contains("call _callee\n"), "unbalanced call preserved:\n{joined}");
        assert!(joined.contains("call _callee2\n"), "inline asm call preserved:\n{joined}");
        assert!(joined.contains("jp _callee3\n") || joined.ends_with("jp _callee3"), "normal call folded:\n{joined}");
    }

    #[test]
    fn skips_tiny_loop_sequences_and_never_nests_outlined_calls() {
        // Repeated 4-byte pair inside a backward-branched loop (savings = 3*4 - (3*3 + 5) ...
        // wait: 3 occurrences * 4B = 12B vs 3*3 + 5 = 14B; test 4 occurrences * 4B = 16B -> savings = 4 < 6).
        // Outside a loop, savings = 4 >= 2 outlines cleanly; inside a loop, it must NOT outline.
        let loop_asm = "\
hot_fn:
hot_fn_bb0:
    ld l, (ix + 2)
    ld e, (ix + 3)
    ld l, (ix + 2)
    ld e, (ix + 3)
    ld l, (ix + 2)
    ld e, (ix + 3)
    ld l, (ix + 2)
    ld e, (ix + 3)
    djnz hot_fn_bb0
    ret
";
        let mut loop_items = parse_zilog_to_lir(loop_asm);
        assert_eq!(
            outline_lir(&mut loop_items),
            0,
            "small sequence saving < 6B inside loop should not pay +27T call overhead"
        );

        let cold_asm = "\
cold_a:
    ld l, (ix + 2)
    ld e, (ix + 3)
    ret
cold_b:
    ld l, (ix + 2)
    ld e, (ix + 3)
    ret
cold_c:
    ld l, (ix + 2)
    ld e, (ix + 3)
    ret
cold_d:
    ld l, (ix + 2)
    ld e, (ix + 3)
    ret
";
        let mut cold_items = parse_zilog_to_lir(cold_asm);
        assert_eq!(outline_lir(&mut cold_items), 1);

        // Even with larger repeated blocks containing an earlier `__aun_out_0` call, a second
        // outline pass must never pull `call __aun_out_0` inside another `__aun_out_*` helper.
        let nested_asm = "\
f1:
    ld a, (0xC000)
    add a, 0x05
    ld d, a
    ld e, 0x01
    call _ext_a
    ld (0xC001), a
    add a, 0x02
    ld (0xC002), a
    ret
f2:
    ld a, (0xC000)
    add a, 0x05
    ld d, a
    ld e, 0x01
    call _ext_a
    ld (0xC001), a
    add a, 0x02
    ld (0xC002), a
    ret
f3:
    ld a, (0xC000)
    add a, 0x05
    ld d, a
    ld e, 0x01
    call _ext_a
    ld (0xC001), a
    add a, 0x02
    ld (0xC002), a
    ret
";
        let mut nested_items = parse_zilog_to_lir(nested_asm);
        outline_lir(&mut nested_items);
        let mut in_helper = false;
        for it in &nested_items {
            match it {
                LirItem::Label(l) => in_helper = l.starts_with("__aun_out_"),
                LirItem::Inst(LirInst::Call { target, .. }) if in_helper => {
                    assert!(
                        !target.starts_with("__aun_out_"),
                        "outlined helper must not call another __aun_out_* helper: {target}"
                    );
                }
                _ => {}
            }
        }
    }
}
