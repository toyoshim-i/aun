//! LIR machine outliner for Z80 code-size compression.
//!
//! Identifies repeated straight-line `LirInst` sequences within single basic blocks,
//! extracts them into shared `__aun_out_<id>` helper routines when net encoded bytes
//! (accounting for `call` overhead + helper `ret`/tail-`jp`) decrease, and collapses
//! trailing `call target; ret` sequences into `jp target`.

use crate::lir::{LirInst, LirItem, Reg16};
use std::collections::{HashMap, HashSet, VecDeque};

/// Collects Z80 symbol names for `extern "asm"` functions that take any `in stack` argument.
/// Calling such callees pops caller-pushed arguments above the return address, so `outline_lir`
/// must never split a stack-argument call away from its caller's stack setup.
pub fn collect_stack_cleanup_extern_syms(program: &aun_ir::IrProgram) -> HashSet<String> {
    let mut out = HashSet::new();
    for ef in &program.externs {
        if ef
            .params
            .iter()
            .any(|(_, _, reg)| reg.eq_ignore_ascii_case("stack"))
        {
            out.insert(ef.name.clone());
        }
    }
    out
}

fn is_outlinable(inst: &LirInst, forbidden_calls: &HashSet<String>) -> bool {
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
        LirInst::Call { cond: None, target } => {
            !target.starts_with("__aun_out_") && !forbidden_calls.contains(target.as_str())
        }
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

fn is_main_event_loop_span(
    items: &[LirItem],
    t_idx: usize,
    idx: usize,
    backedge: &LirInst,
    fn_labels: &HashMap<&str, usize>,
    name_to_fn: &HashMap<String, usize>,
) -> bool {
    let has_external_call = items[t_idx..=idx].iter().any(|it| {
        matches!(
            it,
            LirItem::Inst(LirInst::Call { target, .. })
                if !name_to_fn.contains_key(target.as_str())
        )
    });
    if has_external_call {
        return true;
    }
    let is_unconditional_backedge = matches!(
        backedge,
        LirInst::Jp { cond: None, .. } | LirInst::Jr { cond: None, .. }
    );
    if !is_unconditional_backedge {
        return false;
    }
    let has_exit = items[t_idx..idx].iter().any(|it| match it {
        LirItem::Inst(LirInst::Ret(_)) => true,
        LirItem::Inst(
            LirInst::Jp { target, .. } | LirInst::Jr { target, .. } | LirInst::Djnz(target),
        ) => fn_labels
            .get(target.as_str())
            .is_none_or(|&dst| dst < t_idx || dst > idx),
        _ => false,
    });
    !has_exit
}

/// Outlines profitable repeated LIR instruction sequences into `__aun_out_<n>` helpers
/// and returns the number of helper routines generated.
pub fn outline_lir(items: &mut Vec<LirItem>) -> usize {
    outline_lir_with_forbidden_calls(items, &HashSet::new())
}

/// Like `outline_lir`, while never outlining or tail-collapsing `Call`s in `forbidden_calls`
/// (such as SDCC `--sdcccall(1)` `in stack` externs whose callee cleans up caller stack args).
pub fn outline_lir_with_forbidden_calls(
    items: &mut Vec<LirItem>,
    forbidden_calls: &HashSet<String>,
) -> usize {
    outline_lir_with_options(items, forbidden_calls, "__aun_out_")
}

/// Like `outline_lir_with_forbidden_calls`, allowing callers to namespace helper labels
/// (e.g. `"__aun_out_ovl{blob_id}_"` for HSM overlay blobs). Note: `helper_prefix` MUST start
/// with `"__aun_out_"` so outlined-helper detection and stack-depth checks apply uniformly.
pub fn outline_lir_with_options(
    items: &mut Vec<LirItem>,
    forbidden_calls: &HashSet<String>,
    helper_prefix: &str,
) -> usize {
    debug_assert!(helper_prefix.starts_with("__aun_out_"));
    const MIN_LEN: usize = 2;
    const MAX_LEN: usize = 12;
    const MAX_ITERS: usize = 128;

    let mut helpers: Vec<LirItem> = Vec::new();
    let mut helper_id = 0usize;

    for _ in 0..MAX_ITERS {
        // Identify intra-function loops (`jp`/`jr`/`djnz` backward branches and `djnz`) and
        // build an intra-module LIR call graph across functions so `loop_hot` propagates
        // transitively along `call` / `jp` tail-call edges to callees reachable from any loop.
        // Exception (#39): `main`/`_main`'s top-level frame/event loop (no exit edge, or
        // containing an external I/O/HAL call) must not mark every UI/scene handler `loop_hot`.
        let mut in_loop = vec![false; items.len()];
        let mut item_fn: Vec<Option<usize>> = vec![None; items.len()];
        let mut fn_names: Vec<String> = Vec::new();
        let mut name_to_fn: HashMap<String, usize> = HashMap::new();
        let mut fn_label_pos: Vec<HashMap<&str, usize>> = Vec::new();
        {
            let mut cur_fn_name: Option<&str> = None;
            let mut cur_fn_id: Option<usize> = None;
            for (idx, it) in items.iter().enumerate() {
                if let LirItem::Label(l) = it {
                    if !crate::peephole::is_intra_fn_label(l, cur_fn_name)
                        && !l.starts_with("__aun_out_")
                    {
                        let fid = fn_names.len();
                        fn_names.push(l.clone());
                        name_to_fn.insert(l.clone(), fid);
                        fn_label_pos.push(HashMap::new());
                        cur_fn_name = Some(l.as_str());
                        cur_fn_id = Some(fid);
                    }
                    if let Some(fid) = cur_fn_id {
                        fn_label_pos[fid].insert(l.as_str(), idx);
                    }
                }
                item_fn[idx] = cur_fn_id;
            }
        }
        let mut in_cmp_ovf_stub = false;
        for (idx, it) in items.iter().enumerate() {
            let Some(fid) = item_fn[idx] else {
                in_cmp_ovf_stub = false;
                continue;
            };
            match it {
                LirItem::Label(l) => {
                    in_cmp_ovf_stub = l.contains("_cmp_ovf_");
                    continue;
                }
                LirItem::Comment(_) | LirItem::Blank => continue,
                LirItem::Inst(
                    inst @ (LirInst::Jp { target, .. }
                    | LirInst::Jr { target, .. }
                    | LirInst::Djnz(target)),
                ) => {
                    if in_cmp_ovf_stub {
                        continue;
                    }
                    if let Some(&t_idx) = fn_label_pos[fid].get(target.as_str())
                        && t_idx < idx
                    {
                        let is_main = matches!(fn_names[fid].as_str(), "main" | "_main");
                        if is_main
                            && is_main_event_loop_span(
                                items,
                                t_idx,
                                idx,
                                inst,
                                &fn_label_pos[fid],
                                &name_to_fn,
                            )
                        {
                            continue;
                        }
                        for slot in &mut in_loop[t_idx..=idx] {
                            *slot = true;
                        }
                    } else if matches!(inst, LirInst::Djnz(_)) {
                        in_loop[idx] = true;
                    }
                }
                _ => {
                    in_cmp_ovf_stub = false;
                }
            }
        }

        let mut callees: Vec<Vec<usize>> = vec![Vec::new(); fn_names.len()];
        let mut loop_hot = vec![false; fn_names.len()];
        for (idx, it) in items.iter().enumerate() {
            let Some(caller_fid) = item_fn[idx] else {
                continue;
            };
            if let LirItem::Inst(LirInst::Call { target, .. } | LirInst::Jp { target, .. }) = it
                && let Some(&callee_fid) = name_to_fn.get(target.as_str())
            {
                if callee_fid != caller_fid {
                    callees[caller_fid].push(callee_fid);
                }
                if in_loop[idx] {
                    loop_hot[callee_fid] = true;
                }
            }
        }
        let mut queue: VecDeque<usize> = loop_hot
            .iter()
            .enumerate()
            .filter_map(|(fid, &hot)| hot.then_some(fid))
            .collect();
        while let Some(u) = queue.pop_front() {
            for &v in &callees[u] {
                if !loop_hot[v] {
                    loop_hot[v] = true;
                    queue.push_back(v);
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
                && is_outlinable(inst, forbidden_calls)
            {
                let start = idx;
                let mut seq = Vec::new();
                while idx < items.len()
                    && let LirItem::Inst(i2) = &items[idx]
                    && is_outlinable(i2, forbidden_calls)
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
            // Never outline occurrences inside detected loops or inside loop-hot callees
            // reachable from a loop (avoid +27T call/ret per iteration).
            let positions: Vec<usize> = positions
                .into_iter()
                .filter(|&p| {
                    let end = (p + slice.len()).min(in_loop.len());
                    !(p..end).any(|i| in_loop[i] || item_fn[i].is_some_and(|fid| loop_hot[fid]))
                })
                .collect();
            if positions.len() < 2 {
                continue;
            }
            let seq_bytes: usize = slice.iter().map(LirInst::byte_len).sum();
            // Suppress trivial <= 9 B fragments even in cold code where call/ret 27T dominates.
            if seq_bytes <= 9 {
                continue;
            }
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

        let helper_name = format!("{helper_prefix}{helper_id}");
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
    collapse_tail_calls_lir(items, forbidden_calls);

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

fn collapse_tail_calls_lir(items: &mut Vec<LirItem>, forbidden_calls: &HashSet<String>) {
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
            && !forbidden_calls.contains(target.as_str())
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
    ld d, a
    ld e, 0x10
    add hl, de
    ld b, a
    ret
fn_b:
    push ix
    pop de
    add hl, de
    ld a, (hl)
    or a
    ld d, a
    ld e, 0x10
    add hl, de
    inc (hl)
    ret
fn_c:
    push ix
    pop de
    add hl, de
    ld a, (hl)
    or a
    ld d, a
    ld e, 0x10
    add hl, de
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
        collapse_tail_calls_lir(&mut items, &HashSet::new());
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
    ld h, (ix + 4)
    ld d, (ix + 5)
    ret
cold_b:
    ld l, (ix + 2)
    ld e, (ix + 3)
    ld h, (ix + 4)
    ld d, (ix + 5)
    ret
cold_c:
    ld l, (ix + 2)
    ld e, (ix + 3)
    ld h, (ix + 4)
    ld d, (ix + 5)
    ret
cold_d:
    ld l, (ix + 2)
    ld e, (ix + 3)
    ld h, (ix + 4)
    ld d, (ix + 5)
    ret
";
        let mut cold_items = parse_zilog_to_lir(cold_asm);
        assert_eq!(outline_lir(&mut cold_items), 1);

        // Cold <= 9B snippets must NOT be outlined since call+ret (27T) dominates.
        let small_cold_asm = "\
cold_9a:
    ld l, (ix + 2)
    ld h, (ix + 3)
    ld a, (ix + 4)
    ret
cold_9b:
    ld l, (ix + 2)
    ld h, (ix + 3)
    ld a, (ix + 4)
    ret
cold_9c:
    ld l, (ix + 2)
    ld h, (ix + 3)
    ld a, (ix + 4)
    ret
cold_9d:
    ld l, (ix + 2)
    ld h, (ix + 3)
    ld a, (ix + 4)
    ret
";
        let mut small_cold_items = parse_zilog_to_lir(small_cold_asm);
        assert_eq!(
            outline_lir(&mut small_cold_items),
            0,
            "trivial <= 9B fragments should not be outlined even in cold code"
        );

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

    #[test]
    fn skips_large_loops_across_intra_fn_labels_outlines_only_cold_and_skips_4b_pairs() {
        // Large repeated sequence inside a loop containing intra-function `_cmp_0` / `_sk_0`
        // labels must NOT be outlined, while cold occurrences in separate functions outline cleanly.
        let mixed_asm = "\
hot_loop:
hot_loop_bb0:
    ld l, (ix + 1)
    ld h, (ix + 2)
    inc hl
    inc hl
hot_loop_cmp_0:
    ld (ix + 1), l
    ld (ix + 2), h
hot_loop_sk_0:
    ld l, (ix + 1)
    ld h, (ix + 2)
    inc hl
    inc hl
    ld (ix + 1), l
    ld (ix + 2), h
    djnz hot_loop_bb0
    ret
cold_x:
    ld l, (ix + 1)
    ld h, (ix + 2)
    inc hl
    inc hl
    ld (ix + 1), l
    ld (ix + 2), h
    ret
cold_y:
    ld l, (ix + 1)
    ld h, (ix + 2)
    inc hl
    inc hl
    ld (ix + 1), l
    ld (ix + 2), h
    ld a, 1
    ret
";
        let mut mixed_items = parse_zilog_to_lir(mixed_asm);
        let helpers = outline_lir(&mut mixed_items);
        assert_eq!(helpers, 1);
        let outlined_calls: usize = mixed_items
            .iter()
            .filter(|it| matches!(it, LirItem::Inst(LirInst::Call { target, .. } | LirInst::Jp { target, .. }) if target == "__aun_out_0"))
            .count();
        assert_eq!(
            outlined_calls, 2,
            "only cold occurrences (cold_x, cold_y) should be outlined; loop body untouched:\n{mixed_items:?}"
        );

        // Tiny 4-byte `srl h; rr l` repeated outside any loop must NOT be outlined.
        let mut tiny_asm = String::from("shifts:\n");
        for sep in ["inc a", "inc b", "inc c", "inc d", "inc e", "dec a"] {
            tiny_asm.push_str("    srl h\n    rr l\n");
            tiny_asm.push_str(&format!("    {sep}\n"));
        }
        tiny_asm.push_str("    ret\n");
        let mut tiny_items = parse_zilog_to_lir(&tiny_asm);
        assert_eq!(
            outline_lir(&mut tiny_items),
            0,
            "tiny <= 9B fragments should never be outlined"
        );
    }

    #[test]
    fn propagates_loop_hot_along_call_and_tail_jp_edges_while_outlining_cold() {
        let asm = "\
physics_step_frame:
    ld b, 4
physics_step_frame_bb0:
    call step_ball_substep
    djnz physics_step_frame_bb0
    ret
step_ball_substep:
    ld l, (ix + 1)
    ld h, (ix + 2)
    inc hl
    inc hl
    ld (ix + 1), l
    ld (ix + 2), h
    call sync_ball_px
    jp collide_hull
sync_ball_px:
    ld l, (ix + 1)
    ld h, (ix + 2)
    inc hl
    inc hl
    ld (ix + 1), l
    ld (ix + 2), h
    ret
collide_hull:
    ld l, (ix + 1)
    ld h, (ix + 2)
    inc hl
    inc hl
    ld (ix + 1), l
    ld (ix + 2), h
    ret
cold_init_a:
    ld l, (ix + 1)
    ld h, (ix + 2)
    inc hl
    inc hl
    ld (ix + 1), l
    ld (ix + 2), h
    ret
cold_init_b:
    ld l, (ix + 1)
    ld h, (ix + 2)
    inc hl
    inc hl
    ld (ix + 1), l
    ld (ix + 2), h
    ld a, 1
    ret
";
        let mut items = parse_zilog_to_lir(asm);
        let helpers = outline_lir(&mut items);
        assert_eq!(helpers, 1, "only cold duplicates should outline");
        let outlined_calls: usize = items
            .iter()
            .filter(|it| {
                matches!(
                    it,
                    LirItem::Inst(LirInst::Call { target, .. } | LirInst::Jp { target, .. })
                        if target == "__aun_out_0"
                )
            })
            .count();
        assert_eq!(
            outlined_calls, 2,
            "loop-hot callees must not outline, only cold_init_a/b:\n{items:?}"
        );
    }

    #[test]
    fn forbidden_calls_prevents_outlining_and_tail_collapsing_stack_arg_extern() {
        let asm = "\
stage_a:
    ld a, 5
    push af
    inc sp
    ld hl, 0xEC00
    call _stack_callee
    ld a, (0xC000)
    or a
    ld b, 1
    ld c, 2
    ld d, 3
    ret
stage_b:
    ld a, 5
    push af
    inc sp
    ld hl, 0xEC00
    call _stack_callee
    ld a, (0xC000)
    or a
    ld b, 1
    ld c, 2
    ld d, 3
    ret
tail_caller:
    call _stack_callee
    ret
";
        let mut items = parse_zilog_to_lir(asm);
        let mut forbidden = HashSet::new();
        forbidden.insert("_stack_callee".to_string());
        outline_lir_with_forbidden_calls(&mut items, &forbidden);
        let in_helper = items
            .iter()
            .skip_while(|it| !matches!(it, LirItem::Label(l) if l.starts_with("__aun_out_")))
            .any(|it| {
                matches!(
                    it,
                    LirItem::Inst(LirInst::Call { target, .. } | LirInst::Jp { target, .. })
                        if target == "_stack_callee"
                )
            });
        assert!(
            !in_helper,
            "outlined helpers must not contain call/jp to stack-arg callee:\n{items:?}"
        );
        let tail_preserved = items
            .windows(2)
            .any(|w| {
                matches!(
                    (&w[0], &w[1]),
                    (
                        LirItem::Inst(LirInst::Call { target, .. }),
                        LirItem::Inst(LirInst::Ret(None))
                    ) if target == "_stack_callee"
                )
            });
        assert!(
            tail_preserved,
            "stack-arg call followed by ret must not tail-collapse to jp:\n{items:?}"
        );
    }

    #[test]
    fn main_event_loop_retains_cold_outlining_while_inner_loop_stays_hot() {
        let asm = "\
main:
    jr main_loop
main_loop:
    call _hw_vsync
    ld b, 4
main_inner:
    call inner_hot
    djnz main_inner
    call ui_screen_a
    call ui_screen_b
    jp main_loop
inner_hot:
    ld l, (ix + 1)
    ld h, (ix + 2)
    inc hl
    inc hl
    ld (ix + 1), l
    ld (ix + 2), h
    ret
ui_screen_a:
    ld l, (ix + 1)
    ld h, (ix + 2)
    inc hl
    inc hl
    ld (ix + 1), l
    ld (ix + 2), h
    ret
ui_screen_b:
    ld l, (ix + 1)
    ld h, (ix + 2)
    inc hl
    inc hl
    ld (ix + 1), l
    ld (ix + 2), h
    ld a, 1
    ret
";
        let mut items = parse_zilog_to_lir(asm);
        let helpers = outline_lir(&mut items);
        assert_eq!(helpers, 1, "cold ui handlers must outline even from main event loop");
        let outlined_calls: usize = items
            .iter()
            .filter(|it| {
                matches!(
                    it,
                    LirItem::Inst(LirInst::Call { target, .. } | LirInst::Jp { target, .. })
                        if target == "__aun_out_0"
                )
            })
            .count();
        assert_eq!(
            outlined_calls, 2,
            "only ui_screen_a and ui_screen_b should outline; inner_hot stays hot:\n{items:?}"
        );
    }
}
