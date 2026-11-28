use super::*;

fn reg_byte_width(op: &str) -> u16 {
    if matches!(
        op.trim().to_ascii_lowercase().as_str(),
        "hl" | "de" | "bc" | "ix" | "iy" | "sp"
    ) {
        2
    } else {
        1
    }
}

fn mem_ranges_overlap(a0: MemAddr, w0: u16, a1: MemAddr, w1: u16) -> bool {
    match (a0, a1) {
        (MemAddr::Fixed(x), MemAddr::Fixed(y))
        | (MemAddr::Uchiguri(x), MemAddr::Uchiguri(y)) => {
            let s0 = x as u32;
            let e0 = s0 + w0 as u32;
            let s1 = y as u32;
            let e1 = s1 + w1 as u32;
            s0 < e1 && s1 < e0
        }
        _ => false,
    }
}

fn inst_touches_mem_range_sz(kind: &LineKind, addr: MemAddr, sz: u16) -> bool {
    let LineKind::Inst { mnemonic, ops } = kind else {
        return false;
    };
    if mnemonic == "ld" && ops.len() == 2 {
        if let Some(wa) = parse_mem_ref(&ops[0])
            && mem_ranges_overlap(wa, reg_byte_width(&ops[1]), addr, sz)
        {
            return true;
        }
        if let Some(ra) = parse_mem_ref(&ops[1])
            && mem_ranges_overlap(ra, reg_byte_width(&ops[0]), addr, sz)
        {
            return true;
        }
    }
    false
}

fn addr_in_sb(addr: MemAddr, uchiguri_base: Option<u16>, sz: u16, lo: u16, hi: u16) -> bool {
    let resolved = match (addr, uchiguri_base) {
        (MemAddr::Uchiguri(off), Some(base)) => base.wrapping_add(off),
        (MemAddr::Fixed(a), None) => a,
        _ => return false,
    };
    resolved >= lo && (resolved as u32) + (sz as u32) <= (hi as u32)
}

/// Verifies `[addr, addr + sz)` is not read again after `load_idx` before overwrite or IR BB end.
fn sb_slot_dead_after_load(
    kinds: &[LineKind],
    label_pos: &HashMap<String, usize>,
    load_idx: usize,
    end: usize,
    addr: MemAddr,
    sz: u16,
) -> bool {
    let mut saw_branch_or_synth = false;
    let mut max_fwd_target = load_idx;
    for k in (load_idx + 1)..end {
        match &kinds[k] {
            LineKind::EmptyOrComment => {}
            LineKind::Label(lbl) => {
                if is_bb_label(lbl) && k >= max_fwd_target {
                    return true;
                }
                if is_bb_label(lbl) {
                    return false;
                }
                saw_branch_or_synth = true;
            }
            LineKind::Directive => return k >= max_fwd_target,
            LineKind::Inst {
                mnemonic: km,
                ops: kops,
            } => {
                if matches!(km.as_str(), "ret" | "reti" | "halt") && kops.is_empty() {
                    if k >= max_fwd_target {
                        return true;
                    }
                    continue;
                }
                if matches!(km.as_str(), "jp" | "jr") && kops.len() == 1 {
                    if is_bb_label(&kops[0]) {
                        if k >= max_fwd_target {
                            return true;
                        }
                        continue;
                    }
                    if let Some(&tp) = label_pos.get(&kops[0])
                        && tp > k
                        && tp < end
                    {
                        max_fwd_target = max_fwd_target.max(tp);
                        saw_branch_or_synth = true;
                        continue;
                    }
                    return false;
                }
                if km == "ret" && kops.len() == 1 {
                    continue;
                }
                if matches!(km.as_str(), "jp" | "jr" | "djnz") {
                    if kops.last().is_some_and(|t| is_bb_label(t)) {
                        continue;
                    }
                    if let Some(t) = kops.last()
                        && let Some(&tp) = label_pos.get(t)
                        && tp > k
                        && tp < end
                    {
                        max_fwd_target = max_fwd_target.max(tp);
                        saw_branch_or_synth = true;
                        continue;
                    }
                    return false;
                }
                if km == "ld" && kops.len() == 2 {
                    if let Some(ra) = parse_mem_ref(&kops[1])
                        && mem_ranges_overlap(ra, reg_byte_width(&kops[0]), addr, sz)
                    {
                        return false;
                    }
                    if !saw_branch_or_synth
                        && let Some(wa) = parse_mem_ref(&kops[0])
                        && wa == addr
                        && reg_byte_width(&kops[1]) >= sz
                    {
                        return true;
                    }
                }
            }
        }
    }
    true
}

/// Returns `true` if both `H` and `L` are overwritten before being read across `[start..end)`.
fn hl_overwritten_before_read(lines: &[String], start: usize, end: usize) -> bool {
    let mut h_dead = false;
    let mut l_dead = false;
    for line in &lines[start..end] {
        let (rh, wh) = inst_reads_writes_reg8(line, 'h');
        let (rl, wl) = inst_reads_writes_reg8(line, 'l');
        if (rh && !h_dead) || (rl && !l_dead) {
            return false;
        }
        if wh {
            h_dead = true;
        }
        if wl {
            l_dead = true;
        }
        if h_dead && l_dead {
            return true;
        }
    }
    false
}

/// Promotes single-block scratch spills (`ld (addr), hl` -> `ld de/hl, (addr)` and
/// `ld (addr), a` -> `ld r, (addr)`) in `fn_sb_ranges` (or function-unique private slots
/// in `fn_private_ranges`) to `DE`, `E`, or `D` when the target register(s) are untouched
/// in between and safe afterward.
pub(super) fn promote_single_block_spills_to_de(
    lines: &[String],
    fn_private_ranges: &HashMap<String, Vec<(u16, u16)>>,
    fn_sb_ranges: &HashMap<String, (u16, u16)>,
) -> (Vec<String>, bool) {
    let uchiguri_base = detect_uchiguri_base(lines);
    let kinds: Vec<LineKind> = lines.iter().map(|l| classify_line(l)).collect();
    let label_pos = build_label_positions(&kinds);
    let mut combined_ranges: HashMap<String, Vec<(u16, u16)>> = fn_private_ranges.clone();
    for (k, &r) in fn_sb_ranges {
        combined_ranges.entry(k.clone()).or_insert_with(|| vec![r]);
    }
    let fn_spans = partition_fn_spans(&kinds, &combined_ranges);

    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut changed = false;

    for (start, end, _) in fn_spans {
        let sb_range = if let LineKind::Label(lbl) = &kinds[start] {
            fn_sb_ranges.get(lbl).copied()
        } else {
            None
        };
        let priv_ranges = if let LineKind::Label(lbl) = &kinds[start] {
            fn_private_ranges.get(lbl).map(Vec::as_slice)
        } else {
            None
        };
        if sb_range.is_none() && priv_ranges.is_none_or(|r| r.is_empty()) {
            out.extend_from_slice(&lines[start..end]);
            continue;
        }

        let mut ref_count: HashMap<MemAddr, usize> = HashMap::new();
        let mut span_label_refs: HashMap<String, (usize, usize)> = HashMap::new();
        for (idx, kind) in kinds[start..end].iter().enumerate() {
            let abs_idx = start + idx;
            if let LineKind::Inst { mnemonic, ops } = kind {
                if mnemonic == "ld" && ops.len() == 2 {
                    for op in ops {
                        if let Some(a) = parse_mem_ref(op) {
                            *ref_count.entry(a).or_insert(0) += 1;
                        }
                    }
                } else if matches!(mnemonic.as_str(), "jp" | "jr" | "djnz" | "call")
                    && let Some(t) = ops.last()
                    && label_pos.contains_key(t)
                {
                    span_label_refs
                        .entry(t.clone())
                        .and_modify(|(mn, mx)| {
                            *mn = (*mn).min(abs_idx);
                            *mx = (*mx).max(abs_idx);
                        })
                        .or_insert((abs_idx, abs_idx));
                }
            }
        }

        let mut replace: HashMap<usize, Vec<String>> = HashMap::new();
        let mut in_guarded = false;
        let mut i = start;
        while i < end {
            if matches!(kinds[i], LineKind::EmptyOrComment) {
                let t = lines[i].trim();
                if t == "; @aun:volatile_begin" || t == "; @aun:asm_begin" {
                    in_guarded = true;
                } else if t == "; @aun:volatile_end" || t == "; @aun:asm_end" {
                    in_guarded = false;
                }
                i += 1;
                continue;
            }
            if in_guarded {
                i += 1;
                continue;
            }
            if let LineKind::Inst { mnemonic, ops } = &kinds[i]
                && mnemonic == "ld"
                && ops.len() == 2
                && let Some(addr) = parse_mem_ref(&ops[0])
            {
                let is_sb16 =
                    sb_range.is_some_and(|(lo, hi)| addr_in_sb(addr, uchiguri_base, 2, lo, hi));
                let is_priv16_unique = priv_ranges.is_some_and(|r| {
                    in_private_ranges(addr, uchiguri_base, r)
                        && ref_count.get(&addr).copied() == Some(2)
                        && ref_count.get(&addr.add(1)).copied().unwrap_or(0) == 0
                        && addr
                            .sub_one()
                            .is_none_or(|p| ref_count.get(&p).copied().unwrap_or(0) == 0)
                });
                let is_sb8 =
                    sb_range.is_some_and(|(lo, hi)| addr_in_sb(addr, uchiguri_base, 1, lo, hi));
                let is_priv8_unique = priv_ranges.is_some_and(|r| {
                    in_private_ranges(addr, uchiguri_base, r)
                        && ref_count.get(&addr).copied() == Some(2)
                        && addr
                            .sub_one()
                            .is_none_or(|p| ref_count.get(&p).copied().unwrap_or(0) == 0)
                });
                if ops[1].eq_ignore_ascii_case("hl") && (is_sb16 || is_priv16_unique) {
                    let mut found_16: Option<(usize, String)> = None;
                    let mut crossed_syn: Vec<String> = Vec::new();
                    for j in (i + 1)..end {
                        match &kinds[j] {
                            LineKind::EmptyOrComment => {
                                if lines[j].trim().starts_with("; @aun:") {
                                    break;
                                }
                            }
                            LineKind::Label(lbl) => {
                                if is_bb_label(lbl)
                                    || span_label_refs
                                        .get(lbl)
                                        .is_some_and(|&(mn, _)| mn <= i)
                                {
                                    break;
                                }
                                crossed_syn.push(lbl.clone());
                            }
                            LineKind::Directive => break,
                            LineKind::Inst {
                                mnemonic: m2,
                                ops: ops2,
                            } => {
                                if m2 == "ld"
                                    && ops2.len() == 2
                                    && let Some(ra) = parse_mem_ref(&ops2[1])
                                    && (ra == addr
                                        || (ra == addr.add(1) && ops2[0].eq_ignore_ascii_case("a")))
                                {
                                    let dst = ops2[0].trim().to_ascii_lowercase();
                                    let is_8_ok = dst == "a" && (ra == addr || ra == addr.add(1));
                                    let tag = if ra == addr.add(1) {
                                        "a_hi".to_string()
                                    } else {
                                        dst
                                    };
                                    if (matches!(tag.as_str(), "de" | "hl") || is_8_ok)
                                        && crossed_syn.iter().all(|l| {
                                            span_label_refs
                                                .get(l)
                                                .is_none_or(|&(mn, mx)| mn > i && mx < j)
                                        })
                                    {
                                        found_16 = Some((j, tag));
                                    }
                                    break;
                                }
                                if matches!(
                                    m2.as_str(),
                                    "call" | "rst" | "jp" | "jr" | "djnz" | "ret" | "reti"
                                        | "retn" | "halt" | "exx"
                                ) {
                                    break;
                                }
                                if inst_touches_mem_range_sz(&kinds[j], addr, 2) {
                                    break;
                                }
                                if inst_reads_writes_reg8(&lines[j], 'd') != (false, false)
                                    || inst_reads_writes_reg8(&lines[j], 'e') != (false, false)
                                {
                                    break;
                                }
                            }
                        }
                    }
                    if let Some((load_idx, load_dst)) = found_16
                        && (is_priv16_unique
                            || sb_slot_dead_after_load(&kinds, &label_pos, load_idx, end, addr, 2))
                    {
                        if load_dst == "a" || load_dst == "a_hi" {
                            let src_hl = if load_dst == "a_hi" { 'h' } else { 'l' };
                            let mut did_promote = false;
                            for r in ['e', 'd'] {
                                let untouched = ((i + 1)..load_idx).all(|k| {
                                    inst_reads_writes_reg8(&lines[k], r) == (false, false)
                                });
                                if untouched
                                    && reg8_dead_after(lines, start, end, load_idx + 1, r)
                                {
                                    replace.insert(i, vec![format!("    ld {r}, {src_hl}")]);
                                    replace.insert(load_idx, vec![format!("    ld a, {r}")]);
                                    changed = true;
                                    did_promote = true;
                                    break;
                                }
                            }
                            if did_promote {
                                i = load_idx + 1;
                                continue;
                            }
                        } else if load_dst == "de" {
                            let store_repl = if hl_overwritten_before_read(lines, i + 1, load_idx) {
                                vec!["    ex de, hl".to_string()]
                            } else {
                                vec!["    ld d, h".to_string(), "    ld e, l".to_string()]
                            };
                            replace.insert(i, store_repl);
                            replace.insert(load_idx, Vec::new());
                            changed = true;
                            i = load_idx + 1;
                            continue;
                        } else if load_dst == "hl"
                            && reg_pair_dead_after(lines, start, end, load_idx + 1, 'd', 'e')
                        {
                            let store_repl = if hl_overwritten_before_read(lines, i + 1, load_idx) {
                                vec!["    ex de, hl".to_string()]
                            } else {
                                vec!["    ld d, h".to_string(), "    ld e, l".to_string()]
                            };
                            replace.insert(i, store_repl);
                            replace.insert(load_idx, vec!["    ex de, hl".to_string()]);
                            changed = true;
                            i = load_idx + 1;
                            continue;
                        }
                    }
                } else if ops[1].eq_ignore_ascii_case("a") && (is_sb8 || is_priv8_unique) {
                    let mut found_8: Option<(usize, char)> = None;
                    let mut crossed_syn: Vec<String> = Vec::new();
                    for j in (i + 1)..end {
                        match &kinds[j] {
                            LineKind::EmptyOrComment => {
                                if lines[j].trim().starts_with("; @aun:") {
                                    break;
                                }
                            }
                            LineKind::Label(lbl) => {
                                if is_bb_label(lbl)
                                    || span_label_refs
                                        .get(lbl)
                                        .is_some_and(|&(mn, _)| mn <= i)
                                {
                                    break;
                                }
                                crossed_syn.push(lbl.clone());
                            }
                            LineKind::Directive => break,
                            LineKind::Inst {
                                mnemonic: m2,
                                ops: ops2,
                            } => {
                                if m2 == "ld"
                                    && ops2.len() == 2
                                    && parse_mem_ref(&ops2[1]) == Some(addr)
                                {
                                    let dst = ops2[0].trim().to_ascii_lowercase();
                                    if dst.len() == 1
                                        && matches!(
                                            dst.as_str(),
                                            "a" | "b" | "c" | "d" | "e" | "h" | "l"
                                        )
                                        && crossed_syn.iter().all(|l| {
                                            span_label_refs
                                                .get(l)
                                                .is_none_or(|&(mn, mx)| mn > i && mx < j)
                                        })
                                    {
                                        found_8 = Some((j, dst.chars().next().unwrap()));
                                    }
                                    break;
                                }
                                if matches!(
                                    m2.as_str(),
                                    "call" | "rst" | "jp" | "jr" | "djnz" | "ret" | "reti"
                                        | "retn" | "halt" | "exx"
                                ) {
                                    break;
                                }
                                if inst_touches_mem_range_sz(&kinds[j], addr, 1) {
                                    break;
                                }
                            }
                        }
                    }
                    if let Some((load_idx, dst_char)) = found_8
                        && (is_priv8_unique
                            || sb_slot_dead_after_load(&kinds, &label_pos, load_idx, end, addr, 1))
                    {
                        let reg_order = if dst_char == 'd' { ['d', 'e'] } else { ['e', 'd'] };
                        let mut promoted = false;
                        for r in reg_order {
                            let untouched = ((i + 1)..load_idx).all(|k| {
                                inst_reads_writes_reg8(&lines[k], r) == (false, false)
                            });
                            if untouched
                                && (dst_char == r
                                    || reg8_dead_after(lines, start, end, load_idx + 1, r))
                            {
                                replace.insert(i, vec![format!("    ld {r}, a")]);
                                replace.insert(
                                    load_idx,
                                    if dst_char == r {
                                        Vec::new()
                                    } else {
                                        vec![format!("    ld {dst_char}, {r}")]
                                    },
                                );
                                changed = true;
                                i = load_idx + 1;
                                promoted = true;
                                break;
                            }
                        }
                        if promoted {
                            continue;
                        }
                    }
                }
            }
            i += 1;
        }

        for idx in start..end {
            if let Some(repl) = replace.remove(&idx) {
                out.extend(repl);
            } else {
                out.push(lines[idx].clone());
            }
        }
    }

    (out, changed)
}

/// Promotes a single-write / single-read 16-bit private RAM spill (`ld (S), hl` ... `ld hl/de/bc, (S)`)
/// within the same basic block to `BC` (`ld b, h; ld c, l` ... `ld h, b; ld l, c`) whenever
/// `BC` is untouched in between, `BC` is dead afterward (or overwritten by `ld bc, (S)`),
/// and `S` is referenced nowhere else in the function (or is single-block scratch dead after load).
pub(super) fn promote_single_block_spills_to_bc(
    lines: &[String],
    fn_private_ranges: &HashMap<String, Vec<(u16, u16)>>,
    fn_sb_ranges: &HashMap<String, (u16, u16)>,
    no_bc_fns: &HashSet<String>,
) -> (Vec<String>, bool) {
    let uchiguri_base = detect_uchiguri_base(lines);
    let kinds: Vec<LineKind> = lines.iter().map(|l| classify_line(l)).collect();
    let label_pos = build_label_positions(&kinds);
    let fn_spans = partition_fn_spans(&kinds, fn_private_ranges);

    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut changed = false;

    for (start, end, range_opt) in fn_spans {
        let Some(ranges) = range_opt.filter(|r| !r.is_empty()) else {
            out.extend_from_slice(&lines[start..end]);
            continue;
        };
        let is_no_bc = matches!(&kinds[start], LineKind::Label(l) if no_bc_fns.contains(l));
        if is_no_bc {
            out.extend_from_slice(&lines[start..end]);
            continue;
        }
        let sb_range = if let LineKind::Label(lbl) = &kinds[start] {
            fn_sb_ranges.get(lbl).copied()
        } else {
            None
        };

        // Count all memory references (reads and writes) to each address in this function.
        let mut ref_count: HashMap<MemAddr, usize> = HashMap::new();
        for kind in &kinds[start..end] {
            if let LineKind::Inst { mnemonic, ops } = kind
                && mnemonic == "ld"
                && ops.len() == 2
            {
                for op in ops {
                    if let Some(a) = parse_mem_ref(op) {
                        *ref_count.entry(a).or_insert(0) += 1;
                    }
                }
            }
        }

        // Map line indices to replacement lines when a pair is promoted.
        let mut replace: HashMap<usize, Vec<String>> = HashMap::new();
        let mut in_guarded = false;
        let mut i = start;
        while i < end {
            if matches!(kinds[i], LineKind::EmptyOrComment) {
                let t = lines[i].trim();
                if t == "; @aun:volatile_begin" || t == "; @aun:asm_begin" {
                    in_guarded = true;
                } else if t == "; @aun:volatile_end" || t == "; @aun:asm_end" {
                    in_guarded = false;
                }
                i += 1;
                continue;
            }
            if in_guarded {
                i += 1;
                continue;
            }
            if let LineKind::Inst { mnemonic, ops } = &kinds[i]
                && mnemonic == "ld"
                && ops.len() == 2
                && let Some(addr) = parse_mem_ref(&ops[0])
                && ops[1].eq_ignore_ascii_case("hl")
                && in_private_ranges(addr, uchiguri_base, ranges)
                && ref_count.get(&addr.add(1)).copied().unwrap_or(0) == 0
                && addr.sub_one().is_none_or(|prev| ref_count.get(&prev).copied().unwrap_or(0) == 0)
            {
                let is_sb =
                    sb_range.is_some_and(|(lo, hi)| addr_in_sb(addr, uchiguri_base, 2, lo, hi));
                let is_fn_unique = ref_count.get(&addr).copied() == Some(2);
                if is_sb || is_fn_unique {
                    // Scan forward in the same basic block for the matching `ld hl/de/bc, (addr)`
                    let mut found_read: Option<(usize, String)> = None;
                    for j in (i + 1)..end {
                        match &kinds[j] {
                            LineKind::Label(_) | LineKind::Directive => break,
                            LineKind::EmptyOrComment => {
                                if lines[j].trim().starts_with("; @aun:") {
                                    break;
                                }
                            }
                            LineKind::Inst {
                                mnemonic: m2,
                                ops: ops2,
                            } => {
                                if m2 == "ld"
                                    && ops2.len() == 2
                                    && parse_mem_ref(&ops2[1]) == Some(addr)
                                {
                                    let dst = ops2[0].trim().to_ascii_lowercase();
                                    if matches!(dst.as_str(), "hl" | "de" | "bc") {
                                        found_read = Some((j, dst));
                                    }
                                    break;
                                }
                                if matches!(
                                    m2.as_str(),
                                    "call" | "rst" | "jp" | "jr" | "djnz" | "ret" | "reti"
                                        | "retn" | "halt" | "exx"
                                ) {
                                    break;
                                }
                                if inst_reads_writes_reg8(&lines[j], 'b') != (false, false)
                                    || inst_reads_writes_reg8(&lines[j], 'c') != (false, false)
                                {
                                    break;
                                }
                                if inst_touches_mem_range_sz(&kinds[j], addr, 2) {
                                    break;
                                }
                            }
                        }
                    }
                    if let Some((j, load_dst)) = found_read
                        && (is_fn_unique
                            || sb_slot_dead_after_load(&kinds, &label_pos, j, end, addr, 2))
                        && (load_dst == "bc"
                            || reg_pair_dead_after(lines, start, end, j + 1, 'b', 'c'))
                    {
                        replace.insert(i, vec!["    ld b, h".to_string(), "    ld c, l".to_string()]);
                        let load_repl = match load_dst.as_str() {
                            "bc" => Vec::new(),
                            "hl" => vec!["    ld h, b".to_string(), "    ld l, c".to_string()],
                            "de" => vec!["    ld d, b".to_string(), "    ld e, c".to_string()],
                            _ => unreachable!(),
                        };
                        replace.insert(j, load_repl);
                        changed = true;
                        i = j + 1;
                        continue;
                    }
                }
            }
            i += 1;
        }

        for idx in start..end {
            if let Some(repl) = replace.remove(&idx) {
                out.extend(repl);
            } else {
                out.push(lines[idx].clone());
            }
        }
    }

    (out, changed)
}

use std::collections::BTreeSet;

struct PrivateFnCfg {
    blk_ranges: Vec<(usize, usize)>,
    eff_ends: Vec<usize>,
    succs: Vec<Vec<usize>>,
    hl_rmw_insts: HashMap<usize, MemAddr>,
    safe_ranges: Vec<(u16, u16)>,
}

fn parse_u16_imm(op: &str) -> Option<u16> {
    let t = op.trim().trim_start_matches('#');
    if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        u16::from_str_radix(hex, 16).ok()
    } else {
        None
    }
}

fn build_private_fn_cfg(
    lines: &[String],
    kinds: &[LineKind],
    start: usize,
    end: usize,
    uchiguri_base: Option<u16>,
    ranges: &[(u16, u16)],
) -> Option<PrivateFnCfg> {
    let label_pos = build_label_positions(kinds);
    let mut blk_ranges: Vec<(usize, usize)> = Vec::new();
    let mut label_to_blk: HashMap<String, usize> = HashMap::new();
    let mut cur_blk_start = start;
    let mut dw_targets = false;
    let mut hl_rmw_insts: HashMap<usize, MemAddr> = HashMap::new();
    let mut excluded_range_idxs: HashSet<usize> = HashSet::new();

    for idx in start..end {
        match &kinds[idx] {
            LineKind::Label(lbl) => {
                if idx > cur_blk_start {
                    blk_ranges.push((cur_blk_start, idx));
                }
                let bid = blk_ranges.len();
                label_to_blk.insert(lbl.clone(), bid);
                cur_blk_start = idx;
            }
            LineKind::Inst { mnemonic, ops } => {
                if mnemonic == "jp" && ops.len() == 1 && ops[0].trim().starts_with('(') {
                    return None;
                }
                if mnemonic == "ld" && ops.len() == 2 {
                    let priv_imm = if uchiguri_base.is_none() {
                        parse_u16_imm(&ops[1])
                            .filter(|&imm| in_ranges(imm, ranges))
                            .map(|imm| (imm, MemAddr::Fixed(imm)))
                    } else {
                        uchiguri_base.and_then(|base| {
                            parse_imm_uchiguri_off(&ops[1])
                                .filter(|&off| in_ranges(base.wrapping_add(off), ranges))
                                .map(|off| (base.wrapping_add(off), MemAddr::Uchiguri(off)))
                        })
                    };
                    if let Some((abs_imm, imm_addr)) = priv_imm {
                        let is_rule_3d2 = ops[0].eq_ignore_ascii_case("hl")
                            && next_non_comment(&kinds[..end], idx + 1).is_some_and(|ni| {
                                matches!(
                                    &kinds[ni],
                                    LineKind::Inst { mnemonic: nm, ops: nops }
                                        if matches!(nm.as_str(), "inc" | "dec")
                                            && nops.len() == 1
                                            && nops[0].eq_ignore_ascii_case("(hl)")
                                ) && reg_hl_dead_after(&kinds[..end], &label_pos, ni + 1)
                            });
                        if is_rule_3d2 {
                            let ni = next_non_comment(&kinds[..end], idx + 1).unwrap();
                            hl_rmw_insts.insert(ni, imm_addr);
                        } else {
                            for (ri, &(rlo, rhi)) in ranges.iter().enumerate() {
                                if abs_imm >= rlo && abs_imm < rhi {
                                    excluded_range_idxs.insert(ri);
                                }
                            }
                        }
                    }
                }
                if matches!(
                    mnemonic.as_str(),
                    "jp" | "jr" | "djnz" | "ret" | "reti" | "retn" | "halt"
                ) {
                    blk_ranges.push((cur_blk_start, idx + 1));
                    cur_blk_start = idx + 1;
                }
            }
            _ => {
                let code = lines[idx].split(';').next().unwrap_or("").trim();
                if let Some(rest) = code.strip_prefix(".dw ")
                    && rest.split(',').any(|p| !p.trim().is_empty())
                {
                    dw_targets = true;
                }
            }
        }
    }
    if cur_blk_start < end {
        blk_ranges.push((cur_blk_start, end));
    }
    let safe_ranges: Vec<(u16, u16)> = ranges
        .iter()
        .enumerate()
        .filter_map(|(ri, &r)| (!excluded_range_idxs.contains(&ri)).then_some(r))
        .collect();
    if dw_targets || blk_ranges.is_empty() || safe_ranges.is_empty() {
        return None;
    }

    let nb = blk_ranges.len();
    let mut eff_ends = vec![0usize; nb];
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); nb];
    for (b, &(bs, be)) in blk_ranges.iter().enumerate() {
        let mut falls_through = true;
        let mut blk_end = be;
        for idx in bs..be {
            let LineKind::Inst { mnemonic, ops } = &kinds[idx] else {
                continue;
            };
            if matches!(mnemonic.as_str(), "ret" | "reti" | "retn" | "halt") && ops.is_empty() {
                falls_through = false;
                blk_end = idx + 1;
                break;
            }
            if matches!(mnemonic.as_str(), "jp" | "jr") && ops.len() == 1 {
                if let Some(&tgt) = label_to_blk.get(&ops[0]) {
                    succs[b].push(tgt);
                }
                falls_through = false;
                blk_end = idx + 1;
                break;
            }
            if matches!(mnemonic.as_str(), "jp" | "jr" | "djnz")
                && let Some(t) = ops.last()
                && let Some(&tgt) = label_to_blk.get(t)
            {
                succs[b].push(tgt);
            }
        }
        eff_ends[b] = blk_end;
        if falls_through && b + 1 < nb {
            succs[b].push(b + 1);
        }
    }

    Some(PrivateFnCfg {
        blk_ranges,
        eff_ends,
        succs,
        hl_rmw_insts,
        safe_ranges,
    })
}

fn reg8_slot_index(r: &str) -> Option<usize> {
    match r.trim().to_ascii_lowercase().as_str() {
        "a" => Some(0),
        "b" => Some(1),
        "c" => Some(2),
        "d" => Some(3),
        "e" => Some(4),
        "h" => Some(5),
        "l" => Some(6),
        _ => None,
    }
}

fn reg_byte_indices(op: &str) -> Option<Vec<(u16, usize)>> {
    match op.trim().to_ascii_lowercase().as_str() {
        "a" => Some(vec![(0, 0)]),
        "b" => Some(vec![(0, 1)]),
        "c" => Some(vec![(0, 2)]),
        "d" => Some(vec![(0, 3)]),
        "e" => Some(vec![(0, 4)]),
        "h" => Some(vec![(0, 5)]),
        "l" => Some(vec![(0, 6)]),
        "hl" => Some(vec![(0, 6), (1, 5)]),
        "de" => Some(vec![(0, 4), (1, 3)]),
        "bc" => Some(vec![(0, 2), (1, 1)]),
        _ => None,
    }
}

#[derive(Clone, Default, PartialEq, Eq)]
struct DupSpillState {
    equiv: HashMap<MemAddr, BTreeSet<MemAddr>>,
    reg_bytes: [BTreeSet<MemAddr>; 7],
}

impl DupSpillState {
    fn clear_regs(&mut self) {
        for r in &mut self.reg_bytes {
            r.clear();
        }
    }

    fn clear_all(&mut self) {
        self.equiv.clear();
        self.clear_regs();
    }

    fn intersect_with(&mut self, other: &Self) {
        self.equiv.retain(|k, set_a| {
            if let Some(set_b) = other.equiv.get(k) {
                set_a.retain(|x| set_b.contains(x));
                !set_a.is_empty()
            } else {
                false
            }
        });
        for (ra, rb) in self.reg_bytes.iter_mut().zip(other.reg_bytes.iter()) {
            ra.retain(|x| rb.contains(x));
        }
    }

    fn kill_addr_byte(&mut self, byte_a: MemAddr) {
        if let Some(partners) = self.equiv.remove(&byte_a) {
            for p in partners {
                if let Some(pset) = self.equiv.get_mut(&p) {
                    pset.remove(&byte_a);
                    if pset.is_empty() {
                        self.equiv.remove(&p);
                    }
                }
            }
        }
        for r in &mut self.reg_bytes {
            r.remove(&byte_a);
        }
    }

    fn link_equiv(&mut self, a: MemAddr, b: MemAddr) {
        if a == b {
            return;
        }
        let mut class = BTreeSet::new();
        class.insert(a);
        class.insert(b);
        if let Some(sa) = self.equiv.get(&a) {
            class.extend(sa.iter().copied());
        }
        if let Some(sb) = self.equiv.get(&b) {
            class.extend(sb.iter().copied());
        }
        for &x in &class {
            let mut others = class.clone();
            others.remove(&x);
            self.equiv.insert(x, others);
        }
    }

    fn step(
        &mut self,
        line: &str,
        kind: &LineKind,
        uchiguri_base: Option<u16>,
        ranges: &[(u16, u16)],
        fn_name: Option<&str>,
    ) {
        let LineKind::Inst { mnemonic, ops } = kind else {
            return;
        };
        if mnemonic == "ld" && ops.len() == 2 {
            // Store `ld (addr), reg`
            if let Some(addr) = parse_mem_ref(&ops[0]) {
                let w = reg_byte_width(&ops[1]);
                if in_private_ranges(addr, uchiguri_base, ranges) {
                    for off in 0..w {
                        self.kill_addr_byte(addr.add(off));
                    }
                    if let Some(r_bytes) = reg_byte_indices(&ops[1]) {
                        for (off, r_idx) in r_bytes {
                            let ba = addr.add(off);
                            let existing: Vec<MemAddr> =
                                self.reg_bytes[r_idx].iter().copied().collect();
                            for other in existing {
                                self.link_equiv(ba, other);
                            }
                            self.reg_bytes[r_idx].insert(ba);
                        }
                    }
                }
                return;
            }
            // Load `ld reg, (addr)`
            if let Some(addr) = parse_mem_ref(&ops[1]) {
                if let Some(r_bytes) = reg_byte_indices(&ops[0]) {
                    let is_priv = in_private_ranges(addr, uchiguri_base, ranges);
                    for (off, r_idx) in r_bytes {
                        self.reg_bytes[r_idx].clear();
                        if is_priv {
                            let ba = addr.add(off);
                            self.reg_bytes[r_idx].insert(ba);
                            if let Some(partners) = self.equiv.get(&ba) {
                                self.reg_bytes[r_idx].extend(partners.iter().copied());
                            }
                        }
                    }
                } else {
                    self.clear_regs_written_by(line);
                }
                return;
            }
            // Register-to-register `ld r1, r2`
            if let (Some(d), Some(s)) = (reg8_slot_index(&ops[0]), reg8_slot_index(&ops[1])) {
                if d != s {
                    self.reg_bytes[d] = self.reg_bytes[s].clone();
                }
                return;
            }
            self.clear_regs_written_by(line);
            return;
        }
        if mnemonic == "ex"
            && ops.len() == 2
            && ((ops[0].eq_ignore_ascii_case("de") && ops[1].eq_ignore_ascii_case("hl"))
                || (ops[0].eq_ignore_ascii_case("hl") && ops[1].eq_ignore_ascii_case("de")))
        {
            self.reg_bytes.swap(3, 5); // d <-> h
            self.reg_bytes.swap(4, 6); // e <-> l
            return;
        }
        if matches!(mnemonic.as_str(), "push" | "bit" | "cp" | "nop" | "jp" | "jr" | "ret") {
            return;
        }
        if mnemonic == "call" {
            self.clear_regs();
            if ops.len() != 1 || ops.last().map(String::as_str) == fn_name {
                self.equiv.clear();
            }
            return;
        }
        if matches!(mnemonic.as_str(), "rst" | "exx" | "reti" | "retn" | "halt") {
            self.clear_all();
            return;
        }
        self.clear_regs_written_by(line);
    }

    fn clear_regs_written_by(&mut self, line: &str) {
        const REGS: [char; 7] = ['a', 'b', 'c', 'd', 'e', 'h', 'l'];
        for (idx, r) in REGS.into_iter().enumerate() {
            if inst_reads_writes_reg8(line, r).1 {
                self.reg_bytes[idx].clear();
            }
        }
    }
}

/// Coalesces duplicate private RAM spills (`ld (S1), hl; ld (S2), hl` or `ld (S1), de; ld (S2), de`)
/// within `fn_private_ranges` by forwarding reads of redundant copies to a canonical private slot,
/// allowing `eliminate_dead_scratch_stores` to delete the redundant spill store.
pub(super) fn coalesce_duplicate_private_spills(
    lines: &[String],
    fn_private_ranges: &HashMap<String, Vec<(u16, u16)>>,
) -> (Vec<String>, bool) {
    let uchiguri_base = detect_uchiguri_base(lines);
    let kinds: Vec<LineKind> = lines.iter().map(|l| classify_line(l)).collect();
    let fn_spans = partition_fn_spans(&kinds, fn_private_ranges);

    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut changed = false;

    for (start, end, range_opt) in fn_spans {
        let Some(ranges) = range_opt.filter(|r| !r.is_empty()) else {
            out.extend_from_slice(&lines[start..end]);
            continue;
        };
        let fn_name = match &kinds[start] {
            LineKind::Label(lbl) => Some(lbl.as_str()),
            _ => None,
        };
        let Some(cfg) = build_private_fn_cfg(lines, &kinds, start, end, uchiguri_base, ranges)
        else {
            out.extend_from_slice(&lines[start..end]);
            continue;
        };
        let safe_ranges = &cfg.safe_ranges;

        let mut guarded = vec![false; end - start];
        let mut in_guard = false;
        for idx in start..end {
            let t = lines[idx].trim();
            if t == "; @aun:volatile_begin" || t == "; @aun:asm_begin" {
                in_guard = true;
            } else if t == "; @aun:volatile_end" || t == "; @aun:asm_end" {
                in_guard = false;
            }
            guarded[idx - start] = in_guard;
        }

        let nb = cfg.blk_ranges.len();
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); nb];
        for (b, succ_list) in cfg.succs.iter().enumerate() {
            for &s in succ_list {
                preds[s].push(b);
            }
        }

        let mut in_states: Vec<Option<DupSpillState>> = vec![None; nb];
        let mut out_states: Vec<Option<DupSpillState>> = vec![None; nb];
        in_states[0] = Some(DupSpillState::default());

        for _ in 0..(nb * 4 + 4) {
            let mut df_changed = false;
            for b in 0..nb {
                let st_in = if b == 0 {
                    DupSpillState::default()
                } else {
                    let mut acc: Option<DupSpillState> = None;
                    for &p in &preds[b] {
                        if let Some(ref po) = out_states[p] {
                            match &mut acc {
                                None => acc = Some(po.clone()),
                                Some(cur) => cur.intersect_with(po),
                            }
                        }
                    }
                    let Some(s) = acc else { continue };
                    s
                };
                if in_states[b].as_ref() != Some(&st_in) {
                    in_states[b] = Some(st_in.clone());
                }
                let mut cur = st_in;
                let (bs, _) = cfg.blk_ranges[b];
                for idx in bs..cfg.eff_ends[b] {
                    if guarded[idx - start] {
                        cur.clear_all();
                    } else {
                        cur.step(
                            &lines[idx],
                            &kinds[idx],
                            uchiguri_base,
                            safe_ranges,
                            fn_name,
                        );
                    }
                }
                if out_states[b].as_ref() != Some(&cur) {
                    out_states[b] = Some(cur);
                    df_changed = true;
                }
            }
            if !df_changed {
                break;
            }
        }

        // Collect private slot bases `(addr, width)` and check for overlapping byte accesses.
        let mut slot_access_widths: HashMap<MemAddr, BTreeSet<u16>> = HashMap::new();
        for idx in start..end {
            if let LineKind::Inst { mnemonic, ops } = &kinds[idx]
                && mnemonic == "ld"
                && ops.len() == 2
            {
                if let Some(a) = parse_mem_ref(&ops[0])
                    && in_private_ranges(a, uchiguri_base, safe_ranges)
                {
                    slot_access_widths
                        .entry(a)
                        .or_default()
                        .insert(reg_byte_width(&ops[1]));
                }
                if let Some(a) = parse_mem_ref(&ops[1])
                    && in_private_ranges(a, uchiguri_base, safe_ranges)
                {
                    slot_access_widths
                        .entry(a)
                        .or_default()
                        .insert(reg_byte_width(&ops[0]));
                }
            }
        }
        let clean_slot = |addr: MemAddr, w: u16| -> bool {
            let Some(ws) = slot_access_widths.get(&addr) else {
                return false;
            };
            if ws.len() != 1 || !ws.contains(&w) {
                return false;
            }
            if w == 2 {
                !slot_access_widths.contains_key(&addr.add(1))
                    && addr
                        .sub_one()
                        .is_none_or(|p| !slot_access_widths.get(&p).is_some_and(|pw| pw.contains(&2)))
            } else {
                addr.sub_one()
                    .is_none_or(|p| !slot_access_widths.get(&p).is_some_and(|pw| pw.contains(&2)))
            }
        };

        // At each private read `ld dst, (X)`, compute equivalent partner bases `Y`.
        let mut read_candidates: HashMap<usize, (MemAddr, u16, Vec<MemAddr>)> = HashMap::new();
        let mut must_keep: HashSet<(MemAddr, u16)> = HashSet::new();

        for (b, &(bs, _)) in cfg.blk_ranges.iter().enumerate() {
            let Some(mut cur) = in_states[b].clone() else {
                continue;
            };
            for idx in bs..cfg.eff_ends[b] {
                if guarded[idx - start] {
                    cur.clear_all();
                    continue;
                }
                if let LineKind::Inst { mnemonic, ops } = &kinds[idx]
                    && mnemonic == "ld"
                    && ops.len() == 2
                    && let Some(addr) = parse_mem_ref(&ops[1])
                    && in_private_ranges(addr, uchiguri_base, safe_ranges)
                    && reg_byte_indices(&ops[0]).is_some()
                {
                    let w = reg_byte_width(&ops[0]);
                    if clean_slot(addr, w) {
                        let partners: Vec<MemAddr> = if w == 1 {
                            cur.equiv
                                .get(&addr)
                                .into_iter()
                                .flatten()
                                .copied()
                                .filter(|&y| y != addr && clean_slot(y, 1))
                                .collect()
                        } else if let (Some(s0), Some(s1)) =
                            (cur.equiv.get(&addr), cur.equiv.get(&addr.add(1)))
                        {
                            s0.iter()
                                .copied()
                                .filter(|&y| y != addr && s1.contains(&y.add(1)) && clean_slot(y, 2))
                                .collect()
                        } else {
                            Vec::new()
                        };
                        if partners.is_empty() {
                            must_keep.insert((addr, w));
                        } else {
                            read_candidates.insert(idx, (addr, w, partners));
                        }
                    } else {
                        must_keep.insert((addr, w));
                    }
                }
                cur.step(
                    &lines[idx],
                    &kinds[idx],
                    uchiguri_base,
                    safe_ranges,
                    fn_name,
                );
            }
        }

        let is_scratch = |addr: MemAddr| -> bool {
            let Some(scratch_r) = safe_ranges.first() else {
                return false;
            };
            in_private_ranges(addr, uchiguri_base, std::slice::from_ref(scratch_r))
        };
        let addr_ord = |addr: MemAddr| -> u16 {
            match addr {
                MemAddr::Uchiguri(o) => o,
                MemAddr::Fixed(a) => a,
            }
        };
        let rank_slot = |addr: MemAddr, w: u16| -> (u8, u8, u16) {
            (
                u8::from(!must_keep.contains(&(addr, w))),
                u8::from(is_scratch(addr)),
                addr_ord(addr),
            )
        };

        let mut replace_line: HashMap<usize, String> = HashMap::new();
        for (idx, (addr, w, partners)) in read_candidates {
            let my_rank = rank_slot(addr, w);
            if let Some(best_y) = partners
                .into_iter()
                .filter(|&y| rank_slot(y, w) < my_rank)
                .min_by_key(|&y| rank_slot(y, w))
                && let LineKind::Inst { ops, .. } = &kinds[idx]
            {
                replace_line.insert(idx, format!("    ld {}, ({})", ops[0], best_y.format_imm()));
                changed = true;
            }
        }

        for idx in start..end {
            if let Some(repl) = replace_line.remove(&idx) {
                out.push(repl);
            } else {
                out.push(lines[idx].clone());
            }
        }
    }

    (out, changed)
}

/// Deletes dead stores `ld (0xXXXX), reg` to the *current function's* own private addresses
/// (scratch slots and non-address-taken locals) via CFG backward dataflow liveness over private
/// byte addresses, along with an immediately preceding pure `ld reg, imm` when the very next
/// surviving instruction overwrites `reg` without reading it.
pub(super) fn eliminate_dead_scratch_stores(
    lines: &[String],
    fn_private_ranges: &HashMap<String, Vec<(u16, u16)>>,
) -> (Vec<String>, bool) {
    let uchiguri_base = detect_uchiguri_base(lines);
    let kinds: Vec<LineKind> = lines.iter().map(|l| classify_line(l)).collect();
    let fn_spans = partition_fn_spans(&kinds, fn_private_ranges);

    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut changed = false;
    let label_pos = build_label_positions(&kinds);

    for (start, end, range_opt) in fn_spans {
        let Some(ranges) = range_opt.filter(|r| !r.is_empty()) else {
            out.extend_from_slice(&lines[start..end]);
            continue;
        };
        let Some(cfg) = build_private_fn_cfg(lines, &kinds, start, end, uchiguri_base, ranges)
        else {
            out.extend_from_slice(&lines[start..end]);
            continue;
        };

        let blk_ranges = cfg.blk_ranges;
        let eff_ends = cfg.eff_ends;
        let succs = cfg.succs;
        let hl_rmw_insts = cfg.hl_rmw_insts;
        let safe_ranges = cfg.safe_ranges;
        let nb = blk_ranges.len();
        let mut use_bytes: Vec<HashSet<MemAddr>> = vec![HashSet::new(); nb];
        let mut kill_bytes: Vec<HashSet<MemAddr>> = vec![HashSet::new(); nb];

        for (b, &(bs, _)) in blk_ranges.iter().enumerate() {
            for idx in bs..eff_ends[b] {
                let LineKind::Inst { mnemonic, ops } = &kinds[idx] else {
                    continue;
                };
                if let Some(&addr) = hl_rmw_insts.get(&idx) {
                    if !kill_bytes[b].contains(&addr) {
                        use_bytes[b].insert(addr);
                    }
                    kill_bytes[b].insert(addr);
                }
                if mnemonic == "ld" && ops.len() == 2 {
                    if let Some(addr) = parse_mem_ref(&ops[1]) {
                        let w = reg_byte_width(&ops[0]);
                        for off in 0..w {
                            let byte_a = addr.add(off);
                            if !kill_bytes[b].contains(&byte_a) {
                                use_bytes[b].insert(byte_a);
                            }
                        }
                    }
                    if let Some(addr) = parse_mem_ref(&ops[0]) {
                        let w = reg_byte_width(&ops[1]);
                        for off in 0..w {
                            kill_bytes[b].insert(addr.add(off));
                        }
                    }
                }
            }
        }

        let mut live_in: Vec<HashSet<MemAddr>> = vec![HashSet::new(); nb];
        let mut live_out: Vec<HashSet<MemAddr>> = vec![HashSet::new(); nb];
        loop {
            let mut df_changed = false;
            for b in (0..nb).rev() {
                let mut lo = HashSet::new();
                for &s in &succs[b] {
                    lo.extend(live_in[s].iter().copied());
                }
                let mut li = use_bytes[b].clone();
                for &byte_a in &lo {
                    if !kill_bytes[b].contains(&byte_a) {
                        li.insert(byte_a);
                    }
                }
                if lo != live_out[b] {
                    live_out[b] = lo;
                    df_changed = true;
                }
                if li != live_in[b] {
                    live_in[b] = li;
                    df_changed = true;
                }
            }
            if !df_changed {
                break;
            }
        }

        let mut drop = vec![false; end - start];
        for (b, &(bs, _)) in blk_ranges.iter().enumerate() {
            let mut live = live_out[b].clone();
            for idx in (bs..eff_ends[b]).rev() {
                let LineKind::Inst { mnemonic, ops } = &kinds[idx] else {
                    continue;
                };
                if let Some(&addr) = hl_rmw_insts.get(&idx) {
                    live.insert(addr);
                    continue;
                }
                if mnemonic == "ld" && ops.len() == 2 {
                    if let Some(addr) = parse_mem_ref(&ops[0])
                        && in_private_ranges(addr, uchiguri_base, &safe_ranges)
                    {
                        let w = reg_byte_width(&ops[1]);
                        let any_live = (0..w).any(|off| live.contains(&addr.add(off)));
                        if !any_live {
                            drop[idx - start] = true;
                            changed = true;
                            let reg = ops[1].to_ascii_lowercase();
                            let next_surviving = ((idx + 1)..end).find(|&ni| {
                                !matches!(kinds[ni], LineKind::EmptyOrComment)
                                    && !drop[ni - start]
                            });
                            let next_overwrites = next_surviving.is_some_and(|ni| {
                                match &kinds[ni] {
                                    LineKind::Inst {
                                        mnemonic: nm,
                                        ops: nops,
                                    } => {
                                        (nm == "ld"
                                            && nops.len() == 2
                                            && nops[0].eq_ignore_ascii_case(&reg)
                                            && !nops[1].eq_ignore_ascii_case(&reg)
                                            && !nops[1]
                                                .to_ascii_lowercase()
                                                .contains(&format!("({reg})")))
                                            || (nm == "xor"
                                                && nops.len() == 1
                                                && nops[0].eq_ignore_ascii_case("a")
                                                && reg == "a")
                                    }
                                    _ => false,
                                }
                            });
                            let reg_dead = next_overwrites
                                || match reg.as_str() {
                                    "a" => reg_a_dead_after(&kinds[..end], &label_pos, idx + 1),
                                    "hl" => reg_hl_dead_after(&kinds[..end], &label_pos, idx + 1),
                                    _ => false,
                                };
                            if reg_dead
                                && let Some(prev_inst) = (bs..idx)
                                    .rev()
                                    .find(|&pi| !matches!(kinds[pi], LineKind::EmptyOrComment))
                                && !drop[prev_inst - start]
                                && let LineKind::Inst {
                                    mnemonic: pm,
                                    ops: pops,
                                } = &kinds[prev_inst]
                                && ((pm == "ld"
                                    && pops.len() == 2
                                    && pops[0].eq_ignore_ascii_case(&reg)
                                    && !pops[1].eq_ignore_ascii_case(&reg)
                                    && (!pops[1].starts_with('(')
                                        || parse_mem_ref(&pops[1]).is_some_and(|src_a| {
                                            in_private_ranges(src_a, uchiguri_base, &safe_ranges)
                                        })))
                                    || (pm == "xor"
                                        && pops.len() == 1
                                        && pops[0].eq_ignore_ascii_case("a")
                                        && reg == "a"
                                        && flags_dead_after(&kinds[..end], idx + 1)))
                            {
                                drop[prev_inst - start] = true;
                            }
                            continue;
                        }
                        for off in 0..w {
                            live.remove(&addr.add(off));
                        }
                    } else if let Some(addr) = parse_mem_ref(&ops[0]) {
                        let w = reg_byte_width(&ops[1]);
                        for off in 0..w {
                            live.remove(&addr.add(off));
                        }
                    }
                    if let Some(addr) = parse_mem_ref(&ops[1]) {
                        let w = reg_byte_width(&ops[0]);
                        for off in 0..w {
                            live.insert(addr.add(off));
                        }
                    }
                }
            }
        }

        for idx in start..end {
            if !drop[idx - start] {
                out.push(lines[idx].clone());
            }
        }
    }
    (out, changed)
}

fn is_pure_bb_branch_stub(
    kinds: &[LineKind],
    label_pos: &HashMap<String, usize>,
    target: &str,
    end: usize,
) -> bool {
    if !target.contains("_cmp_ovf_") {
        return false;
    }
    let Some(&pos) = label_pos.get(target) else {
        return false;
    };
    let mut saw_jump = false;
    for k in (pos + 1)..end {
        match &kinds[k] {
            LineKind::EmptyOrComment => {}
            LineKind::Label(_) | LineKind::Directive => break,
            LineKind::Inst { mnemonic, ops } => {
                if matches!(mnemonic.as_str(), "jp" | "jr")
                    && ops.last().is_some_and(|t| is_bb_label(t))
                {
                    saw_jump = true;
                } else {
                    return false;
                }
            }
        }
    }
    saw_jump
}

/// Per-basic-block backward dead-store elimination for single-block colored scratch slots
/// (`fn_sb_ranges`): because these addresses are provably never live across basic-block
/// boundaries (labels, jumps, returns), any store `ld (addr), reg` whose bytes are not read
/// before the next overwrite or end of the basic block is dead and removed!
pub(super) fn eliminate_dead_bb_scratch_stores(
    lines: &[String],
    fn_sb_ranges: &HashMap<String, (u16, u16)>,
) -> (Vec<String>, bool) {
    let uchiguri_base = detect_uchiguri_base(lines);
    let kinds: Vec<LineKind> = lines.iter().map(|l| classify_line(l)).collect();
    let label_pos = build_label_positions(&kinds);
    let wrapped: HashMap<String, Vec<(u16, u16)>> = fn_sb_ranges
        .iter()
        .map(|(k, &r)| (k.clone(), vec![r]))
        .collect();
    let fn_spans = partition_fn_spans(&kinds, &wrapped);

    let mut drop = vec![false; lines.len()];
    let mut changed = false;

    for (start, end, range_opt) in fn_spans {
        let Some(ranges) = range_opt else {
            continue;
        };
        let (lo, hi) = ranges[0];
        let mut has_backward_ref: HashSet<String> = HashSet::new();
        for (idx, kind) in kinds.iter().enumerate().take(end).skip(start) {
            if let LineKind::Inst { mnemonic, ops } = kind
                && matches!(mnemonic.as_str(), "jp" | "jr" | "djnz")
                && let Some(t) = ops.last()
                && label_pos.get(t).is_some_and(|&lp| lp < idx)
            {
                has_backward_ref.insert(t.clone());
            }
        }

        let mut live_bytes: HashSet<MemAddr> = HashSet::new();
        let mut all_sb_live = false;
        let mut live_at_label: HashMap<String, (HashSet<MemAddr>, bool)> = HashMap::new();

        for idx in (start..end).rev() {
            match &kinds[idx] {
                LineKind::EmptyOrComment => {}
                LineKind::Directive => {
                    live_bytes.clear();
                    all_sb_live = false;
                    live_at_label.clear();
                }
                LineKind::Label(lbl) => {
                    if is_bb_label(lbl) || idx == start {
                        live_bytes.clear();
                        all_sb_live = false;
                        live_at_label.clear();
                    } else if has_backward_ref.contains(lbl) {
                        all_sb_live = true;
                        live_at_label.insert(lbl.clone(), (live_bytes.clone(), true));
                    } else {
                        live_at_label.insert(lbl.clone(), (live_bytes.clone(), all_sb_live));
                    }
                }
                LineKind::Inst { mnemonic, ops } => {
                    if matches!(mnemonic.as_str(), "ret" | "reti" | "halt") && ops.is_empty() {
                        live_bytes.clear();
                        all_sb_live = false;
                        continue;
                    }
                    if matches!(mnemonic.as_str(), "jp" | "jr") && ops.len() == 1 {
                        if is_bb_label(&ops[0])
                            || is_pure_bb_branch_stub(&kinds, &label_pos, &ops[0], end)
                        {
                            live_bytes.clear();
                            all_sb_live = false;
                        } else if let Some((tl, ta)) = live_at_label.get(&ops[0]) {
                            live_bytes = tl.clone();
                            all_sb_live = *ta;
                        } else {
                            all_sb_live = true;
                        }
                        continue;
                    }
                    if matches!(mnemonic.as_str(), "jp" | "jr" | "djnz" | "ret") {
                        if mnemonic == "ret" {
                            continue;
                        }
                        if let Some(t) = ops.last() {
                            if is_bb_label(t)
                                || is_pure_bb_branch_stub(&kinds, &label_pos, t, end)
                            {
                                continue;
                            } else if let Some((tl, ta)) = live_at_label.get(t) {
                                live_bytes.extend(tl.iter().copied());
                                all_sb_live |= *ta;
                                continue;
                            }
                        }
                        all_sb_live = true;
                        continue;
                    }
                    if mnemonic == "ld" && ops.len() == 2 {
                        // Store `ld (addr), reg`
                        if let Some(addr) = parse_mem_ref(&ops[0]) {
                            let is_16 = matches!(
                                ops[1].to_ascii_lowercase().as_str(),
                                "hl" | "de" | "bc" | "ix" | "iy" | "sp"
                            );
                            if !all_sb_live && addr_in_sb(addr, uchiguri_base, 1, lo, hi) {
                                let lo_live = live_bytes.contains(&addr);
                                let hi_live = is_16 && live_bytes.contains(&addr.add(1));
                                if !lo_live && !hi_live {
                                    drop[idx] = true;
                                    changed = true;
                                    continue;
                                }
                            }
                            live_bytes.remove(&addr);
                            if is_16 {
                                live_bytes.remove(&addr.add(1));
                            }
                            continue;
                        }
                        // Load `ld reg, (addr)`
                        if let Some(addr) = parse_mem_ref(&ops[1]) {
                            let is_16 = matches!(
                                ops[0].to_ascii_lowercase().as_str(),
                                "hl" | "de" | "bc" | "ix" | "iy" | "sp"
                            );
                            live_bytes.insert(addr);
                            if is_16 {
                                live_bytes.insert(addr.add(1));
                            }
                            continue;
                        }
                    }
                }
            }
        }
    }

    if !changed {
        return (lines.to_vec(), false);
    }
    let out: Vec<String> = lines
        .iter()
        .enumerate()
        .filter(|&(i, _)| !drop[i])
        .map(|(_, l)| l.clone())
        .collect();
    (out, true)
}

#[derive(Clone, Copy)]
enum StackSpillKind {
    Reg16(&'static str),
    RegA,
}

impl StackSpillKind {
    fn from_reg(op: &str) -> Option<Self> {
        if op.eq_ignore_ascii_case("hl") {
            Some(Self::Reg16("hl"))
        } else if op.eq_ignore_ascii_case("de") {
            Some(Self::Reg16("de"))
        } else if op.eq_ignore_ascii_case("bc") {
            Some(Self::Reg16("bc"))
        } else if op.eq_ignore_ascii_case("a") {
            Some(Self::RegA)
        } else {
            None
        }
    }

    fn width(self) -> u16 {
        match self {
            Self::Reg16(_) => 2,
            Self::RegA => 1,
        }
    }

    fn lo_reg8(self) -> &'static str {
        match self {
            Self::Reg16("hl") => "l",
            Self::Reg16("de") => "e",
            Self::Reg16("bc") => "c",
            _ => "a",
        }
    }

    fn push_lines(self) -> Vec<String> {
        match self {
            Self::Reg16(r) => vec![format!("    push {r}")],
            Self::RegA => vec!["    push af".to_string()],
        }
    }

    fn pop_lines(self) -> Vec<String> {
        match self {
            Self::Reg16(r) => vec![format!("    pop {r}")],
            Self::RegA => vec!["    pop af".to_string()],
        }
    }
}

/// Returns `true` if `pop af` at `load_idx` cannot observe/corrupt live CPU flags.
fn pop_af_flags_safe(kinds: &[LineKind], start: usize) -> bool {
    let mut k = start;
    while k < kinds.len() {
        match &kinds[k] {
            LineKind::EmptyOrComment => k += 1,
            LineKind::Label(_) | LineKind::Directive => return false,
            kind @ LineKind::Inst { mnemonic, ops } => {
                if reads_flags(kind)
                    || (mnemonic == "ex" && ops.first().is_some_and(|o| o.eq_ignore_ascii_case("af")))
                {
                    return false;
                }
                if clobbers_all_flags(kind) {
                    return true;
                }
                if (matches!(mnemonic.as_str(), "ret" | "reti") && ops.is_empty())
                    || (matches!(mnemonic.as_str(), "call") && ops.len() == 1)
                    || (matches!(mnemonic.as_str(), "jp" | "jr")
                        && ops.len() == 1
                        && is_bb_label(&ops[0]))
                {
                    return true;
                }
                if matches!(
                    mnemonic.as_str(),
                    "ld" | "push" | "pop" | "ex" | "nop" | "di" | "ei" | "out" | "inc" | "dec"
                ) {
                    k += 1;
                    continue;
                }
                return false;
            }
        }
    }
    false
}

/// Promotes remaining balanced, LIFO-nested single-block scratch spills (`ld (nn), hl/de/bc/a`
/// ... `ld hl/de/bc/a, (nn)`) in `fn_sb_ranges` and function-unique private locals to `push`/`pop`
/// (including 16-bit store -> 8-bit load spills via `push af`/`pop af` or `push rr`/`pop af`).
pub(super) fn promote_single_block_spills_to_stack(
    lines: &[String],
    fn_private_ranges: &HashMap<String, Vec<(u16, u16)>>,
    fn_sb_ranges: &HashMap<String, (u16, u16)>,
) -> (Vec<String>, bool) {
    let uchiguri_base = detect_uchiguri_base(lines);
    let kinds: Vec<LineKind> = lines.iter().map(|l| classify_line(l)).collect();
    let label_pos = build_label_positions(&kinds);
    let mut combined_ranges: HashMap<String, Vec<(u16, u16)>> = fn_private_ranges.clone();
    for (k, &r) in fn_sb_ranges {
        combined_ranges.entry(k.clone()).or_insert_with(|| vec![r]);
    }
    let fn_spans = partition_fn_spans(&kinds, &combined_ranges);

    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut changed = false;

    for (start, end, _) in fn_spans {
        let sb_range = if let LineKind::Label(lbl) = &kinds[start] {
            fn_sb_ranges.get(lbl).copied()
        } else {
            None
        };
        let priv_ranges = if let LineKind::Label(lbl) = &kinds[start] {
            fn_private_ranges.get(lbl).map(Vec::as_slice)
        } else {
            None
        };
        if sb_range.is_none() && priv_ranges.is_none_or(|r| r.is_empty()) {
            out.extend_from_slice(&lines[start..end]);
            continue;
        }

        let mut ref_count: HashMap<MemAddr, usize> = HashMap::new();
        let mut imm_taken_ranges: Vec<(u16, u16)> = Vec::new();
        for kind in &kinds[start..end] {
            if let LineKind::Inst { mnemonic, ops } = kind
                && mnemonic == "ld"
                && ops.len() == 2
            {
                for op in ops {
                    if let Some(a) = parse_mem_ref(op) {
                        *ref_count.entry(a).or_insert(0) += 1;
                    }
                }
                if let Some(pr) = priv_ranges {
                    let abs_imm = if uchiguri_base.is_none() {
                        parse_u16_imm(&ops[1]).filter(|&imm| in_ranges(imm, pr))
                    } else {
                        uchiguri_base.and_then(|base| {
                            parse_imm_uchiguri_off(&ops[1])
                                .map(|off| base.wrapping_add(off))
                                .filter(|&imm| in_ranges(imm, pr))
                        })
                    };
                    if let Some(imm) = abs_imm {
                        for &(rlo, rhi) in pr {
                            if imm >= rlo && imm < rhi {
                                imm_taken_ranges.push((rlo, rhi));
                            }
                        }
                    }
                }
            }
        }

        let mut replace: HashMap<usize, Vec<String>> = HashMap::new();
        // Partition the function into straight-line intra-block regions separated by labels/directives.
        let mut reg_start = start;
        while reg_start < end {
            if matches!(kinds[reg_start], LineKind::Label(_) | LineKind::Directive) {
                reg_start += 1;
                continue;
            }
            let mut reg_end = reg_start;
            while reg_end < end
                && !matches!(kinds[reg_end], LineKind::Label(_) | LineKind::Directive)
            {
                reg_end += 1;
            }

            #[derive(Clone)]
            struct StackCand {
                store_idx: usize,
                load_idx: usize,
                push: Vec<String>,
                pop: Vec<String>,
            }
            let mut cands: Vec<StackCand> = Vec::new();

            for i in reg_start..reg_end {
                let LineKind::Inst { mnemonic, ops } = &kinds[i] else {
                    continue;
                };
                if mnemonic != "ld" || ops.len() != 2 {
                    continue;
                }
                let Some(addr) = parse_mem_ref(&ops[0]) else {
                    continue;
                };
                let Some(store_kind) = StackSpillKind::from_reg(&ops[1]) else {
                    continue;
                };
                let sz = store_kind.width();
                let is_sb =
                    sb_range.is_some_and(|(lo, hi)| addr_in_sb(addr, uchiguri_base, sz, lo, hi));
                let is_priv_safe = priv_ranges.is_some_and(|pr| {
                    in_private_ranges(addr, uchiguri_base, pr)
                        && !in_private_ranges(addr, uchiguri_base, &imm_taken_ranges)
                        && addr
                            .sub_one()
                            .is_none_or(|p| ref_count.get(&p).copied().unwrap_or(0) == 0)
                });
                if !is_sb && !is_priv_safe {
                    continue;
                }

                let mut depth: isize = 0;
                let mut found: Option<(usize, StackSpillKind, bool)> = None;
                for j in (i + 1)..reg_end {
                    let LineKind::Inst {
                        mnemonic: m2,
                        ops: ops2,
                    } = &kinds[j]
                    else {
                        continue;
                    };
                    // Abort on any branch, jump, return, or SP mutation.
                    if matches!(
                        m2.as_str(),
                        "jp" | "jr" | "djnz" | "ret" | "reti" | "retn" | "halt" | "rst"
                    ) {
                        break;
                    }
                    if m2 == "call" && ops2.len() != 1 {
                        break; // conditional call
                    }
                    if m2 == "ld" && ops2.first().is_some_and(|o| o.eq_ignore_ascii_case("sp")) {
                        break;
                    }
                    if ops2.iter().any(|o| o.eq_ignore_ascii_case("(sp)"))
                        || ((m2 == "add" || m2 == "inc" || m2 == "dec")
                            && ops2.first().is_some_and(|o| o.eq_ignore_ascii_case("sp")))
                    {
                        break;
                    }

                    if m2 == "push" {
                        depth += 1;
                        continue;
                    }
                    if m2 == "pop" {
                        depth -= 1;
                        if depth < 0 {
                            break;
                        }
                        continue;
                    }

                    if m2 == "ld" && ops2.len() == 2 {
                        if let Some(load_addr) = parse_mem_ref(&ops2[1])
                            && mem_ranges_overlap(load_addr, reg_byte_width(&ops2[0]), addr, sz)
                        {
                            if depth == 0
                                && let Some(lk) = StackSpillKind::from_reg(&ops2[0])
                            {
                                if load_addr == addr && lk.width() == sz {
                                    found = Some((j, lk, false));
                                } else if sz == 2
                                    && matches!(lk, StackSpillKind::RegA)
                                    && (load_addr == addr || load_addr == addr.add(1))
                                {
                                    found = Some((j, lk, load_addr == addr.add(1)));
                                }
                            }
                            break;
                        }
                        if let Some(st_addr) = parse_mem_ref(&ops2[0])
                            && mem_ranges_overlap(st_addr, reg_byte_width(&ops2[1]), addr, sz)
                        {
                            break;
                        }
                    }
                }

                let Some((load_idx, load_kind, is_hi_byte)) = found else {
                    continue;
                };

                let is_fn_unique = is_priv_safe
                    && if is_hi_byte {
                        ref_count.get(&addr).copied() == Some(1)
                            && ref_count.get(&addr.add(1)).copied() == Some(1)
                    } else {
                        ref_count.get(&addr).copied() == Some(2)
                            && (sz == 1 || ref_count.get(&addr.add(1)).copied().unwrap_or(0) == 0)
                    };
                if !is_sb && !is_fn_unique {
                    continue;
                }

                // Verify `[addr, addr + sz)` is not read again after `load_idx` before overwrite or BB end.
                if is_sb && !sb_slot_dead_after_load(&kinds, &label_pos, load_idx, end, addr, sz) {
                    continue;
                }

                if matches!(load_kind, StackSpillKind::RegA)
                    && !pop_af_flags_safe(&kinds[..reg_end], load_idx + 1)
                {
                    continue;
                }

                let (push_lines, pop_lines) = if sz == 2 && matches!(load_kind, StackSpillKind::RegA) {
                    if is_hi_byte {
                        (store_kind.push_lines(), vec!["    pop af".to_string()])
                    } else {
                        if !reg_a_dead_after(&kinds[..reg_end], &label_pos, i + 1) {
                            continue;
                        }
                        (
                            vec![
                                format!("    ld a, {}", store_kind.lo_reg8()),
                                "    push af".to_string(),
                            ],
                            vec!["    pop af".to_string()],
                        )
                    }
                } else {
                    (store_kind.push_lines(), load_kind.pop_lines())
                };

                cands.push(StackCand {
                    store_idx: i,
                    load_idx,
                    push: push_lines,
                    pop: pop_lines,
                });
            }

            // Select a maximal strictly LIFO-nested / disjoint subset (innermost first).
            cands.sort_by_key(|c| (c.load_idx - c.store_idx, c.store_idx));
            let mut chosen: Vec<(usize, usize)> = Vec::new();
            for c in cands {
                let compatible = chosen.iter().all(|&(cs, cl)| {
                    c.load_idx < cs
                        || cl < c.store_idx
                        || (cs < c.store_idx && c.load_idx < cl)
                        || (c.store_idx < cs && cl < c.load_idx)
                });
                if compatible {
                    chosen.push((c.store_idx, c.load_idx));
                    replace.insert(c.store_idx, c.push);
                    replace.insert(c.load_idx, c.pop);
                    changed = true;
                }
            }

            reg_start = reg_end;
        }

        for idx in start..end {
            if let Some(repl) = replace.remove(&idx) {
                out.extend(repl);
            } else {
                out.push(lines[idx].clone());
            }
        }
    }

    (out, changed)
}
