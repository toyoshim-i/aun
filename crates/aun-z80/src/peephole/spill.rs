use super::*;

/// Promotes a single-write / single-read 16-bit private RAM spill (`ld (S), hl` ... `ld hl, (S)`)
/// within the same basic block to `BC` (`ld b, h; ld c, l` ... `ld h, b; ld l, c`) whenever
/// `BC` is untouched in between and `S` is referenced nowhere else in the function.
pub(super) fn promote_single_block_spills_to_bc(
    lines: &[String],
    fn_private_ranges: &HashMap<String, Vec<(u16, u16)>>,
    fn_sb_ranges: &HashMap<String, (u16, u16)>,
    no_bc_fns: &HashSet<String>,
) -> (Vec<String>, bool) {
    let kinds: Vec<LineKind> = lines.iter().map(|l| classify_line(l)).collect();
    let fn_spans = partition_fn_spans(&kinds, fn_private_ranges);

    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut changed = false;

    for (start, end, range_opt) in fn_spans {
        let Some(ranges) = range_opt else {
            out.extend_from_slice(&lines[start..end]);
            continue;
        };
        // Skip BC spill promotion in functions that already use `C`/`BC` or must preserve `C` for WPO callers
        let is_no_bc = matches!(&kinds[start], LineKind::Label(l) if no_bc_fns.contains(l));
        let fn_uses_c = is_no_bc || kinds[start..end].iter().any(|k| {
            if let LineKind::Inst { ops, .. } = k {
                ops.iter().any(|o| {
                    let ol = o.to_ascii_lowercase();
                    matches!(ol.as_str(), "c" | "bc" | "(c)" | "(bc)")
                })
            } else {
                false
            }
        });
        if fn_uses_c {
            out.extend_from_slice(&lines[start..end]);
            continue;
        }
        let sb_range = if let LineKind::Label(lbl) = &kinds[start] {
            fn_sb_ranges.get(lbl).copied()
        } else {
            None
        };

        // Count all memory references (reads and writes) to each address in this function.
        let mut ref_count: HashMap<u16, usize> = HashMap::new();
        for kind in &kinds[start..end] {
            if let LineKind::Inst { mnemonic, ops } = kind
                && mnemonic == "ld"
                && ops.len() == 2
            {
                for op in ops {
                    if let Some(a) = parse_mem_addr(op) {
                        *ref_count.entry(a).or_insert(0) += 1;
                    }
                }
            }
        }

        // Map line indices to replacement lines when a pair is promoted.
        let mut replace: HashMap<usize, Vec<String>> = HashMap::new();
        let mut i = start;
        while i < end {
            if let LineKind::Inst { mnemonic, ops } = &kinds[i]
                && mnemonic == "ld"
                && ops.len() == 2
                && let Some(addr) = parse_mem_addr(&ops[0])
                && ops[1].eq_ignore_ascii_case("hl")
                && in_ranges(addr, ranges)
                && ref_count.get(&addr.wrapping_add(1)).copied().unwrap_or(0) == 0
                && (addr == 0 || ref_count.get(&(addr - 1)).copied().unwrap_or(0) == 0)
            {
                let is_sb = sb_range.is_some_and(|(lo, hi)| addr >= lo && addr < hi);
                let is_fn_unique = ref_count.get(&addr).copied() == Some(2);
                if is_sb || is_fn_unique {
                    // Scan forward in the same basic block for the matching `ld hl, (addr)`
                    let mut found_read: Option<usize> = None;
                    for j in (i + 1)..end {
                        if let LineKind::Inst {
                            mnemonic: m2,
                            ops: ops2,
                        } = &kinds[j]
                            && m2 == "ld"
                            && ops2.len() == 2
                            && ops2[0].eq_ignore_ascii_case("hl")
                            && parse_mem_addr(&ops2[1]) == Some(addr)
                        {
                            found_read = Some(j);
                            break;
                        }
                        if inst_touches_bc(&kinds[j]) {
                            break;
                        }
                    }
                    // For a shared single-block scratch slot, ensure `addr` is not read again
                    // in the remainder of this basic block after `j` (before overwrite or BB end).
                    let safe_after = if is_fn_unique {
                        true
                    } else if let Some(j) = found_read {
                        let mut ok = true;
                        for k in (j + 1)..end {
                            match &kinds[k] {
                                LineKind::Label(lbl) => {
                                    // True IR BB boundary: sb scratch is dead; synthetic label: conservative!
                                    if !is_bb_label(lbl) {
                                        ok = false;
                                    }
                                    break;
                                }
                                LineKind::Directive => break,
                                LineKind::Inst { mnemonic: km, ops: kops } => {
                                    if matches!(km.as_str(), "ret" | "reti" | "halt") && kops.is_empty() {
                                        break;
                                    }
                                    if matches!(km.as_str(), "jp" | "jr") && kops.len() == 1 {
                                        if !is_bb_label(&kops[0]) {
                                            ok = false;
                                        }
                                        break;
                                    }
                                    // Conditional branch / djnz inside IR block: conservative!
                                    if matches!(km.as_str(), "jp" | "jr" | "djnz" | "ret") {
                                        ok = false;
                                        break;
                                    }
                                    if km == "ld" && kops.len() == 2 {
                                        if parse_mem_addr(&kops[1]) == Some(addr) {
                                            ok = false;
                                            break;
                                        }
                                        if parse_mem_addr(&kops[0]) == Some(addr) {
                                            break;
                                        }
                                    }
                                }
                                LineKind::EmptyOrComment => {}
                            }
                        }
                        ok
                    } else {
                        false
                    };
                    if safe_after && let Some(j) = found_read {
                        replace.insert(i, vec!["    ld b, h".to_string(), "    ld c, l".to_string()]);
                        replace.insert(j, vec!["    ld h, b".to_string(), "    ld l, c".to_string()]);
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

/// Deletes stores `ld (0xXXXX), reg` to the *current function's* own private addresses
/// (scratch slots and non-address-taken locals) that are never read anywhere within that function,
/// along with an immediately preceding pure `ld reg, imm` when the very next instruction
/// overwrites `reg` without reading it.
pub(super) fn eliminate_dead_scratch_stores(
    lines: &[String],
    fn_private_ranges: &HashMap<String, Vec<(u16, u16)>>,
) -> (Vec<String>, bool) {
    let kinds: Vec<LineKind> = lines.iter().map(|l| classify_line(l)).collect();
    let fn_spans = partition_fn_spans(&kinds, fn_private_ranges);

    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut changed = false;

    for (start, end, range_opt) in fn_spans {
        let Some(ranges) = range_opt else {
            out.extend_from_slice(&lines[start..end]);
            continue;
        };

        let mut read_addrs: HashSet<u16> = HashSet::new();
        for kind in &kinds[start..end] {
            if let LineKind::Inst { mnemonic, ops } = kind
                && mnemonic == "ld"
                && ops.len() == 2
                && let Some(addr) = parse_mem_addr(&ops[1])
            {
                read_addrs.insert(addr);
                if matches!(
                    ops[0].to_ascii_lowercase().as_str(),
                    "hl" | "de" | "bc" | "ix" | "iy"
                ) {
                    read_addrs.insert(addr.wrapping_add(1));
                }
            }
        }

        let mut i = start;
        while i < end {
            if let LineKind::Inst { mnemonic, ops } = &kinds[i]
                && mnemonic == "ld"
                && ops.len() == 2
                && let Some(addr) = parse_mem_addr(&ops[0])
                && in_ranges(addr, ranges)
                && !read_addrs.contains(&addr)
                && !(matches!(
                    ops[1].to_ascii_lowercase().as_str(),
                    "hl" | "de" | "bc" | "ix" | "iy"
                ) && read_addrs.contains(&addr.wrapping_add(1)))
            {
                // Drop an immediately preceding pure `ld reg, imm` ONLY if the next instruction
                // unconditionally overwrites `reg` without reading it (NEVER across `ret`!).
                let reg = ops[1].to_ascii_lowercase();
                let next_overwrites = next_non_comment(&kinds[..end], i + 1).is_some_and(|ni| {
                    match &kinds[ni] {
                        LineKind::Inst {
                            mnemonic: nm,
                            ops: nops,
                        } => {
                            (nm == "ld"
                                && nops.len() == 2
                                && nops[0].eq_ignore_ascii_case(&reg)
                                && !nops[1].eq_ignore_ascii_case(&reg)
                                && !nops[1].to_ascii_lowercase().contains(&format!("({reg})")))
                                || (nm == "xor"
                                    && nops.len() == 1
                                    && nops[0].eq_ignore_ascii_case("a")
                                    && reg == "a")
                        }
                        _ => false,
                    }
                });
                // Never pop `xor a` (modifies flags); only pop pure flag-preserving `ld reg, imm`
                if next_overwrites
                    && let Some(prev_line) = out.last()
                    && let LineKind::Inst {
                        mnemonic: pm,
                        ops: pops,
                    } = classify_line(prev_line)
                    && pm == "ld"
                    && pops.len() == 2
                    && pops[0].eq_ignore_ascii_case(&reg)
                    && !pops[1].starts_with('(')
                {
                    out.pop();
                }
                changed = true;
                i += 1;
                continue;
            }
            out.push(lines[i].clone());
            i += 1;
        }
    }
    (out, changed)
}

/// Per-basic-block backward dead-store elimination for single-block colored scratch slots
/// (`fn_sb_ranges`): because these addresses are provably never live across basic-block
/// boundaries (labels, jumps, returns), any store `ld (addr), reg` whose bytes are not read
/// before the next overwrite or end of the basic block is dead and removed!
pub(super) fn eliminate_dead_bb_scratch_stores(
    lines: &[String],
    fn_sb_ranges: &HashMap<String, (u16, u16)>,
) -> (Vec<String>, bool) {
    let kinds: Vec<LineKind> = lines.iter().map(|l| classify_line(l)).collect();
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
        let mut live_bytes: HashSet<u16> = HashSet::new();
        // When we encounter a synthetic intra-IR-block label or conditional branch,
        // conservatively treat all `sb` scratch slots as live until the next true IR BB boundary!
        let mut all_sb_live = false;

        for idx in (start..end).rev() {
            match &kinds[idx] {
                LineKind::EmptyOrComment => {}
                LineKind::Directive => {
                    live_bytes.clear();
                    all_sb_live = false;
                }
                LineKind::Label(lbl) => {
                    if is_bb_label(lbl) || idx == start {
                        live_bytes.clear();
                        all_sb_live = false;
                    } else {
                        // Synthetic intra-block label (e.g. `_mul8_lp`, `_sh_dn`, `_cmp_`):
                        // values stored before this label may be read after it!
                        all_sb_live = true;
                    }
                }
                LineKind::Inst { mnemonic, ops } => {
                    if matches!(mnemonic.as_str(), "ret" | "reti" | "halt") && ops.is_empty() {
                        live_bytes.clear();
                        all_sb_live = false;
                        continue;
                    }
                    if matches!(mnemonic.as_str(), "jp" | "jr") && ops.len() == 1 {
                        if is_bb_label(&ops[0]) {
                            live_bytes.clear();
                            all_sb_live = false;
                        } else {
                            all_sb_live = true;
                        }
                        continue;
                    }
                    if matches!(mnemonic.as_str(), "jp" | "jr" | "djnz" | "ret") {
                        // Conditional branch: execution may fall through or branch to synthetic label
                        if ops.last().is_none_or(|t| !is_bb_label(t)) {
                            all_sb_live = true;
                        }
                        continue;
                    }
                    if mnemonic == "ld" && ops.len() == 2 {
                        // Store `ld (addr), reg`
                        if let Some(addr) = parse_mem_addr(&ops[0]) {
                            let is_16 = matches!(
                                ops[1].to_ascii_lowercase().as_str(),
                                "hl" | "de" | "bc" | "ix" | "iy" | "sp"
                            );
                            if !all_sb_live && addr >= lo && addr < hi {
                                let lo_live = live_bytes.contains(&addr);
                                let hi_live = is_16 && live_bytes.contains(&addr.wrapping_add(1));
                                if !lo_live && !hi_live {
                                    drop[idx] = true;
                                    changed = true;
                                    continue;
                                }
                            }
                            live_bytes.remove(&addr);
                            if is_16 {
                                live_bytes.remove(&addr.wrapping_add(1));
                            }
                            continue;
                        }
                        // Load `ld reg, (addr)`
                        if let Some(addr) = parse_mem_addr(&ops[1]) {
                            let is_16 = matches!(
                                ops[0].to_ascii_lowercase().as_str(),
                                "hl" | "de" | "bc" | "ix" | "iy" | "sp"
                            );
                            live_bytes.insert(addr);
                            if is_16 {
                                live_bytes.insert(addr.wrapping_add(1));
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
