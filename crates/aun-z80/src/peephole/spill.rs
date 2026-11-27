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

fn inst_touches_mem_range(kind: &LineKind, addr: MemAddr) -> bool {
    let LineKind::Inst { mnemonic, ops } = kind else {
        return false;
    };
    if mnemonic == "ld" && ops.len() == 2 {
        if let Some(wa) = parse_mem_ref(&ops[0])
            && mem_ranges_overlap(wa, reg_byte_width(&ops[1]), addr, 2)
        {
            return true;
        }
        if let Some(ra) = parse_mem_ref(&ops[1])
            && mem_ranges_overlap(ra, reg_byte_width(&ops[0]), addr, 2)
        {
            return true;
        }
    }
    false
}

fn addr_in_sb(
    addr: MemAddr,
    uchiguri_base: Option<u16>,
    sz: u16,
    lo: u16,
    hi: u16,
) -> bool {
    let resolved = match (addr, uchiguri_base) {
        (MemAddr::Uchiguri(off), Some(base)) => base.wrapping_add(off),
        (MemAddr::Fixed(a), None) => a,
        _ => return false,
    };
    resolved >= lo && (resolved as u32) + (sz as u32) <= (hi as u32)
}

/// Promotes a single-write / single-read 16-bit private RAM spill (`ld (S), hl` ... `ld hl, (S)`)
/// within the same basic block to `BC` (`ld b, h; ld c, l` ... `ld h, b; ld l, c`) whenever
/// `BC` is untouched in between and `S` is referenced nowhere else in the function.
pub(super) fn promote_single_block_spills_to_bc(
    lines: &[String],
    fn_private_ranges: &HashMap<String, Vec<(u16, u16)>>,
    fn_sb_ranges: &HashMap<String, (u16, u16)>,
    no_bc_fns: &HashSet<String>,
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
            // Skip BC spill promotion in functions that already use `C`/`BC` or preserve `B`/`C` across calls.
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
        let mut i = start;
        while i < end {
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
                    sb_range.is_some_and(|(lo, hi)| addr_in_sb(addr, uchiguri_base, 1, lo, hi));
                let is_fn_unique = ref_count.get(&addr).copied() == Some(2);
                if is_sb || is_fn_unique {
                    // Scan forward in the same basic block for the matching `ld hl, (addr)`
                    let mut found_read: Option<usize> = None;
                    for j in (i + 1)..end {
                        match &kinds[j] {
                            LineKind::Label(_) | LineKind::Directive => break,
                            LineKind::EmptyOrComment => continue,
                            LineKind::Inst {
                                mnemonic: m2,
                                ops: ops2,
                            } => {
                                if m2 == "ld"
                                    && ops2.len() == 2
                                    && ops2[0].eq_ignore_ascii_case("hl")
                                    && parse_mem_ref(&ops2[1]) == Some(addr)
                                {
                                    found_read = Some(j);
                                    break;
                                }
                                if inst_touches_bc(&kinds[j]) {
                                    break;
                                }
                                if inst_touches_mem_range(&kinds[j], addr) {
                                    break;
                                }
                            }
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
                                        if let Some(ra) = parse_mem_ref(&kops[1])
                                            && mem_ranges_overlap(ra, reg_byte_width(&kops[0]), addr, 2)
                                        {
                                            ok = false;
                                            break;
                                        }
                                        if let Some(wa) = parse_mem_ref(&kops[0])
                                            && wa == addr
                                            && reg_byte_width(&kops[1]) == 2
                                        {
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

    let parse_u16_imm = |op: &str| -> Option<u16> {
        let t = op.trim().trim_start_matches('#');
        if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
            u16::from_str_radix(hex, 16).ok()
        } else {
            None
        }
    };

    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut changed = false;

    for (start, end, range_opt) in fn_spans {
        let Some(ranges) = range_opt.filter(|r| !r.is_empty()) else {
            out.extend_from_slice(&lines[start..end]);
            continue;
        };

        // Partition `[start, end)` into basic blocks separated by labels/directives,
        // aborting conservatively if any indirect jump or immediate private address load exists.
        let mut blk_ranges: Vec<(usize, usize)> = Vec::new();
        let mut label_to_blk: HashMap<String, usize> = HashMap::new();
        let mut cur_blk_start = start;
        let mut conservative_bail = false;
        let mut dw_targets: Vec<String> = Vec::new();

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
                    if mnemonic == "jp"
                        && ops.len() == 1
                        && ops[0].trim().starts_with('(')
                    {
                        conservative_bail = true;
                        break;
                    }
                    if mnemonic == "ld" && ops.len() == 2 {
                        let loads_priv_imm = (uchiguri_base.is_none()
                            && parse_u16_imm(&ops[1]).is_some_and(|imm| in_ranges(imm, ranges)))
                            || uchiguri_base.is_some_and(|base| {
                                parse_imm_uchiguri_off(&ops[1])
                                    .is_some_and(|off| in_ranges(base.wrapping_add(off), ranges))
                            });
                        if loads_priv_imm {
                            conservative_bail = true;
                            break;
                        }
                    }
                }
                _ => {
                    let code = lines[idx].split(';').next().unwrap_or("").trim();
                    if let Some(rest) = code.strip_prefix(".dw ") {
                        for p in rest.split(',') {
                            let s = p.trim();
                            if !s.is_empty() {
                                dw_targets.push(s.to_string());
                            }
                        }
                    }
                }
            }
        }
        if !conservative_bail && cur_blk_start < end {
            blk_ranges.push((cur_blk_start, end));
        }
        if conservative_bail || !dw_targets.is_empty() || blk_ranges.is_empty() {
            out.extend_from_slice(&lines[start..end]);
            continue;
        }

        let nb = blk_ranges.len();
        let mut eff_ends = vec![0usize; nb];
        let mut succs: Vec<Vec<usize>> = vec![Vec::new(); nb];
        let mut use_bytes: Vec<HashSet<MemAddr>> = vec![HashSet::new(); nb];
        let mut kill_bytes: Vec<HashSet<MemAddr>> = vec![HashSet::new(); nb];

        for (b, &(bs, be)) in blk_ranges.iter().enumerate() {
            let mut falls_through = true;
            let mut blk_end = be;
            for idx in bs..be {
                let LineKind::Inst { mnemonic, ops } = &kinds[idx] else {
                    continue;
                };
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
                if matches!(mnemonic.as_str(), "ret" | "reti" | "retn" | "halt")
                    && ops.is_empty()
                {
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
                if mnemonic == "ld" && ops.len() == 2 {
                    if let Some(addr) = parse_mem_ref(&ops[0])
                        && in_private_ranges(addr, uchiguri_base, ranges)
                    {
                        let w = reg_byte_width(&ops[1]);
                        let any_live = (0..w).any(|off| live.contains(&addr.add(off)));
                        if !any_live {
                            drop[idx - start] = true;
                            changed = true;
                            let reg = ops[1].to_ascii_lowercase();
                            let next_overwrites =
                                next_non_comment(&kinds[..end], idx + 1).is_some_and(|ni| {
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
                            if next_overwrites
                                && let Some(prev_inst) = (bs..idx)
                                    .rev()
                                    .find(|&pi| !matches!(kinds[pi], LineKind::EmptyOrComment))
                                && !drop[prev_inst - start]
                                && let LineKind::Inst {
                                    mnemonic: pm,
                                    ops: pops,
                                } = &kinds[prev_inst]
                                && pm == "ld"
                                && pops.len() == 2
                                && pops[0].eq_ignore_ascii_case(&reg)
                                && !pops[1].starts_with('(')
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
        let mut live_bytes: HashSet<MemAddr> = HashSet::new();
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

    fn push_line(self) -> String {
        match self {
            Self::Reg16(r) => format!("    push {r}"),
            Self::RegA => "    push af".to_string(),
        }
    }

    fn pop_line(self) -> String {
        match self {
            Self::Reg16(r) => format!("    pop {r}"),
            Self::RegA => "    pop af".to_string(),
        }
    }
}

fn mem_access_width(op: &str) -> u16 {
    if matches!(
        op.to_ascii_lowercase().as_str(),
        "hl" | "de" | "bc" | "ix" | "iy" | "sp"
    ) {
        2
    } else {
        1
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

/// Promotes remaining balanced, LIFO-nested single-block scratch spills (`ld (nn), hl/de/a`
/// ... `ld hl/de/a, (nn)`) in `fn_sb_ranges` to `push`/`pop` (2 bytes vs 6-7 bytes).
pub(super) fn promote_single_block_spills_to_stack(
    lines: &[String],
    fn_sb_ranges: &HashMap<String, (u16, u16)>,
) -> (Vec<String>, bool) {
    let uchiguri_base = detect_uchiguri_base(lines);
    let kinds: Vec<LineKind> = lines.iter().map(|l| classify_line(l)).collect();
    let dummy_ranges: HashMap<String, Vec<(u16, u16)>> = fn_sb_ranges
        .iter()
        .map(|(k, &r)| (k.clone(), vec![r]))
        .collect();
    let fn_spans = partition_fn_spans(&kinds, &dummy_ranges);

    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut changed = false;

    for (start, end, _) in fn_spans {
        let sb_range = if let LineKind::Label(lbl) = &kinds[start] {
            fn_sb_ranges.get(lbl).copied()
        } else {
            None
        };
        let Some((sb_lo, sb_hi)) = sb_range else {
            out.extend_from_slice(&lines[start..end]);
            continue;
        };

        let mut replace: HashMap<usize, String> = HashMap::new();
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
                push: String,
                pop: String,
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
                if !addr_in_sb(addr, uchiguri_base, sz, sb_lo, sb_hi) {
                    continue;
                }

                let mut depth: isize = 0;
                let mut found: Option<(usize, StackSpillKind)> = None;
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
                            && mem_ranges_overlap(load_addr, mem_access_width(&ops2[0]), addr, sz)
                        {
                            if load_addr == addr
                                && depth == 0
                                && let Some(lk) = StackSpillKind::from_reg(&ops2[0])
                                && lk.width() == sz
                            {
                                found = Some((j, lk));
                            }
                            break;
                        }
                        if let Some(st_addr) = parse_mem_ref(&ops2[0])
                            && mem_ranges_overlap(st_addr, mem_access_width(&ops2[1]), addr, sz)
                        {
                            break;
                        }
                    }
                }

                let Some((load_idx, load_kind)) = found else {
                    continue;
                };

                // Verify `[addr, addr + sz)` is not read again after `load_idx` before overwrite or BB end.
                let mut dead_after = true;
                for k in (load_idx + 1)..end {
                    match &kinds[k] {
                        LineKind::EmptyOrComment => {}
                        LineKind::Label(lbl) => {
                            if !is_bb_label(lbl) {
                                dead_after = false;
                            }
                            break;
                        }
                        LineKind::Directive => break,
                        LineKind::Inst {
                            mnemonic: km,
                            ops: kops,
                        } => {
                            if matches!(km.as_str(), "ret" | "reti" | "halt") && kops.is_empty() {
                                break;
                            }
                            if matches!(km.as_str(), "jp" | "jr") && kops.len() == 1 {
                                if !is_bb_label(&kops[0]) {
                                    dead_after = false;
                                }
                                break;
                            }
                            if matches!(km.as_str(), "jp" | "jr" | "djnz" | "ret") {
                                dead_after = false;
                                break;
                            }
                            if km == "ld" && kops.len() == 2 {
                                if let Some(ra) = parse_mem_ref(&kops[1])
                                    && mem_ranges_overlap(ra, mem_access_width(&kops[0]), addr, sz)
                                {
                                    dead_after = false;
                                    break;
                                }
                                if let Some(wa) = parse_mem_ref(&kops[0])
                                    && wa == addr
                                    && mem_access_width(&kops[1]) >= sz
                                {
                                    break;
                                }
                            }
                        }
                    }
                }
                if !dead_after {
                    continue;
                }

                if matches!(load_kind, StackSpillKind::RegA)
                    && !pop_af_flags_safe(&kinds[..reg_end], load_idx + 1)
                {
                    continue;
                }

                cands.push(StackCand {
                    store_idx: i,
                    load_idx,
                    push: store_kind.push_line(),
                    pop: load_kind.pop_line(),
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
                out.push(repl);
            } else {
                out.push(lines[idx].clone());
            }
        }
    }

    (out, changed)
}
