//! Lightweight Z80 peephole optimizer operating on emitted assembly text.
//!
//! Transformations are strictly local within basic blocks (never crossing labels
//! or directives) and preserve Z80 flag semantics.

use std::collections::{HashMap, HashSet};

/// Runs peephole optimization passes on Aun-generated Z80 assembly until convergence.
pub fn optimize_asm(src: &str) -> String {
    optimize_asm_per_fn(src, &HashMap::new())
}

/// Runs peephole optimization plus per-function dead private-slot store elimination
/// and single-block `BC` spill promotion.
///
/// `fn_private_ranges` maps each top-level Z80 function label to its own exclusive
/// `(base, end)` RAM intervals (scratch slots and non-address-taken locals) so dead-store
/// elimination never confuses sibling functions' overlaid Uchiguri locals with another
/// function's private slots.
pub fn optimize_asm_per_fn(
    src: &str,
    fn_private_ranges: &HashMap<String, Vec<(u16, u16)>>,
) -> String {
    let mut lines: Vec<String> = src.lines().map(String::from).collect();
    for _ in 0..10 {
        let (next, changed1) = run_pass(&lines);
        let (next1b, changed1b) = eliminate_redundant_reg_reloads(&next);
        let (next2, changed2) = if fn_private_ranges.is_empty() {
            (next1b, false)
        } else {
            eliminate_dead_scratch_stores(&next1b, fn_private_ranges)
        };
        let (next3, changed3) = if fn_private_ranges.is_empty() {
            (next2, false)
        } else {
            promote_single_block_spills_to_bc(&next2, fn_private_ranges)
        };
        lines = next3;
        if !changed1 && !changed1b && !changed2 && !changed3 {
            break;
        }
    }
    let mut out = lines.join("\n");
    if src.ends_with('\n') {
        out.push('\n');
    }
    out
}

/// CFG-aware register-value cache across basic blocks within each function:
/// Tracks which fixed memory addresses `(0xXXXX)` currently match `HL`, `DE`, and `A`
/// (and which `(ix + d)` 16-bit pair currently matches `HL`), propagating states into
/// intra-function `_bb<digits>` labels via predecessor intersection.
#[derive(Clone, Default, PartialEq, Eq)]
struct RegCache {
    hl_addrs: HashSet<u16>,
    de_addrs: HashSet<u16>,
    a_addrs: HashSet<u16>,
    hl_ix_disp: Option<i8>,
}

impl RegCache {
    fn clear(&mut self) {
        self.hl_addrs.clear();
        self.de_addrs.clear();
        self.a_addrs.clear();
        self.hl_ix_disp = None;
    }

    fn intersect_with(&mut self, other: &RegCache) {
        self.hl_addrs.retain(|x| other.hl_addrs.contains(x));
        self.de_addrs.retain(|x| other.de_addrs.contains(x));
        self.a_addrs.retain(|x| other.a_addrs.contains(x));
        if self.hl_ix_disp != other.hl_ix_disp {
            self.hl_ix_disp = None;
        }
    }

    fn invalidate_addr_bytes(&mut self, addr: u16, width: u16) {
        let a_start = addr as u32;
        let a_end = a_start + width as u32;
        let bad = |set: &mut HashSet<u16>, reg_w: u16| {
            set.retain(|&x| {
                let x_start = x as u32;
                let x_end = x_start + reg_w as u32;
                x_end <= a_start || a_end <= x_start
            });
        };
        bad(&mut self.hl_addrs, 2);
        bad(&mut self.de_addrs, 2);
        bad(&mut self.a_addrs, 1);
    }
}

fn is_bb_label(lbl: &str) -> bool {
    lbl.rsplit_once("_bb")
        .is_some_and(|(_, d)| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
}

fn parse_ix_disp(op: &str) -> Option<i8> {
    let inner = op.trim().strip_prefix('(')?.strip_suffix(')')?.trim();
    let rest = inner
        .strip_prefix("ix")
        .or_else(|| inner.strip_prefix("IX"))?
        .trim();
    if rest.is_empty() {
        return Some(0);
    }
    if let Some(p) = rest.strip_prefix('+') {
        p.trim().parse::<i8>().ok()
    } else if let Some(m) = rest.strip_prefix('-') {
        m.trim().parse::<i8>().ok().and_then(|v| v.checked_neg())
    } else {
        None
    }
}

fn eliminate_redundant_reg_reloads(lines: &[String]) -> (Vec<String>, bool) {
    let kinds: Vec<LineKind> = lines.iter().map(|l| classify_line(l)).collect();

    // First, count incoming edges for every `_bb<digits>` label so we know whether we have
    // seen all predecessors (straight-line / forward-DAG blocks only; loop headers reset).
    let mut label_pos: HashMap<String, usize> = HashMap::new();
    for (idx, k) in kinds.iter().enumerate() {
        if let LineKind::Label(lbl) = k {
            label_pos.insert(lbl.clone(), idx);
        }
    }
    let mut has_back_edge: HashSet<String> = HashSet::new();
    for (idx, k) in kinds.iter().enumerate() {
        if let LineKind::Inst { mnemonic, ops } = k
            && matches!(mnemonic.as_str(), "jp" | "jr" | "djnz")
            && let Some(target) = ops.last()
            && let Some(&tpos) = label_pos.get(target)
            && tpos <= idx
        {
            has_back_edge.insert(target.clone());
        }
    }

    let mut incoming: HashMap<String, Option<RegCache>> = HashMap::new();
    let record_edge = |target: &str, state: &RegCache, map: &mut HashMap<String, Option<RegCache>>| {
        if !is_bb_label(target) {
            return;
        }
        match map.get_mut(target) {
            None => {
                map.insert(target.to_string(), Some(state.clone()));
            }
            Some(Some(existing)) => {
                existing.intersect_with(state);
            }
            Some(None) => {}
        }
    };

    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut changed = false;
    let mut cache = RegCache::default();
    let mut reachable = true;

    let mut i = 0usize;
    while i < lines.len() {
        match &kinds[i] {
            LineKind::EmptyOrComment => {
                out.push(lines[i].clone());
                i += 1;
            }
            LineKind::Directive => {
                cache.clear();
                reachable = true;
                out.push(lines[i].clone());
                i += 1;
            }
            LineKind::Label(lbl) => {
                if is_bb_label(lbl) && !has_back_edge.contains(lbl) {
                    if reachable {
                        record_edge(lbl, &cache, &mut incoming);
                    }
                    cache = incoming.remove(lbl).flatten().unwrap_or_default();
                } else {
                    cache.clear();
                }
                reachable = true;
                out.push(lines[i].clone());
                i += 1;
            }
            LineKind::Inst { mnemonic, ops } => {
                // Recognize 2-instruction `ld l, (ix + d); ld h, (ix + d+1)`
                if mnemonic == "ld"
                    && ops.len() == 2
                    && ops[0].eq_ignore_ascii_case("l")
                    && let Some(d0) = parse_ix_disp(&ops[1])
                    && let Some(ni) = next_non_comment(&kinds, i + 1)
                    && let LineKind::Inst {
                        mnemonic: m2,
                        ops: ops2,
                    } = &kinds[ni]
                    && m2 == "ld"
                    && ops2.len() == 2
                    && ops2[0].eq_ignore_ascii_case("h")
                    && parse_ix_disp(&ops2[1]) == Some(d0.wrapping_add(1))
                {
                    if cache.hl_ix_disp == Some(d0) {
                        // Already in HL! Drop both instructions.
                        changed = true;
                        i = ni + 1;
                        continue;
                    }
                    cache.hl_addrs.clear();
                    cache.hl_ix_disp = Some(d0);
                    out.push(lines[i].clone());
                    for line in &lines[i + 1..=ni] {
                        out.push(line.clone());
                    }
                    i = ni + 1;
                    continue;
                }

                if mnemonic == "ld" && ops.len() == 2 {
                    let dst_l = ops[0].to_ascii_lowercase();
                    let src_l = ops[1].to_ascii_lowercase();

                    // Check if `ld reg, (0xXXXX)` is already satisfied by the register cache!
                    if let Some(addr) = parse_mem_addr(&ops[1]) {
                        let already_loaded = match dst_l.as_str() {
                            "hl" => cache.hl_addrs.contains(&addr),
                            "de" => cache.de_addrs.contains(&addr),
                            "a" => cache.a_addrs.contains(&addr),
                            _ => false,
                        };
                        if already_loaded {
                            changed = true;
                            i += 1;
                            continue;
                        }
                        // Cross-register copy from already-cached 16-bit pair (2B/8T vs 3B/16T)
                        if dst_l == "hl" && cache.de_addrs.contains(&addr) {
                            out.push("    ld h, d".to_string());
                            out.push("    ld l, e".to_string());
                            cache.hl_addrs = cache.de_addrs.clone();
                            cache.hl_ix_disp = None;
                            changed = true;
                            i += 1;
                            continue;
                        }
                        if dst_l == "de" && cache.hl_addrs.contains(&addr) {
                            out.push("    ld d, h".to_string());
                            out.push("    ld e, l".to_string());
                            cache.de_addrs = cache.hl_addrs.clone();
                            changed = true;
                            i += 1;
                            continue;
                        }
                        // Byte-slice reload from already-cached 16-bit register (e.g., 16->8 cast!)
                        if dst_l == "a" {
                            if cache.hl_addrs.contains(&addr) {
                                out.push("    ld a, l".to_string());
                                cache.a_addrs.clear();
                                cache.a_addrs.insert(addr);
                                changed = true;
                                i += 1;
                                continue;
                            }
                            if addr > 0 && cache.hl_addrs.contains(&(addr - 1)) {
                                out.push("    ld a, h".to_string());
                                cache.a_addrs.clear();
                                cache.a_addrs.insert(addr);
                                changed = true;
                                i += 1;
                                continue;
                            }
                            if cache.de_addrs.contains(&addr) {
                                out.push("    ld a, e".to_string());
                                cache.a_addrs.clear();
                                cache.a_addrs.insert(addr);
                                changed = true;
                                i += 1;
                                continue;
                            }
                            if addr > 0 && cache.de_addrs.contains(&(addr - 1)) {
                                out.push("    ld a, d".to_string());
                                cache.a_addrs.clear();
                                cache.a_addrs.insert(addr);
                                changed = true;
                                i += 1;
                                continue;
                            }
                        }
                        // Update cache for the loaded register
                        match dst_l.as_str() {
                            "hl" => {
                                cache.hl_addrs.clear();
                                cache.hl_addrs.insert(addr);
                                cache.hl_ix_disp = None;
                            }
                            "de" => {
                                cache.de_addrs.clear();
                                cache.de_addrs.insert(addr);
                            }
                            "a" => {
                                cache.a_addrs.clear();
                                cache.a_addrs.insert(addr);
                            }
                            "h" | "l" => {
                                cache.hl_addrs.clear();
                                cache.hl_ix_disp = None;
                            }
                            "d" | "e" => cache.de_addrs.clear(),
                            "ix" => cache.hl_ix_disp = None,
                            _ => {}
                        }
                        out.push(lines[i].clone());
                        i += 1;
                        continue;
                    }

                    // Store `ld (0xXXXX), reg`: invalidate overlapping cached addresses, then record `addr`
                    if let Some(addr) = parse_mem_addr(&ops[0]) {
                        let w = if matches!(src_l.as_str(), "hl" | "de" | "bc" | "sp" | "ix" | "iy")
                        {
                            2
                        } else {
                            1
                        };
                        cache.invalidate_addr_bytes(addr, w);
                        cache.hl_ix_disp = None;
                        match src_l.as_str() {
                            "hl" => {
                                cache.hl_addrs.insert(addr);
                            }
                            "de" => {
                                cache.de_addrs.insert(addr);
                            }
                            "a" => {
                                cache.a_addrs.insert(addr);
                            }
                            _ => {}
                        }
                        out.push(lines[i].clone());
                        i += 1;
                        continue;
                    }

                    // Indirect memory write `ld (ix + d), ...`: only invalidates hl_ix_disp if overlapping,
                    // and conservatively clears fixed-address caches (pointer may alias globals/locals).
                    if dst_l.contains('(') {
                        cache.hl_addrs.clear();
                        cache.de_addrs.clear();
                        cache.a_addrs.clear();
                        if let Some(wd) = parse_ix_disp(&ops[0]) {
                            if let Some(hd) = cache.hl_ix_disp
                                && (wd == hd || wd == hd.wrapping_add(1))
                            {
                                cache.hl_ix_disp = None;
                            }
                        } else {
                            cache.hl_ix_disp = None;
                        }
                        out.push(lines[i].clone());
                        i += 1;
                        continue;
                    }

                    // Register-to-register or immediate load invalidates only the modified register
                    match dst_l.as_str() {
                        "hl" | "h" | "l" => {
                            cache.hl_addrs.clear();
                            cache.hl_ix_disp = None;
                        }
                        "de" | "d" | "e" => cache.de_addrs.clear(),
                        "a" => cache.a_addrs.clear(),
                        "ix" => cache.hl_ix_disp = None,
                        _ => {}
                    }
                    out.push(lines[i].clone());
                    i += 1;
                    continue;
                }

                // `ex de, hl` swaps `hl_addrs` and `de_addrs`!
                if mnemonic == "ex"
                    && ops.len() == 2
                    && ops[0].eq_ignore_ascii_case("de")
                    && ops[1].eq_ignore_ascii_case("hl")
                {
                    std::mem::swap(&mut cache.hl_addrs, &mut cache.de_addrs);
                    cache.hl_ix_disp = None;
                    out.push(lines[i].clone());
                    i += 1;
                    continue;
                }

                // Pure flag-only instructions preserve caches
                if matches!(mnemonic.as_str(), "bit" | "cp") {
                    out.push(lines[i].clone());
                    i += 1;
                    continue;
                }
                if matches!(
                    mnemonic.as_str(),
                    "add" | "adc" | "sub" | "sbc" | "and" | "or" | "xor" | "cpl" | "neg"
                        | "rlca" | "rrca" | "rla" | "rra"
                ) {
                    if ops.first().is_some_and(|o| o.eq_ignore_ascii_case("hl")) {
                        cache.hl_addrs.clear();
                        cache.hl_ix_disp = None;
                    } else if ops.first().is_some_and(|o| o.eq_ignore_ascii_case("ix")) {
                        cache.hl_ix_disp = None;
                    } else {
                        cache.a_addrs.clear();
                    }
                    out.push(lines[i].clone());
                    i += 1;
                    continue;
                }

                // Branches: propagate current cache to target `_bb<digits>` label!
                if matches!(mnemonic.as_str(), "jp" | "jr") {
                    if let Some(target) = ops.last() {
                        record_edge(target, &cache, &mut incoming);
                    }
                    if ops.len() == 1 {
                        // Unconditional jump: next line is unreachable until next label
                        cache.clear();
                        reachable = false;
                    }
                    out.push(lines[i].clone());
                    i += 1;
                    continue;
                }
                if matches!(mnemonic.as_str(), "ret" | "reti" | "halt") && ops.is_empty() {
                    cache.clear();
                    reachable = false;
                    out.push(lines[i].clone());
                    i += 1;
                    continue;
                }

                // Anything else (calls, shifts, inc/dec, ldir, push/pop): conservatively clear
                cache.clear();
                out.push(lines[i].clone());
                i += 1;
            }
        }
    }
    (out, changed)
}

fn inst_touches_bc(kind: &LineKind) -> bool {
    match kind {
        LineKind::Inst { mnemonic, ops } => {
            if matches!(
                mnemonic.as_str(),
                "call"
                    | "jp"
                    | "jr"
                    | "ret"
                    | "reti"
                    | "retn"
                    | "rst"
                    | "djnz"
                    | "ldi"
                    | "ldd"
                    | "ldir"
                    | "lddr"
                    | "cpi"
                    | "cpd"
                    | "cpir"
                    | "cpdr"
                    | "ini"
                    | "ind"
                    | "inir"
                    | "indr"
                    | "outi"
                    | "outd"
                    | "otir"
                    | "otdr"
                    | "exx"
                    | "halt"
            ) {
                return true;
            }
            ops.iter().any(|op| {
                let t = op.trim().to_ascii_lowercase();
                matches!(t.as_str(), "b" | "c" | "bc" | "(bc)" | "(c)")
            })
        }
        LineKind::Label(_) | LineKind::Directive => true,
        LineKind::EmptyOrComment => false,
    }
}

/// Promotes a single-write / single-read 16-bit private RAM spill (`ld (S), hl` ... `ld hl, (S)`)
/// within the same basic block to `BC` (`ld b, h; ld c, l` ... `ld h, b; ld l, c`) whenever
/// `BC` is untouched in between and `S` is referenced nowhere else in the function.
fn promote_single_block_spills_to_bc(
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
                && ref_count.get(&addr).copied() == Some(2)
                && ref_count.get(&addr.wrapping_add(1)).copied().unwrap_or(0) == 0
                && (addr == 0 || ref_count.get(&(addr - 1)).copied().unwrap_or(0) == 0)
            {
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
                if let Some(j) = found_read {
                    replace.insert(i, vec!["    ld b, h".to_string(), "    ld c, l".to_string()]);
                    replace.insert(j, vec!["    ld h, b".to_string(), "    ld l, c".to_string()]);
                    changed = true;
                    i = j + 1;
                    continue;
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

fn parse_mem_addr(op: &str) -> Option<u16> {
    let inner = op.trim().strip_prefix('(')?.strip_suffix(')')?.trim();
    if let Some(hex) = inner.strip_prefix("0x").or_else(|| inner.strip_prefix("0X")) {
        u16::from_str_radix(hex, 16).ok()
    } else {
        None
    }
}

fn in_ranges(addr: u16, ranges: &[(u16, u16)]) -> bool {
    ranges.iter().any(|&(lo, hi)| addr >= lo && addr < hi)
}

fn partition_fn_spans<'a>(
    kinds: &[LineKind],
    fn_private_ranges: &'a HashMap<String, Vec<(u16, u16)>>,
) -> Vec<(usize, usize, Option<&'a [(u16, u16)]>)> {
    let mut fn_spans: Vec<(usize, usize, Option<&'a [(u16, u16)]>)> = Vec::new();
    let mut cur_start = 0usize;
    let mut cur_range: Option<&'a [(u16, u16)]> = None;
    for (idx, kind) in kinds.iter().enumerate() {
        if let LineKind::Label(lbl) = kind
            && let Some(ranges) = fn_private_ranges.get(lbl)
        {
            if idx > cur_start {
                fn_spans.push((cur_start, idx, cur_range));
            }
            cur_start = idx;
            cur_range = Some(ranges.as_slice());
        } else if let LineKind::Label(lbl) = kind
            && !lbl.contains("_bb")
            && !lbl.contains("_sat_")
            && !lbl.contains("_cmp_")
            && !lbl.contains("_palloc_")
            && !lbl.contains("_pfree_")
            && !lbl.contains("_pget_")
            && !lbl.contains("_tr_")
            && !lbl.contains("_idx_")
            && !lbl.contains("_assert_")
            && !lbl.contains("_cast_")
            && !lbl.contains("_fxint_")
            && !lbl.contains("_sh")
            && cur_range.is_some()
        {
            if idx > cur_start {
                fn_spans.push((cur_start, idx, cur_range));
            }
            cur_start = idx;
            cur_range = None;
        }
    }
    if cur_start < kinds.len() {
        fn_spans.push((cur_start, kinds.len(), cur_range));
    }
    fn_spans
}

/// Deletes stores `ld (0xXXXX), reg` to the *current function's* own private addresses
/// (scratch slots and non-address-taken locals) that are never read anywhere within that function,
/// along with an immediately preceding pure `ld reg, imm` when the very next instruction
/// overwrites `reg` without reading it.
fn eliminate_dead_scratch_stores(
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

#[derive(Debug, Clone, PartialEq, Eq)]
enum LineKind {
    EmptyOrComment,
    Label(String),
    Directive,
    Inst { mnemonic: String, ops: Vec<String> },
}

fn classify_line(raw: &str) -> LineKind {
    let code = raw.split(';').next().unwrap_or("").trim();
    if code.is_empty() {
        return LineKind::EmptyOrComment;
    }
    if code.starts_with('.') {
        return LineKind::Directive;
    }
    if let Some(lbl) = code.strip_suffix(':') {
        let lbl = lbl.trim();
        if !lbl.is_empty() && !lbl.contains(' ') {
            return LineKind::Label(lbl.to_string());
        }
    }
    // If a line has `label: inst`, treat as directive/opaque boundary
    if code.contains(':') {
        return LineKind::Directive;
    }
    let mut parts = code.splitn(2, |c: char| c.is_whitespace());
    let mnemonic = parts.next().unwrap().to_ascii_lowercase();
    let tail = parts.next().unwrap_or("").trim();
    let ops = if tail.is_empty() {
        Vec::new()
    } else {
        tail.split(',').map(|s| s.trim().to_string()).collect()
    };
    LineKind::Inst { mnemonic, ops }
}

fn is_mem_operand(op: &str) -> bool {
    let t = op.trim();
    t.starts_with('(') && t.ends_with(')') && !t.eq_ignore_ascii_case("(hl)")
        && !t.eq_ignore_ascii_case("(bc)")
        && !t.eq_ignore_ascii_case("(de)")
        && !t.eq_ignore_ascii_case("(sp)")
        && !t.eq_ignore_ascii_case("(c)")
}

fn is_zero_imm(op: &str) -> bool {
    let t = op.trim().trim_start_matches('#');
    matches!(t, "0" | "0x0" | "0x00" | "0X0" | "0X00" | "$0" | "$00")
}

fn reads_flags(kind: &LineKind) -> bool {
    match kind {
        LineKind::Inst { mnemonic, ops } => match mnemonic.as_str() {
            "jr" | "jp" | "call" => ops.len() == 2,
            "ret" => ops.len() == 1,
            "adc" | "sbc" | "rla" | "rra" | "rl" | "rr" | "ccf" | "daa" => true,
            "push" => ops.first().is_some_and(|o| o.eq_ignore_ascii_case("af")),
            _ => false,
        },
        _ => false,
    }
}

/// Returns `true` if `kind` unconditionally overwrites all arithmetic flags (`S`, `Z`, `P/V`, `C`)
/// without reading any prior flag.
fn clobbers_all_flags(kind: &LineKind) -> bool {
    match kind {
        LineKind::Inst { mnemonic, ops } => match mnemonic.as_str() {
            "and" | "or" | "xor" | "cp" | "sub" | "neg" => true,
            "add" => ops.first().is_some_and(|o| o.eq_ignore_ascii_case("a")),
            "srl" | "sra" | "sla" => true,
            _ => false,
        },
        _ => false,
    }
}

/// Returns `true` if modifying flags at `start` cannot affect any subsequent flag consumer
/// within the current basic block.
fn flags_dead_after(kinds: &[LineKind], start: usize) -> bool {
    let mut k = start;
    while k < kinds.len() {
        match &kinds[k] {
            LineKind::EmptyOrComment => k += 1,
            LineKind::Label(_) | LineKind::Directive => return false,
            kind @ LineKind::Inst { mnemonic, .. } => {
                if reads_flags(kind) {
                    return false;
                }
                if clobbers_all_flags(kind) {
                    return true;
                }
                // Flag-preserving instructions: keep scanning forward
                if matches!(mnemonic.as_str(), "ld" | "push" | "pop" | "ex" | "nop" | "di" | "ei") {
                    k += 1;
                    continue;
                }
                // Unconditional control transfer or unknown flag behavior: conservative
                return false;
            }
        }
    }
    false
}

fn next_non_comment(kinds: &[LineKind], start: usize) -> Option<usize> {
    (start..kinds.len()).find(|&idx| !matches!(kinds[idx], LineKind::EmptyOrComment))
}

fn run_pass(lines: &[String]) -> (Vec<String>, bool) {
    let kinds: Vec<LineKind> = lines.iter().map(|l| classify_line(l)).collect();
    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut changed = false;
    let mut i = 0usize;

    while i < lines.len() {
        match &kinds[i] {
            LineKind::Inst { mnemonic, ops } => {
                // Rule 1: eliminate redundant `ld r, r`
                if mnemonic == "ld" && ops.len() == 2 {
                    let d = ops[0].to_ascii_lowercase();
                    let s = ops[1].to_ascii_lowercase();
                    if d == s && matches!(d.as_str(), "a" | "b" | "c" | "d" | "e" | "h" | "l") {
                        changed = true;
                        i += 1;
                        continue;
                    }
                }

                // Rule 2: `cp 0` -> `or a`
                if mnemonic == "cp"
                    && ((ops.len() == 1 && is_zero_imm(&ops[0]))
                        || (ops.len() == 2
                            && ops[0].eq_ignore_ascii_case("a")
                            && is_zero_imm(&ops[1])))
                {
                    out.push("    or a".to_string());
                    changed = true;
                    i += 1;
                    continue;
                }

                // Rule 3: `ld a, 0` -> `xor a` when flags are provably dead afterward
                if mnemonic == "ld"
                    && ops.len() == 2
                    && ops[0].eq_ignore_ascii_case("a")
                    && is_zero_imm(&ops[1])
                    && flags_dead_after(&kinds, i + 1)
                {
                    out.push("    xor a".to_string());
                    changed = true;
                    i += 1;
                    continue;
                }

                // Look ahead to the next non-comment line for 2-instruction patterns
                if let Some(ni) = next_non_comment(&kinds, i + 1) {
                    // Rule 4: `ld (addr), reg` followed by `ld reg, (addr)` (reg in {a, hl, de})
                    if mnemonic == "ld"
                        && ops.len() == 2
                        && is_mem_operand(&ops[0])
                        && matches!(
                            ops[1].to_ascii_lowercase().as_str(),
                            "a" | "hl" | "de"
                        )
                        && let LineKind::Inst {
                            mnemonic: m2,
                            ops: ops2,
                        } = &kinds[ni]
                        && m2 == "ld"
                        && ops2.len() == 2
                        && ops2[0].eq_ignore_ascii_case(&ops[1])
                        && ops2[1].eq_ignore_ascii_case(&ops[0])
                    {
                        // Keep the store and any comments between i and ni, drop the redundant reload
                        out.push(lines[i].clone());
                        for line in &lines[i + 1..ni] {
                            out.push(line.clone());
                        }
                        changed = true;
                        i = ni + 1;
                        continue;
                    }

                    // Rule 4b: `ld (S), hl; ld hl, <X>; ld de, (S)` -> `ld (S), hl; ex de, hl; ld hl, <X>`
                    // (when <X> is not `(S)`, letting dead-scratch DCE subsequently drop `ld (S), hl`!)
                    if mnemonic == "ld"
                        && ops.len() == 2
                        && is_mem_operand(&ops[0])
                        && ops[1].eq_ignore_ascii_case("hl")
                        && let LineKind::Inst {
                            mnemonic: m2,
                            ops: ops2,
                        } = &kinds[ni]
                        && m2 == "ld"
                        && ops2.len() == 2
                        && ops2[0].eq_ignore_ascii_case("hl")
                        && !ops2[1].eq_ignore_ascii_case(&ops[0])
                        && let Some(ni2) = next_non_comment(&kinds, ni + 1)
                        && let LineKind::Inst {
                            mnemonic: m3,
                            ops: ops3,
                        } = &kinds[ni2]
                        && m3 == "ld"
                        && ops3.len() == 2
                        && ops3[0].eq_ignore_ascii_case("de")
                        && ops3[1].eq_ignore_ascii_case(&ops[0])
                    {
                        out.push(lines[i].clone());
                        out.push("    ex de, hl".to_string());
                        out.push(lines[ni].clone());
                        changed = true;
                        i = ni2 + 1;
                        continue;
                    }

                    // Rule 5: tail-call `call sym; ret` -> `jp sym`
                    if mnemonic == "call"
                        && ops.len() == 1
                        && let LineKind::Inst {
                            mnemonic: m2,
                            ops: ops2,
                        } = &kinds[ni]
                        && m2 == "ret"
                        && ops2.is_empty()
                    {
                        out.push(format!("    jp {}", ops[0]));
                        for line in &lines[i + 1..ni] {
                            out.push(line.clone());
                        }
                        changed = true;
                        i = ni + 1;
                        continue;
                    }

                    // Rule 7: Diamond tail-merging — if `<inst>; jp L_next` is immediately followed by
                    // `L_else: <inst>; L_next:` (identical instruction!), drop `<inst>; jp L_next` so
                    // control falls through into `L_else`!
                    if let LineKind::Inst {
                        mnemonic: jm,
                        ops: jops,
                    } = &kinds[ni]
                        && (jm == "jp" || jm == "jr")
                        && jops.len() == 1
                        && is_bb_label(&jops[0])
                        && let Some(lbl_idx) = next_non_comment(&kinds, ni + 1)
                        && let LineKind::Label(else_lbl) = &kinds[lbl_idx]
                        && is_bb_label(else_lbl)
                        && let Some(else_inst_idx) = next_non_comment(&kinds, lbl_idx + 1)
                        && kinds[else_inst_idx] == kinds[i]
                        && let Some(after_else) = next_non_comment(&kinds, else_inst_idx + 1)
                        && let LineKind::Label(next_lbl) = &kinds[after_else]
                        && next_lbl == &jops[0]
                    {
                        changed = true;
                        i = lbl_idx;
                        continue;
                    }

                    // Rule 6: `jp lbl` / `jr lbl` falling through to an intra-function `_bb<digits>` label
                    // (never drops jumps targeting `__aun_noop_ret` or functions, preserving #[smc] stubs!)
                    if (mnemonic == "jp" || mnemonic == "jr")
                        && ops.len() == 1
                        && ops[0]
                            .rsplit_once("_bb")
                            .is_some_and(|(_, d)| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
                    {
                        let target = &ops[0];
                        let mut k = ni;
                        let mut falls_through = false;
                        while k < kinds.len() {
                            match &kinds[k] {
                                LineKind::EmptyOrComment => k += 1,
                                LineKind::Label(lbl) => {
                                    if lbl == target {
                                        falls_through = true;
                                        break;
                                    }
                                    k += 1;
                                }
                                _ => break,
                            }
                        }
                        if falls_through {
                            for line in &lines[i + 1..ni] {
                                out.push(line.clone());
                            }
                            changed = true;
                            i = ni;
                            continue;
                        }
                    }
                }

                out.push(lines[i].clone());
                i += 1;
            }
            _ => {
                out.push(lines[i].clone());
                i += 1;
            }
        }
    }

    (out, changed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eliminates_redundant_store_reload_and_tail_call() {
        let input = "\
foo:
    ld a, 0
    ld (0xC000), a
    ld a, (0xC000)
    ld hl, 0x1234
    ld (0xC002), hl
    ld hl, (0xC002)
    cp 0
    ld b, b
    jp foo_bb1
foo_bb1:
    call bar
    ret
";
        let got = optimize_asm(input);
        assert!(got.contains("xor a"));
        assert!(!got.contains("ld a, 0"));
        assert_eq!(got.matches("0xC000").count(), 1);
        assert_eq!(got.matches("0xC002").count(), 1);
        assert!(got.contains("or a"));
        assert!(!got.contains("cp 0"));
        assert!(!got.contains("ld b, b"));
        assert!(!got.contains("jp foo_bb1"));
        assert!(got.contains("jp bar"));
        assert!(!got.contains("call bar"));
    }

    #[test]
    fn preserves_flags_before_conditional_branch_and_across_labels() {
        let input = "\
    cp b
    ld a, 0
    jr z, done
    ld (0xC000), a
done:
    ld a, (0xC000)
    ret
";
        let got = optimize_asm(input);
        assert!(got.contains("ld a, 0"), "must not clobber flags before jr z");
        assert_eq!(
            got.matches("0xC000").count(),
            2,
            "must not eliminate reload across label"
        );
    }

    #[test]
    fn preserves_return_value_before_ret_and_sibling_locals() {
        let input = "\
answer:
    ld a, 0x2A
    ld (0xC000), a
    ld a, (0xC000)
    ret
sibling:
    ld a, 0x77
    ld (0xC000), a
    ld hl, 0xC000
    ld a, (hl)
    ret
";
        let mut ranges = HashMap::new();
        // `answer`'s scratch is 0xC000..0xC002, while `sibling`'s local is at 0xC000 (scratch at 0xC004..0xC006)
        ranges.insert("answer".to_string(), vec![(0xC000, 0xC002)]);
        ranges.insert("sibling".to_string(), vec![(0xC004, 0xC006)]);

        let got = optimize_asm_per_fn(input, &ranges);
        // `answer` MUST keep `ld a, 0x2A` before `ret`!
        assert!(
            got.contains("ld a, 0x2A"),
            "must preserve return value load before ret:\n{got}"
        );
        // `sibling` MUST keep its live local store `ld (0xC000), a`!
        assert!(
            got.contains("ld (0xC000), a"),
            "must not delete sibling's local store:\n{got}"
        );
    }
}
