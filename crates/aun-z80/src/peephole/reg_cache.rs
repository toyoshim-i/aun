use super::*;

/// CFG-aware register-value cache across basic blocks within each function:
/// Tracks which fixed/relocatable memory addresses currently match `HL`, `DE`, and `A`
/// (and which `(ix + d)` 16-bit pair currently matches `HL`), propagating states into
/// intra-function `_bb<digits>` labels via predecessor intersection.
#[derive(Clone, Default, PartialEq, Eq)]
struct RegCache {
    hl_addrs: HashSet<MemAddr>,
    de_addrs: HashSet<MemAddr>,
    a_addrs: HashSet<MemAddr>,
    hl_ix_disp: Option<i8>,
    a_eq_reg: Option<char>,
}

impl RegCache {
    fn clear(&mut self) {
        self.hl_addrs.clear();
        self.de_addrs.clear();
        self.a_addrs.clear();
        self.hl_ix_disp = None;
        self.a_eq_reg = None;
    }

    fn intersect_with(&mut self, other: &RegCache) {
        self.hl_addrs.retain(|x| other.hl_addrs.contains(x));
        self.de_addrs.retain(|x| other.de_addrs.contains(x));
        self.a_addrs.retain(|x| other.a_addrs.contains(x));
        if self.hl_ix_disp != other.hl_ix_disp {
            self.hl_ix_disp = None;
        }
        if self.a_eq_reg != other.a_eq_reg {
            self.a_eq_reg = None;
        }
    }

    fn invalidate_addr_bytes(&mut self, addr: MemAddr, width: u16) {
        let (is_u, a_start) = match addr {
            MemAddr::Uchiguri(o) => (true, o as u32),
            MemAddr::Fixed(a) => (false, a as u32),
        };
        let a_end = a_start + width as u32;
        let bad = |set: &mut HashSet<MemAddr>, reg_w: u16| {
            set.retain(|&x| {
                let (x_u, x_start) = match x {
                    MemAddr::Uchiguri(o) => (true, o as u32),
                    MemAddr::Fixed(a) => (false, a as u32),
                };
                if x_u != is_u {
                    return true;
                }
                let x_end = x_start + reg_w as u32;
                x_end <= a_start || a_end <= x_start
            });
        };
        bad(&mut self.hl_addrs, 2);
        bad(&mut self.de_addrs, 2);
        bad(&mut self.a_addrs, 1);
    }
}

pub(super) fn eliminate_redundant_reg_reloads(
    lines: &[String],
    fn_private_ranges: &HashMap<String, Vec<(u16, u16)>>,
) -> (Vec<String>, bool) {
    let uchiguri_base = detect_uchiguri_base(lines);
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
    let mut cur_private: Option<&Vec<(u16, u16)>> = None;
    let mut cur_fn_prefix: Option<String> = None;
    let mut reachable = true;
    let mut in_volatile = false;

    let mut i = 0usize;
    while i < lines.len() {
        match &kinds[i] {
            LineKind::EmptyOrComment => {
                let t = lines[i].trim();
                if t == "; @aun:volatile_begin" {
                    cache.clear();
                    in_volatile = true;
                } else if t == "; @aun:volatile_end" {
                    cache.clear();
                    in_volatile = false;
                }
                out.push(lines[i].clone());
                i += 1;
            }
            LineKind::Directive => {
                cache.clear();
                cur_private = None;
                cur_fn_prefix = None;
                reachable = true;
                out.push(lines[i].clone());
                i += 1;
            }
            LineKind::Label(lbl) => {
                if let Some(ranges) = fn_private_ranges.get(lbl) {
                    cur_private = Some(ranges);
                    cur_fn_prefix = Some(format!("{lbl}_"));
                } else if !cur_fn_prefix
                    .as_deref()
                    .is_some_and(|p| lbl.starts_with(p))
                {
                    cur_private = None;
                    cur_fn_prefix = None;
                }
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
                if in_volatile {
                    cache.clear();
                    out.push(lines[i].clone());
                    i += 1;
                    continue;
                }
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

                    // Check if `ld reg, (mem)` is already satisfied by the register cache!
                    if let Some(addr) = parse_mem_ref(&ops[1]) {
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
                                cache.a_eq_reg = None;
                                changed = true;
                                i += 1;
                                continue;
                            }
                            if let Some(prev) = addr.sub_one()
                                && cache.hl_addrs.contains(&prev)
                            {
                                out.push("    ld a, h".to_string());
                                cache.a_addrs.clear();
                                cache.a_addrs.insert(addr);
                                cache.a_eq_reg = None;
                                changed = true;
                                i += 1;
                                continue;
                            }
                            if cache.de_addrs.contains(&addr) {
                                out.push("    ld a, e".to_string());
                                cache.a_addrs.clear();
                                cache.a_addrs.insert(addr);
                                cache.a_eq_reg = None;
                                changed = true;
                                i += 1;
                                continue;
                            }
                            if let Some(prev) = addr.sub_one()
                                && cache.de_addrs.contains(&prev)
                            {
                                out.push("    ld a, d".to_string());
                                cache.a_addrs.clear();
                                cache.a_addrs.insert(addr);
                                cache.a_eq_reg = None;
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
                                cache.a_eq_reg = None;
                            }
                            "h" | "l" => {
                                cache.hl_addrs.clear();
                                cache.hl_ix_disp = None;
                            }
                            "d" | "e" => cache.de_addrs.clear(),
                            "b" => {
                                if cache.a_eq_reg == Some('b') {
                                    cache.a_eq_reg = None;
                                }
                            }
                            "c" => {
                                if cache.a_eq_reg == Some('c') {
                                    cache.a_eq_reg = None;
                                }
                            }
                            "bc" => cache.a_eq_reg = None,
                            "ix" => cache.hl_ix_disp = None,
                            _ => {}
                        }
                        out.push(lines[i].clone());
                        i += 1;
                        continue;
                    }

                    // Store `ld (mem), reg`: invalidate overlapping cached addresses, then record `addr`
                    if let Some(addr) = parse_mem_ref(&ops[0]) {
                        let w = if matches!(src_l.as_str(), "hl" | "de" | "bc" | "sp" | "ix" | "iy")
                        {
                            2
                        } else {
                            1
                        };
                        cache.invalidate_addr_bytes(addr, w);
                        // Storing to a non-escaping local/scratch in this function cannot alias `(ix + d)`.
                        if !cur_private.is_some_and(|r| in_private_ranges(addr, uchiguri_base, r)) {
                            cache.hl_ix_disp = None;
                        }
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
                    // and retains non-escaping private locals/scratch that pointers cannot alias.
                    if dst_l.contains('(') {
                        if let Some(priv_ranges) = cur_private {
                            cache
                                .hl_addrs
                                .retain(|&a| in_private_ranges(a, uchiguri_base, priv_ranges));
                            cache
                                .de_addrs
                                .retain(|&a| in_private_ranges(a, uchiguri_base, priv_ranges));
                            cache
                                .a_addrs
                                .retain(|&a| in_private_ranges(a, uchiguri_base, priv_ranges));
                        } else {
                            cache.hl_addrs.clear();
                            cache.de_addrs.clear();
                            cache.a_addrs.clear();
                        }
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

                    // Eliminate redundant `ld a, b/c` or `ld b/c, a` when A already equals that reg!
                    let is_noop_a_reg = matches!(
                        (dst_l.as_str(), src_l.as_str(), cache.a_eq_reg),
                        ("a", "c", Some('c'))
                            | ("c", "a", Some('c'))
                            | ("a", "b", Some('b'))
                            | ("b", "a", Some('b'))
                    );
                    if is_noop_a_reg {
                        changed = true;
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
                        "a" => {
                            cache.a_addrs.clear();
                            cache.a_eq_reg = match src_l.as_str() {
                                "b" => Some('b'),
                                "c" => Some('c'),
                                _ => None,
                            };
                        }
                        "b" => {
                            cache.a_eq_reg = if src_l == "a" {
                                Some('b')
                            } else if cache.a_eq_reg == Some('b') {
                                None
                            } else {
                                cache.a_eq_reg
                            };
                        }
                        "c" => {
                            cache.a_eq_reg = if src_l == "a" {
                                Some('c')
                            } else if cache.a_eq_reg == Some('c') {
                                None
                            } else {
                                cache.a_eq_reg
                            };
                        }
                        "bc" => cache.a_eq_reg = None,
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
                        let is_or_a =
                            mnemonic == "or" && ops.len() == 1 && ops[0].eq_ignore_ascii_case("a");
                        if !is_or_a {
                            cache.a_addrs.clear();
                            cache.a_eq_reg = None;
                        }
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

                // `push` modifies SP/stack only, preserving all register & RAM caches.
                if mnemonic == "push" {
                    out.push(lines[i].clone());
                    i += 1;
                    continue;
                }
                // `inc`/`dec` and single-register bit/shift ops invalidate only the target register.
                let reg_target = match mnemonic.as_str() {
                    "inc" | "dec" | "sla" | "sra" | "srl" | "rl" | "rr" | "rlc" | "rrc"
                        if ops.len() == 1 && !ops[0].contains('(') =>
                    {
                        Some(ops[0].to_ascii_lowercase())
                    }
                    "set" | "res" if ops.len() == 2 && !ops[1].contains('(') => {
                        Some(ops[1].to_ascii_lowercase())
                    }
                    _ => None,
                };
                if let Some(r) = reg_target {
                    match r.as_str() {
                        "hl" | "h" | "l" => {
                            cache.hl_addrs.clear();
                            cache.hl_ix_disp = None;
                        }
                        "de" | "d" | "e" => cache.de_addrs.clear(),
                        "a" => {
                            cache.a_addrs.clear();
                            cache.a_eq_reg = None;
                        }
                        "b" if cache.a_eq_reg == Some('b') => cache.a_eq_reg = None,
                        "c" if cache.a_eq_reg == Some('c') => cache.a_eq_reg = None,
                        "bc" => cache.a_eq_reg = None,
                        "ix" => cache.hl_ix_disp = None,
                        _ => {}
                    }
                    out.push(lines[i].clone());
                    i += 1;
                    continue;
                }

                // Anything else (calls, ldir, pop, indirect inc/dec): conservatively clear
                cache.clear();
                out.push(lines[i].clone());
                i += 1;
            }
        }
    }
    (out, changed)
}
