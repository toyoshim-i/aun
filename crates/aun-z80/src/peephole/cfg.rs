use super::*;

fn est_inst_bytes(kind: &LineKind) -> usize {
    let LineKind::Inst { mnemonic, ops } = kind else {
        return 0;
    };
    let has_idx = ops
        .iter()
        .any(|o| parse_ix_disp(o).is_some() || o.to_ascii_lowercase().contains("(iy"));
    let has_mem = ops
        .iter()
        .any(|o| parse_mem_addr(o).is_some() || o.trim().starts_with("(__aun_"));
    let is_imm = |s: &str| {
        let t = s.trim().trim_start_matches('#');
        t.starts_with("0x") || t.starts_with("0X") || (!t.is_empty() && t.chars().all(|c| c.is_ascii_digit()))
    };
    match mnemonic.as_str() {
        "ret" | "nop" | "scf" | "ccf" | "cpl" | "daa" | "di" | "ei" | "halt" | "rlca"
        | "rrca" | "rla" | "rra" | "exx" => 1,
        "ex" => {
            if ops.iter().any(|o| {
                let l = o.to_ascii_lowercase();
                l.contains("ix") || l.contains("iy")
            }) {
                2
            } else {
                1
            }
        }
        "push" | "pop" => {
            if ops
                .first()
                .is_some_and(|o| matches!(o.to_ascii_lowercase().as_str(), "ix" | "iy"))
            {
                2
            } else {
                1
            }
        }
        "jp" | "call" => 3,
        "jr" | "djnz" | "in" | "out" => 2,
        "rst" => 1,
        "neg" | "reti" | "retn" | "rld" | "rrd" | "ldi" | "ldd" | "ldir" | "lddr" | "cpi"
        | "cpd" | "cpir" | "cpdr" => 2,
        "bit" | "set" | "res" | "rlc" | "rrc" | "rl" | "rr" | "sla" | "sra" | "srl" => {
            if has_idx { 4 } else { 2 }
        }
        "inc" | "dec" => {
            if has_idx {
                3
            } else if ops
                .first()
                .is_some_and(|o| matches!(o.to_ascii_lowercase().as_str(), "ix" | "iy"))
            {
                2
            } else {
                1
            }
        }
        "add" | "adc" | "sbc" | "sub" | "and" | "xor" | "or" | "cp" => {
            if has_idx {
                3
            } else if ops
                .first()
                .is_some_and(|o| matches!(o.to_ascii_lowercase().as_str(), "hl" | "ix" | "iy"))
            {
                if mnemonic == "add" && ops.first().is_some_and(|o| o.eq_ignore_ascii_case("hl"))
                {
                    1
                } else {
                    2
                }
            } else if ops.last().is_some_and(|o| is_imm(o)) {
                2
            } else {
                1
            }
        }
        "ld" if ops.len() == 2 => {
            let dst = ops[0].trim().to_ascii_lowercase();
            let src = ops[1].trim().to_ascii_lowercase();
            let is_16 = |s: &str| matches!(s, "bc" | "de" | "hl" | "ix" | "iy" | "sp");
            if has_idx {
                if is_imm(&src) { 4 } else { 3 }
            } else if has_mem {
                if matches!(dst.as_str(), "a" | "hl") || matches!(src.as_str(), "a" | "hl") {
                    3
                } else {
                    4
                }
            } else if is_16(&dst) {
                if dst == "sp" && src == "hl" {
                    1
                } else if is_imm(&src) {
                    if matches!(dst.as_str(), "ix" | "iy") { 4 } else { 3 }
                } else {
                    2
                }
            } else if is_imm(&src) {
                2
            } else {
                1
            }
        }
        _ => 2,
    }
}

/// Tail-merging (cross-jumping) for `ret`- and `jp <label>`-terminated basic blocks within
/// the same function: when two `_bb<digits>` blocks share an identical trailing instruction
/// sequence (`common_len >= 2`, `common_bytes >= 5`) ending in `ret` or unconditional `jp <label>`,
/// replaces the earlier block's duplicate suffix with `jp <label>` targeting the later block's
/// shared suffix (reusing the later block's `_bb<digits>` label if the entire block matches, or
/// inserting a fresh `{prefix}_bb<N>` label).
pub(super) fn merge_ret_tails(lines: &[String]) -> (Vec<String>, bool) {
    let mut cur_lines = lines.to_vec();
    let mut any_changed = false;

    for _ in 0..64 {
        let kinds: Vec<LineKind> = cur_lines.iter().map(|l| classify_line(l)).collect();

        // Collect instruction-only spans for each `_bb<digits>` block ending in `ret` or `jp <label>`.
        let mut tail_blocks: Vec<(String, Vec<usize>)> = Vec::new();
        let mut existing_labels: HashSet<String> = HashSet::new();
        let mut cur_fn_prefix: Option<String> = None;
        let mut cur_insts: Vec<usize> = Vec::new();

        for (idx, kind) in kinds.iter().enumerate() {
            match kind {
                LineKind::Label(lbl) => {
                    existing_labels.insert(lbl.clone());
                    cur_insts.clear();
                    if let Some((prefix, _)) = lbl.rsplit_once("_bb")
                        && is_bb_label(lbl)
                    {
                        cur_fn_prefix = Some(prefix.to_string());
                    } else {
                        cur_fn_prefix = None;
                    }
                }
                LineKind::Directive => {
                    cur_fn_prefix = None;
                    cur_insts.clear();
                }
                LineKind::Inst { mnemonic, ops } => {
                    if let Some(ref prefix) = cur_fn_prefix {
                        cur_insts.push(idx);
                        if (mnemonic == "ret" && ops.is_empty())
                            || (mnemonic == "jp" && ops.len() == 1)
                        {
                            tail_blocks.push((prefix.clone(), std::mem::take(&mut cur_insts)));
                            cur_fn_prefix = None;
                        } else if matches!(mnemonic.as_str(), "jr" | "reti" | "retn" | "halt")
                            && ops.len() <= 1
                        {
                            cur_fn_prefix = None;
                            cur_insts.clear();
                        }
                    }
                }
                LineKind::EmptyOrComment => {}
            }
        }

        // Find the pair within the same function sharing the largest trailing byte sequence
        // (`common_bytes >= 5`, `common_len >= 2`).
        let mut best: Option<(String, usize, usize, usize, usize)> = None;
        for a in 0..tail_blocks.len() {
            for b in (a + 1)..tail_blocks.len() {
                if tail_blocks[a].0 != tail_blocks[b].0 {
                    continue;
                }
                let ia = &tail_blocks[a].1;
                let ib = &tail_blocks[b].1;
                let mut k = 0usize;
                let mut bytes = 0usize;
                while k < ia.len() && k < ib.len() {
                    let kind_a = &kinds[ia[ia.len() - 1 - k]];
                    if kind_a == &kinds[ib[ib.len() - 1 - k]] {
                        bytes += est_inst_bytes(kind_a);
                        k += 1;
                    } else {
                        break;
                    }
                }
                if k >= 2
                    && bytes >= 5
                    && best
                        .as_ref()
                        .is_none_or(|&(_, _, _, best_bytes, best_k)| (bytes, k) > (best_bytes, best_k))
                {
                    best = Some((tail_blocks[a].0.clone(), a, b, bytes, k));
                }
            }
        }

        let Some((prefix, a_idx, b_idx, _, common_len)) = best else {
            break;
        };

        let ia = &tail_blocks[a_idx].1;
        let ib = &tail_blocks[b_idx].1;
        let a_start_line = ia[ia.len() - common_len];
        let a_end_line = *ia.last().unwrap();
        let b_start_line = ib[ib.len() - common_len];

        // Reuse the immediately preceding `_bb<digits>` label when the entire block B matches.
        let reused_lbl = if common_len == ib.len() {
            let mut prev = b_start_line;
            let mut found = None;
            while prev > 0 {
                prev -= 1;
                match &kinds[prev] {
                    LineKind::EmptyOrComment => continue,
                    LineKind::Label(lbl) if is_bb_label(lbl) => {
                        found = Some(lbl.clone());
                        break;
                    }
                    _ => break,
                }
            }
            found
        } else {
            None
        };

        let (tail_lbl, insert_label) = if let Some(lbl) = reused_lbl {
            (lbl, false)
        } else {
            let mut n = 9000u32;
            let lbl = loop {
                let cand = format!("{prefix}_bb{n}");
                if !existing_labels.contains(&cand) {
                    break cand;
                }
                n += 1;
            };
            (lbl, true)
        };

        let mut next: Vec<String> = Vec::with_capacity(cur_lines.len() + 1);
        for (idx, line) in cur_lines.iter().enumerate() {
            if idx == a_start_line {
                next.push(format!("    jp {tail_lbl}"));
            } else if idx > a_start_line && idx <= a_end_line {
                continue;
            } else if idx == b_start_line && insert_label {
                next.push(format!("{tail_lbl}:"));
                next.push(line.clone());
            } else {
                next.push(line.clone());
            }
        }

        cur_lines = next;
        any_changed = true;
    }

    (cur_lines, any_changed)
}

/// Jump-threading for compiler-generated `_bb<digits>` trampoline labels:
/// if a `_bb<digits>` label `L1` is immediately followed by an unconditional `jp L2` / `jr L2`
/// (where `L2` is also `_bb<digits>` and `L2 != L1`), rewrites all `jp/jr [cc,] L1` to target `L2`.
pub(super) fn thread_empty_jumps(lines: &[String]) -> (Vec<String>, bool) {
    let kinds: Vec<LineKind> = lines.iter().map(|l| classify_line(l)).collect();
    let mut redirect: HashMap<String, String> = HashMap::new();

    for i in 0..kinds.len() {
        if let LineKind::Label(lbl) = &kinds[i]
            && is_bb_label(lbl)
        {
            // Skip past comments and adjacent labels to find the first instruction
            let mut k = i + 1;
            while k < kinds.len()
                && matches!(kinds[k], LineKind::EmptyOrComment | LineKind::Label(_))
            {
                k += 1;
            }
            if k < kinds.len()
                && let LineKind::Inst { mnemonic, ops } = &kinds[k]
                && (mnemonic == "jp" || mnemonic == "jr")
                && ops.len() == 1
                && is_bb_label(&ops[0])
                && &ops[0] != lbl
            {
                redirect.insert(lbl.clone(), ops[0].clone());
            }
        }
    }
    if redirect.is_empty() {
        return (lines.to_vec(), false);
    }

    let mut out = Vec::with_capacity(lines.len());
    let mut changed = false;
    for (i, line) in lines.iter().enumerate() {
        // Only thread `jp` (never `jr`, which has a +-127 byte range limit) and guard cycles
        if let LineKind::Inst { mnemonic, ops } = &kinds[i]
            && mnemonic == "jp"
            && let Some(target) = ops.last()
            && let Some(new_tgt) = redirect.get(target)
            && redirect.get(new_tgt).map(String::as_str) != Some(target.as_str())
        {
            if ops.len() == 1 {
                out.push(format!("    jp {new_tgt}"));
            } else {
                out.push(format!("    jp {}, {new_tgt}", ops[0]));
            }
            changed = true;
            continue;
        }
        out.push(line.clone());
    }
    (out, changed)
}

fn is_uncond_term(kind: &LineKind) -> bool {
    match kind {
        LineKind::Directive => true,
        LineKind::Inst { mnemonic, ops } => match mnemonic.as_str() {
            "ret" | "reti" | "retn" | "halt" => ops.is_empty(),
            "jp" | "jr" => ops.len() == 1,
            _ => false,
        },
        _ => false,
    }
}

fn rewrite_cond_returns(lines: &[String]) -> (Vec<String>, bool) {
    let kinds: Vec<LineKind> = lines.iter().map(|l| classify_line(l)).collect();
    let mut pure_ret_labels: HashSet<String> = HashSet::new();

    let mut i = 0usize;
    while i < kinds.len() {
        if let LineKind::Label(lbl) = &kinds[i]
            && is_bb_label(lbl)
        {
            let mut k = i + 1;
            while k < kinds.len() && matches!(kinds[k], LineKind::EmptyOrComment) {
                k += 1;
            }
            if k < kinds.len()
                && let LineKind::Inst { mnemonic, ops } = &kinds[k]
                && mnemonic == "ret"
                && ops.is_empty()
            {
                pure_ret_labels.insert(lbl.clone());
            }
        }
        i += 1;
    }

    if pure_ret_labels.is_empty() {
        return (lines.to_vec(), false);
    }

    let mut out = Vec::with_capacity(lines.len());
    let mut changed = false;
    for (idx, line) in lines.iter().enumerate() {
        if let LineKind::Inst { mnemonic, ops } = &kinds[idx]
            && matches!(mnemonic.as_str(), "jp" | "jr")
            && let Some(target) = ops.last()
            && pure_ret_labels.contains(target)
        {
            if ops.len() == 1 {
                out.push("    ret".to_string());
                changed = true;
                continue;
            }
            if ops.len() == 2 {
                let cc = ops[0].to_ascii_lowercase();
                if matches!(cc.as_str(), "z" | "nz" | "c" | "nc" | "po" | "pe" | "p" | "m") {
                    out.push(format!("    ret {cc}"));
                    changed = true;
                    continue;
                }
            }
        }
        out.push(line.clone());
    }
    (out, changed)
}

fn remove_unreachable_bb_blocks(lines: &[String]) -> (Vec<String>, bool) {
    let kinds: Vec<LineKind> = lines.iter().map(|l| classify_line(l)).collect();
    let mut referenced: HashSet<String> = HashSet::new();
    for kind in &kinds {
        if let LineKind::Inst { mnemonic, ops } = kind {
            if matches!(mnemonic.as_str(), "jp" | "jr" | "djnz" | "call")
                && let Some(t) = ops.last()
            {
                referenced.insert(t.clone());
            }
        }
    }
    for raw in lines {
        let code = raw.split(';').next().unwrap_or("").trim();
        if let Some(rest) = code.strip_prefix(".dw ") {
            for part in rest.split(',') {
                referenced.insert(part.trim().to_string());
            }
        }
    }

    let mut drop = vec![false; lines.len()];
    let mut changed = false;
    let mut i = 0usize;
    while i < kinds.len() {
        if let LineKind::Label(lbl) = &kinds[i]
            && is_bb_label(lbl)
            && !referenced.contains(lbl)
        {
            // Only remove if immediately preceded (ignoring blanks/comments) by an unconditional
            // terminator or directive (so control cannot fall through into `lbl`).
            let mut p = i;
            let mut has_fallthrough = true;
            while p > 0 {
                p -= 1;
                match &kinds[p] {
                    LineKind::EmptyOrComment => continue,
                    other => {
                        has_fallthrough = !is_uncond_term(other);
                        break;
                    }
                }
            }
            if !has_fallthrough {
                drop[i] = true;
                changed = true;
                let mut k = i + 1;
                while k < kinds.len()
                    && !matches!(kinds[k], LineKind::Label(_) | LineKind::Directive)
                {
                    drop[k] = true;
                    k += 1;
                }
                i = k;
                continue;
            }
        }
        i += 1;
    }
    if !changed {
        return (lines.to_vec(), false);
    }
    let out = lines
        .iter()
        .enumerate()
        .filter(|&(idx, _)| !drop[idx])
        .map(|(_, l)| l.clone())
        .collect();
    (out, true)
}

fn deduplicate_identical_blocks(lines: &[String]) -> (Vec<String>, bool) {
    let kinds: Vec<LineKind> = lines.iter().map(|l| classify_line(l)).collect();
    let mut has_jr_or_dw_ref: HashSet<String> = HashSet::new();
    for (idx, kind) in kinds.iter().enumerate() {
        if let LineKind::Inst { mnemonic, ops } = kind
            && matches!(mnemonic.as_str(), "jr" | "djnz")
            && let Some(t) = ops.last()
        {
            has_jr_or_dw_ref.insert(t.clone());
        }
        let code = lines[idx].split(';').next().unwrap_or("").trim();
        if let Some(rest) = code.strip_prefix(".dw ") {
            for part in rest.split(',') {
                has_jr_or_dw_ref.insert(part.trim().to_string());
            }
        }
    }

    struct BlockInfo {
        fn_prefix: String,
        label: String,
        start_line: usize,
        end_line: usize,
        body: Vec<LineKind>,
        no_fallthrough_in: bool,
        no_short_ref: bool,
    }

    let mut blocks: Vec<BlockInfo> = Vec::new();
    let mut i = 0usize;
    while i < kinds.len() {
        if let LineKind::Label(lbl) = &kinds[i]
            && is_bb_label(lbl)
            && let Some((prefix, _)) = lbl.rsplit_once("_bb")
        {
            let mut p = i;
            let mut no_fallthrough = false;
            while p > 0 {
                p -= 1;
                match &kinds[p] {
                    LineKind::EmptyOrComment => continue,
                    other => {
                        no_fallthrough = is_uncond_term(other);
                        break;
                    }
                }
            }
            let mut k = i + 1;
            let mut body = Vec::new();
            while k < kinds.len() && !matches!(kinds[k], LineKind::Label(_) | LineKind::Directive) {
                if matches!(kinds[k], LineKind::Inst { .. }) {
                    body.push(kinds[k].clone());
                }
                k += 1;
            }
            let ends_cleanly = body.last().is_some_and(|last| match last {
                LineKind::Inst { mnemonic, ops } => {
                    (mnemonic == "ret" && ops.is_empty()) || (mnemonic == "jp" && ops.len() == 1)
                }
                _ => false,
            });
            if ends_cleanly {
                blocks.push(BlockInfo {
                    fn_prefix: prefix.to_string(),
                    label: lbl.clone(),
                    start_line: i,
                    end_line: k,
                    body,
                    no_fallthrough_in: no_fallthrough,
                    no_short_ref: !has_jr_or_dw_ref.contains(lbl),
                });
            }
            i = k;
            continue;
        }
        i += 1;
    }

    let mut redirect: HashMap<String, String> = HashMap::new();
    let mut drop = vec![false; lines.len()];
    let mut canonical: Vec<usize> = Vec::new();
    for bi in 0..blocks.len() {
        let mut matched_canon = None;
        for &ci in &canonical {
            if blocks[bi].fn_prefix == blocks[ci].fn_prefix && blocks[bi].body == blocks[ci].body {
                matched_canon = Some(ci);
                break;
            }
        }
        if let Some(ci) = matched_canon {
            if blocks[bi].no_fallthrough_in
                && blocks[bi].no_short_ref
                && blocks[bi].label != blocks[ci].label
            {
                redirect.insert(blocks[bi].label.clone(), blocks[ci].label.clone());
                for line_idx in blocks[bi].start_line..blocks[bi].end_line {
                    drop[line_idx] = true;
                }
                continue;
            }
        }
        canonical.push(bi);
    }

    if redirect.is_empty() {
        return (lines.to_vec(), false);
    }

    let mut out = Vec::with_capacity(lines.len());
    for (idx, line) in lines.iter().enumerate() {
        if drop[idx] {
            continue;
        }
        if let LineKind::Inst { mnemonic, ops } = &kinds[idx]
            && mnemonic == "jp"
            && let Some(target) = ops.last()
            && let Some(canon_lbl) = redirect.get(target)
        {
            if ops.len() == 1 {
                out.push(format!("    jp {canon_lbl}"));
            } else {
                out.push(format!("    jp {}, {canon_lbl}", ops[0]));
            }
            continue;
        }
        out.push(line.clone());
    }
    (out, true)
}

/// Simplifies conditional/unconditional jumps targeting pure `ret` blocks into `ret [cc]`,
/// deduplicates identical `ret`/`jp`-terminated `_bb` blocks, and deletes orphaned `_bb` blocks.
pub(super) fn simplify_cond_returns_and_dedup_blocks(lines: &[String]) -> (Vec<String>, bool) {
    let (s1, c1) = rewrite_cond_returns(lines);
    let (s2, c2) = remove_unreachable_bb_blocks(&s1);
    let (s3, c3) = deduplicate_identical_blocks(&s2);
    (s3, c1 || c2 || c3)
}
