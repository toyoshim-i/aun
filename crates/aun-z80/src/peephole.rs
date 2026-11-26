//! Lightweight Z80 peephole optimizer operating on emitted assembly text.
//!
//! Transformations are strictly local within basic blocks (never crossing labels
//! or directives) and preserve Z80 flag semantics.

use std::collections::{HashMap, HashSet};

/// Runs peephole optimization passes on Aun-generated Z80 assembly until convergence.
pub fn optimize_asm(src: &str) -> String {
    optimize_asm_per_fn(src, &HashMap::new())
}

/// Runs peephole optimization plus per-function dead scratch-slot store elimination.
///
/// `fn_scratch_ranges` maps each top-level Z80 function label to its own exclusive
/// `(scratch_base, scratch_end)` interval so dead-store elimination never confuses
/// sibling functions' overlaid Uchiguri locals with another function's scratch slots.
pub fn optimize_asm_per_fn(src: &str, fn_scratch_ranges: &HashMap<String, (u16, u16)>) -> String {
    let mut lines: Vec<String> = src.lines().map(String::from).collect();
    for _ in 0..8 {
        let (next, changed1) = run_pass(&lines);
        let (next2, changed2) = if fn_scratch_ranges.is_empty() {
            (next, false)
        } else {
            eliminate_dead_scratch_stores(&next, fn_scratch_ranges)
        };
        lines = next2;
        if !changed1 && !changed2 {
            break;
        }
    }
    let mut out = lines.join("\n");
    if src.ends_with('\n') {
        out.push('\n');
    }
    out
}

fn parse_mem_addr(op: &str) -> Option<u16> {
    let inner = op.trim().strip_prefix('(')?.strip_suffix(')')?.trim();
    if let Some(hex) = inner.strip_prefix("0x").or_else(|| inner.strip_prefix("0X")) {
        u16::from_str_radix(hex, 16).ok()
    } else {
        None
    }
}

/// Deletes stores `ld (0xXXXX), reg` to the *current function's* own scratch addresses
/// that are never read anywhere within that function, along with an immediately preceding
/// pure `ld reg, imm` when the very next instruction overwrites `reg` without reading it.
fn eliminate_dead_scratch_stores(
    lines: &[String],
    fn_scratch_ranges: &HashMap<String, (u16, u16)>,
) -> (Vec<String>, bool) {
    let kinds: Vec<LineKind> = lines.iter().map(|l| classify_line(l)).collect();

    // Partition lines into per-function slices keyed by top-level labels in `fn_scratch_ranges`
    let mut fn_spans: Vec<(usize, usize, Option<(u16, u16)>)> = Vec::new();
    let mut cur_start = 0usize;
    let mut cur_range: Option<(u16, u16)> = None;
    for (idx, kind) in kinds.iter().enumerate() {
        if let LineKind::Label(lbl) = kind
            && let Some(&range) = fn_scratch_ranges.get(lbl)
        {
            if idx > cur_start {
                fn_spans.push((cur_start, idx, cur_range));
            }
            cur_start = idx;
            cur_range = Some(range);
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
            // Non-function section label (e.g. `_tmask`, `_vtbl_`, `__aun_panic`): stop current fn
            if idx > cur_start {
                fn_spans.push((cur_start, idx, cur_range));
            }
            cur_start = idx;
            cur_range = None;
        }
    }
    if cur_start < lines.len() {
        fn_spans.push((cur_start, lines.len(), cur_range));
    }

    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut changed = false;

    for (start, end, range_opt) in fn_spans {
        let Some((lo, hi)) = range_opt else {
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
                if matches!(ops[0].to_ascii_lowercase().as_str(), "hl" | "de" | "bc") {
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
                && addr >= lo
                && addr < hi
                && !read_addrs.contains(&addr)
                && !(matches!(ops[1].to_ascii_lowercase().as_str(), "hl" | "de" | "bc")
                    && read_addrs.contains(&addr.wrapping_add(1)))
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
            "adc" | "sbc" | "rla" | "rra" | "ccf" => true,
            _ => false,
        },
        _ => false,
    }
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

                // Rule 3: `ld a, 0` -> `xor a` when next instruction does not read flags
                if mnemonic == "ld"
                    && ops.len() == 2
                    && ops[0].eq_ignore_ascii_case("a")
                    && is_zero_imm(&ops[1])
                {
                    let next_reads = next_non_comment(&kinds, i + 1)
                        .is_some_and(|ni| reads_flags(&kinds[ni]));
                    if !next_reads {
                        out.push("    xor a".to_string());
                        changed = true;
                        i += 1;
                        continue;
                    }
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

                    // Rule 6: `jp lbl` / `jr lbl` immediately followed by `lbl:` -> drop jump
                    if (mnemonic == "jp" || mnemonic == "jr")
                        && ops.len() == 1
                        && let LineKind::Label(lbl) = &kinds[ni]
                        && ops[0] == *lbl
                    {
                        for line in &lines[i + 1..ni] {
                            out.push(line.clone());
                        }
                        changed = true;
                        i = ni;
                        continue;
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
        ranges.insert("answer".to_string(), (0xC000, 0xC002));
        ranges.insert("sibling".to_string(), (0xC004, 0xC006));

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
