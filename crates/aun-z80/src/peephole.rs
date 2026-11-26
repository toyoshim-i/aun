//! Lightweight Z80 peephole optimizer operating on emitted assembly text.
//!
//! Transformations are strictly local within basic blocks (never crossing labels
//! or directives) and preserve Z80 flag semantics.

/// Runs peephole optimization passes on Aun-generated Z80 assembly until convergence.
pub fn optimize_asm(src: &str) -> String {
    let mut lines: Vec<String> = src.lines().map(String::from).collect();
    for _ in 0..8 {
        let (next, changed) = run_pass(&lines);
        lines = next;
        if !changed {
            break;
        }
    }
    let mut out = lines.join("\n");
    if src.ends_with('\n') {
        out.push('\n');
    }
    out
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
    let mut out = Vec::with_capacity(lines.len());
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
}
