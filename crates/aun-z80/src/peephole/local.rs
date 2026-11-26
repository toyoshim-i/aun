use super::*;

pub(super) fn run_pass(lines: &[String]) -> (Vec<String>, bool) {
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

                // Rule 3b: `ld a, (ix + d); inc/dec a; ld (ix + d), a` -> `inc/dec (ix + d)` (7B -> 3B!)
                // when `a` is dead afterward (next instruction overwrites `a` or jumps to a `_bb` label).
                if mnemonic == "ld"
                    && ops.len() == 2
                    && ops[0].eq_ignore_ascii_case("a")
                    && let Some(disp) = parse_ix_disp(&ops[1])
                    && let Some(n1) = next_non_comment(&kinds, i + 1)
                    && let LineKind::Inst {
                        mnemonic: m1,
                        ops: ops1,
                    } = &kinds[n1]
                    && matches!(m1.as_str(), "inc" | "dec")
                    && ops1.len() == 1
                    && ops1[0].eq_ignore_ascii_case("a")
                    && let Some(n2) = next_non_comment(&kinds, n1 + 1)
                    && let LineKind::Inst {
                        mnemonic: m2,
                        ops: ops2,
                    } = &kinds[n2]
                    && m2 == "ld"
                    && ops2.len() == 2
                    && parse_ix_disp(&ops2[0]) == Some(disp)
                    && ops2[1].eq_ignore_ascii_case("a")
                {
                    let ix_op = ops[1].clone();
                    // Check if immediately followed by `ld a, (ix + d); or a; jp/jr z/nz _bb`
                    let mut folded_test = false;
                    if let Some(n3) = next_non_comment(&kinds, n2 + 1)
                        && let LineKind::Inst {
                            mnemonic: m3,
                            ops: ops3,
                        } = &kinds[n3]
                        && m3 == "ld"
                        && ops3.len() == 2
                        && ops3[0].eq_ignore_ascii_case("a")
                        && parse_ix_disp(&ops3[1]) == Some(disp)
                        && let Some(n4) = next_non_comment(&kinds, n3 + 1)
                        && let LineKind::Inst {
                            mnemonic: m4,
                            ops: ops4,
                        } = &kinds[n4]
                        && m4 == "or"
                        && ops4.len() == 1
                        && ops4[0].eq_ignore_ascii_case("a")
                        && let Some(n5) = next_non_comment(&kinds, n4 + 1)
                        && let LineKind::Inst {
                            mnemonic: m5,
                            ops: ops5,
                        } = &kinds[n5]
                        && matches!(m5.as_str(), "jp" | "jr")
                        && ops5.len() == 2
                        && matches!(ops5[0].to_ascii_lowercase().as_str(), "z" | "nz")
                        && is_bb_label(&ops5[1])
                    {
                        out.push(format!("    {m1} {ix_op}"));
                        changed = true;
                        i = n4 + 1;
                        folded_test = true;
                    }
                    if folded_test {
                        continue;
                    }
                    let a_dead = next_non_comment(&kinds, n2 + 1).is_some_and(|n3| match &kinds[n3] {
                        LineKind::Label(lbl) => is_bb_label(lbl),
                        LineKind::Inst {
                            mnemonic: m3,
                            ops: ops3,
                        } => {
                            (m3 == "ld"
                                && ops3.len() == 2
                                && ops3[0].eq_ignore_ascii_case("a")
                                && !ops3[1].eq_ignore_ascii_case("a"))
                                || (m3 == "xor"
                                    && ops3.len() == 1
                                    && ops3[0].eq_ignore_ascii_case("a"))
                                || (matches!(m3.as_str(), "jp" | "jr")
                                    && ops3.len() == 1
                                    && is_bb_label(&ops3[0]))
                        }
                        _ => false,
                    });
                    if a_dead {
                        out.push(format!("    {m1} {ix_op}"));
                        changed = true;
                        i = n2 + 1;
                        continue;
                    }
                }

                // Rule 3c: `ld a, (hl); inc/dec a; ld (hl), a` -> `inc/dec (hl)` (3B -> 1B!) when `a` is dead
                if mnemonic == "ld"
                    && ops.len() == 2
                    && ops[0].eq_ignore_ascii_case("a")
                    && ops[1].eq_ignore_ascii_case("(hl)")
                    && let Some(n1) = next_non_comment(&kinds, i + 1)
                    && let LineKind::Inst {
                        mnemonic: m1,
                        ops: ops1,
                    } = &kinds[n1]
                    && matches!(m1.as_str(), "inc" | "dec")
                    && ops1.len() == 1
                    && ops1[0].eq_ignore_ascii_case("a")
                    && let Some(n2) = next_non_comment(&kinds, n1 + 1)
                    && let LineKind::Inst {
                        mnemonic: m2,
                        ops: ops2,
                    } = &kinds[n2]
                    && m2 == "ld"
                    && ops2.len() == 2
                    && ops2[0].eq_ignore_ascii_case("(hl)")
                    && ops2[1].eq_ignore_ascii_case("a")
                {
                    let a_dead = next_non_comment(&kinds, n2 + 1).is_some_and(|n3| match &kinds[n3] {
                        LineKind::Label(lbl) => is_bb_label(lbl),
                        LineKind::Inst {
                            mnemonic: m3,
                            ops: ops3,
                        } => {
                            (m3 == "ld"
                                && ops3.len() == 2
                                && ops3[0].eq_ignore_ascii_case("a")
                                && !ops3[1].eq_ignore_ascii_case("a"))
                                || (m3 == "xor"
                                    && ops3.len() == 1
                                    && ops3[0].eq_ignore_ascii_case("a"))
                                || (matches!(m3.as_str(), "jp" | "jr")
                                    && ops3.len() == 1
                                    && is_bb_label(&ops3[0]))
                        }
                        _ => false,
                    });
                    if a_dead {
                        out.push(format!("    {m1} (hl)"));
                        changed = true;
                        i = n2 + 1;
                        continue;
                    }
                }

                // Rule 3d: `ld a, c; inc/dec a; ld c, a` -> `inc/dec c` (3B -> 1B!) when `a` is dead
                if mnemonic == "ld"
                    && ops.len() == 2
                    && ops[0].eq_ignore_ascii_case("a")
                    && ops[1].eq_ignore_ascii_case("c")
                    && let Some(n1) = next_non_comment(&kinds, i + 1)
                    && let LineKind::Inst {
                        mnemonic: m1,
                        ops: ops1,
                    } = &kinds[n1]
                    && matches!(m1.as_str(), "inc" | "dec")
                    && ops1.len() == 1
                    && ops1[0].eq_ignore_ascii_case("a")
                    && let Some(n2) = next_non_comment(&kinds, n1 + 1)
                    && let LineKind::Inst {
                        mnemonic: m2,
                        ops: ops2,
                    } = &kinds[n2]
                    && m2 == "ld"
                    && ops2.len() == 2
                    && ops2[0].eq_ignore_ascii_case("c")
                    && ops2[1].eq_ignore_ascii_case("a")
                {
                    let a_dead = next_non_comment(&kinds, n2 + 1).is_some_and(|n3| match &kinds[n3] {
                        LineKind::Label(lbl) => is_bb_label(lbl),
                        LineKind::Inst {
                            mnemonic: m3,
                            ops: ops3,
                        } => {
                            (m3 == "ld"
                                && ops3.len() == 2
                                && ops3[0].eq_ignore_ascii_case("a")
                                && !ops3[1].eq_ignore_ascii_case("a"))
                                || (m3 == "xor"
                                    && ops3.len() == 1
                                    && ops3[0].eq_ignore_ascii_case("a"))
                                || (matches!(m3.as_str(), "jp" | "jr")
                                    && ops3.len() == 1
                                    && is_bb_label(&ops3[0]))
                        }
                        _ => false,
                    });
                    if a_dead {
                        out.push(format!("    {m1} c"));
                        changed = true;
                        i = n2 + 1;
                        continue;
                    }
                }

                // Rule 3e: consecutive `ld (ix + d), 0` zero-init run (>= 2) -> `xor a; ld (ix + d), a ...`
                if mnemonic == "ld"
                    && ops.len() == 2
                    && parse_ix_disp(&ops[0]).is_some()
                    && is_zero_imm(&ops[1])
                {
                    let mut run_idxs = vec![i];
                    let mut cur = i;
                    while let Some(ni) = next_non_comment(&kinds, cur + 1) {
                        if let LineKind::Inst {
                            mnemonic: nm,
                            ops: nops,
                        } = &kinds[ni]
                            && nm == "ld"
                            && nops.len() == 2
                            && parse_ix_disp(&nops[0]).is_some()
                            && is_zero_imm(&nops[1])
                        {
                            run_idxs.push(ni);
                            cur = ni;
                        } else {
                            break;
                        }
                    }
                    if run_idxs.len() >= 2 {
                        let last_idx = *run_idxs.last().unwrap();
                        let a_dead = next_non_comment(&kinds, last_idx + 1).is_some_and(|n3| {
                            match &kinds[n3] {
                                LineKind::Label(lbl) => is_bb_label(lbl),
                                LineKind::Inst {
                                    mnemonic: m3,
                                    ops: ops3,
                                } => {
                                    (m3 == "ld"
                                        && ops3.len() == 2
                                        && ops3[0].eq_ignore_ascii_case("a")
                                        && !ops3[1].eq_ignore_ascii_case("a"))
                                        || (m3 == "xor"
                                            && ops3.len() == 1
                                            && ops3[0].eq_ignore_ascii_case("a"))
                                        || (matches!(m3.as_str(), "jp" | "jr")
                                            && ops3.len() == 1
                                            && is_bb_label(&ops3[0]))
                                }
                                _ => false,
                            }
                        });
                        if a_dead && flags_dead_after(&kinds, last_idx + 1) {
                            out.push("    xor a".to_string());
                            for &ri in &run_idxs {
                                if let LineKind::Inst { ops: rops, .. } = &kinds[ri] {
                                    out.push(format!("    ld {}, a", rops[0]));
                                }
                            }
                            changed = true;
                            i = last_idx + 1;
                            continue;
                        }
                    }
                }

                // Look ahead to the next non-comment line for 2-instruction patterns
                if let Some(ni) = next_non_comment(&kinds, i + 1) {
                    // Rule 1b: `ld r1, r2; ld r2, r1` (8-bit regs) -> `ld r1, r2`
                    if mnemonic == "ld"
                        && ops.len() == 2
                        && matches!(
                            ops[0].to_ascii_lowercase().as_str(),
                            "a" | "b" | "c" | "d" | "e" | "h" | "l"
                        )
                        && matches!(
                            ops[1].to_ascii_lowercase().as_str(),
                            "a" | "b" | "c" | "d" | "e" | "h" | "l"
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
                        out.push(lines[i].clone());
                        for line in &lines[i + 1..ni] {
                            out.push(line.clone());
                        }
                        changed = true;
                        i = ni + 1;
                        continue;
                    }

                    // Rule 1c: `ld r1, a; ld a, r2; <xor|and|or> r1` -> `ld r1, a; <xor|and|or> r2`
                    if mnemonic == "ld"
                        && ops.len() == 2
                        && matches!(
                            ops[0].to_ascii_lowercase().as_str(),
                            "b" | "c" | "d" | "e" | "h" | "l"
                        )
                        && ops[1].eq_ignore_ascii_case("a")
                        && let LineKind::Inst {
                            mnemonic: m2,
                            ops: ops2,
                        } = &kinds[ni]
                        && m2 == "ld"
                        && ops2.len() == 2
                        && ops2[0].eq_ignore_ascii_case("a")
                        && matches!(
                            ops2[1].to_ascii_lowercase().as_str(),
                            "b" | "c" | "d" | "e" | "h" | "l"
                        )
                        && !ops2[1].eq_ignore_ascii_case(&ops[0])
                        && let Some(ni2) = next_non_comment(&kinds, ni + 1)
                        && let LineKind::Inst {
                            mnemonic: m3,
                            ops: ops3,
                        } = &kinds[ni2]
                        && matches!(m3.as_str(), "xor" | "and" | "or")
                        && ops3.len() == 1
                        && ops3[0].eq_ignore_ascii_case(&ops[0])
                    {
                        out.push(lines[i].clone());
                        out.push(format!("    {m3} {}", ops2[1]));
                        changed = true;
                        i = ni2 + 1;
                        continue;
                    }

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
