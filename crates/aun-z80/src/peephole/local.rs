use super::*;

pub(super) fn run_pass(lines: &[String]) -> (Vec<String>, bool) {
    let kinds: Vec<LineKind> = lines.iter().map(|l| classify_line(l)).collect();
    let label_pos = build_label_positions(&kinds);
    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut changed = false;
    let mut i = 0usize;
    let mut a_is_zero = false;

    while i < lines.len() {
        match &kinds[i] {
            LineKind::Inst { mnemonic, ops } => {
                // Rule 0: track whether `A == 0` in the current linear basic block and fold
                // `ld (ix + d), 0` -> `ld (ix + d), a` or drop redundant `ld a, 0` / `xor a`.
                if a_is_zero {
                    if mnemonic == "ld"
                        && ops.len() == 2
                        && parse_ix_disp(&ops[0]).is_some()
                        && is_zero_imm(&ops[1])
                    {
                        out.push(format!("    ld {}, a", ops[0]));
                        changed = true;
                        i += 1;
                        continue;
                    }
                    if mnemonic == "ld"
                        && ops.len() == 2
                        && ops[0].eq_ignore_ascii_case("a")
                        && is_zero_imm(&ops[1])
                    {
                        changed = true;
                        i += 1;
                        continue;
                    }
                    if mnemonic == "xor"
                        && ops.len() == 1
                        && ops[0].eq_ignore_ascii_case("a")
                        && flags_dead_after(&kinds, i + 1)
                    {
                        changed = true;
                        i += 1;
                        continue;
                    }
                }

                // Compute updated `a_is_zero` state for fallthrough past the instruction at `i`
                // (used whenever we keep the original instruction or transform cleanly).
                let next_a_is_zero = if (mnemonic == "xor"
                    && ops.len() == 1
                    && ops[0].eq_ignore_ascii_case("a"))
                    || (mnemonic == "ld"
                        && ops.len() == 2
                        && ops[0].eq_ignore_ascii_case("a")
                        && is_zero_imm(&ops[1]))
                {
                    true
                } else if !a_is_zero
                    || matches!(
                        mnemonic.as_str(),
                        "call" | "jp" | "jr" | "djnz" | "ret" | "reti" | "retn" | "rst"
                            | "halt" | "exx" | "ex"
                    )
                    || inst_writes_a_without_reading(mnemonic, ops)
                    || matches!(
                        mnemonic.as_str(),
                        "and" | "or" | "xor" | "sub" | "sbc" | "adc" | "add" | "neg" | "cpl"
                            | "rla" | "rra" | "rlca" | "rrca"
                    )
                    || (matches!(
                        mnemonic.as_str(),
                        "inc" | "dec" | "sla" | "sra" | "srl" | "rl" | "rr"
                    ) && ops.first().is_some_and(|o| o.eq_ignore_ascii_case("a")))
                {
                    false
                } else {
                    true
                };

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

                // Rule 2b: Redundant `or a` elimination:
                // - After `and`/`or`/`xor` (unless `daa` reads H): `Z/S/P/V/N/C` are already identical to `or a`.
                // - After `sub`/`add a`/`inc a`/`dec a`/`srl a`/`sla a`/`sra a`: drop `or a` when
                //   the immediately next instruction only reads Z/NZ (or flags are dead).
                if let Some(ni) = next_non_comment(&kinds, i + 1)
                    && let LineKind::Inst {
                        mnemonic: nm,
                        ops: nops,
                    } = &kinds[ni]
                    && nm == "or"
                    && nops.len() == 1
                    && nops[0].eq_ignore_ascii_case("a")
                {
                    let is_bitwise_a = matches!(mnemonic.as_str(), "and" | "or" | "xor");
                    let next_is_daa = next_non_comment(&kinds, ni + 1).is_some_and(|nn| {
                        matches!(&kinds[nn], LineKind::Inst { mnemonic: dm, .. } if dm == "daa")
                    });
                    let sets_z_from_a = (mnemonic == "sub")
                        || (mnemonic == "add"
                            && ops.first().is_some_and(|o| o.eq_ignore_ascii_case("a")))
                        || (matches!(
                            mnemonic.as_str(),
                            "inc" | "dec" | "srl" | "sla" | "sra"
                        ) && ops.first().is_some_and(|o| o.eq_ignore_ascii_case("a")));
                    let only_z_after = next_non_comment(&kinds, ni + 1).is_some_and(|nn| {
                        matches!(
                            &kinds[nn],
                            LineKind::Inst { mnemonic: jm, ops: jops }
                            if matches!(jm.as_str(), "jp" | "jr" | "call" | "ret")
                                && jops.first().is_some_and(|c| matches!(c.to_ascii_lowercase().as_str(), "z" | "nz"))
                        )
                    }) || flags_dead_after(&kinds, ni + 1);
                    if (is_bitwise_a && !next_is_daa) || (sets_z_from_a && only_z_after) {
                        out.push(lines[i].clone());
                        a_is_zero = next_a_is_zero;
                        changed = true;
                        i = ni + 1;
                        continue;
                    }
                }

                // Rule 3: `ld a, 0` -> `xor a` when flags are provably dead afterward
                if mnemonic == "ld"
                    && ops.len() == 2
                    && ops[0].eq_ignore_ascii_case("a")
                    && is_zero_imm(&ops[1])
                    && flags_dead_after(&kinds, i + 1)
                {
                    out.push("    xor a".to_string());
                    a_is_zero = true;
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

                // Rule 3d: `ld a, c/b; inc/dec a; ld c/b, a` -> `inc/dec c/b` (3B -> 1B!) when `a` is dead
                if mnemonic == "ld"
                    && ops.len() == 2
                    && ops[0].eq_ignore_ascii_case("a")
                    && (ops[1].eq_ignore_ascii_case("c") || ops[1].eq_ignore_ascii_case("b"))
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
                    && ops2[0].eq_ignore_ascii_case(&ops[1])
                    && ops2[1].eq_ignore_ascii_case("a")
                {
                    let reg = ops[1].to_ascii_lowercase();
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
                        out.push(format!("    {m1} {reg}"));
                        changed = true;
                        i = n2 + 1;
                        continue;
                    }
                }

                // Rule 3d2: `ld a, (addr); inc/dec a; ld (addr), a` -> `ld hl, addr; inc/dec (hl)` (7B -> 4B!)
                // when `A` and `HL` are provably overwritten before read (`inc/dec (hl)` sets identical flags).
                if mnemonic == "ld"
                    && ops.len() == 2
                    && ops[0].eq_ignore_ascii_case("a")
                    && let Some(addr) = parse_mem_addr(&ops[1])
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
                    && parse_mem_addr(&ops2[0]) == Some(addr)
                    && ops2[1].eq_ignore_ascii_case("a")
                    && reg_a_dead_after(&kinds, &label_pos, n2 + 1)
                    && reg_hl_dead_after(&kinds, &label_pos, n2 + 1)
                {
                    out.push(format!("    ld hl, 0x{addr:04X}"));
                    out.push(format!("    {m1} (hl)"));
                    a_is_zero = false;
                    changed = true;
                    i = n2 + 1;
                    continue;
                }

                // Rule 3e: zero-init runs of `ld (ix + d), 0` (spanning A/flag-preserving `ld (ix+d), imm`) -> `xor a; ld (ix + d), a ...`
                if mnemonic == "ld"
                    && ops.len() == 2
                    && parse_ix_disp(&ops[0]).is_some()
                    && is_zero_imm(&ops[1])
                {
                    let mut span_idxs = vec![i];
                    let mut zero_count = 1usize;
                    let mut last_zero_end = span_idxs.len();
                    let mut cur = i;
                    while let Some(ni) = next_non_comment(&kinds, cur + 1) {
                        if let LineKind::Inst {
                            mnemonic: nm,
                            ops: nops,
                        } = &kinds[ni]
                            && nm == "ld"
                            && nops.len() == 2
                            && parse_ix_disp(&nops[0]).is_some()
                            && !nops[1].eq_ignore_ascii_case("a")
                        {
                            span_idxs.push(ni);
                            if is_zero_imm(&nops[1]) {
                                zero_count += 1;
                                last_zero_end = span_idxs.len();
                            }
                            cur = ni;
                        } else {
                            break;
                        }
                    }
                    span_idxs.truncate(last_zero_end);
                    if zero_count >= 2 {
                        let last_idx = *span_idxs.last().unwrap();
                        let a_dead = reg_a_dead_after(&kinds, &label_pos, last_idx + 1)
                            || next_non_comment(&kinds, last_idx + 1).is_some_and(|n3| {
                                matches!(&kinds[n3], LineKind::Label(lbl) if is_bb_label(lbl))
                            });
                        if a_dead && flags_dead_after(&kinds, last_idx + 1) {
                            out.push("    xor a".to_string());
                            for &ri in &span_idxs {
                                if let LineKind::Inst { ops: rops, .. } = &kinds[ri] {
                                    if is_zero_imm(&rops[1]) {
                                        out.push(format!("    ld {}, a", rops[0]));
                                    } else {
                                        out.push(lines[ri].clone());
                                    }
                                }
                            }
                            a_is_zero = true;
                            changed = true;
                            i = last_idx + 1;
                            continue;
                        }
                    }
                }

                // Look ahead to the next non-comment line for 2-instruction patterns
                if let Some(ni) = next_non_comment(&kinds, i + 1) {
                    // Rule 1d: dead consecutive `ld dst1, src1; ld dst2, src2` where `dst2`
                    // completely overwrites `dst1` without `src2` reading any byte of `dst1`.
                    if mnemonic == "ld"
                        && ops.len() == 2
                        && let LineKind::Inst {
                            mnemonic: m2,
                            ops: ops2,
                        } = &kinds[ni]
                        && m2 == "ld"
                        && ops2.len() == 2
                    {
                        let d1 = ops[0].to_ascii_lowercase();
                        let d2 = ops2[0].to_ascii_lowercase();
                        let s2 = ops2[1].trim().to_ascii_lowercase();
                        let d1_is_reg = matches!(
                            d1.as_str(),
                            "a" | "b" | "c" | "d" | "e" | "h" | "l" | "hl" | "de" | "bc"
                        );
                        let d2_overwrites_d1 = d1 == d2
                            || (d2 == "hl" && matches!(d1.as_str(), "h" | "l"))
                            || (d2 == "de" && matches!(d1.as_str(), "d" | "e"))
                            || (d2 == "bc" && matches!(d1.as_str(), "b" | "c"));
                        let s2_reads_d1 = match d1.as_str() {
                            "a" => s2 == "a",
                            "b" => matches!(s2.as_str(), "b" | "bc" | "(bc)"),
                            "c" => matches!(s2.as_str(), "c" | "bc" | "(bc)"),
                            "d" => matches!(s2.as_str(), "d" | "de" | "(de)"),
                            "e" => matches!(s2.as_str(), "e" | "de" | "(de)"),
                            "h" => matches!(s2.as_str(), "h" | "hl" | "(hl)"),
                            "l" => matches!(s2.as_str(), "l" | "hl" | "(hl)"),
                            "hl" => matches!(s2.as_str(), "h" | "l" | "hl" | "(hl)"),
                            "de" => matches!(s2.as_str(), "d" | "e" | "de" | "(de)"),
                            "bc" => matches!(s2.as_str(), "b" | "c" | "bc" | "(bc)"),
                            _ => true,
                        };
                        if d1_is_reg && d2_overwrites_d1 && !s2_reads_d1 {
                            for line in &lines[i + 1..ni] {
                                out.push(line.clone());
                            }
                            changed = true;
                            i = ni;
                            continue;
                        }
                    }

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
                a_is_zero = next_a_is_zero;
                i += 1;
            }
            LineKind::Label(_) | LineKind::Directive => {
                out.push(lines[i].clone());
                a_is_zero = false;
                i += 1;
            }
            LineKind::EmptyOrComment => {
                out.push(lines[i].clone());
                i += 1;
            }
        }
    }

    (out, changed)
}
