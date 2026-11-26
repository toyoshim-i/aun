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

    #[test]
    fn promotes_lifo_nested_hl_and_af_spills_to_push_pop() {
        let input = "\
foo:
    ld (0xC010), hl
    ld (0xC012), a
    ld a, (0xD000)
    or a
    ld (0xD001), a
    ld a, (0xC012)
    ld (0xD002), a
    call bar
    ld de, (0xC010)
    ret
";
        let mut priv_ranges = HashMap::new();
        priv_ranges.insert("foo".to_string(), vec![(0xC010, 0xC014)]);
        let mut sb_ranges = HashMap::new();
        sb_ranges.insert("foo".to_string(), (0xC010, 0xC014));

        // Force `BC` unavailable by referencing `c` in `foo` so stack promotion triggers!
        let input_with_c = input.replace("    ld a, (0xD000)\n", "    ld c, a\n    ld a, (0xD000)\n");
        let got = optimize_asm_with_sb(&input_with_c, &priv_ranges, &sb_ranges);
        assert!(got.contains("push hl"), "outer 16-bit spill should use push hl:\n{got}");
        assert!(got.contains("pop de"), "outer 16-bit fill should use pop de:\n{got}");
        assert!(got.contains("push af"), "inner 8-bit spill should use push af:\n{got}");
        assert!(got.contains("pop af"), "inner 8-bit fill should use pop af:\n{got}");
        assert!(!got.contains("0xC010"), "scratch 0xC010 eliminated:\n{got}");
        assert!(!got.contains("0xC012"), "scratch 0xC012 eliminated:\n{got}");
    }

    #[test]
    fn does_not_promote_crossing_spills_or_flag_sensitive_pop_af() {
        // 1. Flag-sensitive `pop af` before `jr z`: must keep `ld (0xC012), a` / `ld a, (0xC012)`!
        let flag_sensitive = "\
bar:
    ld (0xC012), a
    cp 0x05
    ld a, (0xC012)
    jr z, bar_bb1
bar_bb1:
    ret
";
        let mut sb_ranges = HashMap::new();
        sb_ranges.insert("bar".to_string(), (0xC010, 0xC016));
        let got1 = optimize_asm_with_sb(flag_sensitive, &HashMap::new(), &sb_ranges);
        assert!(!got1.contains("pop af"), "must not corrupt flags before jr z:\n{got1}");

        // 2. Crossing intervals: `(s1=A, l1=A)` crosses `(s2=B, l2=B)` -> at most one promoted!
        let crossing = "\
baz:
    ld c, a
    ld (0xC010), hl
    ld hl, (0xD000)
    ld (0xC012), hl
    ld hl, (0xC010)
    ld (0xD002), hl
    ld hl, (0xC012)
    ret
";
        sb_ranges.insert("baz".to_string(), (0xC010, 0xC016));
        let got2 = optimize_asm_with_sb(crossing, &HashMap::new(), &sb_ranges);
        assert_eq!(
            got2.matches("push ").count(),
            1,
            "crossing intervals must promote at most one spill:\n{got2}"
        );
    }

    #[test]
    fn does_not_promote_bc_spill_across_shared_sb_write_or_non_hl_read() {
        let mut priv_ranges = HashMap::new();
        priv_ranges.insert("f".to_string(), vec![(0xC100, 0xC102)]);
        let mut sb_ranges = HashMap::new();
        sb_ranges.insert("f".to_string(), (0xC100, 0xC102));

        let non_hl_lines: Vec<String> = vec![
            "f:".into(),
            "    ld (0xC100), hl".into(),
            "    ld hl, (0xD000)".into(),
            "    ld de, (0xC100)".into(),
            "    ld hl, (0xC100)".into(),
            "    ret".into(),
        ];
        let (out1, ch1) = super::spill::promote_single_block_spills_to_bc(
            &non_hl_lines,
            &priv_ranges,
            &sb_ranges,
            &HashSet::new(),
        );
        assert!(
            !ch1 && out1 == non_hl_lines,
            "intervening non-hl read must prevent BC promotion:\n{}",
            out1.join("\n")
        );

        let overwrite_lines: Vec<String> = vec![
            "f:".into(),
            "    ld (0xC100), hl".into(),
            "    ld (0xC100), de".into(),
            "    ld hl, (0xC100)".into(),
            "    ret".into(),
        ];
        let (out2, ch2) = super::spill::promote_single_block_spills_to_bc(
            &overwrite_lines,
            &priv_ranges,
            &sb_ranges,
            &HashSet::new(),
        );
        assert!(
            !ch2 && out2 == overwrite_lines,
            "intervening write to same sb address must prevent BC promotion:\n{}",
            out2.join("\n")
        );
    }

    #[test]
    fn phase_c1a_eliminates_or_a_folds_ram_inc_and_ix_zero_runs() {
        // 1. Redundant `or a` after `and`/`or`/`xor` and Z-only after `sub`/`inc a`, but NOT before `jp c`!
        let ora_src = "\
f1:
    and 0x01
    or a
    jr z, f1_bb1
f1_bb1:
    sub 0x03
    or a
    jp nz, f1_bb2
f1_bb2:
    sub 0x03
    or a
    jp c, f1_bb3
f1_bb3:
    ret
";
        let got_ora = optimize_asm(ora_src);
        assert_eq!(got_ora.matches("or a").count(), 1, "only `sub; or a; jp c` keeps `or a`:\n{got_ora}");

        // 2. RAM inc/dec via `HL` only when both `A` and `HL` are dead afterward.
        let ram_inc = "\
f2:
    ld a, (0xC100)
    inc a
    ld (0xC100), a
    ld a, (0xC200)
    ld hl, 0xC300
    ld (hl), a
    ld a, (0xC100)
    dec a
    ld (0xC100), a
    ret
";
        let got_inc = optimize_asm(ram_inc);
        assert!(got_inc.contains("ld hl, 0xC100\n    inc (hl)"), "dead A & HL should fold:\n{got_inc}");
        assert!(got_inc.contains("ld a, (0xC100)\n    dec a"), "live A/HL at ret must not fold:\n{got_inc}");

        // 3. Zero-run across `ld (ix+d), imm` when `xor a` / `ld a, imm` overwrites A afterward,
        //    plus direct `ld (ix+d), 0` -> `ld (ix+d), a` while `a_is_zero`.
        let zero_run = "\
f3:
    ld (ix + 8), 0
    ld (ix + 9), 12
    ld (ix + 10), 0
    xor a
    ld (0xC000), a
    ld (ix + 11), 0
    ld a, 1
    or a
    ret
";
        let got_zero = optimize_asm(zero_run);
        assert!(!got_zero.contains(", 0"), "all IX zero stores should use `a`:\n{got_zero}");
        assert_eq!(got_zero.matches("xor a").count(), 1, "trailing redundant `xor a` should be dropped:\n{got_zero}");
    }

    #[test]
    fn phase_c1b_drops_dead_ld_pairs_and_preserves_dependent_loads() {
        let src = "\
f4:
    ld a, 0x30
    ld a, (0xC100)
    ld hl, 0x1000
    ld a, (hl)
    ld h, 0
    ld hl, (hl)
    ld c, a
    ld bc, 0x1234
    ret
";
        let got = optimize_asm(src);
        assert!(!got.contains("ld a, 0x30"), "dead `ld a, 0x30` should be dropped:\n{got}");
        assert!(got.contains("ld hl, 0x1000\n    ld a, (hl)"), "ld hl followed by ld a,(hl) must stay:\n{got}");
        assert!(got.contains("ld h, 0\n    ld hl, (hl)"), "ld hl,(hl) reads h so ld h, 0 must stay:\n{got}");
        assert!(!got.contains("ld c, a"), "ld bc, imm completely overwrites c:\n{got}");
    }

    #[test]
    fn phase_c2b_branch_aware_ram_inc_folds_when_both_arms_kill_hl_and_a() {
        let both_kill = "\
lp_fn:
    ld a, (0xC100)
    inc a
    ld (0xC100), a
    jr lp_fn_bb1
lp_fn_bb1:
    ld a, (0xC100)
    cp 4
    jr c, lp_fn_bb2
lp_fn_bb3:
    ld hl, 0x0000
    ret
lp_fn_bb2:
    ld hl, 0x1000
    ret
";
        let got_fold = optimize_asm(both_kill);
        assert!(
            got_fold.contains("ld hl, 0xC100\n    inc (hl)"),
            "when both branch arms overwrite HL (and A is overwritten at header), should fold:\n{got_fold}"
        );

        let one_reads_hl = "\
lp_fn:
    ld a, (0xC100)
    inc a
    ld (0xC100), a
    jr lp_fn_bb1
lp_fn_bb1:
    ld a, (0xC100)
    cp 4
    jr c, lp_fn_bb2
lp_fn_bb3:
    ld a, (hl)
    ret
lp_fn_bb2:
    ld hl, 0x1000
    ret
";
        let got_keep = optimize_asm(one_reads_hl);
        assert!(
            !got_keep.contains("inc (hl)") && got_keep.contains("ld a, (0xC100)\n    inc a"),
            "when either branch arm reads HL, must NOT clobber HL:\n{got_keep}"
        );
    }

    #[test]
    fn phase_c3a_tail_merges_jp_terminated_blocks() {
        let src = "\
tail_fn:
    jp tail_fn_bb1
tail_fn_bb1:
    ld a, (ix + 0)
    or a
    jr z, tail_fn_bb2
    push ix
    pop de
    ld a, c
    push bc
    call foo
    pop bc
    jp tail_fn_bb95
tail_fn_bb2:
    ld a, (ix + 1)
    or a
    jr z, tail_fn_bb3
    push ix
    pop de
    ld a, c
    push bc
    call foo
    pop bc
    jp tail_fn_bb95
tail_fn_bb3:
    xor a
    ret
tail_fn_bb95:
    ld a, 1
    ret
";
        let got = optimize_asm(src);
        assert_eq!(
            got.matches("call foo").count(),
            1,
            "jp-terminated identical suffixes should tail-merge into one shared suffix:\n{got}"
        );
    }

    #[test]
    fn cancels_double_ex_de_hl_and_tail_calls_across_bc_save() {
        let src = "\
tail_bc:
    ld e, (hl)
    inc hl
    ld d, (hl)
    ex de, hl
    ex de, hl
    add hl, de
    push bc
    call target_fn
    pop bc
    ret
keep_single_ex_and_cond:
    ex de, hl
    ld a, l
    or a
    jr z, keep_single_ex_and_cond_bb1
    push bc
    call other_fn
    pop bc
    ret nz
keep_single_ex_and_cond_bb1:
    ret
";
        let got = optimize_asm(src);
        assert!(
            !got.contains("ex de, hl\n    ex de, hl"),
            "double ex de, hl should be cancelled:\n{got}"
        );
        assert!(
            got.contains("    jp target_fn\n"),
            "push bc; call target_fn; pop bc; ret should become jp target_fn:\n{got}"
        );
        assert!(
            got.contains("call other_fn"),
            "conditional ret nz must NOT collapse push bc; call; pop bc:\n{got}"
        );
        assert_eq!(
            got.matches("ex de, hl").count(),
            1,
            "single ex de, hl in keep_single_ex_and_cond must remain:\n{got}"
        );
    }
