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

    #[test]
    fn foundation_phase1_a_eq_reg_ret_cc_dedup_global_dse_and_local() {
        // 1. `a_eq_reg` cache across `or a` & branches, plus `ret cc`, block dedup, and `xor a; xor a` / `push af; ld a, (ix+d); ld e, a; pop af`
        let src = "\
demo:
    ld c, a
demo_bb0:
    ld a, c
    or a
    jr z, demo_bb1
    push af
    ld a, (ix + 17)
    ld e, a
    pop af
    cp 3
    jr c, demo_bb2
    xor a
    xor a
    ret
demo_bb1:
    ret
demo_bb2:
    ld a, 1
    jp demo_bb9
demo_bb3:
    ld a, 1
    jp demo_bb9
demo_bb9:
    ret
";
        let got = optimize_asm(src);
        assert!(
            !got.contains("ld a, c"),
            "a_eq_reg should eliminate redundant `ld a, c` after `ld c, a`:\n{got}"
        );
        assert!(
            got.contains("ld e, (ix + 17)") && !got.contains("push af"),
            "push af; ld a, (ix + 17); ld e, a; pop af should become ld e, (ix + 17):\n{got}"
        );
        assert!(
            got.contains("ret z") && !got.contains("demo_bb1:"),
            "jr z targeting pure ret block should become ret z and delete demo_bb1:\n{got}"
        );
        assert!(
            !got.contains("xor a\n    xor a"),
            "double xor a should collapse to single xor a:\n{got}"
        );

        // 2. Block deduplication for identical ret/jp-terminated blocks (no fallthrough, no jr ref)
        let dup_src = "\
dedup:
    or a
    jp z, dedup_bb1
    ld a, 2
    jp dedup_bb2
dedup_bb1:
    ld a, 5
    ret
dedup_bb2:
    ld a, 5
    ret
";
        let got_dup = optimize_asm(dup_src);
        assert_eq!(
            got_dup.matches("ld a, 5").count(),
            1,
            "identical blocks dedup_bb1 and dedup_bb2 should deduplicate:\n{got_dup}"
        );

        // 3. CFG Global DSE drops dead private store overwritten along all paths (even if read in a separate block!)
        let dse_src = "\
gfn:
    ld (0xC010), a
gfn_bb1:
    ld a, 7
    ld (0xC010), a
    ld a, (0xC010)
    ret
";
        let mut ranges = HashMap::new();
        ranges.insert("gfn".to_string(), vec![(0xC010, 0xC020)]);
        let got_dse = optimize_asm_per_fn(dse_src, &ranges);
        assert_eq!(
            got_dse.matches("ld (0xC010), a").count(),
            0,
            "dead initial store overwritten in gfn_bb1 (and subsequent reg-cached store) should both be eliminated:\n{got_dse}"
        );
        // Verify when read later on a branch vs overwritten on another branch, live store is KEPT:
        let dse_live_src = "\
gfn2:
    ld (0xC010), a
    ld a, b
    or a
    jr z, gfn2_bb1
    ld a, 7
    ld (0xC010), a
gfn2_bb1:
    ld a, (0xC010)
    ret
";
        let mut ranges2 = HashMap::new();
        ranges2.insert("gfn2".to_string(), vec![(0xC010, 0xC020)]);
        let got_live = optimize_asm_per_fn(dse_live_src, &ranges2);
        assert!(
            got_live.contains("ld (0xC010), a"),
            "store live along fallthrough/jr edge must NOT be deleted by Global DSE:\n{got_live}"
        );
    }

    #[test]
    fn preserves_live_stores_across_mul_and_cmp32_intra_fn_labels() {
        let src = "\
foo:
    ld (0xC010), a
    ld d, a
    xor a
    ld b, 8
foo_mul8_lp_0:
    add a, a
    sla d
    jr nc, foo_mul8_sk_1
    add a, e
foo_mul8_sk_1:
    djnz foo_mul8_lp_0
    ld (0xC012), hl
    or a
    sbc hl, de
    jr nz, foo_cmp32_ne_0
    nop
foo_cmp32_ne_0:
    ld hl, (0xC012)
    ld a, (0xC010)
    add a, l
    ret
foo_2:
    xor a
    ret
";
        let mut ranges = HashMap::new();
        ranges.insert("foo".to_string(), vec![(0xC010, 0xC020)]);
        ranges.insert("foo_2".to_string(), Vec::new());
        let got = optimize_asm_per_fn(src, &ranges);
        assert!(
            got.contains("ld (0xC010), a"),
            "store before _mul8_ label read after label must NOT be deleted:\n{got}"
        );
        assert!(
            got.contains("ld (0xC012), hl"),
            "store before _cmp32_ label read after label must NOT be deleted:\n{got}"
        );
    }

    #[test]
    fn preserves_tail_call_across_inline_asm_and_unbalanced_stack() {
        // 1. Inline asm markers prevent tail-call rewrite even when call sits at end of block
        let inline_src = "\
foo:
; @aun:asm_begin
    push hl
    call _callee
; @aun:asm_end
    ret
";
        let got_inline = optimize_asm(inline_src);
        assert!(
            got_inline.contains("call _callee") && got_inline.contains("ret"),
            "call inside inline asm must NOT rewrite to jp:\n{got_inline}"
        );
        assert!(
            !got_inline.contains("jp _callee"),
            "inline asm call must NOT become jp:\n{got_inline}"
        );

        // 2. Unbalanced SP (`push hl; call _callee; ret`) even without markers must NOT rewrite
        let unbal_src = "\
bar:
    push hl
    call _callee
    ret
";
        let got_unbal = optimize_asm(unbal_src);
        assert!(
            got_unbal.contains("call _callee") && got_unbal.contains("ret"),
            "call with non-zero stack depth must NOT rewrite to jp:\n{got_unbal}"
        );
        assert!(!got_unbal.contains("jp _callee"), "got:\n{got_unbal}");

        // 3. Normal balanced tail-call still rewrites to jp
        let normal_src = "\
baz:
    call foo
    ret
";
        let got_normal = optimize_asm(normal_src);
        assert!(
            got_normal.contains("jp foo") && !got_normal.contains("call foo"),
            "normal balanced tail call should rewrite to jp:\n{got_normal}"
        );
    }

    #[test]
    fn preserves_volatile_reads_and_writes() {
        let src = "\
poll:
; @aun:volatile_begin
    ld a, (0x0099)
; @aun:volatile_end
    ld (0xC010), a
; @aun:volatile_begin
    ld a, (0x0099)
; @aun:volatile_end
    ld (0xC011), a
; @aun:volatile_begin
    xor a
    ld (0x0099), a
; @aun:volatile_end
; @aun:volatile_begin
    xor a
    ld (0x0099), a
; @aun:volatile_end
    ret
";
        let got = optimize_asm(src);
        assert_eq!(
            got.matches("ld a, (0x0099)").count(),
            2,
            "consecutive volatile reads must NOT be eliminated:\n{got}"
        );
        assert_eq!(
            got.matches("ld (0x0099), a").count(),
            2,
            "consecutive volatile writes must NOT be eliminated:\n{got}"
        );
    }

    #[test]
    fn strips_unreferenced_bb_labels_and_inverts_branch_over_jump() {
        let src = "\
foo:
    jp c, foo_bb1
foo_bb10:
    jp foo_bb3
foo_bb1:
    ld bc, #0x0FFF
foo_bb3:
    inc a
    ret
foo_bb8:
    .dw foo_bb9
foo_bb9:
    ret
";
        let got = optimize_asm(src);
        assert!(
            got.contains("jp nc, foo_bb3"),
            "expected inverted branch `jp nc, foo_bb3`:\n{got}"
        );
        assert!(
            !got.contains("foo_bb10:") && !got.contains("foo_bb1:") && !got.contains("foo_bb8:"),
            "unreferenced `_bb` labels should be stripped:\n{got}"
        );
        assert!(
            got.contains("foo_bb3:") && got.contains("foo_bb9:"),
            "referenced `_bb` labels (`foo_bb3` jump target, `foo_bb9` `.dw` target) must NOT be stripped:\n{got}"
        );
    }

    #[test]
    fn reg_cache_preserves_private_locals_across_ix_store_push_and_inc() {
        let asm = "\
foo:
    ld a, (0xC010)
    ld hl, (0xC020)
    ld (ix + 2), 0x05
    push bc
    inc c
    ld a, (0xC010)
    ld hl, (0xC020)
    ret
";
        let mut priv_ranges = HashMap::new();
        priv_ranges.insert("foo".to_string(), vec![(0xC010u16, 0xC012u16)]);
        let got = optimize_asm_full(asm, &priv_ranges, &HashMap::new(), &HashSet::new());
        // Private local 0xC010 survives `ld (ix + 2)`, `push bc`, and `inc c` -> 2nd load eliminated.
        assert_eq!(
            got.matches("(0xC010)").count(),
            1,
            "private local load should remain cached across ld (ix+2), push bc, inc c:\n{got}"
        );
        // Global/untracked 0xC020 must be invalidated by `ld (ix + 2)` -> 2nd load preserved.
        assert_eq!(
            got.matches("(0xC020)").count(),
            2,
            "non-private load must be invalidated on ld (ix+2):\n{got}"
        );
    }

    #[test]
    fn dead_scratch_store_preserves_store_across_unlabeled_cond_branch_fallthrough() {
        let asm = "\
foo:
    ld hl, #0x0010
    ld (__aun_ram + 10), hl
foo_bb2:
    ld hl, (__aun_ram + 10)
    inc hl
    ld (__aun_ram + 10), hl
    dec c
    jp nz, foo_bb2
    ld hl, #0x1000
    ld (__aun_ram + 10), hl
foo_bb4:
    ld hl, (__aun_ram + 10)
    ret
";
        let mut priv_ranges = HashMap::new();
        priv_ranges.insert("foo".to_string(), vec![(0xC000u16, 0xC020u16)]);
        let got = optimize_asm_full(asm, &priv_ranges, &HashMap::new(), &HashSet::new());
        // The store `ld (__aun_ram + 10), hl` before `jp nz, foo_bb2` is read on the backedge
        // inside `foo_bb2`, even though there is no label between `jp nz, foo_bb2` and the
        // subsequent fallthrough overwrite `ld (__aun_ram + 10), hl`.
        assert_eq!(
            got.matches("ld (__aun_ram + 10), hl").count(),
            3,
            "store before conditional backedge must not be deleted across fallthrough overwrite:\n{got}"
        );
    }

    #[test]
    fn tail_merge_uses_non_bb_label_and_preserves_sb_scratch_store() {
        let asm = "\
tfn:
    jp tfn_bb1
tfn_bb1:
    or a
    jr z, tfn_bb2
    ld hl, 0xC100
    ld (0xC010), hl
    call helper1
    ld a, (0xC010)
    call foo
    ld a, 1
    ret
tfn_bb2:
    ld hl, 0xC200
    ld (0xC010), hl
    call helper2
    ld a, (0xC010)
    call foo
    ld a, 1
    ret
";
        let mut priv_ranges = HashMap::new();
        priv_ranges.insert("tfn".to_string(), vec![(0xC000u16, 0xC020u16)]);
        let mut sb_ranges = HashMap::new();
        sb_ranges.insert("tfn".to_string(), (0xC010u16, 0xC012u16));
        let got = optimize_asm_full(asm, &priv_ranges, &sb_ranges, &HashSet::new());
        assert!(
            got.contains("tfn_tail9000:"),
            "expected tail merge to synthesize _tail label instead of _bb:\n{got}"
        );
        assert_eq!(
            got.matches("ld (0xC010), hl").count(),
            2,
            "both single-block scratch writes before the merged tail must be preserved:\n{got}"
        );
    }

    #[test]
    fn sequential_hl_pointer_walk_folds_inc_dec_and_add_hl_de() {
        let asm = "\
walk:
    ld hl, (0xC000)
    inc hl
    ld a, (hl)
    ld (0xD000), a
    ld hl, (0xC000)
    inc hl
    inc hl
    ld a, (hl)
    ld (0xD001), a
    ld hl, (0xC000)
    ld (0xD002), hl
    ld hl, (0xC000)
    ld de, 3
    add hl, de
    ld (0xD004), hl
    ld de, 0x1111
    xor a
    ret
";
        let got = optimize_asm(asm);
        assert_eq!(
            got.matches("ld hl, (0xC000)").count(),
            1,
            "all subsequent reloads of (0xC000) should fold into inc/dec hl:\n{got}"
        );
        assert!(
            !got.contains("ld de, 3") && !got.contains("add hl, de"),
            "Pattern B should fold `ld hl, (0xC000); ld de, 3; add hl, de` into inc hl:\n{got}"
        );
        assert!(
            got.contains("dec hl\n    dec hl"),
            "walking back from +2 to 0 should emit two `dec hl`:\n{got}"
        );

        // Private base pointer survives indirect `ld (hl), a` store!
        let priv_asm = "\
walk_priv:
    ld hl, (0xC000)
    inc hl
    ld (hl), a
    ld hl, (0xC000)
    ld (0xD000), hl
    ret
";
        let mut priv_ranges = HashMap::new();
        priv_ranges.insert("walk_priv".to_string(), vec![(0xC000, 0xC002)]);
        let got_priv = optimize_asm_per_fn(priv_asm, &priv_ranges);
        assert_eq!(
            got_priv.matches("ld hl, (0xC000)").count(),
            1,
            "private base pointer should fold across `ld (hl), a`:\n{got_priv}"
        );
        assert!(
            got_priv.contains("dec hl"),
            "private base walk back from +1 to 0 should emit `dec hl`:\n{got_priv}"
        );
    }

    #[test]
    fn sequential_hl_pointer_walk_negative_cases() {
        // 1. DE live after Pattern B (`ld a, e`) -> must NOT fold
        let de_live = "\
neg_de:
    ld hl, (0xC000)
    inc hl
    ld a, (hl)
    ld (0xD000), a
    ld hl, (0xC000)
    ld de, 2
    add hl, de
    ld (0xD002), hl
    ld a, e
    xor a
    ret
";
        let got_de = optimize_asm(de_live);
        assert!(
            got_de.contains("ld de, 2") && got_de.contains("add hl, de"),
            "Pattern B must NOT fold when DE is live afterward:\n{got_de}"
        );

        // 2. Carry flag live after Pattern B (`adc a, 0`) -> must NOT fold
        let carry_live = "\
neg_carry:
    ld hl, (0xC000)
    inc hl
    ld a, (hl)
    ld (0xD000), a
    ld hl, (0xC000)
    ld de, 2
    add hl, de
    ld (0xD002), hl
    ld de, 0x0000
    adc a, 0
    ret
";
        let got_carry = optimize_asm(carry_live);
        assert!(
            got_carry.contains("ld de, 2") && got_carry.contains("add hl, de"),
            "Pattern B must NOT fold when carry flag is live afterward:\n{got_carry}"
        );

        // 3. Base memory overwritten -> hl_base_off cleared, must reload
        let base_written = "\
neg_base:
    ld hl, (0xC000)
    inc hl
    ld a, (hl)
    ld (0xD000), a
    xor a
    ld (0xC000), a
    ld hl, (0xC000)
    ld (0xD002), hl
    ret
";
        let got_base = optimize_asm(base_written);
        assert_eq!(
            got_base.matches("ld hl, (0xC000)").count(),
            2,
            "base write must invalidate hl_base_off so second load is preserved:\n{got_base}"
        );

        // 4. H/L modified directly -> hl_base_off cleared, must reload
        let hl_clobbered = "\
neg_hl:
    ld hl, (0xC000)
    inc hl
    ld a, (hl)
    ld (0xD000), a
    ld h, 0
    ld (0xD001), hl
    ld hl, (0xC000)
    ld (0xD003), hl
    ret
";
        let got_hl = optimize_asm(hl_clobbered);
        assert_eq!(
            got_hl.matches("ld hl, (0xC000)").count(),
            2,
            "modifying H must invalidate hl_base_off so second load is preserved:\n{got_hl}"
        );
    }

    #[test]
    fn promotes_single_block_spills_to_de_and_e_d() {
        let asm = "\
sp_de:
    ld (0xC010), hl
    ld hl, (0xD000)
    ld (0xD002), hl
    ld de, (0xC010)
    ld (0xD004), de
    ld hl, (0xD006)
    ld (0xC012), hl
    ld hl, (0xD008)
    ld (0xD00A), hl
    ld hl, (0xC012)
    ld (0xD00C), hl
    ld de, 0x1234
    ld a, (0xD00E)
    ld (0xC014), a
    ld a, (0xD010)
    ld (0xD012), a
    ld a, (0xC014)
    ld (0xD014), a
    ld e, 0x55
    ld a, (0xD016)
    ld (0xC015), a
    ld e, 0x01
    ld a, (0xD017)
    ld (0xD018), a
    ld a, (0xC015)
    ld (0xD01A), a
    ld d, 0x66
    ret
";
        let mut priv_ranges = HashMap::new();
        priv_ranges.insert("sp_de".to_string(), vec![(0xC010, 0xC020)]);
        let mut sb_ranges = HashMap::new();
        sb_ranges.insert("sp_de".to_string(), (0xC010, 0xC020));
        let got = optimize_asm_full(asm, &priv_ranges, &sb_ranges, &HashSet::new());
        assert!(!got.contains("0xC010"), "16-bit DE spill should promote:\n{got}");
        assert!(!got.contains("0xC012"), "16-bit HL spill via ex de, hl should promote:\n{got}");
        assert!(got.contains("ex de, hl"), "expected ex de, hl for 0xC012:\n{got}");
        assert!(!got.contains("0xC014"), "8-bit E spill should promote:\n{got}");
        assert!(got.contains("ld e, a") && got.contains("ld a, e"), "expected E spill:\n{got}");
        assert!(!got.contains("0xC015"), "8-bit D spill should promote when E is busy:\n{got}");
        assert!(got.contains("ld d, a") && got.contains("ld a, d"), "expected D spill:\n{got}");
    }

    #[test]
    fn promotes_per_region_bc_spill_even_when_other_block_uses_c() {
        let asm = "\
bc_fn:
    ld c, a
    ld (0xD000), a
    jp bc_fn_bb1
bc_fn_bb1:
    ld (0xC010), hl
    ld hl, (0xD006)
    ld (0xD008), hl
    ld de, 0x0001
    ld (0xD002), de
    ld hl, (0xC010)
    ld (0xD004), hl
    ld bc, 0x0000
    ret
";
        let mut priv_ranges = HashMap::new();
        priv_ranges.insert("bc_fn".to_string(), vec![(0xC010, 0xC012)]);
        let mut sb_ranges = HashMap::new();
        sb_ranges.insert("bc_fn".to_string(), (0xC010, 0xC012));
        let got = optimize_asm_full(asm, &priv_ranges, &sb_ranges, &HashSet::new());
        assert!(
            !got.contains("0xC010"),
            "BC spill in bc_fn_bb1 should promote even though entry block uses c:\n{got}"
        );
        assert!(
            got.contains("ld b, h") && got.contains("ld c, l") && got.contains("ld h, b"),
            "expected BC spill promotion in bc_fn_bb1:\n{got}"
        );
    }

    #[test]
    fn does_not_promote_de_or_bc_spills_when_target_live_after() {
        let mut sb_ranges = HashMap::new();
        sb_ranges.insert("neg".to_string(), (0xC010, 0xC014));
        let mut priv_ranges = HashMap::new();
        priv_ranges.insert("neg".to_string(), vec![(0xC010, 0xC014)]);

        // 1. DE live after `ld hl, (0xC010)` -> must NOT promote to ex de, hl
        let de_live: Vec<String> = vec![
            "neg:".into(),
            "    ld (0xC010), hl".into(),
            "    ld hl, (0xD000)".into(),
            "    ld hl, (0xC010)".into(),
            "    ld a, e".into(),
            "    ret".into(),
        ];
        let empty_priv = HashMap::new();
        let (out1, ch1) =
            super::spill::promote_single_block_spills_to_de(&de_live, &empty_priv, &sb_ranges);
        assert!(!ch1 && out1 == de_live, "DE live after must block DE promotion:\n{}", out1.join("\n"));

        // 2. Both E and D live after 8-bit `ld a, (0xC010)` -> must NOT promote
        let ed_live: Vec<String> = vec![
            "neg:".into(),
            "    ld (0xC010), a".into(),
            "    ld a, (0xD000)".into(),
            "    ld a, (0xC010)".into(),
            "    add a, e".into(),
            "    add a, d".into(),
            "    ret".into(),
        ];
        let (out2, ch2) =
            super::spill::promote_single_block_spills_to_de(&ed_live, &empty_priv, &sb_ranges);
        assert!(!ch2 && out2 == ed_live, "E and D live after must block 8-bit spill:\n{}", out2.join("\n"));

        // 3. C live after `ld hl, (0xC010)` -> must NOT promote to BC
        let bc_live: Vec<String> = vec![
            "neg:".into(),
            "    ld (0xC010), hl".into(),
            "    ld de, 0x0001".into(),
            "    ld hl, (0xC010)".into(),
            "    ld a, c".into(),
            "    ret".into(),
        ];
        let (out3, ch3) = super::spill::promote_single_block_spills_to_bc(
            &bc_live,
            &priv_ranges,
            &sb_ranges,
            &HashSet::new(),
        );
        assert!(!ch3 && out3 == bc_live, "C live after must block BC promotion:\n{}", out3.join("\n"));
    }

    #[test]
    fn issue42_eliminates_redundant_push_pop_and_dead_const_spills_with_3d2() {
        let asm = "\
pack_fn:
    ld hl, 0x0001
    ld (0xC010), hl
    ld hl, 0x0002
    ld (0xC012), hl
    ld a, (0xC020)
    inc a
    ld (0xC020), a
    ld a, (0xC014)
    ld (0xC015), a
    ld e, a
    ld hl, (0xC030)
    push hl
    ld a, (ix + 1)
    or e
    ld e, a
    pop hl
    push hl
    ld a, (ix + 2)
    ld l, a
    ld h, 0
    add hl, hl
    ld d, h
    ld e, l
    pop hl
    ret
";
        let mut priv_ranges = HashMap::new();
        priv_ranges.insert(
            "pack_fn".to_string(),
            vec![(0xC010, 0xC018), (0xC020, 0xC021)],
        );
        let got = optimize_asm_per_fn(asm, &priv_ranges);
        assert!(!got.contains("0xC010"), "dead const spill 0xC010 must be eliminated:\n{got}");
        assert!(!got.contains("0xC012"), "dead const spill 0xC012 must be eliminated:\n{got}");
        assert!(!got.contains("0x0001"), "dead ld hl, 0x0001 must be eliminated:\n{got}");
        assert!(!got.contains("0x0002"), "dead ld hl, 0x0002 must be eliminated:\n{got}");
        assert!(!got.contains("0xC015"), "dead scratch store 0xC015 must be eliminated:\n{got}");
        assert!(!got.contains("push hl"), "unnecessary push hl must be eliminated:\n{got}");
        assert!(!got.contains("pop hl"), "unnecessary pop hl must be eliminated:\n{got}");
        assert!(got.contains("ex de, hl"), "push hl..ld d,h;ld e,l;pop hl must rewrite to ex de, hl:\n{got}");
        assert!(got.contains("inc (hl)"), "Rule 3d2 inc (hl) must still work:\n{got}");
    }

    #[test]
    fn coalesces_duplicate_private_spills_eliminates_ix_reloads_and_promotes_ex_de_hl_across_syn_lbl() {
        // 1. Duplicate private RAM spill coalescing + `ex de, hl` across `_sh16_dn_` synthetic label.
        let mut priv_ranges = HashMap::new();
        priv_ranges.insert(
            "fn_a".to_string(),
            vec![(0xC000u16, 0xC020u16)],
        );
        let dup_asm: Vec<String> = vec![
            "fn_a:".into(),
            "fn_a_bb0:".into(),
            "    ld e, (ix + 4)".into(),
            "    ld d, (ix + 5)".into(),
            "    ld (0xC002), de".into(),
            "    ld (0xC004), de".into(),
            "    ld (0xC006), hl".into(),
            "_sh16_dn_2:".into(),
            "    srl h".into(),
            "    rr l".into(),
            "    ld de, (0xC006)".into(),
            "    add hl, de".into(),
            "    ld hl, (0xC004)".into(),
            "    add hl, hl".into(),
            "    ret".into(),
        ];
        let mut sb_ranges = HashMap::new();
        sb_ranges.insert("fn_a".to_string(), (0xC000u16, 0xC020u16));
        let opt = super::optimize_asm_full(
            &dup_asm.join("\n"),
            &priv_ranges,
            &sb_ranges,
            &HashSet::new(),
        );
        assert!(
            !opt.contains("ld (0xC004), de") && !opt.contains("0xC004"),
            "duplicate private spill slot 0xC004 should be coalesced to 0xC002:\n{opt}"
        );
        assert!(
            !opt.contains("ld (0xC006), hl") && !opt.contains("ld de, (0xC006)"),
            "single-block spill across _sh16_dn_ label should promote cleanly without 0xC006:\n{opt}"
        );

        // 2. Redundant IX reloads into HL, DE, and A (and invalidation after `ld (ix + 4), a`).
        let ix_asm: Vec<String> = vec![
            "fn_ix:".into(),
            "fn_ix_bb0:".into(),
            "    ld l, (ix + 4)".into(),
            "    ld h, (ix + 5)".into(),
            "    ld a, (ix + 4)".into(),
            "    add a, l".into(),
            "    ld e, (ix + 4)".into(),
            "    ld d, (ix + 5)".into(),
            "    add hl, de".into(),
            "    ld (ix + 4), a".into(),
            "    ld e, (ix + 4)".into(),
            "    ld d, (ix + 5)".into(),
            "    ret".into(),
        ];
        let ix_opt = super::optimize_asm_full(
            &ix_asm.join("\n"),
            &HashMap::new(),
            &HashMap::new(),
            &HashSet::new(),
        );
        let ix_loads = ix_opt.matches("(ix").count();
        assert!(
            ix_opt.contains("ld a, l") && ix_opt.contains("ld d, h") && ix_opt.contains("ld e, l"),
            "expected IX reloads in A and DE to reuse HL:\n{ix_opt}"
        );
        assert!(
            ix_opt.contains("ld (ix + 4), a") && ix_opt.contains("ld e, (ix + 4)"),
            "expected store to (ix + 4) to invalidate cached byte 4 and force reload:\n{ix_opt}"
        );
        assert!(
            ix_loads <= 5,
            "expected redundant (ix+d) reloads eliminated (got {ix_loads}):\n{ix_opt}"
        );
    }

    #[test]
    fn issue51_reg_cache_walks_hl_across_add_hl_de_and_invalidates_on_clobber() {
        // Positive: `ld hl, (__aun_ram + 10); ld de, 0x0009; add hl, de` leaves `hl_base_off` at +9,
        // so the subsequent `ld hl, (__aun_ram + 10); ld de, 0x000A; add hl, de` folds to `inc hl`.
        let walk_asm = [
            "fn_walk:",
            "fn_walk_bb0:",
            "    ld hl, (__aun_ram + 10)",
            "    ld de, 0x0009",
            "    add hl, de",
            "    ld a, (hl)",
            "    ld (0xC000), a",
            "    ld hl, (__aun_ram + 10)",
            "    ld de, 0x000A",
            "    add hl, de",
            "    ld a, (hl)",
            "    ld (0xC001), a",
            "    ld de, 0x0000",
            "    or a",
            "    ret",
        ]
        .join("\n");
        let opt = super::optimize_asm(&walk_asm);
        assert!(
            opt.contains("inc hl"),
            "expected second offset load (+10 after +9) to fold to inc hl:\n{opt}"
        );
        assert_eq!(
            opt.matches("ld hl, (__aun_ram + 10)").count(),
            1,
            "expected only one reload of (__aun_ram + 10):\n{opt}"
        );

        // Negative: overwriting `(__aun_ram + 10)` or clobbering `HL` via `call` must invalidate `hl_base_off`.
        let clobber_asm = [
            "fn_clobber:",
            "fn_clobber_bb0:",
            "    ld hl, (__aun_ram + 10)",
            "    ld de, 0x0009",
            "    add hl, de",
            "    ld a, (hl)",
            "    call helper_fn",
            "    ld hl, (__aun_ram + 10)",
            "    ld de, 0x000A",
            "    add hl, de",
            "    ld a, (hl)",
            "    ld de, 0x0000",
            "    or a",
            "    ret",
        ]
        .join("\n");
        let opt_clobber = super::optimize_asm(&clobber_asm);
        assert_eq!(
            opt_clobber.matches("ld hl, (__aun_ram + 10)").count(),
            2,
            "call must invalidate HL base+off tracking and preserve second reload:\n{opt_clobber}"
        );
    }
