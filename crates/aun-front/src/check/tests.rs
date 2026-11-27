    use super::*;
    use crate::parser::Parser;

    #[test]
    fn reject_ref_in_struct() {
        let src = "struct Bad { r: &u8 }";
        let file = Parser::parse_str(0, src).unwrap();
        let errs = Checker::check_and_lower(&file).unwrap_err();
        assert!(errs.iter().any(|d| d.message.contains("second-class")));
    }

    #[test]
    fn reject_mixed_width_arithmetic() {
        let src = "fn f(a: u8, b: u16) -> u16 { a + b }";
        let file = Parser::parse_str(0, src).unwrap();
        let errs = Checker::check_and_lower(&file).unwrap_err();
        assert!(errs.iter().any(|d| d.message.contains("strict type mismatch")));
    }

    #[test]
    fn reject_mutable_aliasing_allow_disjoint_fields() {
        let src = r#"
            struct Pair { a: u8, b: u8 }
            fn swap(x: &mut u8, y: &mut u8) { let t: u8 = *x; *x = *y; *y = t; }
            fn ok() {
                let mut p: Pair = Pair { a: 1, b: 2 };
                swap(&mut p.a, &mut p.b);
            }
        "#;
        let file = Parser::parse_str(0, src).unwrap();
        Checker::check_and_lower(&file).expect("disjoint field borrows should pass");

        let bad_src = r#"
            fn swap(x: &mut u8, y: &mut u8) {}
            fn bad() {
                let mut a: u8 = 1;
                swap(&mut a, &mut a);
            }
        "#;
        let bad_file = Parser::parse_str(0, bad_src).unwrap();
        let errs = Checker::check_and_lower(&bad_file).unwrap_err();
        assert!(errs.iter().any(|d| d.message.contains("aliasing violation")));
    }

    #[test]
    fn reject_forbidden_static_transition_and_strict_dynamic_goto() {
        let bad_static = r#"
            machine Ball {
                state Inactive;
                state InShooter { power: u8 };
                state MagnetHeld;
                transitions {
                    Inactive -> InShooter;
                }
                on InShooter {
                    fn step(&mut self) {
                        goto MagnetHeld;
                    }
                }
            }
        "#;
        let file = Parser::parse_str(0, bad_static).unwrap();
        let errs = Checker::check_and_lower(&file).unwrap_err();
        assert!(errs
            .iter()
            .any(|d| d.message.contains("forbidden static state transition")));

        let bad_strict = r#"
            #[strict_transitions]
            machine Door {
                state Closed;
                state Open;
                transitions { Closed -> Open; }
            }
            fn f() {
                let mut d: Door = Door::new();
                d.goto(Open);
            }
        "#;
        let file2 = Parser::parse_str(0, bad_strict).unwrap();
        let errs2 = Checker::check_and_lower(&file2).unwrap_err();
        assert!(errs2
            .iter()
            .any(|d| d.message.contains("strict_transitions")));
    }

    #[test]
    fn static_bank_conflict_and_di_checking() {
        let base_target = r#"
            target pc88sr : z80 {
                space MainRam @ 0x0100..0xBFFF;
                stack MainStack @ 0xBE00..0xBFFF;
                bank_group HighWindow @ 0xC000..0xFFFF {
                    bank HighMainRam @ 0xC000..0xEFFF;
                    bank VramAlu @ 0xC000..0xFFFF requires di;
                    default HighMainRam;
                    switch(to) {
                        HighMainRam => { io_out(0x35, 0x00); }
                        VramAlu     => { io_out(0x35, 0x80); }
                    }
                }
            }
            static mut SCORE: u8 in HighMainRam;
            extern "asm" {
                fn gfx_fill(c: u8 in A) requires bank(VramAlu), di clobbers(AF);
            }
        "#;

        // OK: calling gfx_fill inside `with bank(VramAlu)`
        let ok_src = format!(
            "{base_target}\nfn draw() {{ let s: u8 = SCORE; with bank(VramAlu) {{ gfx_fill(s); }} }}"
        );
        let ok_file = Parser::parse_str(0, &ok_src).unwrap();
        Checker::check_and_lower(&ok_file).expect("valid bank usage should pass");

        // Error: accessing `SCORE in HighMainRam` while `VramAlu` is mapped!
        let bad_conflict = format!(
            "{base_target}\nfn bad() {{ with bank(VramAlu) {{ gfx_fill(SCORE); }} }}"
        );
        let bad_file = Parser::parse_str(0, &bad_conflict).unwrap();
        let errs = Checker::check_and_lower(&bad_file).unwrap_err();
        assert!(errs.iter().any(|d| d.message.contains("bank conflict")));

        // Error: calling `gfx_fill` without `with bank(VramAlu)`
        let bad_precond = format!("{base_target}\nfn bad2() {{ gfx_fill(1u8); }}");
        let bad_file2 = Parser::parse_str(0, &bad_precond).unwrap();
        let errs2 = Checker::check_and_lower(&bad_file2).unwrap_err();
        assert!(errs2.iter().any(|d| d.message.contains("bank conflict")));
        assert!(errs2.iter().any(|d| d.message.contains("requires `di`")));
    }

    #[test]
    fn mem_write_fixed_addr_and_extern_static_lowering() {
        let src = r#"
            struct GameState {
                score: u16,
                lives: u8,
            }

            static mut G_STATE: GameState @ 0xEC00;

            extern "asm" {
                static mut _EXT_COUNT: u16;
                static _EXT_FLAGS: u8;
            }

            pub fn tick(v: u8) {
                G_STATE.lives = v;
                G_STATE.score = 1234u16;
                _EXT_COUNT = _EXT_COUNT +% 1u16;
                mem_write_u8(0xF3C8, _EXT_FLAGS +% v);
                mem_write_u16(0xF3CA, G_STATE.score);
            }
        "#;
        let prog = crate::compile_str(0, src).expect("should compile");
        assert_eq!(prog.globals.len(), 3);
        assert_eq!(prog.globals[0].fixed_addr, Some(0xEC00));
        assert!(!prog.globals[0].is_extern);
        assert_eq!(prog.globals[1].fixed_addr, None);
        assert!(prog.globals[1].is_extern);
        assert!(prog.globals[1].mutable);
        assert!(prog.globals[2].is_extern);
        assert!(!prog.globals[2].mutable);

        let bad = r#"
            extern "asm" {
                static _EXT_RO: u8;
            }
            fn write_ro() {
                _EXT_RO = 1u8;
            }
        "#;
        let errs = crate::compile_str(0, bad).unwrap_err();
        assert!(
            errs.iter().any(|d| d.message.contains("immutable")),
            "expected immutability error, got {errs:?}"
        );
    }

    #[test]
    fn include_directive_and_extern_rom_const_lowering() {
        let dir = std::env::temp_dir().join(format!(
            "aun_front_inc_test_{}",
            std::process::id()
        ));
        let sub = dir.join("sub");
        std::fs::create_dir_all(&sub).unwrap();

        let main_path = dir.join("main.aun");
        let sub_path = sub.join("tables.aun");
        let leaf_path = sub.join("leaf.aun");

        std::fs::write(
            &leaf_path,
            "const BASE_OFFSET: u8 = 10u8;\nconst ROM_TBL: [u8; 4] = [1u8, 2u8, 3u8, 4u8];\n",
        )
        .unwrap();
        std::fs::write(&sub_path, "include!(\"leaf.aun\");\n").unwrap();
        std::fs::write(
            &main_path,
            r#"
                include!("sub/tables.aun");

                extern "asm" {
                    const _EXT_LUT: [u8; 4];
                }

                pub fn lookup(i: u8) -> u8 {
                    ROM_TBL[i] +% _EXT_LUT[i] +% BASE_OFFSET
                }
            "#,
        )
        .unwrap();

        let (_src, prog) = crate::compile_file(&main_path).expect("compile_file should succeed");
        assert_eq!(prog.globals.len(), 2);
        assert!(prog.globals[0].in_rom && !prog.globals[0].is_extern);
        assert_eq!(prog.globals[1].name, "_EXT_LUT");
        assert!(prog.globals[1].in_rom && prog.globals[1].is_extern && !prog.globals[1].mutable);

        // Cyclic include must error cleanly
        let cyc_a = dir.join("cyc_a.aun");
        let cyc_b = dir.join("cyc_b.aun");
        std::fs::write(&cyc_a, "include!(\"cyc_b.aun\");\n").unwrap();
        std::fs::write(&cyc_b, "include!(\"cyc_a.aun\");\n").unwrap();
        let cyc_errs = crate::compile_file(&cyc_a).unwrap_err();
        assert!(
            cyc_errs.iter().any(|d| d.message.contains("cyclic include")),
            "expected cyclic include error, got {cyc_errs:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn lowers_deduped_rom_strings_byte_strings_and_array_ref_slices() {
        let src = r#"
            const GREET: [u8; 5] = b"HELLO";

            fn sum_slice(s: &const [u8]) -> u16 {
                let mut total: u16 = s.len();
                let mut i: u16 = 0u16;
                while i < s.len() {
                    total = total +% (s[i] as u16);
                    i = i +% 1u16;
                }
                total
            }

            fn write_ref_arr(buf: &mut [u8; 4], s: &[u8]) {
                buf[0] = s[0];
                buf[1] = "XYZ"[2u8];
                buf[2] = (buf.len() as u8);
            }

            pub fn run() -> u16 {
                let mut local_arr: [u8; 4] = b"ABCD";
                write_ref_arr(&mut local_arr, "HI");
                let r_arr: &[u8; 4] = &local_arr;
                sum_slice(r_arr) +% sum_slice("AB") +% sum_slice(b"AB") +% (GREET[0] as u16)
            }
        "#;
        let prog = crate::compile_str(0, src).expect("Issue #9 sample should compile");
        let rom_str_globals: Vec<_> = prog
            .globals
            .iter()
            .filter(|g| g.name.starts_with("__aun_str_"))
            .collect();
        // `"AB"` and `b"AB"` must deduplicate into one ROM global!
        assert_eq!(
            rom_str_globals.iter().filter(|g| g.init == b"AB").count(),
            1,
            "expected deduplicated `__aun_str_*` for \"AB\": {rom_str_globals:?}"
        );
        assert!(rom_str_globals.iter().all(|g| g.in_rom && !g.mutable));
        assert_eq!(prog.globals[0].init, b"HELLO");

        // Immutable slice write must be rejected
        let bad = r#"
            fn bad_write(buf: &[u8; 4], s: &const [u8]) {
                buf[0] = s[0];
            }
        "#;
        let errs = crate::compile_str(0, bad).unwrap_err();
        assert!(
            errs.iter().any(|d| d.message.contains("immutable")),
            "expected immutability error on `&[u8; 4]`, got {errs:?}"
        );
    }

    #[test]
    fn lowers_addr_of_fn_fixed_calls_stack_args_and_volatile() {
        let src = r#"
            volatile static mut LOCAL_TICK: u8 = 0u8;

            extern "asm" {
                volatile static mut _g_vrtc_ticks: u8;
                fn _bios_chput(c: u8 in a) @ 0x00A2;
                fn _ext_cb(x: u8 in a);
                fn _c_stack_fn(a: u8 in a, b: u8 in l, c: u8 in stack, d: u16 in stack) -> u16 in de;
            }

            fn my_fn() {}

            pub fn entry(f: u8) -> u16 {
                let p0: u16 = addr_of(my_fn);
                let p1: u16 = addr_of(_ext_cb);
                let p2: u16 = addr_of(_bios_chput);
                _bios_chput(0x41u8);
                call_u16(p0);
                call_u16(p1, 7u8);
                let v1 = _g_vrtc_ticks +% LOCAL_TICK;
                let v2 = mem_read_u8_volatile(0x0099u16);
                mem_write_u16_volatile(0xF000u16, p2);
                _c_stack_fn(1u8, f, v1 +% v2, 0x1234u16)
            }
        "#;
        let prog = crate::compile_str(0, src).expect("Issue #10 sample should compile");
        assert!(prog.globals[0].volatile);
        assert!(prog.globals[1].volatile);
        assert_eq!(prog.externs[0].fixed_addr, Some(0x00A2));
        assert_eq!(prog.externs[2].params[2].2, "stack");
        assert_eq!(prog.externs[2].params[3].2, "stack");

        let entry = prog.functions.iter().find(|f| f.name == "entry").unwrap();
        let all_insts: Vec<_> = entry.blocks.iter().flat_map(|b| &b.insts).collect();
        assert!(all_insts.iter().any(|i| matches!(i, aun_ir::IrInst::AddrFn { target: aun_ir::IrFnTarget::Fn(_), .. })));
        assert!(all_insts.iter().any(|i| matches!(i, aun_ir::IrInst::AddrFn { target: aun_ir::IrFnTarget::Extern(1), .. })));
        assert!(all_insts.iter().any(|i| matches!(i, aun_ir::IrInst::ConstInt { val: 0x00A2, .. })));
        assert!(all_insts.iter().any(|i| matches!(i, aun_ir::IrInst::CallIndirect { .. })));
        assert!(all_insts.iter().filter(|i| matches!(i, aun_ir::IrInst::LoadPtr { volatile: true, .. })).count() >= 3);
        assert!(all_insts.iter().any(|i| matches!(i, aun_ir::IrInst::StorePtr { volatile: true, .. })));
    }
