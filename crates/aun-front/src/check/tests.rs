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
