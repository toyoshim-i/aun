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

    #[test]
    fn resolves_inline_mod_use_pub_and_qualified_paths() {
        let src = r#"
            mod math {
                pub const BIAS: u8 = 10u8;
                pub enum Mode: u8 { Fast = 1, Slow = 2 }
                pub struct Vec2 { x: u8, y: u8 }
                fn secret_val() -> u8 { 99u8 }
                pub fn add(v: &Vec2, m: Mode) -> u8 {
                    v.x +% v.y +% (m as u8)
                }
                pub mod inner {
                    pub fn delta(x: u8) -> u8 {
                        super::secret_val() -% x
                    }
                }
            }

            use math::{Vec2, Mode, add as m_add};
            use math::inner::*;

            pub fn run() -> u8 {
                let a = Vec2 { x: 3u8, y: 4u8 };
                let b = math::Vec2 { x: math::BIAS, y: 1u8 };
                m_add(&a, Mode::Fast) +% math::add(&b, math::Mode::Slow) +% delta(9u8)
            }
        "#;
        let prog = crate::compile_str(0, src).expect("inline mod/use should compile");
        assert!(prog.functions.iter().any(|f| f.name == "run" && f.is_pub));
        assert!(prog.functions.iter().any(|f| f.name == "math__add" && !f.is_pub));
        assert!(prog.functions.iter().any(|f| f.name == "math__inner__delta" && !f.is_pub));
    }

    #[test]
    fn rejects_private_module_and_private_items() {
        let priv_fn = r#"
            mod secret {
                fn hidden() -> u8 { 1u8 }
            }
            pub fn main() -> u8 { secret::hidden() }
        "#;
        let errs = crate::compile_str(0, priv_fn).unwrap_err();
        assert!(
            errs.iter().any(|d| d.message.contains("private")),
            "expected privacy error, got: {errs:?}"
        );

        let priv_submod = r#"
            mod outer {
                mod priv_inner {
                    pub fn f() -> u8 { 2u8 }
                }
            }
            use outer::priv_inner::f;
            pub fn main() -> u8 { f() }
        "#;
        let errs2 = crate::compile_str(0, priv_submod).unwrap_err();
        assert!(
            errs2.iter().any(|d| d.message.contains("private")),
            "expected submodule privacy error, got: {errs2:?}"
        );
    }

    #[test]
    fn compile_file_resolves_external_mod_tree_and_multi_unit_lto() {
        let tmp = std::env::temp_dir().join(format!("aun_front_mod_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("audio")).unwrap();
        std::fs::create_dir_all(tmp.join("ui")).unwrap();

        std::fs::write(
            tmp.join("main.aun"),
            r#"
mod audio;
mod ui;
use audio::{Tone, sfx::blip};
use ui::badge;

fn helper() -> u8 { 5u8 }

extern "asm" {
    fn _hal_ping(v: u8 in a) clobbers(A);
}

pub fn entry() -> u8 {
    let t = Tone { pitch: 12u8 };
    _hal_ping(t.pitch);
    helper() +% t.pitch +% blip() +% badge()
}
"#,
        )
        .unwrap();

        std::fs::write(
            tmp.join("audio.aun"),
            r#"
pub mod sfx;
pub struct Tone { pitch: u8 }
"#,
        )
        .unwrap();
        std::fs::write(
            tmp.join("audio/sfx.aun"),
            r#"
pub fn blip() -> u8 { 7u8 }
"#,
        )
        .unwrap();
        std::fs::write(
            tmp.join("ui/mod.aun"),
            r#"
pub fn badge() -> u8 { 9u8 }
"#,
        )
        .unwrap();

        let (src, prog) =
            crate::compile_file(&tmp.join("main.aun")).expect("external mod tree should compile");
        assert!(src.contains("mod audio;"));
        assert!(prog.functions.iter().any(|f| f.name == "entry" && f.is_pub));
        assert!(prog.functions.iter().any(|f| f.name == "audio__sfx__blip" && !f.is_pub));
        assert!(prog.functions.iter().any(|f| f.name == "ui__badge" && !f.is_pub));

        // Now test multi-file compile_files(&[main_lto, sound]) with duplicate extern & private names
        std::fs::write(
            tmp.join("sound.aun"),
            r#"
static mut STATE: u8 = 3u8;
extern "asm" {
    fn _hal_ping(v: u8 in a) clobbers(A);
}
fn helper() -> u8 { STATE }
pub fn sound_step() -> u8 {
    _hal_ping(1u8);
    helper() +% 10u8
}
"#,
        )
        .unwrap();
        std::fs::write(
            tmp.join("main_unit.aun"),
            r#"
static mut STATE: u8 = 4u8;
extern "asm" {
    fn _hal_ping(v: u8 in a) clobbers(A);
}
fn helper() -> u8 { STATE }
pub fn run_all() -> u8 {
    helper() +% sound::sound_step()
}
"#,
        )
        .unwrap();

        let m_path = tmp.join("main_unit.aun");
        let s_path = tmp.join("sound.aun");
        let (_, lto_prog) = crate::compile_files(&[m_path.as_path(), s_path.as_path()])
            .expect("multi-file compile_files should succeed");
        assert_eq!(
            lto_prog.externs.iter().filter(|e| e.name == "_hal_ping").count(),
            1,
            "duplicate extern_asm declarations across units should be deduplicated"
        );
        assert!(lto_prog.functions.iter().any(|f| f.name == "run_all" && f.is_pub));
        assert!(lto_prog.functions.iter().any(|f| f.name == "sound_step" && f.is_pub));
        assert!(lto_prog.functions.iter().any(|f| f.name == "main_unit__helper" && !f.is_pub));
        assert!(lto_prog.functions.iter().any(|f| f.name == "sound__helper" && !f.is_pub));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn cross_module_and_forward_scalar_consts_resolve_before_aggregates() {
        let tmp = std::env::temp_dir().join(format!("aun_issue16_consts_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        std::fs::write(
            tmp.join("game_types.aun"),
            r#"
pub const KP_1: u8 = 0x06u8;
pub const KP_2: u8 = 0x07u8;
pub const CHILD_ARR: [u8; 2] = [super::PARENT_CONST, KP_1];
"#,
        )
        .unwrap();
        std::fs::write(
            tmp.join("main.aun"),
            r#"
mod game_types;
use game_types::*;

const PARENT_CONST: u8 = 0x42u8;
const LD_NUM_PORT: [u8; 2] = [KP_1, KP_2];
const CHAIN_TOP: i8 = -CHAIN_MID;
const CHAIN_MID: i8 = CHAIN_BASE;
const CHAIN_BASE: i8 = 12i8;
static mut ST_PORT: [u8; 2] = [KP_2, KP_1];

pub fn read_vals() -> u8 {
    LD_NUM_PORT[0] +% ST_PORT[0] +% CHILD_ARR[0] +% (CHAIN_TOP as u8)
}
"#,
        )
        .unwrap();

        let (_, prog) = crate::compile_file(&tmp.join("main.aun"))
            .expect("parent aggregate referencing child mod const should compile");
        let ld = prog
            .globals
            .iter()
            .find(|g| g.name == "LD_NUM_PORT")
            .expect("LD_NUM_PORT global should exist");
        assert_eq!(ld.init, vec![0x06, 0x07]);

        let child_arr = prog
            .globals
            .iter()
            .find(|g| g.name == "game_types__CHILD_ARR")
            .expect("CHILD_ARR global should exist");
        assert_eq!(child_arr.init, vec![0x42, 0x06]);

        let st = prog
            .globals
            .iter()
            .find(|g| g.name == "ST_PORT")
            .expect("ST_PORT static should exist");
        assert_eq!(st.init, vec![0x07, 0x06]);

        let _ = std::fs::remove_dir_all(&tmp);

        let cyclic_src = r#"
const X: u8 = Y;
const Y: u8 = X;
pub fn f() -> u8 { X }
"#;
        let cyclic_file = Parser::parse_str(0, cyclic_src).unwrap();
        let errs = Checker::check_and_lower(&cyclic_file).unwrap_err();
        assert!(errs
            .iter()
            .any(|d| d.message.contains("const initializer must be a compile-time constant")));
    }

    #[test]
    fn ch559_mcs51_target_spaces_isr_using_extern_and_guards() {
        let good_src = r#"
            target ch559 : mcs51 {
                space Data  : data  @ 0x0000..0x007F;
                space XData : xdata @ 0x0000..0x17FF;
                space Sfr   : sfr   @ 0x0080..0x00FF;
                space Bit   : bit   @ 0x0080..0x00FF;
                stack IStack @ 0x0080..0x00FF;

                interrupt_controller Ch559Int {
                    disable => asm!("clr EA");
                    enable  => asm!("setb EA");
                    source Usb @ irq(8);
                }
            }

            volatile static mut EA: bool in Bit @ 0xAF;
            volatile static mut UEP0_CTRL: u8 in Sfr @ 0xDC;
            #[align(64)]
            static mut EP0_BUF: [u8; 64] in XData;
            static DESCRIPTOR: [u8; 4] = [0x12u8, 0x01u8, 0x00u8, 0x02u8];
            const TABLE_C: [u8; 2] = [1u8, 2u8];

            extern "asm" {
                static mut _USB_DMA: u16 in XData;
                fn _ch_delay(ms: u8 in dpl);
            }

            #[interrupt(Usb, using(1))]
            fn usb_isr() {
                EA = false;
                UEP0_CTRL = 0x02u8;
                EP0_BUF[0] = DESCRIPTOR[0] +% TABLE_C[0];
                _USB_DMA = 0u16;
                _ch_delay(1u8);
                EA = true;
            }
        "#;
        let file = Parser::parse_str(0, good_src).expect("CH559 target should parse");
        let prog = Checker::check_and_lower(&file).expect("CH559 target should lower cleanly");
        let target = prog.target.as_ref().unwrap();
        assert_eq!(target.arch, "mcs51");
        assert_eq!(target.spaces[0].kind, IrAddrSpace::Data);
        assert_eq!(target.spaces[1].kind, IrAddrSpace::XData);
        assert_eq!(target.spaces[2].kind, IrAddrSpace::Sfr);
        assert_eq!(target.spaces[3].kind, IrAddrSpace::Bit);
        assert_eq!(target.interrupt_controllers[0].sources[0].mode, "irq");

        let isr = prog.functions.iter().find(|f| f.name == "usb_isr").unwrap();
        assert_eq!(isr.interrupt_source.as_deref(), Some("Usb"));
        assert_eq!(isr.isr_using, Some(1));

        let ep0 = prog.globals.iter().find(|g| g.name == "EP0_BUF").unwrap();
        assert!(!ep0.in_rom);
        assert_eq!(ep0.align, 64);
        assert_eq!(ep0.space.as_deref(), Some("XData"));

        let usb_dma = prog.globals.iter().find(|g| g.name == "_USB_DMA").unwrap();
        assert!(usb_dma.is_extern);
        assert_eq!(usb_dma.space.as_deref(), Some("XData"));
        assert_eq!(prog.externs[0].clobbers, None);

        // Verify symbolic slice ops and MemFill are emitted without Z80 u32 packing or overlapping MemCopy
        let slice_src = r#"
            struct Pair { a: u8, b: u8 }
            fn sum_slice(s: &[u8]) -> u16 { (s[0] as u16) +% s.len() }
            pub fn test_ops() -> u16 {
                let a: [u16; 6] = [0x1234u16; 6];
                let p: [Pair; 4] = [Pair { a: 1u8, b: 2u8 }; 4];
                let bytes: [u8; 3] = [10u8, 20u8, 30u8];
                sum_slice(&bytes) +% (p[3].b as u16) +% a[5]
            }
        "#;
        let sfile = Parser::parse_str(0, slice_src).unwrap();
        let sprog = Checker::check_and_lower(&sfile).unwrap();
        let sum_insts: Vec<&IrInst> = sprog.functions[0].blocks[0].insts.iter().collect();
        assert!(sum_insts.iter().any(|i| matches!(i, IrInst::SlicePtr { .. })));
        assert!(sum_insts.iter().any(|i| matches!(i, IrInst::SliceLen { .. })));
        let test_insts: Vec<&IrInst> = sprog.functions[1].blocks[0].insts.iter().collect();
        assert!(test_insts.iter().any(|i| matches!(i, IrInst::MakeSlice { .. })));
        assert!(
            test_insts
                .iter()
                .filter(|i| matches!(i, IrInst::MemFill { .. }))
                .count()
                >= 2
        );
        assert!(
            !test_insts.iter().any(|i| matches!(i, IrInst::MemCopy { .. })),
            "array repeat should use MemFill instead of overlapping MemCopy"
        );

        // Verify guards: &volatile, &Sfr, &Bit, #[smc] on mcs51, io_out on mcs51, unknown space
        for (bad_src, expected_err) in [
            (
                "volatile static mut IO: u8 @ 0xF000; fn f() { let p = &IO; }",
                "volatile",
            ),
            (
                "target ch : mcs51 { space Sfr : sfr @ 0x80..0xFF; } volatile static mut P1: u8 in Sfr @ 0x90; fn f() { let p = &P1; }",
                "volatile",
            ),
            (
                "target ch : mcs51 { space Sfr : sfr @ 0x80..0xFF; } static mut P1: u8 in Sfr @ 0x90; fn f() { let p = &P1; }",
                "address-space static",
            ),
            (
                "target ch : mcs51 { } #[smc] machine M { state A; } fn f() {}",
                "#[smc]",
            ),
            (
                "target ch : mcs51 { } fn f() { io_out(0x10u8, 1u8); }",
                "io_out",
            ),
            (
                "target ch : mcs51 { space XData : xdata @ 0..0xFF; } static mut V: u8 in Bogus; fn f() {}",
                "unknown target address space",
            ),
        ] {
            let f = Parser::parse_str(0, bad_src).unwrap();
            let errs = Checker::check_and_lower(&f).unwrap_err();
            assert!(
                errs.iter().any(|d| d.message.contains(expected_err)),
                "expected `{expected_err}` error for `{bad_src}`, got: {errs:?}"
            );
        }
    }
