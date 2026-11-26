//! Z80 backend: WPO-ABI, instruction selection, *Uchiguri* static RAM overlay,
//! peephole optimizer, built-in SDASZ80-compatible assembler, and linker.

pub mod analysis;
pub mod asm;
pub mod codegen;
pub mod lir;
pub mod outliner;
pub mod peephole;
pub mod regalloc;
pub mod sdasz80;
pub mod uchiguri;

use aun_ir::IrProgram;
use std::collections::HashMap;
use uchiguri::{FnRamFrame, GlobalRamEntry};

/// A compiled code overlay binary linked at its `overlay_slot` base address.
#[derive(Debug, Clone)]
pub struct OverlayOutput {
    pub slot_name: String,
    pub state_name: String,
    pub origin: u16,
    pub bin: Vec<u8>,
    pub asm_listing: String,
}

/// Result of assembling and linking a Z80 program.
#[derive(Debug, Default)]
pub struct LinkOutput {
    pub resident_bin: Vec<u8>,
    pub overlays: Vec<OverlayOutput>,
    pub asm_listing: String,
    pub symbols: HashMap<String, u16>,
    pub code_origin: u16,
    pub ram_base: u16,
    pub overlay_base: u16,
    pub ram_end: u16,
    pub ram_bytes: u16,
    pub max_stack_bytes: u16,
    pub total_tstates: u32,
    pub fn_tstates: HashMap<String, u32>,
    pub globals: Vec<GlobalRamEntry>,
    pub fn_frames: Vec<FnRamFrame>,
}

/// Compilation and check-elimination options for `aun-z80`.
#[derive(Debug, Clone, Copy)]
pub struct CompileOptions {
    pub release: bool,
    pub check_bounds: bool,
    pub check_transitions: bool,
    pub check_overflow: bool,
    pub check_bank: bool,
    pub code_origin: u16,
    pub ram_base: u16,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self::debug()
    }
}

impl CompileOptions {
    pub const fn debug() -> Self {
        Self {
            release: false,
            check_bounds: true,
            check_transitions: true,
            check_overflow: true,
            check_bank: true,
            code_origin: 0x0100,
            ram_base: 0xC000,
        }
    }

    pub const fn release() -> Self {
        Self {
            release: true,
            check_bounds: false,
            check_transitions: false,
            check_overflow: false,
            check_bank: false,
            code_origin: 0x0100,
            ram_base: 0xC000,
        }
    }
}

/// Compiles an `IrProgram` (and optional external SDASZ80 `.s` sources) into Z80 machine code.
pub fn compile_z80(program: &IrProgram) -> Result<LinkOutput, String> {
    compile_z80_with_options(program, &[], CompileOptions::debug())
}

pub fn compile_z80_with_asm(
    program: &IrProgram,
    external_asm_sources: &[&str],
    code_origin: u16,
    ram_base: u16,
) -> Result<LinkOutput, String> {
    compile_z80_with_options(
        program,
        external_asm_sources,
        CompileOptions {
            code_origin,
            ram_base,
            ..CompileOptions::debug()
        },
    )
}

/// Compiles an `IrProgram` into relocatable SDASZ80 `.s` assembly for linking with `sdasz80`/`sdldz80`.
pub fn compile_z80_sdasz80(
    module_name: &str,
    program: &IrProgram,
    opts: CompileOptions,
) -> Result<(String, u16, u16), String> {
    let mut opt_prog = program.clone();
    aun_ir::opt::optimize_program(&mut opt_prog);
    let program = &opt_prog;
    if cfg!(debug_assertions)
        && let Err(errs) = aun_ir::verify::verify_program(program)
    {
        panic!("MIR verification failed:\n{}", errs.join("\n"));
    }
    let analysis = analysis::ProgramAnalysis::analyze(program);
    let layout = uchiguri::allocate_uchiguri_with_analysis(program, &analysis, opts.ram_base)?;
    let raw_asm = codegen::emit_z80_asm_with_analysis(program, &analysis, &layout, opts)?;
    let fn_private_ranges = build_fn_private_ranges(program, &analysis, &layout);
    let fn_sb_ranges = build_fn_sb_ranges(&analysis, &layout);
    let no_bc_fns = analysis.no_bc_fns(program);
    let opt_asm = peephole::optimize_asm_full(&raw_asm, &fn_private_ranges, &fn_sb_ranges, &no_bc_fns);
    let sdas = sdasz80::emit_sdasz80(module_name, &opt_asm, program, &layout);
    let ram_bytes = layout.ram_end.saturating_sub(layout.ram_base);
    Ok((sdas, ram_bytes, layout.max_stack_bytes))
}

fn build_fn_sb_ranges(
    analysis: &analysis::ProgramAnalysis,
    layout: &uchiguri::RamLayout,
) -> HashMap<String, (u16, u16)> {
    let mut out = HashMap::new();
    for frame in &layout.fn_frames {
        let has_inline_asm = analysis
            .fns
            .get(frame.func_id as usize)
            .is_some_and(|fa| fa.has_inline_asm);
        if has_inline_asm {
            continue;
        }
        if let Some(&r) = layout.fn_sb_scratch_ranges.get(&frame.func_id) {
            out.insert(analysis::z80_sym(&frame.name), r);
        }
    }
    out
}

fn build_fn_private_ranges(
    program: &IrProgram,
    analysis: &analysis::ProgramAnalysis,
    layout: &uchiguri::RamLayout,
) -> HashMap<String, Vec<(u16, u16)>> {
    let mut fn_private_ranges: HashMap<String, Vec<(u16, u16)>> = HashMap::new();
    for frame in &layout.fn_frames {
        let mut ranges: Vec<(u16, u16)> = Vec::new();
        let fa = analysis.fns.get(frame.func_id as usize);
        if fa.is_some_and(|a| a.has_inline_asm) {
            continue;
        }
        if frame.scratch_size > 0 {
            ranges.push((
                frame.scratch_base,
                frame.scratch_base.wrapping_add(frame.scratch_size),
            ));
        }
        if let Some(func) = program.functions.get(frame.func_id as usize)
            && let Some(fa) = fa
        {
            for (slot_idx, local) in func.locals.iter().enumerate() {
                if !fa.escaped_locals[slot_idx]
                    && let Some(&addr) = layout.slot_addrs.get(&(func.id, local.id))
                {
                    let sz = program.type_size(&local.ty);
                    if sz > 0 {
                        ranges.push((addr, addr.wrapping_add(sz)));
                    }
                }
            }
        }
        if !ranges.is_empty() {
            fn_private_ranges.insert(analysis::z80_sym(&frame.name), ranges);
        }
    }
    fn_private_ranges
}

pub fn compile_z80_with_options(
    program: &IrProgram,
    external_asm_sources: &[&str],
    opts: CompileOptions,
) -> Result<LinkOutput, String> {
    let mut opt_prog = program.clone();
    aun_ir::opt::optimize_program(&mut opt_prog);
    let program = &opt_prog;
    if cfg!(debug_assertions)
        && let Err(errs) = aun_ir::verify::verify_program(program)
    {
        panic!("MIR verification failed:\n{}", errs.join("\n"));
    }
    let analysis = analysis::ProgramAnalysis::analyze(program);
    let layout = uchiguri::allocate_uchiguri_with_analysis(program, &analysis, opts.ram_base)?;
    let raw_asm = codegen::emit_z80_asm_with_analysis(program, &analysis, &layout, opts)?;
    let fn_private_ranges = build_fn_private_ranges(program, &analysis, &layout);
    let fn_sb_ranges = build_fn_sb_ranges(&analysis, &layout);
    let no_bc_fns = analysis.no_bc_fns(program);
    let mut full_asm = peephole::optimize_asm_full(&raw_asm, &fn_private_ranges, &fn_sb_ranges, &no_bc_fns);
    for ext in external_asm_sources {
        full_asm.push('\n');
        full_asm.push_str(ext);
    }
    let assembled = asm::assemble_z80(&full_asm, opts.code_origin)?;

    // Compute per-function T-states using sorted top-level function/extern symbol boundaries
    let mut top_syms: Vec<(u16, String)> = Vec::new();
    for f in &program.functions {
        let sym = analysis::z80_sym(&f.name);
        if !f.is_test
            && let Some(&addr) = assembled.symbols.get(&sym)
        {
            top_syms.push((addr, f.name.clone()));
        }
    }
    for e in &program.externs {
        if let Some(&addr) = assembled.symbols.get(&e.name) {
            top_syms.push((addr, e.name.clone()));
        }
    }
    if let Some(&addr) = assembled.symbols.get("__aun_panic") {
        top_syms.push((addr, "__aun_panic".to_string()));
    }
    top_syms.sort_by_key(|(addr, _)| *addr);
    top_syms.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

    let mut fn_tstates: HashMap<String, u32> = HashMap::new();
    for (idx, (start, name)) in top_syms.iter().enumerate() {
        let end = top_syms
            .get(idx + 1)
            .map(|(a, _)| *a)
            .unwrap_or(u16::MAX);
        let ts: u32 = assembled
            .inst_tstates
            .iter()
            .filter(|(pc, _)| *pc >= *start && *pc < end)
            .map(|(_, t)| *t as u32)
            .sum();
        fn_tstates.insert(name.clone(), ts);
    }

    // Assemble real overlay binaries for each `#[overlay(Slot)]` state
    let mut overlays = Vec::new();
    if let Some(target) = &program.target {
        for mdef in &program.machines {
            for st in &mdef.states {
                if let Some(slot_name) = &st.overlay_slot
                    && let Some(slot_def) =
                        target.overlay_slots.iter().find(|s| &s.name == slot_name)
                {
                    let raw_ovl = codegen::emit_overlay_z80_asm(
                        program,
                        &analysis,
                        &layout,
                        opts,
                        slot_name,
                        &st.name,
                        slot_def.start,
                        &assembled.symbols,
                    )?;
                    let ovl_asm = peephole::optimize_asm_full(
                        &raw_ovl,
                        &fn_private_ranges,
                        &fn_sb_ranges,
                        &no_bc_fns,
                    );
                    let ovl_assembled = asm::assemble_z80(&ovl_asm, slot_def.start)?;
                    let max_ovl_size =
                        slot_def.end.saturating_sub(slot_def.start) as usize + 1;
                    if ovl_assembled.bytes.len() > max_ovl_size {
                        return Err(format!(
                            "overlay `{}` exceeds slot `{}` budget ({} > {} bytes)",
                            st.name,
                            slot_name,
                            ovl_assembled.bytes.len(),
                            max_ovl_size
                        ));
                    }
                    overlays.push(OverlayOutput {
                        slot_name: slot_name.clone(),
                        state_name: st.name.clone(),
                        origin: slot_def.start,
                        bin: ovl_assembled.bytes,
                        asm_listing: ovl_asm,
                    });
                }
            }
        }
        // Static stack budget verification against `target.stacks`
        if let Some(stack) = target.stacks.first() {
            let budget = stack.end.saturating_sub(stack.start).saturating_add(1);
            if layout.max_stack_bytes > budget {
                return Err(format!(
                    "static stack depth {} bytes exceeds stack `{}` budget ({} bytes)",
                    layout.max_stack_bytes, stack.name, budget
                ));
            }
        }
    }

    Ok(LinkOutput {
        resident_bin: assembled.bytes,
        overlays,
        asm_listing: full_asm,
        symbols: assembled.symbols,
        code_origin: assembled.origin,
        ram_base: layout.ram_base,
        overlay_base: layout.overlay_base,
        ram_end: layout.ram_end,
        ram_bytes: layout.ram_end.saturating_sub(layout.ram_base),
        max_stack_bytes: layout.max_stack_bytes,
        total_tstates: assembled.total_tstates,
        fn_tstates,
        globals: layout.global_entries,
        fn_frames: layout.fn_frames,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use aun_ir::*;

    #[test]
    fn compile_and_link_with_external_sdasz80_ffi() {
        // Build a tiny IR program calling an extern "asm" routine `hal_out_fe(v: u8 in A)`
        let mut prog = IrProgram::default();
        prog.externs.push(IrExternFn {
            name: "hal_out_fe".into(),
            params: vec![("v".into(), IrType::Int(IntType::U8), "A".into())],
            ret: None,
            clobbers: vec!["F".into()],
            requires_di: false,
            requires_banks: vec![],
        });
        prog.functions.push(IrFunction {
            id: 0,
            name: "main".into(),
            param_count: 0,
            is_pub: false,
            locals: vec![],
            ret_ty: IrType::Unit,
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            blocks: vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::ConstInt {
                        dst: 0,
                        ty: IrType::Int(IntType::U8),
                        val: 0x55,
                    },
                    IrInst::ExternCall {
                        dst: None,
                        extern_fn: 0,
                        args: vec![0],
                    },
                ],
                term: Terminator::Return(None),
            }],
            next_val: 1,
        });

        let hal_s = r#"
            .area _CODE
            .globl hal_out_fe
        hal_out_fe:
            out (0xFE), a
            ret
        "#;

        let out = compile_z80_with_asm(&prog, &[hal_s], 0x0100, 0xC000)
            .expect("Z80 compile + link should succeed");
        assert!(out.symbols.contains_key("main"));
        assert!(out.symbols.contains_key("hal_out_fe"));
        assert!(!out.resident_bin.is_empty());
        assert!(out.total_tstates > 0);
        // Peephole should optimize `call hal_out_fe; ret` -> `jp hal_out_fe`
        assert!(out.asm_listing.contains("jp hal_out_fe"));
        // Stack depth accounts for ExternCall
        assert_eq!(out.max_stack_bytes, 4);
    }

    #[test]
    fn codegen_phase1_edge_cases_assemble_cleanly() {
        // Exercises UnOp, Cast, IndexAddr, Assert, comparisons (8/16, signed/unsigned),
        // q8_8::int(), q10_6::int(), and multi-arg function calls.
        let mut prog = IrProgram::default();
        prog.globals.push(IrGlobal {
            name: "ARR".into(),
            ty: IrType::Array {
                elem: Box::new(IrType::Int(IntType::U16)),
                len: 8,
            },
            mutable: true,
            in_rom: false,
            align: 1,
            soa: false,
            space: None,
            init: vec![],
        });

        // fn add2(a: u8, b: u8) -> u8
        prog.functions.push(IrFunction {
            id: 0,
            name: "add2".into(),
            param_count: 2,
            is_pub: false,
            locals: vec![
                LocalSlot {
                    id: 0,
                    name: "a".into(),
                    ty: IrType::Int(IntType::U8),
                    mutable: false,
                },
                LocalSlot {
                    id: 1,
                    name: "b".into(),
                    ty: IrType::Int(IntType::U8),
                    mutable: false,
                },
            ],
            ret_ty: IrType::Int(IntType::U8),
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            blocks: vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::LoadLocal {
                        dst: 0,
                        slot: 0,
                        ty: IrType::Int(IntType::U8),
                    },
                    IrInst::LoadLocal {
                        dst: 1,
                        slot: 1,
                        ty: IrType::Int(IntType::U8),
                    },
                    IrInst::BinOp {
                        dst: 2,
                        op: IrBinOp::Add,
                        mode: ArithMode::Wrapping,
                        ty: IrType::Int(IntType::U8),
                        lhs: 0,
                        rhs: 1,
                        site_id: 0,
                    },
                ],
                term: Terminator::Return(Some(2)),
            }],
            next_val: 3,
        });

        // fn main()
        prog.functions.push(IrFunction {
            id: 1,
            name: "main".into(),
            param_count: 0,
            is_pub: false,
            locals: vec![],
            ret_ty: IrType::Unit,
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            blocks: vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::ConstInt {
                        dst: 0,
                        ty: IrType::Int(IntType::I8),
                        val: -5,
                    },
                    IrInst::UnOp {
                        dst: 1,
                        op: IrUnOp::Neg,
                        mode: ArithMode::Wrapping,
                        ty: IrType::Int(IntType::I8),
                        val: 0,
                        site_id: 0,
                    },
                    IrInst::Cast {
                        dst: 2,
                        val: 1,
                        from_ty: IrType::Int(IntType::I8),
                        to_ty: IrType::Int(IntType::I16),
                    },
                    IrInst::FixedInt {
                        dst: 3,
                        val: 2,
                        base: IntType::I16,
                        frac_bits: 6, // q10_6::int()
                    },
                    IrInst::FixedInt {
                        dst: 4,
                        val: 2,
                        base: IntType::I16,
                        frac_bits: 8, // q8_8::int()
                    },
                    IrInst::BinOp {
                        dst: 5,
                        op: IrBinOp::Lt,
                        mode: ArithMode::Checked,
                        ty: IrType::Int(IntType::I16),
                        lhs: 3,
                        rhs: 4,
                        site_id: 0,
                    },
                    IrInst::UnOp {
                        dst: 6,
                        op: IrUnOp::Not,
                        mode: ArithMode::Wrapping,
                        ty: IrType::Bool,
                        val: 5,
                        site_id: 0,
                    },
                    IrInst::Assert {
                        cond: 6,
                        site_id: 1,
                    },
                    IrInst::AddrGlobal { dst: 7, global: 0 },
                    IrInst::ConstInt {
                        dst: 8,
                        ty: IrType::Int(IntType::U8),
                        val: 3,
                    },
                    IrInst::IndexAddr {
                        dst: 9,
                        base: 7,
                        index: 8,
                        elem_size: 2,
                        len: 8,
                        site_id: 2,
                    },
                    IrInst::LoadPtr {
                        dst: 10,
                        addr: 9,
                        ty: IrType::Int(IntType::U16),
                    },
                    IrInst::Call {
                        dst: Some(11),
                        func: 0,
                        args: vec![8, 8],
                    },
                ],
                term: Terminator::Return(None),
            }],
            next_val: 12,
        });

        let out = compile_z80(&prog).expect("Phase 1 edge cases should compile and assemble");
        assert!(out.symbols.contains_key("main"));
        assert!(out.symbols.contains_key("add2"));
        assert!(out.symbols.contains_key("__aun_panic"));
        assert!(out.fn_tstates.get("main").copied().unwrap_or(0) > 0);
        assert!(out.fn_tstates.get("add2").copied().unwrap_or(0) > 0);
    }

    #[test]
    fn e2e_phase2_state_machine_smc_and_pool_to_z80() {
        let src = r#"
            #[smc]
            machine GamePhase {
                state Title;
                state Playing { frames: u16 };
                transitions {
                    Title   -> Playing;
                    Playing -> Title;
                }
                on Title {
                    fn tick(&mut self) { goto Playing { frames: 0u16 }; }
                }
                on Playing {
                    fn tick(&mut self) { self.frames = self.frames +% 1u16; }
                }
            }

            machine Ball {
                x: q8_8,
                state Inactive;
                state InShooter { power: u8 };
                state InPlay { vx: q8_8 };
                transitions {
                    Inactive  -> InShooter;
                    InShooter -> InPlay | Inactive;
                    InPlay    -> Inactive;
                }
                on InShooter {
                    fn on_enter(&mut self) { self.x = 1.0q8_8; }
                    fn step(&mut self) {
                        goto InPlay { vx: 2.0q8_8 };
                    }
                }
                on InPlay {
                    fn step(&mut self) {
                        self.x = self.x + self.vx;
                    }
                }
            }

            #[layout(soa, align(256))]
            static mut BALLS: Pool<Ball, 4>;

            fn main() {
                let mut g: GamePhase = GamePhase::new();
                g.tick();

                let h: Handle<Ball> = BALLS.alloc(Ball::new());
                let b: &mut Ball = BALLS.get_mut(h);
                b.goto(InShooter { power: 8u8 });
                b.step();
                b.step();
                BALLS.free(h);
            }
        "#;

        let prog = aun_front::compile_str(0, src).expect("Phase 2 frontend should succeed");
        let out = compile_z80(&prog).expect("Phase 2 Z80 compile & assemble should succeed");

        assert!(out.symbols.contains_key("main"));
        assert!(out.symbols.contains_key("GamePhase_disp_tick"));
        assert!(out.symbols.contains_key("Ball_disp_step"));
        assert!(out.symbols.contains_key("Ball__InShooter__step"));
        assert!(out.symbols.contains_key("Ball__InPlay__step"));
        // Verify 256-byte page alignment on BALLS Pool global
        let balls = out.globals.iter().find(|g| g.name == "BALLS").unwrap();
        assert_eq!(balls.addr & 0x00FF, 0, "BALLS must be 256B aligned");
        // Verify SMC direct JP patching emitted for singleton GamePhase, vtable for pooled Ball
        assert!(out.asm_listing.contains("ld (GamePhase_disp_tick+1), hl"));
        assert!(!out.resident_bin.is_empty());
        assert!(out.total_tstates > 0);
    }

    #[test]
    fn e2e_phase3_pc88_target_banks_isr_and_overlays() {
        let src = r#"
            target pc88sr : z80 {
                space MainRam   @ 0x0100..0xBFFF;
                stack MainStack @ 0xBE00..0xBFFF;

                bank_group HighWindow @ 0xC000..0xFFFF {
                    bank HighMainRam @ 0xC000..0xEFFF;
                    bank VramAlu     @ 0xC000..0xFFFF requires di;
                    default HighMainRam;
                    switch(to) {
                        HighMainRam => { io_out(0x35, 0x00); io_out(0x5F, 0x00); }
                        VramAlu     => { io_out(0x35, 0x80); }
                    }
                }

                interrupt_controller Pc88Int {
                    disable => asm!("di");
                    enable  => asm!("ei");
                    source Vrtc @ im2(0xF302) { rearm => io_out(0xE4, 0x07); }
                }

                overlay_slot SceneCode @ 0x8000..0x9FFF {
                    load(id: OverlayId) => hal_load_overlay(id, 0x8000);
                }
            }

            static mut FRAME_COUNT: u8 in HighMainRam;

            extern "asm" {
                fn gfx_clear_alu(col: u8 in A) requires bank(VramAlu), di clobbers(AF);
                fn hal_load_overlay(id: u8 in A) clobbers(AF, BC, DE, HL);
            }

            #[interrupt(Vrtc)]
            fn on_vrtc() {
                FRAME_COUNT = FRAME_COUNT +% 1u8;
            }

            machine App {
                #[overlay(SceneCode)] state Title { cursor: u8 };
                #[overlay(SceneCode)] state Stage1 { lives: u8 };
                transitions {
                    Title  -> Stage1;
                    Stage1 -> Title;
                }
                on Title {
                    fn update(&mut self) {
                        self.cursor = self.cursor +% 1u8;
                    }
                }
                on Stage1 {
                    fn update(&mut self) {
                        self.lives = self.lives -| 1u8;
                    }
                }
            }

            fn main() {
                let mut app: App = App::new();
                app.update();
                app.goto(Stage1 { lives: 3u8 });
                app.update();
                with bank(VramAlu) {
                    gfx_clear_alu(0u8);
                }
            }
        "#;

        let hal_s = "gfx_clear_alu:\n    ret\nhal_load_overlay:\n    ret\n";
        let prog = aun_front::compile_str(0, src).expect("Phase 3 frontend should succeed");
        let out = compile_z80_with_asm(&prog, &[hal_s], 0x0100, 0xC000)
            .expect("Phase 3 Z80 compile + overlay link should succeed");

        // Banked global sits in HighMainRam (0xC000..0xEFFF), while unbanked Uchiguri RAM sits
        // strictly below HighWindow (0xC000) and MainStack (0xBE00) so VramAlu never unmaps locals!
        let fc = out.globals.iter().find(|g| g.name == "FRAME_COUNT").unwrap();
        assert!((0xC000..=0xEFFF).contains(&fc.addr));
        assert!(out.ram_end <= 0xBE00, "Uchiguri RAM must stay below MainStack & HighWindow");

        assert!(out.asm_listing.contains("di\n    ld a, 0x80\n    out (0x35), a"));
        assert!(out.asm_listing.contains("reti"));
        // Transition to overlay state invokes hal_load_overlay from resident code
        assert!(out.asm_listing.contains("call hal_load_overlay"));
        // Overlay methods must NOT be in resident_bin; they live in their own overlay binaries
        assert!(!out.symbols.contains_key("App__Title__update"));
        assert!(!out.symbols.contains_key("App__Stage1__update"));
        assert_eq!(out.overlays.len(), 2);
        assert_eq!(out.overlays[0].state_name, "Title");
        assert_eq!(out.overlays[0].origin, 0x8000);
        assert!(out.overlays[0].bin.len() > 4, "Title overlay must contain real method code");
        assert!(out.overlays[0].asm_listing.contains("App__Title__update:"));
        assert_eq!(out.overlays[1].state_name, "Stage1");
        assert_eq!(out.overlays[1].origin, 0x8000);
        assert!(out.overlays[1].bin.len() > 4, "Stage1 overlay must contain real method code");
        assert!(out.overlays[1].asm_listing.contains("App__Stage1__update:"));
    }

    #[test]
    fn mdxnt_drv_ssa_roundtrip_verifies_cleanly() {
        let src = include_str!("../../../bench/mdxnt/src/mdxnt_drv.aun");
        let mut prog = aun_front::compile_str(0, src).expect("mdxnt_drv compile should succeed");
        aun_ir::opt::optimize_program(&mut prog);
        aun_ir::verify::verify_program(&prog).expect("optimized mdxnt_drv MIR must verify");

        let mut phi_count = 0usize;
        for f in &mut prog.functions {
            aun_ir::ssa::mem2reg(f);
            for bb in &f.blocks {
                phi_count += bb
                    .insts
                    .iter()
                    .filter(|i| matches!(i, IrInst::Phi { .. }))
                    .count();
            }
        }
        assert!(
            phi_count > 0,
            "mem2reg should insert Phi nodes on mdxnt_drv.aun"
        );
        aun_ir::verify::verify_program(&prog).expect("mdxnt_drv SSA MIR must verify");

        for f in &mut prog.functions {
            aun_ir::sccp::sccp(f);
            aun_ir::gvn::gvn(f);
            aun_ir::licm::licm(f);
            aun_ir::osr::osr(f);
            aun_ir::opt::dce(f);
        }
        aun_ir::verify::verify_program(&prog).expect("mdxnt_drv SSA+SCCP+GVN+LICM+OSR MIR must verify");

        for f in &mut prog.functions {
            aun_ir::ssa::destruct_ssa(f);
            assert!(
                !f.blocks
                    .iter()
                    .any(|b| b.insts.iter().any(|i| matches!(i, IrInst::Phi { .. }))),
                "fn `{}` still contains Phi after destruct_ssa",
                f.name
            );
        }
        aun_ir::verify::verify_program(&prog).expect("mdxnt_drv MIR after destruct_ssa must verify");

        let mut prog2 = aun_front::compile_str(0, src).expect("mdxnt_drv compile should succeed");
        aun_ir::opt::optimize_program(&mut prog2);
        aun_ir::verify::verify_program(&prog2).expect("mdxnt_drv optimize_program MIR must verify");
    }
}
