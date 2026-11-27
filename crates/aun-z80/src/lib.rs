//! Z80 backend: WPO-ABI, instruction selection, *Uchiguri* static RAM overlay,
//! peephole optimizer, built-in SDASZ80-compatible assembler, and linker.

pub mod analysis;
pub mod asm;
pub mod codegen;
pub mod jumptable;
pub mod lir;
pub mod loop_opt;
pub mod outliner;
pub mod overlay;
pub mod peephole;
pub mod regalloc;
pub mod sdasz80;
pub mod uchiguri;

use aun_ir::IrProgram;
pub use overlay::ExternalAsm;
use std::collections::{HashMap, HashSet};
use uchiguri::{FnRamFrame, GlobalRamEntry};

/// A compiled code overlay binary linked at its space/`overlay_slot` base address.
#[derive(Debug, Clone)]
pub struct OverlayOutput {
    pub id: u8,
    pub slot_name: String,
    pub state_name: String,
    pub parent_state: Option<String>,
    pub space_name: String,
    pub space_id: u8,
    pub origin: u16,
    pub budget_end: u16,
    pub bin: Vec<u8>,
    pub asm_listing: String,
    pub asm_files: Vec<String>,
    pub symbols: HashMap<String, u16>,
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
    let is_release = !opts.check_bounds && !opts.check_transitions && !opts.check_overflow;
    loop_opt::optimize_z80_program(&mut opt_prog, is_release);
    uchiguri::split_isr_shared_reentrant_fns(&mut opt_prog);
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
    let (sdas, extra_stack, ram_bytes) =
        sdasz80::emit_sdasz80_with_metrics(module_name, &opt_asm, program, &layout);
    Ok((sdas, ram_bytes, layout.max_stack_bytes.saturating_add(extra_stack)))
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
        if !fa.is_some_and(|a| a.has_inline_asm) {
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
        }
        // Always register every top-level function symbol so `partition_fn_spans` sees
        // every function boundary even when a callee has zero private RAM ranges.
        fn_private_ranges.insert(analysis::z80_sym(&frame.name), ranges);
    }
    fn_private_ranges
}

pub fn compile_z80_with_options(
    program: &IrProgram,
    external_asm_sources: &[&str],
    opts: CompileOptions,
) -> Result<LinkOutput, String> {
    let ext_asms: Vec<ExternalAsm> = external_asm_sources
        .iter()
        .enumerate()
        .map(|(i, src)| ExternalAsm {
            name: format!("ext_{i}.s"),
            source: (*src).to_string(),
            owner_state: None,
        })
        .collect();
    compile_z80_with_external_asms(program, &ext_asms, opts)
}

/// Compiles `program` with optional state-owned external `.s` modules, running
/// Call-graph LCA *Uchiguri* overlay partitioning and top-down multi-pass linking.
pub fn compile_z80_with_external_asms(
    program: &IrProgram,
    external_asms: &[ExternalAsm],
    opts: CompileOptions,
) -> Result<LinkOutput, String> {
    let mut opt_prog = program.clone();
    aun_ir::opt::optimize_program(&mut opt_prog);
    let is_release = !opts.check_bounds && !opts.check_transitions && !opts.check_overflow;
    loop_opt::optimize_z80_program(&mut opt_prog, is_release);
    uchiguri::split_isr_shared_reentrant_fns(&mut opt_prog);
    let program = &opt_prog;
    if cfg!(debug_assertions)
        && let Err(errs) = aun_ir::verify::verify_program(program)
    {
        panic!("MIR verification failed:\n{}", errs.join("\n"));
    }

    let plan = overlay::partition_program(program, external_asms)?;
    let analysis = analysis::ProgramAnalysis::analyze(program);
    let layout = uchiguri::allocate_uchiguri_with_analysis(program, &analysis, opts.ram_base)?;
    let fn_private_ranges = build_fn_private_ranges(program, &analysis, &layout);
    let fn_sb_ranges = build_fn_sb_ranges(&analysis, &layout);
    let no_bc_fns = analysis.no_bc_fns(program);

    let build_resident = |resolved_syms: &HashMap<String, u16>,
                          resolved_spans: &[(u16, u16)],
                          pass1: bool|
     -> Result<(String, asm::AsmOutput, u16), String> {
        let raw_asm = codegen::emit_z80_asm_with_overlays(
            program,
            &analysis,
            &layout,
            opts,
            &plan,
            resolved_syms,
            resolved_spans,
            pass1,
        )?;
        let opt_asm =
            peephole::optimize_asm_full(&raw_asm, &fn_private_ranges, &fn_sb_ranges, &no_bc_fns);
        let mut lir_items = lir::parse_zilog_to_lir(&opt_asm);
        jumptable::compress_jump_tables(program, &mut lir_items);
        outliner::outline_lir(&mut lir_items);
        let extra_stack = outliner::max_outlined_stack_bytes(&lir_items);
        lir::relax_jp_to_jr(&mut lir_items);
        let mut full_asm = lir::format_zilog_lir(&lir_items);
        for (ei, ext) in external_asms.iter().enumerate() {
            if plan.ext_asm_owner.get(ei) == Some(&overlay::OverlayOwner::Resident) {
                full_asm.push_str("\n        .area _CODE\n");
                full_asm.push_str(&ext.source);
            }
        }
        let assembled = asm::assemble_z80_with_ram(&full_asm, opts.code_origin, layout.ram_end)?;
        Ok((full_asm, assembled, extra_stack))
    };

    // Pass 1: assemble Resident (emitting unique placeholders for unresolved overlay calls)
    let empty_syms = HashMap::new();
    let (mut full_asm, mut assembled, mut extra_stack) =
        build_resident(&empty_syms, &[], true)?;
    let mut overlays = Vec::new();

    if !plan.blobs.is_empty() {
        let primary_space_name = program
            .target
            .as_ref()
            .and_then(|t| t.spaces.first().map(|s| s.name.clone()))
            .unwrap_or_else(|| "MainRam".to_string());
        let mut space_global_hwm: HashMap<String, u16> = HashMap::new();
        for g in &layout.global_entries {
            if let Some(ref sp) = g.space {
                let end = g.addr.wrapping_add(g.size);
                space_global_hwm
                    .entry(sp.clone())
                    .and_modify(|cur| *cur = (*cur).max(end))
                    .or_insert(end);
            }
        }

        let space_bounds = |space_name: &str| -> (u16, u16) {
            if let Some(target) = &program.target {
                if let Some(sp) = target.spaces.iter().find(|s| s.name == space_name) {
                    return (sp.start, sp.end);
                }
                for bg in &target.bank_groups {
                    if let Some(bk) = bg.banks.iter().find(|b| b.name == space_name) {
                        return (bk.start, bk.end);
                    }
                }
            }
            (0x0000, 0xFFFF)
        };

        let mut res_end = opts.code_origin.wrapping_add(assembled.bytes.len() as u16);
        let max_iters = 2;
        for _iter in 0..max_iters {
            let mut resolved_overlay_syms: HashMap<String, u16> = HashMap::new();
            let mut resolved_blob_spans: Vec<(u16, u16)> = vec![(0, 0); plan.blobs.len()];
            let mut blob_syms: Vec<HashMap<String, u16>> = vec![HashMap::new(); plan.blobs.len()];
            let mut pass_overlays = Vec::with_capacity(plan.blobs.len());

            for (bi, blob) in plan.blobs.iter().enumerate() {
                let mdef = &program.machines[blob.machine_id as usize];
                let (sp_start, sp_end) = space_bounds(&blob.space_name);

                let (origin, budget_end) = if let Some((s, e)) = blob.origin_hint {
                    (s, e)
                } else {
                    let anc_same_space = mdef
                        .ancestors(blob.state_id)
                        .into_iter()
                        .rev()
                        .skip(1)
                        .filter_map(|aid| plan.state_blob_idx.get(&(blob.machine_id, aid)).copied())
                        .find(|&abi| plan.blobs[abi].space_name == blob.space_name);
                    let origin = if let Some(abi) = anc_same_space {
                        let (ao, al) = resolved_blob_spans[abi];
                        ao.wrapping_add(al)
                    } else if blob.space_name == primary_space_name {
                        res_end
                    } else {
                        sp_start.max(
                            space_global_hwm
                                .get(&blob.space_name)
                                .copied()
                                .unwrap_or(sp_start),
                        )
                    };

                    let mut budget_end = sp_end;
                    if layout.ram_base >= origin && layout.ram_base <= sp_end {
                        budget_end = budget_end.min(layout.ram_base.saturating_sub(1));
                    }
                    if let Some(target) = &program.target {
                        for st in &target.stacks {
                            if st.start >= origin && st.start <= sp_end {
                                budget_end = budget_end.min(st.start.saturating_sub(1));
                            }
                        }
                        for bg in &target.bank_groups {
                            let is_bank_of_bg = bg.banks.iter().any(|b| b.name == blob.space_name);
                            if !is_bank_of_bg && bg.start >= origin && bg.start <= sp_end {
                                budget_end = budget_end.min(bg.start.saturating_sub(1));
                            }
                        }
                    }
                    (origin, budget_end)
                };

                // Build visible_syms = resident_symbols U all ancestor boundary blob symbols
                let mut visible_syms = assembled.symbols.clone();
                let mut inherited_names: HashSet<String> = HashSet::new();
                for aid in mdef.ancestors(blob.state_id) {
                    if aid != blob.state_id
                        && let Some(&abi) = plan.state_blob_idx.get(&(blob.machine_id, aid))
                    {
                        for (k, &v) in &blob_syms[abi] {
                            visible_syms.insert(k.clone(), v);
                            inherited_names.insert(k.clone());
                        }
                    }
                }

                let raw_ovl = codegen::emit_overlay_blob_z80_asm(
                    program,
                    &analysis,
                    &layout,
                    opts,
                    &plan,
                    blob,
                    external_asms,
                    origin,
                    &visible_syms,
                )?;
                let ovl_asm = peephole::optimize_asm_full(
                    &raw_ovl,
                    &fn_private_ranges,
                    &fn_sb_ranges,
                    &no_bc_fns,
                );
                let ovl_assembled = asm::assemble_z80(&ovl_asm, origin)?;
                let size = ovl_assembled.bytes.len();
                let max_ovl_size = budget_end.saturating_sub(origin) as usize + 1;
                if origin > budget_end || size > max_ovl_size {
                    return Err(format!(
                        "overlay `{}` exceeds `{}` budget (0x{:04X}, {} > {} bytes, limit 0x{:04X})",
                        blob.state_name,
                        blob.slot_name.as_deref().unwrap_or(&blob.space_name),
                        origin,
                        size,
                        max_ovl_size,
                        budget_end
                    ));
                }

                resolved_blob_spans[bi] = (origin, size as u16);
                let mut own_syms = HashMap::new();
                for (k, &v) in &ovl_assembled.symbols {
                    if !assembled.symbols.contains_key(k) && !inherited_names.contains(k) {
                        own_syms.insert(k.clone(), v);
                    }
                }
                for &fid in &blob.fn_ids {
                    let f = &program.functions[fid];
                    let sym = analysis::z80_sym(&f.name);
                    if let Some(&addr) = ovl_assembled.symbols.get(&sym) {
                        resolved_overlay_syms.insert(sym.clone(), addr);
                        own_syms.insert(sym, addr);
                    }
                }
                blob_syms[bi] = own_syms.clone();

                let parent_state = blob
                    .parent_state_id
                    .map(|pid| format!("{}::{}", mdef.name, mdef.states[pid as usize].name));
                let asm_files: Vec<String> = blob
                    .ext_asm_indices
                    .iter()
                    .filter_map(|&ei| external_asms.get(ei).map(|e| e.name.clone()))
                    .collect();
                pass_overlays.push(OverlayOutput {
                    id: blob.blob_id,
                    slot_name: blob
                        .slot_name
                        .clone()
                        .unwrap_or_else(|| blob.space_name.clone()),
                    state_name: blob.state_name.clone(),
                    parent_state,
                    space_name: blob.space_name.clone(),
                    space_id: blob.space_id,
                    origin,
                    budget_end,
                    bin: ovl_assembled.bytes,
                    asm_listing: ovl_asm,
                    asm_files,
                    symbols: own_syms,
                });
            }

            // Pass 3: re-assemble Resident with concrete overlay addresses & LoadOp (dst, size)
            let (new_full_asm, new_assembled, new_extra_stack) =
                build_resident(&resolved_overlay_syms, &resolved_blob_spans, false)?;
            let new_res_end = opts.code_origin.wrapping_add(new_assembled.bytes.len() as u16);
            full_asm = new_full_asm;
            assembled = new_assembled;
            extra_stack = new_extra_stack;
            overlays = pass_overlays;
            if new_res_end == res_end {
                break;
            }
            res_end = new_res_end;
        }
    }

    // Compute per-function T-states using sorted Resident function/extern symbol boundaries
    let mut top_syms: Vec<(u16, String)> = Vec::new();
    for f in &program.functions {
        let sym = analysis::z80_sym(&f.name);
        if !f.is_test
            && !plan.is_fn_in_overlay(f.id as usize)
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
    if let Some(&addr) = assembled.symbols.get("__aun_jtbl_dispatch") {
        top_syms.push((addr, "__aun_jtbl_dispatch".to_string()));
    }
    for (sym, &addr) in &assembled.symbols {
        if sym.starts_with("__aun_out_") {
            top_syms.push((addr, sym.clone()));
        }
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

    let max_stack_bytes = layout.max_stack_bytes.saturating_add(extra_stack);
    if let Some(target) = &program.target
        && let Some(stack) = target.stacks.first()
    {
        let budget = stack.end.saturating_sub(stack.start).saturating_add(1);
        if max_stack_bytes > budget {
            return Err(format!(
                "static stack depth {} bytes exceeds stack `{}` budget ({} bytes)",
                max_stack_bytes, stack.name, budget
            ));
        }
    }

    let final_ram_end = assembled.ram_end.max(layout.ram_end);
    Ok(LinkOutput {
        resident_bin: assembled.bytes,
        overlays,
        asm_listing: full_asm,
        symbols: assembled.symbols,
        code_origin: assembled.origin,
        ram_base: layout.ram_base,
        overlay_base: layout.overlay_base,
        ram_end: final_ram_end,
        ram_bytes: final_ram_end.saturating_sub(layout.ram_base),
        max_stack_bytes,
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
            clobbers: Some(vec!["F".into()]),
            requires_di: false,
            requires_banks: vec![],
            fixed_addr: None,
            no_callback: false,
            owner_state: None,
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
            isr_using: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            owner_state: None,
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
            volatile: false,
            fixed_addr: None,
            is_extern: false,
            align: 1,
            soa: false,
            space: None,
            owner_state: None,
            init: vec![],
        });

        // fn add2(a: u8, b: u8) -> u8
        prog.functions.push(IrFunction {
            id: 0,
            name: "add2".into(),
            param_count: 2,
            is_pub: true,
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
            isr_using: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            owner_state: None,
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
            isr_using: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            owner_state: None,
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
                        volatile: false,
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
        assert!(
            out.asm_listing.contains("ld (GamePhase_disp_tick+1), hl")
                || out.asm_listing.contains("ld (GamePhase_disp_tick + 1), hl"),
            "expected SMC JP patch in:\n{}",
            out.asm_listing
        );
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

    #[test]
    fn mem_write_fixed_addr_and_extern_static_sdasz80() {
        let src = r#"
            struct GameState {
                lives: u8,
                score: u16,
            }

            static mut G_STATE: GameState @ 0xEC00;

            extern "asm" {
                static mut _EXT_VAL: u16;
            }

            pub fn update_hw(v: u8, w: u16) {
                G_STATE.lives = v;
                G_STATE.score = w;
                _EXT_VAL = _EXT_VAL +% 1u16;
                mem_write_u8(0xF3C8, v);
                mem_write_u16(0xF3CA, w);
            }
        "#;
        let prog = aun_front::compile_str(0, src).expect("compile should succeed");
        let layout = uchiguri::allocate_uchiguri(&prog, 0xC000).expect("layout should succeed");
        assert_eq!(
            layout.overlay_base, layout.ram_base,
            "fixed-address and extern statics must consume 0 bytes of __aun_ram globals"
        );
        assert!(layout
            .global_entries
            .iter()
            .any(|g| g.name == "G_STATE" && g.addr == 0xEC00));
        assert!(!layout
            .global_entries
            .iter()
            .any(|g| g.name == "_EXT_VAL"));

        let (sdas, _ram_bytes, _max_stack) = compile_z80_sdasz80(
            "hw_test",
            &prog,
            CompileOptions {
                ram_base: 0xC000,
                ..CompileOptions::release()
            },
        )
        .expect("compile_z80_sdasz80 should succeed");

        assert!(sdas.contains(".globl  _EXT_VAL"), "missing .globl _EXT_VAL:\n{sdas}");
        assert!(sdas.contains("(0xEC00)"), "expected direct access to 0xEC00:\n{sdas}");
        assert!(sdas.contains("(0xF3C8)"), "expected direct write to 0xF3C8:\n{sdas}");
        assert!(sdas.contains("(0xF3CA)"), "expected direct write to 0xF3CA:\n{sdas}");
    }

    #[test]
    fn trims_unused_aun_ram_tail_and_omits_empty_data_area() {
        // 1. All locals register-promoted + fixed address: __aun_ram completely omitted!
        let src_zero = r#"
            static mut HW: u8 @ 0xF3C8;
            pub fn bump(v: u8) {
                HW = v +% 1u8;
            }
        "#;
        let prog0 = aun_front::compile_str(0, src_zero).unwrap();
        let (sdas0, ram0, _) = compile_z80_sdasz80("trim0", &prog0, CompileOptions::release()).unwrap();
        assert_eq!(ram0, 0, "expected 0 trimmed RAM bytes:\n{sdas0}");
        assert!(!sdas0.contains("__aun_ram:"), "unexpected __aun_ram in:\n{sdas0}");

        // 2. Global + unused scratch tail: only live offsets retained; array base Ld16Imm preserves array len
        let src_arr = r#"
            static mut ARR: [u8; 4] = [0u8; 4];
            pub fn write_idx(i: u8, val: u8) {
                ARR[i] = val;
            }
        "#;
        let prog1 = aun_front::compile_str(0, src_arr).unwrap();
        let (sdas1, ram1, _) = compile_z80_sdasz80("trim_arr", &prog1, CompileOptions::release()).unwrap();
        assert!(ram1 >= 4, "array bounds must be preserved by Ld16Imm scan (got {ram1}):\n{sdas1}");
    }

    #[test]
    fn folds_rom_addr_const_return_direct_u8_to_u16_cast_and_branchless_cmp() {
        let src = r#"
            const TBL: [u8; 4] = [10u8, 20u8, 30u8, 40u8];
            extern "asm" {
                fn use_ptr(p: u16 in HL) clobbers(AF, HL);
            }

            pub fn fold_addr_and_cast(idx: u8, off: u16) -> u8 {
                let mut i: u8 = idx;
                i = i +% 1u8;
                let base_plus: u16 = addr_of(TBL) +% off;
                use_ptr(base_plus);
                i = i +% 1u8;
                let mut s: u8 = 0u8;
                while s < 2u8 {
                    let wide: u16 = (i as u16) +% off;
                    let lt_flag: bool = i < 3u8;
                    let ge_flag: bool = wide >= 10u16;
                    if lt_flag && ge_flag {
                        return i;
                    }
                    s = s +% 1u8;
                    i = i +% 1u8;
                }
                0u8
            }
        "#;
        let prog = aun_front::compile_str(0, src).unwrap();
        let (sdas, _ram, _) = compile_z80_sdasz80("fold3", &prog, CompileOptions::release()).unwrap();
        assert!(
            sdas.contains("ld l, c\n        ld h, #0x00")
                || sdas.contains("ld l, b\n        ld h, #0x00"),
            "unsigned u8->u16 cast should read reg directly without A:\n{sdas}"
        );
        assert!(
            !sdas.contains("xor a\n        xor a"),
            "constant return 0 must not emit double xor a:\n{sdas}"
        );
        assert!(
            sdas.contains("rla"),
            "non-fused < / >= comparison should emit branchless rla:\n{sdas}"
        );
    }

    #[test]
    fn keeps_raw_u16_literals_distinct_from_aun_ram_in_sdasz80() {
        let src = r#"
            static mut ACC: u16 = 0u16;
            struct Cmd { addr: u16, val: u8 }
            extern "asm" {
                fn send_cmd(addr: u16 in HL, val: u8 in A) clobbers(AF, HL);
            }

            pub fn run_raw_and_ram(v: u8) -> u16 {
                let c: Cmd = Cmd { addr: 0xC000u16, val: v };
                ACC = ACC +% (c.val as u16);
                send_cmd(c.addr, c.val);
                mem_write_u8(0xC002u16, v);
                ACC
            }
        "#;
        let prog = aun_front::compile_str(0, src).unwrap();
        let (sdas, ram_bytes, _) = compile_z80_sdasz80(
            "raw_u16",
            &prog,
            CompileOptions {
                ram_base: 0xC000,
                ..CompileOptions::release()
            },
        )
        .expect("compile_z80_sdasz80 should succeed");
        assert!(ram_bytes >= 2, "expected ACC in __aun_ram, got {ram_bytes}:\n{sdas}");
        assert!(
            sdas.contains("#0xC000"),
            "raw literal 0xC000 inside [ram_base, ram_end) must remain #0xC000:\n{sdas}"
        );
        assert!(
            sdas.contains("(0xC002)"),
            "raw MMIO address 0xC002 must remain (0xC002):\n{sdas}"
        );
        assert!(
            sdas.contains("__aun_ram"),
            "static ACC / locals must still use __aun_ram:\n{sdas}"
        );
        assert!(
            !sdas.contains("#__aun_ram"),
            "user literal 0xC000 must not be rewritten to #__aun_ram:\n{sdas}"
        );
    }

    #[test]
    fn preserves_inline_asm_tail_call_after_push() {
        let src = r#"
            extern "asm" {
                fn _callee();
            }

            pub fn wrapper(x: u16) {
                asm!("push hl; call _callee");
            }
        "#;
        let prog = aun_front::compile_str(0, src).unwrap();
        let (sdas, _, _) = compile_z80_sdasz80("issue8", &prog, CompileOptions::release())
            .expect("compile_z80_sdasz80 should succeed");
        assert!(
            sdas.contains("call") && sdas.contains("_callee") && sdas.contains("ret"),
            "inline asm `push hl; call _callee` before ret must stay `call _callee` + `ret`:\n{sdas}"
        );
        assert!(
            !sdas.contains("jp _callee") && !sdas.contains("jp      _callee"),
            "inline asm call after push hl must NOT be tail-called to jp:\n{sdas}"
        );
    }

    #[test]
    fn direct_aggregate_init_memset_const_mul16_and_addr_folding() {
        let src = r#"
            const LUT_A: [u8; 4] = [10u8, 20u8, 30u8, 40u8];
            const LUT_B: [u8; 4] = [1u8, 2u8, 3u8, 4u8];

            extern "asm" {
                fn _take2(a: u16 in hl, b: u16 in de);
            }

            pub fn init_line_and_mul(row: u16, idx: u8) -> u16 {
                let mut line: [u8; 44] = [b' '; 44];
                line[idx] = LUT_A[0];
                _take2(addr_of(LUT_A), addr_of(LUT_B));
                row *% 120u16 +% (line[idx] as u16)
            }
        "#;
        let prog = aun_front::compile_str(0, src).expect("Issue #11 sample should compile");
        let (sdas, ram_bytes, _max_stack) =
            compile_z80_sdasz80("issue11", &prog, CompileOptions::release())
                .expect("compile_z80_sdasz80 should succeed");
        // Single 44-byte allocation (not 88 bytes double-allocated!)
        assert!(
            ram_bytes <= 56,
            "expected single-allocated 44B array without double temp buffer (got {ram_bytes}B):\n{sdas}"
        );
        // Compact djnz memset loop
        assert!(
            sdas.contains("djnz"),
            "expected compact djnz memset for [b' '; 44]:\n{sdas}"
        );
        // No dead `ld hl, #0x0078` spill in constant 16-bit multiply
        assert!(
            !sdas.contains("0x0078"),
            "const 120u16 in row *% 120u16 should fold directly to shift-add:\n{sdas}"
        );
    }
}
