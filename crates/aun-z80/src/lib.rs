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
pub mod wcet;

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
    pub aligned_size: usize,
    pub asm_listing: String,
    pub asm_files: Vec<String>,
    pub symbols: HashMap<String, u16>,
}

fn align_up_usize(val: usize, align: u16) -> usize {
    if align <= 1 {
        val
    } else {
        let a = align as usize;
        val.div_ceil(a) * a
    }
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
    pub fn_wcet_tstates: HashMap<String, u32>,
    pub globals: Vec<GlobalRamEntry>,
    pub fn_frames: Vec<FnRamFrame>,
}

/// Compilation and check-elimination options for `aun-z80`.
#[derive(Debug, Clone)]
pub struct CompileOptions {
    pub release: bool,
    pub check_bounds: bool,
    pub check_transitions: bool,
    pub check_overflow: bool,
    pub check_bank: bool,
    pub code_origin: u16,
    pub ram_base: u16,
    pub auto_ram_base: bool,
    pub overlay_load_align: u16,
    pub relocatable_ram: bool,
    pub symbol_prefix: Option<String>,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self::debug()
    }
}

impl CompileOptions {
    pub fn debug() -> Self {
        Self {
            release: false,
            check_bounds: true,
            check_transitions: true,
            check_overflow: true,
            check_bank: true,
            code_origin: 0x0100,
            ram_base: 0xC000,
            auto_ram_base: false,
            overlay_load_align: 1,
            relocatable_ram: false,
            symbol_prefix: None,
        }
    }

    pub fn release() -> Self {
        Self {
            release: true,
            check_bounds: false,
            check_transitions: false,
            check_overflow: false,
            check_bank: false,
            code_origin: 0x0100,
            ram_base: 0xC000,
            auto_ram_base: false,
            overlay_load_align: 1,
            relocatable_ram: false,
            symbol_prefix: None,
        }
    }
}

pub(crate) fn fn_export_alias(f: &aun_ir::IrFunction, symbol_prefix: Option<&str>) -> Option<String> {
    let sym = analysis::z80_sym(&f.name);
    if let Some(ref exp) = f.export_name {
        return (exp != &sym).then(|| exp.clone());
    }
    if (f.is_pub || f.name == "main")
        && let Some(pfx) = symbol_prefix
        && !pfx.is_empty()
        && !sym.starts_with(pfx)
    {
        return Some(format!("{pfx}{sym}"));
    }
    None
}

pub(crate) fn global_export_alias(g: &aun_ir::IrGlobal, symbol_prefix: Option<&str>) -> Option<String> {
    if g.is_extern {
        return None;
    }
    if let Some(ref exp) = g.export_name {
        return (exp != &g.name).then(|| exp.clone());
    }
    if !g.name.starts_with("__aun_")
        && g.owner_state.is_none()
        && let Some(pfx) = symbol_prefix
        && !pfx.is_empty()
        && !g.name.starts_with(pfx)
    {
        return Some(format!("{pfx}{}", g.name));
    }
    None
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
    mut opts: CompileOptions,
) -> Result<(String, u16, u16), String> {
    opts.relocatable_ram = true;
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
    let raw_asm = codegen::emit_z80_asm_with_analysis(program, &analysis, &layout, opts.clone())?;
    let fn_private_ranges = build_fn_private_ranges(program, &analysis, &layout);
    let fn_sb_ranges = build_fn_sb_ranges(&analysis, &layout);
    let no_bc_fns = analysis.no_bc_fns(program);
    let opt_asm = peephole::optimize_asm_full(&raw_asm, &fn_private_ranges, &fn_sb_ranges, &no_bc_fns);
    let (sdas, extra_stack, ram_bytes) = sdasz80::emit_sdasz80_with_options(
        module_name,
        &opt_asm,
        program,
        &layout,
        opts.symbol_prefix.as_deref(),
    );
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
            is_startup: false,
        })
        .collect();
    compile_z80_with_external_asms(program, &ext_asms, opts)
}

/// Compiles `program` with optional state-owned external `.s` modules, running
/// Call-graph LCA *Uchiguri* overlay partitioning and top-down multi-pass linking.
pub fn compile_z80_with_external_asms(
    program: &IrProgram,
    external_asms: &[ExternalAsm],
    mut opts: CompileOptions,
) -> Result<LinkOutput, String> {
    if opts.overlay_load_align != 0 && !opts.overlay_load_align.is_power_of_two() {
        return Err(format!(
            "invalid overlay_load_align `{}` (must be a power of two)",
            opts.overlay_load_align
        ));
    }
    let ovl_align = opts.overlay_load_align.max(1);

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

    let mut implicit_global_externs: HashSet<String> = HashSet::new();
    for ext in &program.externs {
        implicit_global_externs.insert(ext.name.clone());
    }
    for g in &program.globals {
        if g.is_extern {
            implicit_global_externs.insert(g.name.clone());
        }
    }

    let pfx = opts.symbol_prefix.as_deref();
    let mut fn_aliases: HashMap<String, String> = HashMap::new();
    for f in &program.functions {
        if !f.is_test
            && !f.blocks.is_empty()
            && !plan.is_fn_in_overlay(f.id as usize)
            && let Some(alias) = fn_export_alias(f, pfx)
        {
            fn_aliases.insert(analysis::z80_sym(&f.name), alias);
        }
    }
    let mut rom_global_aliases: HashMap<String, String> = HashMap::new();
    for (gid, g) in program.globals.iter().enumerate() {
        if g.in_rom
            && !g.is_extern
            && plan.rom_global_owner.get(gid) == Some(&overlay::OverlayOwner::Resident)
            && let Some(alias) = global_export_alias(g, pfx)
        {
            rom_global_aliases.insert(g.name.clone(), alias);
        }
    }
    let has_resident_main = program.functions.iter().any(|f| {
        !f.is_test
            && !f.blocks.is_empty()
            && f.name == "main"
            && !plan.is_fn_in_overlay(f.id as usize)
    });
    let forbidden_calls = outliner::collect_stack_cleanup_extern_syms(program);
    let no_bc_fns = analysis.no_bc_fns(program);

    let assemble_resident = |cur_layout: &uchiguri::RamLayout,
                             cur_opts: &CompileOptions,
                             private_ranges: &HashMap<String, Vec<(u16, u16)>>,
                             sb_ranges: &HashMap<String, (u16, u16)>,
                             resolved_syms: &HashMap<String, u16>,
                             resolved_spans: &[(u16, u16)],
                             pass1: bool|
     -> Result<(String, asm::AsmOutput, u16), String> {
        let mut ram_global_aliases: Vec<(String, u16)> = Vec::new();
        for entry in &cur_layout.global_entries {
            if let Some(g) = program.globals.get(entry.global_id as usize)
                && !g.in_rom
                && let Some(alias) = global_export_alias(g, cur_opts.symbol_prefix.as_deref())
            {
                ram_global_aliases.push((alias, entry.addr));
            }
        }
        let raw_asm = codegen::emit_z80_asm_with_overlays(
            program,
            &analysis,
            cur_layout,
            cur_opts.clone(),
            &plan,
            resolved_syms,
            resolved_spans,
            pass1,
        )?;
        let opt_asm = peephole::optimize_asm_full(&raw_asm, private_ranges, sb_ranges, &no_bc_fns);
        let mut lir_items = lir::parse_zilog_to_lir(&opt_asm);
        jumptable::compress_jump_tables(program, &mut lir_items);
        outliner::outline_lir_with_forbidden_calls(&mut lir_items, &forbidden_calls);
        let extra_stack = outliner::max_outlined_stack_bytes(&lir_items);
        lir::relax_jp_to_jr(&mut lir_items);

        let mut aliased_items = Vec::with_capacity(
            lir_items.len() + fn_aliases.len() + rom_global_aliases.len(),
        );
        for item in lir_items {
            if let lir::LirItem::Label(ref lbl) = item
                && let Some(alias) = fn_aliases.get(lbl).or_else(|| rom_global_aliases.get(lbl))
            {
                aliased_items.push(lir::LirItem::Label(alias.clone()));
            }
            aliased_items.push(item);
        }

        let mut full_asm = String::new();
        let mut resident_startups: Vec<(usize, &ExternalAsm)> = Vec::new();
        let mut resident_others: Vec<(usize, &ExternalAsm)> = Vec::new();
        for (ei, ext) in external_asms.iter().enumerate() {
            if plan.ext_asm_owner.get(ei) == Some(&overlay::OverlayOwner::Resident) {
                if ext.is_startup || overlay::contains_header_area(&ext.source) {
                    resident_startups.push((ei, ext));
                } else {
                    resident_others.push((ei, ext));
                }
            }
        }
        let mut body_items = Vec::with_capacity(aliased_items.len());
        {
            use std::fmt::Write as _;
            let mut emitted_init_end = false;
            for item in aliased_items {
                match item {
                    lir::LirItem::Org(addr) => {
                        if resident_startups.is_empty() {
                            let _ = writeln!(full_asm, ".org 0x{addr:04X}");
                        }
                    }
                    lir::LirItem::Equ(name, addr) => {
                        let _ = writeln!(full_asm, "{name} = 0x{addr:04X}");
                        if name == "__aun_ram" && !emitted_init_end {
                            let _ = writeln!(
                                full_asm,
                                "__aun_init_ram_end = 0x{:04X}",
                                cur_layout.init_ram_end
                            );
                            emitted_init_end = true;
                        }
                    }
                    other => body_items.push(other),
                }
            }
            if !emitted_init_end {
                let _ = writeln!(
                    full_asm,
                    "__aun_init_ram_end = 0x{:04X}",
                    cur_layout.init_ram_end
                );
            }
            for (alias, addr) in &ram_global_aliases {
                let _ = writeln!(full_asm, "{alias} = 0x{addr:04X}");
            }
        }
        if resident_startups.is_empty() {
            if has_resident_main {
                full_asm.push_str("__aun_start:\n");
                full_asm.push_str("    call __aun_startup_init\n");
                full_asm.push_str("    jp main\n\n");
            }
            full_asm.push_str(&lir::format_zilog_lir(&body_items));
        } else {
            for (ei, ext) in &resident_startups {
                use std::fmt::Write as _;
                let _ = writeln!(full_asm, "\n; @aun:ext_begin {ei}");
                full_asm.push_str(&ext.source);
                if !full_asm.ends_with('\n') {
                    full_asm.push('\n');
                }
                full_asm.push_str("; @aun:ext_end\n");
            }
            full_asm.push_str("\n        .area _CODE\n");
            full_asm.push_str(&lir::format_zilog_lir(&body_items));
        }

        // Callable startup data initializer and resident initializer ROM bytes
        full_asm.push_str("\n__aun_startup_init:\n");
        if !cur_layout.init_bytes.is_empty() {
            full_asm.push_str("    ld hl, s__INITIALIZER\n");
            full_asm.push_str("    ld de, s__INITIALIZED\n");
            full_asm.push_str("    ld bc, l__INITIALIZER\n");
            full_asm.push_str("    ldir\n");
        }
        full_asm.push_str("    ret\n");
        if !cur_layout.init_bytes.is_empty() {
            use std::fmt::Write as _;
            full_asm.push_str("\n__aun_init_rom:\n");
            for chunk in cur_layout.init_bytes.chunks(16) {
                let bytes: Vec<String> = chunk.iter().map(|b| format!("0x{b:02X}")).collect();
                let _ = writeln!(full_asm, "    .db {}", bytes.join(", "));
            }
        }

        for (ei, ext) in &resident_others {
            use std::fmt::Write as _;
            let _ = writeln!(full_asm, "\n        .area _CODE\n; @aun:ext_begin {ei}");
            full_asm.push_str(&ext.source);
            if !full_asm.ends_with('\n') {
                full_asm.push('\n');
            }
            full_asm.push_str("; @aun:ext_end\n");
        }
        let assembled = asm::assemble_z80_with_ram_and_externs(
            &full_asm,
            cur_opts.code_origin,
            cur_layout.ram_end,
            &implicit_global_externs,
        )?;
        Ok((full_asm, assembled, extra_stack))
    };

    if opts.auto_ram_base {
        let primary_space = program.target.as_ref().and_then(|t| t.spaces.first());
        let (sp_start, sp_end) = primary_space
            .map(|s| (s.start, s.end))
            .unwrap_or((0x0000, 0xFFFF));
        let mut ram_ceiling: u32 = (sp_end as u32) + 1;
        if let Some(target) = &program.target {
            for st in &target.stacks {
                if st.start >= sp_start && st.start <= sp_end {
                    ram_ceiling = ram_ceiling.min(st.start as u32);
                }
            }
            for bg in &target.bank_groups {
                if bg.start > sp_start && bg.start <= sp_end && bg.end >= sp_end {
                    ram_ceiling = ram_ceiling.min(bg.start as u32);
                }
            }
        }
        let max_global_align: u16 = program
            .globals
            .iter()
            .filter(|g| !g.in_rom && !g.is_extern && g.fixed_addr.is_none() && g.space.is_none())
            .map(|g| g.align.max(1))
            .max()
            .unwrap_or(1);
        let align_down = |addr: u16| -> u16 {
            let a = max_global_align.max(1);
            addr & !(a - 1)
        };
        let probe_base = (align_up_usize(sp_start as usize, max_global_align)
            .min(ram_ceiling.saturating_sub(1) as usize)
            .min(0xFFFF)) as u16;
        let probe_layout = uchiguri::allocate_uchiguri_with_analysis(program, &analysis, probe_base)?;
        let probe_priv = build_fn_private_ranges(program, &analysis, &probe_layout);
        let probe_sb = build_fn_sb_ranges(&analysis, &probe_layout);
        let probe_opts = CompileOptions {
            ram_base: probe_base,
            ..opts.clone()
        };
        let empty_syms = HashMap::new();
        let (_, probe_assembled, _) = assemble_resident(
            &probe_layout,
            &probe_opts,
            &probe_priv,
            &probe_sb,
            &empty_syms,
            &[],
            true,
        )?;
        let ext_ram_bytes = probe_assembled.ram_end.saturating_sub(probe_layout.ram_end) as u32;
        let uchiguri_bytes = probe_layout.ram_end.saturating_sub(probe_layout.ram_base) as u32;
        let total_ram = uchiguri_bytes + ext_ram_bytes;
        if total_ram > ram_ceiling || ram_ceiling - total_ram < sp_start as u32 {
            return Err(format!(
                "auto ram_base placement failed: RAM ({total_ram} bytes) does not fit in 0x{sp_start:04X}..0x{:04X}",
                ram_ceiling.saturating_sub(1)
            ));
        }
        let mut candidate = align_down(ram_ceiling.saturating_sub(total_ram).min(0xFFFF) as u16);
        if total_ram > 0 {
            for _ in 0..3 {
                if (candidate as u32) < (sp_start as u32) {
                    return Err(format!(
                        "auto ram_base placement failed: RAM exceeds lower space bound 0x{sp_start:04X}"
                    ));
                }
                let cand_layout =
                    uchiguri::allocate_uchiguri_with_analysis(program, &analysis, candidate)?;
                let cand_end_excl = (cand_layout.ram_end as u32) + ext_ram_bytes;
                if cand_end_excl <= ram_ceiling {
                    break;
                }
                let overshoot = (cand_end_excl - ram_ceiling) as u16;
                candidate = align_down(candidate.saturating_sub(overshoot.max(1)));
            }
        }
        opts.ram_base = candidate;
    }

    let layout = uchiguri::allocate_uchiguri_with_analysis(program, &analysis, opts.ram_base)?;
    let fn_private_ranges = build_fn_private_ranges(program, &analysis, &layout);
    let fn_sb_ranges = build_fn_sb_ranges(&analysis, &layout);

    let build_resident = |resolved_syms: &HashMap<String, u16>,
                          resolved_spans: &[(u16, u16)],
                          pass1: bool|
     -> Result<(String, asm::AsmOutput, u16), String> {
        assemble_resident(
            &layout,
            &opts,
            &fn_private_ranges,
            &fn_sb_ranges,
            resolved_syms,
            resolved_spans,
            pass1,
        )
    };

    // Pass 1: assemble Resident (emitting unique placeholders for unresolved overlay calls)
    let empty_syms = HashMap::new();
    let (mut full_asm, mut assembled, mut extra_stack) =
        build_resident(&empty_syms, &[], true)?;
    let mut overlays = Vec::new();
    let mut overlay_extra_stack: u16 = 0;

    if !plan.blobs.is_empty() {
        let is_internal_lir_sym =
            |name: &str| name.starts_with("__aun_out_") || name.starts_with("__aun_jt_");
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

        let mut res_end = assembled.origin.wrapping_add(assembled.bytes.len() as u16);
        let max_iters = 3;
        for _iter in 0..max_iters {
            let mut resolved_overlay_syms: HashMap<String, u16> = HashMap::new();
            let mut resolved_blob_spans: Vec<(u16, u16)> = vec![(0, 0); plan.blobs.len()];
            let mut blob_syms: Vec<HashMap<String, u16>> = vec![HashMap::new(); plan.blobs.len()];
            let mut pass_overlays = Vec::with_capacity(plan.blobs.len());
            let mut blob_extra_stacks: Vec<u16> = vec![0; plan.blobs.len()];

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
                        ao.wrapping_add(align_up_usize(al as usize, ovl_align) as u16)
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

                // Build visible_syms = resident_symbols U all ancestor boundary blob symbols,
                // excluding compiler-internal LIR labels (`__aun_out_*`, `__aun_jt_*`) while
                // preserving `__aun_jtbl_dispatch` so overlays can share a single dispatcher.
                let mut visible_syms: HashMap<String, u16> = assembled
                    .symbols
                    .iter()
                    .filter(|(k, _)| !is_internal_lir_sym(k))
                    .map(|(k, &v)| (k.clone(), v))
                    .collect();
                let mut inherited_names: HashSet<String> = HashSet::new();
                for aid in mdef.ancestors(blob.state_id) {
                    if aid != blob.state_id
                        && let Some(&abi) = plan.state_blob_idx.get(&(blob.machine_id, aid))
                    {
                        for (k, &v) in &blob_syms[abi] {
                            if !is_internal_lir_sym(k) {
                                visible_syms.insert(k.clone(), v);
                                inherited_names.insert(k.clone());
                            }
                        }
                    }
                }

                let raw_ovl = codegen::emit_overlay_blob_z80_asm(
                    program,
                    &analysis,
                    &layout,
                    opts.clone(),
                    &plan,
                    blob,
                    origin,
                    &visible_syms,
                )?;
                let ovl_opt_asm = peephole::optimize_asm_full(
                    &raw_ovl,
                    &fn_private_ranges,
                    &fn_sb_ranges,
                    &no_bc_fns,
                );
                let mut ovl_lir_items = lir::parse_zilog_to_lir(&ovl_opt_asm);
                let jtbl_prefix = format!("__aun_jt_ovl{}_", blob.blob_id);
                let reuse_dispatch = visible_syms.contains_key("__aun_jtbl_dispatch");
                jumptable::compress_jump_tables_with_options(
                    program,
                    &mut ovl_lir_items,
                    &jtbl_prefix,
                    reuse_dispatch,
                );
                let out_prefix = format!("__aun_out_ovl{}_", blob.blob_id);
                outliner::outline_lir_with_options(
                    &mut ovl_lir_items,
                    &forbidden_calls,
                    &out_prefix,
                );
                let ovl_extra_stack = outliner::max_outlined_stack_bytes(&ovl_lir_items);
                lir::relax_jp_to_jr(&mut ovl_lir_items);

                let mut ovl_fn_aliases: HashMap<String, String> = HashMap::new();
                for &fid in &blob.fn_ids {
                    let f = &program.functions[fid];
                    if !f.is_test
                        && !f.blocks.is_empty()
                        && let Some(alias) = fn_export_alias(f, pfx)
                    {
                        ovl_fn_aliases.insert(analysis::z80_sym(&f.name), alias);
                    }
                }
                let ovl_aliased_items = if ovl_fn_aliases.is_empty() {
                    ovl_lir_items
                } else {
                    let mut out = Vec::with_capacity(ovl_lir_items.len() + ovl_fn_aliases.len());
                    for item in ovl_lir_items {
                        if let lir::LirItem::Label(ref lbl) = item
                            && let Some(alias) = ovl_fn_aliases.get(lbl)
                        {
                            out.push(lir::LirItem::Label(alias.clone()));
                        }
                        out.push(item);
                    }
                    out
                };

                let mut ovl_asm = lir::format_zilog_lir(&ovl_aliased_items);
                codegen::append_overlay_external_asms(&mut ovl_asm, blob, external_asms);

                let ovl_assembled = asm::assemble_z80_with_ram_and_externs(
                    &ovl_asm,
                    origin,
                    0xC000,
                    &implicit_global_externs,
                )?;
                let size = ovl_assembled.bytes.len();
                let aligned_size = align_up_usize(size, ovl_align);
                let max_ovl_size = budget_end.saturating_sub(origin) as usize + 1;
                if origin > budget_end || aligned_size > max_ovl_size {
                    if aligned_size != size {
                        return Err(format!(
                            "overlay `{}` exceeds `{}` budget (0x{:04X}, {} bytes, aligned {} bytes to {}B > {} bytes, limit 0x{:04X})",
                            blob.state_name,
                            blob.slot_name.as_deref().unwrap_or(&blob.space_name),
                            origin,
                            size,
                            aligned_size,
                            ovl_align,
                            max_ovl_size,
                            budget_end
                        ));
                    }
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
                blob_extra_stacks[bi] = ovl_extra_stack;
                let mut own_syms = HashMap::new();
                for (k, &v) in &ovl_assembled.symbols {
                    if !assembled.symbols.contains_key(k)
                        && !inherited_names.contains(k)
                        && !is_internal_lir_sym(k)
                    {
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
                    aligned_size,
                    asm_listing: ovl_asm,
                    asm_files,
                    symbols: own_syms,
                });
            }

            let mut machine_max_extra: HashMap<u32, u16> = HashMap::new();
            for (_bi, blob) in plan.blobs.iter().enumerate() {
                let mdef = &program.machines[blob.machine_id as usize];
                let path_stack: u16 = mdef
                    .ancestors(blob.state_id)
                    .into_iter()
                    .filter_map(|aid| plan.state_blob_idx.get(&(blob.machine_id, aid)).copied())
                    .fold(0u16, |acc, abi| acc.saturating_add(blob_extra_stacks[abi]));
                machine_max_extra
                    .entry(blob.machine_id)
                    .and_modify(|cur| *cur = (*cur).max(path_stack))
                    .or_insert(path_stack);
            }
            overlay_extra_stack = machine_max_extra
                .values()
                .copied()
                .fold(0u16, |acc, v| acc.saturating_add(v));

            // Pass 3: re-assemble Resident with concrete overlay addresses & LoadOp (dst, size)
            let (new_full_asm, new_assembled, new_extra_stack) =
                build_resident(&resolved_overlay_syms, &resolved_blob_spans, false)?;
            let new_res_end = new_assembled
                .origin
                .wrapping_add(new_assembled.bytes.len() as u16);
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
    for sym_rt in [
        "__aun_start",
        "__aun_startup_init",
        "__aun_init_rom",
        "__aun_panic",
        "__aun_jtbl_dispatch",
    ] {
        if let Some(&addr) = assembled.symbols.get(sym_rt) {
            top_syms.push((addr, sym_rt.to_string()));
        }
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

    let max_stack_bytes = layout
        .max_stack_bytes
        .saturating_add(extra_stack)
        .saturating_add(overlay_extra_stack);
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

    let wcet_res = wcet::compute_wcet(program, &full_asm, &assembled.symbols, &overlays);
    wcet::verify_cycle_budgets(program, &wcet_res)?;

    let final_ram_end = assembled.ram_end.max(layout.ram_end);
    let ram_bytes = final_ram_end.saturating_sub(layout.ram_base);
    let ram_end_excl: u32 = (layout.ram_base as u32) + (ram_bytes as u32);
    if let Some(target) = &program.target {
        if ram_bytes > 0 {
            for st in &target.stacks {
                if (layout.ram_base as u32) <= (st.end as u32) && ram_end_excl > (st.start as u32) {
                    return Err(format!(
                        "RAM 0x{:04X}..0x{:04X} ({} bytes) overlaps stack `{}` (0x{:04X}..0x{:04X})",
                        layout.ram_base,
                        (ram_end_excl - 1) as u16,
                        ram_bytes,
                        st.name,
                        st.start,
                        st.end
                    ));
                }
            }
            if let Some(sp) = target
                .spaces
                .iter()
                .find(|s| s.start <= layout.ram_base && layout.ram_base <= s.end)
                && ram_end_excl > (sp.end as u32) + 1
            {
                return Err(format!(
                    "RAM 0x{:04X}..0x{:04X} ({} bytes) exceeds space `{}` upper bound 0x{:04X}",
                    layout.ram_base,
                    (ram_end_excl - 1) as u16,
                    ram_bytes,
                    sp.name,
                    sp.end
                ));
            }
        }
        if !assembled.bytes.is_empty() {
            let res_end_excl = (assembled.origin as u32) + (assembled.bytes.len() as u32);
            if let Some(sp) = target
                .spaces
                .iter()
                .find(|s| s.start <= assembled.origin && assembled.origin <= s.end)
            {
                if ram_bytes > 0
                    && (layout.ram_base >= sp.start && layout.ram_base <= sp.end)
                    && (assembled.origin as u32) < ram_end_excl
                    && res_end_excl > (layout.ram_base as u32)
                {
                    return Err(format!(
                        "resident code 0x{:04X}..0x{:04X} ({} bytes) overlaps RAM starting at 0x{:04X}",
                        assembled.origin,
                        (res_end_excl - 1) as u16,
                        assembled.bytes.len(),
                        layout.ram_base
                    ));
                }
                for st in &target.stacks {
                    if (st.start >= sp.start && st.end <= sp.end)
                        && (assembled.origin as u32) <= (st.end as u32)
                        && res_end_excl > (st.start as u32)
                    {
                        return Err(format!(
                            "resident code 0x{:04X}..0x{:04X} ({} bytes) overlaps stack `{}` (0x{:04X}..0x{:04X})",
                            assembled.origin,
                            (res_end_excl - 1) as u16,
                            assembled.bytes.len(),
                            st.name,
                            st.start,
                            st.end
                        ));
                    }
                }
                if res_end_excl > (sp.end as u32) + 1 {
                    return Err(format!(
                        "resident code 0x{:04X}..0x{:04X} ({} bytes) exceeds space `{}` upper bound 0x{:04X}",
                        assembled.origin,
                        (res_end_excl - 1) as u16,
                        assembled.bytes.len(),
                        sp.name,
                        sp.end
                    ));
                }
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
        ram_end: final_ram_end,
        ram_bytes,
        max_stack_bytes,
        total_tstates: assembled.total_tstates,
        fn_tstates,
        fn_wcet_tstates: wcet_res.fn_wcet_tstates,
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
            mock_target: None,
            export_name: None,
            cycle_budget: None,
            loop_bounds: HashMap::new(),
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
            owned_by_irq: None,
            export_name: None,
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
            mock_target: None,
            export_name: None,
            cycle_budget: None,
            loop_bounds: HashMap::new(),
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
            mock_target: None,
            export_name: None,
            cycle_budget: None,
            loop_bounds: HashMap::new(),
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
        let src = include_str!("../../../bench/suite/mdxnt_drv.aun");
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
        // Overlapping LDIR memset for [b' '; 44]
        assert!(
            sdas.contains("ldir"),
            "expected overlapping ldir memset for [b' '; 44]:\n{sdas}"
        );
        // No dead `ld hl, #0x0078` spill in constant 16-bit multiply
        assert!(
            !sdas.contains("0x0078"),
            "const 120u16 in row *% 120u16 should fold directly to shift-add:\n{sdas}"
        );
    }

    #[test]
    fn static_struct_array_ix_promotion_analysis_and_soundness() {
        let src = r#"
            struct Ball {
                x: i16,
                y: i16,
                vx: i16,
                vy: i16,
                hp: u8,
                flags: u8,
            }

            struct GameState {
                score: u16,
                balls: [Ball; 4],
            }

            struct BigElem {
                pad: [u8; 126],
                a: u8,
                b: u8,
                c: u8,
            }

            static mut G_STATE: GameState = GameState {
                score: 0u16,
                balls: [Ball { x: 0i16, y: 0i16, vx: 0i16, vy: 0i16, hp: 0u8, flags: 0u8 }; 4],
            };
            static mut BIG_ARR: [BigElem; 2] = [BigElem { pad: [0u8; 126], a: 0u8, b: 0u8, c: 0u8 }; 2];

            pub fn promoted_ball(bi: u8) -> i16 {
                G_STATE.balls[bi].x = G_STATE.balls[bi].x +% G_STATE.balls[bi].vx;
                G_STATE.balls[bi].y = G_STATE.balls[bi].y +% G_STATE.balls[bi].vy;
                G_STATE.balls[bi].hp = G_STATE.balls[bi].hp -% 1u8;
                G_STATE.balls[bi].x +% G_STATE.balls[bi].y
            }

            pub fn few_accesses(bi: u8) -> i16 {
                G_STATE.balls[bi].x +% G_STATE.balls[bi].y
            }

            pub fn mutated_index(bi: u8) -> i16 {
                let mut idx: u8 = bi;
                let a: i16 = G_STATE.balls[idx].x;
                idx = idx +% 1u8;
                let b: i16 = G_STATE.balls[idx].y;
                let c: i16 = G_STATE.balls[idx].vx;
                a +% b +% c
            }

            pub fn disp_out_of_range(i: u8) -> u8 {
                BIG_ARR[i].a +% BIG_ARR[i].b +% BIG_ARR[i].c
            }

            pub fn ix_param_priority(b: &mut Ball, idx: u8) -> i16 {
                b.x = b.x +% 1i16;
                G_STATE.balls[idx].x +% G_STATE.balls[idx].y +% G_STATE.balls[idx].vx
            }

            fn helper_bump(b: &mut Ball) {
                b.flags = b.flags +% 1u8;
            }

            pub fn step_ball_test(bi: u8) -> i16 {
                if G_STATE.balls[bi].hp == 0u8 {
                    return -1i16;
                }
                G_STATE.balls[bi].x = G_STATE.balls[bi].x +% G_STATE.balls[bi].vx;
                G_STATE.balls[bi].y = G_STATE.balls[bi].y +% G_STATE.balls[bi].vy;
                if G_STATE.balls[bi].x < 0i16 {
                    G_STATE.balls[bi].vx = -G_STATE.balls[bi].vx;
                }
                G_STATE.balls[bi].hp = G_STATE.balls[bi].hp -% 1u8;
                helper_bump(&mut G_STATE.balls[bi]);
                G_STATE.balls[bi].x +% G_STATE.balls[bi].y +% (G_STATE.balls[bi].hp as i16) +% (G_STATE.balls[bi].flags as i16)
            }
        "#;
        let mut prog = aun_front::compile_str(0, src).expect("front compile");
        aun_ir::opt::optimize_program(&mut prog);
        loop_opt::optimize_z80_program(&mut prog, true);
        let analysis = analysis::ProgramAnalysis::analyze(&prog);
        let fa_of = |name: &str| {
            let idx = prog.functions.iter().position(|f| f.name == name).unwrap();
            &analysis.fns[idx]
        };
        assert!(
            fa_of("step_ball_test")
                .ix_static_elem
                .as_ref()
                .is_some_and(|ise| ise.hoist_param == Some(0)),
            "expected step_ball_test (with inlined &mut helper) to promote static elem ptr to IX"
        );

        assert!(
            fa_of("promoted_ball")
                .ix_static_elem
                .as_ref()
                .is_some_and(|ise| ise.hoist_param == Some(0)),
            "expected promoted_ball to hoist static struct-array elem ptr to IX"
        );
        assert!(
            fa_of("few_accesses").ix_static_elem.is_none(),
            "fewer than 3 accesses should not promote to IX"
        );
        assert!(
            fa_of("mutated_index")
                .ix_static_elem
                .as_ref()
                .is_none_or(|ise| ise.hoist_param.is_none()),
            "mutated index must never be hoisted to prologue"
        );
        assert!(
            fa_of("disp_out_of_range").ix_static_elem.is_none(),
            "field displacement > 124 must not promote to IX"
        );
        assert!(
            fa_of("ix_param_priority").ix_param_slot == Some(0)
                && fa_of("ix_param_priority").ix_static_elem.is_none(),
            "existing ix_param_slot must take priority over ix_static_elem"
        );
    }

    #[test]
    fn post_inline_static_ix_base_cse_with_branches_and_ref_aliases() {
        let src = r#"
            struct Ball {
                x: i16,
                y: i16,
                vx: i16,
                vy: i16,
                px: u8,
                py: u8,
                active: u8,
                pad: [u8; 22],
            }

            static mut BALLS: [Ball; 3] = [Ball {
                x: 0i16, y: 0i16, vx: 0i16, vy: 0i16, px: 0u8, py: 0u8, active: 0u8, pad: [0u8; 22],
            }; 3];

            fn bump_ref(b: &mut Ball) {
                b.x = b.x +% 1i16;
            }

            fn sync_ball_px(bi: u8) {
                let b: &mut Ball = &mut BALLS[bi as u16];
                b.px = (b.x >> 4i16) as u8;
                b.py = (b.y >> 4i16) as u8;
            }

            fn guard_right_divider(bi: u8) {
                let b: &mut Ball = &mut BALLS[bi as u16];
                if b.x > 1600i16 && b.vx > 0i16 {
                    b.x = 1600i16;
                    b.vx = -b.vx;
                }
            }

            pub fn step_ball_post_inline(bi: u8, mode: u8) -> i16 {
                let b: &mut Ball = if mode == 0u8 {
                    let s: &mut Ball = &mut BALLS[bi as u16];
                    s.y = s.y +% 2i16;
                    s
                } else {
                    let p: &mut Ball = &mut BALLS[bi as u16];
                    p.x = p.x +% p.vx;
                    guard_right_divider(bi);
                    p
                };
                bump_ref(b);
                sync_ball_px(bi);
                b.x +% b.y +% (b.px as i16) +% (b.py as i16)
            }

            pub fn cold_a(bi: u8) { sync_ball_px(bi); }
            pub fn cold_b(bi: u8) { sync_ball_px(bi); }
            pub fn hot_driver() {
                let mut i: u8 = 0u8;
                while i < 3u8 {
                    let _r: i16 = step_ball_post_inline(i, i & 1u8);
                    i = i +% 1u8;
                }
            }
        "#;
        let mut prog = aun_front::compile_str(0, src).expect("front compile");
        aun_ir::opt::optimize_program(&mut prog);
        loop_opt::optimize_z80_program(&mut prog, true);
        let analysis = analysis::ProgramAnalysis::analyze(&prog);
        let fn_idx = prog
            .functions
            .iter()
            .position(|f| f.name == "step_ball_post_inline")
            .unwrap();
        let func = &prog.functions[fn_idx];
        let fa = &analysis.fns[fn_idx];

        let total_index_addrs: usize = func
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|i| matches!(i, aun_ir::IrInst::IndexAddr { .. }))
            .count();
        let ise = fa
            .ix_static_elem
            .as_ref()
            .expect("step_ball_post_inline must promote BALLS[bi] to IX");
        assert_eq!(ise.hoist_param, Some(0), "must hoist bi (slot 0) to prologue");
        assert!(
            fa.ix_local_slot.is_none(),
            "hoisted static IX must take priority over local __phi slot"
        );
        assert_eq!(
            ise.index_vals.len(),
            total_index_addrs,
            "all {total_index_addrs} IndexAddrs across branches/inlined helpers must be captured in ise.index_vals (got {})",
            ise.index_vals.len()
        );
    }

    #[test]
    fn signed_i16_cmp_avoids_xor_0x80_in_bin_and_sdasz80() {
        let src = r#"
            pub fn branch_cmps(a: i16, b: i16) -> u8 {
                let mut m: u8 = 0u8;
                if a < b { m = m | 1u8; }
                if a <= b { m = m | 2u8; }
                if a > b { m = m | 4u8; }
                if a >= b { m = m | 8u8; }
                if a < 0i16 { m = m | 16u8; }
                if a >= 0i16 { m = m | 32u8; }
                if a > 0i16 { m = m | 64u8; }
                if a <= 0i16 { m = m | 128u8; }
                m
            }

            pub fn bool_cmps(a: i16, b: i16) -> u8 {
                let c0: bool = a < b;
                let c1: bool = a <= b;
                let c2: bool = a > b;
                let c3: bool = a >= b;
                let c4: bool = a < 0i16;
                let c5: bool = a >= 0i16;
                let c6: bool = a > 0i16;
                let c7: bool = a <= 0i16;
                (c0 as u8) +% ((c1 as u8) << 1) +% ((c2 as u8) << 2) +% ((c3 as u8) << 3)
                    +% ((c4 as u8) << 4) +% ((c5 as u8) << 5) +% ((c6 as u8) << 6) +% ((c7 as u8) << 7)
            }
        "#;
        let prog = aun_front::compile_str(0, src).expect("front compile");
        let bin_out = compile_z80_with_options(&prog, &[], CompileOptions::release())
            .expect("compile_z80_with_options");
        let (sdas, _, _) = compile_z80_sdasz80("cmp_i16", &prog, CompileOptions::release())
            .expect("compile_z80_sdasz80");

        for (label, text) in [("bin", &bin_out.asm_listing), ("sdasz80", &sdas)] {
            assert!(
                !text.contains("xor 0x80") && !text.contains("xor #0x80") && !text.contains("xor\t0x80"),
                "signed i16 compares must not emit xor 0x80 in {label}:\n{text}"
            );
            assert!(
                text.contains("sbc hl, de") && text.contains("jp pe,") && text.contains("bit 7, h"),
                "expected native sbc hl, de + jp pe + bit 7, h in {label}:\n{text}"
            );
        }
    }

    #[test]
    fn small_array_u8_mul_avoids_mul_hl_helper_and_respects_bounds() {
        let src = r#"
            struct Item33 {
                a: u8,
                b: u16,
                pad: [u8; 30],
            }

            static mut SMALL_ARR: [Item33; 3] = [
                Item33 { a: 1u8, b: 10u16, pad: [0u8; 30] },
                Item33 { a: 2u8, b: 20u16, pad: [0u8; 30] },
                Item33 { a: 3u8, b: 30u16, pad: [0u8; 30] },
            ];
            static mut BIG_ARR: [Item33; 9] = [
                Item33 { a: 0u8, b: 0u16, pad: [0u8; 30] },
                Item33 { a: 0u8, b: 0u16, pad: [0u8; 30] },
                Item33 { a: 0u8, b: 0u16, pad: [0u8; 30] },
                Item33 { a: 0u8, b: 0u16, pad: [0u8; 30] },
                Item33 { a: 0u8, b: 0u16, pad: [0u8; 30] },
                Item33 { a: 0u8, b: 0u16, pad: [0u8; 30] },
                Item33 { a: 0u8, b: 0u16, pad: [0u8; 30] },
                Item33 { a: 0u8, b: 0u16, pad: [0u8; 30] },
                Item33 { a: 0u8, b: 0u16, pad: [0u8; 30] },
            ];

            pub fn read_small_u8(i: u8) -> u16 {
                let mut acc: u16 = 0u16;
                acc = acc +% (SMALL_ARR[i].a as u16);
                acc = acc +% SMALL_ARR[i].b;
                acc = acc +% (SMALL_ARR[i].pad[0] as u16);
                acc
            }

            pub fn read_small_u16(i: u16) -> u16 {
                SMALL_ARR[i].b
            }

            pub fn read_big_u8(i: u8, j: u8) -> u16 {
                BIG_ARR[i].b +% BIG_ARR[j].b
            }
        "#;
        let prog = aun_front::compile_str(0, src).expect("front compile");

        // Debug mode: bounds checks must still be emitted, while 8-bit inline mul handles SMALL_ARR.
        let dbg_out = compile_z80_with_options(&prog, &[], CompileOptions::debug())
            .expect("debug compile");
        assert!(
            dbg_out.asm_listing.contains("__aun_panic"),
            "debug mode must keep bounds checks on small array"
        );
        assert!(
            dbg_out.asm_listing.contains("call __aun_mul_hl_33"),
            "large array [Item33; 9] (8*33=264 > 255) should still use __aun_mul_hl_33 in debug:\n{}",
            dbg_out.asm_listing
        );

        // Compile SMALL_ARR-only program in release and debug to assert zero __aun_mul_*_33 emission.
        let small_only_src = r#"
            struct Item33 {
                a: u8,
                b: u16,
                pad: [u8; 30],
            }
            static mut SMALL_ARR: [Item33; 3] = [
                Item33 { a: 1u8, b: 10u16, pad: [0u8; 30] },
                Item33 { a: 2u8, b: 20u16, pad: [0u8; 30] },
                Item33 { a: 3u8, b: 30u16, pad: [0u8; 30] },
            ];
            pub fn read_small_hoisted(i: u8) -> u16 {
                let mut acc: u16 = 0u16;
                acc = acc +% (SMALL_ARR[i].a as u16);
                acc = acc +% SMALL_ARR[i].b;
                acc = acc +% (SMALL_ARR[i].pad[0] as u16);
                acc
            }
            pub fn read_small_u16(i: u16) -> u16 {
                SMALL_ARR[i].b
            }
        "#;
        let small_prog = aun_front::compile_str(0, small_only_src).expect("small front");
        let rel_out = compile_z80_with_options(&small_prog, &[], CompileOptions::release())
            .expect("release compile");
        let (rel_sdas, _, _) =
            compile_z80_sdasz80("small_mul", &small_prog, CompileOptions::release())
                .expect("release sdasz80");
        let dbg_small = compile_z80_with_options(&small_prog, &[], CompileOptions::debug())
            .expect("debug small compile");

        for (label, text) in [
            ("release bin", &rel_out.asm_listing),
            ("release sdasz80", &rel_sdas),
            ("debug bin", &dbg_small.asm_listing),
        ] {
            assert!(
                !text.contains("__aun_mul_hl_33") && !text.contains("__aun_mul_a_33"),
                "{label} must not reference or emit __aun_mul_hl_33/__aun_mul_a_33 for [Item33; 3]:\n{text}"
            );
            assert!(
                text.contains("add a, a"),
                "{label} must emit 8-bit shift-add multiply:\n{text}"
            );
        }
    }

    #[test]
    fn slice_index_release_eliminates_dead_len_reload() {
        let src = r#"
            pub fn read_pair(a: u16, s: &const [u8], i: u16, j: u16) -> u8 {
                let x: u8 = s[i];
                let y: u8 = s[j];
                x +% y +% (a as u8)
            }
        "#;
        let prog = aun_front::compile_str(0, src).expect("front compile");
        let bin_out = compile_z80_with_options(&prog, &[], CompileOptions::release())
            .expect("compile_z80_with_options");
        let (sdas, _, _) = compile_z80_sdasz80("slice_idx_len", &prog, CompileOptions::release())
            .expect("compile_z80_sdasz80");

        for (label, text) in [("bin", &bin_out.asm_listing), ("sdasz80", &sdas)] {
            assert!(
                !text.contains("ld d, h") && !text.contains("ld e, l"),
                "release slice indexing must not copy unused fat-ptr .len into DE in {label}:\n{text}"
            );
            // Stack slice param `s` is at IX+6 (ptr at +6/+7, len at +8/+9): .len must never be loaded.
            assert!(
                !text.contains("(ix+8)") && !text.contains("(ix + 8)") && !text.contains("8 (ix)"),
                "release slice indexing must not load unused fat-ptr .len from IX+8 in {label}:\n{text}"
            );
        }
    }

    #[test]
    fn issue42_u8_rot_shifts_u16_bitpack_and_ldir_memset_codegen() {
        let src = r#"
            struct KeyFields {
                a: u8,
                b: u8,
                c: u8,
                d: u8,
                flag: bool,
                arr: [u8; 4],
            }

            pub fn u8_shifts(x: u8, s: i8) -> u16 {
                let s5 = x << 5;
                let s6 = x << 6;
                let s7 = x << 7;
                let r5 = x >> 5;
                let r6 = x >> 6;
                let r7 = x >> 7;
                let sr5 = (s >> 5) as u8;
                ((s5 ^ s6 ^ s7 ^ r5 ^ r6 ^ r7) as u16) | ((sr5 as u16) << 8)
            }

            pub fn pack_key(k: &KeyFields) -> u16 {
                let v0 = k.arr[1];
                let v1 = k.arr[2];
                ((k.a as u16))
                    | (((k.flag as u16) << 5))
                    | (((k.b & 0x07u8) as u16) << 2)
                    | ((v0 as u16) << 8)
                    | ((v1 as u16) << 12)
                    | ((k.c as u16) << 15)
            }

            pub fn memset_short_and_long() -> u8 {
                let mut short_buf: [u8; 8] = [0x11u8; 8];
                let mut long_buf: [u8; 44] = [0x20u8; 44];
                short_buf[0] +% long_buf[43]
            }
        "#;
        let prog = aun_front::compile_str(0, src).expect("front compile");
        let (sdas, _, _) =
            compile_z80_sdasz80("issue42_unit", &prog, CompileOptions::release())
                .expect("compile_z80_sdasz80");

        let fn_body = |name: &str| -> String {
            let mut out = String::new();
            let mut in_fn = false;
            for line in sdas.lines() {
                let t = line.trim();
                if t == format!("{name}:") {
                    in_fn = true;
                } else if in_fn && t.ends_with(':') && !t.starts_with(&format!("{name}_")) && !t.starts_with(&format!("{name}::")) {
                    break;
                }
                if in_fn {
                    out.push_str(line);
                    out.push('\n');
                }
            }
            out
        };

        let sh_asm = fn_body("u8_shifts");
        assert!(sh_asm.contains("rrca") && sh_asm.contains("#0xE0"), "expected rrca + #0xE0:\n{sh_asm}");
        assert!(sh_asm.contains("rlca") && sh_asm.contains("#0x07"), "expected rlca + #0x07:\n{sh_asm}");
        assert!(sh_asm.contains("sra\ta") || sh_asm.contains("sra a"), "expected exact sra a for i8 >> 5:\n{sh_asm}");

        let pack_asm = fn_body("pack_key");
        let pack_after_prologue = pack_asm
            .split_once("pop\tix")
            .or_else(|| pack_asm.split_once("pop ix"))
            .map(|(_, rest)| rest)
            .unwrap_or(&pack_asm);
        assert!(
            !pack_after_prologue.contains("push\thl") && !pack_after_prologue.contains("push hl"),
            "pack_key body should not push hl:\n{pack_asm}"
        );
        assert!(
            !pack_after_prologue.contains("pop\thl") && !pack_after_prologue.contains("pop hl"),
            "pack_key body should not pop hl:\n{pack_asm}"
        );
        assert!(
            !pack_asm.contains("add\thl")
                && !pack_asm.contains("add hl")
                && !pack_asm.contains("ex\tde")
                && !pack_asm.contains("ex de")
                && !pack_asm.contains("__aun_ram")
                && !pack_asm.contains("ld\th, #0")
                && !pack_asm.contains("ld h, 0"),
            "pack_key should accumulate bytes in D/E without HL widening or __aun_ram:\n{pack_asm}"
        );
        assert!(
            (pack_asm.contains("or\te") || pack_asm.contains("or e"))
                && (pack_asm.contains("or\td") || pack_asm.contains("or d")),
            "pack_key should use 8-bit or e / or d accumulators:\n{pack_asm}"
        );

        let ms_asm = fn_body("memset_short_and_long");
        assert!(ms_asm.contains("djnz"), "short 8-byte memset should keep djnz:\n{ms_asm}");
        assert!(ms_asm.contains("ldir"), "long 44-byte memset should use overlapping ldir:\n{ms_asm}");
    }

    #[test]
    fn issue43_signed_i16_fallthrough_ovf_tail_and_high_byte_cmps() {
        let src = r#"
            pub fn gen_cmp(a: i16, b: i16) -> u8 {
                if a < b { 1u8 } else { 2u8 }
            }

            pub fn aligned_cmps(x: i16) -> u8 {
                let mut m: u8 = 0u8;
                if x < 0x0400i16 { m = m | 1u8; }
                if x >= -1024i16 { m = m | 2u8; }
                if x <= 0x03FFi16 { m = m | 4u8; }
                if x > -1025i16 { m = m | 8u8; }
                m
            }

            pub fn unaligned_le(x: i16) -> u8 {
                if x <= 0x0400i16 { 1u8 } else { 0u8 }
            }
        "#;
        let prog = aun_front::compile_str(0, src).expect("front compile");
        let bin_out = compile_z80_with_options(&prog, &[], CompileOptions::release())
            .expect("compile_z80_with_options");
        let (sdas, _, _) = compile_z80_sdasz80("issue43_cmp", &prog, CompileOptions::release())
            .expect("compile_z80_sdasz80");

        for (label, text) in [("bin", &bin_out.asm_listing), ("sdasz80", &sdas)] {
            // Fall-through signed i16 compare emits out-of-line `_cmp_ovf_` tail stub.
            assert!(
                text.contains("_cmp_ovf_"),
                "expected out-of-line _cmp_ovf_ tail stub in {label}:\n{text}"
            );
            // Aligned high-byte compares (`aligned_cmps`) avoid `sbc hl, de`.
            let fn_aligned_start = text.find("aligned_cmps:").expect("aligned_cmps: label");
            let fn_aligned_end = text[fn_aligned_start..]
                .find("\nunaligned_le:")
                .map(|o| fn_aligned_start + o)
                .unwrap_or(text.len());
            let aligned_slice = &text[fn_aligned_start..fn_aligned_end];
            assert!(
                !aligned_slice.contains("sbc hl, de")
                    && (aligned_slice.contains("xor 0x80") || aligned_slice.contains("xor #0x80"))
                    && (aligned_slice.contains("cp 0x84") || aligned_slice.contains("cp #0x84")),
                "256B-aligned compares should lower via high-byte H without sbc hl, de in {label}:\n{aligned_slice}"
            );
            // `x <= 0x0400` is NOT equivalent to high-byte compare, so `unaligned_le` keeps 16-bit compare.
            let unaligned_slice = &text[fn_aligned_end..];
            assert!(
                unaligned_slice.contains("sbc hl, de"),
                "unaligned x <= 0x0400 must still emit 16-bit sbc hl, de in {label}:\n{unaligned_slice}"
            );
        }
    }

    #[test]
    fn issue51_codegen_hl_walk_and_invalidation_negative_cases() {
        let src = r#"
            pub struct Item {
                tag: u8,
                pad: [u8; 8],
                x_px: u8,
                y_px: u8,
                w16: u16,
                w2: u16,
            }
            static mut ITEMS: [Item; 4] = [
                Item { tag: 0u8, pad: [0u8; 8], x_px: 0u8, y_px: 0u8, w16: 0u16, w2: 0u16 }; 4
            ];
            static mut OUT_X: u8 = 0u8;
            static mut OUT_Y: u8 = 0u8;

            fn touch_extern() {
                asm!("nop");
            }

            pub fn walk_adjacent(bi: u8, dummy: &mut Item) -> u8 {
                // Passing `dummy` claims IX, forcing `&ITEMS[bi]` to live as a spilled HL pointer.
                dummy.tag = 1u8;
                let b: &mut Item = &mut ITEMS[bi as u16];
                OUT_X = b.x_px;
                OUT_Y = b.y_px;
                OUT_X +% OUT_Y
            }

            pub fn walk_invalidate_on_call(bi: u8, dummy: &mut Item) -> u8 {
                dummy.tag = 1u8;
                let b: &mut Item = &mut ITEMS[bi as u16];
                let x: u8 = b.x_px;
                touch_extern();
                let y: u8 = b.y_px;
                x +% y
            }

            pub fn walk_after_16bit_read(bi: u8, dummy: &mut Item) -> u16 {
                dummy.tag = 1u8;
                let b: &mut Item = &mut ITEMS[bi as u16];
                let w1: u16 = b.w16;
                let w2: u16 = b.w2;
                w1 +% w2
            }
        "#;
        let prog = aun_front::compile_str(0, src).expect("front compile");
        let out = compile_z80_with_options(&prog, &[], CompileOptions::release())
            .expect("compile_z80_with_options");
        let fn_body = |name: &str| -> String {
            let start = out.asm_listing.find(&format!("{name}:")).expect(name);
            let rest = &out.asm_listing[start..];
            let end = rest[1..]
                .find("\nwalk_")
                .or_else(|| rest[1..].find("\ntouch_"))
                .or_else(|| rest[1..].find("\n;"))
                .map(|o| o + 1)
                .unwrap_or(rest.len());
            rest[..end].to_string()
        };

        let adj = fn_body("walk_adjacent");
        assert!(
            adj.contains("inc hl"),
            "walk_adjacent should walk HL from +9 to +10 via inc hl:\n{adj}"
        );
        assert!(
            !adj.contains("0x000A"),
            "walk_adjacent must not emit ld de, 0x000A; add hl, de:\n{adj}"
        );

        let inv = fn_body("walk_invalidate_on_call");
        assert!(
            inv.contains("call touch_extern") && inv.contains("0x000A"),
            "walk_invalidate_on_call must invalidate HL across call and reload +10:\n{inv}"
        );

        let w16 = fn_body("walk_after_16bit_read");
        assert!(
            !w16.contains("0x000D") && w16.matches("inc hl").count() >= 2,
            "walk_after_16bit_read should update HL offset after 16-bit read (+11..+12 -> +13 via inc hl without 0x000D add):\n{w16}"
        );
    }

    #[test]
    fn issue52_u16_bitpack_8bit_de_accumulators_and_soundness() {
        let src = r#"
            pub fn pack_bbb88(b: u8, c: u8, d: u8) -> u16 {
                (mem_read_u8(0xF01Eu16) as u16)
                    | (((mem_read_u8(0xF022u16) & 0x03u8) as u16) << 3)
                    | (((mem_read_u8(0xF01Fu16) & 0x07u8) as u16) << 5)
                    | ((b as u16) << 8)
                    | ((c as u16) << 10)
                    | ((d as u16) << 13)
            }

            pub fn pack_if_shr_u16mask(cond: bool, a: u8, b: u8, s: u8, p: u8, q: u8) -> u16 {
                let add_m = (((a +% b) & 0x03u8) as u16) << 8;
                let sel = if cond { a & 0x03u8 } else { b & 0x07u8 };
                (sel as u16)
                    | (((p as u16) & 0x0003u16) << 3)
                    | (((s >> 5) as u16) << 5)
                    | add_m
                    | ((q as u16) << 10)
            }

            pub fn pack_mid_clobber(a: u8, b: u8, p: u8, q: u8) -> u16 {
                (((p & 0x07u8) as u16) << 2)
                    | ((((a +% b) & 0x03u8) as u16) << 8)
                    | ((q as u16) << 10)
            }

            pub fn pack_unmasked_spill(a: u8, b: u8) -> u16 {
                ((a as u16) << 3) | ((b as u16) << 10)
            }

            pub fn pack_mixed_if_spill(cond: bool, a: u8, b: u8) -> u16 {
                let v = if cond { a & 0x03u8 } else { b };
                (v as u16) << 5
            }
        "#;
        let prog = aun_front::compile_str(0, src).expect("front compile");
        let (sdas, _, _) =
            compile_z80_sdasz80("issue52_unit", &prog, CompileOptions::release())
                .expect("compile_z80_sdasz80");

        let fn_body = |name: &str| -> String {
            let mut out = String::new();
            let mut in_fn = false;
            for line in sdas.lines() {
                let t = line.trim();
                if t == format!("{name}:") {
                    in_fn = true;
                } else if in_fn
                    && t.ends_with(':')
                    && !t.starts_with(&format!("{name}_"))
                    && !t.starts_with(&format!("{name}::"))
                {
                    break;
                }
                if in_fn {
                    out.push_str(line);
                    out.push('\n');
                }
            }
            out
        };

        for name in ["pack_bbb88", "pack_if_shr_u16mask"] {
            let asm = fn_body(name);
            assert!(
                !asm.contains("add\thl") && !asm.contains("add hl"),
                "{name} must not widen byte-contained shifts to add hl, hl:\n{asm}"
            );
            assert!(
                !asm.contains(", hl")
                    && !asm.contains("hl, (__aun_ram")
                    && !asm.contains("ld\th, #0")
                    && !asm.contains("ld h, 0"),
                "{name} must not spill/reload HL in __aun_ram or zero-extend ld h, 0:\n{asm}"
            );
            assert!(
                (asm.contains("or\te") || asm.contains("or e"))
                    && (asm.contains("or\td") || asm.contains("or d")),
                "{name} should accumulate low/high bytes in E and D:\n{asm}"
            );
        }

        let mid_asm = fn_body("pack_mid_clobber");
        assert!(
            !mid_asm.contains("add\thl") && !mid_asm.contains("add hl"),
            "pack_mid_clobber must keep byte-contained shifts 8-bit even across E-clobbering add:\n{mid_asm}"
        );

        let spill_asm = fn_body("pack_unmasked_spill");
        assert!(
            spill_asm.contains("add\thl, hl") || spill_asm.contains("add hl, hl"),
            "pack_unmasked_spill must preserve upper-byte spillover of unmasked u8 << 3:\n{spill_asm}"
        );

        let mixed_asm = fn_body("pack_mixed_if_spill");
        assert!(
            mixed_asm.contains("rrca")
                && (mixed_asm.contains("ld\th, a") || mixed_asm.contains("ld h, a")),
            "pack_mixed_if_spill must preserve upper-byte spillover when one if-arm is unmasked:\n{mixed_asm}"
        );
    }

    #[test]
    fn issue53_overlay_blobs_run_lir_passes_without_symbol_collisions_and_track_stack() {
        let make_src = |resident_jtbl: bool| -> String {
            let res_dispatch = if resident_jtbl {
                r#"
                pub fn res_dispatch(k: u8) -> u8 {
                    if k == 0u8 { return 11u8; }
                    if k == 1u8 { return 22u8; }
                    if k == 2u8 { return 33u8; }
                    if k == 3u8 { return 44u8; }
                    if k == 4u8 { return 55u8; }
                    if k == 5u8 { return 66u8; }
                    if k == 6u8 { return 77u8; }
                    if k == 7u8 { return 88u8; }
                    if k == 8u8 { return 99u8; }
                    if k == 9u8 { return 110u8; }
                    if k == 10u8 { return 121u8; }
                    if k == 11u8 { return 132u8; }
                    if k == 12u8 { return 143u8; }
                    if k == 13u8 { return 154u8; }
                    0u8
                }
                "#
            } else {
                ""
            };
            format!(
                r#"
                target test_ovl : z80 {{
                    space MainRam : flat @ 0x0100..0xBFFF;
                    stack MainStack @ 0xBF00..0xBFFF;
                    on_swap(ops: &const [LoadOp]) => noop_swap(ops);
                }}

                static mut OUT_VAL: u16 = 0u16;
                extern "asm" {{
                    fn ovl_ext_tag() -> u8 in A;
                }}
                fn noop_swap(_ops: &const [LoadOp]) {{}}

                {res_dispatch}

                machine Scene {{
                    state Idle;
                    #[swap_boundary]
                    state Parent {{
                        #[swap_boundary]
                        state ChildA;
                        #[swap_boundary]
                        state ChildB;
                        initial ChildA;
                    }}
                    transitions {{
                        Idle -> Parent;
                        Parent::ChildA -> Parent::ChildB;
                        Parent -> Idle;
                    }}
                    on Idle {{}}
                    on Parent {{
                        fn parent_dispatch(&mut self, x: u8) -> u16 {{
                            let b: u16 = ovl_ext_tag() as u16;
                            if x == 0u8 {{ return 0x101u16 +% b; }}
                            if x == 1u8 {{ return 0x102u16 +% b; }}
                            if x == 2u8 {{ return 0x103u16 +% b; }}
                            if x == 3u8 {{ return 0x104u16 +% b; }}
                            if x == 4u8 {{ return 0x105u16 +% b; }}
                            if x == 5u8 {{ return 0x106u16 +% b; }}
                            if x == 6u8 {{ return 0x107u16 +% b; }}
                            if x == 7u8 {{ return 0x108u16 +% b; }}
                            if x == 8u8 {{ return 0x109u16 +% b; }}
                            if x == 9u8 {{ return 0x10Au16 +% b; }}
                            if x == 10u8 {{ return 0x10Bu16 +% b; }}
                            if x == 11u8 {{ return 0x10Cu16 +% b; }}
                            if x == 12u8 {{ return 0x10Du16 +% b; }}
                            if x == 13u8 {{ return 0x10Eu16 +% b; }}
                            0x199u16
                        }}
                    }}
                    on Parent::ChildA {{
                        fn compute(&mut self, x: u8) -> u16 {{
                            let a: u16 = ((x as u16) *% 3u16) +% 11u16;
                            OUT_VAL = (OUT_VAL +% a) ^ 0x0055u16;
                            if x == 0u8 {{ return self.parent_dispatch(0u8) +% OUT_VAL; }}
                            OUT_VAL = (OUT_VAL +% a) ^ 0x0055u16;
                            if x == 1u8 {{ return self.parent_dispatch(1u8) +% OUT_VAL; }}
                            OUT_VAL = (OUT_VAL +% a) ^ 0x0055u16;
                            if x == 2u8 {{ return self.parent_dispatch(2u8) +% OUT_VAL; }}
                            OUT_VAL = (OUT_VAL +% a) ^ 0x0055u16;
                            if x == 3u8 {{ return self.parent_dispatch(3u8) +% OUT_VAL; }}
                            OUT_VAL = (OUT_VAL +% a) ^ 0x0055u16;
                            if x == 4u8 {{ return self.parent_dispatch(4u8) +% OUT_VAL; }}
                            OUT_VAL = (OUT_VAL +% a) ^ 0x0055u16;
                            self.parent_dispatch(x) +% OUT_VAL
                        }}
                    }}
                    on Parent::ChildB {{
                        fn compute(&mut self, x: u8) -> u16 {{
                            if x == 0u8 {{ return 1u16; }}
                            if x == 1u8 {{ return 2u16; }}
                            if x == 2u8 {{ return 3u16; }}
                            if x == 3u8 {{ return 4u16; }}
                            if x == 4u8 {{ return 5u16; }}
                            if x == 5u8 {{ return 6u16; }}
                            if x == 6u8 {{ return 7u16; }}
                            if x == 7u8 {{ return 8u16; }}
                            if x == 8u8 {{ return 9u16; }}
                            if x == 9u8 {{ return 10u16; }}
                            if x == 10u8 {{ return 11u16; }}
                            if x == 11u8 {{ return 12u16; }}
                            if x == 12u8 {{ return 13u16; }}
                            if x == 13u8 {{ return 14u16; }}
                            99u16
                        }}
                    }}
                }}

                pub fn main(mode: u8, x: u8) -> u16 {{
                    let mut m: Scene = Scene::new();
                    let _res: u8 = if mode == 0u8 {{
                        {}
                    }} else {{
                        0u8
                    }};
                    // Repeated Resident sequence for outliner when resident_jtbl == true
                    if x == 250u8 {{
                        OUT_VAL = (OUT_VAL *% 3u16) +% 7u16 +% (mode as u16);
                    }} else if x == 251u8 {{
                        OUT_VAL = (OUT_VAL *% 3u16) +% 7u16 +% (mode as u16);
                    }}
                    if mode == 1u8 {{
                        m.goto(Parent::ChildA {{}});
                    }} else {{
                        m.goto(Parent::ChildB {{}});
                    }}
                    let r: u16 = m.compute(x);
                    m.goto(Idle {{}});
                    r
                }}
            "#,
                if resident_jtbl { "res_dispatch(x)" } else { "0u8" }
            )
        };

        let ext_s = ExternalAsm {
            name: "ovl_ext.s".to_string(),
            source: "ovl_ext_tag:\n    ld a, #0x2A\n    ld h, h ; untouchable-ext\n    ret\n"
                .to_string(),
            owner_state: Some("Scene::Parent".to_string()),
            is_startup: false,
        };

        for resident_jtbl in [true, false] {
            let src = make_src(resident_jtbl);
            let prog = aun_front::compile_str(0, &src).expect("front compile");
            let out = compile_z80_with_external_asms(
                &prog,
                &[ext_s.clone()],
                CompileOptions {
                    code_origin: 0x0100,
                    ram_base: 0x9000,
                    ..CompileOptions::release()
                },
            )
            .expect("compile_z80_with_external_asms");
            assert_eq!(out.overlays.len(), 3, "Parent, ChildA, ChildB");

            let parent = &out.overlays[0];
            let child_a = &out.overlays[1];
            let child_b = &out.overlays[2];

            // 1. Namespaced jump tables and reused `__aun_jtbl_dispatch`
            assert!(
                parent.asm_listing.contains("__aun_jt_ovl0_0:"),
                "parent overlay should run jumptable compression:\n{}",
                parent.asm_listing
            );
            assert!(
                child_b.asm_listing.contains("__aun_jt_ovl2_0:"),
                "child B overlay should use ovl2-prefixed jump table:\n{}",
                child_b.asm_listing
            );
            if resident_jtbl {
                assert!(
                    out.symbols.contains_key("__aun_jtbl_dispatch"),
                    "Resident should define __aun_jtbl_dispatch"
                );
                assert!(
                    !parent.asm_listing.contains("__aun_jtbl_dispatch:")
                        && !child_a.asm_listing.contains("__aun_jtbl_dispatch:")
                        && !child_b.asm_listing.contains("__aun_jtbl_dispatch:"),
                    "overlays must reuse Resident __aun_jtbl_dispatch without redefining it"
                );
            } else {
                assert!(
                    parent.asm_listing.contains("__aun_jtbl_dispatch:"),
                    "parent overlay should emit __aun_jtbl_dispatch when Resident has none:\n{}",
                    parent.asm_listing
                );
                assert!(
                    !child_a.asm_listing.contains("__aun_jtbl_dispatch:")
                        && !child_b.asm_listing.contains("__aun_jtbl_dispatch:"),
                    "child overlays must reuse parent __aun_jtbl_dispatch without redefining it"
                );
            }

            // 2. Resident __aun_out_* must not leak as `.equ` into overlays
            assert!(
                !parent.asm_listing.contains("__aun_out_0 =")
                    && !child_a.asm_listing.contains("__aun_out_0 ="),
                "internal LIR symbols must not leak into overlay equ headers"
            );

            // 3. Parent's external .s block appended verbatim after LIR passes
            assert!(
                parent.asm_listing.contains("; untouchable-ext")
                    && parent.asm_listing.contains("ovl_ext_tag:"),
                "parent external .s must remain untouched after LIR"
            );

            // 4. Overlay outlined helpers use per-blob namespacing and contribute to `max_stack_bytes`
            let total_ovl_outlined: usize = out
                .overlays
                .iter()
                .filter(|o| o.asm_listing.contains(&format!("__aun_out_ovl{}_", o.id)))
                .count();
            assert!(
                total_ovl_outlined >= 1,
                "expected at least one overlay to emit namespaced __aun_out_ovl<id>_* helpers"
            );

            // Verify overlay stack accounting includes active-path overlay extra stack
            let res_lir = lir::parse_zilog_to_lir(&out.asm_listing);
            let res_out_extra = outliner::max_outlined_stack_bytes(&res_lir);
            let ovl_extras: Vec<u16> = out
                .overlays
                .iter()
                .map(|o| {
                    let items = lir::parse_zilog_to_lir(&o.asm_listing);
                    outliner::max_outlined_stack_bytes(&items)
                })
                .collect();
            let worst_path = ovl_extras[0] + ovl_extras[1].max(ovl_extras[2]);
            let base_no_res_or_ovl = crate::uchiguri::allocate_uchiguri(&prog, 0x9000)
                .unwrap()
                .max_stack_bytes;
            assert!(
                out.max_stack_bytes >= res_out_extra + worst_path,
                "max_stack_bytes ({}) must reflect overlay outlined stack (base={}, res_out={}, ovl={:?})",
                out.max_stack_bytes,
                base_no_res_or_ovl,
                res_out_extra,
                ovl_extras
            );
        }
    }

    #[test]
    fn issue54_ram_stack_bounds_overlay_alignment_and_auto_ram_base() {
        // 1. RAM-vs-stack overlap (via external .s _BSS pushing final_ram_end past stack.start)
        let stack_src = r#"
            target pc88_bounds : z80 {
                space MainRam   @ 0x0100..0xBFFF;
                stack MainStack @ 0xBF00..0xBFFF;
            }
            static mut WORK: [u8; 32] = [0u8; 32];
            pub fn main() -> u8 { WORK[0] }
        "#;
        let prog = aun_front::compile_str(0, stack_src).expect("front");
        let big_ext = ExternalAsm {
            name: "ext_bss.s".to_string(),
            source: "        .area _BSS\next_big_buf: .ds 300\n        .area _CODE\n".to_string(),
            owner_state: None,
            is_startup: false,
        };
        let err = compile_z80_with_external_asms(
            &prog,
            &[big_ext.clone()],
            CompileOptions {
                code_origin: 0x0100,
                ram_base: 0xBE00,
                ..CompileOptions::release()
            },
        )
        .expect_err("expected RAM + ext .s _BSS > stack.start error");
        assert!(
            err.contains("overlaps stack `MainStack`"),
            "unexpected stack overlap error: {err}"
        );

        // 2. RAM upper space bound check (and allow separate space above stack)
        let upper_src = r#"
            target upper_bounds : z80 {
                space LowRam @ 0x0100..0x8FFF;
                stack Stk    @ 0x8F00..0x8FFF;
                space HiRam  @ 0xC000..0xC03F;
            }
            static mut BIG: [u8; 80] = [0u8; 80];
            pub fn main() -> u8 { BIG[0] }
        "#;
        let upper_prog = aun_front::compile_str(0, upper_src).expect("front");
        let err_hi = compile_z80_with_options(
            &upper_prog,
            &[],
            CompileOptions {
                code_origin: 0x0100,
                ram_base: 0xC000,
                ..CompileOptions::release()
            },
        )
        .expect_err("expected RAM exceeding HiRam end 0xC03F error");
        assert!(
            err_hi.contains("exceeds space `HiRam`"),
            "unexpected upper space error: {err_hi}"
        );

        // 3. Auto ram_base packs RAM + aligned globals + external .s _BSS cleanly below MainStack
        let auto_src = r#"
            target auto_ram : z80 {
                space MainRam   @ 0x0100..0xBFFF;
                stack MainStack @ 0xBF00..0xBFFF;
            }
            #[align(256)]
            static mut ALIGNED_TBL: [u8; 16] = [1u8; 16];
            static mut COUNTER: u16 = 0u16;
            pub fn main() -> u16 {
                COUNTER = COUNTER +% (ALIGNED_TBL[0] as u16);
                COUNTER
            }
        "#;
        let auto_prog = aun_front::compile_str(0, auto_src).expect("front");
        let auto_out = compile_z80_with_external_asms(
            &auto_prog,
            &[big_ext],
            CompileOptions {
                code_origin: 0x0100,
                auto_ram_base: true,
                ..CompileOptions::release()
            },
        )
        .expect("auto ram_base should converge cleanly");
        assert_eq!(auto_out.ram_base & 0x00FF, 0, "ram_base respects #[align(256)]");
        assert!(
            auto_out.ram_end <= 0xBF00,
            "auto ram_end 0x{:04X} must not overlap stack @ 0xBF00",
            auto_out.ram_end
        );
        assert!(
            0xBF00 - auto_out.ram_end < 256,
            "auto ram_base should pack within one alignment unit below stack (ram_base=0x{:04X}, ram_end=0x{:04X})",
            auto_out.ram_base,
            auto_out.ram_end
        );

        // 4. Overlay load alignment enforcement (256B sector granularity + parent/child origin alignment)
        let ovl_src = r#"
            target ovl_align : z80 {
                space MainRam   @ 0x0100..0xBFFF;
                stack MainStack @ 0xBF00..0xBFFF;
                on_swap(ops: &const [LoadOp]) => do_swap(ops);
            }
            fn do_swap(_ops: &const [LoadOp]) {}
            machine M {
                state Idle;
                #[swap_boundary]
                state Parent {
                    #[swap_boundary]
                    state Child;
                    initial Child;
                }
                transitions {
                    Idle -> Parent;
                    Parent -> Idle;
                }
                on Idle {}
                on Parent {
                    fn p_step(&mut self, x: u8) -> u8 { x +% 1u8 }
                }
                on Parent::Child {
                    fn c_step(&mut self, x: u8) -> u8 { self.p_step(x) +% 2u8 }
                }
            }
            pub fn main() -> u8 {
                let mut m: M = M::new();
                m.goto(Parent::Child {});
                m.c_step(5u8)
            }
        "#;
        let ovl_prog = aun_front::compile_str(0, ovl_src).expect("front");
        let base_out = compile_z80_with_options(
            &ovl_prog,
            &[],
            CompileOptions {
                code_origin: 0x0100,
                ram_base: 0x8000,
                overlay_load_align: 1,
                ..CompileOptions::release()
            },
        )
        .expect("baseline compile");
        let p_origin = base_out.overlays[0].origin;
        let p_raw_size = base_out.overlays[0].bin.len();
        assert!(p_raw_size < 256 && p_raw_size > 0);

        // Set ram_base so Parent raw size fits (`p_origin + p_raw_size <= tight_ram`),
        // yet 256B-aligned load (`p_origin + 256`) overlaps `tight_ram`.
        let tight_ram = p_origin + (p_raw_size as u16) + 16;
        assert!(
            tight_ram < p_origin + 256,
            "expected gap between raw end and 256B aligned end"
        );
        // Byte-exact passes:
        compile_z80_with_options(
            &ovl_prog,
            &[],
            CompileOptions {
                code_origin: 0x0100,
                ram_base: tight_ram,
                overlay_load_align: 1,
                ..CompileOptions::release()
            },
        )
        .expect("byte-exact fit succeeds");
        // 256B sector-aligned rejects:
        let align_err = compile_z80_with_options(
            &ovl_prog,
            &[],
            CompileOptions {
                code_origin: 0x0100,
                ram_base: tight_ram,
                overlay_load_align: 256,
                ..CompileOptions::release()
            },
        )
        .expect_err("256B aligned overlay load must reject overlap with tight_ram");
        assert!(
            align_err.contains("aligned 256 bytes to 256B"),
            "unexpected aligned overlay error: {align_err}"
        );

        // When plenty of room exists, Child origin after Parent is aligned up to 256B boundary
        let aligned_ok = compile_z80_with_options(
            &ovl_prog,
            &[],
            CompileOptions {
                code_origin: 0x0100,
                auto_ram_base: true,
                overlay_load_align: 256,
                ..CompileOptions::release()
            },
        )
        .expect("aligned overlays + auto_ram_base succeed");
        assert_eq!(aligned_ok.overlays[0].aligned_size, 256);
        assert_eq!(
            aligned_ok.overlays[1].origin,
            aligned_ok.overlays[0].origin + 256,
            "child overlay must start after parent's 256B aligned footprint"
        );
    }
}
