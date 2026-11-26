//! Phase 1 Z80 code generator lowering `aun-ir` + *Uchiguri* static RAM layout to Z80 assembly.

mod binop;
mod func;
mod inst;
mod runtime;

pub use crate::analysis::z80_sym;
use crate::analysis::{FnAnalysis, ProgramAnalysis};
use crate::uchiguri::RamLayout;
use crate::CompileOptions;
use aun_ir::*;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write;

#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) enum IndexKey {
    RegC,
    Mem(u16, u16), // (addr, byte_size)
}

/// Shared read-only program environment for emitting all functions into Z80 assembly.
pub(super) struct EmitGlobalEnv<'a> {
    pub program: &'a IrProgram,
    pub analysis: &'a ProgramAnalysis,
    pub layout: &'a RamLayout,
    pub opts: CompileOptions,
    pub shared_muls: &'a BTreeSet<u16>,
}

/// Mutable per-function / per-basic-block lowering state used while emitting a function.
pub(super) struct EmitCtx<'a> {
    pub program: &'a IrProgram,
    pub analysis: &'a ProgramAnalysis,
    pub func: &'a IrFunction,
    pub fa: &'a FnAnalysis,
    pub layout: &'a RamLayout,
    pub scratch_base: u16,
    pub val_offsets: &'a [u16],
    pub opts: CompileOptions,
    pub shared_muls: &'a BTreeSet<u16>,
    pub fused_skip_insts: &'a HashSet<(usize, usize)>,
    pub fused_xor_shift: &'a HashMap<ValueId, (ValueId, IrBinOp, u8)>,
    pub fused_q10_6_u8: &'a HashMap<ValueId, ValueId>,
    pub block_nonneg_ix: Option<i8>,
    pub nonneg_vals: BTreeSet<ValueId>,
    pub const_vals: HashMap<ValueId, i64>,
    pub const_addrs: HashMap<ValueId, u16>,
    pub rom_addrs: HashMap<ValueId, (String, u16)>,
    pub val_aliases: HashMap<ValueId, u16>,
    pub ix_addrs: HashMap<ValueId, i8>,
    pub ix_param_addr_vals: BTreeSet<ValueId>,
    pub ix_val_loads: HashMap<ValueId, (i8, u16)>,
    pub val_in_c: BTreeSet<ValueId>,
    pub shift_cse: HashMap<(u16, bool, bool), (u8, u16)>,
    pub index_cse: HashMap<(String, IndexKey, u16), u16>,
    pub label_counter: u32,
    pub needs_panic_stub: &'a mut bool,
    pub asm: &'a mut String,
}

impl<'a> EmitCtx<'a> {
    pub(super) fn reset_bb(&mut self, block_nonneg_ix: Option<i8>) {
        self.shift_cse.clear();
        self.index_cse.clear();
        self.ix_val_loads.clear();
        self.val_in_c.clear();
        self.nonneg_vals.clear();
        self.block_nonneg_ix = block_nonneg_ix;
    }

    pub(super) fn val_addr(&self, v: ValueId) -> u16 {
        let off = self
            .val_offsets
            .get(v as usize)
            .copied()
            .unwrap_or_else(|| (v as u16).wrapping_mul(2));
        self.scratch_base.wrapping_add(off)
    }

    pub(super) fn val_loc(&self, v: ValueId) -> u16 {
        self.val_aliases
            .get(&v)
            .copied()
            .unwrap_or_else(|| self.val_addr(v))
    }

    pub(super) fn fresh_label(&mut self, tag: &str) -> String {
        let id = self.label_counter;
        self.label_counter += 1;
        format!("{}_{tag}_{id}", z80_sym(&self.func.name))
    }

    pub(super) fn emit_load_a_from(&mut self, v: ValueId) {
        if let Some(&c) = self.const_vals.get(&v) {
            let b = (c & 0xFF) as u8;
            if b == 0 {
                writeln!(self.asm, "    xor a").unwrap();
            } else {
                writeln!(self.asm, "    ld a, 0x{b:02X}").unwrap();
            }
        } else if self.val_in_c.contains(&v) {
            writeln!(self.asm, "    ld a, c").unwrap();
        } else if let Some(&(disp, _)) = self.ix_val_loads.get(&v) {
            writeln!(self.asm, "    ld a, (ix + {disp})").unwrap();
        } else {
            let va = self.val_loc(v);
            writeln!(self.asm, "    ld a, (0x{va:04X})").unwrap();
        }
    }

    pub(super) fn emit_load_hl_from(&mut self, v: ValueId) {
        if let Some(&c) = self.const_vals.get(&v) {
            writeln!(self.asm, "    ld hl, 0x{:04X}", c as u16).unwrap();
        } else if let Some(&(disp, _)) = self.ix_val_loads.get(&v) {
            writeln!(self.asm, "    ld l, (ix + {disp})").unwrap();
            writeln!(self.asm, "    ld h, (ix + {})", disp + 1).unwrap();
        } else {
            let va = self.val_loc(v);
            writeln!(self.asm, "    ld hl, (0x{va:04X})").unwrap();
        }
    }

    pub(super) fn emit_load_de_from(&mut self, v: ValueId) {
        if let Some(&c) = self.const_vals.get(&v) {
            writeln!(self.asm, "    ld de, 0x{:04X}", c as u16).unwrap();
        } else if let Some(&(disp, _)) = self.ix_val_loads.get(&v) {
            writeln!(self.asm, "    ld e, (ix + {disp})").unwrap();
            writeln!(self.asm, "    ld d, (ix + {})", disp + 1).unwrap();
        } else {
            let va = self.val_loc(v);
            writeln!(self.asm, "    ld de, (0x{va:04X})").unwrap();
        }
    }
}

/// Emits Z80 assembly text for all non-test functions in `program` (Debug defaults).
pub fn emit_z80_asm(
    program: &IrProgram,
    layout: &RamLayout,
    code_origin: u16,
) -> Result<String, String> {
    emit_z80_asm_with_options(
        program,
        layout,
        CompileOptions {
            code_origin,
            ..CompileOptions::debug()
        },
    )
}

/// Emits Z80 assembly text with explicit debug/release check options.
pub fn emit_z80_asm_with_options(
    program: &IrProgram,
    layout: &RamLayout,
    opts: CompileOptions,
) -> Result<String, String> {
    let analysis = ProgramAnalysis::analyze(program);
    emit_z80_asm_with_analysis(program, &analysis, layout, opts)
}

/// Emits Z80 assembly text using precomputed `ProgramAnalysis`.
pub fn emit_z80_asm_with_analysis(
    program: &IrProgram,
    analysis: &ProgramAnalysis,
    layout: &RamLayout,
    opts: CompileOptions,
) -> Result<String, String> {
    let mut asm = String::new();
    let mode = if opts.release { "release" } else { "debug" };
    writeln!(asm, "; Generated by aun-z80 ({mode})").unwrap();
    writeln!(asm, ".org 0x{:04X}", opts.code_origin).unwrap();
    writeln!(
        asm,
        "; Uchiguri RAM: 0x{:04X}..0x{:04X} ({} bytes), max stack: {} bytes",
        layout.ram_base,
        layout.ram_end,
        layout.ram_end.saturating_sub(layout.ram_base),
        layout.max_stack_bytes
    )
    .unwrap();
    for g in &layout.global_entries {
        writeln!(asm, "{} = 0x{:04X}", g.name, g.addr).unwrap();
    }

    let mut needs_panic_stub = false;
    let mut needs_noop_ret = false;

    // Count non-power-of-two IndexAddr/Pool `elem_size` constants across the program:
    // outline any `k >= 5` used >= 2 times into a shared `__aun_mul_hl_<k>` helper (3B call!).
    let mut mul_counts: HashMap<u16, usize> = HashMap::new();
    for f in &program.functions {
        if f.is_test || f.overlay_state.is_some() {
            continue;
        }
        for bb in &f.blocks {
            for inst in &bb.insts {
                if let IrInst::IndexAddr { elem_size, .. }
                | IrInst::PoolElemAddr { elem_size, .. } = inst
                    && *elem_size >= 5
                    && !elem_size.is_power_of_two()
                {
                    *mul_counts.entry(*elem_size).or_default() += 1;
                }
            }
        }
    }
    let shared_muls: BTreeSet<u16> = mul_counts
        .into_iter()
        .filter(|&(_, cnt)| cnt >= 2)
        .map(|(k, _)| k)
        .collect();

    let env = EmitGlobalEnv {
        program,
        analysis,
        layout,
        opts,
        shared_muls: &shared_muls,
    };

    for func in &program.functions {
        if func.is_test || func.overlay_state.is_some() {
            continue;
        }
        func::emit_function(&env, func, &mut needs_panic_stub, &mut asm)?;
    }

    // Emit shared `__aun_mul_a_<k>` / `__aun_mul_hl_<k>` helpers (clobbers HL, DE; preserves A/BC/IX)
    for &k in &shared_muls {
        writeln!(asm, "\n__aun_mul_a_{k}:").unwrap();
        writeln!(asm, "    ld l, a").unwrap();
        writeln!(asm, "    ld h, 0").unwrap();
        writeln!(asm, "__aun_mul_hl_{k}:").unwrap();
        runtime::emit_mul_hl_const(k, &mut asm);
        writeln!(asm, "    ret").unwrap();
    }

    // Emit state machine dispatch tables, SMC stubs, and transition bitmasks
    for mdef in &program.machines {
        let mut method_names: BTreeSet<String> = BTreeSet::new();
        let mut has_on_exit = false;
        for st in &mdef.states {
            if st.on_exit.is_some() {
                has_on_exit = true;
            }
            for m in st.methods.keys() {
                method_names.insert(m.clone());
            }
        }
        // Transition bitmask table (omitted in Release when transition checks are disabled!)
        if opts.check_transitions {
            writeln!(asm, "\n{}_tmask:", mdef.name).unwrap();
            for &mask in &mdef.transitions {
                writeln!(asm, "    .dw 0x{:04X}", (mask & 0xFFFF) as u16).unwrap();
            }
        }
        if has_on_exit {
            needs_noop_ret = true;
            runtime::emit_dispatch_stub(
                program,
                mdef,
                "on_exit",
                false,
                |st| st.on_exit,
                &mut asm,
            );
        }
        for method in &method_names {
            needs_noop_ret = true;
            runtime::emit_dispatch_stub(
                program,
                mdef,
                method,
                mdef.smc,
                |st| st.methods.get(method).copied(),
                &mut asm,
            );
        }
    }

    if needs_noop_ret {
        writeln!(asm, "\n__aun_noop_ret:").unwrap();
        writeln!(asm, "    ret").unwrap();
    }

    // Emit ROM constant tables (`const TABLE: [T; N] = [...]`) directly in code space
    for g in &program.globals {
        if !g.in_rom {
            continue;
        }
        writeln!(asm, "\n{}:", g.name).unwrap();
        for chunk in g.init.chunks(16) {
            let bytes: Vec<String> = chunk.iter().map(|b| format!("0x{b:02X}")).collect();
            writeln!(asm, "    .db {}", bytes.join(", ")).unwrap();
        }
    }

    let user_defined_panic =
        program.functions.iter().any(|f| !f.is_test && f.name == "__aun_panic")
            || program.externs.iter().any(|e| e.name == "__aun_panic");
    if needs_panic_stub && !user_defined_panic {
        writeln!(asm, "\n__aun_panic:").unwrap();
        writeln!(asm, "    di").unwrap();
        writeln!(asm, "    halt").unwrap();
    }

    Ok(asm)
}

/// Emits Z80 assembly for a single `#[overlay(Slot)]` state binary linked at `slot_origin`.
///
/// The overlay starts with a 3-byte-per-method jump table at `slot_origin` (ordered by
/// `on_enter`, `on_exit`, then sorted dispatched methods) so resident code can reference
/// overlay entry points at deterministic addresses, followed by the actual method bodies.
#[allow(clippy::too_many_arguments)]
pub fn emit_overlay_z80_asm(
    program: &IrProgram,
    analysis: &ProgramAnalysis,
    layout: &RamLayout,
    opts: CompileOptions,
    slot_name: &str,
    state_name: &str,
    slot_origin: u16,
    resident_symbols: &HashMap<String, u16>,
) -> Result<String, String> {
    let mut asm = String::new();
    writeln!(
        asm,
        "; Overlay {slot_name}::{state_name} linked at 0x{slot_origin:04X}"
    )
    .unwrap();
    writeln!(asm, ".org 0x{slot_origin:04X}").unwrap();

    // Import resident symbols and global RAM addresses as `.equ`
    let mut sorted_syms: Vec<(&String, &u16)> = resident_symbols.iter().collect();
    sorted_syms.sort_by_key(|(k, _)| k.as_str());
    for (name, addr) in sorted_syms {
        writeln!(asm, "{name} = 0x{addr:04X}").unwrap();
    }
    for g in &layout.global_entries {
        if !resident_symbols.contains_key(&g.name) {
            writeln!(asm, "{} = 0x{:04X}", g.name, g.addr).unwrap();
        }
    }

    let ovl_fns: Vec<&IrFunction> = program
        .functions
        .iter()
        .filter(|f| {
            !f.is_test
                && f.overlay_state
                    .as_ref()
                    .is_some_and(|(sl, st)| sl == slot_name && st == state_name)
        })
        .collect();

    // Fixed jump-table header at `slot_origin`
    writeln!(asm, "ovl_{state_name}_entry:").unwrap();
    if ovl_fns.is_empty() {
        writeln!(asm, "    ret").unwrap();
        return Ok(asm);
    }
    for f in &ovl_fns {
        writeln!(asm, "    jp {}", z80_sym(&f.name)).unwrap();
    }

    let mut needs_panic_stub = false;
    let empty_muls = BTreeSet::new();
    let env = EmitGlobalEnv {
        program,
        analysis,
        layout,
        opts,
        shared_muls: &empty_muls,
    };
    for func in ovl_fns {
        func::emit_function(&env, func, &mut needs_panic_stub, &mut asm)?;
    }

    Ok(asm)
}
