//! Phase 1 Z80 code generator lowering `aun-ir` + *Uchiguri* static RAM layout to Z80 assembly.

mod binop;
pub(crate) mod func;
mod inst;
mod runtime;

pub use crate::analysis::z80_sym;
use crate::analysis::{ConstAddrSrc, FnAnalysis, ProgramAnalysis};
use crate::overlay::{ExternalAsm, LoadOpSpec, OverlayOwner, OverlayPlan};
use crate::regalloc::{RegLoc, RegMask};
use crate::uchiguri::RamLayout;
use crate::CompileOptions;
use aun_ir::*;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub(super) struct TrampolineKey {
    pub machine_id: u32,
    pub from_state: Option<u8>,
    pub to_state: u8,
    pub copy_start: u16,
    pub copy_end: u16,
}

impl TrampolineKey {
    pub(super) fn label(self) -> String {
        match self.from_state {
            Some(f) => format!(
                "__aun_tr_m{}_f{}_t{}_{}_{}",
                self.machine_id, f, self.to_state, self.copy_start, self.copy_end
            ),
            None => format!(
                "__aun_tr_m{}_dyn_t{}_{}_{}",
                self.machine_id, self.to_state, self.copy_start, self.copy_end
            ),
        }
    }
}

/// Resolves a function call/jump target symbol or concrete overlay address.
/// During Pass 1 (before overlays are linked), emits deterministic non-zero `0xF001 + fid`
/// placeholders so Resident instructions keep fixed width across Pass 1 -> Pass 3.
pub(super) fn resolve_fn_target(
    program: &IrProgram,
    plan: &OverlayPlan,
    resolved_overlay_syms: &HashMap<String, u16>,
    pass1_placeholders: bool,
    fid: u32,
) -> String {
    let sym = z80_sym(&program.functions[fid as usize].name);
    if !plan.is_fn_in_overlay(fid as usize) {
        return sym;
    }
    if let Some(&addr) = resolved_overlay_syms.get(&sym) {
        return format!("0x{addr:04X}");
    }
    if pass1_placeholders {
        return format!("0x{:04X}", 0xF001u16.wrapping_add(fid as u16));
    }
    sym
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum IndexKey {
    Reg(RegLoc),
    Mem(u16, u16), // (addr, byte_size)
    Const(i64),
}

/// Fused 8-bit term `((x_v & mask8) as u16) << k` in a `u16` shift or `BitOr` pack.
#[derive(Clone, Copy, Debug)]
pub(super) struct U8PackTerm {
    pub x_v: ValueId,
    pub k: u8,
    pub mask8: u8,
}

/// Action for an instruction participating in a `DE`-accumulated `u16` bit-pack `BitOr` chain.
#[derive(Clone, Debug)]
pub(super) enum BitPackAction {
    U8Term {
        pre_other: Option<ValueId>,
        term: U8PackTerm,
    },
    Other16 {
        val: ValueId,
        val2: Option<ValueId>,
    },
    Finish {
        tail_u8: Option<U8PackTerm>,
        tail_other1: Option<ValueId>,
        tail_other2: Option<ValueId>,
        const_lo: u8,
        const_hi: u8,
    },
}

/// Tracks optional runtime helpers referenced during Z80 codegen so unused helpers
/// add 0 bytes to emitted programs.
#[derive(Default)]
pub(super) struct RuntimeNeeds {
    pub panic_stub: bool,
    pub jp_iy: bool,
    pub cmp_s16: bool,
    pub mul8: bool,
    pub mul32: bool,
    pub mulu16_32: bool,
    pub muls16_32: bool,
    pub udiv32_8: bool,
    pub udiv32: bool,
    pub sdiv32: bool,
    pub udiv16: bool,
    pub sdiv16: bool,
    pub udiv8: bool,
    pub sdiv8: bool,
    pub trampolines: BTreeSet<TrampolineKey>,
}

/// Shared read-only program environment for emitting all functions into Z80 assembly.
pub(super) struct EmitGlobalEnv<'a> {
    pub program: &'a IrProgram,
    pub analysis: &'a ProgramAnalysis,
    pub layout: &'a RamLayout,
    pub opts: CompileOptions,
    pub shared_muls: &'a BTreeSet<u16>,
    pub use_cmp_s16_helper: bool,
    pub plan: &'a OverlayPlan,
    pub resolved_overlay_syms: &'a HashMap<String, u16>,
    pub pass1_placeholders: bool,
}

fn count_cold_signed_i16_cmps(program: &IrProgram, analysis: &ProgramAnalysis) -> usize {
    let mut count = 0usize;
    for (fi, f) in program.functions.iter().enumerate() {
        if f.is_test
            || f.blocks.is_empty()
            || analysis.fns.get(fi).is_some_and(|fa| fa.is_loop_hot)
        {
            continue;
        }
        let cfg = aun_ir::cfg::Cfg::build(f);
        let mut loop_bbs: HashSet<BlockId> = HashSet::new();
        for lp in cfg.natural_loops() {
            loop_bbs.extend(lp.body.iter().copied());
        }
        let mut consts: HashMap<ValueId, i64> = HashMap::new();
        for bb in &f.blocks {
            for inst in &bb.insts {
                if let IrInst::ConstInt { dst, val, .. } = inst {
                    consts.insert(*dst, *val);
                }
            }
        }
        for bb in &f.blocks {
            if loop_bbs.contains(&bb.id) {
                continue;
            }
            for inst in &bb.insts {
                let IrInst::BinOp {
                    op, ty, lhs, rhs, ..
                } = inst
                else {
                    continue;
                };
                if !matches!(
                    op,
                    IrBinOp::Lt | IrBinOp::Le | IrBinOp::Gt | IrBinOp::Ge
                ) || program.type_size(ty) != 2
                    || !crate::analysis::is_signed_ty(ty)
                {
                    continue;
                }
                let is_hi_aligned = if let Some(&c) = consts.get(rhs) {
                    let c16 = c as i16;
                    match op {
                        IrBinOp::Lt | IrBinOp::Ge => (c16 as u16) & 0x00FF == 0,
                        IrBinOp::Le | IrBinOp::Gt => {
                            c16 < i16::MAX && ((c16 + 1) as u16) & 0x00FF == 0
                        }
                        _ => false,
                    }
                } else if let Some(&c) = consts.get(lhs) {
                    let c16 = c as i16;
                    match op {
                        IrBinOp::Gt | IrBinOp::Le => (c16 as u16) & 0x00FF == 0,
                        IrBinOp::Ge | IrBinOp::Lt => {
                            c16 < i16::MAX && ((c16 + 1) as u16) & 0x00FF == 0
                        }
                        _ => false,
                    }
                } else {
                    false
                };
                if !is_hi_aligned {
                    count += 1;
                }
            }
        }
    }
    count
}

pub(super) fn fmt_uchiguri_imm(ram_base: u16, addr: u16) -> String {
    let off = addr.wrapping_sub(ram_base);
    if off == 0 {
        "__aun_ram".to_string()
    } else {
        format!("__aun_ram + {off}")
    }
}

pub(super) fn fmt_uchiguri_mem(ram_base: u16, addr: u16) -> String {
    format!("({})", fmt_uchiguri_imm(ram_base, addr))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum HlBase {
    RamPtr(u16),
    BcReg,
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
    pub fused_ix_step: &'a HashMap<ValueId, i16>,
    pub fused_reg_step: &'a HashMap<ValueId, i16>,
    pub fused_index_u8_cast: &'a HashMap<ValueId, ValueId>,
    pub fused_xor_shift: &'a HashMap<ValueId, (ValueId, IrBinOp, u8)>,
    pub fused_q10_6_u8: &'a HashMap<ValueId, ValueId>,
    pub fused_hi8_u8: &'a HashMap<ValueId, ValueId>,
    pub fused_u8_shl16: &'a HashMap<ValueId, U8PackTerm>,
    pub fused_bitor_u8_term: &'a HashMap<ValueId, (ValueId, U8PackTerm)>,
    pub bitpack_actions: &'a HashMap<ValueId, BitPackAction>,
    pub fused_divrem_partner: &'a HashMap<ValueId, (IrBinOp, ValueId)>,
    pub u8_bound_vals: &'a HashSet<ValueId>,
    pub u16_bound_vals: &'a HashSet<ValueId>,
    pub i16_bound_vals: &'a HashSet<ValueId>,
    pub fn_possible_bits: &'a HashMap<ValueId, u16>,
    pub fn_slot_possible_bits: &'a HashMap<SlotId, u16>,
    pub body_save_ix: bool,
    pub addr_taken_locals: &'a HashSet<SlotId>,
    pub escaped_slot_ranges: &'a [(u16, u16)],
    pub block_nonneg_ix: Option<i8>,
    pub nonneg_vals: BTreeSet<ValueId>,
    pub const_vals: HashMap<ValueId, i64>,
    pub possible_bits: HashMap<ValueId, u16>,
    pub local_possible_bits: HashMap<SlotId, u16>,
    pub de_d_init: bool,
    pub de_e_init: bool,
    pub const_addrs: HashMap<ValueId, (u16, bool)>,
    pub const_addr_srcs: HashMap<ValueId, ConstAddrSrc>,
    pub const_addr_loads: HashMap<ValueId, (u16, bool)>,
    pub rom_addrs: HashMap<ValueId, (String, u16)>,
    pub val_aliases: HashMap<ValueId, u16>,
    pub val_from_local: HashMap<ValueId, SlotId>,
    pub const_slices: HashMap<ValueId, (ValueId, ValueId)>,
    pub ix_addrs: HashMap<ValueId, i8>,
    pub outer_ix_addrs: HashMap<ValueId, i8>,
    pub ix_param_addr_vals: BTreeSet<ValueId>,
    pub ix_val_loads: HashMap<ValueId, (i8, u16)>,
    pub aligned256_addrs: HashMap<ValueId, (u8, ValueId)>,
    pub val_in_reg: HashMap<ValueId, RegLoc>,
    pub shift_cse: HashMap<(u16, bool, bool), (u8, u16)>,
    pub index_cse: HashMap<(String, IndexKey, u16), u16>,
    pub hl_ptr: Option<(HlBase, i32)>,
    pub hl_val: Option<ValueId>,
    pub label_counter: u32,
    pub plan: &'a OverlayPlan,
    pub resolved_overlay_syms: &'a HashMap<String, u16>,
    pub pass1_placeholders: bool,
    pub cur_bb_is_tail_return_none: bool,
    pub cur_bb_in_loop: bool,
    pub cur_bb_inst_len: usize,
    pub use_cmp_s16_helper: bool,
    pub tail_jumped_trampoline: bool,
    pub tail_cold_stubs: String,
    pub runtime_needs: &'a mut RuntimeNeeds,
    pub asm: &'a mut String,
}

impl<'a> EmitCtx<'a> {
    pub(super) fn possible_bits(&self, v: ValueId) -> u16 {
        if let Some(&c) = self.const_vals.get(&v) {
            return c as u16;
        }
        self.possible_bits
            .get(&v)
            .copied()
            .or_else(|| self.fn_possible_bits.get(&v).copied())
            .unwrap_or(0xFFFF)
    }

    pub(super) fn record_ix_addr(
        &mut self,
        bb_idx: usize,
        inst_idx: usize,
        dst: ValueId,
        disp: i8,
    ) {
        self.ix_addrs.insert(dst, disp);
        if let Some(ref reg) = self.fa.ix_inner_region {
            let bb_id = self.func.blocks[bb_idx].id;
            if !reg.inner_bbs.contains(&bb_id)
                && !(reg.pre_bb_idx == bb_idx && inst_idx >= reg.init_inst_idx)
            {
                self.outer_ix_addrs.insert(dst, disp);
            }
        }
    }

    pub(super) fn reset_bb(&mut self, block_nonneg_ix: Option<i8>, bb_id: BlockId) {
        self.shift_cse.clear();
        self.index_cse.clear();
        self.hl_ptr = None;
        self.hl_val = None;
        if self
            .fa
            .ix_inner_region
            .as_ref()
            .is_some_and(|r| r.inner_bbs.contains(&bb_id))
        {
            self.ix_addrs.clear();
        } else if self.fa.ix_inner_region.is_some() {
            self.ix_addrs = self.outer_ix_addrs.clone();
        } else if self.fa.ix_local_slot.is_some() {
            self.ix_addrs.clear();
        }
        self.ix_val_loads.clear();
        self.const_addr_loads.clear();
        self.aligned256_addrs.clear();
        self.val_in_reg.clear();
        self.nonneg_vals.clear();
        self.local_possible_bits.clear();
        self.de_d_init = false;
        self.de_e_init = false;
        self.block_nonneg_ix = block_nonneg_ix;
        self.tail_jumped_trampoline = false;
    }

    pub(super) fn invalidate_hl_ram(&mut self, addr: u16, sz: u16) {
        if let Some((HlBase::RamPtr(ptr_addr), _)) = self.hl_ptr {
            let p_start = ptr_addr as u32;
            let p_end = p_start + 2;
            let a_start = addr as u32;
            let a_end = a_start + (sz.max(1) as u32);
            if p_start < a_end && a_start < p_end {
                self.hl_ptr = None;
            }
        }
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

    pub(super) fn fmt_ram_imm(&self, addr: u16) -> String {
        fmt_uchiguri_imm(self.layout.ram_base, addr)
    }

    pub(super) fn fmt_ram_mem(&self, addr: u16) -> String {
        fmt_uchiguri_mem(self.layout.ram_base, addr)
    }

    pub(super) fn fmt_addr_imm(&self, addr: u16, is_uchiguri: bool) -> String {
        if is_uchiguri {
            self.fmt_ram_imm(addr)
        } else {
            format!("0x{addr:04X}")
        }
    }

    pub(super) fn fmt_addr_mem(&self, addr: u16, is_uchiguri: bool) -> String {
        if is_uchiguri {
            self.fmt_ram_mem(addr)
        } else {
            format!("(0x{addr:04X})")
        }
    }

    pub(super) fn fmt_val_mem(&self, v: ValueId) -> String {
        self.fmt_val_addr_mem(v, 0)
    }

    pub(super) fn fmt_val_addr_mem(&self, v: ValueId, byte_off: u16) -> String {
        if let Some(&(addr, is_u)) = self.const_addr_loads.get(&v) {
            self.fmt_addr_mem(addr.wrapping_add(byte_off), is_u)
        } else {
            self.fmt_ram_mem(self.val_loc(v).wrapping_add(byte_off))
        }
    }

    pub(super) fn fmt_dst_mem(&self, dst: ValueId) -> String {
        self.fmt_ram_mem(self.val_addr(dst))
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
        } else if let Some(&reg) = self.val_in_reg.get(&v) {
            writeln!(self.asm, "    ld a, {}", reg.reg8_name()).unwrap();
        } else if let Some(&(disp, _)) = self.ix_val_loads.get(&v) {
            writeln!(self.asm, "    ld a, (ix + {disp})").unwrap();
        } else {
            let va = self.fmt_val_mem(v);
            writeln!(self.asm, "    ld a, {va}").unwrap();
        }
    }

    pub(super) fn emit_load_hl_from(&mut self, v: ValueId) {
        if self.hl_val == Some(v) {
            return;
        }
        self.hl_ptr = None;
        self.hl_val = None;
        if let Some(&c) = self.const_vals.get(&v) {
            writeln!(self.asm, "    ld hl, 0x{:04X}", c as u16).unwrap();
        } else if let Some(&(addr, is_u)) = self.const_addrs.get(&v) {
            let imm = self.fmt_addr_imm(addr, is_u);
            writeln!(self.asm, "    ld hl, {imm}").unwrap();
        } else if let Some((sym, off)) = self.rom_addrs.get(&v) {
            if *off == 0 {
                writeln!(self.asm, "    ld hl, {sym}").unwrap();
            } else {
                writeln!(self.asm, "    ld hl, {sym} + {off}").unwrap();
            }
        } else if self.val_in_reg.get(&v) == Some(&RegLoc::BC) {
            writeln!(self.asm, "    ld h, b").unwrap();
            writeln!(self.asm, "    ld l, c").unwrap();
        } else if let Some(&(disp, _)) = self.ix_val_loads.get(&v) {
            writeln!(self.asm, "    ld l, (ix + {disp})").unwrap();
            writeln!(self.asm, "    ld h, (ix + {})", disp + 1).unwrap();
        } else if let Some(&disp) = self.ix_addrs.get(&v) {
            writeln!(self.asm, "    push ix").unwrap();
            writeln!(self.asm, "    pop hl").unwrap();
            match disp {
                0 => {}
                1..=3 => {
                    for _ in 0..disp {
                        writeln!(self.asm, "    inc hl").unwrap();
                    }
                }
                -3..=-1 => {
                    for _ in 0..(-disp) {
                        writeln!(self.asm, "    dec hl").unwrap();
                    }
                }
                _ => {
                    writeln!(self.asm, "    push de").unwrap();
                    writeln!(self.asm, "    ld de, 0x{:04X}", disp as i16 as u16).unwrap();
                    writeln!(self.asm, "    add hl, de").unwrap();
                    writeln!(self.asm, "    pop de").unwrap();
                }
            }
        } else {
            let va = self.fmt_val_mem(v);
            writeln!(self.asm, "    ld hl, {va}").unwrap();
        }
    }

    pub(super) fn emit_load_de_from(&mut self, v: ValueId) {
        if let Some(&c) = self.const_vals.get(&v) {
            writeln!(self.asm, "    ld de, 0x{:04X}", c as u16).unwrap();
        } else if let Some(&(addr, is_u)) = self.const_addrs.get(&v) {
            let imm = self.fmt_addr_imm(addr, is_u);
            writeln!(self.asm, "    ld de, {imm}").unwrap();
        } else if let Some((sym, off)) = self.rom_addrs.get(&v) {
            if *off == 0 {
                writeln!(self.asm, "    ld de, {sym}").unwrap();
            } else {
                writeln!(self.asm, "    ld de, {sym} + {off}").unwrap();
            }
        } else if self.val_in_reg.get(&v) == Some(&RegLoc::BC) {
            writeln!(self.asm, "    ld d, b").unwrap();
            writeln!(self.asm, "    ld e, c").unwrap();
        } else if let Some(&(disp, _)) = self.ix_val_loads.get(&v) {
            writeln!(self.asm, "    ld e, (ix + {disp})").unwrap();
            writeln!(self.asm, "    ld d, (ix + {})", disp + 1).unwrap();
        } else if let Some(&disp) = self.ix_addrs.get(&v) {
            if disp == 0 {
                writeln!(self.asm, "    push ix").unwrap();
                writeln!(self.asm, "    pop de").unwrap();
            } else {
                writeln!(self.asm, "    push hl").unwrap();
                writeln!(self.asm, "    push ix").unwrap();
                writeln!(self.asm, "    pop hl").unwrap();
                match disp {
                    1..=3 => {
                        for _ in 0..disp {
                            writeln!(self.asm, "    inc hl").unwrap();
                        }
                    }
                    -3..=-1 => {
                        for _ in 0..(-disp) {
                            writeln!(self.asm, "    dec hl").unwrap();
                        }
                    }
                    _ => {
                        writeln!(self.asm, "    ld de, 0x{:04X}", disp as i16 as u16).unwrap();
                        writeln!(self.asm, "    add hl, de").unwrap();
                    }
                }
                writeln!(self.asm, "    ex de, hl").unwrap();
                writeln!(self.asm, "    pop hl").unwrap();
            }
        } else {
            let va = self.fmt_val_mem(v);
            writeln!(self.asm, "    ld de, {va}").unwrap();
        }
    }

    pub(super) fn emit_load_hlde_from(&mut self, v: ValueId) {
        self.hl_ptr = None;
        self.hl_val = None;
        if let Some(&(ptr, len)) = self.const_slices.get(&v) {
            self.emit_load_de_from(ptr);
            self.emit_load_hl_from(len);
        } else if let Some(&c) = self.const_vals.get(&v) {
            let u = c as u32;
            writeln!(self.asm, "    ld de, 0x{:04X}", (u & 0xFFFF) as u16).unwrap();
            writeln!(self.asm, "    ld hl, 0x{:04X}", (u >> 16) as u16).unwrap();
        } else if let Some(&(disp, _)) = self.ix_val_loads.get(&v) {
            writeln!(self.asm, "    ld e, (ix + {disp})").unwrap();
            writeln!(self.asm, "    ld d, (ix + {})", disp + 1).unwrap();
            writeln!(self.asm, "    ld l, (ix + {})", disp + 2).unwrap();
            writeln!(self.asm, "    ld h, (ix + {})", disp + 3).unwrap();
        } else if let Some(&(va, is_u)) = self.const_addr_loads.get(&v) {
            let m0 = self.fmt_addr_mem(va, is_u);
            let m2 = self.fmt_addr_mem(va.wrapping_add(2), is_u);
            writeln!(self.asm, "    ld de, {m0}").unwrap();
            writeln!(self.asm, "    ld hl, {m2}").unwrap();
        } else {
            let va = self.val_loc(v);
            let m0 = self.fmt_ram_mem(va);
            let m2 = self.fmt_ram_mem(va.wrapping_add(2));
            writeln!(self.asm, "    ld de, {m0}").unwrap();
            writeln!(self.asm, "    ld hl, {m2}").unwrap();
        }
    }

    pub(super) fn emit_copy32_from_val(&mut self, v: ValueId, dst: u16, dst_uchiguri: bool) {
        if let Some(&c) = self.const_vals.get(&v) {
            self.emit_const32_to(dst, dst_uchiguri, c);
        } else if let Some(&(disp, _)) = self.ix_val_loads.get(&v) {
            let d0 = self.fmt_addr_mem(dst, dst_uchiguri);
            let d2 = self.fmt_addr_mem(dst.wrapping_add(2), dst_uchiguri);
            writeln!(self.asm, "    ld l, (ix + {disp})").unwrap();
            writeln!(self.asm, "    ld h, (ix + {})", disp + 1).unwrap();
            writeln!(self.asm, "    ld {d0}, hl").unwrap();
            writeln!(self.asm, "    ld l, (ix + {})", disp + 2).unwrap();
            writeln!(self.asm, "    ld h, (ix + {})", disp + 3).unwrap();
            writeln!(self.asm, "    ld {d2}, hl").unwrap();
        } else if let Some(&(va, is_u)) = self.const_addr_loads.get(&v) {
            self.emit_copy32(va, is_u, dst, dst_uchiguri);
        } else {
            self.emit_copy32(self.val_loc(v), true, dst, dst_uchiguri);
        }
    }

    pub(super) fn emit_store_dst_a(&mut self, dst: ValueId) {
        if let Some(&reg) = self.fa.direct_reg_dsts.get(&dst) {
            match reg {
                RegLoc::B => writeln!(self.asm, "    ld b, a").unwrap(),
                RegLoc::C => writeln!(self.asm, "    ld c, a").unwrap(),
                RegLoc::BC => unreachable!("1-byte direct_reg_dst cannot be BC"),
            }
            if matches!(self.hl_ptr, Some((HlBase::BcReg, _))) {
                self.hl_ptr = None;
            }
            self.val_in_reg
                .retain(|_, &mut vr| !vr.mask().intersects(reg.mask()));
            self.index_cse.retain(|(_, ik, _), _| match *ik {
                IndexKey::Reg(rk) => !rk.mask().intersects(reg.mask()),
                IndexKey::Mem(..) | IndexKey::Const(_) => true,
            });
        } else {
            let da = self.val_addr(dst);
            self.invalidate_hl_ram(da, 1);
            let da_mem = self.fmt_dst_mem(dst);
            writeln!(self.asm, "    ld {da_mem}, a").unwrap();
        }
    }

    pub(super) fn emit_store_dst_hl(&mut self, dst: ValueId) {
        self.hl_ptr = None;
        self.hl_val = None;
        if let Some(&RegLoc::BC) = self.fa.direct_reg_dsts.get(&dst) {
            writeln!(self.asm, "    ld b, h").unwrap();
            writeln!(self.asm, "    ld c, l").unwrap();
            self.val_in_reg
                .retain(|_, &mut vr| !RegMask::BC.intersects(vr.mask()));
            self.index_cse.retain(|(_, ik, _), _| match *ik {
                IndexKey::Reg(rk) => !rk.mask().intersects(RegMask::BC),
                IndexKey::Mem(..) | IndexKey::Const(_) => true,
            });
        } else {
            let da_mem = self.fmt_dst_mem(dst);
            writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
        }
    }

    pub(super) fn emit_store_hlde_to(&mut self, addr: u16, is_uchiguri: bool) {
        let m0 = self.fmt_addr_mem(addr, is_uchiguri);
        let m2 = self.fmt_addr_mem(addr.wrapping_add(2), is_uchiguri);
        writeln!(self.asm, "    ld {m0}, de").unwrap();
        writeln!(self.asm, "    ld {m2}, hl").unwrap();
    }

    pub(super) fn emit_const32_to(&mut self, addr: u16, is_uchiguri: bool, val: i64) {
        self.hl_ptr = None;
        self.hl_val = None;
        let u = val as u32;
        let lo = (u & 0xFFFF) as u16;
        let hi = (u >> 16) as u16;
        let m0 = self.fmt_addr_mem(addr, is_uchiguri);
        let m2 = self.fmt_addr_mem(addr.wrapping_add(2), is_uchiguri);
        writeln!(self.asm, "    ld hl, 0x{lo:04X}").unwrap();
        writeln!(self.asm, "    ld {m0}, hl").unwrap();
        if hi != lo {
            writeln!(self.asm, "    ld hl, 0x{hi:04X}").unwrap();
        }
        writeln!(self.asm, "    ld {m2}, hl").unwrap();
    }

    pub(super) fn emit_copy32(
        &mut self,
        src: u16,
        src_uchiguri: bool,
        dst: u16,
        dst_uchiguri: bool,
    ) {
        if src == dst && src_uchiguri == dst_uchiguri {
            return;
        }
        self.hl_ptr = None;
        self.hl_val = None;
        let s0 = self.fmt_addr_mem(src, src_uchiguri);
        let d0 = self.fmt_addr_mem(dst, dst_uchiguri);
        let s2 = self.fmt_addr_mem(src.wrapping_add(2), src_uchiguri);
        let d2 = self.fmt_addr_mem(dst.wrapping_add(2), dst_uchiguri);
        writeln!(self.asm, "    ld hl, {s0}").unwrap();
        writeln!(self.asm, "    ld {d0}, hl").unwrap();
        writeln!(self.asm, "    ld hl, {s2}").unwrap();
        writeln!(self.asm, "    ld {d2}, hl").unwrap();
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
    let plan = crate::overlay::partition_program(program, &[])?;
    let empty_syms = HashMap::new();
    emit_z80_asm_with_overlays(
        program,
        analysis,
        layout,
        opts,
        &plan,
        &empty_syms,
        &[],
        true,
    )
}

/// Emits Resident Z80 assembly using an `OverlayPlan`, optionally resolving overlay addresses
/// and `(dst, size)` spans on Pass 3 (`pass1_placeholders = false`).
#[allow(clippy::too_many_arguments)]
pub fn emit_z80_asm_with_overlays(
    program: &IrProgram,
    analysis: &ProgramAnalysis,
    layout: &RamLayout,
    opts: CompileOptions,
    plan: &OverlayPlan,
    resolved_overlay_syms: &HashMap<String, u16>,
    resolved_blob_spans: &[(u16, u16)],
    pass1_placeholders: bool,
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
    writeln!(asm, "__aun_ram = 0x{:04X}", layout.ram_base).unwrap();
    for g in &layout.global_entries {
        writeln!(asm, "{} = 0x{:04X}", g.name, g.addr).unwrap();
    }

    let mut runtime_needs = RuntimeNeeds::default();
    let mut needs_noop_ret = false;

    // Count non-power-of-two IndexAddr/Pool `elem_size` constants across Resident functions:
    // outline any `k >= 5` used >= 2 times into a shared `__aun_mul_hl_<k>` helper (3B call!).
    let mut mul_counts: HashMap<u16, usize> = HashMap::new();
    for f in &program.functions {
        if f.is_test || plan.is_fn_in_overlay(f.id as usize) {
            continue;
        }
        for bb in &f.blocks {
            for inst in &bb.insts {
                match inst {
                    IrInst::IndexAddr { elem_size, len, .. }
                        if *elem_size >= 5
                            && !elem_size.is_power_of_two()
                            && !runtime::is_small_u8_array_mul(*len, *elem_size) =>
                    {
                        *mul_counts.entry(*elem_size).or_default() += 1;
                    }
                    IrInst::PoolElemAddr { elem_size, .. }
                        if *elem_size >= 5 && !elem_size.is_power_of_two() =>
                    {
                        *mul_counts.entry(*elem_size).or_default() += 1;
                    }
                    _ => {}
                }
            }
        }
    }
    let shared_muls: BTreeSet<u16> = mul_counts
        .into_iter()
        .filter(|&(_, cnt)| cnt >= 2)
        .map(|(k, _)| k)
        .collect();

    let use_cmp_s16_helper = count_cold_signed_i16_cmps(program, analysis) >= 3;
    let env = EmitGlobalEnv {
        program,
        analysis,
        layout,
        opts: opts.clone(),
        shared_muls: &shared_muls,
        use_cmp_s16_helper,
        plan,
        resolved_overlay_syms,
        pass1_placeholders,
    };

    for func in &program.functions {
        if func.is_test || plan.is_fn_in_overlay(func.id as usize) || func.blocks.is_empty() {
            continue;
        }
        func::emit_function(&env, func, &mut runtime_needs, &mut asm)?;
    }

    if runtime_needs.jp_iy {
        writeln!(asm, "\n__aun_jp_iy:").unwrap();
        writeln!(asm, "    jp (iy)").unwrap();
    }
    if runtime_needs.cmp_s16 || (use_cmp_s16_helper && !plan.blobs.is_empty()) {
        runtime::emit_cmp_s16_helper(&mut asm);
    }
    if runtime_needs.mul8 {
        runtime::emit_mul8_helper(&mut asm);
    }
    if runtime_needs.mul32 {
        runtime::emit_mul32_helper(&mut asm);
    }
    if runtime_needs.mulu16_32 || runtime_needs.muls16_32 {
        runtime::emit_mulu16_32_helper(&mut asm);
    }
    if runtime_needs.muls16_32 {
        runtime::emit_muls16_32_helper(&mut asm);
    }
    if runtime_needs.udiv32_8 {
        runtime::emit_udiv32_8_helper(&mut asm);
    }
    if runtime_needs.udiv32 || runtime_needs.sdiv32 {
        runtime::emit_udiv32_helper(&mut asm);
    }
    if runtime_needs.sdiv32 {
        runtime::emit_sdiv32_helper(&mut asm);
    }
    if runtime_needs.udiv16 || runtime_needs.sdiv16 {
        runtime::emit_udiv16_helper(&mut asm);
    }
    if runtime_needs.sdiv16 {
        runtime::emit_sdiv16_helper(&mut asm);
    }
    if runtime_needs.udiv8 || runtime_needs.sdiv8 {
        runtime::emit_udiv8_helper(&mut asm);
    }
    if runtime_needs.sdiv8 {
        runtime::emit_sdiv8_helper(&mut asm);
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
                plan,
                resolved_overlay_syms,
                pass1_placeholders,
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
                plan,
                resolved_overlay_syms,
                pass1_placeholders,
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

    // Ensure Resident emits swap trampolines referenced by ANY function (including overlays)
    for func in &program.functions {
        if func.is_test || func.blocks.is_empty() {
            continue;
        }
        for bb in &func.blocks {
            for inst in &bb.insts {
                if let IrInst::StateTransition {
                    machine_id,
                    from_state,
                    to_state,
                    field_vals,
                    ..
                } = inst
                    && plan.machine_has_boundaries[*machine_id as usize]
                {
                    let mut cs = u16::MAX;
                    let mut ce = 0u16;
                    for (rel_off, _, fty) in field_vals {
                        let fsz = program.type_size(fty);
                        if fsz > 0 {
                            cs = cs.min(*rel_off);
                            ce = ce.max(rel_off.saturating_add(fsz));
                        }
                    }
                    if ce <= cs {
                        cs = 0;
                        ce = 0;
                    }
                    runtime_needs.trampolines.insert(TrampolineKey {
                        machine_id: *machine_id,
                        from_state: *from_state,
                        to_state: *to_state,
                        copy_start: cs,
                        copy_end: ce,
                    });
                }
            }
        }
    }

    if !runtime_needs.trampolines.is_empty() {
        runtime::emit_swap_tables_and_trampolines(
            program,
            layout,
            &opts,
            plan,
            resolved_overlay_syms,
            resolved_blob_spans,
            pass1_placeholders,
            &runtime_needs.trampolines,
            &mut runtime_needs.panic_stub,
            &mut asm,
        );
    }

    // Emit ROM constant tables (`const TABLE: [T; N] = [...]`) owned by Resident code
    for (gid, g) in program.globals.iter().enumerate() {
        if !g.in_rom
            || g.is_extern
            || plan.rom_global_owner.get(gid) != Some(&OverlayOwner::Resident)
        {
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
    if runtime_needs.panic_stub && !user_defined_panic {
        writeln!(asm, "\n__aun_panic:").unwrap();
        writeln!(asm, "    di").unwrap();
        writeln!(asm, "    halt").unwrap();
    }

    Ok(asm)
}

/// Emits Aun-generated Z80 assembly for a single HSM/overlay blob (`LoadOpSpec`) linked
/// at `origin` against `visible_syms` (`resident_symbols ∪ ancestor_blob_symbols`).
/// External `.s` blocks owned by the blob should be appended via `append_overlay_external_asms`
/// after running peephole and LIR optimization passes on the Aun-generated assembly.
#[allow(clippy::too_many_arguments)]
pub fn emit_overlay_blob_z80_asm(
    program: &IrProgram,
    analysis: &ProgramAnalysis,
    layout: &RamLayout,
    opts: CompileOptions,
    plan: &OverlayPlan,
    blob: &LoadOpSpec,
    origin: u16,
    visible_syms: &HashMap<String, u16>,
) -> Result<String, String> {
    let mut asm = String::new();
    writeln!(
        asm,
        "; Overlay blob #{} ({}) in space `{}` linked at 0x{:04X}",
        blob.blob_id, blob.state_name, blob.space_name, origin
    )
    .unwrap();
    writeln!(asm, ".org 0x{:04X}", origin).unwrap();

    writeln!(asm, "__aun_ram = 0x{:04X}", layout.ram_base).unwrap();
    let global_names: HashSet<&str> = layout
        .global_entries
        .iter()
        .map(|g| g.name.as_str())
        .collect();
    for g in &layout.global_entries {
        writeln!(asm, "{} = 0x{:04X}", g.name, g.addr).unwrap();
    }

    // Collect local definitions belonging to this blob so we never duplicate them as `.equ`
    let mut blob_local_defs: HashSet<String> = HashSet::new();
    for &fid in &blob.fn_ids {
        blob_local_defs.insert(z80_sym(&program.functions[fid].name));
    }
    for &gid in &blob.rom_global_ids {
        blob_local_defs.insert(program.globals[gid].name.clone());
    }

    let mut sorted_syms: Vec<(&String, &u16)> = visible_syms
        .iter()
        .filter(|(k, _)| {
            k.as_str() != "__aun_ram"
                && !global_names.contains(k.as_str())
                && !blob_local_defs.contains(k.as_str())
        })
        .collect();
    sorted_syms.sort_by_key(|(k, _)| k.as_str());
    for (name, addr) in sorted_syms {
        writeln!(asm, "{name} = 0x{addr:04X}").unwrap();
    }

    let mut runtime_needs = RuntimeNeeds::default();
    let empty_muls = BTreeSet::new();
    let empty_resolved: HashMap<String, u16> = HashMap::new();
    let use_cmp_s16_helper = count_cold_signed_i16_cmps(program, analysis) >= 3;
    let env = EmitGlobalEnv {
        program,
        analysis,
        layout,
        opts: opts.clone(),
        shared_muls: &empty_muls,
        use_cmp_s16_helper,
        plan,
        resolved_overlay_syms: &empty_resolved,
        pass1_placeholders: false,
    };
    for &fid in &blob.fn_ids {
        let func = &program.functions[fid];
        if !func.is_test && !func.blocks.is_empty() {
            func::emit_function(&env, func, &mut runtime_needs, &mut asm)?;
        }
    }

    for &gid in &blob.rom_global_ids {
        let g = &program.globals[gid];
        if let Some(alias) = crate::global_export_alias(g, opts.symbol_prefix.as_deref()) {
            writeln!(asm, "\n{alias}:").unwrap();
            writeln!(asm, "{}:", g.name).unwrap();
        } else {
            writeln!(asm, "\n{}:", g.name).unwrap();
        }
        for chunk in g.init.chunks(16) {
            let bytes: Vec<String> = chunk.iter().map(|b| format!("0x{b:02X}")).collect();
            writeln!(asm, "    .db {}", bytes.join(", ")).unwrap();
        }
    }

    if runtime_needs.jp_iy && !visible_syms.contains_key("__aun_jp_iy") {
        writeln!(asm, "\n__aun_jp_iy:").unwrap();
        writeln!(asm, "    jp (iy)").unwrap();
    }
    if runtime_needs.cmp_s16 && !visible_syms.contains_key("__aun_cmp_s16") {
        runtime::emit_cmp_s16_helper(&mut asm);
    }
    if runtime_needs.mul8 && !visible_syms.contains_key("__aun_mul8") {
        runtime::emit_mul8_helper(&mut asm);
    }
    if runtime_needs.mul32 && !visible_syms.contains_key("__aun_mul32") {
        runtime::emit_mul32_helper(&mut asm);
    }
    if (runtime_needs.mulu16_32 || runtime_needs.muls16_32)
        && !visible_syms.contains_key("__aun_mulu16_32")
    {
        runtime::emit_mulu16_32_helper(&mut asm);
    }
    if runtime_needs.muls16_32 && !visible_syms.contains_key("__aun_muls16_32") {
        runtime::emit_muls16_32_helper(&mut asm);
    }
    if runtime_needs.udiv32_8 && !visible_syms.contains_key("__aun_udivrem32_8") {
        runtime::emit_udiv32_8_helper(&mut asm);
    }
    if (runtime_needs.udiv32 || runtime_needs.sdiv32)
        && !visible_syms.contains_key("__aun_udiv32")
    {
        runtime::emit_udiv32_helper(&mut asm);
    }
    if runtime_needs.sdiv32 && !visible_syms.contains_key("__aun_sdiv32") {
        runtime::emit_sdiv32_helper(&mut asm);
    }
    if (runtime_needs.udiv16 || runtime_needs.sdiv16)
        && !visible_syms.contains_key("__aun_udiv16")
    {
        runtime::emit_udiv16_helper(&mut asm);
    }
    if runtime_needs.sdiv16 && !visible_syms.contains_key("__aun_sdiv16") {
        runtime::emit_sdiv16_helper(&mut asm);
    }
    if (runtime_needs.udiv8 || runtime_needs.sdiv8) && !visible_syms.contains_key("__aun_udiv8") {
        runtime::emit_udiv8_helper(&mut asm);
    }
    if runtime_needs.sdiv8 && !visible_syms.contains_key("__aun_sdiv8") {
        runtime::emit_sdiv8_helper(&mut asm);
    }
    if runtime_needs.panic_stub && !visible_syms.contains_key("__aun_panic") {
        writeln!(asm, "\n__aun_panic:").unwrap();
        writeln!(asm, "    di").unwrap();
        writeln!(asm, "    halt").unwrap();
    }

    Ok(asm)
}

/// Appends external `.s` modules owned by `blob` after peephole/LIR passes have finished
/// optimizing the Aun-generated portion of the overlay assembly.
pub fn append_overlay_external_asms(
    asm: &mut String,
    blob: &LoadOpSpec,
    external_asms: &[ExternalAsm],
) {
    for &ext_idx in &blob.ext_asm_indices {
        if let Some(ext) = external_asms.get(ext_idx) {
            writeln!(
                asm,
                "\n        ; external asm `{}`\n        .area _CODE\n; @aun:ext_begin {}",
                ext.name, ext_idx
            )
            .unwrap();
            asm.push_str(&ext.source);
            if !asm.ends_with('\n') {
                asm.push('\n');
            }
            writeln!(asm, "; @aun:ext_end").unwrap();
        }
    }
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
    if !resident_symbols.contains_key("__aun_ram") {
        writeln!(asm, "__aun_ram = 0x{:04X}", layout.ram_base).unwrap();
    }
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

    let mut runtime_needs = RuntimeNeeds::default();
    let empty_muls = BTreeSet::new();
    let plan = crate::overlay::partition_program(program, &[])?;
    let empty_resolved: HashMap<String, u16> = HashMap::new();
    let use_cmp_s16_helper = count_cold_signed_i16_cmps(program, analysis) >= 3;
    let env = EmitGlobalEnv {
        program,
        analysis,
        layout,
        opts: opts.clone(),
        shared_muls: &empty_muls,
        use_cmp_s16_helper,
        plan: &plan,
        resolved_overlay_syms: &empty_resolved,
        pass1_placeholders: false,
    };
    for func in ovl_fns {
        func::emit_function(&env, func, &mut runtime_needs, &mut asm)?;
    }
    if runtime_needs.jp_iy && !resident_symbols.contains_key("__aun_jp_iy") {
        writeln!(asm, "\n__aun_jp_iy:").unwrap();
        writeln!(asm, "    jp (iy)").unwrap();
    }
    if runtime_needs.cmp_s16 && !resident_symbols.contains_key("__aun_cmp_s16") {
        runtime::emit_cmp_s16_helper(&mut asm);
    }
    if runtime_needs.mul8 && !resident_symbols.contains_key("__aun_mul8") {
        runtime::emit_mul8_helper(&mut asm);
    }
    if runtime_needs.mul32 && !resident_symbols.contains_key("__aun_mul32") {
        runtime::emit_mul32_helper(&mut asm);
    }
    if (runtime_needs.mulu16_32 || runtime_needs.muls16_32)
        && !resident_symbols.contains_key("__aun_mulu16_32")
    {
        runtime::emit_mulu16_32_helper(&mut asm);
    }
    if runtime_needs.muls16_32 && !resident_symbols.contains_key("__aun_muls16_32") {
        runtime::emit_muls16_32_helper(&mut asm);
    }
    if runtime_needs.udiv32_8 && !resident_symbols.contains_key("__aun_udivrem32_8") {
        runtime::emit_udiv32_8_helper(&mut asm);
    }
    if (runtime_needs.udiv32 || runtime_needs.sdiv32)
        && !resident_symbols.contains_key("__aun_udiv32")
    {
        runtime::emit_udiv32_helper(&mut asm);
    }
    if runtime_needs.sdiv32 && !resident_symbols.contains_key("__aun_sdiv32") {
        runtime::emit_sdiv32_helper(&mut asm);
    }
    if (runtime_needs.udiv16 || runtime_needs.sdiv16)
        && !resident_symbols.contains_key("__aun_udiv16")
    {
        runtime::emit_udiv16_helper(&mut asm);
    }
    if runtime_needs.sdiv16 && !resident_symbols.contains_key("__aun_sdiv16") {
        runtime::emit_sdiv16_helper(&mut asm);
    }
    if (runtime_needs.udiv8 || runtime_needs.sdiv8)
        && !resident_symbols.contains_key("__aun_udiv8")
    {
        runtime::emit_udiv8_helper(&mut asm);
    }
    if runtime_needs.sdiv8 && !resident_symbols.contains_key("__aun_sdiv8") {
        runtime::emit_sdiv8_helper(&mut asm);
    }

    Ok(asm)
}
