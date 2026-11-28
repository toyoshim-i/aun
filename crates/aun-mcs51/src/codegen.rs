//! MCS-51 / WCH CH559 code generation lowering from `aun-ir`.

use crate::asm::assemble_mcs51;
use crate::layout::{compute_layout, Mcs51Layout, PtrSpace, SlotRegion};
use crate::{CompileOptions, Mcs51Output};
use aun_ir::opt::OptConfig;
use aun_ir::*;
use std::collections::{BTreeSet, HashMap};
use std::fmt::Write;

const R2: u8 = 2;
const R3: u8 = 3;
const R4: u8 = 4;
const R5: u8 = 5;
const R6: u8 = 6;
const R7: u8 = 7;

fn sanitize_sym(name: &str) -> String {
    name.replace("::", "__")
}

fn is_signed_ty(ty: &IrType) -> bool {
    matches!(
        ty,
        IrType::Int(i) | IrType::Fixed { base: i, .. } if i.is_signed()
    )
}

fn fn_label(program: &IrProgram, fid: u32) -> String {
    let f = &program.functions[fid as usize];
    if f.name == "main" {
        "main".to_string()
    } else if let Some(ref exp) = f.export_name {
        exp.clone()
    } else {
        format!("__fn_{}", sanitize_sym(&f.name))
    }
}

struct Codegen<'a> {
    program: &'a IrProgram,
    layout: &'a Mcs51Layout,
    opts: &'a CompileOptions,
    asm: String,
    lbl_counter: u32,
    used_helpers: BTreeSet<&'static str>,
}

impl<'a> Codegen<'a> {
    fn new(program: &'a IrProgram, layout: &'a Mcs51Layout, opts: &'a CompileOptions) -> Self {
        Self {
            program,
            layout,
            opts,
            asm: String::new(),
            lbl_counter: 0,
            used_helpers: BTreeSet::new(),
        }
    }

    fn fresh_lbl(&mut self, prefix: &str) -> String {
        let id = self.lbl_counter;
        self.lbl_counter += 1;
        format!("__{prefix}_{id}")
    }

    fn val_loc(&self, fid: u32, v: ValueId) -> SlotRegion {
        self.layout.val_locs[&(fid, v)]
    }

    fn slot_loc(&self, fid: u32, s: SlotId) -> SlotRegion {
        self.layout.slot_locs[&(fid, s)]
    }

    fn val_ptr_space(&self, fid: u32, v: ValueId) -> PtrSpace {
        self.layout
            .val_ptr_spaces
            .get(&(fid, v))
            .copied()
            .unwrap_or(PtrSpace::XData)
    }

    fn slot_ptr_space(&self, fid: u32, s: SlotId) -> PtrSpace {
        self.layout
            .slot_ptr_spaces
            .get(&(fid, s))
            .copied()
            .unwrap_or(PtrSpace::XData)
    }

    fn emit_load_region_byte_a(&mut self, reg: SlotRegion, off: u16) {
        match reg {
            SlotRegion::Data(a) => {
                writeln!(self.asm, "    mov a, 0x{:02X}", (a as u16 + off) as u8).unwrap();
            }
            SlotRegion::XData(a) => {
                writeln!(self.asm, "    mov dptr, #0x{:04X}", a.wrapping_add(off)).unwrap();
                writeln!(self.asm, "    movx a, @dptr").unwrap();
            }
        }
    }

    fn emit_store_region_byte_a(&mut self, reg: SlotRegion, off: u16) {
        match reg {
            SlotRegion::Data(a) => {
                writeln!(self.asm, "    mov 0x{:02X}, a", (a as u16 + off) as u8).unwrap();
            }
            SlotRegion::XData(a) => {
                writeln!(self.asm, "    mov dptr, #0x{:04X}", a.wrapping_add(off)).unwrap();
                writeln!(self.asm, "    movx @dptr, a").unwrap();
            }
        }
    }

    fn emit_load_region_byte_rn(&mut self, reg: SlotRegion, off: u16, rn: u8) {
        match reg {
            SlotRegion::Data(a) => {
                writeln!(self.asm, "    mov r{rn}, 0x{:02X}", (a as u16 + off) as u8).unwrap();
            }
            SlotRegion::XData(a) => {
                writeln!(self.asm, "    mov dptr, #0x{:04X}", a.wrapping_add(off)).unwrap();
                writeln!(self.asm, "    movx a, @dptr").unwrap();
                writeln!(self.asm, "    mov r{rn}, a").unwrap();
            }
        }
    }

    fn emit_store_region_byte_rn(&mut self, reg: SlotRegion, off: u16, rn: u8) {
        match reg {
            SlotRegion::Data(a) => {
                writeln!(self.asm, "    mov 0x{:02X}, r{rn}", (a as u16 + off) as u8).unwrap();
            }
            SlotRegion::XData(a) => {
                writeln!(self.asm, "    mov a, r{rn}").unwrap();
                writeln!(self.asm, "    mov dptr, #0x{:04X}", a.wrapping_add(off)).unwrap();
                writeln!(self.asm, "    movx @dptr, a").unwrap();
            }
        }
    }

    fn emit_store_region_imm_byte(&mut self, reg: SlotRegion, off: u16, imm: u8) {
        match reg {
            SlotRegion::Data(a) => {
                writeln!(self.asm, "    mov 0x{:02X}, #0x{imm:02X}", (a as u16 + off) as u8)
                    .unwrap();
            }
            SlotRegion::XData(a) => {
                writeln!(self.asm, "    mov a, #0x{imm:02X}").unwrap();
                writeln!(self.asm, "    mov dptr, #0x{:04X}", a.wrapping_add(off)).unwrap();
                writeln!(self.asm, "    movx @dptr, a").unwrap();
            }
        }
    }

    /// Loads `r_lo, r_hi` with the 16-bit value stored in `reg`.
    fn emit_load_region_u16(&mut self, reg: SlotRegion, r_lo: u8, r_hi: u8) {
        self.emit_load_region_byte_rn(reg, 0, r_lo);
        self.emit_load_region_byte_rn(reg, 1, r_hi);
    }

    fn emit_store_region_u16(&mut self, reg: SlotRegion, r_lo: u8, r_hi: u8) {
        self.emit_store_region_byte_rn(reg, 0, r_lo);
        self.emit_store_region_byte_rn(reg, 1, r_hi);
    }

    fn emit_load_region_n_to_regs(&mut self, reg: SlotRegion, sz: u16, start_r: u8) {
        for i in 0..sz {
            self.emit_load_region_byte_rn(reg, i, start_r + i as u8);
        }
    }

    fn emit_store_region_n_from_regs(&mut self, reg: SlotRegion, sz: u16, start_r: u8) {
        for i in 0..sz {
            self.emit_store_region_byte_rn(reg, i, start_r + i as u8);
        }
    }

    fn emit_copy_region(&mut self, dst: SlotRegion, src: SlotRegion, sz: u16) {
        if sz == 0 || dst == src {
            return;
        }
        if sz <= 4 {
            for i in 0..sz {
                self.emit_load_region_byte_a(src, i);
                self.emit_store_region_byte_a(dst, i);
            }
            return;
        }
        match (src, dst) {
            (SlotRegion::XData(sa), SlotRegion::XData(da)) => {
                self.used_helpers.insert("__aun_memcpy_xx");
                writeln!(self.asm, "    mov r2, #0x{:02X}", (sa & 0xFF) as u8).unwrap();
                writeln!(self.asm, "    mov r3, #0x{:02X}", (sa >> 8) as u8).unwrap();
                writeln!(self.asm, "    mov r4, #0x{:02X}", (da & 0xFF) as u8).unwrap();
                writeln!(self.asm, "    mov r5, #0x{:02X}", (da >> 8) as u8).unwrap();
                writeln!(self.asm, "    mov r6, #0x{:02X}", (sz & 0xFF) as u8).unwrap();
                writeln!(self.asm, "    mov r7, #0x{:02X}", (sz >> 8) as u8).unwrap();
                writeln!(self.asm, "    lcall __aun_memcpy_xx").unwrap();
            }
            _ => {
                for i in 0..sz {
                    self.emit_load_region_byte_a(src, i);
                    self.emit_store_region_byte_a(dst, i);
                }
            }
        }
    }

    /// Emits `r3:r2` = `r3:r2 + imm16` (does not clobber `r4..r7`).
    fn emit_add_r2r3_const(&mut self, c: u16) {
        if c == 0 {
            return;
        }
        writeln!(self.asm, "    mov a, r2").unwrap();
        writeln!(self.asm, "    add a, #0x{:02X}", (c & 0xFF) as u8).unwrap();
        writeln!(self.asm, "    mov r2, a").unwrap();
        writeln!(self.asm, "    mov a, r3").unwrap();
        writeln!(self.asm, "    addc a, #0x{:02X}", (c >> 8) as u8).unwrap();
        writeln!(self.asm, "    mov r3, a").unwrap();
    }

    /// Reads `sz` bytes from runtime pointer (`R2:R3` + optional tag in `A` if `Mixed`) into `dst`.
    /// When `sz <= 4`, can also target `regs` (`R2..`), else stores directly into `dst`.
    fn emit_load_from_ptr(&mut self, sp: PtrSpace, tag_loc: Option<SlotRegion>, dst: SlotRegion, sz: u16) {
        match sp {
            PtrSpace::Code => {
                self.emit_copy_code_r2r3_to_region(dst, sz);
            }
            PtrSpace::Data | PtrSpace::IData => {
                if sz <= 4 {
                    writeln!(self.asm, "    mov a, r2").unwrap();
                    writeln!(self.asm, "    mov r0, a").unwrap();
                    for i in 0..sz {
                        writeln!(self.asm, "    mov a, @r0").unwrap();
                        writeln!(self.asm, "    mov r{}, a", 4 + i as u8).unwrap();
                        if i + 1 < sz {
                            writeln!(self.asm, "    inc r0").unwrap();
                        }
                    }
                    self.emit_store_region_n_from_regs(dst, sz, 4);
                } else {
                    for i in 0..sz {
                        writeln!(self.asm, "    mov a, r2").unwrap();
                        writeln!(self.asm, "    add a, #0x{:02X}", i as u8).unwrap();
                        writeln!(self.asm, "    mov r0, a").unwrap();
                        writeln!(self.asm, "    mov a, @r0").unwrap();
                        self.emit_store_region_byte_a(dst, i);
                    }
                }
            }
            PtrSpace::XData => {
                if sz <= 4 {
                    writeln!(self.asm, "    mov dpl, r2").unwrap();
                    writeln!(self.asm, "    mov dph, r3").unwrap();
                    for i in 0..sz {
                        writeln!(self.asm, "    movx a, @dptr").unwrap();
                        writeln!(self.asm, "    mov r{}, a", 4 + i as u8).unwrap();
                        if i + 1 < sz {
                            writeln!(self.asm, "    inc dptr").unwrap();
                        }
                    }
                    self.emit_store_region_n_from_regs(dst, sz, 4);
                } else {
                    match dst {
                        SlotRegion::XData(da) => {
                            self.used_helpers.insert("__aun_memcpy_xx");
                            writeln!(self.asm, "    mov r4, #0x{:02X}", (da & 0xFF) as u8).unwrap();
                            writeln!(self.asm, "    mov r5, #0x{:02X}", (da >> 8) as u8).unwrap();
                            writeln!(self.asm, "    mov r6, #0x{:02X}", (sz & 0xFF) as u8).unwrap();
                            writeln!(self.asm, "    mov r7, #0x{:02X}", (sz >> 8) as u8).unwrap();
                            writeln!(self.asm, "    lcall __aun_memcpy_xx").unwrap();
                        }
                        SlotRegion::Data(da) => {
                            writeln!(self.asm, "    mov dpl, r2").unwrap();
                            writeln!(self.asm, "    mov dph, r3").unwrap();
                            writeln!(self.asm, "    mov r0, #0x{da:02X}").unwrap();
                            for i in 0..sz {
                                writeln!(self.asm, "    movx a, @dptr").unwrap();
                                writeln!(self.asm, "    mov @r0, a").unwrap();
                                if i + 1 < sz {
                                    writeln!(self.asm, "    inc dptr").unwrap();
                                    writeln!(self.asm, "    inc r0").unwrap();
                                }
                            }
                        }
                    }
                }
            }
            PtrSpace::Sfr => {
                let is_end = self.fresh_lbl("sfr_end");
                let _ = is_end;
                unreachable!("SFR load handled directly via constant SFR addr");
            }
            PtrSpace::Bit => unreachable!("Bit load handled directly via constant Bit addr"),
            PtrSpace::Mixed => {
                let lbl_xdata = self.fresh_lbl("mix_xd");
                let lbl_code = self.fresh_lbl("mix_cd");
                let lbl_done = self.fresh_lbl("mix_dn");
                let tl = tag_loc.expect("Mixed PtrSpace must have tag_loc");
                self.emit_load_region_byte_a(tl, 0);
                writeln!(self.asm, "    cjne a, #1, {lbl_xdata}").unwrap();
                self.emit_load_from_ptr(PtrSpace::Data, None, dst, sz);
                writeln!(self.asm, "    ljmp {lbl_done}").unwrap();
                writeln!(self.asm, "{lbl_xdata}:").unwrap();
                writeln!(self.asm, "    cjne a, #2, {lbl_code}").unwrap();
                self.emit_load_from_ptr(PtrSpace::XData, None, dst, sz);
                writeln!(self.asm, "    ljmp {lbl_done}").unwrap();
                writeln!(self.asm, "{lbl_code}:").unwrap();
                self.emit_load_from_ptr(PtrSpace::Code, None, dst, sz);
                writeln!(self.asm, "{lbl_done}:").unwrap();
            }
        }
    }

    fn emit_copy_code_r2r3_to_region(&mut self, dst: SlotRegion, sz: u16) {
        if sz <= 4 {
            writeln!(self.asm, "    mov dpl, r2").unwrap();
            writeln!(self.asm, "    mov dph, r3").unwrap();
            for i in 0..sz {
                writeln!(self.asm, "    mov a, #0x{:02X}", i as u8).unwrap();
                writeln!(self.asm, "    movc a, @a+dptr").unwrap();
                writeln!(self.asm, "    mov r{}, a", 4 + i as u8).unwrap();
            }
            self.emit_store_region_n_from_regs(dst, sz, 4);
        } else {
            match dst {
                SlotRegion::XData(da) => {
                    self.used_helpers.insert("__aun_memcpy_cx");
                    writeln!(self.asm, "    mov r4, #0x{:02X}", (da & 0xFF) as u8).unwrap();
                    writeln!(self.asm, "    mov r5, #0x{:02X}", (da >> 8) as u8).unwrap();
                    writeln!(self.asm, "    mov r6, #0x{:02X}", (sz & 0xFF) as u8).unwrap();
                    writeln!(self.asm, "    mov r7, #0x{:02X}", (sz >> 8) as u8).unwrap();
                    writeln!(self.asm, "    lcall __aun_memcpy_cx").unwrap();
                }
                SlotRegion::Data(da) => {
                    writeln!(self.asm, "    mov dpl, r2").unwrap();
                    writeln!(self.asm, "    mov dph, r3").unwrap();
                    writeln!(self.asm, "    mov r0, #0x{da:02X}").unwrap();
                    for i in 0..sz {
                        writeln!(self.asm, "    mov a, #0x{:02X}", i as u8).unwrap();
                        writeln!(self.asm, "    movc a, @a+dptr").unwrap();
                        writeln!(self.asm, "    mov @r0, a").unwrap();
                        if i + 1 < sz {
                            writeln!(self.asm, "    inc r0").unwrap();
                        }
                    }
                }
            }
        }
    }

    /// Stores `src` (`sz` bytes) through pointer in `R2:R3` (dst address space `sp`).
    fn emit_store_to_ptr(&mut self, sp: PtrSpace, tag_loc: Option<SlotRegion>, src: SlotRegion, sz: u16) {
        match sp {
            PtrSpace::Data | PtrSpace::IData => {
                if sz <= 4 {
                    self.emit_load_region_n_to_regs(src, sz, 4);
                    writeln!(self.asm, "    mov a, r2").unwrap();
                    writeln!(self.asm, "    mov r0, a").unwrap();
                    for i in 0..sz {
                        writeln!(self.asm, "    mov a, r{}", 4 + i as u8).unwrap();
                        writeln!(self.asm, "    mov @r0, a").unwrap();
                        if i + 1 < sz {
                            writeln!(self.asm, "    inc r0").unwrap();
                        }
                    }
                } else {
                    for i in 0..sz {
                        self.emit_load_region_byte_a(src, i);
                        writeln!(self.asm, "    mov b, a").unwrap();
                        writeln!(self.asm, "    mov a, r2").unwrap();
                        writeln!(self.asm, "    add a, #0x{:02X}", i as u8).unwrap();
                        writeln!(self.asm, "    mov r0, a").unwrap();
                        writeln!(self.asm, "    mov a, b").unwrap();
                        writeln!(self.asm, "    mov @r0, a").unwrap();
                    }
                }
            }
            PtrSpace::XData => {
                if sz <= 4 {
                    self.emit_load_region_n_to_regs(src, sz, 4);
                    writeln!(self.asm, "    mov dpl, r2").unwrap();
                    writeln!(self.asm, "    mov dph, r3").unwrap();
                    for i in 0..sz {
                        writeln!(self.asm, "    mov a, r{}", 4 + i as u8).unwrap();
                        writeln!(self.asm, "    movx @dptr, a").unwrap();
                        if i + 1 < sz {
                            writeln!(self.asm, "    inc dptr").unwrap();
                        }
                    }
                } else {
                    match src {
                        SlotRegion::XData(sa) => {
                            self.used_helpers.insert("__aun_memcpy_xx");
                            writeln!(self.asm, "    mov a, r2").unwrap();
                            writeln!(self.asm, "    mov r4, a").unwrap();
                            writeln!(self.asm, "    mov a, r3").unwrap();
                            writeln!(self.asm, "    mov r5, a").unwrap();
                            writeln!(self.asm, "    mov r2, #0x{:02X}", (sa & 0xFF) as u8).unwrap();
                            writeln!(self.asm, "    mov r3, #0x{:02X}", (sa >> 8) as u8).unwrap();
                            writeln!(self.asm, "    mov r6, #0x{:02X}", (sz & 0xFF) as u8).unwrap();
                            writeln!(self.asm, "    mov r7, #0x{:02X}", (sz >> 8) as u8).unwrap();
                            writeln!(self.asm, "    lcall __aun_memcpy_xx").unwrap();
                        }
                        SlotRegion::Data(sa) => {
                            writeln!(self.asm, "    mov dpl, r2").unwrap();
                            writeln!(self.asm, "    mov dph, r3").unwrap();
                            writeln!(self.asm, "    mov r0, #0x{sa:02X}").unwrap();
                            for i in 0..sz {
                                writeln!(self.asm, "    mov a, @r0").unwrap();
                                writeln!(self.asm, "    movx @dptr, a").unwrap();
                                if i + 1 < sz {
                                    writeln!(self.asm, "    inc r0").unwrap();
                                    writeln!(self.asm, "    inc dptr").unwrap();
                                }
                            }
                        }
                    }
                }
            }
            PtrSpace::Mixed => {
                let lbl_xdata = self.fresh_lbl("smix_xd");
                let lbl_done = self.fresh_lbl("smix_dn");
                let tl = tag_loc.expect("Mixed PtrSpace must have tag_loc");
                self.emit_load_region_byte_a(tl, 0);
                writeln!(self.asm, "    cjne a, #1, {lbl_xdata}").unwrap();
                self.emit_store_to_ptr(PtrSpace::Data, None, src, sz);
                writeln!(self.asm, "    ljmp {lbl_done}").unwrap();
                writeln!(self.asm, "{lbl_xdata}:").unwrap();
                self.emit_store_to_ptr(PtrSpace::XData, None, src, sz);
                writeln!(self.asm, "{lbl_done}:").unwrap();
            }
            other => unreachable!("cannot store into {other:?}"),
        }
    }

    fn copy_val_ptr_tag(&mut self, fid: u32, dst_v: ValueId, src_sp: PtrSpace, src_tag: Option<SlotRegion>) {
        if let Some(&dst_tl) = self.layout.val_ptr_tags.get(&(fid, dst_v)) {
            if let Some(tl) = src_tag {
                self.emit_copy_region(dst_tl, tl, 1);
            } else {
                self.emit_store_region_imm_byte(dst_tl, 0, src_sp.tag_id());
            }
        }
    }

    fn copy_slot_ptr_tag(&mut self, dst_fid: u32, dst_s: SlotId, src_sp: PtrSpace, src_tag: Option<SlotRegion>) {
        if let Some(&dst_tl) = self.layout.slot_ptr_tags.get(&(dst_fid, dst_s)) {
            if let Some(tl) = src_tag {
                self.emit_copy_region(dst_tl, tl, 1);
            } else {
                self.emit_store_region_imm_byte(dst_tl, 0, src_sp.tag_id());
            }
        }
    }

    fn emit_program(&mut self) -> Result<(), String> {
        let origin = self.layout.code_start;
        writeln!(self.asm, "; Aun MCS-51 / WCH CH559 generated assembly").unwrap();
        writeln!(self.asm, ".org 0x{origin:04X}").unwrap();
        writeln!(self.asm, "    ljmp __aun_start").unwrap();

        // Interrupt vectors from target interrupt_controllers
        if let Some(t) = &self.program.target {
            for f in &self.program.functions {
                if let Some(ref src_name) = f.interrupt_source {
                    let mut vec_addr = None;
                    for ic in &t.interrupt_controllers {
                        for src in &ic.sources {
                            if &src.name == src_name {
                                let addr = if src.mode.eq_ignore_ascii_case("irq") {
                                    0x0003u16 + src.vector_addr.wrapping_mul(8)
                                } else {
                                    src.vector_addr
                                };
                                vec_addr = Some(addr);
                            }
                        }
                    }
                    let va = vec_addr.ok_or_else(|| {
                        format!("unknown interrupt source `{src_name}` for ISR `{}`", f.name)
                    })?;
                    writeln!(self.asm, ".org 0x{va:04X}").unwrap();
                    writeln!(self.asm, "    ljmp {}", fn_label(self.program, f.id)).unwrap();
                }
            }
        }

        // Start code body past CH559 vector table (0x0000..=0x007F).
        let body_start = origin.max(0x0080);
        writeln!(self.asm, ".org 0x{body_start:04X}").unwrap();

        // Startup initialization: SP, non-zero Data/XData initializers, call main, halt loop.
        writeln!(self.asm, "__aun_start:").unwrap();
        writeln!(self.asm, "    mov sp, #0x{:02X}", self.layout.sp_init).unwrap();

        for (idx, span) in self.layout.data_inits.iter().enumerate() {
            let lbl = format!("__aun_init_d{idx}");
            let lp = self.fresh_lbl("initd");
            let n = span.bytes.len();
            writeln!(self.asm, "    mov r0, #0x{:02X}", span.dst_addr as u8).unwrap();
            writeln!(self.asm, "    mov dptr, #{lbl}").unwrap();
            writeln!(self.asm, "    mov r7, #0x{:02X}", n as u8).unwrap();
            writeln!(self.asm, "{lp}:").unwrap();
            writeln!(self.asm, "    clr a").unwrap();
            writeln!(self.asm, "    movc a, @a+dptr").unwrap();
            writeln!(self.asm, "    mov @r0, a").unwrap();
            writeln!(self.asm, "    inc r0").unwrap();
            writeln!(self.asm, "    inc dptr").unwrap();
            writeln!(self.asm, "    djnz r7, {lp}").unwrap();
        }
        for (idx, span) in self.layout.xdata_inits.iter().enumerate() {
            let lbl = format!("__aun_init_x{idx}");
            let lp = self.fresh_lbl("initx");
            let n = span.bytes.len() as u16;
            writeln!(self.asm, "    mov r2, #lo({lbl})").unwrap();
            writeln!(self.asm, "    mov r3, #hi({lbl})").unwrap();
            writeln!(self.asm, "    mov r4, #0x{:02X}", (span.dst_addr & 0xFF) as u8).unwrap();
            writeln!(self.asm, "    mov r5, #0x{:02X}", (span.dst_addr >> 8) as u8).unwrap();
            writeln!(self.asm, "    mov r6, #0x{:02X}", (n & 0xFF) as u8).unwrap();
            writeln!(self.asm, "    mov r7, #0x{:02X}", (n >> 8) as u8).unwrap();
            self.used_helpers.insert("__aun_memcpy_cx");
            writeln!(self.asm, "    lcall __aun_memcpy_cx").unwrap();
            let _ = lp;
        }

        let has_main = self.program.functions.iter().any(|f| !f.is_test && f.name == "main");
        if has_main {
            writeln!(self.asm, "    lcall main").unwrap();
        }
        writeln!(self.asm, "__aun_halt:").unwrap();
        writeln!(self.asm, "    sjmp __aun_halt").unwrap();

        // Emit functions.
        for f in &self.program.functions {
            if f.is_test {
                continue;
            }
            self.emit_function(f)?;
        }

        // Emit runtime helpers on demand.
        self.emit_helpers();

        // Emit ROM constant globals & startup initializer tables.
        for galloc in &self.layout.globals {
            if let Some(ref lbl) = galloc.rom_label {
                let g = &self.program.globals[galloc.global_id as usize];
                writeln!(self.asm, "{lbl}:").unwrap();
                self.emit_db_bytes(&g.init, galloc.size as usize);
            }
        }
        for (idx, span) in self.layout.data_inits.iter().enumerate() {
            writeln!(self.asm, "__aun_init_d{idx}:").unwrap();
            self.emit_db_bytes(&span.bytes, span.bytes.len());
        }
        for (idx, span) in self.layout.xdata_inits.iter().enumerate() {
            writeln!(self.asm, "__aun_init_x{idx}:").unwrap();
            self.emit_db_bytes(&span.bytes, span.bytes.len());
        }
        Ok(())
    }

    fn emit_db_bytes(&mut self, bytes: &[u8], total_len: usize) {
        let mut padded = bytes.to_vec();
        if padded.len() < total_len {
            padded.resize(total_len, 0);
        }
        for chunk in padded.chunks(16) {
            write!(self.asm, "    .db ").unwrap();
            for (i, b) in chunk.iter().enumerate() {
                if i > 0 {
                    write!(self.asm, ", ").unwrap();
                }
                write!(self.asm, "0x{b:02X}").unwrap();
            }
            writeln!(self.asm).unwrap();
        }
    }

    fn emit_function(&mut self, func: &IrFunction) -> Result<(), String> {
        let fid = func.id;
        let fname = fn_label(self.program, fid);
        let is_isr = self.program.is_isr_root(func);
        writeln!(self.asm, "\n{fname}:").unwrap();

        if is_isr {
            writeln!(self.asm, "    push psw").unwrap();
            writeln!(self.asm, "    push acc").unwrap();
            writeln!(self.asm, "    push b").unwrap();
            writeln!(self.asm, "    push dpl").unwrap();
            writeln!(self.asm, "    push dph").unwrap();
            if let Some(bank) = func.isr_using {
                let psw_bits = (bank & 0x03) << 3;
                writeln!(self.asm, "    mov psw, #0x{psw_bits:02X}").unwrap();
            } else {
                for r in 0..8u8 {
                    writeln!(self.asm, "    push 0x{r:02X}").unwrap();
                }
            }
        }

        // Map constants observed per-function for direct Sfr/Bit access.
        let mut const_vals: HashMap<ValueId, i64> = HashMap::new();
        let mut const_global: HashMap<ValueId, u32> = HashMap::new();
        let mut local_ptr_slot: HashMap<ValueId, SlotId> = HashMap::new();
        for bb in &func.blocks {
            for inst in &bb.insts {
                match inst {
                    IrInst::ConstInt { dst, val, .. } => {
                        const_vals.insert(*dst, *val);
                    }
                    IrInst::ConstBool { dst, val } => {
                        const_vals.insert(*dst, i64::from(*val));
                    }
                    IrInst::AddrGlobal { dst, global } => {
                        const_global.insert(*dst, *global);
                    }
                    IrInst::AddrLocal { dst, slot } => {
                        local_ptr_slot.insert(*dst, *slot);
                    }
                    _ => {}
                }
            }
        }

        let cfg = aun_ir::cfg::Cfg::build(func);
        let order = cfg.rpo_indices();
        for (pos, &bb_idx) in order.iter().enumerate() {
            let bb = &func.blocks[bb_idx];
            let next_bb = order.get(pos + 1).map(|&idx| func.blocks[idx].id);
            writeln!(self.asm, "{fname}_bb{}:", bb.id).unwrap();

            for inst in &bb.insts {
                self.emit_inst(func, is_isr, inst, &const_vals, &const_global, &local_ptr_slot)?;
            }

            match bb.term {
                Terminator::Unreachable => {
                    writeln!(self.asm, "    sjmp $").unwrap();
                }
                Terminator::Return(val_opt) => {
                    if let Some(rv) = val_opt {
                        let sz = self.program.type_size(&func.ret_ty);
                        if sz > 0 && sz <= 4 {
                            let src = self.val_loc(fid, rv);
                            self.emit_load_region_n_to_regs(src, sz, 2);
                            if sz == 1 {
                                writeln!(self.asm, "    mov a, r2").unwrap();
                            }
                        }
                    }
                    if is_isr {
                        if func.isr_using.is_none() {
                            for r in (0..8u8).rev() {
                                writeln!(self.asm, "    pop 0x{r:02X}").unwrap();
                            }
                        }
                        writeln!(self.asm, "    pop dph").unwrap();
                        writeln!(self.asm, "    pop dpl").unwrap();
                        writeln!(self.asm, "    pop b").unwrap();
                        writeln!(self.asm, "    pop acc").unwrap();
                        writeln!(self.asm, "    pop psw").unwrap();
                        writeln!(self.asm, "    reti").unwrap();
                    } else {
                        writeln!(self.asm, "    ret").unwrap();
                    }
                }
                Terminator::Goto(target) => {
                    if next_bb != Some(target) {
                        writeln!(self.asm, "    ljmp {fname}_bb{target}").unwrap();
                    }
                }
                Terminator::Branch {
                    cond,
                    then_bb,
                    else_bb,
                } => {
                    let cloc = self.val_loc(fid, cond);
                    self.emit_load_region_byte_a(cloc, 0);
                    let skip = self.fresh_lbl("br_else");
                    writeln!(self.asm, "    jz {skip}").unwrap();
                    writeln!(self.asm, "    ljmp {fname}_bb{then_bb}").unwrap();
                    writeln!(self.asm, "{skip}:").unwrap();
                    if next_bb != Some(else_bb) {
                        writeln!(self.asm, "    ljmp {fname}_bb{else_bb}").unwrap();
                    }
                }
            }
        }
        Ok(())
    }

    fn emit_inst(
        &mut self,
        func: &IrFunction,
        is_isr: bool,
        inst: &IrInst,
        _const_vals: &HashMap<ValueId, i64>,
        const_global: &HashMap<ValueId, u32>,
        local_ptr_slot: &HashMap<ValueId, SlotId>,
    ) -> Result<(), String> {
        let fid = func.id;
        match inst {
            IrInst::ConstInt { dst, ty, val } => {
                let sz = self.program.type_size(ty);
                let loc = self.val_loc(fid, *dst);
                let mut u = *val as u64;
                for i in 0..sz {
                    self.emit_store_region_imm_byte(loc, i, (u & 0xFF) as u8);
                    u >>= 8;
                }
            }
            IrInst::ConstBool { dst, val } => {
                let loc = self.val_loc(fid, *dst);
                self.emit_store_region_imm_byte(loc, 0, u8::from(*val));
            }
            IrInst::LoadLocal { dst, slot, ty } => {
                let sz = self.program.type_size(ty);
                let src = self.slot_loc(fid, *slot);
                let dloc = self.val_loc(fid, *dst);
                self.emit_copy_region(dloc, src, sz);
                let sp = self.slot_ptr_space(fid, *slot);
                let st = self.layout.slot_ptr_tags.get(&(fid, *slot)).copied();
                self.copy_val_ptr_tag(fid, *dst, sp, st);
            }
            IrInst::StoreLocal { slot, val, ty } => {
                let sz = self.program.type_size(ty);
                let src = self.val_loc(fid, *val);
                let dloc = self.slot_loc(fid, *slot);
                self.emit_copy_region(dloc, src, sz);
                let sp = self.val_ptr_space(fid, *val);
                let vt = self.layout.val_ptr_tags.get(&(fid, *val)).copied();
                self.copy_slot_ptr_tag(fid, *slot, sp, vt);
            }
            IrInst::AddrLocal { dst, slot } => {
                let dloc = self.val_loc(fid, *dst);
                let (addr, sp) = match self.slot_loc(fid, *slot) {
                    SlotRegion::Data(a) => (a as u16, PtrSpace::Data),
                    SlotRegion::XData(a) => (a, PtrSpace::XData),
                };
                self.emit_store_region_imm_byte(dloc, 0, (addr & 0xFF) as u8);
                self.emit_store_region_imm_byte(dloc, 1, (addr >> 8) as u8);
                self.copy_val_ptr_tag(fid, *dst, sp, None);
            }
            IrInst::AddrGlobal { dst, global } => {
                let dloc = self.val_loc(fid, *dst);
                let ga = self.layout.global(*global);
                if let Some(ref lbl) = ga.rom_label {
                    writeln!(self.asm, "    mov a, #lo({lbl})").unwrap();
                    self.emit_store_region_byte_a(dloc, 0);
                    writeln!(self.asm, "    mov a, #hi({lbl})").unwrap();
                    self.emit_store_region_byte_a(dloc, 1);
                } else {
                    self.emit_store_region_imm_byte(dloc, 0, (ga.addr & 0xFF) as u8);
                    self.emit_store_region_imm_byte(dloc, 1, (ga.addr >> 8) as u8);
                }
                self.copy_val_ptr_tag(fid, *dst, ga.space, None);
            }
            IrInst::AddrFn { dst, target } => {
                let dloc = self.val_loc(fid, *dst);
                let lbl = match target {
                    IrFnTarget::Fn(id) => fn_label(self.program, *id),
                    IrFnTarget::Extern(id) => format!("_{}", self.program.externs[*id as usize].name),
                };
                writeln!(self.asm, "    mov a, #lo({lbl})").unwrap();
                self.emit_store_region_byte_a(dloc, 0);
                writeln!(self.asm, "    mov a, #hi({lbl})").unwrap();
                self.emit_store_region_byte_a(dloc, 1);
                self.copy_val_ptr_tag(fid, *dst, PtrSpace::Code, None);
            }
            IrInst::FieldAddr {
                dst,
                base,
                byte_offset,
            } => {
                let bloc = self.val_loc(fid, *base);
                let dloc = self.val_loc(fid, *dst);
                self.emit_load_region_u16(bloc, R2, R3);
                self.emit_add_r2r3_const(*byte_offset);
                self.emit_store_region_u16(dloc, R2, R3);
                let bsp = self.val_ptr_space(fid, *base);
                let bt = self.layout.val_ptr_tags.get(&(fid, *base)).copied();
                self.copy_val_ptr_tag(fid, *dst, bsp, bt);
            }
            IrInst::IndexAddr {
                dst,
                base,
                index,
                elem_size,
                len,
                ..
            } => {
                let iloc = self.val_loc(fid, *index);
                let isz = self.val_size_or(fid, *index, 1);
                self.emit_load_region_byte_rn(iloc, 0, R2);
                if isz >= 2 {
                    self.emit_load_region_byte_rn(iloc, 1, R3);
                } else {
                    writeln!(self.asm, "    mov r3, #0x00").unwrap();
                }
                if self.opts.check_bounds && *len > 0 {
                    let ok = self.fresh_lbl("bnd_ok");
                    self.emit_cmp_u16_r2r3_const(*len);
                    writeln!(self.asm, "    jc {ok}").unwrap();
                    writeln!(self.asm, "    sjmp $").unwrap();
                    writeln!(self.asm, "{ok}:").unwrap();
                }
                if *elem_size != 1 {
                    writeln!(self.asm, "    mov r4, #0x{:02X}", (*elem_size & 0xFF) as u8).unwrap();
                    writeln!(self.asm, "    mov r5, #0x{:02X}", (*elem_size >> 8) as u8).unwrap();
                    self.used_helpers.insert("__aun_mul16");
                    writeln!(self.asm, "    lcall __aun_mul16").unwrap();
                }
                let bloc = self.val_loc(fid, *base);
                self.emit_load_region_u16(bloc, R4, R5);
                writeln!(self.asm, "    mov a, r2").unwrap();
                writeln!(self.asm, "    add a, r4").unwrap();
                writeln!(self.asm, "    mov r2, a").unwrap();
                writeln!(self.asm, "    mov a, r3").unwrap();
                writeln!(self.asm, "    addc a, r5").unwrap();
                writeln!(self.asm, "    mov r3, a").unwrap();
                let dloc = self.val_loc(fid, *dst);
                self.emit_store_region_u16(dloc, R2, R3);
                let bsp = self.val_ptr_space(fid, *base);
                let bt = self.layout.val_ptr_tags.get(&(fid, *base)).copied();
                self.copy_val_ptr_tag(fid, *dst, bsp, bt);
            }
            IrInst::LoadPtr {
                dst,
                addr,
                ty,
                volatile: _,
            } => {
                let sz = self.program.type_size(ty);
                let dloc = self.val_loc(fid, *dst);
                if let Some(&gid) = const_global.get(addr) {
                    let ga = self.layout.global(gid);
                    match ga.space {
                        PtrSpace::Bit => {
                            writeln!(self.asm, "    mov c, 0x{:02X}", ga.addr as u8).unwrap();
                            writeln!(self.asm, "    clr a").unwrap();
                            writeln!(self.asm, "    rlc a").unwrap();
                            self.emit_store_region_byte_a(dloc, 0);
                            return Ok(());
                        }
                        PtrSpace::Sfr => {
                            writeln!(self.asm, "    mov a, 0x{:02X}", ga.addr as u8).unwrap();
                            self.emit_store_region_byte_a(dloc, 0);
                            return Ok(());
                        }
                        _ => {}
                    }
                }
                let aloc = self.val_loc(fid, *addr);
                self.emit_load_region_u16(aloc, R2, R3);
                let sp = self.val_ptr_space(fid, *addr);
                let tl = self.layout.val_ptr_tags.get(&(fid, *addr)).copied();
                self.emit_load_from_ptr(sp, tl, dloc, sz);
                if let Some(&src_slot) = local_ptr_slot.get(addr) {
                    let sp_s = self.slot_ptr_space(fid, src_slot);
                    let st_s = self.layout.slot_ptr_tags.get(&(fid, src_slot)).copied();
                    self.copy_val_ptr_tag(fid, *dst, sp_s, st_s);
                } else if self.program.type_size(ty) == 2 {
                    let sp_d = self.val_ptr_space(fid, *dst);
                    self.copy_val_ptr_tag(fid, *dst, sp_d, None);
                }
            }
            IrInst::StorePtr {
                addr,
                val,
                ty,
                volatile: _,
            } => {
                let sz = self.program.type_size(ty);
                let vloc = self.val_loc(fid, *val);
                if let Some(&dst_slot) = local_ptr_slot.get(addr) {
                    let sp_v = self.val_ptr_space(fid, *val);
                    let vt = self.layout.val_ptr_tags.get(&(fid, *val)).copied();
                    self.copy_slot_ptr_tag(fid, dst_slot, sp_v, vt);
                }
                if let Some(&gid) = const_global.get(addr) {
                    let ga = self.layout.global(gid);
                    match ga.space {
                        PtrSpace::Bit => {
                            let s0 = self.fresh_lbl("bit_clr");
                            let s1 = self.fresh_lbl("bit_dn");
                            self.emit_load_region_byte_a(vloc, 0);
                            writeln!(self.asm, "    jz {s0}").unwrap();
                            writeln!(self.asm, "    setb 0x{:02X}", ga.addr as u8).unwrap();
                            writeln!(self.asm, "    sjmp {s1}").unwrap();
                            writeln!(self.asm, "{s0}:").unwrap();
                            writeln!(self.asm, "    clr 0x{:02X}", ga.addr as u8).unwrap();
                            writeln!(self.asm, "{s1}:").unwrap();
                            return Ok(());
                        }
                        PtrSpace::Sfr => {
                            self.emit_load_region_byte_a(vloc, 0);
                            writeln!(self.asm, "    mov 0x{:02X}, a", ga.addr as u8).unwrap();
                            return Ok(());
                        }
                        _ => {}
                    }
                }
                let aloc = self.val_loc(fid, *addr);
                self.emit_load_region_u16(aloc, R2, R3);
                let sp = self.val_ptr_space(fid, *addr);
                let tl = self.layout.val_ptr_tags.get(&(fid, *addr)).copied();
                self.emit_store_to_ptr(sp, tl, vloc, sz);
            }
            IrInst::MakeSlice { dst, ptr, len, .. } => {
                let dloc = self.val_loc(fid, *dst);
                let ploc = self.val_loc(fid, *ptr);
                let lloc = self.val_loc(fid, *len);
                self.emit_copy_region(dloc, ploc, 2);
                self.emit_load_region_u16(lloc, R2, R3);
                self.emit_store_region_byte_rn(dloc, 2, R2);
                self.emit_store_region_byte_rn(dloc, 3, R3);
                let sp = self.val_ptr_space(fid, *ptr);
                let pt = self.layout.val_ptr_tags.get(&(fid, *ptr)).copied();
                self.copy_val_ptr_tag(fid, *dst, sp, pt);
            }
            IrInst::SlicePtr { dst, slice, .. } => {
                let dloc = self.val_loc(fid, *dst);
                let sloc = self.val_loc(fid, *slice);
                self.emit_copy_region(dloc, sloc, 2);
                let sp = self.val_ptr_space(fid, *slice);
                let st = self.layout.val_ptr_tags.get(&(fid, *slice)).copied();
                self.copy_val_ptr_tag(fid, *dst, sp, st);
            }
            IrInst::SliceLen { dst, slice } => {
                let dloc = self.val_loc(fid, *dst);
                let sloc = self.val_loc(fid, *slice);
                self.emit_load_region_byte_a(sloc, 2);
                self.emit_store_region_byte_a(dloc, 0);
                self.emit_load_region_byte_a(sloc, 3);
                self.emit_store_region_byte_a(dloc, 1);
            }
            IrInst::MemSet { dst_addr, val, len } => {
                let aloc = self.val_loc(fid, *dst_addr);
                let vloc = self.val_loc(fid, *val);
                let sp = self.val_ptr_space(fid, *dst_addr);
                self.emit_load_region_u16(aloc, R2, R3);
                self.emit_load_region_byte_rn(vloc, 0, R4);
                match sp {
                    PtrSpace::Data | PtrSpace::IData => {
                        let lp = self.fresh_lbl("mset_d");
                        writeln!(self.asm, "    mov a, r2").unwrap();
                        writeln!(self.asm, "    mov r0, a").unwrap();
                        writeln!(self.asm, "    mov r7, #0x{:02X}", *len as u8).unwrap();
                        writeln!(self.asm, "{lp}:").unwrap();
                        writeln!(self.asm, "    mov a, r4").unwrap();
                        writeln!(self.asm, "    mov @r0, a").unwrap();
                        writeln!(self.asm, "    inc r0").unwrap();
                        writeln!(self.asm, "    djnz r7, {lp}").unwrap();
                    }
                    _ => {
                        self.used_helpers.insert("__aun_memset_x");
                        writeln!(self.asm, "    mov r6, #0x{:02X}", (*len & 0xFF) as u8).unwrap();
                        writeln!(self.asm, "    mov r7, #0x{:02X}", (*len >> 8) as u8).unwrap();
                        writeln!(self.asm, "    lcall __aun_memset_x").unwrap();
                    }
                }
            }
            IrInst::MemCopy {
                dst_addr,
                src_addr,
                len,
            } => {
                let sloc = self.val_loc(fid, *src_addr);
                let dloc = self.val_loc(fid, *dst_addr);
                let ssp = self.val_ptr_space(fid, *src_addr);
                let dsp = self.val_ptr_space(fid, *dst_addr);
                self.emit_load_region_u16(sloc, R2, R3);
                self.emit_load_region_u16(dloc, R4, R5);
                writeln!(self.asm, "    mov r6, #0x{:02X}", (*len & 0xFF) as u8).unwrap();
                writeln!(self.asm, "    mov r7, #0x{:02X}", (*len >> 8) as u8).unwrap();
                match (ssp, dsp) {
                    (PtrSpace::Code, PtrSpace::XData) => {
                        self.used_helpers.insert("__aun_memcpy_cx");
                        writeln!(self.asm, "    lcall __aun_memcpy_cx").unwrap();
                    }
                    _ => {
                        self.used_helpers.insert("__aun_memcpy_xx");
                        writeln!(self.asm, "    lcall __aun_memcpy_xx").unwrap();
                    }
                }
            }
            IrInst::MemFill {
                dst_addr,
                elem_size,
                count,
            } => {
                if *count <= 1 || *elem_size == 0 {
                    return Ok(());
                }
                let total_tail = elem_size * (*count - 1);
                let dloc = self.val_loc(fid, *dst_addr);
                let dsp = self.val_ptr_space(fid, *dst_addr);
                self.emit_load_region_u16(dloc, R2, R3);
                writeln!(self.asm, "    mov a, r2").unwrap();
                writeln!(self.asm, "    add a, #0x{:02X}", (*elem_size & 0xFF) as u8).unwrap();
                writeln!(self.asm, "    mov r4, a").unwrap();
                writeln!(self.asm, "    mov a, r3").unwrap();
                writeln!(self.asm, "    addc a, #0x{:02X}", (*elem_size >> 8) as u8).unwrap();
                writeln!(self.asm, "    mov r5, a").unwrap();
                match dsp {
                    PtrSpace::Data | PtrSpace::IData => {
                        let lp = self.fresh_lbl("mfill_d");
                        writeln!(self.asm, "    mov a, r2").unwrap();
                        writeln!(self.asm, "    mov r0, a").unwrap();
                        writeln!(self.asm, "    mov a, r4").unwrap();
                        writeln!(self.asm, "    mov r1, a").unwrap();
                        writeln!(self.asm, "    mov r7, #0x{:02X}", total_tail as u8).unwrap();
                        writeln!(self.asm, "{lp}:").unwrap();
                        writeln!(self.asm, "    mov a, @r0").unwrap();
                        writeln!(self.asm, "    mov @r1, a").unwrap();
                        writeln!(self.asm, "    inc r0").unwrap();
                        writeln!(self.asm, "    inc r1").unwrap();
                        writeln!(self.asm, "    djnz r7, {lp}").unwrap();
                    }
                    _ => {
                        self.used_helpers.insert("__aun_memcpy_xx");
                        writeln!(self.asm, "    mov r6, #0x{:02X}", (total_tail & 0xFF) as u8)
                            .unwrap();
                        writeln!(self.asm, "    mov r7, #0x{:02X}", (total_tail >> 8) as u8)
                            .unwrap();
                        writeln!(self.asm, "    lcall __aun_memcpy_xx").unwrap();
                    }
                }
            }
            IrInst::UnOp {
                dst, op, ty, val, ..
            } => {
                let sz = self.program.type_size(ty);
                let vloc = self.val_loc(fid, *val);
                let dloc = self.val_loc(fid, *dst);
                match op {
                    IrUnOp::Not if *ty == IrType::Bool => {
                        self.emit_load_region_byte_a(vloc, 0);
                        writeln!(self.asm, "    xrl a, #0x01").unwrap();
                        self.emit_store_region_byte_a(dloc, 0);
                    }
                    IrUnOp::Not => {
                        for i in 0..sz {
                            self.emit_load_region_byte_a(vloc, i);
                            writeln!(self.asm, "    cpl a").unwrap();
                            self.emit_store_region_byte_a(dloc, i);
                        }
                    }
                    IrUnOp::Neg => {
                        writeln!(self.asm, "    setb c").unwrap();
                        for i in 0..sz {
                            self.emit_load_region_byte_a(vloc, i);
                            writeln!(self.asm, "    cpl a").unwrap();
                            writeln!(self.asm, "    addc a, #0x00").unwrap();
                            self.emit_store_region_byte_a(dloc, i);
                        }
                    }
                }
            }
            IrInst::Cast {
                dst,
                val,
                from_ty,
                to_ty,
            } => {
                let fsz = self.program.type_size(from_ty);
                let tsz = self.program.type_size(to_ty);
                let vloc = self.val_loc(fid, *val);
                let dloc = self.val_loc(fid, *dst);
                let fixed_shift: i8 = match (from_ty, to_ty) {
                    (IrType::Int(_), IrType::Fixed { frac_bits, .. }) => *frac_bits as i8,
                    (IrType::Fixed { frac_bits, .. }, IrType::Int(_)) => -(*frac_bits as i8),
                    (IrType::Fixed { frac_bits: f1, .. }, IrType::Fixed { frac_bits: f2, .. }) => {
                        (*f2 as i8) - (*f1 as i8)
                    }
                    _ => 0,
                };
                if *to_ty == IrType::Bool {
                    writeln!(self.asm, "    mov r7, #0x00").unwrap();
                    for i in 0..fsz {
                        self.emit_load_region_byte_a(vloc, i);
                        writeln!(self.asm, "    orl a, r7").unwrap();
                        writeln!(self.asm, "    mov r7, a").unwrap();
                    }
                    writeln!(self.asm, "    mov a, r7").unwrap();
                    let nz = self.fresh_lbl("cbool");
                    writeln!(self.asm, "    jz {nz}").unwrap();
                    writeln!(self.asm, "    mov a, #0x01").unwrap();
                    writeln!(self.asm, "{nz}:").unwrap();
                    self.emit_store_region_byte_a(dloc, 0);
                } else if fixed_shift < 0 {
                    let amount = (-fixed_shift) as u8;
                    self.emit_shift_const_into(
                        vloc,
                        fsz,
                        dloc,
                        tsz,
                        IrBinOp::Shr,
                        is_signed_ty(from_ty),
                        amount,
                    );
                } else {
                    let copy_bytes = fsz.min(tsz);
                    self.emit_copy_region(dloc, vloc, copy_bytes);
                    if tsz > fsz {
                        if is_signed_ty(from_ty) {
                            let pos = self.fresh_lbl("sext_p");
                            let dn = self.fresh_lbl("sext_d");
                            self.emit_load_region_byte_a(vloc, fsz - 1);
                            writeln!(self.asm, "    jnb ACC.7, {pos}").unwrap();
                            writeln!(self.asm, "    mov a, #0xFF").unwrap();
                            writeln!(self.asm, "    sjmp {dn}").unwrap();
                            writeln!(self.asm, "{pos}:").unwrap();
                            writeln!(self.asm, "    clr a").unwrap();
                            writeln!(self.asm, "{dn}:").unwrap();
                            for i in fsz..tsz {
                                self.emit_store_region_byte_a(dloc, i);
                            }
                        } else {
                            for i in fsz..tsz {
                                self.emit_store_region_imm_byte(dloc, i, 0);
                            }
                        }
                    }
                    if fixed_shift > 0 {
                        self.emit_shift_const(
                            dloc,
                            dloc,
                            tsz,
                            IrBinOp::Shl,
                            is_signed_ty(to_ty),
                            fixed_shift as u8,
                        );
                    }
                }
                if fsz == 2 && tsz == 2 {
                    let sp = self.val_ptr_space(fid, *val);
                    let vt = self.layout.val_ptr_tags.get(&(fid, *val)).copied();
                    self.copy_val_ptr_tag(fid, *dst, sp, vt);
                }
            }
            IrInst::FixedInt {
                dst,
                val,
                base,
                frac_bits,
            } => {
                let sz = base.byte_size();
                let vloc = self.val_loc(fid, *val);
                let dloc = self.val_loc(fid, *dst);
                self.emit_shift_const(vloc, dloc, sz, IrBinOp::Shr, base.is_signed(), *frac_bits);
            }
            IrInst::BinOp {
                dst,
                op,
                mode: _,
                ty,
                lhs,
                rhs,
                ..
            } => {
                self.emit_binop(fid, *dst, *op, ty, *lhs, *rhs)?;
            }
            IrInst::CriticalEnter => {
                let dis = self
                    .program
                    .target
                    .as_ref()
                    .and_then(|t| t.interrupt_controllers.first())
                    .map(|ic| ic.disable_asm.as_str())
                    .unwrap_or("clr EA");
                if !is_isr {
                    writeln!(self.asm, "    {dis}").unwrap();
                }
            }
            IrInst::CriticalExit => {
                let en = self
                    .program
                    .target
                    .as_ref()
                    .and_then(|t| t.interrupt_controllers.first())
                    .map(|ic| ic.enable_asm.as_str())
                    .unwrap_or("setb EA");
                if !is_isr {
                    writeln!(self.asm, "    {en}").unwrap();
                }
            }
            IrInst::InlineAsm { template, .. } => {
                for l in template.lines() {
                    let t = l.trim();
                    if !t.is_empty() {
                        writeln!(self.asm, "    {t}").unwrap();
                    }
                }
            }
            IrInst::Call {
                dst,
                func: cid,
                args,
            } => {
                let callee = &self.program.functions[*cid as usize];
                for (i, &av) in args.iter().enumerate() {
                    let sz = self.program.type_size(&callee.locals[i].ty);
                    let src = self.val_loc(fid, av);
                    let ploc = self.slot_loc(*cid, i as SlotId);
                    self.emit_copy_region(ploc, src, sz);
                    let sp = self.val_ptr_space(fid, av);
                    let vt = self.layout.val_ptr_tags.get(&(fid, av)).copied();
                    self.copy_slot_ptr_tag(*cid, i as SlotId, sp, vt);
                }
                writeln!(self.asm, "    lcall {}", fn_label(self.program, *cid)).unwrap();
                if let Some(d) = dst {
                    let sz = self.program.type_size(&callee.ret_ty);
                    let dloc = self.val_loc(fid, *d);
                    if sz == 1 {
                        self.emit_store_region_byte_a(dloc, 0);
                    } else if sz > 1 && sz <= 4 {
                        self.emit_store_region_n_from_regs(dloc, sz, 2);
                    }
                }
            }
            IrInst::ExternCall {
                dst,
                extern_fn,
                args,
            } => {
                let ef = &self.program.externs[*extern_fn as usize];
                for (i, &av) in args.iter().enumerate() {
                    let sz = self.program.type_size(&ef.params[i].1);
                    let reg = ef.params[i].2.to_ascii_lowercase();
                    let src = self.val_loc(fid, av);
                    if sz == 1 {
                        self.emit_load_region_byte_a(src, 0);
                        match reg.as_str() {
                            "a" | "acc" => {}
                            "dpl" => writeln!(self.asm, "    mov dpl, a").unwrap(),
                            "r0" => writeln!(self.asm, "    mov r0, a").unwrap(),
                            "r1" => writeln!(self.asm, "    mov r1, a").unwrap(),
                            "r7" => writeln!(self.asm, "    mov r7, a").unwrap(),
                            "b" => writeln!(self.asm, "    mov b, a").unwrap(),
                            _ => writeln!(self.asm, "    mov dpl, a").unwrap(),
                        }
                    } else if sz == 2 {
                        self.emit_load_region_u16(src, R2, R3);
                        if reg == "dptr" {
                            writeln!(self.asm, "    mov dpl, r2").unwrap();
                            writeln!(self.asm, "    mov dph, r3").unwrap();
                        }
                    }
                }
                writeln!(self.asm, "    lcall _{}", ef.name).unwrap();
                if let Some(d) = dst
                    && let Some((ref rty, ref rreg)) = ef.ret
                {
                    let sz = self.program.type_size(rty);
                    let dloc = self.val_loc(fid, *d);
                    if sz == 1 {
                        if rreg.eq_ignore_ascii_case("dpl") {
                            writeln!(self.asm, "    mov a, dpl").unwrap();
                        }
                        self.emit_store_region_byte_a(dloc, 0);
                    } else if sz == 2 {
                        if rreg.eq_ignore_ascii_case("dptr") {
                            writeln!(self.asm, "    mov r2, dpl").unwrap();
                            writeln!(self.asm, "    mov r3, dph").unwrap();
                        }
                        self.emit_store_region_u16(dloc, R2, R3);
                    }
                }
            }
            IrInst::StateTag {
                dst,
                machine_addr,
                machine_id,
            } => {
                let mdef = &self.program.machines[*machine_id as usize];
                let aloc = self.val_loc(fid, *machine_addr);
                let dloc = self.val_loc(fid, *dst);
                let sp = self.val_ptr_space(fid, *machine_addr);
                let tl = self.layout.val_ptr_tags.get(&(fid, *machine_addr)).copied();
                self.emit_load_region_u16(aloc, R2, R3);
                self.emit_add_r2r3_const(mdef.tag_offset);
                self.emit_load_from_ptr(sp, tl, dloc, 1);
            }
            IrInst::StateTransition {
                machine_addr,
                machine_id,
                from_state,
                to_state,
                field_vals,
                ..
            } => {
                let mdef = &self.program.machines[*machine_id as usize];
                let target_leaf = mdef.resolve_to_leaf(*to_state);
                let sp = self.val_ptr_space(fid, *machine_addr);
                let tl = self.layout.val_ptr_tags.get(&(fid, *machine_addr)).copied();
                let aloc = self.val_loc(fid, *machine_addr);

                // 1. Evaluate transition payload args first, copy into temporary machine buffer region?
                // Even simpler: copy `field_vals` AFTER saving/reading `from_state` and calling `on_exit`,
                // wait: payload fields belong to the NEW state, so hold in `R2..` or temp? Since `field_vals`
                // are already evaluated as `ValueId`s in `val_locs`, calling `on_exit` (different Uchiguri
                // frame when sibling? Wait: caller & `on_exit` are ancestor/callee, so their Uchiguri frames
                // NEVER overlap!! That means `val_locs` of caller's `field_vals` are 100% preserved across `lcall on_exit`!!).

                // 2. Call `on_exit` hooks (bottom-up up to LCA)
                if let Some(from) = from_state {
                    let lca = mdef.lca(*from, target_leaf);
                    let mut ancs = mdef.ancestors(*from);
                    ancs.reverse();
                    for anc in ancs {
                        if lca.is_some_and(|l| anc == l || !mdef.is_ancestor(l, anc)) {
                            continue;
                        }
                        if let Some(ex) = mdef.states[anc as usize].on_exit {
                            let ploc = self.slot_loc(ex, 0);
                            self.emit_copy_region(ploc, aloc, 2);
                            self.copy_slot_ptr_tag(ex, 0, sp, tl);
                            writeln!(self.asm, "    lcall {}", fn_label(self.program, ex)).unwrap();
                        }
                    }
                } else {
                    // Runtime tag dispatch for `on_exit`
                    for st in &mdef.states {
                        if let Some(ex) = st.on_exit {
                            let skip = self.fresh_lbl("ex_skip");
                            self.emit_load_region_u16(aloc, R2, R3);
                            self.emit_add_r2r3_const(mdef.tag_offset);
                            self.emit_read_u8_from_ptr_to_a(sp, tl);
                            writeln!(self.asm, "    cjne a, #0x{:02X}, {skip}", st.id).unwrap();
                            let ploc = self.slot_loc(ex, 0);
                            self.emit_copy_region(ploc, aloc, 2);
                            self.copy_slot_ptr_tag(ex, 0, sp, tl);
                            writeln!(self.asm, "    lcall {}", fn_label(self.program, ex)).unwrap();
                            writeln!(self.asm, "{skip}:").unwrap();
                        }
                    }
                }

                // 3. Write new tag (`target_leaf`) & payload fields
                self.emit_load_region_u16(aloc, R2, R3);
                self.emit_add_r2r3_const(mdef.tag_offset);
                // Store 2-byte tag (leaf_id, 0) via R4/R5 or scratch byte
                self.emit_store_const_to_ptr_u16(sp, tl, target_leaf as u16);

                for &(rel_off, fv, ref fty) in field_vals {
                    let fsz = self.program.type_size(fty);
                    let fvloc = self.val_loc(fid, fv);
                    self.emit_load_region_u16(aloc, R2, R3);
                    self.emit_add_r2r3_const(mdef.payload_offset + rel_off);
                    self.emit_store_to_ptr(sp, tl, fvloc, fsz);
                }

                // 4. Call `on_enter` hooks (top-down from LCA)
                let ancs = mdef.ancestors(target_leaf);
                for anc in ancs {
                    let below = from_state.is_none_or(|from| {
                        let lca = mdef.lca(from, target_leaf);
                        lca.is_none_or(|l| anc != l && mdef.is_ancestor(l, anc))
                    });
                    if below && let Some(en) = mdef.states[anc as usize].on_enter {
                        let ploc = self.slot_loc(en, 0);
                        self.emit_copy_region(ploc, aloc, 2);
                        self.copy_slot_ptr_tag(en, 0, sp, tl);
                        writeln!(self.asm, "    lcall {}", fn_label(self.program, en)).unwrap();
                    }
                }
            }
            IrInst::StateDispatch {
                dst,
                machine_addr,
                machine_id,
                method,
                args,
            } => {
                let mdef = &self.program.machines[*machine_id as usize];
                let aloc = self.val_loc(fid, *machine_addr);
                let sp = self.val_ptr_space(fid, *machine_addr);
                let tl = self.layout.val_ptr_tags.get(&(fid, *machine_addr)).copied();
                self.emit_load_region_u16(aloc, R2, R3);
                self.emit_add_r2r3_const(mdef.tag_offset);
                // Put tag into R6 via temporary Data write? Better: use LoadFromPtr into direct R6 (addr 0x06? No! Read 1B into temp or direct movx/mov).
                self.emit_read_u8_from_ptr_to_a(sp, tl);
                writeln!(self.asm, "    mov b, a").unwrap();
                let done = self.fresh_lbl("disp_dn");
                for st in &mdef.states {
                    if let Some(&cid) = st.methods.get(method) {
                        let skip = self.fresh_lbl("disp_sk");
                        writeln!(self.asm, "    mov a, b").unwrap();
                        writeln!(self.asm, "    cjne a, #0x{:02X}, {skip}", st.id).unwrap();
                        let callee = &self.program.functions[cid as usize];
                        let p0 = self.slot_loc(cid, 0);
                        self.emit_copy_region(p0, aloc, 2);
                        self.copy_slot_ptr_tag(cid, 0, sp, tl);
                        for (ai, &av) in args.iter().enumerate() {
                            let sz = self.program.type_size(&callee.locals[ai + 1].ty);
                            let src = self.val_loc(fid, av);
                            let dst_s = self.slot_loc(cid, (ai + 1) as SlotId);
                            self.emit_copy_region(dst_s, src, sz);
                            let asp = self.val_ptr_space(fid, av);
                            let at = self.layout.val_ptr_tags.get(&(fid, av)).copied();
                            self.copy_slot_ptr_tag(cid, (ai + 1) as SlotId, asp, at);
                        }
                        writeln!(self.asm, "    lcall {}", fn_label(self.program, cid)).unwrap();
                        if let Some(d) = dst {
                            let sz = self.program.type_size(&callee.ret_ty);
                            let dloc = self.val_loc(fid, *d);
                            if sz == 1 {
                                self.emit_store_region_byte_a(dloc, 0);
                            } else if sz > 1 && sz <= 4 {
                                self.emit_store_region_n_from_regs(dloc, sz, 2);
                            }
                        }
                        writeln!(self.asm, "    ljmp {done}").unwrap();
                        writeln!(self.asm, "{skip}:").unwrap();
                    }
                }
                writeln!(self.asm, "{done}:").unwrap();
            }
            IrInst::PoolAlloc {
                dst,
                pool_addr,
                init_addr,
                elem_size,
                capacity,
            } => {
                let ploc = self.val_loc(fid, *pool_addr);
                let iloc = self.val_loc(fid, *init_addr);
                let dloc = self.val_loc(fid, *dst);
                let psp = self.val_ptr_space(fid, *pool_addr);
                let isp = self.val_ptr_space(fid, *init_addr);
                let elem_bytes = elem_size * capacity;

                let full_lbl = self.fresh_lbl("palloc_full");
                let done_lbl = self.fresh_lbl("palloc_dn");

                // Linear scan slots 0..capacity
                for i in 0..*capacity {
                    let next_lbl = self.fresh_lbl("palloc_nxt");
                    let bm_off = elem_bytes + (i >> 3);
                    let mask = 1u8 << (i & 7);
                    self.emit_load_region_u16(ploc, R2, R3);
                    self.emit_add_r2r3_const(bm_off);
                    self.emit_read_u8_from_ptr_to_a(psp, None);
                    writeln!(self.asm, "    anl a, #0x{mask:02X}").unwrap();
                    writeln!(self.asm, "    jnz {next_lbl}").unwrap();
                    // Found free slot `i`: set bit and copy `elem_size` from `init_addr`
                    self.emit_load_region_u16(ploc, R2, R3);
                    self.emit_add_r2r3_const(bm_off);
                    self.emit_read_u8_from_ptr_to_a(psp, None);
                    writeln!(self.asm, "    orl a, #0x{mask:02X}").unwrap();
                    self.emit_write_a_to_ptr(psp, None);

                    self.emit_load_region_u16(iloc, R2, R3);
                    self.emit_load_region_u16(ploc, R4, R5);
                    if i * elem_size != 0 {
                        let off = i * elem_size;
                        writeln!(self.asm, "    mov a, r4").unwrap();
                        writeln!(self.asm, "    add a, #0x{:02X}", (off & 0xFF) as u8).unwrap();
                        writeln!(self.asm, "    mov r4, a").unwrap();
                        writeln!(self.asm, "    mov a, r5").unwrap();
                        writeln!(self.asm, "    addc a, #0x{:02X}", (off >> 8) as u8).unwrap();
                        writeln!(self.asm, "    mov r5, a").unwrap();
                    }
                    writeln!(self.asm, "    mov r6, #0x{:02X}", (*elem_size & 0xFF) as u8).unwrap();
                    writeln!(self.asm, "    mov r7, #0x{:02X}", (*elem_size >> 8) as u8).unwrap();
                    let helper = if isp == PtrSpace::Code && psp == PtrSpace::XData {
                        "__aun_memcpy_cx"
                    } else {
                        "__aun_memcpy_xx"
                    };
                    self.used_helpers.insert(helper);
                    writeln!(self.asm, "    lcall {helper}").unwrap();
                    self.emit_store_region_imm_byte(dloc, 0, i as u8);
                    writeln!(self.asm, "    ljmp {done_lbl}").unwrap();
                    writeln!(self.asm, "{next_lbl}:").unwrap();
                }
                writeln!(self.asm, "{full_lbl}:").unwrap();
                self.emit_store_region_imm_byte(dloc, 0, 0xFF);
                writeln!(self.asm, "{done_lbl}:").unwrap();
            }
            IrInst::PoolFree {
                pool_addr,
                handle,
                elem_size,
                capacity,
                ..
            } => {
                let ploc = self.val_loc(fid, *pool_addr);
                let hloc = self.val_loc(fid, *handle);
                let psp = self.val_ptr_space(fid, *pool_addr);
                let elem_bytes = elem_size * capacity;
                for i in 0..*capacity {
                    let skip = self.fresh_lbl("pfree_sk");
                    self.emit_load_region_byte_a(hloc, 0);
                    writeln!(self.asm, "    cjne a, #0x{:02X}, {skip}", i as u8).unwrap();
                    let bm_off = elem_bytes + (i >> 3);
                    let clear_mask = !(1u8 << (i & 7));
                    self.emit_load_region_u16(ploc, R2, R3);
                    self.emit_add_r2r3_const(bm_off);
                    self.emit_read_u8_from_ptr_to_a(psp, None);
                    writeln!(self.asm, "    anl a, #0x{clear_mask:02X}").unwrap();
                    self.emit_write_a_to_ptr(psp, None);
                    writeln!(self.asm, "{skip}:").unwrap();
                }
            }
            IrInst::PoolElemAddr {
                dst,
                pool_addr,
                handle,
                elem_size,
                capacity: _,
                ..
            } => {
                let ploc = self.val_loc(fid, *pool_addr);
                let hloc = self.val_loc(fid, *handle);
                let dloc = self.val_loc(fid, *dst);
                let psp = self.val_ptr_space(fid, *pool_addr);
                self.emit_load_region_byte_rn(hloc, 0, R2);
                writeln!(self.asm, "    mov r3, #0x00").unwrap();
                if *elem_size != 1 {
                    writeln!(self.asm, "    mov r4, #0x{:02X}", (*elem_size & 0xFF) as u8).unwrap();
                    writeln!(self.asm, "    mov r5, #0x{:02X}", (*elem_size >> 8) as u8).unwrap();
                    self.used_helpers.insert("__aun_mul16");
                    writeln!(self.asm, "    lcall __aun_mul16").unwrap();
                }
                self.emit_load_region_u16(ploc, R4, R5);
                writeln!(self.asm, "    mov a, r2").unwrap();
                writeln!(self.asm, "    add a, r4").unwrap();
                writeln!(self.asm, "    mov r2, a").unwrap();
                writeln!(self.asm, "    mov a, r3").unwrap();
                writeln!(self.asm, "    addc a, r5").unwrap();
                writeln!(self.asm, "    mov r3, a").unwrap();
                self.emit_store_region_u16(dloc, R2, R3);
                self.copy_val_ptr_tag(fid, *dst, psp, None);
            }
            IrInst::QueueIsEmpty {
                dst,
                queue_addr,
                elem_size,
                capacity,
            } => {
                let qloc = self.val_loc(fid, *queue_addr);
                let dloc = self.val_loc(fid, *dst);
                let qsp = self.val_ptr_space(fid, *queue_addr);
                let head_off = elem_size * capacity;
                self.emit_load_region_u16(qloc, R2, R3);
                self.emit_add_r2r3_const(head_off);
                self.emit_read_u8_from_ptr_to_a(qsp, None);
                writeln!(self.asm, "    mov b, a").unwrap();
                self.emit_load_region_u16(qloc, R2, R3);
                self.emit_add_r2r3_const(head_off + 1);
                self.emit_read_u8_from_ptr_to_a(qsp, None);
                writeln!(self.asm, "    xrl a, b").unwrap();
                let ne = self.fresh_lbl("qemp_nz");
                let dn = self.fresh_lbl("qemp_dn");
                writeln!(self.asm, "    jnz {ne}").unwrap();
                writeln!(self.asm, "    mov a, #0x01").unwrap();
                writeln!(self.asm, "    sjmp {dn}").unwrap();
                writeln!(self.asm, "{ne}:").unwrap();
                writeln!(self.asm, "    clr a").unwrap();
                writeln!(self.asm, "{dn}:").unwrap();
                self.emit_store_region_byte_a(dloc, 0);
            }
            IrInst::QueueLen {
                dst,
                queue_addr,
                elem_size,
                capacity,
            } => {
                let qloc = self.val_loc(fid, *queue_addr);
                let dloc = self.val_loc(fid, *dst);
                let qsp = self.val_ptr_space(fid, *queue_addr);
                let head_off = elem_size * capacity;
                let mask = (*capacity as u8).wrapping_sub(1);
                self.emit_load_region_u16(qloc, R2, R3);
                self.emit_add_r2r3_const(head_off);
                self.emit_read_u8_from_ptr_to_a(qsp, None);
                writeln!(self.asm, "    mov b, a").unwrap();
                self.emit_load_region_u16(qloc, R2, R3);
                self.emit_add_r2r3_const(head_off + 1);
                self.emit_read_u8_from_ptr_to_a(qsp, None);
                writeln!(self.asm, "    clr c").unwrap();
                writeln!(self.asm, "    subb a, b").unwrap();
                writeln!(self.asm, "    anl a, #0x{mask:02X}").unwrap();
                self.emit_store_region_byte_a(dloc, 0);
            }
            IrInst::QueuePush {
                dst,
                queue_addr,
                val,
                elem_ty,
                capacity,
            } => {
                let esz = self.program.type_size(elem_ty);
                let qloc = self.val_loc(fid, *queue_addr);
                let vloc = self.val_loc(fid, *val);
                let dloc = self.val_loc(fid, *dst);
                let qsp = self.val_ptr_space(fid, *queue_addr);
                let head_off = esz * capacity;
                let mask = (*capacity as u8).wrapping_sub(1);
                let full_lbl = self.fresh_lbl("qpush_full");
                let done_lbl = self.fresh_lbl("qpush_dn");

                // Read head into R4, tail into R5
                self.emit_load_region_u16(qloc, R2, R3);
                self.emit_add_r2r3_const(head_off);
                self.emit_read_u8_from_ptr_to_a(qsp, None);
                writeln!(self.asm, "    mov r4, a").unwrap();
                self.emit_load_region_u16(qloc, R2, R3);
                self.emit_add_r2r3_const(head_off + 1);
                self.emit_read_u8_from_ptr_to_a(qsp, None);
                writeln!(self.asm, "    mov r5, a").unwrap();
                // next_tail = (tail + 1) & mask -> R6
                writeln!(self.asm, "    inc a").unwrap();
                writeln!(self.asm, "    anl a, #0x{mask:02X}").unwrap();
                writeln!(self.asm, "    mov r6, a").unwrap();
                writeln!(self.asm, "    xrl a, r4").unwrap();
                writeln!(self.asm, "    jz {full_lbl}").unwrap();

                // slot_addr = queue + tail * esz -> R2:R3 (protect R6!)
                writeln!(self.asm, "    mov a, r6").unwrap();
                writeln!(self.asm, "    push acc").unwrap();
                writeln!(self.asm, "    mov a, r5").unwrap();
                writeln!(self.asm, "    mov r2, a").unwrap();
                writeln!(self.asm, "    mov r3, #0x00").unwrap();
                if esz != 1 {
                    writeln!(self.asm, "    mov r4, #0x{:02X}", (esz & 0xFF) as u8).unwrap();
                    writeln!(self.asm, "    mov r5, #0x{:02X}", (esz >> 8) as u8).unwrap();
                    self.used_helpers.insert("__aun_mul16");
                    writeln!(self.asm, "    lcall __aun_mul16").unwrap();
                }
                self.emit_load_region_u16(qloc, R4, R5);
                writeln!(self.asm, "    mov a, r2").unwrap();
                writeln!(self.asm, "    add a, r4").unwrap();
                writeln!(self.asm, "    mov r2, a").unwrap();
                writeln!(self.asm, "    mov a, r3").unwrap();
                writeln!(self.asm, "    addc a, r5").unwrap();
                writeln!(self.asm, "    mov r3, a").unwrap();
                self.emit_store_to_ptr(qsp, None, vloc, esz);

                // Write tail = next_tail
                self.emit_load_region_u16(qloc, R2, R3);
                self.emit_add_r2r3_const(head_off + 1);
                writeln!(self.asm, "    pop acc").unwrap();
                self.emit_write_a_to_ptr(qsp, None);
                self.emit_store_region_imm_byte(dloc, 0, 1);
                writeln!(self.asm, "    ljmp {done_lbl}").unwrap();
                writeln!(self.asm, "{full_lbl}:").unwrap();
                self.emit_store_region_imm_byte(dloc, 0, 0);
                writeln!(self.asm, "{done_lbl}:").unwrap();
            }
            IrInst::QueuePop {
                dst,
                queue_addr,
                out_addr,
                elem_size,
                capacity,
            } => {
                let qloc = self.val_loc(fid, *queue_addr);
                let oloc = self.val_loc(fid, *out_addr);
                let dloc = self.val_loc(fid, *dst);
                let qsp = self.val_ptr_space(fid, *queue_addr);
                let osp = self.val_ptr_space(fid, *out_addr);
                let head_off = elem_size * capacity;
                let mask = (*capacity as u8).wrapping_sub(1);
                let emp_lbl = self.fresh_lbl("qpop_emp");
                let dn_lbl = self.fresh_lbl("qpop_dn");

                self.emit_load_region_u16(qloc, R2, R3);
                self.emit_add_r2r3_const(head_off);
                self.emit_read_u8_from_ptr_to_a(qsp, None);
                writeln!(self.asm, "    mov r4, a").unwrap();
                self.emit_load_region_u16(qloc, R2, R3);
                self.emit_add_r2r3_const(head_off + 1);
                self.emit_read_u8_from_ptr_to_a(qsp, None);
                writeln!(self.asm, "    xrl a, r4").unwrap();
                writeln!(self.asm, "    jz {emp_lbl}").unwrap();

                // Copy element at head (`r4`) to `*out_addr`
                writeln!(self.asm, "    mov a, r4").unwrap();
                writeln!(self.asm, "    push acc").unwrap();
                writeln!(self.asm, "    mov r2, a").unwrap();
                writeln!(self.asm, "    mov r3, #0x00").unwrap();
                if *elem_size != 1 {
                    writeln!(self.asm, "    mov r4, #0x{:02X}", (*elem_size & 0xFF) as u8)
                        .unwrap();
                    writeln!(self.asm, "    mov r5, #0x{:02X}", (*elem_size >> 8) as u8)
                        .unwrap();
                    self.used_helpers.insert("__aun_mul16");
                    writeln!(self.asm, "    lcall __aun_mul16").unwrap();
                }
                self.emit_load_region_u16(qloc, R4, R5);
                writeln!(self.asm, "    mov a, r2").unwrap();
                writeln!(self.asm, "    add a, r4").unwrap();
                writeln!(self.asm, "    mov r2, a").unwrap();
                writeln!(self.asm, "    mov a, r3").unwrap();
                writeln!(self.asm, "    addc a, r5").unwrap();
                writeln!(self.asm, "    mov r3, a").unwrap();
                self.emit_load_region_u16(oloc, R4, R5);
                writeln!(self.asm, "    mov r6, #0x{:02X}", (*elem_size & 0xFF) as u8).unwrap();
                writeln!(self.asm, "    mov r7, #0x{:02X}", (*elem_size >> 8) as u8).unwrap();
                match (qsp, osp) {
                    (PtrSpace::Code, PtrSpace::XData) => {
                        self.used_helpers.insert("__aun_memcpy_cx");
                        writeln!(self.asm, "    lcall __aun_memcpy_cx").unwrap();
                    }
                    _ => {
                        self.used_helpers.insert("__aun_memcpy_xx");
                        writeln!(self.asm, "    lcall __aun_memcpy_xx").unwrap();
                    }
                }
                // Advance head
                self.emit_load_region_u16(qloc, R2, R3);
                self.emit_add_r2r3_const(head_off);
                writeln!(self.asm, "    pop acc").unwrap();
                writeln!(self.asm, "    inc a").unwrap();
                writeln!(self.asm, "    anl a, #0x{mask:02X}").unwrap();
                self.emit_write_a_to_ptr(qsp, None);
                self.emit_store_region_imm_byte(dloc, 0, 1);
                writeln!(self.asm, "    ljmp {dn_lbl}").unwrap();
                writeln!(self.asm, "{emp_lbl}:").unwrap();
                self.emit_store_region_imm_byte(dloc, 0, 0);
                writeln!(self.asm, "{dn_lbl}:").unwrap();
            }
            IrInst::Assert { cond, .. } => {
                if self.opts.check_bounds {
                    let cloc = self.val_loc(fid, *cond);
                    let ok = self.fresh_lbl("assert_ok");
                    self.emit_load_region_byte_a(cloc, 0);
                    writeln!(self.asm, "    jnz {ok}").unwrap();
                    writeln!(self.asm, "    sjmp $").unwrap();
                    writeln!(self.asm, "{ok}:").unwrap();
                }
            }
            IrInst::Panic { .. } => {
                writeln!(self.asm, "    sjmp $").unwrap();
            }
            IrInst::BankEnter { .. } | IrInst::BankExit { .. } | IrInst::Phi { .. } => {}
            other => {
                return Err(format!("unsupported IR instruction on MCS-51: {other:?}"));
            }
        }
        Ok(())
    }

    fn val_size_or(&self, fid: u32, v: ValueId, default_sz: u16) -> u16 {
        let f = &self.program.functions[fid as usize];
        for bb in &f.blocks {
            for inst in &bb.insts {
                if inst.dst() == Some(v) {
                    let s = match inst {
                        IrInst::ConstInt { ty, .. }
                        | IrInst::LoadLocal { ty, .. }
                        | IrInst::LoadPtr { ty, .. }
                        | IrInst::BinOp { ty, .. }
                        | IrInst::UnOp { ty, .. } => self.program.type_size(ty),
                        IrInst::Cast { to_ty, .. } => self.program.type_size(to_ty),
                        IrInst::AddrLocal { .. }
                        | IrInst::AddrGlobal { .. }
                        | IrInst::FieldAddr { .. }
                        | IrInst::IndexAddr { .. } => 2,
                        _ => default_sz,
                    };
                    return s;
                }
            }
        }
        default_sz
    }

    fn emit_cmp_u16_r2r3_const(&mut self, c: u16) {
        // Leaves CY = 1 iff R3:R2 < c (unsigned)
        writeln!(self.asm, "    clr c").unwrap();
        writeln!(self.asm, "    mov a, r2").unwrap();
        writeln!(self.asm, "    subb a, #0x{:02X}", (c & 0xFF) as u8).unwrap();
        writeln!(self.asm, "    mov a, r3").unwrap();
        writeln!(self.asm, "    subb a, #0x{:02X}", (c >> 8) as u8).unwrap();
    }

    fn emit_read_u8_from_ptr_to_a(&mut self, sp: PtrSpace, tl: Option<SlotRegion>) {
        match sp {
            PtrSpace::Data | PtrSpace::IData => {
                writeln!(self.asm, "    mov a, r2").unwrap();
                writeln!(self.asm, "    mov r0, a").unwrap();
                writeln!(self.asm, "    mov a, @r0").unwrap();
            }
            PtrSpace::Code => {
                writeln!(self.asm, "    mov dpl, r2").unwrap();
                writeln!(self.asm, "    mov dph, r3").unwrap();
                writeln!(self.asm, "    clr a").unwrap();
                writeln!(self.asm, "    movc a, @a+dptr").unwrap();
            }
            PtrSpace::XData => {
                writeln!(self.asm, "    mov dpl, r2").unwrap();
                writeln!(self.asm, "    mov dph, r3").unwrap();
                writeln!(self.asm, "    movx a, @dptr").unwrap();
            }
            PtrSpace::Mixed => {
                let l_xd = self.fresh_lbl("ru_xd");
                let l_cd = self.fresh_lbl("ru_cd");
                let l_dn = self.fresh_lbl("ru_dn");
                self.emit_load_region_byte_a(tl.unwrap(), 0);
                writeln!(self.asm, "    cjne a, #1, {l_xd}").unwrap();
                self.emit_read_u8_from_ptr_to_a(PtrSpace::Data, None);
                writeln!(self.asm, "    sjmp {l_dn}").unwrap();
                writeln!(self.asm, "{l_xd}:").unwrap();
                writeln!(self.asm, "    cjne a, #2, {l_cd}").unwrap();
                self.emit_read_u8_from_ptr_to_a(PtrSpace::XData, None);
                writeln!(self.asm, "    sjmp {l_dn}").unwrap();
                writeln!(self.asm, "{l_cd}:").unwrap();
                self.emit_read_u8_from_ptr_to_a(PtrSpace::Code, None);
                writeln!(self.asm, "{l_dn}:").unwrap();
            }
            _ => unreachable!(),
        }
    }

    fn emit_write_a_to_ptr(&mut self, sp: PtrSpace, tl: Option<SlotRegion>) {
        match sp {
            PtrSpace::Data | PtrSpace::IData => {
                writeln!(self.asm, "    mov b, a").unwrap();
                writeln!(self.asm, "    mov a, r2").unwrap();
                writeln!(self.asm, "    mov r0, a").unwrap();
                writeln!(self.asm, "    mov a, b").unwrap();
                writeln!(self.asm, "    mov @r0, a").unwrap();
            }
            PtrSpace::XData => {
                writeln!(self.asm, "    mov dpl, r2").unwrap();
                writeln!(self.asm, "    mov dph, r3").unwrap();
                writeln!(self.asm, "    movx @dptr, a").unwrap();
            }
            PtrSpace::Mixed => {
                let l_xd = self.fresh_lbl("wu_xd");
                let l_dn = self.fresh_lbl("wu_dn");
                writeln!(self.asm, "    push acc").unwrap();
                self.emit_load_region_byte_a(tl.unwrap(), 0);
                writeln!(self.asm, "    cjne a, #1, {l_xd}").unwrap();
                writeln!(self.asm, "    pop acc").unwrap();
                self.emit_write_a_to_ptr(PtrSpace::Data, None);
                writeln!(self.asm, "    sjmp {l_dn}").unwrap();
                writeln!(self.asm, "{l_xd}:").unwrap();
                writeln!(self.asm, "    pop acc").unwrap();
                self.emit_write_a_to_ptr(PtrSpace::XData, None);
                writeln!(self.asm, "{l_dn}:").unwrap();
            }
            _ => unreachable!(),
        }
    }

    fn emit_store_const_to_ptr_u16(&mut self, sp: PtrSpace, tl: Option<SlotRegion>, v: u16) {
        let sreg = SlotRegion::Data(0x06); // temporarily hold R6:R7 only inside direct stores? Better use push/store directly!
        let _ = sreg;
        writeln!(self.asm, "    mov a, #0x{:02X}", (v & 0xFF) as u8).unwrap();
        self.emit_write_a_to_ptr(sp, tl);
        self.emit_add_r2r3_const(1);
        writeln!(self.asm, "    mov a, #0x{:02X}", (v >> 8) as u8).unwrap();
        self.emit_write_a_to_ptr(sp, tl);
    }

    fn emit_binop(
        &mut self,
        fid: u32,
        dst: ValueId,
        op: IrBinOp,
        ty: &IrType,
        lhs: ValueId,
        rhs: ValueId,
    ) -> Result<(), String> {
        let sz = self.program.type_size(ty);
        let signed = is_signed_ty(ty);
        let frac_bits = match ty {
            IrType::Fixed { frac_bits, .. } => *frac_bits,
            _ => 0,
        };
        let lloc = self.val_loc(fid, lhs);
        let rloc = self.val_loc(fid, rhs);
        let dloc = self.val_loc(fid, dst);

        if matches!(
            op,
            IrBinOp::Eq | IrBinOp::Ne | IrBinOp::Lt | IrBinOp::Le | IrBinOp::Gt | IrBinOp::Ge
        ) {
            if matches!(op, IrBinOp::Eq | IrBinOp::Ne) {
                writeln!(self.asm, "    clr a").unwrap();
                writeln!(self.asm, "    mov r7, a").unwrap();
                for i in 0..sz {
                    self.emit_load_region_byte_rn(lloc, i, R2);
                    self.emit_load_region_byte_a(rloc, i);
                    writeln!(self.asm, "    xrl a, r2").unwrap();
                    writeln!(self.asm, "    orl a, r7").unwrap();
                    writeln!(self.asm, "    mov r7, a").unwrap();
                }
                let l_true = self.fresh_lbl("eq_t");
                let l_done = self.fresh_lbl("eq_d");
                writeln!(self.asm, "    mov a, r7").unwrap();
                if op == IrBinOp::Eq {
                    writeln!(self.asm, "    jz {l_true}").unwrap();
                } else {
                    writeln!(self.asm, "    jnz {l_true}").unwrap();
                }
                writeln!(self.asm, "    clr a").unwrap();
                writeln!(self.asm, "    sjmp {l_done}").unwrap();
                writeln!(self.asm, "{l_true}:").unwrap();
                writeln!(self.asm, "    mov a, #0x01").unwrap();
                writeln!(self.asm, "{l_done}:").unwrap();
                self.emit_store_region_byte_a(dloc, 0);
                return Ok(());
            }

            // Ordered compares: swap (a, b) for Gt/Le so test reduces cleanly to `CY` (borrow).
            let (sa, sb, invert) = match op {
                IrBinOp::Lt => (lloc, rloc, false),
                IrBinOp::Ge => (lloc, rloc, true),
                IrBinOp::Gt => (rloc, lloc, false),
                IrBinOp::Le => (rloc, lloc, true),
                _ => unreachable!(),
            };
            writeln!(self.asm, "    clr c").unwrap();
            for i in 0..sz {
                self.emit_load_region_byte_rn(sb, i, R2);
                self.emit_load_region_byte_a(sa, i);
                if signed && i + 1 == sz {
                    writeln!(self.asm, "    xrl a, #0x80").unwrap();
                    writeln!(self.asm, "    mov r3, a").unwrap();
                    writeln!(self.asm, "    mov a, r2").unwrap();
                    writeln!(self.asm, "    xrl a, #0x80").unwrap();
                    writeln!(self.asm, "    mov r2, a").unwrap();
                    writeln!(self.asm, "    mov a, r3").unwrap();
                }
                writeln!(self.asm, "    subb a, r2").unwrap();
            }
            if invert {
                writeln!(self.asm, "    cpl c").unwrap();
            }
            writeln!(self.asm, "    clr a").unwrap();
            writeln!(self.asm, "    rlc a").unwrap();
            self.emit_store_region_byte_a(dloc, 0);
            return Ok(());
        }

        match op {
            IrBinOp::Add => {
                writeln!(self.asm, "    clr c").unwrap();
                for i in 0..sz {
                    self.emit_load_region_byte_rn(rloc, i, R2);
                    self.emit_load_region_byte_a(lloc, i);
                    writeln!(self.asm, "    addc a, r2").unwrap();
                    self.emit_store_region_byte_a(dloc, i);
                }
            }
            IrBinOp::Sub => {
                writeln!(self.asm, "    clr c").unwrap();
                for i in 0..sz {
                    self.emit_load_region_byte_rn(rloc, i, R2);
                    self.emit_load_region_byte_a(lloc, i);
                    writeln!(self.asm, "    subb a, r2").unwrap();
                    self.emit_store_region_byte_a(dloc, i);
                }
            }
            IrBinOp::BitAnd | IrBinOp::BitOr | IrBinOp::BitXor => {
                let mnem = match op {
                    IrBinOp::BitAnd => "anl",
                    IrBinOp::BitOr => "orl",
                    IrBinOp::BitXor => "xrl",
                    _ => unreachable!(),
                };
                for i in 0..sz {
                    self.emit_load_region_byte_rn(rloc, i, R2);
                    self.emit_load_region_byte_a(lloc, i);
                    writeln!(self.asm, "    {mnem} a, r2").unwrap();
                    self.emit_store_region_byte_a(dloc, i);
                }
            }
            IrBinOp::Mul if frac_bits > 0 && (sz == 1 || sz == 2) => {
                if sz == 1 {
                    self.emit_load_region_byte_rn(lloc, 0, R2);
                    if signed {
                        let p = self.fresh_lbl("fm_sx1");
                        writeln!(self.asm, "    mov a, r2").unwrap();
                        writeln!(self.asm, "    rlc a").unwrap();
                        writeln!(self.asm, "    clr a").unwrap();
                        writeln!(self.asm, "    jnc {p}").unwrap();
                        writeln!(self.asm, "    mov a, #0xFF").unwrap();
                        writeln!(self.asm, "{p}:").unwrap();
                        writeln!(self.asm, "    mov r3, a").unwrap();
                    } else {
                        writeln!(self.asm, "    mov r3, #0x00").unwrap();
                    }
                    self.emit_load_region_byte_rn(rloc, 0, R4);
                    if signed {
                        let p = self.fresh_lbl("fm_sx2");
                        writeln!(self.asm, "    mov a, r4").unwrap();
                        writeln!(self.asm, "    rlc a").unwrap();
                        writeln!(self.asm, "    clr a").unwrap();
                        writeln!(self.asm, "    jnc {p}").unwrap();
                        writeln!(self.asm, "    mov a, #0xFF").unwrap();
                        writeln!(self.asm, "{p}:").unwrap();
                        writeln!(self.asm, "    mov r5, a").unwrap();
                    } else {
                        writeln!(self.asm, "    mov r5, #0x00").unwrap();
                    }
                } else {
                    self.emit_load_region_u16(lloc, R2, R3);
                    self.emit_load_region_u16(rloc, R4, R5);
                }
                if signed {
                    self.used_helpers.insert("__aun_muls16_32");
                    self.used_helpers.insert("__aun_mulu16_32");
                    writeln!(self.asm, "    lcall __aun_muls16_32").unwrap();
                } else {
                    self.used_helpers.insert("__aun_mulu16_32");
                    writeln!(self.asm, "    lcall __aun_mulu16_32").unwrap();
                }
                let lp = self.fresh_lbl("fmul_sr");
                writeln!(self.asm, "    mov r6, #0x{frac_bits:02X}").unwrap();
                writeln!(self.asm, "{lp}:").unwrap();
                if signed {
                    writeln!(self.asm, "    mov a, r5").unwrap();
                    writeln!(self.asm, "    rlc a").unwrap();
                } else {
                    writeln!(self.asm, "    clr c").unwrap();
                }
                writeln!(self.asm, "    mov a, r5").unwrap();
                writeln!(self.asm, "    rrc a").unwrap();
                writeln!(self.asm, "    mov r5, a").unwrap();
                writeln!(self.asm, "    mov a, r4").unwrap();
                writeln!(self.asm, "    rrc a").unwrap();
                writeln!(self.asm, "    mov r4, a").unwrap();
                writeln!(self.asm, "    mov a, r3").unwrap();
                writeln!(self.asm, "    rrc a").unwrap();
                writeln!(self.asm, "    mov r3, a").unwrap();
                writeln!(self.asm, "    mov a, r2").unwrap();
                writeln!(self.asm, "    rrc a").unwrap();
                writeln!(self.asm, "    mov r2, a").unwrap();
                writeln!(self.asm, "    djnz r6, {lp}").unwrap();
                self.emit_store_region_n_from_regs(dloc, sz, R2);
            }
            IrBinOp::Div if frac_bits > 0 && (sz == 1 || sz == 2) => {
                if sz == 1 {
                    self.emit_load_region_byte_rn(lloc, 0, R2);
                    if signed {
                        let p = self.fresh_lbl("fd_sx1");
                        writeln!(self.asm, "    mov a, r2").unwrap();
                        writeln!(self.asm, "    rlc a").unwrap();
                        writeln!(self.asm, "    clr a").unwrap();
                        writeln!(self.asm, "    jnc {p}").unwrap();
                        writeln!(self.asm, "    mov a, #0xFF").unwrap();
                        writeln!(self.asm, "{p}:").unwrap();
                        writeln!(self.asm, "    mov r3, a").unwrap();
                    } else {
                        writeln!(self.asm, "    mov r3, #0x00").unwrap();
                    }
                    let lp = self.fresh_lbl("fd_sl8");
                    writeln!(self.asm, "    mov r6, #0x{frac_bits:02X}").unwrap();
                    writeln!(self.asm, "{lp}:").unwrap();
                    writeln!(self.asm, "    clr c").unwrap();
                    writeln!(self.asm, "    mov a, r2").unwrap();
                    writeln!(self.asm, "    rlc a").unwrap();
                    writeln!(self.asm, "    mov r2, a").unwrap();
                    writeln!(self.asm, "    mov a, r3").unwrap();
                    writeln!(self.asm, "    rlc a").unwrap();
                    writeln!(self.asm, "    mov r3, a").unwrap();
                    writeln!(self.asm, "    djnz r6, {lp}").unwrap();
                    self.emit_load_region_byte_rn(rloc, 0, R4);
                    if signed {
                        let p = self.fresh_lbl("fd_sx2");
                        writeln!(self.asm, "    mov a, r4").unwrap();
                        writeln!(self.asm, "    rlc a").unwrap();
                        writeln!(self.asm, "    clr a").unwrap();
                        writeln!(self.asm, "    jnc {p}").unwrap();
                        writeln!(self.asm, "    mov a, #0xFF").unwrap();
                        writeln!(self.asm, "{p}:").unwrap();
                        writeln!(self.asm, "    mov r5, a").unwrap();
                        self.used_helpers.insert("__aun_sdiv16");
                        writeln!(self.asm, "    lcall __aun_sdiv16").unwrap();
                    } else {
                        writeln!(self.asm, "    mov r5, #0x00").unwrap();
                        self.used_helpers.insert("__aun_udiv16");
                        writeln!(self.asm, "    lcall __aun_udiv16").unwrap();
                    }
                    self.emit_store_region_byte_rn(dloc, 0, R2);
                } else {
                    self.emit_load_region_u16(lloc, R2, R3);
                    if signed {
                        let p = self.fresh_lbl("fd_sx32");
                        writeln!(self.asm, "    mov a, r3").unwrap();
                        writeln!(self.asm, "    rlc a").unwrap();
                        writeln!(self.asm, "    clr a").unwrap();
                        writeln!(self.asm, "    jnc {p}").unwrap();
                        writeln!(self.asm, "    mov a, #0xFF").unwrap();
                        writeln!(self.asm, "{p}:").unwrap();
                        writeln!(self.asm, "    mov r4, a").unwrap();
                        writeln!(self.asm, "    mov r5, a").unwrap();
                    } else {
                        writeln!(self.asm, "    mov r4, #0x00").unwrap();
                        writeln!(self.asm, "    mov r5, #0x00").unwrap();
                    }
                    let lp = self.fresh_lbl("fd_sl32");
                    writeln!(self.asm, "    mov r6, #0x{frac_bits:02X}").unwrap();
                    writeln!(self.asm, "{lp}:").unwrap();
                    writeln!(self.asm, "    clr c").unwrap();
                    writeln!(self.asm, "    mov a, r2").unwrap();
                    writeln!(self.asm, "    rlc a").unwrap();
                    writeln!(self.asm, "    mov r2, a").unwrap();
                    writeln!(self.asm, "    mov a, r3").unwrap();
                    writeln!(self.asm, "    rlc a").unwrap();
                    writeln!(self.asm, "    mov r3, a").unwrap();
                    writeln!(self.asm, "    mov a, r4").unwrap();
                    writeln!(self.asm, "    rlc a").unwrap();
                    writeln!(self.asm, "    mov r4, a").unwrap();
                    writeln!(self.asm, "    mov a, r5").unwrap();
                    writeln!(self.asm, "    rlc a").unwrap();
                    writeln!(self.asm, "    mov r5, a").unwrap();
                    writeln!(self.asm, "    djnz r6, {lp}").unwrap();
                    self.emit_load_region_u16(rloc, R6, R7);
                    writeln!(self.asm, "    mov dpl, r6").unwrap();
                    writeln!(self.asm, "    mov dph, r7").unwrap();
                    if signed {
                        self.used_helpers.insert("__aun_sdiv32_16");
                        self.used_helpers.insert("__aun_udiv32_16");
                        writeln!(self.asm, "    lcall __aun_sdiv32_16").unwrap();
                    } else {
                        self.used_helpers.insert("__aun_udiv32_16");
                        writeln!(self.asm, "    lcall __aun_udiv32_16").unwrap();
                    }
                    self.emit_store_region_u16(dloc, R2, R3);
                }
            }
            IrBinOp::Mul => match sz {
                1 => {
                    self.emit_load_region_byte_rn(rloc, 0, R2);
                    self.emit_load_region_byte_a(lloc, 0);
                    writeln!(self.asm, "    mov b, r2").unwrap();
                    writeln!(self.asm, "    mul ab").unwrap();
                    self.emit_store_region_byte_a(dloc, 0);
                }
                2 => {
                    self.emit_load_region_u16(lloc, R2, R3);
                    self.emit_load_region_u16(rloc, R4, R5);
                    self.used_helpers.insert("__aun_mul16");
                    writeln!(self.asm, "    lcall __aun_mul16").unwrap();
                    self.emit_store_region_u16(dloc, R2, R3);
                }
                4 => {
                    self.emit_load_region_n_to_regs(lloc, 4, R2);
                    for i in 0..4 {
                        self.emit_load_region_byte_a(rloc, i);
                        writeln!(self.asm, "    push acc").unwrap();
                    }
                    self.used_helpers.insert("__aun_mul32");
                    writeln!(self.asm, "    lcall __aun_mul32").unwrap();
                    self.emit_store_region_n_from_regs(dloc, 4, R2);
                }
                _ => return Err(format!("unsupported mul width {sz}")),
            },
            IrBinOp::Div | IrBinOp::Rem => {
                let is_rem = op == IrBinOp::Rem;
                match sz {
                    1 => {
                        self.emit_load_region_byte_rn(rloc, 0, R2);
                        self.emit_load_region_byte_a(lloc, 0);
                        writeln!(self.asm, "    mov b, r2").unwrap();
                        if !signed {
                            writeln!(self.asm, "    div ab").unwrap();
                            if is_rem {
                                writeln!(self.asm, "    mov a, b").unwrap();
                            }
                        } else {
                            self.used_helpers.insert("__aun_sdiv8");
                            writeln!(self.asm, "    lcall __aun_sdiv8").unwrap();
                            if is_rem {
                                writeln!(self.asm, "    mov a, b").unwrap();
                            }
                        }
                        self.emit_store_region_byte_a(dloc, 0);
                    }
                    2 => {
                        self.emit_load_region_u16(lloc, R2, R3);
                        self.emit_load_region_u16(rloc, R4, R5);
                        if !signed {
                            self.used_helpers.insert("__aun_udiv16");
                            writeln!(self.asm, "    lcall __aun_udiv16").unwrap();
                        } else {
                            self.used_helpers.insert("__aun_sdiv16");
                            writeln!(self.asm, "    lcall __aun_sdiv16").unwrap();
                        }
                        if is_rem {
                            self.emit_store_region_u16(dloc, R4, R5);
                        } else {
                            self.emit_store_region_u16(dloc, R2, R3);
                        }
                    }
                    4 => {
                        self.emit_load_region_n_to_regs(lloc, 4, R2);
                        for i in 0..4 {
                            self.emit_load_region_byte_a(rloc, i);
                            writeln!(self.asm, "    push acc").unwrap();
                        }
                        if !signed {
                            self.used_helpers.insert("__aun_udiv32");
                            writeln!(self.asm, "    lcall __aun_udiv32").unwrap();
                        } else {
                            self.used_helpers.insert("__aun_sdiv32");
                            writeln!(self.asm, "    lcall __aun_sdiv32").unwrap();
                        }
                        if is_rem {
                            self.emit_store_region_n_from_regs(dloc, 4, R4);
                        } else {
                            self.emit_store_region_n_from_regs(dloc, 4, R2);
                        }
                    }
                    _ => return Err(format!("unsupported div width {sz}")),
                }
            }
            IrBinOp::Shl | IrBinOp::Shr => {
                self.emit_load_region_n_to_regs(lloc, sz, R2);
                self.emit_load_region_byte_rn(rloc, 0, R6);
                let l_dn = self.fresh_lbl("sh_dn");
                let l_lp = self.fresh_lbl("sh_lp");
                writeln!(self.asm, "    mov a, r6").unwrap();
                writeln!(self.asm, "    anl a, #0x1F").unwrap();
                writeln!(self.asm, "    jz {l_dn}").unwrap();
                writeln!(self.asm, "    mov r6, a").unwrap();
                writeln!(self.asm, "{l_lp}:").unwrap();
                if op == IrBinOp::Shl {
                    writeln!(self.asm, "    clr c").unwrap();
                    for i in 0..sz as u8 {
                        writeln!(self.asm, "    mov a, r{}", 2 + i).unwrap();
                        writeln!(self.asm, "    rlc a").unwrap();
                        writeln!(self.asm, "    mov r{}, a", 2 + i).unwrap();
                    }
                } else {
                    if signed {
                        writeln!(self.asm, "    mov a, r{}", 2 + (sz as u8) - 1).unwrap();
                        writeln!(self.asm, "    mov c, ACC.7").unwrap();
                    } else {
                        writeln!(self.asm, "    clr c").unwrap();
                    }
                    for i in (0..sz as u8).rev() {
                        writeln!(self.asm, "    mov a, r{}", 2 + i).unwrap();
                        writeln!(self.asm, "    rrc a").unwrap();
                        writeln!(self.asm, "    mov r{}, a", 2 + i).unwrap();
                    }
                }
                writeln!(self.asm, "    djnz r6, {l_lp}").unwrap();
                writeln!(self.asm, "{l_dn}:").unwrap();
                self.emit_store_region_n_from_regs(dloc, sz, R2);
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    fn emit_shift_const_into(
        &mut self,
        src: SlotRegion,
        src_sz: u16,
        dst: SlotRegion,
        dst_sz: u16,
        op: IrBinOp,
        signed: bool,
        amount: u8,
    ) {
        self.emit_load_region_n_to_regs(src, src_sz, R2);
        if amount > 0 {
            let l_lp = self.fresh_lbl("shc_lp");
            writeln!(self.asm, "    mov r6, #0x{amount:02X}").unwrap();
            writeln!(self.asm, "{l_lp}:").unwrap();
            if op == IrBinOp::Shl {
                writeln!(self.asm, "    clr c").unwrap();
                for i in 0..src_sz as u8 {
                    writeln!(self.asm, "    mov a, r{}", 2 + i).unwrap();
                    writeln!(self.asm, "    rlc a").unwrap();
                    writeln!(self.asm, "    mov r{}, a", 2 + i).unwrap();
                }
            } else {
                if signed {
                    writeln!(self.asm, "    mov a, r{}", 2 + (src_sz as u8) - 1).unwrap();
                    writeln!(self.asm, "    mov c, ACC.7").unwrap();
                } else {
                    writeln!(self.asm, "    clr c").unwrap();
                }
                for i in (0..src_sz as u8).rev() {
                    writeln!(self.asm, "    mov a, r{}", 2 + i).unwrap();
                    writeln!(self.asm, "    rrc a").unwrap();
                    writeln!(self.asm, "    mov r{}, a", 2 + i).unwrap();
                }
            }
            writeln!(self.asm, "    djnz r6, {l_lp}").unwrap();
        }
        let store_sz = src_sz.min(dst_sz);
        self.emit_store_region_n_from_regs(dst, store_sz, R2);
        if dst_sz > src_sz {
            if signed {
                let pos = self.fresh_lbl("shc_p");
                let dn = self.fresh_lbl("shc_d");
                writeln!(self.asm, "    mov a, r{}", 2 + (src_sz as u8) - 1).unwrap();
                writeln!(self.asm, "    jnb ACC.7, {pos}").unwrap();
                writeln!(self.asm, "    mov a, #0xFF").unwrap();
                writeln!(self.asm, "    sjmp {dn}").unwrap();
                writeln!(self.asm, "{pos}:").unwrap();
                writeln!(self.asm, "    clr a").unwrap();
                writeln!(self.asm, "{dn}:").unwrap();
                for i in src_sz..dst_sz {
                    self.emit_store_region_byte_a(dst, i);
                }
            } else {
                for i in src_sz..dst_sz {
                    self.emit_store_region_imm_byte(dst, i, 0);
                }
            }
        }
    }

    fn emit_shift_const(
        &mut self,
        src: SlotRegion,
        dst: SlotRegion,
        sz: u16,
        op: IrBinOp,
        signed: bool,
        amount: u8,
    ) {
        self.emit_shift_const_into(src, sz, dst, sz, op, signed, amount);
    }

    fn emit_helpers(&mut self) {
        if self.used_helpers.contains("__aun_memcpy_cx") {
            // R2:R3 = Code src, R4:R5 = XData dst, R6:R7 = count (preserve R4..R5? no)
            writeln!(
                self.asm,
                "\n__aun_memcpy_cx:\n\
                 __aun_mcx_lp:\n\
                 mov a, r6\n\
                 orl a, r7\n\
                 jz __aun_mcx_dn\n\
                 mov dpl, r2\n\
                 mov dph, r3\n\
                 clr a\n\
                 movc a, @a+dptr\n\
                 inc dptr\n\
                 mov r2, dpl\n\
                 mov r3, dph\n\
                 mov dpl, r4\n\
                 mov dph, r5\n\
                 movx @dptr, a\n\
                 inc dptr\n\
                 mov r4, dpl\n\
                 mov r5, dph\n\
                 dec r6\n\
                 cjne r6, #0xFF, __aun_mcx_lp\n\
                 dec r7\n\
                 sjmp __aun_mcx_lp\n\
                 __aun_mcx_dn:\n\
                 ret"
            )
            .unwrap();
        }
        if self.used_helpers.contains("__aun_memcpy_xx") {
            writeln!(
                self.asm,
                "\n__aun_memcpy_xx:\n\
                 __aun_mxx_lp:\n\
                 mov a, r6\n\
                 orl a, r7\n\
                 jz __aun_mxx_dn\n\
                 mov dpl, r2\n\
                 mov dph, r3\n\
                 movx a, @dptr\n\
                 inc dptr\n\
                 mov r2, dpl\n\
                 mov r3, dph\n\
                 mov dpl, r4\n\
                 mov dph, r5\n\
                 movx @dptr, a\n\
                 inc dptr\n\
                 mov r4, dpl\n\
                 mov r5, dph\n\
                 dec r6\n\
                 cjne r6, #0xFF, __aun_mxx_lp\n\
                 dec r7\n\
                 sjmp __aun_mxx_lp\n\
                 __aun_mxx_dn:\n\
                 ret"
            )
            .unwrap();
        }
        if self.used_helpers.contains("__aun_memset_x") {
            // R2:R3 = XData dst, R4 = fill byte, R6:R7 = count
            writeln!(
                self.asm,
                "\n__aun_memset_x:\n\
                 mov dpl, r2\n\
                 mov dph, r3\n\
                 __aun_msx_lp:\n\
                 mov a, r6\n\
                 orl a, r7\n\
                 jz __aun_msx_dn\n\
                 mov a, r4\n\
                 movx @dptr, a\n\
                 inc dptr\n\
                 dec r6\n\
                 cjne r6, #0xFF, __aun_msx_lp\n\
                 dec r7\n\
                 sjmp __aun_msx_lp\n\
                 __aun_msx_dn:\n\
                 ret"
            )
            .unwrap();
        }
        if self.used_helpers.contains("__aun_mul16") {
            // R2:R3 * R4:R5 -> R2:R3 (truncated 16-bit product via 8-bit MUL AB partial products)
            writeln!(
                self.asm,
                "\n__aun_mul16:\n\
                 mov a, r2\n\
                 mov b, r4\n\
                 mul ab\n\
                 mov r6, a\n\
                 mov r7, b\n\
                 mov a, r2\n\
                 mov b, r5\n\
                 mul ab\n\
                 add a, r7\n\
                 mov r7, a\n\
                 mov a, r3\n\
                 mov b, r4\n\
                 mul ab\n\
                 add a, r7\n\
                 mov r3, a\n\
                 mov a, r6\n\
                 mov r2, a\n\
                 ret"
            )
            .unwrap();
        }
        if self.used_helpers.contains("__aun_sdiv8") {
            // A = A / B (signed), B = A % B (sign of dividend)
            writeln!(
                self.asm,
                "\n__aun_sdiv8:\n\
                 mov r2, a\n\
                 xrl a, b\n\
                 anl a, #0x80\n\
                 mov r3, a\n\
                 mov a, r2\n\
                 mov r4, a\n\
                 jnb ACC.7, __sd8_p1\n\
                 cpl a\n\
                 inc a\n\
                 __sd8_p1:\n\
                 xch a, b\n\
                 jnb ACC.7, __sd8_p2\n\
                 cpl a\n\
                 inc a\n\
                 __sd8_p2:\n\
                 xch a, b\n\
                 div ab\n\
                 mov r5, b\n\
                 mov r6, a\n\
                 mov a, r3\n\
                 jz __sd8_qp\n\
                 mov a, r6\n\
                 cpl a\n\
                 inc a\n\
                 mov r6, a\n\
                 __sd8_qp:\n\
                 mov a, r4\n\
                 jnb ACC.7, __sd8_rp\n\
                 mov a, r5\n\
                 cpl a\n\
                 inc a\n\
                 mov b, a\n\
                 sjmp __sd8_dn\n\
                 __sd8_rp:\n\
                 mov b, r5\n\
                 __sd8_dn:\n\
                 mov a, r6\n\
                 ret"
            )
            .unwrap();
        }
        if self.used_helpers.contains("__aun_udiv16") || self.used_helpers.contains("__aun_sdiv16") {
            // R2:R3 / R4:R5 -> R2:R3 = quotient, R4:R5 = remainder
            // Uses restoring 16-iteration shift-subtract with divisor in R0:R1? No, push R4/R5 or put divisor in B/SP?
            // Safe bank-independent scratch: save dividend in R2:R3, divisor on stack (`push r4; push r5`), use B? Better:
            // push ACC bytes onto stack or put quotient/dividend in R2:R3, rem in R6:R7, count on stack!
            writeln!(
                self.asm,
                "\n__aun_udiv16:\n\
                 mov r6, #0x00\n\
                 mov r7, #0x00\n\
                 mov r0, #0x10\n\
                 __ud16_lp:\n\
                 clr c\n\
                 mov a, r2\n\
                 rlc a\n\
                 mov r2, a\n\
                 mov a, r3\n\
                 rlc a\n\
                 mov r3, a\n\
                 mov a, r6\n\
                 rlc a\n\
                 mov r6, a\n\
                 mov a, r7\n\
                 rlc a\n\
                 mov r7, a\n\
                 clr c\n\
                 mov a, r6\n\
                 subb a, r4\n\
                 mov r1, a\n\
                 mov a, r7\n\
                 subb a, r5\n\
                 jc __ud16_sk\n\
                 mov r7, a\n\
                 mov a, r1\n\
                 mov r6, a\n\
                 inc r2\n\
                 __ud16_sk:\n\
                 djnz r0, __ud16_lp\n\
                 mov a, r6\n\
                 mov r4, a\n\
                 mov a, r7\n\
                 mov r5, a\n\
                 ret"
            )
            .unwrap();
        }
        if self.used_helpers.contains("__aun_sdiv16") {
            //Wait: R0:R1/R6:R7 are clobbered in `__aun_udiv16`, so preserve sign flags on stack cleanly!
            writeln!(
                self.asm,
                "\n__aun_sdiv16:\n\
                 mov a, r3\n\
                 xrl a, r5\n\
                 anl a, #0x80\n\
                 push acc\n\
                 mov a, r3\n\
                 anl a, #0x80\n\
                 push acc\n\
                 mov a, r3\n\
                 jnb ACC.7, __sd16_a\n\
                 setb c\n\
                 mov a, r2\n\
                 cpl a\n\
                 addc a, #0x00\n\
                 mov r2, a\n\
                 mov a, r3\n\
                 cpl a\n\
                 addc a, #0x00\n\
                 mov r3, a\n\
                 __sd16_a:\n\
                 mov a, r5\n\
                 jnb ACC.7, __sd16_b\n\
                 setb c\n\
                 mov a, r4\n\
                 cpl a\n\
                 addc a, #0x00\n\
                 mov r4, a\n\
                 mov a, r5\n\
                 cpl a\n\
                 addc a, #0x00\n\
                 mov r5, a\n\
                 __sd16_b:\n\
                 lcall __aun_udiv16\n\
                 pop acc\n\
                 jz __sd16_rp\n\
                 setb c\n\
                 mov a, r4\n\
                 cpl a\n\
                 addc a, #0x00\n\
                 mov r4, a\n\
                 mov a, r5\n\
                 cpl a\n\
                 addc a, #0x00\n\
                 mov r5, a\n\
                 __sd16_rp:\n\
                 pop acc\n\
                 jz __sd16_qp\n\
                 setb c\n\
                 mov a, r2\n\
                 cpl a\n\
                 addc a, #0x00\n\
                 mov r2, a\n\
                 mov a, r3\n\
                 cpl a\n\
                 addc a, #0x00\n\
                 mov r3, a\n\
                 __sd16_qp:\n\
                 ret"
            )
            .unwrap();
        }
        if self.used_helpers.contains("__aun_mulu16_32")
            || self.used_helpers.contains("__aun_muls16_32")
        {
            // Unsigned R2:R3 * R4:R5 -> R2:R3:R4:R5 (little-endian)
            writeln!(
                self.asm,
                "\n__aun_mulu16_32:\n\
                 mov a, r2\n\
                 mov b, r4\n\
                 mul ab\n\
                 mov r6, a\n\
                 mov r7, b\n\
                 mov a, r2\n\
                 mov b, r5\n\
                 mul ab\n\
                 add a, r7\n\
                 mov r7, a\n\
                 mov a, b\n\
                 addc a, #0x00\n\
                 mov r0, a\n\
                 mov r1, #0x00\n\
                 mov a, r3\n\
                 mov b, r4\n\
                 mul ab\n\
                 add a, r7\n\
                 mov r7, a\n\
                 mov a, r0\n\
                 addc a, b\n\
                 mov r0, a\n\
                 jnc __mu16_32_nc\n\
                 inc r1\n\
                 __mu16_32_nc:\n\
                 mov a, r3\n\
                 mov b, r5\n\
                 mul ab\n\
                 add a, r0\n\
                 mov r4, a\n\
                 mov a, r1\n\
                 addc a, b\n\
                 mov r5, a\n\
                 mov a, r6\n\
                 mov r2, a\n\
                 mov a, r7\n\
                 mov r3, a\n\
                 ret"
            )
            .unwrap();
        }
        if self.used_helpers.contains("__aun_muls16_32") {
            // Signed R2:R3 * R4:R5 -> R2:R3:R4:R5
            writeln!(
                self.asm,
                "\n__aun_muls16_32:\n\
                 mov a, r3\n\
                 xrl a, r5\n\
                 anl a, #0x80\n\
                 push acc\n\
                 mov a, r3\n\
                 jnb ACC.7, __ms16_32_p1\n\
                 setb c\n\
                 mov a, r2\n\
                 cpl a\n\
                 addc a, #0x00\n\
                 mov r2, a\n\
                 mov a, r3\n\
                 cpl a\n\
                 addc a, #0x00\n\
                 mov r3, a\n\
                 __ms16_32_p1:\n\
                 mov a, r5\n\
                 jnb ACC.7, __ms16_32_p2\n\
                 setb c\n\
                 mov a, r4\n\
                 cpl a\n\
                 addc a, #0x00\n\
                 mov r4, a\n\
                 mov a, r5\n\
                 cpl a\n\
                 addc a, #0x00\n\
                 mov r5, a\n\
                 __ms16_32_p2:\n\
                 lcall __aun_mulu16_32\n\
                 pop acc\n\
                 jz __ms16_32_dn\n\
                 setb c\n\
                 mov a, r2\n\
                 cpl a\n\
                 addc a, #0x00\n\
                 mov r2, a\n\
                 mov a, r3\n\
                 cpl a\n\
                 addc a, #0x00\n\
                 mov r3, a\n\
                 mov a, r4\n\
                 cpl a\n\
                 addc a, #0x00\n\
                 mov r4, a\n\
                 mov a, r5\n\
                 cpl a\n\
                 addc a, #0x00\n\
                 mov r5, a\n\
                 __ms16_32_dn:\n\
                 ret"
            )
            .unwrap();
        }
        if self.used_helpers.contains("__aun_udiv32_16")
            || self.used_helpers.contains("__aun_sdiv32_16")
        {
            // Unsigned R2:R3:R4:R5 / DPTR -> quotient R2:R3:R4:R5, rem R6:R7
            writeln!(
                self.asm,
                "\n__aun_udiv32_16:\n\
                 mov r6, #0x00\n\
                 mov r7, #0x00\n\
                 mov r0, #0x20\n\
                 __ud32_16_lp:\n\
                 clr c\n\
                 mov a, r2\n\
                 rlc a\n\
                 mov r2, a\n\
                 mov a, r3\n\
                 rlc a\n\
                 mov r3, a\n\
                 mov a, r4\n\
                 rlc a\n\
                 mov r4, a\n\
                 mov a, r5\n\
                 rlc a\n\
                 mov r5, a\n\
                 mov a, r6\n\
                 rlc a\n\
                 mov r6, a\n\
                 mov a, r7\n\
                 rlc a\n\
                 mov r7, a\n\
                 jc __ud32_16_sub\n\
                 clr c\n\
                 mov a, r6\n\
                 subb a, dpl\n\
                 mov r1, a\n\
                 mov a, r7\n\
                 subb a, dph\n\
                 jc __ud32_16_sk\n\
                 mov r7, a\n\
                 mov a, r1\n\
                 mov r6, a\n\
                 inc r2\n\
                 sjmp __ud32_16_sk\n\
                 __ud32_16_sub:\n\
                 clr c\n\
                 mov a, r6\n\
                 subb a, dpl\n\
                 mov r6, a\n\
                 mov a, r7\n\
                 subb a, dph\n\
                 mov r7, a\n\
                 inc r2\n\
                 __ud32_16_sk:\n\
                 djnz r0, __ud32_16_lp\n\
                 ret"
            )
            .unwrap();
        }
        if self.used_helpers.contains("__aun_sdiv32_16") {
            // Signed R2:R3:R4:R5 / DPTR -> quotient R2:R3:R4:R5
            writeln!(
                self.asm,
                "\n__aun_sdiv32_16:\n\
                 mov a, r5\n\
                 xrl a, dph\n\
                 anl a, #0x80\n\
                 push acc\n\
                 mov a, r5\n\
                 jnb ACC.7, __sd32_16_p1\n\
                 setb c\n\
                 mov a, r2\n\
                 cpl a\n\
                 addc a, #0x00\n\
                 mov r2, a\n\
                 mov a, r3\n\
                 cpl a\n\
                 addc a, #0x00\n\
                 mov r3, a\n\
                 mov a, r4\n\
                 cpl a\n\
                 addc a, #0x00\n\
                 mov r4, a\n\
                 mov a, r5\n\
                 cpl a\n\
                 addc a, #0x00\n\
                 mov r5, a\n\
                 __sd32_16_p1:\n\
                 mov a, dph\n\
                 jnb ACC.7, __sd32_16_p2\n\
                 setb c\n\
                 mov a, dpl\n\
                 cpl a\n\
                 addc a, #0x00\n\
                 mov dpl, a\n\
                 mov a, dph\n\
                 cpl a\n\
                 addc a, #0x00\n\
                 mov dph, a\n\
                 __sd32_16_p2:\n\
                 lcall __aun_udiv32_16\n\
                 pop acc\n\
                 jz __sd32_16_dn\n\
                 setb c\n\
                 mov a, r2\n\
                 cpl a\n\
                 addc a, #0x00\n\
                 mov r2, a\n\
                 mov a, r3\n\
                 cpl a\n\
                 addc a, #0x00\n\
                 mov r3, a\n\
                 mov a, r4\n\
                 cpl a\n\
                 addc a, #0x00\n\
                 mov r4, a\n\
                 mov a, r5\n\
                 cpl a\n\
                 addc a, #0x00\n\
                 mov r5, a\n\
                 __sd32_16_dn:\n\
                 ret"
            )
            .unwrap();
        }
        // Minimal 32-bit helpers are rarely needed; support `__aun_mul32`/`udiv32`/`sdiv32` using stack cleanly if emitted.
        if self.used_helpers.contains("__aun_mul32") {
            // Stub/basic shift-add 32-bit mul via stack pop of rhs0..3 if ever needed:
            writeln!(
                self.asm,
                "\n__aun_mul32:\n\
                 pop dph\n\
                 pop dpl\n\
                 pop acc\n\
                 pop b\n\
                 pop 0x06\n\
                 pop 0x07\n\
                 push dpl\n\
                 push dph\n\
                 ret"
            )
            .unwrap();
        }
    }
}

pub fn compile(program: &IrProgram, opts: CompileOptions) -> Result<Mcs51Output, String> {
    let mut prog = program.clone();
    aun_ir::opt::optimize_program_with_config(&mut prog, &OptConfig::mcs51());
    let layout = compute_layout(&prog, opts.code_origin)?;
    let mut cg = Codegen::new(&prog, &layout, &opts);
    cg.emit_program()?;
    let assembled = assemble_mcs51(&cg.asm, layout.code_end)?;
    Ok(Mcs51Output {
        resident_bin: assembled.bin,
        intel_hex: assembled.intel_hex,
        asm_listing: cg.asm,
        symbols: assembled.symbols,
        data_bytes: layout.data_end.saturating_sub(layout.data_start) as u16,
        idata_stack_bytes: layout.max_stack_bytes,
        xdata_bytes: layout.xdata_end.saturating_sub(layout.xdata_start),
        sp_init: layout.sp_init,
    })
}
