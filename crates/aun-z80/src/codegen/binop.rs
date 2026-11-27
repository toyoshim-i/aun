use super::runtime::emit_mul_hl_const;
use super::EmitCtx;
use crate::analysis::is_signed_ty;
use crate::regalloc::RegLoc;
use aun_ir::*;
use std::fmt::Write;

fn invert_cc(cc: &'static str) -> &'static str {
    match cc {
        "z" => "nz",
        "nz" => "z",
        "c" => "nc",
        "nc" => "c",
        _ => unreachable!(),
    }
}

impl<'a> EmitCtx<'a> {
    fn emit_bool_from_cc(&mut self, cc: &str, da: u16) {
        match cc {
            "c" => {
                writeln!(self.asm, "    ld a, 0").unwrap();
                writeln!(self.asm, "    rla").unwrap();
            }
            "nc" => {
                writeln!(self.asm, "    ccf").unwrap();
                writeln!(self.asm, "    ld a, 0").unwrap();
                writeln!(self.asm, "    rla").unwrap();
            }
            _ => {
                let done = self.fresh_label("cmp");
                writeln!(self.asm, "    ld a, 1").unwrap();
                writeln!(self.asm, "    jr {cc}, {done}").unwrap();
                writeln!(self.asm, "    xor a").unwrap();
                writeln!(self.asm, "{done}:").unwrap();
            }
        }
        let da_mem = self.fmt_ram_mem(da);
        writeln!(self.asm, "    ld {da_mem}, a").unwrap();
    }

    fn emit_load_word32(&mut self, v: ValueId, hi: bool, reg: &str, flip_msb: bool) {
        if let Some(&c) = self.const_vals.get(&v) {
            let u = c as u32;
            let mut w = if hi { (u >> 16) as u16 } else { u as u16 };
            if hi && flip_msb {
                w ^= 0x8000;
            }
            writeln!(self.asm, "    ld {reg}, 0x{w:04X}").unwrap();
        } else {
            let va = self.val_loc(v);
            let addr = self.fmt_ram_mem(if hi { va.wrapping_add(2) } else { va });
            writeln!(self.asm, "    ld {reg}, {addr}").unwrap();
            if hi && flip_msb {
                let hi_r = if reg == "hl" { "h" } else { "d" };
                writeln!(self.asm, "    ld a, {hi_r}").unwrap();
                writeln!(self.asm, "    xor 0x80").unwrap();
                writeln!(self.asm, "    ld {hi_r}, a").unwrap();
            }
        }
    }

    fn emit_cmp32_flags(&mut self, a_val: ValueId, b_val: ValueId, signed: bool, ordered: bool) {
        if !ordered {
            let ne_lbl = self.fresh_label("cmp32_ne");
            self.emit_load_word32(a_val, false, "hl", false);
            self.emit_load_word32(b_val, false, "de", false);
            writeln!(self.asm, "    or a").unwrap();
            writeln!(self.asm, "    sbc hl, de").unwrap();
            writeln!(self.asm, "    ld a, h").unwrap();
            writeln!(self.asm, "    or l").unwrap();
            writeln!(self.asm, "    jr nz, {ne_lbl}").unwrap();
            self.emit_load_word32(a_val, true, "hl", false);
            self.emit_load_word32(b_val, true, "de", false);
            writeln!(self.asm, "    or a").unwrap();
            writeln!(self.asm, "    sbc hl, de").unwrap();
            writeln!(self.asm, "    ld a, h").unwrap();
            writeln!(self.asm, "    or l").unwrap();
            writeln!(self.asm, "{ne_lbl}:").unwrap();
            return;
        }
        self.emit_load_word32(a_val, false, "hl", false);
        self.emit_load_word32(b_val, false, "de", false);
        writeln!(self.asm, "    or a").unwrap();
        writeln!(self.asm, "    sbc hl, de").unwrap();
        if signed {
            writeln!(self.asm, "    push af").unwrap();
            self.emit_load_word32(a_val, true, "hl", true);
            self.emit_load_word32(b_val, true, "de", true);
            writeln!(self.asm, "    pop af").unwrap();
        } else {
            self.emit_load_word32(a_val, true, "hl", false);
            self.emit_load_word32(b_val, true, "de", false);
        }
        writeln!(self.asm, "    sbc hl, de").unwrap();
    }

    pub(super) fn emit_fused_cmp_branch(
        &mut self,
        fname: &str,
        op: IrBinOp,
        ty: &IrType,
        lhs: ValueId,
        rhs: ValueId,
        branch: (BlockId, BlockId, Option<BlockId>),
    ) {
        let (then_bb, else_bb, next_bb_id) = branch;
        let signed = is_signed_ty(ty);
        let sz = self.program.type_size(ty);

        let emit_branch_cc = |cc: &'static str, asm: &mut String| {
            if next_bb_id == Some(then_bb) {
                writeln!(asm, "    jp {}, {fname}_bb{else_bb}", invert_cc(cc)).unwrap();
            } else {
                writeln!(asm, "    jp {cc}, {fname}_bb{then_bb}").unwrap();
                if next_bb_id != Some(else_bb) {
                    writeln!(asm, "    jp {fname}_bb{else_bb}").unwrap();
                }
            }
        };

        // Fast idiom 1: signed 16-bit `x < 0` or `x >= 0` -> `bit 7, h`!
        if sz == 2
            && signed
            && self.const_vals.get(&rhs) == Some(&0)
            && matches!(op, IrBinOp::Lt | IrBinOp::Ge)
        {
            self.emit_load_hl_from(lhs);
            writeln!(self.asm, "    bit 7, h").unwrap();
            let cc = if op == IrBinOp::Lt { "nz" } else { "z" };
            emit_branch_cc(cc, self.asm);
            return;
        }

        // Fast idiom 2: 16-bit `x == 0` / `x != 0`, or unsigned `x > 0` / `x <= 0` -> `ld a, h; or l`!
        if sz == 2
            && self.const_vals.get(&rhs) == Some(&0)
            && (matches!(op, IrBinOp::Eq | IrBinOp::Ne)
                || (!signed && matches!(op, IrBinOp::Gt | IrBinOp::Le)))
        {
            self.emit_load_hl_from(lhs);
            writeln!(self.asm, "    ld a, h").unwrap();
            writeln!(self.asm, "    or l").unwrap();
            let cc = match op {
                IrBinOp::Eq | IrBinOp::Le => "z",
                IrBinOp::Ne | IrBinOp::Gt => "nz",
                _ => unreachable!(),
            };
            emit_branch_cc(cc, self.asm);
            return;
        }

        // Fast idiom 2b: 32-bit `x == 0` / `x != 0`, or unsigned `x > 0` / `x <= 0`
        if sz == 4
            && self.const_vals.get(&rhs) == Some(&0)
            && (matches!(op, IrBinOp::Eq | IrBinOp::Ne)
                || (!signed && matches!(op, IrBinOp::Gt | IrBinOp::Le)))
        {
            self.emit_load_hlde_from(lhs);
            writeln!(self.asm, "    ld a, e").unwrap();
            writeln!(self.asm, "    or d").unwrap();
            writeln!(self.asm, "    or l").unwrap();
            writeln!(self.asm, "    or h").unwrap();
            let cc = match op {
                IrBinOp::Eq | IrBinOp::Le => "z",
                IrBinOp::Ne | IrBinOp::Gt => "nz",
                _ => unreachable!(),
            };
            emit_branch_cc(cc, self.asm);
            return;
        }

        // Fast idiom 3: 8-bit `x == 0` / `x != 0`, or unsigned `x > 0` / `x <= 0` -> `or a`!
        if sz == 1
            && self.const_vals.get(&rhs) == Some(&0)
            && (matches!(op, IrBinOp::Eq | IrBinOp::Ne)
                || (!signed && matches!(op, IrBinOp::Gt | IrBinOp::Le)))
        {
            self.emit_load_a_from(lhs);
            writeln!(self.asm, "    or a").unwrap();
            let cc = match op {
                IrBinOp::Eq | IrBinOp::Le => "z",
                IrBinOp::Ne | IrBinOp::Gt => "nz",
                _ => unreachable!(),
            };
            emit_branch_cc(cc, self.asm);
            return;
        }

        // Fast idiom 3b: unsigned 8-bit `x > C` / `x <= C` (C < 255) -> `cp (C+1); nc/c`!
        if sz == 1
            && !signed
            && matches!(op, IrBinOp::Gt | IrBinOp::Le)
            && let Some(&c_raw) = self.const_vals.get(&rhs)
            && (0..255).contains(&c_raw)
        {
            self.emit_load_a_from(lhs);
            writeln!(self.asm, "    cp 0x{:02X}", (c_raw + 1) as u8).unwrap();
            let cc = if op == IrBinOp::Gt { "nc" } else { "c" };
            emit_branch_cc(cc, self.asm);
            return;
        }

        // Fast idiom 3c: signed 8-bit `x < 0` / `x >= 0` -> `bit 7, a`, and `x > 0` / `x <= 0` -> `dec a; cp 0x7F`!
        if sz == 1 && signed && self.const_vals.get(&rhs) == Some(&0) {
            self.emit_load_a_from(lhs);
            match op {
                IrBinOp::Lt | IrBinOp::Ge => {
                    writeln!(self.asm, "    bit 7, a").unwrap();
                    let cc = if op == IrBinOp::Lt { "nz" } else { "z" };
                    emit_branch_cc(cc, self.asm);
                    return;
                }
                IrBinOp::Gt | IrBinOp::Le => {
                    writeln!(self.asm, "    dec a").unwrap();
                    writeln!(self.asm, "    cp 0x7F").unwrap();
                    let cc = if op == IrBinOp::Gt { "c" } else { "nc" };
                    emit_branch_cc(cc, self.asm);
                    return;
                }
                _ => {}
            }
        }

        let (a_val, b_val, cc) = match op {
            IrBinOp::Eq => (lhs, rhs, "z"),
            IrBinOp::Ne => (lhs, rhs, "nz"),
            IrBinOp::Lt => (lhs, rhs, "c"),
            IrBinOp::Ge => (lhs, rhs, "nc"),
            IrBinOp::Le => (rhs, lhs, "nc"),
            IrBinOp::Gt => (rhs, lhs, "c"),
            _ => unreachable!(),
        };
        let ordered = !matches!(op, IrBinOp::Eq | IrBinOp::Ne);

        match sz {
            1 => {
                if !signed || !ordered {
                    if let Some(&c) = self.const_vals.get(&b_val) {
                        self.emit_load_a_from(a_val);
                        writeln!(self.asm, "    cp 0x{:02X}", (c & 0xFF) as u8).unwrap();
                    } else if let Some(&reg) = self.val_in_reg.get(&b_val)
                        && reg.byte_size() == 1
                    {
                        self.emit_load_a_from(a_val);
                        writeln!(self.asm, "    cp {}", reg.reg8_name()).unwrap();
                    } else if let Some(&(disp, _)) = self.ix_val_loads.get(&b_val) {
                        self.emit_load_a_from(a_val);
                        writeln!(self.asm, "    cp (ix + {disp})").unwrap();
                    } else {
                        let b_addr = self.fmt_val_mem(b_val);
                        writeln!(self.asm, "    ld a, {b_addr}").unwrap();
                        writeln!(self.asm, "    ld e, a").unwrap();
                        self.emit_load_a_from(a_val);
                        writeln!(self.asm, "    cp e").unwrap();
                    }
                } else if let Some(&c) = self.const_vals.get(&b_val) {
                    let cb = (c as u8) ^ 0x80;
                    if let Some(&ca) = self.const_vals.get(&a_val) {
                        writeln!(self.asm, "    ld a, 0x{:02X}", (ca as u8) ^ 0x80).unwrap();
                    } else {
                        self.emit_load_a_from(a_val);
                        writeln!(self.asm, "    xor 0x80").unwrap();
                    }
                    writeln!(self.asm, "    cp 0x{cb:02X}").unwrap();
                } else {
                    self.emit_load_a_from(b_val);
                    writeln!(self.asm, "    xor 0x80").unwrap();
                    writeln!(self.asm, "    ld e, a").unwrap();
                    if let Some(&ca) = self.const_vals.get(&a_val) {
                        writeln!(self.asm, "    ld a, 0x{:02X}", (ca as u8) ^ 0x80).unwrap();
                    } else {
                        self.emit_load_a_from(a_val);
                        writeln!(self.asm, "    xor 0x80").unwrap();
                    }
                    writeln!(self.asm, "    cp e").unwrap();
                }
                emit_branch_cc(cc, self.asm);
            }
            2 => {
                if let Some(&c) = self.const_vals.get(&a_val) {
                    let mut w = c as u16;
                    if signed && ordered {
                        w ^= 0x8000;
                    }
                    writeln!(self.asm, "    ld hl, 0x{w:04X}").unwrap();
                } else {
                    self.emit_load_hl_from(a_val);
                    if signed && ordered {
                        writeln!(self.asm, "    ld a, h").unwrap();
                        writeln!(self.asm, "    xor 0x80").unwrap();
                        writeln!(self.asm, "    ld h, a").unwrap();
                    }
                }
                if let Some(&c) = self.const_vals.get(&b_val) {
                    let mut w = c as u16;
                    if signed && ordered {
                        w ^= 0x8000;
                    }
                    writeln!(self.asm, "    ld de, 0x{w:04X}").unwrap();
                } else {
                    self.emit_load_de_from(b_val);
                    if signed && ordered {
                        writeln!(self.asm, "    ld a, d").unwrap();
                        writeln!(self.asm, "    xor 0x80").unwrap();
                        writeln!(self.asm, "    ld d, a").unwrap();
                    }
                }
                writeln!(self.asm, "    or a").unwrap();
                writeln!(self.asm, "    sbc hl, de").unwrap();
                emit_branch_cc(cc, self.asm);
            }
            4 => {
                self.emit_cmp32_flags(a_val, b_val, signed, ordered);
                emit_branch_cc(cc, self.asm);
            }
            _ => {}
        }
    }

    pub(super) fn emit_binop(
        &mut self,
        dst: ValueId,
        op: IrBinOp,
        mode: ArithMode,
        ty: &IrType,
        lhs: ValueId,
        rhs: ValueId,
    ) {
        let la = self.val_loc(lhs);
        let ra = self.val_loc(rhs);
        let da = self.val_addr(dst);
        let la_mem = self.fmt_ram_mem(la);
        let ra_mem = self.fmt_ram_mem(ra);
        let da_mem = self.fmt_ram_mem(da);
        let signed = is_signed_ty(ty);

        // Handle comparisons first (result is always 1-byte Bool in A)
        let is_cmp = matches!(
            op,
            IrBinOp::Eq | IrBinOp::Ne | IrBinOp::Lt | IrBinOp::Le | IrBinOp::Gt | IrBinOp::Ge
        );
        if is_cmp {
            let done = self.fresh_label("cmp");
            let sz = self.program.type_size(ty);
            // Fast idioms for comparisons against 0 / constants (0 scratch spills!)
            if sz == 1
                && self.const_vals.get(&rhs) == Some(&0)
                && (matches!(op, IrBinOp::Eq | IrBinOp::Ne)
                    || (!signed && matches!(op, IrBinOp::Gt | IrBinOp::Le)))
            {
                self.emit_load_a_from(lhs);
                writeln!(self.asm, "    or a").unwrap();
                let cc = match op {
                    IrBinOp::Eq | IrBinOp::Le => "z",
                    IrBinOp::Ne | IrBinOp::Gt => "nz",
                    _ => unreachable!(),
                };
                writeln!(self.asm, "    ld a, 1").unwrap();
                writeln!(self.asm, "    jr {cc}, {done}").unwrap();
                writeln!(self.asm, "    xor a").unwrap();
                writeln!(self.asm, "{done}:").unwrap();
                writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                return;
            }
            if sz == 2
                && self.const_vals.get(&rhs) == Some(&0)
                && (matches!(op, IrBinOp::Eq | IrBinOp::Ne)
                    || (!signed && matches!(op, IrBinOp::Gt | IrBinOp::Le)))
            {
                self.emit_load_hl_from(lhs);
                writeln!(self.asm, "    ld a, h").unwrap();
                writeln!(self.asm, "    or l").unwrap();
                let cc = match op {
                    IrBinOp::Eq | IrBinOp::Le => "z",
                    IrBinOp::Ne | IrBinOp::Gt => "nz",
                    _ => unreachable!(),
                };
                writeln!(self.asm, "    ld a, 1").unwrap();
                writeln!(self.asm, "    jr {cc}, {done}").unwrap();
                writeln!(self.asm, "    xor a").unwrap();
                writeln!(self.asm, "{done}:").unwrap();
                writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                return;
            }
            if sz == 4
                && self.const_vals.get(&rhs) == Some(&0)
                && (matches!(op, IrBinOp::Eq | IrBinOp::Ne)
                    || (!signed && matches!(op, IrBinOp::Gt | IrBinOp::Le)))
            {
                self.emit_load_hlde_from(lhs);
                writeln!(self.asm, "    ld a, e").unwrap();
                writeln!(self.asm, "    or d").unwrap();
                writeln!(self.asm, "    or l").unwrap();
                writeln!(self.asm, "    or h").unwrap();
                let cc = match op {
                    IrBinOp::Eq | IrBinOp::Le => "z",
                    IrBinOp::Ne | IrBinOp::Gt => "nz",
                    _ => unreachable!(),
                };
                writeln!(self.asm, "    ld a, 1").unwrap();
                writeln!(self.asm, "    jr {cc}, {done}").unwrap();
                writeln!(self.asm, "    xor a").unwrap();
                writeln!(self.asm, "{done}:").unwrap();
                writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                return;
            }
            if sz == 1
                && !signed
                && matches!(op, IrBinOp::Gt | IrBinOp::Le)
                && let Some(&c_raw) = self.const_vals.get(&rhs)
                && (0..255).contains(&c_raw)
            {
                self.emit_load_a_from(lhs);
                writeln!(self.asm, "    cp 0x{:02X}", (c_raw + 1) as u8).unwrap();
                let cc = if op == IrBinOp::Gt { "nc" } else { "c" };
                writeln!(self.asm, "    ld a, 1").unwrap();
                writeln!(self.asm, "    jr {cc}, {done}").unwrap();
                writeln!(self.asm, "    xor a").unwrap();
                writeln!(self.asm, "{done}:").unwrap();
                writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                return;
            }

            let (a_val, b_val, cc) = match op {
                IrBinOp::Eq => (lhs, rhs, "z"),
                IrBinOp::Ne => (lhs, rhs, "nz"),
                IrBinOp::Lt => (lhs, rhs, "c"),
                IrBinOp::Ge => (lhs, rhs, "nc"),
                IrBinOp::Le => (rhs, lhs, "nc"),
                IrBinOp::Gt => (rhs, lhs, "c"),
                _ => unreachable!(),
            };
            let ordered = !matches!(op, IrBinOp::Eq | IrBinOp::Ne);
            match sz {
                1 => {
                    if !signed || !ordered {
                        if let Some(&c) = self.const_vals.get(&b_val) {
                            self.emit_load_a_from(a_val);
                            writeln!(self.asm, "    cp 0x{:02X}", (c & 0xFF) as u8).unwrap();
                        } else if let Some(&reg) = self.val_in_reg.get(&b_val)
                            && reg.byte_size() == 1
                        {
                            self.emit_load_a_from(a_val);
                            writeln!(self.asm, "    cp {}", reg.reg8_name()).unwrap();
                        } else if let Some(&(disp, _)) = self.ix_val_loads.get(&b_val) {
                            self.emit_load_a_from(a_val);
                            writeln!(self.asm, "    cp (ix + {disp})").unwrap();
                        } else {
                            self.emit_load_a_from(b_val);
                            writeln!(self.asm, "    ld e, a").unwrap();
                            self.emit_load_a_from(a_val);
                            writeln!(self.asm, "    cp e").unwrap();
                        }
                    } else if let Some(&c) = self.const_vals.get(&b_val) {
                        let cb = (c as u8) ^ 0x80;
                        if let Some(&ca) = self.const_vals.get(&a_val) {
                            writeln!(self.asm, "    ld a, 0x{:02X}", (ca as u8) ^ 0x80).unwrap();
                        } else {
                            self.emit_load_a_from(a_val);
                            writeln!(self.asm, "    xor 0x80").unwrap();
                        }
                        writeln!(self.asm, "    cp 0x{cb:02X}").unwrap();
                    } else {
                        self.emit_load_a_from(b_val);
                        writeln!(self.asm, "    xor 0x80").unwrap();
                        writeln!(self.asm, "    ld e, a").unwrap();
                        if let Some(&ca) = self.const_vals.get(&a_val) {
                            writeln!(self.asm, "    ld a, 0x{:02X}", (ca as u8) ^ 0x80).unwrap();
                        } else {
                            self.emit_load_a_from(a_val);
                            writeln!(self.asm, "    xor 0x80").unwrap();
                        }
                        writeln!(self.asm, "    cp e").unwrap();
                    }
                    self.emit_bool_from_cc(cc, da);
                }
                2 => {
                    if let Some(&c) = self.const_vals.get(&a_val) {
                        let mut w = c as u16;
                        if signed && ordered {
                            w ^= 0x8000;
                        }
                        writeln!(self.asm, "    ld hl, 0x{w:04X}").unwrap();
                    } else {
                        self.emit_load_hl_from(a_val);
                        if signed && ordered {
                            writeln!(self.asm, "    ld a, h").unwrap();
                            writeln!(self.asm, "    xor 0x80").unwrap();
                            writeln!(self.asm, "    ld h, a").unwrap();
                        }
                    }
                    if let Some(&c) = self.const_vals.get(&b_val) {
                        let mut w = c as u16;
                        if signed && ordered {
                            w ^= 0x8000;
                        }
                        writeln!(self.asm, "    ld de, 0x{w:04X}").unwrap();
                    } else {
                        self.emit_load_de_from(b_val);
                        if signed && ordered {
                            writeln!(self.asm, "    ld a, d").unwrap();
                            writeln!(self.asm, "    xor 0x80").unwrap();
                            writeln!(self.asm, "    ld d, a").unwrap();
                        }
                    }
                    writeln!(self.asm, "    or a").unwrap();
                    writeln!(self.asm, "    sbc hl, de").unwrap();
                    self.emit_bool_from_cc(cc, da);
                }
                4 => {
                    self.emit_cmp32_flags(a_val, b_val, signed, ordered);
                    self.emit_bool_from_cc(cc, da);
                }
                _ => {}
            }
            return;
        }

        match self.program.type_size(ty) {
            1 => {
                if matches!(op, IrBinOp::Shl | IrBinOp::Shr) {
                    if let Some(&k_raw) = self.const_vals.get(&rhs) {
                        let k = k_raw.clamp(0, 8) as u8;
                        if let Some(&cl) = self.const_vals.get(&lhs) {
                            writeln!(self.asm, "    ld a, 0x{:02X}", (cl & 0xFF) as u8).unwrap();
                        } else {
                            self.emit_load_a_from(lhs);
                        }
                        match (op, k) {
                            (_, 0) => {}
                            (IrBinOp::Shl, 8) => writeln!(self.asm, "    xor a").unwrap(),
                            (IrBinOp::Shr, 8) if !signed => writeln!(self.asm, "    xor a").unwrap(),
                            (IrBinOp::Shr, 8) => {
                                writeln!(self.asm, "    add a, a").unwrap();
                                writeln!(self.asm, "    sbc a, a").unwrap();
                            }
                            (IrBinOp::Shl, 1..=4) => {
                                for _ in 0..k {
                                    writeln!(self.asm, "    add a, a").unwrap();
                                }
                            }
                            (IrBinOp::Shr, 1..=3) => {
                                for _ in 0..k {
                                    if signed {
                                        writeln!(self.asm, "    sra a").unwrap();
                                    } else {
                                        writeln!(self.asm, "    srl a").unwrap();
                                    }
                                }
                            }
                            _ => {
                                let loop_lbl = self.fresh_label("sh_lp");
                                writeln!(self.asm, "    ld b, {k}").unwrap();
                                writeln!(self.asm, "{loop_lbl}:").unwrap();
                                match op {
                                    IrBinOp::Shl => writeln!(self.asm, "    add a, a").unwrap(),
                                    IrBinOp::Shr if signed => {
                                        writeln!(self.asm, "    sra a").unwrap()
                                    }
                                    IrBinOp::Shr => writeln!(self.asm, "    srl a").unwrap(),
                                    _ => unreachable!(),
                                }
                                writeln!(self.asm, "    djnz {loop_lbl}").unwrap();
                            }
                        }
                        writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                        return;
                    }
                    if self
                        .const_vals
                        .get(&lhs)
                        .is_some_and(|&c| (c & 0xFF) == 0)
                    {
                        writeln!(self.asm, "    xor a").unwrap();
                        writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                        return;
                    }
                    let loop_lbl = self.fresh_label("sh_lp");
                    let done_lbl = self.fresh_label("sh_dn");
                    if self.val_in_reg.get(&lhs) == Some(&RegLoc::B) {
                        self.emit_load_a_from(rhs);
                        writeln!(self.asm, "    or a").unwrap();
                        writeln!(self.asm, "    ld e, a").unwrap();
                        writeln!(self.asm, "    ld a, b").unwrap();
                        writeln!(self.asm, "    ld b, e").unwrap();
                    } else {
                        self.emit_load_a_from(rhs);
                        writeln!(self.asm, "    or a").unwrap();
                        writeln!(self.asm, "    ld b, a").unwrap();
                        self.emit_load_a_from(lhs);
                    }
                    writeln!(self.asm, "    jr z, {done_lbl}").unwrap();
                    writeln!(self.asm, "{loop_lbl}:").unwrap();
                    match op {
                        IrBinOp::Shl => writeln!(self.asm, "    add a, a").unwrap(),
                        IrBinOp::Shr if signed => writeln!(self.asm, "    sra a").unwrap(),
                        IrBinOp::Shr => writeln!(self.asm, "    srl a").unwrap(),
                        _ => unreachable!(),
                    }
                    writeln!(self.asm, "    djnz {loop_lbl}").unwrap();
                    writeln!(self.asm, "{done_lbl}:").unwrap();
                    writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                    return;
                }
                // Fast 8-bit `C - x` (including `0 - x` -> `neg`) (non-saturating)
                if mode != ArithMode::Saturating
                    && op == IrBinOp::Sub
                    && let Some(&cl_raw) = self.const_vals.get(&lhs)
                {
                    let cl = (cl_raw & 0xFF) as u8;
                    if let Some(&cr) = self.const_vals.get(&rhs) {
                        let b = cl.wrapping_sub((cr & 0xFF) as u8);
                        if b == 0 {
                            writeln!(self.asm, "    xor a").unwrap();
                        } else {
                            writeln!(self.asm, "    ld a, 0x{b:02X}").unwrap();
                        }
                    } else if cl == 0 {
                        self.emit_load_a_from(rhs);
                        writeln!(self.asm, "    neg").unwrap();
                    } else {
                        self.emit_load_a_from(rhs);
                        writeln!(self.asm, "    ld e, a").unwrap();
                        writeln!(self.asm, "    ld a, 0x{cl:02X}").unwrap();
                        writeln!(self.asm, "    sub e").unwrap();
                    }
                    writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                    return;
                }
                // Fast 8-bit immediate arithmetic/bitwise when `rhs` is a constant (non-saturating)
                if mode != ArithMode::Saturating
                    && let Some(&c_raw) = self.const_vals.get(&rhs)
                    && matches!(
                        op,
                        IrBinOp::Add
                            | IrBinOp::Sub
                            | IrBinOp::BitAnd
                            | IrBinOp::BitOr
                            | IrBinOp::BitXor
                    )
                {
                    let c = (c_raw & 0xFF) as u8;
                    self.emit_load_a_from(lhs);
                    match (op, c) {
                        (IrBinOp::Add, 0)
                        | (IrBinOp::Sub, 0)
                        | (IrBinOp::BitOr, 0)
                        | (IrBinOp::BitXor, 0) => {}
                        (IrBinOp::Add, 1) => writeln!(self.asm, "    inc a").unwrap(),
                        (IrBinOp::Add, 2) => {
                            writeln!(self.asm, "    inc a").unwrap();
                            writeln!(self.asm, "    inc a").unwrap();
                        }
                        (IrBinOp::Add, _) => writeln!(self.asm, "    add a, 0x{c:02X}").unwrap(),
                        (IrBinOp::Sub, 1) => writeln!(self.asm, "    dec a").unwrap(),
                        (IrBinOp::Sub, 2) => {
                            writeln!(self.asm, "    dec a").unwrap();
                            writeln!(self.asm, "    dec a").unwrap();
                        }
                        (IrBinOp::Sub, _) => writeln!(self.asm, "    sub 0x{c:02X}").unwrap(),
                        (IrBinOp::BitAnd, 0) => writeln!(self.asm, "    xor a").unwrap(),
                        (IrBinOp::BitAnd, 0xFF) => {}
                        (IrBinOp::BitAnd, _) => writeln!(self.asm, "    and 0x{c:02X}").unwrap(),
                        (IrBinOp::BitOr, 0xFF) => writeln!(self.asm, "    ld a, 0xFF").unwrap(),
                        (IrBinOp::BitOr, _) => writeln!(self.asm, "    or 0x{c:02X}").unwrap(),
                        (IrBinOp::BitXor, 0xFF) => writeln!(self.asm, "    cpl").unwrap(),
                        (IrBinOp::BitXor, _) => writeln!(self.asm, "    xor 0x{c:02X}").unwrap(),
                        _ => unreachable!(),
                    }
                    writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                    return;
                }
                // Direct `B`/`C` or `(ix + d)` RHS for 8-bit non-saturating ALU ops (0 scratch!)
                if mode != ArithMode::Saturating
                    && matches!(
                        op,
                        IrBinOp::Add
                            | IrBinOp::Sub
                            | IrBinOp::BitAnd
                            | IrBinOp::BitOr
                            | IrBinOp::BitXor
                    )
                {
                    if let Some(&reg) = self.val_in_reg.get(&rhs)
                        && reg.byte_size() == 1
                    {
                        let r = reg.reg8_name();
                        self.emit_load_a_from(lhs);
                        match op {
                            IrBinOp::Add => writeln!(self.asm, "    add a, {r}").unwrap(),
                            IrBinOp::Sub => writeln!(self.asm, "    sub {r}").unwrap(),
                            IrBinOp::BitAnd => writeln!(self.asm, "    and {r}").unwrap(),
                            IrBinOp::BitOr => writeln!(self.asm, "    or {r}").unwrap(),
                            IrBinOp::BitXor => writeln!(self.asm, "    xor {r}").unwrap(),
                            _ => unreachable!(),
                        }
                        writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                        return;
                    }
                    if let Some(&(rdisp, _)) = self.ix_val_loads.get(&rhs) {
                        self.emit_load_a_from(lhs);
                        match op {
                            IrBinOp::Add => writeln!(self.asm, "    add a, (ix + {rdisp})").unwrap(),
                            IrBinOp::Sub => writeln!(self.asm, "    sub (ix + {rdisp})").unwrap(),
                            IrBinOp::BitAnd => {
                                writeln!(self.asm, "    and (ix + {rdisp})").unwrap()
                            }
                            IrBinOp::BitOr => writeln!(self.asm, "    or (ix + {rdisp})").unwrap(),
                            IrBinOp::BitXor => {
                                writeln!(self.asm, "    xor (ix + {rdisp})").unwrap()
                            }
                            _ => unreachable!(),
                        }
                        writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                        return;
                    }
                }
                self.emit_load_a_from(rhs);
                writeln!(self.asm, "    ld e, a").unwrap();
                self.emit_load_a_from(lhs);
                match (op, mode) {
                    (IrBinOp::Add, ArithMode::Saturating) if !signed => {
                        let lbl = self.fresh_label("sat");
                        writeln!(self.asm, "    add a, e").unwrap();
                        writeln!(self.asm, "    jr nc, {lbl}").unwrap();
                        writeln!(self.asm, "    ld a, 0xFF").unwrap();
                        writeln!(self.asm, "{lbl}:").unwrap();
                    }
                    (IrBinOp::Sub, ArithMode::Saturating) if !signed => {
                        let lbl = self.fresh_label("sat");
                        writeln!(self.asm, "    sub e").unwrap();
                        writeln!(self.asm, "    jr nc, {lbl}").unwrap();
                        writeln!(self.asm, "    xor a").unwrap();
                        writeln!(self.asm, "{lbl}:").unwrap();
                    }
                    (IrBinOp::Add, ArithMode::Saturating) => {
                        let lbl = self.fresh_label("sat");
                        writeln!(self.asm, "    add a, e").unwrap();
                        writeln!(self.asm, "    jp po, {lbl}").unwrap();
                        writeln!(self.asm, "    ld a, 0x7F").unwrap();
                        writeln!(self.asm, "    jp m, {lbl}").unwrap();
                        writeln!(self.asm, "    ld a, 0x80").unwrap();
                        writeln!(self.asm, "{lbl}:").unwrap();
                    }
                    (IrBinOp::Sub, ArithMode::Saturating) => {
                        let lbl = self.fresh_label("sat");
                        writeln!(self.asm, "    sub e").unwrap();
                        writeln!(self.asm, "    jp po, {lbl}").unwrap();
                        writeln!(self.asm, "    ld a, 0x7F").unwrap();
                        writeln!(self.asm, "    jp m, {lbl}").unwrap();
                        writeln!(self.asm, "    ld a, 0x80").unwrap();
                        writeln!(self.asm, "{lbl}:").unwrap();
                    }
                    (IrBinOp::Add, _) => writeln!(self.asm, "    add a, e").unwrap(),
                    (IrBinOp::Sub, _) => writeln!(self.asm, "    sub e").unwrap(),
                    (IrBinOp::BitAnd, _) => writeln!(self.asm, "    and e").unwrap(),
                    (IrBinOp::BitOr, _) => writeln!(self.asm, "    or e").unwrap(),
                    (IrBinOp::BitXor, _) => writeln!(self.asm, "    xor e").unwrap(),
                    (IrBinOp::Mul, _) => {
                        // 8x8 -> low 8 bits: A = lhs, E = rhs
                        let lp = self.fresh_label("mul8_lp");
                        let sk = self.fresh_label("mul8_sk");
                        writeln!(self.asm, "    ld d, a").unwrap();
                        writeln!(self.asm, "    xor a").unwrap();
                        writeln!(self.asm, "    ld b, 8").unwrap();
                        writeln!(self.asm, "{lp}:").unwrap();
                        writeln!(self.asm, "    add a, a").unwrap();
                        writeln!(self.asm, "    sla d").unwrap();
                        writeln!(self.asm, "    jr nc, {sk}").unwrap();
                        writeln!(self.asm, "    add a, e").unwrap();
                        writeln!(self.asm, "{sk}:").unwrap();
                        writeln!(self.asm, "    djnz {lp}").unwrap();
                    }
                    (IrBinOp::Div, _) => {
                        let helper = if signed {
                            self.runtime_needs.sdiv8 = true;
                            "__aun_sdiv8"
                        } else {
                            self.runtime_needs.udiv8 = true;
                            "__aun_udiv8"
                        };
                        writeln!(self.asm, "    call {helper}").unwrap();
                        self.val_in_reg
                            .retain(|_, &mut vr| !crate::regalloc::RegMask::BC.intersects(vr.mask()));
                    }
                    (IrBinOp::Rem, _) => {
                        let helper = if signed {
                            self.runtime_needs.sdiv8 = true;
                            "__aun_srem8"
                        } else {
                            self.runtime_needs.udiv8 = true;
                            "__aun_urem8"
                        };
                        writeln!(self.asm, "    call {helper}").unwrap();
                        self.val_in_reg
                            .retain(|_, &mut vr| !crate::regalloc::RegMask::BC.intersects(vr.mask()));
                    }
                    other => unreachable!("unhandled 1-byte binop {other:?}"),
                }
                writeln!(self.asm, "    ld {da_mem}, a").unwrap();
            }
            2 => {
                // Fused 16-bit `x ^ (x << k)` / `x ^ (x >> k)` idioms (e.g. xorshift16)
                if op == IrBinOp::BitXor
                    && let Some(&(src_v, sop, k)) = self.fused_xor_shift.get(&dst)
                {
                    if let Some(&c) = self.const_vals.get(&src_v) {
                        writeln!(self.asm, "    ld hl, 0x{:04X}", c as u16).unwrap();
                    } else {
                        let sa = self.fmt_val_mem(src_v);
                        writeln!(self.asm, "    ld hl, {sa}").unwrap();
                    }
                    match (sop, k) {
                        (IrBinOp::Shl, 7) => {
                            writeln!(self.asm, "    ld a, h").unwrap();
                            writeln!(self.asm, "    rra").unwrap();
                            writeln!(self.asm, "    ld a, l").unwrap();
                            writeln!(self.asm, "    rra").unwrap();
                            writeln!(self.asm, "    ld d, a").unwrap();
                            writeln!(self.asm, "    ld a, 0").unwrap();
                            writeln!(self.asm, "    rra").unwrap();
                            writeln!(self.asm, "    xor l").unwrap();
                            writeln!(self.asm, "    ld l, a").unwrap();
                            writeln!(self.asm, "    ld a, h").unwrap();
                            writeln!(self.asm, "    xor d").unwrap();
                            writeln!(self.asm, "    ld h, a").unwrap();
                        }
                        (IrBinOp::Shl, 8) => {
                            writeln!(self.asm, "    ld a, h").unwrap();
                            writeln!(self.asm, "    xor l").unwrap();
                            writeln!(self.asm, "    ld h, a").unwrap();
                        }
                        (IrBinOp::Shr, 8..=15) => {
                            writeln!(self.asm, "    ld a, h").unwrap();
                            for _ in 0..(k - 8) {
                                writeln!(self.asm, "    srl a").unwrap();
                            }
                            writeln!(self.asm, "    xor l").unwrap();
                            writeln!(self.asm, "    ld l, a").unwrap();
                        }
                        _ => unreachable!(),
                    }
                    writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                    return;
                }
                // 16-bit multiplication: constant shift-and-add or general 16x16 loop
                if op == IrBinOp::Mul {
                    if let Some(&c) = self.const_vals.get(&rhs) {
                        if let Some(&cl) = self.const_vals.get(&lhs) {
                            writeln!(
                                self.asm,
                                "    ld hl, 0x{:04X}",
                                (cl as u16).wrapping_mul(c as u16)
                            )
                            .unwrap();
                        } else {
                            self.emit_load_hl_from(lhs);
                            emit_mul_hl_const(c as u16, self.asm);
                        }
                    } else if let Some(&c) = self.const_vals.get(&lhs) {
                        self.emit_load_hl_from(rhs);
                        emit_mul_hl_const(c as u16, self.asm);
                    } else {
                        let lp = self.fresh_label("mul16_lp");
                        let sk = self.fresh_label("mul16_sk");
                        writeln!(self.asm, "    ld de, {la_mem}").unwrap();
                        writeln!(self.asm, "    ld bc, {ra_mem}").unwrap();
                        writeln!(self.asm, "    ld hl, 0").unwrap();
                        writeln!(self.asm, "    ld a, 16").unwrap();
                        writeln!(self.asm, "{lp}:").unwrap();
                        writeln!(self.asm, "    add hl, hl").unwrap();
                        writeln!(self.asm, "    sla c").unwrap();
                        writeln!(self.asm, "    rl b").unwrap();
                        writeln!(self.asm, "    jr nc, {sk}").unwrap();
                        writeln!(self.asm, "    add hl, de").unwrap();
                        writeln!(self.asm, "{sk}:").unwrap();
                        writeln!(self.asm, "    dec a").unwrap();
                        writeln!(self.asm, "    jr nz, {lp}").unwrap();
                    }
                    writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                    return;
                }
                // Compact 16-bit negation: `0 - x` (6 bytes, no DE clobber)
                if op == IrBinOp::Sub && self.const_vals.get(&lhs) == Some(&0) {
                    if let Some(&c) = self.const_vals.get(&rhs) {
                        writeln!(self.asm, "    ld hl, 0x{:04X}", (0u16).wrapping_sub(c as u16))
                            .unwrap();
                    } else {
                        self.emit_load_hl_from(rhs);
                        writeln!(self.asm, "    xor a").unwrap();
                        writeln!(self.asm, "    sub l").unwrap();
                        writeln!(self.asm, "    ld l, a").unwrap();
                        writeln!(self.asm, "    sbc a, a").unwrap();
                        writeln!(self.asm, "    sub h").unwrap();
                        writeln!(self.asm, "    ld h, a").unwrap();
                    }
                    writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                    return;
                }
                if matches!(op, IrBinOp::Shl | IrBinOp::Shr) {
                    let is_shl = op == IrBinOp::Shl;
                    if let Some(&k_raw) = self.const_vals.get(&rhs) {
                        let orig_k = k_raw.clamp(0, 16) as u8;
                        let (used_cse, src_addr, k) = if let Some(&(prev_k, prev_da)) =
                            self.shift_cse.get(&(la, is_shl, signed))
                            && prev_k < orig_k
                            && orig_k <= 7
                        {
                            (true, prev_da, orig_k - prev_k)
                        } else {
                            (false, la, orig_k)
                        };
                        if used_cse {
                            let src_mem = self.fmt_ram_mem(src_addr);
                            writeln!(self.asm, "    ld hl, {src_mem}").unwrap();
                        } else {
                            self.emit_load_hl_from(lhs);
                        }
                        match (op, k) {
                            (_, 0) => {}
                            (IrBinOp::Shl, 16) => writeln!(self.asm, "    ld hl, 0").unwrap(),
                            (IrBinOp::Shl, 1..=4) => {
                                for _ in 0..k {
                                    writeln!(self.asm, "    add hl, hl").unwrap();
                                }
                            }
                            (IrBinOp::Shl, 7) => {
                                // (HL << 7): H = (H<<7)|(L>>1), L = (L<<7)
                                writeln!(self.asm, "    ld a, h").unwrap();
                                writeln!(self.asm, "    rra").unwrap();
                                writeln!(self.asm, "    ld a, l").unwrap();
                                writeln!(self.asm, "    rra").unwrap();
                                writeln!(self.asm, "    ld h, a").unwrap();
                                writeln!(self.asm, "    ld a, 0").unwrap();
                                writeln!(self.asm, "    rra").unwrap();
                                writeln!(self.asm, "    ld l, a").unwrap();
                            }
                            (IrBinOp::Shl, 8) => {
                                writeln!(self.asm, "    ld h, l").unwrap();
                                writeln!(self.asm, "    ld l, 0").unwrap();
                            }
                            (IrBinOp::Shl, 9..=15) => {
                                writeln!(self.asm, "    ld h, l").unwrap();
                                writeln!(self.asm, "    ld l, 0").unwrap();
                                for _ in 0..(k - 8) {
                                    writeln!(self.asm, "    sla h").unwrap();
                                }
                            }
                            (IrBinOp::Shr, 16) if !signed => {
                                writeln!(self.asm, "    ld hl, 0").unwrap()
                            }
                            (IrBinOp::Shr, 1..=3) => {
                                for _ in 0..k {
                                    if signed {
                                        writeln!(self.asm, "    sra h").unwrap();
                                    } else {
                                        writeln!(self.asm, "    srl h").unwrap();
                                    }
                                    writeln!(self.asm, "    rr l").unwrap();
                                }
                            }
                            (IrBinOp::Shr, 8) if !signed => {
                                writeln!(self.asm, "    ld l, h").unwrap();
                                writeln!(self.asm, "    ld h, 0").unwrap();
                            }
                            (IrBinOp::Shr, 9..=15) if !signed => {
                                writeln!(self.asm, "    ld l, h").unwrap();
                                writeln!(self.asm, "    ld h, 0").unwrap();
                                for _ in 0..(k - 8) {
                                    writeln!(self.asm, "    srl l").unwrap();
                                }
                            }
                            _ => {
                                let loop_lbl = self.fresh_label("sh16_lp");
                                writeln!(self.asm, "    ld b, {k}").unwrap();
                                writeln!(self.asm, "{loop_lbl}:").unwrap();
                                match op {
                                    IrBinOp::Shl => writeln!(self.asm, "    add hl, hl").unwrap(),
                                    IrBinOp::Shr if signed => {
                                        writeln!(self.asm, "    sra h").unwrap();
                                        writeln!(self.asm, "    rr l").unwrap();
                                    }
                                    IrBinOp::Shr => {
                                        writeln!(self.asm, "    srl h").unwrap();
                                        writeln!(self.asm, "    rr l").unwrap();
                                    }
                                    _ => unreachable!(),
                                }
                                writeln!(self.asm, "    djnz {loop_lbl}").unwrap();
                            }
                        }
                        writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                        if orig_k > 0 && orig_k < 8 {
                            self.shift_cse.insert((la, is_shl, signed), (orig_k, da));
                        }
                        return;
                    }
                    self.emit_load_hl_from(lhs);
                    let loop_lbl = self.fresh_label("sh16_lp");
                    let done_lbl = self.fresh_label("sh16_dn");
                    self.emit_load_a_from(rhs);
                    writeln!(self.asm, "    or a").unwrap();
                    writeln!(self.asm, "    jr z, {done_lbl}").unwrap();
                    writeln!(self.asm, "    ld b, a").unwrap();
                    writeln!(self.asm, "{loop_lbl}:").unwrap();
                    match op {
                        IrBinOp::Shl => writeln!(self.asm, "    add hl, hl").unwrap(),
                        IrBinOp::Shr if signed => {
                            writeln!(self.asm, "    sra h").unwrap();
                            writeln!(self.asm, "    rr l").unwrap();
                        }
                        IrBinOp::Shr => {
                            writeln!(self.asm, "    srl h").unwrap();
                            writeln!(self.asm, "    rr l").unwrap();
                        }
                        _ => unreachable!(),
                    }
                    writeln!(self.asm, "    djnz {loop_lbl}").unwrap();
                    writeln!(self.asm, "{done_lbl}:").unwrap();
                    writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                    return;
                }
                // Compact 16-bit add/sub by small constant 0..=3 using `inc hl` / `dec hl`
                if matches!(op, IrBinOp::Add | IrBinOp::Sub)
                    && let Some(&c_raw) = self.const_vals.get(&rhs)
                    && (0..=3).contains(&c_raw)
                {
                    if let Some(&cl) = self.const_vals.get(&lhs) {
                        let res = if op == IrBinOp::Add {
                            (cl as u16).wrapping_add(c_raw as u16)
                        } else {
                            (cl as u16).wrapping_sub(c_raw as u16)
                        };
                        writeln!(self.asm, "    ld hl, 0x{res:04X}").unwrap();
                    } else {
                        self.emit_load_hl_from(lhs);
                        let inst = if op == IrBinOp::Add {
                            "inc hl"
                        } else {
                            "dec hl"
                        };
                        for _ in 0..c_raw {
                            writeln!(self.asm, "    {inst}").unwrap();
                        }
                    }
                    writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                    return;
                }
                // Compact 16-bit bitwise ops with constant mask (avoids `ld de, imm` + redundant high/low byte ops)
                if matches!(op, IrBinOp::BitAnd | IrBinOp::BitOr | IrBinOp::BitXor)
                    && let Some(&c_raw) = self.const_vals.get(&rhs)
                {
                    let w = c_raw as u16;
                    let lo = (w & 0xFF) as u8;
                    let hi = (w >> 8) as u8;
                    if let Some(&cl) = self.const_vals.get(&lhs) {
                        let lw = cl as u16;
                        let res = match op {
                            IrBinOp::BitAnd => lw & w,
                            IrBinOp::BitOr => lw | w,
                            IrBinOp::BitXor => lw ^ w,
                            _ => unreachable!(),
                        };
                        writeln!(self.asm, "    ld hl, 0x{res:04X}").unwrap();
                    } else {
                        self.emit_load_hl_from(lhs);
                        match op {
                            IrBinOp::BitAnd => {
                                match lo {
                                    0x00 => writeln!(self.asm, "    ld l, 0").unwrap(),
                                    0xFF => {}
                                    _ => {
                                        writeln!(self.asm, "    ld a, l").unwrap();
                                        writeln!(self.asm, "    and 0x{lo:02X}").unwrap();
                                        writeln!(self.asm, "    ld l, a").unwrap();
                                    }
                                }
                                match hi {
                                    0x00 => writeln!(self.asm, "    ld h, 0").unwrap(),
                                    0xFF => {}
                                    _ => {
                                        writeln!(self.asm, "    ld a, h").unwrap();
                                        writeln!(self.asm, "    and 0x{hi:02X}").unwrap();
                                        writeln!(self.asm, "    ld h, a").unwrap();
                                    }
                                }
                            }
                            IrBinOp::BitOr => {
                                match lo {
                                    0x00 => {}
                                    0xFF => writeln!(self.asm, "    ld l, 0xFF").unwrap(),
                                    _ => {
                                        writeln!(self.asm, "    ld a, l").unwrap();
                                        writeln!(self.asm, "    or 0x{lo:02X}").unwrap();
                                        writeln!(self.asm, "    ld l, a").unwrap();
                                    }
                                }
                                match hi {
                                    0x00 => {}
                                    0xFF => writeln!(self.asm, "    ld h, 0xFF").unwrap(),
                                    _ => {
                                        writeln!(self.asm, "    ld a, h").unwrap();
                                        writeln!(self.asm, "    or 0x{hi:02X}").unwrap();
                                        writeln!(self.asm, "    ld h, a").unwrap();
                                    }
                                }
                            }
                            IrBinOp::BitXor => {
                                if lo != 0x00 {
                                    writeln!(self.asm, "    ld a, l").unwrap();
                                    if lo == 0xFF {
                                        writeln!(self.asm, "    cpl").unwrap();
                                    } else {
                                        writeln!(self.asm, "    xor 0x{lo:02X}").unwrap();
                                    }
                                    writeln!(self.asm, "    ld l, a").unwrap();
                                }
                                if hi != 0x00 {
                                    writeln!(self.asm, "    ld a, h").unwrap();
                                    if hi == 0xFF {
                                        writeln!(self.asm, "    cpl").unwrap();
                                    } else {
                                        writeln!(self.asm, "    xor 0x{hi:02X}").unwrap();
                                    }
                                    writeln!(self.asm, "    ld h, a").unwrap();
                                }
                            }
                            _ => unreachable!(),
                        }
                    }
                    writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                    return;
                }
                if op == IrBinOp::Add && self.val_in_reg.get(&rhs) == Some(&RegLoc::BC) {
                    self.emit_load_hl_from(lhs);
                    writeln!(self.asm, "    add hl, bc").unwrap();
                    writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                    return;
                }
                if op == IrBinOp::Add && self.val_in_reg.get(&lhs) == Some(&RegLoc::BC) {
                    self.emit_load_hl_from(rhs);
                    writeln!(self.asm, "    add hl, bc").unwrap();
                    writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                    return;
                }
                self.emit_load_hl_from(lhs);
                self.emit_load_de_from(rhs);
                match op {
                    IrBinOp::Add => writeln!(self.asm, "    add hl, de").unwrap(),
                    IrBinOp::Sub => {
                        writeln!(self.asm, "    or a").unwrap();
                        writeln!(self.asm, "    sbc hl, de").unwrap();
                    }
                    IrBinOp::BitAnd => {
                        writeln!(self.asm, "    ld a, l").unwrap();
                        writeln!(self.asm, "    and e").unwrap();
                        writeln!(self.asm, "    ld l, a").unwrap();
                        writeln!(self.asm, "    ld a, h").unwrap();
                        writeln!(self.asm, "    and d").unwrap();
                        writeln!(self.asm, "    ld h, a").unwrap();
                    }
                    IrBinOp::BitOr => {
                        writeln!(self.asm, "    ld a, l").unwrap();
                        writeln!(self.asm, "    or e").unwrap();
                        writeln!(self.asm, "    ld l, a").unwrap();
                        writeln!(self.asm, "    ld a, h").unwrap();
                        writeln!(self.asm, "    or d").unwrap();
                        writeln!(self.asm, "    ld h, a").unwrap();
                    }
                    IrBinOp::BitXor => {
                        writeln!(self.asm, "    ld a, l").unwrap();
                        writeln!(self.asm, "    xor e").unwrap();
                        writeln!(self.asm, "    ld l, a").unwrap();
                        writeln!(self.asm, "    ld a, h").unwrap();
                        writeln!(self.asm, "    xor d").unwrap();
                        writeln!(self.asm, "    ld h, a").unwrap();
                    }
                    IrBinOp::Div => {
                        let helper = if signed {
                            self.runtime_needs.sdiv16 = true;
                            "__aun_sdiv16"
                        } else {
                            self.runtime_needs.udiv16 = true;
                            "__aun_udiv16"
                        };
                        writeln!(self.asm, "    call {helper}").unwrap();
                        self.val_in_reg
                            .retain(|_, &mut vr| !crate::regalloc::RegMask::BC.intersects(vr.mask()));
                    }
                    IrBinOp::Rem => {
                        let helper = if signed {
                            self.runtime_needs.sdiv16 = true;
                            "__aun_srem16"
                        } else {
                            self.runtime_needs.udiv16 = true;
                            "__aun_urem16"
                        };
                        writeln!(self.asm, "    call {helper}").unwrap();
                        self.val_in_reg
                            .retain(|_, &mut vr| !crate::regalloc::RegMask::BC.intersects(vr.mask()));
                    }
                    other => unreachable!("unhandled 2-byte binop {other:?}"),
                }
                writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
            }
            4 => {
                let da2_mem = self.fmt_ram_mem(da.wrapping_add(2));
                match op {
                    IrBinOp::Add => {
                        self.emit_load_word32(lhs, false, "hl", false);
                        self.emit_load_word32(rhs, false, "de", false);
                        writeln!(self.asm, "    add hl, de").unwrap();
                        writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                        self.emit_load_word32(lhs, true, "hl", false);
                        self.emit_load_word32(rhs, true, "de", false);
                        writeln!(self.asm, "    adc hl, de").unwrap();
                        writeln!(self.asm, "    ld {da2_mem}, hl").unwrap();
                    }
                    IrBinOp::Sub => {
                        self.emit_load_word32(lhs, false, "hl", false);
                        self.emit_load_word32(rhs, false, "de", false);
                        writeln!(self.asm, "    or a").unwrap();
                        writeln!(self.asm, "    sbc hl, de").unwrap();
                        writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                        self.emit_load_word32(lhs, true, "hl", false);
                        self.emit_load_word32(rhs, true, "de", false);
                        writeln!(self.asm, "    sbc hl, de").unwrap();
                        writeln!(self.asm, "    ld {da2_mem}, hl").unwrap();
                    }
                    IrBinOp::BitAnd | IrBinOp::BitOr | IrBinOp::BitXor => {
                        let op_mnem = match op {
                            IrBinOp::BitAnd => "and",
                            IrBinOp::BitOr => "or",
                            IrBinOp::BitXor => "xor",
                            _ => unreachable!(),
                        };
                        for (hi, off) in [(false, 0u16), (true, 2u16)] {
                            self.emit_load_word32(lhs, hi, "hl", false);
                            self.emit_load_word32(rhs, hi, "de", false);
                            writeln!(self.asm, "    ld a, l").unwrap();
                            writeln!(self.asm, "    {op_mnem} e").unwrap();
                            writeln!(self.asm, "    ld l, a").unwrap();
                            writeln!(self.asm, "    ld a, h").unwrap();
                            writeln!(self.asm, "    {op_mnem} d").unwrap();
                            writeln!(self.asm, "    ld h, a").unwrap();
                            let d_off_mem = self.fmt_ram_mem(da.wrapping_add(off));
                            writeln!(self.asm, "    ld {d_off_mem}, hl").unwrap();
                        }
                    }
                    IrBinOp::Shl | IrBinOp::Shr => {
                        if let Some(&k_raw) = self.const_vals.get(&rhs) {
                            let k = k_raw.max(0);
                            if k == 0 {
                                if let Some(&c) = self.const_vals.get(&lhs) {
                                    self.emit_const32_to(da, true, c);
                                } else {
                                    self.emit_copy32(la, true, da, true);
                                }
                            } else if k >= 32 {
                                if op == IrBinOp::Shr && signed {
                                    self.emit_load_hlde_from(lhs);
                                    writeln!(self.asm, "    ld a, h").unwrap();
                                    writeln!(self.asm, "    add a, a").unwrap();
                                    writeln!(self.asm, "    sbc a, a").unwrap();
                                    writeln!(self.asm, "    ld e, a").unwrap();
                                    writeln!(self.asm, "    ld d, a").unwrap();
                                    writeln!(self.asm, "    ld l, a").unwrap();
                                    writeln!(self.asm, "    ld h, a").unwrap();
                                    self.emit_store_hlde_to(da, true);
                                } else {
                                    self.emit_const32_to(da, true, 0);
                                }
                            } else if k == 16 {
                                if op == IrBinOp::Shl {
                                    writeln!(self.asm, "    ld hl, 0").unwrap();
                                    writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                                    self.emit_load_word32(lhs, false, "hl", false);
                                    writeln!(self.asm, "    ld {da2_mem}, hl").unwrap();
                                } else {
                                    self.emit_load_word32(lhs, true, "de", false);
                                    writeln!(self.asm, "    ld {da_mem}, de").unwrap();
                                    if signed {
                                        writeln!(self.asm, "    ld a, d").unwrap();
                                        writeln!(self.asm, "    add a, a").unwrap();
                                        writeln!(self.asm, "    sbc a, a").unwrap();
                                        writeln!(self.asm, "    ld l, a").unwrap();
                                        writeln!(self.asm, "    ld h, a").unwrap();
                                    } else {
                                        writeln!(self.asm, "    ld hl, 0").unwrap();
                                    }
                                    writeln!(self.asm, "    ld {da2_mem}, hl").unwrap();
                                }
                            } else {
                                let lp = self.fresh_label("sh32_lp");
                                self.emit_load_hlde_from(lhs);
                                writeln!(self.asm, "    ld b, {k}").unwrap();
                                writeln!(self.asm, "{lp}:").unwrap();
                                if op == IrBinOp::Shl {
                                    writeln!(self.asm, "    sla e").unwrap();
                                    writeln!(self.asm, "    rl d").unwrap();
                                    writeln!(self.asm, "    adc hl, hl").unwrap();
                                } else {
                                    if signed {
                                        writeln!(self.asm, "    sra h").unwrap();
                                    } else {
                                        writeln!(self.asm, "    srl h").unwrap();
                                    }
                                    writeln!(self.asm, "    rr l").unwrap();
                                    writeln!(self.asm, "    rr d").unwrap();
                                    writeln!(self.asm, "    rr e").unwrap();
                                }
                                writeln!(self.asm, "    djnz {lp}").unwrap();
                                self.emit_store_hlde_to(da, true);
                            }
                        } else {
                            let lp = self.fresh_label("sh32_lp");
                            let dn = self.fresh_label("sh32_dn");
                            self.emit_load_hlde_from(lhs);
                            self.emit_load_a_from(rhs);
                            writeln!(self.asm, "    or a").unwrap();
                            writeln!(self.asm, "    jr z, {dn}").unwrap();
                            writeln!(self.asm, "    ld b, a").unwrap();
                            writeln!(self.asm, "{lp}:").unwrap();
                            if op == IrBinOp::Shl {
                                writeln!(self.asm, "    sla e").unwrap();
                                writeln!(self.asm, "    rl d").unwrap();
                                writeln!(self.asm, "    adc hl, hl").unwrap();
                            } else {
                                if signed {
                                    writeln!(self.asm, "    sra h").unwrap();
                                } else {
                                    writeln!(self.asm, "    srl h").unwrap();
                                }
                                writeln!(self.asm, "    rr l").unwrap();
                                writeln!(self.asm, "    rr d").unwrap();
                                writeln!(self.asm, "    rr e").unwrap();
                            }
                            writeln!(self.asm, "    djnz {lp}").unwrap();
                            writeln!(self.asm, "{dn}:").unwrap();
                            self.emit_store_hlde_to(da, true);
                        }
                    }
                    IrBinOp::Mul | IrBinOp::Div | IrBinOp::Rem => {
                        self.emit_load_hlde_from(rhs);
                        writeln!(self.asm, "    push hl").unwrap();
                        writeln!(self.asm, "    push de").unwrap();
                        self.emit_load_hlde_from(lhs);
                        let helper = match (op, signed) {
                            (IrBinOp::Mul, _) => {
                                self.runtime_needs.mul32 = true;
                                "__aun_mul32"
                            }
                            (IrBinOp::Div, false) => {
                                self.runtime_needs.udiv32 = true;
                                "__aun_udiv32"
                            }
                            (IrBinOp::Rem, false) => {
                                self.runtime_needs.udiv32 = true;
                                "__aun_urem32"
                            }
                            (IrBinOp::Div, true) => {
                                self.runtime_needs.sdiv32 = true;
                                "__aun_sdiv32"
                            }
                            (IrBinOp::Rem, true) => {
                                self.runtime_needs.sdiv32 = true;
                                "__aun_srem32"
                            }
                            _ => unreachable!(),
                        };
                        writeln!(self.asm, "    call {helper}").unwrap();
                        writeln!(self.asm, "    pop bc").unwrap();
                        writeln!(self.asm, "    pop bc").unwrap();
                        self.val_in_reg
                            .retain(|_, &mut vr| !crate::regalloc::RegMask::BC.intersects(vr.mask()));
                        self.emit_store_hlde_to(da, true);
                    }
                    other => unreachable!("unhandled 4-byte binop {other:?}"),
                }
            }
            _ => {}
        }
    }
}
