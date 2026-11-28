use super::runtime::emit_mul_hl_const;
use super::{BitPackAction, EmitCtx, U8PackTerm};
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

/// Emits constant left shift of `A` by `k in 0..=8` (clobbers `F`; preserves `BC, DE, HL, IX`).
fn emit_shift_a_left_const(asm: &mut String, k: u8) {
    emit_shift_a_left_const_for_bits(asm, k, 0x00FF);
}

/// Emits constant left shift of `A` by `k in 0..=8`, omitting the `and` mask after `rrca`
/// when `eff_x_bits` proves all bits shifted past bit 7 are already zero.
fn emit_shift_a_left_const_for_bits(asm: &mut String, k: u8, eff_x_bits: u16) {
    match k {
        0 => {}
        1..=4 => {
            for _ in 0..k {
                writeln!(asm, "    add a, a").unwrap();
            }
        }
        5..=7 => {
            for _ in 0..(8 - k) {
                writeln!(asm, "    rrca").unwrap();
            }
            if (eff_x_bits & !(0xFFu16 >> k)) != 0 {
                writeln!(asm, "    and 0x{:02X}", 0xFFu8 << k).unwrap();
            }
        }
        8.. => writeln!(asm, "    xor a").unwrap(),
    }
}

fn emit_finish_byte_into_reg(asm: &mut String, reg_dst: &str, reg_src: &str, init: bool, c: u8) {
    match (init, c) {
        (false, 0) => writeln!(asm, "    ld {reg_dst}, 0").unwrap(),
        (false, _) | (true, 0xFF) => writeln!(asm, "    ld {reg_dst}, 0x{c:02X}").unwrap(),
        (true, 0) => writeln!(asm, "    ld {reg_dst}, {reg_src}").unwrap(),
        (true, _) => {
            writeln!(asm, "    ld a, {reg_src}").unwrap();
            writeln!(asm, "    or 0x{c:02X}").unwrap();
            writeln!(asm, "    ld {reg_dst}, a").unwrap();
        }
    }
}

fn emit_apply_const_to_hl_byte(asm: &mut String, reg: &str, c: u8) {
    match c {
        0 => {}
        0xFF => writeln!(asm, "    ld {reg}, 0xFF").unwrap(),
        _ => {
            writeln!(asm, "    ld a, {reg}").unwrap();
            writeln!(asm, "    or 0x{c:02X}").unwrap();
            writeln!(asm, "    ld {reg}, a").unwrap();
        }
    }
}

/// Computes `(A as u16) << k` (`k in 1..=7`) into `HL`, optionally ORing with `DE` (`HL = DE | (A << k)`).
fn emit_u8_shl16_from_a(asm: &mut String, k: u8, or_into_de: bool) {
    match k {
        1..=4 => {
            writeln!(asm, "    ld l, a").unwrap();
            writeln!(asm, "    ld h, 0").unwrap();
            for _ in 0..k {
                writeln!(asm, "    add hl, hl").unwrap();
            }
            if or_into_de {
                writeln!(asm, "    ld a, l").unwrap();
                writeln!(asm, "    or e").unwrap();
                writeln!(asm, "    ld l, a").unwrap();
                writeln!(asm, "    ld a, h").unwrap();
                writeln!(asm, "    or d").unwrap();
                writeln!(asm, "    ld h, a").unwrap();
            }
        }
        5 | 6 => {
            for _ in 0..(8 - k) {
                writeln!(asm, "    rrca").unwrap();
            }
            writeln!(asm, "    ld l, a").unwrap();
            writeln!(asm, "    and 0x{:02X}", (1u8 << k) - 1).unwrap();
            writeln!(asm, "    ld h, a").unwrap();
            writeln!(asm, "    ld a, l").unwrap();
            writeln!(asm, "    and 0x{:02X}", 0xFFu8 << k).unwrap();
            if or_into_de {
                writeln!(asm, "    or e").unwrap();
                writeln!(asm, "    ld l, a").unwrap();
                writeln!(asm, "    ld a, h").unwrap();
                writeln!(asm, "    or d").unwrap();
                writeln!(asm, "    ld h, a").unwrap();
            } else {
                writeln!(asm, "    ld l, a").unwrap();
            }
        }
        7 => {
            writeln!(asm, "    srl a").unwrap();
            writeln!(asm, "    ld h, a").unwrap();
            writeln!(asm, "    ld a, 0").unwrap();
            writeln!(asm, "    rra").unwrap();
            if or_into_de {
                writeln!(asm, "    or e").unwrap();
                writeln!(asm, "    ld l, a").unwrap();
                writeln!(asm, "    ld a, h").unwrap();
                writeln!(asm, "    or d").unwrap();
                writeln!(asm, "    ld h, a").unwrap();
            } else {
                writeln!(asm, "    ld l, a").unwrap();
            }
        }
        _ => unreachable!(),
    }
}

/// Emits unrolled constant shift on `HL` for `k in 0..=16` (clobbers `A, F`; preserves `BC, DE, IX`).
pub(super) fn emit_shift16_hl_const(asm: &mut String, op: IrBinOp, signed: bool, k: u8) {
    match (op, k) {
        (_, 0) => {}
        (IrBinOp::Shl, 16..) => writeln!(asm, "    ld hl, 0").unwrap(),
        (IrBinOp::Shl, 1..=6) => {
            for _ in 0..k {
                writeln!(asm, "    add hl, hl").unwrap();
            }
        }
        (IrBinOp::Shl, 7) => {
            // (HL << 7): H = (H<<7)|(L>>1), L = (L<<7)
            writeln!(asm, "    ld a, h").unwrap();
            writeln!(asm, "    rra").unwrap();
            writeln!(asm, "    ld a, l").unwrap();
            writeln!(asm, "    rra").unwrap();
            writeln!(asm, "    ld h, a").unwrap();
            writeln!(asm, "    ld a, 0").unwrap();
            writeln!(asm, "    rra").unwrap();
            writeln!(asm, "    ld l, a").unwrap();
        }
        (IrBinOp::Shl, 8) => {
            writeln!(asm, "    ld h, l").unwrap();
            writeln!(asm, "    ld l, 0").unwrap();
        }
        (IrBinOp::Shl, 9..=15) => {
            writeln!(asm, "    ld h, l").unwrap();
            writeln!(asm, "    ld l, 0").unwrap();
            for _ in 0..(k - 8) {
                writeln!(asm, "    sla h").unwrap();
            }
        }
        (IrBinOp::Shr, 1..=3) => {
            for _ in 0..k {
                if signed {
                    writeln!(asm, "    sra h").unwrap();
                } else {
                    writeln!(asm, "    srl h").unwrap();
                }
                writeln!(asm, "    rr l").unwrap();
            }
        }
        (IrBinOp::Shr, 4..=6) => {
            let n = 8 - k;
            if signed {
                writeln!(asm, "    ld a, h").unwrap();
                writeln!(asm, "    add a, a").unwrap();
                writeln!(asm, "    sbc a, a").unwrap();
            } else {
                writeln!(asm, "    xor a").unwrap();
            }
            for _ in 0..n {
                writeln!(asm, "    add hl, hl").unwrap();
                writeln!(asm, "    rla").unwrap();
            }
            writeln!(asm, "    ld l, h").unwrap();
            writeln!(asm, "    ld h, a").unwrap();
        }
        (IrBinOp::Shr, 7) => {
            writeln!(asm, "    add hl, hl").unwrap();
            writeln!(asm, "    ld l, h").unwrap();
            if signed {
                writeln!(asm, "    sbc a, a").unwrap();
            } else {
                writeln!(asm, "    ld a, 0").unwrap();
                writeln!(asm, "    rla").unwrap();
            }
            writeln!(asm, "    ld h, a").unwrap();
        }
        (IrBinOp::Shr, 8..=14) => {
            writeln!(asm, "    ld l, h").unwrap();
            if signed {
                writeln!(asm, "    ld a, h").unwrap();
                writeln!(asm, "    add a, a").unwrap();
                writeln!(asm, "    sbc a, a").unwrap();
                writeln!(asm, "    ld h, a").unwrap();
                for _ in 0..(k - 8) {
                    writeln!(asm, "    sra l").unwrap();
                }
            } else {
                writeln!(asm, "    ld h, 0").unwrap();
                for _ in 0..(k - 8) {
                    writeln!(asm, "    srl l").unwrap();
                }
            }
        }
        (IrBinOp::Shr, 15) if !signed => {
            writeln!(asm, "    add hl, hl").unwrap();
            writeln!(asm, "    ld a, 0").unwrap();
            writeln!(asm, "    rla").unwrap();
            writeln!(asm, "    ld l, a").unwrap();
            writeln!(asm, "    ld h, 0").unwrap();
        }
        (IrBinOp::Shr, 16..) if !signed => {
            writeln!(asm, "    ld hl, 0").unwrap();
        }
        (IrBinOp::Shr, 15..) => {
            writeln!(asm, "    ld a, h").unwrap();
            writeln!(asm, "    add a, a").unwrap();
            writeln!(asm, "    sbc a, a").unwrap();
            writeln!(asm, "    ld l, a").unwrap();
            writeln!(asm, "    ld h, a").unwrap();
        }
        _ => unreachable!(),
    }
}

impl<'a> EmitCtx<'a> {
    fn emit_load_hi_byte_a_from(&mut self, v: ValueId) {
        if let Some(&c) = self.const_vals.get(&v) {
            let b = ((c as u16) >> 8) as u8;
            if b == 0 {
                writeln!(self.asm, "    xor a").unwrap();
            } else {
                writeln!(self.asm, "    ld a, 0x{b:02X}").unwrap();
            }
        } else if self.val_in_reg.get(&v) == Some(&RegLoc::BC) {
            writeln!(self.asm, "    ld a, b").unwrap();
        } else if let Some(&(disp, _)) = self.ix_val_loads.get(&v) {
            writeln!(self.asm, "    ld a, (ix + {})", disp + 1).unwrap();
        } else {
            let hi_mem = self.fmt_val_addr_mem(v, 1);
            writeln!(self.asm, "    ld a, {hi_mem}").unwrap();
        }
    }

    fn emit_load_masked_u8_term_into_a(&mut self, term: U8PackTerm) -> u16 {
        let orig_bits = self.possible_bits(term.x_v) & 0x00FF;
        let eff_bits = orig_bits & (term.mask8 as u16);
        if eff_bits == 0 || term.k >= 16 {
            writeln!(self.asm, "    xor a").unwrap();
            return 0;
        }
        if let Some(&c) = self.const_vals.get(&term.x_v) {
            let b = ((c as u16) & (term.mask8 as u16)) as u8;
            if b == 0 {
                writeln!(self.asm, "    xor a").unwrap();
            } else {
                writeln!(self.asm, "    ld a, 0x{b:02X}").unwrap();
            }
            return eff_bits;
        }
        self.emit_load_a_from(term.x_v);
        if (orig_bits & !(term.mask8 as u16)) != 0 {
            if term.mask8 == 0 {
                writeln!(self.asm, "    xor a").unwrap();
            } else {
                writeln!(self.asm, "    and 0x{:02X}", term.mask8).unwrap();
            }
        }
        eff_bits
    }

    fn emit_u8_pack_term_into_hl(&mut self, term: U8PackTerm) {
        let eff_bits = self.possible_bits(term.x_v) & 0x00FF & (term.mask8 as u16);
        let term_bits = if term.k >= 16 {
            0
        } else {
            eff_bits.wrapping_shl(term.k as u32)
        };
        if term_bits == 0 {
            writeln!(self.asm, "    ld hl, 0").unwrap();
        } else if (term_bits & 0xFF00) == 0 {
            let eff = self.emit_load_masked_u8_term_into_a(term);
            emit_shift_a_left_const_for_bits(self.asm, term.k, eff);
            writeln!(self.asm, "    ld l, a").unwrap();
            writeln!(self.asm, "    ld h, 0").unwrap();
        } else if (term_bits & 0x00FF) == 0 {
            let eff = self.emit_load_masked_u8_term_into_a(term);
            emit_shift_a_left_const_for_bits(self.asm, term.k - 8, eff);
            writeln!(self.asm, "    ld h, a").unwrap();
            writeln!(self.asm, "    ld l, 0").unwrap();
        } else {
            let _eff = self.emit_load_masked_u8_term_into_a(term);
            emit_u8_shl16_from_a(self.asm, term.k, false);
        }
    }

    fn emit_load_u8_term_direct_reg(&mut self, reg_dst: &str, term: U8PackTerm) -> bool {
        let orig_bits = self.possible_bits(term.x_v) & 0x00FF;
        if (orig_bits & !(term.mask8 as u16)) != 0 || self.const_vals.contains_key(&term.x_v) {
            return false;
        }
        if let Some(&reg) = self.val_in_reg.get(&term.x_v) {
            writeln!(self.asm, "    ld {reg_dst}, {}", reg.reg8_name()).unwrap();
            true
        } else if let Some(&(disp, _)) = self.ix_val_loads.get(&term.x_v) {
            writeln!(self.asm, "    ld {reg_dst}, (ix + {disp})").unwrap();
            true
        } else {
            false
        }
    }

    fn emit_u8_term_into_de(&mut self, term: U8PackTerm) {
        let x_bits = self.possible_bits(term.x_v) & 0x00FF & (term.mask8 as u16);
        let term_bits = if term.k >= 16 {
            0
        } else {
            x_bits.wrapping_shl(term.k as u32)
        };
        if term_bits == 0 {
            return;
        }
        if (term_bits & 0xFF00) == 0 {
            if !self.de_e_init && term.k == 0 && self.emit_load_u8_term_direct_reg("e", term) {
                self.de_e_init = true;
                return;
            }
            let eff = self.emit_load_masked_u8_term_into_a(term);
            emit_shift_a_left_const_for_bits(self.asm, term.k, eff);
            if self.de_e_init {
                writeln!(self.asm, "    or e").unwrap();
            }
            writeln!(self.asm, "    ld e, a").unwrap();
            self.de_e_init = true;
        } else if (term_bits & 0x00FF) == 0 {
            if !self.de_d_init && term.k == 8 && self.emit_load_u8_term_direct_reg("d", term) {
                self.de_d_init = true;
                return;
            }
            let eff = self.emit_load_masked_u8_term_into_a(term);
            emit_shift_a_left_const_for_bits(self.asm, term.k - 8, eff);
            if self.de_d_init {
                writeln!(self.asm, "    or d").unwrap();
            }
            writeln!(self.asm, "    ld d, a").unwrap();
            self.de_d_init = true;
        } else {
            let _eff = self.emit_load_masked_u8_term_into_a(term);
            match (self.de_d_init, self.de_e_init) {
                (false, false) => {
                    emit_u8_shl16_from_a(self.asm, term.k, false);
                    writeln!(self.asm, "    ex de, hl").unwrap();
                }
                (true, true) => {
                    emit_u8_shl16_from_a(self.asm, term.k, true);
                    writeln!(self.asm, "    ex de, hl").unwrap();
                }
                (false, true) => {
                    emit_u8_shl16_from_a(self.asm, term.k, false);
                    writeln!(self.asm, "    ld a, l").unwrap();
                    writeln!(self.asm, "    or e").unwrap();
                    writeln!(self.asm, "    ld e, a").unwrap();
                    writeln!(self.asm, "    ld d, h").unwrap();
                }
                (true, false) => {
                    emit_u8_shl16_from_a(self.asm, term.k, false);
                    writeln!(self.asm, "    ld a, h").unwrap();
                    writeln!(self.asm, "    or d").unwrap();
                    writeln!(self.asm, "    ld d, a").unwrap();
                    writeln!(self.asm, "    ld e, l").unwrap();
                }
            }
            self.de_d_init = true;
            self.de_e_init = true;
        }
    }

    fn emit_other16_into_de(&mut self, val: ValueId) {
        let bits = self.possible_bits(val);
        if bits == 0 {
            return;
        }
        let non_addr = !self.const_addrs.contains_key(&val) && !self.rom_addrs.contains_key(&val);
        if (bits & 0xFF00) == 0 && non_addr {
            self.emit_load_a_from(val);
            if self.de_e_init {
                writeln!(self.asm, "    or e").unwrap();
            }
            writeln!(self.asm, "    ld e, a").unwrap();
            self.de_e_init = true;
        } else if (bits & 0x00FF) == 0
            && non_addr
            && self.ix_val_loads.get(&val).is_none_or(|&(d, _)| d <= 126)
        {
            self.emit_load_hi_byte_a_from(val);
            if self.de_d_init {
                writeln!(self.asm, "    or d").unwrap();
            }
            writeln!(self.asm, "    ld d, a").unwrap();
            self.de_d_init = true;
        } else if !self.de_d_init && !self.de_e_init {
            self.emit_load_de_from(val);
            self.de_d_init = true;
            self.de_e_init = true;
        } else {
            self.emit_load_hl_from(val);
            if self.de_e_init {
                writeln!(self.asm, "    ld a, l").unwrap();
                writeln!(self.asm, "    or e").unwrap();
                writeln!(self.asm, "    ld e, a").unwrap();
            } else {
                writeln!(self.asm, "    ld e, l").unwrap();
            }
            if self.de_d_init {
                writeln!(self.asm, "    ld a, h").unwrap();
                writeln!(self.asm, "    or d").unwrap();
                writeln!(self.asm, "    ld d, a").unwrap();
            } else {
                writeln!(self.asm, "    ld d, h").unwrap();
            }
            self.de_d_init = true;
            self.de_e_init = true;
        }
    }

    pub(super) fn emit_bitpack_action(&mut self, dst: ValueId, action: BitPackAction) {
        match action {
            BitPackAction::U8Term { pre_other, term } => {
                if let Some(other) = pre_other {
                    self.emit_other16_into_de(other);
                }
                self.emit_u8_term_into_de(term);
            }
            BitPackAction::Other16 { val, val2 } => {
                self.emit_other16_into_de(val);
                if let Some(v2) = val2 {
                    self.emit_other16_into_de(v2);
                }
            }
            BitPackAction::Finish {
                tail_u8,
                tail_other1,
                tail_other2,
                const_lo,
                const_hi,
            } => {
                if let Some(term) = tail_u8 {
                    let x_bits = self.possible_bits(term.x_v) & 0x00FF & (term.mask8 as u16);
                    let term_bits = if term.k >= 16 {
                        0
                    } else {
                        x_bits.wrapping_shl(term.k as u32)
                    };
                    if term_bits != 0 && (term_bits & 0xFF00) == 0 {
                        if !self.de_e_init
                            && const_lo == 0
                            && term.k == 0
                            && self.emit_load_u8_term_direct_reg("l", term)
                        {
                            // L loaded directly
                        } else {
                            let eff = self.emit_load_masked_u8_term_into_a(term);
                            emit_shift_a_left_const_for_bits(self.asm, term.k, eff);
                            if self.de_e_init {
                                writeln!(self.asm, "    or e").unwrap();
                            }
                            if const_lo == 0xFF {
                                writeln!(self.asm, "    ld l, 0xFF").unwrap();
                            } else {
                                if const_lo != 0 {
                                    writeln!(self.asm, "    or 0x{const_lo:02X}").unwrap();
                                }
                                writeln!(self.asm, "    ld l, a").unwrap();
                            }
                        }
                        emit_finish_byte_into_reg(self.asm, "h", "d", self.de_d_init, const_hi);
                        self.de_d_init = false;
                        self.de_e_init = false;
                        self.emit_store_dst_hl(dst);
                        return;
                    }
                    if term_bits != 0 && (term_bits & 0x00FF) == 0 {
                        if !self.de_d_init
                            && const_hi == 0
                            && term.k == 8
                            && self.emit_load_u8_term_direct_reg("h", term)
                        {
                            // H loaded directly
                        } else {
                            let eff = self.emit_load_masked_u8_term_into_a(term);
                            emit_shift_a_left_const_for_bits(self.asm, term.k - 8, eff);
                            if self.de_d_init {
                                writeln!(self.asm, "    or d").unwrap();
                            }
                            if const_hi == 0xFF {
                                writeln!(self.asm, "    ld h, 0xFF").unwrap();
                            } else {
                                if const_hi != 0 {
                                    writeln!(self.asm, "    or 0x{const_hi:02X}").unwrap();
                                }
                                writeln!(self.asm, "    ld h, a").unwrap();
                            }
                        }
                        emit_finish_byte_into_reg(self.asm, "l", "e", self.de_e_init, const_lo);
                        self.de_d_init = false;
                        self.de_e_init = false;
                        self.emit_store_dst_hl(dst);
                        return;
                    }
                    if term_bits != 0 {
                        let _eff = self.emit_load_masked_u8_term_into_a(term);
                        match (self.de_d_init, self.de_e_init) {
                            (false, false) => emit_u8_shl16_from_a(self.asm, term.k, false),
                            (true, true) => emit_u8_shl16_from_a(self.asm, term.k, true),
                            (false, true) => {
                                writeln!(self.asm, "    ld d, 0").unwrap();
                                emit_u8_shl16_from_a(self.asm, term.k, true);
                            }
                            (true, false) => {
                                writeln!(self.asm, "    ld e, 0").unwrap();
                                emit_u8_shl16_from_a(self.asm, term.k, true);
                            }
                        }
                        emit_apply_const_to_hl_byte(self.asm, "l", const_lo);
                        emit_apply_const_to_hl_byte(self.asm, "h", const_hi);
                        self.de_d_init = false;
                        self.de_e_init = false;
                        self.emit_store_dst_hl(dst);
                        return;
                    }
                } else if let Some(other1) = tail_other1 {
                    if tail_other2.is_none()
                        && !self.de_d_init
                        && !self.de_e_init
                        && const_lo == 0
                        && const_hi == 0
                    {
                        self.emit_load_hl_from(other1);
                        self.de_d_init = false;
                        self.de_e_init = false;
                        self.emit_store_dst_hl(dst);
                        return;
                    }
                    self.emit_other16_into_de(other1);
                    if let Some(other2) = tail_other2 {
                        self.emit_other16_into_de(other2);
                    }
                }

                if self.de_d_init && self.de_e_init && const_lo == 0 && const_hi == 0 {
                    writeln!(self.asm, "    ex de, hl").unwrap();
                } else if !self.de_d_init && !self.de_e_init {
                    let w = ((const_hi as u16) << 8) | (const_lo as u16);
                    writeln!(self.asm, "    ld hl, 0x{w:04X}").unwrap();
                } else {
                    emit_finish_byte_into_reg(self.asm, "l", "e", self.de_e_init, const_lo);
                    emit_finish_byte_into_reg(self.asm, "h", "d", self.de_d_init, const_hi);
                }
                self.de_d_init = false;
                self.de_e_init = false;
                self.emit_store_dst_hl(dst);
            }
        }
    }

    fn emit_signed_lt_branch(
        &mut self,
        fname: &str,
        want_lt: bool,
        then_bb: BlockId,
        else_bb: BlockId,
        next_bb_id: Option<BlockId>,
    ) {
        let ovf_lbl = self.fresh_label("cmp_ovf");
        if next_bb_id == Some(then_bb) {
            let no_ovf_false = if want_lt { "p" } else { "m" };
            let ovf_false = if want_lt { "m" } else { "p" };
            writeln!(self.asm, "    jp pe, {ovf_lbl}").unwrap();
            writeln!(self.asm, "    jp {no_ovf_false}, {fname}_bb{else_bb}").unwrap();
            writeln!(self.tail_cold_stubs, "{ovf_lbl}:").unwrap();
            writeln!(
                self.tail_cold_stubs,
                "    jp {ovf_false}, {fname}_bb{else_bb}"
            )
            .unwrap();
            writeln!(self.tail_cold_stubs, "    jp {fname}_bb{then_bb}").unwrap();
        } else {
            let no_ovf_true = if want_lt { "m" } else { "p" };
            let ovf_true = if want_lt { "p" } else { "m" };
            writeln!(self.asm, "    jp pe, {ovf_lbl}").unwrap();
            writeln!(self.asm, "    jp {no_ovf_true}, {fname}_bb{then_bb}").unwrap();
            if next_bb_id != Some(else_bb) {
                writeln!(self.asm, "    jp {fname}_bb{else_bb}").unwrap();
            }
            writeln!(self.tail_cold_stubs, "{ovf_lbl}:").unwrap();
            writeln!(
                self.tail_cold_stubs,
                "    jp {ovf_true}, {fname}_bb{then_bb}"
            )
            .unwrap();
            writeln!(self.tail_cold_stubs, "    jp {fname}_bb{else_bb}").unwrap();
        }
    }

    fn emit_signed_lt_bool(&mut self, want_lt: bool, dst: ValueId) {
        let ovf_lbl = self.fresh_label("cmp_ovf");
        let false_lbl = self.fresh_label("cmp_f");
        let done_lbl = self.fresh_label("cmp");
        let no_ovf_true = if want_lt { "m" } else { "p" };
        let ovf_true = if want_lt { "p" } else { "m" };
        writeln!(self.asm, "    ld a, 1").unwrap();
        writeln!(self.asm, "    jp pe, {ovf_lbl}").unwrap();
        writeln!(self.asm, "    jp {no_ovf_true}, {done_lbl}").unwrap();
        writeln!(self.asm, "{false_lbl}:").unwrap();
        writeln!(self.asm, "    xor a").unwrap();
        writeln!(self.asm, "{done_lbl}:").unwrap();
        self.emit_store_dst_a(dst);

        writeln!(self.tail_cold_stubs, "{ovf_lbl}:").unwrap();
        writeln!(self.tail_cold_stubs, "    jp {ovf_true}, {done_lbl}").unwrap();
        writeln!(self.tail_cold_stubs, "    jp {false_lbl}").unwrap();
    }

    /// Matches signed `i16` ordered comparisons reducible to an exact signed high-byte `H`
    /// comparison against `h_k` (`want_lt`: `H < h_k`, otherwise `H >= h_k`):
    /// - `x < K` / `x >= K` when `(K & 0x00FF) == 0` (`h_k = (K >> 8) as i8`)
    /// - `x <= M` / `x > M` when `M < i16::MAX` and `((M + 1) & 0x00FF) == 0` (`h_k = ((M + 1) >> 8) as i8`)
    fn match_signed_i16_high_byte_cmp(
        &self,
        op: IrBinOp,
        lhs: ValueId,
        rhs: ValueId,
    ) -> Option<(ValueId, bool, i8)> {
        let (norm_op, var_v, c_raw) =
            if self.const_vals.contains_key(&lhs) && !self.const_vals.contains_key(&rhs) {
                let flipped = match op {
                    IrBinOp::Eq => IrBinOp::Eq,
                    IrBinOp::Ne => IrBinOp::Ne,
                    IrBinOp::Lt => IrBinOp::Gt,
                    IrBinOp::Le => IrBinOp::Ge,
                    IrBinOp::Gt => IrBinOp::Lt,
                    IrBinOp::Ge => IrBinOp::Le,
                    _ => return None,
                };
                (flipped, rhs, *self.const_vals.get(&lhs)?)
            } else {
                (op, lhs, *self.const_vals.get(&rhs)?)
            };
        if self.const_addrs.contains_key(&var_v) || self.rom_addrs.contains_key(&var_v) {
            return None;
        }
        let c_i16 = c_raw as i16;
        match norm_op {
            IrBinOp::Lt | IrBinOp::Ge if (c_i16 as u16) & 0x00FF == 0 => {
                Some((var_v, norm_op == IrBinOp::Lt, (c_i16 >> 8) as i8))
            }
            IrBinOp::Le | IrBinOp::Gt
                if c_i16 < i16::MAX && ((c_i16 + 1) as u16) & 0x00FF == 0 =>
            {
                Some((var_v, norm_op == IrBinOp::Le, ((c_i16 + 1) >> 8) as i8))
            }
            _ => None,
        }
    }

    fn emit_signed_i16_high_byte_xor_cp(&mut self, var_v: ValueId, h_k: i8) {
        if let Some(&cv) = self.const_vals.get(&var_v) {
            let hb = (((cv as i16) >> 8) as u8) ^ 0x80;
            writeln!(self.asm, "    ld a, 0x{hb:02X}").unwrap();
        } else {
            if self.val_in_reg.get(&var_v) == Some(&RegLoc::BC) {
                writeln!(self.asm, "    ld a, b").unwrap();
            } else if let Some(&(disp, _)) = self.ix_val_loads.get(&var_v)
                && let Some(hi_d) = disp.checked_add(1)
            {
                writeln!(self.asm, "    ld a, (ix + {hi_d})").unwrap();
            } else {
                let hi_mem = self.fmt_val_addr_mem(var_v, 1);
                writeln!(self.asm, "    ld a, {hi_mem}").unwrap();
            }
            writeln!(self.asm, "    xor 0x80").unwrap();
        }
        let cb = (h_k as u8) ^ 0x80;
        writeln!(self.asm, "    cp 0x{cb:02X}").unwrap();
    }

    fn emit_bool_from_cc(&mut self, cc: &str, dst: ValueId) {
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
        self.emit_store_dst_a(dst);
    }

    pub(super) fn emit_load_word32(&mut self, v: ValueId, hi: bool, reg: &str, flip_msb: bool) {
        if let Some(&c) = self.const_vals.get(&v) {
            let u = c as u32;
            let mut w = if hi { (u >> 16) as u16 } else { u as u16 };
            if hi && flip_msb {
                w ^= 0x8000;
            }
            writeln!(self.asm, "    ld {reg}, 0x{w:04X}").unwrap();
        } else if let Some(&(disp, _)) = self.ix_val_loads.get(&v) {
            let d0 = if hi { disp + 2 } else { disp };
            let (lo_r, hi_r) = if reg == "hl" { ("l", "h") } else { ("e", "d") };
            writeln!(self.asm, "    ld {lo_r}, (ix + {d0})").unwrap();
            writeln!(self.asm, "    ld {hi_r}, (ix + {})", d0 + 1).unwrap();
            if hi && flip_msb {
                writeln!(self.asm, "    ld a, {hi_r}").unwrap();
                writeln!(self.asm, "    xor 0x80").unwrap();
                writeln!(self.asm, "    ld {hi_r}, a").unwrap();
            }
        } else {
            let addr = self.fmt_val_addr_mem(v, if hi { 2 } else { 0 });
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
            if let (Some(&ca), Some(&cb)) =
                (self.const_vals.get(&a_val), self.const_vals.get(&b_val))
            {
                if (ca as u32) == (cb as u32) {
                    writeln!(self.asm, "    xor a").unwrap();
                } else {
                    writeln!(self.asm, "    ld a, 1").unwrap();
                    writeln!(self.asm, "    or a").unwrap();
                }
                return;
            }
            let (var_a, var_b) = if self.const_vals.contains_key(&a_val)
                && !self.const_vals.contains_key(&b_val)
            {
                (b_val, a_val)
            } else {
                (a_val, b_val)
            };
            let const_b = self.const_vals.get(&var_b).map(|&c| c as u32);
            let ne_lbl = self.fresh_label("cmp32_ne");
            self.emit_load_word32(var_a, false, "hl", false);
            if const_b.is_some_and(|u| (u & 0xFFFF) == 0) {
                writeln!(self.asm, "    ld a, h").unwrap();
                writeln!(self.asm, "    or l").unwrap();
            } else {
                self.emit_load_word32(var_b, false, "de", false);
                writeln!(self.asm, "    or a").unwrap();
                writeln!(self.asm, "    sbc hl, de").unwrap();
            }
            writeln!(self.asm, "    jr nz, {ne_lbl}").unwrap();
            self.emit_load_word32(var_a, true, "hl", false);
            if const_b.is_some_and(|u| (u >> 16) == 0) {
                writeln!(self.asm, "    ld a, h").unwrap();
                writeln!(self.asm, "    or l").unwrap();
            } else {
                self.emit_load_word32(var_b, true, "de", false);
                writeln!(self.asm, "    or a").unwrap();
                writeln!(self.asm, "    sbc hl, de").unwrap();
            }
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

        // Fast idiom 1: signed 16-bit comparisons against `0` (`bit 7, h` / `ld a, h; or l`)
        if sz == 2 && signed {
            let zero_cmp = if self.const_vals.get(&rhs) == Some(&0) {
                Some((lhs, op))
            } else if self.const_vals.get(&lhs) == Some(&0) {
                let flipped = match op {
                    IrBinOp::Eq => IrBinOp::Eq,
                    IrBinOp::Ne => IrBinOp::Ne,
                    IrBinOp::Lt => IrBinOp::Gt,
                    IrBinOp::Le => IrBinOp::Ge,
                    IrBinOp::Gt => IrBinOp::Lt,
                    IrBinOp::Ge => IrBinOp::Le,
                    _ => unreachable!(),
                };
                Some((rhs, flipped))
            } else {
                None
            };
            if let Some((var_v, norm_op)) = zero_cmp {
                match norm_op {
                    IrBinOp::Lt | IrBinOp::Ge => {
                        self.emit_load_hl_from(var_v);
                        writeln!(self.asm, "    bit 7, h").unwrap();
                        let cc = if norm_op == IrBinOp::Lt { "nz" } else { "z" };
                        emit_branch_cc(cc, self.asm);
                        return;
                    }
                    IrBinOp::Gt | IrBinOp::Le => {
                        let (true_bb, false_bb) = if norm_op == IrBinOp::Gt {
                            (then_bb, else_bb)
                        } else {
                            (else_bb, then_bb)
                        };
                        self.emit_load_hl_from(var_v);
                        writeln!(self.asm, "    ld a, h").unwrap();
                        writeln!(self.asm, "    or l").unwrap();
                        writeln!(self.asm, "    jp z, {fname}_bb{false_bb}").unwrap();
                        writeln!(self.asm, "    bit 7, h").unwrap();
                        if next_bb_id == Some(true_bb) {
                            writeln!(self.asm, "    jp nz, {fname}_bb{false_bb}").unwrap();
                        } else {
                            writeln!(self.asm, "    jp z, {fname}_bb{true_bb}").unwrap();
                            if next_bb_id != Some(false_bb) {
                                writeln!(self.asm, "    jp {fname}_bb{false_bb}").unwrap();
                            }
                        }
                        return;
                    }
                    _ => {}
                }
            }
            if let Some((var_v, want_lt, h_k)) = self.match_signed_i16_high_byte_cmp(op, lhs, rhs) {
                if h_k == 0 {
                    if self.val_in_reg.get(&var_v) == Some(&RegLoc::BC) {
                        writeln!(self.asm, "    bit 7, b").unwrap();
                    } else if let Some(&(disp, _)) = self.ix_val_loads.get(&var_v)
                        && let Some(hi_d) = disp.checked_add(1)
                    {
                        writeln!(self.asm, "    bit 7, (ix + {hi_d})").unwrap();
                    } else {
                        self.emit_load_hl_from(var_v);
                        writeln!(self.asm, "    bit 7, h").unwrap();
                    }
                    emit_branch_cc(if want_lt { "nz" } else { "z" }, self.asm);
                } else {
                    self.emit_signed_i16_high_byte_xor_cp(var_v, h_k);
                    emit_branch_cc(if want_lt { "c" } else { "nc" }, self.asm);
                }
                return;
            }
        }

        // Fast idiom 2: 16-bit compact unsigned/equality constant compares
        if sz == 2 {
            let (norm_op, var_v, const_opt) = if self.const_vals.contains_key(&lhs)
                && !self.const_vals.contains_key(&rhs)
            {
                let flipped = match op {
                    IrBinOp::Eq => IrBinOp::Eq,
                    IrBinOp::Ne => IrBinOp::Ne,
                    IrBinOp::Lt => IrBinOp::Gt,
                    IrBinOp::Le => IrBinOp::Ge,
                    IrBinOp::Gt => IrBinOp::Lt,
                    IrBinOp::Ge => IrBinOp::Le,
                    _ => unreachable!(),
                };
                (flipped, rhs, self.const_vals.get(&lhs).copied())
            } else {
                (op, lhs, self.const_vals.get(&rhs).copied())
            };
            if let Some(c_raw) = const_opt {
                match (norm_op, c_raw, signed) {
                    (IrBinOp::Eq, 0, _)
                    | (IrBinOp::Lt, 1, false)
                    | (IrBinOp::Le, 0, false) => {
                        if self.val_in_reg.get(&var_v) == Some(&RegLoc::BC) {
                            writeln!(self.asm, "    ld a, b").unwrap();
                            writeln!(self.asm, "    or c").unwrap();
                        } else {
                            self.emit_load_hl_from(var_v);
                            writeln!(self.asm, "    ld a, h").unwrap();
                            writeln!(self.asm, "    or l").unwrap();
                        }
                        emit_branch_cc("z", self.asm);
                        return;
                    }
                    (IrBinOp::Ne, 0, _)
                    | (IrBinOp::Ge, 1, false)
                    | (IrBinOp::Gt, 0, false) => {
                        if self.val_in_reg.get(&var_v) == Some(&RegLoc::BC) {
                            writeln!(self.asm, "    ld a, b").unwrap();
                            writeln!(self.asm, "    or c").unwrap();
                        } else {
                            self.emit_load_hl_from(var_v);
                            writeln!(self.asm, "    ld a, h").unwrap();
                            writeln!(self.asm, "    or l").unwrap();
                        }
                        emit_branch_cc("nz", self.asm);
                        return;
                    }
                    (IrBinOp::Eq | IrBinOp::Ne, 1, _) => {
                        self.emit_load_hl_from(var_v);
                        writeln!(self.asm, "    dec hl").unwrap();
                        writeln!(self.asm, "    ld a, h").unwrap();
                        writeln!(self.asm, "    or l").unwrap();
                        let cc = if norm_op == IrBinOp::Eq { "z" } else { "nz" };
                        emit_branch_cc(cc, self.asm);
                        return;
                    }
                    (IrBinOp::Lt, 256, false) | (IrBinOp::Le, 255, false) => {
                        self.emit_load_hl_from(var_v);
                        writeln!(self.asm, "    ld a, h").unwrap();
                        writeln!(self.asm, "    or a").unwrap();
                        emit_branch_cc("z", self.asm);
                        return;
                    }
                    (IrBinOp::Ge, 256, false) | (IrBinOp::Gt, 255, false) => {
                        self.emit_load_hl_from(var_v);
                        writeln!(self.asm, "    ld a, h").unwrap();
                        writeln!(self.asm, "    or a").unwrap();
                        emit_branch_cc("nz", self.asm);
                        return;
                    }
                    _ => {}
                }
            }
            if let Some((var_v, want_lt, h_k)) = self.match_signed_i16_high_byte_cmp(op, lhs, rhs) {
                if h_k == 0 {
                    if self.val_in_reg.get(&var_v) == Some(&RegLoc::BC) {
                        writeln!(self.asm, "    bit 7, b").unwrap();
                    } else if let Some(&(disp, _)) = self.ix_val_loads.get(&var_v)
                        && let Some(hi_d) = disp.checked_add(1)
                    {
                        writeln!(self.asm, "    bit 7, (ix + {hi_d})").unwrap();
                    } else {
                        self.emit_load_hl_from(var_v);
                        writeln!(self.asm, "    bit 7, h").unwrap();
                    }
                    emit_branch_cc(if want_lt { "nz" } else { "z" }, self.asm);
                } else {
                    self.emit_signed_i16_high_byte_xor_cp(var_v, h_k);
                    emit_branch_cc(if want_lt { "c" } else { "nc" }, self.asm);
                }
                return;
            }
        }

        // Fast idiom 2b: 32-bit `x == 0` / `x != 0`, or unsigned `x > 0` / `x <= 0`
        if sz == 4 {
            let zero_target = if self.const_vals.get(&rhs) == Some(&0)
                && (matches!(op, IrBinOp::Eq | IrBinOp::Ne)
                    || (!signed && matches!(op, IrBinOp::Gt | IrBinOp::Le)))
            {
                Some((
                    lhs,
                    match op {
                        IrBinOp::Eq | IrBinOp::Le => "z",
                        IrBinOp::Ne | IrBinOp::Gt => "nz",
                        _ => unreachable!(),
                    },
                ))
            } else if self.const_vals.get(&lhs) == Some(&0)
                && (matches!(op, IrBinOp::Eq | IrBinOp::Ne)
                    || (!signed && matches!(op, IrBinOp::Lt | IrBinOp::Ge)))
            {
                Some((
                    rhs,
                    match op {
                        IrBinOp::Eq | IrBinOp::Ge => "z",
                        IrBinOp::Ne | IrBinOp::Lt => "nz",
                        _ => unreachable!(),
                    },
                ))
            } else {
                None
            };
            if let Some((var_v, cc)) = zero_target {
                self.emit_load_hlde_from(var_v);
                writeln!(self.asm, "    ld a, e").unwrap();
                writeln!(self.asm, "    or d").unwrap();
                writeln!(self.asm, "    or l").unwrap();
                writeln!(self.asm, "    or h").unwrap();
                emit_branch_cc(cc, self.asm);
                return;
            }
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
                emit_branch_cc(cc, self.asm);
            }
            2 => {
                if signed && ordered {
                    let (sa, sb, want_lt) = match op {
                        IrBinOp::Lt => (lhs, rhs, true),
                        IrBinOp::Ge => (lhs, rhs, false),
                        IrBinOp::Gt => (rhs, lhs, true),
                        IrBinOp::Le => (rhs, lhs, false),
                        _ => unreachable!(),
                    };
                    self.emit_load_hl_from(sa);
                    self.emit_load_de_from(sb);
                    writeln!(self.asm, "    or a").unwrap();
                    writeln!(self.asm, "    sbc hl, de").unwrap();
                    self.emit_signed_lt_branch(fname, want_lt, then_bb, else_bb, next_bb_id);
                } else {
                    self.emit_load_hl_from(a_val);
                    self.emit_load_de_from(b_val);
                    writeln!(self.asm, "    or a").unwrap();
                    writeln!(self.asm, "    sbc hl, de").unwrap();
                    emit_branch_cc(cc, self.asm);
                }
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
        let signed = is_signed_ty(ty);
        let frac_bits = match ty {
            IrType::Fixed { frac_bits, .. } => *frac_bits,
            _ => 0,
        };

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
                self.emit_store_dst_a(dst);
                return;
            }
            if sz == 2 && signed {
                let zero_cmp = if self.const_vals.get(&rhs) == Some(&0) {
                    Some((lhs, op))
                } else if self.const_vals.get(&lhs) == Some(&0) {
                    let flipped = match op {
                        IrBinOp::Eq => IrBinOp::Eq,
                        IrBinOp::Ne => IrBinOp::Ne,
                        IrBinOp::Lt => IrBinOp::Gt,
                        IrBinOp::Le => IrBinOp::Ge,
                        IrBinOp::Gt => IrBinOp::Lt,
                        IrBinOp::Ge => IrBinOp::Le,
                        _ => unreachable!(),
                    };
                    Some((rhs, flipped))
                } else {
                    None
                };
                if let Some((var_v, norm_op)) = zero_cmp {
                    match norm_op {
                        IrBinOp::Lt | IrBinOp::Ge => {
                            self.emit_load_hl_from(var_v);
                            writeln!(self.asm, "    add hl, hl").unwrap();
                            if norm_op == IrBinOp::Ge {
                                writeln!(self.asm, "    ccf").unwrap();
                            }
                            writeln!(self.asm, "    ld a, 0").unwrap();
                            writeln!(self.asm, "    rla").unwrap();
                            self.emit_store_dst_a(dst);
                            return;
                        }
                        IrBinOp::Gt => {
                            self.emit_load_hl_from(var_v);
                            writeln!(self.asm, "    ld a, h").unwrap();
                            writeln!(self.asm, "    or l").unwrap();
                            writeln!(self.asm, "    jr z, {done}").unwrap();
                            writeln!(self.asm, "    ld a, 1").unwrap();
                            writeln!(self.asm, "    bit 7, h").unwrap();
                            writeln!(self.asm, "    jr z, {done}").unwrap();
                            writeln!(self.asm, "    xor a").unwrap();
                            writeln!(self.asm, "{done}:").unwrap();
                            self.emit_store_dst_a(dst);
                            return;
                        }
                        IrBinOp::Le => {
                            self.emit_load_hl_from(var_v);
                            writeln!(self.asm, "    ld a, h").unwrap();
                            writeln!(self.asm, "    or l").unwrap();
                            writeln!(self.asm, "    ld a, 1").unwrap();
                            writeln!(self.asm, "    jr z, {done}").unwrap();
                            writeln!(self.asm, "    bit 7, h").unwrap();
                            writeln!(self.asm, "    jr nz, {done}").unwrap();
                            writeln!(self.asm, "    xor a").unwrap();
                            writeln!(self.asm, "{done}:").unwrap();
                            self.emit_store_dst_a(dst);
                            return;
                        }
                        _ => {}
                    }
                }
                if let Some((var_v, want_lt, h_k)) =
                    self.match_signed_i16_high_byte_cmp(op, lhs, rhs)
                {
                    if h_k == 0 {
                        self.emit_load_hl_from(var_v);
                        writeln!(self.asm, "    add hl, hl").unwrap();
                        if !want_lt {
                            writeln!(self.asm, "    ccf").unwrap();
                        }
                        writeln!(self.asm, "    ld a, 0").unwrap();
                        writeln!(self.asm, "    rla").unwrap();
                        self.emit_store_dst_a(dst);
                    } else {
                        self.emit_signed_i16_high_byte_xor_cp(var_v, h_k);
                        self.emit_bool_from_cc(if want_lt { "c" } else { "nc" }, dst);
                    }
                    return;
                }
            }
            if sz == 2
                && self.const_vals.get(&rhs) == Some(&0)
                && (matches!(op, IrBinOp::Eq | IrBinOp::Ne)
                    || (!signed && matches!(op, IrBinOp::Gt | IrBinOp::Le)))
            {
                if self.val_in_reg.get(&lhs) == Some(&RegLoc::BC) {
                    writeln!(self.asm, "    ld a, b").unwrap();
                    writeln!(self.asm, "    or c").unwrap();
                } else {
                    self.emit_load_hl_from(lhs);
                    writeln!(self.asm, "    ld a, h").unwrap();
                    writeln!(self.asm, "    or l").unwrap();
                }
                let cc = match op {
                    IrBinOp::Eq | IrBinOp::Le => "z",
                    IrBinOp::Ne | IrBinOp::Gt => "nz",
                    _ => unreachable!(),
                };
                writeln!(self.asm, "    ld a, 1").unwrap();
                writeln!(self.asm, "    jr {cc}, {done}").unwrap();
                writeln!(self.asm, "    xor a").unwrap();
                writeln!(self.asm, "{done}:").unwrap();
                self.emit_store_dst_a(dst);
                return;
            }
            if sz == 4 {
                let zero_target = if self.const_vals.get(&rhs) == Some(&0)
                    && (matches!(op, IrBinOp::Eq | IrBinOp::Ne)
                        || (!signed && matches!(op, IrBinOp::Gt | IrBinOp::Le)))
                {
                    Some((
                        lhs,
                        match op {
                            IrBinOp::Eq | IrBinOp::Le => "z",
                            IrBinOp::Ne | IrBinOp::Gt => "nz",
                            _ => unreachable!(),
                        },
                    ))
                } else if self.const_vals.get(&lhs) == Some(&0)
                    && (matches!(op, IrBinOp::Eq | IrBinOp::Ne)
                        || (!signed && matches!(op, IrBinOp::Lt | IrBinOp::Ge)))
                {
                    Some((
                        rhs,
                        match op {
                            IrBinOp::Eq | IrBinOp::Ge => "z",
                            IrBinOp::Ne | IrBinOp::Lt => "nz",
                            _ => unreachable!(),
                        },
                    ))
                } else {
                    None
                };
                if let Some((var_v, cc)) = zero_target {
                    self.emit_load_hlde_from(var_v);
                    writeln!(self.asm, "    ld a, e").unwrap();
                    writeln!(self.asm, "    or d").unwrap();
                    writeln!(self.asm, "    or l").unwrap();
                    writeln!(self.asm, "    or h").unwrap();
                    writeln!(self.asm, "    ld a, 1").unwrap();
                    writeln!(self.asm, "    jr {cc}, {done}").unwrap();
                    writeln!(self.asm, "    xor a").unwrap();
                    writeln!(self.asm, "{done}:").unwrap();
                    self.emit_store_dst_a(dst);
                    return;
                }
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
                self.emit_store_dst_a(dst);
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
                    self.emit_bool_from_cc(cc, dst);
                }
                2 => {
                    if signed && ordered {
                        let (sa, sb, want_lt) = match op {
                            IrBinOp::Lt => (lhs, rhs, true),
                            IrBinOp::Ge => (lhs, rhs, false),
                            IrBinOp::Gt => (rhs, lhs, true),
                            IrBinOp::Le => (rhs, lhs, false),
                            _ => unreachable!(),
                        };
                        self.emit_load_hl_from(sa);
                        self.emit_load_de_from(sb);
                        writeln!(self.asm, "    or a").unwrap();
                        writeln!(self.asm, "    sbc hl, de").unwrap();
                        self.emit_signed_lt_bool(want_lt, dst);
                    } else {
                        self.emit_load_hl_from(a_val);
                        self.emit_load_de_from(b_val);
                        writeln!(self.asm, "    or a").unwrap();
                        writeln!(self.asm, "    sbc hl, de").unwrap();
                        self.emit_bool_from_cc(cc, dst);
                    }
                }
                4 => {
                    self.emit_cmp32_flags(a_val, b_val, signed, ordered);
                    self.emit_bool_from_cc(cc, dst);
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
                            (IrBinOp::Shl, 1..=7) => emit_shift_a_left_const(self.asm, k),
                            (IrBinOp::Shr, 1..=3) => {
                                for _ in 0..k {
                                    if signed {
                                        writeln!(self.asm, "    sra a").unwrap();
                                    } else {
                                        writeln!(self.asm, "    srl a").unwrap();
                                    }
                                }
                            }
                            (IrBinOp::Shr, 5..=7) if !signed => {
                                for _ in 0..(8 - k) {
                                    writeln!(self.asm, "    rlca").unwrap();
                                }
                                writeln!(self.asm, "    and 0x{:02X}", 0xFFu8 >> k).unwrap();
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
                        self.emit_store_dst_a(dst);
                        return;
                    }
                    if self
                        .const_vals
                        .get(&lhs)
                        .is_some_and(|&c| (c & 0xFF) == 0)
                    {
                        writeln!(self.asm, "    xor a").unwrap();
                        self.emit_store_dst_a(dst);
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
                    self.emit_store_dst_a(dst);
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
                    self.emit_store_dst_a(dst);
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
                    self.emit_store_dst_a(dst);
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
                        self.emit_store_dst_a(dst);
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
                        self.emit_store_dst_a(dst);
                        return;
                    }
                }
                // Fixed-point 8-bit Mul / Div require intermediate widening by `frac_bits`
                if frac_bits > 0 && matches!(op, IrBinOp::Mul | IrBinOp::Div) {
                    if op == IrBinOp::Mul {
                        if let (Some(&cl), Some(&cr)) =
                            (self.const_vals.get(&lhs), self.const_vals.get(&rhs))
                        {
                            let raw = if signed {
                                ((cl as i8 as i32) * (cr as i8 as i32)) >> frac_bits
                            } else {
                                (((cl as u8 as u32) * (cr as u8 as u32)) >> frac_bits) as i32
                            };
                            let b = raw as u8;
                            if b == 0 {
                                writeln!(self.asm, "    xor a").unwrap();
                            } else {
                                writeln!(self.asm, "    ld a, 0x{b:02X}").unwrap();
                            }
                            self.emit_store_dst_a(dst);
                            return;
                        }
                        let pow2_side = self
                            .const_vals
                            .get(&rhs)
                            .filter(|&&c| {
                                c == 0
                                    || if signed {
                                        let v = c as i8;
                                        v > 0 && (v as u8).is_power_of_two()
                                    } else {
                                        (c as u8).is_power_of_two()
                                    }
                            })
                            .map(|&c| (lhs, c))
                            .or_else(|| {
                                self.const_vals
                                    .get(&lhs)
                                    .filter(|&&c| {
                                        c == 0
                                            || if signed {
                                                let v = c as i8;
                                                v > 0 && (v as u8).is_power_of_two()
                                            } else {
                                                (c as u8).is_power_of_two()
                                            }
                                    })
                                    .map(|&c| (rhs, c))
                            });
                        if let Some((x_val, c)) = pow2_side {
                            if c == 0 {
                                writeln!(self.asm, "    xor a").unwrap();
                            } else {
                                let p = (c as u8).trailing_zeros() as u8;
                                self.emit_load_a_from(x_val);
                                if p < frac_bits {
                                    for _ in 0..(frac_bits - p) {
                                        if signed {
                                            writeln!(self.asm, "    sra a").unwrap();
                                        } else {
                                            writeln!(self.asm, "    srl a").unwrap();
                                        }
                                    }
                                } else if p > frac_bits {
                                    emit_shift_a_left_const(self.asm, p - frac_bits);
                                }
                            }
                            self.emit_store_dst_a(dst);
                            return;
                        }
                        self.emit_load_a_from(rhs);
                        writeln!(self.asm, "    ld l, a").unwrap();
                        if signed {
                            writeln!(self.asm, "    add a, a").unwrap();
                            writeln!(self.asm, "    sbc a, a").unwrap();
                            writeln!(self.asm, "    ld h, a").unwrap();
                        } else {
                            writeln!(self.asm, "    ld h, 0").unwrap();
                        }
                        writeln!(self.asm, "    push hl").unwrap();
                        self.emit_load_a_from(lhs);
                        writeln!(self.asm, "    ld e, a").unwrap();
                        if signed {
                            writeln!(self.asm, "    add a, a").unwrap();
                            writeln!(self.asm, "    sbc a, a").unwrap();
                            writeln!(self.asm, "    ld d, a").unwrap();
                        } else {
                            writeln!(self.asm, "    ld d, 0").unwrap();
                        }
                        writeln!(self.asm, "    pop bc").unwrap();
                        if signed {
                            self.runtime_needs.muls16_32 = true;
                            self.runtime_needs.mulu16_32 = true;
                            writeln!(self.asm, "    call __aun_muls16_32").unwrap();
                        } else {
                            self.runtime_needs.mulu16_32 = true;
                            writeln!(self.asm, "    call __aun_mulu16_32").unwrap();
                        }
                        writeln!(self.asm, "    ex de, hl").unwrap();
                        emit_shift16_hl_const(self.asm, IrBinOp::Shr, signed, frac_bits);
                        writeln!(self.asm, "    ld a, l").unwrap();
                        self.val_in_reg
                            .retain(|_, &mut vr| !crate::regalloc::RegMask::BC.intersects(vr.mask()));
                        self.emit_store_dst_a(dst);
                        return;
                    } else {
                        // 8-bit Fixed Div: ((lhs << frac_bits) / rhs) via 16-bit signed/unsigned div.
                        self.emit_load_a_from(rhs);
                        writeln!(self.asm, "    ld l, a").unwrap();
                        if signed {
                            writeln!(self.asm, "    add a, a").unwrap();
                            writeln!(self.asm, "    sbc a, a").unwrap();
                            writeln!(self.asm, "    ld h, a").unwrap();
                        } else {
                            writeln!(self.asm, "    ld h, 0").unwrap();
                        }
                        writeln!(self.asm, "    push hl").unwrap();
                        self.emit_load_a_from(lhs);
                        writeln!(self.asm, "    ld l, a").unwrap();
                        if signed {
                            writeln!(self.asm, "    add a, a").unwrap();
                            writeln!(self.asm, "    sbc a, a").unwrap();
                            writeln!(self.asm, "    ld h, a").unwrap();
                        } else {
                            writeln!(self.asm, "    ld h, 0").unwrap();
                        }
                        emit_shift16_hl_const(self.asm, IrBinOp::Shl, false, frac_bits);
                        writeln!(self.asm, "    pop de").unwrap();
                        let helper = if signed {
                            self.runtime_needs.sdiv16 = true;
                            "__aun_sdiv16"
                        } else {
                            self.runtime_needs.udiv16 = true;
                            "__aun_udiv16"
                        };
                        writeln!(self.asm, "    call {helper}").unwrap();
                        writeln!(self.asm, "    ld a, l").unwrap();
                        self.val_in_reg
                            .retain(|_, &mut vr| !crate::regalloc::RegMask::BC.intersects(vr.mask()));
                        self.emit_store_dst_a(dst);
                        return;
                    }
                }
                // Fast 8-bit constant multiply via inline shift-add (0 B clobber!)
                if op == IrBinOp::Mul
                    && let Some((x_val, c_raw)) = self
                        .const_vals
                        .get(&rhs)
                        .map(|&c| (lhs, c))
                        .or_else(|| self.const_vals.get(&lhs).map(|&c| (rhs, c)))
                {
                    let k = (c_raw & 0xFF) as u8;
                    match k {
                        0 => writeln!(self.asm, "    xor a").unwrap(),
                        1 => self.emit_load_a_from(x_val),
                        0xFF => {
                            self.emit_load_a_from(x_val);
                            writeln!(self.asm, "    neg").unwrap();
                        }
                        0x80 => {
                            self.emit_load_a_from(x_val);
                            writeln!(self.asm, "    rrca").unwrap();
                            writeln!(self.asm, "    and 0x80").unwrap();
                        }
                        _ if k.is_power_of_two() => {
                            self.emit_load_a_from(x_val);
                            for _ in 0..k.trailing_zeros() {
                                writeln!(self.asm, "    add a, a").unwrap();
                            }
                        }
                        _ => {
                            self.emit_load_a_from(x_val);
                            writeln!(self.asm, "    ld e, a").unwrap();
                            let top_bit = 7 - k.leading_zeros() as usize;
                            for bit in (0..top_bit).rev() {
                                writeln!(self.asm, "    add a, a").unwrap();
                                if (k & (1u8 << bit)) != 0 {
                                    writeln!(self.asm, "    add a, e").unwrap();
                                }
                            }
                        }
                    }
                    self.emit_store_dst_a(dst);
                    return;
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
                        self.runtime_needs.mul8 = true;
                        writeln!(self.asm, "    call __aun_mul8").unwrap();
                        self.val_in_reg
                            .retain(|_, &mut vr| !crate::regalloc::RegMask::B.intersects(vr.mask()));
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
                self.emit_store_dst_a(dst);
            }
            2 => {
                // Fused 16-bit `x ^ (x << k)` / `x ^ (x >> k)` idioms (e.g. xorshift16)
                if op == IrBinOp::BitXor
                    && let Some(&(src_v, sop, k)) = self.fused_xor_shift.get(&dst)
                {
                    if let Some(&c) = self.const_vals.get(&src_v) {
                        writeln!(self.asm, "    ld hl, 0x{:04X}", c as u16).unwrap();
                    } else {
                        self.emit_load_hl_from(src_v);
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
                    self.emit_store_dst_hl(dst);
                    return;
                }
                // DE-accumulated `u16` bit-pack `BitOr` chains
                if let Some(action) = self.bitpack_actions.get(&dst).cloned() {
                    self.emit_bitpack_action(dst, action);
                    return;
                }
                // Fused 16-bit `acc | ((u8 as u16) << k)` / `acc | (u8 as u16)` bit-pack chains
                if op == IrBinOp::BitOr
                    && let Some(&(acc_v, term)) = self.fused_bitor_u8_term.get(&dst)
                {
                    let x_bits = self.possible_bits(term.x_v) & 0x00FF & (term.mask8 as u16);
                    let term_bits = if term.k >= 16 {
                        0
                    } else {
                        x_bits.wrapping_shl(term.k as u32)
                    };
                    if term_bits == 0 {
                        self.emit_load_hl_from(acc_v);
                    } else if (term_bits & 0xFF00) == 0 {
                        self.emit_load_hl_from(acc_v);
                        let eff = self.emit_load_masked_u8_term_into_a(term);
                        emit_shift_a_left_const_for_bits(self.asm, term.k, eff);
                        writeln!(self.asm, "    or l").unwrap();
                        writeln!(self.asm, "    ld l, a").unwrap();
                    } else if (term_bits & 0x00FF) == 0 {
                        self.emit_load_hl_from(acc_v);
                        let eff = self.emit_load_masked_u8_term_into_a(term);
                        emit_shift_a_left_const_for_bits(self.asm, term.k - 8, eff);
                        writeln!(self.asm, "    or h").unwrap();
                        writeln!(self.asm, "    ld h, a").unwrap();
                    } else {
                        self.emit_load_hl_from(acc_v);
                        writeln!(self.asm, "    ex de, hl").unwrap();
                        let _eff = self.emit_load_masked_u8_term_into_a(term);
                        emit_u8_shl16_from_a(self.asm, term.k, true);
                    }
                    self.emit_store_dst_hl(dst);
                    return;
                }
                // Fixed-point 16-bit Mul / Div: widen intermediate by `frac_bits`
                if frac_bits > 0 && matches!(op, IrBinOp::Mul | IrBinOp::Div) {
                    if op == IrBinOp::Mul {
                        if let (Some(&cl), Some(&cr)) =
                            (self.const_vals.get(&lhs), self.const_vals.get(&rhs))
                        {
                            let raw = if signed {
                                ((cl as i16 as i64) * (cr as i16 as i64)) >> frac_bits
                            } else {
                                (((cl as u16 as u64) * (cr as u16 as u64)) >> frac_bits) as i64
                            };
                            writeln!(self.asm, "    ld hl, 0x{:04X}", raw as u16).unwrap();
                            self.emit_store_dst_hl(dst);
                            return;
                        }
                        let pow2_side = self
                            .const_vals
                            .get(&rhs)
                            .filter(|&&c| {
                                c == 0
                                    || if signed {
                                        let v = c as i16;
                                        v > 0 && (v as u16).is_power_of_two()
                                    } else {
                                        (c as u16).is_power_of_two()
                                    }
                            })
                            .map(|&c| (lhs, c))
                            .or_else(|| {
                                self.const_vals
                                    .get(&lhs)
                                    .filter(|&&c| {
                                        c == 0
                                            || if signed {
                                                let v = c as i16;
                                                v > 0 && (v as u16).is_power_of_two()
                                            } else {
                                                (c as u16).is_power_of_two()
                                            }
                                    })
                                    .map(|&c| (rhs, c))
                            });
                        if let Some((x_val, c)) = pow2_side {
                            if c == 0 {
                                writeln!(self.asm, "    ld hl, 0").unwrap();
                            } else {
                                let p = (c as u16).trailing_zeros() as u8;
                                self.emit_load_hl_from(x_val);
                                if p < frac_bits {
                                    emit_shift16_hl_const(
                                        self.asm,
                                        IrBinOp::Shr,
                                        signed,
                                        frac_bits - p,
                                    );
                                } else if p > frac_bits {
                                    emit_shift16_hl_const(
                                        self.asm,
                                        IrBinOp::Shl,
                                        signed,
                                        p - frac_bits,
                                    );
                                }
                            }
                            self.emit_store_dst_hl(dst);
                            return;
                        }
                        self.emit_load_hl_from(rhs);
                        writeln!(self.asm, "    push hl").unwrap();
                        self.emit_load_de_from(lhs);
                        writeln!(self.asm, "    pop bc").unwrap();
                        if signed {
                            self.runtime_needs.muls16_32 = true;
                            self.runtime_needs.mulu16_32 = true;
                            writeln!(self.asm, "    call __aun_muls16_32").unwrap();
                        } else {
                            self.runtime_needs.mulu16_32 = true;
                            writeln!(self.asm, "    call __aun_mulu16_32").unwrap();
                        }
                        if frac_bits == 0 {
                            writeln!(self.asm, "    ex de, hl").unwrap();
                        } else if frac_bits < 8 {
                            for _ in 0..(8 - frac_bits) {
                                writeln!(self.asm, "    sla e").unwrap();
                                writeln!(self.asm, "    rl d").unwrap();
                                writeln!(self.asm, "    rl l").unwrap();
                            }
                            writeln!(self.asm, "    ld h, l").unwrap();
                            writeln!(self.asm, "    ld l, d").unwrap();
                        } else if frac_bits == 8 {
                            writeln!(self.asm, "    ld h, l").unwrap();
                            writeln!(self.asm, "    ld l, d").unwrap();
                        } else if frac_bits < 16 {
                            for _ in 0..(frac_bits - 8) {
                                if signed {
                                    writeln!(self.asm, "    sra h").unwrap();
                                } else {
                                    writeln!(self.asm, "    srl h").unwrap();
                                }
                                writeln!(self.asm, "    rr l").unwrap();
                                writeln!(self.asm, "    rr d").unwrap();
                            }
                            writeln!(self.asm, "    ld h, l").unwrap();
                            writeln!(self.asm, "    ld l, d").unwrap();
                        }
                        self.val_in_reg
                            .retain(|_, &mut vr| !crate::regalloc::RegMask::BC.intersects(vr.mask()));
                        self.emit_store_dst_hl(dst);
                        return;
                    } else {
                        // 16-bit Fixed Div: ((lhs << frac_bits) / rhs) via 32-bit signed/unsigned div.
                        self.emit_load_de_from(rhs);
                        if signed {
                            writeln!(self.asm, "    ld a, d").unwrap();
                            writeln!(self.asm, "    add a, a").unwrap();
                            writeln!(self.asm, "    sbc a, a").unwrap();
                            writeln!(self.asm, "    ld h, a").unwrap();
                            writeln!(self.asm, "    ld l, a").unwrap();
                        } else {
                            writeln!(self.asm, "    ld hl, 0").unwrap();
                        }
                        writeln!(self.asm, "    push hl").unwrap();
                        writeln!(self.asm, "    push de").unwrap();
                        self.emit_load_de_from(lhs);
                        if signed {
                            writeln!(self.asm, "    ld a, d").unwrap();
                            writeln!(self.asm, "    add a, a").unwrap();
                            writeln!(self.asm, "    sbc a, a").unwrap();
                            writeln!(self.asm, "    ld h, a").unwrap();
                            writeln!(self.asm, "    ld l, a").unwrap();
                        } else {
                            writeln!(self.asm, "    ld hl, 0").unwrap();
                        }
                        if frac_bits == 8 {
                            writeln!(self.asm, "    ld h, l").unwrap();
                            writeln!(self.asm, "    ld l, d").unwrap();
                            writeln!(self.asm, "    ld d, e").unwrap();
                            writeln!(self.asm, "    ld e, 0").unwrap();
                        } else if frac_bits < 8 {
                            let lp = self.fresh_label("fdiv_sh");
                            writeln!(self.asm, "    ld b, {frac_bits}").unwrap();
                            writeln!(self.asm, "{lp}:").unwrap();
                            writeln!(self.asm, "    sla e").unwrap();
                            writeln!(self.asm, "    rl d").unwrap();
                            writeln!(self.asm, "    rl l").unwrap();
                            writeln!(self.asm, "    rl h").unwrap();
                            writeln!(self.asm, "    djnz {lp}").unwrap();
                        } else {
                            writeln!(self.asm, "    ld h, l").unwrap();
                            writeln!(self.asm, "    ld l, d").unwrap();
                            writeln!(self.asm, "    ld d, e").unwrap();
                            writeln!(self.asm, "    ld e, 0").unwrap();
                            let rem_sh = frac_bits - 8;
                            if rem_sh > 0 {
                                let lp = self.fresh_label("fdiv_sh");
                                writeln!(self.asm, "    ld b, {rem_sh}").unwrap();
                                writeln!(self.asm, "{lp}:").unwrap();
                                writeln!(self.asm, "    sla d").unwrap();
                                writeln!(self.asm, "    rl l").unwrap();
                                writeln!(self.asm, "    rl h").unwrap();
                                writeln!(self.asm, "    djnz {lp}").unwrap();
                            }
                        }
                        let helper = if signed {
                            self.runtime_needs.sdiv32 = true;
                            self.runtime_needs.udiv32 = true;
                            "__aun_sdiv32"
                        } else {
                            self.runtime_needs.udiv32 = true;
                            "__aun_udiv32"
                        };
                        writeln!(self.asm, "    call {helper}").unwrap();
                        writeln!(self.asm, "    pop bc").unwrap();
                        writeln!(self.asm, "    pop bc").unwrap();
                        writeln!(self.asm, "    ex de, hl").unwrap();
                        self.val_in_reg
                            .retain(|_, &mut vr| !crate::regalloc::RegMask::BC.intersects(vr.mask()));
                        self.emit_store_dst_hl(dst);
                        return;
                    }
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
                    self.emit_store_dst_hl(dst);
                    return;
                }
                // Compact 16-bit negation: `0 - x` (6 bytes, no DE clobber)
                if op == IrBinOp::Sub
                    && mode != ArithMode::Saturating
                    && self.const_vals.get(&lhs) == Some(&0)
                {
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
                    self.emit_store_dst_hl(dst);
                    return;
                }
                if matches!(op, IrBinOp::Shl | IrBinOp::BitAnd)
                    && let Some(&term) = self.fused_u8_shl16.get(&dst)
                {
                    self.emit_u8_pack_term_into_hl(term);
                    self.emit_store_dst_hl(dst);
                    return;
                }
                if matches!(op, IrBinOp::Shl | IrBinOp::Shr) {
                    let is_shl = op == IrBinOp::Shl;
                    if let Some(&k_raw) = self.const_vals.get(&rhs) {
                        let orig_k = k_raw.clamp(0, 16) as u8;
                        let can_cse = !self.const_addr_loads.contains_key(&lhs)
                            && !self.const_vals.contains_key(&lhs)
                            && !self.ix_val_loads.contains_key(&lhs);
                        let (used_cse, src_addr, k) = if can_cse
                            && let Some(&(prev_k, prev_da)) =
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
                        emit_shift16_hl_const(self.asm, op, signed, k);
                        self.emit_store_dst_hl(dst);
                        if can_cse
                            && orig_k > 0
                            && orig_k < 8
                            && !self.fa.direct_reg_dsts.contains_key(&dst)
                        {
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
                    self.emit_store_dst_hl(dst);
                    return;
                }
                // Compact 16-bit add/sub by small constant 0..=3 using `inc hl` / `dec hl`
                if matches!(op, IrBinOp::Add | IrBinOp::Sub)
                    && mode != ArithMode::Saturating
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
                    self.emit_store_dst_hl(dst);
                    return;
                }
                // Compact 16-bit bitwise ops with constant mask (avoids `ld de, imm` + redundant high/low byte ops)
                if matches!(op, IrBinOp::BitAnd | IrBinOp::BitOr | IrBinOp::BitXor)
                    && let Some((var_v, c_raw)) = self
                        .const_vals
                        .get(&rhs)
                        .map(|&c| (lhs, c))
                        .or_else(|| self.const_vals.get(&lhs).map(|&c| (rhs, c)))
                {
                    let w = c_raw as u16;
                    let lo = (w & 0xFF) as u8;
                    let hi = (w >> 8) as u8;
                    if let Some(&cl) = self.const_vals.get(&var_v) {
                        let lw = cl as u16;
                        let res = match op {
                            IrBinOp::BitAnd => lw & w,
                            IrBinOp::BitOr => lw | w,
                            IrBinOp::BitXor => lw ^ w,
                            _ => unreachable!(),
                        };
                        writeln!(self.asm, "    ld hl, 0x{res:04X}").unwrap();
                    } else {
                        self.emit_load_hl_from(var_v);
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
                    self.emit_store_dst_hl(dst);
                    return;
                }
                if op == IrBinOp::Add
                    && mode != ArithMode::Saturating
                    && self.val_in_reg.get(&rhs) == Some(&RegLoc::BC)
                {
                    self.emit_load_hl_from(lhs);
                    writeln!(self.asm, "    add hl, bc").unwrap();
                    self.emit_store_dst_hl(dst);
                    return;
                }
                if op == IrBinOp::Add
                    && mode != ArithMode::Saturating
                    && self.val_in_reg.get(&lhs) == Some(&RegLoc::BC)
                {
                    self.emit_load_hl_from(rhs);
                    writeln!(self.asm, "    add hl, bc").unwrap();
                    self.emit_store_dst_hl(dst);
                    return;
                }
                if op == IrBinOp::BitOr {
                    let l_bits = self.possible_bits(lhs);
                    let r_bits = self.possible_bits(rhs);
                    if l_bits == 0 {
                        self.emit_load_hl_from(rhs);
                        self.emit_store_dst_hl(dst);
                        return;
                    }
                    if r_bits == 0 {
                        self.emit_load_hl_from(lhs);
                        self.emit_store_dst_hl(dst);
                        return;
                    }
                    let half_opt = if (r_bits & 0x00FF) == 0 {
                        Some((lhs, rhs, true))
                    } else if (r_bits & 0xFF00) == 0 {
                        Some((lhs, rhs, false))
                    } else if (l_bits & 0x00FF) == 0 {
                        Some((rhs, lhs, true))
                    } else if (l_bits & 0xFF00) == 0 {
                        Some((rhs, lhs, false))
                    } else {
                        None
                    };
                    if let Some((hl_op, byte_op, is_lo_zero)) = half_opt
                        && !self.const_addrs.contains_key(&byte_op)
                        && !self.rom_addrs.contains_key(&byte_op)
                    {
                        if is_lo_zero {
                            let ix_ok = self
                                .ix_val_loads
                                .get(&byte_op)
                                .is_none_or(|&(disp, _)| disp <= 126);
                            if ix_ok {
                                self.emit_load_hl_from(hl_op);
                                self.emit_load_hi_byte_a_from(byte_op);
                                writeln!(self.asm, "    or h").unwrap();
                                writeln!(self.asm, "    ld h, a").unwrap();
                                self.emit_store_dst_hl(dst);
                                return;
                            }
                        } else {
                            self.emit_load_hl_from(hl_op);
                            self.emit_load_a_from(byte_op);
                            writeln!(self.asm, "    or l").unwrap();
                            writeln!(self.asm, "    ld l, a").unwrap();
                            self.emit_store_dst_hl(dst);
                            return;
                        }
                    }
                }
                self.emit_load_hl_from(lhs);
                self.emit_load_de_from(rhs);
                match (op, mode) {
                    (IrBinOp::Add, ArithMode::Saturating) if !signed => {
                        let lbl = self.fresh_label("sat16");
                        writeln!(self.asm, "    add hl, de").unwrap();
                        writeln!(self.asm, "    jr nc, {lbl}").unwrap();
                        writeln!(self.asm, "    ld hl, 0xFFFF").unwrap();
                        writeln!(self.asm, "{lbl}:").unwrap();
                    }
                    (IrBinOp::Sub, ArithMode::Saturating) if !signed => {
                        let lbl = self.fresh_label("sat16");
                        writeln!(self.asm, "    or a").unwrap();
                        writeln!(self.asm, "    sbc hl, de").unwrap();
                        writeln!(self.asm, "    jr nc, {lbl}").unwrap();
                        writeln!(self.asm, "    ld hl, 0x0000").unwrap();
                        writeln!(self.asm, "{lbl}:").unwrap();
                    }
                    (IrBinOp::Add, ArithMode::Saturating) => {
                        let lbl = self.fresh_label("sat16");
                        writeln!(self.asm, "    or a").unwrap();
                        writeln!(self.asm, "    adc hl, de").unwrap();
                        writeln!(self.asm, "    jp po, {lbl}").unwrap();
                        writeln!(self.asm, "    ld hl, 0x7FFF").unwrap();
                        writeln!(self.asm, "    jp m, {lbl}").unwrap();
                        writeln!(self.asm, "    ld hl, 0x8000").unwrap();
                        writeln!(self.asm, "{lbl}:").unwrap();
                    }
                    (IrBinOp::Sub, ArithMode::Saturating) => {
                        let lbl = self.fresh_label("sat16");
                        writeln!(self.asm, "    or a").unwrap();
                        writeln!(self.asm, "    sbc hl, de").unwrap();
                        writeln!(self.asm, "    jp po, {lbl}").unwrap();
                        writeln!(self.asm, "    ld hl, 0x7FFF").unwrap();
                        writeln!(self.asm, "    jp m, {lbl}").unwrap();
                        writeln!(self.asm, "    ld hl, 0x8000").unwrap();
                        writeln!(self.asm, "{lbl}:").unwrap();
                    }
                    (IrBinOp::Add, _) => writeln!(self.asm, "    add hl, de").unwrap(),
                    (IrBinOp::Sub, _) => {
                        writeln!(self.asm, "    or a").unwrap();
                        writeln!(self.asm, "    sbc hl, de").unwrap();
                    }
                    (IrBinOp::BitAnd, _) => {
                        writeln!(self.asm, "    ld a, l").unwrap();
                        writeln!(self.asm, "    and e").unwrap();
                        writeln!(self.asm, "    ld l, a").unwrap();
                        writeln!(self.asm, "    ld a, h").unwrap();
                        writeln!(self.asm, "    and d").unwrap();
                        writeln!(self.asm, "    ld h, a").unwrap();
                    }
                    (IrBinOp::BitOr, _) => {
                        writeln!(self.asm, "    ld a, l").unwrap();
                        writeln!(self.asm, "    or e").unwrap();
                        writeln!(self.asm, "    ld l, a").unwrap();
                        writeln!(self.asm, "    ld a, h").unwrap();
                        writeln!(self.asm, "    or d").unwrap();
                        writeln!(self.asm, "    ld h, a").unwrap();
                    }
                    (IrBinOp::BitXor, _) => {
                        writeln!(self.asm, "    ld a, l").unwrap();
                        writeln!(self.asm, "    xor e").unwrap();
                        writeln!(self.asm, "    ld l, a").unwrap();
                        writeln!(self.asm, "    ld a, h").unwrap();
                        writeln!(self.asm, "    xor d").unwrap();
                        writeln!(self.asm, "    ld h, a").unwrap();
                    }
                    (IrBinOp::Div, _) => {
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
                    (IrBinOp::Rem, _) => {
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
                self.emit_store_dst_hl(dst);
            }
            4 => {
                let da_mem = self.fmt_ram_mem(da);
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
                            } else if k >= 32 || (op == IrBinOp::Shr && signed && k >= 31) {
                                if op == IrBinOp::Shr && signed {
                                    self.emit_load_word32(lhs, true, "hl", false);
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
                            } else if op == IrBinOp::Shl && (24..=31).contains(&k) {
                                self.emit_load_word32(lhs, false, "hl", false);
                                if k == 24 {
                                    writeln!(self.asm, "    ld h, l").unwrap();
                                } else {
                                    writeln!(self.asm, "    ld a, l").unwrap();
                                    for _ in 0..(k - 24) {
                                        writeln!(self.asm, "    add a, a").unwrap();
                                    }
                                    writeln!(self.asm, "    ld h, a").unwrap();
                                }
                                writeln!(self.asm, "    ld l, 0").unwrap();
                                writeln!(self.asm, "    ld de, 0").unwrap();
                                self.emit_store_hlde_to(da, true);
                            } else if op == IrBinOp::Shr && (24..=31).contains(&k) {
                                self.emit_load_word32(lhs, true, "hl", false);
                                for _ in 0..(k - 24) {
                                    if signed {
                                        writeln!(self.asm, "    sra h").unwrap();
                                    } else {
                                        writeln!(self.asm, "    srl h").unwrap();
                                    }
                                }
                                writeln!(self.asm, "    ld e, h").unwrap();
                                if signed {
                                    writeln!(self.asm, "    ld a, h").unwrap();
                                    writeln!(self.asm, "    add a, a").unwrap();
                                    writeln!(self.asm, "    sbc a, a").unwrap();
                                    writeln!(self.asm, "    ld d, a").unwrap();
                                    writeln!(self.asm, "    ld l, a").unwrap();
                                    writeln!(self.asm, "    ld h, a").unwrap();
                                } else {
                                    writeln!(self.asm, "    ld d, 0").unwrap();
                                    writeln!(self.asm, "    ld hl, 0").unwrap();
                                }
                                self.emit_store_hlde_to(da, true);
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
                            } else if (17..=23).contains(&k) {
                                if op == IrBinOp::Shl {
                                    writeln!(self.asm, "    ld hl, 0").unwrap();
                                    writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                                    self.emit_load_word32(lhs, false, "hl", false);
                                    emit_shift16_hl_const(self.asm, IrBinOp::Shl, false, (k - 16) as u8);
                                    writeln!(self.asm, "    ld {da2_mem}, hl").unwrap();
                                } else {
                                    self.emit_load_word32(lhs, true, "hl", false);
                                    emit_shift16_hl_const(self.asm, IrBinOp::Shr, signed, (k - 16) as u8);
                                    writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                                    if signed {
                                        writeln!(self.asm, "    ld a, h").unwrap();
                                        writeln!(self.asm, "    add a, a").unwrap();
                                        writeln!(self.asm, "    sbc a, a").unwrap();
                                        writeln!(self.asm, "    ld l, a").unwrap();
                                        writeln!(self.asm, "    ld h, a").unwrap();
                                    } else {
                                        writeln!(self.asm, "    ld hl, 0").unwrap();
                                    }
                                    writeln!(self.asm, "    ld {da2_mem}, hl").unwrap();
                                }
                            } else if k == 8 {
                                self.emit_load_hlde_from(lhs);
                                if op == IrBinOp::Shl {
                                    writeln!(self.asm, "    ld h, l").unwrap();
                                    writeln!(self.asm, "    ld l, d").unwrap();
                                    writeln!(self.asm, "    ld d, e").unwrap();
                                    writeln!(self.asm, "    ld e, 0").unwrap();
                                } else {
                                    writeln!(self.asm, "    ld e, d").unwrap();
                                    writeln!(self.asm, "    ld d, l").unwrap();
                                    writeln!(self.asm, "    ld l, h").unwrap();
                                    if signed {
                                        writeln!(self.asm, "    ld a, h").unwrap();
                                        writeln!(self.asm, "    add a, a").unwrap();
                                        writeln!(self.asm, "    sbc a, a").unwrap();
                                        writeln!(self.asm, "    ld h, a").unwrap();
                                    } else {
                                        writeln!(self.asm, "    ld h, 0").unwrap();
                                    }
                                }
                                self.emit_store_hlde_to(da, true);
                            } else if op == IrBinOp::Shl && (9..=11).contains(&k) {
                                self.emit_load_hlde_from(lhs);
                                writeln!(self.asm, "    ld h, l").unwrap();
                                writeln!(self.asm, "    ld l, d").unwrap();
                                writeln!(self.asm, "    ld d, e").unwrap();
                                writeln!(self.asm, "    ld e, 0").unwrap();
                                for _ in 0..(k - 8) {
                                    writeln!(self.asm, "    sla d").unwrap();
                                    writeln!(self.asm, "    adc hl, hl").unwrap();
                                }
                                self.emit_store_hlde_to(da, true);
                            } else if op == IrBinOp::Shl && (12..=15).contains(&k) {
                                let lp = self.fresh_label("sh32_lp");
                                self.emit_load_hlde_from(lhs);
                                writeln!(self.asm, "    ld h, l").unwrap();
                                writeln!(self.asm, "    ld l, d").unwrap();
                                writeln!(self.asm, "    ld d, e").unwrap();
                                writeln!(self.asm, "    ld e, 0").unwrap();
                                writeln!(self.asm, "    ld b, {}", k - 8).unwrap();
                                writeln!(self.asm, "{lp}:").unwrap();
                                writeln!(self.asm, "    sla d").unwrap();
                                writeln!(self.asm, "    adc hl, hl").unwrap();
                                writeln!(self.asm, "    djnz {lp}").unwrap();
                                self.val_in_reg
                                    .retain(|_, &mut vr| !crate::regalloc::RegMask::B.intersects(vr.mask()));
                                self.emit_store_hlde_to(da, true);
                            } else if op == IrBinOp::Shr && (9..=10).contains(&k) {
                                self.emit_load_hlde_from(lhs);
                                writeln!(self.asm, "    ld e, d").unwrap();
                                writeln!(self.asm, "    ld d, l").unwrap();
                                writeln!(self.asm, "    ld l, h").unwrap();
                                for _ in 0..(k - 8) {
                                    if signed {
                                        writeln!(self.asm, "    sra l").unwrap();
                                    } else {
                                        writeln!(self.asm, "    srl l").unwrap();
                                    }
                                    writeln!(self.asm, "    rr d").unwrap();
                                    writeln!(self.asm, "    rr e").unwrap();
                                }
                                if signed {
                                    writeln!(self.asm, "    ld a, l").unwrap();
                                    writeln!(self.asm, "    add a, a").unwrap();
                                    writeln!(self.asm, "    sbc a, a").unwrap();
                                    writeln!(self.asm, "    ld h, a").unwrap();
                                } else {
                                    writeln!(self.asm, "    ld h, 0").unwrap();
                                }
                                self.emit_store_hlde_to(da, true);
                            } else if op == IrBinOp::Shr && (11..=15).contains(&k) {
                                let lp = self.fresh_label("sh32_lp");
                                self.emit_load_hlde_from(lhs);
                                writeln!(self.asm, "    ld e, d").unwrap();
                                writeln!(self.asm, "    ld d, l").unwrap();
                                writeln!(self.asm, "    ld l, h").unwrap();
                                writeln!(self.asm, "    ld b, {}", k - 8).unwrap();
                                writeln!(self.asm, "{lp}:").unwrap();
                                if signed {
                                    writeln!(self.asm, "    sra l").unwrap();
                                } else {
                                    writeln!(self.asm, "    srl l").unwrap();
                                }
                                writeln!(self.asm, "    rr d").unwrap();
                                writeln!(self.asm, "    rr e").unwrap();
                                writeln!(self.asm, "    djnz {lp}").unwrap();
                                if signed {
                                    writeln!(self.asm, "    ld a, l").unwrap();
                                    writeln!(self.asm, "    add a, a").unwrap();
                                    writeln!(self.asm, "    sbc a, a").unwrap();
                                    writeln!(self.asm, "    ld h, a").unwrap();
                                } else {
                                    writeln!(self.asm, "    ld h, 0").unwrap();
                                }
                                self.val_in_reg
                                    .retain(|_, &mut vr| !crate::regalloc::RegMask::B.intersects(vr.mask()));
                                self.emit_store_hlde_to(da, true);
                            } else if (op == IrBinOp::Shl && k <= 3)
                                || (op == IrBinOp::Shr && k <= 2)
                            {
                                self.emit_load_hlde_from(lhs);
                                for _ in 0..k {
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
                                }
                                self.emit_store_hlde_to(da, true);
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
                                self.val_in_reg
                                    .retain(|_, &mut vr| !crate::regalloc::RegMask::B.intersects(vr.mask()));
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
                    IrBinOp::Mul
                        if (self.u16_bound_vals.contains(&lhs)
                            && self.u8_bound_vals.contains(&rhs))
                            || (self.u16_bound_vals.contains(&rhs)
                                && self.u8_bound_vals.contains(&lhs)) =>
                    {
                        let (op16, op8) = if self.u16_bound_vals.contains(&lhs)
                            && self.u8_bound_vals.contains(&rhs)
                        {
                            (lhs, rhs)
                        } else {
                            (rhs, lhs)
                        };
                        if let Some(&c) = self.const_vals.get(&op16) {
                            writeln!(self.asm, "    ld de, 0x{:04X}", (c as u32 & 0xFFFF) as u16)
                                .unwrap();
                        } else {
                            self.emit_load_de_from(op16);
                        }
                        if let Some(&c) = self.const_vals.get(&op8) {
                            writeln!(self.asm, "    ld a, {}", (c as u32 & 0xFF) as u8).unwrap();
                        } else {
                            self.emit_load_a_from(op8);
                        }
                        let lp = self.fresh_label("m16x8_lp");
                        let sk = self.fresh_label("m16x8_sk");
                        writeln!(self.asm, "    ld c, a").unwrap();
                        writeln!(self.asm, "    ld hl, 0").unwrap();
                        writeln!(self.asm, "    xor a").unwrap();
                        writeln!(self.asm, "    ld b, 8").unwrap();
                        writeln!(self.asm, "{lp}:").unwrap();
                        writeln!(self.asm, "    add hl, hl").unwrap();
                        writeln!(self.asm, "    rla").unwrap();
                        writeln!(self.asm, "    sla c").unwrap();
                        writeln!(self.asm, "    jr nc, {sk}").unwrap();
                        writeln!(self.asm, "    add hl, de").unwrap();
                        writeln!(self.asm, "    adc a, 0").unwrap();
                        writeln!(self.asm, "{sk}:").unwrap();
                        writeln!(self.asm, "    djnz {lp}").unwrap();
                        writeln!(self.asm, "    ex de, hl").unwrap();
                        writeln!(self.asm, "    ld l, a").unwrap();
                        writeln!(self.asm, "    ld h, 0").unwrap();
                        self.val_in_reg
                            .retain(|_, &mut vr| !crate::regalloc::RegMask::BC.intersects(vr.mask()));
                        self.emit_store_hlde_to(da, true);
                    }
                    IrBinOp::Mul
                        if !signed
                            && self.u16_bound_vals.contains(&lhs)
                            && self.u16_bound_vals.contains(&rhs) =>
                    {
                        self.emit_load_word32(lhs, false, "de", false);
                        self.emit_load_word32(rhs, false, "bc", false);
                        self.runtime_needs.mulu16_32 = true;
                        writeln!(self.asm, "    call __aun_mulu16_32").unwrap();
                        self.val_in_reg
                            .retain(|_, &mut vr| !crate::regalloc::RegMask::BC.intersects(vr.mask()));
                        self.emit_store_hlde_to(da, true);
                    }
                    IrBinOp::Mul
                        if signed
                            && self.i16_bound_vals.contains(&lhs)
                            && self.i16_bound_vals.contains(&rhs) =>
                    {
                        self.emit_load_word32(lhs, false, "de", false);
                        self.emit_load_word32(rhs, false, "bc", false);
                        self.runtime_needs.mulu16_32 = true;
                        self.runtime_needs.muls16_32 = true;
                        writeln!(self.asm, "    call __aun_muls16_32").unwrap();
                        self.val_in_reg
                            .retain(|_, &mut vr| !crate::regalloc::RegMask::BC.intersects(vr.mask()));
                        self.emit_store_hlde_to(da, true);
                    }
                    op @ (IrBinOp::Div | IrBinOp::Rem)
                        if !signed
                            && (self
                                .const_vals
                                .get(&rhs)
                                .is_some_and(|&c| (1..=255).contains(&c))
                                || (self.u8_bound_vals.contains(&rhs)
                                    && !self.const_vals.contains_key(&rhs))) =>
                    {
                        if let Some(&c) = self.const_vals.get(&rhs) {
                            writeln!(self.asm, "    ld c, 0x{:02X}", (c & 0xFF) as u8).unwrap();
                        } else {
                            let rhs_m = self.fmt_val_addr_mem(rhs, 0);
                            writeln!(self.asm, "    ld a, {rhs_m}").unwrap();
                            writeln!(self.asm, "    ld c, a").unwrap();
                        }
                        self.emit_load_hlde_from(lhs);
                        self.runtime_needs.udiv32_8 = true;
                        writeln!(self.asm, "    call __aun_udivrem32_8").unwrap();
                        let emit_store_rem32 = |this: &mut EmitCtx<'_>, addr: u16| {
                            let r0 = this.fmt_ram_mem(addr);
                            let r2 = this.fmt_ram_mem(addr.wrapping_add(2));
                            writeln!(this.asm, "    ld l, a").unwrap();
                            writeln!(this.asm, "    ld h, 0").unwrap();
                            writeln!(this.asm, "    ld {r0}, hl").unwrap();
                            writeln!(this.asm, "    ld l, h").unwrap();
                            writeln!(this.asm, "    ld {r2}, hl").unwrap();
                        };
                        if let Some(&(_, partner_dst)) = self.fused_divrem_partner.get(&dst) {
                            let partner_addr = self.val_loc(partner_dst);
                            let (quot_addr, rem_addr) = if op == IrBinOp::Div {
                                (da, partner_addr)
                            } else {
                                (partner_addr, da)
                            };
                            self.emit_store_hlde_to(quot_addr, true);
                            emit_store_rem32(self, rem_addr);
                        } else if op == IrBinOp::Div {
                            self.emit_store_hlde_to(da, true);
                        } else {
                            emit_store_rem32(self, da);
                        }
                        self.val_in_reg
                            .retain(|_, &mut vr| !crate::regalloc::RegMask::BC.intersects(vr.mask()));
                    }
                    op @ (IrBinOp::Div | IrBinOp::Rem)
                        if !signed && self.fused_divrem_partner.contains_key(&dst) =>
                    {
                        let (_, partner_dst) = self.fused_divrem_partner[&dst];
                        self.emit_load_hlde_from(rhs);
                        writeln!(self.asm, "    push hl").unwrap();
                        writeln!(self.asm, "    push de").unwrap();
                        self.emit_load_hlde_from(lhs);
                        self.runtime_needs.udiv32 = true;
                        writeln!(self.asm, "    call __aun_udivrem32").unwrap();
                        let partner_addr = self.val_loc(partner_dst);
                        let (quot_addr, rem_addr) = if op == IrBinOp::Div {
                            (da, partner_addr)
                        } else {
                            (partner_addr, da)
                        };
                        self.emit_store_hlde_to(quot_addr, true);
                        writeln!(self.asm, "    pop de").unwrap();
                        writeln!(self.asm, "    pop hl").unwrap();
                        self.emit_store_hlde_to(rem_addr, true);
                        self.val_in_reg
                            .retain(|_, &mut vr| !crate::regalloc::RegMask::BC.intersects(vr.mask()));
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
