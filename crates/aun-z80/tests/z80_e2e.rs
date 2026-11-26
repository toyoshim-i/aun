//! Self-contained Z80 machine-code E2E verification test for `mdxnt_drv.aun`
//! (and comparison against the hand-written Z80 reference `mdxnt_drv_z80.rel`).

use aun_z80::{CompileOptions, compile_z80_sdasz80};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// ---------------------------------------------------------------------------
// Minimal Mock YM2203 / PC-88 HAL (sdcccall(1))
// ---------------------------------------------------------------------------

const MOCK_HAL_ASM: &str = r#"
        .module mock_hal
        .optsdcc -mz80 sdcccall(1)
        .area   _DATA
        .area   _INITIALIZED
        .area   _INITIALIZER
        .area   _CODE

; void mdxnt_opn_write(uint8_t reg, uint8_t val) -- A = reg, L = val
_mdxnt_opn_write::
        out     (0xF0), a
        ret

; void mdxnt_hw_enable_irq(uint8_t init_tb) -- A = init_tb
_mdxnt_hw_enable_irq::
        out     (0xF1), a
        ret

; void mdxnt_hw_disable_irq(void)
_mdxnt_hw_disable_irq::
        out     (0xF2), a
        ret

; void mdxnt_hw_ack_timer_b(void)
_mdxnt_hw_ack_timer_b::
        out     (0xF3), a
        ret

; void mdxnt_enter_crit(void) -- preserves all regs/flags
_mdxnt_enter_crit::
        out     (0xF4), a
        ret

; void mdxnt_leave_crit(void) -- preserves all regs/flags
_mdxnt_leave_crit::
        out     (0xF5), a
        ret
"#;

#[derive(Debug, Clone)]
struct MockHal {
    regs: [u8; 256],
    last_tb: u8,
    irq_enabled: bool,
    ack_count: u32,
    in_crit: bool,
    opn_writes: Vec<(u8, u8)>,
}

impl Default for MockHal {
    fn default() -> Self {
        Self {
            regs: [0; 256],
            last_tb: 0,
            irq_enabled: false,
            ack_count: 0,
            in_crit: false,
            opn_writes: Vec::new(),
        }
    }
}

impl MockHal {
    fn reset(&mut self) {
        self.regs = [0; 256];
        self.last_tb = 0;
        self.irq_enabled = false;
        self.ack_count = 0;
        self.in_crit = false;
        self.opn_writes.clear();
    }

    fn on_out(&mut self, port: u8, a: u8, l: u8) {
        match port {
            0xF0 => {
                self.regs[a as usize] = l;
                self.opn_writes.push((a, l));
            }
            0xF1 => {
                self.last_tb = a;
                self.irq_enabled = true;
                self.in_crit = false;
            }
            0xF2 => {
                self.irq_enabled = false;
                self.in_crit = false;
            }
            0xF3 => {
                self.ack_count += 1;
            }
            0xF4 => {
                self.in_crit = true;
            }
            0xF5 => {
                self.in_crit = false;
            }
            other => panic!("unexpected Z80 OUT port 0x{other:02X} (A=0x{a:02X}, L=0x{l:02X})"),
        }
    }

    /// Filters out Timer-B ack writes (0x27 <- 0x3A) if needed, or returns all OPN writes.
    fn filtered_opn_writes(&self) -> Vec<(u8, u8)> {
        self.opn_writes
            .iter()
            .copied()
            .filter(|&(r, v)| !(r == 0x27 && v == 0x3A))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Clean, Accurate Z80 CPU Interpreter (64 KB RAM)
// ---------------------------------------------------------------------------

const FLAG_C: u8 = 0x01;
const FLAG_N: u8 = 0x02;
const FLAG_PV: u8 = 0x04;
const FLAG_X: u8 = 0x08;
const FLAG_H: u8 = 0x10;
const FLAG_Y: u8 = 0x20;
const FLAG_Z: u8 = 0x40;
const FLAG_S: u8 = 0x80;

#[inline]
fn parity_even(v: u8) -> bool {
    v.count_ones() % 2 == 0
}

#[derive(Clone, Copy)]
enum IndexMode {
    Hl,
    Ix,
    Iy,
}

struct Z80 {
    ram: Vec<u8>,
    a: u8,
    f: u8,
    b: u8,
    c: u8,
    d: u8,
    e: u8,
    h: u8,
    l: u8,
    a_alt: u8,
    f_alt: u8,
    b_alt: u8,
    c_alt: u8,
    d_alt: u8,
    e_alt: u8,
    h_alt: u8,
    l_alt: u8,
    ix: u16,
    iy: u16,
    sp: u16,
    pc: u16,
    i: u8,
    r: u8,
    iff1: bool,
    iff2: bool,
    im: u8,
    halted: bool,
    tstates: u64,
    steps: u64,
}

impl Z80 {
    fn new() -> Self {
        Self {
            ram: vec![0; 65536],
            a: 0,
            f: 0,
            b: 0,
            c: 0,
            d: 0,
            e: 0,
            h: 0,
            l: 0,
            a_alt: 0,
            f_alt: 0,
            b_alt: 0,
            c_alt: 0,
            d_alt: 0,
            e_alt: 0,
            h_alt: 0,
            l_alt: 0,
            ix: 0,
            iy: 0,
            sp: 0xF000,
            pc: 0,
            i: 0,
            r: 0,
            iff1: false,
            iff2: false,
            im: 0,
            halted: false,
            tstates: 0,
            steps: 0,
        }
    }

    #[inline]
    fn read8(&self, addr: u16) -> u8 {
        self.ram[addr as usize]
    }

    #[inline]
    fn write8(&mut self, addr: u16, val: u8) {
        self.ram[addr as usize] = val;
    }

    #[inline]
    fn read16(&self, addr: u16) -> u16 {
        let lo = self.read8(addr) as u16;
        let hi = self.read8(addr.wrapping_add(1)) as u16;
        lo | (hi << 8)
    }

    #[inline]
    fn write16(&mut self, addr: u16, val: u16) {
        self.write8(addr, val as u8);
        self.write8(addr.wrapping_add(1), (val >> 8) as u8);
    }

    #[inline]
    fn fetch8(&mut self) -> u8 {
        let v = self.read8(self.pc);
        self.pc = self.pc.wrapping_add(1);
        v
    }

    #[inline]
    fn fetch16(&mut self) -> u16 {
        let lo = self.fetch8() as u16;
        let hi = self.fetch8() as u16;
        lo | (hi << 8)
    }

    #[inline]
    fn push16(&mut self, val: u16) {
        self.sp = self.sp.wrapping_sub(2);
        self.write16(self.sp, val);
    }

    #[inline]
    fn pop16(&mut self) -> u16 {
        let val = self.read16(self.sp);
        self.sp = self.sp.wrapping_add(2);
        val
    }

    #[inline]
    fn af(&self) -> u16 {
        ((self.a as u16) << 8) | (self.f as u16)
    }
    #[inline]
    fn set_af(&mut self, v: u16) {
        self.a = (v >> 8) as u8;
        self.f = v as u8;
    }
    #[inline]
    fn bc(&self) -> u16 {
        ((self.b as u16) << 8) | (self.c as u16)
    }
    #[inline]
    fn set_bc(&mut self, v: u16) {
        self.b = (v >> 8) as u8;
        self.c = v as u8;
    }
    #[inline]
    fn de(&self) -> u16 {
        ((self.d as u16) << 8) | (self.e as u16)
    }
    #[inline]
    fn set_de(&mut self, v: u16) {
        self.d = (v >> 8) as u8;
        self.e = v as u8;
    }
    #[inline]
    fn hl(&self) -> u16 {
        ((self.h as u16) << 8) | (self.l as u16)
    }
    #[inline]
    fn set_hl(&mut self, v: u16) {
        self.h = (v >> 8) as u8;
        self.l = v as u8;
    }

    #[inline]
    fn idx_reg(&self, mode: IndexMode) -> u16 {
        match mode {
            IndexMode::Hl => self.hl(),
            IndexMode::Ix => self.ix,
            IndexMode::Iy => self.iy,
        }
    }

    #[inline]
    fn set_idx_reg(&mut self, mode: IndexMode, v: u16) {
        match mode {
            IndexMode::Hl => self.set_hl(v),
            IndexMode::Ix => self.ix = v,
            IndexMode::Iy => self.iy = v,
        }
    }

    fn get_rp(&self, idx: u8, mode: IndexMode) -> u16 {
        match idx & 3 {
            0 => self.bc(),
            1 => self.de(),
            2 => self.idx_reg(mode),
            3 => self.sp,
            _ => unreachable!(),
        }
    }

    fn set_rp(&mut self, idx: u8, mode: IndexMode, v: u16) {
        match idx & 3 {
            0 => self.set_bc(v),
            1 => self.set_de(v),
            2 => self.set_idx_reg(mode, v),
            3 => self.sp = v,
            _ => unreachable!(),
        }
    }

    fn get_rp2(&self, idx: u8, mode: IndexMode) -> u16 {
        match idx & 3 {
            0 => self.bc(),
            1 => self.de(),
            2 => self.idx_reg(mode),
            3 => self.af(),
            _ => unreachable!(),
        }
    }

    fn set_rp2(&mut self, idx: u8, mode: IndexMode, v: u16) {
        match idx & 3 {
            0 => self.set_bc(v),
            1 => self.set_de(v),
            2 => self.set_idx_reg(mode, v),
            3 => self.set_af(v),
            _ => unreachable!(),
        }
    }

    #[inline]
    fn get_r8_direct(&self, idx: u8, mode: IndexMode) -> u8 {
        match idx & 7 {
            0 => self.b,
            1 => self.c,
            2 => self.d,
            3 => self.e,
            4 => match mode {
                IndexMode::Hl => self.h,
                IndexMode::Ix => (self.ix >> 8) as u8,
                IndexMode::Iy => (self.iy >> 8) as u8,
            },
            5 => match mode {
                IndexMode::Hl => self.l,
                IndexMode::Ix => self.ix as u8,
                IndexMode::Iy => self.iy as u8,
            },
            7 => self.a,
            other => panic!("get_r8_direct called with idx={other}"),
        }
    }

    #[inline]
    fn set_r8_direct(&mut self, idx: u8, mode: IndexMode, v: u8) {
        match idx & 7 {
            0 => self.b = v,
            1 => self.c = v,
            2 => self.d = v,
            3 => self.e = v,
            4 => match mode {
                IndexMode::Hl => self.h = v,
                IndexMode::Ix => self.ix = (self.ix & 0x00FF) | ((v as u16) << 8),
                IndexMode::Iy => self.iy = (self.iy & 0x00FF) | ((v as u16) << 8),
            },
            5 => match mode {
                IndexMode::Hl => self.l = v,
                IndexMode::Ix => self.ix = (self.ix & 0xFF00) | (v as u16),
                IndexMode::Iy => self.iy = (self.iy & 0xFF00) | (v as u16),
            },
            7 => self.a = v,
            other => panic!("set_r8_direct called with idx={other}"),
        }
    }

    #[inline]
    fn cond(&self, cc: u8) -> bool {
        match cc & 7 {
            0 => (self.f & FLAG_Z) == 0,  // NZ
            1 => (self.f & FLAG_Z) != 0,  // Z
            2 => (self.f & FLAG_C) == 0,  // NC
            3 => (self.f & FLAG_C) != 0,  // C
            4 => (self.f & FLAG_PV) == 0, // PO
            5 => (self.f & FLAG_PV) != 0, // PE
            6 => (self.f & FLAG_S) == 0,  // P
            7 => (self.f & FLAG_S) != 0,  // M
            _ => unreachable!(),
        }
    }

    #[inline]
    fn sz_xy(&self, v: u8) -> u8 {
        (v & (FLAG_S | FLAG_X | FLAG_Y)) | if v == 0 { FLAG_Z } else { 0 }
    }

    fn alu(&mut self, op: u8, val: u8) {
        let a = self.a;
        match op & 7 {
            0 => {
                // ADD
                let sum = (a as u16) + (val as u16);
                let res = sum as u8;
                let h = ((a & 0x0F) + (val & 0x0F)) > 0x0F;
                let pv = ((a ^ res) & (val ^ res) & 0x80) != 0;
                self.a = res;
                self.f = self.sz_xy(res)
                    | if h { FLAG_H } else { 0 }
                    | if pv { FLAG_PV } else { 0 }
                    | if sum > 0xFF { FLAG_C } else { 0 };
            }
            1 => {
                // ADC
                let cin = if (self.f & FLAG_C) != 0 { 1u16 } else { 0u16 };
                let sum = (a as u16) + (val as u16) + cin;
                let res = sum as u8;
                let h = ((a & 0x0F) as u16 + (val & 0x0F) as u16 + cin) > 0x0F;
                let pv = ((a ^ res) & (val ^ res) & 0x80) != 0;
                self.a = res;
                self.f = self.sz_xy(res)
                    | if h { FLAG_H } else { 0 }
                    | if pv { FLAG_PV } else { 0 }
                    | if sum > 0xFF { FLAG_C } else { 0 };
            }
            2 => {
                // SUB
                let diff = (a as i16) - (val as i16);
                let res = diff as u8;
                let h = (a & 0x0F) < (val & 0x0F);
                let pv = ((a ^ val) & (a ^ res) & 0x80) != 0;
                self.a = res;
                self.f = self.sz_xy(res)
                    | FLAG_N
                    | if h { FLAG_H } else { 0 }
                    | if pv { FLAG_PV } else { 0 }
                    | if diff < 0 { FLAG_C } else { 0 };
            }
            3 => {
                // SBC
                let cin = if (self.f & FLAG_C) != 0 { 1i16 } else { 0i16 };
                let diff = (a as i16) - (val as i16) - cin;
                let res = diff as u8;
                let h = ((a & 0x0F) as i16 - (val & 0x0F) as i16 - cin) < 0;
                let pv = ((a ^ val) & (a ^ res) & 0x80) != 0;
                self.a = res;
                self.f = self.sz_xy(res)
                    | FLAG_N
                    | if h { FLAG_H } else { 0 }
                    | if pv { FLAG_PV } else { 0 }
                    | if diff < 0 { FLAG_C } else { 0 };
            }
            4 => {
                // AND
                let res = a & val;
                self.a = res;
                self.f = self.sz_xy(res) | FLAG_H | if parity_even(res) { FLAG_PV } else { 0 };
            }
            5 => {
                // XOR
                let res = a ^ val;
                self.a = res;
                self.f = self.sz_xy(res) | if parity_even(res) { FLAG_PV } else { 0 };
            }
            6 => {
                // OR
                let res = a | val;
                self.a = res;
                self.f = self.sz_xy(res) | if parity_even(res) { FLAG_PV } else { 0 };
            }
            7 => {
                // CP (X/Y come from val, A unchanged)
                let diff = (a as i16) - (val as i16);
                let res = diff as u8;
                let h = (a & 0x0F) < (val & 0x0F);
                let pv = ((a ^ val) & (a ^ res) & 0x80) != 0;
                self.f = (res & FLAG_S)
                    | if res == 0 { FLAG_Z } else { 0 }
                    | (val & (FLAG_X | FLAG_Y))
                    | FLAG_N
                    | if h { FLAG_H } else { 0 }
                    | if pv { FLAG_PV } else { 0 }
                    | if diff < 0 { FLAG_C } else { 0 };
            }
            _ => unreachable!(),
        }
    }

    fn inc8(&mut self, v: u8) -> u8 {
        let res = v.wrapping_add(1);
        let h = (v & 0x0F) == 0x0F;
        let pv = v == 0x7F;
        self.f = (self.f & FLAG_C)
            | self.sz_xy(res)
            | if h { FLAG_H } else { 0 }
            | if pv { FLAG_PV } else { 0 };
        res
    }

    fn dec8(&mut self, v: u8) -> u8 {
        let res = v.wrapping_sub(1);
        let h = (v & 0x0F) == 0x00;
        let pv = v == 0x80;
        self.f = (self.f & FLAG_C)
            | self.sz_xy(res)
            | FLAG_N
            | if h { FLAG_H } else { 0 }
            | if pv { FLAG_PV } else { 0 };
        res
    }

    fn add16(&mut self, a: u16, b: u16) -> u16 {
        let sum = (a as u32) + (b as u32);
        let res = sum as u16;
        let h = ((a & 0x0FFF) + (b & 0x0FFF)) > 0x0FFF;
        self.f = (self.f & (FLAG_S | FLAG_Z | FLAG_PV))
            | (((res >> 8) as u8) & (FLAG_X | FLAG_Y))
            | if h { FLAG_H } else { 0 }
            | if sum > 0xFFFF { FLAG_C } else { 0 };
        res
    }

    fn adc_hl(&mut self, rr: u16) {
        let hl = self.hl();
        let cin = if (self.f & FLAG_C) != 0 { 1u32 } else { 0u32 };
        let sum = (hl as u32) + (rr as u32) + cin;
        let res = sum as u16;
        let h = ((hl & 0x0FFF) as u32 + (rr & 0x0FFF) as u32 + cin) > 0x0FFF;
        let pv = ((hl ^ res) & (rr ^ res) & 0x8000) != 0;
        self.set_hl(res);
        self.f = (((res >> 8) as u8) & (FLAG_S | FLAG_X | FLAG_Y))
            | if res == 0 { FLAG_Z } else { 0 }
            | if h { FLAG_H } else { 0 }
            | if pv { FLAG_PV } else { 0 }
            | if sum > 0xFFFF { FLAG_C } else { 0 };
    }

    fn sbc_hl(&mut self, rr: u16) {
        let hl = self.hl();
        let cin = if (self.f & FLAG_C) != 0 { 1i32 } else { 0i32 };
        let diff = (hl as i32) - (rr as i32) - cin;
        let res = diff as u16;
        let h = ((hl & 0x0FFF) as i32 - (rr & 0x0FFF) as i32 - cin) < 0;
        let pv = ((hl ^ rr) & (hl ^ res) & 0x8000) != 0;
        self.set_hl(res);
        self.f = (((res >> 8) as u8) & (FLAG_S | FLAG_X | FLAG_Y))
            | FLAG_N
            | if res == 0 { FLAG_Z } else { 0 }
            | if h { FLAG_H } else { 0 }
            | if pv { FLAG_PV } else { 0 }
            | if diff < 0 { FLAG_C } else { 0 };
    }

    fn rot_shift(&mut self, op: u8, v: u8) -> u8 {
        let cin = if (self.f & FLAG_C) != 0 { 1u8 } else { 0u8 };
        let (res, cout) = match op & 7 {
            0 => (v.rotate_left(1), (v & 0x80) != 0),           // RLC
            1 => (v.rotate_right(1), (v & 0x01) != 0),          // RRC
            2 => ((v << 1) | cin, (v & 0x80) != 0),             // RL
            3 => ((v >> 1) | (cin << 7), (v & 0x01) != 0),      // RR
            4 => (v << 1, (v & 0x80) != 0),                     // SLA
            5 => ((v >> 1) | (v & 0x80), (v & 0x01) != 0),      // SRA
            6 => ((v << 1) | 1, (v & 0x80) != 0),               // SLL
            7 => (v >> 1, (v & 0x01) != 0),                     // SRL
            _ => unreachable!(),
        };
        self.f = self.sz_xy(res)
            | if parity_even(res) { FLAG_PV } else { 0 }
            | if cout { FLAG_C } else { 0 };
        res
    }

    fn daa(&mut self) {
        let mut a = self.a;
        let mut corr = 0u8;
        let mut carry = (self.f & FLAG_C) != 0;
        if (self.f & FLAG_H) != 0 || (a & 0x0F) > 9 {
            corr |= 0x06;
        }
        if carry || a > 0x99 {
            corr |= 0x60;
            carry = true;
        }
        let h = if (self.f & FLAG_N) != 0 {
            let new_h = (a & 0x0F) < (corr & 0x0F);
            a = a.wrapping_sub(corr);
            new_h
        } else {
            let new_h = (a & 0x0F) + (corr & 0x0F) > 0x0F;
            a = a.wrapping_add(corr);
            new_h
        };
        self.a = a;
        self.f = self.sz_xy(a)
            | (self.f & FLAG_N)
            | if h { FLAG_H } else { 0 }
            | if parity_even(a) { FLAG_PV } else { 0 }
            | if carry { FLAG_C } else { 0 };
    }

    fn bit_test(&mut self, bit: u8, val: u8) {
        let z = (val & (1u8 << bit)) == 0;
        self.f = (self.f & FLAG_C)
            | FLAG_H
            | if z { FLAG_Z | FLAG_PV } else { 0 }
            | if bit == 7 && !z { FLAG_S } else { 0 }
            | (val & (FLAG_X | FLAG_Y));
    }

    fn step(&mut self, hal: &mut MockHal) {
        if self.halted {
            self.tstates += 4;
            return;
        }
        self.steps += 1;
        let start_pc = self.pc;
        let mut mode = IndexMode::Hl;
        let mut op = self.fetch8();
        let mut prefix_t: u64 = 0;

        while op == 0xDD || op == 0xFD {
            mode = if op == 0xDD {
                IndexMode::Ix
            } else {
                IndexMode::Iy
            };
            prefix_t += 4;
            op = self.fetch8();
        }

        if op == 0xED {
            self.step_ed(start_pc, hal);
            return;
        }
        if op == 0xCB {
            match mode {
                IndexMode::Hl => self.step_cb(),
                IndexMode::Ix | IndexMode::Iy => self.step_idx_cb(mode),
            }
            return;
        }

        self.tstates += prefix_t;
        let x = op >> 6;
        let y = (op >> 3) & 7;
        let z = op & 7;
        let p = y >> 1;
        let q = y & 1;

        match x {
            0 => match z {
                0 => match y {
                    0 => self.tstates += 4, // NOP
                    1 => {
                        std::mem::swap(&mut self.a, &mut self.a_alt);
                        std::mem::swap(&mut self.f, &mut self.f_alt);
                        self.tstates += 4;
                    }
                    2 => {
                        let d = self.fetch8() as i8;
                        self.b = self.b.wrapping_sub(1);
                        if self.b != 0 {
                            self.pc = self.pc.wrapping_add_signed(d as i16);
                            self.tstates += 13;
                        } else {
                            self.tstates += 8;
                        }
                    }
                    3 => {
                        let d = self.fetch8() as i8;
                        self.pc = self.pc.wrapping_add_signed(d as i16);
                        self.tstates += 12;
                    }
                    4..=7 => {
                        let d = self.fetch8() as i8;
                        if self.cond(y - 4) {
                            self.pc = self.pc.wrapping_add_signed(d as i16);
                            self.tstates += 12;
                        } else {
                            self.tstates += 7;
                        }
                    }
                    _ => unreachable!(),
                },
                1 => {
                    if q == 0 {
                        let nn = self.fetch16();
                        self.set_rp(p, mode, nn);
                        self.tstates += 10;
                    } else {
                        let lhs = self.idx_reg(mode);
                        let rhs = self.get_rp(p, mode);
                        let res = self.add16(lhs, rhs);
                        self.set_idx_reg(mode, res);
                        self.tstates += 11;
                    }
                }
                2 => match y {
                    0 => {
                        self.write8(self.bc(), self.a);
                        self.tstates += 7;
                    }
                    1 => {
                        self.a = self.read8(self.bc());
                        self.tstates += 7;
                    }
                    2 => {
                        self.write8(self.de(), self.a);
                        self.tstates += 7;
                    }
                    3 => {
                        self.a = self.read8(self.de());
                        self.tstates += 7;
                    }
                    4 => {
                        let nn = self.fetch16();
                        let v = self.idx_reg(mode);
                        self.write16(nn, v);
                        self.tstates += 16;
                    }
                    5 => {
                        let nn = self.fetch16();
                        let v = self.read16(nn);
                        self.set_idx_reg(mode, v);
                        self.tstates += 16;
                    }
                    6 => {
                        let nn = self.fetch16();
                        self.write8(nn, self.a);
                        self.tstates += 13;
                    }
                    7 => {
                        let nn = self.fetch16();
                        self.a = self.read8(nn);
                        self.tstates += 13;
                    }
                    _ => unreachable!(),
                },
                3 => {
                    let v = self.get_rp(p, mode);
                    let res = if q == 0 {
                        v.wrapping_add(1)
                    } else {
                        v.wrapping_sub(1)
                    };
                    self.set_rp(p, mode, res);
                    self.tstates += 6;
                }
                4 => {
                    if y == 6 {
                        let addr = self.eff_addr(mode);
                        let v = self.read8(addr);
                        let res = self.inc8(v);
                        self.write8(addr, res);
                        self.tstates += if matches!(mode, IndexMode::Hl) { 11 } else { 19 };
                    } else {
                        let v = self.get_r8_direct(y, mode);
                        let res = self.inc8(v);
                        self.set_r8_direct(y, mode, res);
                        self.tstates += 4;
                    }
                }
                5 => {
                    if y == 6 {
                        let addr = self.eff_addr(mode);
                        let v = self.read8(addr);
                        let res = self.dec8(v);
                        self.write8(addr, res);
                        self.tstates += if matches!(mode, IndexMode::Hl) { 11 } else { 19 };
                    } else {
                        let v = self.get_r8_direct(y, mode);
                        let res = self.dec8(v);
                        self.set_r8_direct(y, mode, res);
                        self.tstates += 4;
                    }
                }
                6 => {
                    if y == 6 {
                        let addr = self.eff_addr(mode);
                        let n = self.fetch8();
                        self.write8(addr, n);
                        self.tstates += if matches!(mode, IndexMode::Hl) { 10 } else { 15 };
                    } else {
                        let n = self.fetch8();
                        self.set_r8_direct(y, mode, n);
                        self.tstates += 7;
                    }
                }
                7 => {
                    match y {
                        0 => {
                            // RLCA
                            let c = (self.a & 0x80) != 0;
                            self.a = self.a.rotate_left(1);
                            self.f = (self.f & (FLAG_S | FLAG_Z | FLAG_PV))
                                | (self.a & (FLAG_X | FLAG_Y))
                                | if c { FLAG_C } else { 0 };
                        }
                        1 => {
                            // RRCA
                            let c = (self.a & 0x01) != 0;
                            self.a = self.a.rotate_right(1);
                            self.f = (self.f & (FLAG_S | FLAG_Z | FLAG_PV))
                                | (self.a & (FLAG_X | FLAG_Y))
                                | if c { FLAG_C } else { 0 };
                        }
                        2 => {
                            // RLA
                            let cin = if (self.f & FLAG_C) != 0 { 1 } else { 0 };
                            let cout = (self.a & 0x80) != 0;
                            self.a = (self.a << 1) | cin;
                            self.f = (self.f & (FLAG_S | FLAG_Z | FLAG_PV))
                                | (self.a & (FLAG_X | FLAG_Y))
                                | if cout { FLAG_C } else { 0 };
                        }
                        3 => {
                            // RRA
                            let cin = if (self.f & FLAG_C) != 0 { 0x80 } else { 0 };
                            let cout = (self.a & 0x01) != 0;
                            self.a = (self.a >> 1) | cin;
                            self.f = (self.f & (FLAG_S | FLAG_Z | FLAG_PV))
                                | (self.a & (FLAG_X | FLAG_Y))
                                | if cout { FLAG_C } else { 0 };
                        }
                        4 => self.daa(),
                        5 => {
                            // CPL
                            self.a = !self.a;
                            self.f = (self.f & (FLAG_S | FLAG_Z | FLAG_PV | FLAG_C))
                                | (self.a & (FLAG_X | FLAG_Y))
                                | FLAG_H
                                | FLAG_N;
                        }
                        6 => {
                            // SCF
                            self.f = (self.f & (FLAG_S | FLAG_Z | FLAG_PV))
                                | (self.a & (FLAG_X | FLAG_Y))
                                | FLAG_C;
                        }
                        7 => {
                            // CCF
                            let old_c = (self.f & FLAG_C) != 0;
                            self.f = (self.f & (FLAG_S | FLAG_Z | FLAG_PV))
                                | (self.a & (FLAG_X | FLAG_Y))
                                | if old_c { FLAG_H } else { FLAG_C };
                        }
                        _ => unreachable!(),
                    }
                    self.tstates += 4;
                }
                _ => unreachable!(),
            },
            1 => {
                if y == 6 && z == 6 {
                    // HALT (not affected by DD/FD prefix)
                    self.halted = true;
                    self.tstates += 4;
                } else if y == 6 || z == 6 {
                    // When (IX+d)/(IY+d) is involved, the other H/L operand is NOT IXH/IXL
                    let addr = self.eff_addr(mode);
                    if y == 6 {
                        let v = self.get_r8_direct(z, IndexMode::Hl);
                        self.write8(addr, v);
                    } else {
                        let v = self.read8(addr);
                        self.set_r8_direct(y, IndexMode::Hl, v);
                    }
                    self.tstates += if matches!(mode, IndexMode::Hl) { 7 } else { 15 };
                } else {
                    let v = self.get_r8_direct(z, mode);
                    self.set_r8_direct(y, mode, v);
                    self.tstates += 4;
                }
            }
            2 => {
                let v = if z == 6 {
                    let addr = self.eff_addr(mode);
                    self.tstates += if matches!(mode, IndexMode::Hl) { 7 } else { 15 };
                    self.read8(addr)
                } else {
                    self.tstates += 4;
                    self.get_r8_direct(z, mode)
                };
                self.alu(y, v);
            }
            3 => match z {
                0 => {
                    if self.cond(y) {
                        self.pc = self.pop16();
                        self.tstates += 11;
                    } else {
                        self.tstates += 5;
                    }
                }
                1 => {
                    if q == 0 {
                        let v = self.pop16();
                        self.set_rp2(p, mode, v);
                        self.tstates += 10;
                    } else {
                        match p {
                            0 => {
                                self.pc = self.pop16();
                                self.tstates += 10;
                            }
                            1 => {
                                std::mem::swap(&mut self.b, &mut self.b_alt);
                                std::mem::swap(&mut self.c, &mut self.c_alt);
                                std::mem::swap(&mut self.d, &mut self.d_alt);
                                std::mem::swap(&mut self.e, &mut self.e_alt);
                                std::mem::swap(&mut self.h, &mut self.h_alt);
                                std::mem::swap(&mut self.l, &mut self.l_alt);
                                self.tstates += 4;
                            }
                            2 => {
                                self.pc = self.idx_reg(mode);
                                self.tstates += 4;
                            }
                            3 => {
                                self.sp = self.idx_reg(mode);
                                self.tstates += 6;
                            }
                            _ => unreachable!(),
                        }
                    }
                }
                2 => {
                    let nn = self.fetch16();
                    if self.cond(y) {
                        self.pc = nn;
                    }
                    self.tstates += 10;
                }
                3 => match y {
                    0 => {
                        self.pc = self.fetch16();
                        self.tstates += 10;
                    }
                    2 => {
                        let port = self.fetch8();
                        hal.on_out(port, self.a, self.l);
                        self.tstates += 11;
                    }
                    3 => {
                        let _port = self.fetch8();
                        self.a = 0;
                        self.tstates += 11;
                    }
                    4 => {
                        let top = self.read16(self.sp);
                        let cur = self.idx_reg(mode);
                        self.write16(self.sp, cur);
                        self.set_idx_reg(mode, top);
                        self.tstates += 19;
                    }
                    5 => {
                        // EX DE, HL (unaffected by DD/FD)
                        let de = self.de();
                        let hl = self.hl();
                        self.set_de(hl);
                        self.set_hl(de);
                        self.tstates += 4;
                    }
                    6 => {
                        self.iff1 = false;
                        self.iff2 = false;
                        self.tstates += 4;
                    }
                    7 => {
                        self.iff1 = true;
                        self.iff2 = true;
                        self.tstates += 4;
                    }
                    other => panic!("unsupported Z80 opcode 0x{op:02X} (y={other}) at PC=0x{start_pc:04X}"),
                },
                4 => {
                    let nn = self.fetch16();
                    if self.cond(y) {
                        self.push16(self.pc);
                        self.pc = nn;
                        self.tstates += 17;
                    } else {
                        self.tstates += 10;
                    }
                }
                5 => {
                    if q == 0 {
                        let v = self.get_rp2(p, mode);
                        self.push16(v);
                        self.tstates += 11;
                    } else if p == 0 {
                        let nn = self.fetch16();
                        self.push16(self.pc);
                        self.pc = nn;
                        self.tstates += 17;
                    } else {
                        unreachable!();
                    }
                }
                6 => {
                    let n = self.fetch8();
                    self.alu(y, n);
                    self.tstates += 7;
                }
                7 => {
                    self.push16(self.pc);
                    self.pc = (y as u16) * 8;
                    self.tstates += 11;
                }
                _ => unreachable!(),
            },
            _ => unreachable!(),
        }
    }

    #[inline]
    fn eff_addr(&mut self, mode: IndexMode) -> u16 {
        match mode {
            IndexMode::Hl => self.hl(),
            IndexMode::Ix => {
                let d = self.fetch8() as i8;
                self.ix.wrapping_add_signed(d as i16)
            }
            IndexMode::Iy => {
                let d = self.fetch8() as i8;
                self.iy.wrapping_add_signed(d as i16)
            }
        }
    }

    fn step_cb(&mut self) {
        let op = self.fetch8();
        let x = op >> 6;
        let y = (op >> 3) & 7;
        let z = op & 7;
        let v = if z == 6 {
            self.read8(self.hl())
        } else {
            self.get_r8_direct(z, IndexMode::Hl)
        };

        match x {
            0 => {
                let res = self.rot_shift(y, v);
                if z == 6 {
                    self.write8(self.hl(), res);
                    self.tstates += 15;
                } else {
                    self.set_r8_direct(z, IndexMode::Hl, res);
                    self.tstates += 8;
                }
            }
            1 => {
                self.bit_test(y, v);
                self.tstates += if z == 6 { 12 } else { 8 };
            }
            2 => {
                let res = v & !(1u8 << y);
                if z == 6 {
                    self.write8(self.hl(), res);
                    self.tstates += 15;
                } else {
                    self.set_r8_direct(z, IndexMode::Hl, res);
                    self.tstates += 8;
                }
            }
            3 => {
                let res = v | (1u8 << y);
                if z == 6 {
                    self.write8(self.hl(), res);
                    self.tstates += 15;
                } else {
                    self.set_r8_direct(z, IndexMode::Hl, res);
                    self.tstates += 8;
                }
            }
            _ => unreachable!(),
        }
    }

    fn step_idx_cb(&mut self, mode: IndexMode) {
        let d = self.fetch8() as i8;
        let op = self.fetch8();
        let addr = self.idx_reg(mode).wrapping_add_signed(d as i16);
        let x = op >> 6;
        let y = (op >> 3) & 7;
        let z = op & 7;
        let v = self.read8(addr);

        match x {
            0 => {
                let res = self.rot_shift(y, v);
                self.write8(addr, res);
                if z != 6 {
                    self.set_r8_direct(z, IndexMode::Hl, res);
                }
                self.tstates += 23;
            }
            1 => {
                self.bit_test(y, v);
                self.tstates += 20;
            }
            2 => {
                let res = v & !(1u8 << y);
                self.write8(addr, res);
                if z != 6 {
                    self.set_r8_direct(z, IndexMode::Hl, res);
                }
                self.tstates += 23;
            }
            3 => {
                let res = v | (1u8 << y);
                self.write8(addr, res);
                if z != 6 {
                    self.set_r8_direct(z, IndexMode::Hl, res);
                }
                self.tstates += 23;
            }
            _ => unreachable!(),
        }
    }

    fn step_ed(&mut self, start_pc: u16, hal: &mut MockHal) {
        let op = self.fetch8();
        let x = op >> 6;
        let y = (op >> 3) & 7;
        let z = op & 7;
        let p = y >> 1;
        let q = y & 1;

        match x {
            1 => match z {
                0 => {
                    let v = 0u8;
                    if y != 6 {
                        self.set_r8_direct(y, IndexMode::Hl, v);
                    }
                    self.f = (self.f & FLAG_C) | self.sz_xy(v) | if parity_even(v) { FLAG_PV } else { 0 };
                    self.tstates += 12;
                }
                1 => {
                    let v = if y == 6 {
                        0
                    } else {
                        self.get_r8_direct(y, IndexMode::Hl)
                    };
                    hal.on_out(self.c, v, self.l);
                    self.tstates += 12;
                }
                2 => {
                    let rr = self.get_rp(p, IndexMode::Hl);
                    if q == 0 {
                        self.sbc_hl(rr);
                    } else {
                        self.adc_hl(rr);
                    }
                    self.tstates += 15;
                }
                3 => {
                    let nn = self.fetch16();
                    if q == 0 {
                        let v = self.get_rp(p, IndexMode::Hl);
                        self.write16(nn, v);
                    } else {
                        let v = self.read16(nn);
                        self.set_rp(p, IndexMode::Hl, v);
                    }
                    self.tstates += 20;
                }
                4 => {
                    // NEG
                    let a = self.a;
                    self.a = 0;
                    self.alu(2, a);
                    self.tstates += 8;
                }
                5 => {
                    // RETN / RETI
                    self.iff1 = self.iff2;
                    self.pc = self.pop16();
                    self.tstates += 14;
                }
                6 => {
                    self.im = match y {
                        0 | 1 | 4 | 5 => 0,
                        2 | 6 => 1,
                        3 | 7 => 2,
                        _ => 0,
                    };
                    self.tstates += 8;
                }
                7 => {
                    match y {
                        0 => {
                            self.i = self.a;
                            self.tstates += 9;
                        }
                        1 => {
                            self.r = self.a;
                            self.tstates += 9;
                        }
                        2 => {
                            self.a = self.i;
                            self.f = (self.f & FLAG_C) | self.sz_xy(self.a) | if self.iff2 { FLAG_PV } else { 0 };
                            self.tstates += 9;
                        }
                        3 => {
                            self.a = self.r;
                            self.f = (self.f & FLAG_C) | self.sz_xy(self.a) | if self.iff2 { FLAG_PV } else { 0 };
                            self.tstates += 9;
                        }
                        4 => {
                            // RRD
                            let m = self.read8(self.hl());
                            let new_m = (m >> 4) | (self.a << 4);
                            self.a = (self.a & 0xF0) | (m & 0x0F);
                            self.write8(self.hl(), new_m);
                            self.f = (self.f & FLAG_C) | self.sz_xy(self.a) | if parity_even(self.a) { FLAG_PV } else { 0 };
                            self.tstates += 18;
                        }
                        5 => {
                            // RLD
                            let m = self.read8(self.hl());
                            let new_m = (m << 4) | (self.a & 0x0F);
                            self.a = (self.a & 0xF0) | (m >> 4);
                            self.write8(self.hl(), new_m);
                            self.f = (self.f & FLAG_C) | self.sz_xy(self.a) | if parity_even(self.a) { FLAG_PV } else { 0 };
                            self.tstates += 18;
                        }
                        _ => self.tstates += 8,
                    }
                }
                _ => unreachable!(),
            },
            2 => match op {
                0xA0 | 0xA8 | 0xB0 | 0xB8 => {
                    // LDI, LDD, LDIR, LDDR
                    let dec = (op & 0x08) != 0;
                    let rep = (op & 0x10) != 0;
                    let val = self.read8(self.hl());
                    self.write8(self.de(), val);
                    if dec {
                        self.set_hl(self.hl().wrapping_sub(1));
                        self.set_de(self.de().wrapping_sub(1));
                    } else {
                        self.set_hl(self.hl().wrapping_add(1));
                        self.set_de(self.de().wrapping_add(1));
                    }
                    let bc = self.bc().wrapping_sub(1);
                    self.set_bc(bc);
                    self.f = (self.f & (FLAG_S | FLAG_Z | FLAG_C)) | if bc != 0 { FLAG_PV } else { 0 };
                    if rep && bc != 0 {
                        self.pc = self.pc.wrapping_sub(2);
                        self.tstates += 21;
                    } else {
                        self.tstates += 16;
                    }
                }
                0xA1 | 0xA9 | 0xB1 | 0xB9 => {
                    // CPI, CPD, CPIR, CPDR
                    let dec = (op & 0x08) != 0;
                    let rep = (op & 0x10) != 0;
                    let val = self.read8(self.hl());
                    let diff = (self.a as i16) - (val as i16);
                    let res = diff as u8;
                    let h = (self.a & 0x0F) < (val & 0x0F);
                    if dec {
                        self.set_hl(self.hl().wrapping_sub(1));
                    } else {
                        self.set_hl(self.hl().wrapping_add(1));
                    }
                    let bc = self.bc().wrapping_sub(1);
                    self.set_bc(bc);
                    self.f = (self.f & FLAG_C)
                        | (res & FLAG_S)
                        | if res == 0 { FLAG_Z } else { 0 }
                        | if h { FLAG_H } else { 0 }
                        | if bc != 0 { FLAG_PV } else { 0 }
                        | FLAG_N;
                    if rep && bc != 0 && res != 0 {
                        self.pc = self.pc.wrapping_sub(2);
                        self.tstates += 21;
                    } else {
                        self.tstates += 16;
                    }
                }
                other => panic!("unsupported Z80 ED opcode 0xED 0x{other:02X} at PC=0x{start_pc:04X}"),
            },
            _ => panic!("unsupported Z80 ED opcode 0xED 0x{op:02X} at PC=0x{start_pc:04X}"),
        }
    }

    fn call(&mut self, hal: &mut MockHal, addr: u16, hl_arg: u16, max_steps: u64) -> u64 {
        self.sp = 0xF000;
        self.write8(0xFFFF, 0x76); // HALT sentinel
        self.push16(0xFFFF);
        self.set_hl(hl_arg);
        self.pc = addr;
        self.halted = false;
        let start_t = self.tstates;
        let start_steps = self.steps;

        while !self.halted {
            if self.steps - start_steps >= max_steps {
                panic!(
                    "Z80 execution exceeded max_steps={max_steps} at PC=0x{:04X} (entry=0x{addr:04X}, SP=0x{:04X}, AF=0x{:04X}, BC=0x{:04X}, DE=0x{:04X}, HL=0x{:04X}, IX=0x{:04X}, IY=0x{:04X})",
                    self.pc,
                    self.sp,
                    self.af(),
                    self.bc(),
                    self.de(),
                    self.hl(),
                    self.ix,
                    self.iy,
                );
            }
            self.step(hal);
        }
        assert_eq!(
            self.pc, 0x0000, // fetched 0x76 at 0xFFFF -> PC wrapped to 0x0000
            "expected HALT at 0xFFFF, halted at PC=0x{:04X}",
            self.pc.wrapping_sub(1)
        );
        self.tstates - start_t
    }
}

// ---------------------------------------------------------------------------
// Assembler / Linker / Intel HEX / Map Loader
// ---------------------------------------------------------------------------

struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    fn new(tag: &str) -> Self {
        let mut dir = std::env::temp_dir();
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        dir.push(format!("aun_z80_e2e_{tag}_{pid}_{nanos}"));
        fs::create_dir_all(&dir).expect("create temp dir");
        Self { path: dir }
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

struct LinkedImage {
    ram: Vec<u8>,
    symbols: HashMap<String, u16>,
    init_copy: Option<(u16, u16, u16)>, // (src_initializer, dst_initialized, len)
}

fn assemble_sdasz80(s_path: &Path, rel_path: &Path) {
    let out = Command::new("sdasz80")
        .arg("-lo")
        .arg(rel_path)
        .arg(s_path)
        .output()
        .expect("run sdasz80");
    if !out.status.success() {
        panic!(
            "sdasz80 failed on {}:\nstdout: {}\nstderr: {}",
            s_path.display(),
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

fn link_sdldz80(out_base: &Path, rels: &[&Path]) -> LinkedImage {
    let mut cmd = Command::new("sdldz80");
    cmd.arg("-n")
        .arg("-m")
        .arg("-i")
        .arg("-j")
        .arg("-b")
        .arg("_CODE=0x0100")
        .arg("-b")
        .arg("_INITIALIZER=0x2800")
        .arg("-b")
        .arg("_DATA=0xC000")
        .arg("-b")
        .arg("_INITIALIZED=0xC400")
        .arg(out_base);
    for rel in rels {
        cmd.arg(rel);
    }
    let out = cmd.output().expect("run sdldz80");
    if !out.status.success() {
        panic!(
            "sdldz80 failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }

    let ihx_path = out_base.with_extension("ihx");
    let noi_path = out_base.with_extension("noi");
    let map_path = out_base.with_extension("map");

    let ram = parse_ihx(&fs::read_to_string(&ihx_path).expect("read .ihx"));
    let symbols = parse_noi(&fs::read_to_string(&noi_path).expect("read .noi"));
    let init_copy = parse_map_initializer(&fs::read_to_string(&map_path).expect("read .map"));

    LinkedImage {
        ram,
        symbols,
        init_copy,
    }
}

fn parse_ihx(ihx: &str) -> Vec<u8> {
    let mut ram = vec![0u8; 65536];
    for line in ihx.lines() {
        let line = line.trim();
        if line.is_empty() || !line.starts_with(':') {
            continue;
        }
        let byte_count = usize::from_str_radix(&line[1..3], 16).expect("ihx byte_count");
        let addr = usize::from_str_radix(&line[3..7], 16).expect("ihx addr");
        let rec_type = u8::from_str_radix(&line[7..9], 16).expect("ihx rec_type");
        if rec_type == 0x00 {
            for i in 0..byte_count {
                let off = 9 + i * 2;
                let b = u8::from_str_radix(&line[off..off + 2], 16).expect("ihx data byte");
                ram[(addr + i) & 0xFFFF] = b;
            }
        } else if rec_type == 0x01 {
            break;
        }
    }
    ram
}

fn parse_noi(noi: &str) -> HashMap<String, u16> {
    let mut syms = HashMap::new();
    for line in noi.lines() {
        let mut parts = line.split_whitespace();
        if parts.next() == Some("DEF")
            && let (Some(name), Some(addr_str)) = (parts.next(), parts.next())
        {
            let hex = addr_str.trim_start_matches("0x").trim_start_matches("0X");
            if let Ok(addr) = u32::from_str_radix(hex, 16) {
                syms.insert(name.to_string(), addr as u16);
            }
        }
    }
    syms
}

fn parse_map_initializer(map: &str) -> Option<(u16, u16, u16)> {
    let mut initer_addr = None;
    let mut initer_size = None;
    let mut inited_addr = None;
    for line in map.lines() {
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() >= 3 {
            if cols[0] == "_INITIALIZER"
                && let (Ok(a), Ok(s)) = (u16::from_str_radix(cols[1], 16), u16::from_str_radix(cols[2], 16))
            {
                initer_addr = Some(a);
                initer_size = Some(s);
            } else if cols[0] == "_INITIALIZED"
                && let Ok(a) = u16::from_str_radix(cols[1], 16)
            {
                inited_addr = Some(a);
            }
        }
    }
    match (initer_addr, inited_addr, initer_size) {
        (Some(src), Some(dst), Some(len)) if len > 0 => Some((src, dst, len)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Driver Harness (calls exported sdcccall(1) entry points on Z80)
// ---------------------------------------------------------------------------

struct DriverVm {
    cpu: Z80,
    hal: MockHal,
    symbols: HashMap<String, u16>,
    init_copy: Option<(u16, u16, u16)>,
    base_ram: Vec<u8>,
    total_cycles: u64,
}

impl DriverVm {
    fn from_image(img: LinkedImage) -> Self {
        let mut vm = Self {
            cpu: Z80::new(),
            hal: MockHal::default(),
            symbols: img.symbols,
            init_copy: img.init_copy,
            base_ram: img.ram,
            total_cycles: 0,
        };
        vm.reset();
        vm
    }

    fn reset(&mut self) {
        self.cpu = Z80::new();
        self.cpu.ram = self.base_ram.clone();
        if let Some((src, dst, len)) = self.init_copy {
            for i in 0..len {
                let b = self.cpu.read8(src.wrapping_add(i));
                self.cpu.write8(dst.wrapping_add(i), b);
            }
        }
        self.hal.reset();
        self.total_cycles = 0;
    }

    fn load_data(&mut self, addr: u16, data: &[u8]) {
        for (i, &b) in data.iter().enumerate() {
            self.cpu.write8(addr.wrapping_add(i as u16), b);
        }
    }

    fn sym(&self, name: &str) -> u16 {
        *self
            .symbols
            .get(name)
            .unwrap_or_else(|| panic!("missing symbol `{name}`"))
    }

    fn call_void(&mut self, name: &str) {
        let addr = self.sym(name);
        self.total_cycles += self.cpu.call(&mut self.hal, addr, 0, 500_000);
    }

    fn call_u16(&mut self, name: &str, arg: u16) {
        let addr = self.sym(name);
        self.total_cycles += self.cpu.call(&mut self.hal, addr, arg, 500_000);
    }

    fn call_ret_u8(&mut self, name: &str) -> u8 {
        let addr = self.sym(name);
        self.total_cycles += self.cpu.call(&mut self.hal, addr, 0, 500_000);
        self.cpu.a
    }
}

// ---------------------------------------------------------------------------
// Ported Driver Host Scenarios (test_driver_host.c)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
struct ScenarioSnapshot {
    regs_after_fm: [u8; 256],
    writes_fm: Vec<(u8, u8)>,
    ticks_ssg_regs: Vec<[u8; 256]>,
    writes_ssg: Vec<(u8, u8)>,
    total_cycles: u64,
}

fn run_both_scenarios(vm: &mut DriverVm) -> ScenarioSnapshot {
    // -----------------------------------------------------------------------
    // 1. test_fm_con4_and_pitch
    // -----------------------------------------------------------------------
    #[rustfmt::skip]
    const SONG: &[u8] = &[
        b'M', b'D', b'X', b'T', 6, 220,
        20, 0,          // voice_tbl_off = 20
        46, 0,          // ch A off = 46
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Voice @0 (26 bytes): DT/ML, TL=0, KS/AR, DR, SR, SL/RR, FB/CON=0x04
        1, 1, 1, 1,
        0, 0, 0, 0,
        31, 31, 31, 31,
        0, 0, 0, 0,
        0, 0, 0, 0,
        15, 15, 15, 15,
        0x04, 0,
        // Ch A stream: @0, v8 (FM_VOL_TL[8] = 16), note 57 (o4 a) for 16 ticks, end
        0xFD, 0,
        0xFB, 8,
        0x80 + 57, 15,
        0xF1, 0, 0,
    ];

    vm.reset();
    vm.load_data(0x8000, SONG);
    vm.call_void("_mdxnt_init");
    assert!(vm.hal.irq_enabled, "mdxnt_init should enable IRQ");
    assert_eq!(vm.hal.last_tb, 220, "mdxnt_init should set Timer-B = 220");

    vm.call_u16("_mdxnt_play_bgm", 0x8000);
    vm.call_void("_mdxnt_tick_isr");

    // CON 4 carriers are OP2 (0x48) and OP4 (0x4C); modulators OP1 (0x40) & OP3 (0x44) stay 0
    assert_eq!(vm.hal.regs[0x40], 0, "OP1 TL should stay 0");
    assert_eq!(vm.hal.regs[0x44], 0, "OP3 TL should stay 0");
    assert_eq!(vm.hal.regs[0x48], 16, "OP2 TL should be 16");
    assert_eq!(vm.hal.regs[0x4C], 16, "OP4 TL should be 16");

    // A4 = Block 4, F_num = 1040 (0x410): reg 0xA4 = (4 << 3) | 4 = 0x24, reg 0xA0 = 0x10
    assert_eq!(vm.hal.regs[0xA4], 0x24, "FM A4 high reg mismatch");
    assert_eq!(vm.hal.regs[0xA0], 0x10, "FM A4 low reg mismatch");

    let regs_after_fm = vm.hal.regs;
    vm.call_void("_mdxnt_shutdown");
    assert!(!vm.hal.irq_enabled, "mdxnt_shutdown should disable IRQ");
    assert!(!vm.hal.in_crit, "left in critical section after shutdown");
    let writes_fm = vm.hal.filtered_opn_writes();

    // -----------------------------------------------------------------------
    // 2. test_ssg_tie_and_se_restore
    // -----------------------------------------------------------------------
    #[rustfmt::skip]
    const BGM: &[u8] = &[
        b'M', b'D', b'X', b'T', 6, 220,
        0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        20, 0,          // ch E (SSG ch.B = hw_ch 1)
        0, 0,
        // Ch E: @0, v12, q4, note 81 (o6 a, TP=71) 8t, v15, &, note 81 8t, end
        0xFD, 0,
        0xFB, 12,
        0xF8, 4,
        0x80 + 81, 7,
        0xFB, 15,
        0xF7,
        0x80 + 81, 7,
        0xF1, 0, 0,
    ];
    #[rustfmt::skip]
    const SE: &[u8] = &[
        b'M', b'D', b'X', b'T', 6, 220,
        0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        20, 0,
        0, 0,
        0xFD, 0,
        0xFB, 10,
        0x80 + 48, 1,
        0xF1, 0, 0,
    ];

    let fm_cycles = vm.total_cycles;
    vm.reset();
    vm.load_data(0x8000, BGM);
    vm.load_data(0x9000, SE);

    let mut ticks_ssg_regs = Vec::new();
    vm.call_void("_mdxnt_init");
    vm.call_u16("_mdxnt_play_bgm", 0x8000);

    // Tick 1: BGM starts o6 a -> rounded TP = (2269 + 16) >> 5 = 71, vol = 12
    vm.call_void("_mdxnt_tick_isr");
    let tp1 = (vm.hal.regs[2] as u16) | ((vm.hal.regs[3] as u16) << 8);
    assert_eq!(tp1, 71, "Tick 1 SSG TP expected 71");
    assert_eq!(vm.hal.regs[9], 12, "Tick 1 SSG vol expected 12");
    ticks_ssg_regs.push(vm.hal.regs);

    // Ticks 2..5: tie lookahead across v15 must keep note sustained (no gate-off at tick 4!)
    for _ in 0..4 {
        vm.call_void("_mdxnt_tick_isr");
        ticks_ssg_regs.push(vm.hal.regs);
    }
    assert_eq!(vm.hal.regs[9], 12, "Tick 5 SSG vol expected 12");

    // Trigger SE on SSG ch.B: SE hijacks hw_ch 1 with TP=477, vol=10
    vm.call_u16("_mdxnt_play_se", 0x9000);
    assert_eq!(vm.call_ret_u8("_mdxnt_se_busy"), 1, "SE should be busy");

    vm.call_void("_mdxnt_tick_isr"); // SE tick 1
    let tp_se = (vm.hal.regs[2] as u16) | ((vm.hal.regs[3] as u16) << 8);
    assert_eq!(tp_se, 477, "SE tick 1 SSG TP expected 477");
    assert_eq!(vm.hal.regs[9], 10, "SE tick 1 SSG vol expected 10");
    ticks_ssg_regs.push(vm.hal.regs);

    vm.call_void("_mdxnt_tick_isr"); // Tick 7: SE tick 2
    ticks_ssg_regs.push(vm.hal.regs);

    vm.call_void("_mdxnt_tick_isr"); // Tick 8: SE ends -> BGM restores (TP=71, vol=12)
    assert_eq!(vm.call_ret_u8("_mdxnt_se_busy"), 0, "SE should have finished");
    let tp_restored = (vm.hal.regs[2] as u16) | ((vm.hal.regs[3] as u16) << 8);
    assert_eq!(tp_restored, 71, "BGM restore SSG TP expected 71");
    assert_eq!(vm.hal.regs[9], 12, "BGM restore SSG vol expected 12");
    ticks_ssg_regs.push(vm.hal.regs);

    vm.call_void("_mdxnt_tick_isr"); // Tick 9: BGM executes v15 & tied note -> vol updates to 15
    assert_eq!(vm.hal.regs[9], 15, "Tick 9 SSG vol expected 15");
    ticks_ssg_regs.push(vm.hal.regs);

    vm.call_void("_mdxnt_shutdown");
    assert!(!vm.hal.in_crit, "left in critical section after shutdown");
    let writes_ssg = vm.hal.filtered_opn_writes();

    ScenarioSnapshot {
        regs_after_fm,
        writes_fm,
        ticks_ssg_regs,
        writes_ssg,
        total_cycles: fm_cycles + vm.total_cycles,
    }
}

#[test]
fn z80_e2e_aun_mdxnt_driver_matches_mock_and_hand_asm() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir.join("..").join("..");
    let aun_src_path = repo_root.join("bench/mdxnt/src/mdxnt_drv.aun");
    let hand_rel_path = repo_root.join("bench/mdxnt/build/mdxnt_drv_z80.rel");

    let aun_src = fs::read_to_string(&aun_src_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", aun_src_path.display()));
    let prog = aun_front::compile_str(0, &aun_src)
        .unwrap_or_else(|errs| panic!("aun_front compile failed: {errs:?}"));

    let (sdas, ram_bytes, max_stack) = compile_z80_sdasz80(
        "mdxnt_drv",
        &prog,
        CompileOptions {
            ram_base: 0xC000,
            ..CompileOptions::release()
        },
    )
    .expect("compile_z80_sdasz80 failed");

    let tmp = TempDirGuard::new("mdxnt");
    let hal_s = tmp.path.join("mock_hal.s");
    let hal_rel = tmp.path.join("mock_hal.rel");
    fs::write(&hal_s, MOCK_HAL_ASM).expect("write mock_hal.s");
    assemble_sdasz80(&hal_s, &hal_rel);

    // 1. Freshly compiled Aun Z80 machine code
    let aun_s = tmp.path.join("mdxnt_drv_aun.s");
    let aun_rel = tmp.path.join("mdxnt_drv_aun.rel");
    fs::write(&aun_s, &sdas).expect("write mdxnt_drv_aun.s");
    assemble_sdasz80(&aun_s, &aun_rel);

    let aun_img = link_sdldz80(
        &tmp.path.join("aun_test"),
        &[hal_rel.as_path(), aun_rel.as_path()],
    );
    let mut aun_vm = DriverVm::from_image(aun_img);
    let aun_snap = run_both_scenarios(&mut aun_vm);

    eprintln!(
        "[z80_e2e] Aun mdxnt_drv: RAM={ram_bytes}B, max_stack={max_stack}B, total_cycles={}",
        aun_snap.total_cycles
    );

    // 2. Compare against hand-written Z80 ASM reference if present
    if hand_rel_path.exists() {
        let hand_img = link_sdldz80(
            &tmp.path.join("hand_test"),
            &[hal_rel.as_path(), hand_rel_path.as_path()],
        );
        let mut hand_vm = DriverVm::from_image(hand_img);
        let hand_snap = run_both_scenarios(&mut hand_vm);

        eprintln!(
            "[z80_e2e] Hand Z80 ASM mdxnt_drv_z80: total_cycles={}",
            hand_snap.total_cycles
        );

        assert_eq!(
            aun_snap.regs_after_fm, hand_snap.regs_after_fm,
            "FM final OPN register state differs between Aun and hand Z80 ASM"
        );
        assert_eq!(
            aun_snap.ticks_ssg_regs, hand_snap.ticks_ssg_regs,
            "Per-tick SSG OPN register snapshots differ between Aun and hand Z80 ASM"
        );
        assert_eq!(
            aun_snap.writes_fm, hand_snap.writes_fm,
            "FM OPN write sequence differs between Aun and hand Z80 ASM"
        );
        assert_eq!(
            aun_snap.writes_ssg, hand_snap.writes_ssg,
            "SSG OPN write sequence differs between Aun and hand Z80 ASM"
        );
    }
}
