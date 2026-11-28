//! Self-contained Z80 machine-code E2E verification test for `mdxnt_drv.aun`
//! (and comparison against the hand-written Z80 reference `mdxnt_drv_z80.rel`).

use aun_z80::{CompileOptions, compile_z80, compile_z80_sdasz80};
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
    let aun_src_path = repo_root.join("bench/suite/mdxnt_drv.aun");
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

#[test]
fn z80_e2e_u32_and_i32_ops_and_sdcccall1_abi() {
    let src = r#"
struct ScoreBox {
    lo: u8,
    total: u32,
    delta: i16,
}

static mut G_BOX: ScoreBox;

pub fn ret_const_u32() -> u32 {
    0x12345678u32
}

pub fn score_accum(cur: u32, bonus: u32) -> u32 {
    let mut s: u32 = cur +% bonus;
    if s > 999999999u32 {
        s = 999999999u32;
    }
    s
}

pub fn arg_u8_u32(a: u8, b: u32) -> u32 {
    b +% (a as u32)
}

pub fn arg_u8_u8_u32(a: u8, b: u8, c: u32) -> u32 {
    c +% (a as u32) +% (b as u32)
}

pub fn arg_u32_u8(a: u32, b: u8) -> u32 {
    a -% (b as u32)
}

pub fn bit_and_shift_u32(x: u32, sh: u8) -> u32 {
    let masked: u32 = (x & 0x0FF0F00Fu32) | 0x00010000u32;
    let moved: u32 = ((x << 4u32) >> (sh as u32)) >> 16u32;
    masked ^ (moved << 16u32)
}

pub fn mul_div_rem_u32(a: u32, b: u32) -> u32 {
    ((a *% 1000u32) / b) +% (a % b)
}

pub fn signed_i32_ops(a: i32, b: i32) -> i32 {
    let mut r: i32 = -a;
    if a < b && b >= -100i32 {
        r = (a / b) +% (a % b);
    }
    let q: i32 = r >> 2i32;
    q -% 1i32
}

fn add_box(boxp: &mut ScoreBox, add: u16, neg8: i8) -> u32 {
    boxp.total = boxp.total +% (add as u32) +% ((neg8 as i32) as u32);
    boxp.total
}

pub fn run_box_test(init_v: u32, add: u16, neg8: i8) -> u32 {
    G_BOX.lo = 7u8;
    G_BOX.total = init_v;
    G_BOX.delta = neg8 as i16;
    add_box(&mut G_BOX, add, neg8)
}
"#;

    let prog = aun_front::compile_str(0, src).expect("32-bit aun_front compile should succeed");
    // 1. Verify built-in assembler encodes every emitted 32-bit instruction cleanly.
    let built_in = compile_z80(&prog).expect("built-in compile_z80 should assemble all u32/i32 code");
    assert!(built_in.symbols.contains_key("ret_const_u32"));
    assert!(built_in.symbols.contains_key("__aun_mul32"));
    assert!(built_in.symbols.contains_key("__aun_udiv32"));
    assert!(built_in.symbols.contains_key("__aun_sdiv32"));

    // 2. Compile via sdasz80 + sdldz80 and execute in Z80 emulator under sdcccall(1).
    let (sdas, _ram_bytes, _max_stack) = compile_z80_sdasz80(
        "u32_test",
        &prog,
        CompileOptions {
            ram_base: 0xC000,
            ..CompileOptions::release()
        },
    )
    .expect("compile_z80_sdasz80 failed");

    let tmp = TempDirGuard::new("u32");
    let hal_s = tmp.path.join("mock_hal.s");
    let hal_rel = tmp.path.join("mock_hal.rel");
    fs::write(&hal_s, MOCK_HAL_ASM).expect("write mock_hal.s");
    assemble_sdasz80(&hal_s, &hal_rel);

    let asm_path = tmp.path.join("u32_test.s");
    let rel_path = tmp.path.join("u32_test.rel");
    fs::write(&asm_path, &sdas).expect("write u32_test.s");
    assemble_sdasz80(&asm_path, &rel_path);
    let img = link_sdldz80(
        &tmp.path.join("u32_img"),
        &[hal_rel.as_path(), rel_path.as_path()],
    );

    #[derive(Clone, Copy)]
    enum StackArg {
        U8(u8),
        U16(u16),
        U32(u32),
    }

    let mut vm = DriverVm::from_image(img);
    let call32 = |vm: &mut DriverVm,
                  sym: &str,
                  setup: &dyn Fn(&mut Z80),
                  stack_args_rtl: &[StackArg]| -> u32 {
        let addr = vm.sym(sym);
        vm.cpu.sp = 0xF000;
        for &arg in stack_args_rtl {
            match arg {
                StackArg::U8(b) => {
                    vm.cpu.sp = vm.cpu.sp.wrapping_sub(1);
                    vm.cpu.write8(vm.cpu.sp, b);
                }
                StackArg::U16(w) => vm.cpu.push16(w),
                StackArg::U32(d) => {
                    vm.cpu.push16((d >> 16) as u16);
                    vm.cpu.push16(d as u16);
                }
            }
        }
        vm.cpu.write8(0xFFFF, 0x76);
        vm.cpu.push16(0xFFFF);
        setup(&mut vm.cpu);
        vm.cpu.pc = addr;
        vm.cpu.halted = false;
        while !vm.cpu.halted {
            assert!(vm.cpu.steps < 5_000_000, "Z80 step timeout calling {sym}");
            vm.cpu.step(&mut vm.hal);
        }
        assert_eq!(vm.cpu.sp, 0xF000, "sdcccall(1) stack cleanup mismatch on {sym}");
        ((vm.cpu.hl() as u32) << 16) | (vm.cpu.de() as u32)
    };
    let set_hlde = |cpu: &mut Z80, v: u32| {
        cpu.set_de(v as u16);
        cpu.set_hl((v >> 16) as u16);
    };

    assert_eq!(call32(&mut vm, "_ret_const_u32", &|_| {}, &[]), 0x1234_5678);

    // 9-digit score math + clamping (HLDE + 4-byte stacked bonus)
    let r1 = call32(
        &mut vm,
        "_score_accum",
        &|c| set_hlde(c, 123_456_789),
        &[StackArg::U32(70_000)],
    );
    assert_eq!(r1, 123_526_789);
    let r_clamp = call32(
        &mut vm,
        "_score_accum",
        &|c| set_hlde(c, 999_999_000),
        &[StackArg::U32(5_000)],
    );
    assert_eq!(r_clamp, 999_999_999);

    // Mixed arg ABIs: (u8, u32), (u8, u8, u32), (u32, u8)
    assert_eq!(
        call32(
            &mut vm,
            "_arg_u8_u32",
            &|c| c.a = 42,
            &[StackArg::U32(0x1234_5670)],
        ),
        0x1234_5670 + 42
    );
    assert_eq!(
        call32(
            &mut vm,
            "_arg_u8_u8_u32",
            &|c| {
                c.a = 10;
                c.e = 25;
            },
            &[StackArg::U32(0x0001_FFF0)],
        ),
        0x0001_FFF0 + 35
    );
    assert_eq!(
        call32(
            &mut vm,
            "_arg_u32_u8",
            &|c| set_hlde(c, 0x0001_0005),
            &[StackArg::U8(10)],
        ),
        0x0000_FFFB
    );

    // Bitwise & shifts
    let x = 0xABCDEF12u32;
    let sh = 5u8;
    let exp_bit = ((x & 0x0FF0_F00F) | 0x0001_0000) ^ ((((x << 4) >> sh) >> 16) << 16);
    assert_eq!(
        call32(
            &mut vm,
            "_bit_and_shift_u32",
            &|c| set_hlde(c, x),
            &[StackArg::U8(sh)],
        ),
        exp_bit
    );

    // 32-bit unsigned mul / div / rem
    let ma = 123_456u32;
    let mb = 789u32;
    assert_eq!(
        call32(
            &mut vm,
            "_mul_div_rem_u32",
            &|c| set_hlde(c, ma),
            &[StackArg::U32(mb)],
        ),
        (ma.wrapping_mul(1000) / mb).wrapping_add(ma % mb)
    );

    // Signed i32: negative comparisons, div, rem, arith shr, neg
    let sa: i32 = -123_457;
    let sb: i32 = -17;
    let exp_s1 = ((sa / sb).wrapping_add(sa % sb) >> 2).wrapping_sub(1) as u32;
    assert_eq!(
        call32(
            &mut vm,
            "_signed_i32_ops",
            &|c| set_hlde(c, sa as u32),
            &[StackArg::U32(sb as u32)],
        ),
        exp_s1
    );
    let sa2: i32 = 1000;
    let sb2: i32 = -200;
    let exp_s2 = ((-sa2) >> 2).wrapping_sub(1) as u32;
    assert_eq!(
        call32(
            &mut vm,
            "_signed_i32_ops",
            &|c| set_hlde(c, sa2 as u32),
            &[StackArg::U32(sb2 as u32)],
        ),
        exp_s2
    );

    // 4-byte struct fields via IX + sign/zero-extending casts
    let exp_box = 1_000_000u32.wrapping_add(500).wrapping_add((-7i8 as i32) as u32);
    assert_eq!(
        call32(
            &mut vm,
            "_run_box_test",
            &|c| set_hlde(c, 1_000_000),
            &[StackArg::U8(-7i8 as u8), StackArg::U16(500)],
        ),
        exp_box
    );
}

#[test]
fn z80_e2e_mem_write_fixed_addr_and_extern_statics() {
    let ext_asm = r#"
        .module ext_data
        .optsdcc -mz80 sdcccall(1)
        .area   _DATA
_EXT_U16::
        .ds     2
_EXT_STRUCT::
        .ds     4
        .area   _INITIALIZED
        .area   _INITIALIZER
        .area   _CODE
"#;

    let src = r#"
        struct CapsuleState {
            x: u8,
            y: u8,
            score: u16,
        }

        struct ExtPair {
            lo: u8,
            hi: u8,
            w: u16,
        }

        static mut G_STATE: CapsuleState @ 0xEC00;

        extern "asm" {
            static mut _EXT_U16: u16;
            static mut _EXT_STRUCT: ExtPair;
        }

        pub fn run_io_and_shared(tag: u8, tvram_word: u16) -> u16 {
            G_STATE.x = tag;
            G_STATE.y = mem_read_u8(0xEC10) +% 3u8;
            G_STATE.score = mem_read_u16(0xEC12) +% tvram_word;

            _EXT_U16 = _EXT_U16 +% 0x0102u16;
            _EXT_STRUCT.lo = G_STATE.x;
            _EXT_STRUCT.hi = G_STATE.y;
            _EXT_STRUCT.w = _EXT_U16;

            mem_write_u8(0xF3C8, G_STATE.y);
            mem_write_u16(0xF3CA, G_STATE.score);

            G_STATE.score
        }
    "#;

    let prog = aun_front::compile_str(0, src).expect("aun_front compile should succeed");
    let layout = aun_z80::uchiguri::allocate_uchiguri(&prog, 0xC000).expect("layout succeeds");
    assert_eq!(
        layout.overlay_base, layout.ram_base,
        "fixed-address and extern statics must consume 0B of __aun_ram globals"
    );
    let (sdas, _ram_bytes, _max_stack) = compile_z80_sdasz80(
        "io_shared",
        &prog,
        CompileOptions {
            ram_base: 0xC000,
            ..CompileOptions::release()
        },
    )
    .expect("compile_z80_sdasz80 should succeed");

    // Emitted instructions access 0xEC00 / _EXT_* directly without touching __aun_ram
    assert!(
        !sdas.contains("(__aun_ram"),
        "expected no instruction references to __aun_ram:\n{sdas}"
    );
    assert!(sdas.contains(".globl  _EXT_U16"));
    assert!(sdas.contains(".globl  _EXT_STRUCT"));

    let tmp = TempDirGuard::new("issue2_sdas");
    let ext_s = tmp.path.join("ext_data.s");
    let ext_rel = tmp.path.join("ext_data.rel");
    fs::write(&ext_s, ext_asm).unwrap();
    assemble_sdasz80(&ext_s, &ext_rel);

    let aun_s = tmp.path.join("io_shared.s");
    let aun_rel = tmp.path.join("io_shared.rel");
    fs::write(&aun_s, &sdas).unwrap();
    assemble_sdasz80(&aun_s, &aun_rel);

    let img = link_sdldz80(
        &tmp.path.join("issue2_img"),
        &[ext_rel.as_path(), aun_rel.as_path()],
    );

    let mut vm = DriverVm::from_image(img);
    let ext_u16_addr = vm.sym("_EXT_U16");
    let ext_struct_addr = vm.sym("_EXT_STRUCT");
    let fn_addr = vm.sym("_run_io_and_shared");

    // Seed input memory & extern global before calling
    vm.cpu.write8(0xEC10, 0x20);
    vm.cpu.write16(0xEC12, 0x1000);
    vm.cpu.write16(ext_u16_addr, 0x0304);

    // Call `run_io_and_shared(0x7A, 0x0055)` under sdcccall(1): A = u8, DE = u16
    vm.cpu.sp = 0xF000;
    vm.cpu.write8(0xFFFF, 0x76);
    vm.cpu.push16(0xFFFF);
    vm.cpu.a = 0x7A;
    vm.cpu.set_de(0x0055);
    vm.cpu.pc = fn_addr;
    vm.cpu.halted = false;
    while !vm.cpu.halted {
        assert!(vm.cpu.steps < 1_000_000);
        vm.cpu.step(&mut vm.hal);
    }
    assert_eq!(vm.cpu.sp, 0xF000);
    // 16-bit sdcccall(1) return in DE
    assert_eq!(vm.cpu.de(), 0x1055);

    // Verify fixed-address struct @ 0xEC00
    assert_eq!(vm.cpu.read8(0xEC00), 0x7A);
    assert_eq!(vm.cpu.read8(0xEC01), 0x23);
    assert_eq!(vm.cpu.read16(0xEC02), 0x1055);

    // Verify mem_write_u8 / mem_write_u16 targets (TVRAM 0xF3C8 / 0xF3CA)
    assert_eq!(vm.cpu.read8(0xF3C8), 0x23);
    assert_eq!(vm.cpu.read16(0xF3CA), 0x1055);

    // Verify extern "asm" statics (_EXT_U16 and _EXT_STRUCT)
    assert_eq!(vm.cpu.read16(ext_u16_addr), 0x0406);
    assert_eq!(vm.cpu.read8(ext_struct_addr), 0x7A);
    assert_eq!(vm.cpu.read8(ext_struct_addr + 1), 0x23);
    assert_eq!(vm.cpu.read16(ext_struct_addr + 2), 0x0406);
}

#[test]
fn z80_e2e_include_and_extern_rom_const() {
    let ext_asm = r#"
        .module ext_rom_data
        .optsdcc -mz80 sdcccall(1)
        .area   _DATA
        .area   _INITIALIZED
        .area   _INITIALIZER
        .area   _CODE
_EXT_LUT::
        .db     0x10, 0x20, 0x30, 0x40
"#;

    let tmp = TempDirGuard::new("issue3_include");
    let data_dir = tmp.path.join("gen");
    fs::create_dir_all(&data_dir).unwrap();
    fs::write(
        data_dir.join("table_data.aun"),
        "const LOCAL_BIAS: u8 = 5u8;\nconst STAGE_LUT: [u8; 4] = [1u8, 3u8, 7u8, 9u8];\n",
    )
    .unwrap();
    let main_aun = tmp.path.join("main.aun");
    fs::write(
        &main_aun,
        r#"
            include!("gen/table_data.aun");

            extern "asm" {
                const _EXT_LUT: [u8; 4];
            }

            pub fn sample_tables(idx: u8) -> u8 {
                STAGE_LUT[idx] +% _EXT_LUT[idx] +% LOCAL_BIAS
            }
        "#,
    )
    .unwrap();

    let (_src, prog) = aun_front::compile_file(&main_aun).expect("compile_file should succeed");
    let (sdas, _ram_bytes, _max_stack) = compile_z80_sdasz80(
        "inc_test",
        &prog,
        CompileOptions {
            ram_base: 0xC000,
            ..CompileOptions::release()
        },
    )
    .expect("compile_z80_sdasz80 should succeed");

    assert!(sdas.contains(".globl  _EXT_LUT"));
    assert!(sdas.contains("STAGE_LUT:"));
    assert!(!sdas.contains("_EXT_LUT:"));

    let ext_s = tmp.path.join("ext_rom.s");
    let ext_rel = tmp.path.join("ext_rom.rel");
    fs::write(&ext_s, ext_asm).unwrap();
    assemble_sdasz80(&ext_s, &ext_rel);

    let aun_s = tmp.path.join("inc_test.s");
    let aun_rel = tmp.path.join("inc_test.rel");
    fs::write(&aun_s, &sdas).unwrap();
    assemble_sdasz80(&aun_s, &aun_rel);

    let img = link_sdldz80(
        &tmp.path.join("issue3_img"),
        &[ext_rel.as_path(), aun_rel.as_path()],
    );

    let mut vm = DriverVm::from_image(img);
    let fn_addr = vm.sym("_sample_tables");
    for (idx, expected) in [(0u8, 1 + 0x10 + 5), (1, 3 + 0x20 + 5), (2, 7 + 0x30 + 5), (3, 9 + 0x40 + 5)] {
        vm.cpu.sp = 0xF000;
        vm.cpu.write8(0xFFFF, 0x76);
        vm.cpu.push16(0xFFFF);
        vm.cpu.a = idx;
        vm.cpu.pc = fn_addr;
        vm.cpu.halted = false;
        while !vm.cpu.halted {
            assert!(vm.cpu.steps < 1_000_000);
            vm.cpu.step(&mut vm.hal);
        }
        assert_eq!(vm.cpu.sp, 0xF000);
        assert_eq!(vm.cpu.a, expected, "mismatch for idx={idx}");
    }
}

#[test]
fn z80_e2e_div_rem_u8_i8_u16_i16() {
    let src = r#"
        pub fn div_u16(a: u16, b: u16) -> u16 { a / b }
        pub fn rem_u16(a: u16, b: u16) -> u16 { a % b }
        pub fn div_i16(a: i16, b: i16) -> i16 { a / b }
        pub fn rem_i16(a: i16, b: i16) -> i16 { a % b }
        pub fn div_u8(a: u8, b: u8) -> u8 { a / b }
        pub fn rem_u8(a: u8, b: u8) -> u8 { a % b }
        pub fn div_i8(a: i8, b: i8) -> i8 { a / b }
        pub fn rem_i8(a: i8, b: i8) -> i8 { a % b }

        pub fn loop_live_bc(n: u8, x: u16, d: u16) -> u16 {
            let mut acc: u16 = 0u16;
            let mut i: u8 = 0u8;
            while i < n {
                let q: u16 = (x +% (i as u16)) / d;
                let r: u16 = (x +% (i as u16)) % (d as u8 as u16 +% 1u16);
                let q8: u8 = (i +% 10u8) / 3u8;
                let r8: u8 = (i +% 10u8) % 3u8;
                acc = acc +% q +% r +% (q8 as u16) +% (r8 as u16);
                i = i +% 1u8;
            }
            acc
        }
    "#;

    let prog = aun_front::compile_str(0, src).expect("compile should succeed");
    let (sdas, _ram, _stack) = compile_z80_sdasz80(
        "div_rem_e2e",
        &prog,
        CompileOptions {
            ram_base: 0xC000,
            ..CompileOptions::release()
        },
    )
    .expect("compile_z80_sdasz80 should succeed");

    let stub_asm = r#"
        .module issue7_stub
        .optsdcc -mz80 sdcccall(1)
        .area   _DATA
        .area   _INITIALIZED
        .area   _INITIALIZER
        .area   _CODE
"#;
    let tmp = TempDirGuard::new("issue7_div_rem");
    let stub_s = tmp.path.join("stub.s");
    let stub_rel = tmp.path.join("stub.rel");
    fs::write(&stub_s, stub_asm).unwrap();
    assemble_sdasz80(&stub_s, &stub_rel);

    let aun_s = tmp.path.join("div_rem_e2e.s");
    let aun_rel = tmp.path.join("div_rem_e2e.rel");
    fs::write(&aun_s, &sdas).unwrap();
    assemble_sdasz80(&aun_s, &aun_rel);
    let img = link_sdldz80(
        &tmp.path.join("issue7_img"),
        &[stub_rel.as_path(), aun_rel.as_path()],
    );
    let mut vm = DriverVm::from_image(img);

    let call16 = |vm: &mut DriverVm, sym: &str, a: u16, b: u16| -> u16 {
        let addr = vm.sym(sym);
        vm.cpu.sp = 0xF000;
        vm.cpu.write8(0xFFFF, 0x76);
        vm.cpu.push16(0xFFFF);
        vm.cpu.set_hl(a);
        vm.cpu.set_de(b);
        vm.cpu.pc = addr;
        vm.cpu.halted = false;
        while !vm.cpu.halted {
            assert!(vm.cpu.steps < 1_000_000, "infinite loop in {sym}");
            vm.cpu.step(&mut vm.hal);
        }
        assert_eq!(vm.cpu.sp, 0xF000);
        vm.cpu.de()
    };

    let call8 = |vm: &mut DriverVm, sym: &str, a: u8, b: u8| -> u8 {
        let addr = vm.sym(sym);
        vm.cpu.sp = 0xF000;
        vm.cpu.write8(0xFFFF, 0x76);
        vm.cpu.push16(0xFFFF);
        vm.cpu.a = a;
        vm.cpu.e = b;
        vm.cpu.pc = addr;
        vm.cpu.halted = false;
        while !vm.cpu.halted {
            assert!(vm.cpu.steps < 1_000_000, "infinite loop in {sym}");
            vm.cpu.step(&mut vm.hal);
        }
        assert_eq!(vm.cpu.sp, 0xF000);
        vm.cpu.a
    };

    for (a, b) in [
        (0u16, 1u16),
        (1000, 10),
        (65535, 1),
        (65535, 32768),
        (65535, 65535),
        (12345, 97),
        (10, 100),
    ] {
        assert_eq!(call16(&mut vm, "_div_u16", a, b), a / b, "u16 {a}/{b}");
        assert_eq!(call16(&mut vm, "_rem_u16", a, b), a % b, "u16 {a}%{b}");
    }

    for (a, b) in [
        (1000i16, 7i16),
        (-1000, 7),
        (1000, -7),
        (-1000, -7),
        (-32768, 2),
        (-32768, -2),
        (-32768, 3),
        (-32768, -32768),
        (100, -32768),
    ] {
        assert_eq!(
            call16(&mut vm, "_div_i16", a as u16, b as u16) as i16,
            a / b,
            "i16 {a}/{b}"
        );
        assert_eq!(
            call16(&mut vm, "_rem_i16", a as u16, b as u16) as i16,
            a % b,
            "i16 {a}%{b}"
        );
    }

    for (a, b) in [(0u8, 1u8), (255, 1), (255, 128), (255, 255), (200, 13), (7, 20)] {
        assert_eq!(call8(&mut vm, "_div_u8", a, b), a / b, "u8 {a}/{b}");
        assert_eq!(call8(&mut vm, "_rem_u8", a, b), a % b, "u8 {a}%{b}");
    }

    for (a, b) in [
        (100i8, 7i8),
        (-100, 7),
        (100, -7),
        (-100, -7),
        (-128, 2),
        (-128, -2),
        (-128, 3),
        (-128, -128),
    ] {
        assert_eq!(
            call8(&mut vm, "_div_i8", a as u8, b as u8) as i8,
            a / b,
            "i8 {a}/{b}"
        );
        assert_eq!(
            call8(&mut vm, "_rem_i8", a as u8, b as u8) as i8,
            a % b,
            "i8 {a}%{b}"
        );
    }

    // Check loop with live u8/u16 variables across div/rem calls (BC survival check)
    let addr = vm.sym("_loop_live_bc");
    let n: u8 = 5;
    let x: u16 = 1234;
    let d: u16 = 17;
    vm.cpu.sp = 0xF000;
    vm.cpu.write8(0xFFFF, 0x76);
    vm.cpu.a = n;
    vm.cpu.set_de(x);
    vm.cpu.push16(d);
    vm.cpu.push16(0xFFFF);
    vm.cpu.pc = addr;
    vm.cpu.halted = false;
    while !vm.cpu.halted {
        assert!(vm.cpu.steps < 1_000_000);
        vm.cpu.step(&mut vm.hal);
    }
    assert_eq!(vm.cpu.sp, 0xF000);
    let mut expected: u16 = 0;
    for i in 0..n {
        let xi = x.wrapping_add(i as u16);
        let q = xi / d;
        let r = xi % ((d as u8 as u16).wrapping_add(1));
        let q8 = i.wrapping_add(10) / 3;
        let r8 = i.wrapping_add(10) % 3;
        expected = expected
            .wrapping_add(q)
            .wrapping_add(r)
            .wrapping_add(q8 as u16)
            .wrapping_add(r8 as u16);
    }
    assert_eq!(vm.cpu.de(), expected, "loop_live_bc mismatch");
}

#[test]
fn z80_e2e_rom_strings_byte_strings_and_array_ref_slices() {
    let src = r#"
        const HEADER: [u8; 4] = b"AUN!";

        fn fill_from_str(dst: &mut [u8; 6], tag: &const [u8]) {
            let n: u16 = tag.len();
            let mut i: u16 = 0u16;
            while i < n && i < dst.len() {
                dst[i] = tag[i];
                i = i +% 1u16;
            }
            dst[4] = HEADER[0];
            dst[5] = b"\x21"[0];
        }

        fn sum_bytes(s: &[u8]) -> u16 {
            let mut sum: u16 = 0u16;
            let mut i: u16 = 0u16;
            while i < s.len() {
                sum = sum +% (s[i] as u16);
                i = i +% 1u16;
            }
            sum
        }

        pub fn check_strings(flag: u8) -> u16 {
            let mut buf: [u8; 6] = [0u8; 6];
            if flag != 0u8 {
                fill_from_str(&mut buf, "ABCD");
            } else {
                fill_from_str(&mut buf, b"WXYZ");
            }
            sum_bytes(&buf) +% ("OK".len() *% 100u16) +% ("OK"[1] as u16)
        }
    "#;

    let prog = aun_front::compile_str(0, src).expect("compile should succeed");
    let (sdas, _ram, _stack) = compile_z80_sdasz80(
        "str_slice_e2e",
        &prog,
        CompileOptions {
            ram_base: 0xC000,
            ..CompileOptions::release()
        },
    )
    .expect("compile_z80_sdasz80 should succeed");

    let stub_asm = r#"
        .module issue9_stub
        .optsdcc -mz80 sdcccall(1)
        .area   _DATA
        .area   _INITIALIZED
        .area   _INITIALIZER
        .area   _CODE
"#;
    let tmp = TempDirGuard::new("issue9_strings");
    let stub_s = tmp.path.join("stub.s");
    let stub_rel = tmp.path.join("stub.rel");
    fs::write(&stub_s, stub_asm).unwrap();
    assemble_sdasz80(&stub_s, &stub_rel);

    let aun_s = tmp.path.join("str_slice_e2e.s");
    let aun_rel = tmp.path.join("str_slice_e2e.rel");
    fs::write(&aun_s, &sdas).unwrap();
    assemble_sdasz80(&aun_s, &aun_rel);
    let img = link_sdldz80(
        &tmp.path.join("issue9_img"),
        &[stub_rel.as_path(), aun_rel.as_path()],
    );
    let mut vm = DriverVm::from_image(img);
    let fn_addr = vm.sym("_check_strings");

    for flag in [1u8, 0u8] {
        vm.cpu.sp = 0xF000;
        vm.cpu.write8(0xFFFF, 0x76);
        vm.cpu.push16(0xFFFF);
        vm.cpu.a = flag;
        vm.cpu.pc = fn_addr;
        vm.cpu.halted = false;
        while !vm.cpu.halted {
            assert!(vm.cpu.steps < 1_000_000);
            vm.cpu.step(&mut vm.hal);
        }
        assert_eq!(vm.cpu.sp, 0xF000);
        let tag = if flag != 0 { b"ABCD" } else { b"WXYZ" };
        let expected: u16 = tag.iter().map(|&b| b as u16).sum::<u16>()
            + (b'A' as u16)
            + 0x21
            + 2 * 100
            + (b'K' as u16);
        assert_eq!(vm.cpu.de(), expected, "flag={flag}");
    }
}

#[test]
fn z80_e2e_addr_of_fn_fixed_indirect_call_sdcccall_stack_and_volatile() {
    let ext_c = r#"
#include <stdint.h>
uint16_t ext_stack_calc(uint8_t a, uint8_t b, uint8_t c_stack, uint16_t d_stack) {
    return (uint16_t)a + ((uint16_t)b << 1) + ((uint16_t)c_stack * 3u) + d_stack;
}
"#;
    let ext_asm = r#"
        .module issue10_ext
        .optsdcc -mz80 sdcccall(1)
        .area   _DATA
_VDP_TICK::
        .ds     1
        .area   _CODE
        .globl  _ext_cb
_ext_cb::
        add     a, #10
        ld      (0xF201), a
        ret
"#;
    let src = r#"
        struct Pair {
            a: u16,
            b: u16,
        }

        volatile static mut ISR_ACC: u8 @ 0xF208;

        extern "asm" {
            volatile static mut _VDP_TICK: u8;
            fn _ext_cb(x: u8 in a);
            fn _bios_poke(v: u8 in a) @ 0x4000;
            fn _ext_stack_calc(a: u8 in a, b: u8 in l, c: u8 in stack, d: u16 in stack) -> u16 in de;
        }
        extern "asm" {
            const _ext_cb: u8;
        }

        fn bump_pair(p: &mut Pair) {
            p.a = p.a +% 5u16;
            p.b = p.b +% (ISR_ACC as u16);
        }

        fn helper_u16(v: u16) {
            mem_write_u16_volatile(0xF202u16, v +% (_VDP_TICK as u16));
        }

        pub fn check_issue10(flag: u8) -> u16 {
            _bios_poke(0x42u8);
            let poke_addr: u16 = addr_of(_bios_poke);
            call_u16(poke_addr, 0x77u8);

            let cb_addr: u16 = addr_of(_ext_cb);
            call_u16(cb_addr, 5u8);

            let h_addr: u16 = addr_of(helper_u16);
            call_u16(h_addr, 0x1200u16);

            let mut p = Pair { a: 10u16, b: 20u16 };
            let bp_addr: u16 = addr_of(bump_pair);
            call_u16(bp_addr, &mut p);

            let mut acc: u16 = 0u16;
            let mut i: u8 = 0u8;
            while i < 3u8 {
                acc = acc +% _ext_stack_calc(1u8, i, flag, 0x0100u16);
                i = i +% 1u8;
            }
            let r1 = mem_read_u8_volatile(0xF210u16);
            let r2 = mem_read_u8_volatile(0xF210u16);
            acc +% p.a +% p.b +% (r1 as u16) +% (r2 as u16)
        }
"#;
    let prog = aun_front::compile_str(0, src).expect("Issue #10 front compile");
    let (sdas, _ram_bytes, _max_stack) = compile_z80_sdasz80(
        "issue10_e2e",
        &prog,
        CompileOptions {
            ram_base: 0xC000,
            ..CompileOptions::release()
        },
    )
    .expect("compile_z80_sdasz80 should succeed");

    // 1. Fixed-addr extern fn does NOT emit `.globl`, duplicate `_ext_cb` `.globl` deduplicates to 1.
    assert!(
        !sdas.contains(".globl  _bios_poke") && sdas.contains("0x4000"),
        "expected fixed address call 0x4000 without `.globl _bios_poke`:\n{sdas}"
    );
    assert_eq!(
        sdas.lines()
            .filter(|l| l.trim() == ".globl  _ext_cb")
            .count(),
        1,
        "expected deduplicated `.globl _ext_cb`:\n{sdas}"
    );
    // 2. Both consecutive volatile reads from 0xF210 are preserved in final assembly!
    assert_eq!(
        sdas.matches("ld\ta, (0xF210)").count() + sdas.matches("ld      a, (0xF210)").count() + sdas.matches("ld a, (0xF210)").count(),
        2,
        "both consecutive volatile reads must survive optimization:\n{sdas}"
    );

    let tmp = TempDirGuard::new("aun_e2e_issue10");
    let c_path = tmp.path.join("ext.c");
    let c_rel = tmp.path.join("ext.rel");
    fs::write(&c_path, ext_c).unwrap();
    let sdcc_out = Command::new("sdcc")
        .args(["-mz80", "--sdcccall", "1", "-c"])
        .arg(&c_path)
        .arg("-o")
        .arg(&c_rel)
        .output()
        .expect("run sdcc");
    assert!(
        sdcc_out.status.success(),
        "sdcc failed: {}",
        String::from_utf8_lossy(&sdcc_out.stderr)
    );

    let ext_s = tmp.path.join("ext_asm.s");
    let ext_rel = tmp.path.join("ext_asm.rel");
    fs::write(&ext_s, ext_asm).unwrap();
    assemble_sdasz80(&ext_s, &ext_rel);

    let aun_s = tmp.path.join("issue10_e2e.s");
    let aun_rel = tmp.path.join("issue10_e2e.rel");
    fs::write(&aun_s, &sdas).unwrap();
    assemble_sdasz80(&aun_s, &aun_rel);

    let img = link_sdldz80(
        &tmp.path.join("issue10_img"),
        &[c_rel.as_path(), ext_rel.as_path(), aun_rel.as_path()],
    );
    let mut vm = DriverVm::from_image(img);
    let fn_addr = vm.sym("_check_issue10");
    let vdp_addr = vm.sym("_VDP_TICK");

    // Place fixed-address routine at 0x4000: `ld (0xF200), a; ret`
    vm.cpu.write8(0x4000, 0x32);
    vm.cpu.write8(0x4001, 0x00);
    vm.cpu.write8(0x4002, 0xF2);
    vm.cpu.write8(0x4003, 0xC9);
    vm.cpu.write8(vdp_addr, 0x07);
    vm.cpu.write8(0xF208, 3);
    vm.cpu.write8(0xF210, 0x11);

    vm.cpu.sp = 0xF000;
    vm.cpu.write8(0xFFFF, 0x76);
    vm.cpu.push16(0xFFFF);
    vm.cpu.a = 4; // flag = 4
    vm.cpu.pc = fn_addr;
    vm.cpu.halted = false;
    while !vm.cpu.halted {
        assert!(vm.cpu.steps < 1_000_000, "Z80 step timeout");
        vm.cpu.step(&mut vm.hal);
    }
    assert_eq!(vm.cpu.sp, 0xF000);
    assert_eq!(vm.cpu.read8(0xF200), 0x77);
    assert_eq!(vm.cpu.read8(0xF201), 15);
    assert_eq!(vm.cpu.read16(0xF202), 0x1207);
    // acc = sum_{i=0..3} (1 + 2*i + 3*4 + 0x0100) = 269 + 271 + 273 = 813
    // p.a = 10 + 5 = 15, p.b = 20 + 3 = 23, r1 + r2 = 0x11 + 0x11 = 34
    assert_eq!(vm.cpu.de(), 813 + 15 + 23 + 34);
}

#[test]
fn z80_e2e_direct_aggregate_memset_shadowing_and_mul16() {
    let ext_asm = r#"
        .module issue11_ext
        .optsdcc -mz80 sdcccall(1)
        .area   _DATA
        .area   _INITIALIZED
        .area   _INITIALIZER
        .area   _CODE
        .globl  _sum_pair_ptrs
_sum_pair_ptrs::
        ld      a, (hl)
        inc     hl
        ld      b, (hl)
        ex      de, hl
        add     a, (hl)
        ld      e, a
        ld      d, #0
        ld      l, b
        ld      h, #0
        add     hl, de
        ex      de, hl
        ret
"#;
    let src = r#"
        const LUT_A: [u8; 4] = [3u8, 7u8, 11u8, 13u8];
        const LUT_B: [u8; 4] = [5u8, 9u8, 15u8, 19u8];

        struct Pair11 {
            x: u16,
            y: u16,
        }

        extern "asm" {
            fn _sum_pair_ptrs(a: u16 in hl, b: u16 in de) -> u16 in de;
        }

        pub fn check_issue11(row: u16, idx: u16) -> u16 {
            let mut line: [u8; 44] = [b' '; 44];
            let words: [u16; 8] = [0x0102u16; 8];
            let s = Pair11 { x: 10u16, y: 20u16 };
            let s = Pair11 { x: s.y +% 1u16, y: s.x +% 2u16 };
            line[idx] = 10u8;
            let psum = _sum_pair_ptrs(addr_of(LUT_A), addr_of(LUT_B));
            (row *% 120u16)
                +% (line[0] as u16)
                +% (line[43] as u16)
                +% (line[idx] as u16)
                +% words[7]
                +% s.x
                +% s.y
                +% psum
        }
"#;
    let prog = aun_front::compile_str(0, src).expect("Issue #11 front compile");
    let (sdas, ram_bytes, _max_stack) = compile_z80_sdasz80(
        "issue11_e2e",
        &prog,
        CompileOptions {
            ram_base: 0xC000,
            ..CompileOptions::release()
        },
    )
    .expect("compile_z80_sdasz80 should succeed");

    // 44B line + 16B words + 8B shadowed Pair11 structs = 68B + tiny scratch (< 100B)
    assert!(
        ram_bytes < 100,
        "expected no double-allocated [u8; 44] buffer (got {ram_bytes}B):\n{sdas}"
    );
    assert!(
        !sdas.contains("0x0078"),
        "expected folded row *% 120u16 without 0x0078 literal:\n{sdas}"
    );

    let tmp = TempDirGuard::new("aun_e2e_issue11");
    let ext_s = tmp.path.join("ext11.s");
    let ext_rel = tmp.path.join("ext11.rel");
    fs::write(&ext_s, ext_asm).unwrap();
    assemble_sdasz80(&ext_s, &ext_rel);

    let aun_s = tmp.path.join("issue11_e2e.s");
    let aun_rel = tmp.path.join("issue11_e2e.rel");
    fs::write(&aun_s, &sdas).unwrap();
    assemble_sdasz80(&aun_s, &aun_rel);

    let img = link_sdldz80(
        &tmp.path.join("issue11_img"),
        &[ext_rel.as_path(), aun_rel.as_path()],
    );
    let mut vm = DriverVm::from_image(img);
    let fn_addr = vm.sym("_check_issue11");

    vm.cpu.sp = 0xF000;
    vm.cpu.write8(0xFFFF, 0x76);
    vm.cpu.push16(0xFFFF);
    vm.cpu.set_hl(3);
    vm.cpu.set_de(5);
    vm.cpu.pc = fn_addr;
    vm.cpu.halted = false;
    while !vm.cpu.halted {
        assert!(vm.cpu.steps < 1_000_000);
        vm.cpu.step(&mut vm.hal);
    }
    assert_eq!(vm.cpu.sp, 0xF000);
    // 3 * 120 = 360; line[0]=32, line[43]=32, line[5]=10 -> 74; words[7]=0x0102=258;
    // s.x=21, s.y=12 -> 33; psum = 3 + 7 + 5 = 15. Total = 360 + 74 + 258 + 33 + 15 = 740
    assert_eq!(vm.cpu.de(), 740);
}

#[test]
fn z80_e2e_multi_file_mod_use_and_whole_program_lto() {
    let ext_asm = r#"
        .module mod_lto_ext
        .optsdcc -mz80 sdcccall(1)
        .area   _DATA
        .area   _INITIALIZED
        .area   _INITIALIZER
        .area   _CODE
        .globl  _ext_mul2
_ext_mul2::
        add     a, a
        ret
"#;
    let tmp = TempDirGuard::new("aun_e2e_mod_lto");
    fs::write(
        tmp.path.join("utils.aun"),
        r#"
pub struct Pair {
    a: u8,
    b: u8,
}

pub fn pair_sum(p: &Pair) -> u16 {
    (p.a as u16) +% (p.b as u16)
}
"#,
    )
    .unwrap();
    fs::write(
        tmp.path.join("audio.aun"),
        r#"
static mut BIAS: u8;

extern "asm" {
    fn _ext_mul2(v: u8 in a) -> u8 in a clobbers(A, F);
}

fn helper(v: u8) -> u16 {
    (_ext_mul2(v) +% BIAS) as u16
}

pub fn audio_init() {
    BIAS = 5u8;
}

pub fn audio_add(v: u8) -> u16 {
    helper(v) +% 10u16
}
"#,
    )
    .unwrap();
    fs::write(
        tmp.path.join("main.aun"),
        r#"
mod utils;
use utils::{Pair, pair_sum};
use audio::{audio_add, audio_init};

extern "asm" {
    fn _ext_mul2(v: u8 in a) -> u8 in a clobbers(A, F);
}

fn helper(v: u8) -> u16 {
    (_ext_mul2(v) as u16) +% 100u16
}

pub fn main_run(x: u8) -> u16 {
    audio_init();
    let p = Pair { a: x, b: x +% 2u8 };
    pair_sum(&p) +% helper(x) +% audio_add(3u8)
}
"#,
    )
    .unwrap();

    let main_p = tmp.path.join("main.aun");
    let audio_p = tmp.path.join("audio.aun");
    let (_, prog) = aun_front::compile_files(&[main_p.as_path(), audio_p.as_path()])
        .expect("multi-file mod/use LTO front compile");
    let (sdas, _ram_bytes, _max_stack) = compile_z80_sdasz80(
        "mod_lto_e2e",
        &prog,
        CompileOptions {
            ram_base: 0xC000,
            ..CompileOptions::release()
        },
    )
    .expect("compile_z80_sdasz80 should succeed");

    // Both top-level units' `pub fn`s are exported as C ABI wrappers, while submodule `pair_sum` is internal.
    assert!(sdas.contains(".globl  _main_run"));
    assert!(sdas.contains(".globl  _audio_add"));
    assert!(!sdas.contains(".globl  _pair_sum"));

    let ext_s = tmp.path.join("ext_mod.s");
    let ext_rel = tmp.path.join("ext_mod.rel");
    fs::write(&ext_s, ext_asm).unwrap();
    assemble_sdasz80(&ext_s, &ext_rel);

    let aun_s = tmp.path.join("mod_lto_e2e.s");
    let aun_rel = tmp.path.join("mod_lto_e2e.rel");
    fs::write(&aun_s, &sdas).unwrap();
    assemble_sdasz80(&aun_s, &aun_rel);

    let img = link_sdldz80(
        &tmp.path.join("mod_lto_img"),
        &[ext_rel.as_path(), aun_rel.as_path()],
    );
    let mut vm = DriverVm::from_image(img);

    let main_addr = vm.sym("_main_run");
    vm.cpu.sp = 0xF000;
    vm.cpu.write8(0xFFFF, 0x76);
    vm.cpu.push16(0xFFFF);
    vm.cpu.a = 7;
    vm.cpu.pc = main_addr;
    vm.cpu.halted = false;
    while !vm.cpu.halted {
        assert!(vm.cpu.steps < 1_000_000);
        vm.cpu.step(&mut vm.hal);
    }
    assert_eq!(vm.cpu.sp, 0xF000);
    // pair_sum(7, 9) = 16; main::helper(7) = 14 + 100 = 114; audio_add(3) = (6 + 5) + 10 = 21. Total = 151.
    assert_eq!(vm.cpu.de(), 151);

    // Verify `_audio_add` is also directly callable from C/ASM.
    let audio_addr = vm.sym("_audio_add");
    vm.cpu.sp = 0xF000;
    vm.cpu.write8(0xFFFF, 0x76);
    vm.cpu.push16(0xFFFF);
    vm.cpu.a = 4;
    vm.cpu.pc = audio_addr;
    vm.cpu.halted = false;
    while !vm.cpu.halted {
        assert!(vm.cpu.steps < 1_000_000);
        vm.cpu.step(&mut vm.hal);
    }
    assert_eq!(vm.cpu.sp, 0xF000);
    // (4*2 + 5) + 10 = 23
    assert_eq!(vm.cpu.de(), 23);
}

#[test]
fn z80_e2e_bin_links_external_sdasz80_files() {
    let ext_s1 = r#"
        .module ext_data_and_math
        .optsdcc -mz80 sdcccall(1)
        .area   _DATA
_EXT_STATE::
        .ds     1
        .area   _CODE
        .globl  _ext_step
_ext_step::
        ld      hl, #_EXT_STATE
        add     a, (hl)
1$:
        inc     a
        dec     l
        jr      z, 1$
        ld      (_EXT_STATE), a
        ret
"#;
    let ext_s2 = r#"
        .module ext_helper2
        .area   _CODE
        .globl  _ext_double_de
_ext_double_de::
        ex      de, hl
        add     hl, hl
        ex      de, hl
        ret
"#;
    let aun_src = r#"
        extern "asm" {
            static mut _EXT_STATE: u8;
            fn _ext_step(delta: u8 in a) -> u8 in a clobbers(A, H, L, F);
            fn _ext_double_de(val: u16 in de) -> u16 in de clobbers(D, E, H, L, F);
        }

        pub fn main_bin_link(x: u8) -> u16 {
            _EXT_STATE = 10u8;
            let a = _ext_step(x);
            let b = _ext_step(2u8);
            _ext_double_de(((a as u16) +% (b as u16)) +% (_EXT_STATE as u16))
        }
"#;
    let prog = aun_front::compile_str(0, aun_src).expect("front compile");
    let out = aun_z80::compile_z80_with_options(
        &prog,
        &[ext_s1, ext_s2],
        CompileOptions {
            code_origin: 0x0100,
            ram_base: 0xC000,
            ..CompileOptions::release()
        },
    )
    .expect("compile_z80_with_options should link external .s files");

    let ext_state_addr = *out.symbols.get("_EXT_STATE").expect("_EXT_STATE symbol");
    assert!(ext_state_addr >= out.ram_base);
    assert!(out.ram_end >= ext_state_addr + 1);
    assert!(out.symbols.contains_key("_ext_step"));
    assert!(out.symbols.contains_key("_ext_double_de"));
    assert!(out.fn_tstates.contains_key("_ext_step"));

    let mut cpu = Z80::new();
    let mut hal = MockHal::default();
    cpu.ram[out.code_origin as usize..out.code_origin as usize + out.resident_bin.len()]
        .copy_from_slice(&out.resident_bin);
    let fn_addr = *out.symbols.get("main_bin_link").expect("main_bin_link sym");

    cpu.sp = 0xF000;
    cpu.write8(0xFFFF, 0x76);
    cpu.push16(0xFFFF);
    cpu.a = 5;
    cpu.pc = fn_addr;
    cpu.halted = false;
    while !cpu.halted {
        assert!(cpu.steps < 100_000, "Z80 step timeout");
        cpu.step(&mut hal);
    }
    assert_eq!(cpu.sp, 0xF000);
    // _EXT_STATE=10 -> _ext_step(5)=10+5+1=16 -> _EXT_STATE=16 -> _ext_step(2)=16+2+1=19 -> _EXT_STATE=19
    // (16 + 19 + 19) * 2 = 108
    assert_eq!(cpu.read8(ext_state_addr), 19);
    assert_eq!(cpu.hl(), 108);
}

#[test]
fn z80_e2e_sdcccall1_u16_u8_and_extern_callback_uchiguri() {
    let ext_c = r#"
#include <stdint.h>
typedef struct {
    uint16_t score;
    uint8_t fever;
} GameState;

extern uint16_t add_fever(GameState *gs, uint8_t amt);

uint16_t c_call_add_fever(GameState *gs, uint8_t amt) {
    return add_fever(gs, amt) + add_fever(gs, 3);
}
"#;
    let ext_asm = r#"
        .module issue14_ext
        .optsdcc -mz80 sdcccall(1)
        .area   _DATA
_G_ASM_GS::
        .ds     3
        .area   _CODE
        .globl  _add_fever
        .globl  _asm_call_u16_u8
        .globl  _asm_step_with_cb
_asm_call_u16_u8::
        ld      hl, #_G_ASM_GS
        ld      (hl), #0x0A
        inc     hl
        ld      (hl), #0x00
        inc     hl
        ld      (hl), #0x01
        ld      a, #2
        push    af
        inc     sp
        ld      hl, #_G_ASM_GS
        call    _add_fever
        ret
_asm_step_with_cb::
        ; Call back into Aun `pub fn add_fever(gs, amt)` while caller `main_with_cb` has live locals!
        push    af
        inc     sp
        call    _add_fever
        ret
"#;
    let aun_src = r#"
        struct GameState {
            score: u16,
            fever: u8,
        }

        extern "asm" {
            static mut _G_ASM_GS: GameState;
            fn _asm_step_with_cb(gs: &mut GameState in hl, amt: u8 in a) -> u16 in de
                clobbers(AF, BC, DE, HL);
        }

        pub fn add_fever(gs: &mut GameState, amt: u8) -> u16 {
            let mut bonus: u16 = (amt as u16) *% 10u16;
            let mut next_f: u8 = gs.fever +% amt;
            let mut extra: u16 = (next_f as u16) +% bonus;
            gs.fever = next_f;
            gs.score = gs.score +% extra;
            gs.score
        }

        pub fn main_with_cb(amt: u8) -> u16 {
            let mut gs = GameState { score: 100u16, fever: 4u8 };
            let mut saved_a: u16 = (amt as u16) +% 11u16;
            let mut saved_b: u16 = saved_a *% 3u16;
            let mut saved_c: u16 = saved_b +% 7u16;
            let ret_cb = _asm_step_with_cb(&mut gs, amt);
            // If Uchiguri overlapped `main_with_cb` and `add_fever`, saved_a/b/c or gs would corrupt!
            ret_cb +% saved_a +% saved_b +% saved_c +% (gs.fever as u16)
        }
"#;
    let prog = aun_front::compile_str(0, aun_src).expect("Issue #14 front compile");
    let (sdas, _ram_bytes, _max_stack) = compile_z80_sdasz80(
        "issue14_e2e",
        &prog,
        CompileOptions {
            ram_base: 0xC000,
            ..CompileOptions::release()
        },
    )
    .expect("compile_z80_sdasz80 should succeed");

    let tmp = TempDirGuard::new("aun_e2e_issue14");
    let c_path = tmp.path.join("ext14.c");
    let c_rel = tmp.path.join("ext14.rel");
    fs::write(&c_path, ext_c).unwrap();
    let sdcc_out = Command::new("sdcc")
        .args(["-mz80", "--sdcccall", "1", "-c"])
        .arg(&c_path)
        .arg("-o")
        .arg(&c_rel)
        .output()
        .expect("run sdcc");
    assert!(
        sdcc_out.status.success(),
        "sdcc failed: {}",
        String::from_utf8_lossy(&sdcc_out.stderr)
    );

    let ext_s = tmp.path.join("ext14_asm.s");
    let ext_rel = tmp.path.join("ext14_asm.rel");
    fs::write(&ext_s, ext_asm).unwrap();
    assemble_sdasz80(&ext_s, &ext_rel);

    let aun_s = tmp.path.join("issue14_e2e.s");
    let aun_rel = tmp.path.join("issue14_e2e.rel");
    fs::write(&aun_s, &sdas).unwrap();
    assemble_sdasz80(&aun_s, &aun_rel);

    let img = link_sdldz80(
        &tmp.path.join("issue14_img"),
        &[c_rel.as_path(), ext_rel.as_path(), aun_rel.as_path()],
    );
    let mut vm = DriverVm::from_image(img);

    // 1. Hand-written Z80 asm calling `_add_fever` with `--sdcccall 1` `(HL, stack 1B)`
    let asm_caller = vm.sym("_asm_call_u16_u8");
    vm.cpu.sp = 0xF000;
    vm.cpu.write8(0xFFFF, 0x76);
    vm.cpu.push16(0xFFFF);
    vm.cpu.pc = asm_caller;
    vm.cpu.halted = false;
    while !vm.cpu.halted {
        assert!(vm.cpu.steps < 100_000);
        vm.cpu.step(&mut vm.hal);
    }
    assert_eq!(vm.cpu.sp, 0xF000, "callee must clean 1-byte stack arg");
    // score=10, fever=1, amt=2 -> next_f=3, bonus=20, extra=23 -> score=33
    assert_eq!(vm.cpu.de(), 33);

    // 2. SDCC C calling `_add_fever(&gs, 5)` then `_add_fever(&gs, 3)`
    let gs_addr = vm.sym("_G_ASM_GS");
    vm.cpu.write16(gs_addr, 100);
    vm.cpu.write8(gs_addr + 2, 1);
    let c_caller = vm.sym("_c_call_add_fever");
    vm.cpu.sp = 0xF000;
    vm.cpu.sp = vm.cpu.sp.wrapping_sub(1);
    vm.cpu.write8(vm.cpu.sp, 5);
    vm.cpu.write8(0xFFFF, 0x76);
    vm.cpu.push16(0xFFFF);
    vm.cpu.set_hl(gs_addr);
    vm.cpu.pc = c_caller;
    vm.cpu.halted = false;
    while !vm.cpu.halted {
        assert!(vm.cpu.steps < 100_000);
        vm.cpu.step(&mut vm.hal);
    }
    // call1(5): next_f=6, bonus=50, extra=56 -> score=156
    // call2(3): next_f=9, bonus=30, extra=39 -> score=195
    // total = 156 + 195 = 351
    assert_eq!(vm.cpu.de(), 351);
    assert_eq!(vm.cpu.read16(gs_addr), 195);
    assert_eq!(vm.cpu.read8(gs_addr + 2), 9);

    // 3. `main_with_cb(2)` calls extern `_asm_step_with_cb` which calls back into `add_fever`
    let main_cb = vm.sym("_main_with_cb");
    vm.cpu.sp = 0xF000;
    vm.cpu.write8(0xFFFF, 0x76);
    vm.cpu.push16(0xFFFF);
    vm.cpu.a = 2;
    vm.cpu.pc = main_cb;
    vm.cpu.halted = false;
    while !vm.cpu.halted {
        assert!(vm.cpu.steps < 100_000);
        vm.cpu.step(&mut vm.hal);
    }
    assert_eq!(vm.cpu.sp, 0xF000);
    // amt=2, gs={100,4}: cb -> next_f=6, bonus=20, extra=26 -> score=126
    // saved_a=13, saved_b=39, saved_c=46, gs.fever=6 -> 126+13+39+46+6 = 230
    assert_eq!(vm.cpu.de(), 230);
}

#[test]
fn z80_e2e_if_else_struct_copy() {
    let ext_asm = r#"
        .module issue15_ext
        .optsdcc -mz80 sdcccall(1)
        .area   _DATA
        .area   _INITIALIZED
        .area   _INITIALIZER
        .area   _CODE
"#;
    let aun_src = r#"
        struct Pair {
            a: i16,
            b: i16,
        }

        struct SoftSpring2D {
            x: i16,
            y: i16,
            vx: i16,
            vy: i16,
        }

        struct State {
            left: Pair,
            right: Pair,
            spring_l: SoftSpring2D,
            spring_r: SoftSpring2D,
        }

        static mut G: State @ 0xEC00;

        pub fn init_issue15() {
            G.left.a = 10i16;
            G.left.b = 20i16;
            G.right.a = 30i16;
            G.right.b = 40i16;
            G.spring_l.x = 100i16;
            G.spring_l.y = 200i16;
            G.spring_l.vx = 3i16;
            G.spring_l.vy = 4i16;
            G.spring_r.x = 500i16;
            G.spring_r.y = 600i16;
            G.spring_r.vx = 7i16;
            G.spring_r.vy = 8i16;
        }

        pub fn pick_pair(flag: bool) -> i16 {
            let p: Pair = if flag { G.right } else { G.left };
            let sp: SoftSpring2D = if flag { G.spring_r } else { G.spring_l };
            (p.a +% p.b) +% (sp.x +% sp.y +% sp.vx +% sp.vy)
        }
"#;
    let prog = aun_front::compile_str(0, aun_src).expect("Issue #15 front compile");
    let (sdas, _ram_bytes, _max_stack) = compile_z80_sdasz80(
        "issue15_e2e",
        &prog,
        CompileOptions {
            ram_base: 0xC000,
            ..CompileOptions::release()
        },
    )
    .expect("compile_z80_sdasz80 should succeed");

    let tmp = TempDirGuard::new("aun_e2e_issue15");
    let ext_s = tmp.path.join("ext15.s");
    let ext_rel = tmp.path.join("ext15.rel");
    fs::write(&ext_s, ext_asm).unwrap();
    assemble_sdasz80(&ext_s, &ext_rel);

    let aun_s = tmp.path.join("issue15_e2e.s");
    let aun_rel = tmp.path.join("issue15_e2e.rel");
    fs::write(&aun_s, &sdas).unwrap();
    assemble_sdasz80(&aun_s, &aun_rel);

    let img = link_sdldz80(
        &tmp.path.join("issue15_img"),
        &[ext_rel.as_path(), aun_rel.as_path()],
    );
    let mut vm = DriverVm::from_image(img);

    let init_addr = vm.sym("_init_issue15");
    vm.cpu.sp = 0xF000;
    vm.cpu.write8(0xFFFF, 0x76);
    vm.cpu.push16(0xFFFF);
    vm.cpu.pc = init_addr;
    vm.cpu.halted = false;
    while !vm.cpu.halted {
        assert!(vm.cpu.steps < 100_000);
        vm.cpu.step(&mut vm.hal);
    }

    let pick_addr = vm.sym("_pick_pair");
    vm.cpu.sp = 0xF000;
    vm.cpu.write8(0xFFFF, 0x76);
    vm.cpu.push16(0xFFFF);
    vm.cpu.a = 0; // false
    vm.cpu.pc = pick_addr;
    vm.cpu.halted = false;
    while !vm.cpu.halted {
        assert!(vm.cpu.steps < 100_000);
        vm.cpu.step(&mut vm.hal);
    }
    assert_eq!(vm.cpu.sp, 0xF000);
    // (10 + 20) + (100 + 200 + 3 + 4) = 337
    assert_eq!(vm.cpu.de(), 337);

    vm.cpu.sp = 0xF000;
    vm.cpu.write8(0xFFFF, 0x76);
    vm.cpu.push16(0xFFFF);
    vm.cpu.a = 1; // true
    vm.cpu.pc = pick_addr;
    vm.cpu.halted = false;
    while !vm.cpu.halted {
        assert!(vm.cpu.steps < 100_000);
        vm.cpu.step(&mut vm.hal);
    }
    assert_eq!(vm.cpu.sp, 0xF000);
    // (30 + 40) + (500 + 600 + 7 + 8) = 1185
    assert_eq!(vm.cpu.de(), 1185);
}

#[test]
fn z80_e2e_bin_format_runs_outliner_and_tracks_extra_stack() {
    let ext_s = r#"
        .module ext_out_sink
        .optsdcc -mz80 sdcccall(1)
        .area   _DATA
_SINK_BUF::
        .ds     16
        .area   _CODE
        .globl  _emit_row
_emit_row::
        out     (0xF0), a
        ret
"#;
    let aun_src = r#"
        extern "asm" {
            fn _emit_row(port: u8 in a, val: u8 in l) clobbers(F);
        }

        const TABLE: [u8; 8] = [10u8, 20u8, 30u8, 40u8, 50u8, 60u8, 70u8, 80u8];

        pub fn emit_all(bias: u8) -> u16 {
            let mut acc: u16 = 0u16;
            _emit_row(0x10u8, TABLE[0] +% bias);
            _emit_row(0x11u8, TABLE[1] +% bias);
            _emit_row(0x12u8, TABLE[2] +% bias);
            acc = acc +% (TABLE[0] as u16) +% (TABLE[1] as u16) +% (TABLE[2] as u16);

            _emit_row(0x10u8, TABLE[0] +% bias);
            _emit_row(0x11u8, TABLE[1] +% bias);
            _emit_row(0x12u8, TABLE[2] +% bias);
            acc = acc +% (TABLE[3] as u16) +% (TABLE[4] as u16) +% (TABLE[5] as u16);

            _emit_row(0x10u8, TABLE[0] +% bias);
            _emit_row(0x11u8, TABLE[1] +% bias);
            _emit_row(0x12u8, TABLE[2] +% bias);
            acc +% (TABLE[6] as u16) +% (TABLE[7] as u16) +% (bias as u16)
        }
"#;
    let prog = aun_front::compile_str(0, aun_src).expect("front compile");
    let out = aun_z80::compile_z80_with_options(
        &prog,
        &[ext_s],
        CompileOptions {
            code_origin: 0x0100,
            ram_base: 0xC000,
            ..CompileOptions::release()
        },
    )
    .expect("compile_z80_with_options should succeed");

    assert!(
        out.symbols.keys().any(|k| k.starts_with("__aun_out_")),
        "expected outline_lir to emit __aun_out_* helpers in --format bin:\n{}",
        out.asm_listing
    );
    assert!(
        out.max_stack_bytes >= 2,
        "expected extra stack bytes from outlined helpers, got {}",
        out.max_stack_bytes
    );

    let mut cpu = Z80::new();
    let mut hal = MockHal::default();
    cpu.ram[out.code_origin as usize..out.code_origin as usize + out.resident_bin.len()]
        .copy_from_slice(&out.resident_bin);
    let fn_addr = *out.symbols.get("emit_all").expect("emit_all sym");

    cpu.sp = 0xF000;
    cpu.write8(0xFFFF, 0x76);
    cpu.push16(0xFFFF);
    cpu.a = 3;
    cpu.pc = fn_addr;
    cpu.halted = false;
    while !cpu.halted {
        assert!(cpu.steps < 100_000, "Z80 step timeout");
        cpu.step(&mut hal);
    }
    assert_eq!(cpu.sp, 0xF000);
    // sum(10..=80) = 360, + bias(3) = 363
    assert_eq!(cpu.hl(), 363);
    assert_eq!(hal.opn_writes.len(), 9);
    assert_eq!(hal.opn_writes[0], (0x10, 13));
    assert_eq!(hal.opn_writes[8], (0x12, 33));
}

#[test]
fn z80_e2e_wave2_divrem_mul16x8_osr_and_countdown() {
    let aun_src = r#"
        extern "asm" {
            fn _emit_row(reg: u8 in A, val: u8 in C);
        }

        struct Item { a: u8, b: u16 }
        static mut ITEMS: [Item; 8];

        fn helper_mul(x: u16, y: u8) -> u32 {
            (x as u32) *% (y as u32)
        }

        pub fn run_wave2(mult: u8, base_val: u16) -> u32 {
            let mut i: u8 = 0u8;
            while i < 8u8 {
                ITEMS[i as u16].a = i +% 1u8;
                ITEMS[i as u16].b = base_val +% ((i as u16) *% 10u16);
                i = i +% 1u8;
            }
            let mut sum_b: u16 = 0u16;
            let mut j: u8 = 0u8;
            while j < 8u8 {
                sum_b = sum_b +% ITEMS[j as u16].b;
                j = j +% 1u8;
            }
            let prod: u32 = helper_mul(sum_b, mult);
            let mut s: u32 = prod;
            let mut k: u8 = 0u8;
            while k < 4u8 {
                let d: u8 = (s % 10u32) as u8;
                _emit_row(k, d);
                s = s / 10u32;
                k = k +% 1u8;
            }
            prod
        }
    "#;
    let ext_s = r#"
        .globl _emit_row
_emit_row:
        ld l, c
        out (0xF0), a
        ret
    "#;

    let prog = aun_front::compile_str(0, aun_src).expect("front compile");
    let out = aun_z80::compile_z80_with_options(
        &prog,
        &[ext_s],
        CompileOptions {
            code_origin: 0x0100,
            ram_base: 0xC000,
            ..CompileOptions::release()
        },
    )
    .expect("compile_z80_with_options should succeed");

    // 1. Single-caller helper_mul inlined & eliminated; u16*u8 avoids __aun_mul32; Div/Rem fused.
    assert!(!out.symbols.contains_key("helper_mul"));
    assert!(
        !out.symbols.contains_key("__aun_mul32"),
        "u16 * u8 -> u32 must use inline 16x8->32 loop instead of __aun_mul32"
    );
    assert!(
        out.asm_listing.contains("__aun_udivrem32"),
        "adjacent % 10 / 10 must fuse to __aun_udivrem32:\n{}",
        out.asm_listing
    );
    assert!(
        out.asm_listing.contains("djnz ") || out.asm_listing.contains("dec b"),
        "expected djnz or dec b countdown in:\n{}",
        out.asm_listing
    );

    // 2. Verify runtime execution on Z80 emulator:
    // base_val = 100, mult = 250 -> items.b = [100,110,120,130,140,150,160,170], sum_b = 1080
    // prod = 1080 * 250 = 270000 (0x00041EB0 -> HL=0x0004, DE=0x1EB0)
    // low 4 decimal digits of 270000 emitted via _emit_row: 0, 0, 0, 0 (next would be 7, 2)
    // Try base_val = 123, mult = 19 -> items.b sum = 8*123 + 280 = 1264, prod = 1264 * 19 = 24016
    // digits from LSB: 6, 1, 0, 4
    let mut cpu = Z80::new();
    let mut hal = MockHal::default();
    cpu.ram[out.code_origin as usize..out.code_origin as usize + out.resident_bin.len()]
        .copy_from_slice(&out.resident_bin);
    let fn_addr = *out.symbols.get("run_wave2").expect("run_wave2 sym");

    cpu.sp = 0xF000;
    cpu.write8(0xFFFF, 0x76);
    cpu.push16(0xFFFF);
    cpu.a = 19;
    cpu.set_de(123);
    cpu.pc = fn_addr;
    cpu.halted = false;
    while !cpu.halted {
        assert!(cpu.steps < 500_000, "Z80 step timeout");
        cpu.step(&mut hal);
    }
    assert_eq!(cpu.sp, 0xF000);
    let ret32 = ((cpu.hl() as u32) << 16) | (cpu.de() as u32);
    assert_eq!(ret32, 1264 * 19);
    // 1264 * 19 = 24016 -> low 4 decimal digits emitted: 6, 1, 0, 4
    assert_eq!(hal.opn_writes, vec![(0, 6), (1, 1), (2, 0), (3, 4)]);
}

#[test]
fn z80_e2e_state_transition_writes_u32_and_array_payloads() {
    let aun_src = r#"
        extern "asm" {
            fn _log_u8(ch: u8 in A, v: u8 in C);
        }

        static mut OUT_SCORE: u32;

        machine Tracker {
            state A { seed: u32 };
            state B { score: u32, tag: [u8; 3] };
            transitions {
                A -> B;
            }
            on A {
                fn advance(&mut self) {
                    goto B {
                        score: self.seed +% 100000u32,
                        tag: [0x11u8, 0x22u8, 0x33u8],
                    };
                }
                fn report(&mut self) {}
            }
            on B {
                fn advance(&mut self) {}
                fn report(&mut self) {
                    OUT_SCORE = self.score;
                    _log_u8(0u8, self.tag[0]);
                    _log_u8(1u8, self.tag[1]);
                    _log_u8(2u8, self.tag[2]);
                }
            }
        }

        pub fn run_transition(seed_init: u32) -> u32 {
            let mut tr: Tracker = Tracker::new();
            tr.goto(A { seed: seed_init });
            tr.advance();
            tr.report();
            OUT_SCORE
        }
    "#;
    let ext_s = r#"
        .globl _log_u8
_log_u8:
        ld l, c
        out (0xF0), a
        ret
    "#;

    let prog = aun_front::compile_str(0, aun_src).expect("front compile");
    let out = aun_z80::compile_z80_with_options(
        &prog,
        &[ext_s],
        CompileOptions {
            code_origin: 0x0100,
            ram_base: 0xC000,
            ..CompileOptions::release()
        },
    )
    .expect("compile_z80_with_options should succeed");

    let mut cpu = Z80::new();
    let mut hal = MockHal::default();
    cpu.ram[out.code_origin as usize..out.code_origin as usize + out.resident_bin.len()]
        .copy_from_slice(&out.resident_bin);
    let fn_addr = *out.symbols.get("run_transition").expect("run_transition sym");

    // Pass u32 `0x0001_E240` (123456) in HLDE
    cpu.sp = 0xF000;
    cpu.write8(0xFFFF, 0x76);
    cpu.push16(0xFFFF);
    cpu.set_de(0xE240);
    cpu.set_hl(0x0001);
    cpu.pc = fn_addr;
    cpu.halted = false;
    while !cpu.halted {
        assert!(cpu.steps < 50_000, "Z80 step timeout");
        cpu.step(&mut hal);
    }
    assert_eq!(cpu.sp, 0xF000);
    let ret32 = ((cpu.hl() as u32) << 16) | (cpu.de() as u32);
    assert_eq!(ret32, 123_456 + 100_000);
    assert_eq!(hal.opn_writes, vec![(0, 0x11), (1, 0x22), (2, 0x33)]);
}

#[test]
fn z80_e2e_sdasz80_nonzero_static_mut_initializers() {
    let src = r#"
        static mut S_RNG: u16 = 0x5A3Cu16;
        static mut S_TAG: [u8; 4] = [b'A', b'U', b'N', 1u8];
        static mut S_ZERO: u16;

        pub fn step_rng() -> u16 {
            let cur: u16 = S_RNG;
            S_RNG = cur +% 0x0101u16;
            cur +% (S_TAG[3] as u16) +% S_ZERO
        }

        pub fn tag_sum() -> u16 {
            (S_TAG[0] as u16) +% (S_TAG[1] as u16) +% (S_TAG[2] as u16) +% (S_TAG[3] as u16)
        }
    "#;
    let prog = aun_front::compile_str(0, src).expect("front compile");
    let (sdas, ram_bytes, _max_stack) = compile_z80_sdasz80(
        "nz_init",
        &prog,
        CompileOptions {
            ram_base: 0xC000,
            ..CompileOptions::release()
        },
    )
    .expect("compile_z80_sdasz80 failed");

    assert!(sdas.contains(".area _INITIALIZED"), "missing _INITIALIZED:\n{sdas}");
    assert!(sdas.contains(".area _INITIALIZER"), "missing _INITIALIZER:\n{sdas}");
    assert!(sdas.contains("__aun_init:"), "missing __aun_init:\n{sdas}");
    assert!(ram_bytes >= 6, "expected >= 6 initialized bytes, got {ram_bytes}");

    let tmp = TempDirGuard::new("nz_init");
    let hal_s = tmp.path.join("mock_hal.s");
    let hal_rel = tmp.path.join("mock_hal.rel");
    fs::write(&hal_s, MOCK_HAL_ASM).expect("write mock_hal.s");
    assemble_sdasz80(&hal_s, &hal_rel);

    let asm_path = tmp.path.join("nz_init.s");
    let rel_path = tmp.path.join("nz_init.rel");
    fs::write(&asm_path, &sdas).expect("write nz_init.s");
    assemble_sdasz80(&asm_path, &rel_path);
    let img = link_sdldz80(
        &tmp.path.join("nz_init_img"),
        &[hal_rel.as_path(), rel_path.as_path()],
    );

    let mut vm = DriverVm::from_image(img);
    let call_u16 = |vm: &mut DriverVm, sym: &str| -> u16 {
        let addr = vm.sym(sym);
        vm.cpu.sp = 0xF000;
        vm.cpu.write8(0xFFFF, 0x76);
        vm.cpu.push16(0xFFFF);
        vm.cpu.pc = addr;
        vm.cpu.halted = false;
        while !vm.cpu.halted {
            assert!(vm.cpu.steps < 100_000, "Z80 step timeout");
            vm.cpu.step(&mut vm.hal);
        }
        vm.cpu.de()
    };

    assert_eq!(call_u16(&mut vm, "_tag_sum"), 65 + 85 + 78 + 1);
    assert_eq!(call_u16(&mut vm, "_step_rng"), 0x5A3D);
    assert_eq!(call_u16(&mut vm, "_step_rng"), 0x5B3E);
}

#[test]
fn z80_e2e_isr_entry_and_reentrant_helper_isolation() {
    let src = r#"
        static mut G_LOG: [u8; 4];

        #[reentrant]
        fn shared_accum(seed: u8, add: u8) -> u8 {
            let mut local_tmp: u8 = seed +% 1u8;
            if add > 15u8 {
                local_tmp = local_tmp +% add;
            } else {
                local_tmp = local_tmp +% (add *% 2u8);
            }
            local_tmp
        }

        pub fn api_tick(x: u8) -> u8 {
            let mut mloc: u8 = x +% 2u8;
            let r: u8 = shared_accum(mloc, 10u8);
            G_LOG[0] = r;
            r
        }

        #[isr_entry]
        pub fn on_irq(v: u8) {
            let mut iloc: u8 = v +% 3u8;
            G_LOG[1] = shared_accum(iloc, 20u8);
        }
    "#;
    let prog = aun_front::compile_str(0, src).expect("front compile");
    let out = aun_z80::compile_z80_with_options(
        &prog,
        &[],
        CompileOptions {
            code_origin: 0x0100,
            ram_base: 0xC000,
            ..CompileOptions::release()
        },
    )
    .expect("compile_z80_with_options should succeed");

    // Both copies must exist and have distinct Uchiguri frames:
    assert!(out.symbols.contains_key("shared_accum"));
    assert!(out.symbols.contains_key("shared_accum__isr_p0"));
    let api_f = out.fn_frames.iter().find(|f| f.name == "api_tick").unwrap();
    let isr_f = out.fn_frames.iter().find(|f| f.name == "on_irq").unwrap();
    assert!(
        isr_f.locals_base >= api_f.locals_base + api_f.locals_size + api_f.scratch_size,
        "#[isr_entry] locals must not overlap pub fn locals in library mode"
    );

    let mut cpu = Z80::new();
    let mut hal = MockHal::default();
    cpu.ram[out.code_origin as usize..out.code_origin as usize + out.resident_bin.len()]
        .copy_from_slice(&out.resident_bin);

    // Call api_tick(5) -> mloc = 7 -> local_tmp = 8 + (10*2) = 28
    cpu.sp = 0xF000;
    cpu.write8(0xFFFF, 0x76);
    cpu.push16(0xFFFF);
    cpu.a = 5;
    cpu.pc = out.symbols["api_tick"];
    cpu.halted = false;
    while !cpu.halted {
        assert!(cpu.steps < 50_000, "Z80 step timeout");
        cpu.step(&mut hal);
    }
    assert_eq!(cpu.a, 28);

    // Call on_irq(9) -> iloc = 12 -> local_tmp = 13 + 20 = 33
    cpu.sp = 0xF000;
    cpu.push16(0xFFFF);
    cpu.a = 9;
    cpu.pc = out.symbols["on_irq"];
    cpu.halted = false;
    while !cpu.halted {
        assert!(cpu.steps < 50_000, "Z80 step timeout");
        cpu.step(&mut hal);
    }
    let g_log = out.globals.iter().find(|g| g.name == "G_LOG").unwrap().addr;
    assert_eq!(cpu.read8(g_log), 28);
    assert_eq!(cpu.read8(g_log + 1), 33);
}

#[test]
fn z80_e2e_const_shifts_and_widening_mul16_to_32() {
    let src = r#"
        static mut IN_U16: u16 = 0u16;
        static mut IN_I16: i16 = 0i16;
        static mut IN_U32: u32 = 0u32;
        static mut IN_I32: i32 = 0i32;
        static mut IN_A: u16 = 0u16;
        static mut IN_B: u16 = 0u16;
        static mut IN_SA: i16 = 0i16;
        static mut IN_SB: i16 = 0i16;
        static mut OUT_U16: [u16; 12] = [0u16; 12];
        static mut OUT_I16: [i16; 12] = [0i16; 12];
        static mut OUT_U32: [u32; 16] = [0u32; 16];
        static mut OUT_I32: [i32; 16] = [0i32; 16];

        pub fn run_all() {
            test_shifts_and_mul(IN_U16, IN_I16, IN_U32, IN_I32, IN_A, IN_B, IN_SA, IN_SB);
        }

        fn test_shifts_and_mul(u: u16, s: i16, u32v: u32, s32v: i32, a: u16, b: u16, sa: i16, sb: i16) {
            let min_i16: i16 = 0x8000u16 as i16;
            OUT_U16[0] = u << 5u16;
            OUT_U16[1] = u << 6u16;
            OUT_U16[2] = u << 7u16;
            OUT_U16[3] = u << 11u16;
            OUT_U16[4] = u >> 4u16;
            OUT_U16[5] = u >> 5u16;
            OUT_U16[6] = u >> 6u16;
            OUT_U16[7] = u >> 7u16;
            OUT_U16[8] = u >> 11u16;
            OUT_U16[9] = u >> 15u16;
            OUT_U16[10] = u << 0u16;
            OUT_U16[11] = u >> 15u16;

            OUT_I16[0] = s << 5i16;
            OUT_I16[1] = s << 6i16;
            OUT_I16[2] = s >> 4i16;
            OUT_I16[3] = s >> 5i16;
            OUT_I16[4] = s >> 6i16;
            OUT_I16[5] = s >> 7i16;
            OUT_I16[6] = s >> 8i16;
            OUT_I16[7] = s >> 9i16;
            OUT_I16[8] = s >> 14i16;
            OUT_I16[9] = s >> 15i16;
            OUT_I16[10] = s >> 1i16;
            OUT_I16[11] = s << 9i16;

            OUT_U32[0] = u32v << 1u32;
            OUT_U32[1] = u32v << 4u32;
            OUT_U32[2] = u32v << 8u32;
            OUT_U32[3] = u32v << 9u32;
            OUT_U32[4] = u32v << 12u32;
            OUT_U32[5] = u32v << 20u32;
            OUT_U32[6] = u32v << 24u32;
            OUT_U32[7] = u32v << 29u32;
            OUT_U32[8] = u32v >> 1u32;
            OUT_U32[9] = u32v >> 4u32;
            OUT_U32[10] = u32v >> 8u32;
            OUT_U32[11] = u32v >> 10u32;
            OUT_U32[12] = u32v >> 13u32;
            OUT_U32[13] = u32v >> 20u32;
            OUT_U32[14] = u32v >> 24u32;
            OUT_U32[15] = u32v >> 30u32;

            OUT_I32[0] = s32v >> 1i32;
            OUT_I32[1] = s32v >> 4i32;
            OUT_I32[2] = s32v >> 8i32;
            OUT_I32[3] = s32v >> 10i32;
            OUT_I32[4] = s32v >> 13i32;
            OUT_I32[5] = s32v >> 16i32;
            OUT_I32[6] = s32v >> 20i32;
            OUT_I32[7] = s32v >> 24i32;
            OUT_I32[8] = s32v >> 27i32;
            OUT_I32[9] = s32v >> 30i32;
            OUT_I32[10] = s32v >> 31i32;
            OUT_I32[11] = ((0x8000u16 as u32) *% (b as u32)) as i32;
            OUT_I32[12] = ((a as u32) *% (b as u32)) as i32;
            OUT_I32[13] = (sa as i32) *% (sb as i32);
            OUT_I32[14] = (sa as i32) *% 300i32;
            OUT_I32[15] = (min_i16 as i32) *% (min_i16 as i32);
        }
    "#;
    let prog = aun_front::compile_str(0, src).expect("front compile");
    let out = aun_z80::compile_z80_with_options(&prog, &[], CompileOptions::release())
        .expect("compile_z80_with_options should succeed");

    assert!(
        !out.symbols.contains_key("__aun_mul32"),
        "widening 16x16->32 multiply must use __aun_mulu16_32 / __aun_muls16_32 instead of __aun_mul32:\n{}",
        out.asm_listing
    );
    assert!(out.symbols.contains_key("__aun_mulu16_32"));
    assert!(out.symbols.contains_key("__aun_muls16_32"));

    let mut cpu = Z80::new();
    let mut hal = MockHal::default();
    cpu.ram[out.code_origin as usize..out.code_origin as usize + out.resident_bin.len()]
        .copy_from_slice(&out.resident_bin);

    let addr = |name: &str| out.globals.iter().find(|g| g.name == name).unwrap().addr;
    let u_val: u16 = 0xB35A;
    let s_val: i16 = -0x4CA6; // 0xB35A signed
    let u32_val: u32 = 0x9E37_79B9;
    let s32_val: i32 = -0x61C8_8647; // 0x9E37_79B9 signed
    let a_val: u16 = 0xFFFF;
    let b_val: u16 = 0xF00D;
    let sa_val: i16 = -12345;
    let sb_val: i16 = 23456;

    let write_u32 = |cpu: &mut Z80, a: u16, v: u32| {
        cpu.write16(a, (v & 0xFFFF) as u16);
        cpu.write16(a + 2, (v >> 16) as u16);
    };
    cpu.write16(addr("IN_U16"), u_val);
    cpu.write16(addr("IN_I16"), s_val as u16);
    write_u32(&mut cpu, addr("IN_U32"), u32_val);
    write_u32(&mut cpu, addr("IN_I32"), s32_val as u32);
    cpu.write16(addr("IN_A"), a_val);
    cpu.write16(addr("IN_B"), b_val);
    cpu.write16(addr("IN_SA"), sa_val as u16);
    cpu.write16(addr("IN_SB"), sb_val as u16);

    cpu.sp = 0xF000;
    cpu.write8(0xFFFF, 0x76);
    cpu.push16(0xFFFF);
    cpu.pc = out.symbols["run_all"];
    cpu.halted = false;
    while !cpu.halted {
        assert!(cpu.steps < 50_000, "Z80 step timeout");
        cpu.step(&mut hal);
    }

    let read_u32 = |cpu: &Z80, base: u16, idx: u16| -> u32 {
        let lo = cpu.read16(base + idx * 4) as u32;
        let hi = cpu.read16(base + idx * 4 + 2) as u32;
        lo | (hi << 16)
    };

    let ou16 = addr("OUT_U16");
    assert_eq!(cpu.read16(ou16 + 0), u_val << 5);
    assert_eq!(cpu.read16(ou16 + 2), u_val << 6);
    assert_eq!(cpu.read16(ou16 + 4), u_val << 7);
    assert_eq!(cpu.read16(ou16 + 6), u_val << 11);
    assert_eq!(cpu.read16(ou16 + 8), u_val >> 4);
    assert_eq!(cpu.read16(ou16 + 10), u_val >> 5);
    assert_eq!(cpu.read16(ou16 + 12), u_val >> 6);
    assert_eq!(cpu.read16(ou16 + 14), u_val >> 7);
    assert_eq!(cpu.read16(ou16 + 16), u_val >> 11);
    assert_eq!(cpu.read16(ou16 + 18), u_val >> 15);
    assert_eq!(cpu.read16(ou16 + 20), u_val);

    let oi16 = addr("OUT_I16");
    assert_eq!(cpu.read16(oi16 + 0) as i16, s_val << 5);
    assert_eq!(cpu.read16(oi16 + 2) as i16, s_val << 6);
    assert_eq!(cpu.read16(oi16 + 4) as i16, s_val >> 4);
    assert_eq!(cpu.read16(oi16 + 6) as i16, s_val >> 5);
    assert_eq!(cpu.read16(oi16 + 8) as i16, s_val >> 6);
    assert_eq!(cpu.read16(oi16 + 10) as i16, s_val >> 7);
    assert_eq!(cpu.read16(oi16 + 12) as i16, s_val >> 8);
    assert_eq!(cpu.read16(oi16 + 14) as i16, s_val >> 9);
    assert_eq!(cpu.read16(oi16 + 16) as i16, s_val >> 14);
    assert_eq!(cpu.read16(oi16 + 18) as i16, s_val >> 15);
    assert_eq!(cpu.read16(oi16 + 20) as i16, s_val >> 1);
    assert_eq!(cpu.read16(oi16 + 22) as i16, s_val << 9);

    let ou32 = addr("OUT_U32");
    assert_eq!(read_u32(&cpu, ou32, 0), u32_val << 1);
    assert_eq!(read_u32(&cpu, ou32, 1), u32_val << 4);
    assert_eq!(read_u32(&cpu, ou32, 2), u32_val << 8);
    assert_eq!(read_u32(&cpu, ou32, 3), u32_val << 9);
    assert_eq!(read_u32(&cpu, ou32, 4), u32_val << 12);
    assert_eq!(read_u32(&cpu, ou32, 5), u32_val << 20);
    assert_eq!(read_u32(&cpu, ou32, 6), u32_val << 24);
    assert_eq!(read_u32(&cpu, ou32, 7), u32_val << 29);
    assert_eq!(read_u32(&cpu, ou32, 8), u32_val >> 1);
    assert_eq!(read_u32(&cpu, ou32, 9), u32_val >> 4);
    assert_eq!(read_u32(&cpu, ou32, 10), u32_val >> 8);
    assert_eq!(read_u32(&cpu, ou32, 11), u32_val >> 10);
    assert_eq!(read_u32(&cpu, ou32, 12), u32_val >> 13);
    assert_eq!(read_u32(&cpu, ou32, 13), u32_val >> 20);
    assert_eq!(read_u32(&cpu, ou32, 14), u32_val >> 24);
    assert_eq!(read_u32(&cpu, ou32, 15), u32_val >> 30);

    let oi32 = addr("OUT_I32");
    assert_eq!(read_u32(&cpu, oi32, 0) as i32, s32_val >> 1);
    assert_eq!(read_u32(&cpu, oi32, 1) as i32, s32_val >> 4);
    assert_eq!(read_u32(&cpu, oi32, 2) as i32, s32_val >> 8);
    assert_eq!(read_u32(&cpu, oi32, 3) as i32, s32_val >> 10);
    assert_eq!(read_u32(&cpu, oi32, 4) as i32, s32_val >> 13);
    assert_eq!(read_u32(&cpu, oi32, 5) as i32, s32_val >> 16);
    assert_eq!(read_u32(&cpu, oi32, 6) as i32, s32_val >> 20);
    assert_eq!(read_u32(&cpu, oi32, 7) as i32, s32_val >> 24);
    assert_eq!(read_u32(&cpu, oi32, 8) as i32, s32_val >> 27);
    assert_eq!(read_u32(&cpu, oi32, 9) as i32, s32_val >> 30);
    assert_eq!(read_u32(&cpu, oi32, 10) as i32, s32_val >> 31);
    assert_eq!(read_u32(&cpu, oi32, 11), (0x8000u32).wrapping_mul(b_val as u32));
    assert_eq!(read_u32(&cpu, oi32, 12), (a_val as u32).wrapping_mul(b_val as u32));
    assert_eq!(
        read_u32(&cpu, oi32, 13) as i32,
        (sa_val as i32).wrapping_mul(sb_val as i32)
    );
    assert_eq!(
        read_u32(&cpu, oi32, 14) as i32,
        (sa_val as i32).wrapping_mul(300)
    );
    assert_eq!(
        read_u32(&cpu, oi32, 15) as i32,
        (-32768i32).wrapping_mul(-32768i32)
    );
}

#[test]
fn z80_e2e_const_slice_passing_and_iteration() {
    let ext_s = r#"
        .module ext_putch
        .area   _CODE
        .globl  _ext_putch
_ext_putch::
        ld      l, a
        xor     a
        out     (0xF0), a
        ret
"#;
    let src = r#"
        extern "asm" {
            #[no_callback]
            fn _ext_putch(c: u8 in a) clobbers(F, L);
        }

        fn put_str(s: &const [u8]) {
            let mut i: u16 = 0u16;
            while i < s.len() {
                _ext_putch(s[i]);
                i = i +% 1u16;
            }
        }

        pub fn main() {
            put_str("HELLO");
            _ext_putch(0x0Au8);
            put_str("AUN");
        }
    "#;
    let prog = aun_front::compile_str(0, src).expect("front compile");
    let out = aun_z80::compile_z80_with_options(&prog, &[ext_s], CompileOptions::release())
        .expect("compile_z80_with_options should succeed");

    let mut cpu = Z80::new();
    let mut hal = MockHal::default();
    cpu.ram[out.code_origin as usize..out.code_origin as usize + out.resident_bin.len()]
        .copy_from_slice(&out.resident_bin);
    cpu.sp = 0xF000;
    cpu.write8(0xFFFF, 0x76);
    cpu.push16(0xFFFF);
    cpu.pc = out.symbols["main"];
    cpu.halted = false;
    while !cpu.halted {
        assert!(cpu.steps < 50_000, "Z80 step timeout");
        cpu.step(&mut hal);
    }

    let emitted: Vec<u8> = hal.opn_writes.iter().map(|(_, v)| *v).collect();
    assert_eq!(emitted, b"HELLO\nAUN");
    assert!(
        !out.asm_listing.contains("__aun_ram + 14"),
        "slice passing/iteration should not spill redundant MakeSlice/SlicePtr scratch chains:\n{}",
        out.asm_listing
    );
}

#[test]
fn z80_e2e_aligned256_index_ldir_loops_and_index_cse() {
    let src = r#"
        struct Pair {
            a: u8,
            b: u8,
        }

        static mut VRAM: [u8; 256] @ 0xD000 = [0u8; 256];
        static mut FILL_BUF: [u8; 64] = [0u8; 64];
        static mut COPY_DST: [u8; 32] = [0u8; 32];
        static mut PAIRS: [Pair; 8] = [Pair { a: 0u8, b: 0u8 }; 8];

        pub fn run(idx: u8, branch_flag: u8) -> u8 {
            VRAM[idx] = 0x5Au8;
            let mut i: u8 = 0u8;
            while i < 64u8 {
                FILL_BUF[i] = 0x3Cu8;
                i = i +% 1u8;
            }
            let mut j: u8 = 0u8;
            while j < 32u8 {
                COPY_DST[j] = FILL_BUF[j] +% j;
                j = j +% 1u8;
            }
            let mut k: u8 = 0u8;
            while k < 32u8 {
                COPY_DST[k] = FILL_BUF[k];
                k = k +% 1u8;
            }
            PAIRS[idx].a = 0x11u8;
            if branch_flag != 0u8 {
                PAIRS[idx].b = PAIRS[idx].a +% 0x22u8;
            } else {
                PAIRS[idx].b = PAIRS[idx].a +% 0x44u8;
            }
            VRAM[idx] +% COPY_DST[31] +% PAIRS[idx].b
        }
    "#;
    let prog = aun_front::compile_str(0, src).expect("front compile");
    let out = aun_z80::compile_z80_with_options(&prog, &[], CompileOptions::release())
        .expect("compile_z80_with_options should succeed");

    // 1. 256B-aligned fixed address 0xD000 folds into `ld h, 0xD0` without `add hl, de` / `ld de, 0xD000`.
    assert!(
        out.asm_listing.contains("ld h, 0xD0"),
        "expected 256B-aligned VRAM index to emit `ld h, 0xD0`:\n{}",
        out.asm_listing
    );
    assert!(
        !out.asm_listing.contains("ld de, 0xD000"),
        "256B-aligned VRAM index should not add 0xD000 via DE:\n{}",
        out.asm_listing
    );
    // 2. Both the constant fill loop (len=64 >= 16) and copy loop (len=32) lower to `ldir`.
    assert!(
        out.asm_listing.matches("    ldir").count() >= 2,
        "expected >= 2 `ldir` instructions from fill64 + copy32 loops:\n{}",
        out.asm_listing
    );

    let mut cpu = Z80::new();
    let mut hal = MockHal::default();
    cpu.ram[out.code_origin as usize..out.code_origin as usize + out.resident_bin.len()]
        .copy_from_slice(&out.resident_bin);
    cpu.a = 3;
    cpu.e = 1;
    cpu.sp = 0xF000;
    cpu.write8(0xFFFF, 0x76);
    cpu.push16(0xFFFF);
    cpu.pc = out.symbols["run"];
    cpu.halted = false;
    while !cpu.halted {
        assert!(cpu.steps < 50_000, "Z80 step timeout");
        cpu.step(&mut hal);
    }

    assert_eq!(cpu.read8(0xD003), 0x5A);
    let fill_addr = out.globals.iter().find(|g| g.name == "FILL_BUF").unwrap().addr;
    let copy_addr = out.globals.iter().find(|g| g.name == "COPY_DST").unwrap().addr;
    assert_eq!(&cpu.ram[fill_addr as usize..fill_addr as usize + 64], &[0x3Cu8; 64]);
    assert_eq!(&cpu.ram[copy_addr as usize..copy_addr as usize + 32], &[0x3Cu8; 32]);
    assert_eq!(cpu.a, 0x5Au8.wrapping_add(0x3C).wrapping_add(0x33));
}

#[test]
fn z80_e2e_fast_udivrem32_8_mul8_and_cmp16_idioms() {
    let src = r#"
        static mut SINK: u32 = 0u32;
        fn sink(x: u32) {
            SINK = SINK +% x;
        }

        pub fn div32_u8_cases(x: u32, d: u8) -> u32 {
            sink(0u32 / 10u32);
            sink(0u32 % 10u32);
            let q = x / 10u32;
            let r = x % 10u32;
            sink(q);
            sink(r);
            let qmax = 0xFFFF_FFFFu32 / 255u32;
            let rmax = 0xFFFF_FFFFu32 % 255u32;
            let q1 = 42u32 / 1u32;
            let q200 = 0x8000_0000u32 / 200u32;
            let r200 = 0x8000_0000u32 % 200u32;
            let dv = d as u32;
            let qd = x / dv;
            let rd = x % dv;
            qmax +% rmax +% q1 +% q200 +% r200 +% qd +% rd
        }

        pub fn mul8_cases(x: u8, y: u8) -> u8 {
            let a = x *% 0u8;
            let b = x *% 1u8;
            let c = x *% 3u8;
            let d = x *% 6u8;
            let e = x *% 10u8;
            let f = x *% 128u8;
            let g = x *% 255u8;
            let h = x *% y;
            a +% b +% c +% d +% e +% f +% g +% h
        }

        pub fn cmp16_cases(v: u16) -> u8 {
            let mut acc: u8 = 0u8;
            if v < 1u16 { acc = acc +% 1u8; }
            if v >= 1u16 { acc = acc +% 2u8; }
            if v == 1u16 { acc = acc +% 4u8; }
            if v != 1u16 { acc = acc +% 8u8; }
            if v < 256u16 { acc = acc +% 16u8; }
            if v >= 256u16 { acc = acc +% 32u8; }
            if v <= 255u16 { acc = acc +% 64u8; }
            if v > 255u16 { acc = acc +% 128u8; }
            acc
        }

        pub fn test_div32(x: u32) -> u32 {
            SINK = 0u32;
            div32_u8_cases(x, 17u8) +% (x / 10u32) +% (x % 10u32) +% SINK
        }
    "#;
    let prog = aun_front::compile_str(0, src).expect("front compile");
    let out = aun_z80::compile_z80_with_options(&prog, &[], CompileOptions::release())
        .expect("compile_z80_with_options");

    assert!(
        out.asm_listing.contains("__aun_udivrem32_8"),
        "expected fast u32/u8 divider:\n{}",
        out.asm_listing
    );
    assert!(
        !out.symbols.contains_key("__aun_udiv32"),
        "full 32-bit divider should not be linked when divisor fits u8:\n{}",
        out.asm_listing
    );
    assert!(
        out.asm_listing.contains("__aun_mul8"),
        "expected shared __aun_mul8 for dynamic u8 multiply:\n{}",
        out.asm_listing
    );

    let run_fn = |name: &str, hl: u16, de: u16, a: u8| -> (u16, u16, u8) {
        let mut cpu = Z80::new();
        let mut hal = MockHal::default();
        cpu.ram[out.code_origin as usize..out.code_origin as usize + out.resident_bin.len()]
            .copy_from_slice(&out.resident_bin);
        cpu.sp = 0xF000;
        cpu.write8(0xFFFF, 0x76);
        cpu.push16(0xFFFF);
        cpu.set_hl(hl);
        cpu.set_de(de);
        cpu.a = a;
        cpu.pc = out.symbols[name];
        cpu.halted = false;
        while !cpu.halted {
            assert!(cpu.steps < 50_000, "Z80 timeout in {name}");
            cpu.step(&mut hal);
        }
        (cpu.hl(), cpu.de(), cpu.a)
    };

    for &x in &[0u32, 123_456_789u32, 0xFFFF_FFFFu32] {
        let (rhl, rde, _) = run_fn("test_div32", (x >> 16) as u16, x as u16, 0);
        let got32 = ((rhl as u32) << 16) | (rde as u32);
        let expected32 = (x / 10)
            .wrapping_add(x % 10)
            .wrapping_add(0xFFFF_FFFFu32 / 255)
            .wrapping_add(0xFFFF_FFFFu32 % 255)
            .wrapping_add(42)
            .wrapping_add(0x8000_0000u32 / 200)
            .wrapping_add(0x8000_0000u32 % 200)
            .wrapping_add(x / 17)
            .wrapping_add(x % 17)
            .wrapping_add(x / 10)
            .wrapping_add(x % 10);
        assert_eq!(got32, expected32, "u32/u8 mismatch for x={x:#x}");
    }

    for &(vx, vy) in &[(0u8, 0u8), (7, 13), (0xFF, 0xFF), (129, 5)] {
        let (_, _, got8) = run_fn("mul8_cases", 0, vy as u16, vx);
        let expected8 = vx
            .wrapping_mul(0)
            .wrapping_add(vx.wrapping_mul(1))
            .wrapping_add(vx.wrapping_mul(3))
            .wrapping_add(vx.wrapping_mul(6))
            .wrapping_add(vx.wrapping_mul(10))
            .wrapping_add(vx.wrapping_mul(128))
            .wrapping_add(vx.wrapping_mul(255))
            .wrapping_add(vx.wrapping_mul(vy));
        assert_eq!(got8, expected8, "mul8 mismatch for x={vx}, y={vy}");
    }

    for &v in &[0u16, 1, 2, 255, 256, 65535] {
        let (_, _, got_cmp) = run_fn("cmp16_cases", v, 0, 0);
        let mut expected_cmp = 0u8;
        if v < 1 { expected_cmp |= 1; }
        if v >= 1 { expected_cmp |= 2; }
        if v == 1 { expected_cmp |= 4; }
        if v != 1 { expected_cmp |= 8; }
        if v < 256 { expected_cmp |= 16; }
        if v >= 256 { expected_cmp |= 32; }
        if v <= 255 { expected_cmp |= 64; }
        if v > 255 { expected_cmp |= 128; }
        assert_eq!(got_cmp, expected_cmp, "cmp16 mismatch for v={v}");
    }
}

#[test]
fn z80_e2e_local_osr_struct_ptr_ix_promotion() {
    let src = r#"
        struct Entity {
            x: u8,
            y: u8,
            hp: u8,
            flags: u8,
        }

        static mut ENTITIES: [Entity; 8] = [Entity { x: 0u8, y: 0u8, hp: 0u8, flags: 0u8 }; 8];
        static mut EXTRA_ENTITY: Entity = Entity { x: 0u8, y: 0u8, hp: 0u8, flags: 0u8 };

        pub fn update_entities() -> u16 {
            let mut sum: u16 = 0u16;
            let mut i: u8 = 0u8;
            while i < 8u8 {
                let base: u8 = i *% 3u8;
                ENTITIES[i as u16].x = ENTITIES[i as u16].x +% base +% 1u8;
                ENTITIES[i as u16].y = ENTITIES[i as u16].y +% base +% 2u8;
                ENTITIES[i as u16].hp = ENTITIES[i as u16].hp +% 5u8;
                ENTITIES[i as u16].flags = ENTITIES[i as u16].flags ^ 0x0Fu8;
                sum = sum
                    +% (ENTITIES[i as u16].x as u16)
                    +% (ENTITIES[i as u16].y as u16)
                    +% (ENTITIES[i as u16].hp as u16)
                    +% (ENTITIES[i as u16].flags as u16);
                i = i +% 1u8;
            }
            sum
        }

        fn touch_entity_and_update(e: &mut Entity) -> u16 {
            e.x = e.x +% 10u8;
            e.y = e.y +% 20u8;
            let s: u16 = update_entities();
            s +% (e.x as u16) +% (e.y as u16) +% (e.hp as u16) +% (e.flags as u16)
        }

        pub fn test_caller_ix_preserved() -> u16 {
            EXTRA_ENTITY.x = 3u8;
            EXTRA_ENTITY.y = 4u8;
            EXTRA_ENTITY.hp = 5u8;
            EXTRA_ENTITY.flags = 6u8;
            let a: u16 = touch_entity_and_update(&mut EXTRA_ENTITY);
            let b: u16 = touch_entity_and_update(&mut EXTRA_ENTITY);
            a +% b
        }
    "#;
    let prog = aun_front::compile_str(0, src).expect("front compile");
    let out = aun_z80::compile_z80_with_options(&prog, &[], CompileOptions::release())
        .expect("compile_z80_with_options should succeed");

    let update_start = out
        .asm_listing
        .find("\nupdate_entities:")
        .expect("update_entities label in asm");
    let update_rest = &out.asm_listing[update_start + 1..];
    let update_end = update_rest
        .find("\n\n")
        .or_else(|| update_rest.find("\ntouch_entity_and_update:"))
        .or_else(|| update_rest.find("\ntest_caller_ix_preserved:"))
        .unwrap_or(update_rest.len());
    let update_asm = &update_rest[..update_end];

    assert!(
        update_asm.contains("(ix + "),
        "expected update_entities to access Entity fields via (ix + d):\n{update_asm}"
    );
    assert!(
        update_asm.contains("add ix, de") || update_asm.contains("inc ix"),
        "expected update_entities to advance promoted IX pointer in-place:\n{update_asm}"
    );

    let run_fn = |cpu: &mut Z80, name: &str| -> u16 {
        let mut hal = MockHal::default();
        cpu.sp = 0xF000;
        cpu.write8(0xFFFF, 0x76);
        cpu.push16(0xFFFF);
        cpu.pc = out.symbols[name];
        cpu.halted = false;
        while !cpu.halted {
            assert!(cpu.steps < 50_000, "Z80 timeout in {name}");
            cpu.step(&mut hal);
        }
        cpu.hl()
    };

    let mut cpu = Z80::new();
    cpu.ram[out.code_origin as usize..out.code_origin as usize + out.resident_bin.len()]
        .copy_from_slice(&out.resident_bin);
    assert_eq!(run_fn(&mut cpu, "update_entities"), 352);
    assert_eq!(run_fn(&mut cpu, "update_entities"), 464);

    let mut cpu2 = Z80::new();
    cpu2.ram[out.code_origin as usize..out.code_origin as usize + out.resident_bin.len()]
        .copy_from_slice(&out.resident_bin);
    assert_eq!(run_fn(&mut cpu2, "test_caller_ix_preserved"), 942);
}

#[test]
fn z80_e2e_inc_ix_stride1_and_2_bin_format() {
    let src = r#"
        struct Pair {
            lo: u8,
            hi: u8,
        }

        static mut PAIRS: [Pair; 6] = [Pair { lo: 1u8, hi: 2u8 }; 6];
        static mut BYTES: [u8; 6] = [10u8, 20u8, 30u8, 40u8, 50u8, 60u8];

        pub fn step_pairs() -> u16 {
            let mut sum: u16 = 0u16;
            let mut i: u8 = 0u8;
            while i < 6u8 {
                PAIRS[i as u16].lo = PAIRS[i as u16].lo +% i;
                PAIRS[i as u16].hi = PAIRS[i as u16].hi +% (i *% 2u8);
                sum = sum +% (PAIRS[i as u16].lo as u16) +% (PAIRS[i as u16].hi as u16);
                i = i +% 1u8;
            }
            sum
        }

        pub fn step_bytes() -> u16 {
            let mut acc: u16 = 0u16;
            let mut i: u8 = 0u8;
            while i < 6u8 {
                let v: u8 = BYTES[i as u16];
                BYTES[i as u16] = v +% 3u8;
                let doubled: u8 = BYTES[i as u16] +% v;
                BYTES[i as u16] = doubled;
                acc = acc +% (BYTES[i as u16] as u16);
                i = i +% 1u8;
            }
            acc
        }
        pub fn main() -> u16 {
            let mut k: u8 = 0u8;
            while k < 6u8 {
                PAIRS[k as u16].lo = 1u8;
                PAIRS[k as u16].hi = 2u8;
                BYTES[k as u16] = (k +% 1u8) *% 10u8;
                k = k +% 1u8;
            }
            let a: u16 = step_pairs();
            let b: u16 = step_bytes();
            a +% b
        }
    "#;
    let prog = aun_front::compile_str(0, src).expect("front compile");
    let out = aun_z80::compile_z80_with_options(&prog, &[], CompileOptions::release())
        .expect("compile_z80_with_options (--format bin) should assemble inc ix cleanly");

    assert!(
        out.asm_listing.contains("    inc ix"),
        "expected promoted IX pointer step of 1 or 2 to emit `inc ix`:\n{}",
        out.asm_listing
    );

    let mut cpu = Z80::new();
    let mut hal = MockHal::default();
    cpu.ram[out.code_origin as usize..out.code_origin as usize + out.resident_bin.len()]
        .copy_from_slice(&out.resident_bin);
    cpu.sp = 0xF000;
    cpu.write8(0xFFFF, 0x76);
    cpu.push16(0xFFFF);
    cpu.pc = out.symbols["main"];
    cpu.halted = false;
    while !cpu.halted {
        assert!(cpu.steps < 50_000, "Z80 timeout in main");
        cpu.step(&mut hal);
    }
    // Pairs: sum += (1+i) + (2+2i) = 63; Bytes: sum = 2*210 + 18 = 438; total = 501
    assert_eq!(cpu.hl(), 63 + 438);
}

#[test]
fn z80_e2e_hsm_auto_uchiguri_overlays_and_loadop_swap() {
    let src = r#"
        target pc88sr : z80 {
            space MainRam   @ 0x0100..0x9FFF;
            stack MainStack @ 0x9E00..0x9FFF;

            bank_group HighWindow @ 0xC000..0xFFFF {
                bank HighMainRam @ 0xC000..0xEFFF;
                default HighMainRam;
            }

            on_swap(ops: &const [LoadOp]) => mock_swap(ops);
        }

        static mut SWAP_TOTAL_OPS: u8;
        static mut SWAP_CALLS: u8;
        static mut OP_LOG: [u8; 8];
        static mut TRACE: [u16; 8];
        static mut TRACE_LEN: u8;

        extern "asm" {
            // Copies `len` bytes from mock ROM/disk slot (0xA000 + blob_id * 0x0400) to `dst`.
            fn hal_copy_blob(blob_id: u8 in A, dst: u16 in DE) clobbers(AF, BC, DE, HL);
            fn hal_set_copy_len(len: u16 in HL) clobbers(HL);
        }

        #[overlay(App::InGame)]
        extern "asm" {
            fn ext_ingame_bonus() -> u8 in A clobbers(AF);
        }

        fn log_step(v: u16) {
            let idx: u8 = TRACE_LEN;
            TRACE[idx as u16] = v;
            TRACE_LEN = idx +% 1u8;
        }

        fn mock_swap(ops: &const [LoadOp]) {
            let c: u8 = SWAP_CALLS;
            SWAP_CALLS = c +% 1u8;
            let n: u16 = ops.len();
            let mut i: u16 = 0u16;
            while i < n {
                let op: LoadOp = ops[i];
                if op.space != SPACE_MainRam {
                    return;
                }
                let tot: u8 = SWAP_TOTAL_OPS;
                OP_LOG[tot as u16] = op.blob_id;
                SWAP_TOTAL_OPS = tot +% 1u8;
                hal_set_copy_len(op.size);
                hal_copy_blob(op.blob_id, op.dst);
                i = i +% 1u16;
            }
        }

        // Private sub-machine only called inside Stage1 -> auto-partitioned into Stage1 blob!
        machine BossAi {
            state Idle { ticks: u8 };
            state Attack { dmg: u8 };
            transitions {
                Idle -> Attack;
                Attack -> Idle;
            }
            on Idle {
                fn tick(&mut self) -> u8 {
                    let next: u8 = self.ticks +% 5u8;
                    goto Attack { dmg: next };
                    next
                }
            }
            on Attack {
                fn tick(&mut self) -> u8 {
                    self.dmg
                }
            }
        }

        machine App {
            state Title { cursor: u8 };

            #[swap_boundary]
            state InGame {
                score: u16,
                static mut INGAME_BUF: [u8; 4] in HighMainRam;

                #[swap_boundary]
                state Stage1 {
                    hp: u8,
                    static mut S1_BUF: [u8; 4] in HighMainRam;
                };
                #[swap_boundary]
                state Stage2 {
                    wave: u8,
                    static mut S2_BUF: [u8; 4] in HighMainRam;
                };

                initial Stage1;
            }

            transitions {
                Title -> InGame;
                InGame::Stage1 -> InGame::Stage2;
                InGame -> Title;
            }

            on InGame {
                fn on_enter(&mut self) {
                    INGAME_BUF[0] = 0x10u8;
                    log_step(100u16 +% self.score);
                }
                fn on_exit(&mut self) {
                    log_step(900u16);
                }
                fn add_shared(&mut self, pts: u16) {
                    self.score = self.score +% pts;
                }
            }

            on InGame::Stage1 {
                fn on_enter(&mut self) {
                    S1_BUF[0] = self.hp;
                    log_step(200u16 +% (self.hp as u16));
                }
                fn on_exit(&mut self) {
                    log_step(299u16);
                }
                fn update(&mut self) {
                    let mut boss: BossAi = BossAi::new();
                    let b: u8 = boss.tick();
                    let bonus: u8 = ext_ingame_bonus();
                    self.add_shared((b as u16) +% (bonus as u16));
                    log_step(210u16 +% (INGAME_BUF[0] as u16) +% (S1_BUF[0] as u16));
                    goto InGame::Stage2 { wave: 7u8 };
                }
            }

            on InGame::Stage2 {
                fn on_enter(&mut self) {
                    S2_BUF[0] = self.wave;
                    log_step(300u16 +% (self.wave as u16));
                }
                fn update(&mut self) {
                    let bonus: u8 = ext_ingame_bonus();
                    self.add_shared(bonus as u16);
                    log_step(310u16 +% self.score +% (INGAME_BUF[0] as u16) +% (S2_BUF[0] as u16));
                }
            }
        }

        pub fn main() -> u16 {
            let mut app: App = App::new();
            // 1. Title -> InGame::Stage1 (loads [InGame, Stage1])
            app.goto(InGame::Stage1 { score: 20u16, hp: 3u8 });
            // 2. Stage1::update calls BossAi + ext_ingame_bonus, then tail-jumps to Stage2!
            app.update();
            // 3. Now in Stage2: run Stage2::update, then return to Title!
            app.update();
            app.goto(Title { cursor: 1u8 });
            TRACE_LEN as u16
        }
    "#;

    let hal_res_s = r#"
__mock_copy_len:
    .dw 0
hal_set_copy_len:
    ld (__mock_copy_len), hl
    ret
hal_copy_blob:
    ; A = blob_id, DE = dst; src = 0xA000 + blob_id * 0x0400
    add a, a
    add a, a
    add a, 0xA0
    ld h, a
    ld l, 0
    ld bc, (__mock_copy_len)
    ld a, b
    or c
    ret z
    ldir
    ret
"#;
    let ext_ingame_s = r#"
ext_ingame_bonus:
    ld a, 11
    ret
"#;

    let prog = aun_front::compile_str(0, src).expect("HSM frontend should succeed");
    let ext_asms = vec![
        aun_z80::ExternalAsm {
            name: "hal_disk.s".to_string(),
            source: hal_res_s.to_string(),
            owner_state: None,
            is_startup: false,
        },
        aun_z80::ExternalAsm {
            name: "ingame.s".to_string(),
            source: ext_ingame_s.to_string(),
            owner_state: Some("App::InGame".to_string()),
            is_startup: false,
        },
    ];
    let out = aun_z80::compile_z80_with_external_asms(
        &prog,
        &ext_asms,
        CompileOptions {
            code_origin: 0x0100,
            ram_base: 0x8000,
            ..CompileOptions::debug()
        },
    )
    .expect("HSM auto Uchiguri overlay compile should succeed");

    // 1. Verify overlay tree structure, automatic addresses, and auto-partitioning
    assert_eq!(out.overlays.len(), 3, "expected InGame, Stage1, Stage2 blobs");
    let res_end = 0x0100u16 + out.resident_bin.len() as u16;
    assert_eq!(out.overlays[0].state_name, "InGame");
    assert_eq!(out.overlays[0].parent_state, None);
    assert_eq!(out.overlays[0].origin, res_end);
    assert_eq!(out.overlays[0].asm_files, vec!["ingame.s".to_string()]);

    let s1 = &out.overlays[1];
    let s2 = &out.overlays[2];
    assert_eq!(s1.state_name, "InGame::Stage1");
    assert_eq!(s1.parent_state.as_deref(), Some("App::InGame"));
    assert_eq!(s2.state_name, "InGame::Stage2");
    assert_eq!(s2.parent_state.as_deref(), Some("App::InGame"));
    let expected_child_origin = res_end + out.overlays[0].bin.len() as u16;
    assert_eq!(s1.origin, expected_child_origin);
    assert_eq!(s2.origin, expected_child_origin, "sibling Stage1/Stage2 share origin");
    assert!(
        s1.asm_listing.contains("BossAi__"),
        "BossAi methods should be auto-partitioned into Stage1 blob:\n{}",
        s1.asm_listing
    );
    assert!(!out.symbols.contains_key("ext_ingame_bonus"));
    assert!(!out.symbols.contains_key("App__InGame__Stage1__update"));

    // 2. Verify sibling state buffers in HighMainRam share exact addresses after parent's buffer
    let gb_ig = out.globals.iter().find(|g| g.name.ends_with("INGAME_BUF")).unwrap();
    let gb_s1 = out.globals.iter().find(|g| g.name.ends_with("S1_BUF")).unwrap();
    let gb_s2 = out.globals.iter().find(|g| g.name.ends_with("S2_BUF")).unwrap();
    assert!((0xC000..=0xEFFF).contains(&gb_ig.addr));
    assert_eq!(gb_s1.addr, gb_ig.addr + 4);
    assert_eq!(gb_s2.addr, gb_ig.addr + 4);

    // 3. Execute on Z80 with overlay memory initially zeroed & blobs staged at 0xA000 + id * 0x0400!
    let mut cpu = Z80::new();
    let mut hal = MockHal::default();
    cpu.ram[out.code_origin as usize..out.code_origin as usize + out.resident_bin.len()]
        .copy_from_slice(&out.resident_bin);
    for ovl in &out.overlays {
        let disk_addr = 0xA000usize + (ovl.id as usize) * 0x0400;
        cpu.ram[disk_addr..disk_addr + ovl.bin.len()].copy_from_slice(&ovl.bin);
    }
    cpu.sp = 0x9FFF;
    cpu.write8(0xFFFF, 0x76);
    cpu.push16(0xFFFF);
    cpu.pc = out.symbols["main"];
    cpu.halted = false;
    while !cpu.halted {
        assert!(cpu.steps < 100_000, "Z80 timeout in HSM overlay test");
        cpu.step(&mut hal);
    }

    let read_global_u8 = |name: &str| -> u8 {
        let addr = out.globals.iter().find(|g| g.name == name).unwrap().addr;
        cpu.ram[addr as usize]
    };
    let read_global_u16 = |name: &str, idx: usize| -> u16 {
        let addr = out.globals.iter().find(|g| g.name == name).unwrap().addr as usize + idx * 2;
        u16::from_le_bytes([cpu.ram[addr], cpu.ram[addr + 1]])
    };

    assert_eq!(read_global_u8("SWAP_CALLS"), 2, "two boundary swaps (Title->Stage1, Stage1->Stage2)");
    assert_eq!(read_global_u8("SWAP_TOTAL_OPS"), 3, "total 3 LoadOps ([0, 1] then [2])");
    let op_log_addr = out.globals.iter().find(|g| g.name == "OP_LOG").unwrap().addr as usize;
    assert_eq!(&cpu.ram[op_log_addr..op_log_addr + 3], &[0, 1, 2]);

    // Trace events:
    // 0: InGame::on_enter -> 100 + 20 = 120
    // 1: Stage1::on_enter -> 200 + 3 = 203
    // 2: Stage1::update -> 210 + INGAME_BUF[0](0x10=16) + S1_BUF[0](3) = 229
    // 3: Stage1::on_exit -> 299 (NO InGame::on_exit!)
    // 4: Stage2::on_enter -> 300 + 7 = 307 (NO InGame::on_enter!)
    // 5: Stage2::update -> 310 + score(20 + 5 + 11 + 11 = 47) + INGAME_BUF[0](16) + S2_BUF[0](7) = 380
    // 6: InGame::on_exit -> 900
    assert_eq!(cpu.hl(), 7);
    let traces: Vec<u16> = (0..7).map(|i| read_global_u16("TRACE", i)).collect();
    assert_eq!(traces, vec![120, 203, 229, 299, 307, 380, 900]);
}

#[test]
fn z80_rejects_cross_overlay_direct_call() {
    let src = r#"
        target pc88sr : z80 {
            space MainRam @ 0x0100..0xBFFF;
            on_swap(ops: &const [LoadOp]) => noop_swap(ops);
        }
        fn noop_swap(_ops: &const [LoadOp]) {}

        #[overlay(App::Stage1)]
        extern "asm" {
            fn s1_only() -> u8 in A clobbers(AF);
        }

        machine App {
            initial Stage1;
            #[swap_boundary] state Stage1 { a: u8 };
            #[swap_boundary] state Stage2 { b: u8 };
            transitions {
                Stage1 -> Stage2;
                Stage2 -> Stage1;
            }
            on Stage1 {
                fn run(&mut self) -> u8 { s1_only() }
            }
            on Stage2 {
                fn run(&mut self) -> u8 { s1_only() }
            }
        }

        pub fn main() -> u8 {
            let mut app: App = App::new();
            app.run()
        }
    "#;
    let ext = aun_z80::ExternalAsm {
        name: "s1.s".to_string(),
        source: "s1_only:\n    ld a, 1\n    ret\n".to_string(),
        owner_state: Some("App::Stage1".to_string()),
        is_startup: false,
    };
    let prog = aun_front::compile_str(0, src).expect("front");
    let err = aun_z80::compile_z80_with_external_asms(&prog, &[ext], CompileOptions::debug())
        .unwrap_err();
    assert!(
        err.contains("illegal cross-overlay"),
        "expected illegal cross-overlay error, got: {err}"
    );
}

#[test]
fn z80_e2e_wcet_path_sensitive_loops_budgets_and_machine_dispatch() {
    // (a) if/else takes max(then, else) instead of sum
    let if_src = r#"
        static mut OUT_A: u8 = 0u8;
        pub fn branch_cmp(flag: bool, x: u8) -> u8 {
            if flag {
                let a = x +% 1u8;
                let b = a +% 2u8;
                let c = b +% 3u8;
                OUT_A = c;
                c
            } else {
                1u8
            }
        }
    "#;
    let if_prog = aun_front::compile_str(0, if_src).expect("if_src front");
    let if_out = aun_z80::compile_z80_with_options(&if_prog, &[], CompileOptions::release())
        .expect("if_src z80");
    let flat_ts = *if_out.fn_tstates.get("branch_cmp").expect("branch_cmp fn_tstates");
    let wcet_ts = *if_out.fn_wcet_tstates.get("branch_cmp").expect("branch_cmp wcet");
    assert!(
        wcet_ts < flat_ts,
        "path-sensitive WCET ({wcet_ts}T) must be strictly less than flat sum ({flat_ts}T) for if/else"
    );

    // (b) `for i in 0u8..8u8` automatically multiplies body WCET by 8 (and `0u8..=7u8` matches)
    let for_src = r#"
        static mut ACC: u8 = 0u8;
        #[budget(1000T)]
        pub fn sum_eight() -> u8 {
            let mut s: u8 = 0u8;
            for i in 0u8..8u8 {
                s = s +% i;
            }
            ACC = s;
            s
        }
        #[budget(1000T)]
        pub fn sum_eight_incl() -> u8 {
            let mut s: u8 = 0u8;
            for i in 0u8..=7u8 {
                s = s +% i;
            }
            ACC = s;
            s
        }
    "#;
    let for_prog = aun_front::compile_str(0, for_src).expect("for_src front");
    assert_eq!(for_prog.functions[0].loop_bounds.values().copied().collect::<Vec<_>>(), vec![8]);
    assert_eq!(for_prog.functions[1].loop_bounds.values().copied().collect::<Vec<_>>(), vec![8]);
    let for_out = aun_z80::compile_z80_with_options(&for_prog, &[], CompileOptions::release())
        .expect("for_src under 1000T budget");
    let sum8_wcet = *for_out.fn_wcet_tstates.get("sum_eight").unwrap();
    let sum8_flat = *for_out.fn_tstates.get("sum_eight").unwrap();
    assert!(
        sum8_wcet > sum8_flat * 3,
        "for 0u8..8u8 WCET ({sum8_wcet}T) should reflect 8x loop iterations vs flat code ({sum8_flat}T)"
    );

    // (c) `#[max_iters(16)] while ...` multiplies body by 16
    let while_src = r#"
        #[budget(4000T)]
        pub fn wait_loop(n_in: u8) -> u8 {
            let mut n: u8 = n_in;
            let mut acc: u8 = 0u8;
            #[max_iters(16)]
            while n > 0u8 {
                acc = acc +% n;
                n = n -% 1u8;
            }
            acc
        }
        pub fn wait_loop_1(n_in: u8) -> u8 {
            let mut n: u8 = n_in;
            let mut acc: u8 = 0u8;
            #[max_iters(1)]
            while n > 0u8 {
                acc = acc +% n;
                n = n -% 1u8;
            }
            acc
        }
    "#;
    let while_prog = aun_front::compile_str(0, while_src).expect("while_src front");
    let while_out = aun_z80::compile_z80_with_options(&while_prog, &[], CompileOptions::release())
        .expect("while_src under 4000T budget");
    let wait_wcet = *while_out.fn_wcet_tstates.get("wait_loop").unwrap();
    let wait1_wcet = *while_out.fn_wcet_tstates.get("wait_loop_1").unwrap();
    assert!(wait_wcet > wait1_wcet);
    assert_eq!(
        (wait_wcet - wait1_wcet) % 15,
        0,
        "#[max_iters(16)] ({wait_wcet}T) - #[max_iters(1)] ({wait1_wcet}T) must be 15x body WCET"
    );

    // (d) `#[budget(500T)]` passes when under limit and rejects when over limit
    let tight_ok = r#"
        #[budget(500T)]
        pub fn fast_fn(x: u8) -> u8 {
            x +% 1u8
        }
    "#;
    let ok_prog = aun_front::compile_str(0, tight_ok).expect("tight_ok front");
    let ok_out = aun_z80::compile_z80_with_options(&ok_prog, &[], CompileOptions::release())
        .expect("fast_fn should fit under 500T");
    assert!(*ok_out.fn_wcet_tstates.get("fast_fn").unwrap() <= 500);

    let tight_fail = r#"
        #[budget(vblank = 100T)]
        pub fn slow_fn() -> u8 {
            let mut s: u8 = 0u8;
            for i in 0u8..8u8 {
                s = s +% i;
            }
            s
        }
    "#;
    let fail_prog = aun_front::compile_str(0, tight_fail).expect("tight_fail front");
    let fail_err = aun_z80::compile_z80_with_options(&fail_prog, &[], CompileOptions::release())
        .unwrap_err();
    assert!(
        fail_err.contains("slow_fn") && fail_err.contains("vblank") && fail_err.contains("100"),
        "expected cycle budget exceeded error, got: {fail_err}"
    );

    // (e) Unbounded `while` inside `#[budget(...)]` rejects requesting `#[max_iters(N)]`
    let unb_src = r#"
        #[budget(5000)]
        pub fn unb_fn(x_in: u8) -> u8 {
            let mut x: u8 = x_in;
            while x > 0u8 {
                x = x -% 1u8;
            }
            x
        }
    "#;
    let unb_prog = aun_front::compile_str(0, unb_src).expect("unb_src front");
    let unb_err = aun_z80::compile_z80_with_options(&unb_prog, &[], CompileOptions::release())
        .unwrap_err();
    assert!(
        unb_err.contains("unb_fn") && unb_err.contains("#[max_iters(N)]"),
        "expected unbounded loop error asking for #[max_iters(N)], got: {unb_err}"
    );

    // (f) Dispatched `machine` method takes `max` across state handlers (both #[smc] and vtable)
    let mach_src = r#"
        #[smc]
        machine Actor {
            initial Idle;
            state Idle { x: u8 }
            state Heavy { x: u8 }
            transitions {
                Idle -> Heavy;
                Heavy -> Idle;
            }
            on Idle {
                fn step(&mut self) -> u8 {
                    1u8
                }
            }
            on Heavy {
                fn step(&mut self) -> u8 {
                    let mut s: u8 = 0u8;
                    for i in 0u8..8u8 {
                        s = s +% i;
                    }
                    self.x = s;
                    s
                }
            }
        }

        pub fn drive(a: &mut Actor) -> u8 {
            a.step()
        }
    "#;
    let mach_prog = aun_front::compile_str(0, mach_src).expect("mach_src front");
    let mach_out = aun_z80::compile_z80_with_options(&mach_prog, &[], CompileOptions::release())
        .expect("mach_src z80");
    let idle_wcet = *mach_out.fn_wcet_tstates.get("Actor::Idle::step").unwrap();
    let heavy_wcet = *mach_out.fn_wcet_tstates.get("Actor::Heavy::step").unwrap();
    let disp_wcet = *mach_out.fn_wcet_tstates.get("Actor_disp_step").unwrap();
    let drive_wcet = *mach_out.fn_wcet_tstates.get("drive").unwrap();
    assert!(heavy_wcet > idle_wcet);
    assert!(
        disp_wcet >= heavy_wcet,
        "dispatcher WCET ({disp_wcet}T) must be >= max state handler Heavy::step ({heavy_wcet}T)"
    );
    assert!(
        disp_wcet < heavy_wcet + idle_wcet,
        "dispatcher WCET ({disp_wcet}T) must take max, not sum ({}T)",
        heavy_wcet + idle_wcet
    );
    assert!(drive_wcet >= disp_wcet);
}

#[test]
fn z80_e2e_critical_owned_by_and_spsc_queue() {
    let src = r#"
        target pc88sr : z80 {
            space MainRam @ 0x0100..0x7FFF;
            stack MainStack @ 0x7E00..0x7FFF;
            interrupt_controller Pic {
                disable => asm!("di");
                enable  => asm!("ei");
                source Vrtc @ im2(0x7D00);
            }
        }

        struct Event {
            kind: u8,
            arg: u16,
        }

        #[owned_by(Vrtc)]
        static mut VRTC_COUNTER: u16 = 0u16;

        static mut BYTE_Q: Queue<u8, 4>;
        static mut EV_Q: Queue<Event, 4>;
        static EMPTY_EV_Q: Queue<Event, 4>;
        static mut LOG: [u16; 8];
        static mut LOG_LEN: u8 = 0u8;

        fn push_log(v: u16) {
            let idx: u8 = LOG_LEN;
            if idx < 8u8 {
                LOG[idx] = v;
                LOG_LEN = idx +% 1u8;
            }
        }

        #[interrupt(Vrtc)]
        fn on_vrtc() {
            VRTC_COUNTER = VRTC_COUNTER +% 1u16;
            // Nested `critical` inside ISR must NOT emit `ei` before `reti`!
            critical {
                let _ = EV_Q.push(Event { kind: 0xAAu8, arg: VRTC_COUNTER });
            }
        }

        // Also exercise runtime `&mut Queue<Event, 4>` pointer path
        fn pop_event_ref(q: &mut Queue<Event, 4>, out: &mut Event) -> bool {
            q.pop(out)
        }

        pub fn main() -> u16 {
            // 1. `critical { ... }` around #[owned_by(Vrtc)] emits `di`/`ei` (no duplicate for nested)
            let snap: u16 = critical {
                VRTC_COUNTER = VRTC_COUNTER +% 10u16;
                critical { VRTC_COUNTER }
            };
            push_log(snap); // 10

            // 2. Wrap-around FIFO on `Queue<u8, 4>` (usable capacity = 3)
            let mut ok: bool = BYTE_Q.is_empty();
            push_log(if ok { 1u16 } else { 0u16 }); // 1
            let _ = BYTE_Q.push(10u8);
            let _ = BYTE_Q.push(20u8);
            let _ = BYTE_Q.push(30u8);
            let full_push: bool = BYTE_Q.push(40u8);
            push_log(if full_push { 1u16 } else { 0u16 }); // 0
            push_log(BYTE_Q.len() as u16); // 3

            let mut b: u8 = 0u8;
            let _ = BYTE_Q.pop(&mut b);
            push_log(b as u16); // 10
            let _ = BYTE_Q.push(40u8); // wraps tail to 0!
            let _ = BYTE_Q.pop(&mut b);
            push_log(b as u16); // 20
            let _ = BYTE_Q.pop(&mut b);
            push_log(b as u16); // 30
            let _ = BYTE_Q.pop(&mut b);
            push_log(b as u16); // 40
            let empty_pop: bool = BYTE_Q.pop(&mut b);
            push_log(if empty_pop { 1u16 } else { 0u16 }); // 0

            // 3. Wrap-around FIFO on `Queue<Event, 4>` (3-byte struct!)
            let mut ev: Event = Event { kind: 0u8, arg: 0u16 };
            if snap == 10u16 {
                assert!(EV_Q.is_empty());
            }
            // Push 3 via main? Wait: on_vrtc pushes to EV_Q, so let's trigger ISR or test local Queue<Event, 4>!
            let mut ev2: Event = Event { kind: 0u8, arg: 0u16 };
            let mut local_q: Queue<Event, 4> = EMPTY_EV_Q; // zero-initialized local Queue<Event, 4>
            assert!(local_q.push(Event { kind: 1u8, arg: 0x1111u16 }));
            assert!(local_q.push(Event { kind: 2u8, arg: 0x2222u16 }));
            assert!(local_q.push(Event { kind: 3u8, arg: 0x3333u16 }));
            assert!(!local_q.push(Event { kind: 4u8, arg: 0x4444u16 }));
            assert!(local_q.len() == 3u8);
            assert!(pop_event_ref(&mut local_q, &mut ev2) && ev2.kind == 1u8 && ev2.arg == 0x1111u16);
            assert!(local_q.push(Event { kind: 4u8, arg: 0x4444u16 }));
            assert!(pop_event_ref(&mut local_q, &mut ev2) && ev2.kind == 2u8 && ev2.arg == 0x2222u16);
            assert!(pop_event_ref(&mut local_q, &mut ev2) && ev2.kind == 3u8 && ev2.arg == 0x3333u16);
            assert!(pop_event_ref(&mut local_q, &mut ev2) && ev2.kind == 4u8 && ev2.arg == 0x4444u16);
            assert!(!pop_event_ref(&mut local_q, &mut ev2));
            assert!(local_q.is_empty());

            // Return 0 if EV_Q is still empty before ISR fires (ev unused unless ISR runs)
            if EV_Q.pop(&mut ev) {
                ev.arg
            } else {
                snap
            }
        }
    "#;
    let prog = aun_front::compile_str(0, src).expect("front compile");
    let out = aun_z80::compile_z80_with_external_asms(
        &prog,
        &[],
        CompileOptions {
            code_origin: 0x0100,
            ram_base: 0x8000,
            ..CompileOptions::debug()
        },
    )
    .expect("Z80 compile");

    let fn_asm = |label: &str| -> String {
        let mut out_lines = Vec::new();
        let mut in_fn = false;
        let prefix = format!("{label}_");
        for line in out.asm_listing.lines() {
            let trimmed = line.trim();
            if trimmed == format!("{label}:") {
                in_fn = true;
                continue;
            }
            if in_fn {
                if let Some(lbl) = trimmed.strip_suffix(':')
                    && !lbl.contains(' ')
                    && !lbl.starts_with(&prefix)
                {
                    break;
                }
                out_lines.push(line);
            }
        }
        out_lines.join("\n")
    };

    // Verify `main` emits `di` and `ei` (exactly once for the nested critical block)
    let main_asm = fn_asm("main");
    let di_count = main_asm.lines().filter(|l| l.trim() == "di").count();
    let ei_count = main_asm.lines().filter(|l| l.trim() == "ei").count();
    assert_eq!(di_count, 1, "expected exactly 1 `di` in main:\n{main_asm}");
    assert_eq!(ei_count, 1, "expected exactly 1 `ei` in main:\n{main_asm}");

    // Verify `on_vrtc` does NOT emit `di`, and only emits `ei` once right before `reti`
    let isr_asm = fn_asm("on_vrtc");
    assert_eq!(
        isr_asm.lines().filter(|l| l.trim() == "di").count(),
        0,
        "ISR must not emit `di` for nested critical:\n{isr_asm}"
    );
    assert_eq!(
        isr_asm.lines().filter(|l| l.trim() == "ei").count(),
        1,
        "ISR must only emit `ei` once before `reti`:\n{isr_asm}"
    );

    // Run `main` on Z80 CPU emulator
    let mut cpu = Z80::new();
    let mut hal = MockHal::default();
    cpu.ram[out.code_origin as usize..out.code_origin as usize + out.resident_bin.len()]
        .copy_from_slice(&out.resident_bin);
    cpu.sp = 0x7FFF;
    cpu.write8(0xFFFF, 0x76);
    cpu.push16(0xFFFF);
    cpu.pc = out.symbols["main"];
    cpu.halted = false;
    while !cpu.halted {
        assert!(cpu.steps < 50_000, "Z80 timeout in main");
        cpu.step(&mut hal);
    }
    assert_eq!(cpu.hl(), 10);
    assert!(cpu.iff1, "critical exit must have re-enabled interrupts (iff1 == true)");

    let log_addr = out.globals.iter().find(|g| g.name == "LOG").unwrap().addr as usize;
    let logs: Vec<u16> = (0..8)
        .map(|i| u16::from_le_bytes([cpu.ram[log_addr + i * 2], cpu.ram[log_addr + i * 2 + 1]]))
        .collect();
    assert_eq!(logs, vec![10, 1, 0, 3, 10, 20, 30, 40]);

    // Also run `on_vrtc` ISR directly and verify it pushes Event { kind: 0xAA, arg: 11 } to EV_Q!
    cpu.sp = 0x7FFF;
    cpu.push16(0xFFFF);
    cpu.pc = out.symbols["on_vrtc"];
    cpu.halted = false;
    while !cpu.halted {
        assert!(cpu.steps < 60_000, "Z80 timeout in on_vrtc");
        cpu.step(&mut hal);
    }
    let vrtc_cnt_addr = out.globals.iter().find(|g| g.name == "VRTC_COUNTER").unwrap().addr as usize;
    let vrtc_cnt = u16::from_le_bytes([cpu.ram[vrtc_cnt_addr], cpu.ram[vrtc_cnt_addr + 1]]);
    assert_eq!(vrtc_cnt, 11);

    // Now run `main` again: EV_Q now has Event(0xAA, 11) so `EV_Q.pop(&mut ev)` succeeds and returns 11!
    cpu.sp = 0x7FFF;
    cpu.push16(0xFFFF);
    cpu.pc = out.symbols["main"];
    cpu.halted = false;
    while !cpu.halted {
        assert!(cpu.steps < 100_000, "Z80 timeout in second main");
        cpu.step(&mut hal);
    }
    assert_eq!(cpu.hl(), 11);
    assert!(cpu.iff1, "second main must exit cleanly without panic");
}

#[test]
fn z80_e2e_issue33_hot_inlining_and_outliner_loop_suppression() {
    let aun_src = r#"
        extern "asm" {
            fn _hw_tick(tag: u8 in A, val: u8 in L) clobbers(F);
        }

        struct Ball {
            x: u16,
            y: u16,
            vx: i16,
            vy: i16,
            px: u8,
            py: u8,
            active: bool,
        }

        struct State {
            balls: [Ball; 4],
            bounces: u16,
            scratch_sum: u16,
        }

        static mut G_STATE: State;

        fn sync_ball_px(bi: u8) {
            G_STATE.balls[bi as u16].px = (G_STATE.balls[bi as u16].x >> 4u16) as u8;
            G_STATE.balls[bi as u16].py = (G_STATE.balls[bi as u16].y >> 4u16) as u8;
        }

        fn guard_right_divider(bi: u8) {
            if G_STATE.balls[bi as u16].x > 1600u16 && G_STATE.balls[bi as u16].vx > 0i16 {
                G_STATE.balls[bi as u16].x = 1600u16;
                G_STATE.balls[bi as u16].vx = -G_STATE.balls[bi as u16].vx;
                G_STATE.bounces = G_STATE.bounces +% 1u16;
            }
        }

        // Larger loop-hot callee (> 100B, calls extern so not a tiny leaf) containing repeated
        // patterns that must NOT be outlined because it is reachable from physics_step_frame's loop.
        fn collide_hull(bi: u8) {
            if G_STATE.balls[bi as u16].y > 1200u16 {
                G_STATE.balls[bi as u16].y = 1200u16;
                G_STATE.balls[bi as u16].vy = -G_STATE.balls[bi as u16].vy;
                _hw_tick(1u8, G_STATE.balls[bi as u16].px);
                _hw_tick(2u8, G_STATE.balls[bi as u16].py);
                _hw_tick(3u8, bi);
                G_STATE.scratch_sum = G_STATE.scratch_sum +% G_STATE.balls[bi as u16].x;
            } else {
                _hw_tick(1u8, G_STATE.balls[bi as u16].px);
                _hw_tick(2u8, G_STATE.balls[bi as u16].py);
                _hw_tick(3u8, bi);
                G_STATE.scratch_sum = G_STATE.scratch_sum +% G_STATE.balls[bi as u16].y;
            }
            if G_STATE.balls[bi as u16].x < 16u16 {
                G_STATE.balls[bi as u16].x = 16u16;
                _hw_tick(1u8, G_STATE.balls[bi as u16].px);
                _hw_tick(2u8, G_STATE.balls[bi as u16].py);
                _hw_tick(3u8, bi);
            }
        }

        fn step_ball_substep(bi: u8) {
            G_STATE.balls[bi as u16].x = G_STATE.balls[bi as u16].x +% (G_STATE.balls[bi as u16].vx as u16);
            G_STATE.balls[bi as u16].y = G_STATE.balls[bi as u16].y +% (G_STATE.balls[bi as u16].vy as u16);
            guard_right_divider(bi);
            sync_ball_px(bi);
        }

        pub fn init_balls() {
            let mut bi: u8 = 0u8;
            while bi < 4u8 {
                G_STATE.balls[bi as u16].x = 1580u16 +% ((bi as u16) *% 16u16);
                G_STATE.balls[bi as u16].y = 1180u16 +% ((bi as u16) *% 16u16);
                G_STATE.balls[bi as u16].vx = 32i16;
                G_STATE.balls[bi as u16].vy = 32i16;
                G_STATE.balls[bi as u16].active = true;
                sync_ball_px(bi);
                bi = bi +% 1u8;
            }
            G_STATE.bounces = 0u16;
            G_STATE.scratch_sum = 0u16;
        }

        pub fn physics_step_frame() -> u16 {
            let mut bi: u8 = 0u8;
            while bi < 4u8 {
                if G_STATE.balls[bi as u16].active {
                    step_ball_substep(bi);
                    step_ball_substep(bi);
                    collide_hull(bi);
                }
                bi = bi +% 1u8;
            }
            let mut total: u16 = G_STATE.bounces;
            let mut j: u8 = 0u8;
            while j < 4u8 {
                total = total +% (G_STATE.balls[j as u16].px as u16) +% (G_STATE.balls[j as u16].py as u16);
                j = j +% 1u8;
            }
            total
        }

        // Cold functions outside any loop with repeated >= 10B blocks must still outline.
        pub fn cold_stage_a() {
            _hw_tick(0x10u8, 0x21u8);
            _hw_tick(0x11u8, 0x22u8);
            _hw_tick(0x12u8, 0x23u8);
            _hw_tick(0x13u8, 0x24u8);
        }

        pub fn cold_stage_b() {
            _hw_tick(0x10u8, 0x21u8);
            _hw_tick(0x11u8, 0x22u8);
            _hw_tick(0x12u8, 0x23u8);
            _hw_tick(0x13u8, 0x24u8);
            _hw_tick(0x14u8, 0x25u8);
        }
    "#;
    let ext_s = r#"
        .module hw_tick
        .optsdcc -mz80 sdcccall(1)
        .area   _CODE
        .globl  _hw_tick
_hw_tick::
        out     (0xF0), a
        ret
    "#;

    let prog = aun_front::compile_str(0, aun_src).expect("front compile");
    let out = aun_z80::compile_z80_with_options(
        &prog,
        &[ext_s],
        CompileOptions {
            code_origin: 0x0100,
            ram_base: 0xC000,
            ..CompileOptions::release()
        },
    )
    .expect("compile_z80_with_options should succeed");

    // 1. Small private helpers called in hot physics path must inline & vanish.
    assert!(
        !out.symbols.contains_key("guard_right_divider"),
        "guard_right_divider must inline into step_ball_substep:\n{}",
        out.asm_listing
    );
    assert!(
        !out.asm_listing.contains("call sync_ball_px"),
        "sync_ball_px must inline inside hot callers:\n{}",
        out.asm_listing
    );
    // All pub fn exports remain intact.
    for pub_sym in ["init_balls", "physics_step_frame", "cold_stage_a", "cold_stage_b"] {
        assert!(out.symbols.contains_key(pub_sym), "missing pub fn {pub_sym}");
    }

    // 2. Outliner must NOT extract __aun_out_* calls inside physics_step_frame or its loop-hot callees,
    // while cold_stage_a / cold_stage_b still outline cleanly.
    let extract_fn_body = |label: &str| -> String {
        let mut body = Vec::new();
        let mut active = false;
        let prefix = format!("{label}_");
        for line in out.asm_listing.lines() {
            let t = line.trim();
            if t == format!("{label}:") {
                active = true;
                continue;
            }
            if active {
                if let Some(lbl) = t.strip_suffix(':')
                    && !lbl.contains(' ')
                    && !lbl.starts_with(&prefix)
                {
                    break;
                }
                body.push(line);
            }
        }
        body.join("\n")
    };
    for hot_fn in ["physics_step_frame", "step_ball_substep", "collide_hull"] {
        if out.symbols.contains_key(hot_fn) {
            let body = extract_fn_body(hot_fn);
            assert!(
                !body.contains("__aun_out_"),
                "loop-hot function {hot_fn} must not call __aun_out_* helpers:\n{body}"
            );
        }
    }
    assert!(
        out.symbols.keys().any(|k| k.starts_with("__aun_out_")),
        "cold_stage_a/b should still emit __aun_out_* helper:\n{}",
        out.asm_listing
    );

    // 3. Also verify sdasz80 preserves all pub fn exports.
    let (sdas, _, _) =
        compile_z80_sdasz80("issue33_phys", &prog, CompileOptions::release()).expect("sdasz80");
    for c_sym in ["_init_balls::", "_physics_step_frame::", "_cold_stage_a::", "_cold_stage_b::"] {
        assert!(sdas.contains(c_sym), "sdasz80 missing {c_sym}");
    }

    // 4. Execute on Z80 emulator and verify correctness + cycle count.
    let mut cpu = Z80::new();
    let mut hal = MockHal::default();
    cpu.ram[out.code_origin as usize..out.code_origin as usize + out.resident_bin.len()]
        .copy_from_slice(&out.resident_bin);

    cpu.sp = 0xF000;
    cpu.write8(0xFFFF, 0x76);
    cpu.push16(0xFFFF);
    cpu.pc = out.symbols["init_balls"];
    cpu.halted = false;
    while !cpu.halted {
        assert!(cpu.steps < 100_000, "Z80 timeout in init_balls");
        cpu.step(&mut hal);
    }

    let t0 = cpu.tstates;
    cpu.sp = 0xF000;
    cpu.push16(0xFFFF);
    cpu.pc = out.symbols["physics_step_frame"];
    cpu.halted = false;
    while !cpu.halted {
        assert!(cpu.steps < 200_000, "Z80 timeout in physics_step_frame");
        cpu.step(&mut hal);
    }
    let step_tstates = cpu.tstates - t0;
    assert_eq!(cpu.sp, 0xF000);
    // Initial:
    // b0: x=1580, y=1180 -> +2*32 -> x=1644 (>1600 -> 1600, vx=-32, bounce #1), then 1600-32=1568 (px=98); y=1244 (py=77, collide_hull clamps y=1200)
    // b1: x=1596, y=1196 -> +32 -> 1628 (>1600 -> 1600, vx=-32, bounce #2) -> -32 -> 1568 (px=98); y=1260 (py=78)
    // b2: x=1612, y=1212 -> +32 -> 1644 (>1600 -> 1600, vx=-32, bounce #3) -> -32 -> 1568 (px=98); y=1276 (py=79)
    // b3: x=1628, y=1228 -> +32 -> 1660 (>1600 -> 1600, vx=-32, bounce #4) -> -32 -> 1568 (px=98); y=1292 (py=80)
    // total = 4 bounces + 4*98 + (77 + 78 + 79 + 80) = 4 + 392 + 314 = 710
    assert_eq!(cpu.hl(), 710, "unexpected physics_step_frame return value");
    eprintln!("[z80_e2e] issue33 physics_step_frame T-states = {step_tstates}");
}

#[test]
fn z80_e2e_static_struct_array_ix_promotion_bin_and_sdasz80() {
    let src = r#"
        struct Ball {
            x: i16,
            y: i16,
            vx: i16,
            vy: i16,
            active: u8,
            hp: u8,
            flags: u8,
        }

        struct GameState {
            score: u16,
            balls: [Ball; 4],
        }

        static mut G_STATE: GameState = GameState {
            score: 0u16,
            balls: [Ball {
                x: 0i16,
                y: 0i16,
                vx: 0i16,
                vy: 0i16,
                active: 0u8,
                hp: 0u8,
                flags: 0u8,
            }; 4],
        };
        static mut G_CUR: u8 = 0u8;
        static mut OTHER_BALLS: [Ball; 2] = [Ball {
            x: 0i16,
            y: 0i16,
            vx: 0i16,
            vy: 0i16,
            active: 0u8,
            hp: 0u8,
            flags: 0u8,
        }; 2];

        fn helper_bump(b: &mut Ball) {
            b.flags = b.flags +% 1u8;
        }

        fn nested_ix_callee(b: &mut Ball) {
            b.x = b.x +% 1i16;
            b.y = b.y -% 1i16;
            b.flags = b.flags +% 2u8;
        }

        fn nested_other_ix(oi: u8) -> i16 {
            OTHER_BALLS[oi].x = OTHER_BALLS[oi].x +% OTHER_BALLS[oi].vx;
            OTHER_BALLS[oi].y = OTHER_BALLS[oi].y +% OTHER_BALLS[oi].vy;
            OTHER_BALLS[oi].hp = OTHER_BALLS[oi].hp +% 1u8;
            OTHER_BALLS[oi].x +% OTHER_BALLS[oi].y
        }

        fn clobber_cur() {
            G_CUR = 2u8;
        }

        pub fn init_balls() {
            let mut i: u8 = 0u8;
            while i < 4u8 {
                G_STATE.balls[i].x = (i as i16) *% 10i16 -% 15i16;
                G_STATE.balls[i].y = (i as i16) *% 5i16;
                G_STATE.balls[i].vx = 3i16 -% (i as i16);
                G_STATE.balls[i].vy = -2i16;
                G_STATE.balls[i].active = if i == 0u8 { 0u8 } else { 1u8 };
                G_STATE.balls[i].hp = 10u8 +% i;
                G_STATE.balls[i].flags = i;
                i = i +% 1u8;
            }
            G_CUR = 1u8;
            let mut j: u8 = 0u8;
            while j < 2u8 {
                OTHER_BALLS[j].x = 100i16;
                OTHER_BALLS[j].y = 200i16;
                OTHER_BALLS[j].vx = 1i16;
                OTHER_BALLS[j].vy = 2i16;
                OTHER_BALLS[j].active = 1u8;
                OTHER_BALLS[j].hp = 5u8;
                OTHER_BALLS[j].flags = 0u8;
                j = j +% 1u8;
            }
        }

        pub fn step_ball(bi: u8) -> i16 {
            if G_STATE.balls[bi].active == 0u8 {
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

        pub fn sum_ball_in_loop(bi: u8) -> i16 {
            let mut i: u8 = 0u8;
            let mut acc: i16 = 0i16;
            while i < 3u8 {
                acc = acc +% G_STATE.balls[bi].x +% G_STATE.balls[bi].y +% G_STATE.balls[bi].vx;
                i = i +% 1u8;
            }
            acc
        }

        pub fn mutated_idx_soundness(bi: u8) -> i16 {
            let mut idx: u8 = bi;
            let x0: i16 = G_STATE.balls[idx].x;
            idx = idx +% 1u8;
            let y1: i16 = G_STATE.balls[idx].y;
            let vx1: i16 = G_STATE.balls[idx].vx;
            let hp1: i16 = G_STATE.balls[idx].hp as i16;
            x0 +% y1 +% vx1 +% hp1
        }

        pub fn aliasing_call_soundness() -> i16 {
            let a: i16 = G_STATE.balls[G_CUR].x +% G_STATE.balls[G_CUR].y +% G_STATE.balls[G_CUR].vx;
            clobber_cur();
            let b: i16 = G_STATE.balls[G_CUR].x +% G_STATE.balls[G_CUR].y +% G_STATE.balls[G_CUR].vx;
            a +% b
        }

        pub fn nested_ix_calls(bi: u8) -> i16 {
            G_STATE.balls[bi].x = G_STATE.balls[bi].x +% 10i16;
            G_STATE.balls[bi].y = G_STATE.balls[bi].y +% 20i16;
            nested_ix_callee(&mut G_STATE.balls[bi]);
            let o: i16 = nested_other_ix(0u8);
            G_STATE.balls[bi].x +% G_STATE.balls[bi].y +% (G_STATE.balls[bi].flags as i16) +% o
        }

        pub fn touch_nested_twice() -> i16 {
            let mut tmp: Ball = Ball {
                x: 1i16,
                y: 2i16,
                vx: 3i16,
                vy: 4i16,
                active: 1u8,
                hp: 5u8,
                flags: 6u8,
            };
            nested_ix_callee(&mut tmp);
            let o: i16 = nested_other_ix(1u8);
            tmp.x +% tmp.y +% (tmp.flags as i16) +% o
        }
    "#;
    let prog = aun_front::compile_str(0, src).expect("front compile");
    let out = aun_z80::compile_z80_with_options(&prog, &[], CompileOptions::release())
        .expect("compile_z80_with_options release");
    let out_dbg = aun_z80::compile_z80_with_options(&prog, &[], CompileOptions::debug())
        .expect("compile_z80_with_options debug");
    let (sdas, _, _) = compile_z80_sdasz80("balls_ix", &prog, CompileOptions::release())
        .expect("compile_z80_sdasz80");

    let extract_fn = |listing: &str, name: &str| -> String {
        let mut lines = Vec::new();
        let mut inside = false;
        let prefix = format!("{name}_");
        for line in listing.lines() {
            let t = line.trim();
            if t == format!("{name}:") {
                inside = true;
                continue;
            }
            if inside {
                if let Some(lbl) = t.strip_suffix(':')
                    && !lbl.contains(' ')
                    && !lbl.starts_with(&prefix)
                {
                    break;
                }
                lines.push(line);
            }
        }
        lines.join("\n")
    };

    let step_asm = extract_fn(&out.asm_listing, "step_ball");
    assert!(
        step_asm.contains("(ix + "),
        "expected step_ball to access Ball fields via (ix + d):\n{step_asm}"
    );
    assert!(
        step_asm.contains("dec (ix + ") || step_asm.contains("inc (ix + "),
        "expected step_ball to emit inc/dec (ix + d) for hp/flags:\n{step_asm}"
    );
    assert!(
        !step_asm.contains("(__aun_ram + "),
        "expected step_ball not to spill struct-array element pointer to __aun_ram:\n{step_asm}"
    );
    for field_off in ["0x0002", "0x0004", "0x0006", "0x0008", "0x0009", "0x000A"] {
        assert!(
            !step_asm.contains(&format!("ld de, {field_off}")),
            "expected no field offset `ld de, {field_off}` recomputation in step_ball:\n{step_asm}"
        );
    }

    let loop_asm = extract_fn(&out.asm_listing, "sum_ball_in_loop");
    let loop_body = loop_asm
        .split_once("sum_ball_in_loop_bb2:")
        .map(|(_, b)| b)
        .unwrap_or(&loop_asm);
    assert!(
        loop_body.contains("(ix + ") && !loop_body.contains("call ") && !loop_body.contains("ld de,"),
        "expected sum_ball_in_loop loop body to use (ix + d) with IX hoisted out of the loop:\n{loop_asm}"
    );

    let sdas_step = extract_fn(&sdas, "step_ball");
    assert!(
        sdas_step.contains("(ix)") && !sdas_step.contains("(ix + "),
        "expected SDASZ80 syntax d(ix) in step_ball:\n{sdas_step}"
    );

    for (mode_name, compiled) in [("release", &out), ("debug", &out_dbg)] {
        let mut cpu = Z80::new();
        let mut hal = MockHal::default();
        cpu.ram[compiled.code_origin as usize
            ..compiled.code_origin as usize + compiled.resident_bin.len()]
            .copy_from_slice(&compiled.resident_bin);

        let mut call_fn = |cpu: &mut Z80, name: &str, a: u8| -> i16 {
            cpu.sp = 0xF000;
            cpu.write8(0xFFFF, 0x76);
            cpu.push16(0xFFFF);
            cpu.a = a;
            cpu.ix = 0xBEEF;
            cpu.pc = compiled.symbols[name];
            cpu.halted = false;
            while !cpu.halted {
                assert!(cpu.steps < 50_000, "Z80 timeout in {name} ({mode_name})");
                cpu.step(&mut hal);
            }
            assert_eq!(cpu.ix, 0xBEEF, "{name} ({mode_name}) must preserve caller IX");
            cpu.hl() as i16
        };

        call_fn(&mut cpu, "init_balls", 0);
        // bi = 0: active == 0 -> -1
        assert_eq!(call_fn(&mut cpu, "step_ball", 0), -1, "{mode_name}");
        // bi = 1: x=-5+2=-3 (<0 -> vx=2->-2), y=5-2=3, hp=11-1=10, flags=1+1=2 => -3+3+10+2 = 12
        assert_eq!(call_fn(&mut cpu, "step_ball", 1), 12, "{mode_name}");
        // bi = 2: x=5+1=6, y=10-2=8, hp=12-1=11, flags=2+1=3 => 6+8+11+3 = 28
        assert_eq!(call_fn(&mut cpu, "step_ball", 2), 28, "{mode_name}");
        // bi = 1 in loop: x=-3, y=3, vx=-2 => (-3+3-2)*3 = -6
        assert_eq!(call_fn(&mut cpu, "sum_ball_in_loop", 1), -6, "{mode_name}");
        // mutated_idx_soundness(1): balls[1].x (-3) + balls[2].y (8) + balls[2].vx (1) + balls[2].hp (11) = 17
        assert_eq!(call_fn(&mut cpu, "mutated_idx_soundness", 1), 17, "{mode_name}");
        // aliasing_call_soundness(): G_CUR starts at 1 (-3+3-2 = -2), clobber_cur sets G_CUR=2 (6+8+1 = 15) => 13
        assert_eq!(call_fn(&mut cpu, "aliasing_call_soundness", 0), 13, "{mode_name}");
        // nested_ix_calls(1): balls[1] starts x=-3, y=3, flags=2 -> +10/+20 -> +1/-1, flags+2=4 -> x=8, y=22, flags=4
        // nested_other_ix(0): OTHER_BALLS[0] x=100+1=101, y=200+2=202 -> o=303 => 8+22+4+303 = 337
        assert_eq!(call_fn(&mut cpu, "nested_ix_calls", 1), 337, "{mode_name}");
        // touch_nested_twice(): tmp=(2,1,flags=8)->11, OTHER_BALLS[1] x=101, y=202 -> 303 => 314
        assert_eq!(call_fn(&mut cpu, "touch_nested_twice", 0), 314, "{mode_name}");
    }
}

#[test]
fn z80_e2e_signed_i16_compares_exhaustive_no_xor80() {
    let src = r#"
        pub fn cmp_branches(a: i16, b: i16) -> u16 {
            let mut r: u16 = 0u16;
            if a < b  { r = r | 0x0001u16; }
            if a <= b { r = r | 0x0002u16; }
            if a > b  { r = r | 0x0004u16; }
            if a >= b { r = r | 0x0008u16; }
            if a == b { r = r | 0x0010u16; }
            if a != b { r = r | 0x0020u16; }
            if a < 0i16  { r = r | 0x0040u16; }
            if a <= 0i16 { r = r | 0x0080u16; }
            if a > 0i16  { r = r | 0x0100u16; }
            if a >= 0i16 { r = r | 0x0200u16; }
            if a == 0i16 { r = r | 0x0400u16; }
            if a != 0i16 { r = r | 0x0800u16; }
            if 0i16 < a  { r = r | 0x1000u16; }
            if 0i16 >= a { r = r | 0x2000u16; }
            r
        }

        pub fn cmp_bools(a: i16, b: i16) -> u16 {
            let c0: bool = a < b;
            let c1: bool = a <= b;
            let c2: bool = a > b;
            let c3: bool = a >= b;
            let c4: bool = a == b;
            let c5: bool = a != b;
            let c6: bool = a < 0i16;
            let c7: bool = a <= 0i16;
            let c8: bool = a > 0i16;
            let c9: bool = a >= 0i16;
            let c10: bool = a == 0i16;
            let c11: bool = a != 0i16;
            let c12: bool = 0i16 < a;
            let c13: bool = 0i16 >= a;
            let mut r: u16 = 0u16;
            if c0  { r = r | 0x0001u16; }
            if c1  { r = r | 0x0002u16; }
            if c2  { r = r | 0x0004u16; }
            if c3  { r = r | 0x0008u16; }
            if c4  { r = r | 0x0010u16; }
            if c5  { r = r | 0x0020u16; }
            if c6  { r = r | 0x0040u16; }
            if c7  { r = r | 0x0080u16; }
            if c8  { r = r | 0x0100u16; }
            if c9  { r = r | 0x0200u16; }
            if c10 { r = r | 0x0400u16; }
            if c11 { r = r | 0x0800u16; }
            if c12 { r = r | 0x1000u16; }
            if c13 { r = r | 0x2000u16; }
            r
        }
    "#;
    let prog = aun_front::compile_str(0, src).expect("front compile");
    let out_rel = aun_z80::compile_z80_with_options(&prog, &[], CompileOptions::release())
        .expect("compile release");
    let out_dbg = aun_z80::compile_z80_with_options(&prog, &[], CompileOptions::debug())
        .expect("compile debug");
    let (sdas, _, _) = compile_z80_sdasz80("cmp_i16", &prog, CompileOptions::release())
        .expect("compile sdasz80");

    for (label, text) in [
        ("release", &out_rel.asm_listing),
        ("debug", &out_dbg.asm_listing),
        ("sdasz80", &sdas),
    ] {
        assert!(
            !text.contains("xor 0x80") && !text.contains("xor #0x80"),
            "no xor 0x80 should be emitted in {label}:\n{text}"
        );
    }

    let vals: &[i16] = &[-32768, -32767, -2, -1, 0, 1, 2, 32766, 32767];
    for (mode_name, compiled) in [("release", &out_rel), ("debug", &out_dbg)] {
        let mut cpu = Z80::new();
        let mut hal = MockHal::default();
        cpu.ram[compiled.code_origin as usize
            ..compiled.code_origin as usize + compiled.resident_bin.len()]
            .copy_from_slice(&compiled.resident_bin);

        for &a in vals {
            for &b in vals {
                let mut expected: u16 = 0;
                if a < b { expected |= 0x0001; }
                if a <= b { expected |= 0x0002; }
                if a > b { expected |= 0x0004; }
                if a >= b { expected |= 0x0008; }
                if a == b { expected |= 0x0010; }
                if a != b { expected |= 0x0020; }
                if a < 0 { expected |= 0x0040; }
                if a <= 0 { expected |= 0x0080; }
                if a > 0 { expected |= 0x0100; }
                if a >= 0 { expected |= 0x0200; }
                if a == 0 { expected |= 0x0400; }
                if a != 0 { expected |= 0x0800; }
                if 0 < a { expected |= 0x1000; }
                if 0 >= a { expected |= 0x2000; }

                for fn_name in ["cmp_branches", "cmp_bools"] {
                    cpu.sp = 0xF000;
                    cpu.write8(0xFFFF, 0x76);
                    cpu.push16(0xFFFF);
                    cpu.set_hl(a as u16);
                    cpu.set_de(b as u16);
                    cpu.pc = compiled.symbols[fn_name];
                    cpu.halted = false;
                    while !cpu.halted {
                        assert!(cpu.steps < 50_000, "Z80 timeout in {fn_name}");
                        cpu.step(&mut hal);
                    }
                    assert_eq!(
                        cpu.hl(),
                        expected,
                        "{mode_name} {fn_name} mismatch for a={a}, b={b}: got 0x{:04X}, expected 0x{expected:04X}",
                        cpu.hl()
                    );
                }
            }
        }
    }
}

#[test]
fn z80_e2e_u32_eq_avoids_aun_ram_copies() {
    // 1. Minimal module comparing two u32 globals in control flow and bool expressions:
    //    only the 8 bytes for G_A + G_B should exist in __aun_ram (0 bytes of scratch copies).
    let minimal_src = r#"
static mut G_A: u32 = 0u32;
static mut G_B: u32 = 0u32;

pub fn cmp_ne_branch() -> u8 {
    if G_A != G_B {
        1u8
    } else {
        0u8
    }
}

pub fn cmp_eq_bool() -> bool {
    G_A == G_B
}
"#;
    let min_prog = aun_front::compile_str(0, minimal_src).expect("front");
    let min_out = aun_z80::compile_z80_with_options(
        &min_prog,
        &[],
        CompileOptions {
            code_origin: 0x0100,
            ram_base: 0x8000,
            ..CompileOptions::release()
        },
    )
    .expect("Z80 compile minimal u32 eq");
    assert!(
        !min_out.asm_listing.contains("ld (0x800") && !min_out.asm_listing.contains("ld (__aun_ram"),
        "expected zero scratch stores to __aun_ram for u32 ==/!=:\n{}",
        min_out.asm_listing
    );

    let min_sdas = aun_z80::compile_z80_sdasz80(
        "u32_eq_min",
        &min_prog,
        CompileOptions::release(),
    )
    .expect("sdasz80 compile minimal u32 eq");
    assert!(
        min_sdas.0.contains("ld de, (__aun_ram"),
        "expected direct global load into DE in sdasz80:\n{}",
        min_sdas.0
    );
    assert!(
        !min_sdas.0.contains("__aun_ram + 8") && !min_sdas.0.contains("ld (__aun_ram"),
        "expected zero scratch copies to __aun_ram for u32 ==/!=:\n{}",
        min_sdas.0
    );

    // 2. Comprehensive u32 / i32 equality and inequality E2E tests across globals,
    //    constants, IX struct fields, aliasing (same ptr), and runtime array slots.
    let src = r#"
struct Pair32 {
    x: u32,
    y: u32,
}

static mut G_A: u32 = 0u32;
static mut G_B: u32 = 0u32;
static mut G_IA: i32 = 0i32;
static mut G_IB: i32 = 0i32;
static mut G_P1: Pair32 = Pair32 { x: 0u32, y: 0u32 };
static mut G_P2: Pair32 = Pair32 { x: 0u32, y: 0u32 };
static mut G_KEY: u32 = 0u32;
static mut G_IDX: u8 = 0u8;
static mut G_ARR: [u32; 2] = [0u32, 0u32];

fn check_ix_vs_arg(p: &Pair32, k: u32) -> u8 {
    let mut mask: u8 = 0u8;
    if p.x == k {
        mask = mask | 0x01u8;
    }
    if p.y != 0xABCD_0000u32 {
        mask = mask | 0x02u8;
    }
    if p.x == p.y {
        mask = mask | 0x04u8;
    }
    if p.x == p.x {
        mask = mask | 0x08u8;
    }
    mask
}

fn check_ix_alias(p: &Pair32, q: &Pair32) -> bool {
    p.x == q.x
}

pub fn run_u32_eq_suite() -> u16 {
    let mut res: u16 = 0u16;
    if G_A != G_B {
        res = res | 0x0001u16;
    }
    if G_A == G_B {
        res = res | 0x0002u16;
    }
    if G_A == 0u32 {
        res = res | 0x0004u16;
    }
    if 0x1234_0000u32 == G_B {
        res = res | 0x0008u16;
    }
    if G_IA == -1i32 {
        res = res | 0x0010u16;
    }
    if G_IA != G_IB {
        res = res | 0x0020u16;
    }
    let b_eq: bool = G_A == G_B;
    if b_eq {
        res = res | 0x0040u16;
    }
    let ix_bits: u8 = check_ix_vs_arg(&G_P1, G_KEY);
    res = res | ((ix_bits as u16) << 8u16);
    if check_ix_alias(&G_P1, &G_P2) {
        res = res | 0x0080u16;
    }
    if check_ix_alias(&G_P1, &G_P1) && (G_ARR[G_IDX] == G_KEY) {
        res = res | 0x1000u16;
    }
    res
}
"#;
    let prog = aun_front::compile_str(0, src).expect("front");
    let out = aun_z80::compile_z80_with_options(
        &prog,
        &[],
        CompileOptions {
            code_origin: 0x0100,
            ram_base: 0x8000,
            ..CompileOptions::release()
        },
    )
    .expect("Z80 compile u32 eq suite");

    let sdas = aun_z80::compile_z80_sdasz80("u32_eq_suite", &prog, CompileOptions::release())
        .expect("sdasz80 compile u32 eq suite");
    assert!(sdas.0.contains("run_u32_eq_suite"));

    let mut cpu = Z80::new();
    let mut hal = MockHal::default();
    cpu.ram[out.code_origin as usize..out.code_origin as usize + out.resident_bin.len()]
        .copy_from_slice(&out.resident_bin);

    let g_addr = |name: &str| -> usize {
        out.globals.iter().find(|g| g.name == name).unwrap().addr as usize
    };
    let write_u32 = |cpu: &mut Z80, addr: usize, v: u32| {
        let b = v.to_le_bytes();
        cpu.ram[addr..addr + 4].copy_from_slice(&b);
    };

    let cases: &[(u32, u32, i32, i32, u32, u32, u32, u32, u32, u8, u32, u32)] = &[
        // Matching both halves
        (0x1234_5678, 0x1234_5678, -1, -1, 0x9999_0001, 0xABCD_0000, 0x9999_0001, 0x0, 0x9999_0001, 0, 0x9999_0001, 0),
        // Low half differs, high half equal
        (0x1234_5678, 0x1234_5679, -1, 0, 0x1111_2222, 0x1111_2223, 0x1111_2222, 0, 0x1111_2222, 1, 0, 0x1111_2222),
        // Low half equal, high half differs
        (0x1234_5678, 0x1235_5678, 0x7FFF_0000, -0x7FFF_0000, 0xAAAA_5555, 0xAAAA_5555, 0xBBBB_5555, 0, 0xAAAA_5555, 1, 0, 0xFFFF_FFFF),
        // Zero and high-word const checks
        (0, 0x1234_0000, -0x8000_0000, -0x8000_0000, 0x0000_0001, 0xABCD_0000, 0x0000_0002, 0, 0x0000_0001, 0, 0, 0),
    ];

    for &(ga, gb, gia, gib, p1x, p1y, p2x, p2y, key, idx, a0, a1) in cases {
        write_u32(&mut cpu, g_addr("G_A"), ga);
        write_u32(&mut cpu, g_addr("G_B"), gb);
        write_u32(&mut cpu, g_addr("G_IA"), gia as u32);
        write_u32(&mut cpu, g_addr("G_IB"), gib as u32);
        write_u32(&mut cpu, g_addr("G_P1"), p1x);
        write_u32(&mut cpu, g_addr("G_P1") + 4, p1y);
        write_u32(&mut cpu, g_addr("G_P2"), p2x);
        write_u32(&mut cpu, g_addr("G_P2") + 4, p2y);
        write_u32(&mut cpu, g_addr("G_KEY"), key);
        cpu.ram[g_addr("G_IDX")] = idx;
        write_u32(&mut cpu, g_addr("G_ARR"), a0);
        write_u32(&mut cpu, g_addr("G_ARR") + 4, a1);

        cpu.sp = 0x7FFF;
        cpu.write8(0xFFFF, 0x76);
        cpu.push16(0xFFFF);
        cpu.pc = out.symbols["run_u32_eq_suite"];
        cpu.halted = false;
        while !cpu.halted {
            assert!(cpu.steps < 50_000, "Z80 timeout in run_u32_eq_suite");
            cpu.step(&mut hal);
        }

        let mut expected: u16 = 0;
        if ga != gb {
            expected |= 0x0001;
        }
        if ga == gb {
            expected |= 0x0002 | 0x0040;
        }
        if ga == 0 {
            expected |= 0x0004;
        }
        if gb == 0x1234_0000 {
            expected |= 0x0008;
        }
        if gia == -1 {
            expected |= 0x0010;
        }
        if gia != gib {
            expected |= 0x0020;
        }
        if p1x == p2x {
            expected |= 0x0080;
        }
        let mut ix_bits: u16 = 0;
        if p1x == key {
            ix_bits |= 0x01;
        }
        if p1y != 0xABCD_0000 {
            ix_bits |= 0x02;
        }
        if p1x == p1y {
            ix_bits |= 0x04;
        }
        ix_bits |= 0x08;
        expected |= ix_bits << 8;
        let arr_val = if idx == 0 { a0 } else { a1 };
        if arr_val == key {
            expected |= 0x1000;
        }
        assert_eq!(
            cpu.hl(),
            expected,
            "mismatch for case ga={ga:#x} gb={gb:#x} p1=({p1x:#x},{p1y:#x}) key={key:#x}"
        );
    }
}

#[test]
fn z80_e2e_loop_phi_reg_no_ram_roundtrip_and_swap_cycles() {
    // 1. put_2d-style loop: register-allocated loop variables must not round-trip to __aun_ram.
    let put_2d_src = r#"
pub fn put_2d(v_in: u8) -> u16 {
    let mut v: u8 = v_in;
    let mut d: u8 = 0u8;
    while v >= 10u8 {
        v = v -% 10u8;
        d = d +% 1u8;
    }
    ((d as u16) << 8u16) | (v as u16)
}
"#;
    let put_prog = aun_front::compile_str(0, put_2d_src).expect("front");
    let put_out = aun_z80::compile_z80_with_options(
        &put_prog,
        &[],
        CompileOptions {
            code_origin: 0x0100,
            ram_base: 0x8000,
            ..CompileOptions::release()
        },
    )
    .expect("Z80 compile put_2d");
    assert!(
        !put_out.asm_listing.contains("(0x800"),
        "put_2d loop should not access __aun_ram:\n{}",
        put_out.asm_listing
    );

    let put_sdas = aun_z80::compile_z80_sdasz80("put_2d", &put_prog, CompileOptions::release())
        .expect("sdasz80 compile put_2d");
    assert!(
        !put_sdas.0.contains("__aun_ram"),
        "put_2d must not access __aun_ram per iteration:\n{}",
        put_sdas.0
    );

    let mut cpu = Z80::new();
    let mut hal = MockHal::default();
    cpu.ram[put_out.code_origin as usize..put_out.code_origin as usize + put_out.resident_bin.len()]
        .copy_from_slice(&put_out.resident_bin);
    for v_in in [0u8, 1, 9, 10, 11, 19, 42, 99, 100, 200, 255] {
        cpu.sp = 0x7FFF;
        cpu.write8(0xFFFF, 0x76);
        cpu.push16(0xFFFF);
        cpu.a = v_in;
        cpu.pc = put_out.symbols["put_2d"];
        cpu.halted = false;
        while !cpu.halted {
            assert!(cpu.steps < 5_000, "Z80 timeout in put_2d({v_in})");
            cpu.step(&mut hal);
        }
        let expected = (((v_in / 10) as u16) << 8) | ((v_in % 10) as u16);
        assert_eq!(cpu.hl(), expected, "put_2d({v_in}) failed");
    }

    // 2. Swap cycle (a, b = b, a) and 3-way rotation cycle (a, b, c = b, c, a) + u16 Phi accumulators.
    let cycle_src = r#"
static mut C_A: u8 = 0u8;
static mut C_B: u8 = 0u8;
static mut C_C: u8 = 0u8;
static mut C_N: u8 = 0u8;
static mut C_STEP: u16 = 0u16;

pub fn run_swap2() -> u16 {
    let mut a: u8 = C_A;
    let mut b: u8 = C_B;
    let mut n: u8 = C_N;
    while n > 0u8 {
        let t: u8 = a;
        a = b;
        b = t;
        n = n -% 1u8;
    }
    ((a as u16) << 8u16) | (b as u16)
}

pub fn run_rot3() -> u16 {
    let mut a: u8 = C_A;
    let mut b: u8 = C_B;
    let mut c: u8 = C_C;
    let mut n: u8 = C_N;
    while n > 0u8 {
        let t: u8 = a;
        a = b;
        b = c;
        c = t;
        n = n -% 1u8;
    }
    ((a as u16) << 8u16) | ((b as u16) << 4u16) | (c as u16)
}

pub fn run_accum16() -> u16 {
    let mut acc: u16 = 0x1000u16;
    let step: u16 = C_STEP;
    let mut n: u8 = C_N;
    while n > 0u8 {
        acc = acc +% step;
        n = n -% 1u8;
    }
    acc
}
"#;
    let cyc_prog = aun_front::compile_str(0, cycle_src).expect("front");
    let cyc_out = aun_z80::compile_z80_with_options(
        &cyc_prog,
        &[],
        CompileOptions {
            code_origin: 0x0100,
            ram_base: 0x8000,
            ..CompileOptions::release()
        },
    )
    .expect("Z80 compile phi cycle tests");

    let mut cpu = Z80::new();
    let mut hal = MockHal::default();
    cpu.ram[cyc_out.code_origin as usize..cyc_out.code_origin as usize + cyc_out.resident_bin.len()]
        .copy_from_slice(&cyc_out.resident_bin);
    let g = |name: &str| -> usize {
        cyc_out.globals.iter().find(|g| g.name == name).unwrap().addr as usize
    };

    for n in 0u8..=5 {
        cpu.ram[g("C_A")] = 0x12;
        cpu.ram[g("C_B")] = 0xA7;
        cpu.ram[g("C_N")] = n;
        cpu.sp = 0x7FFF;
        cpu.write8(0xFFFF, 0x76);
        cpu.push16(0xFFFF);
        cpu.pc = cyc_out.symbols["run_swap2"];
        cpu.halted = false;
        while !cpu.halted {
            assert!(cpu.steps < 5_000, "Z80 timeout in run_swap2({n})");
            cpu.step(&mut hal);
        }
        let exp2 = if n % 2 == 0 { 0x12A7u16 } else { 0xA712u16 };
        assert_eq!(cpu.hl(), exp2, "run_swap2(n={n}) failed");

        cpu.ram[g("C_A")] = 1;
        cpu.ram[g("C_B")] = 2;
        cpu.ram[g("C_C")] = 3;
        cpu.ram[g("C_N")] = n;
        cpu.sp = 0x7FFF;
        cpu.push16(0xFFFF);
        cpu.pc = cyc_out.symbols["run_rot3"];
        cpu.halted = false;
        while !cpu.halted {
            assert!(cpu.steps < 5_000, "Z80 timeout in run_rot3({n})");
            cpu.step(&mut hal);
        }
        let exp3 = match n % 3 {
            0 => (1 << 8) | (2 << 4) | 3,
            1 => (2 << 8) | (3 << 4) | 1,
            _ => (3 << 8) | (1 << 4) | 2,
        };
        assert_eq!(cpu.hl(), exp3, "run_rot3(n={n}) failed");

        let step = 0x0123u16;
        cpu.ram[g("C_STEP")..g("C_STEP") + 2].copy_from_slice(&step.to_le_bytes());
        cpu.ram[g("C_N")] = n;
        cpu.sp = 0x7FFF;
        cpu.push16(0xFFFF);
        cpu.pc = cyc_out.symbols["run_accum16"];
        cpu.halted = false;
        while !cpu.halted {
            assert!(cpu.steps < 5_000, "Z80 timeout in run_accum16({n})");
            cpu.step(&mut hal);
        }
        assert_eq!(
            cpu.hl(),
            0x1000u16.wrapping_add(step.wrapping_mul(n as u16)),
            "run_accum16(n={n}) failed"
        );
    }
}

#[test]
fn z80_e2e_mulu16_32_and_muls16_32_8x16_fast_path() {
    let src = r#"
static mut IN_UA: u16 = 0u16;
static mut IN_UB: u16 = 0u16;
static mut OUT_U: u32 = 0u32;
static mut IN_SA: i16 = 0i16;
static mut IN_SB: i16 = 0i16;
static mut OUT_S: i32 = 0i32;

pub fn mul_u16_32() -> u32 {
    let r: u32 = (IN_UA as u32) * (IN_UB as u32);
    OUT_U = r;
    r
}

pub fn mul_s16_32() -> i32 {
    let r: i32 = (IN_SA as i32) * (IN_SB as i32);
    OUT_S = r;
    r
}
"#;
    let prog = aun_front::compile_str(0, src).expect("front");
    let out = aun_z80::compile_z80_with_options(
        &prog,
        &[],
        CompileOptions {
            code_origin: 0x0100,
            ram_base: 0x8000,
            ..CompileOptions::release()
        },
    )
    .expect("Z80 compile mul16->32 fast path");

    // WCET must remain finitely boundable through __aun_mulu16_32 / __aun_muls16_32.
    let wcet_dbg = aun_z80::wcet::compute_wcet(&prog, &out.asm_listing, &out.symbols, &out.overlays);
    assert!(
        out.fn_wcet_tstates.get("mul_u16_32").copied().unwrap_or(0) > 0,
        "expected finite WCET for mul_u16_32: bounds={:?}",
        wcet_dbg.fn_bounds
    );
    assert!(
        out.fn_wcet_tstates.get("mul_s16_32").copied().unwrap_or(0) > 0,
        "expected finite WCET for mul_s16_32: bounds={:?}",
        wcet_dbg.fn_bounds
    );

    let sdas = aun_z80::compile_z80_sdasz80("mul16_32", &prog, CompileOptions::release())
        .expect("sdasz80 compile mul16->32");
    assert!(sdas.0.contains("__aun_mulu16_32_8lp"));

    let mut cpu = Z80::new();
    let mut hal = MockHal::default();
    cpu.ram[out.code_origin as usize..out.code_origin as usize + out.resident_bin.len()]
        .copy_from_slice(&out.resident_bin);
    let g = |name: &str| -> usize {
        out.globals.iter().find(|g| g.name == name).unwrap().addr as usize
    };

    let run_u = |cpu: &mut Z80, hal: &mut MockHal, a: u16, b: u16| -> (u32, u64) {
        cpu.ram[g("IN_UA")..g("IN_UA") + 2].copy_from_slice(&a.to_le_bytes());
        cpu.ram[g("IN_UB")..g("IN_UB") + 2].copy_from_slice(&b.to_le_bytes());
        cpu.sp = 0x7FFF;
        cpu.write8(0xFFFF, 0x76);
        cpu.push16(0xFFFF);
        cpu.pc = out.symbols["mul_u16_32"];
        cpu.halted = false;
        cpu.steps = 0;
        let t0 = cpu.tstates;
        while !cpu.halted {
            assert!(cpu.steps < 5_000, "Z80 timeout in mul_u16_32({a}, {b})");
            cpu.step(hal);
        }
        let elapsed = cpu.tstates - t0;
        let out_addr = g("OUT_U");
        let mem_val = u32::from_le_bytes([
            cpu.ram[out_addr],
            cpu.ram[out_addr + 1],
            cpu.ram[out_addr + 2],
            cpu.ram[out_addr + 3],
        ]);
        let reg_val = ((cpu.hl() as u32) << 16) | (cpu.de() as u32);
        assert_eq!(mem_val, reg_val, "OUT_U != HL:DE for {a:#x} * {b:#x}");
        (mem_val, elapsed)
    };

    let run_s = |cpu: &mut Z80, hal: &mut MockHal, a: i16, b: i16| -> (i32, u64) {
        cpu.ram[g("IN_SA")..g("IN_SA") + 2].copy_from_slice(&a.to_le_bytes());
        cpu.ram[g("IN_SB")..g("IN_SB") + 2].copy_from_slice(&b.to_le_bytes());
        cpu.sp = 0x7FFF;
        cpu.write8(0xFFFF, 0x76);
        cpu.push16(0xFFFF);
        cpu.pc = out.symbols["mul_s16_32"];
        cpu.halted = false;
        cpu.steps = 0;
        let t0 = cpu.tstates;
        while !cpu.halted {
            assert!(cpu.steps < 5_000, "Z80 timeout in mul_s16_32({a}, {b})");
            cpu.step(hal);
        }
        let elapsed = cpu.tstates - t0;
        let out_addr = g("OUT_S");
        let mem_val = i32::from_le_bytes([
            cpu.ram[out_addr],
            cpu.ram[out_addr + 1],
            cpu.ram[out_addr + 2],
            cpu.ram[out_addr + 3],
        ]);
        let reg_val = (((cpu.hl() as u32) << 16) | (cpu.de() as u32)) as i32;
        assert_eq!(mem_val, reg_val, "OUT_S != HL:DE for {a} * {b}");
        (mem_val, elapsed)
    };

    let u_edge: &[u16] = &[0, 1, 2, 0x00FF, 0x0100, 0x0101, 0x7FFF, 0x8000, 0x8001, 0xFFFF];
    for &a in u_edge {
        for &b in u_edge {
            let (got, _) = run_u(&mut cpu, &mut hal, a, b);
            assert_eq!(got, (a as u32) * (b as u32), "u16*u16 failed: {a:#x} * {b:#x}");
        }
    }

    let s_edge: &[i16] = &[0, 1, -1, 127, -128, 255, -255, 256, -256, 0x7FFF, -0x7FFF, -32768];
    for &a in s_edge {
        for &b in s_edge {
            let (got, _) = run_s(&mut cpu, &mut hal, a, b);
            assert_eq!(got, (a as i32) * (b as i32), "i16*i16 failed: {a} * {b}");
        }
    }

    // Verify 8x16 fast path triggers on either operand having high byte == 0 and saves T-states.
    let (v_lhs8, t_lhs8) = run_u(&mut cpu, &mut hal, 0x00FF, 0xFFFF);
    let (v_rhs8, t_rhs8) = run_u(&mut cpu, &mut hal, 0xFFFF, 0x00FF);
    let (v_full, t_full) = run_u(&mut cpu, &mut hal, 0x01FF, 0xFFFF);
    assert_eq!(v_lhs8, 0x00FF * 0xFFFF);
    assert_eq!(v_rhs8, 0xFFFF * 0x00FF);
    assert_eq!(v_full, 0x01FF * 0xFFFF);
    assert!(
        t_lhs8 + 150 < t_full,
        "expected LHS hi==0 fast path to save >=150 T-states (lhs8={t_lhs8}, full16={t_full})"
    );
    assert!(
        t_rhs8 + 150 < t_full,
        "expected RHS hi==0 fast path to save >=150 T-states (rhs8={t_rhs8}, full16={t_full})"
    );

    let (vs_fast, ts_fast) = run_s(&mut cpu, &mut hal, -255, 0x7FFF);
    let (vs_full, ts_full) = run_s(&mut cpu, &mut hal, -257, 0x7FFF);
    assert_eq!(vs_fast, -255 * 0x7FFF);
    assert_eq!(vs_full, -257 * 0x7FFF);
    assert!(
        ts_fast + 150 < ts_full,
        "expected signed abs-hi==0 fast path to save >=150 T-states (fast={ts_fast}, full={ts_full})"
    );
}

#[test]
fn z80_e2e_hud_mem_write_ldir_slice_osr_and_aligned256_store() {
    let src = r#"
        struct Hud {
            line: [u8; 44],
        }

        // Force G_HUD onto the next 256B page (0xC100 under default ram_base 0xC000)
        // to test non-zero high-byte folding (`ld h, 0xC1`).
        static mut PAD_BEFORE: [u8; 7] = [0u8; 7];
        #[align(256)]
        static mut G_HUD: Hud = Hud { line: [0u8; 44] };

        pub fn tvram_flush(dst: u16) {
            let col0: u16 = 2u16;
            for i in 0u8..44u8 {
                mem_write_u8(dst +% col0 +% (i as u16), G_HUD.line[i as u16]);
            }
        }

        pub fn put_str(col: u8, s: &const [u8]) {
            let mut c: u8 = col;
            let mut i: u16 = 0u16;
            while i < s.len() {
                G_HUD.line[c as u16] = s[i];
                c = c +% 1u8;
                i = i +% 1u16;
            }
        }

        pub fn put_str_for(col: u8, s: &const [u8]) {
            let mut c: u8 = col;
            for i in 0u16..s.len() {
                G_HUD.line[c as u16] = s[i];
                c = c +% 1u8;
            }
        }

        pub fn read_vram_loop(src: u16) {
            for i in 0u8..44u8 {
                G_HUD.line[i as u16] = mem_read_u8(src +% (i as u16));
            }
        }

        pub fn fill_vram_loop(dst: u16) {
            for i in 0u8..20u8 {
                mem_write_u8(dst +% (i as u16), 0x20u8);
            }
        }

        pub fn run() -> u8 {
            PAD_BEFORE[0] = 0xAAu8;
            put_str(0u8, "SCORE");
            put_str_for(6u8, "0042");
            tvram_flush(0xD800u16);
            fill_vram_loop(0xDA00u16);
            PAD_BEFORE[0]
        }
    "#;
    let prog = aun_front::compile_str(0, src).expect("front compile");
    let out = aun_z80::compile_z80_with_options(&prog, &[], CompileOptions::release())
        .expect("compile_z80_with_options should succeed");

    let fn_asm = |label: &str| -> String {
        let mut inside = false;
        let mut lines = Vec::new();
        for line in out.asm_listing.lines() {
            let trimmed = line.trim();
            if trimmed == format!("{label}:") {
                inside = true;
            } else if inside && trimmed.ends_with(':') && !trimmed.starts_with(&format!("{label}_")) {
                break;
            }
            if inside {
                lines.push(line);
            }
        }
        lines.join("\n")
    };

    let flush_asm = fn_asm("tvram_flush");
    assert!(
        flush_asm.contains("    ldir"),
        "tvram_flush should lower mem_write_u8 copy loop to ldir:\n{flush_asm}"
    );
    let read_asm = fn_asm("read_vram_loop");
    assert!(
        read_asm.contains("    ldir"),
        "read_vram_loop should lower mem_read_u8 copy loop to ldir:\n{read_asm}"
    );
    let fill_asm = fn_asm("fill_vram_loop");
    assert!(
        fill_asm.contains("    ldir"),
        "fill_vram_loop should lower mem_write_u8 fill loop (len=20) to overlapping ldir:\n{fill_asm}"
    );

    let hud_addr = out.globals.iter().find(|g| g.name == "G_HUD").unwrap().addr;
    assert_eq!(hud_addr & 0x00FF, 0, "G_HUD must be 256B-aligned: {hud_addr:#06x}");
    assert_eq!(hud_addr, 0xC100, "G_HUD expected at 0xC100 after PAD_BEFORE");

    for fname in ["put_str", "put_str_for"] {
        let asm = fn_asm(fname);
        assert!(
            asm.contains("ld h, 0xC1"),
            "{fname} should fold 256B-aligned G_HUD.line[c] store into `ld h, 0xC1`:\n{asm}"
        );
        assert!(
            !asm.contains("add hl, de"),
            "{fname} should not emit `add hl, de` for either slice OSR or 256B-aligned store:\n{asm}"
        );
        assert!(
            asm.contains("inc ix") && asm.contains("dec bc"),
            "{fname} should walk slice via OSR IX pointer + BC countdown:\n{asm}"
        );
    }

    // Check sdasz80 mode preserves relocatable `#__aun_ram` (does not hardcode fixed `ld h, #0xC1`).
    let (sdas, _ram_bytes, _max_stack) =
        compile_z80_sdasz80("hud_e2e", &prog, CompileOptions::release())
            .expect("compile_z80_sdasz80 should succeed");
    assert!(
        !sdas.contains("ld h, #0xC1"),
        "sdasz80 must not hardcode fixed 256B page high byte for relocatable __aun_ram:\n{sdas}"
    );
    assert!(
        sdas.contains("ld de, #(__aun_ram + 256)")
            && sdas.contains("ldir")
            && sdas.contains("inc ix"),
        "sdasz80 should still emit ldir and slice OSR `inc ix` while keeping #(__aun_ram + 256) relocatable:\n{sdas}"
    );

    // Execute on Z80 emulator and verify runtime output!
    let mut cpu = Z80::new();
    let mut hal = MockHal::default();
    cpu.ram[out.code_origin as usize..out.code_origin as usize + out.resident_bin.len()]
        .copy_from_slice(&out.resident_bin);
    cpu.sp = 0xF000;
    cpu.write8(0xFFFF, 0x76);
    cpu.push16(0xFFFF);
    cpu.pc = out.symbols["run"];
    cpu.halted = false;
    while !cpu.halted {
        assert!(cpu.steps < 50_000, "Z80 step timeout in run");
        cpu.step(&mut hal);
    }
    assert_eq!(cpu.a, 0xAA);
    assert_eq!(&cpu.ram[0xD802..0xD807], b"SCORE");
    assert_eq!(cpu.ram[0xD807], 0);
    assert_eq!(&cpu.ram[0xD808..0xD80C], b"0042");
    assert_eq!(&cpu.ram[0xD80C..0xD82E], &[0u8; 34]);
    assert_eq!(&cpu.ram[0xDA00..0xDA14], &[0x20u8; 20]);

    // Also test read_vram_loop(0xD900) copies 44 bytes into G_HUD.line
    let pattern: Vec<u8> = (0u8..44u8).map(|i| i ^ 0x5A).collect();
    cpu.ram[0xD900..0xD92C].copy_from_slice(&pattern);
    cpu.sp = 0xF000;
    cpu.push16(0xFFFF);
    cpu.set_hl(0xD900);
    cpu.pc = out.symbols["read_vram_loop"];
    cpu.halted = false;
    while !cpu.halted {
        assert!(cpu.steps < 50_000, "Z80 step timeout in read_vram_loop");
        cpu.step(&mut hal);
    }
    assert_eq!(&cpu.ram[hud_addr as usize..hud_addr as usize + 44], &pattern[..]);
}

#[test]
fn z80_e2e_static_mut_nonzero_init_with_crt0_and_gsinit() {
    let src = r#"
pub static mut COUNTER: u16 = 0x1234u16;
pub static mut BYTE_INIT: u8 = 0x5Au8;
pub static mut TABLE: [u8; 4] = [0x11u8, 0x22u8, 0x33u8, 0x44u8];
pub static mut ZERO_VAL: u16 = 0u16;
pub static mut UNINIT_BUF: [u8; 4];

pub fn main() -> u16 {
    if ZERO_VAL != 0u16 || UNINIT_BUF[0] != 0u8 {
        return 0xDEADu16;
    }
    COUNTER = COUNTER + 1u16;
    COUNTER + (BYTE_INIT as u16) + (TABLE[2] as u16)
}
"#;
    let crt0 = r#"
    .area _CODE
_crt0_entry::
    ; Zero-fill s__DATA .. s__DATA + l__DATA (gsinit pattern)
    ld hl, #s__DATA
    ld bc, #l__DATA
_gs_zero_loop:
    ld a, b
    or c
    jr z, _gs_zero_done
    ld (hl), #0
    inc hl
    dec bc
    jr _gs_zero_loop
_gs_zero_done:
    ; Copy s__INITIALIZER -> s__INITIALIZED (l__INITIALIZER bytes)
    ld hl, #s__INITIALIZER
    ld de, #s__INITIALIZED
    ld bc, #l__INITIALIZER
    ld a, b
    or c
    jr z, _gs_init_done
    ldir
_gs_init_done:
    call _main
    ret
"#;
    let file = aun_front::Parser::parse_str(0, src).expect("parse");
    let prog = aun_front::Checker::check_and_lower(&file).expect("check");
    let mut opts = CompileOptions::release();
    opts.symbol_prefix = Some("_".to_string());
    let ext = vec![aun_z80::ExternalAsm {
        name: "crt0.s".to_string(),
        source: crt0.to_string(),
        owner_state: None,
        is_startup: true,
    }];
    let out = aun_z80::compile_z80_with_external_asms(&prog, &ext, opts)
        .expect("compile_z80_with_external_asms");

    let s_init = out.symbols["s__INITIALIZED"];
    let l_init = out.symbols["l__INITIALIZER"];
    let s_data = out.symbols["s__DATA"];
    assert!(l_init >= 7, "expected >= 7 initialized bytes, got {l_init}");
    assert_eq!(s_data, s_init + l_init);

    let mut cpu = Z80::new();
    let mut hal = MockHal::default();
    // Pre-poison RAM so we verify gsinit zeroes s__DATA without clobbering s__INITIALIZED.
    for b in &mut cpu.ram[0xC000..0xC080] {
        *b = 0xAA;
    }
    cpu.ram[out.code_origin as usize..out.code_origin as usize + out.resident_bin.len()]
        .copy_from_slice(&out.resident_bin);
    cpu.sp = 0xF000;
    cpu.write8(0xFFFF, 0x76);
    cpu.push16(0xFFFF);
    cpu.pc = out.code_origin;
    cpu.halted = false;
    while !cpu.halted {
        assert!(cpu.steps < 100_000, "Z80 timeout in gsinit e2e");
        cpu.step(&mut hal);
    }
    // 0x1235 + 0x5A + 0x33 = 0x12C2
    assert_eq!(cpu.hl(), 0x12C2);
    let table_addr = out.symbols["_TABLE"] as usize;
    assert_eq!(&cpu.ram[table_addr..table_addr + 4], &[0x11, 0x22, 0x33, 0x44]);
    let zero_addr = out.symbols["_ZERO_VAL"] as usize;
    assert_eq!(&cpu.ram[zero_addr..zero_addr + 2], &[0x00, 0x00]);
}

#[test]
fn z80_e2e_default_startup_init_and_reentrant_main() {
    let src = r#"
pub static mut X: u16 = 0x1000u16;
pub static mut Z: u8 = 0u8;

pub fn main() -> u16 {
    X = X + 1u16;
    Z = Z + 1u8;
    X + (Z as u16)
}
"#;
    let file = aun_front::Parser::parse_str(0, src).expect("parse");
    let prog = aun_front::Checker::check_and_lower(&file).expect("check");
    let out = compile_z80(&prog).expect("compile_z80");
    assert!(out.symbols.contains_key("__aun_start"));
    assert!(out.symbols.contains_key("__aun_startup_init"));

    let mut cpu = Z80::new();
    let mut hal = MockHal::default();
    cpu.ram[out.code_origin as usize..out.code_origin as usize + out.resident_bin.len()]
        .copy_from_slice(&out.resident_bin);

    // 1st invocation from code_origin runs __aun_start -> __aun_startup_init -> main.
    cpu.sp = 0xF000;
    cpu.write8(0xFFFF, 0x76);
    cpu.push16(0xFFFF);
    cpu.pc = out.code_origin;
    cpu.halted = false;
    while !cpu.halted {
        assert!(cpu.steps < 100_000);
        cpu.step(&mut hal);
    }
    assert_eq!(cpu.hl(), 0x1002);

    // 2nd invocation directly at `main` must preserve mutated RAM state (not re-init globals).
    cpu.sp = 0xF000;
    cpu.write8(0xFFFF, 0x76);
    cpu.push16(0xFFFF);
    cpu.pc = out.symbols["main"];
    cpu.halted = false;
    while !cpu.halted {
        assert!(cpu.steps < 100_000);
        cpu.step(&mut hal);
    }
    assert_eq!(cpu.hl(), 0x1004);
}

#[test]
fn z80_e2e_external_asm_private_label_scoping_and_symbol_prefix() {
    let src = r#"
#[export_name = "_custom_state"]
pub static mut STATE: u8 = 0x05u8;
pub static mut BASE_VAL: u16 = 0x0100u16;

#[export_name("_stage1_on_target_hit")]
pub fn hit_bonus() -> u16 {
    BASE_VAL + 7u16
}

pub fn main() -> u16 {
    BASE_VAL = BASE_VAL + (STATE as u16);
    BASE_VAL
}
"#;
    let mod_a = r#"
    .area _CODE
ITER = 3
_mod_a_sum::
    ld b, #ITER
    xor a
_psp_loop:
    add a, #0x10
    djnz _psp_loop
    jr 1$
    nop
1$:
    push af
    call _shared_bonus
    ld c, a
    pop af
    add a, c
    ret
_priv_ref:
    .dw _psp_loop
"#;
    let mod_b = r#"
    .area _CODE
ITER = 4
_mod_b_sum::
    ld b, #ITER
    xor a
_psp_loop:
    inc a
    djnz _psp_loop
    jr 1$
    nop
1$:
    push af
    call _shared_bonus
    ld c, a
    pop af
    add a, c
    ret
    .globl _shared_bonus
_shared_bonus:
    ld a, #2
    ret
_priv_ref:
    .dw _psp_loop
"#;
    let crt0 = r#"
    .area _CODE
_crt0_entry::
    call __aun_startup_init
    call _mod_a_sum
    ld e, a
    call _mod_b_sum
    add a, e
    ld e, a
    ld d, #0
    push de
    call _stage1_on_target_hit
    pop de
    add hl, de
    push hl
    call _main
    pop de
    add hl, de
    ld a, (_custom_state)
    ld e, a
    ld d, #0
    add hl, de
    ret
"#;
    let file = aun_front::Parser::parse_str(0, src).expect("parse");
    let prog = aun_front::Checker::check_and_lower(&file).expect("check");
    let mut opts = CompileOptions::release();
    opts.symbol_prefix = Some("_".to_string());
    let ext = vec![
        aun_z80::ExternalAsm {
            name: "crt0.s".to_string(),
            source: crt0.to_string(),
            owner_state: None,
            is_startup: true,
        },
        aun_z80::ExternalAsm {
            name: "mod_a.s".to_string(),
            source: mod_a.to_string(),
            owner_state: None,
            is_startup: false,
        },
        aun_z80::ExternalAsm {
            name: "mod_b.s".to_string(),
            source: mod_b.to_string(),
            owner_state: None,
            is_startup: false,
        },
    ];
    let out = aun_z80::compile_z80_with_external_asms(&prog, &ext, opts)
        .expect("compile_z80_with_external_asms");
    assert!(out.symbols.contains_key("_main"));
    assert!(out.symbols.contains_key("_stage1_on_target_hit"));
    assert!(out.symbols.contains_key("_custom_state"));
    assert!(out.symbols.contains_key("_BASE_VAL"));
    assert!(!out.symbols.contains_key("_psp_loop"));
    assert!(!out.symbols.contains_key("_priv_ref"));
    assert!(!out.symbols.contains_key("ITER"));

    let mut cpu = Z80::new();
    let mut hal = MockHal::default();
    cpu.ram[out.code_origin as usize..out.code_origin as usize + out.resident_bin.len()]
        .copy_from_slice(&out.resident_bin);
    cpu.sp = 0xF000;
    cpu.write8(0xFFFF, 0x76);
    cpu.push16(0xFFFF);
    cpu.pc = out.code_origin;
    cpu.halted = false;
    while !cpu.halted {
        assert!(cpu.steps < 100_000, "Z80 timeout in multi-ext scoping e2e");
        cpu.step(&mut hal);
    }
    // mod_a_sum = 3*0x10 + 2 = 0x32 (50)
    // mod_b_sum = 4*1 + 2 = 6
    // stage1_on_target_hit = 0x0100 + 7 = 0x0107 (263)
    // main = 0x0100 + 5 = 0x0105 (261)
    // _custom_state = 5
    // total = 50 + 6 + 263 + 261 + 5 = 585 (0x0249)
    assert_eq!(cpu.hl(), 585);
}

#[test]
fn z80_e2e_issue38_stack_arg_extern_call_not_outlined() {
    let aun_src = r#"
        extern "asm" {
            fn _callee_stack_u8(ptr: u16 in HL, val: u8 in stack) clobbers(A, B, C, D, E, H, L, F);
        }

        static mut FLAG: u8 = 0u8;
        static mut LOG: u16 = 0u16;

        pub fn stage_hit(kind: u8) -> u8 {
            if kind == 1u8 {
                _callee_stack_u8(0xC800u16, 5u8);
                if FLAG != 0u8 { return 1u8; }
            } else {
                _callee_stack_u8(0xC800u16, 5u8);
                if FLAG != 0u8 { return 2u8; }
            }
            0u8
        }

        pub fn main() -> u16 {
            let r1: u8 = stage_hit(1u8);
            let a1: u16 = LOG;
            let r2: u8 = stage_hit(2u8);
            let a2: u16 = LOG;
            a1 + a2 + (r1 as u16) + (r2 as u16)
        }
    "#;
    // Exact SDCC --sdcccall(1) 1-byte stack arg callee-cleanup:
    // Before `call`: caller executed `push af; inc sp` -> SP decreases by 1, arg byte at [SP+0].
    // After `call`: return address at [SP+0..+1], arg byte at [SP+2].
    // Callee pops return addr (hl), reads arg byte + pops 1 byte (`dec sp; pop af`), jumps `(hl)`.
    let ext_s_exact = r#"
        .area _CODE
        .globl _callee_stack_u8
    _callee_stack_u8:
        ex de, hl       ; ptr in DE
        pop hl          ; return addr in HL
        dec sp          ; align so high byte of AF is the 1-byte stack arg
        pop af          ; A = stack arg, SP += 2 -> net SP += 1 (callee cleaned 1B arg)
        xor e
        xor d
        ld (LOG), a
        ld a, #1
        ld (FLAG), a
        jp (hl)
    "#;
    let prog = aun_front::compile_str(0, aun_src).expect("front");

    // 1. Verify `--format sdasz80` never places `_callee_stack_u8` inside `__aun_out_*`.
    let (sdas_txt, _, _) =
        aun_z80::compile_z80_sdasz80("issue38", &prog, CompileOptions::release())
            .expect("sdasz80");
    if let Some(pos) = sdas_txt.find("__aun_out_") {
        assert!(
            !sdas_txt[pos..].contains("_callee_stack_u8"),
            "outlined helper must never call stack-arg extern `_callee_stack_u8`:\n{sdas_txt}"
        );
    }

    // 2. Verify `--format bin` + Z80 execution preserves SP & return address cleanly.
    let ext = vec![aun_z80::ExternalAsm {
        name: "callee.s".to_string(),
        source: ext_s_exact.to_string(),
        owner_state: None,
        is_startup: false,
    }];
    let out = aun_z80::compile_z80_with_external_asms(&prog, &ext, CompileOptions::release())
        .expect("bin");
    if let Some(pos) = out.asm_listing.find("__aun_out_") {
        assert!(
            !out.asm_listing[pos..].contains("_callee_stack_u8"),
            "outlined helper in bin mode must never call `_callee_stack_u8`:\n{}",
            out.asm_listing
        );
    }

    let mut cpu = Z80::new();
    let mut hal = MockHal::default();
    cpu.ram[out.code_origin as usize..out.code_origin as usize + out.resident_bin.len()]
        .copy_from_slice(&out.resident_bin);
    cpu.sp = 0xF000;
    cpu.write8(0xFFFF, 0x76);
    cpu.push16(0xFFFF);
    cpu.pc = out.code_origin;
    cpu.halted = false;
    while !cpu.halted {
        assert!(cpu.steps < 50_000, "Z80 timeout/corrupt return on #38");
        cpu.step(&mut hal);
    }
    assert_eq!(cpu.sp, 0xF000, "SP must remain balanced across stack-arg calls");
    // A = 5 ^ 0x00 ^ 0xC8 = 205; a1=205, a2=205, r1=1, r2=2 => 413
    assert_eq!(cpu.hl(), 413);
}

#[test]
fn z80_e2e_issue39_main_event_loop_retains_cold_outlining() {
    // Standalone `main() { loop { ... } }` event loops call every scene/menu handler in the
    // binary; transitive `loop_hot` must NOT suppress cold outlining across those handlers, while
    // running inside Z80 emulation with exact semantics.
    let src = r#"
        extern "asm" {
            fn _hw_sync(tick: u8 in A) clobbers(A, B, C, D, E, H, L, F);
        }

        static mut ACC: u16 = 0u16;
        static mut DONE: u8 = 0u8;
        static mut SX: u16 = 10u16;
        static mut SY: u16 = 20u16;
        static mut SZ: u16 = 30u16;

        pub fn scene_cold_a(x: u16, y: u16, z: u16) {
            ACC = ACC + (x ^ 0x13u16) + (y ^ 0x29u16) + (z ^ 0x37u16) + 7u16;
        }

        pub fn scene_cold_b(x: u16, y: u16, z: u16) {
            ACC = ACC + (x ^ 0x13u16) + (y ^ 0x29u16) + (z ^ 0x37u16) + 9u16;
        }

        pub fn scene_cold_c(x: u16, y: u16, z: u16) {
            ACC = ACC + (x ^ 0x13u16) + (y ^ 0x29u16) + (z ^ 0x37u16) + 11u16;
        }

        pub fn main() -> u16 {
            let mut phase: u8 = 0u8;
            loop {
                _hw_sync(phase);
                if phase == 0u8 {
                    scene_cold_a(SX, SY, SZ);
                } else if phase == 1u8 {
                    scene_cold_b(SX, SY, SZ);
                } else if phase == 2u8 {
                    scene_cold_c(SX, SY, SZ);
                } else {
                    DONE = 1u8;
                    break;
                }
                phase = phase + 1u8;
            }
            ACC
        }
    "#;
    let ext_s = r#"
        .area _CODE
        .globl _hw_sync
    _hw_sync:
        ret
    "#;
    let prog = aun_front::compile_str(0, src).expect("front");

    let (sdas_txt, _, _) =
        aun_z80::compile_z80_sdasz80("issue39", &prog, CompileOptions::release())
            .expect("sdasz80");
    assert!(
        sdas_txt.contains("__aun_out_"),
        "cold handlers called from main() event loop must still emit `__aun_out_*` helpers:\n{sdas_txt}"
    );

    let ext = vec![aun_z80::ExternalAsm {
        name: "sync.s".to_string(),
        source: ext_s.to_string(),
        owner_state: None,
        is_startup: false,
    }];
    let out = aun_z80::compile_z80_with_external_asms(&prog, &ext, CompileOptions::release())
        .expect("bin");
    assert!(
        out.asm_listing.contains("__aun_out_"),
        "bin mode must also emit `__aun_out_*` helpers under main() event loop:\n{}",
        out.asm_listing
    );

    let mut cpu = Z80::new();
    let mut hal = MockHal::default();
    cpu.ram[out.code_origin as usize..out.code_origin as usize + out.resident_bin.len()]
        .copy_from_slice(&out.resident_bin);
    cpu.sp = 0xF000;
    cpu.write8(0xFFFF, 0x76);
    cpu.push16(0xFFFF);
    cpu.pc = out.code_origin;
    cpu.halted = false;
    while !cpu.halted {
        assert!(cpu.steps < 50_000, "Z80 timeout on #39");
        cpu.step(&mut hal);
    }
    // (10^0x13) + (20^0x29) + (30^0x37) = 25 + 61 + 41 = 127
    // phase 0: 127 + 7 = 134; phase 1: 127 + 9 = 136; phase 2: 127 + 11 = 138 => total 408
    assert_eq!(cpu.sp, 0xF000);
    assert_eq!(cpu.hl(), 408);
}

#[test]
fn z80_e2e_issue40_phi_inlining_and_small_array_u8_mul() {
    // 33-byte struct matching `[Ball; 3]` (`3 * 33 = 99 <= 255`):
    // 2 + 2 + 1 + 1 + 1 + 26 = 33 bytes
    let src = r#"
        struct Struct33 {
            x: i16,
            y: i16,
            vx: i8,
            vy: i8,
            active: u8,
            pad: [u8; 26],
        }

        static mut ITEMS: [Struct33; 3] = [
            Struct33 { x: -5i16, y: 10i16, vx: 2i8, vy: 0i8, active: 1u8, pad: [1u8; 26] },
            Struct33 { x: 40i16, y: 250i16, vx: 0i8, vy: 3i8, active: 1u8, pad: [2u8; 26] },
            Struct33 { x: 15i16, y: 20i16, vx: 4i8, vy: 5i8, active: 0u8, pad: [3u8; 26] },
        ];

        // Multi-caller hot helper with if/else + &&/|| branches (Phi nodes in SSA).
        fn clamp_u8(v: i16) -> u8 {
            if v < 0i16 {
                0u8
            } else if v > 200i16 {
                200u8
            } else {
                v as u8
            }
        }

        fn is_moving(b: &Struct33) -> bool {
            b.active != 0u8 && (b.vx > 1i8 || b.vy > 1i8)
        }

        pub fn cold_helper_probe(v: i16) -> u8 {
            clamp_u8(v)
        }

        pub fn cold_moving_probe(idx: u8) -> u8 {
            if is_moving(&ITEMS[idx]) { 1u8 } else { 0u8 }
        }

        fn step_one(bi: u8) -> u16 {
            let cx: u8 = clamp_u8(ITEMS[bi].x);
            let cy: u8 = clamp_u8(ITEMS[bi].y);
            let mut bonus: u16 = 0u16;
            if is_moving(&ITEMS[bi]) {
                bonus = 7u16;
            }
            let total: u16 = (cx as u16) +% (cy as u16) +% (ITEMS[bi].pad[3] as u16) +% bonus;
            total
        }

        pub fn main() -> u16 {
            let mut sum: u16 = 0u16;
            let mut i: u8 = 0u8;
            while i < 3u8 {
                sum = sum +% step_one(i);
                i = i +% 1u8;
            }
            sum
        }
    "#;

    let prog = aun_front::compile_str(0, src).expect("front");

    let (sdas_txt, _, _) =
        compile_z80_sdasz80("issue40", &prog, CompileOptions::release()).expect("sdasz80");
    assert!(
        !sdas_txt.contains("call __aun_mul_hl_33") && !sdas_txt.contains("call __aun_mul_a_33"),
        "release sdasz80 must not emit call __aun_mul_hl_33 / __aun_mul_a_33:\n{sdas_txt}"
    );
    assert!(
        !sdas_txt.contains("__aun_mul_hl_33:") && !sdas_txt.contains("__aun_mul_a_33:"),
        "release sdasz80 must not define unused __aun_mul_*_33 helper:\n{sdas_txt}"
    );

    let out = aun_z80::compile_z80_with_options(&prog, &[], CompileOptions::release())
        .expect("release bin");
    assert!(
        !out.asm_listing.contains("call __aun_mul_hl_33")
            && !out.asm_listing.contains("call __aun_mul_a_33"),
        "release bin must not emit call __aun_mul_hl_33 / __aun_mul_a_33:\n{}",
        out.asm_listing
    );
    assert!(
        !out.asm_listing.contains("call step_one"),
        "single-caller hot loop helper step_one must be inlined:\n{}",
        out.asm_listing
    );
    // Extract main() body before next top-level label to confirm clamp_u8/is_moving inlined into hot loop.
    let main_pos = out
        .asm_listing
        .find("\nmain:\n")
        .expect("main label in listing");
    let main_tail = &out.asm_listing[main_pos + 1..];
    let main_end = main_tail
        .find("\ncold_")
        .unwrap_or(main_tail.len());
    let main_asm = &main_tail[..main_end];
    assert!(
        !main_asm.contains("call clamp_u8") && !main_asm.contains("call is_moving"),
        "hot Phi-bearing helpers clamp_u8 and is_moving must inline into main loop:\n{main_asm}"
    );

    // Run both Release and Debug binaries in Z80 emulator and verify exact checksum:
    // bi=0: clamp(-5)=0, clamp(10)=10, pad[3]=1, active=1 && vx=2>1 => bonus=7 => 18
    // bi=1: clamp(40)=40, clamp(250)=200, pad[3]=2, active=1 && vy=3>1 => bonus=7 => 249
    // bi=2: clamp(15)=15, clamp(20)=20, pad[3]=3, active=0 => bonus=0 => 38
    // total = 18 + 249 + 38 = 305
    for (mode_name, opts) in [
        ("release", CompileOptions::release()),
        ("debug", CompileOptions::debug()),
    ] {
        let bin_out = aun_z80::compile_z80_with_options(&prog, &[], opts).expect(mode_name);
        let mut cpu = Z80::new();
        let mut hal = MockHal::default();
        cpu.ram[bin_out.code_origin as usize..bin_out.code_origin as usize + bin_out.resident_bin.len()]
            .copy_from_slice(&bin_out.resident_bin);
        cpu.sp = 0xF000;
        cpu.write8(0xFFFF, 0x76);
        cpu.push16(0xFFFF);
        cpu.pc = bin_out.code_origin;
        cpu.halted = false;
        while !cpu.halted {
            assert!(cpu.steps < 50_000, "Z80 timeout on #40 ({mode_name})");
            cpu.step(&mut hal);
        }
        assert_eq!(cpu.sp, 0xF000, "SP mismatch ({mode_name})");
        assert_eq!(cpu.hl(), 305, "checksum mismatch ({mode_name})");
    }
}

#[test]
fn z80_e2e_issue41_slice_copy_loop_no_dead_len_and_reg_osr() {
    let src = r#"
        pub fn read_at(a: u16, s: &const [u8], idx: u16) -> u8 {
            s[idx] +% (a as u8)
        }

        pub fn put_str(col: u8, s: &const [u8]) -> u8 {
            let mut c: u8 = col;
            let mut i: u16 = 0u16;
            while i < s.len() && c < 44u8 {
                let ch: u8 = s[i];
                if ch == 0u8 {
                    return c;
                }
                mem_write_u8(0xF100u16 +% (c as u16), ch);
                c = c +% 1u8;
                i = i +% 1u16;
            }
            c
        }

        pub fn main() -> u16 {
            let r1: u8 = put_str(2u8, b"HELLO");
            let r2: u8 = put_str(10u8, b"AB\x00XYZ");
            let r3: u8 = put_str(40u8, b"0123456789");
            let r4: u8 = put_str(44u8, b"SKIP");
            let v1: u8 = read_at(1u16, b"WXYZ", 2u16);
            let v2: u8 = read_at(2u16, b"PQRS", 1u16);
            (r1 as u16) +% ((r2 as u16) << 4) +% (r3 as u16) +% (r4 as u16) +% (v1 as u16) +% (v2 as u16)
        }
    "#;
    let prog = aun_front::compile_str(0, src).expect("front");
    let out =
        aun_z80::compile_z80_with_options(&prog, &[], CompileOptions::release()).expect("z80");
    let asm = &out.asm_listing;

    let fn_slice = |name: &str| -> &str {
        let needle = format!("{name}:\n");
        let start = if asm.starts_with(&needle) {
            0
        } else {
            asm.find(&format!("\n{needle}"))
                .map(|p| p + 1)
                .unwrap_or_else(|| panic!("missing function label `{name}:` in:\n{asm}"))
        };
        let rest = &asm[start + needle.len()..];
        let end = rest
            .lines()
            .scan(start + needle.len(), |off, line| {
                let cur = *off;
                *off += line.len() + 1;
                Some((cur, line))
            })
            .find(|(_, line)| {
                !line.is_empty()
                    && !line.starts_with(' ')
                    && !line.starts_with('.')
                    && !line.starts_with(';')
                    && !line.starts_with(&format!("{name}_"))
                    && line.ends_with(':')
            })
            .map(|(cur, _)| cur)
            .unwrap_or(asm.len());
        &asm[start..end]
    };

    let read_at_asm = fn_slice("read_at");
    assert!(
        !read_at_asm.contains("ld d, h")
            && !read_at_asm.contains("ld e, l")
            && !read_at_asm.contains("(ix+8)"),
        "read_at must not reload/copy dead slice .len in release mode:\n{read_at_asm}"
    );

    let put_str_asm = fn_slice("put_str");
    let loop_start = put_str_asm
        .find("ld a, (hl)")
        .unwrap_or_else(|| panic!("expected walking-pointer OSR `ld a, (hl)` in put_str:\n{put_str_asm}"));
    let loop_end = put_str_asm[loop_start..]
        .find("djnz")
        .map(|o| loop_start + o + 4)
        .unwrap_or_else(|| panic!("expected `djnz` countdown in put_str:\n{put_str_asm}"));
    let loop_body = &put_str_asm[loop_start..loop_end];
    assert!(
        loop_body.contains("or a")
            && loop_body.contains("ld (de), a")
            && loop_body.contains("inc hl")
            && loop_body.contains("inc e"),
        "expected tight walking-pointer loop in put_str:\n{loop_body}"
    );
    assert!(
        !loop_body.contains("__aun_ram")
            && !loop_body.contains("ex de, hl")
            && !loop_body.contains("add hl, de"),
        "hot copy loop body must not reload fat-ptr len or spill to __aun_ram:\n{loop_body}"
    );

    let mut cpu = Z80::new();
    let mut hal = MockHal::default();
    cpu.ram[out.code_origin as usize..out.code_origin as usize + out.resident_bin.len()]
        .copy_from_slice(&out.resident_bin);
    // Sentinel values in VRAM around write ranges to catch overruns.
    for addr in 0xF100u16..=0xF135u16 {
        cpu.ram[addr as usize] = 0xAA;
    }
    cpu.sp = 0xF000;
    cpu.write8(0xFFFF, 0x76);
    cpu.push16(0xFFFF);
    cpu.pc = out.code_origin;
    cpu.halted = false;
    while !cpu.halted {
        assert!(cpu.steps < 50_000, "Z80 timeout on #41");
        cpu.step(&mut hal);
    }

    assert_eq!(cpu.sp, 0xF000);
    // r1 = 7, r2 = 12, r3 = 44, r4 = 44, v1 = b'Y' + 1 = 90, v2 = b'Q' + 2 = 83
    // -> 7 + (12 << 4) + 44 + 44 + 90 + 83 = 7 + 192 + 44 + 44 + 90 + 83 = 460
    assert_eq!(
        cpu.hl(),
        460,
        "mismatch: hl={}, vram_02={:?}, vram_0a={:?}, vram_28={:?}\nput_str:\n{put_str_asm}\nread_at:\n{read_at_asm}",
        cpu.hl(),
        &cpu.ram[0xF102..0xF108],
        &cpu.ram[0xF10A..0xF10E],
        &cpu.ram[0xF128..0xF12E],
    );
    assert_eq!(&cpu.ram[0xF102..0xF107], b"HELLO");
    assert_eq!(cpu.ram[0xF107], 0xAA);
    assert_eq!(&cpu.ram[0xF10A..0xF10C], b"AB");
    assert_eq!(cpu.ram[0xF10C], 0xAA);
    assert_eq!(&cpu.ram[0xF128..0xF12C], b"0123");
    assert_eq!(cpu.ram[0xF12C], 0xAA);
}

#[test]
fn z80_e2e_issue42_u8_rot_shifts_u16_bitpack_and_ldir_memset() {
    let src = r#"
        struct PackInput {
            f0: u8,
            f1: u8,
            f2: u8,
            f3: u8,
            f4: u8,
            f5: u8,
            f6: u8,
            f7: u8,
        }

        static mut INPUT: PackInput = PackInput {
            f0: 0u8,
            f1: 0u8,
            f2: 0u8,
            f3: 0u8,
            f4: 0u8,
            f5: 0u8,
            f6: 0u8,
            f7: 0u8,
        };
        static mut GUARD_PRE: u8 = 0xAAu8;
        static mut BUF44: [u8; 44] = [0u8; 44];
        static mut GUARD_POST: u8 = 0x55u8;
        static mut BUF8: [u8; 8] = [0u8; 8];
        static mut SHIFT_OUT: [u8; 9] = [0u8; 9];

        pub fn run_u8_shifts(u: u8, s: i8) {
            SHIFT_OUT[0] = u << 5;
            SHIFT_OUT[1] = u << 6;
            SHIFT_OUT[2] = u << 7;
            SHIFT_OUT[3] = u >> 5;
            SHIFT_OUT[4] = u >> 6;
            SHIFT_OUT[5] = u >> 7;
            SHIFT_OUT[6] = (s >> 5) as u8;
            SHIFT_OUT[7] = (s >> 6) as u8;
            SHIFT_OUT[8] = (s >> 7) as u8;
        }

        pub fn pack_u16_key(p: &PackInput) -> u16 {
            ((p.f0 as u16))
                | (((p.f1 & 0x07u8) as u16) << 2)
                | ((p.f2 as u16) << 5)
                | ((p.f3 as u16) << 6)
                | ((p.f4 as u16) << 7)
                | ((p.f5 as u16) << 8)
                | ((p.f6 as u16) << 12)
                | ((p.f7 as u16) << 15)
        }

        pub fn fill_buffers(v44: u8, v8: u8) {
            let mut i: u8 = 0u8;
            while i < 44u8 {
                BUF44[i] = v44;
                i = i +% 1u8;
            }
            let mut j: u8 = 0u8;
            while j < 8u8 {
                BUF8[j] = v8;
                j = j +% 1u8;
            }
        }

        pub fn main() -> u16 {
            // Test edge values for u8 << 5..7, u8 >> 5..7, and i8 >> 5..7
            let vals: [u8; 6] = [0x00u8, 0xFFu8, 0xA5u8, 0x81u8, 0x5Au8, 0x01u8];
            let mut idx: u8 = 0u8;
            while idx < 6u8 {
                let v = vals[idx];
                let sv = v as i8;
                run_u8_shifts(v, sv);
                if SHIFT_OUT[0] != (v << 5)
                    || SHIFT_OUT[1] != (v << 6)
                    || SHIFT_OUT[2] != (v << 7)
                    || SHIFT_OUT[3] != (v >> 5)
                    || SHIFT_OUT[4] != (v >> 6)
                    || SHIFT_OUT[5] != (v >> 7)
                    || SHIFT_OUT[6] != ((sv >> 5) as u8)
                    || SHIFT_OUT[7] != ((sv >> 6) as u8)
                    || SHIFT_OUT[8] != ((sv >> 7) as u8)
                {
                    return 0xEE00u16 +% (idx as u16);
                }
                idx = idx +% 1u8;
            }

            // Test 44-byte LDIR memset and 8-byte short memset without buffer overrun
            fill_buffers(0x3Cu8, 0x7Eu8);
            if GUARD_PRE != 0xAAu8 || GUARD_POST != 0x55u8 {
                return 0xEE10u16;
            }
            let mut k: u8 = 0u8;
            while k < 44u8 {
                if BUF44[k] != 0x3Cu8 {
                    return 0xEE20u16 +% (k as u16);
                }
                k = k +% 1u8;
            }
            let mut m: u8 = 0u8;
            while m < 8u8 {
                if BUF8[m] != 0x7Eu8 {
                    return 0xEE30u16 +% (m as u16);
                }
                m = m +% 1u8;
            }

            // Test bit-packed u16 key across edge inputs
            INPUT.f0 = 0x03u8;
            INPUT.f1 = 0x05u8; // (0x05 & 7) << 2 = 0x0014
            INPUT.f2 = 0x02u8; // << 5 = 0x0040
            INPUT.f3 = 0x03u8; // << 6 = 0x00C0
            INPUT.f4 = 0x05u8; // << 7 = 0x0280
            INPUT.f5 = 0x0Au8; // << 8 = 0x0A00
            INPUT.f6 = 0x06u8; // << 12 = 0x6000
            INPUT.f7 = 0x01u8; // << 15 = 0x8000
            // 0x0003 | 0x0014 | 0x0040 | 0x00C0 | 0x0280 | 0x0A00 | 0x6000 | 0x8000 = 0xEAD7
            let p1 = pack_u16_key(&INPUT);
            if p1 != 0xEAD7u16 {
                return p1;
            }

            INPUT.f0 = 0x00u8;
            INPUT.f1 = 0xFFu8; // &7 << 2 = 0x001C
            INPUT.f2 = 0xA5u8; // << 5 = 0x14A0
            INPUT.f3 = 0x81u8; // << 6 = 0x2040
            INPUT.f4 = 0xFFu8; // << 7 = 0x7F80
            INPUT.f5 = 0x01u8; // << 8 = 0x0100
            INPUT.f6 = 0x02u8; // << 12 = 0x2000
            INPUT.f7 = 0x00u8;
            // 0x001C | 0x14A0 | 0x2040 | 0x7F80 | 0x0100 | 0x2000 = 0x7FFC
            pack_u16_key(&INPUT)
        }
    "#;
    let prog = aun_front::compile_str(0, src).expect("front");
    let out = aun_z80::compile_z80_with_options(&prog, &[], CompileOptions::release())
        .expect("compile_z80_with_options");

    let fn_asm = |label: &str| -> String {
        let mut inside = false;
        let mut lines = Vec::new();
        for line in out.asm_listing.lines() {
            let trimmed = line.trim();
            if trimmed == format!("{label}:") {
                inside = true;
            } else if inside && trimmed.ends_with(':') && !trimmed.starts_with(&format!("{label}_")) {
                break;
            }
            if inside {
                lines.push(line);
            }
        }
        lines.join("\n")
    };

    let sh_asm = fn_asm("run_u8_shifts");
    assert!(
        sh_asm.contains("rrca") && sh_asm.contains("rlca"),
        "run_u8_shifts must use rrca/rlca for u8 << 5..7 and >> 5..7:\n{sh_asm}"
    );
    assert!(
        sh_asm.contains("sra a"),
        "run_u8_shifts must preserve exact sra a for signed i8 >> 5..7:\n{sh_asm}"
    );

    let fill_asm = fn_asm("fill_buffers");
    assert!(
        fill_asm.contains("    ldir"),
        "fill_buffers must lower 44-byte memset to overlapping ldir:\n{fill_asm}"
    );
    assert!(
        fill_asm.contains("djnz fill_buffers_memset_"),
        "fill_buffers must preserve djnz for 8-byte short memset:\n{fill_asm}"
    );

    let pack_asm = fn_asm("pack_u16_key");
    let pack_body = pack_asm
        .split_once("pop ix")
        .map(|(_, rest)| rest)
        .unwrap_or(&pack_asm);
    assert!(
        !pack_body.contains("push hl") && !pack_body.contains("pop hl"),
        "pack_u16_key must not emit push hl / pop hl across bit-pack chain:\n{pack_asm}"
    );
    assert!(
        pack_body.contains("ex de, hl") && pack_body.contains("or h") && pack_body.contains("or l"),
        "pack_u16_key should use ex de, hl and half-word or h / or l:\n{pack_asm}"
    );

    let mut cpu = Z80::new();
    let mut hal = MockHal::default();
    cpu.ram[out.code_origin as usize..out.code_origin as usize + out.resident_bin.len()]
        .copy_from_slice(&out.resident_bin);
    cpu.sp = 0xF000;
    cpu.write8(0xFFFF, 0x76);
    cpu.push16(0xFFFF);
    cpu.pc = out.code_origin;
    cpu.halted = false;
    while !cpu.halted {
        assert!(cpu.steps < 100_000, "Z80 timeout on #42");
        cpu.step(&mut hal);
    }
    assert_eq!(cpu.sp, 0xF000);
    assert_eq!(cpu.hl(), 0x7FFC);

    // Explicitly verify every shift result in SHIFT_OUT for the last iteration (v = 0x01, s = 1)
    let out_addr = out.globals.iter().find(|g| g.name == "SHIFT_OUT").unwrap().addr;
    assert_eq!(cpu.read8(out_addr), 0x01u8 << 5);
    assert_eq!(cpu.read8(out_addr + 1), 0x01u8 << 6);
    assert_eq!(cpu.read8(out_addr + 2), 0x01u8 << 7);
    assert_eq!(cpu.read8(out_addr + 3), 0x01u8 >> 5);
    assert_eq!(cpu.read8(out_addr + 4), 0x01u8 >> 6);
    assert_eq!(cpu.read8(out_addr + 5), 0x01u8 >> 7);

    // Explicitly verify all 44 bytes of BUF44 and the pre/post guard bytes in emulator RAM
    let guard_pre = out.globals.iter().find(|g| g.name == "GUARD_PRE").unwrap().addr;
    let buf44 = out.globals.iter().find(|g| g.name == "BUF44").unwrap().addr;
    let guard_post = out.globals.iter().find(|g| g.name == "GUARD_POST").unwrap().addr;
    assert_eq!(cpu.read8(guard_pre), 0xAA);
    for i in 0..44u16 {
        assert_eq!(cpu.read8(buf44 + i), 0x3C, "BUF44[{i}] mismatch");
    }
    assert_eq!(cpu.read8(guard_post), 0x55);
}

#[test]
fn z80_e2e_issue43_signed_cmp_fallthrough_high_byte_spills_and_scan() {
    let src = r#"
        struct Ball {
            pad0: u8,
            active: u8,
            x: i16,
            vx: i16,
            ticks: u8,
            last_idx: u8,
        }

        static mut BALLS: [Ball; 6] = [
            Ball { pad0: 0u8, active: 0u8, x: 0i16, vx: 0i16, ticks: 0u8, last_idx: 0u8 }; 6
        ];

        struct Node {
            f0: u16,
            f1: u16,
        }

        pub fn branch_masks(x: i16, y: i16) -> u16 {
            let mut m: u16 = 0u16;
            if x < y { m = m | 0x0001u16; }
            if x <= y { m = m | 0x0002u16; }
            if x > y { m = m | 0x0004u16; }
            if x >= y { m = m | 0x0008u16; }
            if x < 0x0400i16 { m = m | 0x0010u16; }
            if x >= 0x0400i16 { m = m | 0x0020u16; }
            if x <= 0x03FFi16 { m = m | 0x0040u16; }
            if x > 0x03FFi16 { m = m | 0x0080u16; }
            if x < -1024i16 { m = m | 0x0100u16; }
            if x >= -1024i16 { m = m | 0x0200u16; }
            if x <= -1025i16 { m = m | 0x0400u16; }
            if x > -1025i16 { m = m | 0x0800u16; }
            if x <= 0x0400i16 { m = m | 0x1000u16; }
            if x > 0x0400i16 { m = m | 0x2000u16; }
            if x < 0i16 { m = m | 0x4000u16; }
            if x >= -1i16 { m = m | 0x8000u16; }
            m
        }

        pub fn bool_masks(x: i16, y: i16) -> u16 {
            let c0: bool = x < y;
            let c1: bool = x <= y;
            let c2: bool = x > y;
            let c3: bool = x >= y;
            let c4: bool = x < 0x0400i16;
            let c5: bool = x >= 0x0400i16;
            let c6: bool = x <= 0x03FFi16;
            let c7: bool = x > 0x03FFi16;
            let c8: bool = x < -1024i16;
            let c9: bool = x >= -1024i16;
            let c10: bool = x <= -1025i16;
            let c11: bool = x > -1025i16;
            let c12: bool = x <= 0x0400i16;
            let c13: bool = x > 0x0400i16;
            let c14: bool = x < 0i16;
            let c15: bool = x >= -1i16;
            (c0 as u16)
                | ((c1 as u16) << 1)
                | ((c2 as u16) << 2)
                | ((c3 as u16) << 3)
                | ((c4 as u16) << 4)
                | ((c5 as u16) << 5)
                | ((c6 as u16) << 6)
                | ((c7 as u16) << 7)
                | ((c8 as u16) << 8)
                | ((c9 as u16) << 9)
                | ((c10 as u16) << 10)
                | ((c11 as u16) << 11)
                | ((c12 as u16) << 12)
                | ((c13 as u16) << 13)
                | ((c14 as u16) << 14)
                | ((c15 as u16) << 15)
        }

        pub fn ref_masks(n: &mut Node, delta: i16) -> u16 {
            let v: u16 = n.f0 +% n.f1;
            let s1: u16 = n.f0 >> 1;
            n.f1 = v +% (delta as u16);
            s1 +% n.f1
        }

        pub fn step_balls() {
            for k in 0u8..6u8 {
                let ki: u16 = k as u16;
                if BALLS[ki].active != 0u8 {
                    let next_x: i16 = BALLS[ki].x +% BALLS[ki].vx;
                    if next_x < -1024i16 || next_x >= 0x0400i16 {
                        BALLS[ki].vx = 0i16 -% BALLS[ki].vx;
                    } else {
                        BALLS[ki].x = next_x;
                    }
                    BALLS[ki].ticks = BALLS[ki].ticks +% 1u8;
                    BALLS[ki].last_idx = k;
                }
            }
        }

        pub fn verify_signed_cmps() -> u16 {
            let xs: [i16; 10] = [
                (-32767i16 -% 1i16), -1025i16, -1024i16, -1023i16, -1i16,
                0i16, 1023i16, 1024i16, 1025i16, 32767i16,
            ];
            let ys: [i16; 7] = [
                (-32767i16 -% 1i16), -30000i16, -1024i16, -1i16, 0i16, 1024i16, 32767i16,
            ];
            let mut errs: u16 = 0u16;
            for i in 0u8..10u8 {
                let x: i16 = xs[i as u16];
                for j in 0u8..7u8 {
                    let y: i16 = ys[j as u16];
                    let bm: u16 = branch_masks(x, y);
                    let lm: u16 = bool_masks(x, y);
                    if bm != lm {
                        errs = errs +% 1u16;
                    }
                    // Verify exact partitioning invariants:
                    if ((bm & 0x0010u16) ^ ((bm & 0x0020u16) >> 1)) != 0x0010u16 { errs = errs +% 1u16; }
                    if ((bm & 0x0040u16) ^ ((bm & 0x0080u16) >> 1)) != 0x0040u16 { errs = errs +% 1u16; }
                    if ((bm & 0x0100u16) ^ ((bm & 0x0200u16) >> 1)) != 0x0100u16 { errs = errs +% 1u16; }
                    if ((bm & 0x0400u16) ^ ((bm & 0x0800u16) >> 1)) != 0x0400u16 { errs = errs +% 1u16; }
                    if ((bm & 0x1000u16) ^ ((bm & 0x2000u16) >> 1)) != 0x1000u16 { errs = errs +% 1u16; }
                    // Verify 0x0400 <= vs < consistency (< 1024 == <= 1023, and 1025 fails <= 0x0400)
                    if (bm & 0x0010u16) != ((bm & 0x0040u16) >> 2) { errs = errs +% 1u16; }
                    if (bm & 0x0100u16) != ((bm & 0x0400u16) >> 2) { errs = errs +% 1u16; }
                }
            }
            // Also spot-check exact bitmasks on edge inputs via runtime reference
            if branch_masks(1024i16, (-32767i16 -% 1i16)) != 0x9AACu16 { errs = errs +% 100u16; }
            if branch_masks(1025i16, 1025i16) != 0xAAAAu16 { errs = errs +% 1000u16; }
            if branch_masks(1023i16, 1024i16) != 0x9A53u16 { errs = errs +% 10000u16; }
            errs
        }

        pub fn main() -> u16 {
            let cmp_errs: u16 = verify_signed_cmps();
            if cmp_errs != 0u16 {
                return cmp_errs;
            }
            // Specific boundary sanity check for overflow + aligned constants:
            if !(branch_masks((-32767i16 -% 1i16), 30000i16) == 0x5553u16) {
                return 0xEE01u16;
            }
            if !(branch_masks(32767i16, -30000i16) == 0xAAACu16) {
                return 0xEE02u16;
            }
            if !(branch_masks(-1025i16, -1024i16) == 0x5553u16) {
                return 0xEE03u16;
            }
            if !(branch_masks(-1024i16, -1024i16) == 0x5A5Au16) {
                return 0xEE04u16;
            }

            let mut nd: Node = Node { f0: 20u16, f1: 7u16 };
            let r_sum: u16 = ref_masks(&mut nd, -3i16);
            // s1 = 10, n.f1 = 27 - 3 = 24, r_sum = 34
            if r_sum != 34u16 || nd.f1 != 24u16 {
                return 0xEE05u16;
            }

            // Initialize active balls at index 0, 2, 5; indices 1, 3, 4 remain inactive.
            BALLS[0].active = 1u8;
            BALLS[0].x = 1020i16;
            BALLS[0].vx = 5i16;   // bounces at >= 0x0400 (1025 >= 1024)
            BALLS[2].active = 1u8;
            BALLS[2].x = -1020i16;
            BALLS[2].vx = -4i16;  // hits -1024 (>= -1024): moves to -1024
            BALLS[5].active = 1u8;
            BALLS[5].x = -1024i16;
            BALLS[5].vx = -1i16;  // bounces at < -1024 (-1025 < -1024)

            step_balls();

            if BALLS[0].vx != -5i16 || BALLS[0].x != 1020i16 || BALLS[0].ticks != 1u8 || BALLS[0].last_idx != 0u8 {
                return 0xEE06u16;
            }
            if BALLS[1].ticks != 0u8 || BALLS[3].ticks != 0u8 || BALLS[4].ticks != 0u8 {
                return 0xEE07u16;
            }
            if BALLS[2].x != -1024i16 || BALLS[2].vx != -4i16 || BALLS[2].ticks != 1u8 || BALLS[2].last_idx != 2u8 {
                return 0xEE08u16;
            }
            if BALLS[5].vx != 1i16 || BALLS[5].x != -1024i16 || BALLS[5].ticks != 1u8 || BALLS[5].last_idx != 5u8 {
                return 0xEE09u16;
            }
            0x4343u16
        }
    "#;

    let prog = aun_front::compile_str(0, src).expect("front");
    let out = aun_z80::compile_z80_with_options(&prog, &[], CompileOptions::release())
        .expect("compile_z80");

    assert!(
        out.asm_listing.contains("_cmp_ovf_"),
        "expected out-of-line _cmp_ovf_ stubs in listing:\n{}",
        out.asm_listing
    );
    assert!(
        out.asm_listing.contains("ld a, (hl)") && out.asm_listing.contains("djnz "),
        "expected HL + djnz scan loop lowering for step_balls:\n{}",
        out.asm_listing
    );

    let mut cpu = Z80::new();
    let mut hal = MockHal::default();
    cpu.ram[out.code_origin as usize..out.code_origin as usize + out.resident_bin.len()]
        .copy_from_slice(&out.resident_bin);
    cpu.sp = 0xF000;
    cpu.write8(0xFFFF, 0x76);
    cpu.push16(0xFFFF);
    cpu.pc = out.code_origin;
    cpu.halted = false;
    while !cpu.halted {
        assert!(cpu.steps < 500_000, "Z80 timeout on #43");
        cpu.step(&mut hal);
    }
    assert_eq!(cpu.sp, 0xF000);
    assert_eq!(
        cpu.hl(),
        0x4343,
        "unexpected result 0x{:04X} (err code or checksum mismatch)",
        cpu.hl()
    );
}

#[test]
fn z80_e2e_issue47_aggregate_by_value_returns() {
    let src = r#"
        struct S1 { a: u8 }
        struct S2 { a: u8, b: u8 }
        struct S3 { a: u8, b: u16 }
        struct S4 { a: u16, b: u16 }
        struct DiskOp {
            drv: u8,
            flags: u8,
            lba: u16,
            count: u16,
        }

        machine Loader {
            seq: u8,
            state Idle;
            state Ready { bias: u16 };
            initial Idle;
            transitions {
                Idle -> Ready;
            }

            on Idle {
                fn next_op(&mut self, lba: u16) -> DiskOp {
                    self.seq = self.seq +% 1u8;
                    DiskOp { drv: 0u8, flags: self.seq, lba: lba, count: 3u16 }
                }
            }
            on Ready {
                fn next_op(&mut self, lba: u16) -> DiskOp {
                    self.seq = self.seq +% 2u8;
                    DiskOp { drv: 1u8, flags: self.seq, lba: lba +% self.bias, count: 5u16 }
                }
            }
        }

        pub fn make_s1(v: u8) -> S1 { S1 { a: v } }
        pub fn make_s2(a: u8, b: u8) -> S2 { S2 { a: a, b: b } }
        pub fn make_s3(a: u8, b: u16) -> S3 { S3 { a: a, b: b } }
        pub fn make_s4(a: u16, b: u16) -> S4 { S4 { a: a, b: b } }

        pub fn make_op(drv: u8, lba: u16) -> DiskOp {
            if drv == 0u8 {
                return DiskOp { drv: 0u8, flags: 0x11u8, lba: lba, count: 1u16 };
            }
            DiskOp { drv: drv, flags: 0x22u8, lba: lba +% 10u16, count: 4u16 }
        }

        fn chain_op(drv: u8, lba: u16) -> DiskOp {
            return make_op(drv, lba);
        }

        pub fn make_op_cond(flag: bool) -> DiskOp {
            if flag {
                make_op(0u8, 0x0100u16)
            } else {
                make_op(2u8, 0x0200u16)
            }
        }

        pub fn make_arr3(x: u8) -> [u8; 3] {
            [x, x +% 1u8, x +% 2u8]
        }

        pub fn make_arr5(x: u8) -> [u8; 5] {
            [x, x +% 1u8, x +% 2u8, x +% 3u8, x +% 4u8]
        }

        pub fn main() -> u16 {
            let s1: S1 = make_s1(0x5Au8);
            if s1.a != 0x5Au8 { return 0xEE01u16; }

            let s2: S2 = make_s2(0x12u8, 0x34u8);
            if s2.a != 0x12u8 || s2.b != 0x34u8 { return 0xEE02u16; }

            let s3: S3 = make_s3(0x77u8, 0xCAFEu16);
            if s3.a != 0x77u8 || s3.b != 0xCAFEu16 { return 0xEE03u16; }

            let s4: S4 = make_s4(0x1122u16, 0x3344u16);
            if s4.a != 0x1122u16 || s4.b != 0x3344u16 { return 0xEE04u16; }

            let op0: DiskOp = make_op(0u8, 0x1234u16);
            if op0.drv != 0u8 || op0.flags != 0x11u8 || op0.lba != 0x1234u16 || op0.count != 1u16 {
                return 0xEE05u16;
            }

            let op1: DiskOp = chain_op(3u8, 0x0100u16);
            if op1.drv != 3u8 || op1.flags != 0x22u8 || op1.lba != 0x010Au16 || op1.count != 4u16 {
                return 0xEE06u16;
            }

            let opc: DiskOp = make_op_cond(false);
            if opc.drv != 2u8 || opc.lba != 0x020Au16 || opc.count != 4u16 {
                return 0xEE07u16;
            }

            if make_op(1u8, 50u16).lba != 60u16 {
                return 0xEE08u16;
            }

            let a3: [u8; 3] = make_arr3(10u8);
            let a5: [u8; 5] = make_arr5(20u8);
            if a3[0] != 10u8 || a3[2] != 12u8 || a5[0] != 20u8 || a5[4] != 24u8 {
                return 0xEE09u16;
            }

            let mut ldr: Loader = Loader::new();
            let m0: DiskOp = ldr.next_op(0x2000u16);
            if m0.drv != 0u8 || m0.flags != 1u8 || m0.lba != 0x2000u16 || m0.count != 3u16 {
                return 0xEE0Au16;
            }
            ldr.goto(Ready { bias: 0x0040u16 });
            let m1: DiskOp = ldr.next_op(0x2000u16);
            if m1.drv != 1u8 || m1.flags != 3u8 || m1.lba != 0x2040u16 || m1.count != 5u16 {
                return 0xEE0Bu16;
            }
            0x4747u16
        }
    "#;

    let prog = aun_front::compile_str(0, src).expect("front");
    aun_ir::verify::verify_program(&prog).expect("verify");

    for opts in [CompileOptions::debug(), CompileOptions::release()] {
        let out = aun_z80::compile_z80_with_options(&prog, &[], opts.clone())
            .expect("compile_z80");
        let mut cpu = Z80::new();
        let mut hal = MockHal::default();
        cpu.ram[out.code_origin as usize..out.code_origin as usize + out.resident_bin.len()]
            .copy_from_slice(&out.resident_bin);
        cpu.sp = 0xF000;
        cpu.write8(0xFFFF, 0x76);
        cpu.push16(0xFFFF);
        cpu.pc = out.code_origin;
        cpu.halted = false;
        while !cpu.halted {
            assert!(cpu.steps < 500_000, "Z80 timeout on #47");
            cpu.step(&mut hal);
        }
        assert_eq!(cpu.sp, 0xF000);
        assert_eq!(
            cpu.hl(),
            0x4747,
            "unexpected result 0x{:04X} (err code mismatch)",
            cpu.hl()
        );

        let (sdas, _, _) = compile_z80_sdasz80("issue47", &prog, opts)
            .expect("compile_z80_sdasz80");
        assert!(sdas.contains("_make_op::"), "missing _make_op:: in sdasz80:\n{sdas}");
        assert!(sdas.contains("_make_s4::"), "missing _make_s4:: in sdasz80:\n{sdas}");
    }
}

#[test]
fn z80_e2e_issue48_countdown_early_exit_iv_rewrite() {
    let src = r#"
        struct Slot { active: u8, val: u8 }
        static mut SLOTS3: [Slot; 3] = [Slot { active: 0u8, val: 0u8 }; 3];
        static mut GUARD3: [u8; 4] = [0xA5u8; 4];
        static mut FOUND: u8 = 0xFFu8;

        static mut SLOTS6: [Slot; 6] = [Slot { active: 0u8, val: 0u8 }; 6];
        static mut CAPS: [u8; 4] = [0u8; 4];
        static mut CAPS_OUT: [u8; 4] = [0u8; 4];
        static mut CAPS_GUARD: [u8; 4] = [0x5Au8; 4];

        pub fn find_return(v: u8) {
            for i in 0u8..3u8 {
                let ii: u16 = i as u16;
                if SLOTS3[ii].active == 0u8 {
                    SLOTS3[ii].active = 1u8;
                    SLOTS3[ii].val = v;
                    FOUND = i;
                    return;
                }
            }
        }

        pub fn find_break() -> u8 {
            let mut found: u8 = 0xFFu8;
            for i in 0u8..6u8 {
                let ii: u16 = i as u16;
                if SLOTS6[ii].active == 0u8 {
                    found = i;
                    break;
                }
            }
            found
        }

        pub fn collect_capsule() {
            for i in 0u8..4u8 {
                let ii: u16 = i as u16;
                if CAPS[ii] == 0u8 {
                    CAPS[ii] = 1u8;
                    CAPS_OUT[ii] = i +% 10u8;
                    break;
                }
            }
        }

        pub fn find_u16_return(n: u16, want: u16) -> u16 {
            for i in 0u16..n {
                if i == want {
                    return i;
                }
            }
            0xFFFFu16
        }

        pub fn find_u16_break(n: u16, want: u16) -> u16 {
            let mut out: u16 = 0xFFFFu16;
            for i in 0u16..n {
                if i == want {
                    out = i +% 100u16;
                    break;
                }
            }
            out
        }

        pub fn reuse_iv_after_break(stop: u8) -> u8 {
            let mut i: u8 = 0u8;
            while i < 3u8 {
                if i == stop {
                    break;
                }
                i = i +% 1u8;
            }
            while i < 5u8 {
                i = i +% 1u8;
            }
            i
        }

        pub fn main() -> u16 {
            // 1. Sequential find_return: fills slots 0, 1, 2 in order without touching GUARD3.
            find_return(0x11u8);
            if FOUND != 0u8 || SLOTS3[0].active != 1u8 || SLOTS3[0].val != 0x11u8 || SLOTS3[1].active != 0u8 {
                return 0xEE01u16;
            }
            find_return(0x22u8);
            if FOUND != 1u8 || SLOTS3[1].active != 1u8 || SLOTS3[1].val != 0x22u8 || SLOTS3[2].active != 0u8 {
                return 0xEE02u16;
            }
            find_return(0x33u8);
            if FOUND != 2u8 || SLOTS3[2].active != 1u8 || SLOTS3[2].val != 0x33u8 {
                return 0xEE03u16;
            }
            FOUND = 0xFFu8;
            find_return(0x44u8);
            if FOUND != 0xFFu8 || GUARD3[0] != 0xA5u8 || GUARD3[3] != 0xA5u8 {
                return 0xEE04u16;
            }

            // 2. find_break: returns first free index 0..5 or 0xFF.
            if find_break() != 0u8 { return 0xEE05u16; }
            SLOTS6[0].active = 1u8;
            SLOTS6[1].active = 1u8;
            if find_break() != 2u8 { return 0xEE06u16; }
            SLOTS6[2].active = 1u8;
            SLOTS6[3].active = 1u8;
            SLOTS6[4].active = 1u8;
            if find_break() != 5u8 { return 0xEE07u16; }
            SLOTS6[5].active = 1u8;
            if find_break() != 0xFFu8 { return 0xEE08u16; }

            // 3. collect_capsule: writes ascending index into CAPS_OUT[ii] inside break arm.
            collect_capsule();
            collect_capsule();
            if CAPS_OUT[0] != 10u8 || CAPS_OUT[1] != 11u8 || CAPS_OUT[2] != 0u8 || CAPS_GUARD[0] != 0x5Au8 {
                return 0xEE09u16;
            }

            // 4. u16 early return/break & post-loop i reuse.
            if find_u16_return(5u16, 1u16) != 1u16 || find_u16_return(5u16, 9u16) != 0xFFFFu16 {
                return 0xEE0Au16;
            }
            if find_u16_break(6u16, 2u16) != 102u16 || find_u16_break(6u16, 8u16) != 0xFFFFu16 {
                return 0xEE0Bu16;
            }
            if reuse_iv_after_break(1u8) != 5u8 || reuse_iv_after_break(9u8) != 5u8 {
                return 0xEE0Cu16;
            }
            0x4848u16
        }
    "#;

    let prog = aun_front::compile_str(0, src).expect("front");
    aun_ir::verify::verify_program(&prog).expect("verify");

    for (is_release, opts) in [(false, CompileOptions::debug()), (true, CompileOptions::release())] {
        let out = aun_z80::compile_z80_with_options(&prog, &[], opts).expect("compile_z80");
        if is_release {
            assert!(
                out.asm_listing.contains("djnz "),
                "expected scan loops with early exit to still use djnz in release:\n{}",
                out.asm_listing
            );
        }
        let mut cpu = Z80::new();
        let mut hal = MockHal::default();
        cpu.ram[out.code_origin as usize..out.code_origin as usize + out.resident_bin.len()]
            .copy_from_slice(&out.resident_bin);
        cpu.sp = 0xF000;
        cpu.write8(0xFFFF, 0x76);
        cpu.push16(0xFFFF);
        cpu.pc = out.code_origin;
        cpu.halted = false;
        while !cpu.halted {
            assert!(cpu.steps < 500_000, "Z80 timeout on #48");
            cpu.step(&mut hal);
        }
        assert_eq!(cpu.sp, 0xF000);
        assert_eq!(
            cpu.hl(),
            0x4848,
            "unexpected result 0x{:04X} (is_release={is_release})",
            cpu.hl()
        );
    }
}

#[test]
fn z80_e2e_sat16_and_fixed_cast_and_entity_pool() {
    let ep_src = include_str!("../../../bench/suite/entity_pool.aun");
    let driver = r#"
        pub fn sat_u16_add(a: u16, b: u16) -> u16 { a +| b }
        pub fn sat_u16_sub(a: u16, b: u16) -> u16 { a -| b }
        pub fn sat_i16_add(a: i16, b: i16) -> i16 { a +| b }
        pub fn sat_i16_sub(a: i16, b: i16) -> i16 { a -| b }
        pub fn sat_zero_sub(b: i16) -> i16 { 0i16 -| b }
        pub fn sat_small_const(a: i16) -> i16 { a +| 1i16 }

        pub fn fx_mul_half(x: q8_8) -> q8_8 { x *% 0.50q8_8 }
        pub fn fx_mul_qtr(x: q8_8) -> q8_8 { x *% 0.25q8_8 }
        pub fn fx_mul_rt(a: q8_8, b: q8_8) -> q8_8 { a *% b }
        pub fn fx_mul_q10_6(a: q10_6, b: q10_6) -> q10_6 { a *% b }
        pub fn fx_div_q8_8(a: q8_8, b: q8_8) -> q8_8 { a / b }
        pub fn fx_div_q10_6(a: q10_6, b: q10_6) -> q10_6 { a / b }
        pub fn fx_mul_u4_4(a: fixed<u8, 4>, b: fixed<u8, 4>) -> fixed<u8, 4> { a *% b }
        pub fn fx_div_u4_4(a: fixed<u8, 4>, b: fixed<u8, 4>) -> fixed<u8, 4> { a / b }

        pub fn main() -> u16 {
            // 16-bit saturating unsigned +| / -|
            if sat_u16_add(60000u16, 10000u16) != 0xFFFFu16 { return 0xE001u16; }
            if sat_u16_add(100u16, 200u16) != 300u16 { return 0xE002u16; }
            if sat_u16_sub(10u16, 50u16) != 0u16 { return 0xE003u16; }
            if sat_u16_sub(500u16, 123u16) != 377u16 { return 0xE004u16; }

            // 16-bit saturating signed +| / -| (including 0 -| i16::MIN and small const)
            let i16_min: i16 = -32767i16 -% 1i16;
            if sat_i16_add(30000i16, 5000i16) != 32767i16 { return 0xE005u16; }
            if sat_i16_add(-30000i16, -5000i16) != i16_min { return 0xE006u16; }
            if sat_i16_add(1200i16, -500i16) != 700i16 { return 0xE007u16; }
            if sat_i16_sub(30000i16, -5000i16) != 32767i16 { return 0xE008u16; }
            if sat_i16_sub(-30000i16, 5000i16) != i16_min { return 0xE009u16; }
            if sat_zero_sub(i16_min) != 32767i16 { return 0xE00Au16; }
            if sat_small_const(32767i16) != 32767i16 { return 0xE00Bu16; }

            // Fixed-point Cast scaling checks
            let q_u: q8_8 = 5u8 as q8_8;
            let q_i: q10_6 = (-3i8) as q10_6;
            let q_w: q8_8 = q_i as q8_8;
            let q_n: q10_6 = q_u as q10_6;
            let q_int: i16 = (-2.25q8_8) as i16;
            let q_i8: i8 = (-2.25q8_8) as i8;
            if q_u.raw() != 0x0500i16 || q_i.raw() != -192i16 { return 0xE00Cu16; }
            if q_w.raw() != -0x0300i16 || q_n.raw() != 320i16 { return 0xE00Du16; }
            if q_int != -3i16 || q_i8 != -3i8 { return 0xE00Eu16; }

            // Fixed-point Mul & Div scaling checks (pow2 const, runtime signed/unsigned, 8/16-bit)
            if fx_mul_half(-3.00q8_8) != -1.50q8_8 { return 0xE013u16; }
            if fx_mul_qtr(12.00q8_8) != 3.00q8_8 { return 0xE014u16; }
            if fx_mul_rt(1.50q8_8, -2.25q8_8) != -3.375q8_8 { return 0xE015u16; }
            if fx_mul_q10_6(-4.50q10_6, 3.00q10_6) != -13.50q10_6 { return 0xE016u16; }
            if fx_div_q8_8(3.00q8_8, 1.50q8_8) != 2.00q8_8 { return 0xE017u16; }
            if fx_div_q8_8(-3.00q8_8, 2.00q8_8) != -1.50q8_8 { return 0xE018u16; }
            if fx_div_q10_6(-9.00q10_6, -1.50q10_6) != 6.00q10_6 { return 0xE019u16; }
            let u4_25: fixed<u8, 4> = 2.5;
            let u4_15: fixed<u8, 4> = 1.5;
            let u4_30: fixed<u8, 4> = 3.0;
            if fx_mul_u4_4(u4_25, u4_15).raw() != 0x3Cu8 { return 0xE01Au16; }
            if fx_div_u4_4(u4_30, u4_15).raw() != 0x20u8 { return 0xE01Bu16; }

            // Execute entity_pool.aun simulation directly on Z80
            entity_pool_reset();
            if entity_pool_spawn(0u8, 20u8, 100u8, 4u8) != 1u8 || LIVE_MASK != 1u8 {
                return 0xE00Fu16;
            }
            let y1: u16 = entity_pool_step_all(0i8);
            if y1 != 100u16 { return 0xE010u16; }
            let y2: u16 = entity_pool_step_all(0i8);
            if y2 != 98u16 { return 0xE011u16; }
            for _t in 0u8..5u8 {
                let _y: u16 = entity_pool_step_all(0i8);
            }
            if LIVE_MASK != 0u8 { return 0xE012u16; }
            0x5A5Au16
        }
    "#;
    let combined = format!("{ep_src}\n{driver}");
    let prog = aun_front::compile_str(0, &combined).expect("front");
    aun_ir::verify::verify_program(&prog).expect("verify");

    for (is_release, opts) in [(false, CompileOptions::debug()), (true, CompileOptions::release())] {
        let out = aun_z80::compile_z80_with_options(&prog, &[], opts).expect("compile_z80");
        let mut cpu = Z80::new();
        let mut hal = MockHal::default();
        cpu.ram[out.code_origin as usize..out.code_origin as usize + out.resident_bin.len()]
            .copy_from_slice(&out.resident_bin);
        cpu.sp = 0xF000;
        cpu.write8(0xFFFF, 0x76);
        cpu.push16(0xFFFF);
        cpu.pc = out.code_origin;
        cpu.halted = false;
        while !cpu.halted {
            assert!(cpu.steps < 500_000, "Z80 timeout on sat16/fixed/entity_pool");
            cpu.step(&mut hal);
        }
        assert_eq!(cpu.sp, 0xF000);
        assert_eq!(
            cpu.hl(),
            0x5A5A,
            "unexpected result 0x{:04X} (is_release={is_release})",
            cpu.hl()
        );
    }
}
