//! Typed Z80 Low-level Intermediate Representation (LIR).
//!
//! Represents Z80 instructions, directives, labels, and data tables cleanly as
//! structured Rust enums instead of raw text strings, providing:
//! - Exact per-instruction byte-length calculation (`LirInst::byte_len`).
//! - Sound fixed-point `JP -> JR` branch relaxation (`relax_jp_to_jr`).
//! - Zilog (`format_zilog`) and SDASZ80 (`format_sdasz80`) printers from one IR.

use std::collections::HashMap;
use std::fmt::Write;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reg8 {
    A,
    B,
    C,
    D,
    E,
    H,
    L,
}

impl Reg8 {
    pub fn as_str(self) -> &'static str {
        match self {
            Reg8::A => "a",
            Reg8::B => "b",
            Reg8::C => "c",
            Reg8::D => "d",
            Reg8::E => "e",
            Reg8::H => "h",
            Reg8::L => "l",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "a" => Some(Reg8::A),
            "b" => Some(Reg8::B),
            "c" => Some(Reg8::C),
            "d" => Some(Reg8::D),
            "e" => Some(Reg8::E),
            "h" => Some(Reg8::H),
            "l" => Some(Reg8::L),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reg16 {
    BC,
    DE,
    HL,
    SP,
    IX,
    IY,
    AF,
}

impl Reg16 {
    pub fn as_str(self) -> &'static str {
        match self {
            Reg16::BC => "bc",
            Reg16::DE => "de",
            Reg16::HL => "hl",
            Reg16::SP => "sp",
            Reg16::IX => "ix",
            Reg16::IY => "iy",
            Reg16::AF => "af",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "bc" => Some(Reg16::BC),
            "de" => Some(Reg16::DE),
            "hl" => Some(Reg16::HL),
            "sp" => Some(Reg16::SP),
            "ix" => Some(Reg16::IX),
            "iy" => Some(Reg16::IY),
            "af" => Some(Reg16::AF),
            _ => None,
        }
    }

    pub fn is_index(self) -> bool {
        matches!(self, Reg16::IX | Reg16::IY)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexReg {
    IX,
    IY,
}

impl IndexReg {
    pub fn as_str(self) -> &'static str {
        match self {
            IndexReg::IX => "ix",
            IndexReg::IY => "iy",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Cond {
    Z,
    NZ,
    C,
    NC,
    PO,
    PE,
    P,
    M,
}

impl Cond {
    pub fn as_str(self) -> &'static str {
        match self {
            Cond::Z => "z",
            Cond::NZ => "nz",
            Cond::C => "c",
            Cond::NC => "nc",
            Cond::PO => "po",
            Cond::PE => "pe",
            Cond::P => "p",
            Cond::M => "m",
        }
    }

    pub fn is_jr_compatible(self) -> bool {
        matches!(self, Cond::Z | Cond::NZ | Cond::C | Cond::NC)
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "z" => Some(Cond::Z),
            "nz" => Some(Cond::NZ),
            "c" => Some(Cond::C),
            "nc" => Some(Cond::NC),
            "po" => Some(Cond::PO),
            "pe" => Some(Cond::PE),
            "p" => Some(Cond::P),
            "m" => Some(Cond::M),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SymExpr {
    Addr(u16),
    Sym(String),
    SymOffset(String, i16),
}

impl SymExpr {
    pub fn format_zilog(&self) -> String {
        match self {
            SymExpr::Addr(a) => format!("0x{a:04X}"),
            SymExpr::Sym(s) => s.clone(),
            SymExpr::SymOffset(s, off) => {
                if *off >= 0 {
                    format!("{s} + {off}")
                } else {
                    format!("{s} - {}", -(*off as i32))
                }
            }
        }
    }

    pub fn format_sdasz80_mem(&self, ram_base: u16, ram_end: u16) -> String {
        match self {
            SymExpr::Addr(a) if ram_end > ram_base && *a >= ram_base && *a < ram_end => {
                let off = *a - ram_base;
                if off == 0 {
                    "(__aun_ram)".to_string()
                } else {
                    format!("(__aun_ram + {off})")
                }
            }
            SymExpr::Addr(a) => format!("(0x{a:04X})"),
            SymExpr::Sym(s) => format!("({s})"),
            SymExpr::SymOffset(s, off) => {
                if *off >= 0 {
                    format!("({s} + {off})")
                } else {
                    format!("({s} - {})", -(*off as i32))
                }
            }
        }
    }

    pub fn format_sdasz80_imm(&self, ram_base: u16, ram_end: u16) -> String {
        match self {
            SymExpr::Addr(a) if ram_end > ram_base && *a >= ram_base && *a < ram_end => {
                let off = *a - ram_base;
                if off == 0 {
                    "#__aun_ram".to_string()
                } else {
                    format!("#(__aun_ram + {off})")
                }
            }
            SymExpr::Addr(a) => format!("#0x{a:04X}"),
            SymExpr::Sym(s) => format!("#{s}"),
            SymExpr::SymOffset(s, off) => {
                if *off >= 0 {
                    format!("#({s} + {off})")
                } else {
                    format!("#({s} - {})", -(*off as i32))
                }
            }
        }
    }

    pub fn parse(s: &str) -> Self {
        let t = s.trim();
        if let Some(v) = parse_u16(t) {
            return SymExpr::Addr(v);
        }
        if let Some((lhs, rhs)) = t.split_once('+')
            && !lhs.trim().is_empty()
            && let Some(off) = parse_i32_lit(rhs.trim())
            && let Ok(off16) = i16::try_from(off)
        {
            return SymExpr::SymOffset(lhs.trim().to_string(), off16);
        }
        if let Some((lhs, rhs)) = t.split_once(" - ")
            && !lhs.trim().is_empty()
            && let Some(off) = parse_i32_lit(rhs.trim())
            && let Ok(off16) = i16::try_from(-off)
        {
            return SymExpr::SymOffset(lhs.trim().to_string(), off16);
        }
        if let Some((lhs, rhs)) = t.rsplit_once('-')
            && !lhs.trim().is_empty()
            && let Some(off) = parse_i32_lit(rhs.trim())
            && let Ok(off16) = i16::try_from(-off)
        {
            return SymExpr::SymOffset(lhs.trim().to_string(), off16);
        }
        SymExpr::Sym(t.to_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op8 {
    Reg(Reg8),
    Imm(u8),
    HlInd,
    Indexed(IndexReg, i8),
}

impl Op8 {
    pub fn format_zilog(&self) -> String {
        match self {
            Op8::Reg(r) => r.as_str().to_string(),
            Op8::Imm(b) => format!("0x{b:02X}"),
            Op8::HlInd => "(hl)".to_string(),
            Op8::Indexed(reg, d) => format_indexed_zilog(*reg, *d),
        }
    }

    pub fn format_sdasz80(&self) -> String {
        match self {
            Op8::Reg(r) => r.as_str().to_string(),
            Op8::Imm(b) => format!("#0x{b:02X}"),
            Op8::HlInd => "(hl)".to_string(),
            Op8::Indexed(reg, d) => format!("{d}({})", reg.as_str()),
        }
    }

    fn extra_bytes(&self) -> usize {
        match self {
            Op8::Reg(_) | Op8::HlInd => 0,
            Op8::Imm(_) => 1,
            Op8::Indexed(_, _) => 2,
        }
    }
}

fn format_indexed_zilog(reg: IndexReg, disp: i8) -> String {
    if disp == 0 {
        format!("({} + 0)", reg.as_str())
    } else if disp > 0 {
        format!("({} + {disp})", reg.as_str())
    } else {
        format!("({} - {})", reg.as_str(), -(disp as i16))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AluOp8 {
    Add,
    Adc,
    Sub,
    Sbc,
    And,
    Xor,
    Or,
    Cp,
}

impl AluOp8 {
    pub fn as_str(self) -> &'static str {
        match self {
            AluOp8::Add => "add",
            AluOp8::Adc => "adc",
            AluOp8::Sub => "sub",
            AluOp8::Sbc => "sbc",
            AluOp8::And => "and",
            AluOp8::Xor => "xor",
            AluOp8::Or => "or",
            AluOp8::Cp => "cp",
        }
    }

    pub fn uses_a_prefix(self) -> bool {
        matches!(self, AluOp8::Add | AluOp8::Adc | AluOp8::Sbc)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShiftOp {
    Sla,
    Sra,
    Srl,
    Rl,
    Rr,
}

impl ShiftOp {
    pub fn as_str(self) -> &'static str {
        match self {
            ShiftOp::Sla => "sla",
            ShiftOp::Sra => "sra",
            ShiftOp::Srl => "srl",
            ShiftOp::Rl => "rl",
            ShiftOp::Rr => "rr",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LirInst {
    Ld8 { dst: Reg8, src: Op8 },
    LdAInd16(Reg16),
    StInd16A(Reg16),
    LdMem8 { dst: Reg8, addr: SymExpr },
    StMem8 { addr: SymExpr, src: Reg8 },
    LdInd8 { reg: IndexReg, disp: i8, src: Op8 },
    StHlInd8 { src: Op8 },
    Ld16Imm { dst: Reg16, val: SymExpr },
    Ld16Mem { dst: Reg16, addr: SymExpr },
    St16Mem { addr: SymExpr, src: Reg16 },
    LdSp(Reg16),
    Alu8 { op: AluOp8, src: Op8 },
    Add16 { dst: Reg16, src: Reg16 },
    SbcHl { src: Reg16 },
    AdcHl { src: Reg16 },
    Inc8(Op8),
    Dec8(Op8),
    Inc16(Reg16),
    Dec16(Reg16),
    Shift8 { op: ShiftOp, target: Op8 },
    Bit { bit: u8, target: Op8 },
    Set { bit: u8, target: Op8 },
    Res { bit: u8, target: Op8 },
    Neg,
    Cpl,
    Rla,
    Rra,
    Rlca,
    Rrca,
    Ccf,
    Scf,
    Nop,
    Di,
    Ei,
    Halt,
    Ldir,
    Lddr,
    ExDeHl,
    Push(Reg16),
    Pop(Reg16),
    Jp { cond: Option<Cond>, target: String },
    JpInd(Reg16),
    Jr { cond: Option<Cond>, target: String },
    Djnz(String),
    Call { cond: Option<Cond>, target: String },
    Ret(Option<Cond>),
    Reti,
    Retn,
    InAImm(u8),
    InAC,
    OutImmA(u8),
    OutCA,
    Raw(String),
}

impl LirInst {
    /// Exact Z80 encoded byte length (with a 4-byte conservative bound for `Raw`).
    pub fn byte_len(&self) -> usize {
        match self {
            LirInst::Ld8 { src, .. } => 1 + src.extra_bytes(),
            LirInst::LdAInd16(_) | LirInst::StInd16A(_) => 1,
            LirInst::LdMem8 { .. } | LirInst::StMem8 { .. } => 3,
            LirInst::LdInd8 { src, .. } => 3 + usize::from(matches!(src, Op8::Imm(_))),
            LirInst::StHlInd8 { src } => 1 + usize::from(matches!(src, Op8::Imm(_))),
            LirInst::Ld16Imm { dst, .. } => if dst.is_index() { 4 } else { 3 },
            LirInst::Ld16Mem { dst, .. } | LirInst::St16Mem { src: dst, .. } => {
                if *dst == Reg16::HL { 3 } else { 4 }
            }
            LirInst::LdSp(r) => if r.is_index() { 2 } else { 1 },
            LirInst::Alu8 { src, .. } => 1 + src.extra_bytes(),
            LirInst::Add16 { dst, .. } => if dst.is_index() { 2 } else { 1 },
            LirInst::SbcHl { .. } | LirInst::AdcHl { .. } => 2,
            LirInst::Inc8(op) | LirInst::Dec8(op) => 1 + op.extra_bytes(),
            LirInst::Inc16(r) | LirInst::Dec16(r) => if r.is_index() { 2 } else { 1 },
            LirInst::Shift8 { target, .. }
            | LirInst::Bit { target, .. }
            | LirInst::Set { target, .. }
            | LirInst::Res { target, .. } => {
                if matches!(target, Op8::Indexed(_, _)) { 4 } else { 2 }
            }
            LirInst::Push(r) | LirInst::Pop(r) => if r.is_index() { 2 } else { 1 },
            LirInst::Jp { .. } | LirInst::Call { .. } => 3,
            LirInst::JpInd(r) => if r.is_index() { 2 } else { 1 },
            LirInst::Ret(_) => 1,
            LirInst::Jr { .. } | LirInst::Djnz(_) => 2,
            LirInst::Reti | LirInst::Retn => 2,
            LirInst::Neg
            | LirInst::Ldir
            | LirInst::Lddr
            | LirInst::InAImm(_)
            | LirInst::InAC
            | LirInst::OutImmA(_)
            | LirInst::OutCA => 2,
            LirInst::Cpl
            | LirInst::Rla
            | LirInst::Rra
            | LirInst::Rlca
            | LirInst::Rrca
            | LirInst::Ccf
            | LirInst::Scf
            | LirInst::Nop
            | LirInst::Di
            | LirInst::Ei
            | LirInst::Halt
            | LirInst::ExDeHl => 1,
            LirInst::Raw(_) => 4,
        }
    }

    pub fn format_zilog(&self) -> String {
        match self {
            LirInst::Ld8 { dst, src } => format!("ld {}, {}", dst.as_str(), src.format_zilog()),
            LirInst::LdAInd16(r) => format!("ld a, ({})", r.as_str()),
            LirInst::StInd16A(r) => format!("ld ({}), a", r.as_str()),
            LirInst::LdMem8 { dst, addr } => {
                format!("ld {}, ({})", dst.as_str(), addr.format_zilog())
            }
            LirInst::StMem8 { addr, src } => {
                format!("ld ({}), {}", addr.format_zilog(), src.as_str())
            }
            LirInst::LdInd8 { reg, disp, src } => {
                format!("ld {}, {}", format_indexed_zilog(*reg, *disp), src.format_zilog())
            }
            LirInst::StHlInd8 { src } => format!("ld (hl), {}", src.format_zilog()),
            LirInst::Ld16Imm { dst, val } => {
                format!("ld {}, {}", dst.as_str(), val.format_zilog())
            }
            LirInst::Ld16Mem { dst, addr } => {
                format!("ld {}, ({})", dst.as_str(), addr.format_zilog())
            }
            LirInst::St16Mem { addr, src } => {
                format!("ld ({}), {}", addr.format_zilog(), src.as_str())
            }
            LirInst::LdSp(r) => format!("ld sp, {}", r.as_str()),
            LirInst::Alu8 { op, src } => {
                if op.uses_a_prefix() {
                    format!("{} a, {}", op.as_str(), src.format_zilog())
                } else {
                    format!("{} {}", op.as_str(), src.format_zilog())
                }
            }
            LirInst::Add16 { dst, src } => format!("add {}, {}", dst.as_str(), src.as_str()),
            LirInst::SbcHl { src } => format!("sbc hl, {}", src.as_str()),
            LirInst::AdcHl { src } => format!("adc hl, {}", src.as_str()),
            LirInst::Inc8(op) => format!("inc {}", op.format_zilog()),
            LirInst::Dec8(op) => format!("dec {}", op.format_zilog()),
            LirInst::Inc16(r) => format!("inc {}", r.as_str()),
            LirInst::Dec16(r) => format!("dec {}", r.as_str()),
            LirInst::Shift8 { op, target } => format!("{} {}", op.as_str(), target.format_zilog()),
            LirInst::Bit { bit, target } => format!("bit {bit}, {}", target.format_zilog()),
            LirInst::Set { bit, target } => format!("set {bit}, {}", target.format_zilog()),
            LirInst::Res { bit, target } => format!("res {bit}, {}", target.format_zilog()),
            LirInst::Neg => "neg".to_string(),
            LirInst::Cpl => "cpl".to_string(),
            LirInst::Rla => "rla".to_string(),
            LirInst::Rra => "rra".to_string(),
            LirInst::Rlca => "rlca".to_string(),
            LirInst::Rrca => "rrca".to_string(),
            LirInst::Ccf => "ccf".to_string(),
            LirInst::Scf => "scf".to_string(),
            LirInst::Nop => "nop".to_string(),
            LirInst::Di => "di".to_string(),
            LirInst::Ei => "ei".to_string(),
            LirInst::Halt => "halt".to_string(),
            LirInst::Ldir => "ldir".to_string(),
            LirInst::Lddr => "lddr".to_string(),
            LirInst::ExDeHl => "ex de, hl".to_string(),
            LirInst::Push(r) => format!("push {}", r.as_str()),
            LirInst::Pop(r) => format!("pop {}", r.as_str()),
            LirInst::Jp { cond: None, target } => format!("jp {target}"),
            LirInst::Jp {
                cond: Some(cc),
                target,
            } => format!("jp {}, {target}", cc.as_str()),
            LirInst::JpInd(r) => format!("jp ({})", r.as_str()),
            LirInst::Jr { cond: None, target } => format!("jr {target}"),
            LirInst::Jr {
                cond: Some(cc),
                target,
            } => format!("jr {}, {target}", cc.as_str()),
            LirInst::Djnz(target) => format!("djnz {target}"),
            LirInst::Call { cond: None, target } => format!("call {target}"),
            LirInst::Call {
                cond: Some(cc),
                target,
            } => format!("call {}, {target}", cc.as_str()),
            LirInst::Ret(None) => "ret".to_string(),
            LirInst::Ret(Some(cc)) => format!("ret {}", cc.as_str()),
            LirInst::Reti => "reti".to_string(),
            LirInst::Retn => "retn".to_string(),
            LirInst::InAImm(p) => format!("in a, (0x{p:02X})"),
            LirInst::InAC => "in a, (c)".to_string(),
            LirInst::OutImmA(p) => format!("out (0x{p:02X}), a"),
            LirInst::OutCA => "out (c), a".to_string(),
            LirInst::Raw(s) => s.clone(),
        }
    }

    pub fn format_sdasz80(&self, ram_base: u16, ram_end: u16) -> String {
        match self {
            LirInst::Ld8 { dst, src } => format!("ld {}, {}", dst.as_str(), src.format_sdasz80()),
            LirInst::LdMem8 { dst, addr } => format!(
                "ld {}, {}",
                dst.as_str(),
                addr.format_sdasz80_mem(ram_base, ram_end)
            ),
            LirInst::StMem8 { addr, src } => format!(
                "ld {}, {}",
                addr.format_sdasz80_mem(ram_base, ram_end),
                src.as_str()
            ),
            LirInst::LdInd8 { reg, disp, src } => {
                format!("ld {disp}({}), {}", reg.as_str(), src.format_sdasz80())
            }
            LirInst::StHlInd8 { src } => format!("ld (hl), {}", src.format_sdasz80()),
            LirInst::Ld16Imm { dst, val } => format!(
                "ld {}, {}",
                dst.as_str(),
                val.format_sdasz80_imm(ram_base, ram_end)
            ),
            LirInst::Ld16Mem { dst, addr } => format!(
                "ld {}, {}",
                dst.as_str(),
                addr.format_sdasz80_mem(ram_base, ram_end)
            ),
            LirInst::St16Mem { addr, src } => format!(
                "ld {}, {}",
                addr.format_sdasz80_mem(ram_base, ram_end),
                src.as_str()
            ),
            LirInst::Alu8 { op, src } => {
                if op.uses_a_prefix() {
                    format!("{} a, {}", op.as_str(), src.format_sdasz80())
                } else {
                    format!("{} {}", op.as_str(), src.format_sdasz80())
                }
            }
            LirInst::Inc8(op) => format!("inc {}", op.format_sdasz80()),
            LirInst::Dec8(op) => format!("dec {}", op.format_sdasz80()),
            LirInst::Shift8 { op, target } => {
                format!("{} {}", op.as_str(), target.format_sdasz80())
            }
            LirInst::Bit { bit, target } => format!("bit {bit}, {}", target.format_sdasz80()),
            LirInst::Set { bit, target } => format!("set {bit}, {}", target.format_sdasz80()),
            LirInst::Res { bit, target } => format!("res {bit}, {}", target.format_sdasz80()),
            other => other.format_zilog(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LirItem {
    Blank,
    Comment(String),
    Org(u16),
    Equ(String, u16),
    Label(String),
    DataBytes(Vec<u8>),
    DataWords(Vec<SymExpr>),
    Inst(LirInst),
}

impl LirItem {
    pub fn byte_len(&self) -> usize {
        match self {
            LirItem::Blank
            | LirItem::Comment(_)
            | LirItem::Org(_)
            | LirItem::Equ(_, _)
            | LirItem::Label(_) => 0,
            LirItem::DataBytes(v) => v.len(),
            LirItem::DataWords(v) => v.len() * 2,
            LirItem::Inst(inst) => inst.byte_len(),
        }
    }
}

/// Formats `items` back into canonical Zilog assembly syntax.
pub fn format_zilog_lir(items: &[LirItem]) -> String {
    let mut out = String::new();
    for item in items {
        match item {
            LirItem::Blank => out.push('\n'),
            LirItem::Comment(c) => writeln!(out, "{c}").unwrap(),
            LirItem::Org(addr) => writeln!(out, ".org 0x{addr:04X}").unwrap(),
            LirItem::Equ(name, addr) => writeln!(out, "{name} = 0x{addr:04X}").unwrap(),
            LirItem::Label(lbl) => writeln!(out, "{lbl}:").unwrap(),
            LirItem::DataBytes(bytes) => {
                let strs: Vec<String> = bytes.iter().map(|b| format!("0x{b:02X}")).collect();
                writeln!(out, "    .db {}", strs.join(", ")).unwrap();
            }
            LirItem::DataWords(words) => {
                let strs: Vec<String> = words.iter().map(SymExpr::format_zilog).collect();
                writeln!(out, "    .dw {}", strs.join(", ")).unwrap();
            }
            LirItem::Inst(inst) => writeln!(out, "    {}", inst.format_zilog()).unwrap(),
        }
    }
    out
}

/// Iteratively relaxes short intra-function `_bb<digits>` `jp [cc,] target` into
/// 2-byte `jr [cc,] target` using exact per-instruction byte lengths until convergence.
pub fn relax_jp_to_jr(items: &mut [LirItem]) {
    loop {
        let mut label_pos: HashMap<String, (usize, usize)> = HashMap::new(); // lbl -> (item_idx, byte_pos)
        let mut item_byte_pos: Vec<usize> = Vec::with_capacity(items.len());
        let mut raw_prefix: Vec<usize> = Vec::with_capacity(items.len() + 1);
        let mut byte_pos = 0usize;
        let mut raw_count = 0usize;
        raw_prefix.push(0);

        for (idx, item) in items.iter().enumerate() {
            item_byte_pos.push(byte_pos);
            if let LirItem::Label(lbl) = item {
                label_pos.insert(lbl.clone(), (idx, byte_pos));
            }
            if matches!(item, LirItem::Inst(LirInst::Raw(_))) {
                raw_count += 1;
            }
            raw_prefix.push(raw_count);
            byte_pos += item.byte_len();
        }

        let mut changed = false;
        for (idx, item) in items.iter_mut().enumerate() {
            let LirItem::Inst(inst) = item else {
                continue;
            };
            let LirInst::Jp { cond, target } = inst else {
                continue;
            };
            if !cond.is_none_or(Cond::is_jr_compatible) || !is_bb_label(target) {
                continue;
            }
            let Some(&(t_idx, t_pos)) = label_pos.get(target.as_str()) else {
                continue;
            };
            let lo = idx.min(t_idx);
            let hi = idx.max(t_idx);
            if raw_prefix[hi + 1] != raw_prefix[lo] {
                continue;
            }
            let disp = (t_pos as isize) - ((item_byte_pos[idx] + 2) as isize);
            if (-126..=126).contains(&disp) {
                *inst = LirInst::Jr {
                    cond: *cond,
                    target: target.clone(),
                };
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

fn is_bb_label(lbl: &str) -> bool {
    lbl.rsplit_once("_bb")
        .is_some_and(|(_, d)| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
}

/// Parses Aun-emitted Zilog assembly text into typed `LirItem`s (falling back to
/// `LirInst::Raw` only for unparsed inline assembly instructions).
pub fn parse_zilog_to_lir(src: &str) -> Vec<LirItem> {
    src.lines().map(parse_line).collect()
}

fn parse_line(raw: &str) -> LirItem {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return LirItem::Blank;
    }
    if trimmed.starts_with(';') {
        return LirItem::Comment(raw.to_string());
    }
    let code = trimmed.split(';').next().unwrap_or("").trim();
    if code.is_empty() {
        return LirItem::Comment(raw.to_string());
    }
    if let Some(rest) = code.strip_prefix(".org ")
        && let Some(addr) = parse_u16(rest.trim())
    {
        return LirItem::Org(addr);
    }
    if let Some(rest) = code.strip_prefix(".db ") {
        let mut bytes = Vec::new();
        let mut ok = true;
        for part in rest.split(',') {
            if let Some(b) = parse_u8(part.trim()) {
                bytes.push(b);
            } else {
                ok = false;
                break;
            }
        }
        if ok && !bytes.is_empty() {
            return LirItem::DataBytes(bytes);
        }
    }
    if let Some(rest) = code.strip_prefix(".dw ") {
        let words: Vec<SymExpr> = rest.split(',').map(|p| SymExpr::parse(p.trim())).collect();
        if !words.is_empty() {
            return LirItem::DataWords(words);
        }
    }
    if !code.starts_with('.')
        && let Some((name, val)) = code.split_once(" = ")
        && let Some(addr) = parse_u16(val.trim())
    {
        return LirItem::Equ(name.trim().to_string(), addr);
    }
    if let Some(lbl) = code.strip_suffix(':') {
        let lbl = lbl.trim();
        if !lbl.is_empty() && !lbl.contains(' ') {
            return LirItem::Label(lbl.to_string());
        }
    }
    LirItem::Inst(parse_inst(code))
}

fn parse_inst(code: &str) -> LirInst {
    let mut parts = code.splitn(2, |c: char| c.is_whitespace());
    let mnem = parts.next().unwrap_or("").to_ascii_lowercase();
    let tail = parts.next().unwrap_or("").trim();
    let ops: Vec<&str> = if tail.is_empty() {
        Vec::new()
    } else {
        tail.split(',').map(str::trim).collect()
    };

    match (mnem.as_str(), ops.as_slice()) {
        ("neg", []) => LirInst::Neg,
        ("cpl", []) => LirInst::Cpl,
        ("rla", []) => LirInst::Rla,
        ("rra", []) => LirInst::Rra,
        ("rlca", []) => LirInst::Rlca,
        ("rrca", []) => LirInst::Rrca,
        ("ccf", []) => LirInst::Ccf,
        ("scf", []) => LirInst::Scf,
        ("nop", []) => LirInst::Nop,
        ("di", []) => LirInst::Di,
        ("ei", []) => LirInst::Ei,
        ("halt", []) => LirInst::Halt,
        ("ldir", []) => LirInst::Ldir,
        ("lddr", []) => LirInst::Lddr,
        ("reti", []) => LirInst::Reti,
        ("retn", []) => LirInst::Retn,
        ("ret", []) => LirInst::Ret(None),
        ("ret", [c]) => {
            if let Some(cc) = Cond::parse(c) {
                LirInst::Ret(Some(cc))
            } else {
                LirInst::Raw(code.to_string())
            }
        }
        ("ex", [a, b])
            if a.eq_ignore_ascii_case("de") && b.eq_ignore_ascii_case("hl") =>
        {
            LirInst::ExDeHl
        }
        ("push", [r]) => Reg16::parse(r)
            .map(LirInst::Push)
            .unwrap_or_else(|| LirInst::Raw(code.to_string())),
        ("pop", [r]) => Reg16::parse(r)
            .map(LirInst::Pop)
            .unwrap_or_else(|| LirInst::Raw(code.to_string())),
        ("jp", [t]) => {
            if let Some(inner) = strip_parens(t)
                && let Some(r @ (Reg16::HL | Reg16::IX | Reg16::IY)) = Reg16::parse(inner)
            {
                LirInst::JpInd(r)
            } else {
                LirInst::Jp {
                    cond: None,
                    target: (*t).to_string(),
                }
            }
        }
        ("jp", [c, t]) => {
            if let Some(cc) = Cond::parse(c) {
                LirInst::Jp {
                    cond: Some(cc),
                    target: (*t).to_string(),
                }
            } else {
                LirInst::Raw(code.to_string())
            }
        }
        ("jr", [t]) => LirInst::Jr {
            cond: None,
            target: (*t).to_string(),
        },
        ("jr", [c, t]) => {
            if let Some(cc) = Cond::parse(c) {
                LirInst::Jr {
                    cond: Some(cc),
                    target: (*t).to_string(),
                }
            } else {
                LirInst::Raw(code.to_string())
            }
        }
        ("djnz", [t]) => LirInst::Djnz((*t).to_string()),
        ("call", [t]) => LirInst::Call {
            cond: None,
            target: (*t).to_string(),
        },
        ("call", [c, t]) => {
            if let Some(cc) = Cond::parse(c) {
                LirInst::Call {
                    cond: Some(cc),
                    target: (*t).to_string(),
                }
            } else {
                LirInst::Raw(code.to_string())
            }
        }
        ("in", [dst, src]) if dst.eq_ignore_ascii_case("a") => {
            if src.eq_ignore_ascii_case("(c)") {
                LirInst::InAC
            } else if let Some(inner) = strip_parens(src)
                && let Some(p) = parse_u8(inner)
            {
                LirInst::InAImm(p)
            } else {
                LirInst::Raw(code.to_string())
            }
        }
        ("out", [dst, src]) if src.eq_ignore_ascii_case("a") => {
            if dst.eq_ignore_ascii_case("(c)") {
                LirInst::OutCA
            } else if let Some(inner) = strip_parens(dst)
                && let Some(p) = parse_u8(inner)
            {
                LirInst::OutImmA(p)
            } else {
                LirInst::Raw(code.to_string())
            }
        }
        ("inc", [op]) => {
            if let Some(r16) = Reg16::parse(op) {
                LirInst::Inc16(r16)
            } else if let Some(op8) = parse_op8_non_imm(op) {
                LirInst::Inc8(op8)
            } else {
                LirInst::Raw(code.to_string())
            }
        }
        ("dec", [op]) => {
            if let Some(r16) = Reg16::parse(op) {
                LirInst::Dec16(r16)
            } else if let Some(op8) = parse_op8_non_imm(op) {
                LirInst::Dec8(op8)
            } else {
                LirInst::Raw(code.to_string())
            }
        }
        ("sla" | "sra" | "srl" | "rl" | "rr", [op]) => {
            let sop = match mnem.as_str() {
                "sla" => ShiftOp::Sla,
                "sra" => ShiftOp::Sra,
                "srl" => ShiftOp::Srl,
                "rl" => ShiftOp::Rl,
                "rr" => ShiftOp::Rr,
                _ => unreachable!(),
            };
            parse_op8_non_imm(op)
                .map(|target| LirInst::Shift8 { op: sop, target })
                .unwrap_or_else(|| LirInst::Raw(code.to_string()))
        }
        ("bit" | "set" | "res", [b, op]) => {
            if let Some(bit) = b.parse::<u8>().ok().filter(|&v| v < 8)
                && let Some(target) = parse_op8_non_imm(op)
            {
                match mnem.as_str() {
                    "bit" => LirInst::Bit { bit, target },
                    "set" => LirInst::Set { bit, target },
                    "res" => LirInst::Res { bit, target },
                    _ => unreachable!(),
                }
            } else {
                LirInst::Raw(code.to_string())
            }
        }
        ("add", [d, s]) => {
            if let Some(dst) = Reg16::parse(d)
                && let Some(src) = Reg16::parse(s)
            {
                LirInst::Add16 { dst, src }
            } else if d.eq_ignore_ascii_case("a")
                && let Some(src) = parse_op8(s)
            {
                LirInst::Alu8 {
                    op: AluOp8::Add,
                    src,
                }
            } else {
                LirInst::Raw(code.to_string())
            }
        }
        ("adc", [d, s]) => {
            if d.eq_ignore_ascii_case("hl")
                && let Some(src) = Reg16::parse(s)
            {
                LirInst::AdcHl { src }
            } else if d.eq_ignore_ascii_case("a")
                && let Some(src) = parse_op8(s)
            {
                LirInst::Alu8 {
                    op: AluOp8::Adc,
                    src,
                }
            } else {
                LirInst::Raw(code.to_string())
            }
        }
        ("sbc", [d, s]) => {
            if d.eq_ignore_ascii_case("hl")
                && let Some(src) = Reg16::parse(s)
            {
                LirInst::SbcHl { src }
            } else if d.eq_ignore_ascii_case("a")
                && let Some(src) = parse_op8(s)
            {
                LirInst::Alu8 {
                    op: AluOp8::Sbc,
                    src,
                }
            } else {
                LirInst::Raw(code.to_string())
            }
        }
        ("sub" | "and" | "xor" | "or" | "cp", [s]) => {
            let op = match mnem.as_str() {
                "sub" => AluOp8::Sub,
                "and" => AluOp8::And,
                "xor" => AluOp8::Xor,
                "or" => AluOp8::Or,
                "cp" => AluOp8::Cp,
                _ => unreachable!(),
            };
            parse_op8(s)
                .map(|src| LirInst::Alu8 { op, src })
                .unwrap_or_else(|| LirInst::Raw(code.to_string()))
        }
        ("ld", [dst, src]) => parse_ld(dst, src, code),
        _ => LirInst::Raw(code.to_string()),
    }
}

fn parse_ld(dst: &str, src: &str, raw_code: &str) -> LirInst {
    // 1. 16-bit loads & stores
    if let Some(d16) = Reg16::parse(dst) {
        if d16 == Reg16::SP
            && let Some(s16 @ (Reg16::HL | Reg16::IX | Reg16::IY)) = Reg16::parse(src)
        {
            return LirInst::LdSp(s16);
        }
        if let Some(inner) = strip_parens(src) {
            return LirInst::Ld16Mem {
                dst: d16,
                addr: SymExpr::parse(inner),
            };
        }
        return LirInst::Ld16Imm {
            dst: d16,
            val: SymExpr::parse(src),
        };
    }
    if let Some(s16) = Reg16::parse(src)
        && let Some(inner) = strip_parens(dst)
    {
        return LirInst::St16Mem {
            addr: SymExpr::parse(inner),
            src: s16,
        };
    }

    // 2. Indexed store: `ld (ix + d), op`
    if let Some((reg, disp)) = parse_indexed(dst)
        && let Some(op) = parse_op8_reg_or_imm(src)
    {
        return LirInst::LdInd8 {
            reg,
            disp,
            src: op,
        };
    }

    // 3. `(hl)` indirect store: `ld (hl), op`
    if dst.eq_ignore_ascii_case("(hl)")
        && let Some(op) = parse_op8_reg_or_imm(src)
    {
        return LirInst::StHlInd8 { src: op };
    }

    // 4. 8-bit register destination
    if let Some(d8) = Reg8::parse(dst) {
        if let Some(op8) = parse_op8(src) {
            return LirInst::Ld8 { dst: d8, src: op8 };
        }
        if let Some(inner) = strip_parens(src) {
            if d8 == Reg8::A
                && let Some(r16 @ (Reg16::BC | Reg16::DE)) = Reg16::parse(inner)
            {
                return LirInst::LdAInd16(r16);
            }
            return LirInst::LdMem8 {
                dst: d8,
                addr: SymExpr::parse(inner),
            };
        }
    }

    // 5. Direct / (bc) / (de) 8-bit store
    if let Some(inner) = strip_parens(dst)
        && let Some(s8) = Reg8::parse(src)
    {
        if s8 == Reg8::A
            && let Some(r16 @ (Reg16::BC | Reg16::DE)) = Reg16::parse(inner)
        {
            return LirInst::StInd16A(r16);
        }
        return LirInst::StMem8 {
            addr: SymExpr::parse(inner),
            src: s8,
        };
    }

    LirInst::Raw(raw_code.to_string())
}

fn strip_parens(s: &str) -> Option<&str> {
    s.trim().strip_prefix('(')?.strip_suffix(')').map(str::trim)
}

fn parse_indexed(s: &str) -> Option<(IndexReg, i8)> {
    let inner = strip_parens(s)?;
    let lower = inner.to_ascii_lowercase();
    for (prefix, reg) in [("ix", IndexReg::IX), ("iy", IndexReg::IY)] {
        if let Some(rest) = lower.strip_prefix(prefix) {
            let rest = rest.trim();
            if rest.is_empty() {
                return Some((reg, 0));
            }
            if let Some(p) = rest.strip_prefix('+') {
                return p.trim().parse::<i8>().ok().map(|d| (reg, d));
            }
            if let Some(m) = rest.strip_prefix('-') {
                return m
                    .trim()
                    .parse::<i16>()
                    .ok()
                    .and_then(|d| i8::try_from(-d).ok())
                    .map(|d| (reg, d));
            }
        }
    }
    None
}

fn parse_op8_non_imm(s: &str) -> Option<Op8> {
    if let Some(r) = Reg8::parse(s) {
        return Some(Op8::Reg(r));
    }
    if s.trim().eq_ignore_ascii_case("(hl)") {
        return Some(Op8::HlInd);
    }
    if let Some((reg, disp)) = parse_indexed(s) {
        return Some(Op8::Indexed(reg, disp));
    }
    None
}

fn parse_op8_reg_or_imm(s: &str) -> Option<Op8> {
    if let Some(r) = Reg8::parse(s) {
        return Some(Op8::Reg(r));
    }
    parse_u8(s).map(Op8::Imm)
}

fn parse_op8(s: &str) -> Option<Op8> {
    parse_op8_non_imm(s).or_else(|| parse_u8(s).map(Op8::Imm))
}

fn parse_u16(s: &str) -> Option<u16> {
    let t = s.trim();
    if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        u16::from_str_radix(hex, 16).ok()
    } else {
        t.parse::<u16>().ok()
    }
}

fn parse_i32_lit(s: &str) -> Option<i32> {
    let t = s.trim();
    if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        i32::from_str_radix(hex, 16).ok()
    } else {
        t.parse::<i32>().ok()
    }
}

fn parse_u8(s: &str) -> Option<u8> {
    let t = s.trim();
    if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        u8::from_str_radix(hex, 16).ok()
    } else {
        t.parse::<u8>()
            .ok()
            .or_else(|| t.parse::<i8>().ok().map(|v| v as u8))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_byte_lengths() {
        assert_eq!(
            LirInst::Ld8 {
                dst: Reg8::A,
                src: Op8::Reg(Reg8::C)
            }
            .byte_len(),
            1
        );
        assert_eq!(
            LirInst::Ld8 {
                dst: Reg8::A,
                src: Op8::Imm(0x10)
            }
            .byte_len(),
            2
        );
        assert_eq!(
            LirInst::Ld8 {
                dst: Reg8::A,
                src: Op8::Indexed(IndexReg::IX, 5)
            }
            .byte_len(),
            3
        );
        assert_eq!(
            LirInst::LdInd8 {
                reg: IndexReg::IX,
                disp: 1,
                src: Op8::Imm(0)
            }
            .byte_len(),
            4
        );
        assert_eq!(
            LirInst::Ld16Mem {
                dst: Reg16::HL,
                addr: SymExpr::Addr(0xC004)
            }
            .byte_len(),
            3
        );
        assert_eq!(
            LirInst::Ld16Mem {
                dst: Reg16::DE,
                addr: SymExpr::Addr(0xC004)
            }
            .byte_len(),
            4
        );
        assert_eq!(
            LirInst::Bit {
                bit: 7,
                target: Op8::Reg(Reg8::H)
            }
            .byte_len(),
            2
        );
        assert_eq!(
            LirInst::Bit {
                bit: 7,
                target: Op8::Indexed(IndexReg::IX, 3)
            }
            .byte_len(),
            4
        );
    }

    #[test]
    fn parse_and_format_sdasz80() {
        let asm = "\
    ld (0xC004), hl
    ld a, (ix + 5)
    ld hl, FM_FNUM + 2
    ld (ix + 1), 0x00
    add a, 0x10
    bit 7, h
    jp nz, foo_bb3
";
        let items = parse_zilog_to_lir(asm);
        let formatted: Vec<String> = items
            .iter()
            .filter_map(|it| match it {
                LirItem::Inst(i) => {
                    assert!(!matches!(i, LirInst::Raw(_)), "unexpected Raw: {i:?}");
                    Some(i.format_sdasz80(0xC000, 0xC100))
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            formatted,
            vec![
                "ld (__aun_ram + 4), hl",
                "ld a, 5(ix)",
                "ld hl, #(FM_FNUM + 2)",
                "ld 1(ix), #0x00",
                "add a, #0x10",
                "bit 7, h",
                "jp nz, foo_bb3",
            ]
        );
    }

    #[test]
    fn relax_jp_shrinks_in_range_and_preserves_po_and_far() {
        let mut src = String::from("f:\n");
        src.push_str("    jp nz, f_bb1\n");
        src.push_str("    jp po, f_bb1\n");
        src.push_str("    jp f_bb2\n");
        src.push_str("f_bb1:\n");
        for _ in 0..140 {
            src.push_str("    nop\n");
        }
        src.push_str("f_bb2:\n    ret\n");

        let mut items = parse_zilog_to_lir(&src);
        relax_jp_to_jr(&mut items);

        let insts: Vec<&LirInst> = items
            .iter()
            .filter_map(|it| match it {
                LirItem::Inst(i) => Some(i),
                _ => None,
            })
            .collect();
        assert!(matches!(insts[0], LirInst::Jr { cond: Some(Cond::NZ), .. }));
        assert!(matches!(insts[1], LirInst::Jp { cond: Some(Cond::PO), .. }));
        assert!(matches!(insts[2], LirInst::Jp { cond: None, .. }));
    }

    #[test]
    fn parses_edge_cases_and_exact_lengths() {
        let asm = "\
    ld sp, hl
    ld sp, ix
    ld a, (ix - 128)
    ld a, (bc)
    ld (de), a
    jp (ix)
    ld a, -1
    ld hl, SYM + 0x10
";
        let items = parse_zilog_to_lir(asm);
        let insts: Vec<&LirInst> = items
            .iter()
            .filter_map(|it| match it {
                LirItem::Inst(i) => {
                    assert!(!matches!(i, LirInst::Raw(_)), "unexpected Raw: {i:?}");
                    Some(i)
                }
                _ => None,
            })
            .collect();
        let lens: Vec<usize> = insts.iter().map(|i| i.byte_len()).collect();
        assert_eq!(lens, vec![1, 2, 3, 1, 1, 2, 2, 3]);
        let sdas: Vec<String> = insts
            .iter()
            .map(|i| i.format_sdasz80(0xC000, 0xC100))
            .collect();
        assert_eq!(
            sdas,
            vec![
                "ld sp, hl",
                "ld sp, ix",
                "ld a, -128(ix)",
                "ld a, (bc)",
                "ld (de), a",
                "jp (ix)",
                "ld a, #0xFF",
                "ld hl, #(SYM + 16)",
            ]
        );
    }
}
