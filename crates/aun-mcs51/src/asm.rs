//! Built-in two-pass MCS-51 / WCH E8051 assembler and Intel HEX linker.

use std::collections::HashMap;
use std::fmt::Write;

#[derive(Debug, Clone)]
pub struct AssembledOutput {
    pub bin: Vec<u8>,
    pub intel_hex: String,
    pub symbols: HashMap<String, u16>,
}

#[derive(Debug, Clone)]
enum ExprKind {
    Full,
    Low,
    High,
}

#[derive(Debug, Clone)]
enum Stmt {
    Label(String),
    Org(u16),
    Db(Vec<String>),
    Dw(Vec<String>),
    Ds(usize),
    Inst { mnem: String, ops: Vec<String> },
}

fn strip_comment(line: &str) -> &str {
    let mut in_str = false;
    let bytes = line.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'"' {
            in_str = !in_str;
        } else if !in_str && b == b';' {
            return &line[..i];
        }
    }
    line
}

fn split_operands(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut depth = 0i32;
    let mut in_str = false;
    for ch in s.chars() {
        match ch {
            '"' => {
                in_str = !in_str;
                cur.push(ch);
            }
            '(' if !in_str => {
                depth += 1;
                cur.push(ch);
            }
            ')' if !in_str => {
                depth -= 1;
                cur.push(ch);
            }
            ',' if !in_str && depth == 0 => {
                let t = cur.trim();
                if !t.is_empty() {
                    out.push(t.to_string());
                }
                cur.clear();
            }
            _ => cur.push(ch),
        }
    }
    let t = cur.trim();
    if !t.is_empty() {
        out.push(t.to_string());
    }
    out
}

fn named_sfr_addr(name: &str) -> Option<u8> {
    Some(match name.to_ascii_uppercase().as_str() {
        "P0" => 0x80,
        "SP" => 0x81,
        "DPL" => 0x82,
        "DPH" => 0x83,
        "ROM_ADDR_L" => 0x84,
        "ROM_ADDR_H" => 0x85,
        "ROM_CTRL" | "ROM_STATUS" => 0x86,
        "PCON" => 0x87,
        "TCON" => 0x88,
        "TMOD" => 0x89,
        "TL0" => 0x8A,
        "TL1" => 0x8B,
        "TH0" => 0x8C,
        "TH1" => 0x8D,
        "ROM_DATA_L" => 0x8E,
        "ROM_DATA_H" => 0x8F,
        "P1" => 0x90,
        "SCON" => 0x98,
        "SBUF" => 0x99,
        "P2" => 0xA0,
        "SAFE_MOD" => 0xA1,
        "XBUS_AUX" => 0xA2,
        "IE" => 0xA8,
        "P3" => 0xB0,
        "GLOBAL_CFG" => 0xB1,
        "PLL_CFG" => 0xB2,
        "CLOCK_CFG" => 0xB3,
        "IP" => 0xB8,
        "P4_OUT" => 0xC0,
        "T2CON" => 0xC8,
        "T2MOD" => 0xC9,
        "RCAP2L" => 0xCA,
        "RCAP2H" => 0xCB,
        "TL2" => 0xCC,
        "TH2" => 0xCD,
        "PSW" => 0xD0,
        "ACC" => 0xE0,
        "IE_EX" => 0xE8,
        "IP_EX" => 0xE9,
        "B" => 0xF0,
        "SPI0_STAT" => 0xF8,
        "WDOG_COUNT" => 0xFF,
        _ => return None,
    })
}

fn named_bit_addr(name: &str) -> Option<u8> {
    if let Some((base, bit)) = name.split_once('.')
        && let Ok(bit_idx) = bit.trim().parse::<u8>()
        && bit_idx < 8
    {
        if let Some(sfr) = named_sfr_addr(base.trim()) {
            if sfr >= 0x80 && sfr.is_multiple_of(8) {
                return Some(sfr | bit_idx);
            }
        } else if let Ok(v) = parse_int_lit(base.trim())
            && (0x20..=0x2F).contains(&v)
        {
            return Some(((v as u8 - 0x20) << 3) | bit_idx);
        }
    }
    Some(match name.to_ascii_uppercase().as_str() {
        "CY" | "PSW.7" => 0xD7,
        "AC" => 0xD6,
        "F0" => 0xD5,
        "RS1" => 0xD4,
        "RS0" => 0xD3,
        "OV" => 0xD2,
        "EA" => 0xAF,
        "ET2" => 0xAD,
        "ES" => 0xAC,
        "ET1" => 0xAB,
        "EX1" => 0xAA,
        "ET0" => 0xA9,
        "EX0" => 0xA8,
        "TF1" => 0x8F,
        "TR1" => 0x8E,
        "TF0" => 0x8D,
        "TR0" => 0x8C,
        "IE1" => 0x8B,
        "IT1" => 0x8A,
        "IE0" => 0x89,
        "IT0" => 0x88,
        "RI" => 0x98,
        "TI" => 0x99,
        _ => return None,
    })
}

fn parse_int_lit(s: &str) -> Result<i64, String> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        i64::from_str_radix(hex, 16).map_err(|_| format!("invalid hex literal `{s}`"))
    } else if let Some(hex) = s.strip_prefix('$') {
        i64::from_str_radix(hex, 16).map_err(|_| format!("invalid hex literal `{s}`"))
    } else if let Some(bin) = s.strip_prefix("0b").or_else(|| s.strip_prefix("0B")) {
        i64::from_str_radix(bin, 2).map_err(|_| format!("invalid bin literal `{s}`"))
    } else {
        s.parse::<i64>()
            .map_err(|_| format!("invalid integer `{s}`"))
    }
}

fn eval_expr(expr: &str, pc: u16, syms: &HashMap<String, u16>) -> Result<i64, String> {
    let s = expr.trim();
    if s == "$" || s == "." {
        return Ok(pc as i64);
    }
    if let Some(a) = named_sfr_addr(s) {
        return Ok(a as i64);
    }
    if let Some(b) = named_bit_addr(s) {
        return Ok(b as i64);
    }
    if let Some(&v) = syms.get(s) {
        return Ok(v as i64);
    }
    if let Some((lhs, rhs)) = s.rsplit_once('+')
        && !lhs.is_empty()
    {
        return Ok(eval_expr(lhs, pc, syms)? + eval_expr(rhs, pc, syms)?);
    }
    if let Some((lhs, rhs)) = s.rsplit_once('-')
        && !lhs.is_empty()
    {
        return Ok(eval_expr(lhs, pc, syms)? - eval_expr(rhs, pc, syms)?);
    }
    if let Some(inner) = s.strip_prefix("lo(").and_then(|x| x.strip_suffix(')')) {
        return Ok(eval_expr(inner, pc, syms)? & 0xFF);
    }
    if let Some(inner) = s.strip_prefix("hi(").and_then(|x| x.strip_suffix(')')) {
        return Ok((eval_expr(inner, pc, syms)? >> 8) & 0xFF);
    }
    if let Some(inner) = s.strip_prefix('<') {
        return Ok(eval_expr(inner, pc, syms)? & 0xFF);
    }
    if let Some(inner) = s.strip_prefix('>') {
        return Ok((eval_expr(inner, pc, syms)? >> 8) & 0xFF);
    }
    parse_int_lit(s).map_err(|_| format!("undefined symbol or bad expr `{s}`"))
}

fn parse_imm(op: &str, pc: u16, syms: &HashMap<String, u16>, resolve: bool) -> Result<Option<i64>, String> {
    let t = op.trim();
    let Some(rest) = t.strip_prefix('#') else {
        return Ok(None);
    };
    if !resolve {
        return Ok(Some(0));
    }
    let (kind, inner) = if let Some(x) = rest.strip_prefix("lo(").and_then(|s| s.strip_suffix(')')) {
        (ExprKind::Low, x)
    } else if let Some(x) = rest.strip_prefix("hi(").and_then(|s| s.strip_suffix(')')) {
        (ExprKind::High, x)
    } else if let Some(x) = rest.strip_prefix('<') {
        (ExprKind::Low, x)
    } else if let Some(x) = rest.strip_prefix('>') {
        (ExprKind::High, x)
    } else {
        (ExprKind::Full, rest)
    };
    let val = eval_expr(inner, pc, syms)?;
    Ok(Some(match kind {
        ExprKind::Full => val,
        ExprKind::Low => val & 0xFF,
        ExprKind::High => (val >> 8) & 0xFF,
    }))
}

fn reg_code(op: &str) -> Option<u8> {
    match op.trim().to_ascii_uppercase().as_str() {
        "R0" => Some(0),
        "R1" => Some(1),
        "R2" => Some(2),
        "R3" => Some(3),
        "R4" => Some(4),
        "R5" => Some(5),
        "R6" => Some(6),
        "R7" => Some(7),
        _ => None,
    }
}

fn indirect_reg_code(op: &str) -> Option<u8> {
    match op.trim().to_ascii_uppercase().as_str() {
        "@R0" => Some(0),
        "@R1" => Some(1),
        _ => None,
    }
}

fn parse_direct(op: &str, pc: u16, syms: &HashMap<String, u16>, resolve: bool) -> Result<u8, String> {
    if !resolve {
        return Ok(0);
    }
    let v = eval_expr(op, pc, syms)?;
    if (0..=255).contains(&v) {
        Ok(v as u8)
    } else {
        Err(format!("direct address `{op}` ({v}) out of 0..=255 range"))
    }
}

fn parse_bit(op: &str, pc: u16, syms: &HashMap<String, u16>, resolve: bool) -> Result<(bool, u8), String> {
    let t = op.trim();
    let (neg, inner) = if let Some(rest) = t.strip_prefix('/') {
        (true, rest.trim())
    } else {
        (false, t)
    };
    if !resolve {
        return Ok((neg, 0));
    }
    if let Some(b) = named_bit_addr(inner) {
        return Ok((neg, b));
    }
    let v = eval_expr(inner, pc, syms)?;
    if (0..=255).contains(&v) {
        Ok((neg, v as u8))
    } else {
        Err(format!("bit address `{op}` ({v}) out of range"))
    }
}

fn rel_offset(target_str: &str, inst_end_pc: u16, syms: &HashMap<String, u16>, resolve: bool) -> Result<u8, String> {
    if !resolve {
        return Ok(0);
    }
    let target = eval_expr(target_str, inst_end_pc, syms)?;
    let diff = target - (inst_end_pc as i64);
    if !(-128..=127).contains(&diff) {
        return Err(format!(
            "relative branch `{target_str}` at 0x{inst_end_pc:04X} out of range ({diff})"
        ));
    }
    Ok(diff as i8 as u8)
}

fn parse_statements(src: &str) -> Result<Vec<Stmt>, String> {
    let mut stmts = Vec::new();
    for raw in src.lines() {
        let mut line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }
        while let Some(idx) = line.find(':') {
            let before = line[..idx].trim();
            if before.is_empty() || before.contains(char::is_whitespace) || before.starts_with('.') {
                break;
            }
            stmts.push(Stmt::Label(before.to_string()));
            line = line[idx + 1..].trim();
            if line.is_empty() {
                break;
            }
        }
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix(".org").or_else(|| line.strip_prefix(".ORG")) {
            let v = parse_int_lit(rest.trim())?;
            stmts.push(Stmt::Org(v as u16));
            continue;
        }
        if let Some(rest) = line
            .strip_prefix(".db")
            .or_else(|| line.strip_prefix(".byte"))
        {
            stmts.push(Stmt::Db(split_operands(rest)));
            continue;
        }
        if let Some(rest) = line
            .strip_prefix(".dw")
            .or_else(|| line.strip_prefix(".word"))
        {
            stmts.push(Stmt::Dw(split_operands(rest)));
            continue;
        }
        if let Some(rest) = line.strip_prefix(".ds") {
            let n = parse_int_lit(rest.trim())?;
            stmts.push(Stmt::Ds(n as usize));
            continue;
        }
        let mut parts = line.splitn(2, char::is_whitespace);
        let mnem = parts.next().unwrap_or("").trim().to_ascii_lowercase();
        let rest = parts.next().unwrap_or("").trim();
        let ops = if rest.is_empty() {
            Vec::new()
        } else {
            split_operands(rest)
        };
        stmts.push(Stmt::Inst { mnem, ops });
    }
    Ok(stmts)
}

fn encode_inst(
    mnem: &str,
    ops: &[String],
    pc: u16,
    syms: &HashMap<String, u16>,
    resolve: bool,
) -> Result<Vec<u8>, String> {
    let op_up = |idx: usize| -> String {
        ops.get(idx).map(|s| s.trim().to_ascii_uppercase()).unwrap_or_default()
    };
    let need = |n: usize| -> Result<(), String> {
        if ops.len() != n {
            Err(format!("instruction `{mnem}` expected {n} operand(s), got {ops:?}"))
        } else {
            Ok(())
        }
    };

    let bytes = match mnem {
        "nop" => {
            need(0)?;
            vec![0x00]
        }
        "ret" => {
            need(0)?;
            vec![0x22]
        }
        "reti" => {
            need(0)?;
            vec![0x32]
        }
        "mul" => {
            need(1)?;
            if op_up(0) != "AB" {
                return Err(format!("expected `mul ab`, got {ops:?}"));
            }
            vec![0xA4]
        }
        "div" => {
            need(1)?;
            if op_up(0) != "AB" {
                return Err(format!("expected `div ab`, got {ops:?}"));
            }
            vec![0x84]
        }
        "da" => {
            need(1)?;
            if op_up(0) != "A" {
                return Err(format!("expected `da a`, got {ops:?}"));
            }
            vec![0xD4]
        }
        "rr" | "rrc" | "rl" | "rlc" | "swap" => {
            need(1)?;
            if op_up(0) != "A" {
                return Err(format!("expected `{mnem} a`, got {ops:?}"));
            }
            vec![match mnem {
                "rr" => 0x03,
                "rrc" => 0x13,
                "rl" => 0x23,
                "rlc" => 0x33,
                "swap" => 0xC4,
                _ => unreachable!(),
            }]
        }
        "inc" => {
            need(1)?;
            let o = op_up(0);
            if o == "A" {
                vec![0x04]
            } else if o == "DPTR" {
                vec![0xA3]
            } else if let Some(ri) = indirect_reg_code(&ops[0]) {
                vec![0x06 | ri]
            } else if let Some(rn) = reg_code(&ops[0]) {
                vec![0x08 | rn]
            } else {
                vec![0x05, parse_direct(&ops[0], pc, syms, resolve)?]
            }
        }
        "dec" => {
            need(1)?;
            let o = op_up(0);
            if o == "A" {
                vec![0x14]
            } else if let Some(ri) = indirect_reg_code(&ops[0]) {
                vec![0x16 | ri]
            } else if let Some(rn) = reg_code(&ops[0]) {
                vec![0x18 | rn]
            } else {
                vec![0x15, parse_direct(&ops[0], pc, syms, resolve)?]
            }
        }
        "clr" => {
            need(1)?;
            let o = op_up(0);
            if o == "A" {
                vec![0xE4]
            } else if o == "C" {
                vec![0xC3]
            } else {
                let (_, b) = parse_bit(&ops[0], pc, syms, resolve)?;
                vec![0xC2, b]
            }
        }
        "cpl" => {
            need(1)?;
            let o = op_up(0);
            if o == "A" {
                vec![0xF4]
            } else if o == "C" {
                vec![0xB3]
            } else {
                let (_, b) = parse_bit(&ops[0], pc, syms, resolve)?;
                vec![0xB2, b]
            }
        }
        "setb" => {
            need(1)?;
            if op_up(0) == "C" {
                vec![0xD3]
            } else {
                let (_, b) = parse_bit(&ops[0], pc, syms, resolve)?;
                vec![0xD2, b]
            }
        }
        "push" => {
            need(1)?;
            vec![0xC0, parse_direct(&ops[0], pc, syms, resolve)?]
        }
        "pop" => {
            need(1)?;
            vec![0xD0, parse_direct(&ops[0], pc, syms, resolve)?]
        }
        "ljmp" | "lcall" => {
            need(1)?;
            let opcode = if mnem == "ljmp" { 0x02 } else { 0x12 };
            let target = if resolve {
                eval_expr(&ops[0], pc, syms)? as u16
            } else {
                0
            };
            vec![opcode, (target >> 8) as u8, (target & 0xFF) as u8]
        }
        "ajmp" | "acall" => {
            need(1)?;
            let target = if resolve {
                eval_expr(&ops[0], pc, syms)? as u16
            } else {
                0
            };
            let next_pc = pc.wrapping_add(2);
            if resolve && (target & 0xF800) != (next_pc & 0xF800) {
                return Err(format!("{mnem} target 0x{target:04X} crosses 2KB page boundary from 0x{next_pc:04X}"));
            }
            let base = if mnem == "ajmp" { 0x01 } else { 0x11 };
            let hi = (((target >> 8) & 0x07) as u8) << 5;
            vec![base | hi, (target & 0xFF) as u8]
        }
        "jmp" => {
            need(1)?;
            if op_up(0).replace(' ', "") != "@A+DPTR" {
                return Err(format!("expected `jmp @a+dptr`, got {ops:?}"));
            }
            vec![0x73]
        }
        "sjmp" | "jz" | "jnz" | "jc" | "jnc" => {
            need(1)?;
            let opcode = match mnem {
                "sjmp" => 0x80,
                "jz" => 0x60,
                "jnz" => 0x70,
                "jc" => 0x40,
                "jnc" => 0x50,
                _ => unreachable!(),
            };
            let rel = rel_offset(&ops[0], pc.wrapping_add(2), syms, resolve)?;
            vec![opcode, rel]
        }
        "jb" | "jnb" | "jbc" => {
            need(2)?;
            let opcode = match mnem {
                "jb" => 0x20,
                "jnb" => 0x30,
                "jbc" => 0x10,
                _ => unreachable!(),
            };
            let (_, bit) = parse_bit(&ops[0], pc, syms, resolve)?;
            let rel = rel_offset(&ops[1], pc.wrapping_add(3), syms, resolve)?;
            vec![opcode, bit, rel]
        }
        "djnz" => {
            need(2)?;
            if let Some(rn) = reg_code(&ops[0]) {
                let rel = rel_offset(&ops[1], pc.wrapping_add(2), syms, resolve)?;
                vec![0xD8 | rn, rel]
            } else {
                let d = parse_direct(&ops[0], pc, syms, resolve)?;
                let rel = rel_offset(&ops[1], pc.wrapping_add(3), syms, resolve)?;
                vec![0xD5, d, rel]
            }
        }
        "cjne" => {
            need(3)?;
            let dst = op_up(0);
            if dst == "A" {
                if let Some(imm) = parse_imm(&ops[1], pc, syms, resolve)? {
                    let rel = rel_offset(&ops[2], pc.wrapping_add(3), syms, resolve)?;
                    vec![0xB4, imm as u8, rel]
                } else {
                    let d = parse_direct(&ops[1], pc, syms, resolve)?;
                    let rel = rel_offset(&ops[2], pc.wrapping_add(3), syms, resolve)?;
                    vec![0xB5, d, rel]
                }
            } else if let Some(ri) = indirect_reg_code(&ops[0]) {
                let imm = parse_imm(&ops[1], pc, syms, resolve)?
                    .ok_or_else(|| format!("expected immediate in `{mnem} {ops:?}`"))?;
                let rel = rel_offset(&ops[2], pc.wrapping_add(3), syms, resolve)?;
                vec![0xB6 | ri, imm as u8, rel]
            } else if let Some(rn) = reg_code(&ops[0]) {
                let imm = parse_imm(&ops[1], pc, syms, resolve)?
                    .ok_or_else(|| format!("expected immediate in `{mnem} {ops:?}`"))?;
                let rel = rel_offset(&ops[2], pc.wrapping_add(3), syms, resolve)?;
                vec![0xB8 | rn, imm as u8, rel]
            } else {
                return Err(format!("invalid operands for `cjne`: {ops:?}"));
            }
        }
        "add" | "addc" | "subb" => {
            need(2)?;
            if op_up(0) != "A" {
                return Err(format!("expected `{mnem} a, ...`, got {ops:?}"));
            }
            let base = match mnem {
                "add" => 0x20,
                "addc" => 0x30,
                "subb" => 0x90,
                _ => unreachable!(),
            };
            if let Some(imm) = parse_imm(&ops[1], pc, syms, resolve)? {
                vec![base | 0x04, imm as u8]
            } else if let Some(ri) = indirect_reg_code(&ops[1]) {
                vec![base | 0x06 | ri]
            } else if let Some(rn) = reg_code(&ops[1]) {
                vec![base | 0x08 | rn]
            } else {
                vec![base | 0x05, parse_direct(&ops[1], pc, syms, resolve)?]
            }
        }
        "orl" | "anl" | "xrl" => {
            need(2)?;
            let d = op_up(0);
            let base = match mnem {
                "orl" => 0x40,
                "anl" => 0x50,
                "xrl" => 0x60,
                _ => unreachable!(),
            };
            if d == "C" {
                let (neg, b) = parse_bit(&ops[1], pc, syms, resolve)?;
                let opcode = match (mnem, neg) {
                    ("orl", false) => 0x72,
                    ("orl", true) => 0xA0,
                    ("anl", false) => 0x82,
                    ("anl", true) => 0xB0,
                    _ => return Err(format!("unsupported `{mnem} c, ...`")),
                };
                vec![opcode, b]
            } else if d == "A" {
                if let Some(imm) = parse_imm(&ops[1], pc, syms, resolve)? {
                    vec![base | 0x04, imm as u8]
                } else if let Some(ri) = indirect_reg_code(&ops[1]) {
                    vec![base | 0x06 | ri]
                } else if let Some(rn) = reg_code(&ops[1]) {
                    vec![base | 0x08 | rn]
                } else {
                    vec![base | 0x05, parse_direct(&ops[1], pc, syms, resolve)?]
                }
            } else {
                let da = parse_direct(&ops[0], pc, syms, resolve)?;
                if op_up(1) == "A" {
                    vec![base | 0x02, da]
                } else if let Some(imm) = parse_imm(&ops[1], pc, syms, resolve)? {
                    vec![base | 0x03, da, imm as u8]
                } else {
                    return Err(format!("invalid operands for `{mnem}`: {ops:?}"));
                }
            }
        }
        "xch" => {
            need(2)?;
            if op_up(0) != "A" {
                return Err(format!("expected `xch a, ...`, got {ops:?}"));
            }
            if let Some(ri) = indirect_reg_code(&ops[1]) {
                vec![0xC6 | ri]
            } else if let Some(rn) = reg_code(&ops[1]) {
                vec![0xC8 | rn]
            } else {
                vec![0xC5, parse_direct(&ops[1], pc, syms, resolve)?]
            }
        }
        "movc" => {
            need(2)?;
            if op_up(0) != "A" {
                return Err(format!("expected `movc a, ...`, got {ops:?}"));
            }
            let src = op_up(1).replace(' ', "");
            match src.as_str() {
                "@A+DPTR" => vec![0x93],
                "@A+PC" => vec![0x83],
                _ => return Err(format!("invalid movc source `{src}`")),
            }
        }
        "movx" => {
            need(2)?;
            let d = op_up(0).replace(' ', "");
            let s = op_up(1).replace(' ', "");
            match (d.as_str(), s.as_str()) {
                ("A", "@DPTR") => vec![0xE0],
                ("@DPTR", "A") => vec![0xF0],
                ("@DPTR1++", "A") => vec![0xA5],
                ("A", "@R0") => vec![0xE2],
                ("A", "@R1") => vec![0xE3],
                ("@R0", "A") => vec![0xF2],
                ("@R1", "A") => vec![0xF3],
                _ => return Err(format!("invalid movx operands {ops:?}")),
            }
        }
        "mov" => {
            need(2)?;
            let d = op_up(0);
            let s = op_up(1);
            if d == "DPTR" {
                let imm = parse_imm(&ops[1], pc, syms, resolve)?
                    .ok_or_else(|| format!("expected immediate for `mov dptr, ...`: {ops:?}"))?;
                let v = imm as u16;
                vec![0x90, (v >> 8) as u8, (v & 0xFF) as u8]
            } else if d == "C" {
                let (neg, b) = parse_bit(&ops[1], pc, syms, resolve)?;
                if neg {
                    return Err("`mov c, /bit` not supported".into());
                }
                vec![0xA2, b]
            } else if s == "C" {
                let (neg, b) = parse_bit(&ops[0], pc, syms, resolve)?;
                if neg {
                    return Err("`mov /bit, c` not supported".into());
                }
                vec![0x92, b]
            } else if d == "A" {
                if let Some(imm) = parse_imm(&ops[1], pc, syms, resolve)? {
                    vec![0x74, imm as u8]
                } else if let Some(ri) = indirect_reg_code(&ops[1]) {
                    vec![0xE6 | ri]
                } else if let Some(rn) = reg_code(&ops[1]) {
                    vec![0xE8 | rn]
                } else {
                    vec![0xE5, parse_direct(&ops[1], pc, syms, resolve)?]
                }
            } else if let Some(ri) = indirect_reg_code(&ops[0]) {
                if s == "A" {
                    vec![0xF6 | ri]
                } else if let Some(imm) = parse_imm(&ops[1], pc, syms, resolve)? {
                    vec![0x76 | ri, imm as u8]
                } else {
                    vec![0xA6 | ri, parse_direct(&ops[1], pc, syms, resolve)?]
                }
            } else if let Some(rn) = reg_code(&ops[0]) {
                if s == "A" {
                    vec![0xF8 | rn]
                } else if let Some(imm) = parse_imm(&ops[1], pc, syms, resolve)? {
                    vec![0x78 | rn, imm as u8]
                } else {
                    vec![0xA8 | rn, parse_direct(&ops[1], pc, syms, resolve)?]
                }
            } else {
                let da = parse_direct(&ops[0], pc, syms, resolve)?;
                if s == "A" {
                    vec![0xF5, da]
                } else if let Some(imm) = parse_imm(&ops[1], pc, syms, resolve)? {
                    vec![0x75, da, imm as u8]
                } else if let Some(ri) = indirect_reg_code(&ops[1]) {
                    vec![0x86 | ri, da]
                } else if let Some(rn) = reg_code(&ops[1]) {
                    vec![0x88 | rn, da]
                } else {
                    // Note 8051 encoding for `MOV direct_dst, direct_src` is `[0x85, src, dst]`!
                    let sa = parse_direct(&ops[1], pc, syms, resolve)?;
                    vec![0x85, sa, da]
                }
            }
        }
        other => return Err(format!("unsupported MCS-51 instruction `{other}` ({ops:?})")),
    };
    Ok(bytes)
}

pub fn format_intel_hex(segments: &[(u16, Vec<u8>)]) -> String {
    let mut out = String::new();
    for (base, bytes) in segments {
        for (chunk_idx, chunk) in bytes.chunks(16).enumerate() {
            let addr = base.wrapping_add((chunk_idx * 16) as u16);
            let len = chunk.len() as u8;
            let hi = (addr >> 8) as u8;
            let lo = (addr & 0xFF) as u8;
            let mut sum = (len as u16) + (hi as u16) + (lo as u16);
            write!(out, ":{len:02X}{addr:04X}00").unwrap();
            for &b in chunk {
                sum += b as u16;
                write!(out, "{b:02X}").unwrap();
            }
            let cksum = ((!sum).wrapping_add(1) & 0xFF) as u8;
            writeln!(out, "{cksum:02X}").unwrap();
        }
    }
    writeln!(out, ":00000001FF").unwrap();
    out
}

pub fn assemble_mcs51(src: &str, code_limit: u16) -> Result<AssembledOutput, String> {
    let stmts = parse_statements(src)?;
    let mut syms: HashMap<String, u16> = HashMap::new();

    // Pass 1: assign label addresses.
    let mut pc = 0u16;
    for st in &stmts {
        match st {
            Stmt::Label(name) => {
                if syms.insert(name.clone(), pc).is_some() {
                    return Err(format!("duplicate MCS-51 label `{name}`"));
                }
            }
            Stmt::Org(addr) => pc = *addr,
            Stmt::Db(items) => {
                for it in items {
                    let t = it.trim();
                    if t.starts_with('"') && t.ends_with('"') && t.len() >= 2 {
                        pc = pc.wrapping_add((t.len() - 2) as u16);
                    } else {
                        pc = pc.wrapping_add(1);
                    }
                }
            }
            Stmt::Dw(items) => {
                pc = pc.wrapping_add((items.len() * 2) as u16);
            }
            Stmt::Ds(n) => {
                pc = pc.wrapping_add(*n as u16);
            }
            Stmt::Inst { mnem, ops } => {
                let len = encode_inst(mnem, ops, pc, &syms, false)?.len() as u16;
                pc = pc.wrapping_add(len);
            }
        }
    }

    // Pass 2: emit segments (`(addr, bytes)` merged sequentially) and flat bin.
    let mut segments: Vec<(u16, Vec<u8>)> = Vec::new();
    let mut cur_start = 0u16;
    let mut cur_bytes: Vec<u8> = Vec::new();
    let flush = |start: &mut u16, bytes: &mut Vec<u8>, segs: &mut Vec<(u16, Vec<u8>)>| {
        if !bytes.is_empty() {
            segs.push((*start, std::mem::take(bytes)));
        }
    };

    pc = 0;
    let mut max_end = 0u16;
    for st in &stmts {
        match st {
            Stmt::Label(_) => {}
            Stmt::Org(addr) => {
                if *addr != pc {
                    flush(&mut cur_start, &mut cur_bytes, &mut segments);
                    pc = *addr;
                    cur_start = pc;
                }
            }
            Stmt::Db(items) => {
                for it in items {
                    let t = it.trim();
                    if t.starts_with('"') && t.ends_with('"') && t.len() >= 2 {
                        cur_bytes.extend_from_slice(&t.as_bytes()[1..t.len() - 1]);
                        pc = pc.wrapping_add((t.len() - 2) as u16);
                    } else {
                        let v = eval_expr(t, pc, &syms)? as u8;
                        cur_bytes.push(v);
                        pc = pc.wrapping_add(1);
                    }
                }
                max_end = max_end.max(pc);
            }
            Stmt::Dw(items) => {
                for it in items {
                    let v = eval_expr(it, pc, &syms)? as u16;
                    cur_bytes.push((v & 0xFF) as u8);
                    cur_bytes.push((v >> 8) as u8);
                    pc = pc.wrapping_add(2);
                }
                max_end = max_end.max(pc);
            }
            Stmt::Ds(n) => {
                cur_bytes.resize(cur_bytes.len() + *n, 0);
                pc = pc.wrapping_add(*n as u16);
                max_end = max_end.max(pc);
            }
            Stmt::Inst { mnem, ops } => {
                let encoded = encode_inst(mnem, ops, pc, &syms, true)?;
                pc = pc.wrapping_add(encoded.len() as u16);
                cur_bytes.extend_from_slice(&encoded);
                max_end = max_end.max(pc);
            }
        }
    }
    flush(&mut cur_start, &mut cur_bytes, &mut segments);

    if max_end > code_limit.saturating_add(1) {
        return Err(format!(
            "MCS-51 Code ROM overflow: 0x{max_end:04X} exceeds 0x{:04X}",
            code_limit
        ));
    }

    let mut bin = vec![0xFFu8; max_end as usize];
    for (addr, data) in &segments {
        let start = *addr as usize;
        bin[start..start + data.len()].copy_from_slice(data);
    }
    let intel_hex = format_intel_hex(&segments);
    Ok(AssembledOutput {
        bin,
        intel_hex,
        symbols: syms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assembles_8051_core_and_mov_direct_src_dst_order() {
        let asm = "\
            .org 0x0000
            ljmp main
            .org 0x0020
        main:
            mov 0x30, #0x12
            mov 0x31, 0x30
            mov a, 0x31
            add a, #0x08
            mov dptr, #0x0100
            movx @dptr, a
            setb EA
            sjmp $
        ";
        let out = assemble_mcs51(asm, 0xEFFF).expect("assemble");
        // Verify `mov 0x31, 0x30` encodes `[0x85, 0x30, 0x31]`.
        let m_pos = out.symbols["main"] as usize;
        assert_eq!(&out.bin[m_pos..m_pos + 6], &[0x75, 0x30, 0x12, 0x85, 0x30, 0x31]);
        assert!(out.intel_hex.ends_with(":00000001FF\n"));
    }
}
