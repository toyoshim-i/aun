//! Built-in Z80 assembler with SDASZ80 (`.s`) compatibility and T-state accounting.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct AsmOutput {
    pub bytes: Vec<u8>,
    pub origin: u16,
    pub symbols: HashMap<String, u16>,
    pub total_tstates: u32,
    /// `(pc, tstates)` for every assembled instruction in address order.
    pub inst_tstates: Vec<(u16, u8)>,
}

#[derive(Debug, Clone)]
enum ParsedLine {
    Org(u16),
    Equ(String, i32),
    Label(String),
    LocalLabel(String),
    Db(Vec<String>),
    Dw(Vec<String>),
    Ds(u16),
    Inst { mnemonic: String, operands: Vec<String>, line_no: usize },
}

/// Assembles Z80 assembly (standard Zilog or SDASZ80 `.s` syntax) into machine code.
pub fn assemble_z80(src: &str, default_origin: u16) -> Result<AsmOutput, String> {
    let lines = parse_lines(src)?;
    let mut symbols: HashMap<String, u16> = HashMap::new();

    // Pass 1: resolve symbols and label addresses
    let mut pc: u16 = default_origin;
    let mut origin = default_origin;
    let mut origin_set = false;
    let mut current_scope = String::from("__root");

    for item in &lines {
        match item {
            ParsedLine::Org(addr) => {
                if !origin_set {
                    origin = *addr;
                    origin_set = true;
                }
                pc = *addr;
            }
            ParsedLine::Equ(name, val) => {
                symbols.insert(name.clone(), *val as u16);
            }
            ParsedLine::Label(name) => {
                current_scope = name.clone();
                symbols.insert(name.clone(), pc);
            }
            ParsedLine::LocalLabel(num) => {
                symbols.insert(format!("{current_scope}::{num}"), pc);
            }
            ParsedLine::Db(items) => pc = pc.wrapping_add(items.len() as u16),
            ParsedLine::Dw(items) => pc = pc.wrapping_add((items.len() * 2) as u16),
            ParsedLine::Ds(n) => pc = pc.wrapping_add(*n),
            ParsedLine::Inst {
                mnemonic,
                operands,
                line_no,
            } => {
                let (sz, _) = encode_inst(mnemonic, operands, pc, &symbols, &current_scope, false)
                    .map_err(|e| format!("line {line_no}: {e}"))?;
                pc = pc.wrapping_add(sz.len() as u16);
            }
        }
    }

    // Pass 2: emit bytes and count T-states
    let mut bytes = Vec::new();
    let mut total_tstates = 0u32;
    let mut inst_tstates = Vec::new();
    pc = origin;
    current_scope = String::from("__root");

    for item in &lines {
        match item {
            ParsedLine::Org(addr) => {
                if *addr > pc && !bytes.is_empty() {
                    bytes.resize(bytes.len() + (*addr - pc) as usize, 0);
                }
                pc = *addr;
            }
            ParsedLine::Equ(_, _) => {}
            ParsedLine::Label(name) => {
                current_scope = name.clone();
            }
            ParsedLine::LocalLabel(_) => {}
            ParsedLine::Db(items) => {
                for expr in items {
                    let v = eval_expr(expr, &symbols, &current_scope)?;
                    bytes.push((v & 0xFF) as u8);
                    pc = pc.wrapping_add(1);
                }
            }
            ParsedLine::Dw(items) => {
                for expr in items {
                    let v = eval_expr(expr, &symbols, &current_scope)? as u16;
                    bytes.extend_from_slice(&v.to_le_bytes());
                    pc = pc.wrapping_add(2);
                }
            }
            ParsedLine::Ds(n) => {
                bytes.resize(bytes.len() + *n as usize, 0);
                pc = pc.wrapping_add(*n);
            }
            ParsedLine::Inst {
                mnemonic,
                operands,
                line_no,
            } => {
                let (enc, ts) = encode_inst(mnemonic, operands, pc, &symbols, &current_scope, true)
                    .map_err(|e| format!("line {line_no}: {e}"))?;
                inst_tstates.push((pc, ts));
                pc = pc.wrapping_add(enc.len() as u16);
                total_tstates += ts as u32;
                bytes.extend_from_slice(&enc);
            }
        }
    }

    Ok(AsmOutput {
        bytes,
        origin,
        symbols,
        total_tstates,
        inst_tstates,
    })
}

fn parse_lines(src: &str) -> Result<Vec<ParsedLine>, String> {
    let mut out = Vec::new();
    for (idx, raw_line) in src.lines().enumerate() {
        let line_no = idx + 1;
        // Strip `;` comments
        let line = raw_line.split(';').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }

        // Handle `NAME = expr` or `NAME .equ expr` / `NAME equ expr`
        if let Some((lhs, rhs)) = line.split_once('=')
            && !lhs.trim().contains(' ')
            && !lhs.trim().starts_with('.')
        {
            let val = parse_const_num(rhs.trim())
                .ok_or_else(|| format!("line {line_no}: invalid equ value `{rhs}`"))?;
            out.push(ParsedLine::Equ(lhs.trim().to_string(), val));
            continue;
        }
        {
            let tokens: Vec<&str> = line.split_whitespace().collect();
            if tokens.len() >= 3 && (tokens[1].eq_ignore_ascii_case(".equ") || tokens[1].eq_ignore_ascii_case("equ")) {
                let rhs = tokens[2..].join(" ");
                let val = parse_const_num(&rhs)
                    .ok_or_else(|| format!("line {line_no}: invalid equ value `{rhs}`"))?;
                out.push(ParsedLine::Equ(tokens[0].to_string(), val));
                continue;
            }
        }

        // Handle label prefix `foo:` or `1$:`
        let mut rest = line;
        while let Some(colon) = rest.find(':') {
            let before = rest[..colon].trim();
            if before.is_empty() || before.contains(' ') || before.contains(',') {
                break;
            }
            if before.ends_with('$') {
                out.push(ParsedLine::LocalLabel(before.to_string()));
            } else {
                out.push(ParsedLine::Label(before.to_string()));
            }
            rest = rest[colon + 1..].trim();
        }
        if rest.is_empty() {
            continue;
        }

        let mut parts = rest.splitn(2, |c: char| c.is_whitespace());
        let head = parts.next().unwrap();
        let tail = parts.next().unwrap_or("").trim();
        let head_lower = head.to_ascii_lowercase();

        match head_lower.as_str() {
            ".area" | ".module" | ".globl" | ".local" => {
                // Recognized SDASZ80 metadata directives (handled at link layer)
            }
            ".org" | "org" => {
                let addr = parse_const_num(tail)
                    .ok_or_else(|| format!("line {line_no}: invalid .org address `{tail}`"))?;
                out.push(ParsedLine::Org(addr as u16));
            }
            ".db" | ".byte" | "db" | "defb" => {
                let items = split_operands(tail);
                out.push(ParsedLine::Db(items));
            }
            ".dw" | ".word" | "dw" | "defw" => {
                let items = split_operands(tail);
                out.push(ParsedLine::Dw(items));
            }
            ".ds" | ".blkb" | "ds" => {
                let n = parse_const_num(tail)
                    .ok_or_else(|| format!("line {line_no}: invalid .ds count `{tail}`"))?;
                out.push(ParsedLine::Ds(n as u16));
            }
            _ => {
                let operands = if tail.is_empty() {
                    Vec::new()
                } else {
                    split_operands(tail)
                };
                out.push(ParsedLine::Inst {
                    mnemonic: head_lower,
                    operands,
                    line_no,
                });
            }
        }
    }
    Ok(out)
}

fn split_operands(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut paren_depth = 0usize;
    for ch in s.chars() {
        match ch {
            '(' => {
                paren_depth += 1;
                cur.push(ch);
            }
            ')' => {
                paren_depth = paren_depth.saturating_sub(1);
                cur.push(ch);
            }
            ',' if paren_depth == 0 => {
                out.push(cur.trim().to_string());
                cur.clear();
            }
            _ => cur.push(ch),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

fn parse_const_num(s: &str) -> Option<i32> {
    let s = s.trim().trim_start_matches('#');
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        i32::from_str_radix(hex, 16).ok()
    } else if let Some(hex) = s.strip_prefix('$') {
        i32::from_str_radix(hex, 16).ok()
    } else if let Some(bin) = s.strip_prefix("0b").or_else(|| s.strip_prefix("0B")) {
        i32::from_str_radix(bin, 2).ok()
    } else {
        s.parse::<i32>().ok()
    }
}

fn eval_expr(
    raw: &str,
    symbols: &HashMap<String, u16>,
    scope: &str,
) -> Result<i32, String> {
    let s = raw.trim().trim_start_matches('#');
    if let Some(inner) = s.strip_prefix('<') {
        return Ok(eval_expr(inner, symbols, scope)? & 0xFF);
    }
    if let Some(inner) = s.strip_prefix('>') {
        return Ok((eval_expr(inner, symbols, scope)? >> 8) & 0xFF);
    }
    // Simple `sym + off` or `sym - off`
    if let Some((a, b)) = s.split_once('+') {
        return Ok(eval_expr(a, symbols, scope)? + eval_expr(b, symbols, scope)?);
    }
    if let Some(pos) = s.rfind('-')
        && pos > 0
    {
        return Ok(eval_expr(&s[..pos], symbols, scope)? - eval_expr(&s[pos + 1..], symbols, scope)?);
    }
    if let Some(n) = parse_const_num(s) {
        return Ok(n);
    }
    if s.ends_with('$') {
        if let Some(&addr) = symbols.get(&format!("{scope}::{s}")) {
            return Ok(addr as i32);
        }
    }
    if let Some(&addr) = symbols.get(s) {
        return Ok(addr as i32);
    }
    Err(format!("undefined symbol `{s}`"))
}

fn reg8_code(r: &str) -> Option<u8> {
    match r.to_ascii_lowercase().as_str() {
        "b" => Some(0),
        "c" => Some(1),
        "d" => Some(2),
        "e" => Some(3),
        "h" => Some(4),
        "l" => Some(5),
        "(hl)" => Some(6),
        "a" => Some(7),
        _ => None,
    }
}

fn reg16_sp_code(r: &str) -> Option<u8> {
    match r.to_ascii_lowercase().as_str() {
        "bc" => Some(0),
        "de" => Some(1),
        "hl" => Some(2),
        "sp" => Some(3),
        _ => None,
    }
}

fn reg16_af_code(r: &str) -> Option<u8> {
    match r.to_ascii_lowercase().as_str() {
        "bc" => Some(0),
        "de" => Some(1),
        "hl" => Some(2),
        "af" => Some(3),
        _ => None,
    }
}

fn cc_code(cc: &str) -> Option<u8> {
    match cc.to_ascii_lowercase().as_str() {
        "nz" => Some(0),
        "z" => Some(1),
        "nc" => Some(2),
        "c" => Some(3),
        "po" => Some(4),
        "pe" => Some(5),
        "p" => Some(6),
        "m" => Some(7),
        _ => None,
    }
}

fn strip_parens(s: &str) -> Option<&str> {
    let t = s.trim();
    t.strip_prefix('(')?.strip_suffix(')').map(str::trim)
}

fn eval_or_zero(
    expr: &str,
    symbols: &HashMap<String, u16>,
    scope: &str,
    pass2: bool,
) -> Result<i32, String> {
    if pass2 {
        eval_expr(expr, symbols, scope)
    } else {
        Ok(0)
    }
}

fn encode_inst(
    mnemonic: &str,
    ops: &[String],
    pc: u16,
    symbols: &HashMap<String, u16>,
    scope: &str,
    pass2: bool,
) -> Result<(Vec<u8>, u8), String> {
    match mnemonic {
        "nop" => Ok((vec![0x00], 4)),
        "halt" => Ok((vec![0x76], 4)),
        "di" => Ok((vec![0xF3], 4)),
        "ei" => Ok((vec![0xFB], 4)),
        "scf" => Ok((vec![0x37], 4)),
        "ccf" => Ok((vec![0x3F], 4)),
        "cpl" => Ok((vec![0x2F], 4)),
        "neg" => Ok((vec![0xED, 0x44], 8)),
        "exx" => Ok((vec![0xD9], 4)),
        "rlca" => Ok((vec![0x07], 4)),
        "rrca" => Ok((vec![0x0F], 4)),
        "rla" => Ok((vec![0x17], 4)),
        "rra" => Ok((vec![0x1F], 4)),
        "ldi" => Ok((vec![0xED, 0xA0], 16)),
        "ldd" => Ok((vec![0xED, 0xA8], 16)),
        "ldir" => Ok((vec![0xED, 0xB0], 21)),
        "lddr" => Ok((vec![0xED, 0xB8], 21)),
        "reti" => Ok((vec![0xED, 0x4D], 14)),
        "ret" => {
            if ops.is_empty() {
                Ok((vec![0xC9], 10))
            } else if let Some(cc) = cc_code(&ops[0]) {
                Ok((vec![0xC0 | (cc << 3)], 11))
            } else {
                Err(format!("invalid condition `{}` in RET", ops[0]))
            }
        }
        "ex" if ops.len() == 2 => {
            let a = ops[0].to_ascii_lowercase();
            let b = ops[1].to_ascii_lowercase();
            match (a.as_str(), b.as_str()) {
                ("de", "hl") => Ok((vec![0xEB], 4)),
                ("af", "af'") => Ok((vec![0x08], 4)),
                ("(sp)", "hl") => Ok((vec![0xE3], 19)),
                _ => Err(format!("unsupported EX {}, {}", ops[0], ops[1])),
            }
        }
        "push" if ops.len() == 1 => {
            let r = ops[0].to_ascii_lowercase();
            if r == "ix" {
                Ok((vec![0xDD, 0xE5], 15))
            } else if r == "iy" {
                Ok((vec![0xFD, 0xE5], 15))
            } else if let Some(c) = reg16_af_code(&r) {
                Ok((vec![0xC5 | (c << 4)], 11))
            } else {
                Err(format!("invalid PUSH operand `{}`", ops[0]))
            }
        }
        "pop" if ops.len() == 1 => {
            let r = ops[0].to_ascii_lowercase();
            if r == "ix" {
                Ok((vec![0xDD, 0xE1], 14))
            } else if r == "iy" {
                Ok((vec![0xFD, 0xE1], 14))
            } else if let Some(c) = reg16_af_code(&r) {
                Ok((vec![0xC1 | (c << 4)], 10))
            } else {
                Err(format!("invalid POP operand `{}`", ops[0]))
            }
        }
        "ld" if ops.len() == 2 => {
            let dst = ops[0].as_str();
            let src = ops[1].as_str();
            let dst_l = dst.to_ascii_lowercase();
            let src_l = src.to_ascii_lowercase();

            // LD SP, HL
            if dst_l == "sp" && src_l == "hl" {
                return Ok((vec![0xF9], 6));
            }
            // LD A, (BC)/(DE) and reverse
            if dst_l == "a" && src_l == "(bc)" {
                return Ok((vec![0x0A], 7));
            }
            if dst_l == "a" && src_l == "(de)" {
                return Ok((vec![0x1A], 7));
            }
            if dst_l == "(bc)" && src_l == "a" {
                return Ok((vec![0x02], 7));
            }
            if dst_l == "(de)" && src_l == "a" {
                return Ok((vec![0x12], 7));
            }
            // LD r, r' (including (HL))
            if let (Some(d), Some(s)) = (reg8_code(dst), reg8_code(src)) {
                if d == 6 && s == 6 {
                    return Err("LD (HL), (HL) is invalid (HALT)".into());
                }
                let ts = if d == 6 || s == 6 { 7 } else { 4 };
                return Ok((vec![0x40 | (d << 3) | s], ts));
            }
            // LD A, (nn)
            if dst_l == "a"
                && let Some(inner) = strip_parens(src)
            {
                let addr = eval_or_zero(inner, symbols, scope, pass2)? as u16;
                let [lo, hi] = addr.to_le_bytes();
                return Ok((vec![0x3A, lo, hi], 13));
            }
            // LD (nn), A
            if src_l == "a"
                && let Some(inner) = strip_parens(dst)
            {
                let addr = eval_or_zero(inner, symbols, scope, pass2)? as u16;
                let [lo, hi] = addr.to_le_bytes();
                return Ok((vec![0x32, lo, hi], 13));
            }
            // LD HL, (nn) / LD (nn), HL
            if dst_l == "hl"
                && let Some(inner) = strip_parens(src)
            {
                let addr = eval_or_zero(inner, symbols, scope, pass2)? as u16;
                let [lo, hi] = addr.to_le_bytes();
                return Ok((vec![0x2A, lo, hi], 16));
            }
            if src_l == "hl"
                && let Some(inner) = strip_parens(dst)
            {
                let addr = eval_or_zero(inner, symbols, scope, pass2)? as u16;
                let [lo, hi] = addr.to_le_bytes();
                return Ok((vec![0x22, lo, hi], 16));
            }
            // LD rr, (nn) / LD (nn), rr (ED prefix)
            if let Some(rr) = reg16_sp_code(&dst_l)
                && let Some(inner) = strip_parens(src)
            {
                let addr = eval_or_zero(inner, symbols, scope, pass2)? as u16;
                let [lo, hi] = addr.to_le_bytes();
                return Ok((vec![0xED, 0x4B | (rr << 4), lo, hi], 20));
            }
            if let Some(rr) = reg16_sp_code(&src_l)
                && let Some(inner) = strip_parens(dst)
            {
                let addr = eval_or_zero(inner, symbols, scope, pass2)? as u16;
                let [lo, hi] = addr.to_le_bytes();
                return Ok((vec![0xED, 0x43 | (rr << 4), lo, hi], 20));
            }
            // LD r, n
            if let Some(d) = reg8_code(dst) {
                let n = (eval_or_zero(src, symbols, scope, pass2)? & 0xFF) as u8;
                let ts = if d == 6 { 10 } else { 7 };
                return Ok((vec![0x06 | (d << 3), n], ts));
            }
            // LD rr, nn / LD IX, nn / LD IY, nn
            if dst_l == "ix" || dst_l == "iy" {
                let prefix = if dst_l == "ix" { 0xDD } else { 0xFD };
                let nn = eval_or_zero(src, symbols, scope, pass2)? as u16;
                let [lo, hi] = nn.to_le_bytes();
                return Ok((vec![prefix, 0x21, lo, hi], 14));
            }
            if let Some(rr) = reg16_sp_code(&dst_l) {
                let nn = eval_or_zero(src, symbols, scope, pass2)? as u16;
                let [lo, hi] = nn.to_le_bytes();
                return Ok((vec![0x01 | (rr << 4), lo, hi], 10));
            }
            Err(format!("unsupported LD {}, {}", dst, src))
        }
        "inc" if ops.len() == 1 => {
            if let Some(r) = reg8_code(&ops[0]) {
                let ts = if r == 6 { 11 } else { 4 };
                Ok((vec![0x04 | (r << 3)], ts))
            } else if let Some(rr) = reg16_sp_code(&ops[0]) {
                Ok((vec![0x03 | (rr << 4)], 6))
            } else {
                Err(format!("invalid INC operand `{}`", ops[0]))
            }
        }
        "dec" if ops.len() == 1 => {
            if let Some(r) = reg8_code(&ops[0]) {
                let ts = if r == 6 { 11 } else { 4 };
                Ok((vec![0x05 | (r << 3)], ts))
            } else if let Some(rr) = reg16_sp_code(&ops[0]) {
                Ok((vec![0x0B | (rr << 4)], 6))
            } else {
                Err(format!("invalid DEC operand `{}`", ops[0]))
            }
        }
        "add" if ops.len() == 2 => {
            if ops[0].eq_ignore_ascii_case("hl") {
                if let Some(rr) = reg16_sp_code(&ops[1]) {
                    return Ok((vec![0x09 | (rr << 4)], 11));
                }
            }
            if ops[0].eq_ignore_ascii_case("a") {
                return encode_alu8(0, &ops[1], symbols, scope, pass2);
            }
            Err(format!("unsupported ADD {}, {}", ops[0], ops[1]))
        }
        "adc" if ops.len() == 2 => {
            if ops[0].eq_ignore_ascii_case("hl") {
                if let Some(rr) = reg16_sp_code(&ops[1]) {
                    return Ok((vec![0xED, 0x4A | (rr << 4)], 15));
                }
            }
            if ops[0].eq_ignore_ascii_case("a") {
                return encode_alu8(1, &ops[1], symbols, scope, pass2);
            }
            Err(format!("unsupported ADC {}, {}", ops[0], ops[1]))
        }
        "sub" => {
            let rhs = if ops.len() == 2 && ops[0].eq_ignore_ascii_case("a") {
                &ops[1]
            } else if ops.len() == 1 {
                &ops[0]
            } else {
                return Err("SUB expects 1 or 2 operands".into());
            };
            encode_alu8(2, rhs, symbols, scope, pass2)
        }
        "sbc" if ops.len() == 2 => {
            if ops[0].eq_ignore_ascii_case("hl") {
                if let Some(rr) = reg16_sp_code(&ops[1]) {
                    return Ok((vec![0xED, 0x42 | (rr << 4)], 15));
                }
            }
            if ops[0].eq_ignore_ascii_case("a") {
                return encode_alu8(3, &ops[1], symbols, scope, pass2);
            }
            Err(format!("unsupported SBC {}, {}", ops[0], ops[1]))
        }
        "and" => {
            let rhs = if ops.len() == 2 { &ops[1] } else { &ops[0] };
            encode_alu8(4, rhs, symbols, scope, pass2)
        }
        "xor" => {
            let rhs = if ops.len() == 2 { &ops[1] } else { &ops[0] };
            encode_alu8(5, rhs, symbols, scope, pass2)
        }
        "or" => {
            let rhs = if ops.len() == 2 { &ops[1] } else { &ops[0] };
            encode_alu8(6, rhs, symbols, scope, pass2)
        }
        "cp" => {
            let rhs = if ops.len() == 2 { &ops[1] } else { &ops[0] };
            encode_alu8(7, rhs, symbols, scope, pass2)
        }
        "rlc" | "rrc" | "rl" | "rr" | "sla" | "sra" | "srl" if ops.len() == 1 => {
            let base = match mnemonic {
                "rlc" => 0x00,
                "rrc" => 0x08,
                "rl" => 0x10,
                "rr" => 0x18,
                "sla" => 0x20,
                "sra" => 0x28,
                "srl" => 0x38,
                _ => unreachable!(),
            };
            let r = reg8_code(&ops[0]).ok_or_else(|| format!("invalid shift reg `{}`", ops[0]))?;
            let ts = if r == 6 { 15 } else { 8 };
            Ok((vec![0xCB, base | r], ts))
        }
        "bit" | "res" | "set" if ops.len() == 2 => {
            let base = match mnemonic {
                "bit" => 0x40,
                "res" => 0x80,
                "set" => 0xC0,
                _ => unreachable!(),
            };
            let b = eval_or_zero(&ops[0], symbols, scope, pass2)?;
            if !(0..=7).contains(&b) {
                return Err(format!("bit index `{}` out of range 0..=7", ops[0]));
            }
            let r = reg8_code(&ops[1]).ok_or_else(|| format!("invalid bit reg `{}`", ops[1]))?;
            let ts = match (mnemonic, r == 6) {
                ("bit", true) => 12,
                (_, true) => 15,
                (_, false) => 8,
            };
            Ok((vec![0xCB, base | ((b as u8) << 3) | r], ts))
        }
        "jp" => {
            if ops.len() == 1 {
                if ops[0].eq_ignore_ascii_case("(hl)") {
                    return Ok((vec![0xE9], 4));
                }
                let nn = eval_or_zero(&ops[0], symbols, scope, pass2)? as u16;
                let [lo, hi] = nn.to_le_bytes();
                Ok((vec![0xC3, lo, hi], 10))
            } else if ops.len() == 2 {
                let cc = cc_code(&ops[0]).ok_or_else(|| format!("invalid JP cc `{}`", ops[0]))?;
                let nn = eval_or_zero(&ops[1], symbols, scope, pass2)? as u16;
                let [lo, hi] = nn.to_le_bytes();
                Ok((vec![0xC2 | (cc << 3), lo, hi], 10))
            } else {
                Err("invalid JP operands".into())
            }
        }
        "jr" => {
            let (opcode, target_expr) = if ops.len() == 1 {
                (0x18u8, &ops[0])
            } else if ops.len() == 2 {
                let cc = cc_code(&ops[0]).ok_or_else(|| format!("invalid JR cc `{}`", ops[0]))?;
                if cc >= 4 {
                    return Err(format!("JR does not support condition `{}`", ops[0]));
                }
                (0x20 | (cc << 3), &ops[1])
            } else {
                return Err("invalid JR operands".into());
            };
            let disp = if pass2 {
                let target = eval_expr(target_expr, symbols, scope)?;
                let rel = target - (pc as i32 + 2);
                if !(-128..=127).contains(&rel) {
                    return Err(format!("JR target `{target_expr}` out of range ({rel})"));
                }
                rel as i8 as u8
            } else {
                0
            };
            Ok((vec![opcode, disp], 12))
        }
        "djnz" if ops.len() == 1 => {
            let disp = if pass2 {
                let target = eval_expr(&ops[0], symbols, scope)?;
                let rel = target - (pc as i32 + 2);
                if !(-128..=127).contains(&rel) {
                    return Err(format!("DJNZ target `{}` out of range ({rel})", ops[0]));
                }
                rel as i8 as u8
            } else {
                0
            };
            Ok((vec![0x10, disp], 13))
        }
        "call" => {
            if ops.len() == 1 {
                let nn = eval_or_zero(&ops[0], symbols, scope, pass2)? as u16;
                let [lo, hi] = nn.to_le_bytes();
                Ok((vec![0xCD, lo, hi], 17))
            } else if ops.len() == 2 {
                let cc =
                    cc_code(&ops[0]).ok_or_else(|| format!("invalid CALL cc `{}`", ops[0]))?;
                let nn = eval_or_zero(&ops[1], symbols, scope, pass2)? as u16;
                let [lo, hi] = nn.to_le_bytes();
                Ok((vec![0xC4 | (cc << 3), lo, hi], 17))
            } else {
                Err("invalid CALL operands".into())
            }
        }
        "out" if ops.len() == 2 => {
            let port_inner =
                strip_parens(&ops[0]).ok_or_else(|| format!("invalid OUT port `{}`", ops[0]))?;
            if port_inner.eq_ignore_ascii_case("c") {
                let r = reg8_code(&ops[1])
                    .filter(|&c| c != 6)
                    .ok_or_else(|| format!("invalid OUT (C), `{}`", ops[1]))?;
                Ok((vec![0xED, 0x41 | (r << 3)], 12))
            } else if ops[1].eq_ignore_ascii_case("a") {
                let n = (eval_or_zero(port_inner, symbols, scope, pass2)? & 0xFF) as u8;
                Ok((vec![0xD3, n], 11))
            } else {
                Err(format!("unsupported OUT {}, {}", ops[0], ops[1]))
            }
        }
        "in" if ops.len() == 2 => {
            let port_inner =
                strip_parens(&ops[1]).ok_or_else(|| format!("invalid IN port `{}`", ops[1]))?;
            if port_inner.eq_ignore_ascii_case("c") {
                let r = reg8_code(&ops[0])
                    .filter(|&c| c != 6)
                    .ok_or_else(|| format!("invalid IN `{}`, (C)", ops[0]))?;
                Ok((vec![0xED, 0x40 | (r << 3)], 12))
            } else if ops[0].eq_ignore_ascii_case("a") {
                let n = (eval_or_zero(port_inner, symbols, scope, pass2)? & 0xFF) as u8;
                Ok((vec![0xDB, n], 11))
            } else {
                Err(format!("unsupported IN {}, {}", ops[0], ops[1]))
            }
        }
        _ => Err(format!("unsupported instruction `{mnemonic}`")),
    }
}

fn encode_alu8(
    op_idx: u8,
    rhs: &str,
    symbols: &HashMap<String, u16>,
    scope: &str,
    pass2: bool,
) -> Result<(Vec<u8>, u8), String> {
    if let Some(r) = reg8_code(rhs) {
        let ts = if r == 6 { 7 } else { 4 };
        Ok((vec![0x80 | (op_idx << 3) | r], ts))
    } else {
        let n = (eval_or_zero(rhs, symbols, scope, pass2)? & 0xFF) as u8;
        Ok((vec![0xC6 | (op_idx << 3), n], 7))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assemble_sdasz80_and_zilog() {
        let src = r#"
            .area _CODE
            .globl _hal_test
            .org 0x8000
        _hal_test:
            ld a, #0x42
            out (0xFE), a
        1$:
            dec a
            jr nz, 1$
            ld hl, (0xC000)
            jp (hl)
        "#;
        let out = assemble_z80(src, 0x8000).expect("assemble should succeed");
        assert_eq!(out.origin, 0x8000);
        assert_eq!(out.symbols.get("_hal_test"), Some(&0x8000));
        assert_eq!(
            out.bytes,
            vec![
                0x3E, 0x42, // ld a, 0x42
                0xD3, 0xFE, // out (0xFE), a
                0x3D, // dec a
                0x20, 0xFD, // jr nz, -3
                0x2A, 0x00, 0xC0, // ld hl, (0xC000)
                0xE9, // jp (hl)
            ]
        );
        assert_eq!(out.inst_tstates.len(), 6);
    }

    #[test]
    fn assemble_cb_rotates_bits_equ_and_block_ops() {
        let src = r#"
            PORT .equ 0x40
            .org 0x0100
        start:
            sra h
            rr l
            rl h
            bit 7, a
            set 0, (hl)
            res 3, b
            ldir
            in a, (PORT)
            ret
        "#;
        let out = assemble_z80(src, 0x0100).expect("assemble should succeed");
        assert_eq!(out.symbols.get("PORT"), Some(&0x0040));
        assert_eq!(
            out.bytes,
            vec![
                0xCB, 0x2C, // sra h
                0xCB, 0x1D, // rr l
                0xCB, 0x14, // rl h
                0xCB, 0x7F, // bit 7, a
                0xCB, 0xC6, // set 0, (hl)
                0xCB, 0x98, // res 3, b
                0xED, 0xB0, // ldir
                0xDB, 0x40, // in a, (0x40)
                0xC9, // ret
            ]
        );
    }
}
