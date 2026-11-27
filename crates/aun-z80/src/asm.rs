//! Built-in Z80 assembler with SDASZ80 (`.s`) compatibility and T-state accounting.

use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct AsmOutput {
    pub bytes: Vec<u8>,
    pub origin: u16,
    pub symbols: HashMap<String, u16>,
    pub total_tstates: u32,
    /// `(pc, tstates)` for every assembled instruction in address order.
    pub inst_tstates: Vec<(u16, u8)>,
    pub ram_end: u16,
}

#[derive(Debug, Clone)]
pub(crate) enum ParsedLine {
    ExtScope(Option<usize>),
    Globl(Vec<String>),
    Area(bool /* is_data */),
    Org { expr: String, line_no: usize },
    Equ { name: String, expr: String, line_no: usize, is_global: bool },
    Label { name: String, is_global: bool },
    LocalLabel(String),
    Db { items: Vec<String>, line_no: usize },
    Dw { items: Vec<String>, line_no: usize },
    Ds { expr: String, line_no: usize },
    Align { expr: String, line_no: usize },
    Inst { mnemonic: String, operands: Vec<String>, line_no: usize },
}

fn private_sym_key(ext_file: usize, name: &str) -> String {
    format!("__aun_ext{ext_file}::{name}")
}

fn ext_root_scope(ext_file: Option<usize>) -> String {
    match ext_file {
        Some(id) => format!("__aun_ext{id}::__root"),
        None => "__root".to_string(),
    }
}

/// Assembles Z80 assembly (standard Zilog or SDASZ80 `.s` syntax) into machine code,
/// with safe intra-function `JP` -> `JR` branch relaxation.
pub fn assemble_z80(src: &str, default_origin: u16) -> Result<AsmOutput, String> {
    assemble_z80_with_ram(src, default_origin, 0xC000)
}

fn resolve_deferred_equs(
    deferred: &mut Vec<(String, String, String, Option<usize>, u16, usize)>,
    eval_syms: &mut HashMap<String, i32>,
    fail_on_unresolved: bool,
) -> Result<(), String> {
    let mut pending = std::mem::take(deferred);
    let max_rounds = pending.len().saturating_add(1);
    for _ in 0..max_rounds {
        if pending.is_empty() {
            break;
        }
        let mut next = Vec::new();
        let mut progressed = false;
        for entry in pending {
            let (name, expr, scope, ext_file, cur_pc, line_no) = entry;
            match eval_expr_in(&expr, eval_syms, &scope, ext_file, cur_pc) {
                Ok(v) => {
                    eval_syms.insert(name, v);
                    progressed = true;
                }
                Err(err) => {
                    if fail_on_unresolved && !progressed && next.is_empty() {
                        // Keep collecting so we only fail after a full stuck round
                    }
                    next.push((name, expr, scope, ext_file, cur_pc, line_no));
                    let _ = err;
                }
            }
        }
        pending = next;
        if !progressed {
            break;
        }
    }
    if fail_on_unresolved {
        if let Some((_, expr, scope, ext_file, cur_pc, line_no)) = pending.into_iter().next() {
            eval_expr_in(&expr, eval_syms, &scope, ext_file, cur_pc)
                .map_err(|e| format!("line {line_no}: {e}"))?;
        }
    } else {
        *deferred = pending;
    }
    Ok(())
}

fn eval_ds_count(
    expr: &str,
    eval_syms: &HashMap<String, i32>,
    scope: &str,
    ext_file: Option<usize>,
    cur_pc: u16,
) -> Result<u16, String> {
    let n = eval_expr_in(expr, eval_syms, scope, ext_file, cur_pc)?;
    if n < 0 {
        return Err(format!("negative .ds count `{n}`"));
    }
    u16::try_from(n).map_err(|_| format!(".ds count `{n}` out of range"))
}

fn compute_align_pad(
    expr: &str,
    eval_syms: &HashMap<String, i32>,
    scope: &str,
    ext_file: Option<usize>,
    cur_pc: u16,
) -> Result<u16, String> {
    let n = eval_expr_in(expr, eval_syms, scope, ext_file, cur_pc)?;
    if n <= 0 {
        return Err(format!("alignment `{n}` must be positive"));
    }
    let align = n as u32;
    let rem = (cur_pc as u32) % align;
    let pad = if rem == 0 { 0 } else { align - rem };
    u16::try_from(pad).map_err(|_| format!("alignment padding `{pad}` out of range"))
}

/// Assembles Z80 assembly (including SDASZ80 `.area _DATA` RAM allocations starting at
/// `initial_ram_pc`) into machine code, returning the ROM binary and final `ram_end`.
pub fn assemble_z80_with_ram(
    src: &str,
    default_origin: u16,
    initial_ram_pc: u16,
) -> Result<AsmOutput, String> {
    let empty = HashSet::new();
    assemble_z80_with_ram_and_externs(src, default_origin, initial_ram_pc, &empty)
}

/// Same as `assemble_z80_with_ram`, additionally allowing un-exported single-colon labels
/// matching `implicit_global_externs` (Aun `extern "asm"` declarations) to remain global
/// when uniquely defined and not shadowed by an explicit `::` / `.globl` export.
pub fn assemble_z80_with_ram_and_externs(
    src: &str,
    default_origin: u16,
    initial_ram_pc: u16,
    implicit_global_externs: &HashSet<String>,
) -> Result<AsmOutput, String> {
    let mut lines = parse_lines(src)?;
    let mut eval_syms: HashMap<String, i32>;

    // Pre-scan external file `.globl` exports so single-colon `label:` and `NAME = expr`
    // definitions inside external `.s` files stay private unless exported.
    let mut file_globls: HashMap<usize, HashSet<String>> = HashMap::new();
    let mut ext_def_counts: HashMap<String, usize> = HashMap::new();
    {
        let mut scan_ext: Option<usize> = None;
        for item in &lines {
            match item {
                ParsedLine::ExtScope(f) => scan_ext = *f,
                ParsedLine::Globl(syms) => {
                    if let Some(fid) = scan_ext {
                        let entry = file_globls.entry(fid).or_default();
                        for s in syms {
                            entry.insert(s.clone());
                        }
                    }
                }
                ParsedLine::Label { name, .. } | ParsedLine::Equ { name, .. } => {
                    if scan_ext.is_some() {
                        *ext_def_counts.entry(name.clone()).or_default() += 1;
                    }
                }
                _ => {}
            }
        }
    }
    let mut explicit_global_defs: HashSet<String> = HashSet::new();
    {
        let mut scan_ext: Option<usize> = None;
        for item in &lines {
            match item {
                ParsedLine::ExtScope(f) => scan_ext = *f,
                ParsedLine::Label { name, is_global }
                | ParsedLine::Equ { name, is_global, .. } => {
                    if let Some(fid) = scan_ext {
                        if *is_global || file_globls.get(&fid).is_some_and(|g| g.contains(name)) {
                            explicit_global_defs.insert(name.clone());
                        }
                    } else {
                        explicit_global_defs.insert(name.clone());
                    }
                }
                _ => {}
            }
        }
    }
    let resolve_def_key = |name: &str, is_global: bool, cur_ext: Option<usize>| -> (String, bool) {
        let Some(fid) = cur_ext else {
            return (name.to_string(), true);
        };
        let is_exported = is_global
            || file_globls.get(&fid).is_some_and(|g| g.contains(name))
            || (implicit_global_externs.contains(name)
                && !explicit_global_defs.contains(name)
                && ext_def_counts.get(name).copied().unwrap_or(0) == 1);
        if is_exported {
            (name.to_string(), true)
        } else {
            (private_sym_key(fid, name), false)
        }
    };

    // Iterative Pass 1 + JP -> JR branch relaxation for intra-function basic-block labels.
    // Never relaxes `jp` targeting external/function symbols (preserves #[smc] 3-byte `jp` stubs!).
    let mut origin;
    loop {
        eval_syms = HashMap::new();
        let mut user_syms: HashSet<String> = HashSet::new();
        let mut deferred_equs: Vec<(String, String, String, Option<usize>, u16, usize)> =
            Vec::new();
        let mut pc: u16 = default_origin;
        let mut ram_pc: u16 = initial_ram_pc;
        let mut in_data_area = false;
        let mut emitted_any = false;
        origin = default_origin;
        let mut cur_ext_file: Option<usize> = None;
        let mut current_scope = ext_root_scope(None);

        for item in &lines {
            match item {
                ParsedLine::ExtScope(fid) => {
                    cur_ext_file = *fid;
                    current_scope = ext_root_scope(*fid);
                }
                ParsedLine::Globl(_) => {}
                ParsedLine::Area(is_data) => {
                    in_data_area = *is_data;
                }
                ParsedLine::Org { expr, line_no } => {
                    let cur_pc = if in_data_area { ram_pc } else { pc };
                    let addr = eval_expr_in(expr, &eval_syms, &current_scope, cur_ext_file, cur_pc)
                        .map_err(|e| format!("line {line_no}: {e}"))? as u16;
                    if in_data_area {
                        ram_pc = addr;
                    } else {
                        if !emitted_any {
                            origin = addr;
                        }
                        pc = addr;
                    }
                }
                ParsedLine::Equ {
                    name,
                    expr,
                    line_no,
                    is_global,
                } => {
                    let (sym_key, is_pub_sym) = resolve_def_key(name, *is_global, cur_ext_file);
                    if is_pub_sym {
                        user_syms.insert(name.clone());
                    }
                    let cur_pc = if in_data_area { ram_pc } else { pc };
                    match eval_expr_in(expr, &eval_syms, &current_scope, cur_ext_file, cur_pc) {
                        Ok(val) => {
                            eval_syms.insert(sym_key, val);
                        }
                        Err(_) => {
                            deferred_equs.push((
                                sym_key,
                                expr.clone(),
                                current_scope.clone(),
                                cur_ext_file,
                                cur_pc,
                                *line_no,
                            ));
                        }
                    }
                }
                ParsedLine::Label { name, is_global } => {
                    let (sym_key, is_pub_sym) = resolve_def_key(name, *is_global, cur_ext_file);
                    if is_pub_sym {
                        user_syms.insert(name.clone());
                    }
                    current_scope = sym_key.clone();
                    let addr = if in_data_area { ram_pc } else { pc };
                    eval_syms.insert(sym_key, addr as i32);
                }
                ParsedLine::LocalLabel(num) => {
                    let addr = if in_data_area { ram_pc } else { pc };
                    eval_syms.insert(format!("{current_scope}::{num}"), addr as i32);
                }
                ParsedLine::Db { items, .. } => {
                    if in_data_area {
                        ram_pc = ram_pc.wrapping_add(items.len() as u16);
                    } else {
                        if !items.is_empty() {
                            emitted_any = true;
                        }
                        pc = pc.wrapping_add(items.len() as u16);
                    }
                }
                ParsedLine::Dw { items, .. } => {
                    let sz = (items.len() * 2) as u16;
                    if in_data_area {
                        ram_pc = ram_pc.wrapping_add(sz);
                    } else {
                        if sz > 0 {
                            emitted_any = true;
                        }
                        pc = pc.wrapping_add(sz);
                    }
                }
                ParsedLine::Ds { expr, line_no } => {
                    let cur_pc = if in_data_area { ram_pc } else { pc };
                    let n = eval_ds_count(expr, &eval_syms, &current_scope, cur_ext_file, cur_pc)
                        .map_err(|e| format!("line {line_no}: {e}"))?;
                    if in_data_area {
                        ram_pc = ram_pc.wrapping_add(n);
                    } else {
                        if n > 0 {
                            emitted_any = true;
                        }
                        pc = pc.wrapping_add(n);
                    }
                }
                ParsedLine::Align { expr, line_no } => {
                    let cur_pc = if in_data_area { ram_pc } else { pc };
                    let pad = compute_align_pad(expr, &eval_syms, &current_scope, cur_ext_file, cur_pc)
                        .map_err(|e| format!("line {line_no}: {e}"))?;
                    if in_data_area {
                        ram_pc = ram_pc.wrapping_add(pad);
                    } else {
                        if pad > 0 {
                            emitted_any = true;
                        }
                        pc = pc.wrapping_add(pad);
                    }
                }
                ParsedLine::Inst {
                    mnemonic,
                    operands,
                    line_no,
                } => {
                    let (sz, _) = encode_inst(
                        mnemonic,
                        operands,
                        pc,
                        &eval_syms,
                        &current_scope,
                        cur_ext_file,
                        false,
                    )
                    .map_err(|e| format!("line {line_no}: {e}"))?;
                    if in_data_area {
                        ram_pc = ram_pc.wrapping_add(sz.len() as u16);
                    } else {
                        emitted_any = true;
                        pc = pc.wrapping_add(sz.len() as u16);
                    }
                }
            }
        }

        resolve_deferred_equs(&mut deferred_equs, &mut eval_syms, false)?;

        // Synthesize SDASZ80-compatible RAM/INITIALIZER/BSS bounds symbols when not explicitly defined by user `.s`
        let ram_start = eval_syms
            .get("__aun_ram")
            .map(|&v| v as u16)
            .unwrap_or(initial_ram_pc);
        let init_ram_end = eval_syms
            .get("__aun_init_ram_end")
            .map(|&v| (v as u16).max(ram_start))
            .unwrap_or(ram_start);
        let init_len = init_ram_end.saturating_sub(ram_start);
        let init_rom_addr = eval_syms.get("__aun_init_rom").copied().unwrap_or(0);
        let ram_end = ram_pc;
        let ram_size = ram_end.saturating_sub(ram_start);
        let data_size = ram_end.saturating_sub(init_ram_end);
        for (sym, val) in [
            ("__aun_ram_end", ram_end as i32),
            ("__aun_ram_size", ram_size as i32),
            ("s__INITIALIZED", ram_start as i32),
            ("s__INITIALIZER", init_rom_addr),
            ("l__INITIALIZER", init_len as i32),
            ("s__DATA", init_ram_end as i32),
            ("l__DATA", data_size as i32),
            ("s__BSS", init_ram_end as i32),
            ("l__BSS", data_size as i32),
        ] {
            if !user_syms.contains(sym) {
                eval_syms.entry(sym.to_string()).or_insert(val);
            }
        }

        resolve_deferred_equs(&mut deferred_equs, &mut eval_syms, true)?;

        // Try relaxing eligible `jp [nz|z|nc|c,] <bb_label>` -> `jr`
        let mut relaxed_any = false;
        let mut pc2: u16 = origin;
        let mut ram_pc2: u16 = initial_ram_pc;
        let mut in_data2 = false;
        let mut cur_ext2: Option<usize> = None;
        let mut scope2 = ext_root_scope(None);
        for item in &mut lines {
            match item {
                ParsedLine::ExtScope(fid) => {
                    cur_ext2 = *fid;
                    scope2 = ext_root_scope(*fid);
                }
                ParsedLine::Globl(_) => {}
                ParsedLine::Area(is_data) => in_data2 = *is_data,
                ParsedLine::Org { expr, .. } => {
                    let cur_pc = if in_data2 { ram_pc2 } else { pc2 };
                    let addr = eval_expr_in(expr, &eval_syms, &scope2, cur_ext2, cur_pc)? as u16;
                    if in_data2 {
                        ram_pc2 = addr;
                    } else {
                        pc2 = addr;
                    }
                }
                ParsedLine::Equ { .. } => {}
                ParsedLine::Label { name, is_global } => {
                    let (sym_key, _) = resolve_def_key(name, *is_global, cur_ext2);
                    scope2 = sym_key;
                }
                ParsedLine::LocalLabel(_) => {}
                ParsedLine::Db { items, .. } => {
                    if in_data2 {
                        ram_pc2 = ram_pc2.wrapping_add(items.len() as u16);
                    } else {
                        pc2 = pc2.wrapping_add(items.len() as u16);
                    }
                }
                ParsedLine::Dw { items, .. } => {
                    let sz = (items.len() * 2) as u16;
                    if in_data2 {
                        ram_pc2 = ram_pc2.wrapping_add(sz);
                    } else {
                        pc2 = pc2.wrapping_add(sz);
                    }
                }
                ParsedLine::Ds { expr, .. } => {
                    let cur_pc = if in_data2 { ram_pc2 } else { pc2 };
                    let n = eval_ds_count(expr, &eval_syms, &scope2, cur_ext2, cur_pc)?;
                    if in_data2 {
                        ram_pc2 = ram_pc2.wrapping_add(n);
                    } else {
                        pc2 = pc2.wrapping_add(n);
                    }
                }
                ParsedLine::Align { expr, .. } => {
                    let cur_pc = if in_data2 { ram_pc2 } else { pc2 };
                    let pad = compute_align_pad(expr, &eval_syms, &scope2, cur_ext2, cur_pc)?;
                    if in_data2 {
                        ram_pc2 = ram_pc2.wrapping_add(pad);
                    } else {
                        pc2 = pc2.wrapping_add(pad);
                    }
                }
                ParsedLine::Inst {
                    mnemonic,
                    operands,
                    ..
                } => {
                    let old_len =
                        encode_inst(mnemonic, operands, pc2, &eval_syms, &scope2, cur_ext2, false)
                            .map(|(b, _)| b.len() as u16)
                            .unwrap_or(0);
                    if !in_data2 && mnemonic == "jp" {
                        let target_opt = if operands.len() == 1
                            && !operands[0].eq_ignore_ascii_case("(hl)")
                        {
                            Some(operands[0].as_str())
                        } else if operands.len() == 2
                            && matches!(
                                operands[0].to_ascii_lowercase().as_str(),
                                "nz" | "z" | "nc" | "c"
                            )
                        {
                            Some(operands[1].as_str())
                        } else {
                            None
                        };
                        if let Some(target) = target_opt
                            && !scope2.contains("_disp_")
                            && !scope2.starts_with("ovl_")
                            && target
                                .rsplit_once("_bb")
                                .is_some_and(|(_, d)| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
                            && let Some(&taddr) = eval_syms.get(target)
                        {
                            let rel = taddr - (pc2 as i32 + 2);
                            if (-120..=120).contains(&rel) {
                                *mnemonic = "jr".to_string();
                                relaxed_any = true;
                            }
                        }
                    }
                    if in_data2 {
                        ram_pc2 = ram_pc2.wrapping_add(old_len);
                    } else {
                        pc2 = pc2.wrapping_add(old_len);
                    }
                }
            }
        }
        if !relaxed_any {
            break;
        }
    }

    // Pass 2: emit bytes and count T-states
    let mut bytes = Vec::new();
    let mut total_tstates = 0u32;
    let mut inst_tstates = Vec::new();
    let mut pc = origin;
    let mut ram_pc = initial_ram_pc;
    let mut in_data_area = false;
    let mut cur_ext_file: Option<usize> = None;
    let mut current_scope = ext_root_scope(None);

    for item in &lines {
        match item {
            ParsedLine::ExtScope(fid) => {
                cur_ext_file = *fid;
                current_scope = ext_root_scope(*fid);
            }
            ParsedLine::Globl(_) => {}
            ParsedLine::Area(is_data) => in_data_area = *is_data,
            ParsedLine::Org { expr, line_no } => {
                let cur_pc = if in_data_area { ram_pc } else { pc };
                let addr = eval_expr_in(expr, &eval_syms, &current_scope, cur_ext_file, cur_pc)
                    .map_err(|e| format!("line {line_no}: {e}"))? as u16;
                if in_data_area {
                    ram_pc = addr;
                } else {
                    if addr > pc && !bytes.is_empty() {
                        bytes.resize(bytes.len() + (addr - pc) as usize, 0);
                    }
                    pc = addr;
                }
            }
            ParsedLine::Equ { .. } => {}
            ParsedLine::Label { name, is_global } => {
                let (sym_key, _) = resolve_def_key(name, *is_global, cur_ext_file);
                current_scope = sym_key;
            }
            ParsedLine::LocalLabel(_) => {}
            ParsedLine::Db { items, line_no } => {
                if in_data_area {
                    ram_pc = ram_pc.wrapping_add(items.len() as u16);
                } else {
                    for expr in items {
                        let v = eval_expr_in(expr, &eval_syms, &current_scope, cur_ext_file, pc)
                            .map_err(|e| format!("line {line_no}: {e}"))?;
                        bytes.push((v & 0xFF) as u8);
                        pc = pc.wrapping_add(1);
                    }
                }
            }
            ParsedLine::Dw { items, line_no } => {
                if in_data_area {
                    ram_pc = ram_pc.wrapping_add((items.len() * 2) as u16);
                } else {
                    for expr in items {
                        let v = eval_expr_in(expr, &eval_syms, &current_scope, cur_ext_file, pc)
                            .map_err(|e| format!("line {line_no}: {e}"))?
                            as u16;
                        bytes.extend_from_slice(&v.to_le_bytes());
                        pc = pc.wrapping_add(2);
                    }
                }
            }
            ParsedLine::Ds { expr, line_no } => {
                let cur_pc = if in_data_area { ram_pc } else { pc };
                let n = eval_ds_count(expr, &eval_syms, &current_scope, cur_ext_file, cur_pc)
                    .map_err(|e| format!("line {line_no}: {e}"))?;
                if in_data_area {
                    ram_pc = ram_pc.wrapping_add(n);
                } else {
                    bytes.resize(bytes.len() + n as usize, 0);
                    pc = pc.wrapping_add(n);
                }
            }
            ParsedLine::Align { expr, line_no } => {
                let cur_pc = if in_data_area { ram_pc } else { pc };
                let pad = compute_align_pad(expr, &eval_syms, &current_scope, cur_ext_file, cur_pc)
                    .map_err(|e| format!("line {line_no}: {e}"))?;
                if in_data_area {
                    ram_pc = ram_pc.wrapping_add(pad);
                } else {
                    bytes.resize(bytes.len() + pad as usize, 0);
                    pc = pc.wrapping_add(pad);
                }
            }
            ParsedLine::Inst {
                mnemonic,
                operands,
                line_no,
            } => {
                let (enc, ts) = encode_inst(
                    mnemonic,
                    operands,
                    pc,
                    &eval_syms,
                    &current_scope,
                    cur_ext_file,
                    true,
                )
                .map_err(|e| format!("line {line_no}: {e}"))?;
                if in_data_area {
                    ram_pc = ram_pc.wrapping_add(enc.len() as u16);
                } else {
                    inst_tstates.push((pc, ts));
                    pc = pc.wrapping_add(enc.len() as u16);
                    total_tstates += ts as u32;
                    bytes.extend_from_slice(&enc);
                }
            }
        }
    }

    let symbols: HashMap<String, u16> = eval_syms
        .into_iter()
        .filter(|(k, _)| !k.starts_with("__aun_ext"))
        .map(|(k, v)| (k, v as u16))
        .collect();

    Ok(AsmOutput {
        bytes,
        origin,
        symbols,
        total_tstates,
        inst_tstates,
        ram_end: ram_pc,
    })
}

fn strip_asm_comment(raw: &str) -> &str {
    let bytes = raw.as_bytes();
    let mut in_quotes = false;
    let mut escaped = false;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if escaped {
            escaped = false;
            i += 1;
            continue;
        }
        if in_quotes {
            if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_quotes = false;
            }
            i += 1;
            continue;
        }
        match b {
            b'"' => in_quotes = true,
            b'\'' => {
                let prev_is_ident =
                    i > 0 && (bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
                if !prev_is_ident {
                    let mut j = i + 1;
                    let mut esc = false;
                    while j < bytes.len() {
                        if esc {
                            esc = false;
                        } else if bytes[j] == b'\\' {
                            esc = true;
                        } else if bytes[j] == b'\'' {
                            i = j;
                            break;
                        }
                        j += 1;
                    }
                }
            }
            b';' => return raw[..i].trim(),
            _ => {}
        }
        i += 1;
    }
    raw.trim()
}

fn parse_ascii_bytes(tail: &str, nul_term: bool) -> Result<Vec<String>, String> {
    let t = tail.trim();
    let Some(rest) = t.strip_prefix('"') else {
        return Err(format!("expected quoted string literal, found `{tail}`"));
    };
    let mut bytes = Vec::new();
    let mut chars = rest.chars();
    let mut closed = false;
    while let Some(c) = chars.next() {
        match c {
            '"' => {
                closed = true;
                break;
            }
            '\\' => {
                let Some(esc) = chars.next() else {
                    return Err("unterminated escape in string literal".to_string());
                };
                let b = match esc {
                    'n' => b'\n',
                    'r' => b'\r',
                    't' => b'\t',
                    '0' => 0,
                    '\\' => b'\\',
                    '"' => b'"',
                    '\'' => b'\'',
                    other => other as u8,
                };
                bytes.push(b.to_string());
            }
            other => {
                let mut buf = [0u8; 4];
                for &b in other.encode_utf8(&mut buf).as_bytes() {
                    bytes.push(b.to_string());
                }
            }
        }
    }
    if !closed {
        return Err("unterminated string literal".to_string());
    }
    if nul_term {
        bytes.push("0".to_string());
    }
    Ok(bytes)
}

pub(crate) fn parse_lines(src: &str) -> Result<Vec<ParsedLine>, String> {
    let mut out = Vec::new();
    let mut defined_syms: HashSet<String> = HashSet::new();
    let mut file_defined_syms: HashMap<usize, HashSet<String>> = HashMap::new();
    let mut cur_ext_file: Option<usize> = None;
    let mut cond_stack: Vec<(bool /* active */, bool /* parent_active */)> = Vec::new();

    for (idx, raw_line) in src.lines().enumerate() {
        let line_no = idx + 1;
        let raw_trimmed = raw_line.trim();
        if let Some(rest) = raw_trimmed.strip_prefix("; @aun:ext_begin ") {
            if let Ok(fid) = rest.trim().parse::<usize>() {
                cur_ext_file = Some(fid);
                out.push(ParsedLine::ExtScope(Some(fid)));
                continue;
            }
        } else if raw_trimmed == "; @aun:ext_end" {
            cur_ext_file = None;
            out.push(ParsedLine::ExtScope(None));
            continue;
        }
        let line = strip_asm_comment(raw_line);
        if line.is_empty() {
            continue;
        }

        let cur_active = cond_stack.last().map(|&(a, _)| a).unwrap_or(true);
        {
            let mut parts = line.splitn(2, |c: char| c.is_whitespace());
            let head = parts.next().unwrap();
            let tail = parts.next().unwrap_or("").trim();
            let is_def = |sym: &str| -> bool {
                !sym.is_empty()
                    && (defined_syms.contains(sym)
                        || cur_ext_file.is_some_and(|fid| {
                            file_defined_syms.get(&fid).is_some_and(|s| s.contains(sym))
                        }))
            };
            match head.to_ascii_lowercase().as_str() {
                ".ifdef" => {
                    let sym = tail.split_whitespace().next().unwrap_or("");
                    let cond = is_def(sym);
                    cond_stack.push((cur_active && cond, cur_active));
                    continue;
                }
                ".ifndef" => {
                    let sym = tail.split_whitespace().next().unwrap_or("");
                    let cond = !sym.is_empty() && !is_def(sym);
                    cond_stack.push((cur_active && cond, cur_active));
                    continue;
                }
                ".else" => {
                    let Some((active, parent_active)) = cond_stack.last_mut() else {
                        return Err(format!("line {line_no}: unmatched `.else`"));
                    };
                    *active = *parent_active && !*active;
                    continue;
                }
                ".endif" => {
                    if cond_stack.pop().is_none() {
                        return Err(format!("line {line_no}: unmatched `.endif`"));
                    }
                    continue;
                }
                _ => {}
            }
        }
        if !cur_active {
            continue;
        }

        // Handle `NAME == expr` (global), `NAME = expr` (file-local in external `.s`), or `NAME .equ expr` / `NAME equ expr`
        if let Some((lhs, rhs)) = line.split_once("==")
            && !lhs.trim().is_empty()
            && !lhs.trim().contains(' ')
            && !lhs.trim().starts_with('.')
            && !lhs.trim().contains('=')
            && !rhs.trim().is_empty()
        {
            let name = lhs.trim().to_string();
            defined_syms.insert(name.clone());
            out.push(ParsedLine::Equ {
                name,
                expr: rhs.trim().to_string(),
                line_no,
                is_global: true,
            });
            continue;
        }
        if let Some((lhs, rhs)) = line.split_once('=')
            && !lhs.trim().contains(' ')
            && !lhs.trim().starts_with('.')
            && !rhs.trim().is_empty()
        {
            let name = lhs.trim().to_string();
            let is_global = cur_ext_file.is_none();
            if is_global {
                defined_syms.insert(name.clone());
            } else if let Some(fid) = cur_ext_file {
                file_defined_syms.entry(fid).or_default().insert(name.clone());
            }
            out.push(ParsedLine::Equ {
                name,
                expr: rhs.trim().to_string(),
                line_no,
                is_global,
            });
            continue;
        }
        {
            let mut p1 = line.splitn(2, |c: char| c.is_whitespace());
            let first = p1.next().unwrap_or("");
            let after_first = p1.next().unwrap_or("").trim_start();
            let mut p2 = after_first.splitn(2, |c: char| c.is_whitespace());
            let second = p2.next().unwrap_or("");
            let rhs = p2.next().unwrap_or("").trim();
            if !first.is_empty()
                && !first.starts_with('.')
                && (second.eq_ignore_ascii_case(".equ") || second.eq_ignore_ascii_case("equ"))
                && !rhs.is_empty()
            {
                let name = first.to_string();
                let is_global = cur_ext_file.is_none();
                if is_global {
                    defined_syms.insert(name.clone());
                } else if let Some(fid) = cur_ext_file {
                    file_defined_syms.entry(fid).or_default().insert(name.clone());
                }
                out.push(ParsedLine::Equ {
                    name,
                    expr: rhs.to_string(),
                    line_no,
                    is_global,
                });
                continue;
            }
        }

        // Handle label prefix `foo:`, `foo::`, or `1$:`
        let mut rest = line;
        while let Some(colon) = rest.find(':') {
            let before = rest[..colon].trim();
            if before.is_empty()
                || before.contains(' ')
                || before.contains(',')
                || before.contains('"')
                || before.starts_with('.')
            {
                break;
            }
            let after_colon = &rest[colon + 1..];
            let is_double_colon = after_colon.starts_with(':');
            if before.ends_with('$') {
                out.push(ParsedLine::LocalLabel(before.to_string()));
            } else {
                out.push(ParsedLine::Label {
                    name: before.to_string(),
                    is_global: is_double_colon || cur_ext_file.is_none(),
                });
            }
            rest = after_colon.trim_start_matches(':').trim();
        }
        if rest.is_empty() {
            continue;
        }

        let mut parts = rest.splitn(2, |c: char| c.is_whitespace());
        let head = parts.next().unwrap();
        let tail = parts.next().unwrap_or("").trim();
        let head_lower = head.to_ascii_lowercase();

        match head_lower.as_str() {
            ".area" => {
                let area_name = tail
                    .split(|c: char| c.is_whitespace() || c == '(')
                    .next()
                    .unwrap_or("")
                    .to_ascii_lowercase();
                let is_data = matches!(
                    area_name.as_str(),
                    "_data" | "_bss" | "_initialized" | "_dabs"
                );
                out.push(ParsedLine::Area(is_data));
            }
            ".globl" => {
                let syms: Vec<String> = tail
                    .split([',', ' ', '\t'])
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect();
                if !syms.is_empty() {
                    out.push(ParsedLine::Globl(syms));
                }
            }
            ".module" | ".local" | ".optsdcc" => {
                // Recognized SDASZ80 metadata directives (handled at link layer)
            }
            ".org" | "org" => {
                if tail.is_empty() {
                    return Err(format!("line {line_no}: missing .org address"));
                }
                out.push(ParsedLine::Org {
                    expr: tail.to_string(),
                    line_no,
                });
            }
            ".db" | ".byte" | "db" | "defb" => {
                let items = split_operands(tail);
                out.push(ParsedLine::Db { items, line_no });
            }
            ".ascii" => {
                let items = parse_ascii_bytes(tail, false)
                    .map_err(|e| format!("line {line_no}: {e}"))?;
                out.push(ParsedLine::Db { items, line_no });
            }
            ".asciz" | ".ascis" => {
                let items = parse_ascii_bytes(tail, true)
                    .map_err(|e| format!("line {line_no}: {e}"))?;
                out.push(ParsedLine::Db { items, line_no });
            }
            ".dw" | ".word" | "dw" | "defw" => {
                let items = split_operands(tail);
                out.push(ParsedLine::Dw { items, line_no });
            }
            ".ds" | ".blkb" | "ds" => {
                if tail.is_empty() {
                    return Err(format!("line {line_no}: missing .ds count"));
                }
                out.push(ParsedLine::Ds {
                    expr: tail.to_string(),
                    line_no,
                });
            }
            ".align" | ".bndry" => {
                if tail.is_empty() {
                    return Err(format!("line {line_no}: missing {head} boundary"));
                }
                out.push(ParsedLine::Align {
                    expr: tail.to_string(),
                    line_no,
                });
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
    if !cond_stack.is_empty() {
        return Err("unclosed `.ifdef`/`.ifndef` conditional block".to_string());
    }
    Ok(out)
}

fn split_operands(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut paren_depth = 0usize;
    let mut in_double_quote = false;
    let mut escaped = false;
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if escaped {
            cur.push(ch);
            escaped = false;
            i += 1;
            continue;
        }
        if in_double_quote {
            cur.push(ch);
            if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_double_quote = false;
            }
            i += 1;
            continue;
        }
        match ch {
            '"' => {
                in_double_quote = true;
                cur.push(ch);
            }
            '\'' => {
                let prev_is_ident =
                    i > 0 && (chars[i - 1].is_ascii_alphanumeric() || chars[i - 1] == '_');
                if !prev_is_ident {
                    let mut j = i + 1;
                    let mut esc = false;
                    let mut found_close = None;
                    while j < chars.len() {
                        if esc {
                            esc = false;
                        } else if chars[j] == '\\' {
                            esc = true;
                        } else if chars[j] == '\'' {
                            found_close = Some(j);
                            break;
                        }
                        j += 1;
                    }
                    if let Some(end) = found_close {
                        for &c in &chars[i..=end] {
                            cur.push(c);
                        }
                        i = end + 1;
                        continue;
                    }
                }
                cur.push(ch);
            }
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
        i += 1;
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

pub(crate) fn parse_const_num(s: &str) -> Option<i32> {
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

struct ExprParser<'a> {
    src: &'a str,
    bytes: &'a [u8],
    pos: usize,
    symbols: &'a HashMap<String, i32>,
    scope: &'a str,
    ext_file: Option<usize>,
    pc: u16,
}

impl<'a> ExprParser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.bytes.get(self.pos).copied()
    }

    fn eat_str(&mut self, tok: &str) -> bool {
        self.skip_ws();
        if self.src[self.pos..].starts_with(tok) {
            self.pos += tok.len();
            true
        } else {
            false
        }
    }

    fn parse_expr(&mut self) -> Result<i32, String> {
        self.parse_bitor()
    }

    fn parse_bitor(&mut self) -> Result<i32, String> {
        let mut lhs = self.parse_bitxor()?;
        while self.eat_str("|") {
            let rhs = self.parse_bitxor()?;
            lhs |= rhs;
        }
        Ok(lhs)
    }

    fn parse_bitxor(&mut self) -> Result<i32, String> {
        let mut lhs = self.parse_bitand()?;
        while self.eat_str("^") {
            let rhs = self.parse_bitand()?;
            lhs ^= rhs;
        }
        Ok(lhs)
    }

    fn parse_bitand(&mut self) -> Result<i32, String> {
        let mut lhs = self.parse_shift()?;
        while self.eat_str("&") {
            let rhs = self.parse_shift()?;
            lhs &= rhs;
        }
        Ok(lhs)
    }

    fn parse_shift(&mut self) -> Result<i32, String> {
        let mut lhs = self.parse_add()?;
        loop {
            if self.eat_str("<<") {
                let rhs = self.parse_add()?;
                if rhs < 0 {
                    return Err(format!("negative shift count `{rhs}`"));
                }
                lhs = if rhs >= 32 { 0 } else { lhs.wrapping_shl(rhs as u32) };
            } else if self.eat_str(">>") {
                let rhs = self.parse_add()?;
                if rhs < 0 {
                    return Err(format!("negative shift count `{rhs}`"));
                }
                lhs = if rhs >= 32 {
                    if lhs < 0 { -1 } else { 0 }
                } else {
                    lhs.wrapping_shr(rhs as u32)
                };
            } else {
                break;
            }
        }
        Ok(lhs)
    }

    fn parse_add(&mut self) -> Result<i32, String> {
        let mut lhs = self.parse_mul()?;
        loop {
            if self.eat_str("+") {
                let rhs = self.parse_mul()?;
                lhs = lhs.wrapping_add(rhs);
            } else if self.eat_str("-") {
                let rhs = self.parse_mul()?;
                lhs = lhs.wrapping_sub(rhs);
            } else {
                break;
            }
        }
        Ok(lhs)
    }

    fn parse_mul(&mut self) -> Result<i32, String> {
        let mut lhs = self.parse_unary()?;
        loop {
            if self.eat_str("*") {
                let rhs = self.parse_unary()?;
                lhs = lhs.wrapping_mul(rhs);
            } else if self.eat_str("/") {
                let rhs = self.parse_unary()?;
                if rhs == 0 {
                    return Err("division by zero in expression".to_string());
                }
                lhs = lhs.wrapping_div(rhs);
            } else if self.eat_str("%") {
                let rhs = self.parse_unary()?;
                if rhs == 0 {
                    return Err("modulo by zero in expression".to_string());
                }
                lhs = lhs.wrapping_rem(rhs);
            } else {
                break;
            }
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<i32, String> {
        self.skip_ws();
        if self.src[self.pos..].starts_with("<<") || self.src[self.pos..].starts_with(">>") {
            return self.parse_primary();
        }
        match self.peek() {
            Some(b'-') => {
                self.pos += 1;
                Ok(self.parse_unary()?.wrapping_neg())
            }
            Some(b'+') => {
                self.pos += 1;
                self.parse_unary()
            }
            Some(b'~') => {
                self.pos += 1;
                Ok(!self.parse_unary()?)
            }
            Some(b'<') => {
                self.pos += 1;
                Ok(self.parse_unary()? & 0xFF)
            }
            Some(b'>') => {
                self.pos += 1;
                Ok((self.parse_unary()? >> 8) & 0xFF)
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<i32, String> {
        let Some(b) = self.peek() else {
            return Err(format!("unexpected end of expression `{}`", self.src));
        };
        match b {
            b'(' => {
                self.pos += 1;
                let v = self.parse_expr()?;
                if !self.eat_str(")") {
                    return Err(format!("unclosed `(` in expression `{}`", self.src));
                }
                Ok(v)
            }
            b'\'' => self.parse_char_literal(),
            b'$' => {
                self.pos += 1;
                let start = self.pos;
                while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_hexdigit() {
                    self.pos += 1;
                }
                if self.pos == start {
                    return Err(format!("invalid `$` hex literal in `{}`", self.src));
                }
                parse_radix_i32(&self.src[start..self.pos], 16)
            }
            b'.' => {
                let next_is_ident = self
                    .bytes
                    .get(self.pos + 1)
                    .is_some_and(|c| c.is_ascii_alphanumeric() || *c == b'_' || *c == b'$');
                if next_is_ident {
                    return Err(format!(
                        "unexpected `.` prefix in expression `{}`",
                        &self.src[self.pos..]
                    ));
                }
                self.pos += 1;
                Ok(self.pc as i32)
            }
            b'0'..=b'9' => {
                let start = self.pos;
                while self.pos < self.bytes.len()
                    && (self.bytes[self.pos].is_ascii_alphanumeric()
                        || self.bytes[self.pos] == b'_'
                        || self.bytes[self.pos] == b'$')
                {
                    self.pos += 1;
                }
                let tok = &self.src[start..self.pos];
                if tok.ends_with('$') {
                    let key = format!("{}::{tok}", self.scope);
                    return self
                        .symbols
                        .get(&key)
                        .copied()
                        .ok_or_else(|| format!("undefined local label `{tok}`"));
                }
                parse_number_token(tok)
            }
            c if c.is_ascii_alphabetic() || c == b'_' => {
                let start = self.pos;
                while self.pos < self.bytes.len()
                    && (self.bytes[self.pos].is_ascii_alphanumeric()
                        || self.bytes[self.pos] == b'_'
                        || self.bytes[self.pos] == b'$')
                {
                    self.pos += 1;
                }
                let ident = &self.src[start..self.pos];
                if let Some(fid) = self.ext_file
                    && let Some(&val) = self.symbols.get(&private_sym_key(fid, ident))
                {
                    return Ok(val);
                }
                self.symbols
                    .get(ident)
                    .copied()
                    .ok_or_else(|| format!("undefined symbol `{ident}`"))
            }
            _ => Err(format!(
                "unexpected token `{}` in expression `{}`",
                &self.src[self.pos..],
                self.src
            )),
        }
    }

    fn parse_char_literal(&mut self) -> Result<i32, String> {
        self.pos += 1; // skip opening `'`
        let rest = &self.src[self.pos..];
        let mut chars = rest.char_indices();
        let Some((_, ch)) = chars.next() else {
            return Err(format!("unterminated character literal in `{}`", self.src));
        };
        let val = if ch == '\\' {
            let Some((_, esc)) = chars.next() else {
                return Err(format!("unterminated character escape in `{}`", self.src));
            };
            match esc {
                'n' => b'\n' as i32,
                'r' => b'\r' as i32,
                't' => b'\t' as i32,
                '0' => 0,
                '\\' => b'\\' as i32,
                '\'' => b'\'' as i32,
                '"' => b'"' as i32,
                other => other as i32,
            }
        } else {
            ch as i32
        };
        let Some((close_off, '\'')) = chars.next() else {
            return Err(format!("unterminated character literal in `{}`", self.src));
        };
        self.pos += close_off + 1;
        Ok(val)
    }
}

fn parse_radix_i32(digits: &str, radix: u32) -> Result<i32, String> {
    if digits.is_empty() {
        return Err("empty numeric literal".to_string());
    }
    if let Ok(v) = u32::from_str_radix(digits, radix) {
        return Ok(v as i32);
    }
    Err(format!("invalid base-{radix} integer `{digits}`"))
}

fn parse_number_token(tok: &str) -> Result<i32, String> {
    if let Some(hex) = tok.strip_prefix("0x").or_else(|| tok.strip_prefix("0X")) {
        return parse_radix_i32(hex, 16);
    }
    if let Some(bin) = tok.strip_prefix("0b").or_else(|| tok.strip_prefix("0B")) {
        return parse_radix_i32(bin, 2);
    }
    if let Some(hex) = tok.strip_suffix('h').or_else(|| tok.strip_suffix('H'))
        && !hex.is_empty()
        && hex.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return parse_radix_i32(hex, 16);
    }
    tok.parse::<i64>()
        .ok()
        .and_then(|v| {
            if (i32::MIN as i64..=u32::MAX as i64).contains(&v) {
                Some(v as i32)
            } else {
                None
            }
        })
        .ok_or_else(|| format!("invalid integer literal `{tok}`"))
}

fn eval_expr_in(
    raw: &str,
    symbols: &HashMap<String, i32>,
    scope: &str,
    ext_file: Option<usize>,
    pc: u16,
) -> Result<i32, String> {
    let trimmed = raw.trim();
    let s = trimmed.strip_prefix('#').unwrap_or(trimmed).trim();
    let mut parser = ExprParser {
        src: s,
        bytes: s.as_bytes(),
        pos: 0,
        symbols,
        scope,
        ext_file,
        pc,
    };
    let val = parser.parse_expr()?;
    parser.skip_ws();
    if parser.pos != parser.bytes.len() {
        return Err(format!(
            "unexpected trailing token `{}` in expression `{raw}`",
            &parser.src[parser.pos..]
        ));
    }
    Ok(val)
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

fn has_outer_matching_parens(s: &str) -> bool {
    if !s.starts_with('(') || !s.ends_with(')') {
        return false;
    }
    let mut depth = 0usize;
    for (idx, ch) in s.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return idx == s.len() - 1;
                }
            }
            _ => {}
        }
    }
    false
}

fn parse_indexed_ix_iy(
    op: &str,
    symbols: &HashMap<String, i32>,
    scope: &str,
    ext_file: Option<usize>,
    pc: u16,
    pass2: bool,
) -> Result<Option<(u8, i8)>, String> {
    let t = op.trim();
    // Supports `(ix + d)`, `(ix - d)`, `(ix)`, `d(ix)`, and `iy` equivalents
    if has_outer_matching_parens(t) {
        let inner = t[1..t.len() - 1].trim();
        let lower = inner.to_ascii_lowercase();
        let prefix_opt = if lower == "ix"
            || lower
                .strip_prefix("ix")
                .is_some_and(|r| r.starts_with(|c: char| c.is_whitespace() || c == '+' || c == '-'))
        {
            Some((0xDDu8, inner[2..].trim()))
        } else if lower == "iy"
            || lower
                .strip_prefix("iy")
                .is_some_and(|r| r.starts_with(|c: char| c.is_whitespace() || c == '+' || c == '-'))
        {
            Some((0xFDu8, inner[2..].trim()))
        } else {
            None
        };
        if let Some((prefix, rest)) = prefix_opt {
            if rest.is_empty() {
                return Ok(Some((prefix, 0)));
            }
            let v = eval_or_zero(rest, symbols, scope, ext_file, pc, pass2)?;
            if pass2 && !(-128..=127).contains(&v) {
                return Err(format!(
                    "index displacement `{op}` out of range -128..=127 ({v})"
                ));
            }
            return Ok(Some((prefix, v as i8)));
        }
        return Ok(None);
    }

    if let Some(open_idx) = t.rfind('(')
        && t.ends_with(')')
    {
        let reg_part = t[open_idx + 1..t.len() - 1].trim().to_ascii_lowercase();
        let prefix = match reg_part.as_str() {
            "ix" => 0xDDu8,
            "iy" => 0xFDu8,
            _ => return Ok(None),
        };
        let off_str = t[..open_idx].trim();
        let v = if off_str.is_empty() {
            0
        } else {
            eval_or_zero(off_str, symbols, scope, ext_file, pc, pass2)?
        };
        if pass2 && !(-128..=127).contains(&v) {
            return Err(format!(
                "index displacement `{op}` out of range -128..=127 ({v})"
            ));
        }
        return Ok(Some((prefix, v as i8)));
    }
    Ok(None)
}

fn strip_parens(s: &str) -> Option<&str> {
    let t = s.trim();
    if has_outer_matching_parens(t) {
        Some(t[1..t.len() - 1].trim())
    } else {
        None
    }
}

fn eval_or_zero(
    expr: &str,
    symbols: &HashMap<String, i32>,
    scope: &str,
    ext_file: Option<usize>,
    pc: u16,
    pass2: bool,
) -> Result<i32, String> {
    if pass2 {
        eval_expr_in(expr, symbols, scope, ext_file, pc)
    } else {
        Ok(0)
    }
}

pub(crate) fn encode_inst(
    mnemonic: &str,
    ops: &[String],
    pc: u16,
    symbols: &HashMap<String, i32>,
    scope: &str,
    ext_file: Option<usize>,
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
        "retn" => Ok((vec![0xED, 0x45], 14)),
        "im" if ops.len() == 1 => {
            let mode = eval_or_zero(&ops[0], symbols, scope, ext_file, pc, pass2)?;
            let op2 = match mode {
                0 => 0x46,
                1 => 0x56,
                2 => 0x5E,
                _ => {
                    return Err(format!(
                        "invalid IM mode `{}` (expected 0, 1, or 2)",
                        ops[0]
                    ));
                }
            };
            Ok((vec![0xED, op2], 8))
        }
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

            // Interrupt / refresh register loads: LD I/R, A and LD A, I/R
            match (dst_l.as_str(), src_l.as_str()) {
                ("i", "a") => return Ok((vec![0xED, 0x47], 9)),
                ("r", "a") => return Ok((vec![0xED, 0x4F], 9)),
                ("a", "i") => return Ok((vec![0xED, 0x57], 9)),
                ("a", "r") => return Ok((vec![0xED, 0x5F], 9)),
                _ => {}
            }
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
            // LD r, (IX/IY + d) and LD (IX/IY + d), r and LD (IX/IY + d), n
            if let Some((prefix, disp)) =
                parse_indexed_ix_iy(src, symbols, scope, ext_file, pc, pass2)?
                && let Some(d) = reg8_code(dst)
                && d != 6
            {
                return Ok((vec![prefix, 0x46 | (d << 3), disp as u8], 19));
            }
            if let Some((prefix, disp)) =
                parse_indexed_ix_iy(dst, symbols, scope, ext_file, pc, pass2)?
            {
                if let Some(s) = reg8_code(src)
                    && s != 6
                {
                    return Ok((vec![prefix, 0x70 | s, disp as u8], 19));
                }
                let n = (eval_or_zero(src, symbols, scope, ext_file, pc, pass2)? & 0xFF) as u8;
                return Ok((vec![prefix, 0x36, disp as u8, n], 19));
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
                let addr = eval_or_zero(inner, symbols, scope, ext_file, pc, pass2)? as u16;
                let [lo, hi] = addr.to_le_bytes();
                return Ok((vec![0x3A, lo, hi], 13));
            }
            // LD (nn), A
            if src_l == "a"
                && let Some(inner) = strip_parens(dst)
            {
                let addr = eval_or_zero(inner, symbols, scope, ext_file, pc, pass2)? as u16;
                let [lo, hi] = addr.to_le_bytes();
                return Ok((vec![0x32, lo, hi], 13));
            }
            // LD HL/IX/IY, (nn) / LD (nn), HL/IX/IY
            if (dst_l == "hl" || dst_l == "ix" || dst_l == "iy")
                && let Some(inner) = strip_parens(src)
            {
                let addr = eval_or_zero(inner, symbols, scope, ext_file, pc, pass2)? as u16;
                let [lo, hi] = addr.to_le_bytes();
                return match dst_l.as_str() {
                    "ix" => Ok((vec![0xDD, 0x2A, lo, hi], 20)),
                    "iy" => Ok((vec![0xFD, 0x2A, lo, hi], 20)),
                    _ => Ok((vec![0x2A, lo, hi], 16)),
                };
            }
            if (src_l == "hl" || src_l == "ix" || src_l == "iy")
                && let Some(inner) = strip_parens(dst)
            {
                let addr = eval_or_zero(inner, symbols, scope, ext_file, pc, pass2)? as u16;
                let [lo, hi] = addr.to_le_bytes();
                return match src_l.as_str() {
                    "ix" => Ok((vec![0xDD, 0x22, lo, hi], 20)),
                    "iy" => Ok((vec![0xFD, 0x22, lo, hi], 20)),
                    _ => Ok((vec![0x22, lo, hi], 16)),
                };
            }
            // LD rr, (nn) / LD (nn), rr (ED prefix)
            if let Some(rr) = reg16_sp_code(&dst_l)
                && let Some(inner) = strip_parens(src)
            {
                let addr = eval_or_zero(inner, symbols, scope, ext_file, pc, pass2)? as u16;
                let [lo, hi] = addr.to_le_bytes();
                return Ok((vec![0xED, 0x4B | (rr << 4), lo, hi], 20));
            }
            if let Some(rr) = reg16_sp_code(&src_l)
                && let Some(inner) = strip_parens(dst)
            {
                let addr = eval_or_zero(inner, symbols, scope, ext_file, pc, pass2)? as u16;
                let [lo, hi] = addr.to_le_bytes();
                return Ok((vec![0xED, 0x43 | (rr << 4), lo, hi], 20));
            }
            // LD r, n
            if let Some(d) = reg8_code(dst) {
                let n = (eval_or_zero(src, symbols, scope, ext_file, pc, pass2)? & 0xFF) as u8;
                let ts = if d == 6 { 10 } else { 7 };
                return Ok((vec![0x06 | (d << 3), n], ts));
            }
            // LD rr, nn / LD IX, nn / LD IY, nn
            if dst_l == "ix" || dst_l == "iy" {
                let prefix = if dst_l == "ix" { 0xDD } else { 0xFD };
                let nn = eval_or_zero(src, symbols, scope, ext_file, pc, pass2)? as u16;
                let [lo, hi] = nn.to_le_bytes();
                return Ok((vec![prefix, 0x21, lo, hi], 14));
            }
            if let Some(rr) = reg16_sp_code(&dst_l) {
                let nn = eval_or_zero(src, symbols, scope, ext_file, pc, pass2)? as u16;
                let [lo, hi] = nn.to_le_bytes();
                return Ok((vec![0x01 | (rr << 4), lo, hi], 10));
            }
            Err(format!("unsupported LD {}, {}", dst, src))
        }
        "inc" if ops.len() == 1 => {
            if let Some((prefix, d)) =
                parse_indexed_ix_iy(&ops[0], symbols, scope, ext_file, pc, pass2)?
            {
                Ok((vec![prefix, 0x34, d as u8], 23))
            } else if ops[0].eq_ignore_ascii_case("ix") {
                Ok((vec![0xDD, 0x23], 10))
            } else if ops[0].eq_ignore_ascii_case("iy") {
                Ok((vec![0xFD, 0x23], 10))
            } else if let Some(r) = reg8_code(&ops[0]) {
                let ts = if r == 6 { 11 } else { 4 };
                Ok((vec![0x04 | (r << 3)], ts))
            } else if let Some(rr) = reg16_sp_code(&ops[0]) {
                Ok((vec![0x03 | (rr << 4)], 6))
            } else {
                Err(format!("invalid INC operand `{}`", ops[0]))
            }
        }
        "dec" if ops.len() == 1 => {
            if let Some((prefix, d)) =
                parse_indexed_ix_iy(&ops[0], symbols, scope, ext_file, pc, pass2)?
            {
                Ok((vec![prefix, 0x35, d as u8], 23))
            } else if ops[0].eq_ignore_ascii_case("ix") {
                Ok((vec![0xDD, 0x2B], 10))
            } else if ops[0].eq_ignore_ascii_case("iy") {
                Ok((vec![0xFD, 0x2B], 10))
            } else if let Some(r) = reg8_code(&ops[0]) {
                let ts = if r == 6 { 11 } else { 4 };
                Ok((vec![0x05 | (r << 3)], ts))
            } else if let Some(rr) = reg16_sp_code(&ops[0]) {
                Ok((vec![0x0B | (rr << 4)], 6))
            } else {
                Err(format!("invalid DEC operand `{}`", ops[0]))
            }
        }
        "add" if ops.len() == 2 => {
            if ops[0].eq_ignore_ascii_case("hl")
                && let Some(rr) = reg16_sp_code(&ops[1])
            {
                return Ok((vec![0x09 | (rr << 4)], 11));
            }
            if ops[0].eq_ignore_ascii_case("ix") || ops[0].eq_ignore_ascii_case("iy") {
                let prefix = if ops[0].eq_ignore_ascii_case("ix") {
                    0xDD
                } else {
                    0xFD
                };
                let rr = match ops[1].to_ascii_lowercase().as_str() {
                    "bc" => Some(0),
                    "de" => Some(1),
                    r if r.eq_ignore_ascii_case(&ops[0]) => Some(2),
                    "sp" => Some(3),
                    _ => None,
                };
                if let Some(rp) = rr {
                    return Ok((vec![prefix, 0x09 | (rp << 4)], 15));
                }
            }
            if ops[0].eq_ignore_ascii_case("a") {
                return encode_alu8(0, &ops[1], pc, symbols, scope, ext_file, pass2);
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
                return encode_alu8(1, &ops[1], pc, symbols, scope, ext_file, pass2);
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
            encode_alu8(2, rhs, pc, symbols, scope, ext_file, pass2)
        }
        "sbc" if ops.len() == 2 => {
            if ops[0].eq_ignore_ascii_case("hl") {
                if let Some(rr) = reg16_sp_code(&ops[1]) {
                    return Ok((vec![0xED, 0x42 | (rr << 4)], 15));
                }
            }
            if ops[0].eq_ignore_ascii_case("a") {
                return encode_alu8(3, &ops[1], pc, symbols, scope, ext_file, pass2);
            }
            Err(format!("unsupported SBC {}, {}", ops[0], ops[1]))
        }
        "and" => {
            let rhs = if ops.len() == 2 { &ops[1] } else { &ops[0] };
            encode_alu8(4, rhs, pc, symbols, scope, ext_file, pass2)
        }
        "xor" => {
            let rhs = if ops.len() == 2 { &ops[1] } else { &ops[0] };
            encode_alu8(5, rhs, pc, symbols, scope, ext_file, pass2)
        }
        "or" => {
            let rhs = if ops.len() == 2 { &ops[1] } else { &ops[0] };
            encode_alu8(6, rhs, pc, symbols, scope, ext_file, pass2)
        }
        "cp" => {
            let rhs = if ops.len() == 2 { &ops[1] } else { &ops[0] };
            encode_alu8(7, rhs, pc, symbols, scope, ext_file, pass2)
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
            if let Some((prefix, disp)) =
                parse_indexed_ix_iy(&ops[0], symbols, scope, ext_file, pc, pass2)?
            {
                return Ok((vec![prefix, 0xCB, disp as u8, base | 6], 23));
            }
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
            let b = eval_or_zero(&ops[0], symbols, scope, ext_file, pc, pass2)?;
            if !(0..=7).contains(&b) {
                return Err(format!("bit index `{}` out of range 0..=7", ops[0]));
            }
            if let Some((prefix, disp)) =
                parse_indexed_ix_iy(&ops[1], symbols, scope, ext_file, pc, pass2)?
            {
                let ts = if mnemonic == "bit" { 20 } else { 23 };
                return Ok((
                    vec![prefix, 0xCB, disp as u8, base | ((b as u8) << 3) | 6],
                    ts,
                ));
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
                if ops[0].eq_ignore_ascii_case("(ix)") {
                    return Ok((vec![0xDD, 0xE9], 8));
                }
                if ops[0].eq_ignore_ascii_case("(iy)") {
                    return Ok((vec![0xFD, 0xE9], 8));
                }
                let nn = eval_or_zero(&ops[0], symbols, scope, ext_file, pc, pass2)? as u16;
                let [lo, hi] = nn.to_le_bytes();
                Ok((vec![0xC3, lo, hi], 10))
            } else if ops.len() == 2 {
                let cc = cc_code(&ops[0]).ok_or_else(|| format!("invalid JP cc `{}`", ops[0]))?;
                let nn = eval_or_zero(&ops[1], symbols, scope, ext_file, pc, pass2)? as u16;
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
                let target = eval_expr_in(target_expr, symbols, scope, ext_file, pc)?;
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
                let target = eval_expr_in(&ops[0], symbols, scope, ext_file, pc)?;
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
                let nn = eval_or_zero(&ops[0], symbols, scope, ext_file, pc, pass2)? as u16;
                let [lo, hi] = nn.to_le_bytes();
                Ok((vec![0xCD, lo, hi], 17))
            } else if ops.len() == 2 {
                let cc =
                    cc_code(&ops[0]).ok_or_else(|| format!("invalid CALL cc `{}`", ops[0]))?;
                let nn = eval_or_zero(&ops[1], symbols, scope, ext_file, pc, pass2)? as u16;
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
                let n = (eval_or_zero(port_inner, symbols, scope, ext_file, pc, pass2)? & 0xFF) as u8;
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
                let n = (eval_or_zero(port_inner, symbols, scope, ext_file, pc, pass2)? & 0xFF) as u8;
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
    pc: u16,
    symbols: &HashMap<String, i32>,
    scope: &str,
    ext_file: Option<usize>,
    pass2: bool,
) -> Result<(Vec<u8>, u8), String> {
    if let Some((prefix, d)) = parse_indexed_ix_iy(rhs, symbols, scope, ext_file, pc, pass2)? {
        Ok((vec![prefix, 0x86 | (op_idx << 3), d as u8], 19))
    } else if let Some(r) = reg8_code(rhs) {
        let ts = if r == 6 { 7 } else { 4 };
        Ok((vec![0x80 | (op_idx << 3) | r], ts))
    } else {
        let n = (eval_or_zero(rhs, symbols, scope, ext_file, pc, pass2)? & 0xFF) as u8;
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

    #[test]
    fn sdasz80_data_area_ascii_and_ifdef() {
        let src = r#"
            .module ext_test
            .optsdcc -mz80 sdcccall(1)
            FEATURE_A = 1
            .area _DATA
            _ext_buf::
                .ds 4
            _ext_flag::
                .ds 1
            .area _CODE
            .globl _ext_fn
            _ext_fn::
            .ifdef FEATURE_A
                ld a, #0x11
            .else
                ld a, #0x99
            .endif
            .ifndef UNDEF_SYM
                ld (_ext_flag), a
            .endif
                ret
            _ext_msg::
                .ascii "A;"
                .asciz "Z"
        "#;
        let out = assemble_z80_with_ram(src, 0x8000, 0xC100).expect("assemble should succeed");
        assert_eq!(out.symbols.get("_ext_buf"), Some(&0xC100));
        assert_eq!(out.symbols.get("_ext_flag"), Some(&0xC104));
        assert_eq!(out.ram_end, 0xC105);
        assert_eq!(out.symbols.get("_ext_fn"), Some(&0x8000));
        assert_eq!(
            out.bytes,
            vec![
                0x3E, 0x11, // ld a, #0x11
                0x32, 0x04, 0xC1, // ld (0xC104), a
                0xC9, // ret
                b'A', b';', b'Z', 0, // .ascii "A;" + .asciz "Z"
            ]
        );
    }

    #[test]
    fn assemble_inc_dec_ix_iy() {
        let src = r#"
            .org 0x8000
        start:
            inc ix
            dec ix
            inc iy
            dec iy
            INC IX
            DEC IY
        "#;
        let out = assemble_z80(src, 0x8000).expect("inc/dec ix/iy should assemble");
        assert_eq!(
            out.bytes,
            vec![
                0xDD, 0x23, // inc ix
                0xDD, 0x2B, // dec ix
                0xFD, 0x23, // inc iy
                0xFD, 0x2B, // dec iy
                0xDD, 0x23, // INC IX
                0xFD, 0x2B, // DEC IY
            ]
        );
        assert_eq!(out.inst_tstates.len(), 6);
        assert_eq!(out.total_tstates, 60);
    }

    #[test]
    fn assemble_sdasz80_const_expressions_and_pc_dot() {
        let src = r#"
            CH_SIZE = 16
            MASK = ~0x0F & 0xFF
            .org 0x8000
        _chain_stub:
            jr .+2
            ld bc, #(_chain_stub_end - _chain_stub)
            ld a, #-6 & 0xFF
            ld d, #(0 * 80)
            ld e, #'A' + 1
            ld h, #>0x1234
            ld l, #<0x1234
            ld a, #(1 << 4) | (0b101 ^ 0x03)
            ld b, #$0a
            ld c, #0fh
        _chain_stub_end:
        table:
            .dw .-table, _chain_stub_end - _chain_stub
            .db MASK, 10 / 2 + 7 % 4, 'Z' >> 1
        buf:
            .ds 8 * CH_SIZE
        "#;
        let out = assemble_z80(src, 0x8000).expect("const exprs should assemble");
        // jr .+2 (2) + ld bc (3) + ld a (2) + ld d (2) + ld e (2) + ld h (2) + ld l (2)
        // + ld a (2) + ld b (2) + ld c (2) = 21 bytes
        let stub_len = out.symbols["_chain_stub_end"] - out.symbols["_chain_stub"];
        assert_eq!(stub_len, 21);
        assert_eq!(out.symbols["buf"], 0x8000 + 21 + 4 + 3);
        assert_eq!(out.bytes.len(), 21 + 4 + 3 + 128);
        assert_eq!(
            &out.bytes[..28],
            &[
                0x18, 0x00, // jr .+2
                0x01, 21, 0x00, // ld bc, #(_chain_stub_end - _chain_stub)
                0x3E, 0xFA, // ld a, #-6 & 0xFF
                0x16, 0x00, // ld d, #(0 * 80)
                0x1E, b'B', // ld e, #'A' + 1
                0x26, 0x12, // ld h, #>0x1234
                0x2E, 0x34, // ld l, #<0x1234
                0x3E, 0x16, // ld a, #(1 << 4) | (0b101 ^ 0x03)
                0x06, 0x0A, // ld b, #$0a
                0x0E, 0x0F, // ld c, #0fh
                0x00, 0x00, // .dw .-table (at table, so 0)
                21, 0x00, // .dw _chain_stub_end - _chain_stub
                0xF0, // MASK = ~0x0F & 0xFF
                8, // 10 / 2 + 7 % 4 = 5 + 3
                b'Z' >> 1, // 'Z' >> 1 = 45
            ]
        );
    }

    #[test]
    fn assemble_ir_im_and_indexed_cb_ops() {
        let src = r#"
            .org 0x0100
        start:
            ld i, a
            ld r, a
            ld a, i
            ld a, r
            im 0
            im 1
            im 2
            bit 2, 5(ix)
            set 3, -4(iy)
            res 7, (ix + 1)
            srl 2(ix)
            rlc (iy - 3)
            ret
        "#;
        let out = assemble_z80(src, 0x0100).expect("new z80 ops should assemble");
        assert_eq!(
            out.bytes,
            vec![
                0xED, 0x47, // ld i, a
                0xED, 0x4F, // ld r, a
                0xED, 0x57, // ld a, i
                0xED, 0x5F, // ld a, r
                0xED, 0x46, // im 0
                0xED, 0x56, // im 1
                0xED, 0x5E, // im 2
                0xDD, 0xCB, 0x05, 0x56, // bit 2, 5(ix)
                0xFD, 0xCB, 0xFC, 0xDE, // set 3, -4(iy)
                0xDD, 0xCB, 0x01, 0xBE, // res 7, (ix + 1)
                0xDD, 0xCB, 0x02, 0x3E, // srl 2(ix)
                0xFD, 0xCB, 0xFD, 0x06, // rlc (iy - 3)
                0xC9, // ret
            ]
        );
    }

    #[test]
    fn assemble_header_abs_bndry_align_and_ram_bss_symbols() {
        let src = r#"
            __aun_ram = 0xC000
            .area _HEADER (ABS)
            .org 0x0100
        init:
            ld hl, #s__DATA
            ld bc, #l__DATA
            ld de, #s__BSS
            ld hl, #l__BSS
            ld hl, #__aun_ram_end
            ld bc, #__aun_ram_size
            .org 0x0200
        im2_vec:
            .dw 0x0100
            .area _CODE
        lut_pad_start:
            .db 0xAA
            .bndry 256
        lut256:
            .db 0x55
            .align 256
        after_align:
            .db 0x77
            .area _DATA
        ext_d:
            .ds 4
            .area _BSS
        ext_b:
            .ds 6
        "#;
        let out = assemble_z80_with_ram(src, 0x0100, 0xC010).expect("header/align/ram should assemble");
        assert_eq!(out.origin, 0x0100);
        assert_eq!(out.symbols.get("init"), Some(&0x0100));
        assert_eq!(out.symbols.get("im2_vec"), Some(&0x0200));
        assert_eq!(out.symbols.get("lut256"), Some(&0x0300));
        assert_eq!(out.symbols.get("after_align"), Some(&0x0400));
        assert_eq!(out.symbols.get("ext_d"), Some(&0xC010));
        assert_eq!(out.symbols.get("ext_b"), Some(&0xC014));
        assert_eq!(out.ram_end, 0xC01A);
        assert_eq!(out.symbols.get("__aun_ram_end"), Some(&0xC01A));
        assert_eq!(out.symbols.get("__aun_ram_size"), Some(&0x001A));
        assert_eq!(out.symbols.get("s__DATA"), Some(&0xC000));
        assert_eq!(out.symbols.get("l__DATA"), Some(&0x001A));
        assert_eq!(out.symbols.get("s__BSS"), Some(&0xC000));
        assert_eq!(out.symbols.get("l__BSS"), Some(&0x001A));
        assert_eq!(out.bytes[0x0300 - 0x0100], 0x55);
        assert_eq!(out.bytes[0x0400 - 0x0100], 0x77);

        // User-defined symbols must not be overwritten by synthesized defaults
        let custom = r#"
            s__DATA = 0xD000
            l__DATA = 0x0080
            .area _BSS
        my_bss:
            .ds 5
        "#;
        let out2 = assemble_z80_with_ram(custom, 0x0100, 0xC000).expect("custom bounds should assemble");
        assert_eq!(out2.symbols.get("s__DATA"), Some(&0xD000));
        assert_eq!(out2.symbols.get("l__DATA"), Some(&0x0080));
        assert_eq!(out2.symbols.get("s__BSS"), Some(&0xC000));
        assert_eq!(out2.symbols.get("l__BSS"), Some(&5));
    }

    #[test]
    fn asm_per_file_external_scoping_isolates_private_labels_and_equates() {
        let src = r#"
main:
    call _mod_a_entry
    call _mod_b_entry
    ret
; @aun:ext_begin 0
BUF_SZ = 1
_mod_a_entry::
    ld b, BUF_SZ
_psp_loop:
    inc a
    djnz _psp_loop
    jr 1$
    nop
1$:
    call _shared_helper
    ret
_priv_tbl:
    .dw _psp_loop
; @aun:ext_end
; @aun:ext_begin 1
BUF_SZ = 3
_mod_b_entry::
    ld b, BUF_SZ
_psp_loop:
    dec a
    djnz _psp_loop
    jr 1$
    nop
1$:
    jp _shared_helper
    .globl _shared_helper
_shared_helper:
    ret
_priv_tbl:
    .dw _psp_loop
; @aun:ext_end
"#;
        let out = assemble_z80_with_ram(src, 0x0100, 0xC000).expect("scoping should assemble");
        assert!(out.symbols.contains_key("main"));
        assert!(out.symbols.contains_key("_mod_a_entry"));
        assert!(out.symbols.contains_key("_mod_b_entry"));
        assert!(out.symbols.contains_key("_shared_helper"));
        // Private labels and local equates must not leak into global symbol map.
        assert!(!out.symbols.contains_key("_psp_loop"));
        assert!(!out.symbols.contains_key("_priv_tbl"));
        assert!(!out.symbols.contains_key("BUF_SZ"));
        // Two _psp_loop targets must resolve to different addresses: mod_a's before mod_b's.
        let a_entry = out.symbols["_mod_a_entry"];
        let b_entry = out.symbols["_mod_b_entry"];
        assert!(a_entry < b_entry);
    }

    #[test]
    fn asm_initializer_and_initialized_segment_synthesis() {
        // When no __aun_init_ram_end is defined, l__INITIALIZER = 0 and s__DATA == ram_base.
        let empty = r#"
            .area _CODE
        start:
            ret
            .area _BSS
        bss_var:
            .ds 4
        "#;
        let out_empty = assemble_z80_with_ram(empty, 0x0100, 0xC000).expect("empty init");
        assert_eq!(out_empty.symbols.get("s__INITIALIZED"), Some(&0xC000));
        assert_eq!(out_empty.symbols.get("l__INITIALIZER"), Some(&0));
        assert_eq!(out_empty.symbols.get("s__DATA"), Some(&0xC000));
        assert_eq!(out_empty.symbols.get("l__DATA"), Some(&4));

        // When __aun_init_rom and __aun_init_ram_end exist, s__DATA begins after initialized RAM.
        let with_init = r#"
            __aun_ram == 0xC000
            .area _CODE
        start:
            ld hl, s__INITIALIZER
            ld de, s__INITIALIZED
            ld bc, l__INITIALIZER
            ret
        __aun_init_rom:
            .db 0x12, 0x34, 0x56
        __aun_init_ram_end == 0xC003
            .area _BSS
        bss_only:
            .ds 2
        "#;
        let out_init = assemble_z80_with_ram(with_init, 0x0100, 0xC003).expect("with init");
        let init_rom = out_init.symbols["__aun_init_rom"];
        assert_eq!(out_init.symbols.get("s__INITIALIZER"), Some(&init_rom));
        assert_eq!(out_init.symbols.get("s__INITIALIZED"), Some(&0xC000));
        assert_eq!(out_init.symbols.get("l__INITIALIZER"), Some(&3));
        assert_eq!(out_init.symbols.get("s__DATA"), Some(&0xC003));
        assert_eq!(out_init.symbols.get("l__DATA"), Some(&2));
        assert_eq!(out_init.symbols.get("s__BSS"), Some(&0xC003));
        assert_eq!(out_init.symbols.get("l__BSS"), Some(&2));
    }
}
