//! Path-sensitive static Worst-Case Execution Time (WCET) analysis and cycle-budget
//! verification over lowered Z80 assembly.

use crate::analysis::z80_sym;
use crate::asm::{self, ParsedLine};
use crate::lir::{is_bb_label, is_tail_label};
use crate::OverlayOutput;
use aun_ir::{IrFunction, IrProgram};
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

/// Computed WCET bound for a function or loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WcetBound {
    Bounded(u32),
    Unbounded(String),
}

/// Whole-program WCET analysis output.
#[derive(Debug, Clone, Default)]
pub struct WcetResult {
    /// Bounded WCET in T-states keyed by IR function name (or routine symbol).
    pub fn_wcet_tstates: HashMap<String, u32>,
    /// Full WCET status (bounded or unbounded with offending loop/cycle label) keyed by IR/symbol name.
    pub fn_bounds: HashMap<String, WcetBound>,
}

const KNOWN_RUNTIME_SYMS: &[&str] = &[
    "__aun_start",
    "__aun_startup_init",
    "__aun_panic",
    "__aun_jtbl_dispatch",
    "__aun_jp_iy",
    "__aun_noop_ret",
    "__aun_mul8",
    "__aun_mul32",
    "__aun_mulu16_32",
    "__aun_muls16_32",
    "__aun_udiv32",
    "__aun_urem32",
    "__aun_udivrem32",
    "__aun_udivmod32",
    "__aun_sdiv32",
    "__aun_srem32",
    "__aun_sdivmod32",
    "__aun_neg_hlde",
    "__aun_udiv16",
    "__aun_urem16",
    "__aun_udivmod16",
    "__aun_sdiv16",
    "__aun_srem16",
    "__aun_sdivmod16_abs",
    "__aun_neg_hl",
    "__aun_udiv8",
    "__aun_urem8",
    "__aun_udivmod8",
    "__aun_sdiv8",
    "__aun_srem8",
    "__aun_sdivmod8_abs",
    "__aun_udivrem32_8",
];

#[derive(Debug, Clone)]
enum FnLine {
    Label(String),
    Inst { mnemonic: String, operands: Vec<String> },
}

#[derive(Debug, Clone)]
struct FnUnit {
    name: String,
    items: Vec<FnLine>,
    fallthrough_next: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CfgTarget {
    Block(usize),
    Return,
    TailCall(String),
}

#[derive(Debug, Clone)]
struct CfgEdge {
    target: CfgTarget,
    cost: u32,
}

#[derive(Debug, Clone)]
struct CfgBlock {
    labels: Vec<String>,
    base_cost: u32,
    calls: Vec<String>,
    edges: Vec<CfgEdge>,
    /// Immediate constant register loads surviving to the end of this block (`reg -> imm_u8`).
    end_reg8_consts: HashMap<char, u8>,
    /// Registers clobbered inside this block (excluding a trailing `djnz` / `dec r; jr nz` latch).
    body_clobbered_regs: HashSet<char>,
    /// Whether this block ends with `djnz` or `dec r; jr/jp nz`.
    trailing_countdown: Option<(char, usize /* target block */)>,
    has_call: bool,
}

#[derive(Debug, Clone)]
struct FnCfg {
    blocks: Vec<CfgBlock>,
}

#[derive(Debug, Clone)]
struct DispatchSpec {
    use_smc: bool,
    handlers: Vec<String>,
}

/// Computes path-sensitive WCET across resident and overlay Z80 assembly.
pub fn compute_wcet(
    program: &IrProgram,
    resident_asm: &str,
    resident_symbols: &HashMap<String, u16>,
    overlays: &[OverlayOutput],
) -> WcetResult {
    let mut addr_to_sym: HashMap<u16, String> = HashMap::new();
    let mut ir_fn_by_sym: HashMap<String, &IrFunction> = HashMap::new();
    let mut sym_to_ir_name: HashMap<String, String> = HashMap::new();

    for f in &program.functions {
        if f.is_test {
            continue;
        }
        let sym = z80_sym(&f.name);
        sym_to_ir_name.insert(sym.clone(), f.name.clone());
        ir_fn_by_sym.insert(sym.clone(), f);
        if let Some(&addr) = resident_symbols.get(&sym) {
            addr_to_sym.insert(addr, sym.clone());
        }
        for ovl in overlays {
            if let Some(&addr) = ovl.symbols.get(&sym) {
                addr_to_sym.insert(addr, sym.clone());
            }
        }
    }
    for e in &program.externs {
        sym_to_ir_name.insert(e.name.clone(), e.name.clone());
        if let Some(&addr) = resident_symbols.get(&e.name) {
            addr_to_sym.insert(addr, e.name.clone());
        }
        for ovl in overlays {
            if let Some(&addr) = ovl.symbols.get(&e.name) {
                addr_to_sym.insert(addr, e.name.clone());
            }
        }
    }

    let mut dispatchers: HashMap<String, DispatchSpec> = HashMap::new();
    for mdef in &program.machines {
        let mut methods: BTreeSet<String> = BTreeSet::new();
        if mdef.states.iter().any(|s| s.on_exit.is_some()) {
            methods.insert("on_exit".to_string());
        }
        for st in &mdef.states {
            for m in st.methods.keys() {
                methods.insert(m.clone());
            }
        }
        for method in methods {
            let disp_sym = format!("{}_disp_{method}", mdef.name);
            let mut handlers: Vec<String> = Vec::new();
            for st in &mdef.states {
                let fid_opt = if method == "on_exit" {
                    st.on_exit
                } else {
                    st.methods.get(&method).copied()
                };
                if let Some(fid) = fid_opt
                    && let Some(f) = program.functions.get(fid as usize)
                {
                    let hsym = z80_sym(&f.name);
                    if !handlers.contains(&hsym) {
                        handlers.push(hsym);
                    }
                }
            }
            if handlers.is_empty() {
                handlers.push("__aun_noop_ret".to_string());
            }
            dispatchers.insert(
                disp_sym,
                DispatchSpec {
                    use_smc: mdef.smc,
                    handlers,
                },
            );
        }
    }

    let mut module_sources: Vec<&str> = Vec::with_capacity(1 + overlays.len());
    module_sources.push(resident_asm);
    for ovl in overlays {
        module_sources.push(&ovl.asm_listing);
    }

    let parsed_modules: Vec<Vec<ParsedLine>> = module_sources
        .iter()
        .filter_map(|src| asm::parse_lines(src).ok())
        .collect();

    let mut top_syms = collect_top_level_syms(program, &parsed_modules, &addr_to_sym, &dispatchers);
    let mut fn_units: HashMap<String, FnUnit> = HashMap::new();
    let mut data_tables: HashMap<String, Vec<String>> = HashMap::new();

    for lines in &parsed_modules {
        extract_units_and_tables(lines, &mut top_syms, &mut fn_units, &mut data_tables);
    }

    let mut cfgs: HashMap<String, FnCfg> = HashMap::new();
    for (name, unit) in &fn_units {
        cfgs.insert(
            name.clone(),
            build_fn_cfg(unit, &top_syms, &addr_to_sym, &data_tables),
        );
    }

    let extern_names: HashSet<String> = program.externs.iter().map(|e| e.name.clone()).collect();
    let mut memo: HashMap<String, WcetBound> = HashMap::new();
    let mut active: HashSet<String> = HashSet::new();

    // Ensure every function, dispatcher, extern, and runtime helper gets evaluated.
    let mut all_targets: Vec<String> = cfgs.keys().cloned().collect();
    for d in dispatchers.keys() {
        if !cfgs.contains_key(d) {
            all_targets.push(d.clone());
        }
    }
    all_targets.sort();

    for sym in &all_targets {
        eval_fn_wcet(
            sym,
            &cfgs,
            &ir_fn_by_sym,
            &dispatchers,
            &extern_names,
            &mut memo,
            &mut active,
        );
    }

    let mut out = WcetResult::default();
    for (sym, bound) in memo {
        let out_key = sym_to_ir_name.get(&sym).cloned().unwrap_or(sym);
        if let WcetBound::Bounded(ts) = bound {
            out.fn_wcet_tstates.insert(out_key.clone(), ts);
        }
        out.fn_bounds.insert(out_key, bound);
    }
    out
}

/// Enforces `#[budget(...)]` cycle budgets on all IR functions.
pub fn verify_cycle_budgets(program: &IrProgram, wcet: &WcetResult) -> Result<(), String> {
    for func in &program.functions {
        if func.is_test || func.blocks.is_empty() {
            continue;
        }
        let Some((ref label_opt, limit)) = func.cycle_budget else {
            continue;
        };
        let label_desc = label_opt
            .as_ref()
            .map(|l| format!(" (budget `{l}`)"))
            .unwrap_or_default();
        match wcet.fn_bounds.get(&func.name) {
            Some(WcetBound::Bounded(computed)) => {
                if *computed > limit {
                    return Err(format!(
                        "function `{}`{} exceeds cycle budget: worst-case {} T-states > limit {} T-states",
                        func.name, label_desc, computed, limit
                    ));
                }
            }
            Some(WcetBound::Unbounded(loop_lbl)) => {
                return Err(format!(
                    "function `{}`{} contains unbounded loop `{}` and cannot verify cycle budget ({}T); annotate the loop with `#[max_iters(N)]`",
                    func.name, label_desc, loop_lbl, limit
                ));
            }
            None => {
                return Err(format!(
                    "function `{}`{} could not be analyzed for cycle budget ({}T)",
                    func.name, label_desc, limit
                ));
            }
        }
    }
    Ok(())
}

fn is_trampoline_entry(lbl: &str) -> bool {
    if !lbl.starts_with("__aun_tr_") {
        return false;
    }
    // Trampoline labels end with `_{copy_start}_{copy_end}` (two numeric segments).
    let mut parts = lbl.rsplitn(3, '_');
    let p1 = parts.next().unwrap_or("");
    let p2 = parts.next().unwrap_or("");
    !p1.is_empty()
        && !p2.is_empty()
        && p1.bytes().all(|b| b.is_ascii_digit())
        && p2.bytes().all(|b| b.is_ascii_digit())
}

fn is_outlined_entry(lbl: &str) -> bool {
    lbl.strip_prefix("__aun_out_")
        .is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
}

fn resolve_target_sym(target: &str, addr_to_sym: &HashMap<u16, String>) -> String {
    let t = target.trim();
    if let Some(addr) = asm::parse_const_num(t).and_then(|v| u16::try_from(v).ok())
        && let Some(sym) = addr_to_sym.get(&addr)
    {
        return sym.clone();
    }
    t.to_string()
}

fn collect_top_level_syms(
    program: &IrProgram,
    modules: &[Vec<ParsedLine>],
    addr_to_sym: &HashMap<u16, String>,
    dispatchers: &HashMap<String, DispatchSpec>,
) -> HashSet<String> {
    let mut syms: HashSet<String> = HashSet::new();
    for f in &program.functions {
        if !f.is_test {
            syms.insert(z80_sym(&f.name));
        }
    }
    for e in &program.externs {
        syms.insert(e.name.clone());
    }
    for &rt in KNOWN_RUNTIME_SYMS {
        syms.insert(rt.to_string());
    }
    for d in dispatchers.keys() {
        syms.insert(d.clone());
    }
    for lines in modules {
        let mut in_data = false;
        for line in lines {
            match line {
                ParsedLine::Area(is_data) => in_data = *is_data,
                ParsedLine::Label { name: lbl, .. } if !in_data => {
                    if is_trampoline_entry(lbl) || is_outlined_entry(lbl) {
                        syms.insert(lbl.clone());
                    }
                }
                ParsedLine::Inst {
                    mnemonic, operands, ..
                } if !in_data && mnemonic == "call" => {
                    if let Some(raw_tgt) = operands.last() {
                        let tgt = resolve_target_sym(raw_tgt, addr_to_sym);
                        if !tgt.ends_with('$') && !is_bb_label(&tgt) && !is_tail_label(&tgt) {
                            syms.insert(tgt);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    syms
}

fn extract_units_and_tables(
    lines: &[ParsedLine],
    top_syms: &mut HashSet<String>,
    units: &mut HashMap<String, FnUnit>,
    data_tables: &mut HashMap<String, Vec<String>>,
) {
    let mut in_data = false;
    let mut cur: Option<FnUnit> = None;
    let mut cur_has_inst = false;
    let mut pending_data_lbl: Option<String> = None;

    let flush_cur = |cur_opt: &mut Option<FnUnit>,
                     has_inst: bool,
                     next_fn: Option<&str>,
                     units: &mut HashMap<String, FnUnit>| {
        if let Some(mut u) = cur_opt.take()
            && has_inst
        {
            u.fallthrough_next = next_fn.map(str::to_string);
            units.insert(u.name.clone(), u);
        }
    };

    for line in lines {
        match line {
            ParsedLine::Area(is_data) => {
                in_data = *is_data;
                if in_data {
                    flush_cur(&mut cur, cur_has_inst, None, units);
                    cur_has_inst = false;
                    pending_data_lbl = None;
                }
            }
            ParsedLine::ExtScope(_)
            | ParsedLine::Globl(_)
            | ParsedLine::Org { .. }
            | ParsedLine::Equ { .. }
            | ParsedLine::Align { .. } => {}
            _ if in_data => {}
            ParsedLine::Label { name: lbl, .. } => {
                if top_syms.contains(lbl) {
                    flush_cur(&mut cur, cur_has_inst, Some(lbl), units);
                    cur_has_inst = false;
                    pending_data_lbl = Some(lbl.clone());
                    cur = Some(FnUnit {
                        name: lbl.clone(),
                        items: vec![FnLine::Label(lbl.clone())],
                        fallthrough_next: None,
                    });
                } else {
                    pending_data_lbl = Some(lbl.clone());
                    if let Some(ref mut u) = cur {
                        u.items.push(FnLine::Label(lbl.clone()));
                    }
                }
            }
            ParsedLine::LocalLabel(num) => {
                pending_data_lbl = None;
                if let Some(ref mut u) = cur {
                    let scoped = format!("{}::{num}", u.name);
                    u.items.push(FnLine::Label(scoped));
                }
            }
            ParsedLine::Inst {
                mnemonic, operands, ..
            } => {
                pending_data_lbl = None;
                if let Some(ref mut u) = cur {
                    cur_has_inst = true;
                    u.items.push(FnLine::Inst {
                        mnemonic: mnemonic.clone(),
                        operands: operands.clone(),
                    });
                }
            }
            ParsedLine::Dw { items: words, .. } => {
                if let Some(ref dlbl) = pending_data_lbl {
                    data_tables
                        .entry(dlbl.clone())
                        .or_default()
                        .extend(words.iter().cloned());
                }
                if !cur_has_inst {
                    // Data table labeled with a top-level symbol (e.g. `*_vtbl_*`, `__aun_swap_*`)
                    if let Some(ref u) = cur {
                        top_syms.remove(&u.name);
                    }
                    cur = None;
                } else {
                    flush_cur(&mut cur, true, None, units);
                    cur_has_inst = false;
                }
            }
            ParsedLine::Db { .. } | ParsedLine::Ds { .. } => {
                pending_data_lbl = None;
                if !cur_has_inst {
                    if let Some(ref u) = cur {
                        top_syms.remove(&u.name);
                    }
                    cur = None;
                } else {
                    flush_cur(&mut cur, true, None, units);
                    cur_has_inst = false;
                }
            }
        }
    }
    flush_cur(&mut cur, cur_has_inst, None, units);
}

fn empty_block(labels: Vec<String>) -> CfgBlock {
    CfgBlock {
        labels,
        base_cost: 0,
        calls: Vec::new(),
        edges: Vec::new(),
        end_reg8_consts: HashMap::new(),
        body_clobbered_regs: HashSet::new(),
        trailing_countdown: None,
        has_call: false,
    }
}

fn single_reg8_char(s: &str) -> Option<char> {
    let t = s.trim().to_ascii_lowercase();
    match t.as_str() {
        "a" | "b" | "c" | "d" | "e" | "h" | "l" => t.chars().next(),
        _ => None,
    }
}

fn parse_u8_imm(s: &str) -> Option<u8> {
    asm::parse_const_num(s).map(|v| (v & 0xFF) as u8)
}

fn clobbered_regs_of_inst(mnemonic: &str, ops: &[String]) -> Vec<char> {
    match mnemonic {
        "nop" | "di" | "ei" | "scf" | "ccf" | "halt" | "jp" | "jr" | "ret" | "reti"
        | "retn" | "push" | "out" | "cp" | "bit" => vec![],
        "djnz" => vec!['b'],
        "call" | "rst" | "exx" => vec!['a', 'b', 'c', 'd', 'e', 'h', 'l'],
        "ex" if ops.len() == 2 => {
            if ops[0].eq_ignore_ascii_case("de") && ops[1].eq_ignore_ascii_case("hl") {
                vec!['d', 'e', 'h', 'l']
            } else {
                vec!['a']
            }
        }
        "ldir" | "lddr" | "ldi" | "ldd" => vec!['b', 'c', 'd', 'e', 'h', 'l'],
        "neg" | "cpl" | "rla" | "rra" | "rlca" | "rrca" | "daa" | "sub" | "and" | "xor"
        | "or" => vec!['a'],
        "add" | "adc" | "sbc" => {
            if let Some(d) = ops.first() {
                if d.eq_ignore_ascii_case("hl") {
                    vec!['h', 'l']
                } else if let Some(r) = single_reg8_char(d) {
                    vec![r]
                } else {
                    vec![]
                }
            } else {
                vec![]
            }
        }
        "inc" | "dec" | "sla" | "sra" | "srl" | "rl" | "rr" | "rlc" | "rrc" => {
            if let Some(op) = ops.first() {
                match op.trim().to_ascii_lowercase().as_str() {
                    "bc" => vec!['b', 'c'],
                    "de" => vec!['d', 'e'],
                    "hl" => vec!['h', 'l'],
                    _ => single_reg8_char(op).into_iter().collect(),
                }
            } else {
                vec![]
            }
        }
        "pop" => {
            if let Some(op) = ops.first() {
                match op.trim().to_ascii_lowercase().as_str() {
                    "af" => vec!['a'],
                    "bc" => vec!['b', 'c'],
                    "de" => vec!['d', 'e'],
                    "hl" => vec!['h', 'l'],
                    _ => vec![],
                }
            } else {
                vec![]
            }
        }
        "ld" => {
            if let Some(dst) = ops.first() {
                match dst.trim().to_ascii_lowercase().as_str() {
                    "bc" => vec!['b', 'c'],
                    "de" => vec!['d', 'e'],
                    "hl" => vec!['h', 'l'],
                    _ => single_reg8_char(dst).into_iter().collect(),
                }
            } else {
                vec![]
            }
        }
        "in" => ops.first().and_then(|d| single_reg8_char(d)).into_iter().collect(),
        _ => vec!['a', 'b', 'c', 'd', 'e', 'h', 'l'],
    }
}

fn encode_base_tstates(mnemonic: &str, ops: &[String]) -> u32 {
    let dummy_syms = HashMap::new();
    match asm::encode_inst(mnemonic, ops, 0, &dummy_syms, "", None, false) {
        Ok((_, ts)) => ts as u32,
        Err(_) => match mnemonic {
            "retn" => 14,
            "rst" => 11,
            _ => 8,
        },
    }
}

fn build_fn_cfg(
    unit: &FnUnit,
    top_syms: &HashSet<String>,
    addr_to_sym: &HashMap<u16, String>,
    data_tables: &HashMap<String, Vec<String>>,
) -> FnCfg {
    // Pass 1: split `unit.items` into raw blocks (keeping raw instructions per block).
    struct RawBlock {
        labels: Vec<String>,
        insts: Vec<(String, Vec<String>)>,
        falls_through: bool,
    }

    let mut raw_blocks: Vec<RawBlock> = Vec::new();
    let mut label_to_block: HashMap<String, usize> = HashMap::new();

    for item in &unit.items {
        match item {
            FnLine::Label(lbl) => {
                if let Some(last) = raw_blocks.last_mut()
                    && last.insts.is_empty()
                {
                    last.labels.push(lbl.clone());
                    let bi = raw_blocks.len() - 1;
                    label_to_block.insert(lbl.clone(), bi);
                    continue;
                }
                let bi = raw_blocks.len();
                label_to_block.insert(lbl.clone(), bi);
                raw_blocks.push(RawBlock {
                    labels: vec![lbl.clone()],
                    insts: Vec::new(),
                    falls_through: true,
                });
            }
            FnLine::Inst { mnemonic, operands } => {
                if raw_blocks.is_empty() {
                    raw_blocks.push(RawBlock {
                        labels: Vec::new(),
                        insts: Vec::new(),
                        falls_through: true,
                    });
                }
                let last = raw_blocks.last_mut().unwrap();
                last.insts.push((mnemonic.clone(), operands.clone()));
                let is_term = matches!(
                    mnemonic.as_str(),
                    "jp" | "jr" | "djnz" | "ret" | "reti" | "retn" | "halt"
                );
                if is_term {
                    let uncond = matches!(mnemonic.as_str(), "reti" | "retn" | "halt")
                        || (mnemonic == "ret" && operands.is_empty())
                        || ((mnemonic == "jp" || mnemonic == "jr") && operands.len() == 1);
                    last.falls_through = !uncond;
                    raw_blocks.push(RawBlock {
                        labels: Vec::new(),
                        insts: Vec::new(),
                        falls_through: true,
                    });
                }
            }
        }
    }
    // Remove trailing empty anonymous block if the function ended with an unconditional terminator.
    if raw_blocks.len() > 1
        && let Some(last) = raw_blocks.last()
        && last.labels.is_empty()
        && last.insts.is_empty()
    {
        raw_blocks.pop();
    }

    let n = raw_blocks.len();
    let mut blocks: Vec<CfgBlock> = Vec::with_capacity(n);

    for (bi, rb) in raw_blocks.iter().enumerate() {
        let mut blk = empty_block(rb.labels.clone());
        let mut reg8_consts: HashMap<char, u8> = HashMap::new();
        let mut static_bc: Option<u16> = None;
        let mut last_hl_sym: Option<String> = None;

        let resolve_lbl = |raw: &str| -> String {
            let t = raw.trim();
            if t.ends_with('$') {
                format!("{}::{t}", unit.name)
            } else {
                resolve_target_sym(t, addr_to_sym)
            }
        };

        let len = rb.insts.len();
        for (ii, (mnem, ops)) in rb.insts.iter().enumerate() {
            let is_last = ii + 1 == len;
            // Track countdown pattern `dec r; jr/jp nz, target` at block end without marking `r` body-clobbered.
            let is_countdown_dec = len >= 2
                && ii + 2 == len
                && mnem == "dec"
                && ops.len() == 1
                && single_reg8_char(&ops[0]).is_some()
                && {
                    let (ref m2, ref o2) = rb.insts[ii + 1];
                    (m2 == "jr" || m2 == "jp")
                        && o2.len() == 2
                        && o2[0].eq_ignore_ascii_case("nz")
                };
            let is_countdown_term = is_last
                && len >= 2
                && (mnem == "jr" || mnem == "jp")
                && ops.len() == 2
                && ops[0].eq_ignore_ascii_case("nz")
                && rb.insts[ii - 1].0 == "dec"
                && rb.insts[ii - 1].1.len() == 1
                && single_reg8_char(&rb.insts[ii - 1].1[0]).is_some();
            let is_djnz_last = is_last && mnem == "djnz";

            // Compute non-terminator / ldir cost using pre-instruction static_bc / last_hl_sym,
            // then update register/const tracking after handling the instruction.
            let pre_static_bc = static_bc;
            let pre_hl_sym = last_hl_sym.clone();

            if !is_countdown_dec && !is_djnz_last {
                for r in clobbered_regs_of_inst(mnem, ops) {
                    blk.body_clobbered_regs.insert(r);
                    reg8_consts.remove(&r);
                }
            }

            // Track static `ld bc, imm` / `ld r, imm` and `ld hl, sym`
            if mnem == "ld" && ops.len() == 2 {
                let dst_l = ops[0].trim().to_ascii_lowercase();
                if dst_l == "bc" {
                    static_bc = asm::parse_const_num(&ops[1]).map(|v| v as u16);
                    if let Some(bc) = static_bc {
                        reg8_consts.insert('b', (bc >> 8) as u8);
                        reg8_consts.insert('c', (bc & 0xFF) as u8);
                    } else {
                        reg8_consts.remove(&'b');
                        reg8_consts.remove(&'c');
                    }
                } else if dst_l == "hl" {
                    let src = ops[1].trim();
                    if data_tables.contains_key(src) {
                        last_hl_sym = Some(src.to_string());
                    } else {
                        last_hl_sym = None;
                    }
                } else if let Some(r) = single_reg8_char(&dst_l) {
                    if let Some(v) = parse_u8_imm(&ops[1]) {
                        reg8_consts.insert(r, v);
                    } else {
                        reg8_consts.remove(&r);
                    }
                    if r == 'b' || r == 'c' {
                        static_bc = match (reg8_consts.get(&'b'), reg8_consts.get(&'c')) {
                            (Some(&b), Some(&c)) => Some(((b as u16) << 8) | (c as u16)),
                            _ => None,
                        };
                    }
                    if r == 'h' || r == 'l' {
                        last_hl_sym = None;
                    }
                }
            } else {
                if clobbered_regs_of_inst(mnem, ops)
                    .iter()
                    .any(|&r| r == 'b' || r == 'c')
                {
                    static_bc = None;
                }
                if clobbered_regs_of_inst(mnem, ops)
                    .iter()
                    .any(|&r| r == 'h' || r == 'l')
                {
                    last_hl_sym = None;
                }
            }

            // Terminator handling
            if is_last
                && matches!(
                    mnem.as_str(),
                    "jp" | "jr" | "djnz" | "ret" | "reti" | "retn" | "halt"
                )
            {
                match mnem.as_str() {
                    "ret" if ops.is_empty() => {
                        blk.edges.push(CfgEdge {
                            target: CfgTarget::Return,
                            cost: 10,
                        });
                    }
                    "ret" if ops.len() == 1 => {
                        blk.edges.push(CfgEdge {
                            target: CfgTarget::Return,
                            cost: 11,
                        });
                        if bi + 1 < n {
                            blk.edges.push(CfgEdge {
                                target: CfgTarget::Block(bi + 1),
                                cost: 5,
                            });
                        }
                    }
                    "reti" | "retn" => {
                        blk.edges.push(CfgEdge {
                            target: CfgTarget::Return,
                            cost: 14,
                        });
                    }
                    "halt" => {
                        blk.edges.push(CfgEdge {
                            target: CfgTarget::Return,
                            cost: 4,
                        });
                    }
                    "djnz" if ops.len() == 1 => {
                        let tgt_lbl = resolve_lbl(&ops[0]);
                        if let Some(&tbi) = label_to_block.get(&tgt_lbl) {
                            blk.trailing_countdown = Some(('b', tbi));
                            blk.edges.push(CfgEdge {
                                target: CfgTarget::Block(tbi),
                                cost: 13,
                            });
                        } else {
                            blk.edges.push(CfgEdge {
                                target: CfgTarget::TailCall(tgt_lbl),
                                cost: 13,
                            });
                        }
                        if bi + 1 < n {
                            blk.edges.push(CfgEdge {
                                target: CfgTarget::Block(bi + 1),
                                cost: 8,
                            });
                        }
                    }
                    "jp" if ops.len() == 1 => {
                        let op0 = ops[0].trim();
                        if op0.eq_ignore_ascii_case("(hl)") {
                            blk.edges.push(CfgEdge {
                                target: CfgTarget::Return,
                                cost: 4,
                            });
                        } else if op0.eq_ignore_ascii_case("(ix)")
                            || op0.eq_ignore_ascii_case("(iy)")
                        {
                            blk.edges.push(CfgEdge {
                                target: CfgTarget::Return,
                                cost: 8,
                            });
                        } else {
                            let tgt_lbl = resolve_lbl(op0);
                            if tgt_lbl == "__aun_jtbl_dispatch"
                                && let Some(ref tbl) = pre_hl_sym
                                && let Some(words) = data_tables.get(tbl)
                            {
                                let mut seen_tbi = HashSet::new();
                                for w in words {
                                    let wlbl = resolve_lbl(w);
                                    if let Some(&tbi) = label_to_block.get(&wlbl)
                                        && seen_tbi.insert(tbi)
                                    {
                                        // `jp __aun_jtbl_dispatch` (10T) + dispatch helper (54T)
                                        blk.edges.push(CfgEdge {
                                            target: CfgTarget::Block(tbi),
                                            cost: 64,
                                        });
                                    }
                                }
                            } else if let Some(&tbi) = label_to_block.get(&tgt_lbl)
                                && !top_syms.contains(&tgt_lbl)
                            {
                                blk.edges.push(CfgEdge {
                                    target: CfgTarget::Block(tbi),
                                    cost: 10,
                                });
                            } else if let Some(&tbi) = label_to_block.get(&tgt_lbl) {
                                // Self-tail-jump or intra-fn label
                                if tgt_lbl == unit.name {
                                    blk.edges.push(CfgEdge {
                                        target: CfgTarget::TailCall(tgt_lbl),
                                        cost: 10,
                                    });
                                } else {
                                    blk.edges.push(CfgEdge {
                                        target: CfgTarget::Block(tbi),
                                        cost: 10,
                                    });
                                }
                            } else {
                                blk.edges.push(CfgEdge {
                                    target: CfgTarget::TailCall(tgt_lbl),
                                    cost: 10,
                                });
                            }
                        }
                    }
                    "jp" if ops.len() == 2 => {
                        let tgt_lbl = resolve_lbl(&ops[1]);
                        if let Some(&tbi) = label_to_block.get(&tgt_lbl) {
                            if is_countdown_term {
                                let r = single_reg8_char(&rb.insts[ii - 1].1[0]).unwrap();
                                blk.trailing_countdown = Some((r, tbi));
                            }
                            blk.edges.push(CfgEdge {
                                target: CfgTarget::Block(tbi),
                                cost: 10,
                            });
                        } else {
                            blk.edges.push(CfgEdge {
                                target: CfgTarget::TailCall(tgt_lbl),
                                cost: 10,
                            });
                        }
                        if bi + 1 < n {
                            blk.edges.push(CfgEdge {
                                target: CfgTarget::Block(bi + 1),
                                cost: 10,
                            });
                        }
                    }
                    "jr" if ops.len() == 1 => {
                        let tgt_lbl = resolve_lbl(&ops[0]);
                        if let Some(&tbi) = label_to_block.get(&tgt_lbl) {
                            blk.edges.push(CfgEdge {
                                target: CfgTarget::Block(tbi),
                                cost: 12,
                            });
                        } else {
                            blk.edges.push(CfgEdge {
                                target: CfgTarget::TailCall(tgt_lbl),
                                cost: 12,
                            });
                        }
                    }
                    "jr" if ops.len() == 2 => {
                        let tgt_lbl = resolve_lbl(&ops[1]);
                        if let Some(&tbi) = label_to_block.get(&tgt_lbl) {
                            if is_countdown_term {
                                let r = single_reg8_char(&rb.insts[ii - 1].1[0]).unwrap();
                                blk.trailing_countdown = Some((r, tbi));
                            }
                            blk.edges.push(CfgEdge {
                                target: CfgTarget::Block(tbi),
                                cost: 12,
                            });
                        } else {
                            blk.edges.push(CfgEdge {
                                target: CfgTarget::TailCall(tgt_lbl),
                                cost: 12,
                            });
                        }
                        if bi + 1 < n {
                            blk.edges.push(CfgEdge {
                                target: CfgTarget::Block(bi + 1),
                                cost: 7,
                            });
                        }
                    }
                    _ => {}
                }
                continue;
            }

            // Non-terminator instruction
            if mnem == "call" {
                blk.has_call = true;
                blk.base_cost = blk.base_cost.saturating_add(17);
                if let Some(raw_tgt) = ops.last() {
                    blk.calls.push(resolve_lbl(raw_tgt));
                }
            } else if mnem == "ldir" || mnem == "lddr" {
                let ts = if let Some(bc) = pre_static_bc {
                    let count = if bc == 0 { 65536u32 } else { bc as u32 };
                    21u32.saturating_mul(count - 1).saturating_add(16)
                } else {
                    21
                };
                blk.base_cost = blk.base_cost.saturating_add(ts);
            } else {
                blk.base_cost = blk
                    .base_cost
                    .saturating_add(encode_base_tstates(mnem, ops));
            }
        }

        if rb.falls_through && blk.edges.is_empty() {
            if bi + 1 < n {
                blk.edges.push(CfgEdge {
                    target: CfgTarget::Block(bi + 1),
                    cost: 0,
                });
            } else if let Some(ref next_fn) = unit.fallthrough_next {
                blk.edges.push(CfgEdge {
                    target: CfgTarget::TailCall(next_fn.clone()),
                    cost: 0,
                });
            } else {
                blk.edges.push(CfgEdge {
                    target: CfgTarget::Return,
                    cost: 0,
                });
            }
        }
        blk.end_reg8_consts = reg8_consts;
        blocks.push(blk);
    }

    FnCfg { blocks }
}

fn eval_fn_wcet(
    sym: &str,
    cfgs: &HashMap<String, FnCfg>,
    ir_fn_by_sym: &HashMap<String, &IrFunction>,
    dispatchers: &HashMap<String, DispatchSpec>,
    extern_names: &HashSet<String>,
    memo: &mut HashMap<String, WcetBound>,
    active: &mut HashSet<String>,
) -> WcetBound {
    if let Some(cached) = memo.get(sym) {
        return cached.clone();
    }
    if !active.insert(sym.to_string()) {
        return WcetBound::Unbounded(format!("recursion `{sym}`"));
    }

    let bound = if let Some(spec) = dispatchers.get(sym) {
        let mut max_handler: u32 = 10; // at least `__aun_noop_ret`
        let mut err_bound: Option<WcetBound> = None;
        for h in &spec.handlers {
            match eval_fn_wcet(h, cfgs, ir_fn_by_sym, dispatchers, extern_names, memo, active) {
                WcetBound::Bounded(ts) => {
                    max_handler = max_handler.max(ts);
                }
                ub @ WcetBound::Unbounded(_) => {
                    err_bound = Some(ub);
                    break;
                }
            }
        }
        if let Some(ub) = err_bound {
            ub
        } else {
            let stub_overhead = if spec.use_smc {
                10u32
            } else if let Some(cfg) = cfgs.get(sym) {
                match compute_cfg_wcet(
                    sym,
                    cfg,
                    ir_fn_by_sym.get(sym).copied(),
                    cfgs,
                    ir_fn_by_sym,
                    dispatchers,
                    extern_names,
                    memo,
                    active,
                ) {
                    WcetBound::Bounded(ts) => ts,
                    ub @ WcetBound::Unbounded(_) => {
                        active.remove(sym);
                        memo.insert(sym.to_string(), ub.clone());
                        return ub;
                    }
                }
            } else {
                10
            };
            WcetBound::Bounded(stub_overhead.saturating_add(max_handler))
        }
    } else if let Some(cfg) = cfgs.get(sym) {
        compute_cfg_wcet(
            sym,
            cfg,
            ir_fn_by_sym.get(sym).copied(),
            cfgs,
            ir_fn_by_sym,
            dispatchers,
            extern_names,
            memo,
            active,
        )
    } else if extern_names.contains(sym) || sym == "__aun_noop_ret" {
        // Unknown `extern "asm"` body: `ret` = 10T (plus caller's `call` = 17T).
        WcetBound::Bounded(10)
    } else {
        WcetBound::Bounded(10)
    };

    active.remove(sym);
    memo.insert(sym.to_string(), bound.clone());
    bound
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ExitKey {
    Block(usize),
    Return,
    TailCall(String),
}

enum LoopBoundKind {
    TopTested(u32),
    BottomTested(u32),
    Unbounded(String),
}

#[allow(clippy::too_many_arguments)]
fn compute_cfg_wcet(
    fn_name: &str,
    orig_cfg: &FnCfg,
    ir_fn: Option<&IrFunction>,
    all_cfgs: &HashMap<String, FnCfg>,
    ir_fn_by_sym: &HashMap<String, &IrFunction>,
    dispatchers: &HashMap<String, DispatchSpec>,
    extern_names: &HashSet<String>,
    memo: &mut HashMap<String, WcetBound>,
    active: &mut HashSet<String>,
) -> WcetBound {
    if orig_cfg.blocks.is_empty() {
        return WcetBound::Bounded(0);
    }
    let mut blocks = orig_cfg.blocks.clone();
    let n = blocks.len();

    // Precompute block internal call costs; short-circuit immediately if any reachable callee is unbounded.
    // (Wait: we evaluate calls lazily per reachable block so unreachable panics/dead blocks don't hurt,
    // though normally all blocks in `compute_cfg_wcet` are reachable.)

    loop {
        // 1. Reachable blocks from 0 and RPO
        let rpo = compute_rpo(&blocks, 0);
        let reachable: HashSet<usize> = rpo.iter().copied().collect();

        // 2. Iterative dominators on reachable subgraph
        let doms = compute_dominators(&blocks, &rpo, &reachable, n);

        // 3. Collect natural loops grouped by header
        let mut by_header: HashMap<usize, (Vec<usize>, BTreeSet<usize>)> = HashMap::new();
        for &u in &rpo {
            for e in &blocks[u].edges {
                if let CfgTarget::Block(h) = e.target
                    && reachable.contains(&h)
                    && doms[u].contains(&h)
                {
                    let body = collect_natural_loop_body(&blocks, &reachable, h, u);
                    let entry = by_header
                        .entry(h)
                        .or_insert_with(|| (Vec::new(), BTreeSet::new()));
                    if !entry.0.contains(&u) {
                        entry.0.push(u);
                    }
                    entry.1.extend(body);
                }
            }
        }

        if by_header.is_empty() {
            // Check for irreducible cycles in `rpo`
            if has_cycle(&blocks, &rpo) {
                return WcetBound::Unbounded(format!("{fn_name}_irreducible"));
            }
            // DAG longest path in reverse topological order
            let Some(topo) = topo_sort(&blocks, &rpo) else {
                return WcetBound::Unbounded(format!("{fn_name}_cycle"));
            };
            let mut dp: HashMap<usize, u32> = HashMap::new();
            for &u in topo.iter().rev() {
                let mut node_cost = blocks[u].base_cost;
                for c in &blocks[u].calls {
                    match eval_fn_wcet(
                        c,
                        all_cfgs,
                        ir_fn_by_sym,
                        dispatchers,
                        extern_names,
                        memo,
                        active,
                    ) {
                        WcetBound::Bounded(ts) => node_cost = node_cost.saturating_add(ts),
                        ub @ WcetBound::Unbounded(_) => return ub,
                    }
                }
                let mut best_tail = 0u32;
                let mut has_edge = false;
                for e in &blocks[u].edges {
                    let edge_total = match &e.target {
                        CfgTarget::Return => e.cost,
                        CfgTarget::TailCall(tgt) => {
                            let ts = match eval_fn_wcet(
                                tgt,
                                all_cfgs,
                                ir_fn_by_sym,
                                dispatchers,
                                extern_names,
                                memo,
                                active,
                            ) {
                                WcetBound::Bounded(ts) => ts,
                                ub @ WcetBound::Unbounded(_) => return ub,
                            };
                            e.cost.saturating_add(ts)
                        }
                        CfgTarget::Block(v) => e.cost.saturating_add(dp.get(v).copied().unwrap_or(0)),
                    };
                    if !has_edge || edge_total > best_tail {
                        best_tail = edge_total;
                        has_edge = true;
                    }
                }
                dp.insert(u, node_cost.saturating_add(best_tail));
            }
            return WcetBound::Bounded(dp.get(&0).copied().unwrap_or(0));
        }

        // Pick smallest natural loop
        let (header, latches, body) = by_header
            .into_iter()
            .min_by_key(|(h, (_, body))| (body.len(), *h))
            .map(|(h, (l, b))| (h, l, b))
            .unwrap();

        let bound_kind = resolve_loop_bound(
            fn_name,
            header,
            &latches,
            &body,
            &blocks,
            &reachable,
            ir_fn,
        );
        let (max_iters, is_bottom_tested) = match bound_kind {
            LoopBoundKind::TopTested(n) => (n, false),
            LoopBoundKind::BottomTested(n) => (n, true),
            LoopBoundKind::Unbounded(lbl) => return WcetBound::Unbounded(lbl),
        };

        // Build DAG order inside `body` ignoring back-edges `(u in body) -> header`.
        let body_nodes: Vec<usize> = rpo.iter().copied().filter(|u| body.contains(u)).collect();
        let Some(body_topo) = topo_sort_subset(&blocks, &body_nodes, &body, header) else {
            let lbl = blocks[header]
                .labels
                .first()
                .cloned()
                .unwrap_or_else(|| format!("{fn_name}_bb{header}"));
            return WcetBound::Unbounded(lbl);
        };

        // Compute per-node internal cost (base + calls) for `body_topo`.
        let mut node_costs: HashMap<usize, u32> = HashMap::new();
        for &u in &body_topo {
            let mut cost = blocks[u].base_cost;
            for c in &blocks[u].calls {
                match eval_fn_wcet(
                    c,
                    all_cfgs,
                    ir_fn_by_sym,
                    dispatchers,
                    extern_names,
                    memo,
                    active,
                ) {
                    WcetBound::Bounded(ts) => cost = cost.saturating_add(ts),
                    ub @ WcetBound::Unbounded(_) => return ub,
                }
            }
            node_costs.insert(u, cost);
        }

        // DP from `body_topo` in reverse order:
        // - `to_back[u]`: max cost from `u` to a back-edge targeting `header`
        // - `to_exit[u]`: max cost from `u` to each exit target outside `body`
        let mut to_back: HashMap<usize, u32> = HashMap::new();
        let mut to_exit: HashMap<usize, HashMap<ExitKey, u32>> = HashMap::new();

        for &u in body_topo.iter().rev() {
            let nc = node_costs[&u];
            let mut best_back: Option<u32> = None;
            let mut exits: HashMap<ExitKey, u32> = HashMap::new();

            for e in &blocks[u].edges {
                match &e.target {
                    CfgTarget::Block(v) if *v == header => {
                        let cand = nc.saturating_add(e.cost);
                        best_back = Some(best_back.map_or(cand, |cur| cur.max(cand)));
                    }
                    CfgTarget::Block(v) if body.contains(v) => {
                        if let Some(&vb) = to_back.get(v) {
                            let cand = nc.saturating_add(e.cost).saturating_add(vb);
                            best_back = Some(best_back.map_or(cand, |cur| cur.max(cand)));
                        }
                        if let Some(vex) = to_exit.get(v) {
                            for (ek, &vc) in vex {
                                let cand = nc.saturating_add(e.cost).saturating_add(vc);
                                exits
                                    .entry(ek.clone())
                                    .and_modify(|cur| *cur = (*cur).max(cand))
                                    .or_insert(cand);
                            }
                        }
                    }
                    CfgTarget::Block(v) => {
                        let cand = nc.saturating_add(e.cost);
                        exits
                            .entry(ExitKey::Block(*v))
                            .and_modify(|cur| *cur = (*cur).max(cand))
                            .or_insert(cand);
                    }
                    CfgTarget::Return => {
                        let cand = nc.saturating_add(e.cost);
                        exits
                            .entry(ExitKey::Return)
                            .and_modify(|cur| *cur = (*cur).max(cand))
                            .or_insert(cand);
                    }
                    CfgTarget::TailCall(tgt) => {
                        let cand = nc.saturating_add(e.cost);
                        exits
                            .entry(ExitKey::TailCall(tgt.clone()))
                            .and_modify(|cur| *cur = (*cur).max(cand))
                            .or_insert(cand);
                    }
                }
            }

            if let Some(b) = best_back {
                to_back.insert(u, b);
            }
            if !exits.is_empty() {
                to_exit.insert(u, exits);
            }
        }

        let iter_back_cost = to_back.get(&header).copied().unwrap_or(0);
        let h_exits = to_exit.get(&header).cloned().unwrap_or_default();

        // Direct exits from `header` (relevant for top-tested loop exit on iteration `N + 1`).
        let mut header_direct_exits: HashMap<ExitKey, u32> = HashMap::new();
        let h_nc = node_costs[&header];
        for e in &blocks[header].edges {
            let ek_opt = match &e.target {
                CfgTarget::Block(v) if !body.contains(v) => Some(ExitKey::Block(*v)),
                CfgTarget::Return => Some(ExitKey::Return),
                CfgTarget::TailCall(tgt) => Some(ExitKey::TailCall(tgt.clone())),
                _ => None,
            };
            if let Some(ek) = ek_opt {
                let c = h_nc.saturating_add(e.cost);
                header_direct_exits
                    .entry(ek)
                    .and_modify(|cur| *cur = (*cur).max(c))
                    .or_insert(c);
            }
        }

        let mut super_edges: Vec<CfgEdge> = Vec::new();
        for (ek, all_exit_cost) in h_exits {
            let total = if is_bottom_tested {
                if max_iters == 0 {
                    0
                } else {
                    iter_back_cost
                        .saturating_mul(max_iters - 1)
                        .saturating_add(all_exit_cost)
                }
            } else if let Some(&direct_c) = header_direct_exits.get(&ek) {
                let normal = iter_back_cost
                    .saturating_mul(max_iters)
                    .saturating_add(direct_c);
                if max_iters == 0 {
                    direct_c
                } else {
                    let early = iter_back_cost
                        .saturating_mul(max_iters - 1)
                        .saturating_add(all_exit_cost);
                    normal.max(early)
                }
            } else if max_iters == 0 {
                continue;
            } else {
                iter_back_cost
                    .saturating_mul(max_iters - 1)
                    .saturating_add(all_exit_cost)
            };
            let target = match ek {
                ExitKey::Block(v) => CfgTarget::Block(v),
                ExitKey::Return => CfgTarget::Return,
                ExitKey::TailCall(t) => CfgTarget::TailCall(t),
            };
            super_edges.push(CfgEdge {
                target,
                cost: total,
            });
        }

        if super_edges.is_empty() {
            // Infinite loop with no exit edges (even if bounded by `max_iters`, e.g. `loop {}`).
            // Keep cost as `iter_back_cost * max_iters` returning.
            super_edges.push(CfgEdge {
                target: CfgTarget::Return,
                cost: iter_back_cost.saturating_mul(max_iters),
            });
        }

        // Collapse `body` into `header` super-node.
        blocks[header].base_cost = 0;
        blocks[header].calls.clear();
        blocks[header].edges = super_edges;
        blocks[header].trailing_countdown = None;
    }
}

fn resolve_loop_bound(
    fn_name: &str,
    header: usize,
    latches: &[usize],
    body: &BTreeSet<usize>,
    blocks: &[CfgBlock],
    reachable: &HashSet<usize>,
    ir_fn: Option<&IrFunction>,
) -> LoopBoundKind {
    let header_has_direct_exit = blocks[header].edges.iter().any(|e| match e.target {
        CfgTarget::Block(v) => !body.contains(&v),
        CfgTarget::Return | CfgTarget::TailCall(_) => true,
    });
    let is_bottom_shape =
        (latches.len() == 1 && latches[0] == header) || !header_has_direct_exit;

    // (a) Check `ir_fn.loop_bounds` matched against `{fname}_bb{id}` labels on `header`
    if let Some(func) = ir_fn {
        let mut cand_blocks = vec![header];
        // Also check empty trampoline predecessors (`base_cost == 0`, no calls, single edge -> header)
        for &u in reachable {
            if u != header
                && !body.contains(&u)
                && blocks[u].base_cost == 0
                && blocks[u].calls.is_empty()
                && blocks[u].edges.len() == 1
                && matches!(blocks[u].edges[0].target, CfgTarget::Block(v) if v == header)
            {
                cand_blocks.push(u);
            }
        }
        for cb in cand_blocks {
            for lbl in &blocks[cb].labels {
                if let Some((_, id_str)) = lbl.rsplit_once("_bb")
                    && !id_str.is_empty()
                    && id_str.bytes().all(|b| b.is_ascii_digit())
                    && let Ok(bb_id) = id_str.parse::<u32>()
                    && let Some(&bound) = func.loop_bounds.get(&bb_id)
                {
                    return if is_bottom_shape {
                        LoopBoundKind::BottomTested(bound)
                    } else {
                        LoopBoundKind::TopTested(bound)
                    };
                }
            }
        }
    }

    // (b) Check constant `ld b, N` feeding `djnz`, or `ld r, N` feeding `dec r; jr/jp nz`
    if latches.len() == 1
        && let Some((reg, target)) = blocks[latches[0]].trailing_countdown
        && target == header
    {
        let body_clobbers_reg = body.iter().any(|&u| {
            blocks[u].body_clobbered_regs.contains(&reg)
                || (blocks[u].has_call && matches!(reg, 'a' | 'b' | 'c' | 'd' | 'e' | 'h' | 'l'))
        });
        if !body_clobbers_reg {
            let outside_preds: Vec<usize> = reachable
                .iter()
                .copied()
                .filter(|&u| {
                    !body.contains(&u)
                        && blocks[u]
                            .edges
                            .iter()
                            .any(|e| matches!(e.target, CfgTarget::Block(v) if v == header))
                })
                .collect();
            if !outside_preds.is_empty() {
                let mut common: Option<u8> = None;
                let mut ok = true;
                for &p in &outside_preds {
                    if let Some(&v) = blocks[p].end_reg8_consts.get(&reg) {
                        if let Some(prev) = common {
                            if prev != v {
                                ok = false;
                                break;
                            }
                        } else {
                            common = Some(v);
                        }
                    } else {
                        ok = false;
                        break;
                    }
                }
                if ok && let Some(v) = common {
                    let count = if v == 0 { 256u32 } else { v as u32 };
                    return LoopBoundKind::BottomTested(count);
                }
            }
        }
    }

    // (c) Unbounded loop
    let lbl = blocks[header]
        .labels
        .first()
        .cloned()
        .unwrap_or_else(|| format!("{fn_name}_bb{header}"));
    LoopBoundKind::Unbounded(lbl)
}

fn compute_rpo(blocks: &[CfgBlock], entry: usize) -> Vec<usize> {
    let mut visited = vec![false; blocks.len()];
    let mut postorder = Vec::new();
    fn dfs(u: usize, blocks: &[CfgBlock], visited: &mut [bool], po: &mut Vec<usize>) {
        visited[u] = true;
        for e in &blocks[u].edges {
            if let CfgTarget::Block(v) = e.target
                && !visited[v]
            {
                dfs(v, blocks, visited, po);
            }
        }
        po.push(u);
    }
    if entry < blocks.len() {
        dfs(entry, blocks, &mut visited, &mut postorder);
    }
    postorder.reverse();
    postorder
}

fn compute_dominators(
    blocks: &[CfgBlock],
    rpo: &[usize],
    reachable: &HashSet<usize>,
    n: usize,
) -> Vec<HashSet<usize>> {
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &u in rpo {
        for e in &blocks[u].edges {
            if let CfgTarget::Block(v) = e.target
                && reachable.contains(&v)
            {
                preds[v].push(u);
            }
        }
    }

    let mut doms: Vec<HashSet<usize>> = vec![reachable.clone(); n];
    doms[0] = HashSet::from([0]);

    loop {
        let mut changed = false;
        for &u in rpo {
            if u == 0 {
                continue;
            }
            let mut inter: Option<HashSet<usize>> = None;
            for &p in &preds[u] {
                if !reachable.contains(&p) {
                    continue;
                }
                inter = Some(match inter {
                    None => doms[p].clone(),
                    Some(cur) => cur.intersection(&doms[p]).copied().collect(),
                });
            }
            let mut new_set = inter.unwrap_or_default();
            new_set.insert(u);
            if new_set != doms[u] {
                doms[u] = new_set;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    doms
}

fn collect_natural_loop_body(
    blocks: &[CfgBlock],
    reachable: &HashSet<usize>,
    header: usize,
    latch: usize,
) -> BTreeSet<usize> {
    let mut preds: HashMap<usize, Vec<usize>> = HashMap::new();
    for &u in reachable {
        for e in &blocks[u].edges {
            if let CfgTarget::Block(v) = e.target {
                preds.entry(v).or_default().push(u);
            }
        }
    }
    let mut body = BTreeSet::from([header, latch]);
    let mut work = VecDeque::new();
    if latch != header {
        work.push_back(latch);
    }
    while let Some(u) = work.pop_front() {
        if let Some(ps) = preds.get(&u) {
            for &p in ps {
                if body.insert(p) {
                    work.push_back(p);
                }
            }
        }
    }
    body
}

fn has_cycle(blocks: &[CfgBlock], nodes: &[usize]) -> bool {
    let set: HashSet<usize> = nodes.iter().copied().collect();
    topo_sort_subset_all(blocks, nodes, &set).is_none()
}

fn topo_sort(blocks: &[CfgBlock], nodes: &[usize]) -> Option<Vec<usize>> {
    let set: HashSet<usize> = nodes.iter().copied().collect();
    topo_sort_subset_all(blocks, nodes, &set)
}

fn topo_sort_subset_all(
    blocks: &[CfgBlock],
    nodes: &[usize],
    set: &HashSet<usize>,
) -> Option<Vec<usize>> {
    let mut in_deg: HashMap<usize, usize> = nodes.iter().map(|&u| (u, 0usize)).collect();
    for &u in nodes {
        for e in &blocks[u].edges {
            if let CfgTarget::Block(v) = e.target
                && set.contains(&v)
            {
                *in_deg.entry(v).or_default() += 1;
            }
        }
    }
    let mut q: VecDeque<usize> = nodes
        .iter()
        .copied()
        .filter(|u| in_deg[u] == 0)
        .collect();
    let mut out = Vec::with_capacity(nodes.len());
    while let Some(u) = q.pop_front() {
        out.push(u);
        for e in &blocks[u].edges {
            if let CfgTarget::Block(v) = e.target
                && let Some(d) = in_deg.get_mut(&v)
            {
                *d -= 1;
                if *d == 0 {
                    q.push_back(v);
                }
            }
        }
    }
    (out.len() == nodes.len()).then_some(out)
}

fn topo_sort_subset(
    blocks: &[CfgBlock],
    nodes: &[usize],
    body: &BTreeSet<usize>,
    header: usize,
) -> Option<Vec<usize>> {
    let mut in_deg: HashMap<usize, usize> = nodes.iter().map(|&u| (u, 0usize)).collect();
    for &u in nodes {
        for e in &blocks[u].edges {
            if let CfgTarget::Block(v) = e.target
                && v != header
                && body.contains(&v)
            {
                *in_deg.entry(v).or_default() += 1;
            }
        }
    }
    let mut q: VecDeque<usize> = nodes
        .iter()
        .copied()
        .filter(|u| in_deg[u] == 0)
        .collect();
    let mut out = Vec::with_capacity(nodes.len());
    while let Some(u) = q.pop_front() {
        out.push(u);
        for e in &blocks[u].edges {
            if let CfgTarget::Block(v) = e.target
                && v != header
                && body.contains(&v)
                && let Some(d) = in_deg.get_mut(&v)
            {
                *d -= 1;
                if *d == 0 {
                    q.push_back(v);
                }
            }
        }
    }
    (out.len() == nodes.len()).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aun_ir::{BasicBlock, IrExternFn, IrInst, IrType, Terminator};

    fn simple_ir_fn(id: u32, name: &str, budget: Option<(Option<String>, u32)>) -> IrFunction {
        IrFunction {
            id,
            name: name.to_string(),
            is_pub: true,
            inline_hint: Default::default(),
            param_count: 0,
            locals: vec![],
            ret_ty: IrType::Unit,
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            isr_using: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            owner_state: None,
            mock_target: None,
            export_name: None,
            cycle_budget: budget,
            loop_bounds: HashMap::new(),
            blocks: vec![BasicBlock {
                id: 0,
                insts: vec![IrInst::ConstInt {
                    dst: 0,
                    ty: IrType::Int(aun_ir::IntType::U8),
                    val: 1,
                }],
                term: Terminator::Return(None),
            }],
            next_val: 1,
        }
    }

    #[test]
    fn wcet_if_else_takes_max_and_djnz_uses_ld_b_bound() {
        let mut prog = IrProgram::default();
        prog.functions.push(simple_ir_fn(0, "branch_fn", None));
        prog.functions.push(simple_ir_fn(1, "djnz_fn", None));

        // branch_fn:
        //   or a (4)
        //   jr nz, branch_fn_bb1 (12 taken / 7 fallthrough)
        //   nop (4) ; short else
        //   jr branch_fn_bb2 (12)
        // branch_fn_bb1:
        //   nop (4) ; nop (4) ; nop (4) ; nop (4) ; long then (16)
        // branch_fn_bb2:
        //   ret (10)
        // Path then: 4 + 12 + 16 + 10 = 42T
        // Path else: 4 + 7 + 4 + 12 + 10 = 37T -> max = 42T (NOT sum!)
        //
        // djnz_fn:
        //   ld b, 4 (7)
        // djnz_fn_bb1:
        //   nop (4)
        //   djnz djnz_fn_bb1 (13 taken x3 + 8 fallthrough x1)
        //   ret (10)
        // Total: 7 + 3*(4+13) + (4+8) + 10 = 7 + 51 + 12 + 10 = 80T
        let asm = "\
branch_fn:
    or a
    jr nz, branch_fn_bb1
    nop
    jr branch_fn_bb2
branch_fn_bb1:
    nop
    nop
    nop
    nop
branch_fn_bb2:
    ret

djnz_fn:
    ld b, 4
djnz_fn_bb1:
    nop
    djnz djnz_fn_bb1
    ret
";
        let res = compute_wcet(&prog, asm, &HashMap::new(), &[]);
        assert_eq!(res.fn_wcet_tstates.get("branch_fn"), Some(&42));
        assert_eq!(res.fn_wcet_tstates.get("djnz_fn"), Some(&80));
    }

    #[test]
    fn wcet_ldir_static_bc_and_unknown_extern_call() {
        let mut prog = IrProgram::default();
        prog.functions.push(simple_ir_fn(0, "copy_and_ext", None));
        prog.externs.push(IrExternFn {
            name: "unknown_hw".to_string(),
            params: vec![],
            ret: None,
            clobbers: None,
            requires_di: false,
            requires_banks: vec![],
            fixed_addr: None,
            no_callback: false,
            owner_state: None,
        });

        // ld bc, 5 (10)
        // ldir (4*21 + 16 = 100)
        // call unknown_hw (17 + 10 = 27)
        // ret (10)
        // Total = 147T
        let asm = "\
copy_and_ext:
    ld bc, 5
    ldir
    call unknown_hw
    ret
";
        let res = compute_wcet(&prog, asm, &HashMap::new(), &[]);
        assert_eq!(res.fn_wcet_tstates.get("copy_and_ext"), Some(&147));
    }

    #[test]
    fn wcet_budget_rejects_unbounded_and_over_limit() {
        let mut prog = IrProgram::default();
        prog.functions
            .push(simple_ir_fn(0, "unbounded_fn", Some((Some("vblank".into()), 500))));
        prog.functions
            .push(simple_ir_fn(1, "over_fn", Some((Some("sfx".into()), 20))));

        let asm = "\
unbounded_fn:
unbounded_fn_bb1:
    nop
    jp unbounded_fn_bb1

over_fn:
    nop
    nop
    nop
    nop
    ret
";
        let res = compute_wcet(&prog, asm, &HashMap::new(), &[]);
        let err = verify_cycle_budgets(&prog, &res).unwrap_err();
        assert!(
            err.contains("unbounded") && err.contains("#[max_iters(N)]"),
            "expected unbounded loop error asking for #[max_iters(N)], got: {err}"
        );

        prog.functions[0].cycle_budget = None;
        let err2 = verify_cycle_budgets(&prog, &res).unwrap_err();
        assert!(
            err2.contains("over_fn") && err2.contains("sfx") && err2.contains("26") && err2.contains("20"),
            "expected budget exceeded error with label/wcet/limit, got: {err2}"
        );
    }

    #[test]
    fn wcet_dec_r_jr_nz_countdown_bounded() {
        let mut prog = IrProgram::default();
        prog.functions.push(simple_ir_fn(0, "dec_loop_fn", None));

        // dec_loop_fn:
        //   ld a, 4 (7)
        // dec_loop_fn_bb1:
        //   nop (4) ; dec a (4) ; jr nz (12 taken x3 + 7 fallthrough x1)
        //   ret (10)
        // Total: 7 + 3*(4+4+12) + (4+4+7) + 10 = 92T
        let asm = "\
dec_loop_fn:
    ld a, 4
dec_loop_fn_bb1:
    nop
    dec a
    jr nz, dec_loop_fn_bb1
    ret
";
        let res = compute_wcet(&prog, asm, &HashMap::new(), &[]);
        assert_eq!(res.fn_wcet_tstates.get("dec_loop_fn"), Some(&92));
    }
}

