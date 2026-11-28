//! Lightweight Z80 peephole optimizer operating on emitted assembly text.
//!
//! Transformations are strictly local within basic blocks (never crossing labels
//! or directives) and preserve Z80 flag semantics.

use std::collections::{HashMap, HashSet};

mod cfg;
mod local;
mod reg_cache;
mod spill;
#[cfg(test)]
mod tests;

use cfg::{merge_ret_tails, simplify_cond_returns_and_dedup_blocks, thread_empty_jumps};
use local::run_pass;
use reg_cache::eliminate_redundant_reg_reloads;
use spill::{
    coalesce_duplicate_private_spills, eliminate_dead_bb_scratch_stores,
    eliminate_dead_scratch_stores, promote_single_block_spills_to_bc,
    promote_single_block_spills_to_de, promote_single_block_spills_to_stack,
};

/// Runs peephole optimization passes on Aun-generated Z80 assembly until convergence.
pub fn optimize_asm(src: &str) -> String {
    optimize_asm_per_fn(src, &HashMap::new())
}

/// Runs peephole optimization plus per-function dead private-slot store elimination,
/// single-block scratch DCE, single-block `BC` spill promotion, and `ret`-tail merging.
pub fn optimize_asm_per_fn(
    src: &str,
    fn_private_ranges: &HashMap<String, Vec<(u16, u16)>>,
) -> String {
    optimize_asm_with_sb(src, fn_private_ranges, &HashMap::new())
}

pub fn optimize_asm_with_sb(
    src: &str,
    fn_private_ranges: &HashMap<String, Vec<(u16, u16)>>,
    fn_sb_ranges: &HashMap<String, (u16, u16)>,
) -> String {
    optimize_asm_full(src, fn_private_ranges, fn_sb_ranges, &HashSet::new())
}

pub fn optimize_asm_full(
    src: &str,
    fn_private_ranges: &HashMap<String, Vec<(u16, u16)>>,
    fn_sb_ranges: &HashMap<String, (u16, u16)>,
    no_bc_fns: &HashSet<String>,
) -> String {
    let mut lines: Vec<String> = src.lines().map(String::from).collect();
    for _ in 0..10 {
        let (next, changed1) = run_pass(&lines);
        let (next1b, changed1b) = eliminate_redundant_reg_reloads(&next, fn_private_ranges);
        let (next1c, changed1c) = if fn_private_ranges.is_empty() {
            (next1b, false)
        } else {
            coalesce_duplicate_private_spills(&next1b, fn_private_ranges)
        };
        let (next2, changed2) = if fn_private_ranges.is_empty() {
            (next1c, false)
        } else {
            eliminate_dead_scratch_stores(&next1c, fn_private_ranges)
        };
        let (next2b, changed2b) = if fn_sb_ranges.is_empty() {
            (next2, false)
        } else {
            eliminate_dead_bb_scratch_stores(&next2, fn_sb_ranges)
        };
        let (next2c, changed2c) = if fn_sb_ranges.is_empty() && fn_private_ranges.is_empty() {
            (next2b, false)
        } else {
            promote_single_block_spills_to_de(&next2b, fn_private_ranges, fn_sb_ranges)
        };
        let (next3, changed3) = if fn_private_ranges.is_empty() {
            (next2c, false)
        } else {
            promote_single_block_spills_to_bc(&next2c, fn_private_ranges, fn_sb_ranges, no_bc_fns)
        };
        let (next3b, changed3b) = if fn_sb_ranges.is_empty() && fn_private_ranges.is_empty() {
            (next3, false)
        } else {
            promote_single_block_spills_to_stack(&next3, fn_private_ranges, fn_sb_ranges)
        };
        let (next4, changed4) = merge_ret_tails(&next3b);
        let (next4b, changed4b) = simplify_cond_returns_and_dedup_blocks(&next4);
        let (next5, changed5) = thread_empty_jumps(&next4b);
        lines = next5;
        if !changed1
            && !changed1b
            && !changed1c
            && !changed2
            && !changed2b
            && !changed2c
            && !changed3
            && !changed3b
            && !changed4
            && !changed4b
            && !changed5
        {
            break;
        }
    }
    let mut out = lines.join("\n");
    if src.ends_with('\n') {
        out.push('\n');
    }
    out
}

pub(super) fn is_bb_label(lbl: &str) -> bool {
    lbl.rsplit_once("_bb")
        .is_some_and(|(_, d)| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
}

pub(super) fn parse_ix_disp(op: &str) -> Option<i8> {
    let inner = op.trim().strip_prefix('(')?.strip_suffix(')')?.trim();
    let rest = inner
        .strip_prefix("ix")
        .or_else(|| inner.strip_prefix("IX"))?
        .trim();
    if rest.is_empty() {
        return Some(0);
    }
    if let Some(p) = rest.strip_prefix('+') {
        p.trim().parse::<i8>().ok()
    } else if let Some(m) = rest.strip_prefix('-') {
        m.trim().parse::<i8>().ok().and_then(|v| v.checked_neg())
    } else {
        None
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum MemAddr {
    Fixed(u16),
    Uchiguri(u16),
}

impl MemAddr {
    pub(super) fn add(self, off: u16) -> Self {
        match self {
            MemAddr::Fixed(a) => MemAddr::Fixed(a.wrapping_add(off)),
            MemAddr::Uchiguri(o) => MemAddr::Uchiguri(o.wrapping_add(off)),
        }
    }

    pub(super) fn sub_one(self) -> Option<Self> {
        match self {
            MemAddr::Fixed(a) if a > 0 => Some(MemAddr::Fixed(a - 1)),
            MemAddr::Uchiguri(o) if o > 0 => Some(MemAddr::Uchiguri(o - 1)),
            _ => None,
        }
    }

    pub(super) fn format_imm(self) -> String {
        match self {
            MemAddr::Fixed(a) => format!("0x{a:04X}"),
            MemAddr::Uchiguri(0) => "__aun_ram".to_string(),
            MemAddr::Uchiguri(o) => format!("__aun_ram + {o}"),
        }
    }
}

pub(super) fn detect_uchiguri_base(lines: &[String]) -> Option<u16> {
    for line in lines {
        let code = line.split(';').next().unwrap_or("").trim();
        if let Some(rest) = code.strip_prefix("__aun_ram")
            && let Some(val) = rest.trim().strip_prefix('=')
        {
            let v = val.trim();
            if let Some(hex) = v.strip_prefix("0x").or_else(|| v.strip_prefix("0X"))
                && let Ok(n) = u16::from_str_radix(hex, 16)
            {
                return Some(n);
            }
        }
    }
    None
}

pub(super) fn parse_imm_uchiguri_off(op: &str) -> Option<u16> {
    let trimmed = op.trim();
    let (has_hash, t) = match trimmed.strip_prefix('#') {
        Some(rest) => (true, rest.trim()),
        None => (false, trimmed),
    };
    if t == "__aun_ram" {
        return Some(0);
    }
    let inner = if has_hash {
        t.strip_prefix('(')
            .and_then(|s| s.strip_suffix(')'))
            .map(|s| s.trim())
            .unwrap_or(t)
    } else {
        t
    };
    if inner == "__aun_ram" {
        return Some(0);
    }
    if let Some(rest) = inner.strip_prefix("__aun_ram")
        && let Some(off_str) = rest.trim().strip_prefix('+')
    {
        return off_str.trim().parse::<u16>().ok();
    }
    None
}

pub(super) fn parse_mem_ref(op: &str) -> Option<MemAddr> {
    let inner = op.trim().strip_prefix('(')?.strip_suffix(')')?.trim();
    if let Some(hex) = inner.strip_prefix("0x").or_else(|| inner.strip_prefix("0X")) {
        return u16::from_str_radix(hex, 16).ok().map(MemAddr::Fixed);
    }
    if inner == "__aun_ram" {
        return Some(MemAddr::Uchiguri(0));
    }
    if let Some(rest) = inner.strip_prefix("__aun_ram")
        && let Some(off_str) = rest.trim().strip_prefix('+')
    {
        return off_str.trim().parse::<u16>().ok().map(MemAddr::Uchiguri);
    }
    None
}

pub(super) fn in_ranges(addr: u16, ranges: &[(u16, u16)]) -> bool {
    ranges.iter().any(|&(lo, hi)| addr >= lo && addr < hi)
}

pub(super) fn in_private_ranges(
    addr: MemAddr,
    uchiguri_base: Option<u16>,
    ranges: &[(u16, u16)],
) -> bool {
    match (addr, uchiguri_base) {
        (MemAddr::Uchiguri(off), Some(base)) => in_ranges(base.wrapping_add(off), ranges),
        (MemAddr::Fixed(a), None) => in_ranges(a, ranges),
        _ => false,
    }
}

pub(crate) fn is_intra_fn_label(lbl: &str, cur_fn_name: Option<&str>) -> bool {
    if let Some(fn_lbl) = cur_fn_name
        && let Some(suffix) = lbl
            .strip_prefix(fn_lbl.strip_suffix("__ix").unwrap_or(fn_lbl))
            .and_then(|s| s.strip_prefix('_'))
    {
        if (suffix.len() > 2
            && suffix.starts_with("bb")
            && suffix[2..].bytes().all(|b| b.is_ascii_digit()))
            || (suffix.len() > 4
                && suffix.starts_with("tail")
                && suffix[4..].bytes().all(|b| b.is_ascii_digit()))
        {
            return true;
        }
        if suffix.rsplit_once('_').is_some_and(|(tag, digits)| {
            !tag.is_empty() && !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
        }) {
            return true;
        }
    }
    lbl.starts_with("__scan_cold_save_")
        || lbl.starts_with("__scan_reenter_")
        || lbl.contains("_bb")
        || lbl.contains("_tail")
        || lbl.contains("_sat_")
        || lbl.contains("_cmp_")
        || lbl.contains("_cmp32_")
        || lbl.contains("_mul8_")
        || lbl.contains("_mul16_")
        || lbl.contains("_m16x8_")
        || lbl.contains("_memset_")
        || lbl.contains("_div_")
        || lbl.contains("_udiv_")
        || lbl.contains("_sdiv_")
        || lbl.contains("_lp_")
        || lbl.contains("_sk_")
        || lbl.contains("_dn_")
        || lbl.contains("_sh")
        || lbl.contains("_palloc_")
        || lbl.contains("_pfree_")
        || lbl.contains("_pget_")
        || lbl.contains("_tr_")
        || lbl.contains("_idx_")
        || lbl.contains("_assert_")
        || lbl.contains("_cast_")
        || lbl.contains("_fxint_")
}

pub(super) fn partition_fn_spans<'a>(
    kinds: &[LineKind],
    fn_private_ranges: &'a HashMap<String, Vec<(u16, u16)>>,
) -> Vec<(usize, usize, Option<&'a [(u16, u16)]>)> {
    let mut fn_spans: Vec<(usize, usize, Option<&'a [(u16, u16)]>)> = Vec::new();
    let mut cur_start = 0usize;
    let mut cur_range: Option<&'a [(u16, u16)]> = None;
    let mut cur_fn_name: Option<&str> = None;
    for (idx, kind) in kinds.iter().enumerate() {
        if let LineKind::Label(lbl) = kind
            && let Some(ranges) = fn_private_ranges.get(lbl)
        {
            if idx > cur_start {
                fn_spans.push((cur_start, idx, cur_range));
            }
            cur_start = idx;
            cur_range = Some(ranges.as_slice());
            cur_fn_name = Some(lbl.as_str());
        } else if let LineKind::Label(lbl) = kind
            && cur_range.is_some()
            && !is_intra_fn_label(lbl, cur_fn_name)
        {
            if idx > cur_start {
                fn_spans.push((cur_start, idx, cur_range));
            }
            cur_start = idx;
            cur_range = None;
            cur_fn_name = None;
        }
    }
    if cur_start < kinds.len() {
        fn_spans.push((cur_start, kinds.len(), cur_range));
    }
    fn_spans
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum LineKind {
    EmptyOrComment,
    Label(String),
    Directive,
    Inst { mnemonic: String, ops: Vec<String> },
}

pub(super) fn classify_line(raw: &str) -> LineKind {
    let code = raw.split(';').next().unwrap_or("").trim();
    if code.is_empty() {
        return LineKind::EmptyOrComment;
    }
    if code.starts_with('.') {
        return LineKind::Directive;
    }
    if let Some(lbl) = code.strip_suffix(':') {
        let lbl = lbl.trim();
        if !lbl.is_empty() && !lbl.contains(' ') {
            return LineKind::Label(lbl.to_string());
        }
    }
    // If a line has `label: inst`, treat as directive/opaque boundary
    if code.contains(':') {
        return LineKind::Directive;
    }
    let mut parts = code.splitn(2, |c: char| c.is_whitespace());
    let mnemonic = parts.next().unwrap().to_ascii_lowercase();
    let tail = parts.next().unwrap_or("").trim();
    let ops = if tail.is_empty() {
        Vec::new()
    } else {
        tail.split(',').map(|s| s.trim().to_string()).collect()
    };
    LineKind::Inst { mnemonic, ops }
}

pub(super) fn is_mem_operand(op: &str) -> bool {
    let t = op.trim();
    t.starts_with('(') && t.ends_with(')') && !t.eq_ignore_ascii_case("(hl)")
        && !t.eq_ignore_ascii_case("(bc)")
        && !t.eq_ignore_ascii_case("(de)")
        && !t.eq_ignore_ascii_case("(sp)")
        && !t.eq_ignore_ascii_case("(c)")
}

pub(super) fn is_zero_imm(op: &str) -> bool {
    let t = op.trim().trim_start_matches('#');
    matches!(t, "0" | "0x0" | "0x00" | "0X0" | "0X00" | "$0" | "$00")
}

pub(super) fn reads_flags(kind: &LineKind) -> bool {
    match kind {
        LineKind::Inst { mnemonic, ops } => match mnemonic.as_str() {
            "jr" | "jp" | "call" => ops.len() == 2,
            "ret" => ops.len() == 1,
            "adc" | "sbc" | "rla" | "rra" | "rl" | "rr" | "ccf" | "daa" => true,
            "push" => ops.first().is_some_and(|o| o.eq_ignore_ascii_case("af")),
            _ => false,
        },
        _ => false,
    }
}

/// Returns `true` if `kind` unconditionally overwrites all arithmetic flags (`S`, `Z`, `P/V`, `C`)
/// without reading any prior flag.
pub(super) fn clobbers_all_flags(kind: &LineKind) -> bool {
    match kind {
        LineKind::Inst { mnemonic, ops } => match mnemonic.as_str() {
            "and" | "or" | "xor" | "cp" | "sub" | "neg" => true,
            "add" => ops.first().is_some_and(|o| o.eq_ignore_ascii_case("a")),
            "srl" | "sra" | "sla" => true,
            _ => false,
        },
        _ => false,
    }
}

/// Returns `true` if modifying flags at `start` cannot affect any subsequent flag consumer
/// within the current basic block.
pub(super) fn flags_dead_after(kinds: &[LineKind], start: usize) -> bool {
    let mut k = start;
    while k < kinds.len() {
        match &kinds[k] {
            LineKind::EmptyOrComment => k += 1,
            LineKind::Label(_) | LineKind::Directive => return false,
            kind @ LineKind::Inst { mnemonic, .. } => {
                if reads_flags(kind) {
                    return false;
                }
                if clobbers_all_flags(kind) {
                    return true;
                }
                // Flag-preserving instructions: keep scanning forward
                if matches!(mnemonic.as_str(), "ld" | "push" | "pop" | "ex" | "nop" | "di" | "ei") {
                    k += 1;
                    continue;
                }
                // Unconditional control transfer or unknown flag behavior: conservative
                return false;
            }
        }
    }
    false
}

pub(super) fn next_non_comment(kinds: &[LineKind], start: usize) -> Option<usize> {
    (start..kinds.len()).find(|&idx| !matches!(kinds[idx], LineKind::EmptyOrComment))
}

pub(super) fn build_label_positions(kinds: &[LineKind]) -> HashMap<String, usize> {
    let mut map = HashMap::new();
    for (idx, kind) in kinds.iter().enumerate() {
        if let LineKind::Label(lbl) = kind {
            map.insert(lbl.clone(), idx);
        }
    }
    map
}

pub(super) fn inst_reads_a(mnemonic: &str, ops: &[String]) -> bool {
    match mnemonic {
        "xor" | "sub" if ops.len() == 1 && ops[0].eq_ignore_ascii_case("a") => false,
        "and" | "or" | "xor" | "cp" | "sub" | "sbc" | "adc" | "add" | "neg" | "cpl" | "rla"
        | "rra" | "rlca" | "rrca" | "daa" => true,
        "inc" | "dec" | "sla" | "sra" | "srl" | "rl" | "rr" => {
            ops.first().is_some_and(|o| o.eq_ignore_ascii_case("a"))
        }
        "bit" | "set" | "res" => ops.get(1).is_some_and(|o| o.eq_ignore_ascii_case("a")),
        "ld" => ops.get(1).is_some_and(|o| o.eq_ignore_ascii_case("a")),
        "push" => ops.first().is_some_and(|o| o.eq_ignore_ascii_case("af")),
        "out" => ops.get(1).is_some_and(|o| o.eq_ignore_ascii_case("a")),
        _ => true,
    }
}

pub(super) fn inst_writes_a_without_reading(mnemonic: &str, ops: &[String]) -> bool {
    match mnemonic {
        "ld" => {
            ops.len() == 2
                && ops[0].eq_ignore_ascii_case("a")
                && !ops[1].eq_ignore_ascii_case("a")
        }
        "xor" | "sub" => ops.len() == 1 && ops[0].eq_ignore_ascii_case("a"),
        "pop" => ops.first().is_some_and(|o| o.eq_ignore_ascii_case("af")),
        "in" => ops.first().is_some_and(|o| o.eq_ignore_ascii_case("a")),
        _ => false,
    }
}

/// Returns `true` if `A` is provably overwritten before being read on all reachable
/// intra-function control-flow paths starting at `start` (bounded DFS across conditional branches).
pub(super) fn reg_a_dead_after(
    kinds: &[LineKind],
    label_pos: &HashMap<String, usize>,
    start: usize,
) -> bool {
    let mut visited = HashSet::new();
    let mut budget = 48usize;
    reg_a_dead_dfs(kinds, label_pos, start, &mut visited, &mut budget)
}

fn reg_a_dead_dfs(
    kinds: &[LineKind],
    label_pos: &HashMap<String, usize>,
    mut k: usize,
    visited: &mut HashSet<usize>,
    budget: &mut usize,
) -> bool {
    while k < kinds.len() {
        if *budget == 0 {
            return false;
        }
        *budget -= 1;
        if !visited.insert(k) {
            return true;
        }
        match &kinds[k] {
            LineKind::EmptyOrComment => k += 1,
            LineKind::Label(lbl) => {
                if is_bb_label(lbl) {
                    k += 1;
                } else {
                    return false;
                }
            }
            LineKind::Directive => return false,
            LineKind::Inst { mnemonic, ops } => {
                if inst_writes_a_without_reading(mnemonic, ops) {
                    return true;
                }
                if matches!(mnemonic.as_str(), "jp" | "jr") {
                    if ops.len() == 1
                        && is_bb_label(&ops[0])
                        && let Some(&tp) = label_pos.get(&ops[0])
                    {
                        k = tp + 1;
                        continue;
                    }
                    if ops.len() == 2
                        && is_bb_label(&ops[1])
                        && let Some(&tp) = label_pos.get(&ops[1])
                    {
                        return reg_a_dead_dfs(kinds, label_pos, k + 1, visited, budget)
                            && reg_a_dead_dfs(kinds, label_pos, tp + 1, visited, budget);
                    }
                    return false;
                }
                if inst_reads_a(mnemonic, ops) {
                    return false;
                }
                k += 1;
            }
        }
    }
    false
}

/// Returns `true` if both `H` and `L` are provably overwritten before being read on all reachable
/// intra-function control-flow paths starting at `start` (bounded DFS across conditional branches).
pub(super) fn reg_hl_dead_after(
    kinds: &[LineKind],
    label_pos: &HashMap<String, usize>,
    start: usize,
) -> bool {
    let mut visited = HashSet::new();
    let mut budget = 48usize;
    reg_hl_dead_dfs(kinds, label_pos, start, false, false, &mut visited, &mut budget)
}

fn reg_hl_dead_dfs(
    kinds: &[LineKind],
    label_pos: &HashMap<String, usize>,
    mut k: usize,
    mut h_dead: bool,
    mut l_dead: bool,
    visited: &mut HashSet<(usize, u8)>,
    budget: &mut usize,
) -> bool {
    while k < kinds.len() {
        if h_dead && l_dead {
            return true;
        }
        if *budget == 0 {
            return false;
        }
        *budget -= 1;
        let state = u8::from(h_dead) | (u8::from(l_dead) << 1);
        if !visited.insert((k, state)) {
            return true;
        }
        match &kinds[k] {
            LineKind::EmptyOrComment => k += 1,
            LineKind::Label(lbl) => {
                if is_bb_label(lbl) {
                    k += 1;
                } else {
                    return false;
                }
            }
            LineKind::Directive => return false,
            LineKind::Inst { mnemonic, ops } => {
                if ops.iter().any(|o| o.eq_ignore_ascii_case("(hl)")) {
                    return false;
                }
                if matches!(mnemonic.as_str(), "jp" | "jr") {
                    if ops.len() == 1
                        && is_bb_label(&ops[0])
                        && let Some(&tp) = label_pos.get(&ops[0])
                    {
                        k = tp + 1;
                        continue;
                    }
                    if ops.len() == 2
                        && is_bb_label(&ops[1])
                        && let Some(&tp) = label_pos.get(&ops[1])
                    {
                        return reg_hl_dead_dfs(kinds, label_pos, k + 1, h_dead, l_dead, visited, budget)
                            && reg_hl_dead_dfs(
                                kinds,
                                label_pos,
                                tp + 1,
                                h_dead,
                                l_dead,
                                visited,
                                budget,
                            );
                    }
                    return false;
                }
                if mnemonic == "ld" && ops.len() == 2 {
                    let dst = ops[0].to_ascii_lowercase();
                    let src = ops[1].to_ascii_lowercase();
                    if src == "hl" || (src == "h" && !h_dead) || (src == "l" && !l_dead) {
                        return false;
                    }
                    match dst.as_str() {
                        "hl" => return true,
                        "h" => {
                            h_dead = true;
                            k += 1;
                            continue;
                        }
                        "l" => {
                            l_dead = true;
                            k += 1;
                            continue;
                        }
                        _ => {
                            k += 1;
                            continue;
                        }
                    }
                }
                if mnemonic == "pop" && ops.len() == 1 && ops[0].eq_ignore_ascii_case("hl") {
                    return true;
                }
                // Instructions guaranteed not to read/clobber live H or L:
                if matches!(
                    mnemonic.as_str(),
                    "push" | "pop" | "xor" | "and" | "or" | "cp" | "sub" | "adc" | "sbc" | "add"
                        | "inc" | "dec" | "sla" | "sra" | "srl" | "rl" | "rr" | "bit" | "set"
                        | "res" | "neg" | "cpl" | "rla" | "rra" | "rlca" | "rrca" | "scf" | "ccf"
                        | "nop"
                ) && !ops.iter().any(|o| {
                    matches!(
                        o.to_ascii_lowercase().as_str(),
                        "h" | "l" | "hl" | "(hl)"
                    )
                }) {
                    k += 1;
                    continue;
                }
                return false;
            }
        }
    }
    h_dead && l_dead
}

fn op_is_reg8(op: &str, reg: char) -> bool {
    let t = op.trim();
    t.len() == 1 && t.chars().next().is_some_and(|c| c.to_ascii_lowercase() == reg)
}

fn reg8_in_pair(rp: &str, reg: char) -> bool {
    match rp.trim().to_ascii_lowercase().as_str() {
        "af" => reg == 'a',
        "bc" => matches!(reg, 'b' | 'c'),
        "de" => matches!(reg, 'd' | 'e'),
        "hl" => matches!(reg, 'h' | 'l'),
        _ => false,
    }
}

fn op_reads_indirect_regs(op: &str, reg: char) -> bool {
    match op.trim().to_ascii_lowercase().as_str() {
        "(hl)" => matches!(reg, 'h' | 'l'),
        "(de)" => matches!(reg, 'd' | 'e'),
        "(bc)" => matches!(reg, 'b' | 'c'),
        "(c)" => reg == 'c',
        _ => false,
    }
}

/// Returns `(reads, writes)` for 8-bit register `reg` (`'a','b','c','d','e','h','l'`) on `line`.
pub(super) fn inst_reads_writes_reg8(line: &str, reg: char) -> (bool, bool) {
    let reg = reg.to_ascii_lowercase();
    if !matches!(reg, 'a' | 'b' | 'c' | 'd' | 'e' | 'h' | 'l') {
        return (true, true);
    }
    match classify_line(line) {
        LineKind::EmptyOrComment => (false, false),
        LineKind::Label(_) | LineKind::Directive => (true, true),
        LineKind::Inst { mnemonic, ops } => match mnemonic.as_str() {
            "call" | "ret" | "reti" | "retn" | "rst" | "exx" | "halt" => (true, true),
            "jp" | "jr" => {
                if ops.iter().any(|o| o.trim().starts_with('(')) {
                    (true, true)
                } else {
                    (false, false)
                }
            }
            "djnz" => (reg == 'b', reg == 'b'),
            "nop" | "di" | "ei" | "scf" | "ccf" => (false, false),
            "daa" | "cpl" | "neg" | "rla" | "rra" | "rlca" | "rrca" => {
                (reg == 'a', reg == 'a')
            }
            "rld" | "rrd" => (matches!(reg, 'a' | 'h' | 'l'), reg == 'a'),
            "ldi" | "ldd" | "ldir" | "lddr" => {
                let t = matches!(reg, 'b' | 'c' | 'd' | 'e' | 'h' | 'l');
                (t, t)
            }
            "cpi" | "cpd" | "cpir" | "cpdr" => (
                matches!(reg, 'a' | 'b' | 'c' | 'h' | 'l'),
                matches!(reg, 'b' | 'c' | 'h' | 'l'),
            ),
            "ini" | "ind" | "inir" | "indr" | "outi" | "outd" | "otir" | "otdr" => {
                (matches!(reg, 'b' | 'c' | 'h' | 'l'), matches!(reg, 'b' | 'h' | 'l'))
            }
            "push" if ops.len() == 1 => (reg8_in_pair(&ops[0], reg), false),
            "pop" if ops.len() == 1 => (false, reg8_in_pair(&ops[0], reg)),
            "ex" if ops.len() == 2 => {
                let a = ops[0].trim().to_ascii_lowercase();
                let b = ops[1].trim().to_ascii_lowercase();
                if (a == "de" && b == "hl") || (a == "hl" && b == "de") {
                    let t = matches!(reg, 'd' | 'e' | 'h' | 'l');
                    (t, t)
                } else if (a == "(sp)" && b == "hl") || (a == "hl" && b == "(sp)") {
                    let t = matches!(reg, 'h' | 'l');
                    (t, t)
                } else if (a == "(sp)" && (b == "ix" || b == "iy"))
                    || ((a == "ix" || a == "iy") && b == "(sp)")
                {
                    (false, false)
                } else if a == "af" && b == "af'" {
                    (reg == 'a', reg == 'a')
                } else {
                    (true, true)
                }
            }
            "inc" | "dec" if ops.len() == 1 => {
                if reg8_in_pair(&ops[0], reg) && !ops[0].eq_ignore_ascii_case("af") {
                    (true, true)
                } else if op_is_reg8(&ops[0], reg) {
                    (true, true)
                } else if op_reads_indirect_regs(&ops[0], reg) {
                    (true, false)
                } else {
                    (false, false)
                }
            }
            "add" | "adc" | "sbc" => {
                if ops.len() == 2 && ops[0].eq_ignore_ascii_case("hl") {
                    let r = matches!(reg, 'h' | 'l') || reg8_in_pair(&ops[1], reg);
                    let w = matches!(reg, 'h' | 'l');
                    (r, w)
                } else if mnemonic == "add"
                    && ops.len() == 2
                    && (ops[0].eq_ignore_ascii_case("ix") || ops[0].eq_ignore_ascii_case("iy"))
                {
                    (reg8_in_pair(&ops[1], reg), false)
                } else if ops.len() == 1 || (ops.len() == 2 && ops[0].eq_ignore_ascii_case("a")) {
                    let src = ops.last().unwrap();
                    let r = reg == 'a'
                        || op_is_reg8(src, reg)
                        || op_reads_indirect_regs(src, reg);
                    (r, reg == 'a')
                } else {
                    (true, true)
                }
            }
            "sub" | "and" | "or" | "xor" | "cp" => {
                if ops.len() == 1 || (ops.len() == 2 && ops[0].eq_ignore_ascii_case("a")) {
                    let src = ops.last().unwrap();
                    if matches!(mnemonic.as_str(), "xor" | "sub") && src.eq_ignore_ascii_case("a")
                    {
                        (false, reg == 'a')
                    } else {
                        let r = reg == 'a'
                            || op_is_reg8(src, reg)
                            || op_reads_indirect_regs(src, reg);
                        let w = mnemonic != "cp" && reg == 'a';
                        (r, w)
                    }
                } else {
                    (true, true)
                }
            }
            "sla" | "sra" | "srl" | "rl" | "rr" | "rlc" | "rrc" if ops.len() == 1 => {
                if op_is_reg8(&ops[0], reg) {
                    (true, true)
                } else if op_reads_indirect_regs(&ops[0], reg) {
                    (true, false)
                } else {
                    (false, false)
                }
            }
            "bit" if ops.len() == 2 => {
                let r = op_is_reg8(&ops[1], reg) || op_reads_indirect_regs(&ops[1], reg);
                (r, false)
            }
            "set" | "res" if ops.len() == 2 => {
                if op_is_reg8(&ops[1], reg) {
                    (true, true)
                } else if op_reads_indirect_regs(&ops[1], reg) {
                    (true, false)
                } else {
                    (false, false)
                }
            }
            "in" => {
                if ops.len() == 2 {
                    (op_reads_indirect_regs(&ops[1], reg), op_is_reg8(&ops[0], reg))
                } else if ops.len() == 1 {
                    (op_reads_indirect_regs(&ops[0], reg), false)
                } else {
                    (true, true)
                }
            }
            "out" if ops.len() == 2 => {
                let r = op_reads_indirect_regs(&ops[0], reg) || op_is_reg8(&ops[1], reg);
                (r, false)
            }
            "ld" if ops.len() == 2 => {
                let dst = &ops[0];
                let src = &ops[1];
                let r = op_is_reg8(src, reg)
                    || reg8_in_pair(src, reg)
                    || op_reads_indirect_regs(src, reg)
                    || op_reads_indirect_regs(dst, reg);
                let w = op_is_reg8(dst, reg) || reg8_in_pair(dst, reg);
                (r, w)
            }
            _ => (true, true),
        },
    }
}

/// Returns `true` if `reg` is provably overwritten before being read on all reachable
/// intra-function paths starting at `start_idx` within `fn_start..fn_end` (bounded DFS).
pub(super) fn reg8_dead_after(
    lines: &[String],
    fn_start: usize,
    fn_end: usize,
    start_idx: usize,
    reg: char,
) -> bool {
    let reg = reg.to_ascii_lowercase();
    if start_idx < fn_start || start_idx >= fn_end || fn_end > lines.len() {
        return false;
    }
    let fn_lines = &lines[fn_start..fn_end];
    let fn_kinds: Vec<LineKind> = fn_lines.iter().map(|l| classify_line(l)).collect();
    let mut label_pos: HashMap<&str, usize> = HashMap::new();
    for (idx, kind) in fn_kinds.iter().enumerate() {
        if let LineKind::Label(lbl) = kind {
            label_pos.insert(lbl.as_str(), idx);
        }
    }
    let mut visited = HashSet::new();
    let mut budget = 256usize;
    reg8_dead_dfs(
        fn_lines,
        &fn_kinds,
        &label_pos,
        start_idx - fn_start,
        reg,
        &mut visited,
        &mut budget,
    )
}

fn reg8_dead_dfs(
    fn_lines: &[String],
    fn_kinds: &[LineKind],
    label_pos: &HashMap<&str, usize>,
    mut k: usize,
    reg: char,
    visited: &mut HashSet<usize>,
    budget: &mut usize,
) -> bool {
    while k < fn_kinds.len() {
        if *budget == 0 {
            return false;
        }
        *budget -= 1;
        if !visited.insert(k) {
            return true;
        }
        let t = fn_lines[k].trim();
        if t == "; @aun:volatile_begin" || t == "; @aun:asm_begin" {
            return false;
        }
        match &fn_kinds[k] {
            LineKind::EmptyOrComment | LineKind::Label(_) => k += 1,
            LineKind::Directive => return false,
            LineKind::Inst { mnemonic, ops } => {
                if matches!(mnemonic.as_str(), "jp" | "jr") {
                    if ops.len() == 1 && !ops[0].trim().starts_with('(') {
                        if let Some(&tp) = label_pos.get(ops[0].as_str()) {
                            k = tp + 1;
                            continue;
                        }
                        return false;
                    }
                    if ops.len() == 2
                        && let Some(&tp) = label_pos.get(ops[1].as_str())
                    {
                        return reg8_dead_dfs(
                            fn_lines, fn_kinds, label_pos, k + 1, reg, visited, budget,
                        ) && reg8_dead_dfs(
                            fn_lines, fn_kinds, label_pos, tp + 1, reg, visited, budget,
                        );
                    }
                    return false;
                }
                if mnemonic == "djnz" {
                    if reg == 'b' {
                        return false;
                    }
                    if ops.len() == 1
                        && let Some(&tp) = label_pos.get(ops[0].as_str())
                    {
                        return reg8_dead_dfs(
                            fn_lines, fn_kinds, label_pos, k + 1, reg, visited, budget,
                        ) && reg8_dead_dfs(
                            fn_lines, fn_kinds, label_pos, tp + 1, reg, visited, budget,
                        );
                    }
                    return false;
                }
                let (reads, writes) = inst_reads_writes_reg8(&fn_lines[k], reg);
                if reads {
                    return false;
                }
                if writes {
                    return true;
                }
                k += 1;
            }
        }
    }
    false
}

/// Returns `true` if both `hi` and `lo` 8-bit registers are provably overwritten before being read
/// on all reachable intra-function paths starting at `start_idx` within `fn_start..fn_end` (bounded DFS).
pub(super) fn reg_pair_dead_after(
    lines: &[String],
    fn_start: usize,
    fn_end: usize,
    start_idx: usize,
    hi: char,
    lo: char,
) -> bool {
    let hi = hi.to_ascii_lowercase();
    let lo = lo.to_ascii_lowercase();
    if start_idx < fn_start || start_idx >= fn_end || fn_end > lines.len() {
        return false;
    }
    let fn_lines = &lines[fn_start..fn_end];
    let fn_kinds: Vec<LineKind> = fn_lines.iter().map(|l| classify_line(l)).collect();
    let mut label_pos: HashMap<&str, usize> = HashMap::new();
    for (idx, kind) in fn_kinds.iter().enumerate() {
        if let LineKind::Label(lbl) = kind {
            label_pos.insert(lbl.as_str(), idx);
        }
    }
    let mut visited = HashSet::new();
    let mut budget = 256usize;
    reg_pair_dead_dfs(
        fn_lines,
        &fn_kinds,
        &label_pos,
        start_idx - fn_start,
        hi,
        lo,
        false,
        false,
        &mut visited,
        &mut budget,
    )
}

#[allow(clippy::too_many_arguments)]
fn reg_pair_dead_dfs(
    fn_lines: &[String],
    fn_kinds: &[LineKind],
    label_pos: &HashMap<&str, usize>,
    mut k: usize,
    hi: char,
    lo: char,
    mut hi_dead: bool,
    mut lo_dead: bool,
    visited: &mut HashSet<(usize, u8)>,
    budget: &mut usize,
) -> bool {
    while k < fn_kinds.len() {
        if hi_dead && lo_dead {
            return true;
        }
        if *budget == 0 {
            return false;
        }
        *budget -= 1;
        let state = u8::from(hi_dead) | (u8::from(lo_dead) << 1);
        if !visited.insert((k, state)) {
            return true;
        }
        let t = fn_lines[k].trim();
        if t == "; @aun:volatile_begin" || t == "; @aun:asm_begin" {
            return false;
        }
        match &fn_kinds[k] {
            LineKind::EmptyOrComment | LineKind::Label(_) => k += 1,
            LineKind::Directive => return false,
            LineKind::Inst { mnemonic, ops } => {
                if matches!(mnemonic.as_str(), "jp" | "jr") {
                    if ops.len() == 1 && !ops[0].trim().starts_with('(') {
                        if let Some(&tp) = label_pos.get(ops[0].as_str()) {
                            k = tp + 1;
                            continue;
                        }
                        return false;
                    }
                    if ops.len() == 2
                        && let Some(&tp) = label_pos.get(ops[1].as_str())
                    {
                        return reg_pair_dead_dfs(
                            fn_lines, fn_kinds, label_pos, k + 1, hi, lo, hi_dead, lo_dead,
                            visited, budget,
                        ) && reg_pair_dead_dfs(
                            fn_lines, fn_kinds, label_pos, tp + 1, hi, lo, hi_dead, lo_dead,
                            visited, budget,
                        );
                    }
                    return false;
                }
                if mnemonic == "djnz" {
                    if (hi == 'b' && !hi_dead) || (lo == 'b' && !lo_dead) {
                        return false;
                    }
                    if ops.len() == 1
                        && let Some(&tp) = label_pos.get(ops[0].as_str())
                    {
                        return reg_pair_dead_dfs(
                            fn_lines, fn_kinds, label_pos, k + 1, hi, lo, hi_dead, lo_dead,
                            visited, budget,
                        ) && reg_pair_dead_dfs(
                            fn_lines, fn_kinds, label_pos, tp + 1, hi, lo, hi_dead, lo_dead,
                            visited, budget,
                        );
                    }
                    return false;
                }
                let (r_hi, w_hi) = inst_reads_writes_reg8(&fn_lines[k], hi);
                let (r_lo, w_lo) = inst_reads_writes_reg8(&fn_lines[k], lo);
                if (r_hi && !hi_dead) || (r_lo && !lo_dead) {
                    return false;
                }
                if w_hi {
                    hi_dead = true;
                }
                if w_lo {
                    lo_dead = true;
                }
                k += 1;
            }
        }
    }
    hi_dead && lo_dead
}

/// Returns `true` if the carry flag (`C`) is provably overwritten before being read on all
/// reachable intra-function paths starting at `start_idx` within `fn_start..fn_end` (bounded DFS).
pub(super) fn flags_carry_dead_after(
    lines: &[String],
    fn_start: usize,
    fn_end: usize,
    start_idx: usize,
) -> bool {
    if start_idx < fn_start || start_idx >= fn_end || fn_end > lines.len() {
        return false;
    }
    let fn_lines = &lines[fn_start..fn_end];
    let fn_kinds: Vec<LineKind> = fn_lines.iter().map(|l| classify_line(l)).collect();
    let is_aun_helper =
        matches!(fn_kinds.first(), Some(LineKind::Label(l)) if l.starts_with("__aun_"));
    let mut label_pos: HashMap<&str, usize> = HashMap::new();
    for (idx, kind) in fn_kinds.iter().enumerate() {
        if let LineKind::Label(lbl) = kind {
            label_pos.insert(lbl.as_str(), idx);
        }
    }
    let mut visited = HashSet::new();
    let mut budget = 256usize;
    flags_carry_dead_dfs(
        fn_lines,
        &fn_kinds,
        &label_pos,
        start_idx - fn_start,
        is_aun_helper,
        &mut visited,
        &mut budget,
    )
}

fn flags_carry_dead_dfs(
    fn_lines: &[String],
    fn_kinds: &[LineKind],
    label_pos: &HashMap<&str, usize>,
    mut k: usize,
    is_aun_helper: bool,
    visited: &mut HashSet<usize>,
    budget: &mut usize,
) -> bool {
    while k < fn_kinds.len() {
        if *budget == 0 {
            return false;
        }
        *budget -= 1;
        if !visited.insert(k) {
            return true;
        }
        let t = fn_lines[k].trim();
        if t == "; @aun:volatile_begin" || t == "; @aun:asm_begin" {
            return false;
        }
        match &fn_kinds[k] {
            LineKind::EmptyOrComment | LineKind::Label(_) => k += 1,
            LineKind::Directive => return false,
            LineKind::Inst { mnemonic, ops } => match mnemonic.as_str() {
                "adc" | "sbc" | "rla" | "rra" | "rl" | "rr" | "scf" | "ccf" | "daa" => {
                    return false;
                }
                "push" if ops.first().is_some_and(|o| o.eq_ignore_ascii_case("af")) => {
                    return false;
                }
                "pop" if ops.first().is_some_and(|o| o.eq_ignore_ascii_case("af")) => {
                    return false;
                }
                "ex" if ops
                    .iter()
                    .any(|o| o.trim().to_ascii_lowercase().starts_with("af")) =>
                {
                    return false;
                }
                "ret" => {
                    if ops.is_empty() {
                        return !is_aun_helper;
                    }
                    if ops.len() == 1
                        && !ops[0].eq_ignore_ascii_case("c")
                        && !ops[0].eq_ignore_ascii_case("nc")
                        && !is_aun_helper
                    {
                        k += 1;
                        continue;
                    }
                    return false;
                }
                "reti" => return !is_aun_helper,
                "jp" | "jr" => {
                    if ops.len() == 1 && !ops[0].trim().starts_with('(') {
                        if let Some(&tp) = label_pos.get(ops[0].as_str()) {
                            k = tp + 1;
                            continue;
                        }
                        return false;
                    }
                    if ops.len() == 2
                        && !ops[0].eq_ignore_ascii_case("c")
                        && !ops[0].eq_ignore_ascii_case("nc")
                        && let Some(&tp) = label_pos.get(ops[1].as_str())
                    {
                        return flags_carry_dead_dfs(
                            fn_lines,
                            fn_kinds,
                            label_pos,
                            k + 1,
                            is_aun_helper,
                            visited,
                            budget,
                        ) && flags_carry_dead_dfs(
                            fn_lines,
                            fn_kinds,
                            label_pos,
                            tp + 1,
                            is_aun_helper,
                            visited,
                            budget,
                        );
                    }
                    return false;
                }
                "djnz" => {
                    if ops.len() == 1
                        && let Some(&tp) = label_pos.get(ops[0].as_str())
                    {
                        return flags_carry_dead_dfs(
                            fn_lines,
                            fn_kinds,
                            label_pos,
                            k + 1,
                            is_aun_helper,
                            visited,
                            budget,
                        ) && flags_carry_dead_dfs(
                            fn_lines,
                            fn_kinds,
                            label_pos,
                            tp + 1,
                            is_aun_helper,
                            visited,
                            budget,
                        );
                    }
                    return false;
                }
                "or" | "and" | "xor" | "cp" | "sub" | "add" | "sla" | "sra" | "srl"
                | "rlca" | "rrca" | "rlc" | "rrc" | "neg" => return true,
                "ld" | "inc" | "dec" | "bit" | "set" | "res" | "cpl" | "push" | "pop" | "ex"
                | "nop" | "di" | "ei" | "in" | "out" | "ldi" | "ldd" | "ldir" | "lddr"
                | "cpi" | "cpd" | "cpir" | "cpdr" | "rld" | "rrd" => k += 1,
                _ => return false,
            },
        }
    }
    false
}
