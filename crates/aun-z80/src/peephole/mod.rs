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
    eliminate_dead_bb_scratch_stores, eliminate_dead_scratch_stores,
    promote_single_block_spills_to_bc, promote_single_block_spills_to_stack,
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
        let (next2, changed2) = if fn_private_ranges.is_empty() {
            (next1b, false)
        } else {
            eliminate_dead_scratch_stores(&next1b, fn_private_ranges)
        };
        let (next2b, changed2b) = if fn_sb_ranges.is_empty() {
            (next2, false)
        } else {
            eliminate_dead_bb_scratch_stores(&next2, fn_sb_ranges)
        };
        let (next3, changed3) = if fn_private_ranges.is_empty() {
            (next2b, false)
        } else {
            promote_single_block_spills_to_bc(&next2b, fn_private_ranges, fn_sb_ranges, no_bc_fns)
        };
        let (next3b, changed3b) = if fn_sb_ranges.is_empty() {
            (next3, false)
        } else {
            promote_single_block_spills_to_stack(&next3, fn_sb_ranges)
        };
        let (next4, changed4) = merge_ret_tails(&next3b);
        let (next4b, changed4b) = simplify_cond_returns_and_dedup_blocks(&next4);
        let (next5, changed5) = thread_empty_jumps(&next4b);
        lines = next5;
        if !changed1
            && !changed1b
            && !changed2
            && !changed2b
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

pub(super) fn inst_touches_bc(kind: &LineKind) -> bool {
    match kind {
        LineKind::Inst { mnemonic, ops } => {
            if matches!(
                mnemonic.as_str(),
                "call"
                    | "jp"
                    | "jr"
                    | "ret"
                    | "reti"
                    | "retn"
                    | "rst"
                    | "djnz"
                    | "ldi"
                    | "ldd"
                    | "ldir"
                    | "lddr"
                    | "cpi"
                    | "cpd"
                    | "cpir"
                    | "cpdr"
                    | "ini"
                    | "ind"
                    | "inir"
                    | "indr"
                    | "outi"
                    | "outd"
                    | "otir"
                    | "otdr"
                    | "exx"
                    | "halt"
            ) {
                return true;
            }
            ops.iter().any(|op| {
                let t = op.trim().to_ascii_lowercase();
                matches!(t.as_str(), "b" | "c" | "bc" | "(bc)" | "(c)")
            })
        }
        LineKind::Label(_) | LineKind::Directive => true,
        LineKind::EmptyOrComment => false,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

fn is_intra_fn_label(lbl: &str, cur_fn_name: Option<&str>) -> bool {
    if let Some(fn_lbl) = cur_fn_name
        && let Some(suffix) = lbl.strip_prefix(fn_lbl).and_then(|s| s.strip_prefix('_'))
    {
        if suffix.len() > 2
            && suffix.starts_with("bb")
            && suffix[2..].bytes().all(|b| b.is_ascii_digit())
        {
            return true;
        }
        if suffix.rsplit_once('_').is_some_and(|(tag, digits)| {
            !tag.is_empty() && !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
        }) {
            return true;
        }
    }
    lbl.contains("_bb")
        || lbl.contains("_sat_")
        || lbl.contains("_cmp_")
        || lbl.contains("_cmp32_")
        || lbl.contains("_mul8_")
        || lbl.contains("_mul16_")
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
