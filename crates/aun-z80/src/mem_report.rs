//! Structured memory & overlay budget report and facts-only overflow summaries.

use crate::analysis::z80_sym;
use crate::overlay::{OverlayOwner, OverlayPlan};
use crate::uchiguri::GlobalRamEntry;
use aun_ir::IrProgram;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write;

/// Size and classification of one symbol inside Resident, an Overlay, or RAM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolSizeEntry {
    pub name: String,
    pub kind: &'static str,
    pub bytes: usize,
    pub reason: Option<String>,
}

/// Budget & top-symbol summary for one overlay blob.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OverlayBudgetEntry {
    pub state_name: String,
    pub parent_state: Option<String>,
    pub space_name: String,
    pub origin: u16,
    pub size: usize,
    pub aligned_size: usize,
    pub budget: usize,
    pub limit: u16,
    pub overflow: bool,
    pub top_symbols: Vec<SymbolSizeEntry>,
}

/// Full compiler-level memory layout report (available via `--memory-report`,
/// including when a build exceeds its code/overlay/RAM/stack budget).
#[derive(Debug, Clone, Default)]
pub struct MemoryReport {
    pub code_origin: u16,
    pub resident_bytes: usize,
    pub resident_top: Vec<SymbolSizeEntry>,
    pub lca_promoted_resident: Vec<SymbolSizeEntry>,
    pub overlays: Vec<OverlayBudgetEntry>,
    pub ram_base: u16,
    pub ram_bytes: usize,
    pub ram_top: Vec<SymbolSizeEntry>,
    pub max_stack_bytes: u16,
    pub stack_budget: Option<(String, u16)>,
    pub budget_errors: Vec<String>,
}

fn is_internal_code_label(sym: &str) -> bool {
    if sym.starts_with("s__") || sym.starts_with("l__") || sym == "__aun_ram" {
        return true;
    }
    if crate::lir::is_bb_label(sym) || crate::lir::is_tail_label(sym) {
        return true;
    }
    for suffix in [
        "_lp",
        "_sk",
        "_ok",
        "_done",
        "_end",
        "_nz",
        "_z",
        "_neg",
        "_pos",
        "_pos1",
        "_pos2",
        "_de8",
        "_8x16",
        "_16x16",
        "_8lp",
        "_8sk",
    ] {
        if sym.ends_with(suffix) {
            return true;
        }
    }
    for pat in [
        "_sat_",
        "_sat16_",
        "_cmp_",
        "_cmp32_",
        "_adc0_",
        "_sbc0_",
        "_ex_nx",
        "_sw_nx",
        "_en_nx",
        "_ex_done",
        "_sw_done",
        "_m16x8_",
        "_fshr32_",
        "_shr_",
        "_shl_",
    ] {
        if sym.contains(pat) {
            return true;
        }
    }
    false
}

/// Computes exact per-symbol spans inside `[origin, origin + region_bytes)` from
/// assembled symbols, returning the top `limit` largest symbols sorted descending.
pub fn compute_region_top_symbols(
    origin: u16,
    region_bytes: usize,
    syms: &HashMap<String, u16>,
    fn_names: &HashSet<String>,
    rom_names: &HashSet<String>,
    limit: usize,
) -> Vec<SymbolSizeEntry> {
    if region_bytes == 0 {
        return Vec::new();
    }
    let start = origin as usize;
    let end = start + region_bytes;

    // Deduplicate aliases at the same address, preferring Aun fn/const names over prefixed aliases.
    let mut by_addr: BTreeMap<usize, (String, &'static str)> = BTreeMap::new();
    for (name, &addr_u16) in syms {
        let addr = addr_u16 as usize;
        if addr < start || addr >= end || is_internal_code_label(name) {
            continue;
        }
        let kind: &'static str = if fn_names.contains(name) {
            "fn"
        } else if rom_names.contains(name) {
            "const"
        } else if name.starts_with("__aun_") || name.contains("_disp_") || name.contains("_vtbl_")
            || name.contains("_smc_tbl_")
        {
            "rt"
        } else {
            "asm"
        };
        let prio = |k: &str| match k {
            "fn" | "const" => 0,
            "rt" => 1,
            _ => 2,
        };
        match by_addr.get(&addr) {
            Some((prev_name, prev_kind)) => {
                if (prio(kind), name.len()) < (prio(prev_kind), prev_name.len()) {
                    by_addr.insert(addr, (name.clone(), kind));
                }
            }
            None => {
                by_addr.insert(addr, (name.clone(), kind));
            }
        }
    }

    let ordered: Vec<(usize, String, &'static str)> = by_addr
        .into_iter()
        .map(|(a, (n, k))| (a, n, k))
        .collect();
    let mut entries = Vec::with_capacity(ordered.len());
    for i in 0..ordered.len() {
        let (addr, ref name, kind) = ordered[i];
        let next_addr = ordered.get(i + 1).map_or(end, |x| x.0);
        let bytes = next_addr.saturating_sub(addr);
        if bytes > 0 {
            entries.push(SymbolSizeEntry {
                name: name.clone(),
                kind,
                bytes,
                reason: None,
            });
        }
    }
    entries.sort_by(|a, b| b.bytes.cmp(&a.bytes).then_with(|| a.name.cmp(&b.name)));
    entries.truncate(limit);
    entries
}

/// Computes exact byte size map for all top-level symbols in `[origin, origin + region_bytes)`.
pub fn compute_symbol_size_map(
    origin: u16,
    region_bytes: usize,
    syms: &HashMap<String, u16>,
    fn_names: &HashSet<String>,
    rom_names: &HashSet<String>,
) -> HashMap<String, usize> {
    compute_region_top_symbols(origin, region_bytes, syms, fn_names, rom_names, usize::MAX)
        .into_iter()
        .map(|e| (e.name, e.bytes))
        .collect()
}

/// Lists Resident functions that landed in Resident via multi-state LCA promotion
/// or `pub` pinning while having boundary-state callers.
pub fn build_lca_promoted_resident(
    program: &IrProgram,
    plan: &OverlayPlan,
    res_sizes: &HashMap<String, usize>,
    limit: usize,
) -> Vec<SymbolSizeEntry> {
    if !plan.has_any_boundaries() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (fid, f) in program.functions.iter().enumerate() {
        if f.is_test || f.blocks.is_empty() {
            continue;
        }
        if plan.fn_owner.get(fid) != Some(&OverlayOwner::Resident) {
            continue;
        }
        let Some(reason) = plan.fn_resident_reasons.get(&fid) else {
            continue;
        };
        let sym = z80_sym(&f.name);
        let bytes = res_sizes.get(&sym).copied().unwrap_or(0);
        if bytes == 0 {
            continue;
        }
        out.push(SymbolSizeEntry {
            name: f.name.clone(),
            kind: "fn",
            bytes,
            reason: Some(reason.clone()),
        });
    }
    out.sort_by(|a, b| b.bytes.cmp(&a.bytes).then_with(|| a.name.cmp(&b.name)));
    out.truncate(limit);
    out
}

pub fn build_ram_top(globals: &[GlobalRamEntry], limit: usize) -> Vec<SymbolSizeEntry> {
    let mut entries: Vec<SymbolSizeEntry> = globals
        .iter()
        .filter(|g| g.size > 0)
        .map(|g| SymbolSizeEntry {
            name: g.name.clone(),
            kind: "ram",
            bytes: g.size as usize,
            reason: None,
        })
        .collect();
    entries.sort_by(|a, b| b.bytes.cmp(&a.bytes).then_with(|| a.name.cmp(&b.name)));
    entries.truncate(limit);
    entries
}

fn fmt_top_inline(entries: &[SymbolSizeEntry], max_items: usize) -> String {
    entries
        .iter()
        .take(max_items)
        .map(|e| format!("{} {} B", e.name, e.bytes))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Concise facts-only breakdown attached directly to an `E006` overlay overflow error.
#[allow(clippy::too_many_arguments)]
pub fn format_overlay_overflow_summary(
    space_name: &str,
    space_span: Option<(u16, u16)>,
    resident_bytes: usize,
    resident_top: &[SymbolSizeEntry],
    parent_info: Option<(&str, usize)>,
    blob_state: &str,
    blob_bytes: usize,
    blob_top: &[SymbolSizeEntry],
    lca_promoted: &[SymbolSizeEntry],
) -> String {
    let mut out = String::new();
    if let Some((sp_start, sp_end)) = space_span {
        let total = (sp_end as usize).saturating_sub(sp_start as usize) + 1;
        let _ = writeln!(
            out,
            "  space `{space_name}` (0x{sp_start:04X}..0x{sp_end:04X}, {total} B):"
        );
    }
    let res_top_s = fmt_top_inline(resident_top, 3);
    if res_top_s.is_empty() {
        let _ = writeln!(out, "    Resident:        {resident_bytes:>6} B");
    } else {
        let _ = writeln!(
            out,
            "    Resident:        {resident_bytes:>6} B  (top: {res_top_s})"
        );
    }
    if let Some((pname, pbytes)) = parent_info {
        let _ = writeln!(out, "    Parent({pname}): {pbytes:>6} B");
    }
    let ovl_top_s = fmt_top_inline(blob_top, 3);
    if ovl_top_s.is_empty() {
        let _ = writeln!(out, "    {blob_state}: {blob_bytes:>6} B  [OVER]");
    } else {
        let _ = writeln!(
            out,
            "    {blob_state}: {blob_bytes:>6} B  [OVER] (top: {ovl_top_s})"
        );
    }
    if !lca_promoted.is_empty() {
        let items: Vec<String> = lca_promoted
            .iter()
            .take(3)
            .map(|e| {
                if let Some(ref r) = e.reason {
                    format!("{} {} B ({r})", e.name, e.bytes)
                } else {
                    format!("{} {} B", e.name, e.bytes)
                }
            })
            .collect();
        let _ = writeln!(out, "    Resident-pinned/LCA: {}", items.join("; "));
    }
    out.trim_end().to_string()
}

/// Concise facts-only breakdown attached directly to an `E007` RAM overflow error.
pub fn format_ram_overflow_summary(
    ram_bytes: u16,
    ram_top: &[SymbolSizeEntry],
    max_stack_bytes: u16,
) -> String {
    let mut out = String::new();
    let top_s = fmt_top_inline(ram_top, 5);
    if top_s.is_empty() {
        let _ = writeln!(
            out,
            "  RAM total: {ram_bytes} B, static stack depth: {max_stack_bytes} B"
        );
    } else {
        let _ = writeln!(
            out,
            "  RAM total: {ram_bytes} B (top: {top_s}), static stack depth: {max_stack_bytes} B"
        );
    }
    out.trim_end().to_string()
}

/// Formats the full `--memory-report` output for CLI display.
pub fn format_memory_report(report: &MemoryReport) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "=== Aun Memory Report ===");
    let _ = writeln!(
        out,
        "Resident CODE/RODATA: {} B (origin 0x{:04X})",
        report.resident_bytes, report.code_origin
    );
    if !report.resident_top.is_empty() {
        let _ = writeln!(out, "  Top Resident symbols:");
        for e in &report.resident_top {
            let _ = writeln!(out, "    {:>6} B  [{:<5}] {}", e.bytes, e.kind, e.name);
        }
    }
    if !report.lca_promoted_resident.is_empty() {
        let _ = writeln!(
            out,
            "  Resident-pinned / LCA-promoted functions (candidates to demote into overlays):"
        );
        for e in &report.lca_promoted_resident {
            let reason = e.reason.as_deref().unwrap_or("");
            let _ = writeln!(out, "    {:>6} B  {}  -- {}", e.bytes, e.name, reason);
        }
    }
    if !report.overlays.is_empty() {
        let _ = writeln!(out, "Overlays ({}):", report.overlays.len());
        for ovl in &report.overlays {
            let status = if ovl.overflow {
                format!(
                    "OVER by {} B",
                    ovl.aligned_size.saturating_sub(ovl.budget)
                )
            } else {
                format!("{} B free", ovl.budget.saturating_sub(ovl.aligned_size))
            };
            let parent_note = ovl
                .parent_state
                .as_deref()
                .map(|p| format!(" (parent: {p})"))
                .unwrap_or_default();
            let _ = writeln!(
                out,
                "  {}{} in `{}` @ 0x{:04X}..0x{:04X}: {} B / {} B [{}]",
                ovl.state_name,
                parent_note,
                ovl.space_name,
                ovl.origin,
                ovl.limit,
                ovl.size,
                ovl.budget,
                status
            );
            for e in ovl.top_symbols.iter().take(5) {
                let _ = writeln!(out, "    {:>6} B  [{:<5}] {}", e.bytes, e.kind, e.name);
            }
        }
    }
    let stack_note = match &report.stack_budget {
        Some((name, b)) => format!("stack `{name}`: {} / {b} B", report.max_stack_bytes),
        None => format!("static stack: {} B", report.max_stack_bytes),
    };
    let _ = writeln!(
        out,
        "RAM: {} B (base 0x{:04X}), {}",
        report.ram_bytes, report.ram_base, stack_note
    );
    if !report.ram_top.is_empty() {
        let _ = writeln!(out, "  Top RAM globals/buffers:");
        for e in &report.ram_top {
            let _ = writeln!(out, "    {:>6} B  {}", e.bytes, e.name);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CompileOptions, compile_z80_with_options};

    #[test]
    fn overlay_overflow_emits_facts_only_summary_and_memory_report_completes() {
        let src = r#"
            target pc88sr : z80 {
                space MainRam   @ 0x0100..0x04FF;
                stack MainStack @ 0x04C0..0x04FF;

                on_swap(ops: &const [LoadOp]) => dummy_swap(ops);
            }

            fn dummy_swap(_ops: &const [LoadOp]) {}

            // Called from both Title (Resident) and Stage1 (overlay) -> LCA-promoted to Resident!
            fn shared_helper(x: u16) -> u16 {
                let mut acc: u16 = x;
                acc = acc +% (x << 1) +% (x << 2) +% 7u16;
                acc
            }

            const STAGE1_BIG: [u8; 900] = [0x5Au8; 900];

            #[smc]
            machine Game {
                state Title { c: u8 };
                #[swap_boundary]
                state Stage1 { hp: u8 };

                transitions {
                    Title -> Stage1;
                    Stage1 -> Title;
                }

                on Title {
                    fn tick(&mut self) -> u16 {
                        shared_helper(self.c as u16)
                    }
                }

                on Stage1 {
                    fn tick(&mut self) -> u16 {
                        shared_helper(STAGE1_BIG[self.hp as u16] as u16)
                    }
                }
            }

            pub fn main() -> u16 {
                let mut g: Game = Game::new();
                let a: u16 = g.tick();
                g.goto(Stage1 { hp: 1u8 });
                a +% g.tick()
            }
        "#;

        let prog = aun_front::compile_str(0, src).expect("front");

        // 1. Default build fails with E006 + facts-only breakdown (space, Resident, Stage1, LCA).
        let mut opts = CompileOptions::release();
        opts.ram_base = 0xC000;
        let err = compile_z80_with_options(&prog, &[], opts.clone())
            .expect_err("expected Stage1 overlay budget overflow");
        assert!(err.contains("overlay `Stage1` exceeds `MainRam` budget"), "err:\n{err}");
        assert!(err.contains("Resident:"), "expected Resident line in:\n{err}");
        assert!(err.contains("STAGE1_BIG"), "expected top overlay symbol STAGE1_BIG in:\n{err}");
        assert!(
            err.contains("shared_helper") && err.contains("LCA of"),
            "expected LCA promotion reason for shared_helper in:\n{err}"
        );

        // 2. `--memory-report` mode (`allow_budget_overflow = true`) completes and formats full report!
        opts.allow_budget_overflow = true;
        let out = compile_z80_with_options(&prog, &[], opts)
            .expect("allow_budget_overflow should return LinkOutput with MemoryReport");
        assert!(
            !out.memory_report.budget_errors.is_empty(),
            "expected budget_errors recorded"
        );
        let txt = format_memory_report(&out.memory_report);
        assert!(txt.contains("=== Aun Memory Report ==="), "report:\n{txt}");
        assert!(txt.contains("OVER by"), "expected OVER status in:\n{txt}");
        assert!(
            txt.contains("shared_helper") && txt.contains("LCA of"),
            "expected LCA-promoted section in:\n{txt}"
        );
        assert!(txt.contains("STAGE1_BIG"), "expected STAGE1_BIG in:\n{txt}");
    }
}
