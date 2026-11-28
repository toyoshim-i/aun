//! Resident `_CODE` section packer for minimizing `.bndry` / `.align` padding (#66).
//!
//! When external `.s` modules contain alignment directives (e.g. `.bndry 256` for
//! page-aligned LUTs), linear concatenation (`startup -> aun_code -> ext_asm`)
//! can waste up to `N - 1` bytes of padding and cause discontinuous `+N` byte
//! jumps whenever code crosses a boundary.
//!
//! This packer partitions relocatable Aun LIR blocks (functions, RODATA tables,
//! runtime helpers, `__aun_startup_init`, `__aun_init_rom`) and unaligned
//! external `.s` modules into indivisible chunks, places aligned external `.s`
//! modules early, and backfills each pre-boundary gap using an exact 0/1
//! knapsack DP (with greedy Best-Fit Decreasing pre-pass for gaps > 1024B).

use crate::ExternalAsm;
use crate::asm;
use crate::lir::{self, LirInst, LirItem};
use std::collections::HashMap;
use std::fmt::Write;

#[derive(Debug, Clone)]
struct RelocBlock {
    asm: String,
    size: usize,
    is_external: bool,
}

#[derive(Debug, Clone)]
struct AlignedExt<'a> {
    ext_idx: usize,
    ext: &'a ExternalAsm,
    align: usize,
    prefix_bytes: usize,
}

/// Returns true if `item` terminates fallthrough control flow (or is pure data).
fn is_split_barrier(item: &LirItem) -> bool {
    match item {
        LirItem::DataBytes(_) | LirItem::DataWords(_) => true,
        LirItem::Inst(inst) => matches!(
            inst,
            LirInst::Ret(None)
                | LirInst::Reti
                | LirInst::Retn
                | LirInst::Jp { cond: None, .. }
                | LirInst::Jr { cond: None, .. }
                | LirInst::JpInd(_)
                | LirInst::Halt
        ),
        _ => false,
    }
}

/// Returns true if `lbl` is eligible to start a new independently relocatable chunk.
fn is_splittable_top_label(lbl: &str) -> bool {
    if lir::is_bb_label(lbl) || lir::is_tail_label(lbl) {
        return false;
    }
    // Keep contiguous `*_disp_base` + all `*_disp_*` 3-byte JP stubs together!
    if lbl.contains("_disp_") && !lbl.ends_with("_disp_base") {
        return false;
    }
    true
}

/// Partitions `body_items` into indivisible `(is_measurable, byte_len, items)` chunks.
///
/// Guarantees:
/// - Never splits across fallthrough (only splits after unconditional terminator or data).
/// - Never splits alias label pairs (`_foo:` + `foo:`).
/// - Never splits `*_disp_base` from its contiguous `*_disp_*` slots.
/// - Merges any chunks connected by a relative branch (`jr` / `djnz`) target.
/// - Marks any chunk containing `LirInst::Raw(_)` as non-measurable so it is never
///   used for gap-filling where exact byte length is required.
fn split_lir_into_chunks(body_items: &[LirItem]) -> Vec<(bool, usize, Vec<LirItem>)> {
    if body_items.is_empty() {
        return Vec::new();
    }

    // 1. Initial split at candidate top-level labels after a barrier.
    let mut raw_chunks: Vec<Vec<LirItem>> = Vec::new();
    let mut cur: Vec<LirItem> = Vec::new();
    let mut cur_bytes = 0usize;
    let mut last_meaningful_was_barrier = false;

    for item in body_items {
        if let LirItem::Label(lbl) = item
            && cur_bytes > 0
            && last_meaningful_was_barrier
            && is_splittable_top_label(lbl)
        {
            raw_chunks.push(std::mem::take(&mut cur));
            cur_bytes = 0;
            last_meaningful_was_barrier = false;
        }

        let blen = item.byte_len();
        if blen > 0 {
            cur_bytes += blen;
            last_meaningful_was_barrier = is_split_barrier(item);
        }
        cur.push(item.clone());
    }
    if !cur.is_empty() {
        raw_chunks.push(cur);
    }

    // 2. Build label -> chunk_idx map and merge any chunks coupled by `jr`/`djnz` targets.
    let mut lbl_to_chunk: HashMap<String, usize> = HashMap::new();
    for (ci, chunk) in raw_chunks.iter().enumerate() {
        for item in chunk {
            if let LirItem::Label(lbl) = item {
                lbl_to_chunk.insert(lbl.clone(), ci);
            }
        }
    }

    let n = raw_chunks.len();
    let mut merge_end: Vec<usize> = (0..n).collect();
    for (ci, chunk) in raw_chunks.iter().enumerate() {
        for item in chunk {
            let target = match item {
                LirItem::Inst(LirInst::Jr { target, .. }) | LirItem::Inst(LirInst::Djnz(target)) => {
                    Some(target.as_str())
                }
                _ => None,
            };
            if let Some(tgt) = target
                && let Some(&ti) = lbl_to_chunk.get(tgt)
                && ti != ci
            {
                let lo = ci.min(ti);
                let hi = ci.max(ti);
                if merge_end[lo] < hi {
                    merge_end[lo] = hi;
                }
            }
        }
    }

    // Flatten overlapping [lo..=hi] merge intervals into final chunks.
    let mut final_chunks: Vec<(bool, usize, Vec<LirItem>)> = Vec::new();
    let mut i = 0usize;
    while i < n {
        let mut end = merge_end[i];
        let mut j = i + 1;
        while j <= end {
            if merge_end[j] > end {
                end = merge_end[j];
            }
            j += 1;
        }
        let mut merged = Vec::new();
        for c in raw_chunks.iter().take(end + 1).skip(i) {
            merged.extend_from_slice(c);
        }
        let has_raw = merged
            .iter()
            .any(|it| matches!(it, LirItem::Inst(LirInst::Raw(_))));
        let total_len: usize = merged.iter().map(LirItem::byte_len).sum();
        final_chunks.push((!has_raw && total_len > 0, total_len, merged));
        i = end + 1;
    }

    final_chunks
}

/// Selects a deterministic subset of unused relocatable blocks whose total byte
/// length is `<= gap`, maximizing total filled bytes (minimizing leftover padding).
fn select_gap_fillers(pool: &[RelocBlock], used: &[bool], mut gap: usize) -> Vec<usize> {
    if gap == 0 {
        return Vec::new();
    }

    let mut chosen: Vec<usize> = Vec::new();
    let mut temp_used = used.to_vec();

    // For large gaps (> 1024B), greedily pick largest fitting blocks until gap <= 1024.
    const DP_LIMIT: usize = 1024;
    while gap > DP_LIMIT {
        let best = pool
            .iter()
            .enumerate()
            .filter(|&(i, b)| !temp_used[i] && b.size > 0 && b.size <= gap)
            .max_by(|(i1, b1), (i2, b2)| b1.size.cmp(&b2.size).then_with(|| i2.cmp(i1)))
            .map(|(i, _)| i);
        let Some(idx) = best else {
            break;
        };
        temp_used[idx] = true;
        gap -= pool[idx].size;
        chosen.push(idx);
    }

    if gap == 0 || gap > DP_LIMIT {
        return chosen;
    }

    // Exact 0/1 Knapsack DP for `gap <= 1024`:
    // `parent[w] = Some((prev_w, block_idx))` if exact sum `w` is reachable.
    let mut reachable = vec![false; gap + 1];
    let mut parent: Vec<Option<(usize, usize)>> = vec![None; gap + 1];
    reachable[0] = true;
    let mut max_w = 0usize;

    for (i, blk) in pool.iter().enumerate() {
        if temp_used[i] || blk.size == 0 || blk.size > gap {
            continue;
        }
        let sz = blk.size;
        let upper = max_w.min(gap - sz);
        for w in (0..=upper).rev() {
            if reachable[w] && !reachable[w + sz] {
                reachable[w + sz] = true;
                parent[w + sz] = Some((w, i));
                if w + sz > max_w {
                    max_w = w + sz;
                }
            }
        }
        if max_w == gap {
            break;
        }
    }

    let mut dp_chosen = Vec::new();
    let mut cur_w = max_w;
    while cur_w > 0 {
        if let Some((prev_w, idx)) = parent[cur_w] {
            dp_chosen.push(idx);
            cur_w = prev_w;
        } else {
            break;
        }
    }
    dp_chosen.reverse();
    chosen.extend(dp_chosen);
    chosen
}

fn format_ext_block(ext_idx: usize, ext: &ExternalAsm) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "\n        .area _CODE\n; @aun:ext_begin {ext_idx}");
    s.push_str(&ext.source);
    if !s.ends_with('\n') {
        s.push('\n');
    }
    s.push_str("; @aun:ext_end\n");
    s
}

/// Packs relocatable Resident code/data blocks around `.bndry`/`.align` external
/// `.s` modules to minimize alignment padding.
///
/// Returns `None` when no `resident_others` module contains a `_CODE` `.bndry`/`.align`
/// (or if startup PC cannot be statically measured), allowing the caller to fall back
/// to the standard linear layout with zero overhead.
pub fn pack_resident_sections(
    code_origin: u16,
    startup_asm: &str,
    body_items: &[LirItem],
    init_bytes: &[u8],
    resident_others: &[(usize, &ExternalAsm)],
) -> Option<String> {
    if resident_others.is_empty() {
        return None;
    }

    // Check whether any resident external module has a `_CODE` `.align`/`.bndry` > 1.
    let mut aligned_exts: Vec<AlignedExt<'_>> = Vec::new();
    let mut reloc_pool: Vec<RelocBlock> = Vec::new();
    let mut unmeasured_tail: Vec<String> = Vec::new();

    for &(ei, ext) in resident_others {
        if let Some(prof) = asm::measure_code_profile(&ext.source, code_origin)
            && !prof.has_org
        {
            if let Some(align) = prof.first_align {
                aligned_exts.push(AlignedExt {
                    ext_idx: ei,
                    ext,
                    align,
                    prefix_bytes: prof.prefix_bytes,
                });
            } else {
                reloc_pool.push(RelocBlock {
                    asm: format_ext_block(ei, ext),
                    size: prof.prefix_bytes,
                    is_external: true,
                });
            }
        } else {
            unmeasured_tail.push(format_ext_block(ei, ext));
        }
    }

    if aligned_exts.is_empty() {
        return None;
    }

    let mut cur_pc = asm::measure_code_advance(startup_asm, code_origin)? as usize;

    // Partition Aun LIR body into indivisible relocatable chunks.
    let chunks = split_lir_into_chunks(body_items);
    for (is_measurable, sz, items) in chunks {
        let formatted = lir::format_zilog_lir(&items);
        if is_measurable {
            reloc_pool.push(RelocBlock {
                asm: formatted,
                size: sz,
                is_external: false,
            });
        } else {
            unmeasured_tail.push(format!("\n        .area _CODE\n{formatted}"));
        }
    }

    // Add `__aun_startup_init` and `__aun_init_rom` to the relocatable pool.
    {
        let mut init_fn = String::from("\n__aun_startup_init:\n");
        let init_fn_size = if init_bytes.is_empty() {
            init_fn.push_str("    ret\n");
            1
        } else {
            init_fn.push_str("    ld hl, s__INITIALIZER\n");
            init_fn.push_str("    ld de, s__INITIALIZED\n");
            init_fn.push_str("    ld bc, l__INITIALIZER\n");
            init_fn.push_str("    ldir\n");
            init_fn.push_str("    ret\n");
            12
        };
        reloc_pool.push(RelocBlock {
            asm: init_fn,
            size: init_fn_size,
            is_external: false,
        });
    }
    if !init_bytes.is_empty() {
        let mut rom = String::from("\n__aun_init_rom:\n");
        for chunk in init_bytes.chunks(16) {
            let bytes: Vec<String> = chunk.iter().map(|b| format!("0x{b:02X}")).collect();
            let _ = writeln!(rom, "    .db {}", bytes.join(", "));
        }
        reloc_pool.push(RelocBlock {
            asm: rom,
            size: init_bytes.len(),
            is_external: false,
        });
    }

    // Place largest-alignment blocks first (stable sort preserves original order on ties).
    aligned_exts.sort_by(|a, b| b.align.cmp(&a.align));

    let mut used = vec![false; reloc_pool.len()];
    let mut out = String::new();
    let mut need_code_area = true;

    let emit_reloc = |out: &mut String, blk: &RelocBlock, need_code_area: &mut bool| {
        if blk.is_external {
            out.push_str(&blk.asm);
            *need_code_area = true;
        } else {
            if *need_code_area {
                out.push_str("\n        .area _CODE\n");
                *need_code_area = false;
            }
            out.push_str(&blk.asm);
        }
    };

    for aext in &aligned_exts {
        let pre_align_pc = cur_pc + aext.prefix_bytes;
        let rem = pre_align_pc % aext.align;
        let gap = if rem == 0 { 0 } else { aext.align - rem };

        let fillers = select_gap_fillers(&reloc_pool, &used, gap);
        for idx in fillers {
            used[idx] = true;
            emit_reloc(&mut out, &reloc_pool[idx], &mut need_code_area);
            cur_pc += reloc_pool[idx].size;
        }

        out.push_str(&format_ext_block(aext.ext_idx, aext.ext));
        need_code_area = true;
        cur_pc = asm::measure_code_advance(&aext.ext.source, cur_pc as u16)? as usize;
    }

    // Emit remaining relocatable blocks in deterministic order.
    for (i, blk) in reloc_pool.iter().enumerate() {
        if !used[i] {
            emit_reloc(&mut out, blk, &mut need_code_area);
        }
    }

    // Emit any non-measurable / `.org`-bearing blocks last.
    for blk in unmeasured_tail {
        out.push_str(&blk);
    }

    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knapsack_dp_finds_exact_zero_pad_subset() {
        // Greedy would pick 60 + 30 = 90 (10B waste), whereas DP finds 60 + 40 = 100 (0B waste).
        let mk = |size| RelocBlock {
            asm: String::new(),
            size,
            is_external: false,
        };
        let pool = vec![mk(60), mk(45), mk(40), mk(30)];
        let used = vec![false; 4];
        let chosen = select_gap_fillers(&pool, &used, 100);
        let filled: usize = chosen.iter().map(|&i| pool[i].size).sum();
        assert_eq!(filled, 100);
    }

    #[test]
    fn chunk_splitter_preserves_fallthrough_jr_and_disp_base() {
        let src = "\
__aun_mul_a_10:
    add a, a
__aun_mul_a_5:
    add a, a
    ret

fn_a:
    jr fn_a_mid
    nop
fn_a_mid:
    ret

Game_disp_base:
Game_disp_a:
    jp h1
Game_disp_b:
    jp h2

fn_b:
    xor a
    ret
";
        let items = lir::parse_zilog_to_lir(src);
        let chunks = split_lir_into_chunks(&items);
        // Should be exactly 4 indivisible chunks:
        // 1. __aun_mul_a_10 + __aun_mul_a_5 (fallthrough)
        // 2. fn_a + fn_a_mid (connected by jr)
        // 3. Game_disp_base + Game_disp_a + Game_disp_b (contiguous SMC table)
        // 4. fn_b
        assert_eq!(chunks.len(), 4, "chunks: {chunks:#?}");
        assert!(chunks.iter().all(|(m, _, _)| *m));
        assert_eq!(chunks[2].1, 6); // 2 * 3B JP
    }
}
