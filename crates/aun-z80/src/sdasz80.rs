//! Relocatable SDASZ80 `.s` emitter for linking Aun modules into existing SDCC / `sdldz80` builds.

use crate::analysis::z80_sym;
use crate::jumptable::compress_jump_tables;
use crate::lir::{LirInst, LirItem, SymExpr, parse_zilog_to_lir, relax_jp_to_jr};
use crate::outliner::outline_lir;
use crate::uchiguri::RamLayout;
use aun_ir::IrProgram;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

/// Translates optimized Aun Zilog assembly into relocatable SDASZ80 `.s` syntax.
pub fn emit_sdasz80(
    module_name: &str,
    zilog_asm: &str,
    program: &IrProgram,
    layout: &RamLayout,
) -> String {
    emit_sdasz80_with_extra_stack(module_name, zilog_asm, program, layout).0
}

/// Same as `emit_sdasz80`, also returning `(extra_stack_bytes, trimmed_ram_bytes)`.
pub fn emit_sdasz80_with_metrics(
    module_name: &str,
    zilog_asm: &str,
    program: &IrProgram,
    layout: &RamLayout,
) -> (String, u16, u16) {
    emit_sdasz80_with_options(module_name, zilog_asm, program, layout, None)
}

/// Same as `emit_sdasz80_with_metrics`, honoring an explicit `symbol_prefix` override.
pub fn emit_sdasz80_with_options(
    module_name: &str,
    zilog_asm: &str,
    program: &IrProgram,
    layout: &RamLayout,
    symbol_prefix: Option<&str>,
) -> (String, u16, u16) {
    let mut items = parse_zilog_to_lir(zilog_asm);
    compress_jump_tables(program, &mut items);
    outline_lir(&mut items);
    let extra_stack = crate::outliner::max_outlined_stack_bytes(&items);
    relax_jp_to_jr(&mut items);
    let (asm, used_ram_bytes) =
        emit_sdasz80_lir_with_size(module_name, &items, program, layout, symbol_prefix);
    (asm, extra_stack, used_ram_bytes)
}

/// Same as `emit_sdasz80`, also returning any additional stack bytes used by outlined helpers.
pub fn emit_sdasz80_with_extra_stack(
    module_name: &str,
    zilog_asm: &str,
    program: &IrProgram,
    layout: &RamLayout,
) -> (String, u16) {
    let (asm, extra_stack, _) = emit_sdasz80_with_metrics(module_name, zilog_asm, program, layout);
    (asm, extra_stack)
}

/// Emits relocatable SDASZ80 `.s` assembly directly from relaxed typed `LirItem`s:
/// - Places live Uchiguri RAM (globals, locals, scratch) in `.area _DATA` under `__aun_ram`
///   (trimming unused trailing bytes eliminated by register promotion/peephole)
///   and rewrites fixed RAM addresses inside `[ram_base, ram_end)` to `(__aun_ram + off)`.
/// - Emits `.area _CODE`, `.globl` for externs and `pub fn`s (with C-compatible `_name::`
///   entry aliases, plus a 1-instruction `ex de, hl` trampoline for 16-bit `sdcccall(1)` returns).
/// - Renders operands cleanly via `LirInst::format_sdasz80`.
pub fn emit_sdasz80_lir(
    module_name: &str,
    items: &[LirItem],
    program: &IrProgram,
    layout: &RamLayout,
) -> String {
    emit_sdasz80_lir_with_size(module_name, items, program, layout, None).0
}

fn collect_pub_trampoline_stack_slots(
    program: &IrProgram,
    layout: &RamLayout,
    fid: u32,
    param_count: usize,
    ram_base: u16,
    ram_end: u16,
) -> Option<Vec<(u16, u16)>> {
    let func = &program.functions[fid as usize];
    let stack_start = if param_count >= 2 {
        let p0_sz = program.type_size(&func.locals[0].ty);
        let p1_sz = program.type_size(&func.locals[1].ty);
        if matches!(p0_sz, 1 | 2)
            && matches!(p1_sz, 1 | 2)
            && !(p0_sz == 2 && p1_sz == 1)
        {
            2
        } else {
            1
        }
    } else {
        1
    };
    if param_count <= stack_start {
        return None;
    }
    let mut stack_slots: Vec<(u16, u16)> = Vec::new();
    for p_idx in stack_start..param_count {
        if let Some(&p_addr) = layout.slot_addrs.get(&(fid, p_idx as u32))
            && p_addr >= ram_base
            && p_addr < ram_end
        {
            let sz = program.type_size(&func.locals[p_idx].ty);
            if matches!(sz, 1 | 2 | 4) {
                stack_slots.push((p_addr - ram_base, sz));
            } else {
                return None;
            }
        } else {
            return None;
        }
    }
    if stack_slots.is_empty() {
        None
    } else {
        Some(stack_slots)
    }
}

fn compute_used_ram_bytes(
    items: &[LirItem],
    program: &IrProgram,
    layout: &RamLayout,
    sorted_pubs: &[(&String, &(u32, u16, usize, String))],
) -> u16 {
    let data_base = layout.init_ram_end;
    let ram_end = layout.ram_end;
    let ram_bytes = ram_end.saturating_sub(data_base);
    if ram_bytes == 0 {
        return 0;
    }

    let mut used = 0u16;
    let record_ptr_addr = |addr: u16, used: &mut u16| {
        if addr < data_base || addr >= ram_end {
            return;
        }
        for g in &layout.global_entries {
            if addr >= g.addr && addr < g.addr.saturating_add(g.size) {
                *used = (*used).max((g.addr - data_base).saturating_add(g.size));
                return;
            }
        }
        for f in &layout.fn_frames {
            if addr >= f.locals_base && addr < f.locals_base.saturating_add(f.locals_size) {
                *used = (*used).max((f.locals_base - data_base).saturating_add(f.locals_size));
                return;
            }
        }
        *used = ram_bytes;
    };

    for &(_name, &(fid, _ret_sz, param_count, _)) in sorted_pubs {
        if let Some(slots) =
            collect_pub_trampoline_stack_slots(program, layout, fid, param_count, data_base, ram_end)
        {
            for (off, sz) in slots {
                used = used.max(off.saturating_add(sz));
            }
        }
    }

    let aun_ram_offset = |expr: &SymExpr| -> Option<u16> {
        match expr {
            SymExpr::Sym(s) if s == "__aun_ram" => Some(0),
            SymExpr::SymOffset(s, off) if s == "__aun_ram" && *off >= 0 => {
                u16::try_from(*off).ok()
            }
            _ => None,
        }
    };

    for item in items {
        match item {
            LirItem::Inst(inst) => match inst {
                LirInst::LdMem8 { addr, .. } | LirInst::StMem8 { addr, .. } => {
                    if let Some(off) = aun_ram_offset(addr) {
                        used = used.max(off.saturating_add(1));
                    }
                }
                LirInst::Ld16Mem { addr, .. } | LirInst::St16Mem { addr, .. } => {
                    if let Some(off) = aun_ram_offset(addr) {
                        used = used.max(off.saturating_add(2));
                    }
                }
                LirInst::Ld16Imm { val, .. } => {
                    if let Some(off) = aun_ram_offset(val) {
                        record_ptr_addr(data_base.saturating_add(off), &mut used);
                    }
                }
                _ => {}
            },
            LirItem::DataWords(words) => {
                for w in words {
                    if let Some(off) = aun_ram_offset(w) {
                        record_ptr_addr(data_base.saturating_add(off), &mut used);
                    }
                }
            }
            _ => {}
        }
    }

    used.min(ram_bytes)
}

fn rewrite_sym_expr_for_init(expr: &mut SymExpr, init_len: u16) {
    if init_len == 0 {
        return;
    }
    let off = match expr {
        SymExpr::Sym(s) if s == "__aun_ram" => Some(0u16),
        SymExpr::SymOffset(s, o) if s == "__aun_ram" && *o >= 0 => u16::try_from(*o).ok(),
        _ => None,
    };
    if let Some(off) = off {
        if off < init_len {
            *expr = if off == 0 {
                SymExpr::Sym("__aun_init".to_string())
            } else {
                SymExpr::SymOffset("__aun_init".to_string(), off as i16)
            };
        } else {
            let new_off = off - init_len;
            *expr = if new_off == 0 {
                SymExpr::Sym("__aun_ram".to_string())
            } else {
                SymExpr::SymOffset("__aun_ram".to_string(), new_off as i16)
            };
        }
    }
}

fn rewrite_items_for_init(items: &[LirItem], init_len: u16) -> Vec<LirItem> {
    if init_len == 0 {
        return items.to_vec();
    }
    let mut out = items.to_vec();
    for item in &mut out {
        match item {
            LirItem::Inst(inst) => match inst {
                LirInst::LdMem8 { addr, .. }
                | LirInst::StMem8 { addr, .. }
                | LirInst::Ld16Mem { addr, .. }
                | LirInst::St16Mem { addr, .. } => rewrite_sym_expr_for_init(addr, init_len),
                LirInst::Ld16Imm { val, .. } => rewrite_sym_expr_for_init(val, init_len),
                _ => {}
            },
            LirItem::DataWords(words) => {
                for w in words {
                    rewrite_sym_expr_for_init(w, init_len);
                }
            }
            _ => {}
        }
    }
    out
}

fn emit_sdasz80_lir_with_size(
    module_name: &str,
    items: &[LirItem],
    program: &IrProgram,
    layout: &RamLayout,
    symbol_prefix: Option<&str>,
) -> (String, u16) {
    let mut out = String::new();
    let ram_base = layout.ram_base;
    let init_len = layout.init_ram_end.saturating_sub(ram_base);
    let data_base = layout.init_ram_end;
    let ram_end = layout.ram_end;
    let rewritten_items = rewrite_items_for_init(items, init_len);
    let items = &rewritten_items;

    let pfx = symbol_prefix.unwrap_or("_");
    let mut pub_fns: HashMap<String, (u32, u16, usize, String)> = HashMap::new();
    for f in &program.functions {
        if !f.is_test && (f.is_pub || f.export_name.is_some()) && f.overlay_state.is_none() {
            let sym = z80_sym(&f.name);
            let c_sym = if let Some(ref exp) = f.export_name {
                exp.clone()
            } else if sym.starts_with(pfx) {
                sym.clone()
            } else {
                format!("{pfx}{sym}")
            };
            pub_fns.insert(
                sym,
                (f.id, program.type_size(&f.ret_ty), f.param_count, c_sym),
            );
        }
    }

    writeln!(out, "; Generated by aun-z80 (--format sdasz80)").unwrap();
    writeln!(out, "        .module {module_name}").unwrap();
    writeln!(out, "        .optsdcc -mz80 sdcccall(1)").unwrap();

    // External symbols
    let mut emitted_globl: HashSet<String> = HashSet::new();
    for ext in &program.externs {
        if ext.fixed_addr.is_none() && emitted_globl.insert(ext.name.clone()) {
            writeln!(out, "        .globl  {}", ext.name).unwrap();
        }
    }
    for g in &program.globals {
        if g.is_extern && emitted_globl.insert(g.name.clone()) {
            writeln!(out, "        .globl  {}", g.name).unwrap();
        }
    }
    // Public C-compatible exports
    let mut sorted_pubs: Vec<(&String, &(u32, u16, usize, String))> = pub_fns.iter().collect();
    sorted_pubs.sort_by_key(|(k, _)| k.as_str());
    for (_, (_, _, _, c_sym)) in &sorted_pubs {
        if emitted_globl.insert(c_sym.clone()) {
            writeln!(out, "        .globl  {c_sym}").unwrap();
        }
    }

    // Non-zero initialized `static mut` globals in `_INITIALIZED` / `_INITIALIZER`
    if init_len > 0 {
        writeln!(out, "\n        .area _INITIALIZED").unwrap();
        writeln!(out, "__aun_init:").unwrap();
        writeln!(out, "        .ds {init_len}").unwrap();
        writeln!(out, "\n        .area _INITIALIZER").unwrap();
        for chunk in layout.init_bytes.chunks(16) {
            let strs: Vec<String> = chunk.iter().map(|b| format!("0x{b:02X}")).collect();
            writeln!(out, "        .db {}", strs.join(", ")).unwrap();
        }
    }

    // Relocatable Uchiguri RAM block in `_DATA` (trimmed to live referenced tail)
    let used_data_bytes = compute_used_ram_bytes(items, program, layout, &sorted_pubs);
    let used_ram_bytes = init_len.saturating_add(used_data_bytes);
    if used_data_bytes > 0 {
        writeln!(out, "\n        .area _DATA").unwrap();
        writeln!(out, "__aun_ram:").unwrap();
        writeln!(out, "        .ds {used_data_bytes}").unwrap();
    }

    writeln!(out, "\n        .area _CODE").unwrap();

    // Emit C-compatible `_name::` trampolines for:
    // - `pub fn`s with `param_count > 2` (unpacks SDCC `--sdcccall 1` stack args into Uchiguri slots)
    // - `pub fn`s returning 2-byte values (SDCC `--sdcccall 1` expects 16-bit return in `DE`)
    let mut trampoline_fns: HashSet<String> = HashSet::new();
    for &(name, &(fid, ret_sz, param_count, ref c_sym)) in &sorted_pubs {
        if c_sym == name {
            continue;
        }
        if let Some(stack_slots) =
            collect_pub_trampoline_stack_slots(program, layout, fid, param_count, data_base, ram_end)
        {
            writeln!(out, "{c_sym}::").unwrap();
            writeln!(out, "        pop     iy").unwrap();
            let mut si = 0usize;
            while si < stack_slots.len() {
                let (off, sz) = stack_slots[si];
                if sz == 4 {
                    writeln!(out, "        pop     bc").unwrap();
                    writeln!(out, "        ld      (__aun_ram + {off}), bc").unwrap();
                    writeln!(out, "        pop     bc").unwrap();
                    writeln!(out, "        ld      (__aun_ram + {}), bc", off + 2).unwrap();
                    si += 1;
                } else if sz == 2 {
                    writeln!(out, "        pop     bc").unwrap();
                    writeln!(out, "        ld      (__aun_ram + {off}), bc").unwrap();
                    si += 1;
                } else if si + 1 < stack_slots.len()
                    && stack_slots[si + 1].1 == 1
                    && stack_slots[si + 1].0 == off + 1
                {
                    // Two adjacent 1-byte stack slots: pop & store together via BC!
                    writeln!(out, "        pop     bc").unwrap();
                    writeln!(out, "        ld      (__aun_ram + {off}), bc").unwrap();
                    si += 2;
                } else {
                    // Single 1-byte stack arg: pop 2 bytes, restore 1 byte on SP, preserve A
                    writeln!(out, "        pop     bc").unwrap();
                    writeln!(out, "        dec     sp").unwrap();
                    writeln!(out, "        push    af").unwrap();
                    writeln!(out, "        ld      a, c").unwrap();
                    writeln!(out, "        ld      (__aun_ram + {off}), a").unwrap();
                    writeln!(out, "        pop     af").unwrap();
                    si += 1;
                }
            }
            writeln!(out, "        push    iy").unwrap();
            if ret_sz == 2 {
                writeln!(out, "        call    {name}").unwrap();
                writeln!(out, "        ex      de, hl").unwrap();
                writeln!(out, "        ret").unwrap();
            } else {
                writeln!(out, "        jp      {name}").unwrap();
            }
            trampoline_fns.insert(name.clone());
        } else if ret_sz == 2 {
            writeln!(out, "{c_sym}::").unwrap();
            writeln!(out, "        call    {name}").unwrap();
            writeln!(out, "        ex      de, hl").unwrap();
            writeln!(out, "        ret").unwrap();
            trampoline_fns.insert(name.clone());
        }
    }

    for item in items {
        match item {
            LirItem::Blank => out.push('\n'),
            LirItem::Comment(c) => writeln!(out, "{c}").unwrap(),
            LirItem::Org(_) | LirItem::Equ(_, _) => {}
            LirItem::DataBytes(bytes) => {
                let strs: Vec<String> = bytes.iter().map(|b| format!("0x{b:02X}")).collect();
                writeln!(out, "        .db {}", strs.join(", ")).unwrap();
            }
            LirItem::DataWords(words) => {
                let strs: Vec<String> = words.iter().map(|w| w.format_zilog()).collect();
                writeln!(out, "        .dw {}", strs.join(", ")).unwrap();
            }
            LirItem::Label(lbl) => {
                if let Some((_, _, _, c_sym)) = pub_fns.get(lbl)
                    && !trampoline_fns.contains(lbl)
                    && c_sym != lbl
                {
                    writeln!(out, "{c_sym}::").unwrap();
                }
                writeln!(out, "{lbl}:").unwrap();
            }
            LirItem::Inst(inst) => {
                writeln!(out, "        {}", inst.format_sdasz80(ram_base, ram_end)).unwrap();
            }
        }
    }

    (out, used_ram_bytes)
}
