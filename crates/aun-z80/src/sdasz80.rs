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
    let mut items = parse_zilog_to_lir(zilog_asm);
    compress_jump_tables(program, &mut items);
    outline_lir(&mut items);
    let extra_stack = crate::outliner::max_outlined_stack_bytes(&items);
    relax_jp_to_jr(&mut items);
    let (asm, used_ram_bytes) = emit_sdasz80_lir_with_size(module_name, &items, program, layout);
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
    emit_sdasz80_lir_with_size(module_name, items, program, layout).0
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
    let stack_start = if param_count >= 2
        && matches!(program.type_size(&func.locals[0].ty), 1 | 2)
        && matches!(program.type_size(&func.locals[1].ty), 1 | 2)
    {
        2
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
    sorted_pubs: &[(&String, &(u32, u16, usize))],
) -> u16 {
    let ram_base = layout.ram_base;
    let ram_end = layout.ram_end;
    let ram_bytes = ram_end.saturating_sub(ram_base);
    if ram_bytes == 0 {
        return 0;
    }

    let mut used = 0u16;
    let record_ptr_addr = |addr: u16, used: &mut u16| {
        if addr < ram_base || addr >= ram_end {
            return;
        }
        for g in &layout.global_entries {
            if addr >= g.addr && addr < g.addr.saturating_add(g.size) {
                *used = (*used).max((g.addr - ram_base).saturating_add(g.size));
                return;
            }
        }
        for f in &layout.fn_frames {
            if addr >= f.locals_base && addr < f.locals_base.saturating_add(f.locals_size) {
                *used = (*used).max((f.locals_base - ram_base).saturating_add(f.locals_size));
                return;
            }
        }
        *used = ram_bytes;
    };

    for &(_name, &(fid, _ret_sz, param_count)) in sorted_pubs {
        if let Some(slots) =
            collect_pub_trampoline_stack_slots(program, layout, fid, param_count, ram_base, ram_end)
        {
            for (off, sz) in slots {
                used = used.max(off.saturating_add(sz));
            }
        }
    }

    for item in items {
        match item {
            LirItem::Inst(inst) => match inst {
                LirInst::LdMem8 {
                    addr: SymExpr::Addr(a),
                    ..
                }
                | LirInst::StMem8 {
                    addr: SymExpr::Addr(a),
                    ..
                } if *a >= ram_base && *a < ram_end => {
                    used = used.max((*a - ram_base).saturating_add(1));
                }
                LirInst::Ld16Mem {
                    addr: SymExpr::Addr(a),
                    ..
                }
                | LirInst::St16Mem {
                    addr: SymExpr::Addr(a),
                    ..
                } if *a >= ram_base && *a < ram_end => {
                    used = used.max((*a - ram_base).saturating_add(2));
                }
                LirInst::Ld16Imm {
                    val: SymExpr::Addr(a),
                    ..
                } => {
                    record_ptr_addr(*a, &mut used);
                }
                _ => {}
            },
            LirItem::DataWords(words) => {
                for w in words {
                    if let SymExpr::Addr(a) = w {
                        record_ptr_addr(*a, &mut used);
                    }
                }
            }
            _ => {}
        }
    }

    used.min(ram_bytes)
}

fn emit_sdasz80_lir_with_size(
    module_name: &str,
    items: &[LirItem],
    program: &IrProgram,
    layout: &RamLayout,
) -> (String, u16) {
    let mut out = String::new();
    let ram_base = layout.ram_base;
    let ram_end = layout.ram_end;

    let mut pub_fns: HashMap<String, (u32, u16, usize)> = HashMap::new();
    for f in &program.functions {
        if !f.is_test && f.is_pub && f.overlay_state.is_none() {
            pub_fns.insert(
                z80_sym(&f.name),
                (f.id, program.type_size(&f.ret_ty), f.param_count),
            );
        }
    }

    writeln!(out, "; Generated by aun-z80 (--format sdasz80)").unwrap();
    writeln!(out, "        .module {module_name}").unwrap();
    writeln!(out, "        .optsdcc -mz80 sdcccall(1)").unwrap();

    // External symbols
    for ext in &program.externs {
        writeln!(out, "        .globl  {}", ext.name).unwrap();
    }
    for g in &program.globals {
        if g.is_extern {
            writeln!(out, "        .globl  {}", g.name).unwrap();
        }
    }
    // Public C-compatible exports
    let mut sorted_pubs: Vec<(&String, &(u32, u16, usize))> = pub_fns.iter().collect();
    sorted_pubs.sort_by_key(|(k, _)| k.as_str());
    for (name, _) in &sorted_pubs {
        let c_sym = if name.starts_with('_') {
            (*name).clone()
        } else {
            format!("_{name}")
        };
        writeln!(out, "        .globl  {c_sym}").unwrap();
    }

    // Relocatable Uchiguri RAM block in `_DATA` (trimmed to live referenced tail)
    let used_ram_bytes = compute_used_ram_bytes(items, program, layout, &sorted_pubs);
    if used_ram_bytes > 0 {
        writeln!(out, "\n        .area _DATA").unwrap();
        writeln!(out, "__aun_ram:").unwrap();
        writeln!(out, "        .ds {used_ram_bytes}").unwrap();
    }

    writeln!(out, "\n        .area _CODE").unwrap();

    // Emit C-compatible `_name::` trampolines for:
    // - `pub fn`s with `param_count > 2` (unpacks SDCC `--sdcccall 1` stack args into Uchiguri slots)
    // - `pub fn`s returning 2-byte values (SDCC `--sdcccall 1` expects 16-bit return in `DE`)
    let mut trampoline_fns: HashSet<String> = HashSet::new();
    for &(name, &(fid, ret_sz, param_count)) in &sorted_pubs {
        let c_sym = if name.starts_with('_') {
            name.clone()
        } else {
            format!("_{name}")
        };
        if c_sym == *name {
            continue;
        }
        if let Some(stack_slots) =
            collect_pub_trampoline_stack_slots(program, layout, fid, param_count, ram_base, ram_end)
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
                if pub_fns.contains_key(lbl) && !trampoline_fns.contains(lbl) {
                    let c_sym = if lbl.starts_with('_') {
                        lbl.to_string()
                    } else {
                        format!("_{lbl}")
                    };
                    if c_sym != *lbl {
                        writeln!(out, "{c_sym}::").unwrap();
                    }
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
