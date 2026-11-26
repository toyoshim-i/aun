use crate::analysis::z80_sym;
use aun_ir::*;
use std::fmt::Write;

pub(super) fn emit_dispatch_stub<F>(
    program: &IrProgram,
    mdef: &MachineDef,
    method: &str,
    use_smc: bool,
    resolve_fn: F,
    asm: &mut String,
) where
    F: Fn(&StateDef) -> Option<u32>,
{
    let vtbl_lbl = format!("{}_vtbl_{method}", mdef.name);
    let disp_lbl = format!("{}_disp_{method}", mdef.name);

    writeln!(asm, "\n{vtbl_lbl}:").unwrap();
    for st in &mdef.states {
        let target = if let Some(ref slot_name) = st.overlay_slot
            && let Some(fid) = resolve_fn(st)
            && let Some(slot_def) = program
                .target
                .as_ref()
                .and_then(|t| t.overlay_slots.iter().find(|s| &s.name == slot_name))
        {
            // Overlay method entry point in the overlay's fixed 3-byte-per-fn jump table
            let idx = program
                .functions
                .iter()
                .filter(|f| {
                    !f.is_test
                        && f.overlay_state
                            .as_ref()
                            .is_some_and(|(sl, sn)| sl == slot_name && sn == &st.name)
                })
                .position(|f| f.id == fid)
                .unwrap_or(0);
            format!("0x{:04X}", slot_def.start.wrapping_add((idx as u16) * 3))
        } else {
            resolve_fn(st)
                .map(|fid| z80_sym(&program.functions[fid as usize].name))
                .unwrap_or_else(|| "__aun_noop_ret".to_string())
        };
        writeln!(asm, "    .dw {target}").unwrap();
    }

    writeln!(asm, "{disp_lbl}:").unwrap();
    if use_smc {
        // Self-modifying direct JP (10T, 3B) — patched in RAM on every transition
        let init_target = resolve_fn(&mdef.states[mdef.initial_state as usize])
            .map(|fid| z80_sym(&program.functions[fid as usize].name))
            .unwrap_or_else(|| "__aun_noop_ret".to_string());
        writeln!(asm, "    jp {init_target}").unwrap();
    } else {
        // Table-driven dispatch preserving HL = self
        writeln!(asm, "    push hl").unwrap();
        if mdef.tag_offset > 0 {
            writeln!(asm, "    ld de, 0x{:04X}", mdef.tag_offset).unwrap();
            writeln!(asm, "    add hl, de").unwrap();
        }
        writeln!(asm, "    ld a, (hl)").unwrap();
        writeln!(asm, "    add a, a").unwrap();
        writeln!(asm, "    ld e, a").unwrap();
        writeln!(asm, "    ld d, 0").unwrap();
        writeln!(asm, "    ld hl, {vtbl_lbl}").unwrap();
        writeln!(asm, "    add hl, de").unwrap();
        writeln!(asm, "    ld e, (hl)").unwrap();
        writeln!(asm, "    inc hl").unwrap();
        writeln!(asm, "    ld d, (hl)").unwrap();
        writeln!(asm, "    pop hl").unwrap();
        writeln!(asm, "    push de").unwrap();
        writeln!(asm, "    ret").unwrap();
    }
}

pub(super) fn emit_bank_switch(
    program: &IrProgram,
    group: &str,
    banks: &[String],
    asm: &mut String,
) {
    let Some(target) = &program.target else {
        return;
    };
    let Some(bg) = target.bank_groups.iter().find(|g| g.name == group) else {
        return;
    };
    let mut sorted_target = banks.to_vec();
    sorted_target.sort();
    if let Some((_, ops)) = bg.switches.iter().find(|(combo, _)| {
        let mut s = combo.clone();
        s.sort();
        s == sorted_target
    }) {
        for op in ops {
            emit_port_write_op(op, asm);
        }
    }
}

pub(super) fn emit_port_write_op(op: &PortWriteOp, asm: &mut String) {
    let expr = op.val_expr.trim();
    if let Some(hex) = expr.strip_prefix("0x").or_else(|| expr.strip_prefix("0X"))
        && let Ok(val) = u8::from_str_radix(hex, 16)
    {
        if val == 0 {
            writeln!(asm, "    xor a").unwrap();
        } else {
            writeln!(asm, "    ld a, 0x{val:02X}").unwrap();
        }
        writeln!(asm, "    out (0x{:02X}), a", op.port).unwrap();
    } else if let Ok(val) = expr.parse::<u8>() {
        if val == 0 {
            writeln!(asm, "    xor a").unwrap();
        } else {
            writeln!(asm, "    ld a, 0x{val:02X}").unwrap();
        }
        writeln!(asm, "    out (0x{:02X}), a", op.port).unwrap();
    } else if let Some((sym, mask)) = expr.split_once('|') {
        writeln!(asm, "    ld a, ({})", sym.trim()).unwrap();
        writeln!(asm, "    or {}", mask.trim()).unwrap();
        writeln!(asm, "    out (0x{:02X}), a", op.port).unwrap();
    } else if let Some((sym, mask)) = expr.split_once('&') {
        writeln!(asm, "    ld a, ({})", sym.trim()).unwrap();
        writeln!(asm, "    and {}", mask.trim()).unwrap();
        writeln!(asm, "    out (0x{:02X}), a", op.port).unwrap();
    } else {
        writeln!(asm, "    ld a, ({expr})").unwrap();
        writeln!(asm, "    out (0x{:02X}), a", op.port).unwrap();
    }
}

/// Multiplies `HL` by a compile-time constant `k` in-place (clobbers `DE`).
pub(super) fn emit_mul_hl_const(k: u16, asm: &mut String) {
    match k {
        0 => writeln!(asm, "    ld hl, 0").unwrap(),
        1 => {}
        n if n.is_power_of_two() => {
            for _ in 0..n.trailing_zeros() {
                writeln!(asm, "    add hl, hl").unwrap();
            }
        }
        n => {
            // Binary shift-and-add: DE = original HL, start from MSB
            writeln!(asm, "    ld d, h").unwrap();
            writeln!(asm, "    ld e, l").unwrap();
            let msb = 15 - n.leading_zeros();
            for bit in (0..msb).rev() {
                writeln!(asm, "    add hl, hl").unwrap();
                if ((n >> bit) & 1) != 0 {
                    writeln!(asm, "    add hl, de").unwrap();
                }
            }
        }
    }
}

pub(super) fn ranges_overlap(a_start: u16, a_len: u16, b_start: u16, b_len: u16) -> bool {
    let a0 = a_start as u32;
    let a1 = a0 + a_len as u32;
    let b0 = b_start as u32;
    let b1 = b0 + b_len as u32;
    a0 < b1 && b0 < a1
}
