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

/// Emits `__aun_mul32`: 32x32->32 wrapping multiply.
/// Entry: `HLDE` = lhs (`HL` hi16, `DE` lo16), `[sp+2..3]` = rhs_lo, `[sp+4..5]` = rhs_hi.
/// Return: `HLDE` = product (caller cleans stacked rhs). Clobbers `A, B, C, F, IY`. Preserves `IX`.
pub(super) fn emit_mul32_helper(asm: &mut String) {
    writeln!(asm, "\n__aun_mul32:").unwrap();
    writeln!(asm, "    push hl").unwrap();
    writeln!(asm, "    push de").unwrap();
    writeln!(asm, "    ld iy, 0").unwrap();
    writeln!(asm, "    add iy, sp").unwrap();
    writeln!(asm, "    ld de, 0").unwrap();
    writeln!(asm, "    ld h, d").unwrap();
    writeln!(asm, "    ld l, e").unwrap();
    writeln!(asm, "    ld b, 32").unwrap();
    writeln!(asm, "__aun_mul32_lp:").unwrap();
    writeln!(asm, "    sla e").unwrap();
    writeln!(asm, "    rl d").unwrap();
    writeln!(asm, "    rl l").unwrap();
    writeln!(asm, "    rl h").unwrap();
    writeln!(asm, "    sla (iy + 6)").unwrap();
    writeln!(asm, "    rl (iy + 7)").unwrap();
    writeln!(asm, "    rl (iy + 8)").unwrap();
    writeln!(asm, "    rl (iy + 9)").unwrap();
    writeln!(asm, "    jr nc, __aun_mul32_sk").unwrap();
    writeln!(asm, "    push hl").unwrap();
    writeln!(asm, "    ld l, (iy + 0)").unwrap();
    writeln!(asm, "    ld h, (iy + 1)").unwrap();
    writeln!(asm, "    add hl, de").unwrap();
    writeln!(asm, "    ex de, hl").unwrap();
    writeln!(asm, "    pop hl").unwrap();
    writeln!(asm, "    ld a, l").unwrap();
    writeln!(asm, "    adc a, (iy + 2)").unwrap();
    writeln!(asm, "    ld l, a").unwrap();
    writeln!(asm, "    ld a, h").unwrap();
    writeln!(asm, "    adc a, (iy + 3)").unwrap();
    writeln!(asm, "    ld h, a").unwrap();
    writeln!(asm, "__aun_mul32_sk:").unwrap();
    writeln!(asm, "    djnz __aun_mul32_lp").unwrap();
    writeln!(asm, "    pop bc").unwrap();
    writeln!(asm, "    pop bc").unwrap();
    writeln!(asm, "    ret").unwrap();
}

/// Emits `__aun_udiv32` and `__aun_urem32`: unsigned 32-bit div/rem.
/// Entry: `HLDE` = dividend, `[sp+2..5]` = divisor (le).
/// Return: `HLDE` = quotient (`__aun_udiv32`) or remainder (`__aun_urem32`).
/// Clobbers `A, B, C, F, IY`. Preserves `IX`.
pub(super) fn emit_udiv32_helper(asm: &mut String) {
    writeln!(asm, "\n__aun_udiv32:").unwrap();
    writeln!(asm, "    xor a").unwrap();
    writeln!(asm, "    jr __aun_udivmod32").unwrap();
    writeln!(asm, "__aun_urem32:").unwrap();
    writeln!(asm, "    ld a, 1").unwrap();
    writeln!(asm, "__aun_udivmod32:").unwrap();
    writeln!(asm, "    ld bc, 0").unwrap();
    writeln!(asm, "    push bc").unwrap();
    writeln!(asm, "    push bc").unwrap();
    writeln!(asm, "    ld iy, 0").unwrap();
    writeln!(asm, "    add iy, sp").unwrap();
    writeln!(asm, "    ld c, a").unwrap();
    writeln!(asm, "    ld b, 32").unwrap();
    writeln!(asm, "__aun_udiv32_lp:").unwrap();
    writeln!(asm, "    sla e").unwrap();
    writeln!(asm, "    rl d").unwrap();
    writeln!(asm, "    rl l").unwrap();
    writeln!(asm, "    rl h").unwrap();
    writeln!(asm, "    rl (iy + 0)").unwrap();
    writeln!(asm, "    rl (iy + 1)").unwrap();
    writeln!(asm, "    rl (iy + 2)").unwrap();
    writeln!(asm, "    rl (iy + 3)").unwrap();
    writeln!(asm, "    jr c, __aun_udiv32_do_sub").unwrap();
    writeln!(asm, "    ld a, (iy + 3)").unwrap();
    writeln!(asm, "    cp (iy + 9)").unwrap();
    writeln!(asm, "    jr c, __aun_udiv32_no_sub").unwrap();
    writeln!(asm, "    jr nz, __aun_udiv32_do_sub").unwrap();
    writeln!(asm, "    ld a, (iy + 2)").unwrap();
    writeln!(asm, "    cp (iy + 8)").unwrap();
    writeln!(asm, "    jr c, __aun_udiv32_no_sub").unwrap();
    writeln!(asm, "    jr nz, __aun_udiv32_do_sub").unwrap();
    writeln!(asm, "    ld a, (iy + 1)").unwrap();
    writeln!(asm, "    cp (iy + 7)").unwrap();
    writeln!(asm, "    jr c, __aun_udiv32_no_sub").unwrap();
    writeln!(asm, "    jr nz, __aun_udiv32_do_sub").unwrap();
    writeln!(asm, "    ld a, (iy + 0)").unwrap();
    writeln!(asm, "    cp (iy + 6)").unwrap();
    writeln!(asm, "    jr c, __aun_udiv32_no_sub").unwrap();
    writeln!(asm, "__aun_udiv32_do_sub:").unwrap();
    writeln!(asm, "    ld a, (iy + 0)").unwrap();
    writeln!(asm, "    sub (iy + 6)").unwrap();
    writeln!(asm, "    ld (iy + 0), a").unwrap();
    writeln!(asm, "    ld a, (iy + 1)").unwrap();
    writeln!(asm, "    sbc a, (iy + 7)").unwrap();
    writeln!(asm, "    ld (iy + 1), a").unwrap();
    writeln!(asm, "    ld a, (iy + 2)").unwrap();
    writeln!(asm, "    sbc a, (iy + 8)").unwrap();
    writeln!(asm, "    ld (iy + 2), a").unwrap();
    writeln!(asm, "    ld a, (iy + 3)").unwrap();
    writeln!(asm, "    sbc a, (iy + 9)").unwrap();
    writeln!(asm, "    ld (iy + 3), a").unwrap();
    writeln!(asm, "    inc e").unwrap();
    writeln!(asm, "__aun_udiv32_no_sub:").unwrap();
    writeln!(asm, "    djnz __aun_udiv32_lp").unwrap();
    writeln!(asm, "    bit 0, c").unwrap();
    writeln!(asm, "    jr z, __aun_udiv32_done").unwrap();
    writeln!(asm, "    ld e, (iy + 0)").unwrap();
    writeln!(asm, "    ld d, (iy + 1)").unwrap();
    writeln!(asm, "    ld l, (iy + 2)").unwrap();
    writeln!(asm, "    ld h, (iy + 3)").unwrap();
    writeln!(asm, "__aun_udiv32_done:").unwrap();
    writeln!(asm, "    pop bc").unwrap();
    writeln!(asm, "    pop bc").unwrap();
    writeln!(asm, "    ret").unwrap();
}

/// Emits `__aun_sdiv32` and `__aun_srem32` wrapping `__aun_udiv32`/`__aun_urem32`.
/// Truncates toward zero; remainder carries dividend sign.
pub(super) fn emit_sdiv32_helper(asm: &mut String) {
    writeln!(asm, "\n__aun_sdiv32:").unwrap();
    writeln!(asm, "    ld c, 0").unwrap();
    writeln!(asm, "    jr __aun_sdivmod32").unwrap();
    writeln!(asm, "__aun_srem32:").unwrap();
    writeln!(asm, "    ld c, 1").unwrap();
    writeln!(asm, "__aun_sdivmod32:").unwrap();
    writeln!(asm, "    ld iy, 0").unwrap();
    writeln!(asm, "    add iy, sp").unwrap();
    writeln!(asm, "    bit 7, h").unwrap();
    writeln!(asm, "    jr z, __aun_sdiv_lhs_ok").unwrap();
    writeln!(asm, "    set 1, c").unwrap();
    writeln!(asm, "    set 2, c").unwrap();
    writeln!(asm, "    call __aun_neg_hlde").unwrap();
    writeln!(asm, "__aun_sdiv_lhs_ok:").unwrap();
    writeln!(asm, "    ld a, (iy + 5)").unwrap();
    writeln!(asm, "    bit 7, a").unwrap();
    writeln!(asm, "    jr z, __aun_sdiv_rhs_ok").unwrap();
    writeln!(asm, "    ld a, c").unwrap();
    writeln!(asm, "    xor 4").unwrap();
    writeln!(asm, "    ld c, a").unwrap();
    writeln!(asm, "    xor a").unwrap();
    writeln!(asm, "    sub (iy + 2)").unwrap();
    writeln!(asm, "    ld (iy + 2), a").unwrap();
    writeln!(asm, "    ld a, 0").unwrap();
    writeln!(asm, "    sbc a, (iy + 3)").unwrap();
    writeln!(asm, "    ld (iy + 3), a").unwrap();
    writeln!(asm, "    ld a, 0").unwrap();
    writeln!(asm, "    sbc a, (iy + 4)").unwrap();
    writeln!(asm, "    ld (iy + 4), a").unwrap();
    writeln!(asm, "    ld a, 0").unwrap();
    writeln!(asm, "    sbc a, (iy + 5)").unwrap();
    writeln!(asm, "    ld (iy + 5), a").unwrap();
    writeln!(asm, "__aun_sdiv_rhs_ok:").unwrap();
    writeln!(asm, "    push bc").unwrap();
    writeln!(asm, "    ld a, c").unwrap();
    writeln!(asm, "    ld c, (iy + 4)").unwrap();
    writeln!(asm, "    ld b, (iy + 5)").unwrap();
    writeln!(asm, "    push bc").unwrap();
    writeln!(asm, "    ld c, (iy + 2)").unwrap();
    writeln!(asm, "    ld b, (iy + 3)").unwrap();
    writeln!(asm, "    push bc").unwrap();
    writeln!(asm, "    bit 0, a").unwrap();
    writeln!(asm, "    jr nz, __aun_sdiv_call_rem").unwrap();
    writeln!(asm, "    call __aun_udiv32").unwrap();
    writeln!(asm, "    jr __aun_sdiv_after_call").unwrap();
    writeln!(asm, "__aun_sdiv_call_rem:").unwrap();
    writeln!(asm, "    call __aun_urem32").unwrap();
    writeln!(asm, "__aun_sdiv_after_call:").unwrap();
    writeln!(asm, "    pop bc").unwrap();
    writeln!(asm, "    pop bc").unwrap();
    writeln!(asm, "    pop bc").unwrap();
    writeln!(asm, "    bit 0, c").unwrap();
    writeln!(asm, "    jr nz, __aun_sdiv_chk_rem").unwrap();
    writeln!(asm, "    bit 2, c").unwrap();
    writeln!(asm, "    ret z").unwrap();
    writeln!(asm, "    jr __aun_neg_hlde").unwrap();
    writeln!(asm, "__aun_sdiv_chk_rem:").unwrap();
    writeln!(asm, "    bit 1, c").unwrap();
    writeln!(asm, "    ret z").unwrap();
    writeln!(asm, "__aun_neg_hlde:").unwrap();
    writeln!(asm, "    xor a").unwrap();
    writeln!(asm, "    sub e").unwrap();
    writeln!(asm, "    ld e, a").unwrap();
    writeln!(asm, "    ld a, 0").unwrap();
    writeln!(asm, "    sbc a, d").unwrap();
    writeln!(asm, "    ld d, a").unwrap();
    writeln!(asm, "    ld a, 0").unwrap();
    writeln!(asm, "    sbc a, l").unwrap();
    writeln!(asm, "    ld l, a").unwrap();
    writeln!(asm, "    ld a, 0").unwrap();
    writeln!(asm, "    sbc a, h").unwrap();
    writeln!(asm, "    ld h, a").unwrap();
    writeln!(asm, "    ret").unwrap();
}
