use crate::analysis::z80_sym;
use aun_ir::*;
use std::collections::HashMap;
use std::fmt::Write;

pub(super) fn emit_dispatch_stub<F>(
    program: &IrProgram,
    plan: &crate::overlay::OverlayPlan,
    resolved_overlay_syms: &HashMap<String, u16>,
    pass1_placeholders: bool,
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
        let target = resolve_fn(st)
            .map(|fid| {
                super::resolve_fn_target(
                    program,
                    plan,
                    resolved_overlay_syms,
                    pass1_placeholders,
                    fid,
                )
            })
            .unwrap_or_else(|| "__aun_noop_ret".to_string());
        writeln!(asm, "    .dw {target}").unwrap();
    }

    writeln!(asm, "{disp_lbl}:").unwrap();
    if use_smc {
        // Self-modifying direct JP (10T, 3B) — patched in RAM on every transition
        let init_target = resolve_fn(&mdef.states[mdef.initial_state as usize])
            .map(|fid| {
                super::resolve_fn_target(
                    program,
                    plan,
                    resolved_overlay_syms,
                    pass1_placeholders,
                    fid,
                )
            })
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

/// Returns `true` when `IndexAddr` on a fixed-length array (`len > 0`) with a non-power-of-two
/// `elem_size` has all valid byte offsets fitting in `u8` (`(len - 1) * elem_size <= 255`).
pub(super) fn is_small_u8_array_mul(len: u16, elem_size: u16) -> bool {
    len > 0
        && (2..=255).contains(&elem_size)
        && !elem_size.is_power_of_two()
        && ((len as u32) - 1) * (elem_size as u32) <= 255
}

/// Multiplies `A` (`0..=255`) by a compile-time constant `k` (`1..=255`) using `src_reg`
/// (which must already hold the same initial value as `A`, e.g., `"l"`, `"b"`, or `"c"`),
/// and stores the zero-extended 16-bit result into `HL` (`H = 0`, `L = A * k`).
///
/// Precondition: `(A as u16) * k <= 255` for all valid runtime inputs.
/// Clobbers `A, F, H, L`; preserves `B, C, D, E, IX, IY`.
pub(super) fn emit_mul_u8_const(k: u16, src_reg: &str, asm: &mut String) {
    debug_assert!((1..=255).contains(&k));
    debug_assert!(src_reg != "a" && src_reg != "h");
    if k == 1 {
        if src_reg != "l" {
            writeln!(asm, "    ld l, a").unwrap();
        }
        writeln!(asm, "    ld h, 0").unwrap();
        return;
    }
    let msb = 7 - (k as u8).leading_zeros();
    for bit in (0..msb).rev() {
        writeln!(asm, "    add a, a").unwrap();
        if ((k >> bit) & 1) != 0 {
            writeln!(asm, "    add a, {src_reg}").unwrap();
        }
    }
    writeln!(asm, "    ld l, a").unwrap();
    writeln!(asm, "    ld h, 0").unwrap();
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

/// Emits `__aun_mulu16_32`: unsigned widening 16x16->32 multiply.
/// Entry: `DE` = multiplier (u16), `BC` = multiplicand (u16).
/// Return: `HLDE` = product (`HL` hi16, `DE` lo16). Clobbers `A, B, C, F`. Preserves `IX`.
pub(super) fn emit_mulu16_32_helper(asm: &mut String) {
    writeln!(asm, "\n__aun_mulu16_32:").unwrap();
    writeln!(asm, "    ld a, d").unwrap();
    writeln!(asm, "    or a").unwrap();
    writeln!(asm, "    jr z, __aun_mulu16_32_de8").unwrap();
    writeln!(asm, "    ld a, b").unwrap();
    writeln!(asm, "    or a").unwrap();
    writeln!(asm, "    jr nz, __aun_mulu16_32_16x16").unwrap();
    writeln!(asm, "__aun_mulu16_32_8x16:").unwrap();
    writeln!(asm, "    ld hl, 0").unwrap();
    writeln!(asm, "    xor a").unwrap();
    writeln!(asm, "    ld b, 8").unwrap();
    writeln!(asm, "__aun_mulu16_32_8lp:").unwrap();
    writeln!(asm, "    add hl, hl").unwrap();
    writeln!(asm, "    rla").unwrap();
    writeln!(asm, "    sla c").unwrap();
    writeln!(asm, "    jr nc, __aun_mulu16_32_8sk").unwrap();
    writeln!(asm, "    add hl, de").unwrap();
    writeln!(asm, "    adc a, 0").unwrap();
    writeln!(asm, "__aun_mulu16_32_8sk:").unwrap();
    writeln!(asm, "    djnz __aun_mulu16_32_8lp").unwrap();
    writeln!(asm, "    ex de, hl").unwrap();
    writeln!(asm, "    ld l, a").unwrap();
    writeln!(asm, "    ld h, 0").unwrap();
    writeln!(asm, "    ret").unwrap();
    writeln!(asm, "__aun_mulu16_32_de8:").unwrap();
    writeln!(asm, "    ld d, b").unwrap();
    writeln!(asm, "    ld b, e").unwrap();
    writeln!(asm, "    ld e, c").unwrap();
    writeln!(asm, "    ld c, b").unwrap();
    writeln!(asm, "    jr __aun_mulu16_32_8x16").unwrap();
    writeln!(asm, "__aun_mulu16_32_16x16:").unwrap();
    writeln!(asm, "    ld hl, 0").unwrap();
    writeln!(asm, "    ld a, 16").unwrap();
    writeln!(asm, "__aun_mulu16_32_lp:").unwrap();
    writeln!(asm, "    add hl, hl").unwrap();
    writeln!(asm, "    rl e").unwrap();
    writeln!(asm, "    rl d").unwrap();
    writeln!(asm, "    jr nc, __aun_mulu16_32_sk").unwrap();
    writeln!(asm, "    add hl, bc").unwrap();
    writeln!(asm, "    jr nc, __aun_mulu16_32_sk").unwrap();
    writeln!(asm, "    inc de").unwrap();
    writeln!(asm, "__aun_mulu16_32_sk:").unwrap();
    writeln!(asm, "    dec a").unwrap();
    writeln!(asm, "    jr nz, __aun_mulu16_32_lp").unwrap();
    writeln!(asm, "    ex de, hl").unwrap();
    writeln!(asm, "    ret").unwrap();
}

/// Emits `__aun_muls16_32`: signed widening 16x16->32 multiply.
/// Entry: `DE` = lhs (i16), `BC` = rhs (i16).
/// Return: `HLDE` = product (`HL` hi16, `DE` lo16). Clobbers `A, B, C, F`. Preserves `IX`.
pub(super) fn emit_muls16_32_helper(asm: &mut String) {
    writeln!(asm, "\n__aun_muls16_32:").unwrap();
    writeln!(asm, "    ld a, d").unwrap();
    writeln!(asm, "    xor b").unwrap();
    writeln!(asm, "    push af").unwrap();
    writeln!(asm, "    bit 7, d").unwrap();
    writeln!(asm, "    jr z, __aun_muls16_32_pos1").unwrap();
    writeln!(asm, "    xor a").unwrap();
    writeln!(asm, "    sub e").unwrap();
    writeln!(asm, "    ld e, a").unwrap();
    writeln!(asm, "    sbc a, a").unwrap();
    writeln!(asm, "    sub d").unwrap();
    writeln!(asm, "    ld d, a").unwrap();
    writeln!(asm, "__aun_muls16_32_pos1:").unwrap();
    writeln!(asm, "    bit 7, b").unwrap();
    writeln!(asm, "    jr z, __aun_muls16_32_pos2").unwrap();
    writeln!(asm, "    xor a").unwrap();
    writeln!(asm, "    sub c").unwrap();
    writeln!(asm, "    ld c, a").unwrap();
    writeln!(asm, "    sbc a, a").unwrap();
    writeln!(asm, "    sub b").unwrap();
    writeln!(asm, "    ld b, a").unwrap();
    writeln!(asm, "__aun_muls16_32_pos2:").unwrap();
    writeln!(asm, "    call __aun_mulu16_32").unwrap();
    writeln!(asm, "    pop af").unwrap();
    writeln!(asm, "    rla").unwrap();
    writeln!(asm, "    ret nc").unwrap();
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

/// Emits `__aun_udiv32` and `__aun_urem32`: unsigned 32-bit div/rem.
/// Unsigned 32-bit shift-and-subtract division.
/// Entry: `HLDE` = dividend, `[sp+2..5]` = divisor (le).
/// Return: `HLDE` = quotient (`__aun_udiv32` / `__aun_udivrem32`) or remainder (`__aun_urem32`).
/// For `__aun_udivrem32`, `[sp+2..5]` (the stacked divisor frame) is overwritten with remainder.
/// Clobbers `A, B, C, F, IY`. Preserves `IX`.
pub(super) fn emit_udiv32_helper(asm: &mut String) {
    writeln!(asm, "\n__aun_udiv32:").unwrap();
    writeln!(asm, "    xor a").unwrap();
    writeln!(asm, "    jr __aun_udivmod32").unwrap();
    writeln!(asm, "__aun_urem32:").unwrap();
    writeln!(asm, "    ld a, 1").unwrap();
    writeln!(asm, "    jr __aun_udivmod32").unwrap();
    writeln!(asm, "__aun_udivrem32:").unwrap();
    writeln!(asm, "    ld a, 2").unwrap();
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
    writeln!(asm, "    jr nz, __aun_udiv32_want_rem").unwrap();
    writeln!(asm, "    bit 1, c").unwrap();
    writeln!(asm, "    jr z, __aun_udiv32_done").unwrap();
    writeln!(asm, "    ld a, (iy + 0)").unwrap();
    writeln!(asm, "    ld (iy + 6), a").unwrap();
    writeln!(asm, "    ld a, (iy + 1)").unwrap();
    writeln!(asm, "    ld (iy + 7), a").unwrap();
    writeln!(asm, "    ld a, (iy + 2)").unwrap();
    writeln!(asm, "    ld (iy + 8), a").unwrap();
    writeln!(asm, "    ld a, (iy + 3)").unwrap();
    writeln!(asm, "    ld (iy + 9), a").unwrap();
    writeln!(asm, "    jr __aun_udiv32_done").unwrap();
    writeln!(asm, "__aun_udiv32_want_rem:").unwrap();
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

/// Emits `__aun_udiv16` and `__aun_urem16`: unsigned 16-bit div/rem.
/// Entry: `HL` = dividend, `DE` = divisor.
/// Return: `HL` = quotient (`__aun_udiv16`) or remainder (`__aun_urem16`).
/// Clobbers `A, B, C, D, E, H, L, F`. Preserves `IX, IY`.
pub(super) fn emit_udiv16_helper(asm: &mut String) {
    writeln!(asm, "\n__aun_udiv16:").unwrap();
    writeln!(asm, "    call __aun_udivmod16").unwrap();
    writeln!(asm, "    ld h, b").unwrap();
    writeln!(asm, "    ld l, c").unwrap();
    writeln!(asm, "    ret").unwrap();
    writeln!(asm, "__aun_urem16:").unwrap();
    writeln!(asm, "__aun_udivmod16:").unwrap();
    writeln!(asm, "    ld b, h").unwrap();
    writeln!(asm, "    ld c, l").unwrap();
    writeln!(asm, "    ld hl, 0").unwrap();
    writeln!(asm, "    ld a, 16").unwrap();
    writeln!(asm, "__aun_udiv16_lp:").unwrap();
    writeln!(asm, "    sla c").unwrap();
    writeln!(asm, "    rl b").unwrap();
    writeln!(asm, "    adc hl, hl").unwrap();
    writeln!(asm, "    or a").unwrap();
    writeln!(asm, "    sbc hl, de").unwrap();
    writeln!(asm, "    jr nc, __aun_udiv16_set").unwrap();
    writeln!(asm, "    add hl, de").unwrap();
    writeln!(asm, "    jr __aun_udiv16_next").unwrap();
    writeln!(asm, "__aun_udiv16_set:").unwrap();
    writeln!(asm, "    inc c").unwrap();
    writeln!(asm, "__aun_udiv16_next:").unwrap();
    writeln!(asm, "    dec a").unwrap();
    writeln!(asm, "    jr nz, __aun_udiv16_lp").unwrap();
    writeln!(asm, "    ret").unwrap();
}

/// Emits `__aun_sdiv16` and `__aun_srem16` wrapping `__aun_udivmod16`.
/// Entry: `HL` = dividend, `DE` = divisor.
/// Return: `HL` = signed quotient (`__aun_sdiv16`) or remainder (`__aun_srem16`, dividend sign).
/// Clobbers `A, B, C, D, E, H, L, F`. Preserves `IX, IY`.
pub(super) fn emit_sdiv16_helper(asm: &mut String) {
    writeln!(asm, "\n__aun_sdiv16:").unwrap();
    writeln!(asm, "    ld a, h").unwrap();
    writeln!(asm, "    xor d").unwrap();
    writeln!(asm, "    push af").unwrap();
    writeln!(asm, "    call __aun_sdivmod16_abs").unwrap();
    writeln!(asm, "    ld h, b").unwrap();
    writeln!(asm, "    ld l, c").unwrap();
    writeln!(asm, "    pop af").unwrap();
    writeln!(asm, "    bit 7, a").unwrap();
    writeln!(asm, "    ret z").unwrap();
    writeln!(asm, "    jr __aun_neg_hl").unwrap();
    writeln!(asm, "__aun_srem16:").unwrap();
    writeln!(asm, "    ld a, h").unwrap();
    writeln!(asm, "    push af").unwrap();
    writeln!(asm, "    call __aun_sdivmod16_abs").unwrap();
    writeln!(asm, "    pop af").unwrap();
    writeln!(asm, "    bit 7, a").unwrap();
    writeln!(asm, "    ret z").unwrap();
    writeln!(asm, "    jr __aun_neg_hl").unwrap();
    writeln!(asm, "__aun_sdivmod16_abs:").unwrap();
    writeln!(asm, "    bit 7, h").unwrap();
    writeln!(asm, "    jr z, __aun_sdiv16_lhs_ok").unwrap();
    writeln!(asm, "    call __aun_neg_hl").unwrap();
    writeln!(asm, "__aun_sdiv16_lhs_ok:").unwrap();
    writeln!(asm, "    bit 7, d").unwrap();
    writeln!(asm, "    jr z, __aun_sdiv16_rhs_ok").unwrap();
    writeln!(asm, "    ex de, hl").unwrap();
    writeln!(asm, "    call __aun_neg_hl").unwrap();
    writeln!(asm, "    ex de, hl").unwrap();
    writeln!(asm, "__aun_sdiv16_rhs_ok:").unwrap();
    writeln!(asm, "    jp __aun_udivmod16").unwrap();
    writeln!(asm, "__aun_neg_hl:").unwrap();
    writeln!(asm, "    xor a").unwrap();
    writeln!(asm, "    sub l").unwrap();
    writeln!(asm, "    ld l, a").unwrap();
    writeln!(asm, "    sbc a, a").unwrap();
    writeln!(asm, "    sub h").unwrap();
    writeln!(asm, "    ld h, a").unwrap();
    writeln!(asm, "    ret").unwrap();
}

/// Emits `__aun_udiv8` and `__aun_urem8`: unsigned 8-bit div/rem.
/// Entry: `A` = dividend, `E` = divisor.
/// Return: `A` = quotient (`__aun_udiv8`) or remainder (`__aun_urem8`).
/// Clobbers `A, B, D, E, F`. Preserves `BC`'s `C` (clobbers `B`), `HL, IX, IY`.
pub(super) fn emit_udiv8_helper(asm: &mut String) {
    writeln!(asm, "\n__aun_udiv8:").unwrap();
    writeln!(asm, "    call __aun_udivmod8").unwrap();
    writeln!(asm, "    ld a, d").unwrap();
    writeln!(asm, "    ret").unwrap();
    writeln!(asm, "__aun_urem8:").unwrap();
    writeln!(asm, "__aun_udivmod8:").unwrap();
    writeln!(asm, "    ld d, a").unwrap();
    writeln!(asm, "    xor a").unwrap();
    writeln!(asm, "    ld b, 8").unwrap();
    writeln!(asm, "__aun_udiv8_lp:").unwrap();
    writeln!(asm, "    sla d").unwrap();
    writeln!(asm, "    rla").unwrap();
    writeln!(asm, "    cp e").unwrap();
    writeln!(asm, "    jr c, __aun_udiv8_next").unwrap();
    writeln!(asm, "    sub e").unwrap();
    writeln!(asm, "    inc d").unwrap();
    writeln!(asm, "__aun_udiv8_next:").unwrap();
    writeln!(asm, "    djnz __aun_udiv8_lp").unwrap();
    writeln!(asm, "    ret").unwrap();
}

/// Emits `__aun_sdiv8` and `__aun_srem8` wrapping `__aun_udivmod8`.
/// Entry: `A` = dividend, `E` = divisor.
/// Return: `A` = signed quotient (`__aun_sdiv8`) or remainder (`__aun_srem8`, dividend sign).
/// Clobbers `A, B, C, D, E, F`. Preserves `HL, IX, IY`.
pub(super) fn emit_sdiv8_helper(asm: &mut String) {
    writeln!(asm, "\n__aun_sdiv8:").unwrap();
    writeln!(asm, "    ld c, a").unwrap();
    writeln!(asm, "    xor e").unwrap();
    writeln!(asm, "    push af").unwrap();
    writeln!(asm, "    ld a, c").unwrap();
    writeln!(asm, "    call __aun_sdivmod8_abs").unwrap();
    writeln!(asm, "    ld a, d").unwrap();
    writeln!(asm, "    pop bc").unwrap();
    writeln!(asm, "    bit 7, b").unwrap();
    writeln!(asm, "    ret z").unwrap();
    writeln!(asm, "    neg").unwrap();
    writeln!(asm, "    ret").unwrap();
    writeln!(asm, "__aun_srem8:").unwrap();
    writeln!(asm, "    push af").unwrap();
    writeln!(asm, "    call __aun_sdivmod8_abs").unwrap();
    writeln!(asm, "    pop bc").unwrap();
    writeln!(asm, "    bit 7, b").unwrap();
    writeln!(asm, "    ret z").unwrap();
    writeln!(asm, "    neg").unwrap();
    writeln!(asm, "    ret").unwrap();
    writeln!(asm, "__aun_sdivmod8_abs:").unwrap();
    writeln!(asm, "    bit 7, a").unwrap();
    writeln!(asm, "    jr z, __aun_sdiv8_lhs_ok").unwrap();
    writeln!(asm, "    neg").unwrap();
    writeln!(asm, "__aun_sdiv8_lhs_ok:").unwrap();
    writeln!(asm, "    bit 7, e").unwrap();
    writeln!(asm, "    jr z, __aun_sdiv8_rhs_ok").unwrap();
    writeln!(asm, "    ld c, a").unwrap();
    writeln!(asm, "    ld a, e").unwrap();
    writeln!(asm, "    neg").unwrap();
    writeln!(asm, "    ld e, a").unwrap();
    writeln!(asm, "    ld a, c").unwrap();
    writeln!(asm, "__aun_sdiv8_rhs_ok:").unwrap();
    writeln!(asm, "    jp __aun_udivmod8").unwrap();
}

/// Emits `__aun_mul8`: unsigned/wrapping 8x8->8 multiply.
/// Entry: `A` = lhs, `E` = rhs.
/// Return: `A` = low 8 bits of product. Clobbers `B, D, E, F`. Preserves `C, HL, IX, IY`.
pub(super) fn emit_mul8_helper(asm: &mut String) {
    writeln!(asm, "\n__aun_mul8:").unwrap();
    writeln!(asm, "    ld d, a").unwrap();
    writeln!(asm, "    xor a").unwrap();
    writeln!(asm, "    ld b, 8").unwrap();
    writeln!(asm, "__aun_mul8_lp:").unwrap();
    writeln!(asm, "    srl d").unwrap();
    writeln!(asm, "    jr nc, __aun_mul8_skip").unwrap();
    writeln!(asm, "    add a, e").unwrap();
    writeln!(asm, "__aun_mul8_skip:").unwrap();
    writeln!(asm, "    sla e").unwrap();
    writeln!(asm, "    djnz __aun_mul8_lp").unwrap();
    writeln!(asm, "    ret").unwrap();
}

/// Emits `__aun_udivrem32_8`: register-only unsigned `u32 / u8` div/rem (`divisor` in `1..=255`).
/// Entry: `HLDE` = 32-bit unsigned dividend, `C` = non-zero 8-bit unsigned divisor.
/// Return: `HLDE` = 32-bit quotient, `A` = 8-bit remainder.
/// Clobbers `B, F`. Preserves `C, IX, IY` (0 stack, 0 RAM, 0 `IY`).
pub(super) fn emit_udiv32_8_helper(asm: &mut String) {
    writeln!(asm, "\n__aun_udivrem32_8:").unwrap();
    writeln!(asm, "    xor a").unwrap();
    writeln!(asm, "    ld b, 32").unwrap();
    writeln!(asm, "__aun_udivrem32_8_lp:").unwrap();
    writeln!(asm, "    sla e").unwrap();
    writeln!(asm, "    rl d").unwrap();
    writeln!(asm, "    rl l").unwrap();
    writeln!(asm, "    rl h").unwrap();
    writeln!(asm, "    rla").unwrap();
    writeln!(asm, "    jr c, __aun_udivrem32_8_sub").unwrap();
    writeln!(asm, "    cp c").unwrap();
    writeln!(asm, "    jr c, __aun_udivrem32_8_next").unwrap();
    writeln!(asm, "__aun_udivrem32_8_sub:").unwrap();
    writeln!(asm, "    sub c").unwrap();
    writeln!(asm, "    inc e").unwrap();
    writeln!(asm, "__aun_udivrem32_8_next:").unwrap();
    writeln!(asm, "    djnz __aun_udivrem32_8_lp").unwrap();
    writeln!(asm, "    ret").unwrap();
}

/// Returns `true` if `state_id` owns any mutable non-ROM `#[zero_on_enter]` buffer.
pub(super) fn has_zero_state_buffers(program: &IrProgram, machine_id: u32, state_id: u8) -> bool {
    program.globals.iter().any(|g| {
        g.owner_state == Some((machine_id, state_id))
            && g.zero_on_enter
            && !g.in_rom
            && !g.is_extern
            && program.type_size(&g.ty) > 0
    })
}

/// Emits Z80 zero-initialization sequences for any `#[zero_on_enter]` state buffers
/// belonging to states in `enter_states`. Clobbers `A`, `HL`, `DE`, `BC` when returning `true`.
pub(super) fn emit_zero_state_buffers(
    asm: &mut String,
    program: &IrProgram,
    layout: &crate::uchiguri::RamLayout,
    machine_id: u32,
    enter_states: &[u8],
) -> bool {
    let mut emitted = false;
    for entry in &layout.global_entries {
        if entry.size == 0 {
            continue;
        }
        let Some(g) = program.globals.get(entry.global_id as usize) else {
            continue;
        };
        if !g.zero_on_enter || g.in_rom || g.is_extern {
            continue;
        }
        let Some((mid, sid)) = g.owner_state else {
            continue;
        };
        if mid != machine_id || !enter_states.contains(&sid) {
            continue;
        }
        let (addr_expr, addr_next) = if g.fixed_addr.is_none() && g.space.is_none() {
            (
                super::fmt_uchiguri_imm(layout.ram_base, entry.addr),
                super::fmt_uchiguri_imm(layout.ram_base, entry.addr.wrapping_add(1)),
            )
        } else {
            (
                format!("0x{:04X}", entry.addr),
                format!("0x{:04X}", entry.addr.wrapping_add(1)),
            )
        };
        match entry.size {
            1 => {
                writeln!(asm, "    xor a").unwrap();
                writeln!(asm, "    ld ({addr_expr}), a").unwrap();
            }
            2 => {
                writeln!(asm, "    ld hl, 0").unwrap();
                writeln!(asm, "    ld ({addr_expr}), hl").unwrap();
            }
            n => {
                writeln!(asm, "    ld hl, {addr_expr}").unwrap();
                writeln!(asm, "    ld (hl), 0").unwrap();
                writeln!(asm, "    ld de, {addr_next}").unwrap();
                writeln!(asm, "    ld bc, {}", n - 1).unwrap();
                writeln!(asm, "    ldir").unwrap();
            }
        }
        emitted = true;
    }
    emitted
}

/// Emits Resident ROM `LoadOp` tables (`__aun_swap_m{mid}_t{to}`) and deduplicated
/// overlay/HSM transition trampolines (`__aun_tr_...`) so cross-overlay and nested
/// transitions execute safely from Resident code without self-overwriting.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_swap_tables_and_trampolines(
    program: &IrProgram,
    layout: &crate::uchiguri::RamLayout,
    opts: &crate::CompileOptions,
    plan: &crate::overlay::OverlayPlan,
    resolved_overlay_syms: &HashMap<String, u16>,
    resolved_blob_spans: &[(u16, u16)],
    pass1_placeholders: bool,
    trampolines: &std::collections::BTreeSet<super::TrampolineKey>,
    panic_stub: &mut bool,
    asm: &mut String,
) {
    let swap_fn_sym: Option<String> = program
        .target
        .as_ref()
        .and_then(|t| t.on_swap_fn.as_ref())
        .map(|n| z80_sym(n));

    // 1. Emit per-(machine, target_leaf) ROM LoadOp tables
    if swap_fn_sym.is_some() {
        let mut emitted_tables: std::collections::BTreeSet<(u32, u8)> =
            std::collections::BTreeSet::new();
        for key in trampolines {
            if !emitted_tables.insert((key.machine_id, key.to_state)) {
                continue;
            }
            let all_blobs = plan.blobs_for_enter_chain(program, key.machine_id, None, key.to_state);
            if all_blobs.is_empty() {
                continue;
            }
            writeln!(asm, "\n__aun_swap_m{}_t{}:", key.machine_id, key.to_state).unwrap();
            for &bi in &all_blobs {
                let spec = &plan.blobs[bi];
                let (dst, size) = resolved_blob_spans.get(bi).copied().unwrap_or((0, 0));
                writeln!(asm, "    .db 0x{:02X}, 0x{:02X}", spec.blob_id, spec.space_id).unwrap();
                writeln!(asm, "    .dw 0x{dst:04X}, 0x{size:04X}").unwrap();
            }
        }
    }

    let emit_swap_slice = |asm: &mut String, mid: u32, to_state: u8, lca: Option<u8>| {
        let enter_blobs = plan.blobs_for_enter_chain(program, mid, lca, to_state);
        if enter_blobs.is_empty() {
            return;
        }
        if let Some(ref ssym) = swap_fn_sym {
            let all_blobs = plan.blobs_for_enter_chain(program, mid, None, to_state);
            let skip = all_blobs.len().saturating_sub(enter_blobs.len());
            let count = enter_blobs.len();
            let tbl_lbl = format!("__aun_swap_m{mid}_t{to_state}");
            if skip == 0 {
                writeln!(asm, "    ld de, {tbl_lbl}").unwrap();
            } else {
                writeln!(asm, "    ld de, {tbl_lbl} + {}", skip * 6).unwrap();
            }
            writeln!(asm, "    ld hl, {count}").unwrap();
            writeln!(asm, "    call {ssym}").unwrap();
        } else {
            for &bi in &enter_blobs {
                let spec = &plan.blobs[bi];
                if let Some(ref load_fn) = spec.load_fn {
                    let has_load_sym = program
                        .functions
                        .iter()
                        .any(|f| !f.is_test && &f.name == load_fn)
                        || program.externs.iter().any(|e| &e.name == load_fn);
                    if has_load_sym {
                        let arg_val = if spec.slot_name.is_some() {
                            spec.state_id
                        } else {
                            spec.blob_id
                        };
                        writeln!(asm, "    ld a, {arg_val}").unwrap();
                        writeln!(asm, "    call {load_fn}").unwrap();
                    }
                }
            }
        }
    };

    let emit_copy_tag_smc = |asm: &mut String, mdef: &MachineDef, key: super::TrampolineKey| {
        if key.copy_end > key.copy_start
            && let Some(stage_base) = layout.machine_stage_addr
        {
            let copy_len = key.copy_end - key.copy_start;
            let src_addr = stage_base.wrapping_add(key.copy_start);
            let dst_off = mdef.payload_offset.wrapping_add(key.copy_start);
            writeln!(asm, "    pop hl").unwrap();
            writeln!(asm, "    push hl").unwrap();
            if dst_off > 0 {
                writeln!(asm, "    ld de, 0x{dst_off:04X}").unwrap();
                writeln!(asm, "    add hl, de").unwrap();
            }
            writeln!(asm, "    ex de, hl").unwrap();
            writeln!(
                asm,
                "    ld hl, {}",
                super::fmt_uchiguri_imm(layout.ram_base, src_addr)
            )
            .unwrap();
            if copy_len == 1 {
                writeln!(asm, "    ld a, (hl)").unwrap();
                writeln!(asm, "    ld (de), a").unwrap();
            } else {
                writeln!(asm, "    ld bc, {copy_len}").unwrap();
                writeln!(asm, "    ldir").unwrap();
            }
        }

        writeln!(asm, "    pop hl").unwrap();
        writeln!(asm, "    push hl").unwrap();
        if mdef.tag_offset > 0 {
            writeln!(asm, "    ld de, 0x{:04X}", mdef.tag_offset).unwrap();
            writeln!(asm, "    add hl, de").unwrap();
        }
        writeln!(asm, "    ld (hl), {}", key.to_state).unwrap();
        writeln!(asm, "    inc hl").unwrap();
        writeln!(asm, "    ld (hl), 0").unwrap();

        if mdef.smc {
            let mut methods: std::collections::BTreeSet<String> =
                std::collections::BTreeSet::new();
            for st in &mdef.states {
                for m in st.methods.keys() {
                    methods.insert(m.clone());
                }
            }
            for method in methods {
                let target = mdef.states[key.to_state as usize]
                    .methods
                    .get(&method)
                    .map(|&fid| {
                        super::resolve_fn_target(
                            program,
                            plan,
                            resolved_overlay_syms,
                            pass1_placeholders,
                            fid,
                        )
                    })
                    .unwrap_or_else(|| "__aun_noop_ret".to_string());
                writeln!(asm, "    ld hl, {target}").unwrap();
                writeln!(asm, "    ld ({}_disp_{method}+1), hl", mdef.name).unwrap();
            }
        }
    };

    let emit_enter_chain = |asm: &mut String, fns: &[u32]| {
        if fns.is_empty() {
            writeln!(asm, "    ret").unwrap();
            return;
        }
        for (i, &fid) in fns.iter().enumerate() {
            let tgt = super::resolve_fn_target(
                program,
                plan,
                resolved_overlay_syms,
                pass1_placeholders,
                fid,
            );
            if i + 1 < fns.len() {
                writeln!(asm, "    push hl").unwrap();
                writeln!(asm, "    call {tgt}").unwrap();
                writeln!(asm, "    pop hl").unwrap();
            } else {
                writeln!(asm, "    jp {tgt}").unwrap();
            }
        }
    };

    // 2. Emit each deduplicated Resident transition trampoline
    for &key in trampolines {
        let lbl = key.label();
        let mdef = &program.machines[key.machine_id as usize];
        writeln!(asm, "\n{lbl}:").unwrap();

        if let Some(from) = key.from_state {
            writeln!(asm, "    push hl").unwrap();
            let lca = mdef.lca(from, key.to_state);
            for anc in mdef.ancestors(from).into_iter().rev() {
                let below_lca = lca.is_none_or(|l| anc != l && mdef.is_ancestor(l, anc));
                if below_lca && let Some(exit_fn) = mdef.states[anc as usize].on_exit {
                    let tgt = super::resolve_fn_target(
                        program,
                        plan,
                        resolved_overlay_syms,
                        pass1_placeholders,
                        exit_fn,
                    );
                    writeln!(asm, "    pop hl").unwrap();
                    writeln!(asm, "    push hl").unwrap();
                    writeln!(asm, "    call {tgt}").unwrap();
                }
            }

            emit_swap_slice(asm, key.machine_id, key.to_state, lca);
            emit_copy_tag_smc(asm, mdef, key);
            let enter_states: Vec<u8> = mdef
                .ancestors(key.to_state)
                .into_iter()
                .filter(|&a| lca.is_none_or(|l| a != l && mdef.is_ancestor(l, a)))
                .collect();
            emit_zero_state_buffers(asm, program, layout, key.machine_id, &enter_states);
            writeln!(asm, "    pop hl").unwrap();

            let enter_fns: Vec<u32> = enter_states
                .iter()
                .filter_map(|&a| mdef.states[a as usize].on_enter)
                .collect();
            emit_enter_chain(asm, &enter_fns);
            continue;
        }

        // Dynamic `from_state == None`: HL = maddr, DE = site_id (if check_transitions)
        writeln!(asm, "    push hl").unwrap();
        if mdef.tag_offset > 0 {
            writeln!(asm, "    ld bc, 0x{:04X}", mdef.tag_offset).unwrap();
            writeln!(asm, "    add hl, bc").unwrap();
        }
        writeln!(asm, "    ld a, (hl)").unwrap();

        if opts.check_transitions {
            *panic_stub = true;
            writeln!(asm, "    push af").unwrap();
            writeln!(asm, "    add a, a").unwrap();
            writeln!(asm, "    ld c, a").unwrap();
            writeln!(asm, "    ld b, 0").unwrap();
            writeln!(asm, "    ld hl, {}_tmask", mdef.name).unwrap();
            writeln!(asm, "    add hl, bc").unwrap();
            let bit = key.to_state;
            if bit < 8 {
                writeln!(asm, "    ld a, (hl)").unwrap();
                writeln!(asm, "    bit {bit}, a").unwrap();
            } else {
                writeln!(asm, "    inc hl").unwrap();
                writeln!(asm, "    ld a, (hl)").unwrap();
                writeln!(asm, "    bit {}, a", bit - 8).unwrap();
            }
            writeln!(asm, "    jr nz, {lbl}_ok").unwrap();
            writeln!(asm, "    pop af").unwrap();
            writeln!(asm, "    pop hl").unwrap();
            writeln!(asm, "    ex de, hl").unwrap();
            writeln!(asm, "    ld a, {}", PanicCode::InvalidTransition as u8).unwrap();
            writeln!(asm, "    call __aun_panic").unwrap();
            writeln!(asm, "{lbl}_ok:").unwrap();
            writeln!(asm, "    pop af").unwrap();
        }

        // Keep Stack as [ret_pc, old_state, maddr] with A = old_state
        writeln!(asm, "    pop bc").unwrap();
        writeln!(asm, "    push af").unwrap();
        writeln!(asm, "    push bc").unwrap();

        // Dynamic bottom-up on_exit
        if mdef.states.iter().any(|s| s.on_exit.is_some()) {
            if mdef.states.iter().all(|s| s.parent.is_none()) {
                writeln!(asm, "    pop hl").unwrap();
                writeln!(asm, "    push hl").unwrap();
                writeln!(asm, "    call {}_disp_on_exit", mdef.name).unwrap();
            } else {
                let mut exit_cases: Vec<(u8, Vec<u32>)> = Vec::new();
                for st in mdef.states.iter().filter(|s| s.is_leaf) {
                    let lca = mdef.lca(st.id, key.to_state);
                    let fns: Vec<u32> = mdef
                        .ancestors(st.id)
                        .into_iter()
                        .rev()
                        .filter(|&a| lca.is_none_or(|l| a != l && mdef.is_ancestor(l, a)))
                        .filter_map(|a| mdef.states[a as usize].on_exit)
                        .collect();
                    if !fns.is_empty() {
                        exit_cases.push((st.id, fns));
                    }
                }
                if !exit_cases.is_empty() {
                    for (ci, (sid, fns)) in exit_cases.iter().enumerate() {
                        writeln!(asm, "    cp {sid}").unwrap();
                        writeln!(asm, "    jr nz, {lbl}_ex_nx{ci}").unwrap();
                        for &fid in fns {
                            let tgt = super::resolve_fn_target(
                                program,
                                plan,
                                resolved_overlay_syms,
                                pass1_placeholders,
                                fid,
                            );
                            writeln!(asm, "    pop hl").unwrap();
                            writeln!(asm, "    push hl").unwrap();
                            writeln!(asm, "    call {tgt}").unwrap();
                        }
                        writeln!(asm, "    jp {lbl}_ex_done").unwrap();
                        writeln!(asm, "{lbl}_ex_nx{ci}:").unwrap();
                    }
                    writeln!(asm, "{lbl}_ex_done:").unwrap();
                }
            }
        }

        // Dynamic overlay swap suffix strictly below runtime enclosing-boundary LCA
        let all_blobs = plan.blobs_for_enter_chain(program, key.machine_id, None, key.to_state);
        if !all_blobs.is_empty() {
            writeln!(asm, "    pop de").unwrap();
            writeln!(asm, "    pop af").unwrap();
            writeln!(asm, "    push af").unwrap();
            writeln!(asm, "    push de").unwrap();
            for (rev_k, &bi) in all_blobs.iter().enumerate().rev() {
                let b_state = plan.blobs[bi].state_id;
                let st_b = &mdef.states[b_state as usize];
                let (lo, hi) = st_b.leaf_range;
                if lo > 0 {
                    writeln!(asm, "    cp {lo}").unwrap();
                    writeln!(asm, "    jr c, {lbl}_sw_nx{rev_k}").unwrap();
                }
                if hi < 255 {
                    writeln!(asm, "    cp {}", hi + 1).unwrap();
                    writeln!(asm, "    jr nc, {lbl}_sw_nx{rev_k}").unwrap();
                }
                emit_swap_slice(asm, key.machine_id, key.to_state, Some(b_state));
                writeln!(asm, "    jp {lbl}_sw_done").unwrap();
                writeln!(asm, "{lbl}_sw_nx{rev_k}:").unwrap();
            }
            emit_swap_slice(asm, key.machine_id, key.to_state, None);
            writeln!(asm, "{lbl}_sw_done:").unwrap();
        }

        // Staged copy, tag write, SMC patch
        emit_copy_tag_smc(asm, mdef, key);

        // Pop maddr into HL and old_state into A (stack clean!)
        writeln!(asm, "    pop hl").unwrap();
        writeln!(asm, "    pop af").unwrap();

        // Dynamic top-down on_enter strictly below runtime LCA
        let ancs = mdef.ancestors(key.to_state);
        let any_enter = ancs
            .iter()
            .any(|&a| mdef.states[a as usize].on_enter.is_some());
        let any_zero = ancs
            .iter()
            .any(|&a| has_zero_state_buffers(program, key.machine_id, a));
        if !any_enter && !any_zero {
            writeln!(asm, "    ret").unwrap();
        } else if mdef.states.iter().all(|s| s.parent.is_none()) {
            let enter_fns: Vec<u32> = ancs
                .iter()
                .filter_map(|&a| mdef.states[a as usize].on_enter)
                .collect();
            if any_zero {
                if !enter_fns.is_empty() {
                    writeln!(asm, "    push hl").unwrap();
                }
                emit_zero_state_buffers(asm, program, layout, key.machine_id, &ancs);
                if !enter_fns.is_empty() {
                    writeln!(asm, "    pop hl").unwrap();
                }
            }
            emit_enter_chain(asm, &enter_fns);
        } else {
            for (ci, &anc) in ancs.iter().enumerate().rev() {
                let st_anc = &mdef.states[anc as usize];
                let (lo, hi) = st_anc.leaf_range;
                if lo > 0 {
                    writeln!(asm, "    cp {lo}").unwrap();
                    writeln!(asm, "    jr c, {lbl}_en_nx{ci}").unwrap();
                }
                if hi < 255 {
                    writeln!(asm, "    cp {}", hi + 1).unwrap();
                    writeln!(asm, "    jr nc, {lbl}_en_nx{ci}").unwrap();
                }
                let enter_states: Vec<u8> = ancs
                    .iter()
                    .copied()
                    .filter(|&a| a != anc && mdef.is_ancestor(anc, a))
                    .collect();
                let enter_fns: Vec<u32> = enter_states
                    .iter()
                    .filter_map(|&a| mdef.states[a as usize].on_enter)
                    .collect();
                if enter_states
                    .iter()
                    .any(|&a| has_zero_state_buffers(program, key.machine_id, a))
                {
                    if !enter_fns.is_empty() {
                        writeln!(asm, "    push hl").unwrap();
                    }
                    emit_zero_state_buffers(asm, program, layout, key.machine_id, &enter_states);
                    if !enter_fns.is_empty() {
                        writeln!(asm, "    pop hl").unwrap();
                    }
                }
                emit_enter_chain(asm, &enter_fns);
                writeln!(asm, "{lbl}_en_nx{ci}:").unwrap();
            }
            let enter_fns: Vec<u32> = ancs
                .iter()
                .filter_map(|&a| mdef.states[a as usize].on_enter)
                .collect();
            if any_zero {
                if !enter_fns.is_empty() {
                    writeln!(asm, "    push hl").unwrap();
                }
                emit_zero_state_buffers(asm, program, layout, key.machine_id, &ancs);
                if !enter_fns.is_empty() {
                    writeln!(asm, "    pop hl").unwrap();
                }
            }
            emit_enter_chain(asm, &enter_fns);
        }
    }
}
