use super::runtime::{emit_bank_switch, emit_mul_hl_const, ranges_overlap};
use super::{EmitCtx, IndexKey};
use crate::analysis::{
    can_alias_c_load, can_alias_ix_load, can_alias_local_load, extern_clobbers_c,
    is_const_addr_always_folded, is_const_always_folded, is_signed_ty, z80_sym,
};
use aun_ir::*;
use std::collections::BTreeSet;
use std::fmt::Write;

impl<'a> EmitCtx<'a> {
    pub(super) fn emit_inst(
        &mut self,
        bb_idx: usize,
        inst_idx: usize,
        inst: &IrInst,
    ) -> Result<(), String> {
        let ix_param_slot = self.fa.ix_param_slot;
        let reg_c_slot = self.fa.reg_c_slot;
        let fused_cmp_dsts = &self.fa.fused_cmp_dsts;
        let val_sizes = &self.fa.val_sizes;

        // Invalidate same-block shift CSE on any scratch write overlapping `prev_da`
        if let Some(d) = inst.dst() {
            let da = self.val_addr(d);
            let dsz = val_sizes.get(&d).copied().unwrap_or(2).clamp(1, 2);
            self.shift_cse
                .retain(|&(src_a, _, _), &mut (_, pda)| pda != da && src_a != da);
            self.index_cse.retain(|(_, ik, _), &mut res_a| {
                let idx_ok = match *ik {
                    IndexKey::RegC => true,
                    IndexKey::Mem(ia, isz) => !ranges_overlap(ia, isz, da, dsz),
                };
                idx_ok && !ranges_overlap(res_a, 2, da, dsz)
            });
        }
        // Invalidate same-block shift CSE, index CSE, and non-negative IX tracking on memory/local mutations
        match inst {
            IrInst::StoreLocal { slot, ty, .. } => {
                if reg_c_slot == Some(*slot) {
                    self.index_cse.retain(|(_, ik, _), _| *ik != IndexKey::RegC);
                } else {
                    let sa = self.layout.slot_addrs[&(self.func.id, *slot)];
                    let ssz = self.program.type_size(ty).clamp(1, 2);
                    self.shift_cse
                        .retain(|&(src_a, _, _), _| !ranges_overlap(src_a, 2, sa, ssz));
                    self.index_cse.retain(|(_, ik, _), &mut res_a| {
                        let idx_ok = match *ik {
                            IndexKey::RegC => true,
                            IndexKey::Mem(ia, isz) => !ranges_overlap(ia, isz, sa, ssz),
                        };
                        idx_ok && !ranges_overlap(res_a, 2, sa, ssz)
                    });
                }
            }
            IrInst::StorePtr { .. }
            | IrInst::Call { .. }
            | IrInst::ExternCall { .. }
            | IrInst::StateDispatch { .. }
            | IrInst::StateTransition { .. }
            | IrInst::PoolAlloc { .. }
            | IrInst::PoolFree { .. }
            | IrInst::InlineAsm { .. } => {
                self.shift_cse.clear();
                self.index_cse.clear();
                self.block_nonneg_ix = None;
            }
            _ => {}
        }
        match inst {
            IrInst::ConstInt { dst, ty, val } => {
                self.const_vals.insert(*dst, *val);
                if is_const_always_folded(
                    self.program,
                    self.func,
                    *dst,
                    fused_cmp_dsts,
                    self.fused_skip_insts,
                ) {
                    return Ok(());
                }
                let da = self.val_addr(*dst);
                match self.program.type_size(ty) {
                    1 => {
                        let b = (*val & 0xFF) as u8;
                        if b == 0 {
                            writeln!(self.asm, "    xor a").unwrap();
                        } else {
                            writeln!(self.asm, "    ld a, 0x{b:02X}").unwrap();
                        }
                        writeln!(self.asm, "    ld (0x{da:04X}), a").unwrap();
                    }
                    2 => {
                        let w = *val as u16;
                        writeln!(self.asm, "    ld hl, 0x{w:04X}").unwrap();
                        writeln!(self.asm, "    ld (0x{da:04X}), hl").unwrap();
                    }
                    _ => {}
                }
            }
            IrInst::ConstBool { dst, val } => {
                self.const_vals.insert(*dst, if *val { 1 } else { 0 });
                if is_const_always_folded(
                    self.program,
                    self.func,
                    *dst,
                    fused_cmp_dsts,
                    self.fused_skip_insts,
                ) {
                    return Ok(());
                }
                let da = self.val_addr(*dst);
                if *val {
                    writeln!(self.asm, "    ld a, 1").unwrap();
                } else {
                    writeln!(self.asm, "    xor a").unwrap();
                }
                writeln!(self.asm, "    ld (0x{da:04X}), a").unwrap();
            }
            IrInst::LoadLocal { dst, slot, ty } => {
                if ix_param_slot == Some(*slot) {
                    self.ix_addrs.insert(*dst, 0);
                    return Ok(());
                }
                if reg_c_slot == Some(*slot) {
                    if can_alias_c_load(self.program, self.func, bb_idx, inst_idx, *dst, *slot) {
                        self.val_in_c.insert(*dst);
                        return Ok(());
                    }
                    let da = self.val_addr(*dst);
                    writeln!(self.asm, "    ld a, c").unwrap();
                    writeln!(self.asm, "    ld (0x{da:04X}), a").unwrap();
                    return Ok(());
                }
                let sa = self.layout.slot_addrs[&(self.func.id, *slot)];
                if can_alias_local_load(self.func, bb_idx, inst_idx, *dst, *slot) {
                    self.val_aliases.insert(*dst, sa);
                    return Ok(());
                }
                let da = self.val_addr(*dst);
                match self.program.type_size(ty) {
                    1 => {
                        writeln!(self.asm, "    ld a, (0x{sa:04X})").unwrap();
                        writeln!(self.asm, "    ld (0x{da:04X}), a").unwrap();
                    }
                    2 => {
                        writeln!(self.asm, "    ld hl, (0x{sa:04X})").unwrap();
                        writeln!(self.asm, "    ld (0x{da:04X}), hl").unwrap();
                    }
                    _ => {}
                }
            }
            IrInst::StoreLocal { slot, val, ty } => {
                if reg_c_slot == Some(*slot) {
                    if let Some(&c) = self.const_vals.get(val) {
                        writeln!(self.asm, "    ld c, 0x{:02X}", (c & 0xFF) as u8).unwrap();
                    } else if self.val_in_c.contains(val) {
                        // no-op: already in C!
                    } else if let Some(&(disp, _)) = self.ix_val_loads.get(val) {
                        writeln!(self.asm, "    ld c, (ix + {disp})").unwrap();
                    } else {
                        self.emit_load_a_from(*val);
                        writeln!(self.asm, "    ld c, a").unwrap();
                    }
                    return Ok(());
                }
                let sa = self.layout.slot_addrs[&(self.func.id, *slot)];
                let sz = self.program.type_size(ty);
                if let Some(&c) = self.const_vals.get(val) {
                    match sz {
                        1 => {
                            let b = (c & 0xFF) as u8;
                            if b == 0 {
                                writeln!(self.asm, "    xor a").unwrap();
                            } else {
                                writeln!(self.asm, "    ld a, 0x{b:02X}").unwrap();
                            }
                            writeln!(self.asm, "    ld (0x{sa:04X}), a").unwrap();
                        }
                        2 => {
                            writeln!(self.asm, "    ld hl, 0x{:04X}", c as u16).unwrap();
                            writeln!(self.asm, "    ld (0x{sa:04X}), hl").unwrap();
                        }
                        _ => {}
                    }
                } else {
                    match sz {
                        1 => {
                            self.emit_load_a_from(*val);
                            writeln!(self.asm, "    ld (0x{sa:04X}), a").unwrap();
                        }
                        2 => {
                            self.emit_load_hl_from(*val);
                            writeln!(self.asm, "    ld (0x{sa:04X}), hl").unwrap();
                        }
                        _ => {}
                    }
                }
            }
            IrInst::AddrLocal { dst, slot } => {
                if ix_param_slot == Some(*slot) {
                    self.ix_param_addr_vals.insert(*dst);
                    return Ok(());
                }
                let sa = self.layout.slot_addrs[&(self.func.id, *slot)];
                self.const_addrs.insert(*dst, sa);
                if is_const_addr_always_folded(self.func, *dst, self.fused_skip_insts) {
                    return Ok(());
                }
                let da = self.val_addr(*dst);
                writeln!(self.asm, "    ld hl, 0x{sa:04X}").unwrap();
                writeln!(self.asm, "    ld (0x{da:04X}), hl").unwrap();
            }
            IrInst::AddrGlobal { dst, global } => {
                let g = &self.program.globals[*global as usize];
                let da = self.val_addr(*dst);
                if g.in_rom {
                    self.rom_addrs.insert(*dst, (g.name.clone(), 0));
                    if is_const_addr_always_folded(self.func, *dst, self.fused_skip_insts) {
                        return Ok(());
                    }
                    writeln!(self.asm, "    ld hl, {}", g.name).unwrap();
                    writeln!(self.asm, "    ld (0x{da:04X}), hl").unwrap();
                } else {
                    let ga = self.layout.global_addrs[global];
                    self.const_addrs.insert(*dst, ga);
                    if is_const_addr_always_folded(self.func, *dst, self.fused_skip_insts) {
                        return Ok(());
                    }
                    writeln!(self.asm, "    ld hl, 0x{ga:04X}").unwrap();
                    writeln!(self.asm, "    ld (0x{da:04X}), hl").unwrap();
                }
            }
            IrInst::FieldAddr {
                dst,
                base,
                byte_offset,
            } => {
                let da = self.val_addr(*dst);
                if let Some(&disp) = self.ix_addrs.get(base)
                    && let Some(new_disp) = (disp as i32)
                        .checked_add(*byte_offset as i32)
                        .filter(|&d| (-128..=125).contains(&d))
                {
                    self.ix_addrs.insert(*dst, new_disp as i8);
                    return Ok(());
                }
                if let Some(&baddr) = self.const_addrs.get(base) {
                    let folded = baddr.wrapping_add(*byte_offset);
                    self.const_addrs.insert(*dst, folded);
                    if is_const_addr_always_folded(self.func, *dst, self.fused_skip_insts) {
                        return Ok(());
                    }
                    writeln!(self.asm, "    ld hl, 0x{folded:04X}").unwrap();
                    writeln!(self.asm, "    ld (0x{da:04X}), hl").unwrap();
                } else if let Some((sym, off)) = self.rom_addrs.get(base).cloned() {
                    let new_off = off.wrapping_add(*byte_offset);
                    self.rom_addrs.insert(*dst, (sym.clone(), new_off));
                    if is_const_addr_always_folded(self.func, *dst, self.fused_skip_insts) {
                        return Ok(());
                    }
                    if new_off == 0 {
                        writeln!(self.asm, "    ld hl, {sym}").unwrap();
                    } else {
                        writeln!(self.asm, "    ld hl, {sym} + {new_off}").unwrap();
                    }
                    writeln!(self.asm, "    ld (0x{da:04X}), hl").unwrap();
                } else {
                    let ba = self.val_loc(*base);
                    writeln!(self.asm, "    ld hl, (0x{ba:04X})").unwrap();
                    if (1..=3).contains(byte_offset) {
                        for _ in 0..*byte_offset {
                            writeln!(self.asm, "    inc hl").unwrap();
                        }
                    } else if *byte_offset > 0 {
                        writeln!(self.asm, "    ld de, 0x{byte_offset:04X}").unwrap();
                        writeln!(self.asm, "    add hl, de").unwrap();
                    }
                    writeln!(self.asm, "    ld (0x{da:04X}), hl").unwrap();
                }
            }
            IrInst::IndexAddr {
                dst,
                base,
                index,
                elem_size,
                len,
                site_id,
            } => {
                let ba = self.val_loc(*base);
                let ia = self.val_loc(*index);
                let da = self.val_addr(*dst);
                let idx_size = val_sizes.get(index).copied().unwrap_or(1);
                // Constant-index folding: `base[CONST]` -> direct `const_addrs` / `rom_addrs` / `ix_addrs`!
                if let Some(&c_idx) = self.const_vals.get(index)
                    && (!self.opts.check_bounds
                        || *len == 0
                        || (c_idx >= 0 && (c_idx as u16) < *len))
                    && let Some(byte_off) =
                        u16::try_from(c_idx).ok().and_then(|i| i.checked_mul(*elem_size))
                {
                    if let Some(&disp) = self.ix_addrs.get(base)
                        && let Some(new_disp) = (disp as i32)
                            .checked_add(byte_off as i32)
                            .filter(|&d| (-128..=125).contains(&d))
                    {
                        let all_ix_folded = self.func.blocks.iter().all(|b| {
                            b.insts.iter().all(|u| {
                                !u.uses_val(*dst)
                                    || matches!(u, IrInst::FieldAddr { base: ub, .. } if *ub == *dst)
                                    || matches!(u, IrInst::IndexAddr { base: ub, index: ui, .. } if *ub == *dst && *ui != *dst)
                                    || matches!(u, IrInst::LoadPtr { addr: ua, .. } if *ua == *dst)
                                    || matches!(u, IrInst::StorePtr { addr: ua, val: uv, .. } if *ua == *dst && *uv != *dst)
                            }) && !matches!(
                                b.term,
                                Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } if v == *dst
                            )
                        });
                        if all_ix_folded {
                            self.ix_addrs.insert(*dst, new_disp as i8);
                            return Ok(());
                        }
                    }
                    if let Some(&baddr) = self.const_addrs.get(base) {
                        let folded = baddr.wrapping_add(byte_off);
                        self.const_addrs.insert(*dst, folded);
                        if is_const_addr_always_folded(self.func, *dst, self.fused_skip_insts) {
                            return Ok(());
                        }
                        writeln!(self.asm, "    ld hl, 0x{folded:04X}").unwrap();
                        writeln!(self.asm, "    ld (0x{da:04X}), hl").unwrap();
                        return Ok(());
                    }
                    if let Some((sym, off)) = self.rom_addrs.get(base).cloned() {
                        let new_off = off.wrapping_add(byte_off);
                        self.rom_addrs.insert(*dst, (sym.clone(), new_off));
                        if is_const_addr_always_folded(self.func, *dst, self.fused_skip_insts) {
                            return Ok(());
                        }
                        if new_off == 0 {
                            writeln!(self.asm, "    ld hl, {sym}").unwrap();
                        } else {
                            writeln!(self.asm, "    ld hl, {sym} + {new_off}").unwrap();
                        }
                        writeln!(self.asm, "    ld (0x{da:04X}), hl").unwrap();
                        return Ok(());
                    }
                }
                // Same-BB `IndexAddr` CSE: reuse already-computed `base + index * elem_size`!
                let cse_base_key = if let Some(&disp) = self.ix_addrs.get(base) {
                    Some(format!("ix:{disp}"))
                } else if let Some(&baddr) = self.const_addrs.get(base) {
                    Some(format!("ram:{baddr:04X}"))
                } else if let Some((sym, off)) = self.rom_addrs.get(base) {
                    Some(format!("rom:{sym}:{off}"))
                } else {
                    None
                };
                let idx_key = if idx_size == 1 && self.val_in_c.contains(index) {
                    IndexKey::RegC
                } else {
                    IndexKey::Mem(ia, idx_size.clamp(1, 2))
                };
                if let Some(ref bkey) = cse_base_key
                    && let Some(&prev_da) =
                        self.index_cse.get(&(bkey.clone(), idx_key.clone(), *elem_size))
                {
                    writeln!(self.asm, "    ld hl, (0x{prev_da:04X})").unwrap();
                    writeln!(self.asm, "    ld (0x{da:04X}), hl").unwrap();
                    return Ok(());
                }
                let use_mul_a = idx_size == 1
                    && !(self.opts.check_bounds && *len > 0)
                    && self.shared_muls.contains(elem_size);
                if use_mul_a {
                    if self.val_in_c.contains(index) {
                        writeln!(self.asm, "    ld a, c").unwrap();
                    } else {
                        writeln!(self.asm, "    ld a, (0x{ia:04X})").unwrap();
                    }
                    writeln!(self.asm, "    call __aun_mul_a_{elem_size}").unwrap();
                } else {
                    if idx_size == 1 {
                        if self.val_in_c.contains(index) {
                            writeln!(self.asm, "    ld l, c").unwrap();
                        } else {
                            writeln!(self.asm, "    ld a, (0x{ia:04X})").unwrap();
                            writeln!(self.asm, "    ld l, a").unwrap();
                        }
                        writeln!(self.asm, "    ld h, 0").unwrap();
                    } else {
                        writeln!(self.asm, "    ld hl, (0x{ia:04X})").unwrap();
                    }

                    // Bounds check: index < len (eliminated in Release!)
                    if self.opts.check_bounds && *len > 0 {
                        *self.needs_panic_stub = true;
                        let ok_lbl = self.fresh_label("idx_ok");
                        writeln!(self.asm, "    push hl").unwrap();
                        writeln!(self.asm, "    ld de, 0x{len:04X}").unwrap();
                        writeln!(self.asm, "    or a").unwrap();
                        writeln!(self.asm, "    sbc hl, de").unwrap();
                        writeln!(self.asm, "    pop hl").unwrap();
                        writeln!(self.asm, "    jr c, {ok_lbl}").unwrap();
                        writeln!(self.asm, "    ld a, {}", PanicCode::Bounds as u8).unwrap();
                        writeln!(self.asm, "    ld hl, 0x{site_id:04X}").unwrap();
                        writeln!(self.asm, "    call __aun_panic").unwrap();
                        writeln!(self.asm, "{ok_lbl}:").unwrap();
                    }

                    // Multiply HL by constant elem_size (outline repeated non-pow2 sizes into `__aun_mul_hl_<k>`)
                    if self.shared_muls.contains(elem_size) {
                        writeln!(self.asm, "    call __aun_mul_hl_{elem_size}").unwrap();
                    } else {
                        emit_mul_hl_const(*elem_size, self.asm);
                    }
                }

                // Add base address
                if let Some(&disp) = self.ix_addrs.get(base) {
                    writeln!(self.asm, "    push ix").unwrap();
                    writeln!(self.asm, "    pop de").unwrap();
                    writeln!(self.asm, "    add hl, de").unwrap();
                    if disp != 0 {
                        writeln!(self.asm, "    ld de, 0x{:04X}", (disp as i16) as u16).unwrap();
                        writeln!(self.asm, "    add hl, de").unwrap();
                    }
                } else if let Some(&baddr) = self.const_addrs.get(base) {
                    writeln!(self.asm, "    ld de, 0x{baddr:04X}").unwrap();
                    writeln!(self.asm, "    add hl, de").unwrap();
                } else if let Some((sym, off)) = self.rom_addrs.get(base) {
                    if *off == 0 {
                        writeln!(self.asm, "    ld de, {sym}").unwrap();
                    } else {
                        writeln!(self.asm, "    ld de, {sym} + {off}").unwrap();
                    }
                    writeln!(self.asm, "    add hl, de").unwrap();
                } else {
                    writeln!(self.asm, "    ld de, (0x{ba:04X})").unwrap();
                    writeln!(self.asm, "    add hl, de").unwrap();
                }
                writeln!(self.asm, "    ld (0x{da:04X}), hl").unwrap();
                if let Some(bkey) = cse_base_key {
                    let no_self_overlap = match idx_key {
                        IndexKey::RegC => true,
                        IndexKey::Mem(ia_m, isz) => !ranges_overlap(da, 2, ia_m, isz),
                    };
                    if no_self_overlap {
                        self.index_cse.insert((bkey, idx_key, *elem_size), da);
                    }
                }
            }
            IrInst::LoadPtr { dst, addr, ty } => {
                let da = self.val_addr(*dst);
                // If reading the IX-promoted param pointer itself, `dst` IS `IX + 0`!
                if self.ix_param_addr_vals.contains(addr) {
                    self.ix_addrs.insert(*dst, 0);
                    return Ok(());
                }
                if let Some(&disp) = self.ix_addrs.get(addr) {
                    let sz = self.program.type_size(ty);
                    if matches!(sz, 1 | 2)
                        && can_alias_ix_load(
                            self.program,
                            self.func,
                            bb_idx,
                            inst_idx,
                            *dst,
                            fused_cmp_dsts,
                        )
                    {
                        self.ix_val_loads.insert(*dst, (disp, sz));
                        if sz == 2 && self.block_nonneg_ix == Some(disp) {
                            self.nonneg_vals.insert(*dst);
                        }
                        return Ok(());
                    }
                    match sz {
                        1 => {
                            writeln!(self.asm, "    ld a, (ix + {disp})").unwrap();
                            writeln!(self.asm, "    ld (0x{da:04X}), a").unwrap();
                        }
                        2 => {
                            writeln!(self.asm, "    ld l, (ix + {disp})").unwrap();
                            writeln!(self.asm, "    ld h, (ix + {})", disp + 1).unwrap();
                            writeln!(self.asm, "    ld (0x{da:04X}), hl").unwrap();
                            if self.block_nonneg_ix == Some(disp) {
                                self.nonneg_vals.insert(*dst);
                            }
                        }
                        _ => {}
                    }
                    return Ok(());
                }
                if let Some(&fixed_addr) = self.const_addrs.get(addr) {
                    match self.program.type_size(ty) {
                        1 => {
                            writeln!(self.asm, "    ld a, (0x{fixed_addr:04X})").unwrap();
                            writeln!(self.asm, "    ld (0x{da:04X}), a").unwrap();
                        }
                        2 => {
                            writeln!(self.asm, "    ld hl, (0x{fixed_addr:04X})").unwrap();
                            writeln!(self.asm, "    ld (0x{da:04X}), hl").unwrap();
                        }
                        _ => {}
                    }
                } else if let Some((sym, off)) = self.rom_addrs.get(addr) {
                    if *off == 0 {
                        writeln!(self.asm, "    ld hl, {sym}").unwrap();
                    } else {
                        writeln!(self.asm, "    ld hl, {sym} + {off}").unwrap();
                    }
                    match self.program.type_size(ty) {
                        1 => {
                            writeln!(self.asm, "    ld a, (hl)").unwrap();
                            writeln!(self.asm, "    ld (0x{da:04X}), a").unwrap();
                        }
                        2 => {
                            writeln!(self.asm, "    ld e, (hl)").unwrap();
                            writeln!(self.asm, "    inc hl").unwrap();
                            writeln!(self.asm, "    ld d, (hl)").unwrap();
                            writeln!(self.asm, "    ex de, hl").unwrap();
                            writeln!(self.asm, "    ld (0x{da:04X}), hl").unwrap();
                        }
                        _ => {}
                    }
                } else {
                    self.emit_load_hl_from(*addr);
                    match self.program.type_size(ty) {
                        1 => {
                            writeln!(self.asm, "    ld a, (hl)").unwrap();
                            writeln!(self.asm, "    ld (0x{da:04X}), a").unwrap();
                        }
                        2 => {
                            writeln!(self.asm, "    ld e, (hl)").unwrap();
                            writeln!(self.asm, "    inc hl").unwrap();
                            writeln!(self.asm, "    ld d, (hl)").unwrap();
                            writeln!(self.asm, "    ex de, hl").unwrap();
                            writeln!(self.asm, "    ld (0x{da:04X}), hl").unwrap();
                        }
                        _ => {}
                    }
                }
            }
            IrInst::StorePtr { addr, val, ty } => {
                let sz = self.program.type_size(ty);
                if let Some(&disp) = self.ix_addrs.get(addr) {
                    if let Some(&c) = self.const_vals.get(val) {
                        match sz {
                            1 => {
                                writeln!(
                                    self.asm,
                                    "    ld (ix + {disp}), 0x{:02X}",
                                    (c & 0xFF) as u8
                                )
                                .unwrap();
                            }
                            2 => {
                                let [lo, hi] = (c as u16).to_le_bytes();
                                writeln!(self.asm, "    ld (ix + {disp}), 0x{lo:02X}").unwrap();
                                writeln!(self.asm, "    ld (ix + {}), 0x{hi:02X}", disp + 1)
                                    .unwrap();
                            }
                            _ => {}
                        }
                    } else {
                        match sz {
                            1 => {
                                self.emit_load_a_from(*val);
                                writeln!(self.asm, "    ld (ix + {disp}), a").unwrap();
                            }
                            2 => {
                                self.emit_load_hl_from(*val);
                                writeln!(self.asm, "    ld (ix + {disp}), l").unwrap();
                                writeln!(self.asm, "    ld (ix + {}), h", disp + 1).unwrap();
                            }
                            _ => {}
                        }
                    }
                    return Ok(());
                }
                if let Some(&fixed_addr) = self.const_addrs.get(addr) {
                    if let Some(&c) = self.const_vals.get(val) {
                        match sz {
                            1 => {
                                let b = (c & 0xFF) as u8;
                                if b == 0 {
                                    writeln!(self.asm, "    xor a").unwrap();
                                } else {
                                    writeln!(self.asm, "    ld a, 0x{b:02X}").unwrap();
                                }
                                writeln!(self.asm, "    ld (0x{fixed_addr:04X}), a").unwrap();
                            }
                            2 => {
                                writeln!(self.asm, "    ld hl, 0x{:04X}", c as u16).unwrap();
                                writeln!(self.asm, "    ld (0x{fixed_addr:04X}), hl").unwrap();
                            }
                            _ => {}
                        }
                    } else {
                        match sz {
                            1 => {
                                self.emit_load_a_from(*val);
                                writeln!(self.asm, "    ld (0x{fixed_addr:04X}), a").unwrap();
                            }
                            2 => {
                                self.emit_load_hl_from(*val);
                                writeln!(self.asm, "    ld (0x{fixed_addr:04X}), hl").unwrap();
                            }
                            _ => {}
                        }
                    }
                } else {
                    let aa = self.val_loc(*addr);
                    match sz {
                        1 => {
                            writeln!(self.asm, "    ld hl, (0x{aa:04X})").unwrap();
                            if let Some(&c) = self.const_vals.get(val) {
                                writeln!(self.asm, "    ld (hl), 0x{:02X}", (c & 0xFF) as u8)
                                    .unwrap();
                            } else {
                                self.emit_load_a_from(*val);
                                writeln!(self.asm, "    ld (hl), a").unwrap();
                            }
                        }
                        2 => {
                            self.emit_load_de_from(*val);
                            writeln!(self.asm, "    ld hl, (0x{aa:04X})").unwrap();
                            writeln!(self.asm, "    ld (hl), e").unwrap();
                            writeln!(self.asm, "    inc hl").unwrap();
                            writeln!(self.asm, "    ld (hl), d").unwrap();
                        }
                        _ => {}
                    }
                }
            }
            IrInst::UnOp {
                dst, op, ty, val, ..
            } => {
                let va = self.val_loc(*val);
                let da = self.val_addr(*dst);
                match (op, ty, self.program.type_size(ty)) {
                    (IrUnOp::Not, IrType::Bool, _) => {
                        writeln!(self.asm, "    ld a, (0x{va:04X})").unwrap();
                        writeln!(self.asm, "    xor 1").unwrap();
                        writeln!(self.asm, "    ld (0x{da:04X}), a").unwrap();
                    }
                    (IrUnOp::Not, _, 1) => {
                        writeln!(self.asm, "    ld a, (0x{va:04X})").unwrap();
                        writeln!(self.asm, "    cpl").unwrap();
                        writeln!(self.asm, "    ld (0x{da:04X}), a").unwrap();
                    }
                    (IrUnOp::Not, _, 2) => {
                        writeln!(self.asm, "    ld hl, (0x{va:04X})").unwrap();
                        writeln!(self.asm, "    ld a, l").unwrap();
                        writeln!(self.asm, "    cpl").unwrap();
                        writeln!(self.asm, "    ld l, a").unwrap();
                        writeln!(self.asm, "    ld a, h").unwrap();
                        writeln!(self.asm, "    cpl").unwrap();
                        writeln!(self.asm, "    ld h, a").unwrap();
                        writeln!(self.asm, "    ld (0x{da:04X}), hl").unwrap();
                    }
                    (IrUnOp::Neg, _, 1) => {
                        writeln!(self.asm, "    ld a, (0x{va:04X})").unwrap();
                        writeln!(self.asm, "    neg").unwrap();
                        writeln!(self.asm, "    ld (0x{da:04X}), a").unwrap();
                    }
                    (IrUnOp::Neg, _, 2) => {
                        writeln!(self.asm, "    ld hl, (0x{va:04X})").unwrap();
                        writeln!(self.asm, "    xor a").unwrap();
                        writeln!(self.asm, "    sub l").unwrap();
                        writeln!(self.asm, "    ld l, a").unwrap();
                        writeln!(self.asm, "    sbc a, a").unwrap();
                        writeln!(self.asm, "    sub h").unwrap();
                        writeln!(self.asm, "    ld h, a").unwrap();
                        writeln!(self.asm, "    ld (0x{da:04X}), hl").unwrap();
                    }
                    _ => {}
                }
            }
            IrInst::Cast {
                dst,
                val,
                from_ty,
                to_ty,
            } => {
                let da = self.val_addr(*dst);
                if let Some(&src_v) = self.fused_q10_6_u8.get(dst) {
                    // Fused `q10_6::int() as u8`: `(x >> 6) & 0xFF == ((x << 2) >> 8) & 0xFF`
                    // regardless of sign! Only 3 bytes (`add hl, hl; add hl, hl; ld a, h`).
                    if let Some(&c) = self.const_vals.get(&src_v) {
                        let b = ((c >> 6) & 0xFF) as u8;
                        if b == 0 {
                            writeln!(self.asm, "    xor a").unwrap();
                        } else {
                            writeln!(self.asm, "    ld a, 0x{b:02X}").unwrap();
                        }
                    } else {
                        let sa = self.val_loc(src_v);
                        writeln!(self.asm, "    ld hl, (0x{sa:04X})").unwrap();
                        writeln!(self.asm, "    add hl, hl").unwrap();
                        writeln!(self.asm, "    add hl, hl").unwrap();
                        writeln!(self.asm, "    ld a, h").unwrap();
                    }
                    writeln!(self.asm, "    ld (0x{da:04X}), a").unwrap();
                    return Ok(());
                }
                let from_sz = self.program.type_size(from_ty);
                let to_sz = self.program.type_size(to_ty);
                // Zero-cost no-op cast propagation for `(ix + d)` loads (e.g. `c.cur_freq as i16`)!
                if let Some(&(disp, _)) = self.ix_val_loads.get(val)
                    && ((from_sz == 2 && to_sz == 2)
                        || (from_sz == 1
                            && to_sz == 1
                            && (*to_ty != IrType::Bool || *from_ty == IrType::Bool)))
                    && can_alias_ix_load(
                        self.program,
                        self.func,
                        bb_idx,
                        inst_idx,
                        *dst,
                        fused_cmp_dsts,
                    )
                {
                    self.ix_val_loads.insert(*dst, (disp, to_sz));
                    return Ok(());
                }
                match (from_sz, to_sz) {
                    (1, 1) => {
                        self.emit_load_a_from(*val);
                        if *to_ty == IrType::Bool && *from_ty != IrType::Bool {
                            let lbl = self.fresh_label("cast_bool");
                            writeln!(self.asm, "    or a").unwrap();
                            writeln!(self.asm, "    jr z, {lbl}").unwrap();
                            writeln!(self.asm, "    ld a, 1").unwrap();
                            writeln!(self.asm, "{lbl}:").unwrap();
                        }
                        writeln!(self.asm, "    ld (0x{da:04X}), a").unwrap();
                    }
                    (1, 2) => {
                        self.emit_load_a_from(*val);
                        writeln!(self.asm, "    ld l, a").unwrap();
                        if is_signed_ty(from_ty) {
                            writeln!(self.asm, "    add a, a").unwrap();
                            writeln!(self.asm, "    sbc a, a").unwrap();
                            writeln!(self.asm, "    ld h, a").unwrap();
                        } else {
                            writeln!(self.asm, "    ld h, 0").unwrap();
                        }
                        writeln!(self.asm, "    ld (0x{da:04X}), hl").unwrap();
                    }
                    (2, 1) => {
                        if *to_ty == IrType::Bool {
                            let lbl = self.fresh_label("cast_bool");
                            self.emit_load_hl_from(*val);
                            writeln!(self.asm, "    ld a, l").unwrap();
                            writeln!(self.asm, "    or h").unwrap();
                            writeln!(self.asm, "    jr z, {lbl}").unwrap();
                            writeln!(self.asm, "    ld a, 1").unwrap();
                            writeln!(self.asm, "{lbl}:").unwrap();
                            writeln!(self.asm, "    ld (0x{da:04X}), a").unwrap();
                        } else {
                            // Little-endian: low byte is at va / (ix + disp)
                            self.emit_load_a_from(*val);
                            writeln!(self.asm, "    ld (0x{da:04X}), a").unwrap();
                        }
                    }
                    (2, 2) => {
                        self.emit_load_hl_from(*val);
                        writeln!(self.asm, "    ld (0x{da:04X}), hl").unwrap();
                    }
                    _ => {}
                }
            }
            IrInst::FixedInt {
                dst,
                val,
                base,
                frac_bits,
            } => {
                let va = self.val_loc(*val);
                let da = self.val_addr(*dst);
                match (base.byte_size(), *frac_bits) {
                    (2, 8) => {
                        // q8_8::int() -> direct high-byte read (0T shift!), sign/zero extended to 16-bit
                        let va_hi = va.wrapping_add(1);
                        writeln!(self.asm, "    ld a, (0x{va_hi:04X})").unwrap();
                        writeln!(self.asm, "    ld l, a").unwrap();
                        if base.is_signed() {
                            writeln!(self.asm, "    add a, a").unwrap();
                            writeln!(self.asm, "    sbc a, a").unwrap();
                            writeln!(self.asm, "    ld h, a").unwrap();
                        } else {
                            writeln!(self.asm, "    ld h, 0").unwrap();
                        }
                        writeln!(self.asm, "    ld (0x{da:04X}), hl").unwrap();
                    }
                    (2, 6) => {
                        // q10_6::int() -> exact 16-bit shift right by 6 via 2-bit left shift of A:H:L
                        writeln!(self.asm, "    ld hl, (0x{va:04X})").unwrap();
                        if base.is_signed() && !self.nonneg_vals.contains(val) {
                            writeln!(self.asm, "    ld a, h").unwrap();
                            writeln!(self.asm, "    add a, a").unwrap();
                            writeln!(self.asm, "    sbc a, a").unwrap();
                        } else {
                            writeln!(self.asm, "    xor a").unwrap();
                        }
                        writeln!(self.asm, "    add hl, hl").unwrap();
                        writeln!(self.asm, "    rla").unwrap();
                        writeln!(self.asm, "    add hl, hl").unwrap();
                        writeln!(self.asm, "    rla").unwrap();
                        writeln!(self.asm, "    ld l, h").unwrap();
                        writeln!(self.asm, "    ld h, a").unwrap();
                        writeln!(self.asm, "    ld (0x{da:04X}), hl").unwrap();
                    }
                    (2, 0) => {
                        writeln!(self.asm, "    ld hl, (0x{va:04X})").unwrap();
                        writeln!(self.asm, "    ld (0x{da:04X}), hl").unwrap();
                    }
                    (2, fb) => {
                        let lbl = self.fresh_label("fxint");
                        writeln!(self.asm, "    ld hl, (0x{va:04X})").unwrap();
                        writeln!(self.asm, "    ld b, {fb}").unwrap();
                        writeln!(self.asm, "{lbl}:").unwrap();
                        if base.is_signed() {
                            writeln!(self.asm, "    sra h").unwrap();
                        } else {
                            writeln!(self.asm, "    srl h").unwrap();
                        }
                        writeln!(self.asm, "    rr l").unwrap();
                        writeln!(self.asm, "    djnz {lbl}").unwrap();
                        writeln!(self.asm, "    ld (0x{da:04X}), hl").unwrap();
                    }
                    (1, 0) => {
                        writeln!(self.asm, "    ld a, (0x{va:04X})").unwrap();
                        writeln!(self.asm, "    ld (0x{da:04X}), a").unwrap();
                    }
                    (1, fb) => {
                        let lbl = self.fresh_label("fxint");
                        writeln!(self.asm, "    ld a, (0x{va:04X})").unwrap();
                        writeln!(self.asm, "    ld b, {fb}").unwrap();
                        writeln!(self.asm, "{lbl}:").unwrap();
                        if base.is_signed() {
                            writeln!(self.asm, "    sra a").unwrap();
                        } else {
                            writeln!(self.asm, "    srl a").unwrap();
                        }
                        writeln!(self.asm, "    djnz {lbl}").unwrap();
                        writeln!(self.asm, "    ld (0x{da:04X}), a").unwrap();
                    }
                    _ => {}
                }
            }
            IrInst::BinOp {
                dst,
                op,
                mode,
                ty,
                lhs,
                rhs,
                ..
            } => {
                self.emit_binop(*dst, *op, *mode, ty, *lhs, *rhs);
            }
            IrInst::PortOut { port, val } => {
                if let Some(&p) = self.const_vals.get(port) {
                    if let Some(&v) = self.const_vals.get(val) {
                        let b = (v & 0xFF) as u8;
                        if b == 0 {
                            writeln!(self.asm, "    xor a").unwrap();
                        } else {
                            writeln!(self.asm, "    ld a, 0x{b:02X}").unwrap();
                        }
                    } else {
                        let va = self.val_loc(*val);
                        writeln!(self.asm, "    ld a, (0x{va:04X})").unwrap();
                    }
                    writeln!(self.asm, "    out (0x{:02X}), a", (p & 0xFF) as u8).unwrap();
                } else {
                    let pa = self.val_loc(*port);
                    let va = self.val_loc(*val);
                    writeln!(self.asm, "    ld a, (0x{pa:04X})").unwrap();
                    writeln!(self.asm, "    ld c, a").unwrap();
                    writeln!(self.asm, "    ld a, (0x{va:04X})").unwrap();
                    writeln!(self.asm, "    out (c), a").unwrap();
                }
            }
            IrInst::PortIn { dst, port } => {
                let da = self.val_addr(*dst);
                if let Some(&p) = self.const_vals.get(port) {
                    writeln!(self.asm, "    in a, (0x{:02X})", (p & 0xFF) as u8).unwrap();
                } else {
                    let pa = self.val_loc(*port);
                    writeln!(self.asm, "    ld a, (0x{pa:04X})").unwrap();
                    writeln!(self.asm, "    ld c, a").unwrap();
                    writeln!(self.asm, "    in a, (c)").unwrap();
                }
                writeln!(self.asm, "    ld (0x{da:04X}), a").unwrap();
            }
            IrInst::Call {
                dst,
                func: callee_id,
                args,
            } => {
                let cidx = *callee_id as usize;
                let reg_p1 = self.analysis.fns[cidx].uses_reg_param1;
                let ram_start = if reg_p1 { 2 } else { 1 };
                // Write arguments ram_start..N directly into callee's Uchiguri parameter slots
                for (idx, &arg_val) in args.iter().enumerate().skip(ram_start) {
                    let slot_info = {
                        let callee = &self.program.functions[cidx];
                        callee.locals.get(idx).map(|p| {
                            (
                                self.layout.slot_addrs[&(callee.id, p.id)],
                                self.program.type_size(&p.ty),
                            )
                        })
                    };
                    if let Some((pa, psz)) = slot_info {
                        if self.ix_addrs.get(&arg_val) == Some(&0) {
                            writeln!(self.asm, "    ld (0x{pa:04X}), ix").unwrap();
                            continue;
                        }
                        match psz {
                            1 => {
                                self.emit_load_a_from(arg_val);
                                writeln!(self.asm, "    ld (0x{pa:04X}), a").unwrap();
                            }
                            2 => {
                                self.emit_load_hl_from(arg_val);
                                writeln!(self.asm, "    ld (0x{pa:04X}), hl").unwrap();
                            }
                            _ => {}
                        }
                    }
                }
                // Pass argument 1 in E (1-byte) or DE (2-byte) when eligible (before loading A/HL)
                if reg_p1 && let Some(&a1) = args.get(1) {
                    let p1_sz = self.program.functions[cidx]
                        .locals
                        .get(1)
                        .map(|p| self.program.type_size(&p.ty));
                    if let Some(sz) = p1_sz {
                        if self.ix_addrs.get(&a1) == Some(&0) {
                            writeln!(self.asm, "    push ix").unwrap();
                            writeln!(self.asm, "    pop de").unwrap();
                        } else {
                            match sz {
                                1 => {
                                    if let Some(&c) = self.const_vals.get(&a1) {
                                        writeln!(self.asm, "    ld e, 0x{:02X}", (c & 0xFF) as u8)
                                            .unwrap();
                                    } else if self.val_in_c.contains(&a1) {
                                        writeln!(self.asm, "    ld e, c").unwrap();
                                    } else if let Some(&(disp, _)) = self.ix_val_loads.get(&a1) {
                                        writeln!(self.asm, "    ld e, (ix + {disp})").unwrap();
                                    } else {
                                        let aa = self.val_loc(a1);
                                        writeln!(self.asm, "    ld a, (0x{aa:04X})").unwrap();
                                        writeln!(self.asm, "    ld e, a").unwrap();
                                    }
                                }
                                2 => self.emit_load_de_from(a1),
                                _ => {}
                            }
                        }
                    }
                }
                // Pass argument 0 in A (1-byte) or HL (2-byte)
                if let Some(&a0) = args.first() {
                    let p0_sz = self.program.functions[cidx]
                        .locals
                        .first()
                        .map(|p| self.program.type_size(&p.ty));
                    if let Some(sz) = p0_sz {
                        if self.ix_addrs.get(&a0) == Some(&0) {
                            writeln!(self.asm, "    push ix").unwrap();
                            writeln!(self.asm, "    pop hl").unwrap();
                        } else {
                            match sz {
                                1 => self.emit_load_a_from(a0),
                                2 => self.emit_load_hl_from(a0),
                                _ => {}
                            }
                        }
                    }
                }
                let need_save_c = reg_c_slot.is_some()
                    && self
                        .analysis
                        .fn_clobbers_c
                        .get(cidx)
                        .copied()
                        .unwrap_or(true);
                if need_save_c {
                    writeln!(self.asm, "    push bc").unwrap();
                }
                writeln!(
                    self.asm,
                    "    call {}",
                    z80_sym(&self.program.functions[cidx].name)
                )
                .unwrap();
                if need_save_c {
                    writeln!(self.asm, "    pop bc").unwrap();
                }
                if let Some(d) = dst {
                    let da = self.val_addr(*d);
                    let ret_sz = self.program.type_size(&self.program.functions[cidx].ret_ty);
                    match ret_sz {
                        1 => writeln!(self.asm, "    ld (0x{da:04X}), a").unwrap(),
                        2 => writeln!(self.asm, "    ld (0x{da:04X}), hl").unwrap(),
                        _ => {}
                    }
                }
            }
            IrInst::ExternCall {
                dst,
                extern_fn,
                args,
            } => {
                let ef_idx = *extern_fn as usize;
                let param_count = self.program.externs[ef_idx].params.len();
                // Load non-A arguments first so `ld a, (addr); ld r, a` never clobbers an `in A` argument
                let mut deferred_a: Option<ValueId> = None;
                for (arg_val, i) in args.iter().zip(0..param_count) {
                    let (psz, r) = {
                        let (_, pty, reg) = &self.program.externs[ef_idx].params[i];
                        (self.program.type_size(pty), reg.to_ascii_lowercase())
                    };
                    match psz {
                        1 => {
                            if r == "a" {
                                deferred_a = Some(*arg_val);
                            } else if let Some(&c) = self.const_vals.get(arg_val) {
                                writeln!(self.asm, "    ld {r}, 0x{:02X}", (c & 0xFF) as u8)
                                    .unwrap();
                            } else if self.val_in_c.contains(arg_val) {
                                writeln!(self.asm, "    ld {r}, c").unwrap();
                            } else if let Some(&(disp, _)) = self.ix_val_loads.get(arg_val) {
                                writeln!(self.asm, "    ld {r}, (ix + {disp})").unwrap();
                            } else {
                                let aa = self.val_loc(*arg_val);
                                writeln!(self.asm, "    ld a, (0x{aa:04X})").unwrap();
                                writeln!(self.asm, "    ld {r}, a").unwrap();
                            }
                        }
                        2 => {
                            if let Some(&c) = self.const_vals.get(arg_val) {
                                writeln!(self.asm, "    ld {r}, 0x{:04X}", c as u16).unwrap();
                            } else if let Some(&(disp, _)) = self.ix_val_loads.get(arg_val)
                                && r == "hl"
                            {
                                writeln!(self.asm, "    ld l, (ix + {disp})").unwrap();
                                writeln!(self.asm, "    ld h, (ix + {})", disp + 1).unwrap();
                            } else if let Some(&(disp, _)) = self.ix_val_loads.get(arg_val)
                                && r == "de"
                            {
                                writeln!(self.asm, "    ld e, (ix + {disp})").unwrap();
                                writeln!(self.asm, "    ld d, (ix + {})", disp + 1).unwrap();
                            } else {
                                let aa = self.val_loc(*arg_val);
                                writeln!(self.asm, "    ld {r}, (0x{aa:04X})").unwrap();
                            }
                        }
                        _ => {}
                    }
                }
                if let Some(av) = deferred_a {
                    self.emit_load_a_from(av);
                }
                let ef = &self.program.externs[ef_idx];
                let need_save_c = reg_c_slot.is_some() && extern_clobbers_c(ef);
                if need_save_c {
                    writeln!(self.asm, "    push bc").unwrap();
                }
                writeln!(self.asm, "    call {}", ef.name).unwrap();
                if need_save_c {
                    writeln!(self.asm, "    pop bc").unwrap();
                }
                if let (Some(d), Some((rty, rreg))) = (dst, &ef.ret) {
                    let da = self.val_addr(*d);
                    let r = rreg.to_ascii_lowercase();
                    match self.program.type_size(rty) {
                        1 => {
                            if r != "a" {
                                writeln!(self.asm, "    ld a, {r}").unwrap();
                            }
                            writeln!(self.asm, "    ld (0x{da:04X}), a").unwrap();
                        }
                        2 => writeln!(self.asm, "    ld (0x{da:04X}), {r}").unwrap(),
                        _ => {}
                    }
                }
            }
            IrInst::Assert { cond, site_id } => {
                if !self.opts.release {
                    *self.needs_panic_stub = true;
                    let ca = self.val_loc(*cond);
                    let ok_lbl = self.fresh_label("assert_ok");
                    writeln!(self.asm, "    ld a, (0x{ca:04X})").unwrap();
                    writeln!(self.asm, "    or a").unwrap();
                    writeln!(self.asm, "    jr nz, {ok_lbl}").unwrap();
                    writeln!(self.asm, "    ld a, {}", PanicCode::AssertionFailed as u8).unwrap();
                    writeln!(self.asm, "    ld hl, 0x{site_id:04X}").unwrap();
                    writeln!(self.asm, "    call __aun_panic").unwrap();
                    writeln!(self.asm, "{ok_lbl}:").unwrap();
                }
            }
            IrInst::Panic { code, site_id } => {
                *self.needs_panic_stub = true;
                writeln!(self.asm, "    ld a, {}", *code as u8).unwrap();
                writeln!(self.asm, "    ld hl, 0x{site_id:04X}").unwrap();
                writeln!(self.asm, "    call __aun_panic").unwrap();
            }
            IrInst::StateTag {
                dst,
                machine_addr,
                machine_id,
            } => {
                let ma = self.val_loc(*machine_addr);
                let da = self.val_addr(*dst);
                let mdef = &self.program.machines[*machine_id as usize];
                writeln!(self.asm, "    ld hl, (0x{ma:04X})").unwrap();
                if mdef.tag_offset > 0 {
                    writeln!(self.asm, "    ld de, 0x{:04X}", mdef.tag_offset).unwrap();
                    writeln!(self.asm, "    add hl, de").unwrap();
                }
                writeln!(self.asm, "    ld a, (hl)").unwrap();
                writeln!(self.asm, "    ld (0x{da:04X}), a").unwrap();
            }
            IrInst::StateDispatch {
                dst,
                machine_addr,
                machine_id,
                method,
                args,
            } => {
                let mdef = &self.program.machines[*machine_id as usize];
                // Copy args 1..N into every implementing state's Uchiguri parameter slots
                for st in &mdef.states {
                    if let Some(&fid) = st.methods.get(method) {
                        let callee = &self.program.functions[fid as usize];
                        for (idx, &arg_val) in args.iter().enumerate() {
                            if let Some(param_slot) = callee.locals.get(idx + 1) {
                                let aa = self.val_loc(arg_val);
                                let pa = self.layout.slot_addrs[&(callee.id, param_slot.id)];
                                match self.program.type_size(&param_slot.ty) {
                                    1 => {
                                        writeln!(self.asm, "    ld a, (0x{aa:04X})").unwrap();
                                        writeln!(self.asm, "    ld (0x{pa:04X}), a").unwrap();
                                    }
                                    2 => {
                                        writeln!(self.asm, "    ld hl, (0x{aa:04X})").unwrap();
                                        writeln!(self.asm, "    ld (0x{pa:04X}), hl").unwrap();
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                let ma = self.val_loc(*machine_addr);
                writeln!(self.asm, "    ld hl, (0x{ma:04X})").unwrap();
                writeln!(self.asm, "    call {}_disp_{method}", mdef.name).unwrap();
                if let Some(d) = dst {
                    let da = self.val_addr(*d);
                    writeln!(self.asm, "    ld (0x{da:04X}), hl").unwrap();
                }
            }
            IrInst::StateTransition {
                machine_addr,
                machine_id,
                from_state,
                to_state,
                field_vals,
                site_id,
            } => {
                let ma = self.val_loc(*machine_addr);
                let ok_lbl = if from_state.is_none() && self.opts.check_transitions {
                    *self.needs_panic_stub = true;
                    Some(self.fresh_label("tr_ok"))
                } else {
                    None
                };
                let mdef = &self.program.machines[*machine_id as usize];

                // 1. Dynamic transition verification when `from_state` is unknown (eliminated in Release!)
                if let Some(ok_lbl) = ok_lbl {
                    writeln!(self.asm, "    ld hl, (0x{ma:04X})").unwrap();
                    if mdef.tag_offset > 0 {
                        writeln!(self.asm, "    ld de, 0x{:04X}", mdef.tag_offset).unwrap();
                        writeln!(self.asm, "    add hl, de").unwrap();
                    }
                    writeln!(self.asm, "    ld a, (hl)").unwrap();
                    writeln!(self.asm, "    add a, a").unwrap();
                    writeln!(self.asm, "    ld e, a").unwrap();
                    writeln!(self.asm, "    ld d, 0").unwrap();
                    writeln!(self.asm, "    ld hl, {}_tmask", mdef.name).unwrap();
                    writeln!(self.asm, "    add hl, de").unwrap();
                    let bit = *to_state;
                    if bit < 8 {
                        writeln!(self.asm, "    ld a, (hl)").unwrap();
                        writeln!(self.asm, "    bit {bit}, a").unwrap();
                    } else {
                        writeln!(self.asm, "    inc hl").unwrap();
                        writeln!(self.asm, "    ld a, (hl)").unwrap();
                        writeln!(self.asm, "    bit {}, a", bit - 8).unwrap();
                    }
                    writeln!(self.asm, "    jr nz, {ok_lbl}").unwrap();
                    writeln!(self.asm, "    ld a, {}", PanicCode::InvalidTransition as u8)
                        .unwrap();
                    writeln!(self.asm, "    ld hl, 0x{site_id:04X}").unwrap();
                    writeln!(self.asm, "    call __aun_panic").unwrap();
                    writeln!(self.asm, "{ok_lbl}:").unwrap();
                }

                // 2. Call current state's on_exit
                if let Some(from) = from_state {
                    if let Some(exit_fn) = mdef.states[*from as usize].on_exit {
                        let ename = z80_sym(&self.program.functions[exit_fn as usize].name);
                        writeln!(self.asm, "    ld hl, (0x{ma:04X})").unwrap();
                        writeln!(self.asm, "    call {ename}").unwrap();
                    }
                } else if mdef.states.iter().any(|s| s.on_exit.is_some()) {
                    writeln!(self.asm, "    ld hl, (0x{ma:04X})").unwrap();
                    writeln!(self.asm, "    call {}_disp_on_exit", mdef.name).unwrap();
                }

                // 3. Write pre-evaluated payload fields into overlaid RAM region
                for (rel_off, vid, fty) in field_vals {
                    let va = self.val_loc(*vid);
                    let total_off = mdef.payload_offset + *rel_off;
                    writeln!(self.asm, "    ld hl, (0x{ma:04X})").unwrap();
                    if total_off > 0 {
                        writeln!(self.asm, "    ld de, 0x{total_off:04X}").unwrap();
                        writeln!(self.asm, "    add hl, de").unwrap();
                    }
                    match self.program.type_size(fty) {
                        1 => {
                            writeln!(self.asm, "    ld a, (0x{va:04X})").unwrap();
                            writeln!(self.asm, "    ld (hl), a").unwrap();
                        }
                        2 => {
                            writeln!(self.asm, "    ld de, (0x{va:04X})").unwrap();
                            writeln!(self.asm, "    ld (hl), e").unwrap();
                            writeln!(self.asm, "    inc hl").unwrap();
                            writeln!(self.asm, "    ld (hl), d").unwrap();
                        }
                        _ => {}
                    }
                }

                // 4. Write new state tag
                writeln!(self.asm, "    ld hl, (0x{ma:04X})").unwrap();
                if mdef.tag_offset > 0 {
                    writeln!(self.asm, "    ld de, 0x{:04X}", mdef.tag_offset).unwrap();
                    writeln!(self.asm, "    add hl, de").unwrap();
                }
                writeln!(self.asm, "    ld (hl), {}", *to_state).unwrap();
                writeln!(self.asm, "    inc hl").unwrap();
                writeln!(self.asm, "    ld (hl), 0").unwrap();

                // 4b. If target state lives in an #[overlay(Slot)], invoke its `load_fn` from resident code
                if let Some(ref slot_name) = mdef.states[*to_state as usize].overlay_slot
                    && let Some(target) = &self.program.target
                    && let Some(slot_def) =
                        target.overlay_slots.iter().find(|s| &s.name == slot_name)
                    && let Some(ref load_fn) = slot_def.load_fn
                {
                    let has_load_sym = self
                        .program
                        .functions
                        .iter()
                        .any(|f| !f.is_test && &f.name == load_fn)
                        || self.program.externs.iter().any(|e| &e.name == load_fn);
                    if has_load_sym {
                        writeln!(self.asm, "    ld a, {}", *to_state).unwrap();
                        writeln!(self.asm, "    call {load_fn}").unwrap();
                    }
                }

                // 5. If #[smc], patch direct JP target in RAM for each dispatched method
                if mdef.smc {
                    let mut methods: BTreeSet<String> = BTreeSet::new();
                    for st in &mdef.states {
                        for m in st.methods.keys() {
                            methods.insert(m.clone());
                        }
                    }
                    for method in methods {
                        let target = mdef.states[*to_state as usize]
                            .methods
                            .get(&method)
                            .map(|&fid| z80_sym(&self.program.functions[fid as usize].name))
                            .unwrap_or_else(|| "__aun_noop_ret".to_string());
                        writeln!(self.asm, "    ld hl, {target}").unwrap();
                        writeln!(self.asm, "    ld ({}_disp_{method}+1), hl", mdef.name).unwrap();
                    }
                }

                // 6. Call target state's on_enter
                if let Some(enter_fn) = mdef.states[*to_state as usize].on_enter {
                    let ename = z80_sym(&self.program.functions[enter_fn as usize].name);
                    writeln!(self.asm, "    ld hl, (0x{ma:04X})").unwrap();
                    writeln!(self.asm, "    call {ename}").unwrap();
                }
            }
            IrInst::PoolAlloc {
                dst,
                pool_addr,
                init_addr,
                elem_size,
                capacity,
            } => {
                let pa = self.val_loc(*pool_addr);
                let ia = self.val_loc(*init_addr);
                let da = self.val_addr(*dst);
                let scan_lbl = self.fresh_label("palloc_scan");
                let bit_lbl = self.fresh_label("palloc_bit");
                let found_lbl = self.fresh_label("palloc_found");
                let done_lbl = self.fresh_label("palloc_done");
                let shift_done = self.fresh_label("palloc_sd");
                let shift_loop = self.fresh_label("palloc_sl");
                let bm_off = elem_size * capacity;

                // Scan slots 0..capacity-1 for first free bit in alive bitmap
                writeln!(self.asm, "    ld c, 0").unwrap(); // C = candidate index
                writeln!(self.asm, "{scan_lbl}:").unwrap();
                writeln!(self.asm, "    ld a, c").unwrap();
                writeln!(self.asm, "    cp {}", *capacity as u8).unwrap();
                writeln!(self.asm, "    jr c, {bit_lbl}").unwrap();
                writeln!(self.asm, "    ld a, 0xFF").unwrap();
                writeln!(self.asm, "    jr {done_lbl}").unwrap();
                writeln!(self.asm, "{bit_lbl}:").unwrap();
                writeln!(self.asm, "    srl a").unwrap();
                writeln!(self.asm, "    srl a").unwrap();
                writeln!(self.asm, "    srl a").unwrap();
                writeln!(self.asm, "    ld e, a").unwrap();
                writeln!(self.asm, "    ld d, 0").unwrap();
                writeln!(self.asm, "    ld hl, (0x{pa:04X})").unwrap();
                writeln!(self.asm, "    add hl, de").unwrap();
                writeln!(self.asm, "    ld de, 0x{bm_off:04X}").unwrap();
                writeln!(self.asm, "    add hl, de").unwrap();
                // Build bitmask for (C & 7) in B
                writeln!(self.asm, "    ld a, c").unwrap();
                writeln!(self.asm, "    and 7").unwrap();
                writeln!(self.asm, "    ld b, 1").unwrap();
                writeln!(self.asm, "    or a").unwrap();
                writeln!(self.asm, "    jr z, {shift_done}").unwrap();
                writeln!(self.asm, "{shift_loop}:").unwrap();
                writeln!(self.asm, "    sla b").unwrap();
                writeln!(self.asm, "    dec a").unwrap();
                writeln!(self.asm, "    jr nz, {shift_loop}").unwrap();
                writeln!(self.asm, "{shift_done}:").unwrap();
                writeln!(self.asm, "    ld a, (hl)").unwrap();
                writeln!(self.asm, "    and b").unwrap();
                writeln!(self.asm, "    jr z, {found_lbl}").unwrap();
                writeln!(self.asm, "    inc c").unwrap();
                writeln!(self.asm, "    jr {scan_lbl}").unwrap();
                writeln!(self.asm, "{found_lbl}:").unwrap();
                // Mark alive bit
                writeln!(self.asm, "    ld a, (hl)").unwrap();
                writeln!(self.asm, "    or b").unwrap();
                writeln!(self.asm, "    ld (hl), a").unwrap();
                // Copy elem_size bytes from init_addr to pool_addr + C * elem_size using LDIR
                if *elem_size > 0 {
                    writeln!(self.asm, "    push bc").unwrap();
                    writeln!(self.asm, "    ld l, c").unwrap();
                    writeln!(self.asm, "    ld h, 0").unwrap();
                    emit_mul_hl_const(*elem_size, self.asm);
                    writeln!(self.asm, "    ld de, (0x{pa:04X})").unwrap();
                    writeln!(self.asm, "    add hl, de").unwrap();
                    writeln!(self.asm, "    ex de, hl").unwrap(); // DE = dst
                    writeln!(self.asm, "    ld hl, (0x{ia:04X})").unwrap(); // HL = src
                    writeln!(self.asm, "    ld bc, 0x{elem_size:04X}").unwrap();
                    writeln!(self.asm, "    ldir").unwrap();
                    writeln!(self.asm, "    pop bc").unwrap();
                }
                writeln!(self.asm, "    ld a, c").unwrap();
                writeln!(self.asm, "{done_lbl}:").unwrap();
                writeln!(self.asm, "    ld (0x{da:04X}), a").unwrap();
            }
            IrInst::PoolFree {
                pool_addr,
                handle,
                elem_size,
                capacity,
                site_id,
            } => {
                let pa = self.val_loc(*pool_addr);
                let ha = self.val_loc(*handle);
                let bm_off = elem_size * capacity;
                let sd = self.fresh_label("pfree_sd");
                let sl = self.fresh_label("pfree_sl");
                let ok_rng = if self.opts.check_bounds {
                    *self.needs_panic_stub = true;
                    Some((
                        self.fresh_label("pfree_rng"),
                        self.fresh_label("pfree_live"),
                    ))
                } else {
                    None
                };
                writeln!(self.asm, "    ld a, (0x{ha:04X})").unwrap();
                if let Some((ref ok_r, _)) = ok_rng {
                    writeln!(self.asm, "    cp {}", *capacity as u8).unwrap();
                    writeln!(self.asm, "    jr c, {ok_r}").unwrap();
                    writeln!(self.asm, "    ld a, {}", PanicCode::InvalidHandle as u8).unwrap();
                    writeln!(self.asm, "    ld hl, 0x{site_id:04X}").unwrap();
                    writeln!(self.asm, "    call __aun_panic").unwrap();
                    writeln!(self.asm, "{ok_r}:").unwrap();
                }
                writeln!(self.asm, "    ld c, a").unwrap();
                writeln!(self.asm, "    srl a").unwrap();
                writeln!(self.asm, "    srl a").unwrap();
                writeln!(self.asm, "    srl a").unwrap();
                writeln!(self.asm, "    ld e, a").unwrap();
                writeln!(self.asm, "    ld d, 0").unwrap();
                if let Some(&paddr) = self.const_addrs.get(pool_addr) {
                    writeln!(self.asm, "    ld hl, 0x{:04X}", paddr.wrapping_add(bm_off)).unwrap();
                    writeln!(self.asm, "    add hl, de").unwrap();
                } else {
                    writeln!(self.asm, "    ld hl, (0x{pa:04X})").unwrap();
                    writeln!(self.asm, "    add hl, de").unwrap();
                    writeln!(self.asm, "    ld de, 0x{bm_off:04X}").unwrap();
                    writeln!(self.asm, "    add hl, de").unwrap();
                }
                writeln!(self.asm, "    ld a, c").unwrap();
                writeln!(self.asm, "    and 7").unwrap();
                writeln!(self.asm, "    ld b, 1").unwrap();
                writeln!(self.asm, "    or a").unwrap();
                writeln!(self.asm, "    jr z, {sd}").unwrap();
                writeln!(self.asm, "{sl}:").unwrap();
                writeln!(self.asm, "    sla b").unwrap();
                writeln!(self.asm, "    dec a").unwrap();
                writeln!(self.asm, "    jr nz, {sl}").unwrap();
                writeln!(self.asm, "{sd}:").unwrap();
                if let Some((_, ref ok_live)) = ok_rng {
                    writeln!(self.asm, "    ld a, (hl)").unwrap();
                    writeln!(self.asm, "    and b").unwrap();
                    writeln!(self.asm, "    jr nz, {ok_live}").unwrap();
                    writeln!(self.asm, "    ld a, {}", PanicCode::InvalidHandle as u8).unwrap();
                    writeln!(self.asm, "    ld hl, 0x{site_id:04X}").unwrap();
                    writeln!(self.asm, "    call __aun_panic").unwrap();
                    writeln!(self.asm, "{ok_live}:").unwrap();
                }
                writeln!(self.asm, "    ld a, b").unwrap();
                writeln!(self.asm, "    cpl").unwrap();
                writeln!(self.asm, "    and (hl)").unwrap();
                writeln!(self.asm, "    ld (hl), a").unwrap();
            }
            IrInst::PoolElemAddr {
                dst,
                pool_addr,
                handle,
                elem_size,
                capacity,
                site_id,
            } => {
                let pa = self.val_loc(*pool_addr);
                let ha = self.val_loc(*handle);
                let da = self.val_addr(*dst);
                let bm_off = elem_size * capacity;
                let dbg_labels = if self.opts.check_bounds {
                    *self.needs_panic_stub = true;
                    Some((
                        self.fresh_label("pget_rng"),
                        self.fresh_label("pget_live"),
                        self.fresh_label("pget_sd"),
                        self.fresh_label("pget_sl"),
                    ))
                } else {
                    None
                };
                writeln!(self.asm, "    ld a, (0x{ha:04X})").unwrap();
                if let Some((ok_rng, ok_live, sd, sl)) = dbg_labels {
                    writeln!(self.asm, "    cp {}", *capacity as u8).unwrap();
                    writeln!(self.asm, "    jr c, {ok_rng}").unwrap();
                    writeln!(self.asm, "    ld a, {}", PanicCode::InvalidHandle as u8).unwrap();
                    writeln!(self.asm, "    ld hl, 0x{site_id:04X}").unwrap();
                    writeln!(self.asm, "    call __aun_panic").unwrap();
                    writeln!(self.asm, "{ok_rng}:").unwrap();
                    writeln!(self.asm, "    ld c, a").unwrap();
                    writeln!(self.asm, "    srl a").unwrap();
                    writeln!(self.asm, "    srl a").unwrap();
                    writeln!(self.asm, "    srl a").unwrap();
                    writeln!(self.asm, "    ld e, a").unwrap();
                    writeln!(self.asm, "    ld d, 0").unwrap();
                    if let Some(&paddr) = self.const_addrs.get(pool_addr) {
                        writeln!(self.asm, "    ld hl, 0x{:04X}", paddr.wrapping_add(bm_off))
                            .unwrap();
                        writeln!(self.asm, "    add hl, de").unwrap();
                    } else {
                        writeln!(self.asm, "    ld hl, (0x{pa:04X})").unwrap();
                        writeln!(self.asm, "    add hl, de").unwrap();
                        writeln!(self.asm, "    ld de, 0x{bm_off:04X}").unwrap();
                        writeln!(self.asm, "    add hl, de").unwrap();
                    }
                    writeln!(self.asm, "    ld a, c").unwrap();
                    writeln!(self.asm, "    and 7").unwrap();
                    writeln!(self.asm, "    ld b, 1").unwrap();
                    writeln!(self.asm, "    or a").unwrap();
                    writeln!(self.asm, "    jr z, {sd}").unwrap();
                    writeln!(self.asm, "{sl}:").unwrap();
                    writeln!(self.asm, "    sla b").unwrap();
                    writeln!(self.asm, "    dec a").unwrap();
                    writeln!(self.asm, "    jr nz, {sl}").unwrap();
                    writeln!(self.asm, "{sd}:").unwrap();
                    writeln!(self.asm, "    ld a, (hl)").unwrap();
                    writeln!(self.asm, "    and b").unwrap();
                    writeln!(self.asm, "    jr nz, {ok_live}").unwrap();
                    writeln!(self.asm, "    ld a, {}", PanicCode::InvalidHandle as u8).unwrap();
                    writeln!(self.asm, "    ld hl, 0x{site_id:04X}").unwrap();
                    writeln!(self.asm, "    call __aun_panic").unwrap();
                    writeln!(self.asm, "{ok_live}:").unwrap();
                    writeln!(self.asm, "    ld l, c").unwrap();
                } else {
                    // Release: 0-byte check elimination!
                    writeln!(self.asm, "    ld l, a").unwrap();
                }
                writeln!(self.asm, "    ld h, 0").unwrap();
                emit_mul_hl_const(*elem_size, self.asm);
                if let Some(&paddr) = self.const_addrs.get(pool_addr) {
                    writeln!(self.asm, "    ld de, 0x{paddr:04X}").unwrap();
                } else {
                    writeln!(self.asm, "    ld de, (0x{pa:04X})").unwrap();
                }
                writeln!(self.asm, "    add hl, de").unwrap();
                writeln!(self.asm, "    ld (0x{da:04X}), hl").unwrap();
            }
            IrInst::BankEnter {
                group,
                banks,
                needs_di,
            } => {
                if *needs_di {
                    writeln!(self.asm, "    di").unwrap();
                }
                emit_bank_switch(self.program, group, banks, self.asm);
            }
            IrInst::BankExit {
                group,
                restore_banks,
                restore_ei,
            } => {
                emit_bank_switch(self.program, group, restore_banks, self.asm);
                if *restore_ei {
                    writeln!(self.asm, "    ei").unwrap();
                }
            }
            IrInst::InlineAsm { template, .. } => {
                for line in template.split(';') {
                    let t = line.trim();
                    if !t.is_empty() {
                        writeln!(self.asm, "    {t}").unwrap();
                    }
                }
            }
        }
        Ok(())
    }
}
