use super::runtime::{emit_bank_switch, emit_mul_hl_const, ranges_overlap};
use super::{EmitCtx, IndexKey};
use crate::analysis::{
    can_alias_ix_load, can_alias_local_load, can_alias_reg_load, is_const_addr_always_folded,
    is_const_always_folded, is_signed_ty, z80_sym,
};
use crate::regalloc::{extern_clobber_mask, RegLoc, RegMask};
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
        let fused_cmp_dsts = &self.fa.fused_cmp_dsts;
        let val_sizes = &self.fa.val_sizes;

        // Invalidate same-block shift CSE on any scratch write overlapping `prev_da`
        if let Some(d) = inst.dst() {
            let da = self.val_addr(d);
            let dsz = val_sizes.get(&d).copied().unwrap_or(2).clamp(1, 4);
            self.shift_cse
                .retain(|&(src_a, _, _), &mut (_, pda)| pda != da && src_a != da);
            self.index_cse.retain(|(_, ik, _), &mut res_a| {
                let idx_ok = match *ik {
                    IndexKey::Reg(_) => true,
                    IndexKey::Mem(ia, isz) => !ranges_overlap(ia, isz, da, dsz),
                };
                idx_ok && !ranges_overlap(res_a, 2, da, dsz)
            });
        }
        // Invalidate same-block shift CSE, index CSE, and non-negative IX tracking on memory/local mutations
        match inst {
            IrInst::StoreLocal { slot, ty, .. } => {
                if let Some(reg) = self.fa.reg_alloc.slot_reg(*slot) {
                    self.index_cse.retain(|(_, ik, _), _| match *ik {
                        IndexKey::Reg(rk) => !rk.mask().intersects(reg.mask()),
                        IndexKey::Mem(..) => true,
                    });
                } else {
                    let sa = self.layout.slot_addrs[&(self.func.id, *slot)];
                    let ssz = self.program.type_size(ty).clamp(1, 4);
                    self.shift_cse
                        .retain(|&(src_a, _, _), _| !ranges_overlap(src_a, 2, sa, ssz));
                    self.index_cse.retain(|(_, ik, _), &mut res_a| {
                        let idx_ok = match *ik {
                            IndexKey::Reg(_) => true,
                            IndexKey::Mem(ia, isz) => !ranges_overlap(ia, isz, sa, ssz),
                        };
                        idx_ok && !ranges_overlap(res_a, 2, sa, ssz)
                    });
                }
            }
            IrInst::StorePtr { .. }
            | IrInst::LoadPtr { volatile: true, .. }
            | IrInst::MemSet { .. }
            | IrInst::MemCopy { .. }
            | IrInst::MemFill { .. }
            | IrInst::Call { .. }
            | IrInst::CallIndirect { .. }
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
                let da_mem = self.fmt_dst_mem(*dst);
                match self.program.type_size(ty) {
                    1 => {
                        let b = (*val & 0xFF) as u8;
                        if b == 0 {
                            writeln!(self.asm, "    xor a").unwrap();
                        } else {
                            writeln!(self.asm, "    ld a, 0x{b:02X}").unwrap();
                        }
                        writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                    }
                    2 => {
                        let w = *val as u16;
                        writeln!(self.asm, "    ld hl, 0x{w:04X}").unwrap();
                        writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                    }
                    4 => self.emit_const32_to(da, true, *val),
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
                let da_mem = self.fmt_dst_mem(*dst);
                if *val {
                    writeln!(self.asm, "    ld a, 1").unwrap();
                } else {
                    writeln!(self.asm, "    xor a").unwrap();
                }
                writeln!(self.asm, "    ld {da_mem}, a").unwrap();
            }
            IrInst::LoadLocal { dst, slot, ty } => {
                if ix_param_slot == Some(*slot) {
                    self.ix_addrs.insert(*dst, 0);
                    return Ok(());
                }
                if let Some(reg) = self.fa.reg_alloc.slot_reg(*slot) {
                    if can_alias_reg_load(
                        self.program,
                        self.func,
                        &self.fa.reg_alloc,
                        &self.analysis.fn_clobbers,
                        bb_idx,
                        inst_idx,
                        *dst,
                        reg,
                    ) {
                        self.val_in_reg.insert(*dst, reg);
                        return Ok(());
                    }
                    let da_mem = self.fmt_dst_mem(*dst);
                    match reg {
                        RegLoc::C => {
                            writeln!(self.asm, "    ld a, c").unwrap();
                            writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                        }
                        RegLoc::B => {
                            writeln!(self.asm, "    ld a, b").unwrap();
                            writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                        }
                        RegLoc::BC => {
                            writeln!(self.asm, "    ld l, c").unwrap();
                            writeln!(self.asm, "    ld h, b").unwrap();
                            writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                        }
                    }
                    return Ok(());
                }
                let sa = self.layout.slot_addrs[&(self.func.id, *slot)];
                if can_alias_local_load(self.func, bb_idx, inst_idx, *dst, *slot) {
                    self.val_aliases.insert(*dst, sa);
                    return Ok(());
                }
                let da = self.val_addr(*dst);
                let sa_mem = self.fmt_ram_mem(sa);
                let da_mem = self.fmt_dst_mem(*dst);
                match self.program.type_size(ty) {
                    1 => {
                        writeln!(self.asm, "    ld a, {sa_mem}").unwrap();
                        writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                    }
                    2 => {
                        writeln!(self.asm, "    ld hl, {sa_mem}").unwrap();
                        writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                    }
                    4 => self.emit_copy32(sa, true, da, true),
                    _ => {}
                }
            }
            IrInst::StoreLocal { slot, val, ty } => {
                if let Some(reg) = self.fa.reg_alloc.slot_reg(*slot) {
                    match reg {
                        RegLoc::C | RegLoc::B => {
                            let r = reg.reg8_name();
                            if let Some(&c) = self.const_vals.get(val) {
                                writeln!(self.asm, "    ld {r}, 0x{:02X}", (c & 0xFF) as u8)
                                    .unwrap();
                            } else if let Some(&src_r) = self.val_in_reg.get(val) {
                                if src_r != reg {
                                    writeln!(self.asm, "    ld {r}, {}", src_r.reg8_name()).unwrap();
                                }
                            } else if let Some(&(disp, _)) = self.ix_val_loads.get(val) {
                                writeln!(self.asm, "    ld {r}, (ix + {disp})").unwrap();
                            } else {
                                self.emit_load_a_from(*val);
                                writeln!(self.asm, "    ld {r}, a").unwrap();
                            }
                        }
                        RegLoc::BC => {
                            if let Some(&c) = self.const_vals.get(val) {
                                writeln!(self.asm, "    ld bc, 0x{:04X}", c as u16).unwrap();
                            } else if self.val_in_reg.get(val) == Some(&RegLoc::BC) {
                                // already in BC
                            } else if let Some(&(disp, _)) = self.ix_val_loads.get(val) {
                                writeln!(self.asm, "    ld c, (ix + {disp})").unwrap();
                                writeln!(self.asm, "    ld b, (ix + {})", disp + 1).unwrap();
                            } else {
                                self.emit_load_hl_from(*val);
                                writeln!(self.asm, "    ld b, h").unwrap();
                                writeln!(self.asm, "    ld c, l").unwrap();
                            }
                        }
                    }
                    self.val_in_reg
                        .retain(|_, &mut vr| !vr.mask().intersects(reg.mask()));
                    return Ok(());
                }
                let sa = self.layout.slot_addrs[&(self.func.id, *slot)];
                let sa_mem = self.fmt_ram_mem(sa);
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
                            writeln!(self.asm, "    ld {sa_mem}, a").unwrap();
                        }
                        2 => {
                            writeln!(self.asm, "    ld hl, 0x{:04X}", c as u16).unwrap();
                            writeln!(self.asm, "    ld {sa_mem}, hl").unwrap();
                        }
                        4 => self.emit_const32_to(sa, true, c),
                        _ => {}
                    }
                } else {
                    match sz {
                        1 => {
                            self.emit_load_a_from(*val);
                            writeln!(self.asm, "    ld {sa_mem}, a").unwrap();
                        }
                        2 => {
                            self.emit_load_hl_from(*val);
                            writeln!(self.asm, "    ld {sa_mem}, hl").unwrap();
                        }
                        4 => self.emit_copy32(self.val_loc(*val), true, sa, true),
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
                self.const_addrs.insert(*dst, (sa, true));
                if is_const_addr_always_folded(
                    self.program,
                    self.func,
                    *dst,
                    self.fused_skip_insts,
                ) {
                    return Ok(());
                }
                let sa_imm = self.fmt_ram_imm(sa);
                let da_mem = self.fmt_dst_mem(*dst);
                writeln!(self.asm, "    ld hl, {sa_imm}").unwrap();
                writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
            }
            IrInst::AddrGlobal { dst, global } => {
                let g = &self.program.globals[*global as usize];
                let da_mem = self.fmt_dst_mem(*dst);
                if g.in_rom || g.is_extern {
                    self.rom_addrs.insert(*dst, (g.name.clone(), 0));
                    if is_const_addr_always_folded(
                    self.program,
                    self.func,
                    *dst,
                    self.fused_skip_insts,
                ) {
                        return Ok(());
                    }
                    writeln!(self.asm, "    ld hl, {}", g.name).unwrap();
                    writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                } else {
                    let ga = self.layout.global_addrs[global];
                    let is_u = g.fixed_addr.is_none() && g.space.is_none();
                    self.const_addrs.insert(*dst, (ga, is_u));
                    if is_const_addr_always_folded(
                    self.program,
                    self.func,
                    *dst,
                    self.fused_skip_insts,
                ) {
                        return Ok(());
                    }
                    let ga_imm = self.fmt_addr_imm(ga, is_u);
                    writeln!(self.asm, "    ld hl, {ga_imm}").unwrap();
                    writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                }
            }
            IrInst::AddrFn { dst, target } => {
                let sym = match target {
                    IrFnTarget::Fn(fid) => z80_sym(&self.program.functions[*fid as usize].name),
                    IrFnTarget::Extern(eid) => self.program.externs[*eid as usize].name.clone(),
                };
                self.rom_addrs.insert(*dst, (sym.clone(), 0));
                if is_const_addr_always_folded(
                    self.program,
                    self.func,
                    *dst,
                    self.fused_skip_insts,
                ) {
                    return Ok(());
                }
                let da_mem = self.fmt_dst_mem(*dst);
                writeln!(self.asm, "    ld hl, {sym}").unwrap();
                writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
            }
            IrInst::FieldAddr {
                dst,
                base,
                byte_offset,
            } => {
                let da_mem = self.fmt_dst_mem(*dst);
                if let Some(&disp) = self.ix_addrs.get(base)
                    && let Some(new_disp) = (disp as i32)
                        .checked_add(*byte_offset as i32)
                        .filter(|&d| (-128..=125).contains(&d))
                {
                    self.ix_addrs.insert(*dst, new_disp as i8);
                    return Ok(());
                }
                if let Some(&(baddr, is_u)) = self.const_addrs.get(base) {
                    let folded = baddr.wrapping_add(*byte_offset);
                    self.const_addrs.insert(*dst, (folded, is_u));
                    if is_const_addr_always_folded(
                    self.program,
                    self.func,
                    *dst,
                    self.fused_skip_insts,
                ) {
                        return Ok(());
                    }
                    let folded_imm = self.fmt_addr_imm(folded, is_u);
                    writeln!(self.asm, "    ld hl, {folded_imm}").unwrap();
                    writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                } else if let Some((sym, off)) = self.rom_addrs.get(base).cloned() {
                    let new_off = off.wrapping_add(*byte_offset);
                    self.rom_addrs.insert(*dst, (sym.clone(), new_off));
                    if is_const_addr_always_folded(
                    self.program,
                    self.func,
                    *dst,
                    self.fused_skip_insts,
                ) {
                        return Ok(());
                    }
                    if new_off == 0 {
                        writeln!(self.asm, "    ld hl, {sym}").unwrap();
                    } else {
                        writeln!(self.asm, "    ld hl, {sym} + {new_off}").unwrap();
                    }
                    writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                } else {
                    self.emit_load_hl_from(*base);
                    if (1..=3).contains(byte_offset) {
                        for _ in 0..*byte_offset {
                            writeln!(self.asm, "    inc hl").unwrap();
                        }
                    } else if *byte_offset > 0 {
                        writeln!(self.asm, "    ld de, 0x{byte_offset:04X}").unwrap();
                        writeln!(self.asm, "    add hl, de").unwrap();
                    }
                    writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
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
                let ia = self.val_loc(*index);
                let ia_mem = self.fmt_val_mem(*index);
                let da = self.val_addr(*dst);
                let da_mem = self.fmt_dst_mem(*dst);
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
                    if let Some(&(baddr, is_u)) = self.const_addrs.get(base) {
                        let folded = baddr.wrapping_add(byte_off);
                        self.const_addrs.insert(*dst, (folded, is_u));
                        if is_const_addr_always_folded(
                    self.program,
                    self.func,
                    *dst,
                    self.fused_skip_insts,
                ) {
                            return Ok(());
                        }
                        let folded_imm = self.fmt_addr_imm(folded, is_u);
                        writeln!(self.asm, "    ld hl, {folded_imm}").unwrap();
                        writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                        return Ok(());
                    }
                    if let Some((sym, off)) = self.rom_addrs.get(base).cloned() {
                        let new_off = off.wrapping_add(byte_off);
                        self.rom_addrs.insert(*dst, (sym.clone(), new_off));
                        if is_const_addr_always_folded(
                    self.program,
                    self.func,
                    *dst,
                    self.fused_skip_insts,
                ) {
                            return Ok(());
                        }
                        if new_off == 0 {
                            writeln!(self.asm, "    ld hl, {sym}").unwrap();
                        } else {
                            writeln!(self.asm, "    ld hl, {sym} + {new_off}").unwrap();
                        }
                        writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                        return Ok(());
                    }
                }
                // Same-BB `IndexAddr` CSE: reuse already-computed `base + index * elem_size`!
                let cse_base_key = if let Some(&disp) = self.ix_addrs.get(base) {
                    Some(format!("ix:{disp}"))
                } else if let Some(&(baddr, is_u)) = self.const_addrs.get(base) {
                    let kind = if is_u { "ram" } else { "fix" };
                    Some(format!("{kind}:{baddr:04X}"))
                } else if let Some((sym, off)) = self.rom_addrs.get(base) {
                    Some(format!("rom:{sym}:{off}"))
                } else {
                    None
                };
                let reg_idx = self.val_in_reg.get(index).copied().filter(|r| r.byte_size() == 1);
                let idx_key = if idx_size == 1 && let Some(r) = reg_idx {
                    IndexKey::Reg(r)
                } else {
                    IndexKey::Mem(ia, idx_size.clamp(1, 2))
                };
                if let Some(ref bkey) = cse_base_key
                    && let Some(&prev_da) =
                        self.index_cse.get(&(bkey.clone(), idx_key, *elem_size))
                {
                    let pda_mem = self.fmt_ram_mem(prev_da);
                    writeln!(self.asm, "    ld hl, {pda_mem}").unwrap();
                    writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                    return Ok(());
                }
                let use_mul_a = idx_size == 1
                    && !(self.opts.check_bounds && *len > 0)
                    && self.shared_muls.contains(elem_size);
                // Compact 8-bit indexing into an IX struct array field (`ix + disp + u8`) when
                // bounds checks are disabled and `disp + len - 1 <= 255` fits in 8 bits without carry.
                let compact_ix_byte = idx_size == 1
                    && *elem_size == 1
                    && *len > 0
                    && !(self.opts.check_bounds && *len > 0)
                    && self
                        .ix_addrs
                        .get(base)
                        .is_some_and(|&d| d > 0 && (d as u16) + (*len - 1) <= 255);
                if compact_ix_byte {
                    let disp = self.ix_addrs[base] as u8;
                    self.emit_load_a_from(*index);
                    writeln!(self.asm, "    add a, 0x{disp:02X}").unwrap();
                    writeln!(self.asm, "    ld l, a").unwrap();
                    writeln!(self.asm, "    ld h, 0").unwrap();
                    writeln!(self.asm, "    push ix").unwrap();
                    writeln!(self.asm, "    pop de").unwrap();
                    writeln!(self.asm, "    add hl, de").unwrap();
                } else if use_mul_a {
                    if let Some(r) = reg_idx {
                        writeln!(self.asm, "    ld a, {}", r.reg8_name()).unwrap();
                    } else {
                        writeln!(self.asm, "    ld a, {ia_mem}").unwrap();
                    }
                    writeln!(self.asm, "    call __aun_mul_a_{elem_size}").unwrap();
                } else {
                    if idx_size == 1 {
                        if let Some(r) = reg_idx {
                            writeln!(self.asm, "    ld l, {}", r.reg8_name()).unwrap();
                        } else {
                            writeln!(self.asm, "    ld a, {ia_mem}").unwrap();
                            writeln!(self.asm, "    ld l, a").unwrap();
                        }
                        writeln!(self.asm, "    ld h, 0").unwrap();
                    } else {
                        writeln!(self.asm, "    ld hl, {ia_mem}").unwrap();
                    }

                    // Bounds check: index < len (eliminated in Release!)
                    if self.opts.check_bounds && *len > 0 {
                        self.runtime_needs.panic_stub = true;
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

                // Add base address (already added for `compact_ix_byte`)
                if compact_ix_byte {
                } else if let Some(&disp) = self.ix_addrs.get(base) {
                    writeln!(self.asm, "    push ix").unwrap();
                    writeln!(self.asm, "    pop de").unwrap();
                    writeln!(self.asm, "    add hl, de").unwrap();
                    if disp != 0 {
                        writeln!(self.asm, "    ld de, 0x{:04X}", (disp as i16) as u16).unwrap();
                        writeln!(self.asm, "    add hl, de").unwrap();
                    }
                } else if let Some(&(baddr, is_u)) = self.const_addrs.get(base) {
                    let b_imm = self.fmt_addr_imm(baddr, is_u);
                    writeln!(self.asm, "    ld de, {b_imm}").unwrap();
                    writeln!(self.asm, "    add hl, de").unwrap();
                } else if let Some((sym, off)) = self.rom_addrs.get(base) {
                    if *off == 0 {
                        writeln!(self.asm, "    ld de, {sym}").unwrap();
                    } else {
                        writeln!(self.asm, "    ld de, {sym} + {off}").unwrap();
                    }
                    writeln!(self.asm, "    add hl, de").unwrap();
                } else if self.val_in_reg.get(base) == Some(&RegLoc::BC) {
                    writeln!(self.asm, "    add hl, bc").unwrap();
                } else {
                    self.emit_load_de_from(*base);
                    writeln!(self.asm, "    add hl, de").unwrap();
                }
                writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                if let Some(bkey) = cse_base_key {
                    let no_self_overlap = match idx_key {
                        IndexKey::Reg(_) => true,
                        IndexKey::Mem(ia_m, isz) => !ranges_overlap(da, 2, ia_m, isz),
                    };
                    if no_self_overlap {
                        self.index_cse.insert((bkey, idx_key, *elem_size), da);
                    }
                }
            }
            IrInst::LoadPtr {
                dst,
                addr,
                ty,
                volatile,
            } => {
                let da = self.val_addr(*dst);
                // If reading the IX-promoted param pointer itself, `dst` IS `IX + 0`!
                if !*volatile && self.ix_param_addr_vals.contains(addr) {
                    self.ix_addrs.insert(*dst, 0);
                    return Ok(());
                }
                if !*volatile
                    && self.ix_addrs.contains_key(addr)
                    && matches!(self.program.type_size(ty), 1 | 2)
                    && can_alias_ix_load(
                        self.program,
                        self.func,
                        bb_idx,
                        inst_idx,
                        *dst,
                        fused_cmp_dsts,
                    )
                {
                    let disp = self.ix_addrs[addr];
                    let sz = self.program.type_size(ty);
                    self.ix_val_loads.insert(*dst, (disp, sz));
                    if sz == 2 && self.block_nonneg_ix == Some(disp) {
                        self.nonneg_vals.insert(*dst);
                    }
                    return Ok(());
                }
                if *volatile {
                    writeln!(self.asm, "; @aun:volatile_begin").unwrap();
                }
                if let Some(&disp) = self.ix_addrs.get(addr) {
                    let sz = self.program.type_size(ty);
                    match sz {
                        1 => {
                            writeln!(self.asm, "    ld a, (ix + {disp})").unwrap();
                            writeln!(self.asm, "    ld {}, a", self.fmt_ram_mem(da)).unwrap();
                        }
                        2 => {
                            writeln!(self.asm, "    ld l, (ix + {disp})").unwrap();
                            writeln!(self.asm, "    ld h, (ix + {})", disp + 1).unwrap();
                            writeln!(self.asm, "    ld {}, hl", self.fmt_ram_mem(da)).unwrap();
                            if self.block_nonneg_ix == Some(disp) {
                                self.nonneg_vals.insert(*dst);
                            }
                        }
                        4 => {
                            writeln!(self.asm, "    ld l, (ix + {disp})").unwrap();
                            writeln!(self.asm, "    ld h, (ix + {})", disp + 1).unwrap();
                            writeln!(self.asm, "    ld {}, hl", self.fmt_ram_mem(da)).unwrap();
                            writeln!(self.asm, "    ld l, (ix + {})", disp + 2).unwrap();
                            writeln!(self.asm, "    ld h, (ix + {})", disp + 3).unwrap();
                            writeln!(self.asm, "    ld {}, hl", self.fmt_ram_mem(da.wrapping_add(2)))
                                .unwrap();
                        }
                        _ => {}
                    }
                } else if let Some((fixed_addr, is_u)) = self
                    .const_addrs
                    .get(addr)
                    .copied()
                    .or_else(|| self.const_vals.get(addr).map(|&c| (c as u16, false)))
                {
                    let f_mem = self.fmt_addr_mem(fixed_addr, is_u);
                    let da_mem = self.fmt_ram_mem(da);
                    match self.program.type_size(ty) {
                        1 => {
                            writeln!(self.asm, "    ld a, {f_mem}").unwrap();
                            writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                        }
                        2 => {
                            writeln!(self.asm, "    ld hl, {f_mem}").unwrap();
                            writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                        }
                        4 => self.emit_copy32(fixed_addr, is_u, da, true),
                        _ => {}
                    }
                } else if let Some((sym, off)) = self.rom_addrs.get(addr) {
                    if *off == 0 {
                        writeln!(self.asm, "    ld hl, {sym}").unwrap();
                    } else {
                        writeln!(self.asm, "    ld hl, {sym} + {off}").unwrap();
                    }
                    let da_mem = self.fmt_ram_mem(da);
                    match self.program.type_size(ty) {
                        1 => {
                            writeln!(self.asm, "    ld a, (hl)").unwrap();
                            writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                        }
                        2 => {
                            writeln!(self.asm, "    ld e, (hl)").unwrap();
                            writeln!(self.asm, "    inc hl").unwrap();
                            writeln!(self.asm, "    ld d, (hl)").unwrap();
                            writeln!(self.asm, "    ex de, hl").unwrap();
                            writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                        }
                        4 => {
                            let da2_mem = self.fmt_ram_mem(da.wrapping_add(2));
                            writeln!(self.asm, "    ld e, (hl)").unwrap();
                            writeln!(self.asm, "    inc hl").unwrap();
                            writeln!(self.asm, "    ld d, (hl)").unwrap();
                            writeln!(self.asm, "    inc hl").unwrap();
                            writeln!(self.asm, "    ld {da_mem}, de").unwrap();
                            writeln!(self.asm, "    ld e, (hl)").unwrap();
                            writeln!(self.asm, "    inc hl").unwrap();
                            writeln!(self.asm, "    ld d, (hl)").unwrap();
                            writeln!(self.asm, "    ld {da2_mem}, de").unwrap();
                        }
                        _ => {}
                    }
                } else {
                    self.emit_load_hl_from(*addr);
                    let da_mem = self.fmt_ram_mem(da);
                    match self.program.type_size(ty) {
                        1 => {
                            writeln!(self.asm, "    ld a, (hl)").unwrap();
                            writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                        }
                        2 => {
                            writeln!(self.asm, "    ld e, (hl)").unwrap();
                            writeln!(self.asm, "    inc hl").unwrap();
                            writeln!(self.asm, "    ld d, (hl)").unwrap();
                            writeln!(self.asm, "    ex de, hl").unwrap();
                            writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                        }
                        4 => {
                            let da2_mem = self.fmt_ram_mem(da.wrapping_add(2));
                            writeln!(self.asm, "    ld e, (hl)").unwrap();
                            writeln!(self.asm, "    inc hl").unwrap();
                            writeln!(self.asm, "    ld d, (hl)").unwrap();
                            writeln!(self.asm, "    inc hl").unwrap();
                            writeln!(self.asm, "    ld {da_mem}, de").unwrap();
                            writeln!(self.asm, "    ld e, (hl)").unwrap();
                            writeln!(self.asm, "    inc hl").unwrap();
                            writeln!(self.asm, "    ld d, (hl)").unwrap();
                            writeln!(self.asm, "    ld {da2_mem}, de").unwrap();
                        }
                        _ => {}
                    }
                }
                if *volatile {
                    writeln!(self.asm, "; @aun:volatile_end").unwrap();
                }
            }
            IrInst::StorePtr {
                addr,
                val,
                ty,
                volatile,
            } => {
                if *volatile {
                    writeln!(self.asm, "; @aun:volatile_begin").unwrap();
                }
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
                            4 => {
                                let bytes = (c as u32).to_le_bytes();
                                for (off, b) in bytes.into_iter().enumerate() {
                                    writeln!(
                                        self.asm,
                                        "    ld (ix + {}), 0x{b:02X}",
                                        disp + (off as i8)
                                    )
                                    .unwrap();
                                }
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
                            4 => {
                                self.emit_load_hlde_from(*val);
                                writeln!(self.asm, "    ld (ix + {disp}), e").unwrap();
                                writeln!(self.asm, "    ld (ix + {}), d", disp + 1).unwrap();
                                writeln!(self.asm, "    ld (ix + {}), l", disp + 2).unwrap();
                                writeln!(self.asm, "    ld (ix + {}), h", disp + 3).unwrap();
                            }
                            _ => {}
                        }
                    }
                } else if let Some((fixed_addr, is_u)) = self
                    .const_addrs
                    .get(addr)
                    .copied()
                    .or_else(|| self.const_vals.get(addr).map(|&c| (c as u16, false)))
                {
                    let f_mem = self.fmt_addr_mem(fixed_addr, is_u);
                    if let Some(&c) = self.const_vals.get(val) {
                        match sz {
                            1 => {
                                let b = (c & 0xFF) as u8;
                                if b == 0 {
                                    writeln!(self.asm, "    xor a").unwrap();
                                } else {
                                    writeln!(self.asm, "    ld a, 0x{b:02X}").unwrap();
                                }
                                writeln!(self.asm, "    ld {f_mem}, a").unwrap();
                            }
                            2 => {
                                writeln!(self.asm, "    ld hl, 0x{:04X}", c as u16).unwrap();
                                writeln!(self.asm, "    ld {f_mem}, hl").unwrap();
                            }
                            4 => self.emit_const32_to(fixed_addr, is_u, c),
                            _ => {}
                        }
                    } else {
                        match sz {
                            1 => {
                                self.emit_load_a_from(*val);
                                writeln!(self.asm, "    ld {f_mem}, a").unwrap();
                            }
                            2 => {
                                self.emit_load_hl_from(*val);
                                writeln!(self.asm, "    ld {f_mem}, hl").unwrap();
                            }
                            4 => self.emit_copy32(self.val_loc(*val), true, fixed_addr, is_u),
                            _ => {}
                        }
                    }
                } else if let Some((sym, off)) = self.rom_addrs.get(addr).cloned() {
                    let emit_ld_hl_sym = |asm: &mut String| {
                        if off == 0 {
                            writeln!(asm, "    ld hl, {sym}").unwrap();
                        } else {
                            writeln!(asm, "    ld hl, {sym} + {off}").unwrap();
                        }
                    };
                    match sz {
                        1 => {
                            if let Some(&c) = self.const_vals.get(val) {
                                emit_ld_hl_sym(self.asm);
                                writeln!(self.asm, "    ld (hl), 0x{:02X}", (c & 0xFF) as u8)
                                    .unwrap();
                            } else {
                                self.emit_load_a_from(*val);
                                emit_ld_hl_sym(self.asm);
                                writeln!(self.asm, "    ld (hl), a").unwrap();
                            }
                        }
                        2 => {
                            self.emit_load_de_from(*val);
                            emit_ld_hl_sym(self.asm);
                            writeln!(self.asm, "    ld (hl), e").unwrap();
                            writeln!(self.asm, "    inc hl").unwrap();
                            writeln!(self.asm, "    ld (hl), d").unwrap();
                        }
                        4 => {
                            if let Some(&c) = self.const_vals.get(val) {
                                let [b0, b1, b2, b3] = (c as u32).to_le_bytes();
                                emit_ld_hl_sym(self.asm);
                                writeln!(self.asm, "    ld (hl), 0x{b0:02X}").unwrap();
                                writeln!(self.asm, "    inc hl").unwrap();
                                writeln!(self.asm, "    ld (hl), 0x{b1:02X}").unwrap();
                                writeln!(self.asm, "    inc hl").unwrap();
                                writeln!(self.asm, "    ld (hl), 0x{b2:02X}").unwrap();
                                writeln!(self.asm, "    inc hl").unwrap();
                                writeln!(self.asm, "    ld (hl), 0x{b3:02X}").unwrap();
                            } else {
                                let va = self.val_loc(*val);
                                let va_mem = self.fmt_ram_mem(va);
                                let va2_mem = self.fmt_ram_mem(va.wrapping_add(2));
                                writeln!(self.asm, "    ld de, {va_mem}").unwrap();
                                emit_ld_hl_sym(self.asm);
                                writeln!(self.asm, "    ld (hl), e").unwrap();
                                writeln!(self.asm, "    inc hl").unwrap();
                                writeln!(self.asm, "    ld (hl), d").unwrap();
                                writeln!(self.asm, "    inc hl").unwrap();
                                writeln!(self.asm, "    ld de, {va2_mem}").unwrap();
                                writeln!(self.asm, "    ld (hl), e").unwrap();
                                writeln!(self.asm, "    inc hl").unwrap();
                                writeln!(self.asm, "    ld (hl), d").unwrap();
                            }
                        }
                        _ => {}
                    }
                } else {
                    let aa_mem = self.fmt_val_mem(*addr);
                    match sz {
                        1 => {
                            writeln!(self.asm, "    ld hl, {aa_mem}").unwrap();
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
                            writeln!(self.asm, "    ld hl, {aa_mem}").unwrap();
                            writeln!(self.asm, "    ld (hl), e").unwrap();
                            writeln!(self.asm, "    inc hl").unwrap();
                            writeln!(self.asm, "    ld (hl), d").unwrap();
                        }
                        4 => {
                            if let Some(&c) = self.const_vals.get(val) {
                                let [b0, b1, b2, b3] = (c as u32).to_le_bytes();
                                writeln!(self.asm, "    ld hl, {aa_mem}").unwrap();
                                writeln!(self.asm, "    ld (hl), 0x{b0:02X}").unwrap();
                                writeln!(self.asm, "    inc hl").unwrap();
                                writeln!(self.asm, "    ld (hl), 0x{b1:02X}").unwrap();
                                writeln!(self.asm, "    inc hl").unwrap();
                                writeln!(self.asm, "    ld (hl), 0x{b2:02X}").unwrap();
                                writeln!(self.asm, "    inc hl").unwrap();
                                writeln!(self.asm, "    ld (hl), 0x{b3:02X}").unwrap();
                            } else {
                                let va = self.val_loc(*val);
                                let va_mem = self.fmt_ram_mem(va);
                                let va2_mem = self.fmt_ram_mem(va.wrapping_add(2));
                                writeln!(self.asm, "    ld de, {va_mem}").unwrap();
                                writeln!(self.asm, "    ld hl, {aa_mem}").unwrap();
                                writeln!(self.asm, "    ld (hl), e").unwrap();
                                writeln!(self.asm, "    inc hl").unwrap();
                                writeln!(self.asm, "    ld (hl), d").unwrap();
                                writeln!(self.asm, "    inc hl").unwrap();
                                writeln!(self.asm, "    ld de, {va2_mem}").unwrap();
                                writeln!(self.asm, "    ld (hl), e").unwrap();
                                writeln!(self.asm, "    inc hl").unwrap();
                                writeln!(self.asm, "    ld (hl), d").unwrap();
                            }
                        }
                        _ => {}
                    }
                }
                if *volatile {
                    writeln!(self.asm, "; @aun:volatile_end").unwrap();
                }
            }
            IrInst::MemSet { dst_addr, val, len } => {
                if *len == 0 {
                    return Ok(());
                }
                let load_dst_hl = |this: &mut EmitCtx<'_>| {
                    if let Some(&disp) = this.ix_addrs.get(dst_addr) {
                        writeln!(this.asm, "    push ix").unwrap();
                        writeln!(this.asm, "    pop hl").unwrap();
                        if disp != 0 {
                            writeln!(this.asm, "    ld de, 0x{:04X}", disp as i16 as u16).unwrap();
                            writeln!(this.asm, "    add hl, de").unwrap();
                        }
                    } else {
                        this.emit_load_hl_from(*dst_addr);
                    }
                };
                if *len == 1 {
                    load_dst_hl(self);
                    if let Some(&c) = self.const_vals.get(val) {
                        writeln!(self.asm, "    ld (hl), 0x{:02X}", (c & 0xFF) as u8).unwrap();
                    } else {
                        self.emit_load_a_from(*val);
                        writeln!(self.asm, "    ld (hl), a").unwrap();
                    }
                } else if *len <= 256 {
                    load_dst_hl(self);
                    self.emit_load_a_from(*val);
                    let cnt = (*len & 0xFF) as u8;
                    writeln!(self.asm, "    ld b, 0x{cnt:02X}").unwrap();
                    let lp = self.fresh_label("memset");
                    writeln!(self.asm, "{lp}:").unwrap();
                    writeln!(self.asm, "    ld (hl), a").unwrap();
                    writeln!(self.asm, "    inc hl").unwrap();
                    writeln!(self.asm, "    djnz {lp}").unwrap();
                    self.val_in_reg
                        .retain(|_, &mut vr| !RegMask::B.intersects(vr.mask()));
                } else {
                    load_dst_hl(self);
                    self.emit_load_a_from(*val);
                    writeln!(self.asm, "    ld (hl), a").unwrap();
                    writeln!(self.asm, "    ld d, h").unwrap();
                    writeln!(self.asm, "    ld e, l").unwrap();
                    writeln!(self.asm, "    inc de").unwrap();
                    writeln!(self.asm, "    ld bc, 0x{:04X}", *len - 1).unwrap();
                    writeln!(self.asm, "    ldir").unwrap();
                    self.val_in_reg
                        .retain(|_, &mut vr| !RegMask::BC.intersects(vr.mask()));
                }
            }
            IrInst::MemCopy {
                dst_addr,
                src_addr,
                len,
            } => {
                if *len == 0 {
                    return Ok(());
                }
                if let (Some(sa), Some(da)) = (
                    self.const_addrs.get(src_addr),
                    self.const_addrs.get(dst_addr),
                ) && sa == da
                {
                    return Ok(());
                }
                if let Some(&disp) = self.ix_addrs.get(src_addr) {
                    writeln!(self.asm, "    push ix").unwrap();
                    writeln!(self.asm, "    pop hl").unwrap();
                    if disp != 0 {
                        writeln!(self.asm, "    ld de, 0x{:04X}", disp as i16 as u16).unwrap();
                        writeln!(self.asm, "    add hl, de").unwrap();
                    }
                } else {
                    self.emit_load_hl_from(*src_addr);
                }
                if let Some(&disp) = self.ix_addrs.get(dst_addr) {
                    if disp == 0 {
                        writeln!(self.asm, "    push ix").unwrap();
                        writeln!(self.asm, "    pop de").unwrap();
                    } else {
                        writeln!(self.asm, "    push hl").unwrap();
                        writeln!(self.asm, "    push ix").unwrap();
                        writeln!(self.asm, "    pop hl").unwrap();
                        writeln!(self.asm, "    ld de, 0x{:04X}", disp as i16 as u16).unwrap();
                        writeln!(self.asm, "    add hl, de").unwrap();
                        writeln!(self.asm, "    ex de, hl").unwrap();
                        writeln!(self.asm, "    pop hl").unwrap();
                    }
                } else {
                    self.emit_load_de_from(*dst_addr);
                }
                writeln!(self.asm, "    ld bc, 0x{len:04X}").unwrap();
                writeln!(self.asm, "    ldir").unwrap();
                self.val_in_reg
                    .retain(|_, &mut vr| !RegMask::BC.intersects(vr.mask()));
            }
            IrInst::MemFill {
                dst_addr,
                elem_size,
                count,
            } => {
                if *count <= 1 || *elem_size == 0 {
                    return Ok(());
                }
                if let Some(&disp) = self.ix_addrs.get(dst_addr) {
                    writeln!(self.asm, "    push ix").unwrap();
                    writeln!(self.asm, "    pop hl").unwrap();
                    if disp != 0 {
                        writeln!(self.asm, "    ld de, 0x{:04X}", disp as i16 as u16).unwrap();
                        writeln!(self.asm, "    add hl, de").unwrap();
                    }
                } else {
                    self.emit_load_hl_from(*dst_addr);
                }
                if *elem_size <= 4 {
                    writeln!(self.asm, "    ld d, h").unwrap();
                    writeln!(self.asm, "    ld e, l").unwrap();
                    for _ in 0..*elem_size {
                        writeln!(self.asm, "    inc de").unwrap();
                    }
                } else {
                    writeln!(self.asm, "    push hl").unwrap();
                    writeln!(self.asm, "    ld de, 0x{elem_size:04X}").unwrap();
                    writeln!(self.asm, "    add hl, de").unwrap();
                    writeln!(self.asm, "    ex de, hl").unwrap();
                    writeln!(self.asm, "    pop hl").unwrap();
                }
                let copy_bytes = (*count - 1) * *elem_size;
                writeln!(self.asm, "    ld bc, 0x{copy_bytes:04X}").unwrap();
                writeln!(self.asm, "    ldir").unwrap();
                self.val_in_reg
                    .retain(|_, &mut vr| !RegMask::BC.intersects(vr.mask()));
            }
            IrInst::MakeSlice { dst, ptr, len, .. } => {
                let da = self.val_addr(*dst);
                let da_lo = self.fmt_ram_mem(da);
                let da_hi = self.fmt_ram_mem(da.wrapping_add(2));
                if let Some(&disp) = self.ix_addrs.get(ptr) {
                    writeln!(self.asm, "    push ix").unwrap();
                    writeln!(self.asm, "    pop hl").unwrap();
                    if disp != 0 {
                        writeln!(self.asm, "    ld de, 0x{:04X}", disp as i16 as u16).unwrap();
                        writeln!(self.asm, "    add hl, de").unwrap();
                    }
                } else {
                    self.emit_load_hl_from(*ptr);
                }
                writeln!(self.asm, "    ld {da_lo}, hl").unwrap();
                self.emit_load_hl_from(*len);
                writeln!(self.asm, "    ld {da_hi}, hl").unwrap();
            }
            IrInst::SlicePtr { dst, slice, .. } => {
                let va = self.val_loc(*slice);
                let sa_lo = self.fmt_ram_mem(va);
                let da_mem = self.fmt_dst_mem(*dst);
                writeln!(self.asm, "    ld hl, {sa_lo}").unwrap();
                writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
            }
            IrInst::SliceLen { dst, slice } => {
                let va = self.val_loc(*slice);
                let sa_hi = self.fmt_ram_mem(va.wrapping_add(2));
                let da_mem = self.fmt_dst_mem(*dst);
                writeln!(self.asm, "    ld hl, {sa_hi}").unwrap();
                writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
            }
            IrInst::UnOp {
                dst, op, ty, val, ..
            } => {
                let da = self.val_addr(*dst);
                let da_mem = self.fmt_dst_mem(*dst);
                match (op, ty, self.program.type_size(ty)) {
                    (IrUnOp::Not, IrType::Bool, _) => {
                        self.emit_load_a_from(*val);
                        writeln!(self.asm, "    xor 1").unwrap();
                        writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                    }
                    (IrUnOp::Not, _, 1) => {
                        self.emit_load_a_from(*val);
                        writeln!(self.asm, "    cpl").unwrap();
                        writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                    }
                    (IrUnOp::Not, _, 2) => {
                        self.emit_load_hl_from(*val);
                        writeln!(self.asm, "    ld a, l").unwrap();
                        writeln!(self.asm, "    cpl").unwrap();
                        writeln!(self.asm, "    ld l, a").unwrap();
                        writeln!(self.asm, "    ld a, h").unwrap();
                        writeln!(self.asm, "    cpl").unwrap();
                        writeln!(self.asm, "    ld h, a").unwrap();
                        writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                    }
                    (IrUnOp::Not, _, 4) => {
                        self.emit_load_hlde_from(*val);
                        for r in ["e", "d", "l", "h"] {
                            writeln!(self.asm, "    ld a, {r}").unwrap();
                            writeln!(self.asm, "    cpl").unwrap();
                            writeln!(self.asm, "    ld {r}, a").unwrap();
                        }
                        self.emit_store_hlde_to(da, true);
                    }
                    (IrUnOp::Neg, _, 1) => {
                        self.emit_load_a_from(*val);
                        writeln!(self.asm, "    neg").unwrap();
                        writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                    }
                    (IrUnOp::Neg, _, 2) => {
                        self.emit_load_hl_from(*val);
                        writeln!(self.asm, "    xor a").unwrap();
                        writeln!(self.asm, "    sub l").unwrap();
                        writeln!(self.asm, "    ld l, a").unwrap();
                        writeln!(self.asm, "    sbc a, a").unwrap();
                        writeln!(self.asm, "    sub h").unwrap();
                        writeln!(self.asm, "    ld h, a").unwrap();
                        writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                    }
                    (IrUnOp::Neg, _, 4) => {
                        self.emit_load_hlde_from(*val);
                        writeln!(self.asm, "    xor a").unwrap();
                        writeln!(self.asm, "    sub e").unwrap();
                        writeln!(self.asm, "    ld e, a").unwrap();
                        for r in ["d", "l", "h"] {
                            writeln!(self.asm, "    ld a, 0").unwrap();
                            writeln!(self.asm, "    sbc a, {r}").unwrap();
                            writeln!(self.asm, "    ld {r}, a").unwrap();
                        }
                        self.emit_store_hlde_to(da, true);
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
                let da_mem = self.fmt_dst_mem(*dst);
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
                        let sa_mem = self.fmt_val_mem(src_v);
                        writeln!(self.asm, "    ld hl, {sa_mem}").unwrap();
                        writeln!(self.asm, "    add hl, hl").unwrap();
                        writeln!(self.asm, "    add hl, hl").unwrap();
                        writeln!(self.asm, "    ld a, h").unwrap();
                    }
                    writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                    return Ok(());
                }
                let from_sz = self.program.type_size(from_ty);
                let to_sz = self.program.type_size(to_ty);
                if from_sz == 2 && to_sz == 2 {
                    if let Some(&ca) = self.const_addrs.get(val) {
                        self.const_addrs.insert(*dst, ca);
                        if is_const_addr_always_folded(
                            self.program,
                            self.func,
                            *dst,
                            self.fused_skip_insts,
                        ) {
                            return Ok(());
                        }
                    } else if let Some(ra) = self.rom_addrs.get(val).cloned() {
                        self.rom_addrs.insert(*dst, ra);
                        if is_const_addr_always_folded(
                            self.program,
                            self.func,
                            *dst,
                            self.fused_skip_insts,
                        ) {
                            return Ok(());
                        }
                    }
                }
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
                        writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                    }
                    (1, 2) => {
                        if !is_signed_ty(from_ty)
                            && let Some(&reg) = self.val_in_reg.get(val)
                        {
                            writeln!(self.asm, "    ld l, {}", reg.reg8_name()).unwrap();
                            writeln!(self.asm, "    ld h, 0").unwrap();
                        } else if !is_signed_ty(from_ty)
                            && let Some(&(disp, _)) = self.ix_val_loads.get(val)
                        {
                            writeln!(self.asm, "    ld l, (ix + {disp})").unwrap();
                            writeln!(self.asm, "    ld h, 0").unwrap();
                        } else {
                            self.emit_load_a_from(*val);
                            writeln!(self.asm, "    ld l, a").unwrap();
                            if is_signed_ty(from_ty) {
                                writeln!(self.asm, "    add a, a").unwrap();
                                writeln!(self.asm, "    sbc a, a").unwrap();
                                writeln!(self.asm, "    ld h, a").unwrap();
                            } else {
                                writeln!(self.asm, "    ld h, 0").unwrap();
                            }
                        }
                        writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
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
                            writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                        } else {
                            // Little-endian: low byte is at va / (ix + disp)
                            self.emit_load_a_from(*val);
                            writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                        }
                    }
                    (2, 2) => {
                        self.emit_load_hl_from(*val);
                        writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                    }
                    (1, 4) => {
                        self.emit_load_a_from(*val);
                        writeln!(self.asm, "    ld e, a").unwrap();
                        if is_signed_ty(from_ty) {
                            writeln!(self.asm, "    add a, a").unwrap();
                            writeln!(self.asm, "    sbc a, a").unwrap();
                        } else {
                            writeln!(self.asm, "    xor a").unwrap();
                        }
                        writeln!(self.asm, "    ld d, a").unwrap();
                        writeln!(self.asm, "    ld l, a").unwrap();
                        writeln!(self.asm, "    ld h, a").unwrap();
                        self.emit_store_hlde_to(da, true);
                    }
                    (2, 4) => {
                        self.emit_load_de_from(*val);
                        if is_signed_ty(from_ty) {
                            writeln!(self.asm, "    ld a, d").unwrap();
                            writeln!(self.asm, "    add a, a").unwrap();
                            writeln!(self.asm, "    sbc a, a").unwrap();
                        } else {
                            writeln!(self.asm, "    xor a").unwrap();
                        }
                        writeln!(self.asm, "    ld l, a").unwrap();
                        writeln!(self.asm, "    ld h, a").unwrap();
                        self.emit_store_hlde_to(da, true);
                    }
                    (4, 1) => {
                        if *to_ty == IrType::Bool {
                            let lbl = self.fresh_label("cast_bool");
                            self.emit_load_hlde_from(*val);
                            writeln!(self.asm, "    ld a, e").unwrap();
                            writeln!(self.asm, "    or d").unwrap();
                            writeln!(self.asm, "    or l").unwrap();
                            writeln!(self.asm, "    or h").unwrap();
                            writeln!(self.asm, "    jr z, {lbl}").unwrap();
                            writeln!(self.asm, "    ld a, 1").unwrap();
                            writeln!(self.asm, "{lbl}:").unwrap();
                            writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                        } else {
                            self.emit_load_a_from(*val);
                            writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                        }
                    }
                    (4, 2) => {
                        self.emit_load_hl_from(*val);
                        writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                    }
                    (4, 4) => {
                        if let Some(&c) = self.const_vals.get(val) {
                            self.emit_const32_to(da, true, c);
                        } else {
                            self.emit_copy32(self.val_loc(*val), true, da, true);
                        }
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
                let va_mem = self.fmt_val_mem(*val);
                let da_mem = self.fmt_dst_mem(*dst);
                match (base.byte_size(), *frac_bits) {
                    (2, 8) => {
                        // q8_8::int() -> direct high-byte read (0T shift!), sign/zero extended to 16-bit
                        let va_hi_mem = self.fmt_ram_mem(va.wrapping_add(1));
                        writeln!(self.asm, "    ld a, {va_hi_mem}").unwrap();
                        writeln!(self.asm, "    ld l, a").unwrap();
                        if base.is_signed() {
                            writeln!(self.asm, "    add a, a").unwrap();
                            writeln!(self.asm, "    sbc a, a").unwrap();
                            writeln!(self.asm, "    ld h, a").unwrap();
                        } else {
                            writeln!(self.asm, "    ld h, 0").unwrap();
                        }
                        writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                    }
                    (2, 6) => {
                        // q10_6::int() -> exact 16-bit shift right by 6 via 2-bit left shift of A:H:L
                        writeln!(self.asm, "    ld hl, {va_mem}").unwrap();
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
                        writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                    }
                    (2, 0) => {
                        writeln!(self.asm, "    ld hl, {va_mem}").unwrap();
                        writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                    }
                    (2, fb) => {
                        let lbl = self.fresh_label("fxint");
                        writeln!(self.asm, "    ld hl, {va_mem}").unwrap();
                        writeln!(self.asm, "    ld b, {fb}").unwrap();
                        writeln!(self.asm, "{lbl}:").unwrap();
                        if base.is_signed() {
                            writeln!(self.asm, "    sra h").unwrap();
                        } else {
                            writeln!(self.asm, "    srl h").unwrap();
                        }
                        writeln!(self.asm, "    rr l").unwrap();
                        writeln!(self.asm, "    djnz {lbl}").unwrap();
                        writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                    }
                    (1, 0) => {
                        writeln!(self.asm, "    ld a, {va_mem}").unwrap();
                        writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                    }
                    (1, fb) => {
                        let lbl = self.fresh_label("fxint");
                        writeln!(self.asm, "    ld a, {va_mem}").unwrap();
                        writeln!(self.asm, "    ld b, {fb}").unwrap();
                        writeln!(self.asm, "{lbl}:").unwrap();
                        if base.is_signed() {
                            writeln!(self.asm, "    sra a").unwrap();
                        } else {
                            writeln!(self.asm, "    srl a").unwrap();
                        }
                        writeln!(self.asm, "    djnz {lbl}").unwrap();
                        writeln!(self.asm, "    ld {da_mem}, a").unwrap();
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
                        self.emit_load_a_from(*val);
                    }
                    writeln!(self.asm, "    out (0x{:02X}), a", (p & 0xFF) as u8).unwrap();
                } else {
                    let pa_mem = self.fmt_val_mem(*port);
                    let va_mem = self.fmt_val_mem(*val);
                    writeln!(self.asm, "    ld a, {pa_mem}").unwrap();
                    writeln!(self.asm, "    ld c, a").unwrap();
                    writeln!(self.asm, "    ld a, {va_mem}").unwrap();
                    writeln!(self.asm, "    out (c), a").unwrap();
                }
            }
            IrInst::PortIn { dst, port } => {
                let da_mem = self.fmt_dst_mem(*dst);
                if let Some(&p) = self.const_vals.get(port) {
                    writeln!(self.asm, "    in a, (0x{:02X})", (p & 0xFF) as u8).unwrap();
                } else {
                    let pa_mem = self.fmt_val_mem(*port);
                    writeln!(self.asm, "    ld a, {pa_mem}").unwrap();
                    writeln!(self.asm, "    ld c, a").unwrap();
                    writeln!(self.asm, "    in a, (c)").unwrap();
                }
                writeln!(self.asm, "    ld {da_mem}, a").unwrap();
            }
            IrInst::Call {
                dst,
                func: callee_id,
                args,
            } => {
                let cidx = *callee_id as usize;
                let callee_fa = &self.analysis.fns[cidx];
                let reg_p1 = callee_fa.uses_reg_param1;
                let ix_reg_slot = if callee_fa.ix_in_reg {
                    callee_fa.ix_param_slot
                } else {
                    None
                };
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
                        let pa_mem = self.fmt_ram_mem(pa);
                        if self.ix_addrs.get(&arg_val) == Some(&0) {
                            writeln!(self.asm, "    ld {pa_mem}, ix").unwrap();
                            continue;
                        }
                        match psz {
                            1 => {
                                self.emit_load_a_from(arg_val);
                                writeln!(self.asm, "    ld {pa_mem}, a").unwrap();
                            }
                            2 => {
                                self.emit_load_hl_from(arg_val);
                                writeln!(self.asm, "    ld {pa_mem}, hl").unwrap();
                            }
                            4 => {
                                if let Some(&c) = self.const_vals.get(&arg_val) {
                                    self.emit_const32_to(pa, true, c);
                                } else {
                                    self.emit_copy32(self.val_loc(arg_val), true, pa, true);
                                }
                            }
                            _ => {}
                        }
                    }
                }
                // Pass argument 1 in E (1-byte) or DE (2-byte) when eligible and not passed in IX
                if reg_p1
                    && ix_reg_slot != Some(1)
                    && let Some(&a1) = args.get(1)
                {
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
                                    } else if let Some(&reg) = self.val_in_reg.get(&a1) {
                                        writeln!(self.asm, "    ld e, {}", reg.reg8_name()).unwrap();
                                    } else if let Some(&(disp, _)) = self.ix_val_loads.get(&a1) {
                                        writeln!(self.asm, "    ld e, (ix + {disp})").unwrap();
                                    } else {
                                        let aa_mem = self.fmt_val_mem(a1);
                                        writeln!(self.asm, "    ld a, {aa_mem}").unwrap();
                                        writeln!(self.asm, "    ld e, a").unwrap();
                                    }
                                }
                                2 => self.emit_load_de_from(a1),
                                _ => {}
                            }
                        }
                    }
                }
                // Pass argument 0 in A (1-byte) or HL (2-byte) when not passed in IX
                if ix_reg_slot != Some(0)
                    && let Some(&a0) = args.first()
                {
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
                                4 => self.emit_load_hlde_from(a0),
                                _ => {}
                            }
                        }
                    }
                }
                // Pass shared-IX argument directly in IX (without clobbering already-loaded A/HL or E/DE)
                let mut caller_saved_ix = false;
                if let Some(p) = ix_reg_slot
                    && let Some(&ix_arg) = args.get(p as usize)
                    && self.ix_addrs.get(&ix_arg) != Some(&0)
                {
                    if self.fa.ix_param_slot.is_some() {
                        writeln!(self.asm, "    push ix").unwrap();
                        caller_saved_ix = true;
                    }
                    if let Some(&(addr, is_u)) = self.const_addrs.get(&ix_arg) {
                        let imm = self.fmt_addr_imm(addr, is_u);
                        writeln!(self.asm, "    ld ix, {imm}").unwrap();
                    } else if let Some((sym, off)) = self.rom_addrs.get(&ix_arg) {
                        if *off == 0 {
                            writeln!(self.asm, "    ld ix, {sym}").unwrap();
                        } else {
                            writeln!(self.asm, "    ld ix, {sym} + {off}").unwrap();
                        }
                    } else if let Some(&c) = self.const_vals.get(&ix_arg) {
                        writeln!(self.asm, "    ld ix, 0x{:04X}", c as u16).unwrap();
                    } else if let Some(&disp) = self.ix_addrs.get(&ix_arg) {
                        if p == 0 {
                            writeln!(self.asm, "    push ix").unwrap();
                            writeln!(self.asm, "    pop hl").unwrap();
                            writeln!(self.asm, "    push de").unwrap();
                            writeln!(self.asm, "    ld de, 0x{:04X}", disp as i16 as u16).unwrap();
                            writeln!(self.asm, "    add hl, de").unwrap();
                            writeln!(self.asm, "    pop de").unwrap();
                            writeln!(self.asm, "    push hl").unwrap();
                            writeln!(self.asm, "    pop ix").unwrap();
                        } else {
                            writeln!(self.asm, "    push hl").unwrap();
                            writeln!(self.asm, "    push ix").unwrap();
                            writeln!(self.asm, "    pop hl").unwrap();
                            writeln!(self.asm, "    ld de, 0x{:04X}", disp as i16 as u16).unwrap();
                            writeln!(self.asm, "    add hl, de").unwrap();
                            writeln!(self.asm, "    ex de, hl").unwrap();
                            writeln!(self.asm, "    pop hl").unwrap();
                            writeln!(self.asm, "    push de").unwrap();
                            writeln!(self.asm, "    pop ix").unwrap();
                        }
                    } else if self.val_in_reg.get(&ix_arg) == Some(&RegLoc::BC) {
                        writeln!(self.asm, "    push bc").unwrap();
                        writeln!(self.asm, "    pop ix").unwrap();
                    } else if let Some(&(disp, _)) = self.ix_val_loads.get(&ix_arg) {
                        if p == 0 {
                            writeln!(self.asm, "    ld l, (ix + {disp})").unwrap();
                            writeln!(self.asm, "    ld h, (ix + {})", disp + 1).unwrap();
                            writeln!(self.asm, "    push hl").unwrap();
                            writeln!(self.asm, "    pop ix").unwrap();
                        } else {
                            writeln!(self.asm, "    ld e, (ix + {disp})").unwrap();
                            writeln!(self.asm, "    ld d, (ix + {})", disp + 1).unwrap();
                            writeln!(self.asm, "    push de").unwrap();
                            writeln!(self.asm, "    pop ix").unwrap();
                        }
                    } else {
                        let va_mem = self.fmt_val_mem(ix_arg);
                        writeln!(self.asm, "    ld ix, {va_mem}").unwrap();
                    }
                }
                let live_bc = self.fa.reg_alloc.call_live_mask(bb_idx, inst_idx);
                let callee_clob = self
                    .analysis
                    .fn_clobbers
                    .get(cidx)
                    .copied()
                    .unwrap_or(RegMask::BC);
                let need_save_bc = live_bc.intersects(callee_clob);
                if need_save_bc {
                    writeln!(self.asm, "    push bc").unwrap();
                }
                writeln!(
                    self.asm,
                    "    call {}",
                    z80_sym(&self.program.functions[cidx].name)
                )
                .unwrap();
                if need_save_bc {
                    writeln!(self.asm, "    pop bc").unwrap();
                } else {
                    self.val_in_reg
                        .retain(|_, &mut vr| !callee_clob.intersects(vr.mask()));
                }
                if caller_saved_ix {
                    writeln!(self.asm, "    pop ix").unwrap();
                }
                if let Some(d) = dst {
                    let da = self.val_addr(*d);
                    let da_mem = self.fmt_dst_mem(*d);
                    let ret_sz = self.program.type_size(&self.program.functions[cidx].ret_ty);
                    match ret_sz {
                        1 => writeln!(self.asm, "    ld {da_mem}, a").unwrap(),
                        2 => writeln!(self.asm, "    ld {da_mem}, hl").unwrap(),
                        4 => self.emit_store_hlde_to(da, true),
                        _ => {}
                    }
                }
            }
            IrInst::CallIndirect { addr, args } => {
                let direct_target = if let Some(&c) = self.const_vals.get(addr) {
                    Some(format!("0x{:04X}", c as u16))
                } else if let Some(&(ca, is_u)) = self.const_addrs.get(addr) {
                    Some(self.fmt_addr_imm(ca, is_u))
                } else if let Some((sym, off)) = self.rom_addrs.get(addr) {
                    if *off == 0 {
                        Some(sym.clone())
                    } else {
                        Some(format!("{sym} + {off}"))
                    }
                } else {
                    None
                };
                if direct_target.is_none() {
                    self.runtime_needs.jp_iy = true;
                    self.emit_load_hl_from(*addr);
                    writeln!(self.asm, "    push hl").unwrap();
                    writeln!(self.asm, "    pop iy").unwrap();
                }
                if let Some((a1, t1)) = args.get(1) {
                    let sz = self.program.type_size(t1);
                    if self.ix_addrs.get(a1) == Some(&0) {
                        writeln!(self.asm, "    push ix").unwrap();
                        writeln!(self.asm, "    pop de").unwrap();
                    } else {
                        match sz {
                            1 => {
                                if let Some(&c) = self.const_vals.get(a1) {
                                    writeln!(self.asm, "    ld e, 0x{:02X}", (c & 0xFF) as u8)
                                        .unwrap();
                                } else if let Some(&reg) = self.val_in_reg.get(a1) {
                                    writeln!(self.asm, "    ld e, {}", reg.reg8_name()).unwrap();
                                } else if let Some(&(disp, _)) = self.ix_val_loads.get(a1) {
                                    writeln!(self.asm, "    ld e, (ix + {disp})").unwrap();
                                } else {
                                    self.emit_load_a_from(*a1);
                                    writeln!(self.asm, "    ld e, a").unwrap();
                                }
                            }
                            2 => self.emit_load_de_from(*a1),
                            _ => {}
                        }
                    }
                }
                if let Some((a0, t0)) = args.first() {
                    let sz = self.program.type_size(t0);
                    if self.ix_addrs.get(a0) == Some(&0) {
                        writeln!(self.asm, "    push ix").unwrap();
                        writeln!(self.asm, "    pop hl").unwrap();
                    } else {
                        match sz {
                            1 => self.emit_load_a_from(*a0),
                            2 => self.emit_load_hl_from(*a0),
                            4 => self.emit_load_hlde_from(*a0),
                            _ => {}
                        }
                    }
                }
                let live_bc = self.fa.reg_alloc.call_live_mask(bb_idx, inst_idx);
                let need_save_bc = live_bc.intersects(RegMask::BC);
                if need_save_bc {
                    writeln!(self.asm, "    push bc").unwrap();
                }
                if let Some(target) = direct_target {
                    writeln!(self.asm, "    call {target}").unwrap();
                } else {
                    writeln!(self.asm, "    call __aun_jp_iy").unwrap();
                }
                if need_save_bc {
                    writeln!(self.asm, "    pop bc").unwrap();
                } else {
                    self.val_in_reg
                        .retain(|_, &mut vr| !RegMask::BC.intersects(vr.mask()));
                }
            }
            IrInst::ExternCall {
                dst,
                extern_fn,
                args,
            } => {
                let ef_idx = *extern_fn as usize;
                let param_count = self.program.externs[ef_idx].params.len();
                let live_bc = self.fa.reg_alloc.call_live_mask(bb_idx, inst_idx);
                let ext_clob = extern_clobber_mask(&self.program.externs[ef_idx]);
                let need_save_bc = live_bc.intersects(ext_clob);

                let mut stack_args: Vec<(ValueId, u16)> = Vec::new();
                for (arg_val, i) in args.iter().zip(0..param_count) {
                    let (psz, is_stack) = {
                        let (_, pty, reg) = &self.program.externs[ef_idx].params[i];
                        (self.program.type_size(pty), reg.eq_ignore_ascii_case("stack"))
                    };
                    if is_stack {
                        stack_args.push((*arg_val, psz));
                    }
                }
                let has_stack_args = !stack_args.is_empty();
                // Callee cleans stacked args in SDCC --sdcccall(1), so save live BC underneath them.
                if has_stack_args && need_save_bc {
                    writeln!(self.asm, "    push bc").unwrap();
                }
                for (arg_val, psz) in stack_args.into_iter().rev() {
                    match psz {
                        1 => {
                            self.emit_load_a_from(arg_val);
                            writeln!(self.asm, "    push af").unwrap();
                            writeln!(self.asm, "    inc sp").unwrap();
                        }
                        2 => {
                            if self.ix_addrs.get(&arg_val) == Some(&0) {
                                writeln!(self.asm, "    push ix").unwrap();
                            } else {
                                self.emit_load_hl_from(arg_val);
                                writeln!(self.asm, "    push hl").unwrap();
                            }
                        }
                        4 => {
                            self.emit_load_hlde_from(arg_val);
                            writeln!(self.asm, "    push hl").unwrap();
                            writeln!(self.asm, "    push de").unwrap();
                        }
                        _ => {}
                    }
                }

                // Load non-A arguments first so `ld a, (addr); ld r, a` never clobbers an `in A` argument
                let mut deferred_a: Option<ValueId> = None;
                let mut deferred_hlde: Option<ValueId> = None;
                for (arg_val, i) in args.iter().zip(0..param_count) {
                    let (psz, r) = {
                        let (_, pty, reg) = &self.program.externs[ef_idx].params[i];
                        (self.program.type_size(pty), reg.to_ascii_lowercase())
                    };
                    if r == "stack" {
                        continue;
                    }
                    match psz {
                        1 => {
                            if r == "a" {
                                deferred_a = Some(*arg_val);
                            } else if let Some(&c) = self.const_vals.get(arg_val) {
                                writeln!(self.asm, "    ld {r}, 0x{:02X}", (c & 0xFF) as u8)
                                    .unwrap();
                            } else if let Some(&reg) = self.val_in_reg.get(arg_val) {
                                writeln!(self.asm, "    ld {r}, {}", reg.reg8_name()).unwrap();
                            } else if let Some(&(disp, _)) = self.ix_val_loads.get(arg_val) {
                                writeln!(self.asm, "    ld {r}, (ix + {disp})").unwrap();
                            } else {
                                let aa_mem = self.fmt_val_mem(*arg_val);
                                writeln!(self.asm, "    ld a, {aa_mem}").unwrap();
                                writeln!(self.asm, "    ld {r}, a").unwrap();
                            }
                        }
                        2 => {
                            if let Some(&c) = self.const_vals.get(arg_val) {
                                writeln!(self.asm, "    ld {r}, 0x{:04X}", c as u16).unwrap();
                            } else if let Some(&(addr, is_u)) = self.const_addrs.get(arg_val) {
                                let imm = self.fmt_addr_imm(addr, is_u);
                                writeln!(self.asm, "    ld {r}, {imm}").unwrap();
                            } else if let Some((sym, off)) = self.rom_addrs.get(arg_val) {
                                if *off == 0 {
                                    writeln!(self.asm, "    ld {r}, {sym}").unwrap();
                                } else {
                                    writeln!(self.asm, "    ld {r}, {sym} + {off}").unwrap();
                                }
                            } else if self.val_in_reg.get(arg_val) == Some(&RegLoc::BC)
                                && r == "hl"
                            {
                                writeln!(self.asm, "    ld h, b").unwrap();
                                writeln!(self.asm, "    ld l, c").unwrap();
                            } else if self.val_in_reg.get(arg_val) == Some(&RegLoc::BC)
                                && r == "de"
                            {
                                writeln!(self.asm, "    ld d, b").unwrap();
                                writeln!(self.asm, "    ld e, c").unwrap();
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
                                let aa_mem = self.fmt_val_mem(*arg_val);
                                writeln!(self.asm, "    ld {r}, {aa_mem}").unwrap();
                            }
                        }
                        4 if r == "hlde" => {
                            deferred_hlde = Some(*arg_val);
                        }
                        _ => {}
                    }
                }
                if let Some(hv) = deferred_hlde {
                    self.emit_load_hlde_from(hv);
                }
                if let Some(av) = deferred_a {
                    self.emit_load_a_from(av);
                }
                let ef = &self.program.externs[ef_idx];
                if !has_stack_args && need_save_bc {
                    writeln!(self.asm, "    push bc").unwrap();
                }
                if let Some(fixed) = ef.fixed_addr {
                    writeln!(self.asm, "    call 0x{fixed:04X}").unwrap();
                } else {
                    writeln!(self.asm, "    call {}", ef.name).unwrap();
                }
                if need_save_bc {
                    writeln!(self.asm, "    pop bc").unwrap();
                } else {
                    self.val_in_reg
                        .retain(|_, &mut vr| !ext_clob.intersects(vr.mask()));
                }
                if let (Some(d), Some((rty, rreg))) = (dst, &ef.ret) {
                    let da = self.val_addr(*d);
                    let da_mem = self.fmt_dst_mem(*d);
                    let r = rreg.to_ascii_lowercase();
                    match self.program.type_size(rty) {
                        1 => {
                            if r != "a" {
                                writeln!(self.asm, "    ld a, {r}").unwrap();
                            }
                            writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                        }
                        2 => writeln!(self.asm, "    ld {da_mem}, {r}").unwrap(),
                        4 if r == "hlde" => self.emit_store_hlde_to(da, true),
                        _ => {}
                    }
                }
            }
            IrInst::Assert { cond, site_id } => {
                if !self.opts.release {
                    self.runtime_needs.panic_stub = true;
                    let ca_mem = self.fmt_val_mem(*cond);
                    let ok_lbl = self.fresh_label("assert_ok");
                    writeln!(self.asm, "    ld a, {ca_mem}").unwrap();
                    writeln!(self.asm, "    or a").unwrap();
                    writeln!(self.asm, "    jr nz, {ok_lbl}").unwrap();
                    writeln!(self.asm, "    ld a, {}", PanicCode::AssertionFailed as u8).unwrap();
                    writeln!(self.asm, "    ld hl, 0x{site_id:04X}").unwrap();
                    writeln!(self.asm, "    call __aun_panic").unwrap();
                    writeln!(self.asm, "{ok_lbl}:").unwrap();
                }
            }
            IrInst::Panic { code, site_id } => {
                self.runtime_needs.panic_stub = true;
                writeln!(self.asm, "    ld a, {}", *code as u8).unwrap();
                writeln!(self.asm, "    ld hl, 0x{site_id:04X}").unwrap();
                writeln!(self.asm, "    call __aun_panic").unwrap();
            }
            IrInst::StateTag {
                dst,
                machine_addr,
                machine_id,
            } => {
                let ma_mem = self.fmt_val_mem(*machine_addr);
                let da_mem = self.fmt_dst_mem(*dst);
                let mdef = &self.program.machines[*machine_id as usize];
                writeln!(self.asm, "    ld hl, {ma_mem}").unwrap();
                if mdef.tag_offset > 0 {
                    writeln!(self.asm, "    ld de, 0x{:04X}", mdef.tag_offset).unwrap();
                    writeln!(self.asm, "    add hl, de").unwrap();
                }
                writeln!(self.asm, "    ld a, (hl)").unwrap();
                writeln!(self.asm, "    ld {da_mem}, a").unwrap();
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
                                let aa_mem = self.fmt_val_mem(arg_val);
                                let pa_mem = self.fmt_ram_mem(
                                    self.layout.slot_addrs[&(callee.id, param_slot.id)],
                                );
                                match self.program.type_size(&param_slot.ty) {
                                    1 => {
                                        writeln!(self.asm, "    ld a, {aa_mem}").unwrap();
                                        writeln!(self.asm, "    ld {pa_mem}, a").unwrap();
                                    }
                                    2 => {
                                        writeln!(self.asm, "    ld hl, {aa_mem}").unwrap();
                                        writeln!(self.asm, "    ld {pa_mem}, hl").unwrap();
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                let ma_mem = self.fmt_val_mem(*machine_addr);
                writeln!(self.asm, "    ld hl, {ma_mem}").unwrap();
                writeln!(self.asm, "    call {}_disp_{method}", mdef.name).unwrap();
                if let Some(d) = dst {
                    let da_mem = self.fmt_dst_mem(*d);
                    writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
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
                let ma_mem = self.fmt_val_mem(*machine_addr);
                let ok_lbl = if from_state.is_none() && self.opts.check_transitions {
                    self.runtime_needs.panic_stub = true;
                    Some(self.fresh_label("tr_ok"))
                } else {
                    None
                };
                let mdef = &self.program.machines[*machine_id as usize];

                // 1. Dynamic transition verification when `from_state` is unknown (eliminated in Release!)
                if let Some(ok_lbl) = ok_lbl {
                    writeln!(self.asm, "    ld hl, {ma_mem}").unwrap();
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
                        writeln!(self.asm, "    ld hl, {ma_mem}").unwrap();
                        writeln!(self.asm, "    call {ename}").unwrap();
                    }
                } else if mdef.states.iter().any(|s| s.on_exit.is_some()) {
                    writeln!(self.asm, "    ld hl, {ma_mem}").unwrap();
                    writeln!(self.asm, "    call {}_disp_on_exit", mdef.name).unwrap();
                }

                // 3. Write pre-evaluated payload fields into overlaid RAM region
                for (rel_off, vid, fty) in field_vals {
                    let fsz = self.program.type_size(fty);
                    if fsz == 0 {
                        continue;
                    }
                    let total_off = mdef.payload_offset + *rel_off;
                    writeln!(self.asm, "    ld hl, {ma_mem}").unwrap();
                    if total_off > 0 {
                        writeln!(self.asm, "    ld de, 0x{total_off:04X}").unwrap();
                        writeln!(self.asm, "    add hl, de").unwrap();
                    }
                    if !fty.is_scalar() {
                        // Aggregate payload (`[T; N]`, `struct`, child `machine`): `vid` holds source address.
                        writeln!(self.asm, "    push hl").unwrap();
                        if let Some(&disp) = self.ix_addrs.get(vid) {
                            writeln!(self.asm, "    push ix").unwrap();
                            writeln!(self.asm, "    pop hl").unwrap();
                            if disp != 0 {
                                writeln!(self.asm, "    ld de, 0x{:04X}", disp as i16 as u16)
                                    .unwrap();
                                writeln!(self.asm, "    add hl, de").unwrap();
                            }
                        } else {
                            self.emit_load_hl_from(*vid);
                        }
                        writeln!(self.asm, "    pop de").unwrap();
                        writeln!(self.asm, "    ld bc, 0x{fsz:04X}").unwrap();
                        writeln!(self.asm, "    ldir").unwrap();
                        continue;
                    }
                    match fsz {
                        1 => {
                            let va_mem = self.fmt_val_mem(*vid);
                            writeln!(self.asm, "    ld a, {va_mem}").unwrap();
                            writeln!(self.asm, "    ld (hl), a").unwrap();
                        }
                        2 => {
                            let va_mem = self.fmt_val_mem(*vid);
                            writeln!(self.asm, "    ld de, {va_mem}").unwrap();
                            writeln!(self.asm, "    ld (hl), e").unwrap();
                            writeln!(self.asm, "    inc hl").unwrap();
                            writeln!(self.asm, "    ld (hl), d").unwrap();
                        }
                        4 => {
                            if let Some(&c) = self.const_vals.get(vid) {
                                let bytes = (c as u32).to_le_bytes();
                                for (idx, b) in bytes.into_iter().enumerate() {
                                    writeln!(self.asm, "    ld (hl), 0x{b:02X}").unwrap();
                                    if idx < 3 {
                                        writeln!(self.asm, "    inc hl").unwrap();
                                    }
                                }
                            } else {
                                let va = self.val_loc(*vid);
                                let m0 = self.fmt_ram_mem(va);
                                let m2 = self.fmt_ram_mem(va.wrapping_add(2));
                                writeln!(self.asm, "    ld de, {m0}").unwrap();
                                writeln!(self.asm, "    ld (hl), e").unwrap();
                                writeln!(self.asm, "    inc hl").unwrap();
                                writeln!(self.asm, "    ld (hl), d").unwrap();
                                writeln!(self.asm, "    inc hl").unwrap();
                                writeln!(self.asm, "    ld de, {m2}").unwrap();
                                writeln!(self.asm, "    ld (hl), e").unwrap();
                                writeln!(self.asm, "    inc hl").unwrap();
                                writeln!(self.asm, "    ld (hl), d").unwrap();
                            }
                        }
                        _ => {}
                    }
                }

                // 4. Write new state tag
                writeln!(self.asm, "    ld hl, {ma_mem}").unwrap();
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
                    writeln!(self.asm, "    ld hl, {ma_mem}").unwrap();
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
                let pa_mem = self.fmt_val_mem(*pool_addr);
                let ia_mem = self.fmt_val_mem(*init_addr);
                let da_mem = self.fmt_dst_mem(*dst);
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
                writeln!(self.asm, "    ld hl, {pa_mem}").unwrap();
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
                    writeln!(self.asm, "    ld de, {pa_mem}").unwrap();
                    writeln!(self.asm, "    add hl, de").unwrap();
                    writeln!(self.asm, "    ex de, hl").unwrap(); // DE = dst
                    writeln!(self.asm, "    ld hl, {ia_mem}").unwrap(); // HL = src
                    writeln!(self.asm, "    ld bc, 0x{elem_size:04X}").unwrap();
                    writeln!(self.asm, "    ldir").unwrap();
                    writeln!(self.asm, "    pop bc").unwrap();
                }
                writeln!(self.asm, "    ld a, c").unwrap();
                writeln!(self.asm, "{done_lbl}:").unwrap();
                writeln!(self.asm, "    ld {da_mem}, a").unwrap();
            }
            IrInst::PoolFree {
                pool_addr,
                handle,
                elem_size,
                capacity,
                site_id,
            } => {
                let pa_mem = self.fmt_val_mem(*pool_addr);
                let ha_mem = self.fmt_val_mem(*handle);
                let bm_off = elem_size * capacity;
                let sd = self.fresh_label("pfree_sd");
                let sl = self.fresh_label("pfree_sl");
                let ok_rng = if self.opts.check_bounds {
                    self.runtime_needs.panic_stub = true;
                    Some((
                        self.fresh_label("pfree_rng"),
                        self.fresh_label("pfree_live"),
                    ))
                } else {
                    None
                };
                writeln!(self.asm, "    ld a, {ha_mem}").unwrap();
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
                if let Some(&(paddr, is_u)) = self.const_addrs.get(pool_addr) {
                    let bm_imm = self.fmt_addr_imm(paddr.wrapping_add(bm_off), is_u);
                    writeln!(self.asm, "    ld hl, {bm_imm}").unwrap();
                    writeln!(self.asm, "    add hl, de").unwrap();
                } else {
                    writeln!(self.asm, "    ld hl, {pa_mem}").unwrap();
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
                let pa_mem = self.fmt_val_mem(*pool_addr);
                let ha_mem = self.fmt_val_mem(*handle);
                let da_mem = self.fmt_dst_mem(*dst);
                let bm_off = elem_size * capacity;
                let dbg_labels = if self.opts.check_bounds {
                    self.runtime_needs.panic_stub = true;
                    Some((
                        self.fresh_label("pget_rng"),
                        self.fresh_label("pget_live"),
                        self.fresh_label("pget_sd"),
                        self.fresh_label("pget_sl"),
                    ))
                } else {
                    None
                };
                writeln!(self.asm, "    ld a, {ha_mem}").unwrap();
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
                    if let Some(&(paddr, is_u)) = self.const_addrs.get(pool_addr) {
                        let bm_imm = self.fmt_addr_imm(paddr.wrapping_add(bm_off), is_u);
                        writeln!(self.asm, "    ld hl, {bm_imm}").unwrap();
                        writeln!(self.asm, "    add hl, de").unwrap();
                    } else {
                        writeln!(self.asm, "    ld hl, {pa_mem}").unwrap();
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
                if let Some(&(paddr, is_u)) = self.const_addrs.get(pool_addr) {
                    let p_imm = self.fmt_addr_imm(paddr, is_u);
                    writeln!(self.asm, "    ld de, {p_imm}").unwrap();
                } else {
                    writeln!(self.asm, "    ld de, {pa_mem}").unwrap();
                }
                writeln!(self.asm, "    add hl, de").unwrap();
                writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
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
                writeln!(self.asm, "; @aun:asm_begin").unwrap();
                for line in template.split(';') {
                    let t = line.trim();
                    if !t.is_empty() {
                        writeln!(self.asm, "    {t}").unwrap();
                    }
                }
                writeln!(self.asm, "; @aun:asm_end").unwrap();
            }
            IrInst::Phi { .. } => {
                return Err("unexpected Phi in Z80 codegen (run destruct_ssa first)".to_string());
            }
        }
        Ok(())
    }
}
