use super::runtime::{
    emit_bank_switch, emit_mul_hl_const, emit_mul_u8_const, is_small_u8_array_mul, ranges_overlap,
};
use super::{EmitCtx, HlBase, IndexKey};
use crate::analysis::{
    ConstAddrSrc, can_alias_const_addr_load, can_alias_ix_load, can_alias_local_load,
    can_alias_reg_load, is_const_addr_always_folded, is_const_always_folded, is_signed_ty, z80_sym,
};
use crate::regalloc::{
    extern_clobber_mask, inst_internal_clobber_mask, FnRegAlloc, RegLoc, RegMask,
};
use aun_ir::*;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write;

#[inline]
pub(super) fn mask_up_to(n: u16) -> u16 {
    if n == 0 {
        0
    } else {
        0xFFFFu16 >> n.leading_zeros()
    }
}

#[inline]
pub(super) fn type_possible_mask(program: &IrProgram, ty: &IrType) -> Option<u16> {
    if *ty == IrType::Bool {
        Some(1)
    } else {
        match program.type_size(ty) {
            1 => Some(0x00FF),
            2 => Some(0xFFFF),
            _ => None,
        }
    }
}

pub(super) fn eval_inst_possible_bits(
    program: &IrProgram,
    inst: &IrInst,
    u8_bound_vals: &HashSet<ValueId>,
    get_const: impl Fn(ValueId) -> Option<i64>,
    get_bits: impl Fn(ValueId) -> u16,
    get_slot_bits: impl Fn(SlotId, &IrType) -> Option<u16>,
) -> Option<u16> {
    match inst {
        IrInst::ConstInt { ty, val, .. } => {
            if *ty == IrType::Bool {
                Some(((*val != 0) as u16) & 1)
            } else {
                match program.type_size(ty) {
                    1 => Some((*val as u16) & 0x00FF),
                    2 => Some(*val as u16),
                    _ => None,
                }
            }
        }
        IrInst::ConstBool { val, .. } => Some(if *val { 1 } else { 0 }),
        IrInst::Cast {
            val,
            from_ty,
            to_ty,
            ..
        } => {
            let from_sz = program.type_size(from_ty);
            let to_sz = program.type_size(to_ty);
            if matches!(from_ty, IrType::Fixed { .. }) || matches!(to_ty, IrType::Fixed { .. }) {
                return type_possible_mask(program, to_ty);
            }
            if *to_ty == IrType::Bool {
                Some(1)
            } else if to_sz == 1 {
                Some(get_bits(*val) & 0x00FF)
            } else if to_sz == 2 {
                match from_sz {
                    1 if !is_signed_ty(from_ty) => Some(get_bits(*val) & 0x00FF),
                    1 => {
                        let vb = get_bits(*val) & 0x00FF;
                        Some(if (vb & 0x0080) == 0 { vb } else { 0xFFFF })
                    }
                    2 => Some(get_bits(*val)),
                    4 if u8_bound_vals.contains(val) => Some(0x00FF),
                    _ => Some(0xFFFF),
                }
            } else {
                None
            }
        }
        IrInst::BinOp {
            op,
            mode,
            ty,
            lhs,
            rhs,
            ..
        } => {
            let sz = program.type_size(ty);
            if matches!(
                op,
                IrBinOp::Eq | IrBinOp::Ne | IrBinOp::Lt | IrBinOp::Le | IrBinOp::Gt | IrBinOp::Ge
            ) || *ty == IrType::Bool
            {
                Some(1)
            } else if matches!(ty, IrType::Fixed { .. }) && matches!(op, IrBinOp::Mul | IrBinOp::Div)
            {
                type_possible_mask(program, ty)
            } else if sz == 1 {
                let signed = is_signed_ty(ty);
                let lb = get_bits(*lhs) & 0x00FF;
                let rb = get_bits(*rhs) & 0x00FF;
                let b = match op {
                    IrBinOp::BitAnd => lb & rb,
                    IrBinOp::BitOr | IrBinOp::BitXor => lb | rb,
                    IrBinOp::Shl => match get_const(*rhs) {
                        Some(k) if k >= 8 => 0,
                        Some(k) if k >= 0 => (lb << (k as u32)) & 0x00FF,
                        _ => 0x00FF,
                    },
                    IrBinOp::Shr if !signed || (lb & 0x0080) == 0 => match get_const(*rhs) {
                        Some(k) if k >= 8 => 0,
                        Some(k) if k >= 0 => lb >> (k as u32),
                        _ => 0x00FF,
                    },
                    IrBinOp::Rem if !signed => match get_const(*rhs) {
                        Some(m) if m > 0 => mask_up_to(((m as u64 - 1).min(255)) as u16),
                        _ => 0x00FF,
                    },
                    IrBinOp::Div if !signed => match get_const(*rhs) {
                        Some(d) if d > 0 => mask_up_to(((lb as u64) / (d as u64)).min(255) as u16),
                        _ => 0x00FF,
                    },
                    IrBinOp::Add if !signed => {
                        let sum = (lb as u32) + (rb as u32);
                        if *mode == ArithMode::Saturating {
                            mask_up_to(sum.min(255) as u16)
                        } else if sum <= 255 {
                            mask_up_to(sum as u16)
                        } else {
                            0x00FF
                        }
                    }
                    IrBinOp::Mul if !signed => {
                        let prod = (lb as u32) * (rb as u32);
                        if *mode == ArithMode::Saturating {
                            mask_up_to(prod.min(255) as u16)
                        } else if prod <= 255 {
                            mask_up_to(prod as u16)
                        } else {
                            0x00FF
                        }
                    }
                    _ => 0x00FF,
                };
                Some(b)
            } else if sz == 2 {
                let signed = is_signed_ty(ty);
                let lb = get_bits(*lhs);
                let rb = get_bits(*rhs);
                let b = match op {
                    IrBinOp::BitAnd => lb & rb,
                    IrBinOp::BitOr | IrBinOp::BitXor => lb | rb,
                    IrBinOp::Shl => match get_const(*rhs) {
                        Some(k) if k >= 16 => 0,
                        Some(k) if k >= 0 => lb.wrapping_shl(k as u32),
                        _ => 0xFFFF,
                    },
                    IrBinOp::Shr if !signed || (lb & 0x8000) == 0 => match get_const(*rhs) {
                        Some(k) if k >= 16 => 0,
                        Some(k) if k >= 0 => lb >> (k as u32),
                        _ => 0xFFFF,
                    },
                    IrBinOp::Rem if !signed => match get_const(*rhs) {
                        Some(m) if m > 0 => mask_up_to(((m as u64 - 1).min(0xFFFF)) as u16),
                        _ => 0xFFFF,
                    },
                    IrBinOp::Div if !signed => match get_const(*rhs) {
                        Some(d) if d > 0 => {
                            mask_up_to(((lb as u64) / (d as u64)).min(0xFFFF) as u16)
                        }
                        _ => 0xFFFF,
                    },
                    IrBinOp::Add if !signed => {
                        let sum = (lb as u32) + (rb as u32);
                        if *mode == ArithMode::Saturating {
                            mask_up_to(sum.min(0xFFFF) as u16)
                        } else if sum <= 0xFFFF {
                            mask_up_to(sum as u16)
                        } else {
                            0xFFFF
                        }
                    }
                    _ => 0xFFFF,
                };
                Some(b)
            } else {
                None
            }
        }
        IrInst::LoadLocal { slot, ty, .. } => get_slot_bits(*slot, ty),
        IrInst::Phi { ty, incoming, .. } => {
            if *ty == IrType::Bool {
                Some(1)
            } else {
                let mask = type_possible_mask(program, ty)?;
                let mut acc = 0u16;
                for &(_, v) in incoming {
                    acc |= get_bits(v);
                }
                Some(acc & mask)
            }
        }
        IrInst::LoadPtr { ty, .. } | IrInst::UnOp { ty, .. } => type_possible_mask(program, ty),
        IrInst::FixedInt {
            base, frac_bits, ..
        } => {
            if base.byte_size() == 1 {
                Some(0x00FF)
            } else if base.byte_size() == 2
                && !base.is_signed()
                && *frac_bits > 0
                && *frac_bits < 16
            {
                Some(0xFFFFu16 >> (*frac_bits as u32))
            } else {
                None
            }
        }
        IrInst::Call { func: cid, .. } => {
            type_possible_mask(program, &program.functions[*cid as usize].ret_ty)
        }
        IrInst::ExternCall { extern_fn: eid, .. } => program.externs[*eid as usize]
            .ret
            .as_ref()
            .and_then(|(rty, _)| type_possible_mask(program, rty)),
        IrInst::PortIn { .. } | IrInst::StateTag { .. } => Some(0x00FF),
        IrInst::QueueIsEmpty { .. } => Some(1),
        _ => None,
    }
}

impl<'a> EmitCtx<'a> {
    fn clear_hl_tracking(&mut self) {
        self.hl_ptr = None;
        self.hl_val = None;
    }

    fn invalidate_hl_on_store_ptr(&mut self, addr: ValueId, sz: u16) {
        if let Some((HlBase::RamPtr(ptr_addr), _)) = self.hl_ptr {
            if let Some(&(fixed_addr, true)) = self.const_addrs.get(&addr) {
                self.invalidate_hl_ram(fixed_addr, sz);
            } else if self
                .escaped_slot_ranges
                .iter()
                .any(|&(ea, esz)| ranges_overlap(ptr_addr, 2, ea, esz))
            {
                self.hl_ptr = None;
            }
        }
    }

    fn emit_load_ptr_hl(&mut self, ptr: ValueId) {
        if let Some(&disp) = self.ix_addrs.get(&ptr) {
            writeln!(self.asm, "    push ix").unwrap();
            writeln!(self.asm, "    pop hl").unwrap();
            self.emit_add_hl_const(disp as i16 as u16);
        } else {
            self.emit_load_hl_from(ptr);
        }
    }

    fn emit_add_hl_const(&mut self, off: u16) {
        if (1..=3).contains(&off) {
            for _ in 0..off {
                writeln!(self.asm, "    inc hl").unwrap();
            }
        } else if off > 0 {
            writeln!(self.asm, "    ld de, 0x{off:04X}").unwrap();
            writeln!(self.asm, "    add hl, de").unwrap();
        }
    }

    fn update_possible_bits(&mut self, inst: &IrInst) {
        if let IrInst::StoreLocal { slot, val, ty } = inst
            && !self.addr_taken_locals.contains(slot)
            && let Some(mask) = type_possible_mask(self.program, ty)
        {
            let b = self.possible_bits(*val) & mask;
            self.local_possible_bits.insert(*slot, b);
            return;
        }
        let bits = eval_inst_possible_bits(
            self.program,
            inst,
            self.u8_bound_vals,
            |v| self.const_vals.get(&v).copied(),
            |v| self.possible_bits(v),
            |slot, ty| {
                let mask = type_possible_mask(self.program, ty)?;
                if let Some(&b) = self.local_possible_bits.get(&slot) {
                    Some(b & mask)
                } else if let Some(&b) = self.fn_slot_possible_bits.get(&slot) {
                    Some(b & mask)
                } else {
                    Some(mask)
                }
            },
        );
        if let (Some(dst), Some(b)) = (inst.dst(), bits) {
            self.possible_bits.insert(dst, b);
        }
    }

    pub(super) fn emit_inst(
        &mut self,
        bb_idx: usize,
        inst_idx: usize,
        inst: &IrInst,
    ) -> Result<(), String> {
        self.update_possible_bits(inst);
        let ix_slot = self.fa.ix_slot();
        let fused_cmp_dsts = &self.fa.fused_cmp_dsts;
        let val_sizes = &self.fa.val_sizes;
        let preserves_hl = matches!(
            inst,
            IrInst::ConstInt { .. }
                | IrInst::ConstBool { .. }
                | IrInst::LoadLocal { .. }
                | IrInst::StoreLocal { .. }
                | IrInst::AddrLocal { .. }
                | IrInst::AddrGlobal { .. }
                | IrInst::AddrFn { .. }
                | IrInst::FieldAddr { .. }
                | IrInst::IndexAddr { .. }
                | IrInst::LoadPtr {
                    volatile: false,
                    ..
                }
                | IrInst::StorePtr {
                    volatile: false,
                    ..
                }
                | IrInst::Cast { .. }
                | IrInst::Assert { .. }
        );
        if !preserves_hl {
            self.clear_hl_tracking();
        }

        // Invalidate same-block shift CSE on any scratch write overlapping `prev_da`
        if let Some(d) = inst.dst() {
            let da = self.val_addr(d);
            let dsz = val_sizes.get(&d).copied().unwrap_or(2).clamp(1, 4);
            self.shift_cse
                .retain(|&(src_a, _, _), &mut (_, pda)| pda != da && src_a != da);
            self.index_cse.retain(|(_, ik, _), &mut res_a| {
                let idx_ok = match *ik {
                    IndexKey::Reg(_) | IndexKey::Const(_) => true,
                    IndexKey::Mem(ia, isz) => !ranges_overlap(ia, isz, da, dsz),
                };
                idx_ok && !ranges_overlap(res_a, 2, da, dsz)
            });
        }
        // Invalidate same-block shift CSE, index CSE, and non-negative IX tracking on memory/local mutations
        match inst {
            IrInst::StoreLocal { slot, ty, .. } => {
                if self.fa.ix_local_slot == Some(*slot) {
                    self.index_cse.retain(|(b, _, _), _| !b.starts_with("ix:"));
                } else if let Some(reg) = self.fa.reg_alloc.slot_reg(*slot) {
                    self.index_cse.retain(|(_, ik, _), _| match *ik {
                        IndexKey::Reg(rk) => !rk.mask().intersects(reg.mask()),
                        IndexKey::Mem(..) | IndexKey::Const(_) => true,
                    });
                } else {
                    let sa = self.layout.slot_addrs[&(self.func.id, *slot)];
                    let ssz = self.program.type_size(ty).clamp(1, 4);
                    self.shift_cse
                        .retain(|&(src_a, _, _), _| !ranges_overlap(src_a, 2, sa, ssz));
                    self.index_cse.retain(|(_, ik, _), &mut res_a| {
                        let idx_ok = match *ik {
                            IndexKey::Reg(_) | IndexKey::Const(_) => true,
                            IndexKey::Mem(ia, isz) => !ranges_overlap(ia, isz, sa, ssz),
                        };
                        idx_ok && !ranges_overlap(res_a, 2, sa, ssz)
                    });
                }
            }
            IrInst::StorePtr { .. } => {
                self.shift_cse.clear();
                self.block_nonneg_ix = None;
                self.index_cse.retain(|(_, ik, _), _| match *ik {
                    IndexKey::Reg(_) | IndexKey::Const(_) => true,
                    IndexKey::Mem(ia, isz) => !self
                        .escaped_slot_ranges
                        .iter()
                        .any(|&(ea, esz)| ranges_overlap(ia, isz, ea, esz)),
                });
            }
            IrInst::LoadPtr { volatile: true, .. }
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
            | IrInst::QueuePush { .. }
            | IrInst::QueuePop { .. }
            | IrInst::BankEnter { .. }
            | IrInst::BankExit { .. }
            | IrInst::CriticalEnter
            | IrInst::CriticalExit
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
                        self.invalidate_hl_ram(da, 1);
                        let b = (*val & 0xFF) as u8;
                        if b == 0 {
                            writeln!(self.asm, "    xor a").unwrap();
                        } else {
                            writeln!(self.asm, "    ld a, 0x{b:02X}").unwrap();
                        }
                        writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                    }
                    2 => {
                        self.clear_hl_tracking();
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
                let da = self.val_addr(*dst);
                self.invalidate_hl_ram(da, 1);
                let da_mem = self.fmt_dst_mem(*dst);
                if *val {
                    writeln!(self.asm, "    ld a, 1").unwrap();
                } else {
                    writeln!(self.asm, "    xor a").unwrap();
                }
                writeln!(self.asm, "    ld {da_mem}, a").unwrap();
            }
            IrInst::LoadLocal { dst, slot, ty } => {
                if ix_slot == Some(*slot) || self.fa.is_ix_alias_slot(*slot) {
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
                    let da = self.val_addr(*dst);
                    let da_mem = self.fmt_dst_mem(*dst);
                    match reg {
                        RegLoc::C => {
                            self.invalidate_hl_ram(da, 1);
                            writeln!(self.asm, "    ld a, c").unwrap();
                            writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                        }
                        RegLoc::B => {
                            self.invalidate_hl_ram(da, 1);
                            writeln!(self.asm, "    ld a, b").unwrap();
                            writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                        }
                        RegLoc::BC => {
                            self.clear_hl_tracking();
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
                    self.val_from_local.insert(*dst, *slot);
                    return Ok(());
                }
                let da = self.val_addr(*dst);
                let sa_mem = self.fmt_ram_mem(sa);
                let da_mem = self.fmt_dst_mem(*dst);
                match self.program.type_size(ty) {
                    1 => {
                        self.invalidate_hl_ram(da, 1);
                        writeln!(self.asm, "    ld a, {sa_mem}").unwrap();
                        writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                    }
                    2 => {
                        self.clear_hl_tracking();
                        writeln!(self.asm, "    ld hl, {sa_mem}").unwrap();
                        writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                    }
                    4 => {
                        self.clear_hl_tracking();
                        if matches!(ty, IrType::Slice { .. } | IrType::ConstSlice(_))
                            && slice_val_only_uses_ptr(self.func, *dst)
                        {
                            writeln!(self.asm, "    ld hl, {sa_mem}").unwrap();
                            writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                        } else {
                            self.emit_copy32(sa, true, da, true);
                        }
                    }
                    _ => {}
                }
            }
            IrInst::StoreLocal { slot, val, ty } => {
                if self.fa.is_ix_alias_slot(*slot) {
                    return Ok(());
                }
                if self.fa.ix_local_slot == Some(*slot) {
                    let emit_ix_add = |asm: &mut String, k: i16| match k {
                        0 => {}
                        1 => writeln!(asm, "    inc ix").unwrap(),
                        2 => {
                            writeln!(asm, "    inc ix").unwrap();
                            writeln!(asm, "    inc ix").unwrap();
                        }
                        -1 => writeln!(asm, "    dec ix").unwrap(),
                        -2 => {
                            writeln!(asm, "    dec ix").unwrap();
                            writeln!(asm, "    dec ix").unwrap();
                        }
                        _ => {
                            writeln!(asm, "    ld de, 0x{:04X}", k as u16).unwrap();
                            writeln!(asm, "    add ix, de").unwrap();
                        }
                    };
                    if let Some(&k) = self.fused_ix_step.get(val) {
                        emit_ix_add(self.asm, k);
                    } else if let Some(&(addr, is_u)) = self.const_addrs.get(val) {
                        let imm = self.fmt_addr_imm(addr, is_u);
                        writeln!(self.asm, "    ld ix, {imm}").unwrap();
                    } else if let Some((sym, off)) = self.rom_addrs.get(val) {
                        if *off == 0 {
                            writeln!(self.asm, "    ld ix, {sym}").unwrap();
                        } else {
                            writeln!(self.asm, "    ld ix, {sym} + {off}").unwrap();
                        }
                    } else if let Some(&c) = self.const_vals.get(val) {
                        writeln!(self.asm, "    ld ix, 0x{:04X}", c as u16).unwrap();
                    } else if let Some(&disp) = self.ix_addrs.get(val) {
                        emit_ix_add(self.asm, disp as i16);
                    } else if self.val_in_reg.get(val) == Some(&RegLoc::BC) {
                        writeln!(self.asm, "    push bc").unwrap();
                        writeln!(self.asm, "    pop ix").unwrap();
                    } else {
                        self.emit_load_hl_from(*val);
                        writeln!(self.asm, "    push hl").unwrap();
                        writeln!(self.asm, "    pop ix").unwrap();
                    }
                    self.ix_addrs.clear();
                    self.ix_val_loads.clear();
                    self.index_cse.retain(|(b, _, _), _| !b.starts_with("ix:"));
                    return Ok(());
                }
                if !self.fused_reg_step.contains_key(val)
                    && self.fa.direct_reg_dsts.get(val)
                        == self.fa.reg_alloc.slot_reg(*slot).as_ref()
                    && self.fa.direct_reg_dsts.contains_key(val)
                {
                    return Ok(());
                }
                if let Some(reg) = self.fa.reg_alloc.slot_reg(*slot) {
                    if matches!(self.hl_ptr, Some((HlBase::BcReg, _)))
                        && reg.mask().intersects(RegMask::BC)
                    {
                        self.hl_ptr = None;
                    }
                    if let Some(&k) = self.fused_reg_step.get(val) {
                        match reg {
                            RegLoc::BC => {
                                let mnem = if k > 0 { "inc bc" } else { "dec bc" };
                                for _ in 0..k.abs() {
                                    writeln!(self.asm, "    {mnem}").unwrap();
                                }
                            }
                            RegLoc::B | RegLoc::C => {
                                let r = reg.reg8_name();
                                let mnem = if k > 0 { "inc" } else { "dec" };
                                for _ in 0..k.abs() {
                                    writeln!(self.asm, "    {mnem} {r}").unwrap();
                                }
                            }
                        }
                        self.val_in_reg
                            .retain(|_, &mut vr| !vr.mask().intersects(reg.mask()));
                        return Ok(());
                    }
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
                self.invalidate_hl_ram(sa, sz);
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
                            self.clear_hl_tracking();
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
                            self.hl_ptr = Some((HlBase::RamPtr(sa), 0));
                            self.hl_val = Some(*val);
                        }
                        4 => {
                            self.clear_hl_tracking();
                            if let Some(&(ptr, len)) = self.const_slices.get(val) {
                                let sa_lo = self.fmt_ram_mem(sa);
                                let sa_hi = self.fmt_ram_mem(sa.wrapping_add(2));
                                self.emit_load_hl_from(ptr);
                                writeln!(self.asm, "    ld {sa_lo}, hl").unwrap();
                                self.emit_load_hl_from(len);
                                writeln!(self.asm, "    ld {sa_hi}, hl").unwrap();
                                self.clear_hl_tracking();
                            } else {
                                self.emit_copy32_from_val(*val, sa, true);
                            }
                        }
                        _ => {}
                    }
                }
            }
            IrInst::AddrLocal { dst, slot } => {
                if ix_slot == Some(*slot) || self.fa.is_ix_alias_slot(*slot) {
                    self.ix_param_addr_vals.insert(*dst);
                    return Ok(());
                }
                let sa = self.layout.slot_addrs[&(self.func.id, *slot)];
                self.const_addrs.insert(*dst, (sa, true));
                self.const_addr_srcs.insert(*dst, ConstAddrSrc::Local(*slot));
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
                self.clear_hl_tracking();
                writeln!(self.asm, "    ld hl, {sa_imm}").unwrap();
                writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                self.hl_val = Some(*dst);
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
                    self.clear_hl_tracking();
                    writeln!(self.asm, "    ld hl, {}", g.name).unwrap();
                    writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                    self.hl_val = Some(*dst);
                } else {
                    let ga = self.layout.global_addrs[global];
                    let is_u = g.fixed_addr.is_none() && g.space.is_none();
                    self.const_addrs.insert(*dst, (ga, is_u));
                    self.const_addr_srcs.insert(*dst, ConstAddrSrc::Global);
                    if is_const_addr_always_folded(
                    self.program,
                    self.func,
                    *dst,
                    self.fused_skip_insts,
                ) {
                        return Ok(());
                    }
                    self.clear_hl_tracking();
                    let ga_imm = self.fmt_addr_imm(ga, is_u);
                    writeln!(self.asm, "    ld hl, {ga_imm}").unwrap();
                    writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                    self.hl_val = Some(*dst);
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
                self.clear_hl_tracking();
                let da_mem = self.fmt_dst_mem(*dst);
                writeln!(self.asm, "    ld hl, {sym}").unwrap();
                writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                self.hl_val = Some(*dst);
            }
            IrInst::FieldAddr {
                dst,
                base,
                byte_offset,
            } => {
                let da = self.val_addr(*dst);
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
                    if let Some(&src) = self.const_addr_srcs.get(base) {
                        self.const_addr_srcs.insert(*dst, src);
                    }
                    if is_const_addr_always_folded(
                    self.program,
                    self.func,
                    *dst,
                    self.fused_skip_insts,
                ) {
                        return Ok(());
                    }
                    self.clear_hl_tracking();
                    let folded_imm = self.fmt_addr_imm(folded, is_u);
                    writeln!(self.asm, "    ld hl, {folded_imm}").unwrap();
                    writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                    self.hl_val = Some(*dst);
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
                    self.clear_hl_tracking();
                    if new_off == 0 {
                        writeln!(self.asm, "    ld hl, {sym}").unwrap();
                    } else {
                        writeln!(self.asm, "    ld hl, {sym} + {new_off}").unwrap();
                    }
                    writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                    self.hl_val = Some(*dst);
                } else {
                    let base_key = if self.val_in_reg.get(base) == Some(&RegLoc::BC) {
                        Some(HlBase::BcReg)
                    } else if let Some(&(fixed_addr, true)) = self.const_addr_loads.get(base)
                        && matches!(
                            self.const_addr_srcs.get(base),
                            Some(&ConstAddrSrc::Local(_))
                        )
                    {
                        Some(HlBase::RamPtr(fixed_addr))
                    } else if !self.const_vals.contains_key(base)
                        && !self.const_addr_loads.contains_key(base)
                        && !self.ix_val_loads.contains_key(base)
                    {
                        Some(HlBase::RamPtr(self.val_loc(*base)))
                    } else {
                        None
                    };
                    let target_off = *byte_offset as i32;
                    let walked = if let (Some(bk), Some((cur_bk, cur_off))) = (base_key, self.hl_ptr)
                        && bk == cur_bk
                        && (-2..=2).contains(&(target_off - cur_off))
                    {
                        let delta = target_off - cur_off;
                        let step = if delta > 0 { "inc hl" } else { "dec hl" };
                        for _ in 0..delta.abs() {
                            writeln!(self.asm, "    {step}").unwrap();
                        }
                        true
                    } else {
                        false
                    };
                    if !walked {
                        self.emit_load_hl_from(*base);
                        if (1..=3).contains(byte_offset) {
                            for _ in 0..*byte_offset {
                                writeln!(self.asm, "    inc hl").unwrap();
                            }
                        } else if *byte_offset > 0 {
                            writeln!(self.asm, "    ld de, 0x{byte_offset:04X}").unwrap();
                            writeln!(self.asm, "    add hl, de").unwrap();
                        }
                    }
                    writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                    let da_clobbers_base = matches!(
                        base_key,
                        Some(HlBase::RamPtr(pa)) if ranges_overlap(pa, 2, da, 2)
                    );
                    self.invalidate_hl_ram(da, 2);
                    self.hl_ptr = if da_clobbers_base {
                        None
                    } else {
                        base_key.map(|bk| (bk, target_off))
                    };
                    self.hl_val = Some(*dst);
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
                let is_ix_static_elem = self
                    .fa
                    .ix_static_elem
                    .as_ref()
                    .is_some_and(|ise| ise.index_vals.contains(dst));
                if is_ix_static_elem
                    && self.fa.ix_static_elem.as_ref().is_some_and(|ise| {
                        ise.hoist_param.is_some() && (!self.opts.check_bounds || ise.len == 0)
                    })
                {
                    self.ix_addrs.insert(*dst, 0);
                    return Ok(());
                }
                let (eff_index, idx_size) =
                    if let Some(&u8_v) = self.fused_index_u8_cast.get(index) {
                        (u8_v, 1u16)
                    } else {
                        (*index, val_sizes.get(index).copied().unwrap_or(1))
                    };
                let ia = self.val_loc(eff_index);
                let da = self.val_addr(*dst);
                let da_mem = self.fmt_dst_mem(*dst);
                // Constant-index folding: `base[CONST]` -> direct `const_addrs` / `rom_addrs` / `ix_addrs`!
                if let Some(&c_idx) = self.const_vals.get(&eff_index)
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
                        let same_ix_epoch = if let Some(cslot) = self.fa.ix_local_slot {
                            let mut epoch_ok = true;
                            let mut seen_store = false;
                            for (bi, b) in self.func.blocks.iter().enumerate() {
                                if bi != bb_idx {
                                    if b.insts.iter().any(|u| u.uses_val(*dst))
                                        || matches!(
                                            b.term,
                                            Terminator::Return(Some(v))
                                                | Terminator::Branch { cond: v, .. }
                                                if v == *dst
                                        )
                                    {
                                        epoch_ok = false;
                                        break;
                                    }
                                    continue;
                                }
                                for u in b.insts.iter().skip(inst_idx + 1) {
                                    if u.uses_val(*dst) && seen_store {
                                        epoch_ok = false;
                                        break;
                                    }
                                    if matches!(u, IrInst::StoreLocal { slot, .. } if *slot == cslot)
                                    {
                                        seen_store = true;
                                    }
                                }
                                if !epoch_ok {
                                    break;
                                }
                            }
                            epoch_ok
                        } else {
                            true
                        };
                        let all_ix_folded = same_ix_epoch
                            && self.func.blocks.iter().all(|b| {
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
                        if let Some(&src) = self.const_addr_srcs.get(base) {
                            self.const_addr_srcs.insert(*dst, src);
                        }
                        if is_const_addr_always_folded(
                    self.program,
                    self.func,
                    *dst,
                    self.fused_skip_insts,
                ) {
                            return Ok(());
                        }
                        let folded_imm = self.fmt_addr_imm(folded, is_u);
                        self.clear_hl_tracking();
                        writeln!(self.asm, "    ld hl, {folded_imm}").unwrap();
                        writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                        self.hl_val = Some(*dst);
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
                        self.clear_hl_tracking();
                        if new_off == 0 {
                            writeln!(self.asm, "    ld hl, {sym}").unwrap();
                        } else {
                            writeln!(self.asm, "    ld hl, {sym} + {new_off}").unwrap();
                        }
                        writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                        self.hl_val = Some(*dst);
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
                let reg_idx = self
                    .val_in_reg
                    .get(&eff_index)
                    .copied()
                    .filter(|r| r.byte_size() == 1);
                let idx_key = if let Some(&c_idx) = self.const_vals.get(&eff_index) {
                    IndexKey::Const(c_idx)
                } else if idx_size == 1 && let Some(r) = reg_idx {
                    IndexKey::Reg(r)
                } else {
                    IndexKey::Mem(ia, idx_size.clamp(1, 2))
                };
                if let Some(ref bkey) = cse_base_key
                    && !is_ix_static_elem
                    && let Some(&prev_da) =
                        self.index_cse.get(&(bkey.clone(), idx_key, *elem_size))
                {
                    let pda_mem = self.fmt_ram_mem(prev_da);
                    self.clear_hl_tracking();
                    writeln!(self.asm, "    ld hl, {pda_mem}").unwrap();
                    writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                    self.hl_ptr = Some((HlBase::RamPtr(da), 0));
                    self.hl_val = Some(*dst);
                    return Ok(());
                }
                let small_u8_mul = is_small_u8_array_mul(*len, *elem_size);
                let need_bounds_check = self.opts.check_bounds && *len > 0;
                let use_mul_a = idx_size == 1
                    && !need_bounds_check
                    && !small_u8_mul
                    && self.shared_muls.contains(elem_size);
                // Compact 8-bit indexing into an IX struct array field (`ix + disp + u8`) when
                // bounds checks are disabled and `disp + len - 1 <= 255` fits in 8 bits without carry.
                let compact_ix_byte = idx_size == 1
                    && *elem_size == 1
                    && *len > 0
                    && !need_bounds_check
                    && self
                        .ix_addrs
                        .get(base)
                        .is_some_and(|&d| d > 0 && (d as u16) + (*len - 1) <= 255);
                let aligned256_fixed_byte = idx_size == 1
                    && *elem_size == 1
                    && !need_bounds_check
                    && self
                        .const_addrs
                        .get(base)
                        .is_some_and(|&(baddr, is_u)| {
                            (!is_u || !self.opts.relocatable_ram) && (baddr & 0xFF) == 0
                        });
                if aligned256_fixed_byte
                    && can_defer_aligned256(
                        self.program,
                        self.func,
                        bb_idx,
                        inst_idx,
                        *dst,
                        eff_index,
                        &self.const_vals,
                        &self.val_in_reg,
                        &self.val_from_local,
                        &self.ix_val_loads,
                        self.fa.ix_local_slot,
                        &self.fa.reg_alloc,
                        &self.analysis.fn_clobbers,
                        &self.fa.escaped_locals,
                    )
                {
                    let hi = (self.const_addrs[base].0 >> 8) as u8;
                    self.aligned256_addrs.insert(*dst, (hi, eff_index));
                    return Ok(());
                }
                if compact_ix_byte {
                    let disp = self.ix_addrs[base] as u8;
                    self.emit_load_a_from(eff_index);
                    writeln!(self.asm, "    add a, 0x{disp:02X}").unwrap();
                    writeln!(self.asm, "    ld l, a").unwrap();
                    writeln!(self.asm, "    ld h, 0").unwrap();
                    writeln!(self.asm, "    push ix").unwrap();
                    writeln!(self.asm, "    pop de").unwrap();
                    writeln!(self.asm, "    add hl, de").unwrap();
                } else if aligned256_fixed_byte {
                    let hi = (self.const_addrs[base].0 >> 8) as u8;
                    if let Some(r) = reg_idx {
                        writeln!(self.asm, "    ld l, {}", r.reg8_name()).unwrap();
                    } else {
                        self.emit_load_a_from(eff_index);
                        writeln!(self.asm, "    ld l, a").unwrap();
                    }
                    writeln!(self.asm, "    ld h, 0x{hi:02X}").unwrap();
                } else if small_u8_mul && !need_bounds_check {
                    if idx_size == 1 {
                        if let Some(r) = reg_idx {
                            let rname = r.reg8_name();
                            writeln!(self.asm, "    ld a, {rname}").unwrap();
                            emit_mul_u8_const(*elem_size, rname, self.asm);
                        } else {
                            self.emit_load_a_from(eff_index);
                            writeln!(self.asm, "    ld l, a").unwrap();
                            emit_mul_u8_const(*elem_size, "l", self.asm);
                        }
                    } else if self.val_in_reg.get(&eff_index) == Some(&RegLoc::BC) {
                        writeln!(self.asm, "    ld a, c").unwrap();
                        emit_mul_u8_const(*elem_size, "c", self.asm);
                    } else {
                        self.emit_load_a_from(eff_index);
                        writeln!(self.asm, "    ld l, a").unwrap();
                        emit_mul_u8_const(*elem_size, "l", self.asm);
                    }
                } else if use_mul_a {
                    if let Some(r) = reg_idx {
                        writeln!(self.asm, "    ld a, {}", r.reg8_name()).unwrap();
                    } else {
                        self.emit_load_a_from(eff_index);
                    }
                    writeln!(self.asm, "    call __aun_mul_a_{elem_size}").unwrap();
                } else {
                    if idx_size == 1 {
                        if let Some(r) = reg_idx {
                            writeln!(self.asm, "    ld l, {}", r.reg8_name()).unwrap();
                        } else {
                            self.emit_load_a_from(eff_index);
                            writeln!(self.asm, "    ld l, a").unwrap();
                        }
                        writeln!(self.asm, "    ld h, 0").unwrap();
                    } else {
                        self.emit_load_hl_from(eff_index);
                    }

                    // Bounds check: index < len (eliminated in Release!)
                    if need_bounds_check {
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

                    // Multiply HL by constant elem_size (use 8-bit inline for small arrays,
                    // outline repeated non-pow2 sizes into `__aun_mul_hl_<k>`)
                    if small_u8_mul {
                        writeln!(self.asm, "    ld a, l").unwrap();
                        emit_mul_u8_const(*elem_size, "l", self.asm);
                    } else if self.shared_muls.contains(elem_size) {
                        writeln!(self.asm, "    call __aun_mul_hl_{elem_size}").unwrap();
                    } else {
                        emit_mul_hl_const(*elem_size, self.asm);
                    }
                }

                // Add base address (already added for `compact_ix_byte` / `aligned256_fixed_byte`)
                if compact_ix_byte || aligned256_fixed_byte {
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
                if is_ix_static_elem {
                    self.clear_hl_tracking();
                    writeln!(self.asm, "    push hl").unwrap();
                    writeln!(self.asm, "    pop ix").unwrap();
                    self.ix_addrs.insert(*dst, 0);
                    return Ok(());
                }
                writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                self.hl_ptr = Some((HlBase::RamPtr(da), 0));
                self.hl_val = Some(*dst);
                if let Some(bkey) = cse_base_key {
                    let no_self_overlap = match idx_key {
                        IndexKey::Reg(_) | IndexKey::Const(_) => true,
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
                    && !self.fa.direct_reg_dsts.contains_key(dst)
                    && let Some(&disp) = self.ix_addrs.get(addr)
                    && (matches!(self.program.type_size(ty), 1 | 2)
                        || (self.program.type_size(ty) == 4 && disp <= 124))
                    && can_alias_ix_load(
                        self.program,
                        self.func,
                        bb_idx,
                        inst_idx,
                        *dst,
                        fused_cmp_dsts,
                        self.fa.ix_local_slot,
                    )
                {
                    let sz = self.program.type_size(ty);
                    self.ix_val_loads.insert(*dst, (disp, sz));
                    if sz == 2 && self.block_nonneg_ix == Some(disp) {
                        self.nonneg_vals.insert(*dst);
                    }
                    return Ok(());
                }
                if !*volatile
                    && !self.fa.direct_reg_dsts.contains_key(dst)
                    && matches!(self.program.type_size(ty), 1 | 2 | 4)
                    && let Some(&(fixed_addr, is_u)) = self.const_addrs.get(addr)
                    && let Some(&src) = self.const_addr_srcs.get(addr)
                    && can_alias_const_addr_load(
                        self.program,
                        self.func,
                        bb_idx,
                        inst_idx,
                        *dst,
                        src,
                    )
                {
                    self.const_addr_loads.insert(*dst, (fixed_addr, is_u));
                    self.const_addr_srcs.insert(*dst, src);
                    return Ok(());
                }
                let slice_ptr_only = !*volatile
                    && matches!(ty, IrType::Slice { .. } | IrType::ConstSlice(_))
                    && slice_val_only_uses_ptr(self.func, *dst);
                if *volatile {
                    writeln!(self.asm, "; @aun:volatile_begin").unwrap();
                }
                if let Some(&(hi, idx_v)) = self.aligned256_addrs.get(addr) {
                    if let Some(&c_idx) = self.const_vals.get(&idx_v) {
                        let fixed = ((hi as u16) << 8) | ((c_idx & 0xFF) as u16);
                        writeln!(self.asm, "    ld a, (0x{fixed:04X})").unwrap();
                    } else if let Some(&reg_idx) = self
                        .val_in_reg
                        .get(&idx_v)
                        .filter(|r| r.byte_size() == 1)
                    {
                        self.clear_hl_tracking();
                        writeln!(self.asm, "    ld l, {}", reg_idx.reg8_name()).unwrap();
                        writeln!(self.asm, "    ld h, 0x{hi:02X}").unwrap();
                        writeln!(self.asm, "    ld a, (hl)").unwrap();
                    } else {
                        self.clear_hl_tracking();
                        self.emit_load_a_from(idx_v);
                        writeln!(self.asm, "    ld l, a").unwrap();
                        writeln!(self.asm, "    ld h, 0x{hi:02X}").unwrap();
                        writeln!(self.asm, "    ld a, (hl)").unwrap();
                    }
                    self.emit_store_dst_a(*dst);
                } else if let Some(&disp) = self.ix_addrs.get(addr) {
                    let sz = self.program.type_size(ty);
                    match sz {
                        1 => {
                            writeln!(self.asm, "    ld a, (ix + {disp})").unwrap();
                            self.emit_store_dst_a(*dst);
                        }
                        2 => {
                            self.clear_hl_tracking();
                            writeln!(self.asm, "    ld l, (ix + {disp})").unwrap();
                            writeln!(self.asm, "    ld h, (ix + {})", disp + 1).unwrap();
                            self.emit_store_dst_hl(*dst);
                            if self.block_nonneg_ix == Some(disp) {
                                self.nonneg_vals.insert(*dst);
                            }
                        }
                        4 if slice_ptr_only => {
                            self.clear_hl_tracking();
                            writeln!(self.asm, "    ld l, (ix + {disp})").unwrap();
                            writeln!(self.asm, "    ld h, (ix + {})", disp + 1).unwrap();
                            self.emit_store_dst_hl(*dst);
                        }
                        4 => {
                            self.clear_hl_tracking();
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
                    match self.program.type_size(ty) {
                        1 => {
                            writeln!(self.asm, "    ld a, {f_mem}").unwrap();
                            self.emit_store_dst_a(*dst);
                        }
                        2 => {
                            self.clear_hl_tracking();
                            writeln!(self.asm, "    ld hl, {f_mem}").unwrap();
                            self.emit_store_dst_hl(*dst);
                        }
                        4 if slice_ptr_only => {
                            self.clear_hl_tracking();
                            writeln!(self.asm, "    ld hl, {f_mem}").unwrap();
                            self.emit_store_dst_hl(*dst);
                        }
                        4 => self.emit_copy32(fixed_addr, is_u, da, true),
                        _ => {}
                    }
                } else if let Some((sym, off)) = self.rom_addrs.get(addr).cloned() {
                    self.clear_hl_tracking();
                    if off == 0 {
                        writeln!(self.asm, "    ld hl, {sym}").unwrap();
                    } else {
                        writeln!(self.asm, "    ld hl, {sym} + {off}").unwrap();
                    }
                    let da_mem = self.fmt_ram_mem(da);
                    match self.program.type_size(ty) {
                        1 => {
                            writeln!(self.asm, "    ld a, (hl)").unwrap();
                            self.emit_store_dst_a(*dst);
                        }
                        2 => {
                            writeln!(self.asm, "    ld e, (hl)").unwrap();
                            writeln!(self.asm, "    inc hl").unwrap();
                            writeln!(self.asm, "    ld d, (hl)").unwrap();
                            writeln!(self.asm, "    ex de, hl").unwrap();
                            self.emit_store_dst_hl(*dst);
                        }
                        4 if slice_ptr_only => {
                            writeln!(self.asm, "    ld e, (hl)").unwrap();
                            writeln!(self.asm, "    inc hl").unwrap();
                            writeln!(self.asm, "    ld d, (hl)").unwrap();
                            writeln!(self.asm, "    ex de, hl").unwrap();
                            self.emit_store_dst_hl(*dst);
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
                    self.hl_val = Some(*addr);
                    let da_mem = self.fmt_ram_mem(da);
                    match self.program.type_size(ty) {
                        1 => {
                            writeln!(self.asm, "    ld a, (hl)").unwrap();
                            self.emit_store_dst_a(*dst);
                        }
                        2 => {
                            if self.fa.direct_reg_dsts.get(dst) == Some(&RegLoc::BC) {
                                writeln!(self.asm, "    ld c, (hl)").unwrap();
                                writeln!(self.asm, "    inc hl").unwrap();
                                writeln!(self.asm, "    ld b, (hl)").unwrap();
                                self.val_in_reg
                                    .retain(|_, &mut vr| !RegMask::BC.intersects(vr.mask()));
                                self.index_cse.retain(|(_, ik, _), _| match *ik {
                                    IndexKey::Reg(rk) => !rk.mask().intersects(RegMask::BC),
                                    IndexKey::Mem(..) | IndexKey::Const(_) => true,
                                });
                                self.hl_val = None;
                                if matches!(self.hl_ptr, Some((HlBase::BcReg, _))) {
                                    self.hl_ptr = None;
                                } else if let Some((bk, off)) = self.hl_ptr {
                                    self.hl_ptr = Some((bk, off + 1));
                                }
                            } else {
                                let another_field_follows = self.hl_ptr.is_some()
                                    && !self.fa.direct_reg_dsts.contains_key(dst)
                                    && self.func.blocks[bb_idx].insts[inst_idx + 1..]
                                        .iter()
                                        .take_while(|i| {
                                            !matches!(
                                                i,
                                                IrInst::Call { .. }
                                                    | IrInst::CallIndirect { .. }
                                                    | IrInst::ExternCall { .. }
                                                    | IrInst::StateDispatch { .. }
                                                    | IrInst::StateTransition { .. }
                                                    | IrInst::InlineAsm { .. }
                                                    | IrInst::MemCopy { .. }
                                                    | IrInst::MemFill { .. }
                                                    | IrInst::MemSet { .. }
                                                    | IrInst::PoolAlloc { .. }
                                                    | IrInst::PoolFree { .. }
                                                    | IrInst::QueuePush { .. }
                                                    | IrInst::QueuePop { .. }
                                            )
                                        })
                                        .any(|i| matches!(i, IrInst::FieldAddr { .. }));
                                writeln!(self.asm, "    ld e, (hl)").unwrap();
                                writeln!(self.asm, "    inc hl").unwrap();
                                writeln!(self.asm, "    ld d, (hl)").unwrap();
                                if another_field_follows {
                                    writeln!(self.asm, "    ld {da_mem}, de").unwrap();
                                    self.invalidate_hl_ram(da, 2);
                                    if let Some((bk, off)) = self.hl_ptr {
                                        self.hl_ptr = Some((bk, off + 1));
                                    }
                                    self.hl_val = None;
                                } else {
                                    writeln!(self.asm, "    ex de, hl").unwrap();
                                    self.emit_store_dst_hl(*dst);
                                }
                            }
                        }
                        4 if slice_ptr_only => {
                            writeln!(self.asm, "    ld e, (hl)").unwrap();
                            writeln!(self.asm, "    inc hl").unwrap();
                            writeln!(self.asm, "    ld d, (hl)").unwrap();
                            writeln!(self.asm, "    ex de, hl").unwrap();
                            self.emit_store_dst_hl(*dst);
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
                            self.invalidate_hl_ram(da, 4);
                            if let Some((bk, off)) = self.hl_ptr {
                                self.hl_ptr = Some((bk, off + 3));
                            }
                            self.hl_val = None;
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
                self.invalidate_hl_on_store_ptr(*addr, sz);
                if let Some(&(hi, idx_v)) = self.aligned256_addrs.get(addr) {
                    if let Some(&c_idx) = self.const_vals.get(&idx_v) {
                        let fixed = ((hi as u16) << 8) | ((c_idx & 0xFF) as u16);
                        if let Some(&c) = self.const_vals.get(val) {
                            let b = (c & 0xFF) as u8;
                            if b == 0 {
                                writeln!(self.asm, "    xor a").unwrap();
                            } else {
                                writeln!(self.asm, "    ld a, 0x{b:02X}").unwrap();
                            }
                            writeln!(self.asm, "    ld (0x{fixed:04X}), a").unwrap();
                        } else {
                            self.emit_load_a_from(*val);
                            writeln!(self.asm, "    ld (0x{fixed:04X}), a").unwrap();
                        }
                    } else if let Some(&reg_idx) = self
                        .val_in_reg
                        .get(&idx_v)
                        .filter(|r| r.byte_size() == 1)
                    {
                        self.clear_hl_tracking();
                        if let Some(&c) = self.const_vals.get(val) {
                            writeln!(self.asm, "    ld l, {}", reg_idx.reg8_name()).unwrap();
                            writeln!(self.asm, "    ld h, 0x{hi:02X}").unwrap();
                            writeln!(self.asm, "    ld (hl), 0x{:02X}", (c & 0xFF) as u8).unwrap();
                        } else {
                            self.emit_load_a_from(*val);
                            writeln!(self.asm, "    ld l, {}", reg_idx.reg8_name()).unwrap();
                            writeln!(self.asm, "    ld h, 0x{hi:02X}").unwrap();
                            writeln!(self.asm, "    ld (hl), a").unwrap();
                        }
                    } else if let Some(&c) = self.const_vals.get(val) {
                        self.clear_hl_tracking();
                        self.emit_load_a_from(idx_v);
                        writeln!(self.asm, "    ld l, a").unwrap();
                        writeln!(self.asm, "    ld h, 0x{hi:02X}").unwrap();
                        writeln!(self.asm, "    ld (hl), 0x{:02X}", (c & 0xFF) as u8).unwrap();
                    } else if let Some(&reg_val) =
                        self.val_in_reg.get(val).filter(|r| r.byte_size() == 1)
                    {
                        self.clear_hl_tracking();
                        self.emit_load_a_from(idx_v);
                        writeln!(self.asm, "    ld l, a").unwrap();
                        writeln!(self.asm, "    ld h, 0x{hi:02X}").unwrap();
                        writeln!(self.asm, "    ld (hl), {}", reg_val.reg8_name()).unwrap();
                    } else if let Some(&(disp, _)) = self.ix_val_loads.get(val) {
                        self.clear_hl_tracking();
                        self.emit_load_a_from(idx_v);
                        writeln!(self.asm, "    ld l, a").unwrap();
                        writeln!(self.asm, "    ld h, 0x{hi:02X}").unwrap();
                        writeln!(self.asm, "    ld e, (ix + {disp})").unwrap();
                        writeln!(self.asm, "    ld (hl), e").unwrap();
                    } else {
                        self.clear_hl_tracking();
                        self.emit_load_a_from(idx_v);
                        writeln!(self.asm, "    ld l, a").unwrap();
                        writeln!(self.asm, "    ld h, 0x{hi:02X}").unwrap();
                        self.emit_load_a_from(*val);
                        writeln!(self.asm, "    ld (hl), a").unwrap();
                    }
                } else if let Some(&disp) = self.ix_addrs.get(addr) {
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
                                self.clear_hl_tracking();
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
                            4 => self.emit_copy32_from_val(*val, fixed_addr, is_u),
                            _ => {}
                        }
                    }
                } else if let Some((sym, off)) = self.rom_addrs.get(addr).cloned() {
                    self.clear_hl_tracking();
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
                                self.emit_load_word32(*val, false, "de", false);
                                emit_ld_hl_sym(self.asm);
                                writeln!(self.asm, "    ld (hl), e").unwrap();
                                writeln!(self.asm, "    inc hl").unwrap();
                                writeln!(self.asm, "    ld (hl), d").unwrap();
                                writeln!(self.asm, "    inc hl").unwrap();
                                self.emit_load_word32(*val, true, "de", false);
                                writeln!(self.asm, "    ld (hl), e").unwrap();
                                writeln!(self.asm, "    inc hl").unwrap();
                                writeln!(self.asm, "    ld (hl), d").unwrap();
                            }
                        }
                        _ => {}
                    }
                } else {
                    match sz {
                        1 => {
                            self.emit_load_hl_from(*addr);
                            self.hl_val = Some(*addr);
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
                            self.emit_load_hl_from(*addr);
                            writeln!(self.asm, "    ld (hl), e").unwrap();
                            writeln!(self.asm, "    inc hl").unwrap();
                            writeln!(self.asm, "    ld (hl), d").unwrap();
                            if let Some((bk, off)) = self.hl_ptr {
                                self.hl_ptr = Some((bk, off + 1));
                            }
                            self.hl_val = None;
                        }
                        4 => {
                            if let Some(&c) = self.const_vals.get(val) {
                                let [b0, b1, b2, b3] = (c as u32).to_le_bytes();
                                self.emit_load_hl_from(*addr);
                                writeln!(self.asm, "    ld (hl), 0x{b0:02X}").unwrap();
                                writeln!(self.asm, "    inc hl").unwrap();
                                writeln!(self.asm, "    ld (hl), 0x{b1:02X}").unwrap();
                                writeln!(self.asm, "    inc hl").unwrap();
                                writeln!(self.asm, "    ld (hl), 0x{b2:02X}").unwrap();
                                writeln!(self.asm, "    inc hl").unwrap();
                                writeln!(self.asm, "    ld (hl), 0x{b3:02X}").unwrap();
                            } else {
                                self.emit_load_word32(*val, false, "de", false);
                                self.emit_load_hl_from(*addr);
                                writeln!(self.asm, "    ld (hl), e").unwrap();
                                writeln!(self.asm, "    inc hl").unwrap();
                                writeln!(self.asm, "    ld (hl), d").unwrap();
                                writeln!(self.asm, "    inc hl").unwrap();
                                self.emit_load_word32(*val, true, "de", false);
                                writeln!(self.asm, "    ld (hl), e").unwrap();
                                writeln!(self.asm, "    inc hl").unwrap();
                                writeln!(self.asm, "    ld (hl), d").unwrap();
                            }
                            if let Some((bk, off)) = self.hl_ptr {
                                self.hl_ptr = Some((bk, off + 3));
                            }
                            self.hl_val = None;
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
                } else if *len < 12 {
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
                    if let Some(&(addr, is_u)) = self.const_addrs.get(dst_addr) {
                        let dst_imm = self.fmt_addr_imm(addr, is_u);
                        let dst1_imm = self.fmt_addr_imm(addr.wrapping_add(1), is_u);
                        writeln!(self.asm, "    ld hl, {dst_imm}").unwrap();
                        if let Some(&c) = self.const_vals.get(val) {
                            writeln!(self.asm, "    ld (hl), 0x{:02X}", (c & 0xFF) as u8).unwrap();
                        } else {
                            self.emit_load_a_from(*val);
                            writeln!(self.asm, "    ld (hl), a").unwrap();
                        }
                        writeln!(self.asm, "    ld de, {dst1_imm}").unwrap();
                    } else {
                        load_dst_hl(self);
                        if let Some(&c) = self.const_vals.get(val) {
                            writeln!(self.asm, "    ld (hl), 0x{:02X}", (c & 0xFF) as u8).unwrap();
                        } else {
                            self.emit_load_a_from(*val);
                            writeln!(self.asm, "    ld (hl), a").unwrap();
                        }
                        writeln!(self.asm, "    ld d, h").unwrap();
                        writeln!(self.asm, "    ld e, l").unwrap();
                        writeln!(self.asm, "    inc de").unwrap();
                    }
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
                let const_ptr =
                    self.rom_addrs.contains_key(ptr) || self.const_addrs.contains_key(ptr);
                let all_direct_uses = const_ptr
                    && self.const_vals.contains_key(len)
                    && self.func.blocks.iter().all(|b| {
                        b.insts.iter().all(|u| {
                            !u.uses_val(*dst)
                                || matches!(u, IrInst::Call { args, .. } if args.contains(dst))
                                || matches!(u, IrInst::StoreLocal { val, .. } if *val == *dst)
                        }) && !matches!(b.term, Terminator::Branch { cond: v, .. } if v == *dst)
                    });
                if all_direct_uses {
                    self.const_slices.insert(*dst, (*ptr, *len));
                    return Ok(());
                }
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
                if let Some(&s) = self.val_from_local.get(slice)
                    && can_alias_local_load(self.func, bb_idx, inst_idx, *dst, s)
                {
                    let sa = self.layout.slot_addrs[&(self.func.id, s)];
                    self.val_aliases.insert(*dst, sa);
                    return Ok(());
                }
                if let Some(&(addr, is_u)) = self.const_addr_loads.get(slice) {
                    if is_u
                        && let Some(&ConstAddrSrc::Local(s)) = self.const_addr_srcs.get(slice)
                        && can_alias_local_load(self.func, bb_idx, inst_idx, *dst, s)
                    {
                        self.val_aliases.insert(*dst, addr);
                        return Ok(());
                    }
                    let sa_lo = self.fmt_val_addr_mem(*slice, 0);
                    writeln!(self.asm, "    ld hl, {sa_lo}").unwrap();
                    self.emit_store_dst_hl(*dst);
                    return Ok(());
                }
                if let Some(&(disp, _)) = self.ix_val_loads.get(slice) {
                    writeln!(self.asm, "    ld l, (ix + {disp})").unwrap();
                    writeln!(self.asm, "    ld h, (ix + {})", disp + 1).unwrap();
                    self.emit_store_dst_hl(*dst);
                    return Ok(());
                }
                let va = self.val_loc(*slice);
                let sa_lo = self.fmt_ram_mem(va);
                writeln!(self.asm, "    ld hl, {sa_lo}").unwrap();
                self.emit_store_dst_hl(*dst);
            }
            IrInst::SliceLen { dst, slice } => {
                if let Some(&s) = self.val_from_local.get(slice)
                    && can_alias_local_load(self.func, bb_idx, inst_idx, *dst, s)
                {
                    let sa = self.layout.slot_addrs[&(self.func.id, s)];
                    self.val_aliases.insert(*dst, sa.wrapping_add(2));
                    return Ok(());
                }
                if self.const_addr_loads.contains_key(slice) {
                    let sa_hi = self.fmt_val_addr_mem(*slice, 2);
                    writeln!(self.asm, "    ld hl, {sa_hi}").unwrap();
                    self.emit_store_dst_hl(*dst);
                    return Ok(());
                }
                if let Some(&(disp, _)) = self.ix_val_loads.get(slice) {
                    writeln!(self.asm, "    ld l, (ix + {})", disp + 2).unwrap();
                    writeln!(self.asm, "    ld h, (ix + {})", disp + 3).unwrap();
                    self.emit_store_dst_hl(*dst);
                    return Ok(());
                }
                let va = self.val_loc(*slice);
                let sa_hi = self.fmt_ram_mem(va.wrapping_add(2));
                writeln!(self.asm, "    ld hl, {sa_hi}").unwrap();
                self.emit_store_dst_hl(*dst);
            }
            IrInst::UnOp {
                dst, op, ty, val, ..
            } => {
                let da = self.val_addr(*dst);
                match (op, ty, self.program.type_size(ty)) {
                    (IrUnOp::Not, IrType::Bool, _) => {
                        self.emit_load_a_from(*val);
                        writeln!(self.asm, "    xor 1").unwrap();
                        self.emit_store_dst_a(*dst);
                    }
                    (IrUnOp::Not, _, 1) => {
                        self.emit_load_a_from(*val);
                        writeln!(self.asm, "    cpl").unwrap();
                        self.emit_store_dst_a(*dst);
                    }
                    (IrUnOp::Not, _, 2) => {
                        self.emit_load_hl_from(*val);
                        writeln!(self.asm, "    ld a, l").unwrap();
                        writeln!(self.asm, "    cpl").unwrap();
                        writeln!(self.asm, "    ld l, a").unwrap();
                        writeln!(self.asm, "    ld a, h").unwrap();
                        writeln!(self.asm, "    cpl").unwrap();
                        writeln!(self.asm, "    ld h, a").unwrap();
                        self.emit_store_dst_hl(*dst);
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
                        self.emit_store_dst_a(*dst);
                    }
                    (IrUnOp::Neg, _, 2) => {
                        self.emit_load_hl_from(*val);
                        writeln!(self.asm, "    xor a").unwrap();
                        writeln!(self.asm, "    sub l").unwrap();
                        writeln!(self.asm, "    ld l, a").unwrap();
                        writeln!(self.asm, "    sbc a, a").unwrap();
                        writeln!(self.asm, "    sub h").unwrap();
                        writeln!(self.asm, "    ld h, a").unwrap();
                        self.emit_store_dst_hl(*dst);
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
                if let Some(action) = self.bitpack_actions.get(dst).cloned() {
                    self.emit_bitpack_action(*dst, action);
                    return Ok(());
                }
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
                        self.emit_load_hl_from(src_v);
                        writeln!(self.asm, "    add hl, hl").unwrap();
                        writeln!(self.asm, "    add hl, hl").unwrap();
                        writeln!(self.asm, "    ld a, h").unwrap();
                    }
                    self.clear_hl_tracking();
                    self.emit_store_dst_a(*dst);
                    return Ok(());
                }
                if let Some(&src_v) = self.fused_hi8_u8.get(dst) {
                    if let Some(&c) = self.const_vals.get(&src_v) {
                        let b = ((c >> 8) & 0xFF) as u8;
                        if b == 0 {
                            writeln!(self.asm, "    xor a").unwrap();
                        } else {
                            writeln!(self.asm, "    ld a, 0x{b:02X}").unwrap();
                        }
                    } else if self.val_in_reg.get(&src_v) == Some(&RegLoc::BC) {
                        writeln!(self.asm, "    ld a, b").unwrap();
                    } else if let Some(&(disp, _)) = self.ix_val_loads.get(&src_v)
                        && disp <= 126
                    {
                        writeln!(self.asm, "    ld a, (ix + {})", disp + 1).unwrap();
                    } else {
                        let hi_mem = self.fmt_ram_mem(self.val_loc(src_v).wrapping_add(1));
                        writeln!(self.asm, "    ld a, {hi_mem}").unwrap();
                    }
                    self.emit_store_dst_a(*dst);
                    return Ok(());
                }
                let from_sz = self.program.type_size(from_ty);
                let to_sz = self.program.type_size(to_ty);
                let fixed_shift: i8 = match (from_ty, to_ty) {
                    (IrType::Int(_), IrType::Fixed { frac_bits, .. }) => *frac_bits as i8,
                    (IrType::Fixed { frac_bits, .. }, IrType::Int(_)) => -(*frac_bits as i8),
                    (IrType::Fixed { frac_bits: f1, .. }, IrType::Fixed { frac_bits: f2, .. }) => {
                        (*f2 as i8) - (*f1 as i8)
                    }
                    _ => 0,
                };
                if fixed_shift > 0 && *to_ty != IrType::Bool {
                    self.clear_hl_tracking();
                    let k = fixed_shift as u8;
                    match to_sz {
                        1 => {
                            self.emit_load_a_from(*val);
                            if k >= 8 {
                                writeln!(self.asm, "    xor a").unwrap();
                            } else {
                                for _ in 0..k {
                                    writeln!(self.asm, "    add a, a").unwrap();
                                }
                            }
                            self.emit_store_dst_a(*dst);
                        }
                        2 => {
                            if from_sz == 1 && k == 8 {
                                self.emit_load_a_from(*val);
                                writeln!(self.asm, "    ld h, a").unwrap();
                                writeln!(self.asm, "    ld l, 0").unwrap();
                            } else {
                                if from_sz == 1 {
                                    self.emit_load_a_from(*val);
                                    writeln!(self.asm, "    ld l, a").unwrap();
                                    if is_signed_ty(from_ty) {
                                        writeln!(self.asm, "    add a, a").unwrap();
                                        writeln!(self.asm, "    sbc a, a").unwrap();
                                        writeln!(self.asm, "    ld h, a").unwrap();
                                    } else {
                                        writeln!(self.asm, "    ld h, 0").unwrap();
                                    }
                                } else {
                                    self.emit_load_hl_from(*val);
                                }
                                crate::codegen::binop::emit_shift16_hl_const(
                                    self.asm,
                                    IrBinOp::Shl,
                                    is_signed_ty(to_ty),
                                    k,
                                );
                            }
                            self.emit_store_dst_hl(*dst);
                        }
                        4 => {
                            match from_sz {
                                1 => {
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
                                }
                                2 => {
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
                                }
                                _ => self.emit_load_hlde_from(*val),
                            }
                            if k >= 32 {
                                writeln!(self.asm, "    ld de, 0").unwrap();
                                writeln!(self.asm, "    ld hl, 0").unwrap();
                            } else if k > 0 {
                                let lp = self.fresh_label("fshl32");
                                writeln!(self.asm, "    push bc").unwrap();
                                writeln!(self.asm, "    ld b, {k}").unwrap();
                                writeln!(self.asm, "{lp}:").unwrap();
                                writeln!(self.asm, "    sla e").unwrap();
                                writeln!(self.asm, "    rl d").unwrap();
                                writeln!(self.asm, "    rl l").unwrap();
                                writeln!(self.asm, "    rl h").unwrap();
                                writeln!(self.asm, "    djnz {lp}").unwrap();
                                writeln!(self.asm, "    pop bc").unwrap();
                            }
                            self.emit_store_hlde_to(da, true);
                        }
                        _ => {}
                    }
                    return Ok(());
                }
                if fixed_shift < 0 && *to_ty != IrType::Bool {
                    self.clear_hl_tracking();
                    let k = (-fixed_shift) as u8;
                    let from_signed = is_signed_ty(from_ty);
                    match from_sz {
                        1 => {
                            self.emit_load_a_from(*val);
                            if k >= 8 {
                                if from_signed {
                                    writeln!(self.asm, "    add a, a").unwrap();
                                    writeln!(self.asm, "    sbc a, a").unwrap();
                                } else {
                                    writeln!(self.asm, "    xor a").unwrap();
                                }
                            } else {
                                let sinst = if from_signed { "sra a" } else { "srl a" };
                                for _ in 0..k {
                                    writeln!(self.asm, "    {sinst}").unwrap();
                                }
                            }
                            match to_sz {
                                1 => self.emit_store_dst_a(*dst),
                                2 => {
                                    writeln!(self.asm, "    ld l, a").unwrap();
                                    if from_signed {
                                        writeln!(self.asm, "    add a, a").unwrap();
                                        writeln!(self.asm, "    sbc a, a").unwrap();
                                        writeln!(self.asm, "    ld h, a").unwrap();
                                    } else {
                                        writeln!(self.asm, "    ld h, 0").unwrap();
                                    }
                                    self.emit_store_dst_hl(*dst);
                                }
                                4 => {
                                    writeln!(self.asm, "    ld e, a").unwrap();
                                    if from_signed {
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
                                _ => {}
                            }
                        }
                        2 => {
                            self.emit_load_hl_from(*val);
                            crate::codegen::binop::emit_shift16_hl_const(
                                self.asm,
                                IrBinOp::Shr,
                                from_signed,
                                k,
                            );
                            match to_sz {
                                1 => {
                                    writeln!(self.asm, "    ld a, l").unwrap();
                                    self.emit_store_dst_a(*dst);
                                }
                                2 => self.emit_store_dst_hl(*dst),
                                4 => {
                                    writeln!(self.asm, "    ex de, hl").unwrap();
                                    if from_signed {
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
                                _ => {}
                            }
                        }
                        4 => {
                            self.emit_load_hlde_from(*val);
                            if k >= 32 {
                                if from_signed {
                                    writeln!(self.asm, "    ld a, h").unwrap();
                                    writeln!(self.asm, "    add a, a").unwrap();
                                    writeln!(self.asm, "    sbc a, a").unwrap();
                                } else {
                                    writeln!(self.asm, "    xor a").unwrap();
                                }
                                writeln!(self.asm, "    ld e, a").unwrap();
                                writeln!(self.asm, "    ld d, a").unwrap();
                                writeln!(self.asm, "    ld l, a").unwrap();
                                writeln!(self.asm, "    ld h, a").unwrap();
                            } else if k > 0 {
                                let lp = self.fresh_label("fshr32");
                                writeln!(self.asm, "    push bc").unwrap();
                                writeln!(self.asm, "    ld b, {k}").unwrap();
                                writeln!(self.asm, "{lp}:").unwrap();
                                writeln!(
                                    self.asm,
                                    "    {}",
                                    if from_signed { "sra h" } else { "srl h" }
                                )
                                .unwrap();
                                writeln!(self.asm, "    rr l").unwrap();
                                writeln!(self.asm, "    rr d").unwrap();
                                writeln!(self.asm, "    rr e").unwrap();
                                writeln!(self.asm, "    djnz {lp}").unwrap();
                                writeln!(self.asm, "    pop bc").unwrap();
                            }
                            match to_sz {
                                1 => {
                                    writeln!(self.asm, "    ld a, e").unwrap();
                                    self.emit_store_dst_a(*dst);
                                }
                                2 => {
                                    writeln!(self.asm, "    ex de, hl").unwrap();
                                    self.emit_store_dst_hl(*dst);
                                }
                                4 => self.emit_store_hlde_to(da, true),
                                _ => {}
                            }
                        }
                        _ => {}
                    }
                    return Ok(());
                }
                if from_sz == 2 && to_sz == 2 && fixed_shift == 0 {
                    if let Some(&disp) = self.ix_addrs.get(val) {
                        self.ix_addrs.insert(*dst, disp);
                        return Ok(());
                    }
                    if let Some(&ca) = self.const_addrs.get(val) {
                        self.const_addrs.insert(*dst, ca);
                        if let Some(&src) = self.const_addr_srcs.get(val) {
                            self.const_addr_srcs.insert(*dst, src);
                        }
                        if !self.fa.direct_reg_dsts.contains_key(dst)
                            && is_const_addr_always_folded(
                                self.program,
                                self.func,
                                *dst,
                                self.fused_skip_insts,
                            )
                        {
                            return Ok(());
                        }
                    } else if let Some(ra) = self.rom_addrs.get(val).cloned() {
                        self.rom_addrs.insert(*dst, ra);
                        if !self.fa.direct_reg_dsts.contains_key(dst)
                            && is_const_addr_always_folded(
                                self.program,
                                self.func,
                                *dst,
                                self.fused_skip_insts,
                            )
                        {
                            return Ok(());
                        }
                    }
                }
                // Zero-cost no-op cast propagation for `(ix + d)` loads (e.g. `c.cur_freq as i16`)!
                if fixed_shift == 0
                    && !self.fa.direct_reg_dsts.contains_key(dst)
                    && let Some(&(disp, _)) = self.ix_val_loads.get(val)
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
                        self.fa.ix_local_slot,
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
                        self.emit_store_dst_a(*dst);
                    }
                    (1, 2) => {
                        self.clear_hl_tracking();
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
                        self.emit_store_dst_hl(*dst);
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
                            self.emit_store_dst_a(*dst);
                        } else {
                            // Little-endian: low byte is at va / (ix + disp)
                            self.emit_load_a_from(*val);
                            self.emit_store_dst_a(*dst);
                        }
                    }
                    (2, 2) => {
                        self.emit_load_hl_from(*val);
                        self.emit_store_dst_hl(*dst);
                    }
                    (1, 4) => {
                        self.clear_hl_tracking();
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
                        self.clear_hl_tracking();
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
                        self.clear_hl_tracking();
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
                            self.emit_store_dst_a(*dst);
                        } else {
                            self.emit_load_a_from(*val);
                            self.emit_store_dst_a(*dst);
                        }
                    }
                    (4, 2) => {
                        self.emit_load_hl_from(*val);
                        self.emit_store_dst_hl(*dst);
                    }
                    (4, 4) => {
                        self.emit_copy32_from_val(*val, da, true);
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
                let da_mem = self.fmt_dst_mem(*dst);
                match (base.byte_size(), *frac_bits) {
                    (2, 8) => {
                        // q8_8::int() -> direct high-byte read (0T shift!), sign/zero extended to 16-bit
                        if self.val_in_reg.get(val) == Some(&RegLoc::BC) {
                            writeln!(self.asm, "    ld a, b").unwrap();
                        } else if let Some(&(disp, _)) = self.ix_val_loads.get(val)
                            && disp <= 126
                        {
                            writeln!(self.asm, "    ld a, (ix + {})", disp + 1).unwrap();
                        } else if self.ix_val_loads.contains_key(val) {
                            self.emit_load_hl_from(*val);
                            writeln!(self.asm, "    ld a, h").unwrap();
                        } else {
                            let va_hi_mem = self.fmt_val_addr_mem(*val, 1);
                            writeln!(self.asm, "    ld a, {va_hi_mem}").unwrap();
                        }
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
                        self.emit_load_hl_from(*val);
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
                        self.emit_load_hl_from(*val);
                        writeln!(self.asm, "    ld {da_mem}, hl").unwrap();
                    }
                    (2, fb) => {
                        let lbl = self.fresh_label("fxint");
                        self.emit_load_hl_from(*val);
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
                        self.emit_load_a_from(*val);
                        writeln!(self.asm, "    ld {da_mem}, a").unwrap();
                    }
                    (1, fb) => {
                        let lbl = self.fresh_label("fxint");
                        self.emit_load_a_from(*val);
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
                if let Some(&p) = self.const_vals.get(port) {
                    writeln!(self.asm, "    in a, (0x{:02X})", (p & 0xFF) as u8).unwrap();
                } else {
                    let pa_mem = self.fmt_val_mem(*port);
                    writeln!(self.asm, "    ld a, {pa_mem}").unwrap();
                    writeln!(self.asm, "    ld c, a").unwrap();
                    writeln!(self.asm, "    in a, (c)").unwrap();
                }
                self.emit_store_dst_a(*dst);
            }
            IrInst::Call {
                dst,
                func: callee_id,
                args,
            } => {
                let cidx = *callee_id as usize;
                let callee_fa = &self.analysis.fns[cidx];
                let reg_p1 = callee_fa.uses_reg_param1;
                let reg_p2 = callee_fa.uses_reg_param2;
                let caller_hoisted_static_ix = self.fa.ix_static_elem.as_ref().is_some_and(|ise| {
                    ise.hoist_param.is_some() && (!self.opts.check_bounds || ise.len == 0)
                });
                let callee_fast_entry =
                    callee_fa.uses_ix_static_fast_entry(self.opts.check_bounds);
                let callee_only_fast = callee_fa.uses_ix_static_only_fast(self.opts.check_bounds);
                let use_ix_static_fast_call = caller_hoisted_static_ix
                    && callee_fast_entry
                    && self.fa.ix_static_fast_calls.contains(&(bb_idx, inst_idx));
                let skip_fast_idx_slot: Option<SlotId> = if use_ix_static_fast_call
                    && let Some(c_ise) = callee_fa.ix_static_elem.as_ref()
                    && c_ise.hoist_param_dead_in_body
                {
                    c_ise.hoist_param
                } else {
                    None
                };
                let ix_reg_slot = if callee_fa.ix_in_reg {
                    callee_fa.ix_param_slot
                } else {
                    None
                };
                let ram_start = if reg_p2 {
                    3
                } else if reg_p1 {
                    2
                } else {
                    1
                };
                // Write arguments ram_start..N directly into callee's Uchiguri parameter slots
                for (idx, &arg_val) in args.iter().enumerate().skip(ram_start) {
                    if skip_fast_idx_slot == Some(idx as SlotId) {
                        continue;
                    }
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
                                } else if let Some(&(ptr, len)) = self.const_slices.get(&arg_val) {
                                    let pa_lo = self.fmt_ram_mem(pa);
                                    let pa_hi = self.fmt_ram_mem(pa.wrapping_add(2));
                                    self.emit_load_hl_from(ptr);
                                    writeln!(self.asm, "    ld {pa_lo}, hl").unwrap();
                                    self.emit_load_hl_from(len);
                                    writeln!(self.asm, "    ld {pa_hi}, hl").unwrap();
                                } else {
                                    self.emit_copy32_from_val(arg_val, pa, true);
                                }
                            }
                            _ => {}
                        }
                    }
                }
                // Pass argument 1 in E (1-byte) or DE (2-byte) when eligible and not passed in IX
                if reg_p1
                    && ix_reg_slot != Some(1)
                    && skip_fast_idx_slot != Some(1)
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
                    && skip_fast_idx_slot != Some(0)
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
                if reg_p2 && skip_fast_idx_slot != Some(2) && let Some(&a2) = args.get(2) {
                    let p2_sz = self.program.type_size(&self.program.functions[cidx].locals[2].ty);
                    let a0_in_a = ix_reg_slot != Some(0)
                        && skip_fast_idx_slot != Some(0)
                        && self
                            .program
                            .functions[cidx]
                            .locals
                            .first()
                            .is_some_and(|p| self.program.type_size(&p.ty) == 1);
                    if p2_sz == 2 {
                        if self.ix_addrs.get(&a2) == Some(&0) {
                            writeln!(self.asm, "    push ix").unwrap();
                            writeln!(self.asm, "    pop bc").unwrap();
                        } else if let Some(&c) = self.const_vals.get(&a2) {
                            writeln!(self.asm, "    ld bc, 0x{:04X}", c as u16).unwrap();
                        } else if let Some(&(addr, is_u)) = self.const_addrs.get(&a2) {
                            let imm = self.fmt_addr_imm(addr, is_u);
                            writeln!(self.asm, "    ld bc, {imm}").unwrap();
                        } else if let Some((sym, off)) = self.rom_addrs.get(&a2) {
                            if *off == 0 {
                                writeln!(self.asm, "    ld bc, {sym}").unwrap();
                            } else {
                                writeln!(self.asm, "    ld bc, {sym} + {off}").unwrap();
                            }
                        } else if self.val_in_reg.get(&a2) == Some(&RegLoc::BC) {
                            // already in BC
                        } else if let Some(&(disp, _)) = self.ix_val_loads.get(&a2) {
                            writeln!(self.asm, "    ld c, (ix + {disp})").unwrap();
                            writeln!(self.asm, "    ld b, (ix + {})", disp + 1).unwrap();
                        } else {
                            let va_mem = self.fmt_val_mem(a2);
                            writeln!(self.asm, "    ld bc, {va_mem}").unwrap();
                        }
                    } else if p2_sz == 1 {
                        if let Some(&c) = self.const_vals.get(&a2) {
                            writeln!(self.asm, "    ld c, 0x{:02X}", (c & 0xFF) as u8).unwrap();
                        } else if let Some(&reg) = self.val_in_reg.get(&a2) {
                            match reg {
                                RegLoc::C | RegLoc::BC => {}
                                RegLoc::B => writeln!(self.asm, "    ld c, b").unwrap(),
                            }
                        } else if let Some(&(disp, _)) = self.ix_val_loads.get(&a2) {
                            writeln!(self.asm, "    ld c, (ix + {disp})").unwrap();
                        } else {
                            let va_mem = self.fmt_val_mem(a2);
                            if a0_in_a {
                                writeln!(self.asm, "    push af").unwrap();
                                writeln!(self.asm, "    ld a, {va_mem}").unwrap();
                                writeln!(self.asm, "    ld c, a").unwrap();
                                writeln!(self.asm, "    pop af").unwrap();
                            } else {
                                writeln!(self.asm, "    ld a, {va_mem}").unwrap();
                                writeln!(self.asm, "    ld c, a").unwrap();
                            }
                        }
                    }
                }
                // Pass shared-IX argument directly in IX (without clobbering already-loaded A/HL, E/DE, or C/BC)
                let mut caller_saved_ix = false;
                if let Some(p) = ix_reg_slot
                    && let Some(&ix_arg) = args.get(p as usize)
                    && self.ix_addrs.get(&ix_arg) != Some(&0)
                {
                    if self.fa.uses_ix() {
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
                        // Note: if `a2` overwrote `BC`, this is impossible because `ix_arg != a2`? Wait: if `need_save_bc` pushed caller's BC, `a2` won't use `ix_arg` if `ix_reg_slot != 2`.
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
                let callee_sym = z80_sym(&self.program.functions[cidx].name);
                if use_ix_static_fast_call && !callee_only_fast {
                    writeln!(self.asm, "    call {callee_sym}__ix").unwrap();
                } else {
                    writeln!(self.asm, "    call {callee_sym}").unwrap();
                }
                if caller_saved_ix {
                    writeln!(self.asm, "    pop ix").unwrap();
                }
                if need_save_bc {
                    writeln!(self.asm, "    pop bc").unwrap();
                } else {
                    self.val_in_reg
                        .retain(|_, &mut vr| !callee_clob.intersects(vr.mask()));
                }
                if let Some(d) = dst {
                    let da = self.val_addr(*d);
                    let ret_sz = self.program.type_size(&self.program.functions[cidx].ret_ty);
                    match ret_sz {
                        1 => self.emit_store_dst_a(*d),
                        2 => self.emit_store_dst_hl(*d),
                        4 => self.emit_store_hlde_to(da, true),
                        _ => {
                            debug_assert!(false, "unexpected non-unit return size {ret_sz}");
                        }
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
                                self.emit_load_a_from(*arg_val);
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
                                let aa_mem = self.fmt_val_addr_mem(*arg_val, 0);
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
                    self.clear_hl_tracking();
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
                let mdef = &self.program.machines[*machine_id as usize];
                if self.ix_addrs.get(machine_addr) == Some(&0) && mdef.tag_offset <= 127 {
                    writeln!(self.asm, "    ld a, (ix + {})", mdef.tag_offset).unwrap();
                    self.emit_store_dst_a(*dst);
                } else {
                    self.emit_load_hl_from(*machine_addr);
                    if mdef.tag_offset > 0 {
                        writeln!(self.asm, "    ld de, 0x{:04X}", mdef.tag_offset).unwrap();
                        writeln!(self.asm, "    add hl, de").unwrap();
                    }
                    writeln!(self.asm, "    ld a, (hl)").unwrap();
                    self.emit_store_dst_a(*dst);
                }
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
                // (deduped when sibling states share overlapping Uchiguri slots).
                let mut written_slots: std::collections::HashSet<(u16, ValueId)> =
                    std::collections::HashSet::new();
                for st in &mdef.states {
                    if let Some(&fid) = st.methods.get(method) {
                        let callee = &self.program.functions[fid as usize];
                        for (idx, &arg_val) in args.iter().enumerate() {
                            if let Some(param_slot) = callee.locals.get(idx + 1) {
                                let pa = self.layout.slot_addrs[&(callee.id, param_slot.id)];
                                if !written_slots.insert((pa, arg_val)) {
                                    continue;
                                }
                                let pa_mem = self.fmt_ram_mem(pa);
                                if self.ix_addrs.get(&arg_val) == Some(&0) {
                                    writeln!(self.asm, "    ld {pa_mem}, ix").unwrap();
                                    continue;
                                }
                                match self.program.type_size(&param_slot.ty) {
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
                                        } else if let Some(&(ptr, len)) =
                                            self.const_slices.get(&arg_val)
                                        {
                                            let pa_lo = self.fmt_ram_mem(pa);
                                            let pa_hi = self.fmt_ram_mem(pa.wrapping_add(2));
                                            self.emit_load_hl_from(ptr);
                                            writeln!(self.asm, "    ld {pa_lo}, hl").unwrap();
                                            self.emit_load_hl_from(len);
                                            writeln!(self.asm, "    ld {pa_hi}, hl").unwrap();
                                        } else {
                                            self.emit_copy32_from_val(arg_val, pa, true);
                                        }
                                    }
                                    psz => {
                                        debug_assert!(
                                            false,
                                            "unexpected StateDispatch arg size {psz}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                if self.ix_addrs.get(machine_addr) == Some(&0) {
                    writeln!(self.asm, "    push ix").unwrap();
                    writeln!(self.asm, "    pop hl").unwrap();
                } else {
                    self.emit_load_hl_from(*machine_addr);
                }
                writeln!(self.asm, "    call {}_disp_{method}", mdef.name).unwrap();
                if let Some(d) = dst {
                    let ret_sz = mdef
                        .states
                        .iter()
                        .find_map(|st| st.methods.get(method))
                        .map(|&fid| {
                            self.program
                                .type_size(&self.program.functions[fid as usize].ret_ty)
                        })
                        .unwrap_or(2);
                    let da = self.val_addr(*d);
                    let da_mem = self.fmt_ram_mem(da);
                    match ret_sz {
                        1 => writeln!(self.asm, "    ld {da_mem}, a").unwrap(),
                        2 => writeln!(self.asm, "    ld {da_mem}, hl").unwrap(),
                        4 => {
                            let da2 = self.fmt_ram_mem(da.wrapping_add(2));
                            writeln!(self.asm, "    ld {da_mem}, de").unwrap();
                            writeln!(self.asm, "    ld {da2}, hl").unwrap();
                        }
                        _ => {
                            debug_assert!(false, "unexpected non-unit return size {ret_sz}");
                        }
                    }
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
                if self.plan.machine_has_boundaries[*machine_id as usize] {
                    // Stage payload fields into dedicated Uchiguri staging area before Resident trampoline swaps overlays
                    let mut copy_start = u16::MAX;
                    let mut copy_end = 0u16;
                    if !field_vals.is_empty() {
                        let stage_base = self.layout.machine_stage_addr.ok_or_else(|| {
                            "missing machine_stage_addr for boundary machine transition".to_string()
                        })?;
                        for (rel_off, vid, fty) in field_vals {
                            let fsz = self.program.type_size(fty);
                            if fsz == 0 {
                                continue;
                            }
                            copy_start = copy_start.min(*rel_off);
                            copy_end = copy_end.max(rel_off.saturating_add(fsz));
                            let dst_addr = stage_base.wrapping_add(*rel_off);
                            let dst_mem = self.fmt_ram_mem(dst_addr);
                            let dst_imm = self.fmt_ram_imm(dst_addr);
                            if !fty.is_scalar() {
                                if let Some(&disp) = self.ix_addrs.get(vid) {
                                    writeln!(self.asm, "    push ix").unwrap();
                                    writeln!(self.asm, "    pop hl").unwrap();
                                    if disp != 0 {
                                        writeln!(
                                            self.asm,
                                            "    ld de, 0x{:04X}",
                                            disp as i16 as u16
                                        )
                                        .unwrap();
                                        writeln!(self.asm, "    add hl, de").unwrap();
                                    }
                                } else {
                                    self.emit_load_hl_from(*vid);
                                }
                                writeln!(self.asm, "    ld de, {dst_imm}").unwrap();
                                writeln!(self.asm, "    ld bc, 0x{fsz:04X}").unwrap();
                                writeln!(self.asm, "    ldir").unwrap();
                                continue;
                            }
                            match fsz {
                                1 => {
                                    self.emit_load_a_from(*vid);
                                    writeln!(self.asm, "    ld {dst_mem}, a").unwrap();
                                }
                                2 => {
                                    if self.ix_addrs.get(vid) == Some(&0) {
                                        writeln!(self.asm, "    push ix").unwrap();
                                        writeln!(self.asm, "    pop hl").unwrap();
                                    } else {
                                        self.emit_load_hl_from(*vid);
                                    }
                                    writeln!(self.asm, "    ld {dst_mem}, hl").unwrap();
                                }
                                4 => {
                                    if let Some(&c) = self.const_vals.get(vid) {
                                        self.emit_const32_to(dst_addr, true, c);
                                    } else if let Some(&(ptr, len)) = self.const_slices.get(vid) {
                                        let d0 = self.fmt_ram_mem(dst_addr);
                                        let d2 = self.fmt_ram_mem(dst_addr.wrapping_add(2));
                                        self.emit_load_hl_from(ptr);
                                        writeln!(self.asm, "    ld {d0}, hl").unwrap();
                                        self.emit_load_hl_from(len);
                                        writeln!(self.asm, "    ld {d2}, hl").unwrap();
                                    } else {
                                        self.emit_copy32(self.val_loc(*vid), true, dst_addr, true);
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    if copy_end <= copy_start {
                        copy_start = 0;
                        copy_end = 0;
                    }
                    let key = super::TrampolineKey {
                        machine_id: *machine_id,
                        from_state: *from_state,
                        to_state: *to_state,
                        copy_start,
                        copy_end,
                    };
                    let tr_label = key.label();
                    self.runtime_needs.trampolines.insert(key);

                    if self.ix_addrs.get(machine_addr) == Some(&0) {
                        writeln!(self.asm, "    push ix").unwrap();
                        writeln!(self.asm, "    pop hl").unwrap();
                    } else {
                        self.emit_load_hl_from(*machine_addr);
                    }
                    if from_state.is_none() && self.opts.check_transitions {
                        self.runtime_needs.panic_stub = true;
                        writeln!(self.asm, "    ld de, 0x{site_id:04X}").unwrap();
                    }
                    if self.cur_bb_is_tail_return_none && inst_idx + 1 == self.cur_bb_inst_len {
                        if self.body_save_ix {
                            writeln!(self.asm, "    pop ix").unwrap();
                        }
                        writeln!(self.asm, "    jp {tr_label}").unwrap();
                        self.tail_jumped_trampoline = true;
                    } else {
                        writeln!(self.asm, "    call {tr_label}").unwrap();
                    }
                    return Ok(());
                }

                let ma_mem = self.fmt_val_mem(*machine_addr);
                let ok_lbl = if from_state.is_none() && self.opts.check_transitions {
                    self.runtime_needs.panic_stub = true;
                    Some(self.fresh_label("tr_ok"))
                } else {
                    None
                };
                let mdef = &self.program.machines[*machine_id as usize];
                let load_ma_hl = |this: &mut Self| {
                    if this.ix_addrs.get(machine_addr) == Some(&0) {
                        writeln!(this.asm, "    push ix").unwrap();
                        writeln!(this.asm, "    pop hl").unwrap();
                    } else {
                        writeln!(this.asm, "    ld hl, {ma_mem}").unwrap();
                    }
                };

                // 1. Dynamic transition verification when `from_state` is unknown (eliminated in Release!)
                if let Some(ok_lbl) = ok_lbl {
                    if self.ix_addrs.get(machine_addr) == Some(&0) && mdef.tag_offset <= 127 {
                        writeln!(self.asm, "    ld a, (ix + {})", mdef.tag_offset).unwrap();
                    } else {
                        load_ma_hl(self);
                        if mdef.tag_offset > 0 {
                            writeln!(self.asm, "    ld de, 0x{:04X}", mdef.tag_offset).unwrap();
                            writeln!(self.asm, "    add hl, de").unwrap();
                        }
                        writeln!(self.asm, "    ld a, (hl)").unwrap();
                    }
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

                // 2. Call current state's on_exit chain bottom-up strictly below LCA
                if let Some(from) = from_state {
                    let lca = mdef.lca(*from, *to_state);
                    for anc in mdef.ancestors(*from).into_iter().rev() {
                        let below_lca = lca.is_none_or(|l| anc != l && mdef.is_ancestor(l, anc));
                        if below_lca && let Some(exit_fn) = mdef.states[anc as usize].on_exit {
                            let ename = super::resolve_fn_target(
                                self.program,
                                self.plan,
                                self.resolved_overlay_syms,
                                self.pass1_placeholders,
                                exit_fn,
                            );
                            load_ma_hl(self);
                            writeln!(self.asm, "    call {ename}").unwrap();
                        }
                    }
                } else if mdef.states.iter().any(|s| s.on_exit.is_some()) {
                    if mdef.states.iter().all(|s| s.parent.is_none()) {
                        load_ma_hl(self);
                        writeln!(self.asm, "    call {}_disp_on_exit", mdef.name).unwrap();
                    } else {
                        if self.ix_addrs.get(machine_addr) == Some(&0) && mdef.tag_offset <= 127 {
                            writeln!(self.asm, "    ld a, (ix + {})", mdef.tag_offset).unwrap();
                        } else {
                            load_ma_hl(self);
                            if mdef.tag_offset > 0 {
                                writeln!(self.asm, "    ld de, 0x{:04X}", mdef.tag_offset).unwrap();
                                writeln!(self.asm, "    add hl, de").unwrap();
                            }
                            writeln!(self.asm, "    ld a, (hl)").unwrap();
                        }
                        let exit_done = self.fresh_label("exit_done");
                        for st in mdef.states.iter().filter(|s| s.is_leaf) {
                            let lca = mdef.lca(st.id, *to_state);
                            let fns: Vec<u32> = mdef
                                .ancestors(st.id)
                                .into_iter()
                                .rev()
                                .filter(|&a| lca.is_none_or(|l| a != l && mdef.is_ancestor(l, a)))
                                .filter_map(|a| mdef.states[a as usize].on_exit)
                                .collect();
                            if !fns.is_empty() {
                                let next_lbl = self.fresh_label("exit_nx");
                                writeln!(self.asm, "    cp {}", st.id).unwrap();
                                writeln!(self.asm, "    jr nz, {next_lbl}").unwrap();
                                for fid in fns {
                                    let ename = z80_sym(&self.program.functions[fid as usize].name);
                                    load_ma_hl(self);
                                    writeln!(self.asm, "    call {ename}").unwrap();
                                }
                                writeln!(self.asm, "    jp {exit_done}").unwrap();
                                writeln!(self.asm, "{next_lbl}:").unwrap();
                            }
                        }
                        writeln!(self.asm, "{exit_done}:").unwrap();
                    }
                }

                let need_old_state_for_enter = from_state.is_none()
                    && mdef.states.iter().any(|s| {
                        s.parent.is_some()
                            && (s.on_enter.is_some()
                                || super::runtime::has_zero_state_buffers(
                                    self.program,
                                    *machine_id,
                                    s.id,
                                ))
                    });
                if need_old_state_for_enter {
                    if self.ix_addrs.get(machine_addr) == Some(&0) && mdef.tag_offset <= 127 {
                        writeln!(self.asm, "    ld a, (ix + {})", mdef.tag_offset).unwrap();
                    } else {
                        load_ma_hl(self);
                        if mdef.tag_offset > 0 {
                            writeln!(self.asm, "    ld de, 0x{:04X}", mdef.tag_offset).unwrap();
                            writeln!(self.asm, "    add hl, de").unwrap();
                        }
                        writeln!(self.asm, "    ld a, (hl)").unwrap();
                    }
                    writeln!(self.asm, "    push af").unwrap();
                }

                // 3. Write pre-evaluated payload fields into overlaid RAM region
                for (rel_off, vid, fty) in field_vals {
                    let fsz = self.program.type_size(fty);
                    if fsz == 0 {
                        continue;
                    }
                    let total_off = mdef.payload_offset + *rel_off;
                    if self.ix_addrs.get(machine_addr) == Some(&0)
                        && fty.is_scalar()
                        && total_off + fsz <= 128
                    {
                        match fsz {
                            1 => {
                                if let Some(&c) = self.const_vals.get(vid) {
                                    writeln!(
                                        self.asm,
                                        "    ld (ix + {total_off}), 0x{:02X}",
                                        (c & 0xFF) as u8
                                    )
                                    .unwrap();
                                } else {
                                    self.emit_load_a_from(*vid);
                                    writeln!(self.asm, "    ld (ix + {total_off}), a").unwrap();
                                }
                            }
                            2 => {
                                if let Some(&c) = self.const_vals.get(vid) {
                                    let w = c as u16;
                                    writeln!(
                                        self.asm,
                                        "    ld (ix + {total_off}), 0x{:02X}",
                                        (w & 0xFF) as u8
                                    )
                                    .unwrap();
                                    writeln!(
                                        self.asm,
                                        "    ld (ix + {}), 0x{:02X}",
                                        total_off + 1,
                                        (w >> 8) as u8
                                    )
                                    .unwrap();
                                } else {
                                    self.emit_load_hl_from(*vid);
                                    writeln!(self.asm, "    ld (ix + {total_off}), l").unwrap();
                                    writeln!(self.asm, "    ld (ix + {}), h", total_off + 1)
                                        .unwrap();
                                }
                            }
                            4 => {
                                if let Some(&c) = self.const_vals.get(vid) {
                                    for (idx, b) in (c as u32).to_le_bytes().into_iter().enumerate()
                                    {
                                        writeln!(
                                            self.asm,
                                            "    ld (ix + {}), 0x{b:02X}",
                                            total_off + idx as u16
                                        )
                                        .unwrap();
                                    }
                                } else {
                                    self.emit_load_hlde_from(*vid);
                                    writeln!(self.asm, "    ld (ix + {total_off}), e").unwrap();
                                    writeln!(self.asm, "    ld (ix + {}), d", total_off + 1)
                                        .unwrap();
                                    writeln!(self.asm, "    ld (ix + {}), l", total_off + 2)
                                        .unwrap();
                                    writeln!(self.asm, "    ld (ix + {}), h", total_off + 3)
                                        .unwrap();
                                }
                            }
                            _ => {}
                        }
                        continue;
                    }
                    load_ma_hl(self);
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
                            self.emit_load_a_from(*vid);
                            writeln!(self.asm, "    ld (hl), a").unwrap();
                        }
                        2 => {
                            self.emit_load_de_from(*vid);
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
                if self.ix_addrs.get(machine_addr) == Some(&0) && mdef.tag_offset <= 126 {
                    writeln!(
                        self.asm,
                        "    ld (ix + {}), {}",
                        mdef.tag_offset, *to_state
                    )
                    .unwrap();
                    writeln!(self.asm, "    ld (ix + {}), 0", mdef.tag_offset + 1).unwrap();
                } else {
                    load_ma_hl(self);
                    if mdef.tag_offset > 0 {
                        writeln!(self.asm, "    ld de, 0x{:04X}", mdef.tag_offset).unwrap();
                        writeln!(self.asm, "    add hl, de").unwrap();
                    }
                    writeln!(self.asm, "    ld (hl), {}", *to_state).unwrap();
                    writeln!(self.asm, "    inc hl").unwrap();
                    writeln!(self.asm, "    ld (hl), 0").unwrap();
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

                // 6. Zero `#[zero_on_enter]` state buffers and call target state's on_enter chain top-down strictly below LCA
                if let Some(from) = from_state {
                    let lca = mdef.lca(*from, *to_state);
                    let enter_states: Vec<u8> = mdef
                        .ancestors(*to_state)
                        .into_iter()
                        .filter(|&anc| lca.is_none_or(|l| anc != l && mdef.is_ancestor(l, anc)))
                        .collect();
                    if super::runtime::emit_zero_state_buffers(
                        &mut self.asm,
                        self.program,
                        self.layout,
                        *machine_id,
                        &enter_states,
                    ) {
                        self.clear_hl_tracking();
                        self.val_in_reg.clear();
                    }
                    for anc in enter_states {
                        if let Some(enter_fn) = mdef.states[anc as usize].on_enter {
                            let ename = z80_sym(&self.program.functions[enter_fn as usize].name);
                            load_ma_hl(self);
                            writeln!(self.asm, "    call {ename}").unwrap();
                        }
                    }
                } else if need_old_state_for_enter {
                    writeln!(self.asm, "    pop af").unwrap();
                    let enter_done = self.fresh_label("enter_done");
                    let ancs = mdef.ancestors(*to_state);
                    for anc in ancs.iter().rev() {
                        let st_anc = &mdef.states[*anc as usize];
                        let next_lbl = self.fresh_label("ent_nx");
                        let (lo, hi) = st_anc.leaf_range;
                        if lo > 0 {
                            writeln!(self.asm, "    cp {lo}").unwrap();
                            writeln!(self.asm, "    jr c, {next_lbl}").unwrap();
                        }
                        if hi < 255 {
                            writeln!(self.asm, "    cp {}", hi + 1).unwrap();
                            writeln!(self.asm, "    jr nc, {next_lbl}").unwrap();
                        }
                        let enter_states: Vec<u8> = ancs
                            .iter()
                            .copied()
                            .filter(|&a2| a2 != *anc && mdef.is_ancestor(*anc, a2))
                            .collect();
                        if super::runtime::emit_zero_state_buffers(
                            &mut self.asm,
                            self.program,
                            self.layout,
                            *machine_id,
                            &enter_states,
                        ) {
                            self.clear_hl_tracking();
                            self.val_in_reg.clear();
                        }
                        for a2 in enter_states {
                            if let Some(enter_fn) = mdef.states[a2 as usize].on_enter {
                                let ename = z80_sym(&self.program.functions[enter_fn as usize].name);
                                load_ma_hl(self);
                                writeln!(self.asm, "    call {ename}").unwrap();
                            }
                        }
                        writeln!(self.asm, "    jp {enter_done}").unwrap();
                        writeln!(self.asm, "{next_lbl}:").unwrap();
                    }
                    // Fallback: outside all ancestors of `to_state`
                    if super::runtime::emit_zero_state_buffers(
                        &mut self.asm,
                        self.program,
                        self.layout,
                        *machine_id,
                        &ancs,
                    ) {
                        self.clear_hl_tracking();
                        self.val_in_reg.clear();
                    }
                    for &a2 in &ancs {
                        if let Some(enter_fn) = mdef.states[a2 as usize].on_enter {
                            let ename = z80_sym(&self.program.functions[enter_fn as usize].name);
                            load_ma_hl(self);
                            writeln!(self.asm, "    call {ename}").unwrap();
                        }
                    }
                    writeln!(self.asm, "{enter_done}:").unwrap();
                } else {
                    if super::runtime::emit_zero_state_buffers(
                        &mut self.asm,
                        self.program,
                        self.layout,
                        *machine_id,
                        &[*to_state],
                    ) {
                        self.clear_hl_tracking();
                        self.val_in_reg.clear();
                    }
                    if let Some(enter_fn) = mdef.states[*to_state as usize].on_enter {
                        let ename = z80_sym(&self.program.functions[enter_fn as usize].name);
                        load_ma_hl(self);
                        writeln!(self.asm, "    call {ename}").unwrap();
                    }
                }
            }
            IrInst::PoolAlloc {
                dst,
                pool_addr,
                init_addr,
                elem_size,
                capacity,
            } => {
                let bm_off = elem_size * capacity;
                if *capacity > 0 && *capacity <= 8 {
                    let scan_lbl = self.fresh_label("palloc_scan");
                    let found_lbl = self.fresh_label("palloc_found");
                    let done_lbl = self.fresh_label("palloc_done");
                    if let Some(&(paddr, is_u)) = self.const_addrs.get(pool_addr) {
                        let bm_imm = self.fmt_addr_imm(paddr.wrapping_add(bm_off), is_u);
                        writeln!(self.asm, "    ld hl, {bm_imm}").unwrap();
                    } else {
                        self.emit_load_hl_from(*pool_addr);
                        if bm_off > 0 {
                            writeln!(self.asm, "    ld de, 0x{bm_off:04X}").unwrap();
                            writeln!(self.asm, "    add hl, de").unwrap();
                        }
                    }
                    writeln!(self.asm, "    ld d, (hl)").unwrap();
                    writeln!(self.asm, "    ld bc, 0x0100").unwrap();
                    writeln!(self.asm, "{scan_lbl}:").unwrap();
                    writeln!(self.asm, "    ld a, d").unwrap();
                    writeln!(self.asm, "    and b").unwrap();
                    writeln!(self.asm, "    jr z, {found_lbl}").unwrap();
                    writeln!(self.asm, "    sla b").unwrap();
                    writeln!(self.asm, "    inc c").unwrap();
                    writeln!(self.asm, "    ld a, c").unwrap();
                    writeln!(self.asm, "    cp {}", *capacity as u8).unwrap();
                    writeln!(self.asm, "    jr c, {scan_lbl}").unwrap();
                    writeln!(self.asm, "    ld a, 0xFF").unwrap();
                    writeln!(self.asm, "    jr {done_lbl}").unwrap();
                    writeln!(self.asm, "{found_lbl}:").unwrap();
                    writeln!(self.asm, "    ld a, d").unwrap();
                    writeln!(self.asm, "    or b").unwrap();
                    writeln!(self.asm, "    ld (hl), a").unwrap();
                    if *elem_size > 0 {
                        writeln!(self.asm, "    push bc").unwrap();
                        writeln!(self.asm, "    ld l, c").unwrap();
                        writeln!(self.asm, "    ld h, 0").unwrap();
                        emit_mul_hl_const(*elem_size, self.asm);
                        self.emit_load_de_from(*pool_addr);
                        writeln!(self.asm, "    add hl, de").unwrap();
                        writeln!(self.asm, "    ex de, hl").unwrap();
                        if self.ix_addrs.get(init_addr) == Some(&0) {
                            writeln!(self.asm, "    push ix").unwrap();
                            writeln!(self.asm, "    pop hl").unwrap();
                        } else {
                            self.emit_load_hl_from(*init_addr);
                        }
                        writeln!(self.asm, "    ld bc, 0x{elem_size:04X}").unwrap();
                        writeln!(self.asm, "    ldir").unwrap();
                        writeln!(self.asm, "    pop bc").unwrap();
                    }
                    writeln!(self.asm, "    ld a, c").unwrap();
                    writeln!(self.asm, "{done_lbl}:").unwrap();
                    self.emit_store_dst_a(*dst);
                    return Ok(());
                }
                let pa_mem = self.fmt_val_mem(*pool_addr);
                let ia_mem = self.fmt_val_mem(*init_addr);
                let da_mem = self.fmt_dst_mem(*dst);
                let scan_lbl = self.fresh_label("palloc_scan");
                let bit_lbl = self.fresh_label("palloc_bit");
                let found_lbl = self.fresh_label("palloc_found");
                let done_lbl = self.fresh_label("palloc_done");
                let shift_done = self.fresh_label("palloc_sd");
                let shift_loop = self.fresh_label("palloc_sl");

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
                self.emit_load_a_from(*handle);
                if let Some((ref ok_r, _)) = ok_rng {
                    writeln!(self.asm, "    cp {}", *capacity as u8).unwrap();
                    writeln!(self.asm, "    jr c, {ok_r}").unwrap();
                    writeln!(self.asm, "    ld a, {}", PanicCode::InvalidHandle as u8).unwrap();
                    writeln!(self.asm, "    ld hl, 0x{site_id:04X}").unwrap();
                    writeln!(self.asm, "    call __aun_panic").unwrap();
                    writeln!(self.asm, "{ok_r}:").unwrap();
                }
                if *capacity > 0 && *capacity <= 8 {
                    writeln!(self.asm, "    ld b, 1").unwrap();
                    writeln!(self.asm, "    or a").unwrap();
                    writeln!(self.asm, "    jr z, {sd}").unwrap();
                    writeln!(self.asm, "{sl}:").unwrap();
                    writeln!(self.asm, "    sla b").unwrap();
                    writeln!(self.asm, "    dec a").unwrap();
                    writeln!(self.asm, "    jr nz, {sl}").unwrap();
                    writeln!(self.asm, "{sd}:").unwrap();
                    if let Some(&(paddr, is_u)) = self.const_addrs.get(pool_addr) {
                        let bm_imm = self.fmt_addr_imm(paddr.wrapping_add(bm_off), is_u);
                        writeln!(self.asm, "    ld hl, {bm_imm}").unwrap();
                    } else {
                        self.emit_load_hl_from(*pool_addr);
                        if bm_off > 0 {
                            writeln!(self.asm, "    ld de, 0x{bm_off:04X}").unwrap();
                            writeln!(self.asm, "    add hl, de").unwrap();
                        }
                    }
                } else {
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
                }
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
            IrInst::CriticalEnter => {
                let asm_str = self
                    .program
                    .target
                    .as_ref()
                    .and_then(|t| t.interrupt_controllers.first())
                    .map(|ic| ic.disable_asm.as_str())
                    .unwrap_or("di");
                for line in asm_str.split(';') {
                    let t = line.trim();
                    if !t.is_empty() {
                        writeln!(self.asm, "    {t}").unwrap();
                    }
                }
            }
            IrInst::CriticalExit => {
                let asm_str = self
                    .program
                    .target
                    .as_ref()
                    .and_then(|t| t.interrupt_controllers.first())
                    .map(|ic| ic.enable_asm.as_str())
                    .unwrap_or("ei");
                for line in asm_str.split(';') {
                    let t = line.trim();
                    if !t.is_empty() {
                        writeln!(self.asm, "    {t}").unwrap();
                    }
                }
            }
            IrInst::QueuePush {
                dst,
                queue_addr,
                val,
                elem_ty,
                capacity,
            } => {
                let elem_size = self.program.type_size(elem_ty);
                let hd_off = elem_size * *capacity;
                let tl_off = hd_off + 1;
                let mask = (*capacity - 1) as u8;
                let da_mem = self.fmt_dst_mem(*dst);
                let ok_lbl = self.fresh_label("qpush_ok");
                let done_lbl = self.fresh_label("qpush_done");
                let const_q = self.const_addrs.get(queue_addr).copied();

                if let Some((qaddr, is_u)) = const_q {
                    let hd_imm = self.fmt_addr_imm(qaddr.wrapping_add(hd_off), is_u);
                    let tl_mem = self.fmt_addr_mem(qaddr.wrapping_add(tl_off), is_u);
                    let q_imm = self.fmt_addr_imm(qaddr, is_u);
                    writeln!(self.asm, "    ld a, {tl_mem}").unwrap();
                    writeln!(self.asm, "    ld e, a").unwrap();
                    writeln!(self.asm, "    inc a").unwrap();
                    writeln!(self.asm, "    and 0x{mask:02X}").unwrap();
                    writeln!(self.asm, "    ld hl, {hd_imm}").unwrap();
                    writeln!(self.asm, "    cp (hl)").unwrap();
                    writeln!(self.asm, "    jr nz, {ok_lbl}").unwrap();
                    writeln!(self.asm, "    xor a").unwrap();
                    writeln!(self.asm, "    jr {done_lbl}").unwrap();
                    writeln!(self.asm, "{ok_lbl}:").unwrap();
                    writeln!(self.asm, "    push af").unwrap();
                    if elem_size > 0 {
                        writeln!(self.asm, "    ld l, e").unwrap();
                        writeln!(self.asm, "    ld h, 0").unwrap();
                        if elem_size > 1 {
                            emit_mul_hl_const(elem_size, self.asm);
                        }
                        writeln!(self.asm, "    ld de, {q_imm}").unwrap();
                        writeln!(self.asm, "    add hl, de").unwrap();
                    }
                } else {
                    self.emit_load_ptr_hl(*queue_addr);
                    writeln!(self.asm, "    push hl").unwrap();
                    self.emit_add_hl_const(hd_off);
                    writeln!(self.asm, "    ld d, (hl)").unwrap();
                    writeln!(self.asm, "    inc hl").unwrap();
                    writeln!(self.asm, "    ld a, (hl)").unwrap();
                    writeln!(self.asm, "    ld e, a").unwrap();
                    writeln!(self.asm, "    inc a").unwrap();
                    writeln!(self.asm, "    and 0x{mask:02X}").unwrap();
                    writeln!(self.asm, "    cp d").unwrap();
                    writeln!(self.asm, "    jr nz, {ok_lbl}").unwrap();
                    writeln!(self.asm, "    pop hl").unwrap();
                    writeln!(self.asm, "    xor a").unwrap();
                    writeln!(self.asm, "    jr {done_lbl}").unwrap();
                    writeln!(self.asm, "{ok_lbl}:").unwrap();
                    writeln!(self.asm, "    pop hl").unwrap();
                    writeln!(self.asm, "    push af").unwrap();
                    writeln!(self.asm, "    push hl").unwrap();
                    if elem_size > 0 {
                        writeln!(self.asm, "    ld l, e").unwrap();
                        writeln!(self.asm, "    ld h, 0").unwrap();
                        if elem_size > 1 {
                            emit_mul_hl_const(elem_size, self.asm);
                        }
                        writeln!(self.asm, "    pop de").unwrap();
                        writeln!(self.asm, "    push de").unwrap();
                        writeln!(self.asm, "    add hl, de").unwrap();
                    }
                }

                if elem_size > 0 {
                    if elem_ty.is_scalar() && elem_size == 1 {
                        self.emit_load_a_from(*val);
                        writeln!(self.asm, "    ld (hl), a").unwrap();
                    } else if elem_ty.is_scalar() && elem_size == 2 {
                        self.emit_load_de_from(*val);
                        writeln!(self.asm, "    ld (hl), e").unwrap();
                        writeln!(self.asm, "    inc hl").unwrap();
                        writeln!(self.asm, "    ld (hl), d").unwrap();
                    } else if elem_ty.is_scalar() && elem_size == 4 {
                        if let Some(&c) = self.const_vals.get(val) {
                            let [b0, b1, b2, b3] = (c as u32).to_le_bytes();
                            writeln!(self.asm, "    ld (hl), 0x{b0:02X}").unwrap();
                            writeln!(self.asm, "    inc hl").unwrap();
                            writeln!(self.asm, "    ld (hl), 0x{b1:02X}").unwrap();
                            writeln!(self.asm, "    inc hl").unwrap();
                            writeln!(self.asm, "    ld (hl), 0x{b2:02X}").unwrap();
                            writeln!(self.asm, "    inc hl").unwrap();
                            writeln!(self.asm, "    ld (hl), 0x{b3:02X}").unwrap();
                        } else {
                            let va_imm = self.fmt_ram_imm(self.val_loc(*val));
                            writeln!(self.asm, "    ex de, hl").unwrap();
                            writeln!(self.asm, "    ld hl, {va_imm}").unwrap();
                            writeln!(self.asm, "    ld bc, 0x0004").unwrap();
                            writeln!(self.asm, "    ldir").unwrap();
                            self.val_in_reg
                                .retain(|_, &mut vr| !RegMask::BC.intersects(vr.mask()));
                        }
                    } else {
                        writeln!(self.asm, "    ex de, hl").unwrap();
                        if let Some(&disp) = self.ix_addrs.get(val) {
                            if disp != 0 {
                                writeln!(self.asm, "    push de").unwrap();
                            }
                            writeln!(self.asm, "    push ix").unwrap();
                            writeln!(self.asm, "    pop hl").unwrap();
                            if disp != 0 {
                                self.emit_add_hl_const(disp as i16 as u16);
                                writeln!(self.asm, "    pop de").unwrap();
                            }
                        } else {
                            self.emit_load_hl_from(*val);
                        }
                        writeln!(self.asm, "    ld bc, 0x{elem_size:04X}").unwrap();
                        writeln!(self.asm, "    ldir").unwrap();
                        self.val_in_reg
                            .retain(|_, &mut vr| !RegMask::BC.intersects(vr.mask()));
                    }
                }

                if let Some((qaddr, is_u)) = const_q {
                    let tl_mem = self.fmt_addr_mem(qaddr.wrapping_add(tl_off), is_u);
                    writeln!(self.asm, "    pop af").unwrap();
                    writeln!(self.asm, "    ld {tl_mem}, a").unwrap();
                } else {
                    writeln!(self.asm, "    pop hl").unwrap();
                    self.emit_add_hl_const(tl_off);
                    writeln!(self.asm, "    pop af").unwrap();
                    writeln!(self.asm, "    ld (hl), a").unwrap();
                }
                writeln!(self.asm, "    ld a, 1").unwrap();
                writeln!(self.asm, "{done_lbl}:").unwrap();
                writeln!(self.asm, "    ld {da_mem}, a").unwrap();
            }
            IrInst::QueuePop {
                dst,
                queue_addr,
                out_addr,
                elem_size,
                capacity,
            } => {
                let hd_off = *elem_size * *capacity;
                let tl_off = hd_off + 1;
                let mask = (*capacity - 1) as u8;
                let da_mem = self.fmt_dst_mem(*dst);
                let ok_lbl = self.fresh_label("qpop_ok");
                let done_lbl = self.fresh_label("qpop_done");
                let const_q = self.const_addrs.get(queue_addr).copied();

                if let Some((qaddr, is_u)) = const_q {
                    let hd_mem = self.fmt_addr_mem(qaddr.wrapping_add(hd_off), is_u);
                    let tl_imm = self.fmt_addr_imm(qaddr.wrapping_add(tl_off), is_u);
                    let q_imm = self.fmt_addr_imm(qaddr, is_u);
                    writeln!(self.asm, "    ld hl, {tl_imm}").unwrap();
                    writeln!(self.asm, "    ld a, {hd_mem}").unwrap();
                    writeln!(self.asm, "    cp (hl)").unwrap();
                    writeln!(self.asm, "    jr nz, {ok_lbl}").unwrap();
                    writeln!(self.asm, "    xor a").unwrap();
                    writeln!(self.asm, "    jr {done_lbl}").unwrap();
                    writeln!(self.asm, "{ok_lbl}:").unwrap();
                    writeln!(self.asm, "    ld e, a").unwrap();
                    writeln!(self.asm, "    inc a").unwrap();
                    writeln!(self.asm, "    and 0x{mask:02X}").unwrap();
                    writeln!(self.asm, "    push af").unwrap();
                    if *elem_size > 0 {
                        writeln!(self.asm, "    ld l, e").unwrap();
                        writeln!(self.asm, "    ld h, 0").unwrap();
                        if *elem_size > 1 {
                            emit_mul_hl_const(*elem_size, self.asm);
                        }
                        writeln!(self.asm, "    ld de, {q_imm}").unwrap();
                        writeln!(self.asm, "    add hl, de").unwrap();
                    }
                } else {
                    self.emit_load_ptr_hl(*queue_addr);
                    writeln!(self.asm, "    push hl").unwrap();
                    self.emit_add_hl_const(hd_off);
                    writeln!(self.asm, "    ld a, (hl)").unwrap();
                    writeln!(self.asm, "    inc hl").unwrap();
                    writeln!(self.asm, "    cp (hl)").unwrap();
                    writeln!(self.asm, "    jr nz, {ok_lbl}").unwrap();
                    writeln!(self.asm, "    pop hl").unwrap();
                    writeln!(self.asm, "    xor a").unwrap();
                    writeln!(self.asm, "    jr {done_lbl}").unwrap();
                    writeln!(self.asm, "{ok_lbl}:").unwrap();
                    writeln!(self.asm, "    ld e, a").unwrap();
                    writeln!(self.asm, "    inc a").unwrap();
                    writeln!(self.asm, "    and 0x{mask:02X}").unwrap();
                    writeln!(self.asm, "    pop hl").unwrap();
                    writeln!(self.asm, "    push af").unwrap();
                    writeln!(self.asm, "    push hl").unwrap();
                    if *elem_size > 0 {
                        writeln!(self.asm, "    ld l, e").unwrap();
                        writeln!(self.asm, "    ld h, 0").unwrap();
                        if *elem_size > 1 {
                            emit_mul_hl_const(*elem_size, self.asm);
                        }
                        writeln!(self.asm, "    pop de").unwrap();
                        writeln!(self.asm, "    push de").unwrap();
                        writeln!(self.asm, "    add hl, de").unwrap();
                    }
                }

                match *elem_size {
                    0 => {}
                    1 => {
                        if let Some(&disp) = self.ix_addrs.get(out_addr) {
                            writeln!(self.asm, "    ld a, (hl)").unwrap();
                            writeln!(self.asm, "    ld (ix + {disp}), a").unwrap();
                        } else if let Some(&(oa, o_u)) = self.const_addrs.get(out_addr) {
                            let oa_mem = self.fmt_addr_mem(oa, o_u);
                            writeln!(self.asm, "    ld a, (hl)").unwrap();
                            writeln!(self.asm, "    ld {oa_mem}, a").unwrap();
                        } else {
                            writeln!(self.asm, "    ld d, (hl)").unwrap();
                            self.emit_load_hl_from(*out_addr);
                            writeln!(self.asm, "    ld (hl), d").unwrap();
                        }
                    }
                    2 => {
                        writeln!(self.asm, "    ld e, (hl)").unwrap();
                        writeln!(self.asm, "    inc hl").unwrap();
                        writeln!(self.asm, "    ld d, (hl)").unwrap();
                        if let Some(&disp) = self.ix_addrs.get(out_addr) {
                            writeln!(self.asm, "    ld (ix + {disp}), e").unwrap();
                            writeln!(self.asm, "    ld (ix + {}), d", disp + 1).unwrap();
                        } else if let Some(&(oa, o_u)) = self.const_addrs.get(out_addr) {
                            let oa_mem = self.fmt_addr_mem(oa, o_u);
                            writeln!(self.asm, "    ld {oa_mem}, de").unwrap();
                        } else {
                            self.emit_load_hl_from(*out_addr);
                            writeln!(self.asm, "    ld (hl), e").unwrap();
                            writeln!(self.asm, "    inc hl").unwrap();
                            writeln!(self.asm, "    ld (hl), d").unwrap();
                        }
                    }
                    sz => {
                        if let Some(&disp) = self.ix_addrs.get(out_addr) {
                            if disp == 0 {
                                writeln!(self.asm, "    push ix").unwrap();
                                writeln!(self.asm, "    pop de").unwrap();
                            } else {
                                writeln!(self.asm, "    push hl").unwrap();
                                writeln!(self.asm, "    push ix").unwrap();
                                writeln!(self.asm, "    pop hl").unwrap();
                                self.emit_add_hl_const(disp as i16 as u16);
                                writeln!(self.asm, "    ex de, hl").unwrap();
                                writeln!(self.asm, "    pop hl").unwrap();
                            }
                        } else {
                            self.emit_load_de_from(*out_addr);
                        }
                        writeln!(self.asm, "    ld bc, 0x{sz:04X}").unwrap();
                        writeln!(self.asm, "    ldir").unwrap();
                        self.val_in_reg
                            .retain(|_, &mut vr| !RegMask::BC.intersects(vr.mask()));
                    }
                }

                if let Some((qaddr, is_u)) = const_q {
                    let hd_mem = self.fmt_addr_mem(qaddr.wrapping_add(hd_off), is_u);
                    writeln!(self.asm, "    pop af").unwrap();
                    writeln!(self.asm, "    ld {hd_mem}, a").unwrap();
                } else {
                    writeln!(self.asm, "    pop hl").unwrap();
                    self.emit_add_hl_const(hd_off);
                    writeln!(self.asm, "    pop af").unwrap();
                    writeln!(self.asm, "    ld (hl), a").unwrap();
                }
                writeln!(self.asm, "    ld a, 1").unwrap();
                writeln!(self.asm, "{done_lbl}:").unwrap();
                writeln!(self.asm, "    ld {da_mem}, a").unwrap();
            }
            IrInst::QueueIsEmpty {
                dst,
                queue_addr,
                elem_size,
                capacity,
            } => {
                let hd_off = *elem_size * *capacity;
                let tl_off = hd_off + 1;
                let da_mem = self.fmt_dst_mem(*dst);
                let done_lbl = self.fresh_label("qempty_done");
                if let Some((qaddr, is_u)) = self.const_addrs.get(queue_addr).copied() {
                    let hd_mem = self.fmt_addr_mem(qaddr.wrapping_add(hd_off), is_u);
                    let tl_imm = self.fmt_addr_imm(qaddr.wrapping_add(tl_off), is_u);
                    writeln!(self.asm, "    ld a, {hd_mem}").unwrap();
                    writeln!(self.asm, "    ld hl, {tl_imm}").unwrap();
                } else {
                    self.emit_load_ptr_hl(*queue_addr);
                    self.emit_add_hl_const(hd_off);
                    writeln!(self.asm, "    ld a, (hl)").unwrap();
                    writeln!(self.asm, "    inc hl").unwrap();
                }
                writeln!(self.asm, "    cp (hl)").unwrap();
                writeln!(self.asm, "    ld a, 1").unwrap();
                writeln!(self.asm, "    jr z, {done_lbl}").unwrap();
                writeln!(self.asm, "    xor a").unwrap();
                writeln!(self.asm, "{done_lbl}:").unwrap();
                writeln!(self.asm, "    ld {da_mem}, a").unwrap();
            }
            IrInst::QueueLen {
                dst,
                queue_addr,
                elem_size,
                capacity,
            } => {
                let hd_off = *elem_size * *capacity;
                let tl_off = hd_off + 1;
                let mask = (*capacity - 1) as u8;
                let da_mem = self.fmt_dst_mem(*dst);
                if let Some((qaddr, is_u)) = self.const_addrs.get(queue_addr).copied() {
                    let hd_imm = self.fmt_addr_imm(qaddr.wrapping_add(hd_off), is_u);
                    let tl_mem = self.fmt_addr_mem(qaddr.wrapping_add(tl_off), is_u);
                    writeln!(self.asm, "    ld hl, {hd_imm}").unwrap();
                    writeln!(self.asm, "    ld a, {tl_mem}").unwrap();
                    writeln!(self.asm, "    sub (hl)").unwrap();
                } else {
                    self.emit_load_ptr_hl(*queue_addr);
                    self.emit_add_hl_const(hd_off);
                    writeln!(self.asm, "    ld e, (hl)").unwrap();
                    writeln!(self.asm, "    inc hl").unwrap();
                    writeln!(self.asm, "    ld a, (hl)").unwrap();
                    writeln!(self.asm, "    sub e").unwrap();
                }
                writeln!(self.asm, "    and 0x{mask:02X}").unwrap();
                writeln!(self.asm, "    ld {da_mem}, a").unwrap();
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

fn slice_val_only_uses_ptr(func: &IrFunction, v: ValueId) -> bool {
    let mut has_ptr_use = false;
    for bb in &func.blocks {
        for inst in &bb.insts {
            if inst.uses_val(v) {
                if matches!(inst, IrInst::SlicePtr { slice, .. } if *slice == v) {
                    has_ptr_use = true;
                } else {
                    return false;
                }
            }
        }
        if matches!(
            bb.term,
            Terminator::Return(Some(tv)) | Terminator::Branch { cond: tv, .. } if tv == v
        ) {
            return false;
        }
    }
    has_ptr_use
}

#[allow(clippy::too_many_arguments)]
fn can_defer_aligned256(
    program: &IrProgram,
    func: &IrFunction,
    bb_idx: usize,
    inst_idx: usize,
    dst: ValueId,
    eff_index: ValueId,
    const_vals: &HashMap<ValueId, i64>,
    val_in_reg: &HashMap<ValueId, RegLoc>,
    val_from_local: &HashMap<ValueId, SlotId>,
    ix_val_loads: &HashMap<ValueId, (i8, u16)>,
    ix_local_slot: Option<SlotId>,
    reg_alloc: &FnRegAlloc,
    fn_clobbers: &[RegMask],
    escaped_locals: &[bool],
) -> bool {
    for (bi, b) in func.blocks.iter().enumerate() {
        if bi != bb_idx
            && (b.insts.iter().any(|u| u.uses_val(dst))
                || matches!(
                    b.term,
                    Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } if v == dst
                ))
        {
            return false;
        }
    }
    let bb = &func.blocks[bb_idx];
    if matches!(
        bb.term,
        Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } if v == dst
    ) {
        return false;
    }
    let mut last_use: Option<usize> = None;
    for (j, u) in bb.insts.iter().enumerate().skip(inst_idx + 1) {
        if !u.uses_val(dst) {
            continue;
        }
        let ok = match u {
            IrInst::LoadPtr { addr, ty, .. } => *addr == dst && program.type_size(ty) == 1,
            IrInst::StorePtr { addr, val, ty, .. } => {
                *addr == dst && *val != dst && program.type_size(ty) == 1
            }
            _ => false,
        };
        if !ok {
            return false;
        }
        last_use = Some(j);
    }
    let Some(lu) = last_use else {
        return true;
    };
    if const_vals.contains_key(&eff_index) {
        return true;
    }
    if let Some(&reg) = val_in_reg.get(&eff_index) {
        for inst in &bb.insts[inst_idx + 1..lu] {
            match inst {
                IrInst::StoreLocal { slot, .. }
                    if reg_alloc
                        .slot_reg(*slot)
                        .is_some_and(|r| r.mask().intersects(reg.mask())) =>
                {
                    return false;
                }
                IrInst::Call { func: cid, .. }
                    if fn_clobbers
                        .get(*cid as usize)
                        .copied()
                        .unwrap_or(RegMask::BC)
                        .intersects(reg.mask()) =>
                {
                    return false;
                }
                IrInst::CallIndirect { .. } => return false,
                IrInst::ExternCall { extern_fn, .. }
                    if extern_clobber_mask(&program.externs[*extern_fn as usize])
                        .intersects(reg.mask()) =>
                {
                    return false;
                }
                other if inst_internal_clobber_mask(program, other, const_vals)
                    .intersects(reg.mask()) =>
                {
                    return false;
                }
                _ => {}
            }
        }
        return true;
    }
    if let Some(&slot) = val_from_local.get(&eff_index) {
        let is_escaped = escaped_locals.get(slot as usize).copied().unwrap_or(true);
        for inst in &bb.insts[inst_idx + 1..lu] {
            match inst {
                IrInst::StoreLocal { slot: s, .. } if *s == slot => return false,
                IrInst::StorePtr { .. }
                | IrInst::MemSet { .. }
                | IrInst::MemCopy { .. }
                | IrInst::MemFill { .. }
                | IrInst::PoolAlloc { .. }
                | IrInst::PoolFree { .. }
                | IrInst::QueuePush { .. }
                | IrInst::QueuePop { .. }
                | IrInst::BankEnter { .. }
                | IrInst::BankExit { .. }
                | IrInst::CriticalEnter
                | IrInst::CriticalExit
                | IrInst::Call { .. }
                | IrInst::CallIndirect { .. }
                | IrInst::ExternCall { .. }
                | IrInst::StateDispatch { .. }
                | IrInst::StateTransition { .. }
                | IrInst::InlineAsm { .. }
                    if is_escaped =>
                {
                    return false;
                }
                _ => {}
            }
        }
        return true;
    }
    if ix_val_loads.contains_key(&eff_index) {
        for inst in &bb.insts[inst_idx + 1..lu] {
            match inst {
                IrInst::StoreLocal { slot, .. } if ix_local_slot == Some(*slot) => return false,
                IrInst::StorePtr { .. }
                | IrInst::MemSet { .. }
                | IrInst::MemCopy { .. }
                | IrInst::MemFill { .. }
                | IrInst::PoolAlloc { .. }
                | IrInst::PoolFree { .. }
                | IrInst::QueuePush { .. }
                | IrInst::QueuePop { .. }
                | IrInst::BankEnter { .. }
                | IrInst::BankExit { .. }
                | IrInst::CriticalEnter
                | IrInst::CriticalExit
                | IrInst::Call { .. }
                | IrInst::CallIndirect { .. }
                | IrInst::ExternCall { .. }
                | IrInst::StateDispatch { .. }
                | IrInst::StateTransition { .. }
                | IrInst::InlineAsm { .. } => return false,
                _ => {}
            }
        }
        return true;
    }
    false
}
