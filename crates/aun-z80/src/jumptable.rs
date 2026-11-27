//! LIR dense dispatch-chain compression into Z80 jump tables (`__aun_jtbl_dispatch`).
//!
//! Converts consecutive `cp #imm; jr/jp z, target` dispatch chains of >= 5 dense cases
//! into O(1) bounds-checked jump-table lookups (`sub #min; cp #span; jp nc, default;
//! ld hl, #table; jp __aun_jtbl_dispatch`), verifying register/flag deadness before
//! transforming.

use crate::analysis::z80_sym;
use crate::lir::{AluOp8, Cond, LirInst, LirItem, Op8, Reg16, Reg8, ShiftOp, SymExpr};
use aun_ir::{IrProgram, IrType};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
struct RegState {
    a: bool,
    d: bool,
    e: bool,
    h: bool,
    l: bool,
    z_flags: bool, // Z, S, P/V
    c_flag: bool,  // C
}

impl RegState {
    fn bits(self) -> u8 {
        u8::from(self.a)
            | (u8::from(self.d) << 1)
            | (u8::from(self.e) << 2)
            | (u8::from(self.h) << 3)
            | (u8::from(self.l) << 4)
            | (u8::from(self.z_flags) << 5)
            | (u8::from(self.c_flag) << 6)
    }

    fn all_done(self) -> bool {
        self.a && self.d && self.e && self.h && self.l && self.z_flags && self.c_flag
    }

    fn kill_reg8(&mut self, r: Reg8) {
        match r {
            Reg8::A => self.a = true,
            Reg8::D => self.d = true,
            Reg8::E => self.e = true,
            Reg8::H => self.h = true,
            Reg8::L => self.l = true,
            Reg8::B | Reg8::C => {}
        }
    }

    fn kill_reg16(&mut self, r: Reg16) {
        match r {
            Reg16::DE => {
                self.d = true;
                self.e = true;
            }
            Reg16::HL => {
                self.h = true;
                self.l = true;
            }
            Reg16::AF => {
                self.a = true;
                self.z_flags = true;
                self.c_flag = true;
            }
            Reg16::BC | Reg16::SP | Reg16::IX | Reg16::IY => {}
        }
    }

    fn kill_named(&mut self, reg: &str) {
        match reg.trim().to_ascii_uppercase().as_str() {
            "A" => self.a = true,
            "D" => self.d = true,
            "E" => self.e = true,
            "H" => self.h = true,
            "L" => self.l = true,
            "F" | "FLAGS" => {
                self.z_flags = true;
                self.c_flag = true;
            }
            "AF" => self.kill_reg16(Reg16::AF),
            "DE" => self.kill_reg16(Reg16::DE),
            "HL" => self.kill_reg16(Reg16::HL),
            _ => {}
        }
    }

    fn is_live_named(self, reg: &str) -> bool {
        match reg.trim().to_ascii_uppercase().as_str() {
            "A" => !self.a,
            "D" => !self.d,
            "E" => !self.e,
            "H" => !self.h,
            "L" => !self.l,
            "F" | "FLAGS" => !self.z_flags || !self.c_flag,
            "AF" => self.is_live_reg16(Reg16::AF),
            "DE" => self.is_live_reg16(Reg16::DE),
            "HL" => self.is_live_reg16(Reg16::HL),
            _ => false,
        }
    }

    fn is_live_reg8(self, r: Reg8) -> bool {
        match r {
            Reg8::A => !self.a,
            Reg8::D => !self.d,
            Reg8::E => !self.e,
            Reg8::H => !self.h,
            Reg8::L => !self.l,
            Reg8::B | Reg8::C => false,
        }
    }

    fn is_live_reg16(self, r: Reg16) -> bool {
        match r {
            Reg16::DE => !self.d || !self.e,
            Reg16::HL => !self.h || !self.l,
            Reg16::AF => !self.a || !self.z_flags || !self.c_flag,
            Reg16::BC | Reg16::SP | Reg16::IX | Reg16::IY => false,
        }
    }

    fn reads_cond(self, cond: Cond) -> bool {
        match cond {
            Cond::C | Cond::NC => !self.c_flag,
            Cond::Z | Cond::NZ | Cond::PO | Cond::PE | Cond::P | Cond::M => !self.z_flags,
        }
    }

    fn reads_op8(self, op: &Op8) -> bool {
        match op {
            Op8::Reg(r) => self.is_live_reg8(*r),
            Op8::HlInd => !self.h || !self.l,
            Op8::Imm(_) | Op8::Indexed(_, _) => false,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RetReg {
    Void,
    RegA,
    RegHl,
}

#[derive(Clone)]
enum FnCallSpec {
    /// Internal Aun call: checks argument register liveness and kills all caller-saved regs.
    Internal {
        ra: bool,
        rd: bool,
        re: bool,
        rhl: bool,
    },
    /// `extern "asm"` call: explicit parameter registers read, explicit clobbers + ret killed.
    Extern {
        reads: Vec<String>,
        clobbers: Vec<String>,
    },
}

/// Checks forward across all reachable control-flow paths from `start_idx` whether
/// dispatch-clobbered registers (`A`, `D`, `E`, `H`, `L`, `Z/S/PV`, `C`) are never read before
/// being overwritten. On `default` (`!check_de_hl`), `a_preserved` reflects whether `A` arrives
/// intact (true only when `min_v == 0` with no table holes).
fn check_dead_at(
    items: &[LirItem],
    label_pos: &HashMap<String, usize>,
    fn_specs: &HashMap<String, FnCallSpec>,
    start_idx: usize,
    check_de_hl: bool,
    a_preserved: bool,
    cur_fn_ret: Option<RetReg>,
) -> bool {
    let init_k = if check_de_hl {
        RegState::default()
    } else {
        RegState {
            a: a_preserved,
            d: true,
            e: true,
            h: true,
            l: true,
            z_flags: false,
            c_flag: false,
        }
    };
    let mut visited: HashSet<(usize, u8)> = HashSet::new();
    let mut budget = 4096usize;
    check_dead_dfs(
        items,
        label_pos,
        fn_specs,
        start_idx,
        init_k,
        check_de_hl,
        cur_fn_ret,
        &mut visited,
        &mut budget,
    )
}

#[allow(clippy::too_many_arguments)]
fn check_dead_dfs(
    items: &[LirItem],
    label_pos: &HashMap<String, usize>,
    fn_specs: &HashMap<String, FnCallSpec>,
    mut idx: usize,
    mut k: RegState,
    check_de_hl: bool,
    cur_fn_ret: Option<RetReg>,
    visited: &mut HashSet<(usize, u8)>,
    budget: &mut usize,
) -> bool {
    while idx < items.len() {
        if k.all_done() {
            return true;
        }
        if *budget == 0 {
            return false;
        }
        *budget -= 1;
        if !visited.insert((idx, k.bits())) {
            // Already proved or actively exploring cycle without any prior read.
            return true;
        }

        match &items[idx] {
            LirItem::Blank | LirItem::Comment(_) | LirItem::Label(_) => {
                idx += 1;
                continue;
            }
            LirItem::DataBytes(_) | LirItem::DataWords(_) | LirItem::Org(_) | LirItem::Equ(_, _) => {
                return false;
            }
            LirItem::Inst(inst) => match inst {
                LirInst::Ld8 { dst, src } => {
                    if k.reads_op8(src) {
                        return false;
                    }
                    k.kill_reg8(*dst);
                }
                LirInst::LdAInd16(r) => {
                    if k.is_live_reg16(*r) {
                        return false;
                    }
                    k.a = true;
                }
                LirInst::StInd16A(r) => {
                    if k.is_live_reg16(*r) || !k.a {
                        return false;
                    }
                }
                LirInst::LdMem8 { dst, .. } => {
                    k.kill_reg8(*dst);
                }
                LirInst::StMem8 { src, .. } => {
                    if k.is_live_reg8(*src) {
                        return false;
                    }
                }
                LirInst::LdInd8 { src, .. } => {
                    if k.reads_op8(src) {
                        return false;
                    }
                }
                LirInst::StHlInd8 { src } => {
                    if !k.h || !k.l || k.reads_op8(src) {
                        return false;
                    }
                }
                LirInst::Ld16Imm { dst, .. } | LirInst::Ld16Mem { dst, .. } => {
                    k.kill_reg16(*dst);
                }
                LirInst::St16Mem { src, .. } => {
                    if k.is_live_reg16(*src) {
                        return false;
                    }
                }
                LirInst::LdSp(r) => {
                    if k.is_live_reg16(*r) {
                        return false;
                    }
                }
                LirInst::Alu8 { op, src } => {
                    let self_zero = matches!((op, src), (AluOp8::Xor | AluOp8::Sub, Op8::Reg(Reg8::A)));
                    if !self_zero && (!k.a || k.reads_op8(src)) {
                        return false;
                    }
                    if matches!(op, AluOp8::Adc | AluOp8::Sbc) && !k.c_flag {
                        return false;
                    }
                    k.z_flags = true;
                    k.c_flag = true;
                    if *op != AluOp8::Cp {
                        k.a = true;
                    }
                }
                LirInst::Add16 { dst, src } => {
                    if k.is_live_reg16(*dst) || k.is_live_reg16(*src) {
                        return false;
                    }
                    k.kill_reg16(*dst);
                    // ADD rr, rr modifies Carry only; Z, S, P/V are preserved.
                    k.c_flag = true;
                }
                LirInst::SbcHl { src } | LirInst::AdcHl { src } => {
                    if !k.h || !k.l || !k.c_flag || k.is_live_reg16(*src) {
                        return false;
                    }
                    k.h = true;
                    k.l = true;
                    k.z_flags = true;
                    k.c_flag = true;
                }
                LirInst::Inc8(op) | LirInst::Dec8(op) => {
                    if k.reads_op8(op) {
                        return false;
                    }
                    if let Op8::Reg(r) = op {
                        k.kill_reg8(*r);
                    }
                    // INC/DEC 8-bit modifies Z/S/P/V; Carry is preserved.
                    k.z_flags = true;
                }
                LirInst::Shift8 { op, target } => {
                    if matches!(op, ShiftOp::Rl | ShiftOp::Rr) && !k.c_flag {
                        return false;
                    }
                    if k.reads_op8(target) {
                        return false;
                    }
                    if let Op8::Reg(r) = target {
                        k.kill_reg8(*r);
                    }
                    k.z_flags = true;
                    k.c_flag = true;
                }
                LirInst::Bit { target, .. } => {
                    if k.reads_op8(target) {
                        return false;
                    }
                    // BIT modifies Z/S/P/V; Carry is preserved.
                    k.z_flags = true;
                }
                LirInst::Set { target, .. } | LirInst::Res { target, .. } => {
                    if k.reads_op8(target) {
                        return false;
                    }
                }
                LirInst::Inc16(r) | LirInst::Dec16(r) => {
                    if k.is_live_reg16(*r) {
                        return false;
                    }
                    k.kill_reg16(*r);
                }
                LirInst::ExDeHl => {
                    std::mem::swap(&mut k.d, &mut k.h);
                    std::mem::swap(&mut k.e, &mut k.l);
                }
                LirInst::Neg => {
                    if !k.a {
                        return false;
                    }
                    k.a = true;
                    k.z_flags = true;
                    k.c_flag = true;
                }
                LirInst::Cpl => {
                    if !k.a {
                        return false;
                    }
                    k.a = true;
                }
                LirInst::Rlca | LirInst::Rrca => {
                    if !k.a {
                        return false;
                    }
                    k.a = true;
                    k.c_flag = true;
                }
                LirInst::Rla | LirInst::Rra => {
                    if !k.a || !k.c_flag {
                        return false;
                    }
                    k.a = true;
                    k.c_flag = true;
                }
                LirInst::Scf => {
                    // SCF sets Carry only; Z, S, P/V are preserved.
                    k.c_flag = true;
                }
                LirInst::Ccf => {
                    if !k.c_flag {
                        return false;
                    }
                }
                LirInst::Push(r) => {
                    if k.is_live_reg16(*r) {
                        return false;
                    }
                }
                LirInst::Pop(r) => k.kill_reg16(*r),
                LirInst::Jp { cond: None, target } | LirInst::Jr { cond: None, target } => {
                    if let Some(&tpos) = label_pos.get(target) {
                        idx = tpos + 1;
                        continue;
                    }
                    return false;
                }
                LirInst::Jp {
                    cond: Some(cond),
                    target,
                }
                | LirInst::Jr {
                    cond: Some(cond),
                    target,
                } => {
                    if k.reads_cond(*cond) {
                        return false;
                    }
                    let Some(&tpos) = label_pos.get(target) else {
                        return false;
                    };
                    if !check_dead_dfs(
                        items,
                        label_pos,
                        fn_specs,
                        idx + 1,
                        k,
                        check_de_hl,
                        cur_fn_ret,
                        visited,
                        budget,
                    ) {
                        return false;
                    }
                    return check_dead_dfs(
                        items,
                        label_pos,
                        fn_specs,
                        tpos + 1,
                        k,
                        check_de_hl,
                        cur_fn_ret,
                        visited,
                        budget,
                    );
                }
                LirInst::Call { cond: None, target } => match fn_specs.get(target) {
                    Some(FnCallSpec::Internal { ra, rd, re, rhl }) => {
                        if (*ra && !k.a)
                            || (*rd && !k.d)
                            || (*re && !k.e)
                            || (*rhl && (!k.h || !k.l))
                        {
                            return false;
                        }
                        return true;
                    }
                    Some(FnCallSpec::Extern { reads, clobbers }) => {
                        for r in reads {
                            if k.is_live_named(r) {
                                return false;
                            }
                        }
                        for c in clobbers {
                            k.kill_named(c);
                        }
                    }
                    None => return false,
                },
                LirInst::Ret(None) => match cur_fn_ret {
                    Some(RetReg::Void) => return true,
                    Some(RetReg::RegA) => return k.a,
                    Some(RetReg::RegHl) => return !check_de_hl || (k.h && k.l),
                    None => return if check_de_hl { k.a && k.h && k.l } else { k.a },
                },
                LirInst::Nop | LirInst::Di | LirInst::Ei => {}
                _ => return false,
            },
        }
        idx += 1;
    }
    false
}

#[derive(Clone)]
struct ChainEntry {
    start_idx: usize,
    load_inst: Option<LirInst>,
    imm: u8,
    target: String,
    end_idx: usize, // exclusive
}

fn is_a_load(inst: &LirInst) -> bool {
    matches!(
        inst,
        LirInst::Ld8 {
            dst: Reg8::A,
            src: Op8::Reg(_) | Op8::Indexed(_, _),
        } | LirInst::LdMem8 { dst: Reg8::A, .. }
    )
}

/// Scans `items` for dense `cp #imm; jr/jp z, target` dispatch chains and replaces
/// profitable ones with `__aun_jtbl_dispatch` jump-table lookups.
pub fn compress_jump_tables(program: &IrProgram, items: &mut Vec<LirItem>) -> usize {
    let mut fn_specs: HashMap<String, FnCallSpec> = HashMap::new();
    let mut fn_ret_reg: HashMap<String, RetReg> = HashMap::new();
    for f in &program.functions {
        if f.is_test {
            continue;
        }
        let sym = z80_sym(&f.name);
        let ra = f.param_count >= 1 && program.type_size(&f.locals[0].ty) == 1;
        let rhl = f.param_count >= 1 && program.type_size(&f.locals[0].ty) == 2;
        let p1_size = if f.param_count >= 2 && f.overlay_state.is_none() {
            program.type_size(&f.locals[1].ty)
        } else {
            0
        };
        let rd = p1_size >= 2;
        let re = p1_size >= 1;
        fn_specs.insert(sym.clone(), FnCallSpec::Internal { ra, rd, re, rhl });
        let ret_reg = if f.ret_ty == IrType::Unit {
            RetReg::Void
        } else if program.type_size(&f.ret_ty) == 1 {
            RetReg::RegA
        } else {
            RetReg::RegHl
        };
        fn_ret_reg.insert(sym, ret_reg);
    }
    for ext in &program.externs {
        let sym = z80_sym(&ext.name);
        let reads = ext.params.iter().map(|(_, _, reg)| reg.clone()).collect();
        let mut clobbers = ext
            .clobbers
            .clone()
            .unwrap_or_else(|| vec!["AF".into(), "BC".into(), "DE".into(), "HL".into()]);
        if let Some((_, ret_reg)) = &ext.ret {
            clobbers.push(ret_reg.clone());
        }
        fn_specs.insert(sym, FnCallSpec::Extern { reads, clobbers });
    }

    // Count references to every label (jumps, calls, data tables).
    let mut label_refs: HashMap<String, usize> = HashMap::new();
    let mut label_pos: HashMap<String, usize> = HashMap::new();
    for (idx, it) in items.iter().enumerate() {
        match it {
            LirItem::Label(l) => {
                label_pos.insert(l.clone(), idx);
            }
            LirItem::Inst(
                LirInst::Jp { target, .. }
                | LirInst::Jr { target, .. }
                | LirInst::Djnz(target)
                | LirInst::Call { target, .. },
            ) => {
                *label_refs.entry(target.clone()).or_insert(0) += 1;
            }
            LirItem::DataWords(ws) => {
                for w in ws {
                    if let SymExpr::Sym(s) = w {
                        *label_refs.entry(s.clone()).or_insert(0) += 1;
                    }
                }
            }
            _ => {}
        }
    }

    struct ChosenTable {
        replace_start: usize,
        replace_end: usize, // exclusive
        min_v: u8,
        span: u8,
        table_label: String,
        entries: Vec<String>,
        fallthrough_label: String,
        needs_fallthrough_label: bool,
    }

    let mut replacements: Vec<ChosenTable> = Vec::new();
    let mut table_id = 0usize;
    let mut cur_fn_ret: Option<RetReg> = None;
    let mut i = 0usize;

    while i < items.len() {
        if let LirItem::Label(lbl) = &items[i]
            && let Some(&rr) = fn_ret_reg.get(lbl)
        {
            cur_fn_ret = Some(rr);
        }
        // Parse maximal chain starting at `i` (first entry starts at an instruction, not its BB label).
        let Some(e0) = parse_entry(items, i, None, &label_refs, &fn_ret_reg, true) else {
            i += 1;
            continue;
        };

        // Check consistency for subsequent entries.
        let mut chain = vec![e0];
        let mut cur = chain[0].end_idx;
        let mut expected_load = chain[0].load_inst.clone();
        while let Some(ek) = parse_entry(
            items,
            cur,
            expected_load.as_ref(),
            &label_refs,
            &fn_ret_reg,
            false,
        ) {
            if expected_load.is_none() && ek.load_inst.is_some() {
                let lk = ek.load_inst.as_ref().unwrap();
                if chain.len() == 1 && prev_saved_a_into(items, chain[0].start_idx, lk) {
                    expected_load = ek.load_inst.clone();
                } else {
                    break;
                }
            }
            cur = ek.end_idx;
            chain.push(ek);
        }

        if chain.len() < 5 {
            i = cur.max(i + 1);
            continue;
        }

        // Find the best contiguous chain slice `chain[lo..chain.len()]` (suffix) with dense values!
        let mut best_slice: Option<(isize, usize, u8, u8, Vec<String>)> = None;
        for lo in 0..=(chain.len() - 5) {
            // Verify `chain[lo]` either is `lo == 0` or its leading load matches; if `chain[lo]` expects `A` already loaded, keep `load_inst` behavior:
            let rep_start = if lo == 0 {
                // Keep `e0`'s own `load_inst` if present, so replace starting at `cp`!
                find_cp_idx(items, chain[0].start_idx)
            } else {
                // For lo > 0, `A` is live from previous cp or load at `chain[lo]`; replace starting at `chain[lo].start_idx` (or after its load).
                if chain[lo].load_inst.is_some() {
                    find_cp_idx(items, chain[lo].start_idx)
                } else {
                    chain[lo].start_idx
                }
            };
            let rep_end = chain[chain.len() - 1].end_idx;

            let slice = &chain[lo..];
            let mut val_map: BTreeMap<u8, String> = BTreeMap::new();
            for e in slice {
                val_map.entry(e.imm).or_insert_with(|| e.target.clone());
            }
            if val_map.len() < 5 {
                continue;
            }
            let min_v = *val_map.keys().next().unwrap();
            let max_v = *val_map.keys().next_back().unwrap();
            let span_u16 = (max_v as u16) - (min_v as u16) + 1;
            if span_u16 > 127 {
                continue;
            }
            let span = span_u16 as u8;
            if val_map.len() * 2 < span as usize {
                continue;
            }

            // Check that all case targets have dead A, DE, HL, and flags!
            let all_targets_safe = val_map.values().all(|t| {
                label_pos.get(t).is_some_and(|&tp| {
                    check_dead_at(items, &label_pos, &fn_specs, tp + 1, true, false, cur_fn_ret)
                })
            });
            if !all_targets_safe {
                continue;
            }

            // Default path preserves A only when min_v == 0 with no holes (cp #span; jp nc).
            let a_preserved_default = min_v == 0 && val_map.len() == span as usize;
            if !check_dead_at(
                items,
                &label_pos,
                &fn_specs,
                rep_end,
                false,
                a_preserved_default,
                cur_fn_ret,
            ) {
                continue;
            }

            let before_bytes: usize = items[rep_start..rep_end]
                .iter()
                .filter_map(|it| match it {
                    LirItem::Inst(ins) => Some(ins.byte_len()),
                    _ => None,
                })
                .sum();
            let has_top_bound = (min_v as u16) + (span as u16) < 256;
            let after_bytes = usize::from(min_v > 0) * 2
                + usize::from(has_top_bound) * 2
                + 3 // jp nc/c, default
                + 3 // ld hl, #table
                + 3 // jp __aun_jtbl_dispatch
                + (span as usize) * 2;
            let dispatch_overhead = if replacements.is_empty() && table_id == 0 {
                10
            } else {
                0
            };
            let savings = (before_bytes as isize) - ((after_bytes + dispatch_overhead) as isize);
            if savings < 2 {
                continue;
            }

            if best_slice.as_ref().is_none_or(|b| savings > b.0) {
                let mut entries = Vec::with_capacity(span as usize);
                for offset in 0..span {
                    let v = min_v + offset;
                    entries.push(
                        val_map
                            .get(&v)
                            .cloned()
                            .unwrap_or_else(|| "__AUN_JT_DEFAULT__".to_string()),
                    );
                }
                best_slice = Some((savings, rep_start, min_v, span, entries));
            }
        }

        if let Some((_, rep_start, min_v, span, entries)) = best_slice {
            let rep_end = chain[chain.len() - 1].end_idx;
            let (fallthrough_label, needs_fallthrough_label) = match find_next_label(items, rep_end) {
                Some(l) => (l, false),
                None => (format!("__aun_jt_def_{table_id}"), true),
            };
            let table_label = format!("__aun_jt_{table_id}");
            table_id += 1;

            replacements.push(ChosenTable {
                replace_start: rep_start,
                replace_end: rep_end,
                min_v,
                span,
                table_label,
                entries,
                fallthrough_label,
                needs_fallthrough_label,
            });
        }

        i = cur.max(i + 1);
    }

    if replacements.is_empty() {
        return 0;
    }

    let count = replacements.len();
    let mut new_items: Vec<LirItem> = Vec::with_capacity(items.len());
    let mut tables: Vec<LirItem> = Vec::new();
    let mut cur_pos = 0usize;

    for mut rep in replacements {
        new_items.extend_from_slice(&items[cur_pos..rep.replace_start]);

        // If `fallthrough_label` had placeholder empty/synthetic fix, patch entries now:
        if rep.min_v > 0 {
            new_items.push(LirItem::Inst(LirInst::Alu8 {
                op: AluOp8::Sub,
                src: Op8::Imm(rep.min_v),
            }));
        }
        let has_top_bound = (rep.min_v as u16) + (rep.span as u16) < 256;
        let oob_cond = if has_top_bound {
            new_items.push(LirItem::Inst(LirInst::Alu8 {
                op: AluOp8::Cp,
                src: Op8::Imm(rep.span),
            }));
            Cond::NC
        } else {
            Cond::C
        };

        new_items.push(LirItem::Inst(LirInst::Jp {
            cond: Some(oob_cond),
            target: rep.fallthrough_label.clone(),
        }));
        new_items.push(LirItem::Inst(LirInst::Ld16Imm {
            dst: Reg16::HL,
            val: SymExpr::Sym(rep.table_label.clone()),
        }));
        new_items.push(LirItem::Inst(LirInst::Jp {
            cond: None,
            target: "__aun_jtbl_dispatch".to_string(),
        }));

        if rep.needs_fallthrough_label {
            new_items.push(LirItem::Label(rep.fallthrough_label.clone()));
        }

        // Point any hole entries to the fallthrough label.
        for e in &mut rep.entries {
            if e == "__AUN_JT_DEFAULT__" {
                *e = rep.fallthrough_label.clone();
            }
        }

        tables.push(LirItem::Blank);
        tables.push(LirItem::Label(rep.table_label));
        tables.push(LirItem::DataWords(
            rep.entries.into_iter().map(SymExpr::Sym).collect(),
        ));

        cur_pos = rep.replace_end;
    }
    new_items.extend_from_slice(&items[cur_pos..]);

    // Emit shared 10-byte `__aun_jtbl_dispatch` helper + jump tables at the end of `_CODE`.
    new_items.push(LirItem::Blank);
    new_items.push(LirItem::Label("__aun_jtbl_dispatch".to_string()));
    new_items.push(LirItem::Inst(LirInst::Alu8 {
        op: AluOp8::Add,
        src: Op8::Reg(Reg8::A),
    }));
    new_items.push(LirItem::Inst(LirInst::Ld8 {
        dst: Reg8::E,
        src: Op8::Reg(Reg8::A),
    }));
    new_items.push(LirItem::Inst(LirInst::Ld8 {
        dst: Reg8::D,
        src: Op8::Imm(0),
    }));
    new_items.push(LirItem::Inst(LirInst::Add16 {
        dst: Reg16::HL,
        src: Reg16::DE,
    }));
    new_items.push(LirItem::Inst(LirInst::Ld8 {
        dst: Reg8::E,
        src: Op8::HlInd,
    }));
    new_items.push(LirItem::Inst(LirInst::Inc16(Reg16::HL)));
    new_items.push(LirItem::Inst(LirInst::Ld8 {
        dst: Reg8::D,
        src: Op8::HlInd,
    }));
    new_items.push(LirItem::Inst(LirInst::ExDeHl));
    new_items.push(LirItem::Inst(LirInst::JpInd(Reg16::HL)));
    new_items.extend(tables);

    *items = new_items;
    count
}

fn find_next_label(items: &[LirItem], idx: usize) -> Option<String> {
    let mut k = idx;
    while k < items.len() {
        match &items[k] {
            LirItem::Blank | LirItem::Comment(_) => k += 1,
            LirItem::Label(l) => return Some(l.clone()),
            _ => return None,
        }
    }
    None
}

fn find_cp_idx(items: &[LirItem], mut idx: usize) -> usize {
    while idx < items.len() {
        if matches!(
            items[idx],
            LirItem::Inst(LirInst::Alu8 {
                op: AluOp8::Cp,
                src: Op8::Imm(_),
            })
        ) {
            return idx;
        }
        idx += 1;
    }
    idx
}

fn prev_saved_a_into(items: &[LirItem], cp_start: usize, load_inst: &LirInst) -> bool {
    let mut k = cp_start;
    while k > 0 {
        k -= 1;
        match &items[k] {
            LirItem::Blank | LirItem::Comment(_) => continue,
            LirItem::Inst(LirInst::Ld8 { dst, src: Op8::Reg(Reg8::A) }) => {
                return matches!(load_inst, LirInst::Ld8 { dst: Reg8::A, src: Op8::Reg(r) } if r == dst);
            }
            _ => return false,
        }
    }
    false
}

fn parse_entry(
    items: &[LirItem],
    mut idx: usize,
    expected_load: Option<&LirInst>,
    label_refs: &HashMap<String, usize>,
    fn_ret_reg: &HashMap<String, RetReg>,
    is_first: bool,
) -> Option<ChainEntry> {
    let start_idx = idx;
    if !is_first {
        while idx < items.len() && matches!(items[idx], LirItem::Blank | LirItem::Comment(_)) {
            idx += 1;
        }
        if idx < items.len()
            && let LirItem::Label(lbl) = &items[idx]
        {
            if label_refs.get(lbl).copied().unwrap_or(0) > 0 || fn_ret_reg.contains_key(lbl) {
                return None;
            }
            idx += 1;
        }
    }
    while idx < items.len() && matches!(items[idx], LirItem::Blank | LirItem::Comment(_)) {
        idx += 1;
    }

    let mut load_inst: Option<LirInst> = None;

    if idx < items.len()
        && let LirItem::Inst(inst) = &items[idx]
        && is_a_load(inst)
    {
        if !is_first
            && let Some(exp) = expected_load
            && inst != exp
        {
            return None;
        }
        load_inst = Some(inst.clone());
        idx += 1;
    } else if !is_first && expected_load.is_some() {
        return None;
    }

    let LirItem::Inst(LirInst::Alu8 {
        op: AluOp8::Cp,
        src: Op8::Imm(imm),
    }) = items.get(idx)?
    else {
        return None;
    };
    idx += 1;

    let target = match items.get(idx)? {
        LirItem::Inst(LirInst::Jp { cond: Some(Cond::Z), target }) => target.clone(),
        LirItem::Inst(LirInst::Jr { cond: Some(Cond::Z), target }) => target.clone(),
        _ => return None,
    };
    idx += 1;

    Some(ChainEntry {
        start_idx,
        load_inst,
        imm: *imm,
        target,
        end_idx: idx,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lir::{parse_zilog_to_lir, LirItem};

    fn dummy_fn(name: &str, ret_ty: IrType) -> aun_ir::IrFunction {
        aun_ir::IrFunction {
            id: 0,
            name: name.to_string(),
            is_pub: false,
            param_count: 1,
            locals: vec![aun_ir::LocalSlot {
                id: 0,
                name: "x".to_string(),
                ty: IrType::Int(aun_ir::IntType::U8),
                mutable: false,
            }],
            ret_ty,
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            isr_using: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            owner_state: None,
            mock_target: None,
            blocks: vec![],
            next_val: 0,
        }
    }

    #[test]
    fn compresses_dense_cp_chain_into_jump_table() {
        let asm = "\
dispatch:
    ld c, a
    cp 0xF0
    jp z, case_0
dispatch_bb1:
    ld a, c
    cp 0xF1
    jp z, case_1
dispatch_bb2:
    ld a, c
    cp 0xF2
    jp z, case_2
dispatch_bb3:
    ld a, c
    cp 0xF3
    jp z, case_3
dispatch_bb4:
    ld a, c
    cp 0xF4
    jp z, case_4
dispatch_bb5:
    ld a, c
    cp 0xF5
    jp z, case_5
dispatch_bb6:
    ld a, c
    cp 0xF6
    jp z, case_6
dispatch_bb7:
    ld a, c
    cp 0xF8
    jp z, case_8
default_blk:
    ld a, 0
    ret
case_0:
    ld a, 1
    ret
case_1:
    ld a, 2
    ret
case_2:
    ld a, 3
    ret
case_3:
    ld a, 4
    ret
case_4:
    ld a, 5
    ret
case_5:
    ld a, 6
    ret
case_6:
    ld a, 7
    ret
case_8:
    ld a, 9
    ret
";
        let mut prog = IrProgram::default();
        prog.functions
            .push(dummy_fn("dispatch", IrType::Int(aun_ir::IntType::U8)));
        let mut items = parse_zilog_to_lir(asm);
        let n = compress_jump_tables(&prog, &mut items);
        assert_eq!(n, 1);
        assert!(items.iter().any(|it| matches!(
            it,
            LirItem::Label(l) if l == "__aun_jtbl_dispatch"
        )));
        assert!(items.iter().any(|it| matches!(it, LirItem::DataWords(w) if w.len() == 9)));
    }

    #[test]
    fn skips_sparse_or_externally_targeted_chain() {
        let asm = "\
dispatch:
    cp 0x01
    jp z, c0
dispatch_bb1:
    cp 0x40
    jp z, c1
dispatch_bb2:
    cp 0x80
    jp z, c2
dispatch_bb3:
    cp 0xC0
    jp z, c3
dispatch_bb4:
    cp 0xFF
    jp z, c4
    ld a, 0
    ret
other_caller:
    jp dispatch_bb2
c0:
    ld a, 1
    ret
c1:
    ld a, 2
    ret
c2:
    ld a, 3
    ret
c3:
    ld a, 4
    ret
c4:
    ld a, 5
    ret
";
        let mut prog = IrProgram::default();
        prog.functions
            .push(dummy_fn("dispatch", IrType::Int(aun_ir::IntType::U8)));
        let mut items = parse_zilog_to_lir(asm);
        assert_eq!(compress_jump_tables(&prog, &mut items), 0);
    }

    #[test]
    fn compresses_dispatch_loop_with_ex_de_hl_and_extern_call() {
        let asm = "\
loop_top:
    ld a, (0xC000)
    or a
    jr z, exit_fn
dispatch:
    cp 0xF8
    jp z, c_fb
    cp 0xF9
    jp z, c_fb
    cp 0xFA
    jp z, c_fb
    cp 0xFB
    jp z, c_fb
    cp 0xFC
    jp z, c_fc
    cp 0xFD
    jp z, c_fd
    cp 0xFE
    jp z, c_fe
    cp 0xFF
    jp z, c_ff
exit_fn:
    ret
c_fb:
    ld l, (ix + 1)
    ld h, (ix + 2)
    ld e, (hl)
    inc hl
    ld d, (hl)
    ex de, hl
    ld (ix + 0), l
    jr loop_top
c_fc:
    ld l, (ix + 1)
    ld h, (ix + 2)
    ld a, (hl)
    ld l, a
    ld a, 0x26
    call _ext_write
    jr loop_top
c_fd:
    ld (ix + 1), 0x01
    jr loop_top
c_fe:
    ld hl, 0x1234
    ld de, 0x0001
    add hl, de
    ret
c_ff:
    xor a
    ld de, 0x0002
    ld hl, 0x0003
    ret
";
        let mut prog = IrProgram::default();
        prog.functions.push(dummy_fn("loop_top", IrType::Unit));
        prog.externs.push(aun_ir::IrExternFn {
            name: "_ext_write".to_string(),
            params: vec![
                ("reg".to_string(), IrType::Unit, "A".to_string()),
                ("val".to_string(), IrType::Unit, "L".to_string()),
            ],
            ret: None,
            clobbers: Some(vec![
                "A".to_string(),
                "B".to_string(),
                "D".to_string(),
                "F".to_string(),
            ]),
            requires_di: false,
            requires_banks: vec![],
            fixed_addr: None,
            no_callback: false,
            owner_state: None,
        });
        let mut items = parse_zilog_to_lir(asm);
        let n = compress_jump_tables(&prog, &mut items);
        assert_eq!(n, 1);
    }

    #[test]
    fn skips_live_return_reg_partial_flags_and_hole_with_live_a() {
        let u8t = IrType::Int(aun_ir::IntType::U8);

        // 1. Default returns incoming `A` untouched when `min_v > 0` (`sub` would corrupt A!).
        let default_ret_a = "\
dispatch:
    cp 0x10
    jp z, c0
    cp 0x11
    jp z, c1
    cp 0x12
    jp z, c2
    cp 0x13
    jp z, c3
    cp 0x14
    jp z, c4
    cp 0x15
    jp z, c5
    cp 0x16
    jp z, c6
    cp 0x17
    jp z, c7
    ret
c0: ld a, 1
    ret
c1: ld a, 2
    ret
c2: ld a, 3
    ret
c3: ld a, 4
    ret
c4: ld a, 5
    ret
c5: ld a, 6
    ret
c6: ld a, 7
    ret
c7: ld a, 8
    ret
";
        let mut p1 = IrProgram::default();
        p1.functions.push(dummy_fn("dispatch", u8t.clone()));
        assert_eq!(compress_jump_tables(&p1, &mut parse_zilog_to_lir(default_ret_a)), 0);

        // 2. Partial flags preserved by `bit` (C) or `add hl, de` / `scf` (Z) must block jump table!
        let partial_flags = "\
dispatch:
    cp 0x10
    jp z, c0
    cp 0x11
    jp z, c1
    cp 0x12
    jp z, c1
    cp 0x13
    jp z, c1
    cp 0x14
    jp z, c1
    cp 0x15
    jp z, c1
    cp 0x16
    jp z, c1
    cp 0x17
    jp z, c1
    ret
c0:
    bit 0, b
    jr c, c1
    ret
c1:
    ret
";
        let mut p2 = IrProgram::default();
        p2.functions.push(dummy_fn("dispatch", IrType::Unit));
        assert_eq!(compress_jump_tables(&p2, &mut parse_zilog_to_lir(partial_flags)), 0);

        // 3. `min_v == 0` WITH a hole (`0x02` missing) where default returns `A`:
        // holes enter __aun_jtbl_dispatch and double A -> must skip!
        let min0_holes = "\
dispatch:
    cp 0x00
    jp z, c0
    cp 0x01
    jp z, c1
    cp 0x03
    jp z, c3
    cp 0x04
    jp z, c4
    cp 0x05
    jp z, c5
    cp 0x06
    jp z, c6
    cp 0x07
    jp z, c7
    ret
c0: ld a, 10
    ret
c1: ld a, 11
    ret
c3: ld a, 13
    ret
c4: ld a, 14
    ret
c5: ld a, 15
    ret
c6: ld a, 16
    ret
c7: ld a, 17
    ret
";
        let mut p3 = IrProgram::default();
        p3.functions.push(dummy_fn("dispatch", u8t));
        assert_eq!(compress_jump_tables(&p3, &mut parse_zilog_to_lir(min0_holes)), 0);
    }
}
