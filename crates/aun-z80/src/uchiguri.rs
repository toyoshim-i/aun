//! *Uchiguri* v1: Call-graph static RAM overlay allocator and stack depth verifier.

use aun_ir::*;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct GlobalRamEntry {
    pub global_id: u32,
    pub name: String,
    pub addr: u16,
    pub size: u16,
}

#[derive(Debug, Clone)]
pub struct FnRamFrame {
    pub func_id: u32,
    pub name: String,
    pub locals_base: u16,
    pub locals_size: u16,
    pub scratch_base: u16,
    pub scratch_size: u16,
    pub stack_depth: u16,
}

#[derive(Debug, Clone)]
pub struct RamLayout {
    pub ram_base: u16,
    pub overlay_base: u16,
    pub ram_end: u16,
    pub global_addrs: HashMap<u32, u16>,
    /// `(func_id, slot_id) -> fixed RAM address`
    pub slot_addrs: HashMap<(u32, SlotId), u16>,
    /// `func_id -> base address of per-function SSA temporary scratch area`
    pub fn_scratch_bases: HashMap<u32, u16>,
    /// `func_id -> (sb_lo, sb_hi)` sub-range of scratch strictly dead across basic blocks
    pub fn_sb_scratch_ranges: HashMap<u32, (u16, u16)>,
    /// `func_id -> byte offset from scratch_base for each SSA ValueId (0..next_val)`
    pub fn_val_offsets: HashMap<u32, Vec<u16>>,
    pub global_entries: Vec<GlobalRamEntry>,
    pub fn_frames: Vec<FnRamFrame>,
    pub max_stack_bytes: u16,
}

/// Allocates fixed RAM addresses for banked globals, unbanked globals, and non-overlapping
/// call-tree function locals and SSA temporary scratch slots.
pub fn allocate_uchiguri(program: &IrProgram, requested_ram_base: u16) -> Result<RamLayout, String> {
    let analysis = crate::analysis::ProgramAnalysis::analyze(program);
    allocate_uchiguri_with_analysis(program, &analysis, requested_ram_base)
}

/// Allocates Uchiguri RAM using precomputed `ProgramAnalysis`.
pub fn allocate_uchiguri_with_analysis(
    program: &IrProgram,
    analysis: &crate::analysis::ProgramAnalysis,
    requested_ram_base: u16,
) -> Result<RamLayout, String> {
    // If `requested_ram_base` falls inside a switchable `bank_group` window (e.g. PC-88 HighWindow
    // at 0xC000..0xFFFF), automatically pick an unbanked RAM base inside the primary `space`
    // strictly above any `overlay_slot` and below `stack` so switching banks never unmaps locals!
    let mut ram_base = requested_ram_base;
    if let Some(target) = &program.target {
        let in_bank_window = target
            .bank_groups
            .iter()
            .any(|bg| ram_base >= bg.start && ram_base <= bg.end);
        if in_bank_window {
            let min_above_overlays = target
                .overlay_slots
                .iter()
                .map(|s| s.end.saturating_add(1))
                .max()
                .unwrap_or(0x8000);
            let space_start = target.spaces.first().map(|s| s.start).unwrap_or(0x0100);
            ram_base = align_up(min_above_overlays.max(space_start.saturating_add(0x2000)), 256);
        }
    }

    let mut cursor = ram_base;
    let mut bank_cursors: HashMap<String, (u16, u16)> = HashMap::new();
    if let Some(target) = &program.target {
        for bg in &target.bank_groups {
            for b in &bg.banks {
                bank_cursors.insert(b.name.clone(), (b.start, b.end));
            }
        }
    }

    let mut global_addrs = HashMap::new();
    let mut global_entries = Vec::new();

    // 1. Place persistent globals: ROM constants take 0 RAM; banked globals go inside their bank;
    //    unbanked RAM globals are packed sequentially from `ram_base`.
    for (gid, g) in program.globals.iter().enumerate() {
        if g.in_rom {
            continue;
        }
        let align = g.align.max(1);
        let size = program.type_size(&g.ty).max(1);
        let addr = if let Some(ref space_name) = g.space
            && let Some((bcur, bend)) = bank_cursors.get_mut(space_name)
        {
            let placed = align_up(*bcur, align);
            let end = placed.saturating_add(size).saturating_sub(1);
            if end > *bend {
                return Err(format!(
                    "bank `{space_name}` overflow: global `{}` ({size}B) exceeds 0x{bend:04X}",
                    g.name
                ));
            }
            *bcur = placed.saturating_add(size);
            placed
        } else {
            cursor = align_up(cursor, align);
            let placed = cursor;
            cursor = cursor.wrapping_add(size);
            placed
        };
        global_addrs.insert(gid as u32, addr);
        global_entries.push(GlobalRamEntry {
            global_id: gid as u32,
            name: g.name.clone(),
            addr,
            size,
        });
    }

    let overlay_base = cursor;

    // 2. Build static call graph over internal functions and detect extern calls
    let n = program.functions.len();
    let mut callees: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut calls_extern = vec![false; n];
    for (i, f) in program.functions.iter().enumerate() {
        let mut seen = HashSet::new();
        for bb in &f.blocks {
            for inst in &bb.insts {
                match inst {
                    IrInst::Call { func, .. } => {
                        let c = *func as usize;
                        if seen.insert(c) {
                            callees[i].push(c);
                        }
                    }
                    IrInst::StateTransition {
                        machine_id,
                        from_state,
                        to_state,
                        ..
                    } => {
                        let mdef = &program.machines[*machine_id as usize];
                        if let Some(enter_fn) = mdef.states[*to_state as usize].on_enter {
                            let c = enter_fn as usize;
                            if seen.insert(c) {
                                callees[i].push(c);
                            }
                        }
                        if let Some(from) = from_state {
                            if let Some(exit_fn) = mdef.states[*from as usize].on_exit {
                                let c = exit_fn as usize;
                                if seen.insert(c) {
                                    callees[i].push(c);
                                }
                            }
                        } else {
                            calls_extern[i] = true; // dynamic check may call __aun_panic / disp_on_exit
                            for st in &mdef.states {
                                if let Some(exit_fn) = st.on_exit {
                                    let c = exit_fn as usize;
                                    if seen.insert(c) {
                                        callees[i].push(c);
                                    }
                                }
                            }
                        }
                    }
                    IrInst::StateDispatch {
                        machine_id, method, ..
                    } => {
                        calls_extern[i] = true; // dispatch stub return address
                        let mdef = &program.machines[*machine_id as usize];
                        for st in &mdef.states {
                            if let Some(&fid) = st.methods.get(method) {
                                let c = fid as usize;
                                if seen.insert(c) {
                                    callees[i].push(c);
                                }
                            }
                        }
                    }
                    IrInst::ExternCall { .. }
                    | IrInst::Assert { .. }
                    | IrInst::Panic { .. }
                    | IrInst::IndexAddr { .. }
                    | IrInst::PoolAlloc { .. }
                    | IrInst::PoolFree { .. }
                    | IrInst::PoolElemAddr { .. } => {
                        calls_extern[i] = true;
                    }
                    _ => {}
                }
            }
        }
    }

    // 3. Compute frame sizes per function (locals + liveness-colored SSA temporary scratch)
    let mut locals_sizes = vec![0u16; n];
    let mut scratch_sizes = vec![0u16; n];
    let mut dedicated_scratch_bytes = vec![0u16; n];
    let mut frame_sizes = vec![0u16; n];
    let mut local_rel_offsets: Vec<Vec<u16>> = vec![Vec::new(); n];
    let mut fn_val_offsets: HashMap<u32, Vec<u16>> = HashMap::new();
    for (i, f) in program.functions.iter().enumerate() {
        let fa = &analysis.fns[i];
        let (rel_offs, loc_sz) = color_fn_locals(program, f, fa);
        local_rel_offsets[i] = rel_offs;
        locals_sizes[i] = loc_sz;
        let (val_offs, scratch, ded_bytes) = color_fn_scratch(program, f, fa);
        fn_val_offsets.insert(f.id, val_offs);
        scratch_sizes[i] = scratch;
        dedicated_scratch_bytes[i] = ded_bytes;
        frame_sizes[i] = loc_sz.saturating_add(scratch);
    }

    // 4. Bottom-up call-tree overlay offset & stack depth (with recursion check)
    let mut state = vec![0u8; n]; // 0 = unvisited, 1 = visiting, 2 = done
    let mut fn_overlay_offset = vec![0u16; n];
    let mut fn_stack_depth = vec![0u16; n];

    #[allow(clippy::too_many_arguments)]
    fn dfs(
        u: usize,
        program: &IrProgram,
        analysis: &crate::analysis::ProgramAnalysis,
        callees: &[Vec<usize>],
        calls_extern: &[bool],
        frame_sizes: &[u16],
        state: &mut [u8],
        fn_overlay_offset: &mut [u16],
        fn_stack_depth: &mut [u16],
    ) -> Result<(), String> {
        if state[u] == 2 {
            return Ok(());
        }
        if state[u] == 1 {
            if !program.functions[u].is_reentrant {
                return Err(format!(
                    "recursion detected in non-#[reentrant] function `{}`",
                    program.functions[u].name
                ));
            }
            return Ok(());
        }
        state[u] = 1;
        let mut max_child_end = 0u16;
        let mut max_child_stack = if calls_extern[u] { 2u16 } else { 0u16 };
        // Account for callee-saved `push ix` (+2B) and caller-saved `push bc` (+2B for C-reg local)
        let ix_stack = if program.functions[u]
            .locals
            .iter()
            .take(program.functions[u].param_count.min(2))
            .any(|l| matches!(l.ty, IrType::Ref { .. }))
        {
            2u16
        } else {
            0u16
        };
        let bc_stack = if (!callees[u].is_empty() || calls_extern[u])
            && analysis.fns[u].reg_alloc.uses_bc()
        {
            2u16
        } else {
            0u16
        };
        for &v in &callees[u] {
            dfs(
                v,
                program,
                analysis,
                callees,
                calls_extern,
                frame_sizes,
                state,
                fn_overlay_offset,
                fn_stack_depth,
            )?;
            let child_end = fn_overlay_offset[v].saturating_add(frame_sizes[v]);
            max_child_end = max_child_end.max(child_end);
            max_child_stack = max_child_stack.max(fn_stack_depth[v] + 2);
        }
        fn_overlay_offset[u] = max_child_end;
        fn_stack_depth[u] = max_child_stack.saturating_add(ix_stack).saturating_add(bc_stack);
        state[u] = 2;
        Ok(())
    }

    for i in 0..n {
        if !program.functions[i].is_test {
            dfs(
                i,
                program,
                analysis,
                &callees,
                &calls_extern,
                &frame_sizes,
                &mut state,
                &mut fn_overlay_offset,
                &mut fn_stack_depth,
            )?;
        }
    }

    // 5. Partition ISR-reachable functions from main-thread functions (100% ISR isolation)
    //    and reject shared non-reentrant callees reachable from BOTH main and ISR!
    let mut isr_reachable = vec![false; n];
    let mut main_reachable = vec![false; n];
    fn mark_reach(u: usize, callees: &[Vec<usize>], flags: &mut [bool]) {
        if flags[u] {
            return;
        }
        flags[u] = true;
        for &v in &callees[u] {
            mark_reach(v, callees, flags);
        }
    }
    for i in 0..n {
        if program.functions[i].is_test {
            continue;
        }
        if program.functions[i].interrupt_source.is_some() {
            mark_reach(i, &callees, &mut isr_reachable);
        } else if program.functions[i].name == "main" {
            mark_reach(i, &callees, &mut main_reachable);
        }
    }
    for i in 0..n {
        if main_reachable[i] && isr_reachable[i] && !program.functions[i].is_reentrant {
            return Err(format!(
                "re-entrancy hazard: non-#[reentrant] function `{}` is reachable from both `main` and an `#[interrupt]` ISR",
                program.functions[i].name
            ));
        }
    }

    // 6. Assign final concrete addresses: main-thread overlay first, then isolated ISR region
    let mut slot_addrs = HashMap::new();
    let mut fn_scratch_bases = HashMap::new();
    let mut fn_sb_scratch_ranges = HashMap::new();
    let mut fn_frames = Vec::new();
    let mut main_end = overlay_base;
    let mut max_main_stack = 0u16;

    for i in 0..n {
        if program.functions[i].is_test || isr_reachable[i] {
            continue;
        }
        let func = &program.functions[i];
        let fn_base = overlay_base.wrapping_add(fn_overlay_offset[i]);
        for (slot_idx, &rel) in local_rel_offsets[i].iter().enumerate() {
            slot_addrs.insert((func.id, slot_idx as SlotId), fn_base.wrapping_add(rel));
        }
        let scratch_base = fn_base.wrapping_add(locals_sizes[i]);
        fn_scratch_bases.insert(func.id, scratch_base);
        let sb_lo = scratch_base.wrapping_add(dedicated_scratch_bytes[i]);
        let sb_hi = scratch_base.wrapping_add(scratch_sizes[i]);
        if sb_hi > sb_lo {
            fn_sb_scratch_ranges.insert(func.id, (sb_lo, sb_hi));
        }
        let fn_end = fn_base.wrapping_add(frame_sizes[i]);
        main_end = main_end.max(fn_end);
        let total_stack = fn_stack_depth[i] + 2;
        max_main_stack = max_main_stack.max(total_stack);
        fn_frames.push(FnRamFrame {
            func_id: func.id,
            name: func.name.clone(),
            locals_base: fn_base,
            locals_size: locals_sizes[i],
            scratch_base,
            scratch_size: scratch_sizes[i],
            stack_depth: total_stack,
        });
    }

    // Place ISR-reachable functions strictly above `main_end` (zero overlap with main thread)
    let isr_base = main_end;
    let mut max_end = main_end;
    let mut max_isr_stack = 0u16;
    for i in 0..n {
        if program.functions[i].is_test || !isr_reachable[i] {
            continue;
        }
        let func = &program.functions[i];
        let fn_base = isr_base.wrapping_add(fn_overlay_offset[i]);
        for (slot_idx, &rel) in local_rel_offsets[i].iter().enumerate() {
            slot_addrs.insert((func.id, slot_idx as SlotId), fn_base.wrapping_add(rel));
        }
        let scratch_base = fn_base.wrapping_add(locals_sizes[i]);
        fn_scratch_bases.insert(func.id, scratch_base);
        let sb_lo = scratch_base.wrapping_add(dedicated_scratch_bytes[i]);
        let sb_hi = scratch_base.wrapping_add(scratch_sizes[i]);
        if sb_hi > sb_lo {
            fn_sb_scratch_ranges.insert(func.id, (sb_lo, sb_hi));
        }
        let fn_end = fn_base.wrapping_add(frame_sizes[i]);
        max_end = max_end.max(fn_end);
        // Hardware INT return addr (2B) + PUSH AF/BC/DE/HL/IX (10B) = 12B for ISR entry
        let entry_overhead = if func.interrupt_source.is_some() { 12 } else { 2 };
        let total_stack = fn_stack_depth[i] + entry_overhead;
        max_isr_stack = max_isr_stack.max(total_stack);
        fn_frames.push(FnRamFrame {
            func_id: func.id,
            name: func.name.clone(),
            locals_base: fn_base,
            locals_size: locals_sizes[i],
            scratch_base,
            scratch_size: scratch_sizes[i],
            stack_depth: total_stack,
        });
    }

    let max_stack = max_main_stack.saturating_add(max_isr_stack);

    // Verify unbanked Uchiguri RAM does not collide with any declared `stack` or `bank_group`
    if let Some(target) = &program.target
        && max_end > ram_base
    {
        let ram_last = max_end.saturating_sub(1);
        for st in &target.stacks {
            if ram_base <= st.end && st.start <= ram_last {
                return Err(format!(
                    "Uchiguri RAM (0x{ram_base:04X}..0x{max_end:04X}) overlaps stack `{}` (0x{:04X}..0x{:04X})",
                    st.name, st.start, st.end
                ));
            }
        }
        for bg in &target.bank_groups {
            if ram_base <= bg.end && bg.start <= ram_last {
                return Err(format!(
                    "Uchiguri RAM (0x{ram_base:04X}..0x{max_end:04X}) overlaps switchable bank_group `{}` (0x{:04X}..0x{:04X})",
                    bg.name, bg.start, bg.end
                ));
            }
        }
    }

    Ok(RamLayout {
        ram_base,
        overlay_base,
        ram_end: max_end,
        global_addrs,
        slot_addrs,
        fn_scratch_bases,
        fn_sb_scratch_ranges,
        fn_val_offsets,
        global_entries,
        fn_frames,
        max_stack_bytes: max_stack,
    })
}

/// Assigns compact relative byte offsets to `func.locals`:
/// - Parameters (`0..param_count`) and any address-taken (`AddrLocal`) locals get dedicated slots.
/// - Non-address-taken scalar/local slots are interval-colored across flattened instruction
///   positions (conservatively extended over enclosing loop back-edges) so disjoint `if`/`else`
///   and sequential scopes share the exact same Uchiguri RAM bytes.
fn color_fn_locals(
    program: &IrProgram,
    func: &IrFunction,
    fa: &crate::analysis::FnAnalysis,
) -> (Vec<u16>, u16) {
    let n_slots = func.locals.len();
    if n_slots == 0 {
        return (Vec::new(), 0);
    }
    // Synthetic empty-body functions (e.g. unit tests) keep sequential layout
    if func.blocks.iter().all(|b| b.insts.is_empty()) {
        let mut offs = Vec::with_capacity(n_slots);
        let mut cur = 0u16;
        for local in &func.locals {
            offs.push(cur);
            cur = cur.wrapping_add(program.type_size(&local.ty).max(1));
        }
        return (offs, cur);
    }

    // A local slot only escapes if its `AddrLocal` value is used by anything other than
    // direct `LoadPtr(addr)` or `StorePtr(addr, val != addr)`.
    let mut addr_escaped = vec![false; n_slots];
    let mut addr_local_slot: HashMap<ValueId, usize> = HashMap::new();
    let mut min_pos: Vec<Option<usize>> = vec![None; n_slots];
    let mut max_pos: Vec<Option<usize>> = vec![None; n_slots];

    // Map BlockId -> (start_pos, end_pos) in flattened instruction space
    let mut bb_pos: HashMap<BlockId, (usize, usize)> = HashMap::new();
    let mut cur_pos = 0usize;
    for (bi, bb) in func.blocks.iter().enumerate() {
        let start = cur_pos;
        for (ii, inst) in bb.insts.iter().enumerate() {
            match inst {
                IrInst::AddrLocal { dst, slot } => {
                    let s = *slot as usize;
                    if s < n_slots {
                        addr_local_slot.insert(*dst, s);
                        min_pos[s] = Some(min_pos[s].map_or(cur_pos, |m| m.min(cur_pos)));
                        max_pos[s] = Some(max_pos[s].map_or(cur_pos, |m| m.max(cur_pos)));
                        let escapes = func.blocks.iter().any(|b| {
                            b.insts.iter().any(|u| {
                                u.uses_val(*dst)
                                    && !matches!(u, IrInst::LoadPtr { addr, .. } if *addr == *dst)
                                    && !matches!(u, IrInst::StorePtr { addr, val, .. } if *addr == *dst && *val != *dst)
                            }) || matches!(
                                b.term,
                                Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } if v == *dst
                            )
                        });
                        if escapes {
                            addr_escaped[s] = true;
                        }
                    }
                }
                IrInst::LoadPtr { addr, .. } | IrInst::StorePtr { addr, .. } => {
                    if let Some(&s) = addr_local_slot.get(addr) {
                        min_pos[s] = Some(min_pos[s].map_or(cur_pos, |m| m.min(cur_pos)));
                        max_pos[s] = Some(max_pos[s].map_or(cur_pos, |m| m.max(cur_pos)));
                    }
                }
                IrInst::LoadLocal { dst, slot, .. } => {
                    let s = *slot as usize;
                    if s < n_slots {
                        min_pos[s] = Some(min_pos[s].map_or(cur_pos, |m| m.min(cur_pos)));
                        max_pos[s] = Some(max_pos[s].map_or(cur_pos, |m| m.max(cur_pos)));
                        if crate::analysis::can_alias_local_load(func, bi, ii, *dst, *slot) {
                            for (j, uinst) in bb.insts.iter().enumerate().skip(ii + 1) {
                                if uinst.uses_val(*dst) {
                                    let upos = start + j;
                                    max_pos[s] = Some(max_pos[s].map_or(upos, |m| m.max(upos)));
                                }
                            }
                            if matches!(
                                bb.term,
                                Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. }
                                    if v == *dst
                            ) {
                                let tpos = start + bb.insts.len();
                                max_pos[s] = Some(max_pos[s].map_or(tpos, |m| m.max(tpos)));
                            }
                        }
                    }
                }
                IrInst::StoreLocal { slot, .. } => {
                    let s = *slot as usize;
                    if s < n_slots {
                        min_pos[s] = Some(min_pos[s].map_or(cur_pos, |m| m.min(cur_pos)));
                        max_pos[s] = Some(max_pos[s].map_or(cur_pos, |m| m.max(cur_pos)));
                    }
                }
                _ => {}
            }
            cur_pos += 1;
        }
        bb_pos.insert(bb.id, (start, cur_pos));
        cur_pos += 1; // terminator slot
    }

    let ix_param_slot = fa.ix_param_slot;
    let reg_p1 = fa.uses_reg_param1;
    // Register-passed params (0 and optionally 1) are live from prologue entry (`pos = 0`);
    // if non-escaping and not already in IX/B/C/BC, color them so later locals can reuse their RAM!
    for s in 0..func.param_count.min(2) {
        let is_reg_param = s == 0 || (s == 1 && reg_p1);
        if is_reg_param
            && !addr_escaped[s]
            && ix_param_slot != Some(s as SlotId)
            && fa.reg_alloc.slot_reg(s as SlotId).is_none()
        {
            min_pos[s] = Some(0);
            max_pos[s] = Some(max_pos[s].unwrap_or(0));
        }
    }

    // Collect loop intervals from any backward edge (target_start <= src_end)
    let mut loops: Vec<(usize, usize)> = Vec::new();
    for bb in &func.blocks {
        let (_, src_end) = bb_pos[&bb.id];
        let mut check_edge = |target: BlockId| {
            if let Some(&(t_start, _)) = bb_pos.get(&target)
                && t_start <= src_end
            {
                loops.push((t_start, src_end));
            }
        };
        match bb.term {
            Terminator::Goto(t) => check_edge(t),
            Terminator::Branch {
                then_bb, else_bb, ..
            } => {
                check_edge(then_bb);
                check_edge(else_bb);
            }
            _ => {}
        }
    }

    // Extend any slot's live range that crosses a loop boundary (either defined before the loop
    // and used inside/after it, OR defined inside the loop and read after it).
    for s in 0..n_slots {
        if let (Some(mut lo), Some(mut hi)) = (min_pos[s], max_pos[s]) {
            let mut changed = true;
            while changed {
                changed = false;
                for &(l_lo, l_hi) in &loops {
                    let crosses_in = lo < l_lo && hi >= l_lo && hi < l_hi;
                    let crosses_out = lo > l_lo && lo <= l_hi && hi > l_hi;
                    if crosses_in {
                        hi = l_hi;
                        changed = true;
                    }
                    if crosses_out {
                        lo = l_lo;
                        changed = true;
                    }
                }
            }
            min_pos[s] = Some(lo);
            max_pos[s] = Some(hi);
        }
    }

    let mut offsets = vec![0u16; n_slots];
    let mut dedicated_end = 0u16;
    let mut colorable: Vec<(usize, usize, u16, usize)> = Vec::new(); // (lo, hi, size, slot_idx)

    for (s, local) in func.locals.iter().enumerate() {
        let sz = program.type_size(&local.ty).max(1);
        if ix_param_slot == Some(s as SlotId) || fa.reg_alloc.slot_reg(s as SlotId).is_some() {
            // Kept in register (`IX`/`B`/`C`/`BC`) across its live range; never stored in RAM.
            continue;
        }
        let is_reg_param = s < func.param_count && (s == 0 || (s == 1 && reg_p1));
        if (s < func.param_count && !is_reg_param) || addr_escaped[s] {
            offsets[s] = dedicated_end;
            dedicated_end = dedicated_end.wrapping_add(sz);
        } else if let (Some(lo), Some(hi)) = (min_pos[s], max_pos[s]) {
            colorable.push((lo, hi, sz, s));
        }
    }

    // First-fit byte-range coloring for non-overlapping local live ranges
    colorable.sort_by_key(|&(lo, hi, _, s)| (lo, hi, s));
    let mut placed: Vec<(u16, u16, usize, usize)> = Vec::new(); // (off, end_off, lo, hi)
    let mut total_end = dedicated_end;

    for (lo, hi, sz, s) in colorable {
        let mut cand = dedicated_end;
        loop {
            let cand_end = cand.wrapping_add(sz);
            let conflict = placed
                .iter()
                .find(|&&(po, pe, plo, phi)| {
                    // Time intervals overlap AND byte ranges overlap
                    lo <= phi && hi >= plo && cand < pe && cand_end > po
                })
                .copied();
            if let Some((_, pe, _, _)) = conflict {
                cand = pe;
            } else {
                offsets[s] = cand;
                placed.push((cand, cand_end, lo, hi));
                total_end = total_end.max(cand_end);
                break;
            }
        }
    }

    (offsets, total_end)
}

/// Assigns compact byte-packed scratch offsets to SSA `ValueId`s in `func`:
/// - Always-folded constants/addresses and aliased `LoadLocal`s need 0 scratch bytes.
/// - Any `ValueId` used in a different basic block than where it was defined gets a dedicated slot.
/// - Single-block `ValueId`s are byte-range interval-colored per block and share a tiny scratch pool.
/// Returns `(val_offsets, total_scratch_bytes, dedicated_bytes)`.
fn color_fn_scratch(
    program: &IrProgram,
    func: &IrFunction,
    fa: &crate::analysis::FnAnalysis,
) -> (Vec<u16>, u16, u16) {
    let nv = func.next_val as usize;
    if nv == 0 {
        return (Vec::new(), 0, 0);
    }
    let val_sizes = &fa.val_sizes;
    let fused_cmp_dsts = &fa.fused_cmp_dsts;
    let empty_skip: HashSet<(usize, usize)> = HashSet::new();
    let mut def_loc: Vec<Option<(usize, usize)>> = vec![None; nv];
    let mut no_scratch: Vec<bool> = vec![false; nv];
    let mut last_use_same_bb: Vec<Option<usize>> = vec![None; nv];
    let mut cross_block: Vec<bool> = vec![false; nv];

    let ix_param_slot = fa.ix_param_slot;
    let mut ix_vals: HashMap<ValueId, i8> = HashMap::new();
    let mut ix_param_addr_vals: HashSet<ValueId> = HashSet::new();
    for (bi, bb) in func.blocks.iter().enumerate() {
        let mut ix_loaded_vals: HashSet<ValueId> = HashSet::new();
        for (ii, inst) in bb.insts.iter().enumerate() {
            if let Some(d) = inst.dst()
                && (d as usize) < nv
            {
                def_loc[d as usize] = Some((bi, ii));
                let folded = match inst {
                    IrInst::ConstInt { .. } | IrInst::ConstBool { .. } => {
                        crate::analysis::is_const_always_folded(
                            program,
                            func,
                            d,
                            fused_cmp_dsts,
                            &empty_skip,
                        )
                    }
                    IrInst::BinOp { .. } => fused_cmp_dsts.contains(&d),
                    IrInst::AddrLocal { slot, .. } => {
                        if ix_param_slot == Some(*slot) {
                            ix_param_addr_vals.insert(d);
                            true
                        } else {
                            crate::analysis::is_const_addr_always_folded(func, d, &empty_skip)
                        }
                    }
                    IrInst::AddrGlobal { global, .. } => {
                        let g = &program.globals[*global as usize];
                        if g.in_rom {
                            // Only fold ROM globals when all uses are read-only address consumers
                            func.blocks.iter().all(|b| {
                                b.insts.iter().all(|u| {
                                    !u.uses_val(d)
                                        || matches!(
                                            u,
                                            IrInst::FieldAddr { base, .. } if *base == d
                                        )
                                        || matches!(
                                            u,
                                            IrInst::IndexAddr { base, index, .. } if *base == d && *index != d
                                        )
                                        || matches!(u, IrInst::LoadPtr { addr, .. } if *addr == d)
                                }) && !matches!(
                                    b.term,
                                    Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } if v == d
                                )
                            })
                        } else {
                            crate::analysis::is_const_addr_always_folded(func, d, &empty_skip)
                        }
                    }
                    IrInst::LoadLocal { slot, .. } => {
                        if ix_param_slot == Some(*slot) {
                            ix_vals.insert(d, 0);
                            true
                        } else if let Some(reg) = fa.reg_alloc.slot_reg(*slot) {
                            crate::analysis::can_alias_reg_load(
                                program,
                                func,
                                &fa.reg_alloc,
                                bi,
                                ii,
                                d,
                                reg,
                            )
                        } else {
                            crate::analysis::can_alias_local_load(func, bi, ii, d, *slot)
                        }
                    }
                    IrInst::FieldAddr {
                        base, byte_offset, ..
                    } => {
                        if let Some(&disp) = ix_vals.get(base)
                            && let Some(nd) = (disp as i32)
                                .checked_add(*byte_offset as i32)
                                .filter(|&x| (-128..=125).contains(&x))
                        {
                            ix_vals.insert(d, nd as i8);
                            true
                        } else {
                            false
                        }
                    }
                    IrInst::LoadPtr { addr, ty, .. } => {
                        if ix_param_addr_vals.contains(addr) {
                            ix_vals.insert(d, 0);
                            true
                        } else if ix_vals.contains_key(addr)
                            && matches!(program.type_size(ty), 1 | 2)
                            && crate::analysis::can_alias_ix_load(
                                program,
                                func,
                                bi,
                                ii,
                                d,
                                fused_cmp_dsts,
                            )
                        {
                            ix_loaded_vals.insert(d);
                            true
                        } else {
                            false
                        }
                    }
                    IrInst::Cast {
                        from_ty, to_ty, val, ..
                    } => {
                        let from_sz = program.type_size(from_ty);
                        let to_sz = program.type_size(to_ty);
                        if ix_loaded_vals.contains(val)
                            && ((from_sz == 2 && to_sz == 2)
                                || (from_sz == 1
                                    && to_sz == 1
                                    && (*to_ty != IrType::Bool || *from_ty == IrType::Bool)))
                            && crate::analysis::can_alias_ix_load(
                                program,
                                func,
                                bi,
                                ii,
                                d,
                                fused_cmp_dsts,
                            )
                        {
                            ix_loaded_vals.insert(d);
                            true
                        } else {
                            false
                        }
                    }
                    _ => false,
                };
                if folded {
                    no_scratch[d as usize] = true;
                }
            }
        }
    }

    let mut mark_use = |v: ValueId, bi: usize, ii: usize| {
        let idx = v as usize;
        if idx >= nv {
            return;
        }
        match def_loc[idx] {
            Some((dbi, _)) if dbi == bi => {
                last_use_same_bb[idx] = Some(last_use_same_bb[idx].map_or(ii, |cur| cur.max(ii)));
            }
            _ => {
                cross_block[idx] = true;
            }
        }
    };

    for (bi, bb) in func.blocks.iter().enumerate() {
        for (ii, inst) in bb.insts.iter().enumerate() {
            match inst {
                IrInst::BinOp { lhs, rhs, .. } => {
                    mark_use(*lhs, bi, ii);
                    mark_use(*rhs, bi, ii);
                }
                IrInst::UnOp { val, .. }
                | IrInst::Cast { val, .. }
                | IrInst::FixedInt { val, .. }
                | IrInst::StoreLocal { val, .. }
                | IrInst::FieldAddr { base: val, .. }
                | IrInst::LoadPtr { addr: val, .. }
                | IrInst::StateTag {
                    machine_addr: val, ..
                }
                | IrInst::PortIn { port: val, .. }
                | IrInst::Assert { cond: val, .. } => mark_use(*val, bi, ii),
                IrInst::IndexAddr { base, index, .. } => {
                    mark_use(*base, bi, ii);
                    mark_use(*index, bi, ii);
                }
                IrInst::StorePtr { addr, val, .. } => {
                    mark_use(*addr, bi, ii);
                    mark_use(*val, bi, ii);
                }
                IrInst::PortOut { port, val } => {
                    mark_use(*port, bi, ii);
                    mark_use(*val, bi, ii);
                }
                IrInst::Call { args, .. } | IrInst::ExternCall { args, .. } => {
                    for &a in args {
                        mark_use(a, bi, ii);
                    }
                }
                IrInst::StateTransition {
                    machine_addr,
                    field_vals,
                    ..
                } => {
                    mark_use(*machine_addr, bi, ii);
                    for &(_, v, _) in field_vals {
                        mark_use(v, bi, ii);
                    }
                }
                IrInst::StateDispatch {
                    machine_addr, args, ..
                } => {
                    mark_use(*machine_addr, bi, ii);
                    for &a in args {
                        mark_use(a, bi, ii);
                    }
                }
                IrInst::PoolAlloc {
                    pool_addr,
                    init_addr,
                    ..
                } => {
                    mark_use(*pool_addr, bi, ii);
                    mark_use(*init_addr, bi, ii);
                }
                IrInst::PoolFree {
                    pool_addr, handle, ..
                }
                | IrInst::PoolElemAddr {
                    pool_addr, handle, ..
                } => {
                    mark_use(*pool_addr, bi, ii);
                    mark_use(*handle, bi, ii);
                }
                _ => {}
            }
        }
        let term_pos = bb.insts.len();
        if let Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } = bb.term {
            mark_use(v, bi, term_pos);
        }
    }

    let mut offsets = vec![0u16; nv];
    let mut dedicated_bytes = 0u16;
    for v in 0..nv {
        if cross_block[v] && !no_scratch[v] {
            let sz = val_sizes.get(&(v as ValueId)).copied().unwrap_or(2).clamp(1, 2);
            offsets[v] = dedicated_bytes;
            dedicated_bytes = dedicated_bytes.saturating_add(sz);
        }
    }

    // Byte-range interval-color single-block values per block; all blocks share `dedicated_bytes..`
    let mut max_total_bytes = dedicated_bytes;
    for bi in 0..func.blocks.len() {
        // Collect (def_ii, end_ii, size, val_id) for single-block values defined in `bi`
        let mut intervals: Vec<(usize, usize, u16, usize)> = Vec::new();
        for v in 0..nv {
            if !cross_block[v]
                && !no_scratch[v]
                && let Some((dbi, dii)) = def_loc[v]
                && dbi == bi
            {
                let end_ii = last_use_same_bb[v].unwrap_or(dii);
                let sz = val_sizes.get(&(v as ValueId)).copied().unwrap_or(2).clamp(1, 2);
                intervals.push((dii, end_ii, sz, v));
            }
        }
        intervals.sort_by_key(|&(s, e, _, v)| (s, e, v));
        let mut placed: Vec<(u16, u16, usize, usize)> = Vec::new(); // (off, end_off, start, end)
        for (start, end, sz, v) in intervals {
            let mut cand = dedicated_bytes;
            loop {
                let cand_end = cand.saturating_add(sz);
                let conflict = placed
                    .iter()
                    .find(|&&(po, pe, plo, phi)| {
                        start <= phi && end >= plo && cand < pe && cand_end > po
                    })
                    .copied();
                if let Some((_, pe, _, _)) = conflict {
                    cand = pe;
                } else {
                    offsets[v] = cand;
                    placed.push((cand, cand_end, start, end));
                    max_total_bytes = max_total_bytes.max(cand_end);
                    break;
                }
            }
        }
    }

    (offsets, max_total_bytes, dedicated_bytes)
}

fn align_up(addr: u16, align: u16) -> u16 {
    let rem = addr % align;
    if rem == 0 {
        addr
    } else {
        addr.wrapping_add(align - rem)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sibling_functions_share_static_ram() {
        let mut prog = IrProgram::default();
        // fn leaf_a(): 4 bytes local
        prog.functions.push(IrFunction {
            id: 0,
            name: "leaf_a".into(),
            param_count: 0,
            is_pub: false,
            locals: vec![LocalSlot {
                id: 0,
                name: "x".into(),
                ty: IrType::Int(IntType::U32),
                mutable: true,
            }],
            ret_ty: IrType::Unit,
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            blocks: vec![BasicBlock::default()],
            next_val: 0,
        });
        // fn leaf_b(): 4 bytes local
        prog.functions.push(IrFunction {
            id: 1,
            name: "leaf_b".into(),
            param_count: 0,
            is_pub: false,
            locals: vec![LocalSlot {
                id: 0,
                name: "y".into(),
                ty: IrType::Int(IntType::U32),
                mutable: true,
            }],
            ret_ty: IrType::Unit,
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            blocks: vec![BasicBlock::default()],
            next_val: 0,
        });
        // fn main(): 2 bytes local, calls leaf_a and leaf_b
        prog.functions.push(IrFunction {
            id: 2,
            name: "main".into(),
            param_count: 0,
            is_pub: false,
            locals: vec![LocalSlot {
                id: 0,
                name: "z".into(),
                ty: IrType::Int(IntType::U16),
                mutable: true,
            }],
            ret_ty: IrType::Unit,
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            blocks: vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::LoadLocal {
                        dst: 0,
                        slot: 0,
                        ty: IrType::Int(IntType::U16),
                    },
                    IrInst::Call {
                        dst: None,
                        func: 0,
                        args: vec![],
                    },
                    IrInst::Call {
                        dst: None,
                        func: 1,
                        args: vec![],
                    },
                ],
                term: Terminator::Return(None),
            }],
            next_val: 0,
        });

        let layout = allocate_uchiguri(&prog, 0xC000).expect("uchiguri should succeed");
        // leaf_a and leaf_b overlay at 0xC000!
        assert_eq!(layout.slot_addrs[&(0, 0)], 0xC000);
        assert_eq!(layout.slot_addrs[&(1, 0)], 0xC000);
        // main sits above both at 0xC004 (total 6 bytes instead of 10 bytes!)
        assert_eq!(layout.slot_addrs[&(2, 0)], 0xC004);
        assert_eq!(layout.ram_end, 0xC006);
        assert_eq!(layout.max_stack_bytes, 4);
    }

    #[test]
    fn caller_and_callee_scratch_do_not_overlap() {
        let mut prog = IrProgram::default();
        prog.functions.push(IrFunction {
            id: 0,
            name: "callee".into(),
            param_count: 0,
            is_pub: false,
            locals: vec![],
            ret_ty: IrType::Int(IntType::U8),
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            blocks: vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::ConstInt {
                        dst: 0,
                        ty: IrType::Int(IntType::U16),
                        val: 1,
                    },
                    IrInst::ConstInt {
                        dst: 1,
                        ty: IrType::Int(IntType::U16),
                        val: 2,
                    },
                    IrInst::BinOp {
                        dst: 2,
                        op: IrBinOp::Add,
                        mode: ArithMode::Wrapping,
                        ty: IrType::Int(IntType::U16),
                        lhs: 0,
                        rhs: 1,
                        site_id: 0,
                    },
                ],
                term: Terminator::Return(Some(2)),
            }],
            next_val: 3, // 6 bytes live scratch
        });
        prog.functions.push(IrFunction {
            id: 1,
            name: "caller".into(),
            param_count: 0,
            is_pub: false,
            locals: vec![],
            ret_ty: IrType::Unit,
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            blocks: vec![BasicBlock {
                id: 0,
                insts: vec![IrInst::Call {
                    dst: Some(0),
                    func: 0,
                    args: vec![],
                }],
                term: Terminator::Return(None),
            }],
            next_val: 2, // 4 bytes scratch
        });

        let layout = allocate_uchiguri(&prog, 0xC000).expect("uchiguri should succeed");
        assert_eq!(layout.fn_scratch_bases[&0], 0xC000);
        // `callee` folds both `ConstInt`s and needs 2 bytes for the `BinOp::Add` result.
        assert!(layout.fn_scratch_bases[&1] >= layout.fn_scratch_bases[&0] + 2);
        assert!(layout.ram_end > layout.fn_scratch_bases[&1]);
    }

    #[test]
    fn rejects_non_reentrant_recursion() {
        let mut prog = IrProgram::default();
        prog.functions.push(IrFunction {
            id: 0,
            name: "recur".into(),
            param_count: 0,
            is_pub: false,
            locals: vec![],
            ret_ty: IrType::Unit,
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            blocks: vec![BasicBlock {
                id: 0,
                insts: vec![IrInst::Call {
                    dst: None,
                    func: 0,
                    args: vec![],
                }],
                term: Terminator::Return(None),
            }],
            next_val: 0,
        });
        let err = allocate_uchiguri(&prog, 0xC000).expect_err("recursion must fail");
        assert!(err.contains("recursion detected"));
    }

    #[test]
    fn aliased_load_local_extends_slot_liveness_across_subsequent_store() {
        let u8t = IrType::Int(IntType::U8);
        let mut prog = IrProgram {
            externs: vec![IrExternFn {
                name: "_ext_c".into(),
                params: vec![],
                ret: Some((u8t.clone(), "C".into())),
                clobbers: vec!["C".into()],
                requires_di: false,
                requires_banks: vec![],
            }],
            ..Default::default()
        };
        prog.functions.push(IrFunction {
            id: 0,
            name: "alias_span".into(),
            param_count: 0,
            is_pub: false,
            locals: vec![
                LocalSlot {
                    id: 0,
                    name: "s0".into(),
                    ty: u8t.clone(),
                    mutable: false,
                },
                LocalSlot {
                    id: 1,
                    name: "s1".into(),
                    ty: u8t.clone(),
                    mutable: false,
                },
            ],
            ret_ty: u8t.clone(),
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            blocks: vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::ConstInt {
                        dst: 0,
                        ty: u8t.clone(),
                        val: 5,
                    },
                    IrInst::ConstInt {
                        dst: 1,
                        ty: u8t.clone(),
                        val: 1,
                    },
                    IrInst::StoreLocal {
                        slot: 0,
                        val: 0,
                        ty: u8t.clone(),
                    },
                    IrInst::LoadLocal {
                        dst: 2,
                        slot: 0,
                        ty: u8t.clone(),
                    },
                    IrInst::BinOp {
                        dst: 3,
                        op: IrBinOp::Add,
                        mode: ArithMode::Wrapping,
                        ty: u8t.clone(),
                        lhs: 2,
                        rhs: 1,
                        site_id: 0,
                    },
                    IrInst::StoreLocal {
                        slot: 1,
                        val: 3,
                        ty: u8t.clone(),
                    },
                    IrInst::BinOp {
                        dst: 4,
                        op: IrBinOp::Add,
                        mode: ArithMode::Wrapping,
                        ty: u8t.clone(),
                        lhs: 2,
                        rhs: 1,
                        site_id: 0,
                    },
                    IrInst::LoadLocal {
                        dst: 5,
                        slot: 1,
                        ty: u8t.clone(),
                    },
                    IrInst::ExternCall {
                        dst: Some(6),
                        extern_fn: 0,
                        args: vec![],
                    },
                ],
                term: Terminator::Return(Some(4)),
            }],
            next_val: 7,
        });
        let layout = allocate_uchiguri(&prog, 0xC000).expect("uchiguri should succeed");
        assert_ne!(
            layout.slot_addrs[&(0, 0)],
            layout.slot_addrs[&(0, 1)],
            "aliased LoadLocal dst=2 keeps slot 0 live until inst 6, preventing coalescing with slot 1"
        );
    }
}
