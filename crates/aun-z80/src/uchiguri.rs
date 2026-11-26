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
        let (rel_offs, loc_sz) = color_fn_locals(program, f);
        local_rel_offsets[i] = rel_offs;
        locals_sizes[i] = loc_sz;
        let (val_offs, scratch, ded_bytes) = color_fn_scratch(f);
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
        for &v in &callees[u] {
            dfs(
                v,
                program,
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
        fn_stack_depth[u] = max_child_stack;
        state[u] = 2;
        Ok(())
    }

    for i in 0..n {
        if !program.functions[i].is_test {
            dfs(
                i,
                program,
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
fn color_fn_locals(program: &IrProgram, func: &IrFunction) -> (Vec<u16>, u16) {
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
    let mut min_pos: Vec<Option<usize>> = vec![None; n_slots];
    let mut max_pos: Vec<Option<usize>> = vec![None; n_slots];

    // Map BlockId -> (start_pos, end_pos) in flattened instruction space
    let mut bb_pos: HashMap<BlockId, (usize, usize)> = HashMap::new();
    let mut cur_pos = 0usize;
    for bb in &func.blocks {
        let start = cur_pos;
        for inst in &bb.insts {
            match inst {
                IrInst::AddrLocal { dst, slot } => {
                    let s = *slot as usize;
                    if s < n_slots {
                        min_pos[s] = Some(min_pos[s].map_or(cur_pos, |m| m.min(cur_pos)));
                        max_pos[s] = Some(max_pos[s].map_or(cur_pos, |m| m.max(cur_pos)));
                        let escapes = func.blocks.iter().any(|b| {
                            b.insts.iter().any(|u| {
                                crate::codegen::inst_uses_val_pub(u, *dst)
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
                IrInst::LoadLocal { slot, .. } | IrInst::StoreLocal { slot, .. } => {
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

    // Extend any slot's live range that crosses a loop boundary (starts before the loop and
    // is used inside/after it) to cover the full loop. Loop-body-only locals (strictly inside
    // `[l_lo, l_hi]` with no live-in value) stay scoped to their `[lo, hi]` interval!
    for s in 0..n_slots {
        if let (Some(lo), Some(mut hi)) = (min_pos[s], max_pos[s]) {
            let mut changed = true;
            while changed {
                changed = false;
                for &(l_lo, l_hi) in &loops {
                    if lo < l_lo && hi >= l_lo && hi < l_hi {
                        hi = l_hi;
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
        if s < func.param_count || addr_escaped[s] {
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

/// Assigns compact 2-byte scratch offsets to SSA `ValueId`s in `func`:
/// - Any `ValueId` used in a different basic block than where it was defined gets a dedicated slot.
/// - Single-block `ValueId`s are interval-colored per block and share a tiny scratch pool across blocks.
/// Returns `(val_offsets, total_scratch_bytes, dedicated_bytes)`.
fn color_fn_scratch(func: &IrFunction) -> (Vec<u16>, u16, u16) {
    let nv = func.next_val as usize;
    if nv == 0 {
        return (Vec::new(), 0, 0);
    }
    let mut def_loc: Vec<Option<(usize, usize)>> = vec![None; nv];
    let mut last_use_same_bb: Vec<Option<usize>> = vec![None; nv];
    let mut cross_block: Vec<bool> = vec![false; nv];

    let inst_def = |inst: &IrInst| -> Option<ValueId> {
        match inst {
            IrInst::ConstInt { dst, .. }
            | IrInst::ConstBool { dst, .. }
            | IrInst::BinOp { dst, .. }
            | IrInst::UnOp { dst, .. }
            | IrInst::Cast { dst, .. }
            | IrInst::FixedInt { dst, .. }
            | IrInst::LoadLocal { dst, .. }
            | IrInst::AddrLocal { dst, .. }
            | IrInst::AddrGlobal { dst, .. }
            | IrInst::FieldAddr { dst, .. }
            | IrInst::IndexAddr { dst, .. }
            | IrInst::LoadPtr { dst, .. }
            | IrInst::StateTag { dst, .. }
            | IrInst::PoolAlloc { dst, .. }
            | IrInst::PoolElemAddr { dst, .. }
            | IrInst::PortIn { dst, .. } => Some(*dst),
            IrInst::Call { dst, .. }
            | IrInst::ExternCall { dst, .. }
            | IrInst::StateDispatch { dst, .. } => *dst,
            _ => None,
        }
    };

    for (bi, bb) in func.blocks.iter().enumerate() {
        for (ii, inst) in bb.insts.iter().enumerate() {
            if let Some(d) = inst_def(inst)
                && (d as usize) < nv
            {
                def_loc[d as usize] = Some((bi, ii));
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
    let mut dedicated_slots = 0u16;
    for v in 0..nv {
        if cross_block[v] {
            offsets[v] = dedicated_slots.saturating_mul(2);
            dedicated_slots = dedicated_slots.saturating_add(1);
        }
    }

    // Color single-block values per block; all blocks share slots `dedicated_slots..`
    let mut max_total_slots = dedicated_slots;
    for bi in 0..func.blocks.len() {
        // Collect (def_ii, end_ii, val_id) for single-block values defined in `bi`
        let mut intervals: Vec<(usize, usize, usize)> = Vec::new();
        for v in 0..nv {
            if !cross_block[v]
                && let Some((dbi, dii)) = def_loc[v]
                && dbi == bi
            {
                let end_ii = last_use_same_bb[v].unwrap_or(dii);
                intervals.push((dii, end_ii, v));
            }
        }
        intervals.sort_by_key(|&(s, e, v)| (s, e, v));
        // Track end_ii for each local slot color (a slot is free when cur_start > slot_end)
        let mut slot_ends: Vec<usize> = Vec::new();
        for (start, end, v) in intervals {
            let mut chosen = None;
            for (c, se) in slot_ends.iter_mut().enumerate() {
                if start > *se {
                    *se = end;
                    chosen = Some(c as u16);
                    break;
                }
            }
            let color = match chosen {
                Some(c) => c,
                None => {
                    let c = slot_ends.len() as u16;
                    slot_ends.push(end);
                    c
                }
            };
            let global_slot = dedicated_slots.saturating_add(color);
            offsets[v] = global_slot.saturating_mul(2);
            max_total_slots = max_total_slots.max(global_slot.saturating_add(1));
        }
    }

    (
        offsets,
        max_total_slots.saturating_mul(2),
        dedicated_slots.saturating_mul(2),
    )
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
        assert!(layout.fn_scratch_bases[&1] >= layout.fn_scratch_bases[&0] + 4);
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
}
