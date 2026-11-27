//! *Uchiguri* v1: Call-graph static RAM overlay allocator and stack depth verifier.

use aun_ir::*;
use std::collections::{BTreeMap, HashMap, HashSet};

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
    pub init_ram_end: u16,
    pub overlay_base: u16,
    pub ram_end: u16,
    pub init_bytes: Vec<u8>,
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

fn build_uchiguri_callees(program: &IrProgram) -> Vec<Vec<usize>> {
    let n = program.functions.len();
    let mut addr_taken_fns: Vec<usize> = Vec::new();
    let mut addr_taken_seen = HashSet::new();
    for f in &program.functions {
        for bb in &f.blocks {
            for inst in &bb.insts {
                if let IrInst::AddrFn {
                    target: IrFnTarget::Fn(fid),
                    ..
                } = inst
                {
                    let c = *fid as usize;
                    if c < n && addr_taken_seen.insert(c) {
                        addr_taken_fns.push(c);
                    }
                }
            }
        }
    }
    let mut callees: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, f) in program.functions.iter().enumerate() {
        let mut seen = HashSet::new();
        for bb in &f.blocks {
            for inst in &bb.insts {
                match inst {
                    IrInst::Call { func, .. } => {
                        let c = *func as usize;
                        if c < n && seen.insert(c) {
                            callees[i].push(c);
                        }
                    }
                    IrInst::CallIndirect { .. } => {
                        for &c in &addr_taken_fns {
                            if c != i && seen.insert(c) {
                                callees[i].push(c);
                            }
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
                            if c < n && seen.insert(c) {
                                callees[i].push(c);
                            }
                        }
                        if let Some(from) = from_state
                            && let Some(exit_fn) = mdef.states[*from as usize].on_exit
                        {
                            let c = exit_fn as usize;
                            if c < n && seen.insert(c) {
                                callees[i].push(c);
                            }
                        }
                    }
                    IrInst::StateDispatch {
                        machine_id, method, ..
                    } => {
                        let mdef = &program.machines[*machine_id as usize];
                        for st in &mdef.states {
                            if let Some(&fid) = st.methods.get(method) {
                                let c = fid as usize;
                                if c < n && seen.insert(c) {
                                    callees[i].push(c);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    callees
}

fn mark_reach(u: usize, callees: &[Vec<usize>], flags: &mut [bool]) {
    if flags[u] {
        return;
    }
    flags[u] = true;
    for &v in &callees[u] {
        mark_reach(v, callees, flags);
    }
}

/// Auto-clones non-recursive `#[reentrant]` helpers shared between main/pub execution and an ISR
/// priority group (or across distinct ISR priority groups) into private per-ISR copies so each
/// context receives isolated static Uchiguri RAM frames without hardware stack spilling.
pub fn split_isr_shared_reentrant_fns(program: &mut IrProgram) {
    let n = program.functions.len();
    if n == 0 {
        return;
    }
    let callees = build_uchiguri_callees(program);
    let mut main_reachable = vec![false; n];
    let mut isr_groups: BTreeMap<u8, Vec<bool>> = BTreeMap::new();
    for (i, f) in program.functions.iter().enumerate() {
        if f.is_test || f.blocks.is_empty() {
            continue;
        }
        if let Some(prio) = program.isr_priority(f) {
            let flags = isr_groups.entry(prio).or_insert_with(|| vec![false; n]);
            mark_reach(i, &callees, flags);
        } else if f.name == "main" || f.is_pub {
            mark_reach(i, &callees, &mut main_reachable);
        }
    }
    if isr_groups.is_empty() {
        return;
    }

    // Mark functions participating in recursive call cycles (cloning cannot eliminate true recursion).
    let mut in_cycle = vec![false; n];
    for start in 0..n {
        let mut seen = vec![false; n];
        let mut stack = callees[start].clone();
        while let Some(cur) = stack.pop() {
            if cur == start {
                in_cycle[start] = true;
                break;
            }
            if !seen[cur] {
                seen[cur] = true;
                stack.extend(&callees[cur]);
            }
        }
    }

    let prios: Vec<u8> = isr_groups.keys().copied().collect();
    for (idx, prio) in prios.iter().enumerate() {
        let group_reach = &isr_groups[prio];
        let mut clone_map: HashMap<u32, u32> = HashMap::new();
        for i in 0..n {
            let f = &program.functions[i];
            if !group_reach[i] || f.is_test || f.blocks.is_empty() || !f.is_reentrant || in_cycle[i]
            {
                continue;
            }
            let shared_with_other = main_reachable[i]
                || prios[..idx]
                    .iter()
                    .any(|p| isr_groups[p].get(i).copied().unwrap_or(false));
            if !shared_with_other {
                continue;
            }
            let orig_id = f.id;
            let clone_id = program.functions.len() as u32;
            let mut cloned = f.clone();
            cloned.id = clone_id;
            cloned.name = format!("{}__isr_p{prio}", f.name);
            cloned.is_pub = false;
            cloned.is_reentrant = false;
            cloned.interrupt_source = None;
            program.functions.push(cloned);
            clone_map.insert(orig_id, clone_id);
        }
        if clone_map.is_empty() {
            continue;
        }
        let mut rewrite_targets: Vec<usize> = Vec::new();
        for (i, _) in program.functions[..n].iter().enumerate() {
            if group_reach[i]
                && !main_reachable[i]
                && !clone_map.contains_key(&(i as u32))
                && prios
                    .iter()
                    .all(|p| p == prio || !isr_groups[p].get(i).copied().unwrap_or(false))
            {
                rewrite_targets.push(i);
            }
        }
        for &cid in clone_map.values() {
            rewrite_targets.push(cid as usize);
        }
        for fi in rewrite_targets {
            for bb in &mut program.functions[fi].blocks {
                for inst in &mut bb.insts {
                    if let IrInst::Call { func, .. } = inst
                        && let Some(&mapped) = clone_map.get(func)
                    {
                        *func = mapped;
                    }
                }
            }
        }
    }
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

    // 1. Place persistent globals: ROM constants and extern statics take 0 RAM;
    //    Pass 1a packs non-zero-initialized unbanked RAM globals contiguously at `ram_base..init_ram_end`
    //    so `--format sdasz80` can emit `.area _INITIALIZED` / `.area _INITIALIZER`.
    //    Pass 1b places fixed-address, banked, and zero-initialized RAM globals afterwards.
    let mut init_bytes: Vec<u8> = Vec::new();
    let banked_spaces: HashSet<String> = bank_cursors.keys().cloned().collect();
    let is_nonzero_init_ram = |g: &IrGlobal| -> bool {
        !g.in_rom
            && !g.is_extern
            && g.fixed_addr.is_none()
            && g.space
                .as_ref()
                .is_none_or(|s| !banked_spaces.contains(s))
            && g.init.iter().any(|&b| b != 0)
    };
    for (gid, g) in program.globals.iter().enumerate() {
        if !is_nonzero_init_ram(g) {
            continue;
        }
        let align = g.align.max(1);
        let size = program.type_size(&g.ty).max(1);
        let placed = align_up(cursor, align);
        let pad = placed.saturating_sub(cursor) as usize;
        if pad > 0 {
            init_bytes.resize(init_bytes.len() + pad, 0);
        }
        let start_len = init_bytes.len();
        init_bytes.resize(start_len + size as usize, 0);
        let copy_len = g.init.len().min(size as usize);
        init_bytes[start_len..start_len + copy_len].copy_from_slice(&g.init[..copy_len]);
        cursor = placed.wrapping_add(size);
        global_addrs.insert(gid as u32, placed);
        global_entries.push(GlobalRamEntry {
            global_id: gid as u32,
            name: g.name.clone(),
            addr: placed,
            size,
        });
    }
    let init_ram_end = cursor;

    for (gid, g) in program.globals.iter().enumerate() {
        if g.in_rom || g.is_extern || is_nonzero_init_ram(g) {
            continue;
        }
        let align = g.align.max(1);
        let size = program.type_size(&g.ty).max(1);
        let addr = if let Some(fa) = g.fixed_addr {
            fa
        } else if let Some(ref space_name) = g.space
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
    let mut addr_taken_fns: Vec<usize> = Vec::new();
    let mut addr_taken_seen = HashSet::new();
    for f in &program.functions {
        for bb in &f.blocks {
            for inst in &bb.insts {
                if let IrInst::AddrFn {
                    target: IrFnTarget::Fn(fid),
                    ..
                } = inst
                {
                    let c = *fid as usize;
                    if c < n && addr_taken_seen.insert(c) {
                        addr_taken_fns.push(c);
                    }
                }
            }
        }
    }
    let mut callees: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut calls_extern = vec![false; n];
    let mut calls_callback_extern = vec![false; n];
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
                    IrInst::CallIndirect { .. } => {
                        calls_extern[i] = true;
                        calls_callback_extern[i] = true;
                        for &c in &addr_taken_fns {
                            if c != i && seen.insert(c) {
                                callees[i].push(c);
                            }
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
                    IrInst::ExternCall { extern_fn, .. } => {
                        calls_extern[i] = true;
                        let ef = &program.externs[*extern_fn as usize];
                        if !ef.no_callback {
                            let em = crate::regalloc::extern_clobber_mask(ef);
                            if em.contains(crate::regalloc::RegMask::BC) {
                                calls_callback_extern[i] = true;
                            }
                        }
                    }
                    IrInst::Assert { .. }
                    | IrInst::Panic { .. }
                    | IrInst::IndexAddr { .. }
                    | IrInst::PoolAlloc { .. }
                    | IrInst::PoolFree { .. }
                    | IrInst::PoolElemAddr { .. }
                    | IrInst::BinOp {
                        op: IrBinOp::Div | IrBinOp::Rem,
                        ..
                    } => {
                        calls_extern[i] = true;
                    }
                    IrInst::BinOp {
                        op: IrBinOp::Mul,
                        ty,
                        ..
                    } if program.type_size(ty) == 4 => {
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
        let (val_offs, scratch, ded_bytes) =
            color_fn_scratch(program, f, fa, &analysis.fn_clobbers);
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
        // Account for callee-saved `push ix` (+2B), caller-saved `push ix` around
        // `ix_in_reg` calls (+2B), and caller-saved `push bc` (+2B for B/C-reg local).
        let calls_ix_in_reg = callees[u]
            .iter()
            .any(|&v| analysis.fns[v].ix_in_reg);
        let ix_stack = if analysis.fns[u].save_ix
            || (analysis.fns[u].ix_param_slot.is_some() && calls_ix_in_reg)
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

    // 4b. Extern callback isolation: any function that can reach a caller-saved (BC-clobbering)
    //     extern call or indirect call must not overlap its Uchiguri RAM frame with `pub fn`
    //     entry points that external C/ASM code may call back into.
    let mut trans_callback_extern = calls_callback_extern;
    loop {
        let mut changed = false;
        for u in 0..n {
            if !trans_callback_extern[u]
                && callees[u].iter().any(|&v| trans_callback_extern[v])
            {
                trans_callback_extern[u] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let pub_callbacks: Vec<usize> = (0..n)
        .filter(|&v| {
            !program.functions[v].is_test
                && program.functions[v].is_pub
                && !program.is_isr_root(&program.functions[v])
                && program.functions[v].name != "main"
        })
        .collect();
    if !pub_callbacks.is_empty() {
        for u in 0..n {
            let f = &program.functions[u];
            if f.is_test || program.is_isr_root(f) || !trans_callback_extern[u] {
                continue;
            }
            let cb_max_end = pub_callbacks
                .iter()
                .filter(|&&v| {
                    v != u && (!f.is_pub || f.name == "main" || !trans_callback_extern[v])
                })
                .map(|&v| fn_overlay_offset[v].saturating_add(frame_sizes[v]))
                .max()
                .unwrap_or(0);
            fn_overlay_offset[u] = fn_overlay_offset[u].max(cb_max_end);
        }
        for _ in 0..n {
            let mut changed = false;
            for u in 0..n {
                for &v in &callees[u] {
                    let req = fn_overlay_offset[v].saturating_add(frame_sizes[v]);
                    if req > fn_overlay_offset[u] {
                        fn_overlay_offset[u] = req;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }

    // 5. Partition ISR-reachable functions (grouped by interrupt priority) from main-thread
    //    functions (including non-main `pub fn` roots in library mode) and reject shared callees!
    let mut main_reachable = vec![false; n];
    let mut isr_groups: BTreeMap<u8, Vec<bool>> = BTreeMap::new();
    for i in 0..n {
        let f = &program.functions[i];
        if f.is_test {
            continue;
        }
        if let Some(prio) = program.isr_priority(f) {
            let flags = isr_groups.entry(prio).or_insert_with(|| vec![false; n]);
            mark_reach(i, &callees, flags);
        } else if f.name == "main" || f.is_pub {
            mark_reach(i, &callees, &mut main_reachable);
        }
    }
    let mut in_any_isr = vec![false; n];
    for flags in isr_groups.values() {
        for i in 0..n {
            if flags[i] {
                in_any_isr[i] = true;
            }
        }
    }
    for i in 0..n {
        let isr_ctx_count = isr_groups.values().filter(|g| g[i]).count();
        if main_reachable[i] && in_any_isr[i] {
            if !program.functions[i].is_reentrant {
                return Err(format!(
                    "re-entrancy hazard: non-#[reentrant] function `{}` is reachable from both `main`/`pub` and an ISR",
                    program.functions[i].name
                ));
            }
            return Err(format!(
                "re-entrancy hazard: shared #[reentrant] function `{}` could not be isolated into separate main/ISR copies",
                program.functions[i].name
            ));
        }
        if isr_ctx_count >= 2 && !program.functions[i].is_reentrant {
            return Err(format!(
                "re-entrancy hazard: non-#[reentrant] function `{}` is reachable from multiple ISR priority levels",
                program.functions[i].name
            ));
        }
    }

    // 6. Assign final concrete addresses: main-thread overlay first, then per-priority isolated ISR regions
    let mut slot_addrs = HashMap::new();
    let mut fn_scratch_bases = HashMap::new();
    let mut fn_sb_scratch_ranges = HashMap::new();
    let mut fn_frames = Vec::new();
    let mut main_end = overlay_base;
    let mut max_main_stack = 0u16;

    for i in 0..n {
        if program.functions[i].is_test || in_any_isr[i] {
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

    // Place each ISR priority group strictly after main and other ISR priority groups
    let mut cur_isr_base = main_end;
    let mut max_end = main_end;
    let mut max_isr_stack = 0u16;
    for group_reach in isr_groups.values() {
        let mut group_end = cur_isr_base;
        let mut group_max_stack = 0u16;
        for i in 0..n {
            if program.functions[i].is_test || !group_reach[i] {
                continue;
            }
            let func = &program.functions[i];
            let fn_base = cur_isr_base.wrapping_add(fn_overlay_offset[i]);
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
            group_end = group_end.max(fn_end);
            // Hardware INT return addr (2B) + PUSH AF/BC/DE/HL/IX (10B) = 12B; C-side #[isr_entry] = 2B
            let entry_overhead = if func.interrupt_source.is_some() { 12 } else { 2 };
            let total_stack = fn_stack_depth[i] + entry_overhead;
            group_max_stack = group_max_stack.max(total_stack);
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
        cur_isr_base = group_end;
        max_end = max_end.max(group_end);
        max_isr_stack = max_isr_stack.saturating_add(group_max_stack);
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
        init_ram_end,
        overlay_base,
        ram_end: max_end,
        init_bytes,
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

    // Map BlockId -> (start_pos, end_pos) in flattened RPO instruction space
    let cfg = aun_ir::cfg::Cfg::build(func);
    let mut bb_pos: HashMap<BlockId, (usize, usize)> = HashMap::new();
    let mut cur_pos = 0usize;
    for &bi in cfg.rpo_indices() {
        let bb = &func.blocks[bi];
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

    // Collect natural loop intervals from CFG dominance analysis (ignoring forward edges in Vec order)
    let mut loops: Vec<(usize, usize)> = Vec::new();
    for lp in cfg.natural_loops() {
        if let Some(&(l_lo, _)) = bb_pos.get(&lp.header) {
            let l_hi = lp
                .body
                .iter()
                .filter_map(|id| bb_pos.get(id).map(|&(_, end)| end))
                .max()
                .unwrap_or(l_lo);
            loops.push((l_lo, l_hi));
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
    fn_clobbers: &[crate::regalloc::RegMask],
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
    let mut const_addr_vals: HashSet<ValueId> = HashSet::new();
    let cfg = aun_ir::cfg::Cfg::build(func);
    let mut visit_order: Vec<usize> = cfg.rpo_indices().to_vec();
    let mut seen_bi = vec![false; func.blocks.len()];
    for &bi in &visit_order {
        if bi < seen_bi.len() {
            seen_bi[bi] = true;
        }
    }
    for (bi, &visited) in seen_bi.iter().enumerate() {
        if !visited {
            visit_order.push(bi);
        }
    }
    for bi in visit_order {
        let bb = &func.blocks[bi];
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
                            const_addr_vals.insert(d);
                            crate::analysis::is_const_addr_always_folded(
                                program,
                                func,
                                d,
                                &empty_skip,
                            )
                        }
                    }
                    IrInst::AddrGlobal { .. } | IrInst::AddrFn { .. } => {
                        const_addr_vals.insert(d);
                        crate::analysis::is_const_addr_always_folded(
                            program,
                            func,
                            d,
                            &empty_skip,
                        )
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
                                fn_clobbers,
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
                        } else if from_sz == 2
                            && to_sz == 2
                            && const_addr_vals.contains(val)
                        {
                            const_addr_vals.insert(d);
                            crate::analysis::is_const_addr_always_folded(
                                program,
                                func,
                                d,
                                &empty_skip,
                            )
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
            inst.for_each_use(|v| mark_use(v, bi, ii));
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
            let sz = val_sizes.get(&(v as ValueId)).copied().unwrap_or(2).clamp(1, 4);
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
                let sz = val_sizes.get(&(v as ValueId)).copied().unwrap_or(2).clamp(1, 4);
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
            isr_using: None,
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
            isr_using: None,
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
            isr_using: None,
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
            isr_using: None,
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
            isr_using: None,
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
            isr_using: None,
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
                clobbers: Some(vec!["C".into()]),
                requires_di: false,
                requires_banks: vec![],
                fixed_addr: None,
                no_callback: false,
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
            isr_using: None,
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

    #[test]
    fn extern_callback_does_not_overlap_caller_and_pub_callee() {
        let u16t = IrType::Int(IntType::U16);
        let mut prog = IrProgram::default();
        prog.externs.push(IrExternFn {
            name: "_physics_step".into(),
            params: vec![],
            ret: None,
            clobbers: Some(vec!["AF".into(), "BC".into(), "DE".into(), "HL".into()]),
            requires_di: false,
            requires_banks: vec![],
            fixed_addr: None,
            no_callback: false,
        });
        // `pub fn callback()` (id = 0) with a 2-byte address-taken local
        prog.functions.push(IrFunction {
            id: 0,
            name: "callback".into(),
            is_pub: true,
            param_count: 0,
            locals: vec![LocalSlot {
                id: 0,
                name: "cb_loc".into(),
                ty: u16t.clone(),
                mutable: true,
            }],
            ret_ty: IrType::Unit,
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            isr_using: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            blocks: vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::AddrLocal { dst: 0, slot: 0 },
                    IrInst::ConstInt {
                        dst: 1,
                        ty: u16t.clone(),
                        val: 42,
                    },
                    IrInst::StorePtr {
                        addr: 0,
                        val: 1,
                        ty: u16t.clone(),
                        volatile: false,
                    },
                ],
                term: Terminator::Return(None),
            }],
            next_val: 2,
        });
        // `pub fn main()` (id = 1) with an address-taken local calling `_physics_step`
        prog.functions.push(IrFunction {
            id: 1,
            name: "main".into(),
            is_pub: true,
            param_count: 0,
            locals: vec![LocalSlot {
                id: 0,
                name: "main_loc".into(),
                ty: u16t.clone(),
                mutable: true,
            }],
            ret_ty: IrType::Unit,
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            isr_using: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            blocks: vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::AddrLocal { dst: 0, slot: 0 },
                    IrInst::ConstInt {
                        dst: 1,
                        ty: u16t.clone(),
                        val: 7,
                    },
                    IrInst::StorePtr {
                        addr: 0,
                        val: 1,
                        ty: u16t.clone(),
                        volatile: false,
                    },
                    IrInst::ExternCall {
                        dst: None,
                        extern_fn: 0,
                        args: vec![],
                    },
                ],
                term: Terminator::Return(None),
            }],
            next_val: 2,
        });
        let layout = allocate_uchiguri(&prog, 0xC000).expect("uchiguri should succeed");
        assert_ne!(
            layout.slot_addrs[&(0, 0)],
            layout.slot_addrs[&(1, 0)],
            "main calling BC-clobbering extern must not overlap pub callback locals"
        );

        prog.externs[0].no_callback = true;
        let opt_layout = allocate_uchiguri(&prog, 0xC000).expect("uchiguri should succeed");
        assert_eq!(
            opt_layout.slot_addrs[&(0, 0)],
            opt_layout.slot_addrs[&(1, 0)],
            "main calling #[no_callback] BC-clobbering extern may overlay with unrelated pub fn"
        );
        assert!(opt_layout.ram_end < layout.ram_end);
    }

    #[test]
    fn library_mode_pub_fn_and_isr_entry_are_isolated_and_split_reentrant() {
        let u8t = IrType::Int(IntType::U8);
        let mk_fn = |id: u32, name: &str, is_pub: bool, is_reentrant: bool, blocks| IrFunction {
            id,
            name: name.into(),
            is_pub,
            param_count: 0,
            locals: vec![LocalSlot {
                id: 0,
                name: "x".into(),
                ty: u8t.clone(),
                mutable: true,
            }],
            ret_ty: IrType::Unit,
            is_test: false,
            expected_panic: None,
            is_reentrant,
            interrupt_source: None,
            isr_using: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            blocks,
            next_val: 2,
        };
        let write_local_blk = vec![BasicBlock {
            id: 0,
            insts: vec![
                IrInst::AddrLocal { dst: 0, slot: 0 },
                IrInst::ConstInt {
                    dst: 1,
                    ty: u8t.clone(),
                    val: 1,
                },
                IrInst::StorePtr {
                    addr: 0,
                    val: 1,
                    ty: u8t.clone(),
                    volatile: false,
                },
            ],
            term: Terminator::Return(None),
        }];
        let call_shared_blk = vec![BasicBlock {
            id: 0,
            insts: vec![
                IrInst::AddrLocal { dst: 0, slot: 0 },
                IrInst::ConstInt {
                    dst: 1,
                    ty: u8t.clone(),
                    val: 1,
                },
                IrInst::StorePtr {
                    addr: 0,
                    val: 1,
                    ty: u8t.clone(),
                    volatile: false,
                },
                IrInst::Call {
                    dst: None,
                    func: 2,
                    args: vec![],
                },
            ],
            term: Terminator::Return(None),
        }];

        // Non-reentrant helper shared between pub fn (lib mode, no `main`) and #[isr_entry] must error.
        let mut err_prog = IrProgram::default();
        err_prog.functions.push(mk_fn(0, "api_fn", true, false, call_shared_blk.clone()));
        err_prog.functions.push(mk_fn(1, "isr_cb", true, false, call_shared_blk.clone()));
        err_prog.functions.push(mk_fn(2, "shared", false, false, write_local_blk.clone()));
        err_prog.isr_entries.insert(1, 0);
        assert!(allocate_uchiguri(&err_prog, 0xC000).is_err());

        // Marking `shared` #[reentrant] splits it cleanly into `shared` and `shared__isr_p0`.
        err_prog.functions[2].is_reentrant = true;
        split_isr_shared_reentrant_fns(&mut err_prog);
        assert_eq!(err_prog.functions.len(), 4);
        let layout = allocate_uchiguri(&err_prog, 0xC000).expect("split clone isolates frames");
        assert_ne!(layout.slot_addrs[&(0, 0)], layout.slot_addrs[&(1, 0)]);
        assert_ne!(layout.slot_addrs[&(2, 0)], layout.slot_addrs[&(3, 0)]);
    }

    #[test]
    fn multi_priority_isrs_get_disjoint_ram_and_additive_stack() {
        let u8t = IrType::Int(IntType::U8);
        let mk_isr = |id: u32, name: &str| IrFunction {
            id,
            name: name.into(),
            is_pub: false,
            param_count: 0,
            locals: vec![LocalSlot {
                id: 0,
                name: "x".into(),
                ty: u8t.clone(),
                mutable: true,
            }],
            ret_ty: IrType::Unit,
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: Some(name.into()),
            isr_using: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            blocks: vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::AddrLocal { dst: 0, slot: 0 },
                    IrInst::ConstInt {
                        dst: 1,
                        ty: u8t.clone(),
                        val: 5,
                    },
                    IrInst::StorePtr {
                        addr: 0,
                        val: 1,
                        ty: u8t.clone(),
                        volatile: false,
                    },
                ],
                term: Terminator::Return(None),
            }],
            next_val: 2,
        };
        let mut prog = IrProgram::default();
        prog.functions.push(mk_isr(0, "irq_low"));
        prog.functions.push(mk_isr(1, "irq_high"));
        prog.isr_priorities.insert(0, 0);
        prog.isr_priorities.insert(1, 1);

        let layout = allocate_uchiguri(&prog, 0xC000).expect("multi-priority ISRs allocate");
        assert_ne!(
            layout.slot_addrs[&(0, 0)],
            layout.slot_addrs[&(1, 0)],
            "ISRs at distinct priority levels must not share RAM"
        );
        // Each hardware ISR adds 12B save/return stack -> >= 24B total ISR stack.
        assert!(layout.max_stack_bytes >= 24);
    }
}
