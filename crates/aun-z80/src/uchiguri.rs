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
    pub global_entries: Vec<GlobalRamEntry>,
    pub fn_frames: Vec<FnRamFrame>,
    pub max_stack_bytes: u16,
}

/// Allocates fixed RAM addresses for globals and non-overlapping call-tree function locals
/// and SSA temporary scratch slots.
pub fn allocate_uchiguri(program: &IrProgram, ram_base: u16) -> Result<RamLayout, String> {
    let mut cursor = ram_base;
    let mut global_addrs = HashMap::new();
    let mut global_entries = Vec::new();

    // 1. Place persistent globals
    for (gid, g) in program.globals.iter().enumerate() {
        let align = g.align.max(1);
        cursor = align_up(cursor, align);
        let size = program.type_size(&g.ty).max(1);
        global_addrs.insert(gid as u32, cursor);
        global_entries.push(GlobalRamEntry {
            global_id: gid as u32,
            name: g.name.clone(),
            addr: cursor,
            size,
        });
        cursor = cursor.wrapping_add(size);
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

    // 3. Compute frame sizes per function (locals + per-function SSA temporary scratch)
    let mut locals_sizes = vec![0u16; n];
    let mut scratch_sizes = vec![0u16; n];
    let mut frame_sizes = vec![0u16; n];
    let mut local_rel_offsets: Vec<Vec<u16>> = vec![Vec::new(); n];
    for (i, f) in program.functions.iter().enumerate() {
        let mut off = 0u16;
        for slot in &f.locals {
            local_rel_offsets[i].push(off);
            off = off.wrapping_add(program.type_size(&slot.ty).max(1));
        }
        locals_sizes[i] = off;
        let scratch = (f.next_val as u16).saturating_mul(2);
        scratch_sizes[i] = scratch;
        frame_sizes[i] = off.saturating_add(scratch);
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
    let mut isr_reachable = vec![false; n];
    fn mark_isr(u: usize, callees: &[Vec<usize>], isr_reachable: &mut [bool]) {
        if isr_reachable[u] {
            return;
        }
        isr_reachable[u] = true;
        for &v in &callees[u] {
            mark_isr(v, callees, isr_reachable);
        }
    }
    for i in 0..n {
        if !program.functions[i].is_test && program.functions[i].interrupt_source.is_some() {
            mark_isr(i, &callees, &mut isr_reachable);
        }
    }

    // 6. Assign final concrete addresses: main-thread overlay first, then isolated ISR region
    let mut slot_addrs = HashMap::new();
    let mut fn_scratch_bases = HashMap::new();
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
        let fn_end = fn_base.wrapping_add(frame_sizes[i]);
        max_end = max_end.max(fn_end);
        // Hardware INT return addr (2B) + PUSH AF/BC/DE/HL (8B) = 10B for ISR entry
        let entry_overhead = if func.interrupt_source.is_some() { 10 } else { 2 };
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

    Ok(RamLayout {
        ram_base,
        overlay_base,
        ram_end: max_end,
        global_addrs,
        slot_addrs,
        fn_scratch_bases,
        global_entries,
        fn_frames,
        max_stack_bytes: max_stack,
    })
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
            locals: vec![],
            ret_ty: IrType::Int(IntType::U8),
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            interrupt_source: None,
            requires_banks: vec![],
            requires_di: false,
            overlay_state: None,
            blocks: vec![BasicBlock::default()],
            next_val: 3, // 6 bytes scratch
        });
        prog.functions.push(IrFunction {
            id: 1,
            name: "caller".into(),
            param_count: 0,
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
        assert_eq!(layout.fn_scratch_bases[&1], 0xC006);
        assert_eq!(layout.ram_end, 0xC00A);
    }

    #[test]
    fn rejects_non_reentrant_recursion() {
        let mut prog = IrProgram::default();
        prog.functions.push(IrFunction {
            id: 0,
            name: "recur".into(),
            param_count: 0,
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
