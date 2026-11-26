//! *Uchiguri* v1: Call-graph static RAM overlay allocator and stack depth verifier.

use aun_ir::*;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct RamLayout {
    pub ram_base: u16,
    pub ram_end: u16,
    pub global_addrs: HashMap<u32, u16>,
    /// `(func_id, slot_id) -> fixed RAM address`
    pub slot_addrs: HashMap<(u32, SlotId), u16>,
    pub max_stack_bytes: u16,
}

/// Allocates fixed RAM addresses for globals and non-overlapping call-tree function locals.
pub fn allocate_uchiguri(program: &IrProgram, ram_base: u16) -> Result<RamLayout, String> {
    let mut cursor = ram_base;
    let mut global_addrs = HashMap::new();

    // 1. Place persistent globals
    for (gid, g) in program.globals.iter().enumerate() {
        let align = g.align.max(1);
        cursor = align_up(cursor, align);
        global_addrs.insert(gid as u32, cursor);
        cursor = cursor.wrapping_add(g.ty.byte_size(&program.structs).max(1));
    }

    let overlay_base = cursor;

    // 2. Build static call graph over internal functions
    let n = program.functions.len();
    let mut callees: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, f) in program.functions.iter().enumerate() {
        let mut seen = HashSet::new();
        for bb in &f.blocks {
            for inst in &bb.insts {
                if let IrInst::Call { func, .. } = inst {
                    let c = *func as usize;
                    if seen.insert(c) {
                        callees[i].push(c);
                    }
                }
            }
        }
    }

    // 3. Compute frame sizes per function
    let mut frame_sizes = vec![0u16; n];
    let mut local_rel_offsets: Vec<Vec<u16>> = vec![Vec::new(); n];
    for (i, f) in program.functions.iter().enumerate() {
        let mut off = 0u16;
        for slot in &f.locals {
            local_rel_offsets[i].push(off);
            off = off.wrapping_add(slot.ty.byte_size(&program.structs).max(1));
        }
        frame_sizes[i] = off;
    }

    // 4. Bottom-up call-tree overlay offset & stack depth (with recursion check)
    let mut state = vec![0u8; n]; // 0 = unvisited, 1 = visiting, 2 = done
    let mut fn_overlay_offset = vec![0u16; n];
    let mut fn_stack_depth = vec![0u16; n];

    fn dfs(
        u: usize,
        program: &IrProgram,
        callees: &[Vec<usize>],
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
        let mut max_child_stack = 0u16;
        for &v in &callees[u] {
            dfs(
                v,
                program,
                callees,
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
                &frame_sizes,
                &mut state,
                &mut fn_overlay_offset,
                &mut fn_stack_depth,
            )?;
        }
    }

    // 5. Assign final concrete addresses
    let mut slot_addrs = HashMap::new();
    let mut max_end = overlay_base;
    let mut max_stack = 0u16;
    for i in 0..n {
        if program.functions[i].is_test {
            continue;
        }
        let fn_base = overlay_base.wrapping_add(fn_overlay_offset[i]);
        for (slot_idx, &rel) in local_rel_offsets[i].iter().enumerate() {
            slot_addrs.insert((i as u32, slot_idx as SlotId), fn_base.wrapping_add(rel));
        }
        let fn_end = fn_base.wrapping_add(frame_sizes[i]);
        max_end = max_end.max(fn_end);
        max_stack = max_stack.max(fn_stack_depth[i] + 2);
    }

    Ok(RamLayout {
        ram_base,
        ram_end: max_end,
        global_addrs,
        slot_addrs,
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
}
