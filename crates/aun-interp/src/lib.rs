//! Host IR interpreter (`aun-interp`) for `aun test`.

use aun_ir::*;
use std::collections::HashMap;

/// Outcome of executing a single `#[test]` function in `aun-interp`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TestOutcome {
    Passed,
    Failed(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Trap {
    pub code: PanicCode,
    pub site_id: u16,
    pub message: String,
}

/// Host execution engine for `aun-ir`.
#[derive(Debug, Default)]
pub struct Interpreter {
    pub port_out_log: Vec<(u8, u8)>,
    pub port_in_values: HashMap<u8, u8>,
    pub transition_log: Vec<String>,
    pub check_overflow: bool,
    pub check_bounds: bool,
    pub check_transitions: bool,
}

impl Interpreter {
    pub fn new() -> Self {
        Self {
            port_out_log: Vec::new(),
            port_in_values: HashMap::new(),
            transition_log: Vec::new(),
            check_overflow: true,
            check_bounds: true,
            check_transitions: true,
        }
    }

    /// Runs all `#[test]` functions in `program` and returns `(name, outcome)` pairs.
    pub fn run_tests(&mut self, program: &IrProgram) -> Vec<(String, TestOutcome)> {
        let mut results = Vec::new();
        for func in &program.functions {
            if !func.is_test {
                continue;
            }
            let res = self.exec_function(program, func.id, &[]);
            let outcome = match (res, func.expected_panic) {
                (Ok(_), None) => TestOutcome::Passed,
                (Ok(_), Some(expected)) => TestOutcome::Failed(format!(
                    "expected panic `{expected:?}`, but test completed normally"
                )),
                (Err(trap), Some(expected)) if trap.code == expected => TestOutcome::Passed,
                (Err(trap), Some(expected)) => TestOutcome::Failed(format!(
                    "expected panic `{expected:?}`, got `{:?}` at site {}: {}",
                    trap.code, trap.site_id, trap.message
                )),
                (Err(trap), None) => TestOutcome::Failed(format!(
                    "panicked with `{:?}` at site {}: {}",
                    trap.code, trap.site_id, trap.message
                )),
            };
            results.push((func.name.clone(), outcome));
        }
        results
    }

    pub fn exec_function(
        &mut self,
        program: &IrProgram,
        func_id: u32,
        args: &[i64],
    ) -> Result<Option<i64>, Trap> {
        let mut mem = Vec::new();
        let mut global_addrs = Vec::with_capacity(program.globals.len());
        for g in &program.globals {
            let sz = program.type_size(&g.ty) as usize;
            if let Some(fa) = g.fixed_addr {
                let start = fa as usize;
                let end = start + sz.max(1);
                if mem.len() < end {
                    mem.resize(end, 0);
                }
                if g.init.len() == sz {
                    mem[start..start + sz].copy_from_slice(&g.init);
                }
                global_addrs.push(fa as u32);
            } else {
                let addr = mem.len() as u32;
                global_addrs.push(addr);
                if g.init.len() == sz {
                    mem.extend_from_slice(&g.init);
                } else {
                    mem.resize(mem.len() + sz, 0);
                }
            }
        }
        self.exec_frame(program, &mut mem, &global_addrs, func_id, args)
    }

    fn exec_frame(
        &mut self,
        program: &IrProgram,
        mem: &mut Vec<u8>,
        global_addrs: &[u32],
        func_id: u32,
        args: &[i64],
    ) -> Result<Option<i64>, Trap> {
        let func = &program.functions[func_id as usize];
        let frame_base = mem.len();
        let mut slot_addrs = Vec::with_capacity(func.locals.len());
        for slot in &func.locals {
            let addr = mem.len() as u32;
            slot_addrs.push(addr);
            let sz = program.type_size(&slot.ty) as usize;
            mem.resize(mem.len() + sz.max(1), 0);
        }

        // Write arguments into parameter slots
        for (i, &arg_val) in args.iter().enumerate() {
            if let Some(slot) = func.locals.get(i) {
                write_scalar(mem, slot_addrs[i], &slot.ty, arg_val);
            }
        }

        let bb_map: HashMap<BlockId, usize> = func
            .blocks
            .iter()
            .enumerate()
            .map(|(i, b)| (b.id, i))
            .collect();
        let mut vals: Vec<i64> = vec![0; func.next_val as usize];
        let mut cur_bb: BlockId = 0;
        let mut prev_bb: Option<BlockId> = None;

        let result = loop {
            let bb = &func.blocks[bb_map[&cur_bb]];

            // Evaluate all leading Phi nodes simultaneously using values from
            // before block entry.
            let mut phi_updates: Vec<(ValueId, i64)> = Vec::new();
            let mut inst_start = 0usize;
            for inst in &bb.insts {
                match inst {
                    IrInst::Phi { dst, incoming, .. } => {
                        let pred = prev_bb.ok_or_else(|| Trap {
                            code: PanicCode::AssertionFailed,
                            site_id: 0,
                            message: "phi executed in entry block without predecessor".into(),
                        })?;
                        let src_val = incoming
                            .iter()
                            .find(|&&(p, _)| p == pred)
                            .map(|&(_, v)| vals[v as usize])
                            .ok_or_else(|| Trap {
                                code: PanicCode::AssertionFailed,
                                site_id: 0,
                                message: format!("phi missing incoming value from bb{pred}"),
                            })?;
                        phi_updates.push((*dst, src_val));
                        inst_start += 1;
                    }
                    _ => break,
                }
            }
            for (dst, val) in phi_updates {
                vals[dst as usize] = val;
            }

            for inst in &bb.insts[inst_start..] {
                match inst {
                    IrInst::ConstInt { dst, val, .. } => vals[*dst as usize] = *val,
                    IrInst::ConstBool { dst, val } => {
                        vals[*dst as usize] = if *val { 1 } else { 0 };
                    }
                    IrInst::BinOp {
                        dst,
                        op,
                        mode,
                        ty,
                        lhs,
                        rhs,
                        site_id,
                    } => {
                        let a = vals[*lhs as usize];
                        let b = vals[*rhs as usize];
                        let res = self.eval_binop(*op, *mode, ty, a, b, *site_id)?;
                        vals[*dst as usize] = res;
                    }
                    IrInst::UnOp {
                        dst,
                        op,
                        mode,
                        ty,
                        val,
                        site_id,
                    } => {
                        let v = vals[*val as usize];
                        let res = match op {
                            IrUnOp::Not => match ty {
                                IrType::Bool => if v == 0 { 1 } else { 0 },
                                IrType::Int(it) => truncate_int(!v, *it),
                                _ => !v,
                            },
                            IrUnOp::Neg => {
                                let it = match ty {
                                    IrType::Int(it) | IrType::Fixed { base: it, .. } => *it,
                                    _ => IntType::I16,
                                };
                                self.apply_arith_range(-v, it, *mode, *site_id)?
                            }
                        };
                        vals[*dst as usize] = res;
                    }
                    IrInst::Cast {
                        dst,
                        val,
                        from_ty,
                        to_ty,
                    } => {
                        let v = vals[*val as usize];
                        vals[*dst as usize] = eval_cast(v, from_ty, to_ty);
                    }
                    IrInst::FixedInt {
                        dst,
                        val,
                        base,
                        frac_bits,
                    } => {
                        let v = vals[*val as usize];
                        vals[*dst as usize] = truncate_int(v >> *frac_bits, *base);
                    }
                    IrInst::LoadLocal { dst, slot, ty } => {
                        vals[*dst as usize] = read_scalar(mem, slot_addrs[*slot as usize], ty);
                    }
                    IrInst::StoreLocal { slot, val, ty } => {
                        write_scalar(mem, slot_addrs[*slot as usize], ty, vals[*val as usize]);
                    }
                    IrInst::AddrLocal { dst, slot } => {
                        vals[*dst as usize] = slot_addrs[*slot as usize] as i64;
                    }
                    IrInst::AddrGlobal { dst, global } => {
                        vals[*dst as usize] = global_addrs[*global as usize] as i64;
                    }
                    IrInst::FieldAddr {
                        dst,
                        base,
                        byte_offset,
                    } => {
                        vals[*dst as usize] = vals[*base as usize] + (*byte_offset as i64);
                    }
                    IrInst::IndexAddr {
                        dst,
                        base,
                        index,
                        elem_size,
                        len,
                        site_id,
                    } => {
                        let idx = vals[*index as usize];
                        if self.check_bounds && (idx < 0 || idx >= *len as i64) {
                            return Err(Trap {
                                code: PanicCode::Bounds,
                                site_id: *site_id,
                                message: format!("index {idx} out of bounds for length {len}"),
                            });
                        }
                        vals[*dst as usize] = vals[*base as usize] + idx * (*elem_size as i64);
                    }
                    IrInst::LoadPtr { dst, addr, ty } => {
                        vals[*dst as usize] = read_scalar(mem, vals[*addr as usize] as u32, ty);
                    }
                    IrInst::StorePtr { addr, val, ty } => {
                        write_scalar(mem, vals[*addr as usize] as u32, ty, vals[*val as usize]);
                    }
                    IrInst::StateTransition {
                        machine_addr,
                        machine_id,
                        from_state,
                        to_state,
                        field_vals,
                        site_id,
                    } => {
                        let maddr = vals[*machine_addr as usize] as u32;
                        let mdef = &program.machines[*machine_id as usize];
                        let tag_addr = maddr + mdef.tag_offset as u32;
                        let cur_state = mem[tag_addr as usize];

                        // Dynamic transition verification when `from_state` is unknown
                        if from_state.is_none()
                            && self.check_transitions
                            && !mdef.is_transition_allowed(cur_state, *to_state)
                        {
                            let cur_name = mdef
                                .states
                                .get(cur_state as usize)
                                .map(|s| s.name.as_str())
                                .unwrap_or("?");
                            let to_name = &mdef.states[*to_state as usize].name;
                            return Err(Trap {
                                code: PanicCode::InvalidTransition,
                                site_id: *site_id,
                                message: format!(
                                    "illegal state transition `{}: {cur_name} -> {to_name}`",
                                    mdef.name
                                ),
                            });
                        }

                        // Snapshot pre-evaluated payload field values before on_exit
                        let staged: Vec<(u16, i64, IrType)> = field_vals
                            .iter()
                            .map(|(off, vid, ty)| (*off, vals[*vid as usize], ty.clone()))
                            .collect();

                        // 1. Run current state's on_exit
                        if let Some(exit_fn) = mdef
                            .states
                            .get(cur_state as usize)
                            .and_then(|s| s.on_exit)
                        {
                            self.exec_frame(program, mem, global_addrs, exit_fn, &[maddr as i64])?;
                        }

                        // 2. Write target state fields into overlaid payload region
                        let payload_base = maddr + mdef.payload_offset as u32;
                        for (rel_off, fval, fty) in staged {
                            write_scalar(mem, payload_base + rel_off as u32, &fty, fval);
                        }

                        // 3. Update state tag
                        mem[tag_addr as usize] = *to_state;
                        mem[tag_addr as usize + 1] = 0;

                        let cur_name = mdef
                            .states
                            .get(cur_state as usize)
                            .map(|s| s.name.as_str())
                            .unwrap_or("?");
                        let to_name = &mdef.states[*to_state as usize].name;
                        self.transition_log
                            .push(format!("{}: {cur_name} -> {to_name}", mdef.name));

                        // 4. Run target state's on_enter
                        if let Some(enter_fn) = mdef.states[*to_state as usize].on_enter {
                            self.exec_frame(
                                program,
                                mem,
                                global_addrs,
                                enter_fn,
                                &[maddr as i64],
                            )?;
                        }
                    }
                    IrInst::StateDispatch {
                        dst,
                        machine_addr,
                        machine_id,
                        method,
                        args,
                    } => {
                        let maddr = vals[*machine_addr as usize] as u32;
                        let mdef = &program.machines[*machine_id as usize];
                        let cur_state = mem[(maddr + mdef.tag_offset as u32) as usize];
                        if let Some(&handler_id) =
                            mdef.states[cur_state as usize].methods.get(method)
                        {
                            let mut call_args = Vec::with_capacity(args.len() + 1);
                            call_args.push(maddr as i64);
                            for a in args {
                                call_args.push(vals[*a as usize]);
                            }
                            let ret = self.exec_frame(
                                program,
                                mem,
                                global_addrs,
                                handler_id,
                                &call_args,
                            )?;
                            if let (Some(d), Some(rv)) = (dst, ret) {
                                vals[*d as usize] = rv;
                            }
                        }
                    }
                    IrInst::StateTag {
                        dst,
                        machine_addr,
                        machine_id,
                    } => {
                        let maddr = vals[*machine_addr as usize] as u32;
                        let mdef = &program.machines[*machine_id as usize];
                        vals[*dst as usize] = mem[(maddr + mdef.tag_offset as u32) as usize] as i64;
                    }
                    IrInst::PoolAlloc {
                        dst,
                        pool_addr,
                        init_addr,
                        elem_size,
                        capacity,
                    } => {
                        let paddr = vals[*pool_addr as usize] as usize;
                        let iaddr = vals[*init_addr as usize] as usize;
                        let esz = *elem_size as usize;
                        let cap = *capacity as usize;
                        let bitmap_base = paddr + esz * cap;
                        let mut found = 0xFFu8;
                        for idx in 0..cap {
                            let byte_i = bitmap_base + (idx / 8);
                            let bit = 1u8 << (idx % 8);
                            if (mem[byte_i] & bit) == 0 {
                                mem[byte_i] |= bit;
                                let dst_elem = paddr + idx * esz;
                                for b in 0..esz {
                                    mem[dst_elem + b] = mem[iaddr + b];
                                }
                                found = idx as u8;
                                break;
                            }
                        }
                        vals[*dst as usize] = found as i64;
                    }
                    IrInst::PoolFree {
                        pool_addr,
                        handle,
                        elem_size,
                        capacity,
                        site_id,
                    } => {
                        let paddr = vals[*pool_addr as usize] as usize;
                        let h = vals[*handle as usize];
                        let cap = *capacity as i64;
                        let bitmap_base = paddr + (*elem_size as usize) * (*capacity as usize);
                        if h < 0 || h >= cap {
                            if self.check_bounds {
                                return Err(Trap {
                                    code: PanicCode::InvalidHandle,
                                    site_id: *site_id,
                                    message: format!("invalid pool handle {h} on free"),
                                });
                            }
                        } else {
                            let idx = h as usize;
                            let byte_i = bitmap_base + (idx / 8);
                            let bit = 1u8 << (idx % 8);
                            if self.check_bounds && (mem[byte_i] & bit) == 0 {
                                return Err(Trap {
                                    code: PanicCode::InvalidHandle,
                                    site_id: *site_id,
                                    message: format!("double-free of pool handle {h}"),
                                });
                            }
                            mem[byte_i] &= !bit;
                        }
                    }
                    IrInst::PoolElemAddr {
                        dst,
                        pool_addr,
                        handle,
                        elem_size,
                        capacity,
                        site_id,
                    } => {
                        let paddr = vals[*pool_addr as usize] as usize;
                        let h = vals[*handle as usize];
                        let cap = *capacity as i64;
                        let bitmap_base = paddr + (*elem_size as usize) * (*capacity as usize);
                        if h < 0 || h >= cap {
                            if self.check_bounds {
                                return Err(Trap {
                                    code: PanicCode::InvalidHandle,
                                    site_id: *site_id,
                                    message: format!("pool handle {h} out of range 0..{cap}"),
                                });
                            }
                        } else {
                            let idx = h as usize;
                            let byte_i = bitmap_base + (idx / 8);
                            let bit = 1u8 << (idx % 8);
                            if self.check_bounds && (mem[byte_i] & bit) == 0 {
                                return Err(Trap {
                                    code: PanicCode::InvalidHandle,
                                    site_id: *site_id,
                                    message: format!("use-after-free on pool handle {h}"),
                                });
                            }
                            vals[*dst as usize] = (paddr + idx * (*elem_size as usize)) as i64;
                        }
                    }
                    IrInst::Call { dst, func: callee, args } => {
                        let arg_vals: Vec<i64> = args.iter().map(|v| vals[*v as usize]).collect();
                        let ret = self.exec_frame(program, mem, global_addrs, *callee, &arg_vals)?;
                        if let (Some(d), Some(rv)) = (dst, ret) {
                            vals[*d as usize] = rv;
                        }
                    }
                    IrInst::ExternCall { dst, .. } => {
                        // Unmocked externs return 0 in Phase 1 skeleton
                        if let Some(d) = dst {
                            vals[*d as usize] = 0;
                        }
                    }
                    IrInst::PortOut { port, val } => {
                        let p = (vals[*port as usize] & 0xFF) as u8;
                        let v = (vals[*val as usize] & 0xFF) as u8;
                        self.port_out_log.push((p, v));
                    }
                    IrInst::PortIn { dst, port } => {
                        let p = (vals[*port as usize] & 0xFF) as u8;
                        vals[*dst as usize] = *self.port_in_values.get(&p).unwrap_or(&0) as i64;
                    }
                    IrInst::Assert { cond, site_id } => {
                        if vals[*cond as usize] == 0 {
                            return Err(Trap {
                                code: PanicCode::AssertionFailed,
                                site_id: *site_id,
                                message: "assertion failed".into(),
                            });
                        }
                    }
                    IrInst::InlineAsm { .. }
                    | IrInst::BankEnter { .. }
                    | IrInst::BankExit { .. } => {}
                    IrInst::Panic { code, site_id } => {
                        return Err(Trap {
                            code: *code,
                            site_id: *site_id,
                            message: format!("explicit panic `{code:?}`"),
                        });
                    }
                    IrInst::Phi { .. } => {
                        return Err(Trap {
                            code: PanicCode::AssertionFailed,
                            site_id: 0,
                            message: "non-leading phi instruction in basic block".into(),
                        });
                    }
                }
            }

            match bb.term {
                Terminator::Return(v) => break Ok(v.map(|id| vals[id as usize])),
                Terminator::Goto(next) => {
                    prev_bb = Some(cur_bb);
                    cur_bb = next;
                }
                Terminator::Branch {
                    cond,
                    then_bb,
                    else_bb,
                } => {
                    prev_bb = Some(cur_bb);
                    cur_bb = if vals[cond as usize] != 0 {
                        then_bb
                    } else {
                        else_bb
                    };
                }
                Terminator::Unreachable => break Ok(None),
            }
        };

        mem.truncate(frame_base);
        result
    }

    fn eval_binop(
        &self,
        op: IrBinOp,
        mode: ArithMode,
        ty: &IrType,
        a: i64,
        b: i64,
        site_id: u16,
    ) -> Result<i64, Trap> {
        // Comparisons
        match op {
            IrBinOp::Eq => return Ok(if a == b { 1 } else { 0 }),
            IrBinOp::Ne => return Ok(if a != b { 1 } else { 0 }),
            IrBinOp::Lt => return Ok(if a < b { 1 } else { 0 }),
            IrBinOp::Le => return Ok(if a <= b { 1 } else { 0 }),
            IrBinOp::Gt => return Ok(if a > b { 1 } else { 0 }),
            IrBinOp::Ge => return Ok(if a >= b { 1 } else { 0 }),
            _ => {}
        }

        let (it, frac_bits) = match ty {
            IrType::Int(it) => (*it, 0u8),
            IrType::Fixed { base, frac_bits } => (*base, *frac_bits),
            IrType::Bool => (IntType::U8, 0),
            _ => (IntType::I16, 0),
        };

        let raw: i128 = match op {
            IrBinOp::Add => (a as i128) + (b as i128),
            IrBinOp::Sub => (a as i128) - (b as i128),
            IrBinOp::Mul => ((a as i128) * (b as i128)) >> frac_bits,
            IrBinOp::Div => {
                if b == 0 {
                    return Err(Trap {
                        code: PanicCode::DivByZero,
                        site_id,
                        message: "division by zero".into(),
                    });
                }
                ((a as i128) << frac_bits) / (b as i128)
            }
            IrBinOp::Rem => {
                if b == 0 {
                    return Err(Trap {
                        code: PanicCode::DivByZero,
                        site_id,
                        message: "remainder by zero".into(),
                    });
                }
                (a as i128) % (b as i128)
            }
            IrBinOp::BitAnd => return Ok(truncate_int(a & b, it)),
            IrBinOp::BitOr => return Ok(truncate_int(a | b, it)),
            IrBinOp::BitXor => return Ok(truncate_int(a ^ b, it)),
            IrBinOp::Shl => {
                if self.check_overflow && (b < 0 || b >= it.bit_width() as i64) {
                    return Err(Trap {
                        code: PanicCode::Overflow,
                        site_id,
                        message: format!("shift left by {b} overflows `{it}`"),
                    });
                }
                return Ok(truncate_int(a << (b & 31), it));
            }
            IrBinOp::Shr => {
                if self.check_overflow && (b < 0 || b >= it.bit_width() as i64) {
                    return Err(Trap {
                        code: PanicCode::Overflow,
                        site_id,
                        message: format!("shift right by {b} overflows `{it}`"),
                    });
                }
                return Ok(truncate_int(a >> (b & 31), it));
            }
            _ => unreachable!(),
        };

        self.apply_arith_range_i128(raw, it, mode, site_id)
    }

    fn apply_arith_range(
        &self,
        val: i64,
        it: IntType,
        mode: ArithMode,
        site_id: u16,
    ) -> Result<i64, Trap> {
        self.apply_arith_range_i128(val as i128, it, mode, site_id)
    }

    fn apply_arith_range_i128(
        &self,
        val: i128,
        it: IntType,
        mode: ArithMode,
        site_id: u16,
    ) -> Result<i64, Trap> {
        let min = it.min_i64() as i128;
        let max = it.max_i64() as i128;
        match mode {
            ArithMode::Saturating => Ok(val.clamp(min, max) as i64),
            ArithMode::Wrapping => Ok(truncate_int(val as i64, it)),
            ArithMode::Checked => {
                if self.check_overflow && (val < min || val > max) {
                    return Err(Trap {
                        code: PanicCode::Overflow,
                        site_id,
                        message: format!("arithmetic overflow: {val} out of range for `{it}`"),
                    });
                }
                Ok(truncate_int(val as i64, it))
            }
        }
    }
}

fn truncate_int(val: i64, it: IntType) -> i64 {
    match it {
        IntType::U8 => (val as u8) as i64,
        IntType::I8 => (val as i8) as i64,
        IntType::U16 => (val as u16) as i64,
        IntType::I16 => (val as i16) as i64,
        IntType::U32 => (val as u32) as i64,
        IntType::I32 => (val as i32) as i64,
    }
}

fn eval_cast(val: i64, from_ty: &IrType, to_ty: &IrType) -> i64 {
    match (from_ty, to_ty) {
        (IrType::Int(_), IrType::Fixed { base, frac_bits }) => {
            truncate_int(val << *frac_bits, *base)
        }
        (IrType::Fixed { frac_bits, .. }, IrType::Int(to_int)) => {
            truncate_int(val >> *frac_bits, *to_int)
        }
        (
            IrType::Fixed { frac_bits: f1, .. },
            IrType::Fixed {
                base: b2,
                frac_bits: f2,
            },
        ) => {
            let shifted = if f2 >= f1 {
                val << (f2 - f1)
            } else {
                val >> (f1 - f2)
            };
            truncate_int(shifted, *b2)
        }
        (_, IrType::Int(to_int)) => truncate_int(val, *to_int),
        (_, IrType::Bool) => if val != 0 { 1 } else { 0 },
        _ => val,
    }
}

fn read_scalar(mem: &[u8], addr: u32, ty: &IrType) -> i64 {
    let a = addr as usize;
    let byte = |off: usize| mem.get(a + off).copied().unwrap_or(0);
    match ty {
        IrType::Bool => byte(0) as i64,
        IrType::Int(IntType::U8) | IrType::Handle(_) => byte(0) as i64,
        IrType::Int(IntType::I8) => (byte(0) as i8) as i64,
        IrType::Int(IntType::U16) | IrType::Ref { .. } => {
            u16::from_le_bytes([byte(0), byte(1)]) as i64
        }
        IrType::Int(IntType::I16) => i16::from_le_bytes([byte(0), byte(1)]) as i64,
        IrType::Int(IntType::U32) => {
            u32::from_le_bytes([byte(0), byte(1), byte(2), byte(3)]) as i64
        }
        IrType::Int(IntType::I32) => {
            i32::from_le_bytes([byte(0), byte(1), byte(2), byte(3)]) as i64
        }
        IrType::Fixed { base, .. } => read_scalar(mem, addr, &IrType::Int(*base)),
        _ => 0,
    }
}

fn write_scalar(mem: &mut Vec<u8>, addr: u32, ty: &IrType, val: i64) {
    let a = addr as usize;
    let ensure_len = |mem: &mut Vec<u8>, needed: usize| {
        if mem.len() < a + needed {
            mem.resize(a + needed, 0);
        }
    };
    match ty {
        IrType::Bool | IrType::Int(IntType::U8 | IntType::I8) | IrType::Handle(_) => {
            ensure_len(mem, 1);
            mem[a] = (val & 0xFF) as u8;
        }
        IrType::Int(IntType::U16 | IntType::I16) | IrType::Ref { .. } => {
            ensure_len(mem, 2);
            let b = (val as u16).to_le_bytes();
            mem[a] = b[0];
            mem[a + 1] = b[1];
        }
        IrType::Int(IntType::U32 | IntType::I32) => {
            ensure_len(mem, 4);
            let b = (val as u32).to_le_bytes();
            mem[a..a + 4].copy_from_slice(&b);
        }
        IrType::Fixed { base, .. } => write_scalar(mem, addr, &IrType::Int(*base), val),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2e_fixed_point_saturating_wrapping_and_borrows() {
        let src = r#"
            struct Ball {
                x: q8_8,
                vx: q8_8,
            }

            fn step_ball(b: &mut Ball) {
                b.x = b.x + b.vx;
            }

            #[test]
            fn test_physics_and_ops() {
                let mut b: Ball = Ball { x: 10.5q8_8, vx: 1.5q8_8 };
                step_ball(&mut b);
                assert!(b.x.int() == 12i16);

                // Saturating vs wrapping u8
                let a: u8 = 250u8;
                assert!((a +| 20u8) == 255u8);
                assert!((a +% 10u8) == 4u8);

                // Loop + array + port I/O
                let mut arr: [u8; 4] = [0u8; 4];
                for i in 0u8..4u8 {
                    arr[i] = i * 2u8;
                }
                assert!(arr[3] == 6u8);
                io_out(0xFE, arr[3]);
            }

            #[test]
            #[should_panic(Overflow)]
            fn test_checked_overflow_traps() {
                let x: u8 = 250u8;
                let _y: u8 = x + 10u8;
            }

            #[test]
            #[should_panic(Bounds)]
            fn test_array_bounds_traps() {
                let arr: [u8; 2] = [1u8, 2u8];
                let i: u8 = 2u8;
                let _v: u8 = arr[i];
            }
        "#;

        let program = aun_front::compile_str(0, src).expect("compile should succeed");
        let mut interp = Interpreter::new();
        let outcomes = interp.run_tests(&program);
        assert_eq!(outcomes.len(), 3);
        for (name, outcome) in &outcomes {
            assert_eq!(outcome, &TestOutcome::Passed, "test `{name}` failed");
        }
        assert_eq!(interp.port_out_log, vec![(0xFE, 6)]);
        assert_translation_valid(src);
    }

    #[test]
    fn e2e_phase2_state_machine_and_pool() {
        let src = r#"
            struct Input { plunger: u8 }

            machine Ball {
                x: q8_8,

                state Inactive;
                state InShooter { power: u8 };
                state InPlay { vx: q8_8, timer: u16 };
                state MagnetHeld { hold: u8 };

                transitions {
                    Inactive   -> InShooter;
                    InShooter  -> InPlay | Inactive;
                    InPlay     -> MagnetHeld | Inactive;
                    MagnetHeld -> InPlay | Inactive;
                }

                on InShooter {
                    fn on_enter(&mut self) {
                        self.x = 1.0q8_8;
                    }
                    fn step(&mut self, inp: &Input) {
                        if inp.plunger == 0u8 && self.power > 0u8 {
                            goto InPlay {
                                vx: 2.5q8_8,
                                timer: 120u16,
                            };
                        }
                    }
                }

                on InPlay {
                    fn step(&mut self, inp: &Input) {
                        self.x = self.x + self.vx;
                    }
                }
            }

            #[layout(soa, align(256))]
            static mut BALLS: Pool<Ball, 4>;

            #[test]
            fn test_ball_lifecycle_and_pool() {
                let h: Handle<Ball> = BALLS.alloc(Ball::new());
                let b: &mut Ball = BALLS.get_mut(h);
                assert!(is_state!(b, Ball::Inactive));

                b.goto(InShooter { power: 5u8 });
                assert!(is_state!(b, Ball::InShooter));
                assert!(b.x.int() == 1i16);

                let inp: Input = Input { plunger: 0u8 };
                b.step(&inp);
                assert!(is_state!(b, Ball::InPlay));

                b.step(&inp);
                assert!(b.x.int() == 3i16);

                BALLS.free(h);
            }

            #[test]
            #[should_panic(InvalidTransition)]
            fn test_dynamic_illegal_transition_traps() {
                let mut b: Ball = Ball::new();
                b.goto(MagnetHeld { hold: 10u8 });
            }

            #[test]
            #[should_panic(InvalidHandle)]
            fn test_pool_use_after_free_traps() {
                let h: Handle<Ball> = BALLS.alloc(Ball::new());
                BALLS.free(h);
                let _dead: &Ball = BALLS.get(h);
            }
        "#;

        let program = aun_front::compile_str(0, src).expect("Phase 2 compile should succeed");
        // Verify Uchiguri RAM overlay: InPlay (4B) and MagnetHeld (1B) share payload!
        let mdef = &program.machines[0];
        assert_eq!(mdef.total_size, 2 /* x */ + 2 /* tag */ + 4 /* max(0,1,4,1) */);

        let mut interp = Interpreter::new();
        let outcomes = interp.run_tests(&program);
        assert_eq!(outcomes.len(), 3);
        for (name, outcome) in &outcomes {
            assert_eq!(outcome, &TestOutcome::Passed, "test `{name}` failed");
        }
        assert_eq!(
            interp.transition_log,
            vec![
                "Ball: Inactive -> InShooter",
                "Ball: InShooter -> InPlay",
            ]
        );
        assert_translation_valid(src);
    }

    fn assert_translation_valid(src: &str) {
        let prog_raw = aun_front::compile_str(0, src).expect("compile should succeed");
        aun_ir::verify::verify_program(&prog_raw).expect("raw MIR must verify");

        let mut interp_raw = Interpreter::new();
        let outcomes_raw = interp_raw.run_tests(&prog_raw);

        // 1. After mem2reg (SSA with Phi nodes)
        let mut prog_ssa = prog_raw.clone();
        for f in &mut prog_ssa.functions {
            aun_ir::ssa::mem2reg(f);
        }
        aun_ir::verify::verify_program(&prog_ssa).expect("SSA MIR after mem2reg must verify");
        let mut interp_ssa = Interpreter::new();
        let outcomes_ssa = interp_ssa.run_tests(&prog_ssa);
        assert_eq!(outcomes_ssa, outcomes_raw, "mem2reg changed test outcomes");
        assert_eq!(
            interp_ssa.port_out_log, interp_raw.port_out_log,
            "mem2reg changed port_out_log"
        );

        // 2. After destruct_ssa (Phi elimination)
        let mut prog_destruct = prog_ssa.clone();
        for f in &mut prog_destruct.functions {
            aun_ir::ssa::destruct_ssa(f);
        }
        aun_ir::verify::verify_program(&prog_destruct).expect("MIR after destruct_ssa must verify");
        for f in &prog_destruct.functions {
            assert!(
                !f.blocks
                    .iter()
                    .any(|b| b.insts.iter().any(|i| matches!(i, IrInst::Phi { .. }))),
                "fn `{}` still contains Phi after destruct_ssa",
                f.name
            );
        }
        let mut interp_destruct = Interpreter::new();
        let outcomes_destruct = interp_destruct.run_tests(&prog_destruct);
        assert_eq!(
            outcomes_destruct, outcomes_raw,
            "destruct_ssa changed test outcomes"
        );
        assert_eq!(
            interp_destruct.port_out_log, interp_raw.port_out_log,
            "destruct_ssa changed port_out_log"
        );

        // 3. After existing Layer 2 MIR optimizer
        let mut prog_opt = prog_raw.clone();
        aun_ir::opt::optimize_program(&mut prog_opt);
        aun_ir::verify::verify_program(&prog_opt).expect("MIR after optimize_program must verify");
        let mut interp_opt = Interpreter::new();
        let outcomes_opt = interp_opt.run_tests(&prog_opt);
        assert_eq!(
            outcomes_opt, outcomes_raw,
            "optimize_program changed test outcomes"
        );
        assert_eq!(
            interp_opt.port_out_log, interp_raw.port_out_log,
            "optimize_program changed port_out_log"
        );

        // 4. After SSA + SCCP + GVN + LICM + OSR + DCE (both with Phi and after destruct_ssa)
        let mut prog_ssa_opt = prog_ssa.clone();
        for f in &mut prog_ssa_opt.functions {
            for _ in 0..4 {
                let mut changed = false;
                changed |= aun_ir::sccp::sccp(f);
                changed |= aun_ir::gvn::gvn(f);
                changed |= aun_ir::licm::licm(f);
                changed |= aun_ir::osr::osr(f);
                changed |= aun_ir::opt::dce(f);
                if !changed {
                    break;
                }
            }
        }
        aun_ir::verify::verify_program(&prog_ssa_opt)
            .expect("MIR after SSA+SCCP+GVN+LICM+OSR must verify");
        let mut interp_sg = Interpreter::new();
        let outcomes_sg = interp_sg.run_tests(&prog_ssa_opt);
        assert_eq!(
            outcomes_sg, outcomes_raw,
            "SCCP+GVN+LICM+OSR changed test outcomes"
        );
        assert_eq!(
            interp_sg.port_out_log, interp_raw.port_out_log,
            "SCCP+GVN+LICM+OSR changed port_out_log"
        );

        let mut prog_sg_destruct = prog_ssa_opt;
        for f in &mut prog_sg_destruct.functions {
            aun_ir::ssa::destruct_ssa(f);
        }
        aun_ir::verify::verify_program(&prog_sg_destruct)
            .expect("MIR after SSA opts + destruct_ssa must verify");
        for f in &prog_sg_destruct.functions {
            assert!(
                !f.blocks
                    .iter()
                    .any(|b| b.insts.iter().any(|i| matches!(i, IrInst::Phi { .. }))),
                "fn `{}` still contains Phi after destruct_ssa",
                f.name
            );
        }
        let mut interp_sgd = Interpreter::new();
        let outcomes_sgd = interp_sgd.run_tests(&prog_sg_destruct);
        assert_eq!(
            outcomes_sgd, outcomes_raw,
            "SCCP+GVN+LICM+OSR+destruct_ssa changed test outcomes"
        );
        assert_eq!(
            interp_sgd.port_out_log, interp_raw.port_out_log,
            "SCCP+GVN+LICM+OSR+destruct_ssa changed port_out_log"
        );

        // 5. Combined `optimize_function_ssa` + `optimize_function`
        let mut prog_full_ssa = prog_raw.clone();
        for f in &mut prog_full_ssa.functions {
            aun_ir::opt::optimize_function_ssa(f);
            aun_ir::opt::optimize_function(f);
        }
        aun_ir::verify::verify_program(&prog_full_ssa)
            .expect("MIR after optimize_function_ssa + optimize_function must verify");
        let mut interp_full_ssa = Interpreter::new();
        let outcomes_full_ssa = interp_full_ssa.run_tests(&prog_full_ssa);
        assert_eq!(
            outcomes_full_ssa, outcomes_raw,
            "optimize_function_ssa changed test outcomes"
        );
        assert_eq!(
            interp_full_ssa.port_out_log, interp_raw.port_out_log,
            "optimize_function_ssa changed port_out_log"
        );
    }

    #[test]
    fn formal_exhaustive_8bit_sccp_gvn_osr_equivalence() {
        use aun_ir::sccp::{eval_binop as sccp_eval_binop, ConstVal};

        let interp = Interpreter::new();
        let modes = [ArithMode::Wrapping, ArithMode::Saturating, ArithMode::Checked];
        let full_ops = [
            IrBinOp::Add,
            IrBinOp::Sub,
            IrBinOp::Mul,
            IrBinOp::BitAnd,
            IrBinOp::BitOr,
            IrBinOp::BitXor,
            IrBinOp::Eq,
            IrBinOp::Ne,
            IrBinOp::Lt,
            IrBinOp::Le,
            IrBinOp::Gt,
            IrBinOp::Ge,
        ];

        // 1. Exhaustive 8-bit equivalence of SCCP constant evaluation vs Interpreter semantics.
        for it in [IntType::U8, IntType::I8] {
            let ty = IrType::Int(it);
            let range = it.min_i64()..=it.max_i64();
            for &mode in &modes {
                for a in range.clone() {
                    for b in range.clone() {
                        let cv_a = ConstVal::Int(it, a);
                        let cv_b = ConstVal::Int(it, b);
                        for &op in &full_ops {
                            let expected = interp.eval_binop(op, mode, &ty, a, b, 0);
                            let actual = sccp_eval_binop(op, mode, &ty, cv_a, cv_b);
                            match (expected, actual) {
                                (Ok(ev), Some(ConstVal::Int(rt, av))) => {
                                    assert_eq!(rt, it);
                                    assert_eq!(av, ev, "mismatch {it} {mode:?} {op:?}({a},{b})");
                                }
                                (Ok(ev), Some(ConstVal::Bool(ab))) => {
                                    assert_eq!(
                                        if ab { 1 } else { 0 },
                                        ev,
                                        "cmp mismatch {it} {op:?}({a},{b})"
                                    );
                                }
                                (Err(_), None) => {}
                                (e, act) => panic!(
                                    "trap/fold divergence on {it} {mode:?} {op:?}({a},{b}): interp={e:?}, sccp={act:?}"
                                ),
                            }
                        }
                        if b != 0 {
                            for &op in &[IrBinOp::Div, IrBinOp::Rem] {
                                let expected = interp.eval_binop(op, mode, &ty, a, b, 0);
                                let actual = sccp_eval_binop(op, mode, &ty, cv_a, cv_b);
                                match (expected, actual) {
                                    (Ok(ev), Some(ConstVal::Int(rt, av))) => {
                                        assert_eq!(rt, it);
                                        assert_eq!(av, ev);
                                    }
                                    (Err(_), None) => {}
                                    (e, act) => panic!(
                                        "div/rem divergence on {it} {mode:?} {op:?}({a},{b}): interp={e:?}, sccp={act:?}"
                                    ),
                                }
                            }
                        }
                    }
                    for sh in 0..8i64 {
                        let cv_a = ConstVal::Int(it, a);
                        let cv_s = ConstVal::Int(it, sh);
                        for &op in &[IrBinOp::Shl, IrBinOp::Shr] {
                            let ev = interp.eval_binop(op, mode, &ty, a, sh, 0).unwrap();
                            let Some(ConstVal::Int(rt, av)) =
                                sccp_eval_binop(op, mode, &ty, cv_a, cv_s)
                            else {
                                panic!("expected const fold for valid 8-bit shift {op:?}({a}, {sh})");
                            };
                            assert_eq!(rt, it);
                            assert_eq!(av, ev);
                        }
                    }
                }
            }

            // 2. Exhaustive 8-bit algebraic identities used by GVN.
            for x in it.min_i64()..=it.max_i64() {
                for &mode in &[ArithMode::Wrapping, ArithMode::Saturating, ArithMode::Checked] {
                    assert_eq!(interp.eval_binop(IrBinOp::Add, mode, &ty, x, 0, 0).unwrap(), x);
                    assert_eq!(interp.eval_binop(IrBinOp::Sub, mode, &ty, x, 0, 0).unwrap(), x);
                    assert_eq!(interp.eval_binop(IrBinOp::Mul, mode, &ty, x, 1, 0).unwrap(), x);
                    assert_eq!(interp.eval_binop(IrBinOp::Mul, mode, &ty, x, 0, 0).unwrap(), 0);
                    assert_eq!(interp.eval_binop(IrBinOp::Sub, mode, &ty, x, x, 0).unwrap(), 0);
                }
                assert_eq!(interp.eval_binop(IrBinOp::BitAnd, ArithMode::Wrapping, &ty, x, 0, 0).unwrap(), 0);
                assert_eq!(interp.eval_binop(IrBinOp::BitAnd, ArithMode::Wrapping, &ty, x, x, 0).unwrap(), x);
                assert_eq!(interp.eval_binop(IrBinOp::BitOr, ArithMode::Wrapping, &ty, x, 0, 0).unwrap(), x);
                assert_eq!(interp.eval_binop(IrBinOp::BitOr, ArithMode::Wrapping, &ty, x, x, 0).unwrap(), x);
                assert_eq!(interp.eval_binop(IrBinOp::BitXor, ArithMode::Wrapping, &ty, x, 0, 0).unwrap(), x);
                assert_eq!(interp.eval_binop(IrBinOp::BitXor, ArithMode::Wrapping, &ty, x, x, 0).unwrap(), 0);
                assert_eq!(interp.eval_binop(IrBinOp::Shl, ArithMode::Wrapping, &ty, x, 0, 0).unwrap(), x);
                assert_eq!(interp.eval_binop(IrBinOp::Shr, ArithMode::Wrapping, &ty, x, 0, 0).unwrap(), x);
                assert_eq!(interp.eval_binop(IrBinOp::Eq, ArithMode::Checked, &ty, x, x, 0).unwrap(), 1);
                assert_eq!(interp.eval_binop(IrBinOp::Ne, ArithMode::Checked, &ty, x, x, 0).unwrap(), 0);
                assert_eq!(interp.eval_binop(IrBinOp::Lt, ArithMode::Checked, &ty, x, x, 0).unwrap(), 0);
                assert_eq!(interp.eval_binop(IrBinOp::Le, ArithMode::Checked, &ty, x, x, 0).unwrap(), 1);
            }
        }

        // 3. OSR induction-variable ring homomorphism for all 8-bit init values.
        let steps: [u8; 5] = [1, 2, 3, 255, 253];
        let factors: [u8; 7] = [2, 3, 5, 6, 7, 16, 51];
        for init in 0u8..=255 {
            for &step in &steps {
                for &k in &factors {
                    let mut iv = init;
                    let mut sr = init.wrapping_mul(k);
                    let sr_step = step.wrapping_mul(k);
                    for _iter in 0..=16 {
                        assert_eq!(sr, iv.wrapping_mul(k));
                        assert_eq!(sr as i8, (iv as i8).wrapping_mul(k as i8));
                        iv = iv.wrapping_add(step);
                        sr = sr.wrapping_add(sr_step);
                    }
                }
            }
        }
    }

    #[test]
    fn translation_validation_ssa_and_opt() {
        let src = r#"
            fn mix_params(a: u8, b: u8, c: bool) -> u8 {
                let mut res: u8 = a +% b;
                if c {
                    res = res +% 10u8;
                }
                if a == 0u8 {
                    return res;
                }
                res +% a
            }

            #[test]
            fn test_loop_swap_and_critical_edge() {
                let mut a: u8 = 1u8;
                let mut b: u8 = 2u8;
                let mut i: u8 = 0u8;
                while i < 5u8 {
                    let t: u8 = a;
                    a = b;
                    b = t +% 1u8;
                    if (i & 1u8) == 0u8 {
                        a = a +% 3u8;
                    }
                    i = i +% 1u8;
                }
                io_out(0x10, a);
                io_out(0x11, b);
                io_out(0x12, mix_params(4u8, 5u8, true));
                io_out(0x13, mix_params(0u8, 7u8, false));

                // Escaped mutable borrow alongside promotable local
                let mut escaped: u8 = 10u8;
                let mut acc: u8 = 0u8;
                let p: &mut u8 = &mut escaped;
                *p = *p +% 5u8;
                acc = acc +% *p;
                io_out(0x14, acc);

                // Loop-invariant arithmetic and induction-variable Mul/Shl strength reduction
                let base_k: u8 = mix_params(2u8, 3u8, false);
                let mut sum: u8 = 0u8;
                let mut j: u8 = 0u8;
                while j < 6u8 {
                    let inv: u8 = base_k +% 7u8;
                    let scaled: u8 = j *% 3u8;
                    let shifted: u8 = j << 2u8;
                    sum = sum +% inv +% scaled +% shifted;
                    j = j +% 1u8;
                }
                io_out(0x15, sum);
            }
        "#;
        assert_translation_valid(src);
    }

    #[test]
    fn mem_write_and_fixed_addr_static_interp() {
        let src = r#"
            struct State {
                x: u8,
                y: u16,
            }

            static mut G_STATE: State @ 0xEC00;

            #[test]
            fn test_mmio_and_fixed_static() {
                G_STATE.x = 42u8;
                G_STATE.y = 0x1234u16;
                assert!(mem_read_u8(0xEC00) == 42u8);
                assert!(mem_read_u16(0xEC01) == 0x1234u16);

                mem_write_u8(0xF3C8, 0x5Au8);
                mem_write_u16(0xF3CA, 0xBEEFu16);
                assert!(mem_read_u8(0xF3C8) == 0x5Au8);
                assert!(mem_read_u16(0xF3CA) == 0xBEEFu16);
            }
        "#;
        let program = aun_front::compile_str(0, src).expect("compile should succeed");
        let mut interp = Interpreter::new();
        let outcomes = interp.run_tests(&program);
        assert_eq!(outcomes.len(), 1);
        assert_eq!(outcomes[0].1, TestOutcome::Passed);
    }
}
