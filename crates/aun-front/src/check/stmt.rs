use super::*;

impl<'a> FnLowerer<'a> {
    pub(super) fn lower_fn(&mut self, f: &FnDecl) {
        let ret_ty = self.func().ret_ty.clone();
        let tail = self.lower_block(&f.body, Some(&ret_ty));
        if !self.is_terminated() {
            match tail {
                Some((val, ty)) => {
                    if ret_ty != IrType::Unit && ty != ret_ty && ty != IrType::Never {
                        self.checker.error(
                            format!("function `{}` returns `{ty:?}`, expected `{ret_ty:?}`", f.name),
                            f.span,
                        );
                    }
                    if ret_ty == IrType::Unit {
                        self.set_term(Terminator::Return(None));
                    } else {
                        self.set_term(Terminator::Return(Some(val)));
                    }
                }
                None => {
                    if ret_ty != IrType::Unit {
                        self.checker.error(
                            format!("function `{}` must return a value of type `{ret_ty:?}`", f.name),
                            f.span,
                        );
                    }
                    self.set_term(Terminator::Return(None));
                }
            }
        }
    }

    pub(super) fn lower_block(&mut self, block: &Block, expected: Option<&IrType>) -> Option<(ValueId, IrType)> {
        self.scopes.push(HashMap::new());
        for stmt in &block.stmts {
            if self.is_terminated() {
                break;
            }
            self.lower_stmt(stmt);
        }
        let res = if !self.is_terminated() {
            block
                .tail
                .as_ref()
                .and_then(|e| self.lower_expr(e, expected))
        } else {
            None
        };
        self.scopes.pop();
        res
    }

    fn lower_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let {
                mutable,
                name,
                ty,
                init,
                span,
            } => {
                let declared = ty
                    .as_ref()
                    .map(|t| self.checker.resolve_type(t, *span, true));
                let direct_agg_ty = if let Some(ref exp) = declared {
                    if !exp.is_scalar() {
                        Some(exp.clone())
                    } else {
                        None
                    }
                } else if let ExprKind::StructLit { name: sname, .. } = &init.kind {
                    self.checker
                        .struct_ids
                        .get(sname)
                        .map(|&sid| IrType::Struct(sid))
                } else {
                    None
                };
                if let Some(agg_ty) = direct_agg_ty {
                    let slot = self.alloc_local_slot(name.clone(), agg_ty.clone(), *mutable);
                    let dst_addr = self.fresh_val();
                    self.emit(IrInst::AddrLocal {
                        dst: dst_addr,
                        slot,
                    });
                    if self.lower_expr_into(init, dst_addr, &agg_ty).is_none() {
                        return;
                    }
                    // Bind only after lowering `init` so shadowing preserves old bindings cleanly.
                    self.bind_local(name.clone(), slot, agg_ty, *mutable);
                    return;
                }
                let Some((raw_val, init_ty)) = self.lower_expr(init, declared.as_ref()) else {
                    return;
                };
                let (val, final_ty) = if let Some(exp) = declared {
                    if init_ty == IrType::Never {
                        (raw_val, exp)
                    } else if let Some((cval, cty)) =
                        self.coerce_value(raw_val, init_ty.clone(), &exp, *span)
                    {
                        (cval, cty)
                    } else {
                        self.checker.error(
                            format!("type mismatch in `let {name}`: expected `{exp:?}`, found `{init_ty:?}`"),
                            *span,
                        );
                        (raw_val, exp)
                    }
                } else {
                    (raw_val, init_ty)
                };
                let slot = self.alloc_local(name.clone(), final_ty.clone(), *mutable);
                if final_ty.is_scalar() {
                    self.emit(IrInst::StoreLocal {
                        slot,
                        val,
                        ty: final_ty,
                    });
                } else {
                    // Aggregate copy: val holds address of temporary slot
                    self.emit_aggregate_copy_to_local(slot, val, &final_ty);
                }
            }
            Stmt::Assign {
                target,
                op,
                value,
                span,
            } => {
                if let ExprKind::Ident(name) = &target.kind
                    && let Some(v) = self.lookup_var(name)
                    && v.ty.is_scalar()
                {
                    if !v.mutable {
                        self.checker.error("cannot assign to immutable place", *span);
                        return;
                    }
                    let Some((rhs_val, rhs_ty)) = self.lower_expr(value, Some(&v.ty)) else {
                        return;
                    };
                    if rhs_ty != v.ty && rhs_ty != IrType::Never {
                        self.checker.error(
                            format!("type mismatch in assignment: expected `{:?}`, found `{rhs_ty:?}`", v.ty),
                            *span,
                        );
                        return;
                    }
                    let final_val = if *op == AssignOp::Assign {
                        rhs_val
                    } else {
                        let cur = self.fresh_val();
                        self.emit(IrInst::LoadLocal {
                            dst: cur,
                            slot: v.slot,
                            ty: v.ty.clone(),
                        });
                        let bin_op = match op {
                            AssignOp::AddAssign => IrBinOp::Add,
                            AssignOp::SubAssign => IrBinOp::Sub,
                            AssignOp::MulAssign => IrBinOp::Mul,
                            AssignOp::DivAssign => IrBinOp::Div,
                            AssignOp::BitAndAssign => IrBinOp::BitAnd,
                            AssignOp::BitOrAssign => IrBinOp::BitOr,
                            AssignOp::BitXorAssign => IrBinOp::BitXor,
                            AssignOp::ShlAssign => IrBinOp::Shl,
                            AssignOp::ShrAssign => IrBinOp::Shr,
                            AssignOp::Assign => unreachable!(),
                        };
                        let dst = self.fresh_val();
                        let site_id = self.checker.program.intern_site(*span);
                        self.emit(IrInst::BinOp {
                            dst,
                            op: bin_op,
                            mode: ArithMode::Checked,
                            ty: v.ty.clone(),
                            lhs: cur,
                            rhs: rhs_val,
                            site_id,
                        });
                        dst
                    };
                    self.emit(IrInst::StoreLocal {
                        slot: v.slot,
                        val: final_val,
                        ty: v.ty,
                    });
                    return;
                }
                let Some((addr, target_ty, is_mut, is_volatile)) = self.lower_lvalue_ext(target) else {
                    return;
                };
                if !is_mut {
                    self.checker.error("cannot assign to immutable place", *span);
                    return;
                }
                let Some((raw_rhs, rhs_ty)) = self.lower_expr(value, Some(&target_ty)) else {
                    return;
                };
                let rhs_val = if rhs_ty == IrType::Never {
                    raw_rhs
                } else if let Some((cval, _)) =
                    self.coerce_value(raw_rhs, rhs_ty.clone(), &target_ty, *span)
                {
                    cval
                } else {
                    self.checker.error(
                        format!("type mismatch in assignment: expected `{target_ty:?}`, found `{rhs_ty:?}`"),
                        *span,
                    );
                    return;
                };
                let final_val = if *op == AssignOp::Assign {
                    rhs_val
                } else {
                    let cur = self.fresh_val();
                    self.emit(IrInst::LoadPtr {
                        dst: cur,
                        addr,
                        ty: target_ty.clone(),
                        volatile: is_volatile,
                    });
                    let bin_op = match op {
                        AssignOp::AddAssign => IrBinOp::Add,
                        AssignOp::SubAssign => IrBinOp::Sub,
                        AssignOp::MulAssign => IrBinOp::Mul,
                        AssignOp::DivAssign => IrBinOp::Div,
                        AssignOp::BitAndAssign => IrBinOp::BitAnd,
                        AssignOp::BitOrAssign => IrBinOp::BitOr,
                        AssignOp::BitXorAssign => IrBinOp::BitXor,
                        AssignOp::ShlAssign => IrBinOp::Shl,
                        AssignOp::ShrAssign => IrBinOp::Shr,
                        AssignOp::Assign => unreachable!(),
                    };
                    let dst = self.fresh_val();
                    let site_id = self.checker.program.intern_site(*span);
                    self.emit(IrInst::BinOp {
                        dst,
                        op: bin_op,
                        mode: ArithMode::Checked,
                        ty: target_ty.clone(),
                        lhs: cur,
                        rhs: rhs_val,
                        site_id,
                    });
                    dst
                };
                if !target_ty.is_scalar() && *op == AssignOp::Assign {
                    self.emit_aggregate_copy_to_addr(addr, rhs_val, &target_ty);
                    return;
                }
                self.emit(IrInst::StorePtr {
                    addr,
                    val: final_val,
                    ty: target_ty,
                    volatile: is_volatile,
                });
            }
            Stmt::Expr(expr) => {
                self.lower_expr(expr, None);
            }
            Stmt::Return(expr, span) => {
                let ret_ty = self.func().ret_ty.clone();
                match expr {
                    Some(e) => {
                        if let Some((raw_val, ty)) = self.lower_expr(e, Some(&ret_ty)) {
                            let val = if ty == IrType::Never {
                                raw_val
                            } else if let Some((cval, _)) =
                                self.coerce_value(raw_val, ty.clone(), &ret_ty, *span)
                            {
                                cval
                            } else {
                                self.checker.error(
                                    format!("return type mismatch: expected `{ret_ty:?}`, found `{ty:?}`"),
                                    *span,
                                );
                                raw_val
                            };
                            self.unwind_bank_scopes(0);
                            self.set_term(Terminator::Return(Some(val)));
                        }
                    }
                    None => {
                        if ret_ty != IrType::Unit {
                            self.checker.error(
                                format!("missing return value of type `{ret_ty:?}`"),
                                *span,
                            );
                        }
                        self.unwind_bank_scopes(0);
                        self.set_term(Terminator::Return(None));
                    }
                }
            }
            Stmt::Break(span) => {
                if let Some(&(_, break_bb)) = self.loop_stack.last() {
                    self.unwind_bank_scopes(self.loop_stack.len());
                    self.set_term(Terminator::Goto(break_bb));
                } else {
                    self.checker.error("`break` outside of loop", *span);
                }
            }
            Stmt::Continue(span) => {
                if let Some(&(cont_bb, _)) = self.loop_stack.last() {
                    self.unwind_bank_scopes(self.loop_stack.len());
                    self.set_term(Terminator::Goto(cont_bb));
                } else {
                    self.checker.error("`continue` outside of loop", *span);
                }
            }
            Stmt::While { cond, body, span } => {
                let cond_bb = self.new_block();
                let body_bb = self.new_block();
                let exit_bb = self.new_block();
                self.set_term(Terminator::Goto(cond_bb));

                self.cur_bb = cond_bb;
                self.lower_cond(cond, body_bb, exit_bb, "`while` condition must be `bool`", *span);

                self.cur_bb = body_bb;
                self.loop_stack.push((cond_bb, exit_bb));
                self.lower_block(body, None);
                self.loop_stack.pop();
                if !self.is_terminated() {
                    self.set_term(Terminator::Goto(cond_bb));
                }
                self.cur_bb = exit_bb;
            }
            Stmt::Loop { body, .. } => {
                let body_bb = self.new_block();
                let exit_bb = self.new_block();
                self.set_term(Terminator::Goto(body_bb));
                self.cur_bb = body_bb;
                self.loop_stack.push((body_bb, exit_bb));
                self.lower_block(body, None);
                self.loop_stack.pop();
                if !self.is_terminated() {
                    self.set_term(Terminator::Goto(body_bb));
                }
                self.cur_bb = exit_bb;
            }
            Stmt::For {
                var,
                start,
                end,
                inclusive,
                body,
                span,
            } => {
                let Some((sval, sty)) = self.lower_expr(start, None) else {
                    return;
                };
                let Some((eval, ety)) = self.lower_expr(end, Some(&sty)) else {
                    return;
                };
                if sty != ety {
                    self.checker.error(
                        format!("`for` range bounds must have identical types (`{sty:?}` vs `{ety:?}`)"),
                        *span,
                    );
                    return;
                }
                self.scopes.push(HashMap::new());
                let idx_slot = self.alloc_local(var.clone(), sty.clone(), true);
                self.emit(IrInst::StoreLocal {
                    slot: idx_slot,
                    val: sval,
                    ty: sty.clone(),
                });
                let end_slot = self.alloc_local(format!("__end_{var}"), sty.clone(), false);
                self.emit(IrInst::StoreLocal {
                    slot: end_slot,
                    val: eval,
                    ty: sty.clone(),
                });

                let cond_bb = self.new_block();
                let body_bb = self.new_block();
                let step_bb = self.new_block();
                let exit_bb = self.new_block();
                self.set_term(Terminator::Goto(cond_bb));

                self.cur_bb = cond_bb;
                let cur_v = self.fresh_val();
                self.emit(IrInst::LoadLocal {
                    dst: cur_v,
                    slot: idx_slot,
                    ty: sty.clone(),
                });
                let end_v = self.fresh_val();
                self.emit(IrInst::LoadLocal {
                    dst: end_v,
                    slot: end_slot,
                    ty: sty.clone(),
                });
                let cmp = self.fresh_val();
                let site_id = self.checker.program.intern_site(*span);
                self.emit(IrInst::BinOp {
                    dst: cmp,
                    op: if *inclusive { IrBinOp::Le } else { IrBinOp::Lt },
                    mode: ArithMode::Checked,
                    ty: sty.clone(),
                    lhs: cur_v,
                    rhs: end_v,
                    site_id,
                });
                self.set_term(Terminator::Branch {
                    cond: cmp,
                    then_bb: body_bb,
                    else_bb: exit_bb,
                });

                self.cur_bb = body_bb;
                self.loop_stack.push((step_bb, exit_bb));
                self.lower_block(body, None);
                self.loop_stack.pop();
                if !self.is_terminated() {
                    self.set_term(Terminator::Goto(step_bb));
                }

                self.cur_bb = step_bb;
                let cur2 = self.fresh_val();
                self.emit(IrInst::LoadLocal {
                    dst: cur2,
                    slot: idx_slot,
                    ty: sty.clone(),
                });
                let one = self.fresh_val();
                self.emit(IrInst::ConstInt {
                    dst: one,
                    ty: sty.clone(),
                    val: 1,
                });
                let next = self.fresh_val();
                self.emit(IrInst::BinOp {
                    dst: next,
                    op: IrBinOp::Add,
                    mode: ArithMode::Wrapping,
                    ty: sty.clone(),
                    lhs: cur2,
                    rhs: one,
                    site_id,
                });
                self.emit(IrInst::StoreLocal {
                    slot: idx_slot,
                    val: next,
                    ty: sty,
                });
                self.set_term(Terminator::Goto(cond_bb));

                self.scopes.pop();
                self.cur_bb = exit_bb;
            }
        }
    }

    pub(super) fn emit_aggregate_copy_to_addr(
        &mut self,
        dst_base: ValueId,
        src_addr: ValueId,
        ty: &IrType,
    ) {
        let size = self.checker.program.type_size(ty);
        if size >= 4 {
            self.emit(IrInst::MemCopy {
                dst_addr: dst_base,
                src_addr,
                len: size,
            });
            return;
        }
        for off in 0..size {
            let s_addr = if off == 0 {
                src_addr
            } else {
                let a = self.fresh_val();
                self.emit(IrInst::FieldAddr {
                    dst: a,
                    base: src_addr,
                    byte_offset: off,
                });
                a
            };
            let d_addr = if off == 0 {
                dst_base
            } else {
                let a = self.fresh_val();
                self.emit(IrInst::FieldAddr {
                    dst: a,
                    base: dst_base,
                    byte_offset: off,
                });
                a
            };
            let byte = self.fresh_val();
            self.emit(IrInst::LoadPtr {
                dst: byte,
                addr: s_addr,
                ty: IrType::Int(IntType::U8),
                volatile: false,
            });
            self.emit(IrInst::StorePtr {
                addr: d_addr,
                val: byte,
                ty: IrType::Int(IntType::U8),
                volatile: false,
            });
        }
    }

    pub(super) fn emit_aggregate_copy_to_local(&mut self, dst_slot: SlotId, src_addr: ValueId, ty: &IrType) {
        let dst_base = self.fresh_val();
        self.emit(IrInst::AddrLocal {
            dst: dst_base,
            slot: dst_slot,
        });
        self.emit_aggregate_copy_to_addr(dst_base, src_addr, ty);
    }

    /// Lowers an addressable expression to `(addr_val, element_ty, is_mutable)`.
    pub(super) fn lower_lvalue(&mut self, expr: &Expr) -> Option<(ValueId, IrType, bool)> {
        let (addr, ty, is_mut, _) = self.lower_lvalue_ext(expr)?;
        Some((addr, ty, is_mut))
    }

    /// Lowers an addressable expression to `(addr_val, element_ty, is_mutable, is_volatile)`.
    pub(super) fn lower_lvalue_ext(
        &mut self,
        expr: &Expr,
    ) -> Option<(ValueId, IrType, bool, bool)> {
        match &expr.kind {
            ExprKind::Ident(name) => {
                if let Some(v) = self.lookup_var(name) {
                    let addr = self.fresh_val();
                    self.emit(IrInst::AddrLocal {
                        dst: addr,
                        slot: v.slot,
                    });
                    Some((addr, v.ty, v.mutable, false))
                } else if let Some(&(gid, ref gty, gmut)) = self.checker.global_ids.get(name) {
                    let gty = gty.clone();
                    let is_vol = self.checker.program.globals[gid as usize].volatile;
                    if let Some(space) = self.checker.program.globals[gid as usize].space.clone() {
                        self.check_bank_mapped(&space, &format!("accessing `{name}`"), expr.span);
                    }
                    let addr = self.fresh_val();
                    self.emit(IrInst::AddrGlobal {
                        dst: addr,
                        global: gid,
                    });
                    Some((addr, gty, gmut, is_vol))
                } else {
                    self.checker
                        .error(format!("undefined variable `{name}`"), expr.span);
                    None
                }
            }
            ExprKind::Deref(inner) => {
                let (ptr_val, ptr_ty) = self.lower_expr(inner, None)?;
                match ptr_ty {
                    IrType::Ref { mutable, inner } => Some((ptr_val, *inner, mutable, false)),
                    other => {
                        self.checker.error(
                            format!("cannot dereference non-reference type `{other:?}`"),
                            expr.span,
                        );
                        None
                    }
                }
            }
            ExprKind::Field { base, field } => {
                let (mut base_addr, mut base_ty, mut is_mut, mut is_vol) =
                    self.lower_lvalue_ext(base)?;
                // Auto-deref through `&T` / `&mut T`
                if let IrType::Ref { mutable, inner } = base_ty {
                    let loaded = self.fresh_val();
                    self.emit(IrInst::LoadPtr {
                        dst: loaded,
                        addr: base_addr,
                        ty: IrType::Ref {
                            mutable,
                            inner: inner.clone(),
                        },
                        volatile: is_vol,
                    });
                    base_addr = loaded;
                    base_ty = *inner;
                    is_mut = mutable;
                    is_vol = false;
                }
                match base_ty {
                    IrType::Struct(sid) => {
                        let sdef = &self.checker.program.structs[sid as usize];
                        if let Some((_, fty, offset)) =
                            sdef.fields.iter().find(|(n, _, _)| n == field).cloned()
                        {
                            let faddr = self.fresh_val();
                            self.emit(IrInst::FieldAddr {
                                dst: faddr,
                                base: base_addr,
                                byte_offset: offset,
                            });
                            Some((faddr, fty, is_mut, is_vol))
                        } else {
                            self.checker.error(
                                format!("struct `{}` has no field `{field}`", sdef.name),
                                expr.span,
                            );
                            None
                        }
                    }
                    IrType::Machine(mid) => {
                        let mdef = self.checker.program.machines[mid as usize].clone();
                        // Check shared fields first
                        if let Some((_, fty, offset)) = mdef
                            .shared_fields
                            .iter()
                            .find(|(n, _, _)| n == field)
                            .cloned()
                        {
                            let faddr = self.fresh_val();
                            self.emit(IrInst::FieldAddr {
                                dst: faddr,
                                base: base_addr,
                                byte_offset: offset,
                            });
                            return Some((faddr, fty, is_mut, is_vol));
                        }
                        // If inside `on <State>` for this machine, allow state-local fields!
                        if let Some((ctx_mid, Some(sid), _)) = self.machine_ctx
                            && ctx_mid == mid
                            && let Some((_, fty, rel_off)) = mdef.states[sid as usize]
                                .fields
                                .iter()
                                .find(|(n, _, _)| n == field)
                                .cloned()
                        {
                            let faddr = self.fresh_val();
                            self.emit(IrInst::FieldAddr {
                                dst: faddr,
                                base: base_addr,
                                byte_offset: mdef.payload_offset + rel_off,
                            });
                            return Some((faddr, fty, is_mut, is_vol));
                        }
                        self.checker.error(
                            format!("machine `{}` has no accessible field `{field}` in this state", mdef.name),
                            expr.span,
                        );
                        None
                    }
                    other => {
                        self.checker.error(
                            format!("field access on non-struct type `{other:?}`"),
                            expr.span,
                        );
                        None
                    }
                }
            }
            ExprKind::Index { base, index } => {
                let (mut base_addr, base_ty, mut is_mut, is_vol) =
                    if let Some((baddr, bty, bmut, bvol)) = self.lower_lvalue_if_place_ext(base) {
                        match bty {
                            IrType::Ref { mutable, inner } => {
                                let loaded = self.fresh_val();
                                self.emit(IrInst::LoadPtr {
                                    dst: loaded,
                                    addr: baddr,
                                    ty: IrType::Ref {
                                        mutable,
                                        inner: inner.clone(),
                                    },
                                    volatile: bvol,
                                });
                                (loaded, *inner, mutable, false)
                            }
                            ref sty @ (IrType::Slice { mutable: _, ref elem }
                            | IrType::ConstSlice(ref elem)) => {
                                let mutable = matches!(sty, IrType::Slice { mutable: true, .. });
                                let elem = elem.clone();
                                let fat = self.fresh_val();
                                self.emit(IrInst::LoadPtr {
                                    dst: fat,
                                    addr: baddr,
                                    ty: sty.clone(),
                                    volatile: bvol,
                                });
                                let ptr = self.fresh_val();
                                self.emit(IrInst::Cast {
                                    dst: ptr,
                                    val: fat,
                                    from_ty: sty.clone(),
                                    to_ty: IrType::Int(IntType::U16),
                                });
                                (
                                    ptr,
                                    IrType::Array { elem, len: 0 },
                                    mutable,
                                    false,
                                )
                            }
                            other => (baddr, other, bmut, bvol),
                        }
                    } else {
                        let (val, ty) = self.lower_expr(base, None)?;
                        match ty {
                            IrType::Ref { mutable, inner } => (val, *inner, mutable, false),
                            ref sty @ (IrType::Slice { mutable: _, ref elem }
                            | IrType::ConstSlice(ref elem)) => {
                                let mutable = matches!(sty, IrType::Slice { mutable: true, .. });
                                let elem = elem.clone();
                                let ptr = self.fresh_val();
                                self.emit(IrInst::Cast {
                                    dst: ptr,
                                    val,
                                    from_ty: sty.clone(),
                                    to_ty: IrType::Int(IntType::U16),
                                });
                                (
                                    ptr,
                                    IrType::Array { elem, len: 0 },
                                    mutable,
                                    false,
                                )
                            }
                            other => (val, other, false, false),
                        }
                    };
                let _ = (&mut base_addr, &mut is_mut);
                match base_ty {
                    IrType::Array { elem, len } => {
                        let (idx_val, idx_ty) = self.lower_expr(index, None)?;
                        if !matches!(idx_ty, IrType::Int(_)) {
                            self.checker.error("array index must be an integer", index.span);
                            return None;
                        }
                        let elem_size = self.checker.program.type_size(&elem);
                        let site_id = self.checker.program.intern_site(expr.span);
                        let dst = self.fresh_val();
                        self.emit(IrInst::IndexAddr {
                            dst,
                            base: base_addr,
                            index: idx_val,
                            elem_size,
                            len,
                            site_id,
                        });
                        Some((dst, *elem, is_mut, is_vol))
                    }
                    other => {
                        self.checker.error(
                            format!("indexing non-array type `{other:?}`"),
                            expr.span,
                        );
                        None
                    }
                }
            }
            _ => {
                self.checker
                    .error("expression is not an addressable place", expr.span);
                None
            }
        }
    }

    pub(super) fn extract_borrow_place(expr: &Expr) -> Option<BorrowPlace> {
        match &expr.kind {
            ExprKind::Ref { mutable, expr } => {
                let mut place = Self::extract_place_path(expr)?;
                place.mutable = *mutable;
                Some(place)
            }
            _ => None,
        }
    }

    fn extract_place_path(expr: &Expr) -> Option<BorrowPlace> {
        match &expr.kind {
            ExprKind::Ident(name) => Some(BorrowPlace {
                root: name.clone(),
                fields: Vec::new(),
                mutable: false,
            }),
            ExprKind::Field { base, field } => {
                let mut p = Self::extract_place_path(base)?;
                p.fields.push(field.clone());
                Some(p)
            }
            ExprKind::Deref(inner) => Self::extract_place_path(inner),
            _ => None,
        }
    }

    /// Lowers a boolean condition directly into control-flow jumps (`then_bb` / `else_bb`),
    /// eliminating `__logic_tmp` materialization for short-circuit `&&`, `||`, and `!`.
    pub(super) fn lower_cond(
        &mut self,
        expr: &Expr,
        then_bb: BlockId,
        else_bb: BlockId,
        err_msg: &str,
        err_span: Span,
    ) {
        match &expr.kind {
            ExprKind::Binary {
                op: BinOp::And,
                lhs,
                rhs,
                ..
            } => {
                let rhs_bb = self.new_block();
                self.lower_cond(lhs, rhs_bb, else_bb, "logical operator requires `bool`", lhs.span);
                self.cur_bb = rhs_bb;
                self.lower_cond(rhs, then_bb, else_bb, "logical operator requires `bool`", rhs.span);
            }
            ExprKind::Binary {
                op: BinOp::Or,
                lhs,
                rhs,
                ..
            } => {
                let rhs_bb = self.new_block();
                self.lower_cond(lhs, then_bb, rhs_bb, "logical operator requires `bool`", lhs.span);
                self.cur_bb = rhs_bb;
                self.lower_cond(rhs, then_bb, else_bb, "logical operator requires `bool`", rhs.span);
            }
            ExprKind::Unary {
                op: UnaryOp::Not,
                expr: inner,
            } => {
                self.lower_cond(inner, else_bb, then_bb, err_msg, err_span);
            }
            _ => {
                if let Some((cval, cty)) = self.lower_expr(expr, Some(&IrType::Bool)) {
                    if cty != IrType::Bool {
                        self.checker.error(err_msg, err_span);
                    }
                    self.set_term(Terminator::Branch {
                        cond: cval,
                        then_bb,
                        else_bb,
                    });
                }
            }
        }
    }

    pub(super) fn lower_lvalue_if_place_ext(
        &mut self,
        expr: &Expr,
    ) -> Option<(ValueId, IrType, bool, bool)> {
        if matches!(
            expr.kind,
            ExprKind::Ident(_) | ExprKind::Field { .. } | ExprKind::Index { .. } | ExprKind::Deref(_)
        ) {
            self.lower_lvalue_ext(expr)
        } else {
            None
        }
    }

    pub(super) fn lower_state_payload(
        &mut self,
        target: &StateDef,
        fields: &[(String, Expr)],
        span: Span,
    ) -> Option<Vec<(u16, ValueId, IrType)>> {
        let mut out = Vec::with_capacity(fields.len());
        let mut seen = std::collections::HashSet::new();
        for (fname, fexpr) in fields {
            if !seen.insert(fname.as_str()) {
                self.checker.error(
                    format!("duplicate field `{fname}` in state `{}` initializer", target.name),
                    fexpr.span,
                );
                return None;
            }
            let Some((_, fty, rel_off)) =
                target.fields.iter().find(|(n, _, _)| n == fname).cloned()
            else {
                self.checker.error(
                    format!("state `{}` has no field `{fname}`", target.name),
                    fexpr.span,
                );
                return None;
            };
            let (fval, actual_ty) = self.lower_expr(fexpr, Some(&fty))?;
            if actual_ty != fty {
                self.checker.error(
                    format!(
                        "state field `{fname}` type mismatch: expected `{fty:?}`, found `{actual_ty:?}`"
                    ),
                    fexpr.span,
                );
                return None;
            }
            out.push((rel_off, fval, fty));
        }
        if out.len() != target.fields.len() {
            self.checker.error(
                format!(
                    "state `{}` requires {} field(s), got {}",
                    target.name,
                    target.fields.len(),
                    out.len()
                ),
                span,
            );
            return None;
        }
        Some(out)
    }
}
