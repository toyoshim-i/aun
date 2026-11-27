use super::*;

impl<'a> FnLowerer<'a> {
    pub(super) fn pack_fat_slice(
        &mut self,
        ptr_val: ValueId,
        len: u16,
        slice_ty: IrType,
        span: Span,
    ) -> (ValueId, IrType) {
        let p32 = self.fresh_val();
        self.emit(IrInst::Cast {
            dst: p32,
            val: ptr_val,
            from_ty: IrType::Int(IntType::U16),
            to_ty: IrType::Int(IntType::U32),
        });
        let l32 = self.fresh_val();
        self.emit(IrInst::ConstInt {
            dst: l32,
            ty: IrType::Int(IntType::U32),
            val: (len as i64) << 16,
        });
        let fat = self.fresh_val();
        let site_id = self.checker.program.intern_site(span);
        self.emit(IrInst::BinOp {
            dst: fat,
            op: IrBinOp::BitOr,
            mode: ArithMode::Wrapping,
            ty: IrType::Int(IntType::U32),
            lhs: p32,
            rhs: l32,
            site_id,
        });
        (fat, slice_ty)
    }

    pub(super) fn coerce_value(
        &mut self,
        val: ValueId,
        from_ty: IrType,
        expected: &IrType,
        span: Span,
    ) -> Option<(ValueId, IrType)> {
        if &from_ty == expected {
            return Some((val, expected.clone()));
        }
        match (from_ty, expected) {
            (
                IrType::Ref {
                    mutable: true,
                    inner: f_in,
                },
                IrType::Ref {
                    mutable: false,
                    inner: e_in,
                },
            ) if f_in == *e_in => Some((val, expected.clone())),
            (
                IrType::Ref {
                    mutable: f_mut,
                    inner,
                },
                IrType::Slice {
                    mutable: e_mut,
                    elem: e_elem,
                },
            ) if (f_mut || !*e_mut)
                && matches!(
                    inner.as_ref(),
                    IrType::Array { elem, .. } if elem == e_elem
                ) =>
            {
                let IrType::Array { len, .. } = *inner else {
                    unreachable!();
                };
                Some(self.pack_fat_slice(val, len, expected.clone(), span))
            }
            (IrType::Ref { inner, .. }, IrType::ConstSlice(e_elem))
                if matches!(inner.as_ref(), IrType::Array { elem, .. } if elem == e_elem) =>
            {
                let IrType::Array { len, .. } = *inner else {
                    unreachable!();
                };
                Some(self.pack_fat_slice(val, len, expected.clone(), span))
            }
            (
                IrType::ConstSlice(f_elem),
                IrType::Slice {
                    mutable: false,
                    elem: e_elem,
                },
            ) if f_elem == *e_elem => Some((val, expected.clone())),
            (
                IrType::Slice {
                    mutable: f_mut,
                    elem: f_elem,
                },
                IrType::Slice {
                    mutable: e_mut,
                    elem: e_elem,
                },
            ) if (f_mut || !*e_mut) && f_elem == *e_elem => Some((val, expected.clone())),
            (IrType::Slice { elem: f_elem, .. }, IrType::ConstSlice(e_elem))
                if f_elem == *e_elem =>
            {
                Some((val, expected.clone()))
            }
            _ => None,
        }
    }

    fn lower_rom_bytes_expr(
        &mut self,
        bytes: &[u8],
        expected: Option<&IrType>,
        span: Span,
    ) -> Option<(ValueId, IrType)> {
        let Ok(len) = u16::try_from(bytes.len()) else {
            self.checker
                .error("byte/string literal exceeds 65535 bytes", span);
            return None;
        };
        let gid = self.checker.intern_rom_bytes(bytes);
        let addr = self.fresh_val();
        self.emit(IrInst::AddrGlobal {
            dst: addr,
            global: gid,
        });
        let u8_ty = IrType::Int(IntType::U8);
        if let Some(IrType::Array { elem, len: elen }) = expected
            && **elem == u8_ty
            && *elen == len
        {
            return Some((
                addr,
                IrType::Array {
                    elem: Box::new(u8_ty),
                    len,
                },
            ));
        }
        let nat_ty = IrType::Ref {
            mutable: false,
            inner: Box::new(IrType::Array {
                elem: Box::new(u8_ty),
                len,
            }),
        };
        if let Some(exp) = expected
            && let Some(coerced) = self.coerce_value(addr, nat_ty.clone(), exp, span)
        {
            return Some(coerced);
        }
        Some((addr, nat_ty))
    }

    pub(super) fn lower_expr(&mut self, expr: &Expr, expected: Option<&IrType>) -> Option<(ValueId, IrType)> {
        match &expr.kind {
            ExprKind::IntLit { value, suffix } => {
                let ty = if let Some(suf) = suffix {
                    self.checker
                        .resolve_type(&TypeExpr::Named(suf.clone()), expr.span, false)
                } else if let Some(exp @ IrType::Int(_)) = expected {
                    exp.clone()
                } else {
                    IrType::Int(IntType::U8)
                };
                if let IrType::Int(it) = ty {
                    let v = *value as i64;
                    if v < it.min_i64() || v > it.max_i64() {
                        self.checker.error(
                            format!("integer literal `{value}` out of range for `{it}`"),
                            expr.span,
                        );
                    }
                }
                let dst = self.fresh_val();
                self.emit(IrInst::ConstInt {
                    dst,
                    ty: ty.clone(),
                    val: *value as i64,
                });
                Some((dst, ty))
            }
            ExprKind::FixedLit { text, suffix } => {
                let ty = if let Some(suf) = suffix {
                    self.checker
                        .resolve_type(&TypeExpr::Named(suf.clone()), expr.span, false)
                } else if let Some(exp @ IrType::Fixed { .. }) = expected {
                    exp.clone()
                } else {
                    IrType::Fixed {
                        base: IntType::I16,
                        frac_bits: 8,
                    }
                };
                let IrType::Fixed { base, frac_bits } = ty else {
                    self.checker
                        .error("fixed-point literal requires a fixed-point type", expr.span);
                    return None;
                };
                let scaled = self
                    .checker
                    .parse_fixed_scaled(text, base, frac_bits, expr.span)
                    .unwrap_or(0);
                let dst = self.fresh_val();
                self.emit(IrInst::ConstInt {
                    dst,
                    ty: ty.clone(),
                    val: scaled,
                });
                Some((dst, ty))
            }
            ExprKind::BoolLit(b) => {
                let dst = self.fresh_val();
                self.emit(IrInst::ConstBool { dst, val: *b });
                Some((dst, IrType::Bool))
            }
            ExprKind::ByteLit(b) => {
                let dst = self.fresh_val();
                self.emit(IrInst::ConstInt {
                    dst,
                    ty: IrType::Int(IntType::U8),
                    val: *b as i64,
                });
                Some((dst, IrType::Int(IntType::U8)))
            }
            ExprKind::StrLit(s) => self.lower_rom_bytes_expr(s.as_bytes(), expected, expr.span),
            ExprKind::ByteStrLit(bytes) => self.lower_rom_bytes_expr(bytes, expected, expr.span),
            ExprKind::Ident(name) => {
                if let Some((cty, cval)) = self.checker.consts.get(name).cloned() {
                    let dst = self.fresh_val();
                    self.emit(IrInst::ConstInt {
                        dst,
                        ty: cty.clone(),
                        val: cval,
                    });
                    return Some((dst, cty));
                }
                if let Some(v) = self.lookup_var(name) {
                    if v.ty.is_scalar() {
                        let dst = self.fresh_val();
                        self.emit(IrInst::LoadLocal {
                            dst,
                            slot: v.slot,
                            ty: v.ty.clone(),
                        });
                        return Some((dst, v.ty));
                    } else {
                        let addr = self.fresh_val();
                        self.emit(IrInst::AddrLocal {
                            dst: addr,
                            slot: v.slot,
                        });
                        return Some((addr, v.ty));
                    }
                }
                if let Some(&(gid, ref gty, _)) = self.checker.global_ids.get(name) {
                    let gty = gty.clone();
                    if let Some(space) = self.checker.program.globals[gid as usize].space.clone() {
                        self.check_bank_mapped(&space, &format!("reading `{name}`"), expr.span);
                    }
                    let addr = self.fresh_val();
                    self.emit(IrInst::AddrGlobal {
                        dst: addr,
                        global: gid,
                    });
                    if gty.is_scalar() {
                        let is_vol = self.checker.program.globals[gid as usize].volatile;
                        let dst = self.fresh_val();
                        self.emit(IrInst::LoadPtr {
                            dst,
                            addr,
                            ty: gty.clone(),
                            volatile: is_vol,
                        });
                        return Some((dst, gty));
                    } else {
                        return Some((addr, gty));
                    }
                }
                self.checker
                    .error(format!("undefined identifier `{name}`"), expr.span);
                None
            }
            ExprKind::Path(segs) => {
                if segs.len() == 2
                    && let Some((repr, variants)) = self.checker.enums.get(&segs[0])
                    && let Some(&val) = variants.get(&segs[1])
                {
                    let ty = IrType::Int(*repr);
                    let dst = self.fresh_val();
                    self.emit(IrInst::ConstInt {
                        dst,
                        ty: ty.clone(),
                        val,
                    });
                    return Some((dst, ty));
                }
                self.checker
                    .error(format!("unresolved path `{}`", segs.join("::")), expr.span);
                None
            }
            ExprKind::Ref { mutable, expr: inner } => {
                let (addr, ty, place_mut) = self.lower_lvalue(inner)?;
                if *mutable && !place_mut {
                    self.checker.error(
                        "cannot take mutable reference `&mut` of immutable variable or place",
                        expr.span,
                    );
                }
                Some((
                    addr,
                    IrType::Ref {
                        mutable: *mutable,
                        inner: Box::new(ty),
                    },
                ))
            }
            ExprKind::Deref(inner) => {
                let (ptr_val, ptr_ty) = self.lower_expr(inner, None)?;
                let IrType::Ref { inner: elem_ty, .. } = ptr_ty else {
                    self.checker
                        .error("cannot dereference non-reference", expr.span);
                    return None;
                };
                if elem_ty.is_scalar() {
                    let dst = self.fresh_val();
                    self.emit(IrInst::LoadPtr {
                        dst,
                        addr: ptr_val,
                        ty: (*elem_ty).clone(),
                        volatile: false,
                    });
                    Some((dst, *elem_ty))
                } else {
                    Some((ptr_val, *elem_ty))
                }
            }
            ExprKind::Field { .. } | ExprKind::Index { .. } => {
                let (addr, ty, _, is_vol) = self.lower_lvalue_ext(expr)?;
                if ty.is_scalar() {
                    let dst = self.fresh_val();
                    self.emit(IrInst::LoadPtr {
                        dst,
                        addr,
                        ty: ty.clone(),
                        volatile: is_vol,
                    });
                    Some((dst, ty))
                } else {
                    Some((addr, ty))
                }
            }
            ExprKind::Unary { op, expr: inner } => {
                let (val, ty) = self.lower_expr(inner, expected)?;
                let ir_op = match op {
                    UnaryOp::Neg => IrUnOp::Neg,
                    UnaryOp::Not => IrUnOp::Not,
                };
                let dst = self.fresh_val();
                let site_id = self.checker.program.intern_site(expr.span);
                self.emit(IrInst::UnOp {
                    dst,
                    op: ir_op,
                    mode: ArithMode::Checked,
                    ty: ty.clone(),
                    val,
                    site_id,
                });
                Some((dst, ty))
            }
            ExprKind::Binary { op, mode, lhs, rhs } => {
                // Short-circuiting logical && / ||
                if matches!(op, BinOp::And | BinOp::Or) {
                    let (lval, lty) = self.lower_expr(lhs, Some(&IrType::Bool))?;
                    if lty != IrType::Bool {
                        self.checker.error("logical operator requires `bool`", lhs.span);
                    }
                    let res_slot = self.alloc_local("__logic_tmp".into(), IrType::Bool, true);
                    self.emit(IrInst::StoreLocal {
                        slot: res_slot,
                        val: lval,
                        ty: IrType::Bool,
                    });
                    let rhs_bb = self.new_block();
                    let join_bb = self.new_block();
                    if *op == BinOp::And {
                        self.set_term(Terminator::Branch {
                            cond: lval,
                            then_bb: rhs_bb,
                            else_bb: join_bb,
                        });
                    } else {
                        self.set_term(Terminator::Branch {
                            cond: lval,
                            then_bb: join_bb,
                            else_bb: rhs_bb,
                        });
                    }
                    self.cur_bb = rhs_bb;
                    if let Some((rval, rty)) = self.lower_expr(rhs, Some(&IrType::Bool)) {
                        if rty != IrType::Bool {
                            self.checker.error("logical operator requires `bool`", rhs.span);
                        }
                        self.emit(IrInst::StoreLocal {
                            slot: res_slot,
                            val: rval,
                            ty: IrType::Bool,
                        });
                    }
                    self.set_term(Terminator::Goto(join_bb));
                    self.cur_bb = join_bb;
                    let dst = self.fresh_val();
                    self.emit(IrInst::LoadLocal {
                        dst,
                        slot: res_slot,
                        ty: IrType::Bool,
                    });
                    return Some((dst, IrType::Bool));
                }

                let is_cmp = matches!(
                    op,
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
                );
                let operand_expected = if is_cmp { None } else { expected };
                let (lval, lty) = self.lower_expr(lhs, operand_expected)?;
                let (rval, rty) = self.lower_expr(rhs, Some(&lty))?;
                if lty != rty {
                    self.checker.error(
                        format!(
                            "strict type mismatch in binary op: `{lty:?}` vs `{rty:?}` (use explicit `as` cast)"
                        ),
                        expr.span,
                    );
                    return None;
                }
                let ir_op = match op {
                    BinOp::Add => IrBinOp::Add,
                    BinOp::Sub => IrBinOp::Sub,
                    BinOp::Mul => IrBinOp::Mul,
                    BinOp::Div => IrBinOp::Div,
                    BinOp::Rem => IrBinOp::Rem,
                    BinOp::BitAnd => IrBinOp::BitAnd,
                    BinOp::BitOr => IrBinOp::BitOr,
                    BinOp::BitXor => IrBinOp::BitXor,
                    BinOp::Shl => IrBinOp::Shl,
                    BinOp::Shr => IrBinOp::Shr,
                    BinOp::Eq => IrBinOp::Eq,
                    BinOp::Ne => IrBinOp::Ne,
                    BinOp::Lt => IrBinOp::Lt,
                    BinOp::Le => IrBinOp::Le,
                    BinOp::Gt => IrBinOp::Gt,
                    BinOp::Ge => IrBinOp::Ge,
                    BinOp::And | BinOp::Or => unreachable!(),
                };
                let res_ty = if is_cmp { IrType::Bool } else { lty.clone() };
                let dst = self.fresh_val();
                let site_id = self.checker.program.intern_site(expr.span);
                self.emit(IrInst::BinOp {
                    dst,
                    op: ir_op,
                    mode: *mode,
                    ty: lty,
                    lhs: lval,
                    rhs: rval,
                    site_id,
                });
                Some((dst, res_ty))
            }
            ExprKind::Cast { expr: inner, ty } => {
                let to_ty = self.checker.resolve_type(ty, expr.span, false);
                let (val, from_ty) = self.lower_expr(inner, None)?;
                let dst = self.fresh_val();
                self.emit(IrInst::Cast {
                    dst,
                    val,
                    from_ty,
                    to_ty: to_ty.clone(),
                });
                Some((dst, to_ty))
            }
            ExprKind::Call { callee, args } => {
                // Check second-class borrow aliasing across call arguments (AD-3)
                let borrows: Vec<(BorrowPlace, Span)> = args
                    .iter()
                    .filter_map(|a| Self::extract_borrow_place(a).map(|p| (p, a.span)))
                    .collect();
                for i in 0..borrows.len() {
                    for j in (i + 1)..borrows.len() {
                        if borrows[i].0.conflicts_with(&borrows[j].0) {
                            self.checker.error(
                                format!(
                                    "aliasing violation: conflicting borrow of `{}` in function call (AD-3)",
                                    borrows[i].0.root
                                ),
                                borrows[j].1,
                            );
                        }
                    }
                }

                // Built-in port I/O intrinsics: io_out(port, val) / io_in(port)
                if let ExprKind::Ident(name) = &callee.kind {
                    if name == "io_out" {
                        if args.len() != 2 {
                            self.checker
                                .error("`io_out` expects 2 arguments (port: u8, val: u8)", expr.span);
                            return None;
                        }
                        let (p, _) = self.lower_expr(&args[0], Some(&IrType::Int(IntType::U8)))?;
                        let (v, _) = self.lower_expr(&args[1], Some(&IrType::Int(IntType::U8)))?;
                        self.emit(IrInst::PortOut { port: p, val: v });
                        return Some((0, IrType::Unit));
                    }
                    if name == "io_in" {
                        if args.len() != 1 {
                            self.checker
                                .error("`io_in` expects 1 argument (port: u8)", expr.span);
                            return None;
                        }
                        let (p, _) = self.lower_expr(&args[0], Some(&IrType::Int(IntType::U8)))?;
                        let dst = self.fresh_val();
                        self.emit(IrInst::PortIn { dst, port: p });
                        return Some((dst, IrType::Int(IntType::U8)));
                    }
                    if matches!(
                        name.as_str(),
                        "mem_read_u8"
                            | "mem_read_u8_volatile"
                            | "mem_read_u16"
                            | "mem_read_u16_volatile"
                    ) {
                        let volatile = name.ends_with("_volatile");
                        let read_ty = if name.starts_with("mem_read_u16") {
                            IrType::Int(IntType::U16)
                        } else {
                            IrType::Int(IntType::U8)
                        };
                        if args.len() != 1 {
                            self.checker.error(
                                format!("`{name}` expects 1 argument (addr: u16)"),
                                expr.span,
                            );
                            return None;
                        }
                        let (a, _) = self.lower_expr(&args[0], Some(&IrType::Int(IntType::U16)))?;
                        let dst = self.fresh_val();
                        self.emit(IrInst::LoadPtr {
                            dst,
                            addr: a,
                            ty: read_ty.clone(),
                            volatile,
                        });
                        return Some((dst, read_ty));
                    }
                    if matches!(
                        name.as_str(),
                        "mem_write_u8"
                            | "mem_write_u8_volatile"
                            | "mem_write_u16"
                            | "mem_write_u16_volatile"
                    ) {
                        let volatile = name.ends_with("_volatile");
                        let val_ty = if name.starts_with("mem_write_u16") {
                            IrType::Int(IntType::U16)
                        } else {
                            IrType::Int(IntType::U8)
                        };
                        if args.len() != 2 {
                            self.checker.error(
                                format!("`{name}` expects 2 arguments (addr: u16, val)"),
                                expr.span,
                            );
                            return None;
                        }
                        let (a, _) = self.lower_expr(&args[0], Some(&IrType::Int(IntType::U16)))?;
                        let (v, _) = self.lower_expr(&args[1], Some(&val_ty))?;
                        self.emit(IrInst::StorePtr {
                            addr: a,
                            val: v,
                            ty: val_ty,
                            volatile,
                        });
                        return Some((0, IrType::Unit));
                    }
                    if name == "addr_of" {
                        if args.len() != 1 {
                            self.checker
                                .error("`addr_of` expects 1 lvalue or function argument", expr.span);
                            return None;
                        }
                        if let ExprKind::Ident(ref fname) = args[0].kind
                            && self.lookup_var(fname).is_none()
                            && !self.checker.global_ids.contains_key(fname)
                            && let Some(sig) = self.checker.fn_sigs.get(fname).cloned()
                        {
                            let dst = self.fresh_val();
                            if sig.is_extern {
                                if let Some(fixed) =
                                    self.checker.program.externs[sig.id as usize].fixed_addr
                                {
                                    self.emit(IrInst::ConstInt {
                                        dst,
                                        ty: IrType::Int(IntType::U16),
                                        val: fixed as i64,
                                    });
                                } else {
                                    self.emit(IrInst::AddrFn {
                                        dst,
                                        target: IrFnTarget::Extern(sig.id),
                                    });
                                }
                            } else {
                                self.emit(IrInst::AddrFn {
                                    dst,
                                    target: IrFnTarget::Fn(sig.id),
                                });
                            }
                            return Some((dst, IrType::Int(IntType::U16)));
                        }
                        let (addr, _, _) = self.lower_lvalue(&args[0])?;
                        return Some((addr, IrType::Int(IntType::U16)));
                    }
                    if name == "call_u16" {
                        if !(1..=3).contains(&args.len()) {
                            self.checker.error(
                                "`call_u16` expects 1 to 3 arguments (addr: u16, [arg0], [arg1])",
                                expr.span,
                            );
                            return None;
                        }
                        let u16_ty = IrType::Int(IntType::U16);
                        let (raw_addr, aty) = self.lower_expr(&args[0], Some(&u16_ty))?;
                        let Some((addr, _)) =
                            self.coerce_value(raw_addr, aty.clone(), &u16_ty, args[0].span)
                        else {
                            self.checker.error(
                                format!("`call_u16` target must be `u16`, found `{aty:?}`"),
                                args[0].span,
                            );
                            return None;
                        };
                        let mut call_args = Vec::with_capacity(args.len() - 1);
                        for (idx, aexpr) in args[1..].iter().enumerate() {
                            let (av, at) = self.lower_expr(aexpr, None)?;
                            let sz = self.checker.program.type_size(&at);
                            let valid = if idx == 0 {
                                matches!(sz, 1 | 2 | 4)
                            } else {
                                matches!(sz, 1 | 2)
                            };
                            if !valid {
                                self.checker.error(
                                    format!("unsupported argument size {sz}B for `call_u16`"),
                                    aexpr.span,
                                );
                                return None;
                            }
                            call_args.push((av, at));
                        }
                        self.emit(IrInst::CallIndirect {
                            addr,
                            args: call_args,
                        });
                        return Some((0, IrType::Unit));
                    }

                    let Some(sig) = self.checker.fn_sigs.get(name).cloned() else {
                        self.checker
                            .error(format!("undefined function `{name}`"), expr.span);
                        return None;
                    };
                    if args.len() != sig.params.len() {
                        self.checker.error(
                            format!(
                                "function `{name}` expects {} argument(s), found {}",
                                sig.params.len(),
                                args.len()
                            ),
                            expr.span,
                        );
                        return None;
                    }
                    let mut arg_vals = Vec::with_capacity(args.len());
                    for (arg, expected_ty) in args.iter().zip(sig.params.iter()) {
                        let (aval, aty) = self.lower_expr(arg, Some(expected_ty))?;
                        let final_val = if let Some((cval, _)) =
                            self.coerce_value(aval, aty.clone(), expected_ty, arg.span)
                        {
                            cval
                        } else {
                            self.checker.error(
                                format!(
                                    "argument type mismatch in call to `{name}`: expected `{expected_ty:?}`, found `{aty:?}`"
                                ),
                                arg.span,
                            );
                            aval
                        };
                        arg_vals.push(final_val);
                    }
                    let dst = if sig.ret_ty == IrType::Unit {
                        None
                    } else {
                        Some(self.fresh_val())
                    };
                    if sig.is_extern {
                        let ef = self.checker.program.externs[sig.id as usize].clone();
                        for req_bank in &ef.requires_banks {
                            self.check_bank_mapped(
                                req_bank,
                                &format!("calling `{name}`"),
                                expr.span,
                            );
                        }
                        if ef.requires_di && !self.irq_disabled {
                            self.checker.error(
                                format!(
                                    "interrupt safety violation: calling `{name}` requires `di` (use `with bank(...)` or an ISR)"
                                ),
                                expr.span,
                            );
                        }
                        self.emit(IrInst::ExternCall {
                            dst,
                            extern_fn: sig.id,
                            args: arg_vals,
                        });
                    } else {
                        self.emit(IrInst::Call {
                            dst,
                            func: sig.id,
                            args: arg_vals,
                        });
                    }
                    return Some((dst.unwrap_or(0), sig.ret_ty));
                }

                // Fixed-point constructor path: q8_8::from_raw(x) / Machine::new()
                if let ExprKind::Path(segs) = &callee.kind && segs.len() == 2 {
                    if segs[1] == "from_raw" {
                        let target_ty =
                            self.checker
                                .resolve_type(&TypeExpr::Named(segs[0].clone()), expr.span, false);
                        if let IrType::Fixed { base, .. } = target_ty {
                            if args.len() != 1 {
                                self.checker
                                    .error("`from_raw` expects 1 argument", expr.span);
                                return None;
                            }
                            let (v, _) = self.lower_expr(&args[0], Some(&IrType::Int(base)))?;
                            return Some((v, target_ty));
                        }
                    }
                    if segs[1] == "new"
                        && let Some(&mid) = self.checker.machine_ids.get(&segs[0])
                    {
                        let mdef = self.checker.program.machines[mid as usize].clone();
                        let tmp_slot = self.alloc_local(
                            format!("__machine_new_{mid}"),
                            IrType::Machine(mid),
                            true,
                        );
                        let base_addr = self.fresh_val();
                        self.emit(IrInst::AddrLocal {
                            dst: base_addr,
                            slot: tmp_slot,
                        });
                        // Zero-initialize all bytes of the machine (shared fields + payload)
                        let zero_u8 = self.fresh_val();
                        self.emit(IrInst::ConstInt {
                            dst: zero_u8,
                            ty: IrType::Int(IntType::U8),
                            val: 0,
                        });
                        for off in 0..mdef.total_size {
                            let baddr = if off == 0 {
                                base_addr
                            } else {
                                let a = self.fresh_val();
                                self.emit(IrInst::FieldAddr {
                                    dst: a,
                                    base: base_addr,
                                    byte_offset: off,
                                });
                                a
                            };
                            self.emit(IrInst::StorePtr {
                                addr: baddr,
                                val: zero_u8,
                                ty: IrType::Int(IntType::U8),
                                volatile: false,
                            });
                        }
                        // Write initial state tag
                        let tag_addr = self.fresh_val();
                        self.emit(IrInst::FieldAddr {
                            dst: tag_addr,
                            base: base_addr,
                            byte_offset: mdef.tag_offset,
                        });
                        let init_tag = self.fresh_val();
                        self.emit(IrInst::ConstInt {
                            dst: init_tag,
                            ty: IrType::Int(IntType::U16),
                            val: mdef.initial_state as i64,
                        });
                        self.emit(IrInst::StorePtr {
                            addr: tag_addr,
                            val: init_tag,
                            ty: IrType::Int(IntType::U16),
                            volatile: false,
                        });
                        // Call initial state's on_enter if present
                        if let Some(enter_fn) = mdef
                            .states
                            .get(mdef.initial_state as usize)
                            .and_then(|s| s.on_enter)
                        {
                            self.emit(IrInst::Call {
                                dst: None,
                                func: enter_fn,
                                args: vec![base_addr],
                            });
                        }
                        return Some((base_addr, IrType::Machine(mid)));
                    }
                }

                self.checker
                    .error("unsupported call target", expr.span);
                None
            }
            ExprKind::MethodCall {
                receiver,
                method,
                args,
            } => {
                // Check receiver type (handle machine & pool methods by address!)
                if let Some((mut raddr, mut rty, mut rmut, rvol)) = self.lower_lvalue_if_place_ext(receiver) {
                    if let IrType::Ref { mutable, inner } = rty {
                        let loaded = self.fresh_val();
                        self.emit(IrInst::LoadPtr {
                            dst: loaded,
                            addr: raddr,
                            ty: IrType::Ref {
                                mutable,
                                inner: inner.clone(),
                            },
                            volatile: rvol,
                        });
                        raddr = loaded;
                        rty = *inner;
                        rmut = mutable;
                    }
                    match rty {
                        IrType::Machine(mid) => {
                            let mdef = self.checker.program.machines[mid as usize].clone();
                            // Dynamic transition: `ball.goto(StateB { ... })` or `ball.goto(StateB)`
                            if method == "goto" {
                                if !rmut {
                                    self.checker.error("cannot call `.goto()` on immutable machine", expr.span);
                                    return None;
                                }
                                if mdef.strict_transitions {
                                    self.checker.error(
                                        format!("dynamic `.goto()` is forbidden on `#[strict_transitions]` machine `{}`", mdef.name),
                                        expr.span,
                                    );
                                    return None;
                                }
                                if args.len() != 1 {
                                    self.checker.error("`.goto()` expects 1 target state argument", expr.span);
                                    return None;
                                }
                                let (tname, tfields) = match &args[0].kind {
                                    ExprKind::Ident(n) => (n.as_str(), &[][..]),
                                    ExprKind::StructLit { name, fields } => (name.as_str(), fields.as_slice()),
                                    _ => {
                                        self.checker.error("expected state initializer in `.goto(...)`", args[0].span);
                                        return None;
                                    }
                                };
                                let Some(target_state) = mdef.states.iter().find(|s| s.name == tname).cloned() else {
                                    self.checker.error(
                                        format!("unknown state `{tname}` for machine `{}`", mdef.name),
                                        args[0].span,
                                    );
                                    return None;
                                };
                                let field_vals = self.lower_state_payload(&target_state, tfields, args[0].span)?;
                                let site_id = self.checker.program.intern_site(expr.span);
                                self.emit(IrInst::StateTransition {
                                    machine_addr: raddr,
                                    machine_id: mid,
                                    from_state: None,
                                    to_state: target_state.id,
                                    field_vals,
                                    site_id,
                                });
                                return Some((0, IrType::Unit));
                            }
                            // Shared machine method `Machine::method`
                            let shared_name = format!("{}::{method}", mdef.name);
                            if let Some(sig) = self.checker.fn_sigs.get(&shared_name).cloned() {
                                let mut call_args = vec![raddr];
                                for (arg, exp) in args.iter().zip(sig.params.iter().skip(1)) {
                                    let (av, _) = self.lower_expr(arg, Some(exp))?;
                                    call_args.push(av);
                                }
                                let dst = if sig.ret_ty == IrType::Unit {
                                    None
                                } else {
                                    Some(self.fresh_val())
                                };
                                self.emit(IrInst::Call {
                                    dst,
                                    func: sig.id,
                                    args: call_args,
                                });
                                return Some((dst.unwrap_or(0), sig.ret_ty));
                            }
                            // Dispatched state method
                            let mut disp_args = Vec::with_capacity(args.len());
                            for arg in args {
                                let (av, _) = self.lower_expr(arg, None)?;
                                disp_args.push(av);
                            }
                            self.emit(IrInst::StateDispatch {
                                dst: None,
                                machine_addr: raddr,
                                machine_id: mid,
                                method: method.clone(),
                                args: disp_args,
                            });
                            return Some((0, IrType::Unit));
                        }
                        IrType::Pool { elem, capacity } => {
                            let elem_size = self.checker.program.type_size(&elem);
                            match method.as_str() {
                                "alloc" => {
                                    if !rmut {
                                        self.checker.error("cannot call `.alloc()` on immutable pool", expr.span);
                                        return None;
                                    }
                                    if args.len() != 1 {
                                        self.checker.error("`pool.alloc(init)` expects 1 argument", expr.span);
                                        return None;
                                    }
                                    let (init_val, init_ty) = self.lower_expr(&args[0], Some(&elem))?;
                                    if init_ty != *elem {
                                        self.checker.error(
                                            format!(
                                                "`pool.alloc()` type mismatch: expected `{elem:?}`, found `{init_ty:?}`"
                                            ),
                                            args[0].span,
                                        );
                                        return None;
                                    }
                                    let init_addr = if init_ty.is_scalar() {
                                        let tmp = self.alloc_local("__pool_init".into(), init_ty.clone(), true);
                                        self.emit(IrInst::StoreLocal { slot: tmp, val: init_val, ty: init_ty });
                                        let a = self.fresh_val();
                                        self.emit(IrInst::AddrLocal { dst: a, slot: tmp });
                                        a
                                    } else {
                                        init_val
                                    };
                                    let dst = self.fresh_val();
                                    self.emit(IrInst::PoolAlloc {
                                        dst,
                                        pool_addr: raddr,
                                        init_addr,
                                        elem_size,
                                        capacity,
                                    });
                                    return Some((dst, IrType::Handle(elem)));
                                }
                                "free" => {
                                    if !rmut {
                                        self.checker.error("cannot call `.free()` on immutable pool", expr.span);
                                        return None;
                                    }
                                    if args.len() != 1 {
                                        self.checker.error("`pool.free(handle)` expects 1 argument", expr.span);
                                        return None;
                                    }
                                    let expected_h = IrType::Handle(elem);
                                    let (hval, hty) = self.lower_expr(&args[0], Some(&expected_h))?;
                                    if hty != expected_h {
                                        self.checker.error(
                                            format!(
                                                "`pool.free()` handle type mismatch: expected `{expected_h:?}`, found `{hty:?}`"
                                            ),
                                            args[0].span,
                                        );
                                        return None;
                                    }
                                    let site_id = self.checker.program.intern_site(expr.span);
                                    self.emit(IrInst::PoolFree {
                                        pool_addr: raddr,
                                        handle: hval,
                                        elem_size,
                                        capacity,
                                        site_id,
                                    });
                                    return Some((0, IrType::Unit));
                                }
                                "get" | "get_mut" => {
                                    let want_mut = method == "get_mut";
                                    if want_mut && !rmut {
                                        self.checker.error("cannot call `.get_mut()` on immutable pool", expr.span);
                                        return None;
                                    }
                                    if args.len() != 1 {
                                        self.checker.error("`pool.get(handle)` expects 1 argument", expr.span);
                                        return None;
                                    }
                                    let expected_h = IrType::Handle(elem.clone());
                                    let (hval, hty) = self.lower_expr(&args[0], Some(&expected_h))?;
                                    if hty != expected_h {
                                        self.checker.error(
                                            format!(
                                                "`pool.{method}()` handle type mismatch: expected `{expected_h:?}`, found `{hty:?}`"
                                            ),
                                            args[0].span,
                                        );
                                        return None;
                                    }
                                    let site_id = self.checker.program.intern_site(expr.span);
                                    let dst = self.fresh_val();
                                    self.emit(IrInst::PoolElemAddr {
                                        dst,
                                        pool_addr: raddr,
                                        handle: hval,
                                        elem_size,
                                        capacity,
                                        site_id,
                                    });
                                    return Some((
                                        dst,
                                        IrType::Ref {
                                            mutable: want_mut,
                                            inner: elem,
                                        },
                                    ));
                                }
                                _ => {}
                            }
                        }
                        _ => {}
                    }
                }

                let (rval, rty) = self.lower_expr(receiver, None)?;
                if method == "len" && args.is_empty() {
                    match &rty {
                        IrType::Array { len, .. } => {
                            let dst = self.fresh_val();
                            self.emit(IrInst::ConstInt {
                                dst,
                                ty: IrType::Int(IntType::U16),
                                val: *len as i64,
                            });
                            return Some((dst, IrType::Int(IntType::U16)));
                        }
                        IrType::Ref { inner, .. }
                            if matches!(inner.as_ref(), IrType::Array { .. }) =>
                        {
                            let IrType::Array { len, .. } = inner.as_ref() else {
                                unreachable!();
                            };
                            let dst = self.fresh_val();
                            self.emit(IrInst::ConstInt {
                                dst,
                                ty: IrType::Int(IntType::U16),
                                val: *len as i64,
                            });
                            return Some((dst, IrType::Int(IntType::U16)));
                        }
                        IrType::Slice { .. } | IrType::ConstSlice(_) => {
                            let sh = self.fresh_val();
                            self.emit(IrInst::ConstInt {
                                dst: sh,
                                ty: IrType::Int(IntType::U8),
                                val: 16,
                            });
                            let hi = self.fresh_val();
                            let site_id = self.checker.program.intern_site(expr.span);
                            self.emit(IrInst::BinOp {
                                dst: hi,
                                op: IrBinOp::Shr,
                                mode: ArithMode::Wrapping,
                                ty: IrType::Int(IntType::U32),
                                lhs: rval,
                                rhs: sh,
                                site_id,
                            });
                            let dst = self.fresh_val();
                            self.emit(IrInst::Cast {
                                dst,
                                val: hi,
                                from_ty: IrType::Int(IntType::U32),
                                to_ty: IrType::Int(IntType::U16),
                            });
                            return Some((dst, IrType::Int(IntType::U16)));
                        }
                        _ => {}
                    }
                }
                if let IrType::Fixed { base, frac_bits } = rty {
                    if method == "int" && args.is_empty() {
                        let dst = self.fresh_val();
                        self.emit(IrInst::FixedInt {
                            dst,
                            val: rval,
                            base,
                            frac_bits,
                        });
                        return Some((dst, IrType::Int(base)));
                    }
                    if method == "raw" && args.is_empty() {
                        return Some((rval, IrType::Int(base)));
                    }
                }
                self.checker.error(
                    format!("unknown method `{method}` on type `{rty:?}`"),
                    expr.span,
                );
                None
            }
            ExprKind::StructLit { name, fields } => {
                let Some(&sid) = self.checker.struct_ids.get(name) else {
                    self.checker
                        .error(format!("unknown struct `{name}`"), expr.span);
                    return None;
                };
                let sdef = self.checker.program.structs[sid as usize].clone();
                let tmp_slot = self.alloc_local(
                    format!("__struct_tmp_{sid}"),
                    IrType::Struct(sid),
                    true,
                );
                let base_addr = self.fresh_val();
                self.emit(IrInst::AddrLocal {
                    dst: base_addr,
                    slot: tmp_slot,
                });
                for (fname, fexpr) in fields {
                    let Some((_, fty, offset)) =
                        sdef.fields.iter().find(|(n, _, _)| n == fname).cloned()
                    else {
                        self.checker.error(
                            format!("struct `{name}` has no field `{fname}`"),
                            fexpr.span,
                        );
                        continue;
                    };
                    if let Some((fval, actual_ty)) = self.lower_expr(fexpr, Some(&fty)) {
                        if actual_ty != fty {
                            self.checker.error(
                                format!(
                                    "field `{fname}` type mismatch: expected `{fty:?}`, found `{actual_ty:?}`"
                                ),
                                fexpr.span,
                            );
                        }
                        let faddr = self.fresh_val();
                        self.emit(IrInst::FieldAddr {
                            dst: faddr,
                            base: base_addr,
                            byte_offset: offset,
                        });
                        self.emit(IrInst::StorePtr {
                            addr: faddr,
                            val: fval,
                            ty: fty,
                            volatile: false,
                        });
                    }
                }
                Some((base_addr, IrType::Struct(sid)))
            }
            ExprKind::ArrayLit(elems) => {
                let expected_elem = match expected {
                    Some(IrType::Array { elem, .. }) => Some(elem.as_ref().clone()),
                    _ => None,
                };
                let len = elems.len() as u16;
                let mut lowered = Vec::with_capacity(elems.len());
                let mut elem_ty = expected_elem.clone().unwrap_or(IrType::Int(IntType::U8));
                for (i, e) in elems.iter().enumerate() {
                    let (v, t) = self.lower_expr(e, if i == 0 { expected_elem.as_ref() } else { Some(&elem_ty) })?;
                    if i == 0 {
                        elem_ty = t;
                    }
                    lowered.push(v);
                }
                let arr_ty = IrType::Array {
                    elem: Box::new(elem_ty.clone()),
                    len,
                };
                let tmp_slot = self.alloc_local("__arr_tmp".into(), arr_ty.clone(), true);
                let base_addr = self.fresh_val();
                self.emit(IrInst::AddrLocal {
                    dst: base_addr,
                    slot: tmp_slot,
                });
                let esize = self.checker.program.type_size(&elem_ty);
                for (i, v) in lowered.into_iter().enumerate() {
                    let eaddr = self.fresh_val();
                    self.emit(IrInst::FieldAddr {
                        dst: eaddr,
                        base: base_addr,
                        byte_offset: (i as u16) * esize,
                    });
                    self.emit(IrInst::StorePtr {
                        addr: eaddr,
                        val: v,
                        ty: elem_ty.clone(),
                        volatile: false,
                    });
                }
                Some((base_addr, arr_ty))
            }
            ExprKind::ArrayRepeat { val, len } => {
                let expected_elem = match expected {
                    Some(IrType::Array { elem, .. }) => Some(elem.as_ref().clone()),
                    _ => None,
                };
                let (v, elem_ty) = self.lower_expr(val, expected_elem.as_ref())?;
                let arr_ty = IrType::Array {
                    elem: Box::new(elem_ty.clone()),
                    len: *len,
                };
                let tmp_slot = self.alloc_local("__arr_rep".into(), arr_ty.clone(), true);
                let base_addr = self.fresh_val();
                self.emit(IrInst::AddrLocal {
                    dst: base_addr,
                    slot: tmp_slot,
                });
                let esize = self.checker.program.type_size(&elem_ty);
                for i in 0..*len {
                    let eaddr = self.fresh_val();
                    self.emit(IrInst::FieldAddr {
                        dst: eaddr,
                        base: base_addr,
                        byte_offset: i * esize,
                    });
                    self.emit(IrInst::StorePtr {
                        addr: eaddr,
                        val: v,
                        ty: elem_ty.clone(),
                        volatile: false,
                    });
                }
                Some((base_addr, arr_ty))
            }
            ExprKind::If {
                cond,
                then_block,
                else_branch,
            } => {
                let then_bb = self.new_block();
                let else_bb = self.new_block();
                let join_bb = self.new_block();
                self.lower_cond(cond, then_bb, else_bb, "`if` condition must be `bool`", cond.span);

                self.cur_bb = then_bb;
                let then_val = self.lower_block(then_block, expected);
                let then_end_bb = self.cur_bb;
                let then_falls_through = !self.is_terminated();

                let res_ty = then_val
                    .as_ref()
                    .map(|(_, t)| t.clone())
                    .unwrap_or(IrType::Unit);
                let res_slot = if res_ty != IrType::Unit && else_branch.is_some() {
                    Some(self.alloc_local("__if_tmp".into(), res_ty.clone(), true))
                } else {
                    None
                };

                if then_falls_through {
                    self.cur_bb = then_end_bb;
                    if let (Some(slot), Some((v, _))) = (res_slot, then_val) {
                        self.emit(IrInst::StoreLocal {
                            slot,
                            val: v,
                            ty: res_ty.clone(),
                        });
                    }
                    self.set_term(Terminator::Goto(join_bb));
                }

                self.cur_bb = else_bb;
                let else_val = else_branch
                    .as_ref()
                    .and_then(|eb| self.lower_expr(eb, Some(&res_ty)));
                if !self.is_terminated() {
                    if let (Some(slot), Some((v, _))) = (res_slot, else_val) {
                        self.emit(IrInst::StoreLocal {
                            slot,
                            val: v,
                            ty: res_ty.clone(),
                        });
                    }
                    self.set_term(Terminator::Goto(join_bb));
                }

                self.cur_bb = join_bb;
                if let Some(slot) = res_slot {
                    let dst = self.fresh_val();
                    self.emit(IrInst::LoadLocal {
                        dst,
                        slot,
                        ty: res_ty.clone(),
                    });
                    Some((dst, res_ty))
                } else {
                    Some((0, IrType::Unit))
                }
            }
            ExprKind::Block(block) => self.lower_block(block, expected),
            ExprKind::WithBank { banks, body } => {
                let target = self.checker.program.target.clone();
                let mut found_group: Option<(String, bool)> = None;
                if let Some(t) = target {
                    for bg in &t.bank_groups {
                        if banks.iter().all(|b| bg.banks.iter().any(|d| &d.name == b)) {
                            // Check duplicate or overlapping banks in combination (e.g. HighMainRam + VramAlu)
                            let mut resolved_defs: Vec<&BankDef> = Vec::with_capacity(banks.len());
                            let mut seen_b = std::collections::HashSet::new();
                            for b in banks {
                                if !seen_b.insert(b.as_str()) {
                                    self.checker.error(
                                        format!("duplicate bank `{b}` in `with bank(...)`"),
                                        expr.span,
                                    );
                                }
                                if let Some(def) = bg.banks.iter().find(|d| &d.name == b) {
                                    resolved_defs.push(def);
                                }
                            }
                            for i in 0..resolved_defs.len() {
                                for j in (i + 1)..resolved_defs.len() {
                                    if resolved_defs[i].overlaps(resolved_defs[j]) {
                                        self.checker.error(
                                            format!(
                                                "overlapping banks `{}` and `{}` cannot be mapped simultaneously in `{}`",
                                                resolved_defs[i].name, resolved_defs[j].name, bg.name
                                            ),
                                            expr.span,
                                        );
                                    }
                                }
                            }
                            let mut sorted_req = banks.clone();
                            sorted_req.sort();
                            let has_switch = bg.switches.iter().any(|(combo, _)| {
                                let mut s = combo.clone();
                                s.sort();
                                s == sorted_req
                            });
                            if !has_switch {
                                self.checker.error(
                                    format!(
                                        "bank combination `{}` has no matching `switch` arm in `{}`",
                                        banks.join(" + "),
                                        bg.name
                                    ),
                                    expr.span,
                                );
                            }
                            let req_di = resolved_defs.iter().any(|d| d.requires_di);
                            found_group = Some((bg.name.clone(), req_di));
                            break;
                        }
                    }
                }
                let Some((group_name, req_di)) = found_group else {
                    self.checker.error(
                        format!(
                            "unknown bank combination `{}` in `with bank(...)`",
                            banks.join(" + ")
                        ),
                        expr.span,
                    );
                    return self.lower_block(body, expected);
                };

                let prev_banks = self
                    .active_banks
                    .get(&group_name)
                    .cloned()
                    .unwrap_or_default();
                let prev_di = self.irq_disabled;
                let needs_di = req_di && !prev_di;

                self.emit(IrInst::BankEnter {
                    group: group_name.clone(),
                    banks: banks.clone(),
                    needs_di,
                });
                self.active_banks.insert(group_name.clone(), banks.clone());
                if req_di {
                    self.irq_disabled = true;
                }
                self.bank_scope_stack.push(BankScopeEntry {
                    group: group_name.clone(),
                    prev_banks: prev_banks.clone(),
                    restore_ei: needs_di,
                    loop_depth: self.loop_stack.len(),
                });

                let res = self.lower_block(body, expected);

                self.bank_scope_stack.pop();
                self.active_banks.insert(group_name.clone(), prev_banks.clone());
                self.irq_disabled = prev_di;
                if !self.is_terminated() {
                    self.emit(IrInst::BankExit {
                        group: group_name,
                        restore_banks: prev_banks,
                        restore_ei: needs_di,
                    });
                }
                res.or(Some((0, IrType::Unit)))
            }
            ExprKind::Goto { state_name, fields } => {
                let Some((mid, Some(cur_sid), is_hook)) = self.machine_ctx else {
                    self.checker.error(
                        "`goto` can only be used inside an `on <State>` handler block",
                        expr.span,
                    );
                    return None;
                };
                if is_hook {
                    self.checker.error(
                        "`goto` is not permitted directly inside `on_enter` or `on_exit`",
                        expr.span,
                    );
                    return None;
                }
                let mdef = self.checker.program.machines[mid as usize].clone();
                let Some(target) = mdef.states.iter().find(|s| s.name == *state_name).cloned() else {
                    self.checker.error(
                        format!("unknown target state `{state_name}` in machine `{}`", mdef.name),
                        expr.span,
                    );
                    return None;
                };
                // 100% static transition verification!
                if !mdef.is_transition_allowed(cur_sid, target.id) {
                    let from_name = &mdef.states[cur_sid as usize].name;
                    self.checker.error(
                        format!(
                            "forbidden static state transition `{}: {from_name} -> {state_name}`",
                            mdef.name
                        ),
                        expr.span,
                    );
                    return None;
                }
                let field_vals = self.lower_state_payload(&target, fields, expr.span)?;
                let self_var = self.lookup_var("self")?;
                if !matches!(self_var.ty, IrType::Ref { mutable: true, .. }) {
                    self.checker.error(
                        "`goto` requires `&mut self` receiver",
                        expr.span,
                    );
                    return None;
                }
                let self_ptr = self.fresh_val();
                self.emit(IrInst::LoadLocal {
                    dst: self_ptr,
                    slot: self_var.slot,
                    ty: self_var.ty,
                });
                let site_id = self.checker.program.intern_site(expr.span);
                self.unwind_bank_scopes(0);
                self.emit(IrInst::StateTransition {
                    machine_addr: self_ptr,
                    machine_id: mid,
                    from_state: Some(cur_sid),
                    to_state: target.id,
                    field_vals,
                    site_id,
                });
                // AD-11: `goto` is a diverging expression (`!`) — immediately exit handler!
                self.set_term(Terminator::Return(None));
                Some((0, IrType::Never))
            }
            ExprKind::IsState {
                expr: target_expr,
                machine_name,
                state_name,
            } => {
                let Some(&mid) = self.checker.machine_ids.get(machine_name) else {
                    self.checker.error(
                        format!("unknown machine `{machine_name}` in `is_state!`"),
                        expr.span,
                    );
                    return None;
                };
                let Some(sid) = self.checker.program.machines[mid as usize]
                    .states
                    .iter()
                    .find(|s| s.name == *state_name)
                    .map(|s| s.id)
                else {
                    self.checker.error(
                        format!("unknown state `{state_name}` in machine `{machine_name}`"),
                        expr.span,
                    );
                    return None;
                };
                let (mut maddr, mut mty, _) = self.lower_lvalue(target_expr)?;
                if let IrType::Ref { mutable, inner } = mty {
                    let loaded = self.fresh_val();
                    self.emit(IrInst::LoadPtr {
                        dst: loaded,
                        addr: maddr,
                        ty: IrType::Ref {
                            mutable,
                            inner: inner.clone(),
                        },
                        volatile: false,
                    });
                    maddr = loaded;
                    mty = *inner;
                }
                if mty != IrType::Machine(mid) {
                    self.checker.error(
                        format!(
                            "`is_state!` target has type `{mty:?}`, expected machine `{machine_name}`"
                        ),
                        target_expr.span,
                    );
                    return None;
                }
                let tag = self.fresh_val();
                self.emit(IrInst::StateTag {
                    dst: tag,
                    machine_addr: maddr,
                    machine_id: mid,
                });
                let expected_tag = self.fresh_val();
                self.emit(IrInst::ConstInt {
                    dst: expected_tag,
                    ty: IrType::Int(IntType::U8),
                    val: sid as i64,
                });
                let dst = self.fresh_val();
                let site_id = self.checker.program.intern_site(expr.span);
                self.emit(IrInst::BinOp {
                    dst,
                    op: IrBinOp::Eq,
                    mode: ArithMode::Checked,
                    ty: IrType::Int(IntType::U8),
                    lhs: tag,
                    rhs: expected_tag,
                    site_id,
                });
                Some((dst, IrType::Bool))
            }
            ExprKind::Assert { cond } => {
                let (cval, cty) = self.lower_expr(cond, Some(&IrType::Bool))?;
                if cty != IrType::Bool {
                    self.checker
                        .error("`assert!` condition must be `bool`", cond.span);
                }
                let site_id = self.checker.program.intern_site(expr.span);
                self.emit(IrInst::Assert {
                    cond: cval,
                    site_id,
                });
                Some((0, IrType::Unit))
            }
            ExprKind::InlineAsm { template, clobbers } => {
                self.emit(IrInst::InlineAsm {
                    template: template.clone(),
                    clobbers: clobbers.clone(),
                });
                Some((0, IrType::Unit))
            }
        }
    }
}
