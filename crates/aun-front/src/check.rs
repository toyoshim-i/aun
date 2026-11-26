//! Type checker, Second-Class Reference borrow checker, and AST-to-IR lowering.

use crate::ast::*;
use crate::diag::Diagnostic;
use aun_ir::*;
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct FnSig {
    id: u32,
    is_extern: bool,
    params: Vec<IrType>,
    ret_ty: IrType,
}

#[derive(Debug, Clone)]
struct VarInfo {
    slot: SlotId,
    ty: IrType,
    mutable: bool,
}

/// Place path used for second-class alias/borrow checking in call arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
struct BorrowPlace {
    root: String,
    fields: Vec<String>,
    mutable: bool,
}

impl BorrowPlace {
    /// Returns true if `self` and `other` overlap and at least one is mutable.
    fn conflicts_with(&self, other: &BorrowPlace) -> bool {
        if !(self.mutable || other.mutable) {
            return false;
        }
        if self.root != other.root {
            return false;
        }
        // Prefix overlap: e.g. `s` overlaps `s.x`, `s.x` overlaps `s.x`,
        // but `s.x` and `s.y` are disjoint!
        let min_len = self.fields.len().min(other.fields.len());
        self.fields[..min_len] == other.fields[..min_len]
    }
}

pub struct Checker {
    program: IrProgram,
    diagnostics: Vec<Diagnostic>,
    struct_ids: HashMap<String, u32>,
    enums: HashMap<String, (IntType, HashMap<String, i64>)>,
    consts: HashMap<String, (IrType, i64)>,
    global_ids: HashMap<String, (u32, IrType, bool)>,
    fn_sigs: HashMap<String, FnSig>,
}

impl Checker {
    pub fn check_and_lower(file: &File) -> Result<IrProgram, Vec<Diagnostic>> {
        let mut checker = Self {
            program: IrProgram::default(),
            diagnostics: Vec::new(),
            struct_ids: HashMap::new(),
            enums: HashMap::new(),
            consts: HashMap::new(),
            global_ids: HashMap::new(),
            fn_sigs: HashMap::new(),
        };
        checker.collect_items(file);
        checker.lower_functions(file);
        if checker.diagnostics.is_empty() {
            Ok(checker.program)
        } else {
            Err(checker.diagnostics)
        }
    }

    fn error(&mut self, msg: impl Into<String>, span: Span) {
        self.diagnostics.push(Diagnostic::error(msg, span));
    }

    fn resolve_type(&mut self, ty: &TypeExpr, span: Span, allow_ref: bool) -> IrType {
        match ty {
            TypeExpr::Unit => IrType::Unit,
            TypeExpr::Never => IrType::Never,
            TypeExpr::Named(name) => match name.as_str() {
                "bool" => IrType::Bool,
                "u8" => IrType::Int(IntType::U8),
                "i8" => IrType::Int(IntType::I8),
                "u16" => IrType::Int(IntType::U16),
                "i16" => IrType::Int(IntType::I16),
                "u32" => IrType::Int(IntType::U32),
                "i32" => IrType::Int(IntType::I32),
                "q8_8" => IrType::Fixed {
                    base: IntType::I16,
                    frac_bits: 8,
                },
                "q10_6" => IrType::Fixed {
                    base: IntType::I16,
                    frac_bits: 6,
                },
                other => {
                    if let Some(&id) = self.struct_ids.get(other) {
                        IrType::Struct(id)
                    } else if let Some(&(repr, _)) = self.enums.get(other) {
                        IrType::Int(repr)
                    } else {
                        self.error(format!("unknown type `{other}`"), span);
                        IrType::Unit
                    }
                }
            },
            TypeExpr::Fixed { base, frac_bits } => {
                let base_ty = self.resolve_type(base, span, false);
                match base_ty {
                    IrType::Int(it) => {
                        if *frac_bits == 0 || *frac_bits >= it.bit_width() {
                            self.error(
                                format!(
                                    "invalid fractional bits `{frac_bits}` for `{it}` (must be 1..{})",
                                    it.bit_width()
                                ),
                                span,
                            );
                        }
                        IrType::Fixed {
                            base: it,
                            frac_bits: *frac_bits,
                        }
                    }
                    _ => {
                        self.error("`fixed<Base, Frac>` requires an integer Base type", span);
                        IrType::Unit
                    }
                }
            }
            TypeExpr::Ref { mutable, inner } => {
                if !allow_ref {
                    self.error(
                        "second-class references (`&T` / `&mut T`) cannot be stored in structs, statics, or returned from functions (AD-3)",
                        span,
                    );
                }
                let inner_ty = self.resolve_type(inner, span, false);
                IrType::Ref {
                    mutable: *mutable,
                    inner: Box::new(inner_ty),
                }
            }
            TypeExpr::ConstSlice(inner) => {
                let inner_ty = self.resolve_type(inner, span, false);
                IrType::ConstSlice(Box::new(inner_ty))
            }
            TypeExpr::Slice { mutable, inner } => {
                if !allow_ref {
                    self.error(
                        "second-class slices (`&[T]` / `&mut [T]`) cannot be stored in structs, statics, or returned (AD-3)",
                        span,
                    );
                }
                let inner_ty = self.resolve_type(inner, span, false);
                IrType::Ref {
                    mutable: *mutable,
                    inner: Box::new(inner_ty),
                }
            }
            TypeExpr::Array { elem, len } => {
                let elem_ty = self.resolve_type(elem, span, false);
                IrType::Array {
                    elem: Box::new(elem_ty),
                    len: *len,
                }
            }
        }
    }

    fn collect_items(&mut self, file: &File) {
        // Pass 1: register struct names and enums
        for item in &file.items {
            match item {
                Item::Struct(s) => {
                    let id = self.program.structs.len() as u32;
                    self.struct_ids.insert(s.name.clone(), id);
                    self.program.structs.push(StructDef {
                        name: s.name.clone(),
                        fields: Vec::new(),
                        size: 0,
                    });
                }
                Item::Enum(e) => {
                    let repr = match &e.repr {
                        Some(t) => match self.resolve_type(t, e.span, false) {
                            IrType::Int(it) => it,
                            _ => {
                                self.error("enum repr must be an integer type", e.span);
                                IntType::U8
                            }
                        },
                        None => IntType::U8,
                    };
                    let mut map = HashMap::new();
                    let mut next_val = 0i64;
                    for (vname, explicit, _) in &e.variants {
                        let v = explicit.map(|x| x as i64).unwrap_or(next_val);
                        map.insert(vname.clone(), v);
                        next_val = v + 1;
                    }
                    self.enums.insert(e.name.clone(), (repr, map));
                }
                _ => {}
            }
        }

        // Pass 2: resolve struct field layouts (forbid references in struct fields!)
        for item in &file.items {
            if let Item::Struct(s) = item {
                let id = self.struct_ids[&s.name];
                let mut offset = 0u16;
                let mut fields = Vec::new();
                for (fname, fty, fspan) in &s.fields {
                    let ir_ty = self.resolve_type(fty, *fspan, false);
                    let fsize = ir_ty.byte_size(&self.program.structs);
                    fields.push((fname.clone(), ir_ty, offset));
                    offset += fsize;
                }
                let def = &mut self.program.structs[id as usize];
                def.fields = fields;
                def.size = offset;
            }
        }

        // Pass 3: consts, statics, extern "asm", and function signatures
        for item in &file.items {
            match item {
                Item::Const(c) => {
                    let ty = self.resolve_type(&c.ty, c.span, false);
                    if let Some(v) = self.eval_const_expr(&c.value, &ty) {
                        self.consts.insert(c.name.clone(), (ty, v));
                    } else {
                        self.error("const initializer must be a compile-time constant", c.span);
                    }
                }
                Item::Static(st) => {
                    let ty = self.resolve_type(&st.ty, st.span, false);
                    let size = ty.byte_size(&self.program.structs) as usize;
                    let mut init = vec![0u8; size];
                    if let Some(init_expr) = &st.init
                        && let Some(val) = self.eval_const_expr(init_expr, &ty)
                    {
                        let bytes = val.to_le_bytes();
                        for (dst, src) in init.iter_mut().zip(bytes.iter()) {
                            *dst = *src;
                        }
                    }
                    let id = self.program.globals.len() as u32;
                    self.global_ids
                        .insert(st.name.clone(), (id, ty.clone(), st.is_mut));
                    self.program.globals.push(IrGlobal {
                        name: st.name.clone(),
                        ty,
                        mutable: st.is_mut,
                        align: 1,
                        space: st.space.clone(),
                        init,
                    });
                }
                Item::ExternAsm(block) => {
                    for ef in &block.fns {
                        let mut params = Vec::new();
                        let mut sig_params = Vec::new();
                        for p in &ef.params {
                            let pty = self.resolve_type(&p.ty, p.span, true);
                            sig_params.push(pty.clone());
                            params.push((p.name.clone(), pty, p.reg.clone()));
                        }
                        let (ret, ret_ty) = match &ef.ret {
                            Some((rty, reg)) => {
                                let t = self.resolve_type(rty, ef.span, false);
                                (Some((t.clone(), reg.clone())), t)
                            }
                            None => (None, IrType::Unit),
                        };
                        let id = self.program.externs.len() as u32;
                        self.fn_sigs.insert(
                            ef.name.clone(),
                            FnSig {
                                id,
                                is_extern: true,
                                params: sig_params,
                                ret_ty,
                            },
                        );
                        self.program.externs.push(IrExternFn {
                            name: ef.name.clone(),
                            params,
                            ret,
                            clobbers: ef.clobbers.clone(),
                            requires_di: ef.requires_di,
                            requires_banks: ef.requires_banks.clone(),
                        });
                    }
                }
                Item::Fn(f) => {
                    let mut sig_params = Vec::new();
                    for p in &f.params {
                        sig_params.push(self.resolve_type(&p.ty, p.span, true));
                    }
                    let ret_ty = match &f.ret_ty {
                        Some(t) => self.resolve_type(t, f.span, false),
                        None => IrType::Unit,
                    };
                    let id = self.program.functions.len() as u32;
                    self.fn_sigs.insert(
                        f.name.clone(),
                        FnSig {
                            id,
                            is_extern: false,
                            params: sig_params.clone(),
                            ret_ty: ret_ty.clone(),
                        },
                    );

                    let is_test = f.attrs.iter().any(|a| a.name == "test");
                    let is_reentrant = f.attrs.iter().any(|a| a.name == "reentrant");
                    let expected_panic = f
                        .attrs
                        .iter()
                        .find(|a| a.name == "should_panic")
                        .and_then(|a| a.args.first())
                        .and_then(|name| PanicCode::from_name(name));

                    let locals: Vec<LocalSlot> = f
                        .params
                        .iter()
                        .zip(sig_params.into_iter())
                        .enumerate()
                        .map(|(i, (p, ty))| LocalSlot {
                            id: i as SlotId,
                            name: p.name.clone(),
                            ty,
                            mutable: false,
                        })
                        .collect();

                    self.program.functions.push(IrFunction {
                        id,
                        name: f.name.clone(),
                        param_count: locals.len(),
                        locals,
                        ret_ty,
                        is_test,
                        expected_panic,
                        is_reentrant,
                        blocks: vec![BasicBlock::default()],
                        next_val: 0,
                    });
                }
                _ => {}
            }
        }
    }

    fn eval_const_expr(&mut self, expr: &Expr, expected_ty: &IrType) -> Option<i64> {
        match &expr.kind {
            ExprKind::IntLit { value, .. } => Some(*value as i64),
            ExprKind::BoolLit(b) => Some(if *b { 1 } else { 0 }),
            ExprKind::ByteLit(b) => Some(*b as i64),
            ExprKind::FixedLit { text, .. } => {
                if let IrType::Fixed { base, frac_bits } = expected_ty {
                    self.parse_fixed_scaled(text, *base, *frac_bits, expr.span)
                } else {
                    None
                }
            }
            ExprKind::Unary {
                op: UnaryOp::Neg,
                expr: inner,
            } => self.eval_const_expr(inner, expected_ty).map(|v| -v),
            _ => None,
        }
    }

    fn parse_fixed_scaled(
        &mut self,
        text: &str,
        base: IntType,
        frac_bits: u8,
        span: Span,
    ) -> Option<i64> {
        let val: f64 = text.parse().ok()?;
        let scaled = (val * ((1u64 << frac_bits) as f64)).round() as i64;
        if scaled < base.min_i64() || scaled > base.max_i64() {
            self.error(
                format!("fixed-point literal `{text}` out of range for `fixed<{base}, {frac_bits}>`"),
                span,
            );
            return None;
        }
        Some(scaled)
    }

    fn lower_functions(&mut self, file: &File) {
        let mut fn_idx = 0usize;
        for item in &file.items {
            if let Item::Fn(f) = item {
                let mut lowerer = FnLowerer::new(self, fn_idx);
                lowerer.lower_fn(f);
                fn_idx += 1;
            }
        }
    }
}

struct FnLowerer<'a> {
    checker: &'a mut Checker,
    fn_idx: usize,
    cur_bb: BlockId,
    scopes: Vec<HashMap<String, VarInfo>>,
    loop_stack: Vec<(BlockId, BlockId)>, // (continue_bb, break_bb)
}

impl<'a> FnLowerer<'a> {
    fn new(checker: &'a mut Checker, fn_idx: usize) -> Self {
        let mut root_scope = HashMap::new();
        for slot in &checker.program.functions[fn_idx].locals {
            root_scope.insert(
                slot.name.clone(),
                VarInfo {
                    slot: slot.id,
                    ty: slot.ty.clone(),
                    mutable: slot.mutable,
                },
            );
        }
        Self {
            checker,
            fn_idx,
            cur_bb: 0,
            scopes: vec![root_scope],
            loop_stack: Vec::new(),
        }
    }

    fn func(&self) -> &IrFunction {
        &self.checker.program.functions[self.fn_idx]
    }

    fn func_mut(&mut self) -> &mut IrFunction {
        &mut self.checker.program.functions[self.fn_idx]
    }

    fn fresh_val(&mut self) -> ValueId {
        let f = self.func_mut();
        let v = f.next_val;
        f.next_val += 1;
        v
    }

    fn new_block(&mut self) -> BlockId {
        let f = self.func_mut();
        let id = f.blocks.len() as BlockId;
        f.blocks.push(BasicBlock {
            id,
            insts: Vec::new(),
            term: Terminator::Unreachable,
        });
        id
    }

    fn emit(&mut self, inst: IrInst) {
        let bb = self.cur_bb as usize;
        self.func_mut().blocks[bb].insts.push(inst);
    }

    fn set_term(&mut self, term: Terminator) {
        let bb = self.cur_bb as usize;
        if matches!(self.func().blocks[bb].term, Terminator::Unreachable) {
            self.func_mut().blocks[bb].term = term;
        }
    }

    fn is_terminated(&self) -> bool {
        !matches!(
            self.func().blocks[self.cur_bb as usize].term,
            Terminator::Unreachable
        )
    }

    fn lookup_var(&self, name: &str) -> Option<VarInfo> {
        for scope in self.scopes.iter().rev() {
            if let Some(v) = scope.get(name) {
                return Some(v.clone());
            }
        }
        None
    }

    fn alloc_local(&mut self, name: String, ty: IrType, mutable: bool) -> SlotId {
        let f = self.func_mut();
        let id = f.locals.len() as SlotId;
        f.locals.push(LocalSlot {
            id,
            name: name.clone(),
            ty: ty.clone(),
            mutable,
        });
        self.scopes.last_mut().unwrap().insert(
            name,
            VarInfo {
                slot: id,
                ty,
                mutable,
            },
        );
        id
    }

    fn lower_fn(&mut self, f: &FnDecl) {
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

    fn lower_block(&mut self, block: &Block, expected: Option<&IrType>) -> Option<(ValueId, IrType)> {
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
                let Some((val, init_ty)) = self.lower_expr(init, declared.as_ref()) else {
                    return;
                };
                let final_ty = declared.unwrap_or(init_ty.clone());
                if final_ty != init_ty && init_ty != IrType::Never {
                    self.checker.error(
                        format!("type mismatch in `let {name}`: expected `{final_ty:?}`, found `{init_ty:?}`"),
                        *span,
                    );
                }
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
                let Some((addr, target_ty, is_mut)) = self.lower_lvalue(target) else {
                    return;
                };
                if !is_mut {
                    self.checker.error("cannot assign to immutable place", *span);
                    return;
                }
                let Some((rhs_val, rhs_ty)) = self.lower_expr(value, Some(&target_ty)) else {
                    return;
                };
                if rhs_ty != target_ty && rhs_ty != IrType::Never {
                    self.checker.error(
                        format!("type mismatch in assignment: expected `{target_ty:?}`, found `{rhs_ty:?}`"),
                        *span,
                    );
                    return;
                }
                let final_val = if *op == AssignOp::Assign {
                    rhs_val
                } else {
                    let cur = self.fresh_val();
                    self.emit(IrInst::LoadPtr {
                        dst: cur,
                        addr,
                        ty: target_ty.clone(),
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
                self.emit(IrInst::StorePtr {
                    addr,
                    val: final_val,
                    ty: target_ty,
                });
            }
            Stmt::Expr(expr) => {
                self.lower_expr(expr, None);
            }
            Stmt::Return(expr, span) => {
                let ret_ty = self.func().ret_ty.clone();
                match expr {
                    Some(e) => {
                        if let Some((val, ty)) = self.lower_expr(e, Some(&ret_ty)) {
                            if ty != ret_ty && ty != IrType::Never {
                                self.checker.error(
                                    format!("return type mismatch: expected `{ret_ty:?}`, found `{ty:?}`"),
                                    *span,
                                );
                            }
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
                        self.set_term(Terminator::Return(None));
                    }
                }
            }
            Stmt::Break(span) => {
                if let Some(&(_, break_bb)) = self.loop_stack.last() {
                    self.set_term(Terminator::Goto(break_bb));
                } else {
                    self.checker.error("`break` outside of loop", *span);
                }
            }
            Stmt::Continue(span) => {
                if let Some(&(cont_bb, _)) = self.loop_stack.last() {
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
                if let Some((cval, cty)) = self.lower_expr(cond, Some(&IrType::Bool)) {
                    if cty != IrType::Bool {
                        self.checker.error("`while` condition must be `bool`", *span);
                    }
                    self.set_term(Terminator::Branch {
                        cond: cval,
                        then_bb: body_bb,
                        else_bb: exit_bb,
                    });
                }

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

    fn emit_aggregate_copy_to_local(&mut self, dst_slot: SlotId, src_addr: ValueId, ty: &IrType) {
        let size = ty.byte_size(&self.checker.program.structs);
        let dst_base = self.fresh_val();
        self.emit(IrInst::AddrLocal {
            dst: dst_base,
            slot: dst_slot,
        });
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
            });
            self.emit(IrInst::StorePtr {
                addr: d_addr,
                val: byte,
                ty: IrType::Int(IntType::U8),
            });
        }
    }

    /// Lowers an addressable expression to `(addr_val, element_ty, is_mutable)`.
    fn lower_lvalue(&mut self, expr: &Expr) -> Option<(ValueId, IrType, bool)> {
        match &expr.kind {
            ExprKind::Ident(name) => {
                if let Some(v) = self.lookup_var(name) {
                    let addr = self.fresh_val();
                    self.emit(IrInst::AddrLocal {
                        dst: addr,
                        slot: v.slot,
                    });
                    Some((addr, v.ty, v.mutable))
                } else if let Some(&(gid, ref gty, gmut)) = self.checker.global_ids.get(name) {
                    let gty = gty.clone();
                    let addr = self.fresh_val();
                    self.emit(IrInst::AddrGlobal {
                        dst: addr,
                        global: gid,
                    });
                    Some((addr, gty, gmut))
                } else {
                    self.checker
                        .error(format!("undefined variable `{name}`"), expr.span);
                    None
                }
            }
            ExprKind::Deref(inner) => {
                let (ptr_val, ptr_ty) = self.lower_expr(inner, None)?;
                match ptr_ty {
                    IrType::Ref { mutable, inner } => Some((ptr_val, *inner, mutable)),
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
                let (mut base_addr, mut base_ty, mut is_mut) = self.lower_lvalue(base)?;
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
                    });
                    base_addr = loaded;
                    base_ty = *inner;
                    is_mut = mutable;
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
                            Some((faddr, fty, is_mut))
                        } else {
                            self.checker.error(
                                format!("struct `{}` has no field `{field}`", sdef.name),
                                expr.span,
                            );
                            None
                        }
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
                let (mut base_addr, mut base_ty, mut is_mut) = self.lower_lvalue(base)?;
                if let IrType::Ref { mutable, inner } = base_ty {
                    let loaded = self.fresh_val();
                    self.emit(IrInst::LoadPtr {
                        dst: loaded,
                        addr: base_addr,
                        ty: IrType::Ref {
                            mutable,
                            inner: inner.clone(),
                        },
                    });
                    base_addr = loaded;
                    base_ty = *inner;
                    is_mut = mutable;
                }
                match base_ty {
                    IrType::Array { elem, len } => {
                        let (idx_val, idx_ty) = self.lower_expr(index, None)?;
                        if !matches!(idx_ty, IrType::Int(_)) {
                            self.checker.error("array index must be an integer", index.span);
                            return None;
                        }
                        let elem_size = elem.byte_size(&self.checker.program.structs);
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
                        Some((dst, *elem, is_mut))
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

    fn extract_borrow_place(expr: &Expr) -> Option<BorrowPlace> {
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

    fn lower_expr(&mut self, expr: &Expr, expected: Option<&IrType>) -> Option<(ValueId, IrType)> {
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
            ExprKind::StrLit(_) => {
                let dst = self.fresh_val();
                self.emit(IrInst::ConstInt {
                    dst,
                    ty: IrType::Int(IntType::U16),
                    val: 0,
                });
                Some((dst, IrType::ConstSlice(Box::new(IrType::Int(IntType::U8)))))
            }
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
                    let addr = self.fresh_val();
                    self.emit(IrInst::AddrGlobal {
                        dst: addr,
                        global: gid,
                    });
                    if gty.is_scalar() {
                        let dst = self.fresh_val();
                        self.emit(IrInst::LoadPtr {
                            dst,
                            addr,
                            ty: gty.clone(),
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
                    });
                    Some((dst, *elem_ty))
                } else {
                    Some((ptr_val, *elem_ty))
                }
            }
            ExprKind::Field { .. } | ExprKind::Index { .. } => {
                let (addr, ty, _) = self.lower_lvalue(expr)?;
                if ty.is_scalar() {
                    let dst = self.fresh_val();
                    self.emit(IrInst::LoadPtr {
                        dst,
                        addr,
                        ty: ty.clone(),
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
                        if &aty != expected_ty {
                            self.checker.error(
                                format!(
                                    "argument type mismatch in call to `{name}`: expected `{expected_ty:?}`, found `{aty:?}`"
                                ),
                                arg.span,
                            );
                        }
                        arg_vals.push(aval);
                    }
                    let dst = if sig.ret_ty == IrType::Unit {
                        None
                    } else {
                        Some(self.fresh_val())
                    };
                    if sig.is_extern {
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

                // Fixed-point constructor path: q8_8::from_raw(x) / q10_6::from_raw(x)
                if let ExprKind::Path(segs) = &callee.kind
                    && segs.len() == 2
                    && segs[1] == "from_raw"
                {
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

                self.checker
                    .error("unsupported call target", expr.span);
                None
            }
            ExprKind::MethodCall {
                receiver,
                method,
                args,
            } => {
                let (rval, rty) = self.lower_expr(receiver, None)?;
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
                let esize = elem_ty.byte_size(&self.checker.program.structs);
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
                let esize = elem_ty.byte_size(&self.checker.program.structs);
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
                    });
                }
                Some((base_addr, arr_ty))
            }
            ExprKind::If {
                cond,
                then_block,
                else_branch,
            } => {
                let (cval, cty) = self.lower_expr(cond, Some(&IrType::Bool))?;
                if cty != IrType::Bool {
                    self.checker.error("`if` condition must be `bool`", cond.span);
                }
                let then_bb = self.new_block();
                let else_bb = self.new_block();
                let join_bb = self.new_block();
                self.set_term(Terminator::Branch {
                    cond: cval,
                    then_bb,
                    else_bb,
                });

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;

    #[test]
    fn reject_ref_in_struct() {
        let src = "struct Bad { r: &u8 }";
        let file = Parser::parse_str(0, src).unwrap();
        let errs = Checker::check_and_lower(&file).unwrap_err();
        assert!(errs.iter().any(|d| d.message.contains("second-class")));
    }

    #[test]
    fn reject_mixed_width_arithmetic() {
        let src = "fn f(a: u8, b: u16) -> u16 { a + b }";
        let file = Parser::parse_str(0, src).unwrap();
        let errs = Checker::check_and_lower(&file).unwrap_err();
        assert!(errs.iter().any(|d| d.message.contains("strict type mismatch")));
    }

    #[test]
    fn reject_mutable_aliasing_allow_disjoint_fields() {
        let src = r#"
            struct Pair { a: u8, b: u8 }
            fn swap(x: &mut u8, y: &mut u8) { let t: u8 = *x; *x = *y; *y = t; }
            fn ok() {
                let mut p: Pair = Pair { a: 1, b: 2 };
                swap(&mut p.a, &mut p.b);
            }
        "#;
        let file = Parser::parse_str(0, src).unwrap();
        Checker::check_and_lower(&file).expect("disjoint field borrows should pass");

        let bad_src = r#"
            fn swap(x: &mut u8, y: &mut u8) {}
            fn bad() {
                let mut a: u8 = 1;
                swap(&mut a, &mut a);
            }
        "#;
        let bad_file = Parser::parse_str(0, bad_src).unwrap();
        let errs = Checker::check_and_lower(&bad_file).unwrap_err();
        assert!(errs.iter().any(|d| d.message.contains("aliasing violation")));
    }
}
