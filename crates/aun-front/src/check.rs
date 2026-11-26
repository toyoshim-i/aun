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
    machine_ids: HashMap<String, u32>,
    enums: HashMap<String, (IntType, HashMap<String, i64>)>,
    consts: HashMap<String, (IrType, i64)>,
    global_ids: HashMap<String, (u32, IrType, bool)>,
    fn_sigs: HashMap<String, FnSig>,
    current_self_machine: Option<u32>,
}

impl Checker {
    pub fn check_and_lower(file: &File) -> Result<IrProgram, Vec<Diagnostic>> {
        let mut checker = Self {
            program: IrProgram::default(),
            diagnostics: Vec::new(),
            struct_ids: HashMap::new(),
            machine_ids: HashMap::new(),
            enums: HashMap::new(),
            consts: HashMap::new(),
            global_ids: HashMap::new(),
            fn_sigs: HashMap::new(),
            current_self_machine: None,
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
                "Self" => {
                    if let Some(mid) = self.current_self_machine {
                        IrType::Machine(mid)
                    } else {
                        self.error("`Self` is only valid inside a `machine` definition", span);
                        IrType::Unit
                    }
                }
                other => {
                    if let Some(&id) = self.struct_ids.get(other) {
                        IrType::Struct(id)
                    } else if let Some(&id) = self.machine_ids.get(other) {
                        IrType::Machine(id)
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
            TypeExpr::Pool { elem, capacity } => {
                if *capacity == 0 || *capacity > 255 {
                    self.error(
                        format!("`Pool<T, {capacity}>` capacity must be in 1..=255 for 1-byte Handle<T>"),
                        span,
                    );
                }
                let elem_ty = self.resolve_type(elem, span, false);
                if let IrType::Machine(mid) = elem_ty
                    && self.program.machines[mid as usize].smc
                {
                    self.error(
                        format!(
                            "`#[smc]` machine `{}` cannot be stored in `Pool` (SMC requires a singleton instance)",
                            self.program.machines[mid as usize].name
                        ),
                        span,
                    );
                }
                IrType::Pool {
                    elem: Box::new(elem_ty),
                    capacity: *capacity,
                }
            }
            TypeExpr::Handle(elem) => {
                let elem_ty = self.resolve_type(elem, span, false);
                IrType::Handle(Box::new(elem_ty))
            }
        }
    }

    fn collect_items(&mut self, file: &File) {
        // Pass 0: collect target hardware definition if present
        for item in &file.items {
            if let Item::Target(t) = item {
                self.program.target = Some(TargetDef {
                    name: t.name.clone(),
                    arch: t.arch.clone(),
                    spaces: t
                        .spaces
                        .iter()
                        .map(|(n, s, e, _)| SpaceDef {
                            name: n.clone(),
                            start: *s,
                            end: *e,
                        })
                        .collect(),
                    stacks: t
                        .stacks
                        .iter()
                        .map(|(n, s, e, _)| SpaceDef {
                            name: n.clone(),
                            start: *s,
                            end: *e,
                        })
                        .collect(),
                    bank_groups: t
                        .bank_groups
                        .iter()
                        .map(|bg| BankGroupDef {
                            name: bg.name.clone(),
                            start: bg.start,
                            end: bg.end,
                            banks: bg
                                .banks
                                .iter()
                                .map(|(n, s, e, di, _)| BankDef {
                                    name: n.clone(),
                                    start: *s,
                                    end: *e,
                                    requires_di: *di,
                                })
                                .collect(),
                            default_banks: bg.default_banks.clone(),
                            switches: bg
                                .switches
                                .iter()
                                .map(|(combo, ops)| {
                                    (
                                        combo.clone(),
                                        ops.iter()
                                            .map(|(p, v)| PortWriteOp {
                                                port: *p,
                                                val_expr: v.clone(),
                                            })
                                            .collect(),
                                    )
                                })
                                .collect(),
                        })
                        .collect(),
                    interrupt_controllers: t
                        .interrupt_controllers
                        .iter()
                        .map(|ic| IntCtrlDef {
                            name: ic.name.clone(),
                            disable_asm: ic.disable_asm.clone(),
                            enable_asm: ic.enable_asm.clone(),
                            sources: ic
                                .sources
                                .iter()
                                .map(|(n, v, r, _)| IntSourceDef {
                                    name: n.clone(),
                                    vector_addr: *v,
                                    rearm: r
                                        .iter()
                                        .map(|(p, e)| PortWriteOp {
                                            port: *p,
                                            val_expr: e.clone(),
                                        })
                                        .collect(),
                                })
                                .collect(),
                        })
                        .collect(),
                    overlay_slots: t
                        .overlay_slots
                        .iter()
                        .map(|os| OverlaySlotDef {
                            name: os.name.clone(),
                            start: os.start,
                            end: os.end,
                            load_fn: os.load_fn.clone(),
                        })
                        .collect(),
                });
            }
        }

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
                Item::Machine(m) => {
                    let id = self.program.machines.len() as u32;
                    self.machine_ids.insert(m.name.clone(), id);
                    self.program.machines.push(MachineDef {
                        id,
                        name: m.name.clone(),
                        strict_transitions: m.attrs.iter().any(|a| a.name == "strict_transitions"),
                        smc: m.attrs.iter().any(|a| a.name == "smc"),
                        shared_fields: Vec::new(),
                        states: Vec::new(),
                        initial_state: 0,
                        transitions: Vec::new(),
                        tag_offset: 0,
                        payload_offset: 2,
                        total_size: 2,
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

        // Pass 2: resolve struct and machine field layouts (forbid references in fields!)
        for item in &file.items {
            match item {
                Item::Struct(s) => {
                    let id = self.struct_ids[&s.name];
                    let mut offset = 0u16;
                    let mut fields = Vec::new();
                    for (fname, fty, fspan) in &s.fields {
                        let ir_ty = self.resolve_type(fty, *fspan, false);
                        let fsize = self.program.type_size(&ir_ty);
                        fields.push((fname.clone(), ir_ty, offset));
                        offset += fsize;
                    }
                    let def = &mut self.program.structs[id as usize];
                    def.fields = fields;
                    def.size = offset;
                }
                Item::Machine(m) => {
                    let id = self.machine_ids[&m.name];
                    let mut offset = 0u16;
                    let mut shared_fields = Vec::new();
                    for (fname, fty, fspan) in &m.shared_fields {
                        let ir_ty = self.resolve_type(fty, *fspan, false);
                        let fsize = self.program.type_size(&ir_ty);
                        shared_fields.push((fname.clone(), ir_ty, offset));
                        offset += fsize;
                    }
                    let tag_offset = offset;
                    let payload_offset = tag_offset + 2; // 2-byte tag/dispatch pointer

                    let mut states = Vec::new();
                    let mut state_name_to_id = HashMap::new();
                    let mut max_state_size = 0u16;
                    for (idx, st) in m.states.iter().enumerate() {
                        let sid = idx as u8;
                        state_name_to_id.insert(st.name.clone(), sid);
                        let mut soff = 0u16;
                        let mut sfields = Vec::new();
                        for (fname, fty, fspan) in &st.fields {
                            let ir_ty = self.resolve_type(fty, *fspan, false);
                            let fsize = self.program.type_size(&ir_ty);
                            sfields.push((fname.clone(), ir_ty, soff));
                            soff += fsize;
                        }
                        max_state_size = max_state_size.max(soff);
                        let overlay_slot = st
                            .attrs
                            .iter()
                            .find(|a| a.name == "overlay")
                            .and_then(|a| a.args.first().cloned());
                        if let Some(ref slot_name) = overlay_slot
                            && let Some(target) = &self.program.target
                            && !target.overlay_slots.iter().any(|s| &s.name == slot_name)
                        {
                            self.error(
                                format!(
                                    "unknown overlay slot `{slot_name}` in `#[overlay({slot_name})]`"
                                ),
                                st.span,
                            );
                        }
                        states.push(StateDef {
                            id: sid,
                            name: st.name.clone(),
                            fields: sfields,
                            size: soff,
                            on_enter: None,
                            on_exit: None,
                            methods: HashMap::new(),
                            overlay_slot,
                        });
                    }

                    let initial_state = if let Some(init_name) = &m.initial_state {
                        if let Some(&sid) = state_name_to_id.get(init_name) {
                            sid
                        } else {
                            self.error(
                                format!("unknown initial state `{init_name}` in machine `{}`", m.name),
                                m.span,
                            );
                            0
                        }
                    } else {
                        0
                    };

                    let mut transitions = vec![0u32; states.len()];
                    for (from, targets, tspan) in &m.transitions {
                        let Some(&fid) = state_name_to_id.get(from) else {
                            self.error(
                                format!("unknown source state `{from}` in `transitions`"),
                                *tspan,
                            );
                            continue;
                        };
                        for to in targets {
                            if let Some(&tid) = state_name_to_id.get(to) {
                                transitions[fid as usize] |= 1u32 << tid;
                            } else {
                                self.error(
                                    format!("unknown target state `{to}` in `transitions`"),
                                    *tspan,
                                );
                            }
                        }
                    }

                    let def = &mut self.program.machines[id as usize];
                    def.shared_fields = shared_fields;
                    def.states = states;
                    def.initial_state = initial_state;
                    def.transitions = transitions;
                    def.tag_offset = tag_offset;
                    def.payload_offset = payload_offset;
                    def.total_size = payload_offset + max_state_size;
                }
                _ => {}
            }
        }

        // Pass 3: consts, statics, extern "asm", and function signatures
        for item in &file.items {
            match item {
                Item::Const(c) => {
                    let ty = self.resolve_type(&c.ty, c.span, false);
                    if matches!(ty, IrType::Array { .. } | IrType::Struct(_)) {
                        if let Some(init) = self.eval_const_bytes(&c.value, &ty) {
                            let id = self.program.globals.len() as u32;
                            self.global_ids.insert(c.name.clone(), (id, ty.clone(), false));
                            self.program.globals.push(IrGlobal {
                                name: c.name.clone(),
                                ty,
                                mutable: false,
                                in_rom: true,
                                align: 1,
                                soa: false,
                                space: None,
                                init,
                            });
                        } else {
                            self.error(
                                "const aggregate initializer must be a compile-time constant",
                                c.span,
                            );
                        }
                    } else if let Some(v) = self.eval_const_expr(&c.value, &ty) {
                        self.consts.insert(c.name.clone(), (ty, v));
                    } else {
                        self.error("const initializer must be a compile-time constant", c.span);
                    }
                }
                Item::Static(st) => {
                    let ty = self.resolve_type(&st.ty, st.span, false);
                    let size = self.program.type_size(&ty) as usize;
                    let mut init = vec![0u8; size];
                    if let Some(init_expr) = &st.init
                        && let Some(bytes) = self.eval_const_bytes(init_expr, &ty)
                    {
                        for (dst, src) in init.iter_mut().zip(bytes.iter()) {
                            *dst = *src;
                        }
                    }
                    let mut align = 1u16;
                    let mut soa = false;
                    for attr in &st.attrs {
                        if attr.name == "layout" {
                            for arg in &attr.args {
                                if arg == "soa" {
                                    soa = true;
                                } else if let Some(n) = arg
                                    .strip_prefix("align(")
                                    .and_then(|s| s.strip_suffix(')'))
                                    .and_then(|s| s.parse::<u16>().ok())
                                {
                                    align = n;
                                }
                            }
                        } else if attr.name == "align"
                            && let Some(n) = attr.args.first().and_then(|s| s.parse::<u16>().ok())
                        {
                            align = n;
                        }
                    }
                    let id = self.program.globals.len() as u32;
                    self.global_ids
                        .insert(st.name.clone(), (id, ty.clone(), st.is_mut));
                    self.program.globals.push(IrGlobal {
                        name: st.name.clone(),
                        ty,
                        mutable: st.is_mut,
                        in_rom: false,
                        align,
                        soa,
                        space: st.space.clone(),
                        init,
                    });
                }
                Item::Machine(m) => {
                    let mid = self.machine_ids[&m.name];
                    self.current_self_machine = Some(mid);
                    // Register shared methods
                    for method in &m.methods {
                        let mangled = format!("{}::{}", m.name, method.name);
                        self.register_fn_decl(&mangled, method);
                    }
                    // Register per-state `on <State>` methods
                    for ob in &m.on_blocks {
                        let state_id_opt = self.program.machines[mid as usize]
                            .states
                            .iter()
                            .find(|s| s.name == ob.state_name)
                            .map(|s| s.id);
                        let Some(sid) = state_id_opt else {
                            self.error(
                                format!("unknown state `{}` in `on` block of `{}`", ob.state_name, m.name),
                                ob.span,
                            );
                            continue;
                        };
                        for method in &ob.methods {
                            let mangled = format!("{}::{}::{}", m.name, ob.state_name, method.name);
                            let fid = self.register_fn_decl(&mangled, method);
                            let overlay_slot = self.program.machines[mid as usize].states
                                [sid as usize]
                                .overlay_slot
                                .clone();
                            if let Some(slot) = overlay_slot {
                                self.program.functions[fid as usize].overlay_state =
                                    Some((slot, ob.state_name.clone()));
                            }
                            let st = &mut self.program.machines[mid as usize].states[sid as usize];
                            if method.name == "on_enter" {
                                st.on_enter = Some(fid);
                            } else if method.name == "on_exit" {
                                st.on_exit = Some(fid);
                            } else {
                                st.methods.insert(method.name.clone(), fid);
                            }
                        }
                    }
                    self.current_self_machine = None;
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
                    self.register_fn_decl(&f.name, f);
                }
                _ => {}
            }
        }
    }

    fn register_fn_decl(&mut self, sig_name: &str, f: &FnDecl) -> u32 {
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
            sig_name.to_string(),
            FnSig {
                id,
                is_extern: false,
                params: sig_params.clone(),
                ret_ty: ret_ty.clone(),
            },
        );

        let is_test = f.attrs.iter().any(|a| a.name == "test");
        let is_reentrant = f.attrs.iter().any(|a| a.name == "reentrant");
        let interrupt_source = f
            .attrs
            .iter()
            .find(|a| a.name == "interrupt")
            .and_then(|a| a.args.first().cloned());
        if let Some(ref src_name) = interrupt_source {
            if !sig_params.is_empty() || ret_ty != IrType::Unit {
                self.error(
                    format!("`#[interrupt({src_name})]` handler `{sig_name}` must have signature `fn()`"),
                    f.span,
                );
            }
            if let Some(target) = &self.program.target {
                let exists = target
                    .interrupt_controllers
                    .iter()
                    .any(|ic| ic.sources.iter().any(|s| &s.name == src_name));
                if !exists {
                    self.error(
                        format!("unknown interrupt source `{src_name}` in `#[interrupt]`"),
                        f.span,
                    );
                }
            }
        }
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
            name: sig_name.to_string(),
            is_pub: f.is_pub,
            param_count: locals.len(),
            locals,
            ret_ty,
            is_test,
            expected_panic,
            is_reentrant,
            interrupt_source,
            requires_banks: Vec::new(),
            requires_di: false,
            overlay_state: None,
            blocks: vec![BasicBlock::default()],
            next_val: 0,
        });
        id
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
            ExprKind::Ident(name) => self.consts.get(name).map(|(_, v)| *v),
            ExprKind::Unary {
                op: UnaryOp::Neg,
                expr: inner,
            } => self.eval_const_expr(inner, expected_ty).map(|v| -v),
            _ => None,
        }
    }

    fn eval_const_bytes(&mut self, expr: &Expr, expected_ty: &IrType) -> Option<Vec<u8>> {
        let total_size = self.program.type_size(expected_ty) as usize;
        match expected_ty {
            IrType::Int(_) | IrType::Fixed { .. } | IrType::Bool => {
                let val = self.eval_const_expr(expr, expected_ty)?;
                let mut out = vec![0u8; total_size];
                let le = val.to_le_bytes();
                for (d, s) in out.iter_mut().zip(le.iter()) {
                    *d = *s;
                }
                Some(out)
            }
            IrType::Array { elem, len } => {
                let elem_sz = self.program.type_size(elem) as usize;
                let mut out = vec![0u8; total_size];
                match &expr.kind {
                    ExprKind::ArrayLit(elems) => {
                        if elems.len() != *len as usize {
                            return None;
                        }
                        for (i, e) in elems.iter().enumerate() {
                            let eb = self.eval_const_bytes(e, elem)?;
                            let off = i * elem_sz;
                            out[off..off + elem_sz].copy_from_slice(&eb[..elem_sz]);
                        }
                        Some(out)
                    }
                    ExprKind::ArrayRepeat { val, len: rlen } => {
                        if *rlen != *len {
                            return None;
                        }
                        let eb = self.eval_const_bytes(val, elem)?;
                        for i in 0..(*len as usize) {
                            let off = i * elem_sz;
                            out[off..off + elem_sz].copy_from_slice(&eb[..elem_sz]);
                        }
                        Some(out)
                    }
                    _ => None,
                }
            }
            IrType::Struct(sid) => {
                let ExprKind::StructLit { fields, .. } = &expr.kind else {
                    return None;
                };
                let sdef = self.program.structs[*sid as usize].clone();
                let mut out = vec![0u8; total_size];
                for (fname, fexpr) in fields {
                    let fdef = sdef.fields.iter().find(|f| &f.0 == fname)?;
                    let fb = self.eval_const_bytes(fexpr, &fdef.1)?;
                    let off = fdef.2 as usize;
                    let fsz = self.program.type_size(&fdef.1) as usize;
                    out[off..off + fsz].copy_from_slice(&fb[..fsz]);
                }
                Some(out)
            }
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
        for item in &file.items {
            match item {
                Item::Fn(f) => {
                    if let Some(sig) = self.fn_sigs.get(&f.name).cloned() {
                        let mut lowerer = FnLowerer::new(self, sig.id as usize, None);
                        lowerer.lower_fn(f);
                    }
                }
                Item::Machine(m) => {
                    let mid = self.machine_ids[&m.name];
                    self.current_self_machine = Some(mid);
                    for method in &m.methods {
                        let mangled = format!("{}::{}", m.name, method.name);
                        if let Some(sig) = self.fn_sigs.get(&mangled).cloned() {
                            let mut lowerer = FnLowerer::new(
                                self,
                                sig.id as usize,
                                Some((mid, None, false)),
                            );
                            lowerer.lower_fn(method);
                        }
                    }
                    for ob in &m.on_blocks {
                        let state_id_opt = self.program.machines[mid as usize]
                            .states
                            .iter()
                            .find(|s| s.name == ob.state_name)
                            .map(|s| s.id);
                        let Some(sid) = state_id_opt else {
                            continue;
                        };
                        for method in &ob.methods {
                            let mangled = format!("{}::{}::{}", m.name, ob.state_name, method.name);
                            let is_hook = method.name == "on_enter" || method.name == "on_exit";
                            if let Some(sig) = self.fn_sigs.get(&mangled).cloned() {
                                let mut lowerer = FnLowerer::new(
                                    self,
                                    sig.id as usize,
                                    Some((mid, Some(sid), is_hook)),
                                );
                                lowerer.lower_fn(method);
                            }
                        }
                    }
                    self.current_self_machine = None;
                }
                _ => {}
            }
        }
    }
}

#[derive(Debug, Clone)]
struct BankScopeEntry {
    group: String,
    prev_banks: Vec<String>,
    restore_ei: bool,
    loop_depth: usize,
}

struct FnLowerer<'a> {
    checker: &'a mut Checker,
    fn_idx: usize,
    cur_bb: BlockId,
    scopes: Vec<HashMap<String, VarInfo>>,
    loop_stack: Vec<(BlockId, BlockId)>, // (continue_bb, break_bb)
    /// `(machine_id, narrowed_state_id, is_lifecycle_hook)` when inside a `machine` method.
    machine_ctx: Option<(u32, Option<u8>, bool)>,
    /// Current active banks per `bank_group` name.
    active_banks: HashMap<String, Vec<String>>,
    /// Active `with bank(...)` scopes for LIFO unwinding on early return/break/continue/goto.
    bank_scope_stack: Vec<BankScopeEntry>,
    /// Whether interrupts are statically known to be disabled (`di`).
    irq_disabled: bool,
}

impl<'a> FnLowerer<'a> {
    fn new(
        checker: &'a mut Checker,
        fn_idx: usize,
        machine_ctx: Option<(u32, Option<u8>, bool)>,
    ) -> Self {
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
        let mut active_banks = HashMap::new();
        if let Some(target) = &checker.program.target {
            for bg in &target.bank_groups {
                active_banks.insert(bg.name.clone(), bg.default_banks.clone());
            }
        }
        let irq_disabled = checker.program.functions[fn_idx].interrupt_source.is_some();
        Self {
            checker,
            fn_idx,
            cur_bb: 0,
            scopes: vec![root_scope],
            loop_stack: Vec::new(),
            machine_ctx,
            active_banks,
            bank_scope_stack: Vec::new(),
            irq_disabled,
        }
    }

    /// Emits `BankExit` in LIFO order for all active `with bank(...)` scopes with `loop_depth >= min_loop_depth`.
    fn unwind_bank_scopes(&mut self, min_loop_depth: usize) {
        let exits: Vec<BankScopeEntry> = self
            .bank_scope_stack
            .iter()
            .rev()
            .take_while(|e| e.loop_depth >= min_loop_depth)
            .cloned()
            .collect();
        for entry in exits {
            self.emit(IrInst::BankExit {
                group: entry.group,
                restore_banks: entry.prev_banks,
                restore_ei: entry.restore_ei,
            });
        }
    }

    fn check_bank_mapped(&mut self, bank_name: &str, context: &str, span: Span) {
        let Some(target) = self.checker.program.target.clone() else {
            return;
        };
        for bg in &target.bank_groups {
            if bg.banks.iter().any(|b| b.name == bank_name) {
                let mapped = self
                    .active_banks
                    .get(&bg.name)
                    .is_some_and(|list| list.iter().any(|b| b == bank_name));
                if !mapped {
                    let cur = self
                        .active_banks
                        .get(&bg.name)
                        .cloned()
                        .unwrap_or_default();
                    self.checker.error(
                        format!(
                            "bank conflict: {context} requires bank `{bank_name}`, but `{}` currently has `{}` mapped",
                            bg.name,
                            cur.join(" + ")
                        ),
                        span,
                    );
                }
                return;
            }
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
        let size = self.checker.program.type_size(ty);
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
                    if let Some(space) = self.checker.program.globals[gid as usize].space.clone() {
                        self.check_bank_mapped(&space, &format!("accessing `{name}`"), expr.span);
                    }
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
                            return Some((faddr, fty, is_mut));
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
                            return Some((faddr, fty, is_mut));
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
                    if let Some(space) = self.checker.program.globals[gid as usize].space.clone() {
                        self.check_bank_mapped(&space, &format!("reading `{name}`"), expr.span);
                    }
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
                    if name == "mem_read_u8" {
                        if args.len() != 1 {
                            self.checker
                                .error("`mem_read_u8` expects 1 argument (addr: u16)", expr.span);
                            return None;
                        }
                        let (a, _) = self.lower_expr(&args[0], Some(&IrType::Int(IntType::U16)))?;
                        let dst = self.fresh_val();
                        self.emit(IrInst::LoadPtr {
                            dst,
                            addr: a,
                            ty: IrType::Int(IntType::U8),
                        });
                        return Some((dst, IrType::Int(IntType::U8)));
                    }
                    if name == "mem_read_u16" {
                        if args.len() != 1 {
                            self.checker
                                .error("`mem_read_u16` expects 1 argument (addr: u16)", expr.span);
                            return None;
                        }
                        let (a, _) = self.lower_expr(&args[0], Some(&IrType::Int(IntType::U16)))?;
                        let dst = self.fresh_val();
                        self.emit(IrInst::LoadPtr {
                            dst,
                            addr: a,
                            ty: IrType::Int(IntType::U16),
                        });
                        return Some((dst, IrType::Int(IntType::U16)));
                    }
                    if name == "addr_of" {
                        if args.len() != 1 {
                            self.checker
                                .error("`addr_of` expects 1 lvalue argument", expr.span);
                            return None;
                        }
                        let (addr, _, _) = self.lower_lvalue(&args[0])?;
                        return Some((addr, IrType::Int(IntType::U16)));
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
                        let compat = &aty == expected_ty
                            || matches!(
                                (&aty, expected_ty),
                                (
                                    IrType::Ref {
                                        mutable: true,
                                        inner: a_in,
                                    },
                                    IrType::Ref {
                                        mutable: false,
                                        inner: e_in,
                                    }
                                ) if a_in == e_in
                            );
                        if !compat {
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
                if let Some((mut raddr, mut rty, mut rmut)) = self.lower_lvalue_if_place(receiver) {
                    if let IrType::Ref { mutable, inner } = rty {
                        let loaded = self.fresh_val();
                        self.emit(IrInst::LoadPtr {
                            dst: loaded,
                            addr: raddr,
                            ty: IrType::Ref {
                                mutable,
                                inner: inner.clone(),
                            },
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

    fn lower_lvalue_if_place(&mut self, expr: &Expr) -> Option<(ValueId, IrType, bool)> {
        if matches!(
            expr.kind,
            ExprKind::Ident(_) | ExprKind::Field { .. } | ExprKind::Index { .. } | ExprKind::Deref(_)
        ) {
            self.lower_lvalue(expr)
        } else {
            None
        }
    }

    fn lower_state_payload(
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

    #[test]
    fn reject_forbidden_static_transition_and_strict_dynamic_goto() {
        let bad_static = r#"
            machine Ball {
                state Inactive;
                state InShooter { power: u8 };
                state MagnetHeld;
                transitions {
                    Inactive -> InShooter;
                }
                on InShooter {
                    fn step(&mut self) {
                        goto MagnetHeld;
                    }
                }
            }
        "#;
        let file = Parser::parse_str(0, bad_static).unwrap();
        let errs = Checker::check_and_lower(&file).unwrap_err();
        assert!(errs
            .iter()
            .any(|d| d.message.contains("forbidden static state transition")));

        let bad_strict = r#"
            #[strict_transitions]
            machine Door {
                state Closed;
                state Open;
                transitions { Closed -> Open; }
            }
            fn f() {
                let mut d: Door = Door::new();
                d.goto(Open);
            }
        "#;
        let file2 = Parser::parse_str(0, bad_strict).unwrap();
        let errs2 = Checker::check_and_lower(&file2).unwrap_err();
        assert!(errs2
            .iter()
            .any(|d| d.message.contains("strict_transitions")));
    }

    #[test]
    fn static_bank_conflict_and_di_checking() {
        let base_target = r#"
            target pc88sr : z80 {
                space MainRam @ 0x0100..0xBFFF;
                stack MainStack @ 0xBE00..0xBFFF;
                bank_group HighWindow @ 0xC000..0xFFFF {
                    bank HighMainRam @ 0xC000..0xEFFF;
                    bank VramAlu @ 0xC000..0xFFFF requires di;
                    default HighMainRam;
                    switch(to) {
                        HighMainRam => { io_out(0x35, 0x00); }
                        VramAlu     => { io_out(0x35, 0x80); }
                    }
                }
            }
            static mut SCORE: u8 in HighMainRam;
            extern "asm" {
                fn gfx_fill(c: u8 in A) requires bank(VramAlu), di clobbers(AF);
            }
        "#;

        // OK: calling gfx_fill inside `with bank(VramAlu)`
        let ok_src = format!(
            "{base_target}\nfn draw() {{ let s: u8 = SCORE; with bank(VramAlu) {{ gfx_fill(s); }} }}"
        );
        let ok_file = Parser::parse_str(0, &ok_src).unwrap();
        Checker::check_and_lower(&ok_file).expect("valid bank usage should pass");

        // Error: accessing `SCORE in HighMainRam` while `VramAlu` is mapped!
        let bad_conflict = format!(
            "{base_target}\nfn bad() {{ with bank(VramAlu) {{ gfx_fill(SCORE); }} }}"
        );
        let bad_file = Parser::parse_str(0, &bad_conflict).unwrap();
        let errs = Checker::check_and_lower(&bad_file).unwrap_err();
        assert!(errs.iter().any(|d| d.message.contains("bank conflict")));

        // Error: calling `gfx_fill` without `with bank(VramAlu)`
        let bad_precond = format!("{base_target}\nfn bad2() {{ gfx_fill(1u8); }}");
        let bad_file2 = Parser::parse_str(0, &bad_precond).unwrap();
        let errs2 = Checker::check_and_lower(&bad_file2).unwrap_err();
        assert!(errs2.iter().any(|d| d.message.contains("bank conflict")));
        assert!(errs2.iter().any(|d| d.message.contains("requires `di`")));
    }
}
