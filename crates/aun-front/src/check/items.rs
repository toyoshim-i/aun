use super::*;

impl Checker {
    pub(super) fn resolve_addr_space(&self, name: &str) -> Option<IrAddrSpace> {
        self.program
            .target
            .as_ref()
            .and_then(|t| {
                t.spaces
                    .iter()
                    .find(|s| s.name == name)
                    .map(|s| s.kind)
                    .or_else(|| {
                        t.bank_groups
                            .iter()
                            .any(|bg| bg.banks.iter().any(|b| b.name == name))
                            .then_some(IrAddrSpace::Flat)
                    })
            })
            .or_else(|| IrAddrSpace::parse_name(name))
    }

    pub(super) fn resolve_type(&mut self, ty: &TypeExpr, span: Span, allow_ref: bool) -> IrType {
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
                IrType::Slice {
                    mutable: *mutable,
                    elem: Box::new(inner_ty),
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

    pub(super) fn collect_items(&mut self, file: &File) {
        // Pass 0: collect target hardware definition if present
        for item in &file.items {
            if let Item::Include { span, .. } = item {
                self.error("`include!` requires compiling from a file path", *span);
            }
            if let Item::Target(t) = item {
                let mut spaces = Vec::with_capacity(t.spaces.len());
                for (n, explicit_kind, s, e, sp) in &t.spaces {
                    let kind = match explicit_kind {
                        Some(k) => match IrAddrSpace::parse_name(k) {
                            Some(parsed) => parsed,
                            None => {
                                self.error(format!("unknown target space kind `{k}`"), *sp);
                                IrAddrSpace::Flat
                            }
                        },
                        None => IrAddrSpace::parse_name(n).unwrap_or(IrAddrSpace::Flat),
                    };
                    spaces.push(SpaceDef {
                        name: n.clone(),
                        kind,
                        start: *s,
                        end: *e,
                    });
                }
                let (default_disable, default_enable) = match t.arch.as_str() {
                    "mcs51" | "ch559" | "8051" => ("clr EA", "setb EA"),
                    _ => ("di", "ei"),
                };
                self.program.target = Some(TargetDef {
                    name: t.name.clone(),
                    arch: t.arch.clone(),
                    spaces,
                    stacks: t
                        .stacks
                        .iter()
                        .map(|(n, s, e, _)| SpaceDef {
                            name: n.clone(),
                            kind: IrAddrSpace::Flat,
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
                            disable_asm: ic
                                .disable_asm
                                .clone()
                                .unwrap_or_else(|| default_disable.to_string()),
                            enable_asm: ic
                                .enable_asm
                                .clone()
                                .unwrap_or_else(|| default_enable.to_string()),
                            sources: ic
                                .sources
                                .iter()
                                .map(|(n, mode, v, r, _)| IntSourceDef {
                                    name: n.clone(),
                                    mode: mode.clone(),
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
                    let is_smc = m.attrs.iter().any(|a| a.name == "smc");
                    if is_smc
                        && self
                            .program
                            .target
                            .as_ref()
                            .is_some_and(|t| t.arch != "z80")
                    {
                        self.error(
                            "`#[smc]` self-modifying dispatch is only supported on `z80` targets",
                            m.span,
                        );
                    }
                    self.program.machines.push(MachineDef {
                        id,
                        name: m.name.clone(),
                        strict_transitions: m.attrs.iter().any(|a| a.name == "strict_transitions"),
                        smc: is_smc,
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

        // Pass 3a: resolve all const types once and evaluate scalar consts to a fixed point
        // so forward and cross-module scalar const references resolve before aggregate consts
        // and static initializers.
        let mut const_tys: std::collections::VecDeque<IrType> = std::collections::VecDeque::new();
        let mut pending_scalar_consts = Vec::new();
        for item in &file.items {
            if let Item::Const(c) = item {
                let ty = self.resolve_type(&c.ty, c.span, false);
                if !matches!(ty, IrType::Array { .. } | IrType::Struct(_)) {
                    pending_scalar_consts.push((c, ty.clone()));
                }
                const_tys.push_back(ty);
            }
        }
        loop {
            let mut next_pending = Vec::new();
            let mut progressed = false;
            for (c, ty) in pending_scalar_consts {
                let diag_before = self.diagnostics.len();
                if let Some(v) = self.eval_const_expr(&c.value, &ty) {
                    self.consts.insert(c.name.clone(), (ty, v));
                    progressed = true;
                } else if self.diagnostics.len() == diag_before {
                    next_pending.push((c, ty));
                }
            }
            pending_scalar_consts = next_pending;
            if !progressed {
                break;
            }
        }
        for (c, _) in pending_scalar_consts {
            self.error("const initializer must be a compile-time constant", c.span);
        }

        // Pass 3b: aggregate consts, statics, extern "asm", and function signatures
        for item in &file.items {
            match item {
                Item::Const(c) => {
                    let ty = const_tys.pop_front().unwrap();
                    if matches!(ty, IrType::Array { .. } | IrType::Struct(_)) {
                        if let Some(init) = self.eval_const_bytes(&c.value, &ty) {
                            let id = self.program.globals.len() as u32;
                            self.global_ids.insert(c.name.clone(), (id, ty.clone(), false));
                            self.program.globals.push(IrGlobal {
                                name: c.name.clone(),
                                ty,
                                mutable: false,
                                in_rom: true,
                                volatile: false,
                                fixed_addr: None,
                                is_extern: false,
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
                    if let Some(ref sname) = st.space
                        && self.program.target.is_some()
                        && self.resolve_addr_space(sname).is_none()
                    {
                        self.error(format!("unknown target address space `{sname}`"), st.span);
                    }
                    let in_rom = match st.space.as_deref().and_then(|s| self.resolve_addr_space(s)) {
                        Some(IrAddrSpace::Code) => true,
                        Some(
                            IrAddrSpace::Data
                            | IrAddrSpace::IData
                            | IrAddrSpace::XData
                            | IrAddrSpace::Sfr
                            | IrAddrSpace::Bit,
                        ) => false,
                        _ => !st.is_mut && st.fixed_addr.is_none(),
                    };
                    let id = self.program.globals.len() as u32;
                    self.global_ids
                        .insert(st.name.clone(), (id, ty.clone(), st.is_mut));
                    self.program.globals.push(IrGlobal {
                        name: st.name.clone(),
                        ty,
                        mutable: st.is_mut,
                        in_rom,
                        volatile: st.is_volatile,
                        fixed_addr: st.fixed_addr,
                        is_extern: false,
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
                    for es in &block.statics {
                        let ty = self.resolve_type(&es.ty, es.span, false);
                        let in_rom = es.in_rom
                            || es
                                .space
                                .as_deref()
                                .and_then(|s| self.resolve_addr_space(s))
                                == Some(IrAddrSpace::Code);
                        let id = self.program.globals.len() as u32;
                        self.global_ids
                            .insert(es.name.clone(), (id, ty.clone(), es.is_mut));
                        self.program.globals.push(IrGlobal {
                            name: es.name.clone(),
                            ty,
                            mutable: es.is_mut,
                            in_rom,
                            volatile: es.is_volatile,
                            fixed_addr: None,
                            is_extern: true,
                            align: 1,
                            soa: false,
                            space: es.space.clone(),
                            init: Vec::new(),
                        });
                    }
                    let block_no_cb = block.attrs.iter().any(|a| a.name == "no_callback");
                    for ef in &block.fns {
                        let no_callback = block_no_cb
                            || ef.attrs.iter().any(|a| a.name == "no_callback");
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
                            fixed_addr: ef.fixed_addr,
                            no_callback,
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
        let parse_prio = |args: &[String]| -> Option<u8> {
            args.iter().find_map(|arg| {
                arg.strip_prefix("prio=")
                    .or_else(|| {
                        arg.strip_prefix("prio(")
                            .and_then(|s| s.strip_suffix(')'))
                    })
                    .and_then(|s| s.parse::<u8>().ok())
            })
        };
        let interrupt_attr = f.attrs.iter().find(|a| a.name == "interrupt");
        let interrupt_source = interrupt_attr.and_then(|a| a.args.first().cloned());
        if let Some(attr) = interrupt_attr
            && let Some(prio) = parse_prio(&attr.args)
        {
            self.program.isr_priorities.insert(id, prio);
        }
        if let Some(isr_attr) = f.attrs.iter().find(|a| a.name == "isr_entry") {
            let prio = parse_prio(&isr_attr.args).unwrap_or(0);
            self.program.isr_entries.insert(id, prio);
        }
        let isr_using = interrupt_attr.and_then(|a| {
            a.args.iter().skip(1).find_map(|arg| {
                arg.strip_prefix("using(")
                    .and_then(|s| s.strip_suffix(')'))
                    .or_else(|| arg.strip_prefix("using="))
                    .and_then(|s| s.parse::<u8>().ok())
            })
        });
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
            isr_using,
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
                    ExprKind::StrLit(s)
                        if **elem == IrType::Int(IntType::U8) && s.len() == *len as usize =>
                    {
                        Some(s.as_bytes().to_vec())
                    }
                    ExprKind::ByteStrLit(b)
                        if **elem == IrType::Int(IntType::U8) && b.len() == *len as usize =>
                    {
                        Some(b.clone())
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

    pub(super) fn parse_fixed_scaled(
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

    pub(super) fn lower_functions(&mut self, file: &File) {
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

    pub(super) fn intern_rom_bytes(&mut self, bytes: &[u8]) -> u32 {
        if let Some(&id) = self.str_globals.get(bytes) {
            return id;
        }
        let id = self.program.globals.len() as u32;
        let len = bytes.len() as u16;
        self.str_globals.insert(bytes.to_vec(), id);
        self.program.globals.push(IrGlobal {
            name: format!("__aun_str_{id}"),
            ty: IrType::Array {
                elem: Box::new(IrType::Int(IntType::U8)),
                len,
            },
            mutable: false,
            in_rom: true,
            volatile: false,
            fixed_addr: None,
            is_extern: false,
            align: 1,
            soa: false,
            space: None,
            init: bytes.to_vec(),
        });
        id
    }
}
