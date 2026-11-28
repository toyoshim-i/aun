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
            TypeExpr::Queue { elem, capacity } => {
                if !capacity.is_power_of_two() || !(2..=128).contains(capacity) {
                    self.error(
                        format!("`Queue<T, {capacity}>` capacity must be a power of two in 2..=128"),
                        span,
                    );
                }
                let elem_ty = self.resolve_type(elem, span, false);
                if let IrType::Machine(mid) = elem_ty
                    && self.program.machines[mid as usize].smc
                {
                    self.error(
                        format!(
                            "`#[smc]` machine `{}` cannot be stored in `Queue` (SMC requires a singleton instance)",
                            self.program.machines[mid as usize].name
                        ),
                        span,
                    );
                }
                IrType::Queue {
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
                            space: os.space.clone(),
                            load_fn: os.load_fn.clone(),
                        })
                        .collect(),
                    on_swap_fn: t.on_swap_fn.clone(),
                });
                if let Some(ref td) = self.program.target {
                    let u8t = IrType::Int(IntType::U8);
                    for (idx, (sp_name, _, _, _)) in td.all_space_names().into_iter().enumerate() {
                        self.consts.insert(format!("SPACE_{sp_name}"), (u8t.clone(), idx as i64));
                    }
                }
            }
        }

        // Pre-register built-in `struct LoadOp { blob_id: u8, space: u8, dst: u16, size: u16 }`
        // unless user code explicitly defines `struct LoadOp`.
        if !file
            .items
            .iter()
            .any(|it| matches!(it, Item::Struct(s) if s.name == "LoadOp"))
        {
            let u8t = IrType::Int(IntType::U8);
            let u16t = IrType::Int(IntType::U16);
            let id = self.program.structs.len() as u32;
            self.struct_ids.insert("LoadOp".into(), id);
            self.program.structs.push(StructDef {
                name: "LoadOp".into(),
                fields: vec![
                    ("blob_id".into(), u8t.clone(), 0),
                    ("space".into(), u8t, 1),
                    ("dst".into(), u16t.clone(), 2),
                    ("size".into(), u16t, 4),
                ],
                size: 6,
            });
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

                    let mut states: Vec<StateDef> = Vec::new();
                    let mut state_name_to_id: HashMap<String, u8> = HashMap::new();
                    let mut init_child_names: Vec<Option<String>> = Vec::new();
                    let mut pending_statics: Vec<(u8, StaticDecl)> = Vec::new();
                    let mut state_limit_reported = false;

                    #[allow(clippy::too_many_arguments)]
                    fn flatten_state(
                        checker: &mut Checker,
                        st: &StateDecl,
                        parent_id: Option<u8>,
                        prefix: &str,
                        base_off: u16,
                        anc_slot: Option<&str>,
                        states: &mut Vec<StateDef>,
                        state_name_to_id: &mut HashMap<String, u8>,
                        init_child_names: &mut Vec<Option<String>>,
                        pending_statics: &mut Vec<(u8, StaticDecl)>,
                        state_limit_reported: &mut bool,
                    ) -> u16 {
                        if states.len() >= 32 {
                            if !*state_limit_reported {
                                *state_limit_reported = true;
                                checker.error(
                                    "state machine exceeds 32-state limit for 32-bit transition bitmasks",
                                    st.span,
                                );
                            }
                            return 0;
                        }
                        let sid = states.len() as u8;
                        let qname = if prefix.is_empty() {
                            st.name.clone()
                        } else {
                            format!("{prefix}::{}", st.name)
                        };
                        state_name_to_id.insert(qname.clone(), sid);

                        let mut soff = base_off;
                        let mut sfields = Vec::new();
                        for (fname, fty, fspan) in &st.fields {
                            let ir_ty = checker.resolve_type(fty, *fspan, false);
                            let fsize = checker.program.type_size(&ir_ty);
                            sfields.push((fname.clone(), ir_ty, soff));
                            soff += fsize;
                        }
                        let own_size = soff - base_off;

                        let overlay_slot = st
                            .attrs
                            .iter()
                            .find(|a| a.name == "overlay")
                            .and_then(|a| a.args.first().cloned());
                        if let Some(ref slot_name) = overlay_slot {
                            if anc_slot == Some(slot_name.as_str()) {
                                checker.error(
                                    format!(
                                        "nested state `{qname}` cannot share overlay slot `{slot_name}` with an ancestor"
                                    ),
                                    st.span,
                                );
                            }
                            if let Some(target) = &checker.program.target
                                && !target.overlay_slots.iter().any(|s| &s.name == slot_name)
                            {
                                checker.error(
                                    format!(
                                        "unknown overlay slot `{slot_name}` in `#[overlay({slot_name})]`"
                                    ),
                                    st.span,
                                );
                            }
                        }

                        let swap_attr = st.attrs.iter().find(|a| a.name == "swap_boundary");
                        let swap_boundary = swap_attr.is_some();
                        let code_space = swap_attr.and_then(|a| a.args.first().cloned());
                        if let Some(ref sp_name) = code_space
                            && checker.program.target.is_some()
                            && checker.resolve_addr_space(sp_name).is_none()
                        {
                            checker.error(
                                format!(
                                    "unknown target space/bank `{sp_name}` in `#[swap_boundary({sp_name})]`"
                                ),
                                st.span,
                            );
                        }

                        let is_leaf = st.children.is_empty();
                        states.push(StateDef {
                            id: sid,
                            name: qname.clone(),
                            parent: parent_id,
                            is_leaf,
                            initial_child: None,
                            leaf_range: (sid, sid),
                            fields: sfields,
                            size: own_size,
                            on_enter: None,
                            on_exit: None,
                            methods: HashMap::new(),
                            overlay_slot: overlay_slot.clone(),
                            swap_boundary,
                            code_space,
                        });
                        init_child_names.push(st.initial_child.clone());
                        for sd in &st.statics {
                            pending_statics.push((sid, sd.clone()));
                        }

                        let next_anc_slot = overlay_slot.as_deref().or(anc_slot);
                        let mut max_child_size = 0u16;
                        for child in &st.children {
                            let csz = flatten_state(
                                checker,
                                child,
                                Some(sid),
                                &qname,
                                soff,
                                next_anc_slot,
                                states,
                                state_name_to_id,
                                init_child_names,
                                pending_statics,
                                state_limit_reported,
                            );
                            max_child_size = max_child_size.max(csz);
                        }
                        let total_sub = own_size + max_child_size;
                        states[sid as usize].size = total_sub;
                        total_sub
                    }

                    let mut max_state_size = 0u16;
                    for st in &m.states {
                        let sub_sz = flatten_state(
                            self,
                            st,
                            None,
                            "",
                            0,
                            None,
                            &mut states,
                            &mut state_name_to_id,
                            &mut init_child_names,
                            &mut pending_statics,
                            &mut state_limit_reported,
                        );
                        max_state_size = max_state_size.max(sub_sz);
                    }

                    // Resolve `initial_child` for composite states (default to first direct child)
                    for sid in 0..states.len() {
                        if states[sid].is_leaf {
                            continue;
                        }
                        let parent_qname = states[sid].name.clone();
                        let resolved = if let Some(ref ic) = init_child_names[sid] {
                            let full = format!("{parent_qname}::{ic}");
                            if let Some(&cid) =
                                state_name_to_id.get(&full).or_else(|| state_name_to_id.get(ic))
                            {
                                Some(cid)
                            } else {
                                self.error(
                                    format!(
                                        "unknown `initial {ic}` inside state `{parent_qname}`"
                                    ),
                                    m.span,
                                );
                                None
                            }
                        } else {
                            states
                                .iter()
                                .find(|c| c.parent == Some(sid as u8))
                                .map(|c| c.id)
                        };
                        states[sid].initial_child = resolved;
                    }

                    // Compute descendant `leaf_range` for each state (contiguous in DFS pre-order)
                    let parents: Vec<Option<u8>> = states.iter().map(|s| s.parent).collect();
                    let is_leafs: Vec<bool> = states.iter().map(|s| s.is_leaf).collect();
                    let is_anc = |anc: u8, mut desc: u8| -> bool {
                        for _ in 0..=parents.len() {
                            if desc == anc {
                                return true;
                            }
                            match parents[desc as usize] {
                                Some(p) if p != desc => desc = p,
                                _ => return false,
                            }
                        }
                        false
                    };
                    for sid in 0..states.len() {
                        let mut lo = u8::MAX;
                        let mut hi = 0u8;
                        for cid in 0..states.len() {
                            if is_leafs[cid] && is_anc(sid as u8, cid as u8) {
                                lo = lo.min(cid as u8);
                                hi = hi.max(cid as u8);
                            }
                        }
                        if lo <= hi {
                            states[sid].leaf_range = (lo, hi);
                        }
                    }

                    let def_tmp = MachineDef {
                        id,
                        name: m.name.clone(),
                        strict_transitions: false,
                        smc: false,
                        shared_fields: Vec::new(),
                        states,
                        initial_state: 0,
                        transitions: Vec::new(),
                        tag_offset,
                        payload_offset,
                        total_size: 0,
                    };

                    let raw_initial = if let Some(init_name) = &m.initial_state {
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
                    let initial_state = def_tmp.resolve_to_leaf(raw_initial);

                    let mut transitions = vec![0u32; def_tmp.states.len()];
                    for (from, targets, tspan) in &m.transitions {
                        let Some(&fid) = state_name_to_id.get(from) else {
                            self.error(
                                format!("unknown source state `{from}` in `transitions`"),
                                *tspan,
                            );
                            continue;
                        };
                        let f_leaves: Vec<u8> = def_tmp
                            .states
                            .iter()
                            .filter(|s| s.is_leaf && def_tmp.is_ancestor(fid, s.id))
                            .map(|s| s.id)
                            .collect();
                        for to in targets {
                            if let Some(&tid) = state_name_to_id.get(to) {
                                let to_leaf = def_tmp.resolve_to_leaf(tid);
                                for fl in &f_leaves {
                                    transitions[*fl as usize] |= 1u32 << to_leaf;
                                }
                            } else {
                                self.error(
                                    format!("unknown target state `{to}` in `transitions`"),
                                    *tspan,
                                );
                            }
                        }
                    }

                    self.pending_state_statics.insert(id, pending_statics);

                    let def = &mut self.program.machines[id as usize];
                    def.shared_fields = shared_fields;
                    def.states = def_tmp.states;
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
                            let export_name = self.parse_export_name(&c.attrs, "const");
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
                                owner_state: None,
                                owned_by_irq: None,
                                export_name,
                                zero_on_enter: false,
                                export_asm_layout: None,
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
                    let mut space_attr: Option<String> = None;
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
                        } else if attr.name == "space"
                            && let Some(sp) = attr.args.first()
                        {
                            space_attr = Some(sp.clone());
                        }
                    }
                    let effective_space = st.space.clone().or(space_attr);
                    if let Some(ref sname) = effective_space
                        && self.program.target.is_some()
                        && self.resolve_addr_space(sname).is_none()
                    {
                        self.error(format!("unknown target address space `{sname}`"), st.span);
                    }
                    let in_rom = match effective_space
                        .as_deref()
                        .and_then(|s| self.resolve_addr_space(s))
                    {
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
                    let is_banked_space = effective_space.as_deref().is_some_and(|sname| {
                        if let Some(ref td) = self.program.target {
                            !td.spaces.first().is_some_and(|s| s.name == sname)
                        } else {
                            true
                        }
                    });
                    if st.is_mut && !in_rom && is_banked_space && init.iter().any(|&b| b != 0) {
                        let sname = effective_space.as_deref().unwrap_or("bank");
                        self.error(
                            format!(
                                "banked `static mut {}` in `{sname}` cannot have a non-zero initializer (use `= 0` or initialize at runtime)",
                                st.name
                            ),
                            st.span,
                        );
                    }
                    let owned_by_irq = self.parse_owned_by_irq(&st.attrs, st.is_mut, st.span);
                    let export_name = self.parse_export_name(&st.attrs, "static");
                    let _ = self.parse_zero_on_enter(&st.attrs, st.is_mut, false);
                    let export_asm_layout = self.parse_export_asm_layout(&st.attrs);
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
                        space: effective_space,
                        owner_state: None,
                        owned_by_irq,
                        export_name,
                        zero_on_enter: false,
                        export_asm_layout,
                        init,
                    });
                }
                Item::Machine(m) => {
                    let mid = self.machine_ids[&m.name];
                    self.current_self_machine = Some(mid);
                    // Register state-scoped `static mut` buffers declared inside `state S { ... }`
                    if let Some(st_statics) = self.pending_state_statics.remove(&mid) {
                        for (sid, st) in st_statics {
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
                            let mut space_attr: Option<String> = None;
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
                                    && let Some(n) =
                                        attr.args.first().and_then(|s| s.parse::<u16>().ok())
                                {
                                    align = n;
                                } else if attr.name == "space"
                                    && let Some(sp) = attr.args.first()
                                {
                                    space_attr = Some(sp.clone());
                                }
                            }
                            let effective_space = st.space.clone().or(space_attr);
                            if let Some(ref sname) = effective_space
                                && self.program.target.is_some()
                                && self.resolve_addr_space(sname).is_none()
                            {
                                self.error(
                                    format!("unknown target address space `{sname}`"),
                                    st.span,
                                );
                            }
                            if st.is_mut && init.iter().any(|&b| b != 0) {
                                self.error(
                                    format!(
                                        "state-scoped `static mut {}` cannot have a non-zero initializer (state buffers share overlaid RAM; initialize on state entry)",
                                        st.name
                                    ),
                                    st.span,
                                );
                            }
                            let owned_by_irq =
                                self.parse_owned_by_irq(&st.attrs, st.is_mut, st.span);
                            let export_name = self.parse_export_name(&st.attrs, "static");
                            let zero_on_enter =
                                self.parse_zero_on_enter(&st.attrs, st.is_mut, true);
                            let export_asm_layout = self.parse_export_asm_layout(&st.attrs);
                            let gid = self.program.globals.len() as u32;
                            let mangled = format!(
                                "{}__{}__{}",
                                self.program.machines[mid as usize].name,
                                self.program.machines[mid as usize].states[sid as usize]
                                    .name
                                    .replace("::", "_"),
                                st.name
                            );
                            self.state_globals
                                .insert((mid, sid, st.name.clone()), (gid, ty.clone(), st.is_mut));
                            self.program.globals.push(IrGlobal {
                                name: mangled,
                                ty,
                                mutable: st.is_mut,
                                in_rom: false,
                                volatile: st.is_volatile,
                                fixed_addr: st.fixed_addr,
                                is_extern: false,
                                align,
                                soa,
                                space: effective_space,
                                owner_state: Some((mid, sid)),
                                owned_by_irq,
                                export_name,
                                zero_on_enter,
                                export_asm_layout,
                                init,
                            });
                        }
                    }
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
                            self.program.functions[fid as usize].owner_state = Some((mid, sid));
                            let bnd_overlay = {
                                let mdef = &self.program.machines[mid as usize];
                                mdef.enclosing_boundary(sid).and_then(|bsid| {
                                    mdef.states[bsid as usize]
                                        .overlay_slot
                                        .clone()
                                        .map(|sl| (sl, mdef.states[bsid as usize].name.clone()))
                                })
                            };
                            if let Some(ovl) = bnd_overlay {
                                self.program.functions[fid as usize].overlay_state = Some(ovl);
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
                    // Inherit missing non-hook methods top-down from parent states into children
                    let num_states = self.program.machines[mid as usize].states.len();
                    for sid in 0..num_states {
                        if let Some(pid) = self.program.machines[mid as usize].states[sid].parent {
                            let p_methods =
                                self.program.machines[mid as usize].states[pid as usize]
                                    .methods
                                    .clone();
                            let c_methods =
                                &mut self.program.machines[mid as usize].states[sid].methods;
                            for (mname, fid) in p_methods {
                                c_methods.entry(mname).or_insert(fid);
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
                            owner_state: None,
                            owned_by_irq: None,
                            export_name: None,
                            zero_on_enter: false,
                            export_asm_layout: None,
                            init: Vec::new(),
                        });
                    }
                    let block_no_cb = block.attrs.iter().any(|a| a.name == "no_callback");
                    let block_owner_state = block
                        .attrs
                        .iter()
                        .find(|a| a.name == "overlay")
                        .and_then(|a| a.args.first())
                        .and_then(|arg| self.resolve_overlay_owner_state(arg, block.span));
                    for ef in &block.fns {
                        let no_callback = block_no_cb
                            || ef.attrs.iter().any(|a| a.name == "no_callback");
                        let ef_owner_state = ef
                            .attrs
                            .iter()
                            .find(|a| a.name == "overlay")
                            .and_then(|a| a.args.first())
                            .and_then(|arg| self.resolve_overlay_owner_state(arg, ef.span))
                            .or(block_owner_state);
                        let mut wcet_tstates: Option<u32> = None;
                        let mut max_iters: Option<u32> = None;
                        let mut loop_bounds: HashMap<String, u32> = HashMap::new();
                        for attr in &ef.attrs {
                            match attr.name.as_str() {
                                "wcet" => {
                                    if attr.args.len() != 1 {
                                        self.error(
                                            "`#[wcet(...)]` expects a single T-state limit such as `#[wcet(1200T)]`",
                                            attr.span,
                                        );
                                        continue;
                                    }
                                    let raw = attr.args[0].trim();
                                    let digits = raw
                                        .strip_suffix('T')
                                        .or_else(|| raw.strip_suffix('t'))
                                        .unwrap_or(raw);
                                    match digits.parse::<u32>() {
                                        Ok(ts) => {
                                            if wcet_tstates.is_some() {
                                                self.error(
                                                    format!("duplicate `#[wcet(...)]` attribute on `{}`", ef.name),
                                                    attr.span,
                                                );
                                            }
                                            wcet_tstates = Some(ts);
                                        }
                                        Err(_) => {
                                            self.error(
                                                format!("invalid T-state bound `{raw}` in `#[wcet(...)]`"),
                                                attr.span,
                                            );
                                        }
                                    }
                                }
                                "max_iters" => {
                                    if attr.args.is_empty() {
                                        self.error(
                                            "`#[max_iters(...)]` expects `N` or `label = N`",
                                            attr.span,
                                        );
                                        continue;
                                    }
                                    for arg in &attr.args {
                                        let raw = arg.trim();
                                        if let Some((lbl, val_part)) = raw.split_once('=') {
                                            let lbl = lbl.trim();
                                            let val_part = val_part.trim();
                                            if lbl.is_empty() {
                                                self.error(
                                                    "empty loop label in `#[max_iters(...)]`",
                                                    attr.span,
                                                );
                                                continue;
                                            }
                                            match val_part.parse::<u32>() {
                                                Ok(n) => {
                                                    loop_bounds.insert(lbl.to_string(), n);
                                                }
                                                Err(_) => {
                                                    self.error(
                                                        format!(
                                                            "invalid iteration bound `{val_part}` in `#[max_iters(...)]`"
                                                        ),
                                                        attr.span,
                                                    );
                                                }
                                            }
                                        } else {
                                            match raw.parse::<u32>() {
                                                Ok(n) => {
                                                    max_iters = Some(n);
                                                }
                                                Err(_) => {
                                                    self.error(
                                                        format!(
                                                            "invalid iteration bound `{raw}` in `#[max_iters(...)]`"
                                                        ),
                                                        attr.span,
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
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
                                if t.is_aggregate() {
                                    self.error(
                                        format!(
                                            "`extern \"asm\"` function `{}` cannot return aggregate type `{t:?}` by value (pass an `&mut` out-parameter instead)",
                                            ef.name
                                        ),
                                        ef.span,
                                    );
                                }
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
                            owner_state: ef_owner_state,
                            wcet_tstates,
                            max_iters,
                            loop_bounds,
                        });
                    }
                }
                Item::Fn(f) => {
                    let id = self.register_fn_decl(&f.name, f);
                    self.top_fn_ids.push(id);
                }
                _ => {}
            }
        }
    }

    fn resolve_overlay_owner_state(&mut self, arg: &str, span: Span) -> Option<(u32, u8)> {
        if let Some((mname, sname)) = arg.split_once("::")
            && let Some(&mid) = self.machine_ids.get(mname)
        {
            if let Some(st) = self.program.machines[mid as usize]
                .states
                .iter()
                .find(|s| s.name == sname)
            {
                return Some((mid, st.id));
            }
            // Try nested `[mod::]Machine::State...` if module prefix is present
        }
        let segs: Vec<&str> = arg.split("::").collect();
        for split in (1..segs.len()).rev() {
            let m_cand = segs[..split].join("::");
            if let Some(&mid) = self.machine_ids.get(&m_cand) {
                let s_cand = segs[split..].join("::");
                if let Some(st) = self.program.machines[mid as usize]
                    .states
                    .iter()
                    .find(|s| s.name == s_cand)
                {
                    return Some((mid, st.id));
                }
            }
        }
        // Fallback: unambiguous single state name across machines
        let mut matches = Vec::new();
        for m in &self.program.machines {
            if let Some(st) = m.states.iter().find(|s| s.name == arg) {
                matches.push((m.id, st.id));
            }
        }
        if matches.len() == 1 {
            return Some(matches[0]);
        }
        self.error(format!("unknown state `{arg}` in `#[overlay({arg})]`"), span);
        None
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
        if !self.fn_sigs.get(sig_name).is_some_and(|s| s.is_extern) {
            self.fn_sigs.insert(
                sig_name.to_string(),
                FnSig {
                    id,
                    is_extern: false,
                    params: sig_params.clone(),
                    ret_ty: ret_ty.clone(),
                },
            );
        }

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

        let mut locals: Vec<LocalSlot> = f
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

        let mir_ret_ty = if ret_ty.is_aggregate() {
            let sret_slot = locals.len() as SlotId;
            locals.push(LocalSlot {
                id: sret_slot,
                name: "__sret".to_string(),
                ty: IrType::Ref {
                    mutable: true,
                    inner: Box::new(ret_ty.clone()),
                },
                mutable: false,
            });
            self.fn_sret.insert(id, (sret_slot, ret_ty));
            IrType::Unit
        } else {
            ret_ty
        };

        let owner_state = f
            .attrs
            .iter()
            .find(|a| a.name == "overlay")
            .and_then(|a| a.args.first())
            .and_then(|arg| self.resolve_overlay_owner_state(arg, f.span));
        let mock_target = f
            .attrs
            .iter()
            .find(|a| a.name == "mock")
            .and_then(|a| a.args.first().cloned())
            .filter(|s| !s.is_empty());

        let mut cycle_budget = None;
        if let Some(budget_attr) = f.attrs.iter().find(|a| a.name == "budget") {
            if budget_attr.args.len() != 1 {
                self.error(
                    "`#[budget(...)]` expects a single argument such as `#[budget(8000T)]` or `#[budget(vblank = 8000T)]`",
                    budget_attr.span,
                );
            } else {
                let raw = budget_attr.args[0].trim();
                let (label, val_part) = if let Some((lhs, rhs)) = raw.split_once('=') {
                    let lbl = lhs.trim();
                    if lbl.is_empty() {
                        self.error("empty budget label in `#[budget(...)]`", budget_attr.span);
                    }
                    (Some(lbl.to_string()), rhs.trim())
                } else {
                    (None, raw)
                };
                let digits = val_part
                    .strip_suffix('T')
                    .or_else(|| val_part.strip_suffix('t'))
                    .unwrap_or(val_part);
                match digits.parse::<u32>() {
                    Ok(limit) => {
                        cycle_budget = Some((label, limit));
                    }
                    Err(_) => {
                        self.error(
                            format!("invalid T-state limit `{val_part}` in `#[budget(...)]`"),
                            budget_attr.span,
                        );
                    }
                }
            }
        }

        let requires_di = f
            .attrs
            .iter()
            .any(|a| a.name == "requires" && a.args.iter().any(|x| x == "di"));
        let export_name = self.parse_export_name(&f.attrs, "function");

        let mut seen_inline_always = false;
        let mut seen_inline_never = false;
        for attr in &f.attrs {
            match attr.name.as_str() {
                "inline" => {
                    if attr.args.is_empty() {
                        seen_inline_always = true;
                    } else if attr.args.len() == 1 {
                        match attr.args[0].as_str() {
                            "always" => seen_inline_always = true,
                            "never" => seen_inline_never = true,
                            other => {
                                self.error(
                                    format!(
                                        "unknown `#[inline(...)]` mode `{other}` (expected `always` or `never`)"
                                    ),
                                    attr.span,
                                );
                            }
                        }
                    } else {
                        self.error(
                            "`#[inline(...)]` expects at most one argument (`always` or `never`)",
                            attr.span,
                        );
                    }
                }
                "cold" => {
                    if !attr.args.is_empty() {
                        self.error("`#[cold]` takes no arguments", attr.span);
                    }
                    seen_inline_never = true;
                }
                _ => {}
            }
        }
        if seen_inline_always && seen_inline_never {
            self.error(
                format!(
                    "function `{sig_name}` cannot combine `#[inline]`/`#[inline(always)]` with `#[inline(never)]`/`#[cold]`"
                ),
                f.span,
            );
        }
        let inline_hint = if seen_inline_never {
            InlineHint::Never
        } else if seen_inline_always {
            InlineHint::Always
        } else {
            InlineHint::Auto
        };

        self.program.functions.push(IrFunction {
            id,
            name: sig_name.to_string(),
            is_pub: f.is_pub,
            inline_hint,
            param_count: locals.len(),
            locals,
            ret_ty: mir_ret_ty,
            is_test,
            expected_panic,
            is_reentrant,
            interrupt_source,
            isr_using,
            requires_banks: Vec::new(),
            requires_di,
            overlay_state: None,
            owner_state,
            mock_target,
            export_name,
            cycle_budget,
            loop_bounds: HashMap::new(),
            blocks: vec![BasicBlock::default()],
            next_val: 0,
        });
        id
    }

    pub(super) fn eval_const_expr(&mut self, expr: &Expr, expected_ty: &IrType) -> Option<i64> {
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
        let mut top_fn_idx = 0usize;
        for item in &file.items {
            match item {
                Item::Fn(f) => {
                    let id = self.top_fn_ids[top_fn_idx] as usize;
                    top_fn_idx += 1;
                    let mut lowerer = FnLowerer::new(self, id, None);
                    lowerer.lower_fn(f);
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
            owner_state: None,
            owned_by_irq: None,
            export_name: None,
            zero_on_enter: false,
            export_asm_layout: None,
            init: bytes.to_vec(),
        });
        id
    }

    fn parse_export_name(&mut self, attrs: &[Attr], item_kind: &str) -> Option<String> {
        let attr = attrs.iter().find(|a| a.name == "export_name")?;
        if attr.args.len() != 1 || attr.args[0].is_empty() {
            self.error(
                format!("`#[export_name]` on {item_kind} expects a non-empty symbol name string"),
                attr.span,
            );
            return None;
        }
        let name = &attr.args[0];
        if name.contains(char::is_whitespace) || name.starts_with(|c: char| c.is_ascii_digit()) {
            self.error(
                format!("invalid `#[export_name = \"{name}\"]` symbol identifier"),
                attr.span,
            );
            return None;
        }
        Some(name.clone())
    }

    fn parse_export_asm_layout(&mut self, attrs: &[Attr]) -> Option<String> {
        let attr = attrs.iter().find(|a| a.name == "export_asm_layout")?;
        if attr.args.is_empty() {
            return Some(String::new());
        }
        if attr.args.len() > 1 {
            self.error(
                "`#[export_asm_layout]` accepts at most one prefix argument",
                attr.span,
            );
            return None;
        }
        let raw = attr.args[0].trim();
        let prefix = raw
            .strip_prefix('"')
            .and_then(|s| s.strip_suffix('"'))
            .unwrap_or(raw)
            .to_string();
        if prefix.is_empty()
            || prefix.starts_with(|c: char| c.is_ascii_digit())
            || !prefix.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            self.error(
                format!("invalid `#[export_asm_layout(\"{prefix}\")]` identifier prefix"),
                attr.span,
            );
            return None;
        }
        Some(prefix)
    }

    fn parse_zero_on_enter(&mut self, attrs: &[Attr], is_mut: bool, in_state: bool) -> bool {
        let Some(attr) = attrs.iter().find(|a| a.name == "zero_on_enter") else {
            return false;
        };
        if !attr.args.is_empty() {
            self.error("`#[zero_on_enter]` takes no arguments", attr.span);
        }
        if !in_state {
            self.error(
                "`#[zero_on_enter]` is only valid on state-scoped `static mut` buffers inside a `machine` state",
                attr.span,
            );
            return false;
        }
        if !is_mut {
            self.error(
                "`#[zero_on_enter]` requires a mutable `static mut` state buffer",
                attr.span,
            );
            return false;
        }
        true
    }

    fn parse_owned_by_irq(&mut self, attrs: &[Attr], is_mut: bool, span: Span) -> Option<String> {
        let attr = attrs.iter().find(|a| a.name == "owned_by")?;
        if attr.args.len() != 1 {
            self.error(
                "`#[owned_by(IrqName)]` expects 1 interrupt source argument",
                attr.span,
            );
            return None;
        }
        if !is_mut {
            self.error(
                "`#[owned_by(...)]` is only valid on `static mut` globals",
                span,
            );
        }
        let src_name = attr.args[0].clone();
        if let Some(target) = &self.program.target {
            let exists = target
                .interrupt_controllers
                .iter()
                .any(|ic| ic.sources.iter().any(|s| s.name == src_name));
            if !exists {
                self.error(
                    format!("unknown interrupt source `{src_name}` in `#[owned_by]`"),
                    attr.span,
                );
            }
        }
        Some(src_name)
    }

    pub(super) fn check_queue_spsc_safety(&mut self) {
        if self.queue_spsc_accesses.is_empty() {
            return;
        }
        let n = self.program.functions.len();
        let mut callees: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut has_non_isr_caller = vec![false; n];
        let mut has_isr_caller = vec![false; n];
        let mut is_machine_or_addr_taken = vec![false; n];

        for m in &self.program.machines {
            for st in &m.states {
                if let Some(fid) = st.on_enter {
                    is_machine_or_addr_taken[fid as usize] = true;
                }
                if let Some(fid) = st.on_exit {
                    is_machine_or_addr_taken[fid as usize] = true;
                }
                for &fid in st.methods.values() {
                    is_machine_or_addr_taken[fid as usize] = true;
                }
            }
        }
        for (i, f) in self.program.functions.iter().enumerate() {
            let caller_is_isr = self.program.is_isr_root(f);
            for bb in &f.blocks {
                for inst in &bb.insts {
                    match inst {
                        IrInst::Call { func: cid, .. } if (*cid as usize) < n => {
                            let c = *cid as usize;
                            callees[i].push(c);
                            if caller_is_isr {
                                has_isr_caller[c] = true;
                            } else {
                                has_non_isr_caller[c] = true;
                            }
                        }
                        IrInst::AddrFn {
                            target: IrFnTarget::Fn(cid),
                            ..
                        } if (*cid as usize) < n => {
                            is_machine_or_addr_taken[*cid as usize] = true;
                        }
                        IrInst::StateTransition { machine_id, .. } => {
                            let m = &self.program.machines[*machine_id as usize];
                            for st in &m.states {
                                for fid in st.on_enter.into_iter().chain(st.on_exit) {
                                    callees[i].push(fid as usize);
                                    if caller_is_isr {
                                        has_isr_caller[fid as usize] = true;
                                    } else {
                                        has_non_isr_caller[fid as usize] = true;
                                    }
                                }
                            }
                        }
                        IrInst::StateDispatch {
                            machine_id, method, ..
                        } => {
                            let m = &self.program.machines[*machine_id as usize];
                            for st in &m.states {
                                if let Some(&fid) = st.methods.get(method) {
                                    callees[i].push(fid as usize);
                                    if caller_is_isr {
                                        has_isr_caller[fid as usize] = true;
                                    } else {
                                        has_non_isr_caller[fid as usize] = true;
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        let mut isr_reachers: Vec<std::collections::BTreeSet<String>> =
            vec![std::collections::BTreeSet::new(); n];
        let mut work = std::collections::VecDeque::new();
        for (i, f) in self.program.functions.iter().enumerate() {
            if let Some(ref src) = f.interrupt_source {
                isr_reachers[i].insert(src.clone());
                work.push_back(i);
            } else if self.program.isr_entries.contains_key(&f.id) {
                isr_reachers[i].insert(format!("isr_entry:{}", f.name));
                work.push_back(i);
            }
        }
        while let Some(u) = work.pop_front() {
            let srcs: Vec<String> = isr_reachers[u].iter().cloned().collect();
            for &v in &callees[u] {
                let mut added = false;
                for s in &srcs {
                    if isr_reachers[v].insert(s.clone()) {
                        added = true;
                    }
                }
                if added {
                    work.push_back(v);
                }
            }
        }

        let mut main_reachable = vec![false; n];
        let mut mwork = std::collections::VecDeque::new();
        for (i, f) in self.program.functions.iter().enumerate() {
            if self.program.is_isr_root(f) {
                continue;
            }
            if f.is_pub
                || f.is_test
                || f.name == "main"
                || is_machine_or_addr_taken[i]
                || has_non_isr_caller[i]
                || !has_isr_caller[i]
            {
                main_reachable[i] = true;
                mwork.push_back(i);
            }
        }
        while let Some(u) = mwork.pop_front() {
            for &v in &callees[u] {
                if !self.program.is_isr_root(&self.program.functions[v]) && !main_reachable[v] {
                    main_reachable[v] = true;
                    mwork.push_back(v);
                }
            }
        }

        let mut seen: HashMap<(u32, bool), (std::collections::BTreeSet<String>, Span)> =
            HashMap::new();
        let accesses = std::mem::take(&mut self.queue_spsc_accesses);
        for (fn_idx, gid, is_push, span) in accesses {
            let mut ctxs = Vec::new();
            for s in &isr_reachers[fn_idx] {
                ctxs.push(format!("Irq({s})"));
            }
            if main_reachable[fn_idx] || isr_reachers[fn_idx].is_empty() {
                ctxs.push("Main".to_string());
            }
            let entry = seen
                .entry((gid, is_push))
                .or_insert_with(|| (std::collections::BTreeSet::new(), span));
            let prev_len = entry.0.len();
            for c in ctxs {
                entry.0.insert(c);
            }
            if prev_len <= 1 && entry.0.len() > 1 {
                let gname = self.program.globals[gid as usize].name.clone();
                let op = if is_push { "push" } else { "pop" };
                let ctx_list: Vec<String> = entry.0.iter().cloned().collect();
                self.error(
                    format!(
                        "SPSC safety violation: multiple contexts ({}) call `.{op}()` on `{gname}` without `critical`/`di`",
                        ctx_list.join(", ")
                    ),
                    span,
                );
            }
        }
    }
}
