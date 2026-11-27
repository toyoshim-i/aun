//! MCS-51 / WCH CH559 memory layout, pointer-space inference, and *Uchiguri* frame allocator.

use aun_ir::*;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PtrSpace {
    Data,
    IData,
    XData,
    Code,
    Sfr,
    Bit,
    Mixed,
}

impl PtrSpace {
    pub fn from_addr_space(s: IrAddrSpace) -> Self {
        match s {
            IrAddrSpace::Data => Self::Data,
            IrAddrSpace::IData => Self::IData,
            IrAddrSpace::XData | IrAddrSpace::Flat => Self::XData,
            IrAddrSpace::Code => Self::Code,
            IrAddrSpace::Sfr => Self::Sfr,
            IrAddrSpace::Bit => Self::Bit,
        }
    }

    pub fn join(self, other: Self) -> Self {
        if self == other {
            self
        } else {
            Self::Mixed
        }
    }

    pub fn tag_id(self) -> u8 {
        match self {
            Self::Data | Self::IData => 1,
            Self::XData => 2,
            Self::Code => 3,
            Self::Sfr => 4,
            Self::Bit => 5,
            Self::Mixed => 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotRegion {
    Data(u8),
    XData(u16),
}

#[derive(Debug, Clone)]
pub struct GlobalAlloc {
    pub global_id: u32,
    pub space: PtrSpace,
    pub addr: u16,
    pub size: u16,
    pub rom_label: Option<String>,
}

#[derive(Debug, Clone)]
pub struct InitSpan {
    pub dst_addr: u16,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct Mcs51Layout {
    pub data_start: u8,
    pub data_end: u8,
    pub sp_init: u8,
    pub max_stack_bytes: u16,
    pub xdata_start: u16,
    pub xdata_end: u16,
    pub code_start: u16,
    pub code_end: u16,
    pub globals: Vec<GlobalAlloc>,
    pub data_inits: Vec<InitSpan>,
    pub xdata_inits: Vec<InitSpan>,
    /// `(func_id, slot_id) -> SlotRegion`
    pub slot_locs: HashMap<(u32, SlotId), SlotRegion>,
    /// `(func_id, slot_id) -> optional Data address for 1B PtrSpace tag when Mixed`
    pub slot_ptr_tags: HashMap<(u32, SlotId), SlotRegion>,
    /// `(func_id, val_id) -> SlotRegion`
    pub val_locs: HashMap<(u32, ValueId), SlotRegion>,
    /// `(func_id, val_id) -> optional 1B PtrSpace tag when Mixed`
    pub val_ptr_tags: HashMap<(u32, ValueId), SlotRegion>,
    /// `(func_id, val_id) -> static PtrSpace` (exact unless `Mixed`)
    pub val_ptr_spaces: HashMap<(u32, ValueId), PtrSpace>,
    /// `(func_id, slot_id) -> static PtrSpace` (exact unless `Mixed`)
    pub slot_ptr_spaces: HashMap<(u32, SlotId), PtrSpace>,
    /// Call-graph reachable functions per root (`None` = main, `Some(id)` = ISR)
    pub fn_callees: Vec<Vec<usize>>,
}

impl Mcs51Layout {
    pub fn global(&self, id: u32) -> &GlobalAlloc {
        &self.globals[id as usize]
    }
}

fn build_callees(program: &IrProgram) -> Vec<Vec<usize>> {
    let n = program.functions.len();
    let mut addr_taken_fns = Vec::new();
    let mut seen_taken = HashSet::new();
    for f in &program.functions {
        for bb in &f.blocks {
            for inst in &bb.insts {
                if let IrInst::AddrFn {
                    target: IrFnTarget::Fn(fid),
                    ..
                } = inst
                {
                    let c = *fid as usize;
                    if c < n && seen_taken.insert(c) {
                        addr_taken_fns.push(c);
                    }
                }
            }
        }
    }
    let mut callees = vec![Vec::new(); n];
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
                        for anc in mdef.ancestors(*to_state) {
                            let below = from_state.is_none_or(|from| {
                                let lca = mdef.lca(from, *to_state);
                                lca.is_none_or(|l| anc != l && mdef.is_ancestor(l, anc))
                            });
                            if below && let Some(enter_fn) = mdef.states[anc as usize].on_enter {
                                let c = enter_fn as usize;
                                if c < n && seen.insert(c) {
                                    callees[i].push(c);
                                }
                            }
                        }
                        if let Some(from) = from_state {
                            let lca = mdef.lca(*from, *to_state);
                            for anc in mdef.ancestors(*from) {
                                let below = lca.is_none_or(|l| anc != l && mdef.is_ancestor(l, anc));
                                if below && let Some(exit_fn) = mdef.states[anc as usize].on_exit {
                                    let c = exit_fn as usize;
                                    if c < n && seen.insert(c) {
                                        callees[i].push(c);
                                    }
                                }
                            }
                        } else {
                            for st in &mdef.states {
                                if let Some(exit_fn) = st.on_exit {
                                    let c = exit_fn as usize;
                                    if c < n && seen.insert(c) {
                                        callees[i].push(c);
                                    }
                                }
                            }
                        }
                    }
                    IrInst::StateDispatch {
                        machine_id, method, ..
                    } => {
                        let mdef = &program.machines[*machine_id as usize];
                        for st in &mdef.states {
                            if let Some(&cid) = st.methods.get(method) {
                                let c = cid as usize;
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

fn collect_reachable(roots: &[usize], callees: &[Vec<usize>]) -> HashSet<usize> {
    let mut visited = HashSet::new();
    let mut stack: Vec<usize> = roots.to_vec();
    while let Some(u) = stack.pop() {
        if !visited.insert(u) {
            continue;
        }
        for &v in &callees[u] {
            if !visited.contains(&v) {
                stack.push(v);
            }
        }
    }
    visited
}

fn check_acyclic_and_topo(nodes: &HashSet<usize>, callees: &[Vec<usize>], program: &IrProgram) -> Result<Vec<usize>, String> {
    let mut state: HashMap<usize, u8> = HashMap::new(); // 0=unvisited, 1=visiting, 2=done
    let mut order = Vec::new();
    fn dfs(
        u: usize,
        nodes: &HashSet<usize>,
        callees: &[Vec<usize>],
        program: &IrProgram,
        state: &mut HashMap<usize, u8>,
        order: &mut Vec<usize>,
    ) -> Result<(), String> {
        state.insert(u, 1);
        for &v in &callees[u] {
            if !nodes.contains(&v) {
                continue;
            }
            match state.get(&v).copied().unwrap_or(0) {
                1 => {
                    if !program.functions[v].is_reentrant {
                        return Err(format!(
                            "non-reentrant recursion involving `{}` is rejected on MCS-51",
                            program.functions[v].name
                        ));
                    }
                }
                0 => dfs(v, nodes, callees, program, state, order)?,
                _ => {}
            }
        }
        state.insert(u, 2);
        order.push(u);
        Ok(())
    }
    let mut sorted: Vec<usize> = nodes.iter().copied().collect();
    sorted.sort_unstable();
    for u in sorted {
        if state.get(&u).copied().unwrap_or(0) == 0 {
            dfs(u, nodes, callees, program, &mut state, &mut order)?;
        }
    }
    order.reverse(); // callers before callees
    Ok(order)
}

fn is_ptr_like_ty(ty: &IrType) -> bool {
    matches!(
        ty,
        IrType::Ref { .. } | IrType::Slice { .. } | IrType::ConstSlice(_)
    )
}

fn infer_ptr_spaces(
    program: &IrProgram,
    global_spaces: &[PtrSpace],
    slot_spaces: &HashMap<(u32, SlotId), SlotRegion>,
) -> (
    HashMap<(u32, ValueId), PtrSpace>,
    HashMap<(u32, SlotId), PtrSpace>,
) {
    let mut val_sp: HashMap<(u32, ValueId), PtrSpace> = HashMap::new();
    let mut slot_sp: HashMap<(u32, SlotId), PtrSpace> = HashMap::new();

    let join_val = |map: &mut HashMap<(u32, ValueId), PtrSpace>, k: (u32, ValueId), sp: PtrSpace| -> bool {
        match map.get(&k).copied() {
            Some(cur) => {
                let next = cur.join(sp);
                if next != cur {
                    map.insert(k, next);
                    true
                } else {
                    false
                }
            }
            None => {
                map.insert(k, sp);
                true
            }
        }
    };
    let join_slot = |map: &mut HashMap<(u32, SlotId), PtrSpace>, k: (u32, SlotId), sp: PtrSpace| -> bool {
        match map.get(&k).copied() {
            Some(cur) => {
                let next = cur.join(sp);
                if next != cur {
                    map.insert(k, next);
                    true
                } else {
                    false
                }
            }
            None => {
                map.insert(k, sp);
                true
            }
        }
    };

    let mut addr_of_slot: HashMap<(u32, ValueId), SlotId> = HashMap::new();
    for f in &program.functions {
        for bb in &f.blocks {
            for inst in &bb.insts {
                if let IrInst::AddrLocal { dst, slot } = inst {
                    addr_of_slot.insert((f.id, *dst), *slot);
                }
            }
        }
    }

    for _ in 0..32 {
        let mut changed = false;
        for f in &program.functions {
            for bb in &f.blocks {
                for inst in &bb.insts {
                    match inst {
                        IrInst::AddrGlobal { dst, global } => {
                            let sp = global_spaces[*global as usize];
                            changed |= join_val(&mut val_sp, (f.id, *dst), sp);
                        }
                        IrInst::AddrLocal { dst, slot } => {
                            let sp = match slot_spaces.get(&(f.id, *slot)) {
                                Some(SlotRegion::Data(_)) => PtrSpace::Data,
                                Some(SlotRegion::XData(_)) => PtrSpace::XData,
                                None => PtrSpace::Data,
                            };
                            changed |= join_val(&mut val_sp, (f.id, *dst), sp);
                        }
                        IrInst::FieldAddr { dst, base, .. }
                        | IrInst::IndexAddr { dst, base, .. } => {
                            if let Some(&sp) = val_sp.get(&(f.id, *base)) {
                                changed |= join_val(&mut val_sp, (f.id, *dst), sp);
                            }
                        }
                        IrInst::MakeSlice { dst, ptr, .. } => {
                            if let Some(&sp) = val_sp.get(&(f.id, *ptr)) {
                                changed |= join_val(&mut val_sp, (f.id, *dst), sp);
                            }
                        }
                        IrInst::SlicePtr { dst, slice, .. } => {
                            if let Some(&sp) = val_sp.get(&(f.id, *slice)) {
                                changed |= join_val(&mut val_sp, (f.id, *dst), sp);
                            }
                        }
                        IrInst::Cast {
                            dst,
                            val,
                            from_ty,
                            to_ty,
                        } if is_ptr_like_ty(from_ty) || is_ptr_like_ty(to_ty) => {
                            if let Some(&sp) = val_sp.get(&(f.id, *val)) {
                                changed |= join_val(&mut val_sp, (f.id, *dst), sp);
                            }
                        }
                        IrInst::Cast {
                            dst, val, from_ty, to_ty
                        } if program.type_size(from_ty) == 2 && program.type_size(to_ty) == 2 => {
                            if let Some(&sp) = val_sp.get(&(f.id, *val)) {
                                changed |= join_val(&mut val_sp, (f.id, *dst), sp);
                            }
                        }
                        IrInst::LoadLocal { dst, slot, ty }
                            if is_ptr_like_ty(ty) || program.type_size(ty) == 2 =>
                        {
                            if let Some(&sp) = slot_sp.get(&(f.id, *slot)) {
                                changed |= join_val(&mut val_sp, (f.id, *dst), sp);
                            }
                        }
                        IrInst::StoreLocal { slot, val, ty }
                            if is_ptr_like_ty(ty) || program.type_size(ty) == 2 =>
                        {
                            if let Some(&sp) = val_sp.get(&(f.id, *val)) {
                                changed |= join_slot(&mut slot_sp, (f.id, *slot), sp);
                            }
                        }
                        IrInst::LoadPtr { dst, addr, ty, .. }
                            if is_ptr_like_ty(ty) || program.type_size(ty) == 2 =>
                        {
                            if let Some(&slot) = addr_of_slot.get(&(f.id, *addr))
                                && let Some(&sp) = slot_sp.get(&(f.id, slot))
                            {
                                changed |= join_val(&mut val_sp, (f.id, *dst), sp);
                            }
                        }
                        IrInst::StorePtr { addr, val, ty, .. }
                            if is_ptr_like_ty(ty) || program.type_size(ty) == 2 =>
                        {
                            if let Some(&slot) = addr_of_slot.get(&(f.id, *addr))
                                && let Some(&sp) = val_sp.get(&(f.id, *val))
                            {
                                changed |= join_slot(&mut slot_sp, (f.id, slot), sp);
                            }
                        }
                        IrInst::Phi { dst, incoming, ty }
                            if is_ptr_like_ty(ty) || program.type_size(ty) == 2 =>
                        {
                            for (_, v) in incoming {
                                if let Some(&sp) = val_sp.get(&(f.id, *v)) {
                                    changed |= join_val(&mut val_sp, (f.id, *dst), sp);
                                }
                            }
                        }
                        IrInst::Call { dst, func: cid, args } => {
                            for (pi, &av) in args.iter().enumerate() {
                                if let Some(&sp) = val_sp.get(&(f.id, av)) {
                                    changed |= join_slot(&mut slot_sp, (*cid, pi as SlotId), sp);
                                }
                            }
                            if let Some(d) = dst {
                                let callee = &program.functions[*cid as usize];
                                if is_ptr_like_ty(&callee.ret_ty) {
                                    for cbb in &callee.blocks {
                                        if let Terminator::Return(Some(rv)) = cbb.term
                                            && let Some(&sp) = val_sp.get(&(*cid, rv))
                                        {
                                            changed |= join_val(&mut val_sp, (f.id, *d), sp);
                                        }
                                    }
                                }
                            }
                        }
                        IrInst::StateTransition { machine_addr, machine_id, .. } => {
                            if let Some(&sp) = val_sp.get(&(f.id, *machine_addr)) {
                                let mdef = &program.machines[*machine_id as usize];
                                for st in &mdef.states {
                                    if let Some(eid) = st.on_enter {
                                        changed |= join_slot(&mut slot_sp, (eid, 0), sp);
                                    }
                                    if let Some(xid) = st.on_exit {
                                        changed |= join_slot(&mut slot_sp, (xid, 0), sp);
                                    }
                                }
                            }
                        }
                        IrInst::StateDispatch {
                            dst,
                            machine_addr,
                            machine_id,
                            method,
                            args,
                        } => {
                            let mdef = &program.machines[*machine_id as usize];
                            for st in &mdef.states {
                                if let Some(&cid) = st.methods.get(method) {
                                    if let Some(&sp) = val_sp.get(&(f.id, *machine_addr)) {
                                        changed |= join_slot(&mut slot_sp, (cid, 0), sp);
                                    }
                                    for (ai, &av) in args.iter().enumerate() {
                                        if let Some(&sp) = val_sp.get(&(f.id, av)) {
                                            changed |=
                                                join_slot(&mut slot_sp, (cid, (ai + 1) as SlotId), sp);
                                        }
                                    }
                                    if let Some(d) = dst {
                                        let callee = &program.functions[cid as usize];
                                        if is_ptr_like_ty(&callee.ret_ty) {
                                            for cbb in &callee.blocks {
                                                if let Terminator::Return(Some(rv)) = cbb.term
                                                    && let Some(&sp) = val_sp.get(&(cid, rv))
                                                {
                                                    changed |= join_val(&mut val_sp, (f.id, *d), sp);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    (val_sp, slot_sp)
}

pub fn val_byte_sizes(program: &IrProgram, func: &IrFunction) -> Vec<u16> {
    let mut sizes = vec![0u16; func.next_val as usize];
    let mut set = |v: ValueId, sz: u16| {
        if let Some(slot) = sizes.get_mut(v as usize) {
            *slot = (*slot).max(sz);
        }
    };
    for bb in &func.blocks {
        for inst in &bb.insts {
            match inst {
                IrInst::ConstInt { dst, ty, .. } => set(*dst, program.type_size(ty)),
                IrInst::ConstBool { dst, .. } => set(*dst, 1),
                IrInst::BinOp { dst, op, ty, .. } => {
                    let sz = match op {
                        IrBinOp::Eq
                        | IrBinOp::Ne
                        | IrBinOp::Lt
                        | IrBinOp::Le
                        | IrBinOp::Gt
                        | IrBinOp::Ge => 1,
                        _ => program.type_size(ty),
                    };
                    set(*dst, sz);
                }
                IrInst::UnOp { dst, ty, .. } => set(*dst, program.type_size(ty)),
                IrInst::Cast { dst, to_ty, .. } => set(*dst, program.type_size(to_ty)),
                IrInst::FixedInt { dst, base, .. } => set(*dst, base.byte_size()),
                IrInst::LoadLocal { dst, ty, .. } | IrInst::LoadPtr { dst, ty, .. } => {
                    set(*dst, program.type_size(ty));
                }
                IrInst::AddrLocal { dst, .. }
                | IrInst::AddrGlobal { dst, .. }
                | IrInst::AddrFn { dst, .. }
                | IrInst::FieldAddr { dst, .. }
                | IrInst::IndexAddr { dst, .. }
                | IrInst::SlicePtr { dst, .. }
                | IrInst::SliceLen { dst, .. }
                | IrInst::PoolElemAddr { dst, .. } => set(*dst, 2),
                IrInst::MakeSlice { dst, ty, .. } => set(*dst, program.type_size(ty)),
                IrInst::StateTag { dst, .. }
                | IrInst::PoolAlloc { dst, .. }
                | IrInst::QueuePush { dst, .. }
                | IrInst::QueuePop { dst, .. }
                | IrInst::QueueIsEmpty { dst, .. }
                | IrInst::QueueLen { dst, .. }
                | IrInst::PortIn { dst, .. } => set(*dst, 1),
                IrInst::Call { dst: Some(d), func: cid, .. } => {
                    set(*d, program.type_size(&program.functions[*cid as usize].ret_ty));
                }
                IrInst::ExternCall {
                    dst: Some(d),
                    extern_fn,
                    ..
                } => {
                    let sz = program.externs[*extern_fn as usize]
                        .ret
                        .as_ref()
                        .map_or(0, |(t, _)| program.type_size(t));
                    set(*d, sz);
                }
                IrInst::StateDispatch {
                    dst: Some(d),
                    machine_id,
                    method,
                    ..
                } => {
                    let mdef = &program.machines[*machine_id as usize];
                    let sz = mdef
                        .states
                        .iter()
                        .find_map(|s| s.methods.get(method))
                        .map_or(0, |&cid| program.type_size(&program.functions[cid as usize].ret_ty));
                    set(*d, sz);
                }
                IrInst::Phi { dst, ty, .. } => set(*dst, program.type_size(ty)),
                _ => {}
            }
        }
    }
    sizes
}

struct IrqTree<'a> {
    nodes: Vec<usize>,
    callees: &'a [Vec<usize>],
}

impl<'a> IrqTree<'a> {
    fn compute_call_graph_frames(
        &self,
        nodes_set: &HashSet<usize>,
        local_sizes: &[u16],
        program: &IrProgram,
    ) -> HashMap<usize, u16> {
        // Assign disjoint intervals to caller/callee chains and reuse intervals across siblings.
        // Returns start offset relative to region base for each function in `self.nodes`.
        let mut reaches: Vec<HashSet<usize>> = vec![HashSet::new(); program.functions.len()];
        for &u in self.nodes.iter().rev() {
            let mut r = HashSet::new();
            r.insert(u);
            for &v in &self.callees[u] {
                if nodes_set.contains(&v) && v != u {
                    r.insert(v);
                    let child_r: Vec<usize> = reaches[v].iter().copied().collect();
                    r.extend(child_r);
                }
            }
            reaches[u] = r;
        }
        let mut assigned: HashMap<usize, (u16, u16)> = HashMap::new(); // fn -> (start, end)
        for &u in &self.nodes {
            let sz = local_sizes[u];
            let mut busy: Vec<(u16, u16)> = Vec::new();
            for &other in &self.nodes {
                if other == u {
                    continue;
                }
                if let Some(&range) = assigned.get(&other)
                    && (reaches[u].contains(&other) || reaches[other].contains(&u))
                {
                    busy.push(range);
                }
            }
            busy.sort_unstable();
            let mut pos = 0u16;
            for (s, e) in busy {
                if pos + sz <= s {
                    break;
                }
                if e > pos {
                    pos = e;
                }
            }
            assigned.insert(u, (pos, pos + sz));
        }
        assigned.into_iter().map(|(k, (s, _))| (k, s)).collect()
    }
}

pub fn compute_layout(program: &IrProgram, code_origin: u16) -> Result<Mcs51Layout, String> {
    let mut data_range = (0x08u16, 0x7Fu16);
    let mut idata_range = (0x80u16, 0xFFu16);
    let mut xdata_range = (0x0000u16, 0x17FFu16);
    let mut code_range = (code_origin, 0xEFFFu16);

    if let Some(t) = &program.target {
        for sp in &t.spaces {
            match sp.kind {
                IrAddrSpace::Data => data_range = (sp.start.max(0x08), sp.end.min(0x7F)),
                IrAddrSpace::IData => idata_range = (sp.start.max(0x80), sp.end.min(0xFF)),
                IrAddrSpace::XData => xdata_range = (sp.start, sp.end),
                IrAddrSpace::Code => code_range = (sp.start.max(code_origin), sp.end),
                _ => {}
            }
        }
    }

    // Reserve register bank 0 (0x00..=0x07) plus any bank used by `#[interrupt(..., using(N))]`.
    let mut reserved_banks = [false; 4];
    reserved_banks[0] = true;
    for f in &program.functions {
        if let Some(b) = f.isr_using
            && (b as usize) < 4
        {
            reserved_banks[b as usize] = true;
        }
    }
    let mut data_used = [false; 256];
    for (b, &used) in reserved_banks.iter().enumerate() {
        if used {
            for addr in b * 8..(b + 1) * 8 {
                data_used[addr] = true;
            }
        }
    }
    for addr in (data_range.1 as usize + 1)..256 {
        data_used[addr] = true;
    }

    // Check `#[space(Name)]` lookup helper.
    let space_kind_of = |name: &str| -> Option<IrAddrSpace> {
        program
            .target
            .as_ref()?
            .spaces
            .iter()
            .find(|s| s.name == name)
            .map(|s| s.kind)
    };

    let mut global_allocs: Vec<GlobalAlloc> = Vec::with_capacity(program.globals.len());
    let mut data_inits = Vec::new();
    let mut xdata_inits = Vec::new();
    let mut xdata_used_intervals: Vec<(u16, u16)> = Vec::new();

    // First pass: fixed-address & ROM/SFR/Bit globals.
    let mut pending_globals: Vec<usize> = Vec::new();
    for (gid, g) in program.globals.iter().enumerate() {
        let size = program.type_size(&g.ty).max(1);
        let req_space = g
            .space
            .as_deref()
            .and_then(&space_kind_of)
            .or_else(|| g.in_rom.then_some(IrAddrSpace::Code));

        global_allocs.push(GlobalAlloc {
            global_id: gid as u32,
            space: PtrSpace::XData,
            addr: 0,
            size,
            rom_label: None,
        });

        if g.is_extern {
            if let Some(fa) = g.fixed_addr {
                let sp = req_space.map_or(PtrSpace::XData, PtrSpace::from_addr_space);
                global_allocs[gid].space = sp;
                global_allocs[gid].addr = fa;
            }
            continue;
        }

        if g.in_rom || req_space == Some(IrAddrSpace::Code) {
            global_allocs[gid].space = PtrSpace::Code;
            global_allocs[gid].rom_label = Some(format!("__aun_rom_g{gid}"));
            if let Some(fa) = g.fixed_addr {
                global_allocs[gid].addr = fa;
            }
            continue;
        }

        if matches!(req_space, Some(IrAddrSpace::Sfr | IrAddrSpace::Bit)) {
            let fa = g.fixed_addr.ok_or_else(|| {
                format!("global `{}` in Sfr/Bit space requires an `@ addr` address", g.name)
            })?;
            global_allocs[gid].space = PtrSpace::from_addr_space(req_space.unwrap());
            global_allocs[gid].addr = fa;
            continue;
        }

        if let Some(fa) = g.fixed_addr {
            let sp = req_space.map_or(PtrSpace::XData, PtrSpace::from_addr_space);
            global_allocs[gid].space = sp;
            global_allocs[gid].addr = fa;
            match sp {
                PtrSpace::Data => {
                    for off in 0..size {
                        let a = fa as usize + off as usize;
                        if a >= 128 {
                            return Err(format!("Data global `{}` @ 0x{fa:02X} exceeds Data RAM", g.name));
                        }
                        data_used[a] = true;
                    }
                    if g.init.iter().any(|&b| b != 0) {
                        data_inits.push(InitSpan {
                            dst_addr: fa,
                            bytes: g.init.clone(),
                        });
                    }
                }
                PtrSpace::XData => {
                    xdata_used_intervals.push((fa, fa.saturating_add(size)));
                    if g.init.iter().any(|&b| b != 0) {
                        xdata_inits.push(InitSpan {
                            dst_addr: fa,
                            bytes: g.init.clone(),
                        });
                    }
                }
                _ => {}
            }
            continue;
        }
        pending_globals.push(gid);
    }

    fn alloc_data(
        data_used: &mut [bool; 256],
        data_range: (u16, u16),
        sz: u16,
        align: u16,
    ) -> Option<u8> {
        let a = align.max(1) as usize;
        let sz = sz as usize;
        let mut start = data_range.0 as usize;
        while start + sz <= data_range.1 as usize + 1 {
            if !start.is_multiple_of(a) {
                start += a - (start % a);
                continue;
            }
            if (start..start + sz).all(|x| !data_used[x]) {
                for slot in &mut data_used[start..start + sz] {
                    *slot = true;
                }
                return Some(start as u8);
            }
            start += 1;
        }
        None
    }

    fn alloc_xdata(
        xdata_used_intervals: &mut Vec<(u16, u16)>,
        xdata_range: (u16, u16),
        sz: u16,
        align: u16,
    ) -> Result<u16, String> {
        let a = align.max(1);
        xdata_used_intervals.sort_unstable();
        let mut pos = xdata_range.0;
        if !pos.is_multiple_of(a) {
            pos = pos.saturating_add(a - (pos % a));
        }
        for &(s, e) in xdata_used_intervals.iter() {
            if pos.saturating_add(sz) <= s {
                break;
            }
            if e > pos {
                pos = e;
                if !pos.is_multiple_of(a) {
                    pos = pos.saturating_add(a - (pos % a));
                }
            }
        }
        if pos.saturating_add(sz) > xdata_range.1.saturating_add(1) {
            return Err(format!(
                "MCS-51 XData RAM overflow: need {} bytes at 0x{pos:04X}, limit 0x{:04X}",
                sz, xdata_range.1
            ));
        }
        xdata_used_intervals.push((pos, pos + sz));
        Ok(pos)
    }

    for gid in pending_globals {
        let g = &program.globals[gid];
        let size = global_allocs[gid].size;
        let req_space = g.space.as_deref().and_then(&space_kind_of);
        let prefer_data = match req_space {
            Some(IrAddrSpace::Data) => true,
            Some(IrAddrSpace::XData) => false,
            _ => size <= 4 && !matches!(g.ty, IrType::Machine(_) | IrType::Pool { .. } | IrType::Queue { .. }),
        };
        if prefer_data && let Some(addr) = alloc_data(&mut data_used, data_range, size, g.align) {
            global_allocs[gid].space = PtrSpace::Data;
            global_allocs[gid].addr = addr as u16;
            if g.init.iter().any(|&b| b != 0) {
                data_inits.push(InitSpan {
                    dst_addr: addr as u16,
                    bytes: g.init.clone(),
                });
            }
        } else {
            if req_space == Some(IrAddrSpace::Data) {
                return Err(format!("MCS-51 Data RAM overflow for `{}` ({size} B)", g.name));
            }
            let addr = alloc_xdata(&mut xdata_used_intervals, xdata_range, size, g.align)?;
            global_allocs[gid].space = PtrSpace::XData;
            global_allocs[gid].addr = addr;
            if g.init.iter().any(|&b| b != 0) {
                xdata_inits.push(InitSpan {
                    dst_addr: addr,
                    bytes: g.init.clone(),
                });
            }
        }
    }

    // Build call trees: Main tree + separate ISR trees.
    let fn_callees = build_callees(program);
    let mut main_roots = Vec::new();
    let mut isr_roots = Vec::new();
    for f in &program.functions {
        if f.is_test {
            continue;
        }
        if program.is_isr_root(f) {
            isr_roots.push(f.id as usize);
        } else if f.name == "main" || f.is_pub {
            main_roots.push(f.id as usize);
        }
    }
    if main_roots.is_empty() && !program.functions.is_empty() {
        for f in &program.functions {
            if !f.is_test && !program.is_isr_root(f) {
                main_roots.push(f.id as usize);
            }
        }
    }
    let main_reach = collect_reachable(&main_roots, &fn_callees);
    let mut trees: Vec<(Option<usize>, HashSet<usize>, Vec<usize>)> = Vec::new();
    if !main_reach.is_empty() {
        let topo = check_acyclic_and_topo(&main_reach, &fn_callees, program)?;
        trees.push((None, main_reach.clone(), topo));
    }
    for &ir in &isr_roots {
        let mut rset = HashSet::new();
        rset.insert(ir);
        let reach = collect_reachable(&[ir], &fn_callees);
        for &fn_id in &reach {
            if main_reach.contains(&fn_id) && !program.functions[fn_id].is_reentrant {
                return Err(format!(
                    "function `{}` is called by both main thread and ISR `{}` without `#[reentrant]`",
                    program.functions[fn_id].name, program.functions[ir].name
                ));
            }
        }
        let topo = check_acyclic_and_topo(&reach, &fn_callees, program)?;
        trees.push((Some(ir), reach, topo));
    }

    // Decide per-slot & per-val sizes (escaped / large local aggregates go to XData so IRAM stays free).
    #[derive(Clone, Copy)]
    enum RelRegion {
        Data(u16),
        XData(u16),
    }

    let mut fn_data_sizes = vec![0u16; program.functions.len()];
    let mut fn_xdata_sizes = vec![0u16; program.functions.len()];
    let mut fn_slot_rel: Vec<HashMap<SlotId, RelRegion>> = vec![HashMap::new(); program.functions.len()];
    let mut fn_val_rel: Vec<HashMap<ValueId, RelRegion>> = vec![HashMap::new(); program.functions.len()];

    for f in &program.functions {
        if f.is_test {
            continue;
        }
        let fid = f.id as usize;
        let escaped = f.escaped_locals();
        let mut d_off = 0u16;
        let mut x_off = 0u16;
        for (si, sl) in f.locals.iter().enumerate() {
            let sz = program.type_size(&sl.ty).max(1);
            let esc = escaped.get(si).copied().unwrap_or(false);
            if !esc && sz <= 4 && !matches!(sl.ty, IrType::Machine(_) | IrType::Pool { .. } | IrType::Queue { .. }) {
                fn_slot_rel[fid].insert(sl.id, RelRegion::Data(d_off));
                d_off += sz;
            } else {
                fn_slot_rel[fid].insert(sl.id, RelRegion::XData(x_off));
                x_off += sz;
            }
        }
        let vsizes = val_byte_sizes(program, f);
        for (vi, &vsz) in vsizes.iter().enumerate() {
            if vsz == 0 {
                continue;
            }
            if vsz <= 4 {
                fn_val_rel[fid].insert(vi as ValueId, RelRegion::Data(d_off));
                d_off += vsz;
            } else {
                fn_val_rel[fid].insert(vi as ValueId, RelRegion::XData(x_off));
                x_off += vsz;
            }
        }
        fn_data_sizes[fid] = d_off;
        fn_xdata_sizes[fid] = x_off;
    }

    // Color Data frames first; if total Data exceeds budget (leaving 24 bytes for stack), overflow whole Data tree frames to XData.
    let min_stack_reserve = 24u16;
    let first_free_data = data_used[data_range.0 as usize..=data_range.1 as usize]
        .iter()
        .rposition(|&u| u)
        .map_or(data_range.0 as usize, |p| data_range.0 as usize + p + 1);
    let mut data_frame_cursor = first_free_data as u16;
    let mut xdata_frame_cursor = xdata_used_intervals
        .iter()
        .map(|&(_, e)| e)
        .max()
        .unwrap_or(xdata_range.0);

    let mut slot_locs: HashMap<(u32, SlotId), SlotRegion> = HashMap::new();
    let mut val_locs: HashMap<(u32, ValueId), SlotRegion> = HashMap::new();

    for (_, reach_set, topo) in &trees {
        let tree = IrqTree {
            nodes: topo.clone(),
            callees: &fn_callees,
        };
        let d_offsets = tree.compute_call_graph_frames(reach_set, &fn_data_sizes, program);
        let d_total = d_offsets
            .iter()
            .map(|(&fid, &s)| s + fn_data_sizes[fid])
            .max()
            .unwrap_or(0);
        let use_xdata_for_data =
            data_frame_cursor + d_total + min_stack_reserve > idata_range.1 + 1
                || data_frame_cursor + d_total > data_range.1 + 1;

        let x_offsets = tree.compute_call_graph_frames(reach_set, &fn_xdata_sizes, program);
        let x_total = x_offsets
            .iter()
            .map(|(&fid, &s)| s + fn_xdata_sizes[fid])
            .max()
            .unwrap_or(0);

        let d_base = if use_xdata_for_data {
            let b = xdata_frame_cursor;
            xdata_frame_cursor += d_total;
            b
        } else {
            let b = data_frame_cursor;
            for i in b as usize..(b + d_total) as usize {
                data_used[i] = true;
            }
            data_frame_cursor += d_total;
            b
        };
        let x_base = xdata_frame_cursor;
        xdata_frame_cursor += x_total;

        if xdata_frame_cursor > xdata_range.1.saturating_add(1) {
            return Err(format!(
                "MCS-51 XData RAM overflow: {} bytes exceeds 0x{:04X}",
                xdata_frame_cursor, xdata_range.1
            ));
        }

        for &fid in topo {
            let f_id = fid as u32;
            let fdb = d_base + d_offsets.get(&fid).copied().unwrap_or(0);
            let fxb = x_base + x_offsets.get(&fid).copied().unwrap_or(0);
            let materialize = |rel: RelRegion| match rel {
                RelRegion::Data(r) if !use_xdata_for_data => SlotRegion::Data((fdb + r) as u8),
                RelRegion::Data(r) => SlotRegion::XData(fdb + r),
                RelRegion::XData(r) => SlotRegion::XData(fxb + r),
            };
            for (&sid, &rel) in &fn_slot_rel[fid] {
                slot_locs.insert((f_id, sid), materialize(rel));
            }
            for (&vid, &rel) in &fn_val_rel[fid] {
                val_locs.insert((f_id, vid), materialize(rel));
            }
        }
    }

    let global_spaces: Vec<PtrSpace> = global_allocs.iter().map(|g| g.space).collect();
    let (val_ptr_spaces, slot_ptr_spaces) = infer_ptr_spaces(program, &global_spaces, &slot_locs);

    // Allocate 1B runtime PtrSpace tags for any `Mixed` pointers.
    let mut slot_ptr_tags = HashMap::new();
    let mut val_ptr_tags = HashMap::new();
    for (&k, &sp) in &slot_ptr_spaces {
        if sp == PtrSpace::Mixed {
            let loc = if let Some(a) = alloc_data(&mut data_used, data_range, 1, 1) {
                if data_frame_cursor <= a as u16 {
                    data_frame_cursor = a as u16 + 1;
                }
                SlotRegion::Data(a)
            } else {
                let a = xdata_frame_cursor;
                xdata_frame_cursor += 1;
                SlotRegion::XData(a)
            };
            slot_ptr_tags.insert(k, loc);
        }
    }
    for (&k, &sp) in &val_ptr_spaces {
        if sp == PtrSpace::Mixed {
            let loc = if let Some(a) = alloc_data(&mut data_used, data_range, 1, 1) {
                if data_frame_cursor <= a as u16 {
                    data_frame_cursor = a as u16 + 1;
                }
                SlotRegion::Data(a)
            } else {
                let a = xdata_frame_cursor;
                xdata_frame_cursor += 1;
                SlotRegion::XData(a)
            };
            val_ptr_tags.insert(k, loc);
        }
    }

    let mut highest_iram = data_used
        .iter()
        .rposition(|&u| u && false)
        .unwrap_or(0);
    for (idx, &u) in data_used[..=data_range.1 as usize].iter().enumerate() {
        if u {
            highest_iram = idx + 1;
        }
    }
    let highest_iram = (highest_iram as u16).max(data_frame_cursor) as u8;
    // SP points to highest used byte (since 8051 PUSH pre-increments SP); minimum 0x07.
    let sp_init = highest_iram.saturating_sub(1).max(0x07);

    // Compute worst-case upward call stack depth (2 bytes per call + 5..13 bytes per ISR).
    fn max_call_depth(u: usize, callees: &[Vec<usize>], vis: &mut HashSet<usize>) -> u16 {
        if !vis.insert(u) {
            return 2;
        }
        let sub = callees[u]
            .iter()
            .map(|&v| 2 + max_call_depth(v, callees, vis))
            .max()
            .unwrap_or(0);
        vis.remove(&u);
        sub
    }
    let main_stack = main_roots
        .iter()
        .map(|&r| 2 + max_call_depth(r, &fn_callees, &mut HashSet::new()))
        .max()
        .unwrap_or(2);
    // Single-level ISR worst case (push PSW,ACC,B,DPL,DPH (+8 if no `using`) + PC + depth)
    let isr_stack: u16 = isr_roots
        .iter()
        .map(|&ir| {
            let f = &program.functions[ir];
            let reg_save = if f.isr_using.is_some() { 7 } else { 15 };
            reg_save + max_call_depth(ir, &fn_callees, &mut HashSet::new())
        })
        .max()
        .unwrap_or(0);
    let max_stack_bytes = main_stack + isr_stack;
    let stack_avail = idata_range.1.saturating_sub(sp_init as u16);
    if max_stack_bytes > stack_avail {
        return Err(format!(
            "MCS-51 IData stack overflow: need {max_stack_bytes} B above SP=0x{sp_init:02X}, only {stack_avail} B free"
        ));
    }

    Ok(Mcs51Layout {
        data_start: data_range.0 as u8,
        data_end: highest_iram,
        sp_init,
        max_stack_bytes,
        xdata_start: xdata_range.0,
        xdata_end: xdata_frame_cursor,
        code_start: code_range.0,
        code_end: code_range.1,
        globals: global_allocs,
        data_inits,
        xdata_inits,
        slot_locs,
        slot_ptr_tags,
        val_locs,
        val_ptr_tags,
        val_ptr_spaces,
        slot_ptr_spaces,
        fn_callees,
    })
}
