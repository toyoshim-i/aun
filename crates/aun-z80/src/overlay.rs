//! Tree-based *Uchiguri* partitioning for nested HSM code, ROM constants, and external `.s` overlays.

use aun_ir::*;
use std::collections::{HashMap, HashSet};

/// External SDASZ80 assembly input with optional state ownership (`--asm State=file.s` / `[overlay_asm]`)
/// and optional startup (`--startup-asm` / `startup_asm` / `crt0` / `.area _HEADER`) placement.
#[derive(Debug, Clone)]
pub struct ExternalAsm {
    pub name: String,
    pub source: String,
    pub owner_state: Option<String>,
    pub is_startup: bool,
}

impl ExternalAsm {
    pub fn new(name: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            source: source.into(),
            owner_state: None,
            is_startup: false,
        }
    }

    pub fn startup(name: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            source: source.into(),
            owner_state: None,
            is_startup: true,
        }
    }
}

/// Returns `true` when `src` contains an SDASZ80 `.area _HEADER` directive (e.g. `crt0.s`).
pub fn contains_header_area(src: &str) -> bool {
    for raw in src.lines() {
        let code = raw.split(';').next().unwrap_or("").trim();
        if code.is_empty() {
            continue;
        }
        let mut parts = code.splitn(2, |c: char| c.is_whitespace());
        let head = parts.next().unwrap_or("");
        let tail = parts.next().unwrap_or("").trim();
        if head.eq_ignore_ascii_case(".area") {
            let area_name = tail
                .split(|c: char| c.is_whitespace() || c == '(')
                .next()
                .unwrap_or("");
            if area_name.eq_ignore_ascii_case("_HEADER") {
                return true;
            }
        }
    }
    false
}

/// Effective placement owner of a function, ROM constant, or external `.s` unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OverlayOwner {
    Resident,
    /// `(machine_id, boundary_state_id)` where `boundary_state_id` has `swap_boundary` or `overlay_slot`.
    Boundary(u32, u8),
}

/// One hardware transfer/decompression entry emitted into ROM `LoadOp` tables and `manifest.json`.
#[derive(Debug, Clone)]
pub struct LoadOpSpec {
    pub blob_id: u8,
    pub machine_id: u32,
    pub state_id: u8,
    pub state_name: String,
    pub space_name: String,
    pub space_id: u8,
    pub origin_hint: Option<(u16, u16)>,
    pub slot_name: Option<String>,
    pub load_fn: Option<String>,
    pub fn_ids: Vec<usize>,
    pub rom_global_ids: Vec<usize>,
    pub ext_asm_indices: Vec<usize>,
    pub parent_state_id: Option<u8>,
}

/// Complete partitioning result consumed by `uchiguri`, `codegen`, and the top-down linker.
#[derive(Debug, Clone)]
pub struct OverlayPlan {
    pub machine_has_boundaries: Vec<bool>,
    pub fn_owner: Vec<OverlayOwner>,
    pub rom_global_owner: Vec<OverlayOwner>,
    pub ext_asm_owner: Vec<OverlayOwner>,
    pub ext_fn_owner: Vec<OverlayOwner>,
    pub blobs: Vec<LoadOpSpec>,
    /// `(machine_id, state_id) -> index in `blobs` (0-based)`
    pub state_blob_idx: HashMap<(u32, u8), usize>,
}

impl OverlayPlan {
    pub fn has_any_boundaries(&self) -> bool {
        self.machine_has_boundaries.iter().any(|&b| b)
    }

    pub fn is_fn_in_overlay(&self, fn_idx: usize) -> bool {
        self.fn_owner
            .get(fn_idx)
            .is_some_and(|o| *o != OverlayOwner::Resident)
    }

    /// Top-down list of blob indices needed to enter leaf `(mid, to_state)` starting strictly below `lca`.
    pub fn blobs_for_enter_chain(
        &self,
        program: &IrProgram,
        mid: u32,
        lca: Option<u8>,
        to_state: u8,
    ) -> Vec<usize> {
        let mdef = &program.machines[mid as usize];
        let mut out = Vec::new();
        for anc in mdef.ancestors(to_state) {
            let below_lca = lca.is_none_or(|l| anc != l && mdef.is_ancestor(l, anc));
            if below_lca && let Some(&bi) = self.state_blob_idx.get(&(mid, anc)) {
                out.push(bi);
            }
        }
        out
    }
}

/// Resolves `"App::InGame"`, `"mod::App::InGame::Stage1"`, or an unambiguous `"InGame"` state name.
pub fn resolve_state_name(program: &IrProgram, raw: &str) -> Result<(u32, u8), String> {
    let segs: Vec<&str> = raw.split("::").collect();
    for split in (1..segs.len()).rev() {
        let m_cand = segs[..split].join("::");
        if let Some(mdef) = program.machines.iter().find(|m| m.name == m_cand) {
            let s_cand = segs[split..].join("::");
            if let Some(st) = mdef.states.iter().find(|s| s.name == s_cand) {
                return Ok((mdef.id, st.id));
            }
        }
    }
    let mut matches = Vec::new();
    for mdef in &program.machines {
        if let Some(st) = mdef.states.iter().find(|s| s.name == raw) {
            matches.push((mdef.id, st.id));
        }
    }
    if matches.len() == 1 {
        return Ok(matches[0]);
    }
    if matches.len() > 1 {
        return Err(format!(
            "ambiguous overlay state `{raw}` (qualify as `Machine::{raw}`)"
        ));
    }
    Err(format!("unknown state `{raw}` for external assembly assignment"))
}

fn to_boundary_owner(program: &IrProgram, raw: Option<(u32, u8)>) -> OverlayOwner {
    let Some((mid, sid)) = raw else {
        return OverlayOwner::Resident;
    };
    let mdef = &program.machines[mid as usize];
    match mdef.enclosing_boundary(sid) {
        Some(bid) => OverlayOwner::Boundary(mid, bid),
        None => OverlayOwner::Resident,
    }
}

fn owner_lca(program: &IrProgram, a: OverlayOwner, b: OverlayOwner) -> OverlayOwner {
    match (a, b) {
        (OverlayOwner::Resident, _) | (_, OverlayOwner::Resident) => OverlayOwner::Resident,
        (OverlayOwner::Boundary(ma, sa), OverlayOwner::Boundary(mb, sb)) => {
            if ma != mb {
                return OverlayOwner::Resident;
            }
            let mdef = &program.machines[ma as usize];
            match mdef.lca(sa, sb) {
                Some(lca) => match mdef.enclosing_boundary(lca) {
                    Some(bid) => OverlayOwner::Boundary(ma, bid),
                    None => OverlayOwner::Resident,
                },
                None => OverlayOwner::Resident,
            }
        }
    }
}

/// Returns `true` if `callee_owner` is statically guaranteed live while executing in `caller_owner`.
pub fn owner_visible_from(
    program: &IrProgram,
    callee_owner: OverlayOwner,
    caller_owner: OverlayOwner,
) -> bool {
    match (callee_owner, caller_owner) {
        (OverlayOwner::Resident, _) => true,
        (OverlayOwner::Boundary(_, _), OverlayOwner::Resident) => false,
        (OverlayOwner::Boundary(mc, sc), OverlayOwner::Boundary(mu, su)) => {
            mc == mu && program.machines[mc as usize].is_ancestor(sc, su)
        }
    }
}

/// Extracts `.globl` and label symbol names declared inside a Z80 assembly source snippet.
fn extract_asm_defined_symbols(src: &str) -> HashSet<String> {
    let mut syms = HashSet::new();
    for raw in src.lines() {
        let code = raw.split(';').next().unwrap_or("").trim();
        if code.is_empty() {
            continue;
        }
        if let Some(rest) = code
            .strip_prefix(".globl")
            .or_else(|| code.strip_prefix(".GLOBL"))
        {
            for tok in rest.split([',', ' ', '\t']).filter(|s| !s.is_empty()) {
                syms.insert(tok.to_string());
            }
            continue;
        }
        if let Some((lhs, _)) = code.split_once(':') {
            let lbl = lhs.trim();
            if !lbl.is_empty() && !lbl.chars().any(char::is_whitespace) {
                syms.insert(lbl.to_string());
            }
        }
    }
    syms
}

/// Partitions all functions, ROM constants, and external `.s` modules across the HSM boundary tree
/// via call-graph LCA propagation, and enforces static cross-overlay call mutual exclusion.
pub fn partition_program(
    program: &IrProgram,
    ext_asms: &[ExternalAsm],
) -> Result<OverlayPlan, String> {
    let n = program.functions.len();
    let mut machine_has_boundaries = vec![false; program.machines.len()];
    for (mid, mdef) in program.machines.iter().enumerate() {
        machine_has_boundaries[mid] = mdef
            .states
            .iter()
            .any(|s| s.swap_boundary || s.overlay_slot.is_some());
    }

    // 1. Resolve external assembly file ownership and map symbol -> external assembly index.
    let mut ext_asm_owner = Vec::with_capacity(ext_asms.len());
    let mut sym_to_ext_asm: HashMap<String, usize> = HashMap::new();
    for (ei, ext) in ext_asms.iter().enumerate() {
        let owner = match &ext.owner_state {
            Some(s) => {
                let raw = resolve_state_name(program, s)?;
                to_boundary_owner(program, Some(raw))
            }
            None => OverlayOwner::Resident,
        };
        ext_asm_owner.push(owner);
        for sym in extract_asm_defined_symbols(&ext.source) {
            sym_to_ext_asm.insert(sym, ei);
        }
    }

    // 2. Resolve each `IrExternFn` owner from `ef.owner_state` and/or its `.s` file membership.
    let mut ext_fn_owner = vec![OverlayOwner::Resident; program.externs.len()];
    for (eid, ef) in program.externs.iter().enumerate() {
        let explicit = to_boundary_owner(program, ef.owner_state);
        let from_file = sym_to_ext_asm
            .get(&ef.name)
            .map(|&idx| ext_asm_owner[idx])
            .unwrap_or(OverlayOwner::Resident);
        match (explicit, from_file) {
            (OverlayOwner::Resident, other) | (other, OverlayOwner::Resident) => {
                ext_fn_owner[eid] = other;
            }
            (a, b) if a == b => {
                ext_fn_owner[eid] = a;
            }
            (a, b) => {
                return Err(format!(
                    "conflicting overlay ownership for extern `{}`: attribute says {:?} while assembly file belongs to {:?}",
                    ef.name, a, b
                ));
            }
        }
    }

    // 3. Collect explicit function owners (`on <State>` or `#[overlay(State)] fn`) and roots.
    let mut swap_and_load_fns: HashSet<&str> = HashSet::new();
    if let Some(target) = &program.target {
        if let Some(ref swap_fn) = target.on_swap_fn {
            swap_and_load_fns.insert(swap_fn.as_str());
        }
        for slot in &target.overlay_slots {
            if let Some(ref lfn) = slot.load_fn {
                swap_and_load_fns.insert(lfn.as_str());
            }
        }
    }
    let mut addr_taken_fns: HashSet<usize> = HashSet::new();
    for f in &program.functions {
        for bb in &f.blocks {
            for inst in &bb.insts {
                if let IrInst::AddrFn {
                    target: IrFnTarget::Fn(fid),
                    ..
                } = inst
                {
                    addr_taken_fns.insert(*fid as usize);
                }
            }
        }
    }

    let mut pinned = vec![false; n];
    let mut fn_owner_opt: Vec<Option<OverlayOwner>> = vec![None; n];
    for (i, f) in program.functions.iter().enumerate() {
        if f.is_test || f.blocks.is_empty() {
            fn_owner_opt[i] = Some(OverlayOwner::Resident);
            pinned[i] = true;
            continue;
        }
        let is_root = f.name == "main"
            || f.is_pub
            || f.export_name.is_some()
            || f.interrupt_source.is_some()
            || program.isr_entries.contains_key(&f.id)
            || swap_and_load_fns.contains(f.name.as_str())
            || addr_taken_fns.contains(&i);
        if is_root {
            fn_owner_opt[i] = Some(OverlayOwner::Resident);
            pinned[i] = true;
            continue;
        }
        if let Some((mid, sid)) = f.owner_state {
            let b = to_boundary_owner(program, Some((mid, sid)));
            fn_owner_opt[i] = Some(b);
            if b != OverlayOwner::Resident {
                pinned[i] = true;
            }
        } else if let Some((ref slot_name, ref state_name)) = f.overlay_state {
            // Legacy overlay_state fallback
            let mut found = OverlayOwner::Resident;
            for (mid, mdef) in program.machines.iter().enumerate() {
                if let Some(st) = mdef
                    .states
                    .iter()
                    .find(|s| s.name == *state_name && s.overlay_slot.as_deref() == Some(slot_name))
                {
                    found = OverlayOwner::Boundary(mid as u32, st.id);
                    break;
                }
            }
            fn_owner_opt[i] = Some(found);
            if found != OverlayOwner::Resident {
                pinned[i] = true;
            }
        }
    }

    // 4. Propagate owners down direct Call / inline StateTransition / inline StateDispatch edges via LCA.
    for _ in 0..=n.saturating_mul(2).max(8) {
        let mut changed = false;
        for (u, f) in program.functions.iter().enumerate() {
            if f.is_test || f.blocks.is_empty() {
                continue;
            }
            let Some(u_owner) = fn_owner_opt[u] else {
                continue;
            };
            let mut propagate_to = |v: usize| {
                if v >= n || pinned[v] {
                    return;
                }
                let next = match fn_owner_opt[v] {
                    None => u_owner,
                    Some(prev) => owner_lca(program, prev, u_owner),
                };
                if fn_owner_opt[v] != Some(next) {
                    fn_owner_opt[v] = Some(next);
                    changed = true;
                }
            };
            for bb in &f.blocks {
                for inst in &bb.insts {
                    match inst {
                        IrInst::Call { func: v, .. } => propagate_to(*v as usize),
                        IrInst::StateTransition {
                            machine_id,
                            from_state,
                            to_state,
                            ..
                        } if !machine_has_boundaries[*machine_id as usize] => {
                            let mdef = &program.machines[*machine_id as usize];
                            if let Some(enter_fn) = mdef.states[*to_state as usize].on_enter {
                                propagate_to(enter_fn as usize);
                            }
                            if let Some(from) = from_state
                                && let Some(exit_fn) = mdef.states[*from as usize].on_exit
                            {
                                propagate_to(exit_fn as usize);
                            }
                        }
                        IrInst::StateDispatch {
                            machine_id,
                            method,
                            ..
                        } if !machine_has_boundaries[*machine_id as usize] => {
                            let mdef = &program.machines[*machine_id as usize];
                            for st in &mdef.states {
                                if let Some(&fid) = st.methods.get(method) {
                                    propagate_to(fid as usize);
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

    let fn_owner: Vec<OverlayOwner> = fn_owner_opt
        .into_iter()
        .map(|o| o.unwrap_or(OverlayOwner::Resident))
        .collect();

    // 5. Verify static mutual exclusion on all direct Call and ExternCall edges.
    for (u, f) in program.functions.iter().enumerate() {
        if f.is_test || f.blocks.is_empty() {
            continue;
        }
        let u_owner = fn_owner[u];
        for bb in &f.blocks {
            for inst in &bb.insts {
                match inst {
                    IrInst::Call { func: v, .. } => {
                        let vi = *v as usize;
                        if vi < n {
                            let v_owner = fn_owner[vi];
                            if !owner_visible_from(program, v_owner, u_owner) {
                                return Err(format!(
                                    "illegal cross-overlay call from `{}` ({}) to `{}` ({})",
                                    f.name,
                                    fmt_owner(program, u_owner),
                                    program.functions[vi].name,
                                    fmt_owner(program, v_owner),
                                ));
                            }
                        }
                    }
                    IrInst::ExternCall { extern_fn, .. } => {
                        let ei = *extern_fn as usize;
                        if ei < ext_fn_owner.len() {
                            let e_owner = ext_fn_owner[ei];
                            if !owner_visible_from(program, e_owner, u_owner) {
                                return Err(format!(
                                    "illegal cross-overlay extern call from `{}` ({}) to `{}` ({})",
                                    f.name,
                                    fmt_owner(program, u_owner),
                                    program.externs[ei].name,
                                    fmt_owner(program, e_owner),
                                ));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // 6. Partition ROM constants / strings by LCA across referencing functions.
    let mut rom_global_owner_opt: Vec<Option<OverlayOwner>> = vec![None; program.globals.len()];
    for (u, f) in program.functions.iter().enumerate() {
        if f.is_test || f.blocks.is_empty() {
            continue;
        }
        let u_owner = fn_owner[u];
        for bb in &f.blocks {
            for inst in &bb.insts {
                if let IrInst::AddrGlobal { global, .. } = inst {
                    let gid = *global as usize;
                    if gid < program.globals.len()
                        && program.globals[gid].in_rom
                        && !program.globals[gid].is_extern
                    {
                        let next = match rom_global_owner_opt[gid] {
                            None => u_owner,
                            Some(prev) => owner_lca(program, prev, u_owner),
                        };
                        rom_global_owner_opt[gid] = Some(next);
                    }
                }
            }
        }
    }
    let rom_global_owner: Vec<OverlayOwner> = rom_global_owner_opt
        .into_iter()
        .map(|o| o.unwrap_or(OverlayOwner::Resident))
        .collect();

    // 7. Build deterministic DFS-ordered overlay blobs for every boundary state that either uses
    //    legacy `#[overlay(Slot)]` or owns at least one function / ROM const / external `.s`.
    let primary_space_name = program
        .target
        .as_ref()
        .and_then(|t| t.spaces.first().map(|s| s.name.clone()))
        .unwrap_or_else(|| "MainRam".to_string());

    let mut blobs = Vec::new();
    let mut state_blob_idx = HashMap::new();
    for (mid_usize, mdef) in program.machines.iter().enumerate() {
        let mid = mid_usize as u32;
        for st in &mdef.states {
            if !(st.swap_boundary || st.overlay_slot.is_some()) {
                continue;
            }
            let target_owner = OverlayOwner::Boundary(mid, st.id);
            let fn_ids: Vec<usize> = (0..n)
                .filter(|&i| {
                    !program.functions[i].is_test
                        && !program.functions[i].blocks.is_empty()
                        && fn_owner[i] == target_owner
                })
                .collect();
            let rom_global_ids: Vec<usize> = (0..program.globals.len())
                .filter(|&gid| {
                    program.globals[gid].in_rom
                        && !program.globals[gid].is_extern
                        && rom_global_owner[gid] == target_owner
                })
                .collect();
            let ext_asm_indices: Vec<usize> = (0..ext_asm_owner.len())
                .filter(|&ei| ext_asm_owner[ei] == target_owner)
                .collect();

            if st.overlay_slot.is_none()
                && fn_ids.is_empty()
                && rom_global_ids.is_empty()
                && ext_asm_indices.is_empty()
            {
                continue;
            }

            let slot_def = st.overlay_slot.as_ref().and_then(|slot_name| {
                program
                    .target
                    .as_ref()
                    .and_then(|t| t.overlay_slots.iter().find(|s| &s.name == slot_name))
            });
            let space_name = st
                .code_space
                .clone()
                .or_else(|| slot_def.and_then(|s| s.space.clone()))
                .unwrap_or_else(|| primary_space_name.clone());
            let space_id = program
                .target
                .as_ref()
                .and_then(|t| t.space_id(&space_name))
                .unwrap_or(0);
            let origin_hint = slot_def.map(|s| (s.start, s.end));
            let load_fn = slot_def.and_then(|s| s.load_fn.clone());
            let blob_id = blobs.len() as u8;
            state_blob_idx.insert((mid, st.id), blobs.len());
            let parent_state_id = st.parent.and_then(|p| mdef.enclosing_boundary(p));
            blobs.push(LoadOpSpec {
                blob_id,
                machine_id: mid,
                state_id: st.id,
                state_name: st.name.clone(),
                space_name,
                space_id,
                origin_hint,
                slot_name: st.overlay_slot.clone(),
                load_fn,
                fn_ids,
                rom_global_ids,
                ext_asm_indices,
                parent_state_id,
            });
        }
    }

    Ok(OverlayPlan {
        machine_has_boundaries,
        fn_owner,
        rom_global_owner,
        ext_asm_owner,
        ext_fn_owner,
        blobs,
        state_blob_idx,
    })
}

fn fmt_owner(program: &IrProgram, owner: OverlayOwner) -> String {
    match owner {
        OverlayOwner::Resident => "Resident".to_string(),
        OverlayOwner::Boundary(mid, sid) => {
            let mdef = &program.machines[mid as usize];
            format!("{}::{}", mdef.name, mdef.states[sid as usize].name)
        }
    }
}
