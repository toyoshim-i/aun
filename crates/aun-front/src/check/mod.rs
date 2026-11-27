//! Type checker, Second-Class Reference borrow checker, and AST-to-IR lowering.

use crate::ast::*;
use crate::diag::Diagnostic;
use aun_ir::*;
use std::collections::HashMap;

mod expr;
mod items;
mod stmt;
#[cfg(test)]
mod tests;

#[derive(Debug, Clone)]
pub(super) struct FnSig {
    pub(super) id: u32,
    pub(super) is_extern: bool,
    pub(super) params: Vec<IrType>,
    pub(super) ret_ty: IrType,
}

#[derive(Debug, Clone)]
struct VarInfo {
    pub(super) slot: SlotId,
    pub(super) ty: IrType,
    pub(super) mutable: bool,
}

/// Place path used for second-class alias/borrow checking in call arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
struct BorrowPlace {
    pub(super) root: String,
    pub(super) fields: Vec<String>,
    pub(super) mutable: bool,
}

impl BorrowPlace {
    /// Returns true if `self` and `other` overlap and at least one is mutable.
    pub(super) fn conflicts_with(&self, other: &BorrowPlace) -> bool {
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
    pub(super) program: IrProgram,
    pub(super) diagnostics: Vec<Diagnostic>,
    pub(super) struct_ids: HashMap<String, u32>,
    pub(super) machine_ids: HashMap<String, u32>,
    pub(super) enums: HashMap<String, (IntType, HashMap<String, i64>)>,
    pub(super) consts: HashMap<String, (IrType, i64)>,
    pub(super) global_ids: HashMap<String, (u32, IrType, bool)>,
    pub(super) state_globals: HashMap<(u32, u8, String), (u32, IrType, bool)>,
    pub(super) pending_state_statics: HashMap<u32, Vec<(u8, StaticDecl)>>,
    pub(super) str_globals: HashMap<Vec<u8>, u32>,
    pub(super) fn_sigs: HashMap<String, FnSig>,
    pub(super) top_fn_ids: Vec<u32>,
    pub(super) current_self_machine: Option<u32>,
    pub(super) queue_spsc_accesses: Vec<(usize, u32, bool, Span)>,
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
            state_globals: HashMap::new(),
            pending_state_statics: HashMap::new(),
            str_globals: HashMap::new(),
            fn_sigs: HashMap::new(),
            top_fn_ids: Vec::new(),
            current_self_machine: None,
            queue_spsc_accesses: Vec::new(),
        };
        checker.collect_items(file);
        checker.lower_functions(file);
        checker.check_queue_spsc_safety();
        if checker.diagnostics.is_empty() {
            Ok(checker.program)
        } else {
            Err(checker.diagnostics)
        }
    }

    pub(super) fn error(&mut self, msg: impl Into<String>, span: Span) {
        self.diagnostics.push(Diagnostic::error(msg, span));
    }

}

#[derive(Debug, Clone)]
enum ScopeUnwindKind {
    Bank {
        group: String,
        prev_banks: Vec<String>,
        restore_ei: bool,
    },
    Critical,
}

#[derive(Debug, Clone)]
struct ScopeUnwindEntry {
    pub(super) kind: ScopeUnwindKind,
    pub(super) loop_depth: usize,
}

struct FnLowerer<'a> {
    pub(super) checker: &'a mut Checker,
    pub(super) fn_idx: usize,
    pub(super) cur_bb: BlockId,
    pub(super) scopes: Vec<HashMap<String, VarInfo>>,
    pub(super) loop_stack: Vec<(BlockId, BlockId)>, // (continue_bb, break_bb)
    /// `(machine_id, narrowed_state_id, is_lifecycle_hook)` when inside a `machine` method.
    pub(super) machine_ctx: Option<(u32, Option<u8>, bool)>,
    /// Current active banks per `bank_group` name.
    pub(super) active_banks: HashMap<String, Vec<String>>,
    /// Active `with bank(...)` and `critical { ... }` scopes for LIFO unwinding on early return/break/continue/goto.
    pub(super) scope_unwind_stack: Vec<ScopeUnwindEntry>,
    /// Whether interrupts are statically known to be disabled (`di`).
    pub(super) irq_disabled: bool,
    /// Interrupt source name when lowering an `#[interrupt(src)]` handler.
    pub(super) current_isr_source: Option<String>,
}

impl<'a> FnLowerer<'a> {
    pub(super) fn new(
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
        let func = &checker.program.functions[fn_idx];
        let irq_disabled = checker.program.is_isr_root(func) || func.requires_di;
        let current_isr_source = func.interrupt_source.clone();
        Self {
            checker,
            fn_idx,
            cur_bb: 0,
            scopes: vec![root_scope],
            loop_stack: Vec::new(),
            machine_ctx,
            active_banks,
            scope_unwind_stack: Vec::new(),
            irq_disabled,
            current_isr_source,
        }
    }

    /// Emits `BankExit` / `CriticalExit` in LIFO order for all active scopes with `loop_depth >= min_loop_depth`.
    pub(super) fn unwind_bank_scopes(&mut self, min_loop_depth: usize) {
        let exits: Vec<ScopeUnwindEntry> = self
            .scope_unwind_stack
            .iter()
            .rev()
            .take_while(|e| e.loop_depth >= min_loop_depth)
            .cloned()
            .collect();
        for entry in exits {
            match entry.kind {
                ScopeUnwindKind::Bank {
                    group,
                    prev_banks,
                    restore_ei,
                } => self.emit(IrInst::BankExit {
                    group,
                    restore_banks: prev_banks,
                    restore_ei,
                }),
                ScopeUnwindKind::Critical => self.emit(IrInst::CriticalExit),
            }
        }
    }

    pub(super) fn check_global_irq_ownership(&mut self, gid: u32, name: &str, span: Span) {
        let Some(owner) = self.checker.program.globals[gid as usize]
            .owned_by_irq
            .clone()
        else {
            return;
        };
        if self.current_isr_source.as_deref() == Some(owner.as_str()) {
            return;
        }
        if !self.irq_disabled {
            self.checker.error(
                format!(
                    "interrupt safety violation: accessing `{name}` (`#[owned_by({owner})]`) outside `#[interrupt({owner})]` requires `critical {{ ... }}` or `di`"
                ),
                span,
            );
        }
    }

    pub(super) fn check_bank_mapped(&mut self, bank_name: &str, context: &str, span: Span) {
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

    pub(super) fn func(&self) -> &IrFunction {
        &self.checker.program.functions[self.fn_idx]
    }

    pub(super) fn func_mut(&mut self) -> &mut IrFunction {
        &mut self.checker.program.functions[self.fn_idx]
    }

    pub(super) fn fresh_val(&mut self) -> ValueId {
        let f = self.func_mut();
        let v = f.next_val;
        f.next_val += 1;
        v
    }

    pub(super) fn new_block(&mut self) -> BlockId {
        let f = self.func_mut();
        let id = f.blocks.len() as BlockId;
        f.blocks.push(BasicBlock {
            id,
            insts: Vec::new(),
            term: Terminator::Unreachable,
        });
        id
    }

    pub(super) fn emit(&mut self, inst: IrInst) {
        let bb = self.cur_bb as usize;
        self.func_mut().blocks[bb].insts.push(inst);
    }

    pub(super) fn set_term(&mut self, term: Terminator) {
        let bb = self.cur_bb as usize;
        if matches!(self.func().blocks[bb].term, Terminator::Unreachable) {
            self.func_mut().blocks[bb].term = term;
        }
    }

    pub(super) fn is_terminated(&self) -> bool {
        !matches!(
            self.func().blocks[self.cur_bb as usize].term,
            Terminator::Unreachable
        )
    }

    pub(super) fn lookup_var(&self, name: &str) -> Option<VarInfo> {
        for scope in self.scopes.iter().rev() {
            if let Some(v) = scope.get(name) {
                return Some(v.clone());
            }
        }
        None
    }

    pub(super) fn alloc_local_slot(
        &mut self,
        name: String,
        ty: IrType,
        mutable: bool,
    ) -> SlotId {
        let f = self.func_mut();
        let id = f.locals.len() as SlotId;
        f.locals.push(LocalSlot {
            id,
            name,
            ty,
            mutable,
        });
        id
    }

    pub(super) fn bind_local(
        &mut self,
        name: String,
        slot: SlotId,
        ty: IrType,
        mutable: bool,
    ) {
        self.scopes.last_mut().unwrap().insert(
            name,
            VarInfo {
                slot,
                ty,
                mutable,
            },
        );
    }

    pub(super) fn alloc_local(&mut self, name: String, ty: IrType, mutable: bool) -> SlotId {
        let id = self.alloc_local_slot(name.clone(), ty.clone(), mutable);
        self.bind_local(name, id, ty, mutable);
        id
    }

}
