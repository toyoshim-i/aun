//! Target-independent intermediate representation (`aun-ir`) and middle-end passes.

pub mod opt;

use std::collections::HashMap;
use std::fmt;

pub type ValueId = u32;
pub type SlotId = u32;
pub type BlockId = u32;

/// Source span identifier mapped to `file:line:col` for diagnostics and `.sym` manifests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Span {
    pub file_id: u16,
    pub start: u32,
    pub end: u32,
}

impl Span {
    pub const fn new(file_id: u16, start: u32, end: u32) -> Self {
        Self { file_id, start, end }
    }
}

/// Dynamic check categories that can be toggled globally or per module/function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CheckKind {
    Bounds,
    Transition,
    Overflow,
    Bank,
}

/// Runtime panic codes emitted to the user-defined `#[panic_handler]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PanicCode {
    Bounds = 1,
    InvalidHandle = 2,
    InvalidTransition = 3,
    Overflow = 4,
    DivByZero = 5,
    BankMismatch = 6,
    AssertionFailed = 7,
    UserPanic = 8,
}

impl PanicCode {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Bounds" => Some(Self::Bounds),
            "InvalidHandle" => Some(Self::InvalidHandle),
            "InvalidTransition" => Some(Self::InvalidTransition),
            "Overflow" => Some(Self::Overflow),
            "DivByZero" => Some(Self::DivByZero),
            "BankMismatch" => Some(Self::BankMismatch),
            "AssertionFailed" => Some(Self::AssertionFailed),
            "UserPanic" => Some(Self::UserPanic),
            _ => None,
        }
    }
}

/// Arithmetic overflow behavior selected by operator (`+`, `+|`, `+%`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArithMode {
    Checked,
    Wrapping,
    Saturating,
}

/// Base integer width and signedness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
}

impl IntType {
    pub const fn byte_size(self) -> u16 {
        match self {
            IntType::U8 | IntType::I8 => 1,
            IntType::U16 | IntType::I16 => 2,
            IntType::U32 | IntType::I32 => 4,
        }
    }

    pub const fn bit_width(self) -> u8 {
        (self.byte_size() * 8) as u8
    }

    pub const fn is_signed(self) -> bool {
        matches!(self, IntType::I8 | IntType::I16 | IntType::I32)
    }

    pub const fn min_i64(self) -> i64 {
        match self {
            IntType::U8 | IntType::U16 | IntType::U32 => 0,
            IntType::I8 => i8::MIN as i64,
            IntType::I16 => i16::MIN as i64,
            IntType::I32 => i32::MIN as i64,
        }
    }

    pub const fn max_i64(self) -> i64 {
        match self {
            IntType::U8 => u8::MAX as i64,
            IntType::I8 => i8::MAX as i64,
            IntType::U16 => u16::MAX as i64,
            IntType::I16 => i16::MAX as i64,
            IntType::U32 => u32::MAX as i64,
            IntType::I32 => i32::MAX as i64,
        }
    }
}

impl fmt::Display for IntType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            IntType::U8 => "u8",
            IntType::I8 => "i8",
            IntType::U16 => "u16",
            IntType::I16 => "i16",
            IntType::U32 => "u32",
            IntType::I32 => "i32",
        };
        f.write_str(s)
    }
}

/// Core lowered IR type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IrType {
    Unit,
    Never,
    Bool,
    Int(IntType),
    Fixed { base: IntType, frac_bits: u8 },
    Struct(u32),
    Machine(u32),
    Array { elem: Box<IrType>, len: u16 },
    Pool { elem: Box<IrType>, capacity: u16 },
    Handle(Box<IrType>),
    Ref { mutable: bool, inner: Box<IrType> },
    ConstSlice(Box<IrType>),
}

impl IrType {
    pub fn byte_size(&self, structs: &[StructDef]) -> u16 {
        self.byte_size_with_machines(structs, &[])
    }

    pub fn byte_size_with_machines(&self, structs: &[StructDef], machines: &[MachineDef]) -> u16 {
        match self {
            IrType::Unit | IrType::Never => 0,
            IrType::Bool => 1,
            IrType::Int(i) => i.byte_size(),
            IrType::Fixed { base, .. } => base.byte_size(),
            IrType::Struct(id) => structs[*id as usize].size,
            IrType::Machine(id) => machines.get(*id as usize).map(|m| m.total_size).unwrap_or(2),
            IrType::Array { elem, len } => elem.byte_size_with_machines(structs, machines) * len,
            IrType::Pool { elem, capacity } => {
                // Element storage + 1-byte alive bitmap per 8 slots (rounded up)
                let elem_bytes = elem.byte_size_with_machines(structs, machines) * capacity;
                elem_bytes + capacity.div_ceil(8)
            }
            IrType::Handle(_) => 1, // Always 1-byte u8 (AD-12)
            IrType::Ref { .. } => 2,
            IrType::ConstSlice(_) => 4,
        }
    }

    pub fn is_ref(&self) -> bool {
        matches!(self, IrType::Ref { .. })
    }

    pub fn is_scalar(&self) -> bool {
        matches!(
            self,
            IrType::Bool
                | IrType::Int(_)
                | IrType::Fixed { .. }
                | IrType::Handle(_)
                | IrType::Ref { .. }
        )
    }
}

#[derive(Debug, Clone)]
pub struct StructDef {
    pub name: String,
    pub fields: Vec<(String, IrType, u16)>,
    pub size: u16,
}

#[derive(Debug, Clone)]
pub struct StateDef {
    pub id: u8,
    pub name: String,
    pub fields: Vec<(String, IrType, u16)>,
    pub size: u16,
    pub on_enter: Option<u32>,
    pub on_exit: Option<u32>,
    pub methods: HashMap<String, u32>,
    pub overlay_slot: Option<String>,
}

/// First-class state machine definition with overlaid state-local RAM (*Uchiguri*).
#[derive(Debug, Clone)]
pub struct MachineDef {
    pub id: u32,
    pub name: String,
    pub strict_transitions: bool,
    pub smc: bool,
    pub shared_fields: Vec<(String, IrType, u16)>,
    pub states: Vec<StateDef>,
    pub initial_state: u8,
    /// Bitmask of permitted target states per source state ID.
    pub transitions: Vec<u32>,
    /// Byte offset of the 2-byte state tag/dispatch pointer.
    pub tag_offset: u16,
    /// Byte offset where mutually exclusive state-local fields are overlaid.
    pub payload_offset: u16,
    /// Total byte size = `payload_offset + max(state.size)`.
    pub total_size: u16,
}

impl MachineDef {
    pub fn is_transition_allowed(&self, from: u8, to: u8) -> bool {
        self.transitions
            .get(from as usize)
            .is_some_and(|&mask| (mask & (1u32 << to)) != 0)
    }
}

/// Hardware memory space or stack region (`space` / `stack`).
#[derive(Debug, Clone)]
pub struct SpaceDef {
    pub name: String,
    pub start: u16,
    pub end: u16,
}

/// Single bank within a `bank_group`.
#[derive(Debug, Clone)]
pub struct BankDef {
    pub name: String,
    pub start: u16,
    pub end: u16,
    pub requires_di: bool,
}

impl BankDef {
    pub fn overlaps(&self, other: &BankDef) -> bool {
        self.start <= other.end && other.start <= self.end
    }
}

/// Port write action inside a bank switch or interrupt rearm block.
#[derive(Debug, Clone)]
pub struct PortWriteOp {
    pub port: u8,
    pub val_expr: String,
}

/// Multi-banked address window (`bank_group`).
#[derive(Debug, Clone)]
pub struct BankGroupDef {
    pub name: String,
    pub start: u16,
    pub end: u16,
    pub banks: Vec<BankDef>,
    pub default_banks: Vec<String>,
    /// `(target_bank_combo, raw port writes)`
    pub switches: Vec<(Vec<String>, Vec<PortWriteOp>)>,
}

/// Single interrupt source inside an `interrupt_controller`.
#[derive(Debug, Clone)]
pub struct IntSourceDef {
    pub name: String,
    pub vector_addr: u16,
    pub rearm: Vec<PortWriteOp>,
}

/// Hardware interrupt controller declaration.
#[derive(Debug, Clone)]
pub struct IntCtrlDef {
    pub name: String,
    pub disable_asm: String,
    pub enable_asm: String,
    pub sources: Vec<IntSourceDef>,
}

/// Code overlay slot loaded from disk (`overlay_slot`).
#[derive(Debug, Clone)]
pub struct OverlaySlotDef {
    pub name: String,
    pub start: u16,
    pub end: u16,
    pub load_fn: Option<String>,
}

/// Complete hardware target specification (`target <name> : <arch> { ... }`).
#[derive(Debug, Clone)]
pub struct TargetDef {
    pub name: String,
    pub arch: String,
    pub spaces: Vec<SpaceDef>,
    pub stacks: Vec<SpaceDef>,
    pub bank_groups: Vec<BankGroupDef>,
    pub interrupt_controllers: Vec<IntCtrlDef>,
    pub overlay_slots: Vec<OverlaySlotDef>,
}

/// Top-level compiled IR program.
#[derive(Debug, Clone, Default)]
pub struct IrProgram {
    pub target: Option<TargetDef>,
    pub structs: Vec<StructDef>,
    pub machines: Vec<MachineDef>,
    pub globals: Vec<IrGlobal>,
    pub externs: Vec<IrExternFn>,
    pub functions: Vec<IrFunction>,
    pub sites: Vec<Span>,
}

impl IrProgram {
    pub fn intern_site(&mut self, span: Span) -> u16 {
        let id = self.sites.len() as u16;
        self.sites.push(span);
        id
    }

    pub fn type_size(&self, ty: &IrType) -> u16 {
        ty.byte_size_with_machines(&self.structs, &self.machines)
    }
}

/// Static global variable or ROM constant aggregate.
#[derive(Debug, Clone)]
pub struct IrGlobal {
    pub name: String,
    pub ty: IrType,
    pub mutable: bool,
    pub in_rom: bool,
    pub align: u16,
    pub soa: bool,
    pub space: Option<String>,
    pub init: Vec<u8>,
}

/// Register-explicit FFI declaration (`extern "asm"`).
#[derive(Debug, Clone)]
pub struct IrExternFn {
    pub name: String,
    pub params: Vec<(String, IrType, String)>,
    pub ret: Option<(IrType, String)>,
    pub clobbers: Vec<String>,
    pub requires_di: bool,
    pub requires_banks: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct LocalSlot {
    pub id: SlotId,
    pub name: String,
    pub ty: IrType,
    pub mutable: bool,
}

/// IR function (CFG of basic blocks).
#[derive(Debug, Clone)]
pub struct IrFunction {
    pub id: u32,
    pub name: String,
    pub is_pub: bool,
    pub param_count: usize,
    pub locals: Vec<LocalSlot>,
    pub ret_ty: IrType,
    pub is_test: bool,
    pub expected_panic: Option<PanicCode>,
    pub is_reentrant: bool,
    pub interrupt_source: Option<String>,
    pub requires_banks: Vec<String>,
    pub requires_di: bool,
    pub overlay_state: Option<(String, String)>,
    pub blocks: Vec<BasicBlock>,
    pub next_val: ValueId,
}

#[derive(Debug, Clone, Default)]
pub struct BasicBlock {
    pub id: BlockId,
    pub insts: Vec<IrInst>,
    pub term: Terminator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrUnOp {
    Neg,
    Not,
}

#[derive(Debug, Clone)]
pub enum IrInst {
    ConstInt {
        dst: ValueId,
        ty: IrType,
        val: i64,
    },
    ConstBool {
        dst: ValueId,
        val: bool,
    },
    BinOp {
        dst: ValueId,
        op: IrBinOp,
        mode: ArithMode,
        ty: IrType,
        lhs: ValueId,
        rhs: ValueId,
        site_id: u16,
    },
    UnOp {
        dst: ValueId,
        op: IrUnOp,
        mode: ArithMode,
        ty: IrType,
        val: ValueId,
        site_id: u16,
    },
    Cast {
        dst: ValueId,
        val: ValueId,
        from_ty: IrType,
        to_ty: IrType,
    },
    FixedInt {
        dst: ValueId,
        val: ValueId,
        base: IntType,
        frac_bits: u8,
    },
    LoadLocal {
        dst: ValueId,
        slot: SlotId,
        ty: IrType,
    },
    StoreLocal {
        slot: SlotId,
        val: ValueId,
        ty: IrType,
    },
    AddrLocal {
        dst: ValueId,
        slot: SlotId,
    },
    AddrGlobal {
        dst: ValueId,
        global: u32,
    },
    FieldAddr {
        dst: ValueId,
        base: ValueId,
        byte_offset: u16,
    },
    IndexAddr {
        dst: ValueId,
        base: ValueId,
        index: ValueId,
        elem_size: u16,
        len: u16,
        site_id: u16,
    },
    LoadPtr {
        dst: ValueId,
        addr: ValueId,
        ty: IrType,
    },
    StorePtr {
        addr: ValueId,
        val: ValueId,
        ty: IrType,
    },
    /// First-class state machine transition.
    /// `from_state: Some(s)` = statically verified; `None` = dynamically checked in Debug.
    StateTransition {
        machine_addr: ValueId,
        machine_id: u32,
        from_state: Option<u8>,
        to_state: u8,
        /// `(rel_offset_from_payload, value_id, field_ty)` evaluated before `on_exit`.
        field_vals: Vec<(u16, ValueId, IrType)>,
        site_id: u16,
    },
    /// Dispatches a method on a state machine according to its current state tag.
    StateDispatch {
        dst: Option<ValueId>,
        machine_addr: ValueId,
        machine_id: u32,
        method: String,
        args: Vec<ValueId>,
    },
    /// Reads the current state ID tag (`u8`) of a state machine.
    StateTag {
        dst: ValueId,
        machine_addr: ValueId,
        machine_id: u32,
    },
    /// Allocates a slot in a `Pool<T, N>`, returning a 1-byte `Handle<T>` (`0xFF` if full).
    PoolAlloc {
        dst: ValueId,
        pool_addr: ValueId,
        init_addr: ValueId,
        elem_size: u16,
        capacity: u16,
    },
    /// Frees a slot in a `Pool<T, N>` (traps `InvalidHandle` in Debug if invalid/already free).
    PoolFree {
        pool_addr: ValueId,
        handle: ValueId,
        elem_size: u16,
        capacity: u16,
        site_id: u16,
    },
    /// Computes element address in a `Pool<T, N>` (traps `InvalidHandle` in Debug if dead/OOB).
    PoolElemAddr {
        dst: ValueId,
        pool_addr: ValueId,
        handle: ValueId,
        elem_size: u16,
        capacity: u16,
        site_id: u16,
    },
    Call {
        dst: Option<ValueId>,
        func: u32,
        args: Vec<ValueId>,
    },
    ExternCall {
        dst: Option<ValueId>,
        extern_fn: u32,
        args: Vec<ValueId>,
    },
    PortOut {
        port: ValueId,
        val: ValueId,
    },
    PortIn {
        dst: ValueId,
        port: ValueId,
    },
    Assert {
        cond: ValueId,
        site_id: u16,
    },
    InlineAsm {
        template: String,
        clobbers: Vec<String>,
    },
    /// Enters a `with bank(...)` scope, disabling interrupts if required and switching banks.
    BankEnter {
        group: String,
        banks: Vec<String>,
        needs_di: bool,
    },
    /// Exits a `with bank(...)` scope, restoring previous banks and re-enabling interrupts if needed.
    BankExit {
        group: String,
        restore_banks: Vec<String>,
        restore_ei: bool,
    },
    Panic {
        code: PanicCode,
        site_id: u16,
    },
}

#[derive(Debug, Clone, Default)]
pub enum Terminator {
    #[default]
    Unreachable,
    Return(Option<ValueId>),
    Goto(BlockId),
    Branch {
        cond: ValueId,
        then_bb: BlockId,
        else_bb: BlockId,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int_type_sizes() {
        assert_eq!(IntType::U8.byte_size(), 1);
        assert_eq!(IntType::I16.byte_size(), 2);
        assert_eq!(IntType::U32.bit_width(), 32);
        assert!(IntType::I8.is_signed());
        assert!(!IntType::U16.is_signed());
    }
}
