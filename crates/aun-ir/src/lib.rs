//! Target-independent intermediate representation (`aun-ir`) and middle-end passes.

pub mod cfg;
pub mod gvn;
pub mod inline;
pub mod licm;
pub mod opt;
pub mod osr;
pub mod sccp;
pub mod ssa;
pub mod verify;

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
    Slice { mutable: bool, elem: Box<IrType> },
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
            IrType::Slice { .. } | IrType::ConstSlice(_) => 4,
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
                | IrType::Slice { .. }
                | IrType::ConstSlice(_)
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

/// Target memory address space classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum IrAddrSpace {
    #[default]
    Flat,
    Code,
    Data,
    IData,
    XData,
    Sfr,
    Bit,
}

impl IrAddrSpace {
    pub fn parse_name(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "flat" => Some(Self::Flat),
            "code" | "rom" => Some(Self::Code),
            "data" => Some(Self::Data),
            "idata" => Some(Self::IData),
            "xdata" | "xram" => Some(Self::XData),
            "sfr" => Some(Self::Sfr),
            "bit" | "sbit" => Some(Self::Bit),
            _ => None,
        }
    }
}

/// Hardware memory space or stack region (`space` / `stack`).
#[derive(Debug, Clone)]
pub struct SpaceDef {
    pub name: String,
    pub kind: IrAddrSpace,
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
    pub mode: String,
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
    pub volatile: bool,
    pub fixed_addr: Option<u16>,
    pub is_extern: bool,
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
    pub clobbers: Option<Vec<String>>,
    pub requires_di: bool,
    pub requires_banks: Vec<String>,
    pub fixed_addr: Option<u16>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrFnTarget {
    Fn(u32),
    Extern(u32),
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
    pub isr_using: Option<u8>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
        volatile: bool,
    },
    StorePtr {
        addr: ValueId,
        val: ValueId,
        ty: IrType,
        volatile: bool,
    },
    MemSet {
        dst_addr: ValueId,
        val: ValueId,
        len: u16,
    },
    MemCopy {
        dst_addr: ValueId,
        src_addr: ValueId,
        len: u16,
    },
    /// Replicates element 0 (`elem_size` bytes already stored at `dst_addr`) across
    /// `count` consecutive elements (`1..count`). Keeps `MemCopy` strictly non-overlapping.
    MemFill {
        dst_addr: ValueId,
        elem_size: u16,
        count: u16,
    },
    MakeSlice {
        dst: ValueId,
        ptr: ValueId,
        len: ValueId,
        ty: IrType,
    },
    SlicePtr {
        dst: ValueId,
        slice: ValueId,
        ty: IrType,
    },
    SliceLen {
        dst: ValueId,
        slice: ValueId,
    },
    AddrFn {
        dst: ValueId,
        target: IrFnTarget,
    },
    CallIndirect {
        addr: ValueId,
        args: Vec<(ValueId, IrType)>,
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
    /// Static Single Assignment phi node selecting a value based on the predecessor block.
    ///
    /// Must only appear at the beginning of a basic block (before any non-`Phi` instruction).
    Phi {
        dst: ValueId,
        ty: IrType,
        incoming: Vec<(BlockId, ValueId)>,
    },
}

impl IrFunction {
    /// Returns a bit-vec marking local slots whose address escapes (i.e. `AddrLocal` value is
    /// used by anything other than direct `LoadPtr(addr)` or `StorePtr(addr, val != addr)`).
    pub fn escaped_locals(&self) -> Vec<bool> {
        let mut escaped = vec![false; self.locals.len()];
        for bb in &self.blocks {
            for inst in &bb.insts {
                if let IrInst::AddrLocal { dst, slot } = inst {
                    let s = *slot as usize;
                    if s < escaped.len() && !escaped[s] {
                        let escapes = self.blocks.iter().any(|b| {
                            b.insts.iter().any(|u| {
                                u.uses_val(*dst)
                                    && !matches!(u, IrInst::LoadPtr { addr, .. } if *addr == *dst)
                                    && !matches!(
                                        u,
                                        IrInst::StorePtr { addr, val, .. } if *addr == *dst && *val != *dst
                                    )
                            }) || matches!(
                                b.term,
                                Terminator::Return(Some(v)) | Terminator::Branch { cond: v, .. } if v == *dst
                            )
                        });
                        if escapes {
                            escaped[s] = true;
                        }
                    }
                }
            }
        }
        escaped
    }
}

impl IrInst {
    /// Returns the SSA `ValueId` defined by this instruction, if any.
    pub fn dst(&self) -> Option<ValueId> {
        match self {
            IrInst::ConstInt { dst, .. }
            | IrInst::ConstBool { dst, .. }
            | IrInst::BinOp { dst, .. }
            | IrInst::UnOp { dst, .. }
            | IrInst::Cast { dst, .. }
            | IrInst::FixedInt { dst, .. }
            | IrInst::LoadLocal { dst, .. }
            | IrInst::AddrLocal { dst, .. }
            | IrInst::AddrGlobal { dst, .. }
            | IrInst::AddrFn { dst, .. }
            | IrInst::FieldAddr { dst, .. }
            | IrInst::IndexAddr { dst, .. }
            | IrInst::LoadPtr { dst, .. }
            | IrInst::MakeSlice { dst, .. }
            | IrInst::SlicePtr { dst, .. }
            | IrInst::SliceLen { dst, .. }
            | IrInst::StateTag { dst, .. }
            | IrInst::PoolAlloc { dst, .. }
            | IrInst::PoolElemAddr { dst, .. }
            | IrInst::PortIn { dst, .. }
            | IrInst::Phi { dst, .. } => Some(*dst),
            IrInst::Call { dst, .. }
            | IrInst::ExternCall { dst, .. }
            | IrInst::StateDispatch { dst, .. } => *dst,
            _ => None,
        }
    }

    /// Returns a mutable reference to the SSA `ValueId` defined by this instruction, if any.
    pub fn dst_mut(&mut self) -> Option<&mut ValueId> {
        match self {
            IrInst::ConstInt { dst, .. }
            | IrInst::ConstBool { dst, .. }
            | IrInst::BinOp { dst, .. }
            | IrInst::UnOp { dst, .. }
            | IrInst::Cast { dst, .. }
            | IrInst::FixedInt { dst, .. }
            | IrInst::LoadLocal { dst, .. }
            | IrInst::AddrLocal { dst, .. }
            | IrInst::AddrGlobal { dst, .. }
            | IrInst::AddrFn { dst, .. }
            | IrInst::FieldAddr { dst, .. }
            | IrInst::IndexAddr { dst, .. }
            | IrInst::LoadPtr { dst, .. }
            | IrInst::MakeSlice { dst, .. }
            | IrInst::SlicePtr { dst, .. }
            | IrInst::SliceLen { dst, .. }
            | IrInst::StateTag { dst, .. }
            | IrInst::PoolAlloc { dst, .. }
            | IrInst::PoolElemAddr { dst, .. }
            | IrInst::PortIn { dst, .. }
            | IrInst::Phi { dst, .. } => Some(dst),
            IrInst::Call { dst, .. }
            | IrInst::ExternCall { dst, .. }
            | IrInst::StateDispatch { dst, .. } => dst.as_mut(),
            _ => None,
        }
    }

    /// Invokes `f` on every SSA `ValueId` operand read by this instruction.
    pub fn for_each_use(&self, mut f: impl FnMut(ValueId)) {
        match self {
            IrInst::BinOp { lhs, rhs, .. } => {
                f(*lhs);
                f(*rhs);
            }
            IrInst::UnOp { val, .. }
            | IrInst::Cast { val, .. }
            | IrInst::FixedInt { val, .. }
            | IrInst::StoreLocal { val, .. }
            | IrInst::FieldAddr { base: val, .. }
            | IrInst::LoadPtr { addr: val, .. }
            | IrInst::StateTag {
                machine_addr: val, ..
            }
            | IrInst::PortIn { port: val, .. }
            | IrInst::Assert { cond: val, .. } => f(*val),
            IrInst::IndexAddr { base, index, .. } => {
                f(*base);
                f(*index);
            }
            IrInst::StorePtr { addr, val, .. } => {
                f(*addr);
                f(*val);
            }
            IrInst::MemSet { dst_addr, val, .. } => {
                f(*dst_addr);
                f(*val);
            }
            IrInst::MemCopy {
                dst_addr, src_addr, ..
            } => {
                f(*dst_addr);
                f(*src_addr);
            }
            IrInst::MemFill { dst_addr, .. } => f(*dst_addr),
            IrInst::MakeSlice { ptr, len, .. } => {
                f(*ptr);
                f(*len);
            }
            IrInst::SlicePtr { slice, .. } | IrInst::SliceLen { slice, .. } => f(*slice),
            IrInst::PortOut { port, val } => {
                f(*port);
                f(*val);
            }
            IrInst::Call { args, .. } | IrInst::ExternCall { args, .. } => {
                for &a in args {
                    f(a);
                }
            }
            IrInst::CallIndirect { addr, args } => {
                f(*addr);
                for &(a, _) in args {
                    f(a);
                }
            }
            IrInst::StateTransition {
                machine_addr,
                field_vals,
                ..
            } => {
                f(*machine_addr);
                for &(_, v, _) in field_vals {
                    f(v);
                }
            }
            IrInst::StateDispatch {
                machine_addr, args, ..
            } => {
                f(*machine_addr);
                for &a in args {
                    f(a);
                }
            }
            IrInst::PoolAlloc {
                pool_addr,
                init_addr,
                ..
            } => {
                f(*pool_addr);
                f(*init_addr);
            }
            IrInst::PoolFree {
                pool_addr, handle, ..
            }
            | IrInst::PoolElemAddr {
                pool_addr, handle, ..
            } => {
                f(*pool_addr);
                f(*handle);
            }
            IrInst::Phi { incoming, .. } => {
                for &(_, val) in incoming {
                    f(val);
                }
            }
            _ => {}
        }
    }

    /// Invokes `f` mutably on every SSA `ValueId` operand read by this instruction.
    pub fn for_each_use_mut(&mut self, mut f: impl FnMut(&mut ValueId)) {
        match self {
            IrInst::BinOp { lhs, rhs, .. } => {
                f(lhs);
                f(rhs);
            }
            IrInst::UnOp { val, .. }
            | IrInst::Cast { val, .. }
            | IrInst::FixedInt { val, .. }
            | IrInst::StoreLocal { val, .. }
            | IrInst::FieldAddr { base: val, .. }
            | IrInst::LoadPtr { addr: val, .. }
            | IrInst::StateTag {
                machine_addr: val, ..
            }
            | IrInst::PortIn { port: val, .. }
            | IrInst::Assert { cond: val, .. } => f(val),
            IrInst::IndexAddr { base, index, .. } => {
                f(base);
                f(index);
            }
            IrInst::StorePtr { addr, val, .. } => {
                f(addr);
                f(val);
            }
            IrInst::MemSet { dst_addr, val, .. } => {
                f(dst_addr);
                f(val);
            }
            IrInst::MemCopy {
                dst_addr, src_addr, ..
            } => {
                f(dst_addr);
                f(src_addr);
            }
            IrInst::MemFill { dst_addr, .. } => f(dst_addr),
            IrInst::MakeSlice { ptr, len, .. } => {
                f(ptr);
                f(len);
            }
            IrInst::SlicePtr { slice, .. } | IrInst::SliceLen { slice, .. } => f(slice),
            IrInst::PortOut { port, val } => {
                f(port);
                f(val);
            }
            IrInst::Call { args, .. } | IrInst::ExternCall { args, .. } => {
                for a in args {
                    f(a);
                }
            }
            IrInst::CallIndirect { addr, args } => {
                f(addr);
                for (a, _) in args {
                    f(a);
                }
            }
            IrInst::StateTransition {
                machine_addr,
                field_vals,
                ..
            } => {
                f(machine_addr);
                for (_, v, _) in field_vals {
                    f(v);
                }
            }
            IrInst::StateDispatch {
                machine_addr, args, ..
            } => {
                f(machine_addr);
                for a in args {
                    f(a);
                }
            }
            IrInst::PoolAlloc {
                pool_addr,
                init_addr,
                ..
            } => {
                f(pool_addr);
                f(init_addr);
            }
            IrInst::PoolFree {
                pool_addr, handle, ..
            }
            | IrInst::PoolElemAddr {
                pool_addr, handle, ..
            } => {
                f(pool_addr);
                f(handle);
            }
            IrInst::Phi { incoming, .. } => {
                for (_, val) in incoming {
                    f(val);
                }
            }
            _ => {}
        }
    }

    /// Returns `true` if `target` is read as an operand of this instruction.
    pub fn uses_val(&self, target: ValueId) -> bool {
        let mut found = false;
        self.for_each_use(|v| {
            if v == target {
                found = true;
            }
        });
        found
    }
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
