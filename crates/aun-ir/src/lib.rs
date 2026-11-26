//! Target-independent intermediate representation (`aun-ir`) and middle-end passes.

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
    Array { elem: Box<IrType>, len: u16 },
    Ref { mutable: bool, inner: Box<IrType> },
    ConstSlice(Box<IrType>),
}

impl IrType {
    pub fn byte_size(&self, structs: &[StructDef]) -> u16 {
        match self {
            IrType::Unit | IrType::Never => 0,
            IrType::Bool => 1,
            IrType::Int(i) => i.byte_size(),
            IrType::Fixed { base, .. } => base.byte_size(),
            IrType::Struct(id) => structs[*id as usize].size,
            IrType::Array { elem, len } => elem.byte_size(structs) * len,
            IrType::Ref { .. } => 2,        // 16-bit Z80 address
            IrType::ConstSlice(_) => 4,     // (addr: u16, len: u16)
        }
    }

    pub fn is_ref(&self) -> bool {
        matches!(self, IrType::Ref { .. })
    }

    pub fn is_scalar(&self) -> bool {
        matches!(
            self,
            IrType::Bool | IrType::Int(_) | IrType::Fixed { .. } | IrType::Ref { .. }
        )
    }
}

#[derive(Debug, Clone)]
pub struct StructDef {
    pub name: String,
    pub fields: Vec<(String, IrType, u16)>,
    pub size: u16,
}

/// Top-level compiled IR program.
#[derive(Debug, Default)]
pub struct IrProgram {
    pub structs: Vec<StructDef>,
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
}

/// Static global variable.
#[derive(Debug, Clone)]
pub struct IrGlobal {
    pub name: String,
    pub ty: IrType,
    pub mutable: bool,
    pub align: u16,
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
    pub param_count: usize,
    pub locals: Vec<LocalSlot>,
    pub ret_ty: IrType,
    pub is_test: bool,
    pub expected_panic: Option<PanicCode>,
    pub is_reentrant: bool,
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
