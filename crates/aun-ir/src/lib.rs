//! Target-independent intermediate representation (`aun-ir`) and middle-end passes.

use std::fmt;

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
    pub const fn byte_size(self) -> usize {
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
    Ptr(Box<IrType>),
    ConstSlice(Box<IrType>),
}

/// Top-level compiled IR program.
#[derive(Debug, Default)]
pub struct IrProgram {
    pub functions: Vec<IrFunction>,
    pub globals: Vec<IrGlobal>,
    pub externs: Vec<IrExternFn>,
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

/// IR function (CFG of basic blocks).
#[derive(Debug, Clone)]
pub struct IrFunction {
    pub name: String,
    pub params: Vec<(String, IrType)>,
    pub ret_ty: IrType,
    pub is_test: bool,
    pub is_reentrant: bool,
    pub blocks: Vec<BasicBlock>,
}

#[derive(Debug, Clone, Default)]
pub struct BasicBlock {
    pub id: u32,
    pub insts: Vec<IrInst>,
    pub term: Terminator,
}

#[derive(Debug, Clone)]
pub enum IrInst {
    Nop,
    PortOut { port: u32, val: u32 },
    PortIn { dst: u32, port: u32 },
    Panic { code: PanicCode, site_id: u16 },
}

#[derive(Debug, Clone, Default)]
pub enum Terminator {
    #[default]
    Unreachable,
    Return(Option<u32>),
    Goto(u32),
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
