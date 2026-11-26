//! Abstract Syntax Tree (AST) for Aun.

use aun_ir::{ArithMode, Span};

#[derive(Debug, Clone)]
pub struct File {
    pub items: Vec<Item>,
}

#[derive(Debug, Clone)]
pub struct Attr {
    pub name: String,
    pub args: Vec<String>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum Item {
    Fn(FnDecl),
    Struct(StructDecl),
    Enum(EnumDecl),
    Static(StaticDecl),
    Const(ConstDecl),
    ExternAsm(ExternAsmBlock),
}

#[derive(Debug, Clone)]
pub struct FnDecl {
    pub attrs: Vec<Attr>,
    pub is_pub: bool,
    pub name: String,
    pub params: Vec<Param>,
    pub ret_ty: Option<TypeExpr>,
    pub body: Block,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: TypeExpr,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct StructDecl {
    pub attrs: Vec<Attr>,
    pub is_pub: bool,
    pub name: String,
    pub fields: Vec<(String, TypeExpr, Span)>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct EnumDecl {
    pub attrs: Vec<Attr>,
    pub is_pub: bool,
    pub name: String,
    pub repr: Option<TypeExpr>,
    pub variants: Vec<(String, Option<u64>, Span)>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct StaticDecl {
    pub attrs: Vec<Attr>,
    pub is_pub: bool,
    pub is_mut: bool,
    pub name: String,
    pub ty: TypeExpr,
    pub space: Option<String>,
    pub init: Option<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ConstDecl {
    pub is_pub: bool,
    pub name: String,
    pub ty: TypeExpr,
    pub value: Expr,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ExternAsmBlock {
    pub fns: Vec<ExternFnDecl>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ExternFnDecl {
    pub name: String,
    pub params: Vec<ExternParam>,
    pub ret: Option<(TypeExpr, String)>,
    pub requires_banks: Vec<String>,
    pub requires_di: bool,
    pub clobbers: Vec<String>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ExternParam {
    pub name: String,
    pub ty: TypeExpr,
    pub reg: String,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeExpr {
    Unit,
    Never,
    Named(String),
    Fixed { base: Box<TypeExpr>, frac_bits: u8 },
    Ref { mutable: bool, inner: Box<TypeExpr> },
    ConstSlice(Box<TypeExpr>),
    Slice { mutable: bool, inner: Box<TypeExpr> },
    Array { elem: Box<TypeExpr>, len: u16 },
}

#[derive(Debug, Clone)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub tail: Option<Box<Expr>>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Let {
        mutable: bool,
        name: String,
        ty: Option<TypeExpr>,
        init: Expr,
        span: Span,
    },
    Assign {
        target: Expr,
        op: AssignOp,
        value: Expr,
        span: Span,
    },
    Expr(Expr),
    Return(Option<Expr>, Span),
    Break(Span),
    Continue(Span),
    While {
        cond: Expr,
        body: Block,
        span: Span,
    },
    Loop {
        body: Block,
        span: Span,
    },
    For {
        var: String,
        start: Expr,
        end: Expr,
        inclusive: bool,
        body: Block,
        span: Span,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignOp {
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    BitAndAssign,
    BitOrAssign,
    BitXorAssign,
    ShlAssign,
    ShrAssign,
}

#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum ExprKind {
    IntLit { value: u64, suffix: Option<String> },
    FixedLit { text: String, suffix: Option<String> },
    BoolLit(bool),
    ByteLit(u8),
    StrLit(String),
    Ident(String),
    Path(Vec<String>),
    Unary { op: UnaryOp, expr: Box<Expr> },
    Binary { op: BinOp, mode: ArithMode, lhs: Box<Expr>, rhs: Box<Expr> },
    Cast { expr: Box<Expr>, ty: TypeExpr },
    Call { callee: Box<Expr>, args: Vec<Expr> },
    MethodCall { receiver: Box<Expr>, method: String, args: Vec<Expr> },
    Field { base: Box<Expr>, field: String },
    Index { base: Box<Expr>, index: Box<Expr> },
    StructLit { name: String, fields: Vec<(String, Expr)> },
    ArrayLit(Vec<Expr>),
    ArrayRepeat { val: Box<Expr>, len: u16 },
    Ref { mutable: bool, expr: Box<Expr> },
    Deref(Box<Expr>),
    If {
        cond: Box<Expr>,
        then_block: Block,
        else_branch: Option<Box<Expr>>,
    },
    Block(Block),
    Assert { cond: Box<Expr> },
    InlineAsm { template: String, clobbers: Vec<String> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
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
    And,
    Or,
}
