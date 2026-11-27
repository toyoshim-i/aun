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
    Machine(MachineDecl),
    Target(TargetDecl),
    Static(StaticDecl),
    Const(ConstDecl),
    ExternAsm(ExternAsmBlock),
    Include { path: String, span: Span },
    Mod {
        attrs: Vec<Attr>,
        is_pub: bool,
        name: String,
        items: Option<Vec<Item>>,
        span: Span,
    },
    Use(UseDecl),
}

#[derive(Debug, Clone)]
pub struct UseDecl {
    pub attrs: Vec<Attr>,
    pub is_pub: bool,
    pub tree: UseTree,
    pub span: Span,
}

impl Item {
    pub fn attrs(&self) -> &[Attr] {
        match self {
            Item::Fn(d) => &d.attrs,
            Item::Struct(d) => &d.attrs,
            Item::Enum(d) => &d.attrs,
            Item::Machine(d) => &d.attrs,
            Item::Static(d) => &d.attrs,
            Item::Const(d) => &d.attrs,
            Item::ExternAsm(d) => &d.attrs,
            Item::Mod { attrs, .. } => attrs,
            Item::Use(d) => &d.attrs,
            Item::Target(_) | Item::Include { .. } => &[],
        }
    }

    pub fn is_cfg_test(&self) -> bool {
        self.attrs()
            .iter()
            .any(|a| a.name == "cfg" && a.args.iter().any(|arg| arg == "test"))
    }
}

#[derive(Debug, Clone)]
pub enum UseTree {
    Path { seg: String, rest: Box<UseTree> },
    Name { name: String, alias: Option<String> },
    Glob,
    Group(Vec<UseTree>),
}

#[derive(Debug, Clone)]
pub struct TargetDecl {
    pub name: String,
    pub arch: String,
    pub spaces: Vec<(String, Option<String>, u16, u16, Span)>,
    pub stacks: Vec<(String, u16, u16, Span)>,
    pub bank_groups: Vec<BankGroupDecl>,
    pub interrupt_controllers: Vec<IntCtrlDecl>,
    pub overlay_slots: Vec<OverlaySlotDecl>,
    pub on_swap_fn: Option<String>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct BankGroupDecl {
    pub name: String,
    pub start: u16,
    pub end: u16,
    pub banks: Vec<(String, u16, u16, bool /* requires_di */, Span)>,
    pub default_banks: Vec<String>,
    pub switches: Vec<(Vec<String>, Vec<(u8, String)>)>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct IntCtrlDecl {
    pub name: String,
    pub disable_asm: Option<String>,
    pub enable_asm: Option<String>,
    pub sources: Vec<(String, String, u16, Vec<(u8, String)>, Span)>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct OverlaySlotDecl {
    pub name: String,
    pub start: u16,
    pub end: u16,
    pub space: Option<String>,
    pub load_fn: Option<String>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct MachineDecl {
    pub attrs: Vec<Attr>,
    pub is_pub: bool,
    pub name: String,
    pub shared_fields: Vec<(String, TypeExpr, Span)>,
    pub states: Vec<StateDecl>,
    pub initial_state: Option<String>,
    pub transitions: Vec<(String, Vec<String>, Span)>,
    pub methods: Vec<FnDecl>,
    pub on_blocks: Vec<OnStateBlock>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct StateDecl {
    pub attrs: Vec<Attr>,
    pub name: String,
    pub fields: Vec<(String, TypeExpr, Span)>,
    pub children: Vec<StateDecl>,
    pub initial_child: Option<String>,
    pub statics: Vec<StaticDecl>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct OnStateBlock {
    pub state_name: String,
    pub methods: Vec<FnDecl>,
    pub span: Span,
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
    pub is_volatile: bool,
    pub is_mut: bool,
    pub name: String,
    pub ty: TypeExpr,
    pub space: Option<String>,
    pub fixed_addr: Option<u16>,
    pub init: Option<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ConstDecl {
    pub attrs: Vec<Attr>,
    pub is_pub: bool,
    pub name: String,
    pub ty: TypeExpr,
    pub value: Expr,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ExternAsmBlock {
    pub attrs: Vec<Attr>,
    pub fns: Vec<ExternFnDecl>,
    pub statics: Vec<ExternStaticDecl>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ExternStaticDecl {
    pub name: String,
    pub is_volatile: bool,
    pub is_mut: bool,
    pub in_rom: bool,
    pub space: Option<String>,
    pub ty: TypeExpr,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ExternFnDecl {
    pub attrs: Vec<Attr>,
    pub name: String,
    pub params: Vec<ExternParam>,
    pub ret: Option<(TypeExpr, String)>,
    pub requires_banks: Vec<String>,
    pub requires_di: bool,
    pub clobbers: Option<Vec<String>>,
    pub fixed_addr: Option<u16>,
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
    Pool { elem: Box<TypeExpr>, capacity: u16 },
    Handle(Box<TypeExpr>),
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
    ByteStrLit(Vec<u8>),
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
    WithBank {
        banks: Vec<String>,
        body: Block,
    },
    Goto {
        state_name: String,
        fields: Vec<(String, Expr)>,
    },
    IsState {
        expr: Box<Expr>,
        segments: Vec<String>,
    },
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
