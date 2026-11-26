//! Recursive-descent + Pratt parser for Aun.

use crate::ast::*;
use crate::diag::Diagnostic;
use crate::lexer::{Lexer, Token, TokenKind};
use aun_ir::{ArithMode, Span};

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    diagnostics: Vec<Diagnostic>,
}

impl Parser {
    pub fn parse_str(file_id: u16, src: &str) -> Result<File, Vec<Diagnostic>> {
        let tokens = Lexer::new(file_id, src).tokenize()?;
        let mut parser = Self {
            tokens,
            pos: 0,
            diagnostics: Vec::new(),
        };
        let file = parser.parse_file();
        if parser.diagnostics.is_empty() {
            Ok(file)
        } else {
            Err(parser.diagnostics)
        }
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.pos.min(self.tokens.len() - 1)]
    }

    fn peek_kind(&self) -> &TokenKind {
        &self.peek().kind
    }

    fn peek_at(&self, offset: usize) -> &TokenKind {
        let idx = (self.pos + offset).min(self.tokens.len() - 1);
        &self.tokens[idx].kind
    }

    fn bump(&mut self) -> Token {
        let tok = self.peek().clone();
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        tok
    }

    fn check(&self, kind: &TokenKind) -> bool {
        self.peek_kind() == kind
    }

    fn eat(&mut self, kind: &TokenKind) -> bool {
        if self.check(kind) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, kind: &TokenKind, msg: &str) -> Option<Token> {
        if self.check(kind) {
            Some(self.bump())
        } else {
            let span = self.peek().span;
            self.diagnostics.push(Diagnostic::error(
                format!("{msg}, found `{:?}`", self.peek_kind()),
                span,
            ));
            None
        }
    }

    fn expect_ident(&mut self, msg: &str) -> Option<(String, Span)> {
        let tok = self.peek().clone();
        match tok.kind {
            TokenKind::Ident(name) => {
                self.bump();
                Some((name, tok.span))
            }
            _ => {
                self.diagnostics.push(Diagnostic::error(
                    format!("{msg}, found `{:?}`", tok.kind),
                    tok.span,
                ));
                None
            }
        }
    }

    fn parse_file(&mut self) -> File {
        let mut items = Vec::new();
        while !self.check(&TokenKind::Eof) {
            if let Some(item) = self.parse_item() {
                items.push(item);
            } else {
                // Synchronize to next likely item start
                self.bump();
            }
        }
        File { items }
    }

    fn parse_attrs(&mut self) -> Vec<Attr> {
        let mut attrs = Vec::new();
        while self.check(&TokenKind::Hash) {
            let start = self.bump().span;
            if self.expect(&TokenKind::LBracket, "expected `[` after `#`").is_none() {
                break;
            }
            let Some((name, _)) = self.expect_ident("expected attribute name") else {
                break;
            };
            let mut args = Vec::new();
            if self.eat(&TokenKind::LParen) {
                while !self.check(&TokenKind::RParen) && !self.check(&TokenKind::Eof) {
                    let mut arg = String::new();
                    let mut depth = 0usize;
                    while !self.check(&TokenKind::Eof) {
                        if depth == 0 && matches!(self.peek_kind(), TokenKind::Comma | TokenKind::RParen) {
                            break;
                        }
                        let t = self.bump();
                        match t.kind {
                            TokenKind::LParen => {
                                depth += 1;
                                arg.push('(');
                            }
                            TokenKind::RParen => {
                                depth -= 1;
                                arg.push(')');
                            }
                            TokenKind::Ident(s) => arg.push_str(&s),
                            TokenKind::IntLit { value, .. } => arg.push_str(&value.to_string()),
                            TokenKind::Eq => arg.push('='),
                            TokenKind::Comma => arg.push(','),
                            _ => {}
                        }
                    }
                    if !arg.is_empty() {
                        args.push(arg);
                    }
                    if !self.eat(&TokenKind::Comma) {
                        break;
                    }
                }
                self.expect(&TokenKind::RParen, "expected `)` in attribute");
            }
            let end = self
                .expect(&TokenKind::RBracket, "expected `]` after attribute")
                .map(|t| t.span)
                .unwrap_or(start);
            attrs.push(Attr {
                name,
                args,
                span: Span::new(start.file_id, start.start, end.end),
            });
        }
        attrs
    }

    fn parse_item(&mut self) -> Option<Item> {
        let attrs = self.parse_attrs();
        let is_pub = self.eat(&TokenKind::Pub);
        match self.peek_kind() {
            TokenKind::Fn => self.parse_fn(attrs, is_pub).map(Item::Fn),
            TokenKind::Struct => self.parse_struct(attrs, is_pub).map(Item::Struct),
            TokenKind::Enum => self.parse_enum(attrs, is_pub).map(Item::Enum),
            TokenKind::Machine => self.parse_machine(attrs, is_pub).map(Item::Machine),
            TokenKind::Static => self.parse_static(attrs, is_pub).map(Item::Static),
            TokenKind::Const => self.parse_const(is_pub).map(Item::Const),
            TokenKind::Extern => self.parse_extern_asm().map(Item::ExternAsm),
            _ => {
                let span = self.peek().span;
                self.diagnostics.push(Diagnostic::error(
                    format!("expected top-level item, found `{:?}`", self.peek_kind()),
                    span,
                ));
                None
            }
        }
    }

    fn parse_fn(&mut self, attrs: Vec<Attr>, is_pub: bool) -> Option<FnDecl> {
        let start = self.bump().span; // fn
        let (name, _) = self.expect_ident("expected function name")?;
        self.expect(&TokenKind::LParen, "expected `(`")?;
        let mut params = Vec::new();
        while !self.check(&TokenKind::RParen) && !self.check(&TokenKind::Eof) {
            if self.check(&TokenKind::Amp)
                && (matches!(self.peek_at(1), TokenKind::SelfLower)
                    || (matches!(self.peek_at(1), TokenKind::Mut)
                        && matches!(self.peek_at(2), TokenKind::SelfLower)))
            {
                let pspan = self.bump().span; // &
                let mutable = self.eat(&TokenKind::Mut);
                self.bump(); // self
                params.push(Param {
                    name: "self".into(),
                    ty: TypeExpr::Ref {
                        mutable,
                        inner: Box::new(TypeExpr::Named("Self".into())),
                    },
                    span: pspan,
                });
            } else {
                let (pname, pspan) = self.expect_ident("expected parameter name")?;
                self.expect(&TokenKind::Colon, "expected `:` after parameter name")?;
                let ty = self.parse_type()?;
                params.push(Param {
                    name: pname,
                    ty,
                    span: pspan,
                });
            }
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        self.expect(&TokenKind::RParen, "expected `)`")?;
        let ret_ty = if self.eat(&TokenKind::Arrow) {
            Some(self.parse_type()?)
        } else {
            None
        };
        let body = self.parse_block()?;
        let span = Span::new(start.file_id, start.start, body.span.end);
        Some(FnDecl {
            attrs,
            is_pub,
            name,
            params,
            ret_ty,
            body,
            span,
        })
    }

    fn parse_struct(&mut self, attrs: Vec<Attr>, is_pub: bool) -> Option<StructDecl> {
        let start = self.bump().span; // struct
        let (name, _) = self.expect_ident("expected struct name")?;
        self.expect(&TokenKind::LBrace, "expected `{`")?;
        let mut fields = Vec::new();
        while !self.check(&TokenKind::RBrace) && !self.check(&TokenKind::Eof) {
            let (fname, fspan) = self.expect_ident("expected field name")?;
            self.expect(&TokenKind::Colon, "expected `:` after field name")?;
            let fty = self.parse_type()?;
            fields.push((fname, fty, fspan));
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        let end = self.expect(&TokenKind::RBrace, "expected `}`")?.span;
        Some(StructDecl {
            attrs,
            is_pub,
            name,
            fields,
            span: Span::new(start.file_id, start.start, end.end),
        })
    }

    fn parse_machine(&mut self, attrs: Vec<Attr>, is_pub: bool) -> Option<MachineDecl> {
        let start = self.bump().span; // machine
        let (name, _) = self.expect_ident("expected machine name")?;
        self.expect(&TokenKind::LBrace, "expected `{` after machine name")?;

        let mut shared_fields = Vec::new();
        let mut states = Vec::new();
        let mut initial_state = None;
        let mut transitions = Vec::new();
        let mut methods = Vec::new();
        let mut on_blocks = Vec::new();

        while !self.check(&TokenKind::RBrace) && !self.check(&TokenKind::Eof) {
            let item_attrs = self.parse_attrs();
            match self.peek_kind() {
                TokenKind::State => {
                    let sstart = self.bump().span;
                    let (sname, _) = self.expect_ident("expected state name")?;
                    let mut fields = Vec::new();
                    if self.eat(&TokenKind::LBrace) {
                        while !self.check(&TokenKind::RBrace) && !self.check(&TokenKind::Eof) {
                            let (fname, fspan) = self.expect_ident("expected state field name")?;
                            self.expect(&TokenKind::Colon, "expected `:`")?;
                            let fty = self.parse_type()?;
                            fields.push((fname, fty, fspan));
                            if !self.eat(&TokenKind::Comma) {
                                break;
                            }
                        }
                        self.expect(&TokenKind::RBrace, "expected `}`")?;
                        self.eat(&TokenKind::Semi);
                    } else {
                        self.expect(&TokenKind::Semi, "expected `;` or `{` after state name")?;
                    }
                    let end = self.tokens[self.pos.saturating_sub(1)].span.end;
                    states.push(StateDecl {
                        attrs: item_attrs,
                        name: sname,
                        fields,
                        span: Span::new(sstart.file_id, sstart.start, end),
                    });
                }
                TokenKind::Transitions => {
                    self.bump();
                    self.expect(&TokenKind::LBrace, "expected `{` after `transitions`")?;
                    while !self.check(&TokenKind::RBrace) && !self.check(&TokenKind::Eof) {
                        let (from, tspan) = self.expect_ident("expected source state name")?;
                        self.expect(&TokenKind::Arrow, "expected `->` in transition rule")?;
                        let mut targets = Vec::new();
                        let (first_to, _) = self.expect_ident("expected target state name")?;
                        targets.push(first_to);
                        while self.eat(&TokenKind::Pipe) {
                            let (next_to, _) = self.expect_ident("expected target state name")?;
                            targets.push(next_to);
                        }
                        self.expect(&TokenKind::Semi, "expected `;` after transition rule")?;
                        transitions.push((from, targets, tspan));
                    }
                    self.expect(&TokenKind::RBrace, "expected `}`")?;
                }
                TokenKind::On => {
                    self.bump();
                    let (sname, _) = self.expect_ident("expected state name after `on`")?;
                    self.expect(&TokenKind::LBrace, "expected `{`")?;
                    let mut smethods = Vec::new();
                    while !self.check(&TokenKind::RBrace) && !self.check(&TokenKind::Eof) {
                        let mattrs = self.parse_attrs();
                        let mpub = self.eat(&TokenKind::Pub);
                        if let Some(m) = self.parse_fn(mattrs, mpub) {
                            smethods.push(m);
                        } else {
                            self.bump();
                        }
                    }
                    self.expect(&TokenKind::RBrace, "expected `}`")?;
                    on_blocks.push(OnStateBlock {
                        state_name: sname,
                        methods: smethods,
                        span: start,
                    });
                }
                TokenKind::Fn | TokenKind::Pub => {
                    let mpub = self.eat(&TokenKind::Pub);
                    if let Some(m) = self.parse_fn(item_attrs, mpub) {
                        methods.push(m);
                    }
                }
                TokenKind::Ident(id) if id == "initial" => {
                    self.bump();
                    let (iname, _) = self.expect_ident("expected initial state name")?;
                    self.expect(&TokenKind::Semi, "expected `;` after initial state")?;
                    initial_state = Some(iname);
                }
                TokenKind::Ident(_) => {
                    let (fname, fspan) = self.expect_ident("expected shared field name")?;
                    self.expect(&TokenKind::Colon, "expected `:`")?;
                    let fty = self.parse_type()?;
                    self.eat(&TokenKind::Comma);
                    shared_fields.push((fname, fty, fspan));
                }
                other => {
                    let span = self.peek().span;
                    self.diagnostics.push(Diagnostic::error(
                        format!("unexpected token `{other:?}` in `machine {name}`"),
                        span,
                    ));
                    self.bump();
                }
            }
        }
        let end = self.expect(&TokenKind::RBrace, "expected `}`")?.span;
        Some(MachineDecl {
            attrs,
            is_pub,
            name,
            shared_fields,
            states,
            initial_state,
            transitions,
            methods,
            on_blocks,
            span: Span::new(start.file_id, start.start, end.end),
        })
    }

    fn parse_enum(&mut self, attrs: Vec<Attr>, is_pub: bool) -> Option<EnumDecl> {
        let start = self.bump().span; // enum
        let (name, _) = self.expect_ident("expected enum name")?;
        let repr = if self.eat(&TokenKind::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };
        self.expect(&TokenKind::LBrace, "expected `{`")?;
        let mut variants = Vec::new();
        while !self.check(&TokenKind::RBrace) && !self.check(&TokenKind::Eof) {
            let (vname, vspan) = self.expect_ident("expected variant name")?;
            let val = if self.eat(&TokenKind::Eq) {
                match self.bump().kind {
                    TokenKind::IntLit { value, .. } => Some(value),
                    other => {
                        self.diagnostics.push(Diagnostic::error(
                            format!("expected integer discriminant, found `{other:?}`"),
                            vspan,
                        ));
                        None
                    }
                }
            } else {
                None
            };
            variants.push((vname, val, vspan));
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        let end = self.expect(&TokenKind::RBrace, "expected `}`")?.span;
        Some(EnumDecl {
            attrs,
            is_pub,
            name,
            repr,
            variants,
            span: Span::new(start.file_id, start.start, end.end),
        })
    }

    fn parse_static(&mut self, attrs: Vec<Attr>, is_pub: bool) -> Option<StaticDecl> {
        let start = self.bump().span; // static
        let is_mut = self.eat(&TokenKind::Mut);
        let (name, _) = self.expect_ident("expected static variable name")?;
        self.expect(&TokenKind::Colon, "expected `:`")?;
        let ty = self.parse_type()?;
        let space = if self.eat(&TokenKind::In) {
            Some(self.expect_ident("expected memory space or bank name")?.0)
        } else {
            None
        };
        let init = if self.eat(&TokenKind::Eq) {
            Some(self.parse_expr(0)?)
        } else {
            None
        };
        let end = self.expect(&TokenKind::Semi, "expected `;`")?.span;
        Some(StaticDecl {
            attrs,
            is_pub,
            is_mut,
            name,
            ty,
            space,
            init,
            span: Span::new(start.file_id, start.start, end.end),
        })
    }

    fn parse_const(&mut self, is_pub: bool) -> Option<ConstDecl> {
        let start = self.bump().span; // const
        let (name, _) = self.expect_ident("expected const name")?;
        self.expect(&TokenKind::Colon, "expected `:`")?;
        let ty = self.parse_type()?;
        self.expect(&TokenKind::Eq, "expected `=`")?;
        let value = self.parse_expr(0)?;
        let end = self.expect(&TokenKind::Semi, "expected `;`")?.span;
        Some(ConstDecl {
            is_pub,
            name,
            ty,
            value,
            span: Span::new(start.file_id, start.start, end.end),
        })
    }

    fn parse_extern_asm(&mut self) -> Option<ExternAsmBlock> {
        let start = self.bump().span; // extern
        match self.bump().kind {
            TokenKind::StrLit(ref s) if s == "asm" => {}
            other => {
                self.diagnostics.push(Diagnostic::error(
                    format!("expected `\"asm\"` after `extern`, found `{other:?}`"),
                    start,
                ));
                return None;
            }
        }
        self.expect(&TokenKind::LBrace, "expected `{` after `extern \"asm\"`")?;
        let mut fns = Vec::new();
        while !self.check(&TokenKind::RBrace) && !self.check(&TokenKind::Eof) {
            let fstart = self.expect(&TokenKind::Fn, "expected `fn` in `extern \"asm\"`")?.span;
            let (name, _) = self.expect_ident("expected extern function name")?;
            self.expect(&TokenKind::LParen, "expected `(`")?;
            let mut params = Vec::new();
            while !self.check(&TokenKind::RParen) && !self.check(&TokenKind::Eof) {
                let (pname, pspan) = self.expect_ident("expected parameter name")?;
                self.expect(&TokenKind::Colon, "expected `:`")?;
                let ty = self.parse_type()?;
                self.expect(&TokenKind::In, "expected `in <REG>` for extern parameter")?;
                let (reg, _) = self.expect_ident("expected register name")?;
                params.push(ExternParam {
                    name: pname,
                    ty,
                    reg,
                    span: pspan,
                });
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
            self.expect(&TokenKind::RParen, "expected `)`")?;
            let ret = if self.eat(&TokenKind::Arrow) {
                let rty = self.parse_type()?;
                self.expect(&TokenKind::In, "expected `in <REG>` for extern return type")?;
                let (reg, _) = self.expect_ident("expected return register name")?;
                Some((rty, reg))
            } else {
                None
            };
            let mut requires_banks = Vec::new();
            let mut requires_di = false;
            let mut clobbers = Vec::new();
            if self.eat(&TokenKind::Requires) {
                loop {
                    if self.eat(&TokenKind::Bank) {
                        self.expect(&TokenKind::LParen, "expected `(` after `bank`")?;
                        let (bname, _) = self.expect_ident("expected bank name")?;
                        requires_banks.push(bname);
                        self.expect(&TokenKind::RParen, "expected `)`")?;
                    } else if self.eat(&TokenKind::Di) {
                        requires_di = true;
                    } else {
                        break;
                    }
                    if !self.eat(&TokenKind::Comma) {
                        break;
                    }
                }
            }
            if matches!(self.peek_kind(), TokenKind::Ident(s) if s == "clobbers") {
                self.bump();
                self.expect(&TokenKind::LParen, "expected `(` after `clobbers`")?;
                while !self.check(&TokenKind::RParen) && !self.check(&TokenKind::Eof) {
                    let (reg, _) = self.expect_ident("expected register name in `clobbers`")?;
                    clobbers.push(reg);
                    if !self.eat(&TokenKind::Comma) {
                        break;
                    }
                }
                self.expect(&TokenKind::RParen, "expected `)`")?;
            }
            let fend = self.expect(&TokenKind::Semi, "expected `;` after extern fn")?.span;
            fns.push(ExternFnDecl {
                name,
                params,
                ret,
                requires_banks,
                requires_di,
                clobbers,
                span: Span::new(fstart.file_id, fstart.start, fend.end),
            });
        }
        let end = self.expect(&TokenKind::RBrace, "expected `}`")?.span;
        Some(ExternAsmBlock {
            fns,
            span: Span::new(start.file_id, start.start, end.end),
        })
    }

    fn parse_type(&mut self) -> Option<TypeExpr> {
        match self.peek_kind().clone() {
            TokenKind::LParen => {
                self.bump();
                self.expect(&TokenKind::RParen, "expected `)` for unit type")?;
                Some(TypeExpr::Unit)
            }
            TokenKind::Bang => {
                self.bump();
                Some(TypeExpr::Never)
            }
            TokenKind::Bool => {
                self.bump();
                Some(TypeExpr::Named("bool".into()))
            }
            TokenKind::Amp => {
                self.bump();
                if self.eat(&TokenKind::Const) {
                    self.expect(&TokenKind::LBracket, "expected `[` after `&const`")?;
                    let inner = self.parse_type()?;
                    self.expect(&TokenKind::RBracket, "expected `]`")?;
                    return Some(TypeExpr::ConstSlice(Box::new(inner)));
                }
                let mutable = self.eat(&TokenKind::Mut);
                if self.eat(&TokenKind::LBracket) {
                    let inner = self.parse_type()?;
                    self.expect(&TokenKind::RBracket, "expected `]`")?;
                    Some(TypeExpr::Slice {
                        mutable,
                        inner: Box::new(inner),
                    })
                } else {
                    let inner = self.parse_type()?;
                    Some(TypeExpr::Ref {
                        mutable,
                        inner: Box::new(inner),
                    })
                }
            }
            TokenKind::LBracket => {
                self.bump();
                let elem = self.parse_type()?;
                self.expect(&TokenKind::Semi, "expected `;` in array type")?;
                let len = match self.bump() {
                    Token {
                        kind: TokenKind::IntLit { value, .. },
                        ..
                    } => value as u16,
                    other => {
                        self.diagnostics.push(Diagnostic::error(
                            format!("expected array length integer, found `{:?}`", other.kind),
                            other.span,
                        ));
                        return None;
                    }
                };
                self.expect(&TokenKind::RBracket, "expected `]`")?;
                Some(TypeExpr::Array {
                    elem: Box::new(elem),
                    len,
                })
            }
            TokenKind::Ident(name) => {
                self.bump();
                if name == "fixed" && self.eat(&TokenKind::Lt) {
                    let base = self.parse_type()?;
                    self.expect(&TokenKind::Comma, "expected `,` in `fixed<Base, Frac>`")?;
                    let frac_bits = match self.bump() {
                        Token {
                            kind: TokenKind::IntLit { value, .. },
                            ..
                        } => value as u8,
                        other => {
                            self.diagnostics.push(Diagnostic::error(
                                format!("expected fractional bit count, found `{:?}`", other.kind),
                                other.span,
                            ));
                            return None;
                        }
                    };
                    self.expect(&TokenKind::Gt, "expected `>`")?;
                    Some(TypeExpr::Fixed {
                        base: Box::new(base),
                        frac_bits,
                    })
                } else if name == "Pool" && self.eat(&TokenKind::Lt) {
                    let elem = self.parse_type()?;
                    self.expect(&TokenKind::Comma, "expected `,` in `Pool<T, N>`")?;
                    let capacity = match self.bump() {
                        Token {
                            kind: TokenKind::IntLit { value, .. },
                            ..
                        } => value as u16,
                        other => {
                            self.diagnostics.push(Diagnostic::error(
                                format!("expected Pool capacity integer, found `{:?}`", other.kind),
                                other.span,
                            ));
                            return None;
                        }
                    };
                    self.expect(&TokenKind::Gt, "expected `>`")?;
                    Some(TypeExpr::Pool {
                        elem: Box::new(elem),
                        capacity,
                    })
                } else if name == "Handle" && self.eat(&TokenKind::Lt) {
                    let elem = self.parse_type()?;
                    self.expect(&TokenKind::Gt, "expected `>`")?;
                    Some(TypeExpr::Handle(Box::new(elem)))
                } else {
                    Some(TypeExpr::Named(name))
                }
            }
            other => {
                let span = self.peek().span;
                self.diagnostics.push(Diagnostic::error(
                    format!("expected type, found `{other:?}`"),
                    span,
                ));
                None
            }
        }
    }

    fn parse_block(&mut self) -> Option<Block> {
        let start = self.expect(&TokenKind::LBrace, "expected `{`")?.span;
        let mut stmts = Vec::new();
        let mut tail = None;
        while !self.check(&TokenKind::RBrace) && !self.check(&TokenKind::Eof) {
            match self.peek_kind() {
                TokenKind::Let => {
                    if let Some(s) = self.parse_let() {
                        stmts.push(s);
                    }
                }
                TokenKind::Return => {
                    let rstart = self.bump().span;
                    let val = if self.check(&TokenKind::Semi) {
                        None
                    } else {
                        Some(self.parse_expr(0)?)
                    };
                    let end = self.expect(&TokenKind::Semi, "expected `;` after `return`")?.span;
                    stmts.push(Stmt::Return(
                        val,
                        Span::new(rstart.file_id, rstart.start, end.end),
                    ));
                }
                TokenKind::Break => {
                    let bstart = self.bump().span;
                    let end = self.expect(&TokenKind::Semi, "expected `;` after `break`")?.span;
                    stmts.push(Stmt::Break(Span::new(bstart.file_id, bstart.start, end.end)));
                }
                TokenKind::Continue => {
                    let cstart = self.bump().span;
                    let end = self.expect(&TokenKind::Semi, "expected `;` after `continue`")?.span;
                    stmts.push(Stmt::Continue(Span::new(cstart.file_id, cstart.start, end.end)));
                }
                TokenKind::While => {
                    let wstart = self.bump().span;
                    let cond = self.parse_expr_no_struct()?;
                    let body = self.parse_block()?;
                    let span = Span::new(wstart.file_id, wstart.start, body.span.end);
                    stmts.push(Stmt::While { cond, body, span });
                }
                TokenKind::Loop => {
                    let lstart = self.bump().span;
                    let body = self.parse_block()?;
                    let span = Span::new(lstart.file_id, lstart.start, body.span.end);
                    stmts.push(Stmt::Loop { body, span });
                }
                TokenKind::For => {
                    let fstart = self.bump().span;
                    let (var, _) = self.expect_ident("expected loop variable name")?;
                    self.expect(&TokenKind::In, "expected `in` after loop variable")?;
                    let start_expr = self.parse_expr_no_struct()?;
                    let inclusive = if self.eat(&TokenKind::DotDotEq) {
                        true
                    } else {
                        self.expect(&TokenKind::DotDot, "expected `..` or `..=` in `for` range")?;
                        false
                    };
                    let end_expr = self.parse_expr_no_struct()?;
                    let body = self.parse_block()?;
                    let span = Span::new(fstart.file_id, fstart.start, body.span.end);
                    stmts.push(Stmt::For {
                        var,
                        start: start_expr,
                        end: end_expr,
                        inclusive,
                        body,
                        span,
                    });
                }
                _ => {
                    let expr = self.parse_expr(0)?;
                    if let Some(op) = self.try_parse_assign_op() {
                        let value = self.parse_expr(0)?;
                        let end = self.expect(&TokenKind::Semi, "expected `;` after assignment")?.span;
                        let span = Span::new(expr.span.file_id, expr.span.start, end.end);
                        stmts.push(Stmt::Assign {
                            target: expr,
                            op,
                            value,
                            span,
                        });
                    } else if self.eat(&TokenKind::Semi) {
                        stmts.push(Stmt::Expr(expr));
                    } else if self.check(&TokenKind::RBrace) {
                        tail = Some(Box::new(expr));
                    } else if matches!(expr.kind, ExprKind::If { .. } | ExprKind::Block(_)) {
                        stmts.push(Stmt::Expr(expr));
                    } else {
                        let span = self.peek().span;
                        self.diagnostics
                            .push(Diagnostic::error("expected `;` or `}`", span));
                        break;
                    }
                }
            }
        }
        let end = self.expect(&TokenKind::RBrace, "expected `}`")?.span;
        Some(Block {
            stmts,
            tail,
            span: Span::new(start.file_id, start.start, end.end),
        })
    }

    fn parse_let(&mut self) -> Option<Stmt> {
        let start = self.bump().span; // let
        let mutable = self.eat(&TokenKind::Mut);
        let (name, _) = self.expect_ident("expected variable name")?;
        let ty = if self.eat(&TokenKind::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };
        self.expect(&TokenKind::Eq, "expected `=` in `let` binding")?;
        let init = self.parse_expr(0)?;
        let end = self.expect(&TokenKind::Semi, "expected `;` after `let`")?.span;
        Some(Stmt::Let {
            mutable,
            name,
            ty,
            init,
            span: Span::new(start.file_id, start.start, end.end),
        })
    }

    fn try_parse_assign_op(&mut self) -> Option<AssignOp> {
        let op = match self.peek_kind() {
            TokenKind::Eq => AssignOp::Assign,
            TokenKind::PlusEq => AssignOp::AddAssign,
            TokenKind::MinusEq => AssignOp::SubAssign,
            TokenKind::StarEq => AssignOp::MulAssign,
            TokenKind::SlashEq => AssignOp::DivAssign,
            TokenKind::AmpEq => AssignOp::BitAndAssign,
            TokenKind::PipeEq => AssignOp::BitOrAssign,
            TokenKind::CaretEq => AssignOp::BitXorAssign,
            TokenKind::ShlEq => AssignOp::ShlAssign,
            TokenKind::ShrEq => AssignOp::ShrAssign,
            _ => return None,
        };
        self.bump();
        Some(op)
    }

    fn parse_expr_no_struct(&mut self) -> Option<Expr> {
        self.parse_expr_inner(0, false)
    }

    fn parse_expr(&mut self, min_bp: u8) -> Option<Expr> {
        self.parse_expr_inner(min_bp, true)
    }

    fn parse_expr_inner(&mut self, min_bp: u8, allow_struct_lit: bool) -> Option<Expr> {
        let mut lhs = self.parse_prefix(allow_struct_lit)?;

        loop {
            // Postfix cast: `expr as Type`
            if self.check(&TokenKind::As) && 11 >= min_bp {
                self.bump();
                let ty = self.parse_type()?;
                let end = self.tokens[self.pos.saturating_sub(1)].span.end;
                let span = Span::new(lhs.span.file_id, lhs.span.start, end);
                lhs = Expr {
                    kind: ExprKind::Cast {
                        expr: Box::new(lhs),
                        ty,
                    },
                    span,
                };
                continue;
            }

            let Some((l_bp, r_bp, op, mode)) = infix_bp(self.peek_kind()) else {
                break;
            };
            if l_bp < min_bp {
                break;
            }
            self.bump();
            let rhs = self.parse_expr_inner(r_bp, allow_struct_lit)?;
            let span = Span::new(lhs.span.file_id, lhs.span.start, rhs.span.end);
            lhs = Expr {
                kind: ExprKind::Binary {
                    op,
                    mode,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
                span,
            };
        }

        Some(lhs)
    }

    fn parse_prefix(&mut self, allow_struct_lit: bool) -> Option<Expr> {
        let expr = match self.peek_kind().clone() {
            TokenKind::Minus => {
                let start = self.bump().span;
                let inner = self.parse_prefix(allow_struct_lit)?;
                let span = Span::new(start.file_id, start.start, inner.span.end);
                Expr {
                    kind: ExprKind::Unary {
                        op: UnaryOp::Neg,
                        expr: Box::new(inner),
                    },
                    span,
                }
            }
            TokenKind::Bang => {
                let start = self.bump().span;
                let inner = self.parse_prefix(allow_struct_lit)?;
                let span = Span::new(start.file_id, start.start, inner.span.end);
                Expr {
                    kind: ExprKind::Unary {
                        op: UnaryOp::Not,
                        expr: Box::new(inner),
                    },
                    span,
                }
            }
            TokenKind::Amp => {
                let start = self.bump().span;
                let mutable = self.eat(&TokenKind::Mut);
                let inner = self.parse_prefix(allow_struct_lit)?;
                let span = Span::new(start.file_id, start.start, inner.span.end);
                Expr {
                    kind: ExprKind::Ref {
                        mutable,
                        expr: Box::new(inner),
                    },
                    span,
                }
            }
            TokenKind::Star => {
                let start = self.bump().span;
                let inner = self.parse_prefix(allow_struct_lit)?;
                let span = Span::new(start.file_id, start.start, inner.span.end);
                Expr {
                    kind: ExprKind::Deref(Box::new(inner)),
                    span,
                }
            }
            _ => self.parse_primary(allow_struct_lit)?,
        };

        self.parse_postfix(expr)
    }

    fn parse_primary(&mut self, allow_struct_lit: bool) -> Option<Expr> {
        let tok = self.peek().clone();
        match tok.kind {
            TokenKind::IntLit { value, suffix } => {
                self.bump();
                Some(Expr {
                    kind: ExprKind::IntLit { value, suffix },
                    span: tok.span,
                })
            }
            TokenKind::FixedLit { text, suffix } => {
                self.bump();
                Some(Expr {
                    kind: ExprKind::FixedLit { text, suffix },
                    span: tok.span,
                })
            }
            TokenKind::True => {
                self.bump();
                Some(Expr {
                    kind: ExprKind::BoolLit(true),
                    span: tok.span,
                })
            }
            TokenKind::False => {
                self.bump();
                Some(Expr {
                    kind: ExprKind::BoolLit(false),
                    span: tok.span,
                })
            }
            TokenKind::ByteLit(b) => {
                self.bump();
                Some(Expr {
                    kind: ExprKind::ByteLit(b),
                    span: tok.span,
                })
            }
            TokenKind::StrLit(s) => {
                self.bump();
                Some(Expr {
                    kind: ExprKind::StrLit(s),
                    span: tok.span,
                })
            }
            TokenKind::Asm => {
                self.bump();
                self.expect(&TokenKind::Bang, "expected `!` after `asm`")?;
                self.expect(&TokenKind::LParen, "expected `(`")?;
                let template = match self.bump().kind {
                    TokenKind::StrLit(s) => s,
                    other => {
                        self.diagnostics.push(Diagnostic::error(
                            format!("expected assembly string literal, found `{other:?}`"),
                            tok.span,
                        ));
                        return None;
                    }
                };
                let mut clobbers = Vec::new();
                if self.eat(&TokenKind::Comma)
                    && matches!(self.peek_kind(), TokenKind::Ident(s) if s == "clobbers")
                {
                    self.bump();
                    self.expect(&TokenKind::LParen, "expected `(` after `clobbers`")?;
                    while !self.check(&TokenKind::RParen) && !self.check(&TokenKind::Eof) {
                        let (r, _) = self.expect_ident("expected register name")?;
                        clobbers.push(r);
                        if !self.eat(&TokenKind::Comma) {
                            break;
                        }
                    }
                    self.expect(&TokenKind::RParen, "expected `)`")?;
                }
                let end = self.expect(&TokenKind::RParen, "expected `)`")?.span;
                Some(Expr {
                    kind: ExprKind::InlineAsm { template, clobbers },
                    span: Span::new(tok.span.file_id, tok.span.start, end.end),
                })
            }
            TokenKind::Goto => {
                self.bump();
                let (state_name, sspan) = self.expect_ident("expected target state name after `goto`")?;
                let mut fields = Vec::new();
                let mut end = sspan.end;
                if self.eat(&TokenKind::LBrace) {
                    while !self.check(&TokenKind::RBrace) && !self.check(&TokenKind::Eof) {
                        let (fname, _) = self.expect_ident("expected state field name")?;
                        self.expect(&TokenKind::Colon, "expected `:`")?;
                        let val = self.parse_expr(0)?;
                        fields.push((fname, val));
                        if !self.eat(&TokenKind::Comma) {
                            break;
                        }
                    }
                    end = self.expect(&TokenKind::RBrace, "expected `}`")?.span.end;
                }
                Some(Expr {
                    kind: ExprKind::Goto { state_name, fields },
                    span: Span::new(tok.span.file_id, tok.span.start, end),
                })
            }
            TokenKind::SelfLower => {
                self.bump();
                Some(Expr {
                    kind: ExprKind::Ident("self".into()),
                    span: tok.span,
                })
            }
            TokenKind::Ident(name) => {
                self.bump();
                // assert!(cond)
                if name == "assert" && self.eat(&TokenKind::Bang) {
                    self.expect(&TokenKind::LParen, "expected `(` after `assert!`")?;
                    let cond = self.parse_expr(0)?;
                    let end = self.expect(&TokenKind::RParen, "expected `)`")?.span;
                    return Some(Expr {
                        kind: ExprKind::Assert {
                            cond: Box::new(cond),
                        },
                        span: Span::new(tok.span.file_id, tok.span.start, end.end),
                    });
                }
                // is_state!(expr, Machine::State)
                if name == "is_state" && self.eat(&TokenKind::Bang) {
                    self.expect(&TokenKind::LParen, "expected `(` after `is_state!`")?;
                    let target = self.parse_expr(0)?;
                    self.expect(&TokenKind::Comma, "expected `,` in `is_state!`")?;
                    let (machine_name, _) = self.expect_ident("expected machine name")?;
                    self.expect(&TokenKind::ColonColon, "expected `::`")?;
                    let (state_name, _) = self.expect_ident("expected state name")?;
                    let end = self.expect(&TokenKind::RParen, "expected `)`")?.span;
                    return Some(Expr {
                        kind: ExprKind::IsState {
                            expr: Box::new(target),
                            machine_name,
                            state_name,
                        },
                        span: Span::new(tok.span.file_id, tok.span.start, end.end),
                    });
                }
                // Path: Foo::Bar
                if self.check(&TokenKind::ColonColon) {
                    let mut segs = vec![name];
                    let mut end = tok.span.end;
                    while self.eat(&TokenKind::ColonColon) {
                        let (seg, sspan) = self.expect_ident("expected path segment")?;
                        end = sspan.end;
                        segs.push(seg);
                    }
                    return Some(Expr {
                        kind: ExprKind::Path(segs),
                        span: Span::new(tok.span.file_id, tok.span.start, end),
                    });
                }
                // Struct literal: Name { field: val, ... }
                if allow_struct_lit
                    && self.check(&TokenKind::LBrace)
                    && (matches!(self.peek_at(1), TokenKind::RBrace)
                        || (matches!(self.peek_at(1), TokenKind::Ident(_))
                            && matches!(self.peek_at(2), TokenKind::Colon)))
                {
                    self.bump(); // {
                    let mut fields = Vec::new();
                    while !self.check(&TokenKind::RBrace) && !self.check(&TokenKind::Eof) {
                        let (fname, _) = self.expect_ident("expected field name")?;
                        self.expect(&TokenKind::Colon, "expected `:`")?;
                        let val = self.parse_expr(0)?;
                        fields.push((fname, val));
                        if !self.eat(&TokenKind::Comma) {
                            break;
                        }
                    }
                    let end = self.expect(&TokenKind::RBrace, "expected `}`")?.span;
                    return Some(Expr {
                        kind: ExprKind::StructLit { name, fields },
                        span: Span::new(tok.span.file_id, tok.span.start, end.end),
                    });
                }
                Some(Expr {
                    kind: ExprKind::Ident(name),
                    span: tok.span,
                })
            }
            TokenKind::LParen => {
                self.bump();
                let expr = self.parse_expr(0)?;
                self.expect(&TokenKind::RParen, "expected `)`")?;
                Some(expr)
            }
            TokenKind::LBracket => {
                self.bump();
                if self.eat(&TokenKind::RBracket) {
                    return Some(Expr {
                        kind: ExprKind::ArrayLit(Vec::new()),
                        span: tok.span,
                    });
                }
                let first = self.parse_expr(0)?;
                if self.eat(&TokenKind::Semi) {
                    let len = match self.bump() {
                        Token {
                            kind: TokenKind::IntLit { value, .. },
                            ..
                        } => value as u16,
                        other => {
                            self.diagnostics.push(Diagnostic::error(
                                format!("expected array repeat length, found `{:?}`", other.kind),
                                other.span,
                            ));
                            return None;
                        }
                    };
                    let end = self.expect(&TokenKind::RBracket, "expected `]`")?.span;
                    return Some(Expr {
                        kind: ExprKind::ArrayRepeat {
                            val: Box::new(first),
                            len,
                        },
                        span: Span::new(tok.span.file_id, tok.span.start, end.end),
                    });
                }
                let mut elems = vec![first];
                while self.eat(&TokenKind::Comma) {
                    if self.check(&TokenKind::RBracket) {
                        break;
                    }
                    elems.push(self.parse_expr(0)?);
                }
                let end = self.expect(&TokenKind::RBracket, "expected `]`")?.span;
                Some(Expr {
                    kind: ExprKind::ArrayLit(elems),
                    span: Span::new(tok.span.file_id, tok.span.start, end.end),
                })
            }
            TokenKind::If => self.parse_if(),
            TokenKind::LBrace => {
                let block = self.parse_block()?;
                let span = block.span;
                Some(Expr {
                    kind: ExprKind::Block(block),
                    span,
                })
            }
            other => {
                self.diagnostics.push(Diagnostic::error(
                    format!("expected expression, found `{other:?}`"),
                    tok.span,
                ));
                None
            }
        }
    }

    fn parse_if(&mut self) -> Option<Expr> {
        let start = self.bump().span; // if
        let cond = self.parse_expr_no_struct()?;
        let then_block = self.parse_block()?;
        let mut end = then_block.span.end;
        let else_branch = if self.eat(&TokenKind::Else) {
            let e = if self.check(&TokenKind::If) {
                self.parse_if()?
            } else {
                let b = self.parse_block()?;
                let bspan = b.span;
                Expr {
                    kind: ExprKind::Block(b),
                    span: bspan,
                }
            };
            end = e.span.end;
            Some(Box::new(e))
        } else {
            None
        };
        Some(Expr {
            kind: ExprKind::If {
                cond: Box::new(cond),
                then_block,
                else_branch,
            },
            span: Span::new(start.file_id, start.start, end),
        })
    }

    fn parse_postfix(&mut self, mut expr: Expr) -> Option<Expr> {
        loop {
            match self.peek_kind() {
                TokenKind::LParen => {
                    self.bump();
                    let args = self.parse_call_args()?;
                    let end = self.expect(&TokenKind::RParen, "expected `)`")?.span;
                    let span = Span::new(expr.span.file_id, expr.span.start, end.end);
                    expr = Expr {
                        kind: ExprKind::Call {
                            callee: Box::new(expr),
                            args,
                        },
                        span,
                    };
                }
                TokenKind::Dot => {
                    self.bump();
                    let (field, fspan) = if self.check(&TokenKind::Goto) {
                        let t = self.bump();
                        ("goto".to_string(), t.span)
                    } else {
                        self.expect_ident("expected field or method name")?
                    };
                    if self.eat(&TokenKind::LParen) {
                        let args = self.parse_call_args()?;
                        let end = self.expect(&TokenKind::RParen, "expected `)`")?.span;
                        let span = Span::new(expr.span.file_id, expr.span.start, end.end);
                        expr = Expr {
                            kind: ExprKind::MethodCall {
                                receiver: Box::new(expr),
                                method: field,
                                args,
                            },
                            span,
                        };
                    } else {
                        let span = Span::new(expr.span.file_id, expr.span.start, fspan.end);
                        expr = Expr {
                            kind: ExprKind::Field {
                                base: Box::new(expr),
                                field,
                            },
                            span,
                        };
                    }
                }
                TokenKind::LBracket => {
                    self.bump();
                    let index = self.parse_expr(0)?;
                    let end = self.expect(&TokenKind::RBracket, "expected `]`")?.span;
                    let span = Span::new(expr.span.file_id, expr.span.start, end.end);
                    expr = Expr {
                        kind: ExprKind::Index {
                            base: Box::new(expr),
                            index: Box::new(index),
                        },
                        span,
                    };
                }
                _ => break,
            }
        }
        Some(expr)
    }

    fn parse_call_args(&mut self) -> Option<Vec<Expr>> {
        let mut args = Vec::new();
        while !self.check(&TokenKind::RParen) && !self.check(&TokenKind::Eof) {
            args.push(self.parse_expr(0)?);
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        Some(args)
    }
}

fn infix_bp(kind: &TokenKind) -> Option<(u8, u8, BinOp, ArithMode)> {
    use ArithMode::*;
    use BinOp::*;
    let res = match kind {
        TokenKind::PipePipe => (1, 2, Or, Checked),
        TokenKind::AmpAmp => (3, 4, And, Checked),
        TokenKind::EqEq => (5, 6, Eq, Checked),
        TokenKind::BangEq => (5, 6, Ne, Checked),
        TokenKind::Lt => (5, 6, Lt, Checked),
        TokenKind::LtEq => (5, 6, Le, Checked),
        TokenKind::Gt => (5, 6, Gt, Checked),
        TokenKind::GtEq => (5, 6, Ge, Checked),
        TokenKind::Pipe => (7, 8, BitOr, Checked),
        TokenKind::Caret => (9, 10, BitXor, Checked),
        TokenKind::Amp => (11, 12, BitAnd, Checked),
        TokenKind::Shl => (13, 14, Shl, Checked),
        TokenKind::Shr => (13, 14, Shr, Checked),
        TokenKind::Plus => (15, 16, Add, Checked),
        TokenKind::Minus => (15, 16, Sub, Checked),
        TokenKind::PlusPipe => (15, 16, Add, Saturating),
        TokenKind::MinusPipe => (15, 16, Sub, Saturating),
        TokenKind::PlusPercent => (15, 16, Add, Wrapping),
        TokenKind::MinusPercent => (15, 16, Sub, Wrapping),
        TokenKind::Star => (17, 18, Mul, Checked),
        TokenKind::StarPipe => (17, 18, Mul, Saturating),
        TokenKind::StarPercent => (17, 18, Mul, Wrapping),
        TokenKind::Slash => (17, 18, Div, Checked),
        TokenKind::Percent => (17, 18, Rem, Checked),
        _ => return None,
    };
    Some(res)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_phase1_program() {
        let src = r#"
            struct Vec2 {
                x: q8_8,
                y: q8_8,
            }

            extern "asm" {
                fn hal_wait_vsync(prev: u8 in A) -> u16 in DE
                    clobbers(BC, HL, F);
                fn gfx_fill_rect_alu(col: u8 in A, y: u8 in L)
                    requires bank(VramAlu), di
                    clobbers(AF, BC, DE, HL);
            }

            #[test]
            fn add_vec(a: &Vec2, out: &mut Vec2) {
                out.x = a.x +| 1.5q8_8;
                out.y = a.y +% 0.25q8_8;
                io_out(0xFE, 1u8);
                assert!(out.x.int() >= 0);
            }
        "#;

        let file = Parser::parse_str(0, src).expect("parse should succeed");
        assert_eq!(file.items.len(), 3);
        assert!(matches!(file.items[0], Item::Struct(_)));
        assert!(matches!(file.items[1], Item::ExternAsm(_)));
        assert!(matches!(file.items[2], Item::Fn(_)));
    }
}
