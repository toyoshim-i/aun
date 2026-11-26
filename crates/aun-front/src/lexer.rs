//! Lexer for Aun source files (`.aun`).

use crate::diag::Diagnostic;
use aun_ir::Span;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    // Identifiers & literals
    Ident(String),
    IntLit { value: u64, suffix: Option<String> },
    FixedLit { text: String, suffix: Option<String> },
    ByteLit(u8),
    StrLit(String),

    // Keywords
    As,
    Asm,
    Bank,
    BankGroup,
    Bool,
    Break,
    Const,
    Continue,
    Default,
    Di,
    Ei,
    Else,
    Enum,
    Extern,
    False,
    Fn,
    For,
    Goto,
    If,
    In,
    InterruptController,
    Let,
    Loop,
    Machine,
    Match,
    Mod,
    Mut,
    On,
    OverlaySlot,
    Pub,
    Requires,
    Return,
    SelfLower,
    SelfUpper,
    Space,
    Stack,
    State,
    Static,
    Struct,
    Switch,
    Target,
    Transitions,
    True,
    Type,
    Use,
    While,
    With,

    // Operators
    Plus,          // +
    Minus,         // -
    Star,          // *
    Slash,         // /
    Percent,       // %
    PlusPipe,      // +|
    MinusPipe,     // -|
    StarPipe,      // *|
    PlusPercent,   // +%
    MinusPercent,  // -%
    StarPercent,   // *%
    Eq,            // =
    PlusEq,        // +=
    MinusEq,       // -=
    StarEq,        // *=
    SlashEq,       // /=
    AmpEq,         // &=
    PipeEq,        // |=
    CaretEq,       // ^=
    ShlEq,         // <<=
    ShrEq,         // >>=
    EqEq,          // ==
    BangEq,        // !=
    Lt,            // <
    LtEq,          // <=
    Gt,            // >
    GtEq,          // >=
    AmpAmp,        // &&
    PipePipe,      // ||
    Bang,          // !
    Amp,           // &
    Pipe,          // |
    Caret,         // ^
    Shl,           // <<
    Shr,           // >>

    // Punctuation
    LParen,        // (
    RParen,        // )
    LBrace,        // {
    RBrace,        // }
    LBracket,      // [
    RBracket,      // ]
    Comma,         // ,
    Semi,          // ;
    Colon,         // :
    ColonColon,    // ::
    Dot,           // .
    DotDot,        // ..
    DotDotEq,      // ..=
    Arrow,         // ->
    FatArrow,      // =>
    Hash,          // #
    At,            // @

    Eof,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

pub struct Lexer<'a> {
    file_id: u16,
    src: &'a [u8],
    pos: usize,
    diagnostics: Vec<Diagnostic>,
}

impl<'a> Lexer<'a> {
    pub fn new(file_id: u16, src: &'a str) -> Self {
        Self {
            file_id,
            src: src.as_bytes(),
            pos: 0,
            diagnostics: Vec::new(),
        }
    }

    pub fn tokenize(mut self) -> Result<Vec<Token>, Vec<Diagnostic>> {
        let mut tokens = Vec::new();
        loop {
            self.skip_ws_and_comments();
            let start = self.pos;
            let Some(b) = self.peek() else {
                tokens.push(Token {
                    kind: TokenKind::Eof,
                    span: Span::new(self.file_id, start as u32, start as u32),
                });
                break;
            };

            let kind = match b {
                b'b' if self.peek_at(1) == Some(b'\'') => self.lex_byte_lit(start),
                b'A'..=b'Z' | b'a'..=b'z' | b'_' => self.lex_ident_or_kw(start),
                b'0'..=b'9' => self.lex_number(start),
                b'"' => self.lex_string(start),
                _ => self.lex_punct(start),
            };

            if let Some(kind) = kind {
                tokens.push(Token {
                    kind,
                    span: Span::new(self.file_id, start as u32, self.pos as u32),
                });
            }
        }

        if self.diagnostics.is_empty() {
            Ok(tokens)
        } else {
            Err(self.diagnostics)
        }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn peek_at(&self, offset: usize) -> Option<u8> {
        self.src.get(self.pos + offset).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.pos += 1;
        Some(b)
    }

    fn skip_ws_and_comments(&mut self) {
        loop {
            while matches!(self.peek(), Some(b' ' | b'\t' | b'\r' | b'\n')) {
                self.pos += 1;
            }
            if self.peek() == Some(b'/') && self.peek_at(1) == Some(b'/') {
                self.pos += 2;
                while let Some(b) = self.peek() {
                    self.pos += 1;
                    if b == b'\n' {
                        break;
                    }
                }
                continue;
            }
            if self.peek() == Some(b'/') && self.peek_at(1) == Some(b'*') {
                let start = self.pos;
                self.pos += 2;
                let mut depth = 1usize;
                while depth > 0 {
                    match (self.peek(), self.peek_at(1)) {
                        (Some(b'/'), Some(b'*')) => {
                            self.pos += 2;
                            depth += 1;
                        }
                        (Some(b'*'), Some(b'/')) => {
                            self.pos += 2;
                            depth -= 1;
                        }
                        (Some(_), _) => {
                            self.pos += 1;
                        }
                        (None, _) => {
                            self.diagnostics.push(Diagnostic::error(
                                "unterminated block comment",
                                Span::new(self.file_id, start as u32, self.pos as u32),
                            ));
                            break;
                        }
                    }
                }
                continue;
            }
            break;
        }
    }

    fn lex_ident_or_kw(&mut self, start: usize) -> Option<TokenKind> {
        while matches!(
            self.peek(),
            Some(b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_')
        ) {
            self.pos += 1;
        }
        let text = std::str::from_utf8(&self.src[start..self.pos]).unwrap();
        let kind = match text {
            "as" => TokenKind::As,
            "asm" => TokenKind::Asm,
            "bank" => TokenKind::Bank,
            "bank_group" => TokenKind::BankGroup,
            "bool" => TokenKind::Bool,
            "break" => TokenKind::Break,
            "const" => TokenKind::Const,
            "continue" => TokenKind::Continue,
            "default" => TokenKind::Default,
            "di" => TokenKind::Di,
            "ei" => TokenKind::Ei,
            "else" => TokenKind::Else,
            "enum" => TokenKind::Enum,
            "extern" => TokenKind::Extern,
            "false" => TokenKind::False,
            "fn" => TokenKind::Fn,
            "for" => TokenKind::For,
            "goto" => TokenKind::Goto,
            "if" => TokenKind::If,
            "in" => TokenKind::In,
            "interrupt_controller" => TokenKind::InterruptController,
            "let" => TokenKind::Let,
            "loop" => TokenKind::Loop,
            "machine" => TokenKind::Machine,
            "match" => TokenKind::Match,
            "mod" => TokenKind::Mod,
            "mut" => TokenKind::Mut,
            "on" => TokenKind::On,
            "overlay_slot" => TokenKind::OverlaySlot,
            "pub" => TokenKind::Pub,
            "requires" => TokenKind::Requires,
            "return" => TokenKind::Return,
            "self" => TokenKind::SelfLower,
            "Self" => TokenKind::SelfUpper,
            "space" => TokenKind::Space,
            "stack" => TokenKind::Stack,
            "state" => TokenKind::State,
            "static" => TokenKind::Static,
            "struct" => TokenKind::Struct,
            "switch" => TokenKind::Switch,
            "target" => TokenKind::Target,
            "transitions" => TokenKind::Transitions,
            "true" => TokenKind::True,
            "type" => TokenKind::Type,
            "use" => TokenKind::Use,
            "while" => TokenKind::While,
            "with" => TokenKind::With,
            _ => TokenKind::Ident(text.to_string()),
        };
        Some(kind)
    }

    fn lex_number(&mut self, start: usize) -> Option<TokenKind> {
        let mut radix = 10u32;
        if self.peek() == Some(b'0') {
            match self.peek_at(1) {
                Some(b'x' | b'X') => {
                    self.pos += 2;
                    radix = 16;
                }
                Some(b'b' | b'B') => {
                    self.pos += 2;
                    radix = 2;
                }
                _ => {}
            }
        }

        let digits_start = self.pos;
        while let Some(b) = self.peek() {
            let valid = match radix {
                2 => matches!(b, b'0' | b'1' | b'_'),
                10 => matches!(b, b'0'..=b'9' | b'_'),
                16 => b.is_ascii_hexdigit() || b == b'_',
                _ => false,
            };
            if valid {
                self.pos += 1;
            } else {
                break;
            }
        }

        // Fixed-point literal: decimal digits followed by '.' and a digit (not '..').
        if radix == 10 && self.peek() == Some(b'.') && matches!(self.peek_at(1), Some(b'0'..=b'9')) {
            self.pos += 1; // consume '.'
            while matches!(self.peek(), Some(b'0'..=b'9' | b'_')) {
                self.pos += 1;
            }
            let num_end = self.pos;
            let suffix = self.lex_opt_suffix();
            let text = std::str::from_utf8(&self.src[start..num_end])
                .unwrap()
                .replace('_', "");
            return Some(TokenKind::FixedLit { text, suffix });
        }

        let digits_end = self.pos;
        let cleaned = std::str::from_utf8(&self.src[digits_start..digits_end])
            .unwrap()
            .replace('_', "");
        let suffix = self.lex_opt_suffix();

        if cleaned.is_empty() {
            self.diagnostics.push(Diagnostic::error(
                "invalid numeric literal: expected digits",
                Span::new(self.file_id, start as u32, self.pos as u32),
            ));
            return None;
        }

        match u64::from_str_radix(&cleaned, radix) {
            Ok(value) => Some(TokenKind::IntLit { value, suffix }),
            Err(_) => {
                self.diagnostics.push(Diagnostic::error(
                    "integer literal overflows u64",
                    Span::new(self.file_id, start as u32, self.pos as u32),
                ));
                None
            }
        }
    }

    fn lex_opt_suffix(&mut self) -> Option<String> {
        if matches!(self.peek(), Some(b'A'..=b'Z' | b'a'..=b'z')) {
            let s = self.pos;
            while matches!(
                self.peek(),
                Some(b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_')
            ) {
                self.pos += 1;
            }
            Some(std::str::from_utf8(&self.src[s..self.pos]).unwrap().to_string())
        } else {
            None
        }
    }

    fn lex_byte_lit(&mut self, start: usize) -> Option<TokenKind> {
        self.pos += 2; // consume b'
        let byte = match self.bump() {
            Some(b'\\') => match self.bump() {
                Some(b'n') => b'\n',
                Some(b'r') => b'\r',
                Some(b't') => b'\t',
                Some(b'0') => b'\0',
                Some(b'\\') => b'\\',
                Some(b'\'') => b'\'',
                _ => {
                    self.diagnostics.push(Diagnostic::error(
                        "invalid escape in byte literal",
                        Span::new(self.file_id, start as u32, self.pos as u32),
                    ));
                    return None;
                }
            },
            Some(b) if b != b'\'' => b,
            _ => {
                self.diagnostics.push(Diagnostic::error(
                    "empty or unterminated byte literal",
                    Span::new(self.file_id, start as u32, self.pos as u32),
                ));
                return None;
            }
        };
        if self.bump() != Some(b'\'') {
            self.diagnostics.push(Diagnostic::error(
                "unterminated byte literal",
                Span::new(self.file_id, start as u32, self.pos as u32),
            ));
            return None;
        }
        Some(TokenKind::ByteLit(byte))
    }

    fn lex_string(&mut self, start: usize) -> Option<TokenKind> {
        self.pos += 1; // consume '"'
        let mut out = String::new();
        while let Some(b) = self.bump() {
            match b {
                b'"' => return Some(TokenKind::StrLit(out)),
                b'\\' => match self.bump() {
                    Some(b'n') => out.push('\n'),
                    Some(b'r') => out.push('\r'),
                    Some(b't') => out.push('\t'),
                    Some(b'0') => out.push('\0'),
                    Some(b'\\') => out.push('\\'),
                    Some(b'"') => out.push('"'),
                    _ => {
                        self.diagnostics.push(Diagnostic::error(
                            "invalid escape sequence in string literal",
                            Span::new(self.file_id, start as u32, self.pos as u32),
                        ));
                        return None;
                    }
                },
                other => out.push(other as char),
            }
        }
        self.diagnostics.push(Diagnostic::error(
            "unterminated string literal",
            Span::new(self.file_id, start as u32, self.pos as u32),
        ));
        None
    }

    fn lex_punct(&mut self, start: usize) -> Option<TokenKind> {
        let b = self.bump()?;
        let kind = match b {
            b'+' => match self.peek() {
                Some(b'|') => {
                    self.pos += 1;
                    TokenKind::PlusPipe
                }
                Some(b'%') => {
                    self.pos += 1;
                    TokenKind::PlusPercent
                }
                Some(b'=') => {
                    self.pos += 1;
                    TokenKind::PlusEq
                }
                _ => TokenKind::Plus,
            },
            b'-' => match self.peek() {
                Some(b'|') => {
                    self.pos += 1;
                    TokenKind::MinusPipe
                }
                Some(b'%') => {
                    self.pos += 1;
                    TokenKind::MinusPercent
                }
                Some(b'=') => {
                    self.pos += 1;
                    TokenKind::MinusEq
                }
                Some(b'>') => {
                    self.pos += 1;
                    TokenKind::Arrow
                }
                _ => TokenKind::Minus,
            },
            b'*' => match self.peek() {
                Some(b'|') => {
                    self.pos += 1;
                    TokenKind::StarPipe
                }
                Some(b'%') => {
                    self.pos += 1;
                    TokenKind::StarPercent
                }
                Some(b'=') => {
                    self.pos += 1;
                    TokenKind::StarEq
                }
                _ => TokenKind::Star,
            },
            b'/' => {
                if self.peek() == Some(b'=') {
                    self.pos += 1;
                    TokenKind::SlashEq
                } else {
                    TokenKind::Slash
                }
            }
            b'%' => TokenKind::Percent,
            b'=' => match self.peek() {
                Some(b'=') => {
                    self.pos += 1;
                    TokenKind::EqEq
                }
                Some(b'>') => {
                    self.pos += 1;
                    TokenKind::FatArrow
                }
                _ => TokenKind::Eq,
            },
            b'!' => {
                if self.peek() == Some(b'=') {
                    self.pos += 1;
                    TokenKind::BangEq
                } else {
                    TokenKind::Bang
                }
            }
            b'<' => match self.peek() {
                Some(b'=') => {
                    self.pos += 1;
                    TokenKind::LtEq
                }
                Some(b'<') => {
                    self.pos += 1;
                    if self.peek() == Some(b'=') {
                        self.pos += 1;
                        TokenKind::ShlEq
                    } else {
                        TokenKind::Shl
                    }
                }
                _ => TokenKind::Lt,
            },
            b'>' => match self.peek() {
                Some(b'=') => {
                    self.pos += 1;
                    TokenKind::GtEq
                }
                Some(b'>') => {
                    self.pos += 1;
                    if self.peek() == Some(b'=') {
                        self.pos += 1;
                        TokenKind::ShrEq
                    } else {
                        TokenKind::Shr
                    }
                }
                _ => TokenKind::Gt,
            },
            b'&' => match self.peek() {
                Some(b'&') => {
                    self.pos += 1;
                    TokenKind::AmpAmp
                }
                Some(b'=') => {
                    self.pos += 1;
                    TokenKind::AmpEq
                }
                _ => TokenKind::Amp,
            },
            b'|' => match self.peek() {
                Some(b'|') => {
                    self.pos += 1;
                    TokenKind::PipePipe
                }
                Some(b'=') => {
                    self.pos += 1;
                    TokenKind::PipeEq
                }
                _ => TokenKind::Pipe,
            },
            b'^' => {
                if self.peek() == Some(b'=') {
                    self.pos += 1;
                    TokenKind::CaretEq
                } else {
                    TokenKind::Caret
                }
            }
            b'(' => TokenKind::LParen,
            b')' => TokenKind::RParen,
            b'{' => TokenKind::LBrace,
            b'}' => TokenKind::RBrace,
            b'[' => TokenKind::LBracket,
            b']' => TokenKind::RBracket,
            b',' => TokenKind::Comma,
            b';' => TokenKind::Semi,
            b':' => {
                if self.peek() == Some(b':') {
                    self.pos += 1;
                    TokenKind::ColonColon
                } else {
                    TokenKind::Colon
                }
            }
            b'.' => {
                if self.peek() == Some(b'.') {
                    self.pos += 1;
                    if self.peek() == Some(b'=') {
                        self.pos += 1;
                        TokenKind::DotDotEq
                    } else {
                        TokenKind::DotDot
                    }
                } else {
                    TokenKind::Dot
                }
            }
            b'#' => TokenKind::Hash,
            b'@' => TokenKind::At,
            other => {
                self.diagnostics.push(Diagnostic::error(
                    format!("unexpected character `{}`", other as char),
                    Span::new(self.file_id, start as u32, self.pos as u32),
                ));
                return None;
            }
        };
        Some(kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lex_operators_and_literals() {
        let src = "let x: q8_8 = 1.5q8_8 +| 0x10u8 -% 0..=10; /* nested /* comment */ */";
        let tokens = Lexer::new(0, src).tokenize().expect("lex should succeed");
        let kinds: Vec<_> = tokens.into_iter().map(|t| t.kind).collect();
        assert_eq!(
            kinds,
            vec![
                TokenKind::Let,
                TokenKind::Ident("x".into()),
                TokenKind::Colon,
                TokenKind::Ident("q8_8".into()),
                TokenKind::Eq,
                TokenKind::FixedLit {
                    text: "1.5".into(),
                    suffix: Some("q8_8".into()),
                },
                TokenKind::PlusPipe,
                TokenKind::IntLit {
                    value: 0x10,
                    suffix: Some("u8".into()),
                },
                TokenKind::MinusPercent,
                TokenKind::IntLit {
                    value: 0,
                    suffix: None,
                },
                TokenKind::DotDotEq,
                TokenKind::IntLit {
                    value: 10,
                    suffix: None,
                },
                TokenKind::Semi,
                TokenKind::Eof,
            ]
        );
    }
}
