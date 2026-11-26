//! Aun frontend: lexer, parser, AST, type checker, borrow checker, and IR lowering.

pub mod ast;
pub mod check;
pub mod diag;
pub mod lexer;
pub mod parser;

pub use check::Checker;
pub use diag::{Diagnostic, Severity};
pub use parser::Parser;

/// Convenience helper: lex, parse, type/borrow-check, and lower a source string to `IrProgram`.
pub fn compile_str(file_id: u16, src: &str) -> Result<aun_ir::IrProgram, Vec<Diagnostic>> {
    let file = Parser::parse_str(file_id, src)?;
    Checker::check_and_lower(&file)
}
