//! Aun frontend: lexer, parser, AST, type checker, borrow checker, and IR lowering.

pub mod ast;
pub mod diag;
pub mod lexer;
pub mod parser;

pub use diag::{Diagnostic, Severity};
pub use parser::Parser;
