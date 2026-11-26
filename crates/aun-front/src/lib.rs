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

/// Read, recursively expand `include!("...")` directives relative to each file,
/// type/borrow-check, and lower a source file to `(root_source, IrProgram)`.
pub fn compile_file(path: &std::path::Path) -> Result<(String, aun_ir::IrProgram), Vec<Diagnostic>> {
    let root_src = std::fs::read_to_string(path).map_err(|e| {
        vec![Diagnostic::error(
            format!("failed to read `{}`: {e}", path.display()),
            aun_ir::Span::default(),
        )]
    })?;
    let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let mut stack = vec![canonical];
    let mut next_file_id = 1u16;
    let file = load_and_expand_file(0, &root_src, path, &mut stack, &mut next_file_id)?;
    let program = Checker::check_and_lower(&file)?;
    Ok((root_src, program))
}

fn load_and_expand_file(
    file_id: u16,
    src: &str,
    cur_path: &std::path::Path,
    stack: &mut Vec<std::path::PathBuf>,
    next_file_id: &mut u16,
) -> Result<ast::File, Vec<Diagnostic>> {
    let parsed = Parser::parse_str(file_id, src)?;
    let base_dir = cur_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    let mut expanded_items = Vec::with_capacity(parsed.items.len());
    for item in parsed.items {
        match item {
            ast::Item::Include { path: rel, span } => {
                let inc_path = base_dir.join(&rel);
                let canon = std::fs::canonicalize(&inc_path).unwrap_or_else(|_| inc_path.clone());
                if stack.contains(&canon) {
                    return Err(vec![Diagnostic::error(
                        format!("cyclic include detected: `{}`", inc_path.display()),
                        span,
                    )]);
                }
                let inc_src = std::fs::read_to_string(&inc_path).map_err(|e| {
                    vec![Diagnostic::error(
                        format!("failed to include `{}`: {e}", inc_path.display()),
                        span,
                    )]
                })?;
                let child_id = *next_file_id;
                *next_file_id = next_file_id.wrapping_add(1);
                stack.push(canon);
                let child_file =
                    load_and_expand_file(child_id, &inc_src, &inc_path, stack, next_file_id)?;
                stack.pop();
                expanded_items.extend(child_file.items);
            }
            other => expanded_items.push(other),
        }
    }
    Ok(ast::File {
        items: expanded_items,
    })
}
