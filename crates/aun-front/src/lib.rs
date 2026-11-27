//! Aun frontend: lexer, parser, AST, type checker, borrow checker, and IR lowering.

pub mod ast;
pub mod check;
pub mod diag;
pub mod lexer;
pub mod parser;
pub mod resolve;

pub use check::Checker;
pub use diag::{Diagnostic, Severity};
pub use parser::Parser;

/// Convenience helper: lex, parse, resolve modules, type/borrow-check, and lower a source string to `IrProgram`.
/// `#[cfg(test)]` items are stripped.
pub fn compile_str(file_id: u16, src: &str) -> Result<aun_ir::IrProgram, Vec<Diagnostic>> {
    compile_str_with_options(file_id, src, false)
}

/// Test-mode variant of [`compile_str`] that retains `#[cfg(test)]` items.
pub fn compile_str_for_test(file_id: u16, src: &str) -> Result<aun_ir::IrProgram, Vec<Diagnostic>> {
    compile_str_with_options(file_id, src, true)
}

fn compile_str_with_options(
    file_id: u16,
    src: &str,
    include_cfg_test: bool,
) -> Result<aun_ir::IrProgram, Vec<Diagnostic>> {
    let mut file = Parser::parse_str(file_id, src)?;
    file.items = filter_cfg_test(file.items, include_cfg_test);
    let resolved = resolve::resolve_file(file)?;
    Checker::check_and_lower(&resolved)
}

/// Read, recursively expand `include!("...")` and `mod foo;` directives relative to each file,
/// resolve modules, type/borrow-check, and lower a source file to `(root_source, IrProgram)`.
/// `#[cfg(test)]` items are stripped.
pub fn compile_file(path: &std::path::Path) -> Result<(String, aun_ir::IrProgram), Vec<Diagnostic>> {
    compile_file_with_options(path, false)
}

/// Test-mode variant of [`compile_file`] that retains `#[cfg(test)]` items.
pub fn compile_file_for_test(
    path: &std::path::Path,
) -> Result<(String, aun_ir::IrProgram), Vec<Diagnostic>> {
    compile_file_with_options(path, true)
}

fn compile_file_with_options(
    path: &std::path::Path,
    include_cfg_test: bool,
) -> Result<(String, aun_ir::IrProgram), Vec<Diagnostic>> {
    let root_src = std::fs::read_to_string(path).map_err(|e| {
        vec![Diagnostic::error(
            format!("failed to read `{}`: {e}", path.display()),
            aun_ir::Span::default(),
        )]
    })?;
    let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let mut stack = vec![canonical];
    let mut next_file_id = 1u16;
    let file = load_and_expand_file(
        0,
        &root_src,
        path,
        true,
        include_cfg_test,
        &mut stack,
        &mut next_file_id,
    )?;
    let resolved = resolve::resolve_file(file)?;
    let program = Checker::check_and_lower(&resolved)?;
    Ok((root_src, program))
}

/// Compile multiple `.aun` translation units into a single whole-program `IrProgram`
/// (with cross-unit WPO Uchiguri, outlining, `pub fn` exports, and private isolation).
/// `#[cfg(test)]` items are stripped.
pub fn compile_files(
    paths: &[&std::path::Path],
) -> Result<(String, aun_ir::IrProgram), Vec<Diagnostic>> {
    compile_files_with_options(paths, false)
}

/// Test-mode variant of [`compile_files`] that retains `#[cfg(test)]` items.
pub fn compile_files_for_test(
    paths: &[&std::path::Path],
) -> Result<(String, aun_ir::IrProgram), Vec<Diagnostic>> {
    compile_files_with_options(paths, true)
}

fn compile_files_with_options(
    paths: &[&std::path::Path],
    include_cfg_test: bool,
) -> Result<(String, aun_ir::IrProgram), Vec<Diagnostic>> {
    if paths.is_empty() {
        return Err(vec![Diagnostic::error(
            "no input `.aun` files provided",
            aun_ir::Span::default(),
        )]);
    }
    if paths.len() == 1 {
        return compile_file_with_options(paths[0], include_cfg_test);
    }
    let mut first_src = String::new();
    let mut next_file_id = 1u16;
    let mut units = Vec::with_capacity(paths.len());
    for (idx, &path) in paths.iter().enumerate() {
        let src = std::fs::read_to_string(path).map_err(|e| {
            vec![Diagnostic::error(
                format!("failed to read `{}`: {e}", path.display()),
                aun_ir::Span::default(),
            )]
        })?;
        let file_id = if idx == 0 {
            first_src = src.clone();
            0
        } else {
            let fid = next_file_id;
            next_file_id = next_file_id.wrapping_add(1);
            fid
        };
        let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let mut stack = vec![canonical];
        let file = load_and_expand_file(
            file_id,
            &src,
            path,
            true,
            include_cfg_test,
            &mut stack,
            &mut next_file_id,
        )?;
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unit")
            .to_string();
        units.push((stem, file));
    }
    let resolved = resolve::resolve_units(units)?;
    let program = Checker::check_and_lower(&resolved)?;
    Ok((first_src, program))
}

fn filter_cfg_test(items: Vec<ast::Item>, include_cfg_test: bool) -> Vec<ast::Item> {
    if include_cfg_test {
        return items;
    }
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        if item.is_cfg_test() {
            continue;
        }
        match item {
            ast::Item::Mod {
                attrs,
                is_pub,
                name,
                items: Some(inner),
                span,
            } => {
                out.push(ast::Item::Mod {
                    attrs,
                    is_pub,
                    name,
                    items: Some(filter_cfg_test(inner, false)),
                    span,
                });
            }
            ast::Item::ExternAsm(mut block) => {
                block.fns.retain(|ef| {
                    !ef.attrs
                        .iter()
                        .any(|a| a.name == "cfg" && a.args.iter().any(|arg| arg == "test"))
                });
                out.push(ast::Item::ExternAsm(block));
            }
            other => out.push(other),
        }
    }
    out
}

fn load_and_expand_file(
    file_id: u16,
    src: &str,
    cur_path: &std::path::Path,
    is_root: bool,
    include_cfg_test: bool,
    stack: &mut Vec<std::path::PathBuf>,
    next_file_id: &mut u16,
) -> Result<ast::File, Vec<Diagnostic>> {
    let parsed = Parser::parse_str(file_id, src)?;
    let filtered = filter_cfg_test(parsed.items, include_cfg_test);
    let expanded_items = expand_items(
        filtered,
        cur_path,
        is_root,
        include_cfg_test,
        stack,
        next_file_id,
    )?;
    Ok(ast::File {
        items: expanded_items,
    })
}

fn resolve_mod_candidates(cur_path: &std::path::Path, is_root: bool, name: &str) -> Vec<std::path::PathBuf> {
    let base_dir = cur_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    let file_name = cur_path.file_name().and_then(|s| s.to_str()).unwrap_or("");
    let stem = cur_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");

    let mut cands = Vec::new();
    if !is_root && file_name != "mod.aun" && !stem.is_empty() {
        cands.push(base_dir.join(stem).join(format!("{name}.aun")));
        cands.push(base_dir.join(stem).join(name).join("mod.aun"));
    }
    cands.push(base_dir.join(format!("{name}.aun")));
    cands.push(base_dir.join(name).join("mod.aun"));
    cands
}

fn expand_items(
    items: Vec<ast::Item>,
    cur_path: &std::path::Path,
    is_root: bool,
    include_cfg_test: bool,
    stack: &mut Vec<std::path::PathBuf>,
    next_file_id: &mut u16,
) -> Result<Vec<ast::Item>, Vec<Diagnostic>> {
    let base_dir = cur_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    let mut expanded_items = Vec::with_capacity(items.len());
    for item in items {
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
                let child_file = load_and_expand_file(
                    child_id,
                    &inc_src,
                    &inc_path,
                    is_root,
                    include_cfg_test,
                    stack,
                    next_file_id,
                )?;
                stack.pop();
                expanded_items.extend(child_file.items);
            }
            ast::Item::Mod {
                attrs,
                is_pub,
                name,
                items: None,
                span,
            } => {
                let cands = resolve_mod_candidates(cur_path, is_root, &name);
                let Some(mod_path) = cands.iter().find(|p| p.is_file()).cloned() else {
                    let tried: Vec<String> = cands.iter().map(|p| p.display().to_string()).collect();
                    return Err(vec![Diagnostic::error(
                        format!("file not found for module `{name}` (tried: {})", tried.join(", ")),
                        span,
                    )]);
                };
                let canon = std::fs::canonicalize(&mod_path).unwrap_or_else(|_| mod_path.clone());
                if stack.contains(&canon) {
                    return Err(vec![Diagnostic::error(
                        format!("cyclic module dependency detected: `{}`", mod_path.display()),
                        span,
                    )]);
                }
                let mod_src = std::fs::read_to_string(&mod_path).map_err(|e| {
                    vec![Diagnostic::error(
                        format!("failed to read module `{}`: {e}", mod_path.display()),
                        span,
                    )]
                })?;
                let child_id = *next_file_id;
                *next_file_id = next_file_id.wrapping_add(1);
                stack.push(canon);
                let child_file = load_and_expand_file(
                    child_id,
                    &mod_src,
                    &mod_path,
                    false,
                    include_cfg_test,
                    stack,
                    next_file_id,
                )?;
                stack.pop();
                expanded_items.push(ast::Item::Mod {
                    attrs,
                    is_pub,
                    name,
                    items: Some(child_file.items),
                    span,
                });
            }
            ast::Item::Mod {
                attrs,
                is_pub,
                name,
                items: Some(inner),
                span,
            } => {
                let inner_expanded = expand_items(
                    inner,
                    cur_path,
                    false,
                    include_cfg_test,
                    stack,
                    next_file_id,
                )?;
                expanded_items.push(ast::Item::Mod {
                    attrs,
                    is_pub,
                    name,
                    items: Some(inner_expanded),
                    span,
                });
            }
            other => expanded_items.push(other),
        }
    }
    Ok(expanded_items)
}
