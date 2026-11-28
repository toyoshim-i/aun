//! `aun` command-line driver (`aun build`, `aun test`, `aun inspect`, `aun guide`, `aun explain`, `aun mcp`).

mod docs;

use agent_man::RenderMode;
use aun_front::Diagnostic;
use aun_interp::{Interpreter, TestOutcome};
use aun_ir::IrProgram;
use aun_z80::LinkOutput;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn print_usage() {
    eprintln!(
        "aun {} — Aun language compiler and Z80 toolchain\n\n\
         Usage:\n  \
           aun build [-o <dir>] [--release] [--checks=<list>] [--format bin|sdasz80] [--asm <file.s|dir>] [--startup-asm <file.s>] [-I <dir>] [--code-origin <addr>] [--ram-base <addr|auto>] [--overlay-load-align <bytes>] [--symbol-prefix <prefix>] <file.aun|file.s|aun.toml> ...\n  \
           aun test [--strict-externs] [--trace] <file.aun|aun.toml> ...\n  \
           aun inspect [--tstates <fn>] <file.aun>\n  \
           aun guide [topic] [--agent|--terminal|--json]\n  \
           aun explain <code> [--agent|--terminal|--json]\n  \
           aun mcp\n  \
           aun --version",
        env!("CARGO_PKG_VERSION")
    );
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RamBaseConfig {
    Fixed(u16),
    Auto,
}

fn parse_ram_base_config(raw: &str) -> Option<RamBaseConfig> {
    let s = raw.trim().trim_matches('"').trim_matches('\'');
    if s.eq_ignore_ascii_case("auto") {
        Some(RamBaseConfig::Auto)
    } else {
        parse_u16_addr(s).map(RamBaseConfig::Fixed)
    }
}

fn parse_overlay_align(raw: &str) -> Option<u16> {
    let s = raw.trim().trim_matches('"').trim_matches('\'');
    let v = parse_u16_addr(s)?;
    (v >= 1 && v.is_power_of_two()).then_some(v)
}

#[derive(Clone)]
struct AsmInput {
    path: String,
    owner_state: Option<String>,
    is_startup: bool,
}

fn parse_u16_addr(raw: &str) -> Option<u16> {
    let s = raw.trim().trim_matches('"');
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u16::from_str_radix(hex, 16).ok()
    } else {
        s.parse::<u16>().ok()
    }
}

fn collect_asm_arg(
    raw: &str,
    asm_specs: &mut Vec<AsmInput>,
) -> Result<(), String> {
    if let Some((state, path_str)) = raw.split_once('=')
        && !state.trim().is_empty()
        && !path_str.trim().is_empty()
    {
        return collect_asm_with_owner(
            Path::new(path_str.trim()),
            Some(state.trim().to_string()),
            false,
            asm_specs,
        );
    }
    collect_asm_with_owner(Path::new(raw), None, false, asm_specs)
}

fn collect_asm_with_owner(
    path: &Path,
    owner_state: Option<String>,
    is_startup: bool,
    asm_specs: &mut Vec<AsmInput>,
) -> Result<(), String> {
    if path.is_dir() {
        let mut files = Vec::new();
        let entries = fs::read_dir(path)
            .map_err(|e| format!("failed to read asm directory `{}`: {e}", path.display()))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("failed to read directory entry: {e}"))?;
            let p = entry.path();
            if p.is_file()
                && p.extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "s" | "asm"))
            {
                files.push(p);
            }
        }
        files.sort();
        for p in files {
            asm_specs.push(AsmInput {
                path: p.to_string_lossy().into_owned(),
                owner_state: owner_state.clone(),
                is_startup,
            });
        }
    } else {
        asm_specs.push(AsmInput {
            path: path.to_string_lossy().into_owned(),
            owner_state,
            is_startup,
        });
    }
    Ok(())
}

fn strip_toml_comment(line: &str) -> &str {
    let mut in_quotes = false;
    let mut escaped = false;
    for (idx, ch) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match ch {
            '\\' if in_quotes => escaped = true,
            '"' => in_quotes = !in_quotes,
            '#' if !in_quotes => return line[..idx].trim(),
            _ => {}
        }
    }
    line.trim()
}

fn extract_toml_string_list(rhs: &str) -> Vec<String> {
    let inner = rhs.trim().trim_start_matches('[').trim_end_matches(']');
    let mut out = Vec::new();
    let mut chars = inner.chars();
    while let Some(ch) = chars.next() {
        if ch == '"' {
            let mut s = String::new();
            let mut esc = false;
            for c in chars.by_ref() {
                if esc {
                    s.push(c);
                    esc = false;
                } else if c == '\\' {
                    esc = true;
                } else if c == '"' {
                    break;
                } else {
                    s.push(c);
                }
            }
            if !s.is_empty() {
                out.push(s);
            }
        }
    }
    out
}

fn extract_toml_string(rhs: &str) -> Option<String> {
    let t = rhs.trim();
    let quote = if t.starts_with('"') && t.ends_with('"') && t.len() >= 2 {
        '"'
    } else if t.starts_with('\'') && t.ends_with('\'') && t.len() >= 2 {
        '\''
    } else {
        return None;
    };
    let inner = &t[1..t.len() - 1];
    let mut out = String::with_capacity(inner.len());
    let mut esc = false;
    for ch in inner.chars() {
        if esc {
            out.push(ch);
            esc = false;
        } else if quote == '"' && ch == '\\' {
            esc = true;
        } else {
            out.push(ch);
        }
    }
    Some(out)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TomlSection {
    Root,
    OverlayAsm,
    Test,
    Other,
}

#[allow(clippy::too_many_arguments)]
fn parse_aun_toml(
    manifest_path: &Path,
    source_paths: &mut Vec<String>,
    mut test_sources: Option<&mut Vec<String>>,
    asm_paths: &mut Vec<AsmInput>,
    asm_include_dirs: &mut Vec<PathBuf>,
    code_origin: &mut Option<u16>,
    ram_base: &mut Option<RamBaseConfig>,
    overlay_load_align: &mut Option<u16>,
    symbol_prefix: &mut Option<String>,
) -> Result<(), String> {
    let text = fs::read_to_string(manifest_path)
        .map_err(|e| format!("failed to read `{}`: {e}", manifest_path.display()))?;
    let base_dir = manifest_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));

    // Join multi-line `[ ... ]` key-value assignments into logical lines
    let mut statements = Vec::new();
    let mut pending = String::new();
    let mut bracket_depth = 0usize;
    for raw in text.lines() {
        let stripped = strip_toml_comment(raw);
        if stripped.is_empty() {
            continue;
        }
        if !pending.is_empty() {
            pending.push(' ');
        }
        pending.push_str(stripped);
        for ch in stripped.chars() {
            match ch {
                '[' => bracket_depth += 1,
                ']' => bracket_depth = bracket_depth.saturating_sub(1),
                _ => {}
            }
        }
        if bracket_depth == 0 {
            statements.push(std::mem::take(&mut pending));
        }
    }
    if !pending.is_empty() {
        statements.push(pending);
    }

    let mut section = TomlSection::Root;
    for stmt in statements {
        let trimmed = stmt.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') && !trimmed.contains('=') {
            section = match trimmed {
                "[overlay_asm]" => TomlSection::OverlayAsm,
                "[test]" => TomlSection::Test,
                _ => TomlSection::Other,
            };
            continue;
        }
        let Some((lhs, rhs)) = trimmed.split_once('=') else {
            continue;
        };
        let key = lhs.trim().trim_matches('"').trim_matches('\'');
        let val = rhs.trim();
        match section {
            TomlSection::OverlayAsm => {
                for rel in extract_toml_string_list(val) {
                    collect_asm_with_owner(&base_dir.join(rel), Some(key.to_string()), false, asm_paths)?;
                }
            }
            TomlSection::Test => {
                if key == "sources"
                    && let Some(ts) = test_sources.as_deref_mut()
                {
                    for rel in extract_toml_string_list(val) {
                        ts.push(base_dir.join(rel).to_string_lossy().into_owned());
                    }
                }
            }
            TomlSection::Other => {}
            TomlSection::Root => match key {
                "entry" => {
                    if let Some(rel) = extract_toml_string_list(val).into_iter().next() {
                        source_paths.push(base_dir.join(rel).to_string_lossy().into_owned());
                    }
                }
                "sources" => {
                    for rel in extract_toml_string_list(val) {
                        source_paths.push(base_dir.join(rel).to_string_lossy().into_owned());
                    }
                }
                "test_sources" => {
                    if let Some(ts) = test_sources.as_deref_mut() {
                        for rel in extract_toml_string_list(val) {
                            ts.push(base_dir.join(rel).to_string_lossy().into_owned());
                        }
                    }
                }
                "asm_sources" => {
                    for rel in extract_toml_string_list(val) {
                        collect_asm_with_owner(&base_dir.join(rel), None, false, asm_paths)?;
                    }
                }
                "startup_asm" | "crt0" => {
                    for rel in extract_toml_string_list(val) {
                        collect_asm_with_owner(&base_dir.join(rel), None, true, asm_paths)?;
                    }
                }
                "asm_include_dirs" => {
                    for rel in extract_toml_string_list(val) {
                        asm_include_dirs.push(base_dir.join(rel));
                    }
                }
                "code_origin" => {
                    let parsed = parse_u16_addr(val).ok_or_else(|| {
                        format!(
                            "invalid `code_origin` in `{}`: `{val}`",
                            manifest_path.display()
                        )
                    })?;
                    *code_origin = Some(parsed);
                }
                "ram_base" => {
                    let parsed = parse_ram_base_config(val).ok_or_else(|| {
                        format!(
                            "invalid `ram_base` in `{}`: `{val}` (expected address or \"auto\")",
                            manifest_path.display()
                        )
                    })?;
                    *ram_base = Some(parsed);
                }
                "overlay_load_align" | "load_align" | "sector_size" => {
                    let parsed = parse_overlay_align(val).ok_or_else(|| {
                        format!(
                            "invalid `{key}` in `{}`: `{val}` (expected power-of-two byte size)",
                            manifest_path.display()
                        )
                    })?;
                    *overlay_load_align = Some(parsed);
                }
                "symbol_prefix" => {
                    let parsed = extract_toml_string(val).ok_or_else(|| {
                        format!(
                            "invalid `symbol_prefix` in `{}`: `{val}`",
                            manifest_path.display()
                        )
                    })?;
                    *symbol_prefix = Some(parsed);
                }
                _ => {}
            },
        }
    }
    Ok(())
}

fn resolve_asm_include(base_dir: &Path, include_dirs: &[PathBuf], rel: &str) -> PathBuf {
    let direct = base_dir.join(rel);
    if direct.is_file() {
        return direct;
    }
    for dir in include_dirs {
        let candidate = dir.join(rel);
        if candidate.is_file() {
            return candidate;
        }
    }
    direct
}

fn load_external_asm_recursive(
    path: &Path,
    include_dirs: &[PathBuf],
    stack: &mut Vec<PathBuf>,
) -> Result<String, String> {
    let canon = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    if stack.contains(&canon) {
        return Err(format!("cyclic `.include` in assembly: `{}`", path.display()));
    }
    let src = fs::read_to_string(path)
        .map_err(|e| format!("failed to read assembly `{}`: {e}", path.display()))?;
    let base_dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));

    stack.push(canon);
    let mut out = String::with_capacity(src.len());
    for raw in src.lines() {
        let code = raw.split(';').next().unwrap_or("").trim();
        let mut parts = code.splitn(2, |c: char| c.is_whitespace());
        let head = parts.next().unwrap_or("");
        let tail = parts.next().unwrap_or("").trim();
        if head.eq_ignore_ascii_case(".include") || head.eq_ignore_ascii_case("include") {
            let rel = tail.trim().trim_matches(|c| c == '"' || c == '\'');
            if rel.is_empty() {
                stack.pop();
                return Err(format!("invalid `.include` directive in `{}`: `{raw}`", path.display()));
            }
            let inc_path = resolve_asm_include(base_dir, include_dirs, rel);
            let inc_text = load_external_asm_recursive(&inc_path, include_dirs, stack)?;
            out.push_str(&inc_text);
            if !inc_text.ends_with('\n') {
                out.push('\n');
            }
        } else {
            out.push_str(raw);
            out.push('\n');
        }
    }
    stack.pop();
    Ok(out)
}

fn load_external_asm(path: &Path, include_dirs: &[PathBuf]) -> Result<String, String> {
    let mut stack = Vec::new();
    load_external_asm_recursive(path, include_dirs, &mut stack)
}

fn print_diagnostics(diags: Vec<Diagnostic>) {
    let engine = docs::create_doc_engine();
    for d in diags {
        eprintln!("{d}");
        if let Some(hint) = docs::diagnostic_hint(&engine, &d.message, RenderMode::Agent) {
            eprintln!("  {hint}");
        }
    }
}

fn compile_file(path: &str) -> Result<(String, IrProgram), ExitCode> {
    match aun_front::compile_file(Path::new(path)) {
        Ok(res) => Ok(res),
        Err(diags) => {
            print_diagnostics(diags);
            Err(ExitCode::FAILURE)
        }
    }
}

fn compile_files(paths: &[String]) -> Result<(String, IrProgram), ExitCode> {
    let path_refs: Vec<&Path> = paths.iter().map(Path::new).collect();
    match aun_front::compile_files(&path_refs) {
        Ok(res) => Ok(res),
        Err(diags) => {
            print_diagnostics(diags);
            Err(ExitCode::FAILURE)
        }
    }
}

fn compile_files_for_test(paths: &[String]) -> Result<(String, IrProgram), ExitCode> {
    let path_refs: Vec<&Path> = paths.iter().map(Path::new).collect();
    match aun_front::compile_files_for_test(&path_refs) {
        Ok(res) => Ok(res),
        Err(diags) => {
            print_diagnostics(diags);
            Err(ExitCode::FAILURE)
        }
    }
}

/// Converts a byte offset in `src` to 1-based `(line, col)`.
fn offset_to_line_col(src: &str, byte_offset: u32) -> (usize, usize) {
    let clamped = (byte_offset as usize).min(src.len());
    let prefix = &src[..clamped];
    let line = prefix.bytes().filter(|&b| b == b'\n').count() + 1;
    let col = prefix
        .rfind('\n')
        .map(|nl| clamped - nl)
        .unwrap_or(clamped + 1);
    (line, col)
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

fn format_sym_file(path: &str, src: &str, program: &IrProgram, out: &LinkOutput) -> String {
    let mut buf = String::new();
    let _ = writeln!(buf, "; Aun symbol table for {path}");
    let _ = writeln!(buf, "; Code symbols");
    let mut syms: Vec<(&String, &u16)> = out.symbols.iter().collect();
    syms.sort_by(|a, b| a.1.cmp(b.1).then_with(|| a.0.cmp(b.0)));
    for (name, addr) in syms {
        let _ = writeln!(buf, "0x{addr:04X} {name}");
    }

    if !out.globals.is_empty() {
        let _ = writeln!(buf, "\n; Globals (RAM)");
        for g in &out.globals {
            let _ = writeln!(buf, "0x{:04X} {} ; {} bytes", g.addr, g.name, g.size);
        }
    }

    if !program.sites.is_empty() {
        let _ = writeln!(buf, "\n; Panic sites");
        for (idx, span) in program.sites.iter().enumerate() {
            let (line, col) = offset_to_line_col(src, span.start);
            let _ = writeln!(
                buf,
                "site {idx} {path}:{line}:{col} span={}..{}",
                span.start, span.end
            );
        }
    }

    buf
}

fn format_manifest_json(path: &str, src: &str, program: &IrProgram, out: &LinkOutput) -> String {
    let mut buf = String::new();
    let _ = writeln!(buf, "{{");
    let _ = writeln!(buf, "  \"source\": \"{}\",", json_escape(path));
    let _ = writeln!(buf, "  \"code_origin\": \"0x{:04X}\",", out.code_origin);
    let _ = writeln!(buf, "  \"binary_size\": {},", out.resident_bin.len());
    let _ = writeln!(buf, "  \"max_stack_bytes\": {},", out.max_stack_bytes);
    let _ = writeln!(buf, "  \"total_tstates\": {},", out.total_tstates);

    // Uchiguri RAM section
    let _ = writeln!(buf, "  \"uchiguri_ram\": {{");
    let _ = writeln!(buf, "    \"base\": \"0x{:04X}\",", out.ram_base);
    let _ = writeln!(buf, "    \"overlay_base\": \"0x{:04X}\",", out.overlay_base);
    let _ = writeln!(buf, "    \"end\": \"0x{:04X}\",", out.ram_end);
    let _ = writeln!(buf, "    \"bytes\": {},", out.ram_bytes);

    let _ = writeln!(buf, "    \"globals\": [");
    for (i, g) in out.globals.iter().enumerate() {
        let comma = if i + 1 < out.globals.len() { "," } else { "" };
        let _ = writeln!(
            buf,
            "      {{ \"name\": \"{}\", \"addr\": \"0x{:04X}\", \"size\": {} }}{comma}",
            json_escape(&g.name),
            g.addr,
            g.size
        );
    }
    let _ = writeln!(buf, "    ],");

    let _ = writeln!(buf, "    \"functions\": [");
    for (i, f) in out.fn_frames.iter().enumerate() {
        let comma = if i + 1 < out.fn_frames.len() { "," } else { "" };
        let _ = writeln!(
            buf,
            "      {{ \"name\": \"{}\", \"locals_base\": \"0x{:04X}\", \"locals_size\": {}, \"scratch_base\": \"0x{:04X}\", \"scratch_size\": {}, \"stack_depth\": {} }}{comma}",
            json_escape(&f.name),
            f.locals_base,
            f.locals_size,
            f.scratch_base,
            f.scratch_size,
            f.stack_depth
        );
    }
    let _ = writeln!(buf, "    ]");
    let _ = writeln!(buf, "  }},");

    // Function T-states
    let mut fn_ts: Vec<(&String, &u32)> = out.fn_tstates.iter().collect();
    fn_ts.sort_by(|a, b| a.0.cmp(b.0));
    let _ = writeln!(buf, "  \"function_tstates\": {{");
    for (i, (name, ts)) in fn_ts.iter().enumerate() {
        let comma = if i + 1 < fn_ts.len() { "," } else { "" };
        let _ = writeln!(buf, "    \"{}\": {ts}{comma}", json_escape(name));
    }
    let _ = writeln!(buf, "  }},");

    // Panic sites
    let _ = writeln!(buf, "  \"panic_sites\": [");
    for (i, span) in program.sites.iter().enumerate() {
        let (line, col) = offset_to_line_col(src, span.start);
        let comma = if i + 1 < program.sites.len() { "," } else { "" };
        let _ = writeln!(
            buf,
            "    {{ \"site_id\": {i}, \"file\": \"{}\", \"line\": {line}, \"col\": {col}, \"start\": {}, \"end\": {} }}{comma}",
            json_escape(path),
            span.start,
            span.end
        );
    }
    let _ = writeln!(buf, "  ],");

    // Memory spaces and switchable banks
    let mut space_items = Vec::new();
    if let Some(ref td) = program.target {
        for (idx, (sp_name, sp_start, sp_end, is_bank)) in td.all_space_names().into_iter().enumerate() {
            let kind = if is_bank {
                "bank".to_string()
            } else {
                td.spaces
                    .iter()
                    .find(|s| s.name == sp_name)
                    .map(|s| format!("{:?}", s.kind).to_ascii_lowercase())
                    .unwrap_or_else(|| "flat".to_string())
            };
            space_items.push(format!(
                "    {{ \"id\": {idx}, \"name\": \"{}\", \"kind\": \"{kind}\", \"start\": \"0x{sp_start:04X}\", \"end\": \"0x{sp_end:04X}\" }}",
                json_escape(&sp_name)
            ));
        }
    }
    let _ = writeln!(buf, "  \"spaces\": [");
    if !space_items.is_empty() {
        let _ = writeln!(buf, "{}", space_items.join(",\n"));
    }
    let _ = writeln!(buf, "  ],");

    // Overlay / HSM code+const blobs
    let mut blob_items = Vec::new();
    for ovl in &out.overlays {
        let san = ovl.state_name.replace("::", "__").to_ascii_lowercase();
        let file = format!("ovl_{san}.bin");
        let parent_json = match &ovl.parent_state {
            Some(p) => format!("\"{}\"", json_escape(p)),
            None => "null".to_string(),
        };
        let asm_list: Vec<String> = ovl
            .asm_files
            .iter()
            .map(|f| format!("\"{}\"", json_escape(f)))
            .collect();
        let aligned_end: u32 = (ovl.origin as u32) + (ovl.aligned_size as u32);
        blob_items.push(format!(
            "    {{ \"id\": {}, \"state\": \"{}\", \"parent_state\": {parent_json}, \"slot\": \"{}\", \"space\": \"{}\", \"space_id\": {}, \"origin\": \"0x{:04X}\", \"size\": {}, \"aligned_size\": {}, \"aligned_end\": \"0x{:04X}\", \"budget_end\": \"0x{:04X}\", \"file\": \"{}\", \"asm_files\": [{}] }}",
            ovl.id,
            json_escape(&ovl.state_name),
            json_escape(&ovl.slot_name),
            json_escape(&ovl.space_name),
            ovl.space_id,
            ovl.origin,
            ovl.bin.len(),
            ovl.aligned_size,
            aligned_end,
            ovl.budget_end,
            json_escape(&file),
            asm_list.join(", ")
        ));
    }
    let _ = writeln!(buf, "  \"blobs\": [");
    if !blob_items.is_empty() {
        let _ = writeln!(buf, "{}", blob_items.join(",\n"));
    }
    let _ = writeln!(buf, "  ],");

    // Hierarchical state machine state trees
    let mut state_items = Vec::new();
    for mdef in &program.machines {
        for st in &mdef.states {
            let parent_json = match st.parent {
                Some(p) => p.to_string(),
                None => "null".to_string(),
            };
            let space_json = match &st.code_space {
                Some(sp) => format!("\"{}\"", json_escape(sp)),
                None => "null".to_string(),
            };
            let blob_json = out
                .overlays
                .iter()
                .find(|o| o.state_name == st.name)
                .map(|o| o.id.to_string())
                .unwrap_or_else(|| "null".to_string());
            state_items.push(format!(
                "    {{ \"machine\": \"{}\", \"id\": {}, \"name\": \"{}\", \"parent\": {parent_json}, \"is_leaf\": {}, \"swap_boundary\": {}, \"space\": {space_json}, \"blob_id\": {blob_json} }}",
                json_escape(&mdef.name),
                st.id,
                json_escape(&st.name),
                st.is_leaf,
                st.swap_boundary || st.overlay_slot.is_some()
            ));
        }
    }
    let _ = writeln!(buf, "  \"states\": [");
    if !state_items.is_empty() {
        let _ = writeln!(buf, "{}", state_items.join(",\n"));
    }
    let _ = writeln!(buf, "  ],");

    // State-scoped bank/RAM buffers
    let mut buf_items = Vec::new();
    for g in &out.globals {
        if let Some(ref owner) = g.owner_state {
            let sp_json = match &g.space {
                Some(sp) => format!("\"{}\"", json_escape(sp)),
                None => "null".to_string(),
            };
            buf_items.push(format!(
                "    {{ \"name\": \"{}\", \"owner_state\": \"{}\", \"space\": {sp_json}, \"addr\": \"0x{:04X}\", \"size\": {} }}",
                json_escape(&g.name),
                json_escape(owner),
                g.addr,
                g.size
            ));
        }
    }
    let _ = writeln!(buf, "  \"state_buffers\": [");
    if !buf_items.is_empty() {
        let _ = writeln!(buf, "{}", buf_items.join(",\n"));
    }
    let _ = writeln!(buf, "  ]");
    let _ = writeln!(buf, "}}");
    buf
}

fn format_layout_inc(equates: &[(String, u16)]) -> String {
    use std::fmt::Write;
    let mut out = String::from("; Generated by aun build (#[export_asm_layout])\n");
    for (name, val) in equates {
        let _ = writeln!(out, "{name} = 0x{val:04X}");
    }
    out
}

fn cmd_build(args: &[String]) -> ExitCode {
    let mut out_dir = PathBuf::from("target/aun");
    let mut source_paths: Vec<String> = Vec::new();
    let mut asm_paths: Vec<AsmInput> = Vec::new();
    let mut asm_include_dirs: Vec<PathBuf> = Vec::new();
    let mut code_origin_override: Option<u16> = None;
    let mut ram_base_override: Option<RamBaseConfig> = None;
    let mut overlay_align_override: Option<u16> = None;
    let mut cli_symbol_prefix: Option<String> = None;
    let mut toml_symbol_prefix: Option<String> = None;
    let mut release = false;
    let mut explicit_checks: Option<String> = None;
    let mut sdasz80_format = false;
    let mut memory_report = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--out-dir" => {
                let Some(dir) = args.get(i + 1) else {
                    eprintln!("error: `{}` requires a directory argument", args[i]);
                    return ExitCode::FAILURE;
                };
                out_dir = PathBuf::from(dir);
                i += 2;
            }
            a if a.starts_with("--out-dir=") => {
                out_dir = PathBuf::from(&a["--out-dir=".len()..]);
                i += 1;
            }
            a if a.starts_with("-o=") => {
                out_dir = PathBuf::from(&a["-o=".len()..]);
                i += 1;
            }
            "-I" => {
                let Some(dir) = args.get(i + 1) else {
                    eprintln!("error: `-I` requires a directory argument");
                    return ExitCode::FAILURE;
                };
                asm_include_dirs.push(PathBuf::from(dir));
                i += 2;
            }
            a if a.starts_with("-I=") => {
                let dir = &a["-I=".len()..];
                if dir.is_empty() {
                    eprintln!("error: `-I=` requires a directory argument");
                    return ExitCode::FAILURE;
                }
                asm_include_dirs.push(PathBuf::from(dir));
                i += 1;
            }
            a if a.starts_with("-I") => {
                asm_include_dirs.push(PathBuf::from(&a[2..]));
                i += 1;
            }
            "--asm" => {
                let Some(asm_arg) = args.get(i + 1) else {
                    eprintln!("error: `--asm` requires a `.s` file or directory path");
                    return ExitCode::FAILURE;
                };
                if let Err(e) = collect_asm_arg(asm_arg, &mut asm_paths) {
                    eprintln!("error: {e}");
                    return ExitCode::FAILURE;
                }
                i += 2;
            }
            a if a.starts_with("--asm=") => {
                if let Err(e) = collect_asm_arg(&a["--asm=".len()..], &mut asm_paths) {
                    eprintln!("error: {e}");
                    return ExitCode::FAILURE;
                }
                i += 1;
            }
            "--startup-asm" => {
                let Some(asm_arg) = args.get(i + 1) else {
                    eprintln!("error: `--startup-asm` requires a `.s` file or directory path");
                    return ExitCode::FAILURE;
                };
                if let Err(e) = collect_asm_with_owner(Path::new(asm_arg), None, true, &mut asm_paths) {
                    eprintln!("error: {e}");
                    return ExitCode::FAILURE;
                }
                i += 2;
            }
            a if a.starts_with("--startup-asm=") => {
                if let Err(e) = collect_asm_with_owner(
                    Path::new(&a["--startup-asm=".len()..]),
                    None,
                    true,
                    &mut asm_paths,
                ) {
                    eprintln!("error: {e}");
                    return ExitCode::FAILURE;
                }
                i += 1;
            }
            "--code-origin" => {
                let Some(v) = args.get(i + 1) else {
                    eprintln!("error: `--code-origin` requires an address");
                    return ExitCode::FAILURE;
                };
                let Some(addr) = parse_u16_addr(v) else {
                    eprintln!("error: invalid `--code-origin` address `{v}`");
                    return ExitCode::FAILURE;
                };
                code_origin_override = Some(addr);
                i += 2;
            }
            a if a.starts_with("--code-origin=") => {
                let v = &a["--code-origin=".len()..];
                let Some(addr) = parse_u16_addr(v) else {
                    eprintln!("error: invalid `--code-origin` address `{v}`");
                    return ExitCode::FAILURE;
                };
                code_origin_override = Some(addr);
                i += 1;
            }
            "--ram-base" => {
                let Some(v) = args.get(i + 1) else {
                    eprintln!("error: `--ram-base` requires an address or `auto`");
                    return ExitCode::FAILURE;
                };
                let Some(cfg) = parse_ram_base_config(v) else {
                    eprintln!("error: invalid `--ram-base` value `{v}` (expected address or `auto`)");
                    return ExitCode::FAILURE;
                };
                ram_base_override = Some(cfg);
                i += 2;
            }
            a if a.starts_with("--ram-base=") => {
                let v = &a["--ram-base=".len()..];
                let Some(cfg) = parse_ram_base_config(v) else {
                    eprintln!("error: invalid `--ram-base` value `{v}` (expected address or `auto`)");
                    return ExitCode::FAILURE;
                };
                ram_base_override = Some(cfg);
                i += 1;
            }
            "--overlay-load-align" => {
                let Some(v) = args.get(i + 1) else {
                    eprintln!("error: `--overlay-load-align` requires a byte alignment");
                    return ExitCode::FAILURE;
                };
                let Some(align) = parse_overlay_align(v) else {
                    eprintln!("error: invalid `--overlay-load-align` `{v}` (expected power-of-two byte size)");
                    return ExitCode::FAILURE;
                };
                overlay_align_override = Some(align);
                i += 2;
            }
            a if a.starts_with("--overlay-load-align=") => {
                let v = &a["--overlay-load-align=".len()..];
                let Some(align) = parse_overlay_align(v) else {
                    eprintln!("error: invalid `--overlay-load-align` `{v}` (expected power-of-two byte size)");
                    return ExitCode::FAILURE;
                };
                overlay_align_override = Some(align);
                i += 1;
            }
            "--symbol-prefix" => {
                let Some(pfx) = args.get(i + 1) else {
                    eprintln!("error: `--symbol-prefix` requires a prefix string");
                    return ExitCode::FAILURE;
                };
                cli_symbol_prefix = Some(pfx.clone());
                i += 2;
            }
            a if a.starts_with("--symbol-prefix=") => {
                cli_symbol_prefix = Some(a["--symbol-prefix=".len()..].to_string());
                i += 1;
            }
            "--release" => {
                release = true;
                i += 1;
            }
            "--format" => {
                let Some(fmt) = args.get(i + 1) else {
                    eprintln!("error: `--format` requires a format name (`bin` or `sdasz80`)");
                    return ExitCode::FAILURE;
                };
                match fmt.as_str() {
                    "sdasz80" => sdasz80_format = true,
                    "bin" => sdasz80_format = false,
                    other => {
                        eprintln!("error: unknown format `{other}` (expected `bin` or `sdasz80`)");
                        return ExitCode::FAILURE;
                    }
                }
                i += 2;
            }
            a if a.starts_with("--format=") => {
                match &a["--format=".len()..] {
                    "sdasz80" => sdasz80_format = true,
                    "bin" => sdasz80_format = false,
                    other => {
                        eprintln!("error: unknown format `{other}` (expected `bin` or `sdasz80`)");
                        return ExitCode::FAILURE;
                    }
                }
                i += 1;
            }
            "--memory-report" => {
                memory_report = true;
                i += 1;
            }
            a if a.starts_with("--checks=") => {
                explicit_checks = Some(a["--checks=".len()..].to_string());
                i += 1;
            }
            a if a.starts_with('-') => {
                eprintln!("error: unknown option `{a}` for `aun build`");
                return ExitCode::FAILURE;
            }
            file => {
                let p = Path::new(file);
                match p.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()).as_deref() {
                    Some("s" | "asm") => asm_paths.push(AsmInput {
                        path: file.to_string(),
                        owner_state: None,
                        is_startup: false,
                    }),
                    Some("toml") => {
                        if let Err(e) = parse_aun_toml(
                            p,
                            &mut source_paths,
                            None,
                            &mut asm_paths,
                            &mut asm_include_dirs,
                            &mut code_origin_override,
                            &mut ram_base_override,
                            &mut overlay_align_override,
                            &mut toml_symbol_prefix,
                        ) {
                            eprintln!("error: {e}");
                            return ExitCode::FAILURE;
                        }
                    }
                    _ => source_paths.push(file.to_string()),
                }
                i += 1;
            }
        }
    }

    let mut opts = if release {
        aun_z80::CompileOptions::release()
    } else {
        aun_z80::CompileOptions::debug()
    };
    if let Some(addr) = code_origin_override {
        opts.code_origin = addr;
    }
    match ram_base_override {
        Some(RamBaseConfig::Fixed(addr)) => {
            opts.ram_base = addr;
            opts.auto_ram_base = false;
        }
        Some(RamBaseConfig::Auto) => {
            opts.auto_ram_base = true;
        }
        None => {}
    }
    if let Some(align) = overlay_align_override {
        opts.overlay_load_align = align;
    }
    opts.symbol_prefix = cli_symbol_prefix
        .or(toml_symbol_prefix)
        .filter(|s| !s.is_empty());
    if let Some(list) = explicit_checks {
        opts.check_bounds = false;
        opts.check_transitions = false;
        opts.check_overflow = false;
        opts.check_bank = false;
        for item in list.split(',') {
            match item.trim() {
                "bounds" => opts.check_bounds = true,
                "transition" => opts.check_transitions = true,
                "overflow" => opts.check_overflow = true,
                "bank" => opts.check_bank = true,
                "none" | "" => {}
                other => {
                    eprintln!("error: unknown check category `{other}`");
                    return ExitCode::FAILURE;
                }
            }
        }
    }

    if source_paths.is_empty() {
        eprintln!("error: `aun build` requires at least one `.aun` source file");
        return ExitCode::FAILURE;
    }

    if sdasz80_format && !asm_paths.is_empty() {
        eprintln!("error: external .s inputs are only supported with --format bin (for --format sdasz80, link .rel files with sdldz80)");
        return ExitCode::FAILURE;
    }

    let path = source_paths[0].clone();
    let Ok((src, mut program)) = compile_files(&source_paths) else {
        return ExitCode::FAILURE;
    };

    let stem = Path::new(&path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("out");

    if program
        .target
        .as_ref()
        .is_some_and(|t| matches!(t.arch.as_str(), "mcs51" | "ch559" | "8051"))
    {
        let mut mcs_opts = if release {
            aun_mcs51::CompileOptions::release()
        } else {
            aun_mcs51::CompileOptions::debug()
        };
        if let Some(addr) = code_origin_override {
            mcs_opts.code_origin = addr;
        }
        let out = match aun_mcs51::compile_mcs51(&program, mcs_opts) {
            Ok(o) => o,
            Err(err) => {
                docs::print_backend_error(&err);
                return ExitCode::FAILURE;
            }
        };
        if let Err(e) = fs::create_dir_all(&out_dir) {
            eprintln!(
                "error: failed to create output directory `{}`: {e}",
                out_dir.display()
            );
            return ExitCode::FAILURE;
        }
        let bin_path = out_dir.join(format!("{stem}.bin"));
        let ihx_path = out_dir.join(format!("{stem}.ihx"));
        let asm_path = out_dir.join(format!("{stem}.asm"));
        let sym_path = out_dir.join(format!("{stem}.sym"));
        let mut sym_txt = String::new();
        let mut syms: Vec<(&String, &u16)> = out.symbols.iter().collect();
        syms.sort_by(|a, b| a.1.cmp(b.1).then_with(|| a.0.cmp(b.0)));
        for (name, addr) in syms {
            sym_txt.push_str(&format!("{addr:04X} {name}\n"));
        }
        for (file_path, bytes) in [
            (&bin_path, out.resident_bin.as_slice()),
            (&ihx_path, out.intel_hex.as_bytes()),
            (&asm_path, out.asm_listing.as_bytes()),
            (&sym_path, sym_txt.as_bytes()),
        ] {
            if let Err(e) = fs::write(file_path, bytes) {
                eprintln!("error: failed to write `{}`: {e}", file_path.display());
                return ExitCode::FAILURE;
            }
        }
        println!(
            "aun build (mcs51): {} bytes (Data {} B, XData {} B, max stack {} B, SP=0x{:02X})",
            out.resident_bin.len(),
            out.data_bytes,
            out.xdata_bytes,
            out.idata_stack_bytes,
            out.sp_init
        );
        println!("  ihx: {}", ihx_path.display());
        println!("  bin: {}", bin_path.display());
        println!("  asm: {}", asm_path.display());
        println!("  sym: {}", sym_path.display());
        return ExitCode::SUCCESS;
    }

    if sdasz80_format {
        let (sdas, ram_bytes, max_stack) =
            match aun_z80::compile_z80_sdasz80(stem, &program, opts.clone()) {
                Ok(res) => res,
                Err(err) => {
                    docs::print_backend_error(&err);
                    return ExitCode::FAILURE;
                }
            };
        if let Err(e) = fs::create_dir_all(&out_dir) {
            eprintln!(
                "error: failed to create output directory `{}`: {e}",
                out_dir.display()
            );
            return ExitCode::FAILURE;
        }
        let s_path = out_dir.join(format!("{stem}.s"));
        if let Err(e) = fs::write(&s_path, sdas.as_bytes()) {
            eprintln!("error: failed to write `{}`: {e}", s_path.display());
            return ExitCode::FAILURE;
        }
        let inc_path = match aun_z80::collect_sdasz80_layout_equates(&program, &opts) {
            Ok(eqs) if !eqs.is_empty() => {
                let p = out_dir.join(format!("{stem}.layout.inc"));
                if let Err(e) = fs::write(&p, format_layout_inc(&eqs).as_bytes()) {
                    eprintln!("error: failed to write `{}`: {e}", p.display());
                    return ExitCode::FAILURE;
                }
                Some(p)
            }
            _ => None,
        };
        println!(
            "aun build (sdasz80): RAM {ram_bytes} bytes, max stack {max_stack} bytes\n  asm:      {}",
            s_path.display()
        );
        if let Some(p) = inc_path {
            println!("  inc:      {}", p.display());
        }
        return ExitCode::SUCCESS;
    }

    let mut ext_asms = Vec::with_capacity(asm_paths.len());
    for spec in &asm_paths {
        match load_external_asm(Path::new(&spec.path), &asm_include_dirs) {
            Ok(txt) => ext_asms.push(aun_z80::ExternalAsm {
                name: Path::new(&spec.path)
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| spec.path.clone()),
                source: txt,
                owner_state: spec.owner_state.clone(),
                is_startup: spec.is_startup,
            }),
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        }
    }

    let mut ext_referenced_syms: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    for ext in &ext_asms {
        for raw in ext.source.lines() {
            let code = raw.split(';').next().unwrap_or("");
            for tok in code.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
                if !tok.is_empty()
                    && !tok.as_bytes()[0].is_ascii_digit()
                {
                    ext_referenced_syms.insert(tok.to_string());
                }
            }
        }
    }
    let eliminated = aun_ir::opt::eliminate_dead_whole_program_bin(
        &mut program,
        &ext_referenced_syms,
        opts.symbol_prefix.as_deref(),
    );
    for item in &eliminated {
        if item.est_bytes > 0 {
            eprintln!(
                "warning[W001]: unused {} `{}` ({} bytes, eliminated at link time)",
                item.kind, item.name, item.est_bytes
            );
        } else {
            eprintln!(
                "warning[W001]: unused {} `{}` (eliminated at link time)",
                item.kind, item.name
            );
        }
    }

    opts.allow_budget_overflow = memory_report;
    let out = match aun_z80::compile_z80_with_external_asms(&program, &ext_asms, opts) {
        Ok(o) => o,
        Err(err) => {
            docs::print_backend_error(&err);
            return ExitCode::FAILURE;
        }
    };
    if memory_report {
        print!("{}", aun_z80::format_memory_report(&out.memory_report));
        if !out.memory_report.budget_errors.is_empty() {
            for err in &out.memory_report.budget_errors {
                docs::print_backend_error(err);
            }
            return ExitCode::FAILURE;
        }
    }

    if let Err(e) = fs::create_dir_all(&out_dir) {
        eprintln!(
            "error: failed to create output directory `{}`: {e}",
            out_dir.display()
        );
        return ExitCode::FAILURE;
    }

    let bin_path = out_dir.join(format!("{stem}.bin"));
    let asm_path = out_dir.join(format!("{stem}.asm"));
    let sym_path = out_dir.join(format!("{stem}.sym"));
    let manifest_path = out_dir.join(format!("{stem}.manifest.json"));

    let sym_content = format_sym_file(&path, &src, &program, &out);
    let manifest_content = format_manifest_json(&path, &src, &program, &out);

    for (file_path, bytes) in [
        (&bin_path, out.resident_bin.as_slice()),
        (&asm_path, out.asm_listing.as_bytes()),
        (&sym_path, sym_content.as_bytes()),
        (&manifest_path, manifest_content.as_bytes()),
    ] {
        if let Err(e) = fs::write(file_path, bytes) {
            eprintln!("error: failed to write `{}`: {e}", file_path.display());
            return ExitCode::FAILURE;
        }
    }
    for ovl in &out.overlays {
        let san = ovl.state_name.replace("::", "__").to_ascii_lowercase();
        let ovl_bin = out_dir.join(format!("ovl_{san}.bin"));
        let ovl_asm = out_dir.join(format!("ovl_{san}.asm"));
        if let Err(e) = fs::write(&ovl_bin, &ovl.bin) {
            eprintln!("error: failed to write `{}`: {e}", ovl_bin.display());
            return ExitCode::FAILURE;
        }
        if let Err(e) = fs::write(&ovl_asm, ovl.asm_listing.as_bytes()) {
            eprintln!("error: failed to write `{}`: {e}", ovl_asm.display());
            return ExitCode::FAILURE;
        }
    }
    let inc_path = if !out.asm_layout_equates.is_empty() {
        let p = out_dir.join(format!("{stem}.layout.inc"));
        if let Err(e) = fs::write(&p, format_layout_inc(&out.asm_layout_equates).as_bytes()) {
            eprintln!("error: failed to write `{}`: {e}", p.display());
            return ExitCode::FAILURE;
        }
        Some(p)
    } else {
        None
    };

    println!(
        "aun build: {} bytes ({} T-states, RAM {} bytes, max stack {} bytes, {} overlay(s))",
        out.resident_bin.len(),
        out.total_tstates,
        out.ram_bytes,
        out.max_stack_bytes,
        out.overlays.len()
    );
    println!("  bin:      {}", bin_path.display());
    println!("  asm:      {}", asm_path.display());
    println!("  sym:      {}", sym_path.display());
    println!("  manifest: {}", manifest_path.display());
    if let Some(p) = inc_path {
        println!("  inc:      {}", p.display());
    }
    ExitCode::SUCCESS
}

fn cmd_inspect(args: &[String]) -> ExitCode {
    let mut tstates_fn: Option<String> = None;
    let mut source_path: Option<String> = None;
    let mut opts = aun_z80::CompileOptions::debug();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--tstates" => {
                let Some(fn_name) = args.get(i + 1) else {
                    eprintln!("error: `--tstates` requires a function name");
                    return ExitCode::FAILURE;
                };
                tstates_fn = Some(fn_name.clone());
                i += 2;
            }
            a if a.starts_with("--tstates=") => {
                tstates_fn = Some(a["--tstates=".len()..].to_string());
                i += 1;
            }
            "--release" => {
                opts = aun_z80::CompileOptions::release();
                i += 1;
            }
            a if a.starts_with('-') => {
                eprintln!("error: unknown option `{a}` for `aun inspect`");
                return ExitCode::FAILURE;
            }
            file => {
                source_path = Some(file.to_string());
                i += 1;
            }
        }
    }

    let Some(path) = source_path else {
        eprintln!("error: `aun inspect` requires a `.aun` source file");
        return ExitCode::FAILURE;
    };

    let Ok((_, program)) = compile_file(&path) else {
        return ExitCode::FAILURE;
    };

    if program
        .target
        .as_ref()
        .is_some_and(|t| matches!(t.arch.as_str(), "mcs51" | "ch559" | "8051"))
    {
        let mcs_opts = if opts.release {
            aun_mcs51::CompileOptions::release()
        } else {
            aun_mcs51::CompileOptions::debug()
        };
        let out = match aun_mcs51::compile_mcs51(&program, mcs_opts) {
            Ok(o) => o,
            Err(err) => {
                docs::print_backend_error(&err);
                return ExitCode::FAILURE;
            }
        };
        println!("=== Aun Inspect (MCS-51 / CH559): {path} ===");
        println!("Code ROM:    {} bytes", out.resident_bin.len());
        println!(
            "Data RAM:    {} bytes (SP init @ 0x{:02X})",
            out.data_bytes, out.sp_init
        );
        println!("IData Stack: {} bytes max", out.idata_stack_bytes);
        println!("XData RAM:   {} bytes", out.xdata_bytes);
        println!("\nSymbols:");
        let mut syms: Vec<(&String, &u16)> = out.symbols.iter().collect();
        syms.sort_by(|a, b| a.1.cmp(b.1).then_with(|| a.0.cmp(b.0)));
        for (name, addr) in syms {
            println!("  0x{addr:04X}  {name}");
        }
        return ExitCode::SUCCESS;
    }

    let out = match aun_z80::compile_z80_with_options(&program, &[], opts) {
        Ok(o) => o,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };

    if let Some(target_fn) = tstates_fn {
        let ts_opt = out.fn_tstates.get(&target_fn).copied();
        let wcet_opt = out.fn_wcet_tstates.get(&target_fn).copied();
        if ts_opt.is_none() && wcet_opt.is_none() {
            eprintln!("error: function `{target_fn}` not found in `{path}`");
            return ExitCode::FAILURE;
        }
        let addr = out
            .symbols
            .get(&target_fn)
            .or_else(|| out.symbols.get(&target_fn.replace("::", "__")))
            .copied()
            .unwrap_or(0);
        match (ts_opt, wcet_opt) {
            (Some(ts), Some(wcet)) => {
                println!("{target_fn} @ 0x{addr:04X}: {ts} T-states (WCET: {wcet} T-states)");
            }
            (Some(ts), None) => {
                println!("{target_fn} @ 0x{addr:04X}: {ts} T-states (WCET: unbounded)");
            }
            (None, Some(wcet)) => {
                println!("{target_fn} @ 0x{addr:04X}: WCET {wcet} T-states");
            }
            (None, None) => unreachable!(),
        }
        if let Some(frame) = out.fn_frames.iter().find(|f| f.name == target_fn) {
            println!(
                "  locals:  0x{:04X}..0x{:04X} ({} bytes)",
                frame.locals_base,
                frame.locals_base.wrapping_add(frame.locals_size),
                frame.locals_size
            );
            println!(
                "  scratch: 0x{:04X}..0x{:04X} ({} bytes)",
                frame.scratch_base,
                frame.scratch_base.wrapping_add(frame.scratch_size),
                frame.scratch_size
            );
            println!("  stack:   {} bytes", frame.stack_depth);
        }
        return ExitCode::SUCCESS;
    }

    println!("=== Aun Inspect: {path} ===");
    println!(
        "Binary:      {} bytes @ 0x{:04X}",
        out.resident_bin.len(),
        out.code_origin
    );
    println!("Max Stack:   {} bytes", out.max_stack_bytes);
    println!("Total T:     {} T-states", out.total_tstates);

    println!(
        "\nUchiguri RAM Map (0x{:04X}..0x{:04X}, {} bytes):",
        out.ram_base, out.ram_end, out.ram_bytes
    );
    if out.globals.is_empty() {
        println!("  (no globals)");
    } else {
        for g in &out.globals {
            println!(
                "  global {:<20} 0x{:04X}..0x{:04X} ({} bytes)",
                g.name,
                g.addr,
                g.addr.wrapping_add(g.size),
                g.size
            );
        }
    }
    for f in &out.fn_frames {
        let frame_end = f.scratch_base.wrapping_add(f.scratch_size);
        println!(
            "  fn     {:<20} 0x{:04X}..0x{:04X} (locals {}B, scratch {}B, stack {}B)",
            f.name, f.locals_base, frame_end, f.locals_size, f.scratch_size, f.stack_depth
        );
    }

    println!("\nSymbols:");
    let mut syms: Vec<(&String, &u16)> = out.symbols.iter().collect();
    syms.sort_by(|a, b| a.1.cmp(b.1).then_with(|| a.0.cmp(b.0)));
    for (name, addr) in syms {
        println!("  0x{addr:04X}  {name}");
    }

    println!("\nT-State Summary:");
    let mut fn_ts: Vec<(&String, &u32)> = out.fn_tstates.iter().collect();
    fn_ts.sort_by(|a, b| a.0.cmp(b.0));
    for (name, ts) in &fn_ts {
        if let Some(wcet) = out.fn_wcet_tstates.get(*name) {
            println!("  {:<24} {:>6} T  (WCET {:>6} T)", name, ts, wcet);
        } else {
            println!("  {:<24} {:>6} T  (WCET unbounded)", name, ts);
        }
    }
    let mut extra_wcet: Vec<(&String, &u32)> = out
        .fn_wcet_tstates
        .iter()
        .filter(|(k, _)| !out.fn_tstates.contains_key(*k))
        .collect();
    extra_wcet.sort_by(|a, b| a.0.cmp(b.0));
    for (name, wcet) in extra_wcet {
        println!("  {:<24} {:>6}    (WCET {:>6} T)", name, "-", wcet);
    }
    println!("  {:<24} {:>6} T", "TOTAL", out.total_tstates);

    ExitCode::SUCCESS
}

fn cmd_guide(args: &[String]) -> ExitCode {
    let mode = docs::detect_render_mode(args);
    let mut topic: Option<&str> = None;
    for arg in args {
        if matches!(arg.as_str(), "--agent" | "--terminal" | "--json") {
            continue;
        }
        if arg.starts_with('-') {
            eprintln!("error: unknown option `{arg}` for `aun guide`");
            return ExitCode::FAILURE;
        }
        if topic.is_none() {
            topic = Some(arg.as_str());
        }
    }
    let engine = docs::create_doc_engine();
    match engine.guide(topic, mode) {
        Ok(text) => {
            print!("{text}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_explain(args: &[String]) -> ExitCode {
    let mode = docs::detect_render_mode(args);
    let mut code: Option<&str> = None;
    for arg in args {
        if matches!(arg.as_str(), "--agent" | "--terminal" | "--json") {
            continue;
        }
        if arg.starts_with('-') {
            eprintln!("error: unknown option `{arg}` for `aun explain`");
            return ExitCode::FAILURE;
        }
        if code.is_none() {
            code = Some(arg.as_str());
        }
    }
    let Some(query) = code else {
        eprintln!("error: `aun explain` requires a diagnostic code or topic (e.g. `aun explain E001`)");
        return ExitCode::FAILURE;
    };
    let engine = docs::create_doc_engine();
    match engine.explain(query, mode) {
        Ok(text) => {
            print!("{text}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_test(args: &[String]) -> ExitCode {
    let mut source_paths: Vec<String> = Vec::new();
    let mut test_sources: Vec<String> = Vec::new();
    let mut strict_externs = false;

    for arg in args {
        match arg.as_str() {
            "--strict-externs" => strict_externs = true,
            "--trace" => {}
            a if a.starts_with('-') => {
                eprintln!("error: unknown option `{a}` for `aun test`");
                return ExitCode::FAILURE;
            }
            file => {
                let p = Path::new(file);
                if p.extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e.eq_ignore_ascii_case("toml"))
                {
                    let mut ignored_asm = Vec::new();
                    let mut ignored_inc_dirs = Vec::new();
                    let mut ignored_origin = None;
                    let mut ignored_ram = None;
                    let mut ignored_ovl_align = None;
                    let mut ignored_prefix = None;
                    if let Err(e) = parse_aun_toml(
                        p,
                        &mut source_paths,
                        Some(&mut test_sources),
                        &mut ignored_asm,
                        &mut ignored_inc_dirs,
                        &mut ignored_origin,
                        &mut ignored_ram,
                        &mut ignored_ovl_align,
                        &mut ignored_prefix,
                    ) {
                        eprintln!("error: {e}");
                        return ExitCode::FAILURE;
                    }
                } else {
                    source_paths.push(file.to_string());
                }
            }
        }
    }

    for ts in test_sources {
        if !source_paths.contains(&ts) {
            source_paths.push(ts);
        }
    }

    if source_paths.is_empty() {
        eprintln!("error: `aun test` requires at least one `.aun` source file or `aun.toml`");
        return ExitCode::FAILURE;
    }

    let Ok((_, program)) = compile_files_for_test(&source_paths) else {
        return ExitCode::FAILURE;
    };
    let mut interp = Interpreter::new();
    interp.strict_externs = strict_externs;
    let results = interp.run_tests(&program);
    let mut failed = 0usize;
    for (name, outcome) in &results {
        match outcome {
            TestOutcome::Passed => println!("test {name} ... ok"),
            TestOutcome::Failed(msg) => {
                println!("test {name} ... FAILED ({msg})");
                failed += 1;
            }
        }
    }
    println!(
        "\ntest result: {}. {} passed; {} failed",
        if failed == 0 { "ok" } else { "FAILED" },
        results.len() - failed,
        failed
    );
    if failed == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--version" | "-V") => {
            println!("aun {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Some("--help" | "-h") | None => {
            print_usage();
            ExitCode::SUCCESS
        }
        Some("build") => cmd_build(&args[1..]),
        Some("test") => cmd_test(&args[1..]),
        Some("inspect") => cmd_inspect(&args[1..]),
        Some("guide") => cmd_guide(&args[1..]),
        Some("explain") => cmd_explain(&args[1..]),
        Some("mcp") => {
            if let Err(err) = docs::create_doc_engine().serve_stdio() {
                eprintln!("error: MCP server failed: {err}");
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            }
        }
        Some(other) => {
            eprintln!("error: unknown subcommand `{other}`");
            print_usage();
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_build_bin_links_external_s_and_aun_toml() {
        let tmp = env::temp_dir().join(format!("aun_cli_ext_s_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        fs::write(tmp.join("defs.inc"), "ADD_VAL .equ 7\n").unwrap();
        fs::write(
            tmp.join("ext.s"),
            r#"
                .include "defs.inc"
                .area _CODE
                .globl _ext_add
            _ext_add::
                add a, #ADD_VAL
                ret
            "#,
        )
        .unwrap();
        fs::write(
            tmp.join("helper.aun"),
            r#"
                pub fn extra(x: u8) -> u8 {
                    x +% 1u8
                }
            "#,
        )
        .unwrap();
        fs::write(
            tmp.join("main.aun"),
            r#"
                extern "asm" {
                    fn _ext_add(v: u8 in a) -> u8 in a clobbers(A, F);
                }
                pub fn entry(x: u8) -> u8 {
                    _ext_add(x) +% helper::extra(x)
                }
            "#,
        )
        .unwrap();
        fs::write(
            tmp.join("aun.toml"),
            r#"
entry = "main.aun"
sources = ["helper.aun"]
asm_sources = ["ext.s"]
code_origin = 0x0200
ram_base = 0xD000
"#,
        )
        .unwrap();

        let out1 = tmp.join("out_direct");
        let args1: Vec<String> = [
            "--release",
            "--format",
            "bin",
            tmp.join("main.aun").to_str().unwrap(),
            tmp.join("helper.aun").to_str().unwrap(),
            tmp.join("ext.s").to_str().unwrap(),
            "-o",
            out1.to_str().unwrap(),
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(cmd_build(&args1), ExitCode::SUCCESS);
        assert!(out1.join("main.bin").is_file());
        assert!(out1.join("main.asm").is_file());
        assert!(out1.join("main.sym").is_file());
        assert!(out1.join("main.manifest.json").is_file());
        let sym1 = fs::read_to_string(out1.join("main.sym")).unwrap();
        assert!(sym1.contains("_ext_add"));

        let out2 = tmp.join("out_toml");
        let args2: Vec<String> = [
            "--release",
            tmp.join("aun.toml").to_str().unwrap(),
            "-o",
            out2.to_str().unwrap(),
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(cmd_build(&args2), ExitCode::SUCCESS);
        let manifest2 = fs::read_to_string(out2.join("main.manifest.json")).unwrap();
        assert!(manifest2.contains("\"code_origin\": \"0x0200\""));
        assert!(manifest2.contains("\"base\": \"0xD000\""));

        let sdas_err_args: Vec<String> = [
            "--format",
            "sdasz80",
            tmp.join("main.aun").to_str().unwrap(),
            tmp.join("ext.s").to_str().unwrap(),
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(cmd_build(&sdas_err_args), ExitCode::FAILURE);

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn cli_build_overlay_asm_and_full_manifest_json() {
        let tmp = env::temp_dir().join(format!("aun_cli_ovl_manifest_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        fs::write(
            tmp.join("ingame.s"),
            r#"
                .area _CODE
            ext_ingame_step:
                add a, #3
                ret
            "#,
        )
        .unwrap();
        fs::write(
            tmp.join("app.aun"),
            r#"
                target pc88sr : z80 {
                    space MainRam   @ 0x0100..0xBFFF;
                    stack MainStack @ 0xBE00..0xBFFF;
                    bank_group HighWindow @ 0xC000..0xFFFF {
                        bank HighMainRam @ 0xC000..0xEFFF;
                        default HighMainRam;
                    }
                    on_swap(ops: &const [LoadOp]) => app_swap(ops);
                }

                fn app_swap(_ops: &const [LoadOp]) {}

                extern "asm" {
                    fn ext_ingame_step(v: u8 in A) -> u8 in A clobbers(A, F);
                }

                machine App {
                    state Title { cursor: u8 };
                    #[swap_boundary]
                    state InGame {
                        static mut BUF: [u8; 8] in HighMainRam;
                        #[swap_boundary]
                        state Stage1 { wave: u8 };
                        initial Stage1;
                    }
                    transitions {
                        Title -> InGame;
                    }
                    on InGame::Stage1 {
                        fn run(&mut self) -> u8 {
                            BUF[0] = self.wave;
                            ext_ingame_step(self.wave)
                        }
                    }
                }

                pub fn main() -> u8 {
                    let mut app: App = App::new();
                    app.goto(InGame::Stage1 { wave: 4u8 });
                    app.run()
                }
            "#,
        )
        .unwrap();
        fs::write(
            tmp.join("aun.toml"),
            r#"
entry = "app.aun"
code_origin = 0x0100
ram_base = 0x8000

[overlay_asm]
"App::InGame" = "ingame.s"
"#,
        )
        .unwrap();

        let out_toml = tmp.join("out_toml");
        let args_toml: Vec<String> = [
            tmp.join("aun.toml").to_str().unwrap(),
            "-o",
            out_toml.to_str().unwrap(),
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(cmd_build(&args_toml), ExitCode::SUCCESS);
        assert!(out_toml.join("ovl_ingame.bin").is_file());
        assert!(out_toml.join("ovl_ingame__stage1.bin").is_file());
        let manifest = fs::read_to_string(out_toml.join("app.manifest.json")).unwrap();
        assert!(manifest.contains("\"spaces\": ["), "{manifest}");
        assert!(manifest.contains("\"name\": \"HighMainRam\""), "{manifest}");
        assert!(manifest.contains("\"blobs\": ["), "{manifest}");
        assert!(manifest.contains("\"file\": \"ovl_ingame.bin\""), "{manifest}");
        assert!(manifest.contains("\"asm_files\": [\"ingame.s\"]"), "{manifest}");
        assert!(manifest.contains("\"file\": \"ovl_ingame__stage1.bin\""), "{manifest}");
        assert!(manifest.contains("\"states\": ["), "{manifest}");
        assert!(manifest.contains("\"state_buffers\": ["), "{manifest}");
        assert!(manifest.contains("\"owner_state\": \"App::InGame\""), "{manifest}");

        // Also test `--asm StatePath=file.s` directly
        let out_cli = tmp.join("out_cli");
        let args_cli: Vec<String> = [
            tmp.join("app.aun").to_str().unwrap(),
            "--asm",
            &format!("App::InGame={}", tmp.join("ingame.s").display()),
            "--ram-base",
            "0x8000",
            "-o",
            out_cli.to_str().unwrap(),
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(
            cmd_build(&args_cli),
            ExitCode::SUCCESS
        );
        assert!(out_cli.join("ovl_ingame.bin").is_file());
        assert!(out_cli.join("ovl_ingame__stage1.bin").is_file());

        // Verify Issue #54: aun.toml `ram_base = "auto"` + `sector_size = 256` packs RAM right below stack
        // and reports `aligned_size` / `aligned_end` in manifest.json.
        fs::write(
            tmp.join("aun_auto.toml"),
            r#"
entry = "app.aun"
code_origin = 0x0100
ram_base = "auto"
sector_size = 256

[overlay_asm]
"App::InGame" = "ingame.s"
"#,
        )
        .unwrap();
        let out_auto = tmp.join("out_auto");
        let args_auto: Vec<String> = [
            tmp.join("aun_auto.toml").to_str().unwrap(),
            "-o",
            out_auto.to_str().unwrap(),
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(cmd_build(&args_auto), ExitCode::SUCCESS);
        let manifest_auto = fs::read_to_string(out_auto.join("app.manifest.json")).unwrap();
        assert!(
            manifest_auto.contains("\"end\": \"0xBE00\""),
            "auto ram_base should pack RAM flush below stack 0xBE00:\n{manifest_auto}"
        );
        assert!(
            manifest_auto.contains("\"aligned_size\": 256"),
            "manifest should report 256B-aligned overlay size:\n{manifest_auto}"
        );

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn cli_test_multi_file_aun_toml_test_sources_and_strict_externs() {
        let tmp = env::temp_dir().join(format!("aun_cli_test_multi_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        fs::write(
            tmp.join("physics.aun"),
            r#"
                pub static mut _g_val: u8 = 10u8;
                pub fn accel(x: u8) -> u8 { x +% 2u8 }
            "#,
        )
        .unwrap();
        fs::write(
            tmp.join("stage1.aun"),
            r#"
                extern "asm" {
                    static mut _g_val: u8;
                    fn _accel(x: u8 in a) -> u8 in a;
                    fn _sfx(id: u8 in a);
                }

                pub fn step(x: u8) -> u8 {
                    _sfx(1u8);
                    _g_val = _g_val +% _accel(x);
                    _g_val
                }
            "#,
        )
        .unwrap();
        fs::write(
            tmp.join("harness.aun"),
            r#"
                #[cfg(test)]
                #[mock("_sfx")]
                fn mock_sfx(_id: u8) {}

                #[cfg(test)]
                #[test]
                fn test_multi_unit_step() {
                    assert!(stage1::step(3u8) == 15u8);
                    assert!(physics::_g_val == 15u8);
                }
            "#,
        )
        .unwrap();
        fs::write(
            tmp.join("harness_unmocked.aun"),
            r#"
                #[cfg(test)]
                #[test]
                fn test_without_sfx_mock() {
                    assert!(stage1::step(3u8) == 15u8);
                }
            "#,
        )
        .unwrap();

        // 1. Direct multi-file `aun test` with `--strict-externs`
        assert_eq!(
            cmd_test(&[
                "--strict-externs".into(),
                tmp.join("harness.aun").to_string_lossy().into_owned(),
                tmp.join("physics.aun").to_string_lossy().into_owned(),
                tmp.join("stage1.aun").to_string_lossy().into_owned(),
            ]),
            ExitCode::SUCCESS
        );

        // 2. `--strict-externs` fails when `_sfx` is unmocked, while non-strict passes
        assert_eq!(
            cmd_test(&[
                "--strict-externs".into(),
                tmp.join("harness_unmocked.aun").to_string_lossy().into_owned(),
                tmp.join("physics.aun").to_string_lossy().into_owned(),
                tmp.join("stage1.aun").to_string_lossy().into_owned(),
            ]),
            ExitCode::FAILURE
        );
        assert_eq!(
            cmd_test(&[
                tmp.join("harness_unmocked.aun").to_string_lossy().into_owned(),
                tmp.join("physics.aun").to_string_lossy().into_owned(),
                tmp.join("stage1.aun").to_string_lossy().into_owned(),
            ]),
            ExitCode::SUCCESS
        );

        // 3. `aun.toml` with `test_sources = [...]`
        let toml_root = tmp.join("aun_root.toml");
        fs::write(
            &toml_root,
            r#"
entry = "stage1.aun"
sources = ["physics.aun"]
test_sources = ["harness.aun"]
"#,
        )
        .unwrap();
        assert_eq!(
            cmd_test(&[
                "--strict-externs".into(),
                toml_root.to_string_lossy().into_owned(),
            ]),
            ExitCode::SUCCESS
        );

        // 4. `aun.toml` with `[test]` section `sources = [...]`, and `aun build` ignores `[test]`
        let toml_section = tmp.join("aun_section.toml");
        fs::write(
            &toml_section,
            r#"
entry = "stage1.aun"
sources = ["physics.aun"]

[test]
sources = ["harness.aun"]
"#,
        )
        .unwrap();
        assert_eq!(
            cmd_test(&[
                "--strict-externs".into(),
                toml_section.to_string_lossy().into_owned(),
            ]),
            ExitCode::SUCCESS
        );
        // Verify `aun build` strips `#[cfg(test)]` and ignores `[test].sources`
        let sdas_out = tmp.join("out_sdas");
        assert_eq!(
            cmd_build(&[
                "--format".into(),
                "sdasz80".into(),
                toml_section.to_string_lossy().into_owned(),
                "-o".into(),
                sdas_out.to_string_lossy().into_owned(),
            ]),
            ExitCode::SUCCESS
        );
        let asm_txt = fs::read_to_string(sdas_out.join("stage1.s")).unwrap();
        assert!(!asm_txt.contains("mock_sfx"));
        assert!(!asm_txt.contains("test_multi_unit_step"));

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn examples_and_bench_suite_build_and_test_cleanly() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..");
        let tmp = env::temp_dir().join(format!("aun_cli_examples_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        let run_host_tests = |path: &Path| {
            let p_str = path.to_string_lossy().into_owned();
            let (_, prog) = compile_file(&p_str)
                .unwrap_or_else(|_| panic!("failed to compile `{}`", path.display()));
            let mut interp = Interpreter::new();
            let results = interp.run_tests(&prog);
            assert!(
                !results.is_empty(),
                "expected #[test] cases in `{}`",
                path.display()
            );
            for (name, outcome) in results {
                assert!(
                    matches!(outcome, TestOutcome::Passed),
                    "host test `{name}` failed in `{}`: {outcome:?}",
                    path.display()
                );
            }
        };

        // 1. examples/pc88_pinball_mvp.aun: host tests + Z80 bin build
        let pinball = root.join("examples/pc88_pinball_mvp.aun");
        run_host_tests(&pinball);
        let pinball_out = tmp.join("pinball");
        assert_eq!(
            cmd_build(&[
                pinball.to_string_lossy().into_owned(),
                "-o".into(),
                pinball_out.to_string_lossy().into_owned(),
            ]),
            ExitCode::SUCCESS
        );
        assert!(pinball_out.join("pc88_pinball_mvp.bin").is_file());

        // 2. examples/hsm_overlay_app/: host tests + TOML overlay build
        let hsm_dir = root.join("examples/hsm_overlay_app");
        run_host_tests(&hsm_dir.join("src/main.aun"));
        let hsm_out = tmp.join("hsm");
        assert_eq!(
            cmd_build(&[
                hsm_dir.join("aun.toml").to_string_lossy().into_owned(),
                "-o".into(),
                hsm_out.to_string_lossy().into_owned(),
            ]),
            ExitCode::SUCCESS
        );
        assert!(hsm_out.join("main.bin").is_file());
        assert!(hsm_out.join("ovl_ingame.bin").is_file());
        assert!(hsm_out.join("ovl_ingame__stage1.bin").is_file());
        assert!(hsm_out.join("ovl_ingame__stage2.bin").is_file());
        let hsm_manifest = fs::read_to_string(hsm_out.join("main.manifest.json")).unwrap();
        assert!(hsm_manifest.contains("\"blobs\": ["), "{hsm_manifest}");
        assert!(hsm_manifest.contains("\"state_buffers\": ["), "{hsm_manifest}");

        // 3. bench/suite/entity_pool.aun: host tests + sdasz80 build
        let entity_pool = root.join("bench/suite/entity_pool.aun");
        run_host_tests(&entity_pool);
        for name in ["entity_pool.aun", "mdxnt_drv.aun", "game_tally.aun", "sfx_dispatch.aun"] {
            let p = root.join("bench/suite").join(name);
            let out_dir = tmp.join(name);
            assert_eq!(
                cmd_build(&[
                    "--release".into(),
                    "--format".into(),
                    "sdasz80".into(),
                    p.to_string_lossy().into_owned(),
                    "-o".into(),
                    out_dir.to_string_lossy().into_owned(),
                ]),
                ExitCode::SUCCESS,
                "failed sdasz80 build for {name}"
            );
        }

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn cli_build_include_dirs_crt0_startup_and_bndry256_bin() {
        let tmp = env::temp_dir().join(format!("aun_cli_crt0_inc_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        let inc1 = tmp.join("inc_common");
        let inc2 = tmp.join("inc_hw");
        fs::create_dir_all(&inc1).unwrap();
        fs::create_dir_all(&inc2).unwrap();

        fs::write(inc1.join("consts.inc"), "VEC_PAGE .equ 0x0200\n").unwrap();
        fs::write(inc2.join("ports.inc"), "PORT_IRQ .equ 0xE4\n").unwrap();

        fs::write(
            tmp.join("crt0.s"),
            r#"
                .include "consts.inc"
                .include "ports.inc"
                .area _HEADER (ABS)
                .org 0x0100
            _crt0_entry::
                di
                ld sp, #0xF000
                ld hl, #s__DATA
                ld bc, #l__DATA
                ld a, #0x02
                ld i, a
                im 2
                call main
                halt
                .org VEC_PAGE
            _im2_vectors::
                .ds 258
            "#,
        )
        .unwrap();

        fs::write(
            tmp.join("lut256.s"),
            r#"
                .area _CODE
                .bndry 256
                .globl _lut256
            _lut256::
                .db 0x11, 0x22, 0x33, 0x44
            "#,
        )
        .unwrap();

        fs::write(
            tmp.join("main.aun"),
            r#"
                static mut COUNTER: u8 = 0u8;
                extern "asm" {
                    static _lut256: [u8; 4];
                }
                pub fn main() -> u8 {
                    COUNTER = COUNTER +% _lut256[1];
                    COUNTER
                }
            "#,
        )
        .unwrap();

        fs::write(
            tmp.join("aun.toml"),
            r#"
entry = "main.aun"
crt0 = "crt0.s"
asm_sources = ["lut256.s"]
asm_include_dirs = ["inc_common", "inc_hw"]
code_origin = 0x0100
ram_base = 0xC000
"#,
        )
        .unwrap();

        // 1. Build via aun.toml (`crt0` + `asm_include_dirs` + `.bndry 256`)
        let out_toml = tmp.join("out_toml");
        let args_toml: Vec<String> = [
            "--release",
            "--format",
            "bin",
            tmp.join("aun.toml").to_str().unwrap(),
            "-o",
            out_toml.to_str().unwrap(),
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(cmd_build(&args_toml), ExitCode::SUCCESS);
        let bin = fs::read(out_toml.join("main.bin")).unwrap();
        let sym = fs::read_to_string(out_toml.join("main.sym")).unwrap();
        let parse_sym = |name: &str| -> u16 {
            let line = sym
                .lines()
                .find(|l| l.split_whitespace().nth(1) == Some(name))
                .unwrap_or_else(|| panic!("missing symbol `{name}` in:\n{sym}"));
            let hex = line.split_whitespace().next().unwrap().trim_start_matches("0x");
            u16::from_str_radix(hex, 16).unwrap()
        };
        let entry_addr = parse_sym("_crt0_entry");
        let vec_addr = parse_sym("_im2_vectors");
        let main_addr = parse_sym("main");
        let lut_addr = parse_sym("_lut256");
        let s_data = parse_sym("s__DATA");
        let l_data = parse_sym("l__DATA");
        assert_eq!(entry_addr, 0x0100);
        assert_eq!(vec_addr, 0x0200);
        assert!(main_addr >= 0x0302, "main must follow crt0 (got 0x{main_addr:04X})");
        assert_eq!(lut_addr & 0x00FF, 0, "_lut256 must be 256-byte aligned (0x{lut_addr:04X})");
        assert_eq!(s_data, 0xC000);
        assert!(l_data >= 1);
        // First instruction at 0x0100 is `di` (0xF3)
        assert_eq!(bin[0], 0xF3);
        let lut_off = (lut_addr - 0x0100) as usize;
        assert_eq!(&bin[lut_off..lut_off + 4], &[0x11, 0x22, 0x33, 0x44]);

        // 2. Build via direct CLI flags (`-I <dir>`, `-I=<dir>`, `--startup-asm`)
        let out_cli = tmp.join("out_cli");
        let args_cli: Vec<String> = [
            "--release",
            "-I",
            inc1.to_str().unwrap(),
            &format!("-I={}", inc2.display()),
            "--startup-asm",
            tmp.join("crt0.s").to_str().unwrap(),
            "--asm",
            tmp.join("lut256.s").to_str().unwrap(),
            tmp.join("main.aun").to_str().unwrap(),
            "-o",
            out_cli.to_str().unwrap(),
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(cmd_build(&args_cli), ExitCode::SUCCESS);
        assert_eq!(fs::read(out_cli.join("main.bin")).unwrap(), bin);

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn cli_build_symbol_prefix_export_name_and_toml_config() {
        let tmp = env::temp_dir().join(format!("aun_cli_sym_prefix_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        fs::write(
            tmp.join("aun.toml"),
            r#"
entry = "main.aun"
crt0 = "crt0.s"
symbol_prefix = "_"
"#,
        )
        .unwrap();
        fs::write(
            tmp.join("crt0.s"),
            r#"
    .area _CODE
_crt0_entry::
    ld hl, #s__INITIALIZER
    ld de, #s__INITIALIZED
    ld bc, #l__INITIALIZER
    ld a, b
    or c
    jr z, _skip_init
    ldir
_skip_init:
    call _main
    call _stage1_on_target_hit
    ret
"#,
        )
        .unwrap();
        fs::write(
            tmp.join("main.aun"),
            r#"
#[export_name = "_custom_counter"]
pub static mut COUNTER: u16 = 0x1234u16;

pub static mut UNPREFIXED_OK: u8 = 0x5Au8;

#[export_name("_stage1_on_target_hit")]
pub fn hit_handler() -> u16 {
    COUNTER
}

pub fn main() {
    COUNTER = COUNTER + (UNPREFIXED_OK as u16);
}
"#,
        )
        .unwrap();

        let out_toml = tmp.join("out_toml");
        let args_toml: Vec<String> = [
            "--format",
            "bin",
            tmp.join("aun.toml").to_str().unwrap(),
            "-o",
            out_toml.to_str().unwrap(),
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(cmd_build(&args_toml), ExitCode::SUCCESS);
        let sym_toml = fs::read_to_string(out_toml.join("main.sym")).unwrap();
        let has_sym = |sym: &str, name: &str| {
            sym.lines().any(|l| l.split_whitespace().nth(1) == Some(name))
        };
        assert!(has_sym(&sym_toml, "_main"), "missing _main in:\n{sym_toml}");
        assert!(has_sym(&sym_toml, "main"), "missing main in:\n{sym_toml}");
        assert!(
            has_sym(&sym_toml, "_stage1_on_target_hit"),
            "missing _stage1_on_target_hit in:\n{sym_toml}"
        );
        assert!(
            has_sym(&sym_toml, "_custom_counter"),
            "missing _custom_counter in:\n{sym_toml}"
        );
        assert!(
            has_sym(&sym_toml, "_UNPREFIXED_OK"),
            "missing _UNPREFIXED_OK in:\n{sym_toml}"
        );

        // CLI `--symbol-prefix` flag overrides aun.toml
        let out_cli = tmp.join("out_cli");
        fs::write(
            tmp.join("crt0_sym.s"),
            r#"
    .area _CODE
_crt0_entry::
    call __aun_startup_init
    call sym_main
    ret
"#,
        )
        .unwrap();
        let args_cli: Vec<String> = [
            "--format",
            "bin",
            "--startup-asm",
            tmp.join("crt0_sym.s").to_str().unwrap(),
            "--symbol-prefix",
            "sym_",
            tmp.join("main.aun").to_str().unwrap(),
            "-o",
            out_cli.to_str().unwrap(),
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(cmd_build(&args_cli), ExitCode::SUCCESS);
        let sym_cli = fs::read_to_string(out_cli.join("main.sym")).unwrap();
        assert!(has_sym(&sym_cli, "sym_main"), "missing sym_main in:\n{sym_cli}");
        assert!(
            has_sym(&sym_cli, "sym_UNPREFIXED_OK"),
            "missing sym_UNPREFIXED_OK in:\n{sym_cli}"
        );
        assert!(
            has_sym(&sym_cli, "_stage1_on_target_hit"),
            "missing _stage1_on_target_hit in:\n{sym_cli}"
        );
        assert!(
            has_sym(&sym_cli, "_custom_counter"),
            "missing _custom_counter in:\n{sym_cli}"
        );

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn cli_build_emits_layout_inc_for_export_asm_layout() {
        let tmp = env::temp_dir().join(format!("aun_cli_layout_inc_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        let src_path = tmp.join("game.aun");
        fs::write(
            &src_path,
            r#"
target pc88sr : z80 {
    space MainRam @ 0x0100..0xEBFF;
    space HighMainRam @ 0xED00..0xF0FF;
    stack MainStack @ 0xEA00..0xEBFF;
}

struct Stage2State {
    boss_hp: u8,
    timer: u16,
    flags: u8,
}

machine Game {
    initial Stage1;
    state Stage1 { dummy: u8 }
    state Stage2 {
        #[zero_on_enter]
        #[export_asm_layout]
        static mut S2_STATE: Stage2State in HighMainRam;
    }
    transitions {
        Stage1 -> Stage2;
        Stage2 -> Stage1;
    }
    on Stage2 {
        fn bump(&mut self) -> u8 {
            S2_STATE.boss_hp = S2_STATE.boss_hp +% 1u8;
            S2_STATE.boss_hp
        }
    }
}

static mut G: Game = Game::new();

pub fn main() -> u8 {
    G.goto(Stage2 {});
    G.bump()
}
"#,
        )
        .unwrap();

        let out_bin = tmp.join("out_bin");
        let args_bin: Vec<String> = [
            "--release",
            src_path.to_str().unwrap(),
            "-o",
            out_bin.to_str().unwrap(),
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(cmd_build(&args_bin), ExitCode::SUCCESS);
        let inc_bin = fs::read_to_string(out_bin.join("game.layout.inc"))
            .expect("game.layout.inc should be emitted for bin build");
        assert!(inc_bin.contains("S2_STATE = 0xED00"), "{inc_bin}");
        assert!(inc_bin.contains("S2_STATE_SIZE = 0x0004"), "{inc_bin}");
        assert!(inc_bin.contains("STAGE2STATE_SIZE = 0x0004"), "{inc_bin}");
        assert!(inc_bin.contains("STAGE2STATE_OFF_BOSS_HP = 0x0000"), "{inc_bin}");
        assert!(inc_bin.contains("STAGE2STATE_OFF_TIMER = 0x0001"), "{inc_bin}");
        assert!(inc_bin.contains("STAGE2STATE_OFF_FLAGS = 0x0003"), "{inc_bin}");

        let out_sdas = tmp.join("out_sdas");
        let args_sdas: Vec<String> = [
            "--release",
            "--format",
            "sdasz80",
            src_path.to_str().unwrap(),
            "-o",
            out_sdas.to_str().unwrap(),
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(cmd_build(&args_sdas), ExitCode::SUCCESS);
        let sdas_s = fs::read_to_string(out_sdas.join("game.s")).unwrap();
        let inc_sdas = fs::read_to_string(out_sdas.join("game.layout.inc")).unwrap();
        assert!(sdas_s.contains("S2_STATE = 0xED00"), "{sdas_s}");
        assert!(sdas_s.contains("STAGE2STATE_OFF_TIMER = 0x0001"), "{sdas_s}");
        assert!(inc_sdas.contains("S2_STATE = 0xED00"), "{inc_sdas}");
        assert!(inc_sdas.contains("STAGE2STATE_OFF_FLAGS = 0x0003"), "{inc_sdas}");

        let _ = fs::remove_dir_all(&tmp);
    }
}
