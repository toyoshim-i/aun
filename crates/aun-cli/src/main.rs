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
           aun build [-o <dir>] [--release] [--checks=<list>] [--format bin|sdasz80] [--asm <file.s|dir>] [--code-origin <addr>] [--ram-base <addr>] <file.aun|file.s|aun.toml> ...\n  \
           aun test [--trace] <file.aun>\n  \
           aun inspect [--tstates <fn>] <file.aun>\n  \
           aun guide [topic] [--agent|--terminal|--json]\n  \
           aun explain <code> [--agent|--terminal|--json]\n  \
           aun mcp\n  \
           aun --version",
        env!("CARGO_PKG_VERSION")
    );
}

fn parse_u16_addr(raw: &str) -> Option<u16> {
    let s = raw.trim().trim_matches('"');
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u16::from_str_radix(hex, 16).ok()
    } else {
        s.parse::<u16>().ok()
    }
}

fn collect_asm_arg(path: &Path, asm_paths: &mut Vec<String>) -> Result<(), String> {
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
            asm_paths.push(p.to_string_lossy().into_owned());
        }
    } else {
        asm_paths.push(path.to_string_lossy().into_owned());
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

fn parse_aun_toml(
    manifest_path: &Path,
    source_paths: &mut Vec<String>,
    asm_paths: &mut Vec<String>,
    code_origin: &mut Option<u16>,
    ram_base: &mut Option<u16>,
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

    for stmt in statements {
        let Some((lhs, rhs)) = stmt.split_once('=') else {
            continue;
        };
        let key = lhs.trim();
        let val = rhs.trim();
        match key {
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
            "asm_sources" => {
                for rel in extract_toml_string_list(val) {
                    collect_asm_arg(&base_dir.join(rel), asm_paths)?;
                }
            }
            "code_origin" => {
                let parsed = parse_u16_addr(val)
                    .ok_or_else(|| format!("invalid `code_origin` in `{}`: `{val}`", manifest_path.display()))?;
                *code_origin = Some(parsed);
            }
            "ram_base" => {
                let parsed = parse_u16_addr(val)
                    .ok_or_else(|| format!("invalid `ram_base` in `{}`: `{val}`", manifest_path.display()))?;
                *ram_base = Some(parsed);
            }
            _ => {}
        }
    }
    Ok(())
}

fn load_external_asm_recursive(path: &Path, stack: &mut Vec<PathBuf>) -> Result<String, String> {
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
            let rel = tail.trim().trim_matches('"');
            if rel.is_empty() {
                stack.pop();
                return Err(format!("invalid `.include` directive in `{}`: `{raw}`", path.display()));
            }
            let inc_path = base_dir.join(rel);
            let inc_text = load_external_asm_recursive(&inc_path, stack)?;
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

fn load_external_asm(path: &Path) -> Result<String, String> {
    let mut stack = Vec::new();
    load_external_asm_recursive(path, &mut stack)
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
    let _ = writeln!(buf, "  ]");
    let _ = writeln!(buf, "}}");
    buf
}

fn cmd_build(args: &[String]) -> ExitCode {
    let mut out_dir = PathBuf::from("target/aun");
    let mut source_paths: Vec<String> = Vec::new();
    let mut asm_paths: Vec<String> = Vec::new();
    let mut code_origin_override: Option<u16> = None;
    let mut ram_base_override: Option<u16> = None;
    let mut release = false;
    let mut explicit_checks: Option<String> = None;
    let mut sdasz80_format = false;

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
            "--asm" => {
                let Some(asm_arg) = args.get(i + 1) else {
                    eprintln!("error: `--asm` requires a `.s` file or directory path");
                    return ExitCode::FAILURE;
                };
                if let Err(e) = collect_asm_arg(Path::new(asm_arg), &mut asm_paths) {
                    eprintln!("error: {e}");
                    return ExitCode::FAILURE;
                }
                i += 2;
            }
            a if a.starts_with("--asm=") => {
                if let Err(e) = collect_asm_arg(Path::new(&a["--asm=".len()..]), &mut asm_paths) {
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
                    eprintln!("error: `--ram-base` requires an address");
                    return ExitCode::FAILURE;
                };
                let Some(addr) = parse_u16_addr(v) else {
                    eprintln!("error: invalid `--ram-base` address `{v}`");
                    return ExitCode::FAILURE;
                };
                ram_base_override = Some(addr);
                i += 2;
            }
            a if a.starts_with("--ram-base=") => {
                let v = &a["--ram-base=".len()..];
                let Some(addr) = parse_u16_addr(v) else {
                    eprintln!("error: invalid `--ram-base` address `{v}`");
                    return ExitCode::FAILURE;
                };
                ram_base_override = Some(addr);
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
                    Some("s" | "asm") => asm_paths.push(file.to_string()),
                    Some("toml") => {
                        if let Err(e) = parse_aun_toml(
                            p,
                            &mut source_paths,
                            &mut asm_paths,
                            &mut code_origin_override,
                            &mut ram_base_override,
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
    if let Some(addr) = ram_base_override {
        opts.ram_base = addr;
    }
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
    let Ok((src, program)) = compile_files(&source_paths) else {
        return ExitCode::FAILURE;
    };

    let stem = Path::new(&path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("out");

    if sdasz80_format {
        let (sdas, ram_bytes, max_stack) =
            match aun_z80::compile_z80_sdasz80(stem, &program, opts) {
                Ok(res) => res,
                Err(err) => {
                    eprintln!("error: {err}");
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
        println!(
            "aun build (sdasz80): RAM {ram_bytes} bytes, max stack {max_stack} bytes\n  asm:      {}",
            s_path.display()
        );
        return ExitCode::SUCCESS;
    }

    let mut ext_asm_texts = Vec::with_capacity(asm_paths.len());
    for asm_p in &asm_paths {
        match load_external_asm(Path::new(asm_p)) {
            Ok(txt) => ext_asm_texts.push(txt),
            Err(err) => {
                eprintln!("error: {err}");
                return ExitCode::FAILURE;
            }
        }
    }
    let ext_refs: Vec<&str> = ext_asm_texts.iter().map(String::as_str).collect();

    let out = match aun_z80::compile_z80_with_options(&program, &ext_refs, opts) {
        Ok(o) => o,
        Err(err) => {
            eprintln!("error: {err}");
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
        let ovl_bin = out_dir.join(format!("ovl_{}.bin", ovl.state_name.to_ascii_lowercase()));
        let ovl_asm = out_dir.join(format!("ovl_{}.asm", ovl.state_name.to_ascii_lowercase()));
        if let Err(e) = fs::write(&ovl_bin, &ovl.bin) {
            eprintln!("error: failed to write `{}`: {e}", ovl_bin.display());
            return ExitCode::FAILURE;
        }
        if let Err(e) = fs::write(&ovl_asm, ovl.asm_listing.as_bytes()) {
            eprintln!("error: failed to write `{}`: {e}", ovl_asm.display());
            return ExitCode::FAILURE;
        }
    }

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

    let out = match aun_z80::compile_z80_with_options(&program, &[], opts) {
        Ok(o) => o,
        Err(err) => {
            eprintln!("error: {err}");
            return ExitCode::FAILURE;
        }
    };

    if let Some(target_fn) = tstates_fn {
        let Some(&ts) = out.fn_tstates.get(&target_fn) else {
            eprintln!("error: function `{target_fn}` not found in `{path}`");
            return ExitCode::FAILURE;
        };
        let addr = out.symbols.get(&target_fn).copied().unwrap_or(0);
        println!("{target_fn} @ 0x{addr:04X}: {ts} T-states");
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
    for (name, ts) in fn_ts {
        println!("  {:<24} {:>6} T", name, ts);
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
        Some("test") => {
            let Some(path) = args.iter().skip(1).find(|a| !a.starts_with('-')) else {
                eprintln!("error: `aun test` requires a `.aun` source file");
                return ExitCode::FAILURE;
            };
            let Ok((_, program)) = compile_file(path) else {
                return ExitCode::FAILURE;
            };
            let mut interp = Interpreter::new();
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
}
