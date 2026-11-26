//! `aun` command-line driver (`aun build`, `aun test`, `aun inspect`).

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
           aun build [-o <dir>] [--release] [--checks=<list>] <file.aun>\n  \
           aun test [--trace] <file.aun>\n  \
           aun inspect [--tstates <fn>] <file.aun>\n  \
           aun --version",
        env!("CARGO_PKG_VERSION")
    );
}

fn compile_file(path: &str) -> Result<(String, IrProgram), ExitCode> {
    let src = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: failed to read `{path}`: {e}");
            return Err(ExitCode::FAILURE);
        }
    };
    match aun_front::compile_str(0, &src) {
        Ok(prog) => Ok((src, prog)),
        Err(diags) => {
            for d in diags {
                eprintln!("{d}");
            }
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
    let mut source_path: Option<String> = None;
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
                source_path = Some(file.to_string());
                i += 1;
            }
        }
    }

    let mut opts = if release {
        aun_z80::CompileOptions::release()
    } else {
        aun_z80::CompileOptions::debug()
    };
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

    let Some(path) = source_path else {
        eprintln!("error: `aun build` requires a `.aun` source file");
        return ExitCode::FAILURE;
    };

    let Ok((src, program)) = compile_file(&path) else {
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

    let out = match aun_z80::compile_z80_with_options(&program, &[], opts) {
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
        Some(other) => {
            eprintln!("error: unknown subcommand `{other}`");
            print_usage();
            ExitCode::FAILURE
        }
    }
}
