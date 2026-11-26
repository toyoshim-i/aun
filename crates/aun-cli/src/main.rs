//! `aun` command-line driver (`aun build`, `aun test`, `aun inspect`).

use aun_interp::{Interpreter, TestOutcome};
use std::env;
use std::fs;
use std::process::ExitCode;

fn print_usage() {
    eprintln!(
        "aun {} — Aun language compiler and Z80 toolchain\n\n\
         Usage:\n  \
           aun build [--release] [--checks=<list>] <file.aun>\n  \
           aun test [--trace] <file.aun>\n  \
           aun inspect [--tstates <fn>] <file.aun>\n  \
           aun --version",
        env!("CARGO_PKG_VERSION")
    );
}

fn compile_file(path: &str) -> Result<aun_ir::IrProgram, ExitCode> {
    let src = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: failed to read `{path}`: {e}");
            return Err(ExitCode::FAILURE);
        }
    };
    match aun_front::compile_str(0, &src) {
        Ok(prog) => Ok(prog),
        Err(diags) => {
            for d in diags {
                eprintln!("{d}");
            }
            Err(ExitCode::FAILURE)
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
        Some("build") => {
            let Some(path) = args.iter().skip(1).find(|a| !a.starts_with('-')) else {
                eprintln!("error: `aun build` requires a `.aun` source file");
                return ExitCode::FAILURE;
            };
            let Ok(program) = compile_file(path) else {
                return ExitCode::FAILURE;
            };
            match aun_z80::compile_z80(&program) {
                Ok(out) => {
                    println!("aun build: emitted {} bytes", out.resident_bin.len());
                    ExitCode::SUCCESS
                }
                Err(err) => {
                    eprintln!("error: {err}");
                    ExitCode::FAILURE
                }
            }
        }
        Some("test") => {
            let Some(path) = args.iter().skip(1).find(|a| !a.starts_with('-')) else {
                eprintln!("error: `aun test` requires a `.aun` source file");
                return ExitCode::FAILURE;
            };
            let Ok(program) = compile_file(path) else {
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
        Some("inspect") => {
            println!("aun inspect: not yet implemented");
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("error: unknown subcommand `{other}`");
            print_usage();
            ExitCode::FAILURE
        }
    }
}
