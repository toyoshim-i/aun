//! `aun` command-line driver (`aun build`, `aun test`, `aun inspect`).

use std::env;
use std::process::ExitCode;

fn print_usage() {
    eprintln!(
        "aun {} — Aun language compiler and Z80 toolchain\n\n\
         Usage:\n  \
           aun build [--release] [--checks=<list>] [path]\n  \
           aun test [--trace] [path]\n  \
           aun inspect [--tstates <fn>] [path]\n  \
           aun --version",
        env!("CARGO_PKG_VERSION")
    );
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
            let program = aun_ir::IrProgram::default();
            match aun_z80::compile_z80(&program) {
                Ok(out) => {
                    println!("aun build: emitted {} bytes (skeleton)", out.resident_bin.len());
                    ExitCode::SUCCESS
                }
                Err(err) => {
                    eprintln!("error: {err}");
                    ExitCode::FAILURE
                }
            }
        }
        Some("test") => {
            let program = aun_ir::IrProgram::default();
            let mut interp = aun_interp::Interpreter::new();
            let results = interp.run_tests(&program);
            println!("aun test: {} test(s) executed", results.len());
            ExitCode::SUCCESS
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
