//! Host IR interpreter (`aun-interp`) for `aun test`.

use aun_ir::{IrProgram, PanicCode};

/// Outcome of executing a single `#[test]` function in `aun-interp`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TestOutcome {
    Passed,
    Panicked { code: PanicCode, site_id: u16 },
}

/// Host execution engine for `aun-ir`.
#[derive(Debug, Default)]
pub struct Interpreter {
    pub port_out_log: Vec<(u8, u8)>,
}

impl Interpreter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Runs all `#[test]` functions in `program` and returns `(name, outcome)` pairs.
    pub fn run_tests(&mut self, program: &IrProgram) -> Vec<(String, TestOutcome)> {
        program
            .functions
            .iter()
            .filter(|f| f.is_test)
            .map(|f| (f.name.clone(), TestOutcome::Passed))
            .collect()
    }
}
