pub mod asm;
pub mod codegen;
pub mod layout;

use aun_ir::IrProgram;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct CompileOptions {
    pub release: bool,
    pub check_bounds: bool,
    pub check_transitions: bool,
    pub check_overflow: bool,
    pub check_bank: bool,
    pub code_origin: u16,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self::debug()
    }
}

impl CompileOptions {
    pub fn debug() -> Self {
        Self {
            release: false,
            check_bounds: true,
            check_transitions: true,
            check_overflow: true,
            check_bank: true,
            code_origin: 0x0000,
        }
    }

    pub fn release() -> Self {
        Self {
            release: true,
            check_bounds: false,
            check_transitions: false,
            check_overflow: false,
            check_bank: false,
            code_origin: 0x0000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Mcs51Output {
    pub resident_bin: Vec<u8>,
    pub intel_hex: String,
    pub asm_listing: String,
    pub symbols: HashMap<String, u16>,
    pub data_bytes: u16,
    pub idata_stack_bytes: u16,
    pub xdata_bytes: u16,
    pub sp_init: u8,
}

pub fn compile_mcs51(program: &IrProgram, opts: CompileOptions) -> Result<Mcs51Output, String> {
    codegen::compile(program, opts)
}
