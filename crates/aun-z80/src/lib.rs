//! Z80 backend: WPO-ABI, instruction selection, built-in SDASZ80-compatible assembler, and linker.

use aun_ir::IrProgram;

/// Result of assembling and linking a Z80 program.
#[derive(Debug, Default)]
pub struct LinkOutput {
    pub resident_bin: Vec<u8>,
    pub asm_listing: String,
}

/// Compiles an `IrProgram` into Z80 machine code.
pub fn compile_z80(_program: &IrProgram) -> Result<LinkOutput, String> {
    Ok(LinkOutput::default())
}
