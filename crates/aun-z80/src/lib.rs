//! Z80 backend: WPO-ABI, instruction selection, *Uchiguri* static RAM overlay,
//! built-in SDASZ80-compatible assembler, and linker.

pub mod asm;
pub mod codegen;
pub mod uchiguri;

use aun_ir::IrProgram;
use std::collections::HashMap;

/// Result of assembling and linking a Z80 program.
#[derive(Debug, Default)]
pub struct LinkOutput {
    pub resident_bin: Vec<u8>,
    pub asm_listing: String,
    pub symbols: HashMap<String, u16>,
    pub ram_bytes: u16,
    pub max_stack_bytes: u16,
    pub total_tstates: u32,
}

/// Compiles an `IrProgram` (and optional external SDASZ80 `.s` sources) into Z80 machine code.
pub fn compile_z80(program: &IrProgram) -> Result<LinkOutput, String> {
    compile_z80_with_asm(program, &[], 0x0100, 0xC000)
}

pub fn compile_z80_with_asm(
    program: &IrProgram,
    external_asm_sources: &[&str],
    code_origin: u16,
    ram_base: u16,
) -> Result<LinkOutput, String> {
    let layout = uchiguri::allocate_uchiguri(program, ram_base)?;
    let mut full_asm = codegen::emit_z80_asm(program, &layout, code_origin)?;
    for ext in external_asm_sources {
        full_asm.push('\n');
        full_asm.push_str(ext);
    }
    let assembled = asm::assemble_z80(&full_asm, code_origin)?;
    Ok(LinkOutput {
        resident_bin: assembled.bytes,
        asm_listing: full_asm,
        symbols: assembled.symbols,
        ram_bytes: layout.ram_end.saturating_sub(layout.ram_base),
        max_stack_bytes: layout.max_stack_bytes,
        total_tstates: assembled.total_tstates,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use aun_ir::*;

    #[test]
    fn compile_and_link_with_external_sdasz80_ffi() {
        // Build a tiny IR program calling an extern "asm" routine `hal_out_fe(v: u8 in A)`
        let mut prog = IrProgram::default();
        prog.externs.push(IrExternFn {
            name: "hal_out_fe".into(),
            params: vec![("v".into(), IrType::Int(IntType::U8), "A".into())],
            ret: None,
            clobbers: vec!["F".into()],
            requires_di: false,
            requires_banks: vec![],
        });
        prog.functions.push(IrFunction {
            id: 0,
            name: "main".into(),
            param_count: 0,
            locals: vec![],
            ret_ty: IrType::Unit,
            is_test: false,
            expected_panic: None,
            is_reentrant: false,
            blocks: vec![BasicBlock {
                id: 0,
                insts: vec![
                    IrInst::ConstInt {
                        dst: 0,
                        ty: IrType::Int(IntType::U8),
                        val: 0x55,
                    },
                    IrInst::ExternCall {
                        dst: None,
                        extern_fn: 0,
                        args: vec![0],
                    },
                ],
                term: Terminator::Return(None),
            }],
            next_val: 1,
        });

        let hal_s = r#"
            .area _CODE
            .globl hal_out_fe
        hal_out_fe:
            out (0xFE), a
            ret
        "#;

        let out = compile_z80_with_asm(&prog, &[hal_s], 0x0100, 0xC000)
            .expect("Z80 compile + link should succeed");
        assert!(out.symbols.contains_key("main"));
        assert!(out.symbols.contains_key("hal_out_fe"));
        assert!(!out.resident_bin.is_empty());
        assert!(out.total_tstates > 0);
    }
}
