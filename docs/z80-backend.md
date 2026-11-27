---
id: z80-backend
title: Z80 Backend, WPO-ABI, and Built-in Linker
summary: Z80 codegen, register allocation, peephole/outliner, SDCC --sdcccall 1 FFI, and external .s linking.
category: guide
---
# Aun Z80 Backend, WPO-ABI, and Built-in Assembler/Linker (`aun-z80`) v0.1

This document specifies how `aun-z80` lowers `aun-ir` to Z80 machine code, optimizes calling conventions via whole-program analysis, assembles both Aun-generated code and existing SDASZ80 (`.s`) files, and links resident and overlay binaries in one pass.

---

## 1. Whole-Program Optimizing ABI (WPO-ABI)

Because Aun compiles the entire program as a single unit (AD-5), internal functions do not need a rigid, stack-heavy C calling convention.

### 1.1 Internal Calling Convention
- **Register-first parameter passing**:
  - 1st 8-bit arg -> `A`; 2nd -> `E` (or `B`/`C`); 1st 16-bit arg -> `HL`; 2nd -> `DE`; 3rd -> `BC`.
  - Return values: 8-bit -> `A`; 16-bit -> `HL`; 32-bit -> `DEHL`.
  - Remaining arguments (if registers are exhausted) are placed in the callee's *Uchiguri* static parameter slots (never pushed to the stack unless `#[reentrant]`).
- **Whole-Program Register Clobber Propagation**:
  - Bottom-up traversal of the call graph computes the exact `clobbers` set of every internal function.
  - Callers only preserve registers that are *both* live across the call and actually modified by the callee—eliminating redundant `PUSH`/`POP` pairs.
- **Leaf & Single-Caller Functions**:
  - Leaf functions with few live values avoid RAM entirely.
  - Single-caller functions are candidates for automatic inlining when size/speed trade-offs are favorable.
- **Register-Explicit FFI (`extern "asm"`)**:
  - Uses the exact register bindings and `clobbers(...)` declared in the `extern "asm"` signature.

### 1.2 `#[reentrant]` Functions
- Functions explicitly annotated `#[reentrant]` spill live locals across recursive calls to the hardware stack (via `PUSH`/`POP` or `IX` frame only when needed), and contribute `frame_bytes * max_depth` to the static stack budget.

---

## 2. Z80 Instruction Selection & Idioms

### 2.1 Fixed-Point Arithmetic (`q8_8`, `q10_6`, `fixed<B, F>`)
- **`q8_8::int()`**: Direct read of high byte (`LD A, H` or `LD A, (addr + 1)`) — **0T shift**.
- **`q10_6::int()`**: Optimized 2-bit left shift (`ADD HL, HL; ADD HL, HL`) followed by sign-preserving extraction of `H` (or 16-bit shift when full `i16` range `[-512, 511]` is needed).
- **`fixed` multiply (`q8_8 * q8_8`)**: Calls a compact, specialized resident helper (or inline shift-add sequence when one operand is a compile-time constant).
- **Saturating arithmetic (`+|`, `-|`)**:
  - Unsigned `u8 +|`: `ADD A, r; JR NC, $+4; LD A, 0xFF` (compact 4 bytes).
  - Unsigned `u8 -|`: `SUB r; JR NC, $+3; XOR A` (3 bytes).
  - Signed `i8`/`i16` saturating ops check the `P/V` overflow flag (`JP PO, ...`) and clamp to `0x7F`/`0x80` (`0x7FFF`/`0x8000`).

### 2.2 State Machine Dispatch
- **Standard (`JP (HL)`)**:
  - The state tag field stores a 16-bit pointer to the current state's handler (if 1 dispatched method) or state vtable (if multiple dispatched methods).
  - Dispatch sequence:
    ```z80
    LD HL, (ball_state)   ; 16T, 3B (single-method machine)
    JP (HL)               ;  4T, 1B
    ```
- **Self-Modifying Code (`#[smc]` in RAM)**:
  - When a `machine` is marked `#[smc]` and resides in writable RAM, the transition updates the 2-byte immediate target of a `JP xxxx` instruction directly:
    ```z80
    ball_step_dispatch:
        JP state_in_shooter_step  ; 10T, 3B (overwritten on transition)
    ```
  - Saves 10 T-states and frees `HL` at every dispatch site.

### 2.3 `Pool<T, N>` SoA Indexing (`#[layout(soa, align(256))]`)
- Each field array is placed at a 256-byte boundary `0xXX00`.
- Element access with 1-byte `Handle<T>` in `A` (or `L`):
  ```z80
  LD H, hi(BALLS_X_LO)
  LD L, A
  LD E, (HL)              ; low byte of q10_6
  INC H                   ; next 256B page if packed, or LD H, hi(BALLS_X_HI)
  LD D, (HL)              ; high byte of q10_6
  ```
- Iteration over contiguous SoA arrays uses `INC L` (4T) instead of 16-bit pointer addition (`ADD HL, DE`), matching hand-tuned Z80 assembly.

### 2.4 Peephole Optimizer
Post-selection Z80 peephole pass optimizes common sequences:
- `LD A, 0` -> `XOR A` (when flags are dead)
- `CP 0` -> `OR A`
- `CALL foo; RET` -> `JP foo` (tail-call optimization)
- `LD HL, (x); LD DE, (y)` / `EX DE, HL` elimination
- Redundant `LD r, r` and dead flag/register stores
- `JP cc, label` -> `JR cc, label` branch relaxation when displacement fits in `[-128, +127]` (iterated to fixed point during assembly).

---

## 3. Built-in Z80 Assembler (with SDASZ80 `.s` Compatibility)

Per AD-5 and our assembly interop decision, `aun-z80` embeds a complete Z80 assembler supporting both Aun's internal assembly IR and **existing SDASZ80 / ASXXXX (`.s`) source files** (such as `bbb88`'s `hal_pc88.s` and `mdxnt`).

### 3.1 Supported Z80 Instruction Set
- Full documented Z80 ISA (8-bit/16-bit loads, ALU, rotates/shifts, bit ops `CB`, block ops `ED`, `IX`/`IY` indexed ops `DD`/`FD`, `IM 0/1/2`, `RETI`/`RETN`).
- Undocumented half-index register ops (`IXH`, `IXL`, `IYH`, `IYL`) and `SLL` supported optionally via flag (useful for hand-written Z80 libraries).

### 3.2 SDASZ80 (`.s`) Compatibility Layer
To assemble existing `.s` files directly without manual conversion, the assembler parser supports core SDASZ80 / ASXXXX syntax and directives:
- **Directives**: `.area`, `.module`, `.globl`, `.local`, `.org`, `.db` / `.byte`, `.dw` / `.word`, `.ds` / `.blkb`, `.ascii`, `.asciz`, `.equ` / `=`, `.include`, `.if` / `.else` / `.endif`, `.macro` / `.endm`.
- **Syntax conventions**:
  - Immediate prefix `#expr` (e.g., `ld a, #0x40`, `ld hl, #_label`) alongside standard Zilog syntax (`ld a, 0x40`).
  - Indexed offset syntax `offset(ix)` / `offset(iy)` alongside `(ix + offset)` / `(iy + offset)`.
  - High/low byte operators `<expr` (low byte) and `>expr` (high byte).
  - Local numeric labels (`1$:`, `2$:`) scoped between non-local labels.
  - Hex literals `0xNN` and `$NN`.
- **Area Mapping**: `.area _CODE`, `_DATA`, `_BSS`, etc. are mapped to Aun target `space` / `bank` regions via `aun.toml` (with sensible defaults: `_CODE` -> resident code, `_DATA`/`_BSS` -> resident RAM).

---

## 4. Whole-Program Linker & T-State Estimator

### 4.1 Single-Pass Multi-Binary Linking
1. **Symbol Resolution**: Merges Aun-generated object sections and assembled `.s` object sections into one global symbol table.
2. **Placement**:
   - Places fixed-address sections (`@ 0xXXXX`, interrupt vector entries) first.
   - Places 256B-aligned SoA `Pool` arrays on page boundaries with minimal padding waste (backfilling smaller data into alignment gaps!).
   - Places resident code/data in `MainRam` (and designated banks), and each `#[overlay(Slot)]` state's private code/data at the slot's base address.
3. **Branch Relaxation**: Iteratively shrinks `JP` to `JR` where safe, recalculating addresses until convergence.
4. **Verification**:
   - Checks every `space`, `bank`, and `overlay_slot` against its upper bound (failing with exact overflow byte count if exceeded).
   - Verifies static stack budget (`MaxStack <= size(stack)`).
5. **Output**: Emits `resident.bin`, `ovl_*.bin`, `.asm` listing, `.sym`, and `manifest.json`.

### 4.2 Static T-State Estimator (`aun inspect`)
- Every assembled Z80 instruction has exact base T-state costs (taken/not-taken for conditional branches `JR`/`JP`/`CALL`/`RET`, and M1 wait-state profile configurable per target, e.g., PC-8801mkIISR 4MHz + VRAM wait).
- `aun inspect --tstates <fn>` reports best-case / worst-case T-states per basic block and function, and annotates each line in the emitted `.asm` dump.

---

## 6. Relocatable SDASZ80 Output (`--format sdasz80`) & Implemented Optimizations

For incremental migration into existing SDCC / `sdldz80` Z80 projects, `aun build --format sdasz80 [--release] [-o <dir>] <file.aun>` emits relocatable SDASZ80 `.s` assembly (`crates/aun-z80/src/sdasz80.rs`):

- **Areas**: `.area _DATA` with a single `__aun_ram: .ds N` reservation (all fixed *Uchiguri* RAM references rewritten to `(__aun_ram + off)`), and `.area _CODE` containing ROM `const` tables, functions, and `.globl` exports.
- **SDCC `--sdcccall(1)` C ABI Wrappers**: Each `pub fn` exports `_name::` with automatic `ex de, hl` for 2-byte returns and a general `iy`/`bc` stack-arg unpacker for functions with >2 parameters, preserving callee-saved `IX`.
- **Register Promotion & WPO**:
  - `IX` reference-parameter promotion (`detect_ix_param`) with lazy same-block `(ix + d)` loads (`add/sub/and/or/xor/cp (ix + d)`, 16-bit `ld e,(ix+d); ld d,(ix+d+1)`).
  - `C` hot-local promotion (`compute_reg_c_slot`) paired with whole-program call-graph `compute_fn_clobbers_c` to skip `push bc`/`pop bc` around internal calls that preserve `C`.
  - Shared `__aun_mul_hl_<K>` helpers for repeated non-power-of-two struct strides (`k >= 5`).
- **Peephole Convergence Loop**: 7-stage fixed-point optimizer (`crates/aun-z80/src/peephole.rs`) including CFG-aware `RegCache`, dead private/single-block scratch store elimination, `inc/dec (ix+d)` / `inc/dec (hl)` / `inc/dec c`, `BC` spill promotion, `ret`-tail merging, and `jp`->`jr` relaxation.

For full implementation details across all 5 compiler layers, see [Compiler Internals](internals.md).
