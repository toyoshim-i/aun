---
id: mcs51-backend
title: MCS-51 / WCH CH559 Backend and Built-in Assembler
summary: MCS-51 / E8051 1T codegen, Harvard memory spaces, pointer-space inference, Uchiguri IRAM/XData layout, and optimization roadmap.
category: guide
---
# Aun MCS-51 / WCH CH559 Backend (`aun-mcs51`) v0.1

This document specifies how `aun-mcs51` lowers `aun-ir` to MCS-51 / WCH CH559 (`E8051` 1T) machine code, partitions Harvard memory spaces via pointer-space inference and *Uchiguri* call-graph coloring, assembles Intel HEX binaries with its built-in assembler, and details the roadmap from the v0.1 prototype to production CH559 code quality.

For Z80 specifics see [Z80 Backend](z80-backend.md); for target declarations and space syntax see [Target & Memory Model](target-and-memory.md).

---

## 1. Architectural Differences vs Z80

While `aun-z80` targets a flat 64 KB Von Neumann address space (optionally with banked windows and index registers `IX`/`IY`), CH559 (`E8051` 1T) is a strict Harvard microcontroller with tiny internal RAM and rich bit/SFR instructions:

| Dimension | Z80 / PC-88 (`aun-z80`) | WCH CH559 / E8051 (`aun-mcs51`) |
| --- | --- | --- |
| **Address spaces** | Flat 64 KB + SW/HW banks, `io_in`/`io_out` ports | 6 Harvard spaces: `Data` (128B IRAM), `IData` (128B upper RAM + `IStack`), `XData` (6 KB ERAM), `Code` (ROM), `Sfr`, `Bit` |
| **Working registers** | `A`, `B`, `C`, `D`, `E`, `H`, `L`, `IX`, `IY` | `A`, `B`, `DPTR` (CH559 dual `DPTR0`/`DPTR1`), plus 4 hardware register banks (`R0`..`R7` at IRAM `0x00`..`0x1F`) |
| **Register transfers** | `LD r, r'` directly between any pair | **No `mov rX, rY`**: register-to-register transfers must route through `A` |
| **Hardware math** | No `MUL`/`DIV`; relies on helper/constant shifts | 1T `MUL AB` and `DIV AB` hardware 8-bit multiply/divide (`OptConfig::mcs51()` preserves `u8` pow2 mul/div) |
| **Pointer width & indirection** | Uniform 16-bit `HL`/`IX` | Tagged by memory space (`Data` `@r0/@r1` `mov`, `XData` `@dptr` `movx`, `Code` `@a+dptr` `movc`, or 1-byte `Mixed` tag) |
| **Special restrictions** | `#[smc]` self-modifying code allowed in RAM | `#[smc]` forbidden (Harvard ROM), port I/O (`io_in`/`io_out`) forbidden, `&Sfr`/`&Bit` references forbidden |
| **Stack growth** | Downward `PUSH`/`POP` in arbitrary RAM | Upward `PUSH`/`POP` (`SP` pre-increments) inside internal `Data`/`IData` (`sp_init..=0xFF`) |

---

## 2. Memory Model, Pointer-Space Inference, and *Uchiguri* Layout

### 2.1 Memory Spaces and `Mixed` Pointers
`crates/aun-mcs51/src/layout.rs` assigns every global, parameter/local slot, and pointer value one of `PtrSpace`:
- **`Data` (`0x08..=0x7F`)**: Direct internal RAM (register bank 0 reserves `0x00..=0x07`; any `#[interrupt(..., using(N))]` bank reserves `N*8 .. N*8+7`). Zero-init or non-zero initialized via `data_inits` ROM tables.
- **`IData` (`0x80..=0xFF`)**: Indirect-only upper IRAM holding the hardware `IStack`. `sp_init` is placed immediately above allocated `Data` variables and checked statically against worst-case call depth plus ISR register/PSW/PC saves.
- **`XData` (`0x0000..=0x17FF`)**: CH559 6 KB external/on-chip ERAM for larger buffers, state machines, pools, queues, escaped locals, and overflowed call frames.
- **`Code`**: ROM constants and `__aun_rom_g*` tables read via `movc a, @a+dptr`.
- **`Sfr` / `Bit`**: Fixed `@ addr` volatile memory-mapped peripherals and bit-addressable flags.
- **`Mixed`**: Result of joining pointers/slices across distinct static spaces (e.g. passing `&const [u8]` from both a ROM byte-string literal `Code` and a local RAM buffer `XData`/`Data`). Fixpoint `infer_ptr_spaces` tracks flows through `AddrGlobal`, `AddrLocal` (including `LoadPtr`/`StorePtr` across slice slots), `FieldAddr`, `IndexAddr`, `MakeSlice`/`SlicePtr`, `Cast`, `Phi`, `Call`, `StateTransition`, and `StateDispatch`, allocating 1-byte runtime space tags (`tag_id`: `1=Data/IData`, `2=XData`, `3=Code`) in IRAM/XData.

### 2.2 Call-Graph *Uchiguri* Coloring & Startup Initialization
- Call trees are partitioned into the main execution tree and separate ISR trees (sharing without `#[reentrant]` is rejected).
- Functions in an acyclic call tree reuse IRAM/XData offsets across non-concurrent sibling subtrees via `IrqTree::compute_call_graph_frames`.
- If a tree's `Data` frame demand leaves fewer than 24 bytes of headroom below `idata_range.1`, that tree overflows its parameter/local frame cleanly into `XData` (with `RelRegion::Data(u16)` preventing `u8` offset wrap-around when frames exceed 255 bytes).
- Startup (`__aun_start` at `code_origin.max(0x0080)` after the interrupt vector table `0x0000..=0x007F`) sets `SP`, copies `data_inits` and `xdata_inits`, calls `main`, and halts in `__aun_halt: sjmp __aun_halt`.

---

## 3. Calling Convention, Interrupts, and FFI

- **Internal WPO-ABI**: Parameters and *Uchiguri* slots reside in static function frames (`slot_locs`); 8-bit returns pass in `A`, 16-bit returns in `R2:R3` (`lo:hi`), and 32-bit returns in `R2..R5`.
- **Interrupt Handlers (`#[interrupt(Source, using(N))]`)**: Vector table entries at `0x0003 + irq * 8` emit `ljmp <isr_label>`. Entry pushes `PSW`, `ACC`, `B`, `DPL`, `DPH`, sets `PSW` RS1:RS0 if `using(N)` is present (or saves `0x00..=0x07` on `IStack`), and exits with `reti`.
- **External ASM FFI (`extern "asm"`)**: Direct calls bind single 1B/2B arguments into `A`, `DPL`, `R0`, `R1`, `R7`, `B`, or `DPTR` and call `_<name>`.

---

## 4. Built-in 2-Pass Assembler and `sim559` Verification

- `crates/aun-mcs51/src/asm.rs` implements a built-in 2-pass MCS-51 assembler supporting standard 8051 mnemonics, symbolic labels, `lo(...)` / `hi(...)` byte selectors, `.org` / `.db` / `.equ` directives, named CH559 SFR/Bit registers (`ACC`, `B`, `PSW`, `SP`, `DPL`, `DPH`, `EA`, `TR0`, `ET0`, etc.), and both raw flat `.bin` + **Intel HEX (`.ihx`)** output.
- CLI commands `aun build` and `aun inspect` dispatch automatically to `aun-mcs51` when the file declares `target <name> : mcs51` (or `ch559` / `8051`).
- End-to-end tests in `crates/aun-mcs51/tests/ch559_e2e.rs` compile `.aun` sources directly to `.ihx`, load them into the cycle-accurate WCH CH559 simulator ([`sim559`](https://github.com/toyoshim-i/sim559)), and assert real hardware Timer0 IRQs, GPIO port latches, `XRAM`, `Machine`, `Pool`, and `Mixed` slice behavior.

---

## 5. Current v0.1 Baseline Limitations & Improvement Roadmap

While v0.1 passes initial arithmetic/loop, Timer0/Bank1/Queue ISR, and Machine/Pool/Mixed-slice tests on `sim559`, several Z80-inherited assumptions and prototype shortcuts remain before production CH559 firmware (such as porting `chlib` and USB/UART/GPIO sample firmwares) should rely on it.

### Measured v0.1 Baseline (`cfcdb86`)
- Compiling `ch559_e2e_machine_pool_nonzero_init_and_mixed_slices` with `--release` currently produces **4,143 B Code ROM / 934 B XData / 6 B Data** (`SP=0x0D`): every SSA `ValueId` receives its own linear slot without liveness reuse, causing `main` to overflow IRAM (`Data`) into `XData`, and every `ConstInt` is spilled to memory and reloaded rather than folded into `#imm` operands.

### Planned Phases

#### Phase 1 — Frontend & Correctness Gaps (`#44`)
- **Allow non-zero `static mut` initializers in resident `XData` (`crates/aun-front/src/check/items.rs`)**: Distinguish MCS-51 permanent resident spaces (`Data`/`IData`/`XData`) from Z80 banked memory so `static mut X: T in XData = <non-zero>` is accepted alongside `Data` globals.
- **Bank-independent 32-bit math helpers**: Replace the dummy `__aun_mul32` stub and add `__aun_udiv32` / `__aun_sdiv32` (essential for CH559 `FREQ_SYS / baud` UART divisor computation and 32-bit timers).
- **Indirect calls & reentrant stack frames**: Implement `IrInst::CallIndirect` and `#[reentrant]` IRAM `IStack` frame spilling.
- **Safe multi-arg `ExternCall`**: Stage/order parameter moves so loading subsequent arguments never clobbers `A` or `DPTR`, and support `r2..r7` / `u32` bindings.

#### Phase 2 — IRAM Pressure & 8051-Native Instruction Selection (`#45`)
- **Intra-function liveness interval coloring (`layout.rs`)**: Reuse dead temporary `ValueId` slots and allocate short-lived temporaries in register bank `R2..R7` so normal functions stay inside 128 B IRAM (`Data`) rather than falling back to `mov dptr; movx` in `XData`.
- **Immediate `ConstInt` folding (`codegen.rs`)**: Fold known constants into `#imm` instructions (`add a, #imm`, `cjne`, `mov direct, #imm`) instead of storing/reloading every constant through RAM.
- **Native 8051 Bit & SFR lowering**: Emit `jb`/`jnb bit, label` for `Bit` branches, single-instruction `orl`/`anl`/`xrl direct, #imm` for SFR read-modify-write, and `djnz` for bounded countdown loops.

#### Phase 3 — Typed MCS-51 LIR, Peepholes & CH559 Hardware Features (`#46`)
- **MCS-51 LIR & Peephole Pass**: Eliminate redundant `mov dptr` / store-then-reload sequences via `A`/`DPTR`/`R0..R7` tracking, remove dead stores, and relax `ljmp`/`lcall` to `sjmp`/`ajmp`/`acall`.
- **Honor `CompileOptions::release`**: Gate MIR inlining/size tuning and peepholes properly in `codegen::compile`.
- **CH559 Hardware Acceleration & 1T Timing**: Use CH559 `XBUS_AUX` dual-`DPTR` auto-increment (`0xA5` `MOVX @DPTR1++, A`) in `__aun_memcpy_xx`, and add a static 1T E8051 cycle/WCET estimator aligned with `sim559` for `aun inspect`.

#### Validation Milestone — `chlib` Porting & Sample Firmware on `sim559`
- Once Phases 1–2 land, port WCH CH559 HAL modules (`chlib`: system clock/PLL, `Delay`, `UART0`/`UART1`, `GPIO`, `Timer`, `PWM`, `SPI`, `USB`) and sample firmwares to Aun and verify hardware behavior, UART/GPIO output, and IEEE 1364 VCD traces directly in `sim559`.

For broader compiler pipeline details, see [Compiler Internals](internals.md).
