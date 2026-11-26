# Aun

**Aun** is a systems programming language and integrated toolchain designed for the Z80 (with future support for 8086 and 68000), centered on **first-class state machines** and **static memory layout**.

## Overview

- **Z80-First Type System**: Strict integer widths (`u8`, `i8`, `u16`, `i16`, `u32`, `i32`, `bool`) with no implicit 16-bit promotion, first-class fixed-point types (`q8_8`, `q10_6`, `fixed<Base, FracBits>`), second-class references (`&T` / `&mut T`), ROM slices (`&const [T]`), and SoA-capable `Pool<T, N>` with 1-byte `Handle<T>`.
- **First-Class State Machines**: Built-in `machine`, `state`, and `transitions` with zero-overhead static transition verification and optional debug-only dynamic checks.
- **Static RAM Overlay (*Uchiguri*)**: Whole-program call-graph and state-tree interference analysis assigns non-overlapping static RAM addresses without stack frames.
- **Hardware-Aware Target Definitions**: Declarative memory spaces, bank groups, interrupt controllers, and code overlay slots checked at compile time.
- **Integrated Toolchain**: Written in Rust with a built-in IR interpreter (`aun test`), whole-program optimizing Z80 backend, and integrated assembler/linker.

## Documentation

- [Design Specification v1.0](docs/design-spec.md) — Architectural decisions, overview, and roadmap
- [Language Reference v0.1](docs/language-reference.md) — Grammar, types, fixed-point, borrow rules, `Pool`/`Handle`, state machines, and FFI
- [Target & Memory Specification v0.1](docs/target-and-memory.md) — Target DSL, *Uchiguri* static RAM overlay, bank/interrupt safety, and code overlays
- [IR & Interpreter Specification v0.1](docs/ir-and-interpreter.md) — `aun-ir`, optimization passes, and `aun test` host interpreter
- [Z80 Backend & Assembler Specification v0.1](docs/z80-backend.md) — WPO-ABI, Z80 codegen, built-in SDASZ80-compatible assembler, linker, and T-state estimator

## Status

Early development. See the [Implementation Roadmap](docs/design-spec.md#4-implementation-roadmap-rust-workspace-layout) for planned milestones.
