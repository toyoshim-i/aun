# Aun Intermediate Representation (`aun-ir`) and Host Interpreter (`aun-interp`) v0.1

This document specifies the target-independent middle-end IR (`aun-ir`), its optimization pipeline, and the built-in host interpreter (`aun-interp`) powering `aun test`.

---

## 1. Design Goals of `aun-ir`

1. **Target-independent**: Represents Z80, 8086, and 68000 programs without hard-coding CPU registers (except inside `ExternCall` / `InlineAsm` nodes).
2. **Typed SSA with Explicit Memory Places**: Uses SSA virtual registers for values and explicit static memory places (`GlobalId`, `LocalSlotId`, `StateField`) so *Uchiguri* RAM overlay allocation and SoA lowering operate cleanly before instruction selection.
3. **First-Class Domain Ops**: Retains high-level semantics for state transitions, bank scopes, and dynamic checks through middle-end passes so they can be optimized or eliminated before lowering to machine code.

---

## 2. IR Structure

### 2.1 Program Module (`IrProgram`)
- `types`: Lowered type table (`U8`, `I8`, `U16`, `I16`, `U32`, `I32`, `Bool`, `Fixed(Base, Frac)`, `Struct`, `Array`, `Pool`, `Machine`, `ConstSlice`).
- `globals`: Static variables with placement (`SpaceId` / `BankId`, alignment, initial constant bytes).
- `machines`: State machine metadata (shared fields, state layouts, transition bitmasks, handler dispatch tables, overlay slot bindings).
- `functions`: Control-flow graphs of basic blocks (`BasicBlock`) with SSA values (`ValueId`), local slots (`SlotId`), calling attributes (`#[reentrant]`, `#[interrupt]`, `requires bank/di`), and source spans (`SpanId` -> `site_id`).
- `externs`: Register-explicit FFI declarations (`ExternFn`).
- `target`: Parsed target memory spaces, bank groups, interrupt controllers, and overlay slots.

### 2.2 Core Instructions (`IrInst`)

#### Value & Arithmetic Ops
- `Const(Literal)`
- `BinOp { op, mode, lhs, rhs }`
  - `op`: `Add`, `Sub`, `Mul`, `Div`, `Mod`, `And`, `Or`, `Xor`, `Shl`, `Shr`, `Eq`, `Ne`, `Lt`, `Le`, `Gt`, `Ge`
  - `mode`: `Checked` (default), `Wrapping`, `Saturating`
- `UnOp { op: Neg | Not | BitNot, mode, val }`
- `Cast { val, from_ty, to_ty }` (covers integer width casts and fixed-point scaling)
- `FixedInt { val, frac_bits }` / `FixedFrac { val, frac_bits }`

#### Memory & Pool Ops
- `Load { addr: Place }` / `Store { addr: Place, val }`
- `FieldAddr { base, field_idx }` / `IndexAddr { base, index, elem_ty }`
- `PoolAlloc { pool, init }` / `PoolFree { pool, handle }` / `PoolFieldAddr { pool, handle, field_idx }`

#### State Machine Ops (Before Dispatch Lowering)
- `StateDispatch { machine_ptr, method_id, args }`
- `StateTransition { machine_ptr, from_state: Option<StateId>, to_state: StateId, payload, overlay_slot: Option<OverlaySlotId> }`
  - `from_state: Some(s)` = statically verified (no runtime check).
  - `from_state: None` = dynamic transition (paired with `CheckTransition` in Debug).

#### Hardware Effect & FFI Ops
- `BankEnter { group, target_banks, needs_di }` / `BankExit { group, restore_banks, restore_ei }`
- `PortOut { port, val }` / `PortIn { port }`
- `Call { func, args }`
- `ExternCall { extern_fn, args }`
- `InlineAsm { template, clobbers }`

#### Dynamic Check Ops (Zero-Cost Removable)
Every check carries a `CheckKind` (`Bounds`, `Transition`, `Overflow`, `Bank`, `Handle`) and a unique `site_id: u16` mapped to `file:line:col` in `.sym`:
- `CheckBounds { index, len, site_id }`
- `CheckHandle { pool, handle, site_id }`
- `CheckTransition { machine_ptr, target_state, site_id }`
- `CheckOverflow { cond_ok, site_id }` (or lowered from `BinOp(mode: Checked)`)
- `CheckBank { group, expected_banks, site_id }`
- `Panic { code: PanicCode, site_id }`

#### Basic Block Terminators (`Terminator`)
- `Goto(BlockId, args)` (block parameters instead of Phi nodes)
- `Branch { cond, then_bb, else_bb }`
- `Switch { val, cases, default_bb }`
- `Return(Option<ValueId>)`
- `Unreachable` (after `Panic` or diverging `goto`)

---

## 3. Middle-End Pass Pipeline (`aun-ir`)

Passes execute in a deterministic order:

1. **Check Filtering (`strip_checks`)**:
   - Removes `Check*` nodes whose category is disabled by CLI flags (`--release`, `--checks=...`) or function/module `#[checks(...)]` attributes, and downgrades `BinOp(Checked)` to `BinOp(Wrapping)` when `overflow` checks are disabled.
2. **Pool SoA Lowering (`lower_pool_soa`)**:
   - Rewrites `Pool<T, N>` annotated with `#[layout(soa, align(256))]` into separate 256B-aligned static field arrays and replaces `PoolFieldAddr` with direct base+handle indexing.
   - Emits debug `alive_bits` side-table checks only when `bounds` checks are active.
3. **State Machine Lowering (`lower_machines`)**:
   - Lays out shared fields, dispatch pointer tag, and mutually exclusive state-local overlay slots.
   - Lowers `StateTransition` into payload evaluation -> `on_exit` (bottom-up HSM) -> overlay load (if `#[overlay]`) -> state init & tag write -> `on_enter` (top-down HSM).
4. **Bank & Interrupt Optimization (`opt_banks`)**:
   - Hoists `BankEnter`/`BankExit` out of loops when no conflicting memory accesses occur inside the loop.
   - Coalesces adjacent identical bank scopes and removes redundant switches when `BankState` already matches.
5. **Scalar Optimizations (`opt_scalar`)**:
   - Constant folding (including fixed-point literals and shifts), strength reduction (multiply/divide by power-of-two -> shifts), dead code elimination, copy propagation, and small-function inlining.
6. **Whole-Program Call Graph & *Uchiguri* Allocation (`uchiguri_alloc`)**:
   - Computes static reachability from `main` and each `#[interrupt]` ISR.
   - Partitions code/data into resident vs. `#[overlay]` regions.
   - Builds the interference graph across call-tree locals, state trees, and ISRs, assigning concrete relative offsets within each RAM space.
   - Verifies worst-case stack depth against `stack` budgets.

---

## 4. Host Interpreter (`aun-interp`) & `aun test`

`aun-interp` executes `aun-ir` directly on the host machine (before target-specific Z80 lowering, or optionally after `lower_machines` / `lower_pool_soa` to test lowered semantics) with zero external dependencies.

### 4.1 Test Syntax
```aun
#[test]
fn ball_launches_from_shooter() {
    let mut ball = Ball::new();
    ball.goto(InShooter { power: 10 });
    let inp = InputState { plunger: 0 };
    ball.step(&inp);
    assert!(matches!(ball, Ball::InPlay(_)));
}

#[test]
#[should_panic(InvalidTransition)]
fn reject_illegal_ball_transition() {
    let mut ball = Ball::new(); // starts in Inactive
    ball.goto(MagnetHeld { hold_timer: 10 }); // must trap
}
```

### 4.2 Mocking Hardware I/O and `extern "asm"` FFI
To test hardware-facing logic on the host without an emulator:
- **Mock FFI**: Functions marked `#[mock(hal_wait_vsync)]` inside `#[cfg(test)]` modules substitute for `extern "asm"` declarations during `aun test`.
- **Mock I/O Ports**: `aun-interp` maintains a virtual 256-port I/O bus; tests can register port read/write hooks or inspect recorded `out(port, val)` sequences.
- **Virtual Memory Spaces & Banks**: `aun-interp` enforces bank mapping state dynamically, catching any out-of-bank access even across dynamic boundaries.

### 4.3 Rich Diagnostics & Transition Traces
When a test fails or traps:
- Prints the exact `PanicCode`, `site_id`, source file/line, and Aun call stack.
- Prints a **state transition history** for every active `machine` (e.g., `Ball: Inactive -> InShooter (step 1) -> InPlay (step 4) -> [INVALID: MagnetHeld]`).
- `aun test --trace` dumps full state transition logs and virtual port I/O events for debugging.
