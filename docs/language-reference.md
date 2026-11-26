# Aun Language Reference v0.1

This document specifies the lexical structure, grammar, type system, borrow rules, first-class state machines, and FFI semantics of **Aun** (`.aun`).

---

## 1. Lexical Structure

### 1.1 Comments and Whitespace
- Line comments: `// ...`
- Block comments (nestable): `/* ... */`
- Whitespace: ASCII space, tab, `\r`, `\n`.

### 1.2 Identifiers and Keywords
- Identifiers match `[A-Za-z_][A-Za-z0-9_]*`.
- Reserved keywords:
  ```text
  as       asm      bank     bank_group  bool     break    const
  continue default  di       ei          else     enum     extern
  false    fn       for      goto        if       in       interrupt_controller
  let      loop     machine  match       mod      mut      on
  overlay_slot      pub      requires    return   self     Self
  space    stack    state    struct      switch   target   transitions
  true     type     use      while       with
  ```

### 1.3 Literals
- **Integer**: decimal (`42`, `1_000`), hex (`0xFF`, `0xC000`), binary (`0b1010_0101`). Optional type suffix (`42u8`, `-12i16`, `0x8000u16`).
- **Fixed-point**: decimal with fractional part (`0.0`, `1.5`, `-0.25`). Type is inferred from context (`q8_8`, `q10_6`, or `fixed<Base, FracBits>`) or suffixed (`1.5q8_8`, `0.25q10_6`). Rounded to nearest representable value at compile time; out-of-range literals are compile-time errors.
- **Boolean**: `true`, `false`.
- **Byte / String**: `b'A'` (`u8`), `"hello"` (`&const [u8]`, not null-terminated unless explicitly written `"hello\0"`).

---

## 2. Type System

### 2.1 Primitive Types
| Type | Size | Representation | Notes |
| :--- | :---: | :--- | :--- |
| `u8` / `i8` | 1 B | Unsigned / two's-complement signed 8-bit | Native Z80 register width |
| `u16` / `i16` | 2 B | Little-endian 16-bit | Register pair (`HL`, `DE`, `BC`) |
| `u32` / `i32` | 4 B | Little-endian 32-bit | Software / paired ops |
| `bool` | 1 B | `0` (`false`) or `1` (`true`) | |
| `()` | 0 B | Unit type | Zero-sized |
| `!` | 0 B | Never type (diverging) | Produced by `return`, `break`, `continue`, `goto`, panic |

**Strict width rule**: No implicit integer promotion or narrowing. Binary operators require both operands to have the exact same type. Explicit casts use `expr as Type`.

### 2.2 Fixed-Point Types
- `fixed<Base, FracBits>` where `Base` is a signed or unsigned integer type (`i8`, `u8`, `i16`, `u16`, `i32`, `u32`) and `0 < FracBits < bits(Base)`.
- Standard aliases:
  - `q8_8 = fixed<i16, 8>` (range `[-128.0, 127.99609375]`, step `1/256`)
  - `q10_6 = fixed<i16, 6>` (range `[-512.0, 511.984375]`, step `1/64`)
- Built-in methods:
  - `v.raw() -> Base`: returns the underlying scaled integer.
  - `Type::from_raw(raw: Base) -> Self`: constructs from a scaled integer.
  - `v.int() -> Base` (or narrowed integer via `v.int_i8()` / `v.int_i16()`): arithmetic right shift by `FracBits` (floor towards $-\infty$). On Z80, `q8_8::int()` directly reads the high byte `H` (0T shift).
  - `v.frac_raw() -> u8` / `u16`: extracts the low `FracBits` bits.
- Arithmetic:
  - `a + b`, `a - b`: same `fixed<Base, FracBits>` type; direct integer add/sub.
  - `a * b`: widened intermediate product shifted right by `FracBits` (truncation towards $-\infty$ or zero—defined as arithmetic shift right `>> FracBits` for Z80 efficiency).
  - `a / b`: widened dividend `(a << FracBits) / b` (Debug checks division by zero).
  - Casting `int as fixed<B, F>` shifts left by `F`; `fixed as int` performs arithmetic shift right by `F`.

### 2.3 Arithmetic Operators and Overflow Semantics
All integer and fixed-point arithmetic operators follow three explicit modes:

| Mode | Operators | Debug Build | Release Build |
| :--- | :--- | :--- | :--- |
| **Standard (Checked)** | `+`, `-`, `*`, `/` | Traps (`PanicCode::Overflow` / `DivByZero`) | Two's-complement wrapping (`/` UB-free; div-by-zero traps only if `overflow` check enabled) |
| **Wrapping** | `+%`, `-%`, `*%` | Two's-complement wrapping (never traps) | Two's-complement wrapping |
| **Saturating** | `+|`, `-|`, `*|` | Clamps to `[MIN, MAX]` | Clamps to `[MIN, MAX]` |

Bitwise operators (`&`, `|`, `^`, `!`, `<<`, `>>`) operate on integers. Shift amounts `>= bit_width` trap in Debug (`PanicCode::Overflow`).

### 2.4 Compound Types
- **Structs**: `struct Vec2 { x: q8_8, y: q8_8 }`. Layout is packed in declaration order by default (no hidden padding on Z80 unless `#[align(N)]` is specified).
- **Enums (C-like / fieldless)**: `enum Dir : u8 { Up = 0, Down = 1, Left = 2, Right = 3 }`. Backing integer defaults to `u8`.
- **Fixed-size arrays**: `[T; N]`. Indexing `arr[i]` checks `i < N` in Debug (`PanicCode::Bounds`) and is unchecked in Release.
- **Static constant slices**: `&const [T]`. Fat pointer `(addr: u16, len: u16)` (or `len: u8` when statically bounded) referencing immutable ROM / constant data. May be stored in `struct` and `state` fields.

---

## 3. Second-Class References & Borrow Rules (AD-3)

Aun achieves static memory safety with **zero lifetime annotations** by restricting `&T` and `&mut T` to second-class values:

1. **Scope restriction**: `&T` and `&mut T` may appear only as function parameters, local `let` bindings, or temporary expressions.
   - Storing `&T` or `&mut T` inside a `struct`, `state`, `machine`, `Pool`, array, or `static` is a **compile-time error**.
   - Returning `&T` or `&mut T` from a function is a **compile-time error**.
2. **Exclusivity (Aliasing XOR Mutability)**:
   - While a `&mut T` borrow of place `p` is live, no other `&T` or `&mut T` borrow of `p` (or any overlapping sub-place) may exist in the same scope or be passed to the same function call.
   - Disjoint struct fields (`&mut s.a` and `&mut s.b`) may be borrowed simultaneously.
3. **No dangling references**:
   - Because references cannot escape their lexical scope and all storage is statically allocated (or scoped to a `state` whose handler holds `&mut self`), a reference can never outlive its referent.
   - Performing `goto` inside a state handler immediately terminates the handler (Section 5.3), ensuring no local reference to old state fields can be used after a transition.
4. **Mutable slices (`&mut [T]` / `&[T]`)**:
   - Follow the exact same second-class rules as `&mut T` / `&T`.

---

## 4. `Pool<T, N>` and `Handle<T>`

`Pool<T, N>` is a statically allocated fixed-capacity object pool (`1 <= N <= 255` for 1-byte handles; `256 <= N <= 65535` uses 2-byte `u16` handles).

```aun
#[layout(soa, align(256))]
static mut BALLS: Pool<Ball, 16>;
```

### 4.1 Handle Representation
- `Handle<T>` is **always 1 byte (`u8`)** for `N <= 255` in both Debug and Release builds so struct and RAM layouts remain identical across build modes.
- Sentinel value `Handle::<T>::NONE` (`0xFF` for `N <= 255`, reserving `N <= 255` indices `0..N-1`; hence max capacity for 1-byte handle with `NONE` is `255` if `Option`-like sentinel is used, or `255` slots `0..254`).
- **Debug UAF / Bounds Detection**:
  - In Debug builds, the compiler allocates a separate side table (`alive_bits` bitmap and, on host/debug, optional slot epoch metadata) outside the `Pool`'s SoA arrays.
  - Every `pool.get(h)` / `pool.get_mut(h)` / `pool.free(h)` verifies `h < N` and `alive[h] == true`, trapping with `PanicCode::InvalidHandle` on failure.
  - In Release builds, the side table and all checks are completely eliminated (0 bytes RAM, 0 bytes code).

### 4.2 Pool Operations
- `pool.alloc(init: T) -> Handle<T>` (returns `Handle::NONE` if full, or `pool.alloc_or_panic(init)`).
- `pool.free(h: Handle<T>)` (runs `on_exit` if `T` is a `machine`).
- `pool.get(h: Handle<T>) -> &T` / `pool.get_mut(h: Handle<T>) -> &mut T` (second-class references).
- `for h in pool.iter_handles() { ... }` / `for item in pool.iter_mut() { ... }`.

### 4.3 `#[layout(soa, align(256))]`
- Transforms `Pool<Struct, N>` into one contiguous array per field, each aligned to a 256-byte boundary (`0xXX00`).
- On Z80, accessing `pool[h].field` compiles to:
  ```z80
  LD H, hi(POOL_FIELD_BASE)
  LD L, A          ; A = 1-byte Handle<T>
  LD A, (HL)       ; 18 T-states total, zero 16-bit multiply
  ```

---

## 5. First-Class State Machines (`machine` / `state` / `transitions`)

### 5.1 Declaration Syntax
```aun
#[strict_transitions] // optional: forbid dynamic .goto()
machine Ball {
    // Shared fields (live across all states)
    x: q10_6,
    y: q10_6,

    // Exclusive states (state-local fields are RAM-overlaid via Uchiguri)
    state Inactive;
    state InShooter { power: u8 };
    state InPlay { vx: q8_8, vy: q8_8, skill_timer: u16 };
    state MagnetHeld { hold_timer: u8 };

    // Initial state (optional; defaults to the first declared state)
    initial Inactive;

    // Permitted transition graph
    transitions {
        Inactive   -> InShooter;
        InShooter  -> InPlay | Inactive;
        InPlay     -> MagnetHeld | Inactive;
        MagnetHeld -> InPlay | Inactive;
    }

    // Shared methods (current state is unknown unless narrowed via `match`)
    fn reset(&mut self) {
        self.goto(Inactive); // Dynamic transition (checked in Debug unless strict)
    }

    // State-specific handler blocks (current state is statically known)
    on InShooter {
        fn on_enter(&mut self) { self.power = 0; }
        fn on_exit(&mut self)  { }
        fn step(&mut self, inp: &InputState) {
            if inp.plunger == 0 && self.power > 0 {
                goto InPlay {
                    vx: 0.0,
                    vy: -(self.power as q8_8),
                    skill_timer: 120,
                };
            }
        }
    }
}
```

### 5.2 Handler Signatures and Dispatch
- Methods with the same name and signature across multiple `on <State>` blocks form a **dispatched method** callable on `&mut Machine` (e.g., `ball.step(&inp)`).
- If a state does not define a dispatched method `step`, either:
  - A default `fn step(&mut self, ...)` in the `machine` body is called, or
  - If none exists, calling `ball.step(...)` is a no-op only if marked `default fn step(...) {}`; otherwise missing handlers are a compile-time error (exhaustiveness).
- Special lifecycle hooks:
  - `fn on_enter(&mut self)`: runs after the state tag and state-local fields are initialized.
  - `fn on_exit(&mut self)`: runs before the state tag changes and while old state-local fields are still valid.
  - Neither `on_enter` nor `on_exit` may trigger another `goto` directly in v0.1 (prevents re-entrant transition loops; compile-time error).

### 5.3 `goto` Control Flow Semantics
- `goto TargetState { field: expr, ... }` is a **diverging expression of type `!` (never)**.
- Evaluation order:
  1. Evaluate all initializer expressions for `TargetState` (while `self`'s old state fields are still readable!).
  2. Call `on_exit` of the current state (and any nested child machines bottom-up).
  3. Write `TargetState` fields into the overlaid state memory and update the state tag/dispatch pointer.
  4. Call `on_enter` of `TargetState` (and nested child machines top-down).
  5. Immediately return from the enclosing `on <State>` handler (or `match` transition helper).
- Because `goto` has type `!`, it can be used directly as a branch of `if` or `match`, and code following `goto` in the same block is unreachable (compile-time warning/error).
- **Self-transitions** (`StateA -> StateA` when explicitly listed in `transitions`): runs `on_exit` -> re-initializes fields -> runs `on_enter`.

### 5.4 Static vs. Dynamic Transitions
- **Static transition**: `goto StateB { ... }` inside `on StateA { ... }` or `match m { StateA(s) => ... }`. Checked against `transitions` at compile time (0 runtime overhead).
- **Dynamic transition**: `m.goto(StateB { ... })` when `m`'s current state is not statically narrowed.
  - Debug: checks `TRANSITION_MASK[cur_state] & (1 << target_state) != 0`; traps with `PanicCode::InvalidTransition` on failure.
  - Release: check eliminated (0 bytes).
  - Forbidden at compile time when `#[strict_transitions]` is applied to the `machine` or enclosing module.

### 5.5 Hierarchical & Orthogonal State Machines
- **Hierarchical (HSM)**: A `state` field may hold another `machine`. On parent exit, the child's `on_exit` runs first (innermost-out). On parent enter, the parent's `on_enter` runs first, then the child's initial state `on_enter` runs (outermost-in).
- **Orthogonal (Parallel)**: A `struct` or `state` may hold multiple independent `machine` fields side-by-side (e.g., `movement: MoveMachine, weapon: WeaponMachine`).

---

## 6. Control Flow & Expressions

- `if cond { ... } else if cond2 { ... } else { ... }` (usable as statement or expression).
- `match expr { Pattern => expr, ... }` — exhaustive pattern matching over integers, `enum`, and `machine` states (`StateA(s) => ...` binds `s` to a narrowed view of the machine).
- Loops: `loop { ... }`, `while cond { ... }`, `for i in 0..N { ... }` (half-open `..` and inclusive `..=`). Supports `break` and `continue` with optional labels `'outer: loop { ... }`.
- No exceptions, no closures with captured environments, no dynamic dispatch (`dyn Trait`).

---

## 7. Modules, Visibility, and Compilation Unit

- A project is configured by `aun.toml` specifying:
  - `entry = "src/main.aun"`
  - `target = "board/pc88sr.aun"`
  - `asm_sources = ["asm/hal_pc88.s", ...]` (optional SDASZ80-compatible assembly inputs)
- Rust-style module tree: `mod physics;` loads `src/physics.aun` or `src/physics/mod.aun`.
- Items are private to their module by default; `pub` exports them to parent/sibling modules.
- `use physics::Ball;` brings symbols into scope.

---

## 8. FFI, Inline Assembly, and Port I/O

### 8.1 Register-Explicit FFI (`extern "asm"`)
```aun
extern "asm" {
    fn hal_wait_vsync(prev: u8 in A) -> u16 in DE
        clobbers(BC, HL, F);

    fn gfx_fill_rect_alu(col: u8 in A, y: u8 in L, w: u8 in B, h: u8 in C)
        requires bank(VramAlu), di
        clobbers(AF, BC, DE, HL);
}
```
- Valid Z80 register bindings: `A`, `B`, `C`, `D`, `E`, `H`, `L` (8-bit); `BC`, `DE`, `HL`, `IX`, `IY` (16-bit); `DEHL` (32-bit); `F` / `AF` in `clobbers`.
- `requires bank(...), di` attaches compile-time hardware preconditions checked at every call site.

### 8.2 Hardware Port I/O Intrinsics
- `io_out(port: u8, val: u8)` — compiles to Z80 `OUT (n), A` (when `port` is constant) or `OUT (C), r`.
- `io_in(port: u8) -> u8` — compiles to Z80 `IN A, (n)` or `IN r, (C)`.
- Named symmetrically (`io_out` / `io_in`) so `in` remains unambiguous as a keyword for `for x in ...` and `in Space`.
- Both intrinsics are intercepted cleanly by `aun-interp` during host testing via mock handlers.

### 8.3 Inline Assembly
```aun
asm!("di");
asm!("ld a, (hl)", clobbers(A));
```
- Inline `asm!` blocks may specify `clobbers(...)` so the register allocator preserves live values around them.

---

## 9. Attributes Summary

| Attribute | Target | Effect |
| :--- | :--- | :--- |
| `#[layout(soa, align(256))]` | `Pool` / `static` | Expands struct fields into separate 256B-aligned arrays |
| `#[align(N)]` | `static` / `struct` | Forces address alignment to power-of-two `N` |
| `#[strict_transitions]` | `mod` / `machine` | Forbids dynamic `.goto()`; requires statically narrowed transitions |
| `#[smc]` | `machine` / `fn` | Permits self-modifying code (`JP xxxx` dispatch) when placed in RAM |
| `#[reentrant]` | `fn` | Allows recursion using hardware stack frame instead of static RAM |
| `#[overlay(SlotName)]` | `state` | Places state code and data in the named `overlay_slot` |
| `#[checks(off)]` / `#[checks(overflow = off)]` | `mod` / `fn` | Overrides dynamic check generation locally |
| `#[panic_handler]` | `fn` | Registers the global panic handler (`fn(PanicCode, u16) -> !`) |
| `#[interrupt(SourceName)]` | `fn` | Marks an ISR bound to an `interrupt_controller` source |
| `#[test]` | `fn` | Marks a host unit test executed by `aun test` (excluded from target builds) |
