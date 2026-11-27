---
id: target-and-memory
title: Target Definitions, Uchiguri RAM, Banks, and Overlays
summary: Hardware target blocks, static call-graph RAM overlay allocation, bank/DI safety, and overlays.
category: guide
---
# Aun Target Definitions, Static Memory (*Uchiguri*), Banks, and Overlays v0.1

This document specifies how hardware targets are described in Aun, how the compiler performs static RAM overlay allocation (*Uchiguri*), how bank and interrupt safety are verified statically, and how multi-binary code overlays are linked and emitted.

---

## 1. Target Definition DSL (`board/*.aun`)

A target file defines the physical address map, banking hardware, interrupt controller, and code overlay slots for a specific machine (e.g., `board/pc88sr.aun`).

### 1.1 Address Spaces and Stack Budget
```aun
target pc88sr : z80 {
    space MainRam   @ 0x0100..0xBFFF;
    space WorkRam   @ 0xC000..0xEFFF; // optional non-banked or default view
    stack MainStack @ 0xBE00..0xBFFF; // 512 bytes reserved for hardware SP
}
```
- `space <Name> @ <start>..<end>` defines a contiguous inclusive address range (`u16` on Z80).
- `stack <Name> @ <start>..<end>` reserves hardware stack space. The linker initializes `SP = end + 1` (or `end` per target convention) and statically verifies that worst-case stack depth fits within `end - start + 1`.
- Variables and functions default to the primary resident RAM/code space unless annotated with `in <SpaceOrBank>` (e.g., `static mut SCORE: u32 in HighMainRam;`).

### 1.2 Bank Groups (`bank_group`)
```aun
bank_group HighWindow @ 0xC000..0xFFFF {
    bank HighMainRam  @ 0xC000..0xEFFF;
    bank HighSpeedRam @ 0xF000..0xFFFF;
    bank HiddenBgmRam @ 0xF000..0xFFFF requires di;
    bank VramAlu      @ 0xC000..0xFFFF requires di;
    bank VramB        @ 0xC000..0xFFFF requires di;
    bank VramR        @ 0xC000..0xFFFF requires di;
    bank VramG        @ 0xC000..0xFFFF requires di;

    default HighMainRam + HighSpeedRam;

    switch(to) {
        HighMainRam + HighSpeedRam => { io_out(0x35, 0x00); io_out(0x5F, 0); }
        HiddenBgmRam               => { io_out(0x32, shadow_32 | 0x10); }
        VramAlu                    => { io_out(0x32, shadow_32 | 0x40); io_out(0x35, 0x80); }
        VramB                      => { io_out(0x5C, 0); }
        VramR                      => { io_out(0x5D, 0); }
        VramG                      => { io_out(0x5E, 0); }
    }
}
```
- **Bank combinations**: Non-overlapping banks within the same `bank_group` may be active simultaneously (written `BankA + BankB`). Overlapping banks (e.g., `VramAlu` covering `0xC000..0xFFFF`) mutually exclude any bank whose range intersects theirs.
- **`requires di`**: Declares that interrupts must be disabled whenever this bank is mapped (e.g., because an ISR accesses `HighMainRam` or `HighSpeedRam` that would be shadowed).
- **Shadow registers**: Hardware write-only I/O port shadow variables (like `shadow_32`) are normal resident `static mut` variables declared in the target or HAL module and referenced directly by `switch(to)` arms.

### 1.3 Interrupt Controllers (`interrupt_controller`)
```aun
interrupt_controller Pc88Int {
    disable => asm!("di");
    enable  => asm!("ei");
    source Vrtc  @ im2(0xF302) { rearm => io_out(0xE4, shadow_e4); }
    source Sound @ im2(0xF308) { rearm => io_out(0xE4, shadow_e4); }
}
```
- An ISR is bound via `#[interrupt(Vrtc)] fn on_vrtc() { ... }`.
- The compiler automatically wraps the ISR with:
  1. Minimal register save/restore (only registers actually clobbered by the ISR call tree, determined via WPO!).
  2. `rearm` code before `EI; RETI` (or `RETI` per controller config).
  3. Vector table entry generation at the declared address (`0xF302`, `0xF308`).

### 1.4 Memory Swap Handlers (`on_swap`) & Legacy Code Overlay Slots (`overlay_slot`)

For hierarchical overlays managed automatically by the compiler's tree *Uchiguri* code layout, declare an application callback on `target`:
```aun
on_swap(ops: &const [LoadOp]) => hal_swap_overlays(ops);
```
where the compiler pre-registers:
```aun
struct LoadOp {
    blob_id: u8,
    space: u8,   // Matches generated `SPACE_<Name>` u8 constants
    dst: u16,    // Exact linked destination address (0xYYYY)
    size: u16,   // Binary size in bytes
}
```
Custom disk/ROM loading, decompression (e.g. ZX0/RLE), and direct VRAM asset transfers remain inside user callbacks (`on_swap`, `on_enter`) while the compiler guarantees exact static addresses and conflict-free layout across spaces and banks.

For explicit fixed-window slots, legacy `overlay_slot` is also supported (with optional `in Space`):
```aun
overlay_slot SceneCode @ 0x8000..0x9FFF in MainRam {
    load(id: OverlayId) => hal_load_overlay(id, 0x8000);
}
```
- Declares a fixed RAM window into which mutually exclusive overlay binaries (`ovl_<name>.bin`) are loaded on demand.

---

## 2. Static Memory Allocation & *Uchiguri* (RAM Overlay)

Aun eliminates stack frames by placing local variables, temporaries, and state-machine fields at fixed RAM addresses, then overlaying non-simultaneous lifetimes.

### 2.1 Allocation Classes
1. **Persistent Globals (`static` / `static mut`)**: Live for the entire program (or entire overlay state lifetime if scoped to an `#[overlay]` state). Never overlaid with anything else.
2. **State-Local Fields**: Fields declared inside `state S { ... }` of a `machine`. Mutually exclusive states of the same `machine` do not interfere with each other and share the same base RAM slice (`sizeof(machine) = sizeof(shared) + sizeof(tag) + max_i(sizeof(state_i))`).
   - Nested HSMs apply this rule recursively at every level of the state tree.
3. **Function Locals & Call-Frame Temporaries**:
   - Every non-`#[reentrant]` function gets a static activation record (excluding values kept entirely in registers).
   - Two functions `f` and `g` **interfere** iff one is reachable from the other in the static call graph (including indirect calls via state dispatch), OR one is reachable from an ISR while the other is reachable from the main thread (or another nested ISR).
   - Non-interfering functions can share the exact same RAM addresses for their static locals!

### 2.2 Unified Interference Graph & Coloring
The *Uchiguri* allocator constructs a weighted interference graph $G = (V, E)$:
- **Vertices $V$**: Every static local variable / spill slot / call-frame block, plus state-local memory regions.
- **Edges $E$**:
  - `(u, v) in E` if `u` and `v` can be live simultaneously along any call path.
  - `(u, v) in E` if `u` is reachable from `main` and `v` is reachable from any `#[interrupt]` ISR (100% ISR isolation).
  - `(u, v) in E` if `u` and `v` belong to active sibling fields of the same state or struct.
- **Interval / Block Packing**:
  - For call-tree locals, Aun computes a topological bottom-up offset (classic call-tree overlay) refined by graph-coloring interval packing so arbitrarily sized variables (`u8`, `u16`, `[u8; 32]`) pack with zero fragmentation while respecting `#[align(N)]` constraints.
  - Recursion without `#[reentrant]` is detected as a cycle in the call graph and reported as a compile-time error.

### 2.2.1 Call-Graph Boundaries: ISRs, Indirect Calls, External Callbacks, and `#[reentrant]`
- **ISR Isolation**: Functions reachable from `#[interrupt(...)]` ISRs are placed in an isolated RAM region strictly above `main_end`; sharing any non-`#[reentrant]` function across main-thread code and an ISR is a compile-time error (`E004`). Library-mode (`--format sdasz80`) C-called ISR roots (`#[isr_entry]`) and multi-priority nested ISR overlay groups are tracked in issue `#21`.
- **Indirect Calls & State Dispatch**: Taking a function pointer via `addr_of(fn)` conservatively adds call-graph edges from every `CallIndirect` site to all address-taken functions; dispatched `machine` methods add edges to every implementing state's handler.
- **External C Callbacks (`--format sdasz80`)**: Functions that transitively invoke external `extern "asm"` / C routines conservatively avoid overlapping any `pub fn` callback frame so C-to-Aun re-entry cannot corrupt caller locals (leaf extern opt-out is tracked in `#18`).
- **`#[reentrant]` Status**: Non-`#[reentrant]` recursion and main/ISR function sharing are rejected at compile time today; compiler-managed hardware stack frames and main/ISR auto-cloning for shared functions are planned (`#21`).

### 2.3 Static Stack Depth Verification
Even with static locals, Z80 uses the hardware stack for `CALL`/`RET` return addresses (2 bytes per call), `PUSH`/`POP` register saves, `#[reentrant]` frames, and ISR entry/saves.
- At link time, the compiler computes:
  $$\text{MaxStack} = \max_{\text{path from main}} \text{stack\_bytes}(\text{path}) + \sum_{\text{nestable ISRs}} \text{isr\_stack\_bytes}$$
- If `MaxStack > size(stack_region)`, the build fails with the exact worst-case call chain.
- Functions annotated `#[reentrant]` must specify a `#[max_depth(K)]` attribute so the stack bound remains statically provable (unless explicitly opted out with `#[unbounded_stack]`).

---

## 3. Static Bank & Interrupt Effect System

Every function and block has a static **hardware state** `(BankState, IrqState)`:
- `BankState`: map from each `bank_group` to either a known active bank set (e.g., `HighMainRam + HighSpeedRam`), or `Unknown` (at dynamic boundaries).
- `IrqState`: `Enabled`, `Disabled` (`di`), or `Unknown`.

### 3.1 Scoped Bank Switching (`with bank(...)`)
```aun
with bank(VramAlu) {
    // Compiler emits:
    //   1. di (because VramAlu requires di, unless already statically di)
    //   2. switch(VramAlu)
    gfx_fill_rect_alu(0, 16, 40, 80);
    // On scope exit (including early return / goto):
    //   3. restore previous bank
    //   4. ei (only if this scope disabled interrupts)
}
```
- **Static conflict detection**: Accessing any variable `in B` where `B` is not mapped in the current `BankState` is a **compile-time error**.
- **Precondition checking**: Calling a function (Aun or `extern "asm"`) with `requires bank(B), di` requires `BankState` to include `B` and `IrqState == Disabled`; otherwise it is a compile-time error.
- **ISR bank safety**: Every ISR assumes the default bank mapping (because any bank that shadows ISR memory must declare `requires di`, guaranteeing the ISR cannot fire while that bank is active!). If an ISR itself switches banks, it must restore the default bank before returning.

### 3.2 Bank / DI Optimization
- **Hoisting**: When a loop body contains `with bank(B)` and performs no accesses conflicting with `B`, `aun-ir` hoists the bank switch and `di`/`ei` pair outside the loop.
- **Coalescing**: Adjacent `with bank(B) { ... } with bank(B) { ... }` blocks are merged into a single switch.
- **Dynamic boundary checks**: If a function is called with `BankState == Unknown`, Debug builds emit a lightweight shadow-state assertion (`PanicCode::BankMismatch`), eliminated in Release.

---

## 4. Top-Level State x Code Overlay Integration

```aun
machine App {
    total_score: u32, // Resident in MainRam

    #[swap_boundary] state Title { cursor: u8 };
    #[swap_boundary]
    state InGame {
        lives: u8,
        static mut ITEM_BUF: [u8; 64] in HighMainRam;

        #[swap_boundary]
        state Stage1 {
            wave: u8,
            static mut STAGE_BUF: [u8; 32] in HighMainRam;
        };
        #[swap_boundary]
        state Stage2 {
            boss_hp: u8,
            static mut STAGE_BUF: [u8; 48] in HighMainRam; // Shares address with Stage1's STAGE_BUF
        };

        initial Stage1;
    }

    transitions {
        Title          -> InGame;
        InGame::Stage1 -> InGame::Stage2;
        InGame         -> Title;
    }
}
```

### 4.1 Placement, Auto-Layout, and Safety Rules
1. **Resident core & swap trampolines**: Shared `App` fields, the active leaf state tag, `on_swap` / `load` handlers, ISRs, ROM `LoadOp` tables (`__aun_swap_m*_t*`), and deduplicated Resident transition trampolines (`__aun_tr_*`) are placed in resident `MainRam`.
2. **Call-graph LCA & tree *Uchiguri* code layout**:
   - State handlers, private helpers, private sub-machines, ROM constants, and external `.s` modules (`#[overlay(State)] extern "asm"`, `--asm StatePath=file.s`, or `[overlay_asm]` in `aun.toml`) are placed at the LCA enclosing `#[swap_boundary]` / `#[swap_boundary(Space)]` (or `#[overlay(Slot)]`) of all callers.
   - Code and constants are linked top-down in primary `space` immediately after resident code (`res_end`) or in their requested secondary space/bank (`Space`), so mutually exclusive siblings (such as `InGame::Stage1` and `InGame::Stage2`) share exact execution addresses below their parent `InGame` without manual slot addresses.
   - Direct calls or external `.s` references across mutually exclusive boundary subtrees are **compile-time errors**.
   - State-scoped `static mut` buffers (`state S { static mut BUF: ... in Bank; }`) obey the same sibling-sharing tree layout inside their target RAM space/bank and are accessible only inside `S` and its descendants.
3. **LCA Differential Transition Sequence (generated in resident memory)**:
   1. Evaluate and stage transition payload fields into dedicated Uchiguri staging RAM outside any overlay window.
   2. When a `goto` occurs at tail position inside an overlay method, tail-jump (`jp __aun_tr_*`) out of the overlay so the return PC on the hardware stack always stays in intact Resident/Ancestor code.
   3. Run bottom-up `on_exit()` hooks from the active leaf up to `LCA(from, to)`.
   4. Invoke `target.on_swap(ops)` once with the contiguous `&const [LoadOp]` suffix slice covering `(LCA(from, to) .. to]` (skipping shared ancestors such as `InGame` during `Stage1 -> Stage2`), or invoke legacy `overlay_slot.load(...)` hooks.
   5. Copy staged payload fields via `ldir`, write the new leaf tag, patch `#[smc]` dispatch stubs if enabled, and run top-down `on_enter()` hooks along `(LCA(from, to) .. to]`.

---

## 5. Build Artifacts & Manifests

A successful `aun build` produces:
- `<stem>.bin`: Resident binary image linked at its target base address.
- `ovl_<sanitized_state>.bin` / `ovl_<sanitized_state>.asm`: One binary and assembly image per `#[swap_boundary]` / `#[overlay]` blob (`::` sanitized to `__`).
- `<stem>.asm`: Complete human-readable Z80 assembly listing with source line comments and T-state annotations.
- `<stem>.sym`: Symbol table mapping every function, static variable, state handler, and panic `site_id` to its address, bank/overlay ID, and source location (`file:line:col`).
- `<stem>.manifest.json`: Machine-readable build report containing:
  - `spaces`: All target spaces and banks with IDs, kinds, and hardware ranges
  - `blobs`: Overlay blob metadata (`id`, `state`, `parent_state`, `slot`, `space`, `space_id`, `origin`, `size`, `budget_end`, `file`, `asm_files`)
  - `states`: Complete flattened HSM tree per machine (`id`, `name`, `parent`, `is_leaf`, `swap_boundary`, `space`, `blob_id`)
  - `state_buffers`: State-scoped RAM/bank buffers with owning state, space, address, and size
  - *Uchiguri* RAM map, function T-state breakdowns, static worst-case stack depth, and panic `site_id` lookup table
