---
id: idiomatic-aun
title: Idiomatic Aun & Practical Cookbook
summary: How to design programs the Aun way—hierarchical state machines, swap boundaries and LoadOp callbacks, Pool/Handle data modeling, hardware banking/ISRs, and incremental C/ASM adoption.
category: guide
---
# Idiomatic Aun & Practical Cookbook

Aun is built around **explicit state**, **static lifetimes**, and **declarative hardware boundaries**. Rather than forcing C idioms (deep call stacks, heap allocation, global mode booleans) onto small systems, Aun works best when program structure reflects what is live at each point in time.

---

## 1. Thinking in State Machines

### Replace Boolean Mode Flags with `machine` / `state`

Instead of carrying several independent globals (`in_game`, `paused`, `boss_active`) alongside mutually exclusive buffers, model lifecycle states directly in a `machine`. Mutually exclusive state fields share the exact same RAM region automatically via *Uchiguri* without manual unions or casts.

```aun
machine Ball {
    // Shared across all states
    x: q10_6,
    y: q10_6,

    // Exclusive per-state fields (RAM overlaid automatically)
    state Inactive;
    state InShooter { power: u8 };
    state InPlay    { vx: q8_8, vy: q8_8, timer: u16 };

    transitions {
        Inactive  -> InShooter;
        InShooter -> InPlay | Inactive;
        InPlay    -> Inactive;
    }

    on InShooter {
        fn on_enter(&mut self) { self.power = 0u8; }
        fn step(&mut self, fire: bool) {
            if fire {
                goto InPlay { vx: 0.0q8_8, vy: -2.0q8_8, timer: 120u16 };
            }
        }
    }
}
```

### Nested HSM vs. Orthogonal Child Machines

Choose between **hierarchical nesting (`state Parent { state Child ... }`)** and **composition (`struct` / `state` fields)** based on exclusivity:

- **Use a nested `state` subtree** when child states represent sub-modes of the parent and **share parent context and lifetime** (for example `App::InGame::{Stage1, Stage2, Boss}` all sharing `score`, `lives`, and an `ITEM_BUF`).
  - Moving from `InGame::Stage1` to `InGame::Stage2` keeps `InGame`'s fields and `InGame`'s code/buffers live—only `Stage1`'s `on_exit` and `Stage2`'s `on_enter` run (LCA differential hooks).
- **Use separate `machine` fields** when behaviors run **concurrently and independently** (for example `player_move: MoveMachine` alongside `weapon: WeaponMachine` inside an `InGame` state).

### Static `goto` vs. Dynamic `.goto()` and `#[strict_transitions]`

- **Prefer `goto NextState { ... };` inside `on <State>`** or inside narrowed `match m { ... }` arms whenever the outgoing state is known at compile time. The transition is verified statically against `transitions { ... }` with zero runtime validation cost.
- Use `m.goto(...)` (such as from an external controller or `main`) when callers legitimately cannot know which leaf state `m` is currently in.
- Add `#[strict_transitions]` on `machine` or `mod` when you want the compiler to forbid dynamic `m.goto(...)` altogether.

---

## 2. Overlays & Memory Boundaries (`#[swap_boundary]` + `on_swap`)

### Responsibility Split: Language vs. Application

Aun splits overlay management cleanly between what static analysis can prove and what depends on target hardware/storage formats:

- **Compiler / Linker (static)**:
  - Partitions functions, ROM constants, private sub-machines, and state-owned `.s` modules across the HSM tree by **call-graph LCA**—helper code called by both `Stage1` and `Stage2` is promoted automatically into `InGame` (or resident memory if `Title` also calls it).
  - Packs mutually exclusive sibling blobs top-down into the same target `space` / bank without manual slot addresses.
  - Lays out state-scoped `static mut` buffers (`state S { static mut BUF: ... in Bank; }`) as a sibling-sharing tree inside their target RAM/bank space.
  - Emits safe Resident transition trampolines (`__aun_tr_*`) that tail-jump out of an active overlay before swapping, ensuring running code never overwrites itself.
- **Application / HAL (runtime)**:
  - Reads data from disk, cartridge ROM, or RAM disk.
  - Handles decompression (ZX0, LZ, RLE) and direct VRAM/hardware transfers.
  - Implements `on_swap(ops: &const [LoadOp])` to fulfill the exact `&const [LoadOp]` list passed by each transition, and uses `on_enter` for state-specific VRAM or custom binary resource setup.

### Where to Place `#[swap_boundary]`

Mark scenes or coarse modes—**not** every tiny inner state—with `#[swap_boundary]` (or `#[swap_boundary(SpaceName)]` to target a secondary memory space/bank):

```aun
target pc88sr : z80 {
    space MainRam   @ 0x0100..0xBFFF;
    stack MainStack @ 0xBE00..0xBFFF;

    bank_group HighWindow @ 0xC000..0xFFFF {
        bank HighMainRam @ 0xC000..0xEFFF;
        default HighMainRam;
    }

    on_swap(ops: &const [LoadOp]) => app_swap(ops);
}

machine App {
    state Title { cursor: u8 }; // Resident (fast instant UI)

    #[swap_boundary]
    state InGame {
        score: u16,
        static mut STAGE_SHARED: [u8; 256] in HighMainRam; // Shared across Stage1/Stage2

        #[swap_boundary]
        state Stage1 {
            hp: u8,
            static mut ENEMY_SCRATCH: [u8; 512] in HighMainRam;
        };

        #[swap_boundary]
        state Stage2 {
            wave: u8,
            static mut BOSS_SCRATCH: [u8; 512] in HighMainRam; // Overlaid with ENEMY_SCRATCH
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

With this tree:
1. `Title -> InGame::Stage1` invokes `app_swap(ops)` once with `[LoadOp(InGame), LoadOp(Stage1)]`.
2. `Stage1 -> Stage2` invokes `app_swap(ops)` once with `[LoadOp(Stage2)]` only—`InGame`'s code, `score`, and `STAGE_SHARED` stay intact in memory.
3. Moving shared helpers into a common parent state (`on InGame { ... }`) or Resident avoids cross-overlay call errors (see [E005](rules/E005.md)).

### Consuming `manifest.json` in Your Asset/Disk Pipeline

Running `aun build` emits `<stem>.bin`, `ovl_<sanitized_state>.bin`, and `<stem>.manifest.json`. A disk-image or ROM-pack script should consume the JSON manifest rather than hardcoding offsets:

- `blobs`: each `{ id, state, parent_state, space, space_id, origin, size, budget_end, file, asm_files }` tells your disk builder which `ovl_*.bin` corresponds to `op.blob_id`, so `app_swap(ops)` only needs `op.blob_id`, `op.space` (compare against generated `SPACE_<Name>` constants), `op.dst`, and `op.size`.
- `spaces`: verifies target memory windows (`id`, `name`, `kind`, `start`, `end`).
- `states` and `state_buffers`: provides full visibility into the HSM hierarchy and bank/RAM buffer placements.

---

## 3. Data Modeling Without a Heap

### Second-Class References & `Pool<T, N>` / `Handle<T>`

Aun forbids storing `&T` / `&mut T` inside structs or long-lived state fields, and rejects mutable aliasing. Use references purely as short-lived **borrow views**, and store persistent entity links using `Handle<T>` backed by a fixed-capacity `Pool<T, N>`:

```aun
#[layout(soa, align(256))]
static mut ACTORS: Pool<Actor, 16>;

fn tick_actor(h: Handle<Actor>, world_tick: u8) {
    let a: &mut Actor = ACTORS.get_mut(h);
    a.age = world_tick;
}
```

Using `Handle<T>` inside entities and state payloads keeps lifetimes explicit, prevents dangling pointers (generational/liveness checks in Debug builds), and eliminates dynamic allocation.

### Explicit Widths, Overflow Semantics, and Fixed Point

- **Always state integer widths explicitly** (`u8`, `i8`, `u16`, `i16`, `u32`, `i32`); widen deliberately with `as` rather than relying on hidden C promotion rules (see [E001](rules/E001.md)).
- Choose the arithmetic operator that matches the domain invariant:
  - `+` / `-` / `*`: traps on overflow in Debug builds (best for invariants and array/game state where silent wrap would be a bug).
  - `+%` / `-%` / `*%`: explicit modular wrap-around (counters, hash/LFSR, ring indices).
  - `+|` / `-|` / `*|` / `<<|`: saturating arithmetic (clamping HP, timers, velocities, palette fade).
- Prefer fixed-point (`q8_8`, `q10_6`, or `fixed<Base, Frac>`) for sub-pixel kinematics and sound envelopes so host `aun test` and hardware execution match bit-for-bit.
- Use `const TABLE: [u8; N] = [ ... ];` for read-only lookup tables and `&const [T]` for table parameters so constant data stays out of RAM.

---

## 4. Hardware Safety & Incremental Migration

### Bank Switching and Interrupt Discipline

- Bind banked globals/buffers (`in HighMainRam` or state-scoped `static mut ... in HighMainRam`) and bank-dependent `extern "asm"` routines (`requires bank(VramAlu), di`) cleanly so bank mismatches become compile-time errors (see [E004](rules/E004.md)).
- Protect ISR-owned shared state with `#[owned_by(IrqSource)]` so main-thread reads/writes require `critical { ... }` (which automatically elides redundant `di`/`ei` when nested or already inside an ISR), and use lock-free `Queue<T, N>` rings (`push`/`pop`/`is_empty`/`len`) for zero-`di` single-producer/single-consumer ISR-to-main communication.
- Annotate time-critical entry points (such as VBlank handlers) with `#[budget(8000T)]` or `#[budget(vblank = 4500T)]` and bound loops with `#[max_iters(N)]` so static WCET verification catches timing regressions at compile time.
- If two mutually exclusive routines share a switchable window, keep `with bank(...)` tightly scoped around the hardware access rather than holding an overlapping bank across unrelated logic.

### Incremental Adoption in C (`SDCC`) and `.s` Projects

You do not need to rewrite an entire retro project in Aun at once:

1. **Wrap existing assembly cleanly** with `extern "asm"` declaring exact register parameters/returns (`in A`, `in HL`, `in DE`) and accurate `clobbers(...)` (see [E003](rules/E003.md)).
2. **Assign assembly files to overlay subtrees** when migrating banked/overlay scenes:
   - In source: `#[overlay(App::InGame)] extern "asm" { ... }`
   - In `aun.toml`: `[overlay_asm]` mapping `"App::InGame" = ["asm/ingame.s"]`
   - On CLI: `aun build --asm App::InGame=asm/ingame.s ...`
   - Configure `.s` `.include` search roots (`asm_include_dirs` / `-I <dir>`) and optional `startup_asm` / `crt0` startup files in `aun.toml` or on the CLI.
3. **Export Aun modules to an existing C linker (`--format sdasz80`)**: mark C-facing entry points as `pub fn`, and annotate any C-called interrupt routine `#[isr_entry]` so *Uchiguri* isolates its RAM completely from main-thread frames.

### Suggested Development Loop

1. Write `#[test]`, `#[cfg(test)]`, and `#[mock("extern_sym")]` unit tests (with optional multi-file `test_sources = [...]` in `aun.toml`) and run **`aun test`** (or **`aun test --strict-externs`**) on the host interpreter for fast logic, extern-mocking, and state-transition verification (including full LCA `on_enter`/`on_exit` hook traces).
2. Run **`aun build`** (with `aun.toml` for multi-file and overlay `.s` projects) to produce binary/overlay images and `manifest.json`.
3. Inspect memory layout, stack worst-case depth, path-sensitive WCET, and symbol maps with **`aun inspect`**, tightening state splits, `#[max_iters(N)]` bounds, or bank buffers whenever budgets grow.

For full syntax rules, see [Language Reference](language-reference.md) and [Target Definitions and Uchiguri RAM](target-and-memory.md).
