---
id: memory-optimization
title: Memory & Overlay Budget Optimization Playbook
summary: Structured workflow for resolving overlay, resident code, RAM, and stack overflows in idiomatic Aun before considering hand-written assembly.
category: guide
---
# Memory & Overlay Budget Optimization Playbook

When a build fails with `E006` (code/overlay budget overflow) or `E007` (RAM/stack overflow), follow this decision tree **in order**.

> **Anti-pattern: Do NOT jump to rewriting `.aun` into `.s` assembly first.**
> Most overflows on Z80/retro targets come from architectural placement (functions/data accidentally pinned in Resident or a parent FSM, unused/demo data, or unshared RAM buffers), not instruction-level code density. Inspecting `.asm` before checking structural placement wastes time and introduces brittle assembly.

## Step 1: Run `--memory-report` (Works Even When Over Budget)

Re-run `aun build` with `--memory-report`:

```sh
aun build --memory-report ...
```

Unlike manually grepping `.asm` or `.sym`, `--memory-report` uses compiler-internal call-graph and overlay ownership data to show:

- **Per-space budget breakdown**: how much of `MainRam` (or another `space`) is consumed by `Resident`, each ancestor `#[swap_boundary]`, and each leaf overlay.
- **Top symbols per region**: largest functions, `const` tables, and external `.s` modules.
- **LCA promotion callers**: *why* a function or `const` landed in `Resident` or a parent overlay (for example, a single call from `Title` pulling a 1 KB gameplay helper out of `InGame` into `Resident`).
- **RAM & stack breakdown**: largest globals, state buffers, and Uchiguri frame peaks.

## Step 2: Remove Dead Code and Demo/Unused Data

- Check `warning[W001]` messages: in `--format bin`, Aun automatically eliminates unreferenced functions (including unreferenced `pub fn` not referenced by external `.s`) and unreferenced `const`/`static` globals, reporting the bytes reclaimed.
- Look for large `const` tables or debug/demo helpers in `--memory-report` that are still reachable only from a single demo/test branch and remove or gate them behind `#[cfg(test)]`.
- Avoid marking internal helpers `pub` unless external `.s` or cross-module callers require them.

## Step 3: Fix Resident & Parent FSM Bloat (LCA Demotion)

Every byte in `Resident` (or a parent `#[swap_boundary]` such as `InGame`) shrinks the remaining window for **every** child overlay sharing that `space`.

- Inspect the **LCA-promoted to Resident** section of `--memory-report`.
- If a helper is pinned in `Resident` because of one stray call from an unrelated state (e.g., `Title` or `Result` calling an `InGame` helper), break that dependency or move the cold caller into the appropriate subtree.
- Move scene-specific logic, `const` tables, and external `.s` modules (`[overlay_asm]`) out of Resident / parent states into deeper `#[swap_boundary]` leaf states so siblings reuse the same address window.
- For singleton scene FSMs, mark the machine `#[smc]`: Aun places per-state handler tables inside each overlay blob and drops unused `*_vtbl_*` tables from Resident.

## Step 4: Share RAM via Uchiguri State-Scoped Buffers (`E007`)

- Move scene-only `static mut` arrays inside `state S { static mut BUF: ...; }` (optionally with `#[zero_on_enter]`) so mutually exclusive states overlap in RAM automatically.
- Place large buffers into secondary `space` or switchable `bank` windows where the target hardware allows.
- Keep hot struct arrays compact (`u8`/`i8` where range permits) and avoid unnecessary deep call chains to stay within stack budgets.

## Step 5: Only Then Consider Local Code-Size Tuning or `extern "asm"`

Only after Steps 1–4 are clean and a single measured leaf overlay or hot loop still exceeds its budget:

- Check for repeated large arithmetic or wide (`u32`) expressions that can stay `u16`/`u8`.
- Factor shared cold code into a common ancestor `#[swap_boundary]` helper.
- Move only the specific measured bottleneck into `extern "asm"` with an explicit `[overlay_asm]` owner state.

See [E006: Code & Overlay Budget Overflow](rules/E006.md), [E007: RAM & Stack Budget Overflow](rules/E007.md), and [Target Definitions and Uchiguri RAM](target-and-memory.md).
