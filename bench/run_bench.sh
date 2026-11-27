#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

AUN_BIN="${AUN_BIN:-$ROOT/target/debug/aun}"
if [[ ! -x "$AUN_BIN" ]]; then
    cargo build --manifest-path "$ROOT/Cargo.toml" -p aun-cli >/dev/null
    AUN_BIN="$ROOT/target/debug/aun"
fi

area_size() {
    local rel="$1" area="$2"
    awk -v a="$area" '$1 == "A" && $2 == a && $3 == "size" { print strtonum("0x" $4); exit }' "$rel"
}

printf "%-14s | %-18s | %-18s | %-12s\n" "Benchmark" "Aun (CODE / DATA)" "SDCC (CODE / DATA)" "CODE Ratio"
printf "%-14s-+-%-18s-+-%-18s-+-%-12s\n" "--------------" "------------------" "------------------" "------------"

bench_pair() {
    local name="$1" c_file="$2" aun_file="$3" c_inc="${4:-}"
    local out_dir="$TMP/$name"
    mkdir -p "$out_dir"

    "$AUN_BIN" build --release "$aun_file" --format sdasz80 -o "$out_dir" >/dev/null
    local stem
    stem="$(basename "${aun_file%.aun}")"
    sdasz80 -o "$out_dir/${stem}_aun.rel" "$out_dir/${stem}.s"

    local c_flags=(-mz80 --sdcccall 1 --opt-code-size --max-allocs-per-node 10000 -c)
    if [[ -n "$c_inc" ]]; then
        c_flags+=("-I$c_inc")
    fi
    sdcc "${c_flags[@]}" -o "$out_dir/${name}_sdcc.rel" "$c_file"

    local aun_code aun_data sdcc_code sdcc_data sdcc_init sdcc_ram
    aun_code="$(area_size "$out_dir/${stem}_aun.rel" _CODE)"
    aun_data="$(area_size "$out_dir/${stem}_aun.rel" _DATA)"
    aun_code="${aun_code:-0}"
    aun_data="${aun_data:-0}"

    sdcc_code="$(area_size "$out_dir/${name}_sdcc.rel" _CODE)"
    sdcc_data="$(area_size "$out_dir/${name}_sdcc.rel" _DATA)"
    sdcc_init="$(area_size "$out_dir/${name}_sdcc.rel" _INITIALIZED)"
    sdcc_ram=$(( ${sdcc_data:-0} + ${sdcc_init:-0} ))

    local pct
    pct="$(awk -v a="$aun_code" -v s="$sdcc_code" 'BEGIN { printf "%.1f%% (%+d B)", (a * 100.0) / s, a - s }')"
    printf "%-14s | %6d B / %4d B  | %6d B / %4d B  | %s\n" \
        "$name" "$aun_code" "$aun_data" "$sdcc_code" "$sdcc_ram" "$pct"
}

bench_pair "sfx_dispatch" "$ROOT/bench/suite/sfx_dispatch.c" "$ROOT/bench/suite/sfx_dispatch.aun"
bench_pair "game_tally"   "$ROOT/bench/suite/game_tally.c"   "$ROOT/bench/suite/game_tally.aun"
bench_pair "entity_pool"  "$ROOT/bench/suite/entity_pool.c"  "$ROOT/bench/suite/entity_pool.aun"
bench_pair "mdxnt_drv"    "$ROOT/bench/suite/mdxnt_drv.c"    "$ROOT/bench/suite/mdxnt_drv.aun" "$ROOT/bench/suite"
