#include <stdint.h>

typedef struct {
    uint32_t score;
    uint16_t bonus;
    uint8_t lives;
    uint8_t wave;
    uint8_t combo;
    uint8_t flags;
} GameState;

static volatile __at(0xEC00) GameState G_STATE;

static uint8_t read_u8(uint16_t addr) {
    return *(const volatile uint8_t *)addr;
}

static void write_u8(uint16_t addr, uint8_t v) {
    *(volatile uint8_t *)addr = v;
}

static uint32_t add_clamped_score(uint32_t cur, uint32_t delta) {
    uint32_t s = cur + delta;
    if (s > 999999999UL) {
        return 999999999UL;
    }
    return s;
}

void tally_step(uint8_t clears, uint8_t mult) {
    if (clears == 0u) {
        if (G_STATE.combo > 0u) {
            G_STATE.combo = G_STATE.combo - 1u;
        }
        return;
    }
    uint8_t c = G_STATE.combo + clears;
    if (c > 16u) {
        c = 16u;
    }
    G_STATE.combo = c;

    uint16_t base = (uint16_t)clears * 100u;
    uint16_t wave_bonus = (uint16_t)G_STATE.wave * 25u;
    uint32_t pts = ((uint32_t)(base + wave_bonus)) * (uint32_t)(mult + c);
    G_STATE.score = add_clamped_score(G_STATE.score, pts);
    G_STATE.bonus = G_STATE.bonus + base;
    if (G_STATE.bonus >= 5000u) {
        G_STATE.bonus -= 5000u;
        if (G_STATE.lives < 9u) {
            G_STATE.lives += 1u;
        }
        G_STATE.flags |= 0x01u;
    }
}

void tally_draw_score(void) {
    uint32_t s = G_STATE.score;
    uint8_t row = read_u8(0xEC10u);
    uint16_t dst = 0xF3C8u + (uint16_t)row * 80u + 8u;
    for (uint8_t i = 0u; i < 8u; ++i) {
        uint8_t d = (uint8_t)(s % 10u);
        write_u8(dst - i, (uint8_t)('0' + d));
        s /= 10u;
    }
}
