/* Equivalent C benchmark matching bench/suite/entity_pool.aun under SDCC --sdcccall 1. */
#include <stdint.h>

typedef int16_t q8_8;
typedef int16_t q10_6;

enum {
    ST_IDLE = 0,
    ST_LAUNCHING = 1,
    ST_ACTIVE = 2
};

static const q8_8 GRAVITY_TABLE[4] = { 64, 128, 192, 256 }; /* 0.25, 0.50, 0.75, 1.00 in Q8.8 */

typedef struct {
    q10_6 x;
    q10_6 y;
    uint8_t state;
    union {
        struct { uint8_t power; uint8_t kind; } launching;
        struct { q8_8 vx; q8_8 vy; uint8_t gravity_idx; uint8_t ttl; } active;
    } u;
} Actor;

static Actor actors[8];
static uint8_t live_mask;

static inline q8_8 sat_add16(q8_8 a, q8_8 b) {
    int32_t s = (int32_t)a + (int32_t)b;
    if (s > 32767) return 32767;
    if (s < -32768) return -32768;
    return (q8_8)s;
}

void entity_pool_reset(void) {
    for (uint8_t i = 0; i < 8; i++) {
        actors[i].state = ST_IDLE;
    }
    live_mask = 0;
}

uint8_t entity_pool_spawn(uint8_t slot, uint8_t px, uint8_t py, uint8_t power) {
    if (slot >= 8) return 0;
    Actor *a = &actors[slot];
    a->x = (q10_6)((int16_t)px << 6);
    a->y = (q10_6)((int16_t)py << 6);
    a->state = ST_LAUNCHING;
    a->u.launching.power = power;
    a->u.launching.kind = slot;
    live_mask |= (uint8_t)(1u << slot);
    return 1;
}

uint16_t entity_pool_step_all(int8_t wind) {
    uint16_t sum_y = 0;
    for (uint8_t i = 0; i < 8; i++) {
        uint8_t bit = (uint8_t)(1u << i);
        if ((live_mask & bit) == 0) continue;
        Actor *a = &actors[i];
        if (a->state == ST_LAUNCHING) {
            if (a->u.launching.power > 0) {
                uint8_t pwr = a->u.launching.power;
                uint8_t kind = a->u.launching.kind & 3u;
                a->state = ST_ACTIVE;
                a->u.active.vx = (q8_8)((int16_t)wind << 6);
                a->u.active.vy = (q8_8)(-((int16_t)pwr << 7));
                a->u.active.gravity_idx = kind;
                a->u.active.ttl = 4;
            }
        } else if (a->state == ST_ACTIVE) {
            if (a->u.active.ttl == 0) {
                a->state = ST_IDLE;
                live_mask &= (uint8_t)~bit;
                continue;
            }
            a->u.active.vy = sat_add16(a->u.active.vy, GRAVITY_TABLE[a->u.active.gravity_idx]);
            a->u.active.vx = sat_add16(a->u.active.vx, (q8_8)((int16_t)wind << 5));
            a->x = (q10_6)(a->x + (a->u.active.vx >> 2));
            a->y = (q10_6)(a->y + (a->u.active.vy >> 2));
            a->u.active.ttl--;
        }
        if (a->state == ST_IDLE) {
            live_mask &= (uint8_t)~bit;
        } else {
            sum_y += (uint16_t)(a->y >> 6);
        }
    }
    return sum_y;
}
