#include <stdint.h>

extern void _hal_sfx_init(void);
extern uint8_t _hal_se_busy(void);
extern void _hal_play_se(const uint8_t *data) __sdcccall(1);
extern const uint8_t GAME_SE_DATA[];

static uint8_t s_cur_prio;

typedef struct {
    uint16_t off;
    uint8_t prio;
} SfxEntry;

static const SfxEntry SFX_TABLE[18] = {
    {   0u, 0u }, {   0u, 1u }, {  49u, 1u }, {  96u, 1u }, { 145u, 1u }, { 188u, 3u },
    { 241u, 2u }, { 293u, 3u }, { 342u, 3u }, { 398u, 3u }, { 449u, 2u }, { 496u, 3u },
    { 558u, 3u }, { 594u, 2u }, { 641u, 2u }, { 700u, 2u }, { 754u, 2u }, { 807u, 2u },
};

void sfx_init(void) {
    s_cur_prio = 0u;
    _hal_sfx_init();
}

void sfx_play(uint8_t sfx_id) {
    if (sfx_id == 0u || sfx_id >= 18u) {
        return;
    }
    const SfxEntry *e = &SFX_TABLE[sfx_id];
    uint8_t prio = e->prio;
    if (prio < s_cur_prio && _hal_se_busy() != 0u) {
        return;
    }
    s_cur_prio = prio;
    _hal_play_se(GAME_SE_DATA + e->off);
}

void sfx_update(uint8_t tick) {
    if (tick == 0u && _hal_se_busy() == 0u) {
        s_cur_prio = 0u;
    }
}
