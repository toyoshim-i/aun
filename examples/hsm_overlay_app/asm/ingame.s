        .area _CODE
        .globl ext_ingame_bonus

; Assigned to `App::InGame` via [overlay_asm] in aun.toml, so both
; `InGame::Stage1` and `InGame::Stage2` can call it while `InGame` is mapped.
ext_ingame_bonus:
        ld a, #5
        ret
