        .area _CODE
        .globl hal_swap_copy_blob

; In real hardware HAL, this reads from disk/ROM/cartridge into `dst` (DE).
; A = blob_id, DE = dst
hal_swap_copy_blob:
        ret
