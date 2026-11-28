//! End-to-end CH559 / E8051 simulator tests verifying Aun -> MCS-51 codegen on `sim559`.

use aun_mcs51::{compile_mcs51, CompileOptions, Mcs51Output};
use sim559::sim::{RunConfig, Simulator, StopReason};

fn compile_and_run_ch559(src: &str, max_cycles: u64, release: bool) -> (Mcs51Output, Simulator) {
    let prog = aun_front::compile_str(0, src).expect("front-end compile should succeed");
    let opts = if release {
        CompileOptions::release()
    } else {
        CompileOptions::debug()
    };
    let out = compile_mcs51(&prog, opts).expect("mcs51 compile should succeed");
    let mut sim = Simulator::new();
    sim.load_intel_hex(&out.intel_hex, true)
        .expect("load intel hex into sim559");
    let outcome = sim.run(RunConfig {
        max_cycles: Some(max_cycles),
        stop_on_halt: true,
        ..Default::default()
    });
    assert!(
        matches!(outcome.stop_reason, StopReason::HaltLoop { .. }),
        "expected HaltLoop on sim559, got {:?} after {} cycles\nasm:\n{}",
        outcome.stop_reason,
        outcome.cycles_executed,
        out.asm_listing
    );
    (out, sim)
}

#[test]
fn ch559_e2e_arithmetic_control_flow_and_loops() {
    let src = r#"
        target ch559 : mcs51 {
            space Data  : data  @ 0x08..0x7F;
            space IData : idata @ 0x80..0xFF;
            space XData : xdata @ 0x0000..0x17FF;
            space Sfr   : sfr   @ 0x80..0xFF;
            space Bit   : bit   @ 0x80..0xFF;
            stack IStack @ 0x80..0xFF;
        }

        static mut OUT_X: [u8; 16] in XData @ 0x0100;

        fn add_mul_acc(a: u16, b: u16) -> u16 {
            (a *% b) +% 7u16
        }

        fn sum_loop(n: u8) -> u16 {
            let mut acc: u16 = 0u16;
            let mut i: u8 = 1u8;
            while i <= n {
                acc = acc +% (i as u16);
                i = i +% 1u8;
            }
            acc
        }

        pub fn main() {
            // u8 hardware mul/div/rem
            let m8: u8 = 13u8 *% 17u8; // 221
            let q8: u8 = m8 / 10u8;    // 22
            let r8: u8 = m8 % 10u8;    // 1
            OUT_X[0] = m8;
            OUT_X[1] = q8;
            OUT_X[2] = r8;

            // i8 signed mul/div/rem & shifts
            let neg_q: i8 = (-25i8) / 4i8; // -6 = 0xFA
            let neg_r: i8 = (-25i8) % 4i8; // -1 = 0xFF
            let sh: i8 = (-16i8) >> 2i8;   // -4 = 0xFC
            OUT_X[3] = neg_q as u8;
            OUT_X[4] = neg_r as u8;
            OUT_X[5] = sh as u8;

            // u16 call, loop, div/rem
            let r16: u16 = add_mul_acc(123u16, 45u16); // 5535 + 7 = 5542 (0x15A6)
            let s16: u16 = sum_loop(10u8);             // 55
            let d16: u16 = r16 / s16;                  // 5542 / 55 = 100
            let rem16: u16 = r16 % s16;                // 5542 % 55 = 42
            OUT_X[6] = (r16 & 0xFFu16) as u8;
            OUT_X[7] = (r16 >> 8u16) as u8;
            OUT_X[8] = d16 as u8;
            OUT_X[9] = rem16 as u8;

            // Signed i16 compares including boundaries
            let a: i16 = (-32767i16) -% 1i16;
            let b: i16 = 100i16;
            let c: i16 = -50i16;
            let cmp_bits: u8 = (if a < b { 1u8 } else { 0u8 })
                | (if b > c { 2u8 } else { 0u8 })
                | (if c >= a { 4u8 } else { 0u8 })
                | (if b <= c { 8u8 } else { 0u8 });
            OUT_X[10] = cmp_bits; // 1 | 2 | 4 = 7

            // u32 add/sub/eq
            let u0: u32 = 0x0001_00FFu32;
            let u1: u32 = u0 +% 5u32; // 0x0001_0104
            OUT_X[11] = if u1 == 0x0001_0104u32 { 1u8 } else { 0u8 };
            OUT_X[12] = if u1 != u0 { 1u8 } else { 0u8 };
        }
    "#;

    for release in [false, true] {
        let (_out, sim) = compile_and_run_ch559(src, 200_000, release);
        assert_eq!(sim.memory.xram[0x0100], 221);
        assert_eq!(sim.memory.xram[0x0101], 22);
        assert_eq!(sim.memory.xram[0x0102], 1);
        assert_eq!(sim.memory.xram[0x0103], (-6i8) as u8);
        assert_eq!(sim.memory.xram[0x0104], (-1i8) as u8);
        assert_eq!(sim.memory.xram[0x0105], (-4i8) as u8);
        assert_eq!(sim.memory.xram[0x0106], 0xA6);
        assert_eq!(sim.memory.xram[0x0107], 0x15);
        assert_eq!(sim.memory.xram[0x0108], 100);
        assert_eq!(sim.memory.xram[0x0109], 42);
        assert_eq!(sim.memory.xram[0x010A], 7);
        assert_eq!(sim.memory.xram[0x010B], 1);
        assert_eq!(sim.memory.xram[0x010C], 1);
    }
}

#[test]
fn ch559_e2e_sfr_bit_timer0_isr_using_bank1_and_spsc_queue() {
    let src = r#"
        target ch559 : mcs51 {
            space Data  : data  @ 0x08..0x7F;
            space IData : idata @ 0x80..0xFF;
            space XData : xdata @ 0x0000..0x17FF;
            space Sfr   : sfr   @ 0x80..0xFF;
            space Bit   : bit   @ 0x80..0xFF;
            stack IStack @ 0x80..0xFF;

            interrupt_controller Ch559Int {
                disable => asm!("clr EA");
                enable  => asm!("setb EA");
                source Timer0 @ irq(1);
            }
        }

        volatile static mut TMOD: u8 in Sfr @ 0x89;
        volatile static mut TL0: u8  in Sfr @ 0x8A;
        volatile static mut TH0: u8  in Sfr @ 0x8C;
        volatile static mut P1: u8   in Sfr @ 0x90;
        volatile static mut TR0: bool in Bit @ 0x8C;
        volatile static mut ET0: bool in Bit @ 0xA9;
        volatile static mut EA: bool  in Bit @ 0xAF;

        const LUT: [u8; 4] = [0x11u8, 0x22u8, 0x33u8, 0x44u8];

        #[owned_by(Timer0)]
        static mut IRQ_SEQ: u8 = 0u8;

        static mut EV_Q: Queue<u8, 4>;
        static mut OUT_BUF: [u8; 8] in XData @ 0x0200;

        #[interrupt(Timer0, using(1))]
        fn timer0_isr() {
            let val: u8 = LUT[(IRQ_SEQ & 3u8) as u16];
            let _ok: bool = EV_Q.push(val);
            IRQ_SEQ = IRQ_SEQ +% 1u8;
            P1 = IRQ_SEQ;
        }

        pub fn main() {
            // Configure Timer0 Mode 2 (8-bit auto-reload every 32 cycles)
            TMOD = 0x02u8;
            TH0 = 0xE0u8;
            TL0 = 0xE0u8;
            ET0 = true;
            EA = true;
            TR0 = true;

            let mut received: u8 = 0u8;
            let mut sum: u8 = 0u8;
            while received < 3u8 {
                let mut ev: u8 = 0u8;
                if EV_Q.pop(&mut ev) {
                    OUT_BUF[received as u16] = ev;
                    sum = sum +% ev;
                    received = received +% 1u8;
                }
            }

            TR0 = false;
            EA = false;

            let snap: u8 = critical { IRQ_SEQ };
            OUT_BUF[3] = sum;
            OUT_BUF[4] = if snap >= 3u8 { 1u8 } else { 0u8 };
        }
    "#;

    let (_out, sim) = compile_and_run_ch559(src, 200_000, false);
    assert_eq!(sim.memory.xram[0x0200], 0x11);
    assert_eq!(sim.memory.xram[0x0201], 0x22);
    assert_eq!(sim.memory.xram[0x0202], 0x33);
    assert_eq!(sim.memory.xram[0x0203], 0x11 + 0x22 + 0x33);
    assert_eq!(sim.memory.xram[0x0204], 1);
    assert!(sim.peripherals.gpio.read_port_latch(1) >= 3);
}

#[test]
fn ch559_e2e_machine_pool_nonzero_init_and_mixed_slices() {
    let src = r#"
        target ch559 : mcs51 {
            space Data  : data  @ 0x08..0x7F;
            space IData : idata @ 0x80..0xFF;
            space XData : xdata @ 0x0000..0x17FF;
            space Sfr   : sfr   @ 0x80..0xFF;
            space Bit   : bit   @ 0x80..0xFF;
            stack IStack @ 0x80..0xFF;
        }

        struct Bullet {
            x: u8,
            y: u8,
        }

        machine App {
            state Boot;
            state Active { count: u8 };
            transitions {
                Boot -> Active;
            }
            on Boot {
                fn step(&mut self, inc: u8) -> u8 {
                    if inc == 0u8 {
                        return 0u8;
                    }
                    goto Active { count: inc +% G_SEED };
                }
            }
            on Active {
                fn on_enter(&mut self) {
                    G_LOG[0] = G_LOG[0] +% 10u8;
                }
                fn step(&mut self, inc: u8) -> u8 {
                    self.count = self.count +% inc;
                    self.count
                }
            }
        }

        static mut G_SEED: u8 in Data = 40u8;
        static mut G_LOG: [u8; 5] = [5u8, 0u8, 0u8, 0u8, 0u8];
        static mut G_APP: App;
        static mut G_POOL: Pool<Bullet, 4>;
        static mut OUT_R: [u8; 8] in XData @ 0x0300;

        fn sum_bytes(s: &const [u8]) -> u8 {
            let mut acc: u8 = 0u8;
            for i in 0u16..s.len() {
                acc = acc +% s[i];
            }
            acc
        }

        pub fn main() {
            // Non-zero Data & XData init verification
            OUT_R[0] = G_SEED;   // 40
            OUT_R[1] = G_LOG[0]; // 5

            // Machine transitions + on_enter hook + method dispatch
            let r0: u8 = G_APP.step(0u8);       // stays in Boot, returns 0
            let _t: u8 = G_APP.step(2u8);       // Boot -> Active(42), on_enter sets G_LOG[0] = 15
            let r1: u8 = G_APP.step(8u8);       // Active count = 50
            OUT_R[2] = r0;
            OUT_R[3] = r1;
            OUT_R[4] = G_LOG[0]; // 15

            // Pool<Bullet, 4>
            let h0: Handle<Bullet> = G_POOL.alloc(Bullet { x: 11u8, y: 22u8 });
            let h1: Handle<Bullet> = G_POOL.alloc(Bullet { x: 33u8, y: 44u8 });
            G_POOL.free(h0);
            let h2: Handle<Bullet> = G_POOL.alloc(Bullet { x: 7u8, y: 9u8 });
            let b1: &Bullet = G_POOL.get(h1);
            let b2: &Bullet = G_POOL.get(h2);
            OUT_R[5] = b1.x +% b2.y; // 33 + 9 = 42

            // Mixed ConstSlice (ROM string literal + local RAM byte array)
            let local_arr: [u8; 3] = [1u8, 2u8, 3u8];
            let s_rom: u8 = sum_bytes(b"\x0A\x14\x1E"); // 60
            let s_ram: u8 = sum_bytes(&local_arr);      // 6
            OUT_R[6] = s_rom +% s_ram; // 66
        }
    "#;

    let (_out, sim) = compile_and_run_ch559(src, 200_000, false);
    assert_eq!(sim.memory.xram[0x0300], 40);
    assert_eq!(sim.memory.xram[0x0301], 5);
    assert_eq!(sim.memory.xram[0x0302], 0);
    assert_eq!(sim.memory.xram[0x0303], 50);
    assert_eq!(sim.memory.xram[0x0304], 15);
    assert_eq!(sim.memory.xram[0x0305], 42);
    assert_eq!(sim.memory.xram[0x0306], 66);
}

#[test]
fn ch559_e2e_issue47_aggregate_by_value_returns() {
    let src = r#"
        target ch559 : mcs51 {
            space Data  : data  @ 0x08..0x7F;
            space IData : idata @ 0x80..0xFF;
            space XData : xdata @ 0x0000..0x17FF;
            space Sfr   : sfr   @ 0x80..0xFF;
            space Bit   : bit   @ 0x80..0xFF;
            stack IStack @ 0x80..0xFF;
        }

        struct DiskOp {
            drv: u8,
            flags: u8,
            lba: u16,
            count: u16,
        }

        static mut OUT_R: [u8; 16] in XData @ 0x0400;

        pub fn make_op(drv: u8, lba: u16) -> DiskOp {
            if drv == 0u8 {
                return DiskOp { drv: 0u8, flags: 0x11u8, lba: lba, count: 1u16 };
            }
            DiskOp { drv: drv, flags: 0x22u8, lba: lba +% 10u16, count: 4u16 }
        }

        pub fn make_arr5(x: u8) -> [u8; 5] {
            [x, x +% 1u8, x +% 2u8, x +% 3u8, x +% 4u8]
        }

        pub fn main() {
            let op0: DiskOp = make_op(0u8, 0x1234u16);
            let op1: DiskOp = make_op(3u8, 0x0100u16);
            let a5: [u8; 5] = make_arr5(20u8);

            OUT_R[0] = op0.drv;
            OUT_R[1] = op0.flags;
            OUT_R[2] = (op0.lba & 0xFFu16) as u8;
            OUT_R[3] = (op0.lba >> 8u16) as u8;
            OUT_R[4] = op1.drv;
            OUT_R[5] = op1.flags;
            OUT_R[6] = (op1.lba & 0xFFu16) as u8;
            OUT_R[7] = (op1.lba >> 8u16) as u8;
            OUT_R[8] = op1.count as u8;
            OUT_R[9] = a5[0];
            OUT_R[10] = a5[4];
            OUT_R[11] = make_op(2u8, 50u16).lba as u8;
        }
    "#;

    for release in [false, true] {
        let (_out, sim) = compile_and_run_ch559(src, 200_000, release);
        assert_eq!(sim.memory.xram[0x0400], 0);
        assert_eq!(sim.memory.xram[0x0401], 0x11);
        assert_eq!(sim.memory.xram[0x0402], 0x34);
        assert_eq!(sim.memory.xram[0x0403], 0x12);
        assert_eq!(sim.memory.xram[0x0404], 3);
        assert_eq!(sim.memory.xram[0x0405], 0x22);
        assert_eq!(sim.memory.xram[0x0406], 0x0A);
        assert_eq!(sim.memory.xram[0x0407], 0x01);
        assert_eq!(sim.memory.xram[0x0408], 4);
        assert_eq!(sim.memory.xram[0x0409], 20);
        assert_eq!(sim.memory.xram[0x040A], 24);
        assert_eq!(sim.memory.xram[0x040B], 60);
    }
}
