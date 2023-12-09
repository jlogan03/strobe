//! Building this module successfully guarantees that the library is no-std compatible

#![no_std]
#![no_main]

use core::panic::PanicInfo;

use strobe::{add, array};

// #[panic_handler]
// fn panic(_info: &PanicInfo) -> ! {
//     // We can't print, so there's not much to do here
//     loop {}
// }

use panic_never as _;

#[no_mangle]
pub fn _start() -> ! {
    let a = [1.0_f64, 2.0, 3.0, 4.0];
    let b = [1.0_f64, 2.0, 3.0, 4.0];
    let mut c = [0.0; 4];

    let _res = add::<_, 8>(&mut array(&a), &mut array(&b)).eval_into_slice(&mut c);

    loop {} // We don't actually run this, just compile it
}
