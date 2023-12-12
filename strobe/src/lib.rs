//! Fast, low-memory, elementwise array expressions on the stack.
//! Compatible with no-std (and no-alloc) environments.
//!
//! This crate provides array expressions of arbitrary depth and executes
//! without allocation (if an output array is provided) or with a single
//! allocation only for the output array.
//!
//! Vectorization over segments of the data is achieved whenever the
//! inner function is compatible with the compiler's loop vectorizer
//! and matches the available (and enabled) SIMD instruction sets on
//! the target CPU.
//!
//! ## Who is it for?
//! If you're
//! * doing multi-step array operations of size >> 64 elements,
//! * and can formulate your array expression as a tree,
//! * and are bottlenecked on allocation or can't allocate at all,
//! * or need concretely bounded memory usage,
//! * or are memory _or_ CPU usage constrained,
//! * or don't have the standard library,
//! * or want to bag a speedup without hand-vectorizing,
//!
//! then this may be helpful.
//!
//! ## Who is it _not_ for?
//! If, however, you're
//! * doing single-step or small-size array operations,
//! * or can't reasonably formulate your expression as a tree,
//! * or can get the performance you need from a library with excellent ergonomics, like `ndarray`,
//! * or need absolutely the fastest and lowest-memory method possible
//!   and are willing and able and have time to hand-vectorize your application,
//!
//! then this may not be helpful at all.
//!
//! # Example: (A - B) * (C + D) with one allocation
//! ```rust
//! use strobe::{Expr, array, add, sub, mul};
//!
//! // Generate some arrays to operate on
//! const NT: usize = 10_000;
//! let a = vec![1.25_f64; NT];
//! let b = vec![-5.32; NT];
//! let c = vec![1e-3; NT];
//! let d = vec![3.14; NT];
//!
//! // Associate those arrays with inputs to the expression
//! let an: &mut Expr<'_, _, 64> = &mut array(&a);
//! let bn = &mut array(&b);
//! let cn = &mut array(&c);
//! let dn = &mut array(&d);
//!
//! // Build the expression tree, then evaluate,
//! // allocating once for the output array purely for convenience
//! let y = mul(&mut sub(an, bn), &mut add(cn, dn)).eval().unwrap();
//!
//! // Check results for consistency
//! (0..NT).for_each(|i| { assert_eq!(y[i], (a[i] - b[i]) * (c[i] + d[i]) ) });
//! ```
//!
//! # Example: Evaluation with zero allocation
//! While we use a simple example here, any strobe expression can be
//! evaluated into existing storage in this way.
//! ```rust
//! use strobe::{Expr, array, mul};
//!
//! // Generate some arrays to operate on
//! const NT: usize = 10_000;
//! let a = vec![1.25_f64; NT];
//! let an0: &mut Expr<'_, _, 64> = &mut array(&a);  // Two input nodes from `a`, for brevity
//! let an1 = &mut array(&a);
//!
//! // Pre-allocate storage
//! let mut y = vec![0.0; NT];
//!
//! // Build the expression tree, then evaluate into preallocated storage.
//! mul(an0, an1).eval_into(&mut y);
//!
//! // Check results for consistency
//! (0..NT).for_each(|i| { assert_eq!(y[i], a[i] * a[i] ) });
//! ```
//!
//! # Example: Custom expression nodes
//! Many common functions are already implemented. Ones that are not
//! can be assembled using the `unary`, `binary`, `ternary`, and
//! `accumulator` functions along with a matching function pointer
//! or closure.
//! ```rust
//! use strobe::{Expr, array, unary};
//!
//! let x = [0.0_f64, 1.0, 2.0];
//! let mut xn: Expr<'_, _, 64> = array(&x);
//!
//! let sq_func = |a: &[f64], out: &mut [f64]| { (0..a.len()).for_each(|i| {out[i] = a[i].powi(2)}); Ok(()) };
//! let xsq = unary(&mut xn, sq_func).eval().unwrap();
//!
//! (0..x.len()).for_each(|i| {assert_eq!(x[i] * x[i], xsq[i])});
//! ```
//!
//! # Conspicuous Design Decisions and UAQ (Un-Asked Questions)
//! * Why not implement the standard num ops?
//!     * Because we can't guarantee the lifetime of the returned node will match the lifetime of the references it owns,
//!       unless the outside scope is guaranteed to own both the inputs and outputs.
//! * Why abandon the more idiomatic functional programming regime at the interface?
//!     * Same reason we don't have the standard num ops - unfortunately, this usage breaks lifetime guarantees.
//! * Why do expressions need to be strict trees?
//!     * Because we are strictly on the stack, and we need mutable references to each node in order to use intermediate storage,
//!       which we need in order to allow vectorization,
//!       and since we can only have one mutable reference to a anything, this naturally produces a tree structure.
//!     * _However_ since the input arrays do not need to be mutable, more than one expression node can refer to a given input array,
//!       which resolves many (but not all) potential cases where the strict tree structure might prove inadequate.
//! * I can hand-vectorize my array operations and do them even faster with even less memory!
//!     * Cool! Have fun with that.
#![cfg_attr(not(feature = "std"), no_std)]
use num_traits::Num;

#[cfg(test)]
use no_panic::no_panic;

#[cfg(not(feature = "align_selected"))]
compile_error!(
    "At least one alignment feature must be selected from
        align_rust
        align_1
        align_2
        align_4
        align_8
        align_16
        align_32
        align_64
        align_128
        align_256
        align_512
        align_1024
"
);

/// Minimally-constrained numeric element
pub trait Elem: Num + Copy {}
impl<T: Num + Copy> Elem for T {}

/// Unopinionated array type spanning Vec, fixed-size arrays, etc
pub type Array<'a, T> = dyn AsRef<[T]> + 'a;
/// Unopinionated (mutable) array type spanning Vec, fixed-size arrays, etc
pub type ArrayMut<'a, T> = dyn AsMut<[T]> + 'a;

pub mod expr;
pub mod ops;

pub use expr::{AccumulatorFn, BinaryFn, Expr, TernaryFn, UnaryFn};

pub use ops::{
    abs, accumulator, acos, acosh, add, array, asin, atan, atan2, atanh, binary, constant,
    cos, cosh, div, eq, exp, flog10, flog2, ge, gt, le, lt, mul, mul_add, ne, powf, sin,
    sinh, slice, sub, sum, tan, tanh, ternary, unary,
};

impl<'a, T: Elem, const N: usize, K> From<&'a K> for Expr<'a, T, N>
where
    K: AsRef<[T]>,
{
    #[cfg_attr(test, no_panic)]
    fn from(value: &'a K) -> Self {
        array(value)
    }
}

impl<'a, T: Elem, const N: usize> From<&'a [T]> for Expr<'a, T, N> {
    #[cfg_attr(test, no_panic)]
    fn from(value: &'a [T]) -> Self {
        slice(value)
    }
}

#[cfg(test)]
mod test_std {
    /// Make sure we have std enabled for tests
    /// so that they don't get skipped entirely
    #[test]
    fn test_std() {
        assert!(cfg!(feature = "std"));
    }
}

/// Convenience functions for generating random numbers with a fixed seed
/// to use as inputs for tests and benchmarks
#[cfg(test)]
pub(crate) mod randn {
    use rand::distributions::{Distribution, Standard};
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;

    /// Fixed random seed to support repeatable testing
    const SEED: [u8; 32] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7,
        6, 5, 4, 3, 2, 1,
    ];

    /// Get a random number generator with a const seed for repeatable testing
    pub fn rng_fixed_seed() -> StdRng {
        StdRng::from_seed(SEED)
    }

    /// Generate `n` random numbers using provided generator
    pub fn randn<T>(rng: &mut StdRng, n: usize) -> Vec<T>
    where
        Standard: Distribution<T>,
    {
        let out: Vec<T> = (0..n).map(|_| rng.gen::<T>()).collect();
        out
    }
}

#[cfg(all(test, feature = "std"))]
mod test {
    use crate::ops::{fmax, fmin, max, min};

    use super::{expr::*, randn::*, *};

    /// Number of elements to use for tests
    /// to give one full chunk and one partial chunk
    const NT: usize = 67;

    #[test]
    fn test_small_storage() {
        // Make sure parametrized storage size behaves well down to the limit
        // Simple case with depth one
        let mut rng = rng_fixed_seed();
        let x = randn::<f64>(&mut rng, NT);
        let mut xn: Expr<'_, _, 1> = array(&x);
        let out = abs(&mut xn).eval().unwrap();

        // Make sure the values match
        (0..x.len()).for_each(|i| assert_eq!(x[i].abs(), out[i]));
    }

    #[test]
    fn test_abs() {
        // Simple case with depth one
        let mut rng = rng_fixed_seed();
        let x = randn::<f64>(&mut rng, NT);
        let mut xn: Expr<'_, _, 64> = array(&x);
        let out = abs(&mut xn).eval().unwrap();

        // Make sure the values match
        (0..x.len()).for_each(|i| assert_eq!(x[i].abs(), out[i]));
    }

    #[test]
    fn test_atanh() {
        // Simple case with depth one
        let mut rng = rng_fixed_seed();
        let x = randn::<f64>(&mut rng, NT);
        let mut xn: Expr<'_, _, 64> = array(&x);
        let out = atanh(&mut xn).eval().unwrap();

        // Make sure the values match
        (0..x.len()).for_each(|i| assert_eq!(x[i].atanh(), out[i]));
    }

    #[test]
    fn test_acosh() {
        // Simple case with depth one
        let mut rng = rng_fixed_seed();
        let mut x = randn::<f64>(&mut rng, NT);
        (0..NT).for_each(|i| x[i] += 1.5); // Offset to avoid comparing nan values

        let mut xn: Expr<'_, _, 64> = array(&x);
        let out = acosh(&mut xn).eval().unwrap();

        // Make sure the values match
        (0..x.len()).for_each(|i| assert_eq!(x[i].acosh(), out[i]));
    }

    #[test]
    fn test_tanh() {
        // Simple case with depth one
        let mut rng = rng_fixed_seed();
        let x = randn::<f64>(&mut rng, NT);
        let mut xn: Expr<'_, _, 64> = array(&x);
        let out = tanh(&mut xn).eval().unwrap();

        // Make sure the values match
        (0..x.len()).for_each(|i| assert_eq!(x[i].tanh(), out[i]));
    }

    #[test]
    fn test_cosh() {
        // Simple case with depth one
        let mut rng = rng_fixed_seed();
        let x = randn::<f64>(&mut rng, NT);
        let mut xn: Expr<'_, _, 64> = array(&x);
        let out = cosh(&mut xn).eval().unwrap();

        // Make sure the values match
        (0..x.len()).for_each(|i| assert_eq!(x[i].cosh(), out[i]));
    }

    #[test]
    fn test_sinh() {
        // Simple case with depth one
        let mut rng = rng_fixed_seed();
        let x = randn::<f64>(&mut rng, NT);
        let mut xn: Expr<'_, _, 64> = array(&x);
        let out = sinh(&mut xn).eval().unwrap();

        // Make sure the values match
        (0..x.len()).for_each(|i| assert_eq!(x[i].sinh(), out[i]));
    }

    #[test]
    fn test_atan() {
        // Simple case with depth one
        let mut rng = rng_fixed_seed();
        let x = randn::<f64>(&mut rng, NT);
        let mut xn: Expr<'_, _, 64> = array(&x);
        let out = atan(&mut xn).eval().unwrap();

        // Make sure the values match
        (0..x.len()).for_each(|i| assert_eq!(x[i].atan(), out[i]));
    }

    #[test]
    fn test_acos() {
        // Simple case with depth one
        let mut rng = rng_fixed_seed();
        let x = randn::<f64>(&mut rng, NT);
        let mut xn: Expr<'_, _, 64> = array(&x);
        let out = acos(&mut xn).eval().unwrap();

        // Make sure the values match
        (0..x.len()).for_each(|i| assert_eq!(x[i].acos(), out[i]));
    }

    #[test]
    fn test_asin() {
        // Simple case with depth one
        let mut rng = rng_fixed_seed();
        let x = randn::<f64>(&mut rng, NT);
        let mut xn: Expr<'_, _, 64> = array(&x);
        let out = asin(&mut xn).eval().unwrap();

        // Make sure the values match
        (0..x.len()).for_each(|i| assert_eq!(x[i].asin(), out[i]));
    }

    #[test]
    fn test_tan() {
        // Simple case with depth one
        let mut rng = rng_fixed_seed();
        let x = randn::<f64>(&mut rng, NT);
        let mut xn: Expr<'_, _, 64> = array(&x);
        let out = tan(&mut xn).eval().unwrap();

        // Make sure the values match
        (0..x.len()).for_each(|i| assert_eq!(x[i].tan(), out[i]));
    }

    #[test]
    fn test_cos() {
        // Simple case with depth one
        let mut rng = rng_fixed_seed();
        let x = randn::<f64>(&mut rng, NT);
        let mut xn: Expr<'_, _, 64> = array(&x);
        let out = cos(&mut xn).eval().unwrap();

        // Make sure the values match
        (0..x.len()).for_each(|i| assert_eq!(x[i].cos(), out[i]));
    }

    #[test]
    fn test_sin() {
        // Simple case with depth one
        let mut rng = rng_fixed_seed();
        let x = randn::<f64>(&mut rng, NT);
        let mut xn: Expr<'_, _, 64> = array(&x);
        let out = sin(&mut xn).eval().unwrap();

        // Make sure the values match
        (0..x.len()).for_each(|i| assert_eq!(x[i].sin(), out[i]));
    }

    #[test]
    fn test_atan2() {
        // Simple case with depth one
        let mut rng = rng_fixed_seed();
        let x = randn::<f64>(&mut rng, NT);
        let y = randn::<f64>(&mut rng, NT);

        let mut xn: Expr<'_, _, 64> = array(&x);
        let mut yn = array(&y);

        let out = atan2(&mut yn, &mut xn).eval().unwrap();

        // Make sure the values match
        (0..y.len()).for_each(|i| assert_eq!(y[i].atan2(x[i]), out[i]));
    }

    #[test]
    fn test_exp() {
        // Simple case with depth one
        let mut rng = rng_fixed_seed();
        let y = randn::<f64>(&mut rng, NT);

        let mut yn: Expr<'_, _, 64> = array(&y);

        let out = exp(&mut yn).eval().unwrap();

        // Make sure the values match
        (0..y.len()).for_each(|i| assert_eq!(y[i].exp(), out[i]));
    }

    #[test]
    fn test_log10() {
        // Simple case with depth one
        let mut rng = rng_fixed_seed();
        let y = randn::<f64>(&mut rng, NT);

        let mut yn: Expr<'_, _, 64> = array(&y);

        let out = flog10(&mut yn).eval().unwrap();

        // Make sure the values match
        (0..y.len()).for_each(|i| assert_eq!(y[i].log10(), out[i]));
    }

    #[test]
    fn test_log2() {
        // Simple case with depth one
        let mut rng = rng_fixed_seed();
        let y = randn::<f64>(&mut rng, NT);

        let mut yn: Expr<'_, _, 64> = array(&y);

        let out = flog2(&mut yn).eval().unwrap();

        // Make sure the values match
        (0..y.len()).for_each(|i| assert_eq!(y[i].log2(), out[i]));
    }

    #[test]
    fn test_powf() {
        // Simple case with depth one
        let mut rng = rng_fixed_seed();
        let y = randn::<f64>(&mut rng, NT);

        let mut xn: Expr<'_, _, 64> = constant(1.5);
        let mut yn = array(&y);

        let out = powf(&mut yn, &mut xn).eval().unwrap();

        // Make sure the values match
        (0..y.len()).for_each(|i| assert_eq!(y[i].powf(1.5), out[i]));
    }

    #[test]
    fn test_mul_add() {
        // Simple case with one mul_add
        let mut rng = rng_fixed_seed();
        let y = randn::<f64>(&mut rng, NT);
        let z = randn::<f64>(&mut rng, NT);

        let mut xn: Expr<'_, _, 64> = constant(5.0);
        let mut yn = array(&y);
        let mut zn = array(&z);

        let out = mul_add(&mut yn, &mut zn, &mut xn).eval().unwrap();

        // Make sure the values match
        (0..y.len()).for_each(|i| assert_eq!(y[i].mul_add(z[i], 5.0), out[i]));
    }

    #[test]
    fn test_min() {
        let y: Vec<usize> = (0..NT).collect();

        let mut xn: Expr<'_, _, 64> = constant(1);
        let mut yn = array(&y);

        let mut xyn = min(&mut yn, &mut xn);

        let out = xyn.eval().unwrap();

        // Make sure the values match
        (0..y.len()).for_each(|i| assert_eq!(y[i].min(1), out[i]));
    }

    #[test]
    fn test_max() {
        let y: Vec<usize> = (0..NT).collect();

        let mut xn: Expr<'_, _, 64> = constant(1);
        let mut yn = array(&y);

        let mut xyn = max(&mut yn, &mut xn);

        let out = xyn.eval().unwrap();

        // Make sure the values match
        (0..y.len()).for_each(|i| assert_eq!(y[i].max(1), out[i]));
    }

    #[test]
    fn test_fmin() {
        let mut rng = rng_fixed_seed();
        let y = randn::<f64>(&mut rng, NT);

        let mut xn: Expr<'_, _, 64> = constant(0.5);
        let mut yn = array(&y);

        let mut xyn = fmin(&mut yn, &mut xn);

        let out = xyn.eval().unwrap();

        // Make sure the values match
        (0..y.len()).for_each(|i| assert_eq!(y[i].min(0.5), out[i]));
    }

    #[test]
    fn test_fmax() {
        let mut rng = rng_fixed_seed();
        let y = randn::<f64>(&mut rng, NT);

        let mut xn: Expr<'_, _, 64> = constant(0.5);
        let mut yn = array(&y);

        let mut xyn = fmax(&mut yn, &mut xn);

        let out = xyn.eval().unwrap();

        // Make sure the values match
        (0..y.len()).for_each(|i| assert_eq!(y[i].max(0.5), out[i]));
    }

    #[test]
    fn test_lt() {
        // Simple case with one scalar-vector op
        let mut rng = rng_fixed_seed();
        let y = randn::<f64>(&mut rng, NT);

        let mut xn: Expr<'_, _, 64> = constant(0.5);
        let mut yn = array(&y);

        let mut xyn = lt(&mut yn, &mut xn);

        let xy = xyn.eval().unwrap();

        // Make sure the values match
        (0..y.len()).for_each(|i| assert_eq!(f64::from(y[i] < 0.5), xy[i]));
    }

    #[test]
    fn test_le() {
        // Simple case with one scalar-vector op
        let mut rng = rng_fixed_seed();
        let y = randn::<f64>(&mut rng, NT);

        let mut xn: Expr<'_, _, 64> = constant(0.5);
        let mut yn = array(&y);

        let mut xyn = le(&mut yn, &mut xn);

        let xy = xyn.eval().unwrap();

        // Make sure the values match
        (0..y.len()).for_each(|i| assert_eq!(f64::from(y[i] <= 0.5), xy[i]));
    }

    #[test]
    fn test_gt() {
        // Simple case with one scalar-vector op
        let mut rng = rng_fixed_seed();
        let y = randn::<f64>(&mut rng, NT);

        let mut xn: Expr<'_, _, 64> = constant(0.5);
        let mut yn = array(&y);

        let mut xyn = gt(&mut yn, &mut xn);

        let xy = xyn.eval().unwrap();

        // Make sure the values match
        (0..y.len()).for_each(|i| assert_eq!(f64::from(y[i] > 0.5), xy[i]));
    }

    #[test]
    fn test_ge() {
        // Simple case with one scalar-vector op
        let mut rng = rng_fixed_seed();
        let y = randn::<f64>(&mut rng, NT);

        let mut xn: Expr<'_, _, 64> = constant(0.5);
        let mut yn = array(&y);

        let mut xyn = ge(&mut yn, &mut xn);

        let xy = xyn.eval().unwrap();

        // Make sure the values match
        (0..y.len()).for_each(|i| assert_eq!(f64::from(y[i] >= 0.5), xy[i]));
    }

    #[test]
    fn test_eq() {
        // Simple case with one scalar-vector op
        let y = Vec::from([0.0, 0.5, 1.0]);

        let mut xn: Expr<'_, _, 64> = constant(0.5);
        let mut yn = array(&y);

        let mut xyn = eq(&mut yn, &mut xn);

        let xy = xyn.eval().unwrap();

        // Make sure the values match
        (0..y.len()).for_each(|i| assert_eq!(f64::from(y[i] == 0.5), xy[i]));
    }

    #[test]
    fn test_ne() {
        // Simple case with one scalar-vector op
        let y = Vec::from([0.0, 0.5, 1.0]);

        let mut xn: Expr<'_, _, 64> = constant(0.5);
        let mut yn = array(&y);

        let mut xyn = ne(&mut yn, &mut xn);

        let xy = xyn.eval().unwrap();

        // Make sure the values match
        (0..y.len()).for_each(|i| assert_eq!(f64::from(y[i] != 0.5), xy[i]));
    }

    #[test]
    fn test_div() {
        // Simple case with one scalar-vector op
        let mut rng = rng_fixed_seed();
        let y = randn::<f64>(&mut rng, NT);

        let mut xn: Expr<'_, _, 64> = constant(5.0);
        let mut yn = array(&y);

        let mut xyn = div(&mut yn, &mut xn);

        let xy = xyn.eval().unwrap();

        // Make sure the values match
        (0..y.len()).for_each(|i| assert_eq!(y[i] / 5.0, xy[i]));
    }

    #[test]
    fn test_sub() {
        // Simple case with one scalar-vector op
        let mut rng = rng_fixed_seed();
        let y = randn::<f64>(&mut rng, NT);

        let mut xn: Expr<'_, _, 64> = constant(5.0);
        let mut yn = array(&y);

        let mut xyn = sub(&mut xn, &mut yn);

        let xy = xyn.eval().unwrap();

        // Make sure the values match
        (0..y.len()).for_each(|i| assert_eq!(5.0 - y[i], xy[i]));
    }

    #[test]
    fn test_add() {
        // Simple case with one scalar-vector op
        let mut rng = rng_fixed_seed();
        let y = randn::<f64>(&mut rng, NT);

        let mut xn: Expr<'_, _, 64> = constant(5.0);
        let mut yn = array(&y);

        let mut xyn = add(&mut xn, &mut yn);

        let xy = xyn.eval().unwrap();

        // Make sure the values match
        (0..y.len()).for_each(|i| assert_eq!(5.0 + y[i], xy[i]));
    }

    #[test]
    fn test_sum() {
        // Simple case with one multiplication
        let mut rng = rng_fixed_seed();
        let x = randn::<f64>(&mut rng, NT);

        let mut xn: Expr<'_, _, 64> = array(&x);

        let xsumn = sum(&mut xn);

        // We can't evaluate the expression because the scalar output value has
        // usize::MAX length, but we can evaluate the accumulator directly
        let xsum = match xsumn.op {
            Op::Scalar { acc } => acc.unwrap().eval().unwrap(),
            _ => panic!(),
        };

        // Make sure the values match
        let mut v = 0.0;
        (0..x.len()).for_each(|i| v += x[i]);

        assert_eq!(xsum, v);
    }

    #[test]
    fn test_mul_by_sum() {
        // Slightly nontrivial case where a sum is used as an input to
        // a multiplication
        let mut rng = rng_fixed_seed();
        let x = randn::<f64>(&mut rng, NT);
        let y = randn::<f64>(&mut rng, NT);

        let mut xn: Expr<'_, _, 64> = array(&x);
        let mut yn = array(&y);

        let xsumy = mul(&mut sum(&mut xn), &mut yn).eval().unwrap();

        // Make sure the values match
        //    Do the sum manually
        let mut v = 0.0;
        (0..x.len()).for_each(|i| v += x[i]);
        //    Compare
        (0..x.len()).for_each(|i| assert_eq!(y[i] * v, xsumy[i]));
    }

    #[test]
    fn test_from_slice() {
        // Simple case with one scalar-vector multiplication
        let mut rng = rng_fixed_seed();
        let y = &randn::<f64>(&mut rng, NT)[..];

        let mut xn: Expr<'_, _, 64> = constant(5.0);
        let mut yn = y.into();

        let mut xyn = mul(&mut xn, &mut yn);

        let xy = xyn.eval().unwrap();

        // Make sure the values match
        (0..y.len()).for_each(|i| assert_eq!(5.0 * y[i], xy[i]));
    }

    #[test]
    fn test_from_vec() {
        // Simple case with one scalar-vector multiplication
        let mut rng = rng_fixed_seed();
        let y = &randn::<f64>(&mut rng, NT);

        let mut xn: Expr<'_, _, 64> = constant(5.0);
        let mut yn = y.into();

        let mut xyn = mul(&mut xn, &mut yn);

        let xy = xyn.eval().unwrap();

        // Make sure the values match
        (0..y.len()).for_each(|i| assert_eq!(5.0 * y[i], xy[i]));
    }

    #[test]
    fn test_mul_scalar() {
        // Simple case with one scalar-vector multiplication
        let mut rng = rng_fixed_seed();
        let y = randn::<f64>(&mut rng, NT);

        let mut xn: Expr<'_, _, 64> = constant(5.0);
        let mut yn = array(&y);

        let mut xyn = mul(&mut xn, &mut yn);

        let xy = xyn.eval().unwrap();

        // Make sure the values match
        (0..y.len()).for_each(|i| assert_eq!(5.0 * y[i], xy[i]));
    }

    #[test]
    fn test_mul_binary() {
        // Simple case with one multiplication
        let mut rng = rng_fixed_seed();
        let x = randn::<f64>(&mut rng, NT);
        let y = randn::<f64>(&mut rng, NT);

        let mut xn: Expr<'_, _, 64> = array(&x);
        let mut yn = array(&y);

        let f = |a: &[f64], b: &[f64], out: &mut [f64]| {
            (0..a.len()).for_each(|i| out[i] = a[i] * b[i]);
            Ok(())
        };

        let mut xyn = binary(&mut xn, &mut yn, f);

        let xy = xyn.eval().unwrap();

        // Make sure the values match
        (0..x.len()).for_each(|i| assert_eq!(x[i] * y[i], xy[i]));
    }

    #[test]
    fn test_mul() {
        // Simple case with one vector-vector op
        let mut rng = rng_fixed_seed();
        let x = randn::<f64>(&mut rng, NT);
        let y = randn::<f64>(&mut rng, NT);

        let mut xn: Expr<'_, _, 64> = array(&x);
        let mut yn = array(&y);
        let mut xyn = mul(&mut xn, &mut yn);

        let xy = xyn.eval().unwrap();

        // Make sure the values match
        (0..x.len()).for_each(|i| assert_eq!(x[i] * y[i], xy[i]));
    }

    #[test]
    fn test_mul_2x() {
        // Slightly nontrivial case with two multiplications
        let mut rng = rng_fixed_seed();
        let x = randn::<f64>(&mut rng, NT);
        let y = randn::<f64>(&mut rng, NT);
        let z = randn::<f64>(&mut rng, NT);

        let mut xn: Expr<'_, _, 64> = array(&x);
        let mut yn = array(&y);
        let mut zn = array(&z);
        let mut xyn = mul(&mut xn, &mut yn);
        let mut xyzn = mul(&mut xyn, &mut zn);

        let xyz = xyzn.eval().unwrap();

        // Make sure the values match
        (0..x.len()).for_each(|i| assert_eq!(x[i] * y[i] * z[i], xyz[i]));
    }

    #[test]
    fn test_mul_3x() {
        // Slightly nontrivial case with three multiplications
        let mut rng = rng_fixed_seed();
        let x = randn::<f64>(&mut rng, NT);
        let y = randn::<f64>(&mut rng, NT);
        let z = randn::<f64>(&mut rng, NT);
        let w = randn::<f64>(&mut rng, NT);

        let xn: &mut Expr<'_, _, 64> = &mut array(&x);
        let yn = &mut array(&y);
        let zn = &mut array(&z);
        let wn = &mut array(&w);

        let xyn = &mut mul(xn, yn);
        let zwn = &mut mul(zn, wn);

        let xyzw = mul(xyn, zwn).eval().unwrap();

        // Make sure the values match
        // As the expressions get more involved, we start to see roundoff differences due to ordering
        (0..x.len()).for_each(|i| assert!(((x[i] * y[i] * z[i] * w[i]) - xyzw[i]).abs() < 1e-15));
    }
}
