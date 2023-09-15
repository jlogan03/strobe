//! Implementations of operations that can be applied at expression nodes.
//!
//! Broadly speaking, we have three categories of operations:
//! * Identity:     `array`, `slice`, and `constant` are inputs from outside the expression
//! * N-ary:        Operations like `mul`, `add`, etc
//! * Accumulators: Operations that aggregate a full array to a scalar (like `sum`)
//!
//! Custom operations can be assembled using the `unary`, `binary`, `ternary`, and
//! `accumulator` functions, along with a reference to a matching function or closure.
//!
//! For example, you might want to use powi, which isn't treated directly here because
//! it requires configuration using a different type from the element (the integer power):
//! ```rust
//! use strobe::{array, unary};
//!
//! let x = [0.0_f64, 1.0, 2.0];
//! let mut xn = array(&x);  // Input expression node for x
//!
//! let sq_func = |a: &[f64], out: &mut [f64]| { (0..a.len()).for_each(|i| {out[i] = x[i].powi(2)}) };
//! let xsq = unary(&mut xn, &sq_func).eval();
//!
//! (0..x.len()).for_each(|i| {assert_eq!(x[i] * x[i], xsq[i])});
//! ```
use crate::expr::{Accumulator, AccumulatorFn, BinaryFn, Expr, Op, TernaryFn, UnaryFn};
use crate::{Array, Elem};
use num_traits::{Float, MulAdd};

/// Array identity operation. This allows the use of Vec, etc. as inputs.
pub fn array<'a, T: Elem>(v: &'a Array<T>) -> Expr<'a, T> {
    use Op::Array;
    Expr::new(T::zero(), Array { v: v.as_ref() }, v.as_ref().len())
}

/// Slice identity operation. This allows the use of a slice as an input.
pub fn slice<T: Elem>(v: &[T]) -> Expr<'_, T> {
    use Op::Array;
    Expr::new(T::zero(), Array { v }, v.len())
}

/// A scalar identity operation is always either a constant or the
/// output of an accumulator to be used in a downstream expression.
fn scalar<T: Elem>(v: T, acc: Option<Accumulator<'_, T>>) -> Expr<'_, T> {
    use Op::Scalar;
    Expr::new(v, Scalar { acc }, usize::MAX)
}

/// Constant identity operation. This allows use of a constant value as input.
pub fn constant<'a, T: Elem>(v: T) -> Expr<'a, T> {
    scalar(v, None)
}

/// Assemble an arbitrary (1xN)-to-(1x1) operation.
pub fn accumulator<'a, T: Elem>(
    start: T,
    a: &'a mut Expr<'a, T>,
    f: &'a dyn AccumulatorFn<T>,
) -> Accumulator<'a, T> {
    Accumulator {
        v: None,
        start,
        a,
        f,
    }
}

/// Assemble an arbitrary (1xN)-to-(1xN) operation.
pub fn unary<'a, T: Elem>(a: &'a mut Expr<'a, T>, f: &'a dyn UnaryFn<T>) -> Expr<'a, T> {
    use Op::Unary;
    let n = a.len();
    Expr::new(T::zero(), Unary { a, f }, n)
}

/// Assemble an arbitrary (2xN)-to-(1xN) operation.
pub fn binary<'a, T: Elem>(
    a: &'a mut Expr<'a, T>,
    b: &'a mut Expr<'a, T>,
    f: &'a dyn BinaryFn<T>,
) -> Expr<'a, T> {
    use Op::Binary;
    let n = a.len().min(b.len());
    Expr::new(T::zero(), Binary { a, b, f }, n)
}

/// Assemble an arbitrary (3xN)-to-(1xN) operation.
pub fn ternary<'a, T: Elem>(
    a: &'a mut Expr<'a, T>,
    b: &'a mut Expr<'a, T>,
    c: &'a mut Expr<'a, T>,
    f: &'a dyn TernaryFn<T>,
) -> Expr<'a, T> {
    use Op::Ternary;
    let n = a.len().min(b.len().min(c.len()));
    Expr::new(T::zero(), Ternary { a, b, c, f }, n)
}

fn add_inner<T: Elem>(left: &[T], right: &[T], out: &mut [T]) {
    (0..left.len()).for_each(|i| out[i] = left[i] + right[i]);
}

/// Elementwise addition
pub fn add<'a, T: Elem>(left: &'a mut Expr<'a, T>, right: &'a mut Expr<'a, T>) -> Expr<'a, T> {
    binary(left, right, &add_inner)
}

fn sub_inner<T: Elem>(left: &[T], right: &[T], out: &mut [T]) {
    (0..left.len()).for_each(|i| out[i] = left[i] - right[i]);
}

/// Elementwise subtraction
pub fn sub<'a, T: Elem>(left: &'a mut Expr<'a, T>, right: &'a mut Expr<'a, T>) -> Expr<'a, T> {
    binary(left, right, &sub_inner)
}

fn mul_inner<T: Elem>(left: &[T], right: &[T], out: &mut [T]) {
    (0..left.len()).for_each(|i| out[i] = left[i] * right[i]);
}

/// Elementwise multiplication
pub fn mul<'a, T: Elem>(left: &'a mut Expr<'a, T>, right: &'a mut Expr<'a, T>) -> Expr<'a, T> {
    binary(left, right, &mul_inner)
}

fn div_inner<T: Elem>(left: &[T], right: &[T], out: &mut [T]) {
    (0..left.len()).for_each(|i| out[i] = left[i] / right[i]);
}

/// Elementwise division
pub fn div<'a, T: Elem>(numer: &'a mut Expr<'a, T>, denom: &'a mut Expr<'a, T>) -> Expr<'a, T> {
    binary(numer, denom, &div_inner)
}

fn powf_inner<T: Float>(x: &[T], y: &[T], out: &mut [T]) {
    (0..x.len()).for_each(|i| out[i] = x[i].powf(y[i]));
}

/// Elementwise float exponent for float types
pub fn powf<'a, T: Float>(a: &'a mut Expr<'a, T>, b: &'a mut Expr<'a, T>) -> Expr<'a, T> {
    binary(a, b, &powf_inner)
}

fn flog2_inner<T: Float>(x: &[T], out: &mut [T]) {
    (0..x.len()).for_each(|i| out[i] = x[i].log2());
}

/// Elementwise log base 2 for float types
pub fn flog2<'a, T: Float>(a: &'a mut Expr<'a, T>) -> Expr<'a, T> {
    unary(a, &flog2_inner)
}

fn flog10_inner<T: Float>(x: &[T], out: &mut [T]) {
    (0..x.len()).for_each(|i| out[i] = x[i].log10());
}

/// Elementwise log base 10 for float types
pub fn flog10<'a, T: Float>(a: &'a mut Expr<'a, T>) -> Expr<'a, T> {
    unary(a, &flog10_inner)
}

fn exp_inner<T: Float>(x: &[T], out: &mut [T]) {
    (0..x.len()).for_each(|i| out[i] = x[i].exp());
}

/// Elementwise e^x for float types
pub fn exp<'a, T: Float>(a: &'a mut Expr<'a, T>) -> Expr<'a, T> {
    unary(a, &exp_inner)
}

fn atan2_inner<T: Float>(y: &[T], x: &[T], out: &mut [T]) {
    (0..x.len()).for_each(|i| out[i] = y[i].atan2(x[i]));
}

/// Elementwise atan2(y, x) for float types. Produces correct results where atan
/// would produce errors due to the singularity in the tangent function.
///
/// In accordance with tradition, the inputs are taken in (`y`, `x`) order
/// and evaluated like `y.atan2(x)`.
pub fn atan2<'a, T: Float>(y: &'a mut Expr<'a, T>, x: &'a mut Expr<'a, T>) -> Expr<'a, T> {
    binary(y, x, &atan2_inner)
}

fn sin_inner<T: Float>(x: &[T], out: &mut [T]) {
    (0..x.len()).for_each(|i| out[i] = x[i].sin());
}

/// Elementwise sin(x) for float types
pub fn sin<'a, T: Float>(a: &'a mut Expr<'a, T>) -> Expr<'a, T> {
    unary(a, &sin_inner)
}

fn tan_inner<T: Float>(x: &[T], out: &mut [T]) {
    (0..x.len()).for_each(|i| out[i] = x[i].tan());
}

/// Elementwise tan(x) for float types
pub fn tan<'a, T: Float>(a: &'a mut Expr<'a, T>) -> Expr<'a, T> {
    unary(a, &tan_inner)
}

fn cos_inner<T: Float>(x: &[T], out: &mut [T]) {
    (0..x.len()).for_each(|i| out[i] = x[i].cos());
}

/// Elementwise cos(x) for float types
pub fn cos<'a, T: Float>(a: &'a mut Expr<'a, T>) -> Expr<'a, T> {
    unary(a, &cos_inner)
}

fn asin_inner<T: Float>(x: &[T], out: &mut [T]) {
    (0..x.len()).for_each(|i| out[i] = x[i].asin());
}

/// Elementwise asin(x) for float types
pub fn asin<'a, T: Float>(a: &'a mut Expr<'a, T>) -> Expr<'a, T> {
    unary(a, &asin_inner)
}

fn acos_inner<T: Float>(x: &[T], out: &mut [T]) {
    (0..x.len()).for_each(|i| out[i] = x[i].acos());
}

/// Elementwise acos(x) for float types
pub fn acos<'a, T: Float>(a: &'a mut Expr<'a, T>) -> Expr<'a, T> {
    unary(a, &acos_inner)
}

fn atan_inner<T: Float>(x: &[T], out: &mut [T]) {
    (0..x.len()).for_each(|i| out[i] = x[i].atan());
}

/// Elementwise atan(x) for float types
///
/// This function will produce erroneous results near multiple of pi/2.
/// For a version that maintains correctness near singularities in tan(x),
/// see `atan2`.
pub fn atan<'a, T: Float>(a: &'a mut Expr<'a, T>) -> Expr<'a, T> {
    unary(a, &atan_inner)
}

fn sinh_inner<T: Float>(x: &[T], out: &mut [T]) {
    (0..x.len()).for_each(|i| out[i] = x[i].sinh());
}

/// Elementwise sinh(x) for float types
pub fn sinh<'a, T: Float>(a: &'a mut Expr<'a, T>) -> Expr<'a, T> {
    unary(a, &sinh_inner)
}

fn cosh_inner<T: Float>(x: &[T], out: &mut [T]) {
    (0..x.len()).for_each(|i| out[i] = x[i].cosh());
}

/// Elementwise cosh(x) for float types
pub fn cosh<'a, T: Float>(a: &'a mut Expr<'a, T>) -> Expr<'a, T> {
    unary(a, &cosh_inner)
}

fn tanh_inner<T: Float>(x: &[T], out: &mut [T]) {
    (0..x.len()).for_each(|i| out[i] = x[i].tanh());
}

/// Elementwise tanh(x) for float types
pub fn tanh<'a, T: Float>(a: &'a mut Expr<'a, T>) -> Expr<'a, T> {
    unary(a, &tanh_inner)
}

fn asinh_inner<T: Float>(x: &[T], out: &mut [T]) {
    (0..x.len()).for_each(|i| out[i] = x[i].asinh());
}

/// Elementwise asinh(x) for float types
pub fn asinh<'a, T: Float>(a: &'a mut Expr<'a, T>) -> Expr<'a, T> {
    unary(a, &asinh_inner)
}

fn acosh_inner<T: Float>(x: &[T], out: &mut [T]) {
    (0..x.len()).for_each(|i| out[i] = x[i].acosh());
}

/// Elementwise acosh(x) for float types
pub fn acosh<'a, T: Float>(a: &'a mut Expr<'a, T>) -> Expr<'a, T> {
    unary(a, &acosh_inner)
}

fn atanh_inner<T: Float>(x: &[T], out: &mut [T]) {
    (0..x.len()).for_each(|i| out[i] = x[i].atanh());
}

/// Elementwise atanh(x) for float types
pub fn atanh<'a, T: Float>(a: &'a mut Expr<'a, T>) -> Expr<'a, T> {
    unary(a, &atanh_inner)
}

fn abs_inner<T: Float>(x: &[T], out: &mut [T]) {
    (0..x.len()).for_each(|i| out[i] = x[i].abs());
}

/// Elementwise abs(x) for float types
pub fn abs<'a, T: Float>(a: &'a mut Expr<'a, T>) -> Expr<'a, T> {
    unary(a, &abs_inner)
}

fn mul_add_inner<T: Elem + MulAdd<T, Output = T>>(a: &[T], b: &[T], c: &[T], out: &mut [T]) {
    (0..a.len()).for_each(|i| out[i] = a[i].mul_add(b[i], c[i]));
}

/// Elementwise fused multiply-add
///
/// If the compilation target supports FMA (fused multiply-add)
/// and `-Ctarget-feature=fma` is given to rustc, this
/// performs the multiplication and addition in a single operation with
/// a single roundoff error, and can provide a significant improvement
/// in either or both of speed and float error.
///
/// However, if the compilation target does _not_ support FMA
/// or if FMA is not enabled, this will be much slower than a
/// separate multiply and add, because it will not vectorize.
pub fn mul_add<'a, T: Elem + MulAdd<T, Output = T>>(
    a: &'a mut Expr<'a, T>,
    b: &'a mut Expr<'a, T>,
    c: &'a mut Expr<'a, T>,
) -> Expr<'a, T> {
    ternary(a, b, c, &mul_add_inner)
}

fn sum_inner<T: Elem>(x: &[T], v: &mut T) {
    (0..x.len()).for_each(|i| *v = *v + x[i]);
}

/// Cumulative sum of array elements.
///
/// Note that while it is allowed, applying this to an expression with
/// a scalar operation will produce meaningless results.
pub fn sum<'a, T: Elem>(a: &'a mut Expr<'a, T>) -> Expr<'a, T> {
    let acc = Some(accumulator(T::zero(), a, &sum_inner));
    scalar(T::zero(), acc)
}
