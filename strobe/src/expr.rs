//! Stack-only expression graph construction and evaluation.
//!
//! Due to the stack-only limitation, the expression graph
//! must be strictly a tree, as we would otherwise require
//! multiple mutable references to some nodes.
//!
//! Notably, this does _not_ prevent us from using some
//! input arrays more than once, because we only require
//! _immutable_ references to the inputs, and we can have
//! as many of these as we like.
use crate::{ArrayMut, Elem, N};

/// (1xN)-to-(1xN) elementwise array operation.
pub trait UnaryFn<T: Elem>: Fn(&[T], &mut [T]) {}
impl<T: Elem, K: Fn(&[T], &mut [T])> UnaryFn<T> for K {}

/// (2xN)-to-(1xN) elementwise array operation.
pub trait BinaryFn<T: Elem>: Fn(&[T], &[T], &mut [T]) {}
impl<T: Elem, K: Fn(&[T], &[T], &mut [T])> BinaryFn<T> for K {}

/// (3xN)-to-(1xN) elementwise array operation
/// such as fused multiply-add.
pub trait TernaryFn<T>: Fn(&[T], &[T], &[T], &mut [T]) {}
impl<T: Elem, K: Fn(&[T], &[T], &[T], &mut [T])> TernaryFn<T> for K {}

/// (1xN)-to-(1x1) incremental-evaluation operation
/// such as a cumulative sum or product.
pub trait AccumulatorFn<T>: Fn(&[T], &mut T) {}
impl<T: Elem, K: Fn(&[T], &mut T)> AccumulatorFn<T> for K {}

/// Fixed-size and favorably aligned intermediate storage
/// for each expression node.
///
/// 64 is the max align req I'm aware of as of 2023-09-09 .
#[repr(align(64))]
struct Storage<T: Elem>([T; N]);

impl<T: Elem> Storage<T> {
    fn new(v: T) -> Self {
        Self([v; N])
    }
}

/// Operator kinds, categorized by dimensionality
pub(crate) enum Op<'a, T: Elem> {
    /// Array identity
    Array { v: &'a [T] },
    /// Array identity via iterator, useful for interop with non-contiguous arrays
    Iterator {
        v: &'a mut dyn Iterator<Item = &'a T>,
    },
    /// Scalar identity
    Scalar { acc: Option<Accumulator<'a, T>> },
    Unary {
        a: &'a mut Expr<'a, T>,
        f: &'a dyn UnaryFn<T>,
    },
    Binary {
        a: &'a mut Expr<'a, T>,
        b: &'a mut Expr<'a, T>,
        f: &'a dyn BinaryFn<T>,
    },
    Ternary {
        a: &'a mut Expr<'a, T>,
        b: &'a mut Expr<'a, T>,
        c: &'a mut Expr<'a, T>,
        f: &'a dyn TernaryFn<T>,
    },
}

/// A node in an elementwise array expression
/// applying an (MxN)-to-(1xN) operator.
pub struct Expr<'a, T: Elem> {
    storage: Storage<T>,
    pub(crate) op: Op<'a, T>,
    cursor: usize,
    len: usize,
}

impl<'a, T: Elem> Expr<'_, T> {
    pub(crate) fn new(v: T, op: Op<'a, T>, len: usize) -> Expr<'a, T> {
        Expr {
            storage: Storage::new(v),
            op,
            cursor: 0,
            len,
        }
    }

    /// Evaluate the expression, allocating storage for the output.
    ///
    /// Requires at least one array input in the expression in order for
    /// the expression to inherit a finite length.
    ///
    /// # Panics
    /// * If the array length of the input expression does not match the length of the output.
    /// * On any difference between actual and expected source data length (although the intent
    ///   is to logically eliminate this)
    #[cfg(feature = "std")]
    pub fn eval(&mut self) -> Vec<T> {
        let mut out = vec![T::zero(); self.len()];
        self.eval_into(&mut out);
        out
    }

    /// Evaluate the expression, writing values into a preallocated array.
    ///
    /// Requires at least one array input in the expression in order for
    /// the expression to inherit a finite length.
    ///
    /// # Panics
    /// * If the array length of the input expression does not match the length of the output.
    /// * On any difference between actual and expected source data length (although the intent
    ///   is to logically eliminate this)
    pub fn eval_into(&mut self, out: &mut ArrayMut<'a, T>) {
        self.eval_into_slice(out.as_mut());
    }

    /// Evaluate the expression, writing values into a preallocated slice.
    ///
    /// Requires at least one array input in the expression in order for
    /// the expression to inherit a finite length.
    ///
    /// # Panics
    /// * If the array length of the input expression does not match the length of the output.
    /// * On any difference between actual and expected source data length (although the intent
    ///   is to logically eliminate this)
    pub fn eval_into_slice(&mut self, out: &mut [T]) {
        assert_eq!(self.len(), out.len());

        let mut cursor = self.cursor;
        while let Some((x, m)) = self.next() {
            let start = cursor;
            let end = start + m;
            cursor = end;
            out[start..end].copy_from_slice(x);
        }
    }

    /// Evaluate the next chunk
    ///
    /// ## Panics
    /// * On any difference between actual and expected source data length (although the intent
    ///   is to logically eliminate this)
    fn next(&'a mut self) -> Option<(&[T], usize)> {
        use Op::*;
        let n = self.len();
        let mut cursor = self.cursor;

        let ret = match &mut self.op {
            Array { v } => {
                if self.cursor >= n {
                    None
                } else {
                    let end = n.min(self.cursor + N);
                    let start = cursor;
                    let m = end - start;
                    cursor = end;
                    // Copy into local storage to make sure lifetimes line up and align is controlled
                    self.storage.0[0..m].copy_from_slice(&v[start..end]);
                    Some((&self.storage.0[..m], m))
                }
            }
            Iterator { v } => {
                if self.cursor >= n {
                    None
                } else {
                    let end = n.min(self.cursor + N);
                    let start = cursor;
                    let m = end - start;
                    cursor = end;
                    // Copy into local storage to make sure lifetimes line up and align is controlled
                    //
                    // While an unwrap is used here for clarity, there is not much performance
                    // difference between this, which provides some guarantee that we will not
                    // return incorrect results, and the alternative, which is to do something
                    // more idiomatic like `v.take(m).zip(...).for_each(...)`. Because we check
                    // the length of the iterator before the start of evaluation and only take
                    // at most the number of values known to be in the iterator, this unwrap
                    // should never result in a panic unless there has been a real error in logic.
                    (0..m).for_each(|i| self.storage.0[i] = *v.next().unwrap());
                    Some((&self.storage.0[..m], m))
                }
            }
            Scalar { acc } => {
                if let Some(acc) = acc {
                    // If we haven't evaluated the accumulator yet, do it now
                    let v = match acc.v {
                        None => acc.eval(),
                        Some(v) => v,
                    };

                    if self.storage.0[0] != v {
                        self.storage = Storage([v; N]);
                    }
                };

                Some((&self.storage.0[..], N))
            }
            Unary { a, f } => match a.next() {
                Some((x, m)) => {
                    cursor += m;
                    f(&x[0..m], &mut self.storage.0[0..m]);
                    Some((&self.storage.0[0..m], m))
                }
                _ => None,
            },
            Binary { a, b, f } => match (a.next(), b.next()) {
                (Some((x, p)), Some((y, q))) => {
                    let m = p.min(q);
                    cursor += m;
                    f(&x[0..m], &y[0..m], &mut self.storage.0[0..m]);
                    Some((&self.storage.0[0..m], m))
                }
                _ => None,
            },
            Ternary { a, b, c, f } => match (a.next(), b.next(), c.next()) {
                (Some((x, p)), Some((y, q)), Some((z, r))) => {
                    let m = p.min(q.min(r));
                    cursor += m;
                    f(&x[0..m], &y[0..m], &z[0..m], &mut self.storage.0[0..m]);
                    Some((&self.storage.0[0..m], m))
                }
                _ => None,
            },
        };

        self.cursor = cursor;
        ret
    }

    /// The length of the output of the array expression.
    /// This inherits the minimum length of any input.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.len
    }
}

/// A many-to-one adapter for the output of an expression.
pub struct Accumulator<'a, T: Elem> {
    pub(crate) v: Option<T>,
    pub(crate) start: T,
    pub(crate) a: &'a mut Expr<'a, T>,
    pub(crate) f: &'a dyn AccumulatorFn<T>,
}

impl<'a, T: Elem> Accumulator<'a, T> {
    /// Evaluate the input expression and accumulate the output values.
    pub fn eval(&mut self) -> T {
        let f = self.f;
        let mut v = self.start;
        while let Some((x, _m)) = self.a.next() {
            f(x, &mut v);
        }

        self.v = Some(v);
        v
    }
}
