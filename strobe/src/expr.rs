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
use crate::{ArrayMut, Elem};
use core::slice::SliceIndex;

#[cfg(test)]
use no_panic::no_panic;

/// (1xN)-to-(1xN) elementwise array operation.
pub type UnaryFn<T> = fn(&[T], &mut [T]) -> Result<(), &'static str>;

/// (2xN)-to-(1xN) elementwise array operation.
pub type BinaryFn<T> = fn(&[T], &[T], &mut [T]) -> Result<(), &'static str>;

/// (3xN)-to-(1xN) elementwise array operation
/// such as fused multiply-add.
pub type TernaryFn<T> = fn(&[T], &[T], &[T], &mut [T]) -> Result<(), &'static str>;

/// (1xN)-to-(1x1) incremental-evaluation operation
/// such as a cumulative sum or product.
pub type AccumulatorFn<T> = fn(&[T], &mut T) -> Result<(), &'static str>;

/// Operator kinds, categorized by dimensionality
pub(crate) enum Op<'a, T: Elem, const N: usize = 64> {
    /// Array identity
    Array { v: &'a [T] },
    /// Scalar identity
    Scalar { acc: Option<Accumulator<'a, T, N>> },
    Unary {
        a: &'a mut Expr<'a, T, N>,
        f: UnaryFn<T>,
    },
    Binary {
        a: &'a mut Expr<'a, T, N>,
        b: &'a mut Expr<'a, T, N>,
        f: BinaryFn<T>,
    },
    Ternary {
        a: &'a mut Expr<'a, T, N>,
        b: &'a mut Expr<'a, T, N>,
        c: &'a mut Expr<'a, T, N>,
        f: TernaryFn<T>,
    },
}

/// A node in an elementwise array expression
/// applying an (MxN)-to-(1xN) operator.
pub struct Expr<'a, T: Elem, const N: usize = 64> {
    storage: Storage<T, N>,
    pub(crate) op: Op<'a, T, N>,
    cursor: usize,
    len: usize,
}

impl<'a, T: Elem, const N: usize> Expr<'_, T, N> {
    #[cfg_attr(test, no_panic)]
    pub(crate) fn new(v: T, op: Op<'a, T, N>, len: usize) -> Expr<'a, T, N> {
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
    /// # Errors
    /// * If the array length of the input expression does not match the length of the output.
    /// * On any error in a lower-level function during evaluation
    #[cfg(feature = "std")]
    pub fn eval(&mut self) -> Result<Vec<T>, &'static str> {
        let mut out = vec![T::zero(); self.len()];
        self.eval_into(&mut out)?;
        Ok(out)
    }

    /// Evaluate the expression, writing values into a preallocated array.
    ///
    /// Requires at least one array input in the expression in order for
    /// the expression to inherit a finite length.
    ///
    /// # Errors
    /// * If the array length of the input expression does not match the length of the output.
    /// * On any error in a lower-level function during evaluation
    pub fn eval_into(&mut self, out: &mut ArrayMut<'a, T>) -> Result<(), &'static str> {
        self.eval_into_slice(out.as_mut())?;
        Ok(())
    }

    /// Evaluate the expression, writing values into a preallocated slice.
    ///
    /// Requires at least one array input in the expression in order for
    /// the expression to inherit a finite length.
    ///
    /// # Errors
    /// * If the array length of the input expression does not match the length of the output.
    /// * On any error in a lower-level function during evaluation
    pub fn eval_into_slice(&mut self, out: &mut [T]) -> Result<(), &'static str> {
        if self.len() != out.len() {
            return Err("Expression data length does not match output size");
        }

        let mut cursor = self.cursor;
        while let Some((x, m)) = self.next()? {
            let start = cursor;
            let end = start + m;
            cursor = end;
            if out.len() < end {
                return Err("Size mismatch");
            }
            copy_from_slice_fallible(&mut out[start..end], x)?;
        }
        Ok(())
    }

    /// Evaluate the next chunk
    ///
    /// # Errors
    /// * On any error in a lower-level function during evaluation
    #[cfg_attr(test, no_panic)]
    fn next(&'a mut self) -> Result<Option<(&[T], usize)>, &'static str> {
        use Op::*;
        let n = self.len();
        let nstore = self.storage.size();
        let mut cursor = self.cursor;

        let ret = match &mut self.op {
            Array { v } => {
                if self.cursor >= n {
                    None
                } else {
                    let end = n.min(self.cursor + nstore);
                    let start = cursor;
                    let m = end.saturating_sub(start);
                    cursor = end;

                    // Copy into local storage to make sure lifetimes line up and align is controlled
                    let vslice = get_from_slice_fallible(&v, start..end)?;
                    copy_from_slice_fallible(&mut self.storage.0, vslice)?;
                    let used_storage = get_from_slice_fallible(&self.storage.0, ..m)?;
                    Some((used_storage, m))
                }
            }
            Scalar { acc } => {
                if let Some(acc) = acc {
                    // If we haven't evaluated the accumulator yet, do it now
                    let v = match acc.v {
                        None => acc.eval()?,
                        Some(v) => v,
                    };

                    if self.storage.0.is_empty() {
                        return Err("Size mismatch");
                    }

                    if self.storage.0[0] != v {
                        self.storage = Storage::new(v);
                    }
                };

                Some((&self.storage.0[..], N))
            }
            Unary { a, f } => match a.next()? {
                Some((x, m)) => {
                    cursor += m;
                    if self.storage.0.len() < m || x.len() < m {
                        return Err("Size mismatch");
                    }
                    f(&x[..m], &mut self.storage.0[..m])?;
                    Some((&self.storage.0[..m], m))
                }
                _ => None,
            },
            Binary { a, b, f } => match (a.next()?, b.next()?) {
                (Some((x, p)), Some((y, q))) => {
                    let m = p.min(q);
                    cursor += m;
                    if self.storage.0.len() < m || x.len() < m || y.len() < m {
                        return Err("Size mismatch");
                    }
                    f(&x[..m], &y[..m], &mut self.storage.0[..m])?;
                    Some((&self.storage.0[..m], m))
                }
                _ => None,
            },
            Ternary { a, b, c, f } => match (a.next()?, b.next()?, c.next()?) {
                (Some((x, p)), Some((y, q)), Some((z, r))) => {
                    let m = p.min(q.min(r));
                    cursor += m;
                    if self.storage.0.len() < m || x.len() < m || y.len() < m {
                        return Err("Size mismatch");
                    }
                    f(&x[..m], &y[..m], &z[..m], &mut self.storage.0[..m])?;
                    Some((&self.storage.0[..m], m))
                }
                _ => None,
            },
        };

        self.cursor = cursor;
        Ok(ret)
    }

    /// The length of the output of the array expression.
    /// This inherits the minimum length of any input.
    #[allow(clippy::len_without_is_empty)]
    #[cfg_attr(test, no_panic)]
    pub fn len(&self) -> usize {
        self.len
    }
}

/// A many-to-one adapter for the output of an expression.
pub struct Accumulator<'a, T: Elem, const N: usize = 64> {
    pub(crate) v: Option<T>,
    pub(crate) start: T,
    pub(crate) a: &'a mut Expr<'a, T, N>,
    pub(crate) f: AccumulatorFn<T>,
}

impl<'a, T: Elem, const N: usize> Accumulator<'a, T, N> {
    /// Evaluate the input expression and accumulate the output values.
    #[cfg_attr(test, no_panic)]
    pub fn eval(&mut self) -> Result<T, &'static str> {
        let f = self.f;
        let mut v = self.start;
        while let Some((x, _m)) = self.a.next()? {
            f(x, &mut v)?;
        }

        self.v = Some(v);
        Ok(v)
    }
}

/// Fixed-size and favorably aligned intermediate storage
/// for each expression node.
#[cfg(feature = "align_rust")]
struct Storage<T: Elem, const N: usize = 64>([T; N]);

/// Fixed-size and favorably aligned intermediate storage
/// for each expression node.
#[cfg(feature = "align_1")]
#[repr(align(1))]
struct Storage<T: Elem, const N: usize = 64>([T; N]);

/// Fixed-size and favorably aligned intermediate storage
/// for each expression node.
#[cfg(feature = "align_2")]
#[repr(align(2))]
struct Storage<T: Elem, const N: usize = 64>([T; N]);

/// Fixed-size and favorably aligned intermediate storage
/// for each expression node.
#[cfg(feature = "align_4")]
#[repr(align(4))]
struct Storage<T: Elem, const N: usize = 64>([T; N]);

/// Fixed-size and favorably aligned intermediate storage
/// for each expression node.
#[cfg(feature = "align_8")]
#[repr(align(8))]
struct Storage<T: Elem, const N: usize = 64>([T; N]);

/// Fixed-size and favorably aligned intermediate storage
/// for each expression node.
#[cfg(feature = "align_16")]
#[repr(align(16))]
struct Storage<T: Elem, const N: usize = 64>([T; N]);

/// Fixed-size and favorably aligned intermediate storage
/// for each expression node.
#[cfg(feature = "align_32")]
#[repr(align(32))]
struct Storage<T: Elem, const N: usize = 64>([T; N]);

/// Fixed-size and favorably aligned intermediate storage
/// for each expression node.
#[cfg(feature = "align_64")]
#[repr(align(64))]
struct Storage<T: Elem, const N: usize = 64>([T; N]);

/// Fixed-size and favorably aligned intermediate storage
/// for each expression node.
#[cfg(feature = "align_128")]
#[repr(align(128))]
struct Storage<T: Elem, const N: usize = 64>([T; N]);

/// Fixed-size and favorably aligned intermediate storage
/// for each expression node.
#[cfg(feature = "align_256")]
#[repr(align(256))]
struct Storage<T: Elem, const N: usize = 64>([T; N]);

/// Fixed-size and favorably aligned intermediate storage
/// for each expression node.
#[cfg(feature = "align_512")]
#[repr(align(512))]
struct Storage<T: Elem, const N: usize = 64>([T; N]);

/// Fixed-size and favorably aligned intermediate storage
/// for each expression node.
#[cfg(feature = "align_1024")]
#[repr(align(1024))]
struct Storage<T: Elem, const N: usize = 64>([T; N]);

impl<T: Elem, const N: usize> Storage<T, N> {
    #[cfg_attr(test, no_panic)]
    fn new(v: T) -> Self {
        Self([v; N])
    }

    const fn size(&self) -> usize {
        N
    }
}

/// Non-panicking, inlined version of copy_from_slice
#[cfg_attr(test, no_panic)]
#[inline]
fn copy_from_slice_fallible<T: Copy>(to: &mut [T], from: &[T]) -> Result<(), &'static str> {
    let n = to.len();
    if from.len() != n {
        return Err("Size mismatch");
    };

    (0..n).for_each(|i| to[i] = from[i]);
    Ok(())
}

/// Non-panicking, inlined version of core::slice::get
#[cfg_attr(test, no_panic)]
#[inline]
fn get_from_slice_fallible<T, I>(
    s: &[T],
    index: I,
) -> Result<&<I as SliceIndex<[T]>>::Output, &'static str>
where
    I: SliceIndex<[T]>,
{
    let res = s.get(index);
    match res {
        Some(inner) => Ok(inner),
        None => Err("Size mismatch"),
    }
}
