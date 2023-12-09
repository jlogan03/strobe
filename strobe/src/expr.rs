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
use no_panic::no_panic;

/// (1xN)-to-(1xN) elementwise array operation.
pub trait UnaryFn<T: Elem>: Fn(&[T], &mut [T]) -> Result<(), &'static str> {}
impl<T: Elem, K: Fn(&[T], &mut [T]) -> Result<(), &'static str>> UnaryFn<T> for K {}

/// (2xN)-to-(1xN) elementwise array operation.
pub trait BinaryFn<T: Elem>: Fn(&[T], &[T], &mut [T]) -> Result<(), &'static str> {}
impl<T: Elem, K: Fn(&[T], &[T], &mut [T]) -> Result<(), &'static str>> BinaryFn<T> for K {}

/// (3xN)-to-(1xN) elementwise array operation
/// such as fused multiply-add.
pub trait TernaryFn<T>: Fn(&[T], &[T], &[T], &mut [T]) -> Result<(), &'static str> {}
impl<T: Elem, K: Fn(&[T], &[T], &[T], &mut [T]) -> Result<(), &'static str>> TernaryFn<T> for K {}

/// (1xN)-to-(1x1) incremental-evaluation operation
/// such as a cumulative sum or product.
pub trait AccumulatorFn<T>: Fn(&[T], &mut T) -> Result<(), &'static str> {}
impl<T: Elem, K: Fn(&[T], &mut T) -> Result<(), &'static str>> AccumulatorFn<T> for K {}

/// Fixed-size and favorably aligned intermediate storage
/// for each expression node.
///
/// 64 is the max align req I'm aware of as of 2023-09-09 .
#[repr(align(64))]
struct Storage<T: Elem, const N: usize = 64>([T; N]);

impl<T: Elem, const N: usize> Storage<T, N> {
    #[no_panic]
    fn new(v: T) -> Self {
        Self([v; N])
    }

    const fn size(&self) -> usize {
        return N;
    }
}

/// Operator kinds, categorized by dimensionality
pub(crate) enum Op<'a, T: Elem, const N: usize = 64> {
    /// Array identity
    Array { v: &'a [T] },
    /// Array identity via iterator, useful for interop with non-contiguous arrays
    Iterator {
        v: &'a mut dyn Iterator<Item = &'a T>,
    },
    /// Scalar identity
    Scalar { acc: Option<Accumulator<'a, T, N>> },
    Unary {
        a: &'a mut Expr<'a, T, N>,
        f: &'a dyn UnaryFn<T>,
    },
    Binary {
        a: &'a mut Expr<'a, T, N>,
        b: &'a mut Expr<'a, T, N>,
        f: &'a dyn BinaryFn<T>,
    },
    Ternary {
        a: &'a mut Expr<'a, T, N>,
        b: &'a mut Expr<'a, T, N>,
        c: &'a mut Expr<'a, T, N>,
        f: &'a dyn TernaryFn<T>,
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
    #[no_panic]
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
            let outslice = &mut out[start..end];
            if outslice.len() != x.len() {
                return Err("Size mismatch");
            }
            (0..m).for_each(|i| outslice[i] = x[i])
            // outslice.copy_from_slice(x);
        }
        Ok(())
    }

    /// Evaluate the next chunk
    ///
    /// # Errors
    /// * On any error in a lower-level function during evaluation
    #[no_panic]
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
                    let m = end - start;
                    cursor = end;
                    // Copy into local storage to make sure lifetimes line up and align is controlled
                    if self.storage.0.len() < m || v.len() < m {
                        return Err("Size mismatch")
                    }
                    let storeslice = &mut self.storage.0[..m];
                    let vslice = match v.get(start..end) {
                        Some(x) => x,
                        None => return Err("Size mismatch")
                    };
                    if storeslice.len() != vslice.len() {
                        return Err("Size mismatch");
                    }
                    (0..m).for_each(|i| storeslice[i] = vslice[i]);
                    // storeslice.copy_from_slice(vslice);
                    Some((&self.storage.0[..m], m))
                }
            }
            Iterator { v } => {
                if self.cursor >= n {
                    None
                } else {
                    let end = n.min(self.cursor + nstore);
                    let start = cursor;
                    let m = end - start;
                    cursor = end;
                    if self.storage.0.len() < m {
                        return Err("Size mismatch");
                    }
                    // Copy into local storage to make sure lifetimes line up and align is controlled
                    for i in 0..m {
                        let v_inner = v.next();
                        match v_inner {
                            Some(&x) => self.storage.0[i] = x,
                            None => return Err("Input iterator ended early"),
                        }
                    }
                    Some((&self.storage.0[..m], m))
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
                        return Err("Size mismatch")
                    }
                    if self.storage.0[0] != v {
                        self.storage = Storage([v; N]);
                    }
                };

                Some((&self.storage.0[..], N))
            }
            Unary { a, f } => match a.next()? {
                Some((x, m)) => {
                    cursor += m;
                    if self.storage.0.len() < m {
                        return Err("Size mismatch")
                    }
                    f(&x[0..m], &mut self.storage.0[0..m])?;
                    Some((&self.storage.0[0..m], m))
                }
                _ => None,
            },
            Binary { a, b, f } => match (a.next()?, b.next()?) {
                (Some((x, p)), Some((y, q))) => {
                    let m = p.min(q);
                    cursor += m;
                    if self.storage.0.len() < m {
                        return Err("Size mismatch")
                    }
                    f(&x[0..m], &y[0..m], &mut self.storage.0[0..m])?;
                    Some((&self.storage.0[0..m], m))
                }
                _ => None,
            },
            Ternary { a, b, c, f } => match (a.next()?, b.next()?, c.next()?) {
                (Some((x, p)), Some((y, q)), Some((z, r))) => {
                    let m = p.min(q.min(r));
                    cursor += m;
                    if self.storage.0.len() < m {
                        return Err("Size mismatch")
                    }
                    f(&x[0..m], &y[0..m], &z[0..m], &mut self.storage.0[0..m])?;
                    Some((&self.storage.0[0..m], m))
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
    pub fn len(&self) -> usize {
        self.len
    }
}

/// A many-to-one adapter for the output of an expression.
pub struct Accumulator<'a, T: Elem, const N: usize = 64> {
    pub(crate) v: Option<T>,
    pub(crate) start: T,
    pub(crate) a: &'a mut Expr<'a, T, N>,
    pub(crate) f: &'a dyn AccumulatorFn<T>,
}

impl<'a, T: Elem, const N: usize> Accumulator<'a, T, N> {
    /// Evaluate the input expression and accumulate the output values.
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
