# Strobe
Fast, low-memory, elementwise array expressions on the stack.
Compatible with no-std (and no-alloc) environments.

This crate provides array expressions of arbitrary depth and executes
without allocation (if an output array is provided) or with a single
allocation only for the output array.

Vectorization over segments of the data is achieved whenever the
inner function is compatible with the compiler's loop vectorizer
and matches the available (and enabled) SIMD instruction sets on
the target CPU.

## Who is it for?
If you're
* doing multi-step array operations of size >> 64 elements,
* and can formulate your array expression as a tree,
* and are bottlenecked on allocation or can't allocate at all,
* or need concretely bounded memory usage,
* or are memory _or_ CPU usage constrained,
* or don't have the standard library,
* or want to bag a speedup without hand-vectorizing,

then this may be helpful.

## Who is it _not_ for?
If, however, you're
* doing single-step or small-size array operations,
* or can't reasonably formulate your expression as a tree,
* or can get the performance you need from a library with excellent ergonomics, like `ndarray`,
* or need rigorous elimination of all panic branches,
* or need absolutely the fastest and lowest-memory method possible
  and are willing and able and have time to hand-vectorize your application,

then this may not be helpful at all.

# Example: (A - B) * (C + D) with one allocation
```rust
use strobe::{array, add, sub, mul};

// Generate some arrays to operate on
const NT: usize = 10_000;
let a = vec![1.25_f64; NT];
let b = vec![-5.32; NT];
let c = vec![1e-3; NT];
let d = vec![3.14; NT];

// Associate those arrays with inputs to the expression
let an = &mut array(&a);
let bn = &mut array(&b);
let cn = &mut array(&c);
let dn = &mut array(&d);

// Build the expression tree, then evaluate,
// allocating once for the output array purely for convenience
let y = mul(&mut sub(an, bn), &mut add(cn, dn)).eval();

// Check results for consistency
(0..NT).for_each(|i| { assert_eq!(y[i], (a[i] - b[i]) * (c[i] + d[i]) ) });
```

# Example: Evaluation with zero allocation
While we use a simple example here, any strobe expression can be
evaluated into existing storage in this way.
```rust
use strobe::{array, mul};

// Generate some arrays to operate on
const NT: usize = 10_000;
let a = vec![1.25_f64; NT];
let an0 = &mut array(&a);  // Two input nodes from `a`, for brevity
let an1 = &mut array(&a);

// Pre-allocate storage
let mut y = vec![0.0; NT];

// Build the expression tree, then evaluate into preallocated storage.
mul(an0, an1).eval_into(&mut y);

// Check results for consistency
(0..NT).for_each(|i| { assert_eq!(y[i], a[i] * a[i] ) });
```

# Example: Custom expression nodes
Many common functions are already implemented. Ones that are not
can be assembled using the `unary`, `binary`, `ternary`, and
`accumulator` functions along with a matching function pointer
or closure.
```rust
use strobe::{array, unary};

let x = [0.0_f64, 1.0, 2.0];
let mut xn = array(&x);

let sq_func = |a: &[f64], out: &mut [f64]| { (0..a.len()).for_each(|i| {out[i] = x[i].powi(2)}) };
let xsq = unary(&mut xn, &sq_func).eval();

(0..x.len()).for_each(|i| {assert_eq!(x[i] * x[i], xsq[i])});
```

# Conspicuous Design Decisions and UAQ (Un-Asked Questions)
* Why not implement the standard num ops?
    * Because we can't guarantee the lifetime of the returned node will match the lifetime of the references it owns,
      unless the outside scope is guaranteed to own both the inputs and outputs.
* Why abandon the more idiomatic functional programming regime at the interface?
    * Same reason we don't have the standard num ops - unfortunately, this usage breaks lifetime guarantees.
* Why isn't it panic-never compatible?
    * In order to guarantee the lengths of the data and eliminate panic branches in slice operations,
      we would need to use fixed-length input arrays and propagate those lengths with const generics.
      This would give unpleasant ergonomics and, because array lengths would need to be established at
      compile time, this would also prevent any useful interlanguage bindings.
* Why do expressions need to be strict trees?
    * Because we are strictly on the stack, and we need mutable references to each node in order to use intermediate storage,
      which we need in order to allow vectorization,
      and since we can only have one mutable reference to a anything, this naturally produces a tree structure.
    * _However_ since the input arrays do not need to be mutable, more than one expression node can refer to a given input array,
      which resolves many (but not all) potential cases where the strict tree structure might prove inadequate.
* I can hand-vectorize my array operations and do them even faster with even less memory!
    * Cool! Have fun with that.


# License
Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
