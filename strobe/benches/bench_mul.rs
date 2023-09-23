use criterion::*;
use randn::*;
use strobe::mul;

fn bench_mul_1x(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul_1x");
    for size in [
        10, 100, 1000, 10_000, 100_000, 200_000, 300_000, 400_000, 700_000, 1_000_000, 2_000_000,
    ]
    .iter()
    {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("strobe", size), size, |b, &size| {
            let mut rng = rng_fixed_seed();
            let x = randn::<f64>(&mut rng, size);

            b.iter(|| black_box(mul(&mut x[..].into(), &mut x[..].into()).eval()));
        });

        group.bench_with_input(
            BenchmarkId::new("vec_with_alloc", size),
            size,
            |b, &size| {
                let mut rng = rng_fixed_seed();
                let x = randn::<f64>(&mut rng, size);

                b.iter(|| {
                    black_box(
                        x.iter()
                            .zip(x.iter())
                            .map(|(xx, yy)| xx * yy)
                            .collect::<Vec<_>>(),
                    )
                });
            },
        );
    }
    group.finish();
}

fn bench_mul_2x(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul_2x");
    for size in [
        10, 100, 1000, 10_000, 100_000, 200_000, 300_000, 400_000, 700_000, 1_000_000, 2_000_000,
    ]
    .iter()
    {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("strobe", size), size, |b, &size| {
            let mut rng = rng_fixed_seed();
            let x = randn::<f64>(&mut rng, size);

            b.iter(|| {
                black_box(
                    mul(
                        &mut mul(&mut x[..].into(), &mut x[..].into()),
                        &mut x[..].into(),
                    )
                    .eval(),
                )
            });
        });

        group.bench_with_input(
            BenchmarkId::new("vec_with_intermediate_storage", size),
            size,
            |b, &size| {
                let mut rng = rng_fixed_seed();
                let x = randn::<f64>(&mut rng, size);

                let mut xy = vec![0.0; size];

                b.iter(|| {
                    black_box({
                        (0..size).for_each(|i| xy[i] = x[i] * x[i]);
                        let xyz = xy
                            .iter()
                            .zip(x.iter())
                            .map(|(xx, yy)| xx * yy)
                            .collect::<Vec<_>>();
                        xyz
                    })
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("vec_with_intermediate_alloc", size),
            size,
            |b, &size| {
                let mut rng = rng_fixed_seed();
                let x = randn::<f64>(&mut rng, size);

                b.iter(|| {
                    black_box({
                        let xy = x
                            .iter()
                            .zip(x.iter())
                            .map(|(xx, yy)| xx * yy)
                            .collect::<Vec<_>>();
                        let xyz = xy
                            .iter()
                            .zip(x.iter())
                            .map(|(xx, yy)| xx * yy)
                            .collect::<Vec<_>>();
                        xyz
                    })
                });
            },
        );
    }
    group.finish();
}

fn bench_mul_3x(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul_3x");
    for size in [
        10, 100, 1000, 10_000, 100_000, 200_000, 300_000, 400_000, 700_000, 1_000_000, 2_000_000,
    ]
    .iter()
    {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("strobe", size), size, |b, &size| {
            let mut rng = rng_fixed_seed();
            let x = randn::<f64>(&mut rng, size);

            b.iter(|| {
                black_box(
                    mul(
                        &mut mul(
                            &mut mul(&mut x[..].into(), &mut x[..].into()),
                            &mut x[..].into(),
                        ),
                        &mut x[..].into(),
                    )
                    .eval(),
                )
            });
        });

        group.bench_with_input(
            BenchmarkId::new("vec_with_intermediate_storage", size),
            size,
            |b, &size| {
                let mut rng = rng_fixed_seed();
                let x = randn::<f64>(&mut rng, size);

                let mut xy = vec![0.0; size];
                let mut xyz = vec![0.0; size];

                b.iter(|| {
                    black_box({
                        (0..size).for_each(|i| xy[i] = x[i] * x[i]);
                        (0..size).for_each(|i| xyz[i] = xy[i] * x[i]);
                        let xyzw = xyz
                            .iter()
                            .zip(x.iter())
                            .map(|(xx, yy)| xx * yy)
                            .collect::<Vec<_>>();
                        xyzw
                    })
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("vec_with_intermediate_alloc", size),
            size,
            |b, &size| {
                let mut rng = rng_fixed_seed();
                let x = randn::<f64>(&mut rng, size);

                b.iter(|| {
                    black_box({
                        let xy = x
                            .iter()
                            .zip(x.iter())
                            .map(|(xx, yy)| xx * yy)
                            .collect::<Vec<_>>();
                        let xyz = xy
                            .iter()
                            .zip(x.iter())
                            .map(|(xx, yy)| xx * yy)
                            .collect::<Vec<_>>();
                        let xyzw = xyz
                            .iter()
                            .zip(x.iter())
                            .map(|(xx, yy)| xx * yy)
                            .collect::<Vec<_>>();
                        xyzw
                    })
                });
            },
        );
    }
    group.finish();
}

fn bench_mul_4x(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul_4x");
    for size in [
        10, 100, 1000, 10_000, 100_000, 200_000, 300_000, 400_000, 700_000, 1_000_000, 2_000_000,
    ]
    .iter()
    {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("strobe", size), size, |b, &size| {
            let mut rng = rng_fixed_seed();
            let x = randn::<f64>(&mut rng, size);

            b.iter(|| {
                black_box(
                    mul(
                        &mut mul(
                            &mut mul(
                                &mut mul(&mut x[..].into(), &mut x[..].into()),
                                &mut x[..].into(),
                            ),
                            &mut x[..].into(),
                        ),
                        &mut x[..].into(),
                    )
                    .eval(),
                )
            });
        });

        group.bench_with_input(
            BenchmarkId::new("vec_with_intermediate_storage", size),
            size,
            |b, &size| {
                let mut rng = rng_fixed_seed();
                let x = randn::<f64>(&mut rng, size);

                let mut xy = vec![0.0; size];
                let mut xyz = vec![0.0; size];
                let mut xyzw = vec![0.0; size];

                b.iter(|| {
                    black_box({
                        (0..size).for_each(|i| xy[i] = x[i] * x[i]);
                        (0..size).for_each(|i| xyz[i] = xy[i] * x[i]);
                        (0..size).for_each(|i| xyzw[i] = xyz[i] * x[i]);
                        let xyzwv = xyzw
                            .iter()
                            .zip(x.iter())
                            .map(|(xx, yy)| xx * yy)
                            .collect::<Vec<_>>();
                        xyzwv
                    })
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("vec_with_intermediate_alloc", size),
            size,
            |b, &size| {
                let mut rng = rng_fixed_seed();
                let x = randn::<f64>(&mut rng, size);

                b.iter(|| {
                    black_box({
                        let xy = x
                            .iter()
                            .zip(x.iter())
                            .map(|(xx, yy)| xx * yy)
                            .collect::<Vec<_>>();
                        let xyz = xy
                            .iter()
                            .zip(x.iter())
                            .map(|(xx, yy)| xx * yy)
                            .collect::<Vec<_>>();
                        let xyzw = xyz
                            .iter()
                            .zip(x.iter())
                            .map(|(xx, yy)| xx * yy)
                            .collect::<Vec<_>>();
                        let xyzwv = xyzw
                            .iter()
                            .zip(x.iter())
                            .map(|(xx, yy)| xx * yy)
                            .collect::<Vec<_>>();
                        xyzwv
                    })
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches_mul_1x, bench_mul_1x);
criterion_group!(benches_mul_2x, bench_mul_2x);
criterion_group!(benches_mul_3x, bench_mul_3x);
criterion_group!(benches_mul_4x, bench_mul_4x);
criterion_main!(
    benches_mul_1x,
    benches_mul_2x,
    benches_mul_3x,
    benches_mul_4x
);

/// Convenience functions for generating random numbers with a fixed seed
/// to use as inputs for tests and benchmarks
mod randn {
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
