use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fast_poisson::{Poisson2D, Poisson3D};

pub fn criterion_benchmark(c: &mut Criterion) {
    let seed = 0xBADBEEF;
    c.bench_function("Poisson2D", |b| {
        b.iter(|| Poisson2D::new().with_seed(black_box(seed)).generate())
    });
    c.bench_function("Poisson3D", |b| {
        b.iter(|| Poisson3D::new().with_seed(black_box(seed)).generate())
    });

    c.bench_function("Poisson2D with custom dimensions", |b| {
        b.iter(|| {
            Poisson2D::new()
                .with_dimensions([30.0, 20.0], 5.0)
                .with_seed(black_box(seed))
                .generate()
        })
    });
    c.bench_function("Poisson3D with custom dimensions", |b| {
        b.iter(|| {
            Poisson3D::new()
                .with_dimensions([30.0, 20.0, 15.0], 5.0)
                .with_seed(black_box(seed))
                .generate()
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
