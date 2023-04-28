use fast_poisson::Poisson2D;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

#[cfg(not(feature = "single_precision"))]
type Float = f64;
#[cfg(feature = "single_precision")]
type Float = f32;

/// Ensure points remain at minimum radius apart
///
/// Ref #33
#[test]
fn closeness() {
    let seed: u64 = 6980462275800279379;

    let points = Poisson2D::new()
        .with_dimensions([30.0, 20.0], 5.0)
        .with_seed(seed)
        .generate();

    // Test every point against every other point
    for i in 0..(points.len() - 1) {
        // Only need to test points later in the list, since we've already tested i against earlier points
        for j in (i + 1)..points.len() {
            assert!(5.0 <= distance(points[i], points[j]));
        }
    }
}

/// Thoroughly ensure points remain at minimum radius apart
///
/// Ref #33
#[test]
#[ignore = "This test checks 1 million seeds in parallel to ensure minimum distance guarantees are maintained"]
fn closeness_thorough() {
    (0..1_000_000).into_par_iter().for_each(|seed| {
        let points = Poisson2D::new()
            .with_dimensions([30.0, 20.0], 5.0)
            .with_seed(seed)
            .generate();

        if !points.is_empty() {
            // FIXME: Need a test to handle empty results; see #34
            // Test every point against every other point
            for i in 0..(points.len() - 1) {
                // Only need to test points later in the list, since we've already tested i against earlier points
                for j in (i + 1)..points.len() {
                    assert!(5.0 <= distance(points[i], points[j]));
                }
            }
        }
    });
}

fn distance(p1: [Float; 2], p2: [Float; 2]) -> Float {
    ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2)).sqrt()
}
