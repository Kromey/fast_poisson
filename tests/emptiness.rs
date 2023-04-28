use fast_poisson::Poisson2D;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

/// Ensure points remain at minimum radius apart
///
/// Ref #33
#[test]
fn emptiness() {
    for seed in [44244, 698383] {
        let points = Poisson2D::new()
            .with_dimensions([30.0, 20.0], 5.0)
            .with_seed(seed)
            .generate();
    
        // Verify we actually have points
        assert!(!points.is_empty(), "Seed {} produced an empty set of points", seed);
    }
}

/// Thoroughly ensure points remain at minimum radius apart
///
/// Ref #33
#[test]
#[ignore = "This test checks 1 million seeds in parallel to ensure generated points are not empty"]
fn emptiness_thorough() {
    (0..1_000_000).into_par_iter().for_each(|seed| {
        let points = Poisson2D::new()
            .with_dimensions([30.0, 20.0], 5.0)
            .with_seed(seed)
            .generate();
    
            // Verify we actually have points
            assert!(!points.is_empty(), "Seed {} produced an empty set of points", seed);
    });
}
