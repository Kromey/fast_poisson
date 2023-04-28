use fast_poisson::Poisson2D;
use rayon::prelude::*;

#[cfg(not(feature = "single_precision"))]
type Float = f64;
#[cfg(feature = "single_precision")]
type Float = f32;

/// Ensure points remain at minimum radius apart
///
/// Ref #33
#[test]
#[ignore = "This is an intensive test, taking 30 minutes to several hours to run"]
fn million_seeds() {
    (0_u64..1_000_000).into_par_iter().for_each(|s| {
        // Issues may arise at higher-order seeds, so get there quicker
        // Using a large Mersenne Prime so we avoid accidentally re-testing previous seeds
        let seed = s.wrapping_mul(9007199254740991);

        let points_defaults = Poisson2D::new().with_seed(seed).generate();

        let points = Poisson2D::new()
            .with_dimensions([30.0, 20.0], 5.0)
            .with_seed(seed)
            .generate();
        let points2 = Poisson2D::new()
            .with_dimensions([30.0, 20.0], 5.0)
            .with_seed(seed)
            .generate();

        // Ensure we don't have an empty result
        assert!(
            !points_defaults.is_empty(),
            "Seed {} generated 0 points with default parameters",
            seed
        );
        assert!(!points.is_empty(), "Seed {} generated 0 points", seed);

        // Ensure same seed generates same output
        assert_eq!(
            points, points2,
            "Seed {} did not produce deterministic results",
            seed
        );

        // Test every point against every other point
        for i in 0..(points_defaults.len() - 1) {
            // Only need to test points later in the list, since we've already tested i against earlier points
            for j in (i + 1)..points_defaults.len() {
                assert!(
                    0.1 <= distance(points_defaults[i], points_defaults[j]),
                    "Seed {} did not respect radius with default parameters",
                    seed
                );
            }
        }

        for i in 0..(points.len() - 1) {
            // Only need to test points later in the list, since we've already tested i against earlier points
            for j in (i + 1)..points.len() {
                assert!(
                    5.0 <= distance(points[i], points[j]),
                    "Seed {} did not respect radius",
                    seed
                );
            }
        }
    });
}

fn distance(p1: [Float; 2], p2: [Float; 2]) -> Float {
    ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2)).sqrt()
}
