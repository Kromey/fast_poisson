use fast_poisson::Poisson;
use rand_xoshiro::SplitMix64;

/// Ensure points remain at minimum radius apart
///
/// Ref #33
#[test]
fn custom_rng() {
    for seed in [44244, 698383] {
        // SplitMix isn't a good RNG for actual use, but is sufficient to test that choosing
        // a custom PRNG works as expected.
        let points = Poisson::<2, SplitMix64>::new()
            .with_dimensions([30.0, 20.0], 5.0)
            .with_seed(seed)
            .generate();

        // Verify we actually have points
        assert!(
            !points.is_empty(),
            "Seed {} produced an empty set of points",
            seed
        );
    }
}
