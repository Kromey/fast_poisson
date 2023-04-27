use fast_poisson::Poisson2D;

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
        for j in (i+1)..points.len() {
            assert!(5.0 <= distance(points[i], points[j]));
        }
    }
}

fn distance(p1: [f64; 2], p2: [f64; 2]) -> f64 {
    ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2)).sqrt()
}
