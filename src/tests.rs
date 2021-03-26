use super::*;

#[test]
fn new_is_default() {
    let new = Poisson::<2>::new();
    let default: Poisson<2> = Default::default();

    assert_eq!(new.dimensions, default.dimensions);
    assert_eq!(new.radius, default.radius);
    assert_eq!(new.seed, default.seed);
    assert_eq!(new.num_samples, default.num_samples);
}

#[test]
fn unseeded_is_non_deterministic() {
    let a = Poisson::<2>::new().iter();
    let b = Poisson::<2>::new().iter();

    assert!(a.zip(b).any(|(a, b)| a[0] - b[0] > f64::EPSILON || a[1] - b[1] > f64::EPSILON));
}

#[test]
fn iter() {
    let poisson = Poisson::<2>::new();

    for _point in poisson.iter() {}
}

#[test]
fn iter_does_not_consume() {
    let poisson = Poisson::<2>::new();

    for _point in poisson.iter() {}

    for _point in poisson.iter() {}
}

#[test]
fn into_iter() {
    let poisson = Poisson::<2>::new();

    for _point in poisson {}
}

#[test]
fn sample_to_grid() {
    let iter = Poisson::<2>::new().iter();

    for &point in &[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]] {
        let idx = iter.point_to_idx(point);
    
        // Trying to access this will panic if it's out of bound in any way
        // TODO: Should do more robust testing of the results
        let _ = iter.grid[idx];
    }
}

#[test]
fn adding_points() {
    let mut iter = Poisson::<2>::new().iter();
    let point = [0.5, 0.5];

    iter.add_point(point);

    assert!(iter.active.contains(&point));

    let idx = iter.point_to_idx(point);
    assert_eq!(iter.grid[idx], Some(point));
}

#[test]
#[should_panic]
fn point_generation_panics_with_no_point() {
    let mut iter = Poisson::<2>::new().iter();

    iter.generate_random_point();
}

#[test]
fn generate_random_point() {
    let mut iter = Poisson::<2>::new().iter();
    iter.current_sample = Some(([0.5, 0.5], 0));

    for _ in 0..100 {
        let [x, y] = iter.generate_random_point();

        assert!(0.0 <= x);
        assert!(1.0 > x);
        assert!(0.0 <= y);
        assert!(1.0 > y);
    }
}

#[test]
fn in_rectangle() {
    let iter = Poisson::<2>::new().iter();

    // Affirmative tests
    assert!(iter.in_rectangle([0.0, 0.0]));
    assert!(iter.in_rectangle([0.5, 0.5]));

    // Negative tests
    assert!(!iter.in_rectangle([1.0, 1.0]));
    assert!(!iter.in_rectangle([1.0, 2.0]));
    assert!(!iter.in_rectangle([-0.1, 0.0]));
}

#[test]
fn empty_grid_has_no_neighbors() {
    let mut iter = Poisson::<2>::new().iter();
    // Flush the grid
    iter.grid = vec![None; iter.grid.len()];

    assert!(!iter.in_neighborhood([0.1, 0.1]));
    assert!(!iter.in_neighborhood([0.2, 0.2]));
    assert!(!iter.in_neighborhood([1.1, 1.1])); // Out of bounds by definition has no neighbors
}

#[test]
fn distant_point_has_no_neighbors() {
    let mut iter = Poisson::<2>::new().iter();
    // Flush the grid
    iter.grid = vec![None; iter.grid.len()];

    // Add test point
    iter.add_point([0.9, 0.9]);

    assert!(!iter.in_neighborhood([0.1, 0.1]));
    assert!(!iter.in_neighborhood([0.2, 0.2]));
    assert!(!iter.in_neighborhood([0.8, 0.8]));
}

#[test]
fn point_has_neighbors() {
    let mut iter = Poisson::<2>::new().iter();
    // Flush the grid
    iter.grid = vec![None; iter.grid.len()];

    // Add test point
    iter.add_point([0.2, 0.2]);

    assert!(iter.in_neighborhood([0.2, 0.2])); // Same point is a neighbor
    assert!(iter.in_neighborhood([0.2005, 0.2])); // Close point is a neighbor
}

#[test]
fn out_of_bounds_point_is_not_neighbor() {
    let mut iter = Poisson::<2>::new().iter();
    // Flush the grid
    iter.grid = vec![None; iter.grid.len()];

    // Enlarge radius
    iter.pattern.radius = 0.5;
    // Add test point near perimeter
    iter.add_point([0.9, 0.9]);

    assert!(!iter.in_neighborhood([1.1, 1.1])); // Out of bounds by definition has no neighbors
}
