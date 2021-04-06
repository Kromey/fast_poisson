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

    assert!(a
        .zip(b)
        .any(|(a, b)| a[0] - b[0] > f64::EPSILON || a[1] - b[1] > f64::EPSILON));
}

#[test]
fn iter() {
    // 2-dimensional distribution
    let poisson = Poisson2D::new();
    for _point in poisson.iter() {}

    // 3-dimensional distribution
    let poisson = Poisson3D::new();
    for _point in poisson.iter() {}

    // 4-dimensional distribution
    let mut poisson = Poisson4D::new();
    poisson.with_dimensions([1.0; 4], 0.2);
    for _point in poisson.iter() {}

    // For more than 4 dimensions, use `Poisson` directly:
    let mut poisson = Poisson::<7>::new();
    poisson.with_dimensions([1.0; 7], 0.7);
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
fn n_dimensional_grid_size() {
    for n in 1..=3 {
        let mut poisson0 = Poisson::<0>::new();
        let mut poisson1 = Poisson::<1>::new();
        let mut poisson2 = Poisson::<2>::new();
        let mut poisson3 = Poisson::<3>::new();
        let mut poisson4 = Poisson::<4>::new();

        poisson0.dimensions = [];
        poisson1.dimensions = [n as f64];
        poisson2.dimensions = [n as f64; 2];
        poisson3.dimensions = [n as f64; 3];
        poisson4.dimensions = [n as f64; 4];

        let iter0 = poisson0.iter();
        let iter1 = poisson1.iter();
        let iter2 = poisson2.iter();
        let iter3 = poisson3.iter();
        let iter4 = poisson4.iter();

        assert_eq!(iter0.grid.len(), 1);
        assert_eq!(
            iter1.grid.len(),
            (n as f64 / iter2.cell_size).ceil() as usize
        );
        assert_eq!(
            iter2.grid.len(),
            ((n as f64 / iter2.cell_size).ceil() as usize).pow(2)
        );
        assert_eq!(
            iter3.grid.len(),
            ((n as f64 / iter2.cell_size).ceil() as usize).pow(3)
        );
        assert_eq!(
            iter4.grid.len(),
            ((n as f64 / iter2.cell_size).ceil() as usize).pow(4)
        );
    }
}

#[test]
fn n_dimensional_cell_to_idx() {
    let poisson = Poisson::<3> {
        dimensions: [3., 3., 3.],
        ..Default::default()
    };
    let mut iter = poisson.iter();
    // Coerce cell_size to more easily test cell_to_idx function
    iter.cell_size = 1.;

    assert_eq!(iter.cell_to_idx([0, 0, 0]), 0);
    assert_eq!(iter.cell_to_idx([1, 1, 1]), 13);
    assert_eq!(iter.cell_to_idx([1, 2, 1]), 16);
    assert_eq!(iter.cell_to_idx([2, 1, 1]), 22);

    let poisson = Poisson::<2> {
        dimensions: [3., 3.],
        ..Default::default()
    };
    let mut iter = poisson.iter();
    // Coerce cell_size to more easily test cell_to_idx function
    iter.cell_size = 1.;

    assert_eq!(iter.cell_to_idx([0, 0]), 0);
    assert_eq!(iter.cell_to_idx([1, 1]), 4);
    assert_eq!(iter.cell_to_idx([1, 2]), 5);
    assert_eq!(iter.cell_to_idx([2, 1]), 7);
}

#[test]
fn n_dimensional_point_to_cell() {
    let poisson = Poisson::<3> {
        dimensions: [3., 3., 3.],
        ..Default::default()
    };
    let mut iter = poisson.iter();
    // Coerce cell_size to more easily test point_to_cell function
    iter.cell_size = 2.;

    assert_eq!(iter.point_to_cell([0., 0., 0.]), [0, 0, 0]);
    assert_eq!(iter.point_to_cell([1., 1., 1.]), [0, 0, 0]);
    assert_eq!(iter.point_to_cell([1., 2., 1.]), [0, 1, 0]);
    assert_eq!(iter.point_to_cell([2., 3., 1.]), [1, 1, 0]);

    let poisson = Poisson::<2> {
        dimensions: [3., 3.],
        ..Default::default()
    };
    let mut iter = poisson.iter();
    // Coerce cell_size to more easily test point_to_cell function
    iter.cell_size = 2.;

    assert_eq!(iter.point_to_cell([0., 0.]), [0, 0]);
    assert_eq!(iter.point_to_cell([1., 1.]), [0, 0]);
    assert_eq!(iter.point_to_cell([1., 2.]), [0, 1]);
    assert_eq!(iter.point_to_cell([2., 3.]), [1, 1]);
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
fn point_generation_lies_within_radius() {
    let mut iter = Poisson::<2>::new().iter();
    iter.next(); // Ensures there is a current point

    let (initial, _) = iter.current_sample.unwrap();
    for _ in 0..50 {
        let point = iter.generate_random_point();

        let r = point
            .iter()
            .zip(initial.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        assert!(r > iter.distribution.radius);
        assert!(r < iter.distribution.radius * 2.);
    }

    let mut iter = Poisson::<3>::new().iter();
    iter.next(); // Ensures there is a current point

    let (initial, _) = iter.current_sample.unwrap();
    for _ in 0..50 {
        let point = iter.generate_random_point();

        let r = point
            .iter()
            .zip(initial.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        assert!(r > iter.distribution.radius);
        assert!(r < iter.distribution.radius * 2.);
    }
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
fn in_space() {
    let iter = Poisson::<2>::new().iter();

    // Affirmative tests
    assert!(iter.in_space([0.0, 0.0]));
    assert!(iter.in_space([0.5, 0.5]));

    // Negative tests
    assert!(!iter.in_space([1.0, 1.0]));
    assert!(!iter.in_space([1.0, 2.0]));
    assert!(!iter.in_space([-0.1, 0.0]));
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
    iter.distribution.radius = 0.5;
    // Add test point near perimeter
    iter.add_point([0.9, 0.9]);

    assert!(!iter.in_neighborhood([1.1, 1.1])); // Out of bounds by definition has no neighbors
}

#[test]
fn into_vec() {
    let poisson = Poisson2D::new();

    let _vec: Vec<[f64; 2]> = Vec::from(poisson);
}
