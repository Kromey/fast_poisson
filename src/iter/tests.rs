// Copyright 2021 Travis Veazey
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// https://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// https://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use rand_distr::num_traits::ToPrimitive;
use super::*;
use crate::{Poisson2D, Poisson3D};

#[test]
fn adding_points() {
    let mut iter = Poisson::<2>::new().iter();
    let point = [0.5, 0.5];

    iter.add_point(point);

    assert!(iter.active.contains(&point));

    let nearest = iter.sampled.nearest_one::<SquaredEuclidean>(&point);
    assert_eq!(
        (nearest.distance.to_f32().unwrap(), nearest.item),
        (0.0f32, 0u64)
    );
}

#[test]
fn initial_point_not_excluded() {
    for seed in 0..50 {
        let mut iter = Poisson2D::new().with_seed(seed).iter();
        let first_point = iter.active[0];
        let radius = iter.distribution.radius.powi(2); // Square for performance
        if iter.any(|p| (p[0] - first_point[0]).powi(2) + (p[1] - first_point[1]).powi(2) < radius)
        {
            return;
        }
    }

    panic!("Initial point is only found within a void in this distribution");
}

#[test]
fn point_generation_lies_within_radius() {
    let mut iter = Poisson2D::new().iter();
    let initial = [0.5; 2];

    for _ in 0..50 {
        let point = iter.generate_random_point(initial);

        let r = point
            .iter()
            .zip(initial.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<Float>()
            .sqrt();

        assert!(r > iter.distribution.radius);
        assert!(r < iter.distribution.radius * 2.);
    }

    let mut iter = Poisson3D::new().iter();
    let initial = [0.5; 3];

    for _ in 0..50 {
        let point = iter.generate_random_point(initial);

        let r = point
            .iter()
            .zip(initial.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<Float>()
            .sqrt();

        assert!(r > iter.distribution.radius);
        assert!(r < iter.distribution.radius * 2.);
    }
}

#[test]
fn generate_random_point() {
    let mut iter = Poisson::<2>::new().iter();

    for _ in 0..100 {
        let [x, y] = iter.generate_random_point([0.5, 0.5]);

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
fn distant_point_has_no_neighbors() {
    let mut iter = Poisson::<2>::new().iter();
    // Flush the k-d tree
    iter.sampled = KdTree::new();

    // Add test point
    iter.add_point([0.9, 0.9]);

    assert!(!iter.in_neighborhood([0.1, 0.1]));
    assert!(!iter.in_neighborhood([0.2, 0.2]));
    assert!(!iter.in_neighborhood([0.8, 0.8]));
}

#[test]
fn point_has_neighbors() {
    let mut iter = Poisson::<2>::new().iter();
    // Flush the k-d tree
    iter.sampled = KdTree::new();

    // Add test point
    iter.add_point([0.2, 0.2]);

    assert!(iter.in_neighborhood([0.2, 0.2])); // Same point is a neighbor
    assert!(iter.in_neighborhood([0.2005, 0.2])); // Close point is a neighbor
}
