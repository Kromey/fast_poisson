// Copyright 2021 Travis Veazey
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// https://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// https://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use super::*;

#[test]
fn new_is_default() {
    let new = Poisson2D::new();
    let default = Poisson2D::default();

    assert_eq!(new.dimensions, default.dimensions);
    assert_eq!(new.radius, default.radius);
    assert_eq!(new.seed, default.seed);
    assert_eq!(new.num_samples, default.num_samples);
}

#[test]
fn builder_pattern() {
    let _points = Poisson2D::new()
        .with_dimensions([10.0, 10.0], 2.0)
        .with_seed(0xBADBEEF)
        .with_samples(30)
        .generate();
}

#[test]
fn unseeded_is_non_deterministic() {
    let a = Poisson2D::new().iter();
    let b = Poisson2D::new().iter();

    assert!(a
        .zip(b)
        .any(|(a, b)| a[0] - b[0] > Float::EPSILON || a[1] - b[1] > Float::EPSILON));
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

    for _point in &poisson {}

    for _point in poisson.iter() {}

    for _point in &poisson {}
}

#[test]
fn into_iter() {
    let poisson = Poisson::<2>::new();

    for _point in poisson {}
}

#[test]
fn to_vec() {
    let poisson = Poisson2D::new();

    let _vec: Vec<[Float; 2]> = poisson.to_vec();
}

#[test]
fn poisson_equality() {
    let mut poisson = Poisson2D::new();

    // No seed has been specified, so it's not equal to itself
    assert_ne!(poisson, poisson);

    let mut poisson2 = Poisson2D::new();

    // No seed has been specified, so these are not equal
    assert_ne!(poisson, poisson2);

    poisson.with_seed(1337);
    poisson2.with_seed(1337);

    // Now with same seed, these are equal
    assert_eq!(poisson, poisson);
    assert_eq!(poisson2, poisson2);
    assert_eq!(poisson, poisson2);

    poisson2.with_dimensions([2.0, 3.0], 0.5);

    // Different dimension, unequal again
    assert_ne!(poisson, poisson2);
}
