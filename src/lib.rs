// Copyright 2021 Travis Veazey
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// https://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// https://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Generate a Poisson disk distribution.
//!
//! This is an implementation of Bridson's ["Fast Poisson Disk Sampling"][Bridson] algorithm in
//! arbitrary dimensions.
//!
//!  * Iterator-based generation lets you leverage the full power of Rust's
//!    [Iterators](Iterator)
//!  * Lazy evaluation of the distribution means that even complex Iterator chains are as fast as
//!    O(N); with other libraries operations like mapping into another struct become O(N²) or more!
//!  * Using Rust's const generics allows you to consume the distribution with no additional
//!    dependencies
//!
//! # Features
//!
//! These are the optional features you can enable in your Cargo.toml:
//!
//!  * `single_precision` changes the output, and all of the internal calculations, from using
//!    double-precision `f64` to single-precision `f32`. Distributions generated with the
//!    `single-precision` feature are *not* required nor expected to match those generated without
//!    it.
//!  * `small_rng` changes the internal PRNG used to generate the distribution: By default
//!    [`Xoshiro256StarStar`](rand_xoshiro::Xoshiro256StarStar) is used, but with this feature
//!    enabled then [`Xoshiro128StarStar`](rand_xoshiro::Xoshiro128StarStar) is used instead. This
//!    reduces the memory used for the PRNG's state from 256 bits to 128 bits, and may be more
//!    performant for 32-bit systems.
//!  * `derive_serde` automatically derives Serde's Serialize and Deserialize traits for `Poisson`,
//!    This relies on the [`serde_arrays`][sa] crate to allow (de)serializing the const generic arrays
//!    used by `Poisson`.
//!
//! # Requirements
//!
//! This library requires Rust 1.51.0 or later, as it relies on [const generics] to return
//! fixed-length points (e.g. [x, y] or [x, y, z]) without adding additional external dependencies
//! to your code.
//!
//! # Examples
//!
//! ```
//! use fast_poisson::Poisson2D;
//!
//! // Easily generate a simple `Vec`
//! # // Some of these examples look a little hairy because we have to accomodate for the feature
//! # // `single_precision` in doctests, which changes the type of the returned values.
//! # #[cfg(not(feature = "single_precision"))]
//! let points: Vec<[f64; 2]> = Poisson2D::new().generate();
//! # #[cfg(feature = "single_precision")]
//! # let points: Vec<[f32; 2]> = Poisson2D::new().generate();
//!
//! // To fill a box, specify the width and height:
//! let points = Poisson2D::new().with_dimensions([100.0, 100.0], 5.0);
//!
//! // Leverage `Iterator::map` to quickly and easily convert into a custom type in O(N) time!
//! // Also see the `Poisson::to_vec()` method
//! # #[cfg(not(feature = "single_precision"))]
//! struct Point {
//!     x: f64,
//!     y: f64,
//! }
//! # #[cfg(feature = "single_precision")]
//! # struct Point { x: f32, y: f32 }
//! let points = Poisson2D::new().iter().map(|[x, y]| Point { x, y });
//!
//! // Distributions are lazily evaluated; here only 5 points will be calculated!
//! let points = Poisson2D::new().iter().take(5);
//!
//! // `Poisson` can be directly consumed in for loops:
//! for point in Poisson2D::new() {
//!     println!("X: {}; Y: {}", point[0], point[1]);
//! }
//! ```
//!
//! Higher-order Poisson disk distributions are generated just as easily:
//! ```
//! use fast_poisson::{Poisson, Poisson3D, Poisson4D};
//!
//! // 3-dimensional distribution
//! let points_3d = Poisson3D::new().iter();
//!
//! // 4-dimensional distribution
//! let mut points_4d = Poisson4D::new();
//! // To achieve desired levels of performance, you should set a larger radius for higher-order
//! // distributions
//! points_4d.with_dimensions([1.0; 4], 0.2);
//! let points_4d = points_4d.iter();
//!
//! // For more than 4 dimensions, use `Poisson` directly:
//! let mut points_7d = Poisson::<7>::new();
//! points_7d.with_dimensions([1.0; 7], 0.6);
//! let points_7d = points_7d.iter();
//! ```
//!
//! # Upgrading
//!
//! ## 0.4.x
//!
//! This version is 100% backwards-compatible with 0.3.x and 0.2.0, however `fast_poisson` has been
//! relicensed as of this version.
//!
//! Several bugs were identified and fixed in the underlying algorithms; as a result, distributions
//! generated with 0.4.0 will *not* match those generated in earlier versions.
//!
//! ## 0.3.x
//!
//! This version adds no breaking changes and is backwards-compatible with 0.2.0.
//!
//! ## 0.2.0
//!
//! This version adds some breaking changes:
//!
//! ### 2 dimensions no longer assumed
//!
//! In version 0.1.0 you could directly instantiate `Poisson` and get a 2-dimensional distribution.
//! Now you must specifiy that you want 2 dimensions using either `Poisson<2>` or [`Poisson2D`].
//!
//! ### Returned points are arrays
//!
//! In version 0.1.0 the distribution was returned as an iterator over `(f64, f64)` tuples
//! representing each point. To leverage Rust's new const generics feature and support arbitrary
//! dimensions, the N-dimensional points are now `[f64; N]` arrays.
//!
//! ### Builder pattern
//!
//! Use the build pattern to instantiate new distributions. This will not work:
//! ```compile_fail
//! # use fast_poisson::Poisson2D;
//! let poisson = Poisson2D {
//!     width: 100.0,
//!     height: 100.0,
//!     radius: 5.0,
//!     ..Default::default()
//! };
//! let points = poisson.iter();
//! ```
//! Instead, leverage the new builder methods:
//! ```
//! # use fast_poisson::Poisson2D;
//! let mut poisson = Poisson2D::new();
//! poisson.with_dimensions([100.0; 2], 5.0);
//! let points = poisson.iter();
//! ```
//! This change frees me to make additional changes to how internal state is stored without necessarily
//! requiring additional changes to the API.
//!
//! [Bridson]: https://www.cct.lsu.edu/~fharhad/ganbatte/siggraph2007/CD2/content/sketches/0250.pdf
//! [Tulleken]: http://devmag.org.za/2009/05/03/poisson-disk-sampling/
//! [const generics]: https://blog.rust-lang.org/2021/03/25/Rust-1.51.0.html#const-generics-mvp
//! [small_rng]: https://docs.rs/rand/0.8.3/rand/rngs/struct.SmallRng.html
//! [sa]: https://crates.io/crates/serde_arrays

#[cfg(feature = "derive_serde")]
use serde::{Deserialize, Serialize};
#[cfg(test)]
mod tests;

mod iter;
pub use iter::{Iter, Point};

/// [`Poisson`] disk distribution in 2 dimensions
pub type Poisson2D = Poisson<2>;
/// [`Poisson`] disk distribution in 3 dimensions
pub type Poisson3D = Poisson<3>;
/// [`Poisson`] disk distribution in 4 dimensions
pub type Poisson4D = Poisson<4>;

#[cfg(not(feature = "single_precision"))]
type Float = f64;
#[cfg(feature = "single_precision")]
type Float = f32;

/// Poisson disk distribution in N dimensions
///
/// Distributions can be generated for any non-negative number of dimensions, although performance
/// depends upon the volume of the space: for higher-order dimensions you may need to [increase the
/// radius](Poisson::with_dimensions) to achieve the desired level of performance.
///
/// # Equality
///
/// `Poisson` implements `PartialEq` but not `Eq`, because without a specified seed the output of
/// even the same object will be different. That is, the equality of two `Poisson`s is based not on
/// whether or not they were built with the same parameters, but rather on whether or not they will
/// produce the same results once the distribution is generated.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "derive_serde", derive(Serialize, Deserialize))]
pub struct Poisson<const N: usize> {
    /// Dimensions of the box
    #[cfg_attr(feature = "derive_serde", serde(with = "serde_arrays"))]
    dimensions: [Float; N],
    /// Radius around each point that must remain empty
    radius: Float,
    /// Seed to use for the internal RNG
    seed: Option<u64>,
    /// Number of samples to generate and test around each point
    num_samples: u32,
}

impl<const N: usize> Poisson<N> {
    /// Create a new Poisson disk distribution
    ///
    /// By default, `Poisson` will sample each dimension from the semi-open range [0.0, 1.0), using
    /// a radius of 0.1 around each point, and up to 30 random samples around each; the resulting
    /// output will be non-deterministic, meaning it will be different each time.
    ///
    /// See [`Poisson::with_dimensions`] to change the range and radius, [`Poisson::with_samples`]
    /// to change the number of random samples for each point, and [`Poisson::with_seed`] to produce
    /// repeatable results.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Specify the space to be filled and the radius around each point
    ///
    /// To generate a 2-dimensional distribution in a 5×5 square, with no points closer than 1:
    /// ```
    /// # use fast_poisson::Poisson2D;
    /// let mut points = Poisson2D::new().with_dimensions([5.0, 5.0], 1.0).iter();
    ///
    /// assert!(points.all(|p| p[0] >= 0.0 && p[0] < 5.0 && p[1] >= 0.0 && p[1] < 5.0));
    /// ```
    ///
    /// To generate a 3-dimensional distribution in a 3×3×5 prism, with no points closer than 0.75:
    /// ```
    /// # use fast_poisson::Poisson3D;
    /// let mut points = Poisson3D::new().with_dimensions([3.0, 3.0, 5.0], 0.75).iter();
    ///
    /// assert!(points.all(|p| {
    ///     p[0] >= 0.0 && p[0] < 3.0
    ///     && p[1] >= 0.0 && p[1] < 3.0
    ///     && p[2] >= 0.0 && p[2] < 5.0
    /// }));
    /// ```
    pub fn with_dimensions(&mut self, dimensions: [Float; N], radius: Float) -> &mut Self {
        self.dimensions = dimensions;
        self.radius = radius;

        self
    }

    /// Specify the PRNG seed for this distribution
    ///
    /// If no seed is specified then the internal PRNG will be seeded from entropy, providing
    /// non-deterministic and non-repeatable results.
    ///
    /// ```
    /// # use fast_poisson::Poisson2D;
    /// let points = Poisson2D::new().with_seed(0xBADBEEF).iter();
    /// ```
    pub fn with_seed(&mut self, seed: u64) -> &Self {
        self.seed = Some(seed);

        self
    }

    /// Specify the maximum samples to generate around each point
    ///
    /// Note that this is not specifying the number of samples in the resulting distribution, but
    /// rather sets the maximum number of attempts to find a new, valid point around an existing
    /// point for each iteration of the algorithm.
    ///
    /// A higher number may result in better space filling, but may also slow down generation.
    ///
    /// ```
    /// # use fast_poisson::Poisson3D;
    /// let points = Poisson3D::new().with_samples(40).iter();
    /// ```
    pub fn with_samples(&mut self, samples: u32) -> &Self {
        self.num_samples = samples;

        self
    }

    /// Returns an iterator over the points in this distribution
    ///
    /// ```
    /// # use fast_poisson::Poisson3D;
    /// let points = Poisson3D::new();
    ///
    /// for point in points.iter() {
    ///     println!("{:?}", point);
    /// }
    /// ```
    #[must_use]
    pub fn iter(&self) -> Iter<N> {
        Iter::new(self.clone())
    }

    /// Generate the points in this Poisson distribution, collected into a [`Vec`](std::vec::Vec).
    ///
    /// Note that this method does *not* consume the `Poisson`, so you can call it multiple times
    /// to generate multiple `Vec`s; if you have specified a seed, each one will be identical,
    /// whereas they will each be unique if you have not (see [`Poisson::with_seed`]).
    ///
    /// ```
    /// # use fast_poisson::Poisson2D;
    /// let mut poisson = Poisson2D::new();
    ///
    /// let points1 = poisson.generate();
    /// let points2 = poisson.generate();
    ///
    /// // These are not identical because no seed was specified
    /// assert!(points1.iter().zip(points2.iter()).any(|(a, b)| a != b));
    ///
    /// poisson.with_seed(1337);
    ///
    /// let points3 = poisson.generate();
    /// let points4 = poisson.generate();
    ///
    /// // These are identical because a seed was specified
    /// assert!(points3.iter().zip(points4.iter()).all(|(a, b)| a == b));
    /// ```
    pub fn generate(&self) -> Vec<Point<N>> {
        self.iter().collect()
    }

    /// Generate the points in the Poisson distribution, as a [`Vec<T>`](std::vec::Vec).
    ///
    /// This is a shortcut to translating the arrays normally generated into arbitrary types,
    /// with the precondition that the type `T` must implement the `From` trait. This is otherwise
    /// identical to the [`generate`][Poisson::generate] method.
    ///
    /// ```
    /// # use fast_poisson::Poisson2D;
    /// # #[cfg(not(feature = "single_precision"))]
    /// struct Point {
    ///     x: f64,
    ///     y: f64,
    /// }
    /// # #[cfg(feature = "single_precision")]
    /// # struct Point { x: f32, y: f32 }
    ///
    /// # #[cfg(not(feature = "single_precision"))]
    /// impl From<[f64; 2]> for Point {
    ///     fn from(point: [f64; 2]) -> Point {
    ///         Point {
    ///             x: point[0],
    ///             y: point[1],
    ///         }
    ///     }
    /// }
    /// # #[cfg(feature = "single_precision")]
    /// # impl From<[f32; 2]> for Point {
    /// #     fn from(point: [f32; 2]) -> Point {
    /// #         Point {
    /// #             x: point[0],
    /// #             y: point[1],
    /// #         }
    /// #     }
    /// # }
    ///
    /// let points: Vec<Point> = Poisson2D::new().to_vec();
    /// ```
    pub fn to_vec<T>(&self) -> Vec<T>
    where
        T: From<[Float; N]>,
    {
        self.iter().map(|point| point.into()).collect()
    }
}

/// No object is equal, not even to itself, if the seed is unspecified
impl<const N: usize> PartialEq for Poisson<N> {
    fn eq(&self, other: &Self) -> bool {
        self.seed.is_some()
            && other.seed.is_some()
            && self.dimensions == other.dimensions
            && self.radius == other.radius
            && self.seed == other.seed
            && self.num_samples == other.num_samples
    }
}

impl<const N: usize> Default for Poisson<N> {
    fn default() -> Self {
        Poisson::<N> {
            dimensions: [1.0; N],
            radius: 0.1,
            seed: None,
            num_samples: 30,
        }
    }
}

impl<const N: usize> IntoIterator for Poisson<N> {
    type Item = Point<N>;
    type IntoIter = Iter<N>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self)
    }
}

impl<const N: usize> IntoIterator for &Poisson<N> {
    type Item = Point<N>;
    type IntoIter = Iter<N>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// For convenience allow converting to a Vec directly from Poisson
impl<T, const N: usize> From<Poisson<N>> for Vec<T>
where
    T: From<[Float; N]>,
{
    fn from(poisson: Poisson<N>) -> Vec<T> {
        poisson.to_vec()
    }
}

// Hacky way to include README in doc-tests, but works until #[doc(include...)] is stabilized
// https://github.com/rust-lang/cargo/issues/383#issuecomment-720873790
#[cfg(doctest)]
mod test_readme {
    macro_rules! external_doc_test {
        ($x:expr) => {
            #[doc = $x]
            extern "C" {}
        };
    }

    external_doc_test!(include_str!("../README.md"));
}
