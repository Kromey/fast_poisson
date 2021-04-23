//! Generate a Poisson disk distribution.
//!
//! This is an implementation of Bridson's ["Fast Poisson Disk Sampling"][Bridson] algorithm in
//! arbitrary dimensions.
//!
//! # Features
//!
//!  * Iterator-based generation lets you leverage the full power of Rust's
//!    [Iterators](Iterator)
//!  * Lazy evaluation of the distribution means that even complex Iterator chains are as fast as
//!    O(N); with other libraries operations like mapping into another struct become O(N²) or more!
//!  * Using Rust's const generics allows you to consume the distribution with no additional
//!    dependencies
//!
//! This library requires Rust 1.51.0 or later, as it relies on the const generics feature
//! introduced in this version.
//!
//! # Examples
//!
//! To generate a simple Poisson disk pattern in the range (0, 1] for each of the x and y
//! dimensions:
//! ```
//! use fast_poisson::Poisson2D;
//!
//! # // Some of these doctests look a little hairy because we have to accomodate for the feature
//! # // `single_precision` which changes the type of the returned values.
//! # #[cfg(not(feature = "single_precision"))]
//! let points: Vec<[f64; 2]> = Vec::from(Poisson2D::new());
//! # #[cfg(feature = "single_precision")]
//! # let points: Vec<[f32; 2]> = Vec::from(Poisson2D::new());
//! ```
//!
//! To fill a box, specify the width and height:
//! ```
//! use fast_poisson::Poisson2D;
//!
//! let points = Poisson2D::new().with_dimensions([100.0, 100.0], 5.0);
//! ```
//!
//! You have full access to the power of Rust iterator methods to manipulate the distribution,
//! and all within O(N) time because the points are lazily generated within each iteration:
//! ```
//! use fast_poisson::Poisson2D;
//!
//! # #[cfg(not(feature = "single_precision"))]
//! struct Point {
//!     x: f64,
//!     y: f64,
//! }
//! # #[cfg(feature = "single_precision")]
//! # struct Point { x: f32, y: f32 }
//!
//! // Map the Poisson disk points to our `Point` struct in O(N) time!
//! let points = Poisson2D::new().iter().map(|[x, y]| Point { x, y });
//! ```
//!
//! The previous example can also be written concisely to leverage the `From` trait if you want to
//! collect into a `Vec<Point>`:
//! ```
//! # use fast_poisson::Poisson2D;
//! # #[cfg(not(feature = "single_precision"))]
//! # struct Point { x: f64, y: f64 }
//! # #[cfg(feature = "single_precision")]
//! # struct Point { x: f32, y: f32 }
//!
//! # #[cfg(not(feature = "single_precision"))]
//! impl From<[f64; 2]> for Point {
//!     fn from(point: [f64; 2]) -> Point {
//!         Point {
//!             x: point[0],
//!             y: point[1],
//!         }
//!     }
//! }
//! # #[cfg(feature = "single_precision")]
//! # impl From<[f32; 2]> for Point {
//! #     fn from(point: [f32; 2]) -> Point {
//! #         Point {
//! #             x: point[0],
//! #             y: point[1],
//! #         }
//! #     }
//! # }
//!
//! // Could also be written using `.into()`
//! let points: Vec<Point> = Vec::from(Poisson2D::new());
//! ```
//!
//! You can even take just a subset of the distribution without ever spending time calculating the
//! discarded points:
//! ```
//! use fast_poisson::Poisson2D;
//!
//! // Only 5 points from the distribution are actually generated!
//! let points = Poisson2D::new().iter().take(5);
//! ```
//!
//! `Poisson` implements [`IntoIterator`], so you can e.g. directly consume it with a `for` loop:
//! ```
//! use fast_poisson::Poisson2D;
//!
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
//! // To achieve desired levels of performance, you should set a larger radius for
//! // higher-order distributions
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
#[cfg(test)]
mod tests;

use rand::prelude::*;
use rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256StarStar;
use std::iter::FusedIterator;

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
#[derive(Debug, Clone)]
pub struct Poisson<const N: usize> {
    /// Dimensions of the box
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
    /// Currently the same as `Default::default()`
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Specify the space to be filled and the radius around each point
    ///
    /// By default, the output will sample each dimension from the semi-open range [0.0, 1.0). This
    /// method can be used to modify the results to fill any arbitrary space.
    ///
    /// To generate a 2-dimensional distribution in a 5×5 square, with no points closer than 1:
    /// ```
    /// # use fast_poisson::Poisson;
    /// let mut points = Poisson::<2>::new().with_dimensions([5.0, 5.0], 1.0).iter();
    ///
    /// assert!(points.all(|p| p[0] >= 0.0 && p[0] < 5.0 && p[1] >= 0.0 && p[1] < 5.0));
    /// ```
    ///
    /// To generate a 3-dimensional distribution in a 3×3×5 prism, with no points closer than 0.75:
    /// ```
    /// # use fast_poisson::Poisson;
    /// let mut points = Poisson::<3>::new().with_dimensions([3.0, 3.0, 5.0], 0.75).iter();
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
    /// non-deterministic results.
    ///
    /// ```
    /// # use fast_poisson::Poisson;
    /// let points = Poisson::<2>::new().with_seed(0xBADBEEF).iter();
    /// ```
    pub fn with_seed(&mut self, seed: u64) -> &Self {
        self.seed = Some(seed);

        self
    }

    /// Specify the number of samples to generate around each point in the distribution
    ///
    /// A higher number may result in better space filling, but will slow down generation. Note that
    /// specifying a number of samples does not ensure that the final distribution includes this
    /// number of points around each other point; rather, each sample is tested for validity before
    /// being included, so the final distribution will have *up to* the specified number of points
    /// generated from each point.
    ///
    /// By default 30 samples are selected around each point.
    ///
    /// ```
    /// # use fast_poisson::Poisson;
    /// let points = Poisson::<3>::new().with_samples(40).iter();
    /// ```
    pub fn with_samples(&mut self, samples: u32) -> &Self {
        self.num_samples = samples;

        self
    }

    /// Returns an iterator over the points in this distribution
    ///
    /// ```
    /// # use fast_poisson::Poisson;
    /// let points = Poisson::<3>::new();
    ///
    /// for point in points.iter() {
    ///     println!("{:?}", point);
    /// }
    #[must_use]
    pub fn iter(&self) -> PoissonIter<N> {
        PoissonIter::new(self.clone())
    }

    /// Generate the points in this Poisson distribution, collected into a [`Vec`](std::vec::Vec).
    ///
    /// Note that this method does *not* consume the `Poisson`, so you can call it multiple times
    /// to generate multiple `Vec`s; if you have specified a seed, however, each one will be
    /// identical, whereas they will each be unique if you have not.
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
    type IntoIter = PoissonIter<N>;

    fn into_iter(self) -> Self::IntoIter {
        PoissonIter::new(self)
    }
}

/// For convenience allow converting to a Vec directly from Poisson
impl<T, const N: usize> From<Poisson<N>> for Vec<T>
where
    T: From<[Float; N]>,
{
    fn from(poisson: Poisson<N>) -> Vec<T> {
        poisson.iter().map(|point| point.into()).collect()
    }
}

/// A Point is simply an array of Float values
type Point<const N: usize> = [Float; N];

/// A Cell is the grid coordinates containing a given point
type Cell<const N: usize> = [isize; N];

/// An iterator over the points in the Poisson disk distribution
pub struct PoissonIter<const N: usize> {
    /// The distribution from which this iterator was built
    distribution: Poisson<N>,
    /// The RNG
    rng: Xoshiro256StarStar,
    /// The size of each cell in the grid
    cell_size: Float,
    /// The grid stores spatially-oriented samples for fast checking of neighboring sample points
    grid: Vec<Option<Point<N>>>,
    /// A list of valid points that we have not yet visited
    active: Vec<Point<N>>,
    /// The current point we are visiting to generate and test surrounding points
    current_sample: Option<(Point<N>, u32)>,
}

impl<const N: usize> PoissonIter<N> {
    /// Create an iterator over the specified distribution
    fn new(distribution: Poisson<N>) -> Self {
        // We maintain a grid of our samples for faster radius checking
        let cell_size = distribution.radius / (Float::from(2.0)).sqrt();

        // If we were not given a seed, generate one non-deterministically
        let mut rng = match distribution.seed {
            None => Xoshiro256StarStar::from_entropy(),
            Some(seed) => Xoshiro256StarStar::seed_from_u64(seed),
        };

        // Calculate the amount of storage we'll need for our n-dimensional grid, which is stored
        // as a single-dimensional array.
        let grid_size: usize = distribution
            .dimensions
            .iter()
            .map(|n| (n / cell_size).ceil() as usize)
            .product();

        // We have to generate an initial point, just to ensure we've got *something* in the active list
        let mut first_point = [0.0; N];
        for (i, dim) in first_point.iter_mut().zip(distribution.dimensions.iter()) {
            *i = rng.gen::<Float>() * dim;
        }

        let mut iter = PoissonIter {
            distribution,
            rng,
            cell_size,
            grid: vec![None; grid_size],
            active: Vec::new(),
            current_sample: None,
        };
        // Don't forget to add our initial point
        iter.add_point(first_point);

        iter
    }

    /// Add a point to our pattern
    fn add_point(&mut self, point: Point<N>) {
        // Add it to the active list
        self.active.push(point);

        // Now stash this point in our grid
        let idx = self.point_to_idx(point);
        self.grid[idx] = Some(point);
    }

    /// Convert a point into grid cell coordinates
    fn point_to_cell(&self, point: Point<N>) -> Cell<N> {
        let mut cell = [0_isize; N];

        for i in 0..N {
            cell[i] = (point[i] / self.cell_size) as isize;
        }

        cell
    }

    /// Convert a cell into a grid vector index
    fn cell_to_idx(&self, cell: Cell<N>) -> usize {
        cell.iter()
            .zip(self.distribution.dimensions.iter())
            .fold(0, |acc, (pn, dn)| {
                acc * (dn / self.cell_size) as usize + *pn as usize
            })
    }

    /// Convert a point into a grid vector index
    fn point_to_idx(&self, point: Point<N>) -> usize {
        self.cell_to_idx(self.point_to_cell(point))
    }

    /// Generate a random point between `radius` and `2 * radius` away from the given point
    ///
    /// # Panics
    ///
    /// Will panic if `current_sample` is None
    fn generate_random_point(&mut self) -> Point<N> {
        let mut point = self.current_sample.unwrap().0;

        // Pick a random distance away from our point
        let dist = self.distribution.radius * (1.0 + self.rng.gen::<Float>());

        // Generate a randomly distributed vector
        let mut vector: [Float; N] = [0.0; N];
        for i in vector.iter_mut() {
            *i = self.rng.sample(StandardNormal);
        }
        // Now find this new vector's magnitude
        let mag = vector.iter().map(|&x| x.powi(2)).sum::<Float>().sqrt();

        // Dividing each of the vector's components by `mag` will produce a unit vector; then by
        // multiplying each component by `dist`, we'll have a vector pointing `dist` away from the
        // origin. If we then add each of those components to our point, we'll have effectively
        // translated our point by `dist` in a randomly chosen direction.
        // Conveniently, we can do all of this in just one step!
        let translate = dist / mag; // compute this just once!
        for i in 0..N {
            point[i] += vector[i] * translate;
        }

        point
    }

    /// Returns true if the point is within the bounds of our space.
    ///
    /// This is true if 0 ≤ point[i] < dimensions[i]
    fn in_space(&self, point: Point<N>) -> bool {
        point
            .iter()
            .zip(self.distribution.dimensions.iter())
            .all(|(p, d)| *p >= 0. && p < d)
    }

    /// Returns true if the cell is within the bounds of our grid.
    ///
    /// This is true if 0 ≤ `cell[i]` ≤ `ceiling(space[i] / cell_size)`
    fn in_grid(&self, cell: Cell<N>) -> bool {
        cell.iter()
            .zip(self.distribution.dimensions.iter())
            .all(|(c, d)| *c >= 0 && *c < (*d / self.cell_size).ceil() as isize)
    }

    /// Returns true if there is at least one other sample point within `radius` of this point
    fn in_neighborhood(&self, point: Point<N>) -> bool {
        let cell = self.point_to_cell(point);

        // We'll compare to distance squared, so we can skip the square root operation for better performance
        let r_squared = self.distribution.radius.powi(2);

        for mut carry in 0.. {
            let mut neighbor = cell;

            // We can add our current iteration count to visit each neighbor cell
            for i in (&mut neighbor).iter_mut() {
                // We clamp our addition to the range [-2, 2] for each cell
                *i += carry % 5 - 2;
                // Since we modulo by 5 to get the right range, integer division by 5 "advances" us
                carry /= 5;
            }

            if carry > 0 {
                // If we've "overflowed" then we've already tested every neighbor cell
                return false;
            }
            if !self.in_grid(neighbor) {
                // Skip anything beyond the bounds of our grid
                continue;
            }

            if let Some(point2) = self.grid[self.cell_to_idx(neighbor)] {
                let neighbor_dist_squared = point
                    .iter()
                    .zip(point2.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<Float>();

                if neighbor_dist_squared < r_squared {
                    return true;
                }
            }
        }

        // Rust can't tell the previous loop will always reach one of the `return` statements...
        false
    }
}

impl<const N: usize> Iterator for PoissonIter<N> {
    type Item = Point<N>;

    fn next(&mut self) -> Option<Point<N>> {
        if self.current_sample == None && !self.active.is_empty() {
            // Pop points off our active list until it's exhausted
            let point = {
                let i = self.rng.gen_range(0..self.active.len());
                self.active.swap_remove(i)
            };
            self.current_sample = Some((point, 0));
        }

        if let Some((point, mut i)) = self.current_sample {
            while i < self.distribution.num_samples {
                i += 1;
                self.current_sample = Some((point, i));

                // Generate up to `num_samples` random points between radius and 2*radius from the current point
                let point = self.generate_random_point();

                // Ensure we've picked a point inside the bounds of our rectangle, and more than `radius`
                // distance from any other sampled point
                if self.in_space(point) && !self.in_neighborhood(point) {
                    // We've got a good one!
                    self.add_point(point);

                    return Some(point);
                }
            }

            self.current_sample = None;

            return self.next();
        }

        None
    }
}

impl<const N: usize> FusedIterator for PoissonIter<N> {}

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
