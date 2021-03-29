//! Generate a Poisson disk distribution.
//!
//! This is an implementation of Bridson's ["Fast Poisson Disk Sampling"][Bridson] algorithm in
//! arbitrary dimensions.
//!
//! # Features
//!
//!  * Iterator-based generation lets you leverage the full power of Rust's
//!    [Iterators](Iterator)
//!  * Lazy evaluation of the distribution means that even complex Iterator chains are O(N);
//!    with other libraries operations like mapping into another struct become O(N²) or more!
//!  * Using Rust's const generics allows you to consume the distribution with no additional
//!    dependencies
//!
//! # Examples
//!
//! To generate a simple Poisson disk pattern in the range (0, 1] for each of the x and y
//! dimensions:
//! ```
//! use fast_poisson::Poisson2D;
//!
//! let points = Poisson2D::new().iter();
//! ```
//!
//! To fill a box, specify the width and height:
//! ```
//! use fast_poisson::Poisson2D;
//!
//! let points = Poisson2D::new().with_dimensions([100.0, 100.0], 5.0).iter();
//! ```
//!
//! Because [`iter`](Poisson::iter) returns an iterator, you have access to the full power of Rust
//! iterator methods to further manipulate the results, and all within O(N) time because the
//! distribution is lazily generated within each iteration:
//! ```
//! use fast_poisson::Poisson2D;
//!
//! struct Point {
//!     x: f64,
//!     y: f64,
//! }
//!
//! // Map the Poisson disk points to our `Point` struct in O(N) time!
//! let points = Poisson2D::new().iter().map(|[x, y]| Point { x, y });
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
//! [Bridson]: https://www.cct.lsu.edu/~fharhad/ganbatte/siggraph2007/CD2/content/sketches/0250.pdf
//! [Tulleken]: http://devmag.org.za/2009/05/03/poisson-disk-sampling/
#[cfg(test)]
mod tests;

use rand::prelude::*;
use rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256StarStar;

/// [`Poisson`] disk distribution in 2 dimensions
pub type Poisson2D = Poisson<2>;
/// [`Poisson`] disk distribution in 3 dimensions
pub type Poisson3D = Poisson<3>;
/// [`Poisson`] disk distribution in 4 dimensions
pub type Poisson4D = Poisson<4>;

/// Poisson disk distribution in N dimensions
/// 
/// Distributions can be generated for any non-negative number of dimensions, although performance
/// depends upon the volume of the space: for higher-order dimensions you may need to [increase the
/// radius](Poisson::with_dimensions) to achieve the desired level of performance.
#[derive(Debug, Clone)]
pub struct Poisson<const N: usize> {
    /// Dimensions of the box
    dimensions: [f64; N],
    /// Radius around each point that must remain empty
    radius: f64,
    /// Seed to use for the internal RNG
    seed: Option<u64>,
    /// Number of samples to generate and test around each point
    num_samples: u32,
}

impl<const N: usize> Poisson<N> {
    /// Create a new Poisson disk distribution
    ///
    /// Currently the same as `Default::default()`
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
    pub fn with_dimensions(&mut self, dimensions: [f64; N], radius: f64) -> &mut Self {
        self.dimensions = dimensions;
        self.radius = radius;

        self
    }

    /// Specify the RNG seed for this distribution
    ///
    /// If no seed is specified then the internal RNG will be seeded from [`rand::thread_rng()`],
    /// providing non-deterministic results.
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
    pub fn iter(&self) -> PoissonIter<N> {
        PoissonIter::new(self.clone())
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

/// A Point is simply an array of f64 values
type Point<const N: usize> = [f64; N];

/// A Cell is the grid coordinates containing a given point
type Cell<const N: usize> = [isize; N];

/// An iterator over the points in the Poisson disk distribution
pub struct PoissonIter<const N: usize> {
    /// The distribution from which this iterator was built
    distribution: Poisson<N>,
    /// The RNG
    rng: Xoshiro256StarStar,
    /// The size of each cell in the grid
    cell_size: f64,
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
        let cell_size = distribution.radius / (2_f64).sqrt();

        // If we were not given a seed, generate one non-deterministically
        let mut rng = match distribution.seed {
            None => Xoshiro256StarStar::from_rng(rand::thread_rng()).unwrap(),
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
            *i = rng.gen::<f64>() * dim;
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
        let mut cell = [0isize; N];

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
        let dist = self.distribution.radius * (1.0 + self.rng.gen::<f64>());

        // Generate a randomly distributed vector
        let mut vector = [0_f64; N];
        for i in vector.iter_mut() {
            *i = self.rng.sample(StandardNormal);
        }
        // Now find this new vector's magnitude
        let mag = vector.iter().map(|&x| x.powi(2)).sum::<f64>().sqrt();

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
    /// This is true if 0 ≤ cell[i] ≤ ceiling(space[i] / cell_size)
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
                    .sum::<f64>();

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
