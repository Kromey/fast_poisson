//! Generate a Poisson disk distribution.
//!
//! This is an implementation of Bridson's ["Fast Poisson Disk Sampling"][Bridson] algorithm in
//! arbitrary dimensions.
//!
//! # Examples
//!
//! To generate a simple Poisson disk pattern in the range (0, 1] for each of the x and y
//! dimensions:
//! ```
//! use fast_poisson::Poisson;
//!
//! let points = Poisson::<2>::new().iter();
//! ```
//!
//! To fill a box, specify the width and height:
//! ```
//! use fast_poisson::Poisson;
//!
//! let poisson = Poisson::<2> {
//!     dimensions: [100.0, 100.0],
//!     radius: 5.0,
//!     ..Default::default()
//! };
//! let points = poisson.iter();
//! ```
//! **Caution:** If you specify a box size much larger than 1x1, you *should* specify a radius as
//! well. Otherwise the resulting distribution may have *far more* points than you are prepared to
//! handle, and may take longer to generate than expected.
//!
//! Because [`iter`](Poisson::iter) returns an iterator, you have access to the full power of Rust
//! iterator methods to further manipulate the results:
//! ```
//! use fast_poisson::Poisson;
//!
//! struct Point {
//!     x: f64,
//!     y: f64,
//! }
//!
//! // Map the Poisson disk points to our `Point` struct
//! let points = Poisson::<2>::new().iter().map(|[x, y]| Point { x, y });
//! ```
//!
//! Additionally, the iterator is lazily evaluated, meaning that points are only generated as
//! needed:
//! ```
//! use fast_poisson::Poisson;
//!
//! // Only 5 points from the distribution are actually generated!
//! let points = Poisson::<2>::new().iter().take(5);
//! ```
//!
//! You can even use [`Poisson`] directly within a `for` loop!
//! ```
//! use fast_poisson::Poisson;
//!
//! for point in Poisson::<2>::new() {
//!     println!("X: {}; Y: {}", point[0], point[1]);
//! }
//! ```
//!
//! [Bridson]: https://www.cct.lsu.edu/~fharhad/ganbatte/siggraph2007/CD2/content/sketches/0250.pdf
//! [Tulleken]: http://devmag.org.za/2009/05/03/poisson-disk-sampling/
#[cfg(test)]
mod tests;

use rand::prelude::*;
use rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256StarStar;

/// Builder for a Poisson disk distribution
#[derive(Debug, Clone)]
pub struct Poisson<const N: usize> {
    /// Dimensions of the box
    pub dimensions: [f64; N],
    /// Radius around each point that must remain empty
    pub radius: f64,
    /// Seed to use for the internal RNG
    pub seed: Option<u64>,
    /// Number of samples to generate and test around each point
    pub num_samples: u32,
}

impl<const N: usize> Poisson<N> {
    /// Create a new Poisson disk distribution
    ///
    /// Currently the same as `Default::default()`
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns an iterator over the points in this distribution
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
