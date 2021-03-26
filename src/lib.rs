//! Generate a Poisson disk distribution.
//!
//! This is an implementation of Bridson's ["Fast Poisson Disk Sampling"][Bridson] algorithm. At
//! present, however, this library only implements it for 2 dimensions.
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
use rand_xoshiro::Xoshiro256StarStar;
use rand_distr::StandardNormal;

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
    pub fn new() -> Self {
        Self::default()
    }

    pub fn iter(&self) -> PoissonIter<N> {
        PoissonIter::new(self)
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
        self.iter()
    }
}

/// A Point is simply an array of f64 values
type Point<const N: usize> = [f64; N];

/// A Cell is the grid coordinates containing a given point
type Cell<const N: usize> = [isize; N];

/// An iterator over the points in the Poisson disk distribution
pub struct PoissonIter<const N: usize> {
    /// The Pattern from which this iterator was built
    pattern: Poisson<N>,
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
    pub fn new(pattern: &Poisson<N>) -> Self {
        // We maintain a grid of our samples for faster radius checking
        let cell_size = pattern.radius / (2_f64).sqrt();

        // If we were not given a seed, generate one non-deterministically
        let rng = match pattern.seed {
            None => Xoshiro256StarStar::from_rng(rand::thread_rng()).unwrap(),
            Some(seed) => Xoshiro256StarStar::seed_from_u64(seed),
        };

        let grid_size: usize = pattern.dimensions
            .iter()
            .map(|n| (n / cell_size).ceil() as usize)
            .product();

        let mut iter = PoissonIter {
            pattern: pattern.clone(),
            rng,
            cell_size,
            grid: vec![None; grid_size],
            active: Vec::new(),
            current_sample: None,
        };
    
        // We have to generate an initial point, just to ensure we've got *something* in the active list
        let mut first_point = [0.0; N];
        for i in 0..N {
            first_point[i] = iter.rng.gen::<f64>() * pattern.dimensions[i];
        }
        iter.add_point(first_point);

        iter
    }

    /// Add a point to our pattern
    fn add_point(&mut self, point: Point<N>) {
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
        //(cell.0 as f64 * self.pattern.dimensions[1] / self.cell_size) as usize + cell.1 as usize

        cell.iter()
            .zip(self.pattern.dimensions.iter())
            .fold(0, |acc, (pn, dn)| acc * (dn / self.cell_size) as usize + *pn as usize)
    }

    /// Convenience function to go straight from point to grid vector index
    fn point_to_idx(&self, point: Point<N>) -> usize {
        self.cell_to_idx(self.point_to_cell(point))
    }

    /// Generate a random point between `radius` and `2 * radius` away from the given point
    fn generate_random_point(&mut self) -> Point<N> {
        let mut point = self.current_sample.unwrap().0;

        let dist = self.pattern.radius * (1.0 + self.rng.gen::<f64>());

        // Generate a randomly distributed vector
        let mut vector = [0_f64; N];
        for i in 0..N {
            vector[i] = self.rng.sample(StandardNormal);
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
    
    /// Return true if the point is within the bounds of our space.
    ///
    /// This is true if 0 ≤ x < width and 0 ≤ y < height
    fn in_rectangle(&self, point: Point<N>) -> bool {
        point[0] >= 0. && point[0] < self.pattern.dimensions[1] && point[1] >= 0. && point[1] < self.pattern.dimensions[0]
    }
    
    /// Returns true if there is at least one other sample point within `radius` of this point
    fn in_neighborhood(&self, point: Point<N>) -> bool {
        let cell = self.point_to_cell(point);
        let grid_width = (self.pattern.dimensions[1] / self.cell_size) as isize;
        let grid_height = (self.pattern.dimensions[0] / self.cell_size) as isize;

        // We'll compare to distance squared, so we can skip the square root operation for better performance
        let r_squared = self.pattern.radius.powi(2);
    
        for x in cell[0] - 2..=cell[0] + 2 {
            // Make sure we're still in our grid
            if x < 0 || x >= grid_width {
                continue;
            }
            for y in cell[1] - 2..=cell[1] + 2 {
                // Make sure we're still in our grid
                if y < 0 || y >= grid_height {
                    continue;
                }
    
                // If there's a sample here, check that it's not too close to us
                let mut neighbor_cell = [0; N];
                neighbor_cell[0] = x;
                neighbor_cell[1] = y;

                let idx = self.cell_to_idx(neighbor_cell);
                if let Some(point2) = self.grid[idx] {
                    if (point[0] - point2[0]).powi(2) + (point[1] - point2[1]).powi(2) < r_squared {
                        return true;
                    }
                }
            }
        }
    
        // We only make it to here if we find no samples too close
        false
    }
}

impl<const N: usize> Iterator for PoissonIter<N> {
    type Item = Point<N>;

    fn next(&mut self) -> Option<Point<N>> {
        if self.current_sample == None {
            if !self.active.is_empty() {
                // Pop points off our active list until it's exhausted
                let point = {
                    let i = self.rng.gen_range(0..self.active.len());
                    self.active.swap_remove(i)
                };
                self.current_sample = Some((point, 0));
            }
        }

        if let Some((point, mut i)) = self.current_sample {
            while i < self.pattern.num_samples {
                i += 1;
                self.current_sample = Some((point, i));

                // Generate up to `num_samples` random points between radius and 2*radius from the current point
                let point = self.generate_random_point();
    
                // Ensure we've picked a point inside the bounds of our rectangle, and more than `radius`
                // distance from any other sampled point
                if self.in_rectangle(point)
                    && !self.in_neighborhood(point)
                {
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
        extern {}
    };
  }

  external_doc_test!(include_str!("../README.md"));
}
