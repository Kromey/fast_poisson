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
//! let points = Poisson::new().iter();
//! ```
//! 
//! To fill a box, specify the width and height:
//! ```
//! use fast_poisson::Poisson;
//! 
//! let poisson = Poisson {
//!     width: 100.0,
//!     height: 100.0,
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
//! // Map the Poisson disk tuples to our `Point` struct
//! let points = Poisson::new().iter().map(|(x, y)| Point { x, y });
//! ```
//! 
//! Additionally, the iterator is lazily evaluated, meaning that points are only generated as
//! needed:
//! ```
//! use fast_poisson::Poisson;
//! 
//! // Only 5 points from the distribution are actually generated!
//! let points = Poisson::new().iter().take(5);
//! ```
//! 
//! You can even use [`Poisson`] directly within a `for` loop!
//! ```
//! use fast_poisson::Poisson;
//! 
//! for point in Poisson::new() {
//!     println!("X: {}; Y: {}", point.0, point.1);
//! }
//! ```
//!
//! [Bridson]: https://www.cct.lsu.edu/~fharhad/ganbatte/siggraph2007/CD2/content/sketches/0250.pdf
//! [Tulleken]: http://devmag.org.za/2009/05/03/poisson-disk-sampling/
#[cfg(test)]
mod tests;

use rand::prelude::*;
use rand_xoshiro::Xoshiro256StarStar;

/// Builder for a Poisson disk distribution
#[derive(Debug, Clone)]
pub struct Poisson {
    /// Wdith of the box
    pub width: f64,
    /// Height of the box
    pub height: f64,
    /// Radius around each point that must remain empty
    pub radius: f64,
    /// Seed to use for the internal RNG
    pub seed: u64,
    /// Number of samples to generate and test around each point
    pub num_samples: u32,
}

impl Poisson {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn iter(&self) -> PoissonIter {
        PoissonIter::new(self)
    }
}

impl Default for Poisson {
    fn default() -> Self {
        Poisson {
            width: 1.0,
            height: 1.0,
            radius: 0.1,
            seed: 0,
            num_samples: 30,
        }
    }
}

impl IntoIterator for Poisson {
    type Item = Point;
    type IntoIter = PoissonIter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// A Point is simply a two-tuple of f64 values
type Point = (f64, f64);

/// An iterator over the points in the Poisson disk distribution
pub struct PoissonIter {
    /// The Pattern from which this iterator was built
    pattern: Poisson,
    /// The RNG
    rng: Xoshiro256StarStar,
    /// The size of each cell in the grid
    cell_size: f64,
    /// The grid stores spatially-oriented samples for fast checking of neighboring sample points
    grid: Vec<Vec<Option<Point>>>,
    /// A list of valid points that we have not yet visited
    active: Vec<Point>,
    /// The current point we are visiting to generate and test surrounding points
    current_sample: Option<(Point, u32)>,
}

impl PoissonIter {
    pub fn new(pattern: &Poisson) -> Self {
        // We maintain a grid of our samples for faster radius checking
        let cell_size = pattern.radius / (2_f64).sqrt();

        let mut iter = PoissonIter {
            pattern: pattern.clone(),
            rng: Xoshiro256StarStar::seed_from_u64(pattern.seed),
            cell_size,
            grid: vec![vec![None; (pattern.height / cell_size).ceil() as usize]; (pattern.width / cell_size).ceil() as usize],
            active: Vec::new(),
            current_sample: None,
        };
    
        // We have to generate an initial point, just to ensure we've got *something* in the active list
        let first_point = (iter.rng.gen::<f64>() * pattern.width, iter.rng.gen::<f64>() * pattern.height);
        iter.add_point(first_point);

        iter
    }

    /// Add a point to our pattern
    fn add_point(&mut self, point: Point) {
        self.active.push(point);

        // Now stash this point in our grid
        let (x, y) = self.sample_to_grid(point);
        self.grid[x][y] = Some(point);
    }

    /// Convert a sample point into grid cell coordinates
    fn sample_to_grid(&self, point: Point) -> (usize, usize) {
        (
            (point.0 / self.cell_size) as usize,
            (point.1 / self.cell_size) as usize,
        )
    }

    /// Generate a random point between `radius` and `2 * radius` away from the given point
    fn generate_random_point(&mut self) -> Point {
        let point = self.current_sample.unwrap().0;

        let radius = self.pattern.radius * (1.0 + self.rng.gen::<f64>());
        let angle = 2. * std::f64::consts::PI * self.rng.gen::<f64>();
    
        (
            point.0 + radius * angle.cos(),
            point.1 + radius * angle.sin(),
        )
    }
    
    /// Return true if the point is within the bounds of our space.
    ///
    /// This is true if 0 ≤ x < width and 0 ≤ y < height
    fn in_rectangle(&self, point: Point) -> bool {
        point.0 >= 0. && point.0 < self.pattern.width && point.1 >= 0. && point.1 < self.pattern.height
    }
    
    /// Returns true if there is at least one other sample point within `radius` of this point
    fn in_neighborhood(&self, point: Point) -> bool {
        let grid_point = {
            let p = self.sample_to_grid(point);
            (p.0 as isize, p.1 as isize)
        };
        // We'll compare to distance squared, so we can skip the square root operation for better performance
        let r_squared = self.pattern.radius.powi(2);
    
        for x in grid_point.0 - 2..=grid_point.0 + 2 {
            // Make sure we're still in our grid
            if x < 0 || x >= self.grid.len() as isize {
                continue;
            }
            for y in grid_point.1 - 2..=grid_point.1 + 2 {
                // Make sure we're still in our grid
                if y < 0 || y >= self.grid[0].len() as isize {
                    continue;
                }
    
                // If there's a sample here, check that it's not too close to us
                if let Some(point2) = self.grid[x as usize][y as usize] {
                    if (point.0 - point2.0).powi(2) + (point.1 - point2.1).powi(2) < r_squared {
                        return true;
                    }
                }
            }
        }
    
        // We only make it to here if we find no samples too close
        false
    }
}

impl Iterator for PoissonIter {
    type Item = Point;

    fn next(&mut self) -> Option<Point> {
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
