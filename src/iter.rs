// Copyright 2021 Travis Veazey
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// https://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// https://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use super::{Float, Poisson};
use rand::prelude::*;
use rand_distr::StandardNormal;
use std::iter::FusedIterator;

#[cfg(test)]
mod tests;

/// A Point is simply an array of Float values
pub type Point<const N: usize> = [Float; N];

/// A Cell is the grid coordinates containing a given point
type Cell<const N: usize> = [isize; N];

#[cfg(not(feature = "small_rng"))]
type Rand = rand_xoshiro::Xoshiro256StarStar;
#[cfg(feature = "small_rng")]
type Rand = rand_xoshiro::Xoshiro128StarStar;

/// An iterator over the points in the Poisson disk distribution
pub struct Iter<const N: usize> {
    /// The distribution from which this iterator was built
    distribution: Poisson<N>,
    /// The RNG
    rng: Rand,
    /// The size of each cell in the grid
    cell_size: Float,
    /// The grid stores spatially-oriented samples for fast checking of neighboring sample points
    grid: Vec<Option<Point<N>>>,
    /// A list of valid points that we have not yet visited
    active: Vec<Point<N>>,
}

impl<const N: usize> Iter<N> {
    /// Create an iterator over the specified distribution
    pub(crate) fn new(distribution: Poisson<N>) -> Self {
        // We maintain a grid of our samples for faster radius checking
        let cell_size = distribution.radius / (N as Float).sqrt();

        // If we were not given a seed, generate one non-deterministically
        let mut rng = match distribution.seed {
            None => Rand::from_entropy(),
            Some(seed) => Rand::seed_from_u64(seed),
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

        let mut iter = Iter {
            distribution,
            rng,
            cell_size,
            grid: vec![None; grid_size],
            active: Vec::new(),
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
    fn generate_random_point(&mut self, around: Point<N>) -> Point<N> {
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
        let mut point = [0.0; N];
        let translate = dist / mag; // compute this just once!
        for i in 0..N {
            point[i] = around[i] + vector[i] * translate;
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
            for i in neighbor.iter_mut() {
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

impl<const N: usize> Iterator for Iter<N> {
    type Item = Point<N>;

    fn next(&mut self) -> Option<Point<N>> {
        while !self.active.is_empty() {
            let i = self.rng.gen_range(0..self.active.len());

            for _ in 0..self.distribution.num_samples {
                // Generate up to `num_samples` random points between radius and 2*radius from the current point
                let point = self.generate_random_point(self.active[i]);

                // Ensure we've picked a point inside the bounds of our rectangle, and more than `radius`
                // distance from any other sampled point
                if self.in_space(point) && !self.in_neighborhood(point) {
                    // We've got a good one!
                    self.add_point(point);

                    return Some(point);
                }
            }

            self.active.swap_remove(i);
        }

        None
    }
}

impl<const N: usize> FusedIterator for Iter<N> {}
