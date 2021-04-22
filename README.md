# fast_poisson

[![Docs](https://docs.rs/fast_poisson/badge.svg)](https://docs.rs/fast_poisson/)
[![Crates.io](https://img.shields.io/crates/v/fast_poisson.svg)](https://crates.io/crates/fast_poisson)
[![CI](https://github.com/Kromey/fast_poisson/actions/workflows/rust.yml/badge.svg)](https://github.com/Kromey/fast_poisson/actions/workflows/rust.yml)

This is a library for generating Poisson disk distributions using [Bridson's algorithm][Bridson].

Properties of Poisson disk distributions include no two points being closer than a certain radius
and the distribution uniformly filling the space. Poisson disk distributions' blue noise properties
have a variety of applications in procedural generation, including textures, worlds, meshes, and
item placement.

# Usage

`fast_poisson` requires Rust 1.51.0 or later, as it relies on the const generics feature introduced
in this version.

A simple example to generate a `Vec` containing a Poisson distribution within [0, 1) in each
dimension:

```rust
use fast_poisson::Poisson2D;

fn main() {
    let poisson: Vec<[f64; 2]> = Vec::from(Poisson2D::new());
}
```

[Bridson]: https://www.cct.lsu.edu/~fharhad/ganbatte/siggraph2007/CD2/content/sketches/0250.pdf
