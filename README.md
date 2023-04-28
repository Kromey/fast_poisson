# fast_poisson

[![Docs](https://docs.rs/fast_poisson/badge.svg)](https://docs.rs/fast_poisson/)
[![Crates.io](https://img.shields.io/crates/v/fast_poisson.svg)](https://crates.io/crates/fast_poisson)
[![CI](https://github.com/Kromey/fast_poisson/actions/workflows/rust.yml/badge.svg)](https://github.com/Kromey/fast_poisson/actions/workflows/rust.yml)

This is a library for generating Poisson disk distributions using [Bridson's algorithm][Bridson].

Properties of Poisson disk distributions include no two points being closer than a certain radius
and the distribution uniformly filling the space. Poisson disk distributions' blue noise properties
have a variety of applications in procedural generation, including textures, worlds, meshes, and
item placement.

## Usage

A simple example to generate a `Vec` containing a 2D Poisson distribution within [0, 1) in each
dimension:

```rust
use fast_poisson::Poisson2D;

fn main() {
    let poisson = Poisson2D::new().generate();
}
```

See [the documentation](https://docs.rs/fast_poisson/) for more.

## MSRV

`fast_poisson` is tested and supported for Rust version 1.59 or later.

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as
above, without any additional terms or conditions.

[Bridson]: https://www.cct.lsu.edu/~fharhad/ganbatte/siggraph2007/CD2/content/sketches/0250.pdf
