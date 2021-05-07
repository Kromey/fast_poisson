#![cfg(feature = "derive_serde")]

use fast_poisson::Poisson2D;
use serde_json;

#[test]
fn serialize_and_deserialize() {
    // Be sure to set a seed so we can assert equality
    let mut poisson = Poisson2D::new();
    poisson.with_seed(1337);

    let json = serde_json::to_string(&poisson).unwrap();
    let decoded = serde_json::from_str(&json).unwrap();

    assert_eq!(poisson, decoded);
}
