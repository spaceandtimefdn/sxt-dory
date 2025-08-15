//! # Dory
//!
//! Dory is a polynomial commitment scheme with excellent asymptotic performance as well as
//! practical efficiency. It is based on the work of Jonathan Lee (<https://eprint.iacr.org/2020/1274.pdf>).
//!
//! This crate provides a Rust implementation of the commitment scheme, intended to be usable as a
//! building block for other zk/SNARK protocols.

pub mod core;
pub mod primitives;

// Re-export commonly used items at the crate root
pub use core::*;

pub use primitives::{arithmetic, transcript};
