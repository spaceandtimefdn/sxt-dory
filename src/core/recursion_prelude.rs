//! Unified imports for recursion

#[cfg(feature = "recursion")]
pub use jolt_optimizations::ExponentiationSteps;

#[cfg(not(feature = "recursion"))]
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

#[cfg(not(feature = "recursion"))]
#[derive(Debug, Clone, Default, CanonicalSerialize, CanonicalDeserialize)]
/// Used for recursion poly tracking
pub struct ExponentiationSteps;

/// Type alias for GT steps
pub type RecursionOps = Option<Vec<ExponentiationSteps>>;
