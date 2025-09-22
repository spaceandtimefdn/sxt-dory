//! GT offloading abstraction for recursive SNARKs
//!
//! This module provides a clean abstraction for offloading expensive GT exponentiations
//! from the verification circuit to the proof generation phase.

use crate::arithmetic::{Group, Pairing};

#[cfg(feature = "recursion")]
use jolt_optimizations::ExponentiationSteps;
#[cfg(feature = "recursion")]
use std::collections::VecDeque;

/// Context for managing offloaded GT operations
pub struct OffloadContext {
    #[cfg(feature = "recursion")]
    queue: Option<VecDeque<ExponentiationSteps>>,
    #[cfg(not(feature = "recursion"))]
    _phantom: (),
}

impl OffloadContext {
    /// Create a new empty offload context
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "recursion")]
            queue: None,
            #[cfg(not(feature = "recursion"))]
            _phantom: (),
        }
    }

    /// Create an offload context with precomputed steps
    #[cfg(feature = "recursion")]
    pub fn with_steps(steps: Vec<ExponentiationSteps>) -> Self {
        Self {
            queue: Some(VecDeque::from(steps)),
        }
    }

    /// Check if offloading is available
    pub fn is_offloading_enabled(&self) -> bool {
        #[cfg(feature = "recursion")]
        {
            self.queue.is_some()
        }
        #[cfg(not(feature = "recursion"))]
        {
            false
        }
    }

    /// Get the next precomputed result if available
    #[cfg(feature = "recursion")]
    fn pop_result(&mut self) -> Option<ExponentiationSteps> {
        self.queue.as_mut().and_then(|q| q.pop_front())
    }
}

impl Default for OffloadContext {
    fn default() -> Self {
        Self::new()
    }
}


/// Helper function for GT scaling with optional offloading
pub fn scale_gt_with_offload<E>(
    value: &E::GT,
    scalar: &<E::GT as Group>::Scalar,
    ctx: &mut OffloadContext,
) -> E::GT
where
    E: Pairing,
    E::G1: Group,
    E::G2: Group,
    E::GT: Group,
{
    #[cfg(feature = "recursion")]
    {
        // Try to use precomputed result if available
        if let Some(step) = ctx.pop_result() {
            // Convert the precomputed result to the appropriate GT type
            // This is safe because we validate the size compatibility
            debug_assert_eq!(
                std::mem::size_of::<E::GT>(),
                std::mem::size_of_val(&step.result),
                "Size mismatch between GT type and precomputed result"
            );

            // Use unsafe transmute for the conversion
            let precomputed_result: E::GT = unsafe { std::mem::transmute_copy(&step.result) };

            // In debug mode, verify the result matches native computation
            #[cfg(debug_assertions)]
            {
                let native_result = value.scale(scalar);
                if precomputed_result != native_result {
                    panic!(
                        "GT offload mismatch: precomputed result differs from native computation!"
                    );
                }
            }

            return precomputed_result;
        }
    }

    // Fall back to native scaling
    value.scale(scalar)
}