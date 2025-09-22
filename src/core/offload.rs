//! GT offloading abstraction for recursive JOLT
use crate::arithmetic::{Group, Pairing};

#[cfg(feature = "recursion")]
use jolt_optimizations::ExponentiationSteps;
#[cfg(feature = "recursion")]
use std::collections::VecDeque;

/// Context for managing GT offloading operations in recursive SNARKs
pub struct OffloadContext {
    #[cfg(feature = "recursion")]
    queue: Option<VecDeque<ExponentiationSteps>>,
    #[cfg(not(feature = "recursion"))]
    _phantom: (),
}

impl OffloadContext {
    /// Create a new OffloadContext with no offloading enabled
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "recursion")]
            queue: None,
            #[cfg(not(feature = "recursion"))]
            _phantom: (),
        }
    }

    /// Create an OffloadContext with precomputed exponentiation steps for recursion
    #[cfg(feature = "recursion")]
    pub fn with_steps(steps: Vec<ExponentiationSteps>) -> Self {
        Self {
            queue: Some(VecDeque::from(steps)),
        }
    }

    /// Check if GT offloading is enabled
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

/// GT exp but uses offloaded values if available
pub fn scale_gt_with_offload<E>(
    value: &E::GT,
    scalar: &<E::GT as Group>::Scalar,
    #[allow(unused_variables)] ctx: &mut OffloadContext,
) -> E::GT
where
    E: Pairing,
    E::G1: Group,
    E::G2: Group,
    E::GT: Group,
{
    #[cfg(feature = "recursion")]
    {
        // Take a pre-computed value if available
        if let Some(step) = ctx.pop_result() {
            debug_assert_eq!(
                std::mem::size_of::<E::GT>(),
                std::mem::size_of_val(&step.result),
                "Size mismatch between GT type and precomputed result"
            );

            let precomputed_result: E::GT = unsafe { std::mem::transmute_copy(&step.result) };

            // This debug tests correctness of the offloaded values
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
    value.scale(scalar)
}
