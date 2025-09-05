//! Implementation of the extended Dory-Innerproduct protocol.
//! The protocol is outlined in sections 3.3 & 4.3 of the Dory paper.
use crate::arithmetic::{Field, Group, MultiScalarMul};
use crate::builder::{ProofBuilder, VerificationBuilder};
use crate::state::{ProverState, VerifierState};
use crate::messages::FinalBasesMessage;

/// Prover side of extended Dory-innerproduct
/// Follows very closely the prover side of the protocol on Page 24.
#[tracing::instrument(skip_all)]
pub fn inner_product_prove<Builder, State, G1, G2, GT, Scalar, Setup, M1, M2>(
    builder: Builder,
    state: State,
    setup: &Setup,
    num_rounds: usize,
) -> Builder
where
    G1: Group<Scalar = Scalar>,
    G2: Group<Scalar = Scalar>,
    GT: Group,
    Scalar: Field,
    Builder: ProofBuilder<G1 = G1, G2 = G2, GT = GT, Scalar = Scalar>,
    State: ProverState<G1 = G1, G2 = G2, GT = GT, Scalar = Scalar, Setup = Setup>,
    M1: MultiScalarMul<G1>,
    M2: MultiScalarMul<G2>,
{
    let (builder, state) = (0..num_rounds).fold((builder, state), |(builder, state), _| {
        let first_reduce_msg = state.compute_first_reduce_message::<M1, M2>(setup);
        let (challenge, builder) = builder.append_first_reduce_message(first_reduce_msg);

        let state = state.reduce_combine::<M1, M2>(setup, challenge);

        let second_reduce_msg = state.compute_second_reduce_message::<M1, M2>(setup);
        let (challenge, builder) = builder.append_second_reduce_message(second_reduce_msg);

        let folded_state = state.reduce_fold::<M1, M2>(setup, challenge);
        (builder, folded_state)
    });

    // Keep transcripts in sync: derive finalize challenge (prover does not send a message here).
    let (_finalize_challenge, builder) = builder.challenge_finalize();

    // At base case, expose v1', v2' from prover state and append to proof
    let (v1_final, v2_final) = state.final_bases();
    let final_bases = FinalBasesMessage { v1_final, v2_final };
    builder.append_final_bases(final_bases)
}

/// Verifier analogue for the extended Dory-innerproduct
pub fn inner_product_verify<B, State, G1, G2, GT, Scalar, Setup>(
    mut builder: B,
    mut state: State,
    setup: &Setup,
    num_rounds: usize,
) -> Result<(), usize>
where
    G1: Group,
    G2: Group,
    GT: Group,
    Scalar: Field,
    State: VerifierState<G1 = G1, G2 = G2, GT = GT, Scalar = Scalar, Setup = Setup>,
    B: VerificationBuilder<G1 = G1, G2 = G2, GT = GT, Scalar = Scalar>,
{
    // We first check each of the log(n) rounds until we reduce to a statement of size 1
    for idx in 0..(num_rounds) {
        // First update transcript to derive challenges (mut borrow only)
        let first = builder.process_first_reduce_message_at(idx);
        let second = builder.process_second_reduce_message_at(idx);

        // Then borrow messages immutably for state update
        let m1 = builder.first_message(idx);
        let m2 = builder.second_message(idx);
        if !state.dory_reduce_verify_round(setup, m1, m2, first.beta, second.alpha) {
            return Err(idx);
        }
    }
    // Finalize (deferred pairing and linear checks)
    let finalize_challenge = builder.challenge_finalize();
    if !state.finalize(setup, finalize_challenge) {
        return Err(builder.rounds());
    }

    Ok(())
}
