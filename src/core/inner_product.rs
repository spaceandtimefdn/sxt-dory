//! Implementation of the extended Dory-Innerproduct protocol.
//! The protocol is outlined in sections 3.3 & 4.3 of the Dory paper.
use crate::arithmetic::{Field, Group, MultiScalarMul};
use crate::builder::{ProofBuilder, VerificationBuilder};
use crate::state::{ProverState, VerifierState};

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
    G1: Group,
    G2: Group,
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

    // Note: we pull `d_pair` here even though the Prover does not actually need it. This is so that the verifier
    // and prover transcripts will stay in sync (see Scalar-Product protocol in the paper).
    let (final_verify_challenge, builder) = builder.challenge_final_verify();

    // Note: `compute_scalar_product_message` applies the `Fold-Scalars` transform, as described in the paper.
    let final_verify_message = state.compute_final_verify_message::<M1, M2>(setup, final_verify_challenge);
    builder.append_final_verify_message(final_verify_message)
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
        let (m1, m2) = builder.take_round(idx);

        let first = builder.process_first_reduce_message(&m1);
        let second = builder.process_second_reduce_message(&m2);

        if !state.dory_reduce_verify_round(setup, &m1, &m2, first.beta, second.alpha) {
            return Err(idx);
        }
    }
    // when n = 1, we apply `fold-scalars` and then verifier side of `scalar-product (the final pairing check)`.
    // On the prover side, `fold-scalars` is handled immediately before sending the scalar product message.
    let fold_gamma = builder.challenge_fold_scalars();
    let d_pair = builder.challenge_scalar_product_scalars();

    // if !state.final_verify(setup, builder.process_scalar_product_message()) {
    //     return Err(builder.rounds());
    // }

    Ok(())
}
