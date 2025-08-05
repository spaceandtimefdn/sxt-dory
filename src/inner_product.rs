use crate::builder::ProofBuilder;
use crate::state::ProverState;

/// Implementation of the extended Dory-Innerproduct protocol.
///
/// The protocol is outlined in sections 3.3 & 4.3 of the Dory paper.
pub fn inner_product_prove<Builder, State, G1, G2, GT, Scalar, Setup>(
    builder: Builder,
    state: State,
    setup: &Setup,
    num_rounds: usize,
) -> Builder
where
    Builder: ProofBuilder<G1 = G1, G2 = G2, GT = GT, Scalar = Scalar>,
    State: ProverState<G1 = G1, G2 = G2, GT = GT, Scalar = Scalar, Setup = Setup>,
{
    let (builder, state) = (0..num_rounds).fold((builder, state), |(builder, state), _| {
        let (challenge, builder) =
            builder.append_first_reduce_message(state.compute_first_reduce_message(setup));
        let state = state.reduce_combine(setup, challenge);
        let (challenge, builder) =
            builder.append_second_reduce_message(state.compute_second_reduce_message(setup));
        (builder, state.reduce_fold(setup, challenge))
    });
    let (challenge, builder) = builder.challenge_fold_scalars();
    builder.append_scalar_product_message(state.compute_scalar_product_message(setup, challenge))
}

/// These tests are quite convoluted, and are mostly here to ensure 100% code coverage.
#[cfg(test)]
mod tests {
    use mockall::{mock, Sequence};
    use rand::Rng;

    use crate::messages::{
        FirstReduceChallenge, FirstReduceMessage, FoldScalarsChallenge, ScalarProductMessage,
        SecondReduceChallenge, SecondReduceMessage,
    };
    use crate::{inner_product_prove, ProofBuilder, ProverState};

    mock! {
        pub ProofBuilder {}
        impl ProofBuilder for ProofBuilder {
            type G1 = u32;
            type G2 = u32;
            type GT = u32;
            type Scalar = u32;
            fn append_first_reduce_message(
                self,
                message: FirstReduceMessage<u32, u32, u32>,
            ) -> (FirstReduceChallenge<u32>, Self);
            fn append_second_reduce_message(
                self,
                message: SecondReduceMessage<u32, u32, u32>,
            ) -> (SecondReduceChallenge<u32>, Self);
            fn challenge_fold_scalars(self) -> (FoldScalarsChallenge<u32>, Self);
            fn append_scalar_product_message(self, message: ScalarProductMessage<u32, u32>)
                -> Self;
        }
    }

    mock! {
        pub ProverState {}
        impl ProverState for ProverState {
            type G1 = u32;
            type G2 = u32;
            type GT = u32;
            type Scalar = u32;
            type Setup = u32;
            fn compute_first_reduce_message(
                &self,
                setup: &u32,
            ) -> FirstReduceMessage<u32, u32, u32>;
            fn reduce_combine(
                self,
                setup: &u32,
                first_challenge: FirstReduceChallenge<u32>,
            ) -> Self;
            fn compute_second_reduce_message(
                &self,
                setup: &u32,
            ) -> SecondReduceMessage<u32, u32, u32>;
            fn reduce_fold(self, setup: &u32, second_challenge: SecondReduceChallenge<u32>)
                -> Self;
            fn compute_scalar_product_message(
                self,
                setup: &u32,
                fold_challenge: FoldScalarsChallenge<u32>,
            ) -> ScalarProductMessage<u32, u32>;
        }
    }

    #[test]
    fn we_can_prove_inner_product_with_0_rounds() {
        let rng = &mut rand::rng();

        let mock_setup: u32 = rng.random();

        // Final fold and scalar product (no reduce rounds)
        let mock_fold_scalars_challenge: FoldScalarsChallenge<u32> = rng.random();
        let mock_scalar_product_message: ScalarProductMessage<u32, u32> = rng.random();

        let mut builder = MockProofBuilder::new();

        // Final fold challenge (no reduce messages)
        builder
            .expect_challenge_fold_scalars()
            .times(1)
            .returning(move || {
                let mut builder = MockProofBuilder::new();

                // Final scalar product message
                builder
                    .expect_append_scalar_product_message()
                    .times(1)
                    .returning(move |message| {
                        assert_eq!(message, mock_scalar_product_message);
                        MockProofBuilder::new()
                    });
                (mock_fold_scalars_challenge, builder)
            });

        let mut seq = Sequence::new();
        let mut state = MockProverState::new();

        // Final scalar product (no reduce steps)
        state
            .expect_compute_scalar_product_message()
            .times(1)
            .returning(move |setup, challenge| {
                assert_eq!(setup, &mock_setup);
                assert_eq!(challenge, mock_fold_scalars_challenge);
                mock_scalar_product_message
            })
            .in_sequence(&mut seq);

        inner_product_prove(builder, state, &mock_setup, 0);
    }

    #[test]
    fn we_can_prove_inner_product_with_1_round() {
        let rng = &mut rand::rng();

        let mock_setup: u32 = rng.random();

        // Round 0 messages and challenges
        let mock_first_reduce_message_0: FirstReduceMessage<u32, u32, u32> = rng.random();
        let mock_first_reduce_challenge_0: FirstReduceChallenge<u32> = rng.random();
        let mock_second_reduce_message_0: SecondReduceMessage<u32, u32, u32> = rng.random();
        let mock_second_reduce_challenge_0: SecondReduceChallenge<u32> = rng.random();

        // Final fold and scalar product
        let mock_fold_scalars_challenge: FoldScalarsChallenge<u32> = rng.random();
        let mock_scalar_product_message: ScalarProductMessage<u32, u32> = rng.random();

        let mut builder = MockProofBuilder::new();

        // Round 0 - First reduce message
        builder
            .expect_append_first_reduce_message()
            .times(1)
            .returning(move |message| {
                assert_eq!(message, mock_first_reduce_message_0);
                let mut builder = MockProofBuilder::new();

                // Round 0 - Second reduce message
                builder
                    .expect_append_second_reduce_message()
                    .times(1)
                    .returning(move |message| {
                        assert_eq!(message, mock_second_reduce_message_0);
                        let mut builder = MockProofBuilder::new();

                        // Final fold challenge
                        builder
                            .expect_challenge_fold_scalars()
                            .times(1)
                            .returning(move || {
                                let mut builder = MockProofBuilder::new();

                                // Final scalar product message
                                builder
                                    .expect_append_scalar_product_message()
                                    .times(1)
                                    .returning(move |message| {
                                        assert_eq!(message, mock_scalar_product_message);
                                        MockProofBuilder::new()
                                    });
                                (mock_fold_scalars_challenge, builder)
                            });
                        (mock_second_reduce_challenge_0, builder)
                    });
                (mock_first_reduce_challenge_0, builder)
            });

        let mut seq = Sequence::new();
        let mut state = MockProverState::new();

        // Round 0 - First reduce
        state
            .expect_compute_first_reduce_message()
            .times(1)
            .returning(move |setup| {
                assert_eq!(setup, &mock_setup);
                mock_first_reduce_message_0
            })
            .in_sequence(&mut seq);
        state
            .expect_reduce_combine()
            .times(1)
            .returning(move |setup, challenge| {
                assert_eq!(setup, &mock_setup);
                assert_eq!(challenge, mock_first_reduce_challenge_0);
                let mut seq = Sequence::new();
                let mut state = MockProverState::new();

                // Round 0 - Second reduce
                state
                    .expect_compute_second_reduce_message()
                    .times(1)
                    .returning(move |setup| {
                        assert_eq!(setup, &mock_setup);
                        mock_second_reduce_message_0
                    })
                    .in_sequence(&mut seq);
                state
                    .expect_reduce_fold()
                    .times(1)
                    .returning(move |setup, challenge| {
                        assert_eq!(setup, &mock_setup);
                        assert_eq!(challenge, mock_second_reduce_challenge_0);
                        let mut seq = Sequence::new();
                        let mut state = MockProverState::new();

                        // Final scalar product
                        state
                            .expect_compute_scalar_product_message()
                            .times(1)
                            .returning(move |setup, challenge| {
                                assert_eq!(setup, &mock_setup);
                                assert_eq!(challenge, mock_fold_scalars_challenge);
                                mock_scalar_product_message
                            })
                            .in_sequence(&mut seq);
                        state
                    })
                    .in_sequence(&mut seq);
                state
            })
            .in_sequence(&mut seq);

        inner_product_prove(builder, state, &mock_setup, 1);
    }

    #[test]
    #[expect(clippy::too_many_lines)]
    fn we_can_prove_inner_product_with_2_rounds() {
        let rng = &mut rand::rng();

        let mock_setup: u32 = rng.random();

        // Round 0 messages and challenges
        let mock_first_reduce_message_0: FirstReduceMessage<u32, u32, u32> = rng.random();
        let mock_first_reduce_challenge_0: FirstReduceChallenge<u32> = rng.random();
        let mock_second_reduce_message_0: SecondReduceMessage<u32, u32, u32> = rng.random();
        let mock_second_reduce_challenge_0: SecondReduceChallenge<u32> = rng.random();

        // Round 1 messages and challenges
        let mock_first_reduce_message_1: FirstReduceMessage<u32, u32, u32> = rng.random();
        let mock_first_reduce_challenge_1: FirstReduceChallenge<u32> = rng.random();
        let mock_second_reduce_message_1: SecondReduceMessage<u32, u32, u32> = rng.random();
        let mock_second_reduce_challenge_1: SecondReduceChallenge<u32> = rng.random();

        // Final fold and scalar product
        let mock_fold_scalars_challenge: FoldScalarsChallenge<u32> = rng.random();
        let mock_scalar_product_message: ScalarProductMessage<u32, u32> = rng.random();

        let mut builder = MockProofBuilder::new();

        // Round 0 - First reduce message
        builder
            .expect_append_first_reduce_message()
            .times(1)
            .returning(move |message| {
                assert_eq!(message, mock_first_reduce_message_0);
                let mut builder = MockProofBuilder::new();

                // Round 0 - Second reduce message
                builder
                    .expect_append_second_reduce_message()
                    .times(1)
                    .returning(move |message| {
                        assert_eq!(message, mock_second_reduce_message_0);
                        let mut builder = MockProofBuilder::new();

                        // Round 1 - First reduce message
                        builder
                            .expect_append_first_reduce_message()
                            .times(1)
                            .returning(move |message| {
                                assert_eq!(message, mock_first_reduce_message_1);
                                let mut builder = MockProofBuilder::new();

                                // Round 1 - Second reduce message
                                builder
                                    .expect_append_second_reduce_message()
                                    .times(1)
                                    .returning(move |message| {
                                        assert_eq!(message, mock_second_reduce_message_1);
                                        let mut builder = MockProofBuilder::new();

                                        // Final fold challenge
                                        builder.expect_challenge_fold_scalars().times(1).returning(
                                            move || {
                                                let mut builder = MockProofBuilder::new();

                                                // Final scalar product message
                                                builder
                                                    .expect_append_scalar_product_message()
                                                    .times(1)
                                                    .returning(move |message| {
                                                        assert_eq!(
                                                            message,
                                                            mock_scalar_product_message
                                                        );
                                                        MockProofBuilder::new()
                                                    });
                                                (mock_fold_scalars_challenge, builder)
                                            },
                                        );
                                        (mock_second_reduce_challenge_1, builder)
                                    });
                                (mock_first_reduce_challenge_1, builder)
                            });
                        (mock_second_reduce_challenge_0, builder)
                    });
                (mock_first_reduce_challenge_0, builder)
            });

        let mut seq = Sequence::new();
        let mut state = MockProverState::new();

        // Round 0 - First reduce
        state
            .expect_compute_first_reduce_message()
            .times(1)
            .returning(move |setup| {
                assert_eq!(setup, &mock_setup);
                mock_first_reduce_message_0
            })
            .in_sequence(&mut seq);
        state
            .expect_reduce_combine()
            .times(1)
            .returning(move |setup, challenge| {
                assert_eq!(setup, &mock_setup);
                assert_eq!(challenge, mock_first_reduce_challenge_0);
                let mut seq = Sequence::new();
                let mut state = MockProverState::new();

                // Round 0 - Second reduce
                state
                    .expect_compute_second_reduce_message()
                    .times(1)
                    .returning(move |setup| {
                        assert_eq!(setup, &mock_setup);
                        mock_second_reduce_message_0
                    })
                    .in_sequence(&mut seq);
                state
                    .expect_reduce_fold()
                    .times(1)
                    .returning(move |setup, challenge| {
                        assert_eq!(setup, &mock_setup);
                        assert_eq!(challenge, mock_second_reduce_challenge_0);
                        let mut seq = Sequence::new();
                        let mut state = MockProverState::new();

                        // Round 1 - First reduce
                        state
                            .expect_compute_first_reduce_message()
                            .times(1)
                            .returning(move |setup| {
                                assert_eq!(setup, &mock_setup);
                                mock_first_reduce_message_1
                            })
                            .in_sequence(&mut seq);
                        state
                            .expect_reduce_combine()
                            .times(1)
                            .returning(move |setup, challenge| {
                                assert_eq!(setup, &mock_setup);
                                assert_eq!(challenge, mock_first_reduce_challenge_1);
                                let mut seq = Sequence::new();
                                let mut state = MockProverState::new();

                                // Round 1 - Second reduce
                                state
                                    .expect_compute_second_reduce_message()
                                    .times(1)
                                    .returning(move |setup| {
                                        assert_eq!(setup, &mock_setup);
                                        mock_second_reduce_message_1
                                    })
                                    .in_sequence(&mut seq);
                                state
                                    .expect_reduce_fold()
                                    .times(1)
                                    .returning(move |setup, challenge| {
                                        assert_eq!(setup, &mock_setup);
                                        assert_eq!(challenge, mock_second_reduce_challenge_1);
                                        let mut seq = Sequence::new();
                                        let mut state = MockProverState::new();

                                        // Final scalar product
                                        state
                                            .expect_compute_scalar_product_message()
                                            .times(1)
                                            .returning(move |setup, challenge| {
                                                assert_eq!(setup, &mock_setup);
                                                assert_eq!(challenge, mock_fold_scalars_challenge);
                                                mock_scalar_product_message
                                            })
                                            .in_sequence(&mut seq);
                                        state
                                    })
                                    .in_sequence(&mut seq);
                                state
                            })
                            .in_sequence(&mut seq);
                        state
                    })
                    .in_sequence(&mut seq);
                state
            })
            .in_sequence(&mut seq);

        inner_product_prove(builder, state, &mock_setup, 2);
    }
}
