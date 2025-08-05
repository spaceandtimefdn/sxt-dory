use serde::{Deserialize, Serialize};

/// The first prover message in the Dory-Reduce portion (Section 3.2) of the Dory protocol.
///
/// This consists of $D_{1L}$, $D_{1R}$, $D_{2L}$, $D_{2R}$, $E_{1\beta}$, and $E_{2\beta}$.
#[derive(Copy, Clone, Debug, Default, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct FirstReduceMessage<G1, G2, GT> {
    /// $D_{1L}$
    pub d1_left: GT,
    /// $D_{1R}$
    pub d1_right: GT,
    /// $D_{2L}$
    pub d2_left: GT,
    /// $D_{2R}$
    pub d2_right: GT,
    /// $E_{1\beta}$ (extension - Section 4.2 of paper)
    pub e1_beta: G1,
    /// $E_{2\beta}$ (extension - Section 4.2 of paper)
    pub e2_beta: G2,
}

/// The the first verifier challenge in the Dory-Reduce portion (Section 3.2) of the Dory protocol.
///
/// The challenge, $\beta$, is a random scalar. Additionally, $\beta$ must be non-zero because
/// the protocol uses $\beta^{-1}$, which we also include here.
#[derive(Copy, Clone, Debug, Default, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct FirstReduceChallenge<Scalar> {
    /// $\beta$
    pub beta: Scalar,
    /// $\beta^{-1}$
    pub beta_inverse: Scalar,
}

/// The second prover message in the Dory-Reduce portion (Section 3.2) of the Dory protocol.
///
/// This consists of $C_+$, $C_-$, $E_{1+}$, $E_{1-}$, $E_{2+}$, and $E_{2-}$.
#[derive(Copy, Clone, Debug, Default, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SecondReduceMessage<G1, G2, GT> {
    /// $C_+$
    pub c_plus: GT,
    /// $C_-$
    pub c_minus: GT,
    /// $E_{1+}$ (extension - Section 4.2 of paper)
    pub e1_plus: G1,
    /// $E_{1-}$ (extension - Section 4.2 of paper)
    pub e1_minus: G1,
    /// $E_{2+}$ (extension - Section 4.2 of paper)
    pub e2_plus: G2,
    /// $E_{2-}$ (extension - Section 4.2 of paper)
    pub e2_minus: G2,
}

/// The second verifier challenge in the Dory-Reduce portion (Section 3.2) of the Dory protocol.
///
/// The challenge, $\alpha$, is a random scalar. Additionally, $\alpha$ must be non-zero because
/// the protocol uses $\alpha^{-1}$, which we also include here.
#[derive(Copy, Clone, Debug, Default, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct SecondReduceChallenge<Scalar> {
    /// $\alpha$
    pub alpha: Scalar,
    /// $\alpha^{-1}$
    pub alpha_inverse: Scalar,
}

/// The verifier challenge in the Fold-Scalars portion (Section 4.1) of the Dory protocol.
///
/// The challenge, $\gamma$, is a random scalar. Additionally, $\gamma$ must be non-zero because
/// the protocol uses $\gamma^{-1}$, which we also include here.
#[derive(Copy, Clone, Debug, Default, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct FoldScalarsChallenge<Scalar> {
    /// $\gamma$
    pub gamma: Scalar,
    /// $\gamma^{-1}$
    pub gamma_inverse: Scalar,
}

/// The prover message in the Scalar-Product portion (Section 3.1) of the Dory protocol.
///
/// This consists of $E_1$ and $E_2$.
#[derive(Copy, Clone, Debug, Default, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct ScalarProductMessage<G1, G2> {
    /// $E_1$
    pub e1: G1,
    /// $E_2$
    pub e2: G2,
}

/// This module provides random message generation for testing purposes.
/// This could, in theory, easily be hidden behind a `random` feature flag,
/// but there is no obvious use case for this.
#[cfg(test)]
mod random_messages {

    use rand::distr::{Distribution, StandardUniform};

    use super::{
        FirstReduceChallenge, FirstReduceMessage, FoldScalarsChallenge, ScalarProductMessage,
        SecondReduceChallenge, SecondReduceMessage,
    };

    impl<G1, G2, GT> Distribution<FirstReduceMessage<G1, G2, GT>> for StandardUniform
    where
        StandardUniform: Distribution<G1>,
        StandardUniform: Distribution<G2>,
        StandardUniform: Distribution<GT>,
    {
        fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> FirstReduceMessage<G1, G2, GT> {
            FirstReduceMessage {
                d1_left: self.sample(rng),
                d1_right: self.sample(rng),
                d2_left: self.sample(rng),
                d2_right: self.sample(rng),
                e1_beta: self.sample(rng),
                e2_beta: self.sample(rng),
            }
        }
    }

    impl<Scalar> Distribution<FirstReduceChallenge<Scalar>> for StandardUniform
    where
        StandardUniform: Distribution<Scalar>,
    {
        fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> FirstReduceChallenge<Scalar> {
            FirstReduceChallenge {
                beta: self.sample(rng),
                beta_inverse: self.sample(rng),
            }
        }
    }

    impl<G1, G2, GT> Distribution<SecondReduceMessage<G1, G2, GT>> for StandardUniform
    where
        StandardUniform: Distribution<G1>,
        StandardUniform: Distribution<G2>,
        StandardUniform: Distribution<GT>,
    {
        fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> SecondReduceMessage<G1, G2, GT> {
            SecondReduceMessage {
                c_plus: self.sample(rng),
                c_minus: self.sample(rng),
                e1_plus: self.sample(rng),
                e1_minus: self.sample(rng),
                e2_plus: self.sample(rng),
                e2_minus: self.sample(rng),
            }
        }
    }

    impl<Scalar> Distribution<SecondReduceChallenge<Scalar>> for StandardUniform
    where
        StandardUniform: Distribution<Scalar>,
    {
        fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> SecondReduceChallenge<Scalar> {
            SecondReduceChallenge {
                alpha: self.sample(rng),
                alpha_inverse: self.sample(rng),
            }
        }
    }

    impl<Scalar> Distribution<FoldScalarsChallenge<Scalar>> for StandardUniform
    where
        StandardUniform: Distribution<Scalar>,
    {
        fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> FoldScalarsChallenge<Scalar> {
            FoldScalarsChallenge {
                gamma: self.sample(rng),
                gamma_inverse: self.sample(rng),
            }
        }
    }

    impl<G1, G2> Distribution<ScalarProductMessage<G1, G2>> for StandardUniform
    where
        StandardUniform: Distribution<G1>,
        StandardUniform: Distribution<G2>,
    {
        fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> ScalarProductMessage<G1, G2> {
            ScalarProductMessage {
                e1: self.sample(rng),
                e2: self.sample(rng),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::rand_core::impls::fill_bytes_via_next;
    use rand::{Rng, RngCore};

    use super::{
        FirstReduceChallenge, FirstReduceMessage, FoldScalarsChallenge, ScalarProductMessage,
        SecondReduceChallenge, SecondReduceMessage,
    };

    struct MockRng {
        next: u64,
    }
    impl RngCore for MockRng {
        #[expect(clippy::cast_possible_truncation)]
        fn next_u32(&mut self) -> u32 {
            self.next_u64() as u32
        }
        fn next_u64(&mut self) -> u64 {
            self.next += 1;
            self.next - 1
        }
        fn fill_bytes(&mut self, dst: &mut [u8]) {
            fill_bytes_via_next(self, dst);
        }
    }

    #[test]
    fn we_can_generate_sequential_values_from_mock_rng() {
        let mut rng = MockRng { next: 0 };
        assert_eq!(rng.next_u32(), 0);
        assert_eq!(rng.next_u32(), 1);
        assert_eq!(rng.next_u64(), 2);
        assert_eq!(rng.next_u64(), 3);
        let mut buf = [0u8; 10];
        rng.fill_bytes(&mut buf);
        assert_eq!(buf, [4, 0, 0, 0, 0, 0, 0, 0, 5, 0]);
        assert_eq!(rng.random::<u64>(), 6);
        assert_eq!(rng.random::<u32>(), 7);
        assert_eq!(rng.random::<i64>(), 8);
        assert_eq!(rng.random::<i32>(), 9);
    }

    #[test]
    fn we_can_generate_random_messages() {
        let mut rng = MockRng { next: 0 };
        assert_eq!(
            rng.random::<FirstReduceMessage<u32, u64, u32>>(),
            FirstReduceMessage {
                d1_left: 0,
                d1_right: 1,
                d2_left: 2,
                d2_right: 3,
                e1_beta: 4,
                e2_beta: 5,
            }
        );
        assert_eq!(
            rng.random::<FirstReduceChallenge<u32>>(),
            FirstReduceChallenge {
                beta: 6,
                beta_inverse: 7,
            }
        );
        assert_eq!(
            rng.random::<SecondReduceMessage<u32, u64, u32>>(),
            SecondReduceMessage {
                c_plus: 8,
                c_minus: 9,
                e1_plus: 10,
                e1_minus: 11,
                e2_plus: 12,
                e2_minus: 13,
            }
        );
        assert_eq!(
            rng.random::<SecondReduceChallenge<u32>>(),
            SecondReduceChallenge {
                alpha: 14,
                alpha_inverse: 15,
            }
        );
        assert_eq!(
            rng.random::<FoldScalarsChallenge<u32>>(),
            FoldScalarsChallenge {
                gamma: 16,
                gamma_inverse: 17,
            }
        );
        assert_eq!(
            rng.random::<ScalarProductMessage<u32, u64>>(),
            ScalarProductMessage { e1: 18, e2: 19 }
        );
    }
}
