//! Defines the structures which manage state during interactive execution of the prover and verifier
use crate::{
    arithmetic::{Field, Group, MultiScalarMul, Pairing},
    messages::{
        FirstReduceChallenge, FirstReduceMessage,
        SecondReduceChallenge, SecondReduceMessage,
        FinalizeChallenge
    },
};

/// Trait for the state and computation and state of the Dory protocol.
///
/// A type implementing this trait primarily stores the $v_i$ and $s_i$ vectors.
/// The trait methods define the operations needed to compute the messages exchanged.
/// This trait is not responsible for the actual messaging/proving. That is the job of the
/// [`ProofBuilder`](crate::ProofBuilder) trait. This is so that P and V actors can compute things as needed.
pub trait ProverState {
    /// The $\mathbb{G}_1$ group
    type G1: Group;
    /// The $\mathbb{G}_2$ group
    type G2: Group;
    /// The target group, $\mathbb{G}_T$
    type GT: Group;
    /// The scalar, $\mathbb{F}$, field of the groups
    type Scalar: Field;
    /// The setup type. This should contain the public parameters needed for the protocol.
    type Setup;

    /// Computes the [`FirstReduceMessage`] from the state. Specifically:
    /// D₁L = ⟨v₁L, Γ₂′⟩,  D₁R = ⟨v₁R, Γ₂′⟩;
    /// D₂L = ⟨Γ₁′, v₂L⟩,  D₂R = ⟨Γ₁′, v₂R⟩;
    /// E₁β = ⟨Γ₁, s₂⟩,   E₂β = ⟨s₁, Γ₂⟩.
    ///
    /// # Panics
    /// Panics if the state is not in an appropriate round. That is, if the last Reduce round has not been completed. This method
    /// assumes that the vᵢ and sᵢ vectors are of length at least 2.
    #[must_use]
    fn compute_first_reduce_message<M1, M2>(
        &self,
        setup: &Self::Setup,
    ) -> FirstReduceMessage<Self::G1, Self::G2, Self::GT>
    where
        Self::G1: Group,
        Self::G2: Group,
        M1: MultiScalarMul<Self::G1>,
        M2: MultiScalarMul<Self::G2>;
    /// Combines vᵢ with Γᵢ using the [`FirstReduceChallenge`] (step (*) in
    /// Dory-Reduce-Light). Updates:
    /// v₁(new) ← v₁ + β·Γ₁;  v₂(new) ← v₂ + β·Γ₂.
    /// 
    /// Note: can speed up this computation due to same scalar (β) applied to fixed base (Γ₁, Γ₂)
    ///
    /// # Panics
    /// Panics if the state is not in an appropriate round. That is, if the last Reduce round has not been completed. This method
    /// assumes that the $v_i$ and $s_i$ vectors are of length at least 2.
    #[must_use]
    fn reduce_combine<M1, M2>(
        self,
        setup: &Self::Setup,
        first_challenge: FirstReduceChallenge<Self::Scalar>,
    ) -> Self
    where
        M1: MultiScalarMul<Self::G1>,
        M2: MultiScalarMul<Self::G2>;
    /// Computes the [`SecondReduceMessage`] from the state.
    /// In the Light variant, C₊ and C₋ are omitted. After (*), define:
    /// E₁₊ = ⟨v₁R(new), s₂L⟩;  E₁₋ = ⟨v₁L(new), s₂R⟩;
    /// E₂₊ = ⟨s₁L, v₂R(new)⟩;  E₂₋ = ⟨s₁R, v₂L(new)⟩.
    ///
    /// # Panics
    /// Panics if the state is not in an appropriate round. That is, if the last Reduce round has not been completed. This method
    /// assumes that the $v_i$ and $s_i$ vectors are of length at least 2.
    #[must_use]
    fn compute_second_reduce_message<M1, M2>(
        &self,
        setup: &Self::Setup,
    ) -> SecondReduceMessage<Self::G1, Self::G2>
    where
        Self::G1: Group,
        Self::G2: Group,
        M1: MultiScalarMul<Self::G1>,
        M2: MultiScalarMul<Self::G2>;
    /// Folds the vᵢ and sᵢ vectors using the [`SecondReduceChallenge`] (step (**) in
    /// Dory-Reduce-Light). Updates:
    /// v₁′ ← v₁L(new) + α·v₁R(new);  v₂′ ← v₂L(new) + α·v₂R(new);
    /// s₁′ ← α·s₁L + s₁R;           s₂′ ← α·s₂L + s₂R.
    ///
    /// # Panics
    /// Panics if the state is not in an appropriate round. That is, if the last Reduce round has not been completed. This method
    /// assumes that the $v_i$ and $s_i$ vectors are of length at least 2.
    #[must_use]
    fn reduce_fold<M1, M2>(
        self,
        setup: &Self::Setup,
        second_challenge: SecondReduceChallenge<Self::Scalar>,
    ) -> Self
    where
        M1: MultiScalarMul<Self::G1>,
        M2: MultiScalarMul<Self::G2>;

    /// Return base-case group elements after all rounds (nu == 0)
    // #[must_use]
    fn final_bases(&self) -> (Self::G1, Self::G2);

    /// Borrow base-case group elements after all rounds (nu == 0)
    #[must_use]
    fn final_bases_ref(&self) -> (&Self::G1, &Self::G2);

    /// Borrow base-case folded scalar values (s1_final, s2_final)
    #[must_use]
    fn final_scalars_ref(&self) -> (&Self::Scalar, &Self::Scalar);

    // #[must_use]
    // fn final_prove<M1, M2>(
    //     self,
    //     setup: &Self::Setup,
    //     fold_scalars_challenge: FinalizeChallenge<Self::Scalar>,
    // ) -> FinalMessage<Self::G1, Self::G2>
    // where
    //     Self::G1: Group,
    //     Self::G2: Group,
    //     M1: MultiScalarMul<Self::G1>,
    //     M2: MultiScalarMul<Self::G2>;
}

// Verifier
///
/// Trait for the verifier state and computation during the Dory protocol.
///
/// A type implementing this trait maintains verification state and the operations
/// needed to verify the messages from the prover.
pub trait VerifierState {
    /// The $\mathbb{G}_1$ group
    type G1: Group;
    /// The $\mathbb{G}_2$ group
    type G2: Group;
    /// The target group, $\mathbb{G}_T$
    type GT: Group;
    /// The scalar, $\mathbb{F}$, field of the groups
    type Scalar: Field;
    /// The setup type. This should contain the public parameters needed for verification.
    type Setup;

    /// This is the verifier side of the extended Dory-Reduce algorithm in section 3.2 & 4.2 of the paper.
    fn dory_reduce_verify_round(
        &mut self,
        setup: &Self::Setup,
        first_msg: &FirstReduceMessage<Self::G1, Self::G2, Self::GT>,
        second_msg: &SecondReduceMessage<Self::G1, Self::G2>,
        beta: Self::Scalar,
        alpha: Self::Scalar,
    ) -> bool;

    /// Updates D₁ and D₂ in the verifier state (new writeup)
    /// D₁' <- D₁L + α * D₁R + β * (Δ₁L + α * Δ₁R)
    /// D₂' <- D₂L + α * D₂R + β * (Δ₂L + α * Δ₂R)
    fn dory_reduce_verify_update_ds(
        &mut self,
        setup: &Self::Setup,
        d_values: (&Self::GT, &Self::GT, &Self::GT, &Self::GT),
        alpha: Self::Scalar,
        beta: Self::Scalar,
    );

    /// Updates E₁ and E₂ in the extended verifier state (new writeup)
    /// E₁' <- E₁- + α * (E₁ + β * E₁β + α * E₁+)
    /// E₂' <- E₂- + α * (E₂ + β * E₂β + α * E₂+)
    fn dory_reduce_verify_update_es(
        &mut self,
        e_beta_pair: (&Self::G1, &Self::G2),
        e_values: (&Self::G1, &Self::G1, &Self::G2, &Self::G2),
        alpha: Self::Scalar,
        beta: Self::Scalar,
    );

    /// Final verification step for Dory-InnerProduct-Light
    /// Verifies: TBD
    fn finalize(
        &self,
        setup: &Self::Setup,
        gamma_pair: FinalizeChallenge<Self::Scalar>,
    ) -> bool;

    /// Store final base-case group elements sent by the prover
    fn set_final_bases(&mut self, v1_final: Self::G1, v2_final: Self::G2);
}

/// --------- Concrete ProverState and VerifierState ---------------------

pub struct DoryProverState<E: Pairing>
where
    E::G1: Group,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
{
    // these follow notation from the paper
    /// v1 - P witness
    pub v1: Vec<E::G1>,
    /// v2 - P witness
    pub v2: Vec<E::G2>,
    /// s1 - scalars for extended Dory IP (see Section 4)
    pub s1: Vec<<E::G1 as Group>::Scalar>,
    /// s2 - scalars for extended Dory IP (see Section 4)
    pub s2: Vec<<E::G1 as Group>::Scalar>,
    /// number of rounds
    pub nu: usize,
    /// Optional scalar vector v for PCS first-round optimization where v2 = v · Γ2_fin
    pub v2_scalars: Option<std::sync::Arc<[<E::G1 as Group>::Scalar]>>,
}

impl<E: Pairing> DoryProverState<E>
where
    E::G1: Group,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
{
    /// Constructor
    pub fn new(
        v1: Vec<E::G1>,
        v2: Vec<E::G2>,
        s1: Vec<<E::G1 as Group>::Scalar>,
        s2: Vec<<E::G1 as Group>::Scalar>,
        nu: usize,
    ) -> Self {
        Self { v1, v2, s1, s2, nu, v2_scalars: None }
    }

    /// Constructor with PCS first-round scalar vector for v2
    pub fn new_with_v2_scalars(
        v1: Vec<E::G1>,
        v2: Vec<E::G2>,
        s1: Vec<<E::G1 as Group>::Scalar>,
        s2: Vec<<E::G1 as Group>::Scalar>,
        v2_scalars: Vec<<E::G1 as Group>::Scalar>,
        nu: usize,
    ) -> Self {
        Self { v1, v2, s1, s2, nu, v2_scalars: Some(v2_scalars.into()) }
    }
}

/// Verifier state.
/// We track each commitment to be mutated during the interactive protocol
pub struct DoryVerifierState<E: Pairing>
where
    E::G1: Group,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
{
    /// The commitment to v1: <v1,Γ_2>.
    pub d_1: E::GT,
    /// The commitment to v2: <Γ_1,v2>.
    pub d_2: E::GT,
    /// Original D₂ from VMV-Reduce (kept unchanged for Nemo-Finalize)
    pub d_2_orig: E::GT,

    // extended use case:
    /// The commitment to s1: <v1,s2>.
    pub e_1: E::G1,
    /// Original E₁ from VMV-Reduce (kept unchanged for Nemo-Finalize)
    pub e_1_orig: E::G1,
    /// The commitment to s2: <s1,v2>.
    pub e_2: E::G2,

    /// Tensors used for VMV
    /// We only store the underlying evaluation point, not the tensored vector
    pub eval_point_left: std::sync::Arc<[<E::G1 as Group>::Scalar]>,
    /// Tensors used for VMV
    /// We only store the underlying evaluation point, not the tensored vector
    pub eval_point_right: std::sync::Arc<[<E::G1 as Group>::Scalar]>,

    /// Sequence of α challenges sampled across the IP reduction rounds
    pub alpha_challenges: Vec<<E::G1 as Group>::Scalar>,

    /// Base-case group elements provided by prover at the end of IP rounds
    pub v1_final: Option<E::G1>,
    /// Base-case group elements provided by prover at the end of IP rounds
    pub v2_final: Option<E::G2>,

    /// Current round number. Length of v1 and v2 should be 2^nu.
    pub nu: usize,
}
impl<E: Pairing> DoryVerifierState<E>
where
    E::G1: Group,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
{
    /// Constructor
    pub fn new(d_1: E::GT, d_2: E::GT, e_1: E::G1, e_2: E::G2, nu: usize) -> Self {
        Self {
            d_1,
            d_2: d_2.clone(),
            d_2_orig: d_2,
            e_1: e_1.clone(),
            e_1_orig: e_1,
            e_2,
            eval_point_left: {
                let v: Vec<<E::G1 as Group>::Scalar> = Vec::new();
                v.into()
            },
            eval_point_right: {
                let v: Vec<<E::G1 as Group>::Scalar> = Vec::new();
                v.into()
            },
            alpha_challenges: vec![],
            v1_final: None,
            v2_final: None,
            nu,
        }
    }

    /// Constructor
    pub fn new_with_eval_point(
        d_1: E::GT,
        d_2: E::GT,
        e_1: E::G1,
        e_2: E::G2,
        eval_point_left: std::sync::Arc<[<E::G1 as Group>::Scalar]>,
        eval_point_right: std::sync::Arc<[<E::G1 as Group>::Scalar]>,
        nu: usize,
    ) -> Self {
        Self {
            d_1,
            d_2: d_2.clone(),
            d_2_orig: d_2,
            e_1: e_1.clone(),
            e_1_orig: e_1,
            e_2,
            eval_point_left,
            eval_point_right,
            alpha_challenges: vec![],
            v1_final: None,
            v2_final: None,
            nu,
        }
    }
}
