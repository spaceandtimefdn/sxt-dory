//! Dory follows an interactive model. Hence, a "proof" consists of some messages
//! between P and V. We use Prover and Verifier "builders" to manage these messages
//! and the fiat-shamir challenges throughout the implementation.
use crate::transcript::Transcript;
use std::marker::PhantomData;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::{
    arithmetic::{Field, Group},
    messages::{
        FirstReduceChallenge, FirstReduceMessage, FoldScalarsChallenge, ScalarProductChallenge,
        ScalarProductMessage, SecondReduceChallenge, SecondReduceMessage, VMVMessage,
    },
    toy_transcript::ToyTranscript,
};

#[cfg(feature = "recursion")]
use jolt_optimizations::ExponentiationSteps;

/// A serializable proof struct that contains all the messages exchanged
#[derive(Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryProof<G1, G2, GT>
where
    G1: Group,
    G2: Group,
    GT: Group,
{
    /// First prover messages for each round
    pub first_messages: Vec<FirstReduceMessage<G1, G2, GT>>,
    /// Second prover messages for each round
    pub second_messages: Vec<SecondReduceMessage<G1, G2, GT>>,
    /// Final scalar product message
    pub final_message: Option<ScalarProductMessage<G1, G2>>,
    /// Vector-matrix-vector message (for PCS)
    pub vmv_message: Option<VMVMessage<G1, GT>>,
    /// GT exponentiation steps for recursion
    #[cfg(feature = "recursion")]
    pub gt_exponentiation_steps: Vec<ExponentiationSteps>,
}

/// Trait that defines the structure of the Dory proof.
///
/// A type implementing this trait acts as both the transcript and the proof serializer.
/// This is because these two concepts are closely related, and likely should use the same
/// underlying serialization.
pub trait ProofBuilder {
    /// G1 x G2 -> GT
    type Pairing;
    /// The $\mathbb{G}_1$ group
    type G1: Group;
    /// The $\mathbb{G}_2$ group
    type G2: Group;
    /// The target group, $\mathbb{G}_T$
    type GT: Group;
    /// The scalar field, $\mathbb{F}$, of the groups
    type Scalar: Field;

    /// Append a [`FirstReduceMessage`] to the proof and transcript and return a [`FirstReduceChallenge`] drawn from the transcript.
    #[must_use]
    fn append_first_reduce_message(
        self,
        message: FirstReduceMessage<Self::G1, Self::G2, Self::GT>,
    ) -> (FirstReduceChallenge<Self::Scalar>, Self);
    /// Append a [`SecondReduceMessage`] to the proof and transcript and return a [`SecondReduceChallenge`] drawn from the transcript.
    #[must_use]
    fn append_second_reduce_message(
        self,
        message: SecondReduceMessage<Self::G1, Self::G2, Self::GT>,
    ) -> (SecondReduceChallenge<Self::Scalar>, Self);
    /// Draw a [`FoldScalarsChallenge`] from the transcript.
    #[must_use]
    fn challenge_fold_scalars(self) -> (FoldScalarsChallenge<Self::Scalar>, Self);
    /// Append a [`ScalarProductMessage`] to the proof and transcript.
    #[must_use]
    fn append_scalar_product_message(
        self,
        message: ScalarProductMessage<Self::G1, Self::G2>,
    ) -> Self;
    #[must_use]
    /// Append a [`VMVMessage`] to the proof and transcript.
    fn append_vmv_message(self, message: VMVMessage<Self::G1, Self::GT>) -> Self;

    /// Draw a [`ScalarProductChallenge`] from the transcript.
    #[must_use]
    fn challenge_scalar_product_scalars(self) -> (ScalarProductChallenge<Self::Scalar>, Self);

    /// Finalize the proof for recursion by computing GT exponentiation steps
    /// This should be called after all rounds are complete
    #[cfg(feature = "recursion")]
    fn finalize_for_recursion<E>(
        self,
        _setup: &crate::setup::ProverSetup<E>,
        _initial_nu: usize,
    ) -> Self
    where
        E: crate::arithmetic::Pairing<GT = Self::GT>,
        Self::GT: crate::arithmetic::Group + Clone,
        Self: Sized,
    {
        // Default implementation just returns self
        // Concrete implementations can override
        self
    }
}

/// Concrete ProofBuilder to collect messages and perform transcript tasks
#[derive(Clone, Debug)]
pub struct DoryProofBuilder<G1, G2, GT, Scalar, T>
where
    G1: Group<Scalar = Scalar>,
    G2: Group<Scalar = Scalar>,
    GT: Group<Scalar = Scalar>,
    Scalar: Field,
    T: Transcript<Scalar = Scalar>,
{
    /// First prover message for round i
    pub first_messages: Vec<FirstReduceMessage<G1, G2, GT>>,
    /// First reduce challenges for each round (beta, beta_inverse)
    #[cfg(feature = "recursion")]
    pub first_challenges: Vec<FirstReduceChallenge<Scalar>>,
    /// Second reduce challenges for each round (alpha, alpha_inverse)
    #[cfg(feature = "recursion")]
    pub second_challenges: Vec<SecondReduceChallenge<Scalar>>,
    /// Second prover message for round i
    pub second_messages: Vec<SecondReduceMessage<G1, G2, GT>>,
    /// Last Scalar product message at end of protocol
    pub final_message: Option<ScalarProductMessage<G1, G2>>,

    /// vector-matrix-vector message, used to transform general dory into PCS
    pub vmv_message: Option<VMVMessage<G1, GT>>,
    /// GT exponentiation steps for recursion
    #[cfg(feature = "recursion")]
    pub gt_exponentiation_steps: Vec<ExponentiationSteps>,
    /// Delta values from setup for offloading GT operations
    #[cfg(feature = "recursion")]
    pub setup_delta_1l: Vec<GT>,
    #[cfg(feature = "recursion")]
    pub setup_delta_1r: Vec<GT>,
    #[cfg(feature = "recursion")]
    pub setup_delta_2l: Vec<GT>,
    #[cfg(feature = "recursion")]
    pub setup_delta_2r: Vec<GT>,
    /// Fiat shamir
    pub transcript: T,
    /// Phantom
    pub _phantom: PhantomData<(G1, G2, GT, Scalar)>,
}

impl<G1, G2, GT, Scalar, T> DoryProofBuilder<G1, G2, GT, Scalar, T>
where
    G1: Group<Scalar = Scalar>,
    G2: Group<Scalar = Scalar>,
    GT: Group<Scalar = Scalar>,
    Scalar: Field,
    T: Transcript<Scalar = Scalar>,
{
    /// Constructor from new transcript and setup
    #[cfg(feature = "recursion")]
    pub fn new<E>(transcript: T, setup: &crate::setup::ProverSetup<E>) -> Self
    where
        E: crate::arithmetic::Pairing<GT = GT>,
        GT: Clone,
    {
        Self {
            first_messages: Vec::new(),
            first_challenges: Vec::new(),
            second_challenges: Vec::new(),
            second_messages: Vec::new(),
            final_message: None,
            vmv_message: None,
            gt_exponentiation_steps: Vec::new(),
            setup_delta_1l: setup.delta_1l.clone(),
            setup_delta_1r: setup.delta_1r.clone(),
            setup_delta_2l: setup.delta_2l.clone(),
            setup_delta_2r: setup.delta_2r.clone(),
            transcript,
            _phantom: PhantomData,
        }
    }

    /// Constructor from new transcript (without setup for non-recursion mode)
    #[cfg(not(feature = "recursion"))]
    pub fn new(transcript: T) -> Self {
        Self {
            first_messages: Vec::new(),
            second_messages: Vec::new(),
            final_message: None,
            vmv_message: None,
            transcript,
            _phantom: PhantomData,
        }
    }

    /// Constructor to create with ToyTranscript for testing
    #[cfg(feature = "recursion")]
    pub fn new_with_toy_transcript<E>(
        domain: &[u8],
        setup: &crate::setup::ProverSetup<E>,
    ) -> DoryProofBuilder<G1, G2, GT, Scalar, ToyTranscript>
    where
        ToyTranscript: Transcript<Scalar = Scalar>,
        E: crate::arithmetic::Pairing<GT = GT>,
        GT: Clone,
    {
        let transcript = ToyTranscript::new(domain);
        DoryProofBuilder {
            first_messages: Vec::new(),
            first_challenges: Vec::new(),
            second_challenges: Vec::new(),
            second_messages: Vec::new(),
            final_message: None,
            vmv_message: None,
            gt_exponentiation_steps: Vec::new(),
            setup_delta_1l: setup.delta_1l.clone(),
            setup_delta_1r: setup.delta_1r.clone(),
            setup_delta_2l: setup.delta_2l.clone(),
            setup_delta_2r: setup.delta_2r.clone(),
            transcript,
            _phantom: PhantomData,
        }
    }

    /// Constructor to create with ToyTranscript for testing (non-recursion)
    #[cfg(not(feature = "recursion"))]
    pub fn new_with_toy_transcript(
        domain: &[u8],
    ) -> DoryProofBuilder<G1, G2, GT, Scalar, ToyTranscript>
    where
        ToyTranscript: Transcript<Scalar = Scalar>,
    {
        let transcript = ToyTranscript::new(domain);
        DoryProofBuilder {
            first_messages: Vec::new(),
            second_messages: Vec::new(),
            final_message: None,
            vmv_message: None,
            transcript,
            _phantom: PhantomData,
        }
    }

    /// Build a serializable Dory proof
    pub fn build(&self) -> DoryProof<G1, G2, GT> {
        DoryProof {
            first_messages: self.first_messages.clone(),
            second_messages: self.second_messages.clone(),
            final_message: self.final_message.clone(),
            vmv_message: self.vmv_message.clone(),
            #[cfg(feature = "recursion")]
            gt_exponentiation_steps: self.gt_exponentiation_steps.clone(),
        }
    }

    /// Create a DoryProofBuilder from a DoryProof and a fresh transcript
    pub fn from_proof(proof: DoryProof<G1, G2, GT>, transcript: T) -> Self {
        Self {
            first_messages: proof.first_messages,
            #[cfg(feature = "recursion")]
            first_challenges: Vec::new(), // Challenges are not stored in proof, need to be regenerated
            #[cfg(feature = "recursion")]
            second_challenges: Vec::new(),
            second_messages: proof.second_messages,
            final_message: proof.final_message,
            vmv_message: proof.vmv_message,
            #[cfg(feature = "recursion")]
            gt_exponentiation_steps: proof.gt_exponentiation_steps,
            #[cfg(feature = "recursion")]
            setup_delta_1l: Vec::new(),
            #[cfg(feature = "recursion")]
            setup_delta_1r: Vec::new(),
            #[cfg(feature = "recursion")]
            setup_delta_2l: Vec::new(),
            #[cfg(feature = "recursion")]
            setup_delta_2r: Vec::new(),
            transcript,
            _phantom: PhantomData,
        }
    }

    /// Create a DoryProofBuilder from a DoryProof with a default transcript
    pub fn from_proof_no_transcript(proof: DoryProof<G1, G2, GT>) -> Self
    where
        T: Default,
    {
        DoryProofBuilder {
            first_messages: proof.first_messages,
            #[cfg(feature = "recursion")]
            first_challenges: Vec::new(),
            #[cfg(feature = "recursion")]
            second_challenges: Vec::new(),
            second_messages: proof.second_messages,
            final_message: proof.final_message,
            vmv_message: proof.vmv_message,
            #[cfg(feature = "recursion")]
            gt_exponentiation_steps: proof.gt_exponentiation_steps,
            #[cfg(feature = "recursion")]
            setup_delta_1l: Vec::new(),
            #[cfg(feature = "recursion")]
            setup_delta_1r: Vec::new(),
            #[cfg(feature = "recursion")]
            setup_delta_2l: Vec::new(),
            #[cfg(feature = "recursion")]
            setup_delta_2r: Vec::new(),
            transcript: T::default(),
            _phantom: PhantomData,
        }
    }
}

impl<G1Arg, G2Arg, GTArg, ScalarArg, T> ProofBuilder
    for DoryProofBuilder<G1Arg, G2Arg, GTArg, ScalarArg, T>
where
    G1Arg: Group<Scalar = ScalarArg>,
    G2Arg: Group<Scalar = ScalarArg>,
    GTArg: Group<Scalar = ScalarArg> + std::fmt::Debug,
    ScalarArg: Field,
    T: Transcript<Scalar = ScalarArg>,
{
    type G1 = G1Arg;
    type G2 = G2Arg;
    type GT = GTArg;
    type Scalar = ScalarArg;
    type Pairing = ();

    fn append_first_reduce_message(
        mut self,
        message: FirstReduceMessage<Self::G1, Self::G2, Self::GT>,
    ) -> (FirstReduceChallenge<Self::Scalar>, Self) {
        self.transcript.append_group(b"d1_left", &message.d1_left);
        self.transcript.append_group(b"d1_right", &message.d1_right);
        self.transcript.append_group(b"d2_left", &message.d2_left);
        self.transcript.append_group(b"d2_right", &message.d2_right);
        self.transcript.append_group(b"e1_beta", &message.e1_beta);
        self.transcript.append_group(b"e2_beta", &message.e2_beta);

        let beta = self.transcript.challenge_scalar(b"first_reduce_beta");
        let beta_inverse = beta.inv().expect("Inverse for beta must exist");
        let challenge = FirstReduceChallenge { beta, beta_inverse };

        self.first_messages.push(message);
        #[cfg(feature = "recursion")]
        self.first_challenges.push(challenge.clone());
        (challenge, self)
    }

    fn append_second_reduce_message(
        mut self,
        message: SecondReduceMessage<Self::G1, Self::G2, Self::GT>,
    ) -> (SecondReduceChallenge<Self::Scalar>, Self) {
        self.transcript.append_group(b"c_plus", &message.c_plus);
        self.transcript.append_group(b"c_minus", &message.c_minus);
        self.transcript.append_group(b"e1_plus", &message.e1_plus);
        self.transcript.append_group(b"e1_minus", &message.e1_minus);
        self.transcript.append_group(b"e2_plus", &message.e2_plus);
        self.transcript.append_group(b"e2_minus", &message.e2_minus);

        let alpha = self.transcript.challenge_scalar(b"second_reduce_alpha");
        let alpha_inverse = alpha.inv().expect("Inverse for alpha must exist");
        let challenge = SecondReduceChallenge {
            alpha,
            alpha_inverse,
        };

        #[cfg(feature = "recursion")]
        self.second_challenges.push(challenge.clone());

        // GT operation tracking will be done in finalize_for_recursion
        // to ensure correct ordering

        self.second_messages.push(message);
        (challenge, self)
    }

    fn append_scalar_product_message(
        mut self,
        message: ScalarProductMessage<Self::G1, Self::G2>,
    ) -> Self {
        self.transcript.append_group(b"e1", &message.e1);
        self.transcript.append_group(b"e2", &message.e2);
        self.final_message = Some(message);
        self
    }

    fn append_vmv_message(mut self, message: VMVMessage<Self::G1, Self::GT>) -> Self {
        self.transcript.append_group(b"c_eval_vmv", &message.c);
        self.transcript.append_group(b"d2_eval_vmv", &message.d2);
        self.transcript.append_group(b"e1_eval_vmv", &message.e1);
        self.vmv_message = Some(message);
        self
    }

    fn challenge_fold_scalars(mut self) -> (FoldScalarsChallenge<Self::Scalar>, Self) {
        let gamma = self.transcript.challenge_scalar(b"fold_scalars_gamma");
        let gamma_inverse = gamma.inv().expect("Inverse for gamma must exist");
        let challenge: FoldScalarsChallenge<ScalarArg> = FoldScalarsChallenge {
            gamma,
            gamma_inverse,
        };
        (challenge, self)
    }

    fn challenge_scalar_product_scalars(mut self) -> (ScalarProductChallenge<Self::Scalar>, Self) {
        let d = self.transcript.challenge_scalar(b"scalar_product_d");
        let d_inv = d.inv().unwrap();
        let challenge = ScalarProductChallenge {
            d,
            d_inverse: d_inv,
        };
        (challenge, self)
    }

    #[cfg(feature = "recursion")]
    fn finalize_for_recursion<E>(
        mut self,
        setup: &crate::setup::ProverSetup<E>,
        initial_nu: usize,
    ) -> Self
    where
        E: crate::arithmetic::Pairing<GT = Self::GT>,
        Self::GT: crate::arithmetic::Group + Clone,
    {
        // Call the actual implementation method on DoryProofBuilder
        // This is the non-trait method defined below
        DoryProofBuilder::finalize_for_recursion(&mut self, setup, initial_nu);
        self
    }
}

/// Verification analogue of `ProofBuilder`.
pub trait VerificationBuilder {
    /// G1
    type G1: Group;
    /// G2
    type G2: Group;
    /// GT
    type GT: Group;
    /// F_r
    type Scalar: Field;

    /// Number of rounds (nu)
    fn rounds(&mut self) -> usize;

    /// Returns the messages for round[idx]
    fn take_round(
        &mut self,
        idx: usize,
    ) -> (
        FirstReduceMessage<Self::G1, Self::G2, Self::GT>,
        SecondReduceMessage<Self::G1, Self::G2, Self::GT>,
    );

    /// Getter for first msg
    fn first_message(&mut self, idx: usize) -> &FirstReduceMessage<Self::G1, Self::G2, Self::GT>;

    /// Getter for second msg
    fn second_message(&mut self, idx: usize) -> &SecondReduceMessage<Self::G1, Self::G2, Self::GT>;

    /// Consume a FirstReduceMessage, append it to the transcript,
    /// and return β, β⁻¹.
    fn process_first_reduce_message(
        &mut self,
        msg: &FirstReduceMessage<Self::G1, Self::G2, Self::GT>,
    ) -> FirstReduceChallenge<Self::Scalar>;

    /// Consume a SecondReduceMessage, append, and return α, α⁻¹.
    fn process_second_reduce_message(
        &mut self,
        msg: &SecondReduceMessage<Self::G1, Self::G2, Self::GT>,
    ) -> SecondReduceChallenge<Self::Scalar>;

    /// Derive γ, γ⁻¹ after all rounds are ingested.
    fn challenge_fold_scalars(&mut self) -> FoldScalarsChallenge<Self::Scalar>;

    /// Derive d, d^-1 after all rounds are ingested.
    fn challenge_scalar_product_scalars(&mut self) -> ScalarProductChallenge<Self::Scalar>;

    /// Provide the final scalar-product message that the prover sent.
    fn process_scalar_product_message(&self) -> &ScalarProductMessage<Self::G1, Self::G2>;

    /// Process a [`VMVMessage`].
    fn process_vmv_message(&mut self) -> VMVMessage<Self::G1, Self::GT>;
}

/// Concrete Dory verify builder
pub struct DoryVerifyBuilder<G1, G2, GT, Scalar, T>
where
    G1: Group<Scalar = Scalar>,
    G2: Group<Scalar = Scalar>,
    GT: Group<Scalar = Scalar>,
    Scalar: Field,
    T: Transcript<Scalar = Scalar>,
{
    transcript: T,
    first_messages: Vec<FirstReduceMessage<G1, G2, GT>>,
    second_messages: Vec<SecondReduceMessage<G1, G2, GT>>,
    scalar_msg: ScalarProductMessage<G1, G2>,
    vmv_msg: Option<VMVMessage<G1, GT>>,

    _phantom: PhantomData<(G1, G2, GT, Scalar)>,
}

impl<G1, G2, GT, Scalar, T> DoryVerifyBuilder<G1, G2, GT, Scalar, T>
where
    G1: Group<Scalar = Scalar>,
    G2: Group<Scalar = Scalar>,
    GT: Group<Scalar = Scalar>,
    Scalar: Field,
    T: Transcript<Scalar = Scalar>,
{
    /// Build from a serializable `DoryProof` and a fresh transcript.
    /// This is useful when you have a serialized proof that you want to verify.
    pub fn new_from_dory_proof(proof: DoryProof<G1, G2, GT>, transcript: T) -> Self {
        // Extract messages from the proof
        let first_messages = proof.first_messages;
        let second_messages = proof.second_messages;
        let scalar_msg = proof
            .final_message
            .expect("DoryProof must have a final (scalar product) message");
        let vmv_msg = proof.vmv_message;

        Self {
            transcript,
            first_messages,
            second_messages,
            scalar_msg,
            vmv_msg,
            _phantom: PhantomData,
        }
    }

    /// Build from a *proof* (any concrete `DoryProofBuilder`) and a fresh transcript.
    /// The caller is responsible for providing a fresh transcript with the correct domain.
    pub fn new_from_proof(proof: DoryProofBuilder<G1, G2, GT, Scalar, T>, transcript: T) -> Self {
        // destructure
        let DoryProofBuilder {
            first_messages,
            second_messages,
            final_message,
            vmv_message,
            ..
        } = proof;

        let scalar_msg = final_message.expect("proof must contain the scalar-product message");

        Self {
            transcript,
            first_messages,
            second_messages,
            scalar_msg,
            vmv_msg: vmv_message,
            _phantom: PhantomData,
        }
    }
}

impl<G1, G2, GT, Scalar, T> VerificationBuilder for DoryVerifyBuilder<G1, G2, GT, Scalar, T>
where
    G1: Group<Scalar = Scalar>,
    G2: Group<Scalar = Scalar>,
    GT: Group<Scalar = Scalar>,
    Scalar: Field,
    T: Transcript<Scalar = Scalar>,
{
    type G1 = G1;
    type G2 = G2;
    type GT = GT;
    type Scalar = Scalar;

    fn rounds(&mut self) -> usize {
        self.first_messages.len()
    }

    fn take_round(
        &mut self,
        idx: usize,
    ) -> (
        FirstReduceMessage<G1, G2, GT>,
        SecondReduceMessage<G1, G2, GT>,
    ) {
        let m1 = self.first_messages[idx].clone();
        let m2 = self.second_messages[idx].clone();
        (m1, m2)
    }

    fn first_message(&mut self, idx: usize) -> &FirstReduceMessage<G1, G2, GT> {
        &self.first_messages[idx]
    }
    fn second_message(&mut self, idx: usize) -> &SecondReduceMessage<G1, G2, GT> {
        &self.second_messages[idx]
    }

    fn process_first_reduce_message(
        &mut self,
        m: &FirstReduceMessage<G1, G2, GT>,
    ) -> FirstReduceChallenge<Scalar> {
        self.transcript.append_group(b"d1_left", &m.d1_left);
        self.transcript.append_group(b"d1_right", &m.d1_right);
        self.transcript.append_group(b"d2_left", &m.d2_left);
        self.transcript.append_group(b"d2_right", &m.d2_right);
        self.transcript.append_group(b"e1_beta", &m.e1_beta);
        self.transcript.append_group(b"e2_beta", &m.e2_beta);

        let beta = self.transcript.challenge_scalar(b"first_reduce_beta");
        let beta_inv = beta.inv().unwrap();
        FirstReduceChallenge {
            beta,
            beta_inverse: beta_inv,
        }
    }

    fn process_second_reduce_message(
        &mut self,
        m: &SecondReduceMessage<G1, G2, GT>,
    ) -> SecondReduceChallenge<Scalar> {
        self.transcript.append_group(b"c_plus", &m.c_plus);
        self.transcript.append_group(b"c_minus", &m.c_minus);
        self.transcript.append_group(b"e1_plus", &m.e1_plus);
        self.transcript.append_group(b"e1_minus", &m.e1_minus);
        self.transcript.append_group(b"e2_plus", &m.e2_plus);
        self.transcript.append_group(b"e2_minus", &m.e2_minus);

        let alpha = self.transcript.challenge_scalar(b"second_reduce_alpha");
        let alpha_inv = alpha.inv().unwrap();
        SecondReduceChallenge {
            alpha,
            alpha_inverse: alpha_inv,
        }
    }

    fn challenge_fold_scalars(&mut self) -> FoldScalarsChallenge<Scalar> {
        let gamma = self.transcript.challenge_scalar(b"fold_scalars_gamma");
        let gamma_inv = gamma.inv().unwrap();
        FoldScalarsChallenge {
            gamma,
            gamma_inverse: gamma_inv,
        }
    }

    fn challenge_scalar_product_scalars(&mut self) -> ScalarProductChallenge<Self::Scalar> {
        let d = self.transcript.challenge_scalar(b"scalar_product_d");
        let d_inv = d.inv().unwrap();
        ScalarProductChallenge {
            d,
            d_inverse: d_inv,
        }
    }

    fn process_scalar_product_message(&self) -> &ScalarProductMessage<G1, G2> {
        &self.scalar_msg
    }

    fn process_vmv_message(&mut self) -> VMVMessage<G1, GT> {
        let message = self
            .vmv_msg
            .as_ref()
            .expect("VMV message must be present in verify builder");
        self.transcript.append_group(b"c_eval_vmv", &message.c);
        self.transcript.append_group(b"d2_eval_vmv", &message.d2);
        self.transcript.append_group(b"e1_eval_vmv", &message.e1);
        message.clone()
    }
}

/// Additional debug helpers:
impl<G1, G2, GT, Scalar, T> DoryProofBuilder<G1, G2, GT, Scalar, T>
where
    G1: Group<Scalar = Scalar>,
    G2: Group<Scalar = Scalar>,
    GT: Group<Scalar = Scalar>,
    Scalar: Field,
    T: Transcript<Scalar = Scalar>,
{
    /// Finalize the proof for recursion by computing GT exponentiation steps
    /// This must be called after all rounds are complete but before building the proof
    #[cfg(feature = "recursion")]
    pub fn finalize_for_recursion<E>(
        &mut self,
        setup: &crate::setup::ProverSetup<E>,
        initial_nu: usize,
    ) where
        E: crate::arithmetic::Pairing<GT = GT>,
        GT: crate::arithmetic::Group + Clone,
    {
        // Clear any existing steps
        self.gt_exponentiation_steps.clear();

        // Check if we have setup values
        if self.setup_delta_1l.is_empty() {
            println!("WARNING: No setup delta values available for recursion");
            return;
        }

        let num_rounds = self.first_messages.len();
        if num_rounds != self.second_messages.len()
            || num_rounds != self.first_challenges.len()
            || num_rounds != self.second_challenges.len()
        {
            println!("WARNING: Message/challenge count mismatch");
            return;
        }

        let mut nu = initial_nu;
        println!(
            "DEBUG: finalize_for_recursion starting with initial_nu={}, num_rounds={}",
            initial_nu, num_rounds
        );

        // Process each round in the exact order the verifier will
        for round_idx in 0..num_rounds {
            let first_msg = &self.first_messages[round_idx];
            let second_msg = &self.second_messages[round_idx];
            let beta = self.first_challenges[round_idx].beta.clone();
            let beta_inv = self.first_challenges[round_idx].beta_inverse.clone();
            let alpha = self.second_challenges[round_idx].alpha.clone();
            let alpha_inv = self.second_challenges[round_idx].alpha_inverse.clone();

            println!("DEBUG: Processing round {} with nu={}", round_idx, nu);

            // 1. Operations from dory_reduce_verify_update_c
            let (_, steps_c_plus) = second_msg.c_plus.scale_with_steps(&alpha);
            self.gt_exponentiation_steps.push(steps_c_plus);

            let (_, steps_c_minus) = second_msg.c_minus.scale_with_steps(&alpha_inv);
            self.gt_exponentiation_steps.push(steps_c_minus);

            // 2. Operations from dory_reduce_verify_update_ds
            // The verifier does D1 operations (including deltas) first, then D2 operations

            // D1 operations:
            // d1_left.scale(&alpha)
            let (_, steps_d1l) = first_msg.d1_left.scale_with_steps(&alpha);
            self.gt_exponentiation_steps.push(steps_d1l);

            // Then the delta operations for D1 using the current nu value
            if nu < self.setup_delta_1l.len() {
                // delta_1l.scale(&alpha_beta)
                let alpha_beta = alpha.mul(&beta);
                let (_, steps) = self.setup_delta_1l[nu].scale_with_steps(&alpha_beta);
                self.gt_exponentiation_steps.push(steps);

                // delta_1r.scale(&beta)
                let (_, steps) = self.setup_delta_1r[nu].scale_with_steps(&beta);
                self.gt_exponentiation_steps.push(steps);
            }

            // D2 operations:
            // d2_left.scale(&alpha_inv)
            let (_, steps_d2l) = first_msg.d2_left.scale_with_steps(&alpha_inv);
            self.gt_exponentiation_steps.push(steps_d2l);

            // Then the delta operations for D2
            if nu < self.setup_delta_2l.len() {
                // delta_2l.scale(&alpha_inv_beta_inv)
                let alpha_inv_beta_inv = alpha_inv.mul(&beta_inv);
                let (_, steps) = self.setup_delta_2l[nu].scale_with_steps(&alpha_inv_beta_inv);
                self.gt_exponentiation_steps.push(steps);

                // delta_2r.scale(&beta_inv)
                let (_, steps) = self.setup_delta_2r[nu].scale_with_steps(&beta_inv);
                self.gt_exponentiation_steps.push(steps);
            } else if nu >= self.setup_delta_1l.len() {
                println!(
                    "WARNING: nu={} >= setup_delta_1l.len()={}",
                    nu,
                    self.setup_delta_1l.len()
                );
            }

            // Decrement nu as the verifier does after each round
            nu = nu.saturating_sub(1);
        }

        println!(
            "DEBUG: finalize_for_recursion complete, tracked {} GT operations",
            self.gt_exponentiation_steps.len()
        );
    }

    /// Print statistics about the proof structure
    pub fn print_proof_stats(&self) {
        println!("\n=== PROOF STATISTICS ===");
        println!("Number of rounds: {}", self.first_messages.len());
        println!("First reduce messages: {}", self.first_messages.len());
        println!("Second reduce messages: {}", self.second_messages.len());
        println!("Has final message: {}", self.final_message.is_some());
        println!("Has VMV message: {}", self.vmv_message.is_some());

        // Calculate total proof elements
        let total_g1_elements = self.first_messages.iter().map(|_m| 1).sum::<usize>() + // e1_beta per round
                               self.second_messages.iter().map(|_m| 2).sum::<usize>() + // e1_plus + e1_minus per round
                               if self.final_message.is_some() { 1 } else { 0 } + // final e1
                               if self.vmv_message.is_some() { 1 } else { 0 }; // vmv e1

        let total_g2_elements = self.first_messages.iter().map(|_m| 1).sum::<usize>() + // e2_beta per round
                               self.second_messages.iter().map(|_m| 2).sum::<usize>() + // e2_plus + e2_minus per round
                               if self.final_message.is_some() { 1 } else { 0 }; // final e2

        let total_gt_elements = self.first_messages.iter().map(|_m| 4).sum::<usize>() + // d1_left/right + d2_left/right per round
                               self.second_messages.iter().map(|_m| 2).sum::<usize>() + // c_plus + c_minus per round
                               if self.vmv_message.is_some() { 2 } else { 0 }; // vmv c + d2

        println!("Total G1 elements in proof: {}", total_g1_elements);
        println!("Total G2 elements in proof: {}", total_g2_elements);
        println!("Total GT elements in proof: {}", total_gt_elements);
        println!(
            "Total proof elements: {}",
            total_g1_elements + total_g2_elements + total_gt_elements
        );
        println!("========================\n");
    }
}
