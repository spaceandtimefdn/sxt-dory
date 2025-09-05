//! Dory follows an interactive model. Hence, a "proof" consists of some messages
//! between P and V. We use Prover and Verifier "builders" to manage these messages
//! and the fiat-shamir challenges throughout the implementation.
use crate::transcript::Transcript;
use std::marker::PhantomData;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use crate::{
    arithmetic::{Field, Group},
    messages::{
        FirstReduceChallenge, FirstReduceMessage, FinalizeChallenge,
        SecondReduceChallenge, SecondReduceMessage, VMVMessage, FinalBasesMessage,
    },
    toy_transcript::ToyTranscript,
};

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
    pub second_messages: Vec<SecondReduceMessage<G1, G2>>,
    /// Vector-matrix-vector message (for PCS)
    pub vmv_message: Option<VMVMessage<G1, GT>>,
    /// Final base-case group elements
    pub final_bases: Option<FinalBasesMessage<G1, G2>>,
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
        message: SecondReduceMessage<Self::G1, Self::G2>,
    ) -> (SecondReduceChallenge<Self::Scalar>, Self);

    #[must_use]
    /// Append a [`VMVMessage`] to the proof and transcript.
    fn append_vmv_message(self, message: VMVMessage<Self::G1, Self::GT>) -> Self;

    /// Draw a [`FinalizeChallenge`] from the transcript.
    #[must_use]
    fn challenge_finalize(self) -> (FinalizeChallenge<Self::Scalar>, Self);

    /// Append the final base-case group elements.
    #[must_use]
    fn append_final_bases(self, message: FinalBasesMessage<Self::G1, Self::G2>) -> Self;
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
    /// Second prover message for round i
    pub second_messages: Vec<SecondReduceMessage<G1, G2>>,

    /// vector-matrix-vector message, used to transform general dory into PCS
    pub vmv_message: Option<VMVMessage<G1, GT>>,
    /// final base-case group elements
    pub final_bases: Option<FinalBasesMessage<G1, G2>>,
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
    /// Constructor from new transcript
    pub fn new(transcript: T) -> Self {
        Self {
            first_messages: Vec::new(),
            second_messages: Vec::new(),
            vmv_message: None,
            final_bases: None,
            transcript,
            _phantom: PhantomData,
        }
    }

    /// Constructor to create with ToyTranscript for testing
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
            vmv_message: None,
            final_bases: None,
            transcript,
            _phantom: PhantomData,
        }
    }

    /// Build a serializable Dory proof (consumes self to avoid cloning)
    pub fn build(self) -> DoryProof<G1, G2, GT> {
        DoryProof {
            first_messages: self.first_messages,
            second_messages: self.second_messages,
            vmv_message: self.vmv_message,
            final_bases: self.final_bases,
        }
    }

    /// Create a DoryProofBuilder from a DoryProof and a fresh transcript
    pub fn from_proof(proof: DoryProof<G1, G2, GT>, transcript: T) -> Self {
        Self {
            first_messages: proof.first_messages,
            second_messages: proof.second_messages,
            vmv_message: proof.vmv_message,
            final_bases: proof.final_bases,
            transcript,
            _phantom: PhantomData,
        }
    }

    /// Create a DoryProofBuilder from a DoryProof with a default transcript
    pub fn from_proof_no_transcript(proof: DoryProof<G1, G2, GT>) -> Self
    where
        T: Default,
    {
        Self {
            first_messages: proof.first_messages,
            second_messages: proof.second_messages,
            vmv_message: proof.vmv_message,
            final_bases: proof.final_bases,
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
    GTArg: Group<Scalar = ScalarArg>,
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
        let challenge = FirstReduceChallenge { beta };

        self.first_messages.push(message);
        (challenge, self)
    }

    fn append_second_reduce_message(
        mut self,
        message: SecondReduceMessage<Self::G1, Self::G2>,
    ) -> (SecondReduceChallenge<Self::Scalar>, Self) {
        // PCS variant: omit C_+ and C_- from transcript
        self.transcript.append_group(b"e1_plus", &message.e1_plus);
        self.transcript.append_group(b"e1_minus", &message.e1_minus);
        self.transcript.append_group(b"e2_plus", &message.e2_plus);
        self.transcript.append_group(b"e2_minus", &message.e2_minus);

        let alpha = self.transcript.challenge_scalar(b"second_reduce_alpha");
        let challenge = SecondReduceChallenge { alpha };

        self.second_messages.push(message);
        (challenge, self)
    }

    fn append_vmv_message(mut self, message: VMVMessage<Self::G1, Self::GT>) -> Self {
        // PCS variant: remove c from the protocol
        // self.transcript.append_group(b"c_eval_vmv", &message.c);
        self.transcript.append_group(b"d2_eval_vmv", &message.d2);
        self.transcript.append_group(b"e1_eval_vmv", &message.e1);
        self.vmv_message = Some(message);
        self
    }

    fn challenge_finalize(mut self) -> (FinalizeChallenge<Self::Scalar>, Self) {
        let gamma_1 = self.transcript.challenge_scalar(b"finalize_gamma_1");
        let gamma_2 = self.transcript.challenge_scalar(b"finalize_gamma_2");
        let challenge = FinalizeChallenge {
            gamma_1,
            gamma_2,
        };
        (challenge, self)
    }

    fn append_final_bases(mut self, message: FinalBasesMessage<Self::G1, Self::G2>) -> Self {
        // No transcript impact (optional). If desired we could append to transcript too.
        self.final_bases = Some(message);
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
    fn rounds(&self) -> usize;

    /// Returns the messages for round[idx]
    fn take_round(
        &mut self,
        idx: usize,
    ) -> (
        FirstReduceMessage<Self::G1, Self::G2, Self::GT>,
        SecondReduceMessage<Self::G1, Self::G2>,
    );

    /// Getter for first msg
    fn first_message(&self, idx: usize) -> &FirstReduceMessage<Self::G1, Self::G2, Self::GT>;

    /// Getter for second msg
    fn second_message(&self, idx: usize) -> &SecondReduceMessage<Self::G1, Self::G2>;

    /// Consume a FirstReduceMessage, append it to the transcript,
    /// and return β, β⁻¹.
    fn process_first_reduce_message(
        &mut self,
        msg: &FirstReduceMessage<Self::G1, Self::G2, Self::GT>,
    ) -> FirstReduceChallenge<Self::Scalar>;

    /// Append first-reduce message at index to transcript and return β.
    fn process_first_reduce_message_at(
        &mut self,
        idx: usize,
    ) -> FirstReduceChallenge<Self::Scalar>;

    /// Consume a SecondReduceMessage, append, and return α, α⁻¹.
    fn process_second_reduce_message(
        &mut self,
        msg: &SecondReduceMessage<Self::G1, Self::G2>,
    ) -> SecondReduceChallenge<Self::Scalar>;

    /// Append second-reduce message at index to transcript and return α.
    fn process_second_reduce_message_at(
        &mut self,
        idx: usize,
    ) -> SecondReduceChallenge<Self::Scalar>;

    /// Derive gamma_1, gamma_2 after all rounds are ingested.
    fn challenge_finalize(&mut self) -> FinalizeChallenge<Self::Scalar>;

    /// Process a [`VMVMessage`].
    fn process_vmv_message(&mut self) -> VMVMessage<Self::G1, Self::GT>;

    /// Process a [`VMVMessage`] and move it out of the builder without cloning.
    fn process_vmv_message_take(&mut self) -> VMVMessage<Self::G1, Self::GT>;

    /// Process the final base-case group elements from the prover.
    fn process_final_bases(&mut self) -> FinalBasesMessage<Self::G1, Self::G2>;

    /// Move the final base-case group elements out of the builder without cloning.
    fn process_final_bases_take(&mut self) -> FinalBasesMessage<Self::G1, Self::G2>;
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
    second_messages: Vec<SecondReduceMessage<G1, G2>>,
    vmv_msg: Option<VMVMessage<G1, GT>>,
    final_bases: Option<FinalBasesMessage<G1, G2>>,

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
        let vmv_msg = proof.vmv_message;
        let final_bases = proof.final_bases;

        Self {
            transcript,
            first_messages,
            second_messages,
            vmv_msg,
            final_bases,
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
            vmv_message,
            final_bases,
            ..
        } = proof;

        Self {
            transcript,
            first_messages,
            second_messages,
            vmv_msg: vmv_message,
            final_bases,
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

    fn rounds(&self) -> usize {
        self.first_messages.len()
    }

    fn take_round(
        &mut self,
        idx: usize,
    ) -> (
        FirstReduceMessage<G1, G2, GT>,
        SecondReduceMessage<G1, G2>,
    ) {
        let m1 = self.first_messages[idx].clone();
        let m2 = self.second_messages[idx].clone();
        (m1, m2)
    }

    fn first_message(&self, idx: usize) -> &FirstReduceMessage<G1, G2, GT> {
        &self.first_messages[idx]
    }
    fn second_message(&self, idx: usize) -> &SecondReduceMessage<G1, G2> {
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
        FirstReduceChallenge { beta }
    }

    fn process_first_reduce_message_at(
        &mut self,
        idx: usize,
    ) -> FirstReduceChallenge<Scalar> {
        let m = &self.first_messages[idx];
        let transcript = &mut self.transcript;
        transcript.append_group(b"d1_left", &m.d1_left);
        transcript.append_group(b"d1_right", &m.d1_right);
        transcript.append_group(b"d2_left", &m.d2_left);
        transcript.append_group(b"d2_right", &m.d2_right);
        transcript.append_group(b"e1_beta", &m.e1_beta);
        transcript.append_group(b"e2_beta", &m.e2_beta);

        let beta = transcript.challenge_scalar(b"first_reduce_beta");
        FirstReduceChallenge { beta }
    }

    fn process_second_reduce_message(
        &mut self,
        m: &SecondReduceMessage<G1, G2>,
    ) -> SecondReduceChallenge<Scalar> {
        // PCS variant: omit C_+ and C_- from transcript
        self.transcript.append_group(b"e1_plus", &m.e1_plus);
        self.transcript.append_group(b"e1_minus", &m.e1_minus);
        self.transcript.append_group(b"e2_plus", &m.e2_plus);
        self.transcript.append_group(b"e2_minus", &m.e2_minus);

        let alpha = self.transcript.challenge_scalar(b"second_reduce_alpha");
        SecondReduceChallenge { alpha }
    }

    fn process_second_reduce_message_at(
        &mut self,
        idx: usize,
    ) -> SecondReduceChallenge<Scalar> {
        let m = &self.second_messages[idx];
        let transcript = &mut self.transcript;
        // PCS variant: omit C_+ and C_- from transcript
        transcript.append_group(b"e1_plus", &m.e1_plus);
        transcript.append_group(b"e1_minus", &m.e1_minus);
        transcript.append_group(b"e2_plus", &m.e2_plus);
        transcript.append_group(b"e2_minus", &m.e2_minus);

        let alpha = transcript.challenge_scalar(b"second_reduce_alpha");
        SecondReduceChallenge { alpha }
    }

    fn challenge_finalize(&mut self) -> FinalizeChallenge<Self::Scalar> {
        let gamma_1 = self.transcript.challenge_scalar(b"finalize_gamma_1");
        let gamma_2 = self.transcript.challenge_scalar(b"finalize_gamma_2");
        FinalizeChallenge { gamma_1, gamma_2 }
    }

    fn process_vmv_message(&mut self) -> VMVMessage<G1, GT> {
        let message = self
            .vmv_msg
            .as_ref()
            .expect("VMV message must be present in verify builder");
        // PCS variant: remove c from the protocol
        // self.transcript.append_group(b"c_eval_vmv", &message.c);
        self.transcript.append_group(b"d2_eval_vmv", &message.d2);
        self.transcript.append_group(b"e1_eval_vmv", &message.e1);
        message.clone()
    }

    fn process_vmv_message_take(&mut self) -> VMVMessage<G1, GT> {
        let message = self
            .vmv_msg
            .take()
            .expect("VMV message must be present in verify builder");
        // Append to transcript before moving out
        self.transcript.append_group(b"d2_eval_vmv", &message.d2);
        self.transcript.append_group(b"e1_eval_vmv", &message.e1);
        message
    }

    fn process_final_bases(&mut self) -> FinalBasesMessage<G1, G2> {
        self.final_bases
            .as_ref()
            .expect("Final bases must be present in verify builder")
            .clone()
    }

    fn process_final_bases_take(&mut self) -> FinalBasesMessage<G1, G2> {
        self.final_bases
            .take()
            .expect("Final bases must be present in verify builder")
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
    /// Print statistics about the proof structure
    pub fn print_proof_stats(&self) {
        println!("\n=== PROOF STATISTICS ===");
        println!("Number of rounds: {}", self.first_messages.len());
        println!("First reduce messages: {}", self.first_messages.len());
        println!("Second reduce messages: {}", self.second_messages.len());
        println!("Has VMV message: {}", self.vmv_message.is_some());

        // Calculate total proof elements
        let total_g1_elements = self.first_messages.iter().map(|_m| 1).sum::<usize>() + // e1_beta per round
                               self.second_messages.iter().map(|_m| 2).sum::<usize>() + // e1_plus + e1_minus per round
                               if self.vmv_message.is_some() { 1 } else { 0 }; // vmv e1

        let total_g2_elements = self.first_messages.iter().map(|_m| 1).sum::<usize>() + // e2_beta per round
                               self.second_messages.iter().map(|_m| 2).sum::<usize>(); // e2_plus + e2_minus per round

        let total_gt_elements = self.first_messages.iter().map(|_m| 4).sum::<usize>() + // d1_left/right + d2_left/right per round
                               self.second_messages.iter().map(|_m| 0).sum::<usize>() + // PCS: no C terms per round
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
