//! Create / Verify a Dory evaluation proof
use crate::{
    arithmetic::{Field, Group, MultiScalarMul, Pairing},
    builder::{DoryProofBuilder, DoryVerifyBuilder},
    error::DoryError,
    core::{commit_to_rows},
    core::evaluate::{build_dory_prover_state_from_poly, eval_vmv_re_prove, eval_vmv_re_verify},
    inner_product::inner_product_prove,
    inner_product::inner_product_verify,
    poly::Polynomial,
    setup::{ProverSetup, VerifierSetup},
    transcript::Transcript,
};

/// Create a new Dory evaluation proof
pub fn create_evaluation_proof<
  E: Pairing,
  T: Transcript<Scalar = <E::G1 as Group>::Scalar>,
  M1: MultiScalarMul<E::G1>,
  M2: MultiScalarMul<E::G2>,
  P: Polynomial<<E::G1 as Group>::Scalar, E::G1> + ?Sized + Sync,
>(
  initial_transcript: T, // DoryProofBuilder takes ownership of the transcript
  polynomial: &P,
  row_commitments: Option<Vec<E::G1>>,
  point: &[<E::G1 as Group>::Scalar],
  prover_setup: &ProverSetup<E>,
) -> DoryProofBuilder<E::G1, E::G2, E::GT, <E::G1 as Group>::Scalar, T>
where
  E::G1: Group,
  E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
  E::GT: Group<Scalar = <E::G1 as Group>::Scalar>,
  <E::G1 as Group>::Scalar: Field,
{
  // 1. Set σ = (d + 1) / 2 and ν = d - σ
  let d = point.len();
  let sigma = (d + 1) / 2;
  let nu = d - sigma;
  debug_assert!(prover_setup.g1_vec().len() >= (1usize << sigma), "Γ1 length < 2^σ");
  debug_assert!(prover_setup.g2_vec().len() >= (1usize << nu), "Γ2 length < 2^ν");

  // 2. Compute row commits (T` in the paper?)
  let t_vec_prime = row_commitments
      .unwrap_or_else(|| commit_to_rows::<E, M1, P>(polynomial, sigma, nu, prover_setup));

  // 3. Build prover state directly
  let (v_vec, prover_state) = build_dory_prover_state_from_poly::<E, P>(polynomial, point, t_vec_prime);

  // 4. Initialize the DoryProofBuilder
  let proof_builder = DoryProofBuilder::new(initial_transcript);

  // 5. Initial commitments
  let (final_proof_builder, proof_state) =
      eval_vmv_re_prove::<E, T, M1, M2>(proof_builder, prover_state, &v_vec, prover_setup);

  // prove!
  let builder_after_ip = inner_product_prove::<_, _, _, _, _, _, _, M1, M2>(
      final_proof_builder,
      proof_state,
      prover_setup,
      nu,
  );

  builder_after_ip
}

/// Verify a dory evaluation proof
pub fn verify_evaluation_proof<
  E: Pairing,
  T: Transcript<Scalar = <E::G1 as Group>::Scalar>,
  M1: MultiScalarMul<E::G1>,
  M2: MultiScalarMul<E::G2>,
  MGT: MultiScalarMul<E::GT>,
>(
  proof: DoryProofBuilder<E::G1, E::G2, E::GT, <E::G1 as Group>::Scalar, T>,
  commitment_batch: &[E::GT],
  batching_factors: &[<E::G1 as Group>::Scalar],
  evaluations: &[<E::G1 as Group>::Scalar],
  b_points: &[<E::G1 as Group>::Scalar],
  verifier_setup: &VerifierSetup<E>,
  transcript: T,
) -> Result<(), DoryError>
where
  E::G1: Group,
  E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
  E::GT: Group<Scalar = <E::G1 as Group>::Scalar>,
  <E::G1 as Group>::Scalar: Field,
{
  // 1. Compute the MSM of commits and the factors
  let a_commit = MGT::msm(commitment_batch, batching_factors);

  // 2. Compute the product of evaluations and batching factors (batching factors should be 1)
  let product: <E::G1 as Group>::Scalar = evaluations
      .iter()
      .zip(batching_factors)
      .fold(<E::G1 as Group>::Scalar::zero(), |acc, (&e, &f)| {
          acc.add(&e.mul(&f))
      });

  // 3. Set σ = (d + 1) / 2 and ν = d - σ
  let d = b_points.len();
  let sigma = (d + 1) / 2;
  let nu = d - sigma;

  // 4. Create verifier builder from proof
  let verify_builder = DoryVerifyBuilder::new_from_proof(proof, transcript);

  // 5. Eval VMV re verifier side (construct verifier state directly)
  let (verify_builder, verifier_state) = eval_vmv_re_verify::<E, T, M1>(
      verify_builder,
      product,
      b_points,
      a_commit,
      verifier_setup,
  );

  // 6. Dory inner product verify
  inner_product_verify(verify_builder, verifier_state, verifier_setup, nu)
      .map_err(|_| DoryError::InvalidProof)
}
