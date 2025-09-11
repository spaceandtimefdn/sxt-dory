//! Contains the utility required to turn Dory arguments into a full-fledged PCS
//! Primarily makes use of the `eval_vmv_re` protocol

use crate::{
  arithmetic::{Field, Group, MultiScalarMul, Pairing},
  builder::{DoryProofBuilder, DoryVerifyBuilder, VerificationBuilder},
  error::DoryError,
  inner_product::inner_product_verify,
  inner_product_prove,
  messages::VMVMessage,
  poly::Polynomial,
  setup::{ProverSetup, VerifierSetup},
  state::{DoryProverState, DoryVerifierState},
  transcript::Transcript,
  core::commit_to_rows,
  ProofBuilder,
};
// use ark_serialize::CanonicalSerialize;

/// Implements the Eval-VMV-RE protocol from Dory Section 5
/// Proves the VMV relation: polynomial(point) = L^T × M × R
///
/// Note: Randomness terms (rD2, rE1) are omitted since we don't need hiding (yet)
#[tracing::instrument(skip_all)]
fn eval_vmv_re_prove<
  E: Pairing,
  T: Transcript<Scalar = <E::G1 as Group>::Scalar>,
  M1: MultiScalarMul<E::G1>,
  M2: MultiScalarMul<E::G2>,
>(
  mut proof_builder: DoryProofBuilder<E::G1, E::G2, E::GT, <E::G1 as Group>::Scalar, T>,
  mut prover_state: DoryProverState<E>,
  v_vec: &[<E::G1 as Group>::Scalar],
  prover_setup: &ProverSetup<E>,
) -> (
  DoryProofBuilder<E::G1, E::G2, E::GT, <E::G1 as Group>::Scalar, T>,
  DoryProverState<E>,
)
where
  E::G1: Group,
  E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
  E::GT: Group<Scalar = <E::G1 as Group>::Scalar>,
  <E::G1 as Group>::Scalar: Field,
{
  // Validate inputs
  if prover_state.v1.is_empty() || prover_state.s1.is_empty() {
      println!("v1 or s1 is empty in eval_vmv_re_prove");
  }
  if prover_state.nu > 0 && prover_setup.g1_vec().len() < (1 << prover_state.nu) {
      println!("prover_setup.g1_vec doesn't have enough elements for nu");
  }

  // --- Protocol computations ---

  // D₂ = e(⟨Γ₁[nu], ~v⟩, Γ₂,fin)
  // Protocol: D₂ = e(⟨Γ₁,~v⟩, Γ₂,fin) + rD₂·HT (randomness omitted)
  // Slice Γ₁ by the width of v_vec (2^σ), not by ν. This is robust when σ ≠ ν.
  let g1_bases_for_sigma = if v_vec.is_empty() || prover_setup.g1_vec().len() < v_vec.len() {
      &[][..]
  } else {
      &prover_setup.g1_vec()[..v_vec.len()]
  };

  let gamma1_v_inner_product = if g1_bases_for_sigma.is_empty() {
      E::G1::identity()
  } else {
      M1::msm(g1_bases_for_sigma, v_vec)
  };
  let d2 = E::pair(&gamma1_v_inner_product, prover_setup.g_fin());

  // E₁ = ⟨T~₀, ~L⟩
  // Protocol: E₁ = ⟨~L, C₀⟩ + rE₁·H₁ (randomness omitted)
  if prover_state.s2.is_empty() && !prover_state.v1.is_empty() {
      println!("s2 is empty but v1 is not in E₁ calculation");
  }
  let e1 = M1::msm(&prover_state.v1, &prover_state.s2);

  // Create VMV message for transcript
  let vmv_message = VMVMessage {
      d2,
      e1, // note that e2 is calculated by the verifier here
  };
  proof_builder = proof_builder.append_vmv_message(vmv_message);

  // Transform intermediate vector ~v into G2 elements for next phase
  // v₂ = ~v · Γ₂,fin (scalar multiplication in G2)
  // Use fixed-base vectorized MSM since we're scaling the same base (g_fin) by each scalar
  let updated_v2 = M2::fixed_base_vector_msm(
      prover_setup.g_fin(),
      v_vec,
      prover_setup.g1_cache.as_ref(),
      prover_setup.g2_cache.as_ref(),
  );

  // If ν > σ, expand v2 by repeating each entry 2^(ν-σ) times to reach length 2^ν.
  // This aligns with padding the missing right-dimensions by (1,1) tensors.
  let target_len = 1usize << prover_state.nu;
  let base_len = updated_v2.len();
  debug_assert!(base_len == 0 || base_len.is_power_of_two());
  debug_assert!(target_len.is_power_of_two());
  debug_assert!(base_len == 0 || target_len % base_len == 0, "target 2^ν must be multiple of 2^σ");
  let v2_expanded = if base_len > 0 && target_len > base_len {
      let repeat = target_len / base_len;
      let mut out = Vec::with_capacity(target_len);
      for g in updated_v2.iter() {
          for _ in 0..repeat {
              out.push(g.clone());
          }
      }
      out
  } else {
      updated_v2
  };

  // Use expanded v2 and disable scalar shortcut if expansion occurred to avoid length mismatch
  let expanded = v2_expanded.len() == target_len && base_len != target_len;
  prover_state.v2 = v2_expanded;
  if expanded {
      prover_state.v2_scalars = None;
  }

  (proof_builder, prover_state)
}

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

// VERIFIER ANALOGUE:

/// Verifier analogue of `eval_vmv_re` protocol in the paper
fn eval_vmv_re_verify<
  E: Pairing,
  T: Transcript<Scalar = <E::G1 as Group>::Scalar>,
  M1: MultiScalarMul<E::G1>,
>(
  mut verify_builder: DoryVerifyBuilder<E::G1, E::G2, E::GT, <E::G1 as Group>::Scalar, T>,
  y: <E::G1 as Group>::Scalar,
  b_points: &[<E::G1 as Group>::Scalar],
  t: E::GT,
  verifier_setup: &VerifierSetup<E>,
) -> (
  DoryVerifyBuilder<E::G1, E::G2, E::GT, <E::G1 as Group>::Scalar, T>,
  DoryVerifierState<E>,
)
where
  E::G1: Group,
  E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
  E::GT: Group<Scalar = <E::G1 as Group>::Scalar>,
  <E::G1 as Group>::Scalar: Field,
{
  let vmv_message = verify_builder.process_vmv_message_take();

  // Messages from prover
  let d_2 = vmv_message.d2.clone();
  let e_1 = vmv_message.e1.clone();

  // Construct verifier state directly
  let d_1 = t;
  let e_2 = verifier_setup.g_fin.scale(&y);
  let final_verifier_state = DoryVerifierState::new(d_1, d_2, e_1, e_2, b_points.into());

  // Deferred pairing check: handled in finalize. Keep for reference.
  // let pairing_check = E::pair(&vmv_message.e1, &verifier_setup.g_fin);
  // assert!(
  //     vmv_message.d2 == pairing_check,
  //     "Sigma protocol 2 verification failed: d2 != e(e1, Gamma_{2, fin})"
  // );

  // Return the updated verify builder and unchanged verifier state
  // The verifier state conversion should be handled by the caller
  (verify_builder, final_verifier_state)
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
