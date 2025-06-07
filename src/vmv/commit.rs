//! Multilinear polynomial commitmnets as a matrix
use crate::arithmetic::{Field, Group, MultiScalarMul, Pairing};
use crate::poly::Polynomial;
use crate::setup::ProverSetup;

/// Dory's 2-tier homomorphic commitment to multilinear polynomial arranged as matrix
/// Tier 1: Row commitments in G1, Tier 2: Multi-pairing to GT
/// See page 12 of the paper.
pub fn compute_polynomial_commitment<
    E: Pairing<G1 = G1>,
    M1: MultiScalarMul<G1>,
    P: Polynomial<F, G1> + ?Sized,
    F: Field,
    G1: Group<Scalar = F>,
>(
    poly: &P,      // Polynomial
    offset: usize, // Starting position in matrix
    sigma: usize,  // log₂(matrix_width)
    prover_setup: &ProverSetup<E>,
) -> E::GT {
    let num_columns = 1 << sigma;

    // Handle arbitrary offset within the matrix
    let first_row_offset = offset % num_columns; // Column start position
    let rows_offset = offset / num_columns; // Row start position
    let _first_row_len = poly.len().min(num_columns - first_row_offset);

    // TODO(moodlezoup): handle offset
    let row_len = num_columns;
    let row_commitments = poly.commit_rows::<M1>(&prover_setup.g1_vec[..row_len], row_len);

    // let (first_row_coeffs, remaining_coeffs) = coeffs.split_at(first_row_len);
    // let remaining_row_count = (remaining_coeffs.len() + num_columns - 1) / num_columns;

    // --- TIER 1: Compute row commitments in G1 ---

    // let first_row_commit = if first_row_len > 0 {
    //     M1::msm(
    //         &prover_setup.g1_vec[first_row_offset..first_row_offset + first_row_len],
    //         first_row_coeffs,
    //     )
    // } else {
    //     E::G1::identity()
    // };

    // let mut g1_row_commitments = Vec::with_capacity(1 + remaining_row_count);
    // g1_row_commitments.push(first_row_commit);

    // // Remaining row commitments (full rows)
    // for row_coeffs in remaining_coeffs.chunks(num_columns) {
    //     let row_commit = M1::msm(&prover_setup.g1_vec[0..row_coeffs.len()], row_coeffs);
    //     g1_row_commitments.push(row_commit);
    // }

    // --- TIER 2: Multi-pairing to combine row commitments ---

    let g2_elements = &prover_setup.g2_vec[rows_offset..rows_offset + row_commitments.len()];
    E::multi_pair(&row_commitments, g2_elements) // Final commitment in GT
}

/// Create commitment batch, batching factors, and evaluations for verification
/// This provides the values needed for verify_evaluation_proof
pub fn commit_and_evaluate_batch<
    E: Pairing<G1 = G1>,
    M1: MultiScalarMul<G1>,
    P: Polynomial<F, G1> + ?Sized,
    F: Field,
    G1: Group<Scalar = F>,
>(
    poly: &P,
    point: &[F],
    offset: usize,
    sigma: usize,
    prover_setup: &ProverSetup<E>,
) -> (
    Vec<E::GT>, // commitment_batch
    Vec<F>,     // batching_factors
    Vec<F>,     // evaluations
)
where
    F: Field + Clone,
{
    // Compute the commitment to the polynomial
    let commitment =
        compute_polynomial_commitment::<E, M1, P, F, G1>(poly, offset, sigma, prover_setup);

    // Compute the evaluation of the polynomial at the point
    let evaluation = poly.evaluate(point);

    // For a single polynomial, we use a single batching factor of 1
    let commitment_batch = vec![commitment];

    // @TODO(markosg04): support batching
    let batching_factors = vec![F::one()];
    let evaluations = vec![evaluation]; // for now just one evaluation

    (commitment_batch, batching_factors, evaluations)
}
