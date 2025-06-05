//! Multilinear polynomial commitmnets as a matrix
use crate::arithmetic::{Field, Group, MultiScalarMul, MultilinearPolynomial, Pairing};
use crate::poly::compute_polynomial_evaluation;
use crate::setup::ProverSetup;

/// Dory's 2-tier homomorphic commitment to multilinear polynomial arranged as matrix
/// Tier 1: Row commitments in G1, Tier 2: Multi-pairing to GT
/// See page 12 of the paper.
pub fn compute_polynomial_commitment<E: Pairing, M1: MultiScalarMul<E::G1>>(
    polynomial: &MultilinearPolynomial<<E::G1 as Group>::Scalar>, // Polynomial
    offset: usize,                       // Starting position in matrix
    sigma: usize,                        // log₂(matrix_width)
    prover_setup: &ProverSetup<E>,
) -> E::GT {
    let num_columns = 1 << sigma;

    // Handle arbitrary offset within the matrix
    let first_row_offset = offset % num_columns; // Column start position
    let rows_offset = offset / num_columns; // Row start position

    // --- TIER 1: Compute row commitments in G1 ---

    let mut g1_row_commitments = Vec::new();
    
    // We need to handle the offset properly. The polynomial is conceptually
    // arranged as a matrix, and offset tells us where to start in this matrix.
    // We iterate through the polynomial in chunks of size num_columns.
    
    let total_elements = polynomial.len();
    let mut element_idx = 0;
    let mut current_row_offset = first_row_offset;
    
    while element_idx < total_elements {
        let row_remaining = num_columns - current_row_offset;
        let elements_to_process = row_remaining.min(total_elements - element_idx);
        
        // Create a polynomial slice for this row segment
        let row_poly = match polynomial {
            MultilinearPolynomial::LargeScalars(coeffs) => {
                MultilinearPolynomial::LargeScalars(&coeffs[element_idx..element_idx + elements_to_process])
            }
            MultilinearPolynomial::U8Scalars(coeffs) => {
                MultilinearPolynomial::U8Scalars(&coeffs[element_idx..element_idx + elements_to_process])
            }
            MultilinearPolynomial::U16Scalars(coeffs) => {
                MultilinearPolynomial::U16Scalars(&coeffs[element_idx..element_idx + elements_to_process])
            }
            MultilinearPolynomial::U32Scalars(coeffs) => {
                MultilinearPolynomial::U32Scalars(&coeffs[element_idx..element_idx + elements_to_process])
            }
            MultilinearPolynomial::U64Scalars(coeffs) => {
                MultilinearPolynomial::U64Scalars(&coeffs[element_idx..element_idx + elements_to_process])
            }
            MultilinearPolynomial::I64Scalars(coeffs) => {
                MultilinearPolynomial::I64Scalars(&coeffs[element_idx..element_idx + elements_to_process])
            }
        };
        
        // Compute MSM for this row segment
        let row_commit = M1::msm(
            &prover_setup.g1_vec[current_row_offset..current_row_offset + elements_to_process],
            &row_poly,
        );
        g1_row_commitments.push(row_commit);
        
        element_idx += elements_to_process;
        current_row_offset = 0; // After first row, we start at column 0
    }

    // --- TIER 2: Multi-pairing to combine row commitments ---

    let g2_elements = &prover_setup.g2_vec[rows_offset..rows_offset + g1_row_commitments.len()];
    E::multi_pair(&g1_row_commitments, g2_elements) // Final commitment in GT
}

/// Create commitment batch, batching factors, and evaluations for verification
/// This provides the values needed for verify_evaluation_proof
pub fn commit_and_evaluate_batch<E: Pairing, M1: MultiScalarMul<E::G1>>(
    polynomial: &MultilinearPolynomial<<E::G1 as Group>::Scalar>,
    point: &[<E::G1 as Group>::Scalar],
    offset: usize,
    sigma: usize,
    prover_setup: &ProverSetup<E>,
) -> (
    Vec<E::GT>,                    // commitment_batch
    Vec<<E::G1 as Group>::Scalar>, // batching_factors
    Vec<<E::G1 as Group>::Scalar>, // evaluations
)
where
    E::G1: Group,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
    <E::G1 as Group>::Scalar: Field + Clone,
{
    // Compute the commitment to the polynomial
    let commitment = compute_polynomial_commitment::<E, M1>(polynomial, offset, sigma, prover_setup);

    // Compute the evaluation of the polynomial at the point
    let evaluation = compute_polynomial_evaluation(polynomial, point);

    // For a single polynomial, we use a single batching factor of 1
    let commitment_batch = vec![commitment];

    // @TODO(markosg04): support batching
    let batching_factors = vec![<E::G1 as Group>::Scalar::one()];
    let evaluations = vec![evaluation]; // for now just one evaluation

    (commitment_batch, batching_factors, evaluations)
}
