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
) -> (E::GT, Vec<G1>) {
    let num_columns = 1 << sigma;

    let rows_offset = offset / num_columns; // Row start position

    // TODO(moodlezoup): handle offset
    let row_len = num_columns;
    let row_commitments = poly.commit_rows::<M1>(&prover_setup.g1_vec()[..row_len], row_len);

    // --- TIER 2: Multi-pairing to combine row commitments ---

    // Use cached multi-pairing if G2 cache is available, otherwise fall back to regular multi-pairing
    let commitment = if prover_setup.g2_cache.is_some() {
        // Use cached G2 values from prover setup
        E::multi_pair_cached(
            Some(&row_commitments),
            None,
            None, // G1: use runtime points row_commitments
            None,
            Some(row_commitments.len()),
            prover_setup.g2_cache.as_ref(), // G2: use cached elements from rows_offset
        )
    } else {
        // Fall back to regular multi-pairing
        let g2_elements = &prover_setup.g2_vec()[rows_offset..rows_offset + row_commitments.len()];
        E::multi_pair(&row_commitments, g2_elements)
    };

    // Return `row_commitments` because they will come in handy for the opening proof
    (commitment, row_commitments)
}

/// Compute the size split (ν, σ) from variable count and chosen σ
/// 2^ν is the number of rows; 2^σ the number of columns
pub fn compute_nu(num_vars: usize, sigma: usize) -> usize {
    // Enforce symmetric split: sigma = ceil(d/2), nu = floor(d/2)
    let d = num_vars;
    let enforced_sigma = (d + 1) / 2; // ceil(d/2)
    let enforced_nu = d / 2; // floor(d/2)
    debug_assert_eq!(sigma, enforced_sigma, "sigma must equal ceil(d/2)");
    enforced_nu
}

/// Compute the (Pedersen) commitments to the rows of the matrix M that is derived from coeffs `a`.
/// This produces T` in the paper.
pub fn commit_to_rows<E, M1, P>(
  polynomial: &P,
  sigma: usize,
  nu: usize,
  prover_setup: &ProverSetup<E>,
) -> Vec<E::G1>
where
  E: Pairing,
  M1: MultiScalarMul<E::G1>,
  P: Polynomial<<E::G1 as Group>::Scalar, E::G1> + ?Sized,
  E::G1: Group,
  <E::G1 as Group>::Scalar: Field + Clone,
{
  // Use Γ₁[σ] as bases to commit each row of length 2^σ
  debug_assert!(prover_setup.g1_vec().len() >= (1usize << sigma), "Γ1 length < 2^σ");
  let bases = &prover_setup.g1_vec()[..1 << sigma];
  let row_len = 1 << sigma;

  let mut res = polynomial.commit_rows::<M1>(bases, row_len);

  // Pad with identity elements if needed
  while res.len() < (1 << nu) {
      res.push(E::G1::identity());
  }

  res
}