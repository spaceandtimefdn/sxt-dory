//! (multilinear) polynomial utlities
use crate::arithmetic::{Field, Group, MultiScalarMul};

/// multilinear polynomials trait for custom (optimized) primitive operations
/// We provide generic implementations as well
pub trait Polynomial<F: Field, G1: Group<Scalar = F>> {
    /// Get a coefficient at the given index, returns zero if out of bounds
    fn get(&self, index: usize) -> F;

    /// Returns the number of coefficients in the polynomial
    fn len(&self) -> usize;

    /// Evaluates the polynomial at a given point
    fn evaluate(&self, point: &[F]) -> F {
        let len = self.len();
        let mut eval_vec: Vec<F> = vec![F::zero(); len];

        let expected_size = 1 << point.len();
        assert!(
            len <= expected_size,
            "Too many coefficients: got {}, max for {} variables is {}",
            len,
            point.len(),
            expected_size
        );

        multilinear_lagrange_vec(&mut eval_vec, point);

        // Compute inner product <coeffs, eval_vec>
        let mut result = F::zero();
        for (i, eval) in eval_vec.iter().enumerate() {
            let coeff = self.get(i);
            result = result.add(&coeff.mul(eval));
        }
        result
    }

    /// Commits to rows of the polynomial when viewed as a matrix
    fn commit_rows<M1: MultiScalarMul<G1>>(&self, g1_generators: &[G1], row_len: usize) -> Vec<G1> {
        let mut commitments = Vec::new();
        let len = self.len();

        let num_rows = (len + row_len - 1) / row_len;
        for row in 0..num_rows {
            let row_start = row * row_len;
            let row_end = (row_start + row_len).min(len);
            let actual_row_len = row_end - row_start;

            if actual_row_len > 0 {
                let mut row_coeffs = vec![F::zero(); actual_row_len];
                for i in 0..actual_row_len {
                    row_coeffs[i] = self.get(row_start + i);
                }
                let commitment = M1::msm(&g1_generators[..actual_row_len], &row_coeffs);
                commitments.push(commitment);
            }
        }

        commitments
    }

    /// Computes the vector-matrix product v = L^T * M where M is the polynomial as a matrix
    ///
    /// # Arguments
    /// * `left_vec` - The L vector (row evaluation weights)
    /// * `sigma` - log₂(columns) - matrix width
    /// * `nu` - log₂(rows) - matrix height
    ///
    /// # Returns
    /// Result vector v where v[j] = sum_i L[i] * M[i,j]
    #[tracing::instrument(skip_all)]
    fn vector_matrix_product(&self, left_vec: &[F], sigma: usize, nu: usize) -> Vec<F>
    where
        Self: Sync,
    {
        use rayon::prelude::*;

        let cols_per_row = 1 << sigma;
        let len = self.len();
        let num_rows = (1 << nu).min(left_vec.len());

        if num_rows == 0 {
            return vec![F::zero(); cols_per_row];
        }

        let effective_rows: Vec<(usize, &F)> = (0..num_rows)
            .filter_map(|row_idx| {
                let weight = &left_vec[row_idx];
                if !weight.is_zero() {
                    Some((row_idx, weight))
                } else {
                    None
                }
            })
            .collect();

        if effective_rows.is_empty() {
            return vec![F::zero(); cols_per_row];
        }

        (0..cols_per_row)
            .into_par_iter()
            .map(|col_idx| {
                let mut col_sum = F::zero();

                // Process all contributing rows for this column
                for &(row_idx, l_weight) in &effective_rows {
                    let coeff_idx = row_idx * cols_per_row + col_idx;
                    if coeff_idx < len {
                        let coeff = self.get(coeff_idx);
                        let product = l_weight.mul(&coeff);
                        col_sum = col_sum.add(&product);
                    }
                }

                col_sum
            })
            .collect()
    }
}

/// Compute the evaluation of a multilinear polynomial at a given point
/// Uses the lagrange evaluation basis
/// Ref: Section 2.5 of Dory paper.
pub fn compute_polynomial_evaluation<F, G1, P>(poly: &P, point: &[F]) -> F
where
    F: Field,
    G1: Group<Scalar = F>,
    P: Polynomial<F, G1> + ?Sized,
{
    poly.evaluate(point)
}

/// Computes the evaluation vector for a multilinear polynomial at a given point.
///
/// The evaluation vector contains the values of all 2^n multilinear Lagrange basis functions
/// evaluated at the given point. These basis functions are products of the form:
/// (1-x₁)^b₁ * x₁^(1-b₁) * (1-x₂)^b₂ * x₂^(1-b₂) * ... where each bᵢ ∈ {0,1}
///
/// To evaluate a multilinear polynomial with coefficients `coeffs` at `point`:
/// result = coeffs · evaluation_vector
pub fn multilinear_lagrange_vec<F>(v: &mut [F], point: &[F])
where
    F: Field,
{
    assert!(
        v.len() <= (1 << point.len()),
        "Vector length must be at most 2^point.len()"
    );

    // empty point means constant polynomial (all basis functions = 1)
    if point.is_empty() || v.is_empty() {
        v.fill(F::one());
        return;
    }

    // Initialize for first variable: basis functions [1-x₀, x₀]
    let one_minus_p0 = F::one().sub(&point[0]);
    v[0] = one_minus_p0;
    if v.len() > 1 {
        v[1] = point[0];
    }

    // For each subsequent variable, double the active portion of the evaluation vector
    // by splitting each existing value into (value * (1-pᵢ)) and (value * pᵢ)
    for (level, p) in point[1..].iter().enumerate() {
        let mid = 1 << (level + 1); // Size of active portion after previous variables

        // Apply the transformation: right[i] = left[i] * p, left[i] = left[i] * (1-p)
        let one_minus_p = F::one().sub(p);

        if mid >= v.len() {
            // No right portion if we've filled the vector, just multiply all by (1-p)
            for li in v.iter_mut() {
                *li = li.mul(&one_minus_p);
            }
        } else {
            // We can split the vector:
            let (left, right) = v.split_at_mut(mid);
            let k = left.len().min(right.len());

            // Transform paired elements
            for (li, ri) in left[..k].iter_mut().zip(right[..k].iter_mut()) {
                let li_val = *li;
                *ri = li_val.mul(p);
                *li = li_val.mul(&one_minus_p);
            }

            // Handle remaining left elements (when left is longer than right)
            for li in left[k..].iter_mut() {
                *li = li.mul(&one_minus_p);
            }
        }
    }
}

/// Compute vectors L and R for matrix-based polynomial evaluation
/// Given a polynomial arranged as a matrix M, computes L and R such that:
/// polynomial_evaluation(b_point) = L^T × M × R
#[tracing::instrument(skip_all)]
pub fn compute_left_right_vec<F: Field>(
    b_point: &[F],
    sigma: usize, // log₂(max_columns) - matrix width
    nu: usize,    // log₂(vector_length) - matrix length
) -> (Vec<F>, Vec<F>) {
    let mut right_vec = vec![F::zero(); 1 << nu]; // Column evaluation vector
    let mut left_vec = vec![F::zero(); 1 << nu]; // Row evaluation vector
    let point_dim = b_point.len();

    match point_dim {
        // Case 1: Constant polynomial (0 variables)
        0 => {
            right_vec[0] = F::one();
            left_vec[0] = F::one();
            // Matrix is 1×1, so L^T × M × R = 1 × M[0,0] × 1
        }

        // Case 2: All variables fit in columns (single row needed)
        n if n <= sigma => {
            // All variables determine column position
            multilinear_lagrange_vec(&mut right_vec[..1 << point_dim], b_point);
            left_vec[0] = F::one(); // Only need first row
                                    // L^T × M × R = [1, 0, ...] × M × R
        }

        // Case 3: Variables split between rows and columns (no padding)
        n if n <= sigma * 2 => {
            // Split variables: first `nu` for columns, rest for rows
            multilinear_lagrange_vec(&mut right_vec, &b_point[..nu]); // Column vars
            multilinear_lagrange_vec(&mut left_vec[..1 << (point_dim - nu)], &b_point[nu..]);
            // Row vars
            // L^T × M × R where both L and R have meaningful entries
        }

        // Case 4: Too many variables - need column padding
        _ => {
            // Use max column capacity, put remaining variables in rows
            multilinear_lagrange_vec(&mut right_vec[..(1 << sigma)], &b_point[..sigma]); // First σ vars → columns
            multilinear_lagrange_vec(&mut left_vec, &b_point[sigma..]); // Remaining vars → rows
                                                                        // Matrix has padded columns but we only use the first 2^σ columns
        }
    }

    (left_vec, right_vec)
}
