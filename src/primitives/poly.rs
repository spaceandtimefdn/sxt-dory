//! (multilinear) polynomial utlities
use crate::arithmetic::{Field, Group, MultiScalarMul};

/// multilinear polynomials trait for custom (optimized) primitive operations
/// We provide generic implementations as well
pub trait Polynomial<F: Field, G1: Group<Scalar = F>> {
    /// Returns the number of coefficients in the polynomial
    fn len(&self) -> usize {
        len_generic(self)
    }

    /// Evaluates the polynomial at a given point
    fn evaluate(&self, point: &[F]) -> F {
        compute_polynomial_evaluation_generic(self, point)
    }

    /// Commits to rows of the polynomial when viewed as a matrix
    fn commit_rows<M1: MultiScalarMul<G1>>(&self, g1_generators: &[G1], row_len: usize) -> Vec<G1> {
        commit_rows_generic::<F, G1, Self, M1>(self, g1_generators, row_len)
    }

    /// Computes the vector-matrix product L^T * M where M is the polynomial as a matrix
    fn vector_matrix_product(&self, l_vec: &[F]) -> Vec<F> {
        vector_matrix_product_generic(self, l_vec)
    }

    /// Gets coefficient at index
    fn get(&self, index: usize) -> Option<F> {
        get_generic(self, index)
    }

    /// Checks if polynomial is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

}

/// Generic length implementation
fn len_generic<F, G1, P>(poly: &P) -> usize
where
    F: Field,
    G1: Group<Scalar = F>,
    P: Polynomial<F, G1> + ?Sized,
{
    // Find length by checking until we hit None
    let mut len = 0;
    while poly.get(len).is_some() {
        len += 1;
    }
    len
}

/// Generic get implementation
fn get_generic<F, G1, P>(poly: &P, index: usize) -> Option<F>
where
    F: Field + Clone,
    G1: Group<Scalar = F>,
    P: Polynomial<F, G1> + ?Sized,
{
    // This is now just a placeholder since implementations should override this
    // Default behavior: return None (empty polynomial)
    None
}

/// Generic polynomial evaluation function
fn compute_polynomial_evaluation_generic<F, G1, P>(poly: &P, point: &[F]) -> F
where
    F: Field + Clone,
    G1: Group<Scalar = F>,
    P: Polynomial<F, G1> + ?Sized,
{
    let mut eval_vec: Vec<F> = vec![F::zero(); poly.len()];

    let expected_size = 1 << point.len();
    assert!(
        poly.len() <= expected_size,
        "Too many coefficients: got {}, max for {} variables is {}",
        poly.len(),
        point.len(),
        expected_size
    );

    multilinear_lagrange_vec(&mut eval_vec, point);

    // Compute inner product <coeffs, eval_vec>
    let mut result = F::zero();
    for i in 0..poly.len() {
        if let Some(coeff) = poly.get(i) {
            result = result.add(&coeff.mul(&eval_vec[i]));
        }
    }
    result
}

/// Generic commit_rows implementation
fn commit_rows_generic<F, G1, P, M1>(poly: &P, g1_generators: &[G1], row_len: usize) -> Vec<G1>
where
    F: Field + Clone,
    G1: Group<Scalar = F>,
    P: Polynomial<F, G1> + ?Sized,
    M1: MultiScalarMul<G1>,
{
    let mut commitments = Vec::new();
    let total_len = poly.len();

    for row_start in (0..total_len).step_by(row_len) {
        let row_end = (row_start + row_len).min(total_len);
        
        // Collect row coefficients using get()
        let mut row_coeffs = Vec::with_capacity(row_end - row_start);
        for i in row_start..row_end {
            if let Some(coeff) = poly.get(i) {
                row_coeffs.push(coeff);
            } else {
                break; // Stop if we hit None
            }
        }
        
        if !row_coeffs.is_empty() {
            let commitment = M1::msm(&g1_generators[..row_coeffs.len()], &row_coeffs);
            commitments.push(commitment);
        }
    }

    commitments
}

/// Generic vector matrix product implementation
fn vector_matrix_product_generic<F, G1, P>(poly: &P, l_vec: &[F]) -> Vec<F>
where
    F: Field + Clone,
    G1: Group<Scalar = F>,
    P: Polynomial<F, G1> + ?Sized,
{
    // Assuming square matrix for simplicity, adjust as needed
    let n = l_vec.len();
    let mut result = vec![F::zero(); n];

    for row in 0..n {
        for col in 0..n {
            let idx = row * n + col;
            if let Some(coeff) = poly.get(idx) {
                let product = l_vec[row].mul(&coeff);
                result[col] = result[col].add(&product);
            }
        }
    }

    result
}

/// Compute the evaluation of a multilinear polynomial at a given point
/// Uses the lagrange evaluation basis
/// Ref: Section 2.5 of Dory paper.
pub fn compute_polynomial_evaluation<F, G1, P>(poly: &P, point: &[F]) -> F
where
    F: Field + Clone,
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
    F: Field + Clone,
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
        v[1] = point[0].clone();
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
                *ri = li.clone().mul(p);
                *li = li.clone().mul(&one_minus_p);
            }

            // Handle remaining left elements (when left is longer than right)
            for li in left[k..].iter_mut() {
                *li = li.clone().mul(&one_minus_p);
            }
        }
    }
}

/// Compute vectors L and R for matrix-based polynomial evaluation
/// Given a polynomial arranged as a matrix M, computes L and R such that:
/// polynomial_evaluation(b_point) = L^T × M × R
pub fn compute_left_right_vec<F: Field + Clone>(
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

/// Splits evaluation point coordinates into left/right tensors for matrix operations.
/// Outputs can be fed to `multilinear_lagrange_vec` to get the same result as `compute_left_right_vec`.
pub fn compute_l_r_tensors<F: Field + Copy>(
    b_point: &[F],
    sigma: usize,
    nu: usize,
) -> (Vec<F>, Vec<F>) {
    let mut r_coords = vec![F::zero(); 1 << nu]; // Column coordinates
    let mut l_coords = vec![F::zero(); 1 << nu]; // Row coordinates
    let num_vars = b_point.len();

    match num_vars {
        0 => {}

        n if n <= sigma => {
            // All variables → columns
            r_coords[..n].copy_from_slice(b_point);
        }

        n if n <= sigma * 2 => {
            // Split variables between rows and columns
            r_coords.copy_from_slice(&b_point[..nu]);
            l_coords[..(n - nu)].copy_from_slice(&b_point[nu..]);
        }

        _ => {
            // Too many variables: max columns, rest → rows
            r_coords[..sigma].copy_from_slice(&b_point[..sigma]);
            l_coords.copy_from_slice(&b_point[sigma..]);
        }
    }

    (l_coords, r_coords)
}

/// Computes v = L^T × M in Dory's VMV protocol
/// First step of Vector-Matrix-Vector: L^T * M
pub fn compute_v_vec<F, G1, P>(
    a: &P,          // Polynomial coefficients (flattened matrix M)
    left_vec: &[F], // L vector (row evaluation weights)
    sigma: usize,   // log₂(columns) - matrix width
    nu: usize,      // log₂(rows) - matrix height
) -> Vec<F>
where
    F: Field + Clone,
    G1: Group<Scalar = F>,
    P: Polynomial<F, G1> + ?Sized,
{
    let mut v = vec![F::zero(); 1 << nu]; // Result: v = L^T × M
    let cols_per_row = 1 << sigma;

    // Process each row of matrix M
    for row_idx in 0..(1 << nu) {
        if row_idx >= left_vec.len() {
            break;
        }

        let l_weight = &left_vec[row_idx]; // Weight for this row
        let row_start = row_idx * cols_per_row;

        // Add weighted row to result: v += l_weight * row
        for col_idx in 0..cols_per_row {
            if col_idx >= v.len() {
                break;
            }

            let coeff_idx = row_start + col_idx;
            if let Some(a_coeff) = a.get(coeff_idx) {
                let product = l_weight.mul(&a_coeff);
                v[col_idx] = v[col_idx].add(&product);
            }
        }
    }

    v
}
