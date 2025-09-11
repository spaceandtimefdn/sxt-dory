//! Utilities related to the VMV commitment strategy for multilinear polynomials
//! Defines VMV states for both provers, verifiers
use crate::{
    arithmetic::{Field, Group, Pairing},
    poly::{compute_left_right_vec, Polynomial},
    primitives::poly::BitOrdering,
    setup::ProverSetup,
    state::DoryProverState,
    MultiScalarMul,
};

/// Prover structure for computing commitment by VMV
pub struct VMVProverState<E: Pairing>
where
    E::G1: Group,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
{
    /// Evaluations of the columns of the matrix. That is, v = L^T * M.
    /// v[j] = <L, M[_, j]> = sum_{i=0}^{2^nu} L[i] M[i,j], where nu = floor(d / 2)
    pub(super) v_vec: Vec<<E::G1 as Group>::Scalar>,

    /// Commitments to the rows of the matrix.
    /// `T_vec_prime[i] = <M[i, _], Gamma_1[nu]> = sum_{j=0}^{2^nu} M[i,j] Gamma_1[nu][j]`.
    pub(super) t_vec_prime: Vec<<E as Pairing>::G1>,

    /// The left vector, L of LMR.
    pub(super) l_vec: Vec<<E::G1 as Group>::Scalar>,
    /// The right vector, R of LMR.
    pub(super) r_vec: Vec<<E::G1 as Group>::Scalar>,
}

/// Verifier analogue of VMVProverState
pub struct VMVVerifierState<E: Pairing>
where
    E::G1: Group,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
{
    /// The evaluation of the matrix. That is, y = LMR.
    pub(super) y: <E::G1 as Group>::Scalar,
    /// The commitment to the entire matrix. That is, `T = <T_vec_prime, Gamma_2[nu]>`.
    pub(super) t: <E as Pairing>::GT,
    /// The evaluation point.
    pub(super) eval_point: std::sync::Arc<[<E::G1 as Group>::Scalar]>,
}

/// Compute the size of the matrix M that is derived from the coefficients
/// 2^nu is the side length of M
pub fn compute_nu(num_vars: usize, sigma: usize) -> usize {
    if num_vars <= sigma * 2 {
        // No padding needed: prefer square (ν = σ)
        sigma
    } else {
        // Padding needed: columns capped at 2^σ, remaining variables go to rows
        num_vars - sigma
    }
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

/// Build the prover state for the VMV protocol
#[tracing::instrument(skip_all)]
pub fn build_vmv_prover_state<E, P>(
    polynomial: &P,                       // Multilinear polynomial coefficients
    b_point: &[<E::G1 as Group>::Scalar], // Evaluation point ( $v \in \mathbb{R}^d) for d variables
    row_commitments: Vec<E::G1>,
) -> VMVProverState<E>
where
    E: Pairing,
    P: Polynomial<<E::G1 as Group>::Scalar, E::G1> + ?Sized + Sync,
    E::G1: Group,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
{
    let sigma = (b_point.len() + 1) / 2;
    let nu = b_point.len() - sigma;
    let (l_vec, r_vec) = compute_left_right_vec(b_point, BitOrdering::LittleEndian);
    let v_vec = polynomial.vector_matrix_product(&l_vec, sigma, nu);

    VMVProverState {
        v_vec,
        t_vec_prime: row_commitments,
        l_vec,
        r_vec,
    }
}

/// Convert a VMVProverState to a ProverState
pub fn vmv_state_to_dory_prover_state<E: Pairing>(
    vmv_state: VMVProverState<E>,
    _prover_setup: &ProverSetup<E>,
) -> (Vec<<E::G1 as Group>::Scalar>, DoryProverState<E>)
where
    E::G1: Group,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
    <E::G1 as Group>::Scalar: Clone,
{
    // PLACEHOLDER FOR NOW!!
    let nu = vmv_state.r_vec.len();


    // Extract values from VMV state
    // Note: the paper has a typo and we want to actually set s1 = R, s2 = L (as we do below)
    let v_vec = vmv_state.v_vec;
    let r_vec = vmv_state.r_vec;
    let s2 = vmv_state.l_vec; // length 2^nu
    let v1 = vmv_state.t_vec_prime; // row commitments

    // Ensure s1 has length 2^nu by expanding r_vec (length 2^sigma) with block repetition
    let target_len = 1usize << nu;
    debug_assert!(target_len.is_power_of_two());
    debug_assert!(r_vec.len().is_power_of_two());
    // debug_assert!(target_len % r_vec.len() == 0, "2^ν must be a multiple of 2^σ");
    let s1 = if r_vec.len() == target_len {
        r_vec
    } else if r_vec.len() < target_len {
        let repeat = target_len / r_vec.len();
        let mut expanded = Vec::with_capacity(target_len);
        for val in r_vec.iter() {
            for _ in 0..repeat {
                expanded.push(val.clone());
            }
        }
        expanded
    } else {
        // This case should not occur with current compute_nu (guarantees nu >= sigma)
        // Fallback: truncate to target_len (conservative); a better approach would rebalance nu/sigma.
        r_vec.into_iter().take(target_len).collect()
    };

    // eval_vmv_re will calculate v2
    let v2 = Vec::new();

    // Create the ProverState
    let state = DoryProverState::new(v1, v2, s1, s2, nu);

    (v_vec, state)
}
