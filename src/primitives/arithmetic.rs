use std::fmt::Debug;

use rand::Rng;

/// Field elements F_p
pub trait Field: Sized + Clone + Copy + PartialEq + Send + Sync {
    fn zero() -> Self;
    fn one() -> Self;
    fn is_zero(&self) -> bool;

    fn add(&self, rhs: &Self) -> Self;
    fn sub(&self, rhs: &Self) -> Self;
    fn mul(&self, rhs: &Self) -> Self;
    fn inv(&self) -> Option<Self>;

    fn random<R: Rng>(rng: &mut R) -> Self;

    fn from_u64(val: u64) -> Self;
    fn from_i64(val: i64) -> Self;
}

/// Group elements G1 / G2 / GT
pub trait Group: Sized + Clone + PartialEq + Send + Sync + Debug {
    type Scalar: Field;

    fn identity() -> Self;
    fn add(&self, rhs: &Self) -> Self;
    fn neg(&self) -> Self;
    fn scale(&self, k: &Self::Scalar) -> Self;

    fn random<R>(rng: &mut R) -> Self;
}

/// Pairing group G1, G2, GT
pub trait Pairing: Sized + Send + Sync {
    type G1: Group;
    type G2: Group;
    type GT: Group;

    /// e : G1 × G2 → GT
    fn pair(p: &Self::G1, q: &Self::G2) -> Self::GT;

    /// Multi-pairing: computes the product of pairings
    /// Π e(p_i, q_i)
    fn multi_pair(ps: &[Self::G1], qs: &[Self::G2]) -> Self::GT {
        assert_eq!(
            ps.len(),
            qs.len(),
            "multi_pair requires equal length vectors"
        );

        if ps.is_empty() {
            return Self::GT::identity();
        }

        ps.iter()
            .zip(qs.iter())
            .fold(Self::GT::identity(), |acc, (p, q)| {
                acc.add(&Self::pair(p, q))
            })
    }
}

pub trait MultiScalarMul<G: Group> {
    fn msm(bases: &[G], scalars: &[G::Scalar]) -> G;

    /// Fixed-base vectorized scalar multiplication where the same base is scaled by each scalar individually
    /// Computes: [base * scalars[0], base * scalars[1], ..., base * scalars[n-1]]
    fn fixed_base_vector_msm(base: &G, scalars: &[G::Scalar]) -> Vec<G> {
        // Default implementation: scale each scalar individually
        scalars.iter().map(|scalar| base.scale(scalar)).collect()
    }

    /// Fixed-scalar variable-base vectorized multiplication with add: vs[i] = vs[i] + scalar * bases[i]
    /// Modifies vs in place by adding the scaled bases
    /// This is optimized for cases like reduce_fold where we compute v_l = alpha * v_l + v_r
    fn fixed_scalar_variable_with_add(bases: &[G], vs: &mut [G], scalar: &G::Scalar) {
        assert_eq!(bases.len(), vs.len(), "bases and vs must have same length");
        // Default implementation: scale each base and add to vs
        for (base, v) in bases.iter().zip(vs.iter_mut()) {
            *v = v.add(&base.scale(scalar));
        }
    }

    /// Fixed-scalar vectorized multiplication with add: vs[i] = scalar * vs[i] + addends[i]
    /// Modifies vs in place by scaling each element and adding the corresponding addend
    /// This is optimized for cases like reduce_fold where we compute v_l = alpha * v_l + v_r
    fn fixed_scalar_scale_with_add(vs: &mut [G], addends: &[G], scalar: &G::Scalar) {
        assert_eq!(
            vs.len(),
            addends.len(),
            "vs and addends must have same length"
        );
        // Default implementation: scale each vs element and add the corresponding addend
        for (v, addend) in vs.iter_mut().zip(addends.iter()) {
            *v = v.scale(scalar).add(addend);
        }
    }
}
