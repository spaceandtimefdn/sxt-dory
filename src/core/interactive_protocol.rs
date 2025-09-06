//! We implement the interactive protocol for prover <> verifier Dory proofs
//! This mainly involves the messages in the dory-reduce protocol
use rayon::prelude::*;

use crate::{
    arithmetic::{Field, Group, MultiScalarMul, Pairing},
    messages::{FirstReduceChallenge, FirstReduceMessage, SecondReduceChallenge, SecondReduceMessage},
    setup::VerifierSetup,
    state::{DoryProverState, DoryVerifierState, VerifierState},
    poly::fold_eval_from_coords_and_alphas,
};
use crate::curve::SmallScalarMul;

use super::{ProverSetup, FinalizeChallenge};

/// Below is the **prover** side of the interactive protocol for Dory
/// We define the relevant message implementations in the order of communication
impl<E: Pairing> crate::ProverState for DoryProverState<E>
where
    E::G1: Group + SmallScalarMul,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar> + SmallScalarMul,
{
    type G1 = E::G1;
    type G2 = E::G2;
    type GT = E::GT;
    type Scalar = <E::G1 as Group>::Scalar;
    type Setup = ProverSetup<E>;

    /* ---------- First‑Reduce --------------------------------------- */
    #[tracing::instrument(skip_all)]
    fn compute_first_reduce_message<M1, M2>(
        &self,
        setup: &Self::Setup,
    ) -> FirstReduceMessage<Self::G1, Self::G2, Self::GT>
    where
        M1: MultiScalarMul<Self::G1>,
        M2: MultiScalarMul<Self::G2>,
    {
        if self.nu == 0 {
            panic!("Not enough rounds left in prover state");
        }

        // n/2
        let n2 = 1usize << (self.nu - 1);

        let (v1_l, v1_r) = self.v1.split_at(n2);
        let (v2_l, v2_r) = self.v2.split_at(n2);

        /* ---------- COMPUTE D ---------- */
        // Collapsed Γ-vectors of length n/2 (Γ₁′, Γ₂′)
        let g2_prime = &setup.g2_vec()[..1 << (self.nu - 1)];
        let g1_prime = &setup.g1_vec()[..1 << (self.nu - 1)];

        let (d1_left, d1_right, d2_left, d2_right) = {
            // D₁L,R as before (opt: cached G2)
            let (d1_left, d1_right) = if setup.g2_cache.is_some() {
                let g2_prime_count = 1 << (self.nu - 1);
                let d1_left = E::multi_pair_cached(
                    Some(v1_l), None, None, None, Some(g2_prime_count), setup.g2_cache.as_ref(),
                );
                let d1_right = E::multi_pair_cached(
                    Some(v1_r), None, None, None, Some(g2_prime_count), setup.g2_cache.as_ref(),
                );
                (d1_left, d1_right)
            } else {
                (E::multi_pair(v1_l, g2_prime), E::multi_pair(v1_r, g2_prime))
            };

            // D₂L,R: specialize round 0 if v2_scalars present (PCS optimization)
            let (d2_left, d2_right) = if self.v2_scalars.is_some() {
                let v2_scalars = self.v2_scalars.as_ref().unwrap();
                let n2 = 1usize << (self.nu - 1);
                let (v_l, v_r) = v2_scalars.split_at(n2);
                // MSM(Γ₁′, v_{L/R}) in G1
                let m_l = M1::msm(g1_prime, v_l);
                let m_r = M1::msm(g1_prime, v_r);
                // Single pairing with Γ_{2,fin}
                let d2_left = E::pair(&m_l, setup.g_fin());
                let d2_right = E::pair(&m_r, setup.g_fin());
                (d2_left, d2_right)
            } else {
                (E::multi_pair(g1_prime, v2_l), E::multi_pair(g1_prime, v2_r))
            };

            (d1_left, d1_right, d2_left, d2_right)
        };

        /* ---------- COMPUTE E (for extended protocol) ---------- */
        // E₁β = ⟨Γ₁ , s₂⟩
        let e1_beta = M1::msm(&setup.g1_vec()[..1 << self.nu], &self.s2);
        // E₂β = ⟨Γ₂ , s₁⟩
        let e2_beta = M2::msm(&setup.g2_vec()[..1 << self.nu], &self.s1);

        FirstReduceMessage {
            d1_left,
            d1_right,
            d2_left,
            d2_right,
            e1_beta,
            e2_beta,
        }
    }

    /* ---------- Reduce-Combine --------------------------------------- */
    #[tracing::instrument(skip_all)]
    fn reduce_combine<M1, M2>(
        mut self,
        setup: &Self::Setup,
        chall: FirstReduceChallenge,
    ) -> Self
    where
        M1: MultiScalarMul<Self::G1>,
        M2: MultiScalarMul<Self::G2>,
    {
        let beta = chall.beta;

        // Clear PCS first-round scalar optimization after first message use
        if self.v2_scalars.is_some() {
            self.v2_scalars = None;
        }

        let g1_prime = &setup.g1_vec()[..1 << self.nu];
        let g2_prime = &setup.g2_vec()[..1 << self.nu];

        // Prover work P(*):
        // ṽ₁ ← ṽ₁ + β·Γ₁
        // Use cached version if cache is available
        if setup.g1_cache.is_some() && setup.g2_cache.is_some() {
            M1::fixed_scalar_variable_with_add_cached_small(
                g1_prime.len(),
                setup.g1_cache.as_ref(),
                setup.g2_cache.as_ref(),
                &mut self.v1,
                beta,
            );
        } else {
            M1::fixed_scalar_variable_with_add_small(g1_prime, &mut self.v1, beta);
        }

        // ṽ₂ ← ṽ₂ + β·Γ₂
        // Use cached version if cache is available
        if setup.g1_cache.is_some() && setup.g2_cache.is_some() {
            M2::fixed_scalar_variable_with_add_cached_small(
                g2_prime.len(),
                setup.g1_cache.as_ref(),
                setup.g2_cache.as_ref(),
                &mut self.v2,
                beta,
            );
        } else {
            M2::fixed_scalar_variable_with_add_small(g2_prime, &mut self.v2, beta);
        }

        self
    }

    /* ---------- Second‑Reduce -------------------------------------- */
    #[tracing::instrument(skip_all)]
    fn compute_second_reduce_message<M1, M2>(
        &self,
        _setup: &Self::Setup, // not used in this step
    ) -> SecondReduceMessage<Self::G1, Self::G2>
    where
        M1: MultiScalarMul<Self::G1>,
        M2: MultiScalarMul<Self::G2>,
    {
        let n2 = 1usize << (self.nu - 1);

        let (v1_l, v1_r) = self.v1.split_at(n2);
        let (v2_l, v2_r) = self.v2.split_at(n2);
        let (s1_l, s1_r) = self.s1.split_at(n2);
        let (s2_l, s2_r) = self.s2.split_at(n2);

        // ---- E terms (extended protocol) ---------------------------------------
        // Match verifier update: E₁' ← E₁- + α · (E₁ + β · E₁β + α · E₁+)
        // Therefore:
        //   E₁- = ⟨v₁L(new), s₂R⟩,  E₁+ = ⟨v₁R(new), s₂L⟩
        let e1_minus = M1::msm(v1_l, s2_r); // ⟨v₁L, s₂R⟩
        let e1_plus = M1::msm(v1_r, s2_l); // ⟨v₁R, s₂L⟩
        // For E₂, the current assignments already match: E₂- = ⟨v₂L(new), s₁R⟩, E₂+ = ⟨v₂R(new), s₁L⟩
        let e2_plus = M2::msm(v2_r, s1_l); // ⟨v₂R, s₁L⟩
        let e2_minus = M2::msm(v2_l, s1_r); // ⟨v₂L, s₁R⟩

        SecondReduceMessage {
            e1_plus,
            e1_minus,
            e2_plus,
            e2_minus,
        }
    }

    /// On every round, cut vector length in half and fold with α (Nemo semantics):
    ///
    ///   v₁ ← v₁L + α · v₁R
    ///   v₂ ← v₂L + α · v₂R
    ///   s₁ ← α · s₁L + s₁R
    ///   s₂ ← α · s₂L + s₂R
    ///
    /// After folding, all four vectors are truncated to `n/2`.
    #[tracing::instrument(skip_all)]
    fn reduce_fold<M1, M2>(
        mut self,
        _setup: &Self::Setup,
        chall: SecondReduceChallenge,
    ) -> Self
    where
        M1: MultiScalarMul<Self::G1>,
        M2: MultiScalarMul<Self::G2>,
    {
        let alpha = chall.alpha;
        let n2 = 1usize << (self.nu - 1);

        /* ─── fold v-vectors ────────────────────────────────────────────── */
        let (v1_l, v1_r_slice) = self.v1.split_at_mut(n2);
        let v1_r = &*v1_r_slice; // Convert mutable slice to immutable for par_iter()

        let (v2_l, v2_r_slice) = self.v2.split_at_mut(n2);
        let v2_r = &*v2_r_slice;

        M1::fixed_scalar_scale_with_add_small(v1_l, v1_r, alpha);

        M2::fixed_scalar_scale_with_add_small(v2_l, v2_r, alpha);

        self.v1.truncate(n2);
        self.v2.truncate(n2);

        /* ─── fold s-vectors (extended protocol)──────────-────────────────── */
        let (s1_l, s1_r_slice) = self.s1.split_at_mut(n2);
        let s1_r = &*s1_r_slice;

        let (s2_l, s2_r_slice) = self.s2.split_at_mut(n2);
        let s2_r = s2_r_slice;

        s1_l.par_iter_mut()
            .zip(s1_r.par_iter())
            .for_each(|(s_l, s_r_val)| *s_l = s_l.mul_u128(alpha).add(s_r_val));

        s2_l.par_iter_mut()
            .zip(s2_r.par_iter())
            .for_each(|(s_l, s_r_val)| *s_l = s_l.mul_u128(alpha).add(s_r_val));

        self.s1.truncate(n2);
        self.s2.truncate(n2);

        self.nu -= 1;

        self
    }

    /// Expose base-case folded group elements (length-1 vectors)
    fn final_bases(&self) -> (Self::G1, Self::G2) {
        assert_eq!(self.nu, 0, "final_bases called before base case");
        let v1_final = self
            .v1
            .get(0)
            .cloned()
            .unwrap_or_else(|| <E::G1 as Group>::identity());
        let v2_final = self
            .v2
            .get(0)
            .cloned()
            .unwrap_or_else(|| <E::G2 as Group>::identity());
        (v1_final, v2_final)
    }

    /// Borrow base-case folded group elements (length-1 vectors)
    fn final_bases_ref(&self) -> (&Self::G1, &Self::G2) {
        assert_eq!(self.nu, 0, "final_bases_ref called before base case");
        let v1_ref = self
            .v1
            .get(0)
            .expect("v1 empty at base case");
        let v2_ref = self
            .v2
            .get(0)
            .expect("v2 empty at base case");
        (v1_ref, v2_ref)
    }
}

/// Below is the **verifier** side of the interactive protocol for Dory
/// We define the relevant message implementations in the order of communication
impl<E: Pairing> VerifierState for DoryVerifierState<E>
where
    E::G1: Group + SmallScalarMul,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar> + SmallScalarMul,
    E::GT: Group<Scalar = <E::G1 as Group>::Scalar> + SmallScalarMul,
{
    type G1 = E::G1;
    type G2 = E::G2;
    type GT = E::GT;
    type Scalar = <E::G1 as Group>::Scalar;
    type Setup = VerifierSetup<E>;

    /// This is the round i verifier algorithm of the extended Dory-Reduce algorithm in section 3.2 & 4.2 of the paper.
    /// This function should be called after messages are received and challenges are pulled from the transcript.
    fn dory_reduce_verify_round(
        &mut self,
        setup: &Self::Setup,
        first_msg: &FirstReduceMessage<Self::G1, Self::G2, Self::GT>,
        second_msg: &SecondReduceMessage<Self::G1, Self::G2>,
        beta: [u64; 2],
        alpha: [u64; 2],
    ) -> bool {

        // Record the α used for this round for later base-case evaluation of s₁,s₂
        self.alpha_challenges.push(alpha);

        // Update D₁ and D₂ (Nemo semantics, no inverses)
        // D₁' <- D₁L + α * D₁R + β * (Δ₁L + α * Δ₁R)
        // D₂' <- D₂L + α * D₂R + β * (Δ₂L + α * Δ₂R)
        Self::dory_reduce_verify_update_ds(
            self,
            setup,
            (
                &first_msg.d1_left,
                &first_msg.d1_right,
                &first_msg.d2_left,
                &first_msg.d2_right,
            ),
            alpha,
            beta,
        );

        // Update E₁ and E₂ for the **extended** protocol (Nemo semantics)
        // E₁' <- E₁- + α * (E₁ + β * E₁β + α * E₁+)
        // E₂' <- E₂- + α * (E₂ + β * E₂β + α * E₂+)
        Self::dory_reduce_verify_update_es(
            self,
            (&first_msg.e1_beta, &first_msg.e2_beta),
            (
                &second_msg.e1_plus,
                &second_msg.e1_minus,
                &second_msg.e2_plus,
                &second_msg.e2_minus,
            ),
            alpha,
            beta,
        );

        // decrement the rounds
        self.nu -= 1;

        true
    }

    /// From the Nemo-Reduce semantics (no inverses).
    /// Updates `D₁` and `D₂` in verifier state:
    /// * D₁' ← D₁L + α · D₁R + β · (Δ₁L + α · Δ₁R)
    /// * D₂' ← D₂L + α · D₂R + β · (Δ₂L + α · Δ₂R)
    fn dory_reduce_verify_update_ds(
        &mut self,
        setup: &Self::Setup,
        d_values: (&Self::GT, &Self::GT, &Self::GT, &Self::GT),
        alpha: [u64; 2],
        beta: [u64; 2],
    ) {
        let (d_1l, d_1r, d_2l, d_2r) = d_values;

        // Get the precomputed values for the current round
        let delta_1l = &setup.delta_1l[self.nu];
        let delta_1r = &setup.delta_1r[self.nu];
        let delta_2l = &setup.delta_2l[self.nu];
        let delta_2r = &setup.delta_2r[self.nu];

        // D₁' ← D₁L + α·D₁R + β·(Δ₁L + α·Δ₁R)
        let mut new_d_1 = d_1l.add(&d_1r.scale_u128(alpha));
        let delta_1_tail = delta_1l.add(&delta_1r.scale_u128(alpha));
        new_d_1 = new_d_1.add(&delta_1_tail.scale_u128(beta));

        // D₂' ← D₂L + α·D₂R + β·(Δ₂L + α·Δ₂R)
        let mut new_d_2 = d_2l.add(&d_2r.scale_u128(alpha));
        let delta_2_tail = delta_2l.add(&delta_2r.scale_u128(alpha));
        new_d_2 = new_d_2.add(&delta_2_tail.scale_u128(beta));

        self.d_1 = new_d_1;
        self.d_2 = new_d_2;
    }

    /// Nemo-Reduce semantics for E updates (no inverses):
    /// * E₁' ← E₁- + α · (E₁ + β · E₁β + α · E₁+)
    /// * E₂' ← E₂- + α · (E₂ + β · E₂β + α · E₂+)
    fn dory_reduce_verify_update_es(
        &mut self,
        e_beta_pair: (&Self::G1, &Self::G2),
        e_values: (&Self::G1, &Self::G1, &Self::G2, &Self::G2),
        alpha: [u64; 2],
        beta: [u64; 2],
    ) {
        let (e_1beta, e_2beta) = e_beta_pair;
        let (e_1plus, e_1minus, e_2plus, e_2minus) = e_values;

        // E₁' ← E₁- + α · (E₁ + β · E₁β + α · E₁+)
        let inner_e1 = self
            .e_1
            .add(&e_1beta.scale_u128(beta))
            .add(&e_1plus.scale_u128(alpha));
        let new_e_1: Self::G1 = e_1minus.add(&inner_e1.scale_u128(alpha));

        // E₂' ← E₂- + α · (E₂ + β · E₂β + α · E₂+)
        let inner_e2 = self
            .e_2
            .add(&e_2beta.scale_u128(beta))
            .add(&e_2plus.scale_u128(alpha));
        let new_e_2: Self::G2 = e_2minus.add(&inner_e2.scale_u128(alpha));

        self.e_1 = new_e_1;
        self.e_2 = new_e_2;
    }

    fn set_final_bases(&mut self, v1_final: Self::G1, v2_final: Self::G2) {
        self.v1_final = Some(v1_final);
        self.v2_final = Some(v2_final);
    }

    /// Final verification step (Nemo-Finalize, non-ZK)
    /// Implements the deferred VMV pairing check batched with γ₁, γ₂, and
    /// leaves linear G1/G2 checks to be wired once s₁, s₂ base-case values are available.
    fn finalize(
        &self,
        setup: &Self::Setup,
        gamma_pair: FinalizeChallenge,
    ) -> bool {
        // Base case
        assert_eq!(self.nu, 0);

        let gamma_1 = gamma_pair.gamma_1;
        let gamma_2 = gamma_pair.gamma_2;

        let s1_final: <E::G1 as Group>::Scalar =
            fold_eval_from_coords_and_alphas(&self.eval_point_left, &self.alpha_challenges);
        let s2_final: <E::G1 as Group>::Scalar =
            fold_eval_from_coords_and_alphas(&self.eval_point_right, &self.alpha_challenges);

        // Enforce presence of base-case group elements
        let (Some(v1_final), Some(v2_final)) = (&self.v1_final, &self.v2_final) else {
            return false;
        };

        // Linear base-case checks (non-ZK): E1 == s2_final · v1_final, E2 == s1_final · v2_final
        let e1_expected = v1_final.scale(&s2_final);
        if self.e_1 != e1_expected {
            return false;
        }
        let e2_expected = v2_final.scale(&s1_final);
        if self.e_2 != e2_expected {
            return false;
        }

        // GT batched pairing check:
        // e(E1_orig, Γ2_fin) + e(γ1 · E1, Γ2_0) + e(γ2 · Γ1_0, E2)
        //  == D2_orig + γ1 · D1 + γ2 · D2
        let e1_scaled = self.e_1.scale_u128(gamma_1);
        let g1_scaled = setup.g1_0.scale_u128(gamma_2);
        let lhs = E::multi_pair_refs(
            &[&self.e_1_orig, &e1_scaled, &g1_scaled],
            &[&setup.g_fin, &setup.g2_0, &self.e_2],
        );

        let rhs = self
            .d_2_orig
            .add(&self.d_1.scale_u128(gamma_1))
            .add(&self.d_2.scale_u128(gamma_2));

        if lhs != rhs {
            return false;
        }

        true
    }
}
