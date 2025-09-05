//! We implement the interactive protocol for prover <> verifier Dory proofs
//! This mainly involves the messages in the dory-reduce protocol
use rayon::prelude::*;

use crate::{
    arithmetic::{Field, Group, MultiScalarMul, Pairing},
    messages::{FirstReduceChallenge, FirstReduceMessage, SecondReduceChallenge, SecondReduceMessage},
    setup::VerifierSetup,
    state::{DoryProverState, DoryVerifierState, VerifierState},
};

use super::{ProverSetup, FinalizeChallenge};

/// Below is the **prover** side of the interactive protocol for Dory
/// We define the relevant message implementations in the order of communication
impl<E: Pairing> crate::ProverState for DoryProverState<E>
where
    E::G1: Group,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
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

        let (d1_left, d1_right, d2_left, d2_right) =
                // Use cached G2 if available, always use runtime G1
                if setup.g2_cache.is_some() {
                    let g2_prime_count = 1 << (self.nu - 1);

                    // D₁L,R = ⟨v₁L/R , Γ₂′⟩ - v1 is runtime, g2_prime uses cache
                    let d1_left = E::multi_pair_cached(
                        Some(v1_l),
                        None,
                        None, // G1: use runtime points v1_l
                        None,
                        Some(g2_prime_count),
                        setup.g2_cache.as_ref(), // G2: use first 2^(nu-1) cached elements
                    );
                    let d1_right = E::multi_pair_cached(
                        Some(v1_r),
                        None,
                        None, // G1: use runtime points v1_r
                        None,
                        Some(g2_prime_count),
                        setup.g2_cache.as_ref(), // G2: use first 2^(nu-1) cached elements
                    );

                    // D₂L,R = ⟨Γ₁′ , v₂L/R⟩ - g1_prime is runtime, v2 is runtime
                    let d2_left = E::multi_pair(g1_prime, v2_l);
                    let d2_right = E::multi_pair(g1_prime, v2_r);
                    (d1_left, d1_right, d2_left, d2_right)
                } else {
                    // Fallback to regular multi-pairing when cache is not available
                    let d1_left = E::multi_pair(v1_l, g2_prime);
                    let d1_right = E::multi_pair(v1_r, g2_prime);
                    let d2_left = E::multi_pair(g1_prime, v2_l);
                    let d2_right = E::multi_pair(g1_prime, v2_r);
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
        chall: FirstReduceChallenge<Self::Scalar>,
    ) -> Self
    where
        M1: MultiScalarMul<Self::G1>,
        M2: MultiScalarMul<Self::G2>,
    {
        let beta = chall.beta;

        let g1_prime = &setup.g1_vec()[..1 << self.nu];
        let g2_prime = &setup.g2_vec()[..1 << self.nu];

        // Prover work P(*):
        // ṽ₁ ← ṽ₁ + β·Γ₁
        // Use cached version if cache is available
        if setup.g1_cache.is_some() && setup.g2_cache.is_some() {
            M1::fixed_scalar_variable_with_add_cached(
                g1_prime.len(),
                setup.g1_cache.as_ref(),
                setup.g2_cache.as_ref(),
                &mut self.v1,
                &beta,
            );
        } else {
            M1::fixed_scalar_variable_with_add(g1_prime, &mut self.v1, &beta);
        }

        // ṽ₂ ← ṽ₂ + β·Γ₂
        // Use cached version if cache is available
        if setup.g1_cache.is_some() && setup.g2_cache.is_some() {
            M2::fixed_scalar_variable_with_add_cached(
                g2_prime.len(),
                setup.g1_cache.as_ref(),
                setup.g2_cache.as_ref(),
                &mut self.v2,
                &beta,
            );
        } else {
            M2::fixed_scalar_variable_with_add(g2_prime, &mut self.v2, &beta);
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

        // PCS variant: omit C terms (c_plus, c_minus)
        // ---- E terms (extended protocol) ---------------------------------------
        let e1_plus = M1::msm(v1_l, s2_r); // ⟨v₁L, s₂R⟩
        let e1_minus = M1::msm(v1_r, s2_l); // ⟨v₁R, s₂L⟩
        let e2_plus = M2::msm(v2_r, s1_l); // ⟨v₂R, s₁L⟩
        let e2_minus = M2::msm(v2_l, s1_r); // ⟨v₂L, s₁R⟩

        SecondReduceMessage {
            e1_plus,
            e1_minus,
            e2_plus,
            e2_minus,
        }
    }

    /// On every round, cut vector length in half and fold with α (Light semantics):
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
        chall: SecondReduceChallenge<Self::Scalar>,
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

        M1::fixed_scalar_scale_with_add(v1_l, v1_r, &alpha);

        M2::fixed_scalar_scale_with_add(v2_l, v2_r, &alpha);

        self.v1.truncate(n2);
        self.v2.truncate(n2);

        /* ─── fold s-vectors (extended protocol)──────────-────────────────── */
        let (s1_l, s1_r_slice) = self.s1.split_at_mut(n2);
        let s1_r = &*s1_r_slice;

        let (s2_l, s2_r_slice) = self.s2.split_at_mut(n2);
        let s2_r = s2_r_slice;

        s1_l.par_iter_mut()
            .zip(s1_r.par_iter())
            .for_each(|(s_l, s_r_val)| *s_l = s_l.mul(&alpha).add(s_r_val));

        s2_l.par_iter_mut()
            .zip(s2_r.par_iter())
            .for_each(|(s_l, s_r_val)| *s_l = s_l.mul(&alpha).add(s_r_val));

        self.s1.truncate(n2);
        self.s2.truncate(n2);

        self.nu -= 1;

        self
    }
}

/// Below is the **verifier** side of the interactive protocol for Dory
/// We define the relevant message implementations in the order of communication
impl<E: Pairing> VerifierState for DoryVerifierState<E>
where
    E::G1: Group,
    E::G2: Group<Scalar = <E::G1 as Group>::Scalar>,
    E::GT: Group<Scalar = <E::G1 as Group>::Scalar>,
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
        alpha: Self::Scalar,
        beta: Self::Scalar,
    ) -> bool {

        // Update D₁ and D₂ (Light semantics, no inverses)
        // D₁' <- D₁L + α * D₁R + β * (Δ₁L + α * Δ₁R)
        // D₂' <- D₂L + α * D₂R + β * (Δ₂L + α * Δ₂R)
        Self::dory_reduce_verify_update_ds(
            self,
            setup,
            (
                first_msg.d1_left.clone(),
                first_msg.d1_right.clone(),
                first_msg.d2_left.clone(),
                first_msg.d2_right.clone(),
            ),
            alpha.clone(),
            beta.clone(),
        );

        // Update E₁ and E₂ for the **extended** protocol (Light semantics)
        // E₁' <- E₁- + α * (E₁ + β * E₁β + α * E₁+)
        // E₂' <- E₂- + α * (E₂ + β * E₂β + α * E₂+)
        Self::dory_reduce_verify_update_es(
            self,
            (first_msg.e1_beta.clone(), first_msg.e2_beta.clone()),
            (
                second_msg.e1_plus.clone(),
                second_msg.e1_minus.clone(),
                second_msg.e2_plus.clone(),
                second_msg.e2_minus.clone(),
            ),
            alpha.clone(),
            beta.clone(),
        );

        // decrement the rounds
        self.nu -= 1;

        true
    }

    /// From the Dory-Reduce-Light semantics (no inverses).
    /// Updates `D₁` and `D₂` in verifier state:
    /// * D₁' ← D₁L + α · D₁R + β · (Δ₁L + α · Δ₁R)
    /// * D₂' ← D₂L + α · D₂R + β · (Δ₂L + α · Δ₂R)
    fn dory_reduce_verify_update_ds(
        &mut self,
        setup: &Self::Setup,
        d_values: (Self::GT, Self::GT, Self::GT, Self::GT),
        alpha: Self::Scalar,
        beta: Self::Scalar,
    ) {
        let (d_1l, d_1r, d_2l, d_2r) = d_values;

        // Get the precomputed values for the current round
        let delta_1l = &setup.delta_1l[self.nu];
        let delta_1r = &setup.delta_1r[self.nu];
        let delta_2l = &setup.delta_2l[self.nu];
        let delta_2r = &setup.delta_2r[self.nu];

        // D₁' ← D₁L + α·D₁R + β·(Δ₁L + α·Δ₁R)
        let mut new_d_1 = d_1l.add(&d_1r.scale(&alpha));
        let delta_1_tail = delta_1l.add(&delta_1r.scale(&alpha));
        new_d_1 = new_d_1.add(&delta_1_tail.scale(&beta));

        // D₂' ← D₂L + α·D₂R + β·(Δ₂L + α·Δ₂R)
        let mut new_d_2 = d_2l.add(&d_2r.scale(&alpha));
        let delta_2_tail = delta_2l.add(&delta_2r.scale(&alpha));
        new_d_2 = new_d_2.add(&delta_2_tail.scale(&beta));

        self.d_1 = new_d_1;
        self.d_2 = new_d_2;
    }

    /// Extended Dory-Reduce-Light semantics for E updates (no inverses):
    /// * E₁' ← E₁- + α · (E₁ + β · E₁β + α · E₁+)
    /// * E₂' ← E₂- + α · (E₂ + β · E₂β + α · E₂+)
    fn dory_reduce_verify_update_es(
        &mut self,
        e_beta_pair: (Self::G1, Self::G2),
        e_values: (Self::G1, Self::G1, Self::G2, Self::G2),
        alpha: Self::Scalar,
        beta: Self::Scalar,
    ) {
        let (e_1beta, e_2beta) = e_beta_pair;
        let (e_1plus, e_1minus, e_2plus, e_2minus) = e_values;

        // E₁' ← E₁- + α · (E₁ + β · E₁β + α · E₁+)
        let inner_e1 = self
            .e_1
            .add(&e_1beta.scale(&beta))
            .add(&e_1plus.scale(&alpha));
        let new_e_1: Self::G1 = e_1minus.add(&inner_e1.scale(&alpha));

        // E₂' ← E₂- + α · (E₂ + β · E₂β + α · E₂+)
        let inner_e2 = self
            .e_2
            .add(&e_2beta.scale(&beta))
            .add(&e_2plus.scale(&alpha));
        let new_e_2: Self::G2 = e_2minus.add(&inner_e2.scale(&alpha));

        self.e_1 = new_e_1;
        self.e_2 = new_e_2;
    }

    /// Final verification step (Nemo-Finalize, non-ZK)
    /// Implements the deferred VMV pairing check batched with γ₁, γ₂, and
    /// leaves linear G1/G2 checks to be wired once s₁, s₂ base-case values are available.
    fn finalize(
        &self,
        setup: &Self::Setup,
        gamma_pair: FinalizeChallenge<Self::Scalar>,
    ) -> bool {
        // Base case
        assert_eq!(self.nu, 0);

        let gamma_1 = gamma_pair.gamma_1;
        let gamma_2 = gamma_pair.gamma_2;

        // GT batched pairing check:
        // e(E1_orig, Γ2_fin) + e(γ1 · E1, Γ2_0) + e(γ2 · Γ1_0, E2)
        //  == D2_orig + γ1 · D1 + γ2 · D2
        let t0 = E::pair(&self.e_1_orig, &setup.g_fin);
        let t1 = E::pair(&self.e_1.scale(&gamma_1), &setup.g2_0);
        let t2 = E::pair(&setup.g1_0.scale(&gamma_2), &self.e_2);
        let lhs = t0.add(&t1).add(&t2);

        let rhs = self
            .d_2_orig
            .add(&self.d_1.scale(&gamma_1))
            .add(&self.d_2.scale(&gamma_2));

        if lhs != rhs {
            return false;
        }

        // TODO: Linear checks at base case once s₁_final, s₂_final and v₁_final, v₂_final
        // are available on the verifier side. For now, they are deferred.

        true
    }
}
