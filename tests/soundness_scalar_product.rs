#![allow(missing_docs)]
use ark_bn254::{Fq12, Fr, G1Affine};
use ark_ff::UniformRand;
use dory::{
    arithmetic::{Field, Group, MultiScalarMul, Pairing},
    builder::{DoryProofBuilder, DoryVerifyBuilder},
    inner_product::{inner_product_prove, inner_product_verify},
    setup::ProverSetup,
    state::{DoryProverState, DoryVerifierState},
    toy_transcript::ToyTranscript,
};

use dory::curve::{test_rng, ArkBn254Pairing, G2AffineWrapper, OptimizedMsmG1, OptimizedMsmG2};

// Helper function to generate test environment
fn setup_scalar_product_test_environment(
    log_n: usize,
) -> (
    ProverSetup<ArkBn254Pairing>,
    dory::setup::VerifierSetup<ArkBn254Pairing>,
    DoryProverState<ArkBn254Pairing>,
    DoryVerifierState<ArkBn254Pairing>,
) {
    let mut rng = test_rng();
    let vector_size = 1 << log_n;

    // Setup - max_log_n should be 2 * log_n because g1_vec/g2_vec have length sqrt(n)
    let prover_setup = ProverSetup::<ArkBn254Pairing>::new(&mut rng, 2 * log_n);
    let verifier_setup = prover_setup.to_verifier_setup();

    // Generate vectors
    let v1: Vec<G1Affine> = (0..vector_size)
        .map(|_| G1Affine::random(&mut rng))
        .collect();
    let v2: Vec<G2AffineWrapper> = (0..vector_size)
        .map(|_| G2AffineWrapper::random(&mut rng))
        .collect();
    let s1: Vec<Fr> = (0..vector_size).map(|_| Fr::random(&mut rng)).collect();
    let s2: Vec<Fr> = (0..vector_size).map(|_| Fr::random(&mut rng)).collect();

    // Create states
    let prover_state = DoryProverState::new(v1.clone(), v2.clone(), s1.clone(), s2.clone(), log_n);
    let c = ArkBn254Pairing::multi_pair(&v1, &v2);
    let d_1 = ArkBn254Pairing::multi_pair(&v1, &prover_setup.g2_vec()[..1 << log_n]);
    let d_2 = ArkBn254Pairing::multi_pair(&prover_setup.g1_vec()[..1 << log_n], &v2);
    let e_1 = OptimizedMsmG1::msm(&prover_setup.g1_vec()[..1 << log_n], &s2);
    let e_2 = OptimizedMsmG2::msm(&prover_setup.g2_vec()[..1 << log_n], &s1);
    let verifier_state = DoryVerifierState::new(d_1, d_2, e_1, e_2, log_n);

    (prover_setup, verifier_setup, prover_state, verifier_state)
}

#[test]
fn test_wrong_v1_final() {
    println!("=== Testing soundness: wrong v1' (final G1) ===");
    let mut rng = test_rng();
    let domain = b"scalar_product_test";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_scalar_product_test_environment(log_n);

    // Generate proof
    let builder = DoryProofBuilder::<
        G1Affine,
        G2AffineWrapper,
        Fq12,
        Fr,
        ToyTranscript,
    >::new_with_toy_transcript(domain);
    let mut proof_builder =
        inner_product_prove::<_, _, _, _, _, _, _, OptimizedMsmG1, OptimizedMsmG2>(
            builder,
            prover_state,
            &prover_setup,
            log_n,
        );

    if let Some(final_bases) = &mut proof_builder.final_bases {
        final_bases.v1_final = G1Affine::random(&mut rng);
        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);
        assert!(result.is_err(), "Verification should fail with wrong v1'");
    }
}

#[test]
fn test_wrong_v2_final() {
    println!("=== Testing soundness: wrong v2' (final G2) ===");
    let mut rng = test_rng();
    let domain = b"scalar_product_test";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_scalar_product_test_environment(log_n);

    // Generate proof
    let builder = DoryProofBuilder::<
        G1Affine,
        G2AffineWrapper,
        Fq12,
        Fr,
        ToyTranscript,
    >::new_with_toy_transcript(domain);
    let mut proof_builder =
        inner_product_prove::<_, _, _, _, _, _, _, OptimizedMsmG1, OptimizedMsmG2>(
            builder,
            prover_state,
            &prover_setup,
            log_n,
        );

    if let Some(final_bases) = &mut proof_builder.final_bases {
        final_bases.v2_final = G2AffineWrapper::random(&mut rng);
        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);
        assert!(result.is_err(), "Verification should fail with wrong v2'");
    }
}

#[test]
fn test_both_final_bases_wrong() {
    println!("=== Testing soundness: both final bases corrupted ===");
    let mut rng = test_rng();
    let domain = b"scalar_product_test";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_scalar_product_test_environment(log_n);

    // Generate proof
    let builder = DoryProofBuilder::<
        G1Affine,
        G2AffineWrapper,
        Fq12,
        Fr,
        ToyTranscript,
    >::new_with_toy_transcript(domain);
    let mut proof_builder =
        inner_product_prove::<_, _, _, _, _, _, _, OptimizedMsmG1, OptimizedMsmG2>(
            builder,
            prover_state,
            &prover_setup,
            log_n,
        );

    // Tamper with both v1' and v2'
    if let Some(final_bases) = &mut proof_builder.final_bases {
        println!("Tampering with both final bases v1' and v2'...");
        final_bases.v1_final = G1Affine::random(&mut rng);
        final_bases.v2_final = G2AffineWrapper::random(&mut rng);

        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

        assert!(
            result.is_err(),
            "Verification should fail with both final bases corrupted"
        );
    }
}

#[test]
fn test_scaled_final_bases() {
    println!("=== Testing soundness: scaled final bases v1', v2' ===");
    let mut rng = test_rng();
    let domain = b"scalar_product_test";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_scalar_product_test_environment(log_n);

    // Generate proof
    let builder = DoryProofBuilder::<
        G1Affine,
        G2AffineWrapper,
        Fq12,
        Fr,
        ToyTranscript,
    >::new_with_toy_transcript(domain);
    let mut proof_builder =
        inner_product_prove::<_, _, _, _, _, _, _, OptimizedMsmG1, OptimizedMsmG2>(
            builder,
            prover_state,
            &prover_setup,
            log_n,
        );

    // Scale both v1' and v2' by some factor
    if let Some(final_bases) = &mut proof_builder.final_bases {
        println!("Scaling v1' and v2'...");
        let scale = Fr::random(&mut rng);
        let scale_inv = scale.inv().unwrap();

        final_bases.v1_final = final_bases.v1_final.scale(&scale);
        final_bases.v2_final = final_bases.v2_final.scale(&scale_inv);

        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

        assert!(
            result.is_err(),
            "Verification should fail with scaled final bases"
        );
    }
}

#[test]
fn test_relationship_attack_final_bases() {
    println!("=== Testing soundness: relationship attack with final bases ===");
    let domain = b"scalar_product_test";
    let log_n = 8;

    // Generate two different test environments
    let (prover_setup1, _, prover_state1, _) = setup_scalar_product_test_environment(log_n);
    let (prover_setup2, verifier_setup, prover_state2, verifier_state) =
        setup_scalar_product_test_environment(log_n);

    // Generate proof for first state
    let builder1 = DoryProofBuilder::<
        G1Affine,
        G2AffineWrapper,
        Fq12,
        Fr,
        ToyTranscript,
    >::new_with_toy_transcript(domain);
    let proof1 = inner_product_prove::<_, _, _, _, _, _, _, OptimizedMsmG1, OptimizedMsmG2>(
        builder1,
        prover_state1,
        &prover_setup1,
        log_n,
    );

    // Generate proof for second state
    let builder2 = DoryProofBuilder::<
        G1Affine,
        G2AffineWrapper,
        Fq12,
        Fr,
        ToyTranscript,
    >::new_with_toy_transcript(domain);
    let mut proof2 = inner_product_prove::<_, _, _, _, _, _, _, OptimizedMsmG1, OptimizedMsmG2>(
        builder2,
        prover_state2,
        &prover_setup2,
        log_n,
    );

    // Mix final bases from different proof into proof2 and expect failure
    if let (Some(fb1), Some(fb2)) = (&proof1.final_bases, &mut proof2.final_bases) {
        println!("Mixing final bases from different proofs...");
        fb2.v1_final = fb1.v1_final.clone();
        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof2,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);
        assert!(result.is_err());
    }
}

#[test]
fn test_missing_final_bases() {
    println!("=== Testing soundness: missing final bases message ===");
    let domain = b"scalar_product_test";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_scalar_product_test_environment(log_n);

    // Generate proof
    let builder = DoryProofBuilder::<
        G1Affine,
        G2AffineWrapper,
        Fq12,
        Fr,
        ToyTranscript,
    >::new_with_toy_transcript(domain);
    let mut proof_builder =
        inner_product_prove::<_, _, _, _, _, _, _, OptimizedMsmG1, OptimizedMsmG2>(
            builder,
            prover_state,
            &prover_setup,
            log_n,
        );

    // Remove final bases; verification should fail
    proof_builder.final_bases = None;
    let verify_builder =
        DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
            proof_builder,
            ToyTranscript::new(domain),
        );
    let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);
    assert!(result.is_err());
}

#[test]
fn test_pairing_check_via_final_bases_tamper() {
    println!("=== Testing soundness: pairing equation check via final bases tamper ===");
    let mut rng = test_rng();
    let domain = b"scalar_product_test";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_scalar_product_test_environment(log_n);

    // Generate proof
    let builder = DoryProofBuilder::<
        G1Affine,
        G2AffineWrapper,
        Fq12,
        Fr,
        ToyTranscript,
    >::new_with_toy_transcript(domain);
    let mut proof_builder =
        inner_product_prove::<_, _, _, _, _, _, _, OptimizedMsmG1, OptimizedMsmG2>(
            builder,
            prover_state,
            &prover_setup,
            log_n,
        );

    if let Some(final_bases) = &mut proof_builder.final_bases {
        // Tamper v1' to break both linear and pairing batch
        final_bases.v1_final = G1Affine::random(&mut rng);
        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);
        assert!(result.is_err());
    }
}

#[test]
fn test_tamper_after_valid_rounds() {
    println!("=== Testing soundness: tampering final bases after valid rounds ===");
    let mut rng = test_rng();
    let domain = b"scalar_product_test";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_scalar_product_test_environment(log_n);

    // Generate a valid proof
    let builder = DoryProofBuilder::<
        G1Affine,
        G2AffineWrapper,
        Fq12,
        Fr,
        ToyTranscript,
    >::new_with_toy_transcript(domain);
    let mut proof_builder =
        inner_product_prove::<_, _, _, _, _, _, _, OptimizedMsmG1, OptimizedMsmG2>(
            builder,
            prover_state,
            &prover_setup,
            log_n,
        );

    if let Some(final_bases) = &mut proof_builder.final_bases {
        final_bases.v1_final = final_bases.v1_final.add(&G1Affine::random(&mut rng));
        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);
        assert!(result.is_err());
    }
}
