#![allow(missing_docs)]
use std::time::Instant;

use ark_bn254::{Fq12, Fr, G1Affine, G2Affine};
use ark_ff::UniformRand;
use dory::{
    arithmetic::{MultiScalarMul, Pairing},
    builder::{DoryProofBuilder, DoryVerifyBuilder},
    inner_product::{inner_product_prove, inner_product_verify},
    setup::ProverSetup,
    state::{DoryProverState, DoryVerifierState},
    toy_transcript::ToyTranscript,
};

use dory::curve::{test_rng, ArkBn254Pairing, G2AffineWrapper, OptimizedMsmG1, OptimizedMsmG2};

#[test]
fn test_inner_product_verify_should_fail() {
    println!("Starting failing verification test...");
    let total_start = Instant::now();

    // Create deterministic RNG for testing
    let mut rng = test_rng();

    // Test parameters
    let domain = b"test_domain";
    let log_n = 9; // Use a smaller size for faster testing
    let vector_size = 1 << log_n;
    println!("Vector size: {}", vector_size);

    // ----- Setup phase -----
    println!("Creating setup...");
    let prover_setup = ProverSetup::<ArkBn254Pairing>::new(&mut rng, 2 * log_n);
    let verifier_setup = prover_setup.to_verifier_setup();

    // ----- Vector generation phase -----
    println!("Generating random vectors...");
    // Generate random vectors for prover state
    let v1: Vec<G1Affine> = (0..vector_size).map(|_| G1Affine::rand(&mut rng)).collect();
    let v2: Vec<G2AffineWrapper> = (0..vector_size)
        .map(|_| G2AffineWrapper::from(G2Affine::rand(&mut rng)))
        .collect();
    let s1: Vec<Fr> = (0..vector_size).map(|_| Fr::rand(&mut rng)).collect();
    let s2: Vec<Fr> = (0..vector_size).map(|_| Fr::rand(&mut rng)).collect();

    // ----- Initial state calculation phase -----
    println!("Creating initial states...");
    // Create initial state
    let prover_state = DoryProverState::new(v1.clone(), v2.clone(), s1.clone(), s2.clone(), log_n);

    // Create initial value for C (inner product of v1 and v2)
    let c = ArkBn254Pairing::multi_pair(&v1, &v2);

    // Create the initial values for D1 and D2
    let d_1 = ArkBn254Pairing::multi_pair(&v1, &prover_setup.g2_vec()[..1 << log_n]);
    let d_2 = ArkBn254Pairing::multi_pair(&prover_setup.g1_vec()[..1 << log_n], &v2);

    // Create the initial values for E1 and E2
    let e_1 = OptimizedMsmG1::msm(&prover_setup.g1_vec()[..1 << log_n], &s2);
    let e_2 = OptimizedMsmG2::msm(&prover_setup.g2_vec()[..1 << log_n], &s1);

    // Create verifier state
    let _verifier_state = DoryVerifierState::<ArkBn254Pairing>::new_with_s(
        c,
        d_1,
        d_2,
        e_1,
        e_2,
        s1.clone(),
        s2.clone(),
        log_n,
    );

    // ----- Proof generation phase -----
    println!("Generating proof...");

    // Create proof builder
    #[cfg(feature = "recursion")]
    let builder = DoryProofBuilder::<
        G1Affine,
        G2AffineWrapper,
        Fq12,
        Fr,
        ToyTranscript,
    >::new_with_toy_transcript(domain, &prover_setup);

    #[cfg(not(feature = "recursion"))]
    let builder = DoryProofBuilder::<
        G1Affine,
        G2AffineWrapper,
        Fq12,
        Fr,
        ToyTranscript,
    >::new_with_toy_transcript(domain);

    // Generate proof
    let proof_builder = inner_product_prove::<_, _, _, _, _, _, _, OptimizedMsmG1, OptimizedMsmG2>(
        builder,
        prover_state,
        &prover_setup,
        log_n,
    );

    // ----- Tamper with the proof -----
    println!("\n=== Testing tampered proofs ===");

    // Test Case 1: Tamper with a first message
    {
        println!("\n--- Test 1: Tampering with first reduce message ---");

        // Clone the proof builder and tamper with it
        let mut tampered_proof_builder = proof_builder.clone();

        if !tampered_proof_builder.first_messages.is_empty() {
            // Corrupt d1_left in the first message
            println!("Corrupting d1_left in first message...");
            let corrupt_d1_left = Fq12::rand(&mut rng);
            tampered_proof_builder.first_messages[0].d1_left = corrupt_d1_left;

            let verify_transcript = ToyTranscript::new(domain);

            // create a verifier
            let verify_builder = DoryVerifyBuilder::<
                G1Affine,
                G2AffineWrapper,
                Fq12,
                Fr,
                ToyTranscript,
            >::new_from_proof(
                tampered_proof_builder, verify_transcript
            );

            // Test verification
            println!("Verifying corrupted proof...");
            // Recreate verifier state since it doesn't implement Clone
            let verifier_state_copy = DoryVerifierState::<ArkBn254Pairing>::new_with_s(
                c,
                d_1,
                d_2,
                e_1,
                e_2,
                s1.clone(),
                s2.clone(),
                log_n,
            );
            let result =
                inner_product_verify(verify_builder, verifier_state_copy, &verifier_setup, log_n);

            println!("Verification result: {:?}", result);
            assert!(
                result.is_err(),
                "Corrupted first message should cause verification to fail"
            );
            if let Err(round) = result {
                println!("Verification correctly failed at round: {}", round);
            }
        }
    }

    println!("Total test time: {:?}", total_start.elapsed());
    println!("All tests completed successfully!");
}
