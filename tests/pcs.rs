#![allow(missing_docs)]
use ark_bn254::Fr;
use ark_ff::UniformRand;
use dory::*;
use std::time::Instant;

use dory::curve::{
    test_rng, ArkBn254Pairing, DummyMsm, OptimizedMsmG1, OptimizedMsmG2, StandardPolynomial,
};

#[test]
fn test_pcs_api_workflow() {

    let mut rng = test_rng();
    let domain = b"dory_pcs_test";

    // Multilinear polynomial parameters
    let num_variables = 10;
    let sigma = 5; // sigma must be <= max_log_n / 2 for the SRS
    let num_coeffs = 1 << num_variables;

    println!(
        "Testing PCS API with {} variables, {} coefficients, sigma = {}",
        num_variables, num_coeffs, sigma
    );

    // Setup with preloaded SRS file
    let setup_start = Instant::now();
    let srs_path = "./k_5.srs";
    let (mut prover_setup, verifier_setup) =
        setup_with_srs_file::<ArkBn254Pairing, _>(&mut rng, num_variables, Some(srs_path));

    // Initialize cache for performance optimization
    // Try to load cache from disk first, if not found, generate and save
    let g1_cache_path = format!("./g1_cache_k_{}.bin", sigma);
    let g2_cache_path = format!("./g2_cache_k_{}.bin", sigma);
    
    println!("Checking for cached generators...");
    
    // Check if both cache files exist
    if std::path::Path::new(&g1_cache_path).exists() && std::path::Path::new(&g2_cache_path).exists() {
        // Print file sizes
        if let Ok(g1_metadata) = std::fs::metadata(&g1_cache_path) {
            println!("G1 cache file size: {:.2} MB", g1_metadata.len() as f64 / 1_048_576.0);
        }
        if let Ok(g2_metadata) = std::fs::metadata(&g2_cache_path) {
            println!("G2 cache file size: {:.2} MB", g2_metadata.len() as f64 / 1_048_576.0);
        }
        
        println!("Found cache files, loading from disk...");
        let load_start = Instant::now();
        if let Err(e) = prover_setup.load_cache_from_files(&g1_cache_path, &g2_cache_path) {
            println!("Failed to load cache: {}. Regenerating...", e);
            prover_setup.init_cache();
            
            // Save the newly generated cache
            if let Err(e) = prover_setup.save_cache_to_files(&g1_cache_path, &g2_cache_path) {
                println!("Warning: Failed to save cache: {}", e);
            } else {
                println!("✓ Cache saved to {} and {}", g1_cache_path, g2_cache_path);
            }
        } else {
            let load_time = load_start.elapsed();
            println!("✓ Cache loaded successfully from disk in {:?}", load_time);
        }
    } else {
        println!("Cache files not found, generating new cache...");
        let cache_gen_start = Instant::now();
        prover_setup.init_cache();
        let cache_gen_time = cache_gen_start.elapsed();
        println!("✓ Cache generated in {:?}", cache_gen_time);
        
        // Save the cache for future runs
        let save_start = Instant::now();
        if let Err(e) = prover_setup.save_cache_to_files(&g1_cache_path, &g2_cache_path) {
            println!("Warning: Failed to save cache: {}", e);
        } else {
            let save_time = save_start.elapsed();
            println!("✓ Cache saved to {} and {} in {:?}", g1_cache_path, g2_cache_path, save_time);
            
            // Print file sizes after saving
            if let Ok(g1_metadata) = std::fs::metadata(&g1_cache_path) {
                println!("  G1 cache file size: {:.2} MB", g1_metadata.len() as f64 / 1_048_576.0);
            }
            if let Ok(g2_metadata) = std::fs::metadata(&g2_cache_path) {
                println!("  G2 cache file size: {:.2} MB", g2_metadata.len() as f64 / 1_048_576.0);
            }
        }
    }
    
    println!(
        "Cache initialization complete. Has cache: {}",
        prover_setup.has_cache()
    );

    let setup_time = setup_start.elapsed();
    println!("Setup time (including cache): {:?}", setup_time);

    // Random multilinear polynomial coefficients
    let coeffs: Vec<Fr> = (0..num_coeffs).map(|_| Fr::rand(&mut rng)).collect();

    // Random evaluation point (one value per variable)
    let point: Vec<Fr> = (0..num_variables).map(|_| Fr::rand(&mut rng)).collect();

    // Commit to polynomial
    let commit_start = Instant::now();
    let polynomial = StandardPolynomial::new(&coeffs);
    let commitment =
        commit::<ArkBn254Pairing, OptimizedMsmG1, _>(&polynomial, 0, sigma, &prover_setup);
    let commit_time = commit_start.elapsed();
    println!("Commit time: {:?}", commit_time);

    // Evaluate and prove
    let eval_start = Instant::now();
    let transcript = create_transcript(domain);
    let (evaluation, proof) = evaluate::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, _>(
        &StandardPolynomial::new(&coeffs),
        &point,
        sigma,
        &prover_setup,
        transcript,
    );
    let eval_time = eval_start.elapsed();
    println!("Evaluate and prove time: {:?}", eval_time);

    // Print proof statistics before verification consumes it
    proof.print_proof_stats();

    // Verify - create fresh transcript for verification
    let verify_start = Instant::now();
    let verify_transcript = create_transcript(domain);
    let result = verify::<ArkBn254Pairing, _, OptimizedMsmG1, OptimizedMsmG2, DummyMsm<_>>(
        commitment,
        evaluation,
        &point,
        proof,
        sigma,
        &verifier_setup,
        verify_transcript,
    );
    let verify_time = verify_start.elapsed();
    println!("Verify time: {:?}", verify_time);

    let total_time = setup_time + commit_time + eval_time + verify_time;
    println!("Total time: {:?}", total_time);

    assert!(result.is_ok(), "PCS verification should succeed");
    println!("✓ PCS API test passed");
}
