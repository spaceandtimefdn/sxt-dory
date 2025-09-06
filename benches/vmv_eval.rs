#![allow(missing_docs)]
//! Criterion benches for VMV evaluation proving and verification only.
use ark_bn254::{Fr, Fq12};
use ark_ff::UniformRand;
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use dory::curve::{test_rng, ArkBn254Pairing, OptimizedMsmG1, OptimizedMsmG2, StandardPolynomial, DummyMsm};
use dory::setup::ProverSetup;
use dory::toy_transcript::ToyTranscript;
use dory::vmv::evaluate::{create_evaluation_proof, verify_evaluation_proof};
use dory::vmv::{commit_to_rows, compute_nu};

/// Benchmark the opening evaluation proof
/// Benchmark the opening evaluation proof
fn bench_vmv_eval(c: &mut Criterion) {
    // Deterministic RNG for reproducibility across all cases
    let mut rng = test_rng();

    // Use the exact same parameters as the working test
    let dims: &[usize] = &[9]; // d = log2(length), use 9 like the test

    for &d in dims {
        let length: usize = 1usize << d;

        // Fixed polynomial and evaluation point for this dimension (shared across sigma values)
        let a: Vec<Fr> = core::iter::repeat_with(|| Fr::rand(&mut rng))
            .take(length)
            .collect();
        let b_points: Vec<Fr> = core::iter::repeat_with(|| Fr::rand(&mut rng))
            .take(d)
            .collect();
        let polynomial = StandardPolynomial::new(&a);

        // Choose diverse sigma values while keeping SRS size moderate
        // Use roughly d/4, d/3, d/2 (rounded down, unique, >= 1)
        let mut sigma_candidates = vec![
            core::cmp::max(1, d / 4),
            core::cmp::max(1, d / 3),
            core::cmp::max(1, d / 2),
        ];
        sigma_candidates.sort_unstable();
        sigma_candidates.dedup();

        // Current protocol implementation requires d <= 2*sigma (so that nu <= sigma)
        // Otherwise internal vector splits panic (expects length 2^nu across v1,v2,s1,s2)
        let mut valid_sigmas: Vec<usize> = sigma_candidates
            .into_iter()
            .filter(|&s| d <= 2 * s)
            .collect();
        if valid_sigmas.is_empty() {
            valid_sigmas.push(core::cmp::max(1, d / 2));
        }

        for &sigma in &valid_sigmas {
            // Compute nu from (d, sigma) to define rows of the VMV matrix
            let nu: usize = compute_nu(d, sigma);

            // SRS size parameter: same as the working test
            let max_log_n: usize = 9;

            // Build setup outside timing
            let prover_setup = ProverSetup::<ArkBn254Pairing>::new(&mut rng, max_log_n);

            // Precompute padded row commitments OUTSIDE timing (do not time commitment)
            let row_commits = commit_to_rows::<ArkBn254Pairing, OptimizedMsmG1, _>(
                &polynomial,
                sigma,
                nu,
                &prover_setup,
            );

            let bench_name = format!("vmv_eval_prove_2^{}_sigma{}", d, sigma);
            let domain: Vec<u8> = format!("{}_{}__domain", bench_name, std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()).into_bytes();
            c.bench_function(&bench_name, |b| {
                // Only measure proof creation; clone inputs in setup phase per-iter
                b.iter_batched(
                    || row_commits.clone(),
                    |rc| {
                        let transcript = ToyTranscript::new(&domain);
                        let proof = create_evaluation_proof::<
                            ArkBn254Pairing,
                            ToyTranscript,
                            OptimizedMsmG1,
                            OptimizedMsmG2,
                            _,
                        >(
                            transcript,
                            black_box(&polynomial),
                            Some(black_box(rc)),
                            black_box(&b_points),
                            sigma,
                            black_box(&prover_setup),
                        );
                        black_box(proof);
                    },
                    BatchSize::SmallInput,
                )
            });

            // Prepare data for verification bench using the same setup and data as proving
            let verifier_setup = prover_setup.to_verifier_setup();

            let verify_bench_name = format!("vmv_eval_verify_2^{}_sigma{}", d, sigma);

            // Compute commitment batch and evaluations for verification using same data
            let (commitment_batch, batching_factors, evaluations) = dory::curve::commit_and_evaluate_batch::<
                ArkBn254Pairing,
                OptimizedMsmG1,
                Fr,
                <ArkBn254Pairing as dory::arithmetic::Pairing>::G1,
            >(&polynomial, &b_points, 0, sigma, &prover_setup);

            c.bench_function(&verify_bench_name, |b| {
                // Only measure verification; create fresh proof and transcript per-iter
                b.iter_batched(
                    || {
                        let transcript = ToyTranscript::new(&domain);
                        let proof = create_evaluation_proof::<
                            ArkBn254Pairing,
                            ToyTranscript,
                            OptimizedMsmG1,
                            OptimizedMsmG2,
                            _,
                        >(
                            transcript,
                            &polynomial,
                            None,
                            &b_points,
                            sigma,
                            &prover_setup,
                        );
                        proof
                    },
                    |proof_for_iter| {
                        let v_transcript = ToyTranscript::new(&domain);
                        let res = verify_evaluation_proof::<
                            ArkBn254Pairing,
                            ToyTranscript,
                            OptimizedMsmG1,
                            OptimizedMsmG2,
                            DummyMsm<Fq12>,
                        >(
                            black_box(proof_for_iter),
                            black_box(&commitment_batch),
                            black_box(&batching_factors),
                            black_box(&evaluations),
                            black_box(&b_points),
                            sigma,
                            &verifier_setup,
                            v_transcript,
                        );
                        res.expect("Verification should succeed");
                        black_box(());
                    },
                    BatchSize::SmallInput,
                )
            });
        }
    }
}

criterion_group!(benches, bench_vmv_eval);
criterion_main!(benches);
