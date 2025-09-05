use ark_bn254::Fr;
use ark_ff::UniformRand;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dory::curve::{test_rng, ArkBn254Pairing, OptimizedMsmG1, OptimizedMsmG2, StandardPolynomial};
use dory::setup::ProverSetup;
use dory::toy_transcript::ToyTranscript;
use dory::vmv::evaluate::create_evaluation_proof;

fn bench_vmv_eval(c: &mut Criterion) {
    // Parameters: small and fast to start
    let length: usize = 1 << 10; // number of coefficients
    let max_log_n: usize = 10;   // SRS size
    let sigma: usize = 5;        // matrix width log2

    // Derived
    let nu: usize = length.next_power_of_two().trailing_zeros() as usize;

    // Deterministic RNG for reproducibility
    let mut rng = test_rng();

    // Setup and test inputs (outside iter to exclude from timing)
    let prover_setup = ProverSetup::<ArkBn254Pairing>::new(&mut rng, max_log_n);
    let a: Vec<Fr> = core::iter::repeat_with(|| Fr::rand(&mut rng))
        .take(length)
        .collect();
    let b_points: Vec<Fr> = core::iter::repeat_with(|| Fr::rand(&mut rng))
        .take(nu)
        .collect();
    let polynomial = StandardPolynomial::new(&a);

    c.bench_function("vmv_eval_prove_2^10_sigma5", |b| {
        b.iter(|| {
            let transcript = ToyTranscript::new(b"bench_vmv_eval");
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
            black_box(proof);
        })
    });
}

criterion_group!(benches, bench_vmv_eval);
criterion_main!(benches);
