use ark_bn254::{Fr, Fq12};
use ark_ff::UniformRand;
use dory::curve::{test_rng, ArkBn254Pairing, OptimizedMsmG1, OptimizedMsmG2, StandardPolynomial, DummyMsm};
use dory::setup::ProverSetup;
use dory::toy_transcript::ToyTranscript;
use dory::vmv::evaluate::{create_evaluation_proof, verify_evaluation_proof};
use dory::vmv::{commit_to_rows, compute_nu};

fn main() {
    let mut rng = test_rng();
    
    // Simple test case
    let d = 20;
    let sigma = 10;
    let length: usize = 1usize << d;
    
    let a: Vec<Fr> = core::iter::repeat_with(|| Fr::rand(&mut rng))
        .take(length)
        .collect();
    let b_points: Vec<Fr> = core::iter::repeat_with(|| Fr::rand(&mut rng))
        .take(d)
        .collect();
    let polynomial = StandardPolynomial::new(&a);
    
    let nu: usize = compute_nu(d, sigma);
    let max_log_n: usize = 2 * nu;
    let prover_setup = ProverSetup::<ArkBn254Pairing>::new(&mut rng, max_log_n);
    
    let row_commits = commit_to_rows::<ArkBn254Pairing, OptimizedMsmG1, _>(
        &polynomial,
        sigma,
        nu,
        &prover_setup,
    );
    
    let domain = b"test_domain";
    let transcript = ToyTranscript::new(domain);
    let proof = create_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        _,
    >(
        transcript,
        &polynomial,
        Some(row_commits.clone()),
        &b_points,
        sigma,
        &prover_setup,
    );
    
    let verifier_setup = prover_setup.to_verifier_setup();
    let (commitment_batch, batching_factors, evaluations) = dory::curve::commit_and_evaluate_batch::<
        ArkBn254Pairing,
        OptimizedMsmG1,
        Fr,
        <ArkBn254Pairing as dory::arithmetic::Pairing>::G1,
    >(&polynomial, &b_points, 0, sigma, &prover_setup);
    
    let v_transcript = ToyTranscript::new(domain);
    let res = verify_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        DummyMsm<Fq12>,
    >(
        proof,
        &commitment_batch,
        &batching_factors,
        &evaluations,
        &b_points,
        sigma,
        &verifier_setup,
        v_transcript,
    );
    
    match res {
        Ok(()) => println!("Verification succeeded!"),
        Err(e) => println!("Verification failed: {:?}", e),
    }
}
