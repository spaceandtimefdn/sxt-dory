use ark_bn254::{Fr, G1Affine};
use dory::arithmetic::{Field, Group, MultiScalarMul};
use dory::curve::{test_rng, OptimizedMsmG1};
use std::time::Instant;

fn main() {
    let mut rng = test_rng();

    // 2^22 = 4,194,304 elements
    let size = 1 << 22;
    println!("Generating {} random G1 points and Fr scalars...", size);

    let start_gen = Instant::now();

    // Generate random G1 points
    let bases: Vec<G1Affine> = (0..size).map(|_| G1Affine::random(&mut rng)).collect();

    // Generate random Fr scalars
    let scalars: Vec<Fr> = (0..size).map(|_| Fr::random(&mut rng)).collect();

    let gen_duration = start_gen.elapsed();
    println!("Generation took: {:?}", gen_duration);

    println!("Starting G1 MSM of size 2^22...");

    let start = Instant::now();
    let _result = OptimizedMsmG1::msm(&bases, &scalars);
    let duration = start.elapsed();

    println!("G1 MSM completed!");
    println!("Duration: {:?}", duration);
    println!(
        "Elements per second: {:.0}",
        size as f64 / duration.as_secs_f64()
    );
    println!(
        "Average time per element: {:.2} ns",
        duration.as_nanos() as f64 / size as f64
    );
}
