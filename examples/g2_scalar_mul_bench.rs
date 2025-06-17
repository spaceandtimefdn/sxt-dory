use ark_bn254::Fr;
use dory::arithmetic::{Field, Group};
use dory::curve::{test_rng, G2AffineWrapper};
use std::time::Instant;

fn main() {
    let mut rng = test_rng();

    // Generate a random G2 point
    let base_point = G2AffineWrapper::random(&mut rng);

    // Generate random scalars for testing
    let num_ops = 10000;
    let scalars: Vec<Fr> = (0..num_ops).map(|_| Fr::random(&mut rng)).collect();

    println!("Benchmarking {} G2 scalar multiplications...", num_ops);

    let start = Instant::now();

    for scalar in &scalars {
        let _result = base_point.scale(scalar);
    }

    let duration = start.elapsed();
    let ops_per_sec = num_ops as f64 / duration.as_secs_f64();

    println!("Duration: {:?}", duration);
    println!("Operations per second: {:.0}", ops_per_sec);
    println!(
        "Average time per operation: {:.2} μs",
        duration.as_micros() as f64 / num_ops as f64
    );
}
