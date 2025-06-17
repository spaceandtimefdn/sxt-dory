use ark_bn254::{Bn254, G1Affine, G2Affine};
use ark_ec::{
    bn::{G1Prepared as BnG1Prepared, G2Prepared as BnG2Prepared},
    pairing::{MillerLoopOutput, Pairing as ArkPairing},
};
use dory::arithmetic::Group;
use dory::curve::{test_rng, G2AffineWrapper};
use std::time::Instant;

fn main() {
    let mut rng = test_rng();

    // Generate a single pair of points
    let g1_point = G1Affine::random(&mut rng);
    let g2_point = G2AffineWrapper::random(&mut rng);

    println!("Generated single G1 and G2 point pair");

    // Prepare points for Miller loop
    let prepare_start = Instant::now();
    let g1_prepared = BnG1Prepared::from(g1_point);
    let g2_affine: G2Affine = g2_point.into();
    let g2_prepared = BnG2Prepared::from(g2_affine);
    let prepare_duration = prepare_start.elapsed();

    println!("Point preparation time: {:?}", prepare_duration);

    // Test different iteration counts
    let iterations = vec![10000];

    for n in iterations {
        println!("\n=== Miller loop benchmark with {} iterations ===", n);

        // Benchmark n iterations of the same Miller loop
        let miller_start = Instant::now();
        for _ in 0..n {
            let _ml_result =
                Bn254::multi_miller_loop(vec![g1_prepared.clone()], vec![g2_prepared.clone()]);
        }
        let miller_duration = miller_start.elapsed();

        println!("Total duration for {} iterations: {:?}", n, miller_duration);
        println!(
            "Miller loops per second: {:.0}",
            n as f64 / miller_duration.as_secs_f64()
        );
        println!(
            "Average time per Miller loop: {:.2} μs",
            miller_duration.as_micros() as f64 / n as f64
        );

        // Do one final exponentiation for reference
        if n == 1 {
            let ml_result =
                Bn254::multi_miller_loop(vec![g1_prepared.clone()], vec![g2_prepared.clone()]);
            let final_exp_start = Instant::now();
            let _final_result = Bn254::final_exponentiation(ml_result)
                .expect("Final exponentiation should not fail");
            let final_exp_duration = final_exp_start.elapsed();

            println!("Final exponentiation duration: {:?}", final_exp_duration);
            println!(
                "Miller loop vs Final exp ratio: {:.2}:1",
                (miller_duration.as_secs_f64() / n as f64) / final_exp_duration.as_secs_f64()
            );
        }
    }
}
