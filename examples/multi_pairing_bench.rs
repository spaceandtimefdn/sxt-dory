use ark_bn254::{Fr, G1Affine};
use dory::arithmetic::{Group, Pairing};
use dory::curve::{test_rng, ArkBn254Pairing, G2AffineWrapper};
use std::time::Instant;

fn main() {
    let mut rng = test_rng();

    // Test different sizes for multi-pairing
    let sizes = vec![2048];

    for size in sizes {
        println!("\n=== Multi-pairing benchmark with {} pairs ===", size);

        // Generate random G1 points
        let g1_points: Vec<G1Affine> = (0..size).map(|_| G1Affine::random(&mut rng)).collect();

        // Generate random G2 points
        let g2_points: Vec<G2AffineWrapper> = (0..size)
            .map(|_| G2AffineWrapper::random(&mut rng))
            .collect();

        println!("Generated {} G1 and G2 point pairs", size);

        // Benchmark multi_pair
        let start = Instant::now();
        let _result = ArkBn254Pairing::multi_pair(&g1_points, &g2_points);
        let duration = start.elapsed();

        println!("Multi-pairing duration: {:?}", duration);
        println!(
            "Pairings per second: {:.0}",
            size as f64 / duration.as_secs_f64()
        );
        println!(
            "Average time per pairing: {:.2} μs",
            duration.as_micros() as f64 / size as f64
        );

        // Compare with individual pairings for smaller sizes
        if size <= 100 {
            let start_individual = Instant::now();
            let mut individual_result = ArkBn254Pairing::pair(&g1_points[0], &g2_points[0]);
            for i in 1..size {
                let pair_result = ArkBn254Pairing::pair(&g1_points[i], &g2_points[i]);
                individual_result = individual_result.add(&pair_result);
            }
            let individual_duration = start_individual.elapsed();

            println!("Individual pairings duration: {:?}", individual_duration);
            println!(
                "Speedup: {:.2}x",
                individual_duration.as_secs_f64() / duration.as_secs_f64()
            );
        }
    }
}
