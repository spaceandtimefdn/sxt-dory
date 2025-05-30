//! Data structures and generation of the transparent setup for both prover and verifier
use crate::arithmetic::*;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::RngCore;
use std::fs::File;
use std::io::{Read, Write};

/// Dory transparent setup for the prover
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ProverSetup<E: Pairing> {
    /// Γ₁  — column generators (|Γ₁| = n)
    pub g1_vec: Vec<E::G1>,
    /// Γ₂  — row generators    (|Γ₂| = n)
    pub g2_vec: Vec<E::G2>,
    /// H₁  — Pedersen/AFGHO blinding in G1
    pub h1: E::G1,
    /// H₂  — Pedersen/AFGHO blinding in G2
    pub h2: E::G2,
    /// e(H₁, H₂) cached once   
    pub ht: E::GT,
    /// gamma_fin in the paper
    pub g_fin: E::G2,
}

/// Dory transparent setup for the verifier with precomputed values
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct VerifierSetup<E: Pairing> {
    /// Δ₁L[k] = e(Γ₁[..2^(k-1)], Γ₂[..2^(k-1)])
    pub delta_1l: Vec<E::GT>,
    /// Δ₁R[k] = e(Γ₁[2^(k-1)..2^k], Γ₂[..2^(k-1)])
    pub delta_1r: Vec<E::GT>,
    /// Δ₂L[k] = same as Δ₁L[k] -- see paper
    pub delta_2l: Vec<E::GT>,
    /// Δ₂R[k] = e(Γ₁[..2^(k-1)], Γ₂[2^(k-1)..2^k]
    pub delta_2r: Vec<E::GT>,
    /// χ[k] = e(Γ₁[..2^k], Γ₂[..2^k])
    pub chi: Vec<E::GT>,

    /// First element of Γ₁
    pub g1_0: E::G1,
    /// First element of Γ₂
    pub g2_0: E::G2,
    /// H₁ — blinding generator
    pub h1: E::G1,
    /// H₂ — blinding generator
    pub h2: E::G2,
    /// e(H₁, H₂)
    pub ht: E::GT,
    /// $\Gamma_fin$ in the paper
    pub g_fin: E::G2,

    /// maximum # of coeffs 2^max_log_n
    pub max_log_n: usize,
}

impl<E: Pairing> ProverSetup<E> {
    /// Constructor for new prover setup
    pub fn new<R: RngCore>(mut rng: R, max_log_n: usize) -> Self {
        // We want \sqrt(n) generators for each of G1, G2
        // We take max_log_n + 1 to effectively take ceiling for odd max_log_n cases
        let n = 1usize << ((max_log_n + 1) / 2);

        let g1_vec: Vec<_> = (0..n).map(|_| E::G1::random(&mut rng)).collect();
        let g2_vec: Vec<_> = (0..n).map(|_| E::G2::random(&mut rng)).collect();
        let g_fin = E::G2::random(&mut rng);
        let h1 = E::G1::random(&mut rng);
        let h2 = E::G2::random(&mut rng);
        let ht = E::pair(&h1, &h2);

        Self {
            g1_vec,
            g2_vec,
            h1,
            h2,
            ht,
            g_fin,
        }
    }
    /// Convert to verifier side
    pub fn to_verifier_setup(&self) -> VerifierSetup<E> {
        VerifierSetup::from_prover_setup(self)
    }

    /// Save the prover setup to disk (legacy method - saves only prover setup)
    pub fn save_to_file(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::create(filename)?;
        let mut buffer = Vec::new();
        self.serialize_compressed(&mut buffer)?;
        file.write_all(&buffer)?;
        file.flush()?;
        println!(
            "Saved prover setup to {} ({} bytes)",
            filename,
            buffer.len()
        );
        Ok(())
    }

    /// Save both prover and verifier setups to disk in a combined format
    pub fn save_combined_to_file(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::create(filename)?;

        // Write magic marker for combined format
        file.write_all(b"DORY_COMBINED_SRS")?;

        // Serialize prover setup
        let mut prover_buffer = Vec::new();
        self.serialize_compressed(&mut prover_buffer)?;

        // Write prover setup length and data
        let prover_len = prover_buffer.len() as u64;
        file.write_all(&prover_len.to_le_bytes())?;
        file.write_all(&prover_buffer)?;

        // Generate and serialize verifier setup
        let verifier_setup = self.to_verifier_setup();
        let mut verifier_buffer = Vec::new();
        verifier_setup.serialize_compressed(&mut verifier_buffer)?;

        // Write verifier setup length and data
        let verifier_len = verifier_buffer.len() as u64;
        file.write_all(&verifier_len.to_le_bytes())?;
        file.write_all(&verifier_buffer)?;

        file.flush()?;
        println!(
            "Saved combined prover+verifier setup to {} (prover: {} bytes, verifier: {} bytes)",
            filename,
            prover_buffer.len(),
            verifier_buffer.len()
        );
        Ok(())
    }

    /// Load a prover setup from disk (handles both legacy and combined formats)
    pub fn load_from_file(filename: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::open(filename)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        // Check if this is a combined format file
        if buffer.len() >= 17 && &buffer[0..17] == b"DORY_COMBINED_SRS" {
            // Combined format - extract prover setup
            let mut offset = 17;

            // Read prover setup length
            if buffer.len() < offset + 8 {
                return Err("Invalid combined SRS file format".into());
            }
            let prover_len = u64::from_le_bytes([
                buffer[offset],
                buffer[offset + 1],
                buffer[offset + 2],
                buffer[offset + 3],
                buffer[offset + 4],
                buffer[offset + 5],
                buffer[offset + 6],
                buffer[offset + 7],
            ]) as usize;
            offset += 8;

            // Read prover setup data
            if buffer.len() < offset + prover_len {
                return Err("Invalid combined SRS file format".into());
            }
            let prover_data = &buffer[offset..offset + prover_len];
            let setup = Self::deserialize_compressed(prover_data)?;
            println!("Loaded prover setup from combined format file {}", filename);
            Ok(setup)
        } else {
            // Legacy format - entire file is prover setup
            let setup = Self::deserialize_compressed(&buffer[..])?;
            println!("Loaded prover setup from legacy format file {}", filename);
            Ok(setup)
        }
    }
}

impl<E: Pairing> VerifierSetup<E> {
    /// Constructor from an existing prover setup
    pub fn from_prover_setup(prover_setup: &ProverSetup<E>) -> Self {
        // Since g1_vec has length n = 1 << (max_log_n / 2), we have max_log_n = 2 * log2(g1_vec.len())
        let max_log_n = prover_setup.g1_vec.len().trailing_zeros() as usize;

        let mut delta_1l = Vec::with_capacity(max_log_n + 1);
        let mut delta_1r = Vec::with_capacity(max_log_n + 1);
        let mut delta_2r = Vec::with_capacity(max_log_n + 1);
        let mut chi = Vec::with_capacity(max_log_n + 1);

        for k in 0..=max_log_n {
            if k == 0 {
                delta_1l.push(E::GT::identity());
                delta_1r.push(E::GT::identity());
                delta_2r.push(E::GT::identity());
                chi.push(E::pair(&prover_setup.g1_vec[0], &prover_setup.g2_vec[0]));
            } else {
                let half_len = 1 << (k - 1);
                let full_len = 1 << k;

                let g1_first_half = &prover_setup.g1_vec[..half_len];
                let g1_second_half = &prover_setup.g1_vec[half_len..full_len];
                let g2_first_half = &prover_setup.g2_vec[..half_len];
                let g2_second_half = &prover_setup.g2_vec[half_len..full_len];

                // Δ₁L[k] = Δ₂L[k] = e(Γ₁[..2^(k-1)], Γ₂[..2^(k-1)])
                delta_1l.push(E::multi_pair(g1_first_half, g2_first_half));

                // Δ₁R[k] = e(Γ₁[2^(k-1)..2^k], Γ₂[..2^(k-1)])
                delta_1r.push(E::multi_pair(g1_second_half, g2_first_half));

                // Δ₂R[k] = e(Γ₁[..2^(k-1)], Γ₂[2^(k-1)..2^k])
                delta_2r.push(E::multi_pair(g1_first_half, g2_second_half));

                // χ[k] = e(Γ₁[..2^k], Γ₂[..2^k])
                chi.push(E::multi_pair(
                    &prover_setup.g1_vec[..full_len],
                    &prover_setup.g2_vec[..full_len],
                ));
            }
        }

        Self {
            delta_1l: delta_1l.clone(),
            delta_1r,
            delta_2l: delta_1l, // Delta_2L is the same as Delta_1L
            delta_2r,
            chi,
            g1_0: prover_setup.g1_vec[0].clone(),
            g2_0: prover_setup.g2_vec[0].clone(),
            h1: prover_setup.h1.clone(),
            h2: prover_setup.h2.clone(),
            ht: prover_setup.ht.clone(),
            max_log_n,
            g_fin: prover_setup.g_fin.clone(),
        }
    }

    /// Create a new verifier setup
    pub fn new<R: RngCore>(rng: R, max_log_n: usize) -> Self {
        let prover_setup = ProverSetup::new(rng, max_log_n);
        Self::from_prover_setup(&prover_setup)
    }

    /// Load a verifier setup from a combined format file
    pub fn load_from_file(filename: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::open(filename)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        // Check if this is a combined format file
        if buffer.len() >= 17 && &buffer[0..17] == b"DORY_COMBINED_SRS" {
            // Combined format - extract verifier setup
            let mut offset = 17;

            // Read prover setup length
            if buffer.len() < offset + 8 {
                return Err("Invalid combined SRS file format".into());
            }
            let prover_len = u64::from_le_bytes([
                buffer[offset],
                buffer[offset + 1],
                buffer[offset + 2],
                buffer[offset + 3],
                buffer[offset + 4],
                buffer[offset + 5],
                buffer[offset + 6],
                buffer[offset + 7],
            ]) as usize;
            offset += 8 + prover_len; // Skip prover data

            // Read verifier setup length
            if buffer.len() < offset + 8 {
                return Err("Invalid combined SRS file format".into());
            }
            let verifier_len = u64::from_le_bytes([
                buffer[offset],
                buffer[offset + 1],
                buffer[offset + 2],
                buffer[offset + 3],
                buffer[offset + 4],
                buffer[offset + 5],
                buffer[offset + 6],
                buffer[offset + 7],
            ]) as usize;
            offset += 8;

            // Read verifier setup data
            if buffer.len() < offset + verifier_len {
                return Err("Invalid combined SRS file format".into());
            }
            let verifier_data = &buffer[offset..offset + verifier_len];
            let setup = Self::deserialize_compressed(verifier_data)?;
            println!(
                "Loaded verifier setup from combined format file {}",
                filename
            );
            Ok(setup)
        } else {
            return Err("File is not in combined format - verifier setup not available".into());
        }
    }
}


