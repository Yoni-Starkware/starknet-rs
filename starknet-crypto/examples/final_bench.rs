//! Final apples-to-apples benchmark: stock `verify` vs optimized `verify_fast`,
//! averaged over many diverse valid signatures (varied keys and messages) so the
//! result is not tied to one input. Reports the min over several trials.

use std::hint::black_box;
use std::time::Instant;

use starknet_crypto::{
    get_public_key, rfc6979_generate_k, sign, verify, verify_batch_with_nonce_points, verify_fast,
    verify_with_pubkey_point, Felt,
};
use starknet_types_core::curve::AffinePoint;

const CASES: usize = 64;
const TRIALS: usize = 12;

fn build_cases() -> Vec<(Felt, Felt, Felt, Felt)> {
    let salt = Felt::from_hex("0x3c1e9550e66958296d11b60f8e8e7a7ad990d07fa65d5f7652c4a6c87d4e3cc").unwrap();
    (1u64..=CASES as u64)
        .map(|i| {
            let private_key = salt * Felt::from(i.wrapping_mul(0x9E3779B97F4A7C15));
            let message = Felt::from(i).pow(9u64) * Felt::from(0x2545F4914F6CDD1Du64);
            let public_key = get_public_key(&private_key);
            let k = rfc6979_generate_k(&message, &private_key, None);
            let sig = sign(&private_key, &message, &k).unwrap();
            (public_key, message, sig.r, sig.s)
        })
        .collect()
}

/// Min per-verify time over TRIALS, running the whole diverse case-set each trial.
fn min_per_verify<Case>(name: &str, cases: &[Case], mut verify_one: impl FnMut(&Case) -> bool) -> f64 {
    for case in cases {
        black_box(verify_one(case));
    }
    let mut best = f64::MAX;
    for _ in 0..TRIALS {
        let start = Instant::now();
        for case in cases {
            black_box(verify_one(black_box(case)));
        }
        best = best.min(start.elapsed().as_secs_f64() / cases.len() as f64);
    }
    println!("{name:<38} {:>8.2} us/verify   {:>10.0} verify/s", best * 1e6, 1.0 / best);
    best
}

fn main() {
    let cases = build_cases();

    // Correctness: every case verifies true on both, and the two agree.
    for (index, (pk, msg, r, s)) in cases.iter().enumerate() {
        let stock = verify(pk, msg, r, s).ok();
        let fast = verify_fast(pk, msg, r, s).ok();
        assert_eq!(stock, fast, "case {index} disagreement");
        assert_eq!(fast, Some(true), "case {index} should verify");
    }
    println!("verified {} diverse signatures; stock and verify_fast agree on all\n", cases.len());
    println!("profile: release, lto=fat, codegen-units=1, target-cpu=native\n");

    let stock = min_per_verify("stock verify (baseline)", &cases, |(pk, msg, r, s)| {
        verify(pk, msg, r, s).unwrap()
    });
    let fast = min_per_verify("verify_fast (optimized)", &cases, |(pk, msg, r, s)| {
        verify_fast(pk, msg, r, s).unwrap()
    });

    // Full-point variant: precompute the points once (models a caller that already
    // holds the full public key), then verify without any sqrt recovery.
    let points: Vec<(AffinePoint, Felt, Felt, Felt)> = cases
        .iter()
        .map(|(pk, msg, r, s)| (AffinePoint::new_from_x(pk, false).unwrap(), *msg, *r, *s))
        .collect();
    for (point, msg, r, s) in &points {
        assert_eq!(verify_with_pubkey_point(point, msg, r, s).ok(), Some(true));
    }
    let with_point = min_per_verify("verify_with_pubkey_point (full pk)", &points, |(point, msg, r, s)| {
        verify_with_pubkey_point(point, msg, r, s).unwrap()
    });

    println!("\noverall verify speedup (avg over {CASES} cases): {:.2}x", stock / fast);
    println!("with full pubkey point:                          {:.2}x", stock / with_point);

    // Batch MSM verification (signature carries the full nonce point R).
    let salt = Felt::from_hex("0x3c1e9550e66958296d11b60f8e8e7a7ad990d07fa65d5f7652c4a6c87d4e3cc")
        .unwrap();
    let batch_items: Vec<(AffinePoint, Felt, AffinePoint, Felt)> = (1u64..=1024)
        .map(|i| {
            let private_key = salt * Felt::from(i.wrapping_mul(0x2545_F491_4F6C_DD1D));
            let message = Felt::from(i).pow(7u64) * salt;
            let k = rfc6979_generate_k(&message, &private_key, None);
            let signature = sign(&private_key, &message, &k).unwrap();
            let generator = AffinePoint::generator();
            let generator_proj =
                starknet_types_core::curve::ProjectivePoint::from_affine(generator.x(), generator.y())
                    .unwrap();
            let nonce_point = (&generator_proj * k).to_affine().unwrap();
            let public_key_point = (&generator_proj * private_key).to_affine().unwrap();
            (public_key_point, message, nonce_point, signature.s)
        })
        .collect();
    let seed = [42u8; 32];
    println!("\nbatch MSM verify (verify_batch_with_nonce_points, all-or-nothing):");
    for batch_size in [256usize, 1024] {
        let subset = &batch_items[..batch_size];
        assert_eq!(verify_batch_with_nonce_points(subset, &seed).ok(), Some(true));
        let mut best = f64::MAX;
        for _ in 0..3 {
            let start = std::time::Instant::now();
            assert!(verify_batch_with_nonce_points(subset, &seed).unwrap());
            best = best.min(start.elapsed().as_secs_f64() / batch_size as f64);
        }
        println!(
            "  N={batch_size:<5} {:>7.2} us/sig   ({:.1}x vs single verify_with_pubkey_point)",
            best * 1e6,
            with_point / best
        );
    }
}
