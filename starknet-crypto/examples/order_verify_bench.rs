//! Benchmarks the exchange order-ingestion use case: an order of 10 felt fields
//! is hashed to a single message felt, and its STARK ECDSA signature (over that
//! exact hash) is verified.
//!
//! Full matrix: {stock verify, verify_fast, verify_with_pubkey_point}
//!            x {Poseidon(10 felts), blake2s(10 felts, Cairo encoding)}.
//!
//! Every arm verifies a REAL signature over the arm's own hash domain and
//! asserts the result is `true` inside the timed loop, so no arm can pass
//! while computing something trivial. Trials are interleaved round-robin
//! across all arms so thermal/frequency drift hits every arm equally.

use std::hint::black_box;
use std::time::Instant;

use starknet_crypto::{
    get_public_key, poseidon_hash_many, rfc6979_generate_k, sign, verify, verify_fast,
    verify_with_pubkey_point, Felt,
};
use starknet_types_core::curve::AffinePoint;

const ORDERS: usize = 2_000;
const TRIALS: usize = 8;
const THREAD_TRIALS: usize = 5;
const ORDER_FIELDS: usize = 10;

/// blake2s over the 10 order felts using Starknet's Cairo-compatible encoding
/// (starkware cairo_blake2s / starknet-types-core `Blake2Felt252`): each felt
/// encodes as 2 u32 words if < 2^63 or 8 u32 words with an MSB marker otherwise;
/// words serialize little-endian; the 256-bit digest packs into a Felt (mod p).
fn blake_hash_fields(fields: &[Felt; ORDER_FIELDS]) -> Felt {
    use blake2::{Blake2s256, Digest};
    const SMALL_THRESHOLD: Felt = Felt::from_hex_unchecked("0x8000000000000000");
    const BIG_MARKER: u32 = 1 << 31;

    let mut words: Vec<u32> = Vec::with_capacity(ORDER_FIELDS * 8);
    for field in fields {
        let bytes = field.to_bytes_be();
        if *field < SMALL_THRESHOLD {
            words.push(u32::from_be_bytes(bytes[24..28].try_into().unwrap()));
            words.push(u32::from_be_bytes(bytes[28..32].try_into().unwrap()));
        } else {
            let start = words.len();
            for chunk in bytes.chunks_exact(4) {
                words.push(u32::from_be_bytes(chunk.try_into().unwrap()));
            }
            words[start] |= BIG_MARKER;
        }
    }
    let mut byte_stream = Vec::with_capacity(words.len() * 4);
    for word in words {
        byte_stream.extend_from_slice(&word.to_le_bytes());
    }
    let mut hasher = Blake2s256::new();
    hasher.update(&byte_stream);
    let digest = hasher.finalize();
    let mut le_bytes = [0u8; 32];
    le_bytes.copy_from_slice(&digest);
    Felt::from_bytes_le(&le_bytes)
}

struct Order {
    fields: [Felt; ORDER_FIELDS],
    pubkey_x: Felt,
    point: AffinePoint,
    poseidon_r: Felt,
    poseidon_s: Felt,
    blake_r: Felt,
    blake_s: Felt,
}

/// Orders with varied full-width keys and field values; each carries one
/// signature over its Poseidon hash and one over its blake2s hash.
fn build_orders() -> Vec<Order> {
    let salt = Felt::from_hex("0x3c1e9550e66958296d11b60f8e8e7a7ad990d07fa65d5f7652c4a6c87d4e3cc")
        .unwrap();
    (1..=ORDERS as u64)
        .map(|i| {
            let private_key = salt * Felt::from(i.wrapping_mul(0x9e37_79b9_7f4a_7c15));
            let fields: [Felt; ORDER_FIELDS] =
                core::array::from_fn(|j| salt * Felt::from(i * 1000 + j as u64 + 1));
            let pubkey_x = get_public_key(&private_key);
            let point = AffinePoint::new_from_x(&pubkey_x, false).unwrap();

            let poseidon_message = poseidon_hash_many(&fields);
            let poseidon_k = rfc6979_generate_k(&poseidon_message, &private_key, None);
            let poseidon_sig = sign(&private_key, &poseidon_message, &poseidon_k).unwrap();

            let blake_message = blake_hash_fields(&fields);
            let blake_k = rfc6979_generate_k(&blake_message, &private_key, None);
            let blake_sig = sign(&private_key, &blake_message, &blake_k).unwrap();

            Order {
                fields,
                pubkey_x,
                point,
                poseidon_r: poseidon_sig.r,
                poseidon_s: poseidon_sig.s,
                blake_r: blake_sig.r,
                blake_s: blake_sig.s,
            }
        })
        .collect()
}

type ArmFunction<'a> = Box<dyn Fn(&Order) -> bool + 'a>;

fn median(mut times: Vec<f64>) -> f64 {
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

fn main() {
    let orders = build_orders();
    println!(
        "use case: order = {ORDER_FIELDS} felt fields; hash -> message felt -> STARK ECDSA verify"
    );
    println!(
        "{ORDERS} orders, varied keys/values; {TRIALS} trials interleaved round-robin \
         across arms; single thread\n"
    );

    let arms: Vec<(&str, ArmFunction<'_>)> = vec![
        ("poseidon_hash_many(10 felts) alone", Box::new(|o: &Order| {
            black_box(poseidon_hash_many(&o.fields));
            true
        })),
        ("blake2s(10 felts, Cairo encoding) alone", Box::new(|o: &Order| {
            black_box(blake_hash_fields(&o.fields));
            true
        })),
        ("poseidon + stock verify (x-only)", Box::new(|o: &Order| {
            let msg = poseidon_hash_many(&o.fields);
            verify(&o.pubkey_x, &msg, &o.poseidon_r, &o.poseidon_s).unwrap()
        })),
        ("poseidon + verify_fast (x-only)", Box::new(|o: &Order| {
            let msg = poseidon_hash_many(&o.fields);
            verify_fast(&o.pubkey_x, &msg, &o.poseidon_r, &o.poseidon_s).unwrap()
        })),
        ("poseidon + verify_with_pubkey_point (full pk)", Box::new(|o: &Order| {
            let msg = poseidon_hash_many(&o.fields);
            verify_with_pubkey_point(&o.point, &msg, &o.poseidon_r, &o.poseidon_s).unwrap()
        })),
        ("blake2s + stock verify (x-only)", Box::new(|o: &Order| {
            let msg = blake_hash_fields(&o.fields);
            verify(&o.pubkey_x, &msg, &o.blake_r, &o.blake_s).unwrap()
        })),
        ("blake2s + verify_fast (x-only)", Box::new(|o: &Order| {
            let msg = blake_hash_fields(&o.fields);
            verify_fast(&o.pubkey_x, &msg, &o.blake_r, &o.blake_s).unwrap()
        })),
        ("blake2s + verify_with_pubkey_point (full pk)", Box::new(|o: &Order| {
            let msg = blake_hash_fields(&o.fields);
            verify_with_pubkey_point(&o.point, &msg, &o.blake_r, &o.blake_s).unwrap()
        })),
    ];

    // Warm up every arm (also builds the one-time fixed-base G table).
    for (_, arm_function) in &arms {
        for order in orders.iter().take(200) {
            assert!(black_box(arm_function(order)));
        }
    }

    // Interleaved trials: each round times every arm once, so slow drift in
    // clock frequency or thermals affects all arms equally.
    let mut per_arm_times: Vec<Vec<f64>> = vec![Vec::with_capacity(TRIALS); arms.len()];
    for _ in 0..TRIALS {
        for (arm_index, (_, arm_function)) in arms.iter().enumerate() {
            let start = Instant::now();
            for order in &orders {
                assert!(black_box(arm_function(black_box(order))));
            }
            per_arm_times[arm_index].push(start.elapsed().as_secs_f64() / orders.len() as f64);
        }
    }

    let mut best_blake_point = f64::MAX;
    for ((name, _), times) in arms.iter().zip(&per_arm_times) {
        let minimum = times.iter().copied().fold(f64::MAX, f64::min);
        let med = median(times.clone());
        println!(
            "{name:<52} {:>7.2} us/order (min) {:>7.2} (median)   {:>9.0} orders/s/core",
            minimum * 1e6,
            med * 1e6,
            1.0 / minimum
        );
        if *name == "blake2s + verify_with_pubkey_point (full pk)" {
            best_blake_point = minimum;
        }
    }

    // Tampered signatures must fail on every implementation and in both hash
    // domains (guards against an arm accidentally short-circuiting to `true`).
    let mut tamper_checks = 0;
    for order in orders.iter().take(50) {
        let poseidon_bad = poseidon_hash_many(&order.fields) + Felt::ONE;
        let blake_bad = blake_hash_fields(&order.fields) + Felt::ONE;
        for (bad_message, r, s) in [
            (poseidon_bad, order.poseidon_r, order.poseidon_s),
            (blake_bad, order.blake_r, order.blake_s),
        ] {
            assert_eq!(verify(&order.pubkey_x, &bad_message, &r, &s).ok(), Some(false));
            assert_eq!(verify_fast(&order.pubkey_x, &bad_message, &r, &s).ok(), Some(false));
            assert_eq!(
                verify_with_pubkey_point(&order.point, &bad_message, &r, &s).ok(),
                Some(false)
            );
            tamper_checks += 3;
        }
    }
    println!("\ntamper check: {tamper_checks} corrupted verifications rejected on all paths, both hash domains");

    // Thread scaling on the best cell (median of THREAD_TRIALS runs);
    // verification is embarrassingly parallel.
    let single_thread_rate = 1.0 / best_blake_point;
    for n_threads in [4usize, 8, 12] {
        let mut rates = Vec::with_capacity(THREAD_TRIALS);
        for _ in 0..THREAD_TRIALS {
            let start = Instant::now();
            std::thread::scope(|scope| {
                for chunk in orders.chunks(orders.len() / n_threads + 1) {
                    scope.spawn(move || {
                        for o in chunk {
                            let msg = blake_hash_fields(&o.fields);
                            assert!(verify_with_pubkey_point(&o.point, &msg, &o.blake_r, &o.blake_s)
                                .unwrap());
                        }
                    });
                }
            });
            rates.push(orders.len() as f64 / start.elapsed().as_secs_f64());
        }
        let med = median(rates);
        println!(
            "{n_threads:>2} threads: {med:>9.0} orders/s median of {THREAD_TRIALS}  ({:.1}x over single core)",
            med / single_thread_rate
        );
    }
}
