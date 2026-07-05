//! Optimized STARK-curve ECDSA verification.
//!
//! # Public API
//!
//! | Function | Input contract | Semantics | Measured (one x86 laptop core) |
//! |---|---|---|---|
//! | [`verify_fast`] | x-only public key | drop-in for [`crate::verify`] | ~85 us (stock: ~350 us) |
//! | [`verify_with_pubkey_point`] | full public-key point | either y sign accepted | ~63 us |
//! | [`verify_batch`] | x-only, per-item results | same as `verify_fast` | amortizes only `s^-1` |
//! | [`verify_batch_with_nonce_points`] | full `Q` and nonce point `R` | exact equation, all-or-nothing | ~18-22 us/signature at N >= 256 |
//!
//! # Design
//!
//! Relative to the generic double-and-add in [`crate::verify`]:
//!   * `z*w*G` uses a precomputed fixed-base comb for the generator (one mixed
//!     addition per 7-bit window, zero doublings). The ~290 KB table is built
//!     once on first use (~4 ms) behind a `OnceLock`.
//!   * `r*w*Q` uses width-5 wNAF over Jacobian coordinates (`dbl-2007-bl`,
//!     1M+8S doubling with the curve's `a = 1` making `a*Z^4` a plain square;
//!     dedicated SOS squaring makes S < M).
//!   * Public-key y-recovery uses Cipolla's square root, which is O(log p);
//!     the STARK prime's 2-adicity of 192 makes the stock Tonelli-Shanks the
//!     single most expensive step of `verify` (~100 us).
//!   * Scalar-field arithmetic (`s^-1`, `z*w`, `r*w` mod n) runs on fixed-size
//!     `crypto-bigint` Montgomery residues with compile-time parameters.
//!   * The final `x == r` comparison happens projectively (`X == r*Z`), so no
//!     field inversion is spent converting to affine.
//!   * [`verify_batch_with_nonce_points`] checks all signatures at once: a
//!     random linear combination of the per-signature equations
//!     `(z*w)*G + (r*w)*Q - R = 0` is evaluated as one Pippenger bucket
//!     multi-scalar multiplication (soundness error ~2^-128 from 128-bit
//!     seed-derived coefficients).
//!
//! # Semantics
//!
//! The x-only paths preserve chain semantics exactly: the recovered public key
//! may have either y sign, so both `+/-` candidates of `zwG +/- rwQ` are
//! tried, and every input range check matches [`crate::verify`]. One deliberate
//! difference: when a candidate sum is the point at infinity, stock `verify`
//! panics (`to_affine().unwrap()`); this module returns `Ok(false)`, which is
//! the mathematically correct answer (infinity has no affine x-coordinate).
//!
//! The full-point paths verify the exact group equation for the supplied
//! `(Q, R)`; protocols using them must fix a y-parity convention to stay
//! equivalent to on-chain x-only acceptance.
//!
//! Everything here is single-threaded; concurrent callers are safe (the only
//! shared state is the immutable generator table).

use crypto_bigint::modular::runtime_mod::{DynResidue, DynResidueParams};
use crypto_bigint::{ArrayEncoding, U256};
use starknet_curve::curve_params::{ALPHA, BETA, GENERATOR};
use starknet_types_core::curve::{AffinePoint, ProjectivePoint};
use starknet_types_core::felt::Felt;
use std::sync::OnceLock;

use crate::VerifyError;

/// The STARK curve group order (`EC_ORDER`) as a `crypto-bigint` `U256`, used for
/// fixed-size Montgomery scalar-field arithmetic instead of heap-allocating num-bigint.
const SCALAR_MODULUS: U256 =
    U256::from_be_hex("0800000000000010ffffffffffffffffb781126dcae7b2321e66a241adc64d2f");

/// Montgomery parameters for `SCALAR_MODULUS`, computed at compile time so no
/// per-verify call re-derives R^2 and the modular negated inverse.
const SCALAR_PARAMS: DynResidueParams<{ U256::LIMBS }> = DynResidueParams::new(&SCALAR_MODULUS);

fn felt_to_u256(value: &Felt) -> U256 {
    U256::from_be_slice(&value.to_bytes_be())
}

fn u256_to_felt(value: &U256) -> Felt {
    Felt::from_bytes_be(&value.to_be_byte_array().into())
}

const ELEMENT_UPPER_BOUND: Felt = Felt::from_raw([
    576459263475450960,
    18446744073709255680,
    160989183,
    18446743986131435553,
]);

/// wNAF window width for the variable-base (`r*w*Q`) multiply.
const WINDOW: u32 = 5;

/// Window width (bits) for the fixed-base comb over G. 7 divides 252 evenly.
const G_WINDOW: usize = 7;
/// Number of windows: scalars are reduced mod `EC_ORDER < 2^252`, so 252 bits suffice.
const G_WINDOWS: usize = 252 / G_WINDOW;
/// Nonzero digits per window: `1..=2^G_WINDOW - 1`.
const G_DIGITS: usize = (1 << G_WINDOW) - 1;

/// Fixed-base comb table: `table[k][j] = (j + 1) * 2^(G_WINDOW*k) * G` (affine).
/// `scalar * G` is then one mixed add per window, with zero doublings.
fn fixed_base_g_table() -> &'static Vec<Vec<AffinePoint>> {
    static TABLE: OnceLock<Vec<Vec<AffinePoint>>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = Vec::with_capacity(G_WINDOWS);
        let mut window_base = ProjectivePoint::from_affine(GENERATOR.x(), GENERATOR.y()).unwrap();
        for _ in 0..G_WINDOWS {
            // Build the window's multiples projectively, then normalize the whole
            // batch with a single inversion (Montgomery's trick) rather than one
            // per point -- this is a one-time table build, but it keeps cold-start
            // to ~one inversion per window instead of `G_DIGITS`.
            let mut projective = Vec::with_capacity(G_DIGITS);
            let mut multiple = window_base.clone();
            for _ in 0..G_DIGITS {
                projective.push(multiple.clone());
                multiple += &window_base;
            }
            table.push(batch_to_affine(&projective));
            // Advance to the next window: multiply the base by 2^G_WINDOW.
            for _ in 0..G_WINDOW {
                window_base = window_base.double();
            }
        }
        table
    })
}

/// Normalize homogeneous projective points to affine with a single field inversion
/// for the whole slice (Montgomery's batch-inversion trick). Inputs must have Z != 0.
fn batch_to_affine(points: &[ProjectivePoint]) -> Vec<AffinePoint> {
    let mut prefix = Vec::with_capacity(points.len());
    let mut running = Felt::ONE;
    for point in points {
        running = running * point.z();
        prefix.push(running);
    }
    let mut inverse = running.inverse().expect("generator multiples have nonzero Z");
    let mut affine = vec![AffinePoint::identity(); points.len()];
    for index in (0..points.len()).rev() {
        let preceding = if index == 0 { Felt::ONE } else { prefix[index - 1] };
        let z_inverse = inverse * preceding;
        inverse = inverse * points[index].z();
        affine[index] = AffinePoint::new(points[index].x() * z_inverse, points[index].y() * z_inverse)
            .expect("normalized generator multiple is on curve");
    }
    affine
}

/// `scalar * G` via the fixed-base comb: one mixed add per window, zero doublings.
fn fixed_base_mul(scalar: &Felt) -> ProjectivePoint {
    let table = fixed_base_g_table();
    let bits = scalar.to_bits_le();
    let mut acc = ProjectivePoint::identity();
    for (window_index, multiples) in table.iter().enumerate() {
        let mut digit = 0usize;
        for bit_offset in 0..G_WINDOW {
            if bits[window_index * G_WINDOW + bit_offset] {
                digit |= 1 << bit_offset;
            }
        }
        if digit != 0 {
            acc += &multiples[digit - 1];
        }
    }
    acc
}

/// Signed-digit (wNAF) form of `scalar`, little-endian (index 0 = least significant).
/// Nonzero digits are odd and lie in `(-2^W, 2^W)`. Allocation-free arithmetic on
/// 256-bit little-endian limbs.
fn wnaf(scalar: &Felt, width: u32) -> Vec<i8> {
    let bytes = scalar.to_bytes_le();
    let mut limbs = [0u64; 4];
    for (index, limb) in limbs.iter_mut().enumerate() {
        *limb = u64::from_le_bytes(bytes[index * 8..index * 8 + 8].try_into().unwrap());
    }

    let half = 1u64 << (width - 1);
    let full = 1u64 << width;
    let mask = full - 1;

    let mut digits = Vec::with_capacity(256);
    while !is_zero(&limbs) {
        if limbs[0] & 1 == 1 {
            let residue = limbs[0] & mask;
            if residue >= half {
                // digit is negative: subtracting it means adding (full - residue).
                add_small(&mut limbs, full - residue);
                digits.push((residue as i64 - full as i64) as i8);
            } else {
                sub_small(&mut limbs, residue);
                digits.push(residue as i8);
            }
        } else {
            digits.push(0);
        }
        shr_one(&mut limbs);
    }
    digits
}

fn is_zero(limbs: &[u64; 4]) -> bool {
    limbs.iter().all(|&limb| limb == 0)
}

/// `limbs += addend` (addend fits in a limb), with carry propagation.
fn add_small(limbs: &mut [u64; 4], addend: u64) {
    let (sum, mut carry) = limbs[0].overflowing_add(addend);
    limbs[0] = sum;
    let mut index = 1;
    while carry && index < 4 {
        let (next, next_carry) = limbs[index].overflowing_add(1);
        limbs[index] = next;
        carry = next_carry;
        index += 1;
    }
}

/// `limbs -= subtrahend` (subtrahend <= limbs[0]-relevant), with borrow propagation.
fn sub_small(limbs: &mut [u64; 4], subtrahend: u64) {
    let (diff, mut borrow) = limbs[0].overflowing_sub(subtrahend);
    limbs[0] = diff;
    let mut index = 1;
    while borrow && index < 4 {
        let (next, next_borrow) = limbs[index].overflowing_sub(1);
        limbs[index] = next;
        borrow = next_borrow;
        index += 1;
    }
}

/// Logical right shift of the 256-bit value by one bit.
fn shr_one(limbs: &mut [u64; 4]) {
    limbs[0] = (limbs[0] >> 1) | (limbs[1] << 63);
    limbs[1] = (limbs[1] >> 1) | (limbs[2] << 63);
    limbs[2] = (limbs[2] >> 1) | (limbs[3] << 63);
    limbs[3] >>= 1;
}

/// Point in Jacobian coordinates: affine = (X/Z^2, Y/Z^3); infinity has Z = 0.
/// Jacobian doubling costs 1M+8S versus ~7M+5S for the homogeneous formulas the
/// generic path uses, and squarings are cheaper than multiplies (SOS squaring),
/// which is what makes the doubling-dominated wNAF loop faster in this system.
#[derive(Clone, Copy)]
struct JacobianPoint {
    x: Felt,
    y: Felt,
    z: Felt,
}

const JACOBIAN_INFINITY: JacobianPoint = JacobianPoint { x: Felt::ONE, y: Felt::ONE, z: Felt::ZERO };

impl JacobianPoint {
    fn from_affine_point(point: &AffinePoint) -> Self {
        Self { x: point.x(), y: point.y(), z: Felt::ONE }
    }

    fn neg(&self) -> Self {
        Self { x: self.x, y: -self.y, z: self.z }
    }

    /// dbl-2007-bl with the STARK curve's a = 1 (so `a*Z^4` is just `(Z^2)^2`):
    /// 1M + 8S. A 2-torsion input (y = 0) naturally yields Z3 = 0 (infinity).
    fn double(&self) -> Self {
        if self.z == Felt::ZERO {
            return *self;
        }
        let xx = self.x.square();
        let yy = self.y.square();
        let yyyy = yy.square();
        let zz = self.z.square();
        let s_term = {
            let sum_square = (self.x + yy).square();
            let s_half = sum_square - xx - yyyy;
            s_half + s_half
        };
        let m_term = xx + xx + xx + zz.square();
        let t_term = m_term.square() - s_term - s_term;
        let eight_yyyy = {
            let two = yyyy + yyyy;
            let four = two + two;
            four + four
        };
        let z3 = (self.y + self.z).square() - yy - zz;
        Self { x: t_term, y: m_term * (s_term - t_term) - eight_yyyy, z: z3 }
    }

    /// add-2007-bl full Jacobian addition: 11M + 5S. Handles the special cases
    /// (either operand at infinity, doubling, opposite points) exactly.
    fn add(&self, other: &Self) -> Self {
        if self.z == Felt::ZERO {
            return *other;
        }
        if other.z == Felt::ZERO {
            return *self;
        }
        let z1z1 = self.z.square();
        let z2z2 = other.z.square();
        let u1 = self.x * z2z2;
        let u2 = other.x * z1z1;
        let s1 = self.y * other.z * z2z2;
        let s2 = other.y * self.z * z1z1;
        let h = u2 - u1;
        let r_half = s2 - s1;
        if h == Felt::ZERO {
            return if r_half == Felt::ZERO { self.double() } else { JACOBIAN_INFINITY };
        }
        let i_term = {
            let two_h = h + h;
            two_h.square()
        };
        let j_term = h * i_term;
        let r_term = r_half + r_half;
        let v_term = u1 * i_term;
        let x3 = r_term.square() - j_term - v_term - v_term;
        let s1_j = s1 * j_term;
        let y3 = r_term * (v_term - x3) - (s1_j + s1_j);
        let z3 = ((self.z + other.z).square() - z1z1 - z2z2) * h;
        Self { x: x3, y: y3, z: z3 }
    }

    /// Convert to the homogeneous projective system without a field inversion:
    /// affine = (X/Z^2, Y/Z^3), so scaling by Z^3 gives homogeneous (X*Z, Y, Z^3).
    fn to_homogeneous(&self) -> ProjectivePoint {
        if self.z == Felt::ZERO {
            return ProjectivePoint::identity();
        }
        let zz = self.z.square();
        ProjectivePoint::new(self.x * self.z, self.y, zz * self.z)
    }
}

/// `scalar * point` via width-`WINDOW` wNAF over Jacobian coordinates:
/// ~251 doublings (1M+8S each) + ~scalar_bits/(W+1) full additions.
fn windowed_mul(point: &AffinePoint, scalar: &Felt) -> ProjectivePoint {
    // Odd-multiple table: odd_multiples[j] = (2j + 1) * point.
    // Width-W wNAF digits are odd with |d| <= 2^(W-1) - 1, so the max table index
    // is (2^(W-1) - 2)/2 = 2^(W-2) - 1: exactly 2^(W-2) entries are needed.
    let base = JacobianPoint::from_affine_point(point);
    let twice = base.double();
    let table_len = 1usize << (WINDOW - 2);
    let mut odd_multiples = Vec::with_capacity(table_len);
    odd_multiples.push(base);
    for j in 1..table_len {
        odd_multiples.push(odd_multiples[j - 1].add(&twice));
    }

    let digits = wnaf(scalar, WINDOW);
    let mut acc = JACOBIAN_INFINITY;
    for &digit in digits.iter().rev() {
        acc = acc.double();
        if digit > 0 {
            acc = acc.add(&odd_multiples[(digit as usize - 1) / 2]);
        } else if digit < 0 {
            acc = acc.add(&odd_multiples[((-digit) as usize - 1) / 2].neg());
        }
    }
    acc.to_homogeneous()
}

// Fixed exponents for the STARK prime p = 2^251 + 17*2^192 + 1, as little-endian
// u64 limbs. Both are sparse, so exponentiation is dominated by squarings.
//   (p-1)/2 = 2^250 + 17*2^191            (Euler's criterion / Legendre symbol)
//   (p+1)/2 = 2^250 + 17*2^191 + 1        (Cipolla exponent)
const EXP_P_MINUS_1_OVER_2: [u64; 4] = [0, 0, 0x8000_0000_0000_0000, 0x0400_0000_0000_0008];
const EXP_P_PLUS_1_OVER_2: [u64; 4] = [1, 0, 0x8000_0000_0000_0000, 0x0400_0000_0000_0008];

/// `base^exp mod p` (square-and-multiply, MSB first) over the fixed-width exponent.
fn pow_fp(base: &Felt, exp: &[u64; 4]) -> Felt {
    let mut result = Felt::ONE;
    for word in exp.iter().rev() {
        for bit in (0..64).rev() {
            result = result * result;
            if (word >> bit) & 1 == 1 {
                result = result * *base;
            }
        }
    }
    result
}

/// Multiply in F_p2 = F_p[t]/(t^2 - nqr), elements as `(a, b)` meaning `a + b*t`.
fn fp2_mul(x: (Felt, Felt), y: (Felt, Felt), nqr: Felt) -> (Felt, Felt) {
    let (a, b) = x;
    let (c, d) = y;
    (a * c + b * d * nqr, a * d + b * c)
}

/// Square in F_p2.
fn fp2_square(x: (Felt, Felt), nqr: Felt) -> (Felt, Felt) {
    let (a, b) = x;
    let ab = a * b;
    (a * a + b * b * nqr, ab + ab)
}

/// `base^exp` in F_p2 (square-and-multiply, MSB first).
fn fp2_pow(base: (Felt, Felt), exp: &[u64; 4], nqr: Felt) -> (Felt, Felt) {
    let mut result = (Felt::ONE, Felt::ZERO);
    for word in exp.iter().rev() {
        for bit in (0..64).rev() {
            result = fp2_square(result, nqr);
            if (word >> bit) & 1 == 1 {
                result = fp2_mul(result, base, nqr);
            }
        }
    }
    result
}

/// Modular square root on the STARK field via Cipolla's algorithm, which is
/// O(log p) and so avoids the O(2-adicity^2) blowup of Tonelli-Shanks on this
/// prime (2-adicity 192). Returns `None` when `value` is a non-residue.
///
/// There is deliberately no upfront Euler/Legendre residuosity check: the
/// candidate root is validated with a single squaring at the end, which is
/// correct for residues and rejects non-residues (whose candidate cannot
/// square back to `value`). This saves a full (p-1)/2 exponentiation per call.
fn stark_sqrt(value: &Felt) -> Option<Felt> {
    if *value == Felt::ZERO {
        return Some(Felt::ZERO);
    }
    // Find `a` with `a^2 - value` a non-residue, so F_p[t]/(t^2 - (a^2 - value)) is a field.
    let mut a = Felt::from(2u64);
    let nqr = loop {
        let candidate = a * a - *value;
        if candidate == Felt::ZERO {
            // value == a^2, so `a` already is the square root.
            return Some(a);
        }
        if pow_fp(&candidate, &EXP_P_MINUS_1_OVER_2) != Felt::ONE {
            break candidate;
        }
        a = a + Felt::ONE;
    };
    // For a residue, (a + t)^((p+1)/2) lands in F_p and is a square root of `value`.
    let (root, _imag) = fp2_pow((a, Felt::ONE), &EXP_P_PLUS_1_OVER_2, nqr);
    (root * root == *value).then_some(root)
}

/// Input-range checks shared by `verify_fast` and `verify_batch`. `Ok(())` means
/// the tuple is well-formed enough to invert `s` and finish verification.
fn check_ranges(message: &Felt, r: &Felt, s: &Felt) -> Result<(), VerifyError> {
    if message >= &ELEMENT_UPPER_BOUND {
        return Err(VerifyError::InvalidMessageHash);
    }
    if r == &Felt::ZERO || r >= &ELEMENT_UPPER_BOUND {
        return Err(VerifyError::InvalidR);
    }
    if s == &Felt::ZERO || s >= &ELEMENT_UPPER_BOUND {
        return Err(VerifyError::InvalidS);
    }
    Ok(())
}

/// Recover the full public-key point from its x-coordinate. The y sign does not
/// matter for verification because the final check tries both `+/-` candidates.
fn recover_public_key_point(public_key: &Felt) -> Result<AffinePoint, VerifyError> {
    let y_squared = public_key.square() * public_key + ALPHA * public_key + BETA;
    Ok(
        AffinePoint::new(*public_key, stark_sqrt(&y_squared).ok_or(VerifyError::InvalidPublicKey)?)
            .unwrap(),
    )
}

/// Finish verification given the full public-key point and the precomputed scalar
/// inverse `w_residue = s^-1 mod n`.
fn verify_point_with_inverse(
    public_key_point: &AffinePoint,
    message: &Felt,
    r: &Felt,
    w_residue: DynResidue<{ U256::LIMBS }>,
) -> Result<bool, VerifyError> {
    let w = u256_to_felt(&w_residue.retrieve());
    if w == Felt::ZERO || w >= ELEMENT_UPPER_BOUND {
        return Err(VerifyError::InvalidS);
    }

    let message_residue = DynResidue::new(&felt_to_u256(message), SCALAR_PARAMS);
    let r_residue = DynResidue::new(&felt_to_u256(r), SCALAR_PARAMS);
    let zw = u256_to_felt(&(message_residue * w_residue).retrieve());
    let zw_g = fixed_base_mul(&zw);

    let rw = u256_to_felt(&(r_residue * w_residue).retrieve());
    let rw_q = windowed_mul(public_key_point, &rw);

    // Compare the affine x-coordinate against r without a costly inversion:
    // affine_x == r  <=>  X == r * Z  (for a homogeneous projective point, Z != 0).
    let matches_r = |point: ProjectivePoint| {
        let z = point.z();
        z != Felt::ZERO && point.x() == *r * z
    };
    Ok(matches_r(&zw_g + &rw_q) || matches_r(&zw_g - &rw_q))
}

/// Finish verification given the precomputed scalar inverse `w_residue = s^-1 mod n`.
/// Recovers the public-key point first so error precedence matches stock `verify`
/// (`InvalidPublicKey` before the `InvalidS` range check on `w`).
fn verify_with_inverse(
    public_key: &Felt,
    message: &Felt,
    r: &Felt,
    w_residue: DynResidue<{ U256::LIMBS }>,
) -> Result<bool, VerifyError> {
    let public_key_point = recover_public_key_point(public_key)?;
    verify_point_with_inverse(&public_key_point, message, r, w_residue)
}

/// Optimized drop-in for [`crate::verify`]. Returns the same boolean/errors.
pub fn verify_fast(
    public_key: &Felt,
    message: &Felt,
    r: &Felt,
    s: &Felt,
) -> Result<bool, VerifyError> {
    check_ranges(message, r, s)?;
    // Scalar-field arithmetic mod EC_ORDER via fixed-size Montgomery residues
    // (crypto-bigint) instead of heap-allocating num-bigint. `s` is in [1, bound) and
    // EC_ORDER is prime, so the inverse always exists.
    let w_residue = DynResidue::new(&felt_to_u256(s), SCALAR_PARAMS).invert().0;
    verify_with_inverse(public_key, message, r, w_residue)
}

/// Like [`verify_fast`], but takes the full public-key point, skipping the
/// square-root y-recovery entirely (the single largest fixed cost of x-only
/// verification). Either y sign yields the same result, since the final check
/// tries both `+/-` candidates.
///
/// The point must be on the curve (as enforced by `AffinePoint::new`); the
/// identity point is rejected as an invalid public key.
pub fn verify_with_pubkey_point(
    public_key_point: &AffinePoint,
    message: &Felt,
    r: &Felt,
    s: &Felt,
) -> Result<bool, VerifyError> {
    check_ranges(message, r, s)?;
    if public_key_point.is_identity() || !is_on_curve(public_key_point) {
        return Err(VerifyError::InvalidPublicKey);
    }
    let w_residue = DynResidue::new(&felt_to_u256(s), SCALAR_PARAMS).invert().0;
    verify_point_with_inverse(public_key_point, message, r, w_residue)
}

/// Batch-verify many signatures `(public_key, message, r, s)`, returning a result
/// per input in order. The one cost that amortizes across a batch is the scalar
/// inverse `s^-1`: instead of one field inversion each, a single inversion plus
/// `~2N` multiplies covers the whole batch (Montgomery's trick). The public-key
/// square-root and the two scalar multiplications remain per-signature (they cannot
/// be shared, and the STARK pubkey y-sign ambiguity rules out a single batched MSM).
pub fn verify_batch(items: &[(Felt, Felt, Felt, Felt)]) -> Vec<Result<bool, VerifyError>> {
    let one = DynResidue::new(&U256::ONE, SCALAR_PARAMS);

    // Range-check each item; collect s-residues only for the valid ones.
    let mut results: Vec<Result<bool, VerifyError>> = Vec::with_capacity(items.len());
    let mut valid_positions = Vec::new();
    let mut s_residues = Vec::new();
    for (index, (_public_key, message, r, s)) in items.iter().enumerate() {
        match check_ranges(message, r, s) {
            Ok(()) => {
                valid_positions.push(index);
                s_residues.push(DynResidue::new(&felt_to_u256(s), SCALAR_PARAMS));
                results.push(Ok(false)); // placeholder, overwritten below
            }
            Err(error) => results.push(Err(error)),
        }
    }

    // Montgomery batch inversion of all valid `s` values.
    let mut prefix = Vec::with_capacity(s_residues.len());
    let mut running = one;
    for s_residue in &s_residues {
        running *= s_residue;
        prefix.push(running);
    }
    let mut inverse = if s_residues.is_empty() { one } else { running.invert().0 };
    let mut w_residues = vec![one; s_residues.len()];
    for position in (0..s_residues.len()).rev() {
        let preceding = if position == 0 { one } else { prefix[position - 1] };
        w_residues[position] = inverse * preceding;
        inverse *= &s_residues[position];
    }

    for (batch_index, &item_index) in valid_positions.iter().enumerate() {
        let (public_key, message, r, _s) = &items[item_index];
        results[item_index] =
            verify_with_inverse(public_key, message, r, w_residues[batch_index]);
    }
    results
}

/// Whether the affine point satisfies `y^2 = x^3 + ALPHA*x + BETA`. Points
/// arriving from the wire must be validated to rule out invalid-curve attacks.
fn is_on_curve(point: &AffinePoint) -> bool {
    let x = point.x();
    let y = point.y();
    y * y == x * x * x + ALPHA * x + BETA
}

/// Window width for the batch Pippenger MSM, by term count.
fn pippenger_window(term_count: usize) -> usize {
    match term_count {
        0..=512 => 6,
        513..=1024 => 7,
        1025..=2048 => 8,
        _ => 9,
    }
}

/// Pippenger bucket MSM over affine points: computes `sum(scalar_i * point_i)`.
fn pippenger_msm(scalars: &[U256], points: &[AffinePoint]) -> ProjectivePoint {
    let window = pippenger_window(scalars.len());
    let windows_count = 256usize.div_ceil(window);
    let bucket_count = (1usize << window) - 1;
    let mut accumulator = ProjectivePoint::identity();

    for window_index in (0..windows_count).rev() {
        for _ in 0..window {
            accumulator = accumulator.double();
        }
        let mut buckets = vec![ProjectivePoint::identity(); bucket_count];
        for (scalar, point) in scalars.iter().zip(points) {
            let bit_offset = window_index * window;
            let mut digit = 0usize;
            for bit in 0..window {
                let position = bit_offset + bit;
                if position < 256 && scalar.bit_vartime(position) {
                    digit |= 1 << bit;
                }
            }
            if digit != 0 {
                buckets[digit - 1] += point;
            }
        }
        // Suffix sums turn the buckets into sum(digit * bucket[digit]).
        let mut running = ProjectivePoint::identity();
        let mut window_sum = ProjectivePoint::identity();
        for bucket in buckets.iter().rev() {
            running += bucket;
            window_sum += &running;
        }
        accumulator += &window_sum;
    }
    accumulator
}

/// Probabilistic all-or-nothing batch verification of ECDSA signatures whose
/// full nonce point `R` is transmitted alongside `s`; the classic `r` is
/// `x(R)`, range-checked below `2^251` exactly as stock signing guarantees.
///
/// Each item is `(public_key_point, message, nonce_point, s)`. Verifies the
/// exact group equation `(z*w)*G + (r*w)*Q == R` for every item at once via a
/// random linear combination evaluated as one Pippenger multi-scalar
/// multiplication: cost per signature drops well below a single verification
/// for batches of ~64 and larger.
///
/// - Returns `Ok(true)` iff every signature satisfies its equation (soundness
///   error ~2^-128 per batch from the 128-bit random coefficients).
/// - Returns `Ok(false)` if at least one signature is invalid; use bisection
///   over sub-batches (or per-item [`verify_with_pubkey_point`]) to locate it.
/// - Returns `Err` on the first item whose values fail the stock range checks.
///
/// `random_seed` MUST be unpredictable to signature submitters (e.g. from a
/// CSPRNG per batch); a predictable seed lets an attacker craft signatures
/// that cancel in the combination.
///
/// Unlike the x-only [`verify_fast`], this checks the exact `(Q, R)` supplied:
/// chain-equivalent acceptance requires the protocol to fix the y-parity
/// convention for public keys and nonce points.
pub fn verify_batch_with_nonce_points(
    items: &[(AffinePoint, Felt, AffinePoint, Felt)],
    random_seed: &[u8; 32],
) -> Result<bool, VerifyError> {
    if items.is_empty() {
        return Ok(true);
    }

    // Batched w_i = s_i^-1 mod n (Montgomery's trick), with stock range checks.
    let one = DynResidue::new(&U256::ONE, SCALAR_PARAMS);
    let mut s_residues = Vec::with_capacity(items.len());
    for (public_key_point, message, nonce_point, s) in items {
        if public_key_point.is_identity() || !is_on_curve(public_key_point) {
            return Err(VerifyError::InvalidPublicKey);
        }
        if nonce_point.is_identity() || !is_on_curve(nonce_point) {
            return Err(VerifyError::InvalidR);
        }
        check_ranges(message, &nonce_point.x(), s)?;
        s_residues.push(DynResidue::new(&felt_to_u256(s), SCALAR_PARAMS));
    }
    let mut prefix = Vec::with_capacity(s_residues.len());
    let mut running = one;
    for s_residue in &s_residues {
        running *= s_residue;
        prefix.push(running);
    }
    let mut inverse = running.invert().0;
    let mut w_residues = vec![one; s_residues.len()];
    for index in (0..s_residues.len()).rev() {
        let preceding = if index == 0 { one } else { prefix[index - 1] };
        w_residues[index] = inverse * preceding;
        inverse *= &s_residues[index];
    }

    let mut scalars = Vec::with_capacity(2 * items.len() + 1);
    let mut points = Vec::with_capacity(2 * items.len() + 1);
    let mut g_coefficient = DynResidue::new(&U256::ZERO, SCALAR_PARAMS);
    for (index, (public_key_point, message, nonce_point, _s)) in items.iter().enumerate() {
        // Random 128-bit coefficient from a SHA-256 stream over the seed
        // (delta_0 = 1); ~30x cheaper per item than a Poseidon derivation.
        let delta_residue = if index == 0 {
            one
        } else {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(random_seed);
            hasher.update((index as u64).to_be_bytes());
            let digest = hasher.finalize();
            DynResidue::new(&felt_to_u256(&Felt::from_bytes_be_slice(&digest[..16])), SCALAR_PARAMS)
        };
        let w_residue = w_residues[index];

        // r = x(R): `check_ranges` above already enforced r < 2^251 < n, the
        // same convention stock signing guarantees, so no mod-n reduction is
        // needed here.
        let r_residue = DynResidue::new(&felt_to_u256(&nonce_point.x()), SCALAR_PARAMS);
        let w_value = u256_to_felt(&w_residue.retrieve());
        if w_value == Felt::ZERO || w_value >= ELEMENT_UPPER_BOUND {
            return Err(VerifyError::InvalidS);
        }

        g_coefficient =
            g_coefficient + delta_residue * DynResidue::new(&felt_to_u256(message), SCALAR_PARAMS) * w_residue;

        scalars.push((delta_residue * r_residue * w_residue).retrieve());
        points.push(public_key_point.clone());

        scalars.push(delta_residue.retrieve());
        points.push(-nonce_point);
    }
    scalars.push(g_coefficient.retrieve());
    points.push(AffinePoint::generator());

    // The combination sums to the identity iff every equation holds.
    let combination = pippenger_msm(&scalars, &points);
    Ok(combination.to_affine().is_err())
}

/// Like [`verify_batch_with_nonce_points`], but each signature carries only the
/// classic `r` plus the y-parity bit of its nonce point (`+1 bit` on the wire
/// instead of `+32 bytes`). The full `R` is lifted here with one Cipolla square
/// root per signature (square roots, unlike inversions, cannot be batched), so
/// this costs ~16 us/signature more than the full-point variant.
///
/// Items are `(public_key_point, message, r, nonce_y_is_odd, s)`.
pub fn verify_batch_with_nonce_parity(
    items: &[(AffinePoint, Felt, Felt, bool, Felt)],
    random_seed: &[u8; 32],
) -> Result<bool, VerifyError> {
    let mut resolved = Vec::with_capacity(items.len());
    for (public_key_point, message, r, nonce_y_is_odd, s) in items {
        check_ranges(message, r, s)?;
        let y_squared = r.square() * r + ALPHA * r + BETA;
        // A non-liftable r is not the x-coordinate of any curve point: the
        // signature is invalid (stock verify would return false), not malformed.
        let Some(root) = stark_sqrt(&y_squared) else {
            return Ok(false);
        };
        let root_is_odd = root.to_bytes_le()[0] & 1 == 1;
        let y = if root_is_odd == *nonce_y_is_odd { root } else { -root };
        let nonce_point = AffinePoint::new(*r, y).map_err(|_| VerifyError::InvalidR)?;
        resolved.push((public_key_point.clone(), *message, nonce_point, *s));
    }
    verify_batch_with_nonce_points(&resolved, random_seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{get_public_key, rfc6979_generate_k, sign, verify};

    /// Deterministic PRNG (splitmix64) so the fuzz cases are reproducible.
    fn next_random(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A full-width field element (reduced mod p).
    fn random_felt(state: &mut u64) -> Felt {
        let mut bytes = [0u8; 32];
        for chunk in bytes.chunks_mut(8) {
            chunk.copy_from_slice(&next_random(state).to_le_bytes());
        }
        Felt::from_bytes_le(&bytes)
    }

    /// A value `< 2^248 < ELEMENT_UPPER_BOUND`, valid as a message / private key.
    fn random_below_bound(state: &mut u64) -> Felt {
        let mut bytes = [0u8; 32];
        for index in 0..3 {
            bytes[index * 8..index * 8 + 8].copy_from_slice(&next_random(state).to_le_bytes());
        }
        Felt::from_bytes_le(&bytes)
    }

    fn outcome(result: &Result<bool, VerifyError>) -> Option<bool> {
        result.as_ref().ok().copied()
    }

    /// Tens of thousands of fully random `(pk, msg, r, s)` tuples: `verify_fast`
    /// must agree with stock `verify` (almost all are Err or `false`). Random inputs
    /// never hit the point-at-infinity edge (~2^-251), so stock does not panic here.
    #[test]
    fn fuzz_random_tuples_match_stock() {
        let mut state = 0x5EED_0000_1234_ABCD;
        for _ in 0..40_000 {
            let public_key = random_felt(&mut state);
            let message = random_felt(&mut state);
            let r = random_felt(&mut state);
            let s = random_felt(&mut state);
            assert_eq!(
                verify(&public_key, &message, &r, &s).ok(),
                verify_fast(&public_key, &message, &r, &s).ok(),
                "random tuple mismatch: pk={public_key:#x} msg={message:#x} r={r:#x} s={s:#x}"
            );
        }
    }

    /// Real signatures (must verify true) plus tampered variants (must verify false),
    /// checked against stock `verify` and cross-checked through `verify_batch`.
    #[test]
    fn fuzz_real_signatures_match_stock_and_batch() {
        let mut state = 0xABCD_0000_5EED_1234;
        let mut batch_items = Vec::new();
        let mut true_seen = 0u32;
        let mut false_seen = 0u32;
        for _ in 0..4_000 {
            let private_key = random_below_bound(&mut state) + Felt::ONE; // nonzero
            let message = random_below_bound(&mut state);
            let public_key = get_public_key(&private_key);
            let k = rfc6979_generate_k(&message, &private_key, None);
            let sig = sign(&private_key, &message, &k).unwrap();

            // Valid signature.
            assert_eq!(outcome(&verify(&public_key, &message, &sig.r, &sig.s)), Some(true));
            assert_eq!(verify_fast(&public_key, &message, &sig.r, &sig.s).ok(), Some(true));
            true_seen += 1;
            batch_items.push((public_key, message, sig.r, sig.s));

            // Tampered variants must fail identically on both.
            for tampered in [
                (public_key, message + Felt::ONE, sig.r, sig.s),
                (public_key, message, sig.r + Felt::ONE, sig.s),
                (public_key, message, sig.r, sig.s + Felt::ONE),
            ] {
                let (pk, msg, r, s) = tampered;
                assert_eq!(
                    verify(&pk, &msg, &r, &s).ok(),
                    verify_fast(&pk, &msg, &r, &s).ok(),
                    "tampered mismatch"
                );
                if verify_fast(&pk, &msg, &r, &s).ok() == Some(false) {
                    false_seen += 1;
                }
                batch_items.push(tampered);
            }
        }
        assert!(true_seen > 0 && false_seen > 0, "expected both true and false outcomes");

        // verify_batch must return the same per-item results as verify_fast.
        let batch = verify_batch(&batch_items);
        assert_eq!(batch.len(), batch_items.len());
        for (index, (pk, msg, r, s)) in batch_items.iter().enumerate() {
            assert_eq!(
                outcome(&batch[index]),
                verify_fast(pk, msg, r, s).ok(),
                "batch vs verify_fast mismatch at {index}"
            );
        }
    }

    /// `verify_fast` must agree with stock `verify` (compared via `.ok()`, since
    /// `VerifyError` is not `PartialEq`).
    fn assert_parity(public_key: &Felt, message: &Felt, r: &Felt, s: &Felt) {
        assert_eq!(
            verify(public_key, message, r, s).ok(),
            verify_fast(public_key, message, r, s).ok(),
            "verify_fast diverged: pk={public_key:#x} msg={message:#x} r={r:#x} s={s:#x}"
        );
    }

    #[test]
    fn matches_stock_verify_on_valid_and_tampered() {
        let salt = Felt::from_hex("0x3c1e9550e66958296d11b60f8e8e7a7").unwrap();
        for i in 1u64..=32 {
            let private_key = Felt::from(i) * salt;
            let message = Felt::from(i.wrapping_mul(0xabcdef).wrapping_add(7));
            let public_key = get_public_key(&private_key);
            let k = rfc6979_generate_k(&message, &private_key, None);
            let sig = sign(&private_key, &message, &k).unwrap();

            assert_eq!(verify_fast(&public_key, &message, &sig.r, &sig.s).ok(), Some(true));
            assert_parity(&public_key, &message, &sig.r, &sig.s);
            assert_parity(&public_key, &(message + Felt::ONE), &sig.r, &sig.s);
            assert_parity(&public_key, &message, &(sig.r + Felt::ONE), &sig.s);
            assert_parity(&public_key, &message, &sig.r, &(sig.s + Felt::ONE));
        }
    }

    #[test]
    fn cipolla_sqrt_matches_stock_and_rejects_non_residues() {
        let mut non_residue_seen = false;
        for i in 1u64..=500 {
            let value = Felt::from(i);
            match stark_sqrt(&value) {
                Some(root) => {
                    assert_eq!(root * root, value, "sqrt({i}) is not a root");
                    // Stock `Felt::sqrt` agrees a root exists (may be the other sign).
                    let stock = value.sqrt().expect("stock sqrt should also find a root");
                    assert!(root == stock || root == -stock, "root/sign mismatch for {i}");
                }
                None => {
                    non_residue_seen = true;
                    assert!(value.sqrt().is_none(), "stock found a root where Cipolla did not ({i})");
                }
            }
        }
        assert!(non_residue_seen, "expected some non-residues in 1..=500");
    }

    #[test]
    fn batch_matches_per_item_verify() {
        let salt = Felt::from_hex("0x3c1e9550e66958296d11b60f8e8e7a7").unwrap();
        let mut items = Vec::new();
        for i in 1u64..=40 {
            let private_key = Felt::from(i) * salt;
            let message = Felt::from(i.wrapping_mul(0x1234567) + 3);
            let public_key = get_public_key(&private_key);
            let k = rfc6979_generate_k(&message, &private_key, None);
            let sig = sign(&private_key, &message, &k).unwrap();
            items.push((public_key, message, sig.r, sig.s));
        }
        // A tampered (valid-range but wrong) signature and an out-of-range one.
        items.push((items[0].0, items[0].1 + Felt::ONE, items[0].2, items[0].3));
        items.push((get_public_key(&Felt::from(9u64)), Felt::from(3u64), Felt::from(5u64), Felt::ZERO));

        let batch = verify_batch(&items);
        assert_eq!(batch.len(), items.len());
        for (index, (public_key, message, r, s)) in items.iter().enumerate() {
            assert_eq!(
                batch[index].as_ref().ok().copied(),
                verify_fast(public_key, message, r, s).ok(),
                "batch[{index}] disagrees with verify_fast"
            );
        }
    }

    /// The Jacobian wNAF multiply must agree with the generic types-core
    /// double-and-add (`&ProjectivePoint * Felt`) on random and edge scalars.
    #[test]
    fn jacobian_windowed_mul_matches_generic() {
        let mut state = 0xC0FF_EE00_DEAD_BEEF;
        let point = AffinePoint::new_from_x(&get_public_key(&Felt::from(7u64)), true).unwrap();
        let point_proj = ProjectivePoint::from_affine(point.x(), point.y()).unwrap();

        let mut scalars: Vec<Felt> = (0..64).map(|_| random_felt(&mut state)).collect();
        scalars.extend([
            Felt::ZERO,
            Felt::ONE,
            Felt::TWO,
            Felt::THREE,
            Felt::from(15u64),  // largest single wNAF digit
            Felt::from(16u64),  // smallest two-digit case
            Felt::MAX,
        ]);
        for scalar in scalars {
            let via_jacobian = windowed_mul(&point, &scalar);
            let via_generic = &point_proj * scalar;
            // Compare in affine (either representation may differ projectively).
            match (via_jacobian.to_affine(), via_generic.to_affine()) {
                (Ok(a), Ok(b)) => {
                    assert_eq!(a.x(), b.x(), "x mismatch for scalar {scalar:#x}");
                    assert_eq!(a.y(), b.y(), "y mismatch for scalar {scalar:#x}");
                }
                (Err(_), Err(_)) => {} // both infinity (scalar == 0 mod order)
                _ => panic!("infinity disagreement for scalar {scalar:#x}"),
            }
        }
    }

    /// `verify_with_pubkey_point` must agree with `verify_fast` for either y sign
    /// of the public-key point, on valid and tampered signatures.
    #[test]
    fn with_point_matches_x_only_for_both_y_signs() {
        let mut state = 0x0123_4567_89AB_CDEF;
        for _ in 0..200 {
            let private_key = random_below_bound(&mut state) + Felt::ONE;
            let message = random_below_bound(&mut state);
            let public_key = get_public_key(&private_key);
            let k = rfc6979_generate_k(&message, &private_key, None);
            let sig = sign(&private_key, &message, &k).unwrap();

            for y_parity in [false, true] {
                let point = AffinePoint::new_from_x(&public_key, y_parity).unwrap();
                assert_eq!(
                    verify_with_pubkey_point(&point, &message, &sig.r, &sig.s).ok(),
                    Some(true),
                    "valid signature must verify with y_parity={y_parity}"
                );
                assert_eq!(
                    verify_with_pubkey_point(&point, &(message + Felt::ONE), &sig.r, &sig.s).ok(),
                    verify_fast(&public_key, &(message + Felt::ONE), &sig.r, &sig.s).ok(),
                    "tampered mismatch with y_parity={y_parity}"
                );
            }
        }
    }

    #[test]
    fn with_point_rejects_identity() {
        let result = verify_with_pubkey_point(
            &AffinePoint::identity(),
            &Felt::from(2u64),
            &Felt::from(3u64),
            &Felt::from(5u64),
        );
        assert!(matches!(result, Err(VerifyError::InvalidPublicKey)));
    }

    #[test]
    fn rejects_out_of_range_r_and_s() {
        let public_key = get_public_key(&Felt::from(42u64));
        let message = Felt::from(7u64);
        assert!(verify_fast(&public_key, &message, &Felt::ZERO, &Felt::from(5u64)).is_err());
        assert!(verify_fast(&public_key, &message, &Felt::from(5u64), &Felt::ZERO).is_err());
    }

    /// Input (from an adversarial audit) where a candidate `z*w*G + r*w*Q` is the
    /// point at infinity. Stock `verify` panics here via `.to_affine().unwrap()`;
    /// the optimized path must return `Ok(false)` (infinity has no affine x = r).
    #[test]
    fn infinity_candidate_returns_false_instead_of_panicking() {
        let public_key =
            Felt::from_hex("0x21a9091974fd58a932db290fecf11467845ed7108993da01c0849ec2876347b")
                .unwrap();
        let message = Felt::from_hex("0x1").unwrap();
        let r = Felt::from_hex("0x3a282f8608e46c64df4a40eb3e8249500f73e9cc5cb5288f2845d06a57209d6")
            .unwrap();
        let s = Felt::from_hex("0x9e3779b97f4a7c15").unwrap();
        assert_eq!(verify_fast(&public_key, &message, &r, &s).ok(), Some(false));
    }

    // Known-answer tests below check EXACT intermediate values against reference
    // vectors computed with an independent pure-Python integer-arithmetic
    // implementation of the curve group law (no lambdaworks, no Rust) -- so a
    // shared bug between this module and stock `verify` cannot hide behind
    // boolean parity.

    fn affine(x_hex: &str, y_hex: &str) -> AffinePoint {
        AffinePoint::new(Felt::from_hex(x_hex).unwrap(), Felt::from_hex(y_hex).unwrap()).unwrap()
    }

    fn assert_point_eq(point: &ProjectivePoint, expected: &AffinePoint, label: &str) {
        let actual = point.to_affine().unwrap_or_else(|_| panic!("{label}: unexpected infinity"));
        assert_eq!(actual.x(), expected.x(), "{label}: x mismatch");
        assert_eq!(actual.y(), expected.y(), "{label}: y mismatch");
    }

    const TWO_G_X: &str = "0x0759ca09377679ecd535a81e83039658bf40959283187c654c5416f439403cf5";
    const TWO_G_Y: &str = "0x06f524a3400e7708d5c01a28598ad272e7455aa88778b19f93b562d7a9646c41";
    const THREE_G_X: &str = "0x0411494b501a98abd8262b0da1351e17899a0c4ef23dd2f96fec5ba847310b20";
    const THREE_G_Y: &str = "0x07e1b3ebac08924d2c26f409549191fcf94f3bf6f301ed3553e22dfb802f0686";
    const FULL_SCALAR: &str = "0x6f2a1b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f7081920304050607";
    const FULL_G_X: &str = "0x02e503ed721a7e6406cbff41ae74afe4ee3d2d8235b4e0f597412fd585778bef";
    const FULL_G_Y: &str = "0x01f0e92fae4b108704126e9ee49e9f70af474c672b2d6bdee3de64ca48e803e2";
    const ORDER_MINUS_1_G_X: &str =
        "0x01ef15c18599971b7beced415a40f0c7deacfd9b0d1819e03d723d8bc943cfca";
    const ORDER_MINUS_1_G_Y: &str =
        "0x07a997f9f55b68e04841b7fe20b9139d21ac132ee541bc5cd78cfff3c91723e2";
    const Q7_X: &str = "0x0743829e0a179f8afe223fc8112dfc8d024ab6b235fd42283c4f5970259ce7b7";
    const Q7_Y: &str = "0x00e67a0a63cc493225e45b9178a3375596ea2a1d7012628a328dbc14c78cd1b7";
    const FULL_Q7_X: &str = "0x013b66d2f5923d8afcea8b11b4a332267078e34c90fd3c89a5d8a6a2b19c6f9d";
    const FULL_Q7_Y: &str = "0x077e395dea214800b20677f3cfb38c8d9bc90a9986891b9c85e259ea3b51d4e6";
    const SEVENTEEN_Q7_X: &str =
        "0x05e872efefb53f900c760c1aabbe209be5a137d6151e8d75ce307bcd8aedd74d";
    const SEVENTEEN_Q7_Y: &str =
        "0x01521ae45572e09815dc73040f6845b7fd6cae2579ce5c7d162fd96a0355d198";

    /// STARK curve order (= EC_ORDER), for n*P = infinity and (n-1)*G checks.
    const CURVE_ORDER: &str = "0x0800000000000010ffffffffffffffffb781126dcae7b2321e66a241adc64d2f";

    #[test]
    fn jacobian_double_and_add_match_independent_reference() {
        let generator = AffinePoint::generator();
        let g_jacobian = JacobianPoint::from_affine_point(&generator);

        let two_g = g_jacobian.double();
        assert_point_eq(&two_g.to_homogeneous(), &affine(TWO_G_X, TWO_G_Y), "2G = double(G)");

        let three_g_via_add = two_g.add(&g_jacobian);
        assert_point_eq(
            &three_g_via_add.to_homogeneous(),
            &affine(THREE_G_X, THREE_G_Y),
            "3G = 2G + G",
        );

        // Doubling-branch of `add` (equal operands) must agree with `double`.
        let four_g_via_add = two_g.add(&two_g);
        let four_g_via_double = two_g.double();
        assert_point_eq(
            &four_g_via_add.to_homogeneous(),
            &four_g_via_double.to_homogeneous().to_affine().unwrap(),
            "add(P,P) == double(P)",
        );

        // P + (-P) must be infinity.
        let cancelled = g_jacobian.add(&g_jacobian.neg());
        assert!(cancelled.to_homogeneous().to_affine().is_err(), "G + (-G) must be infinity");
    }

    #[test]
    fn fixed_base_mul_known_answers() {
        let generator = AffinePoint::generator();
        assert_point_eq(&fixed_base_mul(&Felt::ONE), &generator, "1*G");
        assert_point_eq(&fixed_base_mul(&Felt::TWO), &affine(TWO_G_X, TWO_G_Y), "2*G");
        assert_point_eq(&fixed_base_mul(&Felt::THREE), &affine(THREE_G_X, THREE_G_Y), "3*G");
        assert_point_eq(
            &fixed_base_mul(&Felt::from_hex(FULL_SCALAR).unwrap()),
            &affine(FULL_G_X, FULL_G_Y),
            "full-width scalar * G",
        );

        let order = Felt::from_hex(CURVE_ORDER).unwrap();
        assert_point_eq(
            &fixed_base_mul(&(order - Felt::ONE)),
            &affine(ORDER_MINUS_1_G_X, ORDER_MINUS_1_G_Y),
            "(n-1)*G",
        );
        assert!(fixed_base_mul(&order).to_affine().is_err(), "n*G must be infinity");
    }

    #[test]
    fn windowed_mul_known_answers() {
        let q7 = affine(Q7_X, Q7_Y);
        assert_point_eq(&windowed_mul(&q7, &Felt::ONE), &q7, "1*Q");
        assert_point_eq(
            &windowed_mul(&q7, &Felt::from(17u64)),
            &affine(SEVENTEEN_Q7_X, SEVENTEEN_Q7_Y),
            "17*Q",
        );
        assert_point_eq(
            &windowed_mul(&q7, &Felt::from_hex(FULL_SCALAR).unwrap()),
            &affine(FULL_Q7_X, FULL_Q7_Y),
            "full-width scalar * Q",
        );

        // n*Q reaches infinity through the H == 0, r != 0 special case of `add`
        // on the final digit; the result must be the point at infinity.
        let order = Felt::from_hex(CURVE_ORDER).unwrap();
        assert!(windowed_mul(&q7, &order).to_affine().is_err(), "n*Q must be infinity");
    }

    #[test]
    fn stark_sqrt_known_answers() {
        // value = k^2 mod p for a hardcoded k: the root must be exactly k or p-k.
        let value = Felt::from_hex(
            "0x03588e3f38a9e4f752b22bd4eb778d58e2d0a23b68c8b1c05d83ff6cd8ab9d04",
        )
        .unwrap();
        let root_a = Felt::from_hex(
            "0x03c1e9550e66958296d11b60f8e8e7a7ad990d07fa65d5f7652c4a6c87d4e3cc",
        )
        .unwrap();
        let root_b = Felt::from_hex(
            "0x043e16aaf1996a8e692ee49f071718585266f2f8059a2a089ad3b593782b1c35",
        )
        .unwrap();
        let root = stark_sqrt(&value).expect("value is a residue");
        assert!(root == root_a || root == root_b, "sqrt returned {root:#x}");

        // 9 = 3^2: roots are exactly {3, p-3}.
        let root_of_nine = stark_sqrt(&Felt::from(9u64)).expect("9 is a residue");
        assert!(
            root_of_nine == Felt::THREE || root_of_nine == -Felt::THREE,
            "sqrt(9) returned {root_of_nine:#x}"
        );

        // 3 is a non-residue mod p (3^((p-1)/2) == p - 1, verified independently).
        assert!(stark_sqrt(&Felt::THREE).is_none(), "3 must have no square root");

        assert_eq!(stark_sqrt(&Felt::ZERO), Some(Felt::ZERO));
        let root_of_one = stark_sqrt(&Felt::ONE).expect("1 is a residue");
        assert!(root_of_one == Felt::ONE || root_of_one == -Felt::ONE);
    }

    #[test]
    fn wnaf_digits_reconstruct_scalar_exactly() {
        let scalars = [
            Felt::ONE,
            Felt::from(15u64),
            Felt::from(16u64),
            Felt::from(17u64),
            Felt::from_hex(FULL_SCALAR).unwrap(),
            Felt::from_hex(CURVE_ORDER).unwrap() - Felt::ONE,
            Felt::MAX,
        ];
        for scalar in scalars {
            let digits = wnaf(&scalar, WINDOW);
            // Reconstruct sum(digit_i * 2^i) in the field; every scalar < p, so
            // exact equality in Felt implies exact integer equality.
            let mut reconstructed = Felt::ZERO;
            let mut power = Felt::ONE;
            for &digit in &digits {
                if digit > 0 {
                    reconstructed = reconstructed + power * Felt::from(digit as u64);
                } else if digit < 0 {
                    reconstructed = reconstructed - power * Felt::from((-digit) as u64);
                }
                power = power + power;
            }
            assert_eq!(reconstructed, scalar, "wNAF digits do not reconstruct scalar");

            for (index, &digit) in digits.iter().enumerate() {
                if digit != 0 {
                    assert!(digit % 2 != 0, "wNAF digit at {index} must be odd");
                    assert!(digit.abs() < 16, "wNAF digit at {index} out of range");
                    // Width-5 window property: the next 4 digits must be zero.
                    for offset in 1..WINDOW as usize {
                        if let Some(&next) = digits.get(index + offset) {
                            assert_eq!(next, 0, "window property violated at {index}+{offset}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn scalar_inverse_known_answer() {
        // w = s^-1 mod n for a hardcoded s, expected value computed independently.
        let s_value = Felt::from_hex(
            "0x0405c3191ab3883ef2b763af35bc5f5d15b3b4e99461d70e84c654a351a7c81b",
        )
        .unwrap();
        let expected_w = Felt::from_hex(
            "0x01ce0310e48aa17f713cbd8f8acc5a88703a359d2ef27d33ef95b8cfce4bcc91",
        )
        .unwrap();
        let w_residue = DynResidue::new(&felt_to_u256(&s_value), SCALAR_PARAMS).invert().0;
        assert_eq!(u256_to_felt(&w_residue.retrieve()), expected_w);
    }

    /// The SAME reference vectors, checked against the STOCK implementation's
    /// stages (lambdaworks generic scalar mult, types-core Tonelli-Shanks sqrt,
    /// num-bigint mod_inverse). This triangulates the Python reference against
    /// code written by third parties: reference <-> ours, reference <-> stock,
    /// and ours <-> stock (parity fuzz) must all hold independently.
    #[test]
    fn stock_stages_match_independent_reference() {
        use crate::fe_utils::mod_inverse;
        use starknet_curve::curve_params::EC_ORDER;

        // Stock scalar multiplication: generic double-and-add via
        // `&ProjectivePoint * Felt` (exactly what stock `mul_by_bits` invokes).
        let generator_proj =
            ProjectivePoint::from_affine(AffinePoint::generator().x(), AffinePoint::generator().y())
                .unwrap();
        assert_point_eq(&(&generator_proj * Felt::TWO), &affine(TWO_G_X, TWO_G_Y), "stock 2*G");
        assert_point_eq(&(&generator_proj * Felt::THREE), &affine(THREE_G_X, THREE_G_Y), "stock 3*G");
        assert_point_eq(
            &(&generator_proj * Felt::from_hex(FULL_SCALAR).unwrap()),
            &affine(FULL_G_X, FULL_G_Y),
            "stock full-width scalar * G",
        );
        let order = Felt::from_hex(CURVE_ORDER).unwrap();
        assert_point_eq(
            &(&generator_proj * (order - Felt::ONE)),
            &affine(ORDER_MINUS_1_G_X, ORDER_MINUS_1_G_Y),
            "stock (n-1)*G",
        );
        assert!((&generator_proj * order).to_affine().is_err(), "stock n*G must be infinity");

        let q7 = affine(Q7_X, Q7_Y);
        let q7_proj = ProjectivePoint::from_affine(q7.x(), q7.y()).unwrap();
        assert_point_eq(
            &(&q7_proj * Felt::from(17u64)),
            &affine(SEVENTEEN_Q7_X, SEVENTEEN_Q7_Y),
            "stock 17*Q",
        );
        assert_point_eq(
            &(&q7_proj * Felt::from_hex(FULL_SCALAR).unwrap()),
            &affine(FULL_Q7_X, FULL_Q7_Y),
            "stock full-width scalar * Q",
        );

        // Stock square root (types-core Tonelli-Shanks): exact root set.
        let value = Felt::from_hex(
            "0x03588e3f38a9e4f752b22bd4eb778d58e2d0a23b68c8b1c05d83ff6cd8ab9d04",
        )
        .unwrap();
        let root_a = Felt::from_hex(
            "0x03c1e9550e66958296d11b60f8e8e7a7ad990d07fa65d5f7652c4a6c87d4e3cc",
        )
        .unwrap();
        let root_b = Felt::from_hex(
            "0x043e16aaf1996a8e692ee49f071718585266f2f8059a2a089ad3b593782b1c35",
        )
        .unwrap();
        let stock_root = value.sqrt().expect("value is a residue");
        assert!(stock_root == root_a || stock_root == root_b, "stock sqrt returned {stock_root:#x}");
        assert!(Felt::THREE.sqrt().is_none(), "stock sqrt: 3 must have no square root");

        // Stock scalar inverse (num-bigint extended GCD): exact value.
        let s_value = Felt::from_hex(
            "0x0405c3191ab3883ef2b763af35bc5f5d15b3b4e99461d70e84c654a351a7c81b",
        )
        .unwrap();
        let expected_w = Felt::from_hex(
            "0x01ce0310e48aa17f713cbd8f8acc5a88703a359d2ef27d33ef95b8cfce4bcc91",
        )
        .unwrap();
        assert_eq!(mod_inverse(&s_value, &EC_ORDER), expected_w, "stock mod_inverse mismatch");
    }

    /// Batch MSM verification must accept valid batches, reject any tampering,
    /// and agree with per-item verification.
    #[test]
    fn batch_with_nonce_points_accepts_valid_and_rejects_tampered() {
        let seed = [7u8; 32];
        let mut state = 0xBA7C_4000_0DDB_A115u64;
        let mut items = Vec::new();
        for _ in 0..96 {
            let private_key = random_below_bound(&mut state) + Felt::ONE;
            let message = random_below_bound(&mut state);
            let k = rfc6979_generate_k(&message, &private_key, None);
            let signature = sign(&private_key, &message, &k).unwrap();
            let nonce_point = fixed_base_mul(&k).to_affine().unwrap();
            // The exact batch equation holds for the TRUE public key point
            // Q = private_key * G (its y-parity is fixed by the private key),
            // which is what a signer publishes under this scheme.
            let public_key_point = fixed_base_mul(&private_key).to_affine().unwrap();
            items.push((public_key_point, message, nonce_point, signature.s));
        }
        assert_eq!(verify_batch_with_nonce_points(&items, &seed).ok(), Some(true));
        // Per-item parity for a few entries.
        for (public_key_point, message, nonce_point, s) in items.iter().take(8) {
            assert_eq!(
                verify_with_pubkey_point(public_key_point, message, &nonce_point.x(), s).ok(),
                Some(true)
            );
        }

        // Any single tampering must sink the whole batch.
        let mut tampered = items.clone();
        tampered[41].3 = tampered[41].3 + Felt::ONE;
        assert_eq!(verify_batch_with_nonce_points(&tampered, &seed).ok(), Some(false));

        let mut tampered_message = items.clone();
        tampered_message[7].1 = tampered_message[7].1 + Felt::ONE;
        assert_eq!(verify_batch_with_nonce_points(&tampered_message, &seed).ok(), Some(false));

        // Wrong y-parity of a transmitted nonce point must fail (exact equation).
        let mut flipped_nonce = items.clone();
        flipped_nonce[3].2 = -&flipped_nonce[3].2;
        assert_eq!(verify_batch_with_nonce_points(&flipped_nonce, &seed).ok(), Some(false));

        // An invalid item mid-batch must error (exercises index remapping),
        // and values at exactly the range bound must match stock semantics.
        let mut mid_invalid = items.clone();
        mid_invalid[1].3 = Felt::ZERO;
        assert!(verify_batch_with_nonce_points(&mid_invalid, &seed).is_err());
        let bound = ELEMENT_UPPER_BOUND;
        assert!(verify_fast(&items[0].0.x(), &bound, &Felt::ONE, &Felt::ONE).is_err());
        assert_eq!(
            verify(&items[0].0.x(), &bound, &Felt::ONE, &Felt::ONE).ok(),
            verify_fast(&items[0].0.x(), &bound, &Felt::ONE, &Felt::ONE).ok()
        );

        // Empty batch is vacuously valid; range violations error out.
        assert_eq!(verify_batch_with_nonce_points(&[], &seed).ok(), Some(true));
        let mut bad_range = items[..4].to_vec();
        bad_range[2].3 = Felt::ZERO;
        assert!(verify_batch_with_nonce_points(&bad_range, &seed).is_err());
    }

    /// Off-curve wire points must be rejected, and the parity-bit batch variant
    /// must agree with the full-point variant.
    #[test]
    fn batch_rejects_off_curve_points_and_parity_variant_matches() {
        let seed = [9u8; 32];
        let mut state = 0x0FFC_0000_C0DE_0001u64;
        let mut point_items = Vec::new();
        let mut parity_items = Vec::new();
        for _ in 0..24 {
            let private_key = random_below_bound(&mut state) + Felt::ONE;
            let message = random_below_bound(&mut state);
            let k = rfc6979_generate_k(&message, &private_key, None);
            let signature = sign(&private_key, &message, &k).unwrap();
            let nonce_point = fixed_base_mul(&k).to_affine().unwrap();
            let public_key_point = fixed_base_mul(&private_key).to_affine().unwrap();
            let nonce_y_is_odd = nonce_point.y().to_bytes_le()[0] & 1 == 1;
            parity_items.push((
                public_key_point.clone(),
                message,
                nonce_point.x(),
                nonce_y_is_odd,
                signature.s,
            ));
            point_items.push((public_key_point, message, nonce_point, signature.s));
        }
        assert_eq!(verify_batch_with_nonce_points(&point_items, &seed).ok(), Some(true));
        assert_eq!(verify_batch_with_nonce_parity(&parity_items, &seed).ok(), Some(true));

        // Wrong parity bit flips R and must fail.
        let mut wrong_parity = parity_items.clone();
        wrong_parity[5].3 = !wrong_parity[5].3;
        assert_eq!(verify_batch_with_nonce_parity(&wrong_parity, &seed).ok(), Some(false));

        // Off-curve nonce point (y tweaked) must be rejected as InvalidR.
        let mut off_curve = point_items.clone();
        let broken = AffinePoint::new_unchecked(off_curve[2].2.x(), off_curve[2].2.y() + Felt::ONE);
        off_curve[2].2 = broken;
        assert!(matches!(
            verify_batch_with_nonce_points(&off_curve, &seed),
            Err(VerifyError::InvalidR)
        ));

        // Off-curve public key must be rejected on both single and batch paths.
        let bad_key = AffinePoint::new_unchecked(point_items[0].0.x(), point_items[0].0.y() + Felt::ONE);
        assert!(matches!(
            verify_with_pubkey_point(&bad_key, &point_items[0].1, &point_items[0].2.x(), &point_items[0].3),
            Err(VerifyError::InvalidPublicKey)
        ));
    }

    /// Every private-key -> public-key pair in the StarkEx precomputed vectors
    /// (generated by starkware's original tooling, independent of this crate)
    /// must match `fixed_base_mul`'s x-coordinate exactly.
    #[test]
    fn fixed_base_mul_matches_precomputed_starkex_keys() {
        let json_data = include_str!("../test-data/keys_precomputed.json");
        let key_map: std::collections::BTreeMap<String, String> =
            serde_json::from_str(json_data).expect("parse keys_precomputed.json");
        assert!(!key_map.is_empty());
        for (private_key, expected_public_key) in key_map {
            let public_key_point = fixed_base_mul(&Felt::from_hex(&private_key).unwrap())
                .to_affine()
                .expect("private key must not map to infinity");
            assert_eq!(
                public_key_point.x(),
                Felt::from_hex(&expected_public_key).unwrap(),
                "fixed_base_mul mismatch for private key {private_key}"
            );
        }
    }
}
