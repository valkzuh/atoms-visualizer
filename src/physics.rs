/// Hydrogen atom quantum physics calculations
/// Based on the Schrödinger equation for hydrogen-like atoms

use std::f32::consts::PI;

/// Represents quantum numbers (n, l, m_l)
/// n: Principal quantum number (1, 2, 3, ...)
/// l: Azimuthal quantum number (0 to n-1)
/// m_l: Magnetic quantum number (-l to l)
#[derive(Debug, Clone, Copy)]
pub struct QuantumNumbers {
    pub n: u32,
    pub l: u32,
    pub m_l: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AngularBasis {
    Complex,
    Real,
}

impl AngularBasis {
    pub fn from_query(value: Option<&str>) -> Self {
        match value.unwrap_or("complex").to_lowercase().as_str() {
            "real" => AngularBasis::Real,
            _ => AngularBasis::Complex,
        }
    }
}

impl QuantumNumbers {
    pub fn new(n: u32, l: u32, m_l: i32) -> Option<Self> {
        // Validate quantum numbers
        if n == 0 || l >= n || m_l.abs() > l as i32 {
            return None;
        }
        Some(QuantumNumbers { n, l, m_l })
    }
}

/// Bohr radius — dimensionless (all distances are in units of a₀)
const BOHR_RADIUS: f32 = 1.0;

/// Calculate the radial wavefunction for hydrogen
/// R_nl(r) for hydrogen atom
pub fn radial_wavefunction(r: f32, n: u32, l: u32) -> f32 {
    if r < 0.0 {
        return 0.0;
    }

    let n_f = n as f32;
    let l_f = l as f32;
    let rho = 2.0 * r / (n_f * BOHR_RADIUS);

    if rho < 0.0 {
        return 0.0;
    }

    // Normalization constant:
    // (2 / (n a0))^(3/2) * sqrt((n-l-1)! / (2n (n+l)!))
    let norm = (2.0 / (n_f * BOHR_RADIUS)).powf(1.5);
    let norm = norm * (factorial(n - l - 1) as f32 / (2.0 * n_f * factorial(n + l) as f32)).sqrt();

    // Exponential decay
    let exp_part = (-rho / 2.0).exp();

    // Radial polynomial (Laguerre polynomial part)
    let poly = laguerre_polynomial(rho, n - l - 1, 2 * l + 1);

    // Multiply by rho^l (rho = 2r / (n a0))
    let rho_power = rho.powf(l_f);

    norm * rho_power * exp_part * poly
}

/// Complex spherical harmonic Y_lm(theta, phi) with Condon-Shortley phase.
/// Returns (re, im).
pub fn spherical_harmonic(theta: f32, phi: f32, l: u32, m_l: i32) -> (f32, f32) {
    let l_f = l as f32;
    let m_abs = m_l.abs() as u32;
    let cos_theta = theta.cos();

    let legendre = associated_legendre(cos_theta, l, m_abs);
    let norm = ((2.0 * l_f + 1.0) / (4.0 * PI)).sqrt();
    let norm = norm
        * (factorial(l - m_abs) as f32 / factorial(l + m_abs) as f32).sqrt();

    let phase = m_abs as f32 * phi;
    let (s, c) = phase.sin_cos();
    let base_re = norm * legendre * c;
    let base_im = norm * legendre * s;

    if m_l >= 0 {
        let cs = if m_abs % 2 == 0 { 1.0 } else { -1.0 };
        (cs * base_re, cs * base_im)
    } else {
        let sign = if m_abs % 2 == 0 { 1.0 } else { -1.0 };
        (sign * base_re, -sign * base_im)
    }
}

/// Real-valued spherical harmonic basis used for chemistry-style orbitals.
/// m > 0 -> cos-like (Re), m < 0 -> sin-like (Im), m = 0 -> Y_l0
pub fn real_spherical_harmonic(theta: f32, phi: f32, l: u32, m_l: i32) -> f32 {
    if m_l == 0 {
        return spherical_harmonic(theta, phi, l, 0).0;
    }
    let m_abs = m_l.abs();
    let (re, im) = spherical_harmonic(theta, phi, l, m_abs as i32);
    let scale = 2.0_f32.sqrt();
    if m_l > 0 {
        scale * re
    } else {
        scale * im
    }
}

/// Angular wavefunction component |Y_lm(theta, phi)|
pub fn angular_wavefunction(theta: f32, phi: f32, l: u32, m_l: i32) -> f32 {
    let (re, im) = spherical_harmonic(theta, phi, l, m_l);
    (re * re + im * im).sqrt()
}

pub fn angular_wavefunction_basis(
    theta: f32,
    phi: f32,
    l: u32,
    m_l: i32,
    basis: AngularBasis,
) -> f32 {
    match basis {
        AngularBasis::Complex => angular_wavefunction(theta, phi, l, m_l),
        AngularBasis::Real => real_spherical_harmonic(theta, phi, l, m_l).abs(),
    }
}

/// Calculate the probability density |ψ|² for a given position in spherical coordinates
pub fn probability_density(r: f32, theta: f32, phi: f32, qn: QuantumNumbers) -> f32 {
    let radial = radial_wavefunction(r, qn.n, qn.l);
    let angular = angular_wavefunction(theta, phi, qn.l, qn.m_l);

    let wavefunction = radial * angular;
    wavefunction * wavefunction
}

pub fn probability_density_basis(
    r: f32,
    theta: f32,
    phi: f32,
    qn: QuantumNumbers,
    basis: AngularBasis,
) -> f32 {
    let radial = radial_wavefunction(r, qn.n, qn.l);
    let angular = angular_wavefunction_basis(theta, phi, qn.l, qn.m_l, basis);

    let wavefunction = radial * angular;
    wavefunction * wavefunction
}

/// Generate sample points from probability distribution for an orbital
pub fn generate_orbital_samples(
    qn: QuantumNumbers,
    num_samples: usize,
    max_radius: f32,
) -> Vec<(f32, f32, f32)> {
    let mut samples = Vec::with_capacity(num_samples);
    let mut rng = rand::thread_rng();

    use rand::Rng;

    // Compute maximum probability density once before the rejection loop
    let max_prob = find_max_probability(qn, max_radius);

    let mut accepted = 0;
    let mut attempts = 0;
    let max_attempts = num_samples * 100; // Prevent infinite loops

    while accepted < num_samples && attempts < max_attempts {
        attempts += 1;

        // Volume-weighted radial sampling: r ~ r² dr via cube-root transform.
        // This gives a uniform 3D spatial proposal, so rejection weight is |ψ|² alone.
        let r = max_radius * rng.gen::<f32>().powf(1.0 / 3.0);

        // Full-sphere theta: cos(theta) uniform in [-1, 1]
        let cos_theta = rng.gen::<f32>() * 2.0 - 1.0;
        let theta = cos_theta.acos();

        let phi = rng.gen::<f32>() * 2.0 * PI;

        // Rejection sampling: accept with probability proportional to |ψ|²
        let prob_density = probability_density(r, theta, phi, qn);

        if rng.gen::<f32>() < prob_density / max_prob {
            // Convert spherical to Cartesian coordinates
            let x = r * theta.sin() * phi.cos();
            let y = r * theta.sin() * phi.sin();
            let z = r * theta.cos();

            samples.push((x, y, z));
            accepted += 1;
        }
    }

    samples
}

pub fn generate_orbital_samples_basis(
    qn: QuantumNumbers,
    num_samples: usize,
    max_radius: f32,
    basis: AngularBasis,
) -> Vec<(f32, f32, f32)> {
    let mut samples = Vec::with_capacity(num_samples);
    let mut rng = rand::thread_rng();

    use rand::Rng;

    let max_prob = find_max_probability_basis(qn, max_radius, basis);

    let mut accepted = 0;
    let mut attempts = 0;
    let max_attempts = num_samples * 100;

    while accepted < num_samples && attempts < max_attempts {
        attempts += 1;

        let r = max_radius * rng.gen::<f32>().powf(1.0 / 3.0);
        let cos_theta = rng.gen::<f32>() * 2.0 - 1.0;
        let theta = cos_theta.acos();
        let phi = rng.gen::<f32>() * 2.0 * PI;

        let prob_density = probability_density_basis(r, theta, phi, qn, basis);

        if rng.gen::<f32>() < prob_density / max_prob {
            let x = r * theta.sin() * phi.cos();
            let y = r * theta.sin() * phi.sin();
            let z = r * theta.cos();
            samples.push((x, y, z));
            accepted += 1;
        }
    }

    samples
}

/// Find approximate maximum probability density for rejection sampling.
/// Scans a 2D (r, theta) grid. Uses quadratic r-spacing to sample densely
/// near the nucleus, where s-type orbitals have their peak.
pub fn find_max_probability(qn: QuantumNumbers, max_radius: f32) -> f32 {
    let mut max_prob = 0.0_f32;
    let r_steps = 100;
    let theta_steps = 20;

    for i in 0..r_steps {
        // Quadratic spacing in r: dense near nucleus, sparse at large r
        let t = (i as f32 + 1.0) / (r_steps as f32);
        let r = max_radius * t * t;
        for j in 0..theta_steps {
            let theta = (j as f32 + 0.5) / (theta_steps as f32) * PI;
            let prob = probability_density(r, theta, 0.0, qn);
            if prob > max_prob {
                max_prob = prob;
            }
        }
    }

    // Explicitly probe very close to the nucleus (catches s-orbital peak at r→0)
    let near_nucleus = probability_density(max_radius * 1e-4, PI / 2.0, 0.0, qn);
    max_prob = max_prob.max(near_nucleus);

    max_prob.max(1e-30) // Guard against division by zero
}

pub fn find_max_probability_basis(
    qn: QuantumNumbers,
    max_radius: f32,
    basis: AngularBasis,
) -> f32 {
    let mut max_prob = 0.0_f32;
    let r_steps = 100;
    let theta_steps = 20;

    for i in 0..r_steps {
        let t = (i as f32 + 1.0) / (r_steps as f32);
        let r = max_radius * t * t;
        for j in 0..theta_steps {
            let theta = (j as f32 + 0.5) / (theta_steps as f32) * PI;
            let prob = probability_density_basis(r, theta, 0.0, qn, basis);
            if prob > max_prob {
                max_prob = prob;
            }
        }
    }

    let near_nucleus =
        probability_density_basis(max_radius * 1e-4, PI / 2.0, 0.0, qn, basis);
    max_prob = max_prob.max(near_nucleus);

    max_prob.max(1e-30)
}

/// Calculate factorial of a u32
pub fn factorial(n: u32) -> u64 {
    (1..=n as u64).product()
}

/// Associated Legendre polynomial P^m_n(x)
pub fn associated_legendre(x: f32, n: u32, m: u32) -> f32 {
    if m > n {
        return 0.0;
    }

    let m_f = m as f32;

    // Base cases
    if m == 0 {
        return legendre_polynomial(x, n);
    }

    let x_sq = x * x;

    // Using recurrence relation for associated Legendre polynomials
    let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
    let pmm = sign * (1.0 - x_sq).powf(m_f / 2.0) * factorial_double(2 * m - 1) as f32;

    if n == m {
        return pmm;
    }

    let pm1m = x * (2.0 * m_f + 1.0) * pmm;

    if n == m + 1 {
        return pm1m;
    }

    // Recurrence relation
    let mut pmn = pmm;
    let mut pm1n = pm1m;

    for i in (m + 2)..=n {
        let i_f = i as f32;
        let pn = ((2.0 * i_f - 1.0) * x * pm1n - (i_f + m_f - 1.0) * pmn) / (i_f - m_f);
        pmn = pm1n;
        pm1n = pn;
    }

    pm1n
}

/// Legendre polynomial P_n(x)
pub fn legendre_polynomial(x: f32, n: u32) -> f32 {
    match n {
        0 => 1.0,
        1 => x,
        _ => {
            let mut p0 = 1.0;
            let mut p1 = x;

            for i in 2..=n {
                let i_f = i as f32;
                let p_new = ((2.0 * i_f - 1.0) * x * p1 - (i_f - 1.0) * p0) / i_f;
                p0 = p1;
                p1 = p_new;
            }

            p1
        }
    }
}

/// Double factorial n!! = n * (n-2) * (n-4) * ... * 1 or 2
pub fn factorial_double(n: u32) -> u64 {
    let mut result = 1u64;
    let mut i = n as i32;

    while i > 0 {
        result *= i as u64;
        i -= 2;
    }

    result
}

/// Generalized Laguerre polynomial L^a_n(x)
pub fn laguerre_polynomial(x: f32, n: u32, alpha: u32) -> f32 {
    if n == 0 {
        return 1.0;
    }

    let mut l0 = 1.0;
    let mut l1 = 1.0 + alpha as f32 - x;

    if n == 1 {
        return l1;
    }

    for i in 2..=n {
        let i_f = i as f32;
        let alpha_f = alpha as f32;

        let l_new = ((2.0 * i_f - 1.0 + alpha_f - x) * l1 - (i_f - 1.0 + alpha_f) * l0) / i_f;
        l0 = l1;
        l1 = l_new;
    }

    l1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_numbers() {
        assert!(QuantumNumbers::new(1, 0, 0).is_some());
        assert!(QuantumNumbers::new(2, 1, -1).is_some());
        assert!(QuantumNumbers::new(2, 1, 0).is_some());
        assert!(QuantumNumbers::new(2, 1, 1).is_some());
        
        assert!(QuantumNumbers::new(0, 0, 0).is_none());
        assert!(QuantumNumbers::new(2, 2, 0).is_none());
        assert!(QuantumNumbers::new(2, 1, 2).is_none());
    }

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
        assert_eq!(factorial(5), 120);
    }

    #[test]
    fn test_radial_wavefunction() {
        let r = BOHR_RADIUS;
        let psi = radial_wavefunction(r, 1, 0);
        assert!(psi > 0.0);
        assert!(!psi.is_nan());
    }
}
