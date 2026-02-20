# Physics Notes

This document is a deeper reference for the physics model used in the visualizer.

## 1. What The Dots Represent

Each dot is a Monte Carlo sample drawn from the probability density |psi|^2. Dots are not electron trajectories. Dense regions indicate higher probability of finding an electron.

## 2. The Hydrogenic Model

In atomic units the time-independent Schrodinger equation is:

[-1/2 * nabla^2 - Z/r] psi(r) = E psi(r)

Separation of variables in spherical coordinates yields:

psi(n,l,m)(r,theta,phi) = R_nl(r) * Y_lm(theta,phi)

Quantum numbers:

- n = 1, 2, 3, ...
- l = 0 to n-1
- m = -l to +l

Energy depends only on n:

E_n = -Z^2 / (2 n^2)

This implies all states with the same n are degenerate in the hydrogenic model.

## 3. Radial Structure

The radial probability density is:

P(r) = r^2 |R_nl(r)|^2

This explains why even s orbitals do not peak at r = 0. Radial nodes are values of r where R_nl(r) = 0.

## 4. Angular Structure

The angular part Y_lm(theta,phi) is a spherical harmonic. It defines lobes and angular nodes. The number of angular nodes is l.

## 5. Real Orbital Basis

Complex spherical harmonics include a phase factor exp(i m phi), so |Y_lm|^2 is independent of phi. This produces azimuthally symmetric shapes (rings and shells) for a single m state.

Chemistry textbooks often show real orbitals formed by linear combinations of m and -m. These real combinations have explicit phi dependence and produce the familiar p, d, and f lobes. The visualizer includes a Real basis option that uses these combinations for orbital and superposition views.

## 6. Superposition And Time Dependence

A stationary eigenstate has only a global phase exp(-i E t). The probability density |psi|^2 is time independent, so a single orbital does not animate.

Superposition of at least two eigenstates with different energies produces time dependence:

psi(r,t) = a * psi1(r) + b * psi2(r) * exp(-i * DeltaE * t)

The probability density includes an interference term:

|psi|^2 = |a psi1|^2 + |b psi2|^2 + 2 Re[a b* psi1 psi2* exp(-i DeltaE t)]

If DeltaE = 0 (degenerate states with the same n), the density is static. The UI may loop the phase for visual continuity, which is not physical time evolution.

## 7. Multi-Electron Densities (LDA)

When available, the visualizer uses OpenMX LDA radial wavefunctions and occupancy for each orbital. Total and valence density views are built by summing occupied orbitals with weights. Single-orbital view uses the selected radial function and Y_lm. LDA data is not m-resolved, so valence lobe mode uses m = 0 as a projection.

## 8. Dots And Bubbles

Dots show Monte Carlo samples. Bubbles reconstruct a smooth surface from those samples. When a phase sign is defined (single orbital and superposition), positive regions are shown in red and negative regions in blue. Density-only modes do not have a sign, so only one surface is shown.

## 9. Sampling Method

Sampling uses rejection sampling in spherical coordinates. Radial samples are drawn from a CDF built from |R_nl|^2 and r^2. Angular samples are accepted according to |Y_lm|^2.

## 10. Fallbacks And Approximations

- PSLibrary data is used as a fallback for single-orbital mode.
- Superposition uses hydrogenic orbitals for any element and scales coordinates by 1/Z.
- This is not a full time-dependent many-body solver.
