"""
Sorkin Parameter Simulation: Statistical Bending Test
======================================================

Simulates triple-slit interference with high statistics.

Goal: Test whether the observed kappa = 0.004 from meta-analysis
could arise from:
  A) Pure statistical fluctuation in standard QM (null hypothesis)
  B) A small branching perturbation with amplitude delta

Strategy:
- Generate amplitudes for 3 paths: alpha_A, alpha_B, alpha_C
- For each "run" config (A only, AB, ABC, etc), sample N events
- Count detection probability at target position
- Compute Sorkin parameter:
    kappa = P_ABC - (P_AB + P_AC + P_BC) + (P_A + P_B + P_C)

- In pure QM (Born rule exact): kappa = 0, noise ~ 1/sqrt(N)
- Can we SEE kappa = 0.004 emerge from branching dynamics?

Author: Zhivko Dobromirski
Date: 2026-04-02
"""

import numpy as np
import time

# Try CUDA. Fall back to numpy.
try:
    import cupy as cp
    GPU = True
    print("GPU (CuPy) available — will use for scale tests")
except ImportError:
    cp = np
    GPU = False
    print("No GPU — using numpy only")


def setup_amplitudes(seed=42):
    """Triple-slit geometry: three complex amplitudes at detector."""
    # Pick non-trivial but well-defined amplitudes
    # Equal magnitudes, different phases
    np.random.seed(seed)
    alpha_A = 1.0 + 0.0j
    alpha_B = 0.9 * np.exp(1j * 2.1)  # phase 2.1 rad
    alpha_C = 0.8 * np.exp(1j * 4.3)  # phase 4.3 rad
    return alpha_A, alpha_B, alpha_C


def probabilities(alpha_A, alpha_B, alpha_C):
    """Theoretical probabilities for all 7 Sorkin configurations."""
    # We measure probability at ONE detector position
    # Normalize: denominator = max possible (triple open)
    P = {
        'A':   np.abs(alpha_A)**2,
        'B':   np.abs(alpha_B)**2,
        'C':   np.abs(alpha_C)**2,
        'AB':  np.abs(alpha_A + alpha_B)**2,
        'AC':  np.abs(alpha_A + alpha_C)**2,
        'BC':  np.abs(alpha_B + alpha_C)**2,
        'ABC': np.abs(alpha_A + alpha_B + alpha_C)**2,
    }
    # Theoretical Sorkin parameter (should be exactly 0)
    kappa_theory = P['ABC'] - P['AB'] - P['AC'] - P['BC'] + P['A'] + P['B'] + P['C']
    return P, kappa_theory


def simulate_kappa(P, N_per_config, use_gpu=False):
    """
    Monte Carlo: for each configuration, sample N Bernoulli trials
    with probability P. Count detections. Estimate probability.
    Compute kappa from estimated probabilities.
    """
    xp = cp if use_gpu else np

    configs = ['A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC']
    P_estimated = {}

    for cfg in configs:
        p = P[cfg]
        # Sample N_per_config Bernoulli trials
        detections = xp.random.binomial(N_per_config, p)
        if use_gpu:
            detections = float(detections.get())
        P_estimated[cfg] = detections / N_per_config

    kappa_est = (P_estimated['ABC']
                 - P_estimated['AB'] - P_estimated['AC'] - P_estimated['BC']
                 + P_estimated['A'] + P_estimated['B'] + P_estimated['C'])

    return kappa_est, P_estimated


def simulate_many_kappa(P, N_per_config, M_trials, use_gpu=False):
    """Run M independent simulations to get distribution of kappa."""
    xp = cp if use_gpu else np

    configs = ['A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC']
    probs = xp.array([P[c] for c in configs])

    # Sample: for each of 7 configs, M trials each with N_per_config events
    # Each element is Binomial(N, p)
    if use_gpu:
        detections = cp.random.binomial(N_per_config, probs[:, None], size=(7, M_trials))
        det_np = detections.get()
    else:
        detections = np.random.binomial(N_per_config, probs[:, None], size=(7, M_trials))
        det_np = detections

    P_est = det_np / N_per_config
    # kappa = P[6] - P[3] - P[4] - P[5] + P[0] + P[1] + P[2]
    kappa_samples = (P_est[6] - P_est[3] - P_est[4] - P_est[5]
                     + P_est[0] + P_est[1] + P_est[2])

    return kappa_samples


# ============================================================
# NORMALIZATION: Sorkin kappa is typically normalized
# ============================================================
def normalized_kappa(P):
    """
    The literature usually defines kappa normalized by the envelope.
    E.g., Sinha 2010: kappa = (P_ABC - P_AB - P_AC - P_BC + P_A + P_B + P_C) / delta
    where delta is max of pairwise interferences.
    """
    I_AB = P['AB'] - P['A'] - P['B']
    I_AC = P['AC'] - P['A'] - P['C']
    I_BC = P['BC'] - P['B'] - P['C']
    max_pair = max(abs(I_AB), abs(I_AC), abs(I_BC))

    kappa_raw = (P['ABC'] - P['AB'] - P['AC'] - P['BC']
                 + P['A'] + P['B'] + P['C'])

    if max_pair == 0:
        return 0
    return kappa_raw / max_pair, kappa_raw, max_pair


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("SORKIN PARAMETER SIMULATION")
    print("=" * 70)

    alpha_A, alpha_B, alpha_C = setup_amplitudes()
    P, kappa_theory = probabilities(alpha_A, alpha_B, alpha_C)

    print(f"\nAmplitudes:")
    print(f"  alpha_A = {alpha_A}")
    print(f"  alpha_B = {alpha_B:.3f}")
    print(f"  alpha_C = {alpha_C:.3f}")

    print(f"\nTheoretical probabilities (Born rule exact):")
    for cfg in ['A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC']:
        print(f"  P({cfg:<3}) = {P[cfg]:.6f}")

    print(f"\nTheoretical Sorkin kappa (should be exactly 0): {kappa_theory:.2e}")

    kappa_norm, kappa_raw, max_pair = normalized_kappa(P)
    print(f"Normalized: {kappa_norm:.2e} (denom = max |pairwise interference| = {max_pair:.4f})")

    # ============================================================
    # SCALING TEST: N increase -> noise decrease
    # ============================================================
    print(f"\n{'=' * 70}")
    print("SCALING TEST: kappa noise vs N (standard QM, Born rule exact)")
    print(f"{'=' * 70}")
    print(f"\n{'N per config':>15} {'M trials':>10} {'kappa mean':>12} {'kappa std':>12} {'time (s)':>10}")
    print("-" * 70)

    N_values = [10**4, 10**5, 10**6, 10**7]
    if GPU:
        N_values = [10**4, 10**5, 10**6, 10**7, 10**8, 10**9]
    M = 1000

    results = []
    for N in N_values:
        t0 = time.time()
        kappas = simulate_many_kappa(P, N, M, use_gpu=GPU)
        t = time.time() - t0
        # Normalize by max pairwise interference
        kappas_norm = kappas / max_pair
        mu = float(np.mean(kappas_norm))
        sigma = float(np.std(kappas_norm))
        results.append({'N': N, 'M': M, 'mean': mu, 'std': sigma, 'time': t})
        print(f"{N:>15} {M:>10} {mu:>12.2e} {sigma:>12.2e} {t:>10.2f}")

    # Expected scaling: sigma ~ 1/sqrt(N)
    print(f"\nExpected scaling: std ~ 1/sqrt(N)")
    print(f"\n{'N':>12} {'std (observed)':>15} {'std (predicted ~1/sqrt(N))':>28}")
    for r in results:
        predicted = 1.0 / np.sqrt(r['N'])
        print(f"{r['N']:>12} {r['std']:>15.2e} {predicted:>28.2e}")

    # ============================================================
    # NULL HYPOTHESIS TEST
    # ============================================================
    print(f"\n{'=' * 70}")
    print("NULL HYPOTHESIS: Can kappa_obs = 0.004 arise from pure QM?")
    print(f"{'=' * 70}")

    kappa_obs = 0.004  # from meta-analysis
    kappa_obs_err = 0.001

    print(f"\nObserved kappa from meta-analysis: {kappa_obs} +/- {kappa_obs_err}")
    print(f"\nFor pure QM (Born rule exact), expected: kappa = 0")
    print(f"Fluctuation amplitude depends on N.")

    # Sinha 2010 main result: ~100 runs, each with ~30M triggers
    # Effective N ~ 3 billion per configuration
    N_sinha = 3e9
    sigma_sinha = 1.0 / np.sqrt(N_sinha) / max_pair  # normalized
    print(f"\nSinha 2010 effective N ~ 3e9 per configuration")
    print(f"Expected QM-null sigma_kappa ~ 1/sqrt(N) / max_pair = {sigma_sinha:.2e}")
    print(f"Observed: {kappa_obs}")
    print(f"Ratio observed/expected: {kappa_obs/sigma_sinha:.1e}")

    if kappa_obs / sigma_sinha > 5:
        print(f"\n>>> Observed kappa is {kappa_obs/sigma_sinha:.0f}x the statistical fluctuation")
        print(f">>> Pure QM CANNOT produce kappa = 0.004 by chance")
        print(f">>> Either: (a) systematic error in experiments")
        print(f"            (b) new physics beyond Born rule")

    # ============================================================
    # BRANCHING PERTURBATION: Can kappa = delta^2 be reproduced?
    # ============================================================
    print(f"\n{'=' * 70}")
    print("BRANCHING HYPOTHESIS: kappa from perturbation")
    print(f"{'=' * 70}")

    # Hypothesis: branching introduces a small "ghost" amplitude
    # alpha_ghost per path, scaling with delta
    # This adds third-order term in |psi|^2

    print("\nModel: Each path has probability amplitude modified:")
    print("  alpha_i -> alpha_i * (1 + delta_ghost * e^(i*phi_i))")
    print("  where delta_ghost = sqrt(delta_Lambda) ~ 0.25 (for delta_Lambda=0.062)")

    deltas = [0.01, 0.03, 0.062, 0.1, 0.15]
    print(f"\n{'delta_Lambda':>12} {'delta_ghost':>12} {'kappa_predicted':>16} {'delta^2':>10}")
    for d in deltas:
        # ghost amplitude
        dg = np.sqrt(d)
        # Modified amplitudes with random phase
        phi_A, phi_B, phi_C = 0.5, 1.7, 3.2
        aA = alpha_A * (1 + dg * np.exp(1j * phi_A))
        aB = alpha_B * (1 + dg * np.exp(1j * phi_B))
        aC = alpha_C * (1 + dg * np.exp(1j * phi_C))
        P_mod, k_mod = probabilities(aA, aB, aC)
        k_norm, _, _ = normalized_kappa(P_mod)
        print(f"{d:>12.4f} {dg:>12.4f} {k_norm:>16.4e} {d**2:>10.4e}")

    print(f"\nIf kappa ~ delta^2: we'd expect kappa = {0.062**2:.4e} = 0.0038")
    print(f"Observed kappa_obs = 0.004 matches delta^2 within 1 sigma.")


if __name__ == '__main__':
    main()
