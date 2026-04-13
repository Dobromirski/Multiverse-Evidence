"""
IGM Heating Consistency Check
==============================

If vacuum branching releases energy, some fraction couples to baryons
and heats the intergalactic medium. Check against CSL bounds.

Standard CSL bound: dE/dt per baryon < ~10^-34 W
(from non-observation of excess IGM/ISM heating)

Our model's energy release rate per m^3:
  ~9.4e23 events/s x E_tension(E_crit) ~ 6.6e-29 J/m^3/s

Question: What fraction of this can couple to baryons without
violating observational bounds?

Author: Zhivko Dobromirski
Date: 2026-04-02
"""

import numpy as np

# Constants
hbar = 1.054571817e-34
G = 6.67430e-11
c = 2.99792458e8
k_B = 1.380649e-23
m_N = 1.6726e-27  # nucleon mass
eV = 1.602176634e-19
H0 = 67.4 * 1000 / 3.086e22
age = 4.35e17
rho_crit = 3 * H0**2 * c**2 / (8 * np.pi * G)

print("=" * 70)
print("IGM HEATING CONSISTENCY CHECK")
print("=" * 70)

# ============================================================
# Our model's predictions (from vacuum_branching_noise.py)
# ============================================================
E_crit = 6.45e-12  # J ~ 40 MeV

# Rate and energy per event
total_rate_per_V = G * E_crit**6 / (6 * hbar**5 * c**8)  # events/m^3/s
E_tension_crit = G * E_crit**3 / (hbar * c**5)
energy_rate = total_rate_per_V * E_tension_crit  # J/m^3/s

print(f"\n--- Our Model ---")
print(f"E_critical: {E_crit/eV/1e6:.1f} MeV")
print(f"Collapse events per m^3 per s: {total_rate_per_V:.2e}")
print(f"Energy per collapse: {E_tension_crit:.2e} J")
print(f"Energy release rate: {energy_rate:.2e} J/m^3/s")

# ============================================================
# Compare with critical density and cosmological time
# ============================================================
total_energy_over_age = energy_rate * age
frac_of_critical = total_energy_over_age / rho_crit
frac_of_Lambda = total_energy_over_age / 5.96e-10

print(f"\n--- Integrated Over Cosmic History ---")
print(f"Total energy released per m^3 over age of universe: {total_energy_over_age:.2e} J/m^3")
print(f"Fraction of critical density: {frac_of_critical:.2%}")
print(f"Fraction of observed Lambda: {frac_of_Lambda:.2%}")

# ============================================================
# IGM heating bounds (CSL literature)
# ============================================================
# Typical CSL bound per nucleon: dE/dt < ~10^-34 W
# Derived from non-observation of excess heating in IGM (~10^4 K, stable)
# Reference: Adler, Bassi, Donadi 2013; Laloe et al. 2014

bound_per_baryon_W = 1e-34  # W = J/s per nucleon

# Baryon density in average universe (not in galaxies)
# Omega_b ~ 0.049, rho_crit ~ 8.6e-27 kg/m^3
# n_b ~ Omega_b * rho_crit / m_N
n_b_average = 0.049 * 8.6e-27 / m_N  # baryons/m^3
print(f"\n--- IGM Heating Bounds ---")
print(f"Average baryon density in universe: {n_b_average:.2e} /m^3")
print(f"CSL bound: dE/dt per baryon < {bound_per_baryon_W:.1e} W")
print(f"Total allowed heating rate per m^3: {bound_per_baryon_W * n_b_average:.2e} J/m^3/s")

allowed_rate = bound_per_baryon_W * n_b_average

# ============================================================
# Maximum coupling fraction
# ============================================================
max_coupling = allowed_rate / energy_rate
print(f"\n--- Compatibility ---")
print(f"Our energy release: {energy_rate:.2e} J/m^3/s")
print(f"Maximum allowed baryon heating: {allowed_rate:.2e} J/m^3/s")
print(f"Maximum fraction that can couple to baryons: {max_coupling:.2e}")
print(f"  (If more than {max_coupling*100:.0e}% of collapse energy couples")
print(f"   to baryons, the model is ruled out by IGM heating)")

if max_coupling > 1:
    print(f"\n  >>> COMFORTABLE: model can couple fully without violating bound")
elif max_coupling > 0.01:
    print(f"\n  >>> OK: model tolerates {max_coupling*100:.2f}% coupling")
elif max_coupling > 1e-6:
    print(f"\n  >>> TIGHT: only {max_coupling*100:.2e}% coupling allowed")
else:
    print(f"\n  >>> VERY TIGHT: coupling must be < {max_coupling:.2e}")

# ============================================================
# Alternative check: what if we assume most energy goes to Lambda?
# ============================================================
print(f"\n--- 'Energy goes to Lambda' interpretation ---")
print(f"If branching energy mostly feeds vacuum energy:")
print(f"  Lambda contribution from branching over cosmic time: {total_energy_over_age:.2e} J/m^3")
print(f"  Observed Lambda: 5.96e-10 J/m^3")
print(f"  This would be: {frac_of_Lambda*100:.1f}% of observed Lambda")
print(f"")
if 0.01 < frac_of_Lambda < 1.0:
    print(f"  >>> Plausible: branching could contribute a significant fraction")
    print(f"      of Lambda without being the entire Lambda")
elif frac_of_Lambda > 1.0:
    print(f"  >>> Over-produces Lambda by factor {frac_of_Lambda:.1f}")
    print(f"      Need efficiency < {1/frac_of_Lambda:.2f} for Lambda route")
else:
    print(f"  >>> Contributes negligibly to Lambda ({frac_of_Lambda*100:.3f}%)")

# ============================================================
# ENERGY BUDGET SUMMARY
# ============================================================
# If all collapse energy went into Lambda, we'd have 5% of observed
# This leaves 95% of Lambda unexplained by branching
# OR if efficiency is higher, we hit IGM bound

print(f"\n{'='*70}")
print("ENERGY BUDGET")
print(f"{'='*70}")
print(f"""
Per m^3 per second, our model releases {energy_rate:.2e} J.
Integrated over cosmic history: {total_energy_over_age:.2e} J/m^3
= {frac_of_Lambda*100:.1f}% of observed Lambda

Where this energy could go (efficiency fractions):

Channel              | Max allowed fraction | Note
---------------------|----------------------|------
Baryon heating (IGM) | {max_coupling:.2e}         | CSL bound
Lambda (vacuum)      | < 1.0                | Model-dependent
Gravitational waves  | < ~1e-7              | NANOGrav bound (from naive calc)
Dark matter??        | unknown              | Speculative
CMB distortions      | < 1e-5               | COBE/Planck bound
Neutrinos            | model-dependent      | Hard to constrain

The {frac_of_Lambda*100:.1f}% Lambda contribution is the most intriguing
because it's in a range where the model could matter but not dominate.
""")

print("=" * 70)
print("VERDICT")
print("=" * 70)
print(f"""
1. Our model is NOT immediately killed by IGM heating.
   Only {max_coupling*100:.1e}% of collapse energy can couple to baryons.

2. If ~5% of branching energy goes to Lambda, it's a measurable
   but non-dominant contribution. This matches our DESI fit
   (delta = 0.062, i.e., Lambda increased by 6% from early universe).

3. The remaining 95% must go into channels with very small baryon
   coupling — potentially gravitational waves (suppressed), dark
   sector interactions, or back into vacuum modes.

4. This is consistent with hypothesis but does NOT uniquely confirm it.
   The model stays alive but cannot claim unique explanatory power.
""")
