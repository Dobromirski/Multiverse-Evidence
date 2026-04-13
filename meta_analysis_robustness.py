"""
Meta-Analysis Robustness: Leave-One-Out Sensitivity
====================================================

Test if the combined kappa = 0.004 is robust to removing any single
study, especially Sinha 2010a which has the smallest error and
dominates the fit.

Also: group correlated measurements (same team/apparatus) and
re-weight accordingly.

Author: Zhivko Dobromirski
Date: 2026-04-02
"""

import numpy as np
from scipy.stats import norm, binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

studies = [
    {'id': 1, 'label': 'Sinha 2010a', 'platform': 'Photon laser+PM', 'team': 'Sinha',
     'kappa': 0.0073, 'sigma': 0.0018},
    {'id': 2, 'label': 'Sinha 2010b', 'platform': 'Photon atten+APD', 'team': 'Sinha',
     'kappa': 0.0034, 'sigma': 0.0038},
    {'id': 3, 'label': 'Sinha 2010c', 'platform': 'Photon heralded SP', 'team': 'Sinha',
     'kappa': 0.0064, 'sigma': 0.0120},
    {'id': 4, 'label': 'Soellner 2012', 'platform': 'Coherent + homodyne', 'team': 'Soellner',
     'kappa': 0.0020, 'sigma': 0.0040},
    {'id': 5, 'label': 'Jin 2017a', 'platform': 'NV center (M1)', 'team': 'Jin',
     'kappa': 0.0017, 'sigma': 0.0045},
    {'id': 6, 'label': 'Jin 2017b', 'platform': 'NV center (M2)', 'team': 'Jin',
     'kappa': -0.0017, 'sigma': 0.0042},
]


def meta_analysis(studies_list):
    """Random-effects DerSimonian-Laird meta-analysis."""
    n = len(studies_list)
    kappas = np.array([s['kappa'] for s in studies_list])
    sigmas = np.array([s['sigma'] for s in studies_list])
    weights_fe = 1.0 / sigmas**2

    # Fixed-effect estimate
    kappa_fe = np.sum(kappas * weights_fe) / np.sum(weights_fe)
    se_fe = 1.0 / np.sqrt(np.sum(weights_fe))

    # Heterogeneity
    Q = np.sum(weights_fe * (kappas - kappa_fe)**2)
    df = n - 1

    # tau^2 (DerSimonian-Laird)
    C = np.sum(weights_fe) - np.sum(weights_fe**2) / np.sum(weights_fe)
    tau_squared = max(0, (Q - df) / C) if C > 0 else 0

    # Random-effects
    weights_re = 1.0 / (sigmas**2 + tau_squared)
    kappa_re = np.sum(kappas * weights_re) / np.sum(weights_re)
    se_re = 1.0 / np.sqrt(np.sum(weights_re))
    z_re = kappa_re / se_re
    p_re = 2 * (1 - norm.cdf(abs(z_re)))

    # Direction test
    n_pos = np.sum(kappas > 0)
    p_sign = binomtest(int(n_pos), n, 0.5).pvalue if n > 1 else 1.0

    return {
        'n': n, 'kappa_fe': kappa_fe, 'se_fe': se_fe,
        'kappa_re': kappa_re, 'se_re': se_re,
        'z_re': z_re, 'p_re': p_re,
        'Q': Q, 'tau_squared': tau_squared,
        'n_positive': int(n_pos), 'p_sign': p_sign
    }


print("=" * 70)
print("META-ANALYSIS ROBUSTNESS (LEAVE-ONE-OUT)")
print("=" * 70)

# Full
full = meta_analysis(studies)
print(f"\n--- FULL (N={full['n']}) ---")
print(f"  kappa_RE = {full['kappa_re']:.5f} +/- {full['se_re']:.5f}")
print(f"  z = {full['z_re']:.2f}, p = {full['p_re']:.6f}")
print(f"  positive/total = {full['n_positive']}/{full['n']}, sign_p = {full['p_sign']:.3f}")

# Leave-one-out
print(f"\n--- LEAVE-ONE-OUT ---")
print(f"\n{'Removed':<20} {'N':>3} {'kappa_RE':>11} {'se_RE':>11} {'z':>7} {'p':>10}")
print("-" * 70)
loo_results = []
for i, s in enumerate(studies):
    remaining = [x for j, x in enumerate(studies) if j != i]
    result = meta_analysis(remaining)
    result['removed'] = s['label']
    loo_results.append(result)
    print(f"{s['label']:<20} {result['n']:>3} {result['kappa_re']:>+11.5f} "
          f"{result['se_re']:>11.5f} {result['z_re']:>+7.2f} {result['p_re']:>10.6f}")

# Leave-one-team-out
print(f"\n--- LEAVE-TEAM-OUT (downweight correlated experiments) ---")
print(f"\nSinha had 3 experiments (2010a/b/c) on same apparatus — correlated.")

teams = ['Sinha', 'Soellner', 'Jin']
for team in teams:
    remaining = [x for x in studies if x['team'] != team]
    if len(remaining) >= 2:
        r = meta_analysis(remaining)
        print(f"\nWithout {team} team ({r['n']} studies):")
        print(f"  kappa_RE = {r['kappa_re']:.5f} +/- {r['se_re']:.5f}")
        print(f"  z = {r['z_re']:.2f}, p = {r['p_re']:.6f}")

# Test: if Sinha 3 are treated as ONE (average them first)
print(f"\n--- TREAT SINHA AS SINGLE POINT ---")
sinha_studies = [s for s in studies if s['team'] == 'Sinha']
sinha_k = np.array([s['kappa'] for s in sinha_studies])
sinha_s = np.array([s['sigma'] for s in sinha_studies])
w = 1.0 / sinha_s**2
sinha_combined_k = np.sum(sinha_k * w) / np.sum(w)
sinha_combined_s = 1.0 / np.sqrt(np.sum(w))

combined = [
    {'label': 'Sinha team (combined)', 'kappa': sinha_combined_k, 'sigma': sinha_combined_s,
     'team': 'Sinha', 'platform': 'Photon'},
    {'label': 'Soellner 2012', 'kappa': 0.0020, 'sigma': 0.0040, 'team': 'Soellner', 'platform': 'Photon'},
    {'label': 'Jin 2017a', 'kappa': 0.0017, 'sigma': 0.0045, 'team': 'Jin', 'platform': 'NV'},
    {'label': 'Jin 2017b', 'kappa': -0.0017, 'sigma': 0.0042, 'team': 'Jin', 'platform': 'NV'},
]
r = meta_analysis(combined)
print(f"\n  Sinha combined: kappa = {sinha_combined_k:.5f} +/- {sinha_combined_s:.5f}")
print(f"  Meta-analysis (4 independent teams):")
print(f"    kappa_RE = {r['kappa_re']:.5f} +/- {r['se_re']:.5f}")
print(f"    z = {r['z_re']:.2f}, p = {r['p_re']:.6f}")

# Compare with prediction
print(f"\n{'=' * 70}")
print(f"COMPARISON WITH delta^2 PREDICTION")
print(f"{'=' * 70}")

delta = 0.0616
delta_sq = delta**2
print(f"\n  delta = {delta} (from DESI fit)")
print(f"  delta^2 = {delta_sq:.5f}")
print(f"\n  Full meta-analysis: kappa = {full['kappa_re']:.5f} -> deviation from delta^2: "
      f"{(full['kappa_re']-delta_sq)/full['se_re']:.2f} sigma")
print(f"  Without Sinha 2010a: kappa = {loo_results[0]['kappa_re']:.5f} -> "
      f"deviation: {(loo_results[0]['kappa_re']-delta_sq)/loo_results[0]['se_re']:.2f} sigma")
print(f"  Sinha-combined 4-team: kappa = {r['kappa_re']:.5f} -> "
      f"deviation: {(r['kappa_re']-delta_sq)/r['se_re']:.2f} sigma")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Forest plot with LOO
ax = axes[0]
labels = ['FULL'] + [f'no {s["label"]}' for s in studies] + ['Sinha-combined']
kappas_plot = [full['kappa_re']] + [r['kappa_re'] for r in loo_results] + [r['kappa_re']]
errors_plot = [full['se_re']] + [r['se_re'] for r in loo_results]
# Recompute Sinha-combined
r_sc = meta_analysis(combined)
errors_plot = errors_plot + [r_sc['se_re']]

y = np.arange(len(labels))
colors = ['black'] + ['blue']*6 + ['green']
for i, (k, e, c) in enumerate(zip(kappas_plot, errors_plot, colors)):
    ax.errorbar(k, y[i], xerr=1.96*e, fmt='o', color=c, capsize=4, markersize=8)

ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
ax.axvline(x=delta_sq, color='red', linestyle='--', alpha=0.7, label=f'delta^2 = {delta_sq:.4f}')
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('Combined kappa', fontsize=12)
ax.set_title('Leave-One-Out Sensitivity', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
ax.invert_yaxis()

# Sign pattern
ax = axes[1]
for i, s in enumerate(studies):
    color = 'blue' if s['kappa'] > 0 else 'red'
    ax.errorbar(s['kappa'], i, xerr=1.96*s['sigma'],
               fmt='o', color=color, capsize=4, markersize=8)
ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
ax.axvline(x=full['kappa_re'], color='green', linestyle='--', alpha=0.7,
          label=f'Full RE: {full["kappa_re"]:.4f}')
ax.axvline(x=delta_sq, color='red', linestyle=':', alpha=0.7,
          label=f'delta^2: {delta_sq:.4f}')
ax.set_yticks(range(6))
ax.set_yticklabels([s['label'] for s in studies], fontsize=8)
ax.set_xlabel('kappa', fontsize=12)
ax.set_title('Individual Studies', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('c:/Projects/Multiverse-Evidence/meta_analysis_robustness.png', dpi=150)
print(f"\nPlot saved: meta_analysis_robustness.png")
