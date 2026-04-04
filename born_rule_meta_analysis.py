"""
Meta-Analysis of Born Rule Tests (Sorkin Parameter kappa)
=========================================================

Posoka 7: Is there a systematic bias in kappa across experiments?

If combinatorial pressure exists, kappa should be consistently
positive or negative (not randomly distributed around zero).

Author: Zhivko Dobromirski
Date: 2026-04-02
"""

import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# DATA: All genuine Born rule kappa measurements
# Excluding: Rengaraj 2018 (looped paths), Namdar 2023 (nonlinear)
# ============================================================

studies = [
    {
        'id': 1, 'author': 'Sinha et al.', 'year': 2010,
        'platform': 'Photon (laser+PM)',
        'kappa': 0.0073, 'sigma': 0.0018,
        'label': 'Sinha 2010a'
    },
    {
        'id': 2, 'author': 'Sinha et al.', 'year': 2010,
        'platform': 'Photon (atten.+APD)',
        'kappa': 0.0034, 'sigma': 0.0038,
        'label': 'Sinha 2010b'
    },
    {
        'id': 3, 'author': 'Sinha et al.', 'year': 2010,
        'platform': 'Photon (heralded SP)',
        'kappa': 0.0064, 'sigma': 0.0120,
        'label': 'Sinha 2010c'
    },
    {
        'id': 4, 'author': 'Soellner et al.', 'year': 2012,
        'platform': 'Coherent light + homodyne',
        'kappa': 0.002, 'sigma': 0.004,
        'label': 'Soellner 2012'
    },
    {
        'id': 5, 'author': 'Jin et al.', 'year': 2017,
        'platform': 'NV center (M1)',
        'kappa': 0.0017, 'sigma': 0.0045,
        'label': 'Jin 2017a'
    },
    {
        'id': 6, 'author': 'Jin et al.', 'year': 2017,
        'platform': 'NV center (M2)',
        'kappa': -0.0017, 'sigma': 0.0042,
        'label': 'Jin 2017b'
    },
]
# Note: Cotter 2017 gives |kappa| < 0.01 but no signed value with error bar.
# Cannot include in quantitative meta-analysis. Noted as qualitative bound.

# ============================================================
# META-ANALYSIS: Random-effects model
# ============================================================

n = len(studies)
kappas = np.array([s['kappa'] for s in studies])
sigmas = np.array([s['sigma'] for s in studies])
weights_fe = 1.0 / sigmas**2  # Fixed-effect weights

print("=" * 70)
print("META-ANALYSIS: Born Rule Sorkin Parameter kappa")
print("=" * 70)

# --- Fixed-effect model ---
kappa_fe = np.sum(kappas * weights_fe) / np.sum(weights_fe)
se_fe = 1.0 / np.sqrt(np.sum(weights_fe))
z_fe = kappa_fe / se_fe
p_fe = 2 * (1 - norm.cdf(abs(z_fe)))

print(f"\n--- Fixed-Effect Model ---")
print(f"  Combined kappa = {kappa_fe:.6f} +/- {se_fe:.6f}")
print(f"  z = {z_fe:.4f}, p = {p_fe:.6f}")
print(f"  95% CI: [{kappa_fe - 1.96*se_fe:.6f}, {kappa_fe + 1.96*se_fe:.6f}]")

# --- Heterogeneity (Q statistic) ---
Q = np.sum(weights_fe * (kappas - kappa_fe)**2)
df = n - 1
p_Q = 1 - norm.cdf(Q, loc=df, scale=np.sqrt(2*df))  # approximate
I_squared = max(0, (Q - df) / Q * 100) if Q > 0 else 0

print(f"\n--- Heterogeneity ---")
print(f"  Q = {Q:.4f} (df = {df})")
print(f"  I-squared = {I_squared:.1f}%")
if I_squared < 25:
    print(f"  Low heterogeneity — studies are consistent")
elif I_squared < 75:
    print(f"  Moderate heterogeneity")
else:
    print(f"  High heterogeneity — studies may be measuring different things")

# --- Random-effects model (DerSimonian-Laird) ---
C = np.sum(weights_fe) - np.sum(weights_fe**2) / np.sum(weights_fe)
tau_squared = max(0, (Q - df) / C)
weights_re = 1.0 / (sigmas**2 + tau_squared)
kappa_re = np.sum(kappas * weights_re) / np.sum(weights_re)
se_re = 1.0 / np.sqrt(np.sum(weights_re))
z_re = kappa_re / se_re
p_re = 2 * (1 - norm.cdf(abs(z_re)))

print(f"\n--- Random-Effects Model (DerSimonian-Laird) ---")
print(f"  tau-squared = {tau_squared:.8f}")
print(f"  Combined kappa = {kappa_re:.6f} +/- {se_re:.6f}")
print(f"  z = {z_re:.4f}, p = {p_re:.6f}")
print(f"  95% CI: [{kappa_re - 1.96*se_re:.6f}, {kappa_re + 1.96*se_re:.6f}]")

# --- Direction test ---
n_positive = np.sum(kappas > 0)
n_negative = np.sum(kappas < 0)
# Sign test: under H0, P(positive) = 0.5
from scipy.stats import binomtest
try:
    from scipy.stats import binomtest
    p_sign = binomtest(n_positive, n, 0.5).pvalue
except:
    # Manual binomial test
    from scipy.stats import binom
    p_sign = 2 * min(binom.cdf(n_positive, n, 0.5), 1 - binom.cdf(n_positive - 1, n, 0.5))

print(f"\n--- Direction Test ---")
print(f"  Positive kappa: {n_positive}/{n}")
print(f"  Negative kappa: {n_negative}/{n}")
print(f"  Sign test p = {p_sign:.4f}")
if p_sign < 0.05:
    print(f"  SIGNIFICANT directional bias!")
else:
    print(f"  No significant directional bias (p > 0.05)")

# --- Per-study summary ---
print(f"\n--- Individual Studies ---")
print(f"  {'Label':<20} {'kappa':>10} {'sigma':>10} {'z':>8} {'p':>8} {'Platform'}")
print(f"  {'-'*75}")
for s in studies:
    z_i = s['kappa'] / s['sigma']
    p_i = 2 * (1 - norm.cdf(abs(z_i)))
    print(f"  {s['label']:<20} {s['kappa']:>10.4f} {s['sigma']:>10.4f} {z_i:>8.2f} {p_i:>8.4f} {s['platform']}")

# ============================================================
# KEY QUESTION: Is the combined kappa significantly different from 0?
# ============================================================
print(f"\n{'='*70}")
print("VERDICT")
print(f"{'='*70}")
print(f"\n  Combined kappa (RE) = {kappa_re:.6f} +/- {se_re:.6f}")
print(f"  p-value = {p_re:.6f}")

if p_re < 0.001:
    print(f"\n  *** HIGHLY SIGNIFICANT (p < 0.001) ***")
    print(f"  Born rule shows systematic deviation!")
elif p_re < 0.01:
    print(f"\n  ** SIGNIFICANT (p < 0.01) **")
elif p_re < 0.05:
    print(f"\n  * MARGINALLY SIGNIFICANT (p < 0.05) *")
else:
    print(f"\n  NOT SIGNIFICANT (p = {p_re:.4f})")
    print(f"  No evidence for systematic bias in kappa")

significance_note = ""
if kappa_re > 0:
    significance_note = "positive (kappa > 0: third-order interference slightly positive)"
else:
    significance_note = "negative (kappa < 0: third-order interference slightly negative)"
print(f"  Direction: {significance_note}")

# Upper bound on combinatorial pressure effect
print(f"\n  Upper bound (95% CL): |kappa| < {abs(kappa_re) + 1.96*se_re:.6f}")
print(f"  This means: if combinatorial pressure exists, its effect on")
print(f"  third-order interference is less than {(abs(kappa_re) + 1.96*se_re)*100:.3f}%")

# ============================================================
# VISUALIZATION
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 7))

# --- Forest Plot ---
ax = axes[0]
y_pos = np.arange(n)
labels = [s['label'] for s in studies]

# Individual studies
for i in range(n):
    color = 'blue' if kappas[i] >= 0 else 'red'
    ax.errorbar(kappas[i], y_pos[i], xerr=1.96*sigmas[i],
               fmt='o', color=color, markersize=8, capsize=4, linewidth=1.5)

# Combined (diamond)
ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
ax.axvline(x=kappa_re, color='green', linestyle='--', alpha=0.7, linewidth=2,
          label=f'Combined: {kappa_re:.4f} (p={p_re:.3f})')
ax.fill_betweenx([-1, n], kappa_re - 1.96*se_re, kappa_re + 1.96*se_re,
                alpha=0.15, color='green')

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Sorkin parameter kappa', fontsize=12)
ax.set_title('Forest Plot: Born Rule Tests', fontsize=14)
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.2, axis='x')
ax.invert_yaxis()

# --- Funnel Plot ---
ax = axes[1]
se_studies = sigmas
ax.scatter(kappas, se_studies, s=80, c='blue', edgecolors='black', zorder=5)
ax.axvline(x=kappa_re, color='green', linestyle='--', alpha=0.7)
ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

# Funnel boundaries (95% CI)
se_range = np.linspace(0.001, max(se_studies)*1.3, 100)
ax.plot(kappa_re - 1.96*se_range, se_range, 'k--', alpha=0.3)
ax.plot(kappa_re + 1.96*se_range, se_range, 'k--', alpha=0.3)

ax.set_xlabel('kappa', fontsize=12)
ax.set_ylabel('Standard error', fontsize=12)
ax.set_title('Funnel Plot (publication bias check)', fontsize=14)
ax.invert_yaxis()
ax.grid(True, alpha=0.2)

# Asymmetry check
if n >= 3:
    # Egger's test approximation
    precision = 1.0 / sigmas
    std_effects = kappas / sigmas
    slope = np.polyfit(precision, std_effects, 1)
    ax.text(0.05, 0.05, f"Egger intercept ~ {slope[1]:.3f}\n(0 = no bias)",
           transform=ax.transAxes, fontsize=9,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# --- Summary panel ---
ax = axes[2]
ax.axis('off')
summary = f"""META-ANALYSIS SUMMARY
{'='*40}

Studies: {n} (2010-2017)
Platforms: Photons, NV center

Fixed-Effect:
  kappa = {kappa_fe:.6f} +/- {se_fe:.6f}
  z = {z_fe:.2f}, p = {p_fe:.4f}

Random-Effects (DL):
  kappa = {kappa_re:.6f} +/- {se_re:.6f}
  z = {z_re:.2f}, p = {p_re:.4f}
  tau2 = {tau_squared:.8f}

Heterogeneity:
  Q = {Q:.2f}, I2 = {I_squared:.1f}%

Direction:
  Positive: {n_positive}/{n}
  Negative: {n_negative}/{n}
  Sign test p = {p_sign:.4f}

95% CI: [{kappa_re-1.96*se_re:.6f}, {kappa_re+1.96*se_re:.6f}]

VERDICT: {'SIGNIFICANT' if p_re < 0.05 else 'NOT SIGNIFICANT'}
Combined kappa is {significance_note}

Upper bound on combinatorial pressure:
|kappa| < {abs(kappa_re)+1.96*se_re:.6f} (95% CL)
"""
ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
       verticalalignment='top', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('c:/Projects/Multiverse-Evidence/born_rule_meta_analysis.png', dpi=150)
print("\nPlot saved: born_rule_meta_analysis.png")
