"""
Expansion-Branching Model
==========================

Three components of branching:
1. SFR (matter-driven, peaks at z~2, declines)
2. Expansion (space itself, continuous, accelerating)
3. Combined

Test: which combination best fits DESI DR2 BAO data?

Lambda(z) = Lambda_0 * [1 + d1*(1-A_sfr(z)) + d2*(H2(z)/H2(0) - 1)]

Author: Zhivko Dobromirski
Date: 2026-04-02
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize, minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# DESI DR2 data
z_DH = np.array([0.510, 0.706, 0.934, 1.321, 1.484, 2.330])
DH_meas = np.array([21.8629, 19.4553, 17.6415, 14.1760, 12.8170, 8.6315])
DH_err = np.array([0.4289, 0.3339, 0.2010, 0.2246, 0.5180, 0.1011])
DM_meas = np.array([13.5876, 17.3507, 21.5756, 27.6009, 30.5119, 38.9890])
DM_err = np.array([0.1684, 0.1799, 0.1618, 0.3246, 0.7636, 0.5317])
DV_meas = np.array([7.9417])
DV_err = np.array([0.0761])

H0, Om, Or = 67.4, 0.315, 9.1e-5
OL = 1 - Om - Or
rd, cc = 147.09, 299792.458

def sfrd(z):
    return 0.015 * (1+z)**2.7 / (1 + ((1+z)/2.9)**5.6) * 0.63

total_sfr, _ = quad(sfrd, 0, 20)
z_cache = np.linspace(0, 25, 2000)
A_cache = np.array([1.0 - quad(sfrd, z, 20)[0]/total_sfr for z in z_cache])
def A_fast(z):
    return float(np.interp(z, z_cache, A_cache))

# H_LCDM^2 / H0^2
def H2_ratio_lcdm(z):
    return Om*(1+z)**3 + Or*(1+z)**4 + OL

# ============================================================
# MODELS
# ============================================================

def lambda_model(z, params, model):
    if model == 'lcdm':
        return 1.0

    elif model == 'sfr_inverted':
        # Winner from previous test
        d = params[0]
        return 1.0 + d * (1.0 - A_fast(z))

    elif model == 'expansion_only':
        # Lambda ~ H^2 (Running Vacuum a la Sola)
        d = params[0]
        return 1.0 + d * (H2_ratio_lcdm(z) - 1.0)

    elif model == 'expansion_log':
        # Lambda ~ ln(H^2) - softer dependence
        d = params[0]
        return 1.0 + d * np.log(H2_ratio_lcdm(z))

    elif model == 'sfr_plus_expansion':
        # Combined: SFR decay + expansion feedback
        d1, d2 = params[0], params[1]
        sfr_comp = d1 * (1.0 - A_fast(z))
        exp_comp = d2 * (H2_ratio_lcdm(z) - 1.0)
        return 1.0 + sfr_comp + exp_comp

    elif model == 'sfr_times_expansion':
        # Multiplicative: branching rate = SFR(z) * Volume(z)
        # Volume grows as a^3, so Volume(z)/Volume(0) = (1+z)^-3... no, comoving volume is constant
        # Physical volume of observable universe grows, but comoving is fixed
        # Rate of expansion = H(z). More expansion = more branching per unit time
        # Total branching rate ~ SFR(z) * H(z) (matter branching amplified by expansion)
        d = params[0]
        sfr_ratio = sfrd(z) / sfrd(0)
        H_ratio = np.sqrt(H2_ratio_lcdm(z))
        combined_rate = sfr_ratio * H_ratio
        combined_rate_0 = 1.0  # at z=0
        return 1.0 + d * (combined_rate - combined_rate_0)

    elif model == 'feedback_positive':
        # Self-reinforcing: Lambda grows because expansion creates branching
        # which adds to Lambda. Parametrize as exponential growth.
        # Lambda(z) = Lambda_0 * exp(d * integral_z^0 H(z')/H0 dz')
        # Simplified: Lambda = Lambda_0 * (1 + d * (scale_factor_integral))
        d = params[0]
        # integral of (1+z)^-1 from z to 0 = ln(1+z)
        # This is ~ "amount of expansion from z to now"
        return 1.0 + d * np.log(1 + z)

    elif model == 'feedback_quadratic':
        # Feedback: Lambda(z)/Lambda_0 = 1 + d * [ln(1+z)]^2
        d = params[0]
        return 1.0 + d * (np.log(1 + z))**2

    elif model == 'volume_branching':
        # Total branching ~ integral of (event_rate * comoving_volume_element)
        # As universe expands, more "space" is branching
        # Rate per unit comoving volume decreases (dilution)
        # But total volume increases
        # Net effect depends on equation of state
        # Simple proxy: total branching ~ (1+z)^(-n) for n < 3
        d, n = params[0], params[1]
        return 1.0 + d * ((1+z)**(-n) - 1.0)

    return 1.0


# Self-consistent H(z) for models that modify Lambda
def H_z(z, params, model):
    lam = lambda_model(z, params, model)
    val = Om*(1+z)**3 + Or*(1+z)**4 + OL * max(lam, 0.01)
    return H0 * np.sqrt(max(val, 1e-10))

def DH_m(z, params, model):
    return cc / (H_z(z, params, model) * rd)

def DM_m(z, params, model):
    r, _ = quad(lambda zp: cc / H_z(zp, params, model), 0, z, limit=100)
    return r / rd

def DV_m(z, params, model):
    dm = DM_m(z, params, model) * rd
    dh = cc / H_z(z, params, model)
    return (z * dm**2 * dh)**(1.0/3.0) / rd

def chi2_func(params, model):
    try:
        s = 0
        for i in range(len(z_DH)):
            s += ((DH_meas[i] - DH_m(z_DH[i], params, model)) / DH_err[i])**2
            s += ((DM_meas[i] - DM_m(z_DH[i], params, model)) / DM_err[i])**2
        s += ((DV_meas[0] - DV_m(0.295, params, model)) / DV_err[0])**2
        return s
    except:
        return 1e6

# ============================================================
# OPTIMIZE ALL
# ============================================================

configs = {
    'lcdm':               {'np': 0, 'bounds': []},
    'sfr_inverted':       {'np': 1, 'bounds': [(-0.2, 0.2)]},
    'expansion_only':     {'np': 1, 'bounds': [(-0.1, 0.1)]},
    'expansion_log':      {'np': 1, 'bounds': [(-0.2, 0.2)]},
    'feedback_positive':  {'np': 1, 'bounds': [(-0.2, 0.2)]},
    'feedback_quadratic': {'np': 1, 'bounds': [(-0.2, 0.2)]},
    'sfr_plus_expansion': {'np': 2, 'bounds': [(-0.2, 0.2), (-0.05, 0.05)]},
    'sfr_times_expansion': {'np': 1, 'bounds': [(-0.2, 0.2)]},
    'volume_branching':   {'np': 2, 'bounds': [(-0.5, 0.5), (0.1, 3.0)]},
}

n_data = 13
results = {}

print("=" * 70)
print("EXPANSION-BRANCHING MODEL COMPARISON")
print("=" * 70)

chi2_lcdm = chi2_func([], 'lcdm')
results['lcdm'] = {'chi2': chi2_lcdm, 'params': [], 'np': 0,
                   'aic': chi2_lcdm, 'bic': chi2_lcdm}

for name, cfg in configs.items():
    if name == 'lcdm':
        continue

    k = cfg['np']
    bounds = cfg['bounds']

    # Grid search
    if k == 1:
        lo, hi = bounds[0]
        ds = np.linspace(lo, hi, 81)
        best_c2, best_p = 1e9, [0]
        for d in ds:
            c2 = chi2_func([d], name)
            if c2 < best_c2:
                best_c2, best_p = c2, [d]
    elif k == 2:
        best_c2, best_p = 1e9, [0, 0]
        for d1 in np.linspace(*bounds[0], 21):
            for d2 in np.linspace(*bounds[1], 21):
                c2 = chi2_func([d1, d2], name)
                if c2 < best_c2:
                    best_c2, best_p = c2, [d1, d2]

    # Refine
    try:
        res = minimize(lambda p: chi2_func(list(p), name), best_p,
                      method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-10})
        if res.fun < best_c2:
            best_p, best_c2 = list(res.x), res.fun
    except:
        pass

    aic = best_c2 + 2*k
    bic = best_c2 + k*np.log(n_data)
    results[name] = {'chi2': best_c2, 'params': best_p, 'np': k, 'aic': aic, 'bic': bic}

# Sort and print
sorted_r = sorted(results.items(), key=lambda x: x[1]['aic'])
best_aic = sorted_r[0][1]['aic']

print(f"\n{'Model':<25} {'k':>2} {'chi2':>8} {'AIC':>8} {'dAIC':>7} | {'Parameters'}")
print("-" * 85)
for name, r in sorted_r:
    daic = r['aic'] - best_aic
    pstr = ', '.join([f'{p:.5f}' for p in r['params']]) if r['params'] else '-'
    marker = ' <<<' if name == sorted_r[0][0] else ''
    print(f"{name:<25} {r['np']:>2} {r['chi2']:>8.2f} {r['aic']:>8.2f} {daic:>+7.2f} | {pstr}{marker}")

# ============================================================
# VISUALIZE
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
colors = {
    'lcdm': 'black', 'sfr_inverted': 'red', 'expansion_only': 'blue',
    'expansion_log': 'cyan', 'feedback_positive': 'green',
    'feedback_quadratic': 'lime', 'sfr_plus_expansion': 'purple',
    'sfr_times_expansion': 'orange', 'volume_branching': 'brown'
}

zp = np.linspace(0.01, 3, 200)

# Plot 1: Lambda(z) for top models
ax = axes[0, 0]
for name, r in sorted_r[:6]:
    lam = [lambda_model(z, r['params'], name) for z in zp]
    ax.plot(zp, lam, '-', color=colors.get(name, 'gray'), linewidth=2,
            label=f"{name} ({r['aic']:.1f})")
ax.axhline(y=1.0, color='black', linestyle=':', alpha=0.3)
ax.set_xlabel('Redshift z', fontsize=12); ax.set_ylabel('Lambda(z)/Lambda_0', fontsize=12)
ax.set_title('Vacuum Energy Evolution', fontsize=14)
ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

# Plot 2: DH/rd
ax = axes[0, 1]
ax.errorbar(z_DH, DH_meas, yerr=DH_err, fmt='ko', ms=8, capsize=5, label='DESI', zorder=5)
for name, r in sorted_r[:5]:
    dh = [DH_m(z, r['params'], name) for z in z_DH]
    ax.plot(z_DH, dh, 'o--', color=colors.get(name, 'gray'), ms=6, label=name, lw=1.5)
ax.set_xlabel('z', fontsize=12); ax.set_ylabel('DH/rd', fontsize=12)
ax.set_title('DH/rd', fontsize=14); ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

# Plot 3: DM/rd
ax = axes[0, 2]
ax.errorbar(z_DH, DM_meas, yerr=DM_err, fmt='ko', ms=8, capsize=5, label='DESI', zorder=5)
for name, r in sorted_r[:5]:
    dm = [DM_m(z, r['params'], name) for z in z_DH]
    ax.plot(z_DH, dm, 'o--', color=colors.get(name, 'gray'), ms=6, label=name, lw=1.5)
ax.set_xlabel('z', fontsize=12); ax.set_ylabel('DM/rd', fontsize=12)
ax.set_title('DM/rd', fontsize=14); ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

# Plot 4: DH residuals
ax = axes[1, 0]
x = np.arange(len(z_DH))
w = 0.15
for j, (name, r) in enumerate(sorted_r[:5]):
    dh = np.array([DH_m(z, r['params'], name) for z in z_DH])
    res = (DH_meas - dh) / DH_err
    ax.bar(x + j*w - 2*w, res, w, color=colors.get(name, 'gray'), alpha=0.7, label=name)
ax.set_xticks(x); ax.set_xticklabels([f'{z:.2f}' for z in z_DH], fontsize=8)
ax.set_ylabel('DH residual (sigma)', fontsize=12)
ax.set_title('DH Residuals', fontsize=14)
ax.axhline(y=0, color='black', alpha=0.3); ax.legend(fontsize=6); ax.grid(True, alpha=0.2)

# Plot 5: AIC ranking
ax = axes[1, 1]
names = [n for n, _ in sorted_r]
aics = [r['aic'] for _, r in sorted_r]
bar_colors = [colors.get(n, 'gray') for n in names]
bars = ax.barh(range(len(names)), aics, color=bar_colors, alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel('AIC (lower = better)', fontsize=12)
ax.set_title('Model Ranking', fontsize=14)
ax.axvline(x=chi2_lcdm, color='black', linestyle=':', label='LCDM')
ax.grid(True, alpha=0.2, axis='x')
# Add dAIC labels
for i, (name, r) in enumerate(sorted_r):
    daic = r['aic'] - best_aic
    ax.text(r['aic'] + 0.3, i, f'dAIC={daic:+.1f}', va='center', fontsize=7)

# Plot 6: Feedback loop diagram as text
ax = axes[1, 2]
ax.axis('off')
txt = """PHYSICAL INTERPRETATION
""" + "="*45 + """

FEEDBACK LOOP HYPOTHESIS:

  Vacuum Energy (Lambda)
        |
        v
  Expansion of Space (H)
        |
        v
  Branching (quantum events in
  expanding spacetime)
        |
        v
  Energy released/consumed
        |
        v
  Back to Lambda (feedback)

POSITIVE FEEDBACK (accelerating):
  More Lambda -> more expansion ->
  more branching -> more Lambda -> ...

NEGATIVE FEEDBACK (stabilizing):
  More Lambda -> more expansion ->
  more branching -> less Lambda -> ...

DATA VERDICT:
"""
w = sorted_r[0]
txt += f"  Best model: {w[0]}\n"
txt += f"  AIC = {w[1]['aic']:.1f} (dAIC vs LCDM = {chi2_lcdm - w[1]['aic']:+.1f})\n"
txt += f"  params = {[f'{p:.4f}' for p in w[1]['params']]}\n"

ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('c:/Projects/Multiverse-Evidence/expansion_branching_results.png', dpi=150)
print("\nPlot saved: expansion_branching_results.png")

# Final
print(f"\n>>> WINNER: {sorted_r[0][0]}")
print(f">>> AIC = {sorted_r[0][1]['aic']:.2f}")
print(f">>> Delta AIC vs LCDM = {chi2_lcdm - sorted_r[0][1]['aic']:+.2f}")
