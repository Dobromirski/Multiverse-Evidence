"""
Hybrid Branching Model: Instantaneous + Cumulative
====================================================

Lambda(z) = Lambda_0 * [1 + d1 * SFR_ratio(z) + d2 * A(z)]

d1: instantaneous component (Lambda responds to current branching rate)
d2: cumulative component (Lambda accumulates from past branching)

Also test: pure models, inverted models, and alternative proxies.

Author: Zhivko Dobromirski
Date: 2026-04-02
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
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
DH_LCDM = np.array([22.7326, 20.1641, 17.5656, 14.0634, 12.8791, 8.6165])
DM_LCDM = np.array([13.4946, 17.6937, 21.9868, 28.0729, 30.2666, 39.1735])

H0, Om, Or = 67.4, 0.315, 9.1e-5
OL = 1 - Om - Or
rd, c = 147.09, 299792.458

def sfrd(z):
    return 0.015 * (1+z)**2.7 / (1 + ((1+z)/2.9)**5.6) * 0.63

sfr0 = sfrd(0)
total_sfr, _ = quad(sfrd, 0, 20)

# Precompute A(z)
z_cache = np.linspace(0, 25, 2000)
A_cache = np.array([1.0 - quad(sfrd, z, 20)[0]/total_sfr for z in z_cache])
def A_fast(z):
    return np.interp(z, z_cache, A_cache)

# ============================================================
# MODEL DEFINITIONS
# ============================================================

def lambda_ratio(z, params, model):
    """Returns Lambda(z)/Lambda_0"""
    if model == 'lcdm':
        return 1.0
    elif model == 'instantaneous':
        d = params[0]
        return 1.0 + d * (sfrd(z)/sfr0 - 1.0)
    elif model == 'cumulative':
        d = params[0]
        return 1.0 + d * A_fast(z)
    elif model == 'hybrid':
        d1, d2 = params[0], params[1]
        return 1.0 + d1 * (sfrd(z)/sfr0 - 1.0) + d2 * A_fast(z)
    elif model == 'matter_density':
        # Lambda ~ matter density (another proxy)
        d = params[0]
        return 1.0 + d * ((1+z)**3 - 1.0)
    elif model == 'hubble_rate':
        # Running vacuum: Lambda ~ H^2 (Sola et al.)
        # H^2/H0^2 = Om*(1+z)^3 + Or*(1+z)^4 + OL
        d = params[0]
        H2_ratio = Om*(1+z)**3 + Or*(1+z)**4 + OL
        return 1.0 + d * (H2_ratio - 1.0)
    elif model == 'linear_z':
        # Simple: Lambda = Lambda_0 * (1 + d*z)
        d = params[0]
        return 1.0 + d * z
    elif model == 'sfr_cumul_inverted':
        # Cumulative but INVERTED: Lambda was higher, decreases as branching accumulates
        d = params[0]
        return 1.0 + d * (1.0 - A_fast(z))
    return 1.0

def H_z(z, params, model):
    lam = lambda_ratio(z, params, model)
    return H0 * np.sqrt(Om*(1+z)**3 + Or*(1+z)**4 + OL * max(lam, 0.01))

def DH_model(z, params, model):
    return c / (H_z(z, params, model) * rd)

def DM_model(z, params, model):
    r, _ = quad(lambda zp: c / H_z(zp, params, model), 0, z)
    return r / rd

def DV_model(z, params, model):
    dm = DM_model(z, params, model) * rd
    dh = c / H_z(z, params, model)
    return (z * dm**2 * dh)**(1.0/3.0) / rd

def chi2(params, model):
    s = 0
    for i in range(len(z_DH)):
        s += ((DH_meas[i] - DH_model(z_DH[i], params, model)) / DH_err[i])**2
        s += ((DM_meas[i] - DM_model(z_DH[i], params, model)) / DM_err[i])**2
    s += ((DV_meas[0] - DV_model(0.295, params, model)) / DV_err[0])**2
    return s

# ============================================================
# OPTIMIZE ALL MODELS
# ============================================================

models = {
    'lcdm':              {'p0': [], 'bounds': None, 'np': 0},
    'instantaneous':     {'p0': [0.03], 'bounds': [(-0.2, 0.2)], 'np': 1},
    'cumulative':        {'p0': [0.1], 'bounds': [(-0.3, 0.3)], 'np': 1},
    'hybrid':            {'p0': [0.03, 0.05], 'bounds': [(-0.2, 0.2), (-0.3, 0.3)], 'np': 2},
    'matter_density':    {'p0': [0.001], 'bounds': [(-0.01, 0.01)], 'np': 1},
    'hubble_rate':       {'p0': [0.01], 'bounds': [(-0.1, 0.1)], 'np': 1},
    'linear_z':          {'p0': [0.01], 'bounds': [(-0.1, 0.1)], 'np': 1},
    'sfr_cumul_inverted': {'p0': [0.05], 'bounds': [(-0.3, 0.3)], 'np': 1},
}

n_data = 13
results = {}

print("=" * 70)
print("COMPREHENSIVE MODEL COMPARISON vs DESI DR2 BAO")
print("=" * 70)

chi2_lcdm = chi2([], 'lcdm')
print(f"\nLCDM baseline: chi2 = {chi2_lcdm:.2f}")

for name, cfg in models.items():
    if name == 'lcdm':
        results[name] = {'chi2': chi2_lcdm, 'params': [], 'np': 0,
                         'aic': chi2_lcdm, 'bic': chi2_lcdm}
        continue

    # Grid search
    if cfg['np'] == 1:
        lo, hi = cfg['bounds'][0]
        ds = np.linspace(lo, hi, 81)
        best_d, best_c2 = None, 1e9
        for d in ds:
            try:
                c2 = chi2([d], name)
                if c2 < best_c2:
                    best_c2, best_d = c2, d
            except:
                pass
        best_params = [best_d]
    elif cfg['np'] == 2:
        best_c2 = 1e9
        best_params = cfg['p0']
        for d1 in np.linspace(*cfg['bounds'][0], 21):
            for d2 in np.linspace(*cfg['bounds'][1], 21):
                try:
                    c2 = chi2([d1, d2], name)
                    if c2 < best_c2:
                        best_c2, best_params = c2, [d1, d2]
                except:
                    pass

    # Refine
    try:
        res = minimize(lambda p: chi2(list(p), name), best_params,
                      method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-10})
        if res.fun < best_c2:
            best_params, best_c2 = list(res.x), res.fun
    except:
        pass

    k = cfg['np']
    aic = best_c2 + 2*k
    bic = best_c2 + k*np.log(n_data)

    results[name] = {'chi2': best_c2, 'params': best_params, 'np': k,
                     'aic': aic, 'bic': bic}

# Print results sorted by AIC
print(f"\n{'Model':<25} {'Params':>6} {'Chi2':>8} {'AIC':>8} {'BIC':>8} {'dAIC':>7} {'dBIC':>7} | {'Parameters'}")
print("-" * 100)

sorted_models = sorted(results.items(), key=lambda x: x[1]['aic'])
best_aic = sorted_models[0][1]['aic']
best_bic = min(r['bic'] for r in results.values())

for name, r in sorted_models:
    daic = r['aic'] - best_aic
    dbic = r['bic'] - best_bic
    pstr = ', '.join([f'{p:.5f}' for p in r['params']]) if r['params'] else '-'
    print(f"{name:<25} {r['np']:>6} {r['chi2']:>8.2f} {r['aic']:>8.2f} {r['bic']:>8.2f} {daic:>+7.2f} {dbic:>+7.2f} | {pstr}")

# ============================================================
# VISUALIZATION
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
colors = {'lcdm': 'black', 'instantaneous': 'red', 'cumulative': 'blue',
          'hybrid': 'purple', 'matter_density': 'green', 'hubble_rate': 'orange',
          'linear_z': 'cyan', 'sfr_cumul_inverted': 'brown'}

# Plot 1: Lambda(z) for top models
ax = axes[0, 0]
zp = np.linspace(0, 3, 200)
for name, r in sorted_models[:5]:
    lam = [lambda_ratio(z, r['params'], name) for z in zp]
    ax.plot(zp, lam, '-', color=colors.get(name, 'gray'), linewidth=2,
            label=f"{name} (AIC={r['aic']:.1f})")
ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel('Lambda(z)/Lambda_0', fontsize=12)
ax.set_title('Vacuum Energy Evolution (top 5)', fontsize=14)
ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

# Plot 2: DH/rd
ax = axes[0, 1]
ax.errorbar(z_DH, DH_meas, yerr=DH_err, fmt='ko', ms=8, capsize=5, label='DESI', zorder=5)
for name, r in sorted_models[:4]:
    dh = [DH_model(z, r['params'], name) for z in z_DH]
    ax.plot(z_DH, dh, 'o--', color=colors.get(name, 'gray'), ms=6, label=name, lw=1.5)
ax.set_xlabel('z', fontsize=12); ax.set_ylabel('DH/rd', fontsize=12)
ax.set_title('DH/rd', fontsize=14)
ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

# Plot 3: DM/rd
ax = axes[0, 2]
ax.errorbar(z_DH, DM_meas, yerr=DM_err, fmt='ko', ms=8, capsize=5, label='DESI', zorder=5)
for name, r in sorted_models[:4]:
    dm = [DM_model(z, r['params'], name) for z in z_DH]
    ax.plot(z_DH, dm, 'o--', color=colors.get(name, 'gray'), ms=6, label=name, lw=1.5)
ax.set_xlabel('z', fontsize=12); ax.set_ylabel('DM/rd', fontsize=12)
ax.set_title('DM/rd', fontsize=14)
ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

# Plot 4: DH residuals for top 4
ax = axes[1, 0]
x = np.arange(len(z_DH))
w = 0.2
for j, (name, r) in enumerate(sorted_models[:4]):
    dh = np.array([DH_model(z, r['params'], name) for z in z_DH])
    res = (DH_meas - dh) / DH_err
    ax.bar(x + j*w - 1.5*w, res, w, color=colors.get(name, 'gray'), alpha=0.7, label=name)
ax.set_xticks(x); ax.set_xticklabels([f'{z:.2f}' for z in z_DH], fontsize=8)
ax.set_ylabel('DH residual (sigma)', fontsize=12)
ax.set_title('DH Residuals per Bin', fontsize=14)
ax.axhline(y=0, color='black', alpha=0.3); ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

# Plot 5: AIC comparison
ax = axes[1, 1]
names_sorted = [n for n, _ in sorted_models]
aics = [r['aic'] for _, r in sorted_models]
bar_colors = [colors.get(n, 'gray') for n in names_sorted]
ax.barh(range(len(names_sorted)), aics, color=bar_colors, alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(names_sorted)))
ax.set_yticklabels(names_sorted, fontsize=9)
ax.set_xlabel('AIC (lower = better)', fontsize=12)
ax.set_title('Model Ranking by AIC', fontsize=14)
ax.axvline(x=chi2_lcdm, color='black', linestyle=':', label='LCDM')
ax.grid(True, alpha=0.2, axis='x')

# Plot 6: Summary text
ax = axes[1, 2]
ax.axis('off')
txt = "RESULTS SUMMARY\n" + "="*45 + "\n\n"
for name, r in sorted_models:
    daic = r['aic'] - best_aic
    pstr = ', '.join([f'{p:.4f}' for p in r['params']]) if r['params'] else 'none'
    txt += f"{name}:\n  chi2={r['chi2']:.2f}, AIC={r['aic']:.2f} (dAIC={daic:+.1f})\n  params=[{pstr}]\n\n"
ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=8,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('c:/Projects/Multiverse-Evidence/hybrid_model_results.png', dpi=150)
print("\nPlot saved: hybrid_model_results.png")

# Winner
w = sorted_models[0]
print(f"\n>>> WINNER: {w[0]} (AIC={w[1]['aic']:.2f}, params={w[1]['params']})")
print(f">>> vs LCDM: Delta AIC = {chi2_lcdm - w[1]['aic']:+.2f}")
