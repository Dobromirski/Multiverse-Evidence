"""
Weighted Branching Decay Model
================================

Посока 11: Различните типове квантови събития имат различна
"branching cost". Λ(t) = Λ₀ × [1 + Σᵢ δᵢ × fᵢ(a)]

Всеки тип събитие има собствена космична история fᵢ(a).
Сравняваме с DESI DR2.

Автор: Живко Добромирски
Дата: 2026-04-02
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# DESI DR2 reference
# ============================================================
DESI_w0 = -0.752
DESI_w0_err = 0.067
DESI_wa = -0.60
DESI_wa_err = 0.29

# ============================================================
# COSMIC HISTORIES f_i(a) for each event type
# ============================================================

def f_vacuum(a):
    """Вакуумни флуктуации: константни per unit comoving volume.
    В expanding universe, comoving volume = const, so f = const = 1."""
    return np.ones_like(np.atleast_1d(a), dtype=float)

def f_thermal(a):
    """Термични флуктуации: ~ baryon density × temperature.
    T ~ 1/a (за radiation) или ~const (за decoupled matter).
    After decoupling (a > 0.001): baryon density ~ a^-3, T_CMB ~ 1/a.
    Rate ~ n × T ~ a^-3 × a^-1 = a^-4 (early), transitions to a^-3 (late)."""
    a = np.atleast_1d(a).astype(float)
    a_dec = 1e-3  # decoupling
    early = a**(-4) * np.exp(-a/a_dec)
    late = a**(-3) * (1 - np.exp(-a/a_dec))
    result = early + late
    # Normalize to f(a=1) = 1
    norm = 1.0  # a^-3 at a=1 = 1
    return result / result[-1] if len(result) > 1 else result

def f_sfr(a):
    """Ядрен синтез в звезди: следва cosmic SFR (Madau & Dickinson 2014).
    SFR(z) ~ (1+z)^2.7 / (1 + ((1+z)/2.9)^5.6)"""
    a = np.atleast_1d(a).astype(float)
    z = np.clip(1.0/a - 1.0, 0, 100)
    sfr = ((1+z)**2.7) / (1 + ((1+z)/2.9)**5.6)
    sfr_now = 1.0  # SFR(z=0) ~ 1 (normalized)
    return sfr / sfr_now

def f_chemistry(a):
    """Химични реакции: растат с metalicity.
    Metalicity ~ integral(SFR), roughly ~ (1 - exp(-t/tau)).
    Approximation: metals grow logistically, reaching ~solar at a~0.6."""
    a = np.atleast_1d(a).astype(float)
    # Metallicity proxy: cumulative star formation
    z = np.clip(1.0/a - 1.0, 0, 100)
    # Simple sigmoid: zero at high z, ~1 at z=0
    metal = 1.0 / (1.0 + np.exp(-8*(a - 0.4)))
    return metal

def f_strong(a):
    """Силно взаимодействие: ~ baryon density ~ a^-3.
    Constant per baryon, density drops with expansion."""
    a = np.atleast_1d(a).astype(float)
    return a**(-3)

def f_agn(a):
    """AGN/черни дупки: следва cosmic AGN luminosity density.
    Peak at z~2-3 (a~0.25-0.33), sharper than SFR."""
    a = np.atleast_1d(a).astype(float)
    z = np.clip(1.0/a - 1.0, 0, 100)
    # AGN luminosity density peaks at z~2.5
    agn = np.exp(-((z - 2.5)**2) / (2 * 1.0**2))
    agn_now = np.exp(-((0 - 2.5)**2) / (2 * 1.0**2))
    return agn / agn_now


# All event types
EVENT_TYPES = {
    'vacuum':    {'func': f_vacuum,    'energy': 0,     'label': 'Vacuum fluctuations'},
    'thermal':   {'func': f_thermal,   'energy': 1e-3,  'label': 'Thermal (meV)'},
    'chemistry': {'func': f_chemistry, 'energy': 1,     'label': 'Chemistry (eV)'},
    'sfr':       {'func': f_sfr,       'energy': 1e6,   'label': 'Nuclear fusion (MeV)'},
    'strong':    {'func': f_strong,    'energy': 1e9,   'label': 'Strong force (GeV)'},
    'agn':       {'func': f_agn,       'energy': 1e12,  'label': 'AGN/BH (TeV)'},
}


# ============================================================
# MODEL: Weighted Λ(a)
# ============================================================

def lambda_weighted(a, weights, event_keys):
    """Λ(a)/Λ₀ = 1 + Σᵢ wᵢ × (fᵢ(a) - fᵢ(1))"""
    result = np.ones_like(np.atleast_1d(a), dtype=float)
    for w, key in zip(weights, event_keys):
        f = EVENT_TYPES[key]['func']
        fa = f(a)
        f1 = f(np.array([1.0]))[0] if hasattr(f(np.array([1.0])), '__len__') else f(np.array([1.0]))
        result = result + w * (fa - f1)
    return result


def compute_w_from_lambda(a_array, lambda_func, *args):
    """w(a) = -1 - (1/3) × d(ln Λ)/d(ln a)"""
    w_values = []
    for a in a_array:
        da = 0.001 * a
        L_m = lambda_func(a - da, *args)
        L_p = lambda_func(a + da, *args)
        L_c = lambda_func(a, *args)
        if hasattr(L_m, '__len__'):
            L_m, L_p, L_c = L_m[0], L_p[0], L_c[0]
        if L_c <= 0:
            w_values.append(-1.0)
            continue
        dL_da = (L_p - L_m) / (2 * da)
        dlnL_dlna = (a / L_c) * dL_da
        w_values.append(-1.0 - (1.0/3.0) * dlnL_dlna)
    return np.array(w_values)


def fit_w0_wa(w_array, a_array):
    A = np.column_stack([np.ones_like(a_array), 1.0 - a_array])
    return np.linalg.lstsq(A, w_array, rcond=None)[0]


def desi_distance(w0, wa):
    return np.sqrt(((w0 - DESI_w0)/DESI_w0_err)**2 + ((wa - DESI_wa)/DESI_wa_err)**2)


# ============================================================
# TEST MODELS
# ============================================================

def main():
    print("=" * 70)
    print("WEIGHTED BRANCHING DECAY MODEL")
    print("Posoka 11: Energy taxonomy of branching events")
    print("=" * 70)

    a_fit = np.linspace(0.2, 1.0, 80)
    a_plot = np.linspace(0.1, 1.0, 200)

    # ---- Define model configurations ----
    configs = {
        'democratic_sfr': {
            'desc': 'SFR only (baseline from v2)',
            'keys': ['sfr'],
            'bounds': [(0.001, 0.5)]
        },
        'democratic_all': {
            'desc': 'All types, equal weight (1 param scales all)',
            'keys': ['vacuum', 'thermal', 'sfr', 'chemistry', 'strong', 'agn'],
            'bounds': [(0.001, 0.3)]  # single delta, applied to all
        },
        'sfr_plus_chemistry': {
            'desc': 'Nuclear + Chemistry (2 params)',
            'keys': ['sfr', 'chemistry'],
            'bounds': [(0.001, 0.5), (0.001, 0.5)]
        },
        'sfr_plus_agn': {
            'desc': 'Nuclear + AGN (2 params)',
            'keys': ['sfr', 'agn'],
            'bounds': [(0.001, 0.5), (0.001, 0.5)]
        },
        'energy_weighted': {
            'desc': 'Energy-proportional: w_i ~ E_i (1 param)',
            'keys': ['thermal', 'chemistry', 'sfr', 'strong', 'agn'],
            'bounds': [(1e-12, 1e-6)]  # single scale factor
        },
        'full_3param': {
            'desc': 'Vacuum + SFR + AGN (3 params)',
            'keys': ['vacuum', 'sfr', 'agn'],
            'bounds': [(0.001, 0.3), (0.001, 0.5), (0.001, 0.5)]
        },
    }

    results = {}

    for config_name, config in configs.items():
        keys = config['keys']
        bounds = config['bounds']
        n_params = len(bounds)

        print(f"\n--- {config_name}: {config['desc']} ({n_params} params) ---")

        def objective(params):
            try:
                if config_name == 'democratic_all':
                    # Single param applied to all
                    weights = [params[0]] * len(keys)
                elif config_name == 'energy_weighted':
                    # Single scale, weighted by energy
                    scale = params[0]
                    energies = [EVENT_TYPES[k]['energy'] for k in keys]
                    weights = [scale * e for e in energies]
                else:
                    weights = list(params)

                w_vals = compute_w_from_lambda(a_fit, lambda_weighted, weights, keys)
                if np.any(np.isnan(w_vals)) or np.any(np.abs(w_vals) > 50):
                    return 1e6
                w0, wa = fit_w0_wa(w_vals, a_fit)
                return desi_distance(w0, wa)
            except:
                return 1e6

        # Grid search
        from itertools import product
        n_grid = max(8, int(30 / n_params))
        grids = [np.linspace(lo, hi, n_grid) for lo, hi in bounds]

        best_dist = 1e6
        best_params = None
        for combo in product(*grids):
            d = objective(list(combo))
            if d < best_dist:
                best_dist = d
                best_params = list(combo)

        # Local refinement
        if best_params:
            try:
                res = minimize(objective, best_params, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
                if res.fun < best_dist:
                    best_params = list(res.x)
                    best_dist = res.fun
            except:
                pass

        # Extract w0, wa
        if config_name == 'democratic_all':
            best_weights = [best_params[0]] * len(keys)
        elif config_name == 'energy_weighted':
            energies = [EVENT_TYPES[k]['energy'] for k in keys]
            best_weights = [best_params[0] * e for e in energies]
        else:
            best_weights = best_params

        w_vals = compute_w_from_lambda(a_fit, lambda_weighted, best_weights, keys)
        w0, wa = fit_w0_wa(w_vals, a_fit)

        results[config_name] = {
            'params': best_params, 'weights': best_weights, 'keys': keys,
            'w0': w0, 'wa': wa, 'dist': best_dist, 'n_params': n_params,
            'desc': config['desc']
        }

        print(f"  params: {[f'{p:.6e}' for p in best_params]}")
        print(f"  weights: {[f'{w:.6e}' for w in best_weights]}")
        print(f"  w0 = {w0:.4f}, wa = {wa:.4f}")
        print(f"  Distance from DESI: {best_dist:.4f} sigma")

    # ============================================================
    # VISUALIZATION
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    colors = {'democratic_sfr': 'red', 'democratic_all': 'blue',
              'sfr_plus_chemistry': 'green', 'sfr_plus_agn': 'orange',
              'energy_weighted': 'purple', 'full_3param': 'brown'}

    # Plot 1: w0-wa plane
    ax1 = axes[0, 0]
    theta = np.linspace(0, 2*np.pi, 100)
    for sigma, alpha in [(1, 0.3), (2, 0.15)]:
        ell_w0 = DESI_w0 + sigma * DESI_w0_err * np.cos(theta)
        ell_wa = DESI_wa + sigma * DESI_wa_err * np.sin(theta)
        ax1.fill(ell_w0, ell_wa, alpha=alpha, color='blue')
    ax1.plot(-1.0, 0.0, 'k+', markersize=15, markeredgewidth=2, label='LCDM')

    for name, res in results.items():
        ax1.plot(res['w0'], res['wa'], 'o', color=colors.get(name, 'gray'),
                markersize=10, label=f"{name} ({res['dist']:.2f}s, {res['n_params']}p)")

    ax1.set_xlabel('w0', fontsize=12)
    ax1.set_ylabel('wa', fontsize=12)
    ax1.set_title('Weighted Models vs DESI DR2', fontsize=13)
    ax1.legend(fontsize=7, loc='upper left')
    ax1.set_xlim(-1.2, -0.4)
    ax1.set_ylim(-2.5, 1.0)
    ax1.grid(True, alpha=0.2)

    # Plot 2: Cosmic histories f_i(a)
    ax2 = axes[0, 1]
    hist_colors = {'vacuum': 'gray', 'thermal': 'cyan', 'chemistry': 'green',
                   'sfr': 'red', 'strong': 'blue', 'agn': 'orange'}
    for key, info in EVENT_TYPES.items():
        fa = info['func'](a_plot)
        # Normalize for display
        fa_norm = fa / np.max(fa) if np.max(fa) > 0 else fa
        ax2.plot(a_plot, fa_norm, '-', color=hist_colors.get(key, 'gray'),
                linewidth=2, label=info['label'])

    ax2.set_xlabel('Scale factor a (a=1 today)', fontsize=12)
    ax2.set_ylabel('f_i(a) / max (normalized)', fontsize=12)
    ax2.set_title('Cosmic Histories of Event Types', fontsize=13)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    # Plot 3: Lambda(a) for best models
    ax3 = axes[1, 0]
    sorted_results = sorted(results.items(), key=lambda x: x[1]['dist'])
    for name, res in sorted_results[:4]:
        L_vals = lambda_weighted(a_plot, res['weights'], res['keys'])
        ax3.plot(a_plot, L_vals, '-', color=colors.get(name, 'gray'),
                linewidth=2, label=f"{name} ({res['dist']:.2f}s)")
    ax3.axhline(y=1.0, color='black', linestyle='--', label='LCDM')
    ax3.set_xlabel('Scale factor a', fontsize=12)
    ax3.set_ylabel('L(a) / L(today)', fontsize=12)
    ax3.set_title('Vacuum Energy Evolution (top 4 models)', fontsize=13)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.2)

    # Plot 4: w(a) for best models
    ax4 = axes[1, 1]
    a_w = np.linspace(0.2, 1.0, 100)
    w_desi = DESI_w0 + DESI_wa * (1.0 - a_w)
    ax4.fill_between(a_w,
                     DESI_w0 - DESI_w0_err + (DESI_wa - DESI_wa_err)*(1-a_w),
                     DESI_w0 + DESI_w0_err + (DESI_wa + DESI_wa_err)*(1-a_w),
                     alpha=0.15, color='blue', label='DESI 1s')
    ax4.plot(a_w, w_desi, 'b--', linewidth=1, alpha=0.5)

    for name, res in sorted_results[:4]:
        w_vals = compute_w_from_lambda(a_w, lambda_weighted, res['weights'], res['keys'])
        ax4.plot(a_w, w_vals, '-', color=colors.get(name, 'gray'),
                linewidth=2, label=name)
    ax4.axhline(y=-1.0, color='black', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Scale factor a', fontsize=12)
    ax4.set_ylabel('w(a)', fontsize=12)
    ax4.set_title('Equation of State w(a)', fontsize=13)
    ax4.legend(fontsize=7)
    ax4.set_ylim(-2.5, 0.5)
    ax4.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('c:/Projects/Multiverse-Evidence/weighted_branching_results.png', dpi=150)
    print("\nPlot saved: weighted_branching_results.png")

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY: All models ranked by DESI distance")
    print("=" * 70)
    print(f"\n{'Model':<30} {'Params':>6} {'w0':>8} {'wa':>8} {'Dist':>8}")
    print("-" * 65)
    print(f"{'LCDM':<30} {'0':>6} {-1.0:>8.4f} {0.0:>8.4f} {desi_distance(-1.0, 0.0):>8.3f}s")

    for name, res in sorted_results:
        print(f"{name:<30} {res['n_params']:>6} {res['w0']:>8.4f} {res['wa']:>8.4f} {res['dist']:>8.3f}s")

    print(f"\n{'DESI DR2 (observed)':<30} {'-':>6} {DESI_w0:>8.4f} {DESI_wa:>8.4f} {'0.000':>8}s")

    # Best per param count
    print("\n--- Best per parameter count ---")
    for np_count in [1, 2, 3]:
        matching = [(n, r) for n, r in sorted_results if r['n_params'] == np_count]
        if matching:
            best = matching[0]
            print(f"  {np_count} param: {best[0]} at {best[1]['dist']:.3f}s")


if __name__ == '__main__':
    main()
