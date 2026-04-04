"""
Multiverse Branching-Decay Model v2
====================================

Проблем с v1: ранната вселена (radiation era) доминира интеграла —
всички late-time модели дават идентичен резултат.

Решение: Моделираме ДИРЕКТНО Λ(a) без да интегрираме от Big Bang.
Физическа мотивация: ранното разклоняване вече е отразено в днешната Λ₀.
DESI мери ПРОМЯНАТА в Λ в близката вселена (z < 3).
Затова моделираме само ефективния branching rate в обозримата история.

Автор: Живко Добромирски
Дата: 2026-04-01
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# DESI DR2 (2025) и ΛCDM референции
# ============================================================
DESI_w0 = -0.752
DESI_w0_err = 0.067
DESI_wa = -0.60
DESI_wa_err = 0.29
LCDM_w0 = -1.0
LCDM_wa = 0.0

# ============================================================
# МОДЕЛИ за Λ(a)/Λ₀
# ============================================================

def lambda_model(a, params, model_type):
    """
    Λ(a)/Λ₀ за различни модели на branching decay.

    Всички модели: Λ(a)/Λ₀ > 1 за a < 1 (повече вакуумна енергия в миналото),
    Λ(a=1)/Λ₀ = 1 (днес).
    """
    if model_type == 'linear':
        # Λ(a) = Λ₀ × [1 + δ(1-a)]
        # Най-прост: линейно намаляване
        delta = params[0]
        return 1.0 + delta * (1.0 - a)

    elif model_type == 'sfr_peak':
        # Branching rate следва cosmic star formation rate
        # SFR пик при z~2 (a~0.33), после спада
        # Λ се изразходва най-бързо при SFR пика
        delta, a_peak, width = params[0], params[1], params[2]
        # Кумулативен ефект: интеграл на SFR-подобна крива от a до 1
        # Приближение: erfc-подобна функция
        decay_profile = np.exp(-((a - a_peak)**2) / (2 * width**2))
        # Нормализираме така че при a=1: Λ/Λ₀ = 1
        norm_at_1 = np.exp(-((1.0 - a_peak)**2) / (2 * width**2))
        return 1.0 + delta * (decay_profile - norm_at_1)

    elif model_type == 'power_law':
        # Λ(a) = Λ₀ × [1 + δ(a⁻ⁿ - 1)]
        # Power law decay — мотивирано от density scaling
        delta, n = params[0], params[1]
        return 1.0 + delta * (a**(-n) - 1.0)

    elif model_type == 'u_curve':
        # Два компонента: ранно разклоняване + късно (структурно) разклоняване
        # Λ(a) = Λ₀ × [1 + δ₁(a⁻ⁿ - 1) + δ₂ × late_boost(a)]
        delta1, n, delta2, a_boost = params[0], params[1], params[2], params[3]
        early = delta1 * (a**(-n) - 1.0)
        # Късен boost: ускоряване при a > a_boost
        late = delta2 * np.maximum(0, (1.0 - a/a_boost)) * np.exp(-(1.0-a)**2 / 0.1)
        return 1.0 + early + late

    elif model_type == 'structure_formation':
        # Базиран на реални наблюдения: cosmic SFR (Madau & Dickinson 2014)
        # Branching rate ~ SFR ~ (1+z)^2.7 / (1+((1+z)/2.9)^5.6)
        # Кумулативен ефект върху Λ
        delta = params[0]
        z = 1.0/np.maximum(a, 0.01) - 1.0
        sfr = ((1+z)**2.7) / (1 + ((1+z)/2.9)**5.6)
        sfr_now = 1.0  # SFR(z=0)
        # Λ ~ remaining vacuum energy after branching up to this point
        # По-високо SFR в миналото = повече branching = повече Λ тогава
        return 1.0 + delta * (sfr - sfr_now)

    elif model_type == 'exponential_decay':
        # Λ(a) = Λ₀ × [1 + δ(e^(-β(a-1)) - 1)]
        # Експоненциално — мотивирано от "горене на гориво"
        delta, beta = params[0], params[1]
        return 1.0 + delta * (np.exp(-beta * (a - 1.0)) - 1.0)

    return np.ones_like(a)


def compute_w(a_array, params, model_type):
    """
    w(a) = -1 - (1/3) × d(ln Λ)/d(ln a)
    Числено диференциране.
    """
    w_values = []
    for a in a_array:
        da = 0.001 * a
        L_minus = lambda_model(a - da, params, model_type)
        L_plus = lambda_model(a + da, params, model_type)
        L_center = lambda_model(a, params, model_type)
        dL_da = (L_plus - L_minus) / (2 * da)
        dlnL_dlna = (a / L_center) * dL_da
        w = -1.0 - (1.0/3.0) * dlnL_dlna
        w_values.append(w)
    return np.array(w_values)


def fit_w0_wa(a_array, w_array):
    """Фитва w(a) = w₀ + wₐ(1-a)"""
    A = np.column_stack([np.ones_like(a_array), 1.0 - a_array])
    result = np.linalg.lstsq(A, w_array, rcond=None)
    return result[0]  # w0, wa


def desi_distance(w0, wa):
    """Разстояние в σ от DESI центъра"""
    return np.sqrt(((w0 - DESI_w0)/DESI_w0_err)**2 + ((wa - DESI_wa)/DESI_wa_err)**2)


# ============================================================
# ОПТИМИЗАЦИЯ: намери параметри, най-близки до DESI
# ============================================================

def optimize_model(model_type, param_bounds, n_params):
    """
    Търси параметри, които минимизират разстоянието до DESI.
    """
    a_fit = np.linspace(0.2, 1.0, 80)

    def objective(params):
        try:
            w_vals = compute_w(a_fit, params, model_type)
            if np.any(np.isnan(w_vals)) or np.any(np.abs(w_vals) > 100):
                return 1e6
            w0, wa = fit_w0_wa(a_fit, w_vals)
            return desi_distance(w0, wa)
        except:
            return 1e6

    best_dist = 1e6
    best_params = None
    best_w0, best_wa = None, None

    # Grid search + local optimization
    from itertools import product
    n_grid = max(5, int(20 / n_params))

    grids = []
    for low, high in param_bounds:
        grids.append(np.linspace(low, high, n_grid))

    for combo in product(*grids):
        params = list(combo)
        dist = objective(params)
        if dist < best_dist:
            best_dist = dist
            best_params = params

    # Local refinement
    if best_params is not None:
        try:
            result = minimize(objective, best_params, method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            if result.fun < best_dist:
                best_params = list(result.x)
                best_dist = result.fun
        except:
            pass

    # Extract w0, wa at best params
    w_vals = compute_w(a_fit, best_params, model_type)
    best_w0, best_wa = fit_w0_wa(a_fit, w_vals)

    return best_params, best_w0, best_wa, best_dist


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("MULTIVERSE BRANCHING-DECAY MODEL v2")
    print("=" * 70)
    print()
    print("Direct parametric Λ(a) models, optimized against DESI DR2")
    print()

    # Модели с техните параметри и граници
    models = {
        'linear': {
            'name': 'Линеен decay',
            'desc': 'L(a) = L0 * [1 + d*(1-a)]',
            'bounds': [(0.001, 1.0)],
            'n_params': 1
        },
        'power_law': {
            'name': 'Power law decay',
            'desc': 'L(a) = L0 * [1 + d*(a^-n - 1)]',
            'bounds': [(0.001, 0.5), (0.1, 3.0)],
            'n_params': 2
        },
        'exponential_decay': {
            'name': 'Exponential decay',
            'desc': 'L(a) = L0 * [1 + d*(exp(-b*(a-1)) - 1)]',
            'bounds': [(0.001, 0.5), (0.5, 10.0)],
            'n_params': 2
        },
        'sfr_peak': {
            'name': 'SFR-peak decay',
            'desc': 'Branching follows star formation rate peak',
            'bounds': [(0.01, 1.0), (0.2, 0.6), (0.05, 0.3)],
            'n_params': 3
        },
        'structure_formation': {
            'name': 'Cosmic SFR (Madau & Dickinson)',
            'desc': 'Branching ~ cosmic star formation rate',
            'bounds': [(0.001, 0.3)],
            'n_params': 1
        }
    }

    results = {}

    for model_type, config in models.items():
        print(f"\n--- {config['name']}: {config['desc']} ---")

        params, w0, wa, dist = optimize_model(
            model_type, config['bounds'], config['n_params'])

        results[model_type] = {
            'params': params, 'w0': w0, 'wa': wa, 'dist': dist,
            'name': config['name']
        }

        param_str = ', '.join([f'{p:.4f}' for p in params])
        print(f"  Optimized params: [{param_str}]")
        print(f"  w0 = {w0:.4f}, wa = {wa:.4f}")
        print(f"  Distance from DESI: {dist:.3f} sigma")

    # ============================================================
    # ВИЗУАЛИЗАЦИЯ
    # ============================================================

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    colors = {'linear': 'red', 'power_law': 'green', 'exponential_decay': 'orange',
              'sfr_peak': 'purple', 'structure_formation': 'brown'}

    a_plot = np.linspace(0.1, 1.0, 200)
    a_w = np.linspace(0.15, 1.0, 100)

    # --- Plot 1: w0-wa plane ---
    ax1 = axes[0, 0]
    theta = np.linspace(0, 2*np.pi, 100)
    for sigma, alpha in [(1, 0.3), (2, 0.15), (3, 0.07)]:
        ell_w0 = DESI_w0 + sigma * DESI_w0_err * np.cos(theta)
        ell_wa = DESI_wa + sigma * DESI_wa_err * np.sin(theta)
        ax1.fill(ell_w0, ell_wa, alpha=alpha, color='blue',
                 label=f'DESI {sigma}s' if sigma <= 2 else '')
        ax1.plot(ell_w0, ell_wa, 'b-', alpha=0.3, linewidth=0.5)

    ax1.plot(LCDM_w0, LCDM_wa, 'k+', markersize=15, markeredgewidth=2, label='LCDM')

    for model_type, res in results.items():
        ax1.plot(res['w0'], res['wa'], 'o', color=colors[model_type],
                 markersize=10, label=f"{res['name']} ({res['dist']:.2f}s)")

    ax1.set_xlabel('w0', fontsize=13)
    ax1.set_ylabel('wa', fontsize=13)
    ax1.set_title('Branching-Decay models vs DESI DR2 (2025)', fontsize=14)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.set_xlim(-1.3, -0.4)
    ax1.set_ylim(-2.5, 1.5)
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
    ax1.axvline(x=-1, color='gray', linestyle=':', alpha=0.3)
    ax1.grid(True, alpha=0.2)

    # --- Plot 2: Lambda(a)/Lambda_0 ---
    ax2 = axes[0, 1]
    for model_type, res in results.items():
        L_vals = lambda_model(a_plot, res['params'], model_type)
        ax2.plot(a_plot, L_vals, '-', color=colors[model_type], linewidth=2,
                 label=res['name'])
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='LCDM')
    ax2.set_xlabel('Scale factor a (a=1 today)', fontsize=12)
    ax2.set_ylabel('L(a) / L(today)', fontsize=12)
    ax2.set_title('Vacuum energy evolution', fontsize=14)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    # --- Plot 3: w(a) ---
    ax3 = axes[1, 0]
    w_desi = DESI_w0 + DESI_wa * (1.0 - a_w)
    ax3.fill_between(a_w,
                     DESI_w0 - DESI_w0_err + (DESI_wa - DESI_wa_err) * (1 - a_w),
                     DESI_w0 + DESI_w0_err + (DESI_wa + DESI_wa_err) * (1 - a_w),
                     alpha=0.15, color='blue', label='DESI 1s band')
    ax3.plot(a_w, w_desi, 'b--', linewidth=1.5, alpha=0.7, label='DESI best fit')

    for model_type, res in results.items():
        w_vals = compute_w(a_w, res['params'], model_type)
        ax3.plot(a_w, w_vals, '-', color=colors[model_type], linewidth=2,
                 label=res['name'])

    ax3.axhline(y=-1.0, color='black', linestyle=':', alpha=0.5, label='w=-1 (LCDM)')
    ax3.set_xlabel('Scale factor a', fontsize=12)
    ax3.set_ylabel('w(a)', fontsize=12)
    ax3.set_title('Equation of state w(a)', fontsize=14)
    ax3.legend(fontsize=7)
    ax3.set_ylim(-2.5, 0.5)
    ax3.grid(True, alpha=0.2)

    # --- Plot 4: Summary table as text ---
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary = "RESULTS SUMMARY\n" + "=" * 50 + "\n\n"
    summary += f"DESI DR2:  w0={DESI_w0:.3f}+-{DESI_w0_err}, wa={DESI_wa:.2f}+-{DESI_wa_err}\n"
    summary += f"LCDM:      w0={LCDM_w0:.3f}, wa={LCDM_wa:.2f}  "
    summary += f"[{desi_distance(LCDM_w0, LCDM_wa):.2f}s from DESI]\n"
    summary += "\n" + "-" * 50 + "\n\n"

    # Sort by distance
    sorted_models = sorted(results.items(), key=lambda x: x[1]['dist'])

    for model_type, res in sorted_models:
        summary += f"{res['name']}:\n"
        summary += f"  w0={res['w0']:.4f}, wa={res['wa']:.4f}\n"
        param_str = ', '.join([f'{p:.4f}' for p in res['params']])
        summary += f"  params=[{param_str}]\n"
        summary += f"  Distance: {res['dist']:.3f}s from DESI\n\n"

    # Quintom crossing
    summary += "-" * 50 + "\n"
    summary += "QUINTOM CROSSING (w crosses -1):\n"
    for model_type, res in sorted_models:
        w_early = compute_w(np.array([0.3]), res['params'], model_type)[0]
        w_late = compute_w(np.array([0.95]), res['params'], model_type)[0]
        crosses = "YES" if (w_early + 1) * (w_late + 1) < 0 else "NO"
        summary += f"  {res['name']}: w(0.3)={w_early:.3f}, w(0.95)={w_late:.3f} -> {crosses}\n"

    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('c:/Projects/Multiverse-Evidence/branching_decay_v2_results.png', dpi=150)
    print("\n\nPlot saved: branching_decay_v2_results.png")

    # Final console summary
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"\n{'Model':<35} {'w0':>8} {'wa':>8} {'dist':>8}")
    print("-" * 65)
    print(f"{'LCDM':<35} {LCDM_w0:>8.4f} {LCDM_wa:>8.4f} {desi_distance(LCDM_w0, LCDM_wa):>8.3f}s")

    for model_type, res in sorted_models:
        print(f"{res['name']:<35} {res['w0']:>8.4f} {res['wa']:>8.4f} {res['dist']:>8.3f}s")

    print(f"\n{'DESI DR2 (observed)':<35} {DESI_w0:>8.4f} {DESI_wa:>8.4f} {'0.000':>8}s")

    # Verdict
    best = sorted_models[0]
    print(f"\n>>> Best model: {best[1]['name']} at {best[1]['dist']:.3f}s from DESI")
    print(f">>> LCDM: {desi_distance(LCDM_w0, LCDM_wa):.3f}s from DESI")
    print(f">>> Improvement: {desi_distance(LCDM_w0, LCDM_wa) - best[1]['dist']:.3f}s closer")

    if best[1]['dist'] < 1.0:
        print("\n*** MODEL IS WITHIN 1 SIGMA OF DESI — STRONG COMPATIBILITY ***")
    elif best[1]['dist'] < 2.0:
        print("\n** Model is within 2 sigma of DESI — compatible **")
    elif best[1]['dist'] < 3.0:
        print("\n* Model is within 3 sigma of DESI — marginally compatible *")
    else:
        print("\nModel is >3 sigma from DESI — poor fit")


if __name__ == '__main__':
    main()
