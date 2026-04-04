"""
Multiverse Branching-Decay Model: First-Level Check
====================================================

Хипотеза: Вакуумната енергия намалява с всяко квантово разклоняване.
  Λ(t) = Λ_initial - N(t) × E_branch

Тест: Превеждаме Λ(t) в параметри на тъмната енергия (w₀, wₐ)
и сравняваме с DESI DR2 (2025) наблюдения.

Автор: Живко Добромирски
Дата: 2026-04-01
"""

import numpy as np
from scipy.integrate import quad
# derivative не е нужен — ползваме finite differences директно
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# КОНСТАНТИ
# ============================================================

# Настоящи космологични параметри (Planck 2018 + DESI 2025)
H0 = 67.4  # km/s/Mpc — Hubble constant
Omega_m0 = 0.315  # Matter density parameter today
Omega_r0 = 9.1e-5  # Radiation density parameter today
Omega_L0 = 1.0 - Omega_m0 - Omega_r0  # Dark energy density today

# DESI DR2 (2025) наблюдавани стойности за динамична тъмна енергия
# w(a) = w0 + wa * (1 - a), където a = 1/(1+z) е scale factor
DESI_w0 = -0.752  # DESI DR2 central value
DESI_w0_err = 0.067
DESI_wa = -0.60  # DESI DR2 central value
DESI_wa_err = 0.29

# ΛCDM предсказания
LCDM_w0 = -1.0
LCDM_wa = 0.0

# ============================================================
# МОДЕЛ: BRANCHING RATE N(t)
# ============================================================

def branching_rate_density(a, model='radiation_dominated'):
    """
    Скорост на разклоняване като функция на scale factor a.

    Физическа мотивация: branching rate ∝ брой квантови взаимодействия
    в единица обем за единица време.

    В ранната вселена (radiation era): rate ∝ T⁴ ∝ a⁻⁴
    В материалната ера: rate ∝ n × σv ∝ a⁻³ (по-бавен спад)

    Параметри:
        a: scale factor (a=1 днес, a→0 в миналото)
        model: 'radiation_dominated', 'mixed', 'structure'
    """
    if model == 'radiation_dominated':
        # Най-прост модел: branching ∝ radiation energy density
        return a**(-4)

    elif model == 'mixed':
        # По-реалистичен: radiation доминира рано, matter по-късно
        # a_eq ≈ Omega_r0/Omega_m0 ≈ 3e-4 (radiation-matter equality)
        a_eq = Omega_r0 / Omega_m0
        return a**(-4) * (1 + a/a_eq)**(-1) + 0.1 * a**(-3)

    elif model == 'structure':
        # С формиране на структура: повече взаимодействия при a > 0.1
        a_eq = Omega_r0 / Omega_m0
        base = a**(-4) * (1 + a/a_eq)**(-1)
        # Структурата добавя взаимодействия при по-късни времена
        structure = 0.5 * np.exp(-((a - 0.5)**2) / (2 * 0.2**2))
        return base + structure

    elif model == 'u_curve_mild':
        # U-крива: ранна вселена (radiation) + късно ускоряване (структура)
        # Физическа мотивация: звезди, химия, живот, технологии
        a_eq = Omega_r0 / Omega_m0
        # Ранен пик: radiation era
        early = a**(-4) * (1 + a/a_eq)**(-1)
        # Късен подем: формиране на структура, звезди (пик при a~0.6, z~0.7)
        # Star formation rate пикира при z~1.5-2 (a~0.33-0.4)
        late = 2.0 * np.exp(-((a - 0.45)**2) / (2 * 0.15**2))
        return early + late

    elif model == 'u_curve_strong':
        # По-силен късен подем — включва експоненциален ръст
        # (повече структура → повече взаимодействия → повече branching)
        a_eq = Omega_r0 / Omega_m0
        early = a**(-4) * (1 + a/a_eq)**(-1)
        # Star formation peak при a~0.4 + експоненциален ръст при a>0.7
        sfr = 3.0 * np.exp(-((a - 0.4)**2) / (2 * 0.12**2))
        # Експоненциален ръст в късна вселена (галактики, черни дупки, живот)
        late_exp = 1.5 * np.exp(3.0 * (a - 1.0))  # расте към a=1
        return early + sfr + late_exp

    elif model == 'u_curve_sfr':
        # Базиран на реалния cosmic star formation rate (Madau & Dickinson 2014)
        # SFR(z) ∝ (1+z)^2.7 / (1 + ((1+z)/2.9)^5.6)
        # Преведено в a: z = 1/a - 1, 1+z = 1/a
        a_eq = Omega_r0 / Omega_m0
        early = a**(-4) * (1 + a/a_eq)**(-1)
        # Cosmic SFR като proxy за branching rate в късната вселена
        z = 1.0/a - 1.0 if a > 0.01 else 99.0
        sfr = ((1+z)**2.7) / (1 + ((1+z)/2.9)**5.6)
        # Нормализираме SFR компонента
        return early + 5.0 * sfr

    elif model == 'late_dominant':
        # Модел, където късното разклоняване ДОМИНИРА
        # (тест: какво ако повечето branching е от структура, не от radiation?)
        a_eq = Omega_r0 / Omega_m0
        early = 0.01 * a**(-4) * (1 + a/a_eq)**(-1)  # потиснат ранен принос
        # Силен късен принос
        z = 1.0/a - 1.0 if a > 0.01 else 99.0
        sfr = ((1+z)**2.7) / (1 + ((1+z)/2.9)**5.6)
        late = 2.0 * np.exp(2.0 * (a - 1.0))
        return early + 10.0 * sfr + late

    return a**(-4)


def cumulative_branching(a, model='radiation_dominated'):
    """
    Кумулативен брой разклонения от Big Bang (a→0) до scale factor a.
    Нормализиран: N(a=1) = 1.
    """
    # Интегрираме branching rate × dt, където dt = da/(a*H(a))
    # H(a) = H0 * sqrt(Omega_r0*a⁻⁴ + Omega_m0*a⁻³ + Omega_L0)

    def integrand(a_prime):
        H_ratio = np.sqrt(Omega_r0 * a_prime**(-4) +
                          Omega_m0 * a_prime**(-3) +
                          Omega_L0)
        rate = branching_rate_density(a_prime, model)
        # dt = da / (a * H), rate × dt = rate / (a * H) * da
        return rate / (a_prime * H_ratio)

    # Интегрираме от малко a (не 0, за да избегнем сингулярност) до a
    a_min = 1e-10
    if a <= a_min:
        return 0.0

    result, _ = quad(integrand, a_min, a, limit=200)
    return result


def compute_N_normalized(a_array, model='radiation_dominated'):
    """
    Изчислява N(a)/N(1) за масив от scale factors.
    """
    # Първо намираме N(a=1) за нормализация
    N_total = cumulative_branching(1.0, model)

    N_values = []
    for a in a_array:
        N_a = cumulative_branching(a, model)
        N_values.append(N_a / N_total)

    return np.array(N_values)


# ============================================================
# ПРЕВОД: Λ(a) → w(a)
# ============================================================

def lambda_ratio(a, delta, model='radiation_dominated', N_cache=None):
    """
    Λ(a)/Λ₀ = 1 + δ × (1 - N(a)/N_total)

    δ > 0 означава: Λ е била по-голяма в миналото (повече "гориво")
    δ = 0 → стандартен ΛCDM

    Параметри:
        a: scale factor
        delta: относителен размер на branching decay ефекта
        model: модел за branching rate
        N_cache: предварително изчислени N стойности (за скорост)
    """
    if N_cache is not None:
        # Интерполираме от кеш
        a_cache, N_cache_vals = N_cache
        N_norm = np.interp(a, a_cache, N_cache_vals)
    else:
        N_norm = compute_N_normalized(np.array([a]), model)[0]

    return 1.0 + delta * (1.0 - N_norm)


def effective_w(a, delta, model='radiation_dominated', N_cache=None):
    """
    Ефективно уравнение на състоянието w(a) от Λ(a).

    За произволно Λ(a):
    w(a) = -1 - (1/3) × d(ln Λ)/d(ln a)

    Ако Λ = const → d(ln Λ)/d(ln a) = 0 → w = -1 (ΛCDM)
    Ако Λ намалява → d(ln Λ)/d(ln a) > 0 → w < -1 (phantom-like)
    Ако Λ намалява все по-бавно → w се връща към -1
    """
    da = 0.001 * a

    L_minus = lambda_ratio(a - da, delta, model, N_cache)
    L_plus = lambda_ratio(a + da, delta, model, N_cache)

    # d(ln Λ)/d(ln a) = (a/Λ) × dΛ/da
    L_center = lambda_ratio(a, delta, model, N_cache)
    dL_da = (L_plus - L_minus) / (2 * da)
    dlnL_dlna = (a / L_center) * dL_da

    return -1.0 - (1.0/3.0) * dlnL_dlna


def fit_w0_wa(delta, model='radiation_dominated'):
    """
    Фитва w(a) = w₀ + wₐ(1-a) към нашия модел.
    Връща (w₀, wₐ).
    """
    # Кеширане на N(a)
    a_cache = np.linspace(0.01, 1.0, 500)
    N_cache_vals = compute_N_normalized(a_cache, model)
    cache = (a_cache, N_cache_vals)

    # Изчисляваме w(a) за набор от a стойности
    a_fit = np.linspace(0.3, 1.0, 50)  # z=0 до z≈2.3 (DESI обхват)
    w_values = np.array([effective_w(a, delta, model, cache) for a in a_fit])

    # Линеен фит: w(a) ≈ w₀ + wₐ(1-a)
    # Матрица: w = [1, (1-a)] × [w₀, wₐ]ᵀ
    A = np.column_stack([np.ones_like(a_fit), 1.0 - a_fit])
    result = np.linalg.lstsq(A, w_values, rcond=None)
    w0_fit, wa_fit = result[0]

    return w0_fit, wa_fit, a_fit, w_values, cache


# ============================================================
# ГЛАВНА СМЕТКА
# ============================================================

def main():
    print("=" * 70)
    print("MULTIVERSE BRANCHING-DECAY MODEL: FIRST-LEVEL CHECK")
    print("=" * 70)
    print()
    print("Хипотеза: Λ(a) = Λ₀ × [1 + δ × (1 - N(a)/N_total)]")
    print("Тест: Сравнение на модела с DESI DR2 (2025) w₀, wₐ")
    print()

    models = ['radiation_dominated', 'u_curve_mild', 'u_curve_strong',
              'u_curve_sfr', 'late_dominant']
    model_names = {
        'radiation_dominated': 'Радиационен (rate ~ T4)',
        'mixed': 'Смесен (radiation + matter)',
        'structure': 'Структурен (+ формиране)',
        'u_curve_mild': 'U-крива умерена (SFR пик)',
        'u_curve_strong': 'U-крива силна (+ експоненц.)',
        'u_curve_sfr': 'Cosmic SFR (Madau & Dickinson)',
        'late_dominant': 'Късно-доминиран (структура >> radiation)'
    }

    # Сканираме δ за всеки модел
    deltas = np.linspace(0.001, 0.5, 100)

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))

    # === PLOT 1: w₀ vs wₐ за различни модели ===
    ax1 = axes[0, 0]

    # DESI confidence ellipse (приблизителна)
    theta = np.linspace(0, 2*np.pi, 100)
    for sigma, alpha in [(1, 0.3), (2, 0.15)]:
        ell_w0 = DESI_w0 + sigma * DESI_w0_err * np.cos(theta)
        ell_wa = DESI_wa + sigma * DESI_wa_err * np.sin(theta)
        ax1.fill(ell_w0, ell_wa, alpha=alpha, color='blue', label=f'DESI {sigma}σ' if sigma == 1 else '')
        ax1.plot(ell_w0, ell_wa, 'b-', alpha=0.5, linewidth=0.5)

    ax1.plot(LCDM_w0, LCDM_wa, 'k+', markersize=15, markeredgewidth=2, label='ΛCDM (w₀=-1, wₐ=0)')

    colors = ['red', 'green', 'orange', 'purple', 'brown']
    best_results = {}

    for idx, model in enumerate(models):
        w0_list, wa_list = [], []
        print(f"\n--- Модел: {model_names[model]} ---")
        print(f"{'δ':>8} | {'w₀':>8} | {'wₐ':>8} | {'Δw₀ от DESI':>12} | {'Δwₐ от DESI':>12}")
        print("-" * 60)

        for i, delta in enumerate(deltas):
            w0, wa, _, _, _ = fit_w0_wa(delta, model)
            w0_list.append(w0)
            wa_list.append(wa)

            if i % 20 == 0:
                dw0 = abs(w0 - DESI_w0)
                dwa = abs(wa - DESI_wa)
                print(f"{delta:8.3f} | {w0:8.4f} | {wa:8.4f} | {dw0:12.4f} | {dwa:12.4f}")

        w0_arr = np.array(w0_list)
        wa_arr = np.array(wa_list)

        # Намираме δ, което е най-близо до DESI
        dist = ((w0_arr - DESI_w0)/DESI_w0_err)**2 + ((wa_arr - DESI_wa)/DESI_wa_err)**2
        best_idx = np.argmin(dist)
        best_delta = deltas[best_idx]
        best_w0 = w0_arr[best_idx]
        best_wa = wa_arr[best_idx]
        best_dist = np.sqrt(dist[best_idx])

        best_results[model] = {
            'delta': best_delta, 'w0': best_w0, 'wa': best_wa, 'sigma_dist': best_dist
        }

        ax1.plot(w0_arr, wa_arr, '-', color=colors[idx], linewidth=2,
                 label=f'{model_names[model]}')
        ax1.plot(best_w0, best_wa, 'o', color=colors[idx], markersize=10)

        print(f"\n  → Най-близо до DESI: δ = {best_delta:.3f}")
        print(f"    w₀ = {best_w0:.4f} (DESI: {DESI_w0}), wₐ = {best_wa:.4f} (DESI: {DESI_wa})")
        print(f"    Разстояние: {best_dist:.2f}σ от DESI центъра")

    ax1.set_xlabel('w₀', fontsize=12)
    ax1.set_ylabel('wₐ', fontsize=12)
    ax1.set_title('Branching-Decay модел vs DESI DR2', fontsize=13)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.set_xlim(-1.3, -0.3)
    ax1.set_ylim(-3.0, 2.0)
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
    ax1.axvline(x=-1, color='gray', linestyle=':', alpha=0.3)
    ax1.grid(True, alpha=0.2)

    # === PLOT 2: Λ(a)/Λ₀ за различни модели (при best δ) ===
    ax2 = axes[0, 1]
    a_plot = np.linspace(0.05, 1.0, 200)

    for idx, model in enumerate(models):
        best = best_results[model]
        N_vals = compute_N_normalized(a_plot, model)
        L_vals = 1.0 + best['delta'] * (1.0 - N_vals)

        ax2.plot(a_plot, L_vals, '-', color=colors[idx], linewidth=2,
                 label=f"{model_names[model]} (δ={best['delta']:.3f})")

    ax2.axhline(y=1.0, color='black', linestyle='--', label='LCDM (L = const)')
    ax2.set_xlabel('Scale factor a (a=1 dnes)', fontsize=12)
    ax2.set_ylabel('L(a) / L0', fontsize=12)
    ax2.set_title('Evolution of vacuum energy (late universe a>0.2)', fontsize=13)
    ax2.set_xlim(0.2, 1.0)  # Focus on late universe where differences matter
    ax2.legend(fontsize=7, loc='upper left')
    ax2.grid(True, alpha=0.2)

    # === PLOT 3: w(a) за различни модели (при best δ) ===
    ax3 = axes[1, 0]
    a_w = np.linspace(0.15, 1.0, 100)

    for idx, model in enumerate(models):
        best = best_results[model]
        _, _, _, _, cache = fit_w0_wa(best['delta'], model)
        w_vals = [effective_w(a, best['delta'], model, cache) for a in a_w]

        ax3.plot(a_w, w_vals, '-', color=colors[idx], linewidth=2,
                 label=f"{model_names[model]}")

        # DESI линеен фит за сравнение
        if idx == 0:
            w_desi = DESI_w0 + DESI_wa * (1.0 - a_w)
            ax3.fill_between(a_w,
                            DESI_w0 - DESI_w0_err + (DESI_wa - DESI_wa_err) * (1 - a_w),
                            DESI_w0 + DESI_w0_err + (DESI_wa + DESI_wa_err) * (1 - a_w),
                            alpha=0.15, color='blue', label='DESI 1σ region')
            ax3.plot(a_w, w_desi, 'b--', linewidth=1, alpha=0.5, label='DESI best fit')

    ax3.axhline(y=-1.0, color='black', linestyle=':', alpha=0.5, label='w = -1 (ΛCDM)')
    ax3.set_xlabel('Scale factor a', fontsize=12)
    ax3.set_ylabel('w(a)', fontsize=12)
    ax3.set_title('Уравнение на състоянието w(a)', fontsize=13)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.2)

    # === PLOT 4: N(a)/N_total — кумулативно разклоняване ===
    ax4 = axes[1, 1]

    for idx, model in enumerate(models):
        N_vals = compute_N_normalized(a_plot, model)
        ax4.plot(a_plot, N_vals, '-', color=colors[idx], linewidth=2,
                 label=f"{model_names[model]}")

    ax4.set_xlabel('Scale factor a', fontsize=12)
    ax4.set_ylabel('N(a) / N(today)', fontsize=12)
    ax4.set_title('Кумулативно разклоняване', fontsize=13)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('c:/Projects/Multiverse-Evidence/branching_decay_results.png', dpi=150)
    print("\n\nГрафиката е запазена: branching_decay_results.png")

    # === ОБОБЩЕНИЕ ===
    print("\n" + "=" * 70)
    print("ОБОБЩЕНИЕ")
    print("=" * 70)

    print(f"\nDESI DR2 наблюдения: w₀ = {DESI_w0} ± {DESI_w0_err}, wₐ = {DESI_wa} ± {DESI_wa_err}")
    print(f"ΛCDM предсказание:  w₀ = {LCDM_w0}, wₐ = {LCDM_wa}")
    print(f"ΛCDM разстояние от DESI: {np.sqrt(((LCDM_w0-DESI_w0)/DESI_w0_err)**2 + ((LCDM_wa-DESI_wa)/DESI_wa_err)**2):.2f}σ")

    print(f"\nBranching-Decay модел резултати:")
    for model in models:
        b = best_results[model]
        print(f"\n  {model_names[model]}:")
        print(f"    Оптимално δ = {b['delta']:.3f}")
        print(f"    w₀ = {b['w0']:.4f}, wₐ = {b['wa']:.4f}")
        print(f"    Разстояние от DESI: {b['sigma_dist']:.2f}σ")

    # Quintom crossing check
    print(f"\n--- Quintom Crossing (w пресича -1) ---")
    for idx, model in enumerate(models):
        best = best_results[model]
        _, _, _, _, cache = fit_w0_wa(best['delta'], model)
        w_early = effective_w(0.2, best['delta'], model, cache)
        w_now = effective_w(0.99, best['delta'], model, cache)
        crosses = (w_early + 1) * (w_now + 1) < 0
        print(f"  {model_names[model]}: w(a=0.2)={w_early:.4f}, w(a=1)={w_now:.4f}, crossing={'ДА' if crosses else 'НЕ'}")

    print("\n" + "=" * 70)
    print("ИНТЕРПРЕТАЦИЯ")
    print("=" * 70)
    print("""
Ако разстоянието от DESI < 2σ: Моделът е СЪВМЕСТИМ с наблюденията.
  → Не доказва хипотезата, но не я отхвърля.
  → Branching-decay е ПОНЕ толкова добро обяснение, колкото ad hoc w₀wₐCDM.
  → Разликата: нашият модел има ФИЗИЧЕСКА МОТИВАЦИЯ.

Ако разстоянието от DESI > 3σ: Моделът НЕ е съвместим.
  → Конкретната форма на N(t) не работи.
  → Може да се опита с друга параметризация, но внимание за overfitting.

ВАЖНО: Дори перфектно съвпадение не доказва мултивселена.
Доказва само, че branching-decay Λ(t) е математически съвместим
с наблюденията — което е необходимо, но не достатъчно условие.
    """)


if __name__ == '__main__':
    main()
