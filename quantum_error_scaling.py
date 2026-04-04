"""
Quantum Computer Error Scaling Analysis
========================================

Хипотеза: Ако комбинаторният натиск от мултивселената е реален,
error rate в квантови компютри трябва да скалира суперлинейно
с броя кубити (след изчистване на известни източници).

Тест: Анализ на публични benchmark данни от множество процесори.

Автор: Живко Добромирски
Дата: 2026-04-02
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import os
import json

# ============================================================
# DATA: Quantum Processor Benchmarks (collected from public sources)
# ============================================================

# Multi-processor comparison data
# Sources: IBM arXiv:2410.00916, Google Willow spec sheet, Quantinuum H2,
#          IonQ specs, Rigetti press releases, academic papers

PROCESSORS = [
    # Superconducting - IBM
    {'name': 'IBM Falcon r5.11 (ibmq_manila)', 'qubits': 5, 'gate_2q_error': 7.5e-3,
     'gate_1q_error': 3.0e-4, 'readout_error': 1.5e-2, 'tech': 'superconducting', 'year': 2021, 'source': 'IBM'},
    {'name': 'IBM Falcon r8 (ibmq_kolkata)', 'qubits': 27, 'gate_2q_error': 1.2e-2,
     'gate_1q_error': 4.0e-4, 'readout_error': 2.0e-2, 'tech': 'superconducting', 'year': 2022, 'source': 'IBM'},
    {'name': 'IBM Eagle r3 (ibm_sherbrooke)', 'qubits': 127, 'gate_2q_error': 7.4e-3,
     'gate_1q_error': 2.4e-4, 'readout_error': 1.35e-2, 'tech': 'superconducting', 'year': 2024, 'source': 'IBM arXiv:2410.00916'},
    {'name': 'IBM Eagle r3 (ibm_brisbane)', 'qubits': 127, 'gate_2q_error': 8.34e-3,
     'gate_1q_error': 2.5e-4, 'readout_error': 1.5e-2, 'tech': 'superconducting', 'year': 2024, 'source': 'IBM'},
    {'name': 'IBM Eagle r3 (ibm_kyiv)', 'qubits': 127, 'gate_2q_error': 1.16e-2,
     'gate_1q_error': 3.0e-4, 'readout_error': 1.8e-2, 'tech': 'superconducting', 'year': 2024, 'source': 'IBM'},
    {'name': 'IBM Heron r1 (ibm_torino)', 'qubits': 133, 'gate_2q_error': 4.77e-3,
     'gate_1q_error': 2.9e-4, 'readout_error': 1.5e-2, 'tech': 'superconducting', 'year': 2024, 'source': 'IBM'},
    {'name': 'IBM Heron r2 (ibm_fez)', 'qubits': 156, 'gate_2q_error': 3.7e-3,
     'gate_1q_error': 2.93e-4, 'readout_error': 1.76e-2, 'tech': 'superconducting', 'year': 2024, 'source': 'IBM arXiv:2410.00916'},

    # Superconducting - Google
    {'name': 'Google Sycamore', 'qubits': 53, 'gate_2q_error': 6.2e-3,
     'gate_1q_error': 1.0e-3, 'readout_error': 5.0e-3, 'tech': 'superconducting', 'year': 2021, 'source': 'Google'},
    {'name': 'Google Willow', 'qubits': 105, 'gate_2q_error': 3.3e-3,
     'gate_1q_error': 3.5e-4, 'readout_error': 7.7e-3, 'tech': 'superconducting', 'year': 2024, 'source': 'Google Willow spec'},

    # Superconducting - Rigetti
    {'name': 'Rigetti Ankaa-2', 'qubits': 84, 'gate_2q_error': 2.0e-2,
     'gate_1q_error': 5.0e-4, 'readout_error': 2.0e-2, 'tech': 'superconducting', 'year': 2023, 'source': 'Rigetti'},
    {'name': 'Rigetti Ankaa-3', 'qubits': 84, 'gate_2q_error': 5.0e-3,
     'gate_1q_error': 3.0e-4, 'readout_error': 1.0e-2, 'tech': 'superconducting', 'year': 2024, 'source': 'Rigetti'},

    # Trapped ions - Quantinuum
    {'name': 'Quantinuum H1', 'qubits': 20, 'gate_2q_error': 1.0e-3,
     'gate_1q_error': 2.0e-5, 'readout_error': 2.0e-3, 'tech': 'trapped-ion', 'year': 2023, 'source': 'Quantinuum'},
    {'name': 'Quantinuum H2', 'qubits': 56, 'gate_2q_error': 1.0e-3,
     'gate_1q_error': 1.0e-4, 'readout_error': 2.0e-3, 'tech': 'trapped-ion', 'year': 2024, 'source': 'Quantinuum H2 benchmark'},

    # Trapped ions - IonQ
    {'name': 'IonQ Aria', 'qubits': 25, 'gate_2q_error': 6.0e-3,
     'gate_1q_error': 6.0e-4, 'readout_error': 3.9e-3, 'tech': 'trapped-ion', 'year': 2024, 'source': 'IonQ'},
    {'name': 'IonQ Forte', 'qubits': 36, 'gate_2q_error': 4.0e-3,
     'gate_1q_error': 2.0e-4, 'readout_error': 5.0e-3, 'tech': 'trapped-ion', 'year': 2024, 'source': 'IonQ'},

    # Neutral atoms
    {'name': 'QuEra Aquila', 'qubits': 256, 'gate_2q_error': 5.0e-3,
     'gate_1q_error': 3.0e-3, 'readout_error': 2.0e-2, 'tech': 'neutral-atom', 'year': 2024, 'source': 'QuEra'},
]


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def power_law(x, a, b, alpha):
    """E(N) = a + b * N^alpha"""
    return a + b * np.power(x, alpha)

def linear(x, a, b):
    """E(N) = a + b * N"""
    return a + b * x

def log_model(x, a, b):
    """E(N) = a + b * ln(N)"""
    return a + b * np.log(x)


def analyze_scaling(processors, label, color, ax_main, ax_residual):
    """Analyze error rate vs qubit count for a set of processors."""

    qubits = np.array([p['qubits'] for p in processors])
    errors = np.array([p['gate_2q_error'] for p in processors])
    names = [p['name'].split('(')[0].strip() for p in processors]

    if len(qubits) < 4:
        print(f"  {label}: Too few data points ({len(qubits)}), skipping fit")
        ax_main.scatter(qubits, errors, s=80, label=f'{label} (N={len(qubits)})',
                       color=color, zorder=5)
        for i, name in enumerate(names):
            ax_main.annotate(name, (qubits[i], errors[i]), fontsize=6,
                           rotation=15, ha='left', va='bottom')
        return None

    # Sort by qubits
    sort_idx = np.argsort(qubits)
    qubits = qubits[sort_idx]
    errors = errors[sort_idx]
    names = [names[i] for i in sort_idx]

    print(f"\n{'='*60}")
    print(f"  {label} ({len(qubits)} processors)")
    print(f"{'='*60}")
    print(f"  {'Processor':<35} {'Qubits':>6} {'2Q Error':>10}")
    print(f"  {'-'*55}")
    for i in range(len(qubits)):
        print(f"  {names[i]:<35} {qubits[i]:>6} {errors[i]:>10.4e}")

    # Correlation
    if len(qubits) >= 4:
        r_pearson, p_pearson = pearsonr(qubits, errors)
        r_spearman, p_spearman = spearmanr(qubits, errors)
        print(f"\n  Pearson r = {r_pearson:.4f} (p = {p_pearson:.4f})")
        print(f"  Spearman rho = {r_spearman:.4f} (p = {p_spearman:.4f})")

    # Fit power law: E(N) = a + b * N^alpha
    try:
        popt_pow, pcov_pow = curve_fit(power_law, qubits, errors,
                                        p0=[0.005, 1e-5, 1.0],
                                        bounds=([0, -np.inf, -5], [0.1, np.inf, 5]),
                                        maxfev=10000)
        a, b, alpha = popt_pow
        residuals_pow = errors - power_law(qubits, *popt_pow)
        ss_res = np.sum(residuals_pow**2)
        ss_tot = np.sum((errors - np.mean(errors))**2)
        r_squared = 1 - ss_res/ss_tot if ss_tot > 0 else 0

        print(f"\n  Power law fit: E(N) = {a:.4e} + {b:.4e} * N^{alpha:.3f}")
        print(f"  R-squared = {r_squared:.4f}")
        print(f"\n  >>> alpha = {alpha:.3f}")
        if alpha > 0.1:
            print(f"  >>> SUPERLINEAR SCALING DETECTED (alpha > 0)")
        elif alpha < -0.1:
            print(f"  >>> SUBLINEAR / IMPROVING with N (alpha < 0)")
        else:
            print(f"  >>> APPROXIMATELY FLAT (alpha ~ 0)")
    except Exception as e:
        print(f"  Power law fit failed: {e}")
        popt_pow = None
        alpha = None

    # Fit linear
    try:
        popt_lin, _ = curve_fit(linear, qubits, errors, p0=[0.005, 1e-5])
        residuals_lin = errors - linear(qubits, *popt_lin)
        print(f"\n  Linear fit: E(N) = {popt_lin[0]:.4e} + {popt_lin[1]:.4e} * N")
    except:
        popt_lin = None

    # Plot
    ax_main.scatter(qubits, errors, s=80, color=color, zorder=5, edgecolors='black',
                   linewidth=0.5)
    for i, name in enumerate(names):
        ax_main.annotate(name, (qubits[i], errors[i]), fontsize=5,
                        rotation=15, ha='left', va='bottom', alpha=0.7)

    x_fit = np.linspace(max(1, min(qubits)*0.8), max(qubits)*1.2, 200)

    if popt_pow is not None:
        ax_main.plot(x_fit, power_law(x_fit, *popt_pow), '--', color=color,
                    linewidth=1.5, alpha=0.7,
                    label=f'{label}: a={alpha:.2f}, R2={r_squared:.3f}')

    if popt_lin is not None:
        ax_main.plot(x_fit, linear(x_fit, *popt_lin), ':', color=color,
                    linewidth=1, alpha=0.4)

    # Residuals
    if popt_pow is not None:
        ax_residual.scatter(qubits, residuals_pow, s=40, color=color, alpha=0.7)
        ax_residual.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    return {'alpha': alpha, 'r_squared': r_squared if popt_pow is not None else None,
            'n_points': len(qubits)}


# ============================================================
# LOAD QUANTINUUM MIRROR BENCHMARKING DATA
# ============================================================

def load_quantinuum_mirror_data():
    """Load mirror benchmarking data from Quantinuum H2 benchmark repo."""
    base_path = "c:/Projects/Multiverse-Evidence/data/quantinuum-h2-benchmark"
    mirror_path = os.path.join(base_path, "mirror_benchmarking", "data")

    results = []
    if os.path.exists(mirror_path):
        for fname in os.listdir(mirror_path):
            if fname.endswith('.json'):
                fpath = os.path.join(mirror_path, fname)
                try:
                    with open(fpath, 'r') as f:
                        data = json.load(f)
                    # Extract qubit count and fidelity
                    if isinstance(data, list):
                        for entry in data:
                            if isinstance(entry, dict):
                                n_qubits = entry.get('n_qubits') or entry.get('width') or entry.get('num_qubits')
                                fidelity = entry.get('fidelity') or entry.get('polarization') or entry.get('success_probability')
                                if n_qubits and fidelity:
                                    results.append({'qubits': n_qubits, 'fidelity': fidelity})
                    elif isinstance(data, dict):
                        n_qubits = data.get('n_qubits') or data.get('width')
                        fidelity = data.get('fidelity') or data.get('polarization')
                        if n_qubits and fidelity:
                            results.append({'qubits': n_qubits, 'fidelity': fidelity})
                except:
                    pass

    return results


# ============================================================
# LOAD METRIQ MIRROR CIRCUIT DATA
# ============================================================

def load_metriq_data():
    """Load mirror circuit data from metriq-data repo."""
    base_path = "c:/Projects/Multiverse-Evidence/data/metriq-data"
    results = []

    # Walk through all JSON files looking for mirror circuit results
    for root, dirs, files in os.walk(base_path):
        for fname in files:
            if fname.endswith('.json'):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        benchmark = data.get('benchmark', '')
                        if 'mirror' in str(benchmark).lower():
                            width = None
                            fidelity = None
                            params = data.get('params', {})
                            if isinstance(params, dict):
                                width = params.get('width')
                            result = data.get('result', {})
                            if isinstance(result, dict):
                                fidelity = result.get('polarization') or result.get('success_probability')
                            device = data.get('device', {})
                            device_name = device.get('name', '') if isinstance(device, dict) else str(device)
                            if width and fidelity:
                                results.append({
                                    'qubits': width,
                                    'fidelity': fidelity,
                                    'device': device_name
                                })
                except:
                    pass

    return results


# ============================================================
# LOAD HISTORICAL 2Q GATE ERRORS
# ============================================================

def load_historical_errors():
    """Load entangled state error data from awesome-quantum-experiments."""
    csv_path = "c:/Projects/Multiverse-Evidence/data/awesome-quantum-experiments/data/entangled_state_error_exp.csv"
    results = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                error = float(row['Entangled State Error'])
                year = int(row['Year'])
                platform = row['Platform']
                title = row['Article Title']
                results.append({
                    'error': error, 'year': year, 'platform': platform, 'title': title
                })
            except:
                pass

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("QUANTUM ERROR SCALING ANALYSIS")
    print("Multiverse Evidence Project — Phase 3")
    print("=" * 70)

    # ---- Analysis 1: Cross-processor error rate vs qubit count ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))

    ax1 = axes[0, 0]
    ax1_res = axes[0, 1]

    # All superconducting
    sc_procs = [p for p in PROCESSORS if p['tech'] == 'superconducting']
    ti_procs = [p for p in PROCESSORS if p['tech'] == 'trapped-ion']
    na_procs = [p for p in PROCESSORS if p['tech'] == 'neutral-atom']

    print("\n\n=== ANALYSIS 1: Cross-Processor 2Q Gate Error vs Qubit Count ===")

    result_sc = analyze_scaling(sc_procs, 'Superconducting', 'blue', ax1, ax1_res)
    result_ti = analyze_scaling(ti_procs, 'Trapped-Ion', 'red', ax1, ax1_res)
    result_na = analyze_scaling(na_procs, 'Neutral-Atom', 'green', ax1, ax1_res)

    ax1.set_xlabel('Number of Qubits', fontsize=12)
    ax1.set_ylabel('2-Qubit Gate Error Rate', fontsize=12)
    ax1.set_title('2Q Gate Error vs Qubit Count (all processors)', fontsize=13)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2, which='both')

    ax1_res.set_xlabel('Number of Qubits', fontsize=12)
    ax1_res.set_ylabel('Residual from power law fit', fontsize=12)
    ax1_res.set_title('Residuals', fontsize=13)
    ax1_res.grid(True, alpha=0.2)

    # ---- Analysis 2: Same-architecture scaling (IBM only) ----
    ax2 = axes[1, 0]

    ibm_procs = [p for p in PROCESSORS if 'IBM' in p['name']]
    ibm_qubits = np.array([p['qubits'] for p in ibm_procs])
    ibm_errors = np.array([p['gate_2q_error'] for p in ibm_procs])
    ibm_names = [p['name'].split('(')[0].strip() for p in ibm_procs]

    sort_idx = np.argsort(ibm_qubits)
    ibm_qubits = ibm_qubits[sort_idx]
    ibm_errors = ibm_errors[sort_idx]
    ibm_names = [ibm_names[i] for i in sort_idx]

    print(f"\n\n=== ANALYSIS 2: IBM-only Scaling ===")
    print(f"  {'Processor':<35} {'Qubits':>6} {'2Q Error':>10}")
    print(f"  {'-'*55}")
    for i in range(len(ibm_qubits)):
        print(f"  {ibm_names[i]:<35} {ibm_qubits[i]:>6} {ibm_errors[i]:>10.4e}")

    ax2.scatter(ibm_qubits, ibm_errors, s=100, color='blue', zorder=5,
               edgecolors='black', linewidth=0.5)
    for i, name in enumerate(ibm_names):
        ax2.annotate(name, (ibm_qubits[i], ibm_errors[i]), fontsize=6,
                    rotation=15, ha='left', va='bottom')

    # Fit
    if len(ibm_qubits) >= 4:
        try:
            popt, pcov = curve_fit(power_law, ibm_qubits, ibm_errors,
                                   p0=[0.005, 1e-5, 0.5],
                                   bounds=([0, -np.inf, -5], [0.1, np.inf, 5]),
                                   maxfev=10000)
            a, b, alpha = popt
            perr = np.sqrt(np.diag(pcov))
            x_fit = np.linspace(3, 200, 200)
            ax2.plot(x_fit, power_law(x_fit, *popt), 'b--', linewidth=1.5,
                    label=f'Power law: alpha={alpha:.3f} +/- {perr[2]:.3f}')

            residuals = ibm_errors - power_law(ibm_qubits, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((ibm_errors - np.mean(ibm_errors))**2)
            r_squared = 1 - ss_res/ss_tot if ss_tot > 0 else 0

            print(f"\n  Power law: E(N) = {a:.4e} + {b:.4e} * N^{alpha:.3f}")
            print(f"  alpha = {alpha:.3f} +/- {perr[2]:.3f}")
            print(f"  R-squared = {r_squared:.4f}")

            r_corr, p_corr = pearsonr(ibm_qubits, ibm_errors)
            print(f"  Pearson r = {r_corr:.4f} (p = {p_corr:.4f})")
        except Exception as e:
            print(f"  Fit failed: {e}")

    ax2.set_xlabel('Number of Qubits', fontsize=12)
    ax2.set_ylabel('2-Qubit Gate Error Rate', fontsize=12)
    ax2.set_title('IBM Processors Only: Error vs Qubit Count', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    # ---- Analysis 3: Historical error improvement per platform ----
    ax3 = axes[1, 1]

    hist_data = load_historical_errors()
    platforms = {}
    for d in hist_data:
        plat = d['platform']
        if plat not in platforms:
            platforms[plat] = {'years': [], 'errors': []}
        platforms[plat]['years'].append(d['year'])
        platforms[plat]['errors'].append(d['error'])

    plat_colors = {
        'Ion traps': 'red', 'Superconducting circuits': 'blue',
        'Neutral atoms': 'green', 'Semiconductor spins': 'purple',
        'Superconducting circuit': 'blue', 'NV centers': 'orange'
    }

    print(f"\n\n=== ANALYSIS 3: Historical 2Q Gate Error Trends ===")
    for plat, data in platforms.items():
        years = np.array(data['years'])
        errors = np.array(data['errors'])
        color = plat_colors.get(plat, 'gray')
        ax3.scatter(years, errors, s=40, color=color, alpha=0.7, label=plat)

        # Exponential fit: error = A * exp(-B * year)
        if len(years) >= 3:
            try:
                log_errors = np.log(errors)
                coeffs = np.polyfit(years, log_errors, 1)
                halving_time = np.log(2) / abs(coeffs[0]) if coeffs[0] != 0 else float('inf')
                years_fit = np.linspace(min(years), max(years), 100)
                errors_fit = np.exp(np.polyval(coeffs, years_fit))
                ax3.plot(years_fit, errors_fit, '--', color=color, alpha=0.5)
                print(f"  {plat}: halving time = {halving_time:.1f} years")
            except:
                pass

    ax3.set_xlabel('Year', fontsize=12)
    ax3.set_ylabel('2-Qubit Gate Error', fontsize=12)
    ax3.set_title('Historical Error Improvement by Platform', fontsize=13)
    ax3.set_yscale('log')
    ax3.legend(fontsize=7, loc='upper right')
    ax3.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('c:/Projects/Multiverse-Evidence/quantum_error_scaling.png', dpi=150)
    print("\n\nPlot saved: quantum_error_scaling.png")

    # ---- SUMMARY ----
    print("\n" + "=" * 70)
    print("SUMMARY & INTERPRETATION")
    print("=" * 70)

    print("""
KEY QUESTION: Does error rate scale superlinearly with qubit count?

If alpha > 0 in E(N) = a + b * N^alpha:
  -> Superlinear scaling = more qubits = disproportionately more errors
  -> Consistent with "combinatorial pressure" hypothesis
  -> BUT: also explainable by crosstalk, frequency crowding, fabrication yield

If alpha <= 0:
  -> No superlinear scaling
  -> Technology improvements dominate
  -> Combinatorial pressure effect is either absent or below noise floor

CRITICAL CAVEAT: Comparing DIFFERENT processors at DIFFERENT qubit counts
conflates scaling with technology improvements. IBM Heron r2 (156 qubits)
has BETTER error rates than older Eagle r3 (127 qubits) because it's newer
technology, not because fewer qubits.

THE CLEAN TEST would be: same chip architecture, same fabrication run,
varying number of ACTIVE qubits. This data exists in the Quantinuum mirror
benchmarking suite and metriq mirror circuit data.
    """)

    # ---- Analysis 4: Try to load mirror benchmarking data ----
    print("\n=== ANALYSIS 4: Mirror Benchmarking (same device, varying width) ===")

    metriq = load_metriq_data()
    if metriq:
        print(f"  Found {len(metriq)} mirror circuit results from metriq-data")
        # Group by device
        devices = {}
        for m in metriq:
            dev = m.get('device', 'unknown')
            if dev not in devices:
                devices[dev] = {'qubits': [], 'fidelity': []}
            devices[dev]['qubits'].append(m['qubits'])
            devices[dev]['fidelity'].append(m['fidelity'])

        for dev, data in devices.items():
            print(f"\n  Device: {dev}")
            for q, f in sorted(zip(data['qubits'], data['fidelity'])):
                error = 1.0 - f if f <= 1.0 else f
                print(f"    Width={q}: fidelity={f:.4f}, error={error:.4f}")
    else:
        print("  No mirror circuit data found in metriq-data")

    quantinuum = load_quantinuum_mirror_data()
    if quantinuum:
        print(f"\n  Found {len(quantinuum)} mirror benchmarking results from Quantinuum H2")
        for q in sorted(quantinuum, key=lambda x: x['qubits']):
            print(f"    N={q['qubits']}: fidelity={q['fidelity']:.4f}")
    else:
        print("  No structured mirror data found in Quantinuum repo (may need manual extraction)")


if __name__ == '__main__':
    main()
