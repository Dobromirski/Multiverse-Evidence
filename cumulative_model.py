"""
Cumulative Branching Model: Branching RELEASES energy
=====================================================

Key insight: branching is like nuclear fission — it releases energy,
not consumes it. The released energy accumulates as vacuum energy.

Lambda(z) = Lambda_0 * [1 + delta * A(z)]
where A(z) = fraction of total branching accumulated by redshift z
A(z=inf) = 0, A(z=0) = 1

Author: Zhivko Dobromirski
Date: 2026-04-02
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
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
DV_LCDM = np.array([8.0533])

H0, Om, Or = 67.4, 0.315, 9.1e-5
OL = 1 - Om - Or
rd = 147.09
c = 299792.458

def sfrd(z):
    return 0.015 * (1+z)**2.7 / (1 + ((1+z)/2.9)**5.6) * 0.63

total_sfr, _ = quad(sfrd, 0, 20)

def accumulated_fraction(z):
    remaining, _ = quad(sfrd, z, 20)
    return 1.0 - remaining / total_sfr

# Precompute A(z) for speed
z_cache = np.linspace(0, 25, 2000)
A_cache = np.array([accumulated_fraction(z) for z in z_cache])

def A_fast(z):
    return np.interp(z, z_cache, A_cache)

# === CUMULATIVE MODEL ===
def H_cumul(z, delta):
    lam = 1.0 + delta * A_fast(z)
    return H0 * np.sqrt(Om*(1+z)**3 + Or*(1+z)**4 + OL * max(lam, 0.01))

def DH_c(z, delta):
    return c / (H_cumul(z, delta) * rd)

def DM_c(z, delta):
    r, _ = quad(lambda zp: c / H_cumul(zp, delta), 0, z)
    return r / rd

def DV_c(z, delta):
    dm = DM_c(z, delta) * rd
    dh = c / H_cumul(z, delta)
    return (z * dm**2 * dh)**(1.0/3.0) / rd

# === INSTANTANEOUS MODEL (for comparison) ===
def H_inst(z, delta):
    lam = 1.0 + delta * (sfrd(z)/sfrd(0) - 1.0)
    return H0 * np.sqrt(Om*(1+z)**3 + Or*(1+z)**4 + OL * max(lam, 0.01))

def DH_i(z, delta):
    return c / (H_inst(z, delta) * rd)

def DM_i(z, delta):
    r, _ = quad(lambda zp: c / H_inst(zp, delta), 0, z)
    return r / rd

def DV_i(z, delta):
    dm = DM_i(z, delta) * rd
    dh = c / H_inst(z, delta)
    return (z * dm**2 * dh)**(1.0/3.0) / rd

# === CHI2 FUNCTIONS ===
def chi2(DH_func, DM_func, DV_func, delta):
    s = 0
    for i in range(len(z_DH)):
        s += ((DH_meas[i] - DH_func(z_DH[i], delta)) / DH_err[i])**2
        s += ((DM_meas[i] - DM_func(z_DH[i], delta)) / DM_err[i])**2
    s += ((DV_meas[0] - DV_func(0.295, delta)) / DV_err[0])**2
    return s

chi2_lcdm = chi2(DH_c, DM_c, DV_c, 0.0)

# Optimize cumulative
print("Optimizing cumulative model...")
res_c = minimize_scalar(lambda d: chi2(DH_c, DM_c, DV_c, d),
                        bounds=(-0.15, 0.15), method='bounded')
dc, chi2c = res_c.x, res_c.fun

# Optimize instantaneous
print("Optimizing instantaneous model...")
res_i = minimize_scalar(lambda d: chi2(DH_i, DM_i, DV_i, d),
                        bounds=(-0.15, 0.15), method='bounded')
di, chi2i = res_i.x, res_i.fun

n = 13
print(f"\n{'='*70}")
print(f"RESULTS")
print(f"{'='*70}")
print(f"  LCDM:          chi2 = {chi2_lcdm:.2f}")
print(f"  Instantaneous:  chi2 = {chi2i:.2f}, delta = {di:.4f}, DAIC = {chi2_lcdm-chi2i-2:+.2f}, DBIC = {chi2_lcdm-chi2i-np.log(n):+.2f}")
print(f"  Cumulative:     chi2 = {chi2c:.2f}, delta = {dc:.4f}, DAIC = {chi2_lcdm-chi2c-2:+.2f}, DBIC = {chi2_lcdm-chi2c-np.log(n):+.2f}")

# Per-bin
print(f"\n--- Per-bin (cumulative, delta={dc:.4f}) ---")
print(f"{'z':>6} | {'DH_L':>8} | {'DH_C':>8} | {'DH_D':>8} | {'res_L':>7} | {'res_C':>7} | {'A(z)':>6}")
print("-" * 65)
mDH, mDM = [], []
for i in range(len(z_DH)):
    dh = DH_c(z_DH[i], dc)
    dm = DM_c(z_DH[i], dc)
    mDH.append(dh)
    mDM.append(dm)
    A = A_fast(z_DH[i])
    rL = (DH_meas[i] - DH_LCDM[i]) / DH_err[i]
    rC = (DH_meas[i] - dh) / DH_err[i]
    print(f"{z_DH[i]:6.3f} | {DH_LCDM[i]:8.4f} | {dh:8.4f} | {DH_meas[i]:8.4f} | {rL:+7.2f} | {rC:+7.2f} | {A:6.3f}")
mDH = np.array(mDH)
mDM = np.array(mDM)

# Physical interpretation
print(f"\n--- Physical interpretation ---")
print(f"  delta = {dc:.4f}")
if dc > 0:
    print(f"  Lambda GROWS over time (branching ADDS energy)")
    print(f"  Lambda today is {dc*100:.1f}% higher than at z=infinity")
    print(f"  Consistent with branching-as-fission hypothesis")
else:
    print(f"  Lambda SHRINKS over time (branching CONSUMES energy)")
    print(f"  Original consumption model preferred")

# === VISUALIZATION ===
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Plot 1: Lambda(z)
ax = axes[0, 0]
zp = np.linspace(0, 3, 200)
lam_c = [1.0 + dc * A_fast(z) for z in zp]
lam_i = [1.0 + di * (sfrd(z)/sfrd(0) - 1.0) for z in zp]
ax.plot(zp, lam_c, 'r-', linewidth=2.5, label=f'Cumulative (d={dc:.4f})')
ax.plot(zp, lam_i, 'b--', linewidth=2, label=f'Instantaneous (d={di:.4f})')
ax.axhline(y=1.0, color='black', linestyle=':', alpha=0.5, label='LCDM')
ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel('Lambda(z) / Lambda_0', fontsize=12)
ax.set_title('Vacuum Energy: Two Models', fontsize=14)
ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

# Plot 2: Accumulated fraction vs SFR rate
ax = axes[0, 1]
A_plot = [A_fast(z) for z in zp]
sfr_norm = [sfrd(z)/max(sfrd(zp_) for zp_ in zp) for z in zp]
ax.plot(zp, A_plot, 'r-', linewidth=2.5, label='Accumulated A(z)')
ax.plot(zp, sfr_norm, 'g--', linewidth=2, label='SFR rate (normalized)')
ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel('Fraction', fontsize=12)
ax.set_title('Accumulation vs Rate', fontsize=14)
ax.legend(fontsize=10); ax.grid(True, alpha=0.2)

# Plot 3: DH/rd
ax = axes[0, 2]
ax.errorbar(z_DH, DH_meas, yerr=DH_err, fmt='ko', ms=8, capsize=5, label='DESI', zorder=5)
ax.plot(z_DH, DH_LCDM, 'bs--', ms=7, label='LCDM', lw=1.5)
ax.plot(z_DH, mDH, 'r^-', ms=8, label=f'Cumulative', lw=2)
ax.set_xlabel('Redshift z', fontsize=12); ax.set_ylabel('DH / rd', fontsize=12)
ax.set_title('DH/rd: Data vs Models', fontsize=14)
ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

# Plot 4: DM/rd
ax = axes[1, 0]
ax.errorbar(z_DH, DM_meas, yerr=DM_err, fmt='ko', ms=8, capsize=5, label='DESI', zorder=5)
ax.plot(z_DH, DM_LCDM, 'bs--', ms=7, label='LCDM', lw=1.5)
ax.plot(z_DH, mDM, 'r^-', ms=8, label='Cumulative', lw=2)
ax.set_xlabel('Redshift z', fontsize=12); ax.set_ylabel('DM / rd', fontsize=12)
ax.set_title('DM/rd: Data vs Models', fontsize=14)
ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

# Plot 5: DH residuals
ax = axes[1, 1]
x = np.arange(len(z_DH))
w = 0.35
rL = (DH_meas - DH_LCDM) / DH_err
rC = (DH_meas - mDH) / DH_err
ax.bar(x - w/2, rL, w, color='blue', alpha=0.6, label='LCDM')
ax.bar(x + w/2, rC, w, color='red', alpha=0.6, label='Cumulative')
ax.set_xticks(x); ax.set_xticklabels([f'z={z:.2f}' for z in z_DH], fontsize=8, rotation=30)
ax.set_ylabel('DH Residual (sigma)', fontsize=12)
ax.set_title('DH Residuals per Bin', fontsize=14)
ax.axhline(y=0, color='black', alpha=0.3); ax.legend(fontsize=10); ax.grid(True, alpha=0.2)

# Plot 6: Chi2 scan
ax = axes[1, 2]
ds = np.linspace(-0.12, 0.12, 49)
c2c = [chi2(DH_c, DM_c, DV_c, d) for d in ds]
c2i = [chi2(DH_i, DM_i, DV_i, d) for d in ds]
ax.plot(ds, c2c, 'r-', lw=2, label='Cumulative')
ax.plot(ds, c2i, 'b--', lw=2, label='Instantaneous')
ax.axhline(y=chi2_lcdm, color='black', linestyle=':', label=f'LCDM ({chi2_lcdm:.1f})')
ax.axvline(x=dc, color='red', linestyle=':', alpha=0.5)
ax.axvline(x=0, color='black', linestyle=':', alpha=0.3)
ax.set_xlabel('delta', fontsize=12); ax.set_ylabel('Chi-squared', fontsize=12)
ax.set_title('Chi2 Landscape', fontsize=14)
ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
ax.set_ylim(10, 80)

fig.text(0.5, 0.01,
    f'LCDM: chi2={chi2_lcdm:.1f} | Cumulative (d={dc:.4f}): chi2={chi2c:.1f}, DAIC={chi2_lcdm-chi2c-2:+.1f} | Instantaneous (d={di:.4f}): chi2={chi2i:.1f}, DAIC={chi2_lcdm-chi2i-2:+.1f}',
    ha='center', fontsize=11, style='italic',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig('c:/Projects/Multiverse-Evidence/cumulative_branching_model.png', dpi=150)
print("\nPlot saved: cumulative_branching_model.png")
