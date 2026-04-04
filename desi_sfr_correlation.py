"""
Direct Correlation: DESI Dark Energy Deviation vs Cosmic SFR
=============================================================

Path 3 + Path 1: Overlay independently measured quantities:
1. DESI DR2 BAO deviations from LCDM at each redshift bin
2. Cosmic Star Formation Rate at the same redshifts

If they correlate — empirical support for branching-decay.
If not — model fails.

Author: Zhivko Dobromirski
Date: 2026-04-02
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# DESI DR2 BAO DATA (from Cobaya/DESI official release)
# arXiv:2503.14738
# ============================================================

# DH/rd = c/(H(z)*rd) measurements
z_DH =       np.array([0.510,   0.706,   0.934,   1.321,   1.484,   2.330])
DH_rd_meas = np.array([21.8629, 19.4553, 17.6415, 14.1760, 12.8170, 8.6315])
DH_rd_err =  np.array([0.4289,  0.3339,  0.2010,  0.2246,  0.5180,  0.1011])

# DM/rd measurements
DM_rd_meas = np.array([13.5876, 17.3507, 21.5756, 27.6009, 30.5119, 38.9890])
DM_rd_err =  np.array([0.1684,  0.1799,  0.1618,  0.3246,  0.7636,  0.5317])

# DV/rd (BGS only)
z_DV = np.array([0.295])
DV_rd_meas = np.array([7.9417])
DV_rd_err = np.array([0.0761])

# LCDM predictions (Planck 2018: H0=67.4, Om=0.315, rd=147.09 Mpc)
DH_rd_LCDM = np.array([22.7326, 20.1641, 17.5656, 14.0634, 12.8791, 8.6165])
DM_rd_LCDM = np.array([13.4946, 17.6937, 21.9868, 28.0729, 30.2666, 39.1735])
DV_rd_LCDM = np.array([8.0533])

# Deviations from LCDM (in sigma)
DH_deviation_sigma = (DH_rd_meas - DH_rd_LCDM) / DH_rd_err
DM_deviation_sigma = (DM_rd_meas - DM_rd_LCDM) / DM_rd_err

# Fractional deviations
DH_frac = (DH_rd_meas - DH_rd_LCDM) / DH_rd_LCDM
DM_frac = (DM_rd_meas - DM_rd_LCDM) / DM_rd_LCDM
DV_frac = (DV_rd_meas - DV_rd_LCDM) / DV_rd_LCDM

# Since DH = c/(H*rd), a LOWER DH means HIGHER H(z)
# So DH_frac < 0 means H(z) is higher than LCDM → more expansion → less dark energy?
# Actually: if Lambda was higher in the past, expansion was faster, H(z) higher, DH lower
# So negative DH_frac is CONSISTENT with higher Lambda in the past

# Combined deviation metric: use DH (more sensitive to dark energy)
# Negative DH_frac → higher H(z) → higher Lambda at that z

# ============================================================
# COSMIC STAR FORMATION RATE
# ============================================================

def sfrd_md14(z):
    """Madau & Dickinson 2014, Chabrier IMF (M_sun/yr/Mpc^3)"""
    return 0.015 * (1+z)**2.7 / (1 + ((1+z)/2.9)**5.6) * 0.63

# Measured IR data at DESI redshifts (Gruppioni+2013, Chabrier IMF)
z_sfr_obs =    np.array([0.30,   0.50,   0.70,   0.90,   1.30,   1.50,   2.30])
sfr_obs =      np.array([0.02395, 0.03015, 0.04564, 0.07233, 0.09106, 0.07751, 0.07751])

# SFR at DESI DH/DM redshifts (using MD14 fit)
sfr_at_desi = sfrd_md14(z_DH)

# ============================================================
# ANALYSIS
# ============================================================

print("=" * 70)
print("DIRECT CORRELATION: DESI Deviations vs Cosmic SFR")
print("=" * 70)

print(f"\n{'z':>6} | {'DH_frac%':>9} | {'DH_sigma':>9} | {'DM_frac%':>9} | {'DM_sigma':>9} | {'SFR':>8}")
print("-" * 65)
for i in range(len(z_DH)):
    print(f"{z_DH[i]:6.3f} | {DH_frac[i]*100:+9.3f} | {DH_deviation_sigma[i]:+9.2f} | "
          f"{DM_frac[i]*100:+9.3f} | {DM_deviation_sigma[i]:+9.2f} | {sfr_at_desi[i]:8.5f}")

# DH deviation vs SFR correlation
# Note: We expect NEGATIVE DH_frac where SFR is HIGH
# (more branching → less Lambda today → but MORE Lambda in the past → higher H → lower DH)
# Actually let's think: if Lambda was HIGHER at z~2 (more fuel), then at z~2:
# H(z) = sqrt(Om*(1+z)^3 + Lambda(z)) → higher Lambda(z) → higher H(z) → lower DH/rd
# So we expect: where SFR is high, DH_frac should be NEGATIVE (or more negative)

print(f"\n=== CORRELATION: DH fractional deviation vs SFR ===")
r_DH_SFR, p_DH_SFR = pearsonr(-DH_frac, sfr_at_desi)  # flip sign: positive = more deviation
rho_DH_SFR, prho_DH_SFR = spearmanr(-DH_frac, sfr_at_desi)
print(f"  Pearson r(-DH_frac, SFR) = {r_DH_SFR:.4f} (p = {p_DH_SFR:.4f})")
print(f"  Spearman rho = {rho_DH_SFR:.4f} (p = {prho_DH_SFR:.4f})")

print(f"\n=== CORRELATION: DM fractional deviation vs SFR ===")
r_DM_SFR, p_DM_SFR = pearsonr(-DM_frac, sfr_at_desi)
rho_DM_SFR, prho_DM_SFR = spearmanr(-DM_frac, sfr_at_desi)
print(f"  Pearson r(-DM_frac, SFR) = {r_DM_SFR:.4f} (p = {p_DM_SFR:.4f})")
print(f"  Spearman rho = {rho_DM_SFR:.4f} (p = {prho_DM_SFR:.4f})")

# Combined deviation: chi-like metric per bin
# delta_chi_i = sqrt((DH_dev_sigma)^2 + (DM_dev_sigma)^2) * sign
combined_dev = np.sqrt(DH_deviation_sigma**2 + DM_deviation_sigma**2)
# Give it the sign of DM deviation (negative = closer distances = higher Lambda in past)
combined_signed = combined_dev * np.sign(-DM_frac)

print(f"\n=== CORRELATION: Combined deviation vs SFR ===")
r_comb, p_comb = pearsonr(combined_signed, sfr_at_desi)
rho_comb, prho_comb = spearmanr(combined_signed, sfr_at_desi)
print(f"  Pearson r(combined, SFR) = {r_comb:.4f} (p = {p_comb:.4f})")
print(f"  Spearman rho = {rho_comb:.4f} (p = {prho_comb:.4f})")

# ============================================================
# PATH 1: Compute our model's DH/rd predictions
# ============================================================

print(f"\n{'='*70}")
print("MODEL PREDICTION: DH/rd from branching-decay Λ(a)")
print(f"{'='*70}")

# Our model: Lambda(a) = Lambda_0 * [1 + delta * (SFR(z(a))/SFR(0) - 1)]
# H^2(z) = H0^2 * [Om*(1+z)^3 + Or*(1+z)^4 + OL * Lambda_ratio(z)]

H0 = 67.4  # km/s/Mpc
Om = 0.315
Or = 9.1e-5
OL = 1 - Om - Or
rd = 147.09  # Mpc, sound horizon

delta_best = 0.1106  # from v2 SFR model

def H_model(z, delta):
    """H(z) for branching-decay model"""
    sfr_z = sfrd_md14(z)
    sfr_0 = sfrd_md14(0)
    lambda_ratio = 1.0 + delta * (sfr_z / sfr_0 - 1.0)
    return H0 * np.sqrt(Om * (1+z)**3 + Or * (1+z)**4 + OL * lambda_ratio)

def DH_model(z, delta):
    """DH/rd = c / (H(z) * rd)"""
    c = 299792.458  # km/s
    return c / (H_model(z, delta) * rd)

def comoving_distance(z, delta):
    """D_M = integral_0^z c/H(z') dz'"""
    from scipy.integrate import quad
    c = 299792.458
    def integrand(zp):
        return c / H_model(zp, delta)
    result, _ = quad(integrand, 0, z)
    return result

def DM_model(z, delta):
    """DM/rd"""
    return comoving_distance(z, delta) / rd

def DV_model(z, delta):
    """DV/rd = (z * DM^2 * DH)^(1/3) / rd"""
    dm = comoving_distance(z, delta)
    c = 299792.458
    dh = c / H_model(z, delta)
    return (z * dm**2 * dh)**(1.0/3.0) / rd

# Compute model predictions
print(f"\ndelta = {delta_best:.4f}")
print(f"\n{'z':>6} | {'DH_LCDM':>8} | {'DH_model':>8} | {'DH_meas':>8} | {'DM_LCDM':>8} | {'DM_model':>8} | {'DM_meas':>8}")
print("-" * 75)

model_DH = []
model_DM = []
for i, z in enumerate(z_DH):
    dh_m = DH_model(z, delta_best)
    dm_m = DM_model(z, delta_best)
    model_DH.append(dh_m)
    model_DM.append(dm_m)
    print(f"{z:6.3f} | {DH_rd_LCDM[i]:8.4f} | {dh_m:8.4f} | {DH_rd_meas[i]:8.4f} | "
          f"{DM_rd_LCDM[i]:8.4f} | {dm_m:8.4f} | {DM_rd_meas[i]:8.4f}")

model_DH = np.array(model_DH)
model_DM = np.array(model_DM)

# Chi-squared comparison
chi2_LCDM_DH = np.sum(((DH_rd_meas - DH_rd_LCDM) / DH_rd_err)**2)
chi2_model_DH = np.sum(((DH_rd_meas - model_DH) / DH_rd_err)**2)
chi2_LCDM_DM = np.sum(((DM_rd_meas - DM_rd_LCDM) / DM_rd_err)**2)
chi2_model_DM = np.sum(((DM_rd_meas - model_DM) / DM_rd_err)**2)

chi2_LCDM_total = chi2_LCDM_DH + chi2_LCDM_DM
chi2_model_total = chi2_model_DH + chi2_model_DM

# BGS
dv_m = DV_model(0.295, delta_best)
chi2_LCDM_DV = ((DV_rd_meas[0] - DV_rd_LCDM[0]) / DV_rd_err[0])**2
chi2_model_DV = ((DV_rd_meas[0] - dv_m) / DV_rd_err[0])**2

chi2_LCDM_all = chi2_LCDM_total + chi2_LCDM_DV
chi2_model_all = chi2_model_total + chi2_model_DV

n_data = 2 * len(z_DH) + 1  # 6 DH + 6 DM + 1 DV = 13
k_LCDM = 0  # no extra params
k_model = 1  # delta

AIC_LCDM = chi2_LCDM_all + 2 * k_LCDM
AIC_model = chi2_model_all + 2 * k_model
BIC_LCDM = chi2_LCDM_all + k_LCDM * np.log(n_data)
BIC_model = chi2_model_all + k_model * np.log(n_data)

print(f"\n{'='*70}")
print("CHI-SQUARED COMPARISON (direct fit to DESI BAO bins)")
print(f"{'='*70}")
print(f"\n  LCDM:            chi2 = {chi2_LCDM_all:.2f} (DH:{chi2_LCDM_DH:.2f} + DM:{chi2_LCDM_DM:.2f} + DV:{chi2_LCDM_DV:.2f})")
print(f"  Branching-decay:  chi2 = {chi2_model_all:.2f} (DH:{chi2_model_DH:.2f} + DM:{chi2_model_DM:.2f} + DV:{chi2_model_DV:.2f})")
print(f"\n  Delta chi2 = {chi2_LCDM_all - chi2_model_all:.2f} (positive = model better)")
print(f"\n  AIC: LCDM={AIC_LCDM:.2f}, Model={AIC_model:.2f}, Delta={AIC_LCDM - AIC_model:.2f}")
print(f"  BIC: LCDM={BIC_LCDM:.2f}, Model={BIC_model:.2f}, Delta={BIC_LCDM - BIC_model:.2f}")
print(f"\n  Interpretation:")
print(f"    Delta AIC > 10: strong evidence for preferred model")
print(f"    Delta AIC 4-7: moderate evidence")
print(f"    Delta AIC < 2: no significant preference")

# ============================================================
# VISUALIZATION
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# --- Plot 1: SFR(z) curve with DESI redshift bins marked ---
ax1 = axes[0, 0]
z_smooth = np.linspace(0, 4, 200)
sfr_smooth = sfrd_md14(z_smooth)
ax1.plot(z_smooth, sfr_smooth * 1000, 'r-', linewidth=2, label='MD14 fit')
ax1.scatter(z_sfr_obs, sfr_obs * 1000, c='red', s=60, zorder=5,
           edgecolors='black', label='IR observations')
for z in z_DH:
    ax1.axvline(x=z, color='blue', alpha=0.3, linestyle='--')
ax1.set_xlabel('Redshift z', fontsize=12)
ax1.set_ylabel('SFRD (10^-3 Msun/yr/Mpc^3)', fontsize=12)
ax1.set_title('Cosmic Star Formation Rate', fontsize=13)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.2)

# --- Plot 2: DH/rd deviations from LCDM ---
ax2 = axes[0, 1]
ax2.errorbar(z_DH, DH_frac*100, yerr=(DH_rd_err/DH_rd_LCDM)*100,
            fmt='bo', markersize=8, capsize=5, label='DESI DH deviation')
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
# Overlay SFR (normalized to same scale)
sfr_norm = sfr_at_desi / np.max(sfr_at_desi) * np.min(DH_frac*100) * 0.8
ax2.plot(z_DH, sfr_norm, 'r^-', markersize=8, alpha=0.7, label='SFR (scaled)')
ax2.set_xlabel('Redshift z', fontsize=12)
ax2.set_ylabel('DH/rd deviation from LCDM (%)', fontsize=12)
ax2.set_title('DESI DH Deviations + SFR overlay', fontsize=13)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.2)

# --- Plot 3: DH/rd — LCDM vs Model vs Measured ---
ax3 = axes[0, 2]
ax3.errorbar(z_DH, DH_rd_meas, yerr=DH_rd_err, fmt='ko', markersize=8,
            capsize=5, label='DESI measured', zorder=5)
ax3.plot(z_DH, DH_rd_LCDM, 'bs--', markersize=7, label='LCDM', linewidth=1.5)
ax3.plot(z_DH, model_DH, 'r^--', markersize=7, label=f'Branching (d={delta_best:.3f})', linewidth=1.5)
ax3.set_xlabel('Redshift z', fontsize=12)
ax3.set_ylabel('DH / rd', fontsize=12)
ax3.set_title('DH/rd: Data vs Models', fontsize=13)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.2)

# --- Plot 4: DM/rd — LCDM vs Model vs Measured ---
ax4 = axes[1, 0]
ax4.errorbar(z_DH, DM_rd_meas, yerr=DM_rd_err, fmt='ko', markersize=8,
            capsize=5, label='DESI measured', zorder=5)
ax4.plot(z_DH, DM_rd_LCDM, 'bs--', markersize=7, label='LCDM', linewidth=1.5)
ax4.plot(z_DH, model_DM, 'r^--', markersize=7, label=f'Branching (d={delta_best:.3f})', linewidth=1.5)
ax4.set_xlabel('Redshift z', fontsize=12)
ax4.set_ylabel('DM / rd', fontsize=12)
ax4.set_title('DM/rd: Data vs Models', fontsize=13)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.2)

# --- Plot 5: Residuals per bin (LCDM vs Model) ---
ax5 = axes[1, 1]
x_pos = np.arange(len(z_DH))
width = 0.35
res_LCDM_DH = (DH_rd_meas - DH_rd_LCDM) / DH_rd_err
res_model_DH = (DH_rd_meas - model_DH) / DH_rd_err
ax5.bar(x_pos - width/2, res_LCDM_DH, width, color='blue', alpha=0.6, label='LCDM residual')
ax5.bar(x_pos + width/2, res_model_DH, width, color='red', alpha=0.6, label='Branching residual')
ax5.set_xticks(x_pos)
ax5.set_xticklabels([f'z={z:.2f}' for z in z_DH], fontsize=8, rotation=30)
ax5.set_ylabel('Residual (sigma)', fontsize=12)
ax5.set_title('DH/rd Residuals by Bin', fontsize=13)
ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.2)

# --- Plot 6: Summary ---
ax6 = axes[1, 2]
ax6.axis('off')
summary = f"""RESULTS SUMMARY
{'='*45}

CORRELATION (Path 3):
  DH deviation vs SFR:
    Pearson r = {r_DH_SFR:.3f} (p = {p_DH_SFR:.3f})
    Spearman rho = {rho_DH_SFR:.3f} (p = {prho_DH_SFR:.3f})

  DM deviation vs SFR:
    Pearson r = {r_DM_SFR:.3f} (p = {p_DM_SFR:.3f})

CHI-SQUARED (Path 1):
  LCDM:     chi2 = {chi2_LCDM_all:.2f} ({n_data} bins, 0 params)
  Branching: chi2 = {chi2_model_all:.2f} ({n_data} bins, 1 param)
  Delta chi2 = {chi2_LCDM_all - chi2_model_all:+.2f}

MODEL SELECTION:
  Delta AIC = {AIC_LCDM - AIC_model:+.2f}
  Delta BIC = {BIC_LCDM - BIC_model:+.2f}

  AIC > 10: strong | 4-7: moderate | <2: none
"""
ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('c:/Projects/Multiverse-Evidence/desi_sfr_correlation.png', dpi=150)
print("\nPlot saved: desi_sfr_correlation.png")
