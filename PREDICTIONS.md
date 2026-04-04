# Pre-registered Predictions: SFR-Inverted Branching Decay Model
## Date: 2026-04-02
## Author: Zhivko Dobromirski

---

## Model

Lambda(z) = Lambda_0 * [1 + delta * (1 - A_sfr(z))]

Where:
- delta = 0.06161 (fitted to DESI DR2 BAO, 2025)
- A_sfr(z) = cumulative fraction of cosmic star formation from Big Bang to redshift z
- SFR(z) follows Madau & Dickinson (2014): SFR = 0.015*(1+z)^2.7 / (1+((1+z)/2.9)^5.6) [Chabrier IMF]

Cosmological parameters (Planck 2018): H0=67.4 km/s/Mpc, Omega_m=0.315, Omega_r=9.1e-5, r_d=147.09 Mpc

## Physical interpretation

Lambda = remaining indeterminacy of the universe. Every quantum branching event (transition from superposition to determinacy) reduces Lambda. The rate of reduction follows the cosmic star formation rate as proxy for quantum event rate. Lambda was higher in the early universe (more indeterminacy) and decreases as branching accumulates.

## Fit to existing data

- DESI DR2 BAO (2025): chi2 = 11.19 (13 data points, 1 parameter)
- LCDM: chi2 = 23.78
- Delta AIC = +10.60 in favor of model ("strong evidence")

## Pre-registered predictions

These numbers are computed on 2026-04-02, BEFORE the release of DESI DR3, Euclid cosmological results, and Roman Space Telescope data.

### DH/rd predictions (DH = c / (H(z) * r_d))

| z | DH/rd (model) | DH/rd (LCDM) | Deviation from LCDM |
|---|--------------|--------------|---------------------|
| 0.150 | 27.5350 | 28.0269 | -1.755% |
| 0.295 | 25.4583 | 25.8425 | -1.486% |
| 0.380 | 24.2542 | 24.5837 | -1.340% |
| 0.510 | 22.4718 | 22.7301 | -1.137% |
| 0.590 | 21.4240 | 21.6457 | -1.024% |
| 0.706 | 19.9840 | 20.1611 | -0.878% |
| 0.850 | 18.3368 | 18.4703 | -0.723% |
| 0.934 | 17.4488 | 17.5621 | -0.645% |
| 1.100 | 15.8472 | 15.9291 | -0.514% |
| 1.321 | 14.0062 | 14.0595 | -0.379% |
| 1.484 | 12.8361 | 12.8751 | -0.303% |
| 1.750 | 11.2139 | 11.2376 | -0.211% |
| 2.000 | 9.9562 | 9.9712 | -0.150% |
| 2.330 | 8.6042 | 8.6126 | -0.097% |
| 2.500 | 8.0173 | 8.0236 | -0.078% |
| 3.000 | 6.6168 | 6.6197 | -0.043% |

### DM/rd predictions (DM = comoving distance / r_d)

| z | DM/rd (model) | DM/rd (LCDM) | Deviation from LCDM |
|---|--------------|--------------|---------------------|
| 0.150 | 4.2883 | 4.3714 | -1.902% |
| 0.295 | 8.1303 | 8.2768 | -1.769% |
| 0.380 | 10.2430 | 10.4197 | -1.696% |
| 0.510 | 13.2792 | 13.4940 | -1.591% |
| 0.590 | 15.0348 | 15.2687 | -1.532% |
| 0.706 | 17.4355 | 17.6924 | -1.452% |
| 0.850 | 20.1927 | 20.4719 | -1.364% |
| 0.934 | 21.6953 | 21.9848 | -1.317% |
| 1.100 | 24.4561 | 24.7617 | -1.234% |
| 1.321 | 27.7492 | 28.0694 | -1.141% |
| 1.484 | 29.9348 | 30.2626 | -1.083% |
| 1.750 | 33.1262 | 33.4621 | -1.004% |
| 2.000 | 35.7677 | 36.1084 | -0.943% |
| 2.330 | 38.8217 | 39.1661 | -0.879% |
| 2.500 | 40.2336 | 40.5792 | -0.852% |
| 3.000 | 43.8741 | 44.2219 | -0.786% |

### w(z) predictions (effective dark energy equation of state)

| z | w(z) model | w(z) LCDM |
|---|-----------|-----------|
| 0.20 | -1.0012 | -1.0000 |
| 0.40 | -1.0021 | -1.0000 |
| 0.60 | -1.0033 | -1.0000 |
| 0.80 | -1.0050 | -1.0000 |
| 1.00 | -1.0070 | -1.0000 |
| 1.20 | -1.0093 | -1.0000 |
| 1.50 | -1.0126 | -1.0000 |
| 2.00 | -1.0162 | -1.0000 |
| 2.50 | -1.0166 | -1.0000 |
| 3.00 | -1.0150 | -1.0000 |

### Key qualitative predictions

1. **DH/rd is systematically LOWER than LCDM at all redshifts** (H(z) is higher = faster expansion)
2. **The deviation is LARGEST at low z** (~1.8% at z=0.15) and **decreases monotonically** toward high z (~0.04% at z=3)
3. **w(z) is slightly phantom** (w < -1) at all redshifts, with maximum deviation at z~2-2.5
4. **w(z) profile has a specific shape**: rises from w~-1.001 at z=0.2 to w~-1.017 at z~2.5, then returns toward -1 at higher z. This shape is distinct from linear CPL (w0+wa*(1-a)) parameterization.

### Falsification criteria

The model is FALSIFIED if:
- DESI DR3 / Euclid measure DH/rd at z=0.5 HIGHER than LCDM prediction (our model requires it to be lower)
- w(z) is measured to be > -1 (quintessence-like) at any redshift — our model requires w < -1 everywhere
- The deviation profile is NOT monotonically decreasing with z — e.g., if the largest deviation is at z > 1.5
- Future measurements converge to LCDM (all deviations < 0.5%) with sufficient precision

### When to check

| Dataset | Expected | What to compare |
|---------|----------|----------------|
| DESI DR3 | 2026-2027 | DH/rd, DM/rd at same z bins with smaller errors |
| Euclid DR1 cosmology | 2026-2027 | BAO + weak lensing, new z coverage |
| Roman Space Telescope | 2028-2029 | SNe + BAO at z=1-3, high precision |
