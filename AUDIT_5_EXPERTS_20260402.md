# Five-Expert Full Project Audit — 2026-04-02

Critical review of the entire project by 5 independent AI experts:
astrophysicist, quantum foundations expert, statistician, programmer,
philosopher of science.

## Consensus blocking problems

1. **CMB constraint kills headline.** All 5 experts flagged: ΛCDM win
   evaporates once Planck θ_star is properly enforced. Current "z<10
   cutoff" is ad hoc epicycle.

2. **ΔAIC = +10.6 is systematically inflated.** Proper corrections
   (full DESI covariance, look-elsewhere for 8 model search, varying
   nuisance parameters) reduce to **~2-5**.

3. **κ signal is marginal, not strong.** With HKSJ random-effects
   correction and proper Sinha grouping: p ~ 0.02-0.05, not p=0.003.

4. **κ = δ² is numerology.** Three functional forms tested
   (linear/quadratic/sqrt), best reported. No derivation.

5. **"130σ above null" is strawman.** Real vs. experimental
   systematic-inclusive errors: ~2σ, not 130σ.

## Consensus strengths

1. **PREDICTIONS.md is the single most valuable artifact.**
   Time-stamped, falsifiable, specific. Will produce automatic verdict
   at DESI DR3.

2. **Self-critical robustness testing.** Published own κ downgrade,
   CMB failure, ISW insufficiency — unusual honesty.

3. **Abandonment of flawed simulation after 5-expert audit.** Shows
   ability to stop rather than accumulate complexity.

## Probability of coincidences under researcher DoF (statistician)

- Conservative: ~4% (1-in-25)
- Skeptical: ~20% (1-in-5)
- Generous: ~1% (1-in-100)

**Range "intriguing, keep watching" — not "extraordinary evidence".**

## Specific bugs found (programmer)

- Growth ODE: wrong initial conditions at a=1e-3 (radiation ~29%)
- Inconsistent κ_obs values across files (0.00442 vs 0.004 vs 0.0073)
- `except: pass` around optimizer calls hides failures
- No seeds in sorkin_scale_test.py
- Hard-coded absolute Windows paths in all plt.savefig

## Missing literature (QF expert)

- **Okon-Sudarsky (2014-2020)** — closest published work to thesis
- Perez-Sahlmann-Sudarsky — CMB from collapse
- Banks-Fischler — N-counting dark energy
- Kent (2015, 2017) — single-world interpretation

## Unanimous next-step recommendation

**WAIT. Don't touch PREDICTIONS.md. Find a physics collaborator.**

Every additional model, relation, or simulation from this point weakens
the project by adding degrees of freedom. The only action that preserves
value is waiting for DESI DR3.

## Verdict on project status

| Category | Status |
|----------|--------|
| Published/submitted | No |
| Peer-reviewable | No (would not pass PRL/JCAP/MNRAS) |
| Possibly publishable if fixed | Foundations of Physics, Entropy |
| Proto-science (Popperian sense) | Yes — form is correct |
| Pseudoscience patterns | Mostly absent |
| Wegener-comparable | Closer than Einstein-comparable |
| Most valuable output | PREDICTIONS.md time-stamp |
