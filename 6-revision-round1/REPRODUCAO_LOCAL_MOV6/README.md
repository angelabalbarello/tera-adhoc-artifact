# Local reproduction package: Movement 6, Table 13 (Attention and Alert rows)

Status: **local package, not published.** It is not part of the public artifact
(`tera-adhoc-artifact`, tag `resubmission-round1` = `5b472ce0`). Do not cite it
as publicly available until it has actually been released.

## What it reproduces

The Attention and Alert rows of Table 13 / Fig. 11 of the Ad Hoc Networks
manuscript (`artigo.tex` SHA-256/16 `176c888b41b6a88c`):

| Row | Mean max_prob | SD (ddof=1) | Fraction ≥ 0.10 | n |
|---|---|---|---|---|
| Attention (A01–A03) | 0.048 | 0.195 | 5.8 % | 240 |
| Alert (L01–L03) | 0.380 | 0.416 | 48.8 % | 240 |

Seed 42, θ = 0.10, 80 episodes per recipe, SENSOR_LEVEL = 4, generator v7.

The Normal and Critical rows are **not** reproduced here. They come from the
factorial test partition of seed 42 (`exp_20260519_055950`) under the
prefix-window protocol, and are covered by the public artifact
(`tab13_fr_reconciliation_seed42.csv`).

## Which model produced the Attention and Alert rows

`modelos_salvos/student_seed42.pt` (SHA-256 `3be34ff08dec0240…`, full hash in
`MANIFEST.sha256`).

What is established from the files:
- It is byte-identical to the student checkpoint of the TERA pipeline run
  `results_bkp/exp_20260517_184912` (pipeline v1.0, `config_hash f91287c4970a02bf`).
- It is not the published factorial student (`f88581a2…`) and not the
  `run_v29` replication checkpoint (`0331c607…`).

What is **not** established:
- The training configuration for `config_hash f91287c4…` was not found among
  the archived `experiment_config.yaml` files. **The training objective of this
  checkpoint cannot be verified from the archive.** Identical hashes prove which
  file was used, not which loss trained it.

Seeds 43–45 are included because they belong to the same run and are used for
the checkpoint-sensitivity statement (L03 = 0.267 / 0.143 / 0.228). Table 13 uses
seed 42 only.

## Contents

```
modelos_salvos/student_seed{42,43,44,45}.pt   frozen checkpoints (inference only)
src/borderline_analysis_gpu_v23.py            original analysis script (unchanged)
src/synthetic_driver_risk_v7.py               generator v7, SHA-256/16 65bbec4a23050639
scripts/rescore_mov6.py                        inference-only driver (no training)
scripts/compare_with_expected.py               comparison + Table 13 summaries
expected/borderline_episode_level.csv          archived episode-level output, seeds 42–45
expected/borderline_summary.csv                archived summary
MANIFEST.sha256                                SHA-256 of every file above
```

`src/` and `modelos_salvos/` are verbatim copies of
`FGCS/Arquitetura/Boderline/` (see `MANIFEST.sha256`).

## How to run

Requirements: Python ≥ 3.10, `torch`, `numpy`, `pandas`, `scikit-learn`,
`matplotlib`. CPU is enough; no GPU is needed.

```
sha256sum -c MANIFEST.sha256
python scripts/rescore_mov6.py 42
python scripts/compare_with_expected.py out/rescored_episode_level_42.csv
```

Expected output: 480 episodes compared; `max |delta max_prob| = 0`; Attention
0.048 / 0.195 / 5.8 %; Alert 0.380 / 0.416 / 48.8 %.

`rescore_mov6.py` calls the functions of the original script
(`build_dataset`, `infer_episodes`, `analyse_borderline`). It does not run the
script's training branch.

## Verified execution

On 2026-10-19 this package was run in a Linux environment (Python 3.13,
torch 2.14.1+cpu, numpy 2.5.3, pandas 3.0.5) with seed 42. Result: 480
episodes identical to `expected/borderline_episode_level.csv` (max difference
0 at the stored 4-decimal precision); summaries as in the table above.
An earlier run on unrounded probabilities reproduced seeds 42 and 43 with a
maximum difference of 3·10⁻⁸.

## Generator version

All results here use generator v7 (`65bbec4a`), the version used for the
manuscript. A counterfactual with generator v7.1 (K1; only recipe L02 changes)
gives Alert 0.378 ± 0.426 (42.1 %). This is reported separately in the decision
package and is **not** a replacement for the Table 13 row.
