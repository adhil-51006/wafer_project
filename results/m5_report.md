# M5 retrieval comparison

Scores are mean ± sample standard deviation across 3 stratified splits. All methods used the same database and queries within each seed.

| Class | Queries/seed | Random | Baseline | Autoencoder | Note |
|---|---:|---:|---:|---:|---|
| none | 2000.0 | 0.852 ± 0.003 | 0.975 ± 0.002 | 0.986 ± 0.003 |  |
| Center | 859.0 | 0.025 ± 0.002 | 0.784 ± 0.014 | 0.803 ± 0.010 |  |
| Donut | 111.0 | 0.002 ± 0.002 | 0.704 ± 0.028 | 0.707 ± 0.018 |  |
| Edge-Ring | 1936.0 | 0.057 ± 0.003 | 0.909 ± 0.009 | 0.951 ± 0.003 |  |
| Edge-Loc | 1038.0 | 0.031 ± 0.001 | 0.377 ± 0.006 | 0.463 ± 0.012 |  |
| Loc | 718.0 | 0.021 ± 0.003 | 0.264 ± 0.008 | 0.303 ± 0.012 |  |
| Scratch | 239.0 | 0.009 ± 0.004 | 0.052 ± 0.015 | 0.121 ± 0.003 |  |
| Random | 173.0 | 0.008 ± 0.004 | 0.812 ± 0.006 | 0.826 ± 0.025 |  |
| Near-full | 30.0 | 0.000 ± 0.000 | 0.920 ± 0.024 | 0.831 ± 0.038 | few queries; noisy |
| macro_defects | — | 0.019 ± 0.001 | 0.603 ± 0.003 | 0.626 ± 0.003 |  |
| macro_all | — | 0.112 ± 0.001 | 0.644 ± 0.003 | 0.666 ± 0.003 |  |

The baseline has the higher mean on: Near-full.
The autoencoder has the higher mean on: none, Center, Donut, Edge-Ring, Edge-Loc, Loc, Scratch, Random.

Both learned and handcrafted retrieval beat random on every defect class.

Near-full is flagged as noisy because each split has fewer than 50 test queries. The confusion CSV files show the full distribution of retrieved labels for each query label.
