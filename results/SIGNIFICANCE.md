# Paired significance: `brainmed-8b-v1__soup-last-a0.3` vs `base-HuatuoGPT-o1-8B`

McNemar's exact test on the items both runs answered. `only B` counts items the
candidate gets right and the baseline gets wrong; `only A` the reverse. Items both
get right or both get wrong are uninformative about which model is better.

| Benchmark | n | base-HuatuoGPT-o1-8B | brainmed-8b-v1__soup-last-a0.3 | Delta | only A | only B | p | verdict (a=0.05) |
|---|---|---|---|---|---|---|---|---|
| MB-op4 | 298 | 59.73 | 62.42 | +2.68 | 30 | 38 | 0.396 | no difference |
| MB-op5 | 298 | 55.37 | 53.36 | -2.01 | 37 | 31 | 0.545 | no difference |
| MedMCQA | 4183 | 63.28 | 64.04 | +0.77 | 358 | 390 | 0.257 | no difference |
| MedQA | 1273 | 77.93 | 76.59 | -1.34 | 103 | 86 | 0.244 | no difference |
| MedXpert | 1449 | 16.98 | 18.63 | +1.66 | 93 | 117 | 0.112 | no difference |
| PubMedQA | 1000 | 80.2 | 79.1 | -1.10 | 30 | 19 | 0.152 | no difference |

## Pooled across benchmarks

- items only `base-HuatuoGPT-o1-8B` gets right: **651**
- items only `brainmed-8b-v1__soup-last-a0.3` gets right: **681**
- McNemar exact p = **0.4269**

Pooled, the two runs are **statistically indistinguishable** (p = 0.427). Any average difference between them is within sampling noise.

Per benchmark: 0 significantly better, 0 significantly worse, 6 indistinguishable.

Pooling treats benchmarks as one sample and so weights them by size; read it
alongside the per-benchmark rows, not instead of them.
