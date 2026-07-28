# Comparison against the published 7B-8B models

Our rows are **measured by this pipeline**. Every other row is **transcribed from**
MedReason (arXiv:2504.00993) and was not re-run.

> Excluded from this table: MedReason-8B.

> Backbone shown with its published values (Huatuo-o1-RL-8B), not the score measured on this harness.

> Harness offset measured on the untouched backbone: +2.89 pt vs the published protocol. Subtract it before reading our-vs-published gaps.

## Table A — Ranking on the paper's six-benchmark suite

| # | Model | MB-op4 | MB-op5 | MedXpert | MedQA | MedMCQA | PubMedQA | Avg |
|---|---|---|---|---|---|---|---|---|
| **1** | **brainmed-8b-final (ours)** | **62.01** | **54.87** | **18.63** | **76.67** | **64.07** | **79.10** | **59.23** |
| **2** | **brainmed-8b-v1__soup-last-a0.3 (ours)** | **62.01** | **52.92** | **18.63** | **76.67** | **64.07** | **79.10** | **58.90** |
| **3** | **brainmed-8b-v1__soup-last-a0.7 (ours)** | **60.71** | **56.49** | **19.32** | **76.12** | **62.87** | **77.90** | **58.90** |
| **4** | **brainmed-8b-v1__soup-last-a0.5 (ours)** | **58.77** | **54.55** | **18.91** | **76.90** | **64.36** | **78.90** | **58.73** |
| **5** | **brainmed-8b-v1__soup-a0.3 (ours)** | **59.42** | **54.87** | **16.43** | **76.28** | **62.54** | **80.00** | **58.26** |
| **6** | **brainmed-8b-v1__soup-a0.15 (ours)** | **60.06** | **50.97** | **16.08** | **77.85** | **63.21** | **81.20** | **58.23** |
| **7** | **brainmed-8b-v1__epoch-3 (ours)** | **58.77** | **56.82** | **18.08** | **72.11** | **61.20** | **77.10** | **57.35** |
| **8** | **brainmed-8b-v1 (ours)** | **56.49** | **50.65** | **17.81** | **72.58** | **60.22** | **77.70** | **55.91** |
| **9** | **brainmed-8b-v1__best (ours)** | **56.49** | **50.65** | **17.81** | **72.58** | **60.22** | **77.70** | **55.91** |
| 10 | Huatuo-o1-RL-8B | 55.20 | 51.30 | 16.70 | 72.60 | 60.40 | 79.20 | 55.90 |
| **11** | **brainmed-8b-v1__epoch-1 (ours)** | **55.52** | **50.00** | **15.87** | **70.62** | **59.77** | **77.50** | **54.88** |
| 12 | Huatuo-o1-SFT-8B | 53.30 | 49.70 | 17.30 | 70.20 | 58.20 | 76.10 | 54.10 |
| 13 | Qwen2.5-Instruct-7B | 50.00 | 41.60 | 12.60 | 57.00 | 55.60 | 72.70 | 48.20 |
| 14 | Llama3.1-Instruct-8B | 43.20 | 40.90 | 14.30 | 58.70 | 56.00 | 75.20 | 48.00 |
| 15 | OpenBioLLM-8B | 39.20 | 35.70 | 10.70 | 57.70 | 54.10 | 74.10 | 45.30 |
| 16 | DeepSeek-Distill-8B | 41.90 | 35.10 | 13.50 | 55.40 | 49.00 | 73.90 | 44.80 |
| 17 | Medical-CoT-8B | 39.30 | 34.10 | 12.60 | 49.00 | 42.60 | 68.00 | 40.90 |
| 18 | BioMistral-7B | 46.40 | 33.10 | 12.40 | 45.00 | 40.20 | 66.90 | 40.70 |
| 19 | Mistral-Instruct-7B | 43.50 | 33.40 | 11.40 | 48.20 | 44.90 | 50.10 | 38.60 |
| 20 | Medical-Llama3-8B | 33.40 | 25.30 | 9.00 | 40.30 | 46.80 | 48.00 | 33.80 |

## Table B — `brainmed-8b-final` vs published **Huatuo-o1-RL-8B**

| Benchmark | Huatuo-o1-RL-8B
(published) | brainmed-8b-final
(measured) | Delta |  |
|---|---|---|---|---|
| MB-op4 | 55.20 | 62.01 | +6.81 | win |
| MB-op5 | 51.30 | 54.87 | +3.57 | win |
| MedXpert | 16.70 | 18.63 | +1.93 | win |
| MedQA | 72.60 | 76.67 | +4.07 | win |
| MedMCQA | 60.40 | 64.07 | +3.67 | win |
| PubMedQA | 79.20 | 79.10 | -0.10 |  |
| **Average** | **55.90** | **59.23** | **+3.33** | **win** |

## Table B — `brainmed-8b-v1` vs published **Huatuo-o1-RL-8B**

| Benchmark | Huatuo-o1-RL-8B
(published) | brainmed-8b-v1
(measured) | Delta |  |
|---|---|---|---|---|
| MB-op4 | 55.20 | 56.49 | +1.29 | win |
| MB-op5 | 51.30 | 50.65 | -0.65 |  |
| MedXpert | 16.70 | 17.81 | +1.11 | win |
| MedQA | 72.60 | 72.58 | -0.02 |  |
| MedMCQA | 60.40 | 60.22 | -0.18 |  |
| PubMedQA | 79.20 | 77.70 | -1.50 |  |
| **Average** | **55.90** | **55.91** | **+0.01** | **win** |

## Table B — `brainmed-8b-v1__best` vs published **Huatuo-o1-RL-8B**

| Benchmark | Huatuo-o1-RL-8B
(published) | brainmed-8b-v1__best
(measured) | Delta |  |
|---|---|---|---|---|
| MB-op4 | 55.20 | 56.49 | +1.29 | win |
| MB-op5 | 51.30 | 50.65 | -0.65 |  |
| MedXpert | 16.70 | 17.81 | +1.11 | win |
| MedQA | 72.60 | 72.58 | -0.02 |  |
| MedMCQA | 60.40 | 60.22 | -0.18 |  |
| PubMedQA | 79.20 | 77.70 | -1.50 |  |
| **Average** | **55.90** | **55.91** | **+0.01** | **win** |

## Table B — `brainmed-8b-v1__epoch-1` vs published **Huatuo-o1-RL-8B**

| Benchmark | Huatuo-o1-RL-8B
(published) | brainmed-8b-v1__epoch-1
(measured) | Delta |  |
|---|---|---|---|---|
| MB-op4 | 55.20 | 55.52 | +0.32 | win |
| MB-op5 | 51.30 | 50.00 | -1.30 |  |
| MedXpert | 16.70 | 15.87 | -0.83 |  |
| MedQA | 72.60 | 70.62 | -1.98 |  |
| MedMCQA | 60.40 | 59.77 | -0.63 |  |
| PubMedQA | 79.20 | 77.50 | -1.70 |  |
| **Average** | **55.90** | **54.88** | **-1.02** |  |

## Table B — `brainmed-8b-v1__epoch-3` vs published **Huatuo-o1-RL-8B**

| Benchmark | Huatuo-o1-RL-8B
(published) | brainmed-8b-v1__epoch-3
(measured) | Delta |  |
|---|---|---|---|---|
| MB-op4 | 55.20 | 58.77 | +3.57 | win |
| MB-op5 | 51.30 | 56.82 | +5.52 | win |
| MedXpert | 16.70 | 18.08 | +1.38 | win |
| MedQA | 72.60 | 72.11 | -0.49 |  |
| MedMCQA | 60.40 | 61.20 | +0.80 | win |
| PubMedQA | 79.20 | 77.10 | -2.10 |  |
| **Average** | **55.90** | **57.35** | **+1.45** | **win** |

## Table B — `brainmed-8b-v1__soup-a0.15` vs published **Huatuo-o1-RL-8B**

| Benchmark | Huatuo-o1-RL-8B
(published) | brainmed-8b-v1__soup-a0.15
(measured) | Delta |  |
|---|---|---|---|---|
| MB-op4 | 55.20 | 60.06 | +4.86 | win |
| MB-op5 | 51.30 | 50.97 | -0.33 |  |
| MedXpert | 16.70 | 16.08 | -0.62 |  |
| MedQA | 72.60 | 77.85 | +5.25 | win |
| MedMCQA | 60.40 | 63.21 | +2.81 | win |
| PubMedQA | 79.20 | 81.20 | +2.00 | win |
| **Average** | **55.90** | **58.23** | **+2.33** | **win** |

## Table B — `brainmed-8b-v1__soup-a0.3` vs published **Huatuo-o1-RL-8B**

| Benchmark | Huatuo-o1-RL-8B
(published) | brainmed-8b-v1__soup-a0.3
(measured) | Delta |  |
|---|---|---|---|---|
| MB-op4 | 55.20 | 59.42 | +4.22 | win |
| MB-op5 | 51.30 | 54.87 | +3.57 | win |
| MedXpert | 16.70 | 16.43 | -0.27 |  |
| MedQA | 72.60 | 76.28 | +3.68 | win |
| MedMCQA | 60.40 | 62.54 | +2.14 | win |
| PubMedQA | 79.20 | 80.00 | +0.80 | win |
| **Average** | **55.90** | **58.26** | **+2.36** | **win** |

## Table B — `brainmed-8b-v1__soup-last-a0.3` vs published **Huatuo-o1-RL-8B**

| Benchmark | Huatuo-o1-RL-8B
(published) | brainmed-8b-v1__soup-last-a0.3
(measured) | Delta |  |
|---|---|---|---|---|
| MB-op4 | 55.20 | 62.01 | +6.81 | win |
| MB-op5 | 51.30 | 52.92 | +1.62 | win |
| MedXpert | 16.70 | 18.63 | +1.93 | win |
| MedQA | 72.60 | 76.67 | +4.07 | win |
| MedMCQA | 60.40 | 64.07 | +3.67 | win |
| PubMedQA | 79.20 | 79.10 | -0.10 |  |
| **Average** | **55.90** | **58.90** | **+3.00** | **win** |

## Table B — `brainmed-8b-v1__soup-last-a0.5` vs published **Huatuo-o1-RL-8B**

| Benchmark | Huatuo-o1-RL-8B
(published) | brainmed-8b-v1__soup-last-a0.5
(measured) | Delta |  |
|---|---|---|---|---|
| MB-op4 | 55.20 | 58.77 | +3.57 | win |
| MB-op5 | 51.30 | 54.55 | +3.25 | win |
| MedXpert | 16.70 | 18.91 | +2.21 | win |
| MedQA | 72.60 | 76.90 | +4.30 | win |
| MedMCQA | 60.40 | 64.36 | +3.96 | win |
| PubMedQA | 79.20 | 78.90 | -0.30 |  |
| **Average** | **55.90** | **58.73** | **+2.83** | **win** |

## Table B — `brainmed-8b-v1__soup-last-a0.7` vs published **Huatuo-o1-RL-8B**

| Benchmark | Huatuo-o1-RL-8B
(published) | brainmed-8b-v1__soup-last-a0.7
(measured) | Delta |  |
|---|---|---|---|---|
| MB-op4 | 55.20 | 60.71 | +5.51 | win |
| MB-op5 | 51.30 | 56.49 | +5.19 | win |
| MedXpert | 16.70 | 19.32 | +2.62 | win |
| MedQA | 72.60 | 76.12 | +3.52 | win |
| MedMCQA | 60.40 | 62.87 | +2.47 | win |
| PubMedQA | 79.20 | 77.90 | -1.30 |  |
| **Average** | **55.90** | **58.90** | **+3.00** | **win** |

## Table D — Eight-benchmark suite (MedReason Table 2 columns)

| Model / data | MedQA | MedMCQA | PubMedQA | MMLU-Pro | MB-op4 | MB-op5 | MedXpert | HLE(med) | Avg |
|---|---|---|---|---|---|---|---|---|---|
| Llama3.1-Instruct-8B (base) | 58.70 | 56.00 | 75.20 | 58.20 | 48.70 | 42.50 | 13.20 | 13.60 | 45.80 |
| Llama3.1-Instruct-8B + Huatuo CoT | 70.20 | 58.20 | 76.10 | 59.90 | 53.30 | 49.70 | 17.30 | 14.60 | 49.90 |
| Llama3.1-Instruct-8B + MedReason | 68.40 | 57.50 | 77.60 | 63.10 | 57.50 | 52.30 | 16.40 | 16.50 | 51.20 |
| Mistral-Instruct-7B (base) | 48.20 | 44.90 | 50.10 | 42.70 | 43.50 | 33.40 | 11.40 | 14.60 | 36.10 |
| Mistral-Instruct-7B + Huatuo CoT | 59.90 | 46.90 | 57.50 | 47.60 | 50.00 | 46.10 | 14.40 | 14.60 | 42.10 |
| Mistral-Instruct-7B + MedReason | 58.70 | 48.90 | 59.20 | 50.80 | 52.30 | 47.10 | 16.60 | 24.30 | 44.70 |
| **brainmed-8b-final (ours)** | **76.67** | **64.07** | **79.10** | **65.73** | **62.01** | **54.87** | **18.63** | **10.68** | **53.97** |
| **brainmed-8b-v1 (ours)** | **72.58** | **60.22** | **77.70** | **61.82** | **56.49** | **50.65** | **17.81** | **20.39** | **52.21** |
