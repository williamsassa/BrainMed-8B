# BrainMed reasoning SFT - evaluation report

Rows marked **(ours)** were measured by this pipeline (greedy decoding, strict prompt,
MedReason scorer with `<answer>` support, training system prompt applied at inference).
All other rows are transcribed from MedReason (arXiv:2504.00993, Table 4) and were
**not** re-run.

## Comparison with published 7B-8B models (MedReason Table 4)

| Model | MB-op4 | MB-op5 | MedXpert | MedQA | MedMCQA | PubMedQA | Avg |
|---|---|---|---|---|---|---|---|
| Llama3.1-Instruct-8B | 43.2 | 40.9 | 14.3 | 58.7 | 56.0 | 75.2 | **48.0** |
| Qwen2.5-Instruct-7B | 50.0 | 41.6 | 12.6 | 57.0 | 55.6 | 72.7 | **48.2** |
| Mistral-Instruct-7B | 43.5 | 33.4 | 11.4 | 48.2 | 44.9 | 50.1 | **38.6** |
| Medical-Llama3-8B | 33.4 | 25.3 | 9.0 | 40.3 | 46.8 | 48.0 | **33.8** |
| OpenBioLLM-8B | 39.2 | 35.7 | 10.7 | 57.7 | 54.1 | 74.1 | **45.3** |
| BioMistral-7B | 46.4 | 33.1 | 12.4 | 45.0 | 40.2 | 66.9 | **40.7** |
| Medical-CoT-8B | 39.3 | 34.1 | 12.6 | 49.0 | 42.6 | 68.0 | **40.9** |
| DeepSeek-Distill-8B | 41.9 | 35.1 | 13.5 | 55.4 | 49.0 | 73.9 | **44.8** |
| Huatuo-o1-SFT-8B | 53.3 | 49.7 | 17.3 | 70.2 | 58.2 | 76.1 | **54.1** |
| Huatuo-o1-RL-8B | 55.2 | 51.3 | 16.7 | 72.6 | 60.4 | 79.2 | **55.9** |
| MedReason-8B | 57.5 | 55.5 | 19.0 | 71.8 | 60.7 | 79.4 | **57.3** |
| **base-HuatuoGPT-o1-8B (ours)** | 59.09 | 55.19 | 17.05 | 77.93 | 63.3 | 80.2 | **58.79** |
| **brainmed-8b-final (ours)** | 62.01 | 54.87 | 18.63 | 76.67 | 64.07 | 79.1 | **59.23** |
| **brainmed-8b-v1 (ours)** | 56.49 | 50.65 | 17.81 | 72.58 | 60.22 | 77.7 | **55.91** |
| **brainmed-8b-v1__best (ours)** | 56.49 | 50.65 | 17.81 | 72.58 | 60.22 | 77.7 | **55.91** |
| **brainmed-8b-v1__epoch-1 (ours)** | 55.52 | 50.0 | 15.87 | 70.62 | 59.77 | 77.5 | **54.88** |
| **brainmed-8b-v1__epoch-2 (ours)** | 56.49 | 50.65 | 17.81 | 72.58 | 60.22 | 77.7 | **55.91** |
| **brainmed-8b-v1__epoch-3 (ours)** | 58.77 | 56.82 | 18.08 | 72.11 | 61.2 | 77.1 | **57.35** |
| **brainmed-8b-v1__last (ours)** | 58.77 | 56.82 | 18.08 | 72.11 | 61.2 | 77.1 | **57.35** |
| **brainmed-8b-v1__soup-a0.15 (ours)** | 60.06 | 50.97 | 16.08 | 77.85 | 63.21 | 81.2 | **58.23** |
| **brainmed-8b-v1__soup-a0.3 (ours)** | 59.42 | 54.87 | 16.43 | 76.28 | 62.54 | 80.0 | **58.26** |
| **brainmed-8b-v1__soup-last-a0.3 (ours)** | 62.01 | 52.92 | 18.63 | 76.67 | 64.07 | 79.1 | **58.9** |
| **brainmed-8b-v1__soup-last-a0.5 (ours)** | 58.77 | 54.55 | 18.91 | 76.9 | 64.36 | 78.9 | **58.73** |
| **brainmed-8b-v1__soup-last-a0.7 (ours)** | 60.71 | 56.49 | 19.32 | 76.12 | 62.87 | 77.9 | **58.9** |


## Table 2 style - effect of the training data (MedReason Table 2)

Published rows compare the *same backbone* trained on Huatuo CoT vs MedReason.
Our row adds the union corpus. Only rows sharing a backbone are comparable:
read the Llama3.1 block against a Llama3.1 run, not against a Huatuo-o1 run.

| Model / data | MedQA | MedMCQA | PubMedQA | MMLU-Pro | MB-op4 | MB-op5 | MedXpert | HLE(med) | Avg |
|---|---|---|---|---|---|---|---|---|---|
| Llama3.1-Instruct-8B (base) | 58.7 | 56.0 | 75.2 | 58.2 | 48.7 | 42.5 | 13.2 | 13.6 | **45.8** |
| Llama3.1-Instruct-8B + Huatuo CoT | 70.2 | 58.2 | 76.1 | 59.9 | 53.3 | 49.7 | 17.3 | 14.6 | **49.9** |
| Llama3.1-Instruct-8B + MedReason | 68.4 | 57.5 | 77.6 | 63.1 | 57.5 | 52.3 | 16.4 | 16.5 | **51.2** |
| Mistral-Instruct-7B (base) | 48.2 | 44.9 | 50.1 | 42.7 | 43.5 | 33.4 | 11.4 | 14.6 | **36.1** |
| Mistral-Instruct-7B + Huatuo CoT | 59.9 | 46.9 | 57.5 | 47.6 | 50.0 | 46.1 | 14.4 | 14.6 | **42.1** |
| Mistral-Instruct-7B + MedReason | 58.7 | 48.9 | 59.2 | 50.8 | 52.3 | 47.1 | 16.6 | 24.3 | **44.7** |
| **base-HuatuoGPT-o1-8B + ours (union) (ours)** | 77.93 | 63.3 | 80.2 | 64.56 | 59.09 | 55.19 | 17.05 | 11.65 | **53.62** |
| **brainmed-8b-final + ours (union) (ours)** | 76.67 | 64.07 | 79.1 | 65.73 | 62.01 | 54.87 | 18.63 | 10.68 | **53.97** |
| **brainmed-8b-v1 + ours (union) (ours)** | 72.58 | 60.22 | 77.7 | 61.82 | 56.49 | 50.65 | 17.81 | 20.39 | **52.21** |

## Table 3 style - common vs challenging averages (MedReason Table 3)

| Model / data | Avg common (4) | Avg challenging (4) | Avg overall (8) |
|---|---|---|---|
| Medical-CoT-8B (base) | 52.1 | 25.4 | 38.75 |
| Medical-CoT-8B + MedReason | 57.4 | 30.6 | 44.0 |
| DeepSeek-Distill-8B (base) | 58.0 | 25.5 | 41.75 |
| DeepSeek-Distill-8B + MedReason | 61.5 | 33.3 | 47.4 |
| **base-HuatuoGPT-o1-8B (ours)** | 71.5 | 35.75 | 53.62 |
| **brainmed-8b-final (ours)** | 71.39 | 36.55 | 53.97 |
| **brainmed-8b-v1 (ours)** | 68.08 | 36.34 | 52.21 |

## HuatuoGPT-o1 Table 1/2 style - core medical benchmarks

`MMLU-Pro (Med)` and `GPQA (Med)` are the merged medical tracks shipped in
`eval_data.json`; Huatuo's Table 1 splits them per track, which the released files
do not allow us to reproduce, so this uses their Table 2 granularity.

| Model | MedQA | MedMCQA | PubMedQA | MMLU-Pro | GPQA(med) | Avg |
|---|---|---|---|---|---|---|
| LLaMA-3.1-8B-Instruct (base) | 58.7 | 56.0 | 75.2 | 58.2 | 44.1 | **58.44** |
| SFT w/ original exam data | 60.0 | 55.5 | 74.1 | 54.3 | 46.9 | **58.16** |
| SFT w/o CoT | 65.2 | 58.1 | 75.4 | 58.5 | 48.7 | **61.18** |
| SFT w/ Simple CoT | 66.6 | 59.2 | 75.4 | 57.0 | 46.7 | **60.98** |
| SFT w/ Complex CoT | 69.0 | 57.9 | 77.7 | 59.4 | 51.0 | **63.0** |
| SFT w/o CoT + RL (PPO) | 66.4 | 58.6 | 76.3 | 60.1 | 49.8 | **62.24** |
| SFT w/ Simple CoT + RL (PPO) | 68.7 | 58.4 | 77.5 | 60.2 | 53.1 | **63.58** |
| SFT w/ Complex CoT + RL (PPO) = o1-8B | 72.6 | 60.4 | 79.2 | 63.1 | 57.5 | **66.56** |
| SFT w/ Complex CoT + RL (DPO) | 72.2 | 58.4 | 77.3 | 60.4 | 52.5 | **64.16** |
| SFT w/ Complex CoT + RL (RLOO) | 71.1 | 60.1 | 78.1 | 60.9 | 58.2 | **65.68** |
| **base-HuatuoGPT-o1-8B (ours)** | 77.93 | 63.3 | 80.2 | 64.56 | 49.74 | **67.15** |
| **brainmed-8b-final (ours)** | 76.67 | 64.07 | 79.1 | 65.73 | 59.74 | **69.06** |
| **brainmed-8b-v1 (ours)** | 72.58 | 60.22 | 77.7 | 61.82 | 49.23 | **64.31** |

## Table 5 style - pipeline ablations

MedReason's Table 5 ablates quality filtering. The equivalent knobs here are
decontamination (`--no_decontaminate`) and MCQ answer-format alignment
(`--no_answer_alignment`). Each additional arm costs one more training run - the
table below fills in from whatever runs exist.

| Run | MedQA | MedMCQA | PubMedQA | MMLU-Pro | MB-op4 | MB-op5 | MedXpert | HLE(med) | Avg (8) | Avg (Table 4, 6) |
|---|---|---|---|---|---|---|---|---|---|---|
| brainmed-8b-final | 76.67 | 64.07 | 79.1 | 65.73 | 62.01 | 54.87 | 18.63 | 10.68 | 53.97 | 59.23 |
| brainmed-8b-v1 | 72.58 | 60.22 | 77.7 | 61.82 | 56.49 | 50.65 | 17.81 | 20.39 | 52.21 | 55.91 |
| brainmed-8b-v1__best | 72.58 | 60.22 | 77.7 |  | 56.49 | 50.65 | 17.81 |  | - | 55.91 |
| brainmed-8b-v1__epoch-1 | 70.62 | 59.77 | 77.5 |  | 55.52 | 50.0 | 15.87 |  | - | 54.88 |
| brainmed-8b-v1__epoch-2 | 72.58 | 60.22 | 77.7 |  | 56.49 | 50.65 | 17.81 |  | - | 55.91 |
| brainmed-8b-v1__epoch-3 | 72.11 | 61.2 | 77.1 |  | 58.77 | 56.82 | 18.08 |  | - | 57.35 |
| brainmed-8b-v1__last | 72.11 | 61.2 | 77.1 |  | 58.77 | 56.82 | 18.08 |  | - | 57.35 |
| brainmed-8b-v1__soup-a0.15 | 77.85 | 63.21 | 81.2 |  | 60.06 | 50.97 | 16.08 |  | - | 58.23 |
| brainmed-8b-v1__soup-a0.3 | 76.28 | 62.54 | 80.0 |  | 59.42 | 54.87 | 16.43 |  | - | 58.26 |
| brainmed-8b-v1__soup-last-a0.3 | 76.67 | 64.07 | 79.1 |  | 62.01 | 52.92 | 18.63 |  | - | 58.9 |
| brainmed-8b-v1__soup-last-a0.5 | 76.9 | 64.36 | 78.9 |  | 58.77 | 54.55 | 18.91 |  | - | 58.73 |
| brainmed-8b-v1__soup-last-a0.7 | 76.12 | 62.87 | 77.9 |  | 60.71 | 56.49 | 19.32 |  | - | 58.9 |

The two averages cover different column sets and must never be compared to each other; `Avg (Table 4, 6)` is the one the published tables use.

## All evaluated benchmarks

Every benchmark scored in this run (10 of them). The paper's averages
cover only the eight it reports; the extras below are measured and kept for the
record but excluded from those averages.

| Benchmark | n | In paper avg | base-HuatuoGPT-o1-8B | brainmed-8b-final | brainmed-8b-v1 | brainmed-8b-v1__best | brainmed-8b-v1__epoch-1 | brainmed-8b-v1__epoch-2 | brainmed-8b-v1__epoch-3 | brainmed-8b-v1__last | brainmed-8b-v1__soup-a0.15 | brainmed-8b-v1__soup-a0.3 | brainmed-8b-v1__soup-last-a0.3 | brainmed-8b-v1__soup-last-a0.5 | brainmed-8b-v1__soup-last-a0.7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MedQA | 1273 | yes | 77.93 | 76.67 | 72.58 | 72.58 | 70.62 | 72.58 | 72.11 | 72.11 | 77.85 | 76.28 | 76.67 | 76.9 | 76.12 |
| MedMCQA | 4183 | yes | 63.3 | 64.07 | 60.22 | 60.22 | 59.77 | 60.22 | 61.2 | 61.2 | 63.21 | 62.54 | 64.07 | 64.36 | 62.87 |
| PubMedQA | 1000 | yes | 80.2 | 79.1 | 77.7 | 77.7 | 77.5 | 77.7 | 77.1 | 77.1 | 81.2 | 80.0 | 79.1 | 78.9 | 77.9 |
| MMLU-Pro | 1535 | yes | 64.56 | 65.73 | 61.82 | - | - | - | - | - | - | - | - | - | - |
| MB-op4 | 308 | yes | 59.09 | 62.01 | 56.49 | 56.49 | 55.52 | 56.49 | 58.77 | 58.77 | 60.06 | 59.42 | 62.01 | 58.77 | 60.71 |
| MB-op5 | 308 | yes | 55.19 | 54.87 | 50.65 | 50.65 | 50.0 | 50.65 | 56.82 | 56.82 | 50.97 | 54.87 | 52.92 | 54.55 | 56.49 |
| MedXpert | 1449 | yes | 17.05 | 18.63 | 17.81 | 17.81 | 15.87 | 17.81 | 18.08 | 18.08 | 16.08 | 16.43 | 18.63 | 18.91 | 19.32 |
| HLE(med) | 103 | yes | 11.65 | 10.68 | 20.39 | - | - | - | - | - | - | - | - | - | - |
| GPQA(med) | 390 | extra | 49.74 | 59.74 | 49.23 | - | - | - | - | - | - | - | - | - | - |
| MedQA-5opt | 1273 | extra | 72.82 | 73.68 | 68.81 | - | - | - | - | - | - | - | - | - | - |

## Paper-comparable suite (8 benchmarks)

| Run | MedQA | MedMCQA | PubMedQA | MMLU-Pro | Avg common | MB-op4 | MB-op5 | MedXpert | HLE(med) | Avg challenging | Avg overall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base-HuatuoGPT-o1-8B | 77.93 | 63.3 | 80.2 | 64.56 | 71.5 | 59.09 | 55.19 | 17.05 | 11.65 | 35.75 | 53.62 |
| brainmed-8b-final | 76.67 | 64.07 | 79.1 | 65.73 | 71.39 | 62.01 | 54.87 | 18.63 | 10.68 | 36.55 | 53.97 |
| brainmed-8b-v1 | 72.58 | 60.22 | 77.7 | 61.82 | 68.08 | 56.49 | 50.65 | 17.81 | 20.39 | 36.34 | 52.21 |
| brainmed-8b-v1__best | 72.58 | 60.22 | 77.7 |  | - | 56.49 | 50.65 | 17.81 |  | - | - |
| brainmed-8b-v1__epoch-1 | 70.62 | 59.77 | 77.5 |  | - | 55.52 | 50.0 | 15.87 |  | - | - |
| brainmed-8b-v1__epoch-2 | 72.58 | 60.22 | 77.7 |  | - | 56.49 | 50.65 | 17.81 |  | - | - |
| brainmed-8b-v1__epoch-3 | 72.11 | 61.2 | 77.1 |  | - | 58.77 | 56.82 | 18.08 |  | - | - |
| brainmed-8b-v1__last | 72.11 | 61.2 | 77.1 |  | - | 58.77 | 56.82 | 18.08 |  | - | - |
| brainmed-8b-v1__soup-a0.15 | 77.85 | 63.21 | 81.2 |  | - | 60.06 | 50.97 | 16.08 |  | - | - |
| brainmed-8b-v1__soup-a0.3 | 76.28 | 62.54 | 80.0 |  | - | 59.42 | 54.87 | 16.43 |  | - | - |
| brainmed-8b-v1__soup-last-a0.3 | 76.67 | 64.07 | 79.1 |  | - | 62.01 | 52.92 | 18.63 |  | - | - |
| brainmed-8b-v1__soup-last-a0.5 | 76.9 | 64.36 | 78.9 |  | - | 58.77 | 54.55 | 18.91 |  | - | - |
| brainmed-8b-v1__soup-last-a0.7 | 76.12 | 62.87 | 77.9 |  | - | 60.71 | 56.49 | 19.32 |  | - | - |

`-` marks an average that cannot be computed: brainmed-8b-v1__best, brainmed-8b-v1__epoch-1, brainmed-8b-v1__epoch-2, brainmed-8b-v1__epoch-3, brainmed-8b-v1__last, brainmed-8b-v1__soup-a0.15, brainmed-8b-v1__soup-a0.3, brainmed-8b-v1__soup-last-a0.3, brainmed-8b-v1__soup-last-a0.5, brainmed-8b-v1__soup-last-a0.7 were scored on a subset of the suite (QUICK sweep). Re-run the full bundle before quoting an 8-benchmark average for them.

## Win conditions

Two things must both hold: the model beats the backbone it started from, and it
beats every published 7B-8B model in MedReason Table 4. The backbone is compared
on numbers measured here; the others on their published rows.

| Run | Condition | Reference | Ours | Delta | Verdict |
|---|---|---|---|---|---|
| brainmed-8b-final | beats backbone (base-HuatuoGPT-o1-8B, measured) | 58.79 | 59.23 | +0.44 | PASS |
| brainmed-8b-final | beats published best (MedReason-8B) | 57.3 | 59.23 | +1.93 | PASS |
| brainmed-8b-final | beats all 11 published models | - | 59.23 | 11/11 | PASS |
| brainmed-8b-v1 | beats backbone (base-HuatuoGPT-o1-8B, measured) | 58.79 | 55.91 | -2.88 | **FAIL** |
| brainmed-8b-v1 | beats published best (MedReason-8B) | 57.3 | 55.91 | -1.39 | **FAIL** |
| brainmed-8b-v1 | beats all 11 published models | - | 55.91 | 10/11 | **FAIL** (MedReason-8B) |
| brainmed-8b-v1__best | beats backbone (base-HuatuoGPT-o1-8B, measured) | 58.79 | 55.91 | -2.88 | **FAIL** |
| brainmed-8b-v1__best | beats published best (MedReason-8B) | 57.3 | 55.91 | -1.39 | **FAIL** |
| brainmed-8b-v1__best | beats all 11 published models | - | 55.91 | 10/11 | **FAIL** (MedReason-8B) |
| brainmed-8b-v1__epoch-1 | beats backbone (base-HuatuoGPT-o1-8B, measured) | 58.79 | 54.88 | -3.91 | **FAIL** |
| brainmed-8b-v1__epoch-1 | beats published best (MedReason-8B) | 57.3 | 54.88 | -2.42 | **FAIL** |
| brainmed-8b-v1__epoch-1 | beats all 11 published models | - | 54.88 | 9/11 | **FAIL** (Huatuo-o1-RL-8B, MedReason-8B) |
| brainmed-8b-v1__epoch-2 | beats backbone (base-HuatuoGPT-o1-8B, measured) | 58.79 | 55.91 | -2.88 | **FAIL** |
| brainmed-8b-v1__epoch-2 | beats published best (MedReason-8B) | 57.3 | 55.91 | -1.39 | **FAIL** |
| brainmed-8b-v1__epoch-2 | beats all 11 published models | - | 55.91 | 10/11 | **FAIL** (MedReason-8B) |
| brainmed-8b-v1__epoch-3 | beats backbone (base-HuatuoGPT-o1-8B, measured) | 58.79 | 57.35 | -1.44 | **FAIL** |
| brainmed-8b-v1__epoch-3 | beats published best (MedReason-8B) | 57.3 | 57.35 | +0.05 | PASS |
| brainmed-8b-v1__epoch-3 | beats all 11 published models | - | 57.35 | 11/11 | PASS |
| brainmed-8b-v1__last | beats backbone (base-HuatuoGPT-o1-8B, measured) | 58.79 | 57.35 | -1.44 | **FAIL** |
| brainmed-8b-v1__last | beats published best (MedReason-8B) | 57.3 | 57.35 | +0.05 | PASS |
| brainmed-8b-v1__last | beats all 11 published models | - | 57.35 | 11/11 | PASS |
| brainmed-8b-v1__soup-a0.15 | beats backbone (base-HuatuoGPT-o1-8B, measured) | 58.79 | 58.23 | -0.56 | **FAIL** |
| brainmed-8b-v1__soup-a0.15 | beats published best (MedReason-8B) | 57.3 | 58.23 | +0.93 | PASS |
| brainmed-8b-v1__soup-a0.15 | beats all 11 published models | - | 58.23 | 11/11 | PASS |
| brainmed-8b-v1__soup-a0.3 | beats backbone (base-HuatuoGPT-o1-8B, measured) | 58.79 | 58.26 | -0.53 | **FAIL** |
| brainmed-8b-v1__soup-a0.3 | beats published best (MedReason-8B) | 57.3 | 58.26 | +0.96 | PASS |
| brainmed-8b-v1__soup-a0.3 | beats all 11 published models | - | 58.26 | 11/11 | PASS |
| brainmed-8b-v1__soup-last-a0.3 | beats backbone (base-HuatuoGPT-o1-8B, measured) | 58.79 | 58.9 | +0.11 | PASS |
| brainmed-8b-v1__soup-last-a0.3 | beats published best (MedReason-8B) | 57.3 | 58.9 | +1.60 | PASS |
| brainmed-8b-v1__soup-last-a0.3 | beats all 11 published models | - | 58.9 | 11/11 | PASS |
| brainmed-8b-v1__soup-last-a0.5 | beats backbone (base-HuatuoGPT-o1-8B, measured) | 58.79 | 58.73 | -0.06 | **FAIL** |
| brainmed-8b-v1__soup-last-a0.5 | beats published best (MedReason-8B) | 57.3 | 58.73 | +1.43 | PASS |
| brainmed-8b-v1__soup-last-a0.5 | beats all 11 published models | - | 58.73 | 11/11 | PASS |
| brainmed-8b-v1__soup-last-a0.7 | beats backbone (base-HuatuoGPT-o1-8B, measured) | 58.79 | 58.9 | +0.11 | PASS |
| brainmed-8b-v1__soup-last-a0.7 | beats published best (MedReason-8B) | 57.3 | 58.9 | +1.60 | PASS |
| brainmed-8b-v1__soup-last-a0.7 | beats all 11 published models | - | 58.9 | 11/11 | PASS |

## Controlled before/after - same harness (base-HuatuoGPT-o1-8B -> brainmed-8b-final)

| Benchmark | Before | After | Delta | Gate |
|---|---|---|---|---|
| MedQA | 77.93 | 76.67 | -1.26 | **REGRESSION** |
| MedMCQA | 63.3 | 64.07 | +0.77 | OK |
| PubMedQA | 80.2 | 79.1 | -1.10 | **REGRESSION** |
| MMLU-Pro | 64.56 | 65.73 | +1.17 | OK |
| MB-op4 | 59.09 | 62.01 | +2.92 | OK |
| MB-op5 | 55.19 | 54.87 | -0.32 | OK |
| MedXpert | 17.05 | 18.63 | +1.58 | OK |
| HLE(med) | 11.65 | 10.68 | -0.97 | **REGRESSION** |
| **Average** | 53.62 | 53.97 | **+0.35** |  |

**GATE FAILED** - 3 benchmark(s) below the -0.5 pt tolerance: MedQA -1.26, PubMedQA -1.10, HLE(med) -0.97.

The fine-tune traded existing strengths for its gains. Before shipping, either lower the learning rate / shorten the schedule, or interpolate towards the backbone with `scripts/weight_soup.py` and re-evaluate.

## Controlled before/after - same harness (base-HuatuoGPT-o1-8B -> brainmed-8b-v1)

| Benchmark | Before | After | Delta | Gate |
|---|---|---|---|---|
| MedQA | 77.93 | 72.58 | -5.35 | **REGRESSION** |
| MedMCQA | 63.3 | 60.22 | -3.08 | **REGRESSION** |
| PubMedQA | 80.2 | 77.7 | -2.50 | **REGRESSION** |
| MMLU-Pro | 64.56 | 61.82 | -2.74 | **REGRESSION** |
| MB-op4 | 59.09 | 56.49 | -2.60 | **REGRESSION** |
| MB-op5 | 55.19 | 50.65 | -4.54 | **REGRESSION** |
| MedXpert | 17.05 | 17.81 | +0.76 | OK |
| HLE(med) | 11.65 | 20.39 | +8.74 | OK |
| **Average** | 53.62 | 52.21 | **-1.41** |  |

**GATE FAILED** - 6 benchmark(s) below the -0.5 pt tolerance: MedQA -5.35, MedMCQA -3.08, PubMedQA -2.50, MMLU-Pro -2.74, MB-op4 -2.60, MB-op5 -4.54.

The fine-tune traded existing strengths for its gains. Before shipping, either lower the learning rate / shorten the schedule, or interpolate towards the backbone with `scripts/weight_soup.py` and re-evaluate.

## Controlled before/after - same harness (base-HuatuoGPT-o1-8B -> brainmed-8b-v1__best)

| Benchmark | Before | After | Delta | Gate |
|---|---|---|---|---|
| MedQA | 77.93 | 72.58 | -5.35 | **REGRESSION** |
| MedMCQA | 63.3 | 60.22 | -3.08 | **REGRESSION** |
| PubMedQA | 80.2 | 77.7 | -2.50 | **REGRESSION** |
| MB-op4 | 59.09 | 56.49 | -2.60 | **REGRESSION** |
| MB-op5 | 55.19 | 50.65 | -4.54 | **REGRESSION** |
| MedXpert | 17.05 | 17.81 | +0.76 | OK |
| **Average** | 58.79 | 55.91 | **-2.88** |  |

**GATE FAILED** - 5 benchmark(s) below the -0.5 pt tolerance: MedQA -5.35, MedMCQA -3.08, PubMedQA -2.50, MB-op4 -2.60, MB-op5 -4.54.

The fine-tune traded existing strengths for its gains. Before shipping, either lower the learning rate / shorten the schedule, or interpolate towards the backbone with `scripts/weight_soup.py` and re-evaluate.

## Controlled before/after - same harness (base-HuatuoGPT-o1-8B -> brainmed-8b-v1__epoch-1)

| Benchmark | Before | After | Delta | Gate |
|---|---|---|---|---|
| MedQA | 77.93 | 70.62 | -7.31 | **REGRESSION** |
| MedMCQA | 63.3 | 59.77 | -3.53 | **REGRESSION** |
| PubMedQA | 80.2 | 77.5 | -2.70 | **REGRESSION** |
| MB-op4 | 59.09 | 55.52 | -3.57 | **REGRESSION** |
| MB-op5 | 55.19 | 50.0 | -5.19 | **REGRESSION** |
| MedXpert | 17.05 | 15.87 | -1.18 | **REGRESSION** |
| **Average** | 58.79 | 54.88 | **-3.91** |  |

**GATE FAILED** - 6 benchmark(s) below the -0.5 pt tolerance: MedQA -7.31, MedMCQA -3.53, PubMedQA -2.70, MB-op4 -3.57, MB-op5 -5.19, MedXpert -1.18.

The fine-tune traded existing strengths for its gains. Before shipping, either lower the learning rate / shorten the schedule, or interpolate towards the backbone with `scripts/weight_soup.py` and re-evaluate.

## Controlled before/after - same harness (base-HuatuoGPT-o1-8B -> brainmed-8b-v1__epoch-2)

| Benchmark | Before | After | Delta | Gate |
|---|---|---|---|---|
| MedQA | 77.93 | 72.58 | -5.35 | **REGRESSION** |
| MedMCQA | 63.3 | 60.22 | -3.08 | **REGRESSION** |
| PubMedQA | 80.2 | 77.7 | -2.50 | **REGRESSION** |
| MB-op4 | 59.09 | 56.49 | -2.60 | **REGRESSION** |
| MB-op5 | 55.19 | 50.65 | -4.54 | **REGRESSION** |
| MedXpert | 17.05 | 17.81 | +0.76 | OK |
| **Average** | 58.79 | 55.91 | **-2.88** |  |

**GATE FAILED** - 5 benchmark(s) below the -0.5 pt tolerance: MedQA -5.35, MedMCQA -3.08, PubMedQA -2.50, MB-op4 -2.60, MB-op5 -4.54.

The fine-tune traded existing strengths for its gains. Before shipping, either lower the learning rate / shorten the schedule, or interpolate towards the backbone with `scripts/weight_soup.py` and re-evaluate.

## Controlled before/after - same harness (base-HuatuoGPT-o1-8B -> brainmed-8b-v1__epoch-3)

| Benchmark | Before | After | Delta | Gate |
|---|---|---|---|---|
| MedQA | 77.93 | 72.11 | -5.82 | **REGRESSION** |
| MedMCQA | 63.3 | 61.2 | -2.10 | **REGRESSION** |
| PubMedQA | 80.2 | 77.1 | -3.10 | **REGRESSION** |
| MB-op4 | 59.09 | 58.77 | -0.32 | OK |
| MB-op5 | 55.19 | 56.82 | +1.63 | OK |
| MedXpert | 17.05 | 18.08 | +1.03 | OK |
| **Average** | 58.79 | 57.35 | **-1.45** |  |

**GATE FAILED** - 3 benchmark(s) below the -0.5 pt tolerance: MedQA -5.82, MedMCQA -2.10, PubMedQA -3.10.

The fine-tune traded existing strengths for its gains. Before shipping, either lower the learning rate / shorten the schedule, or interpolate towards the backbone with `scripts/weight_soup.py` and re-evaluate.

## Controlled before/after - same harness (base-HuatuoGPT-o1-8B -> brainmed-8b-v1__last)

| Benchmark | Before | After | Delta | Gate |
|---|---|---|---|---|
| MedQA | 77.93 | 72.11 | -5.82 | **REGRESSION** |
| MedMCQA | 63.3 | 61.2 | -2.10 | **REGRESSION** |
| PubMedQA | 80.2 | 77.1 | -3.10 | **REGRESSION** |
| MB-op4 | 59.09 | 58.77 | -0.32 | OK |
| MB-op5 | 55.19 | 56.82 | +1.63 | OK |
| MedXpert | 17.05 | 18.08 | +1.03 | OK |
| **Average** | 58.79 | 57.35 | **-1.45** |  |

**GATE FAILED** - 3 benchmark(s) below the -0.5 pt tolerance: MedQA -5.82, MedMCQA -2.10, PubMedQA -3.10.

The fine-tune traded existing strengths for its gains. Before shipping, either lower the learning rate / shorten the schedule, or interpolate towards the backbone with `scripts/weight_soup.py` and re-evaluate.

## Controlled before/after - same harness (base-HuatuoGPT-o1-8B -> brainmed-8b-v1__soup-a0.15)

| Benchmark | Before | After | Delta | Gate |
|---|---|---|---|---|
| MedQA | 77.93 | 77.85 | -0.08 | OK |
| MedMCQA | 63.3 | 63.21 | -0.09 | OK |
| PubMedQA | 80.2 | 81.2 | +1.00 | OK |
| MB-op4 | 59.09 | 60.06 | +0.97 | OK |
| MB-op5 | 55.19 | 50.97 | -4.22 | **REGRESSION** |
| MedXpert | 17.05 | 16.08 | -0.97 | **REGRESSION** |
| **Average** | 58.79 | 58.23 | **-0.56** |  |

**GATE FAILED** - 2 benchmark(s) below the -0.5 pt tolerance: MB-op5 -4.22, MedXpert -0.97.

The fine-tune traded existing strengths for its gains. Before shipping, either lower the learning rate / shorten the schedule, or interpolate towards the backbone with `scripts/weight_soup.py` and re-evaluate.

## Controlled before/after - same harness (base-HuatuoGPT-o1-8B -> brainmed-8b-v1__soup-a0.3)

| Benchmark | Before | After | Delta | Gate |
|---|---|---|---|---|
| MedQA | 77.93 | 76.28 | -1.65 | **REGRESSION** |
| MedMCQA | 63.3 | 62.54 | -0.76 | **REGRESSION** |
| PubMedQA | 80.2 | 80.0 | -0.20 | OK |
| MB-op4 | 59.09 | 59.42 | +0.33 | OK |
| MB-op5 | 55.19 | 54.87 | -0.32 | OK |
| MedXpert | 17.05 | 16.43 | -0.62 | **REGRESSION** |
| **Average** | 58.79 | 58.26 | **-0.54** |  |

**GATE FAILED** - 3 benchmark(s) below the -0.5 pt tolerance: MedQA -1.65, MedMCQA -0.76, MedXpert -0.62.

The fine-tune traded existing strengths for its gains. Before shipping, either lower the learning rate / shorten the schedule, or interpolate towards the backbone with `scripts/weight_soup.py` and re-evaluate.

## Controlled before/after - same harness (base-HuatuoGPT-o1-8B -> brainmed-8b-v1__soup-last-a0.3)

| Benchmark | Before | After | Delta | Gate |
|---|---|---|---|---|
| MedQA | 77.93 | 76.67 | -1.26 | **REGRESSION** |
| MedMCQA | 63.3 | 64.07 | +0.77 | OK |
| PubMedQA | 80.2 | 79.1 | -1.10 | **REGRESSION** |
| MB-op4 | 59.09 | 62.01 | +2.92 | OK |
| MB-op5 | 55.19 | 52.92 | -2.27 | **REGRESSION** |
| MedXpert | 17.05 | 18.63 | +1.58 | OK |
| **Average** | 58.79 | 58.9 | **+0.11** |  |

**GATE FAILED** - 3 benchmark(s) below the -0.5 pt tolerance: MedQA -1.26, PubMedQA -1.10, MB-op5 -2.27.

The fine-tune traded existing strengths for its gains. Before shipping, either lower the learning rate / shorten the schedule, or interpolate towards the backbone with `scripts/weight_soup.py` and re-evaluate.

## Controlled before/after - same harness (base-HuatuoGPT-o1-8B -> brainmed-8b-v1__soup-last-a0.5)

| Benchmark | Before | After | Delta | Gate |
|---|---|---|---|---|
| MedQA | 77.93 | 76.9 | -1.03 | **REGRESSION** |
| MedMCQA | 63.3 | 64.36 | +1.06 | OK |
| PubMedQA | 80.2 | 78.9 | -1.30 | **REGRESSION** |
| MB-op4 | 59.09 | 58.77 | -0.32 | OK |
| MB-op5 | 55.19 | 54.55 | -0.64 | **REGRESSION** |
| MedXpert | 17.05 | 18.91 | +1.86 | OK |
| **Average** | 58.79 | 58.73 | **-0.06** |  |

**GATE FAILED** - 3 benchmark(s) below the -0.5 pt tolerance: MedQA -1.03, PubMedQA -1.30, MB-op5 -0.64.

The fine-tune traded existing strengths for its gains. Before shipping, either lower the learning rate / shorten the schedule, or interpolate towards the backbone with `scripts/weight_soup.py` and re-evaluate.

## Controlled before/after - same harness (base-HuatuoGPT-o1-8B -> brainmed-8b-v1__soup-last-a0.7)

| Benchmark | Before | After | Delta | Gate |
|---|---|---|---|---|
| MedQA | 77.93 | 76.12 | -1.81 | **REGRESSION** |
| MedMCQA | 63.3 | 62.87 | -0.43 | OK |
| PubMedQA | 80.2 | 77.9 | -2.30 | **REGRESSION** |
| MB-op4 | 59.09 | 60.71 | +1.62 | OK |
| MB-op5 | 55.19 | 56.49 | +1.30 | OK |
| MedXpert | 17.05 | 19.32 | +2.27 | OK |
| **Average** | 58.79 | 58.9 | **+0.11** |  |

**GATE FAILED** - 2 benchmark(s) below the -0.5 pt tolerance: MedQA -1.81, PubMedQA -2.30.

The fine-tune traded existing strengths for its gains. Before shipping, either lower the learning rate / shorten the schedule, or interpolate towards the backbone with `scripts/weight_soup.py` and re-evaluate.

## Harness calibration - backbone measured here vs its published row

Backbone `base-HuatuoGPT-o1-8B` against the published **Huatuo-o1-RL-8B**. Small gaps are
expected; a large one means the harness, not the training, is moving the
numbers, and every delta downstream has to be read with that in mind.

| Benchmark | Published | Measured here | Gap |
|---|---|---|---|
| MB-op4 | 55.2 | 59.09 | +3.89 |
| MB-op5 | 51.3 | 55.19 | +3.89 |
| MedXpert | 16.7 | 17.05 | +0.35 |
| MedQA | 72.6 | 77.93 | +5.33 |
| MedMCQA | 60.4 | 63.3 | +2.90 |
| PubMedQA | 79.2 | 80.2 | +1.00 |
| **Mean gap** |  |  | **+2.89** |

Largest single-benchmark gap: **5.33 pt**. That is large enough to confound a data-driven gain - prefer the measured before/after over comparisons with published rows.

## Raw score vs clean subset

The training set **kept** the rows overlapping these benchmarks (the upstream
setting, which is what the published tables were produced under). `Raw` is
therefore the number comparable to the paper; `Clean` excludes the overlapping
items and is the number to quote as an unbiased estimate. A large gap between
the two means the score leans on memorised evaluation items.

| Run | Benchmark | n | Contaminated | Raw | Clean | Bias |
|---|---|---|---|---|---|---|
| base-HuatuoGPT-o1-8B | MedQA | 1273 | 1 | 77.93 | 77.99 | -0.06 |
| base-HuatuoGPT-o1-8B | MMLU-Pro | 1535 | 340 | 64.56 | 64.52 | +0.04 |
| base-HuatuoGPT-o1-8B | MB-op4 | 308 | 35 | 59.09 | 58.97 | +0.12 |
| base-HuatuoGPT-o1-8B | MB-op5 | 308 | 35 | 55.19 | 53.48 | +1.71 |
| base-HuatuoGPT-o1-8B | MedXpert | 1449 | 15 | 17.05 | 16.88 | +0.17 |
| base-HuatuoGPT-o1-8B | HLE(med) | 103 | 30 | 11.65 | 8.22 | +3.43 |
| brainmed-8b-final | MedQA | 1273 | 1 | 76.67 | 76.73 | -0.06 |
| brainmed-8b-final | MMLU-Pro | 1535 | 340 | 65.73 | 65.44 | +0.29 |
| brainmed-8b-final | MB-op4 | 308 | 35 | 62.01 | 62.64 | -0.63 |
| brainmed-8b-final | MB-op5 | 308 | 35 | 54.87 | 54.21 | +0.66 |
| brainmed-8b-final | MedXpert | 1449 | 15 | 18.63 | 18.34 | +0.29 |
| brainmed-8b-final | HLE(med) | 103 | 30 | 10.68 | 6.85 | +3.83 |
| brainmed-8b-v1 | MedQA | 1273 | 1 | 72.58 | 72.56 | +0.02 |
| brainmed-8b-v1 | MMLU-Pro | 1535 | 340 | 61.82 | 60.67 | +1.15 |
| brainmed-8b-v1 | MB-op4 | 308 | 35 | 56.49 | 56.41 | +0.08 |
| brainmed-8b-v1 | MB-op5 | 308 | 35 | 50.65 | 49.08 | +1.57 |
| brainmed-8b-v1 | MedXpert | 1449 | 15 | 17.81 | 17.5 | +0.31 |
| brainmed-8b-v1 | HLE(med) | 103 | 30 | 20.39 | 17.81 | +2.58 |
| brainmed-8b-v1__best | MedQA | 1273 | 1 | 72.58 | 72.56 | +0.02 |
| brainmed-8b-v1__best | MB-op4 | 308 | 35 | 56.49 | 56.41 | +0.08 |
| brainmed-8b-v1__best | MB-op5 | 308 | 35 | 50.65 | 49.08 | +1.57 |
| brainmed-8b-v1__best | MedXpert | 1449 | 15 | 17.81 | 17.5 | +0.31 |
| brainmed-8b-v1__epoch-1 | MedQA | 1273 | 1 | 70.62 | 70.6 | +0.02 |
| brainmed-8b-v1__epoch-1 | MB-op4 | 308 | 35 | 55.52 | 55.31 | +0.21 |
| brainmed-8b-v1__epoch-1 | MB-op5 | 308 | 35 | 50.0 | 48.72 | +1.28 |
| brainmed-8b-v1__epoch-1 | MedXpert | 1449 | 15 | 15.87 | 15.83 | +0.04 |
| brainmed-8b-v1__epoch-2 | MedQA | 1273 | 1 | 72.58 | 72.56 | +0.02 |
| brainmed-8b-v1__epoch-2 | MB-op4 | 308 | 35 | 56.49 | 56.41 | +0.08 |
| brainmed-8b-v1__epoch-2 | MB-op5 | 308 | 35 | 50.65 | 49.08 | +1.57 |
| brainmed-8b-v1__epoch-2 | MedXpert | 1449 | 15 | 17.81 | 17.5 | +0.31 |
| brainmed-8b-v1__epoch-3 | MedQA | 1273 | 1 | 72.11 | 72.17 | -0.06 |
| brainmed-8b-v1__epoch-3 | MB-op4 | 308 | 35 | 58.77 | 57.14 | +1.63 |
| brainmed-8b-v1__epoch-3 | MB-op5 | 308 | 35 | 56.82 | 57.51 | -0.69 |
| brainmed-8b-v1__epoch-3 | MedXpert | 1449 | 15 | 18.08 | 17.78 | +0.30 |
| brainmed-8b-v1__last | MedQA | 1273 | 1 | 72.11 | 72.17 | -0.06 |
| brainmed-8b-v1__last | MB-op4 | 308 | 35 | 58.77 | 57.14 | +1.63 |
| brainmed-8b-v1__last | MB-op5 | 308 | 35 | 56.82 | 57.51 | -0.69 |
| brainmed-8b-v1__last | MedXpert | 1449 | 15 | 18.08 | 17.78 | +0.30 |
| brainmed-8b-v1__soup-a0.15 | MedQA | 1273 | 1 | 77.85 | 77.91 | -0.06 |
| brainmed-8b-v1__soup-a0.15 | MB-op4 | 308 | 35 | 60.06 | 60.44 | -0.38 |
| brainmed-8b-v1__soup-a0.15 | MB-op5 | 308 | 35 | 50.97 | 50.92 | +0.05 |
| brainmed-8b-v1__soup-a0.15 | MedXpert | 1449 | 15 | 16.08 | 15.76 | +0.32 |
| brainmed-8b-v1__soup-a0.3 | MedQA | 1273 | 1 | 76.28 | 76.26 | +0.02 |
| brainmed-8b-v1__soup-a0.3 | MB-op4 | 308 | 35 | 59.42 | 60.07 | -0.65 |
| brainmed-8b-v1__soup-a0.3 | MB-op5 | 308 | 35 | 54.87 | 54.95 | -0.08 |
| brainmed-8b-v1__soup-a0.3 | MedXpert | 1449 | 15 | 16.43 | 16.11 | +0.32 |
| brainmed-8b-v1__soup-last-a0.3 | MedQA | 1273 | 1 | 76.67 | 76.73 | -0.06 |
| brainmed-8b-v1__soup-last-a0.3 | MB-op4 | 308 | 35 | 62.01 | 62.64 | -0.63 |
| brainmed-8b-v1__soup-last-a0.3 | MB-op5 | 308 | 35 | 52.92 | 52.75 | +0.17 |
| brainmed-8b-v1__soup-last-a0.3 | MedXpert | 1449 | 15 | 18.63 | 18.34 | +0.29 |
| brainmed-8b-v1__soup-last-a0.5 | MedQA | 1273 | 1 | 76.9 | 76.97 | -0.07 |
| brainmed-8b-v1__soup-last-a0.5 | MB-op4 | 308 | 35 | 58.77 | 57.88 | +0.89 |
| brainmed-8b-v1__soup-last-a0.5 | MB-op5 | 308 | 35 | 54.55 | 53.11 | +1.44 |
| brainmed-8b-v1__soup-last-a0.5 | MedXpert | 1449 | 15 | 18.91 | 18.55 | +0.36 |
| brainmed-8b-v1__soup-last-a0.7 | MedQA | 1273 | 1 | 76.12 | 76.18 | -0.06 |
| brainmed-8b-v1__soup-last-a0.7 | MB-op4 | 308 | 35 | 60.71 | 59.71 | +1.00 |
| brainmed-8b-v1__soup-last-a0.7 | MB-op5 | 308 | 35 | 56.49 | 54.58 | +1.91 |
| brainmed-8b-v1__soup-last-a0.7 | MedXpert | 1449 | 15 | 19.32 | 19.11 | +0.21 |
| **base-HuatuoGPT-o1-8B** | **Average (8)** |  |  | **53.62** | **52.95** | **+0.67** |
| **brainmed-8b-final** | **Average (8)** |  |  | **53.97** | **53.42** | **+0.55** |
| **brainmed-8b-v1** | **Average (8)** |  |  | **52.21** | **51.49** | **+0.72** |

## Data integrity

- training rows after decontamination: **44351** (removed 0, 0.0%)

Contamination found in the *source* corpus and removed before training:

| Benchmark | n | Leaked items | % | Train rows dropped |
|---|---|---|---|---|
| gpqa_medical.jsonl | 390 | 0 | 0.0% | 0 |
| hle_med.jsonl | 103 | 30 | 29.13% | 30 |
| medbullets_op4.jsonl | 308 | 35 | 11.36% | 35 |
| medbullets_op5.jsonl | 308 | 35 | 11.36% | 35 |
| medmcqa_val.jsonl | 4183 | 0 | 0.0% | 0 |
| medqa_4opt.jsonl | 1273 | 1 | 0.08% | 1 |
| medqa_5opt.jsonl | 1273 | 1 | 0.08% | 1 |
| medxpertqa.jsonl | 1449 | 15 | 1.04% | 17 |
| mmlu_pro_medical.jsonl | 1535 | 340 | 22.15% | 261 |
| pubmedqa_test.jsonl | 1000 | 0 | 0.0% | 0 |

## Answer-format compliance

Share of outputs where the answer was read from an explicit `The answer is X.` rather
than recovered by the scorer's fuzzy fallbacks. A low value means the accuracy above
leans on string matching and should be treated with suspicion.

| Run | Benchmark | Strict-format % | Empty outputs |
|---|---|---|---|
| base-HuatuoGPT-o1-8B | MedQA | 99.84% | 0 |
| base-HuatuoGPT-o1-8B | MedMCQA | 99.81% | 0 |
| base-HuatuoGPT-o1-8B | PubMedQA | 100.0% | 0 |
| base-HuatuoGPT-o1-8B | MMLU-Pro | 99.15% | 0 |
| base-HuatuoGPT-o1-8B | MB-op4 | 100.0% | 0 |
| base-HuatuoGPT-o1-8B | MB-op5 | 99.68% | 0 |
| base-HuatuoGPT-o1-8B | MedXpert | 96.27% | 0 |
| base-HuatuoGPT-o1-8B | HLE(med) | 96.12% | 0 |
| base-HuatuoGPT-o1-8B | GPQA(med) | 98.21% | 0 |
| base-HuatuoGPT-o1-8B | MedQA-5opt | 99.76% | 0 |
| brainmed-8b-final | MedQA | 99.14% | 0 |
| brainmed-8b-final | MedMCQA | 99.83% | 0 |
| brainmed-8b-final | PubMedQA | 99.9% | 0 |
| brainmed-8b-final | MMLU-Pro | 99.48% | 0 |
| brainmed-8b-final | MB-op4 | 99.68% | 0 |
| brainmed-8b-final | MB-op5 | 99.35% | 0 |
| brainmed-8b-final | MedXpert | 99.24% | 0 |
| brainmed-8b-final | HLE(med) | 97.09% | 0 |
| brainmed-8b-final | GPQA(med) | 98.46% | 0 |
| brainmed-8b-final | MedQA-5opt | 98.9% | 0 |
| brainmed-8b-v1 | MedQA | 100.0% | 0 |
| brainmed-8b-v1 | MedMCQA | 99.88% | 0 |
| brainmed-8b-v1 | PubMedQA | 100.0% | 0 |
| brainmed-8b-v1 | MMLU-Pro | 99.61% | 0 |
| brainmed-8b-v1 | MB-op4 | 100.0% | 0 |
| brainmed-8b-v1 | MB-op5 | 100.0% | 0 |
| brainmed-8b-v1 | MedXpert | 100.0% | 0 |
| brainmed-8b-v1 | HLE(med) | 95.15% | 0 |
| brainmed-8b-v1 | GPQA(med) | 99.74% | 0 |
| brainmed-8b-v1 | MedQA-5opt | 99.92% | 0 |
| brainmed-8b-v1__best | MedQA | 100.0% | 0 |
| brainmed-8b-v1__best | MedMCQA | 99.88% | 0 |
| brainmed-8b-v1__best | PubMedQA | 100.0% | 0 |
| brainmed-8b-v1__best | MB-op4 | 100.0% | 0 |
| brainmed-8b-v1__best | MB-op5 | 100.0% | 0 |
| brainmed-8b-v1__best | MedXpert | 100.0% | 0 |
| brainmed-8b-v1__epoch-1 | MedQA | 99.92% | 0 |
| brainmed-8b-v1__epoch-1 | MedMCQA | 99.62% | 0 |
| brainmed-8b-v1__epoch-1 | PubMedQA | 99.9% | 0 |
| brainmed-8b-v1__epoch-1 | MB-op4 | 100.0% | 0 |
| brainmed-8b-v1__epoch-1 | MB-op5 | 100.0% | 0 |
| brainmed-8b-v1__epoch-1 | MedXpert | 99.72% | 0 |
| brainmed-8b-v1__epoch-2 | MedQA | 100.0% | 0 |
| brainmed-8b-v1__epoch-2 | MedMCQA | 99.88% | 0 |
| brainmed-8b-v1__epoch-2 | PubMedQA | 100.0% | 0 |
| brainmed-8b-v1__epoch-2 | MB-op4 | 100.0% | 0 |
| brainmed-8b-v1__epoch-2 | MB-op5 | 100.0% | 0 |
| brainmed-8b-v1__epoch-2 | MedXpert | 100.0% | 0 |
| brainmed-8b-v1__epoch-3 | MedQA | 100.0% | 0 |
| brainmed-8b-v1__epoch-3 | MedMCQA | 99.9% | 0 |
| brainmed-8b-v1__epoch-3 | PubMedQA | 100.0% | 0 |
| brainmed-8b-v1__epoch-3 | MB-op4 | 100.0% | 0 |
| brainmed-8b-v1__epoch-3 | MB-op5 | 100.0% | 0 |
| brainmed-8b-v1__epoch-3 | MedXpert | 100.0% | 0 |
| brainmed-8b-v1__last | MedQA | 100.0% | 0 |
| brainmed-8b-v1__last | MedMCQA | 99.9% | 0 |
| brainmed-8b-v1__last | PubMedQA | 100.0% | 0 |
| brainmed-8b-v1__last | MB-op4 | 100.0% | 0 |
| brainmed-8b-v1__last | MB-op5 | 100.0% | 0 |
| brainmed-8b-v1__last | MedXpert | 100.0% | 0 |
| brainmed-8b-v1__soup-a0.15 | MedQA | 89.32% | 0 |
| brainmed-8b-v1__soup-a0.15 | MedMCQA | 99.19% | 0 |
| brainmed-8b-v1__soup-a0.15 | PubMedQA | 99.6% | 0 |
| brainmed-8b-v1__soup-a0.15 | MB-op4 | 86.69% | 0 |
| brainmed-8b-v1__soup-a0.15 | MB-op5 | 87.66% | 0 |
| brainmed-8b-v1__soup-a0.15 | MedXpert | 92.41% | 0 |
| brainmed-8b-v1__soup-a0.3 | MedQA | 99.69% | 0 |
| brainmed-8b-v1__soup-a0.3 | MedMCQA | 99.64% | 0 |
| brainmed-8b-v1__soup-a0.3 | PubMedQA | 100.0% | 0 |
| brainmed-8b-v1__soup-a0.3 | MB-op4 | 99.03% | 0 |
| brainmed-8b-v1__soup-a0.3 | MB-op5 | 99.68% | 0 |
| brainmed-8b-v1__soup-a0.3 | MedXpert | 99.59% | 0 |
| brainmed-8b-v1__soup-last-a0.3 | MedQA | 99.14% | 0 |
| brainmed-8b-v1__soup-last-a0.3 | MedMCQA | 99.83% | 0 |
| brainmed-8b-v1__soup-last-a0.3 | PubMedQA | 99.9% | 0 |
| brainmed-8b-v1__soup-last-a0.3 | MB-op4 | 99.68% | 0 |
| brainmed-8b-v1__soup-last-a0.3 | MB-op5 | 99.35% | 0 |
| brainmed-8b-v1__soup-last-a0.3 | MedXpert | 99.24% | 0 |
| brainmed-8b-v1__soup-last-a0.5 | MedQA | 99.84% | 0 |
| brainmed-8b-v1__soup-last-a0.5 | MedMCQA | 99.86% | 0 |
| brainmed-8b-v1__soup-last-a0.5 | PubMedQA | 100.0% | 0 |
| brainmed-8b-v1__soup-last-a0.5 | MB-op4 | 100.0% | 0 |
| brainmed-8b-v1__soup-last-a0.5 | MB-op5 | 100.0% | 0 |
| brainmed-8b-v1__soup-last-a0.5 | MedXpert | 100.0% | 0 |
| brainmed-8b-v1__soup-last-a0.7 | MedQA | 100.0% | 0 |
| brainmed-8b-v1__soup-last-a0.7 | MedMCQA | 99.93% | 0 |
| brainmed-8b-v1__soup-last-a0.7 | PubMedQA | 100.0% | 0 |
| brainmed-8b-v1__soup-last-a0.7 | MB-op4 | 100.0% | 0 |
| brainmed-8b-v1__soup-last-a0.7 | MB-op5 | 100.0% | 0 |
| brainmed-8b-v1__soup-last-a0.7 | MedXpert | 100.0% | 0 |
