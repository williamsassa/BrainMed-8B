---
title: BrainMed-8B
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Medical reasoning demo - full fine-tune of HuatuoGPT-o1-8B
---

# BrainMed-8B — demo

Interactive demo of a full-parameter fine-tune of `FreedomIntelligence/HuatuoGPT-o1-8B` on a
union of KG-grounded MedReason traces and verifier-checked medical-o1 traces.

**Research demo, not medical advice.** Not a medical device, not clinically validated, and not
to be used for decisions about any real person.

## Hardware

The model is 8B parameters (~16 GB in bf16). It **will not run on the free CPU tier** — loading
succeeds but a single answer takes many minutes. Assign a GPU under
*Settings → Hardware* (an A10G-small, 24 GB, is comfortable; a T4-small at 16 GB is tight).

## Configuration

| Variable | Purpose |
|---|---|
| `MODEL_ID` | model repo to serve (default `BrainHealthAI/BrainMed-8B`) |
| `HF_TOKEN` | **required while the model repo is private** — add it under *Settings → Secrets* |

## Prompt contract

The model is trained under a fixed system prompt and answers as
`<think>…</think><answer>…</answer>`. The app sends that same system prompt at inference and
splits the two blocks for display. Changing the system prompt is a train/serve mismatch and
measurably lowers accuracy — the field is editable so that can be observed, not so it can be
ignored.
