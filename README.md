# Pixels to Predictions: Multimodal Science QA via QLoRA Fine-Tuning

NYU Tandon Deep Learning (CS-GY 6953), Spring 2026  
Kaggle Competition: Pixels to Predictions: DL Vision Challenge

**Team:** WorkinTilTheAM (Michael Blanchette, Ashley Goluoğlu)

**Public Leaderboard:** 0.80684 (rank 60/106) rip lol

## Overview

QLoRA fine-tuning of SmolVLM-500M-Instruct for multimodal scientific multiple-choice question answering. Given an image (diagram, chart, map, Punnett square) paired with a question and answer choices, the model predicts the correct answer index. Development was constrained to Kaggle Free tier (T4 GPU, 30 hrs/week) with a 5-million trainable parameter cap.

Key contributions:
- **Label masking** on prompt tokens (+15% accuracy vs. naive loss)
- **Systematic ablation** across 6 axes (scoring strategy, DoRA, LoRA rank/modules, prompt engineering)
- Finding that **higher-rank attention-only LoRA outperforms broader MLP coverage** at lower rank, contradicting common guidance

## Repository Structure

```
├── dlfinal-ag8529.ipynb                          # Full notebook (local copy)
├── DLFinal_ag8529                                # Kaggle notebook output (Version 4)
├── NYU_DeepLearning_Spring2026_FinalReport.pdf   # Report
├── LICENSE
└── README.md
```

## Competition Details

- **Task:** Multimodal multiple-choice QA on science questions (2–5 choices)
- **Data:** 3,109 train / 1,048 val / 1,008 test examples with images
- **Metric:** Classification accuracy on hidden test labels
- **Constraints:** SmolVLM-500M-Instruct only, ≤5M trainable params, no external data, offline inference

## How to Reproduce

### On Kaggle (recommended, we did not test on Colab)

1. Go to the [competition page](https://www.kaggle.com/competitions/pixels-to-predictions-dl-vision-challenge) to get the dataset
2. Import `dlfinal-ag8529.ipynb` as a Kaggle notebook
3. Attach the competition dataset as input
4. Enable GPU (T4) 
5. Run all cells. The notebook handles everything end-to-end:
   - Installs/imports/requirements
   - Data examination and preprocessing analysis
   - Ablation runs (results persist to JSON across sessions)
   - Full training with best config
   - Submission CSV generation

### Resuming across sessions

Ablation results are saved to `ablation_results.json` in the output directory. The notebook checks both `/kaggle/working/` and `/kaggle/input/` on startup, so you can add previous results as a dataset to resume without re-running completed ablations.

## Final Config

```
Model:            SmolVLM-500M-Instruct (4-bit NF4 quantized)
LoRA rank:        16
LoRA alpha:       16
LoRA dropout:     0.05
Target modules:   q_proj, v_proj, k_proj, o_proj
DoRA:             disabled
Trainable params: 4,161,536 (83% of 5M budget)
Training:         3 + 3 epochs (second phase at 0.5x LR)
Effective batch:  16 (batch=4, accumulation=4)
max_seq_length:   700
Prompt format:    minimal ("Answer:", no metadata)
Context:          full (lecture + hint)
Scoring:          generate
Seed:             42
```

## Ablation Results Summary

| Config | Val Acc |
|--------|---------|
| attn4 r16, bare prompt | **0.690** |
| attn4 r16, + metadata | 0.680 |
| all7 r6, DoRA, meta+fmt | 0.670 |
| attn4 r16, DoRA | 0.650 |
| all7 r8, meta+fmt | 0.610 |
| all7 r8, bare | 0.600 |
| loglikelihood scoring | 0.550 |

*All ablation runs: 300 steps, 200 eval samples. Full results in notebook and report.*

## Key Findings

1. **Label masking is essential**: training only on answer tokens vs. full sequence improved accuracy by ~15 percentage points
2. **Attention-only LoRA at high rank > all-module LoRA at low rank**: contradicts standard guidance; rank 16 on q/v/k/o outperformed rank 8 on all 7 modules
3. **DoRA hurts**: doubled training time, lower accuracy in every comparison
4. **Simpler prompts win**: no metadata, short "Answer:" format beat elaborate alternatives
5. **Generate scoring is more stable**: log-likelihood peaked early then collapsed

## Requirements

The Kaggle notebook installs its own dependencies. Core packages:

- `transformers`
- `peft`
- `bitsandbytes`
- `accelerate`
- `torch`
- `pillow`

## AI Tooling

Claude (Anthropic) was used as a coding and debugging assistant throughout development, primarily for debugging processor/tokenizer interactions and understanding theory behind ablation decisions. All experimental design decisions were made by the team.

## Results

| Split | Score |
|-------|-------|
| Public leaderboard | 0.80684 |
| Private leaderboard | TBD |
