# ThinkVLN 🧭

ThinkVLN is a vision-language navigation research codebase for embodied agents that must interpret instructions, reason over long-horizon context, and predict navigation actions from visual observations.

This repository brings together the current ThinkVLN training, evaluation, and data-generation stack: actor-style navigation models, chain-of-thought style supervision, automatic annotation pipelines, and closed-loop evaluation tools. It follows earlier StreamVLN explorations, while the repository itself is maintained as the ThinkVLN codebase.

## Overview 🔬

ThinkVLN is organized as a research workspace rather than a single training script. The codebase supports iterative work on model design, supervised fine-tuning, closed-loop policy evaluation, reasoning-oriented dataset construction, and annotation workflows for navigation experiments.

In short, this repo is where we train models, build datasets, generate auxiliary supervision, and evaluate navigation behavior end to end.

## Highlights ✨

- Train ThinkVLN actor models with supervised fine-tuning configs.
- Run closed-loop evaluation for navigation policies.
- Build action and reasoning datasets for VLN-style tasks.
- Generate rollout annotations, trajectory summaries, and supporting supervision.
- Iterate on ThinkVLN, ThinkVLN-Actor, and related research variants in one workspace.

## Repository Map 🗂️

- `thinkvln/models`: model definitions, actor heads, configs, and flow-matching components.
- `thinkvln/engine`: training and inference entry points for SFT, AR, and evaluation helpers.
- `thinkvln/eval`: closed-loop and debug evaluation pipelines.
- `thinkvln/dataset`: dataset builders and dataset classes for training and analysis.
- `thinkvln/datagen`: preprocessing, annotation, rollout, trajectory, and video generation tools.
- `thinkvln/tools`: internal utility tools, web refinement apps, and development scripts.
- `thinkvln/tests`: unit tests and integration-oriented checks for core pipelines.
- `config`: YAML configs for training, evaluation, and data workflows.
- `scripts`: shell and Python entry scripts for common experiment workflows.

## Quick Entry 🚀

Training:

```bash
bash scripts/train_thinkvln_actor.sh config/sft_training.yaml
```

Evaluation:

```bash
python -m thinkvln.eval.close_eval --help
```

Data generation:

```bash
python -m thinkvln.datagen.generation.cot_generation --help
python -m thinkvln.datagen.generation.subtask_determination --help
```

## Project Notes 🧠

- The repository includes research code, experiment configs, and utility scripts under active iteration.
- Some modules still reference legacy `streamvln` components for compatibility and comparison.
- Deployment instructions are intentionally omitted here for now.

## Structure

```text
ThinkVLN/
├── config/
├── scripts/
├── thinkvln/
│   ├── datagen/
│   ├── dataset/
│   ├── engine/
│   ├── eval/
│   ├── habitat_extensions/
│   ├── models/
│   ├── tests/
│   └── tools/
├── streamvln/
├── docs/
└── README.md
```

## License

Add the project license information here before publishing the repository publicly.
