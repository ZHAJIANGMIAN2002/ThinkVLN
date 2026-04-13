# ThinkVLN

ThinkVLN is a vision-language navigation research codebase focused on instruction following, long-horizon reasoning, action prediction, and data generation for embodied navigation.

This repository contains the current ThinkVLN training, evaluation, and dataset tooling used around actor-style navigation models, chain-of-thought style supervision, automatic annotation pipelines, and closed-loop evaluation. It also keeps compatibility code for earlier StreamVLN components, but this repository is maintained as the ThinkVLN codebase.

## What Is In This Repo

- `thinkvln/models`: ThinkVLN model definitions, actor heads, configs, and flow-matching components.
- `thinkvln/engine`: training and inference entry points for SFT, AR, and evaluation helpers.
- `thinkvln/eval`: closed-loop and debug evaluation pipelines.
- `thinkvln/dataset`: dataset builders and dataset classes for training and analysis.
- `thinkvln/datagen`: preprocessing, annotation, rollout, trajectory, and video generation tools.
- `thinkvln/tools`: internal utility tools, web refinement apps, and development scripts.
- `thinkvln/tests`: unit tests and integration-oriented checks for core pipelines.
- `config`: YAML configs for training, evaluation, and data workflows.
- `scripts`: shell and Python entry scripts for common experiments and data processing.

## Main Capabilities

- Train ThinkVLN actor models with supervised fine-tuning configs.
- Run closed-loop evaluation for navigation policies.
- Build and preprocess action / reasoning datasets for VLN-style tasks.
- Generate auxiliary annotations, rollout artifacts, and trajectory summaries.
- Support iterative experiments around ThinkVLN, ThinkVLN-Actor, and related research variants.

## Typical Entry Points

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

## Project Notes

- The repository includes research code, experiment configs, and utility scripts under active iteration.
- Some modules still reference legacy `streamvln` components for compatibility and comparison, but the top-level project identity is ThinkVLN.
- Deployment instructions are intentionally omitted here for now.

## Repository Structure

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
