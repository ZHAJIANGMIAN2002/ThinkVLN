# ThinkVLN

Vision-Language Navigation model with Chain-of-Thought reasoning capabilities.

## Project Structure

```
thinkvln/
├── models/              # Model architectures
│   ├── thinkvln_model.py       # Base VLN model (ThinkVLNForConditionalGeneration)
│   ├── thinkvln_actor.py      # Actor with action/progress heads
│   ├── thinkvln_config.py     # Model config
│   └── actor_config.py        # Actor training config
├── engine/              # Training & inference
│   ├── sft_trainer.py         # SFT training entry point
│   └── inference.py           # Model loading and inference utilities
├── eval/                 # Evaluation scripts
│   ├── env_eval.py            # Habitat simulator evaluation
│   └── openloop_eval.py       # Open-loop evaluation
├── dataset/              # Dataset classes
│   └── dataset.py             # ThinkVLNDataset for mixed action + CoT
├── datagen/              # Data generation & preprocessing
│   ├── generation/            # CoT, subtask, trajectory generation
│   └── preprocessing/         # SFT dataset creation, frame extraction
├── tools/                # Utilities
│   ├── dataset_utils.py       # Data loading helpers
│   ├── web_refinement/        # Web-based annotation tool
│   └── dev_scripts/           # Reference scripts
├── habitat_extensions/   # Habitat simulator extensions
│   ├── maps.py                # Map utilities
│   └── measures.py            # Custom evaluation metrics
└── tests/                # Unit tests
```

## Quick Start

### Training
```bash
# Single GPU
python thinkvln/engine/sft_trainer.py --config config/sft_training.yaml

# Multi-GPU (DeepSpeed)
torchrun --nproc_per_node=4 thinkvln/engine/sft_trainer.py --config config/sft_training.yaml
```

### Evaluation
```bash
# Habitat environment evaluation
python -m thinkvln.eval.env_eval --config_path <path> --model_path <path>

# Open-loop evaluation
python -m thinkvln.eval.openloop_eval --model_path <path> --dataset_path <path>
```

### Data Generation
```bash
# CoT generation
python -m thinkvln.datagen.generation.cot_generation --input_path <path> --output_path <path>

# Subtask determination
python -m thinkvln.datagen.generation.subtask_determination --input_path <path> --output_path <path>
```

## Key Components

### Model (`thinkvln.models`)
- **ThinkVLNModel / ThinkVLNForConditionalGeneration**: Base VLN model based on Qwen3-VL
- **ThinkVLNActor**: Actor with ActionClassificationHead and ProgressRegressionHead for navigation
- Supports LoRA, vision tower freezing, FlashAttention-2

### Engine (`thinkvln.engine`)
- **sft_trainer.py**: SFT training with mixed action and CoT data
- **inference.py**: Model loading, single/batch inference utilities

### Eval (`thinkvln.eval`)
- **env_eval.py**: Evaluate in Habitat simulator
- **openloop_eval.py**: Evaluate predictions without environment

### Dataset (`thinkvln.dataset`)
- **ThinkVLNDataset**: Mixed action + CoT samples, JSONL format

### Datagen (`thinkvln.datagen`)
- **generation/**: CoT reasoning, subtask determination, trajectory generation
- **preprocessing/**: SFT dataset creation, frame extraction

## Data Format

**Action data (JSONL):**
```json
{"episode_key": "...", "instruction": "...", "plan": [...], "actions": [...], "subtask_sequence": [...], "num_frames": N}
```

**CoT data (JSONL):**
```json
{"frame_key": "...", "episode_key": "...", "instruction": "...", "plan": [...], "answer": "..."}
```

## License
See LICENSE file for details.
