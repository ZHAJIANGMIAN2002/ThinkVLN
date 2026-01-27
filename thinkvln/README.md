# ThinkVLN

Vision-Language Navigation model with Chain-of-Thought reasoning capabilities.

## Project Structure

```
thinkvln/
├── models/              # Model architectures
│   ├── thinkvln_sequential.py    # Main ThinkVLN model implementation
│   └── layers/                   # Custom neural network layers
├── engine/              # Training & evaluation infrastructure
│   ├── env_eval.py              # Habitat environment evaluation
│   └── openloop_eval.py         # Open-loop evaluation
├── data/                # Data processing & datasets
│   ├── dataset.py               # PyTorch dataset classes
│   ├── preprocessing/           # Data preprocessing scripts
│   └── generation/              # CoT and trajectory data generation
├── tools/               # Utility tools & scripts
│   ├── inference.py             # Inference utilities
│   ├── web_refinement/          # Web-based annotation tool
│   └── dev_scripts/             # Original development scripts (for reference)
├── habitat_extensions/  # Habitat simulator extensions
│   ├── maps.py                  # Custom map utilities
│   └── measures.py              # Custom evaluation metrics
├── scripts/             # Training/evaluation entry points
│   └── train.py                 # Main training script
├── tests/               # Unit tests
└── datasets/            # Dataset storage (gitignored)
```

## Quick Start

### Training
```bash
python scripts/train.py \
    --model_path ./models/qwen3vl-2 \
    --dataset_path ./datasets/train.json \
    --output_dir ./outputs \
    --batch_size 4 \
    --num_epochs 3
```

### Evaluation
```bash
# Environment evaluation
python -m thinkvln.engine.env_eval \
    --model_path ./outputs/checkpoint-epoch-3 \
    --config_path ./configs/eval.yaml

# Open-loop evaluation
python -m thinkvln.engine.openloop_eval \
    --model_path ./outputs/checkpoint-epoch-3 \
    --dataset_path ./datasets/test.json
```

### Data Generation
```bash
# Generate CoT data
python -m thinkvln.data.generation.cot_generation \
    --input_path ./raw_data/trajectories.json \
    --output_path ./datasets/cot_data.jsonl

# Generate subtask data
python -m thinkvln.data.generation.subtask_determination \
    --input_path ./raw_data/instructions.json \
    --output_path ./datasets/subtasks.jsonl
```

## Key Components

### Model (`thinkvln.models`)
- **ThinkVLNModel**: Sequential VLN model with action head for navigation
- Based on Qwen3-VL with custom action prediction layer

### Engine (`thinkvln.engine`)
- **env_eval.py**: Evaluate model in Habitat simulator environment
- **openloop_eval.py**: Evaluate model predictions without environment interaction

### Data (`thinkvln.data`)
- **dataset.py**: PyTorch Dataset classes for training
- **preprocessing/**: Scripts to prepare SFT datasets, extract frames
- **generation/**: Scripts to generate CoT reasoning, subtasks, and trajectories

### Tools (`thinkvln.tools`)
- **inference.py**: Inference utilities and helpers
- **web_refinement/**: Web interface for manual data refinement
- **deploy/**: Production deployment scripts

## Development

### Adding New Models
Place new model architectures in `thinkvln/models/` and register in `__init__.py`.

### Adding Tests
Add test files to `thinkvln/tests/` following pytest conventions.

### Data Format
Training data should be in JSON format with the following structure:
```json
{
  "messages": [...],
  "images": ["path/to/image.jpg"],
  "action": 0,
  "action_label": 0
}
```

## License
See LICENSE file for details.
