# Migration Guide: Old to New Structure

This document explains the changes from the old to the new thinkvln folder structure.

## Summary of Changes

1. **Separated Models from Engineering**: Models are now in `models/`, training/eval infrastructure in `engine/`
2. **Organized Data Pipeline**: All data-related code consolidated under `data/`
3. **Deploy vs Dev Scripts**: Production scripts in `data/generation/`, dev versions in `tools/dev_scripts/`
4. **Cleaned Up**: Removed `__pycache__`, organized tests, added proper `__init__.py` files

## Path Mapping

### Model Files
| Old Path | New Path |
|----------|----------|
| `thinkvln/model/thinkvln_model_sequential.py` | `thinkvln/models/thinkvln_sequential.py` |
| ~~`thinkvln/model/thinkvln_model.py`~~ | (Deleted, use sequential version) |

### Training & Evaluation
| Old Path | New Path |
|----------|----------|
| `thinkvln/train_sequential.py` | `thinkvln/scripts/train.py` |
| `thinkvln/model/env_eval.py` | `thinkvln/engine/env_eval.py` |
| `thinkvln/model/openloop_eval.py` | `thinkvln/engine/openloop_eval.py` |
| `thinkvln/model/inference.py` | `thinkvln/tools/inference.py` |

### Data Processing
| Old Path | New Path |
|----------|----------|
| `thinkvln/dataset/dataset.py` | `thinkvln/data/dataset.py` |
| `thinkvln/dataset/create_sft_dataset.py` | `thinkvln/data/preprocessing/create_sft_dataset.py` |
| `thinkvln/dataset/convert_cot_to_answer.py` | `thinkvln/data/preprocessing/convert_cot_to_answer.py` |
| `thinkvln/dataset/extract_frames_for_sft.py` | `thinkvln/data/preprocessing/extract_frames.py` |

### Data Generation (Production Scripts)
| Old Path | New Path |
|----------|----------|
| `thinkvln/cot_data/cot_generation_deploy.py` | `thinkvln/data/generation/cot_generation.py` |
| `thinkvln/cot_data/subtask_determination_deploy.py` | `thinkvln/data/generation/subtask_determination.py` |
| `thinkvln/cot_data/subtask_split_deploy.py` | `thinkvln/data/generation/subtask_split.py` |
| `thinkvln/cot_data/streamvln_trajectory_generation.py` | `thinkvln/data/generation/trajectory_generation.py` |
| `thinkvln/cot_data/generate_checking_videos.py` | `thinkvln/data/generation/generate_videos.py` |
| `thinkvln/cot_data/prompt.py` | `thinkvln/data/generation/prompt.py` |

### Development Scripts (Reference Only)
| Old Path | New Path |
|----------|----------|
| `thinkvln/cot_data/cot_generation.py` | `thinkvln/tools/dev_scripts/cot_generation.py` |
| `thinkvln/cot_data/subtask_determination.py` | `thinkvln/tools/dev_scripts/subtask_determination.py` |
| `thinkvln/cot_data/subtask_split.py` | `thinkvln/tools/dev_scripts/subtask_split.py` |

### Tools & Web Interface
| Old Path | New Path |
|----------|----------|
| `thinkvln/cot_data/web_refinement/` | `thinkvln/tools/web_refinement/` |

### Tests
| Old Path | New Path |
|----------|----------|
| `thinkvln/cot_data/tests/test_cot_generation_episode.py` | `thinkvln/tests/test_cot_generation.py` |
| `thinkvln/cot_data/tests/test_subtask_determination_real_data.py` | `thinkvln/tests/test_subtask_determination.py` |
| `thinkvln/cot_data/tests/test_subtask_split.py` | `thinkvln/tests/test_subtask_split.py` |

### Dataset Files
| Old Path | New Path |
|----------|----------|
| `thinkvln/cot_data/*.jsonl` | `thinkvln/datasets/cot_data/*.jsonl` |

### Habitat Extensions
| Old Path | New Path |
|----------|----------|
| `thinkvln/habitat_extensions/maps.py` | `thinkvln/habitat_extensions/maps.py` (unchanged) |
| `thinkvln/habitat_extensions/measures.py` | `thinkvln/habitat_extensions/measures.py` (unchanged) |

## Import Changes

### Before (Old Structure)
```python
from thinkvln.model.thinkvln_model import ThinkVLNModel
from thinkvln.dataset.dataset import ThinkVLNDataset, load_image
```

### After (New Structure)
```python
from thinkvln.models import ThinkVLNModel
from thinkvln.data import ThinkVLNDataset, load_image
```

## Running Commands

### Training
**Before:**
```bash
python thinkvln/train_sequential.py --model_path ./models/qwen3vl-2 --dataset_path data.json
```

**After:**
```bash
python thinkvln/scripts/train.py --model_path ./models/qwen3vl-2 --dataset_path data.json
# OR as a module
python -m thinkvln.scripts.train --model_path ./models/qwen3vl-2 --dataset_path data.json
```

### Evaluation
**Before:**
```bash
python thinkvln/model/env_eval.py --config config.yaml
```

**After:**
```bash
python -m thinkvln.engine.env_eval --config config.yaml
```

### Data Generation
**Before:**
```bash
python thinkvln/cot_data/cot_generation_deploy.py --input data.json
```

**After:**
```bash
python -m thinkvln.data.generation.cot_generation --input data.json
```

## What Was Removed

1. **`__pycache__`** directories - All cleaned up
2. **Backup files** - Removed files like `subtask_determination_back.jsonl`
3. **Old model versions** - Kept only `thinkvln_sequential.py`
4. **Deleted llava files** - Already staged for deletion in git

## What Was Added

1. **`__init__.py`** files in all packages for proper imports
2. **README.md** files for documentation
3. **`.gitignore`** for better git hygiene
4. **Organized folder structure** with clear separation of concerns

## Rollback Instructions

If you need to rollback to the old structure:
```bash
# The old thinkvln folder is still intact, just rename folders
mv thinkvln thinkvln_backup
mv thinkvln_old thinkvln  # if you kept the original
```

## Next Steps

1. Test the training script: `python thinkvln/scripts/train.py --help`
2. Update any external scripts that import from thinkvln
3. Update CI/CD pipelines if any
4. Once verified, delete the old `thinkvln/` folder: `rm -rf thinkvln_old/`
