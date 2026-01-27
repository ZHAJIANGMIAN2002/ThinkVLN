# Keyframe Annotation Refinement Web Interface

A Flask web application for interactively refining keyframe subtask annotations.

## Features

- View keyframes extracted from video trajectories
- See current subtask annotations for each keyframe
- Modify annotations by selecting from plan choices (A, B, C, D format)
- Save changes back to the JSONL file with automatic post-processing
- Visual feedback for modified annotations

## Requirements

- Python 3.7+
- Flask
- OpenCV (cv2)
- NumPy

Install dependencies:
```bash
pip install flask opencv-python numpy
```

## Usage

Run the web server:

```bash
cd streamvln/cot_data/web_refinement
python app.py [OPTIONS]
```

### Options

- `--trajectory_dir`: Path to trajectory data directory (default: `data/trajectory_data/R2R`)
- `--subtask_determination_file`: Path to subtask_determination.jsonl file (default: `streamvln/cot_data/subtask_determination.jsonl`)
- `--subtask_splits_file`: Path to subtask_splits.jsonl file (default: `streamvln/cot_data/subtask_splits.jsonl`)
- `--port`: Port to run the server on (default: 5000)
- `--host`: Host to bind to (default: 0.0.0.0 for server access)

### Example

```bash
python app.py --trajectory_dir /path/to/trajectory_data/R2R --port 5000
```

Then open your browser and navigate to:
- Local access: `http://localhost:5000`
- Remote access (from Mac): `http://<server-ip>:5000`

## How to Use

1. **Select an Episode**: Use the dropdown at the top to select an episode to refine
2. **View Keyframes**: The page displays all keyframes in a grid layout
3. **Review Plan**: The plan choices (A, B, C, D) are shown at the top
4. **Modify Annotations**: For each keyframe:
   - Click on a choice button (A, B, C, D) to change the subtask annotation
   - Modified keyframes are highlighted in red
5. **Save Changes**: Click the "Save Changes" button to update the JSONL file
   - The system automatically regenerates `subtask_sequence` and `subtask_counts`
   - A backup of the original file is created (`.backup`)

## File Format

The application reads and writes JSONL format (one JSON object per line) for `subtask_determination.jsonl`.

Each episode entry contains:
- `episode_key`: Unique identifier
- `keyframes`: List of frame indices that are keyframes
- `vlm_annotations`: List of `{frame_index, subtask_index}` pairs
- `subtask_sequence`: Full sequence for all frames (regenerated on save)
- `subtask_counts`: Count of frames per subtask (regenerated on save)

## Notes

- The application creates a backup file (`.backup`) before saving changes
- Frame extraction happens on-demand and is cached during the session
- The `subtask_sequence` is automatically regenerated using interpolation after saving

