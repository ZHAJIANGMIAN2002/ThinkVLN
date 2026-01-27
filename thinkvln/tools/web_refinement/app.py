#!/usr/bin/env python3
"""
Web interface for refining keyframe subtask annotations.
"""
import os
import json
import argparse
from flask import Flask, render_template, jsonify, request
import sys

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_subtask_determination,
    load_subtask_splits,
    save_subtask_determination,
    extract_frame_from_video,
    frame_to_base64,
    update_episode_annotations,
    get_episode_key
)

# Initialize Flask app with template and static folders
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

# Global state
TRAJECTORY_DIR = None
SUBTASK_DETERMINATION_FILE = None
SUBTASK_SPLITS_FILE = None
EPISODES_DATA = {}
SUBTASK_SPLITS = {}
VIDEO_PATH_CACHE = {}  # Cache episode_key -> video_rel_path


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/api/episodes')
def get_episodes():
    """Get list of all episodes."""
    episodes_list = []
    for episode_key, data in EPISODES_DATA.items():
        episodes_list.append({
            "episode_key": episode_key,
            "episode_id": data.get("episode_id"),
            "scene_id": data.get("scene_id"),
            "num_frames": data.get("num_frames", 0),
            "num_subtasks": data.get("num_subtasks", 0),
            "num_keyframes": len(data.get("keyframes", []))
        })
    
    # Sort by episode_key
    episodes_list.sort(key=lambda x: x["episode_key"])
    return jsonify({"episodes": episodes_list})


@app.route('/api/episode/<episode_key>')
def get_episode(episode_key):
    """Get episode data including keyframes and plan."""
    if episode_key not in EPISODES_DATA:
        return jsonify({"error": "Episode not found"}), 404
    
    episode_data = EPISODES_DATA[episode_key].copy()
    
    # Get plan from subtask_splits
    plan = []
    if episode_key in SUBTASK_SPLITS:
        plan = SUBTASK_SPLITS[episode_key].get("plan", [])
        episode_data["instruction"] = SUBTASK_SPLITS[episode_key].get("instruction", "")
    
    episode_data["plan"] = plan
    
    return jsonify(episode_data)


@app.route('/api/frame/<episode_key>/<int:frame_idx>')
def get_frame(episode_key, frame_idx):
    """Get frame image as base64 encoded JPEG."""
    if episode_key not in EPISODES_DATA:
        return jsonify({"error": "Episode not found"}), 404
    
    # Get video path (use cache if available)
    if episode_key not in VIDEO_PATH_CACHE:
        episode_data = EPISODES_DATA[episode_key]
        scene_id = episode_data.get("scene_id")
        episode_id = episode_data.get("episode_id")
        
        # Try to get video path from summary.json
        video_rel_path = None
        summary_file = os.path.join(TRAJECTORY_DIR, "summary.json")
        if os.path.exists(summary_file):
            with open(summary_file, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if data.get("id") == episode_id and data.get("scene_id") == scene_id:
                                video_rel_path = data.get("video", "")
                                break
                        except json.JSONDecodeError:
                            continue
        
        if not video_rel_path:
            return jsonify({"error": "Video path not found"}), 404
        
        VIDEO_PATH_CACHE[episode_key] = video_rel_path
    else:
        video_rel_path = VIDEO_PATH_CACHE[episode_key]
    
    video_path = os.path.join(TRAJECTORY_DIR, video_rel_path, "trajectory.mp4")
    if not os.path.exists(video_path):
        return jsonify({"error": "Video file not found"}), 404
    
    # Extract frame
    frame_bytes = extract_frame_from_video(video_path, frame_idx)
    if frame_bytes is None:
        return jsonify({"error": "Failed to extract frame"}), 500
    
    # Convert to base64
    frame_b64 = frame_to_base64(frame_bytes)
    return jsonify({"image": f"data:image/jpeg;base64,{frame_b64}"})


@app.route('/api/save', methods=['POST'])
def save_annotations():
    """Save updated annotations."""
    try:
        data = request.json
        episode_key = data.get("episode_key")
        updated_annotations = data.get("annotations", [])
        
        if episode_key not in EPISODES_DATA:
            return jsonify({"error": "Episode not found"}), 404
        
        episode_data = EPISODES_DATA[episode_key]
        num_frames = episode_data.get("num_frames", 0)
        
        # Update episode data
        updated_episode = update_episode_annotations(
            episode_data.copy(),
            updated_annotations,
            num_frames
        )
        
        # Update in-memory data
        EPISODES_DATA[episode_key] = updated_episode
        
        # Save to file
        save_subtask_determination(SUBTASK_DETERMINATION_FILE, EPISODES_DATA)
        
        return jsonify({
            "status": "success",
            "episode_key": episode_key,
            "subtask_sequence": updated_episode["subtask_sequence"],
            "subtask_counts": updated_episode["subtask_counts"]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def initialize_data(trajectory_dir: str, subtask_determination_file: str, subtask_splits_file: str):
    """Initialize global data structures."""
    global TRAJECTORY_DIR, SUBTASK_DETERMINATION_FILE, SUBTASK_SPLITS_FILE
    global EPISODES_DATA, SUBTASK_SPLITS
    
    TRAJECTORY_DIR = trajectory_dir
    SUBTASK_DETERMINATION_FILE = subtask_determination_file
    SUBTASK_SPLITS_FILE = subtask_splits_file
    
    print(f"Loading subtask determination from: {subtask_determination_file}")
    EPISODES_DATA = load_subtask_determination(subtask_determination_file)
    print(f"Loaded {len(EPISODES_DATA)} episodes")
    
    print(f"Loading subtask splits from: {subtask_splits_file}")
    SUBTASK_SPLITS = load_subtask_splits(subtask_splits_file)
    print(f"Loaded {len(SUBTASK_SPLITS)} subtask splits")


def main():
    parser = argparse.ArgumentParser(description="Web interface for keyframe annotation refinement")
    parser.add_argument("--trajectory_dir", type=str, default="data/trajectory_data/R2R",
                       help="Path to trajectory data directory (contains summary.json)")
    parser.add_argument("--subtask_determination_file", type=str,
                       default="streamvln/cot_data/subtask_determination.jsonl",
                       help="Path to subtask_determination.jsonl file")
    parser.add_argument("--subtask_splits_file", type=str,
                       default="streamvln/cot_data/subtask_splits.jsonl",
                       help="Path to subtask_splits.jsonl file")
    parser.add_argument("--port", type=int, default=5000,
                       help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to bind to (0.0.0.0 for server access)")
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute
    # Get the StreamVLN root directory (3 levels up from this file)
    # web_refinement -> cot_data -> streamvln -> StreamVLN
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    # Resolve paths and normalize them
    if os.path.isabs(args.trajectory_dir):
        trajectory_dir = os.path.normpath(os.path.abspath(args.trajectory_dir))
    else:
        trajectory_dir = os.path.normpath(os.path.abspath(os.path.join(base_dir, args.trajectory_dir)))
    
    if os.path.isabs(args.subtask_determination_file):
        subtask_determination_file = os.path.normpath(os.path.abspath(args.subtask_determination_file))
    else:
        subtask_determination_file = os.path.normpath(os.path.abspath(os.path.join(base_dir, args.subtask_determination_file)))
    
    if os.path.isabs(args.subtask_splits_file):
        subtask_splits_file = os.path.normpath(os.path.abspath(args.subtask_splits_file))
    else:
        subtask_splits_file = os.path.normpath(os.path.abspath(os.path.join(base_dir, args.subtask_splits_file)))
    
    # Print resolved paths for debugging
    print(f"\n{'='*80}")
    print("Path Resolution:")
    print(f"{'='*80}")
    print(f"Script directory: {script_dir}")
    print(f"Base directory: {base_dir}")
    print(f"Trajectory directory: {trajectory_dir}")
    print(f"  Exists: {os.path.exists(trajectory_dir)}")
    print(f"Subtask determination file: {subtask_determination_file}")
    print(f"  Exists: {os.path.exists(subtask_determination_file)}")
    print(f"Subtask splits file: {subtask_splits_file}")
    print(f"  Exists: {os.path.exists(subtask_splits_file)}")
    print(f"{'='*80}\n")
    
    # Initialize data
    initialize_data(trajectory_dir, subtask_determination_file, subtask_splits_file)
    
    print(f"\n{'='*80}")
    print("Web Interface for Keyframe Annotation Refinement")
    print(f"{'='*80}")
    print(f"\nServer starting on http://{args.host}:{args.port}")
    print(f"Access from your Mac at: http://<server-ip>:{args.port}")
    print(f"{'='*80}\n")
    
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()

