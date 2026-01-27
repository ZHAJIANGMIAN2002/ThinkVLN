#!/usr/bin/env python3
"""
Extract frames from videos and save them for SFT dataset creation.
"""

import json
import os
import cv2
from pathlib import Path
from typing import Dict, List
import argparse


def load_cot_dataset(cot_file: str) -> List[Dict]:
    """Load COT dataset from JSONL file."""
    data = []
    with open(cot_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_summary_mapping(summary_file: str) -> Dict[str, Dict]:
    """Load summary file to get video path mapping from episode_key."""
    mapping = {}
    with open(summary_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                episode_key = record.get('episode_key')
                if episode_key:
                    mapping[episode_key] = record
    return mapping


def extract_frame_from_video(video_path: str, frame_index: int):
    """
    Extract a specific frame from video.
    
    Args:
        video_path: Path to the video file
        frame_index: Index of the frame to extract (0-based)
    
    Returns:
        The frame (BGR format) or None if failed
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
        
        return frame
    except Exception as e:
        return None


def create_output_directory(output_dir: str, episode_key: str) -> str:
    """Create output directory for episode and return the path."""
    episode_dir = os.path.join(output_dir, episode_key)
    os.makedirs(episode_dir, exist_ok=True)
    return episode_dir


def process_cot_dataset(
    cot_file: str,
    summary_file: str,
    video_base_path: str,
    output_dir: str
) -> Dict:
    """
    Process COT dataset and extract frames.
    """
    
    # Load data
    print(f"Loading COT dataset from {cot_file}...")
    cot_data = load_cot_dataset(cot_file)
    print(f"Loaded {len(cot_data)} COT records")
    
    print(f"Loading summary mapping from {summary_file}...")
    summary_mapping = load_summary_mapping(summary_file)
    print(f"Loaded {len(summary_mapping)} video mappings")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Statistics
    stats = {
        'total': len(cot_data),
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'frame_mapping': []
    }
    
    # Process each COT record
    for idx, record in enumerate(cot_data, 1):
        episode_key = record.get('episode_key')
        step_id = record.get('step_id')
        
        if not episode_key or step_id is None:
            stats['skipped'] += 1
            continue
        
        # Get video info from summary
        if episode_key not in summary_mapping:
            stats['skipped'] += 1
            continue
        
        summary_record = summary_mapping[episode_key]
        video_name = summary_record.get('video')
        
        if not video_name:
            stats['skipped'] += 1
            continue
        
        # Construct video path
        video_path = os.path.join(video_base_path, video_name, 'trajectory.mp4')
        
        if not os.path.exists(video_path):
            print(f"[{idx}/{len(cot_data)}] Video not found: {video_path}")
            stats['failed'] += 1
            continue
        
        # Extract frame
        frame = extract_frame_from_video(video_path, step_id)
        if frame is None:
            print(f"[{idx}/{len(cot_data)}] Failed to extract frame {step_id}")
            stats['failed'] += 1
            continue
        
        # Save frame
        episode_dir = create_output_directory(output_dir, episode_key)
        frame_filename = f"{step_id:06d}.jpg"
        frame_path = os.path.join(episode_dir, frame_filename)
        
        try:
            cv2.imwrite(frame_path, frame)
            full_path = os.path.join(output_dir, episode_key, frame_filename)
            stats['frame_mapping'].append({
                'frame_key': record.get('frame_key'),
                'episode_key': episode_key,
                'step_id': step_id,
                'image_path': full_path
            })
            stats['success'] += 1
            if idx % 10 == 0:
                print(f"[{idx}/{len(cot_data)}] Progress: {stats['success']} extracted")
        except Exception as e:
            print(f"[{idx}/{len(cot_data)}] Failed to save frame: {e}")
            stats['failed'] += 1
    
    # Save frame mapping
    mapping_file = os.path.join(output_dir, 'frame_mapping.json')
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(stats['frame_mapping'], f, ensure_ascii=False, indent=2)
    
    # Print summary
    print(f"\n" + "="*50)
    print(f"Extraction Summary:")
    print(f"  Total records: {stats['total']}")
    print(f"  Successfully extracted: {stats['success']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Frame mapping saved to: {mapping_file}")
    print(f"="*50)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Extract frames from videos for SFT dataset')
    parser.add_argument('--cot-file', type=str, 
                       default='/home/swx/ThinkVLN/data/cot_dataset/cot_dataset_100_answer.jsonl',
                       help='Path to COT answer JSONL file')
    parser.add_argument('--summary-file', type=str,
                       default='/home/swx/ThinkVLN/data/trajectory_data/R2R_back/summary_full.jsonl',
                       help='Path to summary JSONL file')
    parser.add_argument('--video-base-path', type=str,
                       default='/home/swx/ThinkVLN/data/trajectory_data/R2R_back/',
                       help='Base path for videos')
    parser.add_argument('--output-dir', type=str,
                       default='/mnt/swx/dataset/sft-dataset',
                       help='Output directory for extracted frames')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.cot_file):
        print(f"Error: COT file not found: {args.cot_file}")
        return
    
    if not os.path.exists(args.summary_file):
        print(f"Error: Summary file not found: {args.summary_file}")
        return
    
    if not os.path.exists(args.video_base_path):
        print(f"Error: Video base path not found: {args.video_base_path}")
        return
    
    # Process dataset
    process_cot_dataset(
        args.cot_file,
        args.summary_file,
        args.video_base_path,
        args.output_dir
    )


if __name__ == '__main__':
    main()


