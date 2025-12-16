#!/usr/bin/env python3
"""
Extract frames from videos and save them for SFT dataset creation.
Supports concurrent extraction using multiple threads.
"""

import json
import os
import cv2
from pathlib import Path
from typing import Dict, List
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


# Thread-safe lock for writing statistics
stats_lock = Lock()


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


def process_episode(
    episode_key: str,
    records: List[Dict],
    summary_mapping: Dict,
    video_base_path: str,
    output_dir: str,
    stats_lock: Lock
) -> Dict:
    """
    Process all records for a single episode.
    
    Args:
        episode_key: The episode identifier
        records: List of records for this episode
        summary_mapping: Mapping from episode_key to video info
        video_base_path: Base path for videos
        output_dir: Output directory for frames
        stats_lock: Thread-safe lock for statistics
    
    Returns:
        Dictionary with success/failure counts and frame mappings
    """
    local_stats = {
        'episode_key': episode_key,
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'frame_mapping': []
    }
    
    # Get video info from summary
    if episode_key not in summary_mapping:
        local_stats['skipped'] = len(records)
        return local_stats
    
    summary_record = summary_mapping[episode_key]
    video_name = summary_record.get('video')
    
    if not video_name:
        local_stats['skipped'] = len(records)
        return local_stats
    
    # Construct video path
    video_path = os.path.join(video_base_path, video_name, 'trajectory.mp4')
    
    if not os.path.exists(video_path):
        print(f"Video not found for episode {episode_key}: {video_path}")
        local_stats['failed'] = len(records)
        return local_stats
    
    # Process each record for this episode
    for record in records:
        step_id = record.get('step_id')
        
        if step_id is None:
            local_stats['skipped'] += 1
            continue
        
        # Extract frame
        frame = extract_frame_from_video(video_path, step_id)
        if frame is None:
            local_stats['failed'] += 1
            continue
        
        # Save frame
        episode_dir = create_output_directory(output_dir, episode_key)
        frame_filename = f"{step_id:06d}.jpg"
        frame_path = os.path.join(episode_dir, frame_filename)
        
        try:
            cv2.imwrite(frame_path, frame)
            full_path = os.path.join(output_dir, episode_key, frame_filename)
            local_stats['frame_mapping'].append({
                'frame_key': record.get('frame_key'),
                'episode_key': episode_key,
                'step_id': step_id,
                'image_path': full_path
            })
            local_stats['success'] += 1
        except Exception as e:
            print(f"Failed to save frame for episode {episode_key}, step {step_id}: {e}")
            local_stats['failed'] += 1
    
    return local_stats


def process_cot_dataset(
    cot_file: str,
    summary_file: str,
    video_base_path: str,
    output_dir: str,
    num_workers: int = 8
) -> Dict:
    """
    Process COT dataset and extract frames using concurrent workers.
    
    Args:
        cot_file: Path to COT JSONL file
        summary_file: Path to summary JSONL file
        video_base_path: Base path for videos
        output_dir: Output directory for frames
        num_workers: Number of concurrent threads
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
    
    # Group records by episode_key
    episode_records = {}
    for record in cot_data:
        episode_key = record.get('episode_key')
        if episode_key:
            if episode_key not in episode_records:
                episode_records[episode_key] = []
            episode_records[episode_key].append(record)
    
    print(f"Processing {len(episode_records)} episodes with {num_workers} workers...")
    
    # Process episodes concurrently
    all_frame_mappings = []
    total_stats = {
        'total_episodes': len(episode_records),
        'total_records': len(cot_data),
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'completed_episodes': 0
    }
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                process_episode,
                episode_key,
                records,
                summary_mapping,
                video_base_path,
                output_dir,
                stats_lock
            ): episode_key
            for episode_key, records in episode_records.items()
        }
        
        # Process completed tasks
        completed = 0
        for future in as_completed(futures):
            episode_key = futures[future]
            try:
                result = future.result()
                all_frame_mappings.extend(result['frame_mapping'])
                total_stats['success'] += result['success']
                total_stats['failed'] += result['failed']
                total_stats['skipped'] += result['skipped']
                total_stats['completed_episodes'] += 1
                
                completed += 1
                if completed % max(1, len(episode_records) // 10) == 0:
                    print(f"Progress: {completed}/{len(episode_records)} episodes completed "
                          f"({total_stats['success']} frames extracted)")
            except Exception as e:
                print(f"Error processing episode {episode_key}: {e}")
                total_stats['failed'] += len(episode_records[episode_key])
    
    # Save frame mapping
    mapping_file = os.path.join(output_dir, 'frame_mapping.json')
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(all_frame_mappings, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print(f"\n" + "="*60)
    print(f"Extraction Summary:")
    print(f"  Total episodes: {total_stats['total_episodes']}")
    print(f"  Total records: {total_stats['total_records']}")
    print(f"  Successfully extracted: {total_stats['success']}")
    print(f"  Failed: {total_stats['failed']}")
    print(f"  Skipped: {total_stats['skipped']}")
    print(f"  Frame mapping saved to: {mapping_file}")
    print(f"="*60)
    
    return total_stats


def main():
    parser = argparse.ArgumentParser(description='Extract frames from videos for SFT dataset')
    parser.add_argument('--cot-file', type=str, 
                       default='/home/swx/ThinkVLN/data/cot_dataset/cot_dataset_100_answer.jsonl',
                       help='Path to COT answer JSONL file')
    parser.add_argument('--summary-file', type=str,
                       default='/home/swx/ThinkVLN/data/trajectory_data/R2R_back/summary_full.jsonl',
                       help='Path to summary JSONL file')
    parser.add_argument('--video-base-path', type=str,
                       default='/home/swx/ThinkVLN/data/trajectory_data/R2R_back/images',
                       help='Base path for videos')
    parser.add_argument('--output-dir', type=str,
                       default='/mnt/swx/dataset/sft-dataset',
                       help='Output directory for extracted frames')
    parser.add_argument('--num-workers', type=int,
                       default=8,
                       help='Number of concurrent worker threads')
    
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
        args.output_dir,
        args.num_workers
    )


def verify_extraction(
    cot_file: str,
    output_dir: str,
    frame_mapping_file: str
) -> Dict:
    """
    Verify that all frames have been extracted correctly.
    
    Args:
        cot_file: Path to original COT file
        output_dir: Output directory where frames were saved
        frame_mapping_file: Path to frame_mapping.json
    
    Returns:
        Verification statistics
    """
    print(f"\n" + "="*60)
    print("CONSISTENCY CHECK - Verifying Frame Extraction")
    print("="*60)
    
    # Load original COT data
    print("Loading original COT data...")
    cot_data = load_cot_dataset(cot_file)
    print(f"Total COT records: {len(cot_data)}")
    
    # Load frame mapping
    print("Loading frame mapping...")
    try:
        with open(frame_mapping_file, 'r', encoding='utf-8') as f:
            frame_mapping = json.load(f)
        print(f"Total frame mappings: {len(frame_mapping)}")
    except Exception as e:
        print(f"Error loading frame mapping: {e}")
        return {'success': False, 'error': str(e)}
    
    # Create a set of frame_keys from mapping for quick lookup
    mapped_frame_keys = {fm['frame_key'] for fm in frame_mapping}
    
    # Check 1: All COT records have corresponding mappings
    print("\n[Check 1] Verifying all COT records have frame mappings...")
    missing_mappings = []
    for record in cot_data:
        frame_key = record.get('frame_key')
        if frame_key not in mapped_frame_keys:
            missing_mappings.append(frame_key)
    
    if missing_mappings:
        print(f"  ❌ Found {len(missing_mappings)} COT records without frame mappings:")
        for fk in missing_mappings[:5]:
            print(f"    - {fk}")
        if len(missing_mappings) > 5:
            print(f"    ... and {len(missing_mappings) - 5} more")
    else:
        print(f"  ✓ All {len(cot_data)} COT records have frame mappings")
    
    # Check 2: Verify frame files exist
    print("\n[Check 2] Verifying frame files exist...")
    missing_files = []
    for fm in frame_mapping:
        image_path = fm['image_path']
        if not os.path.exists(image_path):
            missing_files.append((fm['frame_key'], image_path))
    
    if missing_files:
        print(f"  ❌ Found {len(missing_files)} missing frame files:")
        for frame_key, path in missing_files[:5]:
            print(f"    - {frame_key}: {path}")
        if len(missing_files) > 5:
            print(f"    ... and {len(missing_files) - 5} more")
    else:
        print(f"  ✓ All {len(frame_mapping)} frame files exist")
    
    # Check 3: Verify frame files are valid images
    print("\n[Check 3] Verifying frame files are valid images...")
    invalid_frames = []
    for fm in frame_mapping:
        image_path = fm['image_path']
        try:
            # Try to read the image with OpenCV
            img = cv2.imread(image_path)
            if img is None:
                invalid_frames.append((fm['frame_key'], image_path, "Cannot read with cv2.imread"))
        except Exception as e:
            invalid_frames.append((fm['frame_key'], image_path, str(e)))
    
    if invalid_frames:
        print(f"  ❌ Found {len(invalid_frames)} invalid frame files:")
        for frame_key, path, reason in invalid_frames[:5]:
            print(f"    - {frame_key}: {reason}")
        if len(invalid_frames) > 5:
            print(f"    ... and {len(invalid_frames) - 5} more")
    else:
        print(f"  ✓ All {len(frame_mapping)} frame files are valid images")
    
    # Check 4: Verify frame counts match
    print("\n[Check 4] Comparing counts...")
    cot_count = len(cot_data)
    mapping_count = len(frame_mapping)
    
    if cot_count == mapping_count:
        print(f"  ✓ Counts match: {cot_count} COT records = {mapping_count} frame mappings")
    else:
        print(f"  ❌ Count mismatch:")
        print(f"    - COT records: {cot_count}")
        print(f"    - Frame mappings: {mapping_count}")
        print(f"    - Difference: {abs(cot_count - mapping_count)}")
    
    # Check 5: Verify all episodes have their directories
    print("\n[Check 5] Verifying episode directories...")
    episodes_in_mapping = {fm['episode_key'] for fm in frame_mapping}
    episodes_with_files = set()
    
    if os.path.exists(output_dir):
        for episode_dir in os.listdir(output_dir):
            episode_path = os.path.join(output_dir, episode_dir)
            if os.path.isdir(episode_path) and episode_dir != '__pycache__':
                episodes_with_files.add(episode_dir)
    
    missing_dirs = episodes_in_mapping - episodes_with_files
    if missing_dirs:
        print(f"  ❌ Found {len(missing_dirs)} episodes in mapping but missing directories")
    else:
        print(f"  ✓ All {len(episodes_in_mapping)} episode directories exist")
    
    # Summary
    print("\n" + "="*60)
    all_checks_passed = (
        len(missing_mappings) == 0 and
        len(missing_files) == 0 and
        len(invalid_frames) == 0 and
        cot_count == mapping_count and
        len(missing_dirs) == 0
    )
    
    if all_checks_passed:
        print("✓ ALL CONSISTENCY CHECKS PASSED!")
        print(f"  Successfully extracted {mapping_count} frames")
        print(f"  All files are valid and accessible")
    else:
        print("❌ CONSISTENCY CHECKS FAILED!")
        print(f"  Issues found:")
        print(f"    - Missing mappings: {len(missing_mappings)}")
        print(f"    - Missing files: {len(missing_files)}")
        print(f"    - Invalid frames: {len(invalid_frames)}")
        print(f"    - Count mismatches: {abs(cot_count - mapping_count)}")
        print(f"    - Missing directories: {len(missing_dirs)}")
    print("="*60)
    
    return {
        'success': all_checks_passed,
        'total_cot_records': cot_count,
        'total_frame_mappings': mapping_count,
        'missing_mappings': len(missing_mappings),
        'missing_files': len(missing_files),
        'invalid_frames': len(invalid_frames),
        'missing_directories': len(missing_dirs),
        'episodes': len(episodes_in_mapping)
    }


def main():
    parser = argparse.ArgumentParser(description='Extract frames from videos for SFT dataset')
    parser.add_argument('--cot-file', type=str, 
                       default='/home/swx/ThinkVLN/data/cot_dataset/cot_dataset_100_answer.jsonl',
                       help='Path to COT answer JSONL file')
    parser.add_argument('--summary-file', type=str,
                       default='/home/swx/ThinkVLN/data/trajectory_data/R2R_back/summary_full.jsonl',
                       help='Path to summary JSONL file')
    parser.add_argument('--video-base-path', type=str,
                       default='/home/swx/ThinkVLN/data/trajectory_data/R2R_back/images',
                       help='Base path for videos')
    parser.add_argument('--output-dir', type=str,
                       default='/mnt/swx/dataset/sft-dataset',
                       help='Output directory for extracted frames')
    parser.add_argument('--num-workers', type=int,
                       default=8,
                       help='Number of concurrent worker threads')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only run verification, skip extraction')
    
    args = parser.parse_args()
    
    # If verify-only mode
    if args.verify_only:
        mapping_file = os.path.join(args.output_dir, 'frame_mapping.json')
        verify_extraction(args.cot_file, args.output_dir, mapping_file)
        return
    
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
        args.output_dir,
        args.num_workers
    )
    
    # Run verification after extraction
    print("\nRunning verification checks...")
    mapping_file = os.path.join(args.output_dir, 'frame_mapping.json')
    verify_extraction(args.cot_file, args.output_dir, mapping_file)


if __name__ == '__main__':
    main()
