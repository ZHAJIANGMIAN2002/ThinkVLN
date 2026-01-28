#!/usr/bin/env python3
"""
Split R2R keyframe images into RGB and Map parts.

Each image is split horizontally:
- Left part: Fixed 640x480 RGB image
- Right part: Dynamic width Map image (remaining part)

Output files are saved in the same directory as input:
- Original: 000025.jpg
- Output: 000025_rgb.jpg and 000025_map.jpg

Usage:
    python split_r2r_images.py --input_dir /mnt/nvme/swx/dataset/R2R_keyframe --demo_only --demo_by_size
"""

import argparse
import os
from pathlib import Path
import cv2
import numpy as np


def check_image_sizes(input_dir, max_samples=100):
    """Check image sizes in the dataset and return size groups"""
    sizes = {}
    count = 0
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        height, width = img.shape[:2]
                        size = (width, height)  # (width, height) format
                        if size not in sizes:
                            sizes[size] = []
                        sizes[size].append(img_path)
                        count += 1
                        if count >= max_samples:
                            break
                except Exception as e:
                    print(f"Error reading {img_path}: {e}")
        if count >= max_samples:
            break
    
    print("=" * 60)
    print("Image Size Statistics:")
    print("=" * 60)
    for size, paths in sorted(sizes.items()):
        print(f"Size {size[0]}x{size[1]} (width x height): {len(paths)} images")
        if len(paths) <= 3:
            for p in paths:
                print(f"  - {p}")
        else:
            print(f"  - {paths[0]}")
            print(f"  - ... ({len(paths)-1} more)")
    print("=" * 60)
    
    return sizes


def split_image(img_path, output_rgb_path, output_map_path, fixed_rgb_width=640, map_left_crop=0):
    """Split an image: left fixed 640x480 RGB, right dynamic Map with left crop"""
    try:
        # Read image using cv2 (BGR format)
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        
        height, width = img.shape[:2]
        
        # Check if image is wide enough
        if width < fixed_rgb_width + map_left_crop:
            raise ValueError(f"Image width {width} is less than fixed RGB width {fixed_rgb_width} + map left crop {map_left_crop}")
        
        # Fixed left part (RGB): 640x480
        rgb_img = img[:, :fixed_rgb_width]
        
        # Dynamic right part (Map): remaining width, with left crop
        map_start = fixed_rgb_width + map_left_crop
        map_img = img[:, map_start:]
        
        # Save both parts
        cv2.imwrite(output_rgb_path, rgb_img)
        cv2.imwrite(output_map_path, map_img)
        
        rgb_size = (fixed_rgb_width, height)
        map_size = (width - map_start, height)
        
        return True, rgb_size, map_size
    except Exception as e:
        print(f"Error splitting {img_path}: {e}")
        return False, None, None


def process_directory(input_dir, output_dir=None, demo_only=False, demo_count=1, demo_by_size=False, fixed_rgb_width=640, map_left_crop=0, delete_original=False):
    """Process all images in the directory"""
    input_path = Path(input_dir)
    
    # If output_dir is not specified, use input_dir (in-place processing)
    if output_dir is None:
        output_path = input_path
    else:
        output_path = Path(output_dir)
    
    # Find all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(input_path.rglob(ext))
    
    print(f"Found {len(image_files)} image files")
    
    # Select demo images by size if requested
    if demo_only and demo_by_size:
        sizes = check_image_sizes(input_dir, max_samples=200)
        demo_files = []
        for size, paths in sorted(sizes.items()):
            if len(paths) > 0:
                demo_files.append(paths[0])  # Take one from each size
                if len(demo_files) >= demo_count:
                    break
        image_files = demo_files
        print(f"\nSelected {len(demo_files)} demo images from different sizes:")
        for f in demo_files:
            img = cv2.imread(f)
            if img is not None:
                h, w = img.shape[:2]
                print(f"  - {Path(f).relative_to(input_path)}: {w}x{h}")
    elif demo_only:
        image_files = image_files[:demo_count]
    
    # Process each image
    processed = 0
    failed = 0
    
    for img_file in image_files:
        # Get file info
        img_path = Path(img_file)
        file_stem = img_path.stem  # e.g., "000025"
        file_ext = img_path.suffix  # e.g., ".jpg"
        
        # Create output paths in the same directory
        if output_dir is None:
            # In-place: save in same directory as input
            output_rgb_path = img_path.parent / f"{file_stem}_rgb{file_ext}"
            output_map_path = img_path.parent / f"{file_stem}_map{file_ext}"
        else:
            # Preserve directory structure in output_dir
            rel_path = img_path.relative_to(input_path)
            output_rgb_path = output_path / rel_path.parent / f"{file_stem}_rgb{file_ext}"
            output_map_path = output_path / rel_path.parent / f"{file_stem}_map{file_ext}"
            output_rgb_path.parent.mkdir(parents=True, exist_ok=True)
            output_map_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Split the image
        success, rgb_size, map_size = split_image(
            str(img_file), 
            str(output_rgb_path), 
            str(output_map_path),
            fixed_rgb_width=fixed_rgb_width,
            map_left_crop=map_left_crop
        )
        
        if success:
            processed += 1
            rel_path = img_path.relative_to(input_path) if output_dir else img_path.name
            
            # Delete original image if requested and both output files exist
            if delete_original:
                if output_rgb_path.exists() and output_map_path.exists():
                    try:
                        img_path.unlink()
                        print(f"✓ Split {rel_path} (original deleted)")
                    except Exception as e:
                        print(f"⚠ Warning: Failed to delete original {rel_path}: {e}")
                        print(f"✓ Split {rel_path}")
                else:
                    print(f"⚠ Warning: Output files missing, keeping original {rel_path}")
                    print(f"✓ Split {rel_path}")
            else:
                print(f"✓ Split {rel_path}")
            
            if processed <= 5 or demo_only:  # Print details for first 5 or all in demo mode
                print(f"  RGB: {rgb_size[0]}x{rgb_size[1]}, Map: {map_size[0]}x{rgb_size[1]}")
                print(f"  Output: {output_rgb_path.name}, {output_map_path.name}")
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Processing complete!")
    print(f"  Processed: {processed}")
    print(f"  Failed: {failed}")
    if output_dir:
        print(f"  Output directory: {output_path}")
    else:
        print(f"  Files saved in-place (same directory as input)")
    print("=" * 60)
    
    return processed, failed


def main():
    parser = argparse.ArgumentParser(description="Split R2R keyframe images into RGB (640x480) and Map parts")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Input directory containing R2R keyframe images")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: same as input_dir, in-place)")
    parser.add_argument("--check_only", action="store_true",
                       help="Only check image sizes, don't split")
    parser.add_argument("--demo_only", action="store_true",
                       help="Process only demo image(s) for verification")
    parser.add_argument("--demo_count", type=int, default=1,
                       help="Number of images to process in demo mode (default: 1)")
    parser.add_argument("--demo_by_size", action="store_true",
                       help="Select demo images from different sizes (use with --demo_only)")
    parser.add_argument("--fixed_rgb_width", type=int, default=640,
                       help="Fixed width for RGB part (default: 640)")
    parser.add_argument("--map_left_crop", type=int, default=0,
                       help="Number of pixels to crop from left side of Map part (default: 0)")
    parser.add_argument("--delete_original", action="store_true",
                       help="Delete original images after successful splitting (default: False)")
    
    args = parser.parse_args()
    
    # Check image sizes first
    print("Checking image sizes...")
    sizes = check_image_sizes(args.input_dir, max_samples=200)
    
    if args.check_only:
        print("\nCheck-only mode: Exiting without splitting images.")
        return
    
    # Process images
    print(f"\nStarting image splitting...")
    print(f"Input directory: {args.input_dir}")
    if args.output_dir:
        print(f"Output directory: {args.output_dir}")
    else:
        print(f"Output: In-place (same directory as input)")
    print(f"Fixed RGB width: {args.fixed_rgb_width}x480")
    if args.map_left_crop > 0:
        print(f"Map left crop: {args.map_left_crop} pixels")
    if args.delete_original:
        print(f"⚠️  WARNING: Original images will be DELETED after successful splitting!")
    
    if args.demo_only:
        if args.demo_by_size:
            print(f"\n⚠️  DEMO MODE: Processing {args.demo_count} image(s) from different sizes")
        else:
            print(f"\n⚠️  DEMO MODE: Only processing {args.demo_count} image(s) for verification")
        print("   Remove --demo_only flag to process all images")
    
    process_directory(
        args.input_dir, 
        args.output_dir,
        demo_only=args.demo_only, 
        demo_count=args.demo_count,
        demo_by_size=args.demo_by_size,
        fixed_rgb_width=args.fixed_rgb_width,
        map_left_crop=args.map_left_crop,
        delete_original=args.delete_original
    )


if __name__ == "__main__":
    main()
