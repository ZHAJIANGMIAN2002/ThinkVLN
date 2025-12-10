#!/usr/bin/env python3
import gzip
import json
import sys
import argparse

def count_json_entries(filepath):
    """Count JSON entries in a gzipped JSON file."""
    
    print(f"Reading {filepath}...")
    
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            data = json.load(f)
    
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    except gzip.BadGzipFile:
        print(f"Error: Not a valid gzip file: {filepath}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"File: {filepath}")
    print(f"{'='*60}")
    print(f"\nRoot type: {type(data).__name__}")
    
    if isinstance(data, list):
        print(f"Total entries: {len(data)}")
        if len(data) > 0:
            print(f"First entry type: {type(data[0]).__name__}")
            if isinstance(data[0], dict):
                print(f"First entry keys: {list(data[0].keys())}")
    
    elif isinstance(data, dict):
        print(f"Top-level keys: {list(data.keys())}")
        print(f"\nDetailed breakdown:")
        
        total_count = 0
        for key, value in data.items():
            if isinstance(value, list):
                count = len(value)
                total_count += count
                print(f"  - '{key}': list with {count} entries")
                if count > 0 and isinstance(value[0], dict):
                    sample_keys = list(value[0].keys())
                    print(f"    Sample entry keys: {sample_keys}")
            elif isinstance(value, dict):
                print(f"  - '{key}': dict with {len(value)} keys")
                print(f"    Keys: {list(value.keys())}")
            else:
                print(f"  - '{key}': {type(value).__name__} = {str(value)[:100]}")
        
        # Check if there's a main "episodes" or similar array
        if 'episodes' in data and isinstance(data['episodes'], list):
            print(f"\n{'='*60}")
            print(f"Total JSON content entries: {len(data['episodes'])}")
            print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count JSON entries in a gzipped JSON file")
    parser.add_argument("filepath", type=str, help="Path to the gzipped JSON file")
    args = parser.parse_args()
    
    count_json_entries(args.filepath)

