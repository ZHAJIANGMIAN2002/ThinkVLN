#!/usr/bin/env python3
import gzip
import json
import sys
import argparse

def inspect_json_file(filepath):
    """Inspect a gzipped JSON file and count entries if it's an array or object."""
    
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
    
    print(f"\nFile structure:")
    print(f"  Root type: {type(data).__name__}")
    
    if isinstance(data, list):
        print(f"  Total entries: {len(data)}")
        if len(data) > 0:
            print(f"  First entry type: {type(data[0]).__name__}")
            print(f"  First entry keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'N/A'}")
    
    elif isinstance(data, dict):
        print(f"  Top-level keys: {list(data.keys())}")
        
        # Try to find arrays in the dict
        for key, value in data.items():
            if isinstance(value, list):
                print(f"  Key '{key}': list with {len(value)} entries")
                if len(value) > 0 and isinstance(value[0], dict):
                    print(f"    - First entry keys: {list(value[0].keys())}")
            elif isinstance(value, dict):
                print(f"  Key '{key}': dict with keys {list(value.keys())}")
            else:
                print(f"  Key '{key}': {type(value).__name__}")
    
    else:
        print(f"  Type: {type(data).__name__}")
        print(f"  Value: {str(data)[:200]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a gzipped JSON file structure")
    parser.add_argument("filepath", type=str, help="Path to the gzipped JSON file")
    args = parser.parse_args()
    
    inspect_json_file(args.filepath)

