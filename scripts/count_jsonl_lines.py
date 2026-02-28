#!/usr/bin/env python3
import gzip
import json
import sys
import argparse

def count_jsonl_lines(filepath):
    """Count the number of JSON lines in a gzipped JSONL file."""
    count = 0
    error_count = 0
    
    print(f"Reading {filepath}...")
    
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                    count += 1
                except json.JSONDecodeError as e:
                    error_count += 1
                    print(f"Warning: Invalid JSON at line {line_num}: {str(e)[:100]}")
                
                # Print progress every 1000 lines
                if line_num % 1000 == 0:
                    print(f"  Processed {line_num} lines, {count} valid JSON objects...")
    
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    except gzip.BadGzipFile:
        print(f"Error: Not a valid gzip file: {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"\nResults:")
    print(f"  Total valid JSON objects: {count}")
    if error_count > 0:
        print(f"  Invalid JSON lines: {error_count}")
    
    return count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count JSON lines in a gzipped JSONL file")
    parser.add_argument("filepath", type=str, help="Path to the gzipped JSONL file")
    args = parser.parse_args()
    
    count = count_jsonl_lines(args.filepath)
    sys.exit(0)

