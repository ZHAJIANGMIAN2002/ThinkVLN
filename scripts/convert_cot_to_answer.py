#!/usr/bin/env python3
"""
Convert COT (Chain of Thought) dataset to format with only the final answer.
Removes the thinking process and keeps only the structured answer format.
"""

import json
import re
import sys
from pathlib import Path


def extract_action(cot_text):
    """
    Extract the final [action] from the COT text.
    """
    # Look for [action] section
    pattern = r'\[action\]\s*([^\n\[\]]+)'
    match = re.search(pattern, cot_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_structured_answer(cot_text):
    """
    Extract the structured answer part from COT text.
    First removes all content before </think> (thinking process),
    then extracts the structured answer part.
    
    Matches the format:
    [localization]
    ...
    [subtask determination]
    ...
    [causal observation]
    ...
    [reason]
    ...
    [action]
    ...
    """
    # First, remove everything before </think> to avoid matching within thinking process
    think_end_pattern = r'</\s*think\s*>'
    think_end_match = re.search(think_end_pattern, cot_text, re.IGNORECASE)
    
    if think_end_match:
        # Remove everything up to and including </think>
        cot_text = cot_text[think_end_match.end():]
    
    # Now find the start of structured answer (first [localization])
    start_pattern = r'\[localization\]'
    start_match = re.search(start_pattern, cot_text, re.IGNORECASE)
    
    if not start_match:
        return None
    
    structured_part = cot_text[start_match.start():]
    
    return structured_part.strip()


def convert_cot_file(input_file, output_file):
    """
    Convert JSONL file with COT to a file with only answers.
    """
    converted_count = 0
    empty_cot_count = 0
    error_count = 0
    answer_lengths = []
    long_answers = []  # Store answers longer than 2000 characters
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())
                cot = data.get('cot', '')
                
                # Remove cot field to reduce file size
                if 'cot' in data:
                    del data['cot']
                
                if not cot or not cot.strip():
                    empty_cot_count += 1
                    # Write original data if COT is empty
                    data['answer'] = None
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    continue
                
                # Extract structured answer
                answer = extract_structured_answer(cot)
                
                if answer:
                    data['answer'] = answer
                    converted_count += 1
                    answer_len = len(answer)
                    answer_lengths.append(answer_len)
                    
                    # Track long answers (might indicate extraction issues)
                    if answer_len > 2000:
                        long_answers.append({
                            'line': line_num,
                            'length': answer_len,
                            'preview': answer[:200] + '...' if len(answer) > 200 else answer,
                            'episode_key': data.get('episode_key', 'unknown')
                        })
                else:
                    data['answer'] = None
                    error_count += 1
                
                # Write the modified data (without cot field)
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON at line {line_num}: {e}", file=sys.stderr)
                error_count += 1
            except Exception as e:
                print(f"Error processing line {line_num}: {e}", file=sys.stderr)
                error_count += 1
    
    print(f"\nConversion Summary:")
    print(f"  Successfully converted: {converted_count}")
    print(f"  Empty COT entries: {empty_cot_count}")
    print(f"  Errors: {error_count}")
    print(f"  Output file: {output_file}")
    
    # Statistics about answer lengths
    if answer_lengths:
        print(f"\nAnswer Length Statistics:")
        print(f"  Total answers: {len(answer_lengths)}")
        print(f"  Min length: {min(answer_lengths)}")
        print(f"  Max length: {max(answer_lengths)}")
        print(f"  Average length: {sum(answer_lengths) / len(answer_lengths):.1f}")
        print(f"  Median length: {sorted(answer_lengths)[len(answer_lengths) // 2]}")
        
        # Count by length ranges
        ranges = [
            (0, 500, "0-500"),
            (500, 1000, "500-1000"),
            (1000, 2000, "1000-2000"),
            (2000, 5000, "2000-5000"),
            (5000, float('inf'), "5000+")
        ]
        print(f"\n  Length distribution:")
        for min_len, max_len, label in ranges:
            count = sum(1 for l in answer_lengths if min_len <= l < max_len)
            if count > 0:
                print(f"    {label}: {count} ({count/len(answer_lengths)*100:.1f}%)")
    
    # Report long answers
    if long_answers:
        print(f"\n⚠️  Found {len(long_answers)} answers longer than 2000 characters:")
        print(f"  (These might indicate extraction issues or non-standard format)\n")
        for item in long_answers[:10]:  # Show first 10
            print(f"  Line {item['line']} (episode: {item['episode_key']}):")
            print(f"    Length: {item['length']} characters")
            print(f"    Preview: {item['preview']}")
            print()
        if len(long_answers) > 10:
            print(f"  ... and {len(long_answers) - 10} more long answers")
    else:
        print(f"\n✓ No unusually long answers found (all < 2000 characters)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_cot_to_answer.py <input_jsonl_file> [output_jsonl_file]")
        print("\nExample:")
        print("  python convert_cot_to_answer.py data/cot_dataset/cot_dataset_100.jsonl")
        print("  python convert_cot_to_answer.py data/cot_dataset/cot_dataset_100.jsonl data/cot_dataset/cot_dataset_100_answer.jsonl")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)
    
    # Generate output filename if not provided
    if len(sys.argv) >= 3:
        output_file = Path(sys.argv[2])
    else:
        # Add '_answer' before .jsonl extension
        output_file = input_file.parent / f"{input_file.stem}_answer.jsonl"
    
    print(f"Converting: {input_file}")
    print(f"Output: {output_file}")
    print()
    
    convert_cot_file(str(input_file), str(output_file))


if __name__ == '__main__':
    main()

