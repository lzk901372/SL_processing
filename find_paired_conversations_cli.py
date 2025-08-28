#!/usr/bin/env python3
"""
Cli version: Find and match all videos with the same Vxx_Sxx_Ixx format.
"""

import os
import glob
import re
import argparse
from typing import List, Tuple, Optional
from collections import defaultdict
from pathlib import Path
import json


def find_mp4_files(root_path: str) -> List[str]:
    """Recursively find all mp4 video files"""
    pattern = os.path.join(root_path, "**", "*.mp4")
    mp4_files = glob.glob(pattern, recursive=True)
    return sorted(mp4_files)


def extract_video_id(filename: str) -> Optional[str]:
    """Extract Vxx_Sxx_Ixx format from file name"""
    pattern = r'(V\d+_S\d+_I\d+)'
    match = re.search(pattern, filename)
    return match.group(1) if match else None


def find_paired_videos(video_files: List[str], auto_skip: bool = False) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Find paired video files"""
    id_to_files = defaultdict(list)
    
    for file_path in video_files:
        filename = os.path.basename(file_path)
        video_id = extract_video_id(filename)
        if video_id:
            id_to_files[video_id].append(file_path)
    
    paired_videos = []
    remaining_videos = []
    
    for video_id, files in id_to_files.items():
        if len(files) == 2:
            # Only two files. Pair success.
            paired_videos.append((files[0], files[1]))
            print(f"✓ Pair success: {video_id}")
        elif len(files) == 1:
            # Only one file. can't pair
            remaining_videos.extend(files)
            print(f"⚠ Can't pair: {video_id} (only one file)")
        else:
            # More than two files
            if auto_skip:
                print(f"⚠ Skipping {video_id} (has {len(files)} files, auto-skip)")
                remaining_videos.extend(files)
            else:
                print(f"❓ Needs attention: {video_id} (has {len(files)} files)")
                for i, file_path in enumerate(files, 1):
                    print(f"  {i}. {os.path.basename(file_path)}")
                
                # Ask user to skip or not
                while True:
                    try:
                        choice = input(f"Choose two files to be paired (input indexes like '1,2' or 'skip')").strip()
                        if choice.lower() == 'skip':
                            remaining_videos.extend(files)
                            print(f"Skipping {video_id}")
                            break
                        
                        selected_indices = [int(x.strip()) - 1 for x in choice.split(',')]
                        if len(selected_indices) == 2 and all(0 <= i < len(files) for i in selected_indices):
                            selected_files = [files[i] for i in selected_indices]
                            paired_videos.append((selected_files[0], selected_files[1]))
                            print(f"✓ Manual match success! {video_id}")
                            
                            for i, file_path in enumerate(files):
                                if i not in selected_indices:
                                    remaining_videos.append(file_path)
                            break
                        else:
                            print("Invalid choice. Please either input two indexes (e.g. '1,2') or 'skip'")
                    except (ValueError, IndexError):
                        print("Error input. Please either input two indexes (e.g. '1,2') or 'skip'")
    
    return paired_videos, remaining_videos


def save_paired_videos_to_txt(paired_videos: List[Tuple[str, str]], output_file: str):
    """Save pair results to txt"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Paired videos list\n")
        f.write("# Format: Video ID | File 1 ID | File 1 Path | File 2 ID | File 2 Path\n")
        f.write("=" * 80 + "\n\n")
        
        for file1, file2 in paired_videos:
            video_id = extract_video_id(os.path.basename(file1))
            f.write(f"{video_id} | {Path(file1).stem.split('_')[-1]} | {file1} | {Path(file2).stem.split('_')[-1]} | {file2}\n")
    
    print(f"✓ Pair results saved to: {output_file}")


def save_paired_videos_to_json(paired_videos: List[Tuple[str, str]], output_file: str):
    """save paired results to json"""
    json_content = {}
    for file1, file2 in paired_videos:
        video_id = extract_video_id(os.path.basename(file1))
        # json_content.append({
        #     "video_prefix": video_id,
        #     "file1_identity": Path(file1).stem.split('_')[-1],
        #     "file1_path": file1,
        #     "file2_identity": Path(file2).stem.split('_')[-1],
        #     "file2_path": file2
        # })
        json_content[video_id] = {
            Path(file1).stem.split('_')[-1]: file1,
            Path(file2).stem.split('_')[-1]: file2,
        }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_content, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description='Find and pair up video files with Vxx_Sxx_Ixx format')
    parser.add_argument('--directory', help='Search directory. Search is recursive', default='/mnt/425b37ab-69e4-4f4d-a719-d3f3d3a7c387/behavior_s2s_data/seamless_interaction_yllee/improvised/train')
    parser.add_argument('-o', '--output', help='Output directory. Default: paired_videos.txt)', default='/home/zliao/seamless_interaction/data/paired_new.txt')
    parser.add_argument('--auto-skip', action='store_true', help='Skip manually-confirmed pairs')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: Not found: {args.directory}")
        return 1
    
    print(f"Searching in {args.directory}")
    print("=" * 50)
    
    # Find all .mp4 video files
    mp4_files = find_mp4_files(args.directory)
    print(f"Found {len(mp4_files)} .mp4 video files")
    
    if not mp4_files:
        print("No video files were found")
        return 0
    
    # Show found videos
    print("\nFound videos:")
    for i, file_path in enumerate(mp4_files, 1):
        filename = os.path.basename(file_path)
        video_id = extract_video_id(filename)
        status = f"ID: {video_id}" if video_id else "Unidentifiable ID"
        print(f"{i:3d}. {filename} ({status})")
    
    print("\n" + "=" * 50)
    print("Start pairing...")
    
    # Finding pairs
    paired_videos, remaining_videos = find_paired_videos(mp4_files, args.auto_skip)
    
    print("\n" + "=" * 50)
    print("Pair result:")
    print(f"Paired videos: {len(paired_videos)} 对")
    print(f"Unpaired videos: {len(remaining_videos)}")
    
    if paired_videos:
        # Save pairs result
        if args.output:
            output_file = args.output
        else:
            output_file = os.path.join(args.directory, "paired_videos.txt")
        
        # save_paired_videos_to_txt(paired_videos, output_file)
        save_paired_videos_to_json(paired_videos, output_file.replace(".txt", ".json"))
        # Info for pair
        print("\nInfo for pair:")
        for i, (file1, file2) in enumerate(paired_videos, 1):
            video_id = extract_video_id(os.path.basename(file1))
            print(f"{i}. {video_id}")
            print(f"   {os.path.basename(file1)}")
            print(f"   {os.path.basename(file2)}")
    
    if remaining_videos:
        print(f"\nUnpaired videos num: ({len(remaining_videos)}):")
        for file_path in remaining_videos:
            filename = os.path.basename(file_path)
            video_id = extract_video_id(filename)
            print(f"  {filename} (ID: {video_id})")
    
    return 0


if __name__ == "__main__":
    exit(main())
