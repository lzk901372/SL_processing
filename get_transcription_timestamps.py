import argparse
import json
import os
from tqdm import tqdm
from typing import List
from glob import glob
from pathlib import Path
import msgpack


paths = {}

def gather_video_paths(video_paths, output_dir):
    for video_path in video_paths:
        if video_path.endswith((".mp4", ".mov", ".MOV")):
            video_key = Path(video_path).stem
            video_input = video_path
            video_output = os.path.join(output_dir, Path(video_path).name)
            paths[video_key] = {
                "input": video_input,
                "output": video_output,
            }


def get_all_mp4_files_glob(directory_path: str) -> List[str]:
    """
    Recursively find all .mp4 files using the glob module.
    
    Args:
        directory_path (str): The root directory path to search in
        
    Returns:
        List[str]: A list of absolute paths to all .mp4 files found
        
    Raises:
        FileNotFoundError: If the directory doesn't exist
        PermissionError: If there's no permission to access the directory
    """
    # Check if directory exists
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory does not exist: {directory_path}")
    
    if not os.path.isdir(directory_path):
        raise ValueError(f"Path is not a directory: {directory_path}")
    
    # Use glob with recursive pattern to find all .mp4 files
    # **/ means search recursively in all subdirectories
    pattern = os.path.join(directory_path, "**", "*.mp4")
    mp4_files = glob(pattern, recursive=True)
    
    # Convert to absolute paths
    mp4_files = [os.path.abspath(file_path) for file_path in mp4_files]
    json_files = [file_path.replace(".mp4", ".json") for file_path in mp4_files]
    npz_files = [file_path.replace(".mp4", ".npz") for file_path in mp4_files]
    wav_files = [file_path.replace(".mp4", ".wav") for file_path in mp4_files]
    
    return mp4_files, json_files, npz_files, wav_files


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", 
                        type=str, 
                        default='/mnt/425b37ab-69e4-4f4d-a719-d3f3d3a7c387/behavior_s2s_data/seamless_interaction_yllee/improvised/train',)
    parser.add_argument("--paired_file",
                        type=str,
                        default='/home/zliao/seamless_interaction/data/paired.json')
    parser.add_argument("--output_dir",
                        type=str,
                        default='/home/zliao/seamless_interaction/data/segments/',
                        # default='/mnt/71e9e143-c56b-4941-a3da-cfc6536c2150/data/seamless_interaction_processed',
                        )
    parser.add_argument("--process_temp_dir",
                        type=str,
                        default='/tmp/pymp-_aaaaaa/')
    
    parser.add_argument("--source_sr",
                        type=int,
                        default=48000)
    parser.add_argument("--target_sr",
                        type=int,
                        default=16000)
    
    parser.add_argument("--min_segment_duration",
                        type=float,
                        default=2.5)
    parser.add_argument("--timestamp_margin",
                        type=float,
                        default=1)
    return parser.parse_args()


# Example usage
if __name__ == "__main__":
    args = parse_arguments()
    
    current_dir = args.input_dir
    output_dir = args.output_dir
    source_sr = args.source_sr
    target_sr = args.target_sr
    min_segment_duration = args.min_segment_duration
    timestamp_margin = args.timestamp_margin
    
    with open(args.paired_file, "r") as f:
        paired_videos = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Using glob version
    mp4_files, _, _, _ = get_all_mp4_files_glob(current_dir)
    
    transcription_timestamps_mp_dumper = open(os.path.join(output_dir, "/home/zliao/seamless_interaction/data/transcription_timestamps.mp"), "wb")
    all_file_listening_timestamps = []
    
    for paired_video in tqdm(paired_videos):
        for mp4_file in [paired_video["file1_path"], paired_video["file2_path"]]:
            
            json_file = mp4_file.replace(".mp4", ".json")
            with open(json_file, "r") as f:
                data = json.load(f)
            transcriptions = data["metadata:transcript"]
            transcriptions_timestamps = [
                {"start": transcription["start"], "end": transcription["end"]}
                for transcription in transcriptions
            ]
            del data
            
            listening_segments = []
            for last, next in zip(transcriptions_timestamps[:-1], transcriptions_timestamps[1:]):
                last_end = last["end"]
                next_start = next["start"]
                if next_start - last_end >= min_segment_duration:
                    listening_segments.append((last_end, next_start))
                    
            all_file_listening_timestamps.append({
                "file_path": mp4_file,
                "listening_segments": listening_segments
            })
    
    packed_data = msgpack.packb(all_file_listening_timestamps, use_bin_type=True)
    transcription_timestamps_mp_dumper.write(packed_data)
    transcription_timestamps_mp_dumper.close()
