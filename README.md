## Data processing pipeline for seamless dataset

- Use `find_paired_conversations_cli.py` to get all paired-up conversations from seamless dataset (original seamless data are not paired up structurally.)

- Use `get_transcriptions_timestamps.py` to obtain timestamps of listening segments for each video according to original annotations.

- Use `optimize_msgpack_and_json.py` to obtain refined listening segment timestamps for each video (Since json file is large, so using msgpack file instead. Can use a small subset of timestamp data and convert to json to verify file structure).

- Run `run_parallel.sh [profile_name]` to extract listening segment data from seamless datasets. `[profile_name]` defines different parallel configurations for data extraction.

- Script `debug_visualization.py` is used for debugging aspect ratio issue.

## Problems

- Running `run_parallel.sh`, which process semaless data with `extract_listening_segments_parallel.py` with parallel processing, will result to incremental OOM on servers.
