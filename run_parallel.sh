#!/bin/bash

# Parallel version run script
# Usage: ./run_parallel.sh [config_name]

# Set default environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/zliao/seamless_interaction

# Set CUDA multiprocessing environment variables (resolve CUDA fork subprocess issues)
export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH=/home/zliao/seamless_interaction

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate seamless

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default configuration
DEFAULT_CONFIG="high_perf"

# Show help information
show_help() {
    echo "Usage: $0 [config_name]"
    echo ""
    echo "Available configurations:"
    echo "  debug      - Debug configuration (single thread, small batch)"
    echo "  balanced   - Balanced configuration (4 threads, medium batch)"
    echo "  high_perf  - High performance configuration (8 threads, large batch) [default]"
    echo "  memory_safe - Memory safe configuration (2 threads, small batch)"
    echo "  mp_safe    - Multiprocessing safe configuration (4 processes, medium batch)"
    echo "  mp_safe_high_perf - Multiprocessing safe high performance configuration (8 processes, large batch)"
    echo "  custom     - Custom configuration"
    echo ""
    echo "Examples:"
    echo "  $0 debug"
    echo "  $0 high_perf"
    echo "  $0 mp_safe"
    echo "  $0 custom --num_workers 16 --batch_size 50"
    echo ""
}

# Run script
run_script() {
    local config_name="$1"
    shift
    
    case "$config_name" in
        "debug")
            echo "🚀 Running debug configuration..."
            python "$SCRIPT_DIR/extract_listening_segments_parallel.py" \
                --num_workers 1 \
                --batch_size 5 \
                --min_duration 1.0 \
                "$@"
            ;;
        "balanced")
            echo "🚀 Running balanced configuration..."
            python "$SCRIPT_DIR/extract_listening_segments_parallel.py" \
                --num_workers 4 \
                --batch_size 10 \
                --motion_analysis \
                "$@"
            ;;
        "high_perf")
            echo "🚀 Running high performance configuration..."
            # Set CUDA multiprocessing environment variables
            export CUDA_LAUNCH_BLOCKING=1
            python "$SCRIPT_DIR/extract_listening_segments_parallel.py" \
                --num_workers 8 \
                --batch_size 20 \
                --use_process_pool \
                # --motion_analysis \
                "$@"
            ;;
        "memory_safe")
            echo "🚀 Running memory safe configuration..."
            python "$SCRIPT_DIR/extract_listening_segments_parallel.py" \
                --num_workers 2 \
                --batch_size 5 \
                "$@"
            ;;
        "mp_safe")
            echo "🚀 Running multiprocessing safe configuration..."
            # Set multiprocessing safe environment variables
            export CUDA_LAUNCH_BLOCKING=1
            export PYTHONPATH=/home/zliao/seamless_interaction
            python "$SCRIPT_DIR/extract_listening_segments_parallel.py" \
                --num_workers 4 \
                --batch_size 10 \
                --use_process_pool \
                --motion_analysis \
                "$@"
            ;;
        "mp_safe_high_perf")
            echo "🚀 Running multiprocessing safe high performance configuration..."
            # Set multiprocessing safe environment variables
            export CUDA_LAUNCH_BLOCKING=1
            export PYTHONPATH=/home/zliao/seamless_interaction
            python "$SCRIPT_DIR/extract_listening_segments_parallel.py" \
                --num_workers 8 \
                --batch_size 20 \
                --use_process_pool \
                "$@"
            ;;
        "custom")
            echo "🚀 Running custom configuration..."
            python "$SCRIPT_DIR/extract_listening_segments_parallel.py" "$@"
            ;;
        "help"|"-h"|"--help")
            show_help
            exit 0
            ;;
        *)
            echo "❌ Unknown configuration: $config_name"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Check parameters
if [ $# -eq 0 ]; then
    echo "Using default configuration: $DEFAULT_CONFIG"
    run_script "$DEFAULT_CONFIG"
else
    run_script "$@"
fi
