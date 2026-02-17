#!/bin/bash

function nsys_profile() {
    local command="$1"
    local logs_dir="${2:-./logs}"

    if [ -z "$command" ]; then
        echo "Usage: nsys_profile '<command>' [logs_dir]"
        echo "Example: nsys_profile 'python3 bench.py --warmup_steps 3'"
        echo "Example: nsys_profile 'python3 bench.py --warmup_steps 3' ./custom_logs"
        return 1
    fi

    local timestamp=$(date +%Y%m%d_%H%M%S)
    local output_dir="$logs_dir/$timestamp"

    mkdir -p "$output_dir"

    local profile_file="$output_dir/profile"

    echo "============================================================"
    echo "NSYS Profile"
    echo "============================================================"
    echo "Command: $command"
    echo "Output: $output_dir"
    echo "============================================================"

    nsys profile \
        --delay 15 \
        --duration 300 \
        --output "$profile_file" \
        --trace cuda,nvtx,osrt \
        --cuda-memory-usage true \
        --force-overwrite true \
        --sample=none \
        --cpuctxsw=none \
        $command

    local full_profile_path="${profile_file}.nsys-rep"
    local qdstrm_path="${profile_file}.qdstrm"

    echo ""
    echo "============================================================"
    echo "Profile saved to: $output_dir"
    echo "============================================================"

    if [ -f "$full_profile_path" ]; then
        echo ""
        echo "CUDA API Summary:"
        nsys stats "$full_profile_path" --report cudaapisum 2>&1 | grep -v "^NOTICE:\|^Processing\|^Generating\|DEPRECATED" | head -15 | tee "$output_dir/cuda_api.txt"

        echo ""
        echo "GPU Kernel Summary:"
        nsys stats "$full_profile_path" --report gpukernsum 2>&1 | grep -v "^NOTICE:\|^Processing\|^Generating\|DEPRECATED" | head -20 | tee "$output_dir/gpu_kernel.txt"

        echo ""
        echo "Results saved to:"
        echo "  - $output_dir/cuda_api.txt"
        echo "  - $output_dir/gpu_kernel.txt"
    elif [ -f "$qdstrm_path" ]; then
        echo ""
        echo "Note: Profile data saved to qdstrm (nsys-rep generation failed)"
        echo "Use NVIDIA Nsight Systems UI to view: $qdstrm_path"
    else
        echo "Profile file not found!"
    fi
}
