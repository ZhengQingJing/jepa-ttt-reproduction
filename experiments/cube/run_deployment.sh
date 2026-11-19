#!/usr/bin/env bash
# Run from this experiment directory after training the model and fitting its reward head.
set -euo pipefail
cd "$(dirname "$0")"
repo="${JEPA_TTT_ROOT:-$(cd ../.. && pwd)}"
py="${PYTHON:-python}"
export JEPA_TTT_ROOT="$repo" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export CUBE_MODELS="${CUBE_MODELS:-$PWD/models}"
output="${CUBE_OUTPUT:-$PWD/runs}"
if (( $# == 0 )); then set -- quadratic delayed; fi
for shift in "$@"; do
    if [[ "$shift" != "quadratic" && "$shift" != "delayed" ]]; then
        echo "Expected quadratic or delayed" >&2
        exit 2
    fi
    args=(--adapter "$PWD/adapter.py" --task cube --shift "$shift" --output "$output" --episodes 500 --eval-n 100 --chunk 4)
    if [[ ! -f "$output/cube/$shift/frozen.json" ]]; then
        "$py" "$repo/src/benchmark.py" frozen "${args[@]}"
    fi
done
for seed in 0 1 2; do
    for shift in "$@"; do
        args=(--adapter "$PWD/adapter.py" --task cube --shift "$shift" --output "$output" --episodes 500 --eval-n 100 --chunk 4)
        if [[ ! -f "$output/cube/$shift/run$seed/ttt.json" ]]; then
            "$py" "$repo/src/benchmark.py" ttt "${args[@]}" --seed "$seed"
        fi
    done
done
