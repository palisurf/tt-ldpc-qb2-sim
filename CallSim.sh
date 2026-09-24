#!/usr/bin/env bash
set -e

# Automatically navigate to the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Activate Tenstorrent Python virtual environment
if [ -f "/home/ttuser/.tenstorrent-venv/bin/activate" ]; then
    source "/home/ttuser/.tenstorrent-venv/bin/activate"
fi

# Set TT-Metal runtime root for device firmware & descriptors
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-/home/ttuser/tt-metal}"

# Ensure C++ binary is compiled
make ldpc_sim -j

# Ensure Results directory exists
mkdir -p Results

# Matrix selection: default to AR4JA_r45_4c_128c_r12.chinn.out if present, otherwise sample_16k.chinn or test_code.chinn
if [ -f "matrices/AR4JA_r45_4c_128c_r12.chinn.out" ]; then
    DEFAULT_MATRIX="matrices/AR4JA_r45_4c_128c_r12.chinn.out"
    DEFAULT_PUNCTURED=512
elif [ -f "matrices/sample_16k.chinn" ]; then
    DEFAULT_MATRIX="matrices/sample_16k.chinn"
    DEFAULT_PUNCTURED=512
else
    DEFAULT_MATRIX="matrices/test_code.chinn"
    DEFAULT_PUNCTURED=0
fi

MATRIX="${CHINN_MATRIX:-$DEFAULT_MATRIX}"
PUNCTURED="${PUNCTURED_NODES:-$DEFAULT_PUNCTURED}"

# Simulation parameters:
# 1. Eb/N0 range 3-tuple (Start, Stop, Step): default (1.5, 3.0, 0.5)
# 2. Max blocks per core (default: 100000)
# 3. Minimum block errors (default: 50)
# 4. Batch size per core per evaluation step (default: 1000)
# 5. Max decoder iterations per codeword (default: 16)
#
# Examples:
#   ./CallSim.sh 1.5:3.0:0.5
#   ./CallSim.sh 1.5,3.0,0.5
#   ./CallSim.sh 100000 50 1000 1.5:3.0:0.5
#   ./CallSim.sh --max_iter 20
EBNO_START="${EBNO_START:-1.5}"
EBNO_END="${EBNO_END:-3.0}"
EBNO_STEP="${EBNO_STEP:-0.5}"
MAX_BLOCKS_PER_CORE="${MAX_BLOCKS_PER_CORE:-100000}"
MIN_ERRORS="${MIN_ERRORS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
MAX_ITER="${MAX_ITER:-16}"

parse_ebn0_tuple() {
    local raw="$1"
    raw="${raw//[()\[\]]/}"
    raw="${raw//[,:\/]/ }"
    read -r s_start s_end s_step <<< "${raw}"
    if [ -n "$s_start" ] && [ -n "$s_end" ] && [ -n "$s_step" ]; then
        EBNO_START="$s_start"
        EBNO_END="$s_end"
        EBNO_STEP="$s_step"
        return 0
    fi
    return 1
}

if [ -n "${EBNO_TUPLE}" ]; then
    parse_ebn0_tuple "${EBNO_TUPLE}" || true
fi

FORWARD_ARGS=()
INT_POS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ebn0|-e)
            parse_ebn0_tuple "$2"
            shift 2
            ;;
        --max_iter|-i)
            MAX_ITER="$2"
            shift 2
            ;;
        --matrix|-c|--chinn)
            MATRIX="$2"
            if [[ "$2" == *"test_code"* ]] && [ -z "${PUNCTURED_NODES}" ]; then
                PUNCTURED=0
            elif [[ "$2" == *"AR4JA"* ]] && [ -z "${PUNCTURED_NODES}" ]; then
                PUNCTURED=512
            fi
            shift 2
            ;;
        --punctured|-p)
            PUNCTURED="$2"
            shift 2
            ;;
        *)
            if parse_ebn0_tuple "$1" 2>/dev/null; then
                shift
            elif [[ "$1" == *.chinn || "$1" == *.chinn.out || "$1" == *.out || -f "$1" ]]; then
                MATRIX="$1"
                if [[ "$1" == *"test_code"* ]] && [ -z "${PUNCTURED_NODES}" ]; then
                    PUNCTURED=0
                elif [[ "$1" == *"AR4JA"* ]] && [ -z "${PUNCTURED_NODES}" ]; then
                    PUNCTURED=512
                fi
                shift
            elif [[ "$1" =~ ^[0-9]+$ ]]; then
                if [ $INT_POS -eq 0 ]; then
                    MAX_BLOCKS_PER_CORE="$1"
                elif [ $INT_POS -eq 1 ]; then
                    MIN_ERRORS="$1"
                elif [ $INT_POS -eq 2 ]; then
                    BATCH_SIZE="$1"
                else
                    FORWARD_ARGS+=("$1")
                fi
                INT_POS=$((INT_POS + 1))
                shift
            else
                FORWARD_ARGS+=("$1")
                shift
            fi
            ;;
    esac
done

echo "=========================================================="
echo "Launching LDPC Simulation with matrix: ${MATRIX}"
echo "Eb/N0 Range (Start:Stop:Step): ${EBNO_START} : ${EBNO_END} : ${EBNO_STEP} dB"
echo "Max blocks per core          : ${MAX_BLOCKS_PER_CORE}"
echo "Min block errors             : ${MIN_ERRORS}"
echo "Batch size per core          : ${BATCH_SIZE}"
echo "Max decoder iterations       : ${MAX_ITER}"
echo "=========================================================="

# Execute Monte Carlo simulation sweep
python3 run_simulation.py \
    --chinn "${MATRIX}" \
    --ebn0_start "${EBNO_START}" \
    --ebn0_end "${EBNO_END}" \
    --ebn0_step "${EBNO_STEP}" \
    --punctured "${PUNCTURED}" \
    --min_errors "${MIN_ERRORS}" \
    --max_blocks_per_core "${MAX_BLOCKS_PER_CORE}" \
    --batch_size "${BATCH_SIZE}" \
    --max_iter "${MAX_ITER}" \
    "${FORWARD_ARGS[@]}"