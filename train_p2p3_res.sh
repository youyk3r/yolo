
#!/usr/bin/env bash

# ============================================================
# Stage 5: P2 / P3 Residual Mode Multi-Seed Experiments
# F05, F10, F11
# 3 models x 3 seeds = 9 runs
# Single GPU, sequential training
# ============================================================

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

MODEL_DIR="$PROJECT_DIR/custom_models"
DATA="$PROJECT_DIR/datasets/M3FD_6ch/rgbt6.yaml"
RUNS_DIR="$PROJECT_DIR/runs/detect"
LOG_DIR="$PROJECT_DIR/runs/queue_logs/stage5_multiseed"

mkdir -p "$LOG_DIR"

STATUS_LOG="$LOG_DIR/queue_status.log"

# ---------- GPU ----------
# Default physical GPU 1
# Use GPU=0 to switch to physical GPU 0
GPU="${GPU:-1}"

# ---------- Training parameters ----------
EPOCHS=200
IMGSZ=640
BATCH=32
WORKERS=8
PATIENCE=100

# ---------- Random seeds ----------
SEEDS=(1 2 3)

# ---------- Experiment labels ----------
EXP_NAMES=(
    "F05"
    "F10"
    "F11"
)

# ---------- Model YAML files ----------
MODELS=(
    "stage5-F05-P2-CGAFF-R5-P3-CGAFF-R3-P4-CMA-R4-P5-ASPP-R3.yaml"
    "stage5-F10-P2-CGAFF-R5-P3-CGAFF-R1-P4-CMA-R4-P5-ASPP-R3.yaml"
    "stage5-F11-P2-CGAFF-R5-P3-CGAFF-R4-P4-CMA-R4-P5-ASPP-R3.yaml"
)

# ---------- Preflight checks ----------
echo "============================================================"
echo "Stage 5 Multi-Seed Training Queue"
echo "Project: $PROJECT_DIR"
echo "Physical GPU: $GPU"
echo "Seeds: ${SEEDS[*]}"
echo "Experiments: ${#MODELS[@]}"
echo "Total runs: $((${#MODELS[@]} * ${#SEEDS[@]}))"
echo "Epochs=$EPOCHS, imgsz=$IMGSZ, batch=$BATCH"
echo "============================================================"

if [[ ! -f "$DATA" ]]; then
    echo "[ERROR] Dataset YAML not found: $DATA"
    exit 1
fi

if ! command -v yolo >/dev/null 2>&1; then
    echo "[ERROR] Cannot find yolo. Activate the zk environment first."
    exit 1
fi

if [[ "${#EXP_NAMES[@]}" -ne "${#MODELS[@]}" ]]; then
    echo "[ERROR] Experiment names and model files do not match."
    exit 1
fi

for model in "${MODELS[@]}"; do
    if [[ ! -f "$MODEL_DIR/$model" ]]; then
        echo "[ERROR] Model YAML not found: $MODEL_DIR/$model"
        exit 1
    fi
    echo "[OK] $model"
done

# Reset status log for this queue invocation
: > "$STATUS_LOG"

echo "[$(date '+%F %T')] QUEUE START GPU=$GPU" \
    | tee -a "$STATUS_LOG"

echo "[$(date '+%F %T')] SEEDS=${SEEDS[*]}" \
    | tee -a "$STATUS_LOG"

SUCCESS=0
FAILED=0
TOTAL=0

# ---------- Sequential multi-seed training ----------
# Model first, then iterate through all seeds.
# F05 seed0 -> F05 seed1 -> F05 seed2
# F10 seed0 -> F10 seed1 -> F10 seed2
# F11 seed0 -> F11 seed1 -> F11 seed2

for i in "${!MODELS[@]}"; do

    model="${MODELS[$i]}"
    exp="${EXP_NAMES[$i]}"

    for seed in "${SEEDS[@]}"; do

        TOTAL=$((TOTAL + 1))

        run_name="${exp}-seed${seed}"
        logfile="$LOG_DIR/${run_name}.log"

        echo
        echo "============================================================"
        echo "[$(date '+%F %T')] START $run_name"
        echo "Experiment: $exp"
        echo "Model: $model"
        echo "GPU=$GPU | Seed=$seed"
        echo "Log: $logfile"
        echo "============================================================"

        echo "[$(date '+%F %T')] START $run_name model=$model" \
            | tee -a "$STATUS_LOG"

        CUDA_VISIBLE_DEVICES="$GPU" yolo detect train \
            model="$MODEL_DIR/$model" \
            data="$DATA" \
            epochs="$EPOCHS" \
            imgsz="$IMGSZ" \
            batch="$BATCH" \
            workers="$WORKERS" \
            device=0 \
            seed="$seed" \
            deterministic=True \
            patience="$PATIENCE" \
            optimizer=auto \
            pretrained=True \
            amp=True \
            cls_remap=True \
            project="$RUNS_DIR" \
            name="stage5-${run_name}" \
            > "$logfile" 2>&1

        result=$?

        if [[ "$result" -eq 0 ]]; then

            SUCCESS=$((SUCCESS + 1))

            echo "[$(date '+%F %T')] SUCCESS $run_name" \
                | tee -a "$STATUS_LOG"

            echo "[SUCCESS] $run_name"

        else

            FAILED=$((FAILED + 1))

            echo "[$(date '+%F %T')] FAILED $run_name exit_code=$result log=$logfile" \
                | tee -a "$STATUS_LOG"

            echo "[FAILED] $run_name (exit code=$result)"
            echo "Continuing with the next run..."

        fi

    done

done

# ---------- Final summary ----------
echo
echo "============================================================"
echo "Stage 5 Multi-Seed Queue Finished"
echo "End time: $(date '+%F %T')"
echo "Total runs: $TOTAL"
echo "Successful: $SUCCESS"
echo "Failed: $FAILED"
echo "Status log: $STATUS_LOG"
echo "============================================================"

echo "[$(date '+%F %T')] QUEUE END total=$TOTAL success=$SUCCESS failed=$FAILED" \
    | tee -a "$STATUS_LOG"

if [[ "$FAILED" -gt 0 ]]; then
    exit 1
fi

exit 0
