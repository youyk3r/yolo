
#!/usr/bin/env bash

# ============================================================
# Stage 5: P2 / P3 Residual Mode Ablation Experiments
# F01-F09
# Single GPU, sequential training
# ============================================================

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

MODEL_DIR="$PROJECT_DIR/custom_models"
DATA="$PROJECT_DIR/datasets/M3FD_6ch/rgbt6.yaml"
RUNS_DIR="$PROJECT_DIR/runs/detect"
LOG_DIR="$PROJECT_DIR/runs/queue_logs/stage5"

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
SEED=0
PATIENCE=100

# ---------- Stage 5 model list ----------
MODELS=(
    "stage5-F01-P2-CGAFF-R1-P3-CGAFF-R3-P4-CMA-R4-P5-ASPP-R3.yaml"
    "stage5-F02-P2-CGAFF-R2-P3-CGAFF-R3-P4-CMA-R4-P5-ASPP-R3.yaml"
    "stage5-F03-P2-CGAFF-R3-P3-CGAFF-R3-P4-CMA-R4-P5-ASPP-R3.yaml"
    "stage5-F04-P2-CGAFF-R4-P3-CGAFF-R3-P4-CMA-R4-P5-ASPP-R3.yaml"
    "stage5-F05-P2-CGAFF-R5-P3-CGAFF-R3-P4-CMA-R4-P5-ASPP-R3.yaml"
    "stage5-F06-P2-CGAFF-R3-P3-CGAFF-R1-P4-CMA-R4-P5-ASPP-R3.yaml"
    "stage5-F07-P2-CGAFF-R3-P3-CGAFF-R2-P4-CMA-R4-P5-ASPP-R3.yaml"
    "stage5-F08-P2-CGAFF-R3-P3-CGAFF-R4-P4-CMA-R4-P5-ASPP-R3.yaml"
    "stage5-F09-P2-CGAFF-R3-P3-CGAFF-R5-P4-CMA-R4-P5-ASPP-R3.yaml"
)

# ---------- Preflight checks ----------
echo "============================================================"
echo "Stage 5 P2/P3 Residual Ablation Queue"
echo "Project: $PROJECT_DIR"
echo "Physical GPU: $GPU"
echo "Experiments: ${#MODELS[@]}"
echo "Epochs=$EPOCHS, imgsz=$IMGSZ, batch=$BATCH, seed=$SEED"
echo "============================================================"

if [[ ! -f "$DATA" ]]; then
    echo "[ERROR] Dataset YAML not found: $DATA"
    exit 1
fi

if ! command -v yolo >/dev/null 2>&1; then
    echo "[ERROR] Cannot find yolo. Activate the zk environment first."
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

echo "[$(date '+%F %T')] TOTAL=${#MODELS[@]} SEED=$SEED" \
    | tee -a "$STATUS_LOG"

SUCCESS=0
FAILED=0
TOTAL=0

# ---------- Sequential training ----------
for model in "${MODELS[@]}"; do

    TOTAL=$((TOTAL + 1))

    run_name="${model%.yaml}-seed${SEED}"
    logfile="$LOG_DIR/${run_name}.log"

    echo
    echo "============================================================"
    echo "[$(date '+%F %T')] START $run_name"
    echo "GPU=$GPU | Seed=$SEED"
    echo "Log: $logfile"
    echo "============================================================"

    echo "[$(date '+%F %T')] START $run_name" \
        | tee -a "$STATUS_LOG"

    CUDA_VISIBLE_DEVICES="$GPU" yolo detect train \
        model="$MODEL_DIR/$model" \
        data="$DATA" \
        epochs="$EPOCHS" \
        imgsz="$IMGSZ" \
        batch="$BATCH" \
        workers="$WORKERS" \
        device=0 \
        seed="$SEED" \
        deterministic=True \
        patience="$PATIENCE" \
        optimizer=auto \
        pretrained=True \
        amp=True \
        cls_remap=True \
        project="$RUNS_DIR" \
        name="$run_name" \
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
        echo "Continuing with the next experiment..."
    fi

done

# ---------- Final summary ----------
echo
echo "============================================================"
echo "Stage 5 queue finished"
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
