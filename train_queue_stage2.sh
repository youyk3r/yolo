
#!/usr/bin/env bash

# ============================================================
# Stage 2: P4-CMA / P5-ASPP residual strategy experiments
# GPU 0: E08-E12
# GPU 1: E13-E17
# ============================================================

set -u

# ---------- 路径配置 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

MODEL_DIR="$PROJECT_DIR/custom_models"
DATA="$PROJECT_DIR/datasets/M3FD_6ch/rgbt6.yaml"
RUNS_DIR="$PROJECT_DIR/runs/detect"
LOG_DIR="$PROJECT_DIR/queue_logs/p4p5/stage2"

mkdir -p "$LOG_DIR" 

STATUS_LOG="$LOG_DIR/queue_status.log"

# ---------- 统一训练参数 ----------
EPOCHS=200
IMGSZ=640
BATCH=32
WORKERS=8
SEED=0

# ---------- 实验配置 ----------
MODELS_GPU0=(
    "stage2-E08-P4-CMA-R1-direct-add-P5-no-enhancement-R0.yaml"
    "stage2-E09-P4-CMA-R2-fixed-0.1-P5-no-enhancement-R0.yaml"
    "stage2-E10-P4-CMA-R3-learnable-scalar-P5-no-enhancement-R0.yaml"
    "stage2-E11-P4-CMA-R4-learnable-channel-weight-P5-no-enhancement-R0.yaml"
    "stage2-E12-P4-CMA-R5-input-adaptive-gate-P5-no-enhancement-R0.yaml"
)

MODELS_GPU1=(
    "stage2-E13-P4-no-enhancement-R0-P5-ASPP-R1-direct-add.yaml"
    "stage2-E14-P4-no-enhancement-R0-P5-ASPP-R2-fixed-0.1.yaml"
    "stage2-E15-P4-no-enhancement-R0-P5-ASPP-R3-learnable-scalar.yaml"
    "stage2-E16-P4-no-enhancement-R0-P5-ASPP-R4-learnable-channel-weight.yaml"
    "stage2-E17-P4-no-enhancement-R0-P5-ASPP-R5-input-adaptive-gate.yaml"
)

# ---------- 检查运行环境 ----------
if ! command -v yolo >/dev/null 2>&1; then
    echo "错误：找不到 yolo 命令，请先激活训练环境。"
    exit 1
fi

if [[ ! -f "$DATA" ]]; then
    echo "错误：数据集配置不存在：$DATA"
    exit 1
fi

# ---------- 检查全部模型文件 ----------
echo "检查阶段二模型 YAML 文件..."

MISSING=0

for model in "${MODELS_GPU0[@]}" "${MODELS_GPU1[@]}"; do
    if [[ -f "$MODEL_DIR/$model" ]]; then
        echo "[OK] $model"
    else
        echo "[缺失] $MODEL_DIR/$model"
        MISSING=1
    fi
done

if [[ "$MISSING" -ne 0 ]]; then
    echo "错误：存在缺失的模型 YAML，停止启动队列。"
    exit 1
fi

# ---------- 初始化队列状态日志 ----------
{
    echo
    echo "============================================================"
    echo "Stage 2 queue started: $(date '+%F %T')"
    echo "GPU 0: E08-E12"
    echo "GPU 1: E13-E17"
    echo "Epochs=$EPOCHS, imgsz=$IMGSZ, batch=$BATCH, seed=$SEED"
    echo "============================================================"
} | tee "$STATUS_LOG"

# ---------- 单 GPU 顺序执行函数 ----------
worker() {
    local gpu="$1"
    shift

    local exp
    local model
    local logfile
    local result
    local failed=0

    while (($# >= 2)); do
        exp="$1"
        model="$2"
        shift 2

        logfile="$LOG_DIR/${model%.yaml}-seed${SEED}.log"

        echo "[$(date '+%F %T')] GPU $gpu START $exp $model" \
            | tee -a "$STATUS_LOG"

        CUDA_VISIBLE_DEVICES="$gpu" yolo detect train \
            model="$MODEL_DIR/$model" \
            data="$DATA" \
            epochs="$EPOCHS" \
            imgsz="$IMGSZ" \
            batch="$BATCH" \
            workers="$WORKERS" \
            device=0 \
            seed="$SEED" \
            deterministic=True \
            patience=100 \
            optimizer=auto \
            pretrained=True \
            amp=True \
            cls_remap=True \
            project="$RUNS_DIR" \
            name="${model%.yaml}" \
            > "$logfile" 2>&1

        result=$?

        if [[ "$result" -eq 0 ]]; then
            echo "[$(date '+%F %T')] GPU $gpu SUCCESS $exp" \
                | tee -a "$STATUS_LOG"
        else
            echo "[$(date '+%F %T')] GPU $gpu FAILED $exp exit_code=$result log=$logfile" \
                | tee -a "$STATUS_LOG"
            failed=1
        fi
    done

    return "$failed"
}

# ---------- GPU 0：P4-CMA，E08-E12 ----------
worker 0 \
    E08 "${MODELS_GPU0[0]}" \
    E09 "${MODELS_GPU0[1]}" \
    # E10 "${MODELS_GPU0[2]}" \
    E11 "${MODELS_GPU0[3]}" \
    E12 "${MODELS_GPU0[4]}" &
PID0=$!

# ---------- GPU 1：P5-ASPP，E13-E17 ----------
worker 1 \
    E13 "${MODELS_GPU1[0]}" \
    E14 "${MODELS_GPU1[1]}" \
    # E15 "${MODELS_GPU1[2]}" \
    E16 "${MODELS_GPU1[3]}" \
    E17 "${MODELS_GPU1[4]}" &
PID1=$!

# ---------- 等待两组队列完成 ----------
wait "$PID0"
STATUS0=$?

wait "$PID1"
STATUS1=$?

echo "[$(date '+%F %T')] GPU 0 queue exit_code=$STATUS0" \
    | tee -a "$STATUS_LOG"

echo "[$(date '+%F %T')] GPU 1 queue exit_code=$STATUS1" \
    | tee -a "$STATUS_LOG"

if [[ "$STATUS0" -eq 0 && "$STATUS1" -eq 0 ]]; then
    echo "[$(date '+%F %T')] Stage 2 ALL EXPERIMENTS SUCCESS" \
        | tee -a "$STATUS_LOG"
    exit 0
else
    echo "[$(date '+%F %T')] Stage 2 FINISHED WITH FAILURES; check logs." \
        | tee -a "$STATUS_LOG"
    exit 1
fi
