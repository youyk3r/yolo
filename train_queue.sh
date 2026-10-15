
#!/usr/bin/env bash

set -u

# ============================================================
# M3FD P4/P5 Dual-Stream Fusion Experiment Queue
#
# 脚本位置：~/zhangke/train_queue.sh
# YOLO 项目：~/zhangke/yolo/
#
# GPU 0：E0 -> E1 -> E2 -> E3（P4）
# GPU 1：E4 -> E5 -> E6 -> E7（P5）
# ============================================================

# 自动定位脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# YOLO 项目目录
PROJECT_DIR="$SCRIPT_DIR/yolo"

# 数据集、模型、结果和日志路径
DATA="$PROJECT_DIR/datasets/M3FD_6ch/rgbt6.yaml"
MODEL_DIR="$PROJECT_DIR/custom_models"
RUNS_DIR="$PROJECT_DIR/runs/detect"
LOG_DIR="$PROJECT_DIR/queue_logs/p4p5"

mkdir -p "$LOG_DIR" "$RUNS_DIR"

# 检查 yolo 命令
if ! command -v yolo >/dev/null 2>&1; then
    echo "[ERROR] 找不到 yolo，请先执行 conda activate zk"
    exit 1
fi

# 检查数据集配置
if [ ! -f "$DATA" ]; then
    echo "[ERROR] 数据集配置不存在：$DATA"
    exit 1
fi

# 检查全部 8 个模型配置
for n in {0..7}; do
    case "$n" in
        0) model="m3fd-p4-none-r0-e00.yaml" ;;
        1) model="m3fd-p4-cma-r3-e01.yaml" ;;
        2) model="m3fd-p4-aspp-r3-e02.yaml" ;;
        3) model="m3fd-p4-nonlocal-r3-e03.yaml" ;;
        4) model="m3fd-p5-none-r0-e04.yaml" ;;
        5) model="m3fd-p5-cma-r3-e05.yaml" ;;
        6) model="m3fd-p5-aspp-r3-e06.yaml" ;;
        7) model="m3fd-p5-nonlocal-r3-e07.yaml" ;;
    esac

    if [ ! -f "$MODEL_DIR/$model" ]; then
        echo "[ERROR] 模型 YAML 不存在：$MODEL_DIR/$model"
        exit 1
    fi
done

# 单个实验训练函数
run_train() {
    local gpu="$1"
    local exp="$2"
    local model="$3"
    local name="${model%.yaml}-seed0"
    local log="$LOG_DIR/${name}.log"
    local status

    echo "[$(date '+%F %T')] START GPU=$gpu EXP=$exp MODEL=$model" \
        | tee -a "$LOG_DIR/queue_status.log"

    CUDA_VISIBLE_DEVICES="$gpu" yolo detect train \
        model="$MODEL_DIR/$model" \
        data="$DATA" \
        epochs=200 \
        imgsz=640 \
        batch=32 \
        workers=8 \
        device=0 \
        seed=0 \
        deterministic=True \
        patience=100 \
        optimizer=auto \
        pretrained=True \
        amp=True \
        cls_remap=True \
        project="$RUNS_DIR" \
        name="$name" \
        > "$log" 2>&1

    status=$?

    if [ "$status" -eq 0 ]; then
        echo "[$(date '+%F %T')] SUCCESS GPU=$gpu EXP=$exp" \
            | tee -a "$LOG_DIR/queue_status.log"
    else
        echo "[$(date '+%F %T')] FAILED GPU=$gpu EXP=$exp EXIT=$status" \
            | tee -a "$LOG_DIR/queue_status.log"
    fi

    return "$status"
}

# 按顺序执行指定实验
worker() {
    local gpu="$1"
    shift

    local failed=0
    local exp
    local model

    while [ "$#" -gt 0 ]; do
        exp="$1"
        model="$2"
        shift 2

        run_train "$gpu" "$exp" "$model" || failed=1
    done

    return "$failed"
}

echo "============================================================"
echo "M3FD P4/P5 Fusion Queue Started"
echo "Project: $PROJECT_DIR"
echo "Dataset: $DATA"
echo "GPU 0: E0 -> E1 -> E2 -> E3"
echo "GPU 1: E4 -> E5 -> E6 -> E7"
echo "Results: $RUNS_DIR"
echo "Logs: $LOG_DIR"
echo "============================================================"

# GPU 0：P4 实验 E0-E3
worker 0 \
    E0 m3fd-p4-none-r0-e00.yaml \
    E1 m3fd-p4-cma-r3-e01.yaml \
    E2 m3fd-p4-aspp-r3-e02.yaml \
    E3 m3fd-p4-nonlocal-r3-e03.yaml &
PID0=$!

# GPU 1：P5 实验 E4-E7
worker 1 \
    E4 m3fd-p5-none-r0-e04.yaml \
    E5 m3fd-p5-cma-r3-e05.yaml \
    E6 m3fd-p5-aspp-r3-e06.yaml \
    E7 m3fd-p5-nonlocal-r3-e07.yaml &
PID1=$!

# 等待两条队列完成
wait "$PID0"
RESULT0=$?

wait "$PID1"
RESULT1=$?

echo "============================================================"
echo "All queues finished: $(date '+%F %T')"
echo "GPU 0 queue result: $RESULT0"
echo "GPU 1 queue result: $RESULT1"
echo "Status log: $LOG_DIR/queue_status.log"
echo "============================================================"

if [ "$RESULT0" -ne 0 ] || [ "$RESULT1" -ne 0 ]; then
    exit 1
fi

exit 0
