#!/usr/bin/env bash
# WaveFormer-B ImageNet-1K 分类训练（单机 16 卡）
#   DROP_PATH_RATE=0.40, AUG.REPROB=0.50
# 用法：bash run_classification.sh
#
# 下面的路径都相对于仓库根目录，可用同名环境变量覆盖，例如：
#   DATA_PATH=data/imagenet OUTPUT=output/classification bash run_classification.sh
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"   # 进入仓库根目录，所有相对路径以此为基准

GPUS=${GPUS:-16}
PORT=${PORT:-29505}
DATA_PATH=${DATA_PATH:-data/imagenet}       # 需包含 train/ 和 val/
OUTPUT=${OUTPUT:-output/classification}    # 权重和日志保存在 ${OUTPUT}/waveformer_base/default/

python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=${GPUS} \
  --master_addr="127.0.0.1" --master_port=${PORT} classification/main.py \
  --cfg classification/configs/waveformer/waveformer_base_224.yaml \
  --batch-size 128 \
  --data-path "${DATA_PATH}" \
  --output "${OUTPUT}" \
  --opts MODEL.DROP_PATH_RATE 0.40 AUG.REPROB 0.50
