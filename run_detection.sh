#!/usr/bin/env bash
# WaveFormer-B COCO 检测训练（Mask R-CNN + FPN，单机 16 卡）
#   1x 多尺度训练，drop_path 0.3，lr 2e-4，milestones [10, 11]，EMA
# 用法：bash run_detection.sh
# 运行前请先激活装有 mmdet / mmcv / mmengine 的环境（例如 conda activate open-mmlab）。
#
# 下面的路径都相对于仓库根目录，可用同名环境变量覆盖，例如：
#   PRETRAINED=output/classification/waveformer_base/default/ckpt_epoch_ema_best.pth bash run_detection.sh
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"   # 进入仓库根目录，所有相对路径以此为基准

GPUS=${GPUS:-16}
PORT=${PORT:-29544}
CONFIG=${CONFIG:-detection/configs/waveformer/mask_rcnn_waveformer_base_fpn_ms_1x_coco.py}
DATA=${DATA:-data/coco}                     # 需包含 train2017/ val2017/ annotations/
WORK_DIR=${WORK_DIR:-output/detection/waveformer_base}
# 默认使用 run_classification.sh 训练得到的 EMA 最优分类权重
PRETRAINED=${PRETRAINED:-output/classification/waveformer_base/default/ckpt_epoch_ema_best.pth}

if [ ! -f "${PRETRAINED}" ]; then
  echo "找不到分类预训练权重: ${PRETRAINED}" >&2
  echo "请先运行 bash run_classification.sh，或用 PRETRAINED=<权重路径> 指定" >&2
  exit 1
fi

PYTHONPATH="detection${PYTHONPATH:+:${PYTHONPATH}}" \
python -m torch.distributed.launch \
    --nnodes=1 --node_rank=0 --nproc_per_node=${GPUS} --master_port=${PORT} \
    detection/tools/train.py \
    "${CONFIG}" \
    --work-dir "${WORK_DIR}" \
    --launcher pytorch \
    --cfg-options \
        "train_dataloader.dataset.data_root=${DATA}" \
        "val_dataloader.dataset.data_root=${DATA}" \
        "val_evaluator.ann_file=${DATA}/annotations/instances_val2017.json" \
        "model.backbone.pretrained=${PRETRAINED}"
