# WaveFormer: Frequency-Time Decoupled Vision Modeling with Wave Equation

**WaveFormer** is the **First** Vision Backbone based on **Wave Equation**.

![demo](figs/fig2.png)

WaveFormer is evaluated as a general-purpose backbone on:
- **Image Classification**
- **Object Detection / Instance Segmentation**
- **More fine-grained applications (coming soon)**

![demo](figs/fig3.png)

## Installation

### Requirements
- Python >= 3.8
- PyTorch >= 1.13 and torchvision
- timm, yacs, termcolor, fvcore
- (Optional) MMDetection 3.x + MMCV (>= 2.0.0, < 2.2.0) + MMEngine (for object detection / instance segmentation)

### Installation

#### 1) Clone the repository
```bash
git clone https://github.com/ZishanShu/WaveFormer.git
cd WaveFormer
```

#### 2) Create the environment

```bash
conda create -n WaveFormer python=3.10 -y
conda activate WaveFormer

# PyTorch with CUDA, e.g. PyTorch 2.1 + CUDA 11.8 (prebuilt MMCV wheels are available for it)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
```

#### 3) Install minimal dependencies via pip

```bash
pip install timm yacs termcolor fvcore
```

#### 4) Integrate with OpenMMLab frameworks

* **MMDetection** for object detection / instance segmentation

```bash
pip install -U openmim
mim install mmengine "mmcv>=2.0.0,<2.2.0"
mim install "mmdet>=3.0.0"
```

## Data Download
Set a data root first:
```bash
export DATA_ROOT=<PATH/TO/DATA_ROOT>
mkdir -p $DATA_ROOT
```

### ImageNet-1K (ILSVRC 2012)

ImageNet-1K requires registration and manual download from the official ImageNet site.
Expected structure:

```
$DATA_ROOT/imagenet/
├── train/  
└── val/    
```

Minimal setup steps:

1. Download the ILSVRC2012 archives (commonly: `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar`)
2. Extract them into the folder below:

```bash
mkdir -p $DATA_ROOT/imagenet
tar -xf <PATH/TO/ILSVRC2012_img_train.tar> -C $DATA_ROOT/imagenet
tar -xf <PATH/TO/ILSVRC2012_img_val.tar>   -C $DATA_ROOT/imagenet
```

### COCO (2017) — for Detection / Instance Segmentation

Expected structure:

```
$DATA_ROOT/coco/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    ├── instances_val2017.json
    └── (other annotation files)
```

Download and extract:

```bash
mkdir -p $DATA_ROOT/coco && cd $DATA_ROOT/coco

# images
wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip

# annotations
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip -q train2017.zip
unzip -q val2017.zip
unzip -q annotations_trainval2017.zip
```

### Point the training scripts to the dataset

The training scripts read the dataset paths from environment variables (relative paths are resolved from the repository root):

* ImageNet-1K: `DATA_PATH=<DATA_ROOT>/imagenet` (default: `data/imagenet`)
* COCO: `DATA=<DATA_ROOT>/coco` (default: `data/coco`)

## Pretrained Weights

Weights are hosted on Hugging Face: [Zishan0602/WaveFormer](https://huggingface.co/Zishan0602/WaveFormer)

| Task | Model | Config | Checkpoint |
| :---: | :---: | :---: | :---: |
| COCO detection (Mask R-CNN, 1x, multi-scale) | WaveFormer-B | [config](detection/configs/waveformer/mask_rcnn_waveformer_base_fpn_ms_1x_coco.py) | [run_det_ms_1x_dp03.pth](https://huggingface.co/Zishan0602/WaveFormer/resolve/main/run_det_ms_1x_dp03.pth) |

Download it into `checkpoints/`

## Model Training and Inference
`run_classification.sh` and `run_detection.sh` train WaveFormer-B on a single node with 16 GPUs by default (set `GPUS=<NUM_GPUS>` to change it) and write checkpoints and logs to `output/`.

### Classification
```bash
DATA_PATH=$DATA_ROOT/imagenet bash run_classification.sh
```
Checkpoints and logs: `output/classification/waveformer_base/default/`

### Detection
Mask R-CNN + FPN, 1x schedule with multi-scale training. The backbone is initialized from the ImageNet-1K checkpoint of the classification step, `output/classification/waveformer_base/default/ckpt_epoch_ema_best.pth` (set `PRETRAINED=<CKPT>` to use another one).
```bash
DATA=$DATA_ROOT/coco bash run_detection.sh
```
Checkpoints and logs: `output/detection/waveformer_base/`

Evaluate the released detection checkpoint on COCO val2017:
```bash
PYTHONPATH=detection python -m torch.distributed.launch --nproc_per_node=<NUM_GPUS> \
  detection/tools/test.py \
  detection/configs/waveformer/mask_rcnn_waveformer_base_fpn_ms_1x_coco.py \
  checkpoints/run_det_ms_1x_dp03.pth \
  --launcher pytorch \
  --cfg-options test_dataloader.dataset.data_root=$DATA_ROOT/coco \
                test_evaluator.ann_file=$DATA_ROOT/coco/annotations/instances_val2017.json
```


## File Structure
```
WaveFormer/
├── run_classification.sh / run_detection.sh
├── classification/
│   ├── models/waveformer.py       # WaveFormer implementation
│   ├── configs/waveformer/        # waveformer_base_224.yaml
│   └── main.py
└── detection/
    ├── waveformer/waveformer.py   # WaveFormer implementation
    ├── model.py                   # Framework adaptation layer (MMDetection)
    ├── configs/waveformer/        # mask_rcnn_waveformer_base_fpn_ms_1x_coco.py
    └── tools/                     # train.py / test.py
```

## Citation
```
@misc{shu2026waveformerfrequencytimedecoupledvision,
      title={WaveFormer: Frequency-Time Decoupled Vision Modeling with Wave Equation}, 
      author={Zishan Shu and Juntong Wu and Wei Yan and Xudong Liu and Hongyu Zhang and Chang Liu and Youdong Mao and Jie Chen},
      year={2026},
      eprint={2601.08602},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.08602}, 
}
```
