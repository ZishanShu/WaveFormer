# WaveFormer: Frequency-Time Decoupled Vision Modeling with Wave Equation

**WaveFormer** is the **First** Vision Backbone based on **Wave Equation**.

![demo](figs/fig2.png)

WaveFormer is evaluated as a general-purpose backbone on:
- **Image Classification**
- **Object Detection / Instance Segmentation**
- **Semantic Segmentation**
- **More fine-grained applications (coming soon)**

![demo](figs/fig3.png)

## Installation

### Requirements
- Python >= 3.7
- PyTorch >= 1.8.0
- timm
- einops
- torch_dct
- (Optional) MMSegmentation (for semantic segmentation)
- (Optional) MMDetection (for object detection / instance segmentation)

### Installation

#### 1) Clone the repository
```bash
git clone https://github.com/ZishanShu/WaveFormer.git
cd WaveFormer
````

#### 2) Create the environment

```bash
conda env create -f WaveFormer.yaml
conda activate WaveFormer
```

#### 3) Install minimal dependencies via pip

```bash
pip install timm einops torch-dct
```

#### 4) Integrate with OpenMMLab frameworks

* **MMSegmentation** for semantic segmentation
* **MMDetection** for object detection / instance segmentation

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

### ADE20K (ADEChallengeData2016) — for Semantic Segmentation

Expected structure (commonly used by MMSegmentation):

```
$DATA_ROOT/ade20k/ADEChallengeData2016/
├── images/
│   ├── training/
│   └── validation/
└── annotations/
    ├── training/
    └── validation/
```

Download and extract:

```bash
mkdir -p $DATA_ROOT/ade20k && cd $DATA_ROOT/ade20k

wget -c http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip -q ADEChallengeData2016.zip
```

### Point your training configs to the dataset

Use the corresponding `--data-path <DATA_PATH>` / `data_root = <DATA_ROOT/...>` fields in your configs:

* ImageNet-1K: `<DATA_ROOT>/imagenet`
* COCO: `<DATA_ROOT>/coco`
* ADE20K: `<DATA_ROOT>/ade20k/ADEChallengeData2016`

## Model Training and Inference
### Classification
```bash
torchrun --nproc_per_node=<NUM_GPUS> main.py \
  --cfg <PATH/TO/CONFIGS/waveformer/classification/waveformer_{tiny|small|base}_224.yaml> \
  --batch-size <BATCH_SIZE> \
  --data-path <DATA_PATH> \
  --output <OUTPUT_DIR>
```

### Detection
```bash
bash train_det.sh configs/waveformer/mask_rcnn_fpn_coco_{tiny|small|base}.py <NUM_GPUS>
```

### Segmentation
```bash
CONFIG=<PATH/TO/CONFIGS/waveformer/seg/upernet_waveformer_160k_ade20k_512x512_{tiny|small|base}.py> \
GPUS=<NUM_GPUS> \
NNODES=${NNODES:-1} \
NODE_RANK=${NODE_RANK:-0} \
PYTHONPATH="$PWD":$PYTHONPATH \
python -m torch.distributed.launch \
  --nnodes=$NNODES \
  --node_rank=$NODE_RANK \
  --nproc_per_node=$GPUS \
  ./tools/train.py $CONFIG \
  --work-dir <WORK_DIR> \
  --launcher pytorch
```


## File Structure
```
waveformer/
├── WaveFormer/              # Core model code
│   ├── WaveFormer.py        # Standard WaveFormer implementation
│   └── WaveFormer_dct.py    # DCT-based implementation
├── ...
└── model.py                 # Framework adaptation layer
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
