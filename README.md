# Long-RVOS: A Comprehensive Benchmark for Long-term Referring Video Object Segmentation

Tianming Liang<sup>1</sup>, Haichao Jiang<sup>1</sup>, Yuting Yang<sup>1</sup>, Chaolei Tan<sup>1</sup>, Shuai Li<sup>2</sup>,  
Wei-Shi Zheng<sup>1</sup>, Jian-Fang Hu<sup>1*</sup>  
<sup>1</sup>Sun Yat-sen University, <sup>2</sup>Shandong University

**Project Page** | **Paper** | **Dataset**

[Project Page](https://isee-laboratory.github.io/Long-RVOS) | [Paper](https://arxiv.org/html/2505.12702v2) | [Dataset](https://huggingface.co/datasets/your-repo/long-rvos)

## 📢 News

- `2025.XX.XX` Long-RVOS benchmark and ReferMo baseline code are released!
- `2025.XX.XX` Dataset is available on HuggingFace Hub.
- `2025.XX.XX` Long-RVOS paper is accepted!

## 🎯 Overview

Long-RVOS is the first large-scale **long-term** referring video object segmentation benchmark, containing 2,000+ videos with an average duration exceeding **60 seconds**. Unlike existing RVOS datasets that focus on short clips (several seconds), Long-RVOS tackles realistic long-video challenges such as:

- **Frequent occlusion** and disappearance-reappearance
- **Shot changes** in long videos
- **Complex spatiotemporal relationships** across extended temporal spans

## 🚀 Environment Setup

```bash
# Clone the repo
git clone https://github.com/iSEE-Laboratory/Long_RVOS.git
cd Long_RVOS

# [Optional] Create a clean Conda environment
conda create -n long_rvos python=3.10 -y
conda activate long_rvos

# PyTorch 
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia

# MultiScaleDeformableAttention
cd models/GroundingDINO/ops
python setup.py build install
python test.py
cd ../../..

# Other dependencies
pip install -r requirements.txt
```

### Install SAM2

ReferMo uses SAM2 for mask propagation. Please install SAM2 following the [official instructions](https://github.com/facebookresearch/segment-anything-2):

```bash
cd sam2
pip install -e .
cd ..
```

Download SAM2 checkpoints and put them in `sam2/checkpoints/`:

```bash
cd sam2/checkpoints
bash download_ckpts.sh
cd ../..
```

### Download Pretrained GroundingDINO

Download pretrained GroundingDINO weights and put them in the `pretrained` directory:

```bash
mkdir pretrained
cd pretrained

wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://github.com/longzw1997/Open-GroundingDino/releases/download/v0.1.0/gdinot-1.8m-odvg.pth

```

## 📦 Data Preparation

### Download Long-RVOS Dataset

The Long-RVOS dataset is available on HuggingFace Hub. Use our download script:

```bash
python scripts/download_dataset.py \
    --repo_id iSEE-Laboratory/Long-RVOS \
    --output_dir data
```

Or manually download from [Google Drive](https://drive.google.com/drive/folders/19GXKf8COc_W3ZHsLvhWTzaPrxRedszac?usp=drive_link) and extract:

```bash
data/
├── long_rvos/
│   ├── train/
│   │   ├── JPEGImages/
│   │   ├── Annotations/
│   │   └── meta_expressions.json
│   ├── valid/
│   │   ├── JPEGImages/
│   │   ├── Annotations/
│   │   └── meta_expressions.json
│   └── test/
│       ├── JPEGImages/
│       └── meta_expressions.json
```


### Extract Motions

If you need to extract motion frames from videos, use:

```bash
python scripts/extract_motion.py --data_dir data/long_rvos --output_dir motions
```

## 🏁 Get Started

### Training

```bash
python main.py -c configs/lrvos_swint.yaml -rm train -bs 2 -ng 8 --version refermo --epochs 6
```

### Inference

```bash
PYTHONPATH=. python eval/inference_lrvos_with_motion.py \
    -ng 8 \
    -ckpt ckpt/refermo_swint.pth \
    --split valid \
    --version refermo
```

### Evaluation

After inference, evaluate the results:

```bash
bash run_eval.sh output/long_rvos/valid/refermo valid
```

## 🙏 Acknowledgements

Our code is built upon [ReferDINO](https://github.com/iSEE-Laboratory/ReferDINO), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), and [SAM2](https://github.com/facebookresearch/sam2). We sincerely appreciate these efforts.

## 📝 Citation

If you find our work helpful for your research, please consider citing our paper:

```bibtex
@article{liang2025longrvos,
  title={Long-RVOS: A Comprehensive Benchmark for Long-term Referring Video Object Segmentation},
  author={Liang, Tianming and Jiang, Haichao and Yang, Yuting and Tan, Chaolei and Li, Shuai and Zheng, Wei-Shi and Hu, Jian-Fang},
  journal={arXiv preprint arXiv:2505.12702},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License. Please refer to the LICENSE file for details.
