<div align="center">

<h2>Long-RVOS: A Comprehensive Benchmark for Long-term Referring Video Object Segmentation</h2>

[Tianming Liang](https://tmliang.github.io/)¹  &emsp;
Haichao Jiang¹  &emsp;
Yuting Yang¹  &emsp;
[Chaolei Tan](https://chaoleitan.github.io/)¹  &emsp;
[Shuai Li](https://scholar.google.com/citations?user=GY1t5OYAAAAJ&hl=en)² &emsp;
[Wei-Shi Zheng](https://www.isee-ai.cn/~zhwshi/)¹  &emsp;
[Jian-Fang Hu](https://isee-ai.cn/~hujianfang/)¹*

¹Sun Yat-sen University &emsp;
²Shandong University

<h3 align="center">
  <a href="https://isee-laboratory.github.io/Long-RVOS/" target='_blank'>Project Page</a> |
  <a href="https://arxiv.org/pdf/2505.12702" target='_blank'>Paper
</h3>

</div>


## 🎯 Overview

Long-RVOS is the first large-scale **long-term** referring video object segmentation benchmark, containing 2,000+ videos with an average duration exceeding **60 seconds**. 

![](assets/sample.png)

## 📦 Dataset Download

The Long-RVOS dataset is available on [HuggingFace Hub](https://huggingface.co/datasets/iSEE-Laboratory/Long-RVOS). Use our download script:

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
│       ├── Annotations/
│       └── meta_expressions.json
```

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

wget https://github.com/longzw1997/Open-GroundingDino/releases/download/v0.1.0/gdinot-1.8m-odvg.pth # default
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth

```

### Motion Extraction

If you need to extract motion frames from videos:

```bash
pip install motion-vector-extractor==1.1.0
```

then,

```bash
python scripts/extract_motion.py --data_dir data/long_rvos --output_dir motions
```

Or you can download our processed motions from [Google Drive](https://drive.google.com/drive/folders/1CCspz-1o3HMlIjFRiXRD9PYzoy8Nuwcj?usp=drive_link) and extract:

```bash
motions/
├── train/
│   ├── motions/
│   └── frame_types.json
├── valid/
│   ├── motions/
│   └── frame_types.json
└── test/
    ├── motions/
    └── frame_types.json
```

## 🌟 Get Started

### Training

```bash
python main.py -c configs/lrvos_swint.yaml -rm train -bs 2 -ng 8 --version refermo --epochs 6
```

Note: you can download our checkpoint from [refermo_swint.pth](https://huggingface.co/liangtm/refermo/blob/main/refermo_swint.pth) and put it in the diretory `ckpt`.

### Inference

```bash
PYTHONPATH=. python eval/inference_lrvos_with_motion.py \
    -ng 8 \
    -ckpt ckpt/refermo_swint.pth \
    --split valid \
    --version refermo
```
> 📌 The results will be saved at `output/long_rvos/{split}/{version}`.
> 
> 📌 We also provide a script `eval/inference_lrvos.py` for ReferDINO-style inference, which does not use motions. 

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
