

# HILA: Improving Semantic Segmentation in Transformers using Hierarchical Inter-Level Attention

This is the official PyTorch implementation of HILA. For technical details, please refer to:

Improving Semantic Segmentation in Transformers using Hierarchical Inter-Level Attention.<br>
[Gary Leung](https://www.cs.toronto.edu/~garyleung/), [Jun Gao](http://www.cs.toronto.edu/~jungao/), [Xiaohui Zeng](https://www.cs.utoronto.ca/~xiaohui/), and [Sanja Fidler](http://www.cs.toronto.edu/~fidler/).<br>

###### University of Toronto  [Project page](https://www.cs.toronto.edu/~garyleung/hila) | [Paper (ArXiv soon)]() 

<div align="center">
  <img src="/docs/resources/performance_graph.png" width="100%"/>
</div>
<p align="center">
  Figure 1: Performance of adding HILA to SegFormer.
</p>

<div align="center">
  <img src="/docs/resources/finished_figure_aligned_v3.gif" width="100%"/>
</div>
<p align="center">
  Figure 2: Adding HILA to pre-existing architectures.
</p>

<div align="center">
  <img src="/docs/resources/patch_diagram.png" width="100%"/>
</div>
<p align="center">
  Figure 3: Top-Down and Bottom-Up Inter-Level Attention.
</p>

## License
```
MIT License

Copyright (c) 2022 Gary Leung, Jun Gao, Xiaohui Zeng, Sanja Fidler

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.                                                                         
```

## Environment Setup

##### Clone this repo
```bash
git clone https://github.com/releasename
cd hila-master
 ```
##### Setup Python environment

```bash
conda create --name hila python=3.7.1
conda activate hila
pip install -r requirements.txt
 ```

##### Dataset setup
We use ADE20K and Cityscapes. For data preparation, please refer to [MMSegmentation v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0).

## Models

Download [trained weights and Imagenet-1K pretrained weights](https://drive.google.com/drive/folders/1tjLob7Qho9YlQoiIE6OmyVi_xaPSV3gz?usp=sharing) and put them in a folder ```pretrained/```. Imagenet-1K pretrained models can be found in the ```pretrained/hila``` folder with model weights being in their respective folders.
Segformer pretrained weights can be found [here](https://github.com/NVlabs/SegFormer).
## Evaluation

Example: evaluate ```SegFormer-B1 + HILA S(2,3,4)``` on ```Cityscapes```:

```
# Single-scale testing
python ./tools/test.py local_configs/hila/segformer/B1/hila.b1.1024x1024.city.160k_S234.py  /path/to/checkpoint_file

# Multi-scale testing
python ./tools/test.py local_configs/hila/segformer/B1/hila.b1.1024x1024.city.160k_S234.py  /path/to/checkpoint_file --aug-test

# F-score testing
python ./tools/test.py local_configs/hila/segformer/B1/hila.b1.1024x1024.city.160k_S234.py  /path/to/checkpoint_file --eval-f1-start 0

# F-score testing (in batches due to memory issues)
python ./tools/test.py local_configs/hila/segformer/B1/hila.b1.1024x1024.city.160k_S234.py  /path/to/checkpoint_file --eval-f1-start 0 --eval-f1-step-size 100
...
python ./tools/test.py local_configs/hila/segformer/B1/hila.b1.1024x1024.city.160k_S234.py  /path/to/checkpoint_file --eval-f1-start 400 --eval-f1-step-size 100

# Modified F-score testing for ADE20K
python ./tools/test.py local_configs/hila/segformer/B1/hila.b1.512x512.ade.160k_S234.py  /path/to/checkpoint_file --eval-f1-start 0 --mod-f1

# Distance Crop testing (eg. 624 by 1248)
python ./tools/test.py local_configs/hila/segformer/B1/hila.b1.1024x1024.city.160k_S234.py  /path/to/checkpoint_file --eval-distance-crop 624 1248
```

## Training

Example: train ```SegFormer-B1 + HILA S(2,3,4)``` on ```Cityscapes```:

```
# GPU training
python ./tools/dist_train.sh local_configs/hila/segformer/B1/hila.b1.1024x1024.city.160k_S234.py <GPU_NUM>
```

## Visualization

Example: visualize ```SegFormer-B1 + HILA S(2,3,4)``` on ```Cityscapes```:

```
# Hierarchical visualization of Stage 4 feature at coor (x, y)
python ./tools/visualize_attention.py local_configs/hila/segformer/B1/hila.b1.1024x1024.city.160k_S234.py \
--show-dir ./path/output/dir/ --save-gt-seg --save-ori-img --data-range i j --attn-coors x y
```

<div align="center">
  <img src="/docs/resources/cityscapes_attention.png" width="100%"/>
</div>
<p align="center">
  Figure 4: Example Visualizations of Hierarchical Attention on Cityscapes.
</p>


## Citation
```
@article{leung2022hila,
  title={HILA: Improving Semantic Segmentation in Transformers using Hierarchical Inter-Level Attention},
  author={Leung, Gary and Gao, Jun and Zeng, Xiaohui and Fidler, Sanja},
  journal={},
  year={2022}
}
```
