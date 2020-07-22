# Image Inpainting using Partial Convolutions with PyTorch
this repository is unofficial implementation of  [Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/abs/1804.07723) [Liu+, **ECCV** 2018].

**Official implementation is [here](https://github.com/NVIDIA/partialconv).**

## Requirements
* Python3.x
* PyTorch 1.5.0
* pillow
* matplotlib

## Usage
* set images under ```./img``` and mask images under ```./mask```.
Then,
```
python3 train.py
```

## References
* Image Inpainting for Irregular Holes Using Partial Convolutions, [Guilin Liu](https://liuguilin1225.github.io/), [Fitsum A. Reda](https://scholar.google.com/citations?user=quZ_qLYAAAAJ&hl=en), [Kevin J. Shih](http://web.engr.illinois.edu/~kjshih2/), [Ting-Chun Wang](https://tcwang0509.github.io/), Andrew Tao, [Bryan Catanzaro](http://ctnzr.io/), **NVIDIA Corporation**, [[arXiv]](https://arxiv.org/abs/1804.07723)
