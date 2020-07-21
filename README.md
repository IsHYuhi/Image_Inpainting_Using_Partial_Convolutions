# Image Inpainting using Partial Convolutions with PyTorch
this repository is unofficial implementation of  [Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/abs/1804.07723) [Liu+, arXiv2018].

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
* Image Inpainting for Irregular Holes Using Partial Convolutions, Gulilin Liu, Fitsum A. Reda, Kevin J. Shih, Ting-Chun Wang, Andrew Tao, Bryan Catanzaro, [[arXiv]](https://arxiv.org/abs/1804.07723)
