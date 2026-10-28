# Third-party notices

This document identifies the principal upstream software and research used by `clipdraw.ipynb`. It does not assign a license to the QMIND repository as a whole.

## CLIPDraw

This notebook is a modified derivative of Kevin Frans' [CLIPDraw implementation](https://github.com/kvfrans/clipdraw).

> Copyright 2022 Kevin Frans

The upstream repository identifies CLIPDraw as licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). QMIND's adaptation reorganizes and documents the notebook, modernizes its Colab installation, adds reproducibility controls and exports, removes unrelated video/pixel utilities, and retains the original optional noun predictor.

Research citation:

Kevin Frans, L. B. Soros, and Olaf Witkowski. “CLIPDraw: Exploring Text-to-Drawing Synthesis through Language-Image Encoders.” 2021. [Paper](https://arxiv.org/abs/2106.14843)

## DiffVG

[DiffVG](https://github.com/BachiLi/diffvg) is a differentiable vector-graphics rasterizer licensed under the [Apache License 2.0](https://github.com/BachiLi/diffvg/blob/master/LICENSE). It accompanies the paper *Differentiable Vector Graphics Rasterization for Editing and Learning* by Tzu-Mao Li, Michal Lukáč, Michaël Gharbi, and Jonathan Ragan-Kelley.

The notebook pins a DiffVG revision and applies runtime compatibility changes that disable its optional TensorFlow bindings, select the C++17 language standard for C++/NVCC compilation, and use a newer pinned pybind11 checkout. These changes are applied only to the temporary Colab checkout.

## OpenAI CLIP

[OpenAI CLIP](https://github.com/openai/CLIP) is distributed under the MIT License:

> MIT License
>
> Copyright (c) 2021 OpenAI
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Build submodules

DiffVG's build uses [pybind11](https://github.com/pybind/pybind11), pinned by this notebook to `v2.13.6`, under its BSD-style license, and the [Thrust](https://github.com/thrust/thrust) submodule under Apache License 2.0. Their license files are retained in the runtime checkouts.

## Runtime Python dependencies

The setup cell installs or uses additional packages including PyTorch, torchvision, NumPy, scikit-image, matplotlib, svgpathtools, cssutils, ftfy, packaging, regex, tqdm, setuptools, and CMake. Each remains subject to its own upstream license and notices as installed in the Colab environment.

Apache 2.0 redistribution conditions include retaining applicable notices, identifying modified files, and providing recipients with the license terms. The linked upstream repositories provide their complete license texts, and the temporary CLIP/DiffVG checkouts retain their license files.
