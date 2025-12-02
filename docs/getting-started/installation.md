
# Installation

## Pixi (Recommended)

We highly recommend using [pixi](https://pixi.sh/) to setup the environment, which is way easier and faster.
If you are interested, please check out the [installation guide for pixi](https://prexsyn.readthedocs.io/en/latest/installation/pixi.html). 
Alternatively, you can follow the steps below to setup the environment manually.

## Conda/Mamba + PyPI

Create and activate conda (mamba) environment:

```bash
conda env create -n prexsyn
conda activate prexsyn
```

Install [PrexSyn Engine](https://github.com/luost26/prexsyn-engine). This package is only available via conda for now. RDKit will be installed as a dependency in this step.

```bash
conda install luost26::prexsyn-engine
```

Setup PrexSyn package. PyTorch and other dependencies will be installed in this step.

```bash
pip install -e .
```
