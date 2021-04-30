# [Re] On the relationship between self-attention and convolutional layers
\[[Original paper](https://arxiv.org/pdf/1911.03584.pdf)\]

Convolutional networks have traditionally been the go-to deep learning method for vision tasks. However, recent work suggests that self-attention layers can behave in a similar manner to convolutional layers, even replicate them. In this work, we experimentally validate the paper "On the Relationship between Self-Attention and Convolutional Layers" which claims that self-attention layers can behave exactly like convolutional layers and supports this claim with theoretical proofs. While it does not achieve SOTA, it provides interesting insights in the functioning of attention mechanisms while processing images. This repository contains our reproducibility code. 

## Organization
Our code is organized as follows:
  - `main`: Reproduces results of the paper in consideration. 
    - `configs`: Configuration files for various experiments.
    - `models`: Network definitions and attention mechanisms.
    - `utils`: Various utilities and helper functions for managing the runs.
    - `main.py`: Training and checkpointing the models.


  - `SAN`: \[[Paper](https://arxiv.org/pdf/2004.13621.pdf)\] For comparison with the main paper, we implement the models in this paper as well. This paper introduces pairwise and patchwise attention mechanisms for use in self-attention networks for vision tasks.
    - `configs`: Configuration files for various experiments.
    - `models`: Network definitions and attention mechanisms.
    - `utils`: Various utilities and helper functions for managing the runs.
    - `main.py`: Training and checkpointing the models.


  - `ViT`: \[[Paper](https://arxiv.org/pdf/2010.11929.pdf)\] We also implement this paper to compare with the main paper. This paper puts forth the Vision Transformer, which performs self-attention over patches of the image rather than individual pixels. However, this model has a two-stage training process (pretraining followed by finetuning) and for fair comparison, we do not pretrain the model in any manner.  
    - `configs`: Configuration files for various experiments.
    - `models`: Network definitions and attention mechanisms.
    - `utils`: Various utilities and helper functions for managing the runs.
    - `main.py`: Training and checkpointing the models.

## Usage
To train the models, clone this repository locally and switch to the directory of the model you wish to train. From here, run `main.py` with 4 CLI arguments:

  - `--config`: Path to configuration file of the experiment to be run.
  - `--output`: Directory in which all checkpoints and model outputs will be placed. Auto-generated with current datetime if not specified by user.
  - `--load`: (Optional) Path to the output directory containing a trained model, named as `best_model.ckpt`.
  - `--task`: Primarily used to choose between training the model, visualizing attention of trained models or computing inference time.
  
**Note.** Since each model has different possible tasks, we request the reader to inspect `main.py` within each folder for more information on available options for `--task`.
  
For instance, to train the model from the main paper with Learned embedding based attention, run the following shell command in `main/`:

```
python3 main.py --config 'configs/learned_2d.yaml' --task 'train'
```

To visualize this trained model's attention as an average over all pixels of the image, run the following command in `main/`:

```
python3 main.py --config 'configs/learned_2d.yaml' --task 'viz' --load 'output/dir/with/trained/model'
```

### Task options
The `--task` argument for different functionalities has been listed below.

**For paper's experiments (`main`)**
  - `train`: Train any of the models except a ResNet model.
  - `resnet_train`: Train a ResNet model.
  - `viz_1`: Visualize attention patterns for a single query pixel in an image. To use this, please create a directory `imgs/` inside `ViT/` and add one image to it.
  - `viz`: Visualize average attention patterns in an image. To use this, please create a directory `imgs/` inside `ViT/` and add one image to it.
  - `gauss_viz`: Visualize attention centers of quadratic embedding based model.
  - `time`: Compute the average throughput and inference time of the model.

**For SAN experiments (`SAN`)**
  - `train`: Train any of the models.
  - `viz`: Visualize average attention patterns for an image. To use this, please create a directory `imgs/` inside `SAN/` and add one image to it.
  - `time`: Compute the average throughput and inference time of the model.

**For Vision Transformer experiments (`ViT`)**
  - `train`: Train any of the models except a ResNet model.
  - `viz_1`: Visualize attention patterns for a single query pixel in an image. To use this, please create a directory `imgs/` inside `ViT/` and add one image to it.
  - `viz`: Visualize average attention patterns in an image. To use this, please create a directory `imgs/` inside `ViT/` and add one image to it.
  - `time`: Compute the average throughput and inference time of the model.
