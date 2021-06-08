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
  - `--load`: (Optional) Path to the output directory containing a trained model, named as `best_model.ckpt`. The output directory is overwritten to this directory if specified.
  - `--image`: (Optional) Path to the image on which any visualization will be performed. Defaults to `imgs/cat_s_000019.png`.
  - `--task`: Primarily used to choose between training the model, visualizing attention of trained models or computing inference time.
  
**Note.** Since each model has different possible tasks, we request the reader to inspect `main.py` within each folder for more information on available options for `--task`.
  
For instance, to train the model from the main paper with Learned embedding based attention (9 heads), run the following shell command in `main/`:

```
python3 main.py --config 'configs/learned2d_withcontent_9heads.yaml' --task 'train'
```

To visualize this trained model's attention as an average over all pixels of the image, run the following command in `main/`:

```
python3 main.py --config 'configs/learned2d_withcontent_9heads.yaml' --task 'viz' --load 'output/dir/with/trained/model'
```

## Experiment configurations
Some details about the major settings within the provided configuration files.

| Name of file                              | Description                                                               |
|-------------------------------------------|---------------------------------------------------------------------------|
| `gaussian_anisotropic.yaml`               | Gaussian encoding with non-isotropic attention heads                      |
| `gaussian_isotropic.yaml`                 | Gaussian encoding with isotropic attention heads                          |
| `learned2d_nocontent_9heads.yaml`         | Learned embedding without content attention, 9 attention heads            |
| `learned2d_nocontent_16heads.yaml`        | Learned embedding without content attention, 16 attention heads           |
| `learned2d_withcontent_9heads.yaml`       | Learned embedding with content attention, 9 attention heads               |
| `learned2d_withcontent_16heads.yaml`      | Learned embedding with content attention, 16 attention heads              |
| `learned2d_withcontent_9heads_hier.yaml`  | Hierarchical learned embedding with content attention, 9 attention heads  |
| `learned2d_withcontent_16heads_hier.yaml` | Hierarchical learned embedding with content attention, 16 attention heads |
| `resnet.yaml`                             | ResNet-18 model                                                           |
| `san_pairwise.yaml`                       | Vanilla SAN (pairwise)                                                    |
| `san_patchwise.yaml`                      | Vanilla SAN (patchwise)                                                   |
| `san_pairwise_hier.yaml`                  | Hierarchical SAN (pairwise)                                               |
| `san_patchwise_hier.yaml`                 | Hierarchical SAN (patchwise)                                              |
| `vit.yaml`                                | Vanilla Vision Transformer                                                |
| `vit_hier.yaml`                           | Hierarchical Vision Transformer                                           |


## Task options
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
  - `time`: Compute the average throughput and inference time of
  

## Comparing model performance
After the models have been trained, one can plot the evolution of train or validation accuracy/loss using `plot_performance.py`. To do so, collect the output directories (containing `trainlogs.txt` in them) into a folder inside `Self-Attention-and-Convolutions`. Then, run the python script using the following CLI arguments.

| Argument   | Default value    | Description                                                                                           |
|------------|------------------|-------------------------------------------------------------------------------------------------------|
| `--root`   | `./data_to_plot` | Path to folder containing the output directories from experiments whose performance is to be plotted. |
| `--which`  | `val`            | Whether to plot train metrics (`train`) or validation metrics (`val`)                                 |
| `--metric` | `accuracy`       | The quantity to plot for chosen experiments, whether loss (`loss`) or accuracy (`accuracy`)           |

For example, to plot validation accuracies for experiments stored in `Self-Attention-and-Convolutions/data_to_plot/`, run the following command inside `Self-Attention-and-Convolutions`:

```
python3 plot_performance.py --root './data_to_plot' --which 'val' --metric 'accuracy'
```

Note that the series in the plot will carry the names of the experiment folders collected inside `--root` in the legend. Please rename the experiment folders in accordance with the names required on the plot legend.