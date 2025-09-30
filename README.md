<h2 align="center">Cellulus-track</h2>

- **[Introduction](#introduction)**
- **[Installation](#installation)**
- **[Getting Started](#getting-started)**
- **[Citation](#citation)**
- **[Issues](#issues)**


### Introduction
Cellulus-track is a deep learning based method which can be used to obtain instance-segmentation and tracking of objects in 2D or 3D microscopy images in an unsupervised fashion i.e. requiring no ground truth labels during training.

### Installation

One could execute these lines of code below to create a new environment and install dependencies.

1. Create a new environment called `cellulus-track`:

```bash
conda create -y -n cellulus-track python==3.9
```

2. Activate the newly-created environment:

```
conda activate cellulus-track
```

3a. If using a GPU, install pytorch cuda dependencies:

```bash
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

3b. otherwise (if using a CPU or MPS), run:

```bash
pip install torch torchvision
```

4. Install the package from github:

```bash
pip install git+https://github.com/h-westmacott/cellulus-track
```


<!-- ### Getting Started

#### With Jupyter Notebooks

Try out a `2D example` or a `3D example` available under the examples tab **[here](https://funkelab.github.io/cellulus)**.

#### From the terminal

Using `cellulus` from the terminal window requires specifying a `train.toml` config file and an `infer.toml` config file. <br>
These files indicate how the training and inference should be performed respectively.

For example, a minimal `train.toml` config file would look as follows:

```toml
[model_config]

num_fmaps = 256
fmap_inc_factor = 3
downsampling_factors = [[2,2],]

[train_config.train_data_config]

container_path = "skin.zarr" # specify path to zarr container, containing raw image dataset
dataset_name = "train/raw"
```

The `train.toml` recipe file can then be used to initiate the model training by running the following line in the terminal window:
```bash
train train.toml
```

Similarly, a minimal `infer.toml` file would look as follows:

```toml
[model_config]

num_fmaps = 256
fmap_inc_factor = 3
checkpoint = "models/best_loss.pth" # path to model weights

[inference_config.dataset_config]

container_path = "skin.zarr" # specify path to zarr container, containing raw image dataset
dataset_name = "test/raw"

[inference_config.prediction_dataset_config]

container_path = "skin.zarr"
dataset_name = "embeddings"

[inference_config.detection_dataset_config]

container_path = "skin.zarr"
dataset_name = "detection"
secondary_dataset_name = "embeddings"

[inference_config.segmentation_dataset_config]

container_path = "skin.zarr"
dataset_name = "segmentation"
secondary_dataset_name = "detection"
```

The `infer.toml` recipe file can be used to apply the trained model weights on raw image data and obtain instance segmentations, by running the following line in the terminal window:

```bash
infer infer.toml
```

### Citation

If you find our work useful in your research, please consider citing:

```bibtex
@misc{wolf2023unsupervised,
      title={Unsupervised Learning of Object-Centric Embeddings for Cell Instance Segmentation in Microscopy Images},
      author={Steffen Wolf and Manan Lalit and Henry Westmacott and Katie McDole and Jan Funke},
      year={2023},
      eprint={2310.08501},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

### Issues

If you encounter any problems, please **[file an issue](https://github.com/funkelab/cellulus/issues)** along with a description. -->
