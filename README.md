## Project Description
Biofuels represent renewable energy sources extracted from organic materials, such as plants or plant remnants, and can serve as an eco-friendly substitute for conventional fossil fuels. Gaining knowledge of plant roots can support research on optimizing nutrient accessibility, improved nutrient absorption, and enhanced plant growth and biomass yield. Flatbed scanners are commonly used to image roots, but require manual segmentation of plant roots for analysis. To scale the current methods to larger studies, an automated analysis is desirable; however, the complex nature of root structures and image noise makes this difficult. To this end, Sordo et al. introduced RhizoNet, a deep learning-based workflow based on a custom residual U-Net and a convex hull post processing to semantically segment plant root scans. In this repo, the nnU-Net model training architecture was adapted to plant root segmentation. nnU-Net is a semantic segmentation method that automatically adapts to a given dataset by analyzing the training data and automatically configuring a pipeline. This framework is currently the state-of-the-art in many biomedical segmentation tasks, despite being an out-of-the-box method.


## Installation
This project relies on [`nnU-Net`](https://github.com/MIC-DKFZ/nnU-Net/tree/master) along with other dependencies specified in a Conda environment file. The setup has been tested on a Linux environment and assumes specific package versions for compatibility. Follow the instructions below to set up everything required for training and evaluation:


#### First, clone this repository and navigate to the base directory
```shell
git clone https://github.com/juampabloheras/rhizo24.git
cd rhizo24
```
**Make sure you are in the `rhizo24` directory when running the following commands.**

#### Pull `nnU-Net` as a submodule
```shell
git clone https://github.com/MIC-DKFZ/nnUNet.git
```
All nnU-Net commands have a `-h` option which gives information on how to use them.


*[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)* **(Recommended)** - Create and activate a `nnunet_env` conda environment using the environment definition provided in `conda.yaml`:

```shell
conda env create -f conda.yaml
conda activate nnunet_env
```

#### Verification

To verify a successful installation, including checking CUDA compatibility, run the following command:

```shell
python -c "import nnunet; import torch; print('nnU-Net successfully imported'); print('CUDA available:', torch.cuda.is_available())"
```


These scripts will handle the setup and configuration for each training approach, using the specified JSON configuration files.

This command will confirm that `nnU-Net` was imported successfully and whether CUDA is available for PyTorch. If CUDA available: False, it may be due to an incompatible CUDA toolkit version; ensure that your installed CUDA version matches the version supported by your PyTorch installation by checking the [PyTorch Get Started Guide](https://pytorch.org/get-started/previous-versions/) for compatible combinations of PyTorch and CUDA versions.


## Training Instructions

To train the model with an 80-20 validation split or a 5-fold cross-validation split, use the appropriate `sbatch` command with the provided Slurm scripts.

**NOTE: Make sure you’re in the `rhizo24` directory when running the following commands in your bash terminal:**


### Training Commands

- **For 5-fold Cross-Validation Training**:
  ```shell
  sbatch scripts/train_nnUNet_5folds.slurm
  ```

- **For 80-20 Validation Split Training**:
  ```shell
  sbatch scripts/train_nnUNet_80_10.slurm
  ```


