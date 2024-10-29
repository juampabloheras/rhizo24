## nnU-Net for plant root segmentation
Biofuels represent renewable energy sources extracted from organic materials and are an eco-friendly alternative to conventional fossil fuels. Gaining knowledge of plant roots can support research on optimizing nutrient accessibility, improved nutrient absorption, and enhanced plant growth and biomass yield, which are determinants of crop productivity and sustainability[^1]. Flatbed scanners are commonly used to image roots, but require manual segmentation of plant roots for analysis. To scale the current methods to larger studies, an automated analysis is desirable; however, the complex nature of root structures and image noise makes this difficult. To this end, Sordo et al. introduced RhizoNet[^2], a deep learning-based workflow based on a custom residual U-Net and a convex hull post processing to semantically segment plant root scans. 

In this repo, the nnU-Net[^3] model training architecture is assessed for plant root segmentation. nnU-Net is a semantic segmentation method that automatically adapts to a given dataset by analyzing the training data and automatically configuring a pipeline. This framework is currently the state-of-the-art in many biomedical segmentation tasks, despite being an out-of-the-box method.

[^1]:York, L. M. et al. Bioenergy underground: Challenges and opportunities for phenotyping roots and the microbiome for sustainable bioenergy crop production. Plant Phenome J. 5, e20028. https://doi.org/10.1002/ppj2.20028 (2022)
[^2]: Sordo, Z., Andeer, P., Sethian, J. et al. RhizoNet segments plant roots to assess biomass and growth for enabling self-driving labs. Sci Rep 14, 12907 (2024). https://doi.org/10.1038/s41598-024-63497-8 
[^3]: Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211. https://doi.org/10.1038/s41592-020-01008-z


## Installation
This project relies on [`nnU-Net`](https://github.com/MIC-DKFZ/nnU-Net/tree/master) along with other dependencies specified in a Conda environment file. The setup has been tested on a Linux environment and assumes specific package versions for compatibility. Follow the instructions below to set up everything required for training and evaluation:

### 1. Clone this Repository
First, clone this repository and navigate to the base directory:

```shell
git clone https://github.com/juampabloheras/rhizo24.git
cd rhizo24
```

**Make sure you are in the `rhizo24` directory when running the following commands.**

### 2. Clone `nnU-Net`
```shell
git clone https://github.com/MIC-DKFZ/nnUNet.git
```
All nnU-Net commands have a `-h` option which gives information on how to use them.

### 3. Create the Conda Environment

*[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)* **(Recommended)** - Create and activate a `nnunet_env` Conda environment using the environment definition provided in `conda.yaml`:

```shell
conda env create -f conda.yaml
conda activate nnunet_env
```

### 4. Verification

To verify a successful installation, including checking CUDA compatibility, run the following command:

```shell
python -c "import nnunet; import torch; print('nnU-Net successfully imported'); print('CUDA available:', torch.cuda.is_available())"
```

This command will confirm that `nnU-Net` was imported successfully and whether CUDA is available for PyTorch. 

> **Note:** If `CUDA available: False`, it may be due to an incompatible CUDA toolkit version. Ensure that your installed CUDA version matches the version supported by your PyTorch installation by checking the [PyTorch Get Started Guide](https://pytorch.org/get-started/previous-versions/) for compatible combinations of PyTorch and CUDA versions.





## Training Instructions

To train the model with an 80-20 validation split or a 5-fold cross-validation split on NERSC, use the appropriate `sbatch` command with the provided Slurm scripts.

**NOTE: Make sure you’re in the `rhizo24` directory when running the following commands in your bash terminal:**


### Commands

- **For 5-fold Cross-Validation Training**:
  ```shell
  sbatch scripts/train_nnUNet_5folds.slurm
  ```

- **For 80-20 Validation Split Training**:
  ```shell
  sbatch scripts/train_nnUNet_80_10.slurm
  ```

## Inference Instructions

After training, to run inference on NERSC with either the 80-20 validation split or the 5-fold cross-validation setup, use the appropriate `sbatch` command with the provided Slurm scripts.

### Commands

- **For 5-Fold Cross-Validation Inference**:
  ```shell
  sbatch scripts/inference_5fold.slurm
  ```

- **For 80-20 Validation Split Inference**:
  ```shell
  sbatch scripts/inference_80_10.slurm
  ```

During inference, metrics and file paths for each processed image are saved in the `final_results` directory as JSON files with a structure similar to the example below:

```json
{
    "image_path": "path/to/imagesTr/rhizo_021_0000.tif",
    "preds_path": "path/to/preds/rhizo_021.tif",
    "label_path": "path/to/labelsTr/rhizo_021.tif",
    "accuracy": 0.9622397685777967,
    "precision": 1.0,
    "IoU": 0.9622397685777967,
    "Dice": 0.9807565660288439,
    "recall": 0.9622397685777967
}
```
![results](https://github.com/user-attachments/assets/f3bea315-1e61-41b3-a38d-dba25170b3af)


