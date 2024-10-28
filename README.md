
## Installation
This project relies on `nnUNet` along with other dependencies specified in a Conda environment file. The setup has been tested on a Linux environment and assumes specific package versions for compatibility. Follow the instructions below to set up everything required for training and evaluation:


### First, clone this repository and navigate to the base directory
```shell
git clone https://github.com/juampabloheras/rhizo24.git
cd rhizo24
```

### Pull nnUNet as a submodule and ensure it's up to date
```shell
git submodule add https://github.com/MIC-DKFZ/nnUNet.git nnUNet
```


*[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)* **(Recommended)** - Create and activate a `nnunet_env` conda environment using the provided environment definition:

```shell
conda env create -f conda.yaml
conda activate nnunet_env
```

### Verification

To verify a successful installation, including checking CUDA compatibility, run the following command:

```shell
python -c "import nnunet; import torch; print('nnUNet successfully imported'); print('CUDA available:', torch.cuda.is_available())"
```

This command will confirm that `nnUNet` was imported successfully and whether CUDA is available for PyTorch. If CUDA available: False, it may be due to an incompatible CUDA toolkit version; ensure that your installed CUDA version matches the version supported by your PyTorch installation by checking the [PyTorch Get Started Guide](https://pytorch.org/get-started/previous-versions/) for compatible combinations of PyTorch and CUDA versions.

