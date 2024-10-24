#!/bin/bash

conda create --name nnunet_env python=3.9 -y
source ~/.bashrc
conda activate nnunet_env
conda install pytorch torchvision torchaudio -c pytorch  -y

if python -c "import torch; x = torch.rand(5, 3); print('cuda avail: ', torch.cuda.is_available())"
then
    echo "Environment compiled successfully."
else
    echo "Failed to import PyTorch. Check your installation."
fi

git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .

echo "nnU-Net setup completed successfully."



