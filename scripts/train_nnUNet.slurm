#!/bin/bash
#SBATCH -N 1 
#SBATCH -C gpu  
#SBATCH -G 1
#SBATCH -q debug  
#SBATCH -J train-nnunet
#SBATCH --mail-type=ALL  
#SBATCH --time=00:30:00  
#SBATCH -A m4055 
#SBATCH --error=slurm_error.txt
#SBATCH --output=slurm_output.txt

cd "$(dirname "$0")"

#!/bin/bash
source ~/.bashrc
conda activate nnunet_env
cd "$(dirname "$0")"

# # Define environment variables, make directories required by nnUNet
DATA_DIR="../original_data"
export nnUNet_raw="../train/nnUNet_raw"
export nnUNet_preprocessed="../train/nnUNet_preprocessed"
export nnUNet_results="../train/nnUNet_results"

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

# # Move data into nnUNet dataset format
DATASET_ID=111
DATASET_NAME=Rhizo24
dataset_string="Dataset${DATASET_ID}_${DATASET_NAME}"
TRAIN_DATASET_PATH="${nnUNet_raw%/}/$dataset_string"
PREPROCESSED_DATA_PATH="${nnUNet_preprocessed%/}/$dataset_string"
RESULTS_DATA_PATH=="${nnUNet_results%/}/$dataset_string"

python3 utils.py nnUNet_name_conversion $DATA_DIR $TRAIN_DATASET_PATH
echo "Name conversion successful!"

# Run nnUNet preprocessing which extracts statistics about the data and gives the parameters for an optimized UNet
cp dataset.json $TRAIN_DATASET_PATH
nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity -pl nnUNetPlannerResEncM
echo "Preprocessing successful!"
cp splits_final.json $PREPROCESSED_DATA_PATH


CUDA_VISIBLE_DEVICES=0 nnUNetv2_train -p nnUNetResEncUNetMPlans  --c $DATASET_ID 2d 0


