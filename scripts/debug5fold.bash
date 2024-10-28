#!/bin/bash
source ~/.bashrc
conda activate nnunet_env_2

# Define environment variables, make directories required by nnUNet
DATA_DIR="original_data"
export nnUNet_raw="train/nnUNet_raw"
export nnUNet_preprocessed="train/nnUNet_preprocessed"
export nnUNet_results="train/nnUNet_results"

# mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

# # Move data into nnUNet dataset format
DATASET_ID=555
DATASET_NAME=Rhizo24
dataset_string="Dataset${DATASET_ID}_${DATASET_NAME}"
TRAIN_DATASET_PATH="${nnUNet_raw%/}/$dataset_string"
PREPROCESSED_DATA_PATH="${nnUNet_preprocessed%/}/$dataset_string"
RESULTS_DATA_PATH=="${nnUNet_results%/}/$dataset_string"

# python3 scripts/utils.py nnUNet_name_conversion $DATA_DIR $TRAIN_DATASET_PATH
echo "Name conversion successful!"

# Run nnUNet preprocessing which extracts statistics about the data and gives the parameters for an optimized UNet
# cp scripts/dataset.json $TRAIN_DATASET_PATH
# nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity -pl nnUNetPlannerResEncM 
# echo "Preprocessing successful!"
# cp scripts/splits_final_5fold.json $PREPROCESSED_DATA_PATH/splits_final.json



TRAINER=nnUNetTrainer_2epochs
PLANS=nnUNetResEncUNetMPlans

wait

# CUDA_VISIBLE_DEVICES=0 nnUNetv2_train -p $PLANS --c $DATASET_ID 2d 0 -tr $TRAINER &
# CUDA_VISIBLE_DEVICES=1 nnUNetv2_train -p $PLANS --c $DATASET_ID 2d 1 -tr $TRAINER &
# CUDA_VISIBLE_DEVICES=2 nnUNetv2_train -p $PLANS --c $DATASET_ID 2d 2 -tr $TRAINER &
# CUDA_VISIBLE_DEVICES=3 nnUNetv2_train -p $PLANS --c $DATASET_ID 2d 3 -tr $TRAINER &

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train -p nnUNetResEncUNetMPlans  --c $DATASET_ID 2d 0 -tr nnUNetTrainer_2epochs &
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train -p nnUNetResEncUNetMPlans  --c $DATASET_ID 2d 1 -tr nnUNetTrainer_2epochs &
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train -p nnUNetResEncUNetMPlans  --c $DATASET_ID 2d 2 -tr nnUNetTrainer_2epochs &
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train -p nnUNetResEncUNetMPlans  --c $DATASET_ID 2d 3 -tr nnUNetTrainer_2epochs &





# Wait for one of the jobs to finish before starting the 5th job
wait -n  # Waits for any background job to finish

# Run the 5th job on the first available GPU
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train -p $PLANS --c $DATASET_ID 2d 4 -tr $TRAINER
