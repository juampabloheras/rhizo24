#!/bin/bash
source ~/.bashrc
conda activate nnunet_env
cd "$(dirname "$0")"
# Define environment variables, make directories required by nnUNet
DATA_DIR="../original_data"
export nnUNet_raw="../train/nnUNet_raw"
export nnUNet_preprocessed="../train/nnUNet_preprocessed"
export nnUNet_results="../train/nnUNet_results"

DATASET_ID=111
DATASET_NAME=Rhizo24
dataset_string="Dataset${DATASET_ID}_${DATASET_NAME}"
TRAIN_DATASET_PATH="${nnUNet_raw%/}/$dataset_string"

INPUT_FOLDER=${TRAIN_DATASET_PATH}/imagesTr
OUTPUT_FOLDER=../final_results
TASK_NAME_OR_ID=111
CONFIGURATION=2d
FOLD=0
PLAN=nnUNetResEncUNetMPlans

nnUNetv2_predict -i $INPUT_FOLDER -o $OUTPUT_FOLDER -d $TASK_NAME_OR_ID -c $CONFIGURATION -f $FOLD -p $PLAN

