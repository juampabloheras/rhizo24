#!/bin/bash
source ~/.bashrc
conda activate nnunet_env_2

# Define environment variables, make directories required by nnUNet
DATA_DIR=original_data
export nnUNet_raw=train/nnUNet_raw
export nnUNet_preprocessed=train/nnUNet_preprocessed
export nnUNet_results=train/nnUNet_results

DATASET_ID=111
DATASET_NAME=Rhizo24
dataset_string=Dataset${DATASET_ID}_${DATASET_NAME}
TRAIN_DATASET_PATH=${nnUNet_raw%/}/$dataset_string

INPUT_FOLDER=${TRAIN_DATASET_PATH}/imagesTr
OUTPUT_FOLDER=final_results/nnunet_preds/split_80_10
TASK_NAME_OR_ID=111
CONFIGURATION=2d
FOLD=0
PLAN=nnUNetResEncUNetMPlans
TRAINER=nnUNetTrainer_750epochs

nnUNetv2_predict -i $INPUT_FOLDER -o $OUTPUT_FOLDER -d $TASK_NAME_OR_ID -c $CONFIGURATION -f $FOLD -p $PLAN -tr $TRAINER

RAW_IMAGE_FOLDER=$INPUT_FOLDER
GT_LABELS_FOLDER=${TRAIN_DATASET_PATH}/labelsTr
PREDS_FOLDER=$OUTPUT_FOLDER
NAME_CONVERSION_PATH=scripts/rhizonet_nnUNet_name_conversion.json
OUT_DIR=final_results/metrics/split_80_10

python3 scripts/utils.py process_and_visualize_subjects $RAW_IMAGE_FOLDER $GT_LABELS_FOLDER $PREDS_FOLDER $NAME_CONVERSION_PATH $OUT_DIR
