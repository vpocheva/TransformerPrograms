#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Define the array of different settings
# declare -a DATASETS=("reverse" "hist" "double_hist" "sort" "most_freq")
declare -a DATASETS=("hist" "double_hist" "sort" "most_freq")
declare -a VOCAB_TYPES=( "V=8 N=8" "V=8 N=16" "V=16 N=16" )
declare -a DATASET_SIZES=("5000" "10000" "20000" "50000" "100000")

# Loop over each combination of dataset, vocab type, and dataset size
for DATASET_SETTINGS in "${DATASETS[@]}"; do
  for VOCAB_TYPE in "${VOCAB_TYPES[@]}"; do
    for DATASET_SIZE in "${DATASET_SIZES[@]}"; do
      # Extract specific settings based on dataset
      if [[ "$DATASET_SETTINGS" == "reverse" ]]; then
        L=3
        H=8
        M=2
      elif [[ "$DATASET_SETTINGS" == "hist" ]]; then
        L=1
        H=4
        M=2
      elif [[ "$DATASET_SETTINGS" == "double_hist" ]]; then
        L=3
        H=4
        M=2
      elif [[ "$DATASET_SETTINGS" == "sort" ]]; then
        L=3
        H=8
        M=4
      elif [[ "$DATASET_SETTINGS" == "most_freq" ]]; then
        L=3
        H=8
        M=4
      fi

      # Set vocab and length
      eval $VOCAB_TYPE

      # Call the original script with current settings

      # print the settings
      echo ""
      echo "DATASET=$DATASET_SETTINGS DATASET_SIZE=$DATASET_SIZE L=$L H=$H M=$M V=$V N=$N"
      echo ""

      DATASET=$DATASET_SETTINGS DATASET_SIZE=$DATASET_SIZE L=$L H=$H M=$M V=$V N=$N \
      sh scripts/rasp.sh
    done
  done
done
