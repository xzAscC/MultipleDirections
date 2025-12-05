#!/bin/bash

# define the models
models=(
  "EleutherAI/pythia-70m"
  # "EleutherAI/pythia-160m"
#   "EleutherAI/pythia-410m"
)

# define the layer and method
method="difference-in-means"

metric="norm"
# loop through the models
for model_name in "${models[@]}"; do
    echo "Running for model=$model_name"
    uv run src/linearity.py \
      --model "$model_name" \
      --concept_vector_dataset_size 300 \
      --concept_vector_alpha 1 \
      --alpha_factor 1000 \
      --linearity_dataset_size 30 \
      --seed 42 \
      --linearity_metric "$metric"
done
