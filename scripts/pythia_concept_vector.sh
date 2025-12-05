#!/bin/bash

# define the models
models=(
  # "EleutherAI/pythia-70m"
  "EleutherAI/pythia-160m"
#   "EleutherAI/pythia-410m"
)

# define the layer and method
method="difference-in-means"
linearity_metric=("lsr" "lss")
remove_flags=("--remove_concept_vector")

# loop through the models
for model_name in "${models[@]}"; do
    for flag in "${remove_flags[@]}"; do
      echo "Running for model=$model_name  flag='$flag'"
      uv run src/linearity.py \
        --model "$model_name" \
        --concept_vector_dataset_size 300 \
        --concept_vector_alpha 1 \
        --alpha_factor 1000 \
        --linearity_dataset_size 30 \
        --seed 42 \
        --concept_vector_pretrained
        --linearity_metric "${linearity_metric[@]}" \
        $flag
      done
done