#!/bin/bash

# define the models
models=(
  "google/gemma-2-2b"
  "Qwen/Qwen3-1.7B"
  "EleutherAI/pythia-70m"
)

# define the concepts
concepts=(
  "safety"
  "language_en_fr"
  "random"
)

# define the layer and method
layer=-1
method="difference-in-means"

# loop through the models and concepts
for model_name in "${models[@]}"; do
  for concept_category in "${concepts[@]}"; do
    echo "Running for model=$model_name, concept=$concept_category"
    python src/concept_vector.py \
      --model "$model_name" \
      --layer "$layer" \
      --concept_category "$concept_category" \
      --method "$method"
  done
done
