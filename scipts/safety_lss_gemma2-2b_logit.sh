concept_vector_path="weights/concept_vectors/gemma-2-2b/random/random_layer26.pt"
model_name="google/gemma-2-2b"
uv run src/linearity.py \
    --model "$model_name" \
    --concept_vector_path "$concept_vector_path" \
    --linearity_metric "lss" \
    --influence_layer "logits" 

