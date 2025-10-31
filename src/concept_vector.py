"""used to measure the linearity of the model"""

import argparse
import transformer_lens
import datasets
import torch
import os
from sklearn.decomposition import IncrementalPCA
from loguru import logger
from tqdm import tqdm


def concept_vector():
    """used to calculate the concept vector of the model"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="EleutherAI/pythia-70m",
        choices=["google/gemma-2-2b-it", "Qwen/Qwen3-1.7B", "EleutherAI/pythia-70m"],
    )
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument(
        "--dataset_path", type=str, default="assets/paired_contexts/en-fr.jsonl"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--methods",
        type=str,
        default="difference-in-means",
        choices=["IncrementalPCA", "difference-in-means"],
    )
    args = parser.parse_args()
    logger.add("logs/concept_vector.log")
    logger.info("Starting concept vector calculation...")
    logger.info(f"Loading model: {args.model}")
    model = transformer_lens.HookedTransformer.from_pretrained(
        args.model, device=args.device, dtype=args.dtype
    )
    logger.info(f"Loading dataset: {args.dataset_path}")
    dataset = datasets.load_dataset("json", data_files=args.dataset_path)["train"]
    logger.info(f"Loaded {len(dataset)} examples")
    if args.methods == "difference-in-means":
        logger.info(f"Calculating concept vectors using difference-in-means...")
        difference_in_means = DifferenceInMeans(
            model, dataset, layer=args.layer, device=args.device
        )
        concept_vectors = difference_in_means.get_concept_vectors()
    else:
        raise ValueError(f"Invalid method: {args.methods}")
    os.makedirs("weights/concept_vectors", exist_ok=True)
    torch.save(concept_vectors, f"weights/concept_vectors/{args.model.split('/')[-1]}_Layer{args.layer}_{args.methods}_{args.dataset_path.split('/')[-1].replace('.jsonl', '')}.pt")
    logger.info(f"Saved concept vectors to weights/concept_vectors/{args.model.split('/')[-1]}_Layer{args.layer}_{args.methods}_{args.dataset_path.split('/')[-1].replace('.jsonl', '')}.pt")
    logger.info(f"Concept vectors shape: {concept_vectors.shape}")
    logger.info(f"Concept vectors: {concept_vectors}")

class DifferenceInMeans:
    def __init__(self, model, dataset, layer, device):
        self.model = model
        self.dataset = dataset
        self.layer = layer
        self.device = device

    def get_concept_vectors(self):
        model_dimension = self.model.cfg.d_model
        positive_concept_vector = torch.zeros(model_dimension, device=self.device)
        negative_concept_vector = torch.zeros(model_dimension, device=self.device)
        token_length = 0
        only_once = True
        for example in tqdm(self.dataset, desc="Calculating concept vectors"):
            positive_context = example["contexts0"]
            _, positive_cache = self.model.run_with_cache(
                positive_context, stop_at_layer=self.layer + 1
            )
            positive_hidden_state = (
                positive_cache[f"blocks.{self.layer}.hook_resid_post"]
                .reshape(-1, model_dimension)
                .mean(dim=0)
            )
            if only_once:
                logger.info(f"hidden state shape: {positive_hidden_state.shape}")
                only_once = False
            positive_concept_vector += positive_hidden_state
            negative_context = example["contexts1"]
            _, negative_cache = self.model.run_with_cache(
                negative_context, stop_at_layer=self.layer + 1
            )
            negative_hidden_state = (
                negative_cache[f"blocks.{self.layer}.hook_resid_post"]
                .reshape(-1, model_dimension)
                .mean(dim=0)
            )
            negative_concept_vector += negative_hidden_state
            token_length += positive_cache[
                f"blocks.{self.layer}.hook_resid_post"
            ].shape[-2]
        logger.info(f"Token length: {token_length}")
        positive_concept_vector /= token_length
        negative_concept_vector /= token_length
        concept_diff = positive_concept_vector - negative_concept_vector
        return torch.nn.functional.normalize(concept_diff, dim=0)


if __name__ == "__main__":
    concept_vector()
