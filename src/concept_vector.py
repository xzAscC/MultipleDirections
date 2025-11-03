"""used to measure the linearity of the model"""

import argparse
import transformer_lens
import datasets
import torch
import os
import gc
from sklearn.decomposition import IncrementalPCA
from loguru import logger
from tqdm import tqdm


def concept_vector():
    """used to calculate the concept vector of the model"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="EleutherAI/pythia-70m ",
        choices=["google/gemma-2-2b-it", "Qwen/Qwen3-1.7B", "EleutherAI/pythia-70m"],
    )
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument(
        "--dataset_path", type=str, default="assets/paired_contexts/en-fr.jsonl", choices=["assets/paired_contexts/en-fr.jsonl", ]
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
        positive_token_length = 0
        negative_token_length = 0
        only_once = True
        for example in tqdm(self.dataset, desc="Calculating concept vectors"):
            torch.cuda.empty_cache()
            gc.collect()
            for context in example["contexts0"]:
                _, positive_cache = self.model.run_with_cache(
                    context, stop_at_layer=self.layer + 1
                )
                positive_hidden_state = (
                    positive_cache[f"blocks.{self.layer}.hook_resid_post"]
                    .reshape(-1, model_dimension)
                    .mean(dim=0)
                )
                positive_concept_vector += positive_hidden_state
            for context in example["contexts1"]:
                _, negative_cache = self.model.run_with_cache(
                    context, stop_at_layer=self.layer + 1
                )
                negative_hidden_state = (
                    negative_cache[f"blocks.{self.layer}.hook_resid_post"]
                    .reshape(-1, model_dimension)
                    .mean(dim=0)
                )
                negative_concept_vector += negative_hidden_state
            positive_token_length += len(example["contexts0"])
            negative_token_length += len(example["contexts1"])
        logger.info(f"Positive token length: {positive_token_length}")
        logger.info(f"Negative token length: {negative_token_length}")
        positive_concept_vector /= positive_token_length
        negative_concept_vector /= negative_token_length
        concept_diff = positive_concept_vector - negative_concept_vector
        return torch.nn.functional.normalize(concept_diff, dim=0)


def random_concept_vector():
    """used to calculate the random concept vector of the model"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-it", choices=["EleutherAI/pythia-70m", "google/gemma-2-2b-it"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()
    model = transformer_lens.HookedTransformer.from_pretrained(
        args.model, device=args.device, dtype=args.dtype
    )
    logger.info(f"Loading model: {args.model}")
    model_dimension = model.cfg.d_model
    logger.info(f"Model dimension: {model_dimension}")
    random_concept_vector = torch.randn(model_dimension)
    random_concept_vector = torch.nn.functional.normalize(random_concept_vector, dim=0)
    random_concept_vector = random_concept_vector.to(args.device)
    os.makedirs("weights/concept_vectors", exist_ok=True)
    torch.save(random_concept_vector, f"weights/concept_vectors/random_concept_vector_{args.model.split('/')[-1]}.pt")
    logger.info(f"Saved random concept vector to weights/concept_vectors/random_concept_vector_{args.model.split('/')[-1]}.pt")
    logger.info(f"Random concept vector shape: {random_concept_vector.shape}")
    logger.info(f"Random concept vector: {random_concept_vector.norm(dim=0).item()}")
    return random_concept_vector

def harmful_concept_vector():
    """used to calculate the harmful concept vector of the model"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-it", choices=["EleutherAI/pythia-70m", "google/gemma-2-2b-it"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()
    model = transformer_lens.HookedTransformer.from_pretrained(
        args.model, device=args.device, dtype=args.dtype)
    logger.info(f"Loading model: {args.model}")
    model_dimension = model.cfg.d_model
    logger.info(f"Model dimension: {model_dimension}")
    harmful_concept_vector = torch.randn(model_dimension)
    harmful_concept_vector = torch.nn.functional.normalize(harmful_concept_vector, dim=0)
    harmful_concept_vector = harmful_concept_vector.to(args.device)
    os.makedirs("weights/concept_vectors", exist_ok=True)
    torch.save(harmful_concept_vector, f"weights/concept_vectors/harmful_concept_vector_{args.model.split('/')[-1]}.pt")
    logger.info(f"Saved harmful concept vector to weights/concept_vectors/harmful_concept_vector_{args.model.split('/')[-1]}.pt")
    logger.info(f"Harmful concept vector shape: {harmful_concept_vector.shape}")
    logger.info(f"Harmful concept vector: {harmful_concept_vector.norm(dim=0).item()}")
    return harmful_concept_vector

if __name__ == "__main__":
    # random_concept_vector()
    # concept_vector()
    
    harmful_concept_vector()