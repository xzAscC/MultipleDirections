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

    # configure the arguments

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="EleutherAI/pythia-70m",
        choices=["google/gemma-2-2b-it", "Qwen/Qwen3-1.7B", "EleutherAI/pythia-70m"],
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="the layer to calculate the concept vector, -1 means the all layers",
    )

    parser.add_argument(
        "--concept_category", type=str, default="safety", choices=["safety", "language"]
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

    # add logger
    logger.add("logs/concept_vector.log")
    logger.info("Starting concept vector calculation...")

    # capture the dataset path
    if args.concept_category == "safety":
        # for all binary dataset, use two datasets to calculate the concept vector
        dataset_folder = "assets/harmbench"
        harmful_dataset_path = os.path.join(dataset_folder, "harmful_data.jsonl")
        harmless_dataset_path = os.path.join(dataset_folder, "harmless_data.jsonl")
        positive_dataset = datasets.load_dataset(
            "json", data_files=harmful_dataset_path
        )["train"]
        negative_dataset = datasets.load_dataset(
            "json", data_files=harmless_dataset_path
        )["train"]
        datset_key = "instruction"
    elif args.concept_category == "language":
        dataset_folder = "assets/language_translation"
        # TODO: add the language translation dataset path
    else:
        raise ValueError(f"Invalid concept category: {args.concept_category}")

    logger.info(f"Loading model: {args.model}")
    model = transformer_lens.HookedTransformer.from_pretrained(
        args.model, device=args.device, dtype=args.dtype
    )

    if args.methods == "difference-in-means":
        logger.info(f"Calculating concept vectors using difference-in-means...")
        difference_in_means = DifferenceInMeans(
            model,
            positive_dataset,
            negative_dataset,
            layer=args.layer,
            device=args.device,
            dataset_key=datset_key,
        )
        concept_vectors = difference_in_means.get_concept_vectors(
            save_path=f"weights/concept_vectors/{args.model.split('/')[-1]}_Layer{args.layer}_{args.methods}_{args.concept_category}.pt",
            is_save=True,
        )
    else:
        raise ValueError(f"Invalid method: {args.methods}")


class DifferenceInMeans:
    def __init__(
        self,
        model: transformer_lens.HookedTransformer,
        positive_dataset: datasets.Dataset,
        negative_dataset: datasets.Dataset,
        layer: int,
        device: str,
        dataset_key: str,
        max_dataset_size: int = 300,
    ):
        """used to calculate the concept vector using difference-in-means

        Args:
            model (transformer_lens.HookedTransformer): the model to calculate the concept vector
            positive_dataset (datasets.Dataset): the positive dataset to calculate the concept vector
            negative_dataset (datasets.Dataset): the negative dataset to calculate the concept vector
            layer (int): the layer to calculate the concept vector
            device (str): the device to calculate the concept vector
            dataset_key (str): the key of the dataset to calculate the concept vector
            max_dataset_size (int, optional): the maximum size of the dataset to calculate the concept vector. Defaults to 300.
        """
        self.model = model
        self.positive_dataset = positive_dataset
        self.negative_dataset = negative_dataset
        if layer == -1:
            self.layers = list(range(self.model.cfg.n_layers))
            logger.info(f"Calculating concept vectors for all layers: {self.layers}")
        else:
            self.layers = [layer]
            logger.info(f"Calculating concept vectors for layer: {layer}")
        self.device = device
        self.dataset_key = dataset_key
        self.max_dataset_size = max_dataset_size

    def get_concept_vectors(self, save_path: str, is_save: bool = False):
        """used to calculate the concept vectors using difference-in-means

        Args:
            save_path (str): the path to save the concept vectors
            is_save (bool, optional): whether to save the concept vectors. Defaults to False.

        Returns:
            torch.Tensor: the concept vectors
        """
        model_dimension = self.model.cfg.d_model
        layer_length = len(self.layers)
        positive_concept_vector = torch.zeros(
            layer_length, model_dimension, device=self.device
        )
        negative_concept_vector = torch.zeros(
            layer_length, model_dimension, device=self.device
        )
        positive_token_length = 0
        negative_token_length = 0
        only_once = True
        for i, example in enumerate(tqdm(
            self.positive_dataset, desc="Calculating positive concept vectors"
        )):
            if i >= self.max_dataset_size:
                break
            torch.cuda.empty_cache()
            gc.collect()
            context = example[self.dataset_key]
            _, positive_cache = self.model.run_with_cache(context)
            for layer in self.layers:
                positive_hidden_state = positive_cache[
                    f"blocks.{layer}.hook_resid_post"
                ].reshape(-1, model_dimension)
                current_token_length = positive_hidden_state.shape[0]
                positive_concept_vector[layer] += positive_hidden_state.sum(dim=0)
                positive_token_length += current_token_length
                
        for i, example in enumerate(tqdm(
            self.negative_dataset, desc="Calculating negative concept vectors"
        )):
            if i >= self.max_dataset_size:
                break
            torch.cuda.empty_cache()
            gc.collect()
            context = example[self.dataset_key]
            _, negative_cache = self.model.run_with_cache(context)
            for layer in self.layers:
                negative_hidden_state = negative_cache[
                    f"blocks.{layer}.hook_resid_post"
                ].reshape(-1, model_dimension)
                current_token_length = negative_hidden_state.shape[0]
                negative_concept_vector[layer] += negative_hidden_state.sum(dim=0)
                negative_token_length += current_token_length
        positive_concept_vector /= positive_token_length
        negative_concept_vector /= negative_token_length
        concept_diff = positive_concept_vector - negative_concept_vector
        concept_diff = torch.nn.functional.normalize(concept_diff, dim=1)

        torch.save(concept_diff, save_path)
        logger.info(f"Concept vector shape: {concept_diff.shape}")
        logger.info(f"Concept vector: {concept_diff.norm(dim=1)}")
        return concept_diff


def random_concept_vector():
    """used to calculate the random concept vector of the model"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-2b-it",
        choices=["EleutherAI/pythia-70m", "google/gemma-2-2b-it"],
    )
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
    torch.save(
        random_concept_vector,
        f"weights/concept_vectors/random_concept_vector_{args.model.split('/')[-1]}.pt",
    )
    logger.info(
        f"Saved random concept vector to weights/concept_vectors/random_concept_vector_{args.model.split('/')[-1]}.pt"
    )
    logger.info(f"Random concept vector shape: {random_concept_vector.shape}")
    logger.info(f"Random concept vector: {random_concept_vector.norm(dim=0).item()}")
    return random_concept_vector


if __name__ == "__main__":
    # random_concept_vector()
    # concept_vector()

    concept_vector()
