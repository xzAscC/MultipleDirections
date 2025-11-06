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
from typing import Union, List

MODEL_LAYERS = {
    "google/gemma-2-2b": 26,
    "Qwen/Qwen3-1.7B": 28,
    "EleutherAI/pythia-70m": 6,
}


def config():
    """config
    Returns:
        argparse.Namespace: the arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        choices=["google/gemma-2-2b", "Qwen/Qwen3-1.7B", "EleutherAI/pythia-70m"],
        help="the model to calculate the concept vector. TODO: add models with checkpoints and same model family",
    )
    parser.add_argument(
        "--layer",
        nargs="+",
        type=int,
        default=[-1],
        help="the layer to calculate the concept vector, -1 means the all layers",
    )

    parser.add_argument(
        "--concept_category",
        type=str,
        default="language_en_fr",
        choices=["safety", "language_en_fr", "random"],
        help="the category of the concept",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--methods",
        type=str,
        default="difference-in-means",
        choices=["IncrementalPCA", "difference-in-means", "rdo"],
    )
    parser.add_argument(
        "--layer_num",
        type=int,
        default=0,
        help="the number of layers, default is 0 and will be calculated automatically",
    )
    parser.add_argument(
        "--max_dataset_size",
        type=int,
        default=300,
        help="the maximum size of the dataset to calculate the concept vector",
    )
    args = parser.parse_args()
    return args


def concept_vector(save_path: str) -> None:
    """used to calculate the concept vector using difference-in-means

    Args:
        save_path (str): the path to save the concept vectors

    Returns:
        None
    """
    args = config()
    max_layers = MODEL_LAYERS[args.model]

    if -1 in args.layer:
        args.layer = list(range(max_layers))
    logger.info(f"Layer: {args.layer}")
    args.layer_num = len(args.layer)
    deepest_layer = max(args.layer)
    shallowest_layer = min(args.layer)

    if args.model not in MODEL_LAYERS:
        raise ValueError(f"Invalid model: {args.model}")

    assert (
        -1 <= shallowest_layer < max_layers and deepest_layer < max_layers
    ), f"layer must be between 0 and {max_layers - 1}"

    # add logger
    logger.add("logs/concept_vector.log")
    logger.info("Starting concept vector calculation...")

    # capture the dataset path
    if args.concept_category == "safety":
        # for all binary dataset, use two datasets to calculate the concept vector
        safety_concept_vector(args, save_path)
    elif args.concept_category == "language_en_fr":
        language_en_fr_concept_vector(args, save_path)
    elif args.concept_category == "random":
        random_concept_vector(args, save_path)
    else:
        raise ValueError(f"Invalid concept category: {args.concept_category}")


class DifferenceInMeans:
    def __init__(
        self,
        model: transformer_lens.HookedTransformer,
        positive_dataset: datasets.Dataset,
        negative_dataset: datasets.Dataset,
        layer: int,
        device: str,
        positive_dataset_key: str,
        negative_dataset_key: str,
        max_dataset_size: int = 300,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """used to calculate the concept vector using difference-in-means

        Args:
            model (transformer_lens.HookedTransformer): the model to calculate the concept vector
            positive_dataset (datasets.Dataset): the positive dataset to calculate the concept vector
            negative_dataset (datasets.Dataset): the negative dataset to calculate the concept vector
            layer (int): the layer to calculate the concept vector
            device (str): the device to calculate the concept vector
            positive_dataset_key (str): the key of the positive dataset to calculate the concept vector
            negative_dataset_key (str): the key of the negative dataset to calculate the concept vector
            max_dataset_size (int, optional): the maximum size of the dataset to calculate the concept vector. Defaults to 300.
        """
        self.model = model
        self.positive_dataset = positive_dataset
        self.negative_dataset = negative_dataset
        if layer == -1:
            self.layers = list(range(self.model.cfg.n_layers))
            logger.info(f"Calculating concept vectors for all layers: {self.layers}")
        else:
            self.layers = layer
            logger.info(f"Calculating concept vectors for layer: {layer}")
        self.device = device
        self.positive_dataset_key = positive_dataset_key
        self.negative_dataset_key = negative_dataset_key
        self.max_dataset_size = max_dataset_size
        self.dtype = dtype

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
            layer_length, model_dimension, device=self.device, dtype=self.dtype
        )
        negative_concept_vector = torch.zeros(
            layer_length, model_dimension, device=self.device, dtype=self.dtype
        )
        positive_token_length = 0
        negative_token_length = 0
        for i, example in tqdm(enumerate(self.positive_dataset), total=len(self.positive_dataset)):
            if i >= self.max_dataset_size:
                break
            torch.cuda.empty_cache()
            gc.collect()
            context = example[self.positive_dataset_key]
            _, positive_cache = self.model.run_with_cache(
                context
            )
            for layer in self.layers:
                positive_hidden_state = positive_cache[
                    f"blocks.{layer}.hook_resid_post"
                ].reshape(-1, model_dimension)
                positive_concept_vector[layer] += positive_hidden_state.sum(dim=0)
                if layer == 0:
                    current_token_length = positive_hidden_state.shape[0]
                    positive_token_length += current_token_length

        for i, example in tqdm(enumerate(self.negative_dataset), total=len(self.negative_dataset)):
            if i >= self.max_dataset_size:
                break
            torch.cuda.empty_cache()
            gc.collect()
            context = example[self.negative_dataset_key]
            _, negative_cache = self.model.run_with_cache(context, stop_at_layer=layer+1)
            for layer in self.layers:
                negative_hidden_state = negative_cache[
                    f"blocks.{layer}.hook_resid_post"
                ].reshape(-1, model_dimension)
                negative_concept_vector[layer] += negative_hidden_state.sum(dim=0)
                if layer == 0:
                    current_token_length = negative_hidden_state.shape[0]
                    negative_token_length += current_token_length
        positive_concept_vector /= positive_token_length
        negative_concept_vector /= negative_token_length
        concept_diff = positive_concept_vector - negative_concept_vector
        concept_diff = torch.nn.functional.normalize(concept_diff, dim=1)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(concept_diff, save_path)
        logger.info(f"Concept vector shape: {concept_diff.shape}")
        logger.info(f"Concept vector: {concept_diff.norm(dim=1)}")
        return concept_diff


def safety_concept_vector(args: argparse.Namespace, save_path: str) -> None:
    dataset_folder = "assets/harmbench"
    harmful_dataset_path = os.path.join(dataset_folder, "harmful_data.jsonl")
    harmless_dataset_path = os.path.join(dataset_folder, "harmless_data.jsonl")
    positive_dataset = datasets.load_dataset("json", data_files=harmful_dataset_path)[
        "train"
    ]
    negative_dataset = datasets.load_dataset("json", data_files=harmless_dataset_path)[
        "train"
    ]
    positive_dataset_key = "instruction"
    negative_dataset_key = "instruction"
    model = transformer_lens.HookedTransformer.from_pretrained(
        args.model, device=args.device, dtype=args.dtype
    )
    save_path = save_path.format(
        model_name=args.model.split("/")[-1],
        method="difference-in-means",
        concept_category="safety",
        layer=args.layer_num,
    )
    get_concept_vectors(
        model=model,
        positive_dataset=positive_dataset,
        negative_dataset=negative_dataset,
        layer=args.layer,
        device=args.device,
        positive_dataset_key=positive_dataset_key,
        negative_dataset_key=negative_dataset_key,
        save_path=save_path,
        methods=args.methods,
        max_dataset_size=args.max_dataset_size,
    )


def get_concept_vectors(
    model: transformer_lens.HookedTransformer,
    positive_dataset: datasets.Dataset,
    negative_dataset: datasets.Dataset,
    layer: int,
    device: str,
    positive_dataset_key: str,
    negative_dataset_key: str,
    methods: str,
    save_path: str,
    max_dataset_size: int = 300,
) -> None:
    """used to get the concept vectors using the specified method

    Args:
        model (transformer_lens.HookedTransformer): the model to get the concept vectors
        positive_dataset (datasets.Dataset): the positive dataset to get the concept vectors
        negative_dataset (datasets.Dataset): the negative dataset to get the concept vectors
        layer (int): the layer to get the concept vectors
        device (str): the device to get the concept vectors
        positive_dataset_key (str): the key of the positive dataset to get the concept vectors
        negative_dataset_key (str): the key of the negative dataset to get the concept vectors
        save_path (str): the path to save the concept vectors
        methods (str): the method to get the concept vectors
        max_dataset_size (int, optional): the maximum size of the dataset to get the concept vectors. Defaults to 300.

    Returns:
        None
    """
    if methods == "difference-in-means":
        difference_in_means = DifferenceInMeans(
            model,
            positive_dataset,
            negative_dataset,
            layer=layer,
            device=device,
            positive_dataset_key=positive_dataset_key,
            negative_dataset_key=negative_dataset_key,
            max_dataset_size=max_dataset_size,
        )
        difference_in_means.get_concept_vectors(
            save_path=save_path,
            is_save=True,
        )
    else:
        raise ValueError(f"Invalid method: {methods}")


def language_en_fr_concept_vector(args: argparse.Namespace, save_path: str) -> None:
    dataset_folder = "assets/language_translation"
    en_dataset_path = os.path.join(dataset_folder, "en.jsonl")
    fr_dataset_path = os.path.join(dataset_folder, "fr.jsonl")
    en_dataset = datasets.load_dataset("json", data_files=en_dataset_path)["train"]
    fr_dataset = datasets.load_dataset("json", data_files=fr_dataset_path)["train"]
    positive_dataset_key = "text"
    negative_dataset_key = "text"
    model = transformer_lens.HookedTransformer.from_pretrained(
        args.model, device=args.device, dtype=args.dtype
    )
    save_path = save_path.format(
        model_name=args.model.split("/")[-1],
        method=args.methods,
        concept_category="language_en_fr",
        layer=args.layer_num,
    )
    get_concept_vectors(
        model=model,
        positive_dataset=en_dataset,
        negative_dataset=fr_dataset,
        layer=args.layer,
        device=args.device,
        positive_dataset_key=positive_dataset_key,
        negative_dataset_key=negative_dataset_key,
        methods=args.methods,
        save_path=save_path,
        max_dataset_size=args.max_dataset_size,
    )


def random_concept_vector(args: argparse.Namespace, save_path: str) -> None:
    """Generate a random concept vector for the model
    Args:
        args (argparse.Namespace): the arguments
        save_path (str): the path to save the concept vectors

    Returns:
        None
    """
    save_path = save_path.format(
        model_name=args.model.split("/")[-1],
        method="random",
        concept_category="random",
        layer=args.layer_num,
    )
    if args.model == "google/gemma-2-2b":
        model_dimension = 2304
    elif args.model == "Qwen/Qwen3-1.7B":
        model_dimension = 2048
    elif args.model == "EleutherAI/pythia-70m":
        model_dimension = 512
    else:
        raise ValueError(f"Invalid model: {args.model}")

    save_path_folder = os.path.dirname(save_path)
    logger.info(f"Save path: {save_path}")
    os.makedirs(save_path_folder, exist_ok=True)
    random_concept_vector = torch.randn(args.layer_num, model_dimension)
    random_concept_vector = torch.nn.functional.normalize(random_concept_vector, dim=1)
    torch.save(random_concept_vector, save_path)
    logger.info(f"Saved random concept vector to {save_path}")
    logger.info(f"Random concept vector shape: {random_concept_vector.shape}")
    logger.info(f"Random concept vector: {random_concept_vector.norm(dim=1)}")
    return random_concept_vector


if __name__ == "__main__":
    save_path = "weights/concept_vectors/{model_name}/{method}/{concept_category}_layer{layer}.pt"

    concept_vector(save_path)
