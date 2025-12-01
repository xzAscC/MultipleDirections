import argparse
import torch
import os
import transformer_lens
import datasets
import gc
from loguru import logger
from tqdm import tqdm
from utils import set_seed

MODEL_LAYERS = {
    "google/gemma-3-270m-it": 18,
    "google/gemma-3-4b-it": 34,  # TODO: this is a multimodal model
    "google/gemma-3-12b-it": 48,
    "Qwen/Qwen3-1.7B": 28,
    "Qwen/Qwen3-8b": 36,
    "Qwen/Qwen3-14B": 40,
    "EleutherAI/pythia-70m": 6,
    "EleutherAI/pythia-410m": 24,
    "EleutherAI/pythia-160m": 12,
}

CONCEPT_CATEGORIES = {
        "safety": ["assets/harmbench", "instruction"],
        "language_en_fr": ["assets/language_translation", "text"],
        "random": ["assets/random_contexts", None],
    }



def config() -> argparse.Namespace:
    """
    Config for linearity
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="EleutherAI/pythia-70m",
        choices=MODEL_LAYERS.keys(),
        help="the model to calculate the linearity",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="the device to use",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["bfloat16", "float16", "float32"],
        help="the dtype to use",
    )
    parser.add_argument("--seed", type=int, default=42, help="the seed to use")
    parser.add_argument(
        "--concept_vector_dataset_size",
        type=int,
        default=300,
        help="the maximum size of the dataset to calculate the concept vector",
    )
    parser.add_argument(
        "--concept_vector_pretrained",
        action="store_true",
        help="whether to use the pretrained concept vector",
    )
    parser.add_argument(
        "--linearity_dataset_size",
        type=int,
        default=30,
        help="the maximum size of the dataset to calculate the linearity",
    )
    parser.add_argument(
        "--linearity_metric",
        type=str,
        default="lss",
        choices=["lss", "lsr", "norm"],
        help="the metric to use to calculate the linearity",
    )
    parser.add_argument(
        "--concept_vector_alpha",
        type=float,
        default=1,
        help="the beginning alpha to use to calculate the concept vector",
    )
    parser.add_argument(
        "--alpha_factor",
        type=int,
        default=1000,
        help="the factor to multiply the alpha by",
    )
    return parser.parse_args()


def linearity() -> None:
    """
    Linearity of llm for random concept vectors and steering concept vectors
    We measure the lss, lsr and norm score under original hidden states difference and the difference after removing the concept vector from the hidden states
    """
    args = config()
    set_seed(args.seed)
    # add logger
    model_name = args.model.split("/")[-1]

    os.makedirs("logs", exist_ok=True)
    os.makedirs(f"assets/linearity/{model_name}", exist_ok=True)
    logger.add("logs/linearity.log")
    logger.info("Starting linearity measurement...")
    logger.info(f"Loading model: {args.model}")
    dtype = getattr(torch, args.dtype)
    max_layers = MODEL_LAYERS[args.model]
    # load model
    model = transformer_lens.HookedTransformer.from_pretrained(
        args.model, device=args.device, dtype=dtype, trust_remote_code=True
    )

    for concept_category_name, concept_category_metadata in CONCEPT_CATEGORIES.items():
        save_path = f"assets/linearity/{model_name}/{concept_category_name}.pt"
        if args.concept_vector_pretrained:
            concept_vector = torch.load(save_path)
        else:
            concept_vector = obtain_concept_vector(
                concept_category_name,
                concept_category_metadata,
                model,
                model_name,
                max_layers,
                device=args.device,
                max_dataset_size=args.concept_vector_dataset_size,
                save_path=save_path,
            )
        


def obtain_concept_vector(
    concept_category_name: str,
    concept_category_metadata: list,
    model: transformer_lens.HookedTransformer,
    model_name: str,
    max_layers: int,
    device: str,
    max_dataset_size: int,
    save_path: str,
    methods: str = "difference-in-means",
) -> torch.Tensor:
    
    """used to obtain the concept vector for the given concept category

    Args:
        concept_category_name (str): the name of the concept category
        concept_category_metadata (list): the metadata of the concept category, the first element is the path to the dataset and the second element is the key of the dataset
        model (transformer_lens.HookedTransformer): the model to obtain the concept vector
        model_name (str): the name of the model
        max_layers (int): the maximum number of layers
        device (str): the device to use
        max_dataset_size (int): the maximum size of the dataset to obtain the concept vector
        save_path (str): the path to save the concept vector
        methods (str): the method to use to obtain the concept vector, default is "difference-in-means"
    Returns:
        torch.Tensor: the concept vector
    """
    dataset_path = concept_category_metadata[0]
    dataset_key = concept_category_metadata[1]
    if concept_category_name == "safety":
        positive_dataset_path = os.path.join(dataset_path, "harmful_data.jsonl")
        negative_dataset_path = os.path.join(dataset_path, "harmless_data.jsonl")
        positive_dataset = datasets.load_dataset(
            "json", data_files=positive_dataset_path, split="train"
        )
        negative_dataset = datasets.load_dataset(
            "json", data_files=negative_dataset_path, split="train"
        )
    elif concept_category_name == "language_en_fr":
        positive_dataset_path = os.path.join(dataset_path, "en.jsonl")
        negative_dataset_path = os.path.join(dataset_path, "fr.jsonl")
        positive_dataset = datasets.load_dataset(
            "json", data_files=positive_dataset_path, split="train"
        )
        negative_dataset = datasets.load_dataset(
            "json", data_files=negative_dataset_path, split="train"
        )
    elif concept_category_name == "random":
        hidden_state_dim = model.cfg.d_model
        concept_vector = torch.randn(max_layers, hidden_state_dim, device=device)
        concept_vector = torch.nn.functional.normalize(concept_vector, dim=1)
        torch.save(concept_vector, save_path)
        logger.info(f"Concept vector shape: {concept_vector.shape}")
        logger.info(f"Concept vector: {concept_vector.norm(dim=1)}")
        torch.cuda.empty_cache()
        gc.collect()
        return concept_vector
    else:
        raise ValueError(f"Invalid concept category: {concept_category_name}")
    
    concept_vector = get_concept_vectors(
        model=model,
        positive_dataset=positive_dataset,
        negative_dataset=negative_dataset,
        layer=max_layers,
        device=device,
        positive_dataset_key=dataset_key,
        negative_dataset_key=dataset_key,
        methods=methods,
        save_path=save_path,
        max_dataset_size=max_dataset_size,
    )
    torch.cuda.empty_cache()
    gc.collect()
    
    return concept_vector


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
    # TODO: more methods
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
        self.layers = list(range(layer))
        self.device = device
        self.positive_dataset_key = positive_dataset_key
        self.negative_dataset_key = negative_dataset_key
        self.max_dataset_size = max_dataset_size
        self.dtype = dtype

    def get_concept_vectors(self, save_path: str, is_save: bool = False)-> torch.Tensor:
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
        positive_dataset_size = len(self.positive_dataset) if self.max_dataset_size > len(self.positive_dataset) else self.max_dataset_size
        for i, example in tqdm(enumerate(self.positive_dataset), total=positive_dataset_size):
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
        negative_dataset_size = len(self.negative_dataset) if self.max_dataset_size > len(self.negative_dataset) else self.max_dataset_size
        for i, example in tqdm(enumerate(self.negative_dataset), total=negative_dataset_size):
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

if __name__ == "__main__":
    linearity()
