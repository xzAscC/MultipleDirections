import argparse
import torch
import os
import transformer_lens
import datasets
import gc
import transformers
import numpy as np
from loguru import logger
from tqdm import tqdm
from utils import set_seed, seed_from_name

# TODO: how to handle the last layer?

MODEL_LAYERS = {
    "google/gemma-3-270m-it": 18,
    "google/gemma-3-4b-it": 34,  # TODO: replace with Gemma2, and add Mistral3
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
    "random": ["assets/harmbench", "instruction"],
    "random1": ["assets/harmbench", "instruction"],
    "random2": ["assets/language_translation", "text"],
    "random3": ["assets/language_translation", "text"],
}


def config() -> argparse.Namespace:
    """
    Config for linearity
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.7B",
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
    parser.add_argument(
        "--remove_concept_vector",
        action="store_true",
        help="whether to remove the concept vector",
    )
    parser.add_argument(
        "--linearity_metric",
        type=str,
        nargs="+",
        default=["lss", "lsr"],
        choices=["lss", "lsr", "norm"],
        help="the metrics to use to calculate the linearity",
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
    logger.info(f"args.linearity_metric: {args.linearity_metric}")

    for concept_category_name, concept_category_metadata in CONCEPT_CATEGORIES.items():
        save_path = f"assets/linearity/{model_name}/{concept_category_name}.pt"
        if args.concept_vector_pretrained:
            concept_vectors = torch.load(save_path)
            if concept_category_name in ["language_en_fr", "random2", "random3"]:
                dataset_path = os.path.join(concept_category_metadata[0], "en.jsonl")
                dataset = datasets.load_dataset(
                    "json", data_files=dataset_path, split="train"
                )
                dataset_key = concept_category_metadata[1]
            elif concept_category_name in ["random1", "safety", "random"]:
                dataset_path = os.path.join(
                    concept_category_metadata[0], "harmful_data.jsonl"
                )
                dataset = datasets.load_dataset(
                    "json", data_files=dataset_path, split="train"
                )
                dataset_key = concept_category_metadata[1]
            else:
                raise ValueError(f"Invalid concept category name: {concept_category_name}")
        else:
            model = transformer_lens.HookedTransformer.from_pretrained(
                args.model, device=args.device, dtype=dtype, trust_remote_code=True
            )
            concept_vectors, dataset, dataset_key = obtain_concept_vector(
                concept_category_name,
                concept_category_metadata,
                model,
                model_name,
                max_layers,
                device=args.device,
                max_dataset_size=args.concept_vector_dataset_size,
                save_path=save_path,
            )
            del model
            torch.cuda.empty_cache()
            gc.collect()
        dataset = dataset.shuffle().select(range(args.linearity_dataset_size))
        input_prompts = dataset[: args.linearity_dataset_size][dataset_key]
        logger.info(f"Concept vectors shape: {concept_vectors.shape}")
        for metric in args.linearity_metric:
            logger.info(f"Measuring {metric} for {concept_category_name}")
            model = transformers.AutoModelForCausalLM.from_pretrained(
                args.model,
                device_map=args.device,
                dtype=dtype,
                trust_remote_code=True,
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                args.model,
                use_fast=True,
                dtype=dtype,
            )

            def _get_layers_container(hf_model):
                """used to get the layers container of the model
                Args:
                    hf_model (transformers.PreTrainedModel): the model to get the layers container
                Returns:
                    layers (list): the layers container of the model
                """
                # Common containers across HF architectures
                candidates = [
                    (hf_model, "gpt_neox", "layers"),
                    (hf_model, "model", "layers"),
                    (hf_model, "transformer", "layers"),
                    (hf_model, "transformer", "h"),
                ]
                for root_obj, root_attr, layers_attr in candidates:
                    root = getattr(root_obj, root_attr, None)
                    if root is None:
                        continue
                    layers = getattr(root, layers_attr, None)
                    if layers is not None:
                        return layers
                raise AttributeError(
                    "Unable to locate transformer layers container on model"
                )

            for layer_idx in tqdm(range(max_layers), desc="Measuring linearity"):
                concept_vector = concept_vectors[layer_idx, :]
                logger.info(f"Concept vector: {concept_vector.shape}")
                # Set padding token if not already set
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                layers_container = _get_layers_container(model)
                target_layer_module = layers_container[layer_idx]
                input_ids = tokenizer(
                    input_prompts, return_tensors="pt", truncation=True, padding=True
                ).to(args.device)
                input_ids = input_ids.input_ids
                output = model(input_ids, output_hidden_states=True)
                last_layer_hidden_states = output.hidden_states[-1]
                dim_middle_states = last_layer_hidden_states.shape[-1]
                seq_len = last_layer_hidden_states.shape[1]
                if metric == "lss":
                    lss_middle_states = torch.zeros(
                        2,
                        seq_len * args.linearity_dataset_size,
                        dim_middle_states,
                        device=torch.device(args.device),
                        dtype=torch.float32,
                    )  # [0, :] is the base state, [1, :] is the modified state
                elif metric == "lsr":
                    last_logits_with_hook = (
                        last_layer_hidden_states.reshape(-1, dim_middle_states)
                        .detach()
                        .to(torch.float32)
                    )
                    seq_diff_norm = torch.zeros(
                        seq_len * args.linearity_dataset_size,
                        device=torch.device(args.device),
                        dtype=torch.float32,
                    )
                elif metric == "norm":
                    mean_norm = torch.zeros(
                        args.alpha_factor,
                        device=torch.device(args.device),
                        dtype=torch.float32,
                    )
                    std_norm = torch.zeros(
                        args.alpha_factor,
                        device=torch.device(args.device),
                        dtype=torch.float32,
                    )
                else:
                    raise ValueError(f"Invalid metric: {metric}")

                for i in tqdm(range(args.alpha_factor), desc="Measuring linearity"):
                    concept_vector_alpha = args.concept_vector_alpha * (i + 1)

                    def _forward_hook(module, inputs, output):
                        # Ensure concept vector matches device/dtype
                        if isinstance(output, tuple):
                            hidden = output[0]
                            vec = concept_vector.to(
                                device=hidden.device, dtype=hidden.dtype
                            )
                            hidden = hidden + concept_vector_alpha * vec
                            return (hidden,) + output[1:]
                        else:
                            vec = concept_vector.to(
                                device=output.device, dtype=output.dtype
                            )
                            return output + concept_vector_alpha * vec

                    hook_handle = target_layer_module.register_forward_hook(
                        _forward_hook
                    )
                    output = model(input_ids, output_hidden_states=True)
                    steered_last_layer_hidden_states = output.hidden_states[-1]
                    hook_handle.remove()
                    if metric == "lss":
                        lss_middle_states = compute_line_shape_score_middle_states(
                            steered_last_layer_hidden_states.reshape(
                                -1, dim_middle_states
                            )
                            .detach()
                            .to(torch.float32),
                            last_layer_hidden_states.reshape(-1, dim_middle_states)
                            .detach()
                            .to(torch.float32),
                            lss_middle_states,
                            i,
                            args.concept_vector_alpha,
                            concept_vector,
                            remove_concept_vector=args.remove_concept_vector,
                        )
                    elif metric == "lsr":
                        last_logits_with_hook, seq_diff_norm = (
                            compute_lsr_middle_states(
                                steered_last_layer_hidden_states.reshape(
                                    -1, dim_middle_states
                                )
                                .detach()
                                .to(torch.float32),
                                last_logits_with_hook,
                                last_layer_hidden_states.reshape(-1, dim_middle_states)
                                .detach()
                                .to(torch.float32),
                                seq_diff_norm,
                                i,
                                args.alpha_factor,
                                args.concept_vector_alpha,
                                concept_vector,
                                remove_concept_vector=args.remove_concept_vector,
                            )
                        )
                    elif metric == "norm":
                        norm_diff = torch.norm(steered_last_layer_hidden_states.reshape(-1, dim_middle_states) - last_layer_hidden_states.reshape(-1, dim_middle_states), p=2, dim=-1) / torch.norm(last_layer_hidden_states.reshape(-1, dim_middle_states), p=2, dim=-1)
                        norm_diff = norm_diff.detach().cpu().numpy()
                        mean_norm[i] = float(np.mean(norm_diff))
                        std_norm[i] = float(np.std(norm_diff))
                    else:
                        raise ValueError(f"Invalid metric: {metric}")
                if metric == "lss":
                    lss = compute_line_shape_score(
                        lss_middle_states[1, :, :],
                        last_layer_hidden_states.reshape(-1, dim_middle_states)
                        .detach()
                        .to(torch.float32),
                        args.alpha_factor,
                    )
                    mean_lss = float(np.mean(lss))
                    std_lss = float(np.std(lss))
                    logger.info(f"Mean LSS: {mean_lss}")
                    logger.info(f"Std LSS: {std_lss}")
                    os.makedirs(f"weights/linearity/lss", exist_ok=True)
                    if args.remove_concept_vector:
                        torch.save(
                            lss,
                            f"assets/linearity/{model_name}/lss_{concept_category_name}_layer{layer_idx}_w_remove.pt",
                        )
                    else:
                        torch.save(
                            lss,
                            f"assets/linearity/{model_name}/lss_{concept_category_name}_layer{layer_idx}_wo_remove.pt",
                        )
                elif metric == "lsr":
                    seq_diff_norm_cpu = seq_diff_norm.cpu().numpy()
                    mean_lsr = float(np.mean(seq_diff_norm_cpu))
                    std_lsr = float(np.std(seq_diff_norm_cpu))
                    logger.info(f"Mean LSR: {mean_lsr}")
                    logger.info(f"Std LSR: {std_lsr}")
                    if args.remove_concept_vector:
                        torch.save(
                            seq_diff_norm,
                            f"assets/linearity/{model_name}/lsr_{concept_category_name}_layer{layer_idx}_w_remove.pt",
                        )
                    else:
                        torch.save(
                            seq_diff_norm,
                            f"assets/linearity/{model_name}/lsr_{concept_category_name}_layer{layer_idx}_wo_remove.pt",
                        )
                elif metric == "norm":
                    if args.remove_concept_vector:
                        torch.save(
                            mean_norm,
                            f"assets/linearity/{model_name}/mean_norm_{concept_category_name}_layer{layer_idx}_w_remove.pt",
                        )
                        torch.save(
                            std_norm,
                            f"assets/linearity/{model_name}/std_norm_{concept_category_name}_layer{layer_idx}_w_remove.pt",
                        )
                    else:
                        torch.save(
                            mean_norm,
                            f"assets/linearity/{model_name}/mean_norm_{concept_category_name}_layer{layer_idx}_wo_remove.pt",
                        )
                        torch.save(
                            std_norm,
                            f"assets/linearity/{model_name}/std_norm_{concept_category_name}_layer{layer_idx}_wo_remove.pt",
                        )
                else:
                    raise ValueError(f"Invalid metric: {metric}")


def compute_line_shape_score(
    lss_final_states: torch.Tensor, initial_state: torch.Tensor, iteration_times: int
):
    """
    Compute Line-Shape Score.

    Args:
        lss_final_states: The final states.
        initial_state: The initial state.
        iteration_times: The iteration times.

    Returns:
        The Line-Shape Score.
    """
    diff = lss_final_states - initial_state
    diff_norm = diff.norm(p=2, dim=-1)
    diff_norm = torch.clamp(diff_norm, min=1e-8)
    return (iteration_times / diff_norm).detach().to(torch.float32).tolist()


def compute_lsr_middle_states(
    logits_with_hook: torch.Tensor,
    last_logits_with_hook: torch.Tensor,
    logits_without_hook: torch.Tensor,
    seq_diff_norm: torch.Tensor,
    i: int,
    alpha_factor: int,
    concept_vector_alpha: float,
    concept_vector: torch.Tensor,
    remove_concept_vector: bool = False,
):
    """
    Compute LSR (per token, streaming over layers, no full storage).
    Args:
        logits_with_hook: The logits with hook.
        last_logits_with_hook: The last logits with hook.
        logits_without_hook: The logits without hook.
        seq_diff_norm: The sequence difference norm.
        i: The index of the alpha.
        alpha_factor: The alpha factor.
        concept_vector_alpha: The alpha of the concept vector.
        concept_vector: The concept vector.
        remove_concept_vector: Whether to remove the concept vector.
    Returns:
        logits_with_hook: The logits with hook.
        seq_diff_norm: The sequence difference norm.
    """
    if remove_concept_vector:
        updated_concept_vector = concept_vector_alpha * concept_vector.to(
            device=logits_with_hook.device, dtype=logits_with_hook.dtype
        )
    else:
        updated_concept_vector = 0
    if i == alpha_factor - 1:
        diff = logits_with_hook - last_logits_with_hook - updated_concept_vector
        diff_norm = torch.norm(diff, p=2, dim=-1)
        diff_norm = torch.clamp(diff_norm, min=1e-8)
        seq_diff_norm += diff_norm
        overall_diff_norm = (
            logits_with_hook - logits_without_hook - updated_concept_vector
        )
        overall_diff_norm = torch.norm(overall_diff_norm, p=2, dim=-1)
        overall_diff_norm = torch.clamp(overall_diff_norm, min=1e-8)
        return logits_with_hook, seq_diff_norm / overall_diff_norm
    else:
        diff = logits_with_hook - last_logits_with_hook - updated_concept_vector
        diff_norm = torch.norm(diff, p=2, dim=-1)
        diff_norm = torch.clamp(diff_norm, min=1e-8)
        seq_diff_norm += diff_norm
        return logits_with_hook, seq_diff_norm


def compute_line_shape_score_middle_states(
    logits_with_hook: torch.Tensor,
    logits_without_hook: torch.Tensor,
    lss_middle_states: torch.Tensor,
    i: int,
    concept_vector_alpha: float,
    concept_vector: torch.Tensor,
    remove_concept_vector: bool = False,
):
    """
    Compute Line-Shape Score (per token, streaming over layers, no full storage).

    Args:
        logits_with_hook: The logits with hook.
        logits_without_hook: The logits without hook.
        lss_middle_states: The middle states.
        i: The index of the alpha.
        concept_vector_alpha: The alpha of the concept vector.
        concept_vector: The concept vector.
        remove_concept_vector: Whether to remove the concept vector.
    Returns:
        lss_middle_states: The middle states.
    """
    if remove_concept_vector:
        updated_concept_vector = concept_vector_alpha * concept_vector
    else:
        updated_concept_vector = 0
    if i == 0:
        lss_middle_states[0, :, :] = logits_without_hook
        diff = logits_with_hook - logits_without_hook - updated_concept_vector
        diff_norm = torch.norm(diff, p=2, dim=-1, keepdim=True)
        diff_norm = torch.clamp(diff_norm, min=1e-8)
        normalized_diff = diff / diff_norm
        lss_middle_states[1, :, :] = (
            normalized_diff + lss_middle_states[0, :, :] - updated_concept_vector
        )
        lss_middle_states[0, :, :] = logits_without_hook
    else:
        diff = logits_with_hook - lss_middle_states[0, :, :] - updated_concept_vector
        diff_norm = torch.norm(diff, p=2, dim=-1, keepdim=True)
        diff_norm = torch.clamp(diff_norm, min=1e-8)
        normalized_diff = diff / diff_norm
        lss_middle_states[1, :, :] = normalized_diff + lss_middle_states[1, :, :]
        lss_middle_states[0, :, :] = logits_with_hook
    return lss_middle_states


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
        datasets.Dataset: the positive dataset
        str: the key of the dataset
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
    elif concept_category_name in ["random", "random1", "random2", "random3"]:
        seed = seed_from_name(concept_category_name)
        set_seed(seed)
        if concept_category_name in ["random", "random1"]:
            positive_dataset_path = os.path.join(dataset_path, "harmful_data.jsonl")
            positive_dataset = datasets.load_dataset(
                "json", data_files=positive_dataset_path, split="train"
            )
        elif concept_category_name in ["random2", "random3"]:
            positive_dataset_path = os.path.join(dataset_path, "en.jsonl")
            positive_dataset = datasets.load_dataset(
                "json", data_files=positive_dataset_path, split="train"
            )
        hidden_state_dim = model.cfg.d_model
        concept_vector = torch.randn(max_layers, hidden_state_dim, device=device)
        concept_vector = torch.nn.functional.normalize(concept_vector, dim=1)
        torch.save(concept_vector, save_path)
        logger.info(f"Concept vector shape: {concept_vector.shape}")
        logger.info(f"Concept vector: {concept_vector.norm(dim=1)}")
        torch.cuda.empty_cache()
        gc.collect()
        return concept_vector, positive_dataset, dataset_key
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

    return concept_vector, positive_dataset, dataset_key


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
) -> torch.Tensor:
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
        torch.Tensor: the concept vectors
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
        concept_vector = difference_in_means.get_concept_vectors(
            save_path=save_path,
            is_save=True,
        )
    else:
        raise ValueError(f"Invalid method: {methods}")
    return concept_vector


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

    def get_concept_vectors(
        self, save_path: str, is_save: bool = False
    ) -> torch.Tensor:
        """used to calculate the concept vectors using difference-in-means

        Args:
            save_path (str): the path to save the concept vectors
            is_save (bool, optional): whether to save the concept vectors. Defaults to False.

        Returns:
            torch.Tensor: the concept vectors
        """
        model_dimension = self.model.cfg.d_model
        layer_length = len(self.layers)
        logger.info(f"layer_length: {layer_length}")
        positive_concept_vector = torch.zeros(
            layer_length, model_dimension, device=self.device, dtype=self.dtype
        )
        negative_concept_vector = torch.zeros(
            layer_length, model_dimension, device=self.device, dtype=self.dtype
        )
        positive_token_length = 0
        negative_token_length = 0
        positive_dataset_size = (
            len(self.positive_dataset)
            if self.max_dataset_size > len(self.positive_dataset)
            else self.max_dataset_size
        )
        for i, example in tqdm(
            enumerate(self.positive_dataset), total=positive_dataset_size
        ):
            if i >= self.max_dataset_size:
                break
            torch.cuda.empty_cache()
            gc.collect()
            context = example[self.positive_dataset_key]
            _, positive_cache = self.model.run_with_cache(context)
            for layer in self.layers:
                positive_hidden_state = positive_cache[
                    f"blocks.{layer}.hook_resid_post"
                ].reshape(-1, model_dimension)
                positive_concept_vector[layer] += positive_hidden_state.sum(dim=0)
                if layer == 0:
                    current_token_length = positive_hidden_state.shape[0]
                    positive_token_length += current_token_length
        negative_dataset_size = (
            len(self.negative_dataset)
            if self.max_dataset_size > len(self.negative_dataset)
            else self.max_dataset_size
        )
        for i, example in tqdm(
            enumerate(self.negative_dataset), total=negative_dataset_size
        ):
            if i >= self.max_dataset_size:
                break
            torch.cuda.empty_cache()
            gc.collect()
            context = example[self.negative_dataset_key]
            _, negative_cache = self.model.run_with_cache(
                context, stop_at_layer=layer + 1
            )
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
