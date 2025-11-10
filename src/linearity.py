import argparse
import torch
import datasets
import os
import numpy as np
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def config():
    """used to configure the arguments
    Returns:
        args: the arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-2b-it",
        choices=["google/gemma-2-2b-it", "Qwen/Qwen3-1.7B", "EleutherAI/pythia-70m"],
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="assets/language_translation/en.jsonl",
        choices=[
            "assets/harmbench/harmful_data.jsonl",
            "assets/harmbench/harmless_data.jsonl",
            "assets/language_translation/en.jsonl",
            "assets/language_translation/fr.jsonl",
        ],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument(
        "--concept_vector_path",
        type=str,
        default="weights/concept_vectors/pythia-70m/difference-in-means/language_en_fr_layer6.pt",
        choices=[
            "weights/concept_vectors/pythia-70m/difference-in-means/safety_layer6.pt",
            "weights/concept_vectors/pythia-70m/difference-in-means/language_en_fr_layer6.pt",
            "weights/concept_vectors/gemma-2-2b/difference-in-means/safety_layer26.pt",
            "weights/concept_vectors/gemma-2-2b/random/random_layer26.pt",
            "weights/concept_vectors/pythia-70m/random/random_layer6.pt",
        ],
    )
    parser.add_argument("--concept_vector_alpha", type=float, default=1e-2)
    parser.add_argument("--random_concept_vector", action="store_true")
    parser.add_argument("--alpha_factor", type=int, default=1000)
    parser.add_argument("--max_dataset_nums", type=int, default=10)
    args = parser.parse_args()
    return args


def linearity():
    """
    Linearity
    """
    args = config()
    # add logger
    logger.add("logs/linearity.log")
    logger.info("Starting linearity measurement...")
    logger.info(f"Loading model: {args.model}")
    dtype = getattr(torch, args.dtype)
    # load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=args.device,
        trust_remote_code=True,
        dtype=dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        dtype=dtype,
    )

    # load dataset
    logger.info(f"Loading dataset: {args.dataset_path}")
    dataset = datasets.load_dataset("json", data_files=args.dataset_path)["train"]
    logger.info(f"Loaded {len(dataset)} examples")
    logger.info(f"Loading concept vector: {args.concept_vector_path}")
    concept_vectors = torch.load(args.concept_vector_path)
    logger.info(f"Concept vector shape: {concept_vectors.shape}")

    if os.path.dirname(args.dataset_path) == "assets/language_translation":
        dataset_key = "text"
    elif os.path.dirname(args.dataset_path) == "assets/harmbench":
        dataset_key = "instruction"
    else:
        raise ValueError(f"Invalid dataset path: {args.dataset_path}")

    layers = model.config.num_hidden_layers
    logger.info(f"Measuring linearity for {layers} layers")
    dataset_nums = args.max_dataset_nums
    input_prompts = dataset[:dataset_nums][dataset_key]

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
        raise AttributeError("Unable to locate transformer layers container on model")

    for layer_idx in tqdm(range(layers), desc="Measuring linearity"):
        concept_vector = concept_vectors[layer_idx, :]
        # Set padding token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        input_ids = tokenizer(
            input_prompts, return_tensors="pt", truncation=True, padding=True
        ).to(args.device)
        input_ids = input_ids.input_ids
        layers_container = _get_layers_container(model)
        target_layer_module = layers_container[layer_idx]

        captured = {"act": None}

        original_output = model(input_ids, output_hidden_states=True)
        logits_without_hook = original_output.logits
        seq_len = logits_without_hook.shape[1]
        lss_middle_states = torch.zeros(
            2,
            seq_len * dataset_nums,
            model.config.vocab_size,
            device=torch.device(args.device),
            dtype=torch.float32,
        )  # [0, :] is the base state, [1, :] is the modified state

        for i in range(args.alpha_factor):
            concept_vector_alpha = args.concept_vector_alpha * (i + 1)

            def _forward_hook(module, inputs, output):
                # Ensure concept vector matches device/dtype
                if isinstance(output, tuple):
                    hidden = output[0]
                    vec = concept_vector.to(device=hidden.device, dtype=hidden.dtype)
                    hidden = hidden + concept_vector_alpha * vec
                    return (hidden,) + output[1:]
                else:
                    vec = concept_vector.to(device=output.device, dtype=output.dtype)
                    return output - concept_vector_alpha * vec

            hook_handle = target_layer_module.register_forward_hook(_forward_hook)

            logits_with_hook = model(input_ids, output_hidden_states=True)
            logits_with_hook = logits_with_hook.logits
            lss_middle_states = compute_line_shape_score_middle_states(
                logits_with_hook.reshape(-1, model.config.vocab_size)
                .detach()
                .to(torch.float32),
                logits_without_hook.reshape(-1, model.config.vocab_size)
                .detach()
                .to(torch.float32),
                lss_middle_states,
                i,
            )
            hook_handle.remove()
        lss = compute_line_shape_score(
            lss_middle_states[1, :, :],
            logits_without_hook.reshape(-1, model.config.vocab_size)
            .detach()
            .to(torch.float32),
            args.alpha_factor,
        )
        mean_lss = float(np.mean(lss))
        std_lss = float(np.std(lss))
        logger.info(f"Mean LSS: {mean_lss}")
        logger.info(f"Std LSS: {std_lss}")
        torch.save(
            lss,
            f"weights/linearity/lss/{args.concept_vector_path.split('/')[-1].replace('.pt', '')}_Layer{layer_idx}.pt",
        )
        logger.info(f"Mean LSS: {mean_lss}")
        logger.info(f"Std LSS: {std_lss}")


def compute_line_shape_score(lss_final_states, initial_state, iteration_times):
    """
    Compute Line-Shape Score (per token, streaming over layers, no full storage).

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


def compute_line_shape_score_middle_states(
    logits_with_hook, logits_without_hook, lss_middle_states, i
):
    """
    Compute Line-Shape Score (per token, streaming over layers, no full storage).

    Args:
        logits_with_hook: The logits with hook.
        logits_without_hook: The logits without hook.
        lss_middle_states: The middle states.
    """
    if i == 0:
        lss_middle_states[0, :, :] = logits_without_hook
        diff = logits_with_hook - logits_without_hook
        diff_norm = torch.norm(diff, p=2, dim=-1, keepdim=True)
        diff_norm = torch.clamp(diff_norm, min=1e-8)
        normalized_diff = diff / diff_norm
        lss_middle_states[1, :, :] = normalized_diff + lss_middle_states[0, :, :]
        lss_middle_states[0, :, :] = logits_without_hook
    else:
        diff = logits_with_hook - lss_middle_states[0, :, :]
        diff_norm = torch.norm(diff, p=2, dim=-1, keepdim=True)
        diff_norm = torch.clamp(diff_norm, min=1e-8)
        normalized_diff = diff / diff_norm
        lss_middle_states[1, :, :] = normalized_diff + lss_middle_states[1, :, :]
        lss_middle_states[0, :, :] = logits_with_hook
    return lss_middle_states


if __name__ == "__main__":
    linearity()
