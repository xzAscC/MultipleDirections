import argparse
import torch
import os
import datasets
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.backends.cuda import sdp_kernel  # TODO: replace


def config():
    """used to configure the arguments
    Returns:
        args: the arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="EleutherAI/pythia-70m",
        choices=["google/gemma-2-2b-it", "Qwen/Qwen3-1.7B", "EleutherAI/pythia-70m"],
    )
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="assets/harmbench/harmful_data.jsonl",
        choices=[
            "assets/paired_contexts/en-fr.jsonl",
            "assets/harmbench/harmful_data.jsonl",
            "assets/harmbench/harmless_data.jsonl",
        ],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument(
        "--concept_vector_path",
        type=str,
        default="weights/concept_vectors/pythia-70m_Layer-1_difference-in-means_safety.pt",
        choices=[
            "weights/concept_vectors/pythia-70m_Layer4_difference-in-means_en-fr.pt",
            "weights/concept_vectors/random_concept_vector_pythia-70m.pt",
            "weights/concept_vectors/pythia-70m_Layer-1_difference-in-means_safety.pt",
            "weights/concept_vectors/pythia-70m_Layer2_difference-in-means_en-fr.pt",
            "weights/concept_vectors/gemma-2-2b-it_Layer16_difference-in-means_en-fr.pt"
            "weights/concept_vectors/random_concept_vector_gemma-2-2b-it.pt",
        ],
    )
    parser.add_argument("--concept_vector_alpha", type=float, default=1e-2)
    parser.add_argument("--random_concept_vector", action="store_true")
    parser.add_argument("--alpha_factor", type=int, default=1000)
    parser.add_argument("--max_dataset_size", type=int, default=50)
    parser.add_argument(
        "--concept_category",
        type=str,
        default="safety",
        choices=["safety", "language_translation"],
    )
    args = parser.parse_args()
    return args


# Resolve target transformer layer module across architectures
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


def linearity():
    """used to measure the linearity of the model"""
    args = config()

    # add logger
    logger.add("logs/linearity.log")
    logger.info("Starting linearity measurement...")
    logger.info(f"Loading model: {args.model}")

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=args.device,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
    )

    # load dataset
    logger.info(f"Loading dataset: {args.dataset_path}")
    dataset = datasets.load_dataset("json", data_files=args.dataset_path)["train"]
    logger.info(f"Loaded {len(dataset)} examples")
    logger.info(f"Loading concept vector: {args.concept_vector_path}")
    concept_vectors = torch.load(args.concept_vector_path)
    logger.info(f"Concept vector shape: {concept_vectors.shape}")

    # measure the linearity of the model
    if args.concept_category == "safety":
        dataset_key = "instruction"
    elif args.concept_category == "language_translation":
        # TODO
        dataset_key = "context"
    else:
        raise ValueError(f"Invalid concept category: {args.concept_category}")
    layers = model.config.num_hidden_layers
    logger.info(f"Measuring linearity for {layers} layers")
    for layer_idx in range(layers):
        logger.info(f"Measuring linearity for layer {layer_idx}")
        idy = 0
        if layer_idx < 3:
            concept_vector = concept_vectors[layer_idx + 3, :]
        else:
            concept_vector = concept_vectors[layer_idx - 3, :]
        progress_bar = tqdm(total=args.max_dataset_size, desc="Measuring linearity")
        each_token_lss = []
        for example in dataset:
            context = example[dataset_key]
            if idy > args.max_dataset_size:
                break
            idy += 1
            tokens = tokenizer(context, return_tensors="pt", truncation=True).to(
                args.device
            )
            input_ids = tokens.input_ids

            layers_container = _get_layers_container(model)
            target_layer_module = layers_container[layer_idx]

            # 1) Capture the actual activation act_base at the target layer (participates in computation graph)
            captured = {"act": None}

            def _capture_hook(module, inputs, output):
                out = output[0] if isinstance(output, tuple) else output
                captured["act"] = out
                return output

            cap_handle = target_layer_module.register_forward_hook(_capture_hook)
            with (
                torch.enable_grad(),
                sdp_kernel(
                    enable_flash=False, enable_mem_efficient=False, enable_math=True
                ),
            ):
                base_outputs = model(input_ids, output_hidden_states=True)
                logits_without_hook = base_outputs.logits
            cap_handle.remove()
            act_base = captured["act"]

            seq_len = (
                int(tokens.attention_mask[0].sum().item())
                if "attention_mask" in tokens
                else act_base.shape[1]
            )

            lss_middle_states = torch.zeros(
                2,
                seq_len,
                model.config.vocab_size,
                device=torch.device(args.device),
                dtype=getattr(torch, args.dtype),
            )  # [0, :] is the base state, [1, :] is the modified state

            for i in range(args.alpha_factor):
                concept_vector_alpha = args.concept_vector_alpha * (i + 1)

                # Add hook to modify activations at the target layer
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
                with (
                    torch.enable_grad(),
                    sdp_kernel(
                        enable_flash=False, enable_mem_efficient=False, enable_math=True
                    ),
                ):
                    outputs_with_hook = model(input_ids, output_hidden_states=True)

                lss_middle_states = compute_line_shape_score_middle_states(
                    outputs_with_hook.logits[0, :seq_len, :],
                    logits_without_hook[0, :seq_len, :],
                    lss_middle_states,
                    i,
                )
                hook_handle.remove()

            lss = compute_line_shape_score(
                lss_middle_states[1, :, :],
                logits_without_hook[0, :seq_len, :],
                args.alpha_factor,
            )
            each_token_lss.extend(lss)
            progress_bar.update(1)
        progress_bar.close()
        # After processing all contexts/examples: per-alpha mean/std over per-token increases
        mean_lss = float(np.mean(each_token_lss))
        std_lss = float(np.std(each_token_lss))
        logger.info(f"Mean LSS: {mean_lss}")
        logger.info(f"Std LSS: {std_lss}")
        torch.save(each_token_lss, f"weights/linearity/lss/linearity_lss_{args.concept_vector_path.split('/')[-1].replace('.pt', '')}_Layer{layer_idx}_random.pt")

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


def compute_jvp(
    hidden_states, concept_vector, target_layer_module, model, input_ids, act_base
) -> torch.Tensor:
    """
    Compute the Jacobian-Vector Product (JVP) for a given hidden state and concept vector.

    Args:
        hidden_states: The hidden states to compute the JVP for.
        concept_vector: The concept vector to compute the JVP for.
        target_layer_module: The target layer module to compute the JVP for.
        model: The model to compute the JVP for.
        input_ids: The input ids to compute the JVP for.
        act_base: The base activation to compute the JVP for.

    Returns:
        The JVP for the given hidden state and concept vector.
    """
    base_jvps = []
    for idx in range(hidden_states.shape[1]):

        def f(act, _idx=idx):
            def _replace_hook(module, inputs, output):
                return (act,) + output[1:] if isinstance(output, tuple) else act

            h = target_layer_module.register_forward_hook(_replace_hook)
            with sdp_kernel(
                enable_flash=False, enable_mem_efficient=False, enable_math=True
            ):
                out = model(
                    input_ids, output_hidden_states=False, use_cache=False
                ).logits[0, _idx, :]
            h.remove()
            return out

        delta_act = torch.zeros_like(act_base)
        delta_act[0, idx, :] = concept_vector.to(
            device=act_base.device, dtype=act_base.dtype
        )
        _, jvp_i = torch.autograd.functional.jvp(
            f, (act_base,), (delta_act,), create_graph=False
        )
        base_jvps.append(jvp_i.detach())

    base_jvp = torch.stack(base_jvps, dim=0)  # [seq_len, vocab]
    return base_jvp



def linearity_token():
    args = config()

    # add logger
    logger.add("logs/linearity.log")
    logger.info("Starting linearity measurement...")
    logger.info(f"Loading model: {args.model}")

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=args.device,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
    )

    # load dataset
    logger.info(f"Loading dataset: {args.dataset_path}")
    dataset = datasets.load_dataset("json", data_files=args.dataset_path)["train"]
    logger.info(f"Loaded {len(dataset)} examples")
    logger.info(f"Loading concept vector: {args.concept_vector_path}")
    concept_vectors = torch.load(args.concept_vector_path)
    logger.info(f"Concept vector shape: {concept_vectors.shape}")

    # measure the linearity of the model
    if args.concept_category == "safety":
        dataset_key = "instruction"
    elif args.concept_category == "language_translation":
        # TODO
        dataset_key = "context"
    else:
        raise ValueError(f"Invalid concept category: {args.concept_category}")
    layers = model.config.num_hidden_layers
    logger.info(f"Measuring linearity for {layers} layers")
    concept_vector = concept_vectors[2, :]
    progress_bar = tqdm(total=args.max_dataset_size, desc="Measuring linearity")
    each_token_lss = []
    idy = 0
    diff_list = []

    for example in dataset:
        context = example[dataset_key]
        if idy > args.max_dataset_size:
            break
        idy += 1
        tokens = tokenizer(context, return_tensors="pt", truncation=True).to(
            args.device
        )
        input_ids = tokens.input_ids

        layers_container = _get_layers_container(model)
        target_layer_module = layers_container[2]

        # 1) Capture the actual activation act_base at the target layer (participates in computation graph)
        captured = {"act": None}

        def _capture_hook(module, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            captured["act"] = out
            return output

        cap_handle = target_layer_module.register_forward_hook(_capture_hook)
        with (
            torch.enable_grad(),
            sdp_kernel(
                enable_flash=False, enable_mem_efficient=False, enable_math=True
            ),
        ):
            base_outputs = model(input_ids, output_hidden_states=True)
            logits_without_hook = base_outputs.logits
        cap_handle.remove()
        act_base = captured["act"]

        seq_len = (
            int(tokens.attention_mask[0].sum().item())
            if "attention_mask" in tokens
            else act_base.shape[1]
        )

        lss_middle_states = torch.zeros(
            2,
            seq_len,
            model.config.vocab_size,
            device=torch.device(args.device),
            dtype=getattr(torch, args.dtype),
        )  # [0, :] is the base state, [1, :] is the modified state

        for i in range(args.alpha_factor):
            concept_vector_alpha = args.concept_vector_alpha * (i + 1)

            # Add hook to modify activations at the target layer
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
            with (
                torch.enable_grad(),
                sdp_kernel(
                    enable_flash=False, enable_mem_efficient=False, enable_math=True
                ),
            ):
                outputs_with_hook = model(input_ids, output_hidden_states=True)

            lss_middle_states = compute_line_shape_score_middle_states(
                outputs_with_hook.logits[0, :seq_len, :],
                logits_without_hook[0, :seq_len, :],
                lss_middle_states,
                i,
            )
            hook_handle.remove()

        lss = compute_line_shape_score(
            lss_middle_states[1, :, :],
            logits_without_hook[0, :seq_len, :],
            args.alpha_factor,
        )
        each_token_lss.extend(lss)
        if max(lss) > 2:
            indices = [i for i, val in enumerate(lss) if val > 2]
            one_diff_list = []
            for i in range(args.alpha_factor):
                concept_vector_alpha = args.concept_vector_alpha * (i + 1)

                # Add hook to modify activations at the target layer
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
                with (
                    torch.enable_grad(),
                    sdp_kernel(
                        enable_flash=False, enable_mem_efficient=False, enable_math=True
                    ),
                ):
                    outputs_with_hook = model(input_ids, output_hidden_states=True)
                    diff = outputs_with_hook.logits[0, indices[0], :] - logits_without_hook[0, indices[0], :]
                    diff_norm = torch.norm(diff, p=2)
                    one_diff_list.append(diff_norm.item())
                hook_handle.remove()
            diff_list.append(one_diff_list)
        progress_bar.update(1)
    progress_bar.close()
    # After processing all contexts/examples: per-alpha mean/std over per-token increases
    torch.save(diff_list, f"weights/linearity/lss/linearity_diff_larger_than_2_pythia-70m_Layer2.pt")



if __name__ == "__main__":
    linearity_token()
