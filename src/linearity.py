import argparse
import torch
import os
import datasets
import transformer_lens
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm
from functorch.experimental import chunk_vmap
from torch.autograd import grad
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.backends.cuda import sdp_kernel

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
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument(
        "--dataset_path", type=str, default="assets/paired_contexts/en-fr.jsonl"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--concept_vector_path",
        type=str,
        default="weights/concept_vectors/random_concept_vector_pythia-70m.pt",
        choices=[
            "weights/concept_vectors/pythia-70m_Layer4_difference-in-means_en-fr.pt",
            "weights/concept_vectors/random_concept_vector_pythia-70m.pt",
            "weights/concept_vectors/pythia-70m_Layer2_difference-in-means_en-fr.pt",
            "weights/concept_vectors/gemma-2-2b-it_Layer16_difference-in-means_en-fr.pt"
            "weights/concept_vectors/random_concept_vector_gemma-2-2b-it.pt"
            
        ],
    )
    parser.add_argument("--concept_vector_alpha", type=float, default=1e-2)
    parser.add_argument("--random_concept_vector", action="store_true")
    parser.add_argument("--alpha_factor", type=int, default=1000)
    args = parser.parse_args()
    return args

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
    concept_vector = torch.load(args.concept_vector_path)
    logger.info(f"Concept vector shape: {concept_vector.shape}")


    # measure the linearity of the model
    only_once = True
    per_alpha_token_increases = [[] for _ in range(args.alpha_factor)]
    idy = 0
    progress_bar = tqdm(total=10, desc="Measuring linearity")
    for example in dataset:
        contexts = example["contexts0"]
        for context in contexts:
            if idy > 10:
                break
            idy += 1
            progress_bar.update(1)
            tokens = tokenizer(context, return_tensors="pt", truncation=True).to(
                args.device
            )
            input_ids = tokens.input_ids
            # outputs = model(input_ids, output_hidden_states=True)
            # # jacobian_without_hook = jacobian(
            # #     outputs.hidden_states[-1], outputs.hidden_states[args.layer], 0, 10
            # # )
            # jacobian_without_hook = jacobian(
            #     outputs.logits, outputs.hidden_states[args.layer], 0, 1
            # ).to(torch.bfloat16)

            # Resolve target transformer layer module across architectures
            def _get_layers_container(hf_model):
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

            layers_container = _get_layers_container(model)
            target_layer_module = layers_container[args.layer - 1]

            # 1) 捕获目标层实际激活 act_base（参与计算图）
            captured = {"act": None}
            def _capture_hook(module, inputs, output):
                out = output[0] if isinstance(output, tuple) else output
                captured["act"] = out
                return output

            cap_handle = target_layer_module.register_forward_hook(_capture_hook)
            with torch.enable_grad(), sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                base_outputs = model(input_ids, output_hidden_states=True)
                logits_without_hook = base_outputs.logits  # 供后续对比
            cap_handle.remove()
            act_base = captured["act"]  # 形状: [1, seq_len, d_model]（或等价）

            # 2) 对每个 token 位置计算一次 JVP，得到 base_jvp: [seq_len, vocab]
            seq_len = int(tokens.attention_mask[0].sum().item()) if "attention_mask" in tokens else act_base.shape[1]
            base_jvps = []
            for idx in range(seq_len):
                def f(act, _idx=idx):
                    def _replace_hook(module, inputs, output):
                        return (act,) + output[1:] if isinstance(output, tuple) else act
                    h = target_layer_module.register_forward_hook(_replace_hook)
                    with sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                        out = model(input_ids, output_hidden_states=False, use_cache=False).logits[0, _idx, :]
                    h.remove()
                    return out

                delta_act = torch.zeros_like(act_base)
                delta_act[0, idx, :] = concept_vector.to(device=act_base.device, dtype=act_base.dtype)
                _, jvp_i = torch.autograd.functional.jvp(f, (act_base,), (delta_act,), create_graph=False)
                base_jvps.append(jvp_i.detach())

            base_jvp = torch.stack(base_jvps, dim=0)  # [seq_len, vocab]

            for i in range(args.alpha_factor):
                concept_vector_alpha = args.concept_vector_alpha * (i + 1)

                # Add hook to modify activations at the target layer
                def _forward_hook(module, inputs, output):
                    # Ensure concept vector matches device/dtype
                    if isinstance(output, tuple):
                        hidden = output[0]
                        vec = concept_vector.to(
                            device=hidden.device, dtype=hidden.dtype
                        )
                        hidden = hidden - concept_vector_alpha * vec
                        return (hidden,) + output[1:]
                    else:
                        vec = concept_vector.to(
                            device=output.device, dtype=output.dtype
                        )
                        return output - concept_vector_alpha * vec

                hook_handle = target_layer_module.register_forward_hook(_forward_hook)
                with torch.enable_grad(), sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                    outputs_with_hook = model(input_ids, output_hidden_states=True)
                hidden_state_diff = outputs_with_hook.logits[0, :seq_len, :] - logits_without_hook[0, :seq_len, :]
                if only_once:
                    logger.info(f"Hidden state diff shape: {hidden_state_diff.shape}")
                    only_once = False
                predicted_output_diff = concept_vector_alpha * base_jvp  # [seq_len, vocab]
                ratios = (hidden_state_diff.to(torch.bfloat16).norm(dim=-1) / predicted_output_diff.to(torch.bfloat16).norm(dim=-1)).tolist()
                per_alpha_token_increases[i].extend(ratios)
                hook_handle.remove()

    progress_bar.close()                    
    # After processing all contexts/examples: per-alpha mean/std over per-token increases
    import numpy as np
    x = [i * args.concept_vector_alpha for i in range(len(per_alpha_token_increases))]
    means = [
        float(np.mean(per_alpha_token_increases[i])) if len(per_alpha_token_increases[i]) > 0 else 0.0
        for i in range(len(per_alpha_token_increases))
    ]
    stds = [
        float(np.std(per_alpha_token_increases[i])) if len(per_alpha_token_increases[i]) > 0 else 0.0
        for i in range(len(per_alpha_token_increases))
    ]
    logger.info(f"counts per alpha: {[len(v) for v in per_alpha_token_increases][:5]} ...")
    plt.errorbar(x, means, yerr=stds, fmt='o', capsize=3, ecolor='gray', alpha=0.9, label='mean ± std')
    plt.plot(x, means, 'r--', linewidth=1)
    plt.xlabel("Alpha", fontsize=12)
    plt.ylabel("||Δlogits|| (per token)", fontsize=12)
    plt.title("Per-token increase vs Alpha (mean ± std)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs("plots/linearity", exist_ok=True)
    plt.savefig(
        f"plots/linearity/predicted_diff_ratios_{args.model.split('/')[-1]}_Layer{args.layer}_concept_vector_{args.concept_vector_path.split('/')[-1].split('.')[0]}_ConceptVectorAlpha{args.concept_vector_alpha}_AlphaFactor{args.alpha_factor}.pdf"
    )
    plt.close()
    os.makedirs("weights/linearity", exist_ok=True)
    torch.save(
        {
            "per_token_increases": per_alpha_token_increases,
            "mean": means,
            "std": stds,
        },
        f"weights/linearity/predicted_diff_ratios_{args.model.split('/')[-1]}_Layer{args.layer}_concept_vector_{args.concept_vector_path.split('/')[-1].split('.')[0]}_ConceptVectorAlpha{args.concept_vector_alpha}_AlphaFactor{args.alpha_factor}.pt",
    )

    logger.info(f"Finished measuring linearity")


if __name__ == "__main__":
    linearity()
