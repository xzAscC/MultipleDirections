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


def linearity():
    """used to measure the linearity of the model"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-2b-it",
        choices=["google/gemma-2-2b-it", "Qwen/Qwen3-1.7B", "EleutherAI/pythia-70m"],
    )
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument(
        "--dataset_path", type=str, default="assets/paired_contexts/en-fr.jsonl"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--concept_vector_path",
        type=str,
        default="weights/concept_vectors/random_concept_vector_gemma-2-2b-it.pt",
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
    logger.add("logs/linearity.log")
    logger.info("Starting linearity measurement...")
    logger.info(f"Loading model: {args.model}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=args.device,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
    )
    # model = transformer_lens.HookedTransformer.from_pretrained(
    #     args.model,
    #     device=args.device,
    #     dtype=getattr(torch, args.dtype),
    # )
    logger.info(f"Loading dataset: {args.dataset_path}")
    dataset = datasets.load_dataset("json", data_files=args.dataset_path)["train"]
    logger.info(f"Loaded {len(dataset)} examples")
    logger.info(f"Loading concept vector: {args.concept_vector_path}")
    concept_vector = torch.load(args.concept_vector_path)
    logger.info(f"Concept vector shape: {concept_vector.shape}")

    def hook_fn(activations, hook):
        return activations - args.concept_vector_alpha * concept_vector

    # measure the linearity of the model
    only_once = True
    predicted_diff_ratios = []
    for example in tqdm(dataset, desc="Measuring linearity"):
        # https://github.com/sugolov/coupling/blob/main/coupling/main.py#L35
        contexts = example["contexts0"]
        for context in contexts:
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

            # 2) 定义 f(act): 用 hook 将该层输出替换为 act，返回目标 token 的 logits 向量
            index = 10  # 与你之前 jacobian 的 index 保持一致
            def f(act):
                def _replace_hook(module, inputs, output):
                    return (act,) + output[1:] if isinstance(output, tuple) else act
                h = target_layer_module.register_forward_hook(_replace_hook)
                with sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                    out = model(input_ids, output_hidden_states=False).logits[0, index, :]  # 形状: [vocab]
                h.remove()
                return out

            # 3) 准备方向向量 delta_act（只在 index 位置注入 concept_vector）
            delta_act = torch.zeros_like(act_base)
            delta_act[0, index, :] = concept_vector.to(device=act_base.device, dtype=act_base.dtype)

            # 4) 一次性 JVP：得到 base_jvp = J @ concept_vector
            _, base_jvp = torch.autograd.functional.jvp(f, (act_base,), (delta_act,), create_graph=False)

            # 5) 在你多 alpha 循环里，用 predicted_output_diff = concept_vector_alpha * base_jvp
            #    注意：这里不需要再算 jacobian_without_hook 了

            for i in tqdm(
                range(args.alpha_factor),
                desc="Running model with hook for different alphas",
            ):
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
                hidden_state_diff = outputs_with_hook.logits - logits_without_hook
                if only_once:
                    logger.info(f"Hidden state diff shape: {hidden_state_diff.shape}")
                    only_once = False
                predicted_output_diff = base_jvp
                predicted_diff_ratios.append(
                    hidden_state_diff[0, 10, :].to(torch.bfloat16).norm(dim=0).item()
                    / predicted_output_diff.to(torch.bfloat16).norm(dim=0).item()
                )
                hook_handle.remove()

            # Create x-axis values (alpha values)
            x = [
                i * args.concept_vector_alpha for i in range(len(predicted_diff_ratios))
            ]
            y = predicted_diff_ratios

            # Create scatter plot
            plt.scatter(x, y, alpha=0.6, s=50)

            # Fit a linear regression line
            x_tensor = torch.tensor(x, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)
            # Use numpy for polynomial fitting since torch doesn't have polyfit
            import numpy as np

            coeffs = np.polyfit(x_tensor.cpu().numpy(), y_tensor.cpu().numpy(), 1)
            poly_fn = np.poly1d(coeffs)

            # Plot the fitted line
            plt.plot(
                x,
                [poly_fn(xi) for xi in x],
                "r--",
                linewidth=2,
                label=f"Fit: y={coeffs[0]:.3f}x+{coeffs[1]:.3f}",
            )

            # Add labels and title
            plt.xlabel("Alpha", fontsize=12)
            plt.ylabel("Predicted Diff Ratio", fontsize=12)
            plt.title("Predicted Diff Ratio vs Alpha", fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)

            os.makedirs("plots/linearity", exist_ok=True)
            plt.savefig(
                f"plots/linearity/predicted_diff_ratios_{args.model.split('/')[-1]}_Layer{args.layer}_concept_vector_{args.concept_vector_path.split('/')[-1].split('.')[0]}_ConceptVectorAlpha{args.concept_vector_alpha}_AlphaFactor{args.alpha_factor}.pdf"
            )
            plt.close()
            os.makedirs("weights/linearity", exist_ok=True)
            torch.save(
                predicted_diff_ratios,
                f"weights/linearity/predicted_diff_ratios_{args.model.split('/')[-1]}_Layer{args.layer}_concept_vector_{args.concept_vector_path.split('/')[-1].split('.')[0]}_ConceptVectorAlpha{args.concept_vector_alpha}_AlphaFactor{args.alpha_factor}.pt",
            )
            exit()
    logger.info(f"Finished measuring linearity")


def jacobian(output, input, index, chunks, index_in=None, device="cuda"):
    """
    Reference: https://github.com/sugolov/coupling/blob/main/coupling/jacobian.py
    Computes the Jacobian of `d{output}/d{input}` from transformer hooks
    by vectorizing over gradients.

    output:     Jacobian wrt this output
    input:      Jacobian wrt this input
    index:      index of output token
    chunks:     number of chunks used to vectorize Jacobian computation
    index_in:   (optional) changes input token of Jacobian if not `index`
    """

    output = output[0, index, :]
    I_N = torch.eye(output.numel()).to(device)

    index_in = index_in if index_in is not None else index

    def get_vjp(v):
        return grad(output, input, v, retain_graph=True)[0][0, index_in, :]

    jacobian = chunk_vmap(get_vjp, chunks=chunks)(I_N)

    return jacobian


if __name__ == "__main__":
    linearity()
