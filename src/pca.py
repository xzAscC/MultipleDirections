import argparse
import torch
import os
import time
import transformer_lens
import huggingface_hub
import datasets
import json
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI
from sklearn.decomposition import IncrementalPCA
from loguru import logger

load_dotenv()

APPRISE_GMAIL = os.getenv("APPRISE_GMAIL")
APPRISE_PWD = os.getenv("APPRISE_PWD")
HF_TOKEN = os.getenv("HF_TOKEN")

huggingface_hub.login(token=HF_TOKEN)


def steering(pretrained=True):
    """
    Steering
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-2b-it",
        choices=["google/gemma-2-2b-it", "Qwen/Qwen3-1.7B"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument(
        "--method",
        type=str,
        default="IncrementalPCA",
        choices=["IncrementalPCA", "dim"],
    )
    parser.add_argument("--torch_type", type=str, default="bfloat16")
    parser.add_argument("--dataset_path", type=str, default="assets/paired_contexts")
    args = parser.parse_args()

    logger.add("logs/pca.log")
    logger.info("Starting PCA Steering...")
    dtype = getattr(torch, args.torch_type)
    model = transformer_lens.HookedTransformer.from_pretrained(
        args.model, device=args.device, dtype=dtype
    )
    input_dataset = []
    steering_dataset = []
    input_target_words = []
    steering_target_words = []
    filenames = os.listdir(args.dataset_path)

    for filename in filenames:
        file_path = os.path.join(args.dataset_path, filename)
        with open(file_path, "r") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    obj = json.loads(raw_line)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON line in {file_path}")
                    continue
                input_dataset.extend(obj["contexts0"])
                steering_dataset.extend(obj["contexts1"])
                input_len = len(obj["contexts0"])
                input_target_words.extend([obj["word0"]] * input_len)
                steering_target_words.extend([obj["word1"]] * input_len)
        logger.info(
            f"Loaded {len(input_dataset)} input examples and {len(steering_dataset)} steering examples"
        )
        if not pretrained:
            # extract the concept vectors
            concept_vectors = extract_concept_vectors(
                steering_dataset, model, args.layer, args.method
            )
            logger.info(f"Concept vectors: {concept_vectors.shape}")
            torch.save(
                concept_vectors,
                f"weights/concept_vectors_Layer{args.layer}_Methods{args.method}_Model{args.model.replace('/', '_')}_FileName{filename.replace('.jsonl', '')}.pt",
            )
        else:
            concept_vectors = torch.load(
                f"weights/concept_vectors_Layer{args.layer}_Methods{args.method}_Model{args.model.replace('/', '_')}_FileName{filename.replace('.jsonl', '')}.pt",
                weights_only=False,
            )
            logger.info(f"Concept vectors: {concept_vectors.shape}")
        pca_num = 1
        concept_vectors = torch.tensor(concept_vectors)

        def hook_fn(activations, hook):
            principal_components = (
                concept_vectors[:pca_num, :]
                .real.to(activations.device)
                .to(activations.dtype)
            )
            # Normalize each principal component
            principal_components = principal_components / torch.norm(
                principal_components, dim=0, keepdim=True
            )
            # Project activations onto the principal components and subtract
            projection = torch.matmul(activations, principal_components.T)
            projection = torch.matmul(projection, principal_components)
            return activations - projection

        logger.info(f"Added hook to layer {args.layer}")
        model.add_hook(f"blocks.{args.layer}.hook_resid_post", hook_fn)
        torch.cuda.empty_cache()
        steering_probs = []
        input_probs = []
        progress_bar = tqdm(total=len(input_dataset), desc="Steering")
        for _, (inputs, target_word, input_word) in enumerate(
            zip(input_dataset, steering_target_words, input_target_words),
        ):
            target_id = model.to_tokens(target_word, prepend_bos=False).squeeze()
            input_id = model.to_tokens(input_word, prepend_bos=False).squeeze()
            input_len = len(input_id) if input_id.dim() > 0 else 1
            _, cache = model.run_with_cache(inputs)
            hs_needed = cache[f"ln_final.hook_normalized"].to(dtype)
            logits = model.unembed(hs_needed[0, -1-input_len, :].to(dtype))
            prob = torch.softmax(logits, dim=-1)
            target_id = target_id[0] if target_id.dim() > 0 else target_id
            input_id = input_id[0] if input_id.dim() > 0 else input_id
            steering_probs.append(prob[target_id])
            input_probs.append(prob[input_id])
            progress_bar.update(1)
        torch.save(
            steering_probs,
            f"weights/steering_probs_pca_Layer{args.layer}_{pca_num}_Methods{args.method}_Model{args.model.replace('/', '_')}_FileName{filename.replace('.jsonl', '')}.pt",
        )
        torch.save(
            input_probs,
            f"weights/input_probs_pca_Layer{args.layer}_{pca_num}_Methods{args.method}_Model{args.model.replace('/', '_')}_FileName{filename.replace('.jsonl', '')}.pt",
        )

        # model.reset_hooks()
        # torch.cuda.empty_cache()
        # steering_probs_no_pca = []
        # input_probs_no_pca = []
        # progress_bar = tqdm(total=len(input_dataset), desc="Steering")
        # for _, (inputs, target_word, input_word) in enumerate(
        #     zip(input_dataset, steering_target_words, input_target_words),
        # ):

        #     target_id = model.to_tokens(target_word, prepend_bos=False).squeeze()
        #     input_id = model.to_tokens(input_word, prepend_bos=False).squeeze()
        #     input_len = len(input_id) if input_id.dim() > 0 else 1                
        #     _, cache = model.run_with_cache(inputs)
        #     hs_needed = cache[f"ln_final.hook_normalized"].to(dtype)
        #     logits = model.unembed(hs_needed[0, -1-input_len, :].to(dtype))
        #     probs = torch.softmax(logits, dim=-1)
        #     target_id = target_id[0] if target_id.dim() > 0 else target_id
        #     input_id = input_id[0] if input_id.dim() > 0 else input_id
        #     steering_probs_no_pca.append(probs[target_id].item())
        #     input_probs_no_pca.append(probs[input_id].item())
        #     progress_bar.update(1)
        # torch.save(
        #     steering_probs_no_pca,
        #     f"weights/steering_probs_original_Layer{args.layer}_Methods{args.method}_Model{args.model.replace('/', '_')}_FileName{filename.replace('.jsonl', '')}.pt",
        # )
        # torch.save(
        #     input_probs_no_pca,
        #     f"weights/input_probs_original_Layer{args.layer}_Methods{args.method}_Model{args.model.replace('/', '_')}_FileName{filename.replace('.jsonl', '')}.pt",
        # )
        exit()


def extract_concept_vectors(steering_dataset, model, layer, method):
    if method == "IncrementalPCA":
        ipca = IncrementalPCA()
        for context in tqdm(steering_dataset, desc="Extracting concept vectors"):
            _, cache = model.run_with_cache(context)
            hs_needed = cache[f"blocks.{layer}.hook_resid_post"]  # [1, bsz, d]
            if method == "IncrementalPCA":
                ipca.partial_fit(hs_needed[0].to(torch.float32).detach().cpu().numpy())
        return ipca.components_


if __name__ == "__main__":
    steering()

    #     model.add_hook(f'blocks.{layer}.hook_resid_post', hook_fn)
