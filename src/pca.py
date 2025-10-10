import argparse
import torch
import os
import transformer_lens
import huggingface_hub
import datasets
import json
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

APPRISE_GMAIL = os.getenv("APPRISE_GMAIL")
APPRISE_PWD = os.getenv("APPRISE_PWD")
HF_TOKEN = os.getenv("HF_TOKEN")

huggingface_hub.login(token=HF_TOKEN)

def concept_steering(concept_vector):
    pass

def eval_steering(concept_response):
    pass

def steering():
    """
    PCA Steering
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--concepts", type=list, default=['safety', 'gender'])
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--method", type=str, default="pca")
    parser.add_argument("--torch_type", type=str, default="bfloat16")
    parser.add_argument("--dataset_name", type=str, default="walledai/HarmBench")
    args = parser.parse_args()
    
    dtype = getattr(torch, args.torch_type)
    model = transformer_lens.HookedTransformer.from_pretrained(args.model, device=args.device, dtype=dtype)
    
    dataset = datasets.load_dataset(args.dataset_name, "contextual", streaming=True)

    concept_vector = extract_concept_vector(dataset, model, args.layer, args.method)
    
    # concept_response = concept_steering(concept_vector)
    
    # eval_steering(concept_response)

def extract_concept_vector(dataset: datasets.Dataset, model: transformer_lens.HookedTransformer, layer: int, method: str, save_path: str='./assets/harmbench_responses.jsonl'):    
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    concept_vectors = []
    for idx, example in tqdm(enumerate(dataset["train"])):
        response = model.generate(example["prompt"], max_new_tokens=1000)
        with open(save_path, 'a') as f:
            f.write(json.dumps({"prompt": example["prompt"], "response": response}) + '\n')
        if idx > 10:
            break

if __name__ == "__main__":
    steering()
    