import json
import torch
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import statistics
import argparse

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_model():
    model_name = "microsoft/deberta-xlarge-mnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = "cpu"
    model = model.to(device)
    model.eval()
    return tokenizer, model, device

def get_bert_embeddings_batch(texts, tokenizer, model, device):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]

def compute_bert_scores_batch(reference_texts, candidate_texts, tokenizer, model, device):
    ref_embeddings = get_bert_embeddings_batch(reference_texts, tokenizer, model, device)
    cand_embeddings = get_bert_embeddings_batch(candidate_texts, tokenizer, model, device)
    return cosine_similarity(ref_embeddings, cand_embeddings, dim=1).tolist()

def calculate_bert_scores(input_json_file, output_json_file=None):
    data = load_json(input_json_file)
    tokenizer, model, device = load_model()
    fields = [
        "condensed_detailed_description_based_on_subgraph_common_func",
        "condensed_detailed_description_based_on_subgraph_prop",
        "condensed_detailed_description_based_on_subgraph_similar_apps",
        "condensed_final_description_round2"
    ]
    bert_scores = {field: [] for field in fields}
    results = []
    batch_size = 16

    for i in tqdm(range(0, len(data), batch_size), desc="Calculating BERT Scores"):
        batch = data[i:i + batch_size]
        reference_texts = [sample["merged_refined_reference_inspected"] for sample in batch]
        for field in fields:
            candidate_texts = [sample.get(field, "") for sample in batch]
            scores = compute_bert_scores_batch(reference_texts, candidate_texts, tokenizer, model, device)
            bert_scores[field].extend(scores)
            for j, sample in enumerate(batch):
                if "bert_scores" not in sample:
                    sample["bert_scores"] = {}
                sample["bert_scores"][field] = scores[j]
                results.append(sample)

    print("\n=== Average BERT Scores ===")
    for field, scores in bert_scores.items():
        if scores:
            try:
                avg_score = statistics.mean(scores)
                print(f"{field}: {avg_score:.4f}")
            except TypeError as e:
                print(f"Error calculating mean for {field}: {e}")
        else:
            print(f"{field}: No scores available")

    if output_json_file:
        with open(output_json_file, 'w', encoding='utf-8') as outf:
            json.dump(results, outf, indent=2, ensure_ascii=False)
        print(f"\nBERT scores saved to {output_json_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", help="Input JSON file path")
    parser.add_argument("--output_json", help="Output JSON file path", default=None)
    args = parser.parse_args()
    calculate_bert_scores(args.input_json, args.output_json)
