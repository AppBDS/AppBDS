import json
from scipy.stats import pearsonr, spearmanr, kendalltau

def is_sample_qualified(item):
    inspectors = ["inspector1", "inspector2", "inspector3"]
    descriptors = ["DescribeCTX", "DescribeCTX_GPT", "AppBDS"]
    metrics = ["Richness", "Specificity", "Semantic Relatedness"]
    for inspector in inspectors:
        inspector_data = item.get(inspector, {})
        for desc in descriptors:
            desc_data = inspector_data.get(desc, {})
            vals = [
                desc_data.get("Richness", None),
                desc_data.get("Specificity", None),
                desc_data.get("Semantic Relatedness", None)
            ]
            if all(v == 0.0 for v in vals if v is not None) and len(vals) == 3 and not any(v is None for v in vals):
                return False
    return True

def collect_pairs_for_metric(merged_data, inspector_key, metric):
    descriptors = ["DescribeCTX", "DescribeCTX_GPT", "AppBDS"]
    pairs = []
    for item in merged_data:
        if not is_sample_qualified(item):
            continue
        inspector_part = item.get(inspector_key, {})
        for desc in descriptors:
            desc_inspector = inspector_part.get(desc, {})
            human_score = desc_inspector.get(metric, None)
            desc_llm = item.get(desc, {})
            llm_score = desc_llm.get(metric, None)
            if human_score is not None and llm_score is not None:
                pairs.append((human_score, llm_score))
    return pairs

def compute_correlations_for_metric(pairs):
    if len(pairs) < 2:
        return None
    human_scores = [p[0] for p in pairs]
    llm_scores = [p[1] for p in pairs]
    pearson_corr, pearson_pval = pearsonr(human_scores, llm_scores)
    spearman_corr, spearman_pval = spearmanr(human_scores, llm_scores)
    kendall_corr, kendall_pval = kendalltau(human_scores, llm_scores)
    return {
        "Pearson":  (pearson_corr, pearson_pval),
        "Spearman": (spearman_corr, spearman_pval),
        "Kendall":  (kendall_corr, kendall_pval)
    }

def compute_correlation_statistics(merged_data, inspector_key):
    metrics = ["Richness", "Specificity", "Semantic Relatedness"]
    results = {}
    for metric in metrics:
        pairs = collect_pairs_for_metric(merged_data, inspector_key, metric)
        corr_result = compute_correlations_for_metric(pairs)
        results[metric] = corr_result
    return results

if __name__ == "__main__":
    merged_json_path = "correlation_materials_3inspector.json"
    with open(merged_json_path, "r", encoding="utf-8") as f:
        merged_data = json.load(f)

    inspector1_results = compute_correlation_statistics(merged_data, inspector_key="inspector1")
    print("=== Inspector1 vs. LLM ===")
    for metric, corr_dict in inspector1_results.items():
        if corr_dict is None:
            print(f"[{metric}] insufficient data for correlation.")
        else:
            pearson = corr_dict["Pearson"]
            spearman = corr_dict["Spearman"]
            kendall = corr_dict["Kendall"]
            print(f"Metric: {metric}")
            print(f"  Pearson r: {pearson[0]:.4f}, p-value: {pearson[1]:.4g}")
            print(f"  Spearman ρ: {spearman[0]:.4f}, p-value: {spearman[1]:.4g}")
            print(f"  Kendall τ: {kendall[0]:.4f}, p-value: {kendall[1]:.4g}")

    inspector2_results = compute_correlation_statistics(merged_data, inspector_key="inspector2")
    print("=== Inspector2 vs. LLM ===")
    for metric, corr_dict in inspector2_results.items():
        if corr_dict is None:
            print(f"[{metric}] insufficient data for correlation.")
        else:
            pearson = corr_dict["Pearson"]
            spearman = corr_dict["Spearman"]
            kendall = corr_dict["Kendall"]
            print(f"Metric: {metric}")
            print(f"  Pearson r: {pearson[0]:.4f}, p-value: {pearson[1]:.4g}")
            print(f"  Spearman ρ: {spearman[0]:.4f}, p-value: {spearman[1]:.4g}")
            print(f"  Kendall τ: {kendall[0]:.4f}, p-value: {kendall[1]:.4g}")

    print("\n=== Inspector3 vs. LLM ===")
    inspector3_results = compute_correlation_statistics(merged_data, inspector_key="inspector3")
    for metric, corr_dict in inspector3_results.items():
        if corr_dict is None:
            print(f"[{metric}] insufficient data for correlation.")
        else:
            pearson = corr_dict["Pearson"]
            spearman = corr_dict["Spearman"]
            kendall = corr_dict["Kendall"]
            print(f"Metric: {metric}")
            print(f"  Pearson r: {pearson[0]:.4f}, p-value: {pearson[1]:.4g}")
            print(f"  Spearman ρ: {spearman[0]:.4f}, p-value: {spearman[1]:.4g}")
            print(f"  Kendall τ: {kendall[0]:.4f}, p-value: {kendall[1]:.4g}")
