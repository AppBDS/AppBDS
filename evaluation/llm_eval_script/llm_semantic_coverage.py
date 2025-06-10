import json, time, statistics, argparse, openai
from typing import List, Dict
from tqdm import tqdm

def run_llm(prompt: str, temperature: float, openai_api_keys: str, max_tokens: int = 1024, engine: str = "gpt-4o") -> str:
    """Call an LLM and return its response. Retries on errors."""
    client = openai.OpenAI(api_key=openai_api_keys)
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt}
    ]
    while True:
        try:
            response = client.chat.completions.create(
                model=engine,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error: {e}, retrying in 2 seconds...")
            time.sleep(2)

def generate_functionality_coverage_prompt(pp_category: str, reference_text: str, description: str) -> str:
    """Generate a prompt to evaluate the semantic coverage of permission-related functionalities."""
    return f"""
You are evaluating the semantic coverage of permission-related functionalities in app descriptions. Focus on how comprehensively the description covers different functional aspects that utilize the {pp_category} permission.

Reference Functionality Text:
{reference_text}

Description to Evaluate:
{description}

Evaluation Framework:
1. Analyze the reference text to identify distinct functional aspects.
2. For each aspect, identify the core functionality and its implementation details, find semantic matches in the description, and assess the coverage level (none/partial/full).

Scoring Criteria (0-10):
0: No meaningful coverage
1-3: Basic coverage of a few aspects
4-6: Moderate coverage across multiple aspects
7-9: Strong coverage of most aspects
10: Comprehensive coverage across all aspects

Guidelines:
- Count functionally equivalent descriptions even if phrasing differs.
- Consider both high-level functionality and specific implementation details.
- Account for implicit functional references within the broader context.
- Weight each functional aspect based on its complexity and significance.
- Evaluate the overall functional narrative beyond point-by-point matching.

Return a single integer (0-10) representing the comprehensive coverage score. No additional explanation needed.
"""

def score_functionality(input_json_file: str, output_json_file: str, openai_api_key: str, engine: str = "gpt-4o"):
    with open(input_json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = []
    functionality_scores = {"dctx_description": []}
    for sample in tqdm(data, desc="Scoring Functionality"):
        app_id = sample["appId"]
        pp_category = sample["pp_category"]
        reference_text = sample["merged_refined_reference_inspected"]
        descriptions = {"dctx_description": sample["dctx_description"]}
        coverage_scores = {}
        for dkey, desc_text in descriptions.items():
            prompt = generate_functionality_coverage_prompt(pp_category, reference_text, desc_text)
            score_val = float(run_llm(prompt, temperature=0.1, openai_api_keys=openai_api_key, engine=engine))
            functionality_scores[dkey].append(score_val)
            coverage_scores[dkey] = {"functionality": score_val}
        results.append({
            "appId": app_id,
            "pp_category": pp_category,
            "coverage_scores": coverage_scores
        })
    if output_json_file:
        with open(output_json_file, 'w', encoding='utf-8') as outf:
            json.dump(results, outf, indent=2, ensure_ascii=False)
    print("\n=== Final Evaluation Statistics ===")
    for dkey in functionality_scores:
        mean_val = statistics.mean(functionality_scores[dkey])
        stdev_val = statistics.stdev(functionality_scores[dkey]) if len(functionality_scores[dkey]) > 1 else 0.0
        print(f"Description Type: {dkey}\n  Functionality - Mean: {mean_val:.2f}, Std Dev: {stdev_val:.2f}")
    print("\nEvaluation completed and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", help="Input JSON file path")
    parser.add_argument("--output_json", help="Output JSON file path", default=None)
    parser.add_argument("--api_key", default="YOUR_OPENAI_API_KEY", help="OpenAI API key")
    parser.add_argument("--engine", default="gpt-4o", help="OpenAI engine name")
    args = parser.parse_args()
    score_functionality(
        input_json_file=args.input_json,
        output_json_file=args.output_json,
        openai_api_key=args.api_key,
        engine=args.engine
    )
