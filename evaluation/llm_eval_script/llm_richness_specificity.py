import json
import time
import statistics
import openai
import argparse
from tqdm import tqdm

def run_llm(prompt: str, temperature: float, openai_api_keys: str, max_tokens: int = 1024, engine: str = "gpt-4o") -> str:
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

def generate_richness_prompt(pp_category: str, reference_text: str, description: str) -> str:
    return f"""
You are evaluating app descriptions for permission-related functionalities.
The reference text provides an overview of functionalities related to the {pp_category} permission and privacy data handling.

Reference text:
{reference_text}

Evaluated description:
{description}

Scoring criteria (0–10) for RICHNESS:
- The reference text is a baseline score of 7.
- "0" means the description lacks diversity in discussing permission-related functionalities.
- "10" means the description covers a wide range of functionalities related to permission use, regardless of direct alignment with the reference text.
- Intermediate scores reflect partial coverage.

No extra explanation. Just return a single integer (0–10).
"""

def generate_specificity_prompt(pp_category: str, reference_text: str, description: str) -> str:
    return f"""
You are evaluating app descriptions for permission-related functionalities.
The reference text provides an overview of functionalities related to the {pp_category} permission and privacy data handling.

Reference text:
{reference_text}

Evaluated description:
{description}

Scoring criteria (0–10) for SPECIFICITY:
- The reference text is a baseline score of 7.
- "0" means the description lacks detailed and precise information on permission-related functionalities.
- "10" means the description provides highly specific and detailed explanations for each permission-related functionality.
- Intermediate scores reflect partial specificity.

No extra explanation. Just return a single integer (0–10).
"""

def score_richness_specificity(input_json_file: str, output_json_file: str, openai_api_key: str, engine: str = "gpt-4o"):
    with open(input_json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    richness_scores = {"dctx_description": []}
    specificity_scores = {"dctx_description": []}

    for sample in tqdm(data, desc="Scoring Richness/Specificity"):
        app_id = sample["appId"]
        pp_category = sample["pp_category"]
        reference_text = sample["merged_refined_reference"]
        descriptions = {"dctx_description": sample["dctx_description"]}
        coverage_scores = {}
        for dkey, desc_text in descriptions.items():
            rprompt = generate_richness_prompt(pp_category, reference_text, desc_text)
            richness_val = float(run_llm(rprompt, temperature=0.1, openai_api_keys=openai_api_key, engine=engine))
            richness_scores[dkey].append(richness_val)
            sprompt = generate_specificity_prompt(pp_category, reference_text, desc_text)
            specificity_val = float(run_llm(sprompt, temperature=0.1, openai_api_keys=openai_api_key, engine=engine))
            specificity_scores[dkey].append(specificity_val)
            coverage_scores[dkey] = {"richness": richness_val, "specificity": specificity_val}
        results.append({"appId": app_id, "pp_category": pp_category, "coverage_scores": coverage_scores})

    if output_json_file:
        with open(output_json_file, 'w', encoding='utf-8') as outf:
            json.dump(results, outf, indent=2, ensure_ascii=False)

    print("\n=== Final Evaluation Statistics ===")
    for dkey in richness_scores.keys():
        r_mean = statistics.mean(richness_scores[dkey])
        r_stdev = statistics.stdev(richness_scores[dkey]) if len(richness_scores[dkey]) > 1 else 0.0
        s_mean = statistics.mean(specificity_scores[dkey])
        s_stdev = statistics.stdev(specificity_scores[dkey]) if len(specificity_scores[dkey]) > 1 else 0.0
        print(f"\nDescription Type: {dkey}")
        print(f"Richness - Mean: {r_mean:.2f}, Std Dev: {r_stdev:.2f}")
        print(f"Specificity - Mean: {s_mean:.2f}, Std Dev: {s_stdev:.2f}")

    print("\nEvaluation completed and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", help="Input JSON file path")
    parser.add_argument("--output_json", help="Output JSON file path", default=None)
    parser.add_argument("--api_key", default="YOUR_OPENAI_API_KEY", help="OpenAI API key")
    parser.add_argument("--engine", default="gpt-4o", help="OpenAI engine name")
    args = parser.parse_args()

    score_richness_specificity(
        input_json_file=args.input_json,
        output_json_file=args.output_json,
        openai_api_key=args.api_key,
        engine=args.engine
    )
