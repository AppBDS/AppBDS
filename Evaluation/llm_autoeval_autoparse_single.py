import json
import openai
import time
from collections import defaultdict
from tqdm import tqdm

def run_llm(prompt, temperature, openai_api_keys, max_tokens=4096, engine="gpt-4o", sys_msg="normal"):
    if "llama" in engine.lower():
        openai.api_key = "EMPTY"
        openai.api_base = "http://localhost:8000/v1"
        engine = openai.Model.list()["data"][0]["id"]
    else:
        client = openai.OpenAI(api_key=openai_api_keys)

    messages = [
        {"role": "system", "content": sys_msg},
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
            result = response.choices[0].message.content
            return result.strip()
        except Exception as e:
            if 'context_length_exceeded' in str(e):
                print("Context length exceeded or string too long.")
                return "context_length_exceeded"
            else:
                print(f"Unexpected error: {e}, retrying in 2 seconds...")
                time.sleep(2)

def generate_richness_prompt(app_name, pp_category, human_label_detail, description, index):
    return (
        "You are an evaluator of app descriptions, focusing on the variety of information provided. "
        "Evaluate the following description for its information richness, considering the range of scenarios and aspects covered. "
        f"The app description to be evaluated is about the {pp_category} permission category.\n\n"
        "Do not focus on depth or specificity, only on the breadth of information.\n\n"
        f"Reference description: {human_label_detail}\n\n"
        f"Description to be evaluated: {description}\n\n"
        "Assume the reference has a richness score of 7. The minimum score is 0. If the description is better, you can give a score higher than 7.\n\n"
        "Please return a single numerical score (0-10) without any additional text or explanation."
    )

def generate_specificity_prompt(app_name, pp_category, human_label_detail, description, index):
    return (
        "You are an evaluator of app descriptions, focusing on the precision and detail of information. "
        "Evaluate the following description for its specificity, considering the depth and accuracy of terms and descriptions. "
        f"The app description to be evaluated is about the {pp_category} permission category.\n\n"
        "Do not consider length or breadth, only the precision and detail.\n\n"
        f"Reference description: {human_label_detail}\n\n"
        f"Description to be evaluated: {description}\n\n"
        "Assume the reference has a specificity score of 7. The minimum score is 0. If the description is better, you can give a score higher than 7.\n\n"
        "Please return a single numerical score (0-10) without any additional text or explanation."
    )

def generate_factuality_prompt(app_name, pp_category, human_label_detail, description, index):
    return (
        "You are an evaluator of app descriptions, focusing on factual accuracy. "
        "Evaluate the following description for its factuality, considering alignment with the reference in terms of detailed meanings and scenarios. "
        f"The app description to be evaluated is about the {pp_category} permission category.\n\n"
        "Deduct points for omissions or contradictions, and add points for including specific information from the reference.\n\n"
        f"Reference description: {human_label_detail}\n\n"
        f"Description to be evaluated: {description}\n\n"
        "Do not assess content beyond the reference unless there are contradictions or errors.\n\n"
        "Please return a single numerical score (0-10) without any additional text or explanation."
    )

def generate_key_semantic_coverage_prompt(app_name, pp_category, human_label_detail, description, index):
    return (
        "You are an evaluator of app descriptions, focusing on key semantic elements. "
        "Evaluate the following description for its coverage of important concepts and information presented in the reference. "
        f"The app description to be evaluated is about the {pp_category} permission category.\n\n"
        "Assess how well the description captures essential elements and ideas.\n\n"
        f"Reference description: {human_label_detail}\n\n"
        f"Description to be evaluated: {description}\n\n"
        "Deduct points for missing important concepts. Do not consider additional details not present in the reference.\n\n"
        "Assume the reference has a key semantic coverage score of 10. The minimum score is 0. If the description matches the reference, you can give a score of 10.\n\n"
        "Please return a single numerical score (0-10) without any additional text or explanation."
    )

def generate_consistency_prompt(app_name, pp_category, human_label_detail, description, index):
    return (
        "You are an evaluator of app descriptions, focusing on consistency. "
        "Evaluate the following description for its consistency with the reference, checking for contradictions or confusing elements. "
        f"The app description to be evaluated is about the {pp_category} permission category.\n\n"
        "Assess whether the description maintains coherence and accurately reflects the information in the reference.\n\n"
        f"Reference description: {human_label_detail}\n\n"
        f"Description to be evaluated: {description}\n\n"
        "Deduct points for contradictions or confusing elements. Do not assess content beyond the reference unless there are contradictions or errors.\n\n"
        "Assume the reference has a consistency score of 10. The minimum score is 0. If the description matches the reference, you can give a score of 10.\n\n"
        "Please return a single numerical score (0-10) without any additional text or explanation."
    )

def generate_key_semantics_quality_prompt(app_name, pp_category, human_label_detail, description, index):
    return (
        "You are an evaluator of app descriptions, focusing on the quality of content related to key semantics identified in the reference. "
        "Evaluate the following description for its key semantics quality, considering how well it captures the key semantics from the reference and elaborates on them with detailed, high-quality information.\n\n"
        f"The app description to be evaluated is about the {pp_category} permission category.\n\n"
        "Assess the parts of the description that discuss the key semantics from the reference, focusing on the richness of information, depth of detail, factual accuracy, and logical coherence.\n\n"
        f"Reference description: {human_label_detail}\n\n"
        f"Description to be evaluated: {description}\n\n"
        "Assume the reference has a key semantics quality score of 7. The minimum score is 0. If the evaluated description performs better, you can give a score higher than 7.\n\n"
        "Please return a single numerical score (0-10) without any additional text or explanation."
    )

def score_descriptions(apps_json_file_path, openai_api_keys, output_json_file_path, exclude_keys, engine="gpt-4o"):
    with open(apps_json_file_path, 'r', encoding='utf-8') as file):
        apps_data = json.load(file)

    results = []
    key_scores = defaultdict(list)

    for app in tqdm(apps_data, desc="Evaluating apps"):
        app_name = app['appId']
        pp_category = app['pp_category']
        human_label_detail = app['descriptions'].get('human_label_detail', '')

        descriptions = [desc for key, desc in app['descriptions'].items() if key not in exclude_keys]  
        description_keys = [key for key in app['descriptions'].keys() if key not in exclude_keys]

        richness_scores = {}
        specificity_scores = {}
        factuality_scores = {}
        key_semantic_coverage_scores = {}
        consistency_scores = {}
        key_semantics_quality_scores = {}

        for i, desc in enumerate(descriptions):
            richness_prompt = generate_richness_prompt(app_name, pp_category, human_label_detail, desc, i)
            richness_score = run_llm(richness_prompt, temperature=0.1, openai_api_keys=openai_api_keys, engine=engine, sys_msg="You are an expert evaluator of app descriptions.")
            richness_scores[description_keys[i]] = int(richness_score)

            specificity_prompt = generate_specificity_prompt(app_name, pp_category, human_label_detail, desc, i)
            specificity_score = run_llm(specificity_prompt, temperature=0.1, openai_api_keys=openai_api_keys, engine=engine, sys_msg="You are an expert evaluator of app descriptions.")
            specificity_scores[description_keys[i]] = int(specificity_score)

            factuality_prompt = generate_factuality_prompt(app_name, pp_category, human_label_detail, desc, i)
            factuality_score = run_llm(factuality_prompt, temperature=0.1, openai_api_keys=openai_api_keys, engine=engine, sys_msg="You are an expert evaluator of app descriptions.")
            factuality_scores[description_keys[i]] = int(factuality_score)

            key_semantic_coverage_prompt = generate_key_semantic_coverage_prompt(app_name, pp_category, human_label_detail, desc, i)
            key_semantic_coverage_score = run_llm(key_semantic_coverage_prompt, temperature=0.1, openai_api_keys=openai_api_keys, engine=engine, sys_msg="You are an expert evaluator of app descriptions.")
            key_semantic_coverage_scores[description_keys[i]] = int(key_semantic_coverage_score)

            consistency_prompt = generate_consistency_prompt(app_name, pp_category, human_label_detail, desc, i)
            consistency_score = run_llm(consistency_prompt, temperature=0.1, openai_api_keys=openai_api_keys, engine=engine, sys_msg="You are an expert evaluator of app descriptions.")
            consistency_scores[description_keys[i]] = int(consistency_score)

            key_semantics_quality_prompt = generate_key_semantics_quality_prompt(app_name, pp_category, human_label_detail, desc, i)
            key_semantics_quality_score = run_llm(key_semantics_quality_prompt, temperature=0.1, openai_api_keys=openai_api_keys, engine=engine, sys_msg="You are an expert evaluator of app descriptions.")
            key_semantics_quality_scores[description_keys[i]] = int(key_semantics_quality_score)

            key_scores[f"Richness {description_keys[i]}"].append(richness_scores[description_keys[i]])
            key_scores[f"Specificity {description_keys[i]}"].append(specificity_scores[description_keys[i]])
            key_scores[f"Factuality {description_keys[i]}"].append(factuality_scores[description_keys[i]])
            key_scores[f"Key Semantic Coverage {description_keys[i]}"].append(key_semantic_coverage_scores[description_keys[i]])
            key_scores[f"Consistency {description_keys[i]}"].append(consistency_scores[description_keys[i]])
            key_scores[f"Key Semantics Quality {description_keys[i]}"].append(key_semantics_quality_scores[description_keys[i]])

        result = {
            "appId": app_name,
            "Richness scores": richness_scores,
            "Specificity scores": specificity_scores,
            "Factuality scores": factuality_scores,
            "Key Semantic Coverage scores": key_semantic_coverage_scores,
            "Consistency scores": consistency_scores,
            "Key Semantics Quality scores": key_semantics_quality_scores
        }
        results.append(result)

        with open(output_json_file_path, 'w', encoding='utf-8') as output_file:
            json.dump(results, output_file, ensure_ascii=False, indent=4)

    avg_scores = {key: sum(scores) / len(scores) for key, scores in key_scores.items()}
    for key, avg_score in avg_scores.items():
        print(f"Average score for {key}: {avg_score:.2f}")

if __name__ == "__main__":
    apps_json_file_path = 'your_input_file_path.json'
    openai_api_keys = 'your_openai_api_key'
    output_json_file_path = 'your_output_file_path.json'

    exclude_keys = []

    print("Evaluating descriptions...")
    score_descriptions(apps_json_file_path, openai_api_keys, output_json_file_path, exclude_keys)
