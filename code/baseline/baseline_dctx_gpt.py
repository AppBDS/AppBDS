import json
import csv
import re
import time
import os
import openai

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
            return response.choices[0].message.content.strip()
        except Exception as e:
            if 'context_length_exceeded' in str(e):
                print("Context length exceeded or string too long.")
                return "context_length_exceeded"
            else:
                print(f"Unexpected error: {e}, retrying in 2 seconds...")
                time.sleep(2)

def load_existing_results(output_json):
    if os.path.exists(output_json) and os.path.getsize(output_json) > 0:
        with open(output_json, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                return {(entry["appId"], entry["pp_category"]) for entry in data}
            except json.JSONDecodeError:
                print(f"Warning: Could not decode {output_json}. Starting fresh.")
                return set()
    return set()

if __name__ == "__main__":
    input_csv = '/path/to/input.csv'
    output_json = '/path/to/output.json'
    openai_api_keys = 'YOUR_OPENAI_API_KEY'

    existing_results = load_existing_results(output_json)

    with open(input_csv, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        total_rows = sum(1 for _ in reader)
        file.seek(0)
        next(reader)
        new_results = []
        
        for idx, row in enumerate(reader):
            app_id = row.get('appId')  
            pp_category = row.get('pp_category')
            pp = row.get('pp')
            ui = row.get('ui')
            cg = row.get('cg')
            optimized_summary = row.get('optimized_summary')

            if not all([app_id, pp_category, pp, ui, cg, optimized_summary]):
                print(f"Skipping row with missing fields: {row}")
                continue

            if (app_id, pp_category) in existing_results:
                print(f"Skipping already processed: {app_id}, {pp_category}")
                continue

            prompt = (
                f"Based ONLY on the provided context information, summarize how this app handles {pp_category} data. "
                f"Focus on clearly evident privacy implications and how these permissions serve the app's core objectives. "
                f"Be concise and factual, avoiding any speculation or unsupported claims.\n\n"
                f"Do not add any uncertain content. The ideal length is 75-150 words, but prioritize clarity and evidence over word count. "
                f"If there is insufficient evidence or information, provide a shorter description rather than including unsubstantiated information. \n\n"
                f"Input Data:\n"
                f"- pp_category: {pp_category}\n"
                f"- Privacy Policy: {pp}\n"
                f"- UI Information: {ui}\n"
                f"- ICCG Function Signatures: {cg}\n\n"
                f"Output Format:\n"
                f"Provide a concise paragraph summarizing the app's handling of {pp_category} data, focusing on confirmed "
                f"functionality, privacy implications, and their connection to the app's objectives. Only include information "
                f"clearly supported by the provided context."
            )

            print(f"Processing: {app_id}, {pp_category}")
            raw_output = run_llm(prompt, temperature=0.3, openai_api_keys=openai_api_keys)

            result = {
                "appId": app_id,
                "pp_category": pp_category,
                "context": {
                    "privacy_policy": pp,
                    "ui_information": ui,
                    "iccg_signatures": cg
                },
                "raw_output": raw_output
            }

            new_results.append(result)
            print(f"Processed {idx + 1} entries")

    if new_results:
        with open(output_json, 'r+', encoding='utf-8') as outfile:
            outfile.seek(0, os.SEEK_END)
            position = outfile.tell()
            if position > 1:
                outfile.seek(position - 1)
                outfile.truncate()
                outfile.write(',\n')
            json.dump(new_results, outfile, ensure_ascii=False, indent=4)
            outfile.write('\n]')
    print(f"Results saved to {output_json}")
