import json
import time
import re
import os
import networkx as nx
from networkx.drawing.nx_pydot import read_dot
from itertools import islice
import numpy as np
import random
import openai
import anthropic
import pandas as pd
import requests

import bert_score
from sentence_transformers import util, SentenceTransformer
from transformers import pipeline
from collections import defaultdict
from typing import Optional, List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity


def run_llm(prompt, temperature, args, max_tokens=4096, engine="gpt-4o", sys_msg="normal"):
    if sys_msg == "des":
        system_message = "You are an AI assistant that helps generate comprehensive and precise app descriptions."
    elif sys_msg == "extract":
        system_message = "You are an AI assistant that extracts filter information from text."
    elif sys_msg == "choose":
        system_message = "You are an AI assistant that helps people make choices based on information."
    elif sys_msg == "gen":
        system_message = "You are an AI assistant that generates text based on prompts."
    else:
        system_message = "You are a helpful AI assistant."
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    if "llama" in engine.lower():
        try:
            import openai
            openai.api_key = "EMPTY"
            openai.api_base = "http://localhost:8000/v1"
            engine = openai.Model.list()["data"][0]["id"]
            client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
            response = client.chat.completions.create(
                model=engine,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return ""
    try:
        import openai
        client = openai.OpenAI(api_key=args.openai_api_keys)
        response = client.chat.completions.create(
            model=engine,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        if args.llm_error_handling_setting == "one_try":
            return ""
        elif args.llm_error_handling_setting == "multi_try":
            try:
                url = "https://api.anthropic.com/v1/messages"
                headers = {
                    "x-api-key": args.anthropic_api_keys,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                }
                data = {
                    "model": "claude-3-5-haiku-20241022",
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "system": system_message,
                    "messages": [{"role": "user", "content": prompt}]
                }
                response = requests.post(url, headers=headers, json=data)
                if response.status_code == 200:
                    response_json = response.json()
                    return response_json["content"][0]["text"]
                else:
                    return ""
            except Exception:
                return ""
        else:
            return ""


def run_llm_seq(prompts, temperature, args, max_tokens=4096, engine="gpt-4o", sys_msg="normal"):
    if sys_msg == "des":
        system_message = "You are an AI assistant that helps generate comprehensive and precise app descriptions."
    elif sys_msg == "extract":
        system_message = "You are an AI assistant that extracts filter information from text."
    elif sys_msg == "choose":
        system_message = "You are an AI assistant that helps people make choices based on information."
    elif sys_msg == "gen":
        system_message = "You are an AI assistant that generates text based on prompts."
    else:
        system_message = "You are a helpful AI assistant."
    if "llama" in engine.lower():
        results = []
        try:
            import openai
            openai.api_key = "EMPTY"
            openai.api_base = "http://localhost:8000/v1"
            engine = openai.Model.list()["data"][0]["id"]
            client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
            messages = [{"role": "system", "content": system_message}]
            for prompt in prompts:
                user_message = {"role": "user", "content": prompt}
                messages.append(user_message)
                try:
                    response = client.chat.completions.create(
                        model=engine,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    result = response.choices[0].message.content
                    results.append(result)
                    messages.append({"role": "assistant", "content": result})
                except Exception:
                    results.append("")
                    messages.append({"role": "assistant", "content": ""})
            return results
        except Exception:
            return [""] * len(prompts)
    results = []
    import openai
    openai_messages = [{"role": "system", "content": system_message}]
    using_claude = False
    claude_messages = []
    for prompt in prompts:
        if not using_claude:
            openai_messages.append({"role": "user", "content": prompt})
        if not using_claude:
            try:
                client = openai.OpenAI(api_key=args.openai_api_keys)
                response = client.chat.completions.create(
                    model=engine,
                    messages=openai_messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                result = response.choices[0].message.content
                results.append(result)
                openai_messages.append({"role": "assistant", "content": result})
                continue
            except Exception:
                if args.llm_error_handling_setting == "one_try":
                    results.append("")
                    openai_messages.append({"role": "assistant", "content": ""})
                    continue
        if args.llm_error_handling_setting == "multi_try" and not using_claude:
            using_claude = True
            claude_messages = []
            for msg in openai_messages[1:]:
                claude_messages.append({"role": msg["role"], "content": msg["content"]})
            if not any(msg["content"] == prompt and msg["role"] == "user" for msg in claude_messages):
                claude_messages.append({"role": "user", "content": prompt})
        if using_claude:
            try:
                url = "https://api.anthropic.com/v1/messages"
                headers = {
                    "x-api-key": args.anthropic_api_keys,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                }
                data = {
                    "model": "claude-3-5-haiku-20241022",
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "system": system_message,
                    "messages": claude_messages
                }
                response = requests.post(url, headers=headers, json=data)
                if response.status_code == 200:
                    response_json = response.json()
                    result = response_json["content"][0]["text"]
                    results.append(result)
                    claude_messages.append({"role": "assistant", "content": result})
                else:
                    results.append("")
                    claude_messages.append({"role": "assistant", "content": ""})
            except Exception:
                results.append("")
                claude_messages.append({"role": "assistant", "content": ""})
    return results


def format_subgraph_node_info(node: Dict) -> str:
    formatted_info = ""
    node_type = node.get('subgraph_node_info', {}).get('node_type', 'Unknown')
    formatted_info += f"\nNode Type: {node_type}"
    formatted_info += f"\nSignature: {node.get('signature', 'N/A')}"
    if 'gui_info' in node and 'activity_summary' in node['gui_info']:
        activity_info = node['gui_info']['activity_summary']
        formatted_info += f"\nActivity Name: {activity_info.get('activity_name', 'N/A')}"
        formatted_info += f"\nActivity Summary: {activity_info.get('concise_summary', 'N/A')}"
    if 'subgraph_node_info' in node:
        formatted_info += f"\nNode Summary: {node['subgraph_node_info'].get('node_summary', 'N/A')}\n"
    return formatted_info


def filter_app_identifiers(text: str, app_name: str, app_id: str) -> str:
    replacements = [
        (app_name, "this app"),
        (app_name.lower(), "this app"),
        (app_name.upper(), "this app"),
        (app_id, "this app"),
        (app_id.lower(), "this app"),
        (app_id.upper(), "this app")
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def clean_text(text: str) -> str:
    text = re.sub(r'(\*+|#+|!+|\-+|\_+|>+)', '', text)
    text = text.strip(" \n\t\r:[]()\"'""''")
    return text.strip()


def extract_formatted_section(text: str) -> str:
    markers = [
        r"PART 2: STRUCTURED OUTPUT",
        r"Based on (?:your|the) (?:above )?analysis,? (?:organize|provide|summarize)",
        r"Structured Output:",
        r"FORMAT(?:TED)? OUTPUT:"
    ]
    for marker in markers:
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            return text[match.end():].strip()
    aspect_match = re.search(r'Aspect\s*1:', text, re.IGNORECASE)
    if aspect_match:
        return text[aspect_match.start():].strip()
    return text


def parse_similar_app_info(detail_text: str) -> Dict[str, str]:
    pattern = r'\(adapted from Similar App\s*([^,\)]+)(?:,\s*based on their\s+(\w+))?\)'
    match = re.search(pattern, detail_text, re.IGNORECASE)
    if match:
        app_id = match.group(1).strip() if match.group(1) else None
        inspiration = match.group(2).lower() if match.group(2) else None
        if inspiration not in ['proposition', 'implementation']:
            inspiration = None
        return {
            "from_similar_app": "True",
            "similar_app_id": app_id,
            "inspiration_type": inspiration
        }
    return {
        "from_similar_app": "False",
        "similar_app_id": None,
        "inspiration_type": None
    }


def parse_aspects_detail(text: str) -> List[Dict[str, Any]]:
    try:
        formatted_text = extract_formatted_section(text)
        unified_text = formatted_text.replace('\n', ' ')
        chunks = re.split(r'(?=Aspect\s*\d+[\s:])', unified_text, flags=re.IGNORECASE)
        results = []
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk or not re.search(r'Aspect\s*\d+', chunk, flags=re.IGNORECASE):
                continue
            patterns = [
                r'Aspect\s*\d+[^:]*:\s*(.*?)\s*Detail:\s*(.+?)(?=Aspect\s*\d+|$)',
                r'Aspect\s*\d+\s*(.*?)\s*Detail\s*(.+?)(?=Aspect\s*\d+|$)',
                r'Aspect\s*\d+[^:]*:\s*(.*?)\s*:\s*(.+?)(?=Aspect\s*\d+|$)'
            ]
            matched = False
            for pattern in patterns:
                match = re.search(pattern, chunk, flags=re.IGNORECASE | re.DOTALL)
                if match:
                    aspect_raw = match.group(1)
                    detail_raw = match.group(2)
                    aspect_clean = clean_text(aspect_raw)
                    detail_clean = clean_text(detail_raw)
                    similar_app_info = parse_similar_app_info(detail_clean)
                    if similar_app_info["from_similar_app"] == "True":
                        detail_clean = re.sub(r'\s*\(adapted from.*?\)', '', detail_clean).strip()
                    result = {"Aspect": aspect_clean, "Detail": detail_clean}
                    results.append(result)
                    matched = True
                    break
        return results
    except Exception:
        return []


def get_similar_app_info(top_similar_apps: List[str], pp_category: str, datas_KB: pd.DataFrame, few_shot_num: int) -> List[Dict[str, Any]]:
    similar_apps_info = []
    for app_id in top_similar_apps:
        matching_row = datas_KB[(datas_KB['appId'] == app_id) & (datas_KB['pp_category'] == pp_category)]
        if not matching_row.empty:
            try:
                description = matching_row['icl_description'].iloc[0] if 'icl_description' in matching_row.columns else None
                propositions_raw = matching_row['propositions'].iloc[0]
                if isinstance(propositions_raw, str):
                    propositions_list = json.loads(propositions_raw)
                    propositions = [list(prop.values())[0] for prop in propositions_list if prop and isinstance(prop, dict)]
                else:
                    propositions = []
                if description and isinstance(description, str) and description.strip():
                    similar_apps_info.append({
                        'app_id': app_id,
                        'description': description.strip(),
                        'propositions': propositions
                    })
                    if len(similar_apps_info) >= few_shot_num:
                        break
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
    return similar_apps_info


def format_similar_app_info(similar_apps_info: List[Dict[str, Any]]) -> str:
    formatted_info = ""
    for i, app_info in enumerate(similar_apps_info, 1):
        formatted_info += f"\nSimilar App {i} (ID: {app_info['app_id']}):"
        if app_info['propositions']:
            formatted_info += "\nPrivacy Policy Propositions:"
            for j, prop in enumerate(app_info['propositions'], 1):
                formatted_info += f"\n  {j}. {prop}"
        formatted_info += f"\nImplementation Description:\n{app_info['description']}\n"
        formatted_info += "\n" + "-"*50 + "\n"
    return formatted_info


def des_gen_based_on_subgraph_parallel(
    cg_file: str,
    cg_node_json: str,
    general_subgraphs_w_summary_dict: Dict[str, Any],
    spec_pp_subgraphs_w_summary_dict: Dict[str, Any],
    spec_similarapps_subgraphs_w_summary_dict: Dict[str, Any],
    top_similar_apps: List[str],
    datas_KB: pd.DataFrame,
    data: Dict[str, Any],
    pp_propositions: List[str],
    args: Any
) -> Tuple[str, str, str, str, str]:
    def build_nodes_info(subgraph_dict):
        nodes_info = "=== Subgraph Node Information ===\n"
        activity_nodes, api_nodes, sdk_nodes, context_nodes, general_nodes = [], [], [], [], []
        for node_id, node in subgraph_dict.items():
            node_type = node.get('subgraph_node_info', {}).get('node_type', '')
            entry_str = format_subgraph_node_info(node)
            if node_type == 'activity':
                activity_nodes.append(entry_str)
            elif node_type == 'api_call':
                api_nodes.append(entry_str)
            elif node_type == 'sdk_call':
                sdk_nodes.append(entry_str)
            elif node_type == 'gui_context':
                context_nodes.append(entry_str)
            else:
                general_nodes.append(entry_str)
        if activity_nodes:
            nodes_info += "\n--- Activity Nodes ---\n" + "\n".join(activity_nodes)
        if api_nodes:
            nodes_info += "\n--- API Call Nodes ---\n" + "\n".join(api_nodes)
        if sdk_nodes:
            nodes_info += "\n--- SDK Call Nodes ---\n" + "\n".join(sdk_nodes)
        if context_nodes:
            nodes_info += "\n--- Context-Rich Nodes ---\n" + "\n".join(context_nodes)
        if general_nodes:
            nodes_info += "\n--- General Nodes ---\n" + "\n".join(general_nodes)
        return nodes_info

    branch1_nodes_info = build_nodes_info(general_subgraphs_w_summary_dict)
    branch2_nodes_info = build_nodes_info(spec_pp_subgraphs_w_summary_dict)
    branch3_nodes_info = build_nodes_info(spec_similarapps_subgraphs_w_summary_dict)
    propositions_info = "=== Privacy Policy Propositions ===\n"
    for idx, prop in enumerate(pp_propositions, 1):
        propositions_info += f"{idx}. {prop}\n"
    similar_apps_formatted = ""
    if args.few_shot_num > 0:
        similar_apps_data = get_similar_app_info(top_similar_apps, data['pp_category'], datas_KB, args.few_shot_num)
        if similar_apps_data:
            similar_apps_formatted = format_similar_app_info(similar_apps_data)
    else:
        similar_apps_data = []
    branch1_nodes_info = filter_app_identifiers(branch1_nodes_info, data['app_name'], data['appId'])
    branch2_nodes_info = filter_app_identifiers(branch2_nodes_info, data['app_name'], data['appId'])
    branch3_nodes_info = filter_app_identifiers(branch3_nodes_info, data['app_name'], data['appId'])
    propositions_info = filter_app_identifiers(propositions_info, data['app_name'], data['appId'])
    similar_apps_formatted = filter_app_identifiers(similar_apps_formatted, data['app_name'], data['appId'])

    branch1_prompt = f"""
BRANCH 1: Permission-Focused Exploration

We are focusing on how the app uses the '{data['pp_category']}' permission/data category.

Below is the subgraph node information, showing code references that might be relevant:
{branch1_nodes_info}

TASK:
1) Identify and infer the major functionalities or features that rely on or related to {data['pp_category']} permission/data.
2) For each functionality, try to reference any specific subgraph details (e.g., APIs, components, UI strings, resource usage) 
   that demonstrate its reliance on {data['pp_category']}.
3) Summarize how these features or functionalities affect the user experience and what benefits they might bring.
4) Present your analysis in a clear, comprehensive format—using bullet points or short paragraphs. 
   Try to give each identified feature a concise name or heading followed by a short explanation.

Note: 
- Combine direct evidence from the subgraph (where available) with your broader knowledge of typical app permission usage.
- Clearly distinguish between what is directly evidenced and what is inferred. 
- Aim for completeness while remaining concise.
"""

    branch2_prompt = f"""
BRANCH 2: Policy-Propositions-Based Exploration

The policy-specific subgraph node info:
{branch2_nodes_info}

The official privacy policy propositions related to '{data['pp_category']}':
{propositions_info}

TASK:
1) Cross-check each proposition against the subgraph node info:
   - Determine if the implementation is Verified, Partially Implemented, or Not Found.
   - Identify any over-implemented or under-implemented aspects compared to the policy claims.

2) Note any implicit or undocumented behaviors:
   - Detect dataflow discrepancies not mentioned in the policy.
   - Surface features or usage patterns in the subgraph that are not stated in the policy.

3) Provide a clear textual analysis referencing relevant subgraph evidence (e.g., APIs, method calls, resource usage).
4) Focus on how '{data['pp_category']}' permission and '{data['pp_category']}'-related privacy data underpins or enables each functionality/feature/scenario:
   - If a proposition is unsupported by the subgraph, explain the gap or uncertainty.
   - If extra functionality beyond the policy claims emerges, highlight it.

Formatting & Style:
- Aim for short paragraphs or bullet points.
- Clearly connect each proposition to the subgraph findings.
"""

    if similar_apps_formatted.strip():
        branch3_prompt = f"""
BRANCH 3: Similar-App-Based Exploration

Similar-app-specific subgraph node info:
{branch3_nodes_info}

We also have info on how other (similar) apps implement '{data['pp_category']}':
{similar_apps_formatted}

TASK:
1) Feature Hypothesis Generation:
   - Identify which permission-related features from these similar apps might also appear in the current app.

2) Evidence Validation:
   - For each hypothesized feature, check the subgraph for supporting or contradicting evidence (e.g., specific API/SDK calls, data flows, or UI traces).
   - Conclude whether each feature is Confirmed (clearly implemented), Potential (partially evidenced), or Ruled Out (contradicted by the subgraph).

3) Gap Identification:
   - Flag any notable features present in similar apps but absent here.
   - Note any unique or novel functionality the current app shows that doesn't appear in those peers.

4) Summarize:
   - Provide a concise textual analysis referencing specific subgraph details, if possible.
   - Use short paragraphs or bullet points. Focus on how '{data['pp_category']}' permission/data is leveraged, or might be leveraged, based on these comparisons.
"""
    else:
        branch3_prompt = f"""
BRANCH 3: Similar-App-Based Exploration

Similar-app-specific subgraph node info:
{branch3_nodes_info}

No direct similar-app data is provided. However, consider potential or novel functionalities that might not be directly obvious from standard usage or policy texts.

TASK:
- Inspect the subgraph for any hidden or advanced usage of {data['pp_category']}.
- Provide a reasoned speculation if something stands out.
"""

    branch1_result = run_llm(branch1_prompt, args.temperature_exploration, args, args.max_tokens, args.LLM_type_generate, sys_msg="normal")
    branch2_result = run_llm(branch2_prompt, args.temperature_exploration, args, args.max_tokens, args.LLM_type_generate, sys_msg="normal")
    branch3_result = run_llm(branch3_prompt, args.temperature_exploration, args, args.max_tokens, args.LLM_type_generate, sys_msg="normal")

    aggregator_prompt_round1 = f"""
AGGREGATOR - Round 1: Comprehensive Analysis Integration

We have three independent branches of analysis examining how this app handles '{data['pp_category']}'.
Each branch relies on different perspectives or data sources:

[BRANCH 1: Permission-Focused Usage]
Data Source & Method:
- Primarily based on general subgraph node info
- Emphasis on how '{data['pp_category']}' enables specific functionalities/features
Analysis Output:
{branch1_result}

[BRANCH 2: Policy-Propositions Correlation]
Data Source & Method:
- Uses official privacy policy propositions with policy-specific subgraph
- Cross-checks each proposition against subgraph evidence
Analysis Output:
{branch2_result}

[BRANCH 3: Similar-App or Novel Discovery]
Data Source & Method:
- Uses similar-app-specific subgraph
- Compares to known implementations from similar apps
- Identifies potential features that might be confirmed, missing, or novel
Analysis Output:
{branch3_result}

TASK:
1) Summarize each branch's approach or reasoning focus, 
   and restate their key findings in your own words.
2) Identify the main points of agreement or consistent conclusions across branches.
3) Highlight any conflicts, discrepancies, or unresolved questions. 
   Suggest possible reasons or ways to reconcile them if applicable.
4) Provide a unified, integrated perspective that merges all findings into 
   a cohesive overall analysis of '{data['pp_category']}' usage in this app.
   (If certain differences remain unresolved, state them explicitly.)

OUTPUT:
A single integrated analysis covering tasks (1)-(4) above.
This analysis is for internal reconciliation only. 
We will generate the final, user-facing description in a subsequent stage.
Please present your summary in a structured, concise manner (e.g., short paragraphs or bullet points).
"""

    aggregator_prompt_round2 = f"""
FINAL STAGE: Authoritative Feature Profile

We have integrated the findings from all previous analyses (Branches 1–3 and the Round 1 unification). 
Now produce a definitive, authoritative summary of how this app uses the '{data['pp_category']}' permission.

Integration Requirements: Feature Clarity and Compliance
   - Distinguish between core and auxiliary functionalities
   - Maintain precise descriptions of each relevant business scenario
   - Emphasize how these functionalities specifically rely on the '{data['pp_category']}' permission
   - Briefly note any compliance transparency and potential implementation risks

Task:
- Present a final app permission-related description that is highly factual and direct.
- Minimize non-substantial or purely descriptive language. Instead, focus on:
  (a) The actual features requiring '{data['pp_category']}'
  (b) The key business scenarios and user impacts
  (c) Concise notes on compliance and risk aspects
- Do NOT directly reference the internal aggregator or any behind-the-scenes analysis.
- Write in short paragraphs or bullet points to ensure clarity. Avoid lengthy adjectives or speculative claims.

Output:
A single cohesive text that thoroughly covers each feature, highlighting its functional purpose, 
how '{data['pp_category']}' is utilized, and any pertinent compliance/risk considerations.
"""

    aggregator_prompts = [aggregator_prompt_round1, aggregator_prompt_round2]
    aggregator_responses = run_llm_seq(
        aggregator_prompts,
        args.temperature_exploration,
        args,
        args.max_tokens,
        args.LLM_type_generate,
        sys_msg="normal"
    )
    aggregator_round1_text = aggregator_responses[0]
    aggregator_round2_text = aggregator_responses[1]

    prompts_dir = "./des_gen_prompts"
    os.makedirs(prompts_dir, exist_ok=True)
    prompts_filename = f"{data['appId']}_{data['pp_category']}.json"
    prompts_filepath = os.path.join(prompts_dir, prompts_filename)
    prompts_dict = {
        "branch1_prompt": branch1_prompt,
        "branch2_prompt": branch2_prompt,
        "branch3_prompt": branch3_prompt,
        "aggregator_prompt_template_round1": aggregator_prompt_round1,
        "aggregator_prompt_round2": aggregator_prompt_round2
    }
    try:
        with open(prompts_filepath, 'w', encoding='utf-8') as f:
            json.dump(prompts_dict, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    return (
        branch1_result,
        branch2_result,
        branch3_result,
        aggregator_round1_text,
        aggregator_round2_text
)
