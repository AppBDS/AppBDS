import json
import time
import re
import os
import requests
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple
from collections import defaultdict

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
        except Exception:
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
    except Exception:
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
                import json
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


def preprocess_texts(texts):
    processed_texts = []
    for text in texts:
        if isinstance(text, str):
            text = text.strip()
            if text:
                processed_texts.append(text)
    return processed_texts


def calculate_embeddings(texts, openai_api_key, method='gpt', batch_sizes=[50, 10, 5, 1]):
    def get_openai_embeddings(text_list, client):
        try:
            response = client.embeddings.create(input=text_list, model="text-embedding-3-small")
            return [data.embedding for data in response.data]
        except Exception as e:
            return None

    def preprocess_texts(texts):
        return texts

    if method == 'gpt':
        import openai
        client = openai.OpenAI(api_key=openai_api_key)
        texts = preprocess_texts(texts)
        if not texts:
            raise ValueError("The input list contains no valid texts.")
        embeddings = {}
        i = 0
        while i < len(texts):
            progress = False
            for batch_size in batch_sizes:
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = get_openai_embeddings(batch_texts, client)
                if batch_embeddings is not None:
                    embeddings.update({text: embedding for text, embedding in zip(batch_texts, batch_embeddings)})
                    i += batch_size
                    progress = True
                    break
                else:
                    if batch_size == batch_sizes[-1]:
                        error_placeholder = "NaN"
                        placeholder_embedding = get_openai_embeddings([error_placeholder] * len(batch_texts), client)
                        if placeholder_embedding is not None:
                            embeddings.update({text: placeholder_embedding[0] for text in batch_texts})
                        else:
                            embeddings.update({text: None for text in batch_texts})
                        i += 1
    return embeddings


def calculate_similarity(embedding1, embedding2):
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


def calculate_similarity_matrix(embeddings1, embeddings2):
    if embeddings1.shape[1] != embeddings2.shape[1]:
        raise ValueError("Embedding dimensions do not match")
    norms1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
    norms2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
    norms1[norms1 == 0] = 1
    norms2[norms2 == 0] = 1
    normalized_embeddings1 = embeddings1 / norms1
    normalized_embeddings2 = embeddings2 / norms2
    similarity_matrix = np.dot(normalized_embeddings1, normalized_embeddings2.T)
    return similarity_matrix


def batch_cosine_similarity(vectors1: np.ndarray, vectors2: np.ndarray) -> np.ndarray:
    epsilon = 1e-8
    norms1 = np.linalg.norm(vectors1, axis=1, keepdims=True)
    norms2 = np.linalg.norm(vectors2, axis=1, keepdims=True)
    normalized1 = np.divide(vectors1, norms1, out=np.zeros_like(vectors1), where=norms1>epsilon)
    normalized2 = np.divide(vectors2, norms2, out=np.zeros_like(vectors2), where=norms2>epsilon)
    return np.dot(normalized1, normalized2.T)


def select_important_nodes(nodes_data: List[dict], activities_dict: Dict[str, List[str]]) -> Dict[str, str]:
    node_map = {node['id']: node for node in nodes_data}
    important_nodes = {}
    for activity_name, node_ids in activities_dict.items():
        onCreate_node = None
        max_length_node = None
        max_length = 0
        for node_id in node_ids:
            if node_id not in node_map:
                continue
            node = node_map[node_id]
            signature = node.get('signature', '')
            body = node.get('body', '')
            if 'onCreate' in signature:
                onCreate_node = node_id
                break
            match = re.search(r'Method body length: (\d+) lines', body)
            body_length = int(match.group(1)) if match else 0
            if body_length > max_length:
                max_length = body_length
                max_length_node = node_id
        important_nodes[activity_name] = onCreate_node if onCreate_node else max_length_node
    return important_nodes


def check_activity_privacy_relevance(activity_info: Dict, privacy_category: str, openai_api_keys: str, args) -> bool:
    activity_name = activity_info.get('gui_info', {}).get('activity_summary', {}).get('activity_name', 'N/A')
    detailed_summary = activity_info.get('gui_info', {}).get('activity_summary', {}).get('detailed_summary', 'N/A')
    concise_summary = activity_info.get('gui_info', {}).get('activity_summary', {}).get('concise_summary', 'N/A')
    prompt = f"""Analyze this Android app activity and identify ANY possible connection to {privacy_category} privacy/permission, including potential, indirect, or contextual relationships.

Activity Information:
1. Activity Name: {activity_name}
2. Detailed Functionality Description: {detailed_summary}
3. Concise Summary: {concise_summary}

Consider ALL possible connections to {privacy_category}:

1. Direct Functionality:
   - Explicit handling or processing of {privacy_category} data
   - Features requiring {privacy_category} permissions
   - UI elements displaying {privacy_category}-related information

2. Supporting or Related Features:
   - Pre- or post-processing of {privacy_category} data
   - Setup or configuration for {privacy_category} features
   - Error handling or status display related to {privacy_category}
   - Any UI elements that might involve {privacy_category} data

3. Indirect or Potential Relations:
   - Part of a feature flow that might involve {privacy_category}
   - Supporting activities for {privacy_category}-related functions
   - Activities that might enhance or supplement {privacy_category} features
   - Any connection to components typically associated with {privacy_category}

4. Contextual Connections:
   - Activities in modules or sections that deal with {privacy_category}
   - Navigation or transition activities around {privacy_category} features
   - Settings or preferences that might affect {privacy_category} handling
   - Any functionality that could potentially use or benefit from {privacy_category} data

5. Future or Conditional Usage:
   - Placeholder or preparatory functionality for {privacy_category} features
   - Conditional features that might involve {privacy_category} in certain cases
   - Extensible components that could incorporate {privacy_category} functionality

Respond with 'RELEVANT' if you can identify ANY possible connection or potential relationship to {privacy_category}, no matter how indirect or speculative.
Only respond with 'NOT_RELEVANT' if the activity is completely unrelated and could not possibly have any connection to {privacy_category} under any circumstance."""
    response = run_llm(prompt, 0.0, args, engine=args.LLM_type_generate, sys_msg="normal")
    if not response:
        return False
    response = response.strip().upper()
    return "NOT_RELEVANT" not in response


def filter_privacy_related_activities(important_activity_nodes: Dict[str, str], 
                                      nodes_data: List[dict],
                                      privacy_category: str,
                                      openai_api_keys: str, args) -> Dict[str, str]:
    node_map = {node['id']: node for node in nodes_data}
    filtered_activities = {}
    total = len(important_activity_nodes)
    for idx, (activity_name, node_id) in enumerate(important_activity_nodes.items(), 1):
        if node_id not in node_map:
            continue
        node_info = node_map[node_id]
        if check_activity_privacy_relevance(node_info, privacy_category, openai_api_keys, args):
            filtered_activities[activity_name] = node_id
    return filtered_activities


def check_privacy_relevance_with_llm(node_info: Dict, privacy_category: str, openai_api_keys: str, args) -> bool:
    prompt = f"""Analyze this Android app node information and identify ANY possible connection to {privacy_category} privacy/permission/functionality, no matter how indirect or speculative the connection might be.

Node Information:
{node_info}

Consider ANY of these possibilities:
1. Current or Potential Usage:
   - Any code that currently uses or might use {privacy_category}
   - Functions that could potentially be extended to use {privacy_category}
   - Components that might interact with {privacy_category}-related features

2. Indirect or Supporting Role:
   - Helper functions or utility code that could support {privacy_category} features
   - Code that processes any data that might be related to {privacy_category}
   - Any UI elements that could display {privacy_category}-related information

3. Contextual or Future Possibilities:
   - Code that belongs to modules/packages that might handle {privacy_category}
   - Functions that could be part of a larger {privacy_category} workflow
   - Any potential future extensions that might involve {privacy_category}
   - Code that could be modified to handle {privacy_category} data

4. General Connection:
   - Any similarity in naming or structure to {privacy_category}-related code
   - Code that handles any type of data that could be derived from {privacy_category}
   - Any connection to general app functionality where {privacy_category} might be relevant

Respond with 'RELEVANT' if you can imagine ANY possible connection or future potential, no matter how indirect or speculative.
Only respond with 'NOT_RELEVANT' if the code is completely and absolutely unrelated to {privacy_category} in any way.

Respond only with 'RELEVANT' or 'NOT_RELEVANT'."""
    response = run_llm(prompt, 0.0, args, engine=args.LLM_type_generate, sys_msg="normal")
    if not response:
        return False
    response = response.strip().upper()
    return "NOT_RELEVANT" not in response


def extract_privacy_nodes(nodes_data: List[dict], topic_calls: Dict[str, List[str]], 
                          privacy_category: str, node_embeddings: Dict[str, np.ndarray], data, 
                          args: Any, is_api: bool = True) -> List[str]:
    directly_matched_nodes = []
    call_types = "API calls" if is_api else "SDK interfaces"
    target_calls = topic_calls.get(privacy_category, [])
    node_dict = {node['id']: node.get('signature', '').lower() for node in nodes_data}
    for node_id, signature in node_dict.items():
        for call in target_calls:
            if call.lower() in signature:
                directly_matched_nodes.append(node_id)
                break
    if len(directly_matched_nodes) >= args.cg_nodes:
        return directly_matched_nodes[:args.cg_nodes]
    target_call_embeddings = calculate_embeddings(target_calls, args.openai_api_keys)
    valid_calls = []
    valid_embeddings = []
    for call in target_calls:
        if call in target_call_embeddings and target_call_embeddings[call] is not None:
            valid_calls.append(call)
            valid_embeddings.append(target_call_embeddings[call])
    if not valid_embeddings:
        return directly_matched_nodes[:args.cg_nodes]
    node_ids = list(node_embeddings.keys())
    node_emb_array = np.array([node_embeddings[node_id] for node_id in node_ids])
    call_emb_array = np.array(valid_embeddings)
    emb_dim = node_emb_array.shape[1]
    if call_emb_array.shape[1] != emb_dim:
        return directly_matched_nodes[:args.cg_nodes]
    similarity_matrix = batch_cosine_similarity(node_emb_array, call_emb_array)
    all_candidates = []
    for i, call in enumerate(valid_calls):
        call_similarities = [(node_ids[j], similarity_matrix[j, i]) for j in range(len(node_ids))]
        sorted_similarities = sorted(call_similarities, key=lambda x: x[1], reverse=True)
        all_candidates.extend(sorted_similarities[:args.cg_nodes])
    sorted_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)
    unique_candidates = []
    seen_nodes = set(directly_matched_nodes)
    for node_id in directly_matched_nodes:
        unique_candidates.append((node_id, 1.0))
    for node_id, similarity in sorted_candidates:
        if node_id not in seen_nodes:
            unique_candidates.append((node_id, similarity))
            seen_nodes.add(node_id)
            if len(unique_candidates) >= args.cg_nodes:
                break
    matched_nodes = directly_matched_nodes.copy()
    nodes_map = {node['id']: node for node in nodes_data if 'id' in node}
    for node_id, _ in unique_candidates[len(directly_matched_nodes):]:
        node_info = nodes_map.get(node_id)
        if node_info and check_privacy_relevance_with_llm(node_info, privacy_category, args.openai_api_keys, args):
            matched_nodes.append(node_id)
    return matched_nodes[:args.cg_nodes]


def get_or_calculate_node_embeddings(nodes_data: List[dict], data, args: Any) -> Dict[str, np.ndarray]:
    signature_to_id = {node['signature']: node['id'] for node in nodes_data if 'signature' in node}
    try:
        with open(os.path.join(args.embedding_json_path, f"{data['appId']}_embeddings.json"), 'r') as f:
            embeddings_data = json.load(f)
        signatures_embeddings = embeddings_data['signatures']
        node_embeddings = {}
        for signature, embedding in signatures_embeddings.items():
            if '<' + signature + '>' in signature_to_id:
                node_embeddings[signature_to_id['<' + signature + '>']] = np.array(embedding)
        if len(node_embeddings) == 0:
            import pdb; pdb.set_trace()
        return node_embeddings
    except:
        pass
    if not nodes_data:
        return {}
    node_texts = {}
    signature_texts = {}
    for node in nodes_data:
        if 'signature' not in node:
            continue
        node_text = f"Signature: {node['signature']}\nBody: {node.get('body', '')}"
        node_texts[node['id']] = node_text
        signature_texts[node['signature']] = node_text
    embeddings = {}
    batch_size = 10
    node_ids = list(node_texts.keys())
    i = 0
    while i < len(node_ids):
        batch_ids = node_ids[i:i + batch_size]
        batch_texts = [node_texts[nid] for nid in batch_ids]
        try:
            batch_embeddings = calculate_embeddings(batch_texts, args.openai_api_keys)
            for nid, emb_text in zip(batch_ids, batch_texts):
                embeddings[nid] = batch_embeddings.get(emb_text, None)
            i += batch_size
        except:
            i += 1
    output_data = {
        "contexts": {},
        "signatures": {}
    }
    id_to_signature = {v: k for k, v in signature_to_id.items()}
    for nid, emb in embeddings.items():
        if nid in id_to_signature and emb is not None:
            signature_key = id_to_signature[nid]
            if signature_key.startswith('<') and signature_key.endswith('>'):
                signature_key = signature_key[1:-1]
            output_data["signatures"][signature_key] = emb.tolist() if isinstance(emb, np.ndarray) else emb
    output_dir = "/mnt/disk1/zichen/DescribeCTX/AppBDS/CCS_code/data/embeddings_json_small"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{data['appId']}.json")
    try:
        with open(output_path, 'w') as f:
            json.dump(output_data, f)
    except:
        pass
    return {nid: np.array(emb) if isinstance(emb, list) else np.array([]) for nid, emb in embeddings.items() if emb is not None}


def extract_context_rich_nodes(nodes_data: List[dict], top_n: int = 10) -> List[str]:
    node_scores = {}
    for node in nodes_data:
        if 'cg_info' not in node:
            continue
        resources = node['cg_info'].get('resources', {})
        score = 0
        layouts = resources.get('layouts', '')
        if isinstance(layouts, dict):
            layout_count = 0
            for value in layouts.values():
                if isinstance(value, str) and value.strip():
                    layout_count += 1
                elif isinstance(value, (list, dict)):
                    layout_count += 1
            score += layout_count * 2
        strings = resources.get('strings', {})
        if isinstance(strings, dict):
            str_count = 0
            for value in strings.values():
                if isinstance(value, str) and value.strip():
                    str_count += 1
                elif isinstance(value, (list, dict)):
                    str_count += len(value)
            score += str_count
        if score > 0:
            node_scores[node['id']] = score
    sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
    top_nodes = [node_id for node_id, _ in sorted_nodes[:top_n]]
    return top_nodes


def general_cg_entity_search(cg_file, cg_node_json, topic_api_calls, topic_sdk_interfaces, data, args):
    try:
        with open(cg_node_json, 'r', encoding='utf-8') as f:
            nodes_data = json.load(f)
    except:
        return {}, [], [], []
    activities_dict = defaultdict(list)
    for node in nodes_data:
        if 'gui_info' not in node:
            continue
        gui_info = node['gui_info']
        if gui_info.get('is_activity') in [True, "true"]:
            activity_summary = gui_info.get('activity_summary', {})
            activity_name = activity_summary.get('activity_name')
            if activity_name:
                activities_dict[activity_name].append(node['id'])
    activities_dict = dict(activities_dict)
    important_activity_nodes = select_important_nodes(nodes_data, activities_dict)
    relevant_activity_nodes = filter_privacy_related_activities(
        important_activity_nodes,
        nodes_data,
        data['pp_category'],
        args.openai_api_keys,
        args
    )
    node_embeddings = get_or_calculate_node_embeddings(nodes_data, data, args)
    api_call_node_ids = extract_privacy_nodes(
        nodes_data, topic_api_calls, data['pp_category'],
        node_embeddings, data, args, is_api=True
    )
    sdk_call_node_ids = extract_privacy_nodes(
        nodes_data, topic_sdk_interfaces, data['pp_category'],
        node_embeddings, data, args, is_api=False
    )
    gui_context_node_ids = extract_context_rich_nodes(nodes_data, args.top_context_nodes)
    all_kinds_node_ids = set()
    all_kinds_node_ids.update(relevant_activity_nodes.values())
    all_kinds_node_ids.update(api_call_node_ids)
    all_kinds_node_ids.update(sdk_call_node_ids)
    all_kinds_node_ids.update(gui_context_node_ids)
    relevant_activity_node_ids = list(relevant_activity_nodes.values())
    all_kinds_node_ids = list(all_kinds_node_ids)
    return relevant_activity_node_ids, list(api_call_node_ids), list(sdk_call_node_ids), list(gui_context_node_ids), all_kinds_node_ids


def check_proposition_relevance_with_llm(node_info: Dict, proposition: str, pp_category: str, openai_api_keys: str, args) -> bool:
    prompt = f"""Analyze the following Android app node information and determine whether it is RELEVANT to the following proposition:

Proposition:
"{proposition}"

Node Information:
{node_info}

DEFINITION OF RELEVANCE:
A node is considered RELEVANT if it meets ANY of the following criteria:
1. DIRECT EVIDENCE: The node provides direct evidence supporting or contradicting the proposition
2. EXPLANATORY VALUE: The node helps explain concepts mentioned in the proposition
3. FUNCTIONAL RELEVANCE: The node implements functionality described in or related to the proposition
4. DATA RELEVANCE: The node handles data types or categories referenced in the proposition
5. CONTEXTUAL ENHANCEMENT: The node provides additional context that enriches understanding of the proposition
6. EXTENSION CONNECTION: The node contains elements that build upon or extend the proposition's concepts
7. INDIRECT UTILITY: The node contains information that could indirectly contribute to analyzing the proposition
8. TECHNICAL IMPLEMENTATION: The node shows how concepts in the proposition might be implemented 
9. LATENT RELATIONSHIP: The node has a non-obvious but meaningful connection to the proposition
10. PERMISSION UTILIZATION: The node might relate to how the app uses {pp_category} permissions
11. FEATURE DEPENDENCY: The node supports app features that may depend on {pp_category} permissions

EVALUATION GUIDELINES:
- Consider both surface-level and deeper connections
- Explore how the node might fit into a broader system related to the proposition
- Identify ways the node might provide additional dimensions or perspectives
- Assess if the node reveals potential implications or applications of the proposition
- Look for connections that might not be immediately obvious but emerge upon deeper analysis
- Examine whether the node might support functionality that would require {pp_category} permissions
- Consider the various ways an app might leverage {pp_category} permissions, and whether this node could be part of those implementation paths

Respond with 'RELEVANT' if you determine ANY possible connection, no matter how speculative or partial.
Only respond with 'NOT_RELEVANT' if there is NO connection whatsoever."""
    response = run_llm(prompt, 0.3, args, engine=args.LLM_type_generate, sys_msg="normal")
    if not response:
        return False
    response = response.strip().upper()
    return "NOT_RELEVANT" not in response


def spec_pp_cg_entity_search(cg_file, cg_node_json, pp_propositions, data, args):
    try:
        with open(cg_node_json, 'r', encoding='utf-8') as f:
            nodes_data = json.load(f)
    except:
        return {}
    node_embeddings = get_or_calculate_node_embeddings(nodes_data, data, args)
    nodes_map = {node['id']: node for node in nodes_data if 'id' in node}
    proposition_embeddings = calculate_embeddings(pp_propositions, args.openai_api_keys)
    proposition_top_nodes = {}
    node_ids = list(node_embeddings.keys())
    node_emb_array = np.array([node_embeddings[node_id] for node_id in node_ids])
    for prop in pp_propositions:
        if prop not in proposition_embeddings or proposition_embeddings[prop] is None:
            proposition_top_nodes[prop] = []
            continue
        prop_embedding = np.array(proposition_embeddings[prop]).reshape(1, -1)
        if prop_embedding.shape[1] != node_emb_array.shape[1]:
            proposition_top_nodes[prop] = []
            continue
        similarities = batch_cosine_similarity(node_emb_array, prop_embedding).flatten()
        node_similarities = list(zip(node_ids, similarities))
        candidates = sorted(node_similarities, key=lambda x: x[1], reverse=True)
        top_candidates = [node_id for node_id, _ in candidates[:args.cg_nodes]]
        filtered_candidates = []
        for node_id in top_candidates:
            node_info = nodes_map.get(node_id)
            if node_info and check_proposition_relevance_with_llm(node_info, prop, data['pp_category'], args.openai_api_keys, args):
                filtered_candidates.append(node_id)
        proposition_top_nodes[prop] = filtered_candidates
    return proposition_top_nodes


def check_similarapp_proposition_relevance_with_llm(node_info: Dict, proposition: str, app_id: str, description: str, pp_category: str, openai_api_keys: str, args) -> bool:
    prompt = f"""Analyze the following Android app node information and determine whether it is RELEVANT to the given proposition.

App ID: {app_id}
App Description: {description}

Proposition:
"{proposition}"

Node Information:
{node_info}

DEFINITION OF RELEVANCE:
A node is considered RELEVANT if it meets ANY of the following criteria:
1. DIRECT CONNECTION: The node explicitly mentions concepts, data, or functionality related to the proposition
2. FUNCTIONAL RELATIONSHIP: The node implements functionality that aligns with what the proposition describes
3. DATA FLOW: The node processes, transmits, or stores data types mentioned in the proposition
4. CONTEXTUAL SUPPORT: The node provides context that helps understand the proposition better
5. EXTENSION POTENTIAL: The node contains elements that could extend or elaborate on the proposition
6. INFRASTRUCTURAL RELEVANCE: The node provides technical infrastructure supporting proposition-related features
7. IMPLICIT CONNECTION: The node has subtle connections that might not be immediately obvious
8. PARTIAL RELEVANCE: The node addresses at least one aspect of the proposition
9. PERMISSION RELATIONSHIP: The node relates to how the app might use {pp_category} permissions
10. FUNCTIONALITY ENABLEMENT: The node supports app features that could potentially utilize {pp_category} permissions

EVALUATION INSTRUCTIONS:
- Look for both explicit and implicit connections
- Consider how the node might interact with other components in the app
- Evaluate if the node contains information that explores additional dimensions of the proposition
- Assess if the node reveals potential implementations or mechanisms related to the proposition
- Determine if the node could provide complementary or contradictory evidence to the proposition
- Analyze whether the node might be related to app functionality that would require {pp_category} permissions
- Consider how the app might use {pp_category} permissions across different features, and if this node supports any of those features

Respond with 'RELEVANT' if you determine ANY possible connection, no matter how speculative or partial.
Only respond with 'NOT_RELEVANT' if there is NO connection whatsoever."""
    response = run_llm(prompt, 0.3, args, engine=args.LLM_type_generate, sys_msg="normal")
    if not response:
        return False
    response = response.strip().upper()
    return "NOT_RELEVANT" not in response


def get_similar_app_info(top_similar_apps: List[str], pp_category: str, datas_KB, few_shot_num: int) -> List[Dict[str, Any]]:
    similar_apps_info = []
    for app_id in top_similar_apps:
        matching_row = datas_KB[(datas_KB['appId'] == app_id) & (datas_KB['pp_category'] == pp_category)]
        if not matching_row.empty:
            try:
                description = matching_row['icl_description'].iloc[0] if 'icl_description' in matching_row.columns else None
                propositions_raw = matching_row['propositions'].iloc[0]
                if isinstance(propositions_raw, str):
                    import json
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
            except:
                continue
    return similar_apps_info


def spec_similarapps_cg_entity_search(cg_file, cg_node_json, top_similar_apps, datas_KB, data, args):
    try:
        with open(cg_node_json, 'r', encoding='utf-8') as f:
            nodes_data = json.load(f)
    except:
        return {}
    nodes_map = {node['id']: node for node in nodes_data if 'id' in node}
    similar_apps_info = get_similar_app_info(top_similar_apps, data['pp_category'], datas_KB, args.few_shot_num)
    app_info_dict = {app_info['app_id']: app_info for app_info in similar_apps_info}
    app_propositions_dict = {app_id: app_info['propositions'] for app_id, app_info in app_info_dict.items()}
    node_embeddings = get_or_calculate_node_embeddings(nodes_data, data, args)
    node_ids = list(node_embeddings.keys())
    node_emb_array = np.array([node_embeddings[node_id] for node_id in node_ids])
    all_propositions = [proposition for propositions in app_propositions_dict.values() for proposition in propositions]
    proposition_embeddings = calculate_embeddings(all_propositions, args.openai_api_keys)
    app_top_nodes = {}
    for app_id, propositions in app_propositions_dict.items():
        app_top_nodes[app_id] = {}
        app_description = app_info_dict[app_id].get('description', '')
        for prop in propositions:
            if prop not in proposition_embeddings or proposition_embeddings[prop] is None:
                app_top_nodes[app_id][prop] = []
                continue
            prop_embedding = np.array(proposition_embeddings[prop]).reshape(1, -1)
            if prop_embedding.shape[1] != node_emb_array.shape[1]:
                app_top_nodes[app_id][prop] = []
                continue
            similarities = batch_cosine_similarity(node_emb_array, prop_embedding).flatten()
            node_similarities = list(zip(node_ids, similarities))
            candidates = sorted(node_similarities, key=lambda x: x[1], reverse=True)
            top_candidates = [node_id for node_id, _ in candidates[:args.prop_cg_nodes]]
            filtered_candidates = []
            for node_id in top_candidates:
                node_info = nodes_map.get(node_id)
                if node_info and check_similarapp_proposition_relevance_with_llm(
                    node_info, 
                    prop, 
                    app_id, 
                    app_description,
                    data['pp_category'], 
                    args.openai_api_keys, 
                    args
                ):
                    filtered_candidates.append(node_id)
            app_top_nodes[app_id][prop] = filtered_candidates
    return app_top_nodes


def general_cg_subgraph_find(MG_cg_dot: nx.MultiDiGraph, 
                             cg_node_json: str,
                             activity_node_ids: List[str], 
                             api_call_node_ids: List[str],
                             sdk_call_node_ids: List[str],
                             gui_context_node_ids: List[str],
                             all_kinds_node_ids: List[str],
                             data: Dict,
                             args: Any) -> Tuple[nx.MultiDiGraph, Dict]:
    dot_filename = f"{data['appId']}_{data['pp_category']}.dot"
    json_filename = f"{data['appId']}_{data['pp_category']}.json"
    dot_path = os.path.join(args.general_subgraph_dot_path, dot_filename)
    json_path = os.path.join(args.general_subgraph_dot_json_path, json_filename)
    all_relevant_nodes = set(all_kinds_node_ids)
    existing_nodes = set(MG_cg_dot.nodes())
    valid_nodes = all_relevant_nodes.intersection(existing_nodes)
    if not valid_nodes:
        return nx.MultiDiGraph(), {}
    connected_nodes = set()
    paths = []
    node_list = list(valid_nodes)
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            source = node_list[i]
            target = node_list[j]
            try:
                if nx.has_path(MG_cg_dot, source, target):
                    path = nx.shortest_path(MG_cg_dot, source, target)
                    paths.extend(path)
                    connected_nodes.update(path)
            except:
                continue
    connected_nodes.update(valid_nodes)
    subgraph = MG_cg_dot.subgraph(connected_nodes).copy()
    subgraph_dict = {}
    for node in subgraph.nodes():
        node_data = MG_cg_dot.nodes[node]
        subgraph_dict[node] = {
            'label': node_data.get('label', ''),
            'attributes': node_data
        }
        if node in activity_node_ids:
            subgraph_dict[node]['type'] = 'activity'
        elif node in api_call_node_ids:
            subgraph_dict[node]['type'] = 'api_call'
        elif node in sdk_call_node_ids:
            subgraph_dict[node]['type'] = 'sdk_call'
        elif node in gui_context_node_ids:
            subgraph_dict[node]['type'] = 'gui_context'
        else:
            subgraph_dict[node]['type'] = 'general'
    try:
        os.makedirs(args.general_subgraph_dot_path, exist_ok=True)
        os.makedirs(args.general_subgraph_dot_json_path, exist_ok=True)
        nx.drawing.nx_pydot.write_dot(subgraph, dot_path)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(subgraph_dict, f, indent=2, ensure_ascii=False)
    except:
        pass
    return subgraph, subgraph_dict


def spec_pp_cg_subgraph_find(MG_cg_dot: nx.MultiDiGraph, 
                             cg_node_json: str,
                             relevant_pp_propositions_node_ids_dict: Dict[str, List[str]], 
                             activity_node_ids: List[str], 
                             api_call_node_ids: List[str],
                             sdk_call_node_ids: List[str],
                             gui_context_node_ids: List[str],
                             all_kinds_node_ids: List[str],
                             data: Dict,
                             args: Any) -> Tuple[nx.MultiDiGraph, Dict]:
    dot_filename = f"{data['appId']}_{data['pp_category']}_pp.dot"
    json_filename = f"{data['appId']}_{data['pp_category']}_pp.json"
    dot_path = os.path.join(args.spec_pp_subgraph_dot_path, dot_filename)
    json_path = os.path.join(args.spec_pp_subgraph_dot_json_path, json_filename)
    pp_nodes = set(node for nodes in relevant_pp_propositions_node_ids_dict.values() for node in nodes)
    all_relevant_nodes = set(all_kinds_node_ids).union(pp_nodes)
    existing_nodes = set(MG_cg_dot.nodes())
    valid_nodes = all_relevant_nodes.intersection(existing_nodes)
    if not valid_nodes:
        return nx.MultiDiGraph(), {}
    connected_nodes = set()
    node_list = list(valid_nodes)
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            source, target = node_list[i], node_list[j]
            try:
                if nx.has_path(MG_cg_dot, source, target):
                    path = nx.shortest_path(MG_cg_dot, source, target)
                    connected_nodes.update(path)
            except:
                continue
    connected_nodes.update(valid_nodes)
    subgraph = MG_cg_dot.subgraph(connected_nodes).copy()
    subgraph_dict = {}
    for node in subgraph.nodes():
        node_data = MG_cg_dot.nodes[node]
        subgraph_dict[node] = {
            'label': node_data.get('label', ''),
            'attributes': node_data
        }
        if node in activity_node_ids:
            subgraph_dict[node]['type'] = 'activity'
        elif node in api_call_node_ids:
            subgraph_dict[node]['type'] = 'api_call'
        elif node in sdk_call_node_ids:
            subgraph_dict[node]['type'] = 'sdk_call'
        elif node in gui_context_node_ids:
            subgraph_dict[node]['type'] = 'gui_context'
        elif node in pp_nodes:
            subgraph_dict[node]['type'] = 'pp_proposition'
        else:
            subgraph_dict[node]['type'] = 'general'
    try:
        os.makedirs(args.spec_pp_subgraph_dot_path, exist_ok=True)
        os.makedirs(args.spec_pp_subgraph_dot_json_path, exist_ok=True)
        nx.drawing.nx_pydot.write_dot(subgraph, dot_path)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(subgraph_dict, f, indent=2, ensure_ascii=False)
    except:
        pass
    return subgraph, subgraph_dict


def spec_similarapps_cg_subgraph_find(MG_cg_dot: nx.MultiDiGraph, 
                                      cg_node_json: str,
                                      relevant_similarapps_node_ids_dict: Dict[str, Dict[str, List[str]]], 
                                      activity_node_ids: List[str], 
                                      api_call_node_ids: List[str],
                                      sdk_call_node_ids: List[str],
                                      gui_context_node_ids: List[str],
                                      all_kinds_node_ids: List[str],
                                      data: Dict,
                                      args: Any) -> Tuple[nx.MultiDiGraph, Dict]:
    dot_filename = f"{data['appId']}_{data['pp_category']}_similarapps.dot"
    json_filename = f"{data['appId']}_{data['pp_category']}_similarapps.json"
    dot_path = os.path.join(args.spec_similarapps_subgraph_dot_path, dot_filename)
    json_path = os.path.join(args.spec_similarapps_subgraph_dot_json_path, json_filename)
    similarapps_nodes = set(node for app_nodes in relevant_similarapps_node_ids_dict.values() for nodes in app_nodes.values() for node in nodes)
    all_relevant_nodes = set(all_kinds_node_ids).union(similarapps_nodes)
    existing_nodes = set(MG_cg_dot.nodes())
    valid_nodes = all_relevant_nodes.intersection(existing_nodes)
    if not valid_nodes:
        return nx.MultiDiGraph(), {}
    connected_nodes = set()
    node_list = list(valid_nodes)
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            source, target = node_list[i], node_list[j]
            try:
                if nx.has_path(MG_cg_dot, source, target):
                    path = nx.shortest_path(MG_cg_dot, source, target)
                    connected_nodes.update(path)
            except:
                continue
    connected_nodes.update(valid_nodes)
    subgraph = MG_cg_dot.subgraph(connected_nodes).copy()
    subgraph_dict = {}
    for node in subgraph.nodes():
        node_data = MG_cg_dot.nodes[node]
        subgraph_dict[node] = {
            'label': node_data.get('label', ''),
            'attributes': node_data
        }
        if node in similarapps_nodes:
            subgraph_dict[node]['type'] = 'similar_app_proposition'
        elif node in activity_node_ids:
            subgraph_dict[node]['type'] = 'activity'
        elif node in api_call_node_ids:
            subgraph_dict[node]['type'] = 'api_call'
        elif node in sdk_call_node_ids:
            subgraph_dict[node]['type'] = 'sdk_call'
        elif node in gui_context_node_ids:
            subgraph_dict[node]['type'] = 'gui_context'
        else:
            subgraph_dict[node]['type'] = 'general'
    try:
        os.makedirs(args.spec_similarapps_subgraph_dot_path, exist_ok=True)
        os.makedirs(args.spec_similarapps_subgraph_dot_json_path, exist_ok=True)
        nx.drawing.nx_pydot.write_dot(subgraph, dot_path)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(subgraph_dict, f, indent=2, ensure_ascii=False)
    except:
        pass
    return subgraph, subgraph_dict


def subgraph_node_summary_gen(
        general_cg_subgraph: nx.MultiDiGraph, general_cg_subgraph_dict: Dict, 
        spec_pp_cg_subgraph: nx.MultiDiGraph, spec_pp_cg_subgraph_dict: Dict,
        spec_similarapps_cg_subgraph: nx.MultiDiGraph, spec_similarapps_cg_subgraph_dict: Dict,
        cg_file: str, cg_node_json: str, data: Dict, args: Any
    ) -> Tuple[Dict, Dict, Dict]:
    def process_subgraph_node_summary(subgraph_dict, all_nodes_map, summary_dir, summary_filename, global_summary):
        summary_path = os.path.join(summary_dir, summary_filename)
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summaries = json.load(f)
                    loaded_dict = {node['id']: node for node in summaries}
                    global_summary.update(loaded_dict)
                    return loaded_dict
            except:
                pass
        summaries = []
        local_summary = {}
        for node_id, node_info in subgraph_dict.items():
            if node_id in global_summary:
                local_summary[node_id] = global_summary[node_id]
                continue
            node_type = node_info.get('type', '')
            if node_id in all_nodes_map:
                current_node = all_nodes_map[node_id].copy()
                if node_type == 'activity':
                    current_node['subgraph_node_info'] = {
                        "node_type": node_type,
                        "node_summary": None,
                        "quality_rate": None
                    }
                    summaries.append(current_node)
                    local_summary[node_id] = current_node
                    global_summary[node_id] = current_node
                    continue
                try:
                    if node_type == 'api_call':
                        node_summary, quality_rate = process_api_call_node(current_node, data, args)
                    elif node_type == 'sdk_call':
                        node_summary, quality_rate = process_sdk_call_node(current_node, data, args)
                    elif node_type == 'gui_context':
                        node_summary, quality_rate = process_gui_context_node(current_node, args)
                    elif node_type in ['pp_proposition', 'similar_app_proposition']:
                        node_summary, quality_rate = process_privacy_relevant_node(current_node, data, args)
                    else:
                        node_summary, quality_rate = process_general_node(current_node, args)
                    current_node['subgraph_node_info'] = {
                        "node_type": node_type,
                        "node_summary": node_summary,
                        "quality_rate": quality_rate
                    }
                    summaries.append(current_node)
                    local_summary[node_id] = current_node
                    global_summary[node_id] = current_node
                except:
                    continue
        try:
            os.makedirs(summary_dir, exist_ok=True)
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(list(local_summary.values()), f, indent=2, ensure_ascii=False)
        except:
            pass
        return local_summary

    try:
        with open(cg_node_json, 'r', encoding='utf-8') as f:
            all_nodes_info = json.load(f)
            all_nodes_map = {node['id']: node for node in all_nodes_info}
    except:
        return {}, {}, {}
    global_summary = {}
    general_summary_dict = process_subgraph_node_summary(
        general_cg_subgraph_dict,
        all_nodes_map,
        args.general_subgraph_node_summary_json,
        f"{data['appId']}_{data['pp_category']}_general.json",
        global_summary
    )
    spec_pp_summary_dict = process_subgraph_node_summary(
        spec_pp_cg_subgraph_dict,
        all_nodes_map,
        args.spec_pp_subgraph_node_summary_json,
        f"{data['appId']}_{data['pp_category']}_pp.json",
        global_summary
    )
    spec_similarapps_summary_dict = process_subgraph_node_summary(
        spec_similarapps_cg_subgraph_dict,
        all_nodes_map,
        args.spec_similarapps_subgraph_node_summary_json,
        f"{data['appId']}_{data['pp_category']}_similarapps.json",
        global_summary
    )
    return general_summary_dict, spec_pp_summary_dict, spec_similarapps_summary_dict


def process_general_node(node_info: Dict, args: Any) -> Tuple[str, int]:
    analysis_prompt = f"""Analyze this program node and provide a concise summary focusing on function name and code logic:

    Signature: {node_info.get('signature', 'N/A')}
    Body: {node_info.get('body', 'N/A')}

    Please provide a concise summary (preferably less than 20 words, but adjust length based on the richness, complexity and significance of the node information). Focus on the main functionality and operations. For simple nodes, be brief; for complex or critical nodes with rich information, include more detail as needed."""
    results = run_llm(
        analysis_prompt, 
        args.temperature_exploration, 
        args,
        args.max_tokens,
        args.LLM_type_generate,
        sys_msg="general"
    )
    summary = results.strip()
    return summary, 0

def process_api_call_node(node_info: Dict, data: Dict, args: Any) -> Tuple[str, int]:
    resources = node_info.get('cg_info', {}).get('resources', {})
    layouts = resources.get('layouts', {})
    strings = resources.get('strings', {})
    analysis_prompt = f"""Analyze this API call node considering its relationship with {data['pp_category']} privacy:

    Signature: {node_info.get('signature', 'N/A')}
    Body: {node_info.get('body', 'N/A')}"""
    if layouts or strings:
        analysis_prompt += f"""
    Resources:
    Layouts: {layouts if layouts else 'N/A'}
    Strings: {strings if strings else 'N/A'}"""
    analysis_prompt += f"""

    Please provide a concise summary (targeting less than 20 words, but adjust based on information richness) focusing on:
    1. Main functionality from code
    2. Relationship with {data['pp_category']} privacy (requesting/using privacy information)

    Note: Length should be proportional to the importance and complexity of the node. Shorter is preferred for simple nodes, but include all critical details for complex ones."""
    results = run_llm(
        analysis_prompt, 
        args.temperature_exploration, 
        args,
        args.max_tokens,
        args.LLM_type_generate,
        sys_msg="general"
    )
    summary = results.strip()
    return summary, 0

def process_sdk_call_node(node_info: Dict, data: Dict, args: Any) -> Tuple[str, int]:
    resources = node_info.get('cg_info', {}).get('resources', {})
    layouts = resources.get('layouts', {})
    strings = resources.get('strings', {})
    analysis_prompt = f"""Analyze this SDK node considering its relationship with {data['pp_category']} privacy:

    Signature: {node_info.get('signature', 'N/A')}
    Body: {node_info.get('body', 'N/A')}"""
    if layouts or strings:
        analysis_prompt += f"""
    Resources:
    Layouts: {layouts if layouts else 'N/A'}
    Strings: {strings if strings else 'N/A'}"""
    analysis_prompt += f"""

    Please provide a concise summary (aim for less than 20 words, but adjust as needed) focusing on:
    1. Main functionality from code
    2. Relationship with {data['pp_category']} privacy (requesting/using privacy information)
    
    Note: Brief summaries are preferred, but detail level should match the node's complexity and privacy significance. Include all essential information even if it exceeds the target length."""
    results = run_llm(
        analysis_prompt, 
        args.temperature_exploration, 
        args,
        args.max_tokens,
        args.LLM_type_generate,
        sys_msg="general"
    )
    summary = results.strip()
    return summary, 0

def process_gui_context_node(node_info: Dict, args: Any) -> Tuple[str, int]:
    resources = node_info.get('cg_info', {}).get('resources', {})
    layouts = resources.get('layouts', {})
    strings = resources.get('strings', {})
    analysis_prompt = f"""Analyze this GUI context node focusing on UI-related functionality:

    Signature: {node_info.get('signature', 'N/A')}
    Body: {node_info.get('body', 'N/A')}"""
    if layouts or strings:
        analysis_prompt += f"""
    Resources:
    Layouts: {layouts if layouts else 'N/A'}
    Strings: {strings if strings else 'N/A'}"""
    analysis_prompt += """
    Please provide a concise summary (preferably less than 40 words, but adjust based on information density) focusing on potential information/functionality implied by the code and UI resources.
    Note: Brevity is valued, but completeness is essential. Shorter summaries are appropriate for simple nodes, while complex nodes with rich implications or details may require more elaboration."""
    results = run_llm(
        analysis_prompt, 
        args.temperature_exploration, 
        args,
        args.max_tokens,
        args.LLM_type_generate,
        sys_msg="general"
    )
    summary = results.strip()
    return summary, 0

def process_privacy_relevant_node(node_info: Dict, data: Dict, args: Any) -> Tuple[str, int]:
    node_type = node_info.get('node_type', '')
    resources = node_info.get('cg_info', {}).get('resources', {})
    layouts = resources.get('layouts', {})
    strings = resources.get('strings', {})
    if node_type == 'pp_proposition':
        analysis_prompt = f"""Analyze this node with a focus on app functionality and implementation details:
    Signature: {node_info.get('signature', 'N/A')}
    Body: {node_info.get('body', 'N/A')}"""
    else:
        analysis_prompt = f"""Analyze this node with a focus on app functionality and implementation details:
    Signature: {node_info.get('signature', 'N/A')}
    Body: {node_info.get('body', 'N/A')}"""
    if layouts or strings:
        analysis_prompt += f"""
    Resources:
    Layouts: {layouts if layouts else 'N/A'}
    Strings: {strings if strings else 'N/A'}"""
    analysis_prompt += f"""

    Please provide a concise summary (preferably less than 40 words, but adjust based on information importance) focusing on:
    1. App functionalities and features contained in or related to this node (not just privacy-specific ones)
    2. Key processing steps and technical implementation details in the node body
    3. Both direct and indirect relationships with {data['pp_category']} privacy aspects
    4. Any potential data flows or user interactions implied by the code

    Note: Cover all significant functionality regardless of whether it explicitly mentions privacy. Include implementation details that might be relevant to understanding how the app works, even if their privacy implications aren't immediately obvious. Prioritize completeness over brevity when important details are present."""
    results = run_llm(
        analysis_prompt, 
        args.temperature_exploration, 
        args,
        args.max_tokens,
        args.LLM_type_generate,
        sys_msg="general"
    )
    summary = results.strip()
    return summary, 0
