from pp_utils import *
from parse import *
import json
import time
import re
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
from transformers import pipeline

import os
import networkx as nx
from networkx.drawing.nx_pydot import read_dot
from itertools import islice

import numpy as np
import random
import openai
from sklearn.metrics.pairwise import cosine_similarity

import time
import bert_score


def run_llm(prompt, temperature, openai_api_keys, max_tokens=4096, engine="gpt-4o", sys_msg="normal"):
    if "llama" in engine.lower():
        openai.api_key = "EMPTY"
        openai.api_base = "http://localhost:8000/v1"
        engine = openai.Model.list()["data"][0]["id"]
    else:
        client = openai.OpenAI(api_key=openai_api_keys)

    system_message = {
        "des": "You are an AI assistant that helps generate comprehensive and precise app descriptions.",
        "extract": "You are an AI assistant that extracts filter information from text.",
        "choose": "You are an AI assistant that helps people make choices based on information.",
        "gen": "You are an AI assistant that generates text based on prompts."
    }.get(sys_msg, "You are a helpful AI assistant.")

    messages = [
        {"role": "system", "content": system_message},
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
            return response.choices[0].message.content
        except Exception as e:
            if 'context_length_exceeded' in str(e):
                print("Context length exceeded, switching to 'gpt-4o' model.")
                engine = "gpt-4o"
            else:
                print(f"Unexpected error: {e}, retrying in 2 seconds...")
                time.sleep(2)


def run_llm_seq(prompts, temperature, openai_api_keys, max_tokens=4096, engine="gpt-4o", sys_msg="normal"):
    if "llama" in engine.lower():
        openai.api_key = "EMPTY"
        openai.api_base = "http://localhost:8000/v1"
        engine = openai.Model.list()["data"][0]["id"]
    else:
        client = openai.OpenAI(api_key=openai_api_keys)

    system_message = {
        "des": "You are an AI assistant that helps generate comprehensive and precise app descriptions.",
        "extract": "You are an AI assistant that extracts filter information from text.",
        "choose": "You are an AI assistant that helps people make choices based on information.",
        "gen": "You are an AI assistant that generates text based on prompts."
    }.get(sys_msg, "You are a helpful AI assistant.")

    messages = [{"role": "system", "content": system_message}]
    results = []

    for prompt in prompts:
        messages.append({"role": "user", "content": prompt})

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
        except Exception as e:
            print(f"OpenAI error, retrying: {str(e)}")
            time.sleep(2)

    return results


def construct_uncvd_str_filter_prompt(app_id, pp_category, uncvd_string_xml_strings, uncvd_layout_hardcoded_strings):
    prompt = (
        f"The following is a list of strings extracted from an app. "
        f"The app may use various privacy permissions, and your task is to identify and extract strings that might indicate the app's use of '{pp_category}' privacy information. "
        "Please use your imagination and reasoning to find non-explicit connections between the strings and the use of '{pp_category}' privacy information. "
        "You do not need to strictly follow the input strings; you can extract similar strings or summarize a few related strings into one. "
        "The final result should be a list of at most 10 strings. Each relevant string should be placed on a new line. "
        "Do not include any additional text or comments. "
        "The output should be formatted as follows:\n\n"
        f"(Relevant String 1)\n"
        f"(Relevant String 2)\n" 
        "...\n\n"
        f"Strings from strings.xml:\n"
        f"{list(uncvd_string_xml_strings)}\n\n"
        f"Strings from layout XMLs:\n"
        f"{list(uncvd_layout_hardcoded_strings)}\n\n"
    )
    return prompt


def uncovered_strings_filter(data, uncvd_string_xml_strings, uncvd_layout_hardcoded_strings, args):
    app_id = data['appId']
    pp_category = data['pp_category']
    prompt = construct_uncvd_str_filter_prompt(app_id, pp_category, uncvd_string_xml_strings, uncvd_layout_hardcoded_strings)
    response = run_llm(prompt, 0.0, args.openai_api_keys, args.max_tokens, args.LLM_type_generate, sys_msg='normal')

    key_uncvd_strings = [line.strip() for line in response.split('\n') if line.strip() and not line.lower().startswith("relevant")]

    return key_uncvd_strings


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
            print(f"Error encountered: {e}")
            return None

    if method == 'gpt':
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
                        print(f"Current text is too long even with batch size 1. Using placeholder embedding. Length: {len(batch_texts)}.")
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
    norms1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
    norms2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)

    norms1[norms1 == 0] = 1
    norms2[norms2 == 0] = 1
    
    normalized_embeddings1 = embeddings1 / norms1
    normalized_embeddings2 = embeddings2 / norms2
    
    similarity_matrix = np.dot(normalized_embeddings1, normalized_embeddings2.T)
    return similarity_matrix


def find_neighbors(node_ids, all_nodes_embeddings, all_node_signatures):
    neighbor_dict = {}
    for node_id in node_ids:
        neighbors = set()
        for i in range(1, 4):
            neighbors.update(nx.single_source_shortest_path_length(cg_dot, node_id, cutoff=i).keys())
        neighbors.discard(node_id)

        node_embedding = all_nodes_embeddings[all_node_signatures[str(node_id)]]
        filtered_neighbors = []
        for neighbor in neighbors:
            neighbor_str = str(neighbor)
            if neighbor_str in all_node_signatures:
                neighbor_embedding = all_nodes_embeddings[all_node_signatures[neighbor_str]]
                similarity = calculate_similarity(node_embedding, neighbor_embedding)
                if similarity > args.similarity_threshold:
                    filtered_neighbors.append(neighbor)
        neighbor_dict[node_id] = filtered_neighbors
    
    return neighbor_dict


def load_embeddings_from_file(file_path):
    with open(file_path, 'r') as file):
        embeddings = json.load(file)
    return embeddings


def cg_entity_search(cg_file, topic_api_calls, data, pp_keyphrases, pp_keywords, pp_propositions, args, uncvd_string_xml_strings, uncvd_layout_hardcoded_strings):
    node_contexts, node_signatures = parse_graph_contexts_ret_dict(cg_file)
    
    json_filename = os.path.splitext(os.path.basename(cg_file))[0] + '_embeddings.json'
    json_full_path = os.path.join(args.embedding_json_path, json_filename)

    all_tgt_texts = pp_keyphrases + pp_keywords
    all_contexts = {context for contexts in node_contexts.values() for context in contexts}

    all_tgt_texts_embeddings = calculate_embeddings(all_tgt_texts, args.openai_api_keys)
    if os.path.exists(json_full_path):
        start_time = time.time()
        print(f"Loading precomputed embeddings from {json_full_path}")
        embeddings_data = load_embeddings_from_file(json_full_path)
        all_contexts_embeddings = embeddings_data.get("contexts", {})
        all_signature_embeddings = embeddings_data.get("signatures", {})
        print(f"Read json time: {time.time() - start_time:.2f} seconds")
        
        default_embedding = [0] * 1536
        
        start_time = time.time()
        unmatched_context_keys = 0
        context_embeddings_list = []
        total_context_keys = 0
        for contexts in node_contexts.values():
            for context in contexts:
                total_context_keys += 1
                embedding = all_contexts_embeddings.get(context, default_embedding)
                context_embeddings_list.append(embedding)
                if embedding == default_embedding:
                    unmatched_context_keys += 1

        context_embeddings = np.array(context_embeddings_list)

        unmatched_signature_keys = 0
        signature_embeddings_list = []
        total_signature_keys = 0
        for sig in node_signatures.values():
            total_signature_keys += 1
            embedding = all_signature_embeddings.get(sig, default_embedding)
            signature_embeddings_list.append(embedding)
            if embedding == default_embedding:
                unmatched_signature_keys += 1

        node_embeddings_matrix = np.array(signature_embeddings_list)
        print(f"Format transfer time: {time.time() - start_time:.2f} seconds")

        print(f"Number of unmatched context keys: {unmatched_context_keys}/{total_context_keys} ({unmatched_context_keys/total_context_keys:.2%})")
        print(f"Number of unmatched signature keys: {unmatched_signature_keys}/{total_signature_keys} ({unmatched_signature_keys/total_signature_keys:.2%})")
    else:
        start_time = time.time()
        print("Precomputed embeddings not found, calculating embeddings...")
        all_contexts = {context for contexts in node_contexts.values() for context in contexts}
        all_contexts_embeddings = calculate_embeddings(list(all_contexts), args.openai_api_keys)
        all_signature_embeddings = calculate_embeddings(list(node_signatures.values()), args.openai_api_keys)
        print(f"Embedding time: {time.time() - start_time:.2f} seconds")

        context_embeddings = np.array([all_contexts_embeddings[context] for contexts in node_contexts.values() for context in contexts])
        node_embeddings_matrix = np.array([all_signature_embeddings[sig] for sig in node_signatures.values()])

    tgt_texts_embeddings = np.array([all_tgt_texts_embeddings[target] for target in all_tgt_texts_embeddings])
    similarity_matrix = calculate_similarity_matrix(context_embeddings, tgt_texts_embeddings)
    node_similarities = {}
    context_idx = 0
    for node_id, contexts in node_contexts.items():
        num_contexts = len(contexts)
        context_similarities = similarity_matrix[context_idx:context_idx + num_contexts]
        if context_similarities.size == 0:
            max_similarities = np.array([0])
        else:
            max_similarities = np.max(context_similarities, axis=1)
        node_similarities[node_id] = np.max(max_similarities)
        context_idx += num_contexts

    sorted_nodes = sorted(node_similarities.items(), key=lambda item: item[1], reverse=True)
    filtered_initial_nodes_id = [node_id for node_id, _ in sorted_nodes[:args.cg_nodes]]

    api_call_contexts = topic_api_calls[data['pp_category']]
    api_call_embeddings = calculate_embeddings(api_call_contexts, args.openai_api_keys)

    api_call_embeddings_matrix = np.array([api_call_embeddings[api] for api in api_call_contexts])
    similarity_matrix = calculate_similarity_matrix(node_embeddings_matrix, api_call_embeddings_matrix)
    node_api_similarities = {}
    node_ids = list(node_signatures.keys())
    for i, node_id in enumerate(node_ids):
        if similarity_matrix[i].size == 0:
            max_similarity = 0
        else:
            max_similarity = np.max(similarity_matrix[i])
        node_api_similarities[node_id] = max_similarity

    sorted_api_nodes = sorted(node_api_similarities.items(), key=lambda item: item[1], reverse=True)
    filtered_api_nodes_id = [node_id for node_id, _ in sorted_api_nodes[:args.cg_nodes]]
    
    return filtered_initial_nodes_id, filtered_api_nodes_id


def find_all_paths(MG_cg_dot, node_list, max_length):
    all_paths = set()
    unconnected_nodes_id = set(node_list)

    for i, start_node_id in enumerate(node_list):
        has_path_to_any = False
        for end_node_id in node_list:
            if start_node_id == end_node_id:
                continue
            try:
                path = nx.shortest_path(MG_cg_dot, source=start_node_id, target=end_node_id)
                if len(path) - 1 <= max_length:
                    all_paths.add(tuple(path))
                    has_path_to_any = True
                    if end_node_id in unconnected_nodes_id:
                        unconnected_nodes_id.remove(end_node_id)
            except nx.NetworkXNoPath:
                continue
        if not has_path_to_any and start_node_id in unconnected_nodes_id:
            unconnected_nodes_id.remove(start_node_id)

    return list(all_paths), list(unconnected_nodes_id)


def is_subpath_of(subpath, path):
    if len(subpath) > len(path):
        return False
    for i in range(len(path) - len(subpath) + 1):
        if path[i:i + len(subpath)] == subpath:
            return True
    return False


def merge_paths(all_paths):
    all_paths.sort(key=len)

    merged_paths = []

    for path in all_paths:
        is_subpath = False
        for merged_path in merged_paths:
            if is_subpath_of(path, merged_path):
                is_subpath = True
                break
        if not is_subpath:
            merged_paths.append(path)

    return merged_paths, len(merged_paths)


def cg_path_find(MG_cg_dot, cg_initial_nodes, cg_api_call_nodes, args, path_each_agent=10, max_path_length=100):
    all_paths, unconnected_nodes_id = find_all_paths(MG_cg_dot, cg_initial_nodes + cg_api_call_nodes, max_path_length)
    merged_paths, path_count = merge_paths(all_paths)

    max_paths = path_each_agent * args.cg_agent_num
    
    if path_count > max_paths:
        selected_paths = random.sample(merged_paths, max_paths)
    elif path_count > path_each_agent:
        selected_paths = []
        i = 0
        while len(selected_paths) < max_paths:
            selected_paths.append(merged_paths[i % path_count])
            i += 1
    else:
        selected_paths = []
        for _ in range(args.cg_agent_num):
            random.shuffle(merged_paths)
            selected_paths.extend(merged_paths)
        selected_paths = selected_paths[:max_paths]
    
    path_groups = [selected_paths[i::args.cg_agent_num] for i in range(args.cg_agent_num)]
    
    return path_groups, unconnected_nodes_id


def format_node_info(node_info):
    class_name = node_info['class_name']
    method_name = node_info['method_name']
    method_params = node_info['method_params']
    return_type = node_info['return_type']
    context_str = node_info['context_str'] if node_info['context_str'] else ''
    
    return f"Class - {class_name}, Method - {return_type} {method_name}({method_params}), Context - {context_str}"


def format_paths(cg_path_groups, all_cg_nodes_info):
    all_path_prompts = []
    
    for group_index, group in enumerate(cg_path_groups):
        group_prompt = []
        for path_index, path in enumerate(group):
            path_prompt = [f"PATH {path_index + 1}:"]
            for node_id in path:
                if str(node_id) in all_cg_nodes_info.keys():
                    node_info = all_cg_nodes_info[str(node_id)]
                    formatted_node_info = format_node_info(node_info)
                    path_prompt.append(f"Node {node_id}: {formatted_node_info}")
                else:
                    continue
            group_prompt.append("\n".join(path_prompt))
        
        all_path_prompts.append("\n\n".join(group_prompt))

    return all_path_prompts


def format_unconnected_nodes(unconnected_nodes, all_cg_nodes_info):
    unconnected_prompt = []

    for node_id in unconnected_nodes:
        node_info = all_cg_nodes_info.get(str(node_id), None)
        if node_info:
            formatted_node_info = format_node_info(node_info)
            unconnected_prompt.append(f"Node {node_id}: {formatted_node_info}")
    
    return "\n".join(unconnected_prompt)


def cg_multiagent_processing(cg_path_groups, unconnected_nodes, all_cg_nodes_info, pp_description, pp_propositions, app_name, pp_category, nested_top5_dict, args, max_unconnected_nodes_pass=50):
    if args.des_gen_prompt_choice == 1:
        all_path_prompts = format_paths(cg_path_groups, all_cg_nodes_info)
        if len(unconnected_nodes) > max_unconnected_nodes_pass:
            unconnected_nodes = unconnected_nodes[:max_unconnected_nodes_pass]
        unconnected_nodes_prompt = format_unconnected_nodes(unconnected_nodes, all_cg_nodes_info)

        all_agents_proposition_insights = []
        for i in range(args.cg_agent_num):
            initial_prompt_template = (
                f"Privacy Category: {pp_category}\n\n"
                "Our ultimate goal is to generate an app description that includes detailed discussions and descriptions of how this app uses privacy information related to the category '{pp_category}'.\n\n"
                "From our previous analysis of the privacy policy, we have developed some propositions regarding the usage of privacy information in this category. "
                "We also provide you with semantic information extracted from the program's call graph and GUI contexts.\n\n"
                "This information includes multiple call graph paths and nodes, which may implicitly contain the program's functionality. It is crucial to understand the function names, method names, and context content of these paths and nodes. "
                "Special attention should be paid to the context content derived from the app layout definitions or referenced textual resources. "
                "Focus on the possible implicit relationships and structures within and between paths. Treat the following content as a knowledge base, and reference this part in the subsequent analysis to help illustrate your points.\n\n"
                "Paths:\n{paths}\n\n"
                "Unconnected Nodes:\n{unconnected_nodes}\n\n"
                "Here are the propositions based on our analysis of the privacy policy. For each proposition, analyze the information provided and determine if there is support, opposition, refinement, or extension to the proposition. "
                "Provide your insights and use the information from the knowledge base to substantiate your points:\n\n"
            )

            propositions_text = "\n".join([f"Proposition {i + 1}: {prop}" for i, prop in enumerate(pp_propositions)])

            analysis_direction = (
                "\n\nPlease analyze the above propositions. For each proposition, provide your insights. "
                "You don't need to copy the information verbatim; feel free to connect different parts of the information and use your thoughts and imagination to make connections. "
                "Additionally, please provide a confidence score (0-100) for the viewpoint you express in your response. "
                "\n\nExpected output format:\n"
                "For each proposition, repeat the proposition and then on a new line write your analysis. "
                "Example:\n"
                "Proposition 1: [Proposition text]\n"
                "Analysis: [Your detailed analysis here]...\n\n"
                "Proposition 2: [Proposition text]\n"
                "Analysis: [Your detailed analysis here]...\n\n"
                "..."
            )

            full_prompt = initial_prompt_template.format(pp_category=pp_category,
                                                        paths=all_path_prompts[i],
                                                        unconnected_nodes=unconnected_nodes_prompt) + propositions_text + analysis_direction

            proposition_insights = run_llm(full_prompt, args.temperature_exploration, args.openai_api_keys, args.max_tokens, args.LLM_type_generate, sys_msg='gen')
            
            all_agents_proposition_insights.append(proposition_insights)

        full_prompt_second = []
        description_gen_prompt = (
            f"In the first round of analysis, we developed and examined several propositions related to how this app uses privacy information categorized under '{pp_category}'. The insights derived from this analysis were provided by an agent, bringing unique perspectives based on the call graph paths, nodes, and other semantic information provided earlier.\n\n"
            "Here is a summary of the propositions and the insights provided by the agent:\n\n"
        )

        for index, insights in enumerate(all_agents_proposition_insights):
            description_gen_prompt += f"Agent {index + 1} Insights:\n"
            description_gen_prompt += f"{insights}\n\n"

        description_gen_prompt += (
            f"Based on the comprehensive analysis and insights for the propositions, you are now tasked to generate a detailed description of how this app manages privacy information related to '{pp_category}'. "
            "While crafting your description, focus specifically on:\n"
            "- Terms and phrases that describe the app's specific functionalities related to the use of privacy information.\n"
            "- The reasons for using the privacy information, providing context as to why such data is necessary for the app's functionality.\n"
            "- Specific functions achieved using the privacy information, illustrating how the app benefits from accessing such data.\n\n"
            "The description should incorporate the understandings from the analyses without explaining the underlying reasons in detail. \n\n"
        )

        if args.few_shot_num != 0:
            description_gen_prompt += "Here are some examples of similar analyses for your reference:\n\n"

            few_shot_num = args.few_shot_num
            few_shot_examples = set()

            example_count = 0
            for i in range(few_shot_num):
                app_id = nested_top5_dict['PP_Sim']['pp_top5_list'][i]
                if app_id in few_shot_examples:
                    continue
                example_count += 1
                few_shot_examples.add(app_id)
                pp_segments = nested_top5_dict['PP_Sim']['pp_segments'][app_id]
                layout_elements = nested_top5_dict['PP_Sim']['layout_elements'][app_id]
                layout_pp_summary = nested_top5_dict['PP_Sim']['layout_pp_summary'][app_id]
                description_gen_prompt += (
                    f"Example {str(example_count)}:\n"
                    f"Input Information:\n"
                    f"Privacy policy segments: {pp_segments}\n"
                    f"GUI context elements: {layout_elements}\n\n"
                    f"Summary:\n"
                    f"{layout_pp_summary}\n\n"
                )

            for i in range(few_shot_num):
                app_id = nested_top5_dict['Layout_Sim']['layout_top5_list'][i]
                if app_id in few_shot_examples:
                    continue
                example_count += 1
                few_shot_examples.add(app_id)
                pp_segments = nested_top5_dict['Layout_Sim']['pp_segments'][app_id]
                layout_elements = nested_top5_dict['Layout_Sim']['layout_elements'][app_id]
                layout_pp_summary = nested_top5_dict['Layout_Sim']['layout_pp_summary'][app_id]
                description_gen_prompt += (
                    f"Example {str(example_count)}:\n"
                    f"Input Information:\n"
                    f"Privacy policy segments: {pp_segments}\n"
                    f"GUI context elements elements: {layout_elements}\n\n"
                    f"Summary:\n"
                    f"{layout_pp_summary}\n\n"
                )

        description_gen_prompt += (
            f"Now, based on the insights provided earlier and the examples given, generate a detailed description of how this app manages privacy information related to '{pp_category}'. "
            "Your description should incorporate terms and phrases that describe the app's specific functionalities related to the use of privacy information, "
            "the reasons for using the privacy information, and specific functions achieved using the privacy information. "
            "While generating the description, consider the thought process and information aspects demonstrated in the examples. "
            "Identify and explore potential implicit connections and similarities between the examples and the app under consideration. "
            "However, ensure that you do not directly copy the information from the examples, but use them to inform and guide your own detailed and original description."
        )

        full_prompt_second.append(description_gen_prompt)

        description_condense_prompt = (
            "Now, please condense the detailed description provided earlier into a concise summary. "
            "This summary should be no more than 75 words. Focus on including critical keywords related to the app’s functionalities that involve privacy information. "
            "Aim to capture the essence of how the app uses privacy data, prioritizing clarity and brevity. "
            "Minimize descriptive phrases and avoid elaborate details; instead, highlight the app's key privacy-related actions and features. "
            "Ensure that the condensed description accurately reflects the main points while being direct and to the point.\n"
        )
        full_prompt_second.append(description_condense_prompt)

        descriptions = run_llm_seq(full_prompt_second, args.temperature_exploration, args.openai_api_keys, args.max_tokens, args.LLM_type_generate, sys_msg='gen')
        detailed_description = descriptions[0]
        concise_description = descriptions[1]

        return detailed_description, concise_description
    
    elif args.des_gen_prompt_choice == 2:
        all_path_prompts = format_paths(cg_path_groups, all_cg_nodes_info)
        if len(unconnected_nodes) > max_unconnected_nodes_pass:
            unconnected_nodes = unconnected_nodes[:max_unconnected_nodes_pass]
        unconnected_nodes_prompt = format_unconnected_nodes(unconnected_nodes, all_cg_nodes_info)

        initial_prompt_template = (
            f"Privacy Category: {pp_category}\n\n"
            "Our ultimate goal is to generate an app description that includes detailed discussions and descriptions of how this app uses privacy information related to the category '{pp_category}'.\n\n"
            "From our previous analysis of the privacy policy, we have developed some propositions regarding the usage of privacy information in this category. "
            "We also provide you with semantic information extracted from the program's call graph and GUI contexts.\n\n"
            "This information includes multiple call graph paths and nodes, which may implicitly contain the program's functionality. It is crucial to understand the function names, method names, and context content of these paths and nodes. "
            "Special attention should be paid to the context content derived from the app layout definitions or referenced textual resources. "
            "Focus on the possible implicit relationships and structures within and between paths. Treat the following content as a knowledge base, and reference this part in the subsequent analysis to help illustrate your points.\n\n"
            "Paths:\n{paths}\n\n"
            "Unconnected Nodes:\n{unconnected_nodes}\n\n"
            "Here are the propositions based on our analysis of the privacy policy. For each proposition, analyze the information provided and determine if there is support, opposition, refinement, or extension to the proposition. "
            "Provide your insights and use the information from the knowledge base to substantiate your points.\n\n"
        )

        propositions_text = "\n".join([f"Proposition {i + 1}: {prop}" for i, prop in enumerate(pp_propositions)])

        analysis_direction = (
            "\n\nAdditionally, please provide a comprehensive analysis that incorporates the insights from the propositions and examples given below. "
            f"Generate a detailed description of how this app manages privacy information related to '{pp_category}'. "
            "Your description should include terms and phrases that describe the app's specific functionalities related to the use of privacy information, "
            "the reasons for using the privacy information, and specific functions achieved using the privacy information. "
            "Consider the thought process and information aspects demonstrated in the examples. "
            "Identify and explore potential implicit connections and similarities between the examples and the app under consideration. "
            "Ensure that you do not directly copy the information from the examples, but use them to inform and guide your own detailed and original description.\n\n"
        )

        full_prompt_first = initial_prompt_template.format(pp_category=pp_category,
                                                    paths=all_path_prompts[0],
                                                    unconnected_nodes=unconnected_nodes_prompt) + propositions_text + analysis_direction

        if args.few_shot_num != 0:
            full_prompt_first += "Here are some examples of similar analyses for your reference:\n\n"
            few_shot_num = args.few_shot_num
            few_shot_examples = set()

            example_count = 0
            for i in range(few_shot_num):
                app_id = nested_top5_dict['PP_Sim']['pp_top5_list'][i]
                if app_id in few_shot_examples:
                    continue
                example_count += 1  
                few_shot_examples.add(app_id)
                pp_segments = nested_top5_dict['PP_Sim']['pp_segments'][app_id]
                layout_elements = nested_top5_dict['PP_Sim']['layout_elements'][app_id]
                layout_pp_summary = nested_top5_dict['PP_Sim']['layout_pp_summary'][app_id]
                full_prompt_first += (
                    f"Example {str(example_count)}:\n"
                    f"Input Information:\n"
                    f"Privacy policy segments: {pp_segments}\n"
                    f"GUI context elements elements: {layout_elements}\n\n"
                    f"Summary:\n"
                    f"{layout_pp_summary}\n\n"
                )

            for i in range(few_shot_num):
                app_id = nested_top5_dict['Layout_Sim']['layout_top5_list'][i]
                if app_id in few_shot_examples:
                    continue
                example_count += 1
                few_shot_examples.add(app_id)
                pp_segments = nested_top5_dict['Layout_Sim']['pp_segments'][app_id]
                layout_elements = nested_top5_dict['Layout_Sim']['layout_elements'][app_id]
                layout_pp_summary = nested_top5_dict['Layout_Sim']['layout_pp_summary'][app_id]
                full_prompt_first += (
                    f"Example {str(example_count)}:\n"
                    f"Input Information:\n"
                    f"Privacy policy segments: {pp_segments}\n"
                    f"GUI context elements elements: {layout_elements}\n\n"
                    f"Summary:\n"
                    f"{layout_pp_summary}\n\n"
                )

        full_prompt_first += (
            f"Now, generate a detailed description of how this app manages privacy information related to '{pp_category}'. "
            "Your description should incorporate terms and phrases that describe the app's specific functionalities related to the use of privacy information, "
            "the reasons for using the privacy information, and specific functions achieved using the privacy information. "
            "While generating the description, consider the thought process and information aspects demonstrated in the examples. "
            "Identify and explore potential implicit connections and similarities between the examples and the app under consideration. "
            "However, ensure that you do not directly copy the information from the examples, but use them to inform and guide your own detailed and original description."
        )

        full_prompt_second = []
        full_prompt_second.append(full_prompt_first)

        description_condense_prompt = (
            "Now, please condense the detailed description provided earlier into a concise summary. "
            "This summary should be no more than 75 words. Focus on including critical keywords related to the app’s functionalities that involve privacy information. "
            "Aim to capture the essence of how the app uses privacy data, prioritizing clarity and brevity. "
            "Minimize descriptive phrases and avoid elaborate details; instead, highlight the app's key privacy-related actions and features. "
            "Ensure that the condensed description accurately reflects the main points while being direct and to the point.\n"
        )
        full_prompt_second.append(description_condense_prompt)

        descriptions = run_llm_seq(full_prompt_second, args.temperature_exploration, openai_api_keys=args.openai_api_keys, max_tokens=args.max_tokens, engine=args.LLM_type_generate, sys_msg='gen')
        detailed_description = descriptions[0]
        concise_description = descriptions[1]

        return detailed_description, concise_description


def evaluate(msg, generated_description, ground_truth_description, openai_api_keys):
    texts = [generated_description, ground_truth_description]
    embeddings = calculate_embeddings(texts, openai_api_keys)
    gen_embedding = np.array(embeddings[generated_description]).reshape(1, -1)
    gt_embedding = np.array(embeddings[ground_truth_description]).reshape(1, -1)

    cos_sim = cosine_similarity(gen_embedding, gt_embedding)[0][0]

    P, R, F1 = bert_score.score([generated_description], [ground_truth_description], lang="en", verbose=True)
    bert_score_f1 = F1.mean().item()

    print(msg, "\n Cosine Similarity: ", cos_sim, "\n Bert-Score: ", float(P.mean().item()), float(R.mean().item()), bert_score_f1)

    return cos_sim, float(P.mean().item()), float(R.mean().item()), bert_score_f1
