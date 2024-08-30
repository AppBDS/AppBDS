from cg_utils import *
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

import openai
from keybert.llm import OpenAI
from keybert import KeyLLM
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
                print(f"Error: {e}")
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


def clean_string(text):
    if isinstance(text, str):
        text = re.sub(r"[‘’“”'\"[\]]", "", text)
    return text


def pp_get_entities(app_name, pp_segments, pp_category, args):
    pp_seg_keywords_ext_prompt = (
        f"In this task, you'll analyze the provided privacy policy segments, which pertain to the privacy category of {pp_category}. "
        "Your objective is to extract keyphrases and keywords related to this privacy category. "
        "Focus specifically on terms and phrases that describe the app's specific functionalities, the reasons for using the privacy information, and the specific functions achieved using the privacy information. "
        "Ensure that each keyphrase or keyword is placed on a new line. "
        "Avoid including any additional text or comments. "
        "The output should be formatted as follows:\n\n"
        "(Keyword 1's content)\n"
        "(Keyword 2's content)\n"
        "...\n\n"
        f"Here are the privacy policy segments for your analysis:\n{pp_segments}\nPlease proceed with the extraction."
    )

    keywords = run_llm(pp_seg_keywords_ext_prompt, args.temperature_exploration, args.openai_api_keys, args.max_tokens, args.LLM_type_generate, sys_msg='normal')

    keyword_list = [line.strip() for line in keywords.split('\n') if line.strip()]
    pp_keywords = []
    seg_remove = ["keyword", "keyphrase", "key phrase", ":"]
    for keyword in keyword_list:
        for phrase in seg_remove:
            keyword = keyword.lower().replace(phrase, "")
        cleaned_keyword = keyword.strip()
        if cleaned_keyword:
            pp_keywords.append(cleaned_keyword)

    return pp_keywords


def phase1_pp_processing(app_id, app_name, pp, pp_category, topic_keywords, args):
    file_path = os.path.join(args.pp_seg_path, f'{app_id}_{pp_category}.txt')

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            pp_segments = file.read()
        print("Loaded existing PP segments from file.")
    else:
        pp_seg_ext_prompt = (
            f"You are about to analyze a privacy policy document. Your task is to extract specific segments related to a particular category of user data. Carefully read the instructions below and then proceed to extract the relevant segments from the provided text.\n\n"
            f"Instructions:\n- Extract the segments that specifically pertain to the '{pp_category}' category within this privacy policy. Focus on details about how the application collects, uses, stores, or shares data in the '{pp_category}' category.\n"
            "- Extract segments that describe the application's functionalities, including specific features, related sentences, phrases, and keywords.\n"
            "- Do not add any additional commentary or content. Just reply with the extracted segments.\n\n"
            f"Below is the privacy policy text for your analysis:\n\n{pp}"
        )

        pp_segments = run_llm(pp_seg_ext_prompt, args.temperature_exploration, args.openai_api_keys, args.max_tokens, args.LLM_type_generate, sys_msg='normal')
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(pp_segments)
        print("Generated and saved PP segments to file.")

    pp_des_prompt = (
        f"As an AI specialized in analyzing digital privacy, your task is to analyze and summarize how an app uses a specific type of user privacy data for its functions and operations. Based on a comprehensive review of the app's Privacy Policy, provide a clear description in a few concise sentences. Focus on the particular privacy data category provided, detailing how this information is utilized by the app in practical terms.\n"
        "The goal is to succinctly convey how the app employs the specified privacy data in its features or activities, emphasizing its direct use and the functional aspect of such data handling. This should include, but is not limited to, how the app accesses, processes, or shares this data in the context of its services. The final description that you generate should be 50 words or fewer.\n\n"
        f"Below is the sample that you will be analyzing:\n- Privacy Data Category of Interest: {pp_category}\n- Privacy Policy segments:\n{pp_segments}\n\nAnswer:"
    )

    pp_description = run_llm(pp_des_prompt, args.temperature_exploration, args.openai_api_keys, args.max_tokens, args.LLM_type_generate, sys_msg='normal')

    pp_seg_extracted_keywords = pp_get_entities(app_name, pp_segments, pp_category, args)

    return pp_segments, pp_description, pp_seg_extracted_keywords


def pp_proposition_generation(pp_segments, pp_keywords, appId, pp_category, args):
    pp_prop_gen_prompt = (
        f"Privacy Category: {pp_category}\n\n"
        f"Based on the following privacy policy segments and identified key phrases, generate a list of propositions regarding the usage of {pp_category} privacy information for this app. "
        "While creating each proposition, attempt to provide details about specific scenarios, functionalities, and operations where the privacy information might be utilized, wherever possible. "
        "Each proposition should be a concise sentence and should be placed on a new line. "
        "Ensure that each proposition expresses a distinct idea and avoid outputting multiple propositions with similar meanings. "
        f"If the privacy policy provides limited or no information about the use of {pp_category} data, use inference to make educated guesses about the app's potential functionalities and their relation to {pp_category} data. "
        "Clearly indicate when a proposition is based on conjecture.\n\n"
        f"Key Phrases from Privacy Policy:\n{', '.join(pp_keywords)}\n\nPrivacy Policy Segments:\n{pp_segments}\n\n"
    )

    pp_keyphr_ext_prompt = (
        "Now, let's think about the next question. Based on the propositions you proposed regarding the usage of privacy information, as well as the original privacy policy segments, extract the keyphrases and keywords related to the privacy category. "
        "Focus specifically on terms and phrases related to the app's specific functionalities, reasons for using the privacy information, and the concrete functions accomplished using the privacy information. "
        "Each keyphrase or keyword should be placed on a new line. "
        "Do not include any additional text or comments. "
        "The output should be formatted as follows:\n\n(Keyphrase 1's content)\n(Keyphrase 2's content)\n...\n\n"
        "Propositions:\nConsider the propositions that you generated in our previous conversation."
    )

    prompts = [pp_prop_gen_prompt, pp_keyphr_ext_prompt]

    results = run_llm_seq(prompts, args.temperature_exploration, args.openai_api_keys, args.max_tokens, args.LLM_type_generate, sys_msg='normal')

    pp_propositions_result_list = [line.strip() for line in results[0].split('\n') if line.strip()]
    pp_propositions = []
    for keyword in pp_propositions_result_list:
        cleaned_keyword = keyword.strip()
        if cleaned_keyword:
            pp_propositions.append(cleaned_keyword)

    pp_keyphrases_result_list = [line.strip() for line in results[1].split('\n') if line.strip()]
    pp_keyphrases = []
    seg_remove = ["keyword", "keyphrase", "key phrase", ":"]
    for keyword in pp_keyphrases_result_list:
        for phrase in seg_remove:
            keyword = keyword.lower().replace(phrase, "")
        cleaned_keyword = keyword.strip()
        if cleaned_keyword:
            pp_keyphrases.append(cleaned_keyword)

    return pp_propositions, pp_keyphrases
