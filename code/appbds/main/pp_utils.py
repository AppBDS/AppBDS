from cg_utils import *
from parse import *
import json, time, re, os, networkx as nx
from networkx.drawing.nx_pydot import read_dot
from itertools import islice
import numpy as np
import openai, anthropic
from keybert.llm import OpenAI
from keybert import KeyLLM
from sklearn.metrics.pairwise import cosine_similarity
import bert_score
from sentence_transformers import util, SentenceTransformer
from transformers import pipeline

def run_llm(prompt, temperature, args, max_tokens=4096, engine="gpt-4o", sys_msg="normal"):
    system_message = ("You are an AI assistant that helps generate comprehensive and precise app descriptions." if sys_msg=="des" 
                      else "You are an AI assistant that extracts filter information from text." if sys_msg=="extract" 
                      else "You are an AI assistant that helps people make choices based on information." if sys_msg=="choose" 
                      else "You are an AI assistant that generates text based on prompts." if sys_msg=="gen" 
                      else "You are a helpful AI assistant.")
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
    if "llama" in engine.lower():
        try:
            openai.api_key = "EMPTY"
            openai.api_base = "http://localhost:8000/v1"
            engine = openai.Model.list()["data"][0]["id"]
            client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
            response = client.chat.completions.create(model=engine, messages=messages, temperature=temperature, max_tokens=max_tokens)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error using local Llama model: {e}")
            return ""
    try:
        client = openai.OpenAI(api_key=args.openai_api_keys)
        response = client.chat.completions.create(model=engine, messages=messages, temperature=temperature, max_tokens=max_tokens)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error using {engine}: {e}")
        if args.llm_error_handling_setting == "one_try":
            print("Error handling setting is 'one_try', returning empty result")
            return ""
        elif args.llm_error_handling_setting == "multi_try":
            print("Error handling setting is 'multi_try', trying Claude model as fallback via direct API call")
            try:
                import requests
                url = "https://api.anthropic.com/v1/messages"
                headers = {"x-api-key": args.anthropic_api_keys, "anthropic-version": "2023-06-01", "content-type": "application/json"}
                data = {"model": "claude-3-5-haiku-20241022", "max_tokens": max_tokens, "temperature": temperature, "system": system_message, "messages": [{"role": "user", "content": prompt}]}
                response = requests.post(url, headers=headers, json=data)
                if response.status_code == 200:
                    response_json = response.json()
                    return response_json["content"][0]["text"]
                else:
                    print(f"Claude API returned error: {response.status_code}, {response.text}")
                    return ""
            except Exception as claude_error:
                print(f"Error using Claude fallback model via direct API: {claude_error}")
                return ""
        else:
            print(f"Unknown error handling setting: {args.llm_error_handling_setting}, returning empty result")
            return ""

def run_llm_seq(prompts, temperature, args, max_tokens=4096, engine="gpt-4o", sys_msg="normal"):
    system_message = ("You are an AI assistant that helps generate comprehensive and precise app descriptions." if sys_msg=="des" 
                      else "You are an AI assistant that extracts filter information from text." if sys_msg=="extract" 
                      else "You are an AI assistant that helps people make choices based on information." if sys_msg=="choose" 
                      else "You are an AI assistant that generates text based on prompts." if sys_msg=="gen" 
                      else "You are a helpful AI assistant.")
    if "llama" in engine.lower():
        results = []
        try:
            openai.api_key = "EMPTY"
            openai.api_base = "http://localhost:8000/v1"
            engine = openai.Model.list()["data"][0]["id"]
            client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
            messages = [{"role": "system", "content": system_message}]
            for prompt in prompts:
                messages.append({"role": "user", "content": prompt})
                try:
                    response = client.chat.completions.create(model=engine, messages=messages, temperature=temperature, max_tokens=max_tokens)
                    result = response.choices[0].message.content
                    results.append(result)
                    messages.append({"role": "assistant", "content": result})
                except Exception as e:
                    print(f"Error in Llama model sequence call: {e}")
                    results.append("")
                    messages.append({"role": "assistant", "content": ""})
            return results
        except Exception as e:
            print(f"Fatal error using local Llama model: {e}")
            return [""] * len(prompts)
    results = []
    openai_messages = [{"role": "system", "content": system_message}]
    using_claude = False
    claude_messages = []
    for prompt in prompts:
        if not using_claude:
            openai_messages.append({"role": "user", "content": prompt})
        if not using_claude:
            try:
                client = openai.OpenAI(api_key=args.openai_api_keys)
                response = client.chat.completions.create(model=engine, messages=openai_messages, temperature=temperature, max_tokens=max_tokens)
                result = response.choices[0].message.content
                results.append(result)
                openai_messages.append({"role": "assistant", "content": result})
                continue
            except Exception as e:
                print(f"Error using {engine} in sequence: {e}")
                if args.llm_error_handling_setting == "one_try":
                    print("Error handling setting is 'one_try', adding empty result")
                    results.append("")
                    openai_messages.append({"role": "assistant", "content": ""})
                    continue
        if args.llm_error_handling_setting == "multi_try" and not using_claude:
            print("Error handling setting is 'multi_try', switching to Claude model via direct API for remaining prompts")
            using_claude = True
            claude_messages = [{"role": msg["role"], "content": msg["content"]} for msg in openai_messages[1:]]
            if not any(msg["content"] == prompt and msg["role"] == "user" for msg in claude_messages):
                claude_messages.append({"role": "user", "content": prompt})
        if using_claude:
            try:
                import requests
                url = "https://api.anthropic.com/v1/messages"
                headers = {"x-api-key": args.anthropic_api_keys, "anthropic-version": "2023-06-01", "content-type": "application/json"}
                data = {"model": "claude-3-5-haiku-20241022", "max_tokens": max_tokens, "temperature": temperature, "system": system_message, "messages": claude_messages}
                response = requests.post(url, headers=headers, json=data)
                if response.status_code == 200:
                    response_json = response.json()
                    result = response_json["content"][0]["text"]
                    results.append(result)
                    claude_messages.append({"role": "assistant", "content": result})
                else:
                    print(f"Claude API returned error: {response.status_code}, {response.text}")
                    results.append("")
                    claude_messages.append({"role": "assistant", "content": ""})
            except Exception as claude_error:
                print(f"Error using Claude in sequence via direct API: {claude_error}")
                results.append("")
                claude_messages.append({"role": "assistant", "content": ""})
    return results

def clean_string(text):
    if isinstance(text, str):
        text = re.sub(r"[‘’“”'\"[\]]", "", text)
    return text

def pp_get_entities(app_name, pp_segments, pp_category, args):
    prompt = (f"In this task, you'll analyze the provided privacy policy segments for the privacy category of {pp_category}. "
              "Extract keyphrases and keywords that describe the app's functionalities, reasons for using the privacy data, and specific functions. "
              "List each keyphrase or keyword on a new line without additional text. "
              f"Privacy Policy Segments:\n{pp_segments}\nPlease proceed with the extraction.")
    keywords = run_llm(prompt, args.temperature_exploration, args, args.max_tokens, args.LLM_type_extract, sys_msg="normal")
    keyword_list = [line.strip() for line in keywords.split('\n') if line.strip()]
    pp_keywords = []
    for keyword in keyword_list:
        for phrase in ["keyword", "keyphrase", "key phrase", ":"]:
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
        prompt = (f"You are about to analyze a privacy policy document. Extract segments related to the \"{pp_category}\" category. "
                  f"Focus on how the app collects, uses, stores, or shares data in the \"{pp_category}\" category. "
                  f"Privacy Policy Text:\n{pp}")
        pp_segments = run_llm(prompt, args.temperature_exploration, args, args.max_tokens, args.LLM_type_extract, sys_msg="normal")
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(pp_segments)
        print("Generated and saved PP segments to file.")
    prompt = (f"As an AI specialized in digital privacy, analyze and summarize how an app uses {pp_category} privacy data. "
              "Provide a clear description in a few concise sentences (50 words or fewer) focusing on practical usage.\n"
              f"Privacy Data Category: {pp_category}\nPrivacy Policy Segments:\n{pp_segments}\nAnswer:")
    pp_description = run_llm(prompt, args.temperature_exploration, args, args.max_tokens, args.LLM_type_extract, sys_msg="normal")
    pp_seg_extracted_keywords = pp_get_entities(app_name, pp_segments, pp_category, args)
    return pp_segments, pp_description, pp_seg_extracted_keywords

def pp_proposition_generation(pp_segments, pp_keywords, appId, pp_category, args):
    prompt = (f"Privacy Category: {pp_category}\n\nBased on the following privacy policy segments and key phrases, generate a list of propositions regarding the usage of {pp_category} privacy data for this app. "
              "Each proposition should be a concise sentence on a new line, expressing a distinct idea. If information is limited, infer potential functionalities and indicate conjecture.\n\n"
              "Format:\n(Proposition 1's content)\n(Proposition 2's content)\n...\n\n"
              "Key Phrases: " + ", ".join(pp_keywords) + "\n\nPrivacy Policy Segments:\n" + pp_segments + "\n\n")
    keyphr_prompt = ("Now, based on the generated propositions and the privacy policy segments, extract keyphrases and keywords related to the privacy category. "
                     "List each on a new line without additional text. Format:\n(Keyphrase 1's content)\n(Keyphrase 2's content)\n...\n\nPropositions:\nConsider the previously generated propositions.")
    prompts = [prompt, keyphr_prompt]
    results = run_llm_seq(prompts, args.temperature_exploration, args, args.max_tokens, args.LLM_type_extract, sys_msg="normal")
    pp_propositions_result = results[0]
    pp_keyphrases_result = results[1]
    pp_propositions = [line.strip() for line in pp_propositions_result.split('\n') if line.strip()]
    pp_keyphrases = []
    for keyword in [line.strip() for line in pp_keyphrases_result.split('\n') if line.strip()]:
        for phrase in ["keyword", "keyphrase", "key phrase", ":"]:
            keyword = keyword.lower().replace(phrase, "")
        cleaned_keyword = keyword.strip()
        if cleaned_keyword:
            pp_keyphrases.append(cleaned_keyword)
    return pp_propositions, pp_keyphrases
