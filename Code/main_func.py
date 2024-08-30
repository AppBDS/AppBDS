import numpy as np
import json
from tqdm import tqdm
import argparse
from pp_utils import *
from cg_utils import *
from parse import *
import random
import os
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import read_dot
from copy import deepcopy
import pandas as pd
import time


def load_processed_app_ids(results_file):
    directory = os.path.dirname(results_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    if not os.path.exists(results_file):
        with open(results_file, 'w') as file:
            json.dump([], file)
        return set(), []
    
    with open(results_file, 'r') as file:
        results = json.load(file)
    return {result['appId'] for result in results}, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_tokens", type=int, default=4096, help="the max length of LLMs output.")
    parser.add_argument("--temperature_exploration", type=float, default=0.7, help="the temperature in exploration stage.")
    parser.add_argument("--temperature_reasoning", type=float, default=0, help="the temperature in reasoning stage.")
    parser.add_argument("--cg_nodes", type=int, default=50, help="choose the number of cg nodes.")
    parser.add_argument("--path_each_agent", type=int, default=10, help="choose the number of paths each agent.")
    parser.add_argument("--max_path_length", type=int, default=50, help="choose the max length of path.")
    parser.add_argument("--max_unconnected_nodes_pass", type=int, default=100, help="number of unconnected nodes that pass to the synthesizer.")
    parser.add_argument("--cg_agent_num", type=int, default=1, help="the number of cg agents.")
    parser.add_argument("--LLM_type_generate", type=str, default="gpt-4o", help="base LLM model.")
    parser.add_argument("--LLM_type_extract", type=str, default="gpt-3.5-turbo-0125", help="base LLM model.")
    parser.add_argument("--num_retain_entity", type=int, default=5, help="Number of entities retained during entities search.")
    parser.add_argument("--batch_index", type=int, default=0, required=False, help="Index of the current batch.")
    parser.add_argument("--total_batches", type=int, default=1, required=False, help="Total number of batches.")
    parser.add_argument("--openai_api_keys", type=str, 
                        default="your_openai_api_key", help="OpenAI API key.")
    parser.add_argument("--cg_path", type=str, help="the path of CG.")
    parser.add_argument("--csv_path", type=str, help="the path of the csv file.")
    parser.add_argument("--cg_ori_path", type=str, help="the path of CG.")
    parser.add_argument("--results_file", type=str, help="the path to the results json file.")
    parser.add_argument("--embedding_json_path", type=str, help="the path to offline embedding results.")
    parser.add_argument("--KB_csv_path", type=str, help="the path to knowledge base csv.")
    parser.add_argument("--pp_seg_path", type=str, help="Path to pre-extracted privacy policy segments.")
    parser.add_argument("--few_shot_num", type=int, default=2, required=False, help="Number of few-shot examples.")
    parser.add_argument("--des_gen_prompt_choice", type=int, default=1, help="Choose the description generation prompt.")

    args = parser.parse_args()

    processed_app_ids, existing_results = load_processed_app_ids(args.results_file)

    csv_data_path = args.csv_path
    datas = pd.read_csv(csv_data_path)
    datas['Summary Information Usage and Privacy Practices (detail)'] = datas['Summary Information Usage and Privacy Practices (detail)'].apply(clean_string)
    datas['Summary Information Usage and Privacy Practices (concise)'] = datas['Summary Information Usage and Privacy Practices (concise)'].apply(clean_string)
    datas['dctx_des'] = datas['dctx_des'].apply(clean_string)

    datas_KB = pd.read_csv(args.KB_csv_path)

    topic_keywords = {
        "CALENDAR": ["date", "time", "event", "calendar"],
        "CAMERA": ["camera", "photo", "video"],
        "LOCATION": ["location"],
        "STORAGE": ["storage"],
        "MICROPHONE": ["microphone"],
        "SMS": ["sms"],
        "CONTACT": ["contacts"]
    }

    topic_api_calls = {
        'CALENDAR': ['startViewCalendarEventInManagedProfile', 'viewCalendarEvents', 'addCalendarEvent', 'deleteCalendarEvent', 'editCalendarEvent', 'getCalendarEvent', 'getCalendarEventInstances', 'getCalendarEvents', 'getCalendarInfo', 'getCalendarInstances', 'getCalendarList', 'getCalendarSync', 'getCalendars', 'getEvent', 'getEvents'],
        'CAMERA': ['startPreview', 'setFlashMode', 'startRecording', 'startCapture', 'takePicture', 'startCapture', 'pictureTaken', 'takePhoto'],
        'CONTACT': ['getContact', 'openContact', 'getContactDetails', 'loadContacts', 'loadContact', 'getContacts', 'getContactList', 'loadContactInformation'],
        'LOCATION': ['getCurrentLocation', 'getLastKnownLocation', 'getLatitude', 'getLongitude', 'getProvider', 'getAccuracy', 'getAltitude', 'getBearing', 'getSpeed'],
        'MICROPHONE': ['startRecording', 'stop', 'stopListening', 'startRecord'],
        'SMS': ['getMessageBody', 'receiveSmsMessage'],
        'STORAGE': ['getExternalStorageDirectory', 'getDownloadCacheDirectory', 'getRootDirectory', 'getExternalStorageState']
    }

    data_splits = np.array_split(datas, args.total_batches)
    batch_data = data_splits[args.batch_index]

    for index, data in tqdm(batch_data.iterrows(), total=batch_data.shape[0], desc=f"Processing batch {args.batch_index}"):
        try:
            if data['appId'] in processed_app_ids:
                print(f"Skipping appId {data['appId']}: Already processed.")
                continue
            if pd.isna(data["privacyPolicy text"]) or not data["privacyPolicy text"].strip():
                print(f"Skipping appId {data['appId']}: No privacy policy text.")
                continue
            existing_files = set(os.listdir(args.embedding_json_path))
            if not any(data['appId'] in file for file in existing_files):
                print(f"Skipping appId {data['appId']}: Embedding file does not exist.")
                continue
            cg_file = os.path.join(args.cg_path, data['appId'] + '.dot')
            cg_ori_file = os.path.join(args.cg_ori_path, data['appId'], data['appId'] + '.dot')
            if not os.path.exists(cg_file) or not os.path.exists(cg_ori_file):
                print(f"Skipping appId {data['appId']}: CG file does not exist.")
                continue

            print(f"Processing app_name {data['app_name']}, appId {data['appId']}, pp_category {data['pp_category']} \n")
            
            ground_truth_description = data['Summary Information Usage and Privacy Practices (detail)']
            ground_truth_concise_description = data['Summary Information Usage and Privacy Practices (concise)']
            dctx_description = data['dctx_des']

            matching_row = datas_KB[(datas_KB['appId'] == data['appId']) & (datas_KB['pp_category'] == data['pp_category'])]

            if not matching_row.empty:
                KB_pp_top5 = matching_row['top5_pp_similarity'].values[0]
                KB_layout_top5 = matching_row['top5_layout_similarity'].values[0]
            else:
                raise ValueError(f"No matching row found for appId: {data['appId']} and pp_category: {data['pp_category']}")

            pp_top5_list = KB_pp_top5.split('\n')
            layout_top5_list = KB_layout_top5.split('\n')

            pp_segments_dict_PP_Sim = {}
            layout_elements_dict_PP_Sim = {}
            layout_summary_dict_PP_Sim = {}
            layout_pp_summary_dict_PP_Sim = {}

            pp_segments_dict_Layout_Sim = {}
            layout_elements_dict_Layout_Sim = {}
            layout_summary_dict_Layout_Sim = {}
            layout_pp_summary_dict_Layout_Sim = {}

            for app_id in pp_top5_list:
                matching_row = datas_KB[(datas_KB['appId'] == app_id) & (datas_KB['pp_category'] == data['pp_category'])]
                if not matching_row.empty:
                    pp_segments_dict_PP_Sim[app_id] = matching_row['pp_segments'].values[0]
                    layout_elements_dict_PP_Sim[app_id] = matching_row['layout_elements'].values[0]
                    layout_summary_dict_PP_Sim[app_id] = matching_row['layout_summary'].values[0]
                    layout_pp_summary_dict_PP_Sim[app_id] = matching_row['layout_pp_summary'].values[0]

            for app_id in layout_top5_list:
                matching_row = datas_KB[(datas_KB['appId'] == app_id) & (datas_KB['pp_category'] == data['pp_category'])]
                if not matching_row.empty:
                    pp_segments_dict_Layout_Sim[app_id] = matching_row['pp_segments'].values[0]
                    layout_elements_dict_Layout_Sim[app_id] = matching_row['layout_elements'].values[0]
                    layout_summary_dict_Layout_Sim[app_id] = matching_row['layout_summary'].values[0]
                    layout_pp_summary_dict_Layout_Sim[app_id] = matching_row['layout_pp_summary'].values[0]

            nested_top5_dict = {
                "PP_Sim": {
                    "pp_top5_list": pp_top5_list,
                    "pp_segments": pp_segments_dict_PP_Sim,
                    "layout_elements": layout_elements_dict_PP_Sim,
                    "layout_summary": layout_summary_dict_PP_Sim,
                    "layout_pp_summary": layout_pp_summary_dict_PP_Sim
                },
                "Layout_Sim": {
                    "layout_top5_list": layout_top5_list,
                    "pp_segments": pp_segments_dict_Layout_Sim,
                    "layout_elements": layout_elements_dict_Layout_Sim,
                    "layout_summary": layout_summary_dict_Layout_Sim,
                    "layout_pp_summary": layout_pp_summary_dict_Layout_Sim
                }
            }

            pp_segments, pp_description, pp_keywords = phase1_pp_processing(data['appId'], data['app_name'], data['privacyPolicy text'], data['pp_category'], topic_keywords[data['pp_category']], args)
            pp_propositions, pp_keyphrases = pp_proposition_generation(pp_segments, pp_keywords, data['appId'], data['pp_category'], args)

            string_xml_strings, layout_hardcoded_strings, uncvd_string_xml_strings, uncvd_layout_hardcoded_strings = uncovered_strings_validation(data['appId'], cg_file)
        
            cg_initial_nodes_id, cg_api_call_nodes_id = cg_entity_search(cg_file, topic_api_calls, data, pp_keyphrases, pp_keywords, pp_propositions, args, uncvd_string_xml_strings, uncvd_layout_hardcoded_strings)

            start_time = time.time()
            cg_dot = read_dot(cg_ori_file)
            print(f"Reading .dot files took {time.time() - start_time:.2f} seconds")
            MG_cg_dot = nx.MultiDiGraph(cg_dot)

            cg_path_groups, unconnected_nodes = cg_path_find(MG_cg_dot, cg_initial_nodes_id, cg_api_call_nodes_id, args, args.path_each_agent, args.max_path_length)
            
            all_cg_nodes_info = parse_graph_nodes_ret_allinfo(cg_file)
            cg_description_detail, cg_description_concise = cg_multiagent_processing(cg_path_groups, unconnected_nodes, all_cg_nodes_info, pp_description, pp_propositions, data['app_name'], data['pp_category'], nested_top5_dict, args, args.max_unconnected_nodes_pass)

            result = {
                'appId': data['appId'],
                'pp_category': data['pp_category'],
                'descriptions': {
                    'human_label_detail': ground_truth_description,
                    'human_label_concise': ground_truth_concise_description,
                    'dctx_description': dctx_description,
                    'pp_description': pp_description,
                    'detailed_description': cg_description_detail,
                    'concise_description': cg_description_concise,
                    'propositions': pp_propositions,
                }
            }
            
            existing_results.append(result)

            with open(args.results_file, 'w') as outfile:
                json.dump(existing_results, outfile, indent=4)

        except Exception as e:
            print(f"Error processing appId {data['appId']}: {e}")
            continue


if __name__ == '__main__':
    main()
