import numpy as np
import json
from tqdm import tqdm
import argparse
from pp_utils import *
from cg_utils import *
from des_gen_utils import *
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
    parser.add_argument("--max_tokens", type=int, default=8192, help="max LLM output length")
    parser.add_argument("--temperature_exploration", type=float, default=0.7, help="exploration stage temperature")
    parser.add_argument("--temperature_reasoning", type=float, default=0.3, help="reasoning stage temperature")
    parser.add_argument("--cg_nodes", type=int, default=10, help="number of cg nodes")
    parser.add_argument("--prop_cg_nodes", type=int, default=4, help="number of branch 3 nodes")
    parser.add_argument("--top_context_nodes", type=int, default=10, help="number of context rich nodes")
    parser.add_argument("--des_gen_based_on_node_or_subgraph", type=str, default="subgraph", help="generate description based on node or subgraph")
    parser.add_argument("--llm_subgraph_node_summary", action="store_true", help="execute subgraph node summary logic")
    parser.add_argument("--LLM_type_generate", type=str, default="gpt-4o", help="base LLM model for generation")
    parser.add_argument("--LLM_type_extract", type=str, default="gpt-4o-mini", help="base LLM model for extraction")
    parser.add_argument("--batch_index", type=int, default=0, required=False, help="current batch index")
    parser.add_argument("--total_batches", type=int, default=1, required=False, help="total number of batches")
    parser.add_argument("--openai_api_keys", type=str, default="YOUR_OPENAI_API_KEY", help="OpenAI API key")
    parser.add_argument("--anthropic_api_keys", type=str, default="YOUR_ANTHROPIC_API_KEY", help="Anthropic API key")
    parser.add_argument("--llm_error_handling_setting", type=str, default="multi_try", help="LLM error handling setting")
    parser.add_argument("--csv_path", type=str, default="path/to/evalset.csv", help="path of the csv file")
    parser.add_argument("--cg_path", type=str, default="path/to/CG", help="path of CG")
    parser.add_argument("--cg_node_json_path", type=str, default="path/to/CG_node_json", help="path of UCG nodes json")
    parser.add_argument("--embedding_json_path", type=str, default="path/to/embeddings_json", help="path to offline embedding results")
    parser.add_argument("--general_subgraph_node_summary_json", type=str, default="path/to/general_subgraph_node_summary_json", help="path of general subgraph node summary jsons")
    parser.add_argument("--general_subgraph_dot_path", type=str, default="path/to/general_subgraph_dot", help="path of general subgraph dot directory")
    parser.add_argument("--general_subgraph_dot_json_path", type=str, default="path/to/general_subgraph_dot_json", help="path of general subgraph json directory")
    parser.add_argument("--spec_pp_subgraph_node_summary_json", type=str, default="path/to/spec_pp_subgraph_node_summary_json", help="path of specific PP subgraph node summary jsons")
    parser.add_argument("--spec_pp_subgraph_dot_path", type=str, default="path/to/spec_pp_subgraph_dot", help="path of specific PP subgraph dot directory")
    parser.add_argument("--spec_pp_subgraph_dot_json_path", type=str, default="path/to/spec_pp_subgraph_dot_json", help="path of specific PP subgraph json directory")
    parser.add_argument("--spec_similarapps_subgraph_node_summary_json", type=str, default="path/to/spec_similarapps_subgraph_node_summary_json", help="path of specific similar apps subgraph node summary jsons")
    parser.add_argument("--spec_similarapps_subgraph_dot_path", type=str, default="path/to/spec_similarapps_subgraph_dot", help="path of specific similar apps subgraph dot directory")
    parser.add_argument("--spec_similarapps_subgraph_dot_json_path", type=str, default="path/to/spec_similarapps_subgraph_dot_json", help="path of specific similar apps subgraph json directory")
    parser.add_argument("--results_file", type=str, default="path/to/results.json", help="path to the results json file")
    parser.add_argument("--KB_csv_path", type=str, default="path/to/kb.csv", help="path to knowledge base csv")
    parser.add_argument("--few_shot_num", type=int, default=4, required=False, help="number of few-shot examples")
    parser.add_argument("--ref_direct_des_gen_prompt_aspect_json", type=str, default="path/to/refined_aspects.json", required=False, help="")
    args = parser.parse_args()

    processed_app_ids, existing_results = load_processed_app_ids(args.results_file)
    datas = pd.read_csv(args.csv_path)
    datas['Summary Information Usage and Privacy Practices (detail)'] = datas['Summary Information Usage and Privacy Practices (detail)'].apply(clean_string)
    datas['Summary Information Usage and Privacy Practices (concise)'] = datas['Summary Information Usage and Privacy Practices (concise)'].apply(clean_string)
    datas['dctx_des'] = datas['dctx_des'].apply(clean_string)
    datas_KB = pd.read_csv(args.KB_csv_path)
    topic_api_calls = {
        'CALENDAR': ['startViewCalendarEventInManagedProfile', 'viewCalendarEvents', 'addCalendarEvent', 'deleteCalendarEvent', 'editCalendarEvent', 'getCalendarEvent', 'getCalendarEventInstances', 'getCalendarEvents', 'getCalendarInfo', 'getCalendarInstances', 'getCalendarList', 'getCalendarSync', 'getCalendars', 'getEvent', 'getEvents'],
        'CAMERA': ['startPreview', 'setFlashMode', 'startRecording', 'startCapture', 'takePicture', 'pictureTaken', 'takePhoto'],
        'CONTACT': ['getContact', 'openContact', 'getContactDetails', 'loadContacts', 'loadContact', 'getContacts', 'getContactList', 'loadContactInformation'],
        'LOCATION': ['getCurrentLocation', 'getLastKnownLocation', 'getLatitude', 'getLongitude', 'getProvider', 'getAccuracy', 'getAltitude', 'getBearing', 'getSpeed'],
        'MICROPHONE': ['startRecording', 'stopListening', 'startRecord'],
        'SMS': ['getMessageBody', 'receiveSmsMessage'],
        'STORAGE': ['getExternalStorageDirectory', 'getDownloadCacheDirectory', 'getRootDirectory', 'getExternalStorageState']
    }
    topic_sdk_interfaces = {
        "LOCATION": ["com.adjust.sdk:adjust-android", "com.amazon.android:aps-admob-adapter", "com.amazon.android:aps-sdk", "com.amplitude:android-sdk", "com.applovin:applovin-sdk", "com.appsflyer:af-android-sdk", "com.braze:android-sdk-location", "com.facebook.android:audience-network-sdk", "com.facebook.android:facebook-android-sdk", "com.flurry.sdk.gx", "com.google.ads.interactivemedia.v3:interactivemedia", "com.google.android.gms:play-services-ads", "com.google.android.gms:play-services-location", "com.google.android.gms:play-services-maps", "com.google.android.gms:play-services-places", "com.google.firebase:firebase-crashlytics", "com.google.firebase:firebase-messaging", "com.google.maps.android:android-maps-utils", "com.here.sdk:here-sdk", "com.indooratlas.android:indooratlas-android-sdk", "com.inmobi.monetization:inmobi-ads-kotlin", "com.ironsource.sdk:mediationsdk", "com.kakao.sdk:v2-navi", "com.mapbox.mapboxsdk:mapbox-android-sdk", "com.mapzen.android:lost", "com.mbridge.msdk.oversea:mbbanner", "com.moengage:moe-android-sdk", "com.moloco.sdk:moloco-sdk", "com.naver.maps:map-sdk", "com.onesignal:onesignal", "com.pangle.global:ads-sdk", "com.sendbird.sdk:sendbird-android-sdk", "com.snap.adkit:adkit", "com.snapchat.kit.sdk:creative", "com.startapp:inapp-sdk", "com.taboola:android-sdk", "com.tealium:library", "com.tencent.mm.opensdk:wechat-sdk-android-without-mta", "com.tenjin:android-sdk", "com.tomtom.sdk:tomtom-navigation", "com.unity3d.ads:unity-ads", "com.vungle:publisher-sdk-android", "com.waze.sdk:waze-sdk", "com.yandex.android:maps.mobile", "com.yandex.android:mobmetricalib", "io.nlopez.smartlocation:library", "org.osmdroid:osmdroid-android"],
        "CALENDAR": ["android.provider.CalendarContract", "biweekly:biweekly", "com.alamkanak.weekview:library", "com.eventful.api:eventful", "com.github.prolificinteractive:material-calendarview", "com.google.android.gms:play-services-calendar", "com.google.apis:google-api-java-client", "com.microsoft.graph:microsoft-graph-java", "com.squareup:android-times-square", "net.fortuna.ical4j:ical4j"],
        "CONTACT": ["android.provider.ContactsContract", "com.auth0.android:auth0", "com.facebook.android:facebook-android-sdk", "com.facebook.android:facebook-login", "com.google.android.gms:play-services-identity", "com.kakao.sdk:v2-talk", "com.linkedin.android.lia:linkedin-android-sdk", "com.microsoft.graph:microsoft-graph-core", "com.microsoft.identity.client:msal", "com.sendbird.sdk:sendbird-android-sdk", "com.tencent.imsdk:imsdk-plus", "com.truecaller.android.sdk:truecaller-sdk", "com.twilio:sync-android", "com.viber.sdk:viber-sdk", "com.vk:android-sdk-core", "io.intercom.android:intercom-sdk"],
        "CAMERA": ["androidx.camera:camera-camera2", "com.camerakit:camerakit", "com.facebook.android:facebook-share", "com.google.android.gms:play-services-mlkit-barcode-scanning", "com.google.android.gms:play-services-vision", "com.google.ar:core", "com.google.zxing:core", "com.journeyapps:zxing-android-embedded", "com.onfido.sdk.capture:onfido-capture-sdk", "com.otaliastudios:cameraview", "com.snapchat.kit.sdk:camera", "com.snapchat.kit.sdk:creative", "com.tencent.mm.opensdk:wechat-sdk-android-without-mta", "com.wonderkiln:camerakit", "io.fotoapparat:fotoapparat", "io.scanbot:sdk-core", "org.opencv:opencv-android"],
        "SMS": ["com.clickatell:clickatell-android-sdk", "com.google.android.gms:play-services-sms", "com.google.firebase:firebase-messaging", "com.infobip:infobip-mobile-messaging-android-sdk", "com.messagebird:messagebird-api", "com.nexmo:client", "com.onesignal:onesignal", "com.plivo:plivo-android-sdk", "com.sinch:android-rtc", "com.textlocal:sdk-android", "com.twilio:twilio-android-sdk", "io.smooch:core"],
        "STORAGE": ["androidx.room:room-runtime", "com.adjust.sdk:adjust-android", "com.adyen.checkout:drop-in", "com.aliyun.dpa:oss-android-sdk", "com.amazonaws:aws-android-sdk-sns", "com.amplifyframework:aws-api", "com.amplitude:android-sdk", "com.android.billingclient:billing", "com.box.sdk:box-android-sdk", "com.braintreepayments.api:card", "com.couchbase.lite:couchbase-lite-android", "com.dropbox.core:dropbox-core-sdk", "com.facebook.android:facebook-android-sdk", "com.github.freshworks:freshchat-android", "com.github.satyan:sugar", "com.google.android.gms:play-services-ads", "com.google.android.gms:play-services-drive", "com.google.android.gms:play-services-wallet", "com.google.code.gson:gson", "com.google.firebase:firebase-config", "com.google.firebase:firebase-crashlytics", "com.google.firebase:firebase-database", "com.google.firebase:firebase-storage", "com.helpshift:android-helpshift-aar", "com.inmobi.monetization:inmobi-ads-kotlin", "com.ironsource.sdk:mediationsdk", "com.ironsource:adqualitysdk", "com.j256.ormlite:ormlite-android", "com.mercadopago.android.px:checkout", "com.michaelpardo:activeandroid", "com.microsoft.azure:azure-storage-android", "com.moengage:moe-android-sdk", "com.paytm.appinvokesdk:appinvokesdk", "com.paytm.easypay:easypay", "com.payumoney.sdkui:plug-n-play", "com.plaid.link:sdk-core", "com.qiniu:qiniu-android-sdk", "com.raizlabs.android:DBFlow-Core", "com.razorpay:checkout", "com.razorpay:customui", "com.revenuecat.purchases:purchases", "com.salesforce.marketingcloud:marketingcloudsdk", "com.salesforce.service:chat-ui", "com.sendbird.sdk:sendbird-android-sdk", "com.snap.adkit:adkit", "com.snapchat.kit.sdk:creative", "com.squareup.moshi:moshi", "com.stripe:stripe-android", "com.tealium:library", "com.tencent.bugly:crashreport", "com.tencent.mm.opensdk:wechat-sdk-android-without-mta", "com.uxcam:uxcam", "com.zendesk:messaging", "com.zendesk:support", "in.juspay:hypersdk", "io.realm:realm-android-library", "io.sentry:sentry-android", "net.zetetic:android-database-sqlcipher", "org.greenrobot:greendao"],
        "MICROPHONE": ["com.github.axet:android-audio-library", "com.google.android.exoplayer:exoplayer-core", "com.google.android.gms:play-services-meet", "com.google.cloud:google-cloud-speech", "com.microsoft.cognitiveservices.speech:client-sdk", "com.sinch:android-rtc", "com.tencent.imsdk:imsdk-plus", "com.twilio.video:twilio-video-android", "com.vonage:client-sdk-android", "io.agora.rtc:full-sdk", "io.intercom.android:intercom-sdk", "org.webrtc:google-webrtc"]
    }
    data_splits = np.array_split(datas, args.total_batches)
    batch_data = data_splits[args.batch_index]
    for _, data in tqdm(batch_data.iterrows(), total=batch_data.shape[0], desc=f"Processing batch {args.batch_index}"):
        try:
            if data['appId'] in processed_app_ids:
                print(f"Skipping appId {data['appId']}: Already processed.")
                continue
            if pd.isna(data["privacyPolicy text"]) or not data["privacyPolicy text"].strip():
                print(f"Skipping appId {data['appId']}: No privacy policy text.")
                continue
            cg_file = os.path.join(args.cg_path, data['appId'] + '.dot')
            cg_node_json = os.path.join(args.cg_node_json_path, data['appId'] + '.json')
            if not os.path.exists(cg_file):
                print(f"Skipping appId {data['appId']}: CG file does not exist.")
                continue
            print(f"Processing app_name {data['app_name']}, appId {data['appId']}, pp_category {data['pp_category']}")
            ground_truth_description = data['Summary Information Usage and Privacy Practices (detail)']
            ground_truth_concise_description = data['Summary Information Usage and Privacy Practices (concise)']
            optimized_summary = data['optimized_summary']
            ground_truth = data['ground_truth']
            ground_truth_list = json.loads(ground_truth)
            dctx_description = data['dctx_des']
            matching_row = datas_KB[(datas_KB['appId'] == data['appId']) & (datas_KB['pp_category'] == data['pp_category'])]
            props_list = json.loads(matching_row['propositions'].iloc[0])
            testapp_propositions = []
            for prop_dict in props_list:
                if isinstance(prop_dict, dict):
                    prop_content = list(prop_dict.values())[0].strip()
                    testapp_propositions.append(prop_content)
            top_similar_apps = []
            try:
                similar_apps_data = json.loads(matching_row['top5_similar_apps'].iloc[0])
                for app_id, similarity in similar_apps_data:
                    top_similar_apps.append(app_id)
            except Exception as e:
                print(f"Error parsing similar apps for {data['appId']}: {e}")
            pp_propositions = testapp_propositions
            if args.des_gen_based_on_node_or_subgraph == "node":
                relevant_activity_node_ids, api_call_node_ids, sdk_call_node_ids, gui_context_node_ids, all_kinds_node_ids = cg_entity_search(
                    cg_file, cg_node_json, topic_api_calls, topic_sdk_interfaces, data, args)
                print("Generating description based on nodes.")
                cg_description_detail, cg_description_concise = des_gen_based_on_node(
                    cg_file, cg_node_json, relevant_activity_node_ids, api_call_node_ids, sdk_call_node_ids,
                    gui_context_node_ids, all_kinds_node_ids, data, pp_description, pp_propositions, nested_top5_dict, args)
                result = {
                    'appId': data['appId'],
                    'pp_category': data['pp_category'],
                    'descriptions': {
                        'human_label_detail': ground_truth_description,
                        'human_label_concise': ground_truth_concise_description,
                        'dctx_description': dctx_description,
                        'pp_description': pp_description,
                        'detailed_description_based_on_node': cg_description_detail,
                        'concise_descriptio_based_on_node': cg_description_concise,
                        'propositions': pp_propositions
                    }
                }
            elif args.des_gen_based_on_node_or_subgraph == "subgraph":
                print("Generating description based on subgraphs.")
                general_subgraph_dot_filename = f"{data['appId']}_{data['pp_category']}.dot"
                general_subgraph_json_filename = f"{data['appId']}_{data['pp_category']}.json"
                spec_pp_subgraph_dot_filename = f"{data['appId']}_{data['pp_category']}_pp.dot"
                spec_pp_subgraph_json_filename = f"{data['appId']}_{data['pp_category']}_pp.json"
                spec_similarapps_subgraph_dot_filename = f"{data['appId']}_{data['pp_category']}_similarapps.dot"
                spec_similarapps_subgraph_json_filename = f"{data['appId']}_{data['pp_category']}_similarapps.json"
                general_subgraph_dot_path = os.path.join(args.general_subgraph_dot_path, general_subgraph_dot_filename)
                general_subgraph_json_path = os.path.join(args.general_subgraph_dot_json_path, general_subgraph_json_filename)
                spec_pp_subgraph_dot_path = os.path.join(args.spec_pp_subgraph_dot_path, spec_pp_subgraph_dot_filename)
                spec_pp_subgraph_json_path = os.path.join(args.spec_pp_subgraph_dot_json_path, spec_pp_subgraph_json_filename)
                spec_similarapps_subgraph_dot_path = os.path.join(args.spec_similarapps_subgraph_dot_path, spec_similarapps_subgraph_dot_filename)
                spec_similarapps_subgraph_json_path = os.path.join(args.spec_similarapps_subgraph_dot_json_path, spec_similarapps_subgraph_json_filename)
                if (os.path.exists(general_subgraph_dot_path) and os.path.exists(general_subgraph_json_path) and
                    os.path.exists(spec_pp_subgraph_dot_path) and os.path.exists(spec_pp_subgraph_json_path) and
                    os.path.exists(spec_similarapps_subgraph_dot_path) and os.path.exists(spec_similarapps_subgraph_json_path)):
                    print("Subgraph and subgraph_dict files exist. Loading them directly...")
                    general_cg_subgraph = nx.drawing.nx_pydot.read_dot(general_subgraph_dot_path)
                    with open(general_subgraph_json_path, 'r', encoding='utf-8') as f:
                        general_cg_subgraph_dict = json.load(f)
                    spec_pp_cg_subgraph = nx.drawing.nx_pydot.read_dot(spec_pp_subgraph_dot_path)
                    with open(spec_pp_subgraph_json_path, 'r', encoding='utf-8') as f:
                        spec_pp_cg_subgraph_dict = json.load(f)
                    spec_similarapps_cg_subgraph = nx.drawing.nx_pydot.read_dot(spec_similarapps_subgraph_dot_path)
                    with open(spec_similarapps_subgraph_json_path, 'r', encoding='utf-8') as f:
                        spec_similarapps_cg_subgraph_dict = json.load(f)
                else:
                    start_time = time.time()
                    relevant_activity_node_ids, api_call_node_ids, sdk_call_node_ids, gui_context_node_ids, all_kinds_node_ids = general_cg_entity_search(
                        cg_file, cg_node_json, topic_api_calls, topic_sdk_interfaces, data, args)
                    relevant_pp_propositions_node_ids_dict = spec_pp_cg_entity_search(
                        cg_file, cg_node_json, pp_propositions, data, args)
                    relevant_similarapps_node_ids_dict = spec_similarapps_cg_entity_search(
                        cg_file, cg_node_json, top_similar_apps, datas_KB, data, args)
                    print(f"Entity search took {time.time() - start_time:.2f} seconds")
                    start_time = time.time()
                    cg_dot = read_dot(cg_file)
                    print(f"Reading .dot files took {time.time() - start_time:.2f} seconds")
                    MG_cg_dot = nx.MultiDiGraph(cg_dot)
                    general_cg_subgraph, general_cg_subgraph_dict = general_cg_subgraph_find(
                        MG_cg_dot, cg_node_json, relevant_activity_node_ids, api_call_node_ids, sdk_call_node_ids,
                        gui_context_node_ids, all_kinds_node_ids, data, args)
                    spec_pp_cg_subgraph, spec_pp_cg_subgraph_dict = spec_pp_cg_subgraph_find(
                        MG_cg_dot, cg_node_json, relevant_pp_propositions_node_ids_dict, relevant_activity_node_ids,
                        api_call_node_ids, sdk_call_node_ids, gui_context_node_ids, all_kinds_node_ids, data, args)
                    spec_similarapps_cg_subgraph, spec_similarapps_cg_subgraph_dict = spec_similarapps_cg_subgraph_find(
                        MG_cg_dot, cg_node_json, relevant_similarapps_node_ids_dict, relevant_activity_node_ids,
                        api_call_node_ids, sdk_call_node_ids, gui_context_node_ids, all_kinds_node_ids, data, args)
                if args.llm_subgraph_node_summary:
                    general_summary_path = os.path.join(args.general_subgraph_node_summary_json, f"{data['appId']}_{data['pp_category']}_general.json")
                    spec_pp_summary_path = os.path.join(args.spec_pp_subgraph_node_summary_json, f"{data['appId']}_{data['pp_category']}_pp.json")
                    spec_similarapps_summary_path = os.path.join(args.spec_similarapps_subgraph_node_summary_json, f"{data['appId']}_{data['pp_category']}_similarapps.json")
                    if all(os.path.exists(path) for path in [general_summary_path, spec_pp_summary_path, spec_similarapps_summary_path]):
                        print("All subgraph summaries already exist. Loading from saved files...")
                        def load_summary(summary_path):
                            try:
                                with open(summary_path, 'r', encoding='utf-8') as f:
                                    summaries = json.load(f)
                                    return {node['id']: node for node in summaries}
                            except Exception as e:
                                print(f"Error loading summary file {summary_path}: {e}")
                                return {}
                        general_subgraphs_w_summary_dict = load_summary(general_summary_path)
                        spec_pp_subgraphs_w_summary_dict = load_summary(spec_pp_summary_path)
                        spec_similarapps_subgraphs_w_summary_dict = load_summary(spec_similarapps_summary_path)
                    else:
                        print("Some subgraph summaries are missing. Generating new summaries...")
                        general_subgraphs_w_summary_dict, spec_pp_subgraphs_w_summary_dict, spec_similarapps_subgraphs_w_summary_dict = subgraph_node_summary_gen(
                            general_cg_subgraph, general_cg_subgraph_dict, spec_pp_cg_subgraph, spec_pp_cg_subgraph_dict,
                            spec_similarapps_cg_subgraph, spec_similarapps_cg_subgraph_dict, cg_file, cg_node_json, data, args)
                    branch1_result, branch2_result, branch3_result, aggregator_text_round1, aggregator_text_round2 = des_gen_based_on_subgraph_parallel(
                        cg_file, cg_node_json, general_subgraphs_w_summary_dict, spec_pp_subgraphs_w_summary_dict,
                        spec_similarapps_subgraphs_w_summary_dict, top_similar_apps, datas_KB, data, pp_propositions, args)
                else:
                    raise NotImplementedError("Subgraph processing without node summary is not implemented yet.")
                result = {
                    'appId': data['appId'],
                    'pp_category': data['pp_category'],
                    'descriptions': {
                        'human_label_detail': ground_truth_description,
                        'human_label_concise': ground_truth_concise_description,
                        'optimized_ground_truth': optimized_summary,
                        'dctx_description': dctx_description,
                        'detailed_description_based_on_subgraph_common_func': branch1_result,
                        'detailed_description_based_on_subgraph_prop': branch2_result,
                        'detailed_description_based_on_subgraph_similar_apps': branch3_result,
                        'final_description_round1': aggregator_text_round1,
                        'final_description_round2': aggregator_text_round2,
                        'propositions': pp_propositions
                    }
                }
            print("Final evaluation:")
            existing_results.append(result)
            with open(args.results_file, 'w') as outfile:
                json.dump(existing_results, outfile, indent=4)
        except Exception as e:
            print(f"Error processing appId {data['appId']}: {e}")
            continue

if __name__ == '__main__':
    main()
