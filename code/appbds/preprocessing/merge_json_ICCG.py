import json
import os
import re
import sys
from typing import List, Dict, Any

class JsonGraphEnricher:
    def __init__(self, json_path: str, dot_path: str):
        self.json_path = json_path
        self.dot_path = dot_path
        self.dot_content = self.load_dot()
        self.nodes = self.load_json()
        self.activity_enriched_count = 0
        self.node_enriched_count = 0
        self.node_map = {node['id']: node for node in self.nodes if isinstance(node, dict)}

    def load_json(self) -> List[Dict[str, Any]]:
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []

    def load_dot(self) -> str:
        try:
            with open(self.dot_path, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            return ""

    def find_node_id_by_signature(self, signature: str) -> List[str]:
        escaped_signature = re.escape(signature)
        pattern = fr'^\s*(\d+)\s*\[\s*label="Signature: <[^>]*{escaped_signature}[^>]*>'
        matches = []
        for line in self.dot_content.split('\n'):
            match = re.match(pattern, line)
            if match:
                matches.append(match.group(1))
        return matches

    def enrich_from_activity_jsons(self, activity_json_dir: str):
        for filename in os.listdir(activity_json_dir):
            if not filename.endswith('.json'):
                continue
            try:
                with open(os.path.join(activity_json_dir, filename), 'r', encoding='utf-8') as f:
                    activity_data = json.load(f)
                for cg_node in activity_data.get('call_graph_nodes', []):
                    signature = cg_node['signature']
                    node_ids = self.find_node_id_by_signature(signature)
                    for node_id in node_ids:
                        if node_id in self.node_map:
                            gui_info = {
                                'is_activity': True,
                                'manifest_info': activity_data['manifest_info'],
                                'resources': activity_data['resources']
                            }
                            self.node_map[node_id]['gui_info'] = gui_info
                            self.activity_enriched_count += 1
            except:
                pass

    def enrich_from_node_json(self, node_json_path: str):
        try:
            with open(node_json_path, 'r', encoding='utf-8') as f:
                node_data = json.load(f)
            nodes_to_process = []
            if isinstance(node_data, dict) and "nodes" in node_data:
                nodes_to_process = node_data["nodes"]
            elif isinstance(node_data, list):
                nodes_to_process = node_data
            for node_info in nodes_to_process:
                if not isinstance(node_info, dict):
                    continue
                signature = node_info.get('signature')
                if not signature:
                    continue
                node_ids = self.find_node_id_by_signature(signature)
                for node_id in node_ids:
                    if node_id in self.node_map:
                        cg_info = {
                            'resources': node_info.get('resources', {})
                        }
                        self.node_map[node_id]['cg_info'] = cg_info
                        self.node_enriched_count += 1
        except:
            raise

    def save_enriched_json(self, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.nodes, f, indent=2, ensure_ascii=False)
        print(f"\nEnrichment Statistics:")
        print(f"Nodes enriched from activity JSONs: {self.activity_enriched_count}")
        print(f"Nodes enriched from node JSON: {self.node_enriched_count}")

def main(app_name: str):
    json_path = f"PATH_TO_ICCG_NODE_JSON/{app_name}.json"
    dot_path = f"PATH_TO_ICCG_DOT/{app_name}.dot"
    activity_json_dir = f"PATH_TO_ACTIVITY_JSON/{app_name}"
    node_json_path = f"PATH_TO_NODE_INFO_JSON/{app_name}.json"
    output_path = f"PATH_TO_OUTPUT_JSON/{app_name}.json"

    enricher = JsonGraphEnricher(json_path, dot_path)
    enricher.enrich_from_activity_jsons(activity_json_dir)
    enricher.enrich_from_node_json(node_json_path)
    enricher.save_enriched_json(output_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)
    app_name = sys.argv[1]
    main(app_name)