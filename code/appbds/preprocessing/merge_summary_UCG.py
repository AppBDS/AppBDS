import os
import json
import sys
from typing import Dict, List

class JsonActivityEnricher:
    def __init__(self, nodes_json_file: str):
        """Initialize and load node JSON data."""
        with open(nodes_json_file, 'r') as f:
            self.nodes = json.load(f)
        self.node_map = {node['id']: node for node in self.nodes if isinstance(node, dict)}

    def process_json_files(self, activity_json_folder: str):
        """Process activity JSON files and update nodes."""
        all_json_files = []
        for root, _, files in os.walk(activity_json_folder):
            for file in files:
                if file.endswith(".json"):
                    all_json_files.append(os.path.join(root, file))

        total_files = len(all_json_files)
        print(f"Total {total_files} files to process.\n")

        for index, json_file in enumerate(all_json_files):
            print(f"Progress: {index + 1}/{total_files} - {json_file}")
            try:
                with open(json_file, 'r') as f:
                    activity_data = json.load(f)

                activity_name = activity_data.get('activity_name', '')
                node_ids = activity_data.get('node_ids', [])
                responses = activity_data.get('responses', [])
                
                if not activity_name or not node_ids:
                    print(f"Skipping {json_file}: Missing activity name or node IDs.")
                    continue

                activity_summary = {
                    'activity_name': activity_name,
                    'number_related_node': len(node_ids),
                    'detailed_summary': responses[0] if len(responses) > 0 else '',
                    'concise_summary': responses[1] if len(responses) > 1 else ''
                }

                updated_count = 0
                for node_id in node_ids:
                    if node_id in self.node_map:
                        node = self.node_map[node_id]
                        if 'gui_info' not in node:
                            node['gui_info'] = {}
                        node['gui_info']['activity_summary'] = activity_summary
                        updated_count += 1

                print(f"Updated activity: {activity_name}, nodes updated: {updated_count}/{len(node_ids)}")

            except Exception as e:
                print(f"Error processing file {json_file}: {str(e)}")

    def save_nodes(self, output_file: str):
        """Save updated node information."""
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.nodes, f, indent=2, ensure_ascii=False)
            print(f"\nSuccessfully saved updated node data to: {output_file}")
        except Exception as e:
            print(f"Error saving file: {str(e)}")

def main(app_name: str):
    """Main function to handle node data updates."""
    nodes_json_file = f'PATH_TO_NODES_JSON/{app_name}.json'
    activity_json_folder = f'PATH_TO_ACTIVITY_JSON/{app_name}'
    nodes_json_output = f'PATH_TO_OUTPUT_JSON/{app_name}.json'
    
    print(f"Processing app: {app_name}")
    enricher = JsonActivityEnricher(nodes_json_file)
    print("Processing activity info...")
    enricher.process_json_files(activity_json_folder)
    enricher.save_nodes(nodes_json_output)
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <app_name>")
        sys.exit(1)
    app_name = sys.argv[1]
    main(app_name)