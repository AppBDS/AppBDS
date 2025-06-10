import os
import sys
import re
import json
from typing import Dict, List, Optional
from xml.etree import ElementTree as ET

class ResourceFinder:
    def __init__(self, apktool_path: str):
        self.res_path = os.path.join(apktool_path, 'res')
        self.public_resources = {}
        self.string_cache = {}
        self.load_public_xml()
        self.load_strings_xml()

    def load_public_xml(self):
        public_path = os.path.join(self.res_path, 'values', 'public.xml')
        try:
            tree = ET.parse(public_path)
            for item in tree.getroot().findall('public'):
                res_type = item.get('type')
                res_name = item.get('name')
                res_id = item.get('id')
                if all([res_type, res_name, res_id]):
                    if res_type not in self.public_resources:
                        self.public_resources[res_type] = {}
                    self.public_resources[res_type][res_name] = {
                        'id': res_id,
                        'name': res_name
                    }
        except:
            pass

    def load_strings_xml(self):
        strings_path = os.path.join(self.res_path, 'values', 'strings.xml')
        try:
            tree = ET.parse(strings_path)
            for string_elem in tree.getroot().findall('.//string'):
                name = string_elem.get('name')
                if name and string_elem.text:
                    self.string_cache[name] = string_elem.text
        except:
            pass

class CallGraphAnalyzer:
    def __init__(self, call_graph_path: str, apktool_path: str):
        self.call_graph_path = call_graph_path
        self.resource_finder = ResourceFinder(apktool_path)

    def analyze_node(self, node_id: str, signature: str, body: List[str]) -> Optional[Dict]:
        full_content = ' '.join(body)
        result = {'layouts': {}, 'strings': {}}
        has_refs = False

        for layout_name, layout_info in self.resource_finder.public_resources.get('layout', {}).items():
            if re.search(r'\b' + re.escape(layout_name) + r'\b', full_content):
                has_refs = True
                layout_path = os.path.join(self.resource_finder.res_path, 'layout', f"{layout_name}.xml")
                try:
                    with open(layout_path) as f:
                        result['layouts'][layout_name] = {
                            'content': f.read(),
                            'id': layout_info['id']
                        }
                except:
                    pass

        for string_name in self.resource_finder.public_resources.get('string', {}):
            if re.search(r'\b' + re.escape(string_name) + r'\b', full_content) and string_name in self.resource_finder.string_cache:
                has_refs = True
                result['strings'][string_name] = self.resource_finder.string_cache[string_name]

        if has_refs:
            return {
                'node_id': node_id,
                'signature': signature,
                'resources': result
            }
        return None

    def analyze_call_graph(self) -> List[Dict]:
        nodes_with_refs = []
        total_nodes = 0
        processed_nodes = 0
        
        with open(self.call_graph_path, 'r', encoding='utf-8') as f:
            for line in f:
                if re.match(r'^\d+\s*\[', line.strip()) and '->' not in line:
                    total_nodes += 1

        try:
            with open(self.call_graph_path, 'r', encoding='utf-8') as f:
                content = f.read()
                main_content = content.split('{', 1)[1].rsplit('}', 1)[0]
                
                current_node = []
                is_node = False

                for line in main_content.splitlines():
                    line = line.strip()
                    if not line or '->' in line:
                        continue

                    if re.match(r'^\d+\s*\[', line):
                        if is_node and current_node:
                            node_info = self.parse_node(' '.join(current_node))
                            if node_info:
                                result = self.analyze_node(
                                    node_info['id'],
                                    node_info['signature'],
                                    node_info['body']
                                )
                                if result:
                                    nodes_with_refs.append(result)
                            processed_nodes += 1
                            print(f"\rProcessing nodes: {processed_nodes}/{total_nodes} ({processed_nodes/total_nodes*100:.1f}%)", end='')
                        is_node = True
                        current_node = [line]
                    elif is_node:
                        current_node.append(line)

                if is_node and current_node:
                    node_info = self.parse_node(' '.join(current_node))
                    if node_info:
                        result = self.analyze_node(
                            node_info['id'],
                            node_info['signature'],
                            node_info['body']
                        )
                        if result:
                            nodes_with_refs.append(result)
                    processed_nodes += 1
                    print(f"\rProcessing nodes: {processed_nodes}/{total_nodes} ({processed_nodes/total_nodes*100:.1f}%)", end='')

            print("\nAnalysis complete!")
            print(f"Found {len(nodes_with_refs)} nodes with resource references")
        except:
            pass
        return nodes_with_refs

    def parse_node(self, node_text: str) -> Optional[Dict]:
        node_match = re.search(r'(\d+)\s*\[\s*label="([\s\S]*?)"\s*\];', node_text)
        if not node_match:
            return None

        node_id = node_match.group(1)
        label_content = node_match.group(2)

        signature_match = re.search(r'Signature:\s*<([^>]+)>', label_content)
        body_content = []
        if '\\lBody:\\l' in label_content:
            body_parts = label_content.split('\\lBody:\\l')[1].strip()
            body_content = [line.strip() for line in body_parts.split('\\l') if line.strip()]

        return {
            'id': node_id,
            'signature': signature_match.group(1) if signature_match else None,
            'body': body_content
        }

def main(call_graph_path, apktool_path, output_path):
    analyzer = CallGraphAnalyzer(call_graph_path, apktool_path)
    results = analyzer.analyze_call_graph()
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'nodes': results}, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(1)
    call_graph_path = sys.argv[1]
    apktool_path = sys.argv[2]
    output_path = sys.argv[3]
    main(call_graph_path, apktool_path, output_path)