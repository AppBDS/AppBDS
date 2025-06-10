import os
import sys
import re
import json
from typing import Dict, List, Set, Optional, Union
from xml.etree import ElementTree as ET

class ManifestParser:
    """Parse AndroidManifest.xml to extract Activity information."""
    def __init__(self, manifest_path: str):
        self.manifest_path = manifest_path
        self.android_ns = {'android': 'http://schemas.android.com/apk/res/android'}

    def extract_activities(self) -> List[Dict]:
        activities = []
        if not os.path.exists(self.manifest_path):
            return activities
        try:
            tree = ET.parse(self.manifest_path)
            root = tree.getroot()
            for activity in root.findall('.//activity'):
                activity_info = {
                    'name': activity.get(f"{{{self.android_ns['android']}}}name"),
                    'label': activity.get(f"{{{self.android_ns['android']}}}label"),
                    'theme': activity.get(f"{{{self.android_ns['android']}}}theme"),
                    'exported': activity.get(f"{{{self.android_ns['android']}}}exported"),
                    'launch_mode': activity.get(f"{{{self.android_ns['android']}}}launchMode")
                }
                if activity_info['name']:
                    activities.append(activity_info)
        except:
            pass
        return activities

class XMLResourceFinder:
    """Handle XML resource files, including parsing public.xml and strings.xml."""
    def __init__(self, apktool_path: str):
        self.apktool_path = apktool_path
        self.res_path = os.path.join(apktool_path, 'res')
        self.string_cache = {}
        self.public_resources = {}
        self.load_public_xml()
        self.load_strings_xml()

    def load_public_xml(self):
        public_path = os.path.join(self.res_path, 'values', 'public.xml')
        if not os.path.exists(public_path):
            return
        try:
            tree = ET.parse(public_path)
            root = tree.getroot()
            for item in root.findall('public'):
                res_type = item.get('type')
                res_name = item.get('name')
                res_id = item.get('id')
                if not all([res_type, res_name, res_id]):
                    continue
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
        if not os.path.exists(strings_path):
            return
        try:
            tree = ET.parse(strings_path)
            root = tree.getroot()
            for string_elem in root.findall('.//string'):
                name = string_elem.get('name')
                if name and string_elem.text:
                    self.string_cache[name] = string_elem.text
        except:
            pass

    def find_activity_resources(self, activity_name: str) -> Dict:
        base_name = activity_name.split('.')[-1].lower()
        base_name = re.sub('activity$', '', base_name)
        result = {
            'layouts': {},
            'menus': {},
            'values': {},
            'resource_ids': {},
            'strings': {}
        }
        layouts = self.public_resources.get('layout', {})
        for res_name, res_info in layouts.items():
            if self.is_resource_related(res_name, base_name):
                layout_path = os.path.join(self.res_path, 'layout', f"{res_name}.xml")
                if os.path.exists(layout_path):
                    try:
                        with open(layout_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            result['layouts'][res_name] = {
                                'content': content,
                                'id': res_info['id']
                            }
                            string_refs = self.extract_string_refs(content)
                            result['strings'].update(string_refs)
                    except:
                        pass
        menus = self.public_resources.get('menu', {})
        for res_name, res_info in menus.items():
            if self.is_resource_related(res_name, base_name):
                menu_path = os.path.join(self.res_path, 'menu', f"{res_name}.xml")
                if os.path.exists(menu_path):
                    try:
                        with open(menu_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            result['menus'][res_name] = {
                                'content': content,
                                'id': res_info['id']
                            }
                            string_refs = self.extract_string_refs(content)
                            result['strings'].update(string_refs)
                    except:
                        pass
        return result

    def is_resource_related(self, res_name: str, base_name: str) -> bool:
        res_name = res_name.lower()
        words = []
        word_start = 0
        for i in range(1, len(base_name)):
            if base_name[i].isupper():
                words.append(base_name[word_start:i].lower())
                word_start = i
        words.append(base_name[word_start:].lower())
        return all(word in res_name for word in words)

    def extract_string_refs(self, content: str) -> Dict[str, str]:
        result = {}
        string_refs = re.findall(r'@string/(\w+)', content)
        for ref in string_refs:
            if ref in self.string_cache:
                result[ref] = self.string_cache[ref]
        return result

    def get_string_value(self, res_id: str) -> Optional[str]:
        if not res_id:
            return None
        if not res_id.startswith('0x'):
            try:
                res_id = f"0x{int(res_id):08x}"
            except ValueError:
                return None
        for res_type, resources in self.public_resources.items():
            for res_name, res_info in resources.items():
                if res_info['id'] == res_id and res_type == 'string':
                    return self.string_cache.get(res_name)
        return None

class ActivityAnalyzer:
    """Analyzer for Android app activities."""
    def __init__(self, manifest_path: str, call_graph_path: str, apktool_path: str):
        self.manifest_path = manifest_path
        self.call_graph_path = call_graph_path
        self.apktool_path = apktool_path
        self.manifest_parser = ManifestParser(manifest_path)
        self.xml_finder = XMLResourceFinder(apktool_path)
        self.call_graph_nodes = self.load_call_graph()

    def load_call_graph(self) -> List[dict]:
        call_graph_nodes = []
        if not os.path.exists(self.call_graph_path):
            return call_graph_nodes
        try:
            with open(self.call_graph_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'digraph' in content and '{' in content and '}' in content:
                    main_content = content.split('{', 1)[1].rsplit('}', 1)[0]
                    current_node = []
                    is_node = False
                    for line in main_content.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        if '->' in line:
                            continue
                        if re.match(r'^\d+\s*\[', line):
                            if is_node and current_node:
                                node_info = self.parse_call_graph_node(' '.join(current_node))
                                if node_info:
                                    call_graph_nodes.append(node_info)
                            is_node = True
                            current_node = [line]
                            continue
                        if is_node:
                            current_node.append(line)
                            if line.endswith('"];') or line.endswith('];'):
                                node_info = self.parse_call_graph_node(' '.join(current_node))
                                if node_info:
                                    call_graph_nodes.append(node_info)
                                is_node = False
                                current_node = []
                    if is_node and current_node:
                        node_info = self.parse_call_graph_node(' '.join(current_node))
                        if node_info:
                            call_graph_nodes.append(node_info)
        except:
            pass
        return call_graph_nodes

    def parse_call_graph_node(self, node_text: str) -> Optional[dict]:
        node_match = re.search(r'(\d+)\s*\[\s*label="([\s\S]*?)"\s*\];', node_text)
        if not node_match:
            return None
        node_id = node_match.group(1)
        label_content = node_match.group(2)
        signature_match = re.search(r'Signature:\s*<([\s\S]*?)>', label_content)
        signature = signature_match.group(1) if signature_match else None
        body_content = []
        if '\\lBody:\\l' in label_content:
            body_parts = label_content.split('\\lBody:\\l')[1].strip()
            body_content = [line.strip() for line in body_parts.split('\\l') if line.strip()]
        return {
            'id': node_id,
            'signature': signature,
            'body': body_content
        }

    def find_activity_call_graph_nodes(self, activity_name: str) -> List[dict]:
        related_nodes = []
        for node in self.call_graph_nodes:
            if node.get('signature') and activity_name in node['signature']:
                related_nodes.append(node)
        return related_nodes

    def extract_resource_refs_from_code(self, body_content: List[str]) -> Dict[str, Dict]:
        result = {
            'layouts': {},
            'strings': {}
        }
        full_content = ' '.join(body_content)
        layout_resources = self.xml_finder.public_resources.get('layout', {})
        for layout_name, layout_info in layout_resources.items():
            pattern = r'\b' + re.escape(layout_name) + r'\b'
            if re.search(pattern, full_content):
                layout_path = os.path.join(self.xml_finder.res_path, 'layout', f"{layout_name}.xml")
                if os.path.exists(layout_path):
                    try:
                        with open(layout_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            result['layouts'][layout_name] = {
                                'content': content,
                                'id': layout_info['id']
                            }
                    except:
                        pass
        string_resources = self.xml_finder.public_resources.get('string', {})
        for string_name in string_resources:
            pattern = r'\b' + re.escape(string_name) + r'\b'
            if re.search(pattern, full_content) and string_name in self.xml_finder.string_cache:
                result['strings'][string_name] = self.xml_finder.string_cache[string_name]
        return result

    def analyze_all(self, output_dir: str = "output") -> None:
        os.makedirs(output_dir, exist_ok=True)
        activities = self.manifest_parser.extract_activities()
        for activity in activities:
            activity_name = activity['name']
            if not activity_name:
                continue
            result = {
                'manifest_info': activity,
                'resources': self.xml_finder.find_activity_resources(activity_name),
                'call_graph_nodes': []
            }
            call_nodes = self.find_activity_call_graph_nodes(activity_name)
            for node in call_nodes:
                node_info = {
                    'signature': node['signature'],
                    'resource': self.extract_resource_refs_from_code(node['body'])
                }
                result['call_graph_nodes'].append(node_info)
            filename = f"{activity_name.split('.')[-1]}.json"
            output_path = os.path.join(output_dir, filename)
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
            except:
                pass
        print(f"Analysis complete. Results saved in {output_dir}/")

def main():
    if len(sys.argv) != 5:
        sys.exit(1)
    manifest_path = sys.argv[1]
    call_graph_path = sys.argv[2]
    apktool_path = sys.argv[3]
    output_dir = sys.argv[4]
    if not os.path.exists(manifest_path):
        sys.exit(1)
    if not os.path.exists(call_graph_path):
        sys.exit(1)
    if not os.path.exists(apktool_path):
        sys.exit(1)
    analyzer = ActivityAnalyzer(manifest_path, call_graph_path, apktool_path)
    analyzer.analyze_all(output_dir)

if __name__ == "__main__":
    main()