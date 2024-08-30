import re
import xml.etree.ElementTree as ET
import os

def parse_strings_xml(strings_xml_path):
    strings_set = set()
    tree = ET.parse(strings_xml_path)
    root = tree.getroot()
    for child in root:
        if child.tag == 'string':
            text_content = ""
            if child.text:
                text_content += child.text
            for sub in child:
                text_content += ET.tostring(sub, encoding='unicode')
                if sub.tail:
                    text_content += sub.tail
            strings_set.add(text_content.strip())
    return strings_set

def parse_graph_contexts_ret_set(dot_path):
    context_set = set()
    with open(dot_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("digraph") and not line.startswith("}") and "->" not in line:
                match = re.match(r'\d+ \[label="(.+?):\s+(.+?)\((.*?)\), Context: (.+)"\];', line)
                if match:
                    context_part = match.group(4).strip()
                    contexts = context_part.split("|#|")
                    for context in contexts:
                        context = context.strip()
                        if context:
                            context_set.add(context)
    return context_set

def parse_layout_xmls(layout_dir):
    hardcoded_strings = set()
    for root_path, dirs, files in os.walk(layout_dir):
        for file in files:
            if file.endswith(".xml"):
                layout_xml_path = os.path.join(root_path, file)
                try:
                    tree = ET.parse(layout_xml_path)
                    root = tree.getroot()
                    for elem in root.iter():
                        for attr in elem.attrib.keys():
                            if 'text' == attr[-4:]:
                                text_value = elem.attrib[attr]
                                if not text_value.startswith('@string/'):
                                    hardcoded_strings.add(text_value)
                except Exception as e:
                    print(f"Error parsing {layout_xml_path}: {e}")
    return hardcoded_strings

def uncovered_strings_validation(app_id, cg_file):
    apktool_output_base = 'your_apktool_output_base_path'

    strings_xml_path = os.path.join(apktool_output_base, app_id, 'res/values/strings.xml')
    layout_dir = os.path.join(apktool_output_base, app_id, 'res/layout')

    if not os.path.exists(strings_xml_path):
        strings_xml_strings = set()
    else:
        strings_xml_strings = parse_strings_xml(strings_xml_path)
    if not os.path.exists(layout_dir):
        layout_hardcoded_strings = set()
    else:
        layout_hardcoded_strings = parse_layout_xmls(layout_dir)
        
    contexts_tuple = parse_graph_contexts_ret_set(cg_file)

    uncvd_string_xml_strings = strings_xml_strings - contexts_tuple
    uncvd_layout_hardcoded_strings = layout_hardcoded_strings - contexts_tuple

    return list(strings_xml_strings), list(layout_hardcoded_strings), list(uncvd_string_xml_strings), list(uncvd_layout_hardcoded_strings)

def parse_graph_contexts_ret_dict(dot_path):
    node_contexts = {}
    node_signatures = {}
    with open(dot_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("digraph") and not line.startswith("}") and "->" not in line:
                match = re.match(r'(\d+) \[label="(.+?):\s+(.+?)\((.*?)\), Context: (.+)"\];', line)
                if match:
                    node_id = match.group(1)
                    class_name = match.group(2)
                    method_name = match.group(3)
                    method_params = match.group(4)
                    context_part = match.group(5).strip()
                    node_signatures[node_id] = f"{class_name}: {method_name}({method_params})"
                    if context_part and context_part != "|#||#|":
                        contexts = context_part.split("|#|")
                        node_contexts[node_id] = [context.strip() for context in contexts if context.strip()]
    return node_contexts, node_signatures

def parse_graph_nodes_ret_allinfo(dot_path):
    node_info = {}
    with open(dot_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("digraph") and not line.startswith("}") and "->" not in line:
                match = re.match(r'(\d+)\s+\[label="(.+?):\s+([a-zA-Z0-9_\.]+)\s+([a-zA-Z0-9_\$]+)\((.*?)\), Context: (.*?)"\];', line)
                if match:
                    node_id = match.group(1)
                    class_name = match.group(2)
                    return_type = match.group(3)
                    method_name = match.group(4)
                    method_params = match.group(5)
                    context_part = match.group(6).strip()

                    if context_part and context_part != "|#||#|":
                        contexts = context_part.split("|#|")
                        context_list = [context.strip() for context in contexts if context.strip()]
                        context_str = ' | '.join(context_list)
                    else:
                        context_list = []
                        context_str = ''

                    node_info[node_id] = {
                        'class_name': class_name,
                        'method_name': method_name,
                        'method_params': method_params,
                        'return_type': return_type,
                        'context_list': context_list,
                        'context_str': context_str
                    }
    return node_info
