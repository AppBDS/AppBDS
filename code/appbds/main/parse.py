import re, os, xml.etree.ElementTree as ET

def parse_strings_xml(strings_xml_path):
    s = set()
    tree = ET.parse(strings_xml_path)
    root = tree.getroot()
    for child in root:
        if child.tag == 'string':
            txt = ""
            if child.text:
                txt += child.text
            for sub in child:
                txt += ET.tostring(sub, encoding='unicode')
                if sub.tail:
                    txt += sub.tail
            s.add(txt.strip())
    return s

def parse_graph_contexts_ret_set(dot_path):
    s = set()
    with open(dot_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("digraph") and not line.startswith("}") and "->" not in line:
                m = re.match(r'\d+ \[label="(.+?):\s+(.+?)\((.*?)\), Context: (.+)"\];', line)
                if m:
                    ctx = m.group(4).strip()
                    for c in ctx.split("|#|"):
                        c = c.strip()
                        if c: s.add(c)
    return s

def parse_layout_xmls(layout_dir):
    s = set()
    for root_path, _, files in os.walk(layout_dir):
        for file in files:
            if file.endswith(".xml"):
                path = os.path.join(root_path, file)
                try:
                    tree = ET.parse(path)
                    for elem in tree.getroot().iter():
                        for attr in elem.attrib:
                            if attr.endswith("text"):
                                txt = elem.attrib[attr]
                                if not txt.startswith('@string/'):
                                    s.add(txt)
                except Exception as e:
                    print(f"Error parsing {path}: {e}")
    return s

def uncovered_strings_validation(app_id, cg_file):
    base = 'path/to/apktool_output'
    sx = os.path.join(base, app_id, 'res/values/strings.xml')
    ld = os.path.join(base, app_id, 'res/layout')
    s1 = parse_strings_xml(sx) if os.path.exists(sx) else set()
    s2 = parse_layout_xmls(ld) if os.path.exists(ld) else set()
    ctx = parse_graph_contexts_ret_set(cg_file)
    return list(s1), list(s2), list(s1 - ctx), list(s2 - ctx)

def parse_graph_contexts_ret_dict(dot_path):
    contexts, signatures = {}, {}
    with open(dot_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("digraph") and not line.startswith("}") and "->" not in line:
                m = re.match(r'(\d+) \[label="(.+?):\s+(.+?)\((.*?)\), Context: (.+)"\];', line)
                if m:
                    nid = m.group(1)
                    cls, mth, params = m.group(2), m.group(3), m.group(4)
                    ctx = m.group(5).strip()
                    signatures[nid] = f"{cls}: {mth}({params})"
                    if ctx and ctx != "|#||#|":
                        contexts[nid] = [c.strip() for c in ctx.split("|#|") if c.strip()]
    return contexts, signatures

def parse_graph_nodes_ret_allinfo(dot_path):
    info = {}
    with open(dot_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("digraph") and not line.startswith("}") and "->" not in line:
                m = re.match(r'(\d+)\s+\[label="(.+?):\s+([a-zA-Z0-9_\.]+)\s+([a-zA-Z0-9_\$]+)\((.*?)\), Context: (.*?)"\];', line)
                if m:
                    nid = m.group(1)
                    cls, rtype, mth, params = m.group(2), m.group(3), m.group(4), m.group(5)
                    ctx = m.group(6).strip()
                    if ctx and ctx != "|#||#|":
                        clist = [c.strip() for c in ctx.split("|#|") if c.strip()]
                        cstr = ' | '.join(clist)
                    else:
                        clist, cstr = [], ''
                    info[nid] = {'class_name': cls, 'method_name': mth, 'method_params': params, 'return_type': rtype, 'context_list': clist, 'context_str': cstr}
    return info
