import re
import json
import os
import sys

def parse_dot_to_json(dot_content):
    nodes = []
    lines = dot_content.split('\n')
    current_node = None
    current_content = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if '->' in line:
            continue

        node_match = re.match(r'^\s*(\d+)\s*\[\s*label="(.*)', line)
        if node_match:
            if current_node and current_content:
                processed_node = process_node(current_node, ''.join(current_content))
                if processed_node:
                    nodes.append(process_node(current_node, ''.join(current_content)))
            current_node = node_match.group(1)
            current_content = [node_match.group(2)]
        else:
            if current_node is not None:
                current_content.append(line)

    if current_node and current_content:
        processed_node = process_node(current_node, ''.join(current_content))
        if processed_node:
            nodes.append(process_node(current_node, ''.join(current_content)))

    return nodes

def process_node(node_id, content):
    content = content.rstrip('"];')
    parts = content.split('\\lBody:\\l')
    if len(parts) != 2:
        return None
    signature = parts[0].replace('Signature: ', '').strip()
    body = parts[1].strip()
    return {
        'id': node_id,
        'signature': signature,
        'body': body
    }

def convert_dot_to_json(app_name, base_dir):
    try:
        dot_path = os.path.join(base_dir, "ICCG", f"{app_name}.dot")
        output_path = os.path.join(base_dir, "ICCG_node_json", f"{app_name}.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(dot_path, 'r', encoding='utf-8') as f:
            dot_content = f.read()
        nodes = parse_dot_to_json(dot_content)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(nodes, f, indent=2, ensure_ascii=False)
        print(f"Successfully converted {dot_path} to {output_path}")
        print(f"Total nodes processed: {len(nodes)}")
    except Exception as e:
        print(f"Error processing {app_name}: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(1)
    app_name = sys.argv[1]
    base_dir = sys.argv[2]
    convert_dot_to_json(app_name, base_dir)