import os
import json
import re
import openai
import time
from typing import List, Dict

def run_llm_seq(prompts, temperature, openai_api_keys, max_tokens=4096, engine="gpt-4o", sys_msg="normal"):
    if "llama" in engine.lower():
        openai.api_key = "EMPTY"
        openai.api_base = "http://localhost:8000/v1"
        engine = openai.Model.list()["data"][0]["id"]
    else:
        client = openai.OpenAI(api_key=openai_api_keys)
    system_message = "You are a helpful AI assistant."
    messages = [{"role": "system", "content": system_message}]
    results = []
    for prompt in prompts:
        try:
            user_message = {"role": "user", "content": prompt}
            current_messages = messages + [user_message]
            response = client.chat.completions.create(
                model=engine,
                messages=current_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            result = response.choices[0].message.content
            results.append(result)
            messages.append(user_message)
            messages.append({"role": "assistant", "content": result})
        except Exception as e:
            time.sleep(2)
            results.append(f"Error: {str(e)}")
    return results

class ActivityAnalyzer:
    def __init__(self, activity_json_path: str, call_graph_path: str, app_id: str, output_dir: str, dot_output_dir: str):
        self.activity_json_path = activity_json_path
        self.call_graph_path = call_graph_path
        self.app_id = app_id
        self.output_dir = output_dir
        self.dot_output_dir = dot_output_dir
        self.activity_data = None
        self.call_graph_content = None
        self.extracted_info = {
            'activity_name': '',
            'manifest_info': {},
            'resources': {
                'layouts': {},
                'strings': {},
            },
            'nodes': []
        }

    def load_data(self):
        try:
            with open(self.activity_json_path, 'r', encoding='utf-8') as f:
                content = f.read()
            fixed_content = self.fix_invalid_escape_sequences(content)
            self.activity_data = json.loads(fixed_content)
            dot_output_path = os.path.join(self.dot_output_dir, f"{self.app_id}.dot")
            if os.path.exists(dot_output_path):
                with open(dot_output_path, 'r', encoding='utf-8') as f:
                    self.call_graph_content = f.read()
            else:
                with open(self.call_graph_path, 'r', encoding='utf-8') as f:
                    self.call_graph_content = f.read()
        except:
            return False
        return True

    @staticmethod
    def fix_invalid_escape_sequences(json_content: str) -> str:
        return re.sub(r'\\l', r'\\\\l', json_content)

    def extract_node_from_call_graph(self, signature: str) -> dict:
        pattern = fr'\d+\s*\[\s*label="Signature:\s*<{re.escape(signature)}>[^"]*?Body:\\l.*?"\s*\];'
        match = re.search(pattern, self.call_graph_content, re.DOTALL)
        if match:
            node_text = match.group(0)
            signature_pattern = r'Signature:\s*<([^>]+)>'
            signature_match = re.search(signature_pattern, node_text)
            body_pattern = r'Body:\\l(.*?)(?="\s*\];)'
            body_match = re.search(body_pattern, node_text, re.DOTALL)
            return {
                'signature': signature_match.group(1) if signature_match else '',
                'body': body_match.group(1).replace('\\l', '\n') if body_match else ''
            }
        return {}

    def extract_info(self):
        if not self.activity_data:
            return
        self.extracted_info['activity_name'] = self.activity_data.get('manifest_info', {}).get('name', '')
        self.extracted_info['manifest_info'] = self.activity_data.get('manifest_info', {})
        resources = self.activity_data.get('resources', {})
        self.extracted_info['resources']['layouts'] = resources.get('layouts', {})
        self.extracted_info['resources']['strings'] = resources.get('strings', {})
        processed_signatures = set()
        for node in self.activity_data.get('call_graph_nodes', []):
            signature = node.get('signature')
            if not signature or signature in processed_signatures:
                continue
            node_info = self.extract_node_from_call_graph(signature)
            if node_info:
                node_data = {
                    'signature': node_info['signature'],
                    'body': node_info['body'],
                    'resources': {
                        'layouts': node.get('resource', {}).get('layouts', {}),
                        'strings': node.get('resource', {}).get('strings', {})
                    }
                }
                self.extracted_info['nodes'].append(node_data)
                processed_signatures.add(signature)

    def filter_app_info(self, text: str) -> str:
        if text is None:
            return ""
        try:
            app_name = self.activity_data.get('manifest_info', {}).get('label', '')
            if app_name and isinstance(app_name, str):
                text = text.replace(app_name, "this app")
            if self.app_id and isinstance(self.app_id, str):
                text = text.replace(self.app_id, "this app")
            return text
        except:
            return text

    def generate_prompts(self) -> List[str]:
        first_prompt = self.generate_analysis_prompt()
        second_prompt = """Based on your previous analysis, please provide a concise summary of ONLY the specific functionality of this activity. 
Your response should:
1. Be within 150 words
2. Focus only on concrete functionality
3. Exclude any descriptive statements or explanations
4. Avoid mentioning implementation details
5. Be direct and to-the-point"""
        return [first_prompt, second_prompt]

    def generate_analysis_prompt(self) -> str:
        prompt_parts = []
        introduction = f"""I am analyzing an Android activity named {self.extracted_info['activity_name']}.
Please help me understand its functionality based on the following information: code implementation, layout definitions and string resources.
... (omitted for brevity) ...
"""
        prompt_parts.append(introduction)
        if self.extracted_info['resources']['layouts']:
            layouts_section = "\nLayout Resources:\n"
            for layout_name, layout_info in self.extracted_info['resources']['layouts'].items():
                layouts_section += f"\n{layout_name} Layout:\n```xml\n{layout_info['content']}\n```"
            prompt_parts.append(layouts_section)
        if self.extracted_info['resources']['strings']:
            strings_section = "\nString Resources:\n```\n"
            for string_name, string_value in self.extracted_info['resources']['strings'].items():
                strings_section += f"{string_name}: {string_value}\n"
            strings_section += "```"
            prompt_parts.append(strings_section)
        nodes_section = "\nActivity Implementation Details:"
        for node in self.extracted_info['nodes']:
            nodes_section += f"\n\nMethod: {node['signature']}\n"
            nodes_section += "Implementation:\n```java\n"
            nodes_section += node['body'] + "\n```"
            if node['resources']['layouts'] or node['resources']['strings']:
                nodes_section += "\nResource References:\n"
                if node['resources']['layouts']:
                    nodes_section += "Layouts:\n"
                    for layout_name, _ in node['resources']['layouts'].items():
                        nodes_section += f"- {layout_name}\n"
                if node['resources']['strings']:
                    nodes_section += "Strings:\n"
                    for string_name, string_value in node['resources']['strings'].items():
                        nodes_section += f"- {string_name}: {string_value}\n"
        prompt_parts.append(nodes_section)
        conclusion = """(Omitted for brevity)"""
        prompt_parts.append(conclusion)
        return "\n".join(prompt_parts)

    def analyze_with_llm(self, openai_api_key: str):
        prompts = self.generate_prompts()
        filtered_prompts = [self.filter_app_info(p) for p in prompts]
        results = run_llm_seq(
            prompts=filtered_prompts,
            temperature=0.7,
            openai_api_keys=openai_api_key,
            max_tokens=8192,
            engine="gpt-4o"
        )
        activity_name = self.extracted_info['activity_name'].split('.')[-1]
        node_ids = set()
        for signature in [node['signature'] for node in self.extracted_info['nodes']]:
            pattern = fr'(\d+)\s*\[\s*label="Signature:\s*<{re.escape(signature)}>'
            matches = re.finditer(pattern, self.call_graph_content)
            for match in matches:
                node_ids.add(match.group(1))
        output_data = {
            'activity_name': activity_name,
            'prompts': filtered_prompts,
            'responses': results,
            'node_ids': list(node_ids)
        }
        os.makedirs(self.output_dir, exist_ok=True)
        json_path = os.path.join(self.output_dir, f"{self.app_id}_{activity_name}_analysis.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        node_summary = json.dumps({
            'activity_name': activity_name,
            'number_related_node': len(node_ids),
            'detailed_summary': results[0] if len(results) > 0 else '',
            'concise_summary': results[1] if len(results) > 1 else ''
        }, ensure_ascii=False)
        dot_output_path = os.path.join(self.dot_output_dir, f"{self.app_id}.dot")
        if os.path.exists(dot_output_path):
            updated_content = self.call_graph_content
        else:
            updated_content = self.call_graph_content
        for node_id in node_ids:
            pattern = fr'({node_id}(?!\s*->)\s*\[\s*label="[^"]*?)"(\s*\];)'
            replacement = fr'\1\\lGUI_info: {node_summary}\2'
            updated_content = re.sub(pattern, replacement, updated_content, flags=re.MULTILINE | re.DOTALL)
        with open(dot_output_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        return output_data

def process_activity(
    activity_json_path: str,
    call_graph_path: str,
    app_id: str,
    output_dir: str,
    dot_output_dir: str,
    openai_api_key: str
    ):
    analyzer = ActivityAnalyzer(
        activity_json_path,
        call_graph_path,
        app_id,
        output_dir,
        dot_output_dir
    )
    if not analyzer.load_data():
        return None
    analyzer.extract_info()
    activity_name = analyzer.extracted_info['activity_name'].split('.')[-1]
    analysis_filename = f"{app_id}_{activity_name}_analysis.json"
    analysis_filepath = os.path.join(output_dir, analysis_filename)
    if os.path.exists(analysis_filepath):
        return None
    return analyzer.analyze_with_llm(openai_api_key)

def main(app_id):
    base_dir = "BASE_DATA_PATH"
    output_dir = os.path.join(base_dir, "UCG_activity_LLM_sumRes", app_id)
    dot_output_dir = "OUTPUT_DOT_PATH"
    activity_json_dir = os.path.join(base_dir, "window_info_json", app_id)
    call_graph_path = os.path.join(base_dir, "ICCG", f"{app_id}.dot")
    openai_api_key = "YOUR_OPENAI_KEY"
    os.makedirs(dot_output_dir, exist_ok=True)
    for filename in os.listdir(activity_json_dir):
        if filename.endswith('.json'):
            activity_json_path = os.path.join(activity_json_dir, filename)
            try:
                result = process_activity(
                    activity_json_path,
                    call_graph_path,
                    app_id,
                    output_dir,
                    dot_output_dir,
                    openai_api_key
                )
            except:
                pass

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.exit(1)
    app_id = sys.argv[1]
    main(app_id)
