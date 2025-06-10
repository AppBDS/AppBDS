import os
import json
import pandas as pd
import time
import re
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from datetime import datetime
import openai
from openai import OpenAI

class PPKBBuilder:
    """
    Permission-related Proposition Knowledge Base Builder
    
    This class implements the complete PP-KB construction pipeline as described in the APPBDS paper,
    including proposition generation, external validation, and description synthesis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PP-KB Builder
        
        Args:
            config: Configuration dictionary containing API keys and paths
        """
        self.config = config
        self.openai_api_key = config.get('openai_api_key')
        self.perplexity_api_key = config.get('perplexity_api_key')
        
        # Initialize clients
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.perplexity_client = OpenAI(
            api_key=self.perplexity_api_key,
            base_url="https://api.perplexity.ai"
        )
        
        # Setup directories
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories for outputs"""
        base_dir = self.config.get('base_dir', './pp_kb_data')
        self.analysis_dir = os.path.join(base_dir, 'perplexity_analysis')
        self.log_dir = os.path.join(base_dir, 'logs')
        
        for directory in [self.analysis_dir, self.log_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def run_llm(self, prompt: str, temperature: float = 0.7, 
                max_tokens: int = 4096, engine: str = "gpt-4o", 
                sys_msg: str = "normal") -> str:
        """
        Execute LLM inference with error handling
        
        Args:
            prompt: Input prompt text
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            engine: Model name
            sys_msg: System message type
            
        Returns:
            Generated response text
        """
        system_messages = {
            "des": "You are an AI assistant that helps generate comprehensive and precise app descriptions.",
            "extract": "You are an AI assistant that extracts filter information from text.",
            "normal": "You are a helpful AI assistant."
        }
        system_message = system_messages.get(sys_msg, system_messages["normal"])
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.openai_client.chat.completions.create(
                    model=engine,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed after {max_retries} attempts: {e}")
                    return ""
                print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(2)
        
        return ""
    
    def run_llm_seq(self, prompts: List[str], temperature: float = 0.7,
                    max_tokens: int = 4096, engine: str = "gpt-4o",
                    sys_msg: str = "normal") -> List[str]:
        """
        Execute sequential LLM inference maintaining conversation context
        
        Args:
            prompts: List of prompts to process sequentially
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            engine: Model name
            sys_msg: System message type
            
        Returns:
            List of generated responses
        """
        system_messages = {
            "des": "You are an AI assistant that helps generate comprehensive and precise app descriptions.",
            "extract": "You are an AI assistant that extracts filter information from text.",
            "normal": "You are a helpful AI assistant."
        }
        system_message = system_messages.get(sys_msg, system_messages["normal"])
        
        messages = [{"role": "system", "content": system_message}]
        results = []
        
        for prompt in prompts:
            messages.append({"role": "user", "content": prompt})
            
            try:
                response = self.openai_client.chat.completions.create(
                    model=engine,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                result = response.choices[0].message.content
                results.append(result)
                messages.append({"role": "assistant", "content": result})
                
            except Exception as e:
                print(f"Error in sequential LLM call: {e}")
                results.append("")
                messages.append({"role": "assistant", "content": ""})
                time.sleep(2)
        
        return results
    
    def generate_privacy_propositions(self, pp_segments: str, app_id: str, 
                                    pp_category: str) -> List[Dict[str, str]]:
        """
        Generate privacy propositions from privacy policy segments
        
        Args:
            pp_segments: Privacy policy text segments
            app_id: Application identifier
            pp_category: Privacy category (e.g., LOCATION, CAMERA)
            
        Returns:
            List of proposition dictionaries
        """
        prompt = f"""Privacy Category: {pp_category}

Based on the following privacy policy segments, generate a list of propositions regarding 
the usage of {pp_category} privacy information for this app. 
While creating each proposition, attempt to provide details about specific scenarios, 
functionalities, and operations where the privacy information might be utilized, wherever possible. 
Each proposition should be a concise sentence and should be placed on a new line. 
Ensure that each proposition expresses a distinct idea and avoid outputting multiple propositions 
with similar meanings. 
If the privacy policy provides limited or no information about the use of {pp_category} data, 
use inference to make educated guesses about the app's potential functionalities and their 
relation to {pp_category} data. Clearly indicate when a proposition is based on conjecture.

The output should be formatted as follows:

(Proposition 1's content)
(Proposition 2's content)
...

Privacy Policy Segments:
{pp_segments}
"""
        
        try:
            results = self.run_llm_seq([prompt], sys_msg='extract')
            propositions_result = results[0]
            raw_propositions = [line.strip() for line in propositions_result.split('\n') if line.strip()]
            
            # Convert to required JSON format
            propositions_json = [
                {f'proposition {i+1}': prop} 
                for i, prop in enumerate(raw_propositions)
            ]
            
            return propositions_json
        
        except Exception as e:
            print(f"Error processing app {app_id}: {str(e)}")
            return []
    
    def get_perplexity_completion(self, messages: List[Dict[str, str]], 
                                app_id: str, pp_category: str, 
                                step_name: str, max_retries: int = 3) -> str:
        """
        Get completion from Perplexity AI with logging
        
        Args:
            messages: Conversation messages
            app_id: Application identifier
            pp_category: Privacy category
            step_name: Name of the analysis step
            max_retries: Maximum retry attempts
            
        Returns:
            Generated response text
        """
        interaction_log = {
            "app_id": app_id,
            "pp_category": pp_category,
            "step": step_name,
            "timestamp": datetime.now().isoformat(),
            "messages": messages,
            "response": None,
            "attempts": []
        }
        
        for attempt in range(max_retries):
            try:
                response = self.perplexity_client.chat.completions.create(
                    model="llama-3.1-sonar-small-128k-online",
                    messages=messages,
                )
                
                response_content = response.choices[0].message.content
                
                # Log successful attempt
                interaction_log["attempts"].append({
                    "attempt_number": attempt + 1,
                    "status": "success",
                    "timestamp": datetime.now().isoformat()
                })
                interaction_log["response"] = response_content
                
                # Save interaction log
                log_file = os.path.join(self.log_dir, 
                                      f"{app_id}_{pp_category}_perplexity_analysis.json")
                
                self._save_log(log_file, interaction_log)
                return response_content
                
            except Exception as e:
                interaction_log["attempts"].append({
                    "attempt_number": attempt + 1,
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                
                if attempt == max_retries - 1:
                    print(f"Failed after {max_retries} attempts: {e}")
                    log_file = os.path.join(self.log_dir, 
                                          f"{app_id}_{pp_category}_perplexity_analysis.json")
                    self._save_log(log_file, interaction_log)
                    return ""
                    
                print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
        
        return ""
    
    def _save_log(self, log_file: str, interaction_log: Dict):
        """Save interaction log to file"""
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    existing_log = json.load(f)
                    if not isinstance(existing_log, list):
                        existing_log = [existing_log]
                    existing_log.append(interaction_log)
                    final_log = existing_log
            else:
                final_log = [interaction_log]
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(final_log, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving log: {e}")
    
    def analyze_app_function(self, app_id: str, pp_category: str) -> str:
        """
        Analyze app functionality using external search
        
        Args:
            app_id: Application identifier
            pp_category: Privacy category
            
        Returns:
            Analysis result text
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a privacy-focused app analyzer. You need to search for and analyze "
                    "apps, focusing on their privacy practices and data usage. Please provide "
                    "detailed, well-structured analysis based on available online information."
                )
            },
            {
                "role": "user",
                "content": f"""First, please search for and identify the app name corresponding to the app ID {app_id}.
Then, using both the app ID and its name, please analyze the app with a focus on its {pp_category} usage.

Please provide:
1. The app name corresponding to this app ID {app_id}
2. Comprehensive description of the app's main functionalities and features
3. Detailed analysis of how this app specifically utilizes {pp_category} data/permission
4. Analysis of potential privacy implications for users regarding {pp_category} usage
5. Any relevant privacy policy details or user agreements related to {pp_category}

Search for the most up-to-date information available online and provide a thorough analysis.
Please clearly state the app name at the beginning of your response."""
            }
        ]
        
        return self.get_perplexity_completion(messages, app_id, pp_category, "app_function_analysis")
    
    def analyze_propositions(self, app_id: str, pp_category: str, 
                           propositions: List[str], app_analysis: str) -> str:
        """
        Analyze propositions accuracy using external validation
        
        Args:
            app_id: Application identifier
            pp_category: Privacy category
            propositions: List of propositions to validate
            app_analysis: Previous app analysis result
            
        Returns:
            Proposition analysis result
        """
        propositions_text = "\n".join([f"{i+1}. {prop}" for i, prop in enumerate(propositions)])
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a privacy-focused app analyzer. You need to evaluate privacy "
                    "statements and verify them against available online information. Provide "
                    "detailed analysis and identify any missing important privacy practices."
                )
            },
            {
                "role": "user",
                "content": f"""Based on online information and the previous analysis, evaluate these privacy propositions for app {app_id} regarding {pp_category} usage.
First, extract the app name from the previous analysis, then use both the app ID and app name for comprehensive information search.

Previous Analysis:
{app_analysis}

Propositions to evaluate:
{propositions_text}

For each proposition:
1. Verify its accuracy using online information (searching with both app ID and app name)
2. Provide detailed information about specific functionality and implementation methods
3. Analyze the completeness of the privacy disclosure
4. Identify any important {pp_category}-related privacy practices not covered

Please clearly reference the app name in your analysis when discussing findings from online sources.
Search for and use the most up-to-date information available for your analysis."""
            }
        ]
        
        return self.get_perplexity_completion(messages, app_id, pp_category, "proposition_analysis")
    
    def read_perplexity_analysis(self, app_id: str, pp_category: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Read previously generated perplexity analysis results
        
        Args:
            app_id: Application identifier
            pp_category: Privacy category
            
        Returns:
            Tuple of (app_analysis, proposition_analysis)
        """
        json_path = os.path.join(self.log_dir, f"{app_id}_{pp_category}_perplexity_analysis.json")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
                return analysis[0]['response'], analysis[1]['response']
        except Exception as e:
            print(f"Error reading analysis for {app_id}_{pp_category}: {e}")
            return None, None
    
    def parse_aspect_detail(self, text: str) -> List[Dict[str, str]]:
        """
        Parse aspect-detail format from LLM output
        
        Args:
            text: Input text containing aspect-detail pairs
            
        Returns:
            List of parsed aspect-detail dictionaries
        """
        text = text.replace('\n', ' ')
        chunks = re.split(r'(?=Aspect\s*\d+\s*:\s*)', text, flags=re.IGNORECASE)
        
        results = []
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            
            match = re.search(r'Aspect\s*\d+\s*:\s*(.*?)Detail\s*:\s*(.*)', chunk, re.IGNORECASE)
            if match:
                aspect_part = match.group(1).strip(' :')
                detail_part = match.group(2).strip(' :')
                results.append({
                    "Aspect": aspect_part.strip(),
                    "Detail": detail_part.strip()
                })
        return results
    
    def format_description_prompt(self, app_id: str, pp_category: str, pp_segments: str,
                                propositions: List[Dict], app_analysis: str, 
                                prop_analysis: str) -> str:
        """
        Format prompt for final description generation
        
        Args:
            app_id: Application identifier
            pp_category: Privacy category
            pp_segments: Privacy policy segments
            propositions: Generated propositions
            app_analysis: App functionality analysis
            prop_analysis: Proposition validation analysis
            
        Returns:
            Formatted prompt string
        """
        prop_texts = [list(prop.values())[0] for prop in propositions]
        props_formatted = "\n".join([f"{i+1}. {prop}" for i, prop in enumerate(prop_texts)])
        
        prompt = f"""Based on the following comprehensive information about an app's privacy practices and functionality, generate a concise technical analysis of how the app uses the specified permission/privacy information.

APP INFORMATION:
App ID: {app_id}
Privacy Category: {pp_category}

PRIVACY POLICY SEGMENTS:
{pp_segments}

PRIVACY PROPOSITIONS:
{props_formatted}

PREVIOUS ANALYSIS:
App Analysis:
{app_analysis}

Proposition Analysis:
{prop_analysis}

TASK:
Generate a technical analysis of how this app implements and uses the {pp_category} permission/privacy information.

CONTENT REQUIREMENTS:
- Focus on concrete functionality descriptions
- Detail specific features using the permission
- Explain technical implementations
- Avoid general statements, security measures, or policy information

OUTPUT FORMAT:
Aspect 1: [Brief Title (less than 5 words)] Detail: [Detailed technical description of how this aspect uses the permission]
Aspect 2: [Brief Title] Detail: [Detailed description]
(continue for all relevant aspects)

Response must be less than 200 words and focus only on technical implementations."""
        
        return prompt
    
    def process_single_app(self, app_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single app through the complete PP-KB pipeline
        
        Args:
            app_data: Dictionary containing app information
            
        Returns:
            Dictionary containing all analysis results
        """
        app_id = app_data['appId']
        pp_category = app_data['pp_category']
        pp_segments = app_data['pp_segments']
        
        print(f"Processing {app_id} for {pp_category}...")
        
        # Step 1: Generate propositions from privacy policy
        print("Step 1: Generating propositions...")
        propositions = self.generate_privacy_propositions(pp_segments, app_id, pp_category)
        
        if not propositions:
            print(f"Failed to generate propositions for {app_id}")
            return {}
        
        # Step 2: Analyze app functionality with external search
        print("Step 2: Analyzing app functionality...")
        app_analysis = self.analyze_app_function(app_id, pp_category)
        time.sleep(3)  # Rate limiting
        
        if not app_analysis:
            print(f"Failed to get app analysis for {app_id}")
            return {}
        
        # Step 3: Validate propositions with external search
        print("Step 3: Validating propositions...")
        prop_contents = [list(prop.values())[0] for prop in propositions]
        prop_analysis = self.analyze_propositions(app_id, pp_category, prop_contents, app_analysis)
        time.sleep(3)  # Rate limiting
        
        if not prop_analysis:
            print(f"Failed to get proposition analysis for {app_id}")
            return {}
        
        # Step 4: Generate final description
        print("Step 4: Generating final description...")
        prompt = self.format_description_prompt(
            app_id, pp_category, pp_segments, propositions, app_analysis, prop_analysis
        )
        
        icl_description = self.run_llm(
            prompt=prompt,
            temperature=0.3,
            max_tokens=1024,
            engine="gpt-4o",
            sys_msg="normal"
        )
        
        # Parse structured output
        parsed_aspects = self.parse_aspect_detail(icl_description)
        
        # Return complete results
        return {
            'appId': app_id,
            'pp_category': pp_category,
            'propositions': propositions,
            'app_analysis': app_analysis,
            'prop_analysis': prop_analysis,
            'icl_description': icl_description,
            'icl_description_json': parsed_aspects
        }
    
    def build_pp_kb(self, input_csv_path: str, output_csv_path: str):
        """
        Build the complete PP-KB from input data
        
        Args:
            input_csv_path: Path to input CSV file
            output_csv_path: Path to output CSV file
        """
        # Read input data
        df = pd.read_csv(input_csv_path)
        
        # Filter relevant apps
        target_apps = df[
            ((df.get('pp_filtered_2', 0) == 1) | (df.get('testset', 0) == 1)) & 
            df['pp_segments'].notna() & 
            df['pp_category'].notna()
        ]
        
        print(f"Processing {len(target_apps)} apps...")
        
        # Initialize new columns
        new_columns = ['propositions', 'app_analysis', 'prop_analysis', 
                      'icl_description', 'icl_description_json']
        for col in new_columns:
            if col not in df.columns:
                df[col] = None
        
        # Process each app
        processed_count = 0
        error_count = 0
        
        for idx, row in tqdm(target_apps.iterrows(), total=len(target_apps), 
                           desc="Building PP-KB"):
            try:
                # Check if already processed
                if pd.notna(row.get('icl_description')):
                    print(f"Skipping {row['appId']} - already processed")
                    continue
                
                # Process the app
                result = self.process_single_app(row.to_dict())
                
                if result:
                    # Update DataFrame
                    df.at[idx, 'propositions'] = json.dumps(result['propositions'])
                    df.at[idx, 'app_analysis'] = result['app_analysis']
                    df.at[idx, 'prop_analysis'] = result['prop_analysis']
                    df.at[idx, 'icl_description'] = result['icl_description']
                    df.at[idx, 'icl_description_json'] = json.dumps(result['icl_description_json'])
                    
                    processed_count += 1
                else:
                    error_count += 1
                
                # Save intermediate results periodically
                if processed_count % 10 == 0:
                    df.to_csv(output_csv_path, index=False)
                    print(f"Saved intermediate results. Processed: {processed_count}, Errors: {error_count}")
                
            except Exception as e:
                print(f"Error processing {row['appId']}: {e}")
                error_count += 1
                continue
        
        # Save final results
        df.to_csv(output_csv_path, index=False)
        
        print(f"\nPP-KB construction completed!")
        print(f"Total processed: {processed_count}")
        print(f"Total errors: {error_count}")
        print(f"Results saved to: {output_csv_path}")


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from file or environment variables
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        Configuration dictionary
    """
    config = {}
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Override with environment variables if available
    config.setdefault('openai_api_key', os.getenv('OPENAI_API_KEY'))
    config.setdefault('perplexity_api_key', os.getenv('PERPLEXITY_API_KEY'))
    config.setdefault('base_dir', os.getenv('PP_KB_BASE_DIR', './pp_kb_data'))
    
    return config


def main():
    """Main function to run PP-KB construction"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Permission-related Proposition Knowledge Base")
    parser.add_argument("--input_csv", required=True, help="Path to input CSV file")
    parser.add_argument("--output_csv", required=True, help="Path to output CSV file")
    parser.add_argument("--config", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Validate required configuration
    if not config.get('openai_api_key'):
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or provide in config file.")
    
    if not config.get('perplexity_api_key'):
        raise ValueError("Perplexity API key not found. Set PERPLEXITY_API_KEY environment variable or provide in config file.")
    
    # Initialize and run PP-KB builder
    builder = PPKBBuilder(config)
    builder.build_pp_kb(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()