"""
Prompt templates for decompressionLM experiments.

Each template is a lambda function that takes domain/concept parameters
and returns a formatted prompt string.

IMPORTANT: These are the tested, working prompts. Do not truncate!
"""

# Flat sampling prompts - same as graph root
flat_concept_list = lambda domain: f"""Generate concepts in {domain} as keywords.
Please output ONE concept per line.
Each concept can be multiple words if needed.
Do not include explanations or extra text.
Please begin from any random concept.
Please use English.
"""

# Graph exploration prompts
graph_root_prompt = lambda domain: f"""Generate concepts in {domain} as keywords.
Please output ONE concept per line.
Each concept can be multiple words if needed.
Do not include explanations or extra text.
Please begin from any random concept.
Please use English.
"""

graph_child_prompt = lambda concept, domain: f"""Generate concepts related to {concept} in {domain} as keywords.
Please output ONE concept per line.
Each concept can be multiple words if needed.
Do not include explanations or extra text.
Please begin from any random concept.
Please use English.
"""

# Single-sequence graph generation (for beam+graph baseline)
# Note: This uses slightly different wording for the beam search method
beam_graph_prompt = lambda domain: f"""Generate a concept graph for {domain} in YAML format.
Use this structure:
- concept: [concept name]
  related:
    - [related concept 1]
    - [related concept 2]
Each concept should be on its own line with proper YAML formatting.
Do not include explanations.
Please use English.
"""

