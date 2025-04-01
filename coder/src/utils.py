import re
from coder.config import PROJECT_ROOT

def extract_python_code(input):
    """
    Extract Python code snippets from a list of input strings.
    """
    output = []
    for input_string in input:
        code_pattern = re.compile(r'```python(.*?)```', re.DOTALL)
        code_segments = code_pattern.findall(input_string)
        combined_code = '\n'.join(code_segments)
        output.append(combined_code)
    output_code = '\n'.join(output)
    return output_code

def load_prompt(agent_name):
    """
    Load system message from file for a given agent.
    """
    prompt_path = PROJECT_ROOT / "coder" / "src" / "prompt" / f"{agent_name}.txt"
    with open(prompt_path, "r") as file:
        prompt = file.read()
    return prompt
