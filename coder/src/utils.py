import re
import json

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

def get_raw_index(file_path):
    raw_index = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            if record['task_id'].split('-')[1] == '2':
                task_id = record['task_id'].split('/')[1].split('-')[0]
                raw_index.append(task_id)
    return raw_index

def get_dyn_index(file_path):
    dyn_index = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            task_id = record['task_id'].split('/')[1]
            dyn_index.append(task_id)
    return dyn_index

def load_raw_dataset(file_path, index_list: list):
    raw_dataset = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            if record['task_id'].split('-')[1] == '2' and record['task_id'].split('/')[1].split('-')[0] in index_list:
                task_id = record['task_id'].split('/')[1].split('-')[0]
                raw_problem = record['raw_problem']
                test_case = record['raw_test_input']
                query = f"""
RAW PROBLEM:\n{raw_problem}
TEST CASE:\n{test_case[0]}
"""
                raw_dataset[task_id] = query
    return raw_dataset

def load_dyn_dataset(raw_file_path, dataset_file_path, index_list):
    dyn_dataset = {}
    raw_codes = {}
    with open(raw_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            id = record['id'].split('-')[0]
            raw_codes[id] = record['code']
    with open(dataset_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            task_id = record['task_id'].split('/')[1]
            if task_id in index_list:
                raw_problem = record['raw_problem']
                new_problem = record['new_problem']
                input_format = record['input_format']
                output_format = record['output_format']
                raw_code = raw_codes[task_id.split('-')[0]]
                test_case = record['test_input']
                query = f"""
Raw Requirement:\n{raw_problem}\n
Raw Code:\n{raw_code}\n
New Requirement:\n{new_problem}\n
Input Format:\n{input_format}\n
Output Format:\n{output_format}\n
Test Case:\n{test_case}\n
                """
                dyn_dataset[task_id] = query
    return dyn_dataset

def load_data_for_pass_raw(code_path, bench_path):
    dataset = {}
    code = {}
    with open(code_path, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            code[record['id']] = record['code']
    with open(bench_path, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            if record['task_id'].split('/')[1].split('-')[1]=='2':
                task_id = record['task_id'].split('/')[1].split('-')[0]
                task = record['raw_problem']
                new_code = code[task_id]
                test_list = record['raw_test_input']
                dataset[task_id] = [task, new_code, test_list]
    return dataset

def load_data_for_pass_dyn(code_path, bench_path):
    dataset = {}
    code = {}
    with open(code_path, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            code[record['id']] = record['code']
    with open(bench_path, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            task_id = record['task_id'].split('/')[1]
            task = record['new_problem']
            new_code = code[task_id]
            test_list = record['test_input']
            dataset[task_id] = [task, new_code, test_list]
    return dataset