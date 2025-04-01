from coder.src.Generate import Generate_agent, Generate_gpt, Generate_qwen, Generate_ds
import json

def get_task_id_text(file_path, error_list):
    task_id_text = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            if record['task_id'].split('-')[1] == '2' and record['task_id'].split('/')[1].split('-')[0] not in error_list:
                task_id = record['task_id'].split('/')[1].split('-')[0]
                raw_problem = record['raw_problem']
                test_case = record['raw_test_input']
                query = f"""
RAW PROBLEM:\n{raw_problem}
TEST CASE:\n{test_case[0]}
"""
                task_id_text[task_id] = query
    return task_id_text

def get_task_id_text_error(file_path, error_list):
    task_id_text = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            if record['task_id'].split('-')[1] == '2' and record['task_id'].split('/')[1].split('-')[0] in error_list:
                task_id = record['task_id'].split('/')[1].split('-')[0]
                raw_problem = record['raw_problem']
                test_case = record['raw_test_input']
                query = f"""
RAW PROBLEM:\n{raw_problem}
TEST CASE:\n{test_case[0]}
"""
                task_id_text[task_id] = query
    return task_id_text

dataset = get_task_id_text('bench/benchmarks/codecontest_dyn.jsonl', [])
print(len(dataset))
print(list(dataset.keys())[:10])

dataset_raw = get_task_id_text('bench/benchmarks/codecontest_dyn.jsonl', [])
error = [list(dataset_raw.keys()) for i in range(5)]
base_model = "gpt-4o-mini"
dataset_name = 'codecontest'
itr = 0
while True:
    if itr == 5:
        print(error)
        break
    for i in range(1,6):
        dataset = get_task_id_text_error(f'bench/benchmarks/codecontest_dyn.jsonl', error[i-1])
        Generator = Generate_agent(dataset, f'coder/experiment/{dataset_name}/code/agent_{base_model}_{i}.jsonl', base_model, 41+i)
        error_list = Generator.run()
        error[i-1] = error_list
    itr += 1
    if all([not i for i in error]):
        break