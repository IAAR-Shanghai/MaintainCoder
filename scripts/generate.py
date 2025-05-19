import os

from coder.src.Generate import Generate_agent, Generate_llm
from coder.src.utils import load_raw_dataset, get_raw_index, load_dyn_dataset, get_dyn_index
from coder.config import PROJECT_ROOT

def generate_raw_agent(dataset_name, api_key, base_url, model_name):
    if dataset_name not in ["apps_dyn", "codecontest_dyn", "humaneval_dyn", "mbpp_dyn", "xcodeeval_dyn"]:
        raise ValueError("Invalid dataset name. Choose from: apps_dyn, codecontest_dyn, humaneval_dyn, mbpp_dyn, xcodeeval_dyn.")
    dataset_path = PROJECT_ROOT / "bench" / "benchmarks" / f"{dataset_name}.jsonl"
    code_path = PROJECT_ROOT / "coder" / "experiment" / dataset_name / "code"
    os.makedirs(code_path, exist_ok=True)
    index = get_raw_index(dataset_path)
    error = [list(index) for i in range(5)]
    itr = 0
    flag = True
    while True:
        if itr == 5:
            print("Error after 5 iterations. Error list:")
            print(error)
            flag = False
            break
        for i in range(5):
            dataset = load_raw_dataset(dataset_path, error[i])
            save_path = code_path / f"agent_{model_name}_{i}.jsonl"
            generator = Generate_agent(dataset, save_path, api_key, base_url, model_name, 42+i)
            error_list = generator.run()
            error[i] = error_list
        itr += 1
        if all([not i for i in error]):
            break
    return flag

def generate_raw_llm(dataset_name, api_key, base_url, model_name, prompt_type):
    if dataset_name not in ["apps_dyn", "codecontests_dyn", "humaneval_dyn", "mbpp_dyn", "xcodeeval_dyn"]:
        raise ValueError("Invalid dataset name. Choose from: apps_dyn, codecontests_dyn, humaneval_dyn, mbpp_dyn, xcodeeval_dyn.")
    dataset_path = PROJECT_ROOT / "bench" / "benchmarks" / f"{dataset_name}.jsonl"
    code_path = PROJECT_ROOT / "coder" / "experiment" / dataset_name / "code"
    os.makedirs(code_path, exist_ok=True)
    index = get_raw_index(dataset_path)
    error = [list(index) for i in range(5)]
    itr = 0
    flag = True
    while True:
        if itr == 5:
            print("Error after 5 iterations. Error list:")
            print(error)
            flag = False
            break
        for i in range(5):
            dataset = load_raw_dataset(dataset_path, error[i])
            save_path = code_path / f"{model_name}_{prompt_type}_{i}.jsonl"
            prompt_path = PROJECT_ROOT / "coder" / "src" / "prompt" / "other.txt"
            generator = Generate_llm(dataset, save_path, prompt_path, api_key, base_url, model_name, prompt_type)
            error_list = generator.run()
            error[i] = error_list
        itr += 1
        if all([not i for i in error]):
            break
    return flag

def generate_dyn(dataset_name, raw_code_name, api_key, base_url, model_name):
    if dataset_name not in ["apps_dyn", "codecontests_dyn", "humaneval_dyn", "mbpp_dyn", "xcodeeval_dyn"]:
        raise ValueError("Invalid dataset name. Choose from: apps_dyn, codecontests_dyn, humaneval_dyn, mbpp_dyn, xcodeeval_dyn.")
    dataset_path = PROJECT_ROOT / "bench" / "benchmarks" / f"{dataset_name}.jsonl"
    code_path = PROJECT_ROOT / "coder" / "experiment" / dataset_name / "code"
    os.makedirs(code_path, exist_ok=True)
    index = get_dyn_index(dataset_path)
    error = [list(index) for i in range(len(raw_code_name))]
    itr = 0
    flag = True
    while True:
        if itr == 5:
            print("Error after 5 iterations. Error list:")
            print(error)
            flag = False
            break
        for i in range(len(raw_code_name)):
            raw_file_path = code_path / f"{raw_code_name[i]}.jsonl"
            dataset = load_dyn_dataset(raw_file_path, dataset_path, error[i])
            save_path = code_path / f"{raw_code_name[i]}_dyn.jsonl"
            prompt_path = PROJECT_ROOT / "coder" / "src" / "prompt" / f"requirement_change_prompt_old2_{dataset_name}.txt"
            generator = Generate_llm(dataset, save_path, prompt_path, api_key, base_url, model_name, "direct")
            error_list = generator.run()
            error[i] = error_list
        itr += 1
        if all([not i for i in error]):
            break
    return flag