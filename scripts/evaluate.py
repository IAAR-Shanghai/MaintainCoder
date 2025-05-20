import json
import os

from coder.src.evaluate.MIscore import calculate_MI
from coder.src.evaluate.passk import Passk
from coder.src.evaluate.similarity import code_sim, ast_sim
from coder.src.utils import load_data_for_pass_raw, load_data_for_pass_dyn
from coder.config import PROJECT_ROOT

bench_root = PROJECT_ROOT / "bench" / "benchmarks"
code_root = PROJECT_ROOT / "coder" / "experiment"

# pass5
def run_pass5_raw(dataset_name, method_name):
    error_path = code_root / dataset_name / "error_detail"
    os.makedirs(error_path, exist_ok=True)
    id_list = list(load_data_for_pass_raw(code_root / dataset_name / "code" / f"{method_name}_1.jsonl",  bench_root / f"{dataset_name}.jsonl").keys())
    pass_list = {}
    for id in id_list:
        pass_list[id] = 5
    for i in range(5):
        dataset_for_eval = load_data_for_pass_raw(code_root / dataset_name / "code" / f"{method_name}_{i}.jsonl",  bench_root / f"{dataset_name}.jsonl")
        evaluator = Passk(dataset_for_eval, slient=True)
        results, failed_list, error_type_list = evaluator.run(max_workers=8)
        for id in failed_list:
            pass_list[id] -= 1
        with open(error_path / f"{method_name}_{i}_error_raw.jsonl", 'w', encoding='utf-8') as file:
            for id in sorted(results.keys()):
                record = results[f'{id}']
                file.write(json.dumps({'id': f'{id}', 'result': record['result'], 'error_type': record['error_type']})+'\n')
    pass_5 = sum([1 for i in pass_list.values() if i !=0]) / len(pass_list)
    return pass_5, pass_list

def run_pass5_dyn(dataset_name, method_name):
    error_path = code_root / dataset_name / "error_detail"
    os.makedirs(error_path, exist_ok=True)
    id_list = list(load_data_for_pass_dyn(code_root / dataset_name / "code" / f"{method_name}_1_dyn.jsonl", bench_root / f"{dataset_name}.jsonl").keys())
    pass_list = {}
    for id in id_list:
        pass_list[id] = 5
    for i in range(5):
        dataset_for_eval = load_data_for_pass_dyn(code_root / dataset_name / "code" / f"{method_name}_{i}_dyn.jsonl",  bench_root / f"{dataset_name}.jsonl")
        evaluator = Passk(dataset_for_eval, slient=True)
        results, failed_list, error_type_list = evaluator.run(max_workers=8)
        for id in failed_list:
            pass_list[id] -= 1
        with open(error_path / f"{method_name}_{i}_error_dyn.jsonl", 'w', encoding='utf-8') as file:
            for id in sorted(results.keys()):
                record = results[f'{id}']
                file.write(json.dumps({'id': f'{id}', 'result': record['result'], 'error_type': record['error_type']})+'\n')
    pass_5 = sum([1 for i in pass_list.values() if i !=0]) / len(pass_list)
    return pass_5, pass_list

# MI相关
def run_cal_MI(dataset_name, method_name):
    MI_score = []
    CC_score = []
    HE_score = []
    SLOC_score = []
    for i in range(5):
        code_path = code_root / dataset_name / "code" / f"{method_name}_{i}.jsonl"
        with open(code_path, 'r', encoding='utf-8') as file:
            for line in file:
                record = json.loads(line)
                id = record['id']
                code = record['code']
                try:
                    mi_score, cc_score, he_score, sloc_score = calculate_MI(code)
                    MI_score.append(mi_score)
                    CC_score.append(cc_score)
                    HE_score.append(he_score)
                    SLOC_score.append(sloc_score)
                except:
                    print(code_path+"-"+id)
    avg_mi_score = sum(MI_score) / len(MI_score) if MI_score else 0
    avg_cc_score = sum(CC_score) / len(CC_score) if CC_score else 0
    avg_he_score = sum(HE_score) / len(HE_score) if HE_score else 0
    avg_sloc_score = sum(SLOC_score) / len(SLOC_score) if SLOC_score else 0
    return avg_mi_score, avg_cc_score, avg_he_score, avg_sloc_score

# 第二轮动态指标
def run_cal_sim(dataset_name, method_name):
    ast_sim_score = []
    code_change_ratio = []
    code_change_abs = []
    sloc_v2 = []
    for i in range(5):
        raw_codes = {}
        raw_code_path = code_root / dataset_name / "code" /  f"{method_name}_{i}.jsonl"
        dyn_code_path = code_root / dataset_name / "code" /  f"{method_name}_{i}_dyn.jsonl"
        with open(raw_code_path, 'r', encoding='utf-8') as file:
            for line in file:
                record = json.loads(line)
                id = record['id']
                raw_codes[id] = record['code']
        with open(dyn_code_path, 'r', encoding='utf-8') as file:
            for line in file:
                record = json.loads(line)
                #if record['id'] == '7-3':
                new_code = record['code']
                raw_code = raw_codes[record['id'].split('-')[0]]
                try:
                    _, _, _, sloc = calculate_MI(new_code)
                    _, _, _, _ = calculate_MI(raw_code)
                    sloc_v2.append(sloc)
                    ast_sim_score.append(ast_sim(raw_code, new_code))
                    code_change_ratio.append(code_sim(raw_code, new_code)[1])
                    code_change_abs.append(code_sim(raw_code, new_code)[0])
                except:
                    print(dyn_code_path+"-"+record['id'])
    avg_ast = sum(ast_sim_score)/len(ast_sim_score)
    avg_code_change_ratio = sum(code_change_ratio)/len(code_change_ratio)
    avg_code_change_abs = sum(code_change_abs)/len(code_change_abs)
    avg_sloc = sum(sloc_v2)/len(sloc_v2)
    return avg_ast, avg_code_change_ratio, avg_code_change_abs, avg_sloc