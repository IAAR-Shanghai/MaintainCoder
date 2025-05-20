import os
import re
from tqdm import tqdm
import sys
import tempfile
import subprocess
import traceback
import concurrent.futures
from threading import Lock
from tqdm import tqdm  # 添加进度条支持

class Passk():
    def __init__(self, dataset, slient=True):
        self.dataset = dataset
        self.slient = slient
        self.print_lock = Lock()
    
    def generate_main_function(self, code, task_id, test_list):
        task_id = re.sub(r'[^\w]', '_', task_id)
        main_function = f"""
{code}

def test_for_{task_id}():
    test_cases = [
"""
        for test in test_list:
            test = test.replace("\n", "\\n")
            test = test.replace('"', '\\"')
            test = test.replace("\\n", "\\\\n")
            test = test.replace("try:\\\\n", "try:\\n")
            test = test.replace("\\\\nexcept", "\\nexcept")
            test = test.replace("as e:\\\\n", "as e:\\n")
            main_function += f'        "{test}",\n'
        
        main_function += f"""
    ]
    
    for test in test_cases:
        exec(test)

test_for_{task_id}()
"""
        return main_function
    
    def test_task(self, task_data):
        task_id, (task, code, test_list) = task_data
        
        with self.print_lock:
            # print(f"测试任务: {task_id}")
            pass
        
        result = {"passed": False, "error_type": None, "result": False}
        error_info = {}
        
        try:
            # 生成完整测试代码
            full_code = self.generate_main_function(code, task_id, test_list)
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w', encoding='utf-8') as temp_file:
                temp_filename = temp_file.name
                temp_file.write(full_code)
            
            # 执行代码
            try:
                proc_result = subprocess.run(
                    [sys.executable, temp_filename],
                    capture_output=True,
                    text=True,
                    timeout=30  # 设置超时时间
                )
                
                # 检查执行结果
                if proc_result.returncode == 0:
                    result["error_type"] = proc_result.stdout.strip()
                    result["result"] = True
                else:
                    result["result"] = False
                    result["error_type"] = proc_result.stderr.strip()
                    error_info = {
                        "result": "fail",
                        "error_type": "runtime_error",
                        "stderr": proc_result.stderr,
                        "stdout": proc_result.stdout
                    }
            except subprocess.TimeoutExpired:
                result["result"] = False
                result["error_type"] = "timeout"
                error_info = {
                    "result": "fail",
                    "error_type": "timeout"
                }
            
            # 删除临时文件
            os.unlink(temp_filename)
            
        except Exception as e:
            result["result"] = False
            result["error_type"] = "exception"
            error_info = {
                "result": "fail",
                "error_type": "exception",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        
        return task_id, result, error_info

    def run(self, max_workers=8):
        """多线程并行执行测试代码"""
        results = {}
        failed_list = []
        error_type_list = {}
        
        # 创建一个进度条
        pbar = tqdm(total=len(self.dataset), desc="执行测试")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(self.test_task, (task_id, data)): task_id 
                for task_id, data in self.dataset.items()
            }
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_task):
                task_id, result, error_info = future.result()
                results[task_id] = result
                
                # 更新失败列表
                if not result.get("result"):
                    failed_list.append(task_id)
                    if error_info:
                        error_type_list[task_id] = error_info
                
                pbar.update(1)
        
        pbar.close()
        return results, failed_list, error_type_list