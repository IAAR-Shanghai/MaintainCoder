import os
from autogen import ConversableAgent, gather_usage_summary
from autogen.coding import DockerCommandLineCodeExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
from tqdm import tqdm

from coder.config import DOCKER_WORKSPACE

def generate_main_function(code, test_list):
    main_function = f"""
{code}

def main():
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
    
    main_function += """
    ]
    
    for test in test_cases:
        exec(test)

if __name__ == "__main__":
    main()
"""
    return main_function

class Passk():
    def __init__(self, dataset, slient=True):
        self.dataset = dataset
        self.slient = slient

    def get_task(self, id):
        return self.dataset[id][0]

    def get_code(self, id):
        return self.dataset[id][1]

    def get_test_list(self, id):
        return self.dataset[id][2]
    
    def get_result(self, id, task, code, test_list):
        user = ConversableAgent(
            name=f"User_{id}",
            human_input_mode="ALWAYS",
            silent=self.slient
        )
        folder_path = DOCKER_WORKSPACE / f'{id}'
        os.makedirs(folder_path, exist_ok=True)
        executor = DockerCommandLineCodeExecutor(
            image="python:3.12-slim",  # Execute code using the given docker image name.
            timeout=10,  # Timeout for each code execution in seconds.
            work_dir=folder_path,  # Use the temporary directory to store the code files.
        )
        Code_Execution_Agent = ConversableAgent(
            name=f"Code_Execution_Agent_{id}",
            human_input_mode="NEVER",
            code_execution_config={"executor": executor},
            is_termination_msg=lambda msg: "TERMINATE" in msg["content"].upper(),
            description="The Code Execution Agent is a Python interpreter that executes code snippets provided by the user. This agent can run Python code and display the output.",
            silent=self.slient
        )

        ConversableAgent.clear_history(Code_Execution_Agent)
        test_code = generate_main_function(code, test_list)
        message = f"```python\n{test_code}\n```"
        chat = user.initiate_chat(Code_Execution_Agent, message=message, max_turns=1)
        result = ConversableAgent.last_message(Code_Execution_Agent)['content']
        error_type = result.split("\n")[-2]
        print(error_type)
        cost = gather_usage_summary([Code_Execution_Agent, user])['usage_including_cached_inference']['total_cost']
        shutil.rmtree(Code_Execution_Agent._code_execution_config["executor"].work_dir)
        Code_Execution_Agent._code_execution_config["executor"].stop()
        if "execution succeeded" in result and "failed" not in result.lower():
            return True, cost, error_type
        else:
            return False, cost, error_type
    
    def process_task(self, id):
        task = self.get_task(id)
        code = self.get_code(id)
        test_list = self.get_test_list(id)
        return self.get_result(id, task, code, test_list)
    
    def run(self):
        success = 0
        failed_list = [] # 失败的id列表
        error_list = [] # 异常的id列表
        failed_type_dict = {} # 失败原因字典
        id_list = list(self.dataset.keys())
        with ThreadPoolExecutor(max_workers=10) as th_executor:
            total_cost = 0
            futures = {th_executor.submit(self.process_task, id):id for id in id_list}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing problems"):
                id = futures[future]
                try:
                    result, cost, error_type = future.result()
                    total_cost += cost
                    if result:
                        success += 1
                        failed_type_dict[id] = {"result": result, "error_type": None}
                    else:
                        failed_list.append(id)
                        failed_type_dict[id] = {"result": result, "error_type": error_type}
                except Exception as e:
                    print(f"Error processing problem {id}: {e}")
                    error_list.append(id)
        print(f"Total cost: {total_cost}")
        print(success)
        print(failed_list)
        return error_list, failed_list, failed_type_dict
