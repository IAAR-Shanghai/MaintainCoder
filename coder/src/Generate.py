from autogen import GroupChat, GroupChatManager, gather_usage_summary
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from tqdm import tqdm
from openai import OpenAI

from coder.src.utils import extract_python_code
from coder.src.agents import AgentManager
from coder.src.custom_conversable_agent import CustomConversableAgent

class Generate_agent():
    def __init__(self, dataset, save_path, api_key, base_url, model_name, seed=42):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.seed = seed
        self.save_path = save_path # 保存的路径
        self.dataset = dataset

    def generate_answer(self, id, query):
        agent_names = [
            ["Requirement_Analysis_Agent", id],
            ["Design_Pattern_Selection_Agent", id],
            ["Framework_Design_Agent", id],
            ["Supervisor_Agent", id],
            ["Code_Implementation_Agent", id],
            ["Test_Case_Generation_Agent", id],
            ["Coordination_Agent", id],
            ["Code_Execution_Agent", id],
            ["User", id],
            ["Library_Installation_Agent", id],
            ["Code_Modification_Agent", id],
            ["Code_Extraction_Agent", id]
        ]
        agent_manager = AgentManager(agent_names, api_key=self.api_key, base_url=self.base_url, model_name=self.model_name, seed=self.seed)
        Requirement_Analysis_Agent = agent_manager.agents["Requirement_Analysis_Agent"]
        Design_Pattern_Selection_Agent = agent_manager.agents["Design_Pattern_Selection_Agent"]
        Framework_Design_Agent = agent_manager.agents["Framework_Design_Agent"]
        Supervisor_Agent = agent_manager.agents["Supervisor_Agent"]
        Code_Implementation_Agent = agent_manager.agents["Code_Implementation_Agent"]
        Test_Case_Generation_Agent = agent_manager.agents["Test_Case_Generation_Agent"]
        Coordination_Agent = agent_manager.agents["Coordination_Agent"]
        Code_Execution_Agent = agent_manager.agents["Code_Execution_Agent"]
        User = agent_manager.agents["User"]
        Library_Installation_Agent = agent_manager.agents["Library_Installation_Agent"]
        Code_Modification_Agent = agent_manager.agents["Code_Modification_Agent"]
        Code_Extraction_Agent = agent_manager.agents["Code_Extraction_Agent"]

        allowed_transitions = {
            User: [Requirement_Analysis_Agent],
            Requirement_Analysis_Agent: [Design_Pattern_Selection_Agent],
            Design_Pattern_Selection_Agent: [Framework_Design_Agent],
        }
        first_group_chat = GroupChat(
            agents = [User, Requirement_Analysis_Agent, Design_Pattern_Selection_Agent, Framework_Design_Agent],
            allowed_or_disallowed_speaker_transitions=allowed_transitions,
            speaker_transitions_type="allowed",
            max_round = 5,
            send_introductions=True,
            messages=[""],
        )
        first_group_manager = GroupChatManager(
            groupchat = first_group_chat,
            # silent=True
        )
        CustomConversableAgent.clear_history(Requirement_Analysis_Agent)
        CustomConversableAgent.clear_history(Design_Pattern_Selection_Agent)
        CustomConversableAgent.clear_history(Framework_Design_Agent)
        CustomConversableAgent.clear_history(Supervisor_Agent)
        CustomConversableAgent.clear_history(Code_Implementation_Agent)
        CustomConversableAgent.clear_history(Test_Case_Generation_Agent)
        CustomConversableAgent.clear_history(Code_Execution_Agent)
        CustomConversableAgent.clear_history(User)
        CustomConversableAgent.clear_history(Coordination_Agent)
        CustomConversableAgent.clear_history(Library_Installation_Agent)
        CustomConversableAgent.clear_history(Code_Modification_Agent)
        CustomConversableAgent.clear_history(Code_Extraction_Agent)
        ## 代码生成阶段
        input_message = query
        assertion = query.split("\n")[-2]
        chat1 = User.initiate_chat(first_group_manager, message=input_message)
        requirement_analysis = list(Requirement_Analysis_Agent.chat_messages.values())[0][-3]['content']
        framework = CustomConversableAgent.last_message(Framework_Design_Agent)['content']


        CustomConversableAgent.clear_history(Framework_Design_Agent)

        Framework_Design_Agent.initiate_chat(Supervisor_Agent, message=f'Primitive problem:{input_message}\nRequirement Analysis:\n{requirement_analysis}\nFramework:\n{framework}', max_turns=6)
        try:
            framework_final = list(Framework_Design_Agent.chat_messages.values())[0][-2]['content']
        except:
            framework_final = framework

        User.initiate_chat(Code_Implementation_Agent, message=f'Primitive problem:{input_message}\nRequirement Analysis:\n{requirement_analysis}\nFramework:{framework_final}\nTest case:{assertion}\n', max_turns=1)
        code = CustomConversableAgent.last_message(Code_Implementation_Agent)['content']

        ## 测试用例生成与调试阶段
        # User.initiate_chat(Test_Case_Generation_Agent, message=f'{input_message}\n\Requirement Analysis:\n{requirement_analysis}\nCode:\n{code}', max_turns=1)
        # test_case = CustomConversableAgent.last_message(Test_Case_Generation_Agent)['content']
        test_case = "```python\n"+assertion+"\n```"
        input = [code, test_case]
        full_code = '```python\n'+extract_python_code(input)+'\n```'

        # User.initiate_chat(Library_Installation_Agent, message=full_code, max_turns=1)
        # bash_code = CustomConversableAgent.last_message(Library_Installation_Agent)['content']
        Code_Modification_Agent.initiate_chat_with_itr(Code_Execution_Agent, message = f"Primitive problem:\n{input_message}\n\Requirement Analysis:\n{requirement_analysis}\nFramework:\n{framework_final}\nCode:\n{full_code}\nTest case:{assertion}\n", max_turns=5)
        code_final = list(Code_Modification_Agent.chat_messages.values())[0][-2]['content']
        User.initiate_chat(Code_Extraction_Agent, message = code_final, max_turns=1)
        final_code = extract_python_code([CustomConversableAgent.last_message(Code_Extraction_Agent)['content']])
        # final_code = extract_python_code([code_final])
        cost = gather_usage_summary([Requirement_Analysis_Agent, Design_Pattern_Selection_Agent, Framework_Design_Agent, Supervisor_Agent, Code_Implementation_Agent, Test_Case_Generation_Agent, Coordination_Agent, Code_Execution_Agent, User, Library_Installation_Agent, Code_Modification_Agent, Code_Extraction_Agent])['usage_including_cached_inference']['total_cost']
        shutil.rmtree(Code_Execution_Agent._code_execution_config["executor"].work_dir)
        Code_Execution_Agent._code_execution_config["executor"].stop()
        return final_code, cost
    
    def process_problem(self, id):
        query = self.dataset[id]
        # print(f"-------------------STARTING-NUMBER{id}-------------------")
        return self.generate_answer(str(id), query)
    
    def run(self):
        id_list = list(self.dataset.keys())
        total_cost = 0
        error_list = []
        with ThreadPoolExecutor(max_workers=1) as th_executor:
            futures = {th_executor.submit(self.process_problem, id): id for id in id_list}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing problems"):
                id = futures[future]
                try:
                    code, cost = future.result()
                    total_cost += cost
                    with open(self.save_path, 'a', encoding='utf-8') as file:
                        json_record = json.dumps({"id": id, "code": code})
                        file.write(json_record + "\n")
                    # print(f"Completed processing for problem {id}")
                except Exception as e:
                    print(f"Error processing problem {id}: {e}")
                    error_list.append(id)
        print(f"Total cost: {total_cost}")
        return error_list


class Generate_llm():
    def __init__(self, dataset, save_path, prompt_path, api_key, base_url, model_name, prompt_type = "direct"):
        self.save_path = save_path
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.prompt_type = prompt_type
        self.prompt_path = prompt_path
        self.dataset = dataset
        with open(self.prompt_path, "r") as file:
            self.prompt = file.read()
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def generate_answer(self, query, prompt_type):
        if prompt_type == "direct":
            completion = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.3,
                top_p=0.95,
                messages=[
                    {"role": "system", "content": self.prompt},
                    {
                        "role": "user",
                        "content": "Generate python code to solve the above mentioned problem:\n"+query
                    }
                ]
            )
        elif prompt_type == "selfplanning":
            completion = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.3,
                top_p=0.95,
                messages=[
                    {
                        "role": "user",
                        "content": "Break down the solution into smaller steps, explain the approach, identify potential challenges, and decide on the algorithms or data structures that would be useful for this task.\nTask:\n"+query
                    }
                ]
            )
            plan = completion.choices[0].message.content
            completion = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.3,
                top_p=0.95,
                messages=[
                    {"role": "system", "content": self.prompt},
                    {
                        "role": "user",
                        "content": f"Problem:\n{query}\nPlanning:\n{plan}\nGenerate python code without any explanation by following the mentioned plans.\n# ----------------\nImportant: Your response must contain only the python code to solve this problem inside ``` block."
                    }
                ]
            )
        elif prompt_type == "COT":
            completion = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.3,
                top_p=0.95,
                messages=[
                    {"role": "system", "content": self.prompt},
                    {
                        "role": "user",
                        "content": "Let's think step by step and generate python code to solve the problem.\n# ----------------\nImportant: Your response must contain only the python code to solve this problem inside ``` block.\n"+query
                    }
                ]
            )
        else:
            raise ValueError("Invalid prompt type")
        # print(completion.choices[0].message.content)
        final_code = extract_python_code([completion.choices[0].message.content])
        return final_code
    
    def process_problem(self, id):
        query = self.dataset[id]
        return self.generate_answer(query, self.prompt_type)
    
    def run(self):
        id_list = list(self.dataset.keys())
        error_list = []
        with ThreadPoolExecutor(max_workers=10) as th_executor:
            futures = {th_executor.submit(self.process_problem, id): id for id in id_list}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing problems"):
                id = futures[future]
                try:
                    code = future.result()
                    with open(self.save_path, 'a', encoding='utf-8') as file:
                        json_record = json.dumps({"id": id, "code": code})
                        file.write(json_record + "\n")
                except Exception as e:
                    print(f"Error processing problem {id}: {e}")
                    error_list.append(id)
        print("All problems processed")
        return error_list

class Generate_agent_no_optimize():
    def __init__(self, dataset, save_path, model_name, seed=42):
        self.model_name = model_name
        self.seed = seed
        self.save_path = save_path # 保存的路径
        self.dataset = dataset

    def generate_answer(self, id, query):
        agent_names = [
            ["Requirement_Analysis_Agent", id],
            ["Design_Pattern_Selection_Agent", id],
            ["Framework_Design_Agent", id],
            ["Supervisor_Agent", id],
            ["Code_Implementation_Agent", id],
            ["Test_Case_Generation_Agent", id],
            ["Coordination_Agent", id],
            ["Code_Execution_Agent", id],
            ["User", id],
            ["Library_Installation_Agent", id],
            ["Code_Modification_Agent", id],
            ["Code_Extraction_Agent", id]
        ]
        agent_manager = AgentManager(agent_names, api_key=self.api_key, base_url=self.base_url, model_name=self.model_name, seed=self.seed)
        Requirement_Analysis_Agent = agent_manager.agents["Requirement_Analysis_Agent"]
        Design_Pattern_Selection_Agent = agent_manager.agents["Design_Pattern_Selection_Agent"]
        Framework_Design_Agent = agent_manager.agents["Framework_Design_Agent"]
        Supervisor_Agent = agent_manager.agents["Supervisor_Agent"]
        Code_Implementation_Agent = agent_manager.agents["Code_Implementation_Agent"]
        Test_Case_Generation_Agent = agent_manager.agents["Test_Case_Generation_Agent"]
        Coordination_Agent = agent_manager.agents["Coordination_Agent"]
        Code_Execution_Agent = agent_manager.agents["Code_Execution_Agent"]
        User = agent_manager.agents["User"]
        Library_Installation_Agent = agent_manager.agents["Library_Installation_Agent"]
        Code_Modification_Agent = agent_manager.agents["Code_Modification_Agent"]
        Code_Extraction_Agent = agent_manager.agents["Code_Extraction_Agent"]

        allowed_transitions = {
            User: [Requirement_Analysis_Agent],
            Requirement_Analysis_Agent: [Design_Pattern_Selection_Agent],
            Design_Pattern_Selection_Agent: [Framework_Design_Agent],
        }
        first_group_chat = GroupChat(
            agents = [User, Requirement_Analysis_Agent, Design_Pattern_Selection_Agent, Framework_Design_Agent],
            allowed_or_disallowed_speaker_transitions=allowed_transitions,
            speaker_transitions_type="allowed",
            max_round = 5,
            send_introductions=True,
            messages=[""],
        )
        first_group_manager = GroupChatManager(
            groupchat = first_group_chat,
            silent=True
        )
        CustomConversableAgent.clear_history(Requirement_Analysis_Agent)
        CustomConversableAgent.clear_history(Design_Pattern_Selection_Agent)
        CustomConversableAgent.clear_history(Framework_Design_Agent)
        CustomConversableAgent.clear_history(Supervisor_Agent)
        CustomConversableAgent.clear_history(Code_Implementation_Agent)
        CustomConversableAgent.clear_history(Test_Case_Generation_Agent)
        CustomConversableAgent.clear_history(Code_Execution_Agent)
        CustomConversableAgent.clear_history(User)
        CustomConversableAgent.clear_history(Coordination_Agent)
        CustomConversableAgent.clear_history(Library_Installation_Agent)
        CustomConversableAgent.clear_history(Code_Modification_Agent)
        CustomConversableAgent.clear_history(Code_Extraction_Agent)
        ## 代码生成阶段
        input_message = query
        assertion = query.split("\n")[-2]
        chat1 = User.initiate_chat(first_group_manager, message=input_message)
        requirement_analysis = list(Requirement_Analysis_Agent.chat_messages.values())[0][-3]['content']
        framework = CustomConversableAgent.last_message(Framework_Design_Agent)['content']


        CustomConversableAgent.clear_history(Framework_Design_Agent)

        # Framework_Design_Agent.initiate_chat(Supervisor_Agent, message=f'Primitive problem:{input_message}\nRequirement Analysis:\n{requirement_analysis}\nFramework:\n{framework}', max_turns=6)
        # try:
        #     framework_final = list(Framework_Design_Agent.chat_messages.values())[0][-2]['content']
        # except:
        #     framework_final = framework

        framework_final = framework

        User.initiate_chat(Code_Implementation_Agent, message=f'Primitive problem:{input_message}\nRequirement Analysis:\n{requirement_analysis}\nFramework:{framework_final}\nTest case:{assertion}\n', max_turns=1)
        code = CustomConversableAgent.last_message(Code_Implementation_Agent)['content']

        ## 测试用例生成与调试阶段
        # User.initiate_chat(Test_Case_Generation_Agent, message=f'{input_message}\n\Requirement Analysis:\n{requirement_analysis}\nCode:\n{code}', max_turns=1)
        # test_case = CustomConversableAgent.last_message(Test_Case_Generation_Agent)['content']
        test_case = "```python\n"+assertion+"\n```"
        input = [code, test_case]
        full_code = '```python\n'+extract_python_code(input)+'\n```'

        # User.initiate_chat(Library_Installation_Agent, message=full_code, max_turns=1)
        # bash_code = CustomConversableAgent.last_message(Library_Installation_Agent)['content']
        Code_Modification_Agent.initiate_chat_with_itr(Code_Execution_Agent, message = f"Primitive problem:\n{input_message}\n\Requirement Analysis:\n{requirement_analysis}\nFramework:\n{framework_final}\nCode:\n{full_code}\nTest case:{assertion}\n", max_turns=5)
        code_final = list(Code_Modification_Agent.chat_messages.values())[0][-2]['content']
        User.initiate_chat(Code_Extraction_Agent, message = code_final, max_turns=1)
        final_code = extract_python_code([CustomConversableAgent.last_message(Code_Extraction_Agent)['content']])
        # final_code = extract_python_code([code_final])
        cost = gather_usage_summary([Requirement_Analysis_Agent, Design_Pattern_Selection_Agent, Framework_Design_Agent, Supervisor_Agent, Code_Implementation_Agent, Test_Case_Generation_Agent, Coordination_Agent, Code_Execution_Agent, User, Library_Installation_Agent, Code_Modification_Agent, Code_Extraction_Agent])['usage_including_cached_inference']['total_cost']
        # cost = 0
        shutil.rmtree(Code_Execution_Agent._code_execution_config["executor"].work_dir)
        Code_Execution_Agent._code_execution_config["executor"].stop()
        return final_code, cost
    
    def process_problem(self, id):
        query = self.dataset[id]
        # print(f"-------------------STARTING-NUMBER{id}-------------------")
        return self.generate_answer(str(id), query)
    
    def run(self):
        id_list = list(self.dataset.keys())
        total_cost = 0
        error_list = []
        with ThreadPoolExecutor(max_workers=5) as th_executor:
            futures = {th_executor.submit(self.process_problem, id): id for id in id_list}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing problems"):
                id = futures[future]
                try:
                    code, cost = future.result()
                    total_cost += cost
                    with open(self.save_path, 'a', encoding='utf-8') as file:
                        json_record = json.dumps({"id": id, "code": code})
                        file.write(json_record + "\n")
                    # print(f"Completed processing for problem {id}")
                except Exception as e:
                    print(f"Error processing problem {id}: {e}")
                    error_list.append(id)
        print(f"Total cost: {total_cost}")
        return error_list
    

class Generate_agent_no_iteration():
    def __init__(self, dataset, save_path, model_name, seed=42):
        self.model_name = model_name
        self.seed = seed
        self.save_path = save_path # 保存的路径
        self.dataset = dataset

    def generate_answer(self, id, query):
        agent_names = [
            ["Requirement_Analysis_Agent", id],
            ["Design_Pattern_Selection_Agent", id],
            ["Framework_Design_Agent", id],
            ["Supervisor_Agent", id],
            ["Code_Implementation_Agent", id],
            ["Test_Case_Generation_Agent", id],
            ["Coordination_Agent", id],
            ["Code_Execution_Agent", id],
            ["User", id],
            ["Library_Installation_Agent", id],
            ["Code_Modification_Agent", id],
            ["Code_Extraction_Agent", id]
        ]
        agent_manager = AgentManager(agent_names, api_key=self.api_key, base_url=self.base_url, model_name=self.model_name, seed=self.seed)
        Requirement_Analysis_Agent = agent_manager.agents["Requirement_Analysis_Agent"]
        Design_Pattern_Selection_Agent = agent_manager.agents["Design_Pattern_Selection_Agent"]
        Framework_Design_Agent = agent_manager.agents["Framework_Design_Agent"]
        Supervisor_Agent = agent_manager.agents["Supervisor_Agent"]
        Code_Implementation_Agent = agent_manager.agents["Code_Implementation_Agent"]
        Test_Case_Generation_Agent = agent_manager.agents["Test_Case_Generation_Agent"]
        Coordination_Agent = agent_manager.agents["Coordination_Agent"]
        Code_Execution_Agent = agent_manager.agents["Code_Execution_Agent"]
        User = agent_manager.agents["User"]
        Library_Installation_Agent = agent_manager.agents["Library_Installation_Agent"]
        Code_Modification_Agent = agent_manager.agents["Code_Modification_Agent"]
        Code_Extraction_Agent = agent_manager.agents["Code_Extraction_Agent"]

        allowed_transitions = {
            User: [Requirement_Analysis_Agent],
            Requirement_Analysis_Agent: [Design_Pattern_Selection_Agent],
            Design_Pattern_Selection_Agent: [Framework_Design_Agent],
        }
        first_group_chat = GroupChat(
            agents = [User, Requirement_Analysis_Agent, Design_Pattern_Selection_Agent, Framework_Design_Agent],
            allowed_or_disallowed_speaker_transitions=allowed_transitions,
            speaker_transitions_type="allowed",
            max_round = 5,
            send_introductions=True,
            messages=[""],
        )
        first_group_manager = GroupChatManager(
            groupchat = first_group_chat,
            silent=True
        )
        CustomConversableAgent.clear_history(Requirement_Analysis_Agent)
        CustomConversableAgent.clear_history(Design_Pattern_Selection_Agent)
        CustomConversableAgent.clear_history(Framework_Design_Agent)
        CustomConversableAgent.clear_history(Supervisor_Agent)
        CustomConversableAgent.clear_history(Code_Implementation_Agent)
        CustomConversableAgent.clear_history(Test_Case_Generation_Agent)
        CustomConversableAgent.clear_history(Code_Execution_Agent)
        CustomConversableAgent.clear_history(User)
        CustomConversableAgent.clear_history(Coordination_Agent)
        CustomConversableAgent.clear_history(Library_Installation_Agent)
        CustomConversableAgent.clear_history(Code_Modification_Agent)
        CustomConversableAgent.clear_history(Code_Extraction_Agent)
        ## 代码生成阶段
        input_message = query
        assertion = query.split("\n")[-2]
        chat1 = User.initiate_chat(first_group_manager, message=input_message)
        requirement_analysis = list(Requirement_Analysis_Agent.chat_messages.values())[0][-3]['content']
        framework = CustomConversableAgent.last_message(Framework_Design_Agent)['content']


        CustomConversableAgent.clear_history(Framework_Design_Agent)

        Framework_Design_Agent.initiate_chat(Supervisor_Agent, message=f'Primitive problem:{input_message}\nRequirement Analysis:\n{requirement_analysis}\nFramework:\n{framework}', max_turns=6)
        try:
            framework_final = list(Framework_Design_Agent.chat_messages.values())[0][-2]['content']
        except:
            framework_final = framework

        User.initiate_chat(Code_Implementation_Agent, message=f'Primitive problem:{input_message}\nRequirement Analysis:\n{requirement_analysis}\nFramework:{framework_final}\nTest case:{assertion}\n', max_turns=1)
        code = CustomConversableAgent.last_message(Code_Implementation_Agent)['content']
        final_code = extract_python_code([code])

        ## 测试用例生成与调试阶段
        # User.initiate_chat(Test_Case_Generation_Agent, message=f'{input_message}\n\Requirement Analysis:\n{requirement_analysis}\nCode:\n{code}', max_turns=1)
        # test_case = CustomConversableAgent.last_message(Test_Case_Generation_Agent)['content']
        # test_case = "```python\n"+assertion+"\n```"
        # input = [code, test_case]
        # full_code = '```python\n'+extract_python_code(input)+'\n```'

        # User.initiate_chat(Library_Installation_Agent, message=full_code, max_turns=1)
        # bash_code = CustomConversableAgent.last_message(Library_Installation_Agent)['content']
        # Code_Modification_Agent.initiate_chat_with_itr(Code_Execution_Agent, message = f"Primitive problem:\n{input_message}\n\Requirement Analysis:\n{requirement_analysis}\nFramework:\n{framework_final}\nCode:\n{full_code}\nTest case:{assertion}\n", max_turns=5)
        # code_final = list(Code_Modification_Agent.chat_messages.values())[0][-2]['content']
        # User.initiate_chat(Code_Extraction_Agent, message = code_final, max_turns=1)
        # final_code = extract_python_code([CustomConversableAgent.last_message(Code_Extraction_Agent)['content']])
        # # final_code = extract_python_code([code_final])
        cost = gather_usage_summary([Requirement_Analysis_Agent, Design_Pattern_Selection_Agent, Framework_Design_Agent, Supervisor_Agent, Code_Implementation_Agent, Test_Case_Generation_Agent, Coordination_Agent, Code_Execution_Agent, User, Library_Installation_Agent, Code_Modification_Agent, Code_Extraction_Agent])['usage_including_cached_inference']['total_cost']
        # cost = 0
        shutil.rmtree(Code_Execution_Agent._code_execution_config["executor"].work_dir)
        # Code_Execution_Agent._code_execution_config["executor"].stop()
        return final_code, cost
    
    def process_problem(self, id):
        query = self.dataset[id]
        # print(f"-------------------STARTING-NUMBER{id}-------------------")
        return self.generate_answer(str(id), query)
    
    def run(self):
        id_list = list(self.dataset.keys())
        total_cost = 0
        error_list = []
        with ThreadPoolExecutor(max_workers=5) as th_executor:
            futures = {th_executor.submit(self.process_problem, id): id for id in id_list}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing problems"):
                id = futures[future]
                try:
                    code, cost = future.result()
                    total_cost += cost
                    with open(self.save_path, 'a', encoding='utf-8') as file:
                        json_record = json.dumps({"id": id, "code": code})
                        file.write(json_record + "\n")
                    # print(f"Completed processing for problem {id}")
                except Exception as e:
                    print(f"Error processing problem {id}: {e}")
                    error_list.append(id)
        print(f"Total cost: {total_cost}")
        return error_list