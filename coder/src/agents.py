import os
from autogen.coding import DockerCommandLineCodeExecutor

from coder.src.utils import load_prompt
from coder.config import DOCKER_WORKSPACE
from coder.src.custom_conversable_agent import CustomConversableAgent


class AgentManager:
    def __init__(self, agent_names, api_key, base_url, model_name, seed=42):
        self.seed = seed
        self.agents = {}
        config_list = [{"model": model_name, "api_key": api_key, "base_url": base_url}]
        self.llm_config = {
            "config_list": config_list,
            'temperature': 0.3,
            'top_p': 0.95,
            'cache_seed': self.seed,
        }
        for name, id in agent_names:
            self.agents[name] = self.create_agent(name, id)

    def create_agent(self, name, id):
        if name == "Code_Execution_Agent":
            folder_path = DOCKER_WORKSPACE / f"{id}"
            os.makedirs(folder_path, exist_ok=True)
            executor = DockerCommandLineCodeExecutor(
                image="python:3.12-slim",  # Execute code using the given docker image name.
                timeout=10,  # Timeout for each code execution in seconds.
                work_dir=folder_path,  # Use the temporary directory to store the code files.
            )
            return CustomConversableAgent(
                name=name+'_'+id,
                human_input_mode="NEVER",
                code_execution_config={"executor": executor},
                is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
                description="The Code Execution Agent is a Python interpreter that executes code snippets provided by the user. This agent can run Python code and display the output.",
                # silent=True
            )
        elif name == "User":
            return CustomConversableAgent(
                name=name+'_'+id,
                human_input_mode="ALWAYS",
                llm_config=self.llm_config,
                # silent=True
            )
        else:
            return CustomConversableAgent(
                name=name+'_'+id,
                system_message=load_prompt(name),
                llm_config=self.llm_config,
                human_input_mode="NEVER",
                is_termination_msg=lambda msg: (
                    "TERMINATE" in msg["content"].upper() or 
                    ("execution succeeded" in msg["content"] if name == "Code_Modification_Agent" else False)
                ) if name in ["Framework_Design_Agent", "Supervisor_Agent", "Code_Implementation_Agent", "Code_Modification_Agent"] else False,
                description=self.get_description(name),
                # silent=True
            )

    def get_description(self, name):
        descriptions = {
            "Requirement_Analysis_Agent": "The Requirement Analysis Agent specializes in dissecting and organizing software requirements into actionable components. With expertise in identifying key challenges and functional modules, this agent ensures that every requirement is mapped to a clear purpose, laying the foundation for efficient and effective system design.",
            "Design_Pattern_Selection_Agent": "The Design Pattern Selection Agent is a seasoned software architect proficient in selecting and applying design patterns. By analyzing functional modules and their challenges, this agent identifies the most suitable design patterns, providing clear justifications for each choice to ensure the system's scalability, maintainability, and robustness.",
            "Framework_Design_Agent": "The Framework Design Agent excels in crafting system architectures by translating functional modules and design patterns into detailed class structures. This agent defines class attributes, methods, and relationships, ensuring a coherent and flexible framework that serves as the backbone of the software system.",
            "Supervisor_Agent": "The Supervisor Agent is responsible for critically evaluating the proposed system design, identifying potential issues, and suggesting optimizations. With a keen eye for detail and a commitment to excellence, this agent ensures that the final framework is error-free, consistent, and optimized for performance.",
            "Code_Implementation_Agent": "The Code Implementation Agent is a Python programming expert focused on translating the system design into high-quality code. This agent implements modules with precision, ensuring the code is clean, maintainable, and adheres to the established design structure, complete with clear comments and explanations.",
            "Test_Case_Generation_Agent": "The Test Case Generation Agent specializes in creating comprehensive test cases to validate the system's functionality and robustness. By analyzing the system design and requirements, this agent formulates test scenarios, edge cases, and expected outcomes, ensuring thorough test coverage and reliable software performance.",
            "Coordination_Agent": "The Coordination Agent responsible for managing feedback between the Code Execution Agent and the Code Implementation Agent.",
            "Code_Execution_Agent": "The Code Execution Agent is a Python interpreter that executes code snippets provided by the user. This agent can run Python code and display the output.",
            "User": "The User agent represents the human user interacting with the system.",
            "Library_Installation_Agent": "The Library Installation Agent is responsible for installing the necessary libraries and dependencies required for the code execution environment.",
            "Code_Extraction_Agent": "The Code Extraction Agent is responsible for identify and extract the original code portion from the input, removing any parts that are used for testing.",
            "Code_Modification_Agent": "The Code Modification Assistant is responsible for modifying the code to meet the requirements."
        }
        return descriptions.get(name, "No description available.")