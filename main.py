import argparse

from coder.config import AGENT_API_KEY, AGENT_BASE_URL, AGENT_MODEL_NAME
from scripts.generate import generate_raw_agent, generate_raw_llm, generate_dyn
from scripts.evaluate import run_pass5_dyn, run_cal_MI, run_cal_sim

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    # 通过argparser传入参数：method，dataset_name, phase_1 or phase_2 or evaluate
    parser = argparse.ArgumentParser(description="Generate code using LLM.")
    parser.add_argument("--method", type=str, required=True, help="Method to use: generate_raw_agent, generate_raw_llm, generate_dyn, skip", choices=["generate_raw_agent", "generate_raw_llm", "generate_dyn", "skip"])
    parser.add_argument("--prompt_type", type=str, default="direct", help="For raw_llm: direct / COT / selfplanning", choices=["direct", "COT", "selfplanning"])
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name: apps_dyn, codecontests_dyn, humaneval_dyn, mbpp_dyn, xcodeeval_dyn", choices=["apps_dyn", "codecontests_dyn", "humaneval_dyn", "mbpp_dyn", "xcodeeval_dyn"])
    parser.add_argument("--evaluate", type=str2bool, required=True, help="Evaluate the generated code: True or False")
    parser.add_argument("--api_key", type=str, default=AGENT_API_KEY, help="API key for the agent")
    parser.add_argument("--base_url", type=str, default=AGENT_BASE_URL, help="Base URL for the agent")
    parser.add_argument("--model_name", type=str, default=AGENT_MODEL_NAME, help="Model name for the agent or llm")
    args = parser.parse_args()

    # 根据传入的参数调用相应的函数
    if args.method == "generate_raw_agent":
        print("Generating raw agent...")
        flag = generate_raw_agent(args.dataset_name, args.api_key, args.base_url, args.model_name)
        if flag:
            print("Generation completed successfully.")
        else:
            print("Generation failed after 5 iterations.")
    elif args.method == "generate_raw_llm":
        print("Generating raw LLM...")
        flag = generate_raw_llm(args.dataset_name, args.api_key, args.base_url, args.model_name, args.prompt_type)
        if flag:
            print("Generation completed successfully.")
        else:
            print("Generation failed after 5 iterations.")
    elif args.method == "generate_dyn":
        raw_code_name = [args.model_name + "_" + args.prompt_type + "_" + str(i) for i in range(5)]
        generate_dyn(args.dataset_name, raw_code_name, args.api_key, args.base_url, args.model_name)
    elif args.method == "skip":
        print("Skipping generation.")
    else:
        print("Invalid method. Please choose from: generate_raw_agent, generate_raw_llm, generate_dyn, skip")

    if args.evaluate == True:
        print("Evaluating the generated code...")
        print("Running Pass5...")
        method_name = args.model_name + "_" + args.prompt_type
        pass_5_dyn, pass_list_dyn = run_pass5_dyn(args.dataset_name, method_name)
        print(f"Pass5 score: {pass_5_dyn}")
        print("Running MI...")
        MI_score, CC_score, _, _ = run_cal_MI(args.dataset_name, method_name)
        print(f"MI score: {MI_score}")
        print(f"CC score: {CC_score}")
        print("Running Similarity...")
        avg_ast, avg_code_change_ratio, avg_code_change_abs, _ = run_cal_sim(args.dataset_name, method_name)
        print(f"AST Similarity: {avg_ast}")
        print(f"Code Change Ratio: {avg_code_change_ratio}")
        print(f"Code Change Absolute: {avg_code_change_abs}")

if __name__ == "__main__":
    main()