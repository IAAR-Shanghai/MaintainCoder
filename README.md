﻿<h1 align="center">MaintainCoder: Maintainable Code Generation <br> Under Dynamic Requirements</h1>

<p align="center">
<a href='https://arxiv.org/pdf/2503.24260'><img src='https://img.shields.io/badge/arXiv-2503.24260-b31b1b.svg'></a>
<a href='https://github.com/IAAR-Shanghai/MaintainCoder'><img src='https://img.shields.io/badge/GitHub-MaintainCoder-blue'></a>
<a href="https://opensource.org/licenses/MIT" target="_blank"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
<a href="https://github.com/IAAR-Shanghai/MaintainCoder" target="_blank"><img alt="GitHub Stars" src="https://img.shields.io/github/stars/IAAR-Shanghai/MaintainCoder?style=social"></a>
</p>

<h5 align="center">If you like our project, please give us a star ⭐ on GitHub for the latest updates!</h5>

## 💡 Overview

**MaintainCoder** is a code generation framework focused on maintainability, combining software design principles with multi-agent collaboration. It outperforms existing methods on dynamic benchmarks from our proposed **MaintainBench**.

<table class="center">
    <tr>
        <td width=100% style="border: none"><img src="image/MaintainCoder.png" style="width:100%"></td>
    </tr>
    <tr>
        <td width="100%" style="border: none; text-align: center; word-wrap: break-word">
              Overview of MaintainCoder.MaintainCoder follows a two-stage pipeline: a framework module builds maintainable architecture from requirements, and a generation module produces code aligned with both design and intent.
      </td>
    </tr>
</table>

<table class="center">
    <tr>
        <td width=100% style="border: none"><img src="image/MaintainBench.png" style="width:100%"></td>
    </tr>
    <tr>
        <td width="100%" style="border: none; text-align: center; word-wrap: break-word">
              The systematic construction of MaintainBench. MaintainBench systematically evolves curated problems via structured requirement changes, co-evolves code and tests, and ensures quality through testing and expert review.
      </td>
    </tr>
</table>

## 🔧 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/IAAR-Shanghai/MaintainCoder.git
cd MaintainCoder
pip install -r requirements.txt
```

## 🏃 Usage

### Command Line Interface

The main functionality is accessible through the `main.py` script which accepts multiple parameters:

```bash
python main.py --method <method> --dataset_name <dataset> --evaluate <true/false> [additional options]
```

### Parameters

- `--method`: Generation method to use
  - `generate_raw_agent`: Use agent-based model to generate code
  - `generate_raw_llm`: Use LLM to generate code with specific prompt types
  - `generate_dyn`: Generate code that adapts to changing requirements
  - `skip`: Skip generation phase

- `--prompt_type`: Type of prompting (for LLM generation)
  - `direct`: Direct prompting
  - `COT`: Chain-of-thought prompting
  - `selfplanning`: Self-planning approach

- `--dataset_name`: Dataset to use
  - Options: `apps_dyn`, `codecontests_dyn`, `humaneval_dyn`, `mbpp_dyn`, `xcodeeval_dyn`

- `--evaluate`: Whether to evaluate the generated code (true/false)

- `--api_key`: API key for the LLM service (optional)
- `--base_url`: Base URL for the API (optional)
- `--model_name`: Model name to use (optional)

## 📊 Evaluation Metrics

The code evaluates several aspects of the generated solutions:

1. **Pass5 Score**: Measures how many generated solutions pass all test cases across 5 iterations
2. **Maintainability Index (MI)**: A software metric that indicates how maintainable the code is
3. **Cyclomatic Complexity (CC)**: A measure of code complexity
4. **AST Similarity**: Measures structural similarity between original and modified code
5. **Code Change Metrics**:
   - **Ratio**: Percentage of code changed
   - **Absolute**: Absolute amount of code changed

## 🌟 Example Commands

Generate code using MaintainCoder on HumanEval dataset:

```bash
python main.py --method generate_raw_agent --dataset_name humaneval_dyn --evaluate false --model_name gpt-4o-mini
```

Generate code using LLM with direct prompting on HumanEval dataset:

```bash
python main.py --method generate_raw_llm --prompt_type direct --dataset_name humaneval_dyn --evaluate false --model_name gpt-4o-mini
```

Generate code adaptations for changing requirements:

```bash
python main.py --method generate_dyn --dataset_name mbpp_dyn --evaluate false --model_name gpt-4o-mini
```

Evaluate previously generated code without generating new code:

```bash
python main.py --method skip --prompt_type direct --dataset_name codecontests_dyn --evaluate true --model_name gpt-4o-mini
```

## ❤️ Acknowledgements

This project benefits from the following open-source projects:
- [AutoGen](https://github.com/microsoft/autogen)
- [FLAML](https://github.com/microsoft/FLAML)

## 📞 Contact

For any questions or feedback, please reach out to us via GitHub Issues.
