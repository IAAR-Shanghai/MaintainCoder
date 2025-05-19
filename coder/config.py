import yaml
from pathlib import Path

def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).resolve().parent.parent

PROJECT_ROOT = get_project_root()
DOCKER_WORKSPACE = PROJECT_ROOT / "coder" / "docker_workspace"

config_path = PROJECT_ROOT / "config" / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

AGENT_API_KEY = config["agent_api_key"]
AGENT_BASE_URL = config["agent_base_url"]
AGENT_MODEL_NAME = config["agent_model_name"]