import yaml
from pathlib import Path

def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).resolve().parent.parent

PROJECT_ROOT = get_project_root()
print(f"Project root: {PROJECT_ROOT}")

config_path = PROJECT_ROOT / "config" / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# API_KEY = config["api_key"]
# BASE_URL = config["base_url"]
# MODEL_NAME = config["model"]