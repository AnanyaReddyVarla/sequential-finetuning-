import yaml
from openai import OpenAI

def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def get_client(role="teacher", config_path="configs/config.yaml"):
    cfg = load_config(config_path)
    api_cfg = cfg[f"{role}_api"]
    client = OpenAI(
        api_key=api_cfg["api_key"],
        base_url=api_cfg["base_url"]
    )
    return client, api_cfg["model"]

def call_model(prompt: str, role="teacher", max_tokens=512, temperature=0.1) -> str:
    client, model = get_client(role)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message.content.strip()
