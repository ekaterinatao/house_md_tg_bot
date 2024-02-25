from gradio_client import Client
from config_data.config import Config, load_config

config: Config = load_config()

client = Client(config.model.hf_client)

def get_answer(user_input):
    answer = client.predict(
        user_input,
        api_name="/predict"
    )
    return answer