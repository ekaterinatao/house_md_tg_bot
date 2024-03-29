from dataclasses import dataclass
import os
from environs import Env


@dataclass
class TgBot:
    token: str
    admin_ids: list[int]


@dataclass
class DataConfig:
    dataset: str
    cls_vec: str


@dataclass
class ModelConfig:
    bi_checkpoint: str
    cross_checkpoint: str
    device: str
    hf_client: str


@dataclass
class Config:
    tg_bot: TgBot
    data: DataConfig
    model: ModelConfig


def load_config(path: str='.env') -> Config:

    env: Env = Env()
    env.read_env()

    return Config(
        tg_bot=TgBot(
            token=env('BOT_TOKEN'), # os.environ['telegram_token'] # for running in huggingface gradio app
            admin_ids=list(map(int, env.list('ADMIN_IDS')))
        ),
        data=DataConfig(
            dataset='ekaterinatao/house_md_context3',
            cls_vec='ekaterinatao/house_md_cls_embeds'
        ),
        model=ModelConfig(
            bi_checkpoint='ekaterinatao/house-md-bot-bert-bi-encoder',
            cross_checkpoint='ekaterinatao/house-md-bot-bert-cross-encoder',
            device='cpu',
            hf_client='ekaterinatao/house_md_bot'
        )
    )