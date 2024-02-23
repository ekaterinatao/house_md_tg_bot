from dataclasses import dataclass
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
            token=env('BOT_TOKEN'),
            admin_ids=list(map(int, env.list('ADMIN_IDS')))
        ),
        data=DataConfig(
            dataset='data/house_dataset.gz',
            cls_vec='ekaterinatao/house_md_cls_embeds'
        ),
        model=ModelConfig(
            bi_checkpoint='ekaterinatao/house-md-bot-bert-bi-encoder',
            cross_checkpoint='ekaterinatao/house-md-bot-bert-cross-encoder',
            device='cpu'
        )
    )