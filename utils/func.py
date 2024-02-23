import torch
import faiss
import numpy as np
import pandas as pd
import datasets
from transformers import AutoTokenizer, AutoModel
from config_data.config import Config, load_config


config: Config = load_config()


def embed_bert_cls(
        text: str, 
        model: AutoModel, 
        tokenizer: AutoTokenizer
) -> np.ndarray:

    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeds = model_output.last_hidden_state[:, 0, :]
    embeds = torch.nn.functional.normalize(embeds)

    return embeds[0].cpu().numpy()


def get_ranked_docs(
    query: str, vec_query_base: np.ndarray, data: pd.DataFrame,
    bi_model: AutoModel, bi_tok: AutoTokenizer, 
    cross_model: AutoModel, cross_tok: AutoTokenizer
) -> str:

    vec_shape = vec_query_base.shape[1]
    index = faiss.IndexFlatL2(vec_shape)
    index.add(vec_query_base)
    xq = embed_bert_cls(query, bi_model, bi_tok)
    D, I = index.search(xq.reshape(1, vec_shape), 50)

    corpus = []
    for i in I[0]:
        corpus.append(data['answer'][i])

    queries = [query] * len(corpus)
    tokenized_texts = cross_tok(
        queries, corpus, max_length=128, padding=True, truncation=True, return_tensors="pt"
    ).to(config.model.device)

    with torch.no_grad():
        ce_scores = cross_model(
            tokenized_texts['input_ids'], tokenized_texts['attention_mask']
        ).last_hidden_state[:, 0, :]
        ce_scores = ce_scores @ ce_scores.T

    scores = ce_scores.cpu().numpy()
    scores_ix = np.argsort(scores)[::-1]

    return corpus[scores_ix[0][0]]


def load_dataset(url: str=config.data.dataset) -> pd.DataFrame:

    house_dataset = pd.read_csv(url, compression='gzip')

    return house_dataset


def load_cls_base(url: str=config.data.cls_vec) -> np.array:

    cls_dataset = datasets.load_dataset(url, split='train')
    cls_base = np.stack([embed for embed in pd.DataFrame(cls_dataset)['cls_embeds']])

    return cls_base


def load_bi_enc_model(
        checkpoint: str=config.model.bi_checkpoint
) -> tuple[AutoTokenizer, AutoModel]:

    bi_model = AutoModel.from_pretrained(checkpoint)
    bi_tok = AutoTokenizer.from_pretrained(checkpoint)

    return bi_model, bi_tok


def load_cross_enc_model(
        checkpoint: str=config.model.cross_checkpoint
) -> tuple[AutoTokenizer, AutoModel]:

    cross_model = AutoModel.from_pretrained(checkpoint)
    cross_tok = AutoTokenizer.from_pretrained(checkpoint)

    return cross_model, cross_tok