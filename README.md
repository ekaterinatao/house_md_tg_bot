# House MD bot

## Description
Retrieval-based chat bot that can imitate conversation with House MD.

## Usage

Try simple demo of the bot on [huggingface](https://huggingface.co/spaces/ekaterinatao/house_md_bot)

**To deploy telegram bot:**  
  
* Create a Telegram bot with [BotFather](https://t.me/botfather)  
* Clone repository `!git clone https://github.com/ekaterinatao/house_md_tg_bot.git`  
* Install dependencies from `requirements.txt`  
* Paste you `BOT_TOKEN` into `.env` (see [example](https://github.com/ekaterinatao/house_md_tg_bot/blob/main/.env.example))
* **Hint:** For running the bot using huggingface gradio app change import of `get_answer` function in [`bot.py` file](https://github.com/ekaterinatao/house_md_tg_bot/blob/main/bot.py) and `tg_bot.token` in [`config.py` file](https://github.com/ekaterinatao/house_md_tg_bot/blob/main/config_data/config.py).  
  
## Tools under the hood
* Telegram bot and `python-telegram-bot` library  
* Huggingface finetuned `distilbert-base-uncased` model for [bi-encoder](https://huggingface.co/ekaterinatao/house-md-bot-bert-bi-encoder) and [crocc-encoder](https://huggingface.co/ekaterinatao/house-md-bot-bert-cross-encoder)  
* Huggingface `Gradio` for demo  
* Preprocessed [dataset](https://huggingface.co/datasets/ekaterinatao/house_md_context3) from House MD series  

## Models
[Here](https://github.com/ekaterinatao/house_md_tg_bot/tree/main/notebooks) you can notebooks with data preprocessing and models finetuning.  

**Retrieval algorithm:**  
* Finetuning `distilbert-base-uncased` model on [House MD dataset](https://huggingface.co/datasets/ekaterinatao/house_md_context3)  
* Formation of the `CLS embeddings` of the context lines preceding the MD House's answer using [bi-encoder](https://huggingface.co/ekaterinatao/house-md-bot-bert-bi-encoder)  
* Selecting MD House's answers from top 50 `CLS embeddings` that are similar to current request. Selection was performed using [faiss](https://github.com/facebookresearch/faiss) search.  
* Re-ranking of the top 50 MD House's answers together with context using [crocc-encoder](https://huggingface.co/ekaterinatao/house-md-bot-bert-cross-encoder)  