from aiogram import Bot, Dispatcher, executor, types
from config_data.config import Config, load_config
from utils.func import (get_ranked_docs, 
                        load_dataset, load_cls_base, 
                        load_bi_enc_model, load_cross_enc_model)

config: Config = load_config()

bot = Bot(token=config.tg_bot.token)
dp = Dispatcher(bot)


@dp.message_handler(commands=["start"])
async def welcome(message: types.Message):
    await message.answer("Hi! Write me anything you want!")


@dp.message_handler(commands=['help'])
async def process_help_command(message: types.Message):
    await message.answer(
        "Write me anything you want."
        "I'll send you House MD answer."
    )


@dp.message_handler()
async def get_answer(message: types.Message):

    user_input = message.text

    await message.answer("Loadind the best answer...\nPlease wait couple of minutes :)")
    dataset = load_dataset()
    cls_base = load_cls_base()
    bi_enc_model = load_bi_enc_model()
    cross_enc_model = load_cross_enc_model()

    answer = get_ranked_docs(
        query=user_input, vec_query_base=cls_base, data=dataset,
        bi_model=bi_enc_model[0], bi_tok=bi_enc_model[1],
        cross_model=cross_enc_model[0], cross_tok=cross_enc_model[1]
    )

    await message.reply(text=answer)


if __name__ == "__main__":
    executor.start_polling(dp)