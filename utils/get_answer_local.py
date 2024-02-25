from utils.func import (get_ranked_docs,
                        load_dataset, load_cls_base,
                        load_bi_enc_model, load_cross_enc_model)


def get_answer(user_input):
    dataset = load_dataset()
    cls_base = load_cls_base()
    bi_enc_model = load_bi_enc_model()
    cross_enc_model = load_cross_enc_model()

    answer = get_ranked_docs(
        query=user_input, vec_query_base=cls_base, data=dataset,
        bi_model=bi_enc_model[0], bi_tok=bi_enc_model[1],
        cross_model=cross_enc_model[0], cross_tok=cross_enc_model[1]
    )
    return answer