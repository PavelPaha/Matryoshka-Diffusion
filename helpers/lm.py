import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


class T5Encoder(T5ForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def embed_dim(self):
        return self.model_dim

    def forward(
        self,
        input_ids,
        attention_mask=None,
        return_dict=None,
        return_penultimate=None,
        *args,
        **kwargs,
    ):
        output = self.encoder.forward(
            input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
            *args,
            **kwargs,
        )
        return output.last_hidden_state

    def load(self):
        pass


def create_tokenizer(text_model_name):
    return T5Tokenizer.from_pretrained(text_model_name)


def create_lm(text_model_name: str = "t5-small"):

    tokenizer = create_tokenizer(text_model_name)
    encoder = T5Encoder.from_pretrained(text_model_name)
    encoder.eval()

    return tokenizer, encoder
