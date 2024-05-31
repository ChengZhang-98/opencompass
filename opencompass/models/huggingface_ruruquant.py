from .huggingface_above_v4_33 import HuggingFaceBaseModel


class HuggingFaceBaseModelRuruQuant(HuggingFaceBaseModel):
    def __init__(
        self,
        path: str,
        q_config: dict,
        model_kwargs: dict = dict(),
        tokenizer_path: str | None = None,
        tokenizer_kwargs: dict = dict(),
        peft_path: str | None = None,
        peft_kwargs: dict = dict(),
        tokenizer_only: bool = False,
        generation_kwargs: dict = dict(),
        max_seq_len: int | None = None,
        pad_token_id: int | None = None,
        stop_words: str | None = [],
        **other_kwargs
    ):
        self.q_config = q_config
        super().__init__(
            path,
            model_kwargs,
            tokenizer_path,
            tokenizer_kwargs,
            peft_path,
            peft_kwargs,
            tokenizer_only,
            generation_kwargs,
            max_seq_len,
            pad_token_id,
            stop_words,
            **other_kwargs
        )


    def _load_model(
        self,
        path: str,
        kwargs: dict,
        peft_path: str | None = None,
        peft_kwargs: dict = dict(),
    ):
        from ruruquant.models.automodel import AutoRuruQuantModelForCausalLM

        self.model = AutoRuruQuantModelForCausalLM.from_pretrained(
            path, self.q_config, **kwargs
        )

        self.model.eval()
        self.model.generation_config.do_sample = False
