import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessor, hash_prompt
from threestudio.utils.misc import cleanup
from threestudio.utils.typing import *

from .prompt_learner import PromptLearner


@threestudio.register("stable-diffusion-prompt-learning-processor")
class StableDiffusionPromptLearningProcessor(PromptProcessor):
    @dataclass
    class Config(PromptProcessor.Config):
        pass

    cfg: Config

    def preprocess_prompt(self, prompt: str) -> str:
        assert os.path.exists(self.cfg.ckpt_path) and self.cfg.ckpt_path.endswith(".pt")
        n_prompts, n_ctx = 1, 0
        ckpt, means, stds, prompt_texts = None, None, None, None
        pdl = False
        ckpt = torch.load(self.cfg.ckpt_path)
        ckpt = ckpt.get("model_state_dict", ckpt)
        for k, v in ckpt.items():
            if k.replace("_orig_mod.", "").replace("module.", "") == "ctx":
                ckpt = OrderedDict({"ctx": v})
        ctx = ckpt["ctx"]
        n_ctx = ctx.shape[-2]
        if ctx.ndim == 4:  # pdl
            pdl = True
            n_prompts = ctx.shape[1]
        self.prompt_learner = PromptLearner(
            pretrained_model_name_or_path=self.cfg.pretrained_model_name_or_path,
            classnames=[prompt],
            n_ctx=n_ctx,
            pdl=pdl,
            n_prompts=n_prompts,
            customize_prompt=prompt,
        )
        missing_keys, unexpected_keys = self.prompt_learner.load_state_dict(ckpt, strict=False)
        assert "ctx" not in missing_keys
        assert len(unexpected_keys) == 0
        self.prompt_learner.sample_fixed_ctx()
        print(f"prompt_learner ctx weights loaded from {self.cfg.ckpt_path}")
        return prompt

    ### these functions are unused, kept for debugging ###
    def configure_text_encoder(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="text_encoder"
        ).to(self.device)

        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

    def destroy_text_encoder(self) -> None:
        del self.tokenizer
        del self.text_encoder
        cleanup()

    def get_text_embeddings(
        self, prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]]
    ) -> Tuple[Float[Tensor, "B 77 768"], Float[Tensor, "B 77 768"]]:
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        # Tokenize text and get embeddings
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_tokens = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(tokens.input_ids.to(self.device))[0]
            uncond_text_embeddings = self.text_encoder(
                uncond_tokens.input_ids.to(self.device)
            )[0]

        return text_embeddings, uncond_text_embeddings

    ###

    # @staticmethod
    def spawn_func(self, pretrained_model_name_or_path, prompts, cache_dir):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        with torch.no_grad():
            text_embeddings, captions = self.prompt_learner.generate_from_fixed_ctx(prompts)

        # tokenizer = AutoTokenizer.from_pretrained(
        #     pretrained_model_name_or_path, subfolder="tokenizer"
        # )
        # text_encoder = CLIPTextModel.from_pretrained(
        #     pretrained_model_name_or_path,
        #     subfolder="text_encoder",
        #     device_map="auto",
        # )
        #
        # with torch.no_grad():
        #     tokens = tokenizer(
        #         prompts,
        #         padding="max_length",
        #         max_length=tokenizer.model_max_length,
        #         return_tensors="pt",
        #     )
        #     text_embeddings = text_encoder(tokens.input_ids.to(text_encoder.device))[0]

        for prompt, embedding in zip(prompts, text_embeddings):
            torch.save(
                embedding,
                os.path.join(
                    cache_dir,
                    f"{hash_prompt(pretrained_model_name_or_path, prompt)}.pt",
                ),
            )

        # del text_encoder
