import copy
import torch
import torch.nn as nn
import numpy as np
from numpy.random import randint
from numpy.random import normal
from einops import repeat, rearrange
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, AutoProcessor
from torch.nn.functional import cosine_similarity


# Input single ctx vector
def interpret_ctx(ctx, tokenizer, embedder, topk=1, print_info=False):
    ranks, ctx_words, dists = [], [], []
    token_embedding = embedder.token_embedding.weight

    # Single context
    distance = torch.cdist(ctx, token_embedding)
    sorted_idxs = torch.argsort(distance, dim=1)
    sorted_idxs = sorted_idxs[:, :topk]

    if print_info:
        print(f"Size of token embedding: {token_embedding.shape}")
        print(f"Size of context: {ctx.shape}")
        print(f"Return the top-{topk} matched words")
        print(f"Size of distance matrix: {distance.shape}")

    for m, idxs in enumerate(sorted_idxs):
        words = [tokenizer.decoder[idx.item()] for idx in idxs]
        dist = [f"{distance[m, idx].item():.4f}" for idx in idxs]
        ranks.append(m+1)
        ctx_words.append(words)
        dists.append(dist)
    return ranks, ctx_words, dists

class PromptLearner(nn.Module):
    """
    PromptLearner class implements learnable prompt embeddings
    Input class idx, output learnable prompt embedding of that class
    """

    def __init__(
        self,
        pretrained_model_name_or_path,
        classnames,
        n_ctx=8,
        n_prompts=10,
        csc=True,
        pdl=True,
        select_prompt="random",
        cls_pos="end",
        dtype=torch.float32,
        num_noise_token=0,
        noise_offset=0,
        use_classname=True,
        customize_prompt=None,
    ):
        super().__init__()
        self.dtype = dtype
        self.pdl = pdl
        self.select_prompt = select_prompt
        self.n_prompts = n_prompts
        self.classnames = classnames
        self.noise_offset = noise_offset
        self.num_noise_token = num_noise_token
        self.use_classname = use_classname
        self.customize_prompt = customize_prompt
        self.fixed_ctx = None
        self.ctx_means = None
        self.ctx_stds = None
        if not self.use_classname or self.customize_prompt is not None:
            self.classnames = ["" for _ in self.classnames]
        n_cls = len(self.classnames) if self.customize_prompt is None else 1

        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.vision_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
        self.text_encoder = CustomTextEncoder(text_encoder.text_model)
        self.vision_encoder.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        ctx_dim = self.text_encoder.final_layer_norm.weight.shape[0]

        # random initialization
        if csc and not pdl:
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=self.dtype)
        elif csc and pdl:
            print("Initializing class-specific contexts with prompt distribution learning")
            ctx_vectors = torch.empty(n_cls, n_prompts, n_ctx, ctx_dim, dtype=self.dtype)
        elif not csc and pdl:
            print("Initializing generic context with prompt distribution learning")
            ctx_vectors = torch.empty(n_prompts, n_ctx, ctx_dim, dtype=self.dtype)
        else:
            print("Initializing generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=self.dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.prompt_prefix = " ".join(["X"] * (n_ctx + num_noise_token))

        print(f"Number of context words (tokens): {n_ctx}")
        print(f"Number of noise (tokens): {num_noise_token}")
        print(f"Prompt distribution learning: {self.pdl}")
        if self.pdl:
            print(f"Number of prompts per class: {n_prompts}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        if self.customize_prompt is None:
            self.classnames = [name.replace("_", " ") for name in self.classnames]
        else:
            self.classnames = [customize_prompt]
        name_lens = [len(self.tokenizer(name).input_ids) - 2 for name in self.classnames]
        prompts = [self.prompt_prefix + " " + name + "." for name in self.classnames]
        self.embedder = self.text_encoder.embeddings

        # tokenized_prompts as an anchor for retrieving position of eos token in each class prompt
        tokenized_prompts = torch.cat([
            self.tokenizer(
                p,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids
            for p in prompts
        ]).to(self.embedder.position_ids.device)
        with torch.no_grad():
            embedding = self.embedder(tokenized_prompts).type(self.dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.n_tokens = self.tokenizer.model_max_length
        self.class_token_position = cls_pos
        self.csc = csc
        self.means = None
        self.stds = None
        self.prompt_texts = None
        self.prompt_freq = np.ones((n_cls, n_prompts))

    def interpret(self, cls_idx=0):
        if self.pdl:
            ctx = self.ctx[cls_idx].mean(dim=0)
        elif self.csc:
            ctx = self.ctx[cls_idx]
        else:
            ctx = self.ctx
        eow = "</w>"
        _, words, _ = interpret_ctx(ctx, tokenizer=self.tokenizer, embedder=self.embedder, topk=1)
        if self.use_classname:
            if self.class_token_position == "end":
                words = words + [[self.classnames[cls_idx]]]
            elif self.class_token_position == "front":
                words = [[self.classnames[cls_idx]]] + words
            elif self.class_token_position == "middle":
                words = words[:len(words) / 2] + [[self.class_token_position[cls_idx]]] + words[len(words) / 2:]
        words = ''.join([word[0].replace(eow, ' ') for word in words])
        words = words.strip(" ")
        return words

    # Input prompts with dimension (B, n_prompt, n_ctx, dim)
    # Output selected prompt
    def select(self, ctx, cls_idx, imgs=None):
        assert ctx.ndim == 4
        bsz = ctx.shape[0]
        idx = None
        if self.select_prompt == "random":
            idx = randint(low=0, high=self.n_prompts, size=bsz)
        elif self.select_prompt == "similarity":
            assert imgs is not None
            imgs = imgs * 0.5 + 0.5
            imgs = self.vision_processor(images=imgs, return_tensors="pt")
            imgs["pixel_values"] = imgs["pixel_values"].to(self.vision_encoder.device)
            img_features = self.vision_encoder(**imgs).pooler_output
            txt_features = self.text_encoder(
                self.concat(ctx, cls_idx=cls_idx),
                pooled=True,
                tokenized_prompts=self.tokenized_prompts.to(cls_idx.device)[cls_idx]
            )
            sim = cosine_similarity(img_features.unsqueeze(-2), txt_features, dim=-1).softmax(
                dim=-1).cpu().detach().numpy()
            sim = sim / self.prompt_freq[cls_idx.cpu().numpy()]
            idx = sim.argmax(axis=-1)
            for i, cls_i in zip(idx, cls_idx):
                self.prompt_freq[cls_i][i] += 1
        else:
            raise ValueError

        ctx = torch.stack([ctx[i, j] for i, j in zip(range(bsz), idx)])
        return ctx

    def concat(self, ctx, cls_idx=None, customize_prompts=None):
        if customize_prompts is not None:
            cls_idx = [0] * len(customize_prompts)
        prefix = self.token_prefix[cls_idx]
        suffix = self.token_suffix[cls_idx]

        # Used in customization only
        if customize_prompts is not None:
            assert self.class_token_position == "end"
            prompts = [self.prompt_prefix + " " + prompt + "." for prompt in customize_prompts]
            tokenized_prompts = torch.cat([
                self.tokenizer(
                    prompt,
                    max_length=self.tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids
                for prompt in prompts
            ]).to(self.embedder.position_ids.device)
            with torch.no_grad():
                embedding = self.embedder(tokenized_prompts).type(self.dtype)
            suffix = embedding[:, 1 + self.n_ctx:, :]  # CLS, EOS
            ctx = repeat(ctx, "l d -> b l d", b=len(customize_prompts))

        if self.num_noise_token > 0:
            noise_vectors = self.noise_offset * torch.randn((prefix.shape[0], self.num_noise_token, prefix.shape[2]),
                                                            device=prefix.device)
            prefix = torch.cat([prefix, noise_vectors], dim=1)
            suffix = suffix[:, :-self.num_noise_token, :]

        if ctx.ndim == 4:  # (B,p,l,d)
            p = ctx.shape[1]
            prefix = repeat(prefix, "b l d -> b p l d", p=p)
            suffix = repeat(suffix, "b l d -> b p l d", p=p)

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (B, 1, dim)
                    ctx,  # (B, n_ctx, dim)
                    suffix,  # (B, *, dim)
                ],
                dim=-2,
            )
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i + 1, :, :]
                class_i = suffix[i:i + 1, :name_len, :]
                suffix_i = suffix[i:i + 1, name_len:, :]
                ctx_i_half1 = ctx[i:i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i:i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=-2,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i + 1, :, :]
                class_i = suffix[i:i + 1, :name_len, :]
                suffix_i = suffix[i:i + 1, name_len:, :]
                ctx_i = ctx[i:i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=-2,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError

        return prompts

    # Input class indices (B,)
    # output tokenized class prompts (B,77) or collection of prompts
    def forward(self, cls_idx, imgs=None, interpret=False):
        ctx = self.ctx[cls_idx] if self.csc else self.ctx
        # if ctx.dim() == 2 and self.pdl:
        #     ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        if self.pdl:
            ctx = self.select(ctx, cls_idx, imgs)

        prompts = self.concat(ctx, cls_idx=cls_idx)
        prompts_hiden_state = self.text_encoder(prompts)  # (B,77,dim)
        if interpret:
            _, caption, _ = interpret_ctx(torch.squeeze(prompts), tokenizer=self.tokenizer, embedder=self.embedder,
                                          topk=1)
            return prompts_hiden_state, caption
        return prompts_hiden_state

    def fit(self):
        if not self.pdl:
            return
        self.means = np.empty((self.n_cls, self.n_tokens, self.ctx_dim))  # (1000*77*1024)
        self.stds = np.empty(self.means.shape)  # (1000,77,1024)
        self.prompt_texts = []
        for cls in tqdm(range(self.n_cls), desc="fit prompt learner distribution"):
            cls_ctx = self.ctx[cls]
            cls_prompts = self.concat(cls_ctx, cls_idx=[cls] * self.n_prompts)
            prompt_text = self.interpret(cls)
            cls_prompts = self.text_encoder(cls_prompts)
            self.means[cls] = cls_prompts.mean(dim=0).detach().cpu().numpy()
            self.stds[cls] = cls_prompts.std(dim=0).detach().cpu().numpy()
            self.prompt_texts.append(prompt_text)

    def sample(self, cls_idx, interpret=False):
        if self.pdl:
            assert self.means is not None and self.stds is not None
            prompt = torch.from_numpy(normal(self.means[cls_idx], self.stds[cls_idx])).unsqueeze(dim=0)
            if interpret:
                caption = self.prompt_texts[cls_idx]
                return prompt, caption
            return prompt
        else:
            return self.forward([cls_idx], interpret=interpret)

    # Only used in customization
    def sample_fixed_ctx(self):
        if not self.pdl:
            self.fixed_ctx = self.ctx
        else:
            assert self.n_cls == 1
            if self.means is None or self.stds is None:
                self.fit()
            self.fixed_ctx = self.sample([0])

            self.ctx_means = np.empty((self.n_cls, self.n_ctx, self.ctx_dim))  # (1*8*1024)
            self.ctx_stds = np.empty(self.ctx_means.shape)
            self.prompt_texts = []
            for cls in range(self.n_cls):  # self.n_cls == 1
                cls_ctx = self.ctx[cls]
                cls_prompts = self.concat(cls_ctx, cls_idx=[cls] * self.n_prompts)
                prompt_text = self.interpret(cls)
                cls_prompts = self.text_encoder(cls_prompts)
                self.means[cls] = cls_prompts.mean(dim=0).detach().cpu().numpy()
                self.stds[cls] = cls_prompts.std(dim=0).detach().cpu().numpy()
                self.prompt_texts.append(prompt_text)
            self.fixed_ctx = torch.from_numpy()

    # Only used in customization
    # Input prompts include views (cat toy, top view), therefore text encoder here after concat
    def generate_from_fixed_ctx(self, prompts: list):
        assert self.fixed_ctx is not None
        if self.pdl:
            raise NotImplementedError
        else:
            ctx = self.fixed_ctx[0]
            prompts = self.concat(ctx, customize_prompts=prompts)
            prompts_hiden_state = self.text_encoder(prompts)  # (B,77,dim)
            captions = []
            for prompt in prompts:
                _, words, _ = interpret_ctx(torch.squeeze(prompt), tokenizer=self.tokenizer, embedder=self.embedder, topk=1)
                caption = ''.join([word[0].replace("</w>", ' ') for word in words])
                caption = caption[:caption.find("<|endoftext|>")].replace("<|startoftext|>", "").strip(" .")
                captions.append(caption+'.')
            return prompts_hiden_state, captions


# Original text_encoder: CLIPTextModel,
#   input (B,77) prompt token indices,
#   convert token into token embeddings
#   output (B,77,1024) prompt embedding
# CustomTextEncoder: token embeddings from PromptLearner
#   input (B,77,1024) prompt token embeddings
#   output (B,77,1024) prompt embedding

class CustomTextEncoder(nn.Module):
    def __init__(self, text_transformer, dtype=torch.float32):
        super().__init__()
        self.encoder = text_transformer.encoder
        self.positional_embedding = text_transformer.embeddings.position_embedding
        self.final_layer_norm = text_transformer.final_layer_norm
        self.dtype = dtype
        self._build_causal_attention_mask = copy.deepcopy(text_transformer._build_causal_attention_mask)
        self.embeddings = text_transformer.embeddings

    def forward(self, prompts, pooled=False, tokenized_prompts=None):
        prompts = prompts + self.positional_embedding.weight.type(self.dtype)
        n_prompts = 1
        if prompts.ndim == 4:
            n_prompts = prompts.shape[1]
            prompts = rearrange(prompts, "b p l d -> (b p) l d")
            if tokenized_prompts is not None:
                tokenized_prompts = repeat(tokenized_prompts, "b l -> (b p) l", p=n_prompts)
        attention_mask = self._build_causal_attention_mask(
            bsz=prompts.shape[0],
            seq_len=prompts.shape[1],
            dtype=prompts.dtype
        ).to(prompts.device)
        prompts = self.encoder(
            inputs_embeds=prompts,  # (B,77,1024) or ((B,P),77,1024) float16
            causal_attention_mask=attention_mask,  # (B,1,77,77) or ((B,P),1,77,1024) float16
        )  # (B,77,1024)
        prompts = self.final_layer_norm(prompts[0]).type(self.dtype)

        # prompts.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eos embedding (eos_token is the highest number in each sequence)
        pooled_prompts = None
        if pooled:
            assert tokenized_prompts is not None
            pooled_prompts = prompts[torch.arange(prompts.shape[0]), tokenized_prompts.argmax(dim=-1)]  # (B,1024)
            # pooled_prompts = pooled_prompts @ self.text_projection  # (B,1024)
        if n_prompts > 1:
            prompts = rearrange(prompts, "(b p) l d -> b p l d", p=n_prompts)
            if pooled_prompts is not None:
                pooled_prompts = rearrange(pooled_prompts, "(b p) d -> b p d", p=n_prompts)

        return prompts if not pooled else pooled_prompts  # (B,(p),77,1024) or (B,(p),1024)
