# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import difflib
import math
import numpy as np
import torch
from typing import Optional, Union, Tuple, List, Callable, Dict
import inspect
from typing import Any
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
import abc

MAX_NUM_WORDS = 77+256
TORCH_DTYPE = torch.bfloat16

class AttentionControl(abc.ABC):

    def step_callback(self, x_t):  # Define that in each diffusion step, the input x_t can be operated
        return x_t

    def between_steps(self):  # Define the operation to be performed after each diffusion step is completed
        return

    @abc.abstractmethod
    def forward(self, attn, place_in_transformer: str):
        raise NotImplementedError

    def __call__(self, attn, place_in_transformer: str):
        h = attn.shape[0]
        attn[h // 2:] = self.forward(attn[h // 2:], place_in_transformer)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl): # Debug

    def forward(self, attn, place_in_transformer: str):
        return attn


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"MM": []}

    def forward(self, attn, place_in_transformer: str):
        return attn

    def between_steps(self):
        return

    def get_average_attention(self):  # Calculate the average attention weight for each category
        average_attention = {
            key: [item / self.cur_step for item in self.attention_store[key]]
            for key in self.attention_store
        }
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self, x_t):
        return x_t

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base):
        raise NotImplementedError

    def forward(self, attn, place_in_transformer: str):  # Execute reweight logic
        super(AttentionControlEdit, self).forward(attn, place_in_transformer)
        if (self.cur_step + 1) <= self.cross_replace_steps:
            attn_base = attn[:, :, :-MAX_NUM_WORDS, -MAX_NUM_WORDS:]
            attn_replace_new = self.replace_cross_attention(attn_base)
            attn[:, :, :-MAX_NUM_WORDS, -MAX_NUM_WORDS:] = attn_replace_new
        return attn

    def __init__(
        self,
        prompts,
        num_steps: int,
        cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
    ):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_steps = int(num_steps * cross_replace_steps)


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base):
        attn_replace = attn_base[:, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(
            self,
            prompt,
            num_steps: int,
            cross_replace_steps: float,
            controller: Optional[AttentionControlEdit] = None,
            tokenizer=None
        ):
        super(AttentionReweight, self).__init__(prompt, num_steps, cross_replace_steps)
        equalizer = get_equalizer(prompt, tokenizer=tokenizer)
        self.equalizer = equalizer.to("cuda")
        self.prev_controller = controller


def get_equalizer(text: str,tokenizer):
    inds = get_quote_inds(text, tokenizer)
    values = torch.tensor(len(inds), dtype=TORCH_DTYPE)
    values = 1+0.2*torch.tanh(0.5*values)
    equalizer = torch.full((1, MAX_NUM_WORDS), 1.0, dtype=TORCH_DTYPE)
    equalizer[:, inds] = values
    return equalizer


def aggregate_attention(
        attention_store: AttentionStore,
        res: int,
        from_where: List[str],
        select: int
):  # Extract and aggregate the attention weights of the specified stage from AttentionStore
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}"]:
            if item.shape[2] == num_pixels:
                cross_maps = item.reshape(item.shape[0], item.shape[1], res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=False)
    # return attn_weight @ value
    return attn_weight


def register_attention_control(model, controller):
    def ca_forward(self, place_in_transformer):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None):
            residual = hidden_states

            batch_size = hidden_states.shape[0]

            # `sample` projections.
            query = self.to_q(hidden_states)
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // self.heads

            query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

            if self.norm_q is not None:
                query = self.norm_q(query)
            if self.norm_k is not None:
                key = self.norm_k(key)

            # `context` projections.
            if encoder_hidden_states is not None:
                encoder_hidden_states_query_proj = self.add_q_proj(encoder_hidden_states)
                encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
                encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

                encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                    batch_size, -1, self.heads, head_dim
                ).transpose(1, 2)
                encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                    batch_size, -1, self.heads, head_dim
                ).transpose(1, 2)
                encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                    batch_size, -1, self.heads, head_dim
                ).transpose(1, 2)

                if self.norm_added_q is not None:
                    encoder_hidden_states_query_proj = self.norm_added_q(encoder_hidden_states_query_proj)
                if self.norm_added_k is not None:
                    encoder_hidden_states_key_proj = self.norm_added_k(encoder_hidden_states_key_proj)

                query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
                key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
                value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

            hidden_states = scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
            hidden_states = controller(hidden_states, place_in_transformer)
            hidden_states = hidden_states @ value
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            if encoder_hidden_states is not None:
                # Split the attention outputs.
                hidden_states, encoder_hidden_states = (
                    hidden_states[:, : residual.shape[1]],
                    hidden_states[:, residual.shape[1]:],
                )
                if not self.context_pre_only:
                    encoder_hidden_states = self.to_add_out(encoder_hidden_states)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            # dropout
            hidden_states = self.to_out[1](hidden_states)

            if encoder_hidden_states is not None:
                return hidden_states, encoder_hidden_states
            else:
                return hidden_states

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:  # If controller not provided
        controller = DummyController()

    def register_recr(net_, count, place_in_transformer):
        if net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward(net_, place_in_transformer)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_transformer)
        return count  # Counter, used to count the number of replaced Attention layers.

    cross_att_count = 0
    sub_nets = model.transformer.named_children()
    for net in sub_nets:
        if "transformer_blocks" in net[0]:
            cross_att_count += register_recr(net[1], 0, "MM")

    controller.num_att_layers = cross_att_count

def get_quote_inds(text: str, tokenizer):  # Input prompt, output the index of the token between the quotation marks and the preceding quotation marks in prompt
    out = []
    text.replace('"', "'")
    words_encode = [tokenizer.decode([item]) for item in tokenizer.encode(text)][:-1]
    between_quote = False
    for i in range(len(words_encode)):
        if between_quote and words_encode[i] != "'":
            out.append(i)
        elif between_quote and words_encode[i] == "'":
            between_quote = False
        elif not between_quote and words_encode[i] == "'":
            between_quote = True
            out.append(i)
    if between_quote:
        raise Exception("miss quote")
    out = [x-256 for x in out]  # sd3
    return np.array(out)


def get_word_inds(text: str, word: str, tokenizer):  # Input prompt, word, and output the index of the token corresponding to this word
    words_encode = [tokenizer.decode([item]) for item in tokenizer.encode(text)][:-1]
    best_matches = difflib.get_close_matches(word, words_encode, n=1, cutoff=0.0)
    best_match = best_matches[0]
    indices = [i for i, x in enumerate(words_encode) if x == best_match]
    if len(indices) == 0:
        raise Exception("miss token")
    indices = [x-256 for x in indices]  # sd3
    return indices


def get_first_quote_inds(text: str, tokenizer):  # Input prompt, output the index of the token of the preceding quotation marks in prompt
    out = []
    text.replace('"', "'")
    words_encode = [tokenizer.decode([item]) for item in tokenizer.encode(text)][:-1]
    between_quote = False
    for i in range(len(words_encode)):
        if words_encode[i] == "'":
            if between_quote:
                between_quote = False
            elif not between_quote and words_encode[i] == "'":
                between_quote = True
                out.append(i)
    if between_quote:
        raise Exception("miss quote")
    out = [x-256 for x in out]  # sd3
    return out


def embed_addition(mask_tokens, prompt, tokenizer, prompt_embeds, addition):
    mask_inds = []
    for mask_token in mask_tokens:
        mask_ind = get_word_inds(prompt, mask_token, tokenizer)
        mask_inds += mask_ind
    mask_inds = sorted(set(mask_inds))
    first_quote_inds = get_first_quote_inds(prompt, tokenizer)
    pointer = 0
    for i in range(len(mask_inds)):
        if i == len(mask_inds) - 1:
            while pointer < len(first_quote_inds):
                prompt_embeds[:, mask_inds[i], :] += addition * prompt_embeds[:, first_quote_inds[pointer],:]
                pointer += 1
            break
        while first_quote_inds[pointer] < mask_inds[i + 1]:
            prompt_embeds[:, mask_inds[i], :] += addition * prompt_embeds[:, first_quote_inds[pointer], :]
            pointer += 1
    return prompt_embeds
