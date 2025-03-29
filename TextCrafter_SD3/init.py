import torch

def init_forwards(self, root_module: torch.nn.Module):
    for name, module in root_module.named_modules():
        if "attn" in name and "transformer_blocks" in name and module.__class__.__name__ == "Attention":
            module.forward = SD3TransformerBlock_init_forward(self, module)

def SD3TransformerBlock_init_forward(self, module):
    def forward(hidden_states=None,
                encoder_hidden_states=None,
                attention_mask=None,
                **cross_attention_kwargs):
        return module.processor(
            module,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask, **cross_attention_kwargs)

    return forward
