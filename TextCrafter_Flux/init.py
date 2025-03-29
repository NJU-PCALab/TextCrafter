import torch

def init_forwards(self, root_module: torch.nn.Module):
    for name, module in root_module.named_modules():
        if (
            "attn" in name
            and "transformer_blocks" in name
            and "single_transformer_blocks" not in name
            and module.__class__.__name__ == "Attention"
        ):
            module.forward = FluxTransformerBlock_init_forward(self, module)           
        elif (
            "attn" in name
            and "single_transformer_blocks" in name
            and module.__class__.__name__ == "Attention"
        ):
            module.forward = FluxSingleTransformerBlock_init_forward(self, module) 

def FluxSingleTransformerBlock_init_forward(self, module):
    def forward(
        hidden_states=None,
        encoder_hidden_states=None,
        image_rotary_emb=None
    ):
        return module.processor(
            module,
            hidden_states=hidden_states,
            image_rotary_emb=image_rotary_emb
        )
    return forward

def FluxTransformerBlock_init_forward(self, module):
    def forward(
        hidden_states=None,
        encoder_hidden_states=None,
        image_rotary_emb=None
    ):
        return module.processor(
            module,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=image_rotary_emb
        )
    return forward