import torch
import utils
from utils import AttentionControl, register_attention_control
from diffusers import FluxPipeline

# Unet->Dit. Modified from https://github.com/google/prompt-to-prompt.git
def pre_generation(
        ldm_flux,
        NUM_DIFFUSION_STEPS=8,
        GUIDANCE_SCALE=3.5,
        MAX_NUM_WORDS=512, # prompt sequence length
        height=1024,
        width=1024,
        seed=0,
        prompt="In a bustling train station, a large banner says 'Faster, Greener, Smarter: The AI Train'. The departure board shows 'Express to Tech Valley: 8:30 AM'. A vending machine with the text 'Please Select'.",
        carrier_list=("banner","board","machine") # carriers keywords
):
    # ldm_flux = FluxPipeline.from_pretrained("/share/dnk/checkpoints/FLUX.1-dev/", torch_dtype=torch.bfloat16).to("cuda")
    # ldm_flux.load_lora_weights('/share/dnk/checkpoints/Hyper-SD', weight_name='Hyper-FLUX.1-dev-8steps-lora.safetensors')
    # ldm_flux.fuse_lora(lora_scale=0.15)
    tokenizer = ldm_flux.tokenizer_2

    g_gpu = torch.Generator(device="cuda").manual_seed(seed)
    inds = []  # Index list of all carriers tokens
    for carrier in carrier_list:
        ind = utils.get_word_inds(prompt, carrier, tokenizer)
        inds += ind
    inds = sorted(set(inds))

    class AttentionLastStore(AttentionControl):
        # Only record attention weight of the last step
        @staticmethod
        def get_empty_store():  # 2 categories of transformer blocks
            return {"MM": [], "single": []}

        def forward(self, attn, place_in_transformer: str):
            # Store the attention weight of the current step to step_store and return the original attention weight attn
            if self.cur_step == NUM_DIFFUSION_STEPS - 1:
                key = f"{place_in_transformer}"
                attn_store = attn[:, :, MAX_NUM_WORDS:, inds]
                self.attention_store[key].append(attn_store)
            return attn

        def between_steps(self):
            return

        def get_average_attention(self):
            # Calculate the average attention weight for each category
            average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                                 self.attention_store}
            return average_attention

        def reset(self):
            # Reset the class status to restart the diffusion task
            super(AttentionLastStore, self).reset()
            self.attention_store = self.get_empty_store()

        def __init__(self):
            super(AttentionLastStore, self).__init__()
            self.attention_store = self.get_empty_store()  # Used to store the attention weights of the last step

    controller = AttentionLastStore()

    register_attention_control(ldm_flux, controller)
    _ = ldm_flux(
        prompt=prompt,
        num_inference_steps=NUM_DIFFUSION_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=g_gpu,
        height=height,
        width=width
    )
    res = height // 16
    # Call aggregate_attention function to extract and process attention weights from attention_store, the result of attention_maps: shape is [res, res, num_tokens], and the attention weight distribution of each token is mapped to the latent space
    attention_maps = utils.aggregate_attention(controller, res=res, from_where=("MM", "single"), select=0)
    max_pixels = []  #Store the coordinates of the point with the maximum attention value
    for i in range(attention_maps.shape[-1]):
        flat_index = torch.argmax(attention_maps[:, :, i])
        row = flat_index // res
        row = row / res
        col = flat_index % res
        col = col / res
        max_pixels.append([col.item(), row.item()])
    return max_pixels


if __name__ == "__main__":
    pre_generation()
