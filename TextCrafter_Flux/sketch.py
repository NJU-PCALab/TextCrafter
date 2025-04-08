import torch
from diffusers import FluxPipeline

# Take 8-steps lora as an example
ckpt_name = "Hyper-FLUX.1-dev-8steps-lora.safetensors"
# Load model, please fill in your access tokens since FLUX.1-dev repo is a gated model.
pipe = FluxPipeline.from_pretrained("/share/dnk/checkpoints/FLUX.1-dev/", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights('/share/dnk/checkpoints/Hyper-SD', weight_name='Hyper-FLUX.1-dev-8steps-lora.safetensors')
pipe.fuse_lora(lora_scale=0.15)
image=pipe(prompt="On the table, a note that says 'TextCrafter'. A coffee cup with the word 'IJCAI2025'.", num_inference_steps=10, guidance_scale=3.5).images[0]
image.save("output.png")
