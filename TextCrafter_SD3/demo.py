import fire
import torch
from textcrafter_pipeline_sd3 import textcrafter_SD3Pipeline
from pre_generation import pre_generation
from rectangles import generate_rectangles_gurobi, visualize_rectangles


@torch.no_grad()
def main(
        pre_generation_steps=8,  # Pre-generation steps
        insulation_steps=3,
        # The number of insulation is higher, which means the object position is more accurately controlled, but the boundaries may be more obvious.
        num_inference_steps=30,  # Sampling steps
        cross_replace_steps=1.0,  # Reweight execution steps(ratio)
        seed=0,
        min_area=0.12,  # Minimum layout area, adjust according to the number of regions
        addition=0.4  # embed addition coefficient
):
    prompt = "A movie poster with the title 'Summer Escape' in large bold white letters at the top, a tagline reading 'Feel the Adventure' in medium italic, a rating with '8.3/10' in small regular, a director name 'J. Doe' in medium, and a release date 'July 2025' in small."
    carrier_list = [
        "title",
        "tagline",
        "rating",
        "name",
        "date"
    ]
    sentence_list = [
        "the title 'Summer Escape' in large bold white letters at the top.",
        "a tagline reading 'Feel the Adventure' in medium italic.",
        "a rating with '8.3/10' in small regular.",
        "a director name 'J. Doe' in medium.",
        "a release date 'July 2025' in small."
    ]

    height, width = 1024, 1024
    max_pixels = pre_generation(
        NUM_DIFFUSION_STEPS=pre_generation_steps,
        height=height,
        width=width,
        seed=seed,
        prompt=prompt,
        carrier_list=carrier_list
    )  # Pre-generation
    rectangles = generate_rectangles_gurobi(points=max_pixels, min_area=min_area)  # Layout-optimizer
    visualize_rectangles(rectangles=rectangles, points=max_pixels)
    torch.cuda.empty_cache()

    insulation_m_offset_list = []  # x_min of bbox
    insulation_n_offset_list = []  # y_min of bbox
    insulation_m_scale_list = []  # width of bbox
    insulation_n_scale_list = []  # height of bbox
    for i, rect in enumerate(rectangles):
        insulation_m_offset_list.append(rect['m_offset'])
        insulation_n_offset_list.append(rect['n_offset'])
        insulation_m_scale_list.append(rect['m_scale'])
        insulation_n_scale_list.append(rect['n_scale'])

    pipe = textcrafter_SD3Pipeline.from_pretrained("/nasdata/dnk/checkpoints/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16).to("cuda")
    pipe.text_encoder.to(torch.bfloat16)
    pipe.text_encoder_2.to(torch.bfloat16)
    pipe.text_encoder_3.to(torch.bfloat16)  # stable diffusion3 bug

    image = pipe(
        sentence_list=sentence_list,
        insulation_m_offset_list=insulation_m_offset_list,
        insulation_n_offset_list=insulation_n_offset_list,
        insulation_m_scale_list=insulation_m_scale_list,
        insulation_n_scale_list=insulation_n_scale_list,
        insulation_steps=insulation_steps,
        carrier_list=carrier_list,
        cross_replace_steps=cross_replace_steps,
        seed=seed,
        addition=addition,
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=3.5,
    ).images[0]

    filename = "demo.png"
    image.save(filename)
    print(f"image saved as {filename}")
    del pipe, image, filename
    torch.cuda.empty_cache()


if __name__ == '__main__':
    fire.Fire(main)
