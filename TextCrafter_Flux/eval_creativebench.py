import fire
import torch
import os

from diffusers import FluxPipeline
from tqdm import tqdm
import json

from textcrafter_pipeline_flux import textcrafter_FluxPipeline
from pre_generation import pre_generation
from rectangles import generate_rectangles_gurobi, visualize_rectangles,generate_rectangles_random,generate_rectangles_fixed

ldm_flux = FluxPipeline.from_pretrained("/share/dnk/checkpoints/FLUX.1-dev/",torch_dtype=torch.bfloat16).to("cuda")
pipe = textcrafter_FluxPipeline.from_pipeline(ldm_flux)

@torch.no_grad()
def inference(
        prompt,
        sentence_list,
        insulation_steps=10,
        num_inference_steps=30,  # Sampling steps
        cross_replace_steps=1.0,  # Reweight execution steps(ratio)
        seed=0,
        height=1024,
        width=1024,
        area=None
):

    rectangles = generate_rectangles_fixed(area=1)  # fixed layout


    insulation_m_offset_list = []  # x_min of bbox
    insulation_n_offset_list = []  # y_min of bbox
    insulation_m_scale_list = []  # width of bbox
    insulation_n_scale_list = []  # height of bbox
    for i, rect in enumerate(rectangles):
        insulation_m_offset_list.append(rect['m_offset'])
        insulation_n_offset_list.append(rect['n_offset'])
        insulation_m_scale_list.append(rect['m_scale'])
        insulation_n_scale_list.append(rect['n_scale'])

    image = pipe(
        sentence_list=sentence_list,
        insulation_m_offset_list=insulation_m_offset_list,
        insulation_n_offset_list=insulation_n_offset_list,
        insulation_m_scale_list=insulation_m_scale_list,
        insulation_n_scale_list=insulation_n_scale_list,
        insulation_steps=insulation_steps,
        carrier_list=None,
        cross_replace_steps=cross_replace_steps,
        seed=seed,
        addition=None,
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=3.5,
    ).images[0]

    return image


def main(
        area=1
        ):

    output_dir = "TextCrafter"

    for benchmark in ("CreativeBench",):
        with open(f"/share/dnk/benchmark/{benchmark}/{area}.json", 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        # 获取 "data_list"
        data_list = json_data.get("data_list")
        for data in tqdm(data_list):
            index = data.get("index")
            # if index < 238: continue
            prompt = data.get("prompt")
            sentence_list = data.get("sentence_list")
            image = inference(prompt, sentence_list,area=1)
            filename = os.path.join(f"/share/dnk/IJCAI-eval/{output_dir}/{benchmark}/{area}", f"{index}.png")
            image.save(filename)


if __name__ == '__main__':
    fire.Fire(main)
