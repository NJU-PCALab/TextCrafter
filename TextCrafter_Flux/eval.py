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
        carrier_list,
        sentence_list,
        min_area=None,
        pre_generation_steps=8,
        insulation_steps=5,
        num_inference_steps=30,  # Sampling steps
        cross_replace_steps=1.0,  # Reweight execution steps(ratio)
        seed=0,
        addition=0.4,  # embed addition coefficient
        height=1024,
        width=1024,
        rectangle_name=None,
        area=None
):
    # max_pixels = pre_generation(
    #     ldm_flux=ldm_flux,
    #     NUM_DIFFUSION_STEPS=pre_generation_steps,
    #     height=height,
    #     width=width,
    #     seed=seed,
    #     prompt=prompt,
    #     carrier_list=carrier_list
    # )  # Pre-generation
    # rectangles = generate_rectangles_gurobi(points=max_pixels, min_area=min_area)  # Layout-optimizer
    # rectangles = generate_rectangles_random(area=area) # random layout
    rectangles = generate_rectangles_fixed(area=area)  # fixed layout

    # visualize_rectangles(rectangles=rectangles, points=max_pixels, filename=rectangle_name)
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

    return image


def main(
        benchmark="MiniBenchmark",
        area=1,
        min_area=None
        ):
    min_area_default = (0.65, 0.3, 0.2, 0.15, 0.12)
    if min_area is None:  # 如果没提供，使用默认值
        min_area = min_area_default[area - 1]

    output_dir = "fixed"

    # for benchmark in ("CVTG","CVTG-Style"):
    for benchmark in ("CVTG-Style",):
        with open(f"/share/dnk/benchmark/{benchmark}/{area}.json", 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        # 获取 "data_list"
        data_list = json_data.get("data_list")
        for data in tqdm(data_list):
            index = data.get("index")
            # if index < 199: continue
            prompt = data.get("prompt")
            carrier_list = data.get("carrier_list")
            sentence_list = data.get("sentence_list")
            rectangle_name = f"/share/dnk/IJCAI-eval/{output_dir}/{benchmark}/{area}/{index}_rec.json"
            image = inference(prompt, carrier_list, sentence_list, min_area=min_area,rectangle_name=rectangle_name,area=area)
            filename = os.path.join(f"/share/dnk/IJCAI-eval/{output_dir}/{benchmark}/{area}", f"{index}.png")
            image.save(filename)


if __name__ == '__main__':
    fire.Fire(main)
