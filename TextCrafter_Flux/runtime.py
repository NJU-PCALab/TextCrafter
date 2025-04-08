import fire
import torch
import os

from diffusers import FluxPipeline
from tqdm import tqdm
import json
import time  # 导入time模块用于测量时间

from textcrafter_pipeline_flux import textcrafter_FluxPipeline
from pre_generation import pre_generation
from rectangles import generate_rectangles_gurobi, visualize_rectangles,generate_rectangles_random

ldm_flux = FluxPipeline.from_pretrained("/share/dnk/checkpoints/FLUX.1-dev/",torch_dtype=torch.bfloat16).to("cuda")
ldm_flux.load_lora_weights('/share/dnk/checkpoints/Hyper-SD', weight_name='Hyper-FLUX.1-dev-8steps-lora.safetensors')
ldm_flux.fuse_lora(lora_scale=0.15)
pipe = textcrafter_FluxPipeline.from_pipeline(ldm_flux)

@torch.no_grad()
def inference(
        prompt,
        carrier_list,
        sentence_list,
        min_area=None,
        pre_generation_steps=2,
        insulation_steps=2,
        num_inference_steps=10,  # Sampling steps
        cross_replace_steps=1.0,  # Reweight execution steps(ratio)
        seed=0,
        addition=0.4,  # embed addition coefficient
        height=1024,
        width=1024,
        rectangle_name=None,
        area=None
):
    max_pixels = pre_generation(
        ldm_flux=ldm_flux,
        NUM_DIFFUSION_STEPS=pre_generation_steps,
        height=height,
        width=width,
        seed=seed,
        prompt=prompt,
        carrier_list=carrier_list
    )  # Pre-generation
    rectangles = generate_rectangles_gurobi(points=max_pixels, min_area=min_area)  # Layout-optimizer
    # rectangles = generate_rectangles_random(area=area) # random layout

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
        area=2,
        min_area=None
):
    min_area_default = (0.65, 0.3, 0.2, 0.15, 0.12)
    if min_area is None:  # 如果没提供，使用默认值
        min_area = min_area_default[area - 1]

    output_dir = "runtime"

    # for benchmark in ("CVTG", "CVTG-Style"):
    for benchmark in ("CVTG",):
        with open(f"/share/dnk/benchmark/{benchmark}/{area}.json", 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        # 获取 "data_list"
        data_list = json_data.get("data_list")

        # 初始化计时相关变量
        sample_count = 0
        total_time = 0
        execution_times = []  # 存储每个样本的执行时间

        for data in tqdm(data_list):
            index = data.get("index")
            # if index < 238: continue
            prompt = data.get("prompt")
            carrier_list = data.get("carrier_list")
            sentence_list = data.get("sentence_list")
            rectangle_name = f"/share/dnk/IJCAI-eval/{output_dir}/{benchmark}/{area}/{index}_rec.json"

            # 计时 inference 函数
            start_time = time.time()
            image = inference(prompt, carrier_list, sentence_list, min_area=min_area, rectangle_name=rectangle_name, area=area)
            end_time = time.time()

            # 计算执行时间
            execution_time = end_time - start_time

            # 跳过第一个样本的计时，统计后面50个样本
            sample_count += 1
            if sample_count > 1:  # 跳过第一个样本
                execution_times.append(execution_time)

            filename = os.path.join(f"/share/dnk/IJCAI-eval/{output_dir}/{benchmark}/{area}", f"{index}.png")
            image.save(filename)

            # 处理51个样本后停止循环（1个跳过 + 20个统计）
            if sample_count >= 21:
                break

        # 计算并打印平均执行时间
        if len(execution_times) > 0:
            avg_time = sum(execution_times) / len(execution_times)
            print(f"Benchmark: {benchmark}, 平均执行时间: {avg_time:.4f} 秒")


if __name__ == '__main__':
    fire.Fire(main)
