from accelerate import Accelerator
import os
from libs.brushnet_CA import BrushNetModel
from libs.unet_2d_condition import UNet2DConditionModel

## path
base_model_name_or_path = "weights/stable-diffusion-v1-5"
pretrained_brushnet_path = "weights/diffuEraser/brushnet"
output_dir = "weights/converted_weights/diffuEraser-model-stage1/checkpoint-1"
input_dir = "diffuEraser-model-stage1/checkpoint-1"

## load models
accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision='fp16',
)

brushnet = BrushNetModel.from_pretrained(pretrained_brushnet_path)
unet_main = UNet2DConditionModel.from_pretrained(
        base_model_name_or_path, subfolder="unet"
    )
unet_main, brushnet, = accelerator.prepare(
        unet_main, brushnet,
    )
accelerator.load_state(input_dir)  

unet_main = accelerator.unwrap_model(unet_main)
brushnet = accelerator.unwrap_model(brushnet)


## save models
unet_main_path = os.path.join(output_dir, "unet_main")
if not os.path.exists(unet_main_path):
    os.makedirs(unet_main_path)
unet_main.save_pretrained(unet_main_path)


brushnet_path = os.path.join(output_dir, "brushnet")
if not os.path.exists(brushnet_path):
    os.makedirs(brushnet_path)
brushnet.save_pretrained(brushnet_path)


print('load done!')
