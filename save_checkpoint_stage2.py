from accelerate import Accelerator
import os
from libs.brushnet_CA import BrushNetModel
from libs.unet_2d_condition import UNet2DConditionModel
from libs.unet_motion_model import MotionAdapter, UNetMotionModel

## path
base_model_name_or_path = "weights/stable-diffusion-v1-5"
pretrained_brushnet_path = "weights/diffuEraser/brushnet"
motion_path = "weights/animatediff-motion-adapter-v1-5-2"
output_dir = "weights/converted_weights/diffuEraser-model-stage2/checkpoint-2"
input_dir = "diffuEraser-model-stage2/checkpoint-2"

## load models
accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision='fp16',
)

unet = UNet2DConditionModel.from_pretrained(
    base_model_name_or_path, subfolder="unet"
)
motion_adapter = MotionAdapter.from_pretrained(motion_path)
unet_main = UNetMotionModel.from_unet2d(unet, motion_adapter)

brushnet = BrushNetModel.from_pretrained(pretrained_brushnet_path)

unet_main, brushnet = accelerator.prepare(
    unet_main, brushnet
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
