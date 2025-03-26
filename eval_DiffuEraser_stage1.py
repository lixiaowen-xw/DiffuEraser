import os
import cv2
from PIL import Image
import numpy as np
import imageio
from dataset.img_util import imfrombytes
from dataset.file_client import FileClient

from transformers import AutoTokenizer, PretrainedConfig
import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
)

from libs.brushnet_CA import BrushNetModel
from libs.unet_2d_condition import UNet2DConditionModel
from diffueraser.pipeline_diffueraser_stage1 import StableDiffusionDiffuEraserPipelineStageOne
    
## args
base_model_name_or_path = "weights/stable-diffusion-v1-5"
vae_path = "weights/sd-vae-ft-mse"
pretrained_stage1_path = "weights/converted_weights/diffuEraser-model-stage1/checkpoint-1"
validation_images=['data/eval/DAVIS/JPEGImages/480p/bear','data/eval/DAVIS/JPEGImages/480p/boat']
validation_masks=['data/eval/DAVIS/Annotations/480p/bear','data/eval/DAVIS/Annotations/480p/boat']
validation_prompts = ["clean background", "clean background"]
output_path = 'outputs/output_stage1'
nframes = 10
seed = None
revision = None

if not os.path.exists(output_path):
    os.makedirs(output_path)

## load models
def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")
    
vae = AutoencoderKL.from_pretrained(vae_path)
noise_scheduler = DDPMScheduler.from_pretrained(base_model_name_or_path, 
        subfolder="scheduler",
        prediction_type="v_prediction",
        timestep_spacing="trailing",
        rescale_betas_zero_snr=True
    )
tokenizer = AutoTokenizer.from_pretrained(
            base_model_name_or_path,
            subfolder="tokenizer",
            use_fast=False,
        )
text_encoder_cls = import_model_class_from_model_name_or_path(base_model_name_or_path,revision)
text_encoder = text_encoder_cls.from_pretrained(
        base_model_name_or_path, subfolder="text_encoder"
    )
brushnet = BrushNetModel.from_pretrained(pretrained_stage1_path, subfolder="brushnet")
unet_main = UNet2DConditionModel.from_pretrained(
    pretrained_stage1_path, subfolder="unet_main",
)

## pipeline
pipeline = StableDiffusionDiffuEraserPipelineStageOne.from_pretrained(
    base_model_name_or_path,
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet_main,
    brushnet=brushnet,
).to("cuda", torch.float16)
pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline.set_progress_bar_config(disable=True)


## inference
def save_videos_grid(video, path: str, duration=125):#fps=8
    outputs = []
    for img in video:
        outputs.append(np.array(img))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, duration=duration, loop=0)

file_client = FileClient('disk')
for validation_prompt, validation_image, validation_mask in zip(validation_prompts, validation_images, validation_masks):
    if os.path.isdir(validation_image):

        frame_list = sorted(os.listdir(validation_image))
        v_len = len(frame_list)
        selected_index = list(range(v_len))[:nframes]

        frames = []
        masks = []
        masked_images = []
        for idx in selected_index:
            frame_path = os.path.join(validation_image, frame_list[idx])

            ## image
            img_bytes = file_client.get(frame_path, 'input')
            img = imfrombytes(img_bytes, float32=False)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            frames.append(img)

            ## mask
            mask_path = os.path.join(validation_mask, str(idx).zfill(5) + '.png')
            mask = Image.open(mask_path).convert('L')
            mask = np.asarray(mask)
            m = np.array(mask > 0).astype(np.uint8)
            m = cv2.dilate(m,
                            cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                            iterations=4)
            mask = Image.fromarray(m * 255)
            masks.append(mask)

            ## masked image
            masked_image = np.array(img)*(1-(np.array(mask)[:,:,np.newaxis].astype(np.float32)/255))
            masked_image = Image.fromarray(masked_image.astype(np.uint8))
            masked_images.append(masked_image)

        validation_masks_input = masks
        validation_images_input = masked_images

        if seed is None:
            generator = None
        else:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        ## forward
        with torch.no_grad():
            images = pipeline(
                num_frames=nframes, prompt=validation_prompt, images=validation_images_input, 
                masks=validation_masks_input, num_inference_steps=50, generator=generator,
                guidance_scale=0.0
            ).frames

        image_name = validation_image.split("/")[-1]
        save_videos_grid(images, f"{output_path}/{image_name}_res.gif")
        save_videos_grid(frames, f"{output_path}/{image_name}_input.gif")
        save_videos_grid(masked_images, f"{output_path}/{image_name}_maskedimages.gif")



