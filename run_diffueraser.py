import torch
import os 
import time
import argparse
from diffueraser.diffueraser import DiffuEraser
from propainter.inference import Propainter, get_device

def main():

    ## input params
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', type=str, default="examples/example3/video.mp4", help='Path to the input video')
    parser.add_argument('--input_mask', type=str, default="examples/example3/mask.mp4" , help='Path to the input mask')
    parser.add_argument('--video_length', type=int, default=10, help='The maximum length of output video')
    parser.add_argument('--mask_dilation_iter', type=int, default=8, help='Adjust it to change the degree of mask expansion')
    parser.add_argument('--max_img_size', type=int, default=960, help='The maximum length of output width and height')
    parser.add_argument('--save_path', type=str, default="results" , help='Path to the output')
    parser.add_argument('--ref_stride', type=int, default=10, help='Propainter params')
    parser.add_argument('--neighbor_length', type=int, default=10, help='Propainter params')
    parser.add_argument('--subvideo_length', type=int, default=50, help='Propainter params')
    parser.add_argument('--base_model_path', type=str, default="weights/stable-diffusion-v1-5" , help='Path to sd1.5 base model')
    parser.add_argument('--vae_path', type=str, default="weights/sd-vae-ft-mse" , help='Path to vae')
    parser.add_argument('--diffueraser_path', type=str, default="weights/diffuEraser" , help='Path to DiffuEraser')
    parser.add_argument('--propainter_model_dir', type=str, default="weights/propainter" , help='Path to priori model')
    args = parser.parse_args()
                  
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    priori_path = os.path.join(args.save_path, "priori.mp4")                        
    output_path = os.path.join(args.save_path, "diffueraser_result.mp4") 
    
    ## model initialization
    device = get_device()
    # PCM params
    ckpt = "2-Step"
    video_inpainting_sd = DiffuEraser(device, args.base_model_path, args.vae_path, args.diffueraser_path, ckpt=ckpt)
    video_inpainting_sd.pipeline.to('cpu')
    propainter = Propainter(args.propainter_model_dir, device=device)
    
    start_time = time.time()

    ## priori
    propainter.forward(args.input_video, args.input_mask, priori_path, video_length=args.video_length, 
                        ref_stride=args.ref_stride, neighbor_length=args.neighbor_length, subvideo_length = args.subvideo_length,
                        mask_dilation = args.mask_dilation_iter) 

    propainter.model.cpu()
    torch.cuda.empty_cache()
    video_inpainting_sd.pipeline.to('cuda').to(torch.bfloat16)

    video_inpainting_sd.pipeline.enable_model_cpu_offload()
    video_inpainting_sd.pipeline.enable_xformers_memory_efficient_attention()
    video_inpainting_sd.pipeline.enable_sequential_cpu_offload()
    
    ## diffueraser
    guidance_scale = None    # The default value is 0.  
    video_inpainting_sd.forward(args.input_video, args.input_mask, priori_path, output_path,
                                max_img_size = args.max_img_size, video_length=args.video_length, mask_dilation_iter=args.mask_dilation_iter,
                                guidance_scale=guidance_scale)
    
    end_time = time.time()  
    inference_time = end_time - start_time  
    print(f"DiffuEraser inference time: {inference_time:.4f} s")

    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()


   