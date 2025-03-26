import os
import random
from PIL import Image
import numpy as np
import csv
from decord import VideoReader, cpu
import torch
import torchvision.transforms as transforms
from dataset.utils import (create_random_shape_with_random_motion, Stack,
                        ToTorchFormatTensor, GroupRandomHorizontalFlip)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer):
        self.args = args

        self.video_root = args.train_data_dir
        self.csv_file = args.csv_file
        self._load_metadata()

        self.nframes = args.nframes
        self.size =  args.resolution

        self.tokenizer = tokenizer
        self._stack = transforms.Compose([
            Stack(),
        ])
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])  # [-1, 1]
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])  # [0, 1]

    def __len__(self):
        return len(self.metadata)
    
    def _load_metadata(self):
        self.metadata = []
        for csv_f in self.csv_file:
            with open(csv_f, 'r', encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    self.metadata.append(row)
    
    def tokenize_captions(self, caption, is_train=True):
        if random.random() < self.args.proportion_empty_prompts:
            caption=""
        elif isinstance(caption, str):
            caption=caption
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            caption=random.choice(caption) if is_train else caption[0]
        else:
            raise ValueError(
                f"Caption column `{self.args.caption_column}` should contain either strings or lists of strings."
            )
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids
    
    def _aug(self, frame, transform=None, state=None):
        if state is not None:
            torch.set_rng_state(state)
        frame_transform = transform(frame) if transform is not None else frame
        return frame_transform
        
    def __getitem__(self, index):

        while True:
            video_path = self.metadata[index][self.args.video_column]
            vid_path = os.path.join(self.video_root, video_path)
            caption = self.metadata[index][self.args.caption_column]

            ## read video
            try:
                video_reader = VideoReader(vid_path, ctx=cpu(0))
                if len(video_reader) < self.nframes:
                    print(f"video length ({len(video_reader)}) is smaller than target length({self.nframes})")
                    index += 1
                    continue
            except:
                index += 1
                print(f"Load video failed! path={vid_path}")
                continue

            ## sample
            fps = video_reader.get_avg_fps()
            frame_stride = max(int(fps // 15), 1)
            if frame_stride != 1:
                all_frames = list(range(0, len(video_reader), frame_stride))
                if len(all_frames) < self.nframes:
                    fs = len(video_reader) // self.nframes
                    assert(fs != 0)
                    all_frames = list(range(0, len(video_reader), fs))
            else:
                all_frames = list(range(len(video_reader)))

            # select a random clip
            rand_idx = random.randint(0, len(all_frames) - self.nframes) 
            frame_indices = all_frames[rand_idx: rand_idx + self.nframes]
            try:
                video = video_reader.get_batch(frame_indices).asnumpy()  # [f h w c]
            except:
                print(f"Get frames failed! path = {vid_path}")
                index += 1
                continue

            # create masks
            img_size = Image.fromarray(video[0]).size         
            all_masks = create_random_shape_with_random_motion(
                len(video), imageHeight=img_size[1], imageWidth=img_size[0])

            break

        assert (video.shape[0] == self.nframes), f'{len(video)}, self.nframes={self.nframes}'
 
        # read video frames
        frames = []
        masks = []
        masked_images = []
        state = torch.get_rng_state()
        for idx in range(self.nframes):
            img = Image.fromarray(video[idx])
            masked_image = img*(1.0 - np.array(all_masks[idx])[:,:,np.newaxis].astype(np.float32)/255) 
            masked_image = Image.fromarray(masked_image.astype(np.uint8))

            img = self._aug(img, self.transform, state)
            frames.append(img)
            masked_image = self._aug(masked_image, self.transform, state) 
            masked_images.append(masked_image)

            mask = Image.fromarray(255-np.array(all_masks[idx])) # hole denoted as 0, reserved as 255 [h,w,1]
            mask = self._aug(mask, self.mask_transform, state)
            masks.append(mask) 
            
            if len(frames) == self.nframes: # random reverse
                if random.random() < 0.5:
                    frames.reverse()
                    masks.reverse()
                    masked_images.reverse()

        # normalizate, to tensors
        frame_tensors = torch.stack(frames) #[-1,1]
        conditioning_pixel_values = torch.stack(masked_images) #[-1,1]
        mask_tensors = torch.stack(masks) #[0,1]
        input_ids = self.tokenize_captions(caption)[0]
    
        return {
                "pixel_values": frame_tensors,
                "conditioning_pixel_values": conditioning_pixel_values,
                "masks":mask_tensors,
                "input_ids": input_ids,
            }
