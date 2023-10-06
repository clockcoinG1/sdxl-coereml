#!/Users/clockcoin/anaconda3/bin/python3.11
from abc import ABC, abstractmethod
import math, os, argparse, datetime, time, logging
from compel import Compel, ReturnedEmbeddingsType
from diffusers import  DiffusionPipeline , loaders, DPMSolverMultistepScheduler
from diffusers.utils.pil_utils import make_image_grid

import torch
import accelerate 
import os
from xattr import xattr

class ImageSaver(ABC):
    @abstractmethod
    def save(self, images):
        pass

class PNGImageSaver(ImageSaver):
    def save(self, images):
        # implementation for saving PNG images
        for i, image in enumerate(images):
            unique_name = f"image_{i}.png"
            image.save(unique_name)
            print(f"Image saved as {unique_name}")

class TextFlasher(ABC):
    @abstractmethod
    def flash_text(self, text):
        pass

class ConsoleTextFlasher(TextFlasher):
    def flash_text(self, text):
        # implementation for flashing text on console
        print(text)


class PipeClass:
    def __init__(self):
        self.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            safety_checker=None,
            use_safetensors=True,
        ).to("mps")
        accelerate.init_on_device(device='mps')
        self.pipe.load_lora_weights(f"{os.environ['HOME']}/Downloads/", weight_name=lora_model)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="dpmsolver++",
        )
    
class ConditioningGenerator:
    def __init__(self, pipe):
        self.compel = Compel(tokenizer=[pipe.tokenizer, pipe.tokenizer_2] , text_encoder=[pipe.text_encoder, pipe.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])

    def generate_conditioning(self, prompt, nprompt):
        conditioning, pooled = self.compel(prompt) 
        nconditioning, npooled = self.compel(nprompt)
        [conditioning, nconditioning] = self.compel.pad_conditioning_tensors_to_same_length([conditioning, nconditioning])
        return conditioning, pooled, nconditioning, npooled

class ImageGenerator:
    def __init__(self, pipe, width, height, num_images, num_steps, conditioning_generator):
        self.pipe = pipe
        self.width = width
        self.height = height
        self.num_images = num_images
        self.num_steps = num_steps
        self.conditioning_generator = conditioning_generator

    def generate_images(self, prompt, nprompt):
        conditioning, pooled, nconditioning, npooled = self.conditioning_generator.generate_conditioning(prompt, nprompt)
        images = []
        for _ in range(self.num_images):
            image = self.pipe.generate_image(self.width, self.height, conditioning, nconditioning, pooled, npooled, self.num_steps)
            images.append(image)
        return images

class Application:
    def __init__(self, image_saver, text_flasher, image_generator):
        self.image_saver = image_saver
        self.text_flasher = text_flasher
        self.image_generator = image_generator

    def run(self, prompt, nprompt):
        images = self.image_generator.generate_images(prompt, nprompt)
        self.image_saver.save(images)
        self.text_flasher.flash_text("Images saved successfully")

width=1024
height=1024
num_steps=25
num_images=1
nprompt = "(low quality, worst quality:1.4), (bad anatomy), (inaccurate limb:1.2), ugly, deformed, (extra arms:1.2)"
prompt=""
lora_model="sdxl.safetensors"

pipe = PipeClass().pipe
conditioning_generator = ConditioningGenerator(pipe)

image_generator = ImageGenerator(pipe, width, height, num_images, num_steps, conditioning_generator)
image_saver = PNGImageSaver()
text_flasher = ConsoleTextFlasher()

app = Application(image_saver, text_flasher, image_generator)
app.run(prompt, nprompt)


# compel = Compel(tokenizer=[pipe.tokenizer, pipe.tokenizer_2] , text_encoder=[pipe.text_encoder, pipe.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])
# conditioning, pooled = compel(prompt) 
# nconditioning, npooled = compel(nprompt)
# [conditioning, nconditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, nconditioning])

# pipe.feature_extractor = None
# pipe.safety_checker = None
# images = generate_images(pipe, width, height, conditioning, nconditioning, pooled, npooled, num_images, num_steps)
# dimensions = f"{height}x{width}_{num_steps}.steps"

# save_images(images[0], prompt, lora_model, dimensions)

# if show:
#     num_images = len(images)
#     sqrt_num_images = math.sqrt(num_images)
#     cols = math.ceil(sqrt_num_images)
#     rows = math.floor(sqrt_num_images)
#     if rows * cols < num_images:
#         rows += 1
#     make_image_grid(images=images, rows=rows , cols=cols).show(title=prompt)

# del pipe
# torch.mps.empty_cache()
# width = 1024
# height = 768
# conditioning = "conditioning"  # replace with your actual value
# nconditioning = "nconditioning"  # replace with your actual value
# pooled = "pooled"  # replace with your actual value
# npooled = "npooled"  # replace with your actual value
# num_images = 10
# num_steps = 5

# image_generator = ImageGenerator(pipe, width, height, conditioning, nconditioning, pooled, npooled, num_images, num_steps)
# app = Application(image_saver, text_flasher, image_generator)
# app.run()
