import torch
from diffusers import AutoPipelineForText2Image
import os
os.environ['HF_HOME'] = '/work1/sinclair/yiwei357/resources/sd-xl/data'
os.environ['TRANSFORMERS_CACHE'] = '/work1/sinclair/yiwei357/resources/sd-xl'
pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipeline = pipeline.to("cuda:0")

batch_size = 16
image_resolution = 512
num_inference_steps = 100
denoising_iterations = 50
prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

images = pipeline(
    prompt=prompt,
    num_images_per_prompt=batch_size,
    height=image_resolution,
    width=image_resolution,
    num_inference_steps=num_inference_steps,
    guidance_scale=0.0,
).images
print("done")