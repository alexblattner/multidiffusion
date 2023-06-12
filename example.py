from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler,ModelMixin, EulerAncestralDiscreteScheduler
from PIL import Image
from io import BytesIO
from textEmbedder import text_embeddings
import torch

from MultiDiffusionPipeline import MultiStableDiffusion
#this was never tested lol
pipe = MultiStableDiffusion.from_pretrained("runwayml/stable-diffusion-v1-5",local_files_only=True, torch_dtype=torch.float16,safety_checker=None,requires_safety_checker=False).to("cuda")
prompt=["a beautiful park","a beautiful sky"]
negative_prompt=["a beautiful sky", "a beautiful park"]

promptE=[]
negative_promptE=[]
for i in range(len(prompt)):
    cond, uncond =text_embeddings(pipe,prompt[i],"", clip_stop_at_last_layers=2)
    promptE.append(cond)
    
buffer=open('Untitled.png', 'rb')
buffer.seek(0)
image_bytes = buffer.read()
imageM = Image.open(BytesIO(image_bytes))
pos=["0:0-512:512",imageM]#the first position should occupy the entirety of the picture size. it works the following: x0:y0-x1:y1 in pixels. the image passed is in black and white
mask_types=[1,12] #the numbers here are essentially the equivalent of z-index in css less strength to background and more strength to mask.
#loras_apply={1:["lora name in dir"]}  this applies the lora for the second area (the mask in this case)

generator = torch.manual_seed(2733424006)
# controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", local_files_only=True,torch_dtype=torch.float16).to("cuda")
# pipe.loadControlnet(controlnet)      this is the only way to load controlnet and also works with multicontrolnet by making the controlnet variable a list of controlnets
image = pipe(
    prompt=None,
    negative_prompt=None,
    prompt_embeds=promptE,
    # negative_prompt_embeds=negative_promptE,
    pos=pos,
    # image=image,
    height=512,
    width=512,
    # controlnet_image=images,
    # controlnet_conditioning_scale=1.0,
    # loras_apply=loras_apply,
    mask_types=mask_types,
    num_inference_steps=20,
    generator=generator,
    guidance_scale=7.0,
).images[0]

image.save("output.png")
