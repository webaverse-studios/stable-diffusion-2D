

import base64
import io
from PIL import Image
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler, EulerDiscreteScheduler
from pathlib import Path

#device = "cuda" means that the model should go in the available GPU, we can
#also make it go to a specific GPU if multiple GPUs are available.
#Example: device = "cuda:2" would cause the model to load into GPU #3
def init_model(local_model_path = "./stable-diffusion-v1-5", device = "cuda"):
  DPM_scheduler = DPMSolverMultistepScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
    trained_betas=None,
    predict_epsilon=True,
    thresholding=False,
    algorithm_type="dpmsolver++",
    solver_type="midpoint",
    lower_order_final=True,
)
  pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    local_model_path,
    revision="fp16", 
    scheduler = DPM_scheduler,
    torch_dtype=torch.float16
  )
  pipe = pipe.to(device)
  return pipe


#'image_path' is a local path to the image
def load_image(image_path):
  init_img = Image.open(image_path).convert("RGB").resize((512, 512))
  #returns a PIL Image
  return init_img


  #'image_path' is a local path to the image or bytearray or bytestream
#reference: https://github.com/pytorch/serve/blob/master/ts/torch_handler/vision_handler.py
def load_image_generalised(image_path):

  path = Path(image_path)

  print('Loading file at => ', path)

  init_img = None
  if isinstance(image_path, str) or path.is_file():
      init_img = Image.open(image_path).convert("RGB").resize((512, 512))
  else:
    # if the image is a string of bytesarray.
    init_img = base64.b64decode(image_path)

  # If the image is sent as bytesarray
  if isinstance(image_path, (bytearray, bytes)):
      init_img = Image.open(io.BytesIO(image_path))
      init_img = init_img.convert("RGB").resize((512, 512))

  
  #returns a PIL Image
  return init_img


def inference(pipe, \
              init_img,\
              prompts = ["blue house", "blacksmith workshop"], \
              strength: float = 0.90,\
              num_inference_steps: int = 20,\
              guidance_scale: float =20,
              negative_pmpt = "isometric, interior, island, farm, monochrome, glowing, text, character, sky, UI, pixelated, blurry",
              req_type = "asset",
              device = "cuda"):
  
  # print(prompts)
  prompts_postproc = None
  images = None
  if req_type == 'asset':
    prompts_postproc = [f'{prompt}, surrounded by completely black, strdwvlly style, completely black background, HD, detailed' for prompt in prompts]
    negative_prompt = [negative_pmpt for x in range(len(prompts_postproc))]
    # print(prompts_postproc[0], '!!!!!!!!!!\n', prompts_postproc[1])

    generator = torch.Generator(device=device).manual_seed(1024)
    with autocast("cuda"):
        images = pipe(prompt=prompts_postproc,\
                    negative_prompt = negative_prompt,\
                    init_image=init_img, 
                    strength=strength, 
                    num_inference_steps = num_inference_steps,
                    guidance_scale=guidance_scale, generator=generator)
    images = images[0]
  else:
    prompts = [x.replace('tile', 'texture') for x in prompts]
    prompts_postproc = [f'{prompt}, studio ghibli style, cartoon style, smlss style' for prompt in prompts]
    negative_prompt = [negative_pmpt for x in range(len(prompts_postproc))]
    # print(prompts_postproc[0], '!!!!!!!!!!\n', prompts_postproc[1])

    generator = torch.Generator(device=device).manual_seed(1024)
    with autocast("cuda"):
        images = pipe(prompt=prompts_postproc,\
                    negative_prompt = negative_prompt,\
                    init_image=init_img, 
                    strength=strength, 
                    num_inference_steps = num_inference_steps,
                    guidance_scale=guidance_scale, generator=generator)
    images = images[0]
    #images = [x.resize((64,64),0).resize((512,512),0) for x in images]
      
  #Returns a List of PIL Images
  return images




  #-------------------------- TEXT2IMG ----------------------------------------

  #device = "cuda" means that the model should go in the available GPU, we can
#also make it go to a specific GPU if multiple GPUs are available.
#Example: device = "cuda:2" would cause the model to load into GPU #3
def init_txt2img_model(local_model_path = "./stable-diffusion-v1-5", device = "cuda"):
  DPM_scheduler = DPMSolverMultistepScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
    trained_betas=None,
    predict_epsilon=True,
    thresholding=False,
    algorithm_type="dpmsolver++",
    solver_type="midpoint",
    lower_order_final=True,
)
  
  pipe = StableDiffusionPipeline.from_pretrained(
    local_model_path,
    revision="fp16", 
    scheduler = DPM_scheduler
    # torch_dtype=torch.float16
  )
  pipe = pipe.to(device)
  return pipe


def inference_txt2img(pipe, \
              prompts = ["blue house", "blacksmith workshop"], \
              strength=0.90,\
              num_inference_steps = 20,\
              guidance_scale=20,
              device = "cuda"):
  
  # print(prompts)
  negative_prompt =  "isometric, terrain, interior, ground, island, farm, at night, dark, ground, monochrome, glowing, text, character, sky, UI, pixelated, blurry, tiled squares"
  prompts_postproc = [f'top-down view of a {prompt}, surrounded by completely black, stardew valley, strdwvlly style, completely black background, HD, detailed, clean lines, realistic' for prompt in prompts]
  negative_prompt = [negative_prompt for x in range(len(prompts_postproc))]
  # print(prompts_postproc[0], '!!!!!!!!!!\n', prompts_postproc[1])

  generator = torch.Generator(device=device).manual_seed(1024)
  with autocast("cuda"):
      images = pipe(prompt=prompts_postproc,\
                  negative_prompt = negative_prompt,\
                  strength=strength, 
                  num_inference_steps = num_inference_steps,
                  guidance_scale=guidance_scale, generator=generator)
      
  #Returns a List of PIL Images
  return images[0]