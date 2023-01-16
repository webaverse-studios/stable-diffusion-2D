

import base64
import io
from PIL import Image
from pathlib import Path

import oneflow as torch
from diffusers import (
    OneFlowDPMSolverMultistepScheduler as DPMSolverMultistepScheduler,
    OneFlowStableDiffusionInpaintPipeline as StableDiffusionInpaintPipeline,
)


#device = "cuda" means that the model should go in the available GPU, we can
#also make it go to a specific GPU if multiple GPUs are available.
#Example: device = "cuda:2" would cause the model to load into GPU #3
def init_model(local_model_path = "./stable-diffusion-2-depth", device = "cuda"):

  #If the model is Depth assisted Img2Img model
  if 'inpainting' in local_model_path:
    dpm_solver = DPMSolverMultistepScheduler.from_config(local_model_path, subfolder="scheduler")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        local_model_path,
        # use_auth_token=True,
        # revision="fp16",
        torch_dtype=torch.float16,
        scheduler=dpm_solver,
        num_inference_steps=20,
    )
    pipe = pipe.to(device)
    return pipe
  else:
    raise Exception('Only OneFlow inpainting pipeline supported currently!')


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
              mask_image, \
              prompts = ["blue house", "blacksmith workshop"], \
              # strength: float = 0.90,\
              num_inference_steps: int = 20,\
              guidance_scale: float = 8,
              negative_pmpt:str = "ugly, blurry",
              req_type = "inpaint",
              device = "cuda",
              seed = 1024):
  
  if req_type == 'inpaint':
    negative_pmpt = [negative_pmpt for negative_pmpt in range(len(prompts))]

    generator = torch.Generator(device=device).manual_seed(seed)

    images = pipe(prompt=prompts, image=init_img, mask_image=mask_image, num_inference_steps = num_inference_steps,\
                    guidance_scale = guidance_scale, negative_prompt = negative_pmpt, generator = generator)

    images = images[0]
  else:
    raise Exception('inference: Only inpainting supported!!')
      
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