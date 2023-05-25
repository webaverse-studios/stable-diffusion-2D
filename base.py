

import base64
import io
from PIL import Image
import numpy as np
import cv2

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler, StableDiffusionDepth2ImgPipeline, UniPCMultistepScheduler
from diffusers.utils import load_image
from pathlib import Path
import openai
from postprocess import convertPILtocv2, convertcv2toPIL

def init_canny_controlnet(local_model_path = "./control_TopdownBalanced_canny"):
  canny_controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(local_model_path).to("cuda")
  canny_controlnet_pipe.safety_checker = lambda images, clip_input: (images, False)
  canny_controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(canny_controlnet_pipe.scheduler.config)
  return canny_controlnet_pipe

#device = "cuda" means that the model should go in the available GPU, we can
#also make it go to a specific GPU if multiple GPUs are available.
#Example: device = "cuda:2" would cause the model to load into GPU #3
def init_model(local_model_path = "./stable-diffusion-2-depth", device = "cuda"):

  #If the model is Depth assisted Img2Img model
  if 'depth' in local_model_path:
    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16
    )
    pipe = pipe.to(device)
    return pipe
  elif 'magenta' in local_model_path:
    DPM_scheduler = DPMSolverMultistepScheduler(
      beta_start=0.00085,
      beta_end=0.012,
      beta_schedule="scaled_linear",
      num_train_timesteps=1000,
      trained_betas=None,
#       predict_epsilon=True,
      thresholding=False,
      algorithm_type="dpmsolver++",
      solver_order=2,
      solver_type="midpoint",
      lower_order_final=True,
    )
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
      local_model_path,
#       'magenta_model',
      revision="fp16", 
      scheduler = DPM_scheduler,
      torch_dtype=torch.float16,
      safety_checker=None
    )
    pipe = pipe.to(device)
    return pipe

  else:
    #for `diffusers_summerstay_strdwvlly_asset_v2` model
    #----------------------------------------
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
      torch_dtype=torch.float16,
      safety_checker=None
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
def load_image_generalised(image_path, resize = False):
  path = Path(image_path)

  print('Loading file at => ', path)

  init_img = None
  if isinstance(image_path, str) or path.is_file():
      init_img = Image.open(image_path).convert("RGB")
  else:
    # if the image is a string of bytesarray.
    init_img = base64.b64decode(image_path)

  # If the image is sent as bytesarray
  if isinstance(image_path, (bytearray, bytes)):
      init_img = Image.open(io.BytesIO(image_path))
      init_img = init_img.convert("RGB")

  
  #returns a PIL Image
  if resize:
    return init_img.resize((512,512))
  else:
    return init_img


def inference(pipe, \
              init_img,\
              prompts = ["blue house", "blacksmith workshop"], \
              strength: float = 0.90,\
              num_inference_steps: int = 20,\
              guidance_scale: float =20,
              negative_pmpt:str = "ugly, contrast, 3D",
              req_type = "asset",
              device = "cuda",
              seed = None):
  
  # print(prompts)
  prompts_postproc = None
  images = None
  if req_type == 'asset':
    #for `diffusers_summerstay_strdwvlly_asset_v2` model
    # prompts_postproc = [f'{prompt}, surrounded by completely black, strdwvlly style, completely black background, HD, detailed' for prompt in prompts]
    # negative_pmpt = "isometric, interior, island, farm, monochrome, glowing, text, character, sky, UI, pixelated, blurry"

    #for `stable-diffusion-2-depth` model
    adjs = [x.split()[0] for x in prompts]
    prompts_postproc = [f'{prompt}, {adj} style, {adj} appearance, {adj}, pixel art' for prompt, adj in zip(prompts,adjs)]

    if negative_pmpt is not None:  
      negative_prompt = [negative_pmpt for x in range(len(prompts_postproc))]
    else:
      negative_prompt = None
    # print(prompts_postproc[0], '!!!!!!!!!!\n', prompts_postproc[1])

    generator = None
    if seed is not None:
      generator = torch.Generator(device=device).manual_seed(seed)

    with autocast("cuda"):
        images = pipe(prompt=prompts_postproc,\
                    negative_prompt = negative_prompt,\
                    image=init_img, 
                    strength=strength, 
                    num_inference_steps = num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator = generator)
    images = images[0]
  else:
    prompts = [x.replace('tile', 'texture') for x in prompts]
    prompts_postproc = [f'{prompt}, studio ghibli style, cartoon style, smlss style' for prompt in prompts]

    if negative_pmpt is not None:  
      negative_prompt = [negative_pmpt for x in range(len(prompts_postproc))]
    else:
      negative_prompt = ["isometric, interior, island, farm, monochrome, glowing, text, character, sky, UI, pixelated, blurry" for x in range(len(prompts_postproc))]
      
    # print(prompts_postproc[0], '!!!!!!!!!!\n', prompts_postproc[1])

    generator = None
    if seed is not None:
      generator = torch.Generator(device=device).manual_seed(seed)

    with autocast("cuda"):
        images = pipe(prompt=prompts_postproc,\
                    negative_prompt = negative_prompt,\
                    image=init_img, 
                    strength=strength, 
                    num_inference_steps = num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator = generator)
    images = images[0]
    #images = [x.resize((64,64),0).resize((512,512),0) for x in images]
      
  #Returns a List of PIL Images
  return images

def inference_with_edge_guidance(canny_controlnet_pipe, init_image, prompts, negative_pmpt, canny_lower, canny_upper, num_inference_steps = 20):
    # This uses the edges from an init image to guide the generation of a new image.
    # it outputs an image in the standard diffusers format
    # The init image is an image whose outline and major shapes you want preserved in the output
    # Canny_lower and Canny_upper are thresholds on which edges will be kept. 100 for lower and 200 for upper is a good starting point for experimentation. They can go from 1 to 255, I think.
    
    if negative_pmpt is not None:  
      negative_prompt = [negative_pmpt for x in range(len(prompts))]
    else:
      negative_prompt = None

    #Converting PIL Image to OpenCV Image
    init_image = cv2.cvtColor(np.array(init_image), cv2.COLOR_RGB2BGR)
    edge_image = cv2.Canny(init_image,canny_lower,canny_upper)
    
    if len(prompts)==1 and len(negative_pmpt)==1:
      prompts = prompts[0]
      negative_prompt = negative_pmpt[0]

    image = canny_controlnet_pipe(prompt=prompts, negative_prompt = negative_prompt, controlnet_hint=edge_image, num_inference_steps = num_inference_steps).images

    return image


def inference_w_gpt(pipe, \
              init_img,\
              prompts = ["blue house", "blacksmith workshop"], \
              strength: float = 0.90,\
              num_inference_steps: int = 20,\
              guidance_scale: float =20,
              negative_pmpt:str = "base, ground, terrain, child's drawing, sillhouette, dark, shadowed, green blob, cast shadow on the ground, background pattern",
              req_type = "asset",
              device = "cuda",
              seed = 1024):
  
  # print(prompts)
  images = None
  if req_type == 'asset':

    images = []
    #for summerstay's magenta model
    adjs = [x.split()[0] for x in prompts]
    adjectives = [f"{adj} world" for adj in adjs]

    for idx in range(len(prompts)):
      prompt = """In creating art for video games, it is important that everything contributes to an overall style. If the style is 'candy world', then everything should be made of candy:
      * tree: gumdrop fruit and licorice bark
      * flower: lollipops with leaves
      For an 'ancient Japan' setting, the items are simply a variation of the items that might be found in ancient Japan. Some might be unchanged:
      * church: a Shinto shrine
      * tree: a gnarled, beautiful cherry tree that looks like a bonsai tree
      * tree stump: tree stump
      * stone: a stone resembling those in zen gardens
      If the style instead is '""" + adjectives[idx] + """' then the items might be:
    * """ + prompts[idx] + """:"""
      outtext = openai.Completion.create(
          model="davinci",
          prompt=prompt,
                max_tokens=256,
          temperature=0.5,
          stop=['\n','.']
          )
      response = outtext.choices[0].text
      print(prompt, '\n--------------------\n')
      print(response, '\n--------------------')

      prompts_postproc = "robust, thick trunk with visible roots, concept art of " + response + ", " + adjectives[idx] + ", game asset surrounded by pure magenta, view from above, studio ghibli and disney style, completely flat magenta background" 

      generator = None
      if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

      with autocast("cuda"):
        image = pipe(prompt=prompts_postproc,\
                    negative_prompt = negative_pmpt[idx],\
                    image=init_img, 
                    strength=strength, 
                    num_inference_steps = num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator = generator)[0][0]
        images.append(image)
  
      
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

def make_background_magenta(PILforeground_source, PILbackground_source, erode_width, magenta_thresh = 0):
    # take a background source that has a pure magenta background, and replace the background on the foreground source with it.
    # this only works well if the background source was used with controlnet to generate the foreground source.
    # erode_width is how much the background should contract before being applied
    foreground_source = convertPILtocv2(PILforeground_source)
    background_source = convertPILtocv2(PILbackground_source)
    (height, width, colors) = foreground_source.shape
    print(foreground_source.shape)
    background_source = cv2.resize(background_source,(width, height), interpolation=cv2.INTER_LINEAR)
    mask =  np.all(background_source == [255,0,255], axis=-1)
    print(mask.dtype)
    if erode_width>0:
        kernel = np.ones((erode_width, erode_width), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8),kernel)
    else:
        kernel = np.ones((-erode_width, -erode_width), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8),kernel)
    foreground_source[mask.astype(bool)] = [255,0,255]

    # opacity = np.where(foreground_source == [255,0,255])
    # print(opacity[0].shape)
    transparent_array = np.zeros((foreground_source.shape[0],foreground_source.shape[1],4),np.uint8)
    transparent_array[:,:,:3]=foreground_source
    transparent_array[:,:,3]= np.where(foreground_source == [255,0,255], [0], [255])[:,:,0]
    # transparent_array[:,:,3]= np.where((foreground_source[0]>(255-magenta_thresh)) & (foreground_source[2]>(255-magenta_thresh)) & (foreground_source[1]<(magenta_thresh)), [0], [255])[:,:,0]
    
    print(transparent_array.shape)

    output = convertcv2toPIL(transparent_array)
    return output
