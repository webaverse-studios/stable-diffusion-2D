import os

import urllib.request

from typing import Any, List
import torch 

from base import init_model, make_background_magenta, load_image_generalised, inference, inference_w_gpt, inference_with_edge_guidance, init_canny_controlnet
from postprocess import cut, cutv2, cut_magenta, remBgPil, splitHeightTo2, splitImageTo9, img2b4
from cog import BasePredictor, BaseModel, File, Input, Path


print('cuda status is',torch.cuda.is_available())

#Magenta model for txt2img generation, does not use ControlNet as there is no input image
# pipe_txt2img = init_model(local_model_path = "./diffusers_TopdownBalanced")

try:  
    mode = os.environ["MODEL"]
    if mode == "trees":
        pipe_tree = init_model(local_model_path = "/AI/summerstay/diffusers/buildings")
    if mode == "furniture":
        pipe_furniture = init_model(local_model_path = "/AI/summerstay/diffusers/furniture")
    if mode == "buildings":
        pipe_building = init_model(local_model_path = "/AI/summerstay/diffusers/trees")
    if mode == "plants":
        pipe_plants = init_model(local_model_path = "/AI/summerstay/diffusers/plants")
    
        


except KeyError: 
    print ("Please set the environment variable MODEL, running all models")
    pipe_building = init_model(local_model_path = "/AI/summerstay/diffusers/buildings", device="cuda:0")
    pipe_furniture = init_model(local_model_path = "/AI/summerstay/diffusers/furniture", device="cuda:1")
    pipe_tree = init_model(local_model_path = "/AI/summerstay/diffusers/trees", device="cuda:2")
    pipe_plants = init_model(local_model_path = "/AI/summerstay/diffusers/plants", device="cuda:0")


def separate_prompts(inp_str: str):
  prompts = [x.strip() for x in inp_str.split(':')]
  return prompts


def run_predict(
        input: str,
        prompts: str = Input(description="Prompts", default="blue house: fire cathedral   "),
        strength: float = Input(description="Denoising strength of Stable Diffusion", default=0.85),
        guidance_scale: float = Input(description="Prompt Guidance strength/Classifier Free Generation strength of Stable Diffusion", default=7.5),
        split : str = Input(description="Decide which split needs to happen", default= None),
        req_type: str = Input(description="Describes whether the request is for an object asset or a tile", default="asset"),
        negative_prompt: str = Input(description="Negative_Prompt", default="base, ground, terrain, child's drawing, sillhouette, dark, shadowed, green blob, cast shadow on the ground, background pattern"),
        num_inference_steps: int = Input(description="Number of denoising steps", default = 20),
        sd_seed:int = Input(description="Seed for SD generations for getting deterministic outputs", default = 1000),
        width:int = Input(description="Width for returning output image", default = 512),
        height:int = Input(description="Height for returning output image", default = 512),
        model: str = Input(description="Model to use based on the type of object you want to create. Options are tree, building, or furniture.", default = "furniture")
    ) -> Any:
        """Run a single prediction on the model"""
        try:
            global pipe_asset_magenta, pipe_asset_pixel, pipe_tree, pipe_building, pipe_furniture
            
            print('Starting Prediction')
            init_img = load_image_generalised(input, resize = True)
            print('Load Image is Completed')

            orig_img_dims = load_image_generalised(input, resize = False).size

            prompts = separate_prompts(prompts)

            print('Prompt Split')

            if negative_prompt is not None:  
                negative_prompt = [negative_prompt for x in range(len(prompts))]

            images = None
            if req_type == 'asset_img2img':
              if model == 'tree':
                used_model = pipe_tree
              elif model == 'building':
                used_model = pipe_building
              elif model == 'plants':
                used_model = pipe_plants
              else:
                used_model = pipe_furniture
              
              print('Model selected about to inference')
              images = inference_w_gpt(used_model, init_img, \
                          prompts = prompts, \
                          negative_pmpt = negative_prompt,
                          strength = strength,
                          guidance_scale = guidance_scale,
                          req_type = req_type,
                          num_inference_steps = num_inference_steps,
                          seed = sd_seed)
              
            print('Type of each image: ', type(images[0]))

            external_ip = urllib.request.urlopen('https://ident.me').read().decode('utf8')

            images_ = []


            if req_type != "tile":
                print('Removing Image Background')
                for gen_image in images:
                    # images_.append(cutv2(gen_image, init_img, outer_tolerance = cut_outer_tol, inner_tolerance = cut_inner_tol, radius = cut_radius))
                    # images_.append(cut_magenta(gen_image, outer_tol))
                    print("Sending Image for REMBG")
                    images_.append(remBgPil(gen_image))
            else:
                for image in images:
                    print('[SKIP] Removing Image Background')
                    images_.append(image)

            if height is None or width is None:
                height = orig_img_dims[0]
                width = orig_img_dims[1]


            print('Resizing Images')
            images_ = [img.resize((height,width)) for img in images_]

            splitted_images = []

            for cutImage in images_:
                if split == "splitHeightTo2":
                    splitted_images.append(splitHeightTo2(cutImage))
                elif split == "splitImageTo9":
                    splitted_images.append(splitImageTo9(cutImage))
                else:
                    splitted_images.append([img2b4(cutImage)])

            print('Splitting Images')

            res = dict()
            res['ip'] = external_ip
            res['file'] = splitted_images

            return res
        except Exception as e:
            print(e)
            return f"Error: {e}"
