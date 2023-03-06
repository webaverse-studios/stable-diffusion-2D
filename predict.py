# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, BaseModel, File, Input, Path
from base import init_model, load_image_generalised, inference, inference_w_gpt, inference_with_edge_guidance, init_canny_controlnet
from postprocess import cut, cutv2, cut_magenta, splitHeightTo2, splitImageTo9, img2b4
from PIL import Image


import base64

import urllib.request

from typing import Any, List
import torch 

print('cuda status is',torch.cuda.is_available())


#strdwvlly style model for generating assets with black background
# pipe_asset = init_model(local_model_path = "./diffusers_summerstay_strdwvlly_asset_v2")
# pipe_asset = init_model(local_model_path = "./stable-diffusion-2-depth")


#Texture model ('smlss style') for generating tiles/textures
# pipe_tile =  init_model(local_model_path = "./diffusers_summerstay_seamless_textures_v1")

pipe_asset_pixel = init_canny_controlnet(local_model_path = "./control_TopdownBalanced_canny.pth")


#Magenta background 2D outdoor asset model trained by summerstay
pipe_asset_magenta = init_model(local_model_path = "./magenta_tree_model")


def separate_prompts(inp_str: str):
  prompts = [x.strip() for x in inp_str.split(':')]
  return prompts



# import unicorn here


class Output(BaseModel):
    file: File
    ip: str


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print('Stable Diffusion started!')
    
    def predict(
        self,
        input: Path = Input(description="Init Image for Img2Img"),
        prompts: str = Input(description="Prompts", default="blue house: fire cathedral   "),
        strength: float = Input(description="Denoising strength of Stable Diffusion", default=0.85),
        guidance_scale: float = Input(description="Prompt Guidance strength/Classifier Free Generation strength of Stable Diffusion", default=7.5),
        split : str = Input(description="Decide which split needs to happen", default="none"),
        req_type: str = Input(description="Describes whether the request is for an object asset or a tile", default="asset"),
        isTree: bool = Input(description="Flag to whether to use Tree model + GPT prompting OR the general Pixel model", default=False),
        negative_prompt: str = Input(description="Negative_Prompt", default="base, ground, terrain, child's drawing, sillhouette, dark, shadowed, green blob, cast shadow on the ground, background pattern"),
        num_inference_steps: int = Input(description="Number of denoising steps", default = 20),
        # cut_inner_tol:int = Input(description="Inner tolerance in `cutv2` strongest component PNG masking ", default = 7),
        outer_tol:int = Input(description="Outer tolerance in `cutv2` strongest component PNG masking ", default = 80),
        # cut_radius:int = Input(description="Radius in `cutv2` strongest component PNG masking ", default = 70),
        sd_seed:int = Input(description="Seed for SD generations for getting deterministic outputs", default = 1024),
        canny_lower:int = Input(description="Canny lower bound for general pixel model with Canny Controlnet", default = 100),
        canny_upper:int = Input(description="Canny upper bound for general pixel model with Canny Controlnet", default = 200),
        width:int = Input(description="Width for returning output image", default = None),
        height:int = Input(description="Height for returning output image", default = None)
    ) -> Any:
        """Run a single prediction on the model"""
        try:
            # global pipe_asset 
            global pipe_tile, pipe_asset_magenta
            
            init_img = load_image_generalised(input, resize = True)

            orig_img_dims = load_image_generalised(input, resize = False).size

            images = None
            if req_type == 'asset':
                if isTree:
                    images = inference_w_gpt(pipe_asset_magenta, init_img, \
                            prompts = separate_prompts(prompts), \
                            negative_pmpt = negative_prompt,
                            strength = strength,
                            guidance_scale = guidance_scale,
                            req_type = req_type,
                            num_inference_steps = num_inference_steps,
                            seed = sd_seed)
                else:
                    images = inference_with_edge_guidance(pipe_asset_pixel, init_img, prompts, negative_prompt , canny_lower, canny_upper, num_inference_steps)

            #else assume it to be a request for tiles
            else:
                images = inference(pipe_tile, init_img, \
                            prompts = separate_prompts(prompts), \
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
                for gen_image in images:
                    # images_.append(cutv2(gen_image, init_img, outer_tolerance = cut_outer_tol, inner_tolerance = cut_inner_tol, radius = cut_radius))
                    images_.append(cut_magenta(gen_image, outer_tol))
            else:
                for image in images:
                    images_.append(image)

            if height is None or width is None:
                height = orig_img_dims[0]
                width = orig_img_dims[1]

            images_ = [img.resize((height,width)) for img in images_]

            splitted_images = []

            for cutImage in images_:
                if split == "splitHeightTo2":
                    splitted_images.append(splitHeightTo2(cutImage))
                elif split == "splitImageTo9":
                    splitted_images.append(splitImageTo9(cutImage))
                else:
                    splitted_images.append([img2b4(cutImage)])

            res = dict()
            res['ip'] = external_ip
            res['file'] = splitted_images

            return res
        except Exception as e:
            return f"Error: {e}"
