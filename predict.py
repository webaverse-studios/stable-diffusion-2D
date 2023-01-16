# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, BaseModel, File, Input, Path
from base import init_model, load_image_generalised, inference
from postprocess import cut, cutv2, splitHeightTo2, splitImageTo9, img2b4
from PIL import Image

import base64

import urllib.request

from typing import Any, List
import torch 

print('cuda status is',torch.cuda.is_available())


#strdwvlly style model for generating assets with black background
# pipe_asset = init_model(local_model_path = "./diffusers_summerstay_strdwvlly_asset_v2")
pipe_inpaint = init_model(local_model_path = "./stable-diffusion-2-inpainting")



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
        input: Path = Input(description="Init Image for Inpaint"),
        mask: Path = Input(description="Mask Image for Inpaint"),
        prompts: str = Input(description="Prompts", default="blue house: fire cathedral   "),
        # strength: float = Input(description="Denoising strength of Stable Diffusion", default=0.85),
        guidance_scale: float = Input(description="Prompt Guidance strength/Classifier Free Generation strength of Stable Diffusion", default=7.5),
        split : str = Input(description="Decide which split needs to happen", default="none"),
        req_type: str = Input(description="Describes whether the request is for an object, currently only \'inpaint\'", default="inpaint"),
        negative_prompt: str = Input(description="Negative_Prompt", default="ugly, blurry"),
        num_inference_steps: int = Input(description="Number of denoising steps", default = 20),
        cut_inner_tol:int = Input(description="Inner tolerance in `cutv2` strongest component PNG masking ", default = 7),
        cut_outer_tol:int = Input(description="Outer tolerance in `cutv2` strongest component PNG masking ", default = 35),
        cut_radius:int = Input(description="Radius in `cutv2` strongest component PNG masking ", default = 70),
        sd_seed:int = Input(description="Seed for SD generations for getting deterministic outputs", default = 1024)
    ) -> Any:
        """Run a single prediction on the model"""
        try:
            global pipe_inpaint

            init_img = load_image_generalised(input)
            mask_image = load_image_generalised(mask)

            images = None
            if req_type == 'inpaint':
                images = inference(pipe_inpaint, init_img, mask_image, \
                            prompts = separate_prompts(prompts), \
                            negative_pmpt = negative_prompt,
                            guidance_scale = guidance_scale,
                            req_type = req_type,
                            num_inference_steps = num_inference_steps,
                            seed = sd_seed)
            
            #else assume it to be a request for tiles
            else:
                raise('predict: only Inpaint supported!')


            external_ip = urllib.request.urlopen('https://ident.me').read().decode('utf8')

            images_ = []

            print('Images are',images)

            if req_type != "tile":
                for gen_image in images:
                    images_.append(cutv2(gen_image, init_img, outer_tolerance = cut_outer_tol, inner_tolerance = cut_inner_tol, radius = cut_radius))
            else:
                for image in images:
                    images_.append(image)

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