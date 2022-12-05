# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, BaseModel, File, Input, Path
from base import init_model, load_image_generalised, inference
from postprocess import cut, splitHeightTo2, splitImageTo9, img2b4
from PIL import Image

import base64

import urllib.request

from typing import Any, List
import torch 

print('cuda status is',torch.cuda.is_available())


pipe = init_model(local_model_path = "./stable-diffusion-v1-5")


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
        split : str = Input(description="Decide which split needs to happen", default="none")
        # negative_prompt: str = Input(description="Negative_Prompt", default="isometric, terrain, interior, ground, island, farm, at night, dark, ground, monochrome, glowing, text, character, sky, UI, pixelated, blurry, tiled squares") 
    ) -> Any:
        """Run a single prediction on the model"""
        try:
            global pipe

            init_img = load_image_generalised(input)

            images = inference(pipe, init_img, \
                            prompts = separate_prompts(prompts), \
                            # negative_prompt= separate_prompts(negative_prompt),
                            strength = strength,
                            guidance_scale = guidance_scale)



            


            external_ip = urllib.request.urlopen('https://ident.me').read().decode('utf8')


            images_ = []

            print('Images are',images)

            for image in images:
                images_.append(cut(image))


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