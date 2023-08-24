# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, BaseModel, File, Input, Path
from base import init_model, make_background_magenta, load_image_generalised, inference, inference_w_gpt, inference_with_edge_guidance, init_canny_controlnet
from postprocess import cut, cutv2, cut_magenta, remBgPil, splitHeightTo2, splitImageTo9, img2b4
from PIL import Image
from run import run_predict

import os

import urllib.request

from typing import Any, List
import torch 




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
        split : str = Input(description="Decide which split needs to happen", default= None),
        req_type: str = Input(description="Describes whether the request is for an object asset or a tile", default="asset"),
        negative_prompt: str = Input(description="Negative_Prompt", default="base, ground, terrain, child's drawing, sillhouette, dark, shadowed, green blob, cast shadow on the ground, background pattern"),
        num_inference_steps: int = Input(description="Number of denoising steps", default = 20),
        sd_seed:int = Input(description="Seed for SD generations for getting deterministic outputs", default = 1000),
        width:int = Input(description="Width for returning output image", default = 512),
        height:int = Input(description="Height for returning output image", default = 512),
        model: str = Input(description="Model to use based on the type of object you want to create. Options are tree, building, or furniture.", default = "furniture")
    ) -> Any:
        return run_predict(input, prompts, strength, guidance_scale, split, req_type, negative_prompt, num_inference_steps, sd_seed, width, height, model)
