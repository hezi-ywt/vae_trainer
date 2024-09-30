import os

import torch
import diffusers
from diffusers.models import AutoencoderKL
from safetensors.torch import load_file, save_file

import logging
logger = logging.getLogger(__name__)


from model import load_vae


vae = load_vae("/home/ywt/lab/stable-diffusion-webui/models/VAE/sdxl.vae.safetensors", "float16")

image_pth = "/home/ywt/lab/sd-scripts/library/vae_trainer/test/6290268.webp"

vae(image_pth)