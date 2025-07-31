import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from peft import PeftModel, PeftConfig
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load base SD pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float32,
    safety_checker=None
).to(device)

# Freeze parts of the model
pipe.unet.requires_grad_(False)
pipe.vae.requires_grad_(False)

# Load LoRA-adapted text encoder
peft_config = PeftConfig.from_pretrained("./checkpoints/text_encoder")
base_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder")
pipe.text_encoder = PeftModel.from_pretrained(base_encoder, "./checkpoints/text_encoder").to(device)

# Replace scheduler
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

# CORRECT tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

def generate_image(prompt, output_path):
    with torch.no_grad():
        pipe.scheduler.set_timesteps(50)

        # Tokenize prompt
        tokens = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)

        # Encode prompt
        prompt_embeds = pipe.text_encoder.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state

        # Latent noise init
        latents = torch.randn((1, pipe.unet.in_channels, pipe.unet.sample_size, pipe.unet.sample_size)).to(device)
        latents = latents * pipe.scheduler.init_noise_sigma

        # Denoising
        for t in pipe.scheduler.timesteps:
            latent_model_input = pipe.scheduler.scale_model_input(latents, t)
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample
            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Decode image
        image = pipe.vae.decode(latents / 0.18215).sample[0]
        image = image.clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        Image.fromarray((image * 255).astype(np.uint8)).save(output_path)
