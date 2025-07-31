import yaml
import torch
from PIL import Image
import numpy as np
from model_dcgan import Generator

# Load config
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)
gan_config = config["gan"]

device = torch.device('cpu')
z_dim = gan_config.get("z_dim", 100)
weights_file = gan_config.get("weights_file", "generator_full.pth")
output_image = gan_config.get("output_image", "output.png")

# Load weights only (the best way!)
generator = Generator(z_dim=z_dim)
generator.load_state_dict(torch.load(weights_file, map_location=device))
generator.eval()

def generate_image():
    z = torch.randn(1, z_dim, 1, 1, device=device)
    print("Noise sample:", z.flatten()[:5])
    with torch.no_grad():
        fake_img = generator(z)
    fake_img = fake_img.detach().cpu().numpy()
    img = fake_img.squeeze()
    img = np.transpose(img, (1, 2, 0))      # (C, H, W) -> (H, W, C)
    img = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(output_image)
    return output_image
