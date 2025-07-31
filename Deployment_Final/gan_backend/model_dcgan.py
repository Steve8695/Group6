import torch
from torch import nn

# Generator model class (needed for both training and inference)
class Generator(nn.Module):
    def __init__(self, z_dim=100, channels=3, features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.ConvTranspose2d(features, channels, 4, 4, 3, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# Discriminator (not needed for Flask inference, but fine to keep)
class Discriminator(nn.Module):
    def __init__(self, channels=3, features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 2, 1, 3, 1, 1, bias=False),
            nn.Sigmoid(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)

# ----------------------------------------------------------
# Only run the training/data/plotting code if this file is
# executed as a script, NOT when it's imported by gan_model.py
# ----------------------------------------------------------

if __name__ == "__main__":
    # You CANNOT use Jupyter magics (!unzip etc.) in Python files.
    # Unzip your data manually or via Python's zipfile module, e.g.:
    # import zipfile; zipfile.ZipFile("/content/sample_data/processed.zip").extractall("processed")

    import os
    import time
    import json
    import torchvision
    from torchvision import transforms, datasets
    from torch.utils.data import DataLoader, Subset
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    from datetime import timedelta
    from tqdm import tqdm
    import warnings
    warnings.filterwarnings("ignore")

    image_size = 128
    batch_size = 128
    sample_size = 16000
    epochs = 10

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Load dataset from local directory
    full_dataset = datasets.ImageFolder(root="/content/processed/processed/train", transform=transform)

    # Filter only the 4 classes
    classes_to_keep = ['food', 'drink', 'inside', 'outside']
    indices_to_keep = [i for i, (_, label) in enumerate(full_dataset.samples)
                    if full_dataset.classes[label] in classes_to_keep]

    filtered_dataset = Subset(full_dataset, indices_to_keep)
    indices_sampled = torch.randperm(len(filtered_dataset))[:sample_size]
    dataset = Subset(filtered_dataset, indices_sampled)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)

    print("Classes used:", classes_to_keep)
    print(f"Using {len(dataset)} images.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    z_dim = 100
    lr = 2e-4

    G = Generator(z_dim=z_dim).to(device)
    D = Discriminator().to(device)

    criterion = nn.BCELoss()
    optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(25, z_dim, 1, 1, device=device)
    overall_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        for real, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            real = real.to(device)
            batch_size = real.size(0)

            noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
            fake = G(noise)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Train D
            D_real = D(real)
            D_fake = D(fake.detach())
            d_loss = criterion(D_real, real_labels) + criterion(D_fake, fake_labels)
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            # Train G
            fake = G(noise)
            g_loss = criterion(D(fake), real_labels)
            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

        # Logging
        epoch_secs = time.time() - epoch_start
        total_secs = time.time() - overall_start
        avg_epoch = total_secs / (epoch + 1)
        remaining = avg_epoch * (epochs - epoch - 1)

        print(f"Epoch {epoch+1:02d}: D_loss={d_loss.item():.4f}  G_loss={g_loss.item():.4f}  "
            f"epoch {timedelta(seconds=int(epoch_secs))}  ETA {timedelta(seconds=int(remaining))}")

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                fake_images = G(fixed_noise).detach().cpu()
            grid = make_grid(fake_images, nrow=5, normalize=True)
            plt.figure(figsize=(6, 6))
            plt.imshow(grid.permute(1, 2, 0))
            plt.axis("off")
            plt.title(f"Generated Images @ Epoch {epoch+1}")
            os.makedirs("generated", exist_ok=True)
            plt.savefig(f"generated/DCGAN_epoch_{epoch+1}.png")
            plt.show()

    print("Training complete.")

    torch.save(G.state_dict(), 'generator_full.pth')
    import pickle
    with open('generator_full.pk', 'wb') as f:
        pickle.dump(G, f)
    print("Models saved successfully!")
