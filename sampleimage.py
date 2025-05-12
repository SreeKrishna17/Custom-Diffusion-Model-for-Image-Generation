import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# fake_images: Tensor of shape (60, 3, 64, 64)
# Simulate 60 random images for demo
fake_images = torch.randn(60, 3, 64, 64)

# Create grid and save
grid = vutils.make_grid(fake_images[:36], nrow=6, normalize=True, scale_each=True)
plt.figure(figsize=(8, 8))
plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
plt.axis("off")
plt.title("Sample Outputs from Model")
plt.tight_layout()
plt.savefig("generated_unet_no_attention.png")
plt.show()
