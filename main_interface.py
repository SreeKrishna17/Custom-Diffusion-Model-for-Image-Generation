from setup_env import *
from image_utils import *
from baseline_unet import *
from advanced_unet import *
from sampling_loop import *
import os
import zipfile

st.markdown("""
## Diffusion-Based Image Generation

This app demonstrates how a diffusion model learns to generate images from **pure random noise** by reversing a gradual corruption process. The result: realistic-looking floral images generated entirely from scratch.

**What this does:**
- Lets you pick between two models: a baseline U-Net and an attention-based U-Net
- Generates 60 unique images from noise using the selected model
- Lets you download all generated images as a ZIP
""")

st.title("Generating Images Using a Diffusion Model")

option = st.radio('Choose a model:', ('Model without Attention', 'Model with Attention'))
if option == 'Model without Attention':
    model = KaushikUnet()
    model.load_state_dict(torch.load("new_linear_model_1090.pt", map_location=torch.device('cpu')))
    st.write("Model without attention is initialized.")

elif option == 'Model with Attention':
    model = Unet(
        dim=img_size,
        channels=3,
        dim_mults=(1, 2, 4,)
    )
    model.load_state_dict(torch.load("model_400pt", map_location=torch.device('cpu')))
    st.write("Model with Attention is initialized.")

model.to(device)

if st.button("Click to generate image from pure noise"):
    samples = generate_images(model, image_size=img_size, batch_size=64, channels=3)
    num_columns = 5
    os.makedirs("generated_output", exist_ok=True)

    for i in range(0, 60, num_columns):
        cols = st.columns(num_columns)
        for col, img_idx in zip(cols, range(i, i + num_columns)):
            reverse_transforms = transforms.Compose([
                transforms.Lambda(lambda t: (t + 1) / 2),
                transforms.Lambda(lambda t: t.permute(1, 2, 0)),
                transforms.Lambda(lambda t: t * 255.),
                transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
                transforms.ToPILImage(),
            ])
            img = reverse_transforms(torch.Tensor((samples[-1][img_idx].reshape(3, img_size, img_size))))
            img.save(f"generated_output/image_{img_idx+1:02d}.png")
            col.image(img, width=150)

    # Create zip for download
    zip_filename = "generated_images.zip"
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        for idx in range(60):
            zipf.write(f"generated_output/image_{idx+1:02d}.png")

    with open(zip_filename, "rb") as f:
        st.download_button("Download All Images", f, file_name=zip_filename, mime="application/zip")
