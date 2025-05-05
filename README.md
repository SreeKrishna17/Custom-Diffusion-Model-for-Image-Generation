# Diffusion-Based Image Generation App

A custom-built image generation system using denoising diffusion probabilistic models (DDPMs), developed entirely from scratch. This project allows users to interactively generate high-resolution floral images using either a baseline or an attention-enhanced U-Net via a Streamlit dashboard.

## What Is Happening?

This application trains a model to reverse the process of adding Gaussian noise to images — essentially learning how to start from noise and create realistic outputs. Inspired by DDPMs like DALL·E 2 and Stable Diffusion (but from scratch!), the model slowly denoises starting from pure noise.

## Project Structure

```
diffusion_project/
├── main_interface.py         # Streamlit UI for model selection and image generation
├── training_pipeline.py      # Training loop using DDPM loss on Oxford Flowers dataset
├── diffusion_core.py         # Core DDPM logic: beta schedule, forward process, and loss
├── baseline_unet.py          # KaushikUnet: simple U-Net architecture without attention
├── advanced_unet.py          # Enhanced U-Net with attention and residual blocks
├── sampling_loop.py          # Reverse diffusion sampling for inference
├── image_utils.py            # Image preprocessing and visualization utilities
├── setup_env.py              # Device configuration and core imports
├── model_400pt               # Pretrained model checkpoint (with attention)
├── new_linear_model_1090.pt  # Pretrained model checkpoint (baseline)
├── requirements.txt          # Python dependencies for training and inference
├── README.md                 # Project overview and usage guide
└── Notebooks/                # Jupyter notebooks for development and experimentation
├── autoencoder_on_a_huggingface_dataset.ipynb                         # Early experiment with autoencoders on Hugging Face datasets
├── Unconditional_Image_Generation_Using_Diffusion_Models.ipynb        # Initial diffusion model implementation and output samples
└── Unconditional_Image_Generation_using_Diffusion_Models_v2_0.ipynb   # Final refined version with annotated steps, visuals, and model comparisons
```

## How to Run

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Launch the Streamlit app
```
streamlit run main_interface.py
```

## Streamlit Interface Overview

When you launch the app:
- Choose between:
  - Model without attention
  - Model with attention 
- Click “Generate image” to sample 60 images from the trained model.
- View images in grid format, with each row representing samples at a time step.

## Notebooks

The `Notebooks/` folder contains:
- Visual explanation of the forward and reverse diffusion processes
- Side-by-side comparisons between noisy and denoised images
- Charts showing loss curves and training stability over epochs

Use these for documenting your experimentation or showcasing model interpretability.

## Key Components Explained

| File | Purpose |
|------|---------|
| diffusion_core.py | Implements beta scheduling, noise addition, and loss |
| baseline_unet.py | Defines a basic encoder-decoder U-Net (KaushikUnet) |
| advanced_unet.py | U-Net with attention + residual connections |
| training_pipeline.py | Loads Oxford Flowers dataset and trains the DDPM |
| sampling_loop.py | Reconstructs images from noise during inference |
| main_interface.py | UI to toggle between models and visualize outputs |
| image_utils.py | Preprocessing steps and image display helpers |
