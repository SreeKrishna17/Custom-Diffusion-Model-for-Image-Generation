# Diffusion-Based Image Generation App

A custom-built image generation system using denoising diffusion probabilistic models (DDPMs), developed entirely from scratch. This project allows users to interactively generate high-resolution floral images using either a baseline or an attention-enhanced U-Net via a Streamlit dashboard.

## What Is Happening?

This application trains a model to reverse the process of adding Gaussian noise to images — essentially learning how to start from noise and create realistic outputs. Inspired by DDPMs like DALL·E 2 and Stable Diffusion (but from scratch!), the model slowly denoises starting from pure noise.

## Project Structure

```
diffusion_project/
├── main_interface.py         # Streamlit UI to run inference
├── training_pipeline.py      # Training loop for DDPM
├── diffusion_core.py         # Core logic for noise scheduling and diffusion
├── baseline_unet.py          # KaushikUnet: Simple baseline U-Net architecture
├── advanced_unet.py          # Improved U-Net with attention modules
├── sampling_loop.py          # Inference pipeline for image generation
├── image_utils.py            # Preprocessing, transforms, and image visualization
├── setup_env.py              # Central import + device setup
├── model_400pt               # Trained model with attention
├── new_linear_model_1090.pt  # Trained baseline model
├── requirements.txt          # Dependencies
├── README.md                 # You're reading it
└── Notebooks/                # Jupyter development logs and visuals
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
