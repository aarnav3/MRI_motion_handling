import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from model import UNet
import pandas as pd
from motion import analyze_artifacts
from  bilateral_filtering import bilateral_filtering, evaluate_bilateral_filtering
from BM3D import bm3d_denoising, evaluate_bm3d

def sharpen_image(image, alpha=1.5, blur_ksize=(5, 5)):
    """
    Sharpens an image using unsharp masking.

    Args:
        image (np.ndarray): Input image
        alpha (float): Strength of sharpening. 
        blur_ksize (tuple): Kernel size for Gaussian blur.

    Returns:
        np.ndarray: Sharpened image (same dtype/shape).
    """
    # Convert to float32 if not already
    img = image.astype(np.float32)

    # Blur the image
    blurred = cv2.GaussianBlur(img, blur_ksize, 0)

    # Subtract and amplify
    sharpened = cv2.addWeighted(img, 1 + alpha, blurred, -alpha, 0)

    # Clips values either between 0 and 1 or 0 and 255
    sharpened = np.clip(sharpened, 0, 1.0 if image.max() <= 1.0 else 255)

    return sharpened.astype(image.dtype)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)

# Load model
model = UNet().to(device)
model.load_state_dict(torch.load("models/filepath.pth", map_location=device))
model.eval()
print("Model Loaded")

# Directories
clean_dir = "clean_reconstruction"
corrupted_dir = "corrupted_reconstruction"
output_dir = "denoised_outputs"
os.makedirs(output_dir, exist_ok=True)

# Metrics
total_psnr = 0
total_mse = 0
total_ssim = 0
count = 0

# Create visualization directory
viz_dir = "visualizations"
os.makedirs(viz_dir, exist_ok=True)

# Counter for visualization examples
viz_count = 0
max_viz_examples = 10

# To update the csv with metrics
metrics = pd.DataFrame(columns=['filename', 'corrupted_psnr', 'corrupted_mse', 'corrupted_ssim', 'bilateral_psnr', 'bilateral_mse', 'bilateral_ssim', 'bm3d_psnr', 'bm3d_mse', 'bm3d_ssim', 'model_mse', 'model_psnr', 'model_ssim'])

# Load test files list
with open('test_files.txt', 'r') as f:
    test_files = [line.strip() for line in f.readlines() if line.strip()]

# Process all corrupted images
for filename in tqdm(test_files):
    clean_path = os.path.join(clean_dir, filename)
    corrupted_path = os.path.join(corrupted_dir, filename)

    clean_img = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
    corrupted_img = cv2.imread(corrupted_path, cv2.IMREAD_GRAYSCALE)

    if clean_img is None or corrupted_img is None:
        print(f"Skipping {filename} — missing file.")
        continue

    #Normalize values
    clean = clean_img.astype(np.float32) / 255.0
    corrupted = corrupted_img.astype(np.float32) / 255.0
    
    corrupted_analysis = analyze_artifacts(clean, corrupted)

    bilateral_filtered = bilateral_filtering(corrupted_image=corrupted)
    bilateral_filtered = bilateral_filtered.astype(np.float32) / 255.0 

    bm3d_filtered = bm3d_denoising(corrupted_image=corrupted)
    bm3d_filtered = np.abs(bm3d_filtered)
    bm3d_filtered = bm3d_filtered.astype(np.float32) / 255.0  
    
    bilateral_analysis = analyze_artifacts(clean, bilateral_filtered)    
    bm3d_analysis = analyze_artifacts(clean, bm3d_filtered)

    input_tensor = torch.tensor(corrupted).unsqueeze(0).unsqueeze(0).to(device)

    # Running the model
    with torch.no_grad():
        output = model(input_tensor).squeeze().cpu().numpy()
        
    output = sharpen_image(output)
    
    analysis = analyze_artifacts(clean, output)

    mse = analysis["MSE"]
    psnr = analysis["PSNR"]
    ssim = analysis["SSIM"]
    
    metrics.loc[count] = {
                'filename': filename,
                'corrupted_psnr': corrupted_analysis["PSNR"],
                'corrupted_mse': corrupted_analysis["MSE"],
                'corrupted_ssim': corrupted_analysis["SSIM"],
                'bilateral_psnr': bilateral_analysis['PSNR'],
                'bilateral_mse': bilateral_analysis['MSE'],
                'bilateral_ssim': bilateral_analysis['SSIM'],
                'bm3d_psnr': bm3d_analysis['PSNR'],
                'bm3d_mse': bm3d_analysis['MSE'],
                'bm3d_ssim': bm3d_analysis['SSIM'],
                'model_mse': mse,
                'model_psnr': psnr,
                'model_ssim': ssim
            }
    
    total_ssim += ssim
    total_mse += mse
    total_psnr += psnr
    count += 1

    # Save output
    denoised_img = (output * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, filename), denoised_img)
    
    if viz_count < max_viz_examples:
        # Create comparison plot and save directly
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        
        axs[0].imshow(clean_img, cmap='gray')
        axs[0].set_title("Clean Image")
        axs[0].axis('off')
        
        axs[1].imshow(corrupted_img, cmap='gray')
        axs[1].set_title("Corrupted Image")  
        axs[1].axis('off')
        
        axs[2].imshow(output, cmap='gray')
        axs[2].set_title("Denoised Output")
        axs[2].axis('off')
        
        plt.suptitle(f"File: {filename}", fontsize=14)
        plt.tight_layout()
        
        # Save to disk
        viz_filename = f"comparison_{viz_count:02d}_{filename.replace('.png', '')}.png"
        plt.savefig(os.path.join(viz_dir, viz_filename), dpi=150, bbox_inches='tight')
        plt.close()  # Frees up memory
        
        viz_count += 1

metrics.to_csv("results/filename.csv")

# Report metrics
if count > 0:
    print(f"\nAverage PSNR: {total_psnr / count:.2f} dB")
    print(f"Average MSE: {total_mse / count:.6f}")
else:
    print("No images processed.")

print(f"Saved {viz_count} visualization examples to '{viz_dir}' directory")
