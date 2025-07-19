import bm3d
import numpy as np
import cv2
import torch
import piq

def bm3d_denoising(corrupted_image, sigma_psd=0.05):
    denoised = bm3d.bm3d(corrupted_image, sigma_psd=sigma_psd)
    return denoised

def evaluate_bm3d(original_image, bm3d_filtered_image):
    '''
    Evaluates the performance of the bm3d method based on MSE, PSNR and SSIM
    
    Parameters: 
        original image - the reconstructed rss image
        bm3d filtered image - the image passed through the bilateral filter
        
    Returns:
        MSE - Mean Squared Error (Pixel differences)
        PSNR - Peak Signal To Noise Ratio (Ratio between maximum power and noise, a higher value is better)
        SSIM - Structural Similarity Index (Measures human percieved image similarity by accounting for contrast etc.)
    '''
    if original_image.shape != bm3d_filtered_image.shape:
        raise ValueError("Original and filtered images must have the same dimensions.")
    
    # Calculate metrics
    mse = np.mean((original_image - bm3d_filtered_image) ** 2)
    psnr = cv2.PSNR(original_image.astype(np.uint8), bm3d_filtered_image.astype(np.uint8))
    
    # Converting the images to torch tensors so that ssim can be used
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_torch = torch.tensor(original_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device).clamp(0.0, 1.0)
    bm3d_torch = torch.tensor(bm3d_filtered_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device).clamp(0.0, 1.0)

    
    # Ensures the device is the same
    if torch.cuda.is_available():
        original_torch = original_torch.cuda()
        bm3d_torch = bm3d_torch.cuda()
        
    ssim_tensor = piq.ssim(original_torch,bm3d_torch)
    ssim = ssim_tensor.cpu().item()

    return {
        'MSE': mse,
        'PSNR': psnr,
        'SSIM': ssim
    }