import cv2
import numpy as np
import piq
import torch

def bilateral_filtering(corrupted_image, d=5, sigma_color=50, sigma_space=50):
    '''
    Uses CV2's bilateral filtering method to return a denoised image. 
    This is used as a baseline comparison for the model. 
    '''
    if corrupted_image is None:
        raise ValueError("Image is None. Check input.")

    # Convert to float32
    corrupted_image = corrupted_image.astype(np.float32)

    # Apply bilateral filter
    filtered_image = cv2.bilateralFilter(corrupted_image, d, sigma_color, sigma_space)

    return filtered_image


def evaluate_bilateral_filtering(original_image, filtered_image):
    '''
    Evaluates the performance of bilateral filtering based on MSE, PSNR and SSIM
    
    Parameters: 
        original image - the reconstructed rss image
        filtered image - the image passed through the bilateral filter
        
    Returns:
        MSE - Mean Squared Error (Pixel differences)
        PSNR - Peak Signal To Noise Ratio (Ratio between maximum power and noise, a higher value is better)
        SSIM - Structural Similarity Index (Measures human percieved image similarity by accounting for contrast etc.)
    '''
    # Compute MSE
    mse = np.mean((original_image - filtered_image) ** 2)

    # Avoid division by zero while computing PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(np.max(original_image) / np.sqrt(mse + 1e-8))
        
    # Converting the images to torch tensors so that ssim can be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_torch = torch.tensor(original_image, dtype=torch.float64).unsqueeze(0).unsqueeze(0).to(device).clamp(0,1)
    filtered_torch = torch.tensor(filtered_image, dtype=torch.float64).unsqueeze(0).unsqueeze(0).to(device).clamp(0,1)

    # Ensuring they're on the same device
    if torch.cuda.is_available():
        original_torch = original_torch.cuda()
        filtered_torch = filtered_torch.cuda()

    ssim_tensor = piq.ssim(original_torch, filtered_torch, data_range=1.0)
    ssim = ssim_tensor.cpu().item()

    return {
        'MSE': mse,
        'PSNR': psnr,
        'SSIM': ssim
    }