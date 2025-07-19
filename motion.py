import numpy as np
import torchio as tio
import torch
import piq

def sim_motion(img, type='motion_blur', severity=1):
    if np.max(img) > 0:
        img /= (np.max(img) + 1e-8)  # Normalize to [0, 1]
    img = tio.ScalarImage(tensor=img[np.newaxis, :, :, np.newaxis])  # Convert to TorchIO ScalarImage

    # This format allows for the expansion of more types of artifacts
    if type == 'motion_blur':
        return motion_blur(img=img, severity=severity)
    else:
        raise ValueError("Unsupported motion type. Use 'affine' or 'motion_blur'.")

def motion_blur(img, num_ghosts=np.random.randint(3, 6), severity=2):
    """
    Simulates motion by overlaying multiple transformed versions of the input image.
    
    Parameters:
        img: TorchIO image (4D tensor: C, H, W, D)
        num_ghosts: Number of motion copies to overlay
        severity: Controls intensity of motion for each copy
        
    Returns:
        np.ndarray: Ghosted 2D image (H, W)
    """
    base_weight = 0.65  # Base weight for the original image (upon which transformations will be overlaid on)
    
    base = img.data.clone().float()
    blend = base * base_weight
    ghost_weight = (1.0 - base_weight) / num_ghosts # Each ghost is equally weighted
    
    for i in range(1, num_ghosts + 1):
        # Randomizes the translation and rotation
        motion = tio.RandomMotion(
            degrees=(severity * 1.5, severity * 1.5),
            translation=(-i * severity * 1.5, i * severity * 1.5),
            num_transforms=np.random.randint(1, 3),
        )
        # Elastically deforms the image to realistically simulate motion
        elastic = tio.RandomElasticDeformation(
            num_control_points=(4, 4, 4),
            max_displacement=severity * i * 3,
            locked_borders=0,
        )

        transformed = elastic(motion(img))
        blend += transformed.data  # Sum up all the transforms
    
    # Average and squeeze
    corrupted = (blend / (num_ghosts + 1)).squeeze().numpy()
    corrupted /= (np.max(corrupted) + 1e-8)
    return corrupted


def analyze_artifacts(original_img, corrupted_img):
    '''
    Evaluates how distorted the image has become
    
    Parameters: 
        original image - the reconstructed rss image
        corrupted image - the image transformed and overlaid using torchio methods
        
    Returns:
        MSE - Mean Squared Error (Pixel differences)
        PSNR - Peak Signal To Noise Ratio (Ratio between maximum power and noise, a higher value is better)
        SSIM - Structural Similarity Index (Measures human percieved image similarity by accounting for contrast etc.)
    '''
    diff = np.abs(corrupted_img - original_img)
    mse = np.mean(diff ** 2)
    psnr = 20 * np.log10(np.max(original_img) / np.sqrt(mse + 1e-8))  # Add epsilon to avoid division by zero
    
    original_torch = torch.tensor(original_img).unsqueeze(0).unsqueeze(0)
    corrupted_torch = torch.tensor(corrupted_img).unsqueeze(0).unsqueeze(0)
    original_torch = original_torch.clamp(0, 1)
    corrupted_torch = corrupted_torch.clamp(0, 1)
    
    ssim_tensor = piq.ssim(original_torch,corrupted_torch)
    ssim = ssim_tensor.item() # type: ignore

    return {
        'MSE': mse,
        'PSNR': psnr,
        'SSIM': ssim
    }
