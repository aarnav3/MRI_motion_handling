import h5py
import numpy as np
import matplotlib.pyplot as plt
from motion import sim_motion, analyze_artifacts
from bilateral_filtering import bilateral_filtering, evaluate_bilateral_filtering
from BM3D import bm3d_denoising, evaluate_bm3d
import os
import pandas as pd

data = pd.DataFrame(columns=['filename', 'slice', 'corrupted_psnr', 'corrupted_mse', 'corrupted_ssim', 'bilateral_psnr', 'bilateral_mse', 'bilateral_ssim', 'bm3d_psnr', 'bm3d_mse', 'bm3d_ssim'])

# The main loop that goes through each of the h5 files and returns the reconstructed images of a few MRI slices to build the dataset
for file in os.listdir('singlecoil_val'):
    if file.endswith('.h5'):
        print(f"Processing {file}")
    
    # Load the reconstructed_rss data
    with h5py.File(f'singlecoil_val/{file}', 'r') as f:
        reconstruction = f['reconstruction_rss'][:]  # type: ignore - shape: (slices, height, width)
        slice_indices = [5,8,19,12,14,15,16,18,20,22,25] # a few distinct slices to create the dataset (there are 30 slices)
         
        for i in slice_indices:
            # Extract and normalize slices
            slice_reconstruction = reconstruction[i] #type: ignore
            img = np.abs(slice_reconstruction) #type: ignore
            img /= (np.max(img) + 1e-8) # Normalize

            filename = f"{file}_slice_{i}.png"
            
            # Save the clean reconstruction
            plt.imsave(f"clean_reconstruction/{filename}", img, cmap='gray')

            # Create and save the corrupted image
            corrupted = sim_motion(img, 'motion_blur', severity=2)
            plt.imsave(f"corrupted_reconstruction/{filename}", corrupted, cmap='gray')

            corrupted_analysis = analyze_artifacts(img, corrupted)

            bilateral_filtered = bilateral_filtering(corrupted_image = corrupted)
            bilateral_filtered = np.abs(bilateral_filtered)
            bilateral_filtered /= (np.max(bilateral_filtered) + 1e-8)
            
            bilateral_analysis = evaluate_bilateral_filtering(img, bilateral_filtered)
    
            bm3d_filtered = bm3d_denoising(corrupted_image=corrupted)
            bm3d_filtered = np.abs(bm3d_filtered)
            bm3d_filtered /= (np.max(bm3d_filtered) + 1e-8)
            
            bm3d_analysis = evaluate_bm3d(img, bm3d_filtered)
            
            # Save the data
            data.loc[len(data)] = {
                'filename': filename,
                'slice': i,
                'corrupted_psnr': corrupted_analysis["PSNR"],
                'corrupted_mse': corrupted_analysis["MSE"],
                'corrupted_ssim': corrupted_analysis["SSIM"],
                'bilateral_psnr': bilateral_analysis['PSNR'],
                'bilateral_mse': bilateral_analysis['MSE'],
                'bilateral_ssim': bilateral_analysis['SSIM'],
                'bm3d_psnr': bm3d_analysis['PSNR'],
                'bm3d_mse': bm3d_analysis['MSE'],
                'bm3d_ssim': bm3d_analysis['SSIM']
            }

# Export data to CSV
data.to_csv('image_generation_results.csv', index=False)