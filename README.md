# MRI Motion Artifact Handling
This system trains a light UNet model capable of running on CUDA-enabled laptops to clean up MRI images with motion artifacts. It first simulates the motion on 320x320 png images of MRI reconstructions from the NYU Langone fastMRI dataset using TorchIO, and uses the reconstructions as ground truth for the model. Images were used instead of k-space because they're less computationally intensive.

This was a project conducted to familiarize myself with image processing and deep learning methods.

## Details
Dataset size: ~2,200 images <br/>
Dataset split: 80:10:10  training:val:testing <br/>
Approx training time: 2 hours

## Evaluation
I compared my model to two standard denoising techniques for MRIs: bilateral filtering and bm3d using MSE (mean squared error), PSNR (peak signal to noise ratio) and SSIM (structural similarity index). The loss function was a weighted average of MSE and SSIM.

| Metric / Artifact Handling Method | Bilateral Filtering | BM3D      | My Model  |
|-----------------------------------|---------------------|-----------|-----------|
| MSE                               | 0.057312            | 0.05235   | 0.005776  |
| PSNR                              | 17.986300           | 13.074181 | 23.089175 |
| SSIM                              | 0.035334            | 0.035316  | 0.694047  |

Although, with an SSIM of ~0.7, my model probably doesn't output images that are clinically viable, given the computational limitations, it's promising and performs a lot better than the algorithmic alternatives.

## Examples

<img width="2549" height="868" alt="image" src="https://github.com/user-attachments/assets/4bef3444-65e8-4ed2-8b84-030ce60d1c59" />
<img width="2539" height="880" alt="image" src="https://github.com/user-attachments/assets/5dd4c752-b025-4316-b15f-b2472f76f9c6" />
<img width="2547" height="872" alt="image" src="https://github.com/user-attachments/assets/7a9a8b87-5bb9-4080-9cb9-b5b20e681899" />



## References
1. Data: https://fastmri.med.nyu.edu/ (the knee singlecoil val dataset was used because it has reconstructed images in addition to just the raw k-space data.)

Data used in the preparation of this article were obtained from the NYU fastMRI Initiative database ([fastmri.med.nyu.edu](https://fastmri.med.nyu.edu)).[citation of Knoll et al Radiol Artif Intell. 2020 Jan 29;2(1):e190007. 
doi: 10.1148/ryai.2020190007. ([https://pubs.rsna.org/doi/10.1148/ryai.2020190007](https://pubs.rsna.org/doi/10.1148/ryai.2020190007)), and the arXiv paper: [https://arxiv.org/abs/1811.08839](https://arxiv.org/abs/1811.08839)] As such, NYU fastMRI investigators provided data but did not participate in analysis or writing of this report. A listing of NYU fastMRI investigators, subject to updates, can be found at:[fastmri.med.nyu.edu](https://fastmri.med.nyu.edu).The primary goal of fastMRI is to test whether machine learning can aid in the reconstruction of medical images.”

2. Module used to simulate motion: https://docs.torchio.org/index.html

[F. Pérez-García, R. Sparks, and S. Ourselin. TorchIO: a Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning. Computer Methods and Programs in Biomedicine (June 2021), p. 106236. ISSN: 0169-2607.doi:10.1016/j.cmpb.2021.106236.](https://doi.org/10.1016/j.cmpb.2021.106236)

3. https://quentin-chappat.netlify.app/project/dl-mri-motion-correction/

4. bm3d: https://pypi.org/project/bm3d/#description
