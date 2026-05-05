import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_metrics(original, processed):
    original = np.array(original).astype(np.float32)
    processed = np.array(processed).astype(np.float32)

    mse = np.mean((original - processed) ** 2)

    if mse == 0:
        psnr = 100
    else:
        psnr = 10 * np.log10((255 ** 2) / mse)

    ssim_val = ssim(original, processed, channel_axis=2, data_range=255)

    return mse, psnr, ssim_val