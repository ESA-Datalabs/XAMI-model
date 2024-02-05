import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ImageNormalize, ZScaleInterval
import numpy as np

def data_norm(data, n_samples=1000, contrast=0.1, max_reject=0.5, min_npixels=10, krej=2.5, max_iterations=5):
        interval = ZScaleInterval(n_samples, contrast, max_reject, min_npixels, krej, max_iterations)
        vmin, vmax = interval.get_limits(data)
        norm = ImageNormalize(vmin=vmin, vmax=vmax)
        return norm

def show_scaled_image(input_path, output_folder):
    hdul = fits.open(input_path)
    image_data = hdul[0].data
    hdul.close()

    # Apply zscale normalization only on non-negative data
    norm = data_norm(image_data[image_data>0])
    normalized_data = norm(image_data)
    
    # Convert masked array to normal NumPy array
    normalized_data = normalized_data.filled(fill_value=-1)

    os.makedirs(output_folder, exist_ok=True)

    # from matplotlib to cv2
    flipped_data = np.flipud(normalized_data)

    output_path = os.path.join(output_folder, os.path.basename(input_path).replace('.fits', '.png'))

    # clip to 255, because some pixels are very big
    scaled_data = np.clip((flipped_data * 255), 0, 255).astype(np.uint8)
    scaled_data = 255 - scaled_data
    
    cv2.imwrite(output_path, scaled_data)

    print(f"FITS file saved as PNG with zscale normalization: {output_path}")
    return output_path

