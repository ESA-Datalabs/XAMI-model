import matplotlib.pyplot as plt
from astropy.convolution import convolve
from astropy.visualization import AsinhStretch, LogStretch, ZScaleInterval, simple_norm
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.background import Background2D, SExtractorBackground, MedianBackground, MMMBackground, ModeEstimatorBackground
from astropy.stats import biweight_location, mad_std, sigma_clipped_stats, SigmaClip
from photutils.segmentation import detect_threshold,  deblend_sources, detect_sources, make_2dgaussian_kernel, SourceFinder, SourceCatalog
from photutils.utils import circular_footprint
import numpy as np
import os
import cv2
from astropy.io import fits


med = r"$\tilde{x}$"
biw_loc = r"$\zeta_{\text{biloc}}$"

# usually the pip install "opencv-python-headless<4.3" solves this problem:
# AttributeError: partially initialized module 'cv2' has no attribute '_registerMatType' (most likely due to a circular import)
# !pip install "opencv-python-headless<4.3"
# !pip install jupyter-bbox-widget

def data_norm(data, n_samples=1000, contrast=0.1, max_reject=0.5, min_npixels=10, krej=2.5, max_iterations=5):
    interval = ZScaleInterval(n_samples, contrast, max_reject, min_npixels, krej, max_iterations)
    vmin, vmax = interval.get_limits(data)
    norm = ImageNormalize(vmin=vmin, vmax=vmax)
    return norm

def image_stretch(data, stretch='log', factor=1):
	"""
	Apply a stretch to an image data array while masking negative input values.

	Parameters:
	- data: numpy.ndarray
		The input image data array. The image is expected to be normalised.
	- stretch: str, optional (default: 'log')
		The type of stretch to apply. Options are 'log' and 'asinh'.
	- factor: float, optional (default: 1)
		The stretching factor to apply. 

	Returns:
	- numpy.ndarray
		The stretched image data array.
	"""

	positive_mask = data > 0
	data_log_stretched = data.copy()

	if stretch == 'asinh':
		stretch = AsinhStretch(a=factor)
	elif stretch == 'log':
		stretch = LogStretch(a=factor)
	else:
		raise ValueError('Invalid stretch option')
	
	data_log_stretched[positive_mask] = stretch(data[positive_mask])

	return data_log_stretched

def rescale_flattened_image(image: np.ndarray, 
                            negative_mask: np.ndarray, 
                            target_mean: float, 
                            target_std: float, 
                            epsilon: float = 1e-8) -> np.ndarray:
    """
    Rescales the pixel values of an image to a specified mean and standard deviation, 
    excluding pixels marked by a negative mask.

    This function flattens the image and the negative mask, filters out the pixels 
    marked by the negative mask, and rescales the values to a target mean and std. 
    Finally, it reconstructs the image from the rescaled flattened pixel values.

    Parameters:
    - image (numpy.ndarray): The original image as a 2D or 3D array.
    - negative_mask (numpy.ndarray): A boolean mask array where True indicates 
      pixels to be excluded from rescaling.
    - target_mean (float): The target mean value for rescaling.
    - target_std (float): The target standard deviation for rescaling.
    - epsilon (float, optional): A small value added to standard deviation to 
      prevent division by zero. Default is 1e-8.

    Returns:
    - final_image (numpy.ndarray): The image after rescaling, with the same shape 
      as the input image.
    """
    
    flat_image = image.flatten()
    flat_mask = negative_mask.flatten()

    non_negative_pixels = flat_image>0

    # Filter out the negative mask pixels
    filtered_pixels = flat_image[non_negative_pixels]

    mean = np.mean(filtered_pixels)
    std = np.std(filtered_pixels)

    rescaled_pixels = (filtered_pixels - mean) / (std + epsilon) * target_std + target_mean

    flat_image[non_negative_pixels] = rescaled_pixels
    final_image = flat_image.reshape(image.shape)

    return final_image

def zscale_image(input_path, output_folder, with_image_stretch=False):
	hdul = fits.open(input_path)
	image_data = hdul[0].data
	hdul.close()
	
	# Apply zscale normalization only on non-negative data
	norm = data_norm(image_data[image_data>0])
	normalized_data = norm(image_data)
	normalized_data = normalized_data.filled(fill_value=-1)
	
	os.makedirs(output_folder, exist_ok=True)
	output_path = os.path.join(output_folder, os.path.basename(input_path).replace('.fits', '.png'))
	
	flipped_data = np.flipud(normalized_data)
	
	# clip to 255
	scaled_data = np.clip((flipped_data * 255), 0, 255).astype(np.uint8)
	
	if with_image_stretch:
		data_log_stretched = image_stretch(flipped_data, stretch='log', factor=500.0)
		cv2.imwrite(output_path, np.clip((data_log_stretched * 255), 0, 255).astype(np.uint8))
		print(f"FITS file saved as PNG with zscale normalization and Log stretch: {output_path}")
	else:
		cv2.imwrite(output_path, scaled_data)
		print(f"FITS file saved as PNG with zscale normalization: {output_path}") 

	return output_path

def detect_and_deblend_sources(data_orig, hw_threshold=0, clip_sigma=3.0, kernel_sigma=3.0, npixels=10, verbose=True):
	"""
	Deblends the input data using background subtraction, convolution, and source detection.

	Args:
		data (numpy.ndarray): Input data array.

	Returns:
		segment_map (numpy.ndarray): Segmentation map of the deblended segments.
		segment_map_finder (numpy.ndarray): Segmentation map of the sources on the segmented image.
		relevant_sources_tbl (astropy.table.Table): Table containing relevant sources (also based on hw threshold) information.
	"""
    
	try: 
		# Initialize SigmaClip with desired parameters
		sigma_clip = SigmaClip(sigma=clip_sigma, 
						#  sigma_lower=clip_sigma, 
						#  sigma_upper= clip_sigma+0.5, 
						 maxiters=10)
		
		# this is used to mask the regions with no data
		coverage_mask = (data_orig == 0)

		threshold = np.percentile(data_orig, 90)  
		bright_source_mask = data_orig > threshold

		# doc here: https://photutils.readthedocs.io/en/stable/api/photutils.background.SExtractorBackground.html
		bkg_estimator = ModeEstimatorBackground()
		bkg = Background2D(data_orig, 
					 box_size=(10, 10), 
					 filter_size=(5,5), 
					 sigma_clip=sigma_clip, 
					#  mask=bright_source_mask, 
					 coverage_mask = coverage_mask,
					 bkg_estimator=bkg_estimator)
		data_orig = data_orig * (~coverage_mask) # mask the regions with no data
		data = data_orig - bkg.background  # subtract the background
		# data = data * (~coverage_mask) # mask the regions with no data
		if verbose:
			print(f"Background: {bkg.background_median}\nBackground RMS: {bkg.background_rms_median}")

		threshold = 1.2 * bkg.background_rms # type: ignore # n-sigma threshold
		kernel = make_2dgaussian_kernel(kernel_sigma, size=5) # enhance the visibility of significant features while reducing the impact 
															  # of random noise or small irrelevant details
		convolved_data = convolve(data, kernel)
		convolved_data = convolved_data * (~coverage_mask) # mask the regions with no data
		# npixels = 10  # minimum number of connected pixels, each greater than threshold, that an object must have to be detected
		segment_map = detect_sources(convolved_data, threshold, npixels=npixels)
		segm_deblend = deblend_sources(convolved_data, segment_map, npixels=npixels, progress_bar=False)

		# Calculate source density
		num_sources = len(np.unique(segment_map)) - 1  # Subtract 1 for the background
		image_area = data.shape[0] * data.shape[1] 
		source_density = num_sources / image_area

		if verbose:
			print(f"Number of sources: {num_sources}\nImage area: {image_area}\nSource density: {source_density}")
			fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 4))
			ax1.imshow(data_orig, cmap='gray', origin='lower', norm=data_norm(data_orig))
			ax1.set_title('Original Data')

			ax2.imshow(data, cmap='gray', origin='lower', norm=data_norm(data))
			ax2.set_title('Background-subtracted Data')
			cmap1 = segment_map.cmap
			ax3.imshow(segment_map.data, cmap=cmap1, interpolation='nearest')
			ax3.set_title('Original Segment')
			cmap2 = segm_deblend.cmap
			ax4.imshow(segm_deblend.data, cmap=cmap2, interpolation='nearest')
			ax4.set_title('Deblended Segments')
			plt.tight_layout()
			plt.savefig('./plots/segmentation_1.png')

		finder = SourceFinder(npixels=10, progress_bar=False)
		segment_map_finder = finder(convolved_data, threshold)
		
		cat = SourceCatalog(data, segm_deblend, convolved_data=convolved_data)
		tbl = cat.to_table()

		tbl['xcentroid'].info.format = '.2f' # type: ignore
		tbl['ycentroid'].info.format = '.2f' # type: ignore
		tbl['kron_flux'].info.format = '.2f' # type: ignore
		tbl['kron_fluxerr'].info.format = '.2f' # type: ignore
		tbl['area'].info.format = '.2f' # type: ignore
		# print(tbl['bbox_xmax'].value - tbl['bbox_xmin'].value)
		# print(tbl['bbox_ymax'].value - tbl['bbox_ymin'].value)
		relevant_sources_tbl = tbl[(abs(tbl['bbox_xmax'].value - tbl['bbox_xmin'].value) > hw_threshold) &  # type: ignore
							(abs(tbl['bbox_ymax'].value - tbl['bbox_ymin'].value) > hw_threshold)] # type: ignore
		# print('flux:', relevant_sources_tbl['kron_flux'].value)

		if verbose:
			fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 12.5))
			ax1.imshow(data, cmap='Greys_r')
			ax1.set_title('Sources')
			ax2.imshow(segment_map_finder, cmap=segm_deblend.cmap, interpolation='nearest')
			ax2.set_title('Deblended Sources')
			cat.plot_kron_apertures(ax=ax1, color='white', lw=1.5)
			cat.plot_kron_apertures(ax=ax2, color='white', lw=1.5)
			plt.tight_layout()
			plt.show()
			plt.close()
	except Exception as e:
		print(e)
		return None, None, None	
	return segment_map, segment_map_finder, relevant_sources_tbl

def mask_with_sigma_clipping(data_2D, sigma=3.0, maxiters=10, sigma_threshold=4.5, footprint_radius=10):
	'''
	prints statistics of the image data, excluding the masked regions.
	'''
	sigma_clip = SigmaClip(sigma=sigma, maxiters=10)
	
	threshold = detect_threshold(data_2D, nsigma=sigma_threshold, sigma_clip=sigma_clip)
	segment_img = detect_sources(data_2D, threshold, npixels=10)
	footprint = circular_footprint(radius=footprint_radius)
	
	# masking can be used to isolate or ignore these sources.
	mask = segment_img.make_source_mask(footprint=footprint)
	# calculates mean, median, and standard deviation of data array, excluding the masked source regions.
	mean, median, std = sigma_clipped_stats(data_2D, sigma=3.0, mask=mask)
	print((mean, median, std))

	plt.subplot(1, 2, 1)
	plt.imshow(data_2D) 
	med = r"$\tilde{x}$"
	biw_loc = r"$\zeta_{\text{biloc}}$"
	plt.title(f'{med}= {np.median(data_2D).round(3)}, {biw_loc} = {biweight_location(data_2D).round(3)}\nmad  = {mad_std(data_2D).round(3)}')

	plt.subplot(1, 2, 2) 
	plt.imshow(mask) 
	plt.title(f'Masked region\n')

	plt.show()
	plt.close()
	return mask, mean, median, std

def get_normalized_centers(data_2D, hw_threshold=30, clip_sigma=2.5, kernel_sigma=1.5, npixels=10):
	segment_map, segment_map_finder, sources_tbl_with_hw_threshold = detect_and_deblend_sources(data_2D, hw_threshold=hw_threshold, clip_sigma=clip_sigma, 
																		  kernel_sigma=kernel_sigma, npixels=npixels, verbose=False)
    
	if segment_map and segment_map_finder and sources_tbl_with_hw_threshold:
		centers_x = [source['xcentroid'] for source in sources_tbl_with_hw_threshold]
		centers_y = [source['ycentroid'] for source in sources_tbl_with_hw_threshold]
		
		def normalize_centers(centers_x, centers_y, image_width, image_height):
				normalized_centers = []
				for x, y in zip(centers_x, centers_y):
						x_normalized = x / (image_width - 1)
						y_normalized = y / (image_height - 1)
						normalized_centers.append((x_normalized, y_normalized))
				return normalized_centers
    		
		normalized_centers = normalize_centers(centers_x, centers_y, data_2D.shape[0], data_2D.shape[1])
		return normalized_centers
	else:
		return None
		
def clahe_algo_image(IMAGE_PATH, clipLimit=3.0, tileGridSize=(8,8)):

    image = cv2.imread(IMAGE_PATH.replace(".fits", ".png"))
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    positive_mask = image_bw > 0
    final_img = image_bw.copy()
    
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    clahe_img = clahe.apply(image_bw[positive_mask])
    final_img[positive_mask] = clahe_img.flatten()
    _, ordinary_img = cv2.threshold(image_bw, 155, 255, cv2.THRESH_BINARY)
    
    plt.figure(figsize=(30, 10))
    plt.subplot(1, 4, 1)
    plt.imshow(image_bw, cmap='viridis')
    plt.title(f'Original Image {IMAGE_PATH.split("/")[-1]}')
    
    plt.subplot(1, 4, 2)
    plt.imshow(ordinary_img, cmap='viridis')
    plt.title('Ordinary Threshold')
    
    plt.subplot(1, 4, 3)
    plt.imshow(final_img, cmap='viridis')
    plt.title('CLAHE Image')
    
    plt.subplot(1, 4, 4)
    plt.hist(final_img[final_img>0], bins=100)
    plt.title('CLAHE Image histogram (nonnegative pixels)')
    plt.show()
    plt.close()

    return final_img

def enhance_contrast_with_adaptive_thresholding(IMAGE_PATH, clipLimit=3.0, tileGridSize=(8,8)):
    image = cv2.imread(IMAGE_PATH)
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding to identify faint regions
    # cv2.ADAPTIVE_THRESH_MEAN_C or cv2.ADAPTIVE_THRESH_GAUSSIAN_C can be used
    adaptive_thresh = cv2.adaptiveThreshold(image_bw, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply CLAHE to the original grayscale image based on the adaptive threshold mask
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    clahe_img = clahe.apply(image_bw)
    
    # Masking: Use the adaptive threshold as a mask to blend the original and CLAHE images
    final_img = np.where(adaptive_thresh == 255, clahe_img, image_bw)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(image_bw, cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(1, 3, 2)
    plt.imshow(adaptive_thresh, cmap='gray')
    plt.title('Adaptive Threshold Mask')
    
    plt.subplot(1, 3, 3)
    plt.imshow(final_img, cmap='gray')
    plt.title('Enhanced Image')
    
    plt.show()

    return final_img

def clahe_algo_image_improved(IMAGE_PATH, clipLimit=1.0, tileGridSize=(4,4), bright_threshold=200):
    image = cv2.imread(IMAGE_PATH.replace(".fits", ".png"))
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Mask for bright regions to avoid enhancing these
    bright_mask = image_bw > bright_threshold
    faint_mask = ~bright_mask

    final_img = image_bw.copy()
    
    # Apply CLAHE only to faint regions
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    faint_img = clahe.apply(image_bw[faint_mask])
    final_img[faint_mask] = faint_img.flatten()

    # Apply a simple threshold to the original image for comparison
    _, ordinary_img = cv2.threshold(image_bw, 155, 255, cv2.THRESH_BINARY)
    
    plt.figure(figsize=(30, 10))
    plt.subplot(1, 4, 1)
    plt.imshow(image_bw, cmap='viridis')
    plt.title(f'Original Image {IMAGE_PATH.split("/")[-1]}')
    
    plt.subplot(1, 4, 2)
    plt.imshow(ordinary_img, cmap='viridis')
    plt.title('Ordinary Threshold')
    
    plt.subplot(1, 4, 3)
    plt.imshow(final_img, cmap='viridis')
    plt.title('CLAHE Image')
    
    plt.subplot(1, 4, 4)
    plt.hist(final_img[final_img>0], bins=100)
    plt.title('CLAHE Image histogram (nonnegative pixels)')
    plt.show()
    plt.close()

    return final_img

def enhance_astronomical_image(image_path, clipLimit=3.0, tileGridSize=(16,16)):
    with fits.open(image_path) as hdul:
        image_data = hdul[0].data
    
    # Estimate the background
    mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
    image_data_subtracted = image_data - median  # Subtract the median as background
    
    # Convert image data to 8-bit for CLAHE using OpenCV, scaling data between 0 and 255
    norm = simple_norm(image_data_subtracted, 'linear', percent=99.5)
    image_scaled = np.clip(norm(image_data_subtracted) * 255, 0, 255).astype('uint8')
    
    # Apply CLAHE to the background-subtracted image
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    clahe_img = clahe.apply(image_scaled)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image_data, cmap='gray', origin='lower', norm=simple_norm(image_data, 'sqrt', percent=99))
    plt.title('Original Image')
    
    plt.subplot(1, 3, 2)
    plt.imshow(image_scaled, cmap='gray', origin='lower')
    plt.title('Background Subtracted')
    
    plt.subplot(1, 3, 3)
    plt.imshow(clahe_img, cmap='gray', origin='lower')
    plt.title('CLAHE Enhanced')
    
    plt.show()
    
from astropy.stats import sigma_clipped_stats
from astropy.io import fits
from astropy.visualization import simple_norm
import matplotlib.pyplot as plt
import numpy as np
import cv2

def selective_clahe_astronomical_image(image_path, clipLimit=3.0, tileGridSize=(1,1)):
    with fits.open(image_path) as hdul:
        image_data = hdul[0].data
    
    # Estimate the background
    mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
    background = median
    image_data_subtracted = image_data - background
    
    # Normalize and scale the subtracted image for CLAHE
    norm = simple_norm(image_data_subtracted, 'linear', percent=99.5)
    image_scaled = np.clip(norm(image_data_subtracted) * 255, 0, 255).astype('uint8')

    # Create a mask for foreground (exclude background)
    _, mask = cv2.threshold(image_scaled, 1, 255, 20) # cv2.THRESH_BINARY is 0

    # Initialize the final image that will combine original background with enhanced foreground
    final_img = image_scaled.copy()

    # Apply CLAHE only to the foreground areas identified by the mask
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    clahe_img_whole = clahe.apply(image_scaled)
    
    # Initialize the final image array
    final_img = np.copy(image_scaled)
    
    # Use the mask to update only the foreground in the final image
    final_img[mask > 0] = clahe_img_whole[mask > 0]

    plt.figure(figsize=(20, 7))
    plt.subplot(1, 4, 1)
    plt.imshow(image_data, cmap='gray', origin='lower', norm=simple_norm(image_data, 'sqrt', percent=99))
    plt.title('Original Image')
    
    plt.subplot(1, 4, 2)
    plt.imshow(image_scaled, cmap='gray', origin='lower')
    plt.title('Background Subtracted')
    
    plt.subplot(1, 4, 3)
    plt.imshow(mask, cmap='gray', origin='lower')
    plt.title('Foreground Mask')
    
    plt.subplot(1, 4, 4)
    plt.imshow(final_img, cmap='gray', origin='lower')
    plt.title('Selective CLAHE Enhanced')
    plt.savefig('./plots/selective_clahe.png')
    
    plt.show()

def darken_background(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Estimate the background using a blur (simple approach)
    background = cv2.GaussianBlur(image, (51, 51), 0)
    
    # Subtract the background, setting a floor at 0 to avoid negative values
    foreground = cv2.subtract(image, background)
    
    # Darken the original background
    # Here, we simply reduce the intensity of the estimated background.
    # Adjust the factor to control the darkness. Values < 1 will darken the background.
    darkened_background = (background * 0.5).astype(np.uint8)
    
    # Combine the darkened background with the original foreground
    final_image = cv2.add(darkened_background, foreground)
    cv2.imwrite('./plots/darkened_background.png', 255-final_image)
    
    # Visualization
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(1, 3, 2)
    plt.imshow(background, cmap='gray')
    plt.title('Estimated Background')
    
    plt.subplot(1, 3, 3)
    plt.imshow(final_image, cmap='gray')
    plt.title('Image with Darkened Background')
    
    plt.show()