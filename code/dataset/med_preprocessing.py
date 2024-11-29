## CT: 
import numpy as np

def apply_windowing(ct_image, window_width, window_level):
    # Calculate the lower and upper bounds for the window
    lower_bound = window_level - (window_width / 2)
    upper_bound = window_level + (window_width / 2)
    
    # Apply windowing by clipping and scaling
    windowed_image = np.clip(ct_image, lower_bound, upper_bound)
    windowed_image = ((windowed_image - lower_bound) / (upper_bound - lower_bound)) * 255.0
    return windowed_image.astype(np.uint8)

# Example usage
# Apply two sets of windowing on the same image
windowed_image_1 = apply_windowing(ct_image, 1500, -160)
windowed_image_2 = apply_windowing(ct_image, 80, 40)

## MRI: 
import numpy as np

def preprocess_mri_image(mri_image):
    # Calculate the 0.5th and 99.5th percentiles
    lower_bound = np.percentile(mri_image, 0.5)
    upper_bound = np.percentile(mri_image, 99.5)
    
    # Clip the image to this range
    clipped_image = np.clip(mri_image, lower_bound, upper_bound)
    
    # Rescale to [0, 255]
    rescaled_image = ((clipped_image - lower_bound) / (upper_bound - lower_bound)) * 255.0
    return rescaled_image.astype(np.uint8)

# Example usage
# Process the MRI image
processed_image = preprocess_mri_image(mri_image)