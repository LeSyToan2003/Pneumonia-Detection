import os
import numpy as np

def load_dataset(convex_hull, files, folder="/kaggle/input/lung-segmentation-dataset-ch0"):
            
    """
    Loads the dataset (images and masks) for lung segmentation.

    Parameters:
    - convex_hull: The subdirectory where the dataset with or without convex hull is stored ("without_convexhull" or "with_convexhull").
    - files: List of file paths relative to the `folder` for images and masks.
    - folder: The root directory containing the dataset (default is the Kaggle input path).

    Returns:
    - img: Numpy array containing the images.
    - mask: Numpy array containing the masks.
    """
    
    
    # Load the images and masks from .npy files
    img = np.load(os.path.join(folder, convex_hull, files[0]))  # Load images
    mask = np.load(os.path.join(folder, convex_hull, files[1]))  # Load masks
    
    # Return the loaded images and masks
    return img, mask