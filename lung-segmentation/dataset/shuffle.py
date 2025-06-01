import numpy as np
import random
from dataset import random_seed

def shuffle(img, mask, img_size=128):
    """
    Shuffles the images and their corresponding masks, and resizes them to the given image size.

    Parameters:
    - img: Numpy array of images (e.g., shape [num_samples, height, width, channels]).
    - mask: Numpy array of masks corresponding to the images.
    - img_size: The size to which the images and masks should be resized (default is 128).

    Returns:
    - img: Reshuffled and resized images.
    - mask: Reshuffled and resized masks.
    """
    
    # Set a fixed random seed for reproducibility
    random_seed(42)

    # Create an array of indices representing the dataset
    idxs = np.arange(np.shape(img)[0])
    
    # Shuffle the indices randomly
    random.shuffle(idxs)
    
    # Initialize empty arrays to store shuffled images and masks
    img_shuffle = np.zeros(np.shape(img))
    mask_shuffle = np.zeros(np.shape(mask))
    
    # Reorder the images and masks according to shuffled indices
    for idx in idxs:
        img_shuffle[idx] = img[idx]    # Reorder images
        mask_shuffle[idx] = mask[idx]  # Reorder masks
    
    # Reshape the images to the specified size and normalize to [0, 1]
    img = img_shuffle.reshape(-1, img_size, img_size, 1) / 255.0

    # Reshape the masks, threshold them to binary (>= 0.5), and convert to float32
    mask = (mask_shuffle.reshape(-1, img_size, img_size, 1) / 255.0 >= 0.5).astype(np.float32)
    
    # Return the shuffled and processed images and masks
    return img, mask