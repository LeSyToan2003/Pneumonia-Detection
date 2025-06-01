import tensorflow as tf
import tf_keras.backend as K

def dsc(y_true, y_pred, smooth=K.epsilon()):
    """
    Calculates the Dice Similarity Coefficient (DSC) for binary segmentation tasks.
    The DSC is a measure of overlap between two binary masks, where a higher value
    indicates a greater similarity between the true and predicted masks.

    Parameters:
    - y_true: Tensor of ground truth labels (True segmentation mask).
    - y_pred: Tensor of predicted labels (Predicted segmentation mask).
    - smooth: Small value to prevent division by zero (default is `K.epsilon()` which is a small value in Keras).

    Returns:
    - dsc: The Dice Similarity Coefficient.
    """
    
    # Convert predictions to binary (0 or 1) based on a threshold of 0.5
    y_pred_bin = K.cast(y_pred >= 0.5, tf.float32)
    
    # Calculate the intersection (common positive pixels between the ground truth and prediction)
    intersection = K.sum(K.abs(y_true * y_pred_bin), axis=[1])
    
    # Calculate the union (total number of positive pixels in the ground truth and prediction)
    union = K.sum(y_true, axis=[1]) + K.sum(y_pred_bin, axis=[1])
    
    # Compute the Dice Similarity Coefficient (DSC), add smoothness to avoid division by zero
    result = K.mean((2.0 * intersection + smooth) / (union + smooth), axis=0)
    
    # Return the Dice score
    return result