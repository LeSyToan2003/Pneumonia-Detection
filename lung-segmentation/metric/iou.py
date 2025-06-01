import tensorflow as tf
import tf_keras.backend as K

def iou(y_true, y_pred, smooth=K.epsilon()):
    """
    Calculates the Intersection over Union (IoU) metric for binary segmentation tasks.
    The IoU is the ratio of the intersection area between the true and predicted masks
    to the union area of the true and predicted masks.

    Parameters:
    - y_true: Tensor of ground truth labels (True segmentation mask).
    - y_pred: Tensor of predicted labels (Predicted segmentation mask).
    - smooth: Small value to prevent division by zero (default is `K.epsilon()` which is a small value in Keras).

    Returns:
    - iou: The Intersection over Union metric.
    """
    
    # Convert predictions to binary (0 or 1) based on a threshold of 0.5
    y_pred_bin = K.cast((y_pred >= 0.5), tf.float32)
    
    # Calculate the intersection (common positive pixels between the ground truth and prediction)
    intersection = K.sum(K.abs(y_true * y_pred_bin), axis=[1])
    
    # Calculate the union (total positive pixels in both ground truth and prediction, excluding intersection)
    union = K.sum(y_true, axis=[1]) + K.sum(y_pred_bin, axis=[1]) - intersection
    
    # Compute the Intersection over Union (IoU), add smoothness to avoid division by zero
    result = K.mean((intersection + smooth) / (union + smooth), axis=0)
    
    # Return the IoU score
    return result