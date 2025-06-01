import os
from tf_keras.callbacks import ModelCheckpoint

def checkpoint(folder):
    # Create a ModelCheckpoint callback to save model weights at the end of each epoch
    cp = ModelCheckpoint(
        os.path.join(folder, "checkpoint/epoch_{epoch:02d}.weights.h5"),  # Save the model weights with a filename format including the epoch number
        save_weights_only=True,  # Save only the model weights, not the entire model
        save_freq="epoch",  # Save the weights at the end of each epoch
        verbose=1  # Print messages to the console when saving the model weights
    )

    # Return the created ModelCheckpoint callback
    return cp