import os
import json

def train(model, x_train, y_train, x_val, y_val, batch_size, epochs, checkpoint, folder):
    """
    Trains a deep learning model with the given training and validation data.

    Parameters:
    - model: The Keras model to be trained (e.g., U-Net, U-Net++ with deep supervision).
    - x_train: Training input data (e.g., images).
    - y_train: Training target labels (e.g., masks).
    - x_val: Validation input data.
    - y_val: Validation target labels.
    - batch_size: Number of samples per training batch.
    - epochs: Number of full passes through the training dataset.
    - checkpoint: Keras ModelCheckpoint callback to save weights during training.
    - folder: Path to the folder where model and training history should be saved.
    """

    # If using 'unetpp_wds' model, duplicate labels 5 times for each output layer
    if model.name == "unetpp_wds":
        y_train = [y_train] * 5
        y_val = [y_val] * 5

    # Fit the model and store training history
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        shuffle=False,
        verbose=True,
        callbacks=[checkpoint]
    )

    # Save the trained model in HDF5 format (.h5)
    model.save(os.path.join(folder, "model", model.name + ".h5"))

    # Save training history as a JSON file
    os.makedirs(os.path.join(folder, "history"), exist_ok=True)
    with open(os.path.join(folder, "history", "history.json"), "w") as json_file:
        json.dump(history.history, json_file)