import os
from tf_keras.optimizers import Adam
from dataset import load, shuffle, visualize
from metric import dsc, iou
from model import define, setup, checkpoint, train

def pipeline(model_name, data_folder, model_folder):
    """
    Runs the full training pipeline:
    - Prepares directories for saving model, checkpoints, and history.
    - Loads and preprocesses training and validation data.
    - Visualizes sample image and mask.
    - Initializes and compiles the model.
    - Trains the model and saves results.

    Parameters:
    - name: The main folder name under which everything will be saved.
    - data_folder: Subfolder specifying which data version to use.
    - model_folder: Subfolder to store model, checkpoints, and training history.
    """

    # Prepare folder structure for model saving
    model_folder = os.path.join(model_name, model_folder)
    os.makedirs(os.path.join(model_folder, "model"), exist_ok=True)
    os.makedirs(os.path.join(model_folder, "checkpoint"), exist_ok=True)
    os.makedirs(os.path.join(model_folder, "history"), exist_ok=True)

    # Load training and validation datasets
    train_img, train_mask, val_img, val_mask = load(convex_hull=data_folder)

    # Shuffle the data (image-mask pairs) for training and validation
    (x_train, y_train), (x_val, y_val) = shuffle(train_img, train_mask), shuffle(val_img, val_mask)

    # Visualize one example of image and its corresponding mask
    visualize(x_train[0], "Image")
    visualize(y_train[0], "Mask")

    # Initialize the model
    model = define(model_name)

    # Display model architecture
    model.summary()

    # Compile the model with learning rate, loss function, and metrics
    setup(model, optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metric=[dsc, iou])

    # Train the model and save outputs
    train(model, x_train, y_train, x_val, y_val, batch_size=64, epochs=50, checkpoint=checkpoint(model_folder), folder=model_folder)