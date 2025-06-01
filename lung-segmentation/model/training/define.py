from keras_unet_collection import models

def define(name, img_size=128):
    """
    Creates and returns a Keras model based on the specified model name.
    Supports different types of models, including U-Net, Attention U-Net, U-Net++, and R2U-Net.
    
    Parameters:
    - name: A string representing the model name. It can be one of the following:
        - 'unet'     : Standard U-Net model
        - 'attunet'  : Attention U-Net model
        - 'unetpp_wods' : U-Net++ without deep supervision
        - 'unetpp_wds'  : U-Net++ with deep supervision
        - 'r2unet'   : R2U-Net model (with residual connections)

    Returns:
    - model: A Keras model object corresponding to the specified model name.
    """
    
    model = None  # Initialize the model variable to None

    # If the model type is "unet"
    if name == "unet":
        model = models.unet_2d(
            input_size=(img_size, img_size, 1),  # Input size of the image (height, width, channels)
            filter_num=[16, 32, 64, 128, 256],  # Number of filters at each level of the U-Net
            stack_num_down=2,  # Number of downsampling blocks
            stack_num_up=2,  # Number of upsampling blocks
            activation="ReLU",  # Activation function used in the layers
            n_labels=1,  # Number of output labels (1 for binary segmentation)
            output_activation="Sigmoid",  # Output activation function (Sigmoid for binary classification)
            batch_norm=True,  # Whether to use batch normalization
            pool=True,  # Whether to use pooling in the downsampling path
            unpool=False,  # Whether to use unpooling in the upsampling path
            name=name  # Name of the model
        )
    # If the model type is "attunet"
    elif name == "attunet":
        model = models.att_unet_2d(
            input_size=(img_size, img_size, 1),
            filter_num=[16, 32, 64, 128, 256],
            stack_num_down=2,
            stack_num_up=2,
            activation="ReLU",
            attention="add",  # Attention mechanism type ("add" or "multiply")
            atten_activation="ReLU",  # Activation function for attention mechanism
            n_labels=1,
            output_activation="Sigmoid",
            batch_norm=True,
            pool=True,
            unpool=False,
            name=name
        )
    # If the model type is "unetpp_wods" (U-Net++ without deep supervision)
    elif name == "unetpp_wods":
        model = models.unet_plus_2d(
            input_size=(img_size, img_size, 1),
            filter_num=[16, 32, 64, 128, 256],
            stack_num_down=2,
            stack_num_up=2,
            activation="ReLU",
            n_labels=1,
            output_activation="Sigmoid",
            batch_norm=True,
            pool=True,
            unpool=False,
            deep_supervision=False,  # No deep supervision
            name=name
        )
    # If the model type is "unetpp_wds" (U-Net++ with deep supervision)
    elif name == "unetpp_wds":
        model = models.unet_plus_2d(
            input_size=(img_size, img_size, 1),
            filter_num=[16, 32, 64, 128, 256],
            stack_num_down=2,
            stack_num_up=2,
            activation="ReLU",
            n_labels=1,
            output_activation="Sigmoid",
            batch_norm=True,
            pool=True,
            unpool=False,
            deep_supervision=True,  # Enable deep supervision
            name=name
        )
    # If the model type is "r2unet" (R2U-Net with residual connections)
    elif name == "r2unet":
        model = models.r2_unet_2d(
            input_size=(img_size, img_size, 1),
            filter_num=[16, 32, 64, 128, 256],
            stack_num_down=2,
            stack_num_up=2,
            recur_num=2,  # Number of residual blocks
            activation="ReLU",
            n_labels=1,
            output_activation="Sigmoid",
            batch_norm=True,
            pool=True,
            unpool=False,
            name=name
        )

    # Return the model object
    return model