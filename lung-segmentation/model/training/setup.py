def setup(model, optimizer, loss, metrics):
    """
    Compiles a Keras model with the given optimizer, loss function, and metrics.
    
    Parameters:
    - model: The Keras model to be compiled (e.g., U-Net, ResNet).
    - lr: The learning rate for the optimizer.
    - loss: The loss function to be used for training the model.
    - metric: The metric(s) to evaluate the model's performance during training (e.g., accuracy, Dice coefficient).
    
    Returns:
    - model: The compiled Keras model.
    """
    
    # Check if the model is "unetpp_wds" (U-Net++ with deep supervision)
    if model.name == "unetpp_wds":
        # Compile the model with deep supervision (multiple output layers)
        model.compile(
            optimizer=optimizer,  # Adam optimizer with a given learning rate
            loss=loss,  # Loss function for training
            metrics={
                # Specify the metric for each of the output layers in U-Net++
                "unetpp_wds_output_sup0_activation": metrics,
                "unetpp_wds_output_sup1_activation": metrics,
                "unetpp_wds_output_sup2_activation": metrics,
                "unetpp_wds_output_sup3_activation": metrics,
                "unetpp_wds_output_final_activation": metrics
            }
        )
    else:
        # Compile the model for other types without deep supervision
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Return the compiled model
    return model