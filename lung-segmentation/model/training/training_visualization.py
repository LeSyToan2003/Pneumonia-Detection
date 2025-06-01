import matplotlib.pyplot as plt

def progress(title, x_label, y_label, data, name, figsize=(5, 5)):
    """
    Visualizes the training process by plotting various metrics over epochs.
    
    Parameters:
    - title: The title of the plot (e.g., "Training Loss and Accuracy").
    - x_label: The label for the x-axis (typically "Epochs").
    - y_label: The label for the y-axis (e.g., "Loss" or "Accuracy").
    - data: A list of metrics data (e.g., loss, accuracy values) for plotting.
    - name: The names corresponding to each data line (e.g., ["Training Loss", "Validation Loss"]).
    - figsize: The size of the figure for the plot (default is (5, 5)).
    """
    
    # Create a figure with the specified size (default is 5x5)
    plt.figure(figsize=figsize)
    
    # Set the title for the plot and specify font size
    plt.title(title, fontsize=15)
    
    # Loop through the data and plot each series
    for i in data:
        plt.plot(i, linewidth=2)  # Plot each data series with a line width of 2
        
    # Add a legend to the plot, specifying the location and font size
    plt.legend(name, loc="center right", fontsize=12)
    
    # Label the x-axis and y-axis, with specified font sizes
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    
    # Set the font size for the x and y ticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Display the plot
    plt.show()