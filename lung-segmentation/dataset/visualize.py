import matplotlib.pyplot as plt

def visualize(img, title, figsize=(2, 2)):
    """
    Visualizes a given image with a specified title.

    Parameters:
    - img: The image to display (can be a 2D or 3D array, typically an image or a batch of images).
    - title: The title to be displayed on the image.
    - figsize: The size of the figure (default is (5, 5)).

    Returns:
    - None
    """
    
    # Create a figure to display the image
    plt.figure(figsize=figsize)
    plt.title(title)  # Set the title for the image
    plt.imshow(img, cmap="gray")  # Show the image in grayscale
    plt.show()  # Display the image