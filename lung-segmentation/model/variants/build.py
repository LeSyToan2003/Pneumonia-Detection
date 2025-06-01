import pipeline

def build(model_name):
    """
    Executes the training pipeline for two dataset variations:
    - One without convex hull preprocessing.
    - One with convex hull preprocessing.
    
    Both models are trained and saved under separate subfolders.

    Parameters:
    - name: Root directory where all results (models, checkpoints, history) will be saved.
    """
    
    # Run pipeline on original segmentation masks
    pipeline(model_name, data_folder="without_convexhull", model_folder="segment")

    # Run pipeline on masks with convex hull preprocessing
    pipeline(model_name, data_folder="with_convexhull", model_folder="segment_convexhull")