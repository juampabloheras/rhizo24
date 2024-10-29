import os
import numpy as np
import glob
import subprocess
import argparse
import json
import matplotlib.pyplot as plt


def timing_decorator(func):
    """
    Usage:
    @timing_decorator
    def func():
        <function body here>
    """
    import time

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(
            f"Function '{func.__name__}' took {elapsed_time:.4f} seconds to complete."
        )
        return result

    return wrapper


class MapLabelValueTransform:
    """
    Maps label values to other values in Numpy array.
    Example usage:
        >> labels_transform = MapLabelValueTransform([0, 85, 170], [0, 1, 0])
        >> transformed_label = labels_transform(label) # Maps 0, 85, 170 valued labels to 0, 1, 0, respectively.
    """

    def __init__(self, source_values, target_values):
        self.source_values = source_values
        self.target_values = target_values

    @staticmethod
    def map_label_values(label, source_values, target_values):
        output_label = np.copy(label)
        for src, tgt in zip(source_values, target_values):
            output_label[label == src] = tgt
        return output_label

    def __call__(self, label):
        return self.map_label_values(label, self.source_values, self.target_values)


def nnUNet_name_conversion(data_dir="data", output_dir="train/nnUNet_raw"):
    """
    Function saves files to the format required by nnUNet, removes labels for rectangular box, and saves all images as .tif
    """

    import numpy as np
    import os
    import glob
    from PIL import Image
    import json

    os.makedirs(output_dir, exist_ok=True)
    image_save_path = os.path.join(output_dir, "imagesTr")
    label_save_path = os.path.join(output_dir, "labelsTr")

    images_paths = sorted(glob.glob(os.path.join(data_dir, "images", "*.tif*")))
    labels_paths = sorted(glob.glob(os.path.join(data_dir, "labels", "*.png*")))

    transformed_images_paths = image_save_path
    transformed_labels_paths = label_save_path
    os.makedirs(transformed_images_paths, exist_ok=True)
    os.makedirs(transformed_labels_paths, exist_ok=True)

    # Define the MONAI transforms
    image_transform = None

    labels_transform = MapLabelValueTransform(
        [0, 85, 170], [0, 1, 0]
    )  # Maps 0, 85, 170 valued labels to 0, 1, 0, respectively.

    names_dict = {}
    for subject_no, path in enumerate(images_paths):
        image_path = images_paths[subject_no]
        label_path = labels_paths[subject_no]

        image_name = image_path.split("/")[-1].split(".")[0]
        label_name = label_path.split("/")[-1].split(".")[0]

        assert image_name in label_name

        image = np.array(Image.open(image_path).convert("RGB"))
        label = np.array(Image.open(label_path))

        transformed_image = Image.fromarray(
            image_transform(image) if image_transform else image
        )
        transformed_label = Image.fromarray(
            labels_transform(label).astype(np.uint8) if labels_transform else label
        )

        names_dict[f"{subject_no:03d}"] = image_name

        # Save the image as a TIFF file
        image_save_name = f"rhizo_{subject_no:03d}_0000.tif"
        labels_save_name = f"rhizo_{subject_no:03d}.tif"
        transformed_image.save(os.path.join(transformed_images_paths, image_save_name))
        transformed_label.save(os.path.join(transformed_labels_paths, labels_save_name))

    file_path = os.path.join(os.getcwd(), f"rhizonet_nnUNet_name_conversion.json")
    with open(file_path, "w") as json_file:
        json.dump(names_dict, json_file, indent=4)


def process_and_visualize_subjects(
    image_save_path,
    label_save_path,
    preds_save_path,
    name_conversion_path,
    results_dir,
    apply_mask=False,
):
    import os
    import glob
    import json
    from tqdm import tqdm
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        jaccard_score,
        f1_score,
        recall_score,
    )
    from matplotlib.colors import ListedColormap

    # Load name conversion dictionary
    with open(name_conversion_path, "r") as file:
        name_dict = json.load(file)

    # Get the number of subjects
    images_paths = sorted(glob.glob(os.path.join(image_save_path, "*.tif*")))
    labels_paths = sorted(glob.glob(os.path.join(label_save_path, "*.tif*")))
    preds_paths = sorted(glob.glob(os.path.join(preds_save_path, "*.tif*")))
    num_subjects = len(images_paths)

    # Process each subject
    for subject_no in tqdm(range(num_subjects)):
        image = np.array(Image.open(images_paths[subject_no]))
        label = np.array(Image.open(labels_paths[subject_no]))
        pred = np.array(Image.open(preds_paths[subject_no]))

        # Filter label and prediction to only include relevant class
        filtered_label = 1
        label = np.where((label == filtered_label), label, 0)
        pred = np.where((pred == filtered_label), pred, 0)

        if apply_mask:
            # Calculate metrics only where label is non-zero
            nonzero_mask = label != 0
            label_flat = label[nonzero_mask].flatten()
            pred_flat = pred[nonzero_mask].flatten()
        else:
            # Calculate metrics over the entire label and prediction arrays
            label_flat = label.flatten()
            pred_flat = pred.flatten()

        # Calculate metrics
        accuracy = accuracy_score(label_flat, pred_flat)
        precision = precision_score(label_flat, pred_flat)
        iou = jaccard_score(label_flat, pred_flat)
        dice = f1_score(label_flat, pred_flat)
        recall = recall_score(label_flat, pred_flat)

        # Visualize images, labels, predictions, and differences
        label = label.astype(np.int16)
        pred = pred.astype(np.int16)
        diff = pred - label

        fig, axes = plt.subplots(1, 4, figsize=(15, 5))
        axes[0].imshow(image)
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(label, cmap="gray")
        axes[1].set_title("Label")
        axes[1].axis("off")

        axes[2].imshow(pred, cmap="gray")
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        cmap = ListedColormap(["red", "black", "white"])
        cax3 = axes[3].imshow(diff, cmap=cmap, vmin=-1, vmax=1)
        axes[3].set_title("Difference (Prediction - Label)")
        axes[3].axis("off")
        cbar3 = fig.colorbar(cax3, ax=axes[3], orientation="vertical")
        cbar3.set_ticks([-1, 0, 1])
        cbar3.set_ticklabels(["-1", "0", "1"])

        plt.tight_layout()

        # Save the visualization and metrics
        subject_name = preds_paths[subject_no].split("_")[-1].split(".")[0]
        rhizonet_name = name_dict.get(subject_name, f"subject_{subject_no}")
        subject_save_dir = os.path.join(results_dir, rhizonet_name)
        os.makedirs(subject_save_dir, exist_ok=True)

        # Save the visualization image
        plt.savefig(os.path.join(subject_save_dir, "results.png"))
        plt.close()

        # Save metrics as JSON
        metrics = {
            "image_path": images_paths[subject_no],
            "preds_path": preds_paths[subject_no],
            "label_path": labels_paths[subject_no],
            "accuracy": accuracy,
            "precision": precision,
            "IoU": iou,
            "Dice": dice,
            "recall": recall,
        }
        with open(os.path.join(subject_save_dir, "metrics.json"), "w") as file:
            json.dump(metrics, file, indent=4)


def vis(plots_dir="plts/", renamed_data_dir="renamed_rhizonet_images/"):
    """
    Function to visualize renamed data. Useful for debugging.
    """
    from PIL import Image
    import matplotlib.pyplot as plt

    for subject_no in [5, 10, 15, 20]:
        image_save_path = os.path.join(renamed_data_dir, "imagesTr")
        label_save_path = os.path.join(renamed_data_dir, "labelsTr")

        images_paths = sorted(
            glob.glob(os.path.join(image_save_path, "*.tif*"))
        )  # All data must be tifs
        labels_paths = sorted(
            glob.glob(os.path.join(label_save_path, "*.tif*"))
        )  # All data must be tifs

        print(f"Image path: {images_paths[subject_no]}")
        print(f"Label path: {labels_paths[subject_no]}")

        image = np.array(Image.open(images_paths[subject_no]))
        label = np.array(Image.open(labels_paths[subject_no]))

        print(f"Image shape: {np.shape(image)}")
        print(f"Label shape: {np.shape(label)}")

        print(f"Label unique: {np.unique(label)}")

        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(image)
        axes[0].set_title("Image")
        axes[0].axis("off")

        cax = axes[1].imshow(label, cmap="gray")
        axes[1].set_title("Label")
        axes[1].axis("off")

        # Add a colorbar to the second subplot
        fig.colorbar(cax, ax=axes[1], orientation="vertical")

        plt.tight_layout()
        plt.show()
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, f"subject_{subject_no}.png"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a specified function with optional arguments."
    )
    parser.add_argument("function_name", type=str, help="The function to run")
    parser.add_argument(
        "additional_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments for the function, passed as strings to the function",
    )
    return parser.parse_args()


if __name__ == "__main__":
    """
    Example usage of a function in this script:
        python3 train.py train arg1 arg2       <-- For function "train" and arguments arg1, arg2
        python3 train.py eval arg1 arg2 arg3   <-- For function "eval" and arguments arg1, arg2, arg3
        python3 train.py sample_fn      <-- For function "sample_fn" with no arguments
    NOTE: all args are passed as strings to the function called.
    """

    args = parse_args()

    # Validate function_name argument
    function_name = args.function_name

    if function_name in globals() and callable(globals()[function_name]):
        # Call the function by name, pass additional arguments as a list
        print(
            f"(@utils.py) Running function: {function_name} with arguments: {args.additional_args}"
        )
        globals()[function_name](*args.additional_args)
    else:
        print(f"Function '{function_name}' not found.")
        print(
            f"Available functions: {[fn for fn in globals() if callable(globals()[fn])]}"
        )
