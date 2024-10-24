import os
import numpy as np
import glob
import subprocess
import argparse


def timing_decorator(func):
    '''
    Usage:
    @timing_decorator
    def func():
        <function body here>
    '''
    import time
    def wrapper(*args, **kwargs):
        start_time = time.time() 
        result = func(*args, **kwargs) 
        end_time = time.time() 
        elapsed_time = end_time - start_time  
        print(f"Function '{func.__name__}' took {elapsed_time:.4f} seconds to complete.")
        return result
    return wrapper

class MapLabelValueTransform:
    '''
    Maps label values to other values in Numpy array. 
    Example usage: 
        >> labels_transform = MapLabelValueTransform([0, 85, 170], [0, 1, 0])
        >> transformed_label = labels_transform(label) # Maps 0, 85, 170 valued labels to 0, 1, 0, respectively.
    '''
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

def nnUNet_name_conversion(data_dir='data', output_dir='train/nnUNet_raw'):
    '''
    Function saves files to the format required by nnUNet, removes labels for rectangular box, and saves all images as .tif
    '''

    import numpy as np
    import os
    import glob
    from PIL import Image
    import json

    os.makedirs(output_dir, exist_ok=True)
    image_save_path = os.path.join(output_dir,'imagesTr')
    label_save_path = os.path.join(output_dir,'labelsTr')

    images_paths = sorted(glob.glob(os.path.join(data_dir, 'images',"*.tif*")))
    labels_paths = sorted(glob.glob(os.path.join(data_dir, 'labels',"*.png*")))

    transformed_images_paths = image_save_path
    transformed_labels_paths = label_save_path
    os.makedirs(transformed_images_paths, exist_ok=True)
    os.makedirs(transformed_labels_paths, exist_ok=True)


    # Define the MONAI transforms
    image_transform = None

    labels_transform = MapLabelValueTransform([0, 85, 170], [0, 1, 0]) # Maps 0, 85, 170 valued labels to 0, 1, 0, respectively.

    names_dict = {}
    for subject_no, path in enumerate(images_paths):
        image_path = images_paths[subject_no]
        label_path = labels_paths[subject_no]

        image_name = image_path.split('/')[-1].split('.')[0]
        label_name = label_path.split('/')[-1].split('.')[0]

        assert image_name in label_name
        
        image = np.array(Image.open(image_path).convert("RGB"))
        label = np.array(Image.open(label_path))

        transformed_image = Image.fromarray(image_transform(image) if image_transform else image)
        transformed_label = Image.fromarray(labels_transform(label).astype(np.uint8) if labels_transform else label)

        names_dict[f'{subject_no:03d}'] = image_name

        # Save the image as a TIFF file
        image_save_name = f'rhizo_{subject_no:03d}_0000.tif'
        labels_save_name = f'rhizo_{subject_no:03d}.tif'
        transformed_image.save(os.path.join(transformed_images_paths, image_save_name))
        transformed_label.save(os.path.join(transformed_labels_paths, labels_save_name))

    file_path = os.path.join(os.getcwd(), f'rhizonet_nnUNet_name_conversion.json')
    with open(file_path, 'w') as json_file:
        json.dump(names_dict, json_file, indent=4) 

def vis(plots_dir = 'plts/', renamed_data_dir = '/pscratch/sd/j/jehr/rhizo24/rhizonet_images'):
    '''
    Function to visualize renamed data. Useful for debugging.
    '''
    from PIL import Image
    import matplotlib.pyplot as plt

    for subject_no in [5, 10, 15, 20]:
        image_save_path = os.path.join(renamed_data_dir,'imagesTr')
        label_save_path = os.path.join(renamed_data_dir,'labelsTr')

        images_paths = sorted(glob.glob(os.path.join(image_save_path,"*.tif*"))) # All data must be tifs
        labels_paths = sorted(glob.glob(os.path.join(label_save_path,"*.tif*"))) # All data must be tifs

        print(f'Image path: {images_paths[subject_no]}')
        print(f'Label path: {labels_paths[subject_no]}')

        image = np.array(Image.open(images_paths[subject_no]))
        label = np.array(Image.open(labels_paths[subject_no]))

        print(f'Image shape: {np.shape(image)}')
        print(f'Label shape: {np.shape(label)}')

        print(f'Label unique: {np.unique(label)}')

        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(image)
        axes[0].set_title("Image")
        axes[0].axis('off')

        cax = axes[1].imshow(label, cmap='gray')
        axes[1].set_title("Label")
        axes[1].axis('off') 

        # Add a colorbar to the second subplot
        fig.colorbar(cax, ax=axes[1], orientation='vertical')

        plt.tight_layout()
        plt.show()
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, f'subject_{subject_no}.png'))






@timing_decorator
def main():
    data_dir = '/pscratch/sd/j/jehr/rhizo24/data'
    nnUNet_name_conversion(data_dir)




def parse_args():
    parser = argparse.ArgumentParser(description="Run a specified function with optional arguments.")
    parser.add_argument('function_name', type=str, help="The function to run")
    parser.add_argument('additional_args', nargs=argparse.REMAINDER, help="Additional arguments for the function, passed as strings to the function")
    return parser.parse_args()

if __name__ == "__main__":
    '''
    Example usage of a function in this script:
        python3 train.py train_seg_synth arg1 arg2
        python3 train.py eval arg1 arg2 arg3
        python3 train.py get_wandb_config
    NOTE: all args are passed as strings to the function called.
    '''

    args = parse_args()

    # Validate function_name argument
    function_name = args.function_name

    if function_name in globals() and callable(globals()[function_name]):
        # Call the function by name, pass additional arguments as a list
        print(f"(@utils.py) Running function: {function_name} with arguments: {args.additional_args}")
        globals()[function_name](*args.additional_args)
    else:
        print(f"Function '{function_name}' not found.")
        print(f"Available functions: {[fn for fn in globals() if callable(globals()[fn])]}")