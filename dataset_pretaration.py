import os
import shutil
import random

def create_directory_structure(base_path: str):
    """
    Creates the necessary directory structure for Keras image data generators.
    
    Parameters:
    base_path (str): The root directory where the sorted dataset will be stored.
    
    Returns:
    None
    """
    splits = ['train', 'val', 'test']
    classes = ['class0_benign', 'class1_malignant']
    
    for split in splits:
        for cls in classes:
            dir_path = os.path.join(base_path, split, cls)
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory: {dir_path}")

def split_and_copy_data(raw_data_path: str, target_base_path: str, train_split: float = 0.8, val_split: float = 0.1):
    """
    Parses raw image filenames, splits them into train/val/test, and copies them to the target structure.
    
    Parameters:
    raw_data_path (str): Path to the unzipped raw Kaggle dataset.
    target_base_path (str): Path where the organized dataset should be built.
    train_split (float): Percentage of data to use for training (default 0.8).
    val_split (float): Percentage of data to use for validation (default 0.1).
    
    Returns:
    None
    """
    #gather all image paths
    all_images = []
    for root, dirs, files in os.walk(raw_data_path):
        for file in files:
            if file.endswith('.png'):
                all_images.append(os.path.join(root, file))
                
    #shuffle to ensure random distribution
    random.seed(42)
    random.shuffle(all_images)
    
    #calculate split indices
    total_images = len(all_images)
    train_end = int(total_images * train_split)
    val_end = train_end + int(total_images * val_split)
    
    train_images = all_images[:train_end]
    val_images = all_images[train_end:val_end]
    test_images = all_images[val_end:]
    
    #helper function to copy files based on their class label in the filename
    def copy_images(image_list, split_name):
        for img_path in image_list:
            filename = os.path.basename(img_path)
            #filenames look like: 8863_idx5_x451_y1451_class0.png
            if 'class0' in filename:
                dest = os.path.join(target_base_path, split_name, 'class0_benign', filename)
            elif 'class1' in filename:
                dest = os.path.join(target_base_path, split_name, 'class1_malignant', filename)
            else:
                continue
            shutil.copy2(img_path, dest)
            
    print("Copying training data...")
    copy_images(train_images, 'train')
    print("Copying validation data...")
    copy_images(val_images, 'val')
    print("Copying testing data...")
    copy_images(test_images, 'test')
    
    print("Data splitting and sorting complete!")

if __name__ == "__main__":
    #paths
    RAW_DATA_DIR = "./raw_idc_data" #extracted file set
    ORGANIZED_DATA_DIR = "./organized_dataset"
    
    #pipeline execution
    create_directory_structure(ORGANIZED_DATA_DIR)
    
    split_and_copy_data(RAW_DATA_DIR, ORGANIZED_DATA_DIR)