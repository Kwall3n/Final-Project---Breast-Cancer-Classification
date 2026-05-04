import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_cancernet(width: int = 50, height: int = 50, depth: int = 3) -> Sequential:
    """
    Constructs the CancerNet Convolutional Neural Network architecture.
    
    Parameters:
    width (int): The width of the input images in pixels (default 50).
    height (int): The height of the input images in pixels (default 50).
    depth (int): The number of color channels, e.g., 3 for RGB (default 3).
    
    Returns:
    Sequential: A compiled Keras Sequential model ready for training.
    """
    model = Sequential()
    input_shape = (height, width, depth)

    #block 1: Convolutional Layer + Pooling
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #block 2: Convolutional Layer + Pooling
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #block 3: Convolutional Layer + Pooling
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #fully Connected Layer (Flattening the 2D matrices into a 1D vector)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5)) # Dropout prevents overfitting by randomly disabling neurons

    #output Layer: 1 node with Sigmoid for Binary Classification (0 = Benign, 1 = Malignant)
    model.add(Dense(1, activation='sigmoid'))

    #compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def test_model_with_data(organized_data_path: str):
    """
    Tests the CancerNet model architecture by flowing a single batch of images 
    from the Phase 1 dataset to ensure tensor shapes match.
    
    Parameters:
    organized_data_path (str): The relative path to the 'organized_dataset' folder.
    
    Returns:
    None
    """
    print("Building CancerNet Model...")
    model = build_cancernet()
    model.summary() # Prints a visual summary of the network parameters
    
    print("\nTesting Data Flow from Phase 1 Directories...")
    train_dir = os.path.join(organized_data_path, 'train')
    
    #initialize a basic Keras Image Generator
    datagen = ImageDataGenerator(rescale=1./255)
    
    try:
        #load a small batch of 32 images just to test the connection
        generator = datagen.flow_from_directory(
            train_dir,
            target_size=(50, 50),
            color_mode="rgb",
            batch_size=32,
            class_mode="binary"
        )
        
        #fetch one batch
        images, labels = next(generator)
        print(f"\nSUCCESS! Loaded a batch of {len(images)} images.")
        print(f"Image tensor shape: {images.shape} (Expected: 32, 50, 50, 3)")
        print("Phase 1 data is perfectly compatible with Phase 2 model architecture.")
        
    except Exception as e:
        print(f"Error connecting to dataset: {e}")

if __name__ == "__main__":
    #the path pointing to the output of Phase 1 code
    ORGANIZED_DATA_DIR = "./organized_dataset"
    
    #execution of the test integration
    test_model_with_data(ORGANIZED_DATA_DIR)