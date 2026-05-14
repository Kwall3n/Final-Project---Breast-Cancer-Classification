import os
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#import the CNN architecture from phase 2
from model_architecture import build_cancernet

def train_cancernet(organized_data_path: str, model_save_path: str, history_save_path: str):
    """
    Trains the CancerNet CNN on the organized IDC dataset.
    
    Parameters:
    organized_data_path (str): Path to the organized train/val/test data.
    model_save_path (str): Path to save the best trained Keras model.
    history_save_path (str): Path to save the training history for visualization.
    """
    print("Initializing Data Generators...")
    train_dir = os.path.join(organized_data_path, 'train')
    val_dir = os.path.join(organized_data_path, 'val')

    #rescale pixel values from 0-255 down to 0-1 
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    #load data in batches of 64
    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(50, 50), color_mode="rgb", batch_size=64, class_mode="binary"
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir, target_size=(50, 50), color_mode="rgb", batch_size=64, class_mode="binary"
    )

    print("Building Model...")
    model = build_cancernet()

    #callbacks to prevent overfitting and save the best weights
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True)

    print("Starting Training")
    #limiting to 10 epochs for local testing
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10, 
        callbacks=[early_stop, checkpoint]
    )

    print("Training Complete! Saving training history...")
    with open(history_save_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    
    print(f"Model successfully saved to {model_save_path}")
    print(f"Training history saved to {history_save_path}")

if __name__ == "__main__":
    ORGANIZED_DATA_DIR = "./organized_dataset"
    MODEL_OUTPUT = "cancernet_best_model.keras"
    HISTORY_OUTPUT = "training_history.pkl"
    
    #execution of the training pipeline
    train_cancernet(ORGANIZED_DATA_DIR, MODEL_OUTPUT, HISTORY_OUTPUT)