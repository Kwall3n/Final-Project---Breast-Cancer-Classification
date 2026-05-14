import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def run_evaluation(history_path, model_path, test_data_path):
    #load history
    with open(history_path, 'rb') as f:
        h = pickle.load(f)
    
    #plot and save learning curves
    epochs = range(1, len(h['accuracy']) + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, h['accuracy'], 'b-', label='Train Acc')
    plt.plot(epochs, h['val_accuracy'], 'r-', label='Val Acc')
    plt.title('Accuracy Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, h['loss'], 'b-', label='Train Loss')
    plt.plot(epochs, h['val_loss'], 'r-', label='Val Loss')
    plt.title('Loss Curve')
    plt.legend()
    
    plt.savefig('learning_curves.png') # SAVES THE IMAGE
    print("✓ Saved learning_curves.png")
    plt.show()

    #load model and test data
    print("Loading model for final test...")
    model = load_model(model_path)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_data_path, target_size=(50, 50), batch_size=32, 
        class_mode='binary', shuffle=False
    )
    
    #predictions
    predictions = model.predict(test_generator)
    y_pred = (predictions > 0.5).astype(int).flatten()
    y_true = test_generator.classes
    
    #confusion matrix and save
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malignant'], 
                yticklabels=['Benign', 'Malignant'])
    plt.title('Final Confusion Matrix')
    plt.savefig('confusion_matrix.png') # SAVES THE IMAGE
    print("✓ Saved confusion_matrix.png")
    plt.show()

    #save report to txt file
    report = classification_report(y_true, y_pred, target_names=['Benign', 'Malignant'])
    with open("evaluation_report.txt", "w") as f:
        f.write(report)
    print("✓ Saved evaluation_report.txt")
    print("\nFinal Results:\n", report)

if __name__ == "__main__":
    run_evaluation('training_history.pkl', 'cancernet_best_model.keras', './organized_dataset/test')