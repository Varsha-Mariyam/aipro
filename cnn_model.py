"""
CNN Model for Aadhaar Fraud Detection
Lightweight, efficient, optimized for CPU training
"""


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np


class FraudCNN:
    """
    Optimized CNN for fraud detection
    Designed for CPU training with limited data
    """
    
    def __init__(self, img_size=(128, 128), batch_size=16):
        """
        Initialize CNN
        
        Args:
            img_size: Input image size (smaller = faster training)
            batch_size: Batch size (smaller = less memory)
        """
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
    
    def build_model(self):
        """
        Build lightweight CNN architecture
        
        Architecture:
        - 4 Conv blocks with BatchNorm + Dropout
        - Global Average Pooling (instead of Flatten)
        - 2 Dense layers
        - Binary classification output
        """
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(*self.img_size, 3)),
            
            # Block 1: Feature extraction
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2: Deeper features
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3: Complex patterns
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 4: High-level features
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),  # Better than Flatten
            layers.Dropout(0.5),
            
            # Classification head
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')  # 0=authentic, 1=fake
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        self.model = model
        
        print("\n🏗️ Model Architecture:")
        print("="*60)
        model.summary()
        print("="*60)
        
        return model
    
    def prepare_data(self, data_dir='Images'):
        """
        Prepare data generators with augmentation
        
        Args:
            data_dir: Root directory containing train/val folders
        """
        
        # Training data augmentation (light augmentation)
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=5,           # Slight rotation
            width_shift_range=0.05,     # Small shifts
            height_shift_range=0.05,
            zoom_range=0.05,
            horizontal_flip=False,      # Don't flip documents
            fill_mode='nearest'
        )
        
        # Validation data (no augmentation, only rescale)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Load training data
        train_generator = train_datagen.flow_from_directory(
            f'{data_dir}/train',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=True,
            color_mode='rgb'
        )
        
        # Load validation data
        val_generator = val_datagen.flow_from_directory(
            f'{data_dir}/valid',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False,
            color_mode='rgb'
        )
        
        print(f"\n📊 Data Loaded:")
        print(f"   Training samples:   {train_generator.samples}")
        print(f"   Validation samples: {val_generator.samples}")
        print(f"   Classes: {train_generator.class_indices}")
        print("="*60)
        
        return train_generator, val_generator
    
    def train(self, data_dir='Images', epochs=30, model_save_path='models/fraud_cnn.h5'):
        """
        Train the CNN model
        
        Args:
            data_dir: Data directory
            epochs: Number of training epochs
            model_save_path: Path to save best model
        """
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Prepare data
        train_gen, val_gen = self.prepare_data(data_dir)
        
        # Callbacks for training
        callbacks = [
            # Save best model based on validation accuracy
            ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Early stopping if no improvement
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        print("\n🚀 Starting training...")
        print("="*60)
        print(f"   Epochs: {epochs}")
        print(f"   Image size: {self.img_size}")
        print(f"   Batch size: {self.batch_size}")
        print("="*60)
        
        # Train model
        self.history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"\n✅ Training complete!")
        print(f"   Model saved: {model_save_path}")
        
        # Plot training history
        self.plot_training_history()
        
        return self.history
    
    def plot_training_history(self):
        """Plot training metrics"""
        
        if self.history is None:
            print("⚠️ No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Val', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Val', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train', linewidth=2)
        axes[1, 0].plot(self.history.history['val_precision'], label='Val', linewidth=2)
        axes[1, 0].set_title('Precision', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train', linewidth=2)
        axes[1, 1].plot(self.history.history['val_recall'], label='Val', linewidth=2)
        axes[1, 1].set_title('Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("📊 Training plots saved: outputs/training_history.png")
        """plt.show() """
    
    def evaluate(self, test_dir):
        """Evaluate model on test data"""
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        print("\n📊 Evaluating model...")
        results = self.model.evaluate(test_generator)
        
        print(f"\n✅ Test Results:")
        print(f"   Loss: {results[0]:.4f}")
        print(f"   Accuracy: {results[1]:.4f}")
        print(f"   Precision: {results[2]:.4f}")
        print(f"   Recall: {results[3]:.4f}")
        
        return results


# Training script
if __name__ == "__main__":
    # Initialize CNN
    fraud_cnn = FraudCNN(img_size=(128, 128), batch_size=16)
    
    # Build model
    fraud_cnn.build_model()
    
    # Train model
    history = fraud_cnn.train(
        data_dir='Images',
        epochs=30,
        model_save_path='models/fraud_cnn.h5'
    )
    
    print("\n🎉 Training complete!")
    print("   Model saved to: models/fraud_cnn.h5")
    print("   Training plots: outputs/training_history.png")
