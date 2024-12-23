import os
from pathlib import Path
from typing import Tuple, List, Optional, Union, Dict
import logging
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import cv2
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SplitPrediction:
    """Represents a prediction for a split point"""
    position: int
    confidence: float
    features: Dict[str, float]

class SplitPointModel:
    """Neural network model for detecting optimal split points in comics"""
    
    def __init__(self,
                 input_height: int = 3000,
                 feature_channels: int = 7,
                 model_path: Optional[Path] = None):
        """
        Initialize the split point detection model.
        
        Args:
            input_height: Height of input image sections
            feature_channels: Number of feature channels (density, edges, etc.)
            model_path: Optional path to load pre-trained model
        """
        self.input_height = input_height
        self.feature_channels = feature_channels
        self.model = self._build_model()
        
        if model_path and model_path.exists():
            self.load_weights(model_path)

    def _build_model(self) -> models.Model:
        """
        Build the neural network architecture.
        
        Returns:
            Compiled TensorFlow model
        """
        inputs = layers.Input(shape=(self.input_height, 1, self.feature_channels))
        
        # Feature extraction layers
        x = layers.Conv2D(32, (7, 1), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (5, 1), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Context understanding
        x = layers.Conv2D(128, (3, 1), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Global features
        global_features = layers.GlobalAveragePooling2D()(x)
        global_features = layers.Dense(256, activation='relu')(global_features)
        
        # Local features processing
        local_features = layers.Conv2D(256, (3, 1), activation='relu', padding='same')(x)
        local_features = layers.Conv2D(1, (1, 1), activation='sigmoid')(local_features)
        
        # Combine global and local features
        local_features = layers.Reshape((self.input_height,))(local_features)
        
        # Output layers
        split_logits = layers.Dense(self.input_height, activation='softmax', name='split_points')(global_features)
        confidence = layers.Dense(1, activation='sigmoid', name='confidence')(global_features)
        
        model = models.Model(
            inputs=inputs,
            outputs=[split_logits, confidence]
        )
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss={
                'split_points': 'categorical_crossentropy',
                'confidence': 'binary_crossentropy'
            },
            metrics={
                'split_points': ['accuracy'],
                'confidence': ['accuracy']
            }
        )
        
        return model

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from image section for model input.
        
        Args:
            image: Image section to analyze
            
        Returns:
            Feature array with shape (height, 1, channels)
        """
        height = image.shape[0]
        features = np.zeros((height, 1, self.feature_channels))
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Feature 1: Content density
        features[:, 0, 0] = self._compute_density(gray)
        
        # Feature 2: Edge detection
        edges = cv2.Canny(gray, 100, 200)
        features[:, 0, 1] = self._compute_density(edges)
        
        # Feature 3: Horizontal lines
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horiz_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horiz_kernel)
        features[:, 0, 2] = self._compute_density(horiz_lines)
        
        # Feature 4: Vertical lines
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        vert_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vert_kernel)
        features[:, 0, 3] = self._compute_density(vert_lines)
        
        # Feature 5: Text regions
        text_regions = self._detect_text_regions(gray)
        features[:, 0, 4] = self._compute_density(text_regions)
        
        # Feature 6: Local contrast
        features[:, 0, 5] = self._compute_local_contrast(gray)
        
        # Feature 7: Position encoding
        features[:, 0, 6] = np.linspace(0, 1, height)
        
        return features

    def _compute_density(self, image: np.ndarray, window_size: int = 50) -> np.ndarray:
        """
        Compute content density using sliding window.
        
        Args:
            image: Input image
            window_size: Size of sliding window
            
        Returns:
            Array of density values
        """
        height = image.shape[0]
        density = np.zeros(height)
        
        for y in range(0, height - window_size + 1):
            window = image[y:y + window_size, :]
            density[y:y + window_size] = np.mean(window > 0)
            
        return density

    def _detect_text_regions(self, image: np.ndarray) -> np.ndarray:
        """
        Detect potential text regions in the image.
        
        Args:
            image: Grayscale image
            
        Returns:
            Binary mask of text regions
        """
        # Simple text detection using local variance
        kernel_size = (25, 25)
        mean, stddev = cv2.meanStdDev(image)
        variance = cv2.GaussianBlur(image, kernel_size, 0)
        threshold = mean + 2 * stddev
        
        return (variance > threshold).astype(np.uint8) * 255

    def _compute_local_contrast(self, image: np.ndarray, window_size: int = 50) -> np.ndarray:
        """
        Compute local contrast values.
        
        Args:
            image: Grayscale image
            window_size: Size of sliding window
            
        Returns:
            Array of local contrast values
        """
        height = image.shape[0]
        contrast = np.zeros(height)
        
        for y in range(0, height - window_size + 1):
            window = image[y:y + window_size, :]
            contrast[y:y + window_size] = np.std(window)
            
        return contrast / 255.0

    def predict_split(self, image: np.ndarray) -> List[SplitPrediction]:
        """
        Predict optimal split points for an image section.
        
        Args:
            image: Image section to analyze
            
        Returns:
            List of SplitPrediction objects
        """
        # Extract features
        features = self.extract_features(image)
        features_batch = np.expand_dims(features, axis=0)
        
        # Get model predictions
        split_logits, confidence = self.model.predict(features_batch, verbose=0)
        
        # Process predictions
        split_probs = split_logits[0]
        confidence_score = float(confidence[0][0])
        
        # Find local maxima in split probabilities
        peaks = self._find_peaks(split_probs)
        
        predictions = []
        for peak in peaks:
            pred = SplitPrediction(
                position=int(peak),
                confidence=float(split_probs[peak]) * confidence_score,
                features={
                    'density': float(features[peak, 0, 0]),
                    'edges': float(features[peak, 0, 1]),
                    'horiz_lines': float(features[peak, 0, 2]),
                    'vert_lines': float(features[peak, 0, 3]),
                    'text': float(features[peak, 0, 4]),
                    'contrast': float(features[peak, 0, 5]),
                }
            )
            predictions.append(pred)
        
        return sorted(predictions, key=lambda x: x.confidence, reverse=True)

    def _find_peaks(self, 
                   probabilities: np.ndarray,
                   min_distance: int = 100,
                   threshold: float = 0.3) -> List[int]:
        """
        Find local maxima in probability distribution.
        
        Args:
            probabilities: Array of split point probabilities
            min_distance: Minimum distance between peaks
            threshold: Minimum probability threshold
            
        Returns:
            List of peak indices
        """
        peaks = []
        length = len(probabilities)
        
        for i in range(1, length - 1):
            if probabilities[i] < threshold:
                continue
                
            if probabilities[i] > probabilities[i - 1] and probabilities[i] > probabilities[i + 1]:
                if not peaks or (i - peaks[-1]) >= min_distance:
                    peaks.append(i)
        
        return peaks

    def save_weights(self, path: Path):
        """Save model weights to file"""
        self.model.save_weights(str(path))
        
    def load_weights(self, path: Path):
        """Load model weights from file"""
        self.model.load_weights(str(path))

    def train(self,
             train_images: List[np.ndarray],
             train_labels: List[np.ndarray],
             validation_split: float = 0.2,
             epochs: int = 50,
             batch_size: int = 32,
             callbacks: List[tf.keras.callbacks.Callback] = None) -> tf.keras.callbacks.History:
        """
        Train the model on a dataset.
        
        Args:
            train_images: List of training images
            train_labels: List of corresponding labels
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Training batch size
            callbacks: Optional list of Keras callbacks
            
        Returns:
            Training history
        """
        # Prepare features
        features = [self.extract_features(img) for img in train_images]
        features = np.stack(features)
        
        # Convert labels to appropriate format
        split_labels = np.array([self._convert_to_split_labels(label) 
                               for label in train_labels])
        confidence_labels = np.ones((len(train_labels), 1))  # Assuming all splits are valid
        
        # Train the model
        history = self.model.fit(
            features,
            {
                'split_points': split_labels,
                'confidence': confidence_labels
            },
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history

    def _convert_to_split_labels(self, label: np.ndarray) -> np.ndarray:
        """
        Convert split point labels to one-hot encoding.
        
        Args:
            label: Array of split point positions
            
        Returns:
            One-hot encoded labels
        """
        one_hot = np.zeros(self.input_height)
        one_hot[label] = 1
        return one_hot

# Example usage:
if __name__ == "__main__":
    # Initialize model
    model = SplitPointModel(
        input_height=3000,
        feature_channels=7
    )
    
    # Example prediction
    test_image = np.random.randint(0, 255, (3000, 800, 3), dtype=np.uint8)
    predictions = model.predict_split(test_image)
    
    # Print predictions
    for pred in predictions:
        print(f"Split point at {pred.position} with confidence {pred.confidence:.2f}")
        print("Feature values:", pred.features)
    
    # Example training (with dummy data)
    num_samples = 10
    train_images = [np.random.randint(0, 255, (3000, 800, 3), dtype=np.uint8) 
                   for _ in range(num_samples)]
    train_labels = [np.array([1000, 2000]) for _ in range(num_samples)]  # Example split points
    
    # Add callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    # Train the model
    history = model.train(
        train_images,
        train_labels,
        epochs=10,
        batch_size=2,
        callbacks=callbacks
    )
    
    print("Training completed!")
    print("Final training accuracy:", history.history['split_points_accuracy'][-1])
    print("Final validation accuracy:", history.history['val_split_points_accuracy'][-1])