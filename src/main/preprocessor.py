import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Union
from dataclasses import dataclass
import json

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """Configuration settings for image preprocessing"""
    # Basic settings
    target_dpi: int = 300
    min_width: int = 800
    max_width: int = 1200
    
    # Enhancement settings
    contrast_factor: float = 1.2
    sharpness_factor: float = 1.1
    brightness_factor: float = 1.0
    
    # Cleaning settings
    denoise_strength: int = 10
    remove_screentone: bool = True
    screentone_threshold: int = 127
    
    # Border settings
    remove_borders: bool = True
    border_threshold: int = 240
    min_border_width: int = 10
    
    # Advanced settings
    adaptive_contrast: bool = True
    preserve_lines: bool = True
    auto_rotate: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'PreprocessingConfig':
        """Create config from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() 
                     if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {field: getattr(self, field) 
                for field in self.__dataclass_fields__}
    
    @classmethod
    def load(cls, path: Path) -> 'PreprocessingConfig':
        """Load config from JSON file"""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))
    
    def save(self, path: Path):
        """Save config to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

class ImagePreprocessor:
    """Handles image preprocessing and enhancement"""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Optional preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        logger.info(f"Initialized preprocessor with config: {self.config}")

    def preprocess(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Apply full preprocessing pipeline to image.
        
        Args:
            image: Input image (OpenCV or PIL format)
            
        Returns:
            Preprocessed image as NumPy array
        """
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        logger.info("Starting preprocessing pipeline")
        
        # Basic preprocessing
        image = self._resize_image(image)
        image = self._auto_rotate(image)
        
        # Enhanced preprocessing
        image = self._enhance_image(image)
        
        # Convert to OpenCV format for advanced processing
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Advanced preprocessing
        cv_image = self._remove_borders(cv_image)
        cv_image = self._clean_image(cv_image)
        cv_image = self._remove_screentones(cv_image)
        
        logger.info("Preprocessing completed")
        return cv_image

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resize image while maintaining aspect ratio and DPI.
        
        Args:
            image: PIL Image to resize
            
        Returns:
            Resized image
        """
        # Calculate target width
        aspect_ratio = image.height / image.width
        target_width = min(max(self.config.min_width,
                             image.width),
                         self.config.max_width)
        target_height = int(target_width * aspect_ratio)
        
        # Resize using high-quality downsampling
        if image.width > target_width:
            return image.resize((target_width, target_height),
                              Image.Resampling.LANCZOS)
        return image

    def _auto_rotate(self, image: Image.Image) -> Image.Image:
        """
        Automatically rotate image if needed.
        
        Args:
            image: PIL Image to rotate
            
        Returns:
            Rotated image if needed
        """
        if not self.config.auto_rotate:
            return image
        
        # Convert to grayscale for line detection
        gray = image.convert('L')
        edges = cv2.Canny(np.array(gray), 50, 150)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            # Find dominant angle
            angles = [line[0][1] for line in lines]
            dominant_angle = np.median(angles)
            
            # Convert to degrees and normalize
            angle_deg = np.degrees(dominant_angle) % 90
            if angle_deg > 45:
                angle_deg -= 90
            
            # Rotate if angle is significant
            if abs(angle_deg) > 0.5:
                logger.info(f"Rotating image by {-angle_deg:.2f} degrees")
                return image.rotate(-angle_deg, Image.Resampling.BICUBIC,
                                  expand=True)
        
        return image

    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """
        Enhance image quality.
        
        Args:
            image: PIL Image to enhance
            
        Returns:
            Enhanced image
        """
        # Apply contrast enhancement
        if self.config.adaptive_contrast:
            # Use adaptive contrast enhancement
            enhancer = ImageEnhance.Contrast(image)
            lab = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
            l_channel = lab[:,:,0]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l_channel)
            lab[:,:,0] = cl
            image = Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
        else:
            # Use basic contrast enhancement
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(self.config.contrast_factor)
        
        # Apply sharpness enhancement
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(self.config.sharpness_factor)
        
        # Apply brightness adjustment
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(self.config.brightness_factor)
        
        return image

    def _clean_image(self, image: np.ndarray) -> np.ndarray:
        """
        Clean and denoise image.
        
        Args:
            image: OpenCV image to clean
            
        Returns:
            Cleaned image
        """
        # Apply denoising if enabled
        if self.config.denoise_strength > 0:
            image = cv2.fastNlMeansDenoisingColored(
                image,
                None,
                self.config.denoise_strength,
                self.config.denoise_strength,
                7,
                21
            )
        
        # Preserve lines if enabled
        if self.config.preserve_lines:
            # Detect edges
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate edges to ensure preservation
            kernel = np.ones((2,2), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Blend edges back into image
            for c in range(3):
                image[:,:,c] = cv2.addWeighted(
                    image[:,:,c], 1,
                    edges, 0.3,
                    0
                )
        
        return image

    def _remove_borders(self, image: np.ndarray) -> np.ndarray:
        """
        Remove white borders from image.
        
        Args:
            image: OpenCV image to process
            
        Returns:
            Image with borders removed
        """
        if not self.config.remove_borders:
            return image
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.config.border_threshold, 255,
                                cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)
            
            # Check if border is significant
            if (w < image.shape[1] - self.config.min_border_width or
                h < image.shape[0] - self.config.min_border_width):
                return image[y:y+h, x:x+w]
        
        return image

    def _remove_screentones(self, image: np.ndarray) -> np.ndarray:
        """
        Remove screentone patterns from image.
        
        Args:
            image: OpenCV image to process
            
        Returns:
            Image with screentones removed
        """
        if not self.config.remove_screentone:
            return image
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # Remove small dots (typical in screentones)
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Blend back with original
        result = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(image, 0.7, result, 0.3, 0)

    def estimate_quality(self, image: np.ndarray) -> Dict[str, float]:
        """
        Estimate quality metrics for the image.
        
        Args:
            image: OpenCV image to analyze
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        # Calculate contrast
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        metrics['contrast'] = gray.std()
        
        # Calculate sharpness
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        metrics['sharpness'] = laplacian.std()
        
        # Calculate noise level
        noise = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        noise_level = np.mean(np.abs(image - noise))
        metrics['noise_level'] = noise_level
        
        # Normalize metrics
        for key in metrics:
            metrics[key] = float(metrics[key])
        
        return metrics

# Example usage
if __name__ == "__main__":
    # Create preprocessor with custom config
    config = PreprocessingConfig(
        target_dpi=300,
        contrast_factor=1.3,
        remove_screentone=True,
        preserve_lines=True
    )
    
    preprocessor = ImagePreprocessor(config)
    
    # Process test image
    test_image_path = Path("test_image.jpg")
    if test_image_path.exists():
        # Load and process image
        image = Image.open(test_image_path)
        processed = preprocessor.preprocess(image)
        
        # Save processed image
        output_path = Path("processed_image.jpg")
        cv2.imwrite(str(output_path), processed)
        
        # Check quality
        quality_metrics = preprocessor.estimate_quality(processed)
        print("Quality metrics:", quality_metrics)
    else:
        print(f"Test image not found: {test_image_path}")