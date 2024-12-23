from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
import os

import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass
import tensorflow as tf
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ImageSegment:
    """Represents a segment of the original image with its properties"""
    start_y: int
    end_y: int
    content_density: float
    split_confidence: float
    segment_height: int = 0

    def __post_init__(self):
        self.segment_height = self.end_y - self.start_y

class ImageProcessor:
    """Main class for processing and splitting comic images"""
    
    def __init__(self, 
                 target_height: int = 2000,
                 min_height: int = 1500,
                 max_height: int = 2500,
                 quality: int = 90,
                 overlap_pixels: int = 50):
        """
        Initialize the image processor with configuration parameters.
        
        Args:
            target_height: Ideal height for output images
            min_height: Minimum acceptable height for splits
            max_height: Maximum acceptable height for splits
            quality: WebP output quality (0-100)
            overlap_pixels: Pixels to overlap between segments
        """
        self.target_height = target_height
        self.min_height = min_height
        self.max_height = max_height
        self.quality = quality
        self.overlap_pixels = overlap_pixels
        self.model = None  # ML model will be loaded when needed

    def load_image(self, image_path: Path) -> Tuple[np.ndarray, Image.Image]:
        """
        Load and prepare an image for processing.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of OpenCV image array and PIL Image
        """
        try:
            # Load with PIL for WebP support
            pil_image = Image.open(image_path)
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to OpenCV format for processing
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return cv_image, pil_image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            raise

    def analyze_content_density(self, image: np.ndarray, window_size: int = 100) -> np.ndarray:
        """
        Analyze the content density across the image height.
        
        Args:
            image: OpenCV image array
            window_size: Size of the sliding window for analysis
            
        Returns:
            Array of content density values
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height = gray.shape[0]
        density_values = np.zeros(height)
        
        for y in range(0, height - window_size, window_size):
            window = gray[y:y + window_size, :]
            # Calculate non-white pixel ratio
            non_white = np.count_nonzero(window < 250) / window.size
            density_values[y:y + window_size] = non_white
            
        return density_values

    def find_split_points(self, 
                         image: np.ndarray,
                         density_values: np.ndarray) -> List[ImageSegment]:
        """
        Find optimal points to split the image using content density and ML predictions.
        
        Args:
            image: OpenCV image array
            density_values: Pre-calculated content density values
            
        Returns:
            List of ImageSegment objects
        """
        height = image.shape[0]
        segments = []
        current_y = 0
        
        while current_y < height:
            # Determine potential split region
            max_region = min(current_y + self.max_height, height)
            min_region = min(current_y + self.min_height, height)
            
            if min_region >= height:
                # Add final segment
                segments.append(ImageSegment(
                    start_y=current_y,
                    end_y=height,
                    content_density=float(np.mean(density_values[current_y:height])),
                    split_confidence=1.0
                ))
                break
            
            # Find optimal split point using density values and ML model
            split_point, confidence = self._find_optimal_split(
                image[current_y:max_region],
                density_values[current_y:max_region]
            )
            
            segments.append(ImageSegment(
                start_y=current_y,
                end_y=current_y + split_point,
                content_density=float(np.mean(density_values[current_y:current_y + split_point])),
                split_confidence=confidence
            ))
            
            current_y += split_point - self.overlap_pixels
            
        return segments

    def _find_optimal_split(self, 
                          image_section: np.ndarray, 
                          density_values: np.ndarray) -> Tuple[int, float]:
        """
        Find the optimal split point within a section of the image.
        
        Args:
            image_section: Section of the image to analyze
            density_values: Content density values for the section
            
        Returns:
            Tuple of (split_point, confidence)
        """
        # TODO: Implement ML model prediction here
        # For now, use simple density-based approach
        target_idx = min(self.target_height, len(density_values))
        window = 100  # Window to search around target height
        
        start_idx = max(0, target_idx - window)
        end_idx = min(len(density_values), target_idx + window)
        
        # Find local minimum in density
        search_region = density_values[start_idx:end_idx]
        local_min_idx = start_idx + np.argmin(search_region)
        
        return local_min_idx, 0.8  # Placeholder confidence

    def save_segment(self,
                    pil_image: Image.Image,
                    segment: ImageSegment,
                    output_path: Path,
                    index: int) -> Path:
        """
        Save an image segment as WebP file.
        
        Args:
            pil_image: Original PIL Image
            segment: ImageSegment object
            output_path: Directory to save the output
            index: Segment index for filename
            
        Returns:
            Path to saved file
        """
        try:
            # Crop segment
            segment_image = pil_image.crop((0, segment.start_y, 
                                          pil_image.width, segment.end_y))
            
            # Prepare output filename
            output_file = output_path / f"segment_{index:03d}.webp"
            
            # Save as WebP
            segment_image.save(
                output_file,
                format='WebP',
                quality=self.quality,
                method=6  # Highest compression effort
            )
            
            return output_file
        except Exception as e:
            logger.error(f"Error saving segment {index}: {str(e)}")
            raise

    def process_image(self, 
                     input_path: Path,
                     output_dir: Path) -> List[Path]:
        """
        Process a single image file.
        
        Args:
            input_path: Path to input image
            output_dir: Directory for output files
            
        Returns:
            List of paths to output files
        """
        logger.info(f"Processing {input_path}")
        
        # Create output directory if needed
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load image
        cv_image, pil_image = self.load_image(input_path)
        
        # Analyze content
        density_values = self.analyze_content_density(cv_image)
        
        # Find split points
        segments = self.find_split_points(cv_image, density_values)
        
        # Save segments
        output_files = []
        for i, segment in enumerate(segments):
            output_file = self.save_segment(pil_image, segment, output_dir, i)
            output_files.append(output_file)
            
        logger.info(f"Created {len(output_files)} segments")
        return output_files

    def process_batch(self,
                     input_files: List[Path],
                     output_dir: Path,
                     progress_callback: Optional[callable] = None) -> Dict[Path, List[Path]]:
        """
        Process multiple images in batch.
        
        Args:
            input_files: List of input image paths
            output_dir: Base directory for outputs
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary mapping input paths to lists of output paths
        """
        results = {}
        
        for input_file in tqdm(input_files, desc="Processing images"):
            # Create specific output directory for this file
            file_output_dir = output_dir / input_file.stem
            
            try:
                output_files = self.process_image(input_file, file_output_dir)
                results[input_file] = output_files
                
                if progress_callback:
                    progress_callback(input_file, True, None)
                    
            except Exception as e:
                logger.error(f"Error processing {input_file}: {str(e)}")
                if progress_callback:
                    progress_callback(input_file, False, str(e))
                results[input_file] = []
                
        return results

# Usage example:
if __name__ == "__main__":
    processor = ImageProcessor(
        target_height=2000,
        min_height=1500,
        max_height=2500,
        quality=90
    )
    
    # Process single image
    input_path = Path("input/comic.jpg")
    output_dir = Path("output")
    
    try:
        output_files = processor.process_image(input_path, output_dir)
        print(f"Successfully processed {input_path}")
        print(f"Created segments: {[str(f) for f in output_files]}")
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")