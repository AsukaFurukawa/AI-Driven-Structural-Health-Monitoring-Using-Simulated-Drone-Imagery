"""
Feature extraction module for structural defect detection.
"""
import cv2
import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass

@dataclass
class DefectFeatures:
    """Container for extracted defect features."""
    location: Tuple[int, int]  # (x, y) coordinates
    size: float  # area in pixels
    severity: float  # normalized severity score
    type: str  # type of defect
    confidence: float  # detection confidence

class StructuralFeatureExtractor:
    """Class for extracting features related to structural defects."""
    
    def __init__(
        self,
        edge_threshold: float = 100,
        area_threshold: float = 100,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the feature extractor.
        
        Args:
            edge_threshold: Threshold for edge detection
            area_threshold: Minimum area for defect detection
            confidence_threshold: Minimum confidence score
        """
        self.edge_threshold = edge_threshold
        self.area_threshold = area_threshold
        self.confidence_threshold = confidence_threshold
        
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges in the image using Canny edge detection.
        
        Args:
            image: Input image array
            
        Returns:
            Binary edge map
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect edges
        edges = cv2.Canny(
            blurred,
            threshold1=self.edge_threshold / 2,
            threshold2=self.edge_threshold
        )
        
        return edges
        
    def detect_contours(
        self,
        image: np.ndarray,
        edges: np.ndarray
    ) -> List[np.ndarray]:
        """
        Detect contours in edge map.
        
        Args:
            image: Original image array
            edges: Binary edge map
            
        Returns:
            List of contours
        """
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by area
        significant_contours = [
            cnt for cnt in contours
            if cv2.contourArea(cnt) > self.area_threshold
        ]
        
        return significant_contours
        
    def analyze_texture(
        self,
        image: np.ndarray,
        region: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyze texture features in a region.
        
        Args:
            image: Input image array
            region: Binary mask defining region of interest
            
        Returns:
            Dictionary of texture features
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Apply mask
        masked = cv2.bitwise_and(gray, gray, mask=region)
        
        # Calculate texture features
        mean = np.mean(masked[region > 0])
        std = np.std(masked[region > 0])
        
        # Calculate GLCM features
        glcm = self._compute_glcm(masked)
        contrast = np.sum(np.square(glcm))
        homogeneity = np.sum(glcm / (1 + np.arange(glcm.shape[0])[:, None]))
        
        return {
            'mean': mean,
            'std': std,
            'contrast': contrast,
            'homogeneity': homogeneity
        }
        
    def _compute_glcm(self, image: np.ndarray) -> np.ndarray:
        """
        Compute Gray-Level Co-occurrence Matrix.
        
        Args:
            image: Grayscale image array
            
        Returns:
            GLCM matrix
        """
        # Quantize to 8 levels
        levels = 8
        quantized = (image / (256 / levels)).astype(np.uint8)
        
        # Compute GLCM
        h, w = quantized.shape
        glcm = np.zeros((levels, levels))
        
        for i in range(h-1):
            for j in range(w-1):
                i_val = quantized[i, j]
                j_val = quantized[i+1, j+1]
                glcm[i_val, j_val] += 1
                
        # Normalize
        glcm = glcm / np.sum(glcm)
        
        return glcm
        
    def extract_features(
        self,
        image: np.ndarray
    ) -> List[DefectFeatures]:
        """
        Extract structural defect features from image.
        
        Args:
            image: Input image array
            
        Returns:
            List of detected defect features
        """
        # Detect edges
        edges = self.detect_edges(image)
        
        # Find contours
        contours = self.detect_contours(image, edges)
        
        defects = []
        for contour in contours:
            # Get contour properties
            area = cv2.contourArea(contour)
            M = cv2.moments(contour)
            
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                continue
                
            # Create mask for texture analysis
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            # Analyze texture
            texture_features = self.analyze_texture(image, mask)
            
            # Calculate severity based on area and texture
            severity = (
                (area / (image.shape[0] * image.shape[1])) * 
                texture_features['contrast']
            )
            
            # Determine defect type based on shape and texture
            if texture_features['contrast'] > 0.5:
                defect_type = 'crack'
            elif texture_features['homogeneity'] > 0.8:
                defect_type = 'spalling'
            else:
                defect_type = 'other'
                
            # Calculate confidence score
            confidence = min(
                1.0,
                (area / self.area_threshold) * 
                texture_features['contrast']
            )
            
            if confidence >= self.confidence_threshold:
                defect = DefectFeatures(
                    location=(cx, cy),
                    size=area,
                    severity=severity,
                    type=defect_type,
                    confidence=confidence
                )
                defects.append(defect)
                
        return defects 