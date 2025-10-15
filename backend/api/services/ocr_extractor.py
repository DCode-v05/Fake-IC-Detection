"""
OCR Extraction Service
Extracts text from IC images using PaddleOCR with angle classification and confidence scoring
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Any, Tuple, Optional


class OCRExtractor:
    """
    Service for extracting text from IC component images using PaddleOCR.
    Provides text extraction with bounding boxes, confidence scores, and preprocessing.
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one OCR model is loaded"""
        if cls._instance is None:
            cls._instance = super(OCRExtractor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize PaddleOCR with optimized settings for speed"""
        if self._initialized:
            return
            
        try:
            from paddleocr import PaddleOCR
            
            # Initialize PaddleOCR with optimized settings for SPEED
            # use_angle_cls=False: Disable angle classification - MUCH FASTER (IC text is usually horizontal)
            # lang='en': English language model
            # GPU will be used automatically if paddlepaddle-gpu is installed
            self.ocr = PaddleOCR(
                use_angle_cls=False,  # Disabled for SPEED - IC text is usually horizontal
                lang='en'
            )
            
            self._initialized = True
            print("✓ OCR Extractor initialized successfully (Speed-Optimized: angle_cls=OFF)")
            
        except ImportError as e:
            print(f"✗ PaddleOCR not installed: {e}")
            print("Install with: pip install paddleocr paddlepaddle")
            self.ocr = None
        except Exception as e:
            print(f"✗ Error initializing OCR Extractor: {e}")
            self.ocr = None
    
    def preprocess_image_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Lightweight preprocessing for faster OCR (optimized for speed)
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image optimized for OCR
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Simple adaptive thresholding only (fastest method)
        # Removed: CLAHE, denoising, sharpening - too slow!
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        return binary
    
    def extract_text(self, image_path: str, preprocess: bool = False) -> Dict[str, Any]:
        """
        Extract text from IC image using PaddleOCR
        
        Args:
            image_path: Path to the IC image file
            preprocess: Whether to apply preprocessing (default: False for speed)
            
        Returns:
            Dictionary containing:
                - success: bool
                - raw_results: Raw OCR results from PaddleOCR
                - extracted_text: List of detected text strings
                - confidence_scores: List of confidence scores per text
                - bounding_boxes: List of bounding box coordinates
                - full_text: All extracted text concatenated
                - error: Error message if extraction failed
        """
        if not self.ocr:
            return {
                "success": False,
                "error": "OCR engine not initialized. Install PaddleOCR."
            }
        
        try:
            # Load image
            if not os.path.exists(image_path):
                return {
                    "success": False,
                    "error": f"Image file not found: {image_path}"
                }
            
            image = cv2.imread(image_path)
            if image is None:
                return {
                    "success": False,
                    "error": "Failed to load image"
                }
            
            # Skip preprocessing for SPEED - PaddleOCR is optimized enough
            # Only preprocess if explicitly requested
            if preprocess:
                processed_image = self.preprocess_image_for_ocr(image)
            else:
                processed_image = image
            
            # Perform OCR (cls=False for speed - angle classification disabled)
            result = self.ocr.ocr(processed_image, cls=False)
            
            # Parse results
            extracted_text = []
            confidence_scores = []
            bounding_boxes = []
            
            if result and result[0]:
                for line in result[0]:
                    # Each line format: [bbox, (text, confidence)]
                    bbox = line[0]  # Bounding box coordinates
                    text_info = line[1]  # (text, confidence)
                    
                    text = text_info[0]
                    confidence = text_info[1]
                    
                    extracted_text.append(text)
                    confidence_scores.append(confidence)
                    bounding_boxes.append(bbox)
            
            # Concatenate all text
            full_text = ' '.join(extracted_text)
            
            return {
                "success": True,
                "raw_results": result,
                "extracted_text": extracted_text,
                "confidence_scores": confidence_scores,
                "bounding_boxes": bounding_boxes,
                "full_text": full_text,
                "total_detections": len(extracted_text),
                "average_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"OCR extraction failed: {str(e)}"
            }
    
    def extract_text_from_array(self, image_array: np.ndarray, preprocess: bool = False) -> Dict[str, Any]:
        """
        Extract text from numpy array image (useful for cropped logo regions)
        
        Args:
            image_array: Image as numpy array
            preprocess: Whether to apply preprocessing (default: False for speed)
            
        Returns:
            Same format as extract_text()
        """
        if not self.ocr:
            return {
                "success": False,
                "error": "OCR engine not initialized"
            }
        
        try:
            # Skip preprocessing for SPEED
            if preprocess:
                processed_image = self.preprocess_image_for_ocr(image_array)
            else:
                processed_image = image_array
            
            # Perform OCR (cls=False for speed)
            result = self.ocr.ocr(processed_image, cls=False)
            
            # Parse results (same as extract_text)
            extracted_text = []
            confidence_scores = []
            bounding_boxes = []
            
            if result and result[0]:
                for line in result[0]:
                    bbox = line[0]
                    text_info = line[1]
                    
                    text = text_info[0]
                    confidence = text_info[1]
                    
                    extracted_text.append(text)
                    confidence_scores.append(confidence)
                    bounding_boxes.append(bbox)
            
            full_text = ' '.join(extracted_text)
            
            return {
                "success": True,
                "raw_results": result,
                "extracted_text": extracted_text,
                "confidence_scores": confidence_scores,
                "bounding_boxes": bounding_boxes,
                "full_text": full_text,
                "total_detections": len(extracted_text),
                "average_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"OCR extraction failed: {str(e)}"
            }
    
    def visualize_ocr_results(self, image_path: str, output_path: str = None) -> Optional[str]:
        """
        Visualize OCR results with bounding boxes on image
        
        Args:
            image_path: Path to input image
            output_path: Path to save annotated image (optional)
            
        Returns:
            Path to annotated image or None if failed
        """
        try:
            # Extract text
            ocr_result = self.extract_text(image_path, preprocess=False)
            
            if not ocr_result["success"]:
                return None
            
            # Load original image
            image = cv2.imread(image_path)
            
            # Draw bounding boxes and text
            for bbox, text, conf in zip(
                ocr_result["bounding_boxes"],
                ocr_result["extracted_text"],
                ocr_result["confidence_scores"]
            ):
                # Convert bbox to integer coordinates
                points = np.array(bbox, dtype=np.int32)
                
                # Draw bounding box
                cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
                
                # Draw text and confidence
                label = f"{text} ({conf:.2f})"
                cv2.putText(
                    image,
                    label,
                    (int(points[0][0]), int(points[0][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )
            
            # Save annotated image
            if output_path is None:
                base_name = os.path.splitext(image_path)[0]
                output_path = f"{base_name}_ocr_annotated.jpg"
            
            cv2.imwrite(output_path, image)
            return output_path
            
        except Exception as e:
            print(f"Visualization error: {e}")
            return None


# Singleton instance
def get_ocr_extractor():
    """Get OCR extractor singleton instance"""
    return OCRExtractor()
