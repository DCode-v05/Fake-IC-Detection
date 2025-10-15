"""
YOLOv8 Logo Detector Service
Replaces the old contour-based logo detection with AI-powered YOLO object detection
Detects IC manufacturer logos directly without manual cropping
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from PIL import Image
import cv2


class YOLOLogoDetector:
    """
    Service for detecting IC manufacturer logos using YOLOv8
    Provides end-to-end logo detection and classification in one step
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one YOLO model is loaded"""
        if cls._instance is None:
            cls._instance = super(YOLOLogoDetector, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize YOLO logo detector"""
        if self._initialized:
            return
        
        self.model = None
        self.model_path = None
        self.class_names = None
        self.load_model()
        self._initialized = True
    
    def load_model(self):
        """
        Load trained YOLOv8 model for logo detection
        """
        try:
            from ultralytics import YOLO
            
            # Model path: prioritize the newly trained model
            project_root = Path(__file__).parent.parent.parent.parent
            possible_paths = [
                project_root / "trained_models" / "best.pt",  # Primary: newly trained model
                project_root / "trained_models" / "yolo_logo_best.pt",  # Alternative name
                project_root / "runs" / "detect" / "yolov8_logo_run" / "weights" / "best.pt",  # Training output
                project_root / "model_training" / "runs" / "detect" / "yolov8_logo_run" / "weights" / "best.pt",
            ]
            
            model_found = False
            for path in possible_paths:
                if path.exists():
                    self.model_path = path
                    model_found = True
                    print(f"✓ Found YOLO model: {path}")
                    break
            
            if not model_found:
                print("⚠️ Warning: YOLO logo model not found. Train the model first using yolo_logo_train_pipeline.py")
                print("   Expected locations:")
                for p in possible_paths:
                    print(f"   - {p}")
                self.model = None
                return
            
            # Load YOLO model
            self.model = YOLO(str(self.model_path))
            self.class_names = self.model.names  # Dictionary: {0: 'class1', 1: 'class2', ...}
            
            print(f"✓ YOLO Logo Detector initialized")
            print(f"  Model: {self.model_path.name}")
            print(f"  Classes: {list(self.class_names.values())}")
            
        except ImportError:
            print("✗ ultralytics not installed. Install with: pip install ultralytics")
            self.model = None
        except Exception as e:
            print(f"✗ Error loading YOLO model: {e}")
            self.model = None
    
    def detect_logos(
        self, 
        image, 
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> Dict[str, Any]:
        """
        Detect logos in IC image using YOLO
        
        Args:
            image: PIL Image, numpy array, or file path
            conf_threshold: Confidence threshold for detections (0-1)
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Dictionary containing:
                - success: bool
                - detections: List of detected logos with bounding boxes
                - best_detection: Highest confidence detection
                - image_with_boxes: Image with drawn bounding boxes
        """
        if not self.model:
            return {
                "success": False,
                "error": "YOLO model not loaded. Train model first.",
                "detections": [],
                "best_detection": None
            }
        
        try:
            # Convert input to format YOLO accepts
            if isinstance(image, str):
                image_path = image
            elif isinstance(image, Image.Image):
                # Save temporarily
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                image.save(temp_file.name)
                image_path = temp_file.name
            elif isinstance(image, np.ndarray):
                # Convert numpy to PIL then save
                import tempfile
                pil_img = Image.fromarray(image)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                pil_img.save(temp_file.name)
                image_path = temp_file.name
            else:
                return {
                    "success": False,
                    "error": "Invalid image format",
                    "detections": []
                }
            
            # Run YOLO inference
            results = self.model.predict(
                source=image_path,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )[0]
            
            # Parse detections
            detections = []
            for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
                x1, y1, x2, y2 = map(int, box.tolist())
                class_id = int(cls)
                class_name = self.class_names[class_id]
                confidence = float(conf)
                
                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "width": x2 - x1,
                        "height": y2 - y1
                    }
                })
            
            # Sort by confidence
            detections.sort(key=lambda d: d["confidence"], reverse=True)
            
            # Get best detection (highest confidence)
            best_detection = detections[0] if detections else None
            
            # Draw boxes on image for visualization
            image_with_boxes = self._draw_detections(image_path, detections)
            
            return {
                "success": True,
                "detections": detections,
                "best_detection": best_detection,
                "total_detections": len(detections),
                "image_with_boxes": image_with_boxes
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Detection failed: {str(e)}",
                "detections": [],
                "best_detection": None
            }
    
    def _draw_detections(self, image_path: str, detections: List[Dict]) -> Optional[np.ndarray]:
        """
        Draw bounding boxes on image
        
        Args:
            image_path: Path to image
            detections: List of detection dictionaries
            
        Returns:
            Image array with drawn boxes or None
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            for det in detections:
                bbox = det["bbox"]
                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                class_name = det["class_name"]
                confidence = det["confidence"]
                
                # Draw rectangle
                color = (0, 255, 0)  # Green
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name} {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            return img
            
        except Exception as e:
            print(f"Error drawing detections: {e}")
            return None
    
    def crop_best_logo(self, image, detection: Dict) -> Optional[Image.Image]:
        """
        Crop the detected logo region from image
        
        Args:
            image: PIL Image or numpy array
            detection: Detection dictionary with bbox
            
        Returns:
            Cropped logo as PIL Image or None
        """
        try:
            # Convert to PIL if needed
            if isinstance(image, np.ndarray):
                pil_img = Image.fromarray(image)
            elif isinstance(image, str):
                pil_img = Image.open(image)
            else:
                pil_img = image
            
            # Extract bbox
            bbox = detection["bbox"]
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            
            # Crop with some padding
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(pil_img.width, x2 + padding)
            y2 = min(pil_img.height, y2 + padding)
            
            cropped = pil_img.crop((x1, y1, x2, y2))
            return cropped
            
        except Exception as e:
            print(f"Error cropping logo: {e}")
            return None
    
    def detect_and_classify(self, image, conf_threshold: float = 0.25) -> Dict[str, Any]:
        """
        One-step detection and classification
        Replaces the old two-step approach (crop then classify)
        
        Args:
            image: Input image (PIL, numpy, or path)
            conf_threshold: Confidence threshold
            
        Returns:
            Dictionary with manufacturer, confidence, and all detection details
        """
        result = self.detect_logos(image, conf_threshold=conf_threshold)
        
        if not result["success"] or not result["best_detection"]:
            return {
                "success": False,
                "manufacturer": None,
                "confidence": 0.0,
                "error": result.get("error", "No logo detected")
            }
        
        best = result["best_detection"]
        
        return {
            "success": True,
            "manufacturer": best["class_name"],
            "confidence": best["confidence"],
            "confidence_percentage": f"{best['confidence'] * 100:.1f}%",
            "bbox": best["bbox"],
            "all_detections": result["detections"],
            "total_detections": result["total_detections"]
        }


# Factory function
def create_yolo_detector():
    """Get YOLO logo detector singleton instance"""
    return YOLOLogoDetector()


# Alias for consistency with other services (ocr_extractor, ic_verifier)
def get_yolo_detector():
    """Get YOLO logo detector singleton instance (alias)"""
    return create_yolo_detector()
