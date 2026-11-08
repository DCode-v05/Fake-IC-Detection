"""
OCR-based Logo Text Detector
Uses PaddleOCR to detect and recognize manufacturer logo text directly from IC images
Replaces YOLO-based visual logo detection with text matching
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import re


class OCRLogoDetector:
    """
    Detects manufacturer logos by finding and matching text in IC images
    Uses PaddleOCR for text detection + recognition
    """
    
    MANUFACTURER_KEYWORDS = {
        "Texas Instruments": [
            r'\bTI\b',
            r'\bTEXAS\s*INSTRUMENTS?\b',
            r'\bT\.I\.?\b'
        ],
        "STMicroelectronics": [
            r'\bST\b',
            r'\bSTM\b',
            r'\bSTMICROELECTRONICS?\b',
            r'\bSTMICRO\b'
        ],
        "NXP Semiconductors": [
            r'\bNXP\b',
            r'\bNXP\s*SEMICONDUCTORS?\b',
            r'\bPHILIPS\b'
        ],
        "ON Semiconductor": [
            r'\bON\b',
            r'\bONSEMI\b',
            r'\bON\s*SEMICONDUCTORS?\b',
            r'\bON\s*SEMI\b'
        ],
        "Microchip": [
            r'\bMICROCHIP\b',
            r'\bMCHP\b',
            r'\bATMEL\b'
        ],
        "Infineon": [
            r'\bINFINEON\b',
            r'\bIFX\b',
            r'\bSIEMENS\b'
        ]
    }
    
    PART_NUMBER_PREFIXES = {
        "STMicroelectronics": [
            r'^STM\d',
            r'^STM',
            r'^L\d{3}',
            r'^M\d{2}[A-Z]',
            r'^[6-9]\d{6}',
            r'^[6-9]\s*\d{6}'
        ],
        "Texas Instruments": [
            r'^TPS\d',
            r'^LM\d',
            r'^TL\d',
            r'^SN\d',
            r'^TMS\d',
            r'^MSP\d'
        ],
        "Microchip": [
            r'^PIC\d',
            r'^ATMEGA\d',
            r'^ATTINY\d',
            r'^MCP\d',
            r'^24[LC]'
        ],
        "NXP Semiconductors": [
            r'^LPC\d',
            r'^MK\d',
            r'^PCF\d',
            r'^TDA\d',
            r'^74[A-Z]+'
        ],
        "ON Semiconductor": [
            r'^NCP\d',
            r'^MC\d',
            r'^CAT\d'
        ],
        "Infineon": [
            r'^TLE\d',
            r'^IRL\d',
            r'^IR[FG]\d',
            r'^BSC\d'
        ]
    }
    
    def __init__(self, ocr_extractor=None):
        """
        Initialize OCR Logo Detector
        
        Args:
            ocr_extractor: Existing OCR extractor instance (to avoid reinitializing PaddleOCR)
        """
        self.ocr_extractor = ocr_extractor
        self._initialized = False
        
        if self.ocr_extractor and hasattr(self.ocr_extractor, 'ocr') and self.ocr_extractor.ocr:
            self._initialized = True
            print("✓ OCR Logo Detector initialized (using shared OCR extractor)")
        else:
            print("✗ OCR Logo Detector: OCR extractor not available")
    
    def _extract_text_regions(self, image) -> List[Dict[str, Any]]:
        """
        Extract all text regions from image using PaddleOCR
        
        Args:
            image: PIL Image or numpy array or file path
            
        Returns:
            List of dictionaries with text, bbox, and confidence
        """
        if not self._initialized:
            return []
        
        try:
            if isinstance(image, Image.Image):
                image = np.array(image)
            elif isinstance(image, str):
                image = np.array(Image.open(image))
            
            result = self.ocr_extractor.ocr.ocr(image)
            
            print(f"📊 OCR Result Type: {type(result)}")
            if result:
                print(f"📊 OCR Result Length: {len(result)}")
                if len(result) > 0:
                    print(f"📊 OCR Result[0] Type: {type(result[0])}")
            
            if not result:
                print("⚠ OCR returned None or empty result")
                return []
            
            ocr_result_obj = result[0] if len(result) > 0 else None
            
            if ocr_result_obj is None:
                print("⚠ OCR result[0] is None")
                return []
            
            text_regions = []
            
            if hasattr(ocr_result_obj, 'rec_texts') or (isinstance(ocr_result_obj, dict) and 'rec_texts' in ocr_result_obj):
                if hasattr(ocr_result_obj, 'rec_texts'):
                    rec_texts = ocr_result_obj.rec_texts
                    rec_scores = ocr_result_obj.rec_scores if hasattr(ocr_result_obj, 'rec_scores') else []
                    rec_polys = ocr_result_obj.rec_polys if hasattr(ocr_result_obj, 'rec_polys') else []
                else:
                    rec_texts = ocr_result_obj.get('rec_texts', [])
                    rec_scores = ocr_result_obj.get('rec_scores', [])
                    rec_polys = ocr_result_obj.get('rec_polys', [])
                
                print(f"📊 Detected Texts: {rec_texts}")
                print(f"📊 Confidence Scores: {rec_scores}")
                
                for i, text in enumerate(rec_texts):
                    if not text:
                        continue
                    
                    confidence = rec_scores[i] if i < len(rec_scores) else 0.0
                    poly = rec_polys[i] if i < len(rec_polys) else None
                    
                    if poly is not None and len(poly) >= 4:
                        x_coords = [point[0] for point in poly]
                        y_coords = [point[1] for point in poly]
                        
                        text_regions.append({
                            'text': str(text),
                            'confidence': float(confidence),
                            'bbox': {
                                'x1': int(min(x_coords)),
                                'y1': int(min(y_coords)),
                                'x2': int(max(x_coords)),
                                'y2': int(max(y_coords))
                            },
                            'bbox_coords': poly.tolist() if hasattr(poly, 'tolist') else poly
                        })
                
                return text_regions
            
            elif isinstance(ocr_result_obj, (list, tuple)):
                ocr_data = ocr_result_obj
                
                for line in ocr_data:
                    if line and len(line) >= 2:
                        try:
                            bbox = line[0]
                            text_info = line[1]
                            
                            if isinstance(text_info, str):
                                text = text_info
                                confidence = 1.0
                            elif isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                text = str(text_info[0]) if text_info[0] else ""
                                confidence = float(text_info[1]) if text_info[1] else 0.0
                            elif isinstance(text_info, dict):
                                text = str(text_info.get('text', ''))
                                confidence = float(text_info.get('confidence', 0.0))
                            else:
                                print(f"⚠ Unexpected text_info type: {type(text_info)}, value: {text_info}")
                                continue
                            
                            if not bbox or len(bbox) < 4:
                                print(f"⚠ Invalid bbox structure: {bbox}")
                                continue
                            
                            x_coords = [point[0] for point in bbox]
                            y_coords = [point[1] for point in bbox]
                            
                            text_regions.append({
                                'text': text,
                                'confidence': confidence,
                                'bbox': {
                                    'x1': int(min(x_coords)),
                                    'y1': int(min(y_coords)),
                                    'x2': int(max(x_coords)),
                                    'y2': int(max(y_coords))
                                },
                                'bbox_coords': bbox
                            })
                        except Exception as line_error:
                            print(f"⚠ Error processing OCR line: {line_error}")
                            continue
            
            return text_regions
            
        except Exception as e:
            import traceback
            print(f"✗ Error extracting text regions: {e}")
            print(traceback.format_exc())
            return []
    
    def _match_manufacturer(self, text: str) -> Optional[Tuple[str, float]]:
        """
        Match text against manufacturer keywords
        
        Args:
            text: Detected text string
            
        Returns:
            Tuple of (manufacturer_name, confidence) or None
        """
        text_upper = text.upper().strip()
        
        if not text_upper:
            return None
        
        for manufacturer, patterns in self.MANUFACTURER_KEYWORDS.items():
            for pattern in patterns:
                try:
                    if re.search(pattern, text_upper, re.IGNORECASE):
                        clean_pattern = re.sub(r'\\[bsw]|\?|\*|\+|\||\[|\]|\(|\)|\{|\}', '', pattern)
                        pattern_length = len(clean_pattern)
                        confidence = min(0.95, 0.75 + (pattern_length / 100))
                        
                        return (manufacturer, confidence)
                except Exception as e:
                    print(f"⚠ Error matching pattern '{pattern}' for {manufacturer}: {e}")
                    continue
        
        return None
    
    def _infer_manufacturer_from_part_number(self, text: str) -> Optional[Tuple[str, float]]:
        """
        Infer manufacturer from part number prefixes (fallback when no logo text found)
        
        Args:
            text: Detected text string (potential part number)
            
        Returns:
            Tuple of (manufacturer_name, confidence) or None
        """
        text_upper = text.upper().strip()
        
        if len(text_upper) < 3:
            return None
        
        clean_text = re.sub(r'[^A-Z0-9]', '', text_upper)
        
        for manufacturer, prefixes in self.PART_NUMBER_PREFIXES.items():
            for prefix_pattern in prefixes:
                try:
                    if re.match(prefix_pattern, clean_text, re.IGNORECASE):
                        clean_pattern = re.sub(r'[\^\$\[\]\(\)\{\}\?\*\+\|\\]', '', prefix_pattern)
                        pattern_length = len(clean_pattern)
                        confidence = min(0.75, 0.60 + (pattern_length / 50))
                        
                        print(f"🔍 Inferred {manufacturer} from part number pattern: '{clean_text}' matches '{prefix_pattern}'")
                        return (manufacturer, confidence)
                except Exception as e:
                    print(f"⚠ Error matching part number pattern '{prefix_pattern}' for {manufacturer}: {e}")
                    continue
        
        return None
    
    def _analyze_visual_patterns(self, text_regions: List[Dict[str, Any]]) -> Optional[Tuple[str, float]]:
        """
        Analyze visual patterns in OCR results to identify graphical logos
        Looks for characteristic combinations and spatial relationships
        
        Args:
            text_regions: List of detected text regions with positions
            
        Returns:
            Tuple of (manufacturer_name, confidence) or None
        """
        if not text_regions or len(text_regions) == 0:
            return None
        
        all_texts = [r.get('text', '').upper().strip() for r in text_regions if r.get('text')]
        all_texts_combined = ' '.join(all_texts)
        
        print(f"🔍 Visual pattern analysis on: {all_texts}")
        
        st_indicators = 0
        
        for text in all_texts:
            if text in ['S', 'ST', 'T', '5', '5T', 'S1', '51']:
                st_indicators += 2
                print(f"  ✓ ST indicator: exact match '{text}'")
            elif text in ['US', 'U5', 'OS', 'O5', 'GS']:
                st_indicators += 2
                print(f"  ✓ ST indicator: OCR misread '{text}' (likely ST)")
            elif len(text) <= 3 and 'ST' in text:
                st_indicators += 3
                print(f"  ✓ ST indicator: contains 'ST' - '{text}'")
        
        for text in all_texts:
            if re.match(r'^[6-9]\d{5,}', text):
                st_indicators += 1
                print(f"  ✓ ST indicator: potential lot code '{text}'")
        
        if len(all_texts) <= 4:
            for text in all_texts:
                if text in ['6', 'C', 'G', 'O', 'E']:
                    st_indicators += 1
                    print(f"  ✓ ST indicator: shape fragment '{text}'")
                    break
            
        if st_indicators >= 2:
            confidence = min(0.85, 0.65 + (st_indicators * 0.05))
            print(f"🎯 Visual pattern match: STMicroelectronics (indicators: {st_indicators}, confidence: {confidence:.2f})")
            return ("STMicroelectronics", confidence)
        
        ti_indicators = 0
        for text in all_texts:
            if text in ['TI', 'T', 'I', 'T.I', 'T.', '.I', '1I', 'T1']:
                ti_indicators += 2
                print(f"  ✓ TI indicator: text match '{text}'")
            if text in ['△', '▲', 'A', '^']:
                ti_indicators += 2
                print(f"  ✓ TI indicator: triangle shape '{text}'")
            
        if ti_indicators >= 2:
            confidence = min(0.80, 0.60 + (ti_indicators * 0.1))
            print(f"🎯 Visual pattern match: Texas Instruments (indicators: {ti_indicators}, confidence: {confidence:.2f})")
            return ("Texas Instruments", confidence)
        
        for text in all_texts:
            if text in ['NXP', 'NX', 'XP']:
                print(f"🎯 Visual pattern match: NXP Semiconductors (text: '{text}')")
                return ("NXP Semiconductors", 0.75)
        
        print(f"  ✗ No visual pattern matched (ST indicators: {st_indicators}, TI indicators: {ti_indicators})")
        return None
    
    def detect_logo_text(self, image, min_confidence: float = 0.5) -> Dict[str, Any]:
        """
        Detect manufacturer logo by finding and matching text in image
        Uses multiple strategies:
        1. Text-based logo detection (e.g., "STMicroelectronics", "TI")
        2. Visual pattern analysis (for graphical logos like ST symbol)
        3. Part number prefix inference (fallback)
        
        Args:
            image: PIL Image, numpy array, or file path
            min_confidence: Minimum OCR confidence for text detection
            
        Returns:
            Dictionary with detection results
        """
        if not self._initialized:
            return {
                'success': False,
                'error': 'OCR Logo Detector not initialized',
                'manufacturer': None,
                'confidence': 0.0
            }
        
        text_regions = self._extract_text_regions(image)
        
        if not text_regions:
            return {
                'success': False,
                'error': 'No text detected in image',
                'manufacturer': None,
                'confidence': 0.0,
                'all_text': []
            }
        
        valid_regions = [r for r in text_regions if r.get('confidence', 0) >= min_confidence * 0.6]
        
        if not valid_regions:
            return {
                'success': False,
                'error': f'No text detected with confidence >= {min_confidence * 0.6}',
                'manufacturer': None,
                'confidence': 0.0,
                'all_text_regions': text_regions
            }
        
        manufacturer_matches = []
        logo_text_regions = []
        marking_text_regions = []
        
        for region in valid_regions:
            text = region.get('text', '')
            if not text or not isinstance(text, str):
                continue
                
            match = self._match_manufacturer(text)
            
            if match:
                manufacturer, match_confidence = match
                combined_confidence = region['confidence'] * match_confidence
                
                manufacturer_matches.append({
                    'manufacturer': manufacturer,
                    'confidence': combined_confidence,
                    'text': text,
                    'bbox': region['bbox'],
                    'ocr_confidence': region['confidence'],
                    'detection_method': 'text_match'
                })
                logo_text_regions.append(region)
            else:
                marking_text_regions.append(region)
        
        if not manufacturer_matches:
            print("🔍 No text-based logo match, trying visual pattern analysis...")
            visual_match = self._analyze_visual_patterns(text_regions)
            
            if visual_match:
                manufacturer, visual_confidence = visual_match
                
                logo_like_regions = []
                for r in text_regions:
                    text = r.get('text', '').upper().strip()
                    if len(text) <= 3 and not re.match(r'^\d{5,}', text):
                        logo_like_regions.append(r)
                
                if logo_like_regions:
                    best_region = max(logo_like_regions, key=lambda r: r.get('confidence', 0))
                else:
                    best_region = max(text_regions, key=lambda r: r.get('confidence', 0))
                
                manufacturer_matches.append({
                    'manufacturer': manufacturer,
                    'confidence': visual_confidence,
                    'text': f"Visual pattern: {', '.join([r.get('text', '') for r in text_regions[:3]])}",
                    'bbox': best_region['bbox'],
                    'ocr_confidence': best_region['confidence'],
                    'detection_method': 'visual_pattern'
                })
                
                logo_text_regions.extend(logo_like_regions)
                marking_text_regions = [r for r in text_regions if r not in logo_like_regions]
                
                print(f"🏷️ Logo regions: {[r.get('text') for r in logo_like_regions]}")
                print(f"📋 Marking regions: {[r.get('text') for r in marking_text_regions]}")
        
        if not manufacturer_matches:
            print("🔍 No visual pattern match, trying part number inference...")
            
            for region in valid_regions:
                text = region.get('text', '')
                if not text:
                    continue
                
                part_match = self._infer_manufacturer_from_part_number(text)
                
                if part_match:
                    manufacturer, part_confidence = part_match
                    combined_confidence = region['confidence'] * part_confidence
                    
                    manufacturer_matches.append({
                        'manufacturer': manufacturer,
                        'confidence': combined_confidence,
                        'text': text,
                        'bbox': region['bbox'],
                        'ocr_confidence': region['confidence'],
                        'detection_method': 'part_number_inference'
                    })
                    logo_text_regions.append(region)
                    marking_text_regions = [r for r in text_regions if r != region]
                    break
        
        if manufacturer_matches:
            best_match = max(manufacturer_matches, key=lambda x: x['confidence'])
            
            detection_method_label = {
                'text_match': 'Logo Text',
                'visual_pattern': 'Visual Pattern Analysis',
                'part_number_inference': 'Part Number Prefix'
            }.get(best_match['detection_method'], 'Unknown')
            
            print(f"✓ Manufacturer detected: {best_match['manufacturer']} via {detection_method_label} (confidence: {best_match['confidence']:.2f})")
            
            return {
                'success': True,
                'manufacturer': best_match['manufacturer'],
                'confidence': best_match['confidence'],
                'confidence_percentage': f"{best_match['confidence'] * 100:.1f}%",
                'logo_text': best_match['text'],
                'bbox': best_match['bbox'],
                'detection_method': detection_method_label,
                'all_matches': manufacturer_matches,
                'logo_regions': logo_text_regions,
                'marking_regions': marking_text_regions,
                'all_text_regions': text_regions
            }
        else:
            print("✗ No manufacturer detected via any method")
            return {
                'success': False,
                'error': 'No manufacturer logo text found',
                'manufacturer': None,
                'confidence': 0.0,
                'marking_regions': marking_text_regions,
                'all_text_regions': text_regions
            }
    
    def extract_ic_markings(self, image, exclude_logo_regions: bool = True) -> Dict[str, Any]:
        """
        Extract IC markings (part numbers, date codes, etc.) from image
        Excludes logo text regions if specified
        
        Args:
            image: PIL Image, numpy array, or file path
            exclude_logo_regions: If True, exclude detected logo text from results
            
        Returns:
            Dictionary with IC marking text
        """
        logo_result = self.detect_logo_text(image)
        
        if exclude_logo_regions and logo_result['success']:
            marking_regions = logo_result.get('marking_regions', [])
        else:
            marking_regions = logo_result.get('all_text_regions', [])
        
        marking_text = []
        for region in marking_regions:
            marking_text.append({
                'text': region['text'],
                'confidence': region['confidence'],
                'bbox': region['bbox']
            })
        
        full_text = ' '.join([r['text'] for r in marking_regions])
        
        return {
            'success': len(marking_regions) > 0,
            'raw_text': full_text,
            'text_regions': marking_text,
            'total_regions': len(marking_regions)
        }
    
    def detect_and_extract_combined(self, image, min_confidence: float = 0.5) -> Dict[str, Any]:
        """
        OPTIMIZED: Combined logo detection and marking extraction in a SINGLE OCR pass
        This is MUCH FASTER than calling detect_logo_text() and extract_ic_markings() separately
        
        Args:
            image: PIL Image or numpy array
            min_confidence: Minimum confidence threshold for manufacturer detection
            
        Returns:
            Dictionary containing:
                - logo_detection: Logo detection result (manufacturer, confidence, bbox, etc.)
                - marking_extraction: IC marking text extraction result
                - text_regions: All detected text regions (shared between both)
        """
        print("⚡ FAST MODE: Combined logo detection + marking extraction (single OCR pass)")
        
        text_regions = self._extract_text_regions(image)
        
        if not text_regions:
            return {
                'success': False,
                'error': 'No text detected in image',
                'logo_detection': {'success': False},
                'marking_extraction': {'success': False, 'raw_text': '', 'text_regions': []}
            }
        
        print(f"📊 Detected {len(text_regions)} text regions in single OCR pass")
        
        logo_result = self._process_logo_detection(text_regions, min_confidence)
        
        if logo_result['success']:
            logo_bbox = logo_result.get('bbox')
            marking_regions = []
            for region in text_regions:
                if logo_bbox and region['bbox'] == logo_bbox:
                    continue
                marking_regions.append(region)
        else:
            marking_regions = text_regions
        
        marking_text = []
        for region in marking_regions:
            marking_text.append({
                'text': region['text'],
                'confidence': region['confidence'],
                'bbox': region['bbox']
            })
        
        full_text = ' '.join([r['text'] for r in marking_regions])
        
        marking_result = {
            'success': len(marking_regions) > 0,
            'raw_text': full_text,
            'text_regions': marking_text,
            'total_regions': len(marking_regions)
        }
        
        print(f"✓ Combined result: Logo={logo_result['manufacturer'] if logo_result['success'] else 'None'}, Markings={len(marking_regions)} regions")
        
        return {
            'success': True,
            'logo_detection': logo_result,
            'marking_extraction': marking_result,
            'text_regions': text_regions,
            'total_ocr_calls': 1
        }
    
    def _process_logo_detection(self, text_regions: List[Dict], min_confidence: float) -> Dict[str, Any]:
        """
        Process text regions to detect manufacturer logo
        (Internal method used by detect_and_extract_combined)
        """
        valid_regions = [r for r in text_regions if r.get('confidence', 0) >= min_confidence]
        
        if not valid_regions:
            return {'success': False, 'error': 'No high-confidence text detected'}
        
        manufacturer_matches = []
        
        for region in valid_regions:
            text = region.get('text', '')
            if not text:
                continue
            
            text_match = self._match_manufacturer(text)
            if text_match:
                manufacturer, match_confidence = text_match
                combined_confidence = region['confidence'] * match_confidence
                
                manufacturer_matches.append({
                    'manufacturer': manufacturer,
                    'confidence': combined_confidence,
                    'text': text,
                    'bbox': region['bbox'],
                    'ocr_confidence': region['confidence'],
                    'detection_method': 'text_pattern'
                })
        
        if not manufacturer_matches:
            for region in valid_regions:
                text = region.get('text', '')
                if not text:
                    continue
                
                part_match = self._infer_manufacturer_from_part_number(text)
                if part_match:
                    manufacturer, part_confidence = part_match
                    combined_confidence = region['confidence'] * part_confidence
                    
                    manufacturer_matches.append({
                        'manufacturer': manufacturer,
                        'confidence': combined_confidence,
                        'text': text,
                        'bbox': region['bbox'],
                        'ocr_confidence': region['confidence'],
                        'detection_method': 'part_number'
                    })
        
        if not manufacturer_matches:
            return {'success': False, 'error': 'No manufacturer detected'}
        
        best_match = max(manufacturer_matches, key=lambda x: x['confidence'])
        
        return {
            'success': True,
            'manufacturer': best_match['manufacturer'],
            'confidence': best_match['confidence'],
            'confidence_percentage': best_match['confidence'] * 100,
            'logo_text': best_match['text'],
            'bbox': best_match['bbox'],
            'all_matches': manufacturer_matches,
            'detection_method': best_match['detection_method']
        }


_ocr_logo_detector = None


def get_ocr_logo_detector(ocr_extractor=None):
    """Get or create OCR logo detector singleton instance"""
    global _ocr_logo_detector
    
    if _ocr_logo_detector is None:
        _ocr_logo_detector = OCRLogoDetector(ocr_extractor)
    
    return _ocr_logo_detector
