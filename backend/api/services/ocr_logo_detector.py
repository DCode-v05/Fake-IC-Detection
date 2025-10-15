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
    
    # Manufacturer text patterns (case-insensitive matching)
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
            r'\bPHILIPS\b'  # NXP was formerly Philips Semiconductors
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
            r'\bATMEL\b'  # Microchip acquired Atmel
        ],
        "Infineon": [
            r'\bINFINEON\b',
            r'\bIFX\b',
            r'\bSIEMENS\b'  # Infineon was formerly Siemens Semiconductors
        ]
    }
    
    # Part number prefixes for manufacturer inference (last resort fallback)
    PART_NUMBER_PREFIXES = {
        "STMicroelectronics": [
            r'^STM\d',      # STM32F103, STM8S...
            r'^STM',        # General ST prefix
            r'^L\d{3}',     # L6599, L7805 voltage regulators
            r'^M\d{2}[A-Z]', # M24C02, M93C46 EEPROMs
            r'^[6-9]\d{6}', # ST lot/date codes: 6007329, 7123456, etc.
            r'^[6-9]\s*\d{6}' # ST lot codes with space: "6 007329"
        ],
        "Texas Instruments": [
            r'^TPS\d',      # TPS54620 power
            r'^LM\d',       # LM324, LM358 op-amps
            r'^TL\d',       # TL072, TL494
            r'^SN\d',       # SN74HC595
            r'^TMS\d',      # TMS320 DSP
            r'^MSP\d'       # MSP430 MCU
        ],
        "Microchip": [
            r'^PIC\d',      # PIC16F, PIC18F
            r'^ATMEGA\d',   # ATmega328P
            r'^ATTINY\d',   # ATtiny85
            r'^MCP\d',      # MCP23017
            r'^24[LC]'      # 24LC256 EEPROM
        ],
        "NXP Semiconductors": [
            r'^LPC\d',      # LPC1768 ARM
            r'^MK\d',       # MK20DX Kinetis
            r'^PCF\d',      # PCF8574 I/O expander
            r'^TDA\d',      # TDA audio chips
            r'^74[A-Z]+'    # 74HC series logic
        ],
        "ON Semiconductor": [
            r'^NCP\d',      # NCP1117 regulators
            r'^MC\d',       # MC34063 switcher
            r'^CAT\d'       # CAT24C series
        ],
        "Infineon": [
            r'^TLE\d',      # TLE4905 sensors
            r'^IRL\d',      # IRLZ44N MOSFETs
            r'^IR[FG]\d',   # IRF640, IRG4PC40W
            r'^BSC\d'       # BSC series
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
            # Convert to numpy array if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            elif isinstance(image, str):
                image = np.array(Image.open(image))
            
            # Run OCR detection + recognition (without cls parameter)
            result = self.ocr_extractor.ocr.ocr(image)
            
            # Debug: Print result structure
            print(f"📊 OCR Result Type: {type(result)}")
            if result:
                print(f"📊 OCR Result Length: {len(result)}")
                if len(result) > 0:
                    print(f"📊 OCR Result[0] Type: {type(result[0])}")
            
            if not result:
                print("⚠ OCR returned None or empty result")
                return []
            
            # Handle PaddleX OCRResult object
            ocr_result_obj = result[0] if len(result) > 0 else None
            
            if ocr_result_obj is None:
                print("⚠ OCR result[0] is None")
                return []
            
            # Extract text data from OCRResult object
            text_regions = []
            
            # Check if it's a PaddleX OCRResult object (has rec_texts attribute)
            if hasattr(ocr_result_obj, 'rec_texts') or (isinstance(ocr_result_obj, dict) and 'rec_texts' in ocr_result_obj):
                # PaddleX format
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
                
                # Process each detected text
                for i, text in enumerate(rec_texts):
                    if not text:
                        continue
                    
                    confidence = rec_scores[i] if i < len(rec_scores) else 0.0
                    poly = rec_polys[i] if i < len(rec_polys) else None
                    
                    if poly is not None and len(poly) >= 4:
                        # Convert polygon to bounding box
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
            
            # Fallback: Handle old PaddleOCR format (list of [bbox, (text, score)])
            elif isinstance(ocr_result_obj, (list, tuple)):
                ocr_data = ocr_result_obj
                
                for line in ocr_data:
                    if line and len(line) >= 2:
                        try:
                            bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                            text_info = line[1]  # Should be (text, confidence)
                            
                            # Handle different text_info formats
                            if isinstance(text_info, str):
                                # If text_info is just a string (text only, no confidence)
                                text = text_info
                                confidence = 1.0  # Default confidence
                            elif isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                # Normal format: (text, confidence)
                                text = str(text_info[0]) if text_info[0] else ""
                                confidence = float(text_info[1]) if text_info[1] else 0.0
                            elif isinstance(text_info, dict):
                                # Dictionary format
                                text = str(text_info.get('text', ''))
                                confidence = float(text_info.get('confidence', 0.0))
                            else:
                                # Unknown format
                                print(f"⚠ Unexpected text_info type: {type(text_info)}, value: {text_info}")
                                continue
                            
                            # Validate bbox structure
                            if not bbox or len(bbox) < 4:
                                print(f"⚠ Invalid bbox structure: {bbox}")
                                continue
                            
                            # Calculate bounding box (top-left and bottom-right)
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
                                'bbox_coords': bbox  # Original polygon coordinates
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
        
        # Skip empty strings
        if not text_upper:
            return None
        
        for manufacturer, patterns in self.MANUFACTURER_KEYWORDS.items():
            for pattern in patterns:
                try:
                    if re.search(pattern, text_upper, re.IGNORECASE):
                        # Calculate confidence based on pattern specificity
                        # Longer matches = higher confidence
                        # Remove regex special characters for length calculation
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
        
        # Skip very short strings (likely noise)
        if len(text_upper) < 3:
            return None
        
        # Remove common non-alphanumeric characters
        clean_text = re.sub(r'[^A-Z0-9]', '', text_upper)
        
        for manufacturer, prefixes in self.PART_NUMBER_PREFIXES.items():
            for prefix_pattern in prefixes:
                try:
                    if re.match(prefix_pattern, clean_text, re.IGNORECASE):
                        # Lower confidence for part number inference (60-75%)
                        # Longer pattern matches = higher confidence
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
        
        # Extract all detected text fragments
        all_texts = [r.get('text', '').upper().strip() for r in text_regions if r.get('text')]
        all_texts_combined = ' '.join(all_texts)
        
        print(f"🔍 Visual pattern analysis on: {all_texts}")
        
        # STMicroelectronics graphical logo patterns
        # The stylized "ST" logo often gets OCR'd as fragments like "S", "T", or similar shapes
        st_indicators = 0
        
        # Check for exact short matches (likely logo text)
        for text in all_texts:
            # Direct ST matches
            if text in ['S', 'ST', 'T', '5', '5T', 'S1', '51']:
                st_indicators += 2
                print(f"  ✓ ST indicator: exact match '{text}'")
            # OCR often misreads ST as "US", "U5", "05", etc.
            elif text in ['US', 'U5', 'OS', 'O5', 'GS']:
                st_indicators += 2
                print(f"  ✓ ST indicator: OCR misread '{text}' (likely ST)")
            # ST substring in short text
            elif len(text) <= 3 and 'ST' in text:
                st_indicators += 3
                print(f"  ✓ ST indicator: contains 'ST' - '{text}'")
        
        # Check for part numbers starting with common ST prefixes
        for text in all_texts:
            # Check if starts with 6, 7, 8, or 9 followed by digits (ST date/lot codes)
            if re.match(r'^[6-9]\d{5,}', text):
                st_indicators += 1
                print(f"  ✓ ST indicator: potential lot code '{text}'")
        
        # Sometimes the ST logo shape is recognized as single characters
        if len(all_texts) <= 4:
            for text in all_texts:
                if text in ['6', 'C', 'G', 'O', 'E']:
                    st_indicators += 1
                    print(f"  ✓ ST indicator: shape fragment '{text}'")
                    break  # Only count once
            
        if st_indicators >= 2:
            confidence = min(0.85, 0.65 + (st_indicators * 0.05))
            print(f"🎯 Visual pattern match: STMicroelectronics (indicators: {st_indicators}, confidence: {confidence:.2f})")
            return ("STMicroelectronics", confidence)
        
        # Texas Instruments triangle logo patterns
        # Often recognized as "TI", "△", or geometric shapes
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
        
        # NXP logo patterns (often has distinctive "NXP" or angular shapes)
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
        
        # Extract all text regions
        text_regions = self._extract_text_regions(image)
        
        if not text_regions:
            return {
                'success': False,
                'error': 'No text detected in image',
                'manufacturer': None,
                'confidence': 0.0,
                'all_text': []
            }
        
        # Filter by OCR confidence (relaxed threshold for visual patterns)
        valid_regions = [r for r in text_regions if r.get('confidence', 0) >= min_confidence * 0.6]  # 60% of threshold
        
        if not valid_regions:
            return {
                'success': False,
                'error': f'No text detected with confidence >= {min_confidence * 0.6}',
                'manufacturer': None,
                'confidence': 0.0,
                'all_text_regions': text_regions
            }
        
        # STRATEGY 1: Try text-based manufacturer matching
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
                # Combined confidence: OCR confidence * pattern match confidence
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
                # Non-logo text (likely part numbers, date codes, etc.)
                marking_text_regions.append(region)
        
        # STRATEGY 2: If no text match found, try visual pattern analysis
        if not manufacturer_matches:
            print("🔍 No text-based logo match, trying visual pattern analysis...")
            visual_match = self._analyze_visual_patterns(text_regions)
            
            if visual_match:
                manufacturer, visual_confidence = visual_match
                
                # Find the most logo-like region (short text, not part numbers)
                # Prefer short text that could be logo misreads (US, OS, etc.)
                logo_like_regions = []
                for r in text_regions:
                    text = r.get('text', '').upper().strip()
                    # Logo text is typically short (1-3 characters)
                    # Part numbers are longer (5+ characters or contain many digits)
                    if len(text) <= 3 and not re.match(r'^\d{5,}', text):
                        logo_like_regions.append(r)
                
                # Use the best logo-like region, or fall back to highest confidence
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
                
                # Mark short logo-like text as logo regions
                logo_text_regions.extend(logo_like_regions)
                # All other regions (including part numbers) are markings
                marking_text_regions = [r for r in text_regions if r not in logo_like_regions]
                
                print(f"🏷️ Logo regions: {[r.get('text') for r in logo_like_regions]}")
                print(f"📋 Marking regions: {[r.get('text') for r in marking_text_regions]}")
        
        # STRATEGY 3: If still no match, try part number prefix inference
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
                    break  # Take first match
        
        # Get best manufacturer match
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
        # First detect logo to identify logo regions
        logo_result = self.detect_logo_text(image)
        
        if exclude_logo_regions and logo_result['success']:
            # Return only non-logo text
            marking_regions = logo_result.get('marking_regions', [])
        else:
            # Return all text
            marking_regions = logo_result.get('all_text_regions', [])
        
        # Extract text from marking regions
        marking_text = []
        for region in marking_regions:
            marking_text.append({
                'text': region['text'],
                'confidence': region['confidence'],
                'bbox': region['bbox']
            })
        
        # Combine all text
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
        
        # Single OCR pass for ALL text
        text_regions = self._extract_text_regions(image)
        
        if not text_regions:
            return {
                'success': False,
                'error': 'No text detected in image',
                'logo_detection': {'success': False},
                'marking_extraction': {'success': False, 'raw_text': '', 'text_regions': []}
            }
        
        print(f"📊 Detected {len(text_regions)} text regions in single OCR pass")
        
        # Process for logo detection
        logo_result = self._process_logo_detection(text_regions, min_confidence)
        
        # Process for marking extraction (exclude logo regions)
        if logo_result['success']:
            logo_bbox = logo_result.get('bbox')
            # Filter out logo regions from markings
            marking_regions = []
            for region in text_regions:
                # Skip if this region is the logo
                if logo_bbox and region['bbox'] == logo_bbox:
                    continue
                marking_regions.append(region)
        else:
            # No logo found, all regions are markings
            marking_regions = text_regions
        
        # Build marking extraction result
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
            'total_ocr_calls': 1  # Only 1 OCR call instead of 2!
        }
    
    def _process_logo_detection(self, text_regions: List[Dict], min_confidence: float) -> Dict[str, Any]:
        """
        Process text regions to detect manufacturer logo
        (Internal method used by detect_and_extract_combined)
        """
        # Filter low confidence text
        valid_regions = [r for r in text_regions if r.get('confidence', 0) >= min_confidence]
        
        if not valid_regions:
            return {'success': False, 'error': 'No high-confidence text detected'}
        
        manufacturer_matches = []
        
        # Try text matching first
        for region in valid_regions:
            text = region.get('text', '')
            if not text:
                continue
            
            text_match = self._match_manufacturer(text)  # Fixed: was _match_manufacturer_text
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
        
        # If no matches, try part number inference
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
        
        # Get best match
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


# Singleton instance
_ocr_logo_detector = None


def get_ocr_logo_detector(ocr_extractor=None):
    """Get or create OCR logo detector singleton instance"""
    global _ocr_logo_detector
    
    if _ocr_logo_detector is None:
        _ocr_logo_detector = OCRLogoDetector(ocr_extractor)
    
    return _ocr_logo_detector
