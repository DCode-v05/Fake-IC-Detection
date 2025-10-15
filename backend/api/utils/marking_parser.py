"""
IC Marking Parser Utilities
Parse and extract structured information from OCR-extracted text from IC components
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from difflib import SequenceMatcher


class MarkingParser:
    """
    Utility class for parsing IC markings and extracting structured information
    such as part numbers, suffixes, date codes, and manufacturer logos
    """
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean OCR text by removing common OCR artifacts and normalizing
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to uppercase for consistency
        cleaned = text.upper()
        
        # Remove common OCR artifacts
        cleaned = cleaned.replace('|', 'I')  # Pipe to I
        cleaned = cleaned.replace('0', 'O').replace('O', '0')  # Normalize O/0
        
        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())
        
        return cleaned
    
    @staticmethod
    def extract_date_code(text: str) -> Optional[Dict[str, Any]]:
        """
        Extract YYWW date code from text
        
        Format: YYWW where YY=year (2 digits), WW=week (2 digits, 01-52)
        Example: 2145 = Year 2021, Week 45
        
        Args:
            text: Text containing potential date code
            
        Returns:
            Dictionary with year, week, raw_code or None if not found
        """
        # Pattern for YYWW format (4 consecutive digits)
        # YY: 00-99 (representing 2000-2099)
        # WW: 01-53 (week of year)
        pattern = r'\b(\d{2})([0-5]\d)\b'
        
        matches = re.findall(pattern, text)
        
        for match in matches:
            yy = match[0]
            ww = match[1]
            
            # Validate week (01-53)
            week_num = int(ww)
            if 1 <= week_num <= 53:
                # Construct full year (assume 2000s)
                year = 2000 + int(yy)
                
                return {
                    "year": year,
                    "week": week_num,
                    "raw_code": yy + ww,
                    "formatted": f"Week {week_num} of {year}"
                }
        
        return None
    
    @staticmethod
    def fuzzy_match(text: str, target: str, threshold: float = 0.8) -> float:
        """
        Calculate fuzzy string match ratio
        
        Args:
            text: Source text
            target: Target text to match against
            threshold: Minimum ratio to consider a match (0.0-1.0)
            
        Returns:
            Match ratio (0.0-1.0)
        """
        return SequenceMatcher(None, text.upper(), target.upper()).ratio()
    
    @staticmethod
    def find_part_number(text: str, known_parts: List[str]) -> Optional[Dict[str, Any]]:
        """
        Find part number in text using fuzzy matching
        
        Args:
            text: OCR extracted text
            known_parts: List of known part numbers to search for
            
        Returns:
            Dictionary with matched part, confidence, position or None
        """
        best_match = None
        best_ratio = 0.0
        best_position = -1
        
        for part in known_parts:
            # Direct substring match (highest confidence)
            if part.upper() in text.upper():
                position = text.upper().find(part.upper())
                return {
                    "part_number": part,
                    "matched_text": text[position:position+len(part)],
                    "confidence": 1.0,
                    "position": position,
                    "match_type": "exact"
                }
            
            # Fuzzy matching for OCR errors
            ratio = MarkingParser.fuzzy_match(text, part)
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = part
                best_position = 0
        
        # Return fuzzy match if above threshold
        if best_ratio >= 0.75:
            return {
                "part_number": best_match,
                "matched_text": text,
                "confidence": best_ratio,
                "position": best_position,
                "match_type": "fuzzy"
            }
        
        return None
    
    @staticmethod
    def find_suffix(text: str, known_suffixes: List[str]) -> Optional[Dict[str, Any]]:
        """
        Find suffix in text
        
        Args:
            text: OCR extracted text
            known_suffixes: List of known suffixes for the part
            
        Returns:
            Dictionary with matched suffix, confidence or None
        """
        for suffix in known_suffixes:
            # Check for exact match
            if suffix.upper() in text.upper():
                position = text.upper().find(suffix.upper())
                return {
                    "suffix": suffix,
                    "matched_text": text[position:position+len(suffix)],
                    "confidence": 1.0,
                    "match_type": "exact"
                }
            
            # Fuzzy match for suffixes
            ratio = MarkingParser.fuzzy_match(text, suffix)
            if ratio >= 0.8:
                return {
                    "suffix": suffix,
                    "matched_text": text,
                    "confidence": ratio,
                    "match_type": "fuzzy"
                }
        
        return None
    
    @staticmethod
    def extract_country_code(text: str) -> Optional[str]:
        """
        Extract country code from text (e.g., USA, CHN, MYS, DEU)
        
        Args:
            text: OCR text
            
        Returns:
            Country code if found, None otherwise
        """
        # Common country codes on ICs
        country_codes = [
            'USA', 'CHN', 'MYS', 'TWN', 'KOR', 'JPN', 'DEU',
            'PHL', 'THA', 'VNM', 'MLT', 'NLD', 'SGP'
        ]
        
        for code in country_codes:
            if code in text.upper():
                return code
        
        return None
    
    @staticmethod
    def parse_ic_marking(
        ocr_text: str,
        ocr_lines: List[str],
        manufacturer: str,
        known_parts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Parse complete IC marking from OCR results
        
        Args:
            ocr_text: Full concatenated OCR text
            ocr_lines: Individual text lines from OCR
            manufacturer: Detected manufacturer name
            known_parts: List of known parts for this manufacturer
            
        Returns:
            Structured parsing results
        """
        result = {
            "success": False,
            "manufacturer": manufacturer,
            "part_number": None,
            "suffix": None,
            "date_code": None,
            "country_code": None,
            "raw_text": ocr_text,
            "confidence": 0.0
        }
        
        # Clean text
        cleaned_text = MarkingParser.clean_text(ocr_text)
        
        # Extract date code
        date_code = MarkingParser.extract_date_code(cleaned_text)
        if date_code:
            result["date_code"] = date_code
        
        # Extract country code
        country = MarkingParser.extract_country_code(cleaned_text)
        if country:
            result["country_code"] = country
        
        # Find part number
        part_numbers = [part["PartNo"] for part in known_parts]
        part_match = MarkingParser.find_part_number(cleaned_text, part_numbers)
        
        if part_match:
            result["part_number"] = part_match
            
            # Find corresponding part in database
            matched_part = None
            for part in known_parts:
                if part["PartNo"] == part_match["part_number"]:
                    matched_part = part
                    break
            
            # Find suffix
            if matched_part:
                suffix_match = MarkingParser.find_suffix(cleaned_text, matched_part["Suffixes"])
                if suffix_match:
                    result["suffix"] = suffix_match
        
        # Calculate overall confidence
        confidences = []
        if result["part_number"]:
            confidences.append(result["part_number"]["confidence"])
        if result["suffix"]:
            confidences.append(result["suffix"]["confidence"])
        if result["date_code"]:
            confidences.append(0.9)  # Date code found adds confidence
        
        if confidences:
            result["confidence"] = sum(confidences) / len(confidences)
            result["success"] = result["confidence"] >= 0.7
        
        return result
    
    @staticmethod
    def validate_date_code_format(date_code: str, expected_format: str = "YYWW") -> bool:
        """
        Validate date code format
        
        Args:
            date_code: Extracted date code
            expected_format: Expected format (default: YYWW)
            
        Returns:
            True if valid format, False otherwise
        """
        if expected_format == "YYWW":
            # Check if 4 digits
            if not re.match(r'^\d{4}$', date_code):
                return False
            
            # Validate week range
            week = int(date_code[2:4])
            return 1 <= week <= 53
        
        return False
    
    @staticmethod
    def extract_logo_text(ocr_lines: List[str], manufacturer_keywords: List[str]) -> Optional[str]:
        """
        Extract manufacturer logo text from OCR lines
        
        Args:
            ocr_lines: List of OCR text lines
            manufacturer_keywords: Keywords for the manufacturer
            
        Returns:
            Matched logo text or None
        """
        for line in ocr_lines:
            cleaned_line = MarkingParser.clean_text(line)
            for keyword in manufacturer_keywords:
                if keyword.upper() in cleaned_line:
                    return keyword
        
        return None


# Convenience functions
def parse_marking(ocr_result: Dict[str, Any], manufacturer: str, known_parts: List[Dict]) -> Dict[str, Any]:
    """
    Convenience function to parse IC marking from OCR result
    
    Args:
        ocr_result: OCR extraction result dictionary
        manufacturer: Detected manufacturer
        known_parts: Known parts from database
        
    Returns:
        Parsed marking information
    """
    parser = MarkingParser()
    
    return parser.parse_ic_marking(
        ocr_text=ocr_result.get("full_text", ""),
        ocr_lines=ocr_result.get("extracted_text", []),
        manufacturer=manufacturer,
        known_parts=known_parts
    )
