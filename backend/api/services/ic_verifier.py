"""
IC Verification Service
Validates IC markings against database to determine authenticity
Falls back to Nexar web database if part not found locally
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from api.services.nexar_service import get_nexar_service


class ICVerifier:
    """
    Service for verifying IC component authenticity by validating markings
    against a database of known genuine parts
    """
    
    _instance = None
    _database_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'database',
        'ic_parts_database.json'
    )
    
    def __new__(cls):
        """Singleton pattern to load database once"""
        if cls._instance is None:
            cls._instance = super(ICVerifier, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize and load IC parts database"""
        if self._initialized:
            return
        
        self.database = None
        self.load_database()
        self._initialized = True
    
    def load_database(self) -> bool:
        """
        Load IC parts database from JSON file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(self._database_path):
                print(f"✗ Database not found at: {self._database_path}")
                return False
            
            with open(self._database_path, 'r', encoding='utf-8') as f:
                self.database = json.load(f)
            
            print(f"✓ IC Database loaded: {self.database['metadata']['total_parts']} parts across {self.database['metadata']['total_manufacturers']} manufacturers")
            return True
            
        except Exception as e:
            print(f"✗ Error loading database: {e}")
            return False
    
    def get_manufacturer_parts(self, manufacturer: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get all parts for a specific manufacturer
        
        Args:
            manufacturer: Manufacturer name
            
        Returns:
            List of parts or None if manufacturer not found
        """
        if not self.database:
            return None
        
        manufacturers = self.database.get("manufacturers", {})
        
        # Try exact match first
        if manufacturer in manufacturers:
            return manufacturers[manufacturer].get("parts", [])
        
        # Try case-insensitive match
        for mfr_name, mfr_data in manufacturers.items():
            if mfr_name.lower() == manufacturer.lower():
                return mfr_data.get("parts", [])
        
        return None
    
    def validate_part_number(
        self,
        part_number: str,
        manufacturer: str
    ) -> Dict[str, Any]:
        """
        Validate if part number exists for manufacturer
        First checks local database, then falls back to Nexar if not found
        
        Args:
            part_number: IC part number to validate
            manufacturer: Manufacturer name
            
        Returns:
            Validation result with matched part details
        """
        parts = self.get_manufacturer_parts(manufacturer)
        
        # Try local database first
        if parts:
            # Check each part
            for part in parts:
                if part["PartNo"].upper() == part_number.upper():
                    return {
                        "valid": True,
                        "source": "local_database",
                        "matched_part": part,
                        "part_number": part["PartNo"],
                        "description": part.get("Description", ""),
                        "package": part.get("Package", [])
                    }
        
        # Not found in local database - try Nexar as fallback
        print(f"🌐 Part {part_number} not found locally, checking Nexar database...")
        nexar_service = get_nexar_service()
        nexar_result = nexar_service.verify_part(part_number, manufacturer)
        
        if nexar_result.get('verified'):
            part_info = nexar_result.get('part_info', {})
            return {
                "valid": True,
                "source": "nexar_web",
                "matched_part": part_info,
                "part_number": part_info.get('part_number', part_number),
                "description": part_info.get('description', ''),
                "package": [],
                "web_verified": True,
                "in_stock": part_info.get('in_stock', False),
                "availability": part_info.get('total_availability', 0),
                "sellers": part_info.get('seller_count', 0)
            }
        
        # Not found in either database
        return {
            "valid": False,
            "source": "not_found",
            "reason": f"Part number '{part_number}' not found in local database or Nexar web database for {manufacturer}"
        }
    
    def validate_suffix(
        self,
        suffix: str,
        part_number: str,
        manufacturer: str
    ) -> Dict[str, Any]:
        """
        Validate if suffix is valid for the part number
        
        Args:
            suffix: Suffix to validate
            part_number: Part number
            manufacturer: Manufacturer name
            
        Returns:
            Validation result
        """
        parts = self.get_manufacturer_parts(manufacturer)
        
        if not parts:
            return {
                "valid": False,
                "reason": "Manufacturer not found"
            }
        
        # Find the part
        matched_part = None
        for part in parts:
            if part["PartNo"].upper() == part_number.upper():
                matched_part = part
                break
        
        if not matched_part:
            return {
                "valid": False,
                "reason": "Part number not found"
            }
        
        # Check suffix
        valid_suffixes = matched_part.get("Suffixes", [])
        
        for valid_suffix in valid_suffixes:
            if valid_suffix.upper() == suffix.upper():
                return {
                    "valid": True,
                    "matched_suffix": valid_suffix,
                    "all_valid_suffixes": valid_suffixes
                }
        
        return {
            "valid": False,
            "reason": f"Suffix '{suffix}' not valid. Valid suffixes: {', '.join(valid_suffixes)}",
            "all_valid_suffixes": valid_suffixes
        }
    
    def validate_date_code(
        self,
        date_code: Dict[str, Any],
        part_number: str,
        manufacturer: str
    ) -> Dict[str, Any]:
        """
        Validate date code format and reasonableness
        
        Args:
            date_code: Date code dictionary with year, week
            part_number: Part number
            manufacturer: Manufacturer name
            
        Returns:
            Validation result
        """
        if not date_code:
            return {
                "valid": False,
                "reason": "No date code provided"
            }
        
        year = date_code.get("year")
        week = date_code.get("week")
        
        # Check format validity
        if not year or not week:
            return {
                "valid": False,
                "reason": "Invalid date code format"
            }
        
        # Check if date is reasonable (not in future)
        current_year = datetime.now().year
        current_week = datetime.now().isocalendar()[1]
        
        if year > current_year:
            return {
                "valid": False,
                "reason": f"Date code year {year} is in the future",
                "suspicious": True
            }
        
        if year == current_year and week > current_week:
            return {
                "valid": False,
                "reason": f"Date code week {week} is in the future",
                "suspicious": True
            }
        
        # Check if date is too old (ICs older than 20 years are suspicious)
        if year < current_year - 20:
            return {
                "valid": True,
                "reason": f"Date code from {year} is very old (possibly remarked)",
                "warning": True,
                "age_years": current_year - year
            }
        
        # Week range validation
        if not (1 <= week <= 53):
            return {
                "valid": False,
                "reason": f"Invalid week number: {week}",
                "suspicious": True
            }
        
        return {
            "valid": True,
            "year": year,
            "week": week,
            "formatted": date_code.get("formatted"),
            "age_years": current_year - year
        }
    
    def validate_country_code(
        self,
        country_code: str,
        part_number: str,
        manufacturer: str
    ) -> Dict[str, Any]:
        """
        Validate country code against expected manufacturing locations
        
        Args:
            country_code: Country code from IC marking
            part_number: Part number
            manufacturer: Manufacturer name
            
        Returns:
            Validation result
        """
        parts = self.get_manufacturer_parts(manufacturer)
        
        if not parts:
            return {
                "valid": False,
                "reason": "Manufacturer not found"
            }
        
        # Find the part
        matched_part = None
        for part in parts:
            if part["PartNo"].upper() == part_number.upper():
                matched_part = part
                break
        
        if not matched_part:
            return {
                "valid": False,
                "reason": "Part number not found"
            }
        
        # Check country codes
        valid_countries = matched_part.get("AdditionalMarkings", {}).get("CountryCode", [])
        
        if country_code in valid_countries:
            return {
                "valid": True,
                "matched_country": country_code,
                "all_valid_countries": valid_countries
            }
        
        return {
            "valid": False,
            "reason": f"Country code '{country_code}' not expected. Valid countries: {', '.join(valid_countries)}",
            "all_valid_countries": valid_countries,
            "suspicious": True
        }
    
    def comprehensive_verification(
        self,
        parsed_marking: Dict[str, Any],
        manufacturer: str,
        logo_confidence: float
    ) -> Dict[str, Any]:
        """
        Perform comprehensive IC verification
        
        Args:
            parsed_marking: Parsed IC marking information
            manufacturer: Detected manufacturer
            logo_confidence: Confidence of logo detection
            
        Returns:
            Comprehensive verification result with authenticity assessment
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "manufacturer": manufacturer,
            "logo_confidence": logo_confidence,
            "verification_status": "unknown",
            "authenticity": "fake",  # Default to fake until proven genuine
            "confidence_score": 0.0,
            "validations": {},
            "warnings": [],
            "errors": [],
            "details": {}
        }
        
        # Check if parsing was successful
        if not parsed_marking.get("success"):
            result["verification_status"] = "failed"
            result["errors"].append("Failed to parse IC markings")
            result["authenticity"] = "fake"  # Can't parse = Fake
            
            # Try to extract any potential part numbers from raw text for Nexar lookup
            raw_text = parsed_marking.get("raw_text", "")
            if raw_text:
                print(f"🔍 Attempting Nexar lookup with raw text: '{raw_text}'")
                # Try to find part number patterns in raw text
                import re
                # Look for common part number patterns (alphanumeric sequences of 5+ chars)
                potential_parts = re.findall(r'\b[A-Z0-9]{5,}\b', raw_text.upper())
                
                if potential_parts:
                    print(f"🔍 Found potential part numbers: {potential_parts}")
                    # Try the first potential part number with Nexar
                    nexar_service = get_nexar_service()
                    for part_candidate in potential_parts:
                        nexar_result = nexar_service.verify_part(part_candidate, manufacturer)
                        if nexar_result.get('verified'):
                            print(f"✓ Found on Nexar: {part_candidate}")
                            part_info = nexar_result.get('part_info', {})
                            result["validations"]["part_number"] = {
                                "valid": True,
                                "source": "nexar_web",
                                "matched_part": part_info,
                                "part_number": part_info.get('part_number', part_candidate),
                                "description": part_info.get('description', ''),
                                "web_verified": True,
                                "in_stock": part_info.get('in_stock', False),
                                "availability": part_info.get('total_availability', 0),
                                "sellers": part_info.get('seller_count', 0)
                            }
                            result["details"]["part_number"] = part_candidate
                            result["details"]["verified_via"] = "Nexar Web Database (fallback)"
                            result["details"]["web_availability"] = part_info.get('total_availability', 0)
                            result["details"]["in_stock"] = part_info.get('in_stock', False)
                            result["details"]["sellers"] = part_info.get('seller_count', 0)
                            result["verification_status"] = "passed"
                            result["authenticity"] = "genuine"  # Found in database = Genuine
                            result["confidence_score"] = 0.70  # Meets minimum threshold
                            result["warnings"].append(
                                f"Part {part_candidate} verified in Nexar database - {part_info.get('seller_count', 0)} authorized sellers"
                            )
                            result["warnings"].append("Marking parsing failed, but part verified in web database")
                            result["errors"] = []  # Clear error since we found it
                            return result
            
            # If Nexar lookup failed but we have manufacturer detection and OCR text
            # If part numbers were extracted but not found in ANY database = FAKE
            if manufacturer and logo_confidence > 0:
                # Part not found in local DB or Nexar = FAKE
                result["verification_status"] = "failed"
                result["authenticity"] = "fake"  # Not in database = Fake
                result["confidence_score"] = 0.0  # No verification = 0% confidence
                result["details"]["detected_manufacturer"] = manufacturer
                result["details"]["manufacturer_confidence"] = logo_confidence
                result["details"]["extracted_text"] = raw_text
                result["details"]["detection_method"] = "Logo and OCR"
                
                # Add extracted text info
                if potential_parts:
                    result["details"]["potential_parts"] = potential_parts
                    result["warnings"].append(f"Potential part numbers detected: {', '.join(potential_parts)}")
                    result["errors"].append(
                        f"⚠️ FAKE IC DETECTED: Part numbers {', '.join(potential_parts)} NOT FOUND in any manufacturer database"
                    )
                else:
                    result["errors"].append("⚠️ FAKE IC DETECTED: No valid part numbers found")
                
                result["errors"].append(
                    f"Verification FAILED: {manufacturer} logo detected but markings do not match any genuine parts"
                )
                
                return result
            
            return result
        
        scores = []
        
        # Validate part number
        part_info = parsed_marking.get("part_number")
        if part_info:
            part_validation = self.validate_part_number(
                part_info["part_number"],
                manufacturer
            )
            result["validations"]["part_number"] = part_validation
            result["details"]["part_number"] = part_info["part_number"]
            
            # Add web verification info if from Nexar
            if part_validation.get("source") == "nexar_web":
                result["details"]["verified_via"] = "Nexar Web Database"
                result["details"]["web_availability"] = part_validation.get("availability", 0)
                result["details"]["in_stock"] = part_validation.get("in_stock", False)
                result["details"]["sellers"] = part_validation.get("sellers", 0)
                result["warnings"].append(
                    f"Part verified via Nexar web database - {part_validation.get('sellers', 0)} sellers found"
                )
            elif part_validation.get("source") == "local_database":
                result["details"]["verified_via"] = "Local Database"
            
            if part_validation["valid"]:
                scores.append(part_info["confidence"])
            else:
                result["errors"].append(part_validation["reason"])
        else:
            result["errors"].append("Part number not detected")
        
        # Validate suffix
        suffix_info = parsed_marking.get("suffix")
        if suffix_info and part_info:
            suffix_validation = self.validate_suffix(
                suffix_info["suffix"],
                part_info["part_number"],
                manufacturer
            )
            result["validations"]["suffix"] = suffix_validation
            result["details"]["suffix"] = suffix_info["suffix"]
            
            if suffix_validation["valid"]:
                scores.append(suffix_info["confidence"])
            else:
                result["warnings"].append(suffix_validation["reason"])
        
        # Validate date code
        date_code = parsed_marking.get("date_code")
        if date_code and part_info:
            date_validation = self.validate_date_code(
                date_code,
                part_info["part_number"],
                manufacturer
            )
            result["validations"]["date_code"] = date_validation
            result["details"]["date_code"] = date_code.get("formatted")
            
            if date_validation["valid"]:
                scores.append(0.8)
                if date_validation.get("warning"):
                    result["warnings"].append(date_validation["reason"])
            else:
                result["errors"].append(date_validation["reason"])
        
        # Validate country code
        country_code = parsed_marking.get("country_code")
        if country_code and part_info:
            country_validation = self.validate_country_code(
                country_code,
                part_info["part_number"],
                manufacturer
            )
            result["validations"]["country_code"] = country_validation
            result["details"]["country_code"] = country_code
            
            if country_validation["valid"]:
                scores.append(0.9)
            else:
                result["warnings"].append(country_validation["reason"])
        
        # Calculate confidence score
        if scores:
            # Include logo detection confidence
            scores.append(logo_confidence)
            result["confidence_score"] = sum(scores) / len(scores)
        
        # Determine authenticity - ONLY TWO OPTIONS: Genuine or Fake
        if result["errors"]:
            # Any errors = Fake IC
            result["authenticity"] = "fake"
            result["verification_status"] = "failed"
        elif result["confidence_score"] >= 0.70:
            # High confidence = Genuine IC
            result["authenticity"] = "genuine"
            result["verification_status"] = "passed"
        else:
            # Below threshold = Fake IC (better safe than sorry)
            result["authenticity"] = "fake"
            result["verification_status"] = "failed"
            if not result["errors"]:
                result["errors"].append(f"Confidence score ({result['confidence_score']*100:.1f}%) below threshold for genuine verification")
        
        return result


# Convenience function
def get_verifier():
    """Get IC verifier singleton instance"""
    return ICVerifier()
