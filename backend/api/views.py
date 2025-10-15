"""
API Views for IC Marking Detection
Handles HTTP requests for image upload and manufacturer prediction
Uses PaddleOCR for both logo text detection and IC marking extraction
"""

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import traceback
import numpy as np

from api.services.ocr_logo_detector import get_ocr_logo_detector
from api.services.ocr_extractor import get_ocr_extractor
from api.services.ic_verifier import get_verifier
from api.utils.image_utils import (
    save_uploaded_image, 
    load_image_from_file,
    validate_image,
    image_to_base64
)
from api.utils.database import create_database
from api.utils.marking_parser import parse_marking


def validate_user_expectations(
    expected_manufacturer: str,
    expected_part_number: str,
    detected_manufacturer: str,
    parsed_marking: dict,
    ocr_result: dict
) -> dict:
    """
    Validate user's expected values against detected values
    
    Args:
        expected_manufacturer: Manufacturer name provided by user
        expected_part_number: Part number provided by user
        detected_manufacturer: Manufacturer detected from image
        parsed_marking: Parsed marking data
        ocr_result: OCR extraction result
        
    Returns:
        Validation result with match status and details
    """
    result = {
        'manufacturer_match': False,
        'part_number_match': False,
        'matches_expectations': False,
        'message': '',
        'details': {}
    }
    
    # Validate manufacturer
    if expected_manufacturer:
        expected_mfr = expected_manufacturer.strip().upper()
        detected_mfr = detected_manufacturer.strip().upper()
        
        # Check for exact match or substring match
        if expected_mfr == detected_mfr or expected_mfr in detected_mfr or detected_mfr in expected_mfr:
            result['manufacturer_match'] = True
            result['details']['manufacturer'] = f"✓ Match: Expected '{expected_manufacturer}', Detected '{detected_manufacturer}'"
        else:
            result['details']['manufacturer'] = f"✗ Mismatch: Expected '{expected_manufacturer}', but detected '{detected_manufacturer}'"
    
    # Validate part number
    if expected_part_number:
        expected_part = expected_part_number.strip().upper().replace(' ', '').replace('-', '')
        
        # Extract detected part number from parsed marking
        detected_part = None
        if parsed_marking and parsed_marking.get('success'):
            part_info = parsed_marking.get('part_number')
            if part_info and isinstance(part_info, dict):
                detected_part = part_info.get('part_number', '').strip().upper().replace(' ', '').replace('-', '')
        
        # Also check raw OCR text
        ocr_text = ocr_result.get('full_text', '').upper().replace(' ', '').replace('-', '')
        
        if detected_part and (expected_part == detected_part or expected_part in detected_part or detected_part in expected_part):
            result['part_number_match'] = True
            result['details']['part_number'] = f"✓ Match: Expected '{expected_part_number}', Found '{detected_part}'"
        elif expected_part in ocr_text:
            result['part_number_match'] = True
            result['details']['part_number'] = f"✓ Match: Expected '{expected_part_number}' found in OCR text"
        else:
            result['details']['part_number'] = f"✗ Mismatch: Expected '{expected_part_number}', but found '{detected_part or 'None'}'"
    
    # Determine overall match
    if expected_manufacturer and expected_part_number:
        # Both provided - both must match
        result['matches_expectations'] = result['manufacturer_match'] and result['part_number_match']
        if result['matches_expectations']:
            result['message'] = f"✓ IC matches user expectations: {expected_manufacturer} {expected_part_number}"
        else:
            mismatches = []
            if not result['manufacturer_match']:
                mismatches.append(f"manufacturer (expected {expected_manufacturer})")
            if not result['part_number_match']:
                mismatches.append(f"part number (expected {expected_part_number})")
            result['message'] = f"✗ IC does NOT match user expectations - Mismatch in {', '.join(mismatches)}"
    elif expected_manufacturer:
        # Only manufacturer provided
        result['matches_expectations'] = result['manufacturer_match']
        if result['matches_expectations']:
            result['message'] = f"✓ Manufacturer matches user expectation: {expected_manufacturer}"
        else:
            result['message'] = f"✗ Manufacturer mismatch: Expected {expected_manufacturer}, detected {detected_manufacturer}"
    elif expected_part_number:
        # Only part number provided
        result['matches_expectations'] = result['part_number_match']
        if result['matches_expectations']:
            result['message'] = f"✓ Part number matches user expectation: {expected_part_number}"
        else:
            result['message'] = f"✗ Part number mismatch: Expected {expected_part_number}"
    
    return result


# Initialize services (singleton pattern)
ocr_logo_detector = None
database = None
ocr_extractor = None
ic_verifier = None


def initialize_services():
    """
    Initialize all services (lazy loading)
    """
    global ocr_logo_detector, database, ocr_extractor, ic_verifier
    
    # Initialize OCR extractor first (needed by OCR logo detector)
    if ocr_extractor is None:
        ocr_extractor = get_ocr_extractor()
        print("✓ OCR extractor initialized")
    
    if ocr_logo_detector is None:
        ocr_logo_detector = get_ocr_logo_detector(ocr_extractor)
        print("✓ OCR logo detector initialized")
    
    if database is None:
        database = create_database()
        print("✓ Database initialized")
    
    if ic_verifier is None:
        ic_verifier = get_verifier()
        print("✓ IC verifier initialized")


@csrf_exempt
@require_http_methods(["POST"])
def detect_manufacturer_stream(request):
    """
    Streaming endpoint for progressive IC manufacturer detection updates
    
    POST /api/detect-stream/
    - Upload IC image
    - Stream progress updates for each verification step
    - Return Server-Sent Events (SSE)
    """
    from django.http import StreamingHttpResponse
    import time
    
    # Parse multipart form data first
    uploaded_file = request.FILES.get('image')
    expected_manufacturer = request.POST.get('expected_manufacturer', None)
    expected_part_number = request.POST.get('expected_part_number', None)
    
    def event_stream():
        """Generator function that yields SSE-formatted progress updates"""
        try:
            # Initialize services
            initialize_services()
            
            # Check if image file is present
            if not uploaded_file:
                yield f"data: {json.dumps({'error': 'No image file provided', 'step': 'error'})}\n\n"
                return
            
            # Step 1: Image Preprocessing
            yield f"data: {json.dumps({'step': 'preprocessing', 'status': 'processing', 'message': 'Preprocessing image...'})}\n\n"
            time.sleep(0.5)
            
            is_valid, error_message = validate_image(uploaded_file)
            if not is_valid:
                yield f"data: {json.dumps({'error': error_message, 'step': 'error'})}\n\n"
                return
            
            image_path = save_uploaded_image(uploaded_file)
            original_image = load_image_from_file(image_path)
            
            yield f"data: {json.dumps({'step': 'preprocessing', 'status': 'completed', 'message': 'Image ready for analysis ✓'})}\n\n"
            
            # Step 2: Logo Detection
            yield f"data: {json.dumps({'step': 'logo_detection', 'status': 'processing', 'message': 'Detecting manufacturer logo...'})}\n\n"
            time.sleep(0.5)
            
            combined_result = ocr_logo_detector.detect_and_extract_combined(original_image, min_confidence=0.5)
            
            if not combined_result.get('success') or not combined_result['logo_detection'].get('success'):
                yield f"data: {json.dumps({'error': combined_result['logo_detection'].get('error', 'No manufacturer logo detected'), 'step': 'error'})}\n\n"
                return
            
            detection_result = combined_result['logo_detection']
            marking_result = combined_result['marking_extraction']
            
            manufacturer = detection_result['manufacturer']
            confidence = detection_result['confidence']
            
            yield f"data: {json.dumps({'step': 'logo_detection', 'status': 'completed', 'message': f'{manufacturer}', 'data': {'manufacturer': manufacturer, 'confidence': float(confidence)}})}\n\n"
            
            # Step 3: OCR Extraction
            yield f"data: {json.dumps({'step': 'ocr_extraction', 'status': 'processing', 'message': 'Extracting IC markings...'})}\n\n"
            time.sleep(0.5)
            
            ocr_result = {
                'success': marking_result['success'],
                'full_text': marking_result.get('raw_text', ''),
                'raw_text': marking_result.get('raw_text', ''),
                'extracted_text': [region['text'] for region in marking_result.get('text_regions', [])],
                'text_regions': marking_result.get('text_regions', [])
            }
            
            ic_marking = ocr_result.get('full_text', 'N/A')
            
            # First, send OCR as processing (yellow)
            # We'll update it later based on verification result
            
            # Step 4: Database Verification
            yield f"data: {json.dumps({'step': 'verification', 'status': 'processing', 'message': 'Verifying authenticity...'})}\n\n"
            time.sleep(0.5)
            
            parsed_marking = None
            verification_result = None
            part_found_in_db = False
            
            if ocr_result.get("success"):
                manufacturer_parts = ic_verifier.get_manufacturer_parts(manufacturer)
                
                if manufacturer_parts:
                    parsed_marking = parse_marking(
                        ocr_result=ocr_result,
                        manufacturer=manufacturer,
                        known_parts=manufacturer_parts
                    )
                    
                    verification_result = ic_verifier.comprehensive_verification(
                        parsed_marking=parsed_marking,
                        manufacturer=manufacturer,
                        logo_confidence=confidence
                    )
                    
                    # Check if part number was found in database
                    if verification_result:
                        details = verification_result.get('details', {})
                        part_found_in_db = details.get('part_number_valid', False)
            
            # Determine status and verification step status
            final_status = 'fake'  # lowercase for frontend compatibility
            final_message = f"Detected as {manufacturer} logo"
            verification_step_status = 'completed'  # Default to completed
            ocr_step_status = 'completed'  # Default to completed
            verification_source = 'local'  # Track where verification came from
            
            if verification_result:
                authenticity = verification_result.get('authenticity', 'fake')
                details = verification_result.get('details', {})
                
                # Check verification source (Nexar vs Local)
                verified_via = details.get('verified_via', '')
                if 'Nexar' in verified_via or 'Web' in verified_via:
                    verification_source = 'nexar'
                else:
                    verification_source = 'local'
                
                if authenticity == 'genuine':
                    final_status = 'genuine'  # lowercase for frontend
                    
                    # Different messages based on source
                    if verification_source == 'nexar':
                        final_message = f"❌ NOT FOUND IN LOCAL DATABASE✅ VERIFIED VIA NEXAR WEB DATABASE"
                    else:
                        final_message = f"✅ VERIFIED AS GENUINE"
                    
                    verification_step_status = 'completed'
                    ocr_step_status = 'completed'
                else:
                    final_status = 'fake'  # lowercase for frontend
                    final_message = f"❌ FAKE DETECTED"
                    verification_step_status = 'failed'  # Mark as failed for fake ICs
                    
                    # If part not found in DB, mark OCR extraction as failed too
                    if not part_found_in_db:
                        ocr_step_status = 'failed'
            else:
                # No verification result means failure
                verification_step_status = 'failed'
                ocr_step_status = 'failed'
                final_status = 'fake'
                final_message = f"❌ FAKE DETECTED"
            
            # Now update OCR step with final status (green if found, red if not found in DB)
            yield f"data: {json.dumps({'step': 'ocr_extraction', 'status': ocr_step_status, 'message': f'{ic_marking[:50]}', 'data': {'marking': ic_marking}})}\n\n"
            
            # Send verification result with source information
            verification_data = {
                'status': final_status,
                'source': verification_source
            }
            yield f"data: {json.dumps({'step': 'verification', 'status': verification_step_status, 'message': final_message, 'data': verification_data})}\n\n"
            
            # User Validation (if expected values provided)
            user_validation = None
            if expected_manufacturer or expected_part_number:
                user_validation = validate_user_expectations(
                    expected_manufacturer=expected_manufacturer or '',
                    expected_part_number=expected_part_number or '',
                    detected_manufacturer=manufacturer,
                    parsed_marking=parsed_marking or {},
                    ocr_result=ocr_result
                )
            
            # Final step: Complete
            time.sleep(0.3)
            complete_data = {
                'step': 'complete',
                'status': 'completed',
                'message': 'Analysis complete',
                'final_status': final_status
            }
            if user_validation:
                complete_data['user_validation'] = user_validation
            
            yield f"data: {json.dumps(complete_data)}\n\n"
            
        except Exception as e:
            print(f"Stream error: {str(e)}")
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e), 'step': 'error'})}\n\n"
    
    response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'
    return response


@csrf_exempt
@require_http_methods(["POST"])
def detect_manufacturer(request):
    """
    Main endpoint for IC manufacturer detection
    
    POST /api/detect/
    - Upload IC image
    - Detect and crop logo
    - Predict manufacturer
    - Optional: expected_manufacturer and expected_part_number for validation
    - Return results
    """
    try:
        # Initialize services
        initialize_services()
        
        # Check if image file is present
        if 'image' not in request.FILES:
            return JsonResponse({
                'success': False,
                'error': 'No image file provided'
            }, status=400)
        
        uploaded_file = request.FILES['image']
        
        # Get optional user inputs for validation
        expected_manufacturer = request.POST.get('expected_manufacturer', None)
        expected_part_number = request.POST.get('expected_part_number', None)
        
        print(f"📝 User Input - Manufacturer: {expected_manufacturer}, Part Number: {expected_part_number}")
        
        # Validate image
        is_valid, error_message = validate_image(uploaded_file)
        if not is_valid:
            return JsonResponse({
                'success': False,
                'error': error_message
            }, status=400)
        
        # Save uploaded image
        image_path = save_uploaded_image(uploaded_file)
        
        # Load image
        original_image = load_image_from_file(image_path)
        
        # OPTIMIZED: Single OCR pass for both logo detection AND marking extraction (MUCH FASTER!)
        print("⚡ Using optimized combined OCR extraction...")
        combined_result = ocr_logo_detector.detect_and_extract_combined(original_image, min_confidence=0.5)
        
        if not combined_result.get('success') or not combined_result['logo_detection'].get('success'):
            return JsonResponse({
                'success': False,
                'error': combined_result['logo_detection'].get('error', 'No manufacturer logo text detected')
            }, status=400)
        
        # Extract logo detection result
        detection_result = combined_result['logo_detection']
        marking_result = combined_result['marking_extraction']
        
        print(f"✓ Completed in 1 OCR pass (was 2 passes before) - {combined_result.get('total_ocr_calls', 1)} call(s)")
        
        # Get prediction result from OCR logo detector
        prediction_result = {
            'manufacturer': detection_result['manufacturer'],
            'confidence': detection_result['confidence'],
            'confidence_percentage': detection_result['confidence_percentage'],
            'logo_text': detection_result.get('logo_text', ''),
            'all_probabilities': {detection_result['manufacturer']: detection_result['confidence']}
        }
        
        # Get top 3 predictions (all manufacturer matches found)
        all_matches = detection_result.get('all_matches', [])
        top_predictions = sorted(
            [{'manufacturer': m['manufacturer'], 'confidence': m['confidence']} 
             for m in all_matches],
            key=lambda x: x['confidence'],
            reverse=True
        )[:3]
        
        # Get cropped logo region for display
        bbox = detection_result.get('bbox')
        if bbox:
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            # Convert PIL Image to numpy array for slicing
            img_array = np.array(original_image)
            cropped_logo = img_array[y1:y2, x1:x2]
        else:
            cropped_logo = np.array(original_image)
        
        # Use the extracted marking text from combined result (already extracted in single OCR pass!)
        ocr_result = {
            'success': marking_result['success'],
            'full_text': marking_result.get('raw_text', ''),  # Parser expects 'full_text'
            'raw_text': marking_result.get('raw_text', ''),
            'extracted_text': [region['text'] for region in marking_result.get('text_regions', [])],  # Parser expects list of strings
            'text_regions': marking_result.get('text_regions', [])
        }
        
        print(f"📋 OCR Result for parser: full_text='{ocr_result['full_text']}', extracted_text={ocr_result['extracted_text']}")
        
        # Step 3: Parse IC markings
        parsed_marking = None
        verification_result = None
        
        if ocr_result.get("success"):
            # Get known parts for the detected manufacturer
            manufacturer_parts = ic_verifier.get_manufacturer_parts(prediction_result['manufacturer'])
            
            print(f"📋 Manufacturer parts found: {len(manufacturer_parts) if manufacturer_parts else 0}")
            
            if manufacturer_parts:
                # Parse the extracted text
                parsed_marking = parse_marking(
                    ocr_result=ocr_result,
                    manufacturer=prediction_result['manufacturer'],
                    known_parts=manufacturer_parts
                )
                
                print(f"📋 Parsed marking type: {type(parsed_marking)}")
                print(f"📋 Parsed marking: {parsed_marking}")
                
                # Step 7: Verify IC authenticity
                verification_result = ic_verifier.comprehensive_verification(
                    parsed_marking=parsed_marking,
                    manufacturer=prediction_result['manufacturer'],
                    logo_confidence=prediction_result['confidence']
                )
                
                print(f"📋 Verification result: {verification_result}")
        
        # Convert images to base64 for response
        cropped_logo_base64 = image_to_base64(cropped_logo)
        
        # USER INPUT VALIDATION (NEW FEATURE)
        user_validation = None
        if expected_manufacturer or expected_part_number:
            user_validation = validate_user_expectations(
                expected_manufacturer=expected_manufacturer,
                expected_part_number=expected_part_number,
                detected_manufacturer=prediction_result['manufacturer'],
                parsed_marking=parsed_marking,
                ocr_result=ocr_result
            )
            print(f"👤 User validation result: {user_validation}")
        
        # Determine final status based on verification - ONLY TWO OPTIONS: genuine or fake
        final_status = 'fake'  # Default to fake until proven genuine
        final_message = f"Detected as {prediction_result['manufacturer']} logo"
        
        # If user validation exists, override status based on user expectations
        if user_validation:
            if user_validation['matches_expectations']:
                final_status = 'genuine'  # Matches expectations = Genuine
                final_message = user_validation['message']
            else:
                final_status = 'fake'  # Doesn't match = Fake
                final_message = user_validation['message']
        elif verification_result:
            authenticity = verification_result.get('authenticity', 'fake')
            if authenticity == 'genuine':
                final_status = 'genuine'
                final_message = f"✅ VERIFIED AS GENUINE {prediction_result['manufacturer']} IC"
            else:
                # Anything not genuine = fake
                final_status = 'fake'
                final_message = f"❌ FAKE IC DETECTED - Markings do not match authorized database"
        
        # Save to database (include verification results)
        inspection_data = {
            'original_filename': uploaded_file.name,
            'image_path': image_path,
            'manufacturer': prediction_result['manufacturer'],
            'confidence': prediction_result['confidence'],
            'all_probabilities': prediction_result['all_probabilities'],
            'top_predictions': top_predictions,
            'ocr_results': {
                'extracted_text': ocr_result.get('extracted_text', []),
                'full_text': ocr_result.get('full_text', ''),
                'average_confidence': ocr_result.get('average_confidence', 0.0)
            } if ocr_result.get('success') else None,
            'parsed_marking': parsed_marking,
            'verification': verification_result,
            'final_status': final_status
        }
        
        inspection_id = database.add_inspection(inspection_data)
        
        # Prepare comprehensive response
        response_data = {
            'success': True,
            'inspection_id': inspection_id,
            'result': {
                'manufacturer': prediction_result['manufacturer'],
                'confidence': prediction_result['confidence'],
                'confidence_percentage': prediction_result['confidence_percentage'],
                'status': final_status,
                'message': final_message
            },
            'logo_detection': {
                'top_predictions': top_predictions,
                'all_probabilities': prediction_result['all_probabilities'],
                'cropped_logo': cropped_logo_base64
            },
            'ocr_extraction': {
                'success': ocr_result.get('success', False),
                'extracted_text': ocr_result.get('text_regions', []),
                'full_text': ocr_result.get('raw_text', ''),
                'confidence': 0.0,  # Will be calculated if needed
                'total_detections': ocr_result.get('total_regions', 0)
            } if ocr_result else None,
            'marking_analysis': None,  # Will be populated below if valid
            'verification': verification_result,
            'user_validation': user_validation  # NEW: Include user validation results
        }
        
        # Safely add marking_analysis if available
        if parsed_marking and isinstance(parsed_marking, dict):
            try:
                response_data['marking_analysis'] = {
                    'part_number': parsed_marking.get('part_number', {}).get('part_number', None) if isinstance(parsed_marking.get('part_number'), dict) else None,
                    'suffix': parsed_marking.get('suffix', {}).get('suffix', None) if isinstance(parsed_marking.get('suffix'), dict) else None,
                    'date_code': parsed_marking.get('date_code', {}).get('formatted', None) if isinstance(parsed_marking.get('date_code'), dict) else None,
                    'country_code': parsed_marking.get('country_code', None)
                }
            except Exception as e:
                print(f"⚠ Error building marking_analysis: {e}")
                response_data['marking_analysis'] = None
        
        return JsonResponse(response_data)
    
    except Exception as e:
        print(f"Error in detect_manufacturer: {str(e)}")
        traceback.print_exc()
        
        return JsonResponse({
            'success': False,
            'error': f"Internal server error: {str(e)}"
        }, status=500)


@require_http_methods(["GET"])
def get_inspection_history(request):
    """
    Get inspection history
    
    GET /api/history/
    - Returns list of past inspections
    """
    try:
        initialize_services()
        
        # Get limit from query params
        limit = request.GET.get('limit', 50)
        try:
            limit = int(limit)
        except ValueError:
            limit = 50
        
        # Get inspections
        inspections = database.get_all_inspections(limit=limit)
        
        return JsonResponse({
            'success': True,
            'count': len(inspections),
            'inspections': inspections
        })
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@require_http_methods(["GET"])
def get_statistics(request):
    """
    Get database statistics
    
    GET /api/statistics/
    - Returns overall statistics
    """
    try:
        initialize_services()
        
        stats = database.get_statistics()
        
        return JsonResponse({
            'success': True,
            'statistics': stats
        })
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@require_http_methods(["GET"])
def health_check(request):
    """
    Health check endpoint
    
    GET /api/health/
    - Check if services are running
    """
    try:
        initialize_services()
        
        return JsonResponse({
            'success': True,
            'status': 'healthy',
            'services': {
                'ocr_logo_detector': ocr_logo_detector is not None,
                'database': database is not None,
                'ocr_extractor': ocr_extractor is not None,
                'ic_verifier': ic_verifier is not None
            },
            'detection_method': 'OCR-based logo text matching'
        })
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'status': 'unhealthy',
            'error': str(e)
        }, status=500)
