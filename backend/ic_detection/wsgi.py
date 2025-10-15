"""
WSGI config for ic_detection project
"""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ic_detection.settings')

application = get_wsgi_application()

# Pre-load OCR models at startup for faster response times
# Only load in the main process (not in Django auto-reloader child processes)
if os.environ.get('RUN_MAIN') == 'true' or os.environ.get('RUN_MAIN') is None:
    print("\n" + "="*60)
    print("🚀 INITIALIZING OCR MODELS AT STARTUP...")
    print("="*60)

    try:
        from api.services.ocr_logo_detector import get_ocr_logo_detector
        from api.services.ocr_extractor import get_ocr_extractor
        from api.services.ic_verifier import get_verifier
        from api.utils.database import create_database
        
        # Initialize all services
        ocr_extractor = get_ocr_extractor()
        print("✅ OCR Extractor loaded successfully")
        
        ocr_logo_detector = get_ocr_logo_detector(ocr_extractor)
        print("✅ OCR Logo Detector loaded successfully")
        
        database = create_database()
        print("✅ Database loaded successfully")
        
        ic_verifier = get_verifier()
        print("✅ IC Verifier loaded successfully")
        
        print("="*60)
        print("✅ ALL MODELS LOADED - READY FOR REQUESTS!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not pre-load models: {e}")
        print("Models will be loaded on first request instead.\n")
