"""
Image Processing Utilities
Helper functions for image handling and processing
"""

import os
import io
import base64
import numpy as np
from PIL import Image
from datetime import datetime


def save_uploaded_image(uploaded_file, upload_dir='media/uploads'):
    """
    Save uploaded image file to disk
    
    Args:
        uploaded_file: Django UploadedFile object
        upload_dir: Directory to save uploads
        
    Returns:
        str: Path to saved file
    """
    # Create upload directory if it doesn't exist
    os.makedirs(upload_dir, exist_ok=True)
    
    # Generate unique filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{uploaded_file.name}"
    filepath = os.path.join(upload_dir, filename)
    
    # Save file
    with open(filepath, 'wb+') as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)
    
    return filepath


def load_image_from_file(filepath):
    """
    Load image from file path
    
    Args:
        filepath: Path to image file
        
    Returns:
        PIL Image: Loaded image
    """
    try:
        image = Image.open(filepath).convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Failed to load image: {str(e)}")


def load_image_from_bytes(image_bytes):
    """
    Load image from bytes
    
    Args:
        image_bytes: Image data as bytes
        
    Returns:
        PIL Image: Loaded image
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Failed to load image from bytes: {str(e)}")


def image_to_base64(image):
    """
    Convert PIL Image or numpy array to base64 string
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        str: Base64 encoded image
    """
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def base64_to_image(base64_string):
    """
    Convert base64 string to PIL Image
    
    Args:
        base64_string: Base64 encoded image
        
    Returns:
        PIL Image: Decoded image
    """
    # Remove data URL prefix if present
    if 'base64,' in base64_string:
        base64_string = base64_string.split('base64,')[1]
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    return image


def resize_image(image, max_size=(800, 800)):
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: PIL Image
        max_size: Maximum (width, height)
        
    Returns:
        PIL Image: Resized image
    """
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image


def validate_image(uploaded_file, max_size_mb=10):
    """
    Validate uploaded image file
    
    Args:
        uploaded_file: Django UploadedFile object
        max_size_mb: Maximum file size in MB
        
    Returns:
        tuple: (is_valid, error_message)
    """
    # Check file size
    if uploaded_file.size > max_size_mb * 1024 * 1024:
        return False, f"File size exceeds {max_size_mb}MB limit"
    
    # Check file extension
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    
    if file_ext not in allowed_extensions:
        return False, f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
    
    # Try to open and verify it's a valid image
    try:
        image = Image.open(uploaded_file)
        image.verify()
        return True, None
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


def cleanup_old_uploads(upload_dir='media/uploads', days_old=7):
    """
    Clean up old uploaded files
    
    Args:
        upload_dir: Directory containing uploads
        days_old: Delete files older than this many days
    """
    import time
    
    if not os.path.exists(upload_dir):
        return
    
    current_time = time.time()
    
    for filename in os.listdir(upload_dir):
        filepath = os.path.join(upload_dir, filename)
        
        # Check if file is older than specified days
        if os.path.isfile(filepath):
            file_age = current_time - os.path.getmtime(filepath)
            if file_age > (days_old * 24 * 3600):
                try:
                    os.remove(filepath)
                    print(f"Deleted old file: {filename}")
                except Exception as e:
                    print(f"Failed to delete {filename}: {str(e)}")
