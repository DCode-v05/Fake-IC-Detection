"""
JSON Database Utilities
Handles reading and writing to JSON-based database
Stores inspection history and results
"""

import json
import os
from datetime import datetime


class JSONDatabase:
    """
    Simple JSON-based database for storing inspection results
    """
    
    def __init__(self, db_path='database/inspections.json'):
        """
        Initialize JSON database
        
        Args:
            db_path: Path to JSON database file
        """
        # Get absolute path
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.db_path = os.path.join(base_dir, db_path)
        
        # Create database directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize database file if it doesn't exist
        if not os.path.exists(self.db_path):
            self._initialize_database()
    
    def _initialize_database(self):
        """Create initial database structure"""
        initial_data = {
            'inspections': [],
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_inspections': 0
            }
        }
        
        with open(self.db_path, 'w') as f:
            json.dump(initial_data, f, indent=2)
    
    def _read_database(self):
        """
        Read database from file
        
        Returns:
            dict: Database content
        """
        try:
            with open(self.db_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading database: {str(e)}")
            return {'inspections': [], 'metadata': {}}
    
    def _write_database(self, data):
        """
        Write database to file
        
        Args:
            data: Database content to write
        """
        try:
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error writing database: {str(e)}")
    
    def add_inspection(self, inspection_data):
        """
        Add new inspection record
        
        Args:
            inspection_data: Dictionary containing inspection details
            
        Returns:
            str: Inspection ID
        """
        # Read current database
        db = self._read_database()
        
        # Generate inspection ID
        inspection_id = f"INS_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(db['inspections']) + 1}"
        
        # Add metadata to inspection
        inspection_record = {
            'id': inspection_id,
            'timestamp': datetime.now().isoformat(),
            **inspection_data
        }
        
        # Add to database
        db['inspections'].append(inspection_record)
        db['metadata']['total_inspections'] = len(db['inspections'])
        db['metadata']['last_updated'] = datetime.now().isoformat()
        
        # Write to file
        self._write_database(db)
        
        return inspection_id
    
    def get_inspection(self, inspection_id):
        """
        Get inspection by ID
        
        Args:
            inspection_id: Inspection ID
            
        Returns:
            dict: Inspection record or None
        """
        db = self._read_database()
        
        for inspection in db['inspections']:
            if inspection['id'] == inspection_id:
                return inspection
        
        return None
    
    def get_all_inspections(self, limit=None):
        """
        Get all inspections
        
        Args:
            limit: Maximum number of inspections to return
            
        Returns:
            list: List of inspection records
        """
        db = self._read_database()
        inspections = db['inspections']
        
        # Sort by timestamp (most recent first)
        inspections.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        if limit:
            return inspections[:limit]
        
        return inspections
    
    def get_inspections_by_manufacturer(self, manufacturer):
        """
        Get inspections for specific manufacturer
        
        Args:
            manufacturer: Manufacturer name
            
        Returns:
            list: Matching inspection records
        """
        db = self._read_database()
        
        return [
            inspection for inspection in db['inspections']
            if inspection.get('manufacturer') == manufacturer
        ]
    
    def get_statistics(self):
        """
        Get database statistics
        
        Returns:
            dict: Statistics
        """
        db = self._read_database()
        
        total_inspections = len(db['inspections'])
        
        # Count by manufacturer
        manufacturer_counts = {}
        for inspection in db['inspections']:
            manufacturer = inspection.get('manufacturer', 'Unknown')
            manufacturer_counts[manufacturer] = manufacturer_counts.get(manufacturer, 0) + 1
        
        # Calculate average confidence
        confidences = [
            inspection.get('confidence', 0) 
            for inspection in db['inspections']
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            'total_inspections': total_inspections,
            'manufacturer_distribution': manufacturer_counts,
            'average_confidence': avg_confidence,
            'last_updated': db['metadata'].get('last_updated', 'N/A')
        }
    
    def clear_database(self):
        """Clear all inspection records"""
        self._initialize_database()
        print("Database cleared")


def create_database(db_path='database/inspections.json'):
    """
    Factory function to create database instance
    
    Args:
        db_path: Path to database file
        
    Returns:
        JSONDatabase: Database instance
    """
    return JSONDatabase(db_path)
