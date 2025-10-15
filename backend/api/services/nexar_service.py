"""
Nexar IC Database Service
Queries Nexar's comprehensive IC component database for parts not found locally
"""

import os
import requests
from typing import Dict, Optional, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GRAPHQL_URL = "https://api.nexar.com/graphql"
ACCESS_TOKEN = os.getenv('NEXAR_ACCESS_TOKEN', '')


class NexarService:
    """
    Service for querying Nexar IC database
    Used as fallback when part is not found in local database
    """
    
    def __init__(self):
        """Initialize Nexar service with access token"""
        self.access_token = ACCESS_TOKEN
        self.graphql_url = GRAPHQL_URL
        
        if not self.access_token or self.access_token == 'your_nexar_access_token_here':
            print("⚠️  Warning: Nexar access token not configured. Web verification disabled.")
            self.enabled = False
        else:
            print("✓ Nexar service initialized with API endpoint:", GRAPHQL_URL)
            self.enabled = True
    
    def test_connection(self) -> bool:
        """
        Test Nexar API connection
        
        Returns:
            True if connection successful, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            # Simple query to test connection
            query = """
            query {
              __schema {
                queryType {
                  name
                }
              }
            }
            """
            
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                self.graphql_url,
                json={"query": query},
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                print("✓ Nexar API connection test successful")
                return True
            else:
                print(f"❌ Nexar API connection test failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Nexar API connection test error: {e}")
            return False
    
    def search_part(self, part_number: str, manufacturer: str = None, limit: int = 3) -> Optional[Dict[str, Any]]:
        """
        Search for IC part in Nexar database
        
        Args:
            part_number: Part number to search (e.g., "LM324", "STM32F103C8T6")
            manufacturer: Optional manufacturer name to narrow search
            limit: Maximum number of results (default: 3)
            
        Returns:
            Dictionary with part information or None if not found
        """
        if not self.enabled:
            return None
        
        try:
            # Build search query (combine part number and manufacturer if provided)
            search_query = part_number
            if manufacturer:
                search_query = f"{manufacturer} {part_number}"
            
            print(f"🔍 Searching Nexar for: '{search_query}'")
            
            # GraphQL query for IC search
            query = """
            query searchIC($q: String!, $limit: Int) {
              supSearchMpn(q: $q, limit: $limit) {
                results {
                  part {
                    mpn
                    manufacturer { 
                      name 
                      id
                    }
                    shortDescription
                    category {
                      name
                      path
                    }
                    specs {
                      attribute {
                        name
                        shortname
                      }
                      value
                    }
                    totalAvail
                    sellers {
                      company { 
                        name 
                        id
                      }
                      offers {
                        inventoryLevel
                        moq
                        prices {
                          currency
                          price
                          quantity
                        }
                      }
                    }
                  }
                }
              }
            }
            """
            
            variables = {
                "q": search_query,
                "limit": limit
            }
            
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            # Make API request with retry logic
            max_retries = 2
            timeout = 30  # 30 second timeout
            response = None
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        self.graphql_url,
                        json={"query": query, "variables": variables},
                        headers=headers,
                        timeout=timeout
                    )
                    break  # Success, exit retry loop
                except requests.exceptions.Timeout:
                    if attempt < max_retries - 1:
                        print(f"⚠️  Nexar API timeout (attempt {attempt + 1}/{max_retries}), retrying...")
                        continue
                    else:
                        print(f"❌ Nexar API timeout after {max_retries} attempts ({timeout}s timeout)")
                        return None
                except requests.exceptions.RequestException as e:
                    print(f"❌ Nexar API request error: {e}")
                    return None
            
            # Check if response was successfully obtained
            if response is None:
                print(f"❌ Nexar API request failed - no response received")
                return None
            
            if response.status_code != 200:
                if response.status_code == 401:
                    print(f"❌ Nexar API authentication failed - Token may be expired (HTTP 401)")
                    print(f"   Please refresh your Nexar access token at: https://portal.nexar.com/")
                elif response.status_code == 403:
                    print(f"❌ Nexar API forbidden - Insufficient permissions (HTTP 403)")
                else:
                    print(f"❌ Nexar API error: HTTP {response.status_code}")
                    try:
                        error_data = response.json()
                        print(f"   Error details: {error_data}")
                    except:
                        print(f"   Response: {response.text[:200]}")
                return None
            
            data = response.json()
            
            # Debug: Print response structure
            if not data:
                print(f"⚠️  Empty response from Nexar API")
                return None
            
            # Check for errors
            if 'errors' in data:
                print(f"❌ Nexar GraphQL error: {data['errors']}")
                return None
            
            # Extract results with safety checks
            if 'data' not in data:
                print(f"⚠️  No 'data' field in Nexar response")
                return None
                
            search_results = data.get('data', {})
            if not search_results or 'supSearchMpn' not in search_results:
                print(f"⚠️  No 'supSearchMpn' field in Nexar response")
                return None
            
            mpn_data = search_results.get('supSearchMpn', {})
            if not mpn_data or 'results' not in mpn_data:
                print(f"⚠️  No 'results' field in Nexar response")
                return None
                
            results = mpn_data.get('results', [])
            
            if not results:
                print(f"ℹ️  No results found in Nexar for: {search_query}")
                return None
            
            # Check if first result is valid
            first_result = results[0] if results else None
            if not first_result:
                print(f"ℹ️  No valid results in Nexar for: {search_query}")
                return None
            
            # Process and return first matching result
            return self._process_result(first_result, part_number)
            
        except requests.exceptions.JSONDecodeError as e:
            print(f"❌ Nexar API response parsing error: {e}")
            return None
        except AttributeError as e:
            print(f"❌ Nexar service attribute error: {e}")
            print(f"   This usually means a None value was accessed")
            import traceback
            traceback.print_exc()
            return None
        except Exception as e:
            print(f"❌ Nexar service error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _process_result(self, result: Dict, search_part_number: str) -> Dict[str, Any]:
        """
        Process Nexar API result into standardized format
        
        Args:
            result: Raw result from Nexar API
            search_part_number: Original search query
            
        Returns:
            Processed part information
        """
        # Safety check
        if not result or not isinstance(result, dict):
            print(f"⚠️  Invalid result format from Nexar API")
            return None
        
        part = result.get('part', {})
        
        # Extract manufacturer info
        manufacturer_data = part.get('manufacturer') or {}
        manufacturer_name = manufacturer_data.get('name', 'Unknown') if isinstance(manufacturer_data, dict) else 'Unknown'
        
        # Extract part number
        mpn = part.get('mpn', search_part_number)
        
        # Extract description
        description = part.get('shortDescription') or ''
        
        # Extract category
        category_data = part.get('category') or {}
        category = category_data.get('name', 'Unknown') if isinstance(category_data, dict) else 'Unknown'
        category_path = category_data.get('path', '') if isinstance(category_data, dict) else ''
        
        # Extract specifications
        specs = {}
        for spec in (part.get('specs') or []):
            if not isinstance(spec, dict):
                continue
            attr = spec.get('attribute') or {}
            if not isinstance(attr, dict):
                continue
            attr_name = attr.get('name') or attr.get('shortname')
            if attr_name:
                specs[attr_name] = spec.get('value')
        
        # Extract availability
        total_avail = part.get('totalAvail') or 0
        
        # Extract sellers info
        sellers = []
        for seller in (part.get('sellers') or [])[:3]:  # Limit to 3 sellers
            if not isinstance(seller, dict):
                continue
            company = seller.get('company') or {}
            offers = seller.get('offers') or []
            
            seller_info = {
                'name': company.get('name', 'Unknown') if isinstance(company, dict) else 'Unknown',
                'inventory': 0,
                'prices': []
            }
            
            for offer in (offers if isinstance(offers, list) else []):
                if not isinstance(offer, dict):
                    continue
                seller_info['inventory'] += offer.get('inventoryLevel') or 0
                for price in (offer.get('prices') or []):
                    if not isinstance(price, dict):
                        continue
                    seller_info['prices'].append({
                        'quantity': price.get('quantity') or 0,
                        'price': price.get('price') or 0,
                        'currency': price.get('currency') or 'USD'
                    })
            
            if seller_info['inventory'] > 0:  # Only include sellers with stock
                sellers.append(seller_info)
        
        return {
            'found': True,
            'source': 'nexar',
            'part_number': mpn,
            'manufacturer': manufacturer_name,
            'description': description,
            'category': category,
            'category_path': category_path,
            'specifications': specs,
            'total_availability': total_avail,
            'in_stock': total_avail > 0,
            'sellers': sellers,
            'seller_count': len(sellers)
        }
    
    def verify_part(self, part_number: str, manufacturer: str) -> Dict[str, Any]:
        """
        Verify if a part exists in Nexar database
        
        Args:
            part_number: Part number to verify
            manufacturer: Manufacturer name
            
        Returns:
            Verification result with status and details
        """
        if not self.enabled:
            return {
                'verified': False,
                'source': 'nexar',
                'message': 'Nexar service not configured',
                'available': False
            }
        
        result = self.search_part(part_number, manufacturer, limit=1)
        
        if result and result.get('found'):
            return {
                'verified': True,
                'source': 'nexar',
                'message': f'Part found in Nexar database',
                'part_info': result,
                'available': result.get('in_stock', False),
                'confidence': 0.9  # High confidence if found in Nexar
            }
        else:
            return {
                'verified': False,
                'source': 'nexar',
                'message': f'Part {part_number} not found in Nexar database',
                'available': False,
                'confidence': 0.0
            }


# Singleton instance
_nexar_service = None

def get_nexar_service() -> NexarService:
    """Get or create Nexar service singleton instance"""
    global _nexar_service
    if _nexar_service is None:
        _nexar_service = NexarService()
    return _nexar_service
