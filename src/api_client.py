import requests
from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv

load_dotenv()

class OrderAPI:
    def __init__(self, api_url: Optional[str] = None):
        """Initialize the OrderAPI with the base URL from environment variables."""
        self.api_url = api_url or os.getenv('MOCK_API_URL') or "http://127.0.0.1:8000/data/"
        if not self.api_url:
            raise ValueError("API URL not found. Please set MOCK_API_URL in .env file")
    
    def get_order_by_id(self, customer_id: int) -> Dict[str, Any]:
        """
        Get orders for a specific customer ID from the mock API.
        
        Args:
            customer_id (int): The ID of the customer to query
            
        Returns:
            Dict[str, Any]: A dictionary containing either:
                - {"orders": [...], "error": None} if successful
                - {"orders": [], "error": "error message"} if there's an error
        """
        try:
            url = f"{self.api_url}customer/{customer_id}"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and "error" in data:
                    return {"orders": [], "error": data["error"]}
                return {"orders": data, "error": None}
            
            return {
                "orders": [], 
                "error": f"API request failed with status code: {response.status_code}"
            }
            
        except requests.RequestException as e:
            return {"orders": [], "error": f"Request failed: {str(e)}"}
        except Exception as e:
            return {"orders": [], "error": f"Unexpected error: {str(e)}"}