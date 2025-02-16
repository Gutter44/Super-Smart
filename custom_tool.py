from typing import Dict, Optional
import requests
import logging

class APITool:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.logger = logging.getLogger(__name__)

    def make_request(self, 
                    url: str, 
                    method: str = "GET", 
                    data: Optional[Dict] = None) -> Dict:
        try:
            self.logger.info(f"Making {method} request to {url}")
            response = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                json=data,
                timeout=30  # Add a timeout to prevent hanging requests
            )
            response.raise_for_status()
            self.logger.info(f"Request to {url} successful")
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {str(e)}")
            return {"error": str(e)}
        except ValueError as e:
            self.logger.error(f"Invalid JSON response: {str(e)}")
            return {"error": "Invalid JSON response"}

