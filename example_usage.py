#!/usr/bin/env python3
"""
Example usage of the Logo Detection API URL batch endpoint
"""

import requests
import json

# API configuration
API_BASE_URL = "http://localhost:8000"
BATCH_ENDPOINT = f"{API_BASE_URL}/api/v1/urls/batch"

def detect_logos_in_urls(urls, confidence_threshold=0.5, max_detections=10):
    """
    Send URLs to the logo detection API for processing.
    
    Args:
        urls (list): List of image URLs to process
        confidence_threshold (float): Minimum confidence for detections (0-1)
        max_detections (int): Maximum number of detections per image
    
    Returns:
        dict: API response with detection results
    """
    payload = {
        "urls": urls,
        "confidence_threshold": confidence_threshold,
        "max_detections": max_detections
    }
    
    try:
        response = requests.post(BATCH_ENDPOINT, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")
        return None

def main():
    # Example URLs (replace with your own)
    example_urls = [
        "https://static.mercdn.net/item/detail/orig/photos/m50578354568_1.jpg?1743788948",
        "https://static.mercdn.net/item/detail/orig/photos/m54128539773_1.jpg?1622825534"
    ]
    
    print("Logo Detection API - Example Usage")
    print("=" * 40)
    
    # Call the API
    results = detect_logos_in_urls(example_urls, confidence_threshold=0.3)
    
    if results:
        print(f"\nProcessed {results['total_urls']} URLs")
        print(f"Successful: {results['processed']}")
        print(f"Failed: {results['failed']}")
        print(f"Total time: {results['processing_time']:.2f}s")
        
        # Show detections
        for url, result in results['results'].items():
            print(f"\n{url}:")
            if result['status'] == 'success':
                if result['detections']:
                    for detection in result['detections']:
                        print(f"  - {detection['brand_name']} ({detection['confidence']:.1%})")
                else:
                    print("  - No logos detected")
            else:
                print(f"  - Error: {result.get('error_message', 'Unknown')}")

if __name__ == "__main__":
    main()