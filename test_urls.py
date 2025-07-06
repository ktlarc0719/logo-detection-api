#!/usr/bin/env python3
"""Test script for URL batch detection API"""

import requests
import json
import time

# Test URLs provided
test_urls = [
    "https://static.mercdn.net/item/detail/orig/photos/m50578354568_1.jpg?1743788948",
    "https://static.mercdn.net/item/detail/orig/photos/m54128539773_1.jpg?1622825534",
    "https://static.mercdn.net/item/detail/orig/photos/m34160011091_1.jpg?1694982289",
    "https://static.mercdn.net/item/detail/orig/photos/m22430349305_1.jpg?1750648148"
]

def test_url_batch_endpoint():
    """Test the URL batch detection endpoint"""
    
    # API endpoint
    api_url = "http://localhost:8000/api/v1/urls/batch"
    
    # Request payload
    payload = {
        "urls": test_urls,
        "confidence_threshold": 0.5,
        "max_detections": 10
    }
    
    print("Testing URL batch detection API...")
    print(f"Sending {len(test_urls)} URLs for processing")
    print("-" * 50)
    
    try:
        # Send request
        start_time = time.time()
        response = requests.post(api_url, json=payload)
        end_time = time.time()
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            print(f"✓ API call successful!")
            print(f"  Total processing time: {end_time - start_time:.2f} seconds")
            print(f"  Total URLs: {result['total_urls']}")
            print(f"  Successfully processed: {result['processed']}")
            print(f"  Failed: {result['failed']}")
            print("\nDetection Results:")
            print("-" * 50)
            
            # Display results for each URL
            for url, image_result in result['results'].items():
                print(f"\nURL: {url}")
                print(f"Status: {image_result['status']}")
                
                if image_result['status'] == 'success':
                    detections = image_result['detections']
                    if detections:
                        print(f"Found {len(detections)} logo(s):")
                        for i, detection in enumerate(detections, 1):
                            print(f"  {i}. Brand: {detection['brand_name']}")
                            print(f"     Confidence: {detection['confidence']:.2%}")
                            print(f"     Category: {detection.get('category', 'N/A')}")
                            print(f"     Bounding Box: {detection['bbox']}")
                    else:
                        print("  No logos detected")
                else:
                    print(f"  Error: {image_result.get('error_message', 'Unknown error')}")
                
        else:
            print(f"✗ API call failed with status code: {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to API. Make sure the server is running.")
    except Exception as e:
        print(f"✗ Unexpected error: {str(e)}")


def test_urls_from_text_endpoint():
    """Test the URLs from text endpoint"""
    
    # API endpoint
    api_url = "http://localhost:8000/api/v1/urls/from-file"
    
    # Convert URLs to newline-delimited text
    urls_text = "\n".join(test_urls)
    
    print("\n\nTesting URLs from text endpoint...")
    print("-" * 50)
    
    try:
        # Send request
        response = requests.post(
            api_url,
            params={
                "confidence_threshold": 0.5,
                "max_detections": 10
            },
            data=urls_text,
            headers={"Content-Type": "text/plain"}
        )
        
        if response.status_code == 200:
            print("✓ Text endpoint test successful!")
        else:
            print(f"✗ Text endpoint test failed: {response.status_code}")
            
    except Exception as e:
        print(f"✗ Text endpoint test error: {str(e)}")


if __name__ == "__main__":
    print("Logo Detection API - URL Batch Testing")
    print("=" * 50)
    
    # Test main batch endpoint
    test_url_batch_endpoint()
    
    # Test text-based endpoint
    # test_urls_from_text_endpoint()