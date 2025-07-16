# URL Batch Detection API

## Overview

The URL Batch Detection API allows you to process multiple image URLs for logo detection in a single request. This is efficient for analyzing product images from e-commerce sites or any collection of online images.

## Endpoints

### 1. Batch URL Processing

**Endpoint:** `POST /api/v1/urls/batch`

**Description:** Process multiple image URLs for logo detection.

**Request Body:**
```json
{
  "urls": [
    "https://example.com/image1.jpg",
    "https://example.com/image2.jpg"
  ],
  "confidence_threshold": 0.5,
  "max_detections": 10
}
```

**Parameters:**
- `urls` (required): Array of image URLs to process (max 100)
- `confidence_threshold` (optional): Minimum confidence score for detections (0.0-1.0, default: 0.5)
- `max_detections` (optional): Maximum number of detections per image (default: 10)

**Response:**
```json
{
  "total_urls": 2,
  "processed": 2,
  "failed": 0,
  "processing_time": 1.234,
  "results": {
    "https://example.com/image1.jpg": {
      "detections": [
        {
          "brand_name": "Apple",
          "confidence": 0.95,
          "category": "Technology",
          "bbox": [100, 100, 200, 200]
        }
      ],
      "processing_time": 0.5,
      "status": "success",
      "error_message": null
    },
    "https://example.com/image2.jpg": {
      "detections": [],
      "processing_time": 0.4,
      "status": "success",
      "error_message": null
    }
  }
}
```

### 2. Process URLs from Text

**Endpoint:** `POST /api/v1/urls/from-file`

**Description:** Process URLs provided as newline-delimited text.

**Request:**
- Content-Type: text/plain
- Body: Newline-separated URLs
- Query Parameters:
  - `confidence_threshold` (optional): 0.0-1.0
  - `max_detections` (optional): integer

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/urls/from-file?confidence_threshold=0.5" \
  -H "Content-Type: text/plain" \
  -d "https://example.com/image1.jpg
https://example.com/image2.jpg"
```

## Usage Examples

### Python Example

```python
import requests

# API endpoint
api_url = "http://localhost:8000/api/v1/urls/batch"

# URLs to process
urls = [
    "https://static.mercdn.net/item/detail/orig/photos/m50578354568_1.jpg",
    "https://static.mercdn.net/item/detail/orig/photos/m54128539773_1.jpg"
]

# Send request
response = requests.post(api_url, json={
    "urls": urls,
    "confidence_threshold": 0.5,
    "max_detections": 10
})

# Process results
if response.status_code == 200:
    results = response.json()
    for url, result in results['results'].items():
        print(f"{url}: {len(result['detections'])} logos found")
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/api/v1/urls/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://example.com/image1.jpg",
      "https://example.com/image2.jpg"
    ],
    "confidence_threshold": 0.5
  }'
```

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid URLs, empty list, too many URLs)
- `422`: Validation error
- `500`: Internal server error

Error responses include detailed messages:
```json
{
  "detail": "URL list cannot be empty"
}
```

## Performance Notes

- Maximum 100 URLs per batch request
- URLs are processed concurrently for optimal performance
- Processing time depends on image size and network speed
- Failed URLs don't stop the batch; they're marked as failed in results

## Testing

Use the provided `test_urls.py` script to test the API:
```bash
python3 test_urls.py
```

Or use the example script:
```bash
python3 example_usage.py
```