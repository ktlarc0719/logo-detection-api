"""
Logo Detection API Management UI
Simple web interface for model management and batch processing
"""

from flask import Flask, render_template, jsonify, request
import requests
import json
from datetime import datetime
import os
from typing import Dict, List, Any

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')

# API base URL - can be configured via environment variable
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/api/model/current', methods=['GET'])
def get_current_model():
    """Get current active model information"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/models/current")
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({
            'error': 'Failed to fetch current model',
            'details': str(e)
        }), 500


@app.route('/api/model/list', methods=['GET'])
def get_model_list():
    """Get list of available models"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/models")
        response.raise_for_status()
        data = response.json()
        
        # Extract model names and current status
        models = []
        for model_name, model_info in data.get('models', {}).items():
            models.append({
                'name': model_name,
                'is_current': model_info.get('is_current', False),
                'loaded': model_info.get('loaded', False),
                'source': model_info.get('source', 'unknown')
            })
        
        return jsonify({
            'models': models,
            'current_model': data.get('current_model', 'unknown')
        })
    except requests.exceptions.RequestException as e:
        return jsonify({
            'error': 'Failed to fetch model list',
            'details': str(e)
        }), 500


@app.route('/api/model/switch', methods=['POST'])
def switch_model():
    """Switch to a different model"""
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        
        if not model_name:
            return jsonify({'error': 'Model name is required'}), 400
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/models/switch",
            params={'model_name': model_name}
        )
        response.raise_for_status()
        
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({
            'error': 'Failed to switch model',
            'details': str(e)
        }), 500


@app.route('/api/process/batch', methods=['POST'])
def process_batch():
    """Process batch of images"""
    try:
        data = request.get_json()
        urls = data.get('urls', [])
        
        if not urls:
            return jsonify({'error': 'No URLs provided'}), 400
        
        # Prepare batch request
        batch_request = {
            'images': [{'id': f'img_{i}', 'url': url} for i, url in enumerate(urls)],
            'model': data.get('model'),
            'confidence_threshold': data.get('confidence_threshold', 0.5)
        }
        
        # Send batch request
        response = requests.post(
            f"{API_BASE_URL}/api/v1/process/batch",
            json=batch_request,
            timeout=300  # 5 minutes timeout for batch processing
        )
        response.raise_for_status()
        
        # Process results for visualization
        results = response.json()
        
        # Add summary statistics
        if 'results' in results:
            total_images = len(results['results'])
            successful = sum(1 for r in results['results'] if r['status'] == 'success')
            total_detections = sum(
                len(r.get('detections', [])) 
                for r in results['results'] 
                if r['status'] == 'success'
            )
            
            results['summary'] = {
                'total_images': total_images,
                'successful': successful,
                'failed': total_images - successful,
                'total_detections': total_detections,
                'average_detections': total_detections / successful if successful > 0 else 0
            }
        
        return jsonify(results)
    except requests.exceptions.Timeout:
        return jsonify({
            'error': 'Batch processing timeout',
            'details': 'The request took too long to process'
        }), 504
    except requests.exceptions.RequestException as e:
        return jsonify({
            'error': 'Failed to process batch',
            'details': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Check API health status"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({
            'status': 'error',
            'api_accessible': False,
            'details': str(e)
        }), 503


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Endpoint not found'}), 404
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Development server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=os.getenv('FLASK_ENV') == 'development'
    )