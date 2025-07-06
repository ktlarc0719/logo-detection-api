#!/usr/bin/env python3
"""
Memory configuration test patterns for 2GB VPS environment
"""

# Test configurations for 2GB VPS
MEMORY_CONFIGS = [
    {
        "name": "Conservative",
        "memory": "1.2g",
        "memory_reservation": "800m",
        "batch_settings": {
            "MAX_CONCURRENT_DETECTIONS": 2,
            "MAX_CONCURRENT_DOWNLOADS": 10,
            "MAX_BATCH_SIZE": 20
        },
        "description": "最も安定性重視、システムに800MB確保"
    },
    {
        "name": "Balanced",
        "memory": "1.5g",
        "memory_reservation": "1g",
        "batch_settings": {
            "MAX_CONCURRENT_DETECTIONS": 2,
            "MAX_CONCURRENT_DOWNLOADS": 15,
            "MAX_BATCH_SIZE": 30
        },
        "description": "バランス型、システムに500MB確保"
    },
    {
        "name": "Performance",
        "memory": "1.7g",
        "memory_reservation": "1.2g",
        "batch_settings": {
            "MAX_CONCURRENT_DETECTIONS": 3,
            "MAX_CONCURRENT_DOWNLOADS": 20,
            "MAX_BATCH_SIZE": 40
        },
        "description": "性能重視、システムに300MB確保"
    },
    {
        "name": "Aggressive",
        "memory": "1.8g",
        "memory_reservation": "1.4g",
        "batch_settings": {
            "MAX_CONCURRENT_DETECTIONS": 3,
            "MAX_CONCURRENT_DOWNLOADS": 25,
            "MAX_BATCH_SIZE": 50
        },
        "description": "最大性能、システムに200MB確保（リスク高）"
    }
]

# Test scenarios
TEST_SCENARIOS = [
    {
        "name": "Light Load",
        "urls": 10,
        "description": "軽負荷テスト"
    },
    {
        "name": "Normal Load",
        "urls": 30,
        "description": "通常負荷テスト"
    },
    {
        "name": "Heavy Load",
        "urls": 50,
        "description": "高負荷テスト"
    },
    {
        "name": "Stress Test",
        "urls": 100,
        "description": "ストレステスト"
    }
]

def generate_docker_command(config):
    """Generate docker run command for testing"""
    cmd = f"""docker run -d \\
    --name logo-detection-test \\
    --memory {config['memory']} \\
    --memory-reservation {config['memory_reservation']} \\
    -p 8000:8000 \\
    -e MAX_CONCURRENT_DETECTIONS={config['batch_settings']['MAX_CONCURRENT_DETECTIONS']} \\
    -e MAX_CONCURRENT_DOWNLOADS={config['batch_settings']['MAX_CONCURRENT_DOWNLOADS']} \\
    -e MAX_BATCH_SIZE={config['batch_settings']['MAX_BATCH_SIZE']} \\
    kentatsujikawadev/logo-detection-api:latest"""
    return cmd

if __name__ == "__main__":
    print("Memory Configuration Test Patterns for 2GB VPS")
    print("=" * 50)
    
    for config in MEMORY_CONFIGS:
        print(f"\n## {config['name']} Configuration")
        print(f"Description: {config['description']}")
        print(f"Memory Limit: {config['memory']}")
        print(f"Memory Reservation: {config['memory_reservation']}")
        print(f"Batch Settings:")
        for key, value in config['batch_settings'].items():
            print(f"  - {key}: {value}")
        print(f"\nDocker Command:")
        print(generate_docker_command(config))
        print("-" * 50)