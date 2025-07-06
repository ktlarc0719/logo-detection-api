#!/usr/bin/env python3
"""
Simple memory test that can run inside Docker container
"""
import requests
import psutil
import time
import json

def test_batch_detection(batch_size=30):
    """Test batch detection with memory monitoring"""
    
    # Sample URLs for testing
    test_urls = [
        "https://www.google.com",
        "https://www.github.com", 
        "https://www.microsoft.com",
        "https://www.apple.com",
        "https://www.amazon.com"
    ] * (batch_size // 5)
    
    test_urls = test_urls[:batch_size]
    
    print(f"\nTesting with {len(test_urls)} URLs...")
    
    # Memory before
    memory_before = psutil.Process().memory_info().rss / 1024 / 1024
    print(f"Memory before: {memory_before:.2f} MB")
    
    # Make request
    start_time = time.time()
    try:
        response = requests.post(
            "http://localhost:8000/api/v1/detect/batch",
            json={"urls": test_urls},
            timeout=300
        )
        
        elapsed = time.time() - start_time
        
        # Memory after
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024
        memory_used = memory_after - memory_before
        
        print(f"Status: {response.status_code}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Memory after: {memory_after:.2f} MB")
        print(f"Memory used: {memory_used:.2f} MB")
        print(f"Speed: {len(test_urls)/elapsed:.2f} URLs/sec")
        
        if response.status_code == 200:
            result = response.json()
            detected = sum(1 for r in result.get("results", []) if r.get("logos_detected"))
            print(f"Logos detected: {detected}/{len(test_urls)}")
            
    except Exception as e:
        print(f"Error: {e}")

def monitor_memory():
    """Monitor current memory usage"""
    print("\n=== Memory Status ===")
    
    # Process memory
    process = psutil.Process()
    process_memory = process.memory_info()
    print(f"Process RSS: {process_memory.rss / 1024 / 1024:.2f} MB")
    print(f"Process VMS: {process_memory.vms / 1024 / 1024:.2f} MB")
    
    # System memory
    vm = psutil.virtual_memory()
    print(f"\nSystem Total: {vm.total / 1024 / 1024 / 1024:.2f} GB")
    print(f"System Available: {vm.available / 1024 / 1024 / 1024:.2f} GB")
    print(f"System Used: {vm.percent:.1f}%")
    
    # Container limits (if available)
    try:
        with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
            limit = int(f.read().strip())
            if limit < 9223372036854775807:  # Not unlimited
                print(f"\nContainer Memory Limit: {limit / 1024 / 1024 / 1024:.2f} GB")
    except:
        pass

if __name__ == "__main__":
    print("Simple Memory Test for Logo Detection API")
    print("=" * 50)
    
    # Check if API is running
    try:
        health = requests.get("http://localhost:8000/api/v1/health")
        if health.status_code != 200:
            print("API is not healthy!")
            exit(1)
    except:
        print("API is not running!")
        exit(1)
    
    # Monitor initial state
    monitor_memory()
    
    # Run tests with different batch sizes
    for batch_size in [10, 20, 30, 40, 50]:
        test_batch_detection(batch_size)
        time.sleep(5)  # Cool down between tests
        
    # Final memory state
    monitor_memory()