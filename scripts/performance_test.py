#!/usr/bin/env python3
"""
Performance test script for batch processing API
Tests different configurations for 2-core 2GB CPU environment
"""

import os
import sys
import time
import json
import psutil
import asyncio
import aiohttp
import statistics
from datetime import datetime
from typing import List, Dict, Tuple
from tabulate import tabulate
import subprocess
from contextlib import contextmanager

# Test configurations
TEST_CONFIGS = [
    # (MAX_CONCURRENT_DETECTIONS, MAX_CONCURRENT_DOWNLOADS, MAX_BATCH_SIZE)
    (1, 5, 10),    # Very conservative
    (2, 5, 10),    # 2 cores, conservative downloads
    (2, 10, 20),   # 2 cores, moderate
    (3, 10, 30),   # Slight oversubscription
    (4, 15, 40),   # More oversubscription
    (2, 20, 50),   # High downloads, moderate detection
    (2, 30, 100),  # Very high downloads, max batch
]

# Sample image URLs (you can replace with actual URLs)
SAMPLE_IMAGE_URLS = [
    "https://static.mercdn.net/item/detail/orig/photos/m47628860505_1.jpg?1740720390",
    "https://static.mercdn.net/item/detail/orig/photos/m44609767648_1.jpg?1750939240",
    "https://static.mercdn.net/item/detail/orig/photos/m88478925057_1.jpg?1662265812",
    "https://static.mercdn.net/item/detail/orig/photos/m96001970769_1.jpg?1750953394",
    "https://static.mercdn.net/item/detail/orig/photos/m48931146074_1.jpg?1750738532",
    "https://static.mercdn.net/item/detail/orig/photos/m38469222607_1.jpg?1750821382",
    "https://static.mercdn.net/item/detail/orig/photos/m88549534663_1.jpg?1750020537",
    "https://static.mercdn.net/item/detail/orig/photos/m78444107154_1.jpg?1729065587",
    "https://static.mercdn.net/item/detail/orig/photos/m99866594532_1.jpg?1697632923",
    "https://static.mercdn.net/item/detail/orig/photos/m21519488926_1.jpg?1748441244",
    "https://static.mercdn.net/item/detail/orig/photos/m30659710715_1.jpg?1704007113",
    "https://static.mercdn.net/item/detail/orig/photos/m83639912369_1.jpg?1704719348",
    "https://static.mercdn.net/item/detail/orig/photos/m96459104257_1.jpg?1750965433",
    "https://static.mercdn.net/item/detail/orig/photos/m13795698389_1.jpg?1731027420",
    "https://static.mercdn.net/item/detail/orig/photos/m62172900593_1.jpg?1734682836",
    "https://static.mercdn.net/item/detail/orig/photos/m67174004244_1.jpg?1750520838",
    "https://static.mercdn.net/item/detail/orig/photos/m40980107539_1.jpg?1724313069",
    "https://static.mercdn.net/item/detail/orig/photos/m48554653940_1.jpg?1748833830",
    "https://static.mercdn.net/item/detail/orig/photos/m54380931388_1.jpg?1743415726",
    "https://static.mercdn.net/item/detail/orig/photos/m72310854705_1.jpg?1749858677",
    "https://static.mercdn.net/item/detail/orig/photos/m37916245048_1.jpg?1746605742",
    "https://static.mercdn.net/item/detail/orig/photos/m13353081345_1.jpg?1746430458",
    "https://static.mercdn.net/item/detail/orig/photos/m91602011457_1.jpg?1750915455",
    "https://static.mercdn.net/item/detail/orig/photos/m12525301933_1.jpg?1742560720",
    "https://static.mercdn.net/item/detail/orig/photos/m45890664235_1.jpg?1750681098",
    "https://static.mercdn.net/item/detail/orig/photos/m76174449115_1.jpg?1750891032",
    "https://static.mercdn.net/item/detail/orig/photos/m19394384485_1.jpg?1750473862",
    "https://static.mercdn.net/item/detail/orig/photos/m92991245040_1.jpg?1721580506",
    "https://static.mercdn.net/item/detail/orig/photos/m37886971886_1.jpg?1747112207",
    "https://static.mercdn.net/item/detail/orig/photos/m43410789903_1.jpg?1710687340",
    "https://static.mercdn.net/item/detail/orig/photos/m63234467141_1.jpg?1748840012",
    "https://static.mercdn.net/item/detail/orig/photos/m88390907152_1.jpg?1750222315",
    "https://auctions.c.yimg.jp/images.auctions.yahoo.co.jp/image/dr000/auc0205/users/3748ec9aace84d1e0e684d5abc3597b92f7edd2e/i-img1200x900-1621327804u0nhtq584531.jpg",
    "https://static.mercdn.net/item/detail/orig/photos/m17019783622_1.jpg?1750845911",
    "https://static.mercdn.net/item/detail/orig/photos/m46628980792_1.jpg?1750838206",
    "https://static.mercdn.net/item/detail/orig/photos/m69922343944_1.jpg?1742221273",
    "https://static.mercdn.net/item/detail/orig/photos/m58141064733_1.jpg?1704708534",
    "https://static.mercdn.net/item/detail/orig/photos/m31544224634_1.jpg?1746570216",
    "https://static.mercdn.net/item/detail/orig/photos/m98903257587_1.jpg?1737184671",
    "https://static.mercdn.net/item/detail/orig/photos/m29569588328_1.jpg?1727088087",
    "https://static.mercdn.net/item/detail/orig/photos/m52187686254_1.jpg?1735913244",
    "https://static.mercdn.net/item/detail/orig/photos/m30649984111_1.jpg?1727088111",
    "https://static.mercdn.net/item/detail/orig/photos/m46862449755_1.jpg?1642093353",
    "https://static.mercdn.net/item/detail/orig/photos/m48640765610_1.jpg?1750869681",
    "https://static.mercdn.net/item/detail/orig/photos/m65837630186_1.jpg?1750592416",
    "https://static.mercdn.net/item/detail/orig/photos/m68314314269_1.jpg?1748249652",
    "https://static.mercdn.net/item/detail/orig/photos/m69474390804_1.jpg?1750898243",
    "https://static.mercdn.net/item/detail/orig/photos/m42093424538_1.jpg?1735970182",
    "https://static.mercdn.net/item/detail/orig/photos/m10888707318_1.jpg?1750563929",
    "https://static.mercdn.net/item/detail/orig/photos/m20928256580_1.jpg?1750573461",
    "https://static.mercdn.net/item/detail/orig/photos/m43062944045_1.jpg?1748951122",
    "https://static.mercdn.net/item/detail/orig/photos/m33939039293_1.jpg?1749109428",
    "https://static.mercdn.net/item/detail/orig/photos/m47776412904_1.jpg?1750651916",
    "https://static.mercdn.net/item/detail/orig/photos/m20257609794_1.jpg?1750908779",
    "https://static.mercdn.net/item/detail/orig/photos/m50478878700_1.jpg?1744008565",
    "https://static.mercdn.net/item/detail/orig/photos/m11932961496_1.jpg?1749888365",
    "https://static.mercdn.net/item/detail/orig/photos/m86009206246_1.jpg?1745666744",
    "https://static.mercdn.net/item/detail/orig/photos/m20720761880_1.jpg?1665988065",
    "https://static.mercdn.net/item/detail/orig/photos/m12836353468_1.jpg?1747197445",
    "https://static.mercdn.net/item/detail/orig/photos/m38274903086_1.jpg?1750918333",
    "https://static.mercdn.net/item/detail/orig/photos/m48781357754_1.jpg?1744338814",
    "https://static.mercdn.net/item/detail/orig/photos/m22127715142_1.jpg?1750916684",
    "https://static.mercdn.net/item/detail/orig/photos/m96044664459_1.jpg?1748874129",
    "https://static.mercdn.net/item/detail/orig/photos/m11630833058_1.jpg?1734667254",
    "https://static.mercdn.net/item/detail/orig/photos/m54843234090_1.jpg?1747284221",
    "https://static.mercdn.net/item/detail/orig/photos/m99115166914_1.jpg?1723525922",
    "https://static.mercdn.net/item/detail/orig/photos/m30833950659_1.jpg?1747040091",
    "https://static.mercdn.net/item/detail/orig/photos/m24640008395_1.jpg?1729658405",
    "https://static.mercdn.net/item/detail/orig/photos/m67319596504_1.jpg?1750508323",
    "https://static.mercdn.net/item/detail/orig/photos/m89482778482_1.jpg?1750908660",
    "https://static.mercdn.net/item/detail/orig/photos/m54998414705_1.jpg?1750766797",
    "https://static.mercdn.net/item/detail/orig/photos/m50158745398_1.jpg?1750923989",
    "https://static.mercdn.net/item/detail/orig/photos/m32864529462_1.jpg?1745754877",
    "https://static.mercdn.net/item/detail/orig/photos/m90399926206_1.jpg?1750822536",
    "https://static.mercdn.net/item/detail/orig/photos/m51809880407_1.jpg?1750860349",
    "https://static.mercdn.net/item/detail/orig/photos/m77863775378_1.jpg?1729405978",
    "https://static.mercdn.net/item/detail/orig/photos/m81400416745_1.jpg?1750925387",
    "https://static.mercdn.net/item/detail/orig/photos/m74348380430_1.jpg?1750908097",
    "https://static.mercdn.net/item/detail/orig/photos/m42477103412_1.jpg?1750574518",
    "https://static.mercdn.net/item/detail/orig/photos/m46255573055_1.jpg?1745727992",
    "https://static.mercdn.net/item/detail/orig/photos/m62033160931_1.jpg?1748938263",
    "https://static.mercdn.net/item/detail/orig/photos/m35087703309_1.jpg?1742816249",
    "https://static.mercdn.net/item/detail/orig/photos/m78809802437_1.jpg?1750922177",
    "https://static.mercdn.net/item/detail/orig/photos/m68596413416_1.jpg?1746443828",
    "https://static.mercdn.net/item/detail/orig/photos/m25480680819_1.jpg?1747102431",
    "https://static.mercdn.net/item/detail/orig/photos/m33871202651_1.jpg?1749025884",
    "https://static.mercdn.net/item/detail/orig/photos/m38841807307_1.jpg?1737770700",
    "https://static.mercdn.net/item/detail/orig/photos/m77911149520_1.jpg?1729708060",
    "https://static.mercdn.net/item/detail/orig/photos/m24901859512_1.jpg?1749718677",
    "https://static.mercdn.net/item/detail/orig/photos/m89914235099_1.jpg?1750830249",
    "https://static.mercdn.net/item/detail/orig/photos/m84303504536_1.jpg?1744025021",
    "https://static.mercdn.net/item/detail/orig/photos/m35236457738_1.jpg?1749891379",
    "https://static.mercdn.net/item/detail/orig/photos/m65054514001_1.jpg?1749737881",
    "https://static.mercdn.net/item/detail/orig/photos/m62695654111_1.jpg?1714443418",
    "https://static.mercdn.net/item/detail/orig/photos/m50578354568_1.jpg?1743788948",
    "https://static.mercdn.net/item/detail/orig/photos/m54128539773_1.jpg?1622825534",
    "https://static.mercdn.net/item/detail/orig/photos/m34160011091_1.jpg?1694982289",
    "https://static.mercdn.net/item/detail/orig/photos/m22430349305_1.jpg?1750648148",
    "https://static.mercdn.net/item/detail/orig/photos/m80918509805_1.jpg?1749565235",
    "https://static.mercdn.net/item/detail/orig/photos/m20795476172_1.jpg?1749611258",
]

class PerformanceMonitor:
    """Monitor CPU and memory usage during test"""
    
    def __init__(self):
        self.cpu_samples = []
        self.memory_samples = []
        self.monitoring = False
        self.monitor_task = None
        
    async def start_monitoring(self):
        """Start monitoring system resources"""
        self.monitoring = True
        self.cpu_samples = []
        self.memory_samples = []
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        
    async def stop_monitoring(self):
        """Stop monitoring and return statistics"""
        self.monitoring = False
        if self.monitor_task:
            await self.monitor_task
            
        return {
            "cpu": {
                "avg": statistics.mean(self.cpu_samples) if self.cpu_samples else 0,
                "max": max(self.cpu_samples) if self.cpu_samples else 0,
                "min": min(self.cpu_samples) if self.cpu_samples else 0,
                "samples": len(self.cpu_samples)
            },
            "memory": {
                "avg_mb": statistics.mean(self.memory_samples) if self.memory_samples else 0,
                "max_mb": max(self.memory_samples) if self.memory_samples else 0,
                "min_mb": min(self.memory_samples) if self.memory_samples else 0,
            }
        }
        
    async def _monitor_loop(self):
        """Monitor loop that samples system resources"""
        while self.monitoring:
            # CPU usage (average across all cores)
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_samples.append(cpu_percent)
            
            # Memory usage in MB
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            self.memory_samples.append(memory_mb)
            
            await asyncio.sleep(0.5)  # Sample every 500ms


@contextmanager
def set_env_vars(**kwargs):
    """Context manager to temporarily set environment variables"""
    old_values = {}
    for key, value in kwargs.items():
        old_values[key] = os.environ.get(key)
        os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


async def generate_batch_request(batch_size: int, batch_id: str) -> Dict:
    """Generate a batch request with sample images"""
    images = []
    for i in range(batch_size):
        # Use different URLs to avoid caching
        url_index = i % len(SAMPLE_IMAGE_URLS)
        base_url = SAMPLE_IMAGE_URLS[url_index]
        # Add timestamp to make URL unique
        url = f"{base_url}?t={int(time.time() * 1000)}"
        
        images.append({
            "id": i,
            "image_url": url
        })
    
    return {
        "batch_id": batch_id,
        "images": images,
        "options": {
            "confidence_threshold": 0.65,
            "max_detections": 10
        }
    }


async def run_batch_test(api_url: str, batch_size: int, test_name: str) -> Tuple[float, Dict]:
    """Run a single batch test and return execution time and response"""
    
    batch_request = await generate_batch_request(batch_size, f"test_{test_name}_{int(time.time())}")
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        
        try:
            async with session.post(
                f"{api_url}/api/v1/process/batch",
                json=batch_request,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                result = await response.json()
                execution_time = time.time() - start_time
                
                if response.status == 200:
                    return execution_time, result
                else:
                    return execution_time, {"error": f"HTTP {response.status}", "detail": result}
                    
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            return execution_time, {"error": "Timeout", "execution_time": execution_time}
        except Exception as e:
            execution_time = time.time() - start_time
            return execution_time, {"error": str(e), "execution_time": execution_time}


async def test_configuration(
    api_url: str,
    max_concurrent_detections: int,
    max_concurrent_downloads: int,
    max_batch_size: int,
    num_runs: int = 3
) -> Dict:
    """Test a specific configuration multiple times"""
    
    print(f"\n{'='*80}")
    print(f"Testing Configuration:")
    print(f"  MAX_CONCURRENT_DETECTIONS: {max_concurrent_detections}")
    print(f"  MAX_CONCURRENT_DOWNLOADS: {max_concurrent_downloads}")
    print(f"  MAX_BATCH_SIZE: {max_batch_size}")
    print(f"{'='*80}\n")
    
    # Test with different batch sizes
    batch_sizes = [10, 20, max_batch_size] if max_batch_size > 20 else [max_batch_size]
    results = []
    
    for batch_size in batch_sizes:
        if batch_size > max_batch_size:
            continue
            
        print(f"\nTesting batch size: {batch_size}")
        
        run_results = []
        monitor = PerformanceMonitor()
        
        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}...", end="", flush=True)
            
            # Start monitoring
            await monitor.start_monitoring()
            
            # Run the test
            execution_time, response = await run_batch_test(
                api_url, 
                batch_size, 
                f"det{max_concurrent_detections}_dl{max_concurrent_downloads}_batch{batch_size}_run{run}"
            )
            
            # Stop monitoring
            stats = await monitor.stop_monitoring()
            
            # Calculate throughput
            if "error" not in response:
                successful = response.get("successful", 0)
                failed = response.get("failed", 0)
                throughput = successful / execution_time if execution_time > 0 else 0
            else:
                successful = 0
                failed = batch_size
                throughput = 0
            
            run_result = {
                "run": run + 1,
                "batch_size": batch_size,
                "execution_time": execution_time,
                "successful": successful,
                "failed": failed,
                "throughput": throughput,
                "cpu_avg": stats["cpu"]["avg"],
                "cpu_max": stats["cpu"]["max"],
                "memory_avg_mb": stats["memory"]["avg_mb"],
                "memory_max_mb": stats["memory"]["max_mb"],
                "error": response.get("error") if "error" in response else None
            }
            
            run_results.append(run_result)
            
            if "error" in response:
                print(f" ERROR: {response['error']}")
            else:
                print(f" Done ({execution_time:.2f}s, {throughput:.2f} img/s)")
            
            # Small delay between runs
            await asyncio.sleep(2)
        
        # Calculate averages for this batch size
        avg_result = {
            "config": f"D:{max_concurrent_detections}/DL:{max_concurrent_downloads}/B:{max_batch_size}",
            "batch_size": batch_size,
            "avg_time": statistics.mean([r["execution_time"] for r in run_results]),
            "avg_throughput": statistics.mean([r["throughput"] for r in run_results]),
            "avg_cpu": statistics.mean([r["cpu_avg"] for r in run_results]),
            "max_cpu": max([r["cpu_max"] for r in run_results]),
            "avg_memory_mb": statistics.mean([r["memory_avg_mb"] for r in run_results]),
            "max_memory_mb": max([r["memory_max_mb"] for r in run_results]),
            "success_rate": statistics.mean([r["successful"] / batch_size * 100 for r in run_results]),
            "runs": run_results
        }
        
        results.append(avg_result)
    
    return {
        "config": {
            "max_concurrent_detections": max_concurrent_detections,
            "max_concurrent_downloads": max_concurrent_downloads,
            "max_batch_size": max_batch_size
        },
        "results": results
    }


async def main():
    """Main test runner"""
    
    # Check if API is running
    api_url = os.getenv("API_URL", "http://localhost:8000")
    
    print(f"Testing API at: {api_url}")
    print(f"System Info:")
    print(f"  CPU Cores: {psutil.cpu_count()}")
    print(f"  Total Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"  Available Memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    
    # Check API health
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api_url}/api/v1/health") as response:
                if response.status != 200:
                    print(f"API health check failed: {response.status}")
                    return
    except Exception as e:
        print(f"Cannot connect to API: {e}")
        print("Please ensure the API is running with: ./scripts/docker-run.sh")
        return
    
    # Store all results
    all_results = []
    
    # Test each configuration
    for det, dl, batch in TEST_CONFIGS:
        # Stop existing container
        subprocess.run(["docker", "stop", "logo-detection-api"], 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["docker", "rm", "logo-detection-api"], 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Set environment variables and restart container
        with set_env_vars(
            MAX_CONCURRENT_DETECTIONS=det,
            MAX_CONCURRENT_DOWNLOADS=dl,
            MAX_BATCH_SIZE=batch
        ):
            print(f"\nStarting container with new configuration...")
            
            # Start container
            result = subprocess.run(
                ["./scripts/docker-run.sh"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"Failed to start container: {result.stderr}")
                continue
            
            # Wait for API to be ready
            print("Waiting for API to be ready...")
            await asyncio.sleep(5)
            
            # Run tests for this configuration
            config_results = await test_configuration(api_url, det, dl, batch)
            all_results.append(config_results)
            
            # Small delay before next configuration
            await asyncio.sleep(2)
    
    # Print summary table
    print(f"\n\n{'='*80}")
    print("PERFORMANCE TEST SUMMARY")
    print(f"{'='*80}\n")
    
    # Create summary table
    summary_data = []
    for config_result in all_results:
        config = config_result["config"]
        for result in config_result["results"]:
            summary_data.append([
                f"D:{config['max_concurrent_detections']}/DL:{config['max_concurrent_downloads']}",
                result["batch_size"],
                f"{result['avg_time']:.2f}",
                f"{result['avg_throughput']:.2f}",
                f"{result['avg_cpu']:.1f}%",
                f"{result['max_cpu']:.1f}%",
                f"{result['avg_memory_mb']:.0f}",
                f"{result['max_memory_mb']:.0f}",
                f"{result['success_rate']:.1f}%"
            ])
    
    headers = [
        "Config (Det/DL)",
        "Batch Size",
        "Avg Time (s)",
        "Throughput (img/s)",
        "Avg CPU",
        "Max CPU",
        "Avg Mem (MB)",
        "Max Mem (MB)",
        "Success Rate"
    ]
    
    print(tabulate(summary_data, headers=headers, tablefmt="grid"))
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"performance_test_results_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "system_info": {
                "cpu_cores": psutil.cpu_count(),
                "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            },
            "test_configs": TEST_CONFIGS,
            "results": all_results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Find optimal configuration
    best_config = None
    best_throughput = 0
    
    for config_result in all_results:
        config = config_result["config"]
        for result in config_result["results"]:
            if result["avg_throughput"] > best_throughput and result["success_rate"] > 95:
                best_throughput = result["avg_throughput"]
                best_config = {
                    "config": config,
                    "batch_size": result["batch_size"],
                    "throughput": result["avg_throughput"],
                    "cpu": result["avg_cpu"]
                }
    
    if best_config:
        print(f"\n{'='*80}")
        print("RECOMMENDED CONFIGURATION FOR 2-CORE 2GB SYSTEM:")
        print(f"{'='*80}")
        print(f"MAX_CONCURRENT_DETECTIONS: {best_config['config']['max_concurrent_detections']}")
        print(f"MAX_CONCURRENT_DOWNLOADS: {best_config['config']['max_concurrent_downloads']}")
        print(f"MAX_BATCH_SIZE: {best_config['config']['max_batch_size']}")
        print(f"Optimal batch size: {best_config['batch_size']}")
        print(f"Expected throughput: {best_config['throughput']:.2f} images/second")
        print(f"Average CPU usage: {best_config['cpu']:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())