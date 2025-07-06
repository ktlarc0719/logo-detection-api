#!/usr/bin/env python3
"""
Simple performance test script for batch processing API
Tests with current configuration without restarting Docker
"""

import asyncio
import aiohttp
import time
import psutil
import statistics
from typing import List, Dict, Tuple
from tabulate import tabulate
import json
from datetime import datetime

# Test different batch sizes with current configuration
BATCH_SIZES = [5, 10, 20, 30, 50, 100]
NUM_RUNS = 3

# Sample image URLs
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


class SimpleMonitor:
    """Simple CPU and memory monitor"""
    
    def __init__(self):
        self.samples = []
        
    def sample(self):
        """Take a sample of CPU and memory"""
        self.samples.append({
            "cpu": psutil.cpu_percent(interval=0.1),
            "memory_mb": psutil.virtual_memory().used / (1024 * 1024),
            "timestamp": time.time()
        })
    
    def get_stats(self):
        """Get statistics from samples"""
        if not self.samples:
            return {"cpu_avg": 0, "cpu_max": 0, "memory_avg_mb": 0, "memory_max_mb": 0}
            
        cpu_values = [s["cpu"] for s in self.samples]
        memory_values = [s["memory_mb"] for s in self.samples]
        
        return {
            "cpu_avg": statistics.mean(cpu_values),
            "cpu_max": max(cpu_values),
            "cpu_min": min(cpu_values),
            "memory_avg_mb": statistics.mean(memory_values),
            "memory_max_mb": max(memory_values),
            "samples": len(self.samples)
        }
    
    def reset(self):
        """Reset samples"""
        self.samples = []


async def generate_batch(batch_size: int) -> Dict:
    """Generate a batch request"""
    images = []
    for i in range(batch_size):
        url = f"{SAMPLE_IMAGE_URLS[i % len(SAMPLE_IMAGE_URLS)]}?t={int(time.time() * 1000)}&id={i}"
        images.append({
            "id": i,
            "image_url": url
        })
    
    return {
        "batch_id": f"test_batch_{batch_size}_{int(time.time())}",
        "images": images,
        "options": {
            "confidence_threshold": 0.8,
            "max_detections": 10
        }
    }


async def monitor_resources(monitor: SimpleMonitor, stop_event: asyncio.Event):
    """Background task to monitor resources"""
    while not stop_event.is_set():
        monitor.sample()
        await asyncio.sleep(0.5)


async def test_batch_size(api_url: str, batch_size: int, run_num: int) -> Dict:
    """Test a specific batch size"""
    
    monitor = SimpleMonitor()
    stop_event = asyncio.Event()
    
    # Start monitoring
    monitor_task = asyncio.create_task(monitor_resources(monitor, stop_event))
    
    # Generate batch
    batch_data = await generate_batch(batch_size)
    
    # Initial CPU sample
    initial_cpu = psutil.cpu_percent(interval=0.1)
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        
        try:
            async with session.post(
                f"{api_url}/api/v1/process/batch",
                json=batch_data,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                result = await response.json()
                execution_time = time.time() - start_time
                
                # Stop monitoring
                stop_event.set()
                await monitor_task
                
                # Get stats
                stats = monitor.get_stats()
                
                if response.status == 200:
                    return {
                        "batch_size": batch_size,
                        "run": run_num,
                        "execution_time": execution_time,
                        "successful": result.get("successful", 0),
                        "failed": result.get("failed", 0),
                        "throughput": result.get("successful", 0) / execution_time if execution_time > 0 else 0,
                        "initial_cpu": initial_cpu,
                        **stats,
                        "status": "success"
                    }
                else:
                    return {
                        "batch_size": batch_size,
                        "run": run_num,
                        "execution_time": execution_time,
                        "successful": 0,
                        "failed": batch_size,
                        "throughput": 0,
                        "error": f"HTTP {response.status}: {result}",
                        "initial_cpu": initial_cpu,
                        **stats,
                        "status": "error"
                    }
                    
        except Exception as e:
            stop_event.set()
            await monitor_task
            stats = monitor.get_stats()
            
            return {
                "batch_size": batch_size,
                "run": run_num,
                "execution_time": time.time() - start_time,
                "successful": 0,
                "failed": batch_size,
                "throughput": 0,
                "error": str(e),
                "initial_cpu": initial_cpu,
                **stats,
                "status": "error"
            }


async def get_current_config(api_url: str) -> Dict:
    """Get current API configuration"""
    try:
        async with aiohttp.ClientSession() as session:
            # Try to get health endpoint which might have config info
            async with session.get(f"{api_url}/api/v1/health") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("system_info", {})
    except:
        pass
    return {}


async def main():
    """Main test runner"""
    
    api_url = "http://localhost:8000"
    
    print("Simple Performance Test for Batch Processing API")
    print("=" * 80)
    print(f"API URL: {api_url}")
    print(f"System Info:")
    print(f"  CPU Cores: {psutil.cpu_count()}")
    print(f"  CPU Freq: {psutil.cpu_freq().current:.0f} MHz" if psutil.cpu_freq() else "  CPU Freq: Unknown")
    print(f"  Total Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"  Available Memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    print("=" * 80)
    
    # Check API health
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api_url}/api/v1/health") as response:
                if response.status != 200:
                    print(f"API health check failed: {response.status}")
                    return
                health_data = await response.json()
                print(f"\nAPI Status: {health_data.get('status', 'Unknown')}")
    except Exception as e:
        print(f"Cannot connect to API: {e}")
        return
    
    # Get current configuration
    config = await get_current_config(api_url)
    if config:
        print(f"\nCurrent API Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    print(f"\nTesting batch sizes: {BATCH_SIZES}")
    print(f"Runs per batch size: {NUM_RUNS}")
    print("\n")
    
    all_results = []
    
    # Test each batch size
    for batch_size in BATCH_SIZES:
        print(f"\nTesting batch size: {batch_size}")
        batch_results = []
        
        for run in range(NUM_RUNS):
            print(f"  Run {run + 1}/{NUM_RUNS}...", end="", flush=True)
            
            result = await test_batch_size(api_url, batch_size, run + 1)
            batch_results.append(result)
            
            if result["status"] == "success":
                print(f" Done ({result['execution_time']:.2f}s, "
                      f"{result['throughput']:.2f} img/s, "
                      f"CPU: {result['cpu_avg']:.1f}%)")
            else:
                print(f" ERROR: {result.get('error', 'Unknown error')}")
            
            # Wait between runs
            await asyncio.sleep(2)
        
        # Calculate averages for this batch size
        successful_runs = [r for r in batch_results if r["status"] == "success"]
        
        if successful_runs:
            avg_result = {
                "batch_size": batch_size,
                "runs": len(successful_runs),
                "avg_time": statistics.mean([r["execution_time"] for r in successful_runs]),
                "min_time": min([r["execution_time"] for r in successful_runs]),
                "max_time": max([r["execution_time"] for r in successful_runs]),
                "avg_throughput": statistics.mean([r["throughput"] for r in successful_runs]),
                "avg_cpu": statistics.mean([r["cpu_avg"] for r in successful_runs]),
                "max_cpu": max([r["cpu_max"] for r in successful_runs]),
                "avg_memory_mb": statistics.mean([r["memory_avg_mb"] for r in successful_runs]),
                "success_rate": len(successful_runs) / len(batch_results) * 100,
                "raw_results": batch_results
            }
            all_results.append(avg_result)
    
    # Print summary
    print(f"\n\n{'='*80}")
    print("PERFORMANCE TEST SUMMARY")
    print(f"{'='*80}\n")
    
    # Create summary table
    summary_data = []
    for result in all_results:
        summary_data.append([
            result["batch_size"],
            f"{result['runs']}/{NUM_RUNS}",
            f"{result['avg_time']:.2f}",
            f"{result['min_time']:.2f}-{result['max_time']:.2f}",
            f"{result['avg_throughput']:.2f}",
            f"{result['avg_cpu']:.1f}%",
            f"{result['max_cpu']:.1f}%",
            f"{result['avg_memory_mb']:.0f} MB",
            f"{result['success_rate']:.0f}%"
        ])
    
    headers = [
        "Batch Size",
        "Success",
        "Avg Time",
        "Time Range",
        "Throughput\n(img/s)",
        "Avg CPU",
        "Max CPU", 
        "Avg Memory",
        "Success Rate"
    ]
    
    print(tabulate(summary_data, headers=headers, tablefmt="grid"))
    
    # Performance per image analysis
    print(f"\n{'='*80}")
    print("PERFORMANCE PER IMAGE ANALYSIS")
    print(f"{'='*80}\n")
    
    per_image_data = []
    for result in all_results:
        time_per_image = result["avg_time"] / result["batch_size"]
        per_image_data.append([
            result["batch_size"],
            f"{time_per_image:.3f}",
            f"{result['avg_cpu'] / psutil.cpu_count():.1f}%",  # CPU per core
            f"{result['avg_memory_mb'] / result['batch_size']:.1f} MB"
        ])
    
    headers2 = ["Batch Size", "Time/Image (s)", "CPU/Core", "Memory/Image"]
    print(tabulate(per_image_data, headers=headers2, tablefmt="grid"))
    
    # Find optimal batch size
    if all_results:
        # Best throughput
        best_throughput = max(all_results, key=lambda x: x["avg_throughput"])
        
        # Best efficiency (throughput per CPU %)
        efficiency_results = [(r, r["avg_throughput"] / r["avg_cpu"]) for r in all_results if r["avg_cpu"] > 0]
        if efficiency_results:
            best_efficiency = max(efficiency_results, key=lambda x: x[1])
            
            print(f"\n{'='*80}")
            print("RECOMMENDATIONS")
            print(f"{'='*80}")
            print(f"\nBest Throughput:")
            print(f"  Batch Size: {best_throughput['batch_size']}")
            print(f"  Throughput: {best_throughput['avg_throughput']:.2f} images/second")
            print(f"  CPU Usage: {best_throughput['avg_cpu']:.1f}%")
            
            print(f"\nBest Efficiency (Throughput/CPU):")
            print(f"  Batch Size: {best_efficiency[0]['batch_size']}")
            print(f"  Efficiency: {best_efficiency[1]:.3f} images/second per CPU%")
            print(f"  Throughput: {best_efficiency[0]['avg_throughput']:.2f} images/second")
            print(f"  CPU Usage: {best_efficiency[0]['avg_cpu']:.1f}%")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"simple_performance_test_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "system_info": {
                "cpu_cores": psutil.cpu_count(),
                "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            },
            "results": all_results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    asyncio.run(main())