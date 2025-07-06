#!/bin/bash
# VPS Memory Monitoring and Performance Test Script
# This script should be run on the VPS host (not inside Docker)

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
API_URL="http://localhost:8000"
MANAGER_URL="http://localhost:8080"
TEST_SIZES=(5 10 20 30 50)
NUM_RUNS=3

# Sample URLs from simple_performance_test.py
SAMPLE_URLS='[
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
    "https://static.mercdn.net/item/detail/orig/photos/m20928256580_1.jpg?1750573461"
]'

echo -e "${GREEN}🚀 VPS Memory Monitoring and Performance Test${NC}"
echo "================================================"

# Check if API is running
echo -e "${YELLOW}Checking API health...${NC}"
if ! curl -s "$API_URL/api/v1/health" > /dev/null; then
    echo -e "${RED}❌ API is not running!${NC}"
    exit 1
fi
echo -e "${GREEN}✓ API is healthy${NC}"

# Create monitoring script
cat << 'MONITOR_SCRIPT' > /tmp/monitor_memory.sh
#!/bin/bash
LOG_FILE="/tmp/memory_monitor.log"
echo "timestamp,system_total_mb,system_used_mb,system_free_mb,docker_mem_usage,docker_mem_limit" > $LOG_FILE

while true; do
    # System memory
    MEM_INFO=$(free -m | grep Mem | awk '{print $2","$3","$4}')
    
    # Docker stats
    DOCKER_STATS=$(docker stats logo-detection-api --no-stream --format "{{.MemUsage}}" | sed 's/[^0-9.,]//g')
    
    # Timestamp
    TIMESTAMP=$(date +%s)
    
    echo "$TIMESTAMP,$MEM_INFO,$DOCKER_STATS" >> $LOG_FILE
    sleep 2
done
MONITOR_SCRIPT

chmod +x /tmp/monitor_memory.sh

# Create test script
cat << 'TEST_SCRIPT' > /tmp/run_performance_test.sh
#!/bin/bash
source /tmp/test_config.sh

echo "Starting performance tests..."
echo "timestamp,batch_size,run,duration_sec,urls_per_sec,success,error" > /tmp/test_results.csv

for size in "${TEST_SIZES[@]}"; do
    echo ""
    echo "Testing batch size: $size"
    
    # Extract URLs for this batch size
    URLS_JSON=$(echo "$SAMPLE_URLS" | jq -c ".[:$size]")
    
    for run in $(seq 1 $NUM_RUNS); do
        echo "  Run $run/$NUM_RUNS..."
        
        # Record start time
        START_TIME=$(date +%s.%N)
        
        # Make request
        RESPONSE=$(curl -s -X POST "$API_URL/api/v1/detect/batch" \
            -H "Content-Type: application/json" \
            -d "{\"urls\": $URLS_JSON}" \
            -w "\n%{http_code}" \
            --max-time 300)
        
        # Extract status code
        HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
        RESPONSE_BODY=$(echo "$RESPONSE" | head -n-1)
        
        # Calculate duration
        END_TIME=$(date +%s.%N)
        DURATION=$(echo "$END_TIME - $START_TIME" | bc)
        
        # Calculate URLs per second
        if [ "$HTTP_CODE" = "200" ]; then
            URLS_PER_SEC=$(echo "scale=2; $size / $DURATION" | bc)
            SUCCESS=1
            ERROR=""
        else
            URLS_PER_SEC=0
            SUCCESS=0
            ERROR="HTTP_$HTTP_CODE"
        fi
        
        # Log result
        echo "$START_TIME,$size,$run,$DURATION,$URLS_PER_SEC,$SUCCESS,$ERROR" >> /tmp/test_results.csv
        
        # Cool down between runs
        sleep 10
    done
    
    # Longer cool down between different batch sizes
    sleep 20
done

echo ""
echo "Test completed! Results saved to /tmp/test_results.csv"
TEST_SCRIPT

chmod +x /tmp/run_performance_test.sh

# Save test configuration
cat << EOF > /tmp/test_config.sh
API_URL="$API_URL"
TEST_SIZES=(${TEST_SIZES[@]})
NUM_RUNS=$NUM_RUNS
SAMPLE_URLS='$SAMPLE_URLS'
EOF

# Create results analysis script
cat << 'ANALYSIS_SCRIPT' > /tmp/analyze_results.sh
#!/bin/bash
echo ""
echo "=== Performance Test Results ==="
echo ""

# Analyze test results
echo "Batch Size | Avg Duration (s) | Avg URLs/sec | Success Rate"
echo "-----------|------------------|--------------|-------------"

for size in "${TEST_SIZES[@]}"; do
    STATS=$(awk -F',' -v size="$size" '
        $2 == size && NR > 1 {
            total_duration += $4
            total_speed += $5
            total_success += $6
            count++
        }
        END {
            if (count > 0) {
                avg_duration = total_duration / count
                avg_speed = total_speed / count
                success_rate = (total_success / count) * 100
                printf "%10d | %16.2f | %12.2f | %11.0f%%\n", 
                    size, avg_duration, avg_speed, success_rate
            }
        }
    ' /tmp/test_results.csv)
    echo "$STATS"
done

echo ""
echo "=== Memory Usage Analysis ==="
echo ""

# Analyze memory usage
awk -F',' 'NR > 1 {
    if (NR == 2) {
        min_sys = max_sys = $3
        min_docker = max_docker = $5
    } else {
        if ($3 < min_sys) min_sys = $3
        if ($3 > max_sys) max_sys = $3
        if ($5 < min_docker) min_docker = $5
        if ($5 > max_docker) max_docker = $5
    }
    total_sys += $3
    total_docker += $5
    count++
}
END {
    avg_sys = total_sys / count
    avg_docker = total_docker / count
    print "System Memory (MB):"
    printf "  Min: %.0f, Max: %.0f, Avg: %.0f\n", min_sys, max_sys, avg_sys
    print "Docker Memory (MB):"
    printf "  Min: %.0f, Max: %.0f, Avg: %.0f\n", min_docker, max_docker, avg_docker
}' /tmp/memory_monitor.log

# Show current configuration
echo ""
echo "=== Current Configuration ==="
curl -s "$MANAGER_URL/config" | jq -r 'to_entries | map("\(.key): \(.value)") | .[]' | grep -E "(MAX_|MEMORY)"
ANALYSIS_SCRIPT

chmod +x /tmp/analyze_results.sh

# Instructions
echo ""
echo -e "${GREEN}Setup complete! Run the following commands in separate terminals:${NC}"
echo ""
echo -e "${YELLOW}Terminal 1 (Memory Monitor):${NC}"
echo "  /tmp/monitor_memory.sh"
echo ""
echo -e "${YELLOW}Terminal 2 (Performance Test):${NC}"
echo "  /tmp/run_performance_test.sh"
echo ""
echo -e "${YELLOW}After tests complete, analyze results:${NC}"
echo "  /tmp/analyze_results.sh"
echo ""
echo -e "${GREEN}Or use tmux to run all at once:${NC}"
echo "  tmux new-session -d -s monitor '/tmp/monitor_memory.sh'"
echo "  tmux new-session -d -s test '/tmp/run_performance_test.sh'"
echo "  tmux attach -t test"
echo ""
echo -e "${YELLOW}Monitor real-time stats:${NC}"
echo "  watch -n 1 'docker stats --no-stream'"