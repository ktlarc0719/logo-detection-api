#!/bin/bash
# VPS Configuration Performance Test
# Tests different configurations by restarting container with new settings

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
MANAGER_URL="http://localhost:8080"
API_URL="http://localhost:8000"

# Test configurations (from performance_test.py)
declare -A CONFIG_1=(
    ["name"]="Conservative"
    ["MAX_CONCURRENT_DETECTIONS"]="1"
    ["MAX_CONCURRENT_DOWNLOADS"]="5"
    ["MAX_BATCH_SIZE"]="10"
    ["memory"]="1.2g"
    ["memory_reservation"]="800m"
)

declare -A CONFIG_2=(
    ["name"]="Balanced"
    ["MAX_CONCURRENT_DETECTIONS"]="2"
    ["MAX_CONCURRENT_DOWNLOADS"]="15"
    ["MAX_BATCH_SIZE"]="30"
    ["memory"]="1.5g"
    ["memory_reservation"]="1g"
)

declare -A CONFIG_3=(
    ["name"]="Performance"
    ["MAX_CONCURRENT_DETECTIONS"]="3"
    ["MAX_CONCURRENT_DOWNLOADS"]="20"
    ["MAX_BATCH_SIZE"]="40"
    ["memory"]="1.7g"
    ["memory_reservation"]="1.2g"
)

# Sample URLs
SAMPLE_URLS='["https://static.mercdn.net/item/detail/orig/photos/m47628860505_1.jpg?1740720390","https://static.mercdn.net/item/detail/orig/photos/m44609767648_1.jpg?1750939240","https://static.mercdn.net/item/detail/orig/photos/m88478925057_1.jpg?1662265812","https://static.mercdn.net/item/detail/orig/photos/m96001970769_1.jpg?1750953394","https://static.mercdn.net/item/detail/orig/photos/m48931146074_1.jpg?1740738532","https://static.mercdn.net/item/detail/orig/photos/m38469222607_1.jpg?1750821382","https://static.mercdn.net/item/detail/orig/photos/m88549534663_1.jpg?1750020537","https://static.mercdn.net/item/detail/orig/photos/m78444107154_1.jpg?1729065587","https://static.mercdn.net/item/detail/orig/photos/m99866594532_1.jpg?1697632923","https://static.mercdn.net/item/detail/orig/photos/m21519488926_1.jpg?1748441244","https://static.mercdn.net/item/detail/orig/photos/m30659710715_1.jpg?1704007113","https://static.mercdn.net/item/detail/orig/photos/m83639912369_1.jpg?1704719348","https://static.mercdn.net/item/detail/orig/photos/m96459104257_1.jpg?1750965433","https://static.mercdn.net/item/detail/orig/photos/m13795698389_1.jpg?1731027420","https://static.mercdn.net/item/detail/orig/photos/m62172900593_1.jpg?1734682836","https://static.mercdn.net/item/detail/orig/photos/m67174004244_1.jpg?1750520838","https://static.mercdn.net/item/detail/orig/photos/m40980107539_1.jpg?1724313069","https://static.mercdn.net/item/detail/orig/photos/m48554653940_1.jpg?1748833830","https://static.mercdn.net/item/detail/orig/photos/m54380931388_1.jpg?1743415726","https://static.mercdn.net/item/detail/orig/photos/m72310854705_1.jpg?1749858677","https://static.mercdn.net/item/detail/orig/photos/m37916245048_1.jpg?1746605742","https://static.mercdn.net/item/detail/orig/photos/m13353081345_1.jpg?1746430458","https://static.mercdn.net/item/detail/orig/photos/m91602011457_1.jpg?1750915455","https://static.mercdn.net/item/detail/orig/photos/m12525301933_1.jpg?1742560720","https://static.mercdn.net/item/detail/orig/photos/m45890664235_1.jpg?1750681098","https://static.mercdn.net/item/detail/orig/photos/m76174449115_1.jpg?1750891032","https://static.mercdn.net/item/detail/orig/photos/m19394384485_1.jpg?1750473862","https://static.mercdn.net/item/detail/orig/photos/m92991245040_1.jpg?1721580506","https://static.mercdn.net/item/detail/orig/photos/m37886971886_1.jpg?1747112207","https://static.mercdn.net/item/detail/orig/photos/m43410789903_1.jpg?1710687340"]'

echo -e "${GREEN}🚀 VPS Configuration Performance Test${NC}"
echo "================================================"

# Function to update configuration
update_config() {
    local -n config=$1
    echo -e "${YELLOW}Updating configuration: ${config[name]}${NC}"
    
    # Update config via manager API
    curl -X POST "$MANAGER_URL/config" \
        -H "Content-Type: application/json" \
        -d "{
            \"MAX_CONCURRENT_DETECTIONS\": \"${config[MAX_CONCURRENT_DETECTIONS]}\",
            \"MAX_CONCURRENT_DOWNLOADS\": \"${config[MAX_CONCURRENT_DOWNLOADS]}\",
            \"MAX_BATCH_SIZE\": \"${config[MAX_BATCH_SIZE]}\"
        }"
    echo ""
}

# Function to restart container with new memory settings
restart_container() {
    local -n config=$1
    echo -e "${YELLOW}Restarting container with memory settings...${NC}"
    
    curl -X POST "$MANAGER_URL/deploy"
    
    # Wait for container to be ready
    echo "Waiting for API to be ready..."
    sleep 10
    
    # Check health
    for i in {1..30}; do
        if curl -s "$API_URL/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ API is ready${NC}"
            return 0
        fi
        sleep 2
    done
    
    echo -e "${RED}❌ API failed to start${NC}"
    return 1
}

# Function to run performance test
run_test() {
    local -n config=$1
    local batch_size=$2
    local result_file="/tmp/test_${config[name]}_${batch_size}_$(date +%Y%m%d_%H%M%S).json"
    
    echo "  Testing batch size: $batch_size"
    
    # Build URL array for batch using Python
    local urls=$(python3 -c "
import json
urls = $SAMPLE_URLS
print(json.dumps(urls[:$batch_size]))
")
    
    # Run test with timing
    local start_time=$(date +%s.%N)
    
    # Build request body for /api/v1/process/batch
    local images_array=""
    for ((i=0; i<batch_size; i++)); do
        if [ $i -gt 0 ]; then
            images_array+=","
        fi
        images_array+="{\"id\": \"img$i\", \"image_url\": $(echo "$urls" | python3 -c "import json, sys; urls=json.load(sys.stdin); print(json.dumps(urls[$i]))")}"
    done
    
    local request_body="{
        \"batch_id\": \"test-${config[name]}-${batch_size}-$(date +%s)\",
        \"images\": [$images_array],
        \"options\": {
            \"confidence_threshold\": 0.5,
            \"max_detections\": 10,
            \"enable_brand_normalization\": true,
            \"enable_category_classification\": true
        }
    }"
    
    # Make the API call
    local http_response=$(curl -s -X POST "$API_URL/api/v1/process/batch" \
        -H "Content-Type: application/json" \
        -d "$request_body" \
        -o /tmp/response_body.json \
        -w '{"http_code": %{http_code}, "time_total": %{time_total}}')
    
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc)
    
    # Parse HTTP response
    local http_code=$(echo "$http_response" | python3 -c "import json, sys; print(json.load(sys.stdin)['http_code'])")
    
    # Create proper JSON result file
    cat > "$result_file" <<EOF
{
    "config": "${config[name]}",
    "batch_size": $batch_size,
    "duration": $duration,
    "http_code": $http_code,
    "response": $(cat /tmp/response_body.json 2>/dev/null || echo '{}')
}
EOF
    
    echo "    Duration: ${duration}s"
    echo "    HTTP Code: ${http_code}"
    
    # Cool down
    sleep 5
}

# Function to monitor memory during test
monitor_memory() {
    local config_name=$1
    local log_file="/tmp/memory_${config_name}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "timestamp,system_used_mb,docker_mem_mb" > "$log_file"
    
    while true; do
        # System memory
        local mem_used=$(free -m | grep Mem | awk '{print $3}')
        
        # Docker memory (extract just the number)
        local docker_mem=$(docker stats logo-detection-api --no-stream --format "{{.MemUsage}}" | awk -F'/' '{print $1}' | sed 's/[^0-9.]//g')
        
        echo "$(date +%s),$mem_used,$docker_mem" >> "$log_file"
        sleep 2
    done
}

# Main test execution
echo -e "${YELLOW}Starting configuration tests...${NC}"
echo ""

# Create results directory
mkdir -p /tmp/perf_results

# Test each configuration
for config_var in CONFIG_1 CONFIG_2 CONFIG_3; do
    declare -n current_config=$config_var
    
    echo ""
    echo -e "${GREEN}Testing Configuration: ${current_config[name]}${NC}"
    echo "----------------------------------------"
    echo "MAX_CONCURRENT_DETECTIONS: ${current_config[MAX_CONCURRENT_DETECTIONS]}"
    echo "MAX_CONCURRENT_DOWNLOADS: ${current_config[MAX_CONCURRENT_DOWNLOADS]}"
    echo "MAX_BATCH_SIZE: ${current_config[MAX_BATCH_SIZE]}"
    echo "Memory: ${current_config[memory]} (reservation: ${current_config[memory_reservation]})"
    echo ""
    
    # Update configuration
    update_config current_config
    
    # Restart container
    restart_container current_config || continue
    
    # Start memory monitoring in background
    monitor_memory "${current_config[name]}" &
    MONITOR_PID=$!
    
    # Run tests with different batch sizes
    for batch_size in 10 20 30; do
        if [ $batch_size -le ${current_config[MAX_BATCH_SIZE]} ]; then
            run_test current_config $batch_size
        fi
    done
    
    # Stop memory monitoring
    kill $MONITOR_PID 2>/dev/null || true
    
    echo ""
    echo "Configuration test completed"
    echo ""
    
    # Cool down before next configuration
    sleep 20
done

# Generate summary report
echo ""
echo -e "${GREEN}=== Test Summary ===${NC}"
echo ""

# Create summary script
cat << 'SUMMARY_SCRIPT' > /tmp/analyze_config_results.sh
#!/bin/bash

echo "Configuration Performance Summary"
echo "================================="
echo ""

# Process each configuration
for config in Conservative Balanced Performance; do
    echo "Configuration: $config"
    echo "-------------------"
    
    # Find all result files for this config
    for file in /tmp/test_${config}_*.json; do
        if [ -f "$file" ]; then
            # Parse JSON using Python for reliability
            python3 -c "
import json
with open('$file', 'r') as f:
    data = json.load(f)
    batch_size = data.get('batch_size', 'N/A')
    duration = data.get('duration', 'N/A')
    http_code = data.get('http_code', 'N/A')
    if isinstance(duration, (int, float)) and duration > 0:
        urls_per_sec = round(batch_size / duration, 2)
    else:
        urls_per_sec = 'N/A'
    print(f'  Batch {batch_size}: {duration}s ({urls_per_sec} URLs/sec) - HTTP {http_code}')
"
        fi
    done
    
    # Memory stats
    if [ -f "/tmp/memory_${config}.log" ]; then
        avg_docker_mem=$(awk -F',' 'NR>1 && $3 ~ /^[0-9.]+$/ {sum+=$3; count++} END {if(count>0) printf "%.1f", sum/count; else print "N/A"}' "/tmp/memory_${config}.log")
        echo "  Avg Docker Memory: ${avg_docker_mem} MB"
    fi
    
    echo ""
done

# Show successful tests only
echo "=== Successful Tests Only ==="
echo ""
python3 -c "
import json
import glob

results = []
for file in glob.glob('/tmp/test_*.json'):
    with open(file, 'r') as f:
        data = json.load(f)
        if data.get('http_code') == 200:
            results.append({
                'config': data['config'],
                'batch': data['batch_size'],
                'duration': round(data['duration'], 2),
                'speed': round(data['batch_size'] / data['duration'], 2)
            })

# Sort by speed
results.sort(key=lambda x: x['speed'], reverse=True)

print('Config         | Batch | Duration | Speed (URLs/sec)')
print('---------------|-------|----------|----------------')
for r in results:
    print(f\"{r['config']:14} | {r['batch']:5} | {r['duration']:8.2f}s | {r['speed']:15.2f}\")
"
SUMMARY_SCRIPT

chmod +x /tmp/analyze_config_results.sh

echo "Run the following to see results summary:"
echo "  /tmp/analyze_config_results.sh"
echo ""
echo "Individual test results are in: /tmp/test_*.json"
echo "Memory logs are in: /tmp/memory_*.log"