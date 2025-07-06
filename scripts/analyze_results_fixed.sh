#!/bin/bash

echo "Configuration Performance Summary"
echo "================================="
echo ""

# Process each configuration
for config in Conservative Balanced Performance; do
    echo "Configuration: $config"
    echo "-------------------"
    
    # Find all result files for this config (with timestamp)
    for file in /tmp/test_${config}_*_*.json; do
        if [ -f "$file" ]; then
            # Check if file is valid JSON first
            if python3 -c "import json; json.load(open('$file'))" 2>/dev/null; then
                # Parse JSON using Python
                python3 -c "
import json
try:
    with open('$file', 'r') as f:
        data = json.load(f)
        batch_size = data.get('batch_size', 'N/A')
        duration = data.get('duration', 'N/A')
        http_code = data.get('http_code', 'N/A')
        if isinstance(duration, (int, float)) and duration > 0:
            urls_per_sec = round(batch_size / duration, 2)
        else:
            urls_per_sec = 'N/A'
        print(f'  Batch {batch_size}: {duration:.2f}s ({urls_per_sec} URLs/sec) - HTTP {http_code}')
except Exception as e:
    print(f'  Error reading {file}: {e}')
"
            else
                echo "  Invalid JSON in $file"
            fi
        fi
    done
    
    # Memory stats - look for files with timestamp
    mem_files=$(ls -t /tmp/memory_${config}_*.log 2>/dev/null | head -1)
    if [ -n "$mem_files" ]; then
        for mem_file in $mem_files; do
            if [ -f "$mem_file" ]; then
                # Check if file has header or not
                first_line=$(head -1 "$mem_file")
                if [[ "$first_line" =~ ^[0-9]+,[0-9]+,[0-9.]+ ]]; then
                    # No header, process all lines
                    avg_docker_mem=$(awk -F',' '$3 ~ /^[0-9.]+$/ {sum+=$3; count++} END {if(count>0) printf "%.1f", sum/count; else print "N/A"}' "$mem_file")
                else
                    # Has header, skip first line
                    avg_docker_mem=$(awk -F',' 'NR>1 && $3 ~ /^[0-9.]+$/ {sum+=$3; count++} END {if(count>0) printf "%.1f", sum/count; else print "N/A"}' "$mem_file")
                fi
                echo "  Avg Docker Memory: ${avg_docker_mem} MB (from $(basename $mem_file))"
                break
            fi
        done
    fi
    
    echo ""
done

# Show all test files for debugging
echo "=== Test Files Found ==="
ls -la /tmp/test_*.json 2>/dev/null | tail -10

echo ""
echo "=== Memory Files Found ==="
ls -la /tmp/memory_*.log 2>/dev/null | tail -10