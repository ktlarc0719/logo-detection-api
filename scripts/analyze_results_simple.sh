#!/bin/bash
# Simple results analyzer without jq

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
            # Extract values using grep and sed
            batch_size=$(grep -o '"batch_size": [0-9]*' "$file" 2>/dev/null | sed 's/.*: //')
            duration=$(grep -o '"duration": [0-9.]*' "$file" 2>/dev/null | sed 's/.*: //')
            
            if [ -n "$batch_size" ] && [ -n "$duration" ]; then
                # Calculate URLs per second
                if command -v bc >/dev/null 2>&1; then
                    urls_per_sec=$(echo "scale=2; $batch_size / $duration" | bc 2>/dev/null || echo "N/A")
                else
                    # Fallback if bc is not available
                    urls_per_sec="N/A"
                fi
                
                echo "  Batch $batch_size: ${duration}s (${urls_per_sec} URLs/sec)"
            else
                echo "  Error reading file: $file"
                echo "  Content preview:"
                head -5 "$file" 2>/dev/null | sed 's/^/    /'
            fi
        fi
    done
    
    # Memory stats
    mem_file="/tmp/memory_${config}.log"
    if [ -f "$mem_file" ]; then
        # Calculate average Docker memory
        avg_mem=$(awk -F',' 'NR>1 && $3 ~ /^[0-9.]+$/ {sum+=$3; count++} END {if(count>0) printf "%.1f", sum/count; else print "N/A"}' "$mem_file" 2>/dev/null)
        echo "  Avg Docker Memory: ${avg_mem} MB"
    else
        echo "  No memory log found"
    fi
    
    echo ""
done

# Also show raw test results
echo ""
echo "=== Raw Test Results ==="
echo ""
for file in /tmp/test_*.json; do
    if [ -f "$file" ]; then
        echo "File: $file"
        echo "Size: $(ls -lh "$file" | awk '{print $5}')"
        echo "First few lines:"
        head -3 "$file" | sed 's/^/  /'
        echo ""
    fi
done