#!/bin/bash
# Fix and analyze existing results

echo "=== Checking API Status ==="
echo ""

# Check API endpoints
echo "Health check:"
curl -s http://localhost:8000/api/v1/health || curl -s http://localhost:8000/health || echo "Health endpoint not found"
echo ""

echo "Available endpoints:"
curl -s http://localhost:8000/docs | grep -oP '"/[^"]*"' | sort -u | head -20
echo ""

echo "=== Existing Test Results ==="
echo ""

# Parse existing results despite errors
for file in /tmp/test_*.json; do
    if [ -f "$file" ]; then
        config=$(basename "$file" | sed 's/test_\(.*\)_[0-9]*.json/\1/')
        batch=$(basename "$file" | sed 's/.*_\([0-9]*\).json/\1/')
        
        echo "Config: $config, Batch: $batch"
        
        # Check if it was a 404 error
        if grep -q '"error":"Not Found"' "$file"; then
            echo "  Status: 404 Not Found (wrong endpoint)"
        else
            # Try to extract timing
            duration=$(grep -o '"duration": [0-9.]*' "$file" | awk '{print $2}')
            if [ -n "$duration" ]; then
                echo "  Duration: ${duration}s"
            fi
        fi
        echo ""
    fi
done

echo "=== Memory Usage Summary ==="
echo ""

for log in /tmp/memory_*.log; do
    if [ -f "$log" ]; then
        config=$(basename "$log" | sed 's/memory_\(.*\).log/\1/')
        echo "Config: $config"
        
        # Get memory statistics
        if [ -s "$log" ]; then
            awk -F',' 'NR>1 && $3 ~ /^[0-9.]+$/ {
                if (NR==2) {min=$3; max=$3}
                if ($3 < min) min=$3
                if ($3 > max) max=$3
                sum+=$3; count++
            }
            END {
                if (count > 0) {
                    printf "  Min: %.1f MB, Max: %.1f MB, Avg: %.1f MB\n", min, max, sum/count
                } else {
                    print "  No valid data"
                }
            }' "$log"
        else
            echo "  Empty log file"
        fi
        echo ""
    fi
done