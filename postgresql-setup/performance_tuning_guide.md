# PostgreSQL Performance Tuning Guide for ML Workloads

## System Overview
- **CPU**: AMD Ryzen 9 9900X (12 cores / 24 threads)
- **Memory**: 128GB DDR5
- **Storage**: SSD via WSL2 mount
- **Use Case**: Machine Learning with up to 50M records

## Phase-Based Tuning Approach

### Phase 1: Conservative Start (Weeks 1-2)
Start with conservative settings to establish baseline performance.

#### WSL2 Configuration
```bash
# .wslconfig
memory=64GB
processors=24
swap=16GB
```

#### PostgreSQL Configuration
```sql
# Memory
shared_buffers = 8GB              # 12.5% of WSL memory
effective_cache_size = 32GB       # 50% of WSL memory
work_mem = 256MB
maintenance_work_mem = 2GB

# Parallelism
max_parallel_workers = 8
max_parallel_workers_per_gather = 4
```

#### Monitoring Commands
```bash
# Check memory usage
free -h
sudo -u postgres psql -c "SELECT name, setting FROM pg_settings WHERE name LIKE '%buffer%';"

# Monitor cache hit ratio
sudo -u postgres psql -c "SELECT sum(blks_hit)*100.0/sum(blks_hit+blks_read) AS cache_hit_ratio FROM pg_stat_database;"
```

### Phase 2: Moderate Performance (Weeks 3-4)
After validating stability, increase resource allocation.

#### WSL2 Configuration
```bash
# .wslconfig
memory=96GB
processors=24
swap=16GB
```

#### PostgreSQL Configuration
```bash
# Apply Phase 2 settings
sudo ./manage_postgresql.sh
# Select option 5 -> 2 (Apply Phase 2 settings)
```

```sql
# Memory
shared_buffers = 16GB             # 16.7% of WSL memory
effective_cache_size = 64GB       # 66.7% of WSL memory
work_mem = 512MB
maintenance_work_mem = 4GB

# Parallelism
max_parallel_workers = 12
max_parallel_workers_per_gather = 6
```

### Phase 3: Maximum Performance (Week 5+)
For production workloads with proven stability.

#### WSL2 Configuration
```bash
# .wslconfig
memory=112GB
processors=24
swap=16GB
```

#### PostgreSQL Configuration
```bash
# Apply Phase 3 settings
sudo ./manage_postgresql.sh
# Select option 5 -> 3 (Apply Phase 3 settings)
```

```sql
# Memory
shared_buffers = 32GB             # 28.6% of WSL memory
effective_cache_size = 96GB       # 85.7% of WSL memory
work_mem = 1GB
maintenance_work_mem = 8GB

# Parallelism
max_parallel_workers = 24
max_parallel_workers_per_gather = 8
```

## ML-Specific Optimizations

### 1. Bulk Data Loading
```sql
-- Before bulk insert
SET maintenance_work_mem = '8GB';
SET max_parallel_maintenance_workers = 8;
ALTER TABLE your_table SET (autovacuum_enabled = false);

-- After bulk insert
ALTER TABLE your_table SET (autovacuum_enabled = true);
ANALYZE your_table;
```

### 2. Large Table Partitioning
```sql
-- Example: Partition by date for time-series ML data
CREATE TABLE measurements (
    id BIGSERIAL,
    sensor_id INT,
    timestamp TIMESTAMPTZ,
    value FLOAT,
    features JSONB
) PARTITION BY RANGE (timestamp);

-- Create monthly partitions
CREATE TABLE measurements_2024_01 PARTITION OF measurements
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Create index on partitions
CREATE INDEX idx_measurements_2024_01_sensor_timestamp 
    ON measurements_2024_01 (sensor_id, timestamp);
```

### 3. Vector Operations (pgvector)
```sql
-- Enable pgvector for ML embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table with vector column
CREATE TABLE items (
    id BIGSERIAL PRIMARY KEY,
    embedding vector(1536),  -- For OpenAI embeddings
    metadata JSONB
);

-- Create index for similarity search
CREATE INDEX ON items USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);  -- Adjust based on data size
```

### 4. Parallel Aggregations
```sql
-- Enable parallel aggregation for large datasets
SET max_parallel_workers_per_gather = 8;
SET parallel_setup_cost = 0;
SET parallel_tuple_cost = 0;

-- Example: Parallel statistics computation
EXPLAIN (ANALYZE, BUFFERS)
SELECT 
    feature_name,
    AVG(value) as mean,
    STDDEV(value) as std,
    MIN(value) as min,
    MAX(value) as max
FROM feature_data
GROUP BY feature_name;
```

## Monitoring and Troubleshooting

### Key Metrics to Monitor

1. **Cache Hit Ratio** (Target: >99%)
```sql
SELECT 
    datname,
    blks_hit,
    blks_read,
    round(100.0 * blks_hit / (blks_hit + blks_read), 2) as cache_hit_ratio
FROM pg_stat_database
WHERE datname = current_database();
```

2. **Connection Pooling**
```sql
-- Check connection usage
SELECT count(*), state 
FROM pg_stat_activity 
GROUP BY state;

-- Find long-running queries
SELECT pid, now() - query_start as duration, query 
FROM pg_stat_activity 
WHERE state = 'active' 
ORDER BY duration DESC;
```

3. **Table Bloat**
```sql
-- Check table bloat
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
    n_dead_tup,
    n_live_tup,
    round(100.0 * n_dead_tup / nullif(n_live_tup, 0), 2) as dead_pct
FROM pg_stat_user_tables
WHERE n_dead_tup > 1000
ORDER BY n_dead_tup DESC;
```

### Common Issues and Solutions

#### 1. High Memory Usage
```bash
# Check shared memory usage
ipcs -m

# PostgreSQL memory breakdown
sudo -u postgres psql -c "
SELECT 
    name, 
    setting, 
    unit,
    pg_size_pretty(setting::bigint * 
        CASE unit 
            WHEN '8kB' THEN 8192 
            WHEN 'kB' THEN 1024 
            WHEN 'MB' THEN 1024*1024 
            ELSE 1 
        END) as size
FROM pg_settings 
WHERE name IN ('shared_buffers', 'work_mem', 'maintenance_work_mem');"
```

#### 2. Slow Queries
```sql
-- Enable query logging
ALTER SYSTEM SET log_min_duration_statement = 1000;  -- Log queries > 1s
SELECT pg_reload_conf();

-- Check slow query log
sudo tail -f /mnt/data-ssd/postgresql/15/main/log/*.log

-- Use EXPLAIN ANALYZE
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) 
SELECT * FROM your_table WHERE conditions;
```

#### 3. Lock Contention
```sql
-- Find blocking queries
SELECT 
    blocked.pid as blocked_pid,
    blocked.query as blocked_query,
    blocking.pid as blocking_pid,
    blocking.query as blocking_query
FROM pg_stat_activity AS blocked
JOIN pg_stat_activity AS blocking 
    ON blocking.pid = ANY(pg_blocking_pids(blocked.pid))
WHERE blocked.wait_event_type = 'Lock';
```

## Backup and Recovery Strategy

### Automated Backup Script
```bash
#!/bin/bash
# /home/username/backup_postgresql.sh

BACKUP_DIR="/mnt/data-ssd/postgresql/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DB_NAME="ml_development"

# Create backup
pg_dump -U postgres -d $DB_NAME | gzip > "$BACKUP_DIR/${DB_NAME}_${TIMESTAMP}.sql.gz"

# Keep only last 7 days of backups
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete

# Backup configuration files
tar -czf "$BACKUP_DIR/config_${TIMESTAMP}.tar.gz" /etc/postgresql/15/main/
```

### Recovery Procedure
```bash
# Restore from backup
gunzip -c /path/to/backup.sql.gz | psql -U postgres -d target_database

# Point-in-time recovery setup
# Edit postgresql.conf
archive_mode = on
archive_command = 'test ! -f /mnt/data-ssd/postgresql/archive/%f && cp %p /mnt/data-ssd/postgresql/archive/%f'
```

## Advanced Optimizations

### 1. Huge Pages
```bash
# Calculate huge pages needed
sudo -u postgres psql -c "SHOW shared_buffers;"
# If shared_buffers = 32GB, need 16384 huge pages (2MB each)

# Configure huge pages
echo "vm.nr_hugepages = 16384" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# Enable in PostgreSQL
echo "huge_pages = on" | sudo tee -a /etc/postgresql/15/main/conf.d/04-hugepages.conf
```

### 2. NUMA Optimization
```bash
# Check NUMA nodes (on WSL2, usually single node)
numactl --hardware

# PostgreSQL NUMA binding (if multiple nodes)
# Edit systemd service
sudo systemctl edit postgresql@15-main.service
# Add: ExecStart=/usr/bin/numactl --interleave=all /usr/lib/postgresql/15/bin/postgres ...
```

### 3. JIT Compilation
```sql
-- Enable JIT for complex queries
SET jit = on;
SET jit_above_cost = 100000;
SET jit_inline_above_cost = 500000;
SET jit_optimize_above_cost = 500000;

-- Make permanent
ALTER SYSTEM SET jit = on;
SELECT pg_reload_conf();
```

## Performance Testing

### Synthetic Benchmark
```bash
# Install pgbench
sudo apt-get install postgresql-contrib-15

# Initialize pgbench
pgbench -i -s 1000 -U postgres ml_development

# Run benchmark
pgbench -c 10 -j 4 -T 300 -U postgres ml_development
```

### Real-World ML Query Testing
```sql
-- Create test dataset
CREATE TABLE ml_features AS
SELECT 
    generate_series(1, 10000000) as id,
    random() * 1000 as feature1,
    random() * 1000 as feature2,
    array_fill(random()::float, ARRAY[100]) as feature_vector,
    now() - interval '1 year' * random() as created_at;

-- Test parallel scan
EXPLAIN (ANALYZE, BUFFERS)
SELECT 
    date_trunc('day', created_at) as day,
    avg(feature1) as avg_f1,
    stddev(feature2) as std_f2,
    count(*) as count
FROM ml_features
GROUP BY date_trunc('day', created_at)
ORDER BY day;
```

## Maintenance Schedule

### Daily Tasks
- Monitor slow query log
- Check connection count
- Verify cache hit ratio > 99%

### Weekly Tasks
- Run VACUUM ANALYZE on large tables
- Review pgbadger reports
- Check for unused indexes

### Monthly Tasks
- Full database backup
- Review and adjust work_mem based on query patterns
- Update table statistics target for frequently queried columns

### Quarterly Tasks
- Consider moving to next performance phase
- Review partitioning strategy
- Analyze query patterns for optimization opportunities