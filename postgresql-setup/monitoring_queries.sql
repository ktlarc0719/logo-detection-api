-- PostgreSQL Monitoring Queries for ML Workloads
-- Optimized for high-performance hardware and large datasets

-- ============================================
-- 1. SYSTEM OVERVIEW
-- ============================================

-- System Information
SELECT 
    current_setting('server_version') as postgresql_version,
    pg_postmaster_start_time() as server_start_time,
    now() - pg_postmaster_start_time() as uptime,
    current_setting('data_directory') as data_directory,
    pg_size_pretty(sum(pg_database_size(datname))) as total_database_size
FROM pg_database
WHERE datistemplate = false;

-- Current Activity Summary
SELECT 
    count(*) FILTER (WHERE state = 'active') as active_connections,
    count(*) FILTER (WHERE state = 'idle') as idle_connections,
    count(*) FILTER (WHERE state = 'idle in transaction') as idle_in_transaction,
    count(*) FILTER (WHERE wait_event_type IS NOT NULL) as waiting_connections,
    count(*) as total_connections
FROM pg_stat_activity
WHERE pid != pg_backend_pid();

-- ============================================
-- 2. MEMORY USAGE
-- ============================================

-- Memory Settings Overview
SELECT 
    name,
    setting,
    unit,
    CASE 
        WHEN unit = '8kB' THEN pg_size_pretty(setting::bigint * 8192)
        WHEN unit = 'kB' THEN pg_size_pretty(setting::bigint * 1024)
        WHEN unit = 'MB' THEN pg_size_pretty(setting::bigint * 1024 * 1024)
        ELSE setting || ' ' || coalesce(unit, 'units')
    END as human_readable,
    short_desc
FROM pg_settings
WHERE name IN (
    'shared_buffers', 'effective_cache_size', 'work_mem', 
    'maintenance_work_mem', 'temp_buffers', 'wal_buffers',
    'max_connections', 'max_parallel_workers', 'max_parallel_workers_per_gather'
)
ORDER BY name;

-- Shared Buffer Usage
SELECT 
    c.relname,
    pg_size_pretty(count(*) * 8192) as buffered_size,
    round(100.0 * count(*) / (SELECT setting::int FROM pg_settings WHERE name = 'shared_buffers'), 2) as percent_of_shared_buffers
FROM pg_buffercache b
INNER JOIN pg_class c ON b.relfilenode = pg_relation_filenode(c.oid)
WHERE b.reldatabase = (SELECT oid FROM pg_database WHERE datname = current_database())
GROUP BY c.relname
ORDER BY count(*) DESC
LIMIT 20;

-- ============================================
-- 3. PERFORMANCE METRICS
-- ============================================

-- Cache Hit Ratios by Database
SELECT 
    datname,
    blks_hit,
    blks_read,
    CASE 
        WHEN blks_hit + blks_read = 0 THEN 0
        ELSE round(100.0 * blks_hit / (blks_hit + blks_read), 2)
    END as cache_hit_ratio,
    xact_commit,
    xact_rollback,
    deadlocks,
    conflicts
FROM pg_stat_database
WHERE datname NOT IN ('template0', 'template1')
ORDER BY blks_hit + blks_read DESC;

-- Table Cache Hit Ratios
SELECT 
    schemaname,
    tablename,
    heap_blks_read,
    heap_blks_hit,
    CASE 
        WHEN heap_blks_hit + heap_blks_read = 0 THEN 0
        ELSE round(100.0 * heap_blks_hit / (heap_blks_hit + heap_blks_read), 2)
    END as cache_hit_ratio,
    n_tup_ins,
    n_tup_upd,
    n_tup_del
FROM pg_statio_user_tables
WHERE heap_blks_hit + heap_blks_read > 1000  -- Only tables with significant activity
ORDER BY heap_blks_hit + heap_blks_read DESC
LIMIT 20;

-- ============================================
-- 4. QUERY PERFORMANCE
-- ============================================

-- Top 10 Slowest Queries (requires pg_stat_statements)
SELECT 
    substring(query, 1, 100) as query_preview,
    calls,
    round(total_exec_time::numeric, 2) as total_time_ms,
    round(mean_exec_time::numeric, 2) as avg_time_ms,
    round(stddev_exec_time::numeric, 2) as stddev_time_ms,
    round(min_exec_time::numeric, 2) as min_time_ms,
    round(max_exec_time::numeric, 2) as max_time_ms,
    rows
FROM pg_stat_statements
WHERE query NOT LIKE '%pg_stat_statements%'
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Currently Running Long Queries
SELECT 
    pid,
    now() - pg_stat_activity.query_start AS duration,
    usename,
    application_name,
    state,
    wait_event_type,
    wait_event,
    substring(query, 1, 100) as query_preview
FROM pg_stat_activity
WHERE (now() - pg_stat_activity.query_start) > interval '1 minute'
    AND state != 'idle'
ORDER BY duration DESC;

-- ============================================
-- 5. TABLE AND INDEX ANALYSIS
-- ============================================

-- Largest Tables with Toast and Index Sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) AS external_size,
    CASE 
        WHEN pg_relation_size(schemaname||'.'||tablename) = 0 THEN 0
        ELSE round(100.0 * (pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) / pg_relation_size(schemaname||'.'||tablename), 2)
    END as external_pct
FROM pg_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
LIMIT 20;

-- Index Usage Statistics
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan as index_scans,
    idx_tup_read as tuples_read,
    idx_tup_fetch as tuples_fetched,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC
LIMIT 20;

-- Unused Indexes (candidates for removal)
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
FROM pg_stat_user_indexes
WHERE idx_scan = 0
    AND indexrelname NOT LIKE '%_pkey'  -- Keep primary keys
    AND pg_relation_size(indexrelid) > 1024 * 1024  -- Only indexes > 1MB
ORDER BY pg_relation_size(indexrelid) DESC;

-- ============================================
-- 6. VACUUM AND MAINTENANCE
-- ============================================

-- Tables Needing Vacuum
SELECT 
    schemaname,
    tablename,
    n_dead_tup,
    n_live_tup,
    round(100.0 * n_dead_tup / NULLIF(n_live_tup + n_dead_tup, 0), 2) as dead_tuple_pct,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze
FROM pg_stat_user_tables
WHERE n_dead_tup > 1000
ORDER BY n_dead_tup DESC
LIMIT 20;

-- Autovacuum Activity
SELECT 
    schemaname,
    tablename,
    last_autovacuum,
    last_autoanalyze,
    autovacuum_count,
    autoanalyze_count,
    n_tup_ins + n_tup_upd + n_tup_del as total_writes
FROM pg_stat_user_tables
WHERE last_autovacuum IS NOT NULL
ORDER BY last_autovacuum DESC
LIMIT 20;

-- ============================================
-- 7. LOCK MONITORING
-- ============================================

-- Current Lock Waits
SELECT 
    blocked_locks.pid AS blocked_pid,
    blocked_activity.usename AS blocked_user,
    blocking_locks.pid AS blocking_pid,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_statement,
    blocking_activity.query AS current_statement_in_blocking_process,
    blocked_activity.application_name AS blocked_application,
    blocking_activity.application_name AS blocking_application,
    now() - blocked_activity.query_start AS blocked_duration,
    now() - blocking_activity.query_start AS blocking_duration
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks 
    ON blocking_locks.locktype = blocked_locks.locktype
    AND blocking_locks.database IS NOT DISTINCT FROM blocked_locks.database
    AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
    AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
    AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
    AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
    AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
    AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
    AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
    AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
    AND blocking_locks.pid != blocked_locks.pid
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;

-- ============================================
-- 8. REPLICATION AND WAL
-- ============================================

-- WAL Generation Rate
WITH wal_stats AS (
    SELECT 
        pg_current_wal_lsn() as current_lsn,
        pg_stat_get_wal_senders() as wal_senders,
        now() as current_time
)
SELECT 
    pg_size_pretty(pg_wal_lsn_diff(current_lsn, '0/0')) as total_wal_generated,
    current_setting('max_wal_size') as max_wal_size,
    current_setting('checkpoint_timeout') as checkpoint_timeout
FROM wal_stats;

-- Checkpoint Statistics
SELECT 
    checkpoints_timed,
    checkpoints_req,
    checkpoint_write_time,
    checkpoint_sync_time,
    buffers_checkpoint,
    buffers_clean,
    buffers_backend,
    buffers_backend_fsync,
    buffers_alloc
FROM pg_stat_bgwriter;

-- ============================================
-- 9. CONNECTION POOLING ANALYSIS
-- ============================================

-- Connection Distribution by State and Application
SELECT 
    application_name,
    state,
    count(*) as connection_count,
    round(avg(extract(epoch from (now() - state_change))), 2) as avg_state_duration_seconds
FROM pg_stat_activity
WHERE pid != pg_backend_pid()
GROUP BY application_name, state
ORDER BY connection_count DESC;

-- Connection Age Distribution
SELECT 
    CASE 
        WHEN age < interval '1 minute' THEN '< 1 minute'
        WHEN age < interval '5 minutes' THEN '1-5 minutes'
        WHEN age < interval '1 hour' THEN '5-60 minutes'
        WHEN age < interval '1 day' THEN '1-24 hours'
        ELSE '> 1 day'
    END as connection_age,
    count(*) as connection_count
FROM (
    SELECT now() - backend_start as age
    FROM pg_stat_activity
    WHERE pid != pg_backend_pid()
) t
GROUP BY connection_age
ORDER BY connection_age;

-- ============================================
-- 10. ML-SPECIFIC MONITORING
-- ============================================

-- Large Result Set Queries (common in ML workloads)
SELECT 
    substring(query, 1, 100) as query_preview,
    calls,
    rows,
    round(rows::numeric / calls, 2) as avg_rows_per_call,
    round(total_exec_time::numeric, 2) as total_time_ms,
    round(mean_exec_time::numeric, 2) as avg_time_ms
FROM pg_stat_statements
WHERE rows > 10000  -- Queries returning many rows
    AND calls > 10  -- Frequently executed
ORDER BY rows DESC
LIMIT 20;

-- Temporary File Usage (common in large sorts/joins)
SELECT 
    datname,
    temp_files,
    pg_size_pretty(temp_bytes) as temp_size,
    blk_read_time,
    blk_write_time
FROM pg_stat_database
WHERE temp_files > 0
ORDER BY temp_bytes DESC;

-- Parallel Query Usage
SELECT 
    substring(query, 1, 100) as query_preview,
    calls,
    total_exec_time,
    mean_exec_time,
    rows,
    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements
WHERE query LIKE '%Parallel%' 
   OR query LIKE '%Worker%'
ORDER BY calls DESC
LIMIT 20;