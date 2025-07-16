#!/bin/bash
# PostgreSQL Management Script for ML Applications

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PGDATA="/mnt/data-ssd/postgresql/15/main"
BACKUP_DIR="/mnt/data-ssd/postgresql/backups"

# Helper functions
print_menu() {
    echo -e "${BLUE}=================================================="
    echo "PostgreSQL Management Tool"
    echo "==================================================${NC}"
    echo "1. Database Management"
    echo "2. User Management"
    echo "3. Backup & Restore"
    echo "4. Performance Monitoring"
    echo "5. Configuration Management"
    echo "6. Service Control"
    echo "7. Quick Health Check"
    echo "0. Exit"
    echo ""
}

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Database Management Functions
manage_databases() {
    while true; do
        echo -e "\n${BLUE}Database Management${NC}"
        echo "1. List databases"
        echo "2. Create database"
        echo "3. Drop database"
        echo "4. Database size report"
        echo "5. Clone database"
        echo "0. Back to main menu"
        
        read -p "Select option: " db_choice
        
        case $db_choice in
            1)
                print_status "Listing databases..."
                sudo -u postgres psql -c "\l+"
                ;;
            2)
                read -p "Enter database name: " db_name
                read -p "Enter owner (default: postgres): " db_owner
                db_owner=${db_owner:-postgres}
                
                print_status "Creating database $db_name..."
                sudo -u postgres createdb -O "$db_owner" "$db_name"
                print_status "Database $db_name created successfully"
                ;;
            3)
                read -p "Enter database name to drop: " db_name
                read -p "Are you sure? This cannot be undone! (yes/no): " confirm
                
                if [ "$confirm" = "yes" ]; then
                    print_status "Dropping database $db_name..."
                    sudo -u postgres dropdb "$db_name"
                    print_status "Database $db_name dropped"
                else
                    print_warning "Operation cancelled"
                fi
                ;;
            4)
                print_status "Database size report:"
                sudo -u postgres psql -c "
                    SELECT 
                        datname as database,
                        pg_size_pretty(pg_database_size(datname)) as size,
                        pg_database_size(datname) as size_bytes
                    FROM pg_database
                    WHERE datistemplate = false
                    ORDER BY pg_database_size(datname) DESC;"
                ;;
            5)
                read -p "Source database: " source_db
                read -p "New database name: " new_db
                
                print_status "Cloning database $source_db to $new_db..."
                sudo -u postgres psql -c "CREATE DATABASE \"$new_db\" WITH TEMPLATE \"$source_db\";"
                print_status "Database cloned successfully"
                ;;
            0)
                break
                ;;
            *)
                print_error "Invalid option"
                ;;
        esac
    done
}

# User Management Functions
manage_users() {
    while true; do
        echo -e "\n${BLUE}User Management${NC}"
        echo "1. List users"
        echo "2. Create user"
        echo "3. Drop user"
        echo "4. Change password"
        echo "5. Grant privileges"
        echo "6. List user privileges"
        echo "0. Back to main menu"
        
        read -p "Select option: " user_choice
        
        case $user_choice in
            1)
                print_status "Listing users..."
                sudo -u postgres psql -c "\du+"
                ;;
            2)
                read -p "Enter username: " username
                read -s -p "Enter password: " password
                echo
                read -p "Create as superuser? (y/n): " is_super
                
                if [ "$is_super" = "y" ]; then
                    sudo -u postgres psql -c "CREATE USER $username WITH PASSWORD '$password' SUPERUSER;"
                else
                    sudo -u postgres psql -c "CREATE USER $username WITH PASSWORD '$password';"
                fi
                print_status "User $username created"
                ;;
            3)
                read -p "Enter username to drop: " username
                read -p "Are you sure? (yes/no): " confirm
                
                if [ "$confirm" = "yes" ]; then
                    sudo -u postgres psql -c "DROP USER $username;"
                    print_status "User $username dropped"
                else
                    print_warning "Operation cancelled"
                fi
                ;;
            4)
                read -p "Enter username: " username
                read -s -p "Enter new password: " password
                echo
                
                sudo -u postgres psql -c "ALTER USER $username WITH PASSWORD '$password';"
                print_status "Password changed for user $username"
                ;;
            5)
                read -p "Enter username: " username
                read -p "Enter database: " database
                echo "Select privilege level:"
                echo "1. Read only"
                echo "2. Read/Write"
                echo "3. All privileges"
                read -p "Choice: " priv_choice
                
                case $priv_choice in
                    1)
                        sudo -u postgres psql -d "$database" -c "GRANT SELECT ON ALL TABLES IN SCHEMA public TO $username;"
                        ;;
                    2)
                        sudo -u postgres psql -d "$database" -c "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO $username;"
                        ;;
                    3)
                        sudo -u postgres psql -d "$database" -c "GRANT ALL PRIVILEGES ON DATABASE $database TO $username;"
                        ;;
                esac
                print_status "Privileges granted"
                ;;
            6)
                read -p "Enter username: " username
                sudo -u postgres psql -c "
                    SELECT 
                        n.nspname as schema,
                        c.relname as table,
                        c.relkind as type,
                        array_to_string(c.relacl, E'\n') as privileges
                    FROM pg_class c
                    LEFT JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE c.relacl::text LIKE '%$username%'
                    ORDER BY n.nspname, c.relname;"
                ;;
            0)
                break
                ;;
            *)
                print_error "Invalid option"
                ;;
        esac
    done
}

# Backup & Restore Functions
backup_restore() {
    while true; do
        echo -e "\n${BLUE}Backup & Restore${NC}"
        echo "1. Backup database"
        echo "2. Backup all databases"
        echo "3. Restore database"
        echo "4. List backups"
        echo "5. Schedule automatic backup"
        echo "6. Clean old backups"
        echo "0. Back to main menu"
        
        read -p "Select option: " backup_choice
        
        case $backup_choice in
            1)
                read -p "Enter database name: " db_name
                timestamp=$(date +%Y%m%d_%H%M%S)
                backup_file="$BACKUP_DIR/${db_name}_${timestamp}.sql.gz"
                
                mkdir -p "$BACKUP_DIR"
                print_status "Backing up $db_name to $backup_file..."
                sudo -u postgres pg_dump "$db_name" | gzip > "$backup_file"
                print_status "Backup completed: $backup_file"
                ;;
            2)
                timestamp=$(date +%Y%m%d_%H%M%S)
                backup_file="$BACKUP_DIR/all_databases_${timestamp}.sql.gz"
                
                mkdir -p "$BACKUP_DIR"
                print_status "Backing up all databases..."
                sudo -u postgres pg_dumpall | gzip > "$backup_file"
                print_status "Backup completed: $backup_file"
                ;;
            3)
                echo "Available backups:"
                ls -lh "$BACKUP_DIR"/*.sql.gz 2>/dev/null || print_warning "No backups found"
                
                read -p "Enter backup file path: " backup_file
                read -p "Enter target database name: " db_name
                
                if [ -f "$backup_file" ]; then
                    print_status "Restoring $backup_file to $db_name..."
                    gunzip -c "$backup_file" | sudo -u postgres psql -d "$db_name"
                    print_status "Restore completed"
                else
                    print_error "Backup file not found"
                fi
                ;;
            4)
                print_status "Available backups:"
                ls -lh "$BACKUP_DIR"/*.sql.gz 2>/dev/null || print_warning "No backups found"
                ;;
            5)
                print_status "Creating backup cron job..."
                cron_cmd="0 2 * * * /usr/bin/sudo -u postgres pg_dumpall | gzip > $BACKUP_DIR/all_databases_\$(date +\%Y\%m\%d_\%H\%M\%S).sql.gz"
                (crontab -l 2>/dev/null; echo "$cron_cmd") | crontab -
                print_status "Daily backup scheduled at 2 AM"
                ;;
            6)
                read -p "Delete backups older than (days): " days
                print_status "Deleting backups older than $days days..."
                find "$BACKUP_DIR" -name "*.sql.gz" -mtime +$days -delete
                print_status "Old backups cleaned"
                ;;
            0)
                break
                ;;
            *)
                print_error "Invalid option"
                ;;
        esac
    done
}

# Performance Monitoring
monitor_performance() {
    while true; do
        echo -e "\n${BLUE}Performance Monitoring${NC}"
        echo "1. Current connections"
        echo "2. Running queries"
        echo "3. Database statistics"
        echo "4. Table sizes"
        echo "5. Index usage"
        echo "6. Cache hit ratio"
        echo "7. Slow queries"
        echo "8. Lock monitoring"
        echo "0. Back to main menu"
        
        read -p "Select option: " perf_choice
        
        case $perf_choice in
            1)
                print_status "Current connections:"
                sudo -u postgres psql -c "
                    SELECT 
                        pid,
                        usename,
                        application_name,
                        client_addr,
                        state,
                        state_change
                    FROM pg_stat_activity
                    WHERE state IS NOT NULL
                    ORDER BY state_change DESC;"
                ;;
            2)
                print_status "Running queries:"
                sudo -u postgres psql -c "
                    SELECT 
                        pid,
                        now() - pg_stat_activity.query_start AS duration,
                        usename,
                        state,
                        substring(query, 1, 100) AS query
                    FROM pg_stat_activity
                    WHERE state != 'idle'
                    ORDER BY duration DESC;"
                ;;
            3)
                print_status "Database statistics:"
                sudo -u postgres psql -c "
                    SELECT 
                        datname,
                        numbackends as connections,
                        xact_commit as commits,
                        xact_rollback as rollbacks,
                        blks_read as disk_reads,
                        blks_hit as cache_hits,
                        round((blks_hit::float / (blks_hit + blks_read + 0.00001)) * 100, 2) as cache_hit_ratio
                    FROM pg_stat_database
                    WHERE datname NOT IN ('template0', 'template1')
                    ORDER BY connections DESC;"
                ;;
            4)
                read -p "Enter database name: " db_name
                print_status "Table sizes in $db_name:"
                sudo -u postgres psql -d "$db_name" -c "
                    SELECT 
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
                        pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) AS indexes_size
                    FROM pg_tables
                    WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                    LIMIT 20;"
                ;;
            5)
                read -p "Enter database name: " db_name
                print_status "Index usage in $db_name:"
                sudo -u postgres psql -d "$db_name" -c "
                    SELECT 
                        schemaname,
                        tablename,
                        indexname,
                        idx_scan as index_scans,
                        pg_size_pretty(pg_relation_size(indexrelid)) as index_size
                    FROM pg_stat_user_indexes
                    ORDER BY idx_scan DESC
                    LIMIT 20;"
                ;;
            6)
                print_status "Cache hit ratios:"
                sudo -u postgres psql -c "
                    SELECT 
                        'Database Cache' as cache_type,
                        sum(blks_hit) as hits,
                        sum(blks_read) as reads,
                        round(sum(blks_hit)::float / (sum(blks_hit) + sum(blks_read) + 0.00001) * 100, 2) as hit_ratio
                    FROM pg_stat_database
                    UNION ALL
                    SELECT 
                        'Table Cache' as cache_type,
                        sum(heap_blks_hit) as hits,
                        sum(heap_blks_read) as reads,
                        round(sum(heap_blks_hit)::float / (sum(heap_blks_hit) + sum(heap_blks_read) + 0.00001) * 100, 2) as hit_ratio
                    FROM pg_statio_user_tables
                    UNION ALL
                    SELECT 
                        'Index Cache' as cache_type,
                        sum(idx_blks_hit) as hits,
                        sum(idx_blks_read) as reads,
                        round(sum(idx_blks_hit)::float / (sum(idx_blks_hit) + sum(idx_blks_read) + 0.00001) * 100, 2) as hit_ratio
                    FROM pg_statio_user_indexes;"
                ;;
            7)
                print_status "Slow queries (requires pg_stat_statements):"
                sudo -u postgres psql -c "
                    SELECT 
                        substring(query, 1, 80) as query,
                        calls,
                        round(total_exec_time::numeric, 2) as total_time_ms,
                        round(mean_exec_time::numeric, 2) as mean_time_ms,
                        round(stddev_exec_time::numeric, 2) as stddev_time_ms
                    FROM pg_stat_statements
                    WHERE query NOT LIKE '%pg_stat_statements%'
                    ORDER BY mean_exec_time DESC
                    LIMIT 10;"
                ;;
            8)
                print_status "Current locks:"
                sudo -u postgres psql -c "
                    SELECT 
                        pg_stat_activity.pid,
                        pg_stat_activity.usename,
                        pg_stat_activity.query,
                        pg_locks.mode,
                        pg_locks.granted
                    FROM pg_stat_activity
                    JOIN pg_locks ON pg_stat_activity.pid = pg_locks.pid
                    WHERE pg_stat_activity.pid != pg_backend_pid()
                    ORDER BY pg_stat_activity.pid;"
                ;;
            0)
                break
                ;;
            *)
                print_error "Invalid option"
                ;;
        esac
        
        read -p "Press Enter to continue..."
    done
}

# Configuration Management
manage_configuration() {
    while true; do
        echo -e "\n${BLUE}Configuration Management${NC}"
        echo "1. Show current settings"
        echo "2. Apply Phase 2 settings (16GB shared_buffers)"
        echo "3. Apply Phase 3 settings (32GB shared_buffers)"
        echo "4. Edit postgresql.conf"
        echo "5. Reload configuration"
        echo "6. Show configuration changes"
        echo "0. Back to main menu"
        
        read -p "Select option: " config_choice
        
        case $config_choice in
            1)
                print_status "Current memory settings:"
                sudo -u postgres psql -c "
                    SELECT name, setting, unit, short_desc
                    FROM pg_settings
                    WHERE name IN ('shared_buffers', 'effective_cache_size', 'work_mem', 
                                   'maintenance_work_mem', 'max_connections', 
                                   'max_parallel_workers', 'max_parallel_workers_per_gather')
                    ORDER BY name;"
                ;;
            2)
                print_status "Applying Phase 2 configuration..."
                cat << EOF | sudo tee /etc/postgresql/15/main/conf.d/02-phase2.conf
# Phase 2 Configuration - Moderate Performance
# For established workloads

# Memory Settings (Moderate)
shared_buffers = 16GB                   # Increased from 8GB
effective_cache_size = 64GB             # 50% of total RAM
maintenance_work_mem = 4GB              # Increased for faster maintenance
work_mem = 512MB                        # Increased per-operation memory

# Parallel Processing (Increased)
max_parallel_workers_per_gather = 6     # Increased from 4
max_parallel_workers = 12               # Match CPU cores
max_parallel_maintenance_workers = 6    # Increased

# WAL Settings (Performance)
wal_buffers = 64MB
wal_compression = on
wal_init_zero = off
wal_recycle = on
EOF
                sudo systemctl reload postgresql
                print_status "Phase 2 configuration applied"
                ;;
            3)
                print_status "Applying Phase 3 configuration..."
                cat << EOF | sudo tee /etc/postgresql/15/main/conf.d/03-phase3.conf
# Phase 3 Configuration - High Performance
# For production ML workloads

# Memory Settings (Aggressive)
shared_buffers = 32GB                   # 25% of total RAM
effective_cache_size = 96GB             # 75% of total RAM
maintenance_work_mem = 8GB              # Maximum for maintenance
work_mem = 1GB                          # High per-operation memory

# Parallel Processing (Maximum)
max_parallel_workers_per_gather = 8     # High parallelism
max_parallel_workers = 24               # Match thread count
max_parallel_maintenance_workers = 8    # Maximum maintenance parallelism

# Advanced Settings
huge_pages = try                        # Enable huge pages
temp_buffers = 512MB                    # Temporary table memory
max_prepared_transactions = 100         # For distributed transactions

# JIT Compilation
jit = on
jit_above_cost = 100000
jit_inline_above_cost = 500000
jit_optimize_above_cost = 500000
EOF
                sudo systemctl reload postgresql
                print_status "Phase 3 configuration applied"
                ;;
            4)
                sudo nano /etc/postgresql/15/main/postgresql.conf
                ;;
            5)
                print_status "Reloading configuration..."
                sudo systemctl reload postgresql
                print_status "Configuration reloaded"
                ;;
            6)
                print_status "Configuration files that would be loaded:"
                sudo -u postgres psql -c "SELECT name, setting FROM pg_settings WHERE source = 'configuration file';"
                ;;
            0)
                break
                ;;
            *)
                print_error "Invalid option"
                ;;
        esac
    done
}

# Service Control
service_control() {
    while true; do
        echo -e "\n${BLUE}Service Control${NC}"
        echo "1. Start PostgreSQL"
        echo "2. Stop PostgreSQL"
        echo "3. Restart PostgreSQL"
        echo "4. Service status"
        echo "5. Enable auto-start"
        echo "6. Disable auto-start"
        echo "0. Back to main menu"
        
        read -p "Select option: " service_choice
        
        case $service_choice in
            1)
                print_status "Starting PostgreSQL..."
                sudo systemctl start postgresql
                print_status "PostgreSQL started"
                ;;
            2)
                print_status "Stopping PostgreSQL..."
                sudo systemctl stop postgresql
                print_status "PostgreSQL stopped"
                ;;
            3)
                print_status "Restarting PostgreSQL..."
                sudo systemctl restart postgresql
                print_status "PostgreSQL restarted"
                ;;
            4)
                sudo systemctl status postgresql
                ;;
            5)
                sudo systemctl enable postgresql
                print_status "Auto-start enabled"
                ;;
            6)
                sudo systemctl disable postgresql
                print_status "Auto-start disabled"
                ;;
            0)
                break
                ;;
            *)
                print_error "Invalid option"
                ;;
        esac
    done
}

# Quick Health Check
health_check() {
    echo -e "\n${BLUE}PostgreSQL Health Check${NC}"
    echo "=============================="
    
    # Service status
    if systemctl is-active --quiet postgresql; then
        echo -e "Service Status: ${GREEN}Running${NC}"
    else
        echo -e "Service Status: ${RED}Stopped${NC}"
    fi
    
    # Version
    version=$(sudo -u postgres psql -t -c "SELECT version();" | head -1)
    echo "Version: $version"
    
    # Uptime
    uptime=$(sudo -u postgres psql -t -c "SELECT now() - pg_postmaster_start_time() AS uptime;")
    echo "Uptime: $uptime"
    
    # Connection count
    connections=$(sudo -u postgres psql -t -c "SELECT count(*) FROM pg_stat_activity;")
    max_conn=$(sudo -u postgres psql -t -c "SHOW max_connections;")
    echo "Connections: $connections / $max_conn"
    
    # Database sizes
    echo -e "\n${YELLOW}Database Sizes:${NC}"
    sudo -u postgres psql -c "
        SELECT 
            datname as database,
            pg_size_pretty(pg_database_size(datname)) as size
        FROM pg_database
        WHERE datistemplate = false
        ORDER BY pg_database_size(datname) DESC
        LIMIT 5;"
    
    # Cache hit ratio
    echo -e "\n${YELLOW}Cache Performance:${NC}"
    sudo -u postgres psql -c "
        SELECT 
            round(sum(blks_hit)::float / (sum(blks_hit) + sum(blks_read) + 0.00001) * 100, 2) as cache_hit_ratio
        FROM pg_stat_database;"
    
    # Disk usage
    echo -e "\n${YELLOW}Data Directory Usage:${NC}"
    df -h "$PGDATA" | tail -1
    
    read -p "Press Enter to continue..."
}

# Main menu loop
while true; do
    clear
    print_menu
    read -p "Select option: " choice
    
    case $choice in
        1) manage_databases ;;
        2) manage_users ;;
        3) backup_restore ;;
        4) monitor_performance ;;
        5) manage_configuration ;;
        6) service_control ;;
        7) health_check ;;
        0) 
            print_status "Exiting..."
            exit 0
            ;;
        *)
            print_error "Invalid option"
            sleep 1
            ;;
    esac
done