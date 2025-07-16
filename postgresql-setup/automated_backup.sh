#!/bin/bash
# PostgreSQL Automated Backup Script for ML Workloads
# Supports full and incremental backups with compression

set -e

# Configuration
BACKUP_BASE_DIR="/mnt/data-ssd/postgresql/backups"
ARCHIVE_DIR="/mnt/data-ssd/postgresql/archive"
LOG_DIR="/mnt/data-ssd/postgresql/backup_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATE=$(date +%Y%m%d)
RETENTION_DAYS=7
RETENTION_WEEKS=4
RETENTION_MONTHS=3

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Create directories
mkdir -p "$BACKUP_BASE_DIR"/{daily,weekly,monthly}
mkdir -p "$LOG_DIR"
mkdir -p "$ARCHIVE_DIR"

# Logging
LOG_FILE="$LOG_DIR/backup_${TIMESTAMP}.log"
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

# Functions
log_info() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] INFO:${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Check if PostgreSQL is running
check_postgresql() {
    if ! systemctl is-active --quiet postgresql; then
        log_error "PostgreSQL is not running!"
        exit 1
    fi
}

# Perform backup
perform_backup() {
    local backup_type=$1
    local backup_dir="$BACKUP_BASE_DIR/$backup_type"
    local backup_file="$backup_dir/backup_${backup_type}_${TIMESTAMP}.tar.gz"
    
    log_info "Starting $backup_type backup..."
    
    # Create temporary directory
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DIR" EXIT
    
    # Dump all databases
    log_info "Dumping all databases..."
    sudo -u postgres pg_dumpall --clean --if-exists > "$TEMP_DIR/all_databases.sql"
    
    # Dump globals separately (roles, tablespaces)
    log_info "Dumping global objects..."
    sudo -u postgres pg_dumpall --globals-only > "$TEMP_DIR/globals.sql"
    
    # Dump each database separately for flexibility
    databases=$(sudo -u postgres psql -t -c "SELECT datname FROM pg_database WHERE datistemplate = false AND datname != 'postgres';")
    
    for db in $databases; do
        db=$(echo $db | xargs)  # Trim whitespace
        if [ -n "$db" ]; then
            log_info "Dumping database: $db"
            sudo -u postgres pg_dump -Fc -f "$TEMP_DIR/${db}.dump" "$db"
            
            # Also create SQL format for specific tables if needed
            sudo -u postgres pg_dump -f "$TEMP_DIR/${db}.sql" "$db"
        fi
    done
    
    # Copy configuration files
    log_info "Backing up configuration files..."
    sudo cp -r /etc/postgresql/15/main "$TEMP_DIR/config"
    
    # Get database statistics
    log_info "Collecting database statistics..."
    sudo -u postgres psql -c "\l+" > "$TEMP_DIR/database_list.txt"
    sudo -u postgres psql -c "SELECT version();" > "$TEMP_DIR/version.txt"
    
    # Create backup info file
    cat > "$TEMP_DIR/backup_info.txt" <<EOF
Backup Information
==================
Type: $backup_type
Date: $(date)
PostgreSQL Version: $(sudo -u postgres psql -t -c "SELECT version();")
Backup Tool Version: pg_dump $(sudo -u postgres pg_dump --version | head -1)
System Info: $(uname -a)
Disk Usage: $(df -h "$BACKUP_BASE_DIR" | tail -1)

Databases Backed Up:
$databases

Configuration Files:
- postgresql.conf
- pg_hba.conf
- conf.d/*

Backup Contents:
$(ls -la "$TEMP_DIR")
EOF
    
    # Compress backup
    log_info "Compressing backup..."
    tar -czf "$backup_file" -C "$TEMP_DIR" .
    
    # Calculate and store checksum
    log_info "Calculating checksum..."
    sha256sum "$backup_file" > "$backup_file.sha256"
    
    # Verify backup
    log_info "Verifying backup..."
    tar -tzf "$backup_file" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        log_info "Backup verified successfully"
    else
        log_error "Backup verification failed!"
        exit 1
    fi
    
    # Log backup size
    backup_size=$(du -h "$backup_file" | cut -f1)
    log_info "Backup completed: $backup_file (Size: $backup_size)"
    
    # Clean up old backups
    cleanup_old_backups "$backup_type"
}

# Cleanup old backups
cleanup_old_backups() {
    local backup_type=$1
    local retention_days
    
    case $backup_type in
        daily)
            retention_days=$RETENTION_DAYS
            ;;
        weekly)
            retention_days=$((RETENTION_WEEKS * 7))
            ;;
        monthly)
            retention_days=$((RETENTION_MONTHS * 30))
            ;;
    esac
    
    log_info "Cleaning up $backup_type backups older than $retention_days days..."
    find "$BACKUP_BASE_DIR/$backup_type" -name "backup_*.tar.gz" -mtime +$retention_days -delete
    find "$BACKUP_BASE_DIR/$backup_type" -name "backup_*.tar.gz.sha256" -mtime +$retention_days -delete
}

# Perform WAL archiving check
check_wal_archiving() {
    log_info "Checking WAL archiving status..."
    
    archive_mode=$(sudo -u postgres psql -t -c "SHOW archive_mode;" | xargs)
    if [ "$archive_mode" = "on" ]; then
        log_info "WAL archiving is enabled"
        
        # Clean old WAL files
        log_info "Cleaning old WAL archive files..."
        find "$ARCHIVE_DIR" -name "*.partial" -mtime +1 -delete
        find "$ARCHIVE_DIR" -name "0*" -mtime +$RETENTION_DAYS -delete
    else
        log_warning "WAL archiving is not enabled. Consider enabling for point-in-time recovery."
    fi
}

# Generate backup report
generate_report() {
    local report_file="$LOG_DIR/backup_report_${DATE}.html"
    
    log_info "Generating backup report..."
    
    cat > "$report_file" <<EOF
<!DOCTYPE html>
<html>
<head>
    <title>PostgreSQL Backup Report - $DATE</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        .success { color: green; }
        .warning { color: orange; }
        .error { color: red; }
    </style>
</head>
<body>
    <h1>PostgreSQL Backup Report</h1>
    <p>Date: $(date)</p>
    
    <h2>Backup Summary</h2>
    <table>
        <tr>
            <th>Backup Type</th>
            <th>Latest Backup</th>
            <th>Size</th>
            <th>Count</th>
        </tr>
EOF
    
    for type in daily weekly monthly; do
        latest=$(ls -t "$BACKUP_BASE_DIR/$type"/backup_*.tar.gz 2>/dev/null | head -1)
        if [ -n "$latest" ]; then
            size=$(du -h "$latest" | cut -f1)
            count=$(ls "$BACKUP_BASE_DIR/$type"/backup_*.tar.gz 2>/dev/null | wc -l)
            echo "<tr><td>$type</td><td>$(basename "$latest")</td><td>$size</td><td>$count</td></tr>" >> "$report_file"
        else
            echo "<tr><td>$type</td><td class='warning'>No backups</td><td>-</td><td>0</td></tr>" >> "$report_file"
        fi
    done
    
    cat >> "$report_file" <<EOF
    </table>
    
    <h2>Database Sizes</h2>
    <table>
        <tr>
            <th>Database</th>
            <th>Size</th>
        </tr>
EOF
    
    sudo -u postgres psql -t -c "
        SELECT datname, pg_size_pretty(pg_database_size(datname))
        FROM pg_database
        WHERE datistemplate = false
        ORDER BY pg_database_size(datname) DESC;" | \
    while IFS='|' read -r db size; do
        echo "<tr><td>$db</td><td>$size</td></tr>" >> "$report_file"
    done
    
    cat >> "$report_file" <<EOF
    </table>
    
    <h2>Disk Usage</h2>
    <pre>$(df -h "$BACKUP_BASE_DIR")</pre>
    
    <h2>Recent Backup Logs</h2>
    <pre>$(tail -50 "$LOG_FILE")</pre>
</body>
</html>
EOF
    
    log_info "Report generated: $report_file"
}

# Main execution
main() {
    log_info "=== PostgreSQL Backup Script Started ==="
    
    # Check prerequisites
    check_postgresql
    
    # Determine backup type based on day
    day_of_week=$(date +%u)
    day_of_month=$(date +%d)
    
    if [ "$day_of_month" -eq 1 ]; then
        # Monthly backup on 1st of month
        perform_backup "monthly"
    elif [ "$day_of_week" -eq 7 ]; then
        # Weekly backup on Sunday
        perform_backup "weekly"
    else
        # Daily backup
        perform_backup "daily"
    fi
    
    # Check WAL archiving
    check_wal_archiving
    
    # Generate report
    generate_report
    
    # Clean up old logs
    find "$LOG_DIR" -name "backup_*.log" -mtime +30 -delete
    find "$LOG_DIR" -name "backup_report_*.html" -mtime +30 -delete
    
    log_info "=== PostgreSQL Backup Script Completed ==="
}

# Run main function
main

# Exit successfully
exit 0