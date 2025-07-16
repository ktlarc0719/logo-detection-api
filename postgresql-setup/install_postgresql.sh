#!/bin/bash
# PostgreSQL 15+ Installation Script for WSL2 Ubuntu
# For Machine Learning Applications with High-Performance Hardware

set -e

echo "=================================================="
echo "PostgreSQL Installation Script for WSL2"
echo "Optimized for Ryzen 9900X + 128GB Memory"
echo "=================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if running on WSL
if ! grep -q Microsoft /proc/version; then
    print_error "This script is designed for WSL2. Please run on WSL2 Ubuntu."
    exit 1
fi

# Update system packages
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install PostgreSQL 15
print_status "Installing PostgreSQL 15..."
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
sudo apt update
sudo apt install -y postgresql-15 postgresql-client-15 postgresql-contrib-15 postgresql-15-pgvector

# Install additional tools
print_status "Installing additional PostgreSQL tools..."
sudo apt install -y postgresql-server-dev-15 libpq-dev pgadmin4

# Stop PostgreSQL service temporarily
print_status "Stopping PostgreSQL service for configuration..."
sudo systemctl stop postgresql

# Create data directory on Windows-mounted SSD
print_status "Setting up data directory on Windows SSD..."
DATA_DIR="/mnt/data-ssd/postgresql"
sudo mkdir -p "$DATA_DIR/15/main"
sudo chown -R postgres:postgres "$DATA_DIR"
sudo chmod 700 "$DATA_DIR/15/main"

# Backup original configuration
print_status "Backing up original configuration..."
sudo cp /etc/postgresql/15/main/postgresql.conf /etc/postgresql/15/main/postgresql.conf.backup
sudo cp /etc/postgresql/15/main/pg_hba.conf /etc/postgresql/15/main/pg_hba.conf.backup

# Move existing data to new location (if exists)
if [ -d "/var/lib/postgresql/15/main" ]; then
    print_status "Moving existing data to new location..."
    sudo -u postgres rsync -av /var/lib/postgresql/15/main/ "$DATA_DIR/15/main/"
fi

# Update data directory in configuration
print_status "Updating PostgreSQL configuration..."
sudo sed -i "s|^data_directory = .*|data_directory = '$DATA_DIR/15/main'|" /etc/postgresql/15/main/postgresql.conf

# Create initial configuration (Phase 1 - Conservative)
cat << EOF | sudo tee /etc/postgresql/15/main/conf.d/01-initial.conf
# Initial Conservative Configuration for ML Workloads
# Phase 1 - Safe starting point

# Memory Settings (Conservative)
shared_buffers = 8GB                    # 8GB for initial setup
effective_cache_size = 32GB             # 25% of total RAM
maintenance_work_mem = 2GB              # For VACUUM, CREATE INDEX
work_mem = 256MB                        # Per operation memory

# Connection Settings
max_connections = 50                    # Conservative for development
superuser_reserved_connections = 3

# Parallel Processing (Ryzen 9900X optimization)
max_parallel_workers_per_gather = 4     # Start conservative
max_parallel_workers = 8
max_parallel_maintenance_workers = 4
parallel_leader_participation = on

# Write Performance
checkpoint_timeout = 15min
checkpoint_completion_target = 0.9
max_wal_size = 4GB
min_wal_size = 1GB

# Query Planning
random_page_cost = 1.1                  # SSD optimization
effective_io_concurrency = 200          # SSD optimization
default_statistics_target = 100

# Logging
log_destination = 'stderr'
logging_collector = on
log_directory = 'log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_file_mode = 0600
log_truncate_on_rotation = off
log_rotation_age = 1d
log_rotation_size = 1GB
log_min_duration_statement = 1000       # Log queries > 1 second
log_checkpoints = on
log_connections = on
log_disconnections = on
log_duration = off
log_line_prefix = '%m [%p] %q%u@%d '
log_statement = 'ddl'
log_timezone = 'UTC'

# Statistics
track_activities = on
track_counts = on
track_io_timing = on
track_functions = all
track_activity_query_size = 4096

# Autovacuum (important for large datasets)
autovacuum = on
autovacuum_max_workers = 4
autovacuum_naptime = 30s
autovacuum_vacuum_threshold = 50
autovacuum_analyze_threshold = 50
autovacuum_vacuum_scale_factor = 0.05
autovacuum_analyze_scale_factor = 0.05
autovacuum_vacuum_cost_delay = 10ms
autovacuum_vacuum_cost_limit = 1000

# Extensions
shared_preload_libraries = 'pg_stat_statements'
EOF

# Create directory for additional configurations
sudo mkdir -p /etc/postgresql/15/main/conf.d
echo "include_dir = 'conf.d'" | sudo tee -a /etc/postgresql/15/main/postgresql.conf

# Update pg_hba.conf for development access
print_status "Configuring authentication..."
cat << EOF | sudo tee /etc/postgresql/15/main/pg_hba.conf
# PostgreSQL Client Authentication Configuration File
# TYPE  DATABASE        USER            ADDRESS                 METHOD

# Local connections
local   all             postgres                                peer
local   all             all                                     md5

# IPv4 local connections:
host    all             all             127.0.0.1/32            md5
host    all             all             172.16.0.0/12           md5  # WSL network

# IPv6 local connections:
host    all             all             ::1/128                 md5

# Allow connections from Windows host (customize as needed)
# host    all             all             0.0.0.0/0               md5  # DANGEROUS - only for isolated dev
EOF

# Initialize the new data directory
print_status "Initializing new data directory..."
sudo -u postgres /usr/lib/postgresql/15/bin/initdb -D "$DATA_DIR/15/main" --encoding=UTF8 --locale=en_US.UTF-8

# Start PostgreSQL
print_status "Starting PostgreSQL service..."
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Wait for PostgreSQL to start
sleep 5

# Set postgres user password
print_status "Setting postgres user password..."
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';"

# Create pg_stat_statements extension
print_status "Creating extensions..."
sudo -u postgres psql -c "CREATE EXTENSION IF NOT EXISTS pg_stat_statements;"
sudo -u postgres psql -c "CREATE EXTENSION IF NOT EXISTS pgcrypto;"
sudo -u postgres psql -c "CREATE EXTENSION IF NOT EXISTS uuid-ossp;"

# Create a development database
print_status "Creating development database..."
sudo -u postgres createdb ml_development

# Create systemd service for WSL auto-start
print_status "Setting up auto-start service..."
cat << 'EOF' | sudo tee /etc/init.d/postgresql-wsl
#!/bin/bash
### BEGIN INIT INFO
# Provides:          postgresql-wsl
# Required-Start:    $remote_fs $syslog
# Required-Stop:     $remote_fs $syslog
# Default-Start:     2 3 4 5
# Default-Stop:      0 1 6
# Short-Description: PostgreSQL for WSL
# Description:       Start PostgreSQL on WSL boot
### END INIT INFO

case "$1" in
  start)
    echo "Starting PostgreSQL for WSL..."
    sudo service postgresql start
    ;;
  stop)
    echo "Stopping PostgreSQL for WSL..."
    sudo service postgresql stop
    ;;
  restart)
    echo "Restarting PostgreSQL for WSL..."
    sudo service postgresql restart
    ;;
  status)
    sudo service postgresql status
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status}"
    exit 1
    ;;
esac
EOF

sudo chmod +x /etc/init.d/postgresql-wsl
sudo update-rc.d postgresql-wsl defaults

# Create WSL startup script
print_status "Creating WSL startup configuration..."
cat << 'EOF' | tee ~/.postgresql_wsl_startup.sh
#!/bin/bash
# PostgreSQL WSL Startup Script
if service postgresql status | grep -q "down"; then
    echo "Starting PostgreSQL..."
    sudo service postgresql start
fi
EOF

chmod +x ~/.postgresql_wsl_startup.sh

# Add to .bashrc for auto-start
if ! grep -q "postgresql_wsl_startup" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# Auto-start PostgreSQL" >> ~/.bashrc
    echo "~/.postgresql_wsl_startup.sh" >> ~/.bashrc
fi

# Create connection info file
print_status "Creating connection information..."
cat << EOF > ~/postgresql_connection_info.txt
PostgreSQL Installation Complete!
================================

Connection Information:
- Host: localhost (or WSL IP from Windows)
- Port: 5432
- Username: postgres
- Password: postgres
- Database: ml_development

From WSL:
  psql -U postgres -d ml_development

From Windows:
  Use WSL IP address (find with: wsl hostname -I)
  psql -h <WSL_IP> -U postgres -d ml_development

Data Directory: $DATA_DIR/15/main
Configuration: /etc/postgresql/15/main/postgresql.conf
Additional Configs: /etc/postgresql/15/main/conf.d/

Next Steps:
1. Review and adjust memory settings in conf.d/01-initial.conf
2. Create application-specific users and databases
3. Configure backup strategies
4. Monitor performance and adjust settings
EOF

print_status "Installation complete!"
echo ""
cat ~/postgresql_connection_info.txt
echo ""
print_warning "Remember to:"
print_warning "1. Restart your WSL session for auto-start to take effect"
print_warning "2. Create the /mnt/data-ssd mount point if it doesn't exist"
print_warning "3. Adjust Windows Firewall if external access is needed"