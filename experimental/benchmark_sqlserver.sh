#!/bin/bash

# Database configuration
DB_HOST="109.103.170.30"
DB_USER="sa"
DB_PASS="Mateinfo22"
DB_NAME="benchmark_db"
TABLE_NAME="test_table"

# Benchmark parameters
NUM_ROWS=100     # Number of rows to insert
BATCH_SIZE=10     # Number of rows per insert batch
NUM_SELECTS=100     # Number of SELECT queries to run
NUM_UPDATES=100     # Number of UPDATE statements to run

# Create the database
sqlcmd -N disable -C -S "$DB_HOST" -U "$DB_USER" -P "$DB_PASS" -Q "IF NOT EXISTS(SELECT * FROM sys.databases WHERE name = '$DB_NAME') CREATE DATABASE $DB_NAME;"

# Create the table
sqlcmd -N disable -C -S "$DB_HOST" -U "$DB_USER" -P "$DB_PASS" -d "$DB_NAME" -Q "IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='$TABLE_NAME' AND xtype='U') CREATE TABLE $TABLE_NAME (
    id INT PRIMARY KEY IDENTITY(1,1),
    col1 VARCHAR(255),
    col2 INT,
    col3 DATETIME
);"

# Generate and insert data
echo "Generating and inserting data..."
for i in $(seq 1 $NUM_ROWS); do
    UUID=$(uuidgen)
    RANDOM_NUMBER=$(( ( RANDOM % 1000 )  + 1 ))
    CURRENT_TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

    SQL_INSERT="INSERT INTO $TABLE_NAME (col1, col2, col3) VALUES ('$UUID', $RANDOM_NUMBER, '$CURRENT_TIMESTAMP');"
    sqlcmd -N disable -C -S "$DB_HOST" -U "$DB_USER" -P "$DB_PASS" -d "$DB_NAME" -Q "$SQL_INSERT"

    if (( $i % $BATCH_SIZE == 0 )); then
        echo "Inserted $i rows..."
    fi
done
echo "Data generation complete!"

echo "Starting SELECT benchmark..."
# Perform SELECT queries
for i in $(seq 1 $NUM_SELECTS); do
    RANDOM_ID=$(( ( RANDOM % $NUM_ROWS )  + 1 ))
    SQL_SELECT="SELECT * FROM $TABLE_NAME WHERE id = $RANDOM_ID;"

    sqlcmd -N disable -C -S "$DB_HOST" -U "$DB_USER" -P "$DB_PASS" -d "$DB_NAME" -Q "$SQL_SELECT" > /dev/null
done
echo "SELECT benchmark completed!"

echo "Starting UPDATE benchmark..."
# Perform UPDATE statements
for i in $(seq 1 $NUM_UPDATES); do
    RANDOM_ID=$(( ( RANDOM % $NUM_ROWS )  + 1 ))
    NEW_RANDOM_NUMBER=$(( ( RANDOM % 1000 )  + 1 ))

    SQL_UPDATE="UPDATE $TABLE_NAME SET col2 = $NEW_RANDOM_NUMBER WHERE id = $RANDOM_ID;"

    sqlcmd -N disable -C -S "$DB_HOST" -U "$DB_USER" -P "$DB_PASS" -d "$DB_NAME" -Q "$SQL_UPDATE" > /dev/null
done
echo "UPDATE benchmark completed!"

