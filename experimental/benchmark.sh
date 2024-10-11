#!/bin/bash

# Database configuration
DB_HOST="35.202.15.101"
DB_USER="root"
DB_PASS="mateinfo"
DB_NAME="benchmark_db"
TABLE_NAME="test_table"

# Benchmark parameters
NUM_ROWS=100    # Number of rows to insert
BATCH_SIZE=10     # Number of rows per insert batch
NUM_SELECTS=100     # Number of SELECT queries to run
NUM_UPDATES=100     # Number of UPDATE statements to run

# Create the database
mysql --ssl-mode=DISABLED -h "$DB_HOST" -u "$DB_USER" -p"$DB_PASS" -e "CREATE DATABASE IF NOT EXISTS $DB_NAME;"

# Create the table
mysql --ssl-mode=DISABLED -h "$DB_HOST" -u "$DB_USER" -p"$DB_PASS" "$DB_NAME" << EOF
CREATE TABLE IF NOT EXISTS $TABLE_NAME (
  id INT PRIMARY KEY AUTO_INCREMENT,
  col1 VARCHAR(255),
  col2 INT,
  col3 DATETIME
);
EOF

# Generate and insert data
echo "Generating and inserting data..."
for i in $(seq 1 $NUM_ROWS); do
  UUID=$(uuidgen)
  RANDOM_NUMBER=$(( ( RANDOM % 1000 )  + 1 ))
  CURRENT_TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
  
  SQL_INSERT="INSERT INTO $TABLE_NAME (col1, col2, col3) VALUES ('$UUID', $RANDOM_NUMBER, '$CURRENT_TIMESTAMP');"
  mysql --ssl-mode=DISABLED -h "$DB_HOST" -u "$DB_USER" -p"$DB_PASS" "$DB_NAME" -e "$SQL_INSERT"

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

  mysql --ssl-mode=DISABLED -h "$DB_HOST" -u "$DB_USER" -p"$DB_PASS" "$DB_NAME" -e "$SQL_SELECT" > /dev/null
done
echo "SELECT benchmark completed!"

echo "Starting UPDATE benchmark..."
# Perform UPDATE statements
for i in $(seq 1 $NUM_UPDATES); do
  RANDOM_ID=$(( ( RANDOM % $NUM_ROWS )  + 1 ))
  NEW_RANDOM_NUMBER=$(( ( RANDOM % 1000 )  + 1 ))

  SQL_UPDATE="UPDATE $TABLE_NAME SET col2 = $NEW_RANDOM_NUMBER WHERE id = $RANDOM_ID;"

  mysql --ssl-mode=DISABLED -h "$DB_HOST" -u "$DB_USER" -p"$DB_PASS" "$DB_NAME" -e "$SQL_UPDATE" > /dev/null
done
echo "UPDATE benchmark completed!"

