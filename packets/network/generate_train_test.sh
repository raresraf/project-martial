#!/bin/bash

list_of_dirs_train=(
    "/Users/raresraf/code/project-martial/packets/network/mysql_5_6/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/packets/network/mysql_5_6_gcp/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/packets/network/mysql_5_7/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/packets/network/mysql_5_7_gcp/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/packets/network/mysql_8_0/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/packets/network/mysql_8_0_gcp/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/packets/network/postgres_9_6/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/packets/network/postgres_9_6_gcp/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/packets/network/postgres_10/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/packets/network/postgres_10_gcp/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/packets/network/postgres_11/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/packets/network/postgres_11_gcp/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/packets/network/postgres_12/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/packets/network/postgres_12_gcp/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/packets/network/postgres_13/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/packets/network/postgres_13_gcp/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/packets/network/postgres_14/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/packets/network/postgres_14_gcp/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/packets/network/postgres_15/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/packets/network/postgres_15_gcp/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/packets/network/postgres_16/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/packets/network/postgres_16_gcp/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/packets/network/sqlserver_2017_dev/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/packets/network/sqlserver_2017_standard_gcp/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/packets/network/sqlserver_2017_enterprise_gcp/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/packets/network/sqlserver_2019_dev/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/packets/network/sqlserver_2019_standard_gcp/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/packets/network/sqlserver_2019_enterprise_gcp/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/packets/network/sqlserver_2022_dev/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/packets/network/sqlserver_2022_standard_gcp/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/packets/network/sqlserver_2022_enterprise_gcp/scenarios/scenario_2"
)


for directory in "${list_of_dirs_train[@]}"; do
  cd "$directory" || continue
  db_version=$(echo "$directory" | awk -F'/' '{print $(NF-2)}')
  mkdir -p "/Users/raresraf/code/project-martial/train/$db_version/"
  find . -not -name 'combined.bin' -name "*.bin" -print0 | xargs -0 sh -c 'for f; do cat "$f"; printf "\4"; done' > "/Users/raresraf/code/project-martial/train/$db_version/combined.train"
  cd -
done


list_of_dirs_test=(
    "/Users/raresraf/code/project-martial/packets/network/mysql_5_6/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/packets/network/mysql_5_6/scenarios/scenario_3"
    "/Users/raresraf/code/project-martial/packets/network/mysql_5_6_gcp/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/packets/network/mysql_5_7/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/packets/network/mysql_5_7/scenarios/scenario_3"
    "/Users/raresraf/code/project-martial/packets/network/mysql_5_7_gcp/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/packets/network/mysql_8_0/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/packets/network/mysql_8_0/scenarios/scenario_3"
    "/Users/raresraf/code/project-martial/packets/network/mysql_8_0_gcp/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/packets/network/postgres_9_6/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/packets/network/postgres_9_6_gcp/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/packets/network/postgres_10/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/packets/network/postgres_10_gcp/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/packets/network/postgres_11/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/packets/network/postgres_11_gcp/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/packets/network/postgres_12/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/packets/network/postgres_12_gcp/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/packets/network/postgres_13/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/packets/network/postgres_13_gcp/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/packets/network/postgres_14/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/packets/network/postgres_14_gcp/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/packets/network/postgres_15/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/packets/network/postgres_15_gcp/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/packets/network/postgres_16/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/packets/network/postgres_16_gcp/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/packets/network/sqlserver_2017_dev/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/packets/network/sqlserver_2017_standard_gcp/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/packets/network/sqlserver_2017_enterprise_gcp/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/packets/network/sqlserver_2019_dev/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/packets/network/sqlserver_2019_standard_gcp/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/packets/network/sqlserver_2019_enterprise_gcp/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/packets/network/sqlserver_2022_dev/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/packets/network/sqlserver_2022_standard_gcp/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/packets/network/sqlserver_2022_enterprise_gcp/scenarios/scenario_1"
)


for directory in "${list_of_dirs_test[@]}"; do
  cd "$directory" || continue
  db_version=$(echo "$directory" | awk -F'/' '{print $(NF-2)}')
  mkdir -p "/Users/raresraf/code/project-martial/test/$db_version/"
  find . -not -name 'combined.bin' -name "*.bin" -print0 | xargs -0 sh -c 'for f; do cat "$f"; printf "\4"; done' > "/Users/raresraf/code/project-martial/test/$db_version/combined.test"
  cd -
done
