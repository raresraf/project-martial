#!/bin/bash

list_of_dirs=(
    "/Users/raresraf/code/project-martial/experimental/network/mysql_5_6/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/experimental/network/mysql_5_6/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/experimental/network/mysql_5_6_gcp/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/experimental/network/mysql_5_6_gcp/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/experimental/network/mysql_5_7/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/experimental/network/mysql_5_7/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/experimental/network/mysql_5_7_gcp/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/experimental/network/mysql_5_7_gcp/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/experimental/network/mysql_8_0/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/experimental/network/mysql_8_0/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/experimental/network/mysql_8_0_gcp/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/experimental/network/mysql_8_0_gcp/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/experimental/network/postgres_9_6/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/experimental/network/postgres_9_6/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/experimental/network/postgres_10/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/experimental/network/postgres_10/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/experimental/network/postgres_11/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/experimental/network/postgres_11/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/experimental/network/postgres_12/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/experimental/network/postgres_12/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/experimental/network/postgres_13/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/experimental/network/postgres_13/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/experimental/network/postgres_14/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/experimental/network/postgres_14/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/experimental/network/postgres_15/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/experimental/network/postgres_15/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/experimental/network/postgres_16/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/experimental/network/postgres_16/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/experimental/network/sqlserver_2017_dev/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/experimental/network/sqlserver_2017_dev/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/experimental/network/sqlserver_2019_dev/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/experimental/network/sqlserver_2019_dev/scenarios/scenario_2"
    "/Users/raresraf/code/project-martial/experimental/network/sqlserver_2022_dev/scenarios/scenario_1"
    "/Users/raresraf/code/project-martial/experimental/network/sqlserver_2022_dev/scenarios/scenario_2"
)


for directory in "${list_of_dirs[@]}"; do
  cd "$directory" || continue
  find . -not -name 'combined.bin' -name "*.bin" -print0 | xargs -0 sh -c 'for f; do cat "$f"; printf "\4"; done' > combined.bin
  cd -
done