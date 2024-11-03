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
)


for directory in "${list_of_dirs[@]}"; do
  cd "$directory" || continue
  find . -not -name 'combined.bin' -name "*.bin" -print0 | xargs -0 sh -c 'for f; do cat "$f"; printf "\4"; done' > combined.bin
  cd -
done