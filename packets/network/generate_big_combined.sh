#!/bin/bash

find . -not -name 'combined.bin' -name "*.bin" -print0 | xargs -0 sh -c 'for f; do cat "$f"; printf "\4"; done' > combined.bin