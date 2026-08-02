#!/bin/bash

rm -rf ~/mlx-models/GLM-5.2-colibri-int4-g64-with-int8-mtp

python3 ./coli convert \
    --model /mnt/zfs1/noprot/mlx-lm/models/GLM-5.2-colibri-int4-g64-with-int8-mtp \
    --outdir ~/mlx-models/GLM-5.2-colibri-int4-perrow \
    --group-size 0
