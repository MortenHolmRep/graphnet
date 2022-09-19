#!/bin/bash

db=
out="/groups/icecube/qgf305/storage/test/combined_angle/"
gpu=1

python train_model.py --database ${db} --output ${out} --gpu ${gpu}