#!/bin/bash
export PYTHONPATH=/home/i40/jenkea/PycharmProjects/V-GAN/src
while true; do
    python3 text/od_experiment.py
    echo "Script crashed. Restarting..." >&2
    sleep 1
done