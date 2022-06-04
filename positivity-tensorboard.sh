#!/bin/bash
killall tensorboard
sleep 1
tensorboard --host 0.0.0.0 --logdir ./runs &> /dev/null &
./positivity.py "$@"
