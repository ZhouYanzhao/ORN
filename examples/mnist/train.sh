#!/usr/bin/env sh
DIR="$( cd "$(dirname "$0")" ; pwd -P )"
mkdir -p $DIR/snapshot
cd $DIR

# Train ORN
$DIR/../../build/tools/caffe train \
--solver=orn_solver.prototxt \
-gpu all 2>&1 | tee $DIR/snapshot/Rot_ORN_8_Pooling.log

# Train Baseline CNN
$DIR/../../build/tools/caffe train \
--solver=cnn_solver.prototxt \
-gpu all 2>&1 | tee $DIR/snapshot/Rot_CNN.log


# Compare results
python utils.py eval --log="$DIR/snapshot/Rot_CNN.log"
python utils.py eval --log="$DIR/snapshot/Rot_ORN_8_Pooling.log"