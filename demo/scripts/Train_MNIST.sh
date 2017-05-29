#!/usr/bin/env bash
# utilities
function init { 
    if [ -z "${!1}" ]; then export $1=$2; fi
}
function define { 
    export $1=$2
}
function waiting {
    for pid in "$@"
    do
        while [ -e /proc/$pid ]
        do
            sleep 1
        done
    done
}

# global settings
init enableSave true
init notify "false"
init maxEpoch 200
init learningRateDecayRatio 0.5
init removeOldCheckpoints false
init optimMethod "adadelta"
init batchSize 128
init gpuDevice "{1,2}"

# tasks
function Rot_CNN {
    define dataset "MNIST-rot"
    define model "CNN"
    define savePath "logs/Rot_CNN"
    define note "Rot_CNN"
    define customParams "{rho=0.9,eps=1e-6}"
    th train.lua
}

function Rot_ORN_8_Align {
    define dataset "MNIST-rot"
    define model "ORN"
    define savePath "logs/Rot_ORN_8_Align"
    define note "Rot_ORN_8_Align"
    define customParams "{rho=0.9,eps=1e-6,orientation=8,useORAlign=true,useORPooling=false}"
    th train.lua
}

# run tasks
PID=""
(Rot_CNN; Rot_ORN_8_Align) &
PID="$PID $!"
waiting $PID
