#!/bin/bash
HOME=$(pwd)
echo "Compiling cuda kernels..."
cd $HOME/orn/src
rm liborn_kernel.cu.o
nvcc -c -o liborn_kernel.cu.o liborn_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_35
echo "Installing extension..."
cd $HOME
python setup.py clean && python setup.py install
