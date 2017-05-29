#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Install CPU version..."
cd $DIR/cpu
rm -r ./build
luarocks make orn-scm-1.rockspec

echo "Install GPU version..."
cd $DIR/gpu
rm -r ./build
luarocks make cuorn-scm-1.rockspec

echo "Install CuDNN version..."
cd $DIR/cudnn
rm -r ./build
luarocks make cudnnorn-scm-1.rockspec

echo "-------------------------------"
echo "All done!"

