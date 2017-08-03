#!/usr/bin/env sh

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
mkdir -p $DIR/dataset
cd $DIR/dataset

echo "Downloading original MNIST..."

for fname in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte
do
    if [ ! -e $fname ]; then
        wget --no-check-certificate http://yann.lecun.com/exdb/mnist/${fname}.gz
        gunzip ${fname}.gz
    fi
done

echo "Generating MNIST-Rot..."
python ../utils.py rotate --src-idx="train-images-idx3-ubyte" --dst-idx="train-rot-images-idx3-ubyte"
python ../utils.py rotate --src-idx="t10k-images-idx3-ubyte" --dst-idx="t10k-rot-images-idx3-ubyte"

echo "Creating LMDB..."
BUILD=$DIR/../../build/examples/mnist
BACKEND="lmdb"
rm -rf mnist_rot_train_${BACKEND}
rm -rf mnist_rot_test_${BACKEND}
$BUILD/convert_mnist_data.bin train-rot-images-idx3-ubyte \
  train-labels-idx1-ubyte mnist_rot_train_${BACKEND} --backend=${BACKEND}
$BUILD/convert_mnist_data.bin t10k-rot-images-idx3-ubyte \
  t10k-labels-idx1-ubyte mnist_rot_test_${BACKEND} --backend=${BACKEND}

echo "Done."