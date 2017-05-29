# Oriented Response Networks
[[project]](http://zhouyanzhao.github.io/ORN) [[doc]](http://github.com/ZhouYanzhao/ORN) [[arXiv]](https://arxiv.org/pdf/1701.01833)

The torch branch contains:
* the official **torch** implementation of ORN.
* the **MNIST-Variants** demo.

Please follow the instruction below to install it and run the experiment demo. Check the [[doc]](http://github.com/ZhouYanzhao/ORN) for more information.

## Setup
### Prerequisites
* Linux (only tested on ubuntu 14.04LTS)
* NVIDIA GPU + CUDA CuDNN (CPU mode and CUDA without CuDNN mode are also available but significantly slower)
* [Torch7](http://torch.ch/docs/getting-started.html)

### Getting started
You can setup everything via a single command `wget -O - https://git.io/vHCMI | bash` **or** do it manually in case something goes wrong:

1. install the dependencies (required by the demo code):
    * [torchnet](https://github.com/torchnet/torchnet): `luarocks install torchnet`
    * [optnet](https://github.com/fmassa/optimize-net): `luarocks install optnet`

2. clone the torch branch: 

	```bash
	# git version must be greater than 1.9.10
	git clone https://github.com/ZhouYanzhao/ORN.git -b torch --single-branch ORN.torch
	cd ORN.torch
	export DIR=$(pwd)
	```

3. install ORN: 

    ```bash
    cd $DIR/install
    # install the CPU/GPU/CuDNN version ORN.
    bash install.sh
    ```

4. unzip the MNIST dataset:

    ```bash
    cd $DIR/demo/datasets
    unzip MNIST
    ```

5. run the MNIST-Variants demo:

    ```bash
    cd $DIR/demo
    # you can modify the script to test different hyper-parameters
    bash ./scripts/Train_MNIST.sh
    ```

### Trouble shooting
If you run into `'cudnn.find' not found`, update Torch7 to the latest version via `cd <TORCH_DIR> && bash ./update.sh` then re-install everything.

## Citation 
If you use the code in your research, please cite:
```bibtex
@INPROCEEDINGS{Zhou2017ORN,
    author = {Zhou, Yanzhao and Ye, Qixiang and Qiu, Qiang and Jiao, Jianbin},
    title = {Oriented Response Networks},
    booktitle = {CVPR},
    year = {2017}
}
```