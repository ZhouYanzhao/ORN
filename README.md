# Oriented Response Networks (PyTorch)
[[project]](http://zhouyanzhao.github.io/ORN) [[doc]](http://github.com/ZhouYanzhao/ORN) [[arXiv]](https://arxiv.org/pdf/1701.01833)
> Reimplemented to be compatiable with modern versions of PyTorch (tested with 1.12.0).

Note that this simplified version only supports 1x1/3x3 kernels with 4/8 orientation channels. Please check the [master branch](http://github.com/ZhouYanzhao/ORN) for more information.

## Install
```bash
cd path_to_the_setup.py
pip install .
```

## Quick test (rotated MNIST)
```bash
# Train Baseline
python test/mnist-rot/main.py

# Train ORN
python test/mnist-rot/main.py --use-arf
```

## Use ORN in your own project
```python
# 1. Use predefined ORN-upgraded models (VGG, ResNet, etc.)
from orn import models
model = models.or_resnet18_align_8()
# Print the full list
print('\n'.join(dir(models)))

# 2. Use the helper function for model conversion
from torchvision import models
from orn import upgrade_to_orn
model = models.resnet18(weights=None)
upgrade_to_orn(model, num_orientation=8, scale_factor=2,
    classifier=model.fc, features=model.avgpool, invariant_encoding='align')
print(model)

# 3. Use ORN layers
from orn import ORConv, ORAlign1d, ORPool1d
```

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
