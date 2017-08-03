#!/usr/bin/env python
import os
from setuptools import setup, find_packages

setup(
    name="orn",
    version="1.0",
    description="Oriented Response Networks, in CVPR2017",
    url="https://ZhouYanzhao.github.io/ORN",
    author="Yanzhao Zhou",
    author_email="zhouyanzhao215@mails.ucas.ac.cn",
    # Require cffi.
    install_requires=["cffi>=1.0.0"],
    setup_requires=["cffi>=1.0.0"],
    # Exclude the build files.
    packages=find_packages(exclude=["build"]),
    # Package where to put the extensions. Has to be a prefix of build.py.
    ext_package="",
    # Extensions to compile.
    cffi_modules=[
        os.path.join(os.path.dirname(__file__), "build.py:ffi")
    ],
)
