# -*- coding: utf-8 -*-
import sys
import re
import random
import idx2numpy
import argparse
import numpy as np
from scipy.ndimage import rotate
from scipy.misc import imresize
from os.path import basename, abspath, splitext

parser = argparse.ArgumentParser(description="ORN.Caffe MNIST Example")
parser.add_argument("cmd", help="rotate / eval")
parser.add_argument("--src-idx", default="./dataset/train-images-idx3-ubyte", metavar='I',
                    help="path to the original MNIST idx data")
parser.add_argument("--dst-idx", default="./dataset/train-rot-images-idx3-ubyte", metavar='O',
                    help="path to the MNIST-rot idx data")
parser.add_argument("--size", type=int, default=32, metavar='S',
                    help="image size of the MNIST-rot")
parser.add_argument("--log", default="./snapshot/Rot_CNN.log", metavar='L',
                    help="path to the log file")
parser.add_argument("--title", default=None, metavar='T',
                    help="display title for eval results")

args = parser.parse_args()

if args.cmd == "eval":
    with open(args.log, 'r') as file:
        title = args.title if args.title else splitext(basename(abspath(args.log)))[0]
        content = ''.join(file.readlines())
        best_acc = 0.0
        def repl(m):
            global best_acc
            acc = float(m.groups(1)[0])
            if acc > best_acc:
                best_acc = acc
            return None
        re.sub(r"accuracy = ([\d\.]+)", repl, content)
        print('\n'.join([
            "[{}]",
            "Best Test Accuracy = {}%",
            "Best Test Error Rate = {}%"]).format(title, best_acc*100, (1-best_acc)*100))

elif args.cmd == "rotate":
    src = idx2numpy.convert_from_file(args.src_idx)
    size = args.size
    count = src.shape[0]
    dst = np.ndarray((count, size, size), dtype="uint8")
    for i in range(0, count):
        dst[i] = rotate(imresize(src[i], size=(size, size), interp='bilinear'), random.randint(0, 360), reshape=False)
    idx2numpy.convert_to_file(args.dst_idx, dst)

else:
    print("Invalid cmd [{}]".format(args.cmd))