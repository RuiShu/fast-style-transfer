#! /bin/bash
mkdir -p data/train
cd data/train
wget http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
unzip train2014.zip
