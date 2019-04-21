# Fast_R-CNN
simple implement of Faster R-CNN with Pytorch

reference: https://github.com/potterhsu/easy-faster-rcnn.pytorch

check the link bellow to get the details:

https://zhuanlan.zhihu.com/p/62401362

# Usage

to train the faster r-cnn on voc2007&2012

1、setup the base dir of voc dataset in config.py, the dir's structure should be like this:

base_dir
   VOC2007
      Annotations
      ImageSets
      JPEGImages
      SegmentationClass
      SegmentationObject
   VOC2012
      Annotations
      ImageSets
      JPEGImages
      SegmentationClass
      SegmentationObject
