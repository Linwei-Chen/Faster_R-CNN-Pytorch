# Fast_R-CNN
simple implement of Faster R-CNN with Pytorch

reference: https://github.com/potterhsu/easy-faster-rcnn.pytorch

check the link bellow to get the details:

https://zhuanlan.zhihu.com/p/62401362

# Usage

to train the faster r-cnn on voc2007&2012

1、setup the base dir of voc dataset in config.py, the dir's structure should be like this:


 ```
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
```

2、train the faster r-cnn use different backbone network by：
```
python train.py --base_net resnet18 # or resnet50, resnet101
```

3、to test the performance of faster r-cnn use the pic of test_img, run:
```
python test.py
```

# Pseudocode of the project
```
Faster R-CNN：
    features = base_net.forward(img)
    anchors = generate_anchors(features_size, anchor_sizes, anchor_ratios)
    rpn_hidden_conv_out = rpn_hidden_conv3x3_layer.forward(features)
    rpn_cls = rpn_hidden_conv1x1_layer_for_cls(rpn_hidden_conv_out)
    rpn_reg = rpn_hidden_conv1x1_layer_for_reg(rpn_hidden_conv_out)
    if training:
        # 1.rpn part
        inside_img_anchors_indices = find_anchors_inside_img(img, anchors)
        inside_img_anchors = anchors[inside_img_anchors_indice]
        all_positive_anchors_indices, labels = find_positive_anchors(gt_boxes, inside_img_anchors)
        all_negative_anchors_indices = get_exclude(inside_img_anchors, all_positive_anchors)
        shuffle(all_positive_anchors_indices)
        shuffle(all_negative_anchors_indices)
        train_sample_indices = all_positive_anchors_indices[:up_to_128] + all_negative_anchors_indices[:256 - up_to_128]
        rpn_cls_samples = rpn_cls[train_sample_indices]
        gt_labels = labels[train_sample_indices]
        rpn_reg_samples = rpn_reg[train_sample_indices]
        gt_reg = calculate_gt_reg(rpn_reg_samples, gt_boxes)
        rpn_loss = cross_entropy(rpn_cls_samples, gt_labels) + smoothL1(rpn_reg_samples, gt_reg)

        # 2.detection part
        inside_rpn_positive_probs = rpn_cls[positive_class][inside_img_anchors_indices]
        inside_rpn_reg = rpn_reg[inside_img_anchors_indices]
        # non-max-suppression to get proposals
        sorted_indices = descend_sort(inside_rpn_positive_probs)
        top_n_probs_indices = sorted_indices[:pre_nms_top_n]
        top_n_probs = inside_rpn_positive_probs[top_n_probs_indices]
        top_n_inside_img_anchors = inside_img_anchors[top_n_probs_indices]
        top_m_anchors_indices = nms(top_n_inside_img_anchors, top_n_probs, threshold, top_m)
        top_m_anchors = inside_img_anchors[top_m_anchors_indices]
        top_m_anchors_reg = inside_rpn_reg[top_m_anchors_indices]
        rpn_proposals = bounding_box_regression(top_m_anchors, top_m_anchors_reg)
        rpn_proposals_gt_reg, rpn_proposals_gt_label = get_gt(rpn_proposals, gt_labels, get_boxes)
        rpn_proposals_features = roi(rpn_proposals, features)
        detection_conv_out = detection_conv3x3_layer(rpn_proposals_features)
        detection_cls = detection_cls_fc_layer(detection_conv_out)
        detection_reg = detection_reg_fc_layer(detection_conv_out)
        detection_loss = cross_entropy(detection_cls, gt_labels) + smoothL1(detection_reg, rpn_proposals_gt_reg)

        loss = rpn_loss + detection_loss
        return loss

    else if not training:
        rpn_proposals = bounding_regression(anchors, rpn_reg)
        rpn_proposals = clip(min=(0,0), max=(img_h,img_w))
        rpn_probs = softmax(rpn_cls[positive_class])
        top_n_rpn_proposals, top_n_rpn_probs = choose_top_n(rpn_proposals, rpn_probs)
        rpn_proposals = nms(top_n_rpn_proposals, top_n_rpn_probs, threshold, top_m)
        detection_cls, detection_reg = detect_net.forwad(rpn_proposals)
        detection_probs = softmax(detection_cls)
        detection_boxes = bounding_regression(rpn_proposals, detection_reg)
        result_dict = {}
        for class in classes:
            result_dict[class] = nms(detection_boxes, detection_probs[class], threshold)

        return result_dict
```

# Demo Pic
![](https://pic4.zhimg.com/80/v2-5f1f4c292cb7cb879cde54c7cd186287_hd.jpg)

![](https://pic3.zhimg.com/80/v2-bcb92476324b0113fb0979640effd9b6_hd.jpg)

![](https://pic4.zhimg.com/80/v2-1cfe0fa72af880305dfa3c427622a4cb_hd.jpg)

![](https://pic3.zhimg.com/80/v2-2af7ec7553625fa2ea92c502623a28da_hd.jpg)

![](https://pic2.zhimg.com/80/v2-3f9847221dcb3f1d09c215f4651121bd_hd.jpg)

![](https://pic2.zhimg.com/80/v2-426b46425f15823f9a9ac4ba2edc9ec5_hd.jpg)

![](https://pic2.zhimg.com/80/v2-8e080f7f5fcdcdf0f8db8d477f9942c5_hd.jpg)

![](https://pic1.zhimg.com/80/v2-391ddf0105a3b995528da72e941b2a0c_hd.jpg)

![](https://pic3.zhimg.com/80/v2-87dfcea8f15e26931e44fe50ecab3bc2_hd.jpg)

![](https://pic2.zhimg.com/80/v2-4687c1afce158f67174faa2616872e91_hd.jpg)

![](https://pic2.zhimg.com/80/v2-86db7c66a4a8602be8730f40e5a99539_hd.jpg)

![](https://pic1.zhimg.com/80/v2-6b31549ba305105ba353f70b6507a59c_hd.jpg)
