# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: complete_sample.py
# Python  : python3.6
# Time    : 18-7-9 22:42
# Github  : https://github.com/Super-Louis

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def distort_color(image, color_ordering=0):
    if color_ordering == 0: # 图像处理顺序
        image = tf.image.random_brightness(image, max_delta=32/255) # 亮度
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5) # 饱和度
        image = tf.image.random_hue(image, max_delta=0.2) # 色相
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5) # 对比度
    elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32/255)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering == 2:
        ...
    return tf.clip_by_value(image, 0, 1)

def preprocess_for_train(image, height, width, bbox):
    """
    对原始图像进行预处理
    :param image:
    :param height:
    :param width:
    :param bbox: 标注框
    :return:
    """
    if bbox is None: # 如果没有提供标注框，则认为整个图像就是需要关注的部分
        bbox = tf.constant([0,0,1,1], dtype=tf.float32, shape=[1,1,4])

    # 转换图像张量的类型
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # 随机截取图像，减小需要关注的物体大小对算法的影响
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image), bounding_boxes=bbox
    )
    distorted_image = tf.slice(image, bbox_begin, bbox_size)
    # 采用随机方法resize图片
    distorted_image = tf.image.resize_images(distorted_image, [height, width],
                                             method=np.random.randint(4))
    # 随机左右翻转图像
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    # 使用一种随机的顺序调整图像色彩
    distorted_image = distort_color(distorted_image, np.random.randint(2))
    return distorted_image

image_raw_data = tf.gfile.FastGFile('xmind.jpeg', 'rb').read()
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
    # 运行6次得到6种不同的图像
    for i in range(6):
        result = preprocess_for_train(img_data, 299, 299, boxes)
        plt.imshow(result.eval())
        plt.show()