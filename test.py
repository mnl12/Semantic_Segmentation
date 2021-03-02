from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow_examples.models.pix2pix import pix2pix
from PIL import Image
from pascal_voc_cmap import color_map, colors2labels, onehot2mask, labels2colors
from models import segmentation_network
from my_metrics import my_miou_metric, my_accuarcy_metric, other_accuracy, me_others_iou, general_iou
from IPython.display import clear_output
import os

def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


seed=1
any_size=1
BATCH_SIZE = 32
IMAGE_SIZE=(512,512)
n_classes=21
model_name='deeplab_v3'
base_model_name='resnet'
db_path='../dataset/VOCdevkit/VOC2012'
img_path='/JPEGImages/'
mask_path='/SegmentationClass/'
c_map=color_map(n_classes)



if any_size==1:
    sample_image=np.zeros((*IMAGE_SIZE, 3), dtype='float32')
    sample_image_nopad = np.array(Image.open('../dataset/VOCdevkit/VOC2012/JPEGImages/2007_000129.jpg').convert('RGB'), dtype=float)
    sample_image = tf.keras.applications.resnet50.preprocess_input(sample_image)
    nopad_shape=sample_image_nopad.shape
    sample_image[:nopad_shape[0], :nopad_shape[1]]=sample_image_nopad
    sample_mask = np.array(Image.open('../dataset/VOCdevkit/VOC2012/SegmentationClass/2007_000129.png').convert('RGB'), dtype=float)
elif any_size==2:
    sample_image=np.zeros((*IMAGE_SIZE, 3), dtype='float32')
    sample_mask=np.zeros((*IMAGE_SIZE, 3), dtype='float32')
    sample_image_nopad = np.array(Image.open('../dataset/VOCdevkit/VOC2012/JPEGImages/2008_000120.jpg').convert('RGB'), dtype=float)
    sample_image = tf.keras.applications.resnet50.preprocess_input(sample_image)
    nopad_shape=sample_image_nopad.shape
    sample_image[:nopad_shape[0], :nopad_shape[1]]=sample_image_nopad
    sample_mask_nopad = np.array(Image.open('../dataset/VOCdevkit/VOC2012/SegmentationClass/2008_000120.png').convert('RGB'), dtype=float)
    sample_mask[:nopad_shape[0], :nopad_shape[1]]=sample_mask_nopad


else:
    sample_image = np.array(Image.open('../dataset/VOCdevkit/VOC2012/JPEGImages/2007_000129.jpg').convert('RGB').resize(IMAGE_SIZE), dtype=float)
    sample_image=tf.keras.applications.vgg16.preprocess_input(sample_image)
    sample_mask = np.array(Image.open('../dataset/VOCdevkit/VOC2012/SegmentationClass/2007_000129.png').convert('RGB').resize(IMAGE_SIZE, Image.NEAREST))
#sample_mask=sample_mask[..., np.newaxis]

#Model define
OUTPUT_CHANNELS=n_classes+1
model = segmentation_network(base_model_name,model_name, OUTPUT_CHANNELS, IMAGE_SIZE)
checkpint_folder = model_name+'-'+base_model_name+'-'+str(IMAGE_SIZE[0])
checkpoint_path='checkpoints/'+checkpint_folder+'/cp-any448-0015.ckpt'
model.load_weights(checkpoint_path)


def create_binary_mask(pred_mask):
    thre_ind=pred_mask<.5
    pred_mask[thre_ind]=0
    return pred_mask[0]

def create_mask(pred_mask):
    pred_mask1=tf.argmax(pred_mask, axis=-1)
    pred_mask=pred_mask1[0]
    pred_mask=labels2colors(pred_mask, c_map)
    return pred_mask

def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pre_mask=model.predict(image)
            display([image[0], mask[0], create_binary_mask(pred_mask)])
    else:

        pred_mask=model.predict(sample_image[tf.newaxis, ...])
        #pred_mask[:,:,:,21]=0
        pred_img=create_mask(pred_mask)
        if any_size==1:
            pred_img=pred_img[:nopad_shape[0], :nopad_shape[1]]
            sample_imageo=sample_image[:nopad_shape[0], :nopad_shape[1]]
        else:
            sample_imageo=sample_image
        alpha=.7
        sample_maskv=colors2labels(sample_mask, c_map, one_hot=True)
        pred_mask_v=colors2labels(pred_img, c_map, one_hot=True)
        sample_maskv[:,:,-1] = 0
        sample_maskv=tf.cast(sample_maskv, dtype='int64')

        pred_mask_v=tf.cast(pred_mask_v, dtype='int64')
        miou_val=general_iou(sample_maskv[tf.newaxis, ...], pred_mask_v[tf.newaxis, ...])
        acc=tf.keras.metrics.Accuracy()
        acc.update_state(sample_mask[tf.newaxis, ...], pred_img[tf.newaxis, ...])
        print(acc.result().numpy(), miou_val)

        #print(miou_val)
        display([sample_imageo, sample_mask, pred_img])


show_predictions()

