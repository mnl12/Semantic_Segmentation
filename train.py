from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow_examples.models.pix2pix import pix2pix
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
from IPython.display import clear_output
import os
from PIL import Image
from my_metrics import my_miou_metric, others_iou, me_others_iou, general_iou, general_iou_ignore
from pascal_voc_cmap import color_map, colors2labels, onehot2mask, labels2colors, onehot2mask
from generators import pascal_voc_generator
from models import segmentation_network
from utils import display


seed=1
extra_train=0
RAND_TRAIN_VAL=0
shuffle=1
augment=0
BATCH_SIZE = 12
IMAGE_SIZE=(512,512)
ANY_SIZE=1
model_name='deeplab_v3'
base_model_name='resnet'
MASK_SIZE=IMAGE_SIZE
n_classes=21
EPOCHS = 100
c_map=color_map(n_classes)
if extra_train == 0:
    db_path='../dataset/VOCdevkit/VOC2012'
    img_path='/JPEGImages/'
    mask_path='/SegmentationClass/'
else:
    db_path='../dataset'
    img_path='/benchmark_RELEASE/dataset/img/'
    mask_path='/benchmark_RELEASE/dataset/cls/'

img_dir=db_path+img_path
mask_dir=db_path+mask_path
img_dir_test='../dataset/VOCdevkit/VOC2012/JPEGImages/'
mask_dir_test='../dataset/VOCdevkit/VOC2012/SegmentationClass/'
sdf=os.walk(img_path)
db_length=len(os.walk(mask_dir).__next__()[2])
mask_ids=next(os.walk(mask_dir))[2]

if RAND_TRAIN_VAL==1:
    train_ids=mask_ids[:int(len(mask_ids)*.9)]
    test_ids=mask_ids[int(len(mask_ids)*.9):]
else:
    file_path='../dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/'
    file_path_extra='../dataset/benchmark_RELEASE/dataset/'
    if extra_train == 0:
        train_ids=open(file_path+'train.txt', 'r').readlines()
    elif extra_train == 2:
        train_ids = open(file_path_extra + 'train_pascal.txt', 'r').readlines()
    else:
        train_ids1 = set(open(file_path_extra + 'train.txt', 'r').readlines())
        train_ids2 = set(open(file_path_extra + 'val.txt', 'r').readlines())
        tst_ids = set(open(file_path+'val.txt', 'r').readlines())
        train_ids=train_ids1|train_ids2-tst_ids
    test_ids=open(file_path+'val.txt', 'r').readlines()
    train_ids=[train_id.replace('\n', '.png') for train_id in train_ids]
    test_ids = [test_id.replace('\n', '.png') for test_id in test_ids]





TRAIN_LENGTH=len(train_ids)
TEST_LENGTH=len(test_ids)


BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train_generator=pascal_voc_generator(img_dir, mask_dir, train_ids, BATCH_SIZE, IMAGE_SIZE, MASK_SIZE, 1, n_classes, ANY_SIZE, shuffle, augment, extra_train)
test_generator=pascal_voc_generator(img_dir_test, mask_dir_test, test_ids, BATCH_SIZE, IMAGE_SIZE, MASK_SIZE, 1, n_classes, ANY_SIZE, shuffle, 0, 0)
sample_image_batch, sample_mask_batch,weightj=test_generator.__getitem__(1)
display([sample_image_batch[1,...], labels2colors(onehot2mask(sample_mask_batch[1]), c_map)])


#Model define
OUTPUT_CHANNELS=n_classes+1
model = segmentation_network(base_model_name,model_name, OUTPUT_CHANNELS, IMAGE_SIZE)
#model=DeepLabV3Plus(IMAGE_SIZE[0],IMAGE_SIZE[1],OUTPUT_CHANNELS)
model.summary()
w_loss = tf.keras.losses.CategoricalCrossentropy()
def my_loss(y_true, y_pred):
    weights=tf.cast(tf.not_equal(y_true[:,:,:,n_classes], 1), tf.float32)
    weights=weights[...,tf.newaxis]
    return tf.keras.backend.mean(tf.keras.backend.categorical_crossentropy(tf.multiply(y_true, weights), tf.multiply(y_pred, weights)), axis=-1)
initial_learning_rate = 4*1e-5
k_iou_metric=tf.keras.metrics.MeanIoU(num_classes=n_classes)
opt=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate, decay=1e-6)
opt2=tf.keras.optimizers.SGD(learning_rate=initial_learning_rate, momentum=.9, decay=0)
lr_poly=tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate, 35000, end_learning_rate=0, power=0.9)
opt_poly=tf.keras.optimizers.SGD(learning_rate=lr_poly)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', general_iou_ignore])

checkpint_folder = model_name+'-'+base_model_name+'-'+str(IMAGE_SIZE[0])
checkpoint_path='checkpoints/'+checkpint_folder+'/cp-any448-{epoch:04d}.ckpt'
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    period=5)


model_history=model.fit(train_generator, epochs=EPOCHS, validation_data=test_generator, callbacks=[cp_callback])

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()



