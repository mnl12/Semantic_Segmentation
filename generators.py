import tensorflow as tf
import numpy as np
from scipy.io import loadmat
import os
import random
from PIL import Image
from pascal_voc_cmap import color_map, colors2labels, onehot2mask, labels2onehot

class pascal_voc_generator (tf.keras.utils.Sequence):
    def __init__(self, image_path, mask_path, ids, batch_size, image_size, mask_size, normalization, n_classes, any_size, shuffle, augment, extra):
        self.indexes=ids
        self.image_path=image_path
        self.mask_path=mask_path
        self.batch_size=batch_size
        self.image_size=image_size
        self.mask_size=mask_size
        self.batch_size=batch_size
        self.normalization=normalization
        self.shuffle=shuffle
        self.n_classes=n_classes
        self.cmap=self.color_map_c()
        self.any_size=any_size
        self.aug=augment
        self.extra=extra

    def __len__(self):
        return int(len(self.indexes)/float(self.batch_size))

    def color_map_c(self):
        cmap256=color_map(256)
        cmap=np.vstack([cmap256[:self.n_classes], cmap256[-1].reshape(1, 3)])
        return cmap

    def on_epoch_end(self):
        'Updates indexes after each epoch'

        if self.shuffle:
            random.shuffle(self.indexes)


    def get_labels(self, masks):
        labels=[]
        for mask in masks:
            if np.sum(mask)>0:
                label=1
            else:
                label=0
            labels.append(label)
        return np.array(labels)

    def construct_image_batch(self, image_group, BATCH_SIZE):
        # get the max image shape
        #max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))
        max_shape=(*self.image_size, image_group[1].shape[2])

        # construct an image batch object
        image_batch = np.zeros((BATCH_SIZE,) + max_shape, dtype='float32')

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            min_shape = tuple(min(image.shape[d], self.image_size[d]) for d in range(2))
            image_batch[image_index, :min_shape[0], :min_shape[1]] = image[:min_shape[0], :min_shape[1]]

        return image_batch

    def construct_mask_batch(self, image_group, BATCH_SIZE):
        # get the max image shape
        max_shape=(*self.image_size, image_group[1].shape[2])

        # construct an image batch object
        image_batch = np.zeros((BATCH_SIZE,) + max_shape, dtype='float32')
        #image_batch[:, :, :, self.n_classes] = 1


        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            min_shape = tuple(min(image.shape[d], self.image_size[d]) for d in range(2))
            image_batch[image_index, :min_shape[0], :min_shape[1]] = image[:min_shape[0], :min_shape[1]]
            image_batch[:, :, :, self.n_classes] = 0
            image_weights = np.sum(image_batch, axis=-1)


        return image_batch


    def __getitem__(self, index):
        if self.any_size==1:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
            if self.aug==1:
                transform_ops = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
                rand_op = random.choice(transform_ops)
                rot_deg=np.random.randint(-20,20)
                images_orig=self.construct_image_batch([np.asarray(Image.open(self.image_path+k.replace('png','jpg')).convert('RGB'), dtype=float) for k in indexes], self.batch_size)
                images_aug=self.construct_image_batch([np.asarray(Image.open(self.image_path+k.replace('png','jpg')).convert('RGB').rotate(rot_deg), dtype=float) for k in indexes], self.batch_size)
                masks_onehot_orig = self.construct_image_batch([colors2labels(np.array(Image.open(self.mask_path+k).convert('RGB')), self.cmap, True) for k in indexes], self.batch_size)
                masks_onehot_aug = self.construct_mask_batch([colors2labels(np.array(Image.open(self.mask_path+k).convert('RGB').rotate(rot_deg)), self.cmap, True) for k in indexes], self.batch_size)
                images = images_aug
                masks_onehot = masks_onehot_aug
                weights_masks = np.sum(masks_onehot, axis=-1, dtype=int)

            else:
                images=self.construct_image_batch([np.asarray(Image.open(self.image_path+k.replace('png', 'jpg')).convert('RGB'), dtype=float) for k in indexes], self.batch_size)
                if self.extra == 0:
                    masks_onehot = self.construct_mask_batch([colors2labels(np.array(Image.open(self.mask_path+k).convert('RGB')), self.cmap, True) for k in indexes], self.batch_size)
                    #masks=[colors2labels(np.array(Image.open(self.mask_path+k).convert('RGB').resize(self.image_size, Image.NEAREST)), self.cmap, False) for k in indexes]
                else:
                    masks_onehot = self.construct_mask_batch([labels2onehot(np.array(loadmat(self.mask_path+k.replace('png', 'mat'))['GTcls'][0,0][1]), self.n_classes+1) for k in indexes], self.batch_size)
                weights_masks = np.sum(masks_onehot, axis=-1, dtype=int)
        else:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
            images=[np.asarray(Image.open(self.image_path+k.replace('png','jpg')).convert('RGB').resize(self.image_size), dtype=float) for k in indexes]
            masks_onehot = [colors2labels(np.array(Image.open(self.mask_path+k).convert('RGB').resize(self.image_size, Image.NEAREST)), self.cmap, True) for k in indexes]
            weights_masks=np.sum(masks_onehot, axis=-1, dtype=int)



        if self.normalization:
            #images=np.array(images)/255.0
            images=tf.keras.applications.resnet50.preprocess_input(images)
            #masks=np.array(masks, dtype=int)
            masks_onehot = np.array(masks_onehot, dtype=int)


        labels=self.get_labels(masks_onehot)
        return images, masks_onehot,weights_masks
