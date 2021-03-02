import tensorflow as tf
import numpy as np
from tensorflow_examples.models.pix2pix import pix2pix
def Upsample(tensor, size):
    '''bilinear upsampling'''
    name = tensor.name.split('/')[0] + '_upsample'

    def bilinear_upsample(x, size):
        resized = tf.image.resize(
            images=x, size=size)
        return resized
    y = tf.keras.layers.Lambda(lambda x: bilinear_upsample(x, size),
               output_shape=size, name=name)(tensor)
    return y
def ASPP(tensor):
    '''atrous spatial pyramid pooling'''
    dims = tf.keras.backend.int_shape(tensor)

    y_pool = tf.keras.layers.AveragePooling2D(pool_size=(
        dims[1], dims[2]), name='average_pooling')(tensor)
    y_pool = tf.keras.layers.Conv2D(filters=256, kernel_size=1, padding='same',
                    kernel_initializer='he_normal', name='pool_1x1conv2d', use_bias=False)(y_pool)
    y_pool = tf.keras.layers.BatchNormalization(name=f'bn_1')(y_pool)
    y_pool = tf.keras.layers.Activation('relu', name=f'relu_1')(y_pool)

   # y_pool = Upsample(tensor=y_pool, size=[dims[1], dims[2]])
    y_pool = tf.keras.layers.UpSampling2D((dims[1], dims[2]), interpolation='bilinear')(y_pool)

    y_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
                 kernel_initializer='he_normal', name='ASPP_conv2d_d1', use_bias=False)(tensor)
    y_1 = tf.keras.layers.BatchNormalization(name=f'bn_2')(y_1)
    y_1 = tf.keras.layers.Activation('relu', name=f'relu_2')(y_1)

    y_6 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, dilation_rate=6, padding='same',
                 kernel_initializer='he_normal', name='ASPP_conv2d_d6', use_bias=False)(tensor)
    y_6 = tf.keras.layers.BatchNormalization(name=f'bn_3')(y_6)
    y_6 = tf.keras.layers.Activation('relu', name=f'relu_3')(y_6)

    y_12 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, dilation_rate=12, padding='same',
                  kernel_initializer='he_normal', name='ASPP_conv2d_d12', use_bias=False)(tensor)
    y_12 = tf.keras.layers.BatchNormalization(name=f'bn_4')(y_12)
    y_12 = tf.keras.layers.Activation('relu', name=f'relu_4')(y_12)

    y_18 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, dilation_rate=18, padding='same',
                  kernel_initializer='he_normal', name='ASPP_conv2d_d18', use_bias=False)(tensor)
    y_18 = tf.keras.layers.BatchNormalization(name=f'bn_5')(y_18)
    y_18 = tf.keras.layers.Activation('relu', name=f'relu_5')(y_18)

    y = tf.keras.layers.concatenate([y_pool, y_1, y_6, y_12, y_18], name='ASPP_concat')

    y = tf.keras.layers.Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
               kernel_initializer='he_normal', name='ASPP_conv2d_final', use_bias=False)(y)
    y = tf.keras.layers.BatchNormalization(name=f'bn_final')(y)
    y = tf.keras.layers.Activation('relu', name=f'relu_final')(y)
    return y

def vgg16_with_dropout(img_input, dropout_rate):

    WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                          'releases/download/v0.1/'
                          'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

    # Block 1
    x = tf.keras.layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    #x=tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = tf.keras.layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    #x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = tf.keras.layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    #x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    #x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = tf.keras.layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    #x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    #x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = tf.keras.layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    #x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    #x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    model = tf.keras.models.Model(inputs=img_input, outputs=x, name='vgg16')

    weights_path = tf.keras.utils.get_file(
        'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
        WEIGHTS_PATH_NO_TOP,
        cache_subdir='models',
        file_hash='6d6bbae143d832006294945121d1f1fc')
    model.load_weights(weights_path)
    return model



def segmentation_network (base_model_name,decoder_name, n_classes, IMAGE_SIZE):
    OUTPUT_CHANNELS = n_classes
    if base_model_name=='mobilenet':
        base_model = tf.keras.applications.MobileNetV2(input_shape=[*IMAGE_SIZE, 3], include_top=False)
        layer_names = [
            'block_1_expand_relu',  # 64x64
            'block_3_expand_relu',  # 32x32
            'block_6_expand_relu',  # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',  # 4x4
        ]
    elif base_model_name=='vgg':
        base_model = tf.keras.applications.VGG16(input_shape=[*IMAGE_SIZE, 3], include_top=False, weights='imagenet')
        layer_names = [
            'block1_pool',  # 64
            'block2_pool',
            'block3_pool',
            'block4_pool',
            'block5_pool'
        ]
    elif base_model_name == 'resnet':
        base_model = tf.keras.applications.ResNet50(input_shape=[*IMAGE_SIZE, 3], include_top=False, weights='imagenet')
        layer_names = ['conv2_block3_out', 'conv4_block6_out']



    layers = [base_model.get_layer(name).output for name in layer_names]
    #for mlayer in base_model.layers:
    #    mlayer.trainable = False

    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    down_stack.trainable = True

    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),  # 32x32 -> 64x64
    ]
    inputs = tf.keras.Input(shape=[*IMAGE_SIZE, 3])
    enc_outs = down_stack(inputs)



    if decoder_name =='fcn-32':
        y1=enc_outs[-1]
        y1=tf.keras.layers.Conv2D(OUTPUT_CHANNELS, (1,1), padding='same', activation=None, kernel_initializer='zeros')(y1)
        yl=tf.keras.layers.UpSampling2D(size=(32, 32), interpolation='bilinear')(y1)
        y=tf.keras.activations.softmax(yl)
    elif decoder_name =='fcn-16':
        x16=tf.keras.layers.Conv2D(OUTPUT_CHANNELS, (1,1), padding='same', kernel_initializer='zeros')(enc_outs[-2])
        x216=tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 3, strides=2, padding='same', kernel_initializer='zeros', use_bias=False)(enc_outs[-1])
        sumx=tf.keras.layers.Add()([x16, x216])
        sumx = tf.keras.layers.UpSampling2D((16, 16), interpolation='bilinear')(sumx)
        y = tf.keras.layers.Softmax()(sumx)
    elif decoder_name=='fcn-8':
        x316 = tf.keras.layers.Conv2D(OUTPUT_CHANNELS, (1, 1), padding='same')(enc_outs[-3])
        x216=tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 3, strides=2, padding='same')(enc_outs[-2])
        x16 = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 3, strides=4, padding='same')(enc_outs[-1])
        sumx = tf.keras.layers.Add()([x16, x216, x316])
        sumx = tf.keras.layers.UpSampling2D((8, 8), interpolation='bilinear')(sumx)
        y = tf.keras.layers.Softmax()(sumx)
    elif decoder_name=='deeplab':
        y=ASPP(enc_outs[-1])
        y=tf.keras.layers.Conv2D(OUTPUT_CHANNELS, 1, name='final_logits')(y)
        y=tf.keras.layers.UpSampling2D((16, 16), interpolation='bilinear', name='upsampled')(y)
        y=tf.keras.layers.Softmax()(y)
    elif decoder_name=='deeplab_v3':
        y = ASPP(enc_outs[-1])
        x_a = tf.keras.layers.UpSampling2D((4, 4), interpolation='bilinear', name='upsampled')(y)

        x_b = enc_outs[-2]
        x_b = tf.keras.layers.Conv2D(filters=48, kernel_size=1, padding='same',
                     kernel_initializer='he_normal', name='low_level_projection', use_bias=False)(x_b)
        x_b = tf.keras.layers.BatchNormalization(name=f'bn_low_level_projection')(x_b)
        x_b = tf.keras.layers.Activation('relu', name='low_level_activation')(x_b)

        x = tf.keras.layers.concatenate([x_a, x_b], name='decoder_concat')

        x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
                   kernel_initializer='he_normal', name='decoder_conv2d_1', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(name=f'bn_decoder_1')(x)
        x = tf.keras.layers.Activation('relu', name='activation_decoder_1')(x)

        x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
                   kernel_initializer='he_normal', name='decoder_conv2d_2', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(name=f'bn_decoder_2')(x)
        x = tf.keras.layers.Activation('relu', name='activation_decoder_2')(x)

        x=tf.keras.layers.Conv2D(OUTPUT_CHANNELS, 1, name='final_logits')(x)
        x=tf.keras.layers.UpSampling2D((4, 4), interpolation='bilinear', name='upsampled_2')(x)
        y=tf.keras.layers.Softmax()(x)



    elif decoder_name =='unet':
        last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 3, strides=2, padding='same', activation='softmax')
        z = enc_outs[-1]
        skips = reversed(enc_outs[:-1])

        for up, skip in zip(up_stack, skips):
            z = up(z)
            cancat = tf.keras.layers.Concatenate()
            z = cancat([z, skip])

        y = last(z)

    return tf.keras.Model(inputs=inputs, outputs=y)



