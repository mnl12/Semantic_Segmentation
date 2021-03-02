import tensorflow as tf
import matplotlib.pyplot as plt

def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

def create_mask(pred_mask):
    pred_mask_1=tf.argmax(pred_mask, axis=-1)
    pred_mask_c=np.empty((0,pred_mask_1.shape[1],pred_mask_1.shape[2], pred_mask.shape[-1]))
    for pred_mask_i in pred_mask_1:
        pred_mask_i_c=labels2colors(pred_mask_i, c_map)
        pred_mask_i_c=colors2labels(pred_mask_i_c, c_map, True)
        pred_mask_c=np.append(pred_mask_c, pred_mask_i_c[tf.newaxis, ...], axis=0)
    return pred_mask_c

def validate_results():
    miou_v=np.zeros(TEST_LENGTH//BATCH_SIZE)
    for i in range(TEST_LENGTH//BATCH_SIZE):
        sample_image_batch, sample_mask_batch=test_generator.__getitem__(i)
        pred_mask_batch = model.predict(sample_image_batch)
        pred_mask_bin_batch = create_mask(pred_mask_batch)
        miou_val = me_others_iou(sample_mask_batch, pred_mask_bin_batch)
        miou_v[i]=miou_val

    print('IOU value:', np.mean(miou_v))


