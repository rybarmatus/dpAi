import tensorflow as tf
from tensorflow.python.framework.config import set_memory_growth

import multimodal_text_image

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

if __name__ == '__main__':
    data_path = 'D:\\dp2\\web_categories - Copy'
    multimodal_text_image.resize_images_dataset()
    pass
