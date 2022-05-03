from tensorflow.python.framework.config import set_memory_growth

import kmeans
import os
import config
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

if __name__ == '__main__':
    data_path = 'D:\\dp2\\web_categories - Copy'
    # transfer_lr_InceptionResNetV2.tranfser_InceptionResNetV2(32, 0.001, 0.001, 0.4)
    # transfer_lr_binary_InceptionResNetV2.tranfser_InceptionResNetV2(64, 0.001, 0.001, 0.4)
    # transfer_lr_MobileNetV2.mobileNetV2(128, 0.001, 0.001, 0.4)
    # FineTuningCategorical.fineTune(data_path, "all.h5")
    # FineTuningBinaryMobileNet.fineTune(data_path, "finedMobileEshops.h5")
    # FineTuningCategoricalMobileNet.fineTune(data_path, "fineCategoricalMobilenet.h5")
    # kmeans.do_kmeans_after_dim_reduction()
    # kmeans.do_kmeans()
    kmeans.kmeans_elbow()
    pass
    listOfFiles = []
    notPresent = []
    for (dirpath, dirnames, filenames) in os.walk(config.images_path):
        listOfFiles += [os.path.join(file) for file in filenames]

    test = []

    for dirpath, dirnames, filenames in os.walk(config.html_folder):
        for f in filenames:
            tst = f.removesuffix('.html')
            tst += '.png'
            test += [f]
            if str(tst) not in listOfFiles:
                notPresent += [tst]
        # if filenames not in listOfFiles:
        #     print(filenames)
        #     notPresent += filenames


    pass

# cerpane z
# https://www.tensorflow.org/tutorials/images/transfer_learning