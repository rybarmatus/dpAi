import transfer_lr_InceptionResNetV2
import transfer_lr_MobileNetV2
import transfer_lr_binary_InceptionResNetV2
import FineTuningBinary
import FineTuningCategorical
import FineTuningBinaryMobileNet
import FineTuningCategoricalMobileNet
import kmeans

if __name__ == '__main__':
    data_path = 'D:\\dp2\\web_categories - Copy'
    # transfer_lr_InceptionResNetV2.tranfser_InceptionResNetV2(32, 0.001, 0.001, 0.4)
    # transfer_lr_binary_InceptionResNetV2.tranfser_InceptionResNetV2(64, 0.001, 0.001, 0.4)
    # transfer_lr_MobileNetV2.mobileNetV2(128, 0.001, 0.001, 0.4)
    # FineTuningCategorical.fineTune(data_path, "all.h5")
    # FineTuningBinaryMobileNet.fineTune(data_path, "finedMobileEshops.h5")
    # FineTuningCategoricalMobileNet.fineTune(data_path, "fineCategoricalMobilenet.h5")
    kmeans.do_kmeans_after_dim_reduction()
    kmeans.do_kmeans()
    pass

# cerpane z
# https://www.tensorflow.org/tutorials/images/transfer_learning