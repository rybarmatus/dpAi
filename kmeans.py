import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from sklearn.cluster import KMeans

import config
import matplotlib.pyplot as plt


def read_features() -> []:
    import csv
    with open(config.featuresFile, newline='') as f:
        reader = csv.reader(f)
        return list(reader)


def get_or_extract_features(img_list, model) -> []:
    if do_csv_exists():
        return read_features()
    return extract_features(img_list, model)


def do_csv_exists() -> bool:
    from pathlib import Path
    my_file = Path(config.featuresFile)
    return my_file.is_file()


def extract_features(img_list: [], model) -> []:
    features = []
    index = 0

    for img in img_list:
        im = cv2.imread(img)
        im = cv2.resize(im, (config.img_w, config.img_h))
        img = tf.keras.applications.mobilenet.preprocess_input(np.expand_dims(im.copy(), axis=0))
        resnet_feature = model.predict(img)
        resnet_feature_np = np.array(resnet_feature)
        features.append(resnet_feature_np.flatten())

        print(features[index])
        index += 1

    df = pd.DataFrame(features)
    df.to_csv(config.featuresFile, index=False, header=False)
    return features


def do_kmeans():
    train_data: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
        config.k_means_datapath,
        seed=123,
        image_size=(config.img_w, config.img_h),
        batch_size=config.batch_size,
        color_mode='rgb',
        label_mode='categorical',
    )

    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=(config.img_w, config.img_w, 3),
        include_top=False,
    )  # Do not include the ImageNet classifier at the top.

    model = tf.keras.Sequential([base_model])
    features = get_or_extract_features(train_data.file_paths[0:100], model)
    sum_of_squared_distances = []
    K = range(1, 14)
    for k in K:
        model = KMeans(n_clusters=k, verbose=1).fit(features)
        sum_of_squared_distances.append(model.inertia_)
        # plt.scatter(features[0][0], features[1], c=k, cmap='rainbow')
        # plt.show()

    plt.plot(K, sum_of_squared_distances, "bx")
    plt.xlabel("Kvalues")
    plt.ylabel("Sum of Squared Distances")
    plt.title("Elbow Method")
    plt.show()

    pass
