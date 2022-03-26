import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import config


def read_features() -> []:
    import csv
    with open(config.featuresFile, newline='') as f:
        reader = csv.reader(f)
        return list(reader)


def read_features_as_list(file_name: str, with_header=False):
    start = time.time()
    if with_header:
        df = pd.read_csv(file_name)
    else:
        df = pd.read_csv(file_name, header=None)
    end = time.time()
    print('cas na nacitanie csv ', end - start)
    return df.values


def get_or_extract_features(img_list, model):
    if do_csv_exists():
        return read_features_as_list(config.featuresFile), None
    return extract_features(img_list, model)


def do_csv_exists() -> bool:
    from pathlib import Path
    my_file = Path(config.featuresFile)
    return my_file.is_file()


def extract_features(img_list: [], model):
    features = []
    index = 0
    labels_list = []
    img_list.sort()
    # iterovanie ciest k obrazkom stranok
    for img in img_list:
        category = img.split("\\")[3]
        im = cv2.imread(img)
        im = cv2.resize(im, (config.img_w, config.img_h))
        img = tf.keras.applications.mobilenet.preprocess_input(np.expand_dims(im.copy(), axis=0))
        resnet_feature = model.predict(img)
        resnet_feature_np = np.array(resnet_feature)
        feature = resnet_feature_np.flatten()
        features.append(feature)
        labels_list.append(category)
        print(index)
        index += 1

    df = pd.DataFrame(features)
    df.to_csv(config.featuresFile, index=False, header=False)
    return features, labels_list


def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


# https://learnopencv.com/t-sne-for-feature-visualization/
def reduce_dim_2D(features, labels_list):
    tsne = TSNE(n_components=2, verbose=1).fit_transform(features)
    cps_df = pd.DataFrame(columns=['x', 'y', 'label'],
                          data=np.column_stack((tsne,
                                                labels_list)))
    cps_df.to_csv('tsne2.csv', header=True, index=False)


def reduce_dim_3D(features, labels_list):
    tsne = TSNE(n_components=3, verbose=1).fit_transform(features)
    cps_df = pd.DataFrame(columns=['x', 'y', 'z', 'label'],
                          data=np.column_stack((tsne,
                                                labels_list)))
    cps_df.to_csv('tsne3.csv', header=True, index=False)


def plot_data_2D():
    pd.set_option("plotting.backend", "plotly")
    pd.options.plotting.backend = 'plotly'
    print(pd.get_option("plotting.backend"))
    cps_df = pd.read_csv('tsne2.csv')
    # cps_df.columns = ["x", "y", "label"]
    cps_df["x"] = cps_df["x"].apply(lambda x: float(x))
    cps_df["y"] = cps_df["y"].apply(lambda x: float(x))
    cps_df.plot.scatter(x='x', y='y', color='label').show()
    pass


def plot_data_3D():
    import plotly.express as px
    pd.set_option("plotting.backend", "plotly")
    pd.options.plotting.backend = 'plotly'
    print(pd.get_option("plotting.backend"))
    cps_df = pd.read_csv('tsne3.csv')
    # cps_df.columns = ["x", "y", "z", "label"]
    cps_df["x"] = cps_df["x"].apply(lambda x: float(x))
    cps_df["y"] = cps_df["y"].apply(lambda x: float(x))
    cps_df["z"] = cps_df["z"].apply(lambda x: float(x))
    # cps_df.plot.scatter(x='x', y='y', z='z', color='label').show()
    fig = px.scatter_3d(cps_df, x='x', y='y', z='z',
                        color='label')
    fig.show()
    pass


def plot_data_2D_kmeans(model):
    cps_df = pd.read_csv('tsne2.csv')
    cps_df.loc[:, 'label'] = model
    cps_df['label'] = cps_df['label'].apply(lambda x: str(x))

    cps_df.plot.scatter(x='x', y='y', color='label').show()


def plot_data_3D_kmeans(model, k, save_path):
    import plotly.express as px
    cps_df = pd.read_csv('tsne3.csv')
    cps_df.loc[:, 'label'] = model
    cps_df.loc[:, 'size'] = model
    cps_df.loc[:, 'symbol'] = model
    cps_df['label'] = cps_df['label'].apply(lambda x: str(x))
    cps_df['size'] = cps_df['size'].apply(lambda x: 8)
    cps_df['symbol'] = cps_df['symbol'].apply(lambda x: 'circle')
    fig = px.scatter_3d(cps_df, x='x', y='y', z='z',
                        color='label', size='size', size_max=10)
    fig.write_html(save_path + str(k) + ".html")


def do_kmeans():
    pd.options.plotting.backend = 'plotly'

    train_data: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
        config.images_path,
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

    features, labels_list = get_or_extract_features(train_data.file_paths, model)

    # reduce_dim_2D(features, labels_list)
    # reduce_dim_3D(features, labels_list)
    # plot_data_2D()
    # plot_data_3D()

    print("priznaky extrahovane")
    K = range(2, 14)
    for k in K:
        model = KMeans(n_clusters=k, verbose=1).fit_predict(features)
        plot_data_2D_kmeans(model)
        plot_data_3D_kmeans(model, k, "C:\\Users\\snako\\Desktop\\3d before tsnet\\file")


def do_kmeans_after_dim_reduction():
    pd.options.plotting.backend = 'plotly'

    reduced_features_df = pd.read_csv('tsne3.csv')
    reduced_features_df = reduced_features_df.drop(columns='label')
    reduced_features = reduced_features_df.values

    K = range(2, 14)
    for k in K:
        model = KMeans(n_clusters=k, verbose=1).fit_predict(reduced_features)
        plot_data_2D_kmeans(model)
        plot_data_3D_kmeans(model, k, "C:\\Users\\snako\\Desktop\\3d after tsnet\\file")


def kmeans_elbow(train_data, model):
    features, labels_list = get_or_extract_features(train_data.file_paths, model)

    sum_of_squared_distances = []
    K = range(3, 14)
    for k in K:
        model = KMeans(n_clusters=k, verbose=1).fit(features)
        sum_of_squared_distances.append(model.inertia_)

    plt.plot(K, sum_of_squared_distances, "bx")
    plt.xlabel("Kvalues")
    plt.ylabel("Sum of Squared Distances")
    plt.title("Elbow Method")
    plt.show()
