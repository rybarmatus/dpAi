import PIL
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL.Image import Image
from keras import regularizers
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import config
import tensorflow_hub as hub
from tensorflow import keras
import tensorflow_text

def resize_images():
    df = pd.read_csv('text_with_img.csv')
    for page in df['img']:
        baseheight = 300
        print(page)
        filename = page.split("\\")[5]
        dir = page.replace(filename, '')
        dir = dir.replace('dp2 - Copy', 'dp_reduced')
        from pathlib import Path
        Path(dir).mkdir(parents=True, exist_ok=True)
        img = PIL.Image.open(page)
        img = img.resize((baseheight, baseheight), PIL.Image.ANTIALIAS)
        page = page.replace('dp2 - Copy', 'dp_reduced')
        img.save(page)


def find_img_for_text():
    data_path = 'D:\dp2 - Copy\web_categories - Copy'
    train_data = tf.keras.utils.image_dataset_from_directory(
        data_path,
        seed=123,
        image_size=(config.img_w, config.img_w),
        color_mode='rgb',
        label_mode='categorical',
    )
    df = pd.read_csv('web_texts.csv')
    df['page'] = df['page'].apply(lambda v: v.replace('.html', ''))
    df['img'] = df['page'].apply(lambda v: tt(v, train_data.file_paths))
    df = df[df['img'] != '']
    df.to_csv('text_with_img.csv', index=False)
    pass

def visualize(idx):
    df = pd.read_csv('text_with_img.csv')
    current_row = df.iloc[idx]
    image_1 = plt.imread(current_row["img"])
    text_1 = current_row["text"]
    label = current_row["page"]

    plt.subplot(1, 2, 1)
    plt.imshow(image_1)
    plt.axis("off")
    plt.title("Image One")
    plt.subplot(1, 2, 2)
    plt.imshow(image_1)
    plt.axis("off")
    plt.title("Image Two")
    plt.show()

    print(f"Text one: {text_1}")
    print(f"Label: {label}")


def tt(v, files):
    try:
        return [i for i in files if v in i][0]
    except:
        return ''


# casti kodu cerpane z https://github.com/artelab/Image-and-Text-fusion-for-UPMC-Food-101-using-BERT-and-CNNs/blob/main/stacking_early_fusion_UPMC_food101.ipynb
def t():
    find_img_for_text()


# from tensorflow import keras
#
# img_model = keras.Sequential()
#
# img_model.add(keras.applications.MobileNetV2(
#     weights="imagenet",  # Load weights pre-trained on ImageNet.
#     input_shape=(config.img_w, config.img_w, 3),
#     include_top=False,
# ))  # Do not include the ImageNet classifier at the top.
#
# # Freeze the base_model
# img_model.trainable = False
# img_model.add(tf.keras.applications.mobilenet.preprocess_input(img_model))
# img_model.add(img_model, training=False)
# img_model.add(keras.layers.GlobalAveragePooling2D())
# img_model.add(tf.keras.layers.Dense(units=68, activation='relu',
#                           kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),
#                           ))
# img_model.add(keras.layers.Dropout(0.4))  # Regularize with dropout
#
# bert_preprocess = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2")
# bert_encoder = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base/1")
#
# text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
# preprocessed_text = bert_preprocess(text_input)
# outputs = bert_encoder(preprocessed_text)
# net = (outputs['pooled_output'])
#
# net = tf.keras.layers.Dropout(0.1, name="dropout")(net)
# net = tf.keras.layers.Dense(14, activation='softmax', name="output")(net)
#
# text_model = tf.keras.Model(inputs=[text_input], outputs=[net])
#
# image_input = layers.Input(shape = input_shape, dtype=tf.float32,
#                            name = "image")
#
# image_side = model_cnn(image_input)
# text_side = model_lstm([input_word_ids, input_mask, segment_ids])
# # Concatenate features from images and texts
# merged = layers.Concatenate()([image_side, text_side])
# merged = layers.Dense(256, activation = 'relu')(merged)
# output = layers.Dense(nClasses, activation='softmax', name = "class")(merged)
# model = models.Model([input_word_ids, input_mask, segment_ids, image_input], output)

def make_bert_preprocessing_model(sentence_features, seq_length=128):
    bert_preprocess_path = "https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2"

    """Returns Model mapping string features to BERT inputs.

  Args:
    sentence_features: A list with the names of string-valued features.
    seq_length: An integer that defines the sequence length of BERT inputs.

  Returns:
    A Keras Model that can be called on a list or dict of string Tensors
    (with the order or names, resp., given by sentence_features) and
    returns a dict of tensors for input to BERT.
  """

    input_segments = [
        tf.keras.layers.Input(shape=(), dtype=tf.string, name=ft)
        for ft in sentence_features
    ]

    # Tokenize the text to word pieces.
    bert_preprocess = hub.load(bert_preprocess_path)
    tokenizer = hub.KerasLayer(bert_preprocess.tokenize, name="tokenizer")
    segments = [tokenizer(s) for s in input_segments]

    # Optional: Trim segments in a smart way to fit seq_length.
    # Simple cases (like this example) can skip this step and let
    # the next step apply a default truncation to approximately equal lengths.
    truncated_segments = segments

    # Pack inputs. The details (start/end token ids, dict of output tensors)
    # are model-dependent, so this gets loaded from the SavedModel.
    packer = hub.KerasLayer(
        bert_preprocess.bert_pack_inputs,
        arguments=dict(seq_length=seq_length),
        name="packer",
    )
    model_inputs = packer(truncated_segments)
    return keras.Model(input_segments, model_inputs)


def dataframe_to_dataset(dataframe):
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(dataframe.category)
    dataframe["category"] = le.transform(dataframe.category)
    columns = ["img", "text", "category"]
    dataframe = dataframe[columns].copy()
    labels = dataframe.pop("category")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

resize = (128, 128)
bert_input_features = ["input_word_ids", "input_type_ids", "input_mask"]


def preprocess_image(image_path):

    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 3)
    image = tf.image.resize(image, resize)
    return image


def preprocess_text(text_1):
    text_1 = tf.convert_to_tensor([text_1])
    output = bert_preprocess_model([text_1])
    output = {feature: tf.squeeze(output[feature]) for feature in bert_input_features}
    return output


def preprocess_text_and_image(sample):
    image_1 = preprocess_image(sample["img"])
    text = preprocess_text(sample["text"])
    return {"image_1": image_1, "text": text}

batch_size = 32
auto = tf.data.AUTOTUNE


def prepare_dataset(dataframe, training=True):
    ds = dataframe_to_dataset(dataframe)
    if training:
        ds = ds.shuffle(len(train_df))
    ds = ds.map(lambda x, y: (preprocess_text_and_image(x), y)).cache()
    ds = ds.batch(batch_size).prefetch(auto)
    return ds


def project_embeddings(
    embeddings, num_projection_layers, projection_dims, dropout_rate
):
    projected_embeddings = tf.keras.layers.Dense(units=projection_dims)(embeddings)
    for _ in range(num_projection_layers):
        x = tf.nn.gelu(projected_embeddings)
        x = tf.keras.layers.Dense(projection_dims)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.Add()([projected_embeddings, x])
        projected_embeddings = tf.keras.layers.LayerNormalization()(x)
    return projected_embeddings

def create_vision_encoder(
    num_projection_layers, projection_dims, dropout_rate, trainable=False
):
    # Load the pre-trained ResNet50V2 model to be used as the base encoder.
    resnet_v2 = tf.keras.applications.MobileNet(
        include_top=False, weights="imagenet", pooling="avg"
    )
    # Set the trainability of the base encoder.
    for layer in resnet_v2.layers:
        layer.trainable = trainable

    # Receive the images as inputs.
    image_1 = tf.keras.Input(shape=(128, 128, 3), name="img")

    # Preprocess the input image.
    preprocessed_1 = tf.keras.applications.mobilenet.preprocess_input(image_1)

    # Generate the embeddings for the images using the resnet_v2 model
    # concatenate them.
    embeddings = resnet_v2(preprocessed_1)

    # Project the embeddings produced by the model.
    outputs = project_embeddings(
        embeddings, num_projection_layers, projection_dims, dropout_rate
    )
    # Create the vision encoder model.
    return tf.keras.Model([image_1], outputs, name="vision_encoder")

def create_text_encoder(
    num_projection_layers, projection_dims, dropout_rate, trainable=False
):
    # Load the pre-trained BERT model to be used as the base encoder.
    bert = hub.KerasLayer(bert_model_path, name="bert",)
    # Set the trainability of the base encoder.
    bert.trainable = trainable

    # Receive the text as inputs.
    bert_input_features = ["input_type_ids", "input_mask", "input_word_ids"]
    inputs = {
        feature: tf.keras.Input(shape=(128,), dtype=tf.int32, name=feature)
        for feature in bert_input_features
    }

    # Generate embeddings for the preprocessed text using the BERT model.
    embeddings = bert(inputs)["pooled_output"]

    # Project the embeddings produced by the model.
    outputs = project_embeddings(
        embeddings, num_projection_layers, projection_dims, dropout_rate
    )
    # Create the text encoder model.
    return tf.keras.Model(inputs, outputs, name="text_encoder")

def create_multimodal_model(
    num_projection_layers=1,
    projection_dims=128,
    dropout_rate=0.4,
    vision_trainable=False,
    text_trainable=False,
):
    # Receive the images as inputs.
    image_1 = tf.keras.Input(shape=(128, 128, 3), name="image_1")

    # Receive the text as inputs.
    bert_input_features = ["input_type_ids", "input_mask", "input_word_ids"]
    text_inputs = {
        feature: tf.keras.Input(shape=(128,), dtype=tf.int32, name=feature)
        for feature in bert_input_features
    }

    # Create the encoders.
    vision_encoder = create_vision_encoder(
        num_projection_layers, projection_dims, dropout_rate, vision_trainable
    )
    text_encoder = create_text_encoder(
        num_projection_layers, projection_dims, dropout_rate, text_trainable
    )

    # Fetch the embedding projections.
    vision_projections = vision_encoder([image_1])
    text_projections = text_encoder(text_inputs)

    # Concatenate the projections and pass through the classification layer.
    concatenated = tf.keras.layers.Concatenate()([vision_projections, text_projections])
    concatenated = tf.keras.layers.Dense(units=68, activation='relu',
                              kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),
                              )(concatenated)
    concatenated = keras.layers.Dropout(0.4)(concatenated)
    outputs = tf.keras.layers.Dense(14, activation="softmax")(concatenated)
    return tf.keras.Model([image_1, text_inputs], outputs)


if __name__ == '__main__':
    t()
    resize_images()
    exit(0)
    df = pd.read_csv('text_with_img.csv')
    print(df["category"].value_counts())

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["category"].values, random_state=42
    )
    # 5% for validation
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, stratify=train_df["category"].values, random_state=42
    )

    print(f"Total training examples: {len(train_df)}")
    print(f"Total validation examples: {len(val_df)}")
    print(f"Total test examples: {len(test_df)}")

    bert_model_path = (
        "https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base/1"
    )

    bert_preprocess_model = make_bert_preprocessing_model(["text"])
    # plot_model(bert_preprocess_model, show_shapes=True, show_dtype=True)

    train_ds = prepare_dataset(train_df)
    validation_ds = prepare_dataset(val_df, False)
    test_ds = prepare_dataset(test_df, False)

    multimodal_model = create_multimodal_model()

    multimodal_model.summary()

    multimodal_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )

    # multimodal_model.compile(
    #     optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy"
    # )

    early = tf.keras.callbacks.EarlyStopping(patience=3,

                                             restore_best_weights=True,
                                             monitor="val_loss", )

    history = multimodal_model.fit(train_ds, validation_data=validation_ds, epochs=2, callbacks=[early])

    multimodal_model.save("multi_modal.h5")

    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # plot results
    # accuracy
    plt.figure(figsize=(10, 16))
    plt.rcParams['figure.figsize'] = [16, 9]
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.grid'] = True
    plt.rcParams['figure.facecolor'] = 'white'
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.title(
        f'\nTraining and Validation Accuracy. \nTrain Accuracy: {str(acc[-1])}\nValidation Accuracy: {str(val_acc[-1])}')

    # loss
    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title(f'Training and Validation Loss. \nTrain Loss: {str(loss[-1])}\nValidation Loss: {str(val_loss[-1])}')
    plt.xlabel('epoch')
    plt.tight_layout(pad=3.0)
    plt.show()

    accuracy_score = multimodal_model.evaluate(test_ds)
    print(accuracy_score)
    print("Accuracy: {:.4f}%".format(accuracy_score[1] * 100))

    print("Loss: ", accuracy_score[0])

    _, acc = multimodal_model.evaluate(test_ds)
    print(f"Accuracy on the test set: {round(acc * 100, 2)}%.")

    pass
