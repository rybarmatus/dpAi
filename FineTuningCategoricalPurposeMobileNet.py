import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
# casti kodu cerpane z https://keras.io/guides/transfer_learning/
from keras import regularizers
from tensorflow.python.framework.config import set_memory_growth
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import config


# tf.compat.v1.disable_v2_behavior()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

import enum


class SubsetEnum(enum.Enum):
    Train = 'training'
    Val = 'validation'


class LabelModeEnum(enum.Enum):
    Categorical = 'categorical'
    Binary = 'binary'
    Int = 'int'


def get_dataset_from_directory(data_path: str, val_split: float, subset: SubsetEnum,
                               label_mode: LabelModeEnum) -> tf.data.Dataset:
    return tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=val_split,
        subset=subset,
        seed=123,
        image_size=(config.img_w, config.img_w),
        color_mode='rgb',
        label_mode=label_mode,

    )


def fineTune(data_path, outFileName):
    train_dataset = get_dataset_from_directory(data_path, 0.2, SubsetEnum.Train.value, LabelModeEnum.Int.value)

    validation_dataset = get_dataset_from_directory(data_path, 0.2, SubsetEnum.Val.value,
                                                    LabelModeEnum.Int.value)

    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 5)
    validation_dataset = validation_dataset.skip(val_batches // 5)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_ds = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    from tensorflow import keras

    data_augmentation = keras.Sequential()

    base_model = keras.applications.MobileNetV2(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=(config.img_w, config.img_w, 3),
        include_top=False,
    )  # Do not include the ImageNet classifier at the top.

    # Freeze the base_model
    base_model.trainable = False

    # Create new model on top
    inputs = keras.Input(shape=(config.img_w, config.img_w, 3))
    x = data_augmentation(inputs)  # Apply random data augmentation

    # Pre-trained Xception weights requires that input be scaled
    # from (0, 255) to a range of (-1., +1.), the rescaling layer
    x = tf.keras.applications.mobilenet.preprocess_input(x)

    # The base model contains batchnorm layers. We want to keep them in inference mode
    # when we unfreeze the base model for fine-tuning, so we make sure that the
    # base_model is running in inference mode here.
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(units=68, activation='relu',
                              kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),
                              )(x)
    x = keras.layers.Dropout(0.4)(x)  # Regularize with dropout
    outputs = keras.layers.Dense(4, activation='softmax')(x)
    model = keras.Model(inputs, outputs)

    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    early = tf.keras.callbacks.EarlyStopping(patience=3,

                                             restore_best_weights=True,
                                             monitor="val_loss", )

    epochs = 20
    history = model.fit(train_ds, epochs=epochs, validation_data=validation_ds, callbacks=[early])

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

    accuracy_score = model.evaluate(test_dataset)
    print(accuracy_score)
    print("Accuracy: {:.4f}%".format(accuracy_score[1] * 100))

    print("Loss: ", accuracy_score[0])

    base_model.trainable = True
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    epochs = 10
    history_fined = model.fit(train_ds, epochs=epochs, validation_data=validation_ds, callbacks=[early])

    model.save(outFileName)

    acc = history_fined.history['sparse_categorical_accuracy']
    val_acc = history_fined.history['val_sparse_categorical_accuracy']
    loss = history_fined.history['loss']
    val_loss = history_fined.history['val_loss']

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

    accuracy_score = model.evaluate(test_dataset)
    print(accuracy_score)
    print("Accuracy: {:.4f}%".format(accuracy_score[1] * 100))

    print("Loss: ", accuracy_score[0])

    y_pred = []  # store predicted labels
    y_true = []  # store true labels

    # iterate over the dataset
    for image_batch, label_batch in test_dataset:  # use dataset.unbatch() with repeat
        # append true labels
        y_true.append(label_batch)
        # compute predictions
        preds = model.predict(image_batch)
        # append predicted labels
        y_pred.append(np.argmax(preds, axis=- 1))

    # convert the true and predicted labels into tensors

    pd.set_option("plotting.backend", "plotly")
    pd.options.plotting.backend = 'plotly'

    correct_labels = tf.concat([item for item in y_true], axis=0)
    predicted_labels = tf.concat([item for item in y_pred], axis=0)
    ConfusionMatrixDisplay(confusion_matrix(predicted_labels, correct_labels)).plot()
    plt.show()
    print(confusion_matrix(predicted_labels, correct_labels))


def load(data_path, model):
    # train_dataset = get_dataset_from_directory(data_path, 0.2, SubsetEnum.Train.value, LabelModeEnum.Categorical.value)

    validation_dataset = get_dataset_from_directory(data_path, 0.2, SubsetEnum.Val.value,
                                                    LabelModeEnum.Categorical.value)

    val_batches = tf.data.experimental.cardinality(validation_dataset)
    t = validation_dataset.take(val_batches // 5)
    validation_dataset = validation_dataset.skip(val_batches // 5)

    AUTOTUNE = tf.data.AUTOTUNE

    # train_ds = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_ds = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    t_ds = t.prefetch(buffer_size=AUTOTUNE)

    # accuracy_score = model.evaluate(train_ds)
    # print(accuracy_score)
    # print("Accuracy: {:.4f}%".format(accuracy_score[1] * 100))
    #
    # print("Loss: ", accuracy_score[0])

    y_pred = []  # store predicted labels
    y_true = []  # store true labels
    #
    # iterate over the dataset
    for image_batch, label_batch in t_ds:  # use dataset.unbatch() with repeat
        # append true labels
        y_true.append(label_batch)
        # compute predictions
        preds = model.predict(image_batch)
        # append predicted labels
        y_pred.append(np.argmax(preds, axis=- 1))

    # convert the true and predicted labels into tensors
    correct_labels = tf.concat([item for item in y_true], axis=0)
    predicted_labels = tf.concat([item for item in y_pred], axis=0)
    ConfusionMatrixDisplay(confusion_matrix(predicted_labels, correct_labels)).plot()
    plt.show()
    print(confusion_matrix(predicted_labels, correct_labels))



if __name__ == '__main__':
    # import tensorflow as tf
    fineTune('E:\\functional', "fine_tuned_categorical_purpose.h5")
    # from tensorflow import keras

    # history_fined = keras.models.load_model(
    #     'C:\\Users\\snako\\PycharmProjects\\pythonProject\\fine_tuned_categorical_purpose.h5', )
    # load('E:\\functional', history_fined)
    pass
