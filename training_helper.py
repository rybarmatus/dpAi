import enum

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import config


class SubsetEnum(enum.Enum):
    Train = 'training'
    Val = 'validation'


class LabelModeEnum(enum.Enum):
    Categorical = 'categorical'
    Binary = 'binary'
    Int = 'int'


class AccuracyTypeEnum(enum.Enum):
    Categorical = 'categorical_accuracy'
    ValCategorical = 'val_categorical_accuracy'
    SparceCategorical = 'sparse_categorical_accuracy'
    ValSparseCategorical = 'val_sparse_categorical_accuracy'
    Binary = 'binary_accuracy'
    ValBinary = 'val_binary_accuracy'


def get_dataset_from_directory(data_path: str,
                               val_split: float,
                               subset: SubsetEnum,
                               label_mode: LabelModeEnum,
                               width: int = config.img_w,
                               height: int = config.img_h,
                               ) -> tf.data.Dataset:
    return tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=val_split,
        subset=subset.value,
        seed=123,
        image_size=(width, height),
        color_mode='rgb',
        label_mode=label_mode.value,
    )


def plot_accuracy(history, train_type: AccuracyTypeEnum, val_type: AccuracyTypeEnum):
    acc = history.history[train_type.value]
    val_acc = history.history[val_type.value]

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


def plot_loss(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss. \nTrain Loss: {str(loss[-1])}\nValidation Loss: {str(val_loss[-1])}')
    plt.xlabel('epoch')
    plt.tight_layout(pad=3.0)


def plot_training(history, train_type: AccuracyTypeEnum, val_type: AccuracyTypeEnum):
    plot_accuracy(history, train_type, val_type)
    plot_loss(history)
    plt.show()


def do_plot_big_confusion(predicted_labels, correct_labels, name_classes):
    cm = confusion_matrix(predicted_labels, correct_labels)
    ConfusionMatrixDisplay(cm, display_labels=name_classes).plot()
    plt.show()

    cm_df = pd.DataFrame(cm,
                         index=name_classes,
                         columns=name_classes)
    ax = sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, rotation_mode='anchor', ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=-45, rotation_mode='anchor', ha='right')
    ax.figure.savefig('hmx.png', transparent=True, bbox_inches='tight')
    plt.close()

    print("---- CONFUSION MATRIX ----")
    print(confusion_matrix(predicted_labels, correct_labels))


def plot_confusion(predicted_labels, correct_labels, name_classes):
    cm = confusion_matrix(predicted_labels, correct_labels)
    ConfusionMatrixDisplay(cm, display_labels=name_classes).plot()
    plt.show()

    cm_df = pd.DataFrame(cm,
                         index=name_classes,
                         columns=name_classes)
    plt.figure(figsize=(20, 10))
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    plt.show()

    print("---- CONFUSION MATRIX ----")
    print(confusion_matrix(predicted_labels, correct_labels))


def do_evaluate(model: keras.Model, test_dataset: tf.data.Dataset, name_classes,
                plot_big=False):
    y_pred = []  # store predicted labels
    y_true = []  # store true labels

    # convert the true and predicted labels into tensors

    pd.set_option("plotting.backend", "plotly")
    pd.options.plotting.backend = 'plotly'

    # iterate over the dataset
    for image_batch, label_batch in test_dataset:
        # append true labels
        y_true.append(label_batch)
        # compute predictions
        preds = model.predict(image_batch)
        # append predicted labels
        y_pred.append(np.argmax(preds, axis=- 1))

    correct_labels = tf.concat([item for item in y_true], axis=0)
    predicted_labels = tf.concat([item for item in y_pred], axis=0)

    if plot_big:
        do_plot_big_confusion(predicted_labels, correct_labels, name_classes)
    else:
        plot_confusion(predicted_labels, correct_labels, name_classes)


def print_accuracy(model: keras.Model, test_dataset: tf.data.Dataset):
    accuracy_score = model.evaluate(test_dataset)
    print(accuracy_score)
    print("Accuracy: {:.4f}%".format(accuracy_score[1] * 100))

    print("Loss: ", accuracy_score[0])
