# casti kodu cerpane z https://keras.io/guides/transfer_learning/
from keras import regularizers
from tensorflow.python.framework.config import set_memory_growth

from training_helper import *

# tf.compat.v1.disable_v2_behavior()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def fineTune(data_path):
    train_dataset = get_dataset_from_directory(data_path, 0.3, SubsetEnum.Train,
                                               LabelModeEnum.Int, width=config.img_w_fine,
                                               height=config.img_h_fine)

    validation_dataset = get_dataset_from_directory(data_path, 0.3, SubsetEnum.Val,
                                                    LabelModeEnum.Int, width=config.img_w_fine,
                                                    height=config.img_h_fine)

    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 4)
    validation_dataset = validation_dataset.skip(val_batches // 4)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_ds = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    from tensorflow import keras

    name_classes = train_dataset.class_names
    data_augmentation = keras.Sequential()

    base_model = keras.applications.InceptionResNetV2(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=(config.img_w_fine, config.img_h_fine, 3),
        include_top=False,
    )  # Do not include the MobileNet classifier at the top.

    # Freeze the base_model
    base_model.trainable = False

    # Create new model on top
    inputs = keras.Input(shape=(config.img_w_fine, config.img_h_fine, 3))
    x = data_augmentation(inputs)

    x = tf.keras.applications.inception_resnet_v2.preprocess_input(x)

    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(units=68, activation='relu',
                              kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),
                              )(x)
    x = keras.layers.Dropout(0.4)(x)  # Regularize with dropout
    outputs = keras.layers.Dense(14, activation='softmax')(x)
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

    plot_training(history, AccuracyTypeEnum.SparceCategorical, AccuracyTypeEnum.ValSparseCategorical)
    print_accuracy(model, test_dataset)

    base_model.trainable = True
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    epochs = 20
    history_fined = model.fit(train_ds, epochs=epochs, validation_data=validation_ds, callbacks=[early])

    model.save(config.all_categories_weight_name)

    plot_training(history_fined, AccuracyTypeEnum.SparceCategorical, AccuracyTypeEnum.ValSparseCategorical)
    print_accuracy(model, test_dataset)
    do_evaluate(model, test_dataset, name_classes, plot_big=True)


if __name__ == '__main__':
    fineTune(config.all_categories_image_path)
