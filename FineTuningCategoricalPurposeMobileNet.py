from keras import regularizers

from training_helper import *


def fine_tune():
    train_dataset = get_dataset_from_directory(config.purpose_image_path, 0.3, SubsetEnum.Train,
                                               LabelModeEnum.Int)

    validation_dataset = get_dataset_from_directory(config.purpose_image_path, 0.3, SubsetEnum.Val,
                                                    LabelModeEnum.Int)

    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 4)
    validation_dataset = validation_dataset.skip(val_batches // 4)

    name_classes = train_dataset.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_ds = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    from tensorflow import keras

    data_augmentation = keras.Sequential()

    base_model = keras.applications.MobileNetV2(
        weights="imagenet",
        input_shape=(config.img_w, config.img_w, 3),
        include_top=False,
    )

    # Freeze the base_model
    base_model.trainable = False

    # Create new model on top
    inputs = keras.Input(shape=(config.img_w, config.img_w, 3))
    x = data_augmentation(inputs)  # Apply random data augmentation

    x = tf.keras.applications.mobilenet.preprocess_input(x)

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

    plot_training(history, AccuracyTypeEnum.SparceCategorical, AccuracyTypeEnum.ValSparseCategorical)

    print_accuracy(model, test_dataset)

    base_model.trainable = True
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    epochs = 10
    history_fined = model.fit(train_ds, epochs=epochs, validation_data=validation_ds, callbacks=[early])

    model.save(config.purpose_weights_name)

    plot_training(history_fined, AccuracyTypeEnum.SparceCategorical, AccuracyTypeEnum.ValSparseCategorical)
    print_accuracy(model, test_dataset)
    do_evaluate(model, test_dataset, name_classes)


if __name__ == '__main__':
    fine_tune()
    pass
