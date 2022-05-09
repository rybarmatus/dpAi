import config
from training_helper import *


def tranfser_InceptionResNetV2(neurons, l1, l2, dropout):
    import tensorflow as tf
    from keras import regularizers


    train_dataset = get_dataset_from_directory(config.binary_image_path, 0.3, SubsetEnum.Train,
                                               LabelModeEnum.Binary, width=config.img_w_fine,
                                               height=config.img_h_fine)

    validation_dataset = get_dataset_from_directory(config.binary_image_path, 0.3, SubsetEnum.Val,
                                                    LabelModeEnum.Binary, width=config.img_w_fine,
                                                    height=config.img_h_fine)

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

    base_model = keras.applications.InceptionResNetV2(
        weights="imagenet",
        input_shape=(config.img_w_fine, config.img_h_fine, 3),
        include_top=False,
    )

    # Freeze the base_model
    base_model.trainable = False

    # Create new model on top
    inputs = keras.Input(shape=(config.img_w_fine, config.img_h_fine, 3))
    x = data_augmentation(inputs)

    x = tf.keras.applications.inception_resnet_v2.preprocess_input(x)

    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(units=neurons, activation='relu',
                              kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                              )(x)
    x = keras.layers.Dropout(dropout)(x)  # Regularize with dropout
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)

    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()],
    )

    early = tf.keras.callbacks.EarlyStopping(patience=3,
                                             restore_best_weights=True,
                                             monitor="val_loss", )

    epochs = 20
    history = model.fit(train_ds, epochs=epochs, validation_data=validation_ds, callbacks=[early], batch_size=32)

    plot_training(history, AccuracyTypeEnum.Binary, AccuracyTypeEnum.ValBinary)

    print_accuracy(model, test_dataset)
    do_evaluate(model, test_dataset, name_classes)

