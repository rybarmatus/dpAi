# casti kodu cerpane z https://keras.io/guides/transfer_learning/
from keras import regularizers
from tensorflow.python.framework.config import set_memory_growth

from config import batch_size
from training_helper import *

# tf.compat.v1.disable_v2_behavior()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def fineTune(data_path, outFileName):
    train_dataset = get_dataset_from_directory(data_path, 0.3, SubsetEnum.Train,
                                               LabelModeEnum.Binary, width=config.img_w_fine,
                                               height=config.img_h_fine)

    validation_dataset = get_dataset_from_directory(data_path, 0.3, SubsetEnum.Val,
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

    base_model = keras.applications.Xception(
        weights="imagenet",
        input_shape=(config.img_w_fine, config.img_h_fine, 3),
        include_top=False,
    )

    # Freeze the base_model
    base_model.trainable = False

    # Create new model on top
    inputs = keras.Input(shape=(config.img_w_fine, config.img_h_fine, 3))
    x = data_augmentation(inputs)  # Apply random data augmentation

    # Pre-trained Xception weights requires that input be scaled
    # from (0, 255) to a range of (-1., +1.), the rescaling layer
    scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
    x = scale_layer(x)

    # The base model contains batchnorm layers. We want to keep them in inference mode
    # when we unfreeze the base model for fine-tuning, so we make sure that the
    # base_model is running in inference mode here.
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(units=68, activation='relu',
                              kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001),
                              )(x)
    x = keras.layers.Dropout(0.4)(x)  # Regularize with dropout
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

    base_model.trainable = True
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()],
    )

    epochs = 10
    early = tf.keras.callbacks.EarlyStopping(patience=2,
                                             restore_best_weights=True,
                                             monitor="val_loss", )

    history_fined = model.fit(train_ds, epochs=epochs, validation_data=validation_ds, callbacks=[early])

    model.save(outFileName)

    plot_training(history_fined, AccuracyTypeEnum.Binary, AccuracyTypeEnum.ValBinary)
    print_accuracy(model, test_dataset)
    do_evaluate(model, test_dataset, name_classes)


if __name__ == '__main__':
    fineTune(config.binary_image_path, config.binary_weights_name)
