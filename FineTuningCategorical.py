import matplotlib.pyplot as plt
import tensorflow as tf

from keras import regularizers

# casti kodu cerpane z https://keras.io/guides/transfer_learning/

def fineTune(data_path, outFileName):
    img_h = 150
    img_w = 150
    batch_size = 32

    train_data = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=(img_w, img_h),
        color_mode='rgb',
        label_mode='categorical',
    )

    validation_data = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=(img_w, img_h),
        color_mode='rgb',
        label_mode='categorical',
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_data.cache().prefetch(buffer_size=autotune)
    validation_ds = validation_data.cache().prefetch(buffer_size=autotune)

    from tensorflow import keras

    data_augmentation = keras.Sequential()
    # TODO InceptionResnt
    base_model = keras.applications.Xception(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=(img_w, img_h, 3),
        include_top=False,
    )  # Do not include the ImageNet classifier at the top.

    # Freeze the base_model
    base_model.trainable = False

    # Create new model on top
    inputs = keras.Input(shape=(img_w, img_h, 3))
    x = data_augmentation(inputs)  # Apply random data augmentation

    # Pre-trained Xception weights requires that input be scaled
    # from (0, 255) to a range of (-1., +1.), the rescaling layer
    # outputs: `(inputs * scale) + offset`
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
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )

    early = tf.keras.callbacks.EarlyStopping(patience=3,

                                             restore_best_weights=True,
                                             monitor="val_loss", )

    epochs = 20
    history = model.fit(train_ds, epochs=epochs, validation_data=validation_ds, callbacks=[early])

    plot_training(history, AccuracyTypeEnum.SparceCategorical, AccuracyTypeEnum.ValSparseCategorical)

    print_accuracy(model, test_dataset)

    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
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

    accuracy_score = model.evaluate(validation_data)
    print(accuracy_score)
    print("Accuracy: {:.4f}%".format(accuracy_score[1] * 100))

    print("Loss: ", accuracy_score[0])

    base_model.trainable = True
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )

    epochs = 1
    history_fined = model.fit(train_ds, epochs=epochs, validation_data=validation_ds, callbacks=[early])

    model.save(outFileName)

    acc = history_fined.history["categorical_accuracy"]
    val_acc = history_fined.history['val_categorical_accuracy']
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

    accuracy_score = model.evaluate(validation_data)
    print(accuracy_score)
    print("Accuracy: {:.4f}%".format(accuracy_score[1] * 100))

    print("Loss: ", accuracy_score[0])
