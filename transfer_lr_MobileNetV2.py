def mobileNetV2(neurons, l1, l2, dropout):
    import matplotlib.pyplot as plt

    import tensorflow as tf
    from keras import regularizers

    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns

    print(l1, l2, neurons, dropout)

    batch_s = 32
    img_h = 224
    img_w = 224
    data_path = 'D:\\blogs-news-media'

    train_batchsize = 100
    val_batchsize = 10

    train_data = tf.keras.preprocessing.image.ImageDataGenerator(
        data_path,
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=(img_w, img_h),
        batch_size=train_batchsize,
        color_mode='rgb',
        label_mode='binary',
        shuffle=True,
        rescale=1. / 255
    )

    validation_data = tf.keras.preprocessing.image.ImageDataGenerator(
        data_path,
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=(img_w, img_h),
        batch_size=val_batchsize,
        color_mode='rgb',
        label_mode='binary',
        rescale=1. / 255
    )

    train_data = tf.keras.applications.mobilenet.preprocess_input(train_data)
    validation_data = tf.keras.applications.mobilenet.preprocess_input(validation_data)

    autotune = tf.data.AUTOTUNE
    train_data = train_data.cache().prefetch(buffer_size=autotune)
    validation_data = validation_data.cache().prefetch(buffer_size=autotune)

    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(img_w, img_h, 3)  # TODO mensie inputy skusit
    )

    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.BatchNormalization(renorm=True),
        tf.keras.layers.GlobalAveragePooling2D(),
        # tf.keras.layers.Dense(512, activation='relu'), # TODO mensiu siet
        # tf.keras.layers.Dropout(dropout),
        # tf.keras.layers.Dense(units=neurons, activation='relu',
        #                       kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
        #                       ),
        # tf.keras.layers.Dense(256, activation='relu'),
        # tf.keras.layers.Dropout(0.4),  # menej, viac dropoutov + l1/l2 regularizacie
        # tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(units=neurons, activation='relu',
                              kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                              ),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    early = tf.keras.callbacks.EarlyStopping(patience=10,
                                             min_delta=0.001,
                                             restore_best_weights=True)

    # fit model
    history = model.fit(train_data,
                        batch_size=batch_s,
                        verbose=1,
                        validation_data=validation_data,
                        epochs=15,
                        callbacks=[early])  # TODO https://keras.io/api/callbacks/model_checkpoint/

    model.save("Model2.h5")

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
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

