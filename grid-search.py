import matplotlib.pyplot as plt

import tensorflow as tf
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator


def transfer_Mobi():
    gen = ImageDataGenerator(
        rescale=1. / 255.,
        validation_split=0.2,  # training: 80% data, validation: 20% data
        zoom_range=0.3,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )
    train_generator = gen.flow_from_directory(
        'C:\data\dp',
        target_size=(224, 224),
        batch_size=32,
        subset="training",
        color_mode="rgb",
        class_mode='categorical',
        shuffle=True,
        seed=42, )

    validation_generator = gen.flow_from_directory(
        'C:\data\dp',
        target_size=(224, 224),
        batch_size=32,
        subset="validation",
        color_mode="rgb",
        class_mode='categorical',
        shuffle=True,
        seed=42, )

    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)

    )

    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.BatchNormalization(renorm=True),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        # tf.keras.layers.Dense(units=64, activation='relu',
        #                       kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.01),
        #                       bias_regularizer=regularizers.l2(0.01),
        #                       activity_regularizer=regularizers.l2(0.001)
        #                       ),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(13, activation='softmax')
    ])

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    early = tf.keras.callbacks.EarlyStopping(patience=10,
                                             min_delta=0.001,
                                             restore_best_weights=True)

    # fit model
    history = model.fit(train_generator,
                        workers=6,
                        validation_data=validation_generator,
                        epochs=25,
                        callbacks=[early])

    model.save("Model.h5")

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
        '\nTraining and Validation Accuracy. \nTrain Accuracy: {str(acc[-1])}\nValidation Accuracy: {str(val_acc[-1])}')

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

    accuracy_score = model.evaluate(validation_generator)
    print(accuracy_score)
    print("Accuracy: {:.4f}%".format(accuracy_score[1] * 100))

    print("Loss: ", accuracy_score[0])


def tranfser_InceptionResNetV2(neurons, l1, l2, dropout):
    print(l1, l2, neurons, dropout)

    batch_s = 32
    img_h = 150
    img_w = 150
    data_path = 'C:\data\dp'

    train_data = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=(img_w, img_h),
        batch_size=batch_s,
        color_mode='rgb',
        label_mode='categorical',
    )

    validation_data = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=(img_w, img_h),
        batch_size=batch_s,
        color_mode='rgb',
        label_mode='categorical',
    )

    autotune = tf.data.AUTOTUNE
    train_data = train_data.cache().prefetch(buffer_size=autotune)
    validation_data = validation_data.cache().prefetch(buffer_size=autotune)

    base_model = tf.keras.applications.InceptionResNetV2(
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
        # tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(dropout),  # menej, viac dropoutov + l1/l2 regularizacie
        # tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(units=neurons, activation='relu',
                              kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                              ),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(13, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
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

    model.save("Model.h5")

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


if __name__ == '__main__':
    neurons = [100]
    l1s = [0.0001, 0.001]
    l2s = [0.0001, 0.001]
    dropouts = [0.2, 0.5]
    counter = 0
    for neuron in neurons:
        for l1 in l1s:
            for l2 in l2s:
                for dropout in dropouts:
                    if counter < 4:
                        counter += 1
                        continue
                    tranfser_InceptionResNetV2(neuron, l1, l2, dropout)
                    continue
    # transfer_Mobi()
    # tranfser_InceptionResNetV2(neuron, l1, l2, dropout)
    exit()
