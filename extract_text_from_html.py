from bs4 import BeautifulSoup
import os
import re


import config
import translate_text
import pandas as pd
import string

import os

import tensorflow as tf
import tensorflow_hub as hub

import matplotlib.pyplot as plt

# tf.get_logger().setLevel('ERROR')
#
tf.compat.v1.disable_eager_execution()
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
TF_FORCE_GPU_ALLOW_GROWTH = 1
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


# odstrani interpunkciu, nie latinkove slova, lowercase, znaky
def preprocess_text(page_str: str) -> str:
    alpha_words = [word for word in page_str.split() if word.isalpha()]
    page_str = " ".join(alpha_words)
    page_str = page_str.lower()
    page_str = re.sub("([^\x00-\x7F])+", " ", page_str)
    page_str = re.sub(r'\d +', "", page_str)
    page_str = page_str.translate(str.maketrans('', '', string.punctuation))
    page_str = page_str.strip()
    return page_str


def do_extract():
    text_elements = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']
    df = pd.DataFrame(columns=['page', 'category', 'text'])
    index = 0
    for dirpath, dirnames, filenames in os.walk(config.html_folder):
        for f in filenames:
            index += 1
            print(f)
            filename = os.fsdecode(f)
            if filename.endswith(".html"):
                page_str: str = ''
                with open(dirpath + '\\' + filename, encoding="utf8") as fp:

                    soup = BeautifulSoup(fp.read(), "html.parser")
                    title = soup.find('title')
                    if title is not None:
                        page_str += title.text
                    description = soup.find('meta', attrs={'name': 'description'})
                    if "content" in str(description):
                        description = description.get("content")
                        page_str += ' '
                        page_str += description

                    for el in text_elements:
                        found_elements = soup.find_all(el)
                        for found_element_text in found_elements:
                            if found_element_text is not None:
                                page_str += ' '
                                page_str += found_element_text.text
                if page_str.__contains__('403 Forbidden'): continue
                if page_str.__contains__('Problém pri načítaní stránky'): continue
                if page_str.__contains__('Server sa nenašiel'): continue
                if page_str.__contains__('Access denied'): continue
                if page_str.__contains__('The page is temporarily unavailable'): continue
                if page_str.__contains__('Please Wait...'): continue
                if page_str.__contains__('Error 403'): continue
                if page_str.__contains__('Just a moment...'): continue

                if len(page_str) < 1:
                    continue
                page_str = translate_text.translate_if_needed(page_str)

                page = f
                category = dirpath.split('\\')[3]

                page_str = preprocess_text(page_str)
                if len(page_str) < 1:
                    continue
                if page_str.__contains__('see relevant content for'):
                    continue
                df = df.append({'page': page, 'category': category, 'text': page_str}, ignore_index=True)

    df.to_csv(config.web_texts, index=False, header=True)


def get_category_as_num(x: str):
    if x == 'Arts_and_Entertainment':
        return 0
    if x == 'Business_and_Consumer_Services':
        return 1
    if x == 'commerce_and_Shopping':
        return 2
    if x == 'Community_and_Society':
        return 3
    if x == 'Computers_Electronics_and_Technology':
        return 4
    if x == 'Finance':
        return 5
    if x == 'Food_and_Drink':
        return 6
    if x == 'Gambling':
        return 7
    if x == 'Health':
        return 8
    if x == 'Heavy_Industry_and_Engineering':
        return 9
    if x == 'Lifestyle':
        return 10
    if x == 'News_and_Media':
        return 11
    if x == 'Science_and_Education':
        return 12
    if x == 'Sports':
        return 13
    exit(0)


def bert():
    from sklearn.model_selection import train_test_split
    df = pd.read_csv('web_texts.csv')
    df['category'] = df['category'].apply(lambda x: get_category_as_num(x))
    df = df.drop(columns=['page'])

    X_train, X_test, y_train, y_test = train_test_split(df['text'].values, df['category'].values, test_size=0.2,
                                                        random_state=1)

    # y_test = OneHotEncoder().fit_transform(X_test)
    # y_train = OneHotEncoder().fit_transform(X_train)

    bert_preprocess = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2")
    bert_encoder = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base/1")

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessed_text = bert_preprocess(text_input)
    outputs = bert_encoder(preprocessed_text)

    # pridane vrstvy
    net = (outputs['pooled_output'])
    net = tf.keras.layers.Dropout(0.1, name="dropout")(net)
    net = tf.keras.layers.Dense(14, activation='softmax', name="output")(net)

    model = tf.keras.Model(inputs=[text_input], outputs=[net])

    # init_lr = 3e-5
    # optimizer = optimization.create_optimizer(init_lr=init_lr,
    #                                           num_train_steps=num_train_steps,
    #                                           num_warmup_steps=num_warmup_steps,
    #                                           optimizer_type='adamw')

    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                          patience=3,
                                                          restore_best_weights=True)

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    history = model.fit(X_train, y_train, epochs=15, validation_split=0.2, batch_size=32, callbacks=[earlystop_callback])

    model.save('text_classifier.h5')
    y_predicted = model.predict(X_test)
    # y_predicted = y_predicted.flatten()

    loss, accuracy = model.evaluate(X_test)
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')

    history_dict = history.history
    print(history_dict.keys())

    acc = history_dict['sparse_categorical_accuracy']
    val_acc = history_dict['val_sparse_categorical_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)
    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()

    plt.subplot(2, 1, 1)
    # r is for "solid red line"
    plt.plot(epochs, loss, 'r', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.show()


if __name__ == '__main__':
    # do_extract()
    bert()
