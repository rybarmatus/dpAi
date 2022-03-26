import numpy as np
from bs4 import BeautifulSoup
import os

import config
import translate_text
import pandas as pd


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
                arr = np.array([page, category, page_str])
                # df_to_add = pd.DataFrame(columns=['page', 'category', 'text'])
                # df_to_add.loc[0] = arr
                # df = pd.concat([df, df_to_add], axis=1)
                df = df.append({'page': page, 'category': category, 'text': page_str}, ignore_index=True)
                if index == 100:
                    df.to_csv(config.web_texts, index=False, header=True)
                    exit(0)

    df.to_csv(config.web_texts, index=False, header=True)


if __name__ == '__main__':
    do_extract()
