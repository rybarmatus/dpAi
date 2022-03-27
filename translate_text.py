# ceprane z https://towardsdatascience.com/how-to-detect-and-translate-languages-for-nlp-project-dfd52af0c3b5

import time

# from google_trans_new import google_translator
from googletrans import Translator
from langdetect import detect

time.sleep(5)


def translate_if_needed(to_translate: str) -> str:
    if to_translate is None: return ''
    if len(to_translate) < 1: return ''
    translator = Translator()
    # netreba prekladat, lebo to uz je v anglictine
    try:
        lang = detect(to_translate)
        if lang == 'en':
            return to_translate
        return translator.translate(to_translate).text
    except:
        return ''
