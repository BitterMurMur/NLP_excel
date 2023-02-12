import nltk
import numpy as np
from matplotlib import pyplot as plt
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk import BlanklineTokenizer
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


class ExcelParser():
    def __init__(self, excel_name, file_sheetname):
        self.excel_name = excel_name
        self.file_sheetname = file_sheetname

    def read_excel(self):
        data_frame = pd.read_excel(self.excel_name, sheet_name=self.file_sheetname)
        return data_frame.drop (columns=['Unnamed: 0','Unnamed: 25', 'Unnamed: 26'], index=[ 0,1,2,3])

class NLP():

    def __init__(self):
        self.stop_words = ['nan', ' ', '\n', 'тег отсутствует', 'Тег состояния технологического объекта не предусмотрен', 'на режиме', 'вкл', 'выкл','включена', 'отключена',
                           'тег состояния технологического объекта не предусмотрен', 'б/н']
        self.stop_words_for_sents = ['nan', '\n']

    def tokenize_words(self, words):
        #предложения из ячеек тоже будем считать отдельными токенами, т.к. это описание и его менять нельзя
        out_words = []
        for  word in  words:
            for item in word:
                tokens = str(item).lower()
                out_words.append(tokens)
        out_words = self.delete_stopwords_words(out_words)
        self.word_tokens = out_words

    #датасет изначально уже практически токенизирован, просто делаем из листа единое предложение
    def tokenize_sents(self, sents):
        out_sents=[]
        for sent in sents:
            tokens=  self.clean_tokens(str(sent).lower())
            out_sents.append(tokens)
        self.preprocessing_sents(out_sents)
        self.sent_tokens =  out_sents

    def clean_tokens(self, str):
        replace_symbols = ["[", "]", ",", "'", "{","}"]
        out_str = str
        for sym in replace_symbols:
            out_str = out_str.replace(sym, '')
        out_str = out_str.replace('\\n', ' ')
        out_str = out_str.replace('\n', ' ')
        return out_str

    def delete_stopwords_words(self, tokens):
        without_stopwords = [token for token in tokens if token not in self.stop_words]
        return without_stopwords

    def preprocessing_sents(self, tokens):
        cleanr = re.compile('[!@"“’«»#$%&\'()*+,—/:;<=>?^_`{|}~\[\]]')
        for i, token in enumerate(tokens):
            for sym in self.stop_words_for_sents:
                tmp_str = tokens[i].replace(sym, '')
                cleantext = re.sub(cleanr, '', tmp_str)
                rem_num = re.sub('[0-9]+', '', cleantext)
                tokens[i] = rem_num

    def vectorize(self, tokens_sent, most_count):
        new_custom_stop_words = ['поз', 'режиме', 'тег', 'anhk','2022', 'раздел', 'datetime', 'time', '2023', 'blok', 'uv', 'ob',
                                 'другой', 'ей', 'какоий', 'мои', 'ней', 'объекта', 'отсутствует', 'предусмотрен', 'сейчас', 'состояния', 'такой', 'технологического', 'этой']
        stop_words_new = stopwords.words('russian')
        stop_words_new.extend(new_custom_stop_words)
        stop_words_new.extend(self.stop_words)

        vecorize = CountVectorizer(strip_accents='unicode', stop_words=stop_words_new)
        words = vecorize.fit_transform(tokens_sent)
        word_frequency = pd.DataFrame(
          {'word' : vecorize.get_feature_names_out(),
          # получаем частотность слов
            # находя сумму компонент векторов
          'frequency' : np.array(words.sum(axis = 0))[0]
            }).sort_values(by = 'frequency', ascending = False)
        return word_frequency[:most_count]


class Pipeline:

    def __init__(self, excel_name, sheetname):
        self.excel_name = excel_name
        self.excel_sheetname = sheetname



    def show_most_common(self, most_count):
        ew = ExcelParser(self.excel_name, self.excel_sheetname)
        work_data = ew.read_excel()
        nlp = NLP()
        data = work_data.values.tolist()
        nlp.tokenize_words(data)
        fdist = FreqDist(nlp.word_tokens)
        fdist.most_common(most_count)
        fdist.plot(most_count, cumulative=False)

    def show_most_common_vectorized(self, most_count):
        ew = ExcelParser(self.excel_name, self.excel_sheetname)
        work_data = ew.read_excel()
        nlp = NLP()
        data = work_data.values.tolist()
        nlp.tokenize_sents(data)
        word_frequency_filtered = nlp.vectorize(nlp.sent_tokens, most_count)
        fig, ax = plt.subplots()
        ax.plot(word_frequency_filtered['word'], word_frequency_filtered['frequency'])
        ax.grid()
        plt.xticks(rotation=90)
        plt.show()

def main():
    nltk.download('punkt')
    nltk.download('stopwords')
    pipe = Pipeline("test_dataset.xlsx", "Data")
    pipe.show_most_common(30)
    pipe.show_most_common_vectorized(30)


if __name__=="__main__":
    main()