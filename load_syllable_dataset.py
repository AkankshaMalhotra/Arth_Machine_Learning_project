from config import syllable_9,syllable_10, syllable_11, syllable_12
import pandas as pd
import syllable
class load():
    def load(self):
        '''
        1) Loading corpus of words containing the words having syllables from 1 to 6
        2) Loading text files having syllables 7 and 8. Words having syllables 9,10,11,12 are in the form of list.
        3) Creating dataset
        :return:
        '''
        syllable_1_6 = pd.read_excel("syllable.xlsx", headers=None, names=["word", "syllable"])
        with open("Syllable-7", "r") as g:
            syllable_7 = g.read()
        with open("Syllable-8", "r") as g:
            syllable_8 = g.read()
        syllable_7 = syllable_7.split()
        syllable_8 = syllable_8.split()
        syllable_1_6_data = [(syllable_1_6["word"][i].lower(), syllable_1_6["syllable"][i]) for i in
                             xrange(len(syllable_1_6["syllable"]))]
        syllable_7_data = [(i.lower(), 7) for i in syllable_7]
        syllable_8_data = [(i.lower(), 8) for i in syllable_8]
        syllable_9_data = [(i.lower(), 9) for i in syllable_9]
        syllable_10_data = [(i.lower(), 10) for i in syllable_10]
        syllable_11_data = [(i.lower(), 11) for i in syllable_11]
        syllable_12_data = [(i.lower(), 12) for i in syllable_12]
        syllable_data = list(
            set(syllable_1_6_data + syllable_7_data + syllable_8_data + syllable_9_data + syllable_10_data
                + syllable_11_data + syllable_12_data))
        return syllable_data

if __name__ == '__main__':
    load_obj=load()
    data=load_obj.load()
    obj=syllable.Syllable(data)
    accuracy=obj.save_model()
