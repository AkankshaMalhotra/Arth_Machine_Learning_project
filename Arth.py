from bs4 import BeautifulSoup
from scipy import stats
from sklearn.cluster import KMeans
from nltk.stem import WordNetLemmatizer
from simplified_lesk import Lesk
import re
import pickle as cPickle
import nltk
import numpy as np
from config import stopwords
import random
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet
import unicodedata


class Arth:
    def __init__(self, text):
        self.text = re.sub(u"(\u2018|\u2019)", "'", text)

    @staticmethod
    def get_wordnet_pos(words):
        data = []
        treebank_tag = nltk.pos_tag(words)
        for i in treebank_tag:
            if i[1].startswith('J'):
                v = wordnet.ADJ
            elif i[1].startswith('V'):
                v = wordnet.VERB
            elif i[1].startswith('N'):
                v = wordnet.NOUN
            elif i[1].startswith('R'):
                v = wordnet.ADV
            else:
                v = ''
            data.append((i[0], v))
        return data

    def preprocessing(self):
        """
         Implementing preprocessing steps.
         1) Removed HTML tags
         2) Removed Punctuation, numbers and special characters
         3) Word Tokenization of the text
         4) Generates Pos_tags for the words
         5) Find Appropriate lemma for the words
         6) Find Unique Words in the text
         7) Removal of stopwords, i.e. words that aren't important
        :return: preprocessed list of words
        """
        object = BeautifulSoup(self.text, "lxml")
        markup_removed_text = object.get_text()
        markup_removed_text = unicodedata.normalize('NFKD', markup_removed_text).encode('ascii', 'ignore')
        words_only_text = re.sub("[^a-zA-Z]", " ", markup_removed_text)
        words = [i.lower() for i in words_only_text.split()]
        word_postag_tuple = self.get_wordnet_pos(words)
        wordnet_lemmatizer = WordNetLemmatizer()
        lemma = {}
        for word, pos_tag in word_postag_tuple:
            if pos_tag != "":
                lem = wordnet_lemmatizer.lemmatize(word, pos=pos_tag)
                if word not in stopwords:
                    lemma[word] = lem
            else:
                if word not in stopwords:
                    lemma[word] = word
        return lemma

    def sylco(self, word):
        # I have not authored this function, taken from a blog by the person
        word = word.lower()
        # exception_add are words that need extra syllables
        # exception_del are words that need less syllable
        exception_add = ['serious', 'crucial']
        exception_del = ['fortunately', 'unfortunately']
        co_one = ['cool', 'coach', 'coat', 'coal', 'count', 'coin', 'coarse', 'coup', 'coif', 'cook', 'coign', 'coiffe',
                  'coof', 'court']
        co_two = ['coapt', 'coed', 'coinci']
        pre_one = ['preach']
        syls = 0  # added syllable number
        disc = 0  # discarded syllable number
        # 1) if letters < 3 : return 1
        if len(word) <= 3:
            syls = 1
            return syls
        # 2) if doesn't end with "ted" or "tes" or "ses" or "ied" or "ies", discard "es" and "ed" at the end.
        # if it has only 1 vowel or 1 set of consecutive vowels, discard. (like "speed", "fled" etc.)
        if word[-2:] == "es" or word[-2:] == "ed":
            doubleAndtripple_1 = len(re.findall(r'[eaoui][eaoui]', word))
            if doubleAndtripple_1 > 1 or len(re.findall(r'[eaoui][^eaoui]', word)) > 1:
                if word[-3:] == "ted" or word[-3:] == "tes" or word[-3:] == "ses" or word[-3:] == "ied" or word[
                                                                                                           -3:] == "ies":
                    pass
                else:
                    disc += 1
        # 3) discard trailing "e", except where ending is "le"
        le_except = ['whole', 'mobile', 'pole', 'male', 'female', 'hale', 'pale', 'tale', 'sale', 'aisle', 'whale',
                     'while']
        if word[-1:] == "e":
            if word[-2:] == "le" and word not in le_except:
                pass
            else:
                disc += 1

        # 4) check if consecutive vowels exists, triplets or pairs, count them as one.

        doubleAndtripple = len(re.findall(r'[eaoui][eaoui]', word))
        tripple = len(re.findall(r'[eaoui][eaoui][eaoui]', word))
        disc += doubleAndtripple + tripple
        # 5) count remaining vowels in word.
        numVowels = len(re.findall(r'[eaoui]', word))
        # 6) add one if starts with "mc"
        if word[:2] == "mc":
            syls += 1

        # 7) add one if ends with "y" but is not surrouned by vowel
        if word[-1:] == "y" and word[-2] not in "aeoui":
            syls += 1
        # 8) add one if "y" is surrounded by non-vowels and is not in the last word.
        for i, j in enumerate(word):
            if j == "y":
                if (i != 0) and (i != len(word) - 1):
                    if word[i - 1] not in "aeoui" and word[i + 1] not in "aeoui":
                        syls += 1

        # 9) if starts with "tri-" or "bi-" and is followed by a vowel, add one.
        if word[:3] == "tri" and word[3] in "aeoui":
            syls += 1
        if word[:2] == "bi" and word[2] in "aeoui":
            syls += 1

        # 10) if ends with "-ian", should be counted as two syllables, except for "-tian" and "-cian"
        if word[-3:] == "ian":
            # and (word[-4:] != "cian" or word[-4:] != "tian") :
            if word[-4:] == "cian" or word[-4:] == "tian":
                pass
            else:
                syls += 1

        # 11) if starts with "co-" and is followed by a vowel, check if exists in the double syllable dictionary, if not, check if in single dictionary and act accordingly.

        if word[:2] == "co" and word[2] in 'eaoui':
            if word[:4] in co_two or word[:5] in co_two or word[:6] in co_two:
                syls += 1
            elif word[:4] in co_one or word[:5] in co_one or word[:6] in co_one:
                pass
            else:
                syls += 1
        # 12) if starts with "pre-" and is followed by a vowel, check if exists in the double syllable dictionary, if not, check if in single dictionary and act accordingly

        if word[:3] == "pre" and word[3] in 'eaoui':
            if word[:6] in pre_one:
                pass
            else:
                syls += 1
        # 13) check for "-n't" and cross match with dictionary to add syllable.
        negative = ["doesn't", "isn't", "shouldn't", "couldn't", "wouldn't"]
        if word[-3:] == "n't":
            if word in negative:
                syls += 1
            else:
                pass
        # 14) Handling the exceptional words.
        if word in exception_del:
            disc += 1
        if word in exception_add:
            syls += 1
        # calculate the output
        return numVowels - disc + syls

    def syllable_calculater(self, word_list):
        '''

        :param word_list:
        :return: Dictionary of syllables
        '''
        #         syllable_obj = Syllable(word_list)
        #         syllable_dict=syllable_obj.model_load()
        syllable_dict = {}
        for i in word_list:
            syllable_dict[i] = self.sylco(i)
        return syllable_dict

    def usage_calc(self, word_list):
        freq_dict = {}
        with open("freq_dict.pickle", "rb") as g:
            freq = cPickle.load(g)
        for word in word_list:
            if word in freq:
                freq_dict[word] = freq[word]
            else:
                freq_dict[word] = 0
        return freq_dict

    def getfeatures_normalize(self):
        word_list = self.preprocessing()
        syll = self.syllable_calculater(list(word_list.keys()))
        frequency = self.usage_calc(list(word_list.values()))
        feature = []
        vocab = {k: v for k, v in enumerate(word_list)}
        for i in word_list:
            feature.append([frequency[word_list[i]], syll[i]])
        feature = np.array(feature)
        zscor = stats.zscore(feature, axis=0)
        return zscor, vocab

    def clustering(self, show):
        cluster_words = {}
        zsc, vocab = self.getfeatures_normalize()
        db = KMeans(n_clusters=5)
        db.fit(zsc)
        labels = db.labels_
        for i in set(labels):
            cluster_words[i] = []
        for j, k in enumerate(labels):
            cluster_words[k].append(vocab[j])
        return cluster_words

    def corect_ans_gen(self, word, sent_ret):
        leskobj = Lesk(word, sent_ret)
        return leskobj.simplified_lesk(True)

    def sent_retrieval(self, word, sent_tokenize_list):
        for j in sent_tokenize_list:
            if re.search(word, j, re.IGNORECASE):
                return (j)

    def wrong_ans_gen(self, input_word, sent_ret, correct_syns, word_list):
        x=[]
        r=[]
        # for i in sent_ret:
        not_list = [input_word]
        wrong = []
        word = input_word
        try:
            antonym = correct_syns[0].antonyms()
            wrong += [j.definition() for j in antonym if j != []]
        except:
            wrong += [k.synset().definition() for k in correct_syns[0].lemmas()[0].antonyms()]
        word_syn = wordnet.synsets(input_word)
        try:
            syn_wrong = [j.definition() for j in word_syn if
                         not wordnet.wup_similarity(correct_syns[0], j) or wordnet.wup_similarity(correct_syns[0],
                                                                                                  j) < 0.25]
        except:
            x.append(word_syn)
            r.append(correct_syns)
            return
        wrong += syn_wrong
        if len(wrong) >= 3:
            wrong = wrong[:3]
        else:
            len_wrong = len(wrong)
            while word in not_list or len_wrong < 3:
                word1 = list(word_list.keys())
                word = word1[random.randint(0, len(word1) - 1)]
                if len_wrong < 3:
                    if word not in not_list:
                        wrong.append(word)
                        not_list.append(word)
                len_wrong = len(wrong)
            wrong.append(word)
        # wrong_answer[input_word]=wrong
        return wrong

    def word_gen(self):
        clusters = self.clustering(0)
        word_list = self.preprocessing()
        sent_tokenize_list = sent_tokenize(self.text)
        word_selection = []
        data = {}
        random.seed(10)
        for i in clusters:
            data[i] = {}
            count = 0
            ind = random.randint(0, (len(clusters[i]) - 1))
            while count != 5:
                sent_ret = self.sent_retrieval(clusters[i][ind], sent_tokenize_list)
                correct_syns = self.corect_ans_gen(clusters[i][ind], sent_ret)
                if correct_syns:
                    wrong_ans = self.wrong_ans_gen(clusters[i][ind], sent_ret, correct_syns, word_list)
                else:
                    wrong_ans = []
                if wrong_ans != [] and correct_syns != []:
                    word_selection.append(clusters[i][ind])
                    data[i][(clusters[i][ind], sent_ret)] = {}
                    data[i][(clusters[i][ind], sent_ret)]["correct"] = correct_syns[0].definition()
                    data[i][(clusters[i][ind], sent_ret)]["wrong"] = wrong_ans
                    count += 1
                word_selection.append(clusters[i][ind])
                c=0
                while clusters[i][ind] in word_selection or wordnet.synsets(clusters[i][ind]) == []:
                    ind = random.randint(0, (len(clusters[i]) - 1))
                    print(clusters[i][ind])
                    print(word_selection)
                    print("****")
                    if c == 50:
                        break
                    c+=1
        return data, clusters

    def final_text(self, correct, clusters):
        difficult_words = []
        for i in correct:
            if correct[i]/5.0 < 60.0:
                difficult_words.extend(clusters[int(i)])
        difficult_words = set(difficult_words)
        paragraph = self.text.split("\n")
        final = []
        for j in range(len(paragraph)):
            sent_tokenize_list = sent_tokenize(paragraph[j])
            for i in difficult_words:
                if i.lower() in paragraph[j].lower():
                    for k in range(len(sent_tokenize_list)):
                        if i.lower() in sent_tokenize_list[k].lower():
                            try:
                                meaning = self.corect_ans_gen(i, sent_tokenize_list[k])[0].definition()
                            except:
                                meaning = ""
                            sent_tokenize_list[k] = re.sub(" " + i + " ", " " + i + " (" + meaning + ") ",
                                                               sent_tokenize_list[k])
            new_para = " ".join(sent_tokenize_list)
            final.append(new_para)
        return "\n".join(final)

