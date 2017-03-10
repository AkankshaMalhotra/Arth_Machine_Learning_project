from bs4 import BeautifulSoup
import re
import cPickle
from config import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import syllable
import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import random
from nltk.tokenize import sent_tokenize
import simplified_lesk
from nltk.corpus import wordnet

file_name=raw_input()
with open(file_name, "r") as g:
    text = g.read()

class Arth:
    def __init__(self,text):
        self.text=text

    def get_wordnet_pos(self,words):
        data=[]
        treebank_tag=nltk.pos_tag(words)
        for i in treebank_tag:
            if i[1].startswith('J'):
                v=wordnet.ADJ
            elif i[1].startswith('V'):
                v= wordnet.VERB
            elif i[1].startswith('N'):
                v=wordnet.NOUN
            elif i[1].startswith('R'):
                v= wordnet.ADV
            else:
                v= ''
            data.append((i[0],v))
        return data

    def preprocessing(self):
        '''
         Implementing preprocessing steps.
         1) Removed HTML tags
         2) Removed Punctuation, numbers and special characters
         3) Word Tokenization of the text
         4) Generates Pos_tags for the words
         5) Find Appropriate lemma for the words
         6) Find Unique Words in the text
         7) Removal of stopwords, i.e. words that aren't important
        :return: preprocessed list of words
        '''
        object = BeautifulSoup(self.text, "lxml")
        markup_removed_text=object.get_text()
        words_only_text = re.sub("[^a-zA-Z]", " ",markup_removed_text)
        words = [i.lower() for i in words_only_text.split()]
        word_postag_tuple = self.get_wordnet_pos(words)
        wordnet_lemmatizer = WordNetLemmatizer()
        lemma = []
        for word, pos_tag in word_postag_tuple:
            if pos_tag != "":
                lemma.append(wordnet_lemmatizer.lemmatize(word, pos=pos_tag))
            else:
                lemma.append(wordnet_lemmatizer.lemmatize(word))
        unique_words= list(set(lemma))
        removed_stopwords=[i for i in unique_words if i not in stopwords]
        return removed_stopwords

    def syllable_calculater(self,word_list):
        '''

        :param word_list:
        :return: Dictionary of syllables
        '''
        syllable_obj = syllable.Syllable(word_list)
        syllable_dict=syllable_obj.model_load()
        return syllable_dict

    def usage_calc(self,word_list):
        freq_dict={}
        with open("freq_dict.pickle","rb") as g:
            freq=cPickle.load(g)
        for word in word_list:
            if word in freq:
                freq_dict[word]=freq[word]
            else:
                freq_dict[word]=0
        return freq_dict

    def getfeatures_normalize(self):
        word_list=self.preprocessing()
        syll=self.syllable_calculater(word_list)
        frequency=self.usage_calc(word_list)
        feature=[]
        vocab={k:v for k,v in enumerate(word_list)}
        for i in xrange(len(word_list)):
            feature.append((frequency[word_list[i]],syll[word_list[i]]))
        zscor=stats.zscore(feature)
        return zscor,vocab

    def clustering(self,show):
        cluster_words={}
        zsc,vocab=self.getfeatures_normalize()
        db=DBSCAN()
        db.fit(zsc)
        labels = db.labels_
        for i in set(labels):
            cluster_words[i]=[]
        for j, k in enumerate(labels):
            cluster_words[k].append(vocab[j])
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        if show==1:
        # Black removed and is used for noise instead.
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            unique_labels = set(labels)
            colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
            for k, col in zip(unique_labels, colors):
                if k == -1: # Black used for noise.
                    col = 'k'
                class_member_mask = (labels == k)
                xy = zsc[class_member_mask & core_samples_mask]
                plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14, hold=True)
                xy = zsc[class_member_mask & ~core_samples_mask]
                plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6, hold=True)
            plt.show()
        return(cluster_words)

    def corect_ans_gen(self,word,sent_ret):
        leskobj=simplified_lesk.Lesk(word,sent_ret)
        return(leskobj.simplified_lesk(True))

    def sent_retrieval(self,word,sent_tokenize_list):
        for j in sent_tokenize_list:
            if re.search(word, j, re.IGNORECASE):
                return(j)

    def wrong_ans_gen(self,input_word,sent_ret, correct_syns, word_list):
        wrong_answer={}
        # for i in sent_ret:
        not_list = [input_word]
        wrong=[]
        word=input_word
        try:
            antonym=correct_syns[0].antonyms()
            wrong += [j.definition() for j in antonym if j!=[]]
        except:
            wrong+=correct_syns[0].lemmas()[0].antonyms()
        word_syn=wordnet.synsets(input_word)
        syn_wrong=[j for j in word_syn if wordnet.wup_similarity(correct_syns[0],j)<0.25]
        wrong+=syn_wrong
        if len(wrong)>=3:
            wrong=wrong[:3]
        else:
            len_wrong = len(wrong)
            while word in not_list or len_wrong<3:
                word=word_list[random.randint(0,len(word_list)-1)]
                if len_wrong<3:
                    if word not in not_list:
                        wrong.append(word)
                        not_list.append(word)
                len_wrong = len(wrong)
            wrong.append(word)
        # wrong_answer[input_word]=wrong
        return wrong

    def word_gen(self, word_list):
        clusters=self.clustering(0)
        sent_tokenize_list = sent_tokenize(self.text)
        word_selection=[]
        data={}
        for i in clusters:
            data[i]={}
            count=0
            ind = random.randint(0, (len(clusters[i]) - 1))
            while(count!=5):
                sent_ret = self.sent_retrieval(clusters[i][ind],sent_tokenize_list)
                correct_syns = self.corect_ans_gen(clusters[i][ind], sent_ret)
                if correct_syns!=[]:
                    wrong_ans = self.wrong_ans_gen(clusters[i][ind], sent_ret, correct_syns, word_list)
                else:
                    wrong_ans=[]
                if wrong_ans != [] and correct_syns != []:
                    word_selection.append(clusters[i][ind])
                    data[i][clusters[i][ind]] = {}
                    data[i][clusters[i][ind]]["sentence"] = sent_ret
                    data[i][clusters[i][ind]]["correct"] = correct_syns
                    data[i][clusters[i][ind]]["wrong"] = wrong_ans
                    count += 1
                word_selection.append(clusters[i][ind])
                while clusters[i][ind] in word_selection or wordnet.synsets(clusters[i][ind])==[]:
                    ind = random.randint(0, (len(clusters[i]) - 1))
        return(data)




s=Arth(text)
word_list=s.preprocessing()
print s.word_gen(word_list)
