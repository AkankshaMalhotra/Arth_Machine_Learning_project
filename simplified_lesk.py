from nltk.corpus import wordnet
from nltk.corpus import stopwords
import re

class Lesk():
    def __init__(self, word, sentence):
        self.word=word
        self.sentence=sentence

    def remove_stopwords(self,words):
        stop=stopwords.words('english')
        return [word for word in words if word not in stop]

    def compute_overlap_words(self,signature,context):
        overlap=0
        overlap_words=[]
        for word in context:
            count=signature.count(word)
            if(count>0):
                overlap_words.append(word)
                overlap+=1
        return (overlap,overlap_words)

    def simplified_LESK(self, Flag_stopwords=False):
        syns=wordnet.synsets(self.word)
        best_sense=syns[0]
        max_overlap=0
        context=self.sentence.split()
        if(Flag_stopwords):
            context=self.remove_stopwords(context)
            overlaps={}
            for sense in syns:
                signature=sense.definition().split()
                examples=sense.examples()
                for example in examples:
                    signature+=example.split()
            if(Flag_stopwords):
                signature=self.remove_stopwords(signature)
            overlap,overlap_words=self.compute_overlap_words(signature,context)
            overlaps[sense]=(overlap,overlap_words)
            if overlap>max_overlap:
                max_overlap=overlap
                best_sense=sense
        return (best_sense,max_overlap,overlaps)