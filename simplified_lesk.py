from nltk.corpus import wordnet
from nltk.corpus import stopwords
import nltk


class Lesk():
    def __init__(self, word, sentence):
        self.word=word
        self.sentence=sentence

    def remove_stopwords(self,words):
        stop=stopwords.words('english')
        return [word for word in words if word not in stop]

    def overlap_words(self,signature,context):
        overlap=0
        over_words=[]
        for word in context:
            count=signature.count(word)
            if(count>0):
                over_words.append(word)
                overlap+=1
        return (overlap,over_words)

    def get_wordnet_pos(self,words):
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

    def simplified_lesk(self, Flag_stopwords=False):
        max_overlap=0
        signature=[]
        syns=wordnet.synsets(self.word)
        try:
            sense=syns[0]
            context=self.sentence.split()
        except:
            return ([])
        pos=self.get_wordnet_pos(context)
        if (pos):
            syns=[s for s in syns if str(s.pos())==pos]
        if(Flag_stopwords):
            context=self.remove_stopwords(context)
            overlaps={}
            for sen in syns:
                signature+=sen.definition().split()
                examples=sen.examples()
                for example in examples:
                    signature+=example.split()
                if(Flag_stopwords):
                    signature=self.remove_stopwords(signature)
                overlap,over_words=self.compute_overlap_words(signature,context)
                overlaps[sen]=(overlap,over_words)
                if overlap>max_overlap:
                    max_overlap=overlap
                    sense=sen
        return (sense,max_overlap,overlaps)