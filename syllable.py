import re
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
import random
class Syllable():

    def __init__(self,data):
        self.data=data

    def syllables_features(self,word):
        feature = []
        feature.append(len(word))
        feature.append(len(re.findall(r'[bcdfghjklmnpqrstvwxz]+', word)))
        feature.append(len(re.findall(r'[aeiou]', word, flags=re.IGNORECASE)))
        feature.append(self.sylco(word))
        feature.append(len(re.findall(r'[eaoui][eaoui]', word)) + len(re.findall(r'[eaoui][eaoui][eaoui]', word)))
        if word[-1:] == "e":
            if word[-2:] == "le":
                feature.append(1)
            else:
                feature.append(0)
        else:
            feature.append(1)
        feature_vector = np.array(feature)
        return feature_vector

    def sylco(self,word):
        #I have not authored this function, taken from a blog by the person
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

    def build_features(self):
        random.shuffle(self.data)
        feature_vector = np.empty([len(self.data), 6])
        label = np.empty([len(self.data)])
        for i in xrange(len(self.data)):
            feature_vector[i] = self.syllables_features(self.data[i][0])
            label[i] = self.data[i][1]
        return(feature_vector,label)

    def check_accuracy(self, feature_vector, label):
        train_feature=feature_vector[:int(len(feature_vector)*0.80)]
        train_label=label[:int(len(feature_vector)*0.80)]
        test_feature=feature_vector[int(len(feature_vector)*0.80):]
        test_label=label[int(len(feature_vector)*0.80):]
        model=self.train_model(train_feature,train_label)
        accuracy=model.score(test_feature,test_label)
        return accuracy

    def train_model(self, train_features, train_labels):
        clf = GaussianNB()
        clf.fit(train_features, train_labels)
        return clf

    def predict(self, model,word):
        features=self.syllables_features(word)
        features = features.reshape(1, -1)
        return(model.predict(features)[0])

    def save_model(self):
        features, labels=self.build_features()
        accuracy=self.check_accuracy(features,labels)
        model=self.train_model(features,labels)
        joblib.dump(model, 'syllable.pkl')
        return(accuracy)

    def model_load(self):
        print self.data
        model=joblib.load('syllable.pkl')
        syl_dict={word:self.predict(model, word) for word in self.data}
        return(syl_dict)





