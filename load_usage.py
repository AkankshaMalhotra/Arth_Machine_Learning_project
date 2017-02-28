import pandas as pd
import cPickle
usage = pd.read_excel("frequency.xlsx")
headers = usage.columns.values
freq_dict = {usage[headers[1]][i]: usage[headers[2]][i] for i in xrange(len(usage[headers[2]]))}

with open("freq_dict.pickle","wb") as g:
    cPickle.dump(freq_dict,g)
