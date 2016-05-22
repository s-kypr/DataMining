__author__ = 'sofia'

from PIL import Image
import time
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd

import numpy as np

from wordcloud import WordCloud, STOPWORDS


def wordcloud(datafile):

    #remove stop words, the most common words in a language
    vectorizer=CountVectorizer(stop_words='english')

    for word in vectorizer.get_stop_words():
        STOPWORDS.add(word)
    STOPWORDS.add("said")

    pony_mask = np.array(Image.open("../../pinkyB.jpg"))
    wc = WordCloud(background_color="black", max_words=2000, mask=pony_mask,
               stopwords=STOPWORDS)         #maybe width, height??
    
    # generate word cloud
    #wc.generate(text)
    # store to file

    #init dictionary with the five categories
    categoriesSet = set(datafile["Category"])
    categoriesDict = dict.fromkeys(categoriesSet,"")
    print categoriesDict

    #Conditional Selection




    business = datafile.ix[datafile["Category"]=="Business"]

    print business["Content"].size

    # 2735

    for index, row in datafile.iterrows():
        categoriesDict[row["Category"]] += str(row["Content"])

    for category, words in categoriesDict.iteritems():
        wc.generate(words)
        image = wc.to_image()
        image.save("../wordcloud/wordcloud_" + category + ".jpg")
    return





print "Creating wordclouds..."
start = time.clock();
wordcloud(pd.read_csv('../datasets/train_set.csv', sep='\t'))
end = time.clock();
print "Wordclouds created in "+ str(end - start)+" sec"