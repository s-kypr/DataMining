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

    pony_mask = np.array(Image.open("../pinkyB.jpg"))
    wc = WordCloud(background_color="black", max_words=2000, mask=pony_mask, stopwords=STOPWORDS)

    #init dictionary with the five categories
    categoriesSet = set(datafile["Category"])
    categoriesDict = dict.fromkeys(categoriesSet,"")

    #Conditional Selection
    # business = datafile.ix[datafile["Category"]=="Business"]
    # print business["Content"].size

    #fill index with data from cv
    for index, row in datafile.iterrows():
        categoriesDict[row["Category"]] += str(row["Content"])

    for category, text in categoriesDict.iteritems():
        wc.generate(text)
        image = wc.to_image()
        image.save("../wordcloud/wordcloud_" + category + ".jpg")
    return





print "Creating wordclouds..."
start = time.clock();
wordcloud(pd.read_csv('../datasets/train_set.csv', sep='\t'))
end = time.clock();
print "Wordclouds created in "+ str(end - start)+" sec"