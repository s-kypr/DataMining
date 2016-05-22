__author__ = 'sofia'

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
#
# #Read Data
# df=pd.read_csv("train_set.csv",sep="\t")
# le = preprocessing.LabelEncoder()
# le.fit(df["Category"])
# Y_train=le.transform(df["Category"])
# X_train=df['Content']
# vectorizer=CountVectorizer(stop_words='english')
# transformer=TfidfTransformer()
# svd=TruncatedSVD(n_components=10, random_state=42)
# clf=SGDClassifier()
#
# pipeline = Pipeline([
#     ('vect', vectorizer),
#     ('tfidf', transformer),
#     ('svd',svd),
#     ('clf', clf)
# ])
# #Simple Pipeline Fit
# pipeline.fit(X_train,Y_train)
# #Predict the train set
# predicted=pipeline.predict(X_train)
# print(metrics.classification_report(df['Category'], le.inverse_transform(predicted)))

from sklearn.feature_extraction.text import *
import pandas as pd
from sklearn.decomposition import TruncatedSVD

vectorizer=TfidfVectorizer(stop_words='english')
df=pd.read_csv("datasets/train_set.csv",sep="\t")
X_train = vectorizer.fit_transform(df["Content"])
#X_train now ready to be used for training a classifier or for SVD
svd=TruncatedSVD(n_components=200, random_state=42)
X_lsi=svd.fit_transform(X_train)