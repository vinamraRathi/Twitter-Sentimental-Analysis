import numpy as np
import pandas as pd
import gensim
from tqdm import tqdm
from gensim.models.doc2vec import TaggedDocument
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


train_data = pd.read_csv("/home/vinu/PycharmProjects/TwitterSentimentalAnalysis/venv/input/train_E6oV3lV.CSV")
test_data = pd.read_csv("/home/vinu/PycharmProjects/TwitterSentimentalAnalysis/venv/input/test_tweets_anuFYb8.csv")

#cleaning data

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)

    return input_txt

# remove twitter handles

train_data['tidy_tweet'] = np.vectorize(remove_pattern)(train_data['tweet'], "@[\w]*")
test_data['tidy_tweet'] = np.vectorize(remove_pattern)(test_data['tweet'], "@[\w]*")

# remove special characters, numbers, punctuations
train_data['tidy_tweet'] = train_data['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
test_data['tidy_tweet'] = test_data['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")

# remove short words
train_data['tidy_tweet'] = train_data['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
test_data['tidy_tweet'] = test_data['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

"""
train_data['label'].value_counts().plot.bar(color = 'red', figsize = (6, 4))
plt.title('Sentiment Analysis')
plt.xlabel('Sentiments')
plt.ylabel('Count')
plt.show()

length_train = train_data['tweet'].str.len().plot.hist(color = 'pink', figsize = (6, 4))
length_test = test_data['tweet'].str.len().plot.hist(color = 'orange', figsize = (6, 4))


plt.title('Comparison of distribution of tweets in the train and test sets')
plt.xlabel('lenght')
plt.ylabel('count')
plt.legend()
plt.show()
"""

# making a count vectorizer to keep record of which word occur the most
# making an object for the count vectorizer
cv = CountVectorizer(stop_words = 'english', max_features= 2500)
words = cv.fit_transform(train_data.tweet)

#tokenizing the words present in the training set
tokenized_tweet = train_data['tidy_tweet'].apply(lambda x: x.split())
#creating a word to vector model
model_w2v = gensim.models.Word2Vec(tokenized_tweet,size = 200, window=5,min_count=2, sg=1,hs=0,negative = 10, workers=2,seed=34)
model_w2v.train(tokenized_tweet, total_examples=len(train_data['tidy_tweet']), epochs=20)

tqdm.pandas(desc="progress-bar") #to show progress of each lable

#labeling the tweets
def add_label(twt):
    output = []
    for i,s in zip(twt.index, twt):
        output.append(TaggedDocument(s, ["tweet_" + str(i)]))
    return output

labeled_tweets = add_label(tokenized_tweet)

#making a copus of cleaned and processed words with their counts
#for the training dataset

train_corpus = []

for i in range(0, 31962):
    review = re.sub('[^a-zA-Z]',' ',train_data['tidy_tweet'][i])
    review = review.lower()
    review = review.split()

    ps = PorterStemmer()

    # stemming
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]


    # Joining them back with space
    review = ' '.join(review)
    train_corpus.append(review)

# for the test data set

test_corpus = []

for i in range(0, 17197):
    review = re.sub('[^a-zA-Z]',' ',test_data['tidy_tweet'][i])
    review = review.lower()
    review = review.split()

    ps = PorterStemmer()

    # stemming
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]


    # Joining them back with space
    review = ' '.join(review)
    test_corpus.append(review)

# creating bag of words
# for the train corpus

x = cv.fit_transform(train_corpus).toarray()
y = train_data.iloc[:, 1]

# for the test corptoarrayus

x_test = cv.fit_transform(test_corpus).toarray()

# splitting the training data into train and valid sets

x_train, x_valid, y_train, y_valid = train_test_split(x,y, test_size=0.25, random_state= 42)

# Standardization

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_valid = sc.fit_transform(x_valid)
x_test = sc.fit_transform(x_test)

# MODELLING
# CLASSIFICATION OF NEG AND POS TWEETS
# USING RANDOM FOREST CLASSIFIER

#model = RandomForestClassifier()
#model.fit(x_train, y_train)

#y_pred = model.predict(x_valid)

#print("Training Accuracy :", model.score(x_train, y_train))
#print("Validation Accuracy :", model.score(x_valid, y_valid))

#print("F1 score :", f1_score(y_valid, y_pred))

# confusion matrix
#cm = confusion_matrix(y_valid, y_pred)
#print(cm)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_valid)

print("Training Accuracy :", model.score(x_train, y_train))
print("Validation Accuracy :", model.score(x_valid, y_valid))

# calculating the f1 score for the validation set
print("f1 score :", f1_score(y_valid, y_pred))

# confusion matrix
cm = confusion_matrix(y_valid, y_pred)
print(cm)
