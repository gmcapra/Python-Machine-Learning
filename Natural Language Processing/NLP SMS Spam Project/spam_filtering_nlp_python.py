"""
--------------------------------------------------------------------------------------------
Using Natural Language Processing with Python to Filter SMS Spam Messages
--------------------------------------------------------------------------------------------
Gianluca Capraro, Python for Data Science and Machine Learning Project
Created: May 2019
--------------------------------------------------------------------------------------------
In this project, we will use Python with the NLTK library to build a spam detection
filter. This project makes use of the SMSSpamCollection.csv file that can be found in the 
smsspamcollection folder. The data is provided for free use by the UCI Machine Learning
Repository that can be found here:

https://archive.ics.uci.edu/ml/index.php

Please refer to the readme document within the folder for specific information and
appropriate citations regarding this data set.
--------------------------------------------------------------------------------------------
"""

#import natural language processing library
import nltk

"""
--------------------------------------------------------------------------------------------
IF FIRST TIME RUNNING SCRIPT: Set up the NLTK Library
--------------------------------------------------------------------------------------------
This script requires downloading the 'stopwords' package from nltk.
The first time running, when prompted with the downloader:
	- Type 'd'
	- Then type 'stopwords'
	- This should install the necessary package
	- Type 'l' when first prompted for a list of all packages available with nltk
"""
#-------------------------------------------------------------------------------------------
#UNCOMMENT BELOW LINE FOR NLTK DOWNLOADER
#-------------------------------------------------------------------------------------------
#nltk.download_shell()


"""
--------------------------------------------------------------------------------------------
Set up the DataFrame of SMS Messages
- Based on the .csv file, separate the messages at the first tab
- Create columns for the dataframe, include a spam label column, and a column for messages
--------------------------------------------------------------------------------------------
"""
#import pandas for data manipulation
import pandas as pd

#format pd for desired terminal output
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)

#create the sms message dataframe
sms_data = pd.read_csv('smsspamcollection/SMSSpamCollection.csv', sep='\t', names = ['label','message'])

#show the head of our new dataframe
print('\nSMS Data Head:')
print(sms_data.head())
print('\n')

#get the description of the sms data
print('SMS Data Description:')
print(sms_data.describe())
print('\n')

#let's get more specific information, a data description for spam and ham messages
#get the description of the sms data, segmented by message label column
print('SMS Data Description (Spam vs. Ham):')
print(sms_data.groupby('label').describe())
print('\n')

#to better understand the sms messages and get additional data, add message length column
sms_data['length'] = sms_data['message'].apply(len)


"""
--------------------------------------------------------------------------------------------
Exploratory Data Analysis
--------------------------------------------------------------------------------------------
"""
#import data visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt

# using new column of data, get an idea of message length distribution 
# the plot shows signs of a bimodal distribution, which could play into spam determination later on
print('Showing Distribution of Message Length...')
sns.distplot(sms_data['length'], kde = False, bins = 100)
plt.title('Distribution of Message Length (# Characters)')
plt.show()
print('\n')

# let's see if the distribution is significantly different based on spam designation
# spam message length around 150 characters
# ham message length around 50 characters
print('Showing Distribution of Message Length (Spam vs. Ham)...')
sms_data.hist(column = 'length', by = 'label', bins = 100)
plt.xlabel('Message Length (# Characters)')
plt.show()
print('\n')


"""
--------------------------------------------------------------------------------------------
Text Cleaning and Normalization
--------------------------------------------------------------------------------------------
- Prior to being able to use our message data, it needs to be cleaned and normalized
- In this project, we will only perform simple puncuation and stopwords removal
- Additional text normalization methods are available with nltk and will be revisited
	in future projects
--------------------------------------------------------------------------------------------
"""
#import the stopwords list from nltk
from nltk.corpus import stopwords

#import the string library to help manipulate our data
import string

#create function to remove any punctuation or stopwords from a given string
def clean_text(message):
	"""
	- takes in a string message
	- removes punctuation and stopwords
	- returns 'cleaned' message text

	"""
	nopunctuation = [char for char in message if char not in string.punctuation]
	nopunctuation = ' '.join(nopunctuation)
	return [word for word in nopunctuation.split() if word.lower() not in stopwords.words('english')]

#tokenize the message (converting a normal text string into the desired 'clean' version)
sms_data['message'].apply(clean_text)


"""
--------------------------------------------------------------------------------------------
Text Vectorization
--------------------------------------------------------------------------------------------
- Currently we have the messages as lists of tokens
- Need to be transformed into vectors that machine learning models can understand
- To do this, we will use the 'bag of words' model
	- Count how many times a word occurs in each message (term frequency)
	- Weight the count so that frequent tokens get lower weight (inverse document frequency)
	- Normalize vectors to unit length (L2 normalization)
- Each message vector will then have as many dimensions as there are unique words in the corpus

- Use SciKit Learn's CountVectorizer to convert collection of text documents to a matrix of token counts
	- Results in a 2D matrix where the columns represent each message and the rows represent each count of every word in the corpus
	- Will result in many cells with 0 values

- Weigh and Normalize the messages with TF-IDF
	- term-frequency, inverse document frequency

- Now, sms_tfidf represents the message data in vector form
- This data can be used to train the spam filter!
--------------------------------------------------------------------------------------------
"""
#import the countvectorizer from scikit learn
from sklearn.feature_extraction.text import CountVectorizer

# create the bag of words transformer as an instance of count vectorizer
# this line can take a little bit of time to process
bow_transformer = CountVectorizer(analyzer=clean_text).fit(sms_data['message'])

#transform our sms data using BOW
sms_bow = bow_transformer.transform(sms_data['message'])

#normalize using TF-IDF
#import the TF IDF Transformer from SciKit Learn
from sklearn.feature_extraction.text import TfidfTransformer

#create instance of TF IDF transformer
tfidf_transformer = TfidfTransformer().fit(sms_bow)

#transform the BOW corpus into TD-IDF corpus
sms_tfidf = tfidf_transformer.transform(sms_bow)


"""
--------------------------------------------------------------------------------------------
Spam Detection Training
--------------------------------------------------------------------------------------------
- For this project, we will classify text using the Naive Bayes algorithm
	This represents one of the best-known email spam filtering algorithms
- Now that all of our data is cleaned and ready to be used, this model can be trained
	fairly simply
--------------------------------------------------------------------------------------------
"""

#import the Naive Bayes algorithm from SKLearn
from sklearn.naive_bayes import MultinomialNB

#create instance of the model using the NB algorithm
spam_filter = MultinomialNB().fit(sms_tfidf, sms_data['label'])


"""
--------------------------------------------------------------------------------------------
Making a Spam Prediction with our Model
--------------------------------------------------------------------------------------------
"""
print('First attempt at Spam Filtering for First Message in Dataset:')
print('Predicted Label: ', spam_filter.predict(sms_tfidf)[0])
print('Correct Label: ', sms_data.label[0])
print('\n')


"""
--------------------------------------------------------------------------------------------
Model Evaluation
--------------------------------------------------------------------------------------------
Now that we have seen our model make a spam or ham prediction, we could grab predictions
for the entire data set and use SciKit Learn to evaluate its predictive performance vs.
the label information we know to be correct.

However, we will not evaluate the model accuracy based on the same data used for training.
This type of evaluation does not indicate the true predictive capabilities of the model.
To properly evaluate the model, we will use the Train Test Split method to split our data
into separate testing and training sets.
--------------------------------------------------------------------------------------------
"""

#import train_test_split
from sklearn.model_selection import train_test_split

#use train test split to split data
#use 20% of the data for testing set, and the rest for training
message_train, message_test, label_train, label_test = train_test_split(sms_data['message'], sms_data['label'], test_size = 0.2)

#use SKLearn's Pipeline to store workflow
#this allows us to set up all transformations to data for us to use in the future
#in this case, our steps are BOW, TF-IDF, Naive Bayes Classification
#now we can pass message text data directly into the pipeline and it will automatically be cleaned and processed!
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
		('bow', CountVectorizer(analyzer=clean_text)),
		('tfidf', TfidfTransformer()),
		('classifier', MultinomialNB())

	])

#fit the training data to the pipeline
#notice how we no longer need to do each step for message data!
pipeline.fit(message_train, label_train)

#get all predictions from our test data based on the data trained
predictions = pipeline.predict(message_test)

#print the classification report for our model
#compare our predictions to the known testing labels
from sklearn.metrics import classification_report
print('Classification Report for Spam Filter:')
print(classification_report(label_test, predictions))
print('\n')


"""
The classification report demonstrates a fairly good ability to filter messages
into spam or ham categories. However, there is clearly some room for improvement.
Going forward, additional cleaning and text normalization could be performed, 
and more NLP features could be identified in this data set to refine
predictive capabilities.

NOTE: The above pipeline takes in data and transforms it using the given functions.
Try experimenting with different combinations. For example, we could change our
model to make use of the random forest classifier instead of Naive Bayes.

See what happens:
- First, above pipeline declaration: from sklearn.ensemble import RandomForestClassifier
- Then, replace MultinomialNB() with RandomForestClassifier()
"""



"""
Revisit for Additional Resources - Introductions to NLP:
http://www.nltk.org/book/
https://www.kaggle.com/c/word2vec-nlp-tutorial
https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
"""


