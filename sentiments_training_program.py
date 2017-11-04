import nltk
import random
from nltk.corpus import twitter_samples,stopwords
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
import pickle

from statistics import mode
import re
from nltk.tokenize import TweetTokenizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

class VoteClassifier(ClassifierI):
# ClassifierI is used ton make a user defined classifier class
	def __init__(self, *classifier):
		self._classifiers = classifier

	def classify(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)

		return mode(votes)

	def confidence(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)

		choice_votes = votes.count(mode(votes))
		conf = choice_votes/len(votes)
		return conf


tknz = TweetTokenizer()


def find_features(document):
	# words = set(document)
	document = re.sub('[?\'.]','',document)
	features = {}
	for w in tknz.tokenize(document):
		features[w.lower()] = (w.lower() in word_features)
	return features


positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

short_pos = open("positive.txt","r").read()
short_neg = open("negative.txt","r").read()

sentiment_tweets = []
all_words = []

allowed_word_types = ["J"]

for p in short_pos.split('\n'):
	sentiment_tweets.append((p,'positive'))
	words = word_tokenize(p)
	pos = nltk.pos_tag(words)
	for w in pos:
	    if w[1][0] in allowed_word_types:
	        all_words.append(w[0].lower())

    
for p in short_neg.split('\n'):
	sentiment_tweets.append((p,'negative'))
	words = word_tokenize(p)
	pos = nltk.pos_tag(words)
	for w in pos:
		if w[1][0] in allowed_word_types:
			all_words.append(w[0].lower())



positive_tweets_refined = []
negative_tweets_refined = []

for tweet in positive_tweets:
	tweet = str(tweet)
	tweet = re.sub(r'[^a-zA-z0-9:)(-;\s]',r'',tweet)
	positive_tweets_refined.append(tweet)

for tweet in negative_tweets:
	tweet = str(tweet)
	tweet = re.sub(r'[^a-zA-z0-9:)(-;\s]',r'',tweet)
	negative_tweets_refined.append(tweet)

positive_tweets = positive_tweets_refined
negative_tweets = negative_tweets_refined

for i in range(len(positive_tweets)):
	sentiment_tweets.append((positive_tweets[i],'positive'))

for i in range(len(negative_tweets)):
	sentiment_tweets.append((negative_tweets[i],'negetive'))

random.shuffle(sentiment_tweets)

stop_words = stopwords.words('english')

tweets = positive_tweets + negative_tweets

for tweet in tweets:
	for word in tknz.tokenize(tweet):
		if word.lower() not in stop_words and not word.startswith('https'):
			all_words.append(word.lower())


all_words = nltk.FreqDist(all_words)
all_words = (all_words.most_common(5000))
for word, freq in all_words:
	print(word.encode('utf-8'),freq)

word_features = [x[0] for x in all_words]

word_features1 = []

for word in word_features:
	if not word.startswith('https'):
		word_features1.append(word)

word_features = word_features1
word_features_doc = open('word_features.pkl','wb')
pickle.dump(word_features,word_features_doc)

feature_set = [(find_features(tweet), category) for (tweet, category) in sentiment_tweets]
# training_set = feature_set[:9000]
# testing_set = feature_set[9000:] 
# classifier.show_most_informative_features(25)

MultinomialNB_classifier = SklearnClassifier(MultinomialNB())
MultinomialNB_classifier.train(feature_set)
# print('MultinomialNB classifier accuracy',nltk.classify.accuracy(MultinomialNB_classifier, testing_set))
mnb_classifier_doc = open('mnb_classifier.pkl','wb')
pickle.dump(MultinomialNB_classifier,mnb_classifier_doc)
mnb_classifier_doc.close()

# BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
# BernoulliNB_classifier.train(feature_set)
# print('BernoulliNB_classifier accuracy',nltk.classify.accuracy(BernoulliNB_classifier, testing_set))
# bernoulli_classifier_doc = open('bernoulli_classifier.pkl','wb')
# pickle.dump(BernoulliNB_classifier,bernoulli_classifier_doc)
# bernoulli_classifier_doc.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(feature_set)
# print('LogisticRegression classifier accuracy',nltk.classify.accuracy(LogisticRegression_classifier, testing_set))
LogisticRegression_classifier_doc = open('LogisticRegression.pkl','wb')
pickle.dump(LogisticRegression_classifier,LogisticRegression_classifier_doc)
LogisticRegression_classifier_doc.close()

SGDclassifier = SklearnClassifier(SGDClassifier())
SGDclassifier.train(feature_set)
# print('SGDClassifier classifier accuracy',nltk.classify.accuracy(SGDClassifier_classifier, testing_set))
SGDClassifier_doc = open('SGDClassifier.pkl','wb')
pickle.dump(SGDclassifier,SGDClassifier_doc)
SGDClassifier_doc.close()

# SVC_classifier = SklearnClassifier(SVC())
# SVC_classifier.train(feature_set)
# print('SVC classifier accuracy',nltk.classify.accuracy(SVC_classifier, testing_set))
# SVC_classifier_doc = open('SVCClassifier.pkl','wb')
# pickle.dump(SVC_classifier,SVC_classifier_doc)
# SVC_classifier_doc.close()

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(feature_set)
# print('NuSVC classifier accuracy',nltk.classify.accuracy(NuSVC_classifier, testing_set))
NuSVC_classifier_doc = open('NuSVC_classifier.pkl','wb')
pickle.dump(NuSVC_classifier,NuSVC_classifier_doc)
NuSVC_classifier_doc.close()

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(feature_set)
# print('LinearSVC classifier accuracy',nltk.classify.accuracy(LinearSVC_classifier, training_set))
LinearSVC_classifier_doc = open('LinearSVC_classifier.pkl','wb')
pickle.dump(LinearSVC_classifier,LinearSVC_classifier_doc)
LinearSVC_classifier_doc.close()

# save_classifier = open('NaiveBayesClassifier.pickle', 'wb')
# pickle.dump(classifier, save_classifier)
# save_classifier.close()

# with open('LinearSVC_classifier.pkl','rb') as fp:
# 	LinearSVC_classifier = pickle.load(fp)
	
# with open('NuSVC_classifier.pkl','rb') as fp:
# 	NuSVC_classifier = pickle.load(fp)
	
# with open('SGDClassifier.pkl','rb') as fp:
# 	SGDclassifier = pickle.load(fp)

# with open('mnb_classifier.pkl','rb') as fp:
# 	MultinomialNB_classifier = pickle.load(fp)

# with open('LogisticRegression.pkl','rb') as fp:
# 	LogisticRegression_classifier = pickle.load(fp)

voted_classifier = VoteClassifier(LinearSVC_classifier,
								  NuSVC_classifier,
								  SGDclassifier,
								  MultinomialNB_classifier,
								  LogisticRegression_classifier)

def sentiment(document):
	feats = find_features(document)
	return voted_classifier.classify(feats),voted_classifier.confidence(feats)

