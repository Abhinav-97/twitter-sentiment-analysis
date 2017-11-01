import nltk
import random
from nltk.corpus import twitter_samples,stopwords
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
	features = {}
	for w in tknz.tokenize(document):
		features[w.lower()] = (w.lower() in word_features)
	return features


positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')


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

sentiment_tweets = []
for i in range(len(positive_tweets)):
	sentiment_tweets.append((positive_tweets[i],'positive'))

for i in range(len(negative_tweets)):
	sentiment_tweets.append((negative_tweets[i],'negetive'))

random.shuffle(sentiment_tweets)

stop_words = stopwords.words('english')

all_words = []

tweets = positive_tweets + negative_tweets

for tweet in tweets:
	for word in tknz.tokenize(tweet):
		if word.lower() not in stop_words and not word.startswith('https'):
			all_words.append(word.lower())


all_words = nltk.FreqDist(all_words)
all_words = (all_words.most_common(3500))
for word, freq in all_words:
	print(word.encode('utf-8'),freq)

word_features = [x[0] for x in all_words]

word_features1 = []

for word in word_features:
	if not word.startswith('https'):
		word_features1.append(word)

word_features = word_features1
	

print(find_features(sentiment_tweets[1][0]))
feature_set = [(find_features(tweet), category) for (tweet, category) in sentiment_tweets]
training_set = feature_set[:9000]
testing_set = feature_set[9000:] 
# classifier.show_most_informative_features(25)

MultinomialNB_classifier = SklearnClassifier(MultinomialNB())
MultinomialNB_classifier.train(feature_set)
# print('MultinomialNB classifier accuracy',nltk.classify.accuracy(MultinomialNB_classifier, testing_set))
mnb_classifier_doc = open('mnb_classifier.pkl','wb')
pickle.dump(MultinomialNB_classifier,mnb_classifier_doc)
mnb_classifier_doc.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(feature_set)
print('BernoulliNB_classifier accuracy',nltk.classify.accuracy(BernoulliNB_classifier, testing_set))
bernoulli_classifier_doc = open('bernoulli_classifier.pkl','wb')
pickle.dump(BernoulliNB_classifier,bernoulli_classifier_doc)
bernoulli_classifier_doc.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(feature_set)
# print('LogisticRegression classifier accuracy',nltk.classify.accuracy(LogisticRegression_classifier, testing_set))
LogisticRegression_classifier_doc = open('LogisticRegression.pkl','wb')
pickle.dump(LogisticRegression_classifier,LogisticRegression_classifier_doc)
LogisticRegression_classifier_doc.close()

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(feature_set)
# print('SGDClassifier classifier accuracy',nltk.classify.accuracy(SGDClassifier_classifier, testing_set))
SGDClassifier_classifier_doc = open('SGDClassifier.pkl','wb')
pickle.dump(SGDClassifier_classifier,SGDClassifier_classifier_doc)
SGDClassifier_classifier_doc.close()

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(feature_set)
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

voted_classifier = VoteClassifier(LinearSVC_classifier,
								  NuSVC_classifier,
								  SGDClassifier_classifier,
								  MultinomialNB_classifier,
								  LogisticRegression_classifier)

# print("voted classifier accuracy percent", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

# print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:",voted_classifier.confidence(testing_set[0][0])*100)
# print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %:",voted_classifier.confidence(testing_set[1][0])*100)
# print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %:",voted_classifier.confidence(testing_set[2][0])*100)
# print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %:",voted_classifier.confidence(testing_set[3][0])*100)
# print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %:",voted_classifier.confidence(testing_set[4][0])*100)
# print("Classification:", voted_classifier.classify(testing_set[5][0]), "Confidence %:",voted_classifier.confidence(testing_set[5][0])*100)