import sentiments_training_program as s
import json

from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener

api_key = '2hIyn2HcjHm0sM4WQ6MJMkuLi'
api_secret = 'gZtTPBqRc5wIg7JnbKigNyLxTZUI4guOC4MgXq2R47ndHk54Hz'
access_token = '926503251385532416-LNXZXaUXdgF5kHRDfxBy3xVpUxCBmys'
access_secret = 'wKxeZlFMER0FLBR9smxGH7VkL75rs4TIsVMe6GZaE7e0p'
print(s.sentiment('The movie was horrible I will never watch such movie in my life'))
print(s.sentiment('The best movie of the series,would definitely recommend it FIVE stars'))
print(s.sentiment('very good movie,one of the best of all time'))
# sentiments = []
outpu_file = open('output_sentiments_for_ThorRagnorok.txt','wb')
class Listener(StreamListener):

	def on_data(self,data):
		data = json.loads(data)
		tweet = data['text'] 
		sentiment_value,confidence = s.sentiment(tweet)
		# sentiments.append(sentiment_value)
		print(tweet.encode('utf-8'),sentiment_value,confidence)

		return True

	def on_error(self,status):
		print(status)

auth = OAuthHandler(api_key,api_secret)
auth.set_access_token(access_token,access_secret)

twitterStream = Stream(auth,Listener())
for x in range(100):
	twitterStream.filter(track=['Justice League'])
# print(sentiments)