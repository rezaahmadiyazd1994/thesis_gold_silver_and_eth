from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
import pandas_datareader.data as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense,Dropout
import requests
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from bs4 import BeautifulSoup
import tensorflow as tf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from keras.models import model_from_json



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------News Analyeis-------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

#vader
 
# function to print sentiments
# of the sentence.

count_positive = 0
count_negative = 0

def sentiment_scores(sentence):
    global count_negative, count_positive
 
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()
 
    # polarity_scores method of SentimentIntensityAnalyzer
    # object gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(sentence)

    # decide sentiment as positive, negative and neutral
    if(sentiment_dict['compound'] >= 0.05):
	    count_positive = count_positive + 1 
    elif(sentiment_dict['compound'] <= - 0.05):
 	    count_negative = count_negative + 1
          
def clean_tweet(tweet):
	return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


# Python program to convert a list
# to string using join() function
# Function to convert

def listToString(s):

     # initialize an empty string
     str1 = " "

     # return string
     return (str1.join(s))

     # Driver code
     s = filtered_sentence
     print(listToString(s))
          

# get day
today = datetime.now()
day = today.strftime("%d")

day = str(day)
datet = str(today.year)
datem = str(today.month)
timeh = str(today.hour)
timem = str(today.minute)

# web scraping
count_neutral = 0

# get news and tweet
urls = [
'https://www.tradingview.com/symbols/XAUUSD/?exchange=OANDA'
]

stop_words = set(stopwords.words('english'))

for url in urls:
     response = requests.get(url)
     soup = BeautifulSoup(response.content, 'html.parser')
     headlines = soup.find_all("article",class_="card-exterior-Us1ZHpvJ")
     for headline in headlines:
          headline = str(headline)
          word_tokens = word_tokenize(headline)
          filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
          filtered_sentence = []  
          for w in word_tokens:
               if w not in stop_words:
                    filtered_sentence.append(w)
               lemma_word = []
               wordnet_lemmatizer = WordNetLemmatizer()
               for w in filtered_sentence:
                    word1 = wordnet_lemmatizer.lemmatize(w, pos = "n")
                    word2 = wordnet_lemmatizer.lemmatize(word1, pos = "v")
                    word3 = wordnet_lemmatizer.lemmatize(word2, pos = ("a"))
                    lemma_word.append(word3)
          tweet = str(filtered_sentence)
          tweet = BeautifulSoup(tweet, "lxml").text
          sentiment_scores(clean_tweet(tweet)) 

buy_counter = 0
sell_counter = 0


count_all = count_positive + count_negative + 1
percent_positive = (count_positive * 100 / count_all)
percent_negative = (count_negative * 100 / count_all)

signal_news = "Neutral"
p_count = str()
if(percent_positive >= 50):
     buy_counter = buy_counter + 1
     pn = 1
     pns = "Positive"
     signal_news = "Positive"
     p_count = round(percent_positive, 2)
     p_count = str(p_count)
elif(percent_negative > 50):
     sell_counter = sell_counter + 1
     pn = -1
     pns = "Positive"
     signal_news = "Negative"
     p_count = round(percent_negative, 2)
     p_count = str(p_count)
else:
     pn = 0
     pns = "Neutral"
     signal_news = "Neutral"


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------Data Analyeis-------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

# get day
today = datetime.now()
day = today.strftime("%d")

day = str(day)
datet = str(today.year)
datem = str(today.month)
timeh = str(today.hour)
timem = str(today.minute)

# Directory
directory = day+"-"+datem+"-"+datet
  
# Parent Directory path
parent_dir = "date"
  
# Path
path = os.path.join(parent_dir, directory)

#  Data
data = web.DataReader("xauusd", "av-daily", start=datetime(2020, 9, 9), end=datetime.now(), api_key='8WA3JKMUXS09BZKJ')

# Prepare Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['close'].values.reshape(-1,1))

prediction_day = 60

x_train = []
y_train = []

for x in range(prediction_day, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_day:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# Create Artificial Neural Network
if(os.path.isdir(path)): 
    # load json and create model
    json_file = open(path+'\model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(path+"\model.h5")
else:
    model = Sequential()
    model.add(LSTM(units=60,return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(Dropout(0.1))
    model.add(LSTM(units=60,return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(units=60))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    model.fit(x_train, y_train, epochs=100, batch_size=32)

    os.mkdir(path)
    # serialize model to JSON
    model_json = model.to_json()
    with open("date/"+directory+"/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("date/"+directory+"/model.h5")
    print("Saved model to disk")

# Load Test Data
test_data = web.DataReader("xauusd", "av-daily", start=datetime(2020, 9, 9), end=datetime.now(), api_key='8WA3JKMUXS09BZKJ')
actusal_prices = test_data['close'].values

total_dataset = pd.concat((data['close'], test_data['close']), axis=0)
model_input = total_dataset[len(total_dataset) - len(test_data) - prediction_day:].values
model_input = model_input.reshape(-1,1)
model_input = scaler.transform(model_input)

# make predictons on test data
x_test = []
for x in range(prediction_day,len(model_input)):
    x_test.append(model_input[x-prediction_day:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1))

predicted_price = model.predict(x_test)
predicted_price = scaler.inverse_transform(predicted_price)


#predict next day
real_data = [model_input[len(model_input) + 1 - prediction_day:len(model_input + 1),0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data,(real_data.shape[0],real_data.shape[1],1))
prediction =  model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
prediction = float(prediction)
prediction = round(prediction, 2)
prediction = str(prediction)

# get stream live data
url = 'http://www.goldapi.io/api/XAU/USD'
headers = {'x-access-token': 'goldapi-187e5blrlgb9db7n-io'}
resp = requests.get(url, headers=headers)
resp_dict = resp.json()

now_price = resp_dict.get('price')
now_price = str(now_price)



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------Visualization Data-------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

                    

# Plt the Test Prediction
plt.plot(actusal_prices, color="Blue", label=f"Actual Gold Price")
plt.plot(predicted_price,  color="Red", label=f"Prediction Gold Price")
plt.title(f"The price now is: " + now_price + ",    The prediction last price: " +  prediction + ",    Gold News: " +  signal_news + " ("+p_count+"%)")
plt.xlabel("Days")
plt.ylabel(f"Price")
plt.legend()
plt.grid()
plt.show()