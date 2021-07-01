# Sentiment-Analysis-of-Indian-Farmers-Protest-using-Twitter-Data

    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

    # Initialize the random number generator
    import random
    random.seed(0)

    # Ignore the warnings
    import warnings
    warnings.filterwarnings("ignore")

    import os
    for dirname, _, filenames in os.walk('/content/drive/MyDrive/Sentiment Analysis of Indian Farmers Protest using Twitter Data'):
        for filename in filenames:
            print(os.path.join(dirname, filename))
            
    # Read the data
    tweets = pd.read_csv("/content/drive/MyDrive/Sentiment Analysis of Indian Farmers Protest using Twitter Data/Protest tweets.csv")
    tweets.columns
    
    # Read the users data
    users = pd.read_csv("/content/drive/MyDrive/Sentiment Analysis of Indian Farmers Protest using Twitter Data/Tweeted users.csv")
    tweets.head()
    
    # Merge the datasets
    final =pd.merge(tweets, users, on="userId")
    final.head()
    
    # Let us restrict our analysis to the verified accounts only
    import seaborn as sns
    sns.countplot(x="verified",data=final);
    
    final.columns
    final_verified.shape
    check = final_verified.groupby('username')['tweetId'].count()
    print("Maximum tweets from verified account", check.idxmax(), check.max())
    final_unverified = final[final["verified"]==False]
    check2 = final_unverified.groupby('username')['tweetId'].count()
    print("Maximum tweets from unverified account", check2.idxmax(), check2.max())
   
    check3 = final_verified.groupby('location')['tweetId'].count()
    print("Maximum verified tweets from location", check3.idxmax(), check3.max())
    check4 = final_unverified.groupby('location')['tweetId'].count()
    print("Maximum unverified tweets from location", check4.idxmax(), check4.max())
    
    # Sentiment analysis using LSTM
    # Read in sample data with sentiment marked
    data = pd.read_csv( "/content/drive/MyDrive/Sentiment Analysis of Indian Farmers Protest using Twitter Data/Sentiment.csv")
    # Keeping only the neccessary columns
    data = data[['text','sentiment']]
    data.head()
    
    # Data preprocessing for model
    import re
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    data = data[data.sentiment != "Neutral"]
    data['text'] = data['text'].apply(lambda x: x.lower())
    data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

    print(data[ data['sentiment'] == 'Positive'].size)
    print(data[ data['sentiment'] == 'Negative'].size)

    for idx,row in data.iterrows():
        row[0] = row[0].replace('rt',' ')

    vocabSize = 2000
    tokenizer = Tokenizer(num_words=vocabSize, split=' ')
    tokenizer.fit_on_texts(data['text'].values)
    X = tokenizer.texts_to_sequences(data['text'].values)
    X = pad_sequences(X)
    
    # Define the model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

    embed_dim = 128
    lstm_out = 196

    model = Sequential()
    model.add(Embedding(vocabSize, embed_dim,input_length = X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())
    
    # Train the LSTM model
    from sklearn.model_selection import train_test_split

    Y = pd.get_dummies(data['sentiment']).values
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.15, random_state = 42)
    print(X_train.shape,Y_train.shape)
    print(X_test.shape,Y_test.shape)
    
    # Fit the model
    batch_size = 32
    model.fit(X_train, Y_train, epochs = 10, batch_size=batch_size, verbose = 2)
    
    score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
    print("score: %.2f" % (score))
    print("acc: %.2f" % (acc))
    
    import numpy as np

    pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0

    for x in range(len(X_test)):

        result = model.predict(X_test[x].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]

        if np.argmax(result) == np.argmax(Y_test[x]):
            if np.argmax(Y_test[x]) == 0:
                neg_correct += 1
            else:
                pos_correct += 1

        if np.argmax(Y_test[x]) == 0:
            neg_cnt += 1
        else:
            pos_cnt += 1

    print("pos_acc", pos_correct/pos_cnt*100, "%")
    print("neg_acc", neg_correct/neg_cnt*100, "%")
    
    # Analyse sentiment of a sample tweet
    twt = ['Punjab’s lions and Bengals tigers fighting together again. History repeats itself.']
    #vectorizing the tweet by the pre-fitted tokenizer instance
    twt = tokenizer.texts_to_sequences(twt)
    #padding the tweet to have exactly the same shape as `embedding_2` input
    twt = pad_sequences(twt, maxlen=28, dtype='int32', value=0)
    print(twt)
    sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
    if(np.argmax(sentiment) == 0):
        print("negative")
    elif (np.argmax(sentiment) == 1):
        print("positive")


    
  
    
