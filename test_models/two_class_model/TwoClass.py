import pandas as pd
import json
import tensorflow as tf
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from string import punctuation
from tensorflow import keras
from nltk.corpus import stopwords


class TwoClass(object):
    def __init__(self, dataframe, y):
        self._dataframe=dataframe
        self._y=y
        self.cols = [i for i in self._dataframe.columns if i not in [self._y,]][0]
        self.vectorized_dataframe=self._preProcessData()
    
    def VectorizedDataFrame(self):
        return self.vectorized_dataframe

    def _preProcessData(self):
        ps = PorterStemmer()

        StopWords=stopwords.words('english')

        for indx in self._dataframe.index:
            text = self._dataframe.loc[indx,self.cols]
            text= text.split(' ')
            text= ' '.join(i for i in text if '@' not in i)
            text= ''.join(i for i in text if i not in punctuation and not(i.isdigit()) and i!='\t' and i!='\n')
            text= text.split(' ')
            text= ' '.join(ps.stem(i) for i in text if i not in StopWords)
            self._dataframe.loc[indx,'email']=text

        vectorizer = CountVectorizer()
        df_vectorized=vectorizer.fit_transform(self._dataframe[self.cols])

        self.cols_features=vectorizer.get_feature_names_out()

        vectorized_dataframe=pd.DataFrame(df_vectorized.toarray(), columns=self.cols_features)

        self._y_vectorized = 'responsive_rating'
        vectorized_dataframe[self._y_vectorized]=self._dataframe[self._y]
        return vectorized_dataframe

    def DeepNN_model(self, accuracy_threshold = 0.9, n_epochs=15):
        train,test = train_test_split(self.vectorized_dataframe, test_size=0.25)

        self.col_size=len(self.cols_features)

        self._x_train=train[self.cols_features].values
        self._y_train=train[self._y_vectorized].values

        self._x_test=test[self.cols_features].values
        self._y_test=test[self._y_vectorized].values

        self.col_size=len(self.cols_features)

        class myCallback(tf.keras.callbacks.Callback):
          def on_epoch_end(self, n_epochs, logs={}):
            current_acc = logs.get('accuracy')
            if current_acc is not None:
                if( current_acc > accuracy_threshold):
                    self.model.stop_training = True
            else:
                print("Acuracy is found None", logs.keys())

        callbacks = myCallback()

        self.model= keras.Sequential([
                keras.layers.Dense(128, input_shape=(self.col_size,), activation='relu'),
                keras.layers.Dense(64, input_shape=(self.col_size,), activation='sigmoid'),
                keras.layers.Dense(1),
                ])

        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
            )

        self.model.fit(
            self._x_train,
            self._y_train,
            epochs=n_epochs,
            callbacks=[callbacks,])

    def LogisticRegressionModel(self):
        self._lgr = LogisticRegression()
        self._lgr.fit(self._x_train,self._y_train)

    def LogRegPrediction(self):
        return self._lgr.predict(self._x_test)

    def LogRegAccuracy(self):
        count=0
        total=0
        pred=self._lgr.predict(self._x_test)
        for i in (pred==self._y_test):
            if i==True:
                count += 1
                total += 1
        return(count/total)

    def saveDNNmodel(self, filename: str="model.json"):
        import json
        self.modelJSON=json.loads(self.model.to_json())
        with open(filename, "w") as json_file:
            json.dump(self.modelJSON, json_file, indent=2)
        return self.modelJSON
    
    def get_model(self):
        try:
            return self.model
        except:
            raise ValueError('Model has not been trained yet!')
