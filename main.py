import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import MinMaxScaler
import pandas as pd 
import os 
import urllib.request
import requests
import ssl
import datetime
import functools
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, TimeDistributed, RepeatVector
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split




class bitcoin(): 
    def __init__(self): 
        #self.btc_h_path = "https://www.cryptodatadownload.com/cdd/gemini_BTCUSD_1hr.csv"
        self.btc_d_path = "https://www.cryptodatadownload.com/cdd/Coinbase_BTCUSD_d.csv"
        #self.eth_h_path = "https://www.cryptodatadownload.com/cdd/gemini_ETHUSD_1hr.csv"
        self.eth_d_path = "https://www.cryptodatadownload.com/cdd/Coinbase_ETHUSD_d.csv"
        #self.ltc_h_path = "https://www.cryptodatadownload.com/cdd/gemini_LTHUSD_1hr.csv"
        self.ltf_d_path = "https://www.cryptodatadownload.com/cdd/Coinbase_LTCUSD_d.csv"
        plt.style.use('dark_background')
    def get_data(self,update = "yes"):
        if update == "yes":
            ssl._create_default_https_context = ssl._create_unverified_context
            #urllib.request.urlretrieve(self.btc_h_path,filename="bitcoin_hourly.csv")
            urllib.request.urlretrieve(self.btc_d_path,filename="bitcoin_daily.csv")
            #urllib.request.urlretrieve(self.eth_h_path,filename="eth_hourly.csv")
            urllib.request.urlretrieve(self.eth_d_path,filename="eth_daily.csv")
            #urllib.request.urlretrieve(self.ltc_h_path,filename="ltc_hourly.csv")
            urllib.request.urlretrieve(self.ltf_d_path,filename="ltc_daily.csv")
        
    def load_data(self):
        lastdays = 380
        self.btc_hourly = pd.read_csv("bitcoin_daily.csv",skiprows=1)[::-1]
        self.btc_daily = pd.read_csv("bitcoin_daily.csv",skiprows=1)[::-1].iloc[-lastdays:,:]
        self.eth_hourly = pd.read_csv("eth_hourly.csv",skiprows=1)[::-1]
        self.eth_daily = pd.read_csv("eth_daily.csv",skiprows=1)[::-1].iloc[-lastdays:,:]
        self.ltc_hourly = pd.read_csv("ltc_hourly.csv",skiprows=1)[::-1]
        self.ltc_daily = pd.read_csv("ltc_daily.csv",skiprows=1)[::-1].iloc[-lastdays:,:]
        
        print(self.btc_daily.tail())
        
        f = plt.figure(figsize=(20,20))
        ax = f.add_subplot(211)
        ax1 = f.add_subplot(212)
        #self.btc_daily["Date"] = [datetime.datetime.strptime(date, "%Y-%m-%d")np.array(
        #                          for date in self.btc_daily["Date"]]
        ax.plot(self.btc_daily["Date"],self.btc_daily['Close'],label="BTC")
        
        ax.plot(self.eth_daily["Date"],self.eth_daily['Close'],label="ETH")
        ax.plot(self.ltc_daily["Date"],self.ltc_daily['Close'],label="LTC")
        ax.set_title('Stock')
        ax1.plot(self.btc_daily["Date"],self.btc_daily['Volume USD'],label="BTC")
        ax1.plot(self.eth_daily["Date"],self.eth_daily['Volume USD'],label="ETH")
        ax1.plot(self.eth_daily["Date"],self.eth_daily['Volume USD'],label="LTC")
        ax1.set_title('Volume')
        
        
        locator=MaxNLocator(prune='both',nbins=20)
        ax.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_locator(locator)
        ax.xaxis.set_tick_params(rotation=60,labelsize=10)
        ax1.xaxis.set_tick_params(rotation=60,labelsize=10)
        plt.show()
    
    def process_daily_data(self):
        self.dates = self.btc_daily["Date"] 
        data = pd.concat([self.btc_daily.iloc[:,2:],
                       self.eth_daily.iloc[:,2:],
                       self.ltc_daily.iloc[:,2:]],axis=1)
        
        
        data = self.btc_daily.iloc[:,2:]
        print(data.head())
        names = data.columns
        #grad = pd.DataFrame(np.gradient(data.values,axis=0),columns=names)
        #diff = pd.DataFrame(np.diff(data,axis=0),columns=names)
        def gamma(a,b): 
            return ((a+2)/a)*b
        a = pd.Series([functools.reduce(gamma,i) for i in data.values])
        b = pd.Series([functools.reduce(gamma,i[::-1]) for i in data.values])
        c = gamma(data["Open"],data["Close"])
        d = gamma(data["Open"],data["High"])
        e = gamma(data["Open"],data["Low"])
        f = gamma(data["High"],data["Close"])
        g = gamma(data["High"],data["Low"])
        h = gamma(data["Close"],data["Low"])
        i = gamma(data["Open"],data["Volume BTC"])
        j = gamma(data["Open"],data["Volume USD"])
        k = gamma(data["Close"],data["Volume BTC"])
        l = gamma(data["Close"],data["Volume USD"])
        m = gamma(data["High"],data["Volume BTC"])
        n = gamma(data["High"],data["Volume USD"])
        o = gamma(data["Low"],data["Volume BTC"])
        p = gamma(data["Low"],data["Volume USD"])
        q = data["Open"]/data["Low"]
        r = data["Open"]/data["High"]
        s = data["Open"]/data["Close"]
        t = data["Open"]/data["Volume BTC"]
        u = data["Open"]/data["Volume USD"]
        v = data["Open"]/abs(data["High"]-data["Low"])
        to = [data,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v]
        data = pd.concat(to,axis=1)
        X_data = []
        y_data = []
        X_to_predict = []
        
        ws = 10
        n_steps_out = 10
        scaler = MinMaxScaler(feature_range=(-1,1))
        for idx in range(ws,data.shape[0]-n_steps_out):
            X_data.append(scaler.fit_transform(data.iloc[idx-ws:idx,:].values))
            y_data.append(data.iloc[idx:idx+n_steps_out,2:4].values)
        X_to_predict.append(scaler.fit_transform(data.iloc[-ws:,:].values))
        
        print("X_data",np.array(X_data).shape)
        print("X_to_predict",np.array(X_to_predict).shape)
        print("Y",np.array(y_data).shape)
        
        self.ws = ws 
        self.n_steps_out = ws 
        self.X = np.array(X_data)
        self.y = np.array(y_data)
        self.X_to_predict = np.array(X_to_predict)
    
    def model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                            random_state=42)
        # And we create the model
        self.model = Sequential()
        self.model.add(LSTM(50, activation='relu', 
                            input_shape=(self.ws, self.X.shape[2])))
        self.model.add(RepeatVector(self.ws))
        self.model.add(LSTM(50, activation='elu', return_sequences=True))
        #self.model.add(Dense(self.y.shape[2]))
        self.model.add(Dropout(0.1))
        self.model.add(TimeDistributed(Dense(self.y.shape[2])))
        self.model.compile(optimizer='RMSprop', loss='mse',metrics=["mae"])
        
        # Store the model as an attribute of the experiment object
        self.history= self.model.fit(X_train, y_train, epochs=2000,
                                 batch_size=100,verbose=1,validation_data=(X_test,y_test))
        #print(self.history.history)
        f = plt.figure(figsize=(10,10))        
        ax = f.add_subplot(221)
        ax1 = f.add_subplot(222)
        ax2 = f.add_subplot(223)
        ax3 = f.add_subplot(224)
        
        ax.plot(self.history.history["loss"])
        ax.set_title("train loss")
        ax1.plot(self.history.history["mae"])
        ax1.set_title("train mae")
        ax2.plot(self.history.history["val_loss"])
        ax2.set_title("test loss")
        ax3.plot(self.history.history["val_mae"])
        ax3.set_title("test mae")
        yhat = self.model.predict(X_test, verbose=1)
        ypred = self.model.predict(self.X_to_predict, verbose=1)
        print(ypred)
        yhat = [i[0].tolist() for i in yhat]
        #yhat = pd.DataFrame(self.model.predict(X_train, verbose=1)[0])
        #print(yhat)
        plt.show()
        f = plt.figure(figsize=(20,20))
        ax = f.add_subplot(311)
        ax1 = f.add_subplot(312)
        ax2 = f.add_subplot(313)
        ax.plot([i[1] for i in yhat],label="pred")
        ax1.plot([i[1] for i in y_test],label="gt")
        ax2.plot(self.dates[-self.ws:],[i[1] for i in ypred[0]],label="close")
        ax2.plot(self.dates[-self.ws:],[i[0] for i in ypred[0]],label="low")
        plt.legend()
        
        print("test correlation",np.corrcoef([i[1] for i in yhat],
                                             [i[1] for i in y_test])[0][1])
        
    def analyze_market(self): 
        ws = {i:[] for i in [3,5,10]}
        for i in ws:
            for j in range(len(self.btc_daily["Close"])-i):
                ws[i].append(np.std(self.btc_daily["Close"][j:j+i]))
        f = plt.figure(figsize=(20,10))
        ax = f.add_subplot(111)
        for i in ws: 
            ax.plot(ws[i],label=i)
        plt.legend()
        plt.show()
        daily_volatility = []
        for i in range(len(self.btc_daily["High"])):
            daily_volatility.append(self.btc_daily["High"][i]-self.btc_daily["Low"][i])
        f = plt.figure(figsize=(20,10))
        ax = f.add_subplot(211)
        ax1 = f.add_subplot(212)
        ax.plot(self.btc_daily["Date"],daily_volatility,label="Raw volatility")
        ax.plot(self.btc_daily["Date"],self.btc_daily["Close"],label="Close price")
        ax1.plot(self.btc_daily["Date"],self.btc_daily["Volume BTC"],label="Volume price")
        locator=MaxNLocator(prune='both',nbins=30)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_tick_params(rotation=60,labelsize=10)
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_tick_params(rotation=60,labelsize=10)
        plt.legend()
o = bitcoin()
o.get_data("yes")
o.load_data()
#o.analyze_market()
o.process_daily_data()
o.model()
