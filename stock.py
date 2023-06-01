import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
import tensorflow as tf

#Load DataSet
raw = pd.read_csv("AAPL.csv", index_col="Date")
raw = raw.dropna()

#Feature Engineering
def add_orginal(df, df_new):
    df_new['open'] = df['Open']

    df_new['open_1'] = df['Open'].shift(1)

    df_new['close_1'] = df['Close'].shift(1)

    df_new['high_1'] = df['High'].shift(1)

    df_new['low_1'] = df['Low'].shift(1)

    df_new['volume_1'] = df['Volume'].shift(1)

def add_avg_price(df, df_new):
    df_new['avg_price_5'] = df['Close'].rolling(5).mean().shift(1)

    df_new['avg_price_30'] = df['Close'].rolling(21).mean().shift(1)

    df_new['ratio_avg_price_5_30'] = df_new['avg_price_5'] / df_new['avg_price_30']

def add_avg_volume(df, df_new):
    df_new['avg_volume_5'] = df['Volume'].rolling(5).mean().shift(1)

    df_new['avg_volume_30'] = df['Volume'].rolling(21).mean().shift(1)

    df_new['ratio_avg_volume_5_30'] = df_new['avg_volume_5'] / df_new['avg_volume_30']

def add_std_price(df, df_new):
    df_new['std_price_5'] = df['Close'].rolling(5).std().shift(1)

    df_new['std_price_30'] = df['Close'].rolling(21).std().shift(1)
 
    df_new['ratio_std_price_5_30'] = df_new['std_price_5'] / df_new['std_price_30']

def add_std_volume(df, df_new):

    df_new['std_volume_5'] = df['Volume'].rolling(5).std().shift(1)

    df_new['std_volume_30'] = df['Volume'].rolling(21).std().shift(1)

    df_new['ratio_std_volume_5_30'] = df_new['std_volume_5'] / df_new['std_volume_30']

def add_return_feature(df, df_new):
    df_new['return_1'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)).shift(1)

    df_new['return_5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)).shift(1)

    df_new['return_30'] = ((df['Close'] - df['Close'].shift(21)) / df['Close'].shift(21)).shift(1)

    df_new['moving_avg_5'] = df_new['return_1'].rolling(5).mean().shift(1)

    df_new['moving_avg_30'] = df_new['return_1'].rolling(21).mean().shift(1)

def gen_features(df):
    df_new = pd.DataFrame()

    # 6 original features
    add_orginal(df, df_new)

    # 31 generated features
    add_avg_price(df, df_new)

    add_avg_volume(df, df_new)

    add_std_price(df, df_new)

    add_std_volume(df, df_new)

    add_return_feature(df, df_new)

    # the target
    df_new['close'] = df['Close']

    df_new = df_new.dropna(axis=0)

    return df_new

data = gen_features(raw)

#Show Data
print(data)

#Train Test Split
data_train = data[:200]
X_train = data_train.drop('close', axis=1).values
y_train = data_train['close'].values

data_test = data[200:229]
X_test = data_test.drop('close', axis=1).values
y_test = data_test['close'].values

model = Sequential([
Dense(units=32, activation='relu'),
Dense(units=1)
])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

model.fit(X_train, y_train, epochs=100, verbose=True)

predictions = model.predict(X_test)[:,0]

print(f'R^2: {r2_score(y_test, predictions):.3f}')