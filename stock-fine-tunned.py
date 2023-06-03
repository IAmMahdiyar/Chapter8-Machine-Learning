import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

# Load DataSet
raw = pd.read_csv("AAPL.csv", index_col="Date")
raw = raw.dropna()


# Feature Engineering
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

    df_new['avg_price_365'] = df['Close'].rolling(252).mean().shift(1)

    df_new['ratio_avg_price_5_30'] = df_new['avg_price_5'] / df_new['avg_price_30']

    df_new['ratio_avg_price_5_365'] = df_new['avg_price_5'] / df_new['avg_price_365']

    df_new['ratio_avg_price_30_365'] = df_new['avg_price_30'] / df_new['avg_price_365']


def add_avg_volume(df, df_new):
    df_new['avg_volume_5'] = df['Volume'].rolling(5).mean().shift(1)

    df_new['avg_volume_30'] = df['Volume'].rolling(21).mean().shift(1)

    df_new['ratio_avg_volume_5_30'] = df_new['avg_volume_5'] / df_new['avg_volume_30']

    df_new['avg_volume_365'] = df['Volume'].rolling(252).mean().shift(1)

    df_new['ratio_avg_volume_5_365'] = df_new['avg_volume_5'] / df_new['avg_volume_365']

    df_new['ratio_avg_volume_30_365'] = df_new['avg_volume_30'] / df_new['avg_volume_365']


def add_std_price(df, df_new):
    df_new['std_price_5'] = df['Close'].rolling(5).std().shift(1)

    df_new['std_price_30'] = df['Close'].rolling(21).std().shift(1)

    df_new['std_price_365'] = df['Close'].rolling(252).std().shift(1)

    df_new['ratio_std_price_5_30'] = df_new['std_price_5'] / df_new['std_price_30']

    df_new['ratio_std_price_5_365'] = df_new['std_price_5'] / df_new['std_price_365']

    df_new['ratio_std_price_30_365'] = df_new['std_price_30'] / df_new['std_price_365']


def add_std_volume(df, df_new):
    df_new['std_volume_5'] = df['Volume'].rolling(5).std().shift(1)

    df_new['std_volume_30'] = df['Volume'].rolling(21).std().shift(1)

    df_new['std_volume_365'] = df['Volume'].rolling(252).std().shift(1)

    df_new['ratio_std_volume_5_30'] = df_new['std_volume_5'] / df_new['std_volume_30']

    df_new['ratio_std_volume_5_365'] = df_new['std_volume_5'] / df_new['std_volume_365']

    df_new['ratio_std_volume_30_365'] = df_new['std_volume_30'] / df_new['std_volume_365']


def add_return_feature(df, df_new):
    df_new['return_1'] = ((df['Close'] - df['Close'].shift(1))
                          / df['Close'].shift(1)).shift(1)

    df_new['return_5'] = ((df['Close'] - df['Close'].shift(5))
                          / df['Close'].shift(5)).shift(1)

    df_new['return_30'] = ((df['Close'] -
                            df['Close'].shift(21)) / df['Close'].shift(21)).shift(1)

    df_new['return_365'] = ((df['Close'] -
                             df['Close'].shift(252)) / df['Close'].shift(252)).shift(1)

    df_new['moving_avg_5'] = df_new['return_1'].rolling(5).mean().shift(1)
    df_new['moving_avg_30'] = df_new['return_1'].rolling(21).mean().shift(1)
    df_new['moving_avg_365'] = df_new['return_1'].rolling(252).mean().shift(1)


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

# Show Data
print(data)

start_train = '1988-01-01'
end_train = '2018-12-31'
start_test = '2019-01-01'
end_test = '2019-12-31'
data_train = data.loc[start_train:end_train]
X_Train = data_train.drop("close", axis=1).values
Y_Train = data_train["close"].values

data_test = data.loc[start_test:end_test]
X_Test = data_test.drop("close", axis=1).values
Y_Test = data_test["close"].values

scaler = StandardScaler()
X_Train = scaler.fit_transform(X_Train)
X_Test = scaler.transform(X_Test)

HP_Hidden = hp.HParam("hidden_size", hp.Discrete([64, 32, 16]))
HP_Epochs = hp.HParam("epochs", hp.Discrete([100, 200, 500]))
HP_Learning_rate = hp.HParam('learning_rate', hp.RealInterval(0.01, 0.4))
session_num = 0


def fit(hparam, logdir):
    model = keras.Sequential([
        keras.layers.Dense(units=hparam[HP_Hidden], activation='relu'),
        keras.layers.Dense(units=1)
    ])

    model.compile(loss='mean_squared_error', optimizer=tf.optimizers.Adam(hparam[HP_Learning_rate]),
                  metrics=["mean_squared_error"])
    model.fit(X_Train, Y_Train, epochs=hparam[HP_Epochs], verbose=False, validation_data=(X_Test, Y_Test), callbacks=[
        tf.keras.callbacks.TensorBoard(logdir),
        hp.KerasCallback(logdir, hparam),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=200, verbose=0, mode='auto')
    ])

    pred = model.predict(X_Test)
    mse = mean_squared_error(Y_Test, pred)
    r2 = r2_score(Y_Test, pred)

    return mse, r2


def run(hparams, logdir):
    with tf.summary.create_file_writer(logdir).as_default():
        hp.hparams_config(
            hparams=[HP_Hidden, HP_Epochs,
                     HP_Learning_rate],
            metrics=[hp.Metric('mean_squared_error', display_name='mse'),
                     hp.Metric('r2', display_name='r2')], )
        mse, r2 = fit(hparams, logdir)
        tf.summary.scalar('mean_squared_error', mse, step=1)
        tf.summary.scalar('r2', r2, step=1)

for hidden in HP_Hidden.domain.values:
    for epochs in HP_Epochs.domain.values:
        for rate in tf.linspace(HP_Learning_rate.domain.min_value, HP_Learning_rate.domain.max_value, 5):
            param = {
                HP_Hidden: hidden,
                HP_Epochs: epochs,
                HP_Learning_rate: float("%.2f" % float(rate))
            }
            name = "run " + str(session_num)

            print("Starting", name)
            print({h.name: param[h] for h in param})
            run(param, 'logs/' + name)
            session_num = session_num + 1
