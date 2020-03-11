import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Add, Activation
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


df = pd.read_csv('ratings-full.csv')
n_users = df.user.nunique()
n_items = df.item.nunique()
X = np.array(df[['user', 'item']])
y = np.array(df['label'])
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, shuffle=True,
    test_size=0.2)
# print([c.shape for c in [X_trainval, y_trainval, X_test, y_test]])
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval,
    shuffle=True, test_size=0.2)


n_dim = 5
model = Sequential([
    Embedding(n_users + n_items, n_dim, input_length=2),
    Flatten(),
    Dense(units=5, activation='relu'),
    Dense(units=3, activation='sigmoid')
    #, kernel_regularizer=regularizers.l2(0.001))
])

model.compile(
    loss=keras.losses.sparse_categorical_crossentropy,
    # optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
    optimizer=keras.optimizers.Adam(lr=0.01),
    metrics=['accuracy'])

es = keras.callbacks.EarlyStopping(patience=1)

model.fit(X_train, y_train,
          validation_data=(X_valid, y_valid),
          epochs=5, batch_size=1000, callbacks=[es])

print(model.evaluate(X_test, y_test))
