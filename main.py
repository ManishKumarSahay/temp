# flask, scikit-learn, pandas, pickle-mixin
import pandas as pd
from flask import Flask, render_template, request
import pickle
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
import numpy as np


app = Flask(__name__)
Data = pd.read_csv('house_price_data.csv')


X = Data.drop('price',axis =1).values
y = Data['price'].values

#splitting Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

#standardization scaler - fit&transform on train, fit only on test
from sklearn.preprocessing import StandardScaler
s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(np.float))
X_test = s_scaler.transform(X_test.astype(np.float))

# Creating a Neural Network Model
# having 20 nueron is based on the number of available featurs

model = Sequential()

model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')

model.fit(x=X_train,y=y_train,
          validation_data=(X_test,y_test),
          batch_size=64,epochs=200)

y_pred = model.predict(X_test)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    bedrooms = request.form.get('bedrooms')
    bathrooms = request.form.get('bathrooms')
    sqft_living = request.form.get('sqft_living')
    sqft_lot = request.form.get('sqft_lot')
    floors = request.form.get('floors')
    waterfront = request.form.get('waterfront')
    condition = request.form.get('condition')
    yr_built = request.form.get('yr_built')

    print(bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, condition, yr_built)
    input = pd.DataFrame([[bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, condition, yr_built]],
                         columns=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'condition', 'yr_built'])



    input_data_as_numpy_array = np.asarray(input)

    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    std_data = s_scaler.transform(input_data_reshaped)

    prediction = model.predict(std_data)[0]
    print(prediction)
    return str(prediction)


if __name__ == "__main__":
    app.run(debug=True)
