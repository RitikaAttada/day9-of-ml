import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow import keras

def prob(num):
    mp = 'three.keras'
    x = np.array([20,22,25,28,30], dtype=float)
    y = np.array([30,35,50,65,80], dtype=float)
    x = (x - 20)/(30-20)
    y = (y - 30)/(80-30)
    if (os.path.exists(mp)):
        m = keras.models.load_model(mp)
    else:
        m = keras.Sequential([keras.layers.Dense(32, input_shape=(1,), activation='relu'),
                              keras.layers.Dense(16, activation='relu'),
                              keras.layers.Dense(units=1)])
        m.compile(optimizer='adam', loss='mean_squared_error')
        h = m.fit(x,y,epochs=500)
        m.save('three.keras')
        lv = h.history['loss']
        p = m.predict(x)
        plt.figure()
        plt.plot(lv)
        plt.title('loss values')
        plt.show()
        plt.figure()
        plt.plot(y, p, label='actual vs preicted', color='pink')
        plt.grid(True)
        plt.xlabel('actual')
        plt.ylabel('predicted')
        plt.legend()
        plt.show() 
    return m.predict(np.array([(num-20)/(10)]))[0][0]*(50)+30

print(prob(21))
print(prob(23))
print(prob(26)) 