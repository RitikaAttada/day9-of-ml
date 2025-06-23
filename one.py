import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow import keras

def prob(num):
    mp = 'one.keras'
    x = np.arange(1.5,8.0,1.5)
    y = np.array([35, 50, 65, 75, 85], dtype=float)
    x = (x - 1.5)/(7.5-1.5)
    y = (y - 35)/(85-35)
    if (os.path.exists(mp)):
        m = keras.models.load_model(mp)
    else:
        m = keras.Sequential([keras.layers.Dense(32, input_shape=(1,), activation='relu'),
                              keras.layers.Dense(16, activation='relu'),
                              keras.layers.Dense(units=1)])
        m.compile(optimizer='adam', loss='mean_squared_error')
        h = m.fit(x,y,epochs=500)
        m.save('one.keras')
        lv = h.history['loss']
        p = m.predict(x)
        plt.figure()
        plt.plot(lv)
        plt.title('loss values')
        plt.show()
        plt.figure()
        plt.scatter(y, p, label='actual vs preicted', color='pink')
        plt.grid(True)
        plt.xlabel('actual')
        plt.ylabel('predicted')
        plt.legend()
        plt.show() 
    return m.predict(np.array([(num-1.5)/(7.5-1.5)]))[0][0]*(85-35)+35

print(prob(5.5))
print(prob(8.0))
print(prob(1)) 