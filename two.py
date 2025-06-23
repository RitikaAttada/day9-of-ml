import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow import keras

def prob(num):
    mp = 'two.keras'
    x = np.array([1,2,3,5,7], dtype=float)
    y = np.array([15,13,11,8,5], dtype=float)
    x = (x - 1)/(7-1)
    y = (y - 5)/(15-5)
    if (os.path.exists(mp)):
        m = keras.models.load_model(mp)
    else:
        m = keras.Sequential([keras.layers.Dense(32, input_shape=(1,), activation='relu'),
                              keras.layers.Dense(16, activation='relu'),
                              keras.layers.Dense(units=1)])
        m.compile(optimizer='adam', loss='mean_squared_error')
        h = m.fit(x,y,epochs=500)
        m.save('two.keras')
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
    return m.predict(np.array([(num-1)/(7-1)]))[0][0]*(15-5)+5

print(prob(4))
print(prob(8))
print(prob(6)) 