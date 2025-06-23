import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import os

def prob(num1, num2):
    mp = 'six.keras'
    x = np.array([1.2,1.5,1.8,2.0,2.2], dtype=float)
    z = np.array([900,1000,1100,1200,1300], dtype=float)
    y = np.array([22,20,18,15,13], dtype=float)
    x = (x-1.2)/(2.2-1.2)
    z = (z-900)/(1300-900)
    inp = np.column_stack((x,z))
    y = (y -13)/(22-13)
    if (os.path.exists(mp)):
        m = keras.models.load_model(mp)
    else:
        m = keras.Sequential([keras.layers.Dense(32, input_shape=(2,), activation='relu'),
                              keras.layers.Dense(16, activation='relu'),
                              keras.layers.Dense(units=1)])
        m.compile(optimizer ='adam', loss='mean_squared_error')
        h = m.fit(inp, y, epochs=200)
        m.save('six.keras')
        lv = h.history['loss']
        p = m.predict(inp)
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
    return m.predict(np.array([[(num1-1.2)/(2.2-1.2), (num2-900)/(1300-900)]]))[0][0]*(22-13)+13

print(prob(1.6,1050)) 
print(prob(2.0, 1300)) 
print(prob(1,800))