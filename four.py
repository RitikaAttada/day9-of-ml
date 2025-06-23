import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import os

def prob(num1, num2):
    mp = 'four.keras'
    x = np.array([1000,1200,1500,1800,2000], dtype=float)
    z = np.array([2,3,3,4,4], dtype=float)
    y = np.array([50,60,75,90,100], dtype=float)
    x = (x-1000)/(1000)
    z = (z-2)/2
    inp = np.column_stack((x,z))
    y = (y -50)/50
    if (os.path.exists(mp)):
        m = keras.models.load_model(mp)
    else:
        m = keras.Sequential([keras.layers.Dense(32, input_shape=(2,), activation='relu'),
                              keras.layers.Dense(16, activation='relu'),
                              keras.layers.Dense(units=1)])
        m.compile(optimizer ='adam', loss='mean_squared_error')
        h = m.fit(inp, y, epochs=200)
        m.save('four.keras')
        lv = h.history['loss']
        p = m.predict(inp)
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
    return m.predict(np.array([[(num1-1000)/1000, (num2-2)/2]]))[0][0]*50+50

print(prob(1600,3)) 
print(prob(2020,3)) 
print(prob(1400,3))