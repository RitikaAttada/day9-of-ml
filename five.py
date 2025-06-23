import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import os

def prob(num1, num2):
    mp = 'five.keras'
    x = np.array([2,4,5,7,8], dtype=float)
    z = np.array([8,7,6,5,4], dtype=float)
    y = np.array([65,75,78,85,88], dtype=float)
    x = (x-2)/(6)
    z = (z-4)/4
    inp = np.column_stack((x,z))
    y = (y -65)/(88-65)
    if (os.path.exists(mp)):
        m = keras.models.load_model(mp)
    else:
        m = keras.Sequential([keras.layers.Dense(32, input_shape=(2,), activation='relu'),
                              keras.layers.Dense(16, activation='relu'),
                              keras.layers.Dense(units=1)])
        m.compile(optimizer ='adam', loss='mean_squared_error')
        h = m.fit(inp, y, epochs=200)
        m.save('five.keras')
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
    return m.predict(np.array([[(num1-2)/6, (num2-4)/4]]))[0][0]*(88-65)+65

print(prob(6,6)) 
print(prob(9,4)) 
print(prob(8,3))