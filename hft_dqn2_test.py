#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from datetime import datetime, timedelta
import time
from collections import deque
import numpy as np
from tqdm import tqdm
import copy
import tensorflow as tf
import random
from save_and_load import *
from Candle import Candle
from MultiTimeframeCandleManager import MultiTimeframeCandleManager

candles = obj_load("NQ_1")
len(candles)


# In[ ]:





# In[2]:


chart_m15 = tf.keras.layers.Input(shape = (60,4))
chart_m5 = tf.keras.layers.Input(shape = (60,4))
chart_m1 = tf.keras.layers.Input(shape = (60,4))

pdas = tf.keras.layers.Input(shape = (19,))

current_position = tf.keras.layers.Input(shape = (1,))
#scaled_open_profit = tf.keras.layers.Input(shape = (1,))

minutes = tf.keras.layers.Input(shape = (1,))
minutes_embed = tf.keras.layers.Embedding(input_dim=60*24, output_dim=8)(minutes)
minutes_embed_flat = tf.keras.layers.Flatten()(minutes_embed)

f15 = tf.keras.layers.Flatten()(chart_m15)
f5 = tf.keras.layers.Flatten()(chart_m5)
f1 = tf.keras.layers.Flatten()(chart_m1)

#c = tf.keras.layers.Concatenate()([f15, f5, f1, pdas, minutes_embed_flat, current_position, scaled_open_profit])
c = tf.keras.layers.Concatenate()([f15, f5, f1, pdas, minutes_embed_flat, current_position])

lrelu = tf.keras.layers.LeakyReLU(0.05)
d = tf.keras.layers.Dense(1024*8)(c)
d = lrelu(d)
d = tf.keras.layers.Dense(1024*8)(d)
d = lrelu(d)
d = tf.keras.layers.Dense(1024*8)(c)
d = lrelu(d)
d = tf.keras.layers.Dense(1024*8)(d)
d = lrelu(d)

value = tf.keras.layers.Dense(1, activation="linear")(d)
advantage = tf.keras.layers.Dense(2, activation="linear")(d)

q_values = tf.keras.layers.Lambda(
    lambda inputs: inputs[0] + (inputs[1] - tf.reduce_mean(inputs[1], axis=1, keepdims=True))
)([value, advantage])


model = tf.keras.Model(inputs = [chart_m15, chart_m5, chart_m1, pdas, minutes, current_position], outputs = q_values)

model.summary()


# In[3]:


model.load_weights("model.weights.h5")


# In[4]:


def relative (value, center, r):
        return (value - center) / r

def ret_to_scaled_inputs(ret):

    midnight_open, midnight_opening_range_high,midnight_opening_range_low, pdas, current_close, current_time, charts = ret

    center = (midnight_opening_range_high + midnight_opening_range_low) / 2
    r = (midnight_opening_range_high - midnight_opening_range_low) / 2

    pda_rel = []
    pda_rel.append(relative(midnight_open, center, r))
    for pda in pdas:
        pda_rel.append(relative(pda, center, r))
    pda_np = np.array(pda_rel)

    current_minutes = current_time.hour * 60 + current_time.minute

    charts_array = []
    for candlesticks in charts:
        charts_array.append([])
        for candle in candlesticks:
            o = relative(candle.o, center, r)
            h = relative(candle.h, center, r)
            l = relative(candle.l, center, r)
            c = relative(candle.c, center, r)
            charts_array[-1].append([o,h,l,c])

    m15_np = np.array(charts_array[0])
    m5_np = np.array(charts_array[1])
    m1_np = np.array(charts_array[2])

    return [m15_np, m5_np, m1_np, pda_np, current_minutes]


# In[ ]:


m = MultiTimeframeCandleManager()

current_position = 0
entry_price = 0

equity = 0
open_profit = 0
equity_L = [0]

cmm = 0.5

last_close = 0
last_state = None
last_action = 0

#for index in tqdm(range(len(candles))):
for index in range(len(candles)):

        ret = m.push_m1_candle(candles[index])
        midnight_open, midnight_opening_range_high,midnight_opening_range_low, pdas, current_close, current_time, charts = ret
        center = (midnight_opening_range_high + midnight_opening_range_low) / 2
        r = (midnight_opening_range_high - midnight_opening_range_low) / 2


        if(len(m.m15_candles) == 60):
            open_profit = (current_close - entry_price) * current_position

            state = ret_to_scaled_inputs(ret) + [current_position]
            m15_np, m5_np, m1_np, pda_np, current_minutes, pos = state

            equity_L.append(equity+open_profit)
            
            output = model([
                    tf.expand_dims(m15_np, 0),
                    tf.expand_dims(m5_np, 0),
                    tf.expand_dims(m1_np, 0),
                    tf.expand_dims(pda_np, 0),
                    tf.expand_dims(current_minutes, 0),
                    tf.expand_dims(pos, 0)
            ])
            action = np.argmax(output)

            last_close = current_close


            if(action == 0 and current_position != -1):
                equity += open_profit
                print("short at", current_close, " last position return:", open_profit, " equity:", equity)
                current_position = -1
                entry_price = current_close
                equity -= cmm

            if(action == 1 and current_position != 1):
                equity += open_profit
                print("long at", current_close, " last position return:", open_profit, " equity:", equity)
                current_position = 1
                entry_price = current_close
                equity -= cmm


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(equity_l)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




