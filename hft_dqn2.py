#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import datetime, timedelta
import time
from collections import deque
import numpy as np
import copy
import tensorflow as tf
import random
from save_and_load import *
from Candle import Candle
from MultiTimeframeCandleManager import MultiTimeframeCandleManager

candles = obj_load("NQ_1")
len(candles)


# In[ ]:


gamma = 0.99
memory_len = 100000
batch_size = 32

path = "./"

ep_len = 1000

m1 = np.eye(2, dtype="float32")
num_model_inputs = 6


# In[ ]:


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
target_model = tf.keras.Model(inputs = [chart_m15, chart_m5, chart_m1, pdas, minutes, current_position], outputs = q_values)

model.summary()


# In[ ]:


optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00001)
#optimizer = tf.keras.optimizers.SGD(learning_rate = 0.0000000001, momentum = 0.95)


# In[ ]:


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

cmm = 0.5

last_close = 0
last_state = None
last_action = 0

index = 0

def step():

    global index, last_close, last_state, last_action, current_position, entry_price, equity


    sarts = None
    while  sarts == None:

        ret = m.push_m1_candle(candles[index])
        midnight_open, midnight_opening_range_high,midnight_opening_range_low, pdas, current_close, current_time, charts = ret
        center = (midnight_opening_range_high + midnight_opening_range_low) / 2
        r = (midnight_opening_range_high - midnight_opening_range_low) / 2


        if(len(m.m15_candles) == 60):

            state = ret_to_scaled_inputs(ret)
            m15_np, m5_np, m1_np, pda_np, current_minutes = state

            if(last_state != None):

                sarts = []
                
                terminal = 0
                if(index+1 == len(candles)):
                    terminal = 1
                    
                current_close = state[2][-1][3]
                last_close = last_state[2][-1][3]

                ### last position -1 
                state_short = last_state + [-1]
                ## action was -1
                new_state = state + [-1]
                reward = ((current_close - last_close) * -1) / r
                sarts.append([state_short, 0, reward, terminal, new_state])
                ## action was 1
                new_state = state + [1]
                reward = ((current_close - last_close) * 1 - cmm) / r
                sarts.append([state_short, 1, reward, terminal, new_state])

                
                ### last position 1 
                state_long = last_state + [1]
                ## action was -1
                new_state = state + [-1]
                reward = ((current_close - last_close) * -1 - cmm) / r
                sarts.append([state_long, 0, reward, terminal, new_state])
                ## action was 1
                new_state = state + [1]
                reward = ((current_close - last_close) * 1) / r
                sarts.append([state_long, 1, reward, terminal, new_state])


            last_state = state


        index += 1
        if(index == len(candles)):
            index = 0
            current_position = 0
            entry_price = 0
            last_close = 0
            last_state = None
            last_action = 0
            print("env reset")

    return sarts


# In[ ]:





# In[ ]:


@tf.function(reduce_retracing=True)
def get_target_q(next_states, rewards, terminals):
            estimated_q_values_next = target_model(next_states)
            q_batch = tf.math.reduce_max(estimated_q_values_next, axis=1)
            target_q_values = q_batch * gamma * (1-terminals) + rewards
            return target_q_values

@tf.function(reduce_retracing=True)
def tstep(states, masks, rewards, terminals, next_states):

    target_q_values = get_target_q(next_states, rewards, terminals)

    with tf.GradientTape() as t:
        model_return = model(states, training=True)
        mask_return = model_return * masks
        estimated_q_values = tf.math.reduce_sum(mask_return, axis=1)
        #print(estimated_q_values, mask_return, model_return, masks)
        loss_e = tf.math.square(target_q_values - estimated_q_values)
        loss = tf.reduce_mean(loss_e)


    gradient = t.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))

    return loss, tf.reduce_mean(estimated_q_values)


# In[ ]:


def run():
    sarts = step()
    sarts_memory.extend(sarts)

    sarts_sample = random.sample(sarts_memory, min(batch_size, len(sarts_memory)))

    states = [x[0] for x in sarts_sample]
    actions = [x[1] for x in sarts_sample]
    rewards = np.array([x[2] for x in sarts_sample], dtype="float32")
    terminals = np.array([x[3] for x in sarts_sample], dtype="float32")
    next_states = [x[4] for x in sarts_sample]

    next_states_array = []
    for i in range(num_model_inputs):
        next_states_array.append(np.array([x[i] for x in next_states], dtype = "float32"))


    states_array = []
    for i in range(num_model_inputs):
        states_array.append(np.array([x[i] for x in states], dtype = "float32"))


    masks = np.array(m1[actions], dtype="float32")

    loss, q = tstep(states_array, masks, rewards, terminals, next_states_array)

    return loss, q


# In[ ]:


sarts_memory = deque(maxlen = memory_len)

loss_mean = []
q_mean = []

try:
    model.load_weights(path+"model.weights.h5")
    target_model.load_weights(path+"model.weights.h5")

    loss_mean = obj_load(path+"loss")
    q_mean = obj_load(path+"q")
except:
    print("unable to load data")


# In[ ]:


def save():
            model.save_weights(path+"model.weights.h5")
            obj_save(loss_mean, path+"loss")
            obj_save(q_mean, path+"q")
            print("saved progress")


# In[ ]:


t0 = time.time()
safe_after_eps = 10
eps_counter=0

while True:
    eps_counter+=1
    try:
        loss = []
        q = []
        rewards_tmp = []
        actions = []
        progbar = tf.keras.utils.Progbar(ep_len)
        for i in range(ep_len):
            c_loss, c_q = run()
            loss.append(c_loss)
            q.append(c_q)
            
            progbar.update(i+1, values = [("loss", c_loss), ("qv", c_q)])

        loss_mean.append(np.mean(loss))
        q_mean.append(np.mean(q))

        target_model.set_weights(model.get_weights())

        if(eps_counter >= safe_after_eps):
            eps_counter = 0
            save()


    except    KeyboardInterrupt:
        print("")
        print("exit")
        save()
        break


# In[ ]:





# In[ ]:


#import matplotlib.pyplot as plt
#plt.plot(equity_L)
#equity


# In[ ]:





# In[ ]:





# In[ ]:




