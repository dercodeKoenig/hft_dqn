#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
start_time = time.time()
import os
from MultiTimeframeCandleManager import MultiTimeframeCandleManager
from datetime import datetime, timedelta
from collections import deque
import numpy as np
from tqdm import tqdm
import copy
import tensorflow as tf
import random
from save_and_load import *
from Candle import Candle
import matplotlib.pyplot as plt
#from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy('mixed_float16')


# In[6]:


try:
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.TPUStrategy(cluster_resolver)
    print("use tpu strategy")
except:
    strategy = tf.distribute.MirroredStrategy()
strategy


# In[7]:

gamma = 0.99
memory_len = 6000000
sarts_memory = deque(maxlen = memory_len)
batch_size = 1024
e = 2
slm = 1.5
steps_before_learn = 1
num_envs = 64
min_memory_size = 100000

ep_len = 1000
save_eps = 20




num_actions = 3
path = "./"
m1 = np.eye(num_actions, dtype="float32")
num_model_inputs = 6


try:
    sarts_cache = obj_load(path+"sarts_cache")
    for i in tqdm(sarts_cache):
        sarts_memory.append(i)
    del sarts_cache
except Exception as error:
    print("no sarts cache loaded")
    print(error)


# In[9]:


with strategy.scope():

  lrelu = tf.keras.layers.LeakyReLU(0.05)


  chart_m15 = tf.keras.layers.Input(shape = (60,4))
  chart_m5 = tf.keras.layers.Input(shape = (60,4))
  chart_m1 = tf.keras.layers.Input(shape = (60,4))

  pdas = tf.keras.layers.Input(shape = (3*3+3*3+1+12*5+5*3,))

  current_position = tf.keras.layers.Input(shape = (3,))

  minutes = tf.keras.layers.Input(shape = (1,))
  minutes_embed = tf.keras.layers.Embedding(input_dim=60*24, output_dim=8)(minutes)
  minutes_embed_flat = tf.keras.layers.Flatten()(minutes_embed)

  f15 = tf.keras.layers.Flatten()(chart_m15)
  f5 = tf.keras.layers.Flatten()(chart_m5)
  f1 = tf.keras.layers.Flatten()(chart_m1)

  pdas_repeated = tf.keras.layers.Lambda(
  lambda inputs: tf.repeat(tf.expand_dims(inputs, axis = 1), repeats=60, axis=1)
  )(pdas)

  concatenated_m5_at = tf.keras.layers.Concatenate(axis=-1)([chart_m5, pdas_repeated])
  m5_at = tf.keras.layers.Dense(256)(concatenated_m5_at)
  m5_at = lrelu(m5_at)
  m5_at = tf.keras.layers.Dense(128)(m5_at)
  m5_at = lrelu(m5_at)
  m5_at = tf.keras.layers.Dense(64)(m5_at)
  m5_at = lrelu(m5_at)
  m5_at = tf.keras.layers.LSTM(128)(m5_at)

  concatenated_m1_at = tf.keras.layers.Concatenate(axis=-1)([chart_m1, pdas_repeated])
  m1_at = tf.keras.layers.Dense(256)(concatenated_m1_at)
  m1_at = lrelu(m1_at)
  m1_at = tf.keras.layers.Dense(128)(m1_at)
  m1_at = lrelu(m1_at)
  m1_at = tf.keras.layers.Dense(64)(m1_at)
  m1_at = lrelu(m1_at)
  m1_at = tf.keras.layers.LSTM(128)(m1_at)

    
  #c = tf.keras.layers.Concatenate()([f15, f5, f1, pdas, minutes_embed_flat, current_position, scaled_open_profit])
  c = tf.keras.layers.Concatenate()([f15, f5, f1, pdas, minutes_embed_flat, current_position, m1_at, m5_at])

  d = tf.keras.layers.Dense(4096)(c)
  d = lrelu(d)
  d = tf.keras.layers.Dense(4096)(d)
  d = lrelu(d)
  d = tf.keras.layers.Dense(2048)(d)
  d = lrelu(d)
  d = tf.keras.layers.Dense(1024)(d)
  d = lrelu(d)


  value = tf.keras.layers.Dense(1, activation="linear")(d)
  advantage = tf.keras.layers.Dense(num_actions, activation="linear")(d)

  q_values = tf.keras.layers.Lambda(
  lambda inputs: inputs[0] + (inputs[1] - tf.reduce_mean(inputs[1], axis=1, keepdims=True))
  )([value, advantage])

  outputs = tf.keras.layers.Activation('linear', dtype='float32')(q_values)

  model = tf.keras.Model(inputs = [chart_m15, chart_m5, chart_m1, pdas, minutes, current_position], outputs = outputs)
  target_model = tf.keras.Model(inputs = [chart_m15, chart_m5, chart_m1, pdas, minutes, current_position], outputs = outputs)


  optimizer = tf.keras.optimizers.Adam(learning_rate = 0.000001)



model.summary()


# In[10]:



# In[2]:


def relative (value, center, r):
        return (value - center) / r

def ret_to_scaled_inputs(ret):

    midnight_open, midnight_opening_range_high,midnight_opening_range_low, pdas, current_close, current_time, charts = ret


    center = (midnight_opening_range_high + midnight_opening_range_low) / 2
    r = max(0.0001,(midnight_opening_range_high - midnight_opening_range_low) / 2)

    pda_rel = []
    pda_rel.append(relative(midnight_open, center, r))
    for pda in pdas[0:9+9+15]:
        pda_rel.append(relative(pda, center, r))
    for index in range(9+9+15,9+9+15+5*12):
        ## highs lows are like this [h, h_taken, l, l_taken]
        ## the bools should not be scaled
        if (index - 9+9+15) % 2 == 0:
            pda_rel.append(relative(pdas[index], center, r))
        else:
            pda_rel.append(pdas[index])

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


# In[11]:

@tf.function(jit_compile = True)
def inference_step(m15_np, m5_np, m1_np, pda_np, current_minutes, pos_info):
    return model([
        m15_np,
        m5_np,
        m1_np,
        pda_np,
        current_minutes,
        pos_info,
    ])


class Order:
    def __init__(self, limit, stop, tp, direction):
        self.entry = limit
        self.tp = tp
        self.sl = stop
        self.direction = direction

class Position:
    def __init__(self, entry, stop, tp, direction):
        self.entry = entry
        self.tp = tp
        self.sl = stop
        self.direction = direction


inputs = [
    (obj_load("NQ_2"), 0.5),
    (obj_load("ES_2"), 0.3),
    (obj_load("YM_2"), 0.75),
    #("EURUSD_2", 0.00015),
    #("GBPUSD_2", 0.00015)
]

class environment:

    def __init__(self):
        self.reset(True)
        
    def reset(self, first = False):

        if(first):
            self.input_index = random.randint(0,len(inputs)-1)
        
        ob = inputs[self.input_index]
        self.candles = ob[0]
        self.cmm = ob[1]

        self.input_index+=1
        if(self.input_index >= len(inputs)):
            self.input_index = 0
    
        self.m = MultiTimeframeCandleManager()
    
        self.current_position = Position(0,0,0,0)
        self.current_order = None
    
        self.last_state = None
        self.state = None
        self.last_action = 0
        self.ret = None

        
        if(first):
            self.index = random.randint(0,len(self.candles)-100000)
        else:
            self.index = 0

        self.equity = 0
        self.equity_L = [0]
        
        print("env reset, using data:", ob, "-",self.index)
    
        return self.step(2)
    
    
    
    
    def step(self, last_action, rek_skip = True):

        self.last_state = self.state
        
        if self.ret != None and self.last_state != None:
            
            midnight_open, midnight_opening_range_high,midnight_opening_range_low, pdas, current_close, current_time, charts = self.ret
            m15_np, m5_np, m1_np, pda_np, current_minutes, pos_info = self.last_state
            
            if current_minutes >= 16*60+30 and current_minutes < 18*60:
                last_action = 2
            
            elif(random.randint(0,100) >= e):
               pass
            else:
                last_action = random.randint(0,num_actions-1)

            #print(last_action, self.current_position.direction)

            self.current_order = None

            avg_candle_range = np.mean([ i.h - i.l for i in list(charts[2])[55:60]])

            if(last_action == 2 and self.current_position.direction != 0):
                self.equity += self.open_profit
                self.current_position = Position(0,0,0,0)

            if(last_action == 0 and self.current_position.direction == 1):
                self.equity += self.open_profit
                self.current_position = Position(0,0,0,0)

            if(last_action == 0 and self.current_position.direction == 0):
                last_candle_low = charts[2][-2].l
                if ( last_candle_low < current_close ):
                    last_candle_low = None

                pdas = self.m.normal_pdas ## (low, high)

                ## ignore pdas with low below close
                pdas_filtered = []
                for pda in pdas:
                        if(pda[0] > current_close):
                            pdas_filtered.append(pda)
                ### sort
                sorted_by_high = sorted(pdas_filtered, key = lambda x:x[1])
                sorted_by_low = sorted(pdas_filtered, key = lambda x:x[0])

                if(len(pdas_filtered) > 0):

                    ### entry is lowest i can get or immediate rebalance
                    entry = sorted_by_low[0][0]
                    if(last_candle_low != None):
                        entry = min(entry, last_candle_low)


                    sl = entry + avg_candle_range * slm
                    tp = entry  -  abs(entry-sl) * 1000


                    self.current_order = Order(entry, sl, tp, -1)
                    #print("set short order:",entry,sl,tp)



            if(last_action == 1 and self.current_position.direction == -1):
                self.equity += self.open_profit
                self.current_position = Position(0,0,0,0)

            if(last_action == 1 and self.current_position.direction == 0):
                last_candle_high = charts[2][-2].h
                if ( last_candle_high > current_close ):
                    last_candle_high = None
                pdas = self.m.normal_pdas ## (low, high)

                ## ignore pdas with low below close
                pdas_filtered = []
                for pda in pdas:
                        if(pda[1] < current_close):
                            pdas_filtered.append(pda)
                ### sort
                sorted_by_high = sorted(pdas_filtered, key = lambda x:x[1], reverse=True)
                sorted_by_low = sorted(pdas_filtered, key = lambda x:x[0], reverse=True)

                if(len(pdas_filtered) > 0):
                    ### entry is lowest i can get or immediate rebalance
                    entry = sorted_by_high[0][1]
                    if(last_candle_high != None):
                        entry = max(entry, last_candle_high)

                    sl = entry - avg_candle_range * slm
                    tp = entry  +  abs(entry-sl) * 1000

                    self.current_order = Order(entry, sl, tp, 1)
                    #print("set long order:",entry,sl,tp)





        self.index += 1
        if(self.index == len(self.candles)):
            return self.reset()
    
    
        self.sarts = None

        self.ret = self.m.push_m1_candle(self.candles[self.index])
        midnight_open, midnight_opening_range_high,midnight_opening_range_low, pdas, current_close, current_time, charts = self.ret
        center = (midnight_opening_range_high + midnight_opening_range_low) / 2
        r = max(0.0001, (midnight_opening_range_high - midnight_opening_range_low) / 2)



        current_candle_m1 = charts[2][-1]
        #### check tp before filling order so that the same m1 candle will not trigger tp - it is not sure if the candle hit first limit and later tp or reve3rse
        if self.current_position.direction == 1:
            if current_candle_m1.h >= self.current_position.tp:
                pnl = (self.current_position.tp - self.current_position.entry) * self.current_position.direction
                self.equity += pnl
                self.current_position = Position(0,0,0,0)
        if self.current_position.direction == -1:
            if current_candle_m1.l <= self.current_position.tp:
                pnl = (self.current_position.tp - self.current_position.entry) * self.current_position.direction
                self.equity += pnl
                self.current_position = Position(0,0,0,0)

        #### check order
        if self.current_order != None:
            if  self.current_order.direction == 1:
                if current_candle_m1.l < self.current_order.entry:
                    self.current_position = Position(self.current_order.entry, self.current_order.sl, self.current_order.tp, self.current_order.direction)
                    #print("fill long order:",self.current_order.entry, self.current_order.sl, self.current_order.tp)
                    self.equity -= self.cmm
                    self.current_order = None
        if self.current_order != None:
            if  self.current_order.direction == -1:
                if current_candle_m1.h > self.current_order.entry:
                    self.current_position = Position(self.current_order.entry, self.current_order.sl, self.current_order.tp, self.current_order.direction)
                    #print("fill short order:",self.current_order.entry, self.current_order.sl, self.current_order.tp)
                    self.equity -= self.cmm
                    self.current_order = None

        #### check sl
        if self.current_position.direction == 1:
            if current_candle_m1.l <= self.current_position.sl:
                pnl = (self.current_position.sl - self.current_position.entry) * self.current_position.direction
                self.equity += pnl
                self.current_position = Position(0,0,0,0)
        if self.current_position.direction == -1:
            if current_candle_m1.h >= self.current_position.sl:
                pnl = (self.current_position.sl - self.current_position.entry) * self.current_position.direction
                self.equity += pnl
                self.current_position = Position(0,0,0,0)




        if(len(self.m.ndogs) == 5 and len(self.m.fps) == 3 and len(self.m.opening_range_gaps) == 3 and len(self.m.asia_highs_lows) == 3 and len(self.m.london_highs_lows) == 3 and len(self.m.ny_am_highs_lows) == 3 and len(self.m.ny_lunch_highs_lows) == 3 and len(self.m.ny_pm_highs_lows) == 3):


            self.open_profit = (current_close - self.current_position.entry) * self.current_position.direction

            scaled_entry_diff  =  0
            scaled_sl_diff  =  0
            if(self.current_position.direction != 0):
                scaled_entry_diff = (current_close - self.current_position.entry) / r
                scaled_sl_diff = (current_close - self.current_position.sl) / r

            self.state = ret_to_scaled_inputs(self.ret) + [np.array([self.current_position.direction, scaled_entry_diff, scaled_sl_diff])]
            m15_np, m5_np, m1_np, pda_np, current_minutes, pos_info = self.state

            if(self.last_state != None):
                diff = (self.equity+self.open_profit) - self.equity_L[-1]
                self.equity_L.append(self.equity+self.open_profit)
                reward =  (diff) / r
                reward = min(max(reward, -10), 10)
                terminal = 0
                if(self.index+1 == len(self.candles)):
                    terminal = 1

                self.sarts = self.last_state, last_action, reward, terminal, self.state


        if self.sarts != None:
            return self.sarts
        else:
            if(rek_skip):
                while self.sarts==None:
                    self.step(2, rek_skip = False)



# In[3]:


envs = []
for _ in range(num_envs):
    env = environment()
    envs.append(env)


# In[4]:


def step_all_envs():
    
    l_m15_np = []
    l_m5_np = []
    l_m1_np = []
    l_pda_np = [] 
    l_current_minutes =[]
    l_pos_info = []
    
    for i in range(len(envs)):
        m15_np, m5_np, m1_np, pda_np, current_minutes, pos_info = envs[i].state
        l_m15_np.append(m15_np)
        l_m5_np.append(m5_np)
        l_m1_np.append(m1_np)
        l_pda_np .append(pda_np)
        l_current_minutes.append(current_minutes)
        l_pos_info.append(pos_info)
    
    l_m15_np = np.array(l_m15_np)
    l_m5_np = np.array(l_m5_np)
    l_m1_np = np.array(l_m1_np)
    l_pda_np = np.array(l_pda_np)
    l_current_minutes = np.array(l_current_minutes)
    l_pos_info = np.array(l_pos_info)

    results = inference_step(
        l_m15_np,
        l_m5_np,
        l_m1_np,
        l_pda_np,
        l_current_minutes,
        l_pos_info
    )

    sarts_list = []
    for i in range(len(envs)):
        sarts = envs[i].step(np.argmax(results[i]))
        sarts_list.append(sarts)
    
    return sarts_list
    


# In[6]:


@tf.function(jit_compile=True)
def get_target_q(next_states, rewards, terminals):
            estimated_q_values_next = target_model(next_states)
            q_batch = tf.math.reduce_max(estimated_q_values_next, axis=1)
            target_q_values = q_batch * gamma * (1-terminals) + rewards
            return target_q_values

@tf.function(jit_compile=True) # my gpu does not support this
def tstep(data):
    states, masks, rewards, terminals, next_states = data

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


# In[14]:


sarts_index = 0
def get_data(n):
        global sarts_index
        #sarts_index = random.randint(0, len(sarts_memory) - batch_size)
        sarts_sample = [sarts_memory[i] for i in range(sarts_index, sarts_index + batch_size)]
        sarts_index+=batch_size
        if(sarts_index+batch_size >= len(sarts_memory)):
            sarts_index = 0
    



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


        return states_array, masks, rewards, terminals, next_states_array


def run():
    actions = []
    rewards = []
    for _ in range(steps_before_learn):
        all_sarts = step_all_envs()
        sarts_memory.extend(all_sarts)
        
        rewards.append([sarts[2] for sarts in all_sarts])
        actions.append([sarts[1] for sarts in all_sarts])

    if(len(sarts_memory) > min_memory_size):

        distributed_data = (strategy.experimental_distribute_values_from_function(get_data))
        loss, q = strategy.reduce(tf.distribute.ReduceOp.MEAN, strategy.run(tstep, args = (distributed_data,)), axis = None)

        return loss, q, np.mean(rewards) , np.mean(actions)

    else :
        return 0,0, np.mean(rewards) , np.mean(actions)


# In[15]:


loss_mean = []
q_mean = []
rewards = []

try:
    model.load_weights(path+"model.weights.h5")
    target_model.load_weights(path+"model.weights.h5")

    loss_mean = obj_load(path+"loss")
    q_mean = obj_load(path+"q")
    rewards = obj_load(path+"rewards")
except Exception as a:
    print(a)
    print("unable to load data")


def save():
            model.save_weights(path+"model.weights.h5")
            obj_save(loss_mean, path+"loss")
            obj_save(q_mean, path+"q")
            obj_save(rewards, path+"rewards")
            print("saved progress")


# In[16]:



eps_c = 0

while True:
    try:
        eps_c += 1
        loss = []
        q = []
        rewards_tmp = []
        actions = []
        progbar = tf.keras.utils.Progbar(ep_len)
        for i in range(ep_len):
            c_loss, c_q, c_rewards, c_action = run()
            loss.append(c_loss)
            q.append(c_q)
            rewards_tmp.append(c_rewards)
            actions.append(c_action)

            progbar.update(i+1, values = [("loss", c_loss), ("qv", c_q), ("reward", c_rewards), ("avg_action", c_action)])

        loss_mean.append(np.mean(loss))
        q_mean.append(np.mean(q))
        #rewards.append(np.mean(rewards_tmp))
        rewards.extend(rewards_tmp)

        #progbar.update(ep_len, values = [("loss", np.mean(loss)), ("qv", np.mean(q)), ("reward", np.mean(rewards_tmp)), ("avg_action", np.mean(actions))])

        target_model.set_weights(model.get_weights())


        
        if(eps_c >=save_eps):
          eps_c=0
          save()

    except    KeyboardInterrupt:
        print("")
        print("exit")
        break

save()


obj_save(list(sarts_memory), path+"sarts_cache")



# In[ ]:





# In[ ]:





# In[ ]:




