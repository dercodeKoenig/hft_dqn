import time
start_time = time.time()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from MultiTimeframeCandleManager import MultiTimeframeCandleManager
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import copy
import tensorflow as tf
import random
from save_and_load import *
from Candle import Candle
import matplotlib.pyplot as plt

gamma = 0.995
memory_len = 2000000
sarts_memory = deque(maxlen = memory_len)
batch_size = 1
e = 2
slm = 1.5

min_memory_size = batch_size * 20

num_actions = 3

path = "./"

ep_len = 1000

m1 = np.eye(num_actions, dtype="float32")
num_model_inputs = 6


def make_model():
    lrelu = tf.keras.layers.LeakyReLU(0.05)

    chart_m15 = tf.keras.layers.Input(shape=(60, 4))
    chart_m5 = tf.keras.layers.Input(shape=(60, 4))
    chart_m1 = tf.keras.layers.Input(shape=(60, 4))

    pdas = tf.keras.layers.Input(shape=(3 * 3 + 3 * 3 + 1 + 12 * 5 + 5 * 3,))

    current_position = tf.keras.layers.Input(shape=(3,))

    minutes = tf.keras.layers.Input(shape=(1,))
    minutes_embed = tf.keras.layers.Embedding(input_dim=60 * 24, output_dim=8)(minutes)
    minutes_embed_flat = tf.keras.layers.Flatten()(minutes_embed)

    f15 = tf.keras.layers.Flatten()(chart_m15)
    f5 = tf.keras.layers.Flatten()(chart_m5)
    f1 = tf.keras.layers.Flatten()(chart_m1)

    pdas_repeated = tf.keras.layers.Lambda(
        lambda inputs: tf.repeat(tf.expand_dims(inputs, axis=1), repeats=60, axis=1)
    )(pdas)

    concatenated_m5_at = tf.keras.layers.Concatenate(axis=-1)([chart_m5, pdas_repeated])
    m5_at = tf.keras.layers.Dense(128)(concatenated_m5_at)
    m5_at = lrelu(m5_at)
    m5_at = tf.keras.layers.Dense(128)(m5_at)
    m5_at = lrelu(m5_at)
    m5_at = tf.keras.layers.Dense(32)(m5_at)
    m5_at = lrelu(m5_at)
    m5_rnn = tf.keras.layers.GRU(32)(m5_at)

    concatenated_m1_at = tf.keras.layers.Concatenate(axis=-1)([chart_m1, pdas_repeated])
    m1_at = tf.keras.layers.Dense(128)(concatenated_m1_at)
    m1_at = lrelu(m1_at)
    m1_at = tf.keras.layers.Dense(128)(m1_at)
    m1_at = lrelu(m1_at)
    m1_at = tf.keras.layers.Dense(32)(m1_at)
    m1_at = lrelu(m1_at)
    m1_rnn = tf.keras.layers.GRU(32)(m1_at)

    # c = tf.keras.layers.Concatenate()([f15, f5, f1, pdas, minutes_embed_flat, current_position, scaled_open_profit])
    c = tf.keras.layers.Concatenate()([f15, f5, f1, pdas, minutes_embed_flat, current_position, m1_rnn, m5_rnn])

    d = tf.keras.layers.Dense(1024 * 1)(c)
    d = lrelu(d)
    d = tf.keras.layers.Dense(1024 * 1)(d)
    d = lrelu(d)
    d = tf.keras.layers.Dense(1024 * 1)(d)
    d = lrelu(d)

    value = tf.keras.layers.Dense(1, activation="linear")(d)
    advantage = tf.keras.layers.Dense(num_actions, activation="linear")(d)

    q_values = tf.keras.layers.Lambda(
        lambda inputs: inputs[0] + (inputs[1] - tf.reduce_mean(inputs[1], axis=1, keepdims=True))
    )([value, advantage])

    outputs = tf.keras.layers.Activation('linear', dtype='float32')(q_values)

    model = tf.keras.Model(inputs=[chart_m15, chart_m5, chart_m1, pdas, minutes, current_position], outputs=outputs)
    return model

if __name__ == "__main__":
    try:
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
        tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.TPUStrategy(cluster_resolver)
        print("use tpu strategy")
    except:
        strategy = tf.distribute.MirroredStrategy()


    with strategy.scope():
        model = make_model()
        target_model = make_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.000001)
else:
    print("make model...")
    model = make_model()
    print("model made!")



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


equity = 0
equity_L = [0]

inputs = [
    ("NQ_2", 1),
    ("ES_2", 0.75),
    ("YM_2", 1.5),
    ("EURUSD_2", 0.00015),
    ("GBPUSD_2", 0.00015)
]

candles = []
cmm = 0
def reset():
    global index, last_state, last_action, current_position, current_order, equity, m, candles, cmm

    ob = random.choice(inputs)
    candles = obj_load(ob[0])
    cmm = ob[1]

    m = MultiTimeframeCandleManager()

    current_position = Position(0,0,0,0)
    current_order = None

    last_state = None
    last_action = 0

    index = 0

    print("env reset, using data:", ob)


@tf.function()
def inference_step(m15_np, m5_np, m1_np, pda_np, current_minutes, pos_info):
    return model([
        m15_np,
        m5_np,
        m1_np,
        pda_np,
        current_minutes,
        pos_info,
    ])


def step():

    global index, last_state, last_action, current_position, current_order, equity, m


    sarts = None
    while  sarts == None:

        ret = m.push_m1_candle(candles[index])
        midnight_open, midnight_opening_range_high,midnight_opening_range_low, pdas, current_close, current_time, charts = ret
        center = (midnight_opening_range_high + midnight_opening_range_low) / 2
        r = max(0.0001, (midnight_opening_range_high - midnight_opening_range_low) / 2)



        current_candle_m1 = charts[2][-1]
        #### check tp before filling order so that the same m1 candle will not trigger tp - it is not sure if the candle hit first limit and later tp or reve3rse
        if current_position.direction == 1:
            if current_candle_m1.h >= current_position.tp:
                pnl = (current_position.tp - current_position.entry) * current_position.direction
                equity += pnl
                current_position = Position(0,0,0,0)
        if current_position.direction == -1:
            if current_candle_m1.l <= current_position.tp:
                pnl = (current_position.tp - current_position.entry) * current_position.direction
                equity += pnl
                current_position = Position(0,0,0,0)

        #### check order
        if current_order != None:
            if  current_order.direction == 1:
                if current_candle_m1.l < current_order.entry:
                    current_position = Position(current_order.entry, current_order.sl, current_order.tp, current_order.direction)
                    #print("fill long order:",current_order.entry, current_order.sl, current_order.tp)
                    equity -= cmm
                    current_order = None
        if current_order != None:
            if  current_order.direction == -1:
                if current_candle_m1.h > current_order.entry:
                    current_position = Position(current_order.entry, current_order.sl, current_order.tp, current_order.direction)
                    #print("fill short order:",current_order.entry, current_order.sl, current_order.tp)
                    equity -= cmm
                    current_order = None

        #### check sl
        if current_position.direction == 1:
            if current_candle_m1.l <= current_position.sl:
                pnl = (current_position.sl - current_position.entry) * current_position.direction
                equity += pnl
                current_position = Position(0,0,0,0)
        if current_position.direction == -1:
            if current_candle_m1.h >= current_position.sl:
                pnl = (current_position.sl - current_position.entry) * current_position.direction
                equity += pnl
                current_position = Position(0,0,0,0)




        if(len(m.ndogs) == 5 and len(m.fps) == 3 and len(m.opening_range_gaps) == 3 and len(m.asia_highs_lows) == 3 and len(m.london_highs_lows) == 3 and len(m.ny_am_highs_lows) == 3 and len(m.ny_lunch_highs_lows) == 3 and len(m.ny_pm_highs_lows) == 3):


            open_profit = (current_close - current_position.entry) * current_position.direction

            scaled_entry_diff  =  0
            scaled_sl_diff  =  0
            if(current_position.direction != 0):
                scaled_entry_diff = (current_close - current_position.entry) / r
                scaled_sl_diff = (current_close - current_position.sl) / r

            state = ret_to_scaled_inputs(ret) + [np.array([current_position.direction, scaled_entry_diff, scaled_sl_diff])]
            m15_np, m5_np, m1_np, pda_np, current_minutes, pos_info = state

            if(last_state != None):
                diff = (equity+open_profit) - equity_L[-1]
                equity_L.append(equity+open_profit)
                reward =  (diff) / r
                terminal = 0
                if(index+1 == len(candles)):
                    terminal = 1

                sarts = last_state, last_action, reward, terminal, state


            if(random.randint(0,100) > e):

                output = inference_step(
                        tf.expand_dims(m15_np, 0),
                        tf.expand_dims(m5_np, 0),
                        tf.expand_dims(m1_np, 0),
                        tf.expand_dims(pda_np, 0),
                        tf.expand_dims(current_minutes, 0),
                        tf.expand_dims(pos_info, 0)
                )

                last_action = np.argmax(output)
            else:
                last_action = random.randint(0,num_actions-1)

            last_state = state

            current_order = None

            avg_candle_range = np.mean([ i.h - i.l for i in list(charts[2])[55:60]])

            if(last_action == 2 and current_position.direction != 0):
                equity += open_profit
                current_position = Position(0,0,0,0)

            if(last_action == 0 and current_position.direction == 1):
                equity += open_profit
                current_position = Position(0,0,0,0)

            if(last_action == 0 and current_position.direction == 0):
                last_candle_low = charts[2][-2].l
                if ( last_candle_low < current_close ):
                    last_candle_low = None

                pdas = m.normal_pdas ## (low, high)

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


                    current_order = Order(entry, sl, tp, -1)
                    #print("set short order:",entry,sl,tp)



            if(last_action == 1 and current_position.direction == -1):
                equity += open_profit
                current_position = Position(0,0,0,0)

            if(last_action == 1 and current_position.direction == 0):
                last_candle_high = charts[2][-2].h
                if ( last_candle_high > current_close ):
                    last_candle_high = None
                pdas = m.normal_pdas ## (low, high)

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

                    current_order = Order(entry, sl, tp, 1)
                    #print("set long order:",entry,sl,tp)





        index += 1
        if(index == len(candles)):
            reset()

    return sarts

from multiprocessing import Queue, Process

def get_next_sarts(output_q, model_weights_q):
  print("process start")
  reset()
  while True:
    if(model_weights_q.qsize() > 0):
        model.set_weights(model_weights_q.get())
        print("updated weights")


    if(output_q.qsize() < 30):
      r = step()
      output_q.put(r)
    else:
      time.sleep(0.1)


model_qs = []
sarts_qs = []

def update_weights():
    for i in model_qs:
        i.put(model.get_weights())

def get_sarts():
    sarts = []
    for i in sarts_qs:
        if i.qsize() > 0:
            p = i.get()
            sarts.append(p)
    return sarts

@tf.function(jit_compile=True)
def get_target_q(next_states, rewards, terminals):
            estimated_q_values_next = target_model(next_states)
            q_batch = tf.math.reduce_max(estimated_q_values_next, axis=1)
            target_q_values = q_batch * gamma * (1-terminals) + rewards
            return target_q_values

@tf.function(jit_compile=True)
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

def get_data(n):
        r = random.randint(0, len(sarts_memory) - batch_size)
        sarts_sample = [sarts_memory[i] for i in range(r, r + batch_size)]

        states = [x[0] for x in sarts_sample]
        actions = [x[1] for x in sarts_sample]
        rewards = np.array([x[2] for x in sarts_sample], dtype="float32")
        terminals = np.array([x[3] for x in sarts_sample], dtype="float32")
        next_states = [x[4] for x in sarts_sample]

        next_states_array = []
        for i in range(num_model_inputs):
            next_states_array.append(np.array([x[i] for x in next_states], dtype="float32"))

        states_array = []
        for i in range(num_model_inputs):
            states_array.append(np.array([x[i] for x in states], dtype="float32"))

        masks = np.array(m1[actions], dtype="float32")

        return states_array, masks, rewards, terminals, next_states_array


def run():

    sarts_memory.extend(get_sarts())

    while(len(sarts_memory) < min_memory_size):
        time.sleep(10)
        print("waiting for sarts data... - memory size:", len(sarts_memory))
        for _ in range(min_memory_size):
            p = get_sarts()
            if(len(p) > 0):
                sarts_memory.extend(p)
            else:
                break


    distributed_data = (strategy.experimental_distribute_values_from_function(get_data))
    loss, q = strategy.reduce(tf.distribute.ReduceOp.MEAN, strategy.run(tstep, args=(distributed_data,)), axis=None)


    last_rewards = []
    for i in range(len(sarts_memory)-2, len(sarts_memory)):
        last_rewards.append(sarts_memory[i][2])
    last_rewards_mean = np.mean(last_rewards)

    return loss, q, last_rewards_mean



if __name__ == "__main__":

    for _ in range(2):
        x1 = Queue()
        x2 = Queue()

        x = Process(target=get_next_sarts, args=(x1, x2))
        x.daemon = True
        x.start()

        sarts_qs.append(x1)
        model_qs.append(x2)

    loss_mean = []
    q_mean = []
    rewards = []

    try:
        model.load_weights(path + "model.weights.h5")
        target_model.load_weights(path + "model.weights.h5")

        loss_mean = obj_load(path + "loss")
        q_mean = obj_load(path + "q")
        rewards = obj_load(path + "rewards")
    except:
        print("unable to load data")


    update_weights()

    def save():
        model.save_weights(path + "model.weights.h5")
        obj_save(loss_mean, path + "loss")
        obj_save(q_mean, path + "q")
        obj_save(rewards, path + "rewards")
        print("saved progress")


    while True:
        try:
            loss = []
            q = []
            rewards_tmp = []
            progbar = tf.keras.utils.Progbar(ep_len)
            for i in range(ep_len):
                c_loss, c_q, c_rewards = run()
                loss.append(c_loss)
                q.append(c_q)
                rewards_tmp.append(c_rewards)

                progbar.update(i + 1, values=[("loss", c_loss), ("qv", c_q), ("reward", c_rewards)])

            loss_mean.append(np.mean(loss))
            q_mean.append(np.mean(q))
            # rewards.append(np.mean(rewards_tmp))
            rewards.extend(rewards_tmp)

            #progbar.update(ep_len, values = [("loss", np.mean(loss)), ("qv", np.mean(q)), ("reward", np.mean(rewards_tmp))])

            target_model.set_weights(model.get_weights())

            update_weights()

            print("sarts num:", len(sarts_memory))

        except    KeyboardInterrupt:
            print("")
            print("exit")
            import sys
            sys.exit()
            break
