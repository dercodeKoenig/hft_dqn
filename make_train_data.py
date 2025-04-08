import sys
import os
from MultiTimeframeCandleManager import *
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


def make_train_data(candles_file):
    candles = obj_load(candles_file)
    
    train_data = []
    
    m = MultiTimeframeCandleManager()

    index = 0

    batch_index = 0
    
    for _ in range(10000):
        ret = m.push_m1_candle(candles[index])
        index += 1
    
    while True:
        if index >= len(candles)-21:
            break
            
        ret = m.push_m1_candle(candles[index])
        midnight_open, midnight_opening_range_high,midnight_opening_range_low, pdas, current_close, current_time, charts = ret
    
    
        avg_candle_range = np.mean([ i.h - i.l for i in list(charts[2])])
    #avg_candle_range
    
        slv = avg_candle_range * 2
        tpv = avg_candle_range * 6
        c = m.m1_candles[-1].c
    
        slshort = c + slv
        tpshort = c - tpv
        sllong = c - slv
        tplong = c + tpv
    
        short_stop = False
        long_stop = False
        short_hit = False
        long_hit = False
    
        for index_forward in range(index+1, index+20):
            next_candle = candles[index_forward]
            ncl = next_candle.l
            nch = next_candle.h
    
            if short_hit == False and ncl < sllong:
                long_stop = True
            if long_hit == False and nch > slshort:
                short_stop = True
    
            if short_stop == False and ncl < tpshort:
                short_hit = True
                break
            if long_stop == False and nch > tplong:
                long_hit = True
                break
        
        index += 1
        
        #print(long_hit, short_hit, short_stop, long_stop)
        try:
            x = ret_to_scaled_inputs(ret)
        except IndexError:
            continue
        y = 0
        if long_hit:
            y=1
        if short_hit:
            y=-1
    
        train_data.append((x,y))

        if len(train_data) >= 1000000:
            obj_save(train_data, candles_file.split("/")[-1]+"_train_"+str(batch_index))
            batch_index+=1
            train_data = []
    
        

    if len(train_data) > 0:
            obj_save(train_data, candles_file.split("/")[-1]+"_train_"+str(batch_index))

    


candles_file_path = sys.argv[1]
make_train_data(candles_file_path)