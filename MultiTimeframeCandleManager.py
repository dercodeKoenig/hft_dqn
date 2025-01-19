
from collections import deque
import copy
from Candle import Candle


class MultiTimeframeCandleManager:

    index = 0
    
    current_candles = deque(maxlen = 30000)
    
    midnight_open = 0
    midnight_opening_range_low = 0
    midnight_opening_range_high = 0
    
    opening_range_gap_start_3 = 0
    opening_range_gap_ce_3 = 0
    opening_range_gap_end_3 = 0
    
    opening_range_gap_start_2 = 0
    opening_range_gap_ce_2 = 0
    opening_range_gap_end_2 = 0
    
    opening_range_gap_start_1 = 0
    opening_range_gap_ce_1 = 0
    opening_range_gap_end_1 = 0
    
    fp_low_3 = 0
    fp_ce_3 = 0
    fp_high_3 = 0
    
    fp_low_2 = 0
    fp_ce_2 = 0
    fp_high_2 = 0
    
    fp_low_1 = 0
    fp_ce_1 = 0
    fp_high_1 = 0
        
    last_settlement_price = 0


    def __init__(self):
        l = 60
        self.m1_candles = deque(maxlen = l)
        self.m5_candles = deque(maxlen = l)
        self.m15_candles = deque(maxlen = l)
        self.m60_candles = deque(maxlen = 1000)  
        self.d1_candles = deque(maxlen = 1000)   

    

    
    def push_m1_candle(self, __candle):
        candle = copy.deepcopy(__candle)
        candle_minute = candle.t.minute
        candle_hour = candle.t.hour

        self.m1_candles.append(candle)
    
        if(candle.t.hour == 0 and candle.t.minute == 0 and candle.t.second == 0):
            self.midnight_open  = candle.o
            self.midnight_opening_range_low = candle.l
            self.midnight_opening_range_high = candle.h
    
        if(candle.t.hour == 0 and candle.t.minute <= 30):
            self.midnight_opening_range_low = min(self.midnight_opening_range_low, candle.l)
            self.midnight_opening_range_high = max(self.midnight_opening_range_high, candle.h)
    
        
        if(candle.t.hour == 16 and candle.t.minute == 15 and candle.t.second == 0 and len(self.m1_candles)>1):
            last_candle = self.m1_candles[-2]
            self.last_settlement_price = last_candle.c
    
        if(candle.t.hour == 9 and candle.t.minute == 30 and candle.t.second == 0):
            opening_price = candle.o
            
            self.opening_range_gap_start_3 = self.opening_range_gap_start_2
            self.opening_range_gap_start_2 = self.opening_range_gap_start_1
            self.opening_range_gap_end_3 = self.opening_range_gap_end_2
            self.opening_range_gap_end_2 = self.opening_range_gap_end_1
            self.opening_range_gap_ce_3 = self.opening_range_gap_ce_2
            self.opening_range_gap_ce_2 = self.opening_range_gap_ce_1
    
            self.opening_range_gap_end_1 = opening_price
            self.opening_range_gap_start_1 = self.last_settlement_price
            self.opening_range_gap_ce_1 = (opening_price+self.last_settlement_price) / 2
            
    

        # m5 candles
        if len(self.m5_candles) > 0:
            last_minute = self.m5_candles[-1].t.minute
            last_hour = self.m5_candles[-1].t.hour
            minute_rounded = int(candle_minute/5) * 5
            if minute_rounded != last_minute or candle_hour != last_hour:
                
                c = Candle(candle.o,candle.h,candle.l,candle.c,candle.t)
                self.m5_candles.append(c)
            else:
                self.m5_candles[-1].c = candle.c
                self.m5_candles[-1].h = max(candle.h, self.m5_candles[-1].h)
                self.m5_candles[-1].l = min(candle.l, self.m5_candles[-1].l)
        else:

            c = Candle(candle.o,candle.h,candle.l,candle.c,candle.t)
            self.m5_candles.append(c)
            

       # m15 candles
        if len(self.m15_candles) > 0:
            last_minute = self.m15_candles[-1].t.minute
            last_hour = self.m15_candles[-1].t.hour
            minute_rounded = int(candle_minute/15) * 15
            if minute_rounded != last_minute or candle_hour != last_hour:
               
                c = Candle(candle.o,candle.h,candle.l,candle.c,candle.t)
                self.m15_candles.append(c)
            else:
                self.m15_candles[-1].c = candle.c
                self.m15_candles[-1].h = max(candle.h, self.m15_candles[-1].h)
                self.m15_candles[-1].l = min(candle.l, self.m15_candles[-1].l)
        else:
            
            c = Candle(candle.o,candle.h,candle.l,candle.c,candle.t)
            self.m15_candles.append(c)

        # m60 candles
        if len(self.m60_candles) > 0:
            last_minute = self.m60_candles[-1].t.minute
            last_hour = self.m60_candles[-1].t.hour
            
            if candle_hour != last_hour:
                
                c = Candle(candle.o,candle.h,candle.l,candle.c,candle.t)
                self.m60_candles.append(c)
            else:
                self.m60_candles[-1].c = candle.c
                self.m60_candles[-1].h = max(candle.h, self.m60_candles[-1].h)
                self.m60_candles[-1].l = min(candle.l, self.m60_candles[-1].l)
        else:
           
            c = Candle(candle.o,candle.h,candle.l,candle.c,candle.t)
            self.m60_candles.append(c)


       # d1 candles
        if len(self.d1_candles) > 0:
            last_candle_hour = self.m1_candles[-2].t.hour
            #print(candle_hour)
            if candle_hour != last_candle_hour and candle_hour == 18:
                
                c = Candle(candle.o,candle.h,candle.l,candle.c,candle.t)
                self.d1_candles.append(c)
            else:
                self.d1_candles[-1].c = candle.c
                self.d1_candles[-1].h = max(candle.h, self.d1_candles[-1].h)
                self.d1_candles[-1].l = min(candle.l, self.d1_candles[-1].l)
        else:
            
            c = Candle(candle.o,candle.h,candle.l,candle.c,candle.t)
            self.d1_candles.append(c)


        pdas = [
            self.opening_range_gap_start_3,
            self.opening_range_gap_ce_3,
            self.opening_range_gap_end_3,
    
            self.opening_range_gap_start_2,
            self.opening_range_gap_ce_2,
            self.opening_range_gap_end_2,
    
            self.opening_range_gap_start_1,
            self.opening_range_gap_ce_1,
            self.opening_range_gap_end_1,
            ]

        ret_candles =       [
                self.m15_candles,
                 self.m5_candles,
                 self.m1_candles
        ]
        
        return [
            self.midnight_open, 
            self.midnight_opening_range_high,
            self.midnight_opening_range_low, 
            pdas,
    
            self.m1_candles[-1].c, 
            self.m1_candles[-1].t,
            
            ret_candles
        ]


